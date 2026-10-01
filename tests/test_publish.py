"""L11 - the public copy: an allow-list, and checks that refuse anything private or shaped like a key."""
from __future__ import annotations

import pytest

from machine.tools import publish as P

EMAILS = {"someone.private@gmail.com"}


def files(**extra) -> dict[str, bytes]:
    base = {"machine/x.py": b"print('ok')\n", "README.md": b"# hello\n"}
    base.update({k.replace("__", "/").replace("_dot_", "."): v for k, v in extra.items()})
    return base


def test_a_clean_copy_has_no_findings():
    assert P.check(files(), EMAILS) == []


@pytest.mark.parametrize("path", ["strategy/LEARNED.md", "old/build/x.py", "CLAUDE.md", ".gitleaksignore"])
def test_a_private_path_is_refused(path):
    assert any("a private path" in f for f in P.check({path: b"x"}, EMAILS))


@pytest.mark.parametrize("text, name", [
    ("TOKEN = '1234567890:" + "A" * 35 + "'", "telegram bot token"),
    ("key = 'sk-ant-api03-" + "x" * 60 + "'", "anthropic key"),
    ("ghp_" + "a" * 36, "github token"),
    ("AKIA" + "B" * 16, "aws key"),
    ("-----BEGIN " + "OPENSSH PRIVATE KEY-----", "private key"),
    ("C:" + "\\Users\\someone\\Desktop", "a local user folder"),
])
def test_secrets_and_private_text_are_refused(text, name):
    found = P.check({"machine/a.py": ("x = 1\n" + text + "\n").encode()}, EMAILS)
    assert found == [f"machine/a.py:2: {name}"]


def test_a_personal_email_of_the_history_is_refused_but_a_noreply_is_fine():
    assert P.check({"docs/a.md": b"mail someone.private@gmail.com"}, EMAILS)
    assert not P.check({"docs/a.md": b"Co-Authored-By: Claude <noreply@anthropic.com>"}, EMAILS)


def test_the_private_repo_s_name_is_refused_in_any_case():
    found = P.check({"docs/a.md": b"git clone https://github.com/x/Secret-Repo-Name.git"}, EMAILS, {"secret-repo-name"})
    assert found == ["docs/a.md: the private repo's name"]


def test_the_private_name_comes_from_the_remote_not_from_the_code(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin", "git@github.com:someone/Hidden-Name.git"],
                   check=True)
    assert P.private_names(tmp_path) == {"hidden-name"}
    assert "hidden-name" not in P.Path(P.__file__).read_text(encoding="utf-8").lower()


def test_a_test_s_obviously_fake_key_is_not_a_finding():
    assert P.check({"tests/t.py": b"monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-should-never-be-passed-on')"},
                   EMAILS) == []


def test_binary_files_are_not_read_as_text():
    assert P.check({"docs/demo.gif": b"GIF89a sk-ant-api03-" + b"x" * 60}, EMAILS) == []


def test_write_keeps_git_and_replaces_everything_else(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "old.txt").write_text("from the last copy")
    (tmp_path / "gone").mkdir()
    p = P.Plan(files={"machine/a.py": b"a = 1\n", "PRIVATE-PARTS.md": b"# x\n"})
    P.write(p, tmp_path)
    assert (tmp_path / ".git" / "HEAD").is_file()
    assert not (tmp_path / "old.txt").exists() and not (tmp_path / "gone").exists()
    assert (tmp_path / "machine" / "a.py").read_bytes() == b"a = 1\n"


def test_write_refuses_when_the_checks_found_something(tmp_path):
    p = P.Plan(files={"strategy/x.md": b"x"}, findings=["strategy/x.md: a private path"])
    with pytest.raises(P.PublishError, match="not written"):
        P.write(p, tmp_path)
    assert not any(tmp_path.iterdir())


def test_private_parts_names_every_left_out_folder():
    text = P.private_parts()
    for path, _ in P.LEFT_OUT:
        assert f"`{path}`" in text
    assert "skipped" in text and "machine.demo" in text


@pytest.mark.private
def test_the_real_copy_is_clean_and_holds_only_the_allowed_files():
    """The copy of THIS repo's HEAD (the private one): nothing private, the public README and the demo in it."""
    p = P.plan_copy()
    assert p.findings == []
    paths = set(p.files)
    assert not any(x.startswith(("strategy/", "old/", "plan/work/ideas/", "plan/work/trials/")) for x in paths)
    assert {"README.md", "PRIVATE-PARTS.md", "docs/DESIGN.md", "plan/README.md", "plan/work/demo-script.md",
            "plan/work/prior-tries.csv", "machine/demo/run.py", "tests/data/demo/idea.json",
            ".github/workflows/tests.yml", ".gitignore"} <= paths
    assert not ({"CLAUDE.md", "docs/SETUP.md", ".github/workflows/learned-merge.yml", "plan/PROGRESS.md",
                 "plan/likhith.md", "plan/tejas.md"} & paths)
    assert all(not x.startswith("data/") or x in P.PUBLIC_FILES for x in paths)
    assert b"idea-machine" in p.files["README.md"]
