"""L9c: the bot's files reach git every night (machine/telegram/nightly.py) and the pull that knows them
(machine/tools/pull.py). Real git, throwaway repos: a bare "GitHub", the bot's clone (dev), the job's own clone, and
a third clone that plays GitHub's side (a merged pull request, the learned-merge Action, a session's push)."""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from types import SimpleNamespace

import pytest

from machine import config
from machine.telegram import nightly as N
from machine.tools import pull as P

DAY = dt.date(2026, 9, 29)
LESSON = "strategy/learned-inbox/2026-09-29-2130-telegram.md"
PICTURE = "strategy/pictures/idea-001-Q1-his-answer.jpg"
ANSWERS = "plan/work/ideas/idea-001/answers.md"
STATE = "plan/work/ideas/idea-001/state.json"
FAKE_KEY = "123456789:" + "A" * 35                       # the shape of a Telegram bot key, not a real one


def sh(repo, *args, check=True):
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and done.returncode:
        raise AssertionError(f"git {args} failed: {done.stderr}")
    return done


def write(repo, path, text, mode="w"):
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        f.write_bytes(text)
    else:
        with open(f, mode, encoding="utf-8", newline="\n") as fh:
            fh.write(text)


@pytest.fixture
def r(tmp_path, monkeypatch):
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "test")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "test@example.com")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "pull-data")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    github = tmp_path / "github-side"
    sh(tmp_path, "clone", "-q", str(origin), str(github))
    write(github, ".gitignore", "/data/*\n.env\n")
    write(github, "strategy/learned-inbox/README.md", "# inbox\n")
    write(github, ANSWERS, "# Answers - idea-001\n")
    write(github, STATE, '{"state": "QUESTIONS QUEUED"}\n')
    write(github, "machine/x.py", "x = 1\n")
    sh(github, "add", "-A")
    sh(github, "commit", "-q", "-m", "start")
    sh(github, "push", "-q", "origin", "HEAD:main")
    dev = tmp_path / "ML Bot"
    sh(tmp_path, "clone", "-q", str(origin), str(dev))
    return SimpleNamespace(origin=origin, github=github, dev=dev, clone=tmp_path / "ML Bot-bot", tmp=tmp_path)


def bot_writes(dev):
    """What one day of the bot leaves in its clone, plus things that must NEVER be copied."""
    write(dev, LESSON, "# Lesson (Telegram, 29 Sep 2026 21:30) — waiting for a number\n\n> the grab must be clean\n")
    write(dev, PICTURE, b"\xff\xd8\xff\x00jpeg\x00bytes")
    write(dev, ANSWERS, "\n## Q1\n\n> A\n", mode="a")
    write(dev, ".env", f"TELEGRAM_BOT_TOKEN={FAKE_KEY}\n")                    # ignored by git; never copied
    write(dev, "data/telegram/messages.jsonl", '{"update_id": 1}\n')          # data/: never copied
    write(dev, "machine/x.py", "x = 2  # a session's unfinished code\n")     # not a bot file
    write(dev, "strategy/learned-inbox/2026-09-29-2200-session.md", "a session's lesson\n")  # not the bot's


def job(r, **kw):
    return N.run(dev=r.dev, clone=r.clone, today=kw.pop("today", DAY), gh=kw.pop("gh", None), log=lambda s: None,
                 **kw)


def on_branch(r, branch):
    """Files the branch changes against GitHub's main."""
    return set(sh(r.origin, "diff", "--name-only", f"main...{branch}").stdout.split())


def github_merges(r, branch):
    sh(r.github, "fetch", "-q", "origin")
    sh(r.github, "merge", "-q", "--no-ff", "-m", f"merge {branch}", f"origin/{branch}")
    sh(r.github, "push", "-q", "origin", "HEAD:main")


def github_numbers(r, path):
    """What the learned-merge Action does: the inbox file moves to merged/."""
    (r.github / "strategy/learned-inbox/merged").mkdir(parents=True, exist_ok=True)
    sh(r.github, "mv", path, path.replace("learned-inbox/", "learned-inbox/merged/"))
    github_commits(r, "learned-merge: #153")


def github_commits(r, message):
    sh(r.github, "add", "-A")
    sh(r.github, "commit", "-q", "-m", message)
    sh(r.github, "push", "-q", "origin", "HEAD:main")


def fresh_github(r):
    sh(r.github, "pull", "-q", "origin", "main")


# ================================================================================================ which files

def test_patterns_one_star_is_one_name_and_never_data_or_env():
    assert N.matches(LESSON) and N.matches(PICTURE) and N.matches(ANSWERS)
    assert not N.matches("strategy/learned-inbox/merged/2026-09-29-2130-telegram.md")   # only the Action writes it
    assert not N.matches("plan/work/ideas/idea-001/deep/answers.md")
    assert not N.matches("old/strategy/learned-inbox/2026-09-29-2130-telegram.md")
    assert not N.matches("strategy/learned-inbox/2026-09-29-2200-session.md")          # a session's, not the bot's
    assert N.forbidden("data/telegram/sent.jsonl") and N.forbidden(".env") and N.forbidden("x/.env.local")
    assert N.forbidden(PICTURE) is None


def test_compare_link():
    assert N.compare_link("https://github.com/example-owner/example-repo.git", "bot/2026-09-29") == (
        "https://github.com/example-owner/example-repo/compare/main...bot/2026-09-29?expand=1")
    assert N.compare_link("git@github.com:example-owner/example-repo.git", "bot/x").endswith("main...bot/x?expand=1")
    assert N.compare_link("C:/some/folder/origin.git", "bot/x") is None


# ================================================================================================ the job

def test_copies_only_the_bots_files_to_a_branch(r):
    bot_writes(r.dev)
    res = job(r)
    assert res.branch == "bot/2026-09-29" and res.error is None
    assert on_branch(r, res.branch) == {LESSON, PICTURE, ANSWERS}          # no .env, no data/, no code, no session file
    sh(r.github, "fetch", "-q", "origin")
    assert sh(r.github, "show", f"origin/{res.branch}:{ANSWERS}").stdout == "# Answers - idea-001\n\n## Q1\n\n> A\n"
    text = N.message(res)
    assert "1 lesson, 1 picture, 1 answer file" in text and "bot/2026-09-29" in text
    assert set(sh(r.dev, "status", "--porcelain", "-uall").stdout.split()) - {"M", "??"} == {   # his clone untouched
        LESSON, PICTURE, ANSWERS, "machine/x.py", "strategy/learned-inbox/2026-09-29-2200-session.md"}
    assert sh(r.dev, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"


def test_quiet_night_sends_nothing(r):
    res = job(r)
    assert res.branch is None and N.message(res) is None
    assert sh(r.origin, "branch", "--list", "bot/*").stdout == ""


def test_dry_run_changes_nothing(r):
    bot_writes(r.dev)
    res = job(r, dry=True)
    assert res.branch is None and {f["path"] for f in res.of("copy", "join")} == {LESSON, PICTURE, ANSWERS}
    assert sh(r.origin, "branch", "--list", "bot/*").stdout == ""


def test_after_the_merge_nothing_is_copied_again(r):
    bot_writes(r.dev)
    github_merges(r, job(r).branch)
    res = job(r, today=DAY + dt.timedelta(days=1))
    assert res.branch is None
    assert {f["why"] for f in res.files} == {"GitHub has it already"}


def test_a_numbered_lesson_is_not_sent_again(r):
    bot_writes(r.dev)
    github_merges(r, job(r).branch)
    fresh_github(r)
    github_numbers(r, LESSON)
    res = job(r, today=DAY + dt.timedelta(days=1))
    assert res.branch is None
    assert any("numbered it already" in f["why"] for f in res.files)


def test_lines_added_on_both_sides_are_joined(r):
    write(r.github, ANSWERS, "\n## L6.2-Q1\n\n> NO, too near\n", mode="a")                 # a session, on GitHub
    github_commits(r, "an answer from a session")
    write(r.dev, ANSWERS, "\n## Q1\n\n> A\n", mode="a")                                    # the bot, on his PC
    res = job(r)
    assert [f["action"] for f in res.files] == ["join"]
    sh(r.github, "fetch", "-q", "origin")
    assert sh(r.github, "show", f"origin/{res.branch}:{ANSWERS}").stdout == (
        "# Answers - idea-001\n\n## L6.2-Q1\n\n> NO, too near\n\n## Q1\n\n> A\n")


def test_changed_on_both_is_named_not_copied(r):
    write(r.github, STATE, '{"state": "FORM DRAFTED"}\n')
    github_commits(r, "a session moved the idea")
    write(r.dev, STATE, '{"state": "LOCKED"}\n')
    write(r.dev, LESSON, "# Lesson\n\n> words\n")
    res = job(r)
    assert on_branch(r, res.branch) == {LESSON}
    assert res.of("conflict")[0]["path"] == STATE
    assert f"changed both on your PC and on GitHub (a Claude session can join it): {STATE}" in N.message(res)


def test_a_file_with_a_key_inside_is_refused(r):
    write(r.dev, LESSON, f"# Lesson\n\n> my key is {FAKE_KEY}\n")
    res = job(r)
    assert res.branch is None
    assert res.of("refused")[0]["why"].startswith("looks like it holds a key")
    assert "must never go to GitHub" in N.message(res)
    assert FAKE_KEY not in N.message(res)


def test_a_second_run_the_same_day_gets_a_new_branch_and_says_it_has_the_older_one(r):
    bot_writes(r.dev)
    first = job(r)
    write(r.dev, "strategy/learned-inbox/2026-09-29-2350-telegram.md", "# Lesson\n\n> more\n")
    second = job(r)
    assert second.branch == "bot/2026-09-29-2" and second.older == ["bot/2026-09-29"]
    assert on_branch(r, second.branch) == on_branch(r, first.branch) | {
        "strategy/learned-inbox/2026-09-29-2350-telegram.md"}
    assert "also has everything from bot/2026-09-29. Merge only this one" in N.message(second)


def test_a_clone_the_job_did_not_make_is_never_touched(r):
    sh(r.tmp, "clone", "-q", str(r.origin), str(r.clone))
    write(r.clone, "his-own-work.txt", "keep me\n")
    with pytest.raises(N.NightlyError, match="did not make"):
        job(r)
    assert (r.clone / "his-own-work.txt").read_text() == "keep me\n"


def test_gh_opens_the_pull_request_when_it_is_there(r, tmp_path):
    fake = tmp_path / "fake_gh.py"
    fake.write_text("import sys, pathlib\n"
                    "pathlib.Path(sys.argv[0]).with_suffix('.args').write_text(' '.join(sys.argv[1:]))\n"
                    "print('https://github.com/example-owner/example-repo/pull/7')\n")
    bot_writes(r.dev)
    res = job(r, gh=[sys.executable, str(fake)])
    assert res.pr == "https://github.com/example-owner/example-repo/pull/7" and res.link is None
    args = fake.with_suffix(".args").read_text()
    assert args.startswith("pr create --base main --head bot/2026-09-29")
    assert 'press "Merge pull request":\nhttps://github.com/example-owner/example-repo/pull/7' in N.message(res)


def test_a_failure_messages_tejas(r):
    bot_writes(r.dev)
    job(r, dry=True)                                                     # the clone exists
    sh(r.clone, "remote", "set-url", "origin", str(r.tmp / "no-such-github.git"))
    told, state = [], r.tmp / "state"
    code = N.main(["--clone", str(r.clone)], tell=told.append, dev=r.dev, state_dir=state)
    assert code == 1 and len(told) == 1
    assert told[0].startswith("The nightly copy of your bot's files to GitHub did NOT work tonight.")
    assert "could not reach GitHub" in told[0] and "Nothing is lost" in told[0]
    assert N.status_line(state).startswith("Nightly copy to GitHub: FAILED")


def test_main_records_the_run_and_tells_him_once(r):
    bot_writes(r.dev)
    told, state = [], r.tmp / "state"
    assert N.main(["--clone", str(r.clone)], tell=told.append, dev=r.dev, state_dir=state) == 0
    assert len(told) == 1 and "on GitHub" in told[0]
    assert "file" in N.status_line(state) and "bot/" in N.status_line(state)
    assert N.main(["--clone", str(r.clone)], tell=told.append, dev=r.dev, state_dir=state) == 0
    assert len(told) == 2                                               # the same files again: branch -2 (not merged)


# ================================================================================================ the pull

def test_plain_git_pull_refuses_after_the_merge_and_the_tool_pulls(r):
    bot_writes(r.dev)
    (r.dev / "machine/x.py").write_text("x = 1\n")                     # (the session's code put back)
    github_merges(r, job(r).branch)
    assert sh(r.dev, "pull", "-q", check=False).returncode != 0         # the reason this tool exists
    assert P.pull(r.dev, log=lambda s: None) == 0
    assert sh(r.dev, "rev-parse", "HEAD").stdout == sh(r.dev, "rev-parse", "origin/main").stdout
    left = set(sh(r.dev, "status", "--porcelain", "-uall").stdout.split())
    assert LESSON not in left and PICTURE not in left and ANSWERS not in left
    assert (r.dev / PICTURE).read_bytes() == b"\xff\xd8\xff\x00jpeg\x00bytes"


def test_the_pull_keeps_lines_the_bot_added_since(r):
    bot_writes(r.dev)
    (r.dev / "machine/x.py").write_text("x = 1\n")
    github_merges(r, job(r).branch)
    write(r.dev, ANSWERS, "\n## Q2\n\n> B\n", mode="a")                # after the nightly copy
    assert P.pull(r.dev, log=lambda s: None) == 0
    assert (r.dev / ANSWERS).read_text(encoding="utf-8").endswith("> A\n\n## Q2\n\n> B\n")
    assert set(sh(r.dev, "status", "--porcelain", "-uall").stdout.split()) - {"M", "??"} == {
        ANSWERS, "strategy/learned-inbox/2026-09-29-2200-session.md"}


def test_the_pull_stops_on_a_real_difference_and_touches_nothing(r):
    write(r.github, STATE, '{"state": "FORM DRAFTED"}\n')
    github_commits(r, "a session moved the idea")
    write(r.dev, STATE, '{"state": "LOCKED"}\n')
    head = sh(r.dev, "rev-parse", "HEAD").stdout
    said = []
    assert P.pull(r.dev, log=said.append) == 1
    assert sh(r.dev, "rev-parse", "HEAD").stdout == head
    assert (r.dev / STATE).read_text() == '{"state": "LOCKED"}\n'
    assert any(STATE in s for s in said)


def test_the_pull_removes_a_lesson_github_numbered(r):
    bot_writes(r.dev)
    (r.dev / "machine/x.py").write_text("x = 1\n")
    github_merges(r, job(r).branch)
    fresh_github(r)
    github_numbers(r, LESSON)
    assert P.pull(r.dev, log=lambda s: None) == 0
    assert not (r.dev / LESSON).exists()
    assert (r.dev / LESSON.replace("learned-inbox/", "learned-inbox/merged/")).is_file()
