"""The lesson numbering (L9a): highest + 1 with the duplicate #21-#24 and both heading styles present, oldest first,
append only, files moved to merged/, and it refuses to run anywhere but the GitHub Action on main."""
from __future__ import annotations

import re

import pytest

from machine import config
from machine.search.index import LEARNED_HEAD, _learned_entries
from machine.tools import learned_merge as M

MAIN = {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main"}

LEARNED = """# LEARNED.md — everything Tejas has taught, in his own words

> APPEND-ONLY ARCHIVE.

---

## #1  ·  2026-08-30 14:22  ·  FEEDBACK

old words

## #21  ·  2026-09-06  ·  TEJAS — EQUAL HIGHS AND EQUAL LOWS ARE A PAIR.

## #24  ·  2026-09-06 23:30  ·  TEJAS — MY LANGUAGE IS TOO HARD

## #21  ·  2026-09-07 23:01  ·  LESSON · on card **T20240307-00375**

## #24  ·  2026-09-07 23:15  ·  CORRECTION

## #135 — 2026-09-25. "IT SHOULD EVALUATE ITSELF"

## #136 — 2026-09-27. "YES, THAT IS WHAT I WANT"

text of 136 with a word #999 inside that is not a heading
"""

SESSION = """# Lesson (session, 29 Sep 2026, prompt L8) — waiting for a number

> Yes

(his answer to: "Shall I mark L8 as done?")

= **L8 counts as done** with the two unmet marks written down. The rest comes in L9b.
"""

TELEGRAM = """# Lesson (Telegram, 29 Sep 2026 14:02) — waiting for a number

> The gap must be the first one after the break.

- picture: `strategy/pictures/2026-09-29-1402-telegram-m7-his-picture.jpg` (the phone's top bar and buttons cut off)
- Telegram message 7, sent 2026-09-29 14:02:13 +0530, filed by the bot (L9a)
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "strategy" / "learned-inbox" / "merged").mkdir(parents=True)
    (tmp_path / "strategy" / "LEARNED.md").write_bytes(LEARNED.encode("utf-8"))
    (tmp_path / "strategy" / "learned-inbox" / "README.md").write_text("# learned-inbox\n", encoding="utf-8")
    return tmp_path


def _put(repo, name, text):
    (repo / "strategy" / "learned-inbox" / name).write_text(text, encoding="utf-8")


def test_highest_counts_both_heading_styles_and_the_duplicates():
    assert M.highest(LEARNED) == 136
    assert M.highest("## #7  ·  x\n## #3 — y\n") == 7
    assert M.highest("no lessons") == 0


def test_numbering_with_duplicates_present_gives_highest_plus_one(repo):
    _put(repo, "2026-09-29-1402-telegram.md", TELEGRAM)
    todo, problems = M.plan(repo)
    assert [w.n for w in todo] == [137] and problems == []


def test_numbering_follows_the_real_learned_md():
    """On the real file: the highest over both styles (#21-#24 twice) + 1 is where the next lesson starts."""
    text = config.LEARNED_MD.read_text(encoding="utf-8")
    numbers = [int(m.group(1)) for m in re.finditer(r"^## #(\d+)", text, re.MULTILINE)]
    assert len(numbers) > len(set(numbers))                     # the duplicates are really there
    todo, _ = M.plan(config.ROOT)
    if todo:
        assert todo[0].n == max(numbers) + 1 and [w.n for w in todo] == list(range(max(numbers) + 1,
                                                                                    max(numbers) + 1 + len(todo)))


def test_oldest_first_and_a_second_file_in_the_same_minute_after_the_first(repo):
    _put(repo, "2026-09-29-1402-telegram-2.md", TELEGRAM.replace("first one", "second one"))
    _put(repo, "2026-09-29-1402-telegram.md", TELEGRAM)
    _put(repo, "2026-09-28-2020-session.md", SESSION)
    _put(repo, "2026-09-29-0705-session.md", SESSION)
    todo, _ = M.plan(repo)
    assert [(w.n, w.path.name) for w in todo] == [
        (137, "2026-09-28-2020-session.md"), (138, "2026-09-29-0705-session.md"),
        (139, "2026-09-29-1402-telegram.md"), (140, "2026-09-29-1402-telegram-2.md")]


def test_a_file_that_would_break_the_numbering_waits(repo):
    _put(repo, "notes.md", SESSION)
    _put(repo, "2026-09-29-1500-session.md", SESSION + "\n## #12 — a copied heading\n")
    todo, problems = M.plan(repo)
    assert todo == [] and len(problems) == 2
    assert any("no time" in p for p in problems) and any("## #N" in p for p in problems)


def test_titles():
    assert M.title_of(SESSION) == "L8 counts as done with the two unmet marks written down."
    assert M.title_of(TELEGRAM) == 'Lesson (Telegram): "The gap must be the first one after the break."'
    assert M.title_of("<!-- title: HIS OWN TITLE -->\n" + SESSION) == "HIS OWN TITLE"
    answer = TELEGRAM.replace("# Lesson (Telegram,", "# Answer to question Q1, idea-001 (Telegram,")
    assert M.title_of(answer).startswith('Answer to question Q1, idea-001 (Telegram): "The gap')
    assert len(M.title_of("= " + "word " * 100)) <= M.TITLE_MAX + 2


def test_apply_appends_moves_and_never_rewrites(repo):
    _put(repo, "2026-09-29-0705-session.md", SESSION)
    _put(repo, "2026-09-29-1402-telegram.md", TELEGRAM)
    before = (repo / "strategy" / "LEARNED.md").read_bytes()
    done = M.apply(repo, env=MAIN)
    after = (repo / "strategy" / "LEARNED.md").read_bytes()
    assert after.startswith(before)                            # add-only: every old byte is still there, in place
    new = after[len(before):].decode("utf-8")
    assert "## #137 — 2026-09-29. L8 counts as done with the two unmet marks written down.\n\n> Yes" in new
    assert '## #138 — 2026-09-29. Lesson (Telegram): "The gap must be the first one after the break."' in new
    assert "waiting for a number" not in new                   # the inbox heading is dropped
    assert "(from `strategy/learned-inbox/merged/2026-09-29-1402-telegram.md`" in new
    assert sorted(p.name for p in (repo / "strategy" / "learned-inbox").glob("*.md")) == ["README.md"]
    assert sorted(p.name for p in (repo / "strategy" / "learned-inbox" / "merged").iterdir()) == [
        "2026-09-29-0705-session.md", "2026-09-29-1402-telegram.md"]
    assert [w.n for w in done] == [137, 138]
    assert M.apply(repo, env=MAIN) == []                       # nothing left: a second run numbers nothing
    assert "learned-merge: #137-#138 from the inbox (2 files)" in M.message(done)


def test_the_search_reads_the_new_entries(repo):
    _put(repo, "2026-09-29-1402-telegram.md", TELEGRAM)
    M.apply(repo, env=MAIN)
    lines = (repo / "strategy" / "LEARNED.md").read_text(encoding="utf-8").splitlines()
    entries = _learned_entries(lines)
    assert entries[-1].n == 137 and entries[-1].date == "2026-09-29"
    assert sum(1 for ln in lines if LEARNED_HEAD.match(ln)) == 8


def test_apply_keeps_the_files_line_endings(repo):
    crlf = LEARNED.replace("\n", "\r\n").encode("utf-8")
    (repo / "strategy" / "LEARNED.md").write_bytes(crlf)
    _put(repo, "2026-09-29-1402-telegram.md", TELEGRAM)
    M.apply(repo, env=MAIN)
    new = (repo / "strategy" / "LEARNED.md").read_bytes()[len(crlf):]
    assert b"\r\n## #137" in new and new.count(b"\n") == new.count(b"\r\n")


@pytest.mark.parametrize("env", [{}, {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/L9a-bot"},
                                 {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/pull/3/merge"},
                                 {"GITHUB_REF": "refs/heads/main"}])
def test_apply_refuses_off_main_and_changes_nothing(repo, env):
    _put(repo, "2026-09-29-1402-telegram.md", TELEGRAM)
    before = (repo / "strategy" / "LEARNED.md").read_bytes()
    with pytest.raises(SystemExit):
        M.apply(repo, env=env)
    assert (repo / "strategy" / "LEARNED.md").read_bytes() == before
    assert (repo / "strategy" / "learned-inbox" / "2026-09-29-1402-telegram.md").is_file()


def test_the_action_runs_only_on_pushes_to_main():
    text = (config.ROOT / ".github" / "workflows" / "learned-merge.yml").read_text(encoding="utf-8")
    on = re.search(r"^on:\n((?:[ \t]+.*\n|\n)+?)(?=^\S)", text, re.MULTILINE).group(1)
    triggers = re.findall(r"^  (\w+):", on, re.MULTILINE)
    assert triggers == ["push"]                                # no pull_request, no manual run, no schedule
    assert re.search(r"^    branches: \[main\]$", on, re.MULTILINE)
    assert "branches-ignore" not in on and "tags" not in on
    assert re.search(r"^    if: github\.ref == 'refs/heads/main'$", text, re.MULTILINE)
    assert "python -m machine.tools.learned_merge --apply" in text
    assert "ref: main" in text and "cancel-in-progress: false" in text
    assert re.search(r"^permissions:\n  contents: write$", text, re.MULTILINE)


def test_the_other_workflows_never_number_lessons():
    for path in (config.ROOT / ".github" / "workflows").glob("*.yml"):
        if path.name != "learned-merge.yml":
            assert "learned_merge" not in path.read_text(encoding="utf-8"), path.name
