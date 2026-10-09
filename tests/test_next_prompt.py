"""next_prompt on a small made-up PROGRESS table, plus a check that the real table's Needs all name real rows."""
import json

import pytest

from machine import config
from machine.tools import next_prompt
from machine.tools.next_prompt import count_labels, main, parse_progress, plan_for

FAKE = """# Progress — a made-up table for the test

| Stage | ID | Owner | What | Needs | Runs on | Status | Date | Note |
|---|---|---|---|---|---|---|---|---|
| 0 | L0 | Likhith | Workshop | — | ☁ 💻 | ✅ | 2026-09-28 | |
| 0 | T1 | Tejas | One true strategy.md | — | ☁ | ✅ | 2026-09-27 | |
| 0 | L1 | Likhith | Price data | L0 | 💻 | ⬜ | | copy the 3 CSVs first |
| 0 | L2 | Likhith | Cost | L0, L1 | ☁ 💻 | ⬜ | | |
| 1 | L6.2 | Likhith | Liquidity blocks | L0 (soft: T2, T3) | 💻 | ⬜ | | |
| 1 | L5 ⭐ | Likhith + Tejas pairs | Referee | L3, L5-prior | 💻 | ⬜ | | |
| 1 | L8 ⭐ | **Tejas** | Talker | L0, T1 | ☁ | ⬜ | | |
| 1 | L9 | Tejas | Waits for a row that is not there | L99 | ☁ | ⬜ | | |
| 1 | L7 | Likhith | Already taken | L0 | ☁ | 🟡 | 2026-09-28 | |
| 0 | T7 | Tejas | Lower stage, later in the table | T1 | ☁ | ⬜ | | |
| 6 | L18 | Likhith | Locked | L0 | 🪟 | 🔒 | | |
| any | L6-new | either | Add a missing block | L0 | 💻 | ⬜ | | reusable |

## Log

| Date | Who | ID | What happened |
|---|---|---|---|
| 2026-09-28 | Claude | L0 | not a progress row |
"""


@pytest.fixture
def rows():
    return parse_progress(FAKE)


def ids(items):
    return [i.row.id for i in items]


def test_parse(rows):
    by_id = {r.id: r for r in rows}
    assert len(rows) == 12                                   # the Log table is not read
    assert by_id["L5"].owner == "Likhith"                    # "⭐" dropped; the first name is the owner
    assert by_id["L8"].owner == "Tejas"                      # "**Tejas**"
    assert by_id["L6-new"].owner == "either"
    assert by_id["L2"].needs == ["L0", "L1"]
    assert by_id["L5"].needs == ["L3", "L5-prior"]
    assert by_id["L6.2"].needs == ["L0"] and by_id["L6.2"].soft == ["T2", "T3"]
    assert by_id["L0"].needs == []


def test_likhith(rows):
    plan = plan_for(rows, "Likhith")
    assert ids(plan.ready) == ["L1", "L6.2", "L6-new"]       # stage 0, then 1, then "any"
    assert ids(plan.blocked) == ["L2", "L5"]
    assert plan.blocked[0].waits == ["L1"]
    assert plan.blocked[1].waits == ["L3", "L5-prior"]       # rows that are not in the table count as not done
    assert plan.ready[1].soft_missing == ["T2", "T3"]        # soft needs never block
    assert ids(plan.taken) == ["L7"]                         # ✅ and 🔒 rows are not listed


def test_tejas(rows):
    plan = plan_for(rows, "Tejas")
    assert ids(plan.ready) == ["T7", "L8", "L6-new"]         # lowest stage first, even if later in the table
    assert ids(plan.blocked) == ["L9"]
    assert plan.blocked[0].waits == ["L99"]


def test_main_prints_ready_then_blocked(tmp_path, capsys):
    progress = tmp_path / "PROGRESS.md"
    progress.write_text(FAKE, encoding="utf-8")
    assert main(["--who", "likhith", "--file", str(progress)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Ready for Likhith, best first. Paste L1:")
    assert "note: copy the 3 CSVs first" in out
    assert "soft need T2, T3 not done: that step waits" in out
    assert out.index("L1 ") < out.index("Blocked, waiting for:") < out.index("waits for L1 (Likhith, ⬜)")
    assert "waits for L3 (not in the table), L5-prior (not in the table)" in out


def test_unknown_person_is_refused():
    with pytest.raises(SystemExit):
        main(["--who", "Bob"])


LABEL_TABLE = """| Stage | ID | Owner | What | Needs | Runs on | Status | Date | Note |
|---|---|---|---|---|---|---|---|---|
| 4 | L15.1 | Likhith | Labels | — | 💻 | ✅ | 2026-09-30 | |
| 4 | L15.2 ⭐ | Likhith (Tejas if he says yes) | Two models | L15.1 | 💻 | ⬜ | | owner pending; ≥ 600 labels |
| 4 | L15.3 | Likhith | The count in the Needs cell | L15.1 (≥ 1,000 labels) | 💻 | ⬜ | | |
| 5 | L16.1 | Likhith | No count | L15.1 | 🪟 | ⬜ | | |
"""


def test_a_label_count_holds_a_prompt_back_until_it_is_reached():
    """2 Oct 2026: the list said "Paste L15.2" with 16 labels of the 600 it needs; L15.2 stopped at its own gate."""
    rows = parse_progress(LABEL_TABLE)
    by_id = {r.id: r for r in rows}
    assert by_id["L15.2"].labels_needed == 600 and by_id["L15.2"].needs == ["L15.1"]   # read from the Note cell
    assert by_id["L15.3"].labels_needed == 1000 and by_id["L15.3"].needs == ["L15.1"]  # or from the Needs cell
    assert by_id["L16.1"].labels_needed == 0
    asked = []

    def sixteen():
        asked.append(1)
        return 16

    plan = plan_for(rows, "Likhith", labels_now=sixteen)
    assert ids(plan.ready) == ["L16.1"]
    assert ids(plan.blocked) == ["L15.2", "L15.3"]
    assert plan.blocked[0].waits == [] and plan.blocked[0].short == ["600 labels (16 so far)"]
    assert plan.blocked[1].short == ["1,000 labels (16 so far)"]
    assert asked == [1]                                      # counted once for both rows
    plan = plan_for(rows, "Likhith", labels_now=lambda: 600)
    assert ids(plan.ready) == ["L15.2", "L16.1"] and ids(plan.blocked) == ["L15.3"]


def test_labels_are_counted_only_when_a_row_waits_for_them(rows):
    def never():
        raise AssertionError("counted labels for a table that needs none")

    assert ids(plan_for(rows, "Likhith", labels_now=never).ready) == ["L1", "L6.2", "L6-new"]
    label_rows = parse_progress(LABEL_TABLE)
    assert plan_for(label_rows, "Tejas", labels_now=never).ready == []   # L15.x are Likhith's rows


def test_a_count_that_fails_holds_the_prompt_and_says_why():
    def broken():
        raise OSError("replies.jsonl is locked")

    plan = plan_for(parse_progress(LABEL_TABLE), "Likhith", labels_now=broken)
    assert ids(plan.blocked) == ["L15.2", "L15.3"]
    assert plan.blocked[0].short == ["600 labels (could not count them: replies.jsonl is locked)"]


def test_main_prints_the_label_count(tmp_path, capsys, monkeypatch):
    progress = tmp_path / "PROGRESS.md"
    progress.write_text(LABEL_TABLE, encoding="utf-8")
    monkeypatch.setattr(next_prompt, "count_labels", lambda: 16)
    assert main(["--who", "Likhith", "--file", str(progress)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Ready for Likhith, best first. Paste L16.1:")
    assert "L15.2  Two models  · runs on 💻  · waits for 600 labels (16 so far)" in out


def test_count_labels_counts_his_drawings_and_the_charts_he_answered(tmp_path):
    from machine.labels import drawings
    folder = tmp_path / "strong_zone"
    folder.mkdir()
    sent = [{"qid": f"LBL-strong_zone-20261001-0{n}"} for n in (1, 2, 3)]
    said = [{"qid": "LBL-strong_zone-20261001-01", "label": "yes"},
            {"qid": "LBL-strong_zone-20261001-01", "label": "no"},       # he changed his mind: still one label
            {"qid": "LBL-strong_zone-20261001-02", "label": "not sure"}]  # "not sure" is a real answer (T9)
    (folder / "charts.jsonl").write_text("".join(json.dumps(c) + "\n" for c in sent), encoding="utf-8")
    (folder / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in said), encoding="utf-8")
    drawn = len(drawings.labels("strong_zone"))
    assert drawn == 6                                        # his answers on Q78 / Q79 (L15.1)
    assert count_labels(tmp_path) == drawn + 2               # chart 3 was not answered
    assert count_labels(tmp_path / "nothing sent yet") == drawn


def test_real_table_needs_name_real_rows():
    """A typo in plan/PROGRESS.md's Needs column would block a prompt forever — catch it here."""
    rows = parse_progress(config.PROGRESS_FILE.read_text(encoding="utf-8"))
    known = {r.id for r in rows}
    assert {"L0", "L1", "T1", "L5-prior", "L6.1"} <= known
    unknown = {(r.id, n) for r in rows for n in r.needs + r.soft if n not in known}
    assert not unknown, f"Needs that name no row: {sorted(unknown)}"
    assert all(r.owner in ("Tejas", "Likhith", "either") for r in rows), [(r.id, r.owner) for r in rows]


IDEA_TABLE = """| Stage | ID | Owner | What | Needs | Runs on | Status | Date | Note |
|---|---|---|---|---|---|---|---|---|
| 5 | L17a | Likhith | Live feed | — | 🪟 | ✅ | 2026-10-09 | |
| 5 | L17b | Likhith | Paper trades for an idea that passed | L17a | 🪟 📱 | ⬜ | | + an idea that "works" or a L17c plan |
| 5 | L17c | Likhith | Forward test | L17a | 🪟 📱 | ⬜ | | |
"""


def test_l17b_waits_for_an_idea_that_works_or_a_forward_plan():
    """9 Oct 2026: L17a was ✅ and the list said "Paste L17b" - but no idea had passed and none had a forward plan,
    so L17b would stop at its own gate."""
    rows = parse_progress(IDEA_TABLE)
    by_id = {r.id: r for r in rows}
    assert by_id["L17b"].idea_needed and not by_id["L17c"].idea_needed
    plan = plan_for(rows, "Likhith", ideas_now=lambda: 0)
    assert ids(plan.ready) == ["L17c"] and ids(plan.blocked) == ["L17b"]
    assert plan.blocked[0].waits == [] and plan.blocked[0].short == ['an idea that "works" or a forward plan (none yet)']
    assert ids(plan_for(rows, "Likhith", ideas_now=lambda: 1).ready) == ["L17b", "L17c"]

    def broken():
        raise OSError("state.json is locked")

    assert plan_for(rows, "Likhith", ideas_now=broken).blocked[0].short == [
        'an idea that "works" or a forward plan (could not count them: state.json is locked)']
    real = parse_progress(config.PROGRESS_FILE.read_text(encoding="utf-8"))
    assert {r.id: r for r in real}["L17b"].idea_needed                      # the real row says it


def test_count_l17b_ideas_counts_a_live_trial_and_a_written_forward_plan(monkeypatch):
    from machine.ideas import store
    from machine.intake import state as S
    from machine.referee.verdict import WORKS
    states = {"idea-001": {"waits_for": S.PLANNED},                        # its forward plan is written (L17c)
              "idea-002": {"waits_for": S.choice(WORKS, False)[2]},        # the machine's LIVE TRIAL
              "idea-003": {"waits_for": S.FORWARD},                        # its plan not written yet
              "idea-004": {"waits_for": S.PARK}, "idea-005": {}}
    monkeypatch.setattr(store, "ideas", lambda: list(states))
    monkeypatch.setattr(store, "read_state", lambda i: states[i])
    assert next_prompt.count_l17b_ideas() == 2
