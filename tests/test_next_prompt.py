"""next_prompt on a small made-up PROGRESS table, plus a check that the real table's Needs all name real rows."""
import pytest

from machine import config
from machine.tools.next_prompt import main, parse_progress, plan_for

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


def test_real_table_needs_name_real_rows():
    """A typo in plan/PROGRESS.md's Needs column would block a prompt forever — catch it here."""
    rows = parse_progress(config.PROGRESS_FILE.read_text(encoding="utf-8"))
    known = {r.id for r in rows}
    assert {"L0", "L1", "T1", "L5-prior", "L6.1"} <= known
    unknown = {(r.id, n) for r in rows for n in r.needs + r.soft if n not in known}
    assert not unknown, f"Needs that name no row: {sorted(unknown)}"
    assert all(r.owner in ("Tejas", "Likhith", "either") for r in rows), [(r.id, r.owner) for r in rows]
