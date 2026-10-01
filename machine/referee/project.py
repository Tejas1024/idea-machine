"""The whole project's score (L5, step 9): ideas judged, how many "works", and how many "works" luck alone would give.

Read from the referee's trial files (kind "referee" in plan/work/trials/), never from memory.
- false "works" expected by luck: each judged answer that could have said "works" (not already seen) has at most a
  config.P_LIMIT (5 in 100) chance of saying it when the idea has no edge — the noise test (tests/test_referee.py)
  measures the real rate, which is lower. So 20 judged ideas with no edge would still give about 1 "works".
- not a "works" in a row: README section 7 — after 10 in a row the machine asks Tejas: keep going, change direction,
  or stop.
"""
from __future__ import annotations

from pathlib import Path

from machine import config
from machine.referee import splits, trials
from machine.referee.verdict import FRESH, WORDS, WORKS

ASK_AFTER = 10          # README section 7: after 10 answers in a row that are not "works", ask Tejas


def summary(folder: Path | None = None) -> dict:
    files = [f for f in trials.load(folder) if f.get("kind") == "referee"]
    answers = [f["result"]["answer"]["word"] for f in files]
    eligible = sum(w != FRESH for w in answers)
    in_a_row = 0
    for w in reversed(answers):
        if w == WORKS:
            break
        in_a_row += 1
    links = trials.parents(trials.load(folder))
    return {
        "answers": len(answers),
        "families_judged": len({trials.root_of(f["idea_id"], links) for f in files}),
        "by_word": {w: answers.count(w) for w in WORDS},
        "works": answers.count(WORKS),
        "false_works_expected": round(eligible * config.P_LIMIT, 2),
        "not_works_in_a_row": in_a_row,
        "ask_tejas": in_a_row >= ASK_AFTER,
        "check_opened": splits.opened_count(),
    }


def text(s: dict) -> list[str]:
    lines = [
        f"CHECK years opened {s['check_opened']} times in the project",
        f"Project: {s['answers']} answers for {s['families_judged']} ideas · \"works\" {s['works']} · "
        f"luck alone would give about {s['false_works_expected']:g} \"works\" by now "
        f"(up to 5 in every 100 answers, even when no idea has an edge)",
    ]
    if s["ask_tejas"]:
        lines.append(f"{s['not_works_in_a_row']} answers in a row are not \"works\": time to ask Tejas — keep going, "
                     "change direction, or stop (README section 7)")
    return lines
