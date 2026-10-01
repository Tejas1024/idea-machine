"""B4-03 - "Equal highs / lows are a magnet, timing unknown" (strategy.md section 1, line 75).

One event = an equal high or low (two touches, his rule - machine/blocks/equal_levels.py, on the 5-minute, hour and day
charts) at the moment it is known. Counted: does price come to it (a high at or above an equal high) within N days?
His measure names four windows - 1, 3, 5 and 10 days - so there are four versions; the tune years choose one.
The fair comparison: 20 fake levels per real one, born at the same weekday and hour of the same year, as far from
price (in day's ranges), on the same side - the same count.
Close relative, not the same belief: this machine's idea #1 TRADED equal highs / lows (take one side -> the other).
Plan: plan/work/beliefs/equal-magnet.md.
"""
from __future__ import annotations

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-equal-magnet"
DAYS = (1, 3, 5, 10)


def _rows(days: int):
    def rows(w):
        eb = EV.equal_births(w)
        eb = eb[eb["ok"] & (eb["touches"] == 2)].reset_index(drop=True)
        x, pool = EV.magnet(w, eb, days)
        return LV.frame(w, eb["pos"], x, pool, chart=eb["tf"], side=eb["side"])
    return rows


VERSIONS = tuple(
    Belief(FAMILY, f"{d}-day{'s' if d > 1 else ''}", "B4-03", "Equal highs / lows are a magnet, timing unknown",
           "Type 3 Equal highs/lows #1 (L2). Later touch must NOT pass reference; exact strongest; 3>2. A magnet, "
           "timing unknown.",
           "strategy.md section 1, line 75; four-boxes B4-03",
           "an equal high (low) when it is known: does price come to it - a high at or above it (a low at or below) "
           f"- within {d} day{'s' if d > 1 else ''}?",
           f"{d} day{'s' if d > 1 else ''} ({d * 288:,} five-minute candles) from the moment it is known",
           "20 fake levels per real one - born at the same weekday and hour of the same year, as far from price in "
           "day's ranges, on the same side - the same count",
           "share", _rows(d), noun="equal levels", slices=("chart", "side"),
           code=("machine/beliefs/equal_magnet.py", "machine/beliefs/events.py", "machine/beliefs/levels.py"))
    for d in DAYS)
