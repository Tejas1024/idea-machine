"""B4-04a - "Exact equal is the strongest" (strategy.md section 1, line 75; LEARNED #89, #90).

One event = an EXACT equal high or low (the second touch 0-2 cents from the first - #89 "it is perfect if it is the
exact price ... slight one or two point is okay") when it is known. Counted: how much more often price comes to it
within 5 days than to its own 20 fake levels (the B4-03 count, less its fake levels: the magnet's pull, fair for any
distance).
The fair comparison: the same pull of the equal levels that are 3-6 cents apart (still equal by his rule, B1-07), of
the same year and side.
One window, fixed before looking: 5 days (the middle of his four; B4-03 tries all four). Plan:
plan/work/beliefs/equal-exact.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-equal-exact"
DAYS = 5
EXACT_CENTS = 2


def pull(w, touches: int):
    """The equal levels with `touches` touches, each with its pull: reached within 5 days, less its fake levels."""
    def make():
        eb = EV.equal_births(w)
        eb = eb[eb["ok"] & (eb["touches"] == touches)].reset_index(drop=True)
        x, pool = EV.magnet(w, eb, DAYS)
        eb["pull"] = [xi - p.mean() if np.isfinite(xi) and len(p) >= 5 else np.nan for xi, p in zip(x, pool)]
        eb["year"] = LV.year_of(w, eb["pos"])
        return eb
    return w.memo(("equal_pull", touches), make)


def rows(w):
    eb = pull(w, 2)
    eb = eb[np.isfinite(eb["gap_cents"])].reset_index(drop=True)
    return LV.group_frame(w, eb["pos"], eb["pull"], (eb["gap_cents"] <= EXACT_CENTS).to_numpy(),
                          LV.stratum(w, eb["pos"], eb["side"]), chart=eb["tf"], side=eb["side"])


VERSIONS = (
    Belief(FAMILY, "0-2-cents", "B4-04", "Exact equal highs / lows pull price the most",
           "it is perfect if it is the exact price of both equal highs or equal lows uh, but slight one or two point is "
           "okay",
           "LEARNED #89; strategy.md section 1, line 75 ('exact strongest'); four-boxes B4-04",
           "an equal high (low) whose two touches are 0-2 cents apart, when it is known: how much more often price "
           "comes to it within 5 days than to its own 20 fake levels (in 100)",
           "5 days (1,440 five-minute candles) from the moment it is known",
           "the same pull of the equal levels 3-6 cents apart, of the same year and side",
           "share", rows, noun="equal levels", slices=("chart", "side"), groups=True,
           code=("machine/beliefs/equal_exact.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
