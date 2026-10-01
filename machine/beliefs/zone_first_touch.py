"""B4-13a - "A zone is strongest the first time" (strategy.md section 3, line 89; LEARNED #77).

One event = the FIRST touch of one of his strong zones (tiers 1 and 2). Counted: does price turn there (the race of
B4-11a: the next liquidity back the way it came, before as far past the far edge)?
The fair comparison: the 2nd and 3rd touches of his strong zones (the zone ends at the 3rd - rules.py
weak_after_touches, #77 "two to three times"), of the same year and side, the same race. The same zones, so the same
band heights.
Plan: plan/work/beliefs/zone-first-touch.md.
"""
from __future__ import annotations

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-zone-first-touch"


def rows(w):
    t = EV.zone_touches(w)
    t = t[t["tier"].isin((1, 2))].reset_index(drop=True)
    return LV.group_frame(w, t["pos"], t["x"], (t["touch"] == 1).to_numpy(), LV.stratum(w, t["pos"], t["side"]),
                          tier=t["tier"].astype(str), side=t["side"].map({"high": "from below", "low": "from above"}))


VERSIONS = (
    Belief(FAMILY, "first-vs-later", "B4-13", "A zone is strongest the first time",
           "Once. It is touched and uh, respected, then that becomes weak.",
           "LEARNED #77; strategy.md section 3, line 89; four-boxes B4-13",
           "the first touch of one of his strong zones: does price turn - reach the next liquidity back the way it "
           "came - before it goes as far past the zone's far edge?",
           "until one of the two is reached, at most 5 days",
           "the 2nd and 3rd touches of his strong zones, same year and side, the same race",
           "share", rows, noun="zone touches", slices=("tier", "side"), groups=True,
           code=("machine/beliefs/zone_first_touch.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
