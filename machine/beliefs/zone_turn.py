"""B4-11a - "A zone marks where a turn is likely" (strategy.md section 3, line 88).

One event = the FIRST touch of one of his QUIET zones (tier 1, his main zone: machine/blocks/zones.py; rules.py
what_a_strong_zone_is, calm at most 0.74 - #79, #80, #81): a candle entering the band after one wholly outside it.
Counted: does price turn - reach the next liquidity back the way it came, from the edge it met - before it goes as far
past the zone's FAR edge? (levels.turn_race.)
The fair comparison: "random bands of the same size" - 20 fake bands per zone, the same height, born at the same
weekday and hour of the same year, their middle as far from price as the zone's; each gets its first touch the same
way, then the same race, as far.
The rupee half of his sentence ("a sweep + gap at a strong zone is high-probability") is B4-11b (zone_grab_gap.py).
Plan: plan/work/beliefs/zone-turn.md.
"""
from __future__ import annotations

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-zone-turn"


def first_touches(w, tiers=(1, 2)):
    t = EV.zone_touches(w)
    return t[(t["touch"] == 1) & t["tier"].isin(tiers)].reset_index(drop=True)


def rows(w):
    t = first_touches(w, tiers=(1,))
    return LV.frame(w, t["pos"], t["x"], LV.pools(list(t["pool"])), window=t["window"],
                    side=t["side"].map({"high": "from below", "low": "from above"}))


VERSIONS = (
    Belief(FAMILY, "quiet-zones", "B4-11", "A zone marks where a turn is likely",
           "Uses: high-prob sweep+FVG at a zone; marks where reversal is likely",
           "strategy.md section 3, line 88; four-boxes B4-11",
           "the first touch of one of his quiet zones (tier 1, his main zone): does price turn - reach the next "
           "liquidity back the way it came, from the edge it met - before it goes as far past the zone's far edge?",
           "until one of the two is reached, at most 5 days",
           "random bands of the same size: 20 fake bands per zone (the same height, born at the same weekday and hour "
           "of the same year, as far from price), each first touched the same way, then the same race",
           "share", rows, noun="zone touches", slices=("window", "side"),
           code=("machine/beliefs/zone_turn.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
