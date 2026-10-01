"""B4-13b - "Two quiet stretches at the same price make it stronger" (strategy.md section 3, line 89; LEARNED #79).

One event = the first touch of one of his QUIET zones (tier 1) that has at least one EARLIER quiet zone, built from
other candles, at the same price (zones.py `quiet_before`, the measure L6b.2 built for this belief). Counted: its pull
- the turn of B4-11a less its own 20 fake bands' turns.
The fair comparison: the same pull of the first touches of quiet zones with NO earlier quiet zone at their price, of the
same year and side.
Plan: plan/work/beliefs/zone-two-quiet.md.
"""
from __future__ import annotations

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.hour_zones import pulls

FAMILY = "belief-zone-two-quiet"


def rows(w):
    t = pulls(w, tiers=(1,))
    return LV.group_frame(w, t["pos"], t["pull"], (t["quiet_before"] >= 1).to_numpy(),
                          LV.stratum(w, t["pos"], t["side"]), window=t["window"],
                          side=t["side"].map({"high": "from below", "low": "from above"}))


VERSIONS = (
    Belief(FAMILY, "two-vs-one", "B4-13", "Two quiet stretches at one price make a zone stronger",
           "Two quiet stretches at the same price make it stronger.",
           "strategy.md section 3, line 89 (#79); four-boxes B4-13",
           "the first touch of a quiet zone with an earlier quiet zone at the same price: how much more often price "
           "turns there than at its own 20 fake bands (in 100)",
           "until the next liquidity or as far past the far edge, at most 5 days",
           "the same pull of quiet zones with no earlier quiet zone at their price, same year and side",
           "share", rows, noun="zone touches", slices=("window", "side"), groups=True,
           code=("machine/beliefs/zone_two_quiet.py", "machine/beliefs/hour_zones.py", "machine/beliefs/zone_turn.py",
                 "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
