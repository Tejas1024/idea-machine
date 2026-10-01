"""B4-12 - "1-hour zones are stronger than 5-min zones" (strategy.md section 3, line 88: "Hand-drawn 5m (1h stronger)").

One event = the first touch of a zone from the HOUR window (his zones drawn on the hour candles of the last 20 days -
settings "zone_windows"). Counted: its turn (the race of B4-11a) less its own 20 fake bands' turns - the zone's pull,
fair for any band height (an hour zone is taller, and a taller band turns more often by itself).
The fair comparison: the same pull of the first touches of the 5-minute zones (the "today" and "week" windows), of the
same year and side. His strong zones, tiers 1 and 2 (settings "zone_tiers" = tiered).
Plan: plan/work/beliefs/hour-zones.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.zone_turn import first_touches

FAMILY = "belief-hour-zones"


def pulls(w, tiers=(1, 2)):
    t = first_touches(w, tiers)
    t["pull"] = [x - p.mean() if np.isfinite(x) and len(p) >= 5 else np.nan
                 for x, p in zip(t["x"], LV.pools(list(t["pool"])))]
    t["year"] = LV.year_of(w, t["pos"])
    return t


def rows(w):
    t = pulls(w)
    return LV.group_frame(w, t["pos"], t["pull"], (t["window"] == "hour").to_numpy(), LV.stratum(w, t["pos"], t["side"]),
                          tier=t["tier"].astype(str), side=t["side"].map({"high": "from below", "low": "from above"}))


VERSIONS = (
    Belief(FAMILY, "hour-vs-5-minute", "B4-12", "1-hour zones are stronger than 5-minute zones",
           "Hand-drawn 5m (1h stronger).",
           "strategy.md section 3, line 88; LEARNED #78 ('I usually consider five minutes and one hour as well'); "
           "four-boxes B4-12",
           "the first touch of a zone drawn on the hour candles: how much more often price turns there than at its own "
           "20 fake bands (in 100)",
           "until the next liquidity or as far past the far edge, at most 5 days",
           "the same pull of the first touches of his 5-minute zones (today and week windows), same year and side",
           "share", rows, noun="zone touches", slices=("tier", "side"), groups=True,
           code=("machine/beliefs/hour_zones.py", "machine/beliefs/zone_turn.py", "machine/beliefs/events.py",
                 "machine/beliefs/levels.py")),
)
