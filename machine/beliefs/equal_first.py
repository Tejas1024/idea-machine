"""B4-02 - "Equal highs / lows are the #1 liquidity; ranking Equal > Trendline > Session" (strategy.md section 1,
lines 75 and 78).

Version "turns" (a count): the takes of B4-01 (machine/beliefs/grab_then_move.py), the equal highs / lows against
the takes of his other two ranked kinds - session highs / lows and trendlines - of the same year and side: after an
equal level is taken, does price reach the next liquidity the other way MORE often? The race is the same distance
each way, so every kind starts from a coin (50 in 100) whatever its distances.
Version "rupees": his grab + gap trades (machine/beliefs/trades.py), the ones whose grab took equal highs / lows,
against those whose grab took a session high / low or a trendline, same close year and direction.
ALREADY SEEN: the old project raced his grab + gap by the kind of level taken on these years (early_sweep_direction:
"at a trendline, at a session high/low, equal highs/lows swept", 28 tries, 2016-2026).
His whole ranking is shown in B4-01's report (by kind, each against its own fake levels).
Plan: plan/work/beliefs/equal-first.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief

FAMILY = "belief-equal-first"
RANKED = ("equal", "trendline", "session")
SEEN = ("the old project raced his grab + gap on these years by the kind of level taken: early_sweep_direction "
        "(groups 'at a trendline', 'at a session high/low', 'equal highs/lows swept'; 28 tries, 2016-2026: 47-51%)")


def rows(w):
    t = EV.takes(w)
    t = t[t["ok"] & t["family"].isin(RANKED)].reset_index(drop=True)
    return LV.group_frame(w, t["pos"], t["x"], (t["family"] == "equal").to_numpy(), LV.stratum(w, t["pos"], t["side"]),
                          side=t["side"], chart=t["tf"])


def rupees(w):
    t = TR.type1(w)
    fam = t["family"].to_numpy()
    return TR.rows_two_groups(w, fam == "equal", np.isin(fam, ("session", "trendline")), t)


VERSIONS = (
    Belief(FAMILY, "turns", "B4-02", "Equal highs / lows are the #1 liquidity (Equal > Trendline > Session)",
           "Type 3 Equal highs/lows #1 ... Ranking: Equal > Trendline > Session.",
           "strategy.md section 1, lines 75 and 78; four-boxes B4-02",
           "a fresh take of equal highs / lows: does price reach the next liquidity on the other side before it goes "
           "as far past the level? (the race of B4-01)",
           "until one of the two is reached, at most 5 days",
           "the takes of his other ranked kinds - session highs / lows and trendlines - of the same year and side, "
           "the same race",
           "share", rows, noun="takes", slices=("side", "chart"), seen_note=SEEN, groups=True,
           code=("machine/beliefs/equal_first.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
    Belief(FAMILY, "rupees", "B4-02", "Equal highs / lows are the #1 liquidity (Equal > Trendline > Session)",
           "Type 3 Equal highs/lows #1 ... Ranking: Equal > Trendline > Session.",
           "strategy.md section 1, lines 75 and 78; four-boxes B4-02",
           "his grab + gap trade whose grab took equal highs / lows: rupees after the Rs 40 cost, 0.01 lot, a plain "
           "stop at his hedge point, the next liquidity as the target, every setup its own trade",
           "until the stop or the target (no time limit, #68 / no timeout)",
           "his grab + gap trades whose grab took a session high / low or a trendline, same close year and direction",
           "inr", rupees, noun="trades", slices=("direction",), seen_note=SEEN, groups=True,
           code=("machine/beliefs/equal_first.py", *TR.CODE)),
)
