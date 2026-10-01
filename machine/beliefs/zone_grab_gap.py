"""B4-11b - "A sweep + gap at a strong zone is high-probability" (strategy.md section 3, line 88). ALREADY SEEN.

One event = his grab + gap trade whose entry price (the gap's near edge) is INSIDE one of his strong zones alive at the
setup's candle (machine/blocks/zones.py, tiers 1 and 2). Counted: its rupees (machine/beliefs/trades.py).
The fair comparison: his grab + gap trades not at a zone, same close year and direction.
The count half ("a zone marks where a turn is likely") is B4-11a (zone_turn.py).
Plan: plan/work/beliefs/zone-grab-gap.md.
"""
from __future__ import annotations

from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief

FAMILY = "belief-zone-grab-gap"
SEEN = ("the old project split his grab + gap by 'at a zone' on these years: early_sweep_direction (the group 'at a "
        "zone', 2016-2026) and early_sweep_126_combos (the filter 'zone'); zone_hedge_aim_small and double_profit "
        "also used his zones with it (2016-2023)")


def rows(w):
    t = TR.type1(w)
    z = t["at_zone"].to_numpy(bool)
    return TR.rows_two_groups(w, z, ~z, t)


VERSIONS = (
    Belief(FAMILY, "at-zone-vs-not", "B4-11", "A sweep + gap at a strong zone is high-probability",
           "Uses: high-prob sweep+FVG at a zone; marks where reversal is likely",
           "strategy.md section 3, line 88; four-boxes B4-11",
           "his grab + gap trade whose entry price is inside one of his strong zones: rupees after the Rs 40 cost",
           "until the stop or the target",
           "his grab + gap trades whose entry is not inside a zone, same close year and direction",
           "inr", rows, groups=True, noun="trades", slices=("direction",),
           seen_key=("liquidity_sweep+break_of_structure+fvg+strong_zones", "5m", "gap:near"), seen_note=SEEN,
           code=("machine/beliefs/zone_grab_gap.py", *TR.CODE)),
)
