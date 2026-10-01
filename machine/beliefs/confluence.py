"""B4-19 - "More confluence makes a trade better" (strategy.md section 8, line 113). ALREADY SEEN (partly).

His sentence names the things: "More confluence (liquidity + zone + ADR + HTF) makes a trade better, but it is never a
minimum count" (#15, #16, #17, #51, #84, #97 - the count is never a RULE; here it is only measured). One point for each:
  liquidity   the grab took his top-ranked liquidity - equal highs / lows or a trendline ("Equal > Trendline > Session")
  zone        the entry price is inside one of his strong zones
  ADR         a buy at discount, a sell at premium (near his green / red zone)
  HTF         the hour AND the day bias with the trade
One event = his grab + gap trade with 2 or more of the 4. Counted: its rupees (machine/beliefs/trades.py).
The fair comparison: his grab + gap trades with 0 or 1 of them, same close year and direction.
Plan: plan/work/beliefs/confluence.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief

FAMILY = "belief-confluence"
SEEN = ("the old project measured his entries under 750 combinations of filters on these years (combo_750, "
        "2016-2023) and counted confluence for AMBIGUITY-REGISTER A1 - four-boxes B4-19 'partly'")


def points(t) -> np.ndarray:
    return (t["family"].isin(("equal", "trendline")).to_numpy(int) + t["at_zone"].to_numpy(int)
            + t["adr_with"].to_numpy(int) + t["htf_with"].to_numpy(int))


def rows(w):
    t = TR.type1(w)
    k = points(t)
    t = t.assign(points=k.astype(str))
    return TR.rows_two_groups(w, k >= 2, k <= 1, t, slices={"points": "points"})


VERSIONS = (
    Belief(FAMILY, "two-or-more-vs-one-or-none", "B4-19", "More confluence makes a trade better",
           "More confluence (liquidity + zone + ADR + HTF) makes a trade better, but it is never a minimum count - "
           "each setup is judged on its own",
           "strategy.md section 8, line 113; four-boxes B4-19",
           "his grab + gap trade with 2 or more of the 4 things agreeing (his top liquidity, a zone, ADR, the hour and "
           "day): rupees after the Rs 40 cost",
           "until the stop or the target",
           "his grab + gap trades with 0 or 1 of them, same close year and direction",
           "inr", rows, groups=True, noun="trades", slices=("points", "direction"),
           seen_key=TR.SEEN_KEY, seen_note=SEEN,
           code=("machine/beliefs/confluence.py", *TR.CODE)),
)
