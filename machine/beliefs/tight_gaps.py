"""B4-10 - "Prefer tight gaps" (strategy.md section 2, line 83). ALREADY SEEN (partly - four-boxes B4-10).

One event = his grab + gap trade whose entry gap is among the SMALLEST third of that year's gaps (the gap's size in
day's ranges; the thirds are cut inside each year, so the rule is the same in any year). Counted: its rupees
(machine/beliefs/trades.py).
The fair comparison: his grab + gap trades whose gap is among the LARGEST third of that year's, same close year and
direction. The middle third is only looked at.
Plan: plan/work/beliefs/tight-gaps.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief

FAMILY = "belief-tight-gaps"
SEEN = ("the old project tried a minimum gap size on these years: min_gap_size (5 tries, 2016-2023: any / 10 / 20 / "
        "30 / 50 cents - noise), and #119's smallest-gap filter (noise)")


def thirds(t: pd.DataFrame) -> np.ndarray:
    """1 = the smallest third of the setup's year, 3 = the largest (by the gap's size in day's ranges)."""
    year = pd.DatetimeIndex(t["known_at"]).year
    out = np.zeros(len(t), int)
    for y in np.unique(year):
        m = (year == y) & np.isfinite(t["gap_adr"].to_numpy(float))
        lo, hi = np.quantile(t.loc[m, "gap_adr"], [1 / 3, 2 / 3])
        g = t["gap_adr"].to_numpy(float)
        out[m] = np.where(g[m] <= lo, 1, np.where(g[m] > hi, 3, 2))
    return out


def rows(w):
    t = TR.type1(w)
    k = thirds(t)
    return TR.rows_two_groups(w, k == 1, k == 3, t)


VERSIONS = (
    Belief(FAMILY, "smallest-vs-largest-third", "B4-10", "Tight gaps make better trades",
           "Prefer tight; BOS mandatory.",
           "strategy.md section 2, line 83; four-boxes B4-10",
           "his grab + gap trade whose gap is among the smallest third of that year's gaps (size in day's ranges): "
           "rupees after the Rs 40 cost",
           "until the stop or the target",
           "his grab + gap trades whose gap is among the largest third of that year's, same close year and direction",
           "inr", rows, groups=True, noun="trades", slices=("direction",),
           seen_key=TR.SEEN_KEY, seen_note=SEEN,
           code=("machine/beliefs/tight_gaps.py", *TR.CODE)),
)
