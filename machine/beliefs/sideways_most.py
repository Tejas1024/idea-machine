"""B4-22 - "Most of the time gold is sideways" (strategy.md section 14, line 163; section 10, line 131).

One event = one broker week (Monday to Friday, server clock; at least 4 days of candles). The trend measure, chosen
BEFORE looking (four-boxes B4-22: "by one trend measure chosen before looking"; Claude's): the week is SIDEWAYS when it
ended less than half of its range away from where it started - |last close - first open| < 0.5 x (highest - lowest),
on the 5-minute closes. Counted: sideways, yes or no.
The fair comparison: the same week with its moves in random directions - each 5-minute move keeps its size, its sign
is a coin (20 times) - a market with the week's own movement and no memory. "Most of the time" is also shown plainly:
how many weeks in 100 are sideways by this measure.
Plan: plan/work/beliefs/sideways-most.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-sideways-most"
SIDEWAYS_BELOW = 0.5            # |net move| / range - Claude's, chosen before looking
MIN_CANDLES = 4 * 276           # at least 4 trading days of candles


def efficiency(path: np.ndarray) -> np.ndarray:
    """|end - start| / (highest - lowest) of each path (rows)."""
    rng = path.max(axis=-1) - path.min(axis=-1)
    return np.where(rng > 0, np.abs(path[..., -1] - path[..., 0]) / rng, 0.0)


def rows(w):
    monday = (w.t.normalize() - pd.to_timedelta(w.t.dayofweek, unit="D")).as_unit("s").asi8 // 86_400
    rng = w.rng("sideways_most")
    x, pool, pos, month = [], [], [], []
    for wk in np.unique(monday[w.judged]):
        idx = np.flatnonzero(monday == wk)
        idx = idx[w.weekday[idx] <= 4]
        if len(idx) < MIN_CANDLES:
            continue
        path = np.r_[w.open[idx[0]], w.close[idx]]
        steps = np.diff(path)
        x.append(float(efficiency(path[None, :])[0] < SIDEWAYS_BELOW))
        signs = rng.choice((-1.0, 1.0), size=(LV.K_FAKES, len(steps)))
        fake = np.concatenate([np.full((LV.K_FAKES, 1), path[0]), path[0] + np.cumsum(signs * steps, axis=1)], axis=1)
        pool.append((efficiency(fake) < SIDEWAYS_BELOW).astype(float))
        pos.append(idx[-1])
        month.append(int(w.t[idx[0]].month))
    q = np.array(["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"])[(np.array(month, dtype=int) - 1) // 3] if month else []
    return LV.frame(w, np.array(pos, dtype=np.int64), np.array(x), pool, quarter=q)


VERSIONS = (
    Belief(FAMILY, "half-the-range", "B4-22", "Most of the time gold is sideways",
           "Majority of the time gold is SIDEWAYS, not trending -> the grid/sideways assumption usually holds; trends "
           "are the exception.",
           "strategy.md section 14, line 163; section 10, line 131; four-boxes B4-22",
           "one broker week: is it sideways - did it end less than half of its range away from where it started "
           "(on the 5-minute closes)?",
           "one week, Monday to Friday",
           "the same week with each 5-minute move in a random direction (its size kept), 20 times: a market with the "
           "week's own movement and no memory",
           "share", rows, noun="weeks", slices=("quarter",),
           code=("machine/beliefs/sideways_most.py", "machine/beliefs/levels.py")),
)
