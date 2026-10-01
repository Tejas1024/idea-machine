"""B4-08b - "Unfilled gaps are a magnet - gap-ups and gap-downs too" (strategy.md section 2, line 84; LEARNED #87).

One event = a GAP-UP or GAP-DOWN: a 5-minute candle that OPENS at least 0.10 of a day's range away from the close of
the candle before it (the Monday open, the open after the daily break, a jump) - known at that open. Counted: is it
filled - does price come back to the close before the gap - within N days (1, 5, 20: three versions)? The gap candle
itself counts (everything in it happens after its open).
The fair comparison: 20 fake levels per gap, at a candle's OPEN of the same weekday and hour of the same year, as far
from that open (in day's ranges), on the same side - the same count.
Plan: plan/work/beliefs/open-gap-magnet.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.world import DAY

FAMILY = "belief-open-gap-magnet"
DAYS = (1, 5, 20)
MIN_GAP_ADR = 0.10               # Claude's: smaller jumps between two candles are ordinary ticks


def open_gaps(w) -> dict:
    def make():
        prev = np.r_[np.nan, w.close[:-1]]
        jump = (w.open - prev) / w.ruler
        pos = np.flatnonzero(np.isfinite(jump) & (np.abs(jump) >= MIN_GAP_ADR))
        side = np.where(jump[pos] > 0, "low", "high")               # a gap-up leaves the old close below
        dist = np.abs(jump[pos])
        c = w.controls(pos, LV.K_FAKES, w.rng("open_gap_magnet"))
        cc = np.where(c >= 0, c, 0)
        fp = np.where(c >= 0, w.open[cc] + LV.sign(side)[:, None] * dist[:, None] * w.ruler[cc], np.nan)
        kind = np.where(w.weekday[pos] == 0, "Monday open", "other")
        return {"pos": pos, "level": prev[pos], "side": side, "c": c, "fp": fp, "kind": kind}
    return w.memo("open_gaps", make)


def _rows(days: int):
    def rows(w):
        g = open_gaps(w)
        bars = days * DAY
        pos, c = g["pos"], g["c"]
        x = w.reach(pos, g["level"], g["side"], bars)
        k = c.shape[1]
        pool = w.reach(c.ravel(), g["fp"].ravel(), np.repeat(g["side"], k), bars).reshape(c.shape)
        pool = np.where(c >= 0, pool, np.nan)
        out = LV.frame(w, pos, x, LV.pools(pool), side=np.where(g["side"] == "low", "gap-up", "gap-down"),
                       when=g["kind"])
        out.index = out.index - pd.Timedelta(minutes=5)             # known at the gap candle's OPEN
        return out
    return rows


VERSIONS = tuple(
    Belief(FAMILY, f"{d}-day{'s' if d > 1 else ''}", "B4-08", "Gap-ups and gap-downs are a magnet",
           "even when the market is give a gap up, eventually it will come back to that gap up or gap down",
           "LEARNED #87; strategy.md section 2, line 84 ('gap-ups and gap-downs too'); four-boxes B4-08",
           "a candle that opens at least 0.10 of a day's range away from the close before it: does price come back "
           f"to that close within {d} day{'s' if d > 1 else ''} (the gap candle included)?",
           f"{d} day{'s' if d > 1 else ''} ({d * 288:,} five-minute candles) from the gap candle's open",
           "20 fake levels per gap - at a candle's open of the same weekday and hour of the same year, as far from it, "
           "on the same side - the same count",
           "share", _rows(d), noun="gaps", slices=("side", "when"),
           code=("machine/beliefs/open_gap_magnet.py", "machine/beliefs/levels.py"))
    for d in DAYS)
