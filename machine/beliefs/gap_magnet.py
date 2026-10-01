"""B4-08a - "Unfilled gaps are a magnet" (strategy.md section 2, line 84; LEARNED #87): his 3-candle gaps.

One event = a fair value gap on the 5-minute chart (machine/blocks/fvg.py: any size, #62) when its 3rd candle closes.
Counted: is it FILLED - does price come back through the whole gap, to its far edge - within N days? His measure names
1, 5 and 20 days: three versions, the tune years choose one. A buy gap (left by a move up) lies below price: filled by
a low at or below its bottom; a sell gap the other way round.
The fair comparison: 20 fake levels per gap, born at the same weekday and hour of the same year, as far from price as
the gap's far edge (in day's ranges), on the same side - the same count.
The gap-ups and gap-downs of his words are B4-08b (open_gap_magnet.py). Plan: plan/work/beliefs/gap-magnet.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.world import DAY
from machine.blocks import fvg as FVG

FAMILY = "belief-gap-magnet"
DAYS = (1, 5, 20)


def gaps(w) -> dict:
    def make():
        g = FVG.gaps(w.m5, "5m").reset_index(drop=True)
        pos = w.at_or_after(g["known_at"]) - 1
        side = np.where(g["side"] == "buy", "low", "high")         # where the gap lies from price
        far = g["far"].to_numpy(float)
        pp = np.clip(pos, 0, w.n - 1)
        dist = LV.sign(side) * (far - w.close[pp]) / w.ruler[pp]
        ok = (pos >= 0) & np.isfinite(dist) & (dist > 0)
        c = np.full((len(g), LV.K_FAKES), -1, np.int64)
        fp = np.full((len(g), LV.K_FAKES), np.nan)
        if ok.any():
            c[ok], fp[ok] = LV.fake_prices(w, pos[ok], side[ok], dist[ok], w.rng("gap_magnet"))
        size = (g["hi"] - g["lo"]).to_numpy(float) / w.ruler[pp]
        return {"pos": pos[ok], "far": far[ok], "side": side[ok], "c": c[ok], "fp": fp[ok],
                "size": np.where(size <= 0.05, "up to 0.05 of a day", "more than 0.05")[ok]}
    return w.memo("gaps", make)


def _rows(days: int):
    def rows(w):
        g = gaps(w)
        bars = days * DAY
        x = w.reach(g["pos"] + 1, g["far"], g["side"], bars)
        pool = LV.fake_reach(w, g["c"], g["fp"], g["side"], bars)
        return LV.frame(w, g["pos"], x, LV.pools(pool), side=g["side"], gap_size=g["size"])
    return rows


VERSIONS = tuple(
    Belief(FAMILY, f"{d}-day{'s' if d > 1 else ''}", "B4-08", "Unfilled gaps are a magnet (his 3-candle gaps)",
           "market always move towards the unfilled gaps ... eventually it will come back to that gap",
           "LEARNED #87; strategy.md section 2, line 84; four-boxes B4-08",
           "a 3-candle gap (his fair value gap) when it is made: is it filled - does price come back to its far edge - "
           f"within {d} day{'s' if d > 1 else ''}?",
           f"{d} day{'s' if d > 1 else ''} ({d * 288:,} five-minute candles) from the gap's 3rd candle",
           "20 fake levels per gap - born at the same weekday and hour of the same year, as far from price as the "
           "gap's far edge, on the same side - the same count",
           "share", _rows(d), noun="gaps", slices=("side", "gap_size"),
           code=("machine/beliefs/gap_magnet.py", "machine/beliefs/levels.py"))
    for d in DAYS)
