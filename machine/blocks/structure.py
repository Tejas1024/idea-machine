"""His order block and the break of structure (L6.1) - machine/rules.py "what_counts_as_the_break".

His words (LEARNED #56, #57): "the recent order block, that I would consider the low of the recent order block. That
what I consider, it should be broken to consider as a makeup structure, break of structure." · "It is the red candle
in a uptrend, so not the green candle." His arrows on 9 charts (#58, #59, #60) fixed the rest: a WICK is enough.

  SELL side: the most recent RED candle in the rise, after which the move up - the candles until the next red one -
             goes ABOVE that red candle's high (a wick is enough). A wick BELOW its low is the break of structure.
  BUY side:  the mirror - the most recent GREEN candle in the drop, after which the drop goes below its low; a wick
             ABOVE its high is the break.

A block is KNOWN when the candle that went past it closes (the old engine made it usable from the next candle - the
same thing). Blocks of one side confirm in candle order, so "the most recent block" at any candle is simply the last
one confirmed before it.

  order_blocks()       every block: side, block candle, its high / low, the candle that confirmed it, known_at
  breaks()             each block's first break while it is the most recent block of its side
  first_break_after()  the break a setup uses: the first candle after the sweep that wicks through the block that is
                       most recent AT THAT CANDLE, a block no older than settings "order_block_max_age_bars" before
                       the sweep (Claude's 40, from the old engine) - exactly the old engine's SWEPT step.
  BreakSearch          the same search for many setups on the same bars (the blocks found once) - the idea
                       compiler (L7a) uses it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import settings
from machine.blocks._base import bar_close, check_bars, empty, first_hit

BLOCK_COLUMNS = {"known_at": "datetime64[ns]", "side": "object", "block_bar": "datetime64[ns]",
                 "block_high": "float64", "block_low": "float64", "level": "float64", "confirmed_bar": "datetime64[ns]"}
BREAK_COLUMNS = {"known_at": "datetime64[ns]", "side": "object", "bar": "datetime64[ns]", "level": "float64",
                 "block_bar": "datetime64[ns]"}


def _block_arrays(o, h, lo, c):
    """For each side: (block candle index, confirming candle index), in confirmation order."""
    n = len(c)
    red, green = c < o, c > o
    out = {}
    for side, own, cmp_arr, op in (("sell", red, h, ">"), ("buy", green, lo, "<")):
        k = np.flatnonzero(own)
        # the run after candle k lasts until the next candle of k's own colour (dojis do not end it)
        nxt = np.searchsorted(k, k, side="right")
        run_end = np.where(nxt < len(k), k[np.minimum(nxt, len(k) - 1)], n)
        j = first_hit(cmp_arr, k + 1, cmp_arr[k], op, ends=run_end)
        ok = j >= 0
        out[side] = (k[ok], j[ok])
    return out


def order_blocks(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Every order block. level = the price whose wick-through is the break (sell: the block's low; buy: its high)."""
    check_bars(bars, tf)
    o, h, lo, c = (bars[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    if len(c) < 2:
        return empty(BLOCK_COLUMNS)
    closes = bar_close(bars.index, tf)
    parts = []
    for side, (k, j) in _block_arrays(o, h, lo, c).items():
        parts.append(pd.DataFrame({"known_at": closes[j], "side": side, "block_bar": bars.index[k],
                                   "block_high": h[k], "block_low": lo[k],
                                   "level": lo[k] if side == "sell" else h[k], "confirmed_bar": bars.index[j]}))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "side"], kind="stable", ignore_index=True)


def _current(n, k, j):
    """cur[i] = the block candle index that is the most recent block AT candle i (confirmed before i), or -1."""
    mark = np.full(n + 1, -1, np.int64)
    mark[j + 1] = k                                   # known from the candle after the confirming one
    pos = np.where(mark >= 0, np.arange(n + 1), 0)
    np.maximum.accumulate(pos, out=pos)
    return mark[pos][:n]


def breaks(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Each block's first break while it is still the most recent block of its side (a wick is enough)."""
    check_bars(bars, tf)
    o, h, lo, c = (bars[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    n = len(c)
    if n < 3:
        return empty(BREAK_COLUMNS)
    closes = bar_close(bars.index, tf)
    parts = []
    for side, (k, j) in _block_arrays(o, h, lo, c).items():
        if not len(k):
            continue
        start = j + 1                                   # the block is the most recent from here ...
        end = np.append(j[1:] + 1, n)                   # ... until the next block of its side is known
        if side == "sell":
            i = first_hit(lo, start, lo[k], "<", ends=end)
        else:
            i = first_hit(h, start, h[k], ">", ends=end)
        ok = i >= 0
        k, i = k[ok], i[ok]
        parts.append(pd.DataFrame({"known_at": closes[i], "side": side, "bar": bars.index[i],
                                   "level": lo[k] if side == "sell" else h[k], "block_bar": bars.index[k]}))
    if not parts:
        return empty(BREAK_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "side"], kind="stable", ignore_index=True)


def first_break_after(bars: pd.DataFrame, tf: str, side: str, after, max_age_bars: int | None = None) -> dict | None:
    """The break a setup uses (the old engine's SWEPT step): the first candle AFTER `after` (the sweep candle's open
    time) whose wick goes through the most recent block of `side` known at that candle, that block being no older
    than max_age_bars candles before `after`. Returns {known_at, bar, level, block_bar} or None.

    side "sell" = a high was swept, the break is down through a red block's low; "buy" = the mirror.
    """
    if side not in ("sell", "buy"):
        raise ValueError(f"side must be 'sell' or 'buy', not {side!r}")
    check_bars(bars, tf)
    s = int(bars.index.get_indexer([pd.Timestamp(after)])[0])
    if s < 0:
        raise KeyError(f"{after} is not a bar of these bars")
    return BreakSearch(bars, tf, side, max_age_bars).after(s)


class BreakSearch:
    """first_break_after() for many setups on the same bars (the idea compiler, L7a): the order blocks are found once,
    and a search may stop at a last candle (the step's maximum wait)."""

    def __init__(self, bars: pd.DataFrame, tf: str, side: str, max_age_bars: int | None = None):
        if side not in ("sell", "buy"):
            raise ValueError(f"side must be 'sell' or 'buy', not {side!r}")
        check_bars(bars, tf)
        self.index, self.tf, self.side = bars.index, tf, side
        self.age = settings.get("order_block_max_age_bars") if max_age_bars is None else int(max_age_bars)
        o, h, lo, c = (bars[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
        self.h, self.lo, self.n = h, lo, len(c)
        k, j = _block_arrays(o, h, lo, c)[side]
        self.cur = _current(self.n, k, j)

    def after(self, s: int, until: int | None = None, start: int | None = None) -> dict | None:
        """The break for a setup whose sweep candle is at position s: the first candle s+1 ... until (positions,
        until included; None = to the end) that wicks through the most recent block known at it, that block no older
        than the age limit before s. start: look only from this candle on (the grab kept going - his fixed rule
        a_grab_that_keeps_going_is_still_a_grab: wait for the break again after the new extreme); the age limit still
        counts from s."""
        end = self.n if until is None else min(self.n, int(until) + 1)
        first = s + 1 if start is None else max(s + 1, int(start))
        if first >= end:
            return None
        i = np.arange(first, end)
        kk = self.cur[first:end]
        ok = (kk >= 0) & (kk >= s - self.age)
        safe = np.where(kk >= 0, kk, 0)
        hit = ok & ((self.lo[i] < self.lo[safe]) if self.side == "sell" else (self.h[i] > self.h[safe]))
        if not hit.any():
            return None
        q = int(np.argmax(hit))
        b, blk = int(i[q]), int(kk[q])
        return {"known_at": bar_close(self.index[[b]], self.tf)[0], "bar": self.index[b],
                "level": float(self.lo[blk] if self.side == "sell" else self.h[blk]), "block_bar": self.index[blk]}
