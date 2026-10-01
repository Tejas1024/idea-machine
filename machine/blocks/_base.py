"""Shared by every block: checking the bars, the moment a bar's result is KNOWN, and a fast "first bar that reaches
this price" search.

Two kinds of output, both stamped with `known_at` (the moment the thing became known, server clock):
  per-bar frames   index = the bar's open time (like load_bars); the row uses that bar and earlier ones only, so it is
                   known at the bar's CLOSE: known_at = open + timeframe. Checked by guards.assert_no_future.
  event tables     one row per thing found (a swing, a gap, a break ...), sorted by known_at. A row may use only the
                   bars that CLOSED by its known_at. Checked by guards.assert_no_future_events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from machine import guards
from machine.tester.models import TF_MINUTES, known_at_close

PRICE = ("open", "high", "low", "close")

__all__ = ["check_bars", "bar_close", "first_hit", "known_at_close", "TF_MINUTES"]


def check_bars(bars: pd.DataFrame, tf: str, need: tuple[str, ...] = PRICE) -> pd.DataFrame:
    """The bars a block may read: a sorted, unrepeated, time-zone-free (server clock) index, the price columns, and no
    sealed year (LEARNED #96). Returns them unchanged."""
    if tf not in TF_MINUTES:
        raise ValueError(f"timeframe {tf!r} unknown; use one of {', '.join(TF_MINUTES)}.")
    if not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("a block needs bars: a DataFrame whose index is the bars' open times (machine.data.bars).")
    if bars.index.tz is not None:
        raise ValueError("the bars' index must be MT5 server time without a time zone (like load_bars).")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("the bars must be sorted by time with no repeated bar.")
    missing = [c for c in need if c not in bars.columns]
    if missing:
        raise ValueError(f"the bars have no {missing} column(s).")
    if len(bars):
        guards.check_dates(bars.index)
    return bars


def bar_close(index: pd.DatetimeIndex, tf: str) -> pd.DatetimeIndex:
    """Each bar's close = its open time + the timeframe: the moment what it shows is KNOWN (L3's known_at_close)."""
    return pd.DatetimeIndex(index) + pd.Timedelta(minutes=TF_MINUTES[tf])


_OPS = {
    "<=": np.less_equal,
    "<": np.less,
    ">=": np.greater_equal,
    ">": np.greater,
}
_WINDOW = 64
_BATCH = 16384


def first_hit(arr: np.ndarray, starts, x, op: str, ends=None) -> np.ndarray:
    """For each query q: the first index i with starts[q] <= i < ends[q] and `arr[i] op x[q]`, or -1 if none.

    arr: prices (e.g. the lows); x: one price per query; op: "<=", "<", ">=" or ">"; ends: None = to the end of arr.
    The next 64 bars of every query are checked at once; the few queries still open are scanned in growing chunks.
    """
    cmp = _OPS[op]
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    starts = np.asarray(starts, dtype=np.int64)
    x = np.broadcast_to(np.asarray(x, dtype=float), starts.shape)
    ends = np.full(starts.shape, n, np.int64) if ends is None else np.minimum(np.asarray(ends, dtype=np.int64), n)
    out = np.full(len(starts), -1, np.int64)
    if len(starts) == 0 or n == 0:
        return out
    s = np.clip(starts, 0, n)
    padded = np.concatenate([arr, np.full(_WINDOW, np.nan)])     # NaN compares False: past the end is never a hit
    win = sliding_window_view(padded, _WINDOW)
    for b0 in range(0, len(s), _BATCH):
        sl = slice(b0, b0 + _BATCH)
        block = win[s[sl]]
        hit = cmp(block, x[sl, None]) & (np.arange(_WINDOW)[None, :] < (ends[sl] - s[sl])[:, None])
        has = hit.any(axis=1)
        out[sl][has] = s[sl][has] + hit[has].argmax(axis=1)
    rest = np.flatnonzero((out < 0) & (s + _WINDOW < ends))
    for q in rest:
        pos, size, end, xq = int(s[q]) + _WINDOW, 256, int(ends[q]), x[q]
        while pos < end:
            seg = arr[pos:min(pos + size, end)]
            h = cmp(seg, xq)
            if h.any():
                out[q] = pos + int(h.argmax())
                break
            pos += size
            size *= 4
    return out


def empty(columns: dict[str, str]) -> pd.DataFrame:
    """An event table with no rows but the right columns and types."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns.items()})
