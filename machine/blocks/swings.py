"""Swing highs and lows (L6.1).

His definition (strategy.md glossary; strategy-code.md section 3, B3 confirmed): "Swing high/low — local high/low with
2 lower/higher candles each side." A swing high is a candle whose high is STRICTLY higher than the highs of the
2 candles before it AND the 2 after it; a swing low the mirror. The 2 comes from machine/rules.py PARTS
("swing_lookback").

KNOWN only when the 2nd candle after it has closed: known_at = that candle's close. Nothing earlier is used.

What a swing may be used FOR is his too (machine/rules.py): a recent 5-minute swing is a target only (#91); a 1-hour
swing is a target, not an entry (#93); only day and week swings start a setup. The roles are applied in L6.2
(swing_roles). This block only finds them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import rules
from machine.blocks._base import bar_close, check_bars, empty

COLUMNS = {"known_at": "datetime64[ns]", "bar": "datetime64[ns]", "side": "object", "price": "float64"}


def swings(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Every swing high and low: known_at, bar (the swing candle's open time), side ("high" / "low"), price."""
    check_bars(bars, tf, ("high", "low"))
    piv = int(rules.get("swing_lookback"))
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    n = len(h)
    if n < 2 * piv + 1:
        return empty(COLUMNS)
    is_hi = np.zeros(n, bool)
    is_lo = np.zeros(n, bool)
    mid = slice(piv, n - piv)
    hi_ok = np.ones(n - 2 * piv, bool)
    lo_ok = np.ones(n - 2 * piv, bool)
    for d in range(1, piv + 1):                       # strictly beyond each of the piv candles on both sides
        hi_ok &= (h[mid] > h[piv - d:n - piv - d]) & (h[mid] > h[piv + d:n - piv + d])
        lo_ok &= (lo[mid] < lo[piv - d:n - piv - d]) & (lo[mid] < lo[piv + d:n - piv + d])
    is_hi[mid] = hi_ok
    is_lo[mid] = lo_ok
    closes = bar_close(bars.index, tf)
    parts = []
    for side, mask, price in (("high", is_hi, h), ("low", is_lo, lo)):
        k = np.flatnonzero(mask)
        parts.append(pd.DataFrame({"known_at": closes[k + piv], "bar": bars.index[k], "side": side,
                                   "price": price[k]}))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "side"], kind="stable", ignore_index=True)
