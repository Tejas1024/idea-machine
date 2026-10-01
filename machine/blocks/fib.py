"""Fibonacci retracement levels (L6.1).

His words (LEARNED #135, 25 Sep 2026): "in one hour time frame whenever break of structure happen like uh, one up one
side movement and it should break like it should higher high sort of thing then it will come back to exact golden
ratio that is uh, 0.68 it will definitely come to that particular uh, point I, if it comes there it will definitely
give 1 is to 1 the next that is 0.5 is the target even for bullish or bearish".

Fibonacci is NEW to his files: nothing in strategy.md or strategy-code.md defines it, and three points are open
(strategy-understanding section 66bz): F1 0.68 or 0.618 · F2 where the stop is · F3 WHICH break counts. So this
block draws the levels for the plain reading only, and says so:
  "a higher high"  = a swing high (machine.blocks.swings, his 2-candle swing) above the swing high before it;
  the move         = from the lowest low BETWEEN those two swing highs (bottom) to the new high (top);
  a level          = top - ratio x (top - bottom)  (a BUY waits for price to come back down to it);
  the mirror for a lower low (a SELL): level = bottom + ratio x (top - bottom).
Known when the new swing is known (its 2nd candle after has closed). Ratios: settings "fib_ratios" (the idea's own
numbers - a version may change them). Other readings of F3 (his documented break, a strong move after liquidity)
come from the other blocks once T3 / the talker say which one he means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import settings
from machine.blocks._base import check_bars
from machine.blocks.swings import swings


def level_name(ratio: float) -> str:
    return f"level_{ratio:g}"


def fib_levels(bars: pd.DataFrame, tf: str, ratios=None) -> pd.DataFrame:
    """One row per higher high (side "buy") and lower low (side "sell"): known_at, side, from_bar (the bar of the move's
    start), to_bar (the new swing), top, bottom, and one column per ratio (level_0.618 ...)."""
    check_bars(bars, tf, ("high", "low"))
    ratios = tuple(settings.get("fib_ratios") if ratios is None else ratios)
    cols = ["known_at", "side", "from_bar", "to_bar", "top", "bottom", *[level_name(r) for r in ratios]]
    sw = swings(bars, tf)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    rows = []
    for side, which in (("buy", "high"), ("sell", "low")):
        s = sw[sw["side"] == which].sort_values("bar", kind="stable")
        pos = bars.index.get_indexer(pd.DatetimeIndex(s["bar"]))
        price = s["price"].to_numpy()
        known = s["known_at"].to_numpy()
        for q in range(1, len(pos)):
            a, b = pos[q - 1], pos[q]
            if b - a < 2:
                continue
            if side == "buy":
                if not price[q] > price[q - 1]:
                    continue                        # not a higher high
                m = a + 1 + int(np.argmin(lo[a + 1:b]))
                top, bottom = price[q], lo[m]
                lv = [top - r * (top - bottom) for r in ratios]
            else:
                if not price[q] < price[q - 1]:
                    continue                        # not a lower low
                m = a + 1 + int(np.argmax(h[a + 1:b]))
                top, bottom = h[m], price[q]
                lv = [bottom + r * (top - bottom) for r in ratios]
            if top <= bottom:
                continue
            rows.append([known[q], side, bars.index[m], bars.index[b], top, bottom, *lv])
    out = pd.DataFrame(rows, columns=cols)
    if not len(out):
        out = out.astype({"known_at": "datetime64[ns]", "from_bar": "datetime64[ns]", "to_bar": "datetime64[ns]"})
    return out.sort_values(["known_at", "side"], kind="stable", ignore_index=True)
