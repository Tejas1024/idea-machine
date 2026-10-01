"""His hammer (L6b.3) - machine/rules.py "what_a_hammer_is" (LEARNED #115; the plan and the old code said #117).

His words (#115, his R05): "there is a very uh, hammer type of candle which is a signal that it took the liquidity and
uh, buy bought suddenly and the fair value gap is formed that is uh, you know one powerful fair value gap … So the dip,
huge dip, and the fair value gap will give the better trades than normal fair value gaps." strategy.md section 2:
"Hammer + gap: a long wick that takes the liquidity and snaps back, followed by a gap, is a stronger setup - it can be
taken even when the higher timeframe does not agree (#115)" (settings "hammer_beats_the_higher_timeframe").

The shape (rules.py; the numbers are the old engine's, his words give none):
  - the wick on the side the liquidity was taken is at least wick_min (55%) of the candle - a BUY hammer has the long
    wick below (it took the lows and snapped back up), a SELL hammer above;
  - the other wick at most other_wick_max (40%), the body at most body_max (40%);
  - the wick itself at least dip_adr_min (0.12) of a day's range ("huge dip" - measured against the day, so it means the
    same in 2016 and 2023). A day's range = settings "adr_ruler_days" (his ADR, broker days).
Known when the candle closes.

  hammers(bars, tf)     on every bar: buy / sell (is it a hammer that way), the long wick's share of the candle and its
                        size as a share of a day's range, known_at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import rules
from machine.blocks import adr as ADR
from machine.blocks._base import bar_close, check_bars


def hammers(bars: pd.DataFrame, tf: str, ruler=None) -> pd.DataFrame:
    """His hammer on every bar (see the module text). ruler: a day's range on every bar (default his ADR)."""
    check_bars(bars, tf)
    R = rules.get("what_a_hammer_is")
    o, h, lo, c = (bars[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    r = np.asarray(ADR.adr_bars(bars, tf)["ruler"] if ruler is None else ruler, dtype=float)
    rng = h - lo
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    lower, upper = body_lo - lo, h - body_hi
    ok = (rng > 0) & np.isfinite(r) & (r > 0)
    safe = np.where(rng > 0, rng, 1.0)
    body_ok = (rng - lower - upper) / safe <= R["body_max"]
    out = {}
    for side, wick, other in (("buy", lower, upper), ("sell", upper, lower)):
        out[side] = (ok & (wick > 0) & (wick / safe >= R["wick_min"]) & (other / safe <= R["other_wick_max"])
                     & body_ok & (wick >= R["dip_adr_min"] * np.where(ok, r, np.inf)))
        out[f"{side}_wick_share"] = np.where(rng > 0, wick / safe, np.nan)
        out[f"{side}_dip"] = np.where(ok, wick / np.where(ok, r, 1.0), np.nan)
    return pd.DataFrame({"known_at": bar_close(bars.index, tf), **out}, index=bars.index)
