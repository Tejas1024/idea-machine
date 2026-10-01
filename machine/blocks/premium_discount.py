"""Premium and discount: where price sits against his ADR zones today (L6b.3).

His words:
  strategy.md section 7 (T1): "ADR (Avg Daily Range) zones = daily OPEN ± half of 5/10-day range -> red (resistance) +
      green (support) zones. ... At discount (near the green zone) a sell only if the hour AND the day are bearish; at
      premium a buy only if they are bullish. A break below the lower ADR zone can mean a further fall (bad news,
      strong selling) (#76)."
  LEARNED #76 (Q74-C2): "if higher time frame, hourly and daily is also suggesting to bearish, then I would take it. Or
      else I would not take it, I would avoid this" · below the zone: "if it breaks the lower range of ADR, lower zone,
      it will go further low ... it's not like we shouldn't sell below that" - so PAST the zone is the same place, not a
      harder one.
  LEARNED #75: the picture he could not read (no timeframe, the gap invisible, lines in the zones' colours) - pictures
      of these zones must draw them apart from the trade's lines (machine/charts).

The zones are machine/blocks/adr.py's (his Pine study, broker days): red = [up_lo, up_hi] above the day's open, green =
[dn_lo, dn_hi] below it. Where a price sits - two measures, a block parameter (a version, never a guess):
  near_the_zone     (default) his words "near the green zone": PREMIUM = at, inside or past the red zone, or at most
                    settings "near_zone_adr" of a day's range short of it (up_lo - that). DISCOUNT = the mirror at the
                    green zone. Else MIDDLE.
  share_of_the_day  the old engine's: the share of the way from the bottom of the green zone (0) to the top of the red
                    one (1); PREMIUM from settings "premium_share" (0.85), DISCOUNT up to "discount_share" (0.15).
Both numbers are Claude's (his #5 and #76 give none).

  zones(bars, tf, measure)    on every bar: the day's open, the two zones, a day's range, premium_from (at or above =
                              premium), discount_to (at or below = discount); known_at = the bar's close (the day's
                              open is known from its first bar - adr.py).
  place(price, row)           "premium" / "discount" / "middle" for one price on one bar's zones.
  share(price, row)           0 at the bottom of the green zone, 1 at the top of the red zone (can be below 0 / above 1).
  arrivals(bars, tf, side, measure)   the bars where price COMES to a zone: side "top" = the first bar of a run whose
                              high reaches premium (a new day starts a new run); "bottom" = the mirror with the lows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import settings
from machine.blocks import adr as ADR
from machine.blocks._base import bar_close, check_bars

MEASURES = ("near_the_zone", "share_of_the_day")
PREMIUM, DISCOUNT, MIDDLE = "premium", "discount", "middle"


def zones(bars: pd.DataFrame, tf: str, measure: str = "near_the_zone") -> pd.DataFrame:
    """His ADR zones on every bar, and the lines premium starts at and discount ends at (see the module text)."""
    if measure not in MEASURES:
        raise ValueError(f"measure {measure!r}: one of {', '.join(MEASURES)}")
    a = ADR.adr_bars(bars, tf)
    out = a[["day_open", "up_lo", "up_hi", "dn_hi", "dn_lo", "ruler"]].copy()
    if measure == "near_the_zone":
        pad = float(settings.get("near_zone_adr")) * out["ruler"]
        out["premium_from"] = out["up_lo"] - pad
        out["discount_to"] = out["dn_hi"] + pad
    else:
        width = out["up_hi"] - out["dn_lo"]
        out["premium_from"] = out["dn_lo"] + float(settings.get("premium_share")) * width
        out["discount_to"] = out["dn_lo"] + float(settings.get("discount_share")) * width
    out["day"] = bars.index.normalize()
    out["known_at"] = bar_close(bars.index, tf)
    return out


def place(price: float, row) -> str:
    """premium / discount / middle for `price` on one bar's zones (a row of zones()). No zones yet (the first 10 days
    of the bars) = middle. Where the two lines cross (a tiny day), the side of the day's open decides."""
    pf, dt = float(row["premium_from"]), float(row["discount_to"])
    if not (np.isfinite(pf) and np.isfinite(dt)):
        return MIDDLE
    hi, lo = price >= pf, price <= dt
    if hi and lo:
        return PREMIUM if price >= float(row["day_open"]) else DISCOUNT
    return PREMIUM if hi else DISCOUNT if lo else MIDDLE


def share(price: float, row) -> float:
    """0 = the bottom of the green zone, 1 = the top of the red zone."""
    lo, hi = float(row["dn_lo"]), float(row["up_hi"])
    return (price - lo) / (hi - lo) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else np.nan


def arrivals(bars: pd.DataFrame, tf: str, side: str, measure: str = "near_the_zone") -> pd.DataFrame:
    """The bars where price comes to the top (red) or bottom (green) zone: bar, price (the bar's high for the top,
    its low for the bottom), known_at (its close). A run of bars at the zone counts once; a new day starts a new run."""
    if side not in ("top", "bottom"):
        raise ValueError(f"side must be 'top' or 'bottom', not {side!r}")
    check_bars(bars, tf)
    z = zones(bars, tf, measure)
    if side == "top":
        px = bars["high"].to_numpy(dtype=float)
        at = px >= z["premium_from"].to_numpy()
    else:
        px = bars["low"].to_numpy(dtype=float)
        at = px <= z["discount_to"].to_numpy()
    day = z["day"].to_numpy()
    prev_at = np.r_[False, at[:-1]]
    same_day = np.r_[False, day[1:] == day[:-1]]
    k = np.flatnonzero(at & ~(prev_at & same_day))
    return pd.DataFrame({"known_at": z["known_at"].to_numpy()[k], "bar": bars.index[k], "price": px[k]})
