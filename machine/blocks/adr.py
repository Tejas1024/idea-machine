"""ADR - his one indicator (L6.1; strategy-code.md section 16; strategy.md section 7).

His words (strategy.md section 7): "ADR (Avg Daily Range) zones = daily OPEN ± half of 5/10-day range → red
(resistance) + green (support) zones." The formula of his Pine v2 study (strategy-code.md section 16):
    day_range = daily high - daily low, for each of the last 1..10 FINISHED days
    adr_n     = mean of the last n finished days' ranges, n = 5 and 10     (machine/rules.py PARTS "adr_days")
    up band   = [open + min(adr5, adr10)/2 , open + max(adr5, adr10)/2]    (his red, resistance)
    down band = [open - max(adr5, adr10)/2 , open - min(adr5, adr10)/2]    (his green, support)
Days are BROKER days (machine/rules.py "broker_days_not_utc_days"): a day is one server calendar day, as on his MT5.
Today's own range is never used for today (it is not finished).

`ruler` is "a day's range" - the measure behind every number written as a share of a day (settings "adr_ruler_days":
his 10-day ADR; Claude's pick - the old engine's ruler used UTC days, plan/work/rules-missing.md).

  adr_days()  one row per broker day: day, day_open, adr5, adr10, the four band edges, ruler, known_at (the close of
              the day's first bar - when its open price is on the chart).
  adr_bars()  the same values on every bar (index = the bar's open), known_at = the bar's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks._base import bar_close, check_bars

VALUES = ["day_open", "adr5", "adr10", "up_lo", "up_hi", "dn_hi", "dn_lo", "ruler"]


def _days(bars: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, np.ndarray]:
    check_bars(bars, tf)
    if rules.get("broker_days_not_utc_days") != "server":
        raise RuntimeError("rules.py broker_days_not_utc_days must be 'server'")
    key = bars.index.normalize()                              # the broker day = the server calendar day
    g = bars.groupby(key, sort=True)
    d = g.agg(day_open=("open", "first"), high=("high", "max"), low=("low", "min"))
    d["first_bar"] = g.apply(lambda x: x.index[0]).to_numpy()
    rng = d["high"] - d["low"]
    short, long_ = rules.get("adr_days")
    d["adr5"] = rng.rolling(short).mean().shift(1)            # finished days only
    d["adr10"] = rng.rolling(long_).mean().shift(1)
    lo_adr = np.minimum(d["adr5"], d["adr10"]) / 2
    hi_adr = np.maximum(d["adr5"], d["adr10"]) / 2
    d["up_lo"], d["up_hi"] = d["day_open"] + lo_adr, d["day_open"] + hi_adr
    d["dn_hi"], d["dn_lo"] = d["day_open"] - lo_adr, d["day_open"] - hi_adr
    n = int(settings.get("adr_ruler_days"))
    d["ruler"] = d["adr10"] if n == long_ else (d["adr5"] if n == short else rng.rolling(n).mean().shift(1))
    return d, key


def adr_days(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """One row per broker day (see the module text)."""
    d, _ = _days(bars, tf)
    out = d[VALUES].copy()
    out.insert(0, "known_at", bar_close(pd.DatetimeIndex(d["first_bar"]), tf))
    out.index.name = "day"
    return out.reset_index()


def adr_bars(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """His ADR values on every bar (index = the bar's open time, server clock); known_at = the bar's close."""
    d, key = _days(bars, tf)
    out = d[VALUES].reindex(key)
    out.index = bars.index
    out["known_at"] = bar_close(bars.index, tf)
    return out
