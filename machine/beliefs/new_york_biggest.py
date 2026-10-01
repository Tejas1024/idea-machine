"""B4-16 - "NY (5:30-8:30 PM IST) = biggest, news-driven moves. News = 7 PM IST" (strategy.md section 5, line 99).

The news half ("news = 7 PM IST") was measured by L12b.2 from the calendar: 55-61 of 100 of his news come between
6 and 8 PM India time (plan/work/L12b2-notes.md section 3c). This is the session half.
One event = one hour candle of his New York window (his windows, India time: machine/rules.py
what_a_session_high_or_low_is - the candles that open 17:30, 18:30, 19:30 IST). Counted: how far price travelled in it
(high - low) as a share of a day's range.
The fair comparison: the hour candles of his London window (12:30-15:30 IST) in the same broker week. Looked at: the
New York hours with and without a release of his news inside (the "news-driven" part).
Plan: plan/work/beliefs/new-york-biggest.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.blocks import sessions as SES


def _window_hours(bars: pd.DataFrame, name: str) -> np.ndarray:
    """The hour candles that lie inside his window `name` (their India-time open and close)."""
    start, end = next((a, b) for n, a, b in SES.windows() if n == name)
    ist = pd.DatetimeIndex(bars["utc"]) + pd.Timedelta(hours=5, minutes=30)
    t_open = ist.time
    t_close = (ist + pd.Timedelta(hours=1)).time
    return np.array([(o >= start) and (c <= end) and (c > o) for o, c in zip(t_open, t_close)])


FAMILY = "belief-new-york-biggest"


def rows(w):
    h1 = w.market.bars("1h")
    ruler = w.market.ruler("1h", w.fp)
    rng_adr = (h1["high"] - h1["low"]).to_numpy(float) / ruler
    week = (h1.index.normalize() - pd.to_timedelta(h1.index.dayofweek, unit="D")).as_unit("s").asi8 // 86_400
    ny, ld = _window_hours(h1, "New York"), _window_hours(h1, "London")
    ok = np.isfinite(rng_adr)
    london = {}
    for wk in np.unique(week[ld & ok]):
        london[wk] = rng_adr[ld & ok & (week == wk)]
    pos = np.flatnonzero(ny & ok)
    close = h1.index[pos] + pd.Timedelta(hours=1)
    news = LV.news_between(w, h1.index[pos].as_unit("ns").asi8, close.as_unit("ns").asi8)
    out = pd.DataFrame({"x": rng_adr[pos]}, index=close.rename("known_at"))
    out["pool"] = pd.Series([london.get(wk) for wk in week[pos]], index=out.index, dtype=object)
    out["news"] = np.where(news, "his news in the hour", "no news in the hour")
    out["hour"] = [f"{t:%H:%M} IST" for t in (pd.DatetimeIndex(h1["utc"].to_numpy()[pos])
                                              + pd.Timedelta(hours=5, minutes=30))]
    return out[np.isin(out.index.year, w.years)]


VERSIONS = (
    Belief(FAMILY, "new-york-vs-london", "B4-16", "New York has the biggest moves",
           "NY (5:30-8:30 PM IST) = biggest, news-driven moves. News = 7 PM IST.",
           "strategy.md section 5, line 99; four-boxes B4-16 (the news half: L12b.2)",
           "one hour candle of his New York window (opens 17:30, 18:30 or 19:30 India time): how far price travelled in "
           "it, high to low, as a share of a day's range",
           "one hour",
           "the hour candles of his London window (12:30-15:30 India time) of the same broker week",
           "range", rows, noun="New York hours", slices=("news", "hour"),
           code=("machine/beliefs/new_york_biggest.py", "machine/beliefs/levels.py")),
)
