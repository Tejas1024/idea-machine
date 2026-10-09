"""The chart situation at a minute, from live bars (L17a steps 2 and 3) - every part of 'the last 10 times'
(machine/news/similar.py SITUATION), measured exactly as machine/news/reaction.py measured every 2016-2023 release
(tests/test_live.py checks them against that table):
  at()           from the 1-minute bars only: his session window, his 10-day ADR (a day's range), how much of it the day
                 had already used, the move in the 2 hours before
  chart_parts()  his bias on the hour, day and week charts and his two nearest levels (with their names in his words),
                 drawn by HIS OWN blocks on live bars - inside guards.live_trial (T14 use 1, alerts only). The history
                 each needs was measured on 2018-2020 (machine/live/prices.Charts)
  whole()        both; without his charts (MT5 trouble), at() alone - the bias and levels are then LEFT OUT, never
                 guessed (the lookup leaves out a part the question does not carry)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import guards
from machine.blocks import sessions as SES
from machine.data import clock

WHO = "machine.live (L17a, his bias and levels for an alert)"
WHY = "his bias and nearest levels drawn by his own blocks on live bars, for the live alerts only (T14 use 1)"
LEVEL_KEYS = ("up_level_kind", "up_level_dist", "up_level_adr", "dn_level_kind", "dn_level_dist", "dn_level_adr")

MIN = pd.Timedelta(minutes=1)
OPEN_GAP = 5 * MIN                  # reaction.OPEN_GAP_MIN
BEFORE = 120 * MIN                  # reaction.py's "before_120": the move in the 2 hours before ...
BEFORE_SLACK = 30 * MIN             # ... read from a bar at most 30 minutes older than that


def day_adr(m1: pd.DataFrame, day: pd.Timestamp) -> float:
    """His 10-day ADR for the server day `day`: the mean high - low of the 10 finished broker days before it (nan when
    fewer than 10) - machine/blocks/adr.py's ruler."""
    d = m1.groupby(pd.DatetimeIndex(m1.index).normalize()).agg(high=("high", "max"), low=("low", "min"))
    d = d[d.index < pd.Timestamp(day).normalize()].tail(10)
    return float((d["high"] - d["low"]).mean()) if len(d) == 10 else float("nan")


def session_of(t: pd.Timestamp) -> str:
    """His session window holding server minute t (India time), or "outside"."""
    ist = clock.server_to_utc(pd.DatetimeIndex([t])).tz_convert(clock.IST_ZONE)[0]
    m = ist.hour * 60 + ist.minute
    return next((name for name, a, b in SES.windows() if a.hour * 60 + a.minute <= m < b.hour * 60 + b.minute),
                "outside")


def at(m1: pd.DataFrame, t: pd.Timestamp) -> dict:
    """The situation at release minute t (server clock) from bars that opened BEFORE t: p0, session, adr, adr_used,
    before_120, before_120_adr. Empty when the market was shut (no bar in the 5 minutes before t)."""
    t = pd.Timestamp(t)
    idx = pd.DatetimeIndex(m1.index)
    i0 = int(idx.searchsorted(t, "left"))
    if i0 == 0 or t - idx[i0 - 1] > OPEN_GAP:
        return {}
    close = m1["close"].to_numpy(float)
    p0 = float(close[i0 - 1])
    out = {"p0": p0, "session": session_of(t)}
    adr = day_adr(m1.iloc[:i0], t)
    out["adr"] = adr
    d0 = int(idx.searchsorted(t.normalize(), "left"))
    if d0 < i0 and np.isfinite(adr) and adr > 0:
        out["adr_used"] = float((m1["high"].iloc[d0:i0].max() - m1["low"].iloc[d0:i0].min()) / adr)
    j2 = int(idx.searchsorted(t - BEFORE, "left")) - 1
    if j2 >= 0 and (t - BEFORE) - idx[j2] <= BEFORE_SLACK:
        out["before_120"] = p0 - float(close[j2])
        if np.isfinite(adr) and adr > 0:
            out["before_120_adr"] = out["before_120"] / adr
    return out


def chart_parts(c, t: pd.Timestamp) -> dict:
    """His bias (bias_h, bias_d, bias_w: +1 up, -1 down, 0 neutral) and his two nearest levels at minute t (server
    clock): kind (None = no level of his alive on that side), distance in dollars and in days' ranges, and
    `level_names` in his words - by his own blocks on the charts `c` (machine/live/prices.Charts), as reaction.py did.
    The day and week bias come from a market of the day and week charts from 2016 (so the all-time high starts from
    gold's 2011 record, as in every 2016-2023 table); the hour bias from the hour chart alone (it has no all-time-high
    rule); the levels from the 1-minute bars' own market."""
    from machine import settings
    from machine.ideas.compile import Market
    from machine.news import reaction, similar
    fp = settings.fingerprint()
    t = pd.Timestamp(t)
    with guards.live_trial(WHO, WHY):
        dw = Market({"1d": c.d1, "1w": c.w1})
        out = {"bias_h": float(Market({"1h": c.h1}).bias_at("1h", fp, t)),
               "bias_d": float(dw.bias_at("1d", fp, t)), "bias_w": float(dw.bias_at("1w", fp, t))}
        market = Market.from_m1(c.m1)                     # every bar closed by now: as reaction.py, the 5-minute
        ctx = reaction.Measurer(c.m1, market).measure([t], context_only=True).iloc[0]   # candle holding t is there
        if pd.isna(ctx["p0"]):
            return out                                     # the market was shut: no levels at this minute
        for k in LEVEL_KEYS:
            v = ctx[k]
            out[k] = (v if isinstance(v, str) else None) if k.endswith("kind") else float(v)
        row = pd.Series({**{k: out[k] for k in LEVEL_KEYS}, "time": t, "p0": float(ctx["p0"])})
        out["level_names"] = similar.level_names(row, similar.Levels(market))
    return out


def whole(p, t: pd.Timestamp) -> dict:
    """at() and, when his charts can be had (p.charts()), chart_parts(). Any failure there (MT5, the lock) only leaves
    his bias and levels out - the alert still goes - and `charts_why` says why."""
    s = at(p.m1, t)
    if not s:
        return s
    try:
        c = p.charts()
        if c is None:
            return {**s, "charts_why": "no charts"}
        return {**s, **chart_parts(c, t)}
    except Exception as e:
        return {**s, "charts_why": f"{type(e).__name__}: {str(e)[:200]}"}


def for_lookup(s: dict) -> dict:
    """The parts the lookup matches on (similar.SITUATION) that were measured: his bias and levels only when his
    charts were read (a level kind of None there = no level alive - a real value)."""
    keys = ("session", "adr", "adr_used", "before_120_adr", "bias_d", "bias_w") + \
        (("up_level_kind", "up_level_adr", "dn_level_kind", "dn_level_adr") if "up_level_kind" in s else ())
    return {k: s[k] for k in keys if k in s}
