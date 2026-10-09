"""The big-move alert (L17a step 2): gold moved a lot in a short time with no news of his near it. His strategy.md
section 12: "Breaking/unscheduled news (war, etc.) -> need a real-time gold-news feed/alert ... -> flag 'analysis may
change'." The move itself is the feed: no news site has to be read.

THE RULE - Claude's alert setting, not his "big move" of the triangle rule (four-boxes Box 2). Fixed on 5 Oct 2026
before anything was counted; only the two sizes were then set, on 2016-2020 (the tune years - T14: never on live
prices), to about one alert a week:
  move     the highest high - the lowest low of the last LIVE_MOVE_MINUTES (15) closed 1-minute bars (at least
           LIVE_MOVE_MIN_BARS = 12 of them there)
  usual    the median of the same move at the same New York clock minute on the last LIVE_MOVE_USUAL_DAYS (20) New York
           days that had bars (at least 10 of them with a move at that minute) - a busy hour has a bigger usual
  big      move >= LIVE_MOVE_TIMES_USUAL (6) x usual  AND  move >= LIVE_MOVE_SHARE_OF_DAY (0.3) x his 10-day ADR (broker
           days, finished days only - machine/blocks/adr.py's ruler; tests/test_live.py checks they agree)
  no news  no Forex Factory red or orange US release from LIVE_NEWS_CALM_MINUTES before the bar (red 120, orange 40 -
           L12b.2: gold is calm again after about 2 hours / 40 minutes) to LIVE_NEWS_EARLY_MINUTES (5) after it
  quiet    after one alert, none for LIVE_MOVE_QUIET_MINUTES (240): the same move goes on
On 2016-2020 it comes 0.95 times a week (249 in 5 years; no alert in half the weeks; 3 or 4 on 11 days). With no
date given, it found the gold flash crash of 26 Jun 2017, Iran's missiles on 8 Jan 2020, the vaccine news of 9 Nov
2020, the US election night of 2016, the Brexit result and the pound's flash crash of 7 Oct 2016.
    python -m machine.live moves 2016 2020      counts it again on the 1-minute bars of those years
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from machine import config
from machine.data import clock
from machine.host import alert as HA

NY = "America/New_York"
MIN_NS = 60 * 10 ** 9
RATE_WORDS = "On 2016–2020, a move this big with no news came about once a week."


def adr(m1: pd.DataFrame) -> pd.Series:
    """His 10-day ADR on every bar: the mean high - low of the last 10 FINISHED broker days (server calendar days) -
    machine/blocks/adr.py's ruler, made here because the blocks refuse 2024+ bars."""
    day = pd.DatetimeIndex(m1.index).normalize()
    d = m1.groupby(day).agg(high=("high", "max"), low=("low", "min"))
    ruler = (d["high"] - d["low"]).rolling(10).mean().shift(1)
    return pd.Series(ruler.reindex(day).to_numpy(), index=m1.index, name="adr")


def measure(m1: pd.DataFrame) -> pd.DataFrame:
    """One row per closed 1-minute bar (index = its open): move, usual, adr, times_usual, share_of_day. A row uses its
    bar and earlier ones only (the cut-off-the-future test, tests/test_live.py)."""
    w = f"{config.LIVE_MOVE_MINUTES}min"
    hi, lo = m1["high"].rolling(w).max(), m1["low"].rolling(w).min()
    n = m1["high"].rolling(w).count()
    move = (hi - lo).where(n >= config.LIVE_MOVE_MIN_BARS)
    ny = clock.server_to_utc(pd.DatetimeIndex(m1.index)).tz_convert(NY).tz_localize(None)
    cells = pd.DataFrame({"move": move.to_numpy(), "day": ny.normalize(), "minute": ny.hour * 60 + ny.minute})
    piv = cells.pivot_table(index="day", columns="minute", values="move", aggfunc="first", dropna=False)
    usual = piv.rolling(config.LIVE_MOVE_USUAL_DAYS, min_periods=config.LIVE_MOVE_USUAL_MIN_DAYS).median().shift(1)
    rows = usual.index.get_indexer(cells["day"])
    cols = usual.columns.get_indexer(cells["minute"])
    vals = usual.to_numpy()
    ok = (rows >= 0) & (cols >= 0)
    u = np.full(len(cells), np.nan)
    u[ok] = vals[rows[ok], cols[ok]]
    out = pd.DataFrame({"move": move.to_numpy(), "usual": u, "adr": adr(m1).to_numpy()}, index=m1.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["times_usual"] = out["move"] / out["usual"].where(out["usual"] > 0)
        out["share_of_day"] = out["move"] / out["adr"].where(out["adr"] > 0)
    return out


def big(mv: pd.DataFrame) -> pd.Series:
    """The two sizes: at least 6 times the usual AND at least 0.3 of his ADR (a NaN is never big)."""
    return (mv["times_usual"] >= config.LIVE_MOVE_TIMES_USUAL) & (mv["share_of_day"] >= config.LIVE_MOVE_SHARE_OF_DAY)


def near_news(times: Any, news: pd.DataFrame) -> np.ndarray:
    """For each bar time (server clock): is a release of his news (`news`: columns server, colour) from its calm time
    before to 5 minutes after it?"""
    t = pd.DatetimeIndex(times).as_unit("ns").asi8
    out = np.zeros(len(t), dtype=bool)
    for colour, calm in config.LIVE_NEWS_CALM_MINUTES.items():
        r = np.sort(pd.DatetimeIndex(news.loc[news["colour"] == colour, "server"]).as_unit("ns").asi8)
        if len(r):
            out |= np.searchsorted(r, t + config.LIVE_NEWS_EARLY_MINUTES * MIN_NS, "right") > \
                np.searchsorted(r, t - calm * MIN_NS, "left")
    return out


def quiet_times(times: Any) -> pd.DatetimeIndex:
    """The times an alert goes: the first, then the first one LIVE_MOVE_QUIET_MINUTES or more after the last alert."""
    out, last = [], None
    for t in pd.DatetimeIndex(times):
        if last is None or t - last >= pd.Timedelta(minutes=config.LIVE_MOVE_QUIET_MINUTES):
            out.append(t)
            last = t
    return pd.DatetimeIndex(out)


# ============================================================================================================ live

def facts(m1: pd.DataFrame, t: pd.Timestamp, row: pd.Series) -> dict:
    """What the alert says about the move up to bar t: which way (the low first, then the high = up), from and to."""
    win = m1.loc[(m1.index > t - pd.Timedelta(minutes=config.LIVE_MOVE_MINUTES)) & (m1.index <= t)]
    hi_t, lo_t = win["high"].idxmax(), win["low"].idxmin()
    up = bool(lo_t < hi_t) if hi_t != lo_t else bool(win["close"].iloc[-1] >= win["open"].iloc[0])
    high, low = float(win["high"].max()), float(win["low"].min())
    return dict(bar=t, start=win.index[0], end=t + pd.Timedelta(minutes=1), up=up, move=float(row["move"]),
                frm=low if up else high, to=high if up else low, times_usual=float(row["times_usual"]),
                share_of_day=float(row["share_of_day"]), usual=float(row["usual"]), adr=float(row["adr"]))


def check(p, news: pd.DataFrame, last_alert: pd.Timestamp | None) -> dict | None:
    """The big move at the newest closed bar of the live prices `p` (machine/live/prices.Prices), or None: the feed is
    not live, the move is not big, his news is near, or an alert went less than 4 hours ago."""
    if not p.fresh():
        return None
    mv = p.moves()
    t, row = mv.index[-1], mv.iloc[-1]
    if not bool(big(mv.iloc[[-1]]).iloc[0]) or near_news([t], news)[0]:
        return None
    if last_alert is not None and t - pd.Timestamp(last_alert) < pd.Timedelta(minutes=config.LIVE_MOVE_QUIET_MINUTES):
        return None
    f = facts(p.m1, t, row)
    f["inr"] = p.rupees(f["move"])
    return f


def _ist(t: pd.Timestamp) -> str:
    return HA.india(clock.server_to_utc(pd.DatetimeIndex([t]))[0].to_pydatetime()).strftime("%H:%M")


def text(f: dict) -> str:
    way = "UP" if f["up"] else "DOWN"
    out = [f"⚡ Gold moved {way} ${f['move']:,.2f} in {config.LIVE_MOVE_MINUTES} minutes ({_ist(f['start'])}–"
           f"{_ist(f['end'])} India time): {f['frm']:,.2f} → {f['to']:,.2f}.",
           f"That is {f['times_usual']:.0f} times the usual for this time of day, and {round(100 * f['share_of_day'])} "
           f"in 100 of your 10-day ADR (${f['adr']:,.2f})."]
    if f.get("inr"):
        out.append(f"On 0.01 lot: ₹{f['inr']:,.0f}.")
    out += ["No US red or orange news on Forex Factory near now.",
            "It may be news the calendar does not show (a war, a leader's words, a big order). Your analysis may change.",
            RATE_WORDS]
    return "\n".join(out)


# ========================================================================================================= history

def history_news(start: Any, end: Any) -> pd.DataFrame:
    """His news of start..end from the 2015-2023 Forex Factory history (US red and orange rows with an exact time):
    columns server, colour - the same rows the live calendar keeps (machine/live/calendar.events)."""
    from machine.news import forexfactory as FF
    ff = FF.load()
    ff = ff[ff["colour"].isin(config.LIVE_COLOURS) & ff["utc"].notna() & ~ff["timeMasked"].fillna(False).astype(bool)]
    server = clock.utc_to_server(pd.DatetimeIndex(pd.to_datetime(ff["utc"], utc=True)))
    news = pd.DataFrame({"server": server, "colour": ff["colour"].to_numpy()})
    lo, hi = pd.Timestamp(str(start)) - pd.Timedelta(days=1), pd.Timestamp(f"{pd.Timestamp(str(end)).year}-12-31") + \
        pd.Timedelta(days=2)
    return news[(news["server"] >= lo) & (news["server"] <= hi)].reset_index(drop=True)


def count(start: Any = "2016", end: Any = "2020") -> dict:
    """The rule on the 1-minute bars of start..end (2016-2023 only - load_bars refuses later): the alert times and how
    many a week. The tune years set the sizes; nothing here reads a live or sealed price."""
    from machine.data import bars as BARS
    m1 = BARS.load_bars("1m", str(start), str(end))
    mv = measure(m1)
    ok = big(mv).to_numpy() & ~near_news(mv.index, history_news(start, end))
    times = quiet_times(mv.index[ok])
    weeks = (m1.index[-1] - m1.index[0]) / pd.Timedelta(days=7)
    alerts = mv.loc[times].assign(ist=[_ist(t) for t in times])
    return dict(alerts=alerts, per_week=len(times) / weeks, weeks=weeks, minutes=int(mv["move"].notna().sum()))
