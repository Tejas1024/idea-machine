"""The news calendar for the machine (L12): when news comes, what kind, and what was known at any moment.

    from machine.news.calendar import load_events, minutes_to_next, minutes_since_last, is_news_day

    ev = load_events("2019", "2020")                 # red + orange (settings news_events), sorted by time
    bars = load_bars("5m", "2019", "2020")
    to = minutes_to_next(bars.index + pd.Timedelta(minutes=5))   # pass the moment you DECIDE at (a bar's close)

Times: a naive time is SERVER time (the price data's clock, like every bar index); a time with a zone is converted.
The table (data/news/events.parquet, made by `python -m machine.news.build`) has utc / server / ist for every row.

What is known when (the "cut off the future" rule, likhith.md R4):
  - the TIME, event, kind, importance, forecast and previous number of a SCHEDULED release are known in advance
    (the agencies publish their calendars months ahead). WARNING (L12b.1, 30 Sep 2026): MT5's `forecast` is NOT the
    economists' consensus (8 May 2020 jobs: MT5 +139K, the consensus -22,000K) - for a surprise use
    machine.news.surprise (Forex Factory's forecast), never actual - forecast from this table;
  - the ACTUAL number (and the revised previous one) only from the release moment on;
  - an UNSCHEDULED release (the Fed's emergency cuts, 3 and 15 March 2020 - the Fed's own pages mark those meetings
    "unscheduled") is not known at all before it happens.
  minutes_to_next and is_news_day use only scheduled rows; minutes_since_last and last_released use rows that have
  happened. as_known_at(events, t) is the table as the world knew it at t (tests/test_news.py checks every function
  against it with guards.assert_no_future).

importance = MT5's rating: "high" or "moderate". Tejas's news is Forex Factory red AND orange (T7, settings
news_events); MT5 high + moderate stands in for it. MT5's "moderate" is wider than Forex Factory's orange (it
includes bond auctions, oil rig counts, CFTC positions, many Fed speakers): which kinds move gold is measured, not
assumed (plan/work/news.md NQ1, L12b.2).
kind = Claude's grouping of MT5's event names (KINDS below), not Tejas's words.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from machine import config, guards, settings
from machine.data import clock

# settings news_events -> MT5 importance. "forex_factory_red_and_orange" is Tejas's rule (T7); "forex_factory_red" is
# the comparison four-boxes B3-29 asks for ("red only, or red + orange - the machine checks which is best").
IMPORTANCE_FOR = {
    "forex_factory_red_and_orange": ("high", "moderate"),
    "forex_factory_red": ("high",),
}

# kind of each MT5 event name: the first pattern that matches wins. Claude's grouping (L12), for "event type".
KINDS: tuple[tuple[str, str], ...] = (
    ("fed", r"^(FOMC (Statement|Press Conference|Minutes|Economic Projections)|Fed Interest Rate Decision|Fed Chair "
            r"|Fed Monetary Policy Report|Fed Beige Book|Jackson Hole|Fed (CCAR|DFAST) )"),
    ("fed_speaker", r"^(FOMC Member |Fed Governor |Fed Vice Chair)"),
    ("growth", r"^Nonfarm Productivity"),
    ("jobs", r"Nonfarm|Unemployment Rate|Jobless Claims|Average Hourly Earnings|JOLTS|Participation Rate"
             r"|Employment Cost|Employment$|Employment Change|Labor Market"),
    ("inflation", r"CPI|PPI|PCE Price|Price Index|Prices Paid|Inflation Expectations|Unit Labor Costs"),
    ("housing", r"Home Sales|Housing Starts|Building Permits|HPI|Construction Spending"),
    ("energy", r"^EIA |Baker Hughes|WASDE"),
    ("auction", r"Auction$"),
    ("positions", r"^CFTC |^TIC "),
    ("other", r"Federal Budget|Treasury Secretary|President"),
    ("growth", r"."),
)
_KIND_RE = tuple((kind, re.compile(pattern)) for kind, pattern in KINDS)


def event_kind(name: str) -> str:
    """'Nonfarm Payrolls' -> 'jobs', 'CPI m/m' -> 'inflation', 'FOMC Statement' -> 'fed' ... (KINDS)."""
    for kind, pattern in _KIND_RE:
        if pattern.search(name):
            return kind
    raise ValueError(f"no kind for {name!r}")  # never: the last pattern matches anything


def load_events(start: Any = None, end: Any = None, *, importance: tuple[str, ...] | None = None,
                kinds: tuple[str, ...] | None = None, events: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The calendar rows with SERVER time in [start, end] (both ends included, like load_bars), sorted by time.

    importance: default from settings news_events (Tejas: red + orange -> ("high", "moderate")).
    kinds / events: keep only these kinds (KINDS) / these MT5 event names.
    A request reaching 2024 is refused before anything is read (LEARNED #96).
    """
    request = [x for x in (start, end) if x is not None]
    if request:
        guards.check_dates(request)
    if not config.NEWS_EVENTS.is_file():
        raise FileNotFoundError(f"{config.NEWS_EVENTS} is missing - make it with `python -m machine.news.build` "
                                f"(after `python -m machine.news.mt5_export` on the PC with MT5).")
    ev = pd.read_parquet(config.NEWS_EVENTS)
    guards.check_dates(ev[["utc", "server"]])
    if importance is None:
        importance = importance_for(settings.get("news_events"))
    ev = ev[ev["importance"].isin(importance)]
    if kinds is not None:
        ev = ev[ev["kind"].isin(kinds)]
    if events is not None:
        ev = ev[ev["event"].isin(events)]
    day = ev["server"].fillna(ev["day"])
    if start is not None:
        ev = ev[day >= pd.Timestamp(str(start))]
        day = day.loc[ev.index]
    if end is not None:
        ev = ev[day < _end_exclusive(end)]
    return ev.sort_values(["day", "server"], na_position="first").reset_index(drop=True)


def fingerprint(events: pd.DataFrame | None = None) -> str:
    """SHA-256 (first 16) of the calendar's CONTENT - every row's moment, day, event, importance and numbers, sorted -
    so a trial file (R14) can say which calendar it used, and another PC can prove it built the same one."""
    ev = events if events is not None else pd.read_parquet(config.NEWS_EVENTS)
    cols = ["utc", "day", "event", "importance", "actual", "forecast", "previous", "scheduled"]
    text = ev[cols].sort_values(cols[:3], na_position="first").to_csv(index=False, float_format="%.6f")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def importance_for(news_events: str) -> tuple[str, ...]:
    if news_events not in IMPORTANCE_FOR:
        raise ValueError(f"news_events = {news_events!r}: the calendar knows {list(IMPORTANCE_FOR)}.")
    return IMPORTANCE_FOR[news_events]


# ---- what was known at a moment ------------------------------------------------------------------------------------

def as_known_at(events: pd.DataFrame, t: Any) -> pd.DataFrame:
    """The table as the world knew it at moment t: scheduled rows all there, unscheduled rows only once they
    happened, the actual and the revised previous number only from their release on."""
    t_utc = _utc(pd.DatetimeIndex([pd.Timestamp(t)]))[0]
    happened = events["utc"] <= t_utc
    out = events[events["scheduled"] | happened].copy()
    later = ~happened.loc[out.index]
    for col in ("actual", "revised_previous"):
        out.loc[later, col] = np.nan
    out.loc[later, "has_actual"] = False
    return out


def minutes_to_next(times: Any, events: pd.DataFrame | None = None, **filters) -> pd.Series:
    """Minutes from each moment to the next SCHEDULED release at or after it (0 = at that minute; NaN = none left).
    events: a table from load_events (default: load_events(**filters))."""
    idx = _server(times)
    ev = _exact(events if events is not None else load_events(**filters))
    ev = ev[ev["scheduled"]]
    at = ev["server"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(at, idx.to_numpy(dtype="datetime64[ns]"), side="left")
    out = np.full(len(idx), np.nan)
    ok = pos < len(at)
    out[ok] = (at[pos[ok]] - idx.to_numpy(dtype="datetime64[ns]")[ok]) / np.timedelta64(1, "m")
    return pd.Series(out, index=idx, name="minutes_to_next")


def minutes_since_last(times: Any, events: pd.DataFrame | None = None, **filters) -> pd.Series:
    """Minutes from the last release at or before each moment (scheduled or not - it has happened) to that moment."""
    idx = _server(times)
    ev = _exact(events if events is not None else load_events(**filters))
    at = ev["server"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(at, idx.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    out = np.full(len(idx), np.nan)
    ok = pos >= 0
    out[ok] = (idx.to_numpy(dtype="datetime64[ns]")[ok] - at[pos[ok]]) / np.timedelta64(1, "m")
    return pd.Series(out, index=idx, name="minutes_since_last")


def next_event(times: Any, events: pd.DataFrame | None = None, **filters) -> pd.DataFrame:
    """For each moment: the next scheduled release (its minutes away, event, kind, importance, forecast, previous).
    Several releases at the same minute (NFP + unemployment rate + earnings) -> the first by importance."""
    idx = _server(times)
    ev = _exact(events if events is not None else load_events(**filters))
    ev = ev[ev["scheduled"]].sort_values(["server", "mt5_importance"], ascending=[True, False], kind="stable")
    ev = ev.drop_duplicates("server")
    at = ev["server"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(at, idx.to_numpy(dtype="datetime64[ns]"), side="left")
    ok = pos < len(at)
    cols = ["event", "kind", "importance", "forecast", "previous"]
    picked = ev.iloc[np.where(ok, pos, 0)][cols].reset_index(drop=True)
    picked.loc[~ok, :] = None
    picked.insert(0, "minutes_to_next", minutes_to_next(idx, ev).to_numpy())
    picked.index = idx
    return picked


def last_released(times: Any, event: str, column: str = "actual", events: pd.DataFrame | None = None) -> pd.Series:
    """For each moment: `column` (actual, forecast, previous ...) of the latest `event` release at or before it."""
    idx = _server(times)
    ev = events if events is not None else load_events(events=(event,), importance=("high", "moderate", "low"))
    ev = _exact(ev[ev["event"] == event])
    at = ev["server"].to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(at, idx.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    numeric = pd.api.types.is_numeric_dtype(ev[column])
    out = pd.Series(np.nan, index=idx, name=f"{event}: {column}", dtype=float if numeric else object)
    ok = pos >= 0
    out.iloc[np.flatnonzero(ok)] = ev[column].to_numpy()[pos[ok]]
    return out


def is_news_day(days: Any, events: pd.DataFrame | None = None, clock_name: str = "server", **filters) -> pd.Series:
    """True for a day with at least one SCHEDULED release (all-day events count on their day).

    days: dates or times; each is taken as the day it falls on in `clock_name`: "server" (the broker's day, like the
    daily candles - likhith.md B4) or "ist" (India). Naive input is server time.
    """
    idx = _server(days)
    ev = events if events is not None else load_events(**filters)
    ev = ev[ev["scheduled"]]
    if clock_name == "server":
        want = idx.normalize()
        have = ev["server"].dt.normalize()
    elif clock_name == "ist":
        want = clock.to_ist(idx).tz_localize(None).normalize()
        have = ev["ist"].dt.tz_localize(None).dt.normalize()
    else:
        raise ValueError(f"clock_name is 'server' or 'ist', not {clock_name!r}")
    have = pd.DatetimeIndex(have.fillna(ev["day"])).unique()
    return pd.Series(want.isin(have), index=idx, name="is_news_day")


# ---- helpers -------------------------------------------------------------------------------------------------------

def _exact(ev: pd.DataFrame) -> pd.DataFrame:
    """Rows with an exact moment (MT5 time_mode 'date and time'), sorted by it. All-day rows have none."""
    return ev[ev["exact_time"]].sort_values("server", kind="stable")


def _server(times: Any) -> pd.DatetimeIndex:
    if isinstance(times, (str, pd.Timestamp, dt.datetime, dt.date)):
        times = [times]
    idx = pd.DatetimeIndex(times)
    return clock.utc_to_server(idx) if idx.tz is not None else idx


def _utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert("UTC") if idx.tz is not None else clock.server_to_utc(idx)


def _end_exclusive(end: Any) -> pd.Timestamp:
    """'2020' -> 2021-01-01, '2020-03' -> 2020-04-01, '2020-03-05' -> 2020-03-06, a full time -> itself + 1 ns."""
    text = str(end)
    t = pd.Timestamp(text)
    if re.fullmatch(r"\d{4}", text):
        return t + pd.DateOffset(years=1)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return t + pd.DateOffset(months=1)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return t + pd.Timedelta(days=1)
    return t + pd.Timedelta(1, "ns")
