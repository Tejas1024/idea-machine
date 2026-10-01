"""The clock of the price data (L1): MT5 SERVER time -> UTC -> IST (India, where Tejas trades).

MEASURED, not assumed (python -m machine.data.measure_clock -> data/server_offset.csv):
  server time = Eastern European time: UTC+2 in winter, UTC+3 in summer — with the EUROPEAN summer-time dates
  (last Sunday of March to last Sunday of October), NOT the US dates.
  - 2019-2023, prices: MT5 5-min closes against Dukascopy (UTC), every week, shifts -1..+5 hours. All 261 weeks
    are clear and every one matches this rule — including the 17 weeks when the US and Europe differ, where the
    "US summer" rule is wrong every time. In every week the right hour's typical gap between the two feeds is
    1-22 cents; the next-best hour's is at least $0.55 (10 times or more).
  - 2016-2018: no second feed, so the rule 2019-2023 shows is APPLIED. One check agrees: gold's daily stop
    (17:00 New York) leaves server hour 23 empty in exactly the weeks the US and Europe differ — 9 of 9 such
    weeks in 2016-2018, 0 of 148 other weeks — the same pattern as 2019-2023 (17 of 17, 0 of 243).
This is the clock of THIS data file (the old notes say MetaQuotes-Demo). Another broker's server — XM for the live
trial — is measured again, never assumed (the old live path was once 3 hours out: old/CLAUDE-old-phase.md).
The clock only changes at 03:00-04:00 server time on a Sunday, when gold is closed; a time inside that hour
is refused rather than guessed.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

SERVER_ZONE = "Europe/Athens"      # UTC+2 / UTC+3 on EU dates; same rule as Europe/Bucharest (the old engine's)
SERVER_RULE = "UTC+2 in winter, UTC+3 in EU summer time (last Sunday of March to last Sunday of October)"
IST_ZONE = "Asia/Kolkata"          # UTC+5:30, no summer time


def server_to_utc(t: Any) -> Any:
    """Server time (no time zone) -> UTC (with time zone). Takes a time, a DatetimeIndex or a Series."""
    return _map(t, _server_to_utc_index, "server_to_utc", want_naive=True)


def utc_to_server(t: Any) -> Any:
    """A time with a time zone (UTC or any) -> server time (no time zone). Naive input is refused: say its clock."""
    return _map(t, lambda idx: idx.tz_convert(SERVER_ZONE).tz_localize(None), "utc_to_server", want_naive=False)


def to_ist(t: Any) -> Any:
    """-> India time (with time zone). A naive time is SERVER time (the data's own clock); a zoned one is converted."""
    def one(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return (_server_to_utc_index(idx) if idx.tz is None else idx).tz_convert(IST_ZONE)
    return _map(t, one, "to_ist", want_naive=None)


def server_offset_hours(t: Any) -> Any:
    """Server time minus UTC, in hours (2 or 3), for a server time / DatetimeIndex / Series."""
    def one(idx: pd.DatetimeIndex) -> pd.Index:
        utc = _server_to_utc_index(idx).tz_localize(None)
        return pd.Index(((idx - utc) / pd.Timedelta(hours=1)).astype(int))
    return _map(t, one, "server_offset_hours", want_naive=True)


def _server_to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    try:
        return idx.tz_localize(SERVER_ZONE, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    except (ValueError, TypeError) as err:  # pytz-style NonExistent/AmbiguousTimeError are ValueErrors
        raise ValueError(f"server time inside the clock change (a Sunday 03:00-04:00, gold is closed): {err}") from err


def _map(t: Any, fn, name: str, want_naive: bool | None) -> Any:
    if isinstance(t, pd.Series):
        idx = pd.DatetimeIndex(t)
        _check_tz(idx, name, want_naive)
        return pd.Series(fn(idx), index=t.index, name=t.name)
    if isinstance(t, pd.DatetimeIndex):
        _check_tz(t, name, want_naive)
        return fn(t)
    if isinstance(t, (str, dt.datetime, pd.Timestamp)):
        idx = pd.DatetimeIndex([pd.Timestamp(t)])
        _check_tz(idx, name, want_naive)
        return fn(idx)[0]
    raise TypeError(f"{name} takes a time, a DatetimeIndex or a Series of times, not a {type(t).__name__}.")


def _check_tz(idx: pd.DatetimeIndex, name: str, want_naive: bool | None) -> None:
    if want_naive is True and idx.tz is not None:
        raise ValueError(f"{name} takes SERVER time, which has no time zone; this one is {idx.tz}.")
    if want_naive is False and idx.tz is None:
        raise ValueError(f"{name} needs a time with a time zone (e.g. UTC); a naive time's clock is unknown.")
