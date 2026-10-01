"""Session highs and lows (L6.1) - machine/rules.py "what_a_session_high_or_low_is".

His windows, India time (strategy.md section 1): "IST: Asian 4:30–10:30 · London 12:30–15:30 · NY 17:30–20:30".
His rule (LEARNED #85): "now let's just consider the high and low of this session range, whatever it is, as a
liquidity" - the plain high and low of the exact window, no age limit (#83's other scenario is for later).
India time has no summer time, so each window is read from the bars' own UTC column (+5:30), never guessed from the
server clock. A bar belongs to a session when it opens at or after the window start and closes at or before its end.

  names(), name_of()  his three session names, and the name he meant from the way he said it ("NY" -> New York).
  so_far()  per bar inside a window: the session's high and low up to and including this bar (known at its close).
  final()   one row per session and India date: its high and low, the bars they were made on, and known_at = the
            window's end on the server clock. A session is emitted only once the data reaches its end.
Sessions need bars of 1 hour or less (1-hour bars open at :30 India time, so they fit the windows exactly).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from machine import rules
from machine.blocks._base import TF_MINUTES, bar_close, check_bars, empty

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)
FINAL_COLUMNS = {"known_at": "datetime64[ns]", "session": "object", "ist_date": "object", "high": "float64",
                 "low": "float64", "high_bar": "datetime64[ns]", "low_bar": "datetime64[ns]",
                 "first_bar": "datetime64[ns]", "last_bar": "datetime64[ns]", "bars": "int64"}


def windows() -> tuple[tuple[str, dt.time, dt.time], ...]:
    """His three windows from machine/rules.py, as (name, start, end) India times."""
    w = rules.get("what_a_session_high_or_low_is")["windows_ist"]
    return tuple((name, dt.time.fromisoformat(a), dt.time.fromisoformat(b)) for name, a, b in w)


def names() -> tuple[str, ...]:
    """His three sessions as machine/rules.py names them: Asia, London, New York."""
    return tuple(w for w, _, _ in windows())


# His own spellings of the three names (strategy.md section 1: "Asian 4:30–10:30 · London 12:30–15:30 · NY 17:30–20:30").
_SPELLINGS = {"asia": "Asia", "asian": "Asia", "london": "London", "new york": "New York", "newyork": "New York",
              "ny": "New York"}


def name_of(word: str) -> str:
    """One of his session names from the way it was said: "london" / "London session" -> London, "Asian" -> Asia,
    "NY" -> New York. Any other word is refused - the machine never guesses a session he did not name."""
    key = " ".join(str(word).replace("_", " ").lower().split()).removesuffix(" session")
    got = _SPELLINGS.get(key)
    if got not in names():
        raise ValueError(f"session {word!r}: his sessions are {', '.join(names())}")
    return got


def _assign(bars: pd.DataFrame, tf: str):
    check_bars(bars, tf, ("high", "low", "utc"))
    if TF_MINUTES[tf] > 60:
        raise ValueError(f"sessions need bars of 1 hour or less; {tf} bars do not fit his windows.")
    ist = pd.DatetimeIndex(bars["utc"]).tz_convert("UTC").tz_localize(None) + IST_OFFSET
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    day = ist.normalize()
    open_min = ((ist - day) / pd.Timedelta(minutes=1)).to_numpy()
    close_min = open_min + step / pd.Timedelta(minutes=1)
    name = np.full(len(bars), None, dtype=object)
    for w, a, b in windows():
        a_m, b_m = a.hour * 60 + a.minute, b.hour * 60 + b.minute
        name[(open_min >= a_m) & (close_min <= b_m)] = w
    return ist, day, name


def so_far(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Per bar inside a session window (index = the bar's open, server time): session, ist_date, high_so_far,
    low_so_far, known_at (the bar's close)."""
    ist, day, name = _assign(bars, tf)
    m = name != None  # noqa: E711 - an object array
    sub = pd.DataFrame({"session": name[m], "ist_date": day[m].date, "high": bars["high"].to_numpy()[m],
                        "low": bars["low"].to_numpy()[m]}, index=bars.index[m])
    g = sub.groupby(["ist_date", "session"], sort=False)
    out = pd.DataFrame({"session": sub["session"], "ist_date": sub["ist_date"],
                        "high_so_far": g["high"].cummax(), "low_so_far": g["low"].cummin()}, index=sub.index)
    out["known_at"] = bar_close(out.index, tf)
    return out


def final(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """One row per (India date, session) whose window has ended inside the data: high, low, high_bar, low_bar,
    first_bar, last_bar, bars (how many), known_at = the window's end on the server clock."""
    ist, day, name = _assign(bars, tf)
    m = name != None  # noqa: E711
    if not m.any():
        return empty(FINAL_COLUMNS)
    ends = {w: b for w, _, b in windows()}
    idx = bars.index[m]
    sub = pd.DataFrame({"session": name[m], "ist_date": day[m], "high": bars["high"].to_numpy()[m],
                        "low": bars["low"].to_numpy()[m], "bar": idx, "ist": ist[m]})
    g = sub.groupby(["ist_date", "session"], sort=False)
    hi_i, lo_i = g["high"].idxmax(), g["low"].idxmin()    # first bar of the extreme when it repeats
    agg = g.agg(high=("high", "max"), low=("low", "min"), first_bar=("bar", "first"), last_bar=("bar", "last"),
                last_ist=("ist", "last"), bars=("bar", "size"))
    agg["high_bar"] = sub.loc[hi_i.to_numpy(), "bar"].to_numpy()
    agg["low_bar"] = sub.loc[lo_i.to_numpy(), "bar"].to_numpy()
    agg = agg.reset_index()
    end_ist = agg["ist_date"] + pd.to_timedelta([ends[s].hour * 60 + ends[s].minute for s in agg["session"]],
                                                unit="min")
    # the window's end on the server clock: the last bar's server time + (window end - that bar's India time)
    agg["known_at"] = agg["last_bar"] + (end_ist - agg["last_ist"])
    data_end = bar_close(bars.index[[-1]], tf)[0]
    agg = agg[agg["known_at"] <= data_end]
    agg["ist_date"] = agg["ist_date"].dt.date
    out = agg[list(FINAL_COLUMNS)].astype({"bars": "int64"})
    return out.sort_values(["known_at", "session"], kind="stable", ignore_index=True)
