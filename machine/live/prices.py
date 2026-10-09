"""Live gold prices for the live alerts (L17a step 2) - T14 use 1, "live-trial": live prices from today on, plus the
recent days an alert needs, ONLY for alerts and paper trades - never to test, tune or choose (T14 question 1: "Yes").

    from machine.live import prices as LP
    p, why = LP.live(now)         # Prices (the last LIVE_PAST_DAYS of CLOSED 1-minute bars from MT5), or None + why not
    p.m1                          # index = the bar's open, MT5 server clock - like machine.data.bars.load_bars

Read only: no order, ever (hard rule 13). Only MetaQuotes-Demo (config.MT5_SERVER, the server the price data and its
measured clock came from - machine/host/prices.py _open refuses another). Only CLOSED bars: the bar still forming is
left out. The bars pass the lock: guards.check_dates(..., unlock=True, use="live-trial"), one line in
data/sealed_access.log per week (machine/live/opened.py). MT5 trouble never stops a pass: the news alerts go on
without prices, and the pass says why.

THE TAPE: each pass writes the bars that closed in the last LIVE_TAPE_FRESH_MINUTES - what the alerts saw LIVE - to
LIVE_TAPE_DIR/<server day>.csv in L1's own text (machine/host/prices.lines, the text the daily run keeps). The daily
check (machine/live/check.py) compares that tape with MT5's history of the day.
Nothing here is kept for research: no test, tune or research code reads LIVE_DIR.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from machine import config
from machine.data import clock
from machine.host import prices as HP
from machine.live import opened

WHO = "machine.live (L17a, the live alerts)"
WHY = "live gold prices (closed 1-minute bars of the last days) for the live alerts only (T14 use 1)"
EPOCH = pd.Timestamp("1970-01-01")
OPEN_GAP = pd.Timedelta(minutes=5)          # reaction.OPEN_GAP_MIN: no bar within 5 minutes = the market is shut


def server_now(now: dt.datetime) -> pd.Timestamp:
    """`now` (an aware time) on the MT5 server's clock, with no time zone - like the bars."""
    return clock.utc_to_server(pd.DatetimeIndex([pd.Timestamp(now).tz_convert("UTC")]))[0]


def _sec(t: pd.Timestamp) -> int:
    return int((pd.Timestamp(t) - EPOCH) // pd.Timedelta(seconds=1))


def frame(rates) -> pd.DataFrame:
    """MT5's bars as a frame like load_bars: index server_time (the bar's open), open high low close vol spread, utc.
    The index is in nanoseconds, as load_bars' is: pandas 3 would keep MT5's seconds, and a time with microseconds
    ("now") then cannot be looked up in it (5 Oct: it would have stopped every pass with a before-alert due)."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(rates["time"], dtype=np.int64), unit="s"),
                           name="server_time").as_unit("ns")
    m1 = pd.DataFrame({c: np.asarray(rates[c], dtype=float) for c in ("open", "high", "low", "close")}, index=idx)
    m1["vol"] = np.asarray(rates["tick_volume"], dtype=np.int64)
    m1["spread"] = np.asarray(rates["spread"], dtype=np.int64)
    m1["utc"] = clock.server_to_utc(idx)
    return m1


@dataclass
class Charts:
    """What his bias and his levels need (step 3), all CLOSED bars, server clock, frames like load_bars:
    m1  1-minute bars from the Monday on or before LIVE_LEVEL_DAYS ago (his levels: 60 days already gave the same
        levels as the whole history, 2018-2020)
    h1  the hour chart of the last LIVE_HOUR_BIAS_DAYS (his hour bias)
    d1, w1  the day and week charts from LIVE_CHARTS_FROM (his day and week bias hold while their trade travels: only
        the whole history reads them the same) - a week labelled by its Monday, as machine/data/bars.py does."""
    m1: pd.DataFrame
    h1: pd.DataFrame
    d1: pd.DataFrame
    w1: pd.DataFrame


TF_SECONDS = {"m1": 60, "h1": 3600, "d1": 86400, "w1": 7 * 86400}


def level_start(at: pd.Timestamp) -> pd.Timestamp:
    """The Monday 00:00 (server) on or before LIVE_LEVEL_DAYS before `at`: whole days and weeks, as Market.load cuts."""
    d = (pd.Timestamp(at) - pd.Timedelta(days=config.LIVE_LEVEL_DAYS)).normalize()
    return d - pd.Timedelta(days=d.dayofweek)


def charts(now: dt.datetime | None = None, *, mt5=None) -> tuple[Charts | None, str]:
    """The live charts now from MT5 (read only, MetaQuotes-Demo only, closed bars only), or (None, why not). MT5's
    week bars are labelled by their Sunday: moved to the Monday (the same week - 417 of 417 the same, 2016-2023)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    at = server_now(now)
    try:
        mt5 = mt5 or HP._mt5()
        HP._open(mt5)
    except Exception as e:
        return None, f"no charts: {str(e)[:200]}"
    ask = {"m1": (mt5.TIMEFRAME_M1, level_start(at)),
           "h1": (mt5.TIMEFRAME_H1, at - pd.Timedelta(days=config.LIVE_HOUR_BIAS_DAYS)),
           "d1": (mt5.TIMEFRAME_D1, pd.Timestamp(config.LIVE_CHARTS_FROM)),
           "w1": (mt5.TIMEFRAME_W1, pd.Timestamp(config.LIVE_CHARTS_FROM))}
    out = {}
    try:
        for name, (code, start) in ask.items():
            r = mt5.copy_rates_range(config.MT5_SYMBOL, code, _sec(start), _sec(at))
            if r is None or not len(r):
                return None, f"no charts: MT5 gave no {name} bars ({mt5.last_error()})"
            r = r[r["time"] + TF_SECONDS[name] <= _sec(at)].copy()         # closed bars only
            if name == "w1":
                r["time"] += 86400                                          # Sunday -> Monday
                r = r[r["time"] >= _sec(pd.Timestamp(config.LIVE_CHARTS_FROM))]
            out[name] = frame(r)
    finally:
        mt5.shutdown()
    from machine.live import calendar as LC
    opened.week(out["m1"].index, LC.sunday_of(now), use="live-trial", who=WHO, why=WHY)
    return Charts(**out), ""


def history_charts(at: pd.Timestamp) -> Charts:
    """The same charts cut the same way from the 2016-2023 bars, known at server time `at` (a rehearsal, a test):
    closed bars only (load_bars refuses any later year)."""
    from machine.data import bars as BARS
    at = pd.Timestamp(at)

    def closed(tf: str, start: pd.Timestamp, name: str) -> pd.DataFrame:
        b = BARS.load_bars(tf, start, at)
        return b[b.index + pd.Timedelta(seconds=TF_SECONDS[name]) <= at]

    start = pd.Timestamp(config.LIVE_CHARTS_FROM)
    return Charts(closed("1m", level_start(at), "m1"),
                  closed("1h", at - pd.Timedelta(days=config.LIVE_HOUR_BIAS_DAYS), "h1"),
                  closed("1d", start, "d1"), closed("1w", start, "w1"))


@dataclass
class Prices:
    """Closed 1-minute bars known at `at` (server clock), and the rupees per dollar then. charts_fn: where his charts
    come from when an alert needs his bias and levels (MT5 live; the history in a rehearsal) - read once."""
    m1: pd.DataFrame
    at: pd.Timestamp
    usd_inr: float | None = None                # MT5's USDINR now (None: not known - then no rupees are shown)
    source: str = ""
    _moves: pd.DataFrame | None = field(default=None, repr=False)
    charts_fn: Callable[[], Charts | None] | None = field(default=None, repr=False)
    _charts: list = field(default_factory=list, repr=False)

    def charts(self) -> Charts | None:
        if not self._charts:
            self._charts.append(self.charts_fn() if self.charts_fn is not None else None)
        return self._charts[0]

    def newest_close(self) -> pd.Timestamp | None:
        return self.m1.index[-1] + pd.Timedelta(minutes=1) if len(self.m1) else None

    def fresh(self) -> bool:
        """The newest bar closed at most LIVE_FEED_FRESH_MINUTES ago: the market is open and the feed is live."""
        last = self.newest_close()
        return last is not None and self.at - last <= pd.Timedelta(minutes=config.LIVE_FEED_FRESH_MINUTES)

    def before(self, t: pd.Timestamp) -> float | None:
        """The last price before minute t: the close of the last bar that opened before t - reaction.py's p0. None when
        that bar is more than 5 minutes older than t (the market was shut)."""
        i = int(self.m1.index.searchsorted(pd.Timestamp(t), "left")) - 1
        if i < 0 or pd.Timestamp(t) - self.m1.index[i] > OPEN_GAP:
            return None
        return float(self.m1["close"].iloc[i])

    def moves(self) -> pd.DataFrame:
        """machine/live/move.py's measure on these bars (made once)."""
        if self._moves is None:
            from machine.live import move
            self._moves = move.measure(self.m1)
        return self._moves

    def rupees(self, usd: float) -> float | None:
        """A move of `usd` dollars on 0.01 lot (1 ounce) in rupees at today's rate; None when the rate is not known."""
        if not self.usd_inr or not np.isfinite(usd):
            return None
        return abs(float(usd)) * config.OUNCES_PER_LOT * 0.01 * self.usd_inr


def _usd_inr(mt5) -> float | None:
    """Rupees per dollar now: MT5's USDINR bid (a currency price, not gold). None when MT5 has none."""
    try:
        if not mt5.symbol_select("USDINR", True):
            return None
        tick = mt5.symbol_info_tick("USDINR")
        rate = float(tick.bid) if tick is not None else 0.0
        return rate if 40.0 < rate < 200.0 else None
    except Exception:
        return None


def live(now: dt.datetime | None = None, *, mt5=None, days: int | None = None,
         tape_dir: Path | None = None) -> tuple[Prices | None, str]:
    """The live prices now: (Prices, "") or (None, why not). MT5 trouble is a "why not", never an error. A "no" in T14
    for live-trial raises guards.SealedYearsError (the caller stops the price part)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    at = server_now(now)
    try:
        mt5 = mt5 or HP._mt5()
        HP._open(mt5)                                   # starts or joins MT5; refuses another server than MetaQuotes-Demo
    except Exception as e:
        return None, f"no live prices: {str(e)[:200]}"
    try:
        lo = at - pd.Timedelta(days=days or config.LIVE_PAST_DAYS)
        rates = mt5.copy_rates_range(config.MT5_SYMBOL, mt5.TIMEFRAME_M1, _sec(lo), _sec(at) + 60)
        if rates is None or not len(rates):
            return None, f"no live prices: MT5 gave no 1-minute bars ({mt5.last_error()})"
        rates = rates[rates["time"] + 60 <= _sec(at)]   # closed bars only: the one still forming is left out
        rate = _usd_inr(mt5)
    finally:
        mt5.shutdown()
    if not len(rates):
        return None, "no live prices: no closed bar yet"
    m1 = frame(rates)
    from machine.live import calendar as LC
    opened.week(m1.index, LC.sunday_of(now), use="live-trial", who=WHO, why=WHY)
    tape(rates, at, tape_dir)
    return Prices(m1, at, rate, f"MT5 {config.MT5_SERVER}", charts_fn=lambda: charts(now)[0]), ""


# ===================================================================================================== the tape

def tape(rates, at: pd.Timestamp, folder: Path | None = None) -> int:
    """Append to LIVE_TAPE_DIR/<server day>.csv the bars that closed in the last LIVE_TAPE_FRESH_MINUTES (seen LIVE) and
    are newer than the day's last line - in L1's own text. Returns how many were added."""
    base = Path(folder or config.LIVE_TAPE_DIR)
    cut = _sec(at) - config.LIVE_TAPE_FRESH_MINUTES * 60
    fresh = rates[rates["time"] + 60 >= cut]
    if not len(fresh):
        return 0
    added = 0
    days = np.asarray(fresh["time"], dtype=np.int64) // 86400
    for d in np.unique(days):
        rows = fresh[days == d]
        path = base / f"{HP.day_of(int(d) * 86400).isoformat()}.csv"
        last = HP.last_time_text(path)
        if last and last != "time":
            rows = rows[rows["time"] > _sec(pd.Timestamp(last))]
        if not len(rows):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.is_file()
        with open(path, "a", encoding="ascii", newline="\n") as fh:
            fh.write((HP.HEADER + "\n" if new else "") + "\n".join(HP.lines(rows)) + "\n")
        added += len(rows)
    return added
