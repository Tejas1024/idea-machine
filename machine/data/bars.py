"""Gold bars on any timeframe, made from the 1-minute bars (L1).

    from machine.data.bars import load_bars
    d = load_bars("1h", "2016", "2020")      # 2016-01-01 .. 2020-12-31, both ends included (like df.loc)

Every frame: index "server_time" = the bar's OPEN time on the broker's clock (MT5 server time, what his MT5 shows);
columns open high low close (BID prices, $), vol (ticks), spread (points, 1 point = $0.01), n_m1 (1-min bars inside),
utc (the bar's open time in UTC). India time: machine.data.clock.to_ist(d.index).

How bars are built, the way MT5 builds its own (checked: all 561,728 raw MT5 5-min bars 2016-2023):
  open = first open, high = highest high, low = lowest low, close = last close, vol = sum,
  spread = the SMALLEST 1-min spread inside the bar (MT5 stores the minimum), n_m1 = how many 1-min bars.
Days follow the BROKER day: a 1d bar is one server calendar day, 4h bars start at 00/04/08/12/16/20 server time,
a 1w bar is Monday 00:00 to Sunday 23:59 server time (MT5 labels the same week by its Sunday).
Always check_dates: any request or result touching 2024+ is refused (LEARNED #96, #104).
Built bars are cached in data/cache/, keyed by the 1-min data's content hash AND the code that built them (lesson
B19: a cache must carry the fingerprint of what made it).
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from machine import config, guards
from machine.data import clock, hashes

TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
_FLOOR = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}   # floored from midnight: 4h starts 00/04/.../20
PRICE_COLUMNS = ["open", "high", "low", "close", "vol", "spread"]


def load_bars(tf: str, start: Any = None, end: Any = None) -> pd.DataFrame:
    """Bars of timeframe tf (1m 5m 15m 1h 4h 1d 1w) from start to end, server clock, both ends included.

    start / end: a time or text like "2019", "2019-03", "2019-03-11 08:00" (server time); None = from 2016 / to the
    end of 2023. A request that touches 2024 or later is refused before anything is read.
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"timeframe {tf!r} unknown; use one of {', '.join(TIMEFRAMES)}.")
    asked = [x for x in (start, end) if x is not None]
    if asked:
        guards.check_dates(asked)
    bars = _all_bars(tf, _m1_content_hash())
    out = bars.loc[start:end].copy()
    return guards.check_dates(out)


def m1_fingerprint() -> str:
    """The 1-min data's content hash (what a trial file records about the prices it ran on)."""
    return _m1_content_hash()


def _m1_content_hash() -> str:
    if not config.GOLD_M1.is_file():
        raise FileNotFoundError(f"{config.GOLD_M1} is missing — run: python -m machine.data.convert "
                                "(the raw CSVs first: machine/data/README.md)")
    in_file = hashes.parquet_meta(config.GOLD_M1)["content_sha256"]
    recorded = hashes.read_data_hashes().get(_rel(config.GOLD_M1), {}).get("content")
    if recorded != in_file:
        raise ValueError(f"{_rel(config.GOLD_M1)} and {_rel(config.DATA_HASHES)} disagree on the content hash — "
                         "rerun: python -m machine.data.convert")
    return in_file


@lru_cache(maxsize=None)
def _all_bars(tf: str, content: str) -> pd.DataFrame:
    """Every bar 2016-2023 for tf. content (the 1-min content hash) is part of the key, so new data = new bars."""
    if tf == "1m":
        return _with_utc(_read_m1())
    path = config.BARS_CACHE_DIR / f"bars_{tf}_{content[:16]}_{_code_fingerprint()[:12]}.parquet"
    if path.is_file():
        return pd.read_parquet(path).set_index("server_time")
    bars = _with_utc(aggregate(_read_m1(), tf))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    bars.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return bars


def _read_m1() -> pd.DataFrame:
    m1 = pd.read_parquet(config.GOLD_M1, columns=["time", *PRICE_COLUMNS]).set_index("time")
    m1.index = pd.DatetimeIndex(m1.index).as_unit("ns").rename("server_time")
    m1["n_m1"] = 1
    return guards.check_dates(m1)


def bar_open_times(t: pd.DatetimeIndex, tf: str) -> pd.DatetimeIndex:
    """For each 1-min time, the open time of the tf bar it belongs to (the broker day — see the module text)."""
    t = pd.DatetimeIndex(t)
    if tf == "1m":
        return t
    if tf == "1d":
        return t.normalize()
    if tf == "1w":
        return t.normalize() - pd.to_timedelta(t.dayofweek, unit="D")
    if tf in _FLOOR:
        return t.floor(_FLOOR[tf])
    raise ValueError(f"timeframe {tf!r} unknown; use one of {', '.join(TIMEFRAMES)}.")


def aggregate(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1-min bars (index = server time) -> tf bars, the MT5 way (see the module text)."""
    if tf == "1m":
        raise ValueError("aggregate makes 5m 15m 1h 4h 1d 1w, not '1m'.")
    key = bar_open_times(pd.DatetimeIndex(m1.index), tf)
    g = m1.groupby(key.rename("server_time"), sort=True)
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                 vol=("vol", "sum"), spread=("spread", "min"), n_m1=("open", "size"))


def _with_utc(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["utc"] = clock.server_to_utc(pd.DatetimeIndex(bars.index))
    return bars


def _code_fingerprint() -> str:
    """SHA-256 of the code that builds bars (this file + the clock), so changed code never reuses an old cache."""
    h = hashlib.sha256()
    for f in (Path(__file__), Path(clock.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()


def _rel(path: Path) -> str:
    return path.relative_to(config.ROOT).as_posix()
