"""Small fake bars for the block tests (no price data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import settings

# Monday 7 Jan 2019, winter: server time = UTC + 2 (machine/data/README.md)
START = pd.Timestamp("2019-01-07 01:00")
SERVER_MINUS_UTC = pd.Timedelta(hours=2)

# The highest price before made-up bars (they all start in 2019): gold's September 2011 record, because the real
# 2016-2018 daily highs never reached it - on the real bars Market.ath_before() gives the same 1920.94 (checked
# 4 Oct 2026). Handed over, a compile on made-up prices never opens the real price file, which GitHub's CI does not
# have (data/ is not in git): 11 tests failed there on every push.
FAKE_ATH_BEFORE = float(settings.get("all_time_high_before_2016"))


def fake_market(m1: pd.DataFrame, tfs=("5m", "1h", "1d", "1w")):
    """Market.from_m1 for made-up 1-minute bars, with FAKE_ATH_BEFORE."""
    from machine.ideas.compile import Market
    return Market.from_m1(m1, tfs, ath_before=FAKE_ATH_BEFORE)


def bars_from(rows, tf_minutes: int = 5, start=START) -> pd.DataFrame:
    """rows = [(open, high, low, close), ...] -> bars like load_bars (server-time index, utc column)."""
    idx = pd.DatetimeIndex([start + pd.Timedelta(minutes=tf_minutes * i) for i in range(len(rows))],
                           name="server_time").as_unit("ns")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["utc"] = (idx - SERVER_MINUS_UTC).tz_localize("UTC")
    return df


def candles(closes, wick: float = 0.2) -> list[tuple]:
    """Candles that walk from close to close, each with a small wick both ways."""
    out, prev = [], float(closes[0])
    for c in closes:
        c = float(c)
        o = prev
        out.append((o, max(o, c) + wick, min(o, c) - wick, c))
        prev = c
    return out


def random_bars(n: int = 3000, seed: int = 1, tf_minutes: int = 5, start=START) -> pd.DataFrame:
    """A random walk on the 1-cent grid, every day of the week (enough for the no-future tests)."""
    rng = np.random.default_rng(seed)
    close = 1300 + np.cumsum(rng.normal(0, 0.6, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
    rows = np.round(np.c_[open_, high, low, close], 2)
    return bars_from([tuple(r) for r in rows], tf_minutes, start)
