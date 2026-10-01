"""Small fake bars for the block tests (no price data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Monday 7 Jan 2019, winter: server time = UTC + 2 (machine/data/README.md)
START = pd.Timestamp("2019-01-07 01:00")
SERVER_MINUS_UTC = pd.Timedelta(hours=2)


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
