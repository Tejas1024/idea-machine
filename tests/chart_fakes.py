"""Hand-made bars and a hand-made trade for the picture tests (no price data needed - CI runs these)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from machine.charts import draw as D


def bars_5m(n: int = 400, start: str = "2019-03-11 00:00", seed: int = 7) -> pd.DataFrame:
    """A calm random walk of 5-minute candles, MT5 server time, no weekend inside."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="5min")
    close = 1300 + np.cumsum(rng.normal(0, 0.45, n))
    open_ = np.r_[1300.0, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.5, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx).rename_axis(
        "server_time")


def to_1h(m5: pd.DataFrame) -> pd.DataFrame:
    g = m5.resample("1h")
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last()}).dropna()


def fake_trade(m5: pd.DataFrame, i_fill: int, i_exit: int, direction: str = "sell"):
    """Only what the picture builders read of a tester Trade: filled at bar i_fill's close, closed at bar i_exit's."""
    fill, out = float(m5["close"].iloc[i_fill]), float(m5["close"].iloc[i_exit])
    sign = 1 if direction == "buy" else -1
    usd = (out - fill) * sign
    return SimpleNamespace(
        direction=direction, fill_ts=m5.index[i_fill], fill_px=fill, lots=0.01,
        exits=(SimpleNamespace(ts=m5.index[i_exit], px=out, reason="target", lots=0.01),),
        net_inr=usd * 83 - 40)


def good_trade_picture(m5: pd.DataFrame, h1: pd.DataFrame, subject: str = "test trade 1") -> D.Picture:
    """A short buy with a gap box, a target line, a stop line and the ADR band on the hour chart."""
    t = fake_trade(m5, 180, 196, "buy")
    gap = D.Box(m5.index[170], m5.index[180], t.fill_px - 0.70, t.fill_px)
    out_px = t.exits[0].px
    items_5m = [gap, D.Level(out_px, "target", start=m5.index[180]),
                D.Level(t.fill_px - 2.0, "stop", start=m5.index[180])]
    day = m5.index[0]
    adr = D.ADRBand(day, day + pd.Timedelta(hours=24), float(m5["open"].iloc[0]), adr5=9.0, adr10=11.0)
    return D.trade_picture(t, m5, h1, subject=subject, sub="11 Mar 2019 · a test trade", items_5m=items_5m,
                           items_1h=[adr])


def good_question_picture(m5: pd.DataFrame, h1: pd.DataFrame, subject: str = "test question 1") -> D.Picture:
    """Two highs joined by a line, touches numbered 1 and 2; the charts end at the entry candle."""
    w = m5.iloc[120:200]
    i1 = int(np.argmax(w["high"].to_numpy()[:40]))
    i2 = 45 + int(np.argmax(w["high"].to_numpy()[45:70]))
    t1, t2 = w.index[i1], w.index[i2]
    p1, p2 = float(w["high"].iloc[i1]), float(w["high"].iloc[i2])
    entry = w.index[i2 + 2]
    line = D.Line((t1, p1), (t2, p2), touches=(t1, t2), name="equal highs", extend_to=entry)
    q = D.Question("Are 1 and 2 the equal highs you mean?", ("Answer YES, or NO and say what is different.",))
    return D.question_picture(m5, h1, entry_at=entry, question=q, subject=subject, heading="IDEA #1 - A QUESTION",
                              sub="11 Mar 2019 · price made two highs", items_5m=[line], bars_5m=60)


def good_whole_trade_picture(m5: pd.DataFrame, h1: pd.DataFrame, subject: str = "test whole trade 1") -> D.Picture:
    t = fake_trade(m5, 180, 330, "buy")
    gap = D.Box(m5.index[170], m5.index[180], t.fill_px - 0.70, t.fill_px)
    target = D.Level(t.exits[0].px, "target", start=m5.index[180])
    return D.whole_trade_picture(t, m5, h1, subject=subject, sub="11 Mar 2019 · a test trade", items_1h=[target],
                                 point_items={"B1 in": [gap], "B1 C out": [target]}, min_hour_bars=30)
