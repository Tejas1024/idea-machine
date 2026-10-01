"""MEASURE the MT5 server clock against UTC (L1).  Run:  python -m machine.data.measure_clock
(python -m machine.data.convert runs it at the end.)

Two measurements, week by week:

1. Prices, 2019-2023: MT5 5-min closes (server time) against Dukascopy 5-min closes (UTC). For each week and each
   shift h from -1 to +5 hours, match MT5 bar t with the Dukascopy bar at t - h and take the typical gap between the
   two closes (median of |gap - the week's median gap|, so a steady price difference between the two feeds does
   not count). The right shift makes the two feeds move together: its gap is a few cents; a wrong hour is dollars.
2. The daily stop, 2016-2023: gold stops trading for one hour every day at 17:00 New York time. In the weeks when
   New York is on summer time but Europe is not, that stop falls at SERVER 23:00 under the EU rule, and at server
   00:00 under the US rule. So: count the 1-minute bars in server hour 23 on Tuesday-Thursday, every week. This
   needs no second feed, so it also covers 2016-2018, where there is no Dukascopy data. (Calibrated on 2019-2023,
   where the price check gives the answer independently.)

Candidate rules compared with the measured shift:
  US rule  server = UTC+2, or UTC+3 while New York is on summer time  (= New York time + 7 hours)
  EU rule  server = UTC+2, or UTC+3 while Europe is on summer time    (= Eastern European time, the old engine's guess)
They differ only for the 2-3 weeks each March, and the 1 week around each late October / early November, when
the US and Europe are not both on summer (or both on winter) time. Those weeks decide between them.
Writes data/server_offset.csv; prints the summary.
"""
from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from machine import config

SHIFTS = range(-1, 6)             # hours tried: server time minus UTC
PRICE_YEARS = (2019, 2023)        # the years both feeds cover (Dukascopy starts 2018-12-31)
CLEAR_RATIO = 3.0                 # "clear" = the next-best shift's gap is at least 3x the best one
MIN_MATCHED = 200                 # and at least 200 matched 5-min bars that week
NEW_YORK = ZoneInfo("America/New_York")
EUROPE = ZoneInfo("Europe/Athens")  # Eastern European time with EU summer-time dates (same as Bucharest)


def _summer(zone: ZoneInfo, when: dt.datetime) -> bool:
    return bool(when.astimezone(zone).dst())


def rule_offsets(week_starts: pd.DatetimeIndex) -> pd.DataFrame:
    """US-rule and EU-rule offset (hours) for each server week (Monday 00:00), judged at Wednesday 12:00 UTC."""
    mid = [dt.datetime(d.year, d.month, d.day, 12, tzinfo=dt.UTC) + dt.timedelta(days=2) for d in week_starts]
    return pd.DataFrame({"rule_us": [2 + _summer(NEW_YORK, m) for m in mid],
                         "rule_eu": [2 + _summer(EUROPE, m) for m in mid]}, index=week_starts)


def _week(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Monday 00:00 of each time's week (server clock)."""
    return (idx.normalize() - pd.to_timedelta(idx.dayofweek, unit="D")).rename("week")


def measure_prices() -> pd.DataFrame:
    mt5 = pd.read_parquet(config.GOLD_M5_MT5, columns=["time", "close"]).set_index("time")["close"]
    duka = pd.read_parquet(config.GOLD_M5_DUKA, columns=["time", "close"]).set_index("time")["close"]
    duka.index = duka.index.tz_convert("UTC").tz_localize(None)
    y0, y1 = PRICE_YEARS
    mt5 = mt5[(mt5.index.year >= y0) & (mt5.index.year <= y1)]
    week = _week(mt5.index)

    gaps, counts = {}, {}
    for h in SHIFTS:
        other = duka.reindex(mt5.index - pd.Timedelta(hours=h)).to_numpy()
        frame = pd.DataFrame({"week": week, "gap": mt5.to_numpy() - other}).dropna()
        centre = frame.groupby("week")["gap"].transform("median")
        spread = (frame["gap"] - centre).abs()
        gaps[h] = spread.groupby(frame["week"]).median()
        counts[h] = frame.groupby("week").size()
    gaps, counts = pd.DataFrame(gaps), pd.DataFrame(counts).fillna(0).astype(int)

    order = np.argsort(gaps.to_numpy(), axis=1)
    shifts = np.array(list(SHIFTS))
    best = shifts[order[:, 0]]
    second = shifts[order[:, 1]]
    rows = np.arange(len(gaps))
    col = {h: i for i, h in enumerate(SHIFTS)}
    out = pd.DataFrame({
        "measured_offset_h": best,
        "gap_best_usd": gaps.to_numpy()[rows, order[:, 0]].round(4),
        "next_best_offset_h": second,
        "gap_next_usd": gaps.to_numpy()[rows, order[:, 1]].round(4),
        "matched_bars": [counts.iloc[i, col[b]] for i, b in enumerate(best)],
    }, index=gaps.index)
    out["clear"] = (out.gap_next_usd >= CLEAR_RATIO * out.gap_best_usd) & (out.matched_bars >= MIN_MATCHED)
    return out


def measure_stop_at_23h() -> pd.Series:
    """Per week: 1-min bars in server hour 23 on Tue-Thu. 0 = gold's daily stop (17:00 New York) sat at 23:00."""
    idx = pd.DatetimeIndex(pd.read_parquet(config.GOLD_M1, columns=["time"])["time"])
    idx = idx[idx.dayofweek.isin([1, 2, 3])]
    at_23 = pd.Series(idx.hour == 23, index=_week(idx))
    return at_23.groupby(level=0).sum().astype(int).rename("bars_at_23h")


def build() -> pd.DataFrame:
    prices = measure_prices()
    brk = measure_stop_at_23h()
    weeks = prices.index.union(brk.index)
    out = rule_offsets(weeks).join(prices).join(brk)
    out["clear"] = out["clear"].astype("boolean").fillna(False).astype(bool)   # no price check (2016-2018) = False
    for name in ("measured_offset_h", "next_best_offset_h", "matched_bars"):
        out[name] = out[name].astype("Int64")
    out.index.name = "week_start_server"
    out.insert(0, "us_eu_differ", out.rule_us != out.rule_eu)
    return out


def write(table: pd.DataFrame) -> None:
    tmp = config.SERVER_OFFSET_CSV.with_suffix(".csv.tmp")
    table.to_csv(tmp, date_format="%Y-%m-%d", lineterminator="\n")
    os.replace(tmp, config.SERVER_OFFSET_CSV)


def summary(table: pd.DataFrame) -> str:
    t = table[table.measured_offset_h.notna()]
    clear = t[t.clear]
    lines = [f"Server clock, measured week by week ({config.SERVER_OFFSET_CSV.relative_to(config.ROOT).as_posix()})",
             f"  price check {PRICE_YEARS[0]}-{PRICE_YEARS[1]}: {len(t)} weeks, {len(clear)} clear "
             f"(next-best hour's gap >= {CLEAR_RATIO:g}x, >= {MIN_MATCHED} bars)"]
    lines.append("  year  weeks  +2  +3  other | US rule agrees | EU rule agrees | typical gap best / next ($)")
    for y, g in clear.groupby(clear.index.year):
        vc = g.measured_offset_h.value_counts()
        other = len(g) - vc.get(2, 0) - vc.get(3, 0)
        lines.append(f"  {y}  {len(g):5d}  {vc.get(2, 0):2d}  {vc.get(3, 0):2d}  {other:5d} | "
                     f"{(g.measured_offset_h == g.rule_us).sum():8d}/{len(g):<5d} | "
                     f"{(g.measured_offset_h == g.rule_eu).sum():8d}/{len(g):<5d} | "
                     f"{g.gap_best_usd.median():.3f} / {g.gap_next_usd.median():.2f}")
    diff = t[t.us_eu_differ]
    lines.append(f"  the {len(diff)} weeks when the US and Europe differ (they decide the rule):")
    for w, r in diff.iterrows():
        lines.append(f"    week of {w:%Y-%m-%d}: measured +{int(r.measured_offset_h)}  (US rule +{r.rule_us}, "
                     f"EU rule +{r.rule_eu})  gap ${r.gap_best_usd:.3f} vs next ${r.gap_next_usd:.2f}"
                     f"{'' if r.clear else '  NOT CLEAR'}")
    unclear = t[~t.clear]
    if len(unclear):
        lines.append("  weeks that are not clear: " + ", ".join(f"{w:%Y-%m-%d}({r.matched_bars} bars)"
                                                             for w, r in unclear.iterrows()))
    lines.append("  daily stop check (gold stops 17:00 New York): server hour 23 EMPTY in [weeks when US and Europe"
                 " differ: EU rule says all, US rule says none] / [other weeks: both rules say none]:")
    empty = table.bars_at_23h == 0
    parts = []
    for y, g in table.groupby(table.index.year):
        e = empty[g.index]
        parts.append(f"{y}: {e[g.us_eu_differ].sum()}/{g.us_eu_differ.sum()} | {e[~g.us_eu_differ].sum()}/"
                     f"{(~g.us_eu_differ).sum()}")
    lines += ["    " + "   ".join(parts[i:i + 4]) for i in range(0, len(parts), 4)]
    return "\n".join(lines)


def main() -> None:
    table = build()
    write(table)
    print("\n" + summary(table))


if __name__ == "__main__":
    main()
