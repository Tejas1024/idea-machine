"""Build data/news/events.parquet - THE news calendar (L12).   Run:  python -m machine.news.build

Input: data/news/raw/ (python -m machine.news.mt5_export) and data/news/official/ (machine.news.official; fetched
on the first run). The 2024+ file in data/news/sealed/ is never opened here (LEARNED #96).

THE CLOCK TRAP (measured 29 Sep 2026, printed by every build): MT5's calendar gives every old release in server time
converted with the offset of the day it was EXPORTED (+3 h on 29 Sep 2026, EU summer time), not the offset of the
release's own date. So
    utc = MT5 calendar time - (server_minus_gmt_seconds of the export, in its info file)
and the machine's `server` column is then that UTC moment on the price data's measured clock (machine/data/clock.py).
Read with the release date's own offset instead, a winter NFP lands at 9:30 New York - one hour late. The proof,
printed below: with the export's offset 89 of 142 red / orange event kinds sit at ONE New York time each (every
NFP 8:30); with each date's own offset, 3 of 142. The build refuses to write if any NFP is not at 8:30 New York.

Rows: US events MT5 rates high or moderate (= Forex Factory red + orange stand-in, T7), 2015-2023.
Columns:
  utc, server, ist         the release moment (NaT for an all-day event such as Jackson Hole); server = price clock
  day                      the day (server clock; an all-day event: its own date)
  event, event_id, kind    MT5's name and id; kind = Claude's grouping (calendar.KINDS)
  importance, mt5_importance   "high" (3) / "moderate" (2)
  actual, forecast, previous, revised_previous   MT5's numbers (the first-released actual; unit + multiplier below)
  unit, multiplier, period, revision, value_id    as MT5 has them
  has_actual, has_forecast, no_forecast_why       why a forecast is missing: "none for this kind" (FOMC, speeches,
                           auctions ...) or "MT5 has none in 2016 to mid-2017" (a gap in MT5's history)
  exact_time               False for an all-day event
  scheduled                False for the Fed's two emergency meetings (3 and 15 Mar 2020): not known in advance
  source                   where the row comes from (MT5 for every row: it has every actual number 2016-2023)
  checked                  the official check of this row, if one was made (BLS / Fed / BEA)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, guards
from machine.data import clock
from machine.news import calendar, official
from machine.news.mt5_export import INFO_FILE, OPEN_FILE

NY = "America/New_York"
UNITS = {0: "", 1: "percent", 2: "currency", 3: "hours", 4: "jobs", 5: "rigs", 6: "USD", 7: "people",
         8: "mortgage", 9: "votes", 10: "barrels", 11: "cubic feet", 12: "positions", 13: "buildings"}
MULTIPLIERS = {0: "", 1: "thousands", 2: "millions", 3: "billions", 4: "trillions"}
IMPORTANCE = {3: "high", 2: "moderate"}
FORECAST_GAP_BEFORE = pd.Timestamp("2017-07-01")   # MT5 has no forecasts before about mid-2017 (measured below)
KIND_HAS_FORECAST = 0.5                            # a kind "has forecasts" when MT5 has one on half its rows or more


def read_raw(path: Path | None = None) -> tuple[pd.DataFrame, pd.Timedelta, dict]:
    """The MT5 export (2015-2023 only) and the offset its times carry."""
    path = path or config.NEWS_RAW_DIR / OPEN_FILE
    if config.NEWS_SEALED_DIR.resolve() in path.resolve().parents:
        raise guards.SealedYearsError(f"{path} is in the sealed folder. {guards.SEALED_WORDS}.")
    info = dict(line.split("=", 1) for line in
                (path.parent / INFO_FILE).read_text(encoding="utf-8").splitlines() if "=" in line)
    seconds = int(info["server_minus_gmt_seconds"])
    offset = pd.Timedelta(minutes=15 * round(seconds / 900))          # TimeTradeServer - TimeGMT, to 15 minutes
    raw = pd.read_csv(path, sep="\t", dtype={"event_code": str, "source_url": str}, keep_default_na=False,
                      na_values={c: [""] for c in ("actual_x1e6", "forecast_x1e6", "prev_x1e6", "revised_prev_x1e6")})
    raw["mt5_time"] = pd.to_datetime(raw["time_server"], format="%Y.%m.%d %H:%M:%S")
    guards.check_dates(raw["mt5_time"])
    return raw, offset, info


def mt5_to_utc(mt5_time: pd.Series, offset: pd.Timedelta) -> pd.Series:
    """MT5 calendar time -> UTC: minus the offset of the EXPORT day (the clock trap, top of this file)."""
    return (mt5_time - offset).dt.tz_localize("UTC")


def to_events(raw: pd.DataFrame, offset: pd.Timedelta, info: dict) -> pd.DataFrame:
    d = raw[raw["importance"].isin(IMPORTANCE)].copy()
    exact = d["time_mode"].eq(0)
    utc = mt5_to_utc(d["mt5_time"], offset).where(exact)
    out = pd.DataFrame({
        "utc": utc,
        "server": clock.utc_to_server(utc.dropna()).reindex(d.index),
        "ist": utc.dt.tz_convert(clock.IST_ZONE),
    })
    out["day"] = out["server"].dt.normalize().fillna(d["mt5_time"].dt.normalize())
    out["event"] = d["event_name"]
    out["event_id"] = d["event_id"].astype("int64")
    out["kind"] = d["event_name"].map(calendar.event_kind)
    out["importance"] = d["importance"].map(IMPORTANCE)
    out["mt5_importance"] = d["importance"].astype("int8")
    for new, old in (("actual", "actual_x1e6"), ("forecast", "forecast_x1e6"), ("previous", "prev_x1e6"),
                     ("revised_previous", "revised_prev_x1e6")):
        out[new] = pd.to_numeric(d[old], errors="coerce") / 1e6
    out["unit"] = d["unit"].map(UNITS)
    out["multiplier"] = d["multiplier"].map(MULTIPLIERS)
    out["period"] = pd.to_datetime(d["period"], format="%Y.%m.%d %H:%M:%S")
    out["revision"] = d["revision"].astype("int16")
    out["value_id"] = d["value_id"].astype("int64")
    out["has_actual"] = out["actual"].notna()
    out["has_forecast"] = out["forecast"].notna()
    after_gap = out["day"] >= FORECAST_GAP_BEFORE
    kind_has = out[after_gap].groupby("event")["has_forecast"].mean() >= KIND_HAS_FORECAST
    usual = out["event"].map(kind_has).fillna(True).astype(bool)   # a kind seen only in the gap: can't tell -> gap
    out["no_forecast_why"] = np.select(
        [out["has_forecast"], ~usual, out["day"] < FORECAST_GAP_BEFORE],
        ["", "none for this kind", "MT5 has none in 2016 to mid-2017"], "MT5 has none for this release")
    out["exact_time"] = exact
    out["scheduled"] = True
    exported = info.get("exported_at_gmt", "")[:10].replace(".", "-")
    out["source"] = f"MT5 calendar ({info.get('server', '?')}, build {info.get('terminal_build', '?')}, exported " \
                    f"{exported})"
    out["checked"] = ""
    out = out.sort_values(["day", "server", "mt5_importance"], ascending=[True, True, False], kind="stable")
    return out.reset_index(drop=True)


# ---- the checks ----------------------------------------------------------------------------------------------------

def clock_proof(raw: pd.DataFrame, offset: pd.Timedelta) -> tuple[int, int, int]:
    """How many red / orange event kinds (2016-2023, exact times) sit at ONE New York time: with the export's
    offset, and with each date's own offset (the price data's clock). Returns (kinds, one_time_export, one_time_own)."""
    d = raw[raw["importance"].isin(IMPORTANCE) & raw["time_mode"].eq(0) & (raw["mt5_time"] >= "2016-01-01")]
    ny_export = mt5_to_utc(d["mt5_time"], offset).dt.tz_convert(NY).dt.strftime("%H:%M")
    fine = ~((d["mt5_time"].dt.dayofweek == 6) & (d["mt5_time"].dt.hour == 3))   # not inside the clock-change hour
    own = clock.server_to_utc(d.loc[fine, "mt5_time"])
    ny_own = own.dt.tz_convert(NY).dt.strftime("%H:%M")
    one_export = ny_export.groupby(d["event_name"]).nunique().eq(1).sum()
    one_own = ny_own.groupby(d.loc[fine, "event_name"]).nunique().eq(1).sum()
    return d["event_name"].nunique(), int(one_export), int(one_own)


def official_check(ev: pd.DataFrame, releases: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per official release 2016-2023 compared with MT5's row(s) that day: day, New York time, number."""
    ev = ev[ev["exact_time"] & (ev["day"] >= "2016-01-01")].copy()
    ev["ny"] = ev["utc"].dt.tz_convert(NY)
    ev["ny_day"] = ev["ny"].dt.date
    rows = []

    def compare(rel: pd.DataFrame, agency: str, event: str, number: str | None = None, time_col: str = "official_ny"):
        mine = ev[ev["event"] == event].set_index("ny_day")
        for _, r in rel.iterrows():
            m = mine.loc[[r["day"]]] if r["day"] in mine.index else None
            row = {"agency": agency, "event": event, "day": r["day"], "in_mt5": m is not None,
                   "official_ny": r.get(time_col, pd.NaT), "mt5_ny": m["ny"].iloc[0] if m is not None else pd.NaT,
                   "note": r.get("note", "") if isinstance(r.get("note", ""), str) else ""}
            # None = not compared (no official time / number, or no MT5 row); True / False = compared
            row["time_same"] = (row["official_ny"] == row["mt5_ny"]) if (
                m is not None and pd.notna(row["official_ny"])) else None
            if number:
                row["official_number"] = r.get(number, np.nan)
                row["mt5_number"] = m["actual"].iloc[0] if m is not None else np.nan
                rounding = r.get(f"{number}_rounding", 1e-9)     # the agency's headline rounding, if it rounds
                row["number_same"] = bool(abs(row["official_number"] - row["mt5_number"]) <= rounding) if (
                    pd.notna(row["official_number"]) and pd.notna(row["mt5_number"])) else None
            rows.append(row)
        official_days = set(rel["day"])
        for day in sorted(set(mine.index) - official_days):
            rows.append({"agency": agency, "event": event, "day": day, "in_mt5": True, "official_ny": pd.NaT,
                         "mt5_ny": mine.loc[[day], "ny"].iloc[0], "time_same": None, "in_official": False})

    compare(releases["empsit"], "BLS", "Nonfarm Payrolls", "nfp_thousands")
    compare(releases["empsit"], "BLS", "Unemployment Rate", "unemployment_rate")
    compare(releases["cpi"], "BLS", "CPI m/m", "cpi_mm")
    fomc = releases["fomc"]
    decisions = fomc[fomc["title"].str.contains("FOMC statement", case=False)]
    compare(decisions, "Fed", "Fed Interest Rate Decision")
    compare(decisions, "Fed", "FOMC Statement")
    gdp = releases["gdp"].copy()
    mt5_gdp = set(ev.loc[ev["event"] == "GDP q/q", "ny"])
    for i, r in gdp.iterrows():                 # BEA's list against MT5; where they differ, the release's own words
        if r["official_ny"] not in mt5_gdp:
            own = official.bea_release_time(r["link"])
            if pd.notna(own):
                gdp.loc[i, ["day", "official_ny", "note"]] = [own.date(), own, f"BEA's list said "
                                                              f"{r['official_ny']:%Y-%m-%d %H:%M}; the release says "
                                                              f"{own:%Y-%m-%d %H:%M}"]
    compare(gdp, "BEA", "GDP q/q")
    out = pd.DataFrame(rows)
    out["in_official"] = out.get("in_official", pd.Series(True, index=out.index)).fillna(True).astype(bool)
    return out


def _is(value, want: bool) -> bool:
    """True when value is a real True / False (not None, not NaN) equal to want."""
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and bool(value) is want


def mark_checked(ev: pd.DataFrame, check: pd.DataFrame, releases: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ev = ev.copy()
    ny_day = ev["utc"].dt.tz_convert(NY).dt.date
    for _, c in check[check["in_mt5"] & check["in_official"]].iterrows():
        rows = (ev["event"] == c["event"]) & (ny_day == c["day"])
        parts = ["day"]
        if _is(c.get("time_same"), True):
            parts.append(f"{c['official_ny']:%H:%M} New York")
        if _is(c.get("number_same"), True):
            parts.append("first-released number")
        bad = _is(c.get("time_same"), False) or _is(c.get("number_same"), False)
        ev.loc[rows, "checked"] = (f"{c['agency']}: DIFFERS - see official_check.csv" if bad
                                   else f"{c['agency']}: same {', '.join(parts)}")
    # The Fed marks its emergency meetings "(unscheduled)": their releases were not known in advance.
    fomc = releases["fomc"]
    surprise = set(fomc.loc[fomc["unscheduled"] & fomc["title"].str.contains("FOMC statement", case=False), "day"])
    fed_day = ev["event"].isin(["Fed Interest Rate Decision", "FOMC Statement", "FOMC Press Conference"])
    ev.loc[fed_day & ny_day.isin(surprise), "scheduled"] = False
    return ev


def add_missing_fed(ev: pd.DataFrame, releases: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """An FOMC statement on the Fed's pages with no Fed row in MT5 that day gets a row, source = the Fed.
    2016-2023: one - 23 Mar 2020, 8:00 New York, a "(notation vote)" statement (unlimited bond buying; the rate
    unchanged), not known in advance. Same name and rating as MT5's other "FOMC Statement" rows."""
    fomc = releases["fomc"]
    statements = fomc[fomc["title"].str.contains("FOMC statement", case=False) & fomc["official_ny"].notna()]
    fed_rows = ev["event"].isin(["Fed Interest Rate Decision", "FOMC Statement"]) & ev["exact_time"]
    have = set(ev.loc[fed_rows, "utc"].dt.tz_convert(NY).dt.date)
    like = ev[ev["event"] == "FOMC Statement"].iloc[0]
    add = []
    for _, s in statements[~statements["day"].isin(have)].iterrows():
        utc = s["official_ny"].tz_convert("UTC")
        row = like.copy()
        row[["utc", "server", "ist"]] = [utc, clock.utc_to_server(utc), utc.tz_convert(clock.IST_ZONE)]
        row["day"] = row["server"].normalize()
        row[["actual", "forecast", "previous", "revised_previous"]] = np.nan
        row[["event_id", "value_id", "revision"]] = 0
        row["period"] = pd.NaT
        row[["has_actual", "has_forecast"]] = False
        row["scheduled"] = not bool(s["unscheduled"])
        row["source"] = f"Federal Reserve statement page {s['day']:%Y-%m-%d} (MT5 has no row)"
        row["checked"] = f"Fed: {s['official_ny']:%H:%M} New York from the Fed's page"
        add.append(row)
    if not add:
        return ev
    out = pd.concat([ev, pd.DataFrame(add).astype(ev.dtypes.to_dict())], ignore_index=True)
    return out.sort_values(["day", "server", "mt5_importance"], ascending=[True, True, False],
                           kind="stable").reset_index(drop=True)


def price_check(ev: pd.DataFrame, events: tuple[str, ...], window: int = 30) -> pd.DataFrame:
    """Gold's own check of the clock: for each release, the minute (from -window to +window around the calendar's
    server minute) where gold's 1-minute candle is tallest. 0 = the release minute. Uses 2016-2023 prices only."""
    from machine.data.bars import load_bars

    m1 = load_bars("1m", "2016-01-01", "2023-12-29")
    rng = (m1["high"] - m1["low"]).to_numpy()
    at = m1.index.to_numpy(dtype="datetime64[ns]")
    rows = []
    for _, r in ev[ev["event"].isin(events) & ev["exact_time"] & (ev["day"] >= "2016-01-01")].iterrows():
        t0 = np.datetime64(r["server"], "ns")
        lo, hi = np.searchsorted(at, [t0 - np.timedelta64(window, "m"), t0 + np.timedelta64(window + 1, "m")])
        if hi - lo < window:
            continue
        k = lo + int(np.argmax(rng[lo:hi]))
        rows.append({"event": r["event"], "server": r["server"],
                     "tallest_minute": int((at[k] - t0) / np.timedelta64(1, "m")), "range_usd": float(rng[k])})
    return pd.DataFrame(rows)


def nfp_times(ev: pd.DataFrame) -> pd.DataFrame:
    """NFP's India time in January and July, year by year (the prompt expects about 19:00 and 18:00)."""
    nfp = ev[(ev["event"] == "Nonfarm Payrolls") & (ev["day"] >= "2016-01-01")]
    rows = []
    for year in range(2016, 2024):
        row = {"year": year}
        for month, name in ((1, "January"), (7, "July")):
            hit = nfp[(nfp["ist"].dt.year == year) & (nfp["ist"].dt.month == month)]
            row[f"{name} (IST)"] = f"{hit['ist'].iloc[0]:%d %b %H:%M}" if len(hit) else "-"
            row[f"{name} (server)"] = f"{hit['server'].iloc[0]:%H:%M}" if len(hit) else "-"
        rows.append(row)
    return pd.DataFrame(rows)


# ---- the build ----------------------------------------------------------------------------------------------------

def build(fetch_official: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    raw, offset, info = read_raw()
    print(f"MT5 export: {info['server']}, build {info['terminal_build']}, exported {info['exported_at_gmt']} GMT; "
          f"its times carry +{offset / pd.Timedelta(hours=1):g} h (the export day's offset)")
    kinds, one_export, one_own = clock_proof(raw, offset)
    print(f"the clock trap: {one_export} of {kinds} red / orange event kinds sit at ONE New York time with the "
          f"export's offset; {one_own} of {kinds} with each date's own offset")
    ev = to_events(raw, offset, info)
    nfp = ev[(ev["event"] == "Nonfarm Payrolls") & (ev["day"] >= "2016-01-01")]
    ny = nfp["utc"].dt.tz_convert(NY).dt.strftime("%H:%M")
    if not (ny == "08:30").all():
        raise RuntimeError(f"NFP not at 8:30 New York: {nfp.loc[ny != '08:30', 'server'].tolist()} - the clock "
                           f"rule is wrong; nothing written.")
    guards.check_dates(ev[["utc", "server", "ist", "day"]])

    check = None
    if fetch_official:
        releases = official.all_releases()
        check = official_check(ev, releases)
        ev = mark_checked(ev, check, releases)
        ev = add_missing_fed(ev, releases)
        config.NEWS_OFFICIAL_CHECK.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.NEWS_OFFICIAL_CHECK.with_suffix(".tmp")
        check.to_csv(tmp, index=False)
        tmp.replace(config.NEWS_OFFICIAL_CHECK)

    tmp = config.NEWS_EVENTS.with_suffix(".tmp")
    ev.to_parquet(tmp, index=False)
    tmp.replace(config.NEWS_EVENTS)
    return ev, check


def report(ev: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    y = ev["day"].dt.year
    in_years = ev[(y >= 2016) & (y <= 2023)]
    print(f"\n{config.NEWS_EVENTS.relative_to(config.ROOT)}: {len(ev):,} rows ({ev['day'].min():%Y-%m-%d} to "
          f"{ev['day'].max():%Y-%m-%d}); 2016-2023: {len(in_years):,} rows, {in_years['event'].nunique()} event kinds")
    print(f"content fingerprint (calendar.fingerprint): {calendar.fingerprint(ev)}")
    print("\nsources:")
    print(in_years.groupby(["source"]).size().to_string())
    print("\nofficial check (rows marked):")
    print(in_years["checked"].replace("", "(none)").value_counts().to_string())
    print("\nrows by kind and importance, 2016-2023:")
    print(pd.crosstab(in_years["kind"], in_years["importance"], margins=True).to_string())
    print("\nforecast, 2016-2023 (share of rows that have one, among kinds that usually do):")
    usual = in_years[in_years["no_forecast_why"] != "none for this kind"]
    print((usual.groupby(usual["day"].dt.year)["has_forecast"].mean() * 100).round(1).to_string())
    print("\nno forecast, why:")
    print(in_years["no_forecast_why"].replace("", "(has one)").value_counts().to_string())
    print("\nnot scheduled (not known in advance):")
    print(in_years.loc[~in_years["scheduled"], ["ist", "event"]].to_string(index=False))
    nfp = in_years[in_years["event"] == "Nonfarm Payrolls"]
    print(f"\nNFP rows 2016-2023: {len(nfp)} - by year {nfp.groupby(nfp['day'].dt.year).size().to_dict()}")
    print("NFP time, January vs July:")
    print(nfp_times(ev).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-official", action="store_true", help="skip the BLS / Fed / BEA check")
    ap.add_argument("--no-prices", action="store_true", help="skip gold's own check of the clock")
    a = ap.parse_args()
    ev, check = build(fetch_official=not a.no_official)
    report(ev)
    if check is not None:
        print("\nMT5 against the agencies (2016-2023):")
        for (agency, event), g in check.groupby(["agency", "event"], sort=False):
            both = g[g["in_mt5"] & g["in_official"]]
            line = (f"  {agency:3s} {event:28s} official {int(g['in_official'].sum()):3d} · in MT5 "
                    f"{int(g['in_mt5'].sum()):3d} · both {len(both):3d}")
            for col, words in (("time_same", "same New York time"), ("number_same", "same first number")):
                if col in both and both[col].map(lambda v: _is(v, True) or _is(v, False)).any():
                    line += f" · {words} {int(both[col].map(lambda v: _is(v, True)).sum())} of {len(both)}"
            print(line)
        differ = pd.Series(False, index=check.index)
        for col in ("time_same", "number_same"):
            if col in check:
                differ |= check[col].map(lambda v: _is(v, False))
        odd = check[differ | ~check["in_mt5"] | ~check["in_official"]]
        if len(odd):
            print("\nthe rows that differ:")
            print(odd.drop(columns=[c for c in ("time_same", "number_same") if c in odd]).to_string(index=False))
    if not a.no_prices:
        _, offset, _ = read_raw()
        names = ("Nonfarm Payrolls", "CPI m/m", "Fed Interest Rate Decision")
        # the trap: MT5's calendar time read as if it were the price data's server time
        trap = ev.assign(server=(ev["utc"] + offset).dt.tz_localize(None))
        print("\ngold's own check of the clock: where is gold's tallest 1-minute candle within 30 minutes of the "
              "release? (0 or 1 = at the release minute or the next)")
        for label, table in (("this table", ev), ("the trap reading", trap)):
            print(f" {label}:")
            for event, g in price_check(table, names).groupby("event"):
                at0 = g["tallest_minute"].between(0, 1)
                print(f"  {event:28s} {at0.sum():3d} of {len(g)} at the release minute; "
                      f"others at {sorted(g.loc[~at0, 'tallest_minute'].tolist())}")


if __name__ == "__main__":
    main()
