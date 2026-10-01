"""The 2-year Treasury yield, daily 2015-2023 = FRED series DGS2 (L12b.1: the Fed's decisions have no forecast).

    python -m machine.news.yields         # -> data/news/dgs2.csv (in git), checked against FRED's own file

Source: the U.S. Treasury's "Daily Treasury Par Yield Curve Rates", column "2 Yr", one CSV per year 2015-2023 - a
year of 2024 or later is never asked for (LEARNED #96). FRED's DGS2 is this same number (FRED takes it from the Fed's
H.15 table, which takes it from the Treasury). FRED drops this PC's scripts (L12), so FRED's own DGS2 file for
2015-2023 was fetched once through Claude's web fetch and kept as data/news/yields/fred_DGS2_2015_2023.csv; every
build compares the two. 30 Sep 2026: equal on 2,251 of 2,251 days; FRED's other 95 rows are holidays with no number.

When a day's number is known (the "cut off the future" rule): the Treasury takes the day's quotes at about 3:30 pm
New York. The machine treats the number as known only from 16:15 New York on the NEXT day that has a quote (the
Fed's daily H.15 update) - later than it really comes out, never earlier (Claude's reading, the safe side).
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request

import pandas as pd

from machine import config, guards
from machine.news.official import HEADERS

YEARS = range(2015, 2024)
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
                "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
FRED_FILE = "fred_DGS2_2015_2023.csv"
NY = "America/New_York"
QUOTE_TIME = pd.Timedelta(hours=15, minutes=30)      # the Treasury's quotes: about 3:30 pm New York
KNOWN_TIME = pd.Timedelta(hours=16, minutes=15)      # the Fed's H.15 update, the next day with a quote


def treasury_year(year: int) -> pd.DataFrame:
    """The Treasury's file for one year (downloaded once, then read from data/news/yields/)."""
    guards.check_dates([pd.Timestamp(f"{year}-12-31")])
    path = config.NEWS_YIELDS_DIR / f"treasury_par_yield_{year}.csv"
    if not path.is_file():
        with urllib.request.urlopen(urllib.request.Request(TREASURY_URL.format(year=year), headers=HEADERS),
                                    timeout=60) as r:
            raw = r.read()
        if not raw.startswith(b"Date,"):
            raise ConnectionError(f"the Treasury's {year} file does not start with 'Date,' - not saved")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
        time.sleep(2)
    t = pd.read_csv(path)
    return pd.DataFrame({"date": pd.to_datetime(t["Date"], format="%m/%d/%Y"), "dgs2": t["2 Yr"].astype(float)})


def build() -> pd.DataFrame:
    """dgs2.csv from the Treasury's files; the FRED comparison printed (and required to agree where both exist)."""
    two = pd.concat([treasury_year(y) for y in YEARS]).dropna().sort_values("date").reset_index(drop=True)
    guards.check_dates(two[["date"]])
    fred_path = config.NEWS_YIELDS_DIR / FRED_FILE
    if fred_path.is_file():
        fred = pd.read_csv(fred_path, na_values=["."])
        fred = pd.DataFrame({"date": pd.to_datetime(fred["observation_date"]), "fred": fred["DGS2"]}).dropna()
        both = two.merge(fred, on="date", how="outer")
        same = (both["dgs2"] - both["fred"]).abs() < 1e-9
        print(f"FRED DGS2 check: {int(same.sum())} of {len(both)} days equal; Treasury only "
              f"{int(both['fred'].isna().sum())}, FRED only {int(both['dgs2'].isna().sum())}")
        if not same.all():
            raise ValueError("the Treasury's 2-year yield differs from FRED's DGS2 - see the rows above; not written")
    else:
        print(f"FRED DGS2 check: skipped ({fred_path} is not here)")
    tmp = config.NEWS_DGS2.with_suffix(".tmp")
    two.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    tmp.replace(config.NEWS_DGS2)
    print(f"{config.NEWS_DGS2}: {len(two)} days {two['date'].min():%Y-%m-%d} .. {two['date'].max():%Y-%m-%d}")
    return two


def load() -> pd.DataFrame:
    """date, dgs2 (percent), quoted (the quote's moment, UTC), known_from (UTC; NaT for the last day)."""
    two = pd.read_csv(config.NEWS_DGS2, parse_dates=["date"])
    guards.check_dates(two[["date"]])
    return with_moments(two)


def with_moments(two: pd.DataFrame) -> pd.DataFrame:
    two = two.sort_values("date").reset_index(drop=True)
    quoted = (two["date"] + QUOTE_TIME).dt.tz_localize(NY).dt.tz_convert("UTC")
    known = (two["date"].shift(-1) + KNOWN_TIME).dt.tz_localize(NY).dt.tz_convert("UTC")
    return two.assign(quoted=quoted, known_from=known)


def known_at(two: pd.DataFrame, t) -> pd.DataFrame:
    """The yields as the world knew them at moment t (UTC, or naive = server time like every bar)."""
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        from machine.data import clock
        t = clock.server_to_utc(pd.DatetimeIndex([t]))[0]
    return two[two["known_from"] <= t.tz_convert("UTC")]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argparse.ArgumentParser(prog="python -m machine.news.yields", description=__doc__.split("\n\n")[0]
                            ).parse_args(argv)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
