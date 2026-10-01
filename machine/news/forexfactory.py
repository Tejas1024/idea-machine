"""Forex Factory's US calendar 2015-2023: the FORECAST (the economists' consensus) for the surprise (L12b.1).

    python -m machine.news.forexfactory          # fetches the weeks not fetched yet (about 31 minutes the first time)

Why this exists (found 30 Sep 2026): MT5's `forecast` column is NOT the economists' consensus. For the jobs report of
8 May 2020 MT5 says +139K jobs were expected; everyone expected about -22 million (Forex Factory: -22000K). In Sep -
Nov 2020 MT5's "forecasts" were -74,433K, +78,356K and -76,324K jobs. A surprise measured against those numbers is
not a surprise. Forex Factory is also Tejas's own calendar (his news = its red and orange, T7), and each of its rows
carries his colour, the forecast and Forex Factory's own mark "better / worse" for the dollar.

What is kept: only US (USD) rows, only the fields below, one JSON per week in data/news/forexfactory/ (not in git).
A week is Sunday to Saturday on the viewer's clock; `dateline` is the release moment in Unix seconds (UTC), so the
clock of the page does not matter. The last week asked for is 24-30 Dec 2023: no week reaching 2024 is ever
requested, and a row dated 2024 or later would be dropped and counted, never saved (LEARNED #96).
Pages are fetched with Git's curl under our own name (USER_AGENT, no browser disguise), DELAY_SECONDS apart, once
each (a saved week is never fetched again); Forex Factory's robots.txt sets no limit (it lists only a sitemap).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, guards
from machine.news.official import HEADERS

WEEK_URL = "https://www.forexfactory.com/calendar?week={mon}{day}.{year}"
FIRST_SUNDAY = dt.date(2014, 12, 28)       # its week holds 1-3 Jan 2015 (2015 = warm-up for the rolling size)
LAST_SUNDAY = dt.date(2023, 12, 24)        # 24-30 Dec 2023; the next week reaches 2024 and is never asked for
SEALED_FROM = pd.Timestamp("2024-01-01", tz="UTC")
DELAY_SECONDS = 4.0
USER_AGENT = "ML-Bot research (personal, one-time history fetch)"
GIT_CURL = (Path(r"C:\Program Files\Git\mingw64\bin\curl.exe"), Path(r"C:\Program Files\Git\usr\bin\curl.exe"))
KEEP = ("id", "ebaseId", "name", "prefixedName", "dateline", "impactName", "impactClass", "timeLabel", "timeMasked",
        "actual", "forecast", "previous", "revision", "actualBetterWorse", "revisionBetterWorse", "notice", "date")
COLOUR = {"icon--ff-impact-red": "red", "icon--ff-impact-ora": "orange", "icon--ff-impact-yel": "yellow",
          "icon--ff-impact-gra": "grey"}
SUFFIX = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_NUMBER = re.compile(r"^\s*([<>]?)\s*(-?\d+(?:\.\d+)?)\s*([KMBT%]?)\s*$")

_last_fetch = [0.0]


def sundays(first: dt.date = FIRST_SUNDAY, last: dt.date = LAST_SUNDAY) -> list[dt.date]:
    if last + dt.timedelta(days=6) >= dt.date(2024, 1, 1):
        raise guards.SealedYearsError(f"the week of {last} reaches 2024. {guards.SEALED_WORDS}.")
    out, day = [], first
    while day <= last:
        out.append(day)
        day += dt.timedelta(days=7)
    return out


def week_file(sunday: dt.date):
    return config.NEWS_FF_DIR / f"{sunday.isoformat()}.json"


def parse_page(html: str) -> list[dict]:
    """The week's rows from the page: the line `days: [...]` of window.calendarComponentStates."""
    line = next((ln for ln in html.splitlines() if ln.lstrip().startswith("days: [")), None)
    if line is None:
        raise ConnectionError("the page has no calendar in it (blocked, or Forex Factory changed its page)")
    days = json.loads(line.lstrip()[len("days: "):].rstrip().rstrip(","))
    return [e for d in days for e in d["events"]]


def fetch_week(sunday: dt.date) -> tuple[list[dict], int]:
    """(the week's US rows, how many rows dated 2024+ were dropped). Saved to data/news/forexfactory/."""
    path = week_file(sunday)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), 0
    guards.check_dates([pd.Timestamp(sunday + dt.timedelta(days=6))])
    url = WEEK_URL.format(mon=sunday.strftime("%b").lower(), day=sunday.day, year=sunday.year)
    wait = DELAY_SECONDS - (time.monotonic() - _last_fetch[0])
    if wait > 0:
        time.sleep(wait)
    html = _get(url)
    _last_fetch[0] = time.monotonic()
    rows = [{k: e.get(k) for k in KEEP} for e in parse_page(html) if e.get("prefixedName", "").startswith("US ")]
    sealed = [e for e in rows if pd.Timestamp(e["dateline"], unit="s", tz="UTC") >= SEALED_FROM]
    rows = [e for e in rows if e not in sealed]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(path)
    return rows, len(sealed)


def _get(url: str) -> str:
    """The page through Git's curl, which says who it is (USER_AGENT) - no browser disguise. Forex Factory's firewall
    refused Python's own web client and Windows' curl.exe (403 Forbidden, 30 Sep 2026) and served Git for Windows'
    curl with this name. Its robots.txt allows the calendar."""
    curl = next((str(p) for p in GIT_CURL if p.is_file()), None) or shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("no curl on this PC (Git for Windows brings one)")
    args = [curl, "-sS", "--fail", "-m", "60", "--compressed", "-A", USER_AGENT,
            "-H", "Accept: text/html", "-H", f"Accept-Language: {HEADERS['Accept-Language']}"]
    done = subprocess.run(args + [url], capture_output=True, timeout=90)
    if done.returncode:
        raise ConnectionError(done.stderr.decode("utf-8", "replace").strip()[:200])
    return done.stdout.decode("utf-8", errors="replace")


def number(text: str | None) -> tuple[float, str]:
    """'-22000K' -> (-22000000.0, 'K'); '0.3%' -> (0.3, '%'); '<0.25%' -> (0.25, '%'); '' -> (nan, '')."""
    m = _NUMBER.match(text or "")
    if not m:
        return np.nan, ""
    value = float(m.group(2))
    unit = m.group(3)
    return (value if unit == "%" else value * SUFFIX[unit]), unit


def half_step(text: str | None) -> float:
    """Half of the last digit Forex Factory shows, in the number's own units: '-19.4' -> 0.05, '0.2M' -> 50,000,
    '225K' -> 500. Another source's number agrees with Forex Factory's when it is within this (rounding only)."""
    m = _NUMBER.match(text or "")
    if not m:
        return np.nan
    decimals = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
    return 0.5 * 10.0 ** -decimals * (1.0 if m.group(3) == "%" else SUFFIX[m.group(3)])


def load() -> pd.DataFrame:
    """Every saved US row 2015-2023, one row each: utc, name, colour, the numbers as text and as values."""
    files = sorted(config.NEWS_FF_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{config.NEWS_FF_DIR} is empty - run `python -m machine.news.forexfactory`")
    rows = [e for f in files for e in json.loads(f.read_text(encoding="utf-8"))]
    ff = pd.DataFrame(rows).drop_duplicates("id")
    ff["utc"] = pd.to_datetime(ff["dateline"], unit="s", utc=True)
    guards.check_dates(ff[["utc"]])
    ff["name"] = ff["prefixedName"].str.removeprefix("US ")
    ff["colour"] = ff["impactClass"].map(COLOUR).fillna("other")
    for col in ("actual", "forecast", "previous", "revision"):
        parsed = ff[col].map(number)
        ff[f"{col}_value"] = parsed.map(lambda p: p[0])
        ff[f"{col}_unit"] = parsed.map(lambda p: p[1])
    ff["better_worse"] = ff["actualBetterWorse"].map({1: "better", 2: "worse"}).fillna("same or none")
    return ff.sort_values(["utc", "name"]).reset_index(drop=True)


def as_known_at(ff: pd.DataFrame, t) -> pd.DataFrame:
    """The rows as the world knew them at moment t (UTC, or naive = server time): the forecast and previous number
    are known before a release; the actual number only from its minute on."""
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        from machine.data import clock
        t = clock.server_to_utc(pd.DatetimeIndex([t]))[0]
    out = ff.copy()
    later = out["utc"] > t.tz_convert("UTC")
    out.loc[later, "actual"] = None
    out.loc[later, "actual_value"] = np.nan
    out.loc[later, "better_worse"] = "same or none"
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argparse.ArgumentParser(prog="python -m machine.news.forexfactory", description=__doc__.split("\n\n")[0]
                            ).parse_args(argv)
    weeks = sundays()
    todo = [s for s in weeks if not week_file(s).is_file()]
    print(f"{len(weeks)} weeks {weeks[0]} .. {weeks[-1] + dt.timedelta(days=6)}; {len(todo)} to fetch "
          f"(~{len(todo) * DELAY_SECONDS / 60:.0f} min)", flush=True)
    dropped = 0
    for i, sunday in enumerate(todo, 1):
        for attempt in range(3):
            try:
                rows, sealed = fetch_week(sunday)
                break
            except Exception as err:                       # a dropped connection: wait and try again, then stop
                print(f"  {sunday}: {type(err).__name__}: {err}", flush=True)
                if attempt == 2:
                    print("STOPPED - run again later; the weeks already saved are kept.", flush=True)
                    return 1
                time.sleep(30 * (attempt + 1))
        dropped += sealed
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} weeks - {sunday}: {len(rows)} US rows", flush=True)
    ff = load()
    print(f"saved: {len(ff)} US rows {ff['utc'].min():%Y-%m-%d} .. {ff['utc'].max():%Y-%m-%d}; "
          f"colours {ff['colour'].value_counts().to_dict()}; rows dated 2024+ dropped: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
