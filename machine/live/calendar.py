"""This week's news for the live alerts (L17a): Forex Factory's calendar page - his calendar (red + orange, T7), US only.

    from machine.live import calendar as LC
    rows, fetched = LC.week(now)            # this week's US rows: saved, or fetched when older than an hour
    ev = LC.events(rows)                    # his news: one row per number, with its report, sign and times
    for r in LC.releases(ev): ...           # one release per minute and report, with the report's headline number

The page is the one machine/news/forexfactory.py reads for 2015-2023 (parse_page), fetched with Git's curl under
config.LIVE_USER_AGENT (Forex Factory's firewall serves that curl; it refused Python's own client, 30 Sep 2026).
Its numbers are 2024+ news NUMBERS: read only by the live alerts, under T14 use 2 - guards.check_dates(...,
unlock=True, use="news-values"), one line in data/sealed_access.log per week per program run - and saved in
config.LIVE_DIR/ff/, never in config.NEWS_FF_DIR (the 2015-2023 history that tests and research read). No gold price
is read here.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from machine import config
from machine.data import clock
from machine.live import opened
from machine.news import forexfactory as FF
from machine.news import surprise as NS

WHO = "machine.live (L17a, the live news alerts)"
WHY = "this week's Forex Factory news numbers, read only for the live news alerts (T14 use 2)"


def sunday_of(now: dt.datetime) -> dt.date:
    """The Sunday that starts the Forex Factory week holding `now` (an aware time; the week's page by its Sunday, as
    forexfactory.py asks for them). US news never comes on a weekend, so the edge of the week needs no care."""
    d = now.astimezone(dt.timezone.utc).date()
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def week_file(sunday: dt.date, folder: Path | None = None) -> Path:
    return Path(folder or config.LIVE_DIR) / "ff" / f"{sunday.isoformat()}.json"


def _open(rows: list[dict], sunday: dt.date) -> list[dict]:
    """T14 use 2: a week with 2024+ rows is opened and logged once (machine/live/opened.py); every later read checks
    that the T14 line still says yes - a "no" there stops the alerts."""
    times = pd.DatetimeIndex([pd.Timestamp(r["dateline"], unit="s", tz="UTC") for r in rows if r.get("dateline")])
    opened.week(times, sunday, use="news-values", who=WHO, why=WHY)
    return rows


def fetch(sunday: dt.date, *, get: Callable[[str], str] | None = None) -> list[dict]:
    """This week's US rows from the page (the fields forexfactory.KEEP keeps)."""
    url = FF.WEEK_URL.format(mon=sunday.strftime("%b").lower(), day=sunday.day, year=sunday.year)
    html = (get or _get)(url)
    rows = [{k: e.get(k) for k in FF.KEEP} for e in FF.parse_page(html) if e.get("prefixedName", "").startswith("US ")]
    return _open(rows, sunday)


def _get(url: str) -> str:
    curl = next((str(p) for p in FF.GIT_CURL if p.is_file()), None) or shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("no curl on this PC (Git for Windows brings one)")
    args = [curl, "-sS", "--fail", "-m", "45", "--compressed", "-A", config.LIVE_USER_AGENT,
            "-H", "Accept: text/html", "-H", "Accept-Language: en-US,en;q=0.9"]
    done = subprocess.run(args + [url], capture_output=True, timeout=60)
    if done.returncode:
        raise ConnectionError(done.stderr.decode("utf-8", "replace").strip()[:200] or f"curl exit {done.returncode}")
    return done.stdout.decode("utf-8", errors="replace")


def save(sunday: dt.date, rows: list[dict], fetched: dt.datetime, folder: Path | None = None) -> Path:
    path = week_file(sunday, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fetched": fetched.isoformat(timespec="seconds"), "rows": rows}, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def saved(sunday: dt.date, folder: Path | None = None) -> tuple[list[dict], dt.datetime | None]:
    path = week_file(sunday, folder)
    if not path.is_file():
        return [], None
    d = json.loads(path.read_text(encoding="utf-8"))
    return _open(d["rows"], sunday), dt.datetime.fromisoformat(d["fetched"])


def week(now: dt.datetime, *, max_age: dt.timedelta | None = None, folder: Path | None = None,
         get: Callable[[str], str] | None = None) -> tuple[list[dict], dt.datetime]:
    """This week's rows: the saved copy while younger than max_age (default LIVE_WEEK_REFRESH_MINUTES), else the page
    again. A failed fetch keeps the saved copy (and says how old it is); with no copy at all it raises."""
    sunday = sunday_of(now)
    rows, fetched = saved(sunday, folder)
    age = max_age if max_age is not None else dt.timedelta(minutes=config.LIVE_WEEK_REFRESH_MINUTES)
    if fetched is not None and now - fetched < age:
        return rows, fetched
    try:
        rows = fetch(sunday, get=get)
    except (ConnectionError, OSError, ValueError, subprocess.TimeoutExpired):
        if fetched is None:
            raise
        return rows, fetched
    save(sunday, rows, now, folder)
    return rows, now


# ======================================================================================================== his news

def _colour(impact_class: Any) -> str:
    return FF.COLOUR.get(str(impact_class or ""), "")


def ff_names(signs: pd.DataFrame, table: pd.DataFrame | None = None) -> dict[str, Any]:
    """{a Forex Factory name: its row in signs.csv}. signs.csv's ff_name may hold several names ("ISM Services PMI|ISM
    Non-Manufacturing PMI": Forex Factory renamed it); the surprise table adds the names Forex Factory used for each
    event in 2016-2023 (ff_name_found: "FOMC Meeting Minutes" -> FOMC Minutes). The first name wins."""
    out: dict[str, Any] = {}
    for i, cell in signs["ff_name"].items():
        for name in str(cell or "").split("|"):
            if name.strip() and name.strip() not in out:
                out[name.strip()] = i
    if table is None and config.NEWS_SURPRISES.is_file():
        table = pd.read_parquet(config.NEWS_SURPRISES, columns=["event", "ff_name_found"])
    if table is not None:
        row_of = {e: i for i, e in signs["event"].items()}
        pairs = table.dropna(subset=["ff_name_found"])[["ff_name_found", "event"]].drop_duplicates()
        for name, event in pairs.itertuples(index=False):
            if name and name not in out and event in row_of:
                out[name] = row_of[event]
    return out


def events(rows: list[dict], signs: pd.DataFrame | None = None) -> pd.DataFrame:
    """His news (config.LIVE_COLOURS) of the rows, one row per number: the time (utc, server, India), the numbers as
    Forex Factory writes them, and from signs.csv the report it belongs to, its rank in the report, the method and the
    sign. A number signs.csv does not know keeps its own name as the report (no lookup for it). A row with no exact
    time ("All Day", "Tentative") gets no alert: it has no minute to count to."""
    s = NS.load_signs() if signs is None else signs
    by_ff = ff_names(s)
    out = []
    for r in rows:
        colour = _colour(r.get("impactClass"))
        if colour not in config.LIVE_COLOURS or not r.get("dateline") or r.get("timeMasked"):
            continue
        utc = pd.Timestamp(int(r["dateline"]), unit="s", tz="UTC")
        hit = s.loc[by_ff[r["name"]]] if r["name"] in by_ff else None
        out.append(dict(ff_id=r["id"], name=r["name"], colour=colour, utc=utc,
                        actual=r.get("actual") or "", forecast=r.get("forecast") or "", previous=r.get("previous") or "",
                        event=hit["event"] if hit is not None else "", report=hit["report"] if hit is not None else r["name"],
                        rank=int(hit["rank"]) if hit is not None else 99,
                        method=hit["method"] if hit is not None else "flagged",
                        sign=int(hit["sign"]) if hit is not None else 0))
    ev = pd.DataFrame(out, columns=["ff_id", "name", "colour", "utc", "actual", "forecast", "previous", "event", "report",
                                    "rank", "method", "sign"])
    if len(ev):
        ev["server"] = clock.utc_to_server(pd.DatetimeIndex(ev["utc"]))
        ev["ist"] = pd.DatetimeIndex(ev["utc"]).tz_convert("Asia/Kolkata")
        ev = ev.sort_values(["utc", "report", "rank"], kind="stable", ignore_index=True)
    return ev


def releases(ev: pd.DataFrame) -> list[dict]:
    """One release per minute and report: its numbers (headline first) and its colour (the reddest of them). The
    headline = the lowest rank with a forecast-method number (machine/news/surprise.py: NFP = the payrolls)."""
    out = []
    if not len(ev):
        return out
    for (utc, report), g in ev.groupby(["utc", "report"], sort=True):
        g = g.sort_values("rank", kind="stable")
        scored = g[(g["method"] == "forecast") & (g["forecast"] != "")]
        head = scored.iloc[0] if len(scored) else g.iloc[0]
        colour = "red" if (g["colour"] == "red").any() else g["colour"].iloc[0]
        out.append(dict(utc=utc, server=head["server"], ist=head["ist"], report=report, colour=colour,
                        head=head.to_dict(), numbers=g.to_dict("records"),
                        key=f"{pd.Timestamp(utc).strftime('%Y%m%d-%H%M')}-{int(head['ff_id'])}"))
    return sorted(out, key=lambda r: (r["utc"], r["report"]))


def out_now(rel: dict) -> bool:
    """Its headline number is on Forex Factory now."""
    return bool(str(rel["head"].get("actual") or "").strip())


def value(text: str) -> float:
    return FF.number(text)[0] if text else np.nan
