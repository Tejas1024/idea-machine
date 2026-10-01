"""News surprise, measured right (L12b.1).     python -m machine.news.surprise   -> data/news/surprises.parquet

For every release in the calendar (data/news/events.parquet, L12): how far the number landed from what was expected,
turned the right way round for the dollar, and sized against the same event's OWN PAST surprises only.

1. data/news/signs.csv (in git; one line per MT5 event name, explained line by line in its `why` column):
     report / rank   which report the number comes in; rank 1 = THE number used for that report (NFP = the headline
                     payrolls; CPI = core CPI m/m - see its line). If rank 1 is missing that day, the lowest rank there.
     method          "forecast"  surprise = actual - forecast, both from Forex Factory's row for the same release;
                     "2y yield"  the Fed's decisions (no forecast number): the 2-year Treasury yield (FRED DGS2) at the
                                 first daily quote AFTER the release minus the last one BEFORE it, in percentage points
                                 - "daily, measured after the release" (known only from the next quote day, yields.py);
                     "flagged"   no forecast and no such series (speeches, minutes, auctions ...): not scored.
     sign            +1 = a number ABOVE the forecast is good for the dollar; -1 = turned round (jobless claims, the
                     unemployment rates); 0 = flagged.
   The forecast is NOT MT5's: MT5's `forecast` column is not the economists' consensus (forexfactory.py says why).
2. dollar_surprise = sign x surprise: above 0 = good for the dollar, in the event's own unit.
3. z = dollar_surprise / usual_miss. usual_miss = the average size of the same event's last WINDOW (24) surprises
   that were KNOWN BEFORE this release, each capped at CAP (4) times the middle-sized one of those 24 - so one freak
   month (June 2020: 10.5 million jobs off) cannot make every miss for two years look small. Fewer than MIN_PAST (8)
   known past surprises -> not scored. The window ends before the event: the cut-off-the-future test
   (tests/test_surprise.py) checks it.
4. group: "big good for the dollar" (z >= BIG), "small" (|z| < BIG), "big bad for the dollar" (z <= -BIG),
   "not scored" (why_not_scored says why). BIG = 1: a miss bigger than the usual miss.
WINDOW, MIN_PAST, CAP and BIG are Claude's numbers (chosen before any gold price was looked at), not Tejas's rules.

Nothing here reads a gold price: how gold reacted is L12b.2.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from machine import config, guards
from machine.data import clock
from machine.news import forexfactory, yields

WINDOW = 24          # past surprises of the same event that make its "usual miss"
MIN_PAST = 8         # fewer known past surprises -> not scored
CAP = 4.0            # a past miss counts at most CAP x the middle-sized one of the window
BIG = 1.0            # |z| >= BIG -> a big surprise
NY = "America/New_York"
METHODS = ("forecast", "2y yield", "flagged")
GROUPS = ("big good for the dollar", "small", "big bad for the dollar", "not scored")
SIGN_COLUMNS = ["event", "kind", "report", "rank", "method", "sign", "good_for_dollar", "ff_name", "why"]
MULTIPLIER = {"": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9, "trillions": 1e12}


# ---- signs.csv -------------------------------------------------------------------------------------------------------

def load_signs(path=None) -> pd.DataFrame:
    """data/news/signs.csv, checked: one line per event; sign +1 / -1 for a scored method, 0 for flagged; inside a
    report that is scored, every rank different (so "the number used" is never a tie)."""
    s = pd.read_csv(path or config.NEWS_SIGNS, dtype={"ff_name": str}, keep_default_na=False)
    if list(s.columns) != SIGN_COLUMNS:
        raise ValueError(f"signs.csv columns must be {SIGN_COLUMNS}, not {list(s.columns)}")
    s["rank"] = s["rank"].astype(int)
    s["sign"] = s["sign"].astype(int)
    bad = []
    if s["event"].duplicated().any():
        bad.append(f"events twice: {sorted(s.loc[s['event'].duplicated(), 'event'])}")
    if not s["method"].isin(METHODS).all():
        bad.append(f"unknown method: {sorted(set(s['method']) - set(METHODS))}")
    wrong = s[(s["method"] == "flagged") != (s["sign"] == 0)]
    if len(wrong):
        bad.append(f"flagged <-> sign 0 broken for: {list(wrong['event'])}")
    if not s["sign"].isin((-1, 0, 1)).all():
        bad.append("a sign is not +1, -1 or 0")
    scored = s[s["report"].isin(s.loc[s["method"] != "flagged", "report"])]
    tie = scored[scored.duplicated(["report", "rank"], keep=False)]
    if len(tie):
        bad.append(f"two lines share a rank in one report: {tie[['report', 'rank', 'event']].values.tolist()}")
    if (s["why"].str.strip() == "").any():
        bad.append(f"a line without its why: {list(s.loc[s['why'].str.strip() == '', 'event'])}")
    if bad:
        raise ValueError("signs.csv: " + "; ".join(bad))
    return s


# ---- the surprise table -------------------------------------------------------------------------------------------

def score(events: pd.DataFrame, ff: pd.DataFrame, dgs2: pd.DataFrame, signs: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar row: its surprise, dollar_surprise, usual_miss, z, group and known_from.

    events: calendar rows (utc, server, ist, day, event, kind, importance, actual, multiplier, value_id, exact_time,
            scheduled). ff: forexfactory.load() rows (utc, name, colour, actual_value, forecast_value, better_worse).
    dgs2: yields.load() (date, dgs2, quoted, known_from). Every argument is used only as given - pass them as known
    at a moment (as_known_at) and the result is the table as known then.
    """
    missing = sorted(set(events["event"]) - set(signs["event"]))
    if missing:
        raise ValueError(f"signs.csv has no line for {missing} - add them (a sign, or flagged with why)")
    t = events.merge(signs.drop(columns=["kind"]), on="event", how="left", validate="many_to_one")
    t = t.sort_values(["utc", "rank"], kind="stable").reset_index(drop=True)
    t["ny_date"] = t["utc"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    first = t.groupby(["report", "ny_date"])["rank"].transform("min")
    t["headline"] = t["rank"] == first
    t = _add_ff(t, ff)
    t["surprise"] = np.where(t["method"] == "forecast", t["ff_actual"] - t["ff_forecast"], np.nan)
    t["known_from"] = pd.to_datetime(t["utc"].where(t["method"] == "forecast"), utc=True).astype(
        "datetime64[ns, UTC]")
    t["measured"] = np.where(t["method"] == "forecast", "at the release (actual - forecast)", "")
    t = _add_yield(t, dgs2)
    t["dollar_surprise"] = t["sign"] * t["surprise"]
    t = _add_size(t)
    t["group"] = np.select([t["z"] >= BIG, t["z"] <= -BIG, t["z"].abs() < BIG], [GROUPS[0], GROUPS[2], GROUPS[1]],
                           default=GROUPS[3])
    t["why_not_scored"] = _why_not(t)
    cols = ["utc", "server", "ist", "day", "ny_date", "event", "kind", "importance", "value_id", "report", "rank",
            "headline", "method", "sign", "good_for_dollar", "ff_id", "ff_name_found", "ff_colour", "ff_match", "ff_actual",
            "ff_forecast", "ff_better_worse", "mt5_actual", "surprise", "dollar_surprise", "past_n", "usual_miss", "z",
            "group", "why_not_scored", "measured", "known_from", "yield_before", "yield_after"]
    return t[cols]


def _add_ff(t: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """Forex Factory's row for each row whose signs.csv line names one: a name from ff_name at the same minute, else
    the same New York day (the nearest in time). ff_match says which ("same minute" / "same day" / "none"). Every
    such row gets Forex Factory's colour (his red / orange, T7); only "forecast" rows use its numbers."""
    scale = t["multiplier"].fillna("").map(MULTIPLIER).fillna(1.0) if "multiplier" in t else 1.0
    t["mt5_actual"] = t["actual"] * scale                       # MT5's first-released number, in Forex Factory's units
    want = t.loc[t["ff_name"].fillna("") != "", ["ff_name", "utc", "ny_date"]]   # every line: its colour too
    want = want.assign(row=want.index, name=want["ff_name"].str.split("|")).explode("name")
    have = pd.DataFrame({"name": ff["name"], "ff_utc": pd.to_datetime(ff["utc"], utc=True),
                         "ny_date": ff["utc"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize(),
                         "ff_colour": ff["colour"], "ff_better_worse": ff["better_worse"],
                         "ff_actual": ff["actual_value"].astype(float), "ff_forecast": ff["forecast_value"].astype(float),
                         "ff_name_found": ff["name"], "ff_id": ff["id"] if "id" in ff else pd.NA})
    hit = want.merge(have, on=["name", "ny_date"], how="inner")
    hit["gap"] = (hit["ff_utc"] - pd.to_datetime(hit["utc"], utc=True)).abs()
    hit = hit.sort_values(["row", "gap"], kind="stable").drop_duplicates("row").set_index("row")
    hit["ff_match"] = np.where(hit["gap"] == pd.Timedelta(0), "same minute", "same day")
    cols = ["ff_id", "ff_name_found", "ff_colour", "ff_better_worse", "ff_actual", "ff_forecast", "ff_match"]
    t = t.join(hit[cols])
    t["ff_match"] = t["ff_match"].fillna("none")
    return t


def _add_yield(t: pd.DataFrame, dgs2: pd.DataFrame) -> pd.DataFrame:
    """The Fed's decisions: the 2-year yield after the release minus before, one number per decision day."""
    t["yield_before"] = np.nan
    t["yield_after"] = np.nan
    fed = t[t["method"] == "2y yield"]
    quoted = _naive_utc(dgs2["quoted"])
    for _, day in fed.groupby(["report", "ny_date"]):
        exact = day[day["exact_time"]] if "exact_time" in day else day
        if not len(exact):
            continue
        release = _naive_utc(exact["utc"]).min()
        before = np.searchsorted(quoted, release, side="left") - 1           # quoted strictly before the release
        after = np.searchsorted(quoted, release, side="right")               # quoted strictly after it
        if before < 0 or after >= len(dgs2):
            continue
        q = dgs2.iloc[[before, after]]
        t.loc[day.index, "yield_before"] = q["dgs2"].iloc[0]
        t.loc[day.index, "yield_after"] = q["dgs2"].iloc[1]
        t.loc[day.index, "surprise"] = round(q["dgs2"].iloc[1] - q["dgs2"].iloc[0], 6)
        t.loc[day.index, "known_from"] = pd.Timestamp(q["known_from"].iloc[1]).tz_convert("UTC")
        t.loc[day.index, "measured"] = "daily, measured after the release (2-year yield)"
    return t


def _add_size(t: pd.DataFrame) -> pd.DataFrame:
    """usual_miss and z from the same event's past surprises known before this release (the Fed's decisions: from
    past decision days, one number a day)."""
    t["past_n"] = 0
    t["usual_miss"] = np.nan
    t["z"] = np.nan
    fed = (t["method"] == "2y yield") & t["headline"]
    keys = [("event", t["method"] == "forecast"), ("report", fed)]
    for column, mask in keys:
        for _, g in t[mask & t["dollar_surprise"].notna()].groupby(column):
            g = g.sort_values("utc", kind="stable")
            known = _naive_utc(g["known_from"])
            at = _naive_utc(g["utc"])
            size = g["dollar_surprise"].abs().to_numpy()
            order = np.argsort(known, kind="stable")
            known_sorted, size_sorted = known[order], size[order]
            for j, idx in enumerate(g.index):
                n_known = np.searchsorted(known_sorted, at[j], side="left")      # known strictly before the release
                past = size_sorted[max(0, n_known - WINDOW):n_known]
                t.at[idx, "past_n"] = len(past)
                if len(past) < MIN_PAST:
                    continue
                mid = np.median(past)
                usual = np.mean(np.minimum(past, CAP * mid)) if mid > 0 else np.mean(past)
                if usual > 0:
                    t.at[idx, "usual_miss"] = usual
                    t.at[idx, "z"] = g.at[idx, "dollar_surprise"] / usual
    day_z = t[fed].set_index(["report", "ny_date"])[["past_n", "usual_miss", "z"]]
    others = (t["method"] == "2y yield") & ~t["headline"]
    if others.any() and len(day_z):
        got = t.loc[others, ["report", "ny_date"]].merge(day_z, left_on=["report", "ny_date"], right_index=True,
                                                         how="left")
        t.loc[others, ["past_n", "usual_miss", "z"]] = got[["past_n", "usual_miss", "z"]].to_numpy()
    t["past_n"] = t["past_n"].fillna(0).astype(int)
    return t


def _naive_utc(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def _why_not(t: pd.DataFrame) -> pd.Series:
    why = pd.Series("", index=t.index, dtype=object)
    flagged = t["method"] == "flagged"
    why[flagged] = "flagged in signs.csv (no forecast and no market series)"
    fc = t["method"] == "forecast"
    why[fc & (t["ff_match"] == "none")] = "no Forex Factory row for this release"
    why[fc & (t["ff_match"] != "none") & t["ff_forecast"].isna()] = "Forex Factory has no forecast for this release"
    why[fc & (t["ff_match"] != "none") & t["ff_forecast"].notna() & t["ff_actual"].isna()] = \
        "Forex Factory has no actual number for this release"
    fed = t["method"] == "2y yield"
    why[fed & t["surprise"].isna()] = "no 2-year yield quote around it"
    few = t["dollar_surprise"].notna() & t["z"].isna()
    why[few] = "only " + t.loc[few, "past_n"].astype(str) + f" known past surprises (needs {MIN_PAST})"
    why[t["z"].notna()] = ""
    return why


# ---- what was known when ------------------------------------------------------------------------------------------

def latest(times: Any, table: pd.DataFrame, report: str, column: str = "z") -> pd.Series:
    """For each moment (naive = server time): `column` of the latest headline release of `report` KNOWN by then
    (known_from <= the moment). The Fed's 2-year yield number is known only a day after the decision."""
    idx = pd.DatetimeIndex([times] if isinstance(times, (str, pd.Timestamp)) else times)
    utc = idx.tz_convert("UTC") if idx.tz is not None else clock.server_to_utc(idx)
    rows = table[(table["report"] == report) & table["headline"] & table[column].notna()]
    rows = rows.sort_values("known_from", kind="stable")
    known = _naive_utc(rows["known_from"])
    pos = np.searchsorted(known, utc.tz_localize(None).to_numpy(dtype="datetime64[ns]"), side="right") - 1
    out = np.full(len(idx), np.nan)
    ok = pos >= 0
    out[ok] = rows[column].to_numpy(dtype=float)[pos[ok]]
    return pd.Series(out, index=idx, name=f"{report}: {column}")


def build(write: bool = True) -> pd.DataFrame:
    from machine.news import calendar
    events = calendar.load_events(importance=("high", "moderate"))
    table = score(events, forexfactory.load(), yields.load(), load_signs())
    guards.check_dates(table[["utc", "server", "day"]])
    if write:
        tmp = config.NEWS_SURPRISES.with_suffix(".tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(config.NEWS_SURPRISES)
    return table


def load(start: Any = None, end: Any = None) -> pd.DataFrame:
    """The surprise table rows with SERVER day in [start, end] (like load_events). A request reaching 2024 is refused."""
    request = [x for x in (start, end) if x is not None]
    if request:
        guards.check_dates(request)
    if not config.NEWS_SURPRISES.is_file():
        raise FileNotFoundError(f"{config.NEWS_SURPRISES} is missing - make it with `python -m machine.news.surprise`")
    t = pd.read_parquet(config.NEWS_SURPRISES)
    guards.check_dates(t[["utc", "server"]])
    day = t["server"].fillna(t["day"])
    if start is not None:
        t = t[day >= pd.Timestamp(str(start))]
    if end is not None:
        from machine.news.calendar import _end_exclusive
        t = t[day.loc[t.index] < _end_exclusive(end)]
    return t.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argparse.ArgumentParser(prog="python -m machine.news.surprise", description=__doc__.split("\n\n")[0]
                            ).parse_args(argv)
    from machine.news import surprise_report
    table = build()
    surprise_report.report(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
