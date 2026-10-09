"""How big a live number's surprise is (L17a) - the same rule as machine/news/surprise.py (L12b.1), for one number.

    from machine.live import surprise as LS
    s = LS.score(head)     # head: a release's headline row (machine/live/calendar.py), its number out
    s["z"], s["group"]     # 1.8, "big good for the dollar"

dollar surprise = sign x (actual - forecast), both Forex Factory's (signs.csv gives the sign; MT5's forecast is not
the consensus - forexfactory.py). z = dollar surprise / the usual miss: the average size of the SAME event's last
surprise.WINDOW (24) surprises, each capped at surprise.CAP times the middle-sized one; fewer than surprise.MIN_PAST
(8) -> not scored. group: |z| >= surprise.BIG (1) -> big good / big bad for the dollar, else small.

The past surprises are 2016-2023 (data/news/surprises.parquet), so the "usual miss" of a 2026 number is the usual miss
of the end of 2023 (Claude's reading: T14 allows recent 2024+ news numbers to size a surprise, but no 2024+ history
of them is fetched yet - every alert names the years). The Fed's decisions ("2y yield": the 2-year yield of the whole
day) are known only the next day, and speeches / minutes have no number: those are not scored here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from machine import config
from machine.news import surprise as NS
from machine.live import calendar as LC

_table: list[pd.DataFrame] = []


def past_misses(event: str, table: pd.DataFrame | None = None) -> np.ndarray:
    """The sizes of the event's last WINDOW scored surprises (2016-2023), oldest first."""
    if table is None:
        if not _table:
            _table.append(pd.read_parquet(config.NEWS_SURPRISES))
        table = _table[0]
    t = table[(table["event"] == event) & (table["method"] == "forecast") & table["dollar_surprise"].notna()]
    t = t.sort_values("known_from", kind="stable")
    return t["dollar_surprise"].abs().to_numpy(dtype=float)[-NS.WINDOW:]


def score(head: dict, table: pd.DataFrame | None = None) -> dict[str, Any]:
    """{dollar_surprise, usual_miss, past_n, z, group, why} for a headline row whose number is out."""
    out: dict[str, Any] = dict(dollar_surprise=np.nan, usual_miss=np.nan, past_n=0, z=np.nan, group="not scored",
                               why="")
    method = head.get("method")
    if method == "2y yield":
        out["why"] = "the Fed's surprise is the 2-year yield of the whole day, known only tomorrow"
        return out
    if method != "forecast":
        out["why"] = "this news has no forecast number"
        return out
    actual, forecast = LC.value(head.get("actual", "")), LC.value(head.get("forecast", ""))
    if not np.isfinite(actual):
        out["why"] = "the number is not out yet"
        return out
    if not np.isfinite(forecast):
        out["why"] = "Forex Factory has no forecast for it"
        return out
    ds = float(head.get("sign", 0)) * (actual - forecast)
    if table is None and not _table and not config.NEWS_SURPRISES.is_file():
        out.update(dollar_surprise=ds, why="no 2016-2023 surprise table on this PC (data/news/) - the size is not known")
        return out                                     # the alert still goes, with the number and no size
    past = past_misses(head.get("event", ""), table)
    out.update(dollar_surprise=ds, past_n=int(len(past)))
    if len(past) < NS.MIN_PAST:
        out["why"] = f"only {len(past)} past surprises of it (needs {NS.MIN_PAST})"
        return out
    mid = float(np.median(past))
    usual = float(np.mean(np.minimum(past, NS.CAP * mid))) if mid > 0 else float(np.mean(past))
    if usual <= 0:
        out["why"] = "its past surprises were all zero"
        return out
    z = ds / usual
    group = ("big good for the dollar" if z >= NS.BIG else "big bad for the dollar" if z <= -NS.BIG else "small")
    out.update(usual_miss=usual, z=z, group=group)
    return out
