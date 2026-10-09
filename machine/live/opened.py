"""The lock for the live alerts (L17a): 2024+ data is read only under a T14 use Tejas said yes to (machine/guards.py).

    from machine.live import opened
    opened.week(times, sunday, use="live-trial", who=..., why=...)

The first time a week's 2024+ data is read for a use, the lock opens it and logs it: ONE line in
data/sealed_access.log per week and use. LIVE_DIR/opened.json remembers it, as a pass runs every minute in a new
program. Every later read checks again that the T14 line still says yes (no log line): a "no" there stops that part of
the alerts at once.
  news-values  this week's Forex Factory numbers (T14 question 2)       opened.json key: the week's Sunday
  live-trial   live gold prices, the recent days an alert needs (T14 question 1)   key: "live-trial <Sunday>"
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from machine import config, guards

_opened: set[str] = set()                    # the keys this program run opened or checked


def _key(use: str, sunday: dt.date) -> str:
    return sunday.isoformat() if use == "news-values" else f"{use} {sunday.isoformat()}"


def _sealed(times: pd.DatetimeIndex) -> bool:
    if not len(times):
        return False
    start = pd.Timestamp(config.SEALED_FROM)
    return times.max() >= (start.tz_localize("UTC") if times.tz is not None else start)


def _show_t14() -> str:
    try:
        return Path(config.T14_DECISIONS).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return str(config.T14_DECISIONS)


def week(times: pd.DatetimeIndex, sunday: dt.date, *, use: str, who: str, why: str) -> None:
    """Open `use` for this week's 2024+ times (logged once per week), or check that T14 still says yes. Raises
    guards.SealedYearsError when T14 does not say yes to `use`."""
    times = pd.DatetimeIndex(times)
    if not _sealed(times):
        return
    key = _key(use, sunday)
    path = Path(config.LIVE_DIR) / "opened.json"
    try:
        done = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        done = {}
    if key in _opened or key in done:
        if guards.allowed_uses().get(use) is not True:
            raise guards.SealedYearsError(f"{_show_t14()} no longer says \"use: {use} = yes\": this part of the live "
                                          f"alerts stops. {guards.SEALED_WORDS}.")
        _opened.add(key)
        return
    guards.check_dates(times, unlock=True, who=who, why=why, use=use)
    _opened.add(key)
    done[key] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(done), encoding="utf-8")
    tmp.replace(path)
