"""The prices from MT5, every day, sealed (L16.1 steps 1-2).

Every complete SERVER day after the last one kept goes into data/sealed/mt5/<M1|M5>/<server day>.csv, in L1's own
format (time,open,high,low,close,vol,spread - the same text the old export wrote, so one reader reads all 2024+ rows),
and one line per run into data/sealed/mt5/manifest.jsonl: the days, their row counts, first and last bar TIMES and
each file's SHA-256. NEVER parsed: the bars go from MT5 straight to text; no price is computed on, printed, logged or
returned. They stay closed until a T14 use opens them through machine.guards (live-trial for the L17a alerts; the
final exam only with his yes again on that day). This continues L1, which copied 2024 - 14 Aug 2026 the same way.

Rules (hard rules 1 and 10 - read only: no order, no live price used or shown):
  - the clock: MT5 gives bar times in its SERVER clock; a server day is cut at the server's midnight, as L1 did.
  - only MT5_SERVER (MetaQuotes-Demo, where the price data and its measured clock came from); another server = refused.
  - a day is kept only when MT5 already has a bar on a LATER day (a day that may still be loading is never kept half),
    and never today. MT5 is asked twice; the two answers must be the same (its history may still be downloading).
  - a kept day is never written again. If MT5 later gives other bars for it, nothing is overwritten: the run fails
    and says so (the broker changed its history - worth knowing).
  - the MT5 login, when the terminal does not keep one itself: only in .env (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

from machine import config

HEADER = "time,open,high,low,close,vol,spread"
L1_SEALED = {"M1": "gold_m1_2024plus.csv", "M5": "gold_m5_mt5_2024plus.csv"}   # what L1 copied (data/sealed/)
EPOCH = dt.date(1970, 1, 1)
ASK_AGAIN_SECONDS = 3
GOLD_OPENS = dt.time(1, 15)    # the server day's first bar is 01:00 (every kept day); 15 minutes on, a loaded MT5 has one
LOAD_ASK_SECONDS = 10          # while MT5 loads, its newest bar is asked for this often (up to MT5_LOAD_WAIT_SECONDS)


class PriceError(RuntimeError):
    """Why nothing (more) was kept, in plain words."""


# ================================================================================================= where we are

def folder() -> Path:
    return Path(config.SEALED_MT5_DIR)


def manifest() -> list[dict]:
    path = folder() / "manifest.jsonl"
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def last_time_text(path: Path) -> str | None:
    """The first field (the time) of a CSV's last line, read from the file's end. The prices on it are never read."""
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 4096))
        tail = fh.read().splitlines()
    for line in reversed(tail):
        if line.strip():
            return line.split(b",", 1)[0].decode("ascii", "replace").strip()
    return None


def kept_until(tf: str) -> dt.date | None:
    """The last server day kept for `tf`: the newest manifest line, else the last line of L1's sealed file."""
    rows = [r for r in manifest() if r.get("tf") == tf]
    if rows:
        return dt.date.fromisoformat(rows[-1]["to_day"])
    t = last_time_text(Path(config.SEALED_DIR) / L1_SEALED[tf])
    return dt.date.fromisoformat(t[:10]) if t else None


def server_clock(now: dt.datetime | None = None):
    """Now on the server's clock (machine/data/clock.py: UTC+2, +3 on the European summer dates)."""
    import pandas as pd

    from machine.data import clock
    now = now or dt.datetime.now(dt.timezone.utc)
    t = pd.Timestamp(now.astimezone(dt.timezone.utc)).tz_convert("UTC")
    return clock.utc_to_server(pd.DatetimeIndex([t]))[0]


def server_today(now: dt.datetime | None = None) -> dt.date:
    """Today on the server's clock, MT5_DAY_DONE_MINUTES after its midnight."""
    import pandas as pd
    return (server_clock(now) - pd.Timedelta(minutes=config.MT5_DAY_DONE_MINUTES)).date()


def should_have_today(now: dt.datetime | None = None) -> bool:
    """A server weekday (Mon-Fri) past gold's opening: a loaded MT5 has bars of today."""
    s = server_clock(now)
    return s.weekday() < 5 and s.time() >= GOLD_OPENS


def day_of(seconds: int) -> dt.date:
    return EPOCH + dt.timedelta(days=int(seconds) // 86400)


def seconds_of(day: dt.date) -> int:
    return (day - EPOCH).days * 86400


# ================================================================================================= the text

def lines(rates) -> list[str]:
    """MT5's bars as L1's text lines: time in the server clock, each price as MT5 gave it (no rounding, no maths)."""
    out = []
    for r in rates:
        t = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=int(r["time"]))
        out.append(f"{t:%Y-%m-%d %H:%M:%S},{float(r['open'])!r},{float(r['high'])!r},{float(r['low'])!r},"
                   f"{float(r['close'])!r},{int(r['tick_volume'])},{int(r['spread'])}")
    return out


def day_text(rates) -> bytes:
    return (HEADER + "\n" + "\n".join(lines(rates)) + "\n").encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ================================================================================================= MT5

def _mt5():
    import MetaTrader5 as mt5
    return mt5


def _open(mt5) -> dict:
    """Start / join MT5 and check it is the right server. Returns who it is (no secret)."""
    login = config.env("MT5_LOGIN")
    kw = {}
    if login:
        kw = {"login": int(login), "password": config.env("MT5_PASSWORD") or "",
              "server": config.env("MT5_SERVER") or config.MT5_SERVER}
    path = Path(config.MT5_TERMINAL)
    ok = mt5.initialize(path=str(path), **kw) if path.is_file() else mt5.initialize(**kw)
    if not ok:
        raise PriceError(f"MT5 did not start ({mt5.last_error()}) - is it installed and logged in?")
    term, acc = mt5.terminal_info(), mt5.account_info()
    if term is None or not term.connected:
        raise PriceError("MT5 is open but not connected to its server (no internet, or not logged in)")
    if acc is None:
        raise PriceError("MT5 is not logged in to an account")
    if acc.server != config.MT5_SERVER:
        raise PriceError(f"MT5 is logged in to {acc.server}, not {config.MT5_SERVER} (the server the price data and "
                         "its clock came from) - nothing kept")
    if not mt5.symbol_select(config.MT5_SYMBOL, True):
        raise PriceError(f"MT5 has no {config.MT5_SYMBOL}: {mt5.last_error()}")
    return {"server": acc.server, "company": acc.company,
            "account": {0: "demo", 1: "contest", 2: "real"}.get(acc.trade_mode, str(acc.trade_mode))}


def _ask(mt5, tf: str, start: dt.date, end: dt.date, sleep=time.sleep):
    """The bars of server days start..end (both in), asked twice: the same answer, or MT5 is still loading."""
    code = getattr(mt5, f"TIMEFRAME_{tf}")
    lo, hi = seconds_of(start), seconds_of(end + dt.timedelta(days=1))
    answers = []
    for _ in range(3):
        r = mt5.copy_rates_range(config.MT5_SYMBOL, code, lo, hi)
        if r is None:
            raise PriceError(f"MT5 gave no {tf} bars ({mt5.last_error()})")
        r = r[(r["time"] >= lo) & (r["time"] < hi)]
        if answers and len(r) == len(answers[-1]) and sha256(day_text(r)) == sha256(day_text(answers[-1])):
            return r
        answers.append(r)
        sleep(ASK_AGAIN_SECONDS)
    raise PriceError(f"MT5 was still loading its {tf} history (3 answers differed) - tried again tomorrow")


def newest_day(mt5, tf: str) -> dt.date | None:
    """The server day of MT5's newest bar (its TIME only)."""
    r = mt5.copy_rates_from_pos(config.MT5_SYMBOL, getattr(mt5, f"TIMEFRAME_{tf}"), 0, 1)
    return day_of(int(r["time"][-1])) if r is not None and len(r) else None


def newest_loaded(mt5, tf: str, now: dt.datetime | None = None, *, sleep=time.sleep) -> tuple[dt.date | None, int]:
    """MT5's newest day once it has loaded, and the seconds waited. After a power cut the daily run catches up the
    moment Windows starts, and MT5 has only just opened: its newest bar is still the evening before, so yesterday looked
    unfinished and was not kept (6 Oct 2026 08:02: "nothing new to keep" - Monday's daily check of the live feed waited
    a day). On a server weekday past gold's opening, MT5 is asked again every LOAD_ASK_SECONDS until it has a bar of
    today, up to MT5_LOAD_WAIT_SECONDS. A weekend never waits; a market holiday waits once and keeps nothing."""
    today, newest, waited = server_today(now), newest_day(mt5, tf), 0
    if should_have_today(now):
        while (newest is None or newest < today) and waited < config.MT5_LOAD_WAIT_SECONDS:
            sleep(LOAD_ASK_SECONDS)
            waited += LOAD_ASK_SECONDS
            newest = newest_day(mt5, tf)
    return newest, waited


# ================================================================================================= the run

def store(now: dt.datetime | None = None, *, mt5=None, sleep=time.sleep) -> dict:
    """Keep every complete server day after the last one kept, for each timeframe. Returns times and counts only."""
    mt5 = mt5 or _mt5()
    who = _open(mt5)
    try:
        now = now or dt.datetime.now(dt.timezone.utc)
        today = server_today(now)
        out = {**who, "timeframes": {}}
        for tf in config.MT5_STORE_TIMEFRAMES:
            last = kept_until(tf)
            if last is None:
                raise PriceError(f"nothing to continue from: no {L1_SEALED[tf]} in data/sealed/ and no manifest line")
            newest, waited = newest_loaded(mt5, tf, now, sleep=sleep)
            loading = {"waited_s": waited,
                       "no_bar_today": should_have_today(now) and (newest is None or newest < today)}
            end = min(today - dt.timedelta(days=1), (newest or last) - dt.timedelta(days=1))
            start = last + dt.timedelta(days=1)
            if start > end:
                out["timeframes"][tf] = {"from_day": start.isoformat(), "to_day": None, "days": [], **loading}
                continue
            out["timeframes"][tf] = {**_keep(tf, _ask(mt5, tf, start, end, sleep), start, end, who), **loading}
        return out
    finally:
        mt5.shutdown()


def _keep(tf: str, rates, start: dt.date, end: dt.date, who: dict) -> dict:
    base = folder() / tf
    base.mkdir(parents=True, exist_ok=True)
    by_day: dict[dt.date, list] = {}
    for r in rates:
        by_day.setdefault(day_of(int(r["time"])), []).append(r)
    days = []
    for day in sorted(by_day):
        rows = by_day[day]
        raw = day_text(rows)
        path = base / f"{day.isoformat()}.csv"
        if path.is_file():
            if sha256(path.read_bytes()) == sha256(raw):
                continue
            raise PriceError(f"MT5 now gives other {tf} bars for {day} than the ones kept - nothing overwritten")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)                                  # whole or not at all
        first = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=int(rows[0]["time"]))
        last = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=int(rows[-1]["time"]))
        days.append({"day": day.isoformat(), "rows": len(rows), "first": f"{first:%H:%M}", "last": f"{last:%H:%M}",
                     "sha256": sha256(raw)})
    line = {"at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "tf": tf, "symbol": config.MT5_SYMBOL,
            "from_day": start.isoformat(), "to_day": end.isoformat(), "days": days,
            "rows": sum(d["rows"] for d in days), **who}
    path = folder() / "manifest.jsonl"
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def same_as_l1(day: dt.date, tf: str = "M1", *, mt5=None) -> dict:
    """Does MT5 today give the SAME text for `day` as L1's sealed copy? Compares two SHA-256 - no price is read.
    (Used once in L16.1 to prove the format and the clock are L1's.)"""
    mt5 = mt5 or _mt5()
    _open(mt5)
    try:
        rates = _ask(mt5, tf, day, day)
    finally:
        mt5.shutdown()
    prefix = day.isoformat().encode()
    with open(Path(config.SEALED_DIR) / L1_SEALED[tf], "rb") as fh:
        theirs = [ln for ln in fh.read().splitlines() if ln[:10] == prefix]      # the date text only, as L1 split
    mine = day_text(rates).splitlines()[1:]
    return {"day": day.isoformat(), "tf": tf, "rows_l1": len(theirs), "rows_mt5_now": len(mine),
            "same": sha256(b"\n".join(theirs)) == sha256(b"\n".join(mine))}
