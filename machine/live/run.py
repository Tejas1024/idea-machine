"""One pass of the live news alerts (L17a) - what is due now - and a rehearsal on a past week.

    python -m machine.live once                  # what is due now (Task Scheduler runs this every minute)
    python -m machine.live rehearse 2023-10-01   # a 2023 week replayed minute by minute: the alerts it would send
    python -m machine.live status                # the last pass, this week's releases, what was queued

A pass: this week's page (machine/live/calendar.py; again after an hour, or every LIVE_AFTER_REFRESH_SECONDS while a
release waits for its number) -> his news, one release per minute and report -> for each:
    before   from 30 minutes before the release until it: the alert with 'the last 10 times'
    after    from the release until LIVE_AFTER_WAIT_MINUTES after it, as soon as its number is on Forex Factory: the
             number, the surprise (machine/live/surprise.py) and 'the last 10 times like this'. News with no number
             (speeches, minutes) gets the before alert only.
A window missed (the PC was off) is skipped - an alert is never sent late. While he has typed STOP (the bot's
state.json), nothing is queued (the bot would send it after START, hours late) - the pass writes "not sent: STOP".
Each alert is one line in the bot's outbox (the one bot sends it, once per id) and one in LIVE_DIR/sent.jsonl.

Live prices (step 2, T14 use 1 - machine/live/prices.py), once per pass: "gold now" in the before-alert; "gold since
the number" and the situation at the release minute (machine/live/situation.py) in the after-alert; and
    move     the big-move alert, no news near (machine/live/move.py): checked on the newest closed bar every pass
MT5 trouble, or a "no" for live-trial in T14, switches the price lines off - the news alerts go on as in step 1.
The preview: hours before a release, its alerts are built ahead in memory on the live feed (a bug shows then, not at
the alert); a failure tells Tejas once. The heartbeat (machine/host/heartbeat.py) alerts when no pass ran for 10 min.
After the pass (locked_once): the daily check of the live feed against MT5's history (machine/live/check.py).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from machine import config, guards
from machine.host import alert as HA
from machine.live import alerts as LA
from machine.live import calendar as LC
from machine.live import move as LM
from machine.live import prices as LP
from machine.live import situation as LSit
from machine.live import surprise as LS
from machine.news import similar

LOCK_STALE = dt.timedelta(minutes=10)
_tab: list[pd.DataFrame] = []


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def table() -> pd.DataFrame:
    """The lookup's 2016-2023 releases (similar.releases), read once per program run."""
    if not _tab:
        _tab.append(similar.releases())
    return _tab[0]


def log(text: str, folder: Path | None = None) -> None:
    path = Path(folder or config.LIVE_DIR) / "live.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{HA.iso(HA.now())} {text}\n")


def read_sent(folder: Path | None = None) -> set[tuple[str, str]]:
    return {(r.get("kind"), r.get("key")) for r in HA.read_jsonl(Path(folder or config.LIVE_DIR) / "sent.jsonl")}


def to_outbox(line_id: str, text: str, outbox: Path | None = None) -> bool:
    """One line for the bot to send (the bot sends an id once). False = this id is in the outbox already."""
    box = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    if any(line.get("id") == line_id for line in HA.read_jsonl(box)):
        return False
    HA.append_jsonl(box, {"id": line_id, "text": str(text)[:4096], "at": HA.iso(HA.now()), "to": "Tejas",
                          "why": "L17a live news alert", "from": "machine.live (L17a)"})
    return True


# ============================================================================================================ due?

def has_number(rel: dict) -> bool:
    """A release whose after-alert waits for a number: a forecast number, or the Fed's decision."""
    h = rel["head"]
    return (h.get("method") == "forecast" and bool(h.get("forecast"))) or h.get("method") == "2y yield"


def due(rel: dict, now: dt.datetime, sent: set[tuple[str, str]]) -> list[str]:
    t = pd.Timestamp(rel["utc"]).to_pydatetime()
    out = []
    if t - dt.timedelta(minutes=config.LIVE_BEFORE_MINUTES) <= now < t and ("before", rel["key"]) not in sent:
        out.append("before")
    end = t + dt.timedelta(minutes=config.LIVE_AFTER_WAIT_MINUTES)
    if has_number(rel) and t <= now <= end and ("after", rel["key"]) not in sent:
        if LC.out_now(rel) or now >= end - dt.timedelta(minutes=1):
            out.append("after")
    return out


def waiting(rel: dict, now: dt.datetime) -> bool:
    """Its window is open and its number is not on the page yet: the page is read again sooner."""
    t = pd.Timestamp(rel["utc"]).to_pydatetime()
    return has_number(rel) and t <= now <= t + dt.timedelta(minutes=config.LIVE_AFTER_WAIT_MINUTES) and \
        not LC.out_now(rel)


# ===================================================================================================== live prices

Feed = Callable[[dt.datetime], tuple["LP.Prices | None", str]]


def prices_now(now: dt.datetime, feed: Feed | None = None) -> tuple[LP.Prices | None, str]:
    """(Prices, "") or (None, why not). Never raises: a "no" in T14 or MT5 trouble only switches the price lines off."""
    try:
        return (feed or LP.live)(now)
    except guards.SealedYearsError as e:
        return None, f"live prices stopped by the lock: {str(e)[:300]}"
    except Exception as e:
        return None, f"live prices failed: {type(e).__name__}: {str(e)[:200]}"


def since(p: LP.Prices | None, rel: dict) -> dict | None:
    """Gold from the last price before the release minute to the newest closed bar (fresh prices only)."""
    if p is None or not p.fresh():
        return None
    t = pd.Timestamp(rel["server"])
    p0, last = p.before(t), p.newest_close()
    if p0 is None or last is None or last <= t:
        return None
    close = float(p.m1["close"].iloc[-1])
    return dict(p0=p0, last=close, move=close - p0, minutes=(last - t) / pd.Timedelta(minutes=1))


def _situation(p: LP.Prices | None, t: pd.Timestamp | None) -> dict:
    """The situation at minute t from fresh live prices (with his bias and levels when his charts can be read)."""
    if p is None or t is None or not p.fresh():
        return {}
    return LSit.whole(p, t)


def last_move(folder: Path) -> pd.Timestamp | None:
    """The bar of the last big-move alert (sent or held by STOP: either starts the quiet hours)."""
    bars = [r["bar"] for r in HA.read_jsonl(folder / "sent.jsonl") if r.get("kind") == "move" and r.get("bar")]
    return pd.Timestamp(max(bars)) if bars else None


def news_frame(rels: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({"server": [pd.Timestamp(r["server"]) for r in rels], "colour": [r["colour"] for r in rels]},
                        columns=["server", "colour"])


# ===================================================================================================== the preview

def code_stamp() -> str:
    """A short fingerprint of machine/'s code (each .py file's name, size and time): a new version gets a new preview."""
    import hashlib
    root = Path(config.ROOT) / "machine"
    h = hashlib.sha1()
    for f in sorted(root.rglob("*.py")):
        st = f.stat()
        h.update(f"{f.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:12]


def preview(now: dt.datetime, rels: list[dict], p: LP.Prices | None, before: dict | None, *,
            tab: pd.DataFrame | None = None, tell: Callable[[str], Any] | None = None) -> dict | None:
    """The next release within LIVE_PREVIEW_HOURS gets its alerts BUILT AHEAD, in memory, on the live feed, as the pass
    will build them (no outbox, no sent.jsonl): the before-alert with gold now, and - for news with a number - the
    after-alert with its forecast as the number at the minute just past. A crash here is found while there is time to
    fix it (5 Oct: a bug that would have stopped the first real before-alert was found only by a dry run). Once per
    release and code version, and again after LIVE_PREVIEW_EVERY_MINUTES. A failure tells Tejas once.
    `before`: the last preview (state.json). Returns the new record, or None when none was due."""
    soon = now + dt.timedelta(hours=config.LIVE_PREVIEW_HOURS)
    nxt = next((r for r in rels if now < pd.Timestamp(r["utc"]).to_pydatetime() <= soon), None)
    if nxt is None:
        return None
    stamp = code_stamp()
    if before and before.get("key") == nxt["key"] and before.get("code") == stamp and \
            now - dt.datetime.fromisoformat(before["at"]) < dt.timedelta(minutes=config.LIVE_PREVIEW_EVERY_MINUTES):
        return None
    rec = {"key": nxt["key"], "report": nxt["report"], "code": stamp, "at": now.isoformat(timespec="seconds"),
           "with_prices": p is not None and p.fresh()}
    try:
        tab = tab if tab is not None else table()
        s = _situation(p, p.at) if p is not None else {}
        LA.before_text(nxt, similar.lookup(tab, LA.question(nxt)), s)
        rec["with_charts"] = "bias_d" in s
        if has_number(nxt) and p is not None and p.fresh():
            fake = {**nxt, "server": p.m1.index[-1], "head": {**nxt["head"], "actual": nxt["head"].get("forecast")}}
            score = LS.score(fake["head"])
            sit = LSit.for_lookup(_situation(p, fake["server"]))
            LA.after_text(fake, score, similar.lookup(tab, LA.question(fake, score, sit)), since(p, fake))
        rec["ok"] = True
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        if not before or before.get("ok", True) or before.get("key") != nxt["key"]:
            when, say = HA.india(pd.Timestamp(nxt["utc"]).to_pydatetime()), tell or HA.alert
            say(f"⚠️ The live alert for {similar.name_of(nxt['report'])[1]} at {when:%H:%M} India time will probably "
                 f"FAIL: building it ahead just now stopped with an error ({rec['error'][:150]}). A Claude session "
                 "must fix it: data/live/state.json and live.log say more.")
    return rec


# =========================================================================================================== a pass

def _queue(kind: str, key: str, text: str, now: dt.datetime, folder: Path, outbox: Path | None,
           paused: Callable[[], bool], **extra) -> dict:
    line_id = f"live-{kind}-{key}"
    stop = paused()
    queued = False if stop else to_outbox(line_id, text, outbox)
    rec = dict(kind=kind, key=key, id=line_id, at=now.isoformat(timespec="seconds"),
               sent="not sent: STOP" if stop else ("queued" if queued else "already in the outbox"), **extra)
    HA.append_jsonl(folder / "sent.jsonl", rec)
    log(f"{kind} {extra.get('report', '')} {key}: {rec['sent']}".replace("  ", " "), folder)
    return {**rec, "text": text}


def once(now: dt.datetime | None = None, *, rows: list[dict] | None = None, folder: Path | None = None,
         outbox: Path | None = None, get: Callable[[str], str] | None = None, tab: pd.DataFrame | None = None,
         paused: Callable[[], bool] = HA.paused, feed: Feed | None = None,
         tell: Callable[[str], Any] | None = None, ahead: bool | None = None) -> list[dict]:
    """What is due now, queued. rows given (a rehearsal or a test): no page is read. feed: where prices come from
    (default: MT5, machine/live/prices.live). ahead: build the next release's alerts ahead (preview) - by default only
    when the real page is read, so a rehearsal never messages him; tell: how a failed preview reaches him (default
    machine/host/alert.alert, looked up at the call)."""
    now = now or _now()
    folder = Path(folder or config.LIVE_DIR)
    fetched = None
    ahead = (rows is None) if ahead is None else ahead     # decided BEFORE rows is filled from the page below
    if rows is None:
        rows, fetched = LC.week(now, folder=folder, get=get)
    rels = LC.releases(LC.events(rows))
    if fetched is not None and any(waiting(r, now) for r in rels):
        rows, fetched = LC.week(now, folder=folder, get=get,
                                max_age=dt.timedelta(seconds=config.LIVE_AFTER_REFRESH_SECONDS))
        rels = LC.releases(LC.events(rows))
    p, why = prices_now(now, feed)
    sent = read_sent(folder)
    done = []
    for rel in rels:
        for kind in due(rel, now, sent):
            score = LS.score(rel["head"]) if kind == "after" else None
            # before: where gold and his chart are NOW; after: the situation at the release minute (for the lookup)
            s = _situation(p, pd.Timestamp(rel["server"]) if kind == "after" else (p.at if p is not None else None))
            if s.get("charts_why"):
                log(f"{kind} {rel['key']}: no bias or levels ({s['charts_why']})", folder)
            q = LA.question(rel, score, LSit.for_lookup(s) if kind == "after" else {})
            res = similar.lookup(tab if tab is not None else table(), q)
            text = LA.before_text(rel, res, s) if kind == "before" else LA.after_text(rel, score, res, since(p, rel))
            done.append(_queue(kind, rel["key"], text, now, folder, outbox, paused,
                               release=pd.Timestamp(rel["utc"]).isoformat(), report=rel["report"],
                               z=None if not score or score.get("z") != score.get("z") else round(float(score["z"]), 3),
                               group=(score or {}).get("group")))
            sent.add((kind, rel["key"]))
    if p is not None:
        f = LM.check(p, news_frame(rels), last_move(folder))
        if f is not None:
            done.append(_queue("move", f"{f['bar']:%Y%m%d-%H%M}", LM.text(f), now, folder, outbox, paused,
                               bar=f["bar"].isoformat(), move=round(f["move"], 2), up=f["up"],
                               times_usual=round(f["times_usual"], 2), share_of_day=round(f["share_of_day"], 3)))
    last = _state(folder).get("preview")
    try:
        pv = preview(now, rels, p, last, tab=tab, tell=tell or HA.alert) if ahead else None
    except Exception as e:                                 # the preview never stops a pass
        pv = None
        log(f"the preview could not run: {type(e).__name__}: {str(e)[:200]}", folder)
    if pv is not None:
        log(f"preview {pv['report']} {pv['key']}: " + ("ok" if pv["ok"] else f"FAILED {pv['error']}"), folder)
    write_state(now, fetched, rels, folder, p, why, preview=pv or last)
    return done


def _state(folder: Path) -> dict:
    try:
        path = folder / "state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def write_state(now: dt.datetime, fetched: dt.datetime | None, rels: list[dict], folder: Path,
                p: LP.Prices | None = None, why: str = "", preview: dict | None = None) -> None:
    nxt = next((r for r in rels if pd.Timestamp(r["utc"]).to_pydatetime() > now), None)
    path = folder / "state.json"
    before = _state(folder)
    prices = {"newest_bar": p.m1.index[-1].isoformat(), "fresh": p.fresh(), "source": p.source} if p is not None \
        else {"why_not": why}
    if why and why != (before.get("prices") or {}).get("why_not"):
        log(why, folder)                                   # once when it starts, not every minute
    state = {"at": now.isoformat(timespec="seconds"),
             "fetched": fetched.isoformat(timespec="seconds") if fetched else None,
             "releases_this_week": len(rels),
             "next": None if nxt is None else {"report": nxt["report"], "colour": nxt["colour"],
                                               "utc": pd.Timestamp(nxt["utc"]).isoformat()},
             "prices": prices, "preview": preview}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def day_report(rec: dict, folder: Path | None = None, outbox: Path | None = None,
               paused: Callable[[], bool] = HA.paused, now: dt.datetime | None = None) -> dict:
    """The day's report (machine/live/check.day_report) queued for the bot once - id live-day-<day>; STOP holds it."""
    from machine.live import check
    folder = Path(folder or config.LIVE_DIR)
    return _queue("day", rec["day"], check.day_report(rec, folder), now or _now(), folder, outbox, paused)


def locked_once(**kw) -> list[dict] | None:
    """once() unless another pass holds LIVE_DIR/once.lock (younger than LOCK_STALE): None then. After the pass, the
    daily check of the live feed and, for each day it checks, the day's report to Tejas (neither ever stops the alerts:
    a failure is logged)."""
    folder = Path(kw.get("folder") or config.LIVE_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "once.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - lock.stat().st_mtime
        if age < LOCK_STALE.total_seconds():
            return None
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f"{os.getpid()} {HA.iso(HA.now())}".encode())
        os.close(fd)
        done = once(**kw)
        if not kw.get("rows"):
            try:
                from machine.live import check
                for rec in check.run(folder=folder):
                    log(f"check {check.words(rec)}", folder)
                    try:
                        day_report(rec, folder, kw.get("outbox"), kw.get("paused") or HA.paused)
                    except Exception as e:                 # the report never stops a pass or the next day's check
                        log(f"the day's report failed: {type(e).__name__}: {str(e)[:300]}", folder)
            except guards.SealedYearsError:
                pass                                       # T14 says no: the price part already says so, once
            except Exception as e:
                log(f"the daily check failed: {type(e).__name__}: {str(e)[:300]}", folder)
        return done
    finally:
        lock.unlink(missing_ok=True)


# ======================================================================================================= rehearsal

def as_of(rows: list[dict], now: dt.datetime) -> list[dict]:
    """A past week as the page showed it at `now`: no actual number before its release minute."""
    cut = int(now.timestamp())
    return [{**r, "actual": r.get("actual") if (r.get("dateline") or 0) <= cut else ""} for r in rows]


def history_feed(start: Any, end: Any) -> tuple[Feed, pd.DataFrame]:
    """2016-2023 1-minute bars as the live feed of a rehearsal: at `now`, only the bars CLOSED by then, rupees at that
    month's RBI/FBIL rate. The moves table is made once: move.measure passes the cut-off-the-future test, so its rows up
    to `now` are what a live pass would compute then."""
    from machine import costs
    from machine.data import bars as BARS
    m1 = BARS.load_bars("1m", start, end)[["open", "high", "low", "close", "vol", "spread", "utc"]]
    mv = LM.measure(m1)

    def feed(now: dt.datetime) -> tuple[LP.Prices | None, str]:
        at = LP.server_now(now)
        n = int(m1.index.searchsorted(at - pd.Timedelta(minutes=1), "right"))
        p = LP.Prices(m1.iloc[:n], at, costs.usd_inr(at.strftime("%Y-%m-01")), "2016-2023 history (a rehearsal)",
                      charts_fn=lambda: LP.history_charts(at))
        p._moves = mv.iloc[:n]
        return p, ""
    return feed, mv


def rehearse(sunday: dt.date, *, outbox: Path | None = None, tab: pd.DataFrame | None = None,
             prices: bool = True) -> list[dict]:
    """A 2015-2023 week (data/news/forexfactory/<sunday>.json) replayed minute by minute through once(), into a
    temporary folder and outbox: the alerts and their times, as the live pass would have sent them. No page is read,
    nothing reaches the real outbox; refused for a week reaching 2024 (forexfactory.sundays). prices: the week's own
    1-minute bars play the live feed (gold now / since the number / the big-move alert); False = step 1's alerts."""
    from machine.data import clock
    from machine.news import forexfactory as FF
    FF.sundays(sunday, sunday)                                   # refuses a week reaching the sealed years
    rows = json.loads(FF.week_file(sunday).read_text(encoding="utf-8"))
    start = dt.datetime.combine(sunday, dt.time(0, 0), tzinfo=dt.timezone.utc)
    tab = tab if tab is not None else table()
    minutes = set()
    for s in sorted({int(r["dateline"]) for r in rows if r.get("dateline")}):   # the minutes that can hold an alert
        t = dt.datetime.fromtimestamp(s, dt.timezone.utc)
        for m in range(-config.LIVE_BEFORE_MINUTES, config.LIVE_AFTER_WAIT_MINUTES + 1):
            minutes.add(t + dt.timedelta(minutes=m))
    feed: Feed = lambda now: (None, "a rehearsal without prices")             # noqa: E731
    if prices:
        lo = pd.Timestamp(sunday) - pd.Timedelta(days=config.LIVE_PAST_DAYS)
        feed, mv = history_feed(lo, pd.Timestamp(sunday) + pd.Timedelta(days=7))
        cand = mv.index[LM.big(mv).to_numpy() & ~LM.near_news(mv.index, news_frame(LC.releases(LC.events(rows))))]
        for t in clock.server_to_utc(cand + pd.Timedelta(minutes=1)):          # the pass just after each big bar
            minutes.add(t.to_pydatetime())
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        box = Path(outbox) if outbox else folder / "outbox.jsonl"
        for now in sorted(m for m in minutes if start <= m < start + dt.timedelta(days=7)):
            out += once(now, rows=as_of(rows, now - dt.timedelta(minutes=1)), folder=folder, outbox=box, tab=tab,
                        paused=lambda: False, feed=feed)
    return out


def status_lines(folder: Path | None = None) -> list[str]:
    folder = Path(folder or config.LIVE_DIR)
    path = folder / "state.json"
    if not path.is_file():
        return ["Live news alerts (L17a): no pass yet."]
    s = json.loads(path.read_text(encoding="utf-8"))
    at = dt.datetime.fromisoformat(s["at"])
    lines = [f"Live news alerts (L17a): last pass {HA.india(at):%a %d %b %H:%M} India time; "
             f"{s['releases_this_week']} releases of your news this week."]
    if s.get("next"):
        t = pd.Timestamp(s["next"]["utc"]).tz_convert("Asia/Kolkata")
        lines.append(f"Next: {similar.name_of(s['next']['report'])[1]} ({s['next']['colour']}) {t:%a %d %b %H:%M} "
                     "India time.")
    pv = s.get("preview")
    if pv:
        at = HA.india(dt.datetime.fromisoformat(pv["at"]))
        lines.append(f"Built ahead at {at:%H:%M}: {similar.name_of(pv['report'])[1]} - "
                     + ("OK." if pv.get("ok") else f"FAILED ({str(pv.get('error', ''))[:80]})."))
    pr = s.get("prices") or {}
    if pr.get("newest_bar"):
        from machine.data import clock
        t = HA.india(clock.server_to_utc(pd.DatetimeIndex([pr["newest_bar"]]))[0].to_pydatetime())
        lines.append(f"Live prices (MT5): newest 1-minute bar {t:%a %H:%M} India time - "
                     f"{'live' if pr.get('fresh') else 'no new bar (market shut?)'}.")
    elif pr.get("why_not"):
        lines.append(f"Live prices: none now - {pr['why_not'][:160]}.")
    sent = HA.read_jsonl(folder / "sent.jsonl")
    news = [r for r in sent if r.get("kind") in ("before", "after")]
    if news:
        last = news[-1]
        lines.append(f"Last news alert: {last['kind']} {similar.name_of(last['report'])[1]} - {last['sent']}.")
    moves = [r for r in sent if r.get("kind") == "move"]
    if moves:
        from machine.data import clock
        t = HA.india(clock.server_to_utc(pd.DatetimeIndex([moves[-1]["bar"]]))[0].to_pydatetime())
        lines.append(f"Last big-move alert: {t:%a %d %b %H:%M} India time - {moves[-1]['sent']}.")
    checks = HA.read_jsonl(folder / "checks.jsonl")
    if checks:
        from machine.live import check
        lines.append(f"Daily check of the live prices: {check.words(checks[-1])}.")
    return lines
