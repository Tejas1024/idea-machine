"""What the dashboard shows, read from the machine's own files (L16.2). Read only: nothing here writes, sends or
starts anything.

READS is the whole list of files the page may open; tests/test_dashboard.py opens the page with Python's audit hook on
and fails on any other file. Never read: the live alerts' folder (data/live: live prices), the sealed prices
(data/sealed), the bot's outbox and its sent messages (live alert texts), his messages, .env. Free text goes through
machine/dashboard/mask.py first: a text the live alerts can write loses every digit; any other text loses every number
shaped like a gold price (hard rule 1: never a 2024+ price on the page).

In Docker the page cannot ask Windows anything (programs, Task Scheduler, the power log): it shows what the heartbeat
last wrote down, with that beat's time. Times are India time.
"""
from __future__ import annotations

import collections
import datetime as dt
import fnmatch
import json
import os
import re
import threading
import time
from pathlib import Path

from machine import config
from machine.dashboard import mask

INDIA = dt.timezone(dt.timedelta(hours=5, minutes=30))
CACHE_SECONDS = 30          # the page is read again at most this often (it refreshes every minute)
RECENT_DAYS = 7             # daily runs, missed beats and errors this far back
STRIP_HOURS = 24            # the heartbeat strip: one box per hour
SHOWN = 20                  # at most this many alerts and errors

# The only files the page reads: (the config folder, the file pattern inside it).
READS = (
    ("IDEAS_DIR", "*/state.json"), ("IDEAS_DIR", "*/form.json"), ("IDEAS_DIR", "*/versions.jsonl"),
    ("IDEAS_DIR", "*/check-opened.json"), ("IDEAS_DIR", "*/decision.md"),
    ("TRIALS_DIR", "*.json"),
    ("WORK_DIR", "intake-queue.jsonl"), ("WORK_DIR", "intake/*"), ("WORK_DIR", "block-requests.md"),
    ("HOST_DIR", "daily.jsonl"), ("HOST_DIR", "heartbeat.jsonl"), ("HOST_DIR", "alerts.json"),
    ("HOST_DIR", "alerts.log"), ("HOST_DIR", "unsent.jsonl"),
    ("DATA_DIR", "intake/intake.log"),
    ("TELEGRAM_DIR", "state.json"), ("TELEGRAM_DIR", "nightly.jsonl"), ("TELEGRAM_DIR", "bot.log"),
)

STEP_NAMES = {"prices": "Prices kept", "caches": "Price tables", "ideas": "Ideas", "questions": "Weekly questions",
              "mlflow": "Tests recorded"}
PROBLEMS = {"bot": "the Telegram bot is not running", "daily": "today's daily run did not run, or is stuck",
            "disk": "the PC's drive is nearly full", "intake": "an idea's run seems stuck",
            "live": "the live news alerts stopped"}
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(\S*) ?(.*)$")
ANSWER_WORDS = ("works", "chance", "chance — needs fresh years", "not enough trades to tell")


def allowed(path) -> bool:
    """True when the page may open this file (READS)."""
    p = Path(path).resolve()
    for base, pattern in READS:
        try:
            rel = p.relative_to(Path(getattr(config, base)).resolve()).as_posix()
        except ValueError:
            continue
        if fnmatch.fnmatchcase(rel, pattern):
            return True
    return False


# ============================================================================================================ reading

def now() -> dt.datetime:
    return dt.datetime.now(INDIA)


def india(t: dt.datetime) -> dt.datetime:
    return (t if t.tzinfo else t.replace(tzinfo=INDIA)).astimezone(INDIA)


def when(iso: str | None) -> dt.datetime | None:
    """An ISO time from the machine's files (with its offset; the logs' plain times are India time)."""
    try:
        return india(dt.datetime.fromisoformat(str(iso)))
    except (TypeError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    out = []
    for line in _text(path).splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                                  # a line being written right now
    return out


def _json(path: Path) -> dict:
    try:
        return json.loads(_text(path)) or {}
    except ValueError:
        return {}


def _text(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _entries(path: Path) -> list[tuple[dt.datetime, str]]:
    """A log's entries: (its time, its text). A line without a time belongs to the entry above it (a traceback), and
    so does a timed line that starts with spaces (machine/host/alert.log writes a message's next lines that way)."""
    out: list[tuple[dt.datetime, str]] = []
    for line in _text(path).splitlines():
        m = STAMP.match(line)
        t = when(m.group(1) + m.group(2)) if m else None
        if t is not None and not (out and m.group(3)[:1].isspace()):
            out.append((t, m.group(3)))
        elif out and line.strip():
            more = m.group(3).strip() if t is not None else line.strip()
            out[-1] = (out[-1][0], out[-1][1] + "\n" + more)
    return out


# ============================================================================================================ the parts

def ideas(t: dt.datetime) -> dict:
    """His ideas (idea-001 ...): the words, the state, since when, what each waits for; the queue; a pause."""
    from machine.ideas import blocked, store
    from machine.intake import state as intake
    rows = []
    b_lines = {ln.split(" ", 1)[0]: ln for ln in blocked.status_lines(t)}
    try:
        run_now = intake.current()
    except ValueError:                                    # a broken state.json: its row below says so
        run_now = None
    for idea in store.ideas():
        form = _json(Path(config.IDEAS_DIR) / idea / "form.json")
        try:
            s = store.read_state(idea)
        except store.StoreError:
            s = {"state": "RECEIVED" if not form else "FORM DRAFTED", "since": None}
        except ValueError as e:                           # a broken state.json: this row says so, the others show
            s = {"state": "?", "since": None, "waits_for": f"its state.json cannot be read ({type(e).__name__})"}
        wait = s.get("waits_for") or ("nothing" if s["state"] in intake.FINISHED else "the machine")
        if s["state"] not in ("BLOCKED", *intake.FINISHED) and run_now and idea != run_now:
            wait = f"its turn ({intake.label(run_now)} runs now)"
        if s["state"] == "BLOCKED" and idea in b_lines:
            wait = b_lines[idea].split(": ", 1)[-1]
        words = form.get("words") or (s.get("intake") or {}).get("words") or ""
        rows.append({"id": idea, "name": intake.label(idea), "title": mask.prices(form.get("title") or "", years=False),
                     "words": mask.prices(" ".join(str(words).split())[:220], years=False),
                     "state": s["state"], "since": s.get("since"), "waits": mask.prices(wait),
                     "decided": _decision(idea) if s["state"] == "DECIDED" else None,
                     "needs_session": bool(s.get("needs")), "running": idea == run_now})
    try:
        q = intake.pending()
    except ValueError:
        q = []
    return {"rows": rows, "queue": len(q), "queue_since": q[0].get("at") if q else None,
            "paused": mask.prices(intake.paused() or "") or None}


def versions(t: dt.datetime) -> dict:
    """Per idea: the versions tried of the family's 20, the referee's choice on 2016-2020, the answer on the check
    years - the same functions the reports use (machine/agent/research.every_version, machine/mcp/versions.family)."""
    from machine.agent import research
    from machine.ideas import store
    from machine.mcp import versions as V
    from machine.referee import project, trials
    files = trials.load()
    rows = []
    for idea in store.ideas():
        row: dict = {"id": idea}
        try:
            form = store.load_form(idea)
        except Exception:                                 # RECEIVED: no form yet, so no version either
            rows.append({**row, "tried": 0, "waiting": 0, "budget": config.VERSION_BUDGET, "chosen": None,
                         "answer": None, "hedge": None})
            continue
        try:
            parent, _ = V.parent_of(idea, form, files)
            fam = V.family(idea, parent, files)
            every = research.every_version(idea)
        except Exception as e:                            # e.g. two parents named: refused, never guessed
            rows.append({**row, "tried": None, "waiting": 0, "budget": config.VERSION_BUDGET, "chosen": None,
                         "answer": None, "hedge": None,
                         "failed": mask.every_digit(f"{type(e).__name__}: {e}")[:200]})
            continue
        pick = next((r for r in every if r["chosen"]), None)
        row.update(tried=len(fam["tried"]), waiting=len(fam["waiting"]), budget=fam["budget"],
                   family=fam["root"] if fam["root"] != idea else None, chosen=_version(pick, every))
        judged = [f for f in files if f.get("kind") == "referee" and f.get("idea_id") in fam["ideas"]]
        row["answer"] = _answer(judged[-1]) if judged else None
        hedged = [f for f in files if f.get("kind") == "hedging-2x2" and f.get("idea_id") in fam["ideas"]]
        row["hedge"] = _hedge(hedged[-1]) if hedged else None
        rows.append(row)
    newest = {f["idea_id"]: f for f in files                # his whole strategy (L19d): its newest answer
              if f.get("kind") == "referee" and not re.fullmatch(r"idea-\d{3,}", str(f.get("idea_id")))}
    others = [_answer(f, name=name) for name, f in newest.items()]
    kinds = collections.Counter(f.get("kind") for f in files)
    return {"rows": rows, "others": others, "files": len(files), "kinds": dict(kinds.most_common()),
            "project": project.summary()}


def _version(pick: dict | None, every: list[dict]) -> dict | None:
    if pick is None:
        return None
    return {"version": pick["version"], "of": len(every), "about": mask.prices(pick.get("about") or "")[:200],
            "trades": pick.get("trades"), "per_trade": pick.get("per_trade"), "total": pick.get("total"),
            "worst_year": pick.get("worst_year"), "hedged": bool(pick.get("hedged"))}


def _answer(f: dict, name: str | None = None) -> dict:
    """The referee's answer in one trial file. An idea's file says net_inr / per_trade; his whole strategy's (L19d,
    hedged) says net_inr_total / net_inr_per_trade."""
    r = f.get("result") or {}
    a = r.get("answer") or {}
    s = r.get("summary") or {}
    years = sorted(str(y) for y in (r.get("per_year") or {}))
    return {"name": name, "word": a.get("word"), "why": mask.prices(a.get("why") or ""), "made_at": f.get("made_at"),
            "trades": s.get("trades"), "total": _first(s, "net_inr", "net_inr_total"),
            "per_trade": _first(s, "per_trade", "net_inr_per_trade"),
            "worst_year": r.get("worst_year") or s.get("worst_year"),
            "years": f"{years[0]}-{years[-1]}" if years else None}


def _hedge(f: dict) -> dict:
    """His hedge instead of a stop on the idea's chosen version (L19a / L19d, kind hedging-2x2): 2016-2020 only."""
    r = f.get("result") or {}
    h = r.get("his_hedging") or {}
    years = sorted(str(y) for y in r.get("years") or [])
    per = h.get("per_year") or {}
    worst = min(per.items(), key=lambda kv: kv[1].get("net_inr", 0)) if per else None
    return {"trades": h.get("trades"), "total": h.get("net_inr_total"), "per_trade": h.get("net_inr_per_trade"),
            "worst_year": {"year": worst[0], "net_inr": worst[1].get("net_inr")} if worst else None,
            "beats": (r.get("his_beats_random") or {}).get("with_hedging"), "runs": r.get("runs"),
            "choice": mask.prices(r.get("choice") or "") or None, "years": f"{years[0]}-{years[-1]}" if years else None,
            "made_at": f.get("made_at")}


def _first(d: dict, *keys):
    return next((d[k] for k in keys if d.get(k) is not None), None)


def _decision(idea: str) -> str | None:
    """The machine's choice after the report, in decision.md's own words: the bold line under "## The choice", then
    a later "**The machine's choice: ...**" line if a test came after (L19d)."""
    text = _text(Path(config.IDEAS_DIR) / idea / "decision.md")
    if not text:
        return None
    first = re.search(r"^## The choice\s*\n+\*\*(.+?)\*\*", text, re.M)
    later = re.findall(r"^\*\*The machine's choice: (.+?)\*\*", text, re.M)
    words = (first.group(1).strip() if first else "") + (f" Then: {later[-1].strip()}." if later else "")
    return mask.prices(words)[:240] or None


def runs(t: dt.datetime) -> dict:
    """The last daily runs (L16.1), the intake's last changes, the nightly copy to GitHub (L9c), the bot's start."""
    host = Path(config.HOST_DIR)
    since = t - dt.timedelta(days=RECENT_DAYS)
    daily = []
    for r in _jsonl(host / "daily.jsonl"):
        at = when(r.get("at"))
        if at is None or at < since:
            continue
        daily.append({"at": at.isoformat(), "ok": bool(r.get("ok")), "test": r.get("forced"),
                      "finished": r.get("finished"),
                      "steps": [{"name": STEP_NAMES.get(s.get("step"), str(s.get("step"))), "ok": bool(s.get("ok")),
                                 "line": mask.prices(s.get("line") or "")[:300], "seconds": s.get("seconds")}
                                for s in r.get("steps") or []]})
    intake = [{"at": at.isoformat(), "text": mask.prices(text.split("\n", 1)[0])[:240]}
              for at, text in _entries(Path(config.DATA_DIR) / "intake" / "intake.log")[-8:]]
    nightly = [{"at": (when(n.get("at")) or t).isoformat(), "branch": mask.prices(n.get("branch") or ""),
                "files": len(n.get("files") or []), "link": bool(n.get("pr") or n.get("link")),
                "error": mask.every_digit(n.get("error"))[:240] if n.get("error") else None,
                "test": bool(n.get("dry"))}
               for n in _jsonl(Path(config.TELEGRAM_DIR) / "nightly.jsonl")[-3:]]
    bot = _entries(Path(config.TELEGRAM_DIR) / "bot.log")
    started = [at for at, text in bot if text.startswith("bot started")]
    drops = [at for at, text in bot if text.startswith("Telegram not reachable") and at >= t - dt.timedelta(hours=24)]
    tg = _json(Path(config.TELEGRAM_DIR) / "state.json")
    return {"daily": daily[::-1], "intake": intake[::-1], "nightly": nightly[::-1],
            "bot_started": started[-1].isoformat() if started else None, "drops_24h": len(drops),
            "last_drop": drops[-1].isoformat() if drops else None,
            "sending": tg.get("sending", True) is not False if tg else None}


def heartbeat(t: dt.datetime) -> dict:
    """The last beat, one box per hour for the last day, the missed beats with Windows' reason, the open problems."""
    host = Path(config.HOST_DIR)
    beats = []
    for b in _jsonl(host / "heartbeat.jsonl"):
        at = when(b.get("at"))
        if at is not None:
            beats.append((at, b))
    last = None
    if beats:
        at, b = beats[-1]
        last = {"at": at.isoformat(), "bot": b.get("bot"), "disk_free_gb": b.get("disk_free_gb"),
                "problems": [str(p) for p in b.get("problems") or []], "ping": mask.prices(b.get("ping") or "")}
    hour = t.replace(minute=0, second=0, microsecond=0)
    first = beats[0][0] if beats else None
    strip = []
    for k in range(STRIP_HOURS - 1, -1, -1):
        start = hour - dt.timedelta(hours=k)
        n = sum(start <= at < start + dt.timedelta(hours=1) for at, _ in beats)
        if n:
            kind = "beat"
        elif first is None or start + dt.timedelta(hours=1) <= first:
            kind = "before"                               # before the first beat ever
        elif k == 0:
            kind = "waiting"                              # this hour's beat comes at :15
        else:
            kind = "none"
        strip.append({"from": start.isoformat(), "beats": n, "kind": kind})
    since = t - dt.timedelta(days=RECENT_DAYS)
    missed = []
    for at, b in beats:
        m = b.get("missed")
        if m and at >= since:
            missed.append({"at": at.isoformat(), "since": (when(m.get("since")) or at).isoformat(),
                           "minutes": m.get("minutes"),
                           "offs": [{"off": o.get("off"), "on": o.get("on"), "why": mask.prices(o.get("why") or "")}
                                    for o in m.get("offs") or []][-3:]})
    open_ = (_json(host / "alerts.json").get("open") or {})
    problems = [{"key": k, "what": PROBLEMS.get(k, mask.every_digit(k)), "since": (v or {}).get("since"),
                 "told": (v or {}).get("sent")} for k, v in open_.items()]
    return {"last": last, "count": len(beats), "strip": strip, "missed": missed[::-1], "open": problems,
            "unsent": len(_jsonl(host / "unsent.jsonl")), "late_minutes": config.HEARTBEAT_LATE_MINUTES}


def errors(t: dt.datetime, done: dict) -> list[dict]:
    """The newest alerts and errors of the last days: the alerts sent to him (their digits masked: the live alerts
    write there too), failed daily steps, an idea's step that stopped, a failed nightly copy, Telegram drops."""
    since = t - dt.timedelta(days=RECENT_DAYS)
    out = []
    for at, text in _entries(Path(config.HOST_DIR) / "alerts.log"):
        if at < since:
            continue
        m = re.match(r"(sent|NOT SENT YET|NOT SENT)\s*(\(([^)]*)\))?:?\s*(.*)", text, re.S)
        how, kind, body = (m.group(1), m.group(3) or "", m.group(4)) if m else ("", "", text)
        where = {"sent": "message sent to you", "NOT SENT": "message NOT sent", "NOT SENT YET": "message waits"}
        out.append({"at": at.isoformat(), "where": where.get(how, "alerts"),
                    "kind": mask.every_digit(kind)[:60], "text": mask.every_digit(" ".join(body.split()))[:400]})
    for r in (done.get("runs") or {}).get("daily") or []:
        for s in r["steps"]:
            if not s["ok"] and not r["test"]:
                out.append({"at": r["at"], "where": "daily run", "kind": s["name"], "text": s["line"]})
    for at, text in _entries(Path(config.DATA_DIR) / "intake" / "intake.log"):
        head = text.split("\n", 1)[0]
        if at >= since and (" stopped" in head or "Traceback" in text or "needs a" in head):
            out.append({"at": at.isoformat(), "where": "ideas", "kind": "", "text": mask.prices(head)[:300]})
    for n in (done.get("runs") or {}).get("nightly") or []:
        if n["error"]:
            out.append({"at": n["at"], "where": "nightly copy to GitHub", "kind": "", "text": n["error"]})
    r = done.get("runs") or {}
    if r.get("drops_24h"):
        out.append({"at": r["last_drop"], "where": "Telegram bot", "kind": "",
                    "text": f"Telegram could not be reached {r['drops_24h']} time(s) in the last 24 hours. The bot "
                            "tries again by itself; nothing is lost."})
    out.sort(key=lambda e: e["at"] or "", reverse=True)
    return out[:SHOWN]


# ============================================================================================================ right now

def right_now(s: dict, t: dt.datetime) -> list[dict]:
    """The six boxes at the top: good / warning / bad / unknown, each with one line."""
    hb, rn = s.get("heartbeat") or {}, s.get("runs") or {}
    last = hb.get("last")
    tiles = []

    beat_at = when(last["at"]) if last else None
    open_keys = {p["key"] for p in hb.get("open") or []}
    if not last:
        tiles.append(_tile("Heartbeat", "unknown", "No heartbeat on this computer.",
                           "It runs on Tejas's PC every hour (L16.1)."))
    else:
        late = (t - beat_at).total_seconds() / 60 > hb.get("late_minutes", 90)
        tiles.append(_tile("Heartbeat", "bad" if late else "good",
                           f"Last beat {_hm(beat_at, t)}" + (" - LATE" if late else ""),
                           "One comes every hour. Late = the PC was off, or the heartbeat stopped." if late
                           else "One comes every hour, at :15."))

    if last is None or last.get("bot") is None:
        tiles.append(_tile("Telegram bot", "unknown", "Not known", "Only the heartbeat on Tejas's PC can see it."))
    elif not last["bot"]:
        tiles.append(_tile("Telegram bot", "bad", "NOT running", f"At the beat {_hm(beat_at, t)}. It starts when "
                           "Tejas logs in to the PC."))
    elif rn.get("sending") is False:
        tiles.append(_tile("Telegram bot", "warning", "Running, NOT sending", "Tejas typed STOP. START sends again."))
    else:
        started = when(rn.get("bot_started"))
        tiles.append(_tile("Telegram bot", "good", "Running", f"At the beat {_hm(beat_at, t)}"
                           + (f"; started {_hm(started, t)}" if started else "") + "."))

    tiles.append(_daily_tile(rn.get("daily") or [], t, "daily" in open_keys))

    if not last:
        tiles.append(_tile("Live news alerts", "unknown", "Not known", "They run on Tejas's PC every minute (L17a)."))
    elif "live" in open_keys or "live" in (last.get("problems") or []):
        since = next((when(p["since"]) for p in hb["open"] if p["key"] == "live"), None)
        tiles.append(_tile("Live news alerts", "bad", "STOPPED", (f"Since {_hm(since, t)}. " if since else "")
                           + "No news alert and no big-move alert comes."))
    else:
        tiles.append(_tile("Live news alerts", "good", "Running", f"At the beat {_hm(beat_at, t)}. Their prices "
                           "are never shown here."))

    nightly = rn.get("nightly") or []
    if not nightly:
        tiles.append(_tile("Nightly copy to GitHub", "unknown", "No run on this computer", "It runs at 23:30 on "
                           "Tejas's PC (L9c)."))
    else:
        n = nightly[0]
        at = when(n["at"])
        old = at is not None and (t - at) > dt.timedelta(hours=26)
        state = "bad" if n["error"] else ("warning" if old else "good")
        what = "FAILED" if n["error"] else (f"{n['files']} file(s)" + (", link sent to Tejas" if n["link"] else "")
                                            if n["files"] else "nothing new")
        tiles.append(_tile("Nightly copy to GitHub", state, what, f"{_hm(at, t)}"
                           + (" - not run last night (the PC was off?)" if old and not n["error"] else "") + "."))

    probs = hb.get("open") or []
    unsent = hb.get("unsent") or 0
    if probs or unsent:
        line = "; ".join(p["what"] for p in probs) or ""
        if unsent:
            line = (line + "; " if line else "") + f"{unsent} message(s) wait for the network"
        tiles.append(_tile("Open problems", "bad" if probs else "warning", f"{len(probs) + bool(unsent)}", line + "."))
    else:
        tiles.append(_tile("Open problems", "good" if last else "unknown", "None" if last else "Not known",
                           "What the heartbeat watches: the bot, the daily run, the drive, a stuck idea run, "
                           "the live alerts."))
    return tiles


def _daily_tile(daily: list[dict], t: dt.datetime, open_problem: bool) -> dict:
    real = [r for r in daily if not r["test"]]
    today = [r for r in real if when(r["at"]).date() == t.date()]
    hh, mm = (int(x) for x in config.DAILY_AT.split(":"))
    due = t.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if today:
        r = today[0]
        bad = [s["name"] for s in r["steps"] if not s["ok"]]
        return _tile("Daily run", "bad" if bad else "good", "FAILED: " + ", ".join(bad) if bad else "All steps worked",
                     f"Today {_hm(when(r['at']), t)}, {len(r['steps'])} steps.")
    if not daily:
        return _tile("Daily run", "unknown", "No run on this computer", f"It runs at {config.DAILY_AT} on Tejas's PC "
                     "(L16.1).")
    last = real[0] if real else daily[0]
    if open_problem or t >= due + dt.timedelta(hours=2):
        return _tile("Daily run", "bad", "NOT run today", f"It should run at {config.DAILY_AT}. Last: "
                     f"{_hm(when(last['at']), t)}.")
    return _tile("Daily run", "waiting", f"Today at {config.DAILY_AT}", f"Last: {_hm(when(last['at']), t)}, "
                 + ("all steps worked." if last["ok"] else "it FAILED."))


def _tile(name: str, state: str, value: str, line: str) -> dict:
    return {"name": name, "state": state, "value": value, "line": line}


def _hm(at: dt.datetime | None, t: dt.datetime) -> str:
    """'13:15' today, 'Mon 23:30' this week, '28 Sep 23:30' before."""
    if at is None:
        return "?"
    at = india(at)
    if at.date() == t.date():
        return f"{at:%H:%M}"
    if (t.date() - at.date()).days < 7:
        return f"{at:%a %H:%M}"
    return f"{at:%d %b %H:%M}"


# ============================================================================================================ all of it

_cache: dict = {}
_lock = threading.Lock()
_making = threading.Lock()                     # at most one reading made in the background at a time
TOO_OLD_SECONDS = 300                          # nobody looked for this long: read again before showing


def latest() -> dict:
    """For the server: the last reading at once. Older than CACHE_SECONDS: a new one is made in the background, and
    the next load (the page reloads every minute) shows it - in Docker one reading takes seconds (about 1,100 small
    files through Docker's file sharing). Older than TOO_OLD_SECONDS, or none yet: read now."""
    age = time.monotonic() - _cache["at"] if _cache else None
    if age is None or age > TOO_OLD_SECONDS:
        return snapshot()
    if age > CACHE_SECONDS and _making.acquire(blocking=False):
        def make() -> None:
            try:
                snapshot(fresh=True)
            finally:
                _making.release()
        threading.Thread(target=make, daemon=True).start()
    return _cache["snap"]


def snapshot(t: dt.datetime | None = None, *, fresh: bool = False) -> dict:
    """Everything the page shows, as plain data (also the page's /status.json). One part that cannot be read says so;
    the others still show. Kept for CACHE_SECONDS (the page refreshes every minute)."""
    with _lock:
        if not fresh and t is None and _cache and time.monotonic() - _cache["at"] < CACHE_SECONDS:
            return _cache["snap"]
        t = india(t) if t else now()
        out: dict = {"made_at": t.isoformat(timespec="seconds"),
                     "root": os.environ.get("DASHBOARD_FILES") or str(config.ROOT)}
        for name, fn in (("ideas", ideas), ("versions", versions), ("runs", runs), ("heartbeat", heartbeat)):
            try:
                out[name] = fn(t)
            except Exception as e:                        # one bad file never takes the page down
                out[name] = {"failed": mask.every_digit(f"{type(e).__name__}: {e}")[:300]}
        try:
            out["errors"] = errors(t, out)
        except Exception as e:
            out["errors"] = [{"at": t.isoformat(), "where": "this page", "kind": "",
                              "text": mask.every_digit(f"could not read the logs: {type(e).__name__}: {e}")[:300]}]
        out["now"] = right_now(out, t)
        _cache.update(at=time.monotonic(), snap=out)
        return out
