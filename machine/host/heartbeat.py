"""The hourly heartbeat (L16.1 step 3): a missed one alerts Tejas.

Task Scheduler runs it every hour (at :15, away from the jobs that start on the hour) and 2 minutes after he logs in
(a power cut ends with a restart). Each beat is one line in data/host/heartbeat.jsonl. A beat:
    1. sends what an earlier beat or run could not (no network then);
    2. MISSED: more than config.HEARTBEAT_LATE_MINUTES since the last beat -> one alert, with what Windows says
       happened in between (machine.host.power: the power went off at 22:50, it started again at 13:06 ...);
    3. checks what must be true now - each a lasting problem: ONE alert when it starts, one when it is over, never
       every hour (data/host/alerts.json):
           bot      the Telegram bot is not running
           daily    today's daily run has not run (2 hours after its time, the PC on for an hour), or is stuck
           disk     less than config.DISK_FREE_MIN_GB free on the PC's drive
           intake   an intake run has worked for more than config.INTAKE_STUCK_HOURS
           live     the live news alerts (L17a) made no pass for config.LIVE_STALE_MINUTES (a pass every minute -
                    stopped, or each pass crashing): no news or big-move alert comes
    4. pings the outside watcher, when .env has HEARTBEAT_PING_URL (a healthchecks.io check: when beats stop coming,
       because the PC is off, IT tells Tejas - this PC cannot, it is off). With a problem open it pings <url>/fail.

When the PC itself is off, nothing on it can send anything: the missed beat is told when it is back on (step 2), and
in real time only by the outside watcher (step 4).
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import time
import urllib.request
from pathlib import Path

from machine import config
from machine.host import alert, lock, power, procs

DAILY_GRACE = dt.timedelta(hours=2)        # today's daily run is "not run" only this long after its time ...
JUST_STARTED = dt.timedelta(hours=1)       # ... and when the PC has been on this long (a missed run starts by itself)
DAILY_STUCK = dt.timedelta(hours=3)        # a daily run working longer than this is stuck
DAILY = "-m machine.host daily"
LIVE_SETTLE = dt.timedelta(minutes=15)     # after the PC starts, the live alerts get this long before they count


def beat(now: dt.datetime | None = None, *, tell=None, sleep=time.sleep, procs_now: list | None = None,
         offs=None, boot: dt.datetime | None = None, free_gb: float | None = None) -> dict | None:
    """One heartbeat. None = another beat is working. procs_now / offs / boot / free_gb: for the tests."""
    now = now or alert.now()
    with lock.one_run(Path(config.HOST_DIR) / "heartbeat.lock") as mine:
        if not mine:
            return None
        alert.resend(tell=tell, sleep=sleep)
        beats = alert.read_jsonl(_beats_file())
        rec: dict = {"at": alert.iso(now)}
        sent = []
        if beats:
            prev = dt.datetime.fromisoformat(beats[-1]["at"])
            gap = (now - prev).total_seconds() / 60
            rec["gap_minutes"] = round(gap, 1)
            if gap > config.HEARTBEAT_LATE_MINUTES:
                found = power.offs_since(prev) if offs is None else offs
                rec["missed"] = {"since": beats[-1]["at"], "minutes": round(gap),
                                 "offs": [_off(o) for o in found]}
                if alert.alert(missed_text(prev, now, found), tell=tell, sleep=sleep, kind="missed"):
                    sent.append("missed")
        else:
            rec["first"] = True
        ps = procs.processes() if procs_now is None else procs_now
        boot = boot if boot is not None else power.boot_time(now)
        free = free_gb if free_gb is not None else shutil.disk_usage(config.ROOT).free / 1e9
        problems = check(now, ps, boot, free)
        rec.update({"bot": bool(procs.bot(ps)) if ps else None,          # None = Windows could not be asked
                    "disk_free_gb": round(free, 1), "problems": sorted(problems)})
        sent += _tell_changes(problems, now, tell=tell, sleep=sleep)
        rec["ping"] = ping(bool(problems))
        rec["sent"] = sent
        alert.append_jsonl(_beats_file(), rec)
        alert.log("heartbeat", f"beat: problems {sorted(problems) or 'none'}"
                  + (f", MISSED ({rec['missed']['minutes']} min since the last beat)" if "missed" in rec else "")
                  + f", ping {rec['ping']}")
        return rec


def check(now: dt.datetime, ps: list | None, boot: dt.datetime | None, free_gb: float) -> dict[str, str]:
    """The lasting problems now: key -> the alert's text. ps None or empty = Windows could not be asked (a real answer
    holds this program itself): then nothing is said about the bot or an idea's run."""
    out = {}
    ps = ps or []
    if ps and not procs.bot(ps):
        out["bot"] = ("⚠️ The Telegram bot is not running. Your messages wait safely on Telegram; it files them when "
                      "it starts. It starts by itself when you log in to the PC. To start it now, in PowerShell: "
                      "Start-ScheduledTask -TaskName 'ML Bot - the Telegram bot (starts at logon)'")
    d = daily_problem(now, ps, boot)
    if d:
        out["daily"] = d
    if free_gb < config.DISK_FREE_MIN_GB:
        out["disk"] = (f"⚠️ The PC's drive has only {free_gb:.1f} GB free (less than {config.DISK_FREE_MIN_GB} GB). "
                       "The daily run and the tests may stop when it is full.")
    stuck = [p for p in (procs.intake(ps) or []) if p.started
             and now - p.started > dt.timedelta(hours=config.INTAKE_STUCK_HOURS)]
    if stuck:
        out["intake"] = (f"⚠️ An idea's run has been working since {alert.india(stuck[0].started):%a %H:%M} (more than "
                         f"{config.INTAKE_STUCK_HOURS} hours). It may be stuck; no other idea can move while it runs. "
                         "A Claude session can look at data/intake/intake.log.")
    lv = live_problem(now, boot)
    if lv:
        out["live"] = lv
    return out


def live_problem(now: dt.datetime, boot: dt.datetime | None) -> str | None:
    """The live news alerts (L17a) write data/live/state.json at the end of every pass, every minute. Older than
    config.LIVE_STALE_MINUTES - while the PC has been on 15 minutes or more - means they stopped or every pass fails.
    Never installed (no state.json): nothing to say."""
    path = Path(config.LIVE_DIR) / "state.json"
    try:
        at = dt.datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["at"])
    except (OSError, ValueError, KeyError):
        return None
    if (boot and now - boot < LIVE_SETTLE) or now - at <= dt.timedelta(minutes=config.LIVE_STALE_MINUTES):
        return None
    return (f"⚠️ The live news alerts have not run since {alert.india(at):%a %H:%M} (they run every minute). No news "
            "alert and no big-move alert comes until they run again. A Claude session can look at data/live/live.log "
            "and data/live/console.log.")


def daily_problem(now: dt.datetime, ps: list, boot: dt.datetime | None) -> str | None:
    running = procs.programs(ps, DAILY)
    if running and running[0].started and now - running[0].started > DAILY_STUCK:
        return (f"⚠️ Today's daily run started at {alert.india(running[0].started):%H:%M} and has not finished "
                f"(more than {int(DAILY_STUCK.total_seconds() // 3600)} hours). It may be stuck.")
    runs = alert.read_jsonl(Path(config.HOST_DIR) / "daily.jsonl")
    if not runs or running:
        return None                                    # never ran yet (not installed), or running now
    local = alert.india(now)
    hh, mm = (int(x) for x in config.DAILY_AT.split(":"))
    due = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if local < due + DAILY_GRACE or (boot and now - boot < JUST_STARTED):
        return None
    if any(alert.india(dt.datetime.fromisoformat(r["at"])).date() == local.date() for r in runs):
        return None
    return (f"⚠️ Today's daily run has not run (it should have at {config.DAILY_AT}"
            + (f"; the PC has been on since {alert.india(boot):%H:%M}" if boot else "") + ").")


def missed_text(prev: dt.datetime, now: dt.datetime, offs: list) -> str:
    mins = int((now - prev).total_seconds() // 60)
    span = f"{mins // 60} hour{'s' if mins // 60 != 1 else ''} {mins % 60} min" if mins >= 60 else f"{mins} min"
    a, b = alert.india(prev), alert.india(now)
    head = (f"⚠️ The heartbeat was missed: no beat from {a:%a %H:%M} to {b:%a %H:%M} ({span}; one comes every "
            f"hour).")
    if offs:
        from machine.host.daily import off_text
        said = "; ".join(off_text(o) for o in offs[-3:])
        return (head + f"\nWindows says the PC was off: {said}.\nWhile it was off nothing ran: no daily run, no "
                "ideas, no alerts. Everything goes on by itself now.")
    return (head + "\nWindows says the PC did NOT stop, so the heartbeat itself did not run (Task Scheduler). A Claude "
            "session should look at data/host/heartbeat.log.")


def _tell_changes(problems: dict[str, str], now: dt.datetime, *, tell=None, sleep=time.sleep) -> list[str]:
    """One alert when a problem starts, one when it is over (data/host/alerts.json remembers what is open)."""
    path = Path(config.HOST_DIR) / "alerts.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {"open": {}}
    sent = []
    for key, text in problems.items():
        if key not in state["open"]:
            ok = alert.alert(text, tell=tell, sleep=sleep, kind=key)
            state["open"][key] = {"since": alert.iso(now), "sent": ok}
            if ok:
                sent.append(key)
    for key in [k for k in state["open"] if k not in problems]:
        since = alert.india(dt.datetime.fromisoformat(state["open"].pop(key)["since"]))
        over = {"bot": f"✅ The Telegram bot is running again (it was not, from {since:%a %H:%M}).",
                "disk": "✅ The PC's drive has enough free space again.",
                "intake": "✅ The long idea run has finished.",
                "live": f"✅ The live news alerts run again (they did not, from {since:%a %H:%M}).",
                "daily": None}.get(key)                # the daily run's own summary says it ran
        if over and alert.alert(over, tell=tell, sleep=sleep, kind=f"{key}-over"):
            sent.append(f"{key}-over")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)
    return sent


def ping(problem: bool, *, opener=urllib.request.urlopen) -> str:
    url = config.env("HEARTBEAT_PING_URL")
    if not url:
        return "no outside watcher (HEARTBEAT_PING_URL is not in .env)"
    try:
        with opener(url.rstrip("/") + ("/fail" if problem else ""), timeout=10) as r:
            return f"ok ({r.status})" + (" /fail" if problem else "")
    except Exception as e:
        return f"failed ({type(e).__name__})"


def _off(o: power.Off) -> dict:
    return {"off": o.off_at.isoformat() if o.off_at else None, "on": o.on_at.isoformat(), "why": o.why}


def _beats_file() -> Path:
    return Path(config.HOST_DIR) / "heartbeat.jsonl"
