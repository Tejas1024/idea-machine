"""The daily run (L16.1 step 1). Task Scheduler starts it every day at config.DAILY_AT; a day missed while the PC was
off runs as soon as it is on again.

Five steps, in this order, each one on its own - a step that fails is sent to Tejas AT ONCE (an alert), and the other
steps still run:
    prices     MT5 -> data/sealed/mt5/, complete server days only, never parsed (machine.host.prices)
    caches     price tables ready for today's code, old ones removed, the document search refreshed (machine.host.caches)
    ideas      the intake (L10b) started as its own program - it does nothing when a run is working (the bot starts it
               every 30 minutes too); the ideas' states go into the summary
    questions  when the weekly batch went and may go next (the intake sends it on its day - CLAUDE.md rule 16)
    mlflow     every trial file in MLflow (machine.host.mlflow_sync)
Then ONE summary to Tejas: through the bot's outbox, or straight to Telegram when the bot is not running.

One run at a time (data/host/daily.lock). Every run is one line in data/host/daily.jsonl; the log is data/host/daily.log.
    python -m machine.host daily --force-fail mlflow    a TEST: that step fails on purpose (the alert says so)
"""
from __future__ import annotations

import datetime as dt
import time
import traceback
from pathlib import Path

from machine import config
from machine.host import alert, caches, lock, mlflow_sync, power, prices, procs

STEPS = ("prices", "caches", "ideas", "questions", "mlflow")
TITLES = {"prices": "Prices", "caches": "Caches", "ideas": "Ideas", "questions": "Weekly questions",
          "mlflow": "Tests recorded"}


class ForcedFailure(RuntimeError):
    """A step made to fail on purpose, to test that a failure reaches Tejas (L16.1 Done when)."""


# ================================================================================================= the steps

def step_prices(now: dt.datetime) -> tuple[str, dict]:
    out = prices.store()
    return prices_line(out), out


def prices_line(out: dict) -> str:
    m1 = (out.get("timeframes") or {}).get("M1") or {}
    days = m1.get("days") or []
    waited = m1.get("waited_s") or 0
    if m1.get("no_bar_today"):
        late = (f" MT5 still had no bar of today after {waited // 60} minutes of waiting (still loading after the PC "
                "started, or gold is shut today): yesterday waits for the next daily run.")
    else:
        late = f" MT5 was still loading: waited {waited} seconds for it." if waited else ""
    if not days:
        return ("Prices: nothing new to keep (a day is kept once the next day's prices exist; the market may be "
                "closed)." + late)
    first, last = (dt.date.fromisoformat(days[0]["day"]), dt.date.fromisoformat(days[-1]["day"]))
    when = f"{last:%a %d %b}" if first == last else f"{first:%d %b} to {last:%d %b}"
    rows = sum(d["rows"] for d in days)
    return (f"Prices: {len(days)} day{'s' if len(days) != 1 else ''} kept ({when}), {rows:,} one-minute rows - "
            "sealed, nobody opens them." + late)


def step_caches(now: dt.datetime) -> tuple[str, dict]:
    out = caches.tidy()
    return caches.line(out), out


def step_ideas(now: dt.datetime) -> tuple[str, dict]:
    from machine.ideas import store
    from machine.intake import state as S
    if not S.spawn():
        raise RuntimeError("could not start the intake program (python -m machine.intake run)")
    parts, blocked = [], 0
    for idea in store.ideas():
        try:
            s = S.state_of(idea)
        except store.StoreError:
            continue
        if s["state"] == "BLOCKED":
            blocked += 1
        elif s["state"] not in S.FINISHED:
            wait = " ".join(str(s.get("waits_for") or "the machine").split())
            parts.append(f"{S.label(idea)} is {s['state'].lower()} and waits for {_short(wait, 110)}")
    queue = len(S.pending())
    line = "Ideas: " + ("; ".join(parts[:4]) if parts else "none running")
    line += f". {blocked} wait for a block." if blocked else "."
    line += f" New ideas waiting: {queue}." if queue else " No new idea waits."
    return line, {"running": len(parts), "blocked": blocked, "queue": queue}


def step_questions(now: dt.datetime) -> tuple[str, dict]:
    from machine.intake import state as S
    last, nxt = S.last_batch(), S.next_batch()
    if last is None:
        line = "Weekly questions: none sent yet; the first batch goes when a question waits."
    elif nxt is None:
        line = f"Weekly questions: last sent {last:%a %d %b}; the next can go now (it goes when a question waits)."
    else:
        line = f"Weekly questions: last sent {last:%a %d %b}; the next can go on {nxt:%a %d %b}."
    return line, {"last": last.isoformat() if last else None, "next": nxt.isoformat() if nxt else None}


def step_mlflow(now: dt.datetime) -> tuple[str, dict]:
    out = mlflow_sync.sync()
    return mlflow_sync.line(out), {k: out[k] for k in ("files", "in_mlflow", "new", "changed", "only_in_mlflow")}


FUNCTIONS = {"prices": step_prices, "caches": step_caches, "ideas": step_ideas, "questions": step_questions,
             "mlflow": step_mlflow}


# ================================================================================================= the run

def run(now: dt.datetime | None = None, *, force_fail: str | None = None, summary: bool = True, tell=None,
        functions: dict | None = None, sleep=time.sleep) -> dict | None:
    """One daily run. None = another run holds the lock (this one did nothing)."""
    if force_fail is not None and force_fail not in STEPS:
        raise ValueError(f"--force-fail wants one of {', '.join(STEPS)}, not {force_fail!r}")
    now = now or alert.now()
    functions = functions or FUNCTIONS
    with lock.one_run(Path(config.HOST_DIR) / "daily.lock") as mine:
        if not mine:
            alert.log("daily", "another daily run is working - this one does nothing")
            return None
        alert.log("daily", "daily run started" + (f" - TEST: step {force_fail} fails on purpose" if force_fail else ""))
        before = last_run()
        results = []
        for name in STEPS:
            t0 = time.time()
            try:
                if name == force_fail:
                    raise ForcedFailure("made to fail on purpose (python -m machine.host daily --force-fail "
                                        f"{name}), to test that a failure reaches you")
                line, detail = functions[name](now)
                results.append({"step": name, "ok": True, "line": line, "seconds": round(time.time() - t0, 1),
                                "detail": detail})
                alert.log("daily", line)
            except Exception as e:
                why = f"{type(e).__name__}: {e}"
                results.append({"step": name, "ok": False, "line": f"{TITLES[name]}: FAILED - {_short(str(e), 200)}",
                                "error": why, "seconds": round(time.time() - t0, 1)})
                alert.log("daily", f"{name} FAILED: {why}\n{traceback.format_exc()}")
                alert.alert(failure_text(name, e), tell=tell, sleep=sleep, kind=f"daily-{name}")
        rec = {"at": alert.iso(now), "finished": alert.iso(alert.now()), "ok": all(r["ok"] for r in results),
               "forced": force_fail, "steps": results}
        if summary:
            try:
                bot = procs.bot()                          # None = Windows could not be asked
                rec["summary"] = send_summary(now, summary_text(now, results, before, bot), bot, tell=tell,
                                              sleep=sleep)
            except Exception as e:                         # the record is still written; he still hears of it
                rec["summary"] = {"how": "failed", "error": f"{type(e).__name__}: {e}"}
                alert.log("daily", f"the summary FAILED: {e}\n{traceback.format_exc()}")
                alert.alert(f"❌ The daily run's summary could not be made: {_short(str(e), 300)}", tell=tell,
                            sleep=sleep, kind="daily-summary")
        alert.append_jsonl(Path(config.HOST_DIR) / "daily.jsonl", rec)
        alert.log("daily", "daily run finished: " + ("all steps worked" if rec["ok"] else
                  f"{sum(not r['ok'] for r in results)} step(s) FAILED"))
        return rec


def failure_text(step: str, e: Exception) -> str:
    if isinstance(e, ForcedFailure):
        return (f"TEST - nothing is wrong. The daily run's step \"{TITLES[step]}\" was made to fail on purpose, to "
                "check that a failure reaches you. It did. The other steps still ran.")
    return (f"❌ The daily run failed in step \"{TITLES[step]}\": {_short(str(e), 300)}\n"
            "The other steps still ran. The details are in data/host/daily.log on the PC.")


def last_run() -> dict | None:
    rows = alert.read_jsonl(Path(config.HOST_DIR) / "daily.jsonl")
    return rows[-1] if rows else None


# ================================================================================================= the summary

def summary_text(now: dt.datetime, results: list[dict], before: dict | None, bot: list | None) -> str:
    bad = [r for r in results if not r["ok"]]
    head = f"Daily run, {alert.india(now):%a %d %b %H:%M}: " + (f"all {len(results)} steps worked ✅" if not bad else
                                                   f"{len(bad)} of {len(results)} steps FAILED ❌")
    lines = [head, pc_line(now, before), heartbeat_line(now), bot_line(bot)]
    lines += [("❌ " if not r["ok"] else "") + r["line"] for r in results]
    return "\n".join(x for x in lines if x)


def pc_line(now: dt.datetime, before: dict | None) -> str:
    since = dt.datetime.fromisoformat(before["at"]) if before else now - dt.timedelta(days=1)
    offs = power.offs_since(since)
    boot = power.boot_time()
    on = f"PC: on since {alert.india(boot):%a %H:%M}" if boot else "PC: on"
    if not offs:
        return on + ", no restart since the last daily run."
    shown = "; ".join(off_text(o) for o in offs[-3:])
    more = f" (and {len(offs) - 3} more)" if len(offs) > 3 else ""
    return f"{on}. Off {len(offs)} time{'s' if len(offs) != 1 else ''} since the last daily run: {shown}{more}."


def off_text(o: power.Off) -> str:
    on = alert.india(o.on_at)
    if o.off_at is None:
        return f"until {on:%a %H:%M}, {o.why}"
    off = alert.india(o.off_at)
    end = f"{on:%H:%M}" if off.date() == on.date() else f"{on:%a %H:%M}"
    return f"{off:%a %H:%M} to {end}, {o.why}"


def heartbeat_line(now: dt.datetime) -> str:
    beats = alert.read_jsonl(Path(config.HOST_DIR) / "heartbeat.jsonl")
    if not beats:
        return "Heartbeat: not running yet."
    day_ago = now - dt.timedelta(days=1)
    hours = {alert.india(dt.datetime.fromisoformat(b["at"])).replace(minute=0, second=0, microsecond=0)
             for b in beats if dt.datetime.fromisoformat(b["at"]) > day_ago}
    first = dt.datetime.fromisoformat(beats[0]["at"])
    expected = 24 if first <= day_ago else max(1, int((now - first).total_seconds() // 3600) + 1)
    return f"Heartbeat: {min(len(hours), expected)} of the last {expected} hour{'s' if expected != 1 else ''}."


def bot_line(bot: list | None) -> str:
    if bot is None:
        return "Bot: could not check (Windows did not answer)."
    return "Bot: running." if bot else "Bot: NOT running ❌ (your messages wait on Telegram until it starts)."


def send_summary(now: dt.datetime, text: str, bot: list | None, *, tell=None, sleep=time.sleep) -> dict:
    """Through the bot's outbox when the bot runs - or when Windows could not be asked (the usual way); an id is sent
    once: a second run that day gets the time in its id. Straight to Telegram only when the bot is KNOWN to be off."""
    if bot is None or bot:
        line_id = f"daily-{now:%Y-%m-%d}"
        if not alert.to_outbox(line_id, text):
            line_id = f"daily-{now:%Y-%m-%d-%H%M%S}"
            alert.to_outbox(line_id, text)
        return {"how": "outbox", "id": line_id}
    sent = alert.alert(text + "\n(Sent straight to you: the bot is not running.)", tell=tell, sleep=sleep,
                       kind="daily-summary")
    return {"how": "straight", "sent": sent}


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"
