"""The host PC's daily runs from the command line (L16.1).

    .venv\\Scripts\\python -m machine.host daily                     the daily run (Task Scheduler: every day)
    .venv\\Scripts\\python -m machine.host daily --force-fail mlflow  a TEST: that step fails on purpose; the alert says so
    .venv\\Scripts\\python -m machine.host heartbeat                 one heartbeat (Task Scheduler: every hour)
    .venv\\Scripts\\python -m machine.host status                    the last runs and beats, open problems, the two jobs
    .venv\\Scripts\\python -m machine.host offs --days 7             when the PC was off, and why (Windows' own log)
    .venv\\Scripts\\python -m machine.host install                   add both jobs to Task Scheduler
    .venv\\Scripts\\python -m machine.host commands                  print the PowerShell that adds them (to paste)
    .venv\\Scripts\\python -m machine.host uninstall                 remove both jobs
    .venv\\Scripts\\python -m machine.host prices                    keep the new days now (the daily run does it)
    .venv\\Scripts\\python -m machine.host prices --same-as-l1 2026-08-14   does MT5 give L1's text for that day?
    .venv\\Scripts\\python -m machine.host mlflow-sync               every trial file into MLflow now
    .venv\\Scripts\\python -m machine.host mlflow-ui                 MLflow's page on http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from machine import config
from machine.host import alert

COMMANDS = ("daily", "heartbeat", "status", "offs", "install", "commands", "uninstall", "prices", "mlflow-sync",
            "mlflow-ui")


def console() -> None:
    """Task Scheduler starts pythonw with NO console: sys.stdout and sys.stderr are None, and a library that draws a
    progress bar on them stops with an error (tested 4 Oct 2026: tqdm -> "'NoneType' object has no attribute 'write'";
    the document search's model draws one while it loads). Then both write into data/host/console.log."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    path = Path(config.HOST_DIR) / "console.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = fh
    if sys.stderr is None:
        sys.stderr = fh


def main(argv: list[str] | None = None) -> int:
    console()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.host", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("daily")
    d.add_argument("--force-fail", metavar="STEP", help="a TEST: this step fails on purpose")
    d.add_argument("--no-summary", action="store_true", help="send no summary (the alerts still go)")
    sub.add_parser("heartbeat")
    sub.add_parser("status")
    o = sub.add_parser("offs")
    o.add_argument("--days", type=float, default=7)
    sub.add_parser("install")
    sub.add_parser("commands")
    sub.add_parser("uninstall")
    p = sub.add_parser("prices")
    p.add_argument("--same-as-l1", metavar="DAY", help="compare MT5's text for DAY with L1's sealed copy (hashes)")
    sub.add_parser("mlflow-sync")
    u = sub.add_parser("mlflow-ui")
    u.add_argument("--port", type=int, default=5000)
    a = ap.parse_args(argv)

    if a.cmd == "daily":
        from machine.host import daily
        rec = daily.run(force_fail=a.force_fail, summary=not a.no_summary)
        if rec is None:
            print("another daily run is working - this one did nothing")
            return 0
        for r in rec["steps"]:
            print(("ok   " if r["ok"] else "FAIL ") + r["line"])
        print("summary:", json.dumps(rec.get("summary")))
        return 0 if rec["ok"] or rec.get("forced") else 1
    if a.cmd == "heartbeat":
        from machine.host import heartbeat
        rec = heartbeat.beat()
        print(json.dumps(rec, ensure_ascii=False) if rec else "another beat is working")
        return 0
    if a.cmd == "status":
        for line in status_lines():
            print(line)
        return 0
    if a.cmd == "offs":
        from machine.host import daily, power
        offs = power.offs_since(alert.now() - dt.timedelta(days=a.days))
        for x in offs:
            m = x.minutes()
            print(daily.off_text(x) + (f"  [{m:.0f} min]" if m is not None else ""))
        print(f"{len(offs)} time(s) in the last {a.days:g} days")
        return 0
    if a.cmd in ("install", "commands", "uninstall"):
        from machine.host import tasks
        if a.cmd == "install":
            print(tasks.install())
        elif a.cmd == "commands":
            print(tasks.script())
        else:
            tasks.uninstall()
            print(f"removed '{config.DAILY_TASK_NAME}' and '{config.HEARTBEAT_TASK_NAME}' (if they were there)")
        return 0
    if a.cmd == "prices":
        from machine.host import daily, prices
        if a.same_as_l1:
            day = dt.date.fromisoformat(a.same_as_l1)
            for tf in config.MT5_STORE_TIMEFRAMES:
                print(json.dumps(prices.same_as_l1(day, tf)))
            return 0
        out = prices.store()
        print(daily.prices_line(out))
        for tf, x in out["timeframes"].items():
            print(f"  {tf}: {len(x.get('days') or [])} day(s), {x.get('rows', 0):,} rows, {x.get('from_day')} .. "
                  f"{x.get('to_day')}")
        return 0
    if a.cmd == "mlflow-sync":
        from machine.host import mlflow_sync
        out = mlflow_sync.sync()
        print(mlflow_sync.line(out))
        return 0
    if a.cmd == "mlflow-ui":
        from machine.host import mlflow_sync
        uri = mlflow_sync.uri()
        print(f"MLflow on http://127.0.0.1:{a.port} (store {uri}); Ctrl+C stops it")
        return subprocess.call([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri,
                                "--host", "127.0.0.1", "--port", str(a.port)])
    return 2


def status_lines() -> list[str]:
    from machine.host import tasks
    host = Path(config.HOST_DIR)
    out = tasks.status()
    runs = alert.read_jsonl(host / "daily.jsonl")
    if runs:
        r = runs[-1]
        bad = [s["step"] for s in r["steps"] if not s["ok"]]
        out.append(f"Last daily run: {r['at'][:16].replace('T', ' ')}, "
                   + ("all steps worked" if not bad else "FAILED: " + ", ".join(bad))
                   + (" (a TEST)" if r.get("forced") else ""))
    else:
        out.append("Last daily run: none yet")
    beats = alert.read_jsonl(host / "heartbeat.jsonl")
    if beats:
        b = beats[-1]
        out.append(f"Last heartbeat: {b['at'][:16].replace('T', ' ')}, problems: {', '.join(b.get('problems') or []) or 'none'}"
                   f", ping: {b.get('ping')}")
        missed = [x for x in beats if x.get("missed")]
        if missed:
            m = missed[-1]
            out.append(f"Last missed heartbeat: {m['missed']['since'][:16].replace('T', ' ')} -> "
                       f"{m['at'][:16].replace('T', ' ')} ({m['missed']['minutes']} min)")
    else:
        out.append("Last heartbeat: none yet")
    try:
        state = json.loads((host / "alerts.json").read_text(encoding="utf-8"))
        out.append("Open problems: " + (", ".join(state.get("open", {})) or "none"))
    except (OSError, ValueError):
        out.append("Open problems: none")
    waiting = alert.read_jsonl(host / "unsent.jsonl")
    if waiting:
        out.append(f"Messages not sent yet (no network then): {len(waiting)}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
