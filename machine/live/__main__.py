"""python -m machine.live once | rehearse <sunday> [--no-prices] | moves <from> <to> | check | report <day> [--send] |
status | install | commands | uninstall   (L17a)"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from pathlib import Path

from machine import config
from machine.live import run, tasks


def console() -> None:
    """Task Scheduler starts pythonw with NO console (sys.stdout / sys.stderr are None; a library that writes to them
    stops with an error - machine/host's console(), 4 Oct 2026): then both write into LIVE_DIR/console.log."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    path = Path(config.LIVE_DIR) / "console.log"
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
    ap = argparse.ArgumentParser(prog="python -m machine.live", description=run.__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("once", help="what is due now (Task Scheduler runs this every minute)")
    p = sub.add_parser("rehearse", help="a 2015-2023 week replayed minute by minute (no page read, no real outbox)")
    p.add_argument("sunday", help="the week's Sunday, e.g. 2023-10-01")
    p.add_argument("--no-prices", action="store_true", help="step 1's alerts only (no 1-minute bars as the feed)")
    p = sub.add_parser("moves", help="the big-move rule counted on the 1-minute bars of two years (2016-2023 only)")
    p.add_argument("first", help="e.g. 2016")
    p.add_argument("last", help="e.g. 2020")
    sub.add_parser("check", help="the daily check of the live prices against MT5's history, now")
    p = sub.add_parser("report", help="a checked day's report (the pass sends it by itself after the day's check)")
    p.add_argument("day", help="a checked server day, e.g. 2026-10-05")
    p.add_argument("--send", action="store_true", help="queue it for the bot (once per day; STOP holds it)")
    for name in ("status", "install", "commands", "uninstall"):
        sub.add_parser(name)
    a = ap.parse_args(argv)
    if a.cmd == "once":
        try:
            done = run.locked_once()
        except Exception:
            run.log("FAILED: " + traceback.format_exc().strip().replace("\n", " | ")[-900:])
            raise
        if done is None:
            print("another pass is running (once.lock)")
            return 0
        for d in done:
            print(f"{d['kind']} {d.get('report') or d['key']}: {d['sent']}")    # a big move has no report
        return 0
    if a.cmd == "rehearse":
        for d in run.rehearse(dt.date.fromisoformat(a.sunday), prices=not a.no_prices):
            print(f"===== {d['at']} UTC - {d['kind']} - {d.get('report', '')}\n{d['text']}\n")
        return 0
    if a.cmd == "moves":
        from machine.live import move
        c = move.count(a.first, a.last)
        al = c["alerts"]
        print(f"{len(al)} big-move alerts in {a.first}-{a.last}: {c['per_week']:.2f} a week "
              f"({c['weeks']:.0f} weeks, {c['minutes']:,} minutes measured)")
        days = al.groupby(al.index.normalize()).size()
        print(f"days with 1: {(days == 1).sum()}, 2: {(days == 2).sum()}, 3 or more: {(days >= 3).sum()}")
        for t, r in al.sort_values("share_of_day", ascending=False).head(15).iterrows():
            print(f"  {t:%Y-%m-%d} {r['ist']} India time  ${r['move']:.2f} in 15 min = {r['times_usual']:.1f} x usual, "
                  f"{r['share_of_day']:.2f} of a day")
        return 0
    if a.cmd == "check":
        from machine.live import check
        done = check.run()
        print("\n".join(check.words(r) for r in done) or "nothing to check (no new day kept since the last check)")
        return 0
    if a.cmd == "report":
        from machine.host import alert as HA
        from machine.live import check
        rec = next((r for r in HA.read_jsonl(Path(config.LIVE_DIR) / "checks.jsonl") if r.get("day") == a.day), None)
        if rec is None:
            print(f"{a.day} is not checked yet (the day is checked once MT5's history of it is kept)")
            return 1
        print(check.day_report(rec))
        if a.send:
            out = run.day_report(rec)
            print(f"\n{out['id']}: {out['sent']}")
        return 0
    if a.cmd == "status":
        print("\n".join(run.status_lines() + [tasks.status()]))
        return 0
    if a.cmd == "install":
        print(tasks.install())
        return 0
    if a.cmd == "commands":
        print(tasks.script())
        return 0
    if a.cmd == "uninstall":
        tasks.uninstall()
        print("removed")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
