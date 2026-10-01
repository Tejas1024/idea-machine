"""Labels for "his eyes" (L15.1).

    .venv\\Scripts\\python -m machine.labels build        data/labels/<word>.parquet: his drawn labels (+ his Telegram answers)
    .venv\\Scripts\\python -m machine.labels status       how many labels, charts sent, waiting, his T9 answer
    .venv\\Scripts\\python -m machine.labels catalogue    plan/work/labels/his-drawings.md: all his pictures, read
    .venv\\Scripts\\python -m machine.labels send [--dry-run]         today's 10 charts (only after his T9 opt-in)
    .venv\\Scripts\\python -m machine.labels opt-in --answer "opt in" --words "<his words>" --who "<name>"
                                                        T9: his answer, word for word; "opt in" adds the daily task,
                                                        "no" removes it
"""
from __future__ import annotations

import argparse
import sys

from machine import config
from machine.labels import daily, drawings, replies, store, words


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.labels", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "status", "send", "catalogue"):
        p = sub.add_parser(name)
        p.add_argument("--word", default=words.STRONG_ZONE.slug, choices=sorted(words.WORDS))
        if name == "send":
            p.add_argument("--dry-run", action="store_true", help="draw them, send nothing")
    p = sub.add_parser("opt-in")
    p.add_argument("--answer", required=True, choices=("opt in", "no"))
    p.add_argument("--words", required=True, help="his words, word for word")
    p.add_argument("--who", required=True)
    p.add_argument("--no-task", action="store_true", help="record his answer only; leave Task Scheduler alone")
    p = sub.add_parser("install")
    p.add_argument("--at", default=config.LABELS_AT)
    sub.add_parser("uninstall")
    a = ap.parse_args(argv)

    if a.cmd == "build":
        df = store.build(a.word)
        path = store.write(df, a.word)
        print(f"{store.parquet_path(a.word).relative_to(config.ROOT).as_posix()}: {len(df)} labels")
        print(store.summary(df))
        print(f"written: {path}")
    elif a.cmd == "status":
        print(replies.status_line(a.word))
    elif a.cmd == "catalogue":
        path = config.LABELS_WORK_DIR / "his-drawings.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(drawings.page(), encoding="utf-8", newline="\n")
        tmp.replace(path)
        print(f"written: {path.relative_to(config.ROOT).as_posix()} ({len(drawings.PICTURES)} pictures)")
    elif a.cmd == "send":
        try:
            out = daily.send(a.word, dry=a.dry_run)
        except daily.LabelsOff as e:
            daily.log(a.word, f"not sent: {e}")
            print(f"not sent: {e}")
            return 0
        daily.log(a.word, out["said"])
        print(out["said"])
    elif a.cmd == "opt-in":
        path = daily.opt_in(a.answer, a.words, who=a.who)
        print(f"his answer '{a.answer}' added to {path.relative_to(config.ROOT).as_posix()}")
        if not a.no_task:
            print(daily.install() if a.answer == "opt in" else daily.uninstall())
    elif a.cmd == "install":
        print(daily.install(a.at))
    elif a.cmd == "uninstall":
        print(daily.uninstall())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
