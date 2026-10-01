"""The intake from the command line (L10b). Needs the price data (💻) for the steps after the form.

    .venv\\Scripts\\python -m machine.intake "sell when the asian high is taken ..."   queue the idea, then run
    .venv\\Scripts\\python -m machine.intake "..." --queue-only                        only queue it
    .venv\\Scripts\\python -m machine.intake run                                     everything the machine can do now
    .venv\\Scripts\\python -m machine.intake status                                  one line per idea (as STATUS)
    .venv\\Scripts\\python -m machine.intake retry idea-002                          a session fixed what it waited for
    .venv\\Scripts\\python -m machine.intake answer idea-002-shown-1 "YES" --source "..."
                                         a session gives its reading of his words to the 3-setups picture
    .venv\\Scripts\\python -m machine.intake send-now "his words"                  the weekly batch NOW, before its
                                         day - only on his words (CLAUDE.md rule 16); they go into sends.jsonl
    .venv\\Scripts\\python -m machine.intake withdraw idea-002 "his words"        take back a picture of 3 setups he
                                         could not judge (not a NO); the next run draws the same setups anew
"""
from __future__ import annotations

import argparse
import sys

from machine.intake import state as S

COMMANDS = ("run", "status", "retry", "answer", "send-now", "withdraw")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        ap = argparse.ArgumentParser(prog="python -m machine.intake", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("words", nargs="+")
        ap.add_argument("--queue-only", action="store_true")
        a = ap.parse_args(argv)
        line = S.add(" ".join(a.words))
        print(f"queued: \"{line['words']}\" ({len(S.pending())} waiting)")
        if not a.queue_only:
            for x in S.run():
                print(x)
            for x in S.status_lines():
                print(x)
        return 0
    ap = argparse.ArgumentParser(prog="python -m machine.intake", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("status")
    r = sub.add_parser("retry")
    r.add_argument("idea")
    w = sub.add_parser("answer")
    w.add_argument("qid")
    w.add_argument("words")
    w.add_argument("--source", default="a Claude session on the PC")
    x = sub.add_parser("withdraw")
    x.add_argument("idea")
    x.add_argument("why", help="his words: why the picture of 3 setups is taken back")
    n = sub.add_parser("send-now")
    n.add_argument("why", help="his words: why the batch goes before its day")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        S.run()
    elif a.cmd == "status":
        for x in S.status_lines() or ["no idea yet"]:
            print(x)
    elif a.cmd == "retry":
        s = S.retry(a.idea)
        print(f"{a.idea}: {s['state']} - the machine tries again at the next run")
    elif a.cmd == "withdraw":
        from machine.intake import shown
        rec = shown.withdraw(a.idea, a.why)
        S._tell(f"{rec['qid']}-withdrawn", f"{S.label(a.idea).capitalize()}: the picture of 3 setups I sent is taken "
                "back. A new one comes; a reply to the old one changes nothing.", idea=a.idea,
                why="L10b SHOWN taken back")
        print(f"{rec['qid']} taken back; {a.idea} is CONFIRMED & LOCKED again - the next run draws a new picture")
    elif a.cmd == "send-now":
        sent = S.batch_step(S.Tools(), early=a.why)
        print("sent: " + ", ".join(S._read_jsonl(S.work_dir() / "sends.jsonl")[-1].get("shown", [])) if sent
              else "nothing waits to be sent")
    elif a.cmd == "answer":
        print(S.on_reply({"qid": a.qid}, a.words, False, source=a.source) or f"{a.qid} is not one of the intake's")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
