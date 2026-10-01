"""The question loop from the command line (L9b). Needs the price data (💻) for `ask`.

    python -m machine.questions ask idea-001 [--part U2] [--again]   make the questions (search, readings, pictures)
    python -m machine.questions checked idea-001 U2 "understanding §66g" "why it does not answer"
    python -m machine.questions batch --dry-run                      this week's batch, nothing sent
    python -m machine.questions batch                                ... and its lines into the Telegram outbox
    python -m machine.questions status                               every part and question
    python -m machine.questions withdraw idea-001 U2 "why"           take back sent, unanswered questions
    python -m machine.questions answer idea-001-U2-1 "B"             a T5 session files his answer (the bot does
                                                                     this by itself for Telegram replies)
"""
from __future__ import annotations

import argparse
import sys

from machine.questions import ask, batch, interpret
from machine.questions import queue as Q


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.questions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("idea")
    a.add_argument("--part")
    a.add_argument("--again", action="store_true", help="make new questions for a part whose questions still wait")
    c = sub.add_parser("checked")
    c.add_argument("idea")
    c.add_argument("part")
    c.add_argument("place", help="the search's id for the place, e.g. 'understanding §66g'")
    c.add_argument("why")
    b = sub.add_parser("batch")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--budget", type=int)
    sub.add_parser("status")
    x = sub.add_parser("withdraw")
    x.add_argument("idea")
    x.add_argument("part")
    x.add_argument("why")
    w = sub.add_parser("answer")
    w.add_argument("qid")
    w.add_argument("words")
    w.add_argument("--photo", action="store_true", help="he answered with a drawing")
    w.add_argument("--source", default="a T5 session on the PC")
    args = ap.parse_args(argv)

    if args.cmd == "ask":
        parts = [args.part] if args.part else ask.open_ask_parts(args.idea)
        if not parts:
            print(f"{args.idea}: no open ASK part - nothing to ask")
        market = None
        for p in parts:
            if market is None:
                from machine import config
                from machine.ideas.compile import Market
                market = Market.load(str(min(config.TUNE_YEARS)), str(max(config.TUNE_YEARS)))
            rec = ask.make_part(args.idea, p, market=market, again=args.again)
            print(f"{args.idea} {p}: {rec['status']} -> {Q.rel(Q.folder(args.idea, p) / 'part.md')}")
        return 0
    if args.cmd == "checked":
        ask.add_checked(args.idea, args.part, args.place, args.why)
        print(f"written: {args.place} does not answer {args.idea} {args.part}. Now: python -m machine.questions ask "
              f"{args.idea} --part {args.part}")
        return 0
    if args.cmd == "batch":
        out = batch.make(dry=args.dry_run, budget=args.budget)
        print(f"batch {out['date']}{' (DRY RUN - nothing sent)' if out['dry_run'] else ''}: {len(out['chosen'])} "
              f"question(s): {', '.join(out['chosen']) or 'none'}; next week: {', '.join(out['later']) or 'none'}")
        print(f"page: {Q.rel(Q.queue_dir() / 'batches' / out['date'] / 'batch.md')}")
        return 0
    if args.cmd == "withdraw":
        rec = ask.withdraw(args.idea, args.part, args.why)
        print(f"{args.idea} {args.part}: {len(rec['withdrawn'])} question(s) withdrawn. Now: python -m machine.questions "
              f"ask {args.idea} --part {args.part}")
        return 0
    if args.cmd == "status":
        for rec in Q.records():
            qs = ", ".join(f"{q['qid']} {q['status']}" + (f" ({q['answer'].get('letter') or q['answer'].get('kind')})"
                                                          if q.get("answer") else "") for q in rec["questions"])
            print(f"{rec['idea']} {rec['part']}: {rec['status']}" + (f" - {qs}" if qs else ""))
        for h in Q.hand_made():
            print(f"{h['qid']}: made by hand ({h['status'] or 'no status'}) - never sent by the batch")
        return 0
    if args.cmd == "answer":
        found = Q.find(args.qid)
        if not found:
            print(f"no L9b question {args.qid}")
            return 1
        rec, _ = found
        print(interpret.on_answer({"qid": args.qid, "idea": rec["idea"]}, args.words, args.photo,
                                  source=args.source) or "(nothing)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
