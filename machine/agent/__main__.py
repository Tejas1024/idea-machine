"""python -m machine.agent - the research agent (L14.2). machine/agent/research.py says how it works.

    .venv\\Scripts\\python -m machine.agent plan idea-002            the plan: Claude writes it, code checks and SAVES it
    .venv\\Scripts\\python -m machine.agent run idea-002 [--send]    the plan, then exactly the plan, the report, the
                                                                    final message (--send: to his Telegram)
    .venv\\Scripts\\python -m machine.agent replan idea-002 "why"    a session: put the plan aside and make it again
    .venv\\Scripts\\python -m machine.agent show idea-002            the plan and the log
    .venv\\Scripts\\python -m machine.agent noise [--idea idea-002] [--versions 3]   the noise exam (noise.py)
"""
from __future__ import annotations

import argparse
import sys


def _say(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.replace("₹", "Rs ").encode(enc, errors="replace").decode(enc), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "show"):
        p = sub.add_parser(name)
        p.add_argument("idea")
        if name == "run":
            p.add_argument("--send", action="store_true", help="the report and every version to his Telegram")
    p = sub.add_parser("replan")
    p.add_argument("idea")
    p.add_argument("why")
    p = sub.add_parser("noise")
    p.add_argument("--idea", default="idea-002", help="whose saved plan runs on the made-up prices")
    p.add_argument("--versions", type=int, default=3, help="the small budget: the plan's first N versions")
    p.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    from machine.agent import research as R
    if a.cmd == "show":
        _say(R.show(a.idea))
        return 0
    if a.cmd == "plan":
        try:
            plan = R.make_plan(a.idea, say=_say)
        except R.ResearchError as err:
            _say(f"{a.idea}: {err}")
            return 1
        _say(R.plan_text(plan))
        return 0
    if a.cmd == "replan":
        plan = R.replan(a.idea, a.why, say=_say)
        _say(R.plan_text(plan))
        return 0
    if a.cmd == "noise":
        from machine.agent import noise
        out = noise.exam(idea=a.idea, versions=a.versions, seed=a.seed, say=_say)
        return 0 if out["passed"] else 1
    out = R.run(a.idea, send=a.send, say=_say)
    _say(f"\n{out.idea}: {out.status} - {out.said}")
    if out.final:
        _say("\n" + out.final)
    return 0 if out.status in ("reported", "plan saved") else 1


if __name__ == "__main__":
    sys.exit(main())
