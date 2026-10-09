"""python -m machine.strategy - his whole strategy through the research loop (L19d). See machine/strategy/research.py.

    .venv\\Scripts\\python -m machine.strategy plan                  Claude plans; code checks and SAVES the plan
    .venv\\Scripts\\python -m machine.strategy run                   the plan's versions on 2016-2020 (a try each)
    .venv\\Scripts\\python -m machine.strategy judge --who "..."     the referee: ONE version, its 2x2 and slices, the
                                                                    check years opened ONCE for it, judged; reports
    .venv\\Scripts\\python -m machine.strategy reports               the reports again from the files (runs nothing)
    .venv\\Scripts\\python -m machine.strategy decide                the machine's choice after the report (decision.md,
                                                                    written once; LEARNED #162 - it tells, never asks)
    .venv\\Scripts\\python -m machine.strategy show                  the plan and the log
On real prices (2 Oct 2026): run = 6 minutes (the compile, then seconds a version); judge = 40 minutes (the compile
twice, 200 random runs on each window). Use python -u and a long timeout (a background job's default is 30 minutes).
"""
from __future__ import annotations

import argparse
import sys

from machine import config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.strategy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("plan", "run", "judge", "reports", "decide", "parked", "show"))
    ap.add_argument("idea", nargs="?", help="parked: the idea parked for his hedge (idea-NNN)")
    ap.add_argument("--who", help="judge: who opens the check years (goes into check-opened.json)")
    ap.add_argument("--runs", type=int, default=config.RANDOM_RUNS, help="judge: random runs of the 2x2")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from machine.strategy import report as RP
    from machine.strategy import research as SR

    def say(m: str) -> None:
        print(m, flush=True)

    if a.step == "plan":
        plan = SR.make_plan(say=say)
        print(SR.plan_text(plan))
    elif a.step == "run":
        out = SR.run_plan(say=say)
        print(f"stopped: {out['stopped']} - {len(out['ran'])} versions ran")
    elif a.step == "judge":
        if not a.who:
            ap.error("judge needs --who (the check years open once; the file says who opened them)")
        j = SR.judge(who=a.who, n_runs=a.runs, say=say)
        print(j["result"]["answer"]["word"])
        for p in RP.write_all():
            print(p.relative_to(config.ROOT).as_posix())
    elif a.step == "parked":
        if not a.idea:
            ap.error("parked needs the idea (idea-NNN)")
        from machine.strategy import parked
        out = parked.run(a.idea, n_runs=a.runs, log=say)
        print(f"{out['choice']} - {out['means']}")
        for k in ("trial", "page", "decision"):
            print(out[k])
    elif a.step == "decide":
        print(SR.decide(say=say))
    elif a.step == "reports":
        for p in RP.write_all():
            print(p.relative_to(config.ROOT).as_posix())
    else:
        print(SR.show())
    return 0


if __name__ == "__main__":
    sys.exit(main())
