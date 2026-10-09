"""His eyes (L15.2) - see machine/eyes/__init__.py.

    .venv\\Scripts\\python -m machine.eyes exams          made-up answers first -> plan/work/eyes/exams.json + .md
    .venv\\Scripts\\python -m machine.eyes plan           -> plan/work/eyes/plan.json + .md (then commit AND push)
    .venv\\Scripts\\python -m machine.eyes run            scores his held-out answers ONCE -> eyes-model.md, trial files
    .venv\\Scripts\\python -m machine.eyes run --again "<why>"     a second scoring, written next to the first
    .venv\\Scripts\\python -m machine.eyes report         the report again from result.json (no new scoring)
    .venv\\Scripts\\python -m machine.eyes final          both models taught on all his answers -> data/eyes/
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from machine import config


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.eyes", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("exams")
    sub.add_parser("plan")
    p = sub.add_parser("run")
    p.add_argument("--again", default="", help="why his answers are scored again (written in runs.jsonl)")
    sub.add_parser("report")
    sub.add_parser("final")
    a = ap.parse_args(argv)
    t0 = time.time()

    def say(msg: str) -> None:
        print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

    from machine.eyes import labels as LB
    from machine.eyes import run as RUN

    if a.cmd == "exams":
        from machine.eyes import exams
        data = RUN.prepare()
        say(f"{len(data.lab)} charts ready (made-up answers only)")
        out = exams.run_all(data, say=say)
        js, md = exams.write(out)
        print(json.dumps(out["summary"], indent=2))
        say(f"written: {md.relative_to(config.ROOT).as_posix()}")
    elif a.cmd == "plan":
        from machine.eyes import plan
        lab = LB.load()
        ex = _exams()
        if ex is None:
            print("no exams yet: run `python -m machine.eyes exams` first")
            return 1
        p, md = plan.write(plan.make(lab, ex, versions=_versions()))
        print(f"written: {p.relative_to(config.ROOT).as_posix()} + plan.md - commit and push them, then: "
              "python -m machine.eyes run")
    elif a.cmd == "run":
        from machine.data import bars
        from machine.eyes import plan, report
        lab = LB.load()
        fixed = plan.check(lab)                       # refuses before anything is scored
        done = report.runs()
        if done and not a.again:
            print(f"his answers were already scored {len(done)} time(s) ({done[-1]['at']}): a second scoring needs "
                  "--again \"<why>\" and is written next to the first")
            return 1
        data = RUN.prepare(lab)
        say(f"{len(lab)} answers, plan of commit {fixed['plan_commit'][:10]} - scoring the held-out ones")
        r = RUN.measure(data, lab["y"].to_numpy(), say=say)
        from machine.eyes import model_b
        fp = {"labels_sha256": fixed["labels"]["fingerprint"], "label_file": "data/labels/strong_zone.parquet",
              "gold_m1_content_sha256": bars.m1_fingerprint(), "resnet18_weights": model_b.weights_file()}
        line = report.write_all(r, fixed, lab, exams=_exams(), left=LB.left_out(), data_fp=fp,
                                run_no=len(done) + 1, again=a.again)
        print(json.dumps(line, indent=2, ensure_ascii=False))
        say(f"written: {config.EYES_REPORT.relative_to(config.ROOT).as_posix()}")
    elif a.cmd == "report":
        from machine.eyes import plan, report
        lab = LB.load()
        fixed = json.loads(plan.path().read_text(encoding="utf-8"))
        line = report.runs()[-1]
        fixed["plan_commit"] = line.get("plan_commit") or "?"
        out = report.rebuild(fixed, lab, exams=_exams(), left=LB.left_out())
        print(f"written again (no new scoring): {out.relative_to(config.ROOT).as_posix()}")
    elif a.cmd == "final":
        from machine.eyes import final
        out = final.make(say=say)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _exams() -> dict | None:
    p = config.EYES_WORK_DIR / "exams.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _versions() -> dict:
    import lightgbm
    import torch
    import torchvision
    from machine.eyes import model_b
    return {"python": sys.version.split()[0], "lightgbm": lightgbm.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "resnet18_weights": model_b.weights_file()}


if __name__ == "__main__":
    raise SystemExit(main())
