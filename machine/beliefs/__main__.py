"""python -m machine.beliefs rules | counts | tune | retune | open | check  (L13b)

    rules                       the rules for an answer, as printed in every report
    counts [--years 2019]       how many events each belief finds - NO outcome is printed (for checking the build
                                before anything is looked at)
    tune [--only NAME ...]      every version on 2016-2020 (a trial file each), one chosen per belief
    retune --only NAME ... --why "..."   after a FIX to how events are read: the same versions on 2016-2020 again,
                                chosen again by the same rule; the first choice is kept in chosen.json ("before_fix")
    open --who "..."            open the check years ONCE for each belief's chosen version
    check [--only NAME ...] [--note "..."]   the check years 2021-2023, the chosen version, judged the chosen way (a
                                trial file each); --note: a run again after a fix says so
Each step also writes its reports: plan/work/beliefs/<belief>-tune.md / -check.md.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from machine import config
from machine.beliefs import engine, registry
from machine.beliefs.world import World


def _counts(fams, years) -> None:
    w = World.load(years)
    for fam, versions in fams.items():
        for b in versions:
            t0 = time.time()
            raw = b.rows(w)
            has_x = int(np.isfinite(raw["x"].to_numpy(float)).sum()) if len(raw) else 0
            sizes = [0 if p is None else int(np.isfinite(np.asarray(p, dtype=float)).sum()) for p in raw["pool"]]
            enough = int(np.sum(np.array(sizes) >= engine.MIN_MATCH)) if sizes else 0
            print(f"{b.box:<7} {fam:<32} {b.version_id:<22} {len(raw):>7,} {b.noun} found · {has_x:>7,} with an "
                  f"answer · {enough:>7,} with {engine.MIN_MATCH}+ comparison values · {time.time() - t0:5.1f} s",
                  flush=True)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.beliefs", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("rules", "counts", "tune", "retune", "open", "check"))
    ap.add_argument("--only", nargs="*", help="belief names (e.g. equal-magnet)")
    ap.add_argument("--years", nargs="*", type=int, help="counts only: the years to count on (tune years only)")
    ap.add_argument("--who", default="Tejas (with Claude)", help="who opens the check years (goes into the file)")
    ap.add_argument("--why", help="retune: the fix, in plain words (goes into chosen.json and the trial files)")
    ap.add_argument("--note", help="check: a run again after a fix, in plain words (the trial files and the pages)")
    a = ap.parse_args(argv)
    if a.step == "rules":
        print(engine.RULES)
        return 0
    fams = registry.families(a.only)
    if a.step == "counts":
        years = tuple(a.years or config.TUNE_YEARS)
        if set(years) - set(config.TUNE_YEARS):
            raise SystemExit("counts reads the tune years only")
        _counts(fams, years)
        return 0
    if a.step in ("tune", "retune"):
        if a.step == "retune" and not (a.only and a.why):
            raise SystemExit("retune needs --only (the beliefs the fix touched) and --why (the fix)")
        w = World.load(config.TUNE_YEARS)
        done = engine.tune(fams, w) if a.step == "tune" else engine.retune(fams, w, a.why)
        for fam, reps in done.items():
            for r in reps:
                print(r.text(), "\n" + "=" * 118, flush=True)
        print("CHOSEN (one version per belief, for the check years):")
        for fam, c in engine.read_chosen().items():
            if fam in fams:
                print(f"  {fam}: {c['version_id']} · judged {'his way' if c['way'] > 0 else 'the OPPOSITE way'} · "
                      f"{c['why']}")
        return 0
    if a.step == "open":
        engine.open_all(fams, a.who)
        print(f"check years open for {len(fams)} beliefs; opened {engine.splits.opened_count()} times in the project")
        return 0
    chosen = engine.read_chosen()
    w = World.load(config.CHECK_YEARS, opened=[(f, chosen[f]["version_id"]) for f in fams])
    for fam, r in engine.check(fams, w, note=a.note).items():
        print(r.text(), "\n" + "=" * 118, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
