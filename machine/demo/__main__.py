"""python -m machine.demo - one made-up idea through every step of the machine (machine/demo/run.py says how).

    .venv\\Scripts\\python -m machine.demo                   the steps, about 2 minutes
    .venv\\Scripts\\python -m machine.demo --full            + the referee's real answer and the report (about 20 min)
    .venv\\Scripts\\python -m machine.demo --full --out demo-output     ... and save the report page there
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from machine.demo.run import main_steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="choose on 2016-2020, open 2021-2023 once, judge, report")
    ap.add_argument("--out", type=Path, help="--full: save the report page and its picture in this folder")
    ap.add_argument("--keep", type=Path, help="write the demo's files here instead of a temporary folder")
    a = ap.parse_args(argv)
    main_steps(full=a.full, out=a.out, keep=a.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
