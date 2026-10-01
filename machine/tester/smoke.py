"""The smoke run (L3): an idea with no edge, to prove the tester is honest.

Buy at the next open every Monday 10:00 server time, stop $5 below the fill, target $5 above it, 5-min bars, the
TUNE years 2016-2020, 0.01 lot, one trade at a time. With no edge it should lose about the cost (Rs 40 a trade).
If it makes a lot, hunt the bug before believing it (lesson B6).

    python -m machine.tester.smoke
"""
from __future__ import annotations

import sys

import pandas as pd

from machine import config
from machine.data.bars import load_bars
from machine.tester import Signal, known_at_close, run

IDEA, VERSION = "smoke-monday-10", "v1"


def monday_signals(bars_5m: pd.DataFrame) -> list[Signal]:
    """One signal per Monday whose 09:55 bar exists: known at its close, 10:00 (no bar, no signal — a holiday)."""
    idx = pd.DatetimeIndex(bars_5m.index)
    made_by = idx[(idx.dayofweek == 0) & (idx.hour == 9) & (idx.minute == 55)]
    return [Signal(id=f"mon-{t:%Y-%m-%d}", known_at=known_at_close(t, "5m"), direction="buy",
                   entry_type="next_open", levels="from_fill", stop=5.0, targets=(5.0,), lots=0.01,
                   idea_id=IDEA, version_id=VERSION)
            for t in made_by]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    start, end = str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1])
    signals = monday_signals(load_bars("5m", start, end))
    result = run(signals, tf="5m", start=start, end=end,
                 note="L3 smoke run: buy next open every Monday 10:00 server time, stop $5, target $5 — no edge")
    print(result.text())
    table = result.table()
    closed = table[table.status == "closed"]
    print(f"\nGold before cost: {closed.gold_usd.sum():+.2f} dollars over {len(closed)} trades "
          f"(targets {int((closed.closed_by == 'target').sum())}, stops {int((closed.closed_by == 'stop').sum())})")
    print(f"Hours held: median {((closed.exit_time - closed.fill_time).dt.total_seconds() / 3600).median():.1f}")
    print("\nFirst 5 trades:")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[["signal", "fill_time", "fill_px", "stop", "targets", "exit_time", "exit_px", "closed_by",
                     "gold_usd", "net_inr"]].head().to_string(index=False))
    print(f"\nTrial file: {result.trial_path.relative_to(config.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
