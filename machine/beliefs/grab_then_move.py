"""B4-01 - "Liquidity is grabbed first, then the real move" (strategy.md section 1, line 72). ALREADY SEEN.

One event = a fresh take of one of his levels that may start a setup (liquidity.sweeps: session highs / lows, equal
highs / lows, trendlines, day and week swings - a wick past it is enough, #95; a level swept once is finished, #129).
Counted: does price then reach the NEXT liquidity on the other side (the real move) before it goes as far past the
taken level? (machine/beliefs/levels.py: the race after a take.)
The fair comparison: 20 fake levels per real one, born at the same weekday and hour of the same year, as far from
price, on the same side; each is "taken" by its first poke within the real level's life, then the same race, as far.
Plan: plan/work/beliefs/grab-then-move.md.
"""
from __future__ import annotations

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-grab-then-move"
SEEN = ("the old project raced price after his grab + gap on these years: type1_race_causal (4,504 races, 51.2% "
        "his way, 61 tries) and early_sweep_direction (28 tries, 47-51%) - four-boxes B4-01 'yes (Type 1)'")


def rows(w):
    t = EV.takes(w)
    t = t[t["ok"]]
    return LV.frame(w, t["pos"], t["x"], t["pool"], kind=t["family"], side=t["side"])


VERSIONS = (
    Belief(FAMILY, "next-liquidity", "B4-01", "Liquidity is grabbed first, then the real move",
           "Meaning: levels where stops cluster; grabbed first, then the real move.",
           "strategy.md section 1, line 72; four-boxes B4-01",
           "a fresh take of one of his levels that may start a setup (session, equal, trendline, day / week swing): "
           "does price reach the next liquidity on the OTHER side before it goes as far past the taken level?",
           "until one of the two is reached, at most 5 days (1,440 five-minute candles)",
           "20 fake levels per real one - born at the same weekday and hour of the same year, as far from price, on "
           "the same side - taken by their first poke, then the same race, as far",
           "share", rows, noun="takes", slices=("kind", "side"),
           seen_key=("liquidity_sweep+break_of_structure+fvg", "5m", "gap:far"), seen_note=SEEN,
           code=("machine/beliefs/grab_then_move.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
