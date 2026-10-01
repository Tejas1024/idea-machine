"""B4-04b - "3 touches > 2" (strategy.md section 1, line 75).

One event = an equal high or low at the moment its THIRD touch is known (his rule's 3rd touch: machine/blocks/
equal_levels.py). Counted: its pull - how much more often price comes to it within 5 days than to its own 20 fake
levels (the same measure as B4-04a, equal_exact.py).
The fair comparison: the same pull of equal levels at the moment their SECOND touch made them (every equal level), of
the same year and side.
Plan: plan/work/beliefs/equal-three-touches.md.
"""
from __future__ import annotations

import pandas as pd

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.equal_exact import pull

FAMILY = "belief-equal-three-touches"


def rows(w):
    both = pd.concat([pull(w, 3).assign(want=True), pull(w, 2).assign(want=False)], ignore_index=True)
    return LV.group_frame(w, both["pos"], both["pull"], both["want"].to_numpy(),
                          LV.stratum(w, both["pos"], both["side"]), chart=both["tf"], side=both["side"])


VERSIONS = (
    Belief(FAMILY, "third-touch", "B4-04", "Equal highs / lows touched 3 times pull price more than 2 times",
           "Later touch must NOT pass reference; exact strongest; 3>2.",
           "strategy.md section 1, line 75; four-boxes B4-04",
           "an equal high (low) when its 3rd touch is known: how much more often price comes to it within 5 days than "
           "to its own 20 fake levels (in 100)",
           "5 days (1,440 five-minute candles) from the moment the 3rd touch is known",
           "the same pull of equal levels when their 2nd touch made them, of the same year and side",
           "share", rows, noun="equal levels", slices=("chart", "side"), groups=True,
           code=("machine/beliefs/equal_three_touches.py", "machine/beliefs/equal_exact.py",
                 "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
