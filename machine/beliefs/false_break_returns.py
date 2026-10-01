"""B4-14b - "... price goes the other way, and returns days later" (strategy.md section 4, line 96).

One event = his false breakout (the events of B4-14a). Counted: from the close back, does price first go at least 0.25
of a day's range the other way, and then come back to the level - both within 3 days? (events.away_and_back.)
The fair comparison: the same 20 fake false breaks as B4-14a, the same count.
Plan: plan/work/beliefs/false-break-returns.md.
"""
from __future__ import annotations

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-false-break-returns"


def rows(w):
    fb = EV.false_breaks(w)
    return LV.frame(w, fb["decided"], fb["x_back"], LV.pools(list(fb["pool_back"])), kind=fb["family"],
                    side=fb["side"])


VERSIONS = (
    Belief(FAMILY, "back-within-3-days", "B4-14", "After a false breakout price returns to the level days later",
           "False-liquidity (P2): a false breakout consumes the liquidity -> price goes the other way, returns days "
           "later.",
           "strategy.md section 4, line 96; four-boxes B4-14",
           "his false breakout: from the close back, does price first go at least 0.25 of a day's range the other "
           "way, and then come back to the level?",
           "both within 3 days (864 five-minute candles) of the close back",
           "fake false breaks: 20 fake levels per real one (same weekday and hour, same year, as far from price, same "
           "side), poked and closed back the same way, then the same count",
           "share", rows, noun="false breaks", slices=("kind", "side"),
           code=("machine/beliefs/false_break_returns.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
