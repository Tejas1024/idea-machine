"""B4-14a - "A false breakout uses up the liquidity -> price goes the other way" (strategy.md section 4, line 96;
section 14, line 164). ALREADY SEEN.

One event = a take of his level that was his false breakout (patterns.false_breaks' reading: a candle closed back on
the side price came from before any closed beyond, within settings "false_break_close_back_bars" = 1 hour), known at
that close. Counted: from the close back, does price reach the next liquidity the other way before it goes as far past
the level? (levels.turn_race, the candles after the close back.)
The fair comparison: fake false breaks - the 20 fake levels of B4-01, taken by a poke and closed back the same way.
The second half of his sentence ("returns days later") is B4-14b (false_break_returns.py).
Plan: plan/work/beliefs/false-break-other-way.md.
"""
from __future__ import annotations

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-false-break-other-way"
SEEN = ("the old project traded this on these years: p2_false_liquidity (6 tries, 2016-2023: an obvious level taken, "
        "the break false, then a gap the other way; +Rs 6,697, but the hour-and-day control alone +Rs 5,810)")


def rows(w):
    fb = EV.false_breaks(w)
    return LV.frame(w, fb["decided"], fb["x_other_way"], LV.pools(list(fb["pool_other_way"])), kind=fb["family"],
                    side=fb["side"])


VERSIONS = (
    Belief(FAMILY, "next-liquidity", "B4-14", "A false breakout uses up the liquidity - price goes the other way",
           "False-liquidity (P2): a false breakout consumes the liquidity -> price goes the other way, returns days "
           "later.",
           "strategy.md section 4, line 96; section 14, line 164; four-boxes B4-14",
           "a take of his level that closed back within an hour (his false breakout): from that close, does price "
           "reach the next liquidity the other way before it goes as far past the level?",
           "until one of the two is reached, at most 5 days",
           "fake false breaks: 20 fake levels per real one (same weekday and hour, same year, as far from price, same "
           "side), poked and closed back the same way, then the same race",
           "share", rows, noun="false breaks", slices=("kind", "side"),
           seen_key=("liquidity_sweep+false_breakout+fvg", "5m", "gap:near"), seen_note=SEEN,
           code=("machine/beliefs/false_break_other_way.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
