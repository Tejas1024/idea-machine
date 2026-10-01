"""B4-09 - "Hammer + gap is a stronger setup" (strategy.md section 2, line 85; LEARNED #115). ALREADY SEEN.

One event = his grab + gap trade whose GRAB candle is his hammer the trade's way (machine/blocks/hammer.py, rules.py
what_a_hammer_is: a long wick that takes the liquidity and snaps back - the compiler's own _grab_is_hammer). Counted:
its rupees (machine/beliefs/trades.py: a plain stop at his hedge point, the next liquidity as the target, Rs 40).
The fair comparison: his grab + gap trades whose grab candle is NOT a hammer, same close year and direction.
Plan: plan/work/beliefs/hammer-gap.md.
"""
from __future__ import annotations

from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief

FAMILY = "belief-hammer-gap"
SEEN = ("the old project tested a hammer grab against an ordinary grab on these years: hammer_grab (10 tries, "
        "2016-2023, on 5 entries: the 15-minute gap +Rs 6,388 on 40, grab + gap hammers -Rs 1,952)")


def rows(w):
    t = TR.type1(w)
    h = t["hammer"].to_numpy(bool)
    return TR.rows_two_groups(w, h, ~h, t)


VERSIONS = (
    Belief(FAMILY, "hammer-vs-not", "B4-09", "Hammer + gap is a stronger setup",
           "there is a very uh, hammer type of candle which is a signal that it took the liquidity and uh, buy bought "
           "suddenly and the fair value gap is formed that is uh, you know one powerful fair value gap",
           "LEARNED #115; strategy.md section 2, line 85; four-boxes B4-09",
           "his grab + gap trade whose grab candle is his hammer the trade's way: rupees after the Rs 40 cost (a plain "
           "stop at his hedge point, the next liquidity as the target, every setup its own trade)",
           "until the stop or the target",
           "his grab + gap trades whose grab candle is not a hammer, same close year and direction",
           "inr", rows, groups=True, noun="trades", slices=("direction",),
           seen_key=("liquidity_sweep+hammer+fvg", "5m", "gap:near"), seen_note=SEEN,
           code=("machine/beliefs/hammer_gap.py", *TR.CODE)),
)
