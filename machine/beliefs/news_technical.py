"""B4-20 - "Technical rules still work at news (false breakout -> stop-run -> reverse)" (strategy.md section 12,
line 153).

One event = his grab + gap trade whose GRAB candle falls inside his news window: Forex Factory red or orange (T7), from
the release to 120 minutes after it (red) or 40 (orange) - when L12b.2 found 3 of 4 releases calm again
(plan/work/L12b2-notes.md section 3b). Counted: its rupees (machine/beliefs/trades.py).
The fair comparison: his grab + gap trades whose grab is outside every news window, same close year and direction.
How the answer reads: "holds" = his setup does BETTER at news; "the opposite holds" = WORSE; "chance" = no clear
difference - which is what "still works" says (Claude's reading of the three words for this belief).
Not already seen by the old project's list (it never split his grab + gap by news); his grab + gap itself is
(type1_grab_gap). Plan: plan/work/beliefs/news-technical.md.
"""
from __future__ import annotations

from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief

FAMILY = "belief-news-technical"


def rows(w):
    t = TR.type1(w)
    n = t["news"].to_numpy(bool)
    return TR.rows_two_groups(w, n, ~n, t)


VERSIONS = (
    Belief(FAMILY, "news-window-vs-not", "B4-20", "Technical rules still work at news",
           "Technical rules still work at news (false breakout -> stop-run -> reverse).",
           "strategy.md section 12, line 153; four-boxes B4-20",
           "his grab + gap trade whose grab candle is inside his news window (red: the release to 120 minutes after, "
           "orange: to 40): rupees after the Rs 40 cost",
           "until the stop or the target",
           "his grab + gap trades whose grab is outside every news window, same close year and direction",
           "inr", rows, groups=True, noun="trades", slices=("direction",),
           code=("machine/beliefs/news_technical.py", *TR.CODE)),
)
