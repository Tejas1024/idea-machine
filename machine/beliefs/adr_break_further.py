"""B4-18 - "A break below the lower ADR zone can mean a further fall (bad news, strong selling)" (strategy.md section
7, line 107; LEARNED #76).

One event = the first 5-minute candle of a broker day that CLOSES below his green zone (below its bottom: the day's
open less half of the larger of the 5- and 10-day ranges - machine/blocks/adr.py). Counted: from that close, does price
fall further - down to the next liquidity below - before it goes as far back up? (levels.move_race.)
The fair comparison: 20 random moments per event at the same weekday and hour of the same year, the same race as far
(in day's ranges), down first.
Only the lower zone: his words (#76: "if it breaks the lower range of ADR, lower zone, it will go further low").
Plan: plan/work/beliefs/adr-break-further.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import levels as LV
from machine.beliefs.adr_green_bounce import first_of_day
from machine.beliefs.engine import Belief

FAMILY = "belief-adr-break-further"


def rows(w):
    z = w.market.adr_zones(w.fp, "near_the_zone")
    dn_lo = z["dn_lo"].to_numpy(float)
    j = first_of_day(z["day"].to_numpy(), np.isfinite(dn_lo) & (w.close < dn_lo))
    j = j[j + 1 < w.n]
    alive = LV.Alive(w)
    r = LV.next_liquidity_r(w, alive, j, w.close[j], np.full(len(j), "low"))
    x = LV.move_race(w, j + 1, w.close[j], r, -1.0)
    c = w.controls(j, LV.K_FAKES, w.rng("adr_break_further"))
    cc = np.where(c >= 0, c, 0)
    rr = (r / w.ruler[j])[:, None] * w.ruler[cc]
    fx = LV.move_race(w, (cc + 1).ravel(), w.close[cc].ravel(), rr.ravel(), -1.0).reshape(c.shape)
    fx = np.where(c >= 0, fx, np.nan)
    weekday = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[w.weekday[j]]
    return LV.frame(w, j, x, LV.pools(fx), weekday=weekday)


VERSIONS = (
    Belief(FAMILY, "next-liquidity", "B4-18", "A break below the lower ADR zone can mean a further fall",
           "if it breaks the lower range of ADR, lower zone, it will go further low",
           "LEARNED #76; strategy.md section 7, line 107; four-boxes B4-18",
           "the first candle of a day that closes below his green zone: from that close, does price fall to the next "
           "liquidity below before it goes as far back up?",
           "until one of the two is reached, at most 5 days",
           "20 random moments per event at the same weekday and hour of the same year: the same race, as far, down "
           "first",
           "share", rows, noun="breaks", slices=("weekday",),
           code=("machine/beliefs/adr_break_further.py", "machine/beliefs/adr_green_bounce.py",
                 "machine/beliefs/levels.py")),
)
