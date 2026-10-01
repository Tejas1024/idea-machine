"""B4-17 - "Near the green ADR zone -> over-sold today -> likely bounce (esp. no positive news)" (strategy.md section
7, line 107).

One event = the first 5-minute candle of a broker day whose low comes to his GREEN zone (its top: the day's open less
half of the smaller of the 5- and 10-day ranges - machine/blocks/adr.py, his Pine study); and the same the other way
round at the RED zone ("and red", four-boxes B4-17). Counted: from that candle's close, does price bounce - go up to
the next liquidity above
(from the red zone: down to the next below) - before it goes as far the other way? (levels.move_race.)
The fair comparison: 20 random moments per event at the same weekday and hour of the same year, the same race as far
(in day's ranges), the same way. (Fake levels "at the same distance" from the day's open would be the green zone of
another day: the zone IS a distance from the open. So the comparison is "a random touch": price at any moment.)
Plan: plan/work/beliefs/adr-green-bounce.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-adr-green-bounce"


def first_of_day(day: np.ndarray, hit: np.ndarray) -> np.ndarray:
    """The first candle of each day where `hit` is true."""
    pos = np.flatnonzero(hit)
    if not len(pos):
        return pos
    keep = np.r_[True, day[pos][1:] != day[pos][:-1]]
    return pos[keep]


def rows(w):
    z = w.market.adr_zones(w.fp, "near_the_zone")
    day = z["day"].to_numpy()
    alive = LV.Alive(w)
    parts = []
    for name, hit, turn_side, way in (("green", w.low <= z["dn_hi"].to_numpy(float), "high", 1.0),
                                      ("red", w.high >= z["up_lo"].to_numpy(float), "low", -1.0)):
        j = first_of_day(day, hit & np.isfinite(z["dn_hi"].to_numpy(float)))
        j = j[j + 1 < w.n]
        r = LV.next_liquidity_r(w, alive, j, w.close[j], np.full(len(j), turn_side))
        x = LV.move_race(w, j + 1, w.close[j], r, way)
        c = w.controls(j, LV.K_FAKES, w.rng(f"adr_green_bounce_{name}"))
        cc = np.where(c >= 0, c, 0)
        rr = (r / w.ruler[j])[:, None] * w.ruler[cc]
        fx = LV.move_race(w, (cc + 1).ravel(), w.close[cc].ravel(), rr.ravel(), way).reshape(c.shape)
        fx = np.where(c >= 0, fx, np.nan)
        d0 = pd.DatetimeIndex(z.index[j]).normalize()
        news = LV.news_between(w, d0.as_unit("ns").asi8, (d0 + pd.Timedelta(days=1)).as_unit("ns").asi8)
        parts.append(LV.frame(w, j, x, LV.pools(fx), zone=np.full(len(j), name),
                              news=np.where(news, "his news that day", "no news that day")))
    return pd.concat(parts).sort_index(kind="stable")


VERSIONS = (
    Belief(FAMILY, "next-liquidity", "B4-17", "Near the green ADR zone price is over-sold and likely bounces",
           "day-extension (near green -> over-sold today -> likely bounce, esp. no positive news)",
           "strategy.md section 7, line 107; four-boxes B4-17",
           "the first candle of a day that comes to his green zone (or his red zone): from its close, does price "
           "bounce to the next liquidity above (below, at the red zone) before it goes as far the other way?",
           "until one of the two is reached, at most 5 days",
           "20 random moments per event at the same weekday and hour of the same year: the same race, as far, the same "
           "way",
           "share", rows, noun="zone arrivals", slices=("zone", "news"),
           code=("machine/beliefs/adr_green_bounce.py", "machine/beliefs/levels.py")),
)
