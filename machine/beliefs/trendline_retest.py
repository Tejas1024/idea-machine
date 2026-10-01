"""B4-15 - "A broken trendline's retest is a good turning point" (strategy.md section 14, line 165: "Trendline retest
after breakout = valid reversal/'sell' zone (I1)").

One event = the RETEST of a broken trendline of his (3 touches, 5-minute or hour chart - machine/blocks/trendlines.py,
L6b.1): the line BREAKS when a candle CLOSES beyond it by more than settings "line_taken_through_adr" of a day's range
(a real break, not a wick); the retest is the first candle after the break, within 5 days (four-boxes B3-25, D4: "a
trendline retest 5 days"), whose wick comes back to the line (its price on that candle). Counted: does price turn there
- reach the next liquidity the way of the break - before it goes as far back through the line? (levels.turn_race; the
line's price is held at its retest price, a small simplification - the same as L12b.2's.)
The fair comparison ("random touches of old lines"): 20 fake lines per real one - the same slope in day's ranges,
born at the break's weekday and hour of the same year with the close as far beyond them, then retested and raced the
same way. A line nobody drew, touched the same way.
Plan: plan/work/beliefs/trendline-retest.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import settings
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.world import DAY
from machine.blocks import liquidity as LQ
from machine.blocks import trendlines as TL

FAMILY = "belief-trendline-retest"
RETEST_BARS = 5 * DAY


def _retest(w, start, line_at, up_break: bool) -> int:
    """The first candle from `start` within RETEST_BARS whose wick comes back to the line (line_at(i) = its price)."""
    end = min(start + RETEST_BARS, w.n)
    if start >= end:
        return -1
    i = np.arange(start, end)
    px = line_at(i)
    hit = np.flatnonzero(w.low[start:end] <= px) if up_break else np.flatnonzero(w.high[start:end] >= px)
    return int(start + hit[0]) if len(hit) else -1


def retests(w) -> pd.DataFrame:
    def make():
        parts = [w.market.lines(tf, w.fp) for tf in LQ.TRENDLINE_CHARTS if w.market.has(tf)]
        lines = pd.concat([p[p["touches"] == 3] for p in parts if len(p)], ignore_index=True)
        on = TL.on_m5(lines, w.m5).reset_index(drop=True)
        through = float(settings.get("line_taken_through_adr"))
        alive = LV.Alive(w)
        rng = w.rng("trendline_retest")
        out = []
        for ln in on.itertuples():
            s0, gone = int(ln.start), int(ln.gone) if int(ln.gone) >= 0 else w.n
            if s0 >= w.n or s0 >= gone:
                continue
            i = np.arange(s0, min(gone, w.n))
            px = ln.b_price + ln.slope5 * (i - ln.x0)
            beyond = through * ln.ruler
            up = ln.side == "high"                          # a line through highs breaks UP
            br = np.flatnonzero(w.close[i] > px + beyond) if up else np.flatnonzero(w.close[i] < px - beyond)
            if not len(br):
                continue
            b = int(i[br[0]])
            k = _retest(w, b + 1, lambda j: ln.b_price + ln.slope5 * (j - ln.x0), up)
            if k < 0:
                continue
            lp = ln.b_price + ln.slope5 * (k - ln.x0)
            side = "low" if up else "high"                  # after an up-break the line is below price
            r = LV.next_liquidity_r(w, alive, np.array([k]), np.array([lp]), np.array(["high" if up else "low"]))[0]
            x = LV.turn_race(w, np.array([k]), np.array([lp]), np.array([lp]), np.array([side]), np.array([r]))[0]
            # the fake lines: same slope and the close as far beyond, at the break's weekday and hour
            rb = w.ruler[b]
            off = (w.close[b] - (ln.b_price + ln.slope5 * (b - ln.x0))) / rb
            slope_adr = ln.slope5 / rb
            fx = np.full(LV.K_FAKES, np.nan)
            for q, c in enumerate(w.controls(np.array([b]), LV.K_FAKES, rng)[0]):
                if c < 0 or not np.isfinite(r):
                    continue
                rc = w.ruler[c]
                l0 = w.close[c] - off * rc
                kf = _retest(w, c + 1, lambda j: l0 + slope_adr * rc * (j - c), up)
                if kf < 0:
                    continue
                lf = l0 + slope_adr * rc * (kf - c)
                fx[q] = LV.turn_race(w, np.array([kf]), np.array([lf]), np.array([lf]), np.array([side]),
                                     np.array([r / w.ruler[k] * w.ruler[kf]]))[0]
            out.append((k, x, fx, ln.tf, "up (a line through highs)" if up else "down (a line through lows)"))
        return pd.DataFrame(out, columns=["pos", "x", "pool", "chart", "break"])
    return w.memo("trendline_retests", make)


def rows(w):
    t = retests(w)
    return LV.frame(w, t["pos"].to_numpy(np.int64), t["x"], LV.pools(list(t["pool"])) if len(t) else [],
                    chart=t["chart"], broke=t["break"])


VERSIONS = (
    Belief(FAMILY, "retest-within-5-days", "B4-15", "A broken trendline's retest is a good turning point",
           "Trendline retest after breakout = valid reversal/'sell' zone (I1).",
           "strategy.md section 14, line 165; four-boxes B4-15",
           "the first retest (a wick back to the line, within 5 days) of his trendline after a candle CLOSED through "
           "it: does price turn there - reach the next liquidity the way of the break - before it goes as far back "
           "through the line?",
           "the retest within 5 days of the break; then until one of the two is reached, at most 5 days",
           "20 fake lines per real one - the same slope, born at the break's weekday and hour of the same year with the "
           "close as far beyond them - retested and raced the same way",
           "share", rows, noun="retests", slices=("chart", "broke"),
           code=("machine/beliefs/trendline_retest.py", "machine/beliefs/levels.py")),
)
