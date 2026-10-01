"""B4-06 - "Price goes to the recent liquidity first - today's session first" (strategy.md section 1, line 72;
LEARNED #130).

One event = a session high (or low) at the moment his window ends and it is known (machine/blocks/sessions.py final:
Asia, London, New York), with an OLDER level of his on the OTHER side of price - the nearest one known at least a day
before (equal, session, trendline, hour / day / week swing), both within one day's range of price. Counted: which is
reached first - the new session level or the older one?
Why the other side: two levels on the same side are met in the order of their distance, whatever their age.
The fair comparison: 20 fake pairs, born at the same weekday and hour of the same year, the "recent" one as far from
price on the same side, the "older" one as far on the other side - the same race. A walk reaches the nearer of the two
first more often; the fake pairs have the same distances, so that is the same for both.
Plan: plan/work/beliefs/recent-first.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief
from machine.beliefs.world import DAY, DOWN, UP
from machine.blocks import sessions as SES

FAMILY = "belief-recent-first"
NEAR_ADR = 1.0                  # "both near": within one day's range of price - Claude's
MAX_BARS = 5 * DAY


def rows(w):
    fin = SES.final(w.m5, "5m")
    pos = w.at_or_after(fin["known_at"]) - 1
    ok = (pos >= 0) & (pos < w.n)
    fin, pos = fin[ok].reset_index(drop=True), pos[ok]
    alive = LV.Alive(w)
    out = []
    for side, col in (("high", "high"), ("low", "low")):
        recent = fin[col].to_numpy(float)
        close = w.close[pos]
        ruler = w.ruler[pos]
        sg = 1.0 if side == "high" else -1.0
        d_r = sg * (recent - close) / ruler
        other = "low" if side == "high" else "high"
        older, _ = alive.nearest(pos, np.full(len(pos), other), close, known_by=pos + 1 - DAY)
        d_o = -sg * (older - close) / ruler
        keep = np.isfinite(d_r) & np.isfinite(d_o) & (d_r > 0) & (d_o > 0) & (d_r <= NEAR_ADR) & (d_o <= NEAR_ADR)
        p, r_px, o_px = pos[keep], recent[keep], older[keep]
        up, down = (r_px, o_px) if side == "high" else (o_px, r_px)
        ans, _ = w.race(p + 1, up, down, MAX_BARS)
        first = UP if side == "high" else DOWN
        x = np.where((ans == UP) | (ans == DOWN), (ans == first).astype(float), np.nan)
        c = w.controls(p, LV.K_FAKES, w.rng(f"recent_first_{side}"))
        cc = np.where(c >= 0, c, 0)
        fr = w.close[cc] + sg * d_r[keep][:, None] * w.ruler[cc]
        fo = w.close[cc] - sg * d_o[keep][:, None] * w.ruler[cc]
        fu, fd = (fr, fo) if side == "high" else (fo, fr)
        fa, _ = w.race(cc.ravel() + 1, fu.ravel(), fd.ravel(), MAX_BARS)
        fx = np.where((fa == UP) | (fa == DOWN), (fa == first).astype(float), np.nan).reshape(c.shape)
        fx = np.where(c >= 0, fx, np.nan)
        out.append(LV.frame(w, p, x, LV.pools(fx), session=fin["session"].to_numpy()[keep],
                            side=np.full(len(p), "session high" if side == "high" else "session low")))
    return pd.concat(out).sort_index(kind="stable")


VERSIONS = (
    Belief(FAMILY, "session-vs-older", "B4-06", "Price goes to the recent liquidity first (today's session first)",
           "the more priority on the recent liquidity sessions ... today's and the nows that have the more priority "
           "because price moved towards that first",
           "LEARNED #130; strategy.md section 1, line 72; four-boxes B4-06",
           "a session high (low) when his window ends, with an older level of his on the other side of price (known at "
           "least a day before), both within one day's range: is the new session level reached first?",
           "until one of the two is reached, at most 5 days",
           "20 fake pairs per event - born at the same weekday and hour of the same year, the 'recent' one as far "
           "from price on the same side, the 'older' one as far on the other side - the same race",
           "share", rows, noun="session levels", slices=("session", "side"),
           code=("machine/beliefs/recent_first.py", "machine/beliefs/levels.py")),
)
