"""B4-05 - "Equal highs and lows are a pair: after one side is taken, price goes to the other" (strategy.md section 1,
line 75; LEARNED #21, 6 Sep). ALREADY SEEN - by this machine's own idea #1.

One event = a fresh take of equal highs (or lows) while equal levels of the OTHER side are alive (known, not taken):
does price reach the nearest of them before it goes as far past the taken level? (levels.turn_race, R = the distance
between the two sides; at least 0.1 and at most 2.5 day's ranges.)
The fair comparison: fake pairs - the 20 fake levels of B4-01 for each taken level, then the same race to a fake other
side as far away.
Plan: plan/work/beliefs/equal-pair.md.
"""
from __future__ import annotations

import numpy as np

from machine.beliefs import events as EV
from machine.beliefs import levels as LV
from machine.beliefs.engine import Belief

FAMILY = "belief-equal-pair"
SEEN = ("this machine's idea #1 traded this belief on these years (take one side of equal highs / lows -> a break -> "
        "the gap -> aim at the other side; its check years 2021-2023 were opened on 29 Sep 2026: with a plain stop it "
        "lost in all 3 years, 232 of 260 trades reached the grab's extreme first)")


def rows(w):
    def make():
        t = EV.takes(w)
        e = t[(t["family"] == "equal") & (t["birth"] >= 0) & np.isfinite(t["dist_adr"]) & (t["dist_adr"] > 0)]
        e = e.reset_index(drop=True)
        alive = LV.Alive(w)
        other = np.where(e["side"] == "high", "low", "high")
        px, _ = alive.nearest(e["pos"].to_numpy(), other, e["level"].to_numpy(float), kinds=("equal",))
        r = np.abs(e["level"].to_numpy(float) - px)
        ruler = w.ruler[e["pos"].to_numpy()]
        ok = np.isfinite(r) & (r >= LV.NEXT_MIN_ADR * ruler) & (r <= LV.NEXT_MAX_ADR * ruler)
        e = e[ok].reset_index(drop=True)
        r = r[ok]
        side = e["side"].to_numpy()
        x = LV.turn_race(w, e["pos"].to_numpy(), e["level"].to_numpy(float), e["level"].to_numpy(float), side, r)
        c, fp = LV.fake_prices(w, e["birth"].to_numpy(), side, e["dist_adr"].to_numpy(float), w.rng("equal_pair"))
        life_bars = LV.life(w).drop_duplicates("level_id").set_index("level_id").loc[e["level_id"], "life_bars"]
        jf, f, s = EV.fake_take_points(w, c, fp, side, life_bars.to_numpy(np.int64))
        pool = EV.fake_races(w, jf, f, s, r / ruler[ok], LV.K_FAKES)
        return LV.frame(w, e["pos"], x, LV.pools(pool), side=side, chart=e["tf"])
    return w.memo("equal_pair", make)


VERSIONS = (
    Belief(FAMILY, "other-side-first", "B4-05", "Equal highs and lows are a pair: after one side is taken, price "
           "goes to the other",
           "if it goes to the equal highs first, then it will eventually move to the equal lows ... Even if it takes "
           "the equal lows, then it will go to the opposite, opposite.",
           "LEARNED #21 (6 Sep); strategy.md section 1, line 75; four-boxes B4-05",
           "a fresh take of equal highs (lows) while equal levels of the other side are alive: does price reach the "
           "nearest of them before it goes as far past the taken level?",
           "until one of the two is reached, at most 5 days",
           "fake pairs: 20 fake levels per taken one (same weekday and hour, same year, as far from price, same "
           "side), each taken by its first poke, then the same race to a fake other side as far away",
           "share", rows, noun="takes", slices=("side", "chart"), seen_note=SEEN,
           code=("machine/beliefs/equal_pair.py", "machine/beliefs/events.py", "machine/beliefs/levels.py")),
)
