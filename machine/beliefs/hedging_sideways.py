"""B4-21 - "Hedging works best sideways; the grid prefers sideways" (strategy.md section 11, line 144; section 13,
line 160).

One event = a trade of his grab + gap entry managed HIS way (the hedging tester, machine/tester/hedging.py, L19a: his
T12 table, one trade at a time - his rule, M18) that was opened in a SIDEWAYS week, by B4-22's measure fixed before
looking (sideways_most.py: the week ended less than half of its range from where it started). Counted: the whole
trade's rupees after Rs 40 a position (every leg pays it).
The fair comparison: his hedged trades opened in a TRENDING week. There are few of them (L19b: 15 in 2016-2020 - a
hedged trade stays open about two months), so the comparison is every trending-week trade of the judged years, both
directions.
The week's type is known only when the week ends: this says WHEN hedging did well, it is not a rule anyone could trade.
Plan: plan/work/beliefs/hedging-sideways.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import trades as TR
from machine.beliefs.engine import Belief
from machine.beliefs.sideways_most import SIDEWAYS_BELOW, efficiency
from machine.tester import engine as ENG
from machine.tester import hedging

FAMILY = "belief-hedging-sideways"


def week_is_sideways(w) -> dict[int, bool]:
    """{the week's Monday (a day number): sideways} for every broker week of the world with 4 days or more."""
    monday = (w.t.normalize() - pd.to_timedelta(w.t.dayofweek, unit="D")).as_unit("s").asi8 // 86_400
    out = {}
    for wk in np.unique(monday):
        idx = np.flatnonzero(monday == wk)
        idx = idx[w.weekday[idx] <= 4]
        if len(idx) >= 4 * 276:
            path = np.r_[w.open[idx[0]], w.close[idx]]
            out[int(wk)] = bool(efficiency(path[None, :])[0] < SIDEWAYS_BELOW)
    return out


def hedged(w) -> pd.DataFrame:
    def make():
        comp, m1 = TR.compiled(w)
        book = ENG.bars_book(m1, "5m")
        facts = hedging.MarketFacts.build(w.market, w.fp, book, setups=hedging.setups_from(comp.signals, book))
        rep = hedging.replay(comp.signals, facts)
        weeks = week_is_sideways(w)
        rows = []
        for t in rep.trades:
            fill = pd.Timestamp(t.fill_ts)
            wk = int((fill.normalize() - pd.Timedelta(days=fill.dayofweek)).value // 86_400_000_000_000)
            rows.append({"signal": t.signal_id, "direction": t.direction, "known_at": pd.Timestamp(t.known_at),
                         "net_inr": t.net_inr if t.status == "closed" else np.nan,
                         "exit_time": pd.Timestamp(t.exit_ts) if t.status == "closed" else pd.NaT,
                         "week": {True: "sideways", False: "trending"}.get(weeks.get(wk), "short week"),
                         "positions": t.positions})
        return pd.DataFrame(rows, columns=["signal", "direction", "known_at", "net_inr", "exit_time", "week",
                                           "positions"])
    return w.memo("type1_hedged", make)


def rows(w):
    t = hedged(w)
    wk = t["week"].to_numpy()
    return TR.rows_two_groups(w, wk == "sideways", wk == "trending", t, keys=())


VERSIONS = (
    Belief(FAMILY, "sideways-vs-trending-week", "B4-21", "Hedging works best sideways",
           "Context: works best sideways.",
           "strategy.md section 11, line 144; section 13, line 160 ('Grid prefers sideways'); four-boxes B4-21",
           "a trade of his grab + gap managed his way (his hedging, one trade at a time) opened in a sideways week: "
           "the whole trade's rupees after Rs 40 a position",
           "until the whole trade is closed (no timeout, #104)",
           "his hedged trades opened in a trending week (every one of the judged years - there are few)",
           "inr", rows, groups=True, noun="hedged trades", slices=("direction",), seen_key=TR.SEEN_KEY,
           seen_note="his grab + gap with his hedging was run on these years by the old project (type1_grab_gap, "
                     "29 tries, re-counted under every manager) and by this machine (L19a, L19b)",
           code=("machine/beliefs/hedging_sideways.py", "machine/beliefs/sideways_most.py", *TR.CODE)),
)
