"""His grab + gap trades, for the beliefs that are about rupees (L13b): every Type 1 setup on the world's candles, each
replayed ALONE with a plain stop at his hedge point and the next liquidity as the target, with what the setup was made
of.

The form: machine/tester/hedging.py HIS_GRAB_GAP - the Type 1 L19a closes hedged legs with. The same steps, entry,
hedge point, target and do-not-enter rules as L19b's entry-type1 (the same version fingerprint, v-917f35901a):
his obvious liquidity taken -> a break of structure -> the FIRST gap of the breaking move -> a limit at its near edge
(first visit only) -> the hedge point at the grab's extreme -> the next liquidity (#9, #33, #61, #62, #65, #68, #111,
#119). Already seen: prior-tries type1_grab_gap (29 tries, 2016-2023) - every rupee belief built on it says so.

Every setup is its own trade (the tester's trades_at_once = every setup): a belief about a KIND of setup compares
setups, so none is dropped because another trade was on. His rule is one trade at a time (#55, M18); this is a
measurement of setups, never his trading. The hedge is tested as a plain stop (the compiler says so): a setup is
"stronger" when price reaches the target before the hedge point more often, or by more rupees.
Money: rupees after his Rs 40 a position (#54, #115), 0.01 lot. A trade still open at the end of the years has no money
(lesson B6) and is not counted.

What each setup carries (known at the setup's moment, nothing later):
  family      the kind of level the grab took: equal, session, trendline, swing (a day or week swing)
  hammer      the grab's candle is his hammer, the trade's way (the compiler's own _grab_is_hammer, #115)
  gap_adr     the entry gap's size in day's ranges
  at_zone     the entry price is inside one of his strong zones alive at the setup's candle
  adr_with    a buy at discount, a sell at premium (premium_discount.place, the near-the-zone reading)
  htf_with    the hour AND the day bias with the trade (bias.py, the last candles closed by then)
  news        the grab's candle inside his news window: from a release to 120 minutes after it (red), 40 (orange) -
              L12b.2's "calm again" times (plan/work/L12b2-notes.md section 3b)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.beliefs import levels as LV
from machine.blocks import premium_discount as PD
from machine.blocks import zones as ZN
from machine.data import bars as BARS
from machine.ideas.compile import compile_idea
from machine.ideas.form import Idea
from machine.tester import engine as ENG
from machine.tester.hedging import HIS_GRAB_GAP

SEEN_KEY = ("liquidity_sweep+break_of_structure+fvg", "5m", "gap:near")
SEEN = "his grab + gap was tested on these years by the old project: type1_grab_gap (29 tries, 2016-2023)"
NEWS_WINDOW_MIN = {"red": 120, "orange": 40}
CODE = ("machine/beliefs/trades.py", "machine/beliefs/levels.py")


def compiled(w):
    """His Type 1 setups on the world's candles (with the compiler's trace) and the world's 1-minute bars, made once."""
    def make():
        comp = compile_idea(Idea.load(HIS_GRAB_GAP), w.market, trace=True)
        m1 = (w.market.bars("1m") if w.market.has("1m")
              else BARS.load_bars("1m", w.t[0], w.t[-1] + pd.Timedelta(minutes=4)))
        return comp, m1
    return w.memo("type1_compiled", make)


def type1(w) -> pd.DataFrame:
    """One row per filled Type 1 setup: net_inr (NaN when still open at the end), exit_time, direction, and its facts."""
    def make():
        comp, m1 = compiled(w)
        facts = {r["signal"]: r["facts"] for r in comp.trace if r.get("signal")}
        rep = ENG.replay(comp.signals, m1, "5m", trades_at_once=len(comp.signals) + 1)
        sig = {s.id: s for s in comp.signals}
        hammers = w.market.hammers(w.fp)
        zlife = w.market.zone_life(w.fp)
        adr = w.market.adr_zones(w.fp, "near_the_zone")
        windows = news_windows(w)
        rows = []
        for t in rep.trades:
            f = facts[t.signal_id]
            s = sig[t.signal_id]
            buy = t.direction == "buy"
            g = int(w.at_or_after([pd.Timestamp(s.known_at)])[0]) - 1          # the candle that made the signal
            sweep_bar = int(w.at_or_after([pd.Timestamp(f["sweep"]["bar"])])[0])
            here = ZN.alive(zlife, g)
            entry = float(s.entry_price)
            at_zone = bool(((here["lo"] <= entry) & (entry <= here["hi"])).any()) if len(here) else False
            place = PD.place(entry, adr.iloc[g])
            want = 1 if buy else -1
            htf = all(w.market.bias_at(tf, w.fp, s.known_at) == want for tf in ("1h", "1d"))
            rows.append({
                "signal": t.signal_id, "direction": t.direction, "known_at": pd.Timestamp(s.known_at),
                "status": t.status, "net_inr": t.net_inr if t.status == "closed" else np.nan,
                "exit_time": pd.Timestamp(t.exit_ts) if t.status == "closed" else pd.NaT,
                "family": str(f["sweep"]["kind"]).split("_", 1)[0],
                "hammer": bool(hammers[t.direction].iat[sweep_bar]),
                "gap_adr": float(f["gap"]["size"]) / w.ruler[g],
                "at_zone": at_zone,
                "adr_with": place == (PD.DISCOUNT if buy else PD.PREMIUM),
                "htf_with": bool(htf),
                "news": bool(in_windows(windows, w.t[sweep_bar].value)),
            })
        return pd.DataFrame(rows)
    return w.memo("type1_trades", make)


def news_windows(w) -> tuple[np.ndarray, np.ndarray]:
    """His news windows inside the world, server time in ns: (starts, the latest end so far) - release to 120 minutes
    after (red) or 40 (orange)."""
    from machine.data import clock
    from machine.news import forexfactory as FF

    rel = LV.news_times(w)
    ff = FF.load()
    ff = ff[ff["colour"].isin(["red", "orange"])]
    minute = pd.DatetimeIndex(clock.utc_to_server(pd.DatetimeIndex(ff["utc"]).floor("min"))).as_unit("ns").asi8
    red = set(minute[(ff["colour"] == "red").to_numpy()])
    length = np.array([NEWS_WINDOW_MIN["red" if r in red else "orange"] for r in rel], dtype=np.int64)
    ends = rel + length * 60 * 1_000_000_000
    return rel, np.maximum.accumulate(ends) if len(ends) else ends


def in_windows(windows, t_ns) -> bool:
    starts, ends = windows
    k = int(np.searchsorted(starts, t_ns, "right")) - 1
    return k >= 0 and ends[k] > t_ns


def rows_two_groups(w, want: np.ndarray, compare: np.ndarray, t: pd.DataFrame, slices: dict | None = None,
                    keys=("year", "direction")):
    """The engine's rows for "these setups earn more than those" (a two-group belief): the wanted trades (group A) and
    the comparison trades (B), each with its rupees, compared inside the same keys (the close year and the direction;
    none = all together). Only setups MADE in the judged years that also CLOSED in them (open at the end: no money)."""
    made = np.isin(pd.DatetimeIndex(t["known_at"]).year, w.years)     # setups of the warm-up weeks are not events
    closed = t["exit_time"].notna().to_numpy() & np.isin(pd.DatetimeIndex(t["exit_time"]).year, w.years)
    both = (np.asarray(want) | np.asarray(compare)) & made & closed
    e = t[both]
    parts = []
    if "year" in keys:
        parts.append(pd.DatetimeIndex(e["exit_time"]).year.astype(str).to_numpy())
    if "direction" in keys:
        parts.append(e["direction"].to_numpy().astype(str))
    strat = ["|".join(p) for p in zip(*parts)] if parts else ["all"] * len(e)
    out = pd.DataFrame({"x": e["net_inr"].to_numpy(float), "group": np.where(np.asarray(want)[both], "A", "B"),
                        "stratum": np.asarray(strat, dtype=object), "direction": e["direction"].to_numpy()},
                       index=pd.DatetimeIndex(e["exit_time"]).rename("closed"))
    for name, col in (slices or {}).items():
        out[name] = e[col].to_numpy()
    return out.sort_index(kind="stable")
