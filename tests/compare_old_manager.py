"""L19a step 2: the hedging tester (machine/tester/hedging.py) against the old engine's manager
(old/ml/agent/manager.py), trade by trade, on the SAME setups and the SAME market facts, 2016-2020.

    .venv\\Scripts\\python -m tests.compare_old_manager            # prints, writes plan/work/L19a/old-vs-new.md
    .venv\\Scripts\\python -m tests.compare_old_manager --no-write

Tests may import the old code to compare (likhith.md R2); machine/ never does. What both sides get:
  the entries   the new tester's own trades of idea #1 (v-8cc96cf812): each one's fill (5-minute bar, price), hedge
                point and target go into the old manage(d, S, i_fill, direction, entry, first_hedge, target, P).
  his setups    the SAME setups close legs on both sides: the new MarketFacts.setups become the old reversal index S
                (bar = the 5-minute bar that made it, fill = the 5-minute bar its order filled in, px = its order price,
                tgt / hdg = its target / hedge point).
  the market    the same 5-minute and 1-minute bars; the same ruler (the new ADR, as d.adr); the same bias (the new
                hour / day / week bias known at each bar's open); the same liquidity (d.book.nearest is answered by the
                new facts' next_level at that bar's open); the same news (Forex Factory red + orange, put into the old
                manager's news cache).
Two old runs:
  shipped  manager.DEFAULTS as the old engine shipped them: a leg closes only in profit (#105), counted from the price
           it opened at, the whole trade (#112) may close a red leg while hedged, the leg left may end red (#22).
  T12      the switches his T12 table changed: the leg the setup points against, even at a loss, 1:4 from the CURRENT
           entry (#134: hedged_close any_rr, profit_ref current); hedged closes by that leg's own 1:4 only
           (hedged_whole leg); the whole trade decides only the last leg's close (#112: leftover_close basket).
What is left between "T12" and the new tester is how the two engines work: plan/work/L19a-differences.md explains it.
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from machine import config, settings
from machine.data import clock
from machine.data.bars import load_bars
from machine.ideas.compile import Market, compile_idea
from machine.ideas.form import Idea
from machine.tester import engine, hedging

OUT = config.WORK_DIR / "L19a" / "old-vs-new.md"
T12 = {"hedged_close": "any_rr", "profit_ref": "current", "hedged_whole": "leg", "leftover_close": "basket"}
FIVE = pd.Timedelta(minutes=5)


def _old():
    agent = str(config.ROOT / "old" / "ml" / "agent")
    if agent not in sys.path:
        sys.path.insert(0, agent)
    import manager as old_manager            # noqa: E402 - the old engine, for comparing only
    return old_manager


class OldBook:
    """d.book for the old manager: nearest() answered by the new facts at the open of bar i (the same levels the new
    manager reads at that moment). One level (the old callers that ask for more are switched off)."""

    def __init__(self, facts, opens: pd.DatetimeIndex):
        self.facts, self.opens = facts, opens

    def nearest(self, i, side, price, max_adr, adr_here, sweepable_only=True, max_n=1, include_targets=False):
        lv = self.facts.next_level("high" if side > 0 else "low", float(price), self.opens[int(i)],
                                   float(max_adr) * float(adr_here))
        return [] if lv is None else [(round(lv, 2), 0)]


def old_data(market: Market, fp: str, facts, m1: pd.DataFrame) -> SimpleNamespace:
    m5 = market.bars("5m")
    opens = pd.DatetimeIndex(m5.index)
    open_ns = opens.as_unit("ns").asi8
    d = SimpleNamespace()
    d.o, d.h, d.l, d.c = (m5[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    d.t = pd.DatetimeIndex(m5["utc"]).as_unit("s").asi8                  # UTC seconds, as the old news times
    d.spread = m5["spread"].to_numpy(np.float32)
    d.adr = np.asarray(market.ruler("5m", fp), dtype=float)
    for tf, name in (("1h", "bias_h1"), ("1d", "bias_d1"), ("1w", "bias_w1")):
        tab = market.bias(tf, fp)
        known = pd.DatetimeIndex(tab["known_at"]).as_unit("ns").asi8
        k = np.searchsorted(known, open_ns, "right") - 1
        setattr(d, name, np.where(k >= 0, tab["bias"].to_numpy()[np.maximum(k, 0)], 0).astype(np.int8))
    m1_ns = pd.DatetimeIndex(m1.index).as_unit("ns").asi8
    d.m1_at = np.searchsorted(m1_ns, open_ns, "left").astype(np.int64)
    d.m1_h, d.m1_l = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
    d.n, d.n_m1 = len(m5), len(m1)
    d.book = OldBook(facts, opens)
    d.open_ns = open_ns
    return d


def old_index(facts, d) -> dict:
    """His setups as the old manager's reversal index (manager.setup_index's arrays)."""
    ss = facts.setups
    bar = np.searchsorted(d.open_ns, np.array([s.known for s in ss], dtype=np.int64) - FIVE.value, "left")
    fill_ns = facts.book.t[np.array([s.fill_i for s in ss], dtype=np.int64)]
    fill = np.searchsorted(d.open_ns, fill_ns, "right") - 1
    o = np.argsort(bar, kind="stable")
    fill_o = fill[o].astype(np.int64)
    ford = np.argsort(fill_o, kind="stable")
    return dict(bar=bar[o].astype(np.int64), dir=np.array([hedging.SIGN[s.direction] for s in ss], np.int8)[o],
                px=np.array([s.order_px for s in ss], float)[o], fill=fill_o,
                win=np.full(len(ss), 10**9, np.int64), tgt=np.array([s.target for s in ss], float)[o],
                hdg=np.array([s.hedge for s in ss], float)[o], ford=ford, fsorted=fill_o[ford], ids=[ss[k].id for k in o])


def bar_of(d, ts) -> int:
    return int(np.searchsorted(d.open_ns, pd.Timestamp(ts).as_unit("ns").value, "right") - 1)


def new_events(t: hedging.HedgedTrade, d) -> list[tuple]:
    out = []
    for r in t.story:
        kind = {"hedge": "hedge", "news hedge": "hedge", "close": "close", "trail": "trail"}.get(r["what"])
        if kind:
            out.append((kind, bar_of(d, r["ts"]), r["price"], r["words"]))
    return out


def old_events(res: dict) -> list[tuple]:
    out = []
    for i, px, kind, text in res["log"]:
        k = {"hedge": "hedge", "book": "close", "flip": "close", "exit": "close", "reverse": "reverse",
             "trail": "trail"}.get(kind)
        if k:
            out.append((k, int(i), round(float(px), 2), f"{kind}: {text}"))
    return out


def first_split(a: list, b: list, tol: float = 1.0) -> int | None:
    """The first event where the two sides really part: a different kind, more than one 5-minute bar apart, or more
    than `tol` dollars apart. Smaller gaps are the two engines' clocks and prices (counted apart, not a split)."""
    for k in range(max(len(a), len(b))):
        if k >= len(a) or k >= len(b):
            return k
        x, y = a[k], b[k]
        if x[0] != y[0] or abs(x[1] - y[1]) > 1 or abs(x[2] - y[2]) > tol:
            return k
    return None


def near_misses(a: list, b: list, upto: int | None) -> tuple[int, int, int]:
    """Before the split: (events the same, a bar apart, a price apart by more than 5 cents)."""
    n = len(a) if upto is None else upto
    same = bar = px = 0
    for x, y in zip(a[:n], b[:n]):
        if x[1] != y[1]:
            bar += 1
        elif abs(x[2] - y[2]) > 0.05:
            px += 1
        else:
            same += 1
    return same, bar, px


def run(write: bool = True) -> str:
    OM = _old()
    market = Market.load("2016", "2020")
    m1 = load_bars("1m", "2016", "2020")
    book = engine.bars_book(m1, "5m")
    fp = settings.fingerprint(settings.resolve({}))
    facts = hedging.MarketFacts.build(market, fp, book)
    comp = compile_idea(Idea.load(config.IDEAS_DIR / "idea-001" / "form.json"), market,
                        readings={"U1": "A", "U2": "A", "U3": "B"})
    by_id = {s.id: s for s in comp.signals}
    new = hedging.replay(comp.signals, facts)
    d = old_data(market, fp, facts, m1)
    S = old_index(facts, d)
    rel = pd.DatetimeIndex(facts.news[:, 1])
    OM._NEWS["t"] = np.sort(pd.DatetimeIndex(clock.server_to_utc(rel)).as_unit("s").asi8)
    cost_leg = OM.DEFAULTS["cost_rs_001"] / OM.USDINR

    lines = ["# L19a — the old manager against the new hedging tester, trade by trade (made by "
             "`python -m tests.compare_old_manager`)", "",
             f"idea-001 {comp.version_id}, 2016-2020, the new tester's {len(new.trades)} trades. Both sides: the same "
             f"bars, ruler, bias, liquidity, news and his {len(facts.setups)} setups (tests/compare_old_manager.py). "
             "Gold $ = before cost; positions = legs (₹40 each). The old engine's ₹ used a fixed 88 rupees a dollar; "
             "compare the gold dollars and the positions.", "",
             "| trade | new: positions · hedges · $ · end | old with T12's switches | old as shipped |",
             "|---|---|---|---|"]
    detail = ["", "## Where each trade first differs (new tester against the old manager with T12's switches)", ""]
    rows = []
    tot = {"new": [0.0, 0, 0], "T12": [0.0, 0, 0], "shipped": [0.0, 0, 0]}     # gold $, positions, trades (closed)
    for t in new.trades:
        s = by_id[t.signal_id]
        i_fill = bar_of(d, t.fill_ts)
        m = int(np.searchsorted(book.t, pd.Timestamp(t.fill_ts).as_unit("ns").value, "left"))
        entry = t.fill_px + (float(book.sp[m]) if s.direction == "buy" else 0.0)    # the price MT5 shows (a buy: the
        args = (d, S, i_fill, hedging.SIGN[s.direction], entry, float(s.stop), float(s.targets[0]))   # ask), as new
        res = {}
        for name, over in (("T12", T12), ("shipped", {})):
            P = dict(OM.DEFAULTS, **over)
            res[name] = OM.manage(*args, P, near_ath=False, end_bar=d.n - 1)

        def old_cell(r):
            if r is None:
                return "not run (no ruler at the fill)"
            gross = r["usd"] + r["legs"] * cost_leg
            end = "open at end" if r["open_at_end"] else f"{pd.Timestamp(d.open_ns[r['i_end']]):%Y-%m-%d} ({r['how']})"
            return f"{r['legs']} · {r['hedges']} · ${gross:,.2f} · {end}"
        closes = [x for x in t.legs if x.pnl is not None]
        gross_new = sum(x.pnl.gross_usd for x in closes)
        if t.status == "closed":
            tot["new"] = [tot["new"][0] + gross_new, tot["new"][1] + t.positions, tot["new"][2] + 1]
        for name in ("T12", "shipped"):
            r = res[name]
            if r is not None and not r["open_at_end"]:
                g = r["usd"] + r["legs"] * cost_leg
                tot[name] = [tot[name][0] + g, tot[name][1] + r["legs"], tot[name][2] + 1]
        end = "open at end" if t.status != "closed" else f"{t.exit_ts:%Y-%m-%d} ({t.exit_reason})"
        lines.append(f"| {t.signal_id.split('.')[-1]} | {t.positions} · {t.hedges} · ${gross_new:,.2f} · {end} | "
                     f"{old_cell(res['T12'])} | {old_cell(res['shipped'])} |")
        a, b = new_events(t, d), old_events(res["T12"]) if res["T12"] else []
        k = first_split(a, b)
        same, bar, px = near_misses(a, b, k)
        rows.append({"trade": t.signal_id, "split_at": k, "new": a, "old": b})
        detail.append(f"### {t.signal_id.split('.')[-1]} — "
                      f"{'the same all the way' if k is None else f'parts at event {k + 1} of {len(a)} (new)'}; "
                      f"before that: {same} the same, {bar} one bar apart, {px} a few cents to $1 apart")
        if k is not None:
            lo = max(0, k - 1)
            detail.append("")
            detail.append("| | new | old (T12 switches) |")
            detail.append("|---|---|---|")
            for j in range(lo, min(k + 3, max(len(a), len(b)))):
                na = a[j] if j < len(a) else None
                ob = b[j] if j < len(b) else None

                def cell(e):
                    if e is None:
                        return "—"
                    return (f"{e[0]} at {pd.Timestamp(d.open_ns[e[1]]):%Y-%m-%d %H:%M} {e[2]:.2f}: "
                            f"{e[3][:150]}").replace("|", "/")
                detail.append(f"| {j + 1} | {cell(na)} | {cell(ob)} |")
        detail.append("")
    lines.append("| **closed trades, all** | " + " | ".join(
        f"{tot[k][2]} trades · {tot[k][1]} positions · ${tot[k][0]:,.2f}" for k in ("new", "T12", "shipped")) + " |")
    text = "\n".join(lines + detail) + "\n"
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the old manager against the new hedging tester")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(run(write=not a.no_write))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
