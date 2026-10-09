"""Type 1's real order again, with the trades he would really have taken after each stop-out (L20-notes 3c point 7).

L19b made its list of hedged trades without stop-outs. When the plan's account reaches Rs 0 inside a trade, the broker
closes the trade there (THE STOP-OUT, 3c point 2), but the entries that came while that trade would still have been open
are not in L19b's list: the account only waits for the next trade in it. Here, in the REAL ORDER of version 1 (a
reshuffled order cannot be made again), the hedging tester is run again from the minute of each stop-out - L19b's own
set-up for Type 1: the same signals, his management (hedging.Rules.now()), his closers, the checks while an order
waits, the moving hedge - and the plan runs again on the new list, until no new stop-out comes. The replay from the
start must give L19b's recorded trades exactly, or nothing runs.

    .venv\\Scripts\\python -m machine.money.rerun              # a look: every round and the new headline, no file
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd

from machine import config
from machine.money import hedged as H
from machine.money import plan as MP

STOPPED_BY = "the broker closed it at Rs 0 (the plan's account ran out inside it)"
MAX_ROUNDS = 100                                   # a guard: each round's stop-out is later than the one before

Again = Callable[[pd.Timestamp], tuple[list[dict], dict[str, str]]]


class Type1Hedging:
    """L19b's hedging tester for Type 1, set up once exactly as machine.entries.run sets it up (the market of
    2016-2020, Type 1's signals, his closers and their checks, the checks while an order waits, the moving hedge, his
    management). replay(after) runs it on the signals known AFTER a moment: what the one-trade-at-a-time tester does
    when the trade it held ended at that moment (a signal known while a trade is on is skipped, M18)."""

    def __init__(self, st: H.Stream | None = None, start: str = H.FIRST, end: str = H.LAST, log=print):
        from machine import guards, settings
        from machine.entries import closers, forms
        from machine.entries import run as ER
        from machine.ideas.compile import Market, compile_idea
        from machine.tester import engine, hedging
        st = st or H.stream(H.MAIN)
        if st.entry != forms.TYPE1.id:
            raise ValueError("trades are made again only for Type 1, the report's headline (L20-notes 3c point 7)")
        t0 = time.time()
        market = Market.load(start, end, tfs=ER.TFS)
        self.book = engine.bars_book(market.bars("1m"), "5m")
        fp0 = settings.fingerprint(settings.resolve({}))
        comp = compile_idea(forms.TYPE1.idea(), market, readings={})
        if comp.version_id != st.version:
            raise RuntimeError(f"Type 1 compiled today is {comp.version_id}, not L19b's {st.version}: stop and find why")
        got, _ = closers.closer_entries()
        signals = {e.id: comp.signals if (e is forms.TYPE1 and not rd) else
                   compile_idea(e.idea(), market, readings=rd).signals for e, rd in got}
        closing = closers.closing_setups(market, self.book, signals=signals)
        self.facts = hedging.MarketFacts.build(market, fp0, self.book, setups=closing.setups)
        self.facts.setup_signals, self.facts.closers = closing.signals, closing.which
        guards.check_opened(forms.TYPE1.id, [self.book.first, self.book.last], version=comp.version_id)
        self.signals, self.version = comp.signals, comp.version_id
        self.rules = hedging.Rules.now()
        self.cancel = closers.fill_checks(forms.TYPE1, comp.signals, market, fp0, comp.signals)
        self.moves = ER._moves(comp)
        self.replays: list[str] = []                   # every replay asked for: "start" or the moment
        log(f"the hedging tester for Type 1 {self.version}: {len(self.signals)} signals, {len(self.facts.setups)} of "
            f"his setups that close a leg ({time.time() - t0:.0f} s)")

    def replay(self, after: pd.Timestamp | None = None):
        from machine.tester import hedging
        if after is None:
            sigs = self.signals
        else:
            cut = pd.Timestamp(after)
            sigs = [s for s in self.signals if pd.Timestamp(s.known_at) > cut]
        self.replays.append("start" if after is None else str(pd.Timestamp(after)))
        return hedging.replay(sigs, self.facts, "5m", self.rules, moves=self.moves, cancel_by=self.cancel)

    def stories(self, after: pd.Timestamp | None = None) -> tuple[list[dict], dict[str, str]]:
        """The trades of replay(after) in the stories file's shape, and how each one ended."""
        from machine.entries import run as ER
        rep = self.replay(after)
        return ([json.loads(json.dumps(ER.story(t, self.book), ensure_ascii=False)) for t in rep.trades],
                {t.signal_id: t.exit_reason or "" for t in rep.trades})


def same_trades(mine: list[dict], recorded: list[dict]) -> list[int]:
    """Where two lists of stories differ (an empty list: the same trades, leg by leg, to the paisa)."""
    if len(mine) != len(recorded):
        return list(range(max(len(mine), len(recorded))))
    return [i for i, (a, b) in enumerate(zip(mine, recorded)) if a != b]


# ================================================================================================================
# a trade the broker closed: it ends at that minute
# ================================================================================================================

def stop_bar(opened: pd.Timestamp, days: float) -> pd.Timestamp:
    """The minute a trade was stopped: its open + the days to its stop-out, on the minute."""
    return pd.Timestamp(opened) + pd.Timedelta(minutes=int(round(days * 1440)))


def cut(got: dict, signal: str, bar: pd.Timestamp) -> dict:
    """The trade `signal` ends at the minute `bar`, where the broker closed it: its money minute by minute up to that
    minute (its new lows), its lowest point, its money then, the positions it had opened by then (Rs 40 each). The
    stop-out minute must be one of its new lows (the account can only reach Rs 0 at a new low). Nothing else changes."""
    f = got["frame"].copy()
    hit = np.flatnonzero(f["signal"].to_numpy() == signal)
    if len(hit) != 1:
        raise KeyError(f"{signal} is not one trade of the plan")
    j = int(hit[0])
    opened, bar = pd.Timestamp(f.at[j, "open"]), pd.Timestamp(bar)
    d = (bar - opened) / pd.Timedelta(days=1)
    ins = got["inside"][signal]
    k, k0 = ins.t <= d + 1e-9, ins.t0 <= d + 1e-9
    if not k.any() or abs(float(ins.t[k][-1]) - d) > 1e-6:
        raise AssertionError(f"{signal}: {bar} is not one of its new lows - the broker cannot have closed it then")
    new = MP.Inside(ins.t[k], ins.v[k], ins.t0[k0], ins.v0[k0])
    story = next(s for s in got["stories"] if s["signal"] == signal)
    positions = sum(pd.Timestamp(x["opened"]) <= bar for x in story["legs"])
    cost = config.COST_INR_PER_POSITION * positions
    net = float(new.v[-1])
    for col, value in (("close", bar), ("closed_by", STOPPED_BY), ("net_inr", net), ("cost_inr", float(cost)),
                       ("gross_inr", net + cost), ("low_inr", net), ("low0_inr", float(new.v0[-1])),
                       ("low_when", bar), ("days", d), ("positions", positions)):
        f.at[j, col] = value
    return {**got, "frame": f, "inside": {**got["inside"], signal: new}}


# ================================================================================================================
# the real order, made again after every stop-out
# ================================================================================================================

def _plan_on(got: dict, plan: MP.Plan) -> tuple[MP.TradeSet, MP.Sim, pd.DataFrame]:
    ts = MP.TradeSet.build(got["frame"], last_month=H.LAST_MONTH, inside=got["inside"])
    one = MP.simulate(ts, MP.real_order(ts), plan, trace=True)
    tr = one.trace.copy()
    k = tr["trade"].to_numpy(int)
    tr["signal"] = ts.frame["signal"].to_numpy()[k]
    tr["open"] = ts.frame["open"].to_numpy()[k]
    tr["bar"] = [stop_bar(o, e - s) if x else pd.NaT
                 for o, s, e, x in zip(tr["open"], tr["opened_d"], tr["ended_d"], tr["stopped_out"])]
    return ts, one, tr


def made_again(st: H.Stream, book: MP.MinuteBook, needs: pd.DataFrame, first: list[dict], again: Again,
               plan: MP.Plan = MP.VERSION_1, ended: dict[str, str] | None = None, log=print) -> dict:
    """The real order of `plan` on `first` (the list made from the start); at the FIRST stop-out the list is cut there
    and again(that minute) gives the trades from then on; the plan runs again; and so on until no new stop-out.
    Each round's earlier stop-outs must come again at the same minutes (everything before them is the same)."""
    stories, ended = list(first), dict(ended or {})
    cuts: list[tuple[str, pd.Timestamp]] = []
    rounds: list[dict] = []
    heads: list[dict] = []                             # the plan's headline in each round (0 = the list as given)
    while True:
        got = H.stream_trades(st, book, needs, stories=stories, recorded={}, ended=ended)
        for sig, bar in cuts:
            got = cut(got, sig, bar)
        ts, one, tr = _plan_on(got, plan)
        heads.append(MP.headline(one))
        for sig, bar in cuts:
            row = tr[tr["signal"] == sig]
            if len(row) != 1 or not bool(row["stopped_out"].iloc[0]) or row["bar"].iloc[0] != bar:
                raise AssertionError(f"{sig}: the broker closed it at {bar} before, not now - the rounds disagree")
        done = {s for s, _ in cuts}
        new = tr[tr["stopped_out"] & ~tr["signal"].isin(done)].sort_values("bar")
        if not len(new):
            break
        sig, bar = str(new["signal"].iloc[0]), pd.Timestamp(new["bar"].iloc[0])
        if cuts and bar <= cuts[-1][1]:
            raise AssertionError(f"a new stop-out at {bar} is not after the last one ({cuts[-1][1]})")
        if len(cuts) >= MAX_ROUNDS:
            raise RuntimeError(f"more than {MAX_ROUNDS} stop-outs to make again: stop and find why")
        k = next(i for i, s in enumerate(stories) if s["signal"] == sig)
        more, more_ended = again(bar)
        if any(pd.Timestamp(x["opened"]) <= bar for s in more for x in s["legs"]):
            raise AssertionError(f"a trade made again after {bar} opens at or before it")
        dropped = stories[k + 1:]
        kept_after = {s["signal"] for s in dropped} & {s["signal"] for s in more}
        rounds.append({"round": len(rounds) + 1, "signal": sig, "opened": pd.Timestamp(new["open"].iloc[0]),
                       "stopped_at": bar, "account_at_open": float(new["account_at_open"].iloc[0]),
                       "lots": float(new["lots"].iloc[0]), "trades_left_out": len(dropped),
                       "trades_made_again": len(more), "the_same_trades": len(kept_after)})
        log(f"    round {len(rounds)}: {sig} closed by the broker at {bar}; {len(dropped)} later trades of the list "
            f"left out, {len(more)} made again from that minute ({len(kept_after)} of them the same)")
        stories = stories[:k + 1] + more
        ended.update(more_ended)
        cuts.append((sig, bar))
    new_signals = sorted({s["signal"] for s in stories} - {s["signal"] for s in first})
    return {"stories": stories, "ended": ended, "got": got, "ts": ts, "one": one, "trace": tr, "rounds": rounds,
            "cuts": cuts, "head": heads[-1], "first_head": heads[0], "heads": heads,
            "years": MP.year_rows(one, ts), "plan": plan, "new_signals": new_signals, "still": got["still"]}


def main(argv=None) -> int:
    import argparse
    import sys
    from machine.data import bars as BARS
    ap = argparse.ArgumentParser(prog="python -m machine.money.rerun", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    st = H.stream(H.MAIN)
    tester = Type1Hedging(st)
    first, ended = tester.stories()
    recorded, _ = H.load(st)
    bad = same_trades(first, recorded)
    if bad:
        raise RuntimeError(f"the hedging tester made today differs from L19b's recorded trades at {bad[:5]}")
    print(f"the replay from the start = L19b's {len(recorded)} recorded trades, leg by leg")
    book = MP.MinuteBook.build(BARS.load_bars("1m", H.FIRST, H.LAST))
    needs = MP.weekly_need(BARS.load_bars("5m", H.FIRST, H.LAST))
    res = made_again(st, book, needs, first, tester.stories, ended=ended)
    h = res["head"]
    print(f"made again: {len(res['trace'])} trades taken ({len(res['ts'])} in the list), {len(res['rounds'])} rounds; "
          f"your Rs 200,000 -> "
          f"{h['your_money_end']:,.0f} ({h['result']:+,.0f}); {h['blowups']} blow-ups ({h['stopouts']} stop-outs), "
          f"{h['doubles']} doubles, taken out {h['out']:,.0f}, added back {h['added']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
