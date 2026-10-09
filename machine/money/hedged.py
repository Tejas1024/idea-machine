"""His money plan on his HEDGED trades (L20, the second run - plan/work/L20-notes.md section 3).

    .venv\\Scripts\\python -m machine.money.hedged                  # L19b's hedged trades -> money-plan-hedged.md
    .venv\\Scripts\\python -m machine.money.hedged --runs 50 --no-write

The trades: L19b's hedging box - each of his entry types with his hedging (every leg 0.01 lot, Rs 40 a position),
2016-2020, one STREAM per L19b version (STREAMS). Each is read from plan/work/L19b/<entry>/<version>-hedged-stories.jsonl
and must equal that version's trial file to the paisa. Each trade brings its money minute by minute
(plan.trade_minutes, the 2e measure): where the account would reach Rs 0 inside it - THE STOP-OUT (L20-notes 3c).
Type 1 (the main stream) runs version 1, the 1,000 reshuffles and every one-change row (plan.HEDGED_ROWS); every other
stream runs version 1 (the real order and the 1,000 reshuffles).
Writes (write=True):
  plan/work/money-plan-hedged.md             the report, easy English (every sentence passes check_words)
  plan/work/money-plan-hedged/account.jpg    Type 1, the real order: the account's money hour by hour
  plan/work/money-plan-hedged/orders.jpg     Type 1: where your money ended in each of the 1,000 orders
  plan/work/money-plan-hedged/*.csv          Type 1's trades and events, the lots, every stream, the dips, the Mondays
  plan/work/trials/<id>.json                 ONE trial file per stream (kind "money-plan")
The words BLOW-UP, REFILL, a double, THE STOP-OUT: plan/work/L20-notes.md sections 1 and 3 (fixed before the run).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config
from machine.money import plan as MP

L19B = config.WORK_DIR / "L19b"
TRIALS = config.WORK_DIR / "trials"
FIRST, LAST, LAST_MONTH = "2016", "2020", "2020-12"     # L19b's window; 2021-2023 not opened (already seen)
KIND = "money-plan"


@dataclass(frozen=True)
class Stream:
    """One L19b version's hedged trades (L20-notes 3a)."""
    key: str
    entry: str
    version: str
    trial: str                        # L19b's trial file (its id; the file ends with -<entry>-<version>.json)
    words: str                        # the entry type in easy English
    reading: str = ""                 # the reading of his open word, in words ("" = as written)

    @property
    def stories(self) -> Path:
        return L19B / self.entry / f"{self.version}-hedged-stories.jsonl"

    @property
    def table(self) -> Path:
        return L19B / self.entry / f"{self.version}-hedged-trades.csv"

    @property
    def trial_file(self) -> Path:
        return TRIALS / f"{self.trial}-{self.entry}-{self.version}.json"

    @property
    def name(self) -> str:
        return self.words + (f" ({self.reading})" if self.reading else "")


STREAMS: tuple[Stream, ...] = (
    Stream("type1", "entry-type1", "v-917f35901a", "20260930-142057-272805-9079be",
           "Type 1: liquidity taken, a break, the first gap"),
    Stream("zone-break", "entry-zone-break", "v-b52875432e", "20260930-142059-217138-d21bfc",
           "the zone break: a strong zone broken, enter at the retest"),
    Stream("zone-entry", "entry-zone-entry", "v-e926ea367b", "20260930-142100-334738-8f9047",
           "the zone entry: liquidity just past a strong zone taken, enter at the zone"),
    Stream("p1-a", "entry-triangle-false-break", "v-9700e54ac4", "20260930-142059-534235-42da64",
           "P1: the triangle false break", "reading A: 'big move' = the day before moved at least half a day's range"),
    Stream("p1-b", "entry-triangle-false-break", "v-6ecbfb3b3b", "20260930-142059-782333-b80561",
           "P1: the triangle false break", "reading B: 'big move' = at least one day's range"),
    Stream("p1-c", "entry-triangle-false-break", "v-776081ac09", "20260930-142100-065036-338e4c",
           "P1: the triangle false break", "reading C: 'big move' = at least one and a half days' range"),
    Stream("touch-a", "entry-trendline-touch", "v-b60a4013cd", "20260930-142058-669216-b25396",
           "Type 2: the 3rd or 4th touch of a steep trendline",
           "reading A: 'steep' = the line travels at least 0.15 of a day's range in a day"),
    Stream("touch-b", "entry-trendline-touch", "v-a9718f59a8", "20260930-142058-944688-8433ad",
           "Type 2: the 3rd or 4th touch of a steep trendline",
           "reading B: 'steep' = at least 0.53 of a day's range in a day"),
)
MAIN = "type1"
# the second gap (#124), both readings: the same trades as Type 1 to the rupee (L19b-notes point 4) - not a stream
SAME_AS_TYPE1 = (("entry-second-gap", "v-543ca92f7c", "20260930-142057-827300-6070ba"),
                 ("entry-second-gap", "v-eb3c267b19", "20260930-142058-290256-033d0b"))


def stream(key: str) -> Stream:
    for s in STREAMS:
        if s.key == key:
            return s
    raise KeyError(key)


# ================================================================================================================
# the trades: read, checked, and each one minute by minute
# ================================================================================================================

def read_stories(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load(st: Stream) -> tuple[list[dict], dict]:
    """The stream's hedged trades as L19b wrote them - checked against its trial file (the closed trades and their
    rupees, to the paisa) or the run stops."""
    stories = read_stories(st.stories)
    trial = json.loads(st.trial_file.read_text(encoding="utf-8"))
    if (trial.get("idea_id"), trial.get("version_id")) != (st.entry, st.version):
        raise RuntimeError(f"{st.trial_file.name} is not {st.entry} {st.version}")
    rec = trial["result"]["his_hedging"]
    closed = [s for s in stories if s["status"] == "closed"]
    total = round(sum(s["net_inr"] for s in closed), 2)
    want = (int(rec["trades"]), round(float(rec["net_inr_total"]), 2))
    if (len(closed), total) != want:
        raise RuntimeError(f"{st.entry} {st.version}: the stories hold {len(closed)} closed trades, {total} - its "
                           f"trial file {st.trial} says {want[0]}, {want[1]}: stop and find why")
    return stories, {"trial": st.trial, "trades": want[0], "net_inr": want[1],
                     "open_at_end": len(stories) - len(closed)}


def stream_trades(st: Stream, book: MP.MinuteBook, needs: pd.DataFrame, stories: list[dict] | None = None,
                  recorded: dict | None = None, ended: dict[str, str] | None = None) -> dict:
    """The stream as the plan reads it (L20-notes 3a, 3b): one row per CLOSED trade with its Monday number, its money
    minute by minute (Inside), and every trade still open at the end on its own (lesson B6: never in the plan).
    stories/recorded: given by the tests (made-up trades) or by rerun.made_again (trades made again); else read and
    checked by load(). ended: {signal: how it ended} - else L19b's table of the stream."""
    if stories is None:
        stories, recorded = load(st)
    if ended is None:
        ended = {}
        if st.table.exists():
            t = pd.read_csv(st.table)
            ended = dict(zip(t["signal"], t["ended_by"].fillna("")))
    rows, inside, still, hours = [], {}, [], {}
    for s in stories:
        got = MP.trade_minutes(s, book)
        a0, b0 = got["a0"], got["b0"]
        t_ns = book.t[a0:b0]
        inr, inr0 = got["inr"], got["inr0"]
        j = int(np.argmin(inr))
        hours[s["signal"]] = _hourly(t_ns, inr)
        common = {"signal": s["signal"], "direction": s["legs"][0]["direction"], "open": book.idx[a0],
                  "positions": len(s["legs"]), "most_open_at_once": int(got["n_open"].max()),
                  "low_inr": float(inr[j]), "low_when": book.idx[a0 + j]}
        if got["still_open"]:
            still.append({**common, "last_minute": book.idx[b0 - 1], "inr_at_the_end": float(inr[-1]),
                          "days": (t_ns[-1] - t_ns[0]) / MP.DAY_NS})
            continue
        cost = config.COST_INR_PER_POSITION * len(s["legs"])
        rows.append({**common, "close": book.idx[b0 - 1], "closed_by": ended.get(s["signal"], ""),
                     "gross_inr": float(s["net_inr"]) + cost, "cost_inr": cost, "net_inr": float(s["net_inr"]),
                     "low0_inr": float(inr0.min()), "days": (t_ns[-1] - t_ns[0]) / MP.DAY_NS})
        inside[s["signal"]] = MP.Inside.from_minutes((t_ns - t_ns[0]) / MP.DAY_NS, inr, inr0)
    frame = pd.DataFrame(rows)
    got_needs, dropped = MP.attach_need(frame, needs) if len(frame) else (frame, 0)
    still_f = pd.DataFrame(still)
    if len(still_f):
        still_f, _ = MP.attach_need(still_f, needs)
    return {"stream": st, "frame": got_needs, "inside": inside, "still": still_f, "dropped": dropped,
            "dropped_signals": sorted(set(frame["signal"]) - set(got_needs["signal"])) if len(frame) else [],
            "dropped_frame": frame[~frame["signal"].isin(got_needs["signal"])] if len(frame) else frame,
            "recorded": recorded or {}, "hours": hours, "stories": stories, "ended": ended}


def _hourly(t_ns: np.ndarray, inr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The trade's lowest money in each clock hour (for the picture): (hour start ns, lowest)."""
    h = t_ns // (3_600 * 10**9)
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    return h[starts] * 3_600 * 10**9, np.minimum.reduceat(inr, starts)


# ================================================================================================================
# the runs
# ================================================================================================================

def plans_for(st: Stream) -> tuple[MP.Plan, ...]:
    """Type 1: version 1 + rows 1-10; every other stream: version 1 (L20-notes 3a, 3h)."""
    return (MP.VERSION_1, *MP.HEDGED_ROWS) if st.key == MAIN else (MP.VERSION_1,)


def run_stream(got: dict, n_runs: int = MP.RESHUFFLES, log=print) -> dict:
    """One stream through its plans: the real order (traced) and the reshuffles of the 60 months."""
    st = got["stream"]
    ts = MP.TradeSet.build(got["frame"], last_month=LAST_MONTH, inside=got["inside"])
    real, shuf = MP.real_order(ts), MP.reshuffled(ts, n_runs, MP.SEED)
    rows = []
    for plan in plans_for(st):
        one = MP.simulate(ts, real, plan, trace=True)
        many = MP.simulate(ts, shuf, plan)
        rows.append({"plan": plan, "one": one, "many": many, "head": MP.headline(one), "spread": MP.spread(many)})
    h, a = rows[0]["head"], rows[0]["spread"]["all_years"]
    log(f"  {st.key}: {len(ts)} trades; real order {h['result']:+,.0f} ({h['stopouts']} stop-outs, {h['blowups']} "
        f"blow-ups); the middle of {n_runs} orders {a['result_mid']:+,.0f}")
    return {**got, "ts": ts, "rows": rows, "real": real, "shuf": shuf, "n_runs": n_runs}


def still_open_line(res: dict) -> list[dict]:
    """Each trade still open at the end of 2020, at the lots the plan would have given it from the account the real
    order ended with (version 1): its money on the last minute, and whether the account reached Rs 0 inside it."""
    out = []
    one = res["rows"][0]["one"]
    acc = float(one.account[0])
    for r in res["still"].itertuples():
        steps = np.floor(acc / r.need_inr + 1e-9) if acc >= r.need_inr - 1e-9 else 0.0
        low_acc = acc + steps * r.low_inr
        out.append({"signal": r.signal, "opened": r.open, "days": r.days, "positions": r.positions,
                    "inr_at_the_end_001": r.inr_at_the_end, "low_001": r.low_inr, "need_inr": r.need_inr,
                    "account": acc, "lots": steps * config.LOT_STEP, "inr_at_the_end_at_lots": steps * r.inr_at_the_end,
                    "account_lowest_inside": low_acc, "would_stop_out": bool(steps and low_acc <= 1e-9)})
    return out


# ================================================================================================================
# the whole run: every number of the report
# ================================================================================================================

OUT_MD = config.WORK_DIR / "money-plan-hedged.md"
OUT_DIR = config.WORK_DIR / "money-plan-hedged"
NOTES = "plan/work/L20-notes.md"
PLAIN_REPORT = "plan/work/money-plan.md"
PLAIN_MONDAYS = config.WORK_DIR / "money-plan" / "mondays.csv"
FEW = 5                                            # a stream with fewer trades says nothing (said next to it)


def run(n_runs: int = MP.RESHUFFLES, write: bool = True, log=print, streams=STREAMS) -> dict:
    """L19b's hedged trades through the plan; write=True also writes the report, its pictures and tables, and one
    trial file per stream."""
    from machine.data import bars as BARS
    from machine.money import report as R
    t0 = time.time()
    m1 = BARS.load_bars("1m", FIRST, LAST)
    needs = MP.weekly_need(BARS.load_bars("5m", FIRST, LAST))
    same = same_mondays(needs)
    log(f"1-minute bars {m1.index[0]} .. {m1.index[-1]}; {len(needs)} Monday numbers, {same} of them the same as the "
        f"plain run's ({time.time() - t0:.0f} s)")
    data = {"prices": R._rel(config.GOLD_M1), "gold_m1_content_sha256": BARS.m1_fingerprint()}
    again, tester = None, None
    if any(s.key == MAIN for s in streams):              # L20-notes 3c point 7: Type 1's real order made again
        from machine.money import rerun
        tester = rerun.Type1Hedging(stream(MAIN), log=log)
        fresh, _ = tester.stories()
        bad = rerun.same_trades(fresh, load(stream(MAIN))[0])
        if bad:
            raise RuntimeError(f"the hedging tester set up today is not L19b's: its trades differ at {bad[:5]} - stop "
                               "and find why")
        log(f"  the hedging tester set up again: the replay from the start = L19b's {len(fresh)} recorded Type 1 "
            f"trades, leg by leg ({time.time() - t0:.0f} s)")
        again = tester.stories
    out = build(streams, MP.MinuteBook.build(m1), needs, n_runs, log, data=data, again=again)
    out["same_mondays"] = same
    if tester is not None:
        out["again_replays"] = list(tester.replays)
    if write:
        publish(out, log=log)
    return out


def same_mondays(needs: pd.DataFrame) -> int:
    """The Monday numbers of 2016-2020 must be the plain run's (plan/work/money-plan/mondays.csv): how many match
    (every one, or the run stops)."""
    if not PLAIN_MONDAYS.exists():
        return 0
    old = pd.read_csv(PLAIN_MONDAYS, parse_dates=["day"])
    old = old[old["day"] < pd.Timestamp(f"{int(LAST) + 1}-01-01")].reset_index(drop=True)
    new = needs.reset_index(drop=True)
    if len(old) != len(new) or not (old["day"].to_numpy() == new["day"].to_numpy()).all() or \
            not np.allclose(old["need_inr"].to_numpy(), new["need_inr"].to_numpy(), atol=1e-3):
        raise RuntimeError("the Monday numbers of 2016-2020 are not the plain run's (money-plan/mondays.csv)")
    return len(new)


def build(streams, book: MP.MinuteBook, needs: pd.DataFrame, n_runs: int = MP.RESHUFFLES, log=print,
          data: dict | None = None, stories: dict | None = None, again=None) -> dict:
    """Every number of the report (no file is written). stories: {stream key: (stories, recorded)} - the tests'
    made-up trades; else each stream is read and checked (load). again(minute) -> (stories, how each ended): the
    hedging tester from that minute (rerun.Type1Hedging.stories, or the tests' own) - Type 1's real order is made
    again after each stop-out (L20-notes 3c point 7); None: not made again (the report says so)."""
    t0 = time.time()
    res = []
    for st in streams:
        s, rec = (stories or {}).get(st.key, (None, None))
        got = stream_trades(st, book, needs, s, rec)
        if not len(got["frame"]):
            raise RuntimeError(f"{st.key}: no trade left in the plan")
        res.append(run_stream(got, n_runs, log))
    main = next(r for r in res if r["stream"].key == MAIN)
    out = {"data": data or {"prices": "given to build()"}, "needs": needs, "n_runs": n_runs, "streams": res,
           "main": main, "again": None}
    if again is not None:
        from machine.money import rerun
        log("  Type 1, the real order made again after each stop-out (L20-notes 3c point 7):")
        ag = rerun.made_again(main["stream"], book, needs, main["stories"], again, MP.VERSION_1,
                              ended=main["ended"], log=log)
        if ag["rounds"] and abs(ag["first_head"]["result"] - main["rows"][0]["head"]["result"]) > 0.005:
            raise AssertionError("made again: its first round is not the plan's real order")
        out["again"] = ag
    out["facts"] = facts(main, needs)
    out["years"] = MP.year_rows(main["rows"][0]["one"], main["ts"])
    out["still"] = {r["stream"].key: still_open_line(r) for r in res}
    worst = main["rows"][0]["spread"]["worst"]["run"]
    out["worst"] = MP.simulate(main["ts"], main["shuf"].one(worst), MP.VERSION_1, trace=True)
    out["dips"] = dips_table(res)
    out["seconds"] = round(time.time() - t0)
    out["text"] = summary_text(out)
    return out


def trace_table(res: dict, row: int = 0) -> pd.DataFrame:
    """The real order of one plan (row 0 = version 1), one line per trade taken, with the trade's own columns.
    positions_in_the_plan: a trade the broker closed counts only the positions it had opened by then."""
    from machine.money.rerun import stop_bar
    f = res["ts"].frame
    tr = res["rows"][row]["one"].trace.copy()
    for col in ("signal", "direction", "open", "close", "closed_by", "monday", "net_inr", "gross_inr", "cost_inr",
                "positions", "low_inr", "days"):
        tr[col] = f[col].to_numpy()[tr["trade"].to_numpy(int)] if col in f.columns else None
    legs = {s["signal"]: s["legs"] for s in res.get("stories") or []}
    tr["positions_in_the_plan"] = [
        sum(pd.Timestamp(x["opened"]) <= stop_bar(o, e - s) for x in legs[g]) if stopped and g in legs else p
        for g, o, s, e, stopped, p in zip(tr["signal"], tr["open"], tr["opened_d"], tr["ended_d"], tr["stopped_out"],
                                          tr["positions"])]
    return tr


def facts(res: dict, needs: pd.DataFrame) -> dict:
    ts, rows = res["ts"], res["rows"]
    one, many = rows[0]["one"], rows[0]["many"]
    f = ts.frame
    tr = trace_table(res)
    steps = np.round(tr["lots"].to_numpy() / config.LOT_STEP)
    stop = tr["stopped_out"].to_numpy(bool)
    would = steps * tr["net_inr"].to_numpy(float)
    best = [{"open": pd.Timestamp(r.open), "lots": float(r.lots), "inr": float(r.inr), "at_001": float(r.net_inr),
             "stopped": bool(r.stopped_out)} for r in tr.sort_values("inr", ascending=False).head(3).itertuples()]
    year_of_open = pd.DatetimeIndex(f["open"]).year
    monday_year = pd.DatetimeIndex(needs["day"]).year
    return {
        "trades": len(ts), "dropped": res["dropped"], "dropped_frame": res["dropped_frame"],
        "positions": int(f["positions"].sum()), "won": int((f["net_inr"] > 0).sum()),
        "flat_total": float(f["net_inr"].sum()), "flat_gross": float(f["gross_inr"].sum()),
        "flat_cost": float(f["cost_inr"].sum()),
        "flat_by_year": {int(y): float(v) for y, v in f.groupby(year_of_open)["net_inr"].sum().items()},
        "days_mid": float(f["days"].median()), "days_max": float(f["days"].max()),
        "stopped": int(stop.sum()), "stopped_inr": float(tr["inr"].to_numpy()[stop].sum()),
        "stopped_would": float(would[stop].sum()),
        "own_end": int((~stop).sum()), "own_end_inr": float(tr["inr"].to_numpy()[~stop].sum()),
        "under": int(one.under.sum()), "cut_days": float(one.cut_days[0]),
        "lot_min": float(tr["lots"].min()), "lot_max": float(tr["lots"].max()),
        "best3": best, "best3_sum": float(sum(b["inr"] for b in best)),
        "orders_better": int((many.result > one.result[0] + 1e-6).sum()),
        "orders_worse": int((many.result < one.result[0] - 1e-6).sum()),
        "orders_best": float(many.result.max()), "orders_made_money": int((many.result > 0).sum()),
        "refills": [float(e["money_in_out"]) for e in one.events if e["what"] == "blow-up"],
        "doubles_in_his_days": int((many.dbl_days <= HIS_DAYS).sum()), "doubles_all": int(len(many.dbl_days)),
        "runs_with_double_by_year": [int((many.doubles[:, y] > 0).sum()) for y in range(many.paths.n_years)],
        "need_min": float(needs["need_inr"].min()), "need_max": float(needs["need_inr"].max()),
        "need_by_year": {int(y): (float(g.min()), float(g.max())) for y, g in needs.groupby(monday_year)["need_inr"]},
    }


def dips_table(results: list[dict]) -> pd.DataFrame:
    """Every stream's trades (closed, and still open at the end): the lowest point inside at 0.01 lot a position,
    against the money needed for 0.01 lot the week it opened - and what a FULL account (Rs 20,000) would be at that
    point at the plan's lots (the 2e measure on L19b's trades)."""
    rows = []
    for r in results:
        key = r["stream"].key
        for x in r["ts"].frame.itertuples():
            rows.append({"stream": key, "signal": x.signal, "opened": x.open, "ended": x.close, "still_open": False,
                         "days": x.days, "positions": x.positions, "net_inr": x.net_inr, "low_inr": x.low_inr,
                         "low_when": x.low_when, "need_inr": x.need_inr})
        for x in (r["still"].itertuples() if len(r["still"]) else ()):
            rows.append({"stream": key, "signal": x.signal, "opened": x.open, "ended": None, "still_open": True,
                         "days": x.days, "positions": x.positions, "net_inr": None, "low_inr": x.low_inr,
                         "low_when": x.low_when, "need_inr": x.need_inr})
    d = pd.DataFrame(rows)
    size = MP.VERSION_1.account_size
    d["low_in_needs"] = -d["low_inr"] / d["need_inr"]
    d["full_account_steps"] = np.floor(size / d["need_inr"] + 1e-9)
    d["full_account_at_low"] = size + d["full_account_steps"] * d["low_inr"]
    return d


def summary_text(out: dict) -> str:
    from machine.costs import inr
    main, fx = out["main"], out["facts"]
    h, a = main["rows"][0]["head"], main["rows"][0]["spread"]["all_years"]
    lines = [f"MONEY PLAN ON HIS HEDGED TRADES (L20, second run) - Type 1 {main['stream'].version}, {fx['trades']} "
             f"trades 2016-2020, {fx['positions']} positions, Rs 40 a position",
             f"  real order : your Rs 200,000 -> {inr(h['your_money_end'])} ({h['result']:+,.0f}); {h['doubles']} doubles; "
             f"{h['blowups']} blow-ups ({h['stopouts']} stop-outs: the broker closed a trade at Rs 0); taken out "
             f"{inr(h['out'])}, added back {inr(h['added'])}",
             f"  {out['n_runs']} orders: the middle {a['result_mid']:+,.0f}, the bad 10% {a['result_bad']:+,.0f}; "
             f"stop-outs {a['stopouts_mid']:.0f} (bad 10% {a['stopouts_bad']:.0f})",
             f"  0.01 lot a position, no plan: {fx['flat_total']:+,.0f}"]
    ag = out.get("again")
    if ag and ag["rounds"]:
        g = ag["head"]
        lines.append(f"  found again after each stop-out ({len(ag['rounds'])} rounds): {len(ag['trace'])} trades taken "
                     f"({len(ag['ts'])} in the list); your Rs 200,000 -> {inr(g['your_money_end'])} "
                     f"({g['result']:+,.0f}); {g['blowups']} blow-ups "
                     f"({g['stopouts']} stop-outs), {g['doubles']} doubles; taken out {inr(g['out'])}, added back "
                     f"{inr(g['added'])}")
    for r in out["streams"]:
        if r is main:
            continue
        hh, aa = r["rows"][0]["head"], r["rows"][0]["spread"]["all_years"]
        lines.append(f"  {r['stream'].key}: {len(r['ts'])} trades, real {hh['result']:+,.0f} ({hh['stopouts']} "
                     f"stop-outs), middle {aa['result_mid']:+,.0f}")
    return "\n".join(lines)


# ================================================================================================================
# the files: pictures, tables, trial files, the report
# ================================================================================================================

HIS_DAYS = 8                                       # his "doubles in ~7-8 days" (strategy.md line 67, section 12)
HIS_BLOWUPS = (2, 4)                               # capital-plan.md: "2-4 full blow-ups a year accepted"


def publish(out: dict, *, md_path: Path = OUT_MD, folder: Path = OUT_DIR, record: bool = True, log=print) -> None:
    """The pictures, the tables, the trial files (record=True) and the report. The report's words are checked FIRST:
    a report that cannot be written leaves no picture, no table and no trial file behind."""
    from machine.money import report as R
    _checked(report_md(out))
    folder.mkdir(parents=True, exist_ok=True)
    out["pictures"] = [account_picture(out, folder / "account.jpg"), orders_picture(out, folder / "orders.jpg")]
    write_tables(out, folder)
    if record:
        out["trials"] = {r["stream"].key: _record(out, r).name for r in out["streams"]}
    md = _checked(report_md(out))
    tmp = md_path.with_name(md_path.name + ".tmp")
    tmp.write_text(md, encoding="utf-8", newline="\n")
    tmp.replace(md_path)
    log(f"wrote {R._rel(md_path)}, {R._rel(folder)}/" + (f" and {len(out['trials'])} trial files" if record else ""))


def _checked(md: str) -> str:
    from machine.money.report import MoneyReportError
    from machine.report.build import check_words
    problems = check_words(md)
    if problems:
        raise MoneyReportError("the report's words: " + "; ".join(problems[:6]))
    return md


def account_path(out: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Type 1's account in the real order (version 1): inside each trade its lowest money in each hour (the account
    when the trade opened + the lots x the trade's money that hour, until the trade ended), after each trade, and
    after every withdrawal and refill."""
    main = out["main"]
    one = main["rows"][0]["one"]
    tr = trace_table(main)
    pts = [(float(tr["opened_d"].iloc[0]), one.plan.account_size, 0)]
    for r in tr.itertuples():
        h_ns, low = main["hours"][r.signal]
        opened = MP.EPOCH.value + r.opened_d * MP.DAY_NS
        ended = MP.EPOCH.value + r.ended_d * MP.DAY_NS
        steps = round(r.lots / config.LOT_STEP)
        for t, v in zip(h_ns, low):
            if t <= ended:                                    # an hour that started before the open counts from it
                pts.append(((max(t, opened) - MP.EPOCH.value) / MP.DAY_NS,
                            max(r.account_at_open + steps * v, 0.0), 1))
        pts.append((r.ended_d, r.account_after_trade, 2))
    pts += [(e["t"], e["account_after"], 3) for e in one.events if e["what"] in ("blow-up", "a double")]
    pts.sort(key=lambda p: (p[0], p[2]))
    return pd.DatetimeIndex([MP.when(p[0]) for p in pts]), np.array([p[1] for p in pts])


def account_picture(out: dict, path: Path) -> dict:
    """ONE chart: the money in the account hour by hour, the money needed for 0.01 lot, Rs 0, and each stop-out,
    blow-up and double - every thing it names is drawn."""
    import matplotlib
    import matplotlib.pyplot as plt
    from machine.charts import draw as D
    from machine.costs import inr
    from machine.money import report as R
    plt.rcParams["text.parse_math"] = False
    main = out["main"]
    one, h = main["rows"][0]["one"], main["rows"][0]["head"]
    tr = trace_table(main)
    when, money = account_path(out)
    size, double = one.plan.account_size, one.plan.double_at * one.plan.account_size
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.9), dpi=D.DPI)
    texts: list[str] = []
    ax.plot(when, money, color=D.INK, linewidth=1.0, zorder=3, drawstyle="steps-post")
    need_t = pd.DatetimeIndex([MP.when(x) for x in tr["opened_d"]])
    ax.step(need_t, tr["need_inr"], where="post", color=D.RED, linewidth=1.1, zorder=2)
    ax.axhline(0, color=D.INK, linewidth=0.8, zorder=1)
    ax.axhline(size, color=D.MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(double, color=D.GREEN, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    so = [e for e in one.events if e["what"] == "stop-out"]
    bu = [e for e in one.events if e["what"] == "blow-up" and "broker" not in e["why"]]
    dv = [e for e in one.events if e["what"] == "a double"]
    ax.scatter([MP.when(e["t"]) for e in so], [0.0] * len(so), marker="X", s=120, color=D.RED, zorder=6,
               edgecolors="white", linewidths=0.8)
    ax.scatter([MP.when(e["t"]) for e in bu], [e["account_before"] for e in bu], marker="v", s=95, color=D.RED,
               zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter([MP.when(e["t"]) for e in dv], [e["account_before"] for e in dv], marker="^", s=105, color=D.GREEN,
               zorder=5, edgecolors="white", linewidths=0.8)
    top = max(float(money.max()), double) * 1.30
    ax.set_ylim(-0.04 * top, top)
    x0 = pd.Timestamp(f"{FIRST}-01-01") - pd.Timedelta(days=30)
    x1 = pd.Timestamp(f"{int(LAST) + 1}-01-01") + pd.Timedelta(days=30)
    ax.set_xlim(x0, x1)

    def put(y, s, colour):
        texts.append(s.strip())
        ax.text(x1, y, s, ha="left", va="center", fontsize=D.F_AXIS, color=colour, clip_on=False)

    put(size, f" {inr(size)}\n the full account", D.MUTED)
    put(double, f" {inr(double)}\n a double", D.GREEN)
    put(0, " ₹0: the broker\n closes the trade", D.INK)
    put(float(tr["need_inr"].iloc[-1]) + 0.05 * top, " money needed\n for 0.01 lot", D.RED)
    key = [(D.INK, "line", "the money in the account (its lowest point in each hour)", 1),
           (D.RED, "X", f"the broker closed a trade at ₹0 ({h['stopouts']}): the whole account lost", len(so)),
           (D.RED, "v", f"a blow-up after a trade ({len(bu)}): below the red line, refilled from the reserve", len(bu)),
           (D.GREEN, "^", f"a double ({h['doubles']}): the profit taken out, {inr(h['out'])} in all", len(dv))]
    key = [(colour, mark, words) for colour, mark, words, drawn in key if drawn]   # #113: only what is drawn
    handles = []
    for colour, mark, words in key:
        texts.append(words)
        if mark == "line":
            handles.append(plt.Line2D([], [], color=colour, linewidth=1.6, label=words))
        else:
            handles.append(plt.Line2D([], [], color=colour, marker=mark, linestyle="", markersize=9, label=words))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=D.F_AXIS + 0.5,
              labelcolor=D.INK, handlelength=1.6, borderaxespad=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="both", labelsize=D.F_AXIS, colors=D.MUTED)
    years = range(int(FIRST), int(LAST) + 1)
    ax.set_xticks([pd.Timestamp(year=y, month=1, day=1) for y in years], [str(y) for y in years])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=D.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    title = f"The account at the broker, {FIRST}-{LAST}, the real order"
    sub = "Type 1 with your hedging · your money plan as written · ₹40 a position per 0.01 lot"
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=D.F_HEAD, color=D.INK, fontweight="bold")
    fig.text(0.06, 0.905, sub, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    texts += [title, sub]
    fig.subplots_adjust(left=0.085, right=0.84, top=0.85, bottom=0.08)
    return R._save(fig, path, texts)


def orders_picture(out: dict, path: Path) -> dict:
    """ONE chart: where your money ended in each reshuffled order (Type 1, version 1), the real order among them."""
    import matplotlib
    import matplotlib.pyplot as plt
    from machine.charts import draw as D
    from machine.costs import inr
    from machine.money import report as R
    plt.rcParams["text.parse_math"] = False
    main, fx = out["main"], out["facts"]
    one, many = main["rows"][0]["one"], main["rows"][0]["many"]
    end = many.your_money_end
    real_end = float(one.your_money_end[0])
    start = one.plan.your_money
    flat_end = start + fx["flat_total"]
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.8), dpi=D.DPI)
    texts: list[str] = []
    lo = min(float(end.min()), real_end, flat_end) - 8_000
    hi = max(start, float(end.max()), real_end, flat_end) + 8_000
    bins = np.arange(np.floor(lo / 5_000) * 5_000, np.ceil(hi / 5_000) * 5_000 + 1, 5_000)
    counts, _, _ = ax.hist(end, bins=bins, color="#b7c0ca", edgecolor="white", linewidth=0.6, zorder=2)
    top = float(counts.max()) * 1.6
    ax.set_ylim(0, top)
    ax.set_xlim(bins[0], bins[-1])

    def mark(x, colour, words, level, ha):
        ax.axvline(x, color=colour, linewidth=1.8, zorder=4, ymax=level)
        texts.append(words)
        ax.text(x, top * (level + 0.012), words, ha=ha, va="bottom", fontsize=D.F_AXIS + 0.5, color=colour,
                fontweight="bold", zorder=5)

    mid = float(np.percentile(end, 50))
    mark(mid, D.RED, f"the middle order\n{inr(mid)}", 0.60, "center")
    mark(real_end, D.INK, f"the real order\n{inr(real_end)}", 0.74, "center")
    mark(flat_end, D.BLUE, f"the same trades at 0.01 lot\na position, no plan: {inr(flat_end)}", 0.86, "center")
    mark(start, D.GREEN, f"the start\n{inr(start)}", 0.60, "right" if start > mid else "left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="both", labelsize=D.F_AXIS, colors=D.MUTED)
    ylab, xlab = "how many of the orders", f"your money at the end of the {many.paths.n_years} years"
    ax.set_ylabel(ylab, fontsize=D.F_AXIS + 0.5, color=D.MUTED)
    ax.set_xlabel(xlab, fontsize=D.F_AXIS + 0.5, color=D.MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=D.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    title = f"The same months in {many.paths.P:,} other orders"
    sub = (f"grey bars = the {many.paths.P:,} orders · {fx['orders_worse']:,} of them ended with less than the real "
           f"order\nType 1 with your hedging · your money plan as written · ₹40 a position per 0.01 lot")
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=D.F_HEAD, color=D.INK, fontweight="bold")
    fig.text(0.06, 0.925, sub, ha="left", va="top", fontsize=D.F_AXIS, color=D.MUTED, linespacing=1.5)
    texts += [title, sub, xlab, ylab]
    fig.subplots_adjust(left=0.085, right=0.975, top=0.82, bottom=0.115)
    return R._save(fig, path, texts)


def write_tables(out: dict, folder: Path) -> None:
    from machine.money import report as R
    main = out["main"]
    tr, one = trace_table(main), main["rows"][0]["one"]
    t = pd.DataFrame({
        "signal": tr["signal"], "direction": tr["direction"], "opened": tr["open"], "closed": tr["close"],
        "ended_in_the_plan": [MP.when(x).round("s") for x in tr["ended_d"]], "closed_by": tr["closed_by"],
        "stopped_out_at_0": tr["stopped_out"], "positions": tr["positions"],
        "positions_opened_in_the_plan": tr["positions_in_the_plan"], "monday": tr["monday"],
        "money_needed_for_0.01_lot": tr["need_inr"].round(2), "inr_at_0.01_lot": tr["net_inr"].round(2),
        "lowest_point_at_0.01_lot": tr["low_inr"].round(2), "lots": tr["lots"],
        "account_at_open": tr["account_at_open"].round(2), "account_lowest_inside": tr["account_lowest_inside"].round(2),
        "inr_at_that_lot": tr["inr"].round(2), "account_after_the_trade": tr["account_after_trade"].round(2),
        "taken_out": tr["taken_out"].round(2), "days_the_account_waited": tr["days_cut"].round(3)})
    R._csv(t, folder / "trades.csv")
    ev = pd.DataFrame(one.events)
    if len(ev):
        ev.insert(0, "when", [MP.when(x).round("s") for x in ev.pop("t")])
    R._csv(R._rounded(ev, 2), folder / "events.csv")
    lots = [{"year": MP.EPOCH.year + y, "lot": round(s * config.LOT_STEP, 2), "trades": int(c)}
            for y, row in enumerate(one.steps) for s, c in enumerate(row) if c > 0 and s > 0]
    R._csv(pd.DataFrame(lots, columns=["year", "lot", "trades"]), folder / "lots.csv")    # counts, never an average
    R._csv(R._rounded(out["needs"], 4), folder / "mondays.csv")
    R._csv(R._rounded(out["dips"], 3), folder / "dips.csv")
    R._csv(streams_frame(out), folder / "streams.csv")
    ag = out.get("again")
    if ag and ag["rounds"]:                           # Type 1's real order made again after each stop-out (3c point 7)
        g, f = ag["trace"], ag["ts"].frame
        k = g["trade"].to_numpy(int)
        R._csv(pd.DataFrame({
            "signal": g["signal"], "found_again": g["signal"].isin(ag["new_signals"]), "opened": g["open"],
            "closed": f["close"].to_numpy()[k], "closed_by": f["closed_by"].to_numpy()[k],
            "stopped_out_at_0": g["stopped_out"], "positions": f["positions"].to_numpy()[k],
            "money_needed_for_0.01_lot": g["need_inr"].round(2), "inr_at_0.01_lot": f["net_inr"].to_numpy()[k].round(2),
            "lowest_point_at_0.01_lot": f["low_inr"].to_numpy()[k].round(2), "lots": g["lots"],
            "account_at_open": g["account_at_open"].round(2), "account_lowest_inside": g["account_lowest_inside"].round(2),
            "inr_at_that_lot": g["inr"].round(2), "account_after_the_trade": g["account_after_trade"].round(2),
            "taken_out": g["taken_out"].round(2)}), folder / "trades-found-again.csv")
        ev = pd.DataFrame(ag["one"].events)
        if len(ev):
            ev.insert(0, "when", [MP.when(x).round("s") for x in ev.pop("t")])
        R._csv(R._rounded(ev, 2), folder / "events-found-again.csv")
        R._csv(pd.DataFrame(ag["rounds"]), folder / "found-again-rounds.csv")
        new = [s for s in ag["stories"] if s["signal"] in set(ag["new_signals"])]
        path = folder / "type1-found-again-stories.jsonl"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in new), encoding="utf-8", newline="\n")
        tmp.replace(path)


def streams_frame(out: dict) -> pd.DataFrame:
    rows = []
    for r in out["streams"]:
        st, f = r["stream"], r["ts"].frame
        hh, aa = r["rows"][0]["head"], r["rows"][0]["spread"]["all_years"]
        rows.append({"stream": st.key, "entry": st.entry, "version": st.version, "words": st.name,
                     "trades_in_the_plan": len(f), "positions": int(f["positions"].sum()),
                     "before_the_first_monday": r["dropped"], "still_open_at_the_end": len(r["still"]),
                     "inr_at_0.01_lot_a_position": round(float(f["net_inr"].sum()), 2),
                     "real_order_result": round(hh["result"], 2), "stopouts": hh["stopouts"],
                     "blowups": hh["blowups"], "doubles": hh["doubles"], "taken_out": round(hh["out"], 2),
                     "added_back": round(hh["added"], 2), "middle_order_result": round(aa["result_mid"], 2),
                     "bad_10pct_result": round(aa["result_bad"], 2), "orders_with_a_stopout": aa["runs_with_a_stopout"],
                     "trial": out.get("trials", {}).get(st.key, "")})
    return pd.DataFrame(rows)


def _record(out: dict, res: dict) -> Path:
    from machine import trials
    from machine.money import report as R
    st, ts, rows = res["stream"], res["ts"], res["rows"]
    fp = ts.frame[["signal", "open", "close", "gross_inr", "cost_inr", "low_inr", "low0_inr", "need_inr"]]
    lows = {s: [x.t.tolist(), x.v.tolist()] for s, x in zip(ts.frame["signal"], ts.inside)}
    settings = {
        "trades": {"label": "HIS HEDGING (L19b) - not plain trades", "entry": st.entry, "version_id": st.version,
                   "words": st.name, "stories": R._rel(st.stories), "l19b_trial": st.trial,
                   "recorded": res["recorded"], "in_the_plan": len(ts), "before_the_first_monday": res["dropped"],
                   "still_open_at_the_end": len(res["still"]),
                   "window": f"{FIRST}-{LAST} (L19b's; 2021-2023 not opened - every entry is already seen)",
                   "trades_sha256": trials.sha256_of(R._rounded(fp, 6).astype(str).to_dict("list")),
                   "inside_sha256": trials.sha256_of(lows)},
        "version_1": MP.VERSION_1.as_dict(),
        "extra_rows": [r["plan"].as_dict() for r in rows[1:]],
        "inside_a_trade": NOTES + " section 3c (no money in while a trade is open; the broker's stop-out at Rs 0) and "
                                  "3d (row 10)",
        "money_needed_for_0.01_lot": "2 x that week's Monday ADR (the larger of the 5- and 10-day average day range, "
                                     "finished broker days, strategy-code.md section 16 USE 3) x 1 ounce x the rupee "
                                     "rate of that Monday's month; known when the Monday's first 5-minute bar closed",
        "definitions": NOTES + " sections 1 and 3 (committed before the first run on hedged trades)",
        "reshuffles": {"runs": out["n_runs"], "seed": MP.SEED, "blocks": "calendar months, each used once",
                       "months": ts.n_months},
        "year_rule": "a trade counts in the year it OPENED",
        "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
        "already_seen": {"account_doubles_7_8_days": "1 try, 2016-2026 (prior-tries)",
                         "capital_adr_sizing": "10 tries, 2016-2023 (prior-tries)",
                         "the_entry": "tested by the old project on 2016-2023 (L19b's referee rows)"},
        "same_mondays_as_the_plain_run": out.get("same_mondays"),
        "code_sha256": trials.code_sha256([Path(MP.__file__), Path(__file__), Path(R.__file__),
                                           Path(__file__).with_name("rerun.py")]),
    }
    result = {
        "flat_0.01_lot_a_position": {"total": float(ts.frame["net_inr"].sum()),
                                     "price_moves": float(ts.frame["gross_inr"].sum()),
                                     "cost": float(ts.frame["cost_inr"].sum())},
        "version_1": {"real_order": rows[0]["head"], "reshuffles": rows[0]["spread"]},
        "rows": [{"name": r["plan"].name, "change": r["plan"].change, "real_order": r["head"],
                  "reshuffles": {k: r["spread"][k] for k in ("all_years", "any_year", "worst")}} for r in rows[1:]],
        "still_open_at_the_end": out["still"].get(st.key, []),
        "pictures": [{k: p[k] for k in ("path", "pixel_sha256")} for p in out.get("pictures", [])]
        if st.key == MAIN else [],
    }
    if st.key == MAIN:
        result["version_1"]["real_order"] = {**result["version_1"]["real_order"], "years": out["years"],
                                             "facts": {k: v for k, v in out["facts"].items() if k != "dropped_frame"}}
        ag = out.get("again")
        settings["found_again"] = (
            NOTES + " section 3c point 7: in the real order of version 1, after each stop-out the hedging tester (L19b's "
            "set-up for Type 1, machine.money.rerun.Type1Hedging; its replay from the start = L19b's recorded trades, "
            "leg by leg) runs again on the signals known after that minute, and the plan runs again on the new list, "
            "until no new stop-out comes" if ag is not None else "not made again in this run")
        if ag is not None:
            sig = ag["ts"].frame[["signal", "open", "close", "net_inr", "low_inr", "need_inr"]]
            result["found_again"] = {
                "rounds": ag["rounds"], "real_order": ag["head"], "years": ag["years"],
                "trades": len(ag["ts"]), "new_signals": ag["new_signals"],
                "replays": out.get("again_replays", []),
                "still_open_at_the_end": len(ag["still"]),
                "trades_sha256": trials.sha256_of(R._rounded(sig, 6).astype(str).to_dict("list"))}
    return trials.record(
        kind=KIND, idea_id=st.entry, version_id=st.version, parent_idea=None, settings=settings, data=out["data"],
        result=R._plain(result),
        note=f"L20 second run: his money plan on L19b's hedged trades of {st.entry} ({st.name}) - the real order and "
             f"{out['n_runs']} reshuffles of whole months, {len(rows)} setting(s); the same version, not a new one")


# ================================================================================================================
# the report: easy English (CLAUDE.md #113, #114)
# ================================================================================================================

def _inr():
    from machine.costs import inr
    return inr


def signed(x: float) -> str:
    inr = _inr()
    return inr(x) if round(x) <= 0 else "+" + inr(x)


def pc(x: float) -> str:
    return f"{x * 100:+.1f}%"


def num(x, fmt: str = ".0f") -> str:
    return "none" if x is None or not np.isfinite(x) else format(x, fmt)


def days_list(days) -> str:
    d = sorted(int(round(x)) for x in days)
    if not d:
        return "none"
    if len(d) == 1:
        return f"{d[0]} days"
    return ", ".join(str(x) for x in d[:-1]) + f" and {d[-1]} days"


def _days(x: float) -> str:
    d = int(round(x))
    return f"{d} day" if d == 1 else f"{d} days"


def _lot_range(row) -> str:
    used = [s for s, c in enumerate(row) if c > 0 and s > 0]
    if not used:
        return "-"
    return f"{used[0] * 0.01:.2f}" if len(used) == 1 else f"{used[0] * 0.01:.2f} to {used[-1] * 0.01:.2f}"


def _each(refills) -> str:
    """The size of each refill, when there was more than one ("₹20,000 each time" when they are all the same). A
    blow-up that got nothing (the reserve was empty: the plan stopped there) is said on its own."""
    inr = _inr()
    got = [x for x in refills if round(x) > 0]
    empty = "; nothing at the last one: the reserve was empty" if len(got) < len(refills) else ""
    if len(got) < 2:
        return f" ({empty[2:]})" if empty else ""
    lo, hi = min(got), max(got)
    each = f"{inr(lo)} each time" if round(lo) == round(hi) else f"{inr(lo)} to {inr(hi)} each time"
    return f" ({each}{empty})"


def _reserve_words(head: dict) -> str:
    inr = _inr()
    return (f"ran out in {head['reserve_used_up']}" if head["reserve_used_up"]
            else f"{inr(head['reserve_end'])} left")


def again_facts(ag: dict) -> dict:
    """What the report says about Type 1's real order made again after each stop-out (rerun.made_again). Only the
    trades the plan TOOK count (after the reserve is used up, no trade is taken); a trade the broker closed counts the
    positions it had opened by then (rerun.cut)."""
    one, f, g = ag["one"], ag["ts"].frame, ag["trace"]
    taken = f.iloc[g["trade"].to_numpy(int)]
    refills = [float(e["money_in_out"]) for e in one.events if e["what"] == "blow-up"]
    return {"head": ag["head"], "dbl_days": one.dbl_days, "years": ag["years"], "refills": refills,
            "worst": min(ag["years"], key=lambda r: r["result"]), "trades": len(taken),
            "positions": int(taken["positions"].sum()), "new": int(taken["signal"].isin(ag["new_signals"]).sum()),
            "not_taken": len(f) - len(taken), "rounds": len(ag["rounds"])}


def _row(out: dict, name: str) -> dict:
    return next(r for r in out["main"]["rows"] if r["plan"].name == name)


def _mid(out: dict, name: str) -> float:
    return _row(out, name)["spread"]["all_years"]["result_mid"]


ROW_WORDS = {
    "his plan, version 1": "as written",
    "row 1": "take the profit out at +75%, not at a double",
    "row 2": "the account 5% of your money (₹10,000 in, ₹190,000 outside)",
    "row 3": "the account 20% of your money (₹40,000 in, ₹160,000 outside)",
    "row 4": "never take the profit out",
    "row 5": "no chocolate rule: 0.01 lot always",
    "row 6": "your sheet's number: ₹7,100 for 0.01 lot every week, not 2 × Monday's ADR",
    "row 7": "refill only up to the money needed for 0.01 lot",
    "row 8": "between trades, a blow-up only at ₹0 (below the line it still trades 0.01 lot)",
    "row 9": "NOT REAL: before the ₹40 cost",
    "row 10": "the reserve keeps an open trade alive: below the line inside a trade, it tops the account up to ₹20,000",
}


def report_md(out: dict) -> str:
    inr = _inr()
    main, fx, years, n = out["main"], out["facts"], out["years"], out["n_runs"]
    v1 = main["rows"][0]
    h, s, one = v1["head"], v1["spread"], v1["one"]
    a, plan = s["all_years"], one.plan
    start = plan.your_money
    ny = len(years)
    worst_year = min(years, key=lambda r: r["result"])
    lost = h["result"] < 0
    tr = trace_table(main)
    trials_ = out.get("trials", {})
    flat = fx["flat_total"]
    L: list[str] = []
    add = L.append

    add("# Your money plan on your hedged trades (L20, second run)")
    add("")
    add("> For Tejas · made by `python -m machine.money.hedged` · trial files: "
        + (", ".join(f"`{v}`" for v in trials_.values()) if trials_ else "not written")
        + f" · what each word means was fixed before the run: `{NOTES}` sections 1 and 3.")
    add(f"> **YOUR HEDGING.** The trades are L19b's: each of your entry types with your hedging, {FIRST}–{LAST}, "
        f"every position 0.01 lot to start with, ₹40 a position. The first run (`{PLAIN_REPORT}`) used plain trades "
        f"with a stop. This one uses your hedging.")
    add("> **Already seen:** the old project tested every one of these entry types on 2016–2023. It also looked at "
        "\"doubles in 7–8 days\" once and at the capital rule 10 ways. These are not fresh years for any of them.")
    add("")

    # ---------------------------------------------------------------- the answer
    add("## The answer")
    add("")
    slow = not np.isfinite(h["days_mid"]) or h["days_mid"] > HIS_DAYS
    add(f"**On Type 1 with your hedging the plan {'lost' if lost else 'made'} money, and "
        + ("the account did not double every 7–8 days.**" if slow
           else f"the middle double took {h['days_mid']:.0f} days.**"))
    add("")
    ag = out.get("again")
    two = bool(ag and ag["rounds"])
    gx = again_facts(ag) if two else None

    def cells(first: str, second) -> str:
        return f"| {first} |" + (f" {second} |" if two else "")

    add(f"| | the real order, {FIRST}–{LAST} |" + (" the same, with your trades found again after each stop-out |"
                                                   if two else ""))
    add("|---|---|" + ("---|" if two else ""))
    add(cells(f"your {inr(start)} | became **{inr(h['your_money_end'])}** ({signed(h['result'])})",
              gx and f"became **{inr(gx['head']['your_money_end'])}** ({signed(gx['head']['result'])})"))
    add(cells(f"doubles | **{h['doubles']} in {ny} years**" + (f": after {days_list(one.dbl_days)}"
                                                              if len(one.dbl_days) else ""),
              gx and (f"**{gx['head']['doubles']} in {ny} years**" + (f": after {days_list(gx['dbl_days'])}"
                                                                     if len(gx["dbl_days"]) else ""))))
    add(cells(f"blow-ups | **{h['blowups']} in {ny} years**, about {h['blowups'] / ny:.1f} a year (you accept "
              f"{HIS_BLOWUPS[0]} to {HIS_BLOWUPS[1]}). In **{h['stopouts']}** of them the broker closed the trade at "
              f"₹0 while it was still open",
              gx and ((f"**{gx['head']['blowups']}**, then the reserve was empty ({gx['head']['reserve_used_up']}) "
                       f"and the plan stopped" if gx["head"]["reserve_used_up"] else
                       f"**{gx['head']['blowups']} in {ny} years**, about {gx['head']['blowups'] / ny:.1f} a year")
                      + f". In **{gx['head']['stopouts']}** of them the broker closed the trade at ₹0")))
    add(cells(f"taken out | {inr(h['out'])}", gx and inr(gx["head"]["out"])))
    add(cells(f"added back after blow-ups | {inr(h['added'])}{_each(fx['refills'])}",
              gx and f"{inr(gx['head']['added'])}{_each(gx['refills'])}"))
    add(cells(f"the reserve ({inr(plan.reserve_size)} at the start) | {_reserve_words(h)}",
              gx and _reserve_words(gx["head"])))
    add(cells(f"the worst year | {worst_year['year']}: {signed(worst_year['result'])} ({worst_year['blowups']} "
              f"blow-ups, {worst_year['doubles']} doubles)",
              gx and f"{gx['worst']['year']}: {signed(gx['worst']['result'])} ({gx['worst']['blowups']} blow-ups, "
                     f"{gx['worst']['doubles']} doubles)"))
    add(cells(f"trades | {fx['trades']} in {ny} years ({fx['positions']} positions). A trade stayed open "
              f"{_days(fx['days_mid'])} in the middle, {_days(fx['days_max'])} at most",
              gx and (f"{gx['trades']} ({gx['positions']} positions); {gx['new']} of them are not in the first list"
                      + (f". {gx['not_taken']} more came after the reserve was empty: not taken" if gx["not_taken"]
                         else ""))))
    add("")
    if two:
        add(f"**The second column.** L19b found your trades as if no trade was ever closed by the broker. After a "
            f"stop-out, the trades you would have taken in that time are missing from its list, and the account only "
            f"waits ({fx['cut_days']:.0f} days in all). So your hedging was run again from the minute of each "
            f"stop-out ({len(ag['rounds'])} times): the same entries, your management, the same settings. Only the "
            f"real order can be made again: the {n:,} other orders and the one-change rows below use the first "
            f"column's trades.")
        add("")
    elif fx["stopped"] and ag is None:
        add("**Not made again in this run:** the trades after each stop-out (a look only).")
        add("")
    worse, better = fx["orders_worse"] / n, fx["orders_better"] / n
    kind = "a lucky one" if worse >= 0.8 else ("an unlucky one" if better >= 0.8 else "an ordinary one")
    add(f"**The real order was {kind}.** The same months in {n:,} other orders: {fx['orders_worse']:,} ended with "
        f"less, {fx['orders_better']:,} with more. The middle one ended at {inr(start + a['result_mid'])} "
        f"({signed(a['result_mid'])}).")
    rest3 = h["result"] - fx["best3_sum"]
    if fx["trades"] > 3:
        add(f"The 3 best trades made {signed(fx['best3_sum'])}. Without them the real order "
            f"{'loses' if rest3 < 0 else 'makes'} {inr(abs(rest3))}.")
    add("")
    add("**Why it lost:**" if lost else "**Why:**")
    add(f"- The trades themselves {'lose' if flat < 0 else 'make'} {inr(abs(flat))} at 0.01 lot a position, with no "
        f"money plan at all. The price moves made {signed(fx['flat_gross'])}, the ₹40 cost took "
        f"{inr(fx['flat_cost'])} ({fx['positions']} positions).")
    add(f"- The plan puts the whole account behind every trade: lots {fx['lot_min']:.2f} to {fx['lot_max']:.2f}, "
        f"every position of a trade at that lot.")
    if fx["stopped"]:
        w = fx["stopped_would"]
        add(f"- In {fx['stopped']} of the {fx['trades']} trades the money in the account reached ₹0 while the trade "
            f"was open. The broker closed them: {inr(-fx['stopped_inr'])} lost. Left to run to their end, the same "
            f"trades at the same lots would have {'made' if w > 0 else 'lost'} {inr(abs(w))}.")
    add(f"- The {fx['own_end']} trades that ended by themselves {'made' if fx['own_end_inr'] >= 0 else 'lost'} "
        f"{inr(abs(fx['own_end_inr']))} at the plan's lots.")
    if fx["under"]:
        add(f"- In {fx['under']} more {'trade' if fx['under'] == 1 else 'trades'} the account went below the money "
            f"needed for 0.01 lot but stayed above ₹0, and the trade went on to its end.")
    add("")
    add("![The account in the real order](money-plan-hedged/account.jpg)")
    add("")
    marks = {e["what"]: 0 for e in one.events}
    for e in one.events:
        if e["what"] != "blow-up" or "broker" not in e["why"]:
            marks[e["what"]] += 1
    add("How to read it: the black line is the money in the account, at its lowest point in each hour. The red line is "
        "the money needed for 0.01 lot."
        + (" A red cross: the broker closed a trade at ₹0." if marks.get("stop-out") else "")
        + (" A red mark: a blow-up after a trade, and the reserve fills the account back to ₹20,000."
           if marks.get("blow-up") else "")
        + (" A green mark: a double, the profit is taken out." if marks.get("a double") else ""))
    add("")

    # ---------------------------------------------------------------- inside a hedged trade
    add("## Inside a hedged trade")
    add("")
    add("A hedged trade stays open for weeks or months, and the money in the account moves with it. Your words decide "
        "what happens then (strategy.md §12: \"accept occasional blow-ups (real capital stays safe)\"; strategy-code "
        "§18: \"occasional full blow-ups accepted\"):")
    add("- No money goes into the account while a trade is open. The reserve stays outside.")
    add("- If the money in the account reaches ₹0 while the trade is open, the broker closes every position of the "
        "trade. The whole account is lost. That is a blow-up: the reserve fills the account back to ₹20,000.")
    add("- Below the money needed for 0.01 lot, but above ₹0: the trade goes on.")
    add("")
    add("Every Type 1 trade in the real order:")
    add("")
    add("| opened | days open | positions | lot | the account when it opened | its lowest point inside the trade | "
        "how it ended | money at that lot |")
    add("|---|---:|---:|---:|---:|---:|---|---:|")
    for r in tr.itertuples():
        days = r.ended_d - r.opened_d
        how = "the broker closed it at ₹0" if r.stopped_out else "by itself"
        add(f"| {pd.Timestamp(r.open):%d %b %Y} | {days:.0f} | {r.positions_in_the_plan} | {r.lots:.2f} | "
            f"{inr(r.account_at_open)} | {inr(r.account_lowest_inside)} | {how} | {signed(r.inr)} |")
    add("")
    if fx["stopped"]:
        add("- A trade the broker closed counts the positions it had opened by then.")
    if fx["stopped"]:
        add(f"- After the broker closed a trade, the account waited for your next trade in L19b's list. L19b made the "
            f"list without these stop-outs, so the trades that would have come in between are not in it: "
            f"{fx['cut_days']:.0f} days of waiting in all."
            + (" The second column of the answer finds those trades again." if two else ""))
    add("- ₹0 is taken as the broker's line. The real broker closes a little above ₹0 (at a margin level). With "
        "hedged positions that is close to ₹0.")
    if fx["dropped"]:
        d0 = fx["dropped_frame"]
        add(f"- {fx['dropped']} trade opened before the first Monday number (18 Jan 2016), so it is not in the plan: "
            + ", ".join(f"{pd.Timestamp(x.open):%d %b %Y} ({signed(x.net_inr)} at 0.01 lot a position)"
                        for x in d0.itertuples()) + ".")
    add("")

    # ---------------------------------------------------------------- the plan as run
    add("## The plan, as the machine ran it")
    add("")
    add("| your rule | how the machine does it |")
    add("|---|---|")
    add(f"| money needed for 0.01 lot = 2 × that Monday's ADR | in rupees, kept for the week. It was between "
        f"{inr(fx['need_min'])} and {inr(fx['need_max'])} in these {ny} years |")
    add("| the ₹100-of-chocolate rule | lots = the money in the account ÷ the money needed for 0.01 lot, rounded "
        "down to 0.01, 0.02, 0.03 … Every position of the trade (the entry, every hedge) is that lot |")
    add("| one trade at a time | a trade opens only after the last one ended |")
    add(f"| a burner account of about 10% | your money {inr(start)}: {inr(plan.account_size)} in the account, "
        f"{inr(plan.reserve_size)} kept outside (the reserve) |")
    add("| withdraw at each double | when the account reaches 2 × its starting money after a trade, everything above "
        "the starting money is taken out |")
    add("| **a blow-up** | the account below the money needed for 0.01 lot when no trade is open, OR the broker "
        "closed a trade at ₹0 |")
    add(f"| **the refill** | the reserve brings the account back to {inr(plan.account_size)} |")
    add("")

    # ---------------------------------------------------------------- year by year
    add("## Year by year, the real order")
    add("")
    add("| year | trades | money needed for 0.01 lot | lots used | blow-ups | of them at ₹0 | doubles | taken out | "
        "added back | the year's result | growth of your money | the same trades at 0.01 lot a position |")
    add("|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in years:
        y = r["year"] - MP.EPOCH.year
        dd = one.dbl_days[one.dbl_year == y]
        need = (f"{inr(r['need_lo'])} to {inr(r['need_hi'])}" if np.isfinite(r["need_lo"]) else "-")
        add(f"| {r['year']} | {r['trades']} | {need} | {_lot_range(one.steps[y])} | {r['blowups']} | {r['stopouts']} "
            f"| {r['doubles']}{' (' + days_list(dd) + ')' if len(dd) else ''} | {inr(r['taken_out'])} | "
            f"{inr(r['added_back'])} | {signed(r['result'])} | {pc(r['growth'])} | "
            f"{signed(fx['flat_by_year'].get(r['year'], 0.0))} |")
    add(f"| **all** | **{h['taken']}** | | {fx['lot_min']:.2f} to {fx['lot_max']:.2f} | **{h['blowups']}** | "
        f"**{h['stopouts']}** | **{h['doubles']}** | **{inr(h['out'])}** | **{inr(h['added'])}** | "
        f"**{signed(h['result'])}** | | **{signed(flat)}** |")
    add("")
    add("- A trade counts in the year it opened.")
    add("- Growth = the year's result ÷ your money at the start of that year (the reserve + the account + everything "
        "taken out).")
    add("- How many trades at each lot size, year by year: `plan/work/money-plan-hedged/lots.csv`.")
    add("")

    # ---------------------------------------------------------------- the reshuffles
    add(f"## The same months in {n:,} other orders")
    add("")
    add(f"The {main['ts'].n_months} months are put in a random order, each month used once. The trades inside a month "
        f"stay as they were. A long trade pushes the next trades later (one trade at a time). The plan runs through "
        f"each order from the same start. Seed {MP.SEED}.")
    add("")
    add("![Where your money ended in each order](money-plan-hedged/orders.jpg)")
    add("")
    add(f"| over the {ny} years | the middle order | the bad 10% |")
    add("|---|---:|---:|")
    add(f"| your {inr(start)} ends at | {inr(start + a['result_mid'])} ({signed(a['result_mid'])}) | "
        f"{inr(start + a['result_bad'])} or less ({signed(a['result_bad'])}) |")
    add(f"| taken out | {inr(a['out_mid'])} | {inr(a['out_bad'])}{' or less' if round(a['out_bad']) > 0 else ''} |")
    add(f"| added back after blow-ups | {inr(a['added_mid'])} | {inr(a['added_bad'])} or more |")
    add(f"| blow-ups | {a['blowups_mid']:.0f} | {a['blowups_bad']:.0f} or more |")
    add(f"| of them: the broker closed a trade at ₹0 | {a['stopouts_mid']:.0f} | {a['stopouts_bad']:.0f} or more |")
    add(f"| doubles | {a['doubles_mid']:.0f} | {a['runs_no_double']:,} of the {n:,} orders had none |")
    add("")
    add("\"The bad 10%\" = 1 order in 10 was this bad or worse.")
    add((f"None of the {n:,} orders ended above the start. " if fx["orders_made_money"] == 0
         else f"{fx['orders_made_money']:,} of the {n:,} orders ended above the start. ")
        + f"The best one ended at {inr(start + fx['orders_best'])} ({signed(fx['orders_best'])}). "
        + f"In {a['runs_with_a_stopout']:,} of the {n:,} orders the broker closed at least one trade at ₹0.")
    add("")
    add("| year of the plan | growth of your money: middle | bad 10% | blow-ups: middle | bad 10% | orders with a "
        "double that year | days per double: middle | slow 10% |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for y, r in enumerate(s["per_year"]):
        add(f"| {r['year']} | {pc(r['growth_mid'])} | {pc(r['growth_bad'])} | {r['blowups_mid']:.0f} | "
            f"{r['blowups_bad']:.0f} | {fx['runs_with_double_by_year'][y]:,} of {n:,} | {num(r['days_mid'])} | "
            f"{num(r['days_slow'])} |")
    any_ = s["any_year"]
    add(f"| **any year** | {pc(any_['growth_mid'])} | {pc(any_['growth_bad'])} | {any_['blowups_mid']:.0f} | "
        f"{any_['blowups_bad']:.0f} | | {num(any_['days_mid'])} | {num(any_['days_slow'])} |")
    add("")
    add(f"- Days per double counts only the times the account did double ({fx['doubles_all']:,} in all the orders).")
    add(f"- {fx['doubles_in_his_days']:,} of those {fx['doubles_all']:,} doubles came within {HIS_DAYS} days.")
    add("- \"Slow 10%\" = 1 double in 10 took this long or longer.")
    w = s["worst"]
    used = (f"The reserve ran out in year {w['used_up_year']} of the plan." if w["used_up_year"]
            else f"The reserve did not run out: {inr(float(out['worst'].reserve[0]))} of it was left.")
    add("")
    add(f"**The worst order of the {n:,}:** your {inr(start)} ended at {inr(w['your_money_end'])} "
        f"({signed(w['result'])}). {w['blowups']} blow-ups ({w['stopouts']} at ₹0), {w['doubles']} doubles, "
        f"{inr(w['out'])} taken out, {inr(w['added'])} added back. {used}")
    add(f"The reserve ran out in {a['runs_reserve_used_up']:,} of the {n:,} orders.")
    add("")

    # ---------------------------------------------------------------- the rows
    add("## One change at a time (shown, not recommended)")
    add("")
    add("Each row changes ONE thing from the plan above. Same Type 1 trades.")
    add("")
    add("| | the one change | real order: your money ends at | blow-ups | at ₹0 | taken out | added back | "
        f"{n:,} orders: the middle ends at | the bad 10% | orders where the reserve ran out |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in main["rows"]:
        hh, aa, p = r["head"], r["spread"]["all_years"], r["plan"]
        name = "**the plan**" if not p.change else p.name
        stop = f" (the reserve ran out, {hh['reserve_used_up']})" if hh["reserve_used_up"] else ""
        add(f"| {name} | {ROW_WORDS.get(p.name, p.change)} | {inr(hh['your_money_end'])} ({signed(hh['result'])}){stop} "
            f"| {hh['blowups']} | {hh['stopouts']} | {inr(hh['out'])} | {inr(hh['added'])} | "
            f"{inr(p.your_money + aa['result_mid'])} ({signed(aa['result_mid'])}) | "
            f"{inr(p.your_money + aa['result_bad'])} | {aa['runs_reserve_used_up']:,} of {n:,} |")
    add("")
    add("What the rows show:")
    for line in _row_notes(out):
        add(f"- {line}")
    add("")

    # ---------------------------------------------------------------- every stream
    add("## Every entry type with your hedging (the plan as written)")
    add("")
    add("| entry type | trades in the plan | at 0.01 lot a position, no plan | the plan, the real order | blow-ups "
        f"(at ₹0) | doubles | {n:,} orders: the middle | the bad 10% |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in out["streams"]:
        st, f = r["stream"], r["ts"].frame
        hh, aa = r["rows"][0]["head"], r["rows"][0]["spread"]["all_years"]
        few = " — too few trades to say anything" if len(f) < FEW else ""
        add(f"| {st.name}{few} | {len(f)} ({int(f['positions'].sum())} positions) | "
            f"{signed(float(f['net_inr'].sum()))} | {signed(hh['result'])} | {hh['blowups']} ({hh['stopouts']}) | "
            f"{hh['doubles']} | {signed(aa['result_mid'])} | {signed(aa['result_bad'])} |")
    add("")
    add("- The second gap (#124), both readings, took exactly Type 1's trades, so its numbers are Type 1's.")
    add("- Each entry type is run alone here: one trade at a time from that entry type only. Your full way (every "
        "entry together, whichever comes first) is L19d.")
    stills = [(r["stream"], x) for r in out["streams"] for x in out["still"][r["stream"].key]]
    if stills:
        add("")
        add(f"**Trades still open at the end of {LAST}** are not in any number above (a still-open trade is never "
            f"counted). Here is where they stood on its last minute:")
        add("")
        add("| entry type | opened | positions | at 0.01 lot a position | at the lot the plan would give it | the "
            "account's lowest point inside it |")
        add("|---|---|---:|---:|---:|---:|")
        for st, x in stills:
            add(f"| {st.name} | {pd.Timestamp(x['opened']):%d %b %Y} | {x['positions']} | "
                f"{signed(x['inr_at_the_end_001'])} | {signed(x['inr_at_the_end_at_lots'])} ({x['lots']:.2f} lot) | "
                f"{inr(x['account_lowest_inside'])}"
                + (" — the broker would have closed it at ₹0" if x["would_stop_out"] else "") + " |")
    add("")

    _dips_section(out, add)
    _two_readings(out, add)

    # ---------------------------------------------------------------- what would make it wrong
    add("## What would make this wrong")
    add("")
    add("- **Trades that make money after the ₹40 cost.** These years cannot give them: every entry type here is "
        "already seen. Only new prices can (the forward test, L17c).")
    add("- **The ₹0 line.** The real broker closes a little above ₹0, so a little sooner.")
    if fx["stopped"] and two:
        add(f"- **The trades after a trade the broker closed.** In L19b's list the account waited "
            f"{fx['cut_days']:.0f} days for its next trade. With your trades found again from each stop-out, your "
            f"money ends at {inr(gx['head']['your_money_end'])} instead of {inr(h['your_money_end'])} (the answer's "
            f"second column). Only the real order can be made again.")
    elif fx["stopped"]:
        add(f"- **The trades after a trade the broker closed.** The account waited {fx['cut_days']:.0f} days for the "
            f"next trade in L19b's list.")
    add("- **One entry type at a time.** Your full way takes every entry type in one account (L19d).")
    add("- **Your other ways to manage a trade** (L19c). Each would change every trade here.")
    add(f"- **Few trades.** {fx['trades']} trades in {ny} years. The best 3 made {signed(fx['best3_sum'])}.")
    add(f"- **The start.** {inr(plan.account_size)} in the account is your own number (#114, #115). With another "
        f"start every rupee number changes (rows 2 and 3).")
    add("")
    tries = sum(len(r["rows"]) for r in out["streams"])
    add(f"**Tries:** {len(main['rows'])} settings of the plan on Type 1 (the plan + {len(main['rows']) - 1} one-change "
        f"rows) and the plan alone on {len(out['streams']) - 1} other entry types: {tries} in all. Nothing was chosen. "
        f"The old project: 10 tries of the capital rule, 1 of the doubling.")
    add("")

    # ---------------------------------------------------------------- where
    add("## Where the numbers are")
    add("")
    add("- every Type 1 trade with its lot and money: `plan/work/money-plan-hedged/trades.csv`")
    add("- every blow-up, stop-out and double in the real order: `plan/work/money-plan-hedged/events.csv`")
    add("- every entry type in one table: `plan/work/money-plan-hedged/streams.csv`")
    add("- the lowest point inside every trade: `plan/work/money-plan-hedged/dips.csv`")
    add(f"- the {len(out['needs'])} Monday numbers: `plan/work/money-plan-hedged/mondays.csv`")
    if two:
        add("- the answer's second column: every trade `plan/work/money-plan-hedged/trades-found-again.csv`, every "
            "blow-up and stop-out `events-found-again.csv`, each stop-out and what came after it "
            "`found-again-rounds.csv`, the trades found again leg by leg `type1-found-again-stories.jsonl`")
    rec = main["recorded"]
    if rec:
        add(f"- the trades are L19b's recorded ones: Type 1 {rec['trades']} trades, {signed(rec['net_inr'])} at 0.01 "
            f"lot a position (trial `{rec['trial']}`), and so for every entry type")
    add("- the code: `machine/money/hedged.py`, `machine/money/plan.py`, `machine/money/rerun.py`; the tests: "
        "`tests/test_money_hedged.py`, `tests/test_money_plan.py`, `tests/test_money_rerun.py`")
    add("")
    return "\n".join(L)


def _dips_section(out: dict, add) -> None:
    """The 2e measure on L19b's trades: how deep each trade went against the money needed for 0.01 lot."""
    inr = _inr()
    d = out["dips"]
    size = MP.VERSION_1.account_size
    add("## How deep a hedged trade goes")
    add("")
    add("For every trade of every entry type: the lowest point the whole trade reached, next to the money needed for "
        "0.01 lot the week it opened. With the chocolate rule a full account puts every rupee to work, so a dip as "
        "big as the money needed for 0.01 lot takes the whole account.")
    add("")
    add("| entry type | trades | the lowest point was at least the money needed for 0.01 lot | the deepest (0.01 lot a "
        f"position) | a full {inr(size)} account would reach ₹0 inside |")
    add("|---|---:|---:|---:|---:|")
    for r in out["streams"]:
        x = d[d["stream"] == r["stream"].key]
        deep = x.loc[x["low_in_needs"].idxmax()]
        add(f"| {r['stream'].name} | {len(x)} | {int((x['low_in_needs'] >= 1).sum())} | {inr(float(deep['low_inr']))} "
            f"({float(deep['low_in_needs']):.1f} times) | {int((x['full_account_at_low'] <= 0).sum())} |")
    add("")
    add("- \"times\" = the lowest point ÷ the money needed for 0.01 lot that week.")
    add("- Trades still open at the end of 2020 are in this table (their lowest point so far).")
    add("")


def _row_notes(out: dict) -> list[str]:
    inr = _inr()
    n = out["n_runs"]
    mid = lambda name: signed(_mid(out, name))                                     # noqa: E731
    r10 = _row(out, "row 10")
    return [
        f"Row 1 (take out at +75%): middle order {mid('row 1')}. The plan: {mid(MP.VERSION_1.name)}.",
        f"Rows 2 and 3 (the account 5% or 20% of your money): middle order {mid('row 2')} and {mid('row 3')}. At 20% "
        f"the reserve ran out in {_row(out, 'row 3')['spread']['all_years']['runs_reserve_used_up']:,} of the "
        f"{n:,} orders.",
        f"Row 4 (never take out): {mid('row 4')} in the middle order.",
        f"Row 5 (0.01 lot always): {signed(_row(out, 'row 5')['head']['result'])} in the real order, "
        f"{mid('row 5')} in the middle order.",
        f"Row 6 (your sheet's ₹7,100 for 0.01 lot): middle order {mid('row 6')}.",
        f"Row 7 (refill only up to the money needed for 0.01 lot): {_row(out, 'row 7')['head']['blowups']} blow-ups "
        f"in the real order.",
        f"Row 8 (between trades, a blow-up only at ₹0): {_row(out, 'row 8')['head']['blowups']} blow-ups in the real "
        f"order, middle order {mid('row 8')}.",
        f"Row 9 (before the ₹40 cost, NOT REAL): {mid('row 9')} in the middle order. It only shows how much the cost "
        f"takes.",
        f"Row 10 (the reserve keeps an open trade alive): the reserve put {inr(r10['head']['put_in_inside'])} into "
        f"open trades in the real order ({r10['head']['blowups']} blow-ups, {r10['head']['stopouts']} at ₹0); middle "
        f"order {mid('row 10')}, the bad 10% {signed(r10['spread']['all_years']['result_bad'])}. The reserve ran out "
        f"in {r10['spread']['all_years']['runs_reserve_used_up']:,} of the {n:,} orders. This is your #118 question "
        f"(\"if we keep on hedging, how long will the account last?\") with the reserve behind it.",
    ]


def _two_readings(out: dict, add) -> None:
    """The judgement guard (#114): the numbers, two readings and what each would mean, which one the numbers fit -
    and stop. What "fit" means was fixed before the run (L20-notes 3j):
      reading 1 (the lot rule does not fit his hedging) fits when the middle order has at least one trade the broker
        closed at Rs 0 AND keeping the trades alive (row 10) ends the middle order higher than the plan;
      reading 2 (the plan cannot rescue trades that lose after the Rs 40) fits when the trades lose at 0.01 lot a
        position with no plan at all.
    When neither fits, it refuses: the readings must be written again for these trades."""
    inr = _inr()
    fx, n = out["facts"], out["n_runs"]
    head1 = out["main"]["rows"][0]["head"]
    a1 = out["main"]["rows"][0]["spread"]["all_years"]
    mid_v1, mid_r10 = a1["result_mid"], _mid(out, "row 10")
    one_fits = a1["stopouts_mid"] >= 1 and mid_r10 > mid_v1
    two_fits = fx["flat_total"] < 0
    if not (one_fits or two_fits):
        from machine.money.report import MoneyReportError
        raise MoneyReportError("the report's 'two ways to read this' fits neither reading (no trade closed at Rs 0 in "
                               "the middle order, or row 10 not better; and the trades make money at 0.01 lot): "
                               "write hedged._two_readings again for these trades")
    d = out["dips"]
    t1 = d[d["stream"] == MAIN]
    deep, zero = int((t1["low_in_needs"] >= 1).sum()), int((t1["full_account_at_low"] <= 0).sum())
    add("## Two ways to read this")
    add("")
    add("**Reading 1: the money plan does not fit your hedging.** With every rupee at work, one deep dip inside a "
        "hedged trade takes the whole account before the trade can come back. If so, your hedging needs more room "
        "than 2 × the Monday ADR per 0.01 lot (fewer lots, or a reserve behind the trade).")
    add("")
    add("**Reading 2: the plan cannot rescue trades that lose after the ₹40 cost.** If the trades lose on their own, "
        "no money plan makes them pay; it only makes the loss bigger or smaller. If so, the plan can only be judged "
        "on trades that make money after the cost.")
    add("")
    add("**The numbers for reading 1:**")
    add(f"- The broker closed a trade at ₹0 {head1['stopouts']} times in the real order, {a1['stopouts_mid']:.0f} in "
        f"the middle order ({a1['runs_with_a_stopout']:,} of the {n:,} orders had at least one).")
    add(f"- In {deep} of the {len(t1)} Type 1 trades the dip was as big as the money needed for 0.01 lot or bigger. A "
        f"full account would reach ₹0 inside {zero} of them.")
    add(f"- The reserve kept every trade alive (row 10): the middle order {signed(mid_r10)}, against "
        f"{signed(mid_v1)} for the plan.")
    ag = out.get("again")
    if ag and ag["rounds"]:
        g = ag["head"]
        if g["reserve_used_up"]:
            add(f"- With your trades found again after each stop-out (the real order only): the broker closed a trade "
                f"{g['stopouts']} times, and then the reserve was empty ({g['reserve_used_up']}): no trade after that.")
        else:
            add(f"- With your trades found again after each stop-out (the real order only): the broker closed a trade "
                f"{g['stopouts']} times in {len(out['years'])} years, and {inr(g['reserve_end'])} of the reserve was "
                f"left.")
    add("")
    add("**The numbers for reading 2:**")
    add(f"- With no plan at all (0.01 lot a position), the Type 1 trades {'lose' if fx['flat_total'] < 0 else 'make'} "
        f"{inr(abs(fx['flat_total']))}: price moves {signed(fx['flat_gross'])}, cost {inr(fx['flat_cost'])}.")
    add(f"- Before the ₹40 cost (NOT REAL) the plan's middle order is {signed(_mid(out, 'row 9'))}.")
    add(f"- The more money at work, the more it moves. Middle order: account 5% {signed(_mid(out, 'row 2'))}, "
        f"account 10% {signed(mid_v1)}, account 20% {signed(_mid(out, 'row 3'))}.")
    add("")
    if one_fits and two_fits:
        add("**The numbers fit both.** The trades lose after the ₹40 cost on their own, AND the stop-outs cut trades "
            "that would have come back: keeping them alive with the reserve ends the middle order higher.")
    elif two_fits:
        add("**The numbers fit reading 2.** The trades lose after the ₹40 cost on their own. "
            + ("Keeping every trade alive with the reserve did not help: the middle order ends lower (row 10)."
               if a1["stopouts_mid"] >= 1 else "No trade was closed at ₹0 in the middle order."))
    else:
        add("**The numbers fit reading 1.** The trades make money on their own at 0.01 lot a position, but the "
            "stop-outs cut trades that would have come back: keeping them alive with the reserve ends the middle "
            "order higher.")
    add(f"- How much of it is 3 trades: the best 3 made {signed(fx['best3_sum'])} of the real order's "
        f"{signed(head1['result'])}. {fx['trades']} trades; {len(out['main']['rows'])} settings of the plan tried.")
    add("")
    add("**Two things the test does say about the plan itself:**")
    real_days = out["main"]["rows"][0]["one"].dbl_days
    ag = out.get("again")
    found = ag["one"].dbl_days if ag and ag["rounds"] else np.zeros(0)
    quick = int((real_days <= HIS_DAYS).sum() + (found <= HIS_DAYS).sum()) + fx["doubles_in_his_days"]
    fastest = int(round(float(np.min(real_days)))) if len(real_days) else None
    add("- \"Doubles in 7–8 days\" " + ("did not happen. " if quick == 0 else f"came {quick} times in all. ")
        + (f"The fastest double in the real order took {fastest} days. " if fastest else "No double in the real "
           "order. ")
        + (f"With your trades found again: {len(found)} {'double' if len(found) == 1 else 'doubles'}, after "
           f"{days_list(found)}. " if len(found) else "")
        + f"In the {n:,} orders, {fx['doubles_in_his_days']:,} of {fx['doubles_all']:,} doubles came within "
          f"{HIS_DAYS} days. (Already seen by the old project.)")
    add(f"- For your hedging, 2 × the Monday ADR was not enough room in {deep} of {len(t1)} Type 1 trades.")
    add("")
    add("The decision is yours.")
    add("")


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m machine.money.hedged", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=MP.RESHUFFLES, help="how many reshuffles of whole months (1,000)")
    ap.add_argument("--no-write", action="store_true", help="a look only: no report, no pictures, no trial files")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = run(n_runs=args.runs, write=not args.no_write)
    print(out["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
