"""The 2x2 test (L19a step 3): {his entries, random entries} x {his hedging, a plain stop}, year by year.

    .venv\\Scripts\\python -m machine.tester.hedging_2x2                       # idea #1's chosen version, 2016-2020
    .venv\\Scripts\\python -m machine.tester.hedging_2x2 --runs 20 --no-write  # a quick look, no files

Never buy-and-hold (#114): all four boxes trade the same way, so each comparison moves only ONE thing.
  his entries      the idea's own signals (idea #1, version v-8cc96cf812 = the version L10 chose: U1=A, U2=A, U3=B).
  random entries   each of his signals gets a random twin: the same month, direction, entry type, hedge and target
                   distances, cancel distance and waiting time; a random bar close in that month; a limit distance
                   drawn from his own (machine/referee/random_entries: its template and make_signal). The SAME twins
                   go through both managements. config.RANDOM_RUNS runs (200).
  his hedging      machine/tester/hedging.py (his T12 table, version 1).
  a plain stop     the tester (L3): a stop at the hedge point, the same target - what idea #1's L10 report used.
What the ENTRY adds   = his entries minus random entries, inside one management.
What the HEDGING adds = hedging minus the plain stop, for the same entries.
The whole             = his entries with hedging minus random entries with a plain stop.
Money: Rs after Rs 40 a position, a trade counts in the year it closed; trades still open at the end are not counted
(and said). "Beats" compares the TOTAL rupees over the window (every box gets the same signals every month).
Only 2016-2020: idea #1's hidden years are used up (plan/work/ideas/idea-001/decision.md; the machine refuses them).
Writes ONE trial file (kind "hedging-2x2"), plan/work/L19a-2x2.md and plan/work/L19a/ (his trades, one row each, and
each hedged trade's story).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, costs, guards, settings, trials
from machine.data import bars as BARS
from machine.ideas.compile import Market, compile_idea
from machine.ideas.form import Idea
from machine.referee import random_entries as RE
from machine.referee.replays import Segment
from machine.tester import engine, hedging
from machine.tester.run import summarize as stop_summary

IDEA, READINGS = "idea-001", {"U1": "A", "U2": "A", "U3": "B"}       # L10's choice: v-8cc96cf812
OUT_MD = config.WORK_DIR / "L19a-2x2.md"
OUT_DIR = config.WORK_DIR / "L19a"
ARMS = ("his entries + hedging", "his entries + plain stop", "random entries + hedging", "random entries + plain stop")


def twins(seg: Segment, signals, rng: np.random.Generator, run: int) -> list:
    """One random twin per signal (the module text). Returns tester Signals."""
    times = seg.known_times()
    months = times.to_period("M")
    pool = {m: np.flatnonzero(months == m) for m in months.unique()}
    temps = [RE.template(seg, s) for s in signals]
    offsets = np.array([t.offset for t in temps if t.offset is not None])
    out = []
    for k, (s, t) in enumerate(zip(signals, temps)):
        idx = pool.get(pd.Timestamp(s.known_at).to_period("M"))
        if idx is None or not len(idx):
            continue
        when = times[idx[rng.integers(len(idx))]]
        off = float(offsets[rng.integers(len(offsets))]) if t.entry_type == "limit" else None
        out.append(RE.make_signal(seg, t, when, off, f"rand{run}-{k}"))
    return out


def _years(first: pd.Timestamp, last: pd.Timestamp) -> list[int]:
    return list(range(first.year, last.year + 1))


def _stop_by_year(rep: engine.Replay, years) -> tuple[dict, int, int]:
    out = {y: 0.0 for y in years}
    n = 0
    for t in rep.trades:
        if t.status == "closed":
            out[pd.Timestamp(t.exit_ts).year] += t.net_inr
            n += 1
    return out, n, sum(t.status != "closed" for t in rep.trades)


def _hedge_by_year(rep: hedging.HedgeReplay, years) -> tuple[dict, int, int]:
    out = {y: 0.0 for y in years}
    n = 0
    for t in rep.trades:
        if t.status == "closed":
            out[t.exit_ts.year] += t.net_inr
            n += 1
    return out, n, sum(t.status != "closed" for t in rep.trades)


def run(idea: str = IDEA, readings: dict | None = None, start: str = "2016", end: str = "2020",
        n_runs: int = config.RANDOM_RUNS, write: bool = True, log=print) -> dict:
    readings = READINGS if readings is None else readings
    t0 = time.time()
    market = Market.load(start, end)
    form = Idea.load(config.IDEAS_DIR / idea / "form.json")
    comp = compile_idea(form, market, readings=readings)
    m1 = BARS.load_bars("1m", start, end)
    guards.check_opened(idea, [m1.index[0], m1.index[-1]], version=comp.version_id)     # the hidden years: refused
    book = engine.bars_book(m1, comp.tf)
    fp0 = settings.fingerprint(settings.resolve({}))
    facts = hedging.MarketFacts.build(market, fp0, book)
    rules = hedging.Rules.now()
    years = _years(book.first, book.last)
    log(f"{idea} {comp.version_id}: {len(comp.signals)} signals; his setups that can close a leg: {len(facts.setups)} "
        f"of {facts.setup_signals}; {time.time() - t0:.0f} s")

    his_h = hedging.replay(comp.signals, facts, comp.tf, rules)
    his_s = engine.replay(comp.signals, None, comp.tf, config.TRADES_AT_ONCE, book=book)
    seg = Segment(f"{start}-{end}", tuple(years), comp.tf, comp.signals, m1, book, m1["close"].to_numpy(float))
    seed = RE.seed_of([seg])
    rand = {"hedge": [], "stop": []}
    t1 = time.time()
    for r in range(n_runs):
        sig = twins(seg, comp.signals, np.random.default_rng([seed, r]), r)
        rand["hedge"].append(_hedge_by_year(hedging.replay(sig, facts, comp.tf, rules), years))
        rand["stop"].append(_stop_by_year(engine.replay(sig, None, comp.tf, config.TRADES_AT_ONCE, book=book), years))
        if (r + 1) % 20 == 0:
            log(f"  random run {r + 1}/{n_runs} ({time.time() - t1:.0f} s)")

    result = _result(his_h, his_s, rand, years)
    settings_used = {
        "idea_id": idea, "version_id": comp.version_id, "readings": readings, "timeframe": comp.tf,
        "window": {"asked": [start, end], "first_bar": str(book.first), "last_bar": str(book.last)},
        "management_hedging": {"table": hedging.TABLE, "rules": rules.as_dict(),
                               "rows": {m: what for m, what, _ in hedging.RULES}},
        "management_stop": "the tester (L3): a stop at the hedge point, the same target (M02's version)",
        "random_entries": {"runs": n_runs, "seed": seed, "keeps": "month, direction, entry type, hedge and target "
                           "distances, cancel distance, waiting time, lots; the SAME twins in both managements",
                           "changes": "the time (a random bar close in the same month) and a limit's distance "
                           "(drawn from his own)"},
        "closing_setups": {"form": "hedging.HIS_GRAB_GAP (strategy.md section 9, Type 1)", "signals":
                           facts.setup_signals, "filled": len(facts.setups)},
        "news": f"Forex Factory red + orange, exact times: {len(facts.news)} moments in the window",
        "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
        "code_sha256": trials.code_sha256([Path(hedging.__file__), Path(engine.__file__), Path(__file__)]),
        "signals_sha256": trials.sha256_of([s.model_dump(mode="json") for s in comp.signals]),
    }
    data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
            "gold_m1_content_sha256": BARS.m1_fingerprint()}
    out = {"result": result, "settings": settings_used, "his_hedging": hedging.summarize(his_h),
           "his_stop": stop_summary(his_s), "seconds": round(time.time() - t0)}
    if write:
        path = trials.record(kind="hedging-2x2", idea_id=idea, version_id=comp.version_id,
                             parent_idea=comp.parent_idea, settings=settings_used, data=data,
                             result={**result, "his_hedging": out["his_hedging"], "his_stop": out["his_stop"]},
                             note="L19a: the 2x2 test - the chosen version (L10), not a new version; management "
                                  "hedging = his T12 table, version 1")
        out["trial"] = path.name
        _write_trades(his_h, book)
        OUT_MD.write_text(report_md(out), encoding="utf-8")
    return out


def _result(his_h, his_s, rand, years) -> dict:
    hh, n_hh, open_hh = _hedge_by_year(his_h, years)
    hs, n_hs, open_hs = _stop_by_year(his_s, years)
    rh = np.array([[r[0][y] for y in years] for r in rand["hedge"]])
    rs = np.array([[r[0][y] for y in years] for r in rand["stop"]])
    n_rh = np.array([r[1] for r in rand["hedge"]])
    n_rs = np.array([r[1] for r in rand["stop"]])
    open_rh = np.array([r[2] for r in rand["hedge"]])

    def row(vals, trades, open_end, label):
        v = {str(y): round(float(x), 2) for y, x in zip(years, vals)}
        worst = min(v, key=v.get)
        return {"arm": label, "per_year": v, "total": round(float(sum(vals)), 2), "trades": int(trades),
                "open_at_end": int(open_end), "worst_year": worst}

    def rand_row(mat, trades, open_end, label):
        med = np.median(mat, axis=0)
        r = row(med, np.median(trades), np.median(open_end) if open_end is not None else 0, label)
        tot = mat.sum(axis=1)
        r.update({"median_of_runs": True, "total": round(float(np.median(tot)), 2),
                  "total_5_95": [round(float(np.percentile(tot, 5)), 2), round(float(np.percentile(tot, 95)), 2)],
                  "per_year_5_95": {str(y): [round(float(np.percentile(mat[:, k], 5)), 2),
                                             round(float(np.percentile(mat[:, k], 95)), 2)]
                                    for k, y in enumerate(years)}})
        return r

    boxes = [row([hh[y] for y in years], n_hh, open_hh, ARMS[0]), row([hs[y] for y in years], n_hs, open_hs, ARMS[1]),
             rand_row(rh, n_rh, open_rh, ARMS[2]), rand_row(rs, n_rs, None, ARMS[3])]
    his_h_tot, his_s_tot = boxes[0]["total"], boxes[1]["total"]

    def beats(his_total, mat):
        return int((mat.sum(axis=1) < his_total).sum())

    def beats_year(his_vals, mat):
        return {str(y): int((mat[:, k] < his_vals[k]).sum()) for k, y in enumerate(years)}

    hh_v = np.array([hh[y] for y in years])
    hs_v = np.array([hs[y] for y in years])
    adds = {
        "entry_with_hedging": _diff(hh_v, np.median(rh, axis=0), years),
        "entry_with_stop": _diff(hs_v, np.median(rs, axis=0), years),
        "hedging_for_his_entries": _diff(hh_v, hs_v, years),
        "hedging_for_random_entries": _diff(np.median(rh - rs, axis=0), np.zeros(len(years)), years),
        "whole": _diff(hh_v, np.median(rs, axis=0), years),
    }
    return {
        "years": [str(y) for y in years], "boxes": boxes, "adds": adds, "runs": int(len(rh)),
        "his_beats_random": {"with_hedging": beats(his_h_tot, rh), "with_stop": beats(his_s_tot, rs),
                             "with_hedging_by_year": beats_year(hh_v, rh),
                             "with_stop_by_year": beats_year(hs_v, rs)},
        "hedging_beats_stop_in_random_runs": int(((rh - rs).sum(axis=1) > 0).sum()),
    }


def _diff(a, b, years) -> dict:
    d = {str(y): round(float(x - z), 2) for y, x, z in zip(years, a, b)}
    worst = min(d, key=d.get)
    return {"per_year": d, "total": round(float(np.sum(a) - np.sum(b)), 2), "worst_year": worst,
            "years_positive": sum(v > 0 for v in d.values())}


def _write_trades(rep: hedging.HedgeReplay, book) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hedging.trades_table(rep.trades).to_csv(OUT_DIR / "idea-001-hedged-trades.csv", index=False)

    def when(i):
        return None if i is None else str(pd.Timestamp(int(book.t[i])))
    with (OUT_DIR / "idea-001-hedged-stories.jsonl").open("w", encoding="utf-8") as f:
        for t in rep.trades:
            legs = [{"direction": x.direction, "opened": when(x.open_i), "open_px": x.open_px, "why_open": x.why_open,
                     "closed": when(x.close_i), "close_px": x.close_px, "why_closed": x.why_closed,
                     "net_inr": None if x.pnl is None else round(x.pnl.net_inr, 2)} for x in t.legs]
            f.write(json.dumps({"signal": t.signal_id, "status": t.status, "net_inr": t.net_inr,
                                "story": [{**r, "ts": str(r["ts"])} for r in t.story], "legs": legs},
                               ensure_ascii=False) + "\n")


# ================================================================================================================
# the report (easy English)
# ================================================================================================================

def report_md(out: dict) -> str:
    r, s = out["result"], out["settings"]
    inr = costs.inr
    ys = r["years"]
    n = r["runs"]
    L = [f"# L19a — the 2×2 test: his entries or random entries × his hedging or a plain stop",
         "",
         f"> Made by `python -m machine.tester.hedging_2x2` · {s['idea_id']} version {s['version_id']} (the version "
         f"L10 chose; readings {', '.join(f'{k}={v}' for k, v in s['readings'].items())}) · "
         f"{s['window']['first_bar'][:10]} to {s['window']['last_bar'][:10]} (MT5 server time) · 5-minute chart · "
         f"0.01 lot a leg · ₹40 a position, all-in · trial file `{out.get('trial', '(not written)')}`",
         "",
         "Every box trades the same months, the same directions and the same distances. Only ONE thing changes "
         "between two boxes side by side (never buy-and-hold, #114). Rupees are after ₹40 a position. A trade counts "
         "in the year it closed. Random = the middle of "
         f"{n} random runs (half did better, half worse).",
         "",
         "## The four boxes (₹ after cost)",
         "",
         "| box | " + " | ".join(ys) + " | all years | trades | open at end | worst year |",
         "|---|" + "---:|" * (len(ys) + 4)]
    for b in r["boxes"]:
        L.append(f"| {b['arm']} | " + " | ".join(inr(b["per_year"][y]) for y in ys) +
                 f" | **{inr(b['total'])}** | {b['trades']:g} | {b['open_at_end']:g} | {b['worst_year']} |")
    L += ["", "Random boxes: the middle run. The band that holds 90 of 100 random runs, all years: "
          f"with hedging {inr(r['boxes'][2]['total_5_95'][0])} to {inr(r['boxes'][2]['total_5_95'][1])} · "
          f"with a stop {inr(r['boxes'][3]['total_5_95'][0])} to {inr(r['boxes'][3]['total_5_95'][1])}.", ""]
    names = {"entry_with_hedging": "What HIS ENTRY adds, both with hedging",
             "entry_with_stop": "What HIS ENTRY adds, both with a plain stop",
             "hedging_for_his_entries": "What the HEDGING adds, for his entries",
             "hedging_for_random_entries": "What the HEDGING adds, for random entries (middle run)",
             "whole": "The WHOLE: his entries + hedging, against random entries + a stop"}
    L += ["## What each part adds (₹ after cost; + means it helped)", "",
          "| | " + " | ".join(ys) + " | all years | years it helped | worst year |",
          "|---|" + "---:|" * (len(ys) + 3)]
    for k, label in names.items():
        a = r["adds"][k]
        L.append(f"| {label} | " + " | ".join(inr(a["per_year"][y]) for y in ys) +
                 f" | **{inr(a['total'])}** | {a['years_positive']} of {len(ys)} | {a['worst_year']} |")
    hb = r["his_beats_random"]
    L += ["", "## His entries against random entries",
          "",
          f"- With hedging: his entries made more than **{hb['with_hedging']} of {n}** random runs (all years). "
          "By year: " + ", ".join(f"{y} {v}" for y, v in hb["with_hedging_by_year"].items()) + ".",
          f"- With a plain stop: more than **{hb['with_stop']} of {n}**. By year: " +
          ", ".join(f"{y} {v}" for y, v in hb["with_stop_by_year"].items()) + ".",
          f"- Hedging did better than a stop in **{r['hedging_beats_stop_in_random_runs']} of {n}** random runs.",
          ""]
    hh = out["his_hedging"]
    L += ["## His entries with his hedging, in detail", "",
          "```", hedging.summary_text(hh, f"{s['idea_id']} {s['version_id']}"), "```", "",
          "Every trade, one row each: `plan/work/L19a/idea-001-hedged-trades.csv`. What happened in each trade, step "
          "by step: `plan/work/L19a/idea-001-hedged-stories.jsonl`.", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=config.RANDOM_RUNS)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = run(n_runs=a.runs, write=not a.no_write, log=lambda m: print(m, flush=True))
    print(report_md(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
