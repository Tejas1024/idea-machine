"""His entry types through the tester, the hedging tester and the referee's 2x2 (L19b).

    .venv\\Scripts\\python -m machine.entries                          every entry type, every reading, 2016-2020
    .venv\\Scripts\\python -m machine.entries --only entry-zone-entry --runs 20 --no-write     a quick look, no files
    .venv\\Scripts\\python -m machine.entries --counts                 counts only (no money): the counts files
    .venv\\Scripts\\python -m machine.entries --reports                the reports again from the files (runs nothing)
The whole run takes about 70 minutes (each reading compiles under its own settings; 200 random runs of thousands of
signals): start it as its own process, not as a background job that is stopped after 30 minutes.

For each entry type (machine/entries/forms.py) and each reading of its open word (one version each - a try):
  1. compile it on the tune years 2016-2020 (the check years 2021-2023 are not opened: the old project already tested
     every one of these entries on 2016-2023 - plan/work/prior-tries.csv - so they cannot give a fresh answer either);
  2. THE TESTER: a plain stop at the hedge point, the same target (T12 M02's other way);
  3. THE HEDGING TESTER: his management (T12, version 1) - legs closed by ANY of his entries whose words are settled
     (M10, machine/entries/closers.py), the trendline hedge moving with its line (M27), the checks while an order
     waits (#98 B, #87);
  4. THE REFEREE'S 2x2 (L19a, #114 - never buy-and-hold): {his entries, random entries} x {his hedging, a plain stop};
     random entries = one twin per signal (the same month, direction, distances, cancel distance and waiting time; a
     random bar close in that month; a limit distance drawn from his own - machine/tester/hedging_2x2.twins), the
     SAME twins through both. A twin of a trendline entry gets a hedge moving the same way (a line through its own
     hedge, the same slope); the checks while an order waits are his entries' own rules and are not given to twins;
  5. the referee's answer by its fixed rules: every one of these is "already seen", so the best it can say is
     "chance - needs fresh years" (machine/referee/verdict.py) - said on every page, with the old project's rows;
     the weekly luck test and "without the best 3 trades" are printed as facts.
Writes ONE trial file per version (kind "hedging-2x2", counted as a try of that entry family), and
plan/work/L19b/<entry>.md (a report per entry type, easy English, with a picture of the four boxes),
plan/work/L19b/README.md (all of them), and per version in plan/work/L19b/<entry>/: <version>-counts.json (what the
compile found, no money - also for a version with no signal, which has no trial file), every trade (-*.csv) and
each hedged trade's story (.jsonl). The reports are made ONLY from the counts files and the trial files
(load_versions), so --reports makes them again without running anything.
Money: Rs after Rs 40 a position (0.01 lot, all-in, #54, #115); a trade counts in the year it closed; trades still open
at the window end are not counted (and said).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, costs, guards, settings, trials
from machine.data import bars as BARS
from machine.entries import closers, forms
from machine.ideas.compile import Compiled, Market, compile_idea
from machine.referee import prior, stats, verdict
from machine.referee import random_entries as RE
from machine.referee.replays import Segment
from machine.tester import engine, hedging
from machine.tester import hedging_2x2 as H2
from machine.tester.models import TF_MINUTES
from machine.tester.run import summarize as stop_summary

TFS = ("1m", "5m", "1h", "1d", "1w")            # the 1-minute chart: the zone break's lines inside the zone (#98 C)
OUT_DIR = config.WORK_DIR / "L19b"
KIND = "hedging-2x2"                            # the same trial kind as L19a's 2x2
COUNT_KEYS = ("signals", "buy", "sell", "signals_a_week", "weeks", "funnel", "dropped", "notes", "breaks_his_rules",
              "moves", "real_breaks", "real_breaks_a_week", "second_gap_orders")
# the comparison line: the two entries whose words are open, as closers at their loosest reading (the most setups)
ALL_FIVE = {forms.TRENDLINE_TOUCH.id: {"B2-01": "A"}, forms.TRIANGLE_FALSE_BREAK.id: {"B2-02": "A"}}


@dataclass
class Version:
    entry: forms.EntryType
    readings: dict
    version_id: str
    comp: Compiled | None = None                 # None when the version is read back from its files (--reports)
    out: dict = field(default_factory=dict)      # its counts (no money), then what the testers found

    @property
    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.readings.items()) or "as written"

    @property
    def means(self) -> str:
        if not self.readings:
            return "as written"
        idea = self.entry.idea()
        parts = []
        for u in idea.unclear:
            r = next(x for x in u.readings if x.id == self.readings.get(u.id))
            parts.append(f"{u.id} {r.id}: {r.means}")
        return "; ".join(parts)

    @property
    def plain(self) -> str:
        """The reading in his words, not a question code (his #113), for a picture or a table:
        "reading A: 'big move' = the day before it moved at least half a day's range"."""
        if not self.readings:
            return "as written"
        word = re.search(r"'([^']+)'", self.entry.open_word or "")
        parts = []
        for u in self.entry.idea().unclear:
            r = next(x for x in u.readings if x.id == self.readings.get(u.id))
            parts.append(f"reading {r.id}: '{word.group(1) if word else u.part}' = {r.means}")
        return "; ".join(parts)


def _key(rd: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(rd.items()))


# ================================================================================================================
# the run
# ================================================================================================================

def run(only: list[str] | None = None, *, start: str = "2016", end: str = "2020", n_runs: int = config.RANDOM_RUNS,
        write: bool = True, counts_only: bool = False, log=print) -> dict:
    if n_runs < 1 and not counts_only:
        raise ValueError("the 2x2 needs at least one random run")
    t0 = time.time()
    market = Market.load(start, end, tfs=TFS)
    m1 = market.bars("1m")
    book = engine.bars_book(m1, "5m")
    years = list(range(book.first.year, book.last.year + 1))
    fp0 = settings.fingerprint(settings.resolve({}))
    log(f"bars {book.first} .. {book.last} read in {time.time() - t0:.0f} s")
    chosen = [forms.get(x) for x in only] if only else list(forms.ENTRY_TYPES.values())

    # ---- compile: the chosen entry types, every reading; his closers and Type 1 (the opposite setups of #98 B)
    comps: dict[tuple[str, str], Compiled] = {}

    def comp(e: forms.EntryType, rd: dict) -> Compiled:
        k = (e.id, _key(rd))
        if k not in comps:
            t1 = time.time()
            comps[k] = compile_idea(e.idea(), market, readings=rd)
            c = comps[k]
            log(f"  {e.id} [{_key(rd) or 'as written'}] {c.version_id}: {len(c.signals)} signals "
                f"({c.counts()['buy']} buy / {c.counts()['sell']} sell), {time.time() - t1:.0f} s")
        return comps[k]

    type1 = comp(forms.TYPE1, {}).signals
    versions = []
    for e in chosen:
        for rd in e.readings():
            c = comp(e, rd)
            v = Version(e, rd, c.version_id, c)
            v.out = _counts(v, market, fp0, type1)
            if write:
                _write_counts(v)
            versions.append(v)
    if counts_only:
        log(f"counts done in {time.time() - t0:.0f} s")
        return {"versions": versions}
    got, _ = closers.closer_entries()
    closing = closers.closing_setups(market, book, signals={e.id: comp(e, rd).signals for e, rd in got})
    facts = hedging.MarketFacts.build(market, fp0, book, setups=closing.setups)
    facts.setup_signals, facts.closers = closing.signals, closing.which
    got5, _ = closers.closer_entries(ALL_FIVE)
    closing5 = closers.closing_setups(market, book, readings=ALL_FIVE,
                                      signals={e.id: comp(e, rd).signals for e, rd in got5})
    facts5 = hedging.MarketFacts(market, fp0, book, closing5.setups, facts.news, closing5.signals, closing5.which)
    log(f"his setups that close a leg: {len(facts.setups)} filled of {closing.signals} ({closing.which}); with the "
        f"two open ones at reading A: {len(facts5.setups)}; news moments {len(facts.news)}; {time.time() - t0:.0f} s")

    rules = hedging.Rules.now()
    for v in versions:
        t1 = time.time()
        v.out.update(_one(v, market, m1, book, facts, facts5, rules, type1, years, fp0, n_runs, log))
        log(f"  {v.entry.id} [{v.label}] done in {time.time() - t1:.0f} s")
    window = (str(book.first), str(book.last))
    if write:
        data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
                "gold_m1_content_sha256": BARS.m1_fingerprint()}
        for v in versions:
            if v.out.get("signals"):
                v.out["trial"] = _record(v, data, facts, rules, n_runs, (start, end), book).name
        write_reports(versions, facts.closers, window)
    else:                                               # a look only: the reports on the screen, no files
        by_entry: dict[str, list[Version]] = {}
        for v in versions:
            by_entry.setdefault(v.entry.id, []).append(v)
        log(overview_md(by_entry, facts.closers, window))
        for vs in by_entry.values():
            log(report_md(vs, facts.closers, window))
    log(f"all done in {time.time() - t0:.0f} s")
    return {"versions": versions, "facts": facts, "window": window}


def _moves(comp: Compiled, tf: str = "5m") -> dict[str, float]:
    """M27: signal id -> its line's slope in price per candle of the tester's chart."""
    return {sid: float(d["slope"]) * TF_MINUTES[tf] / TF_MINUTES[d["tf"]] for sid, d in comp.hedge_lines.items()}


def _one(v: Version, market, m1, book, facts, facts5, rules, type1, years, fp0, n_runs, log) -> dict:
    """What the testers found for one version (added to its counts): the 2x2, his boxes in detail, the checks."""
    sigs = v.comp.signals
    if not sigs:
        return {}
    second = set(v.out.get("second_gap_ids", ()))
    guards.check_opened(v.entry.id, [book.first, book.last], version=v.version_id)
    cancel = closers.fill_checks(v.entry, sigs, market, fp0, type1)
    moves = _moves(v.comp)
    his_s = engine.replay(sigs, None, "5m", config.TRADES_AT_ONCE, book=book, cancel_by=cancel)
    his_h = hedging.replay(sigs, facts, "5m", rules, moves=moves, cancel_by=cancel)
    his_h5 = hedging.replay(sigs, facts5, "5m", rules, moves=moves, cancel_by=cancel)
    seg = Segment(f"{years[0]}-{years[-1]}", tuple(years), "5m", sigs, m1, book, m1["close"].to_numpy(float))
    seed = RE.seed_of([seg])
    rand = {"hedge": [], "stop": []}
    t1 = time.time()
    for r in range(n_runs):
        tw = H2.twins(seg, sigs, np.random.default_rng([seed, r]), r)
        tmoves = {t.id: moves[sigs[int(t.id.split("-")[1])].id] for t in tw
                  if sigs[int(t.id.split("-")[1])].id in moves}
        rand["hedge"].append(H2._hedge_by_year(hedging.replay(tw, facts, "5m", rules, moves=tmoves), years))
        rand["stop"].append(H2._stop_by_year(engine.replay(tw, None, "5m", config.TRADES_AT_ONCE, book=book), years))
        if (r + 1) % 50 == 0:
            log(f"    random run {r + 1}/{n_runs} ({time.time() - t1:.0f} s)")
    result = H2._result(his_h, his_s, rand, years)
    extra = {}
    if v.entry is forms.SECOND_GAP:                 # how many of the trades each box took were second-gap orders
        extra["second_gap_trades"] = {"hedging": sum(t.signal_id in second for t in his_h.trades),
                                      "stop": sum(t.signal_id in second for t in his_s.trades)}
    return {**extra, "result": result, "seed": seed, "runs": n_runs,
            "his_hedging": hedging.summarize(his_h), "his_stop": stop_summary(his_s),
            "his_hedging_all_five": hedging.summarize(his_h5),
            "fill_checks": {"orders": len(cancel), "cancelled_plain": _cancelled(his_s.not_filled),
                            "cancelled_hedging": _cancelled(his_h.not_filled)},
            "referee": _referee(v.entry, years, his_s, his_h),
            "replays": {"stop": his_s, "hedge": his_h}}


def _cancelled(not_filled) -> int:
    return sum(why == engine.CHECK_BEFORE_FILL for _, why in not_filled)


def _counts_path(entry_id: str, version_id: str) -> Path:
    return OUT_DIR / entry_id / f"{version_id}-counts.json"


def _write_counts(v: Version) -> Path:
    """The version's counts (no money) next to its trades: what the reports need besides the trial file - and the
    only record of a version that made no signal (it has no trial file)."""
    p = _counts_path(v.entry.id, v.version_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"entry": v.entry.id, **v.out}, indent=1, ensure_ascii=False, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(p)
    return p


def _counts(v: Version, market, fp0, type1=None) -> dict:
    """What the compile found, a week (no money). type1: Type 1's signals - the second gap's own orders are the ones
    that are not among them."""
    c = v.comp
    first, last = c.window
    weeks = max((last - first) / pd.Timedelta(days=7), 1e-9)
    dropped: dict[tuple[str, str], int] = {}
    for (w, why), n in c.dropped.items():            # one line per kind of reason: a gap's own prices left out
        key = (w, re.sub(r"\s*\d+\.\d+-\d+\.\d+", "", why))
        dropped[key] = dropped.get(key, 0) + n
    out = {"signals": len(c.signals), "buy": c.counts()["buy"], "sell": c.counts()["sell"],
           "signals_a_week": round(len(c.signals) / weeks, 2), "weeks": round(weeks, 1),
           "funnel": {k: sum(x.values()) for k, x in c.funnel.items()},
           "dropped": sorted(([w, why, n] for (w, why), n in dropped.items()), key=lambda x: -x[2])[:8],
           "notes": list(c.notes), "breaks_his_rules": list(c.resolved.breaks_his_rules),
           "version_id": c.version_id, "readings": dict(v.readings), "means": v.means,
           "moves": len(c.hedge_lines)}
    if v.entry is forms.ZONE_BREAK:
        zb = market.zone_breaks(fp0)
        zb = zb[(zb["known_at"] >= first) & (zb["known_at"] <= last)]
        out["real_breaks"] = len(zb)
        out["real_breaks_a_week"] = round(len(zb) / weeks, 2)
    if v.entry is forms.SECOND_GAP and type1 is not None:   # the orders that are not Type 1's own (#124)
        k1 = {(s.direction, s.known_at, s.entry_price) for s in type1}
        ids = [s.id for s in c.signals if (s.direction, s.known_at, s.entry_price) not in k1]
        out["second_gap_orders"], out["second_gap_ids"] = len(ids), ids
    return out


def _referee(e: forms.EntryType, years, his_s, his_h) -> dict:
    """The referee's answer by its fixed rules (verdict.py) and the facts it would print: already seen -> at best
    'chance - needs fresh years'; the weekly luck test and 'without the best 3 trades', on both managements."""
    seen = prior.already_seen(e.seen, years)
    out = {"seen": seen.seen, "rows": [{"idea_key": r.idea_key, "what": prior.describe(r.idea_key), "tries": r.tries,
                                        "years": f"{r.years[0]}-{r.years[1]}"} for r in seen.rows],
           "old_tries": seen.tries,
           "answer": verdict.FRESH if seen.seen else "not judged here (not in the old list - a referee run first)",
           "why": ("the old project already looked at these years for this entry - only new prices can judge it "
                   "(the forward test, L17c)") if seen.seen else ""}
    for name, trades in (("stop", _stop_trades(his_s)), ("hedging", _hedge_trades(his_h))):
        if len(trades) >= 2:
            p = stats.bootstrap_p(trades["net_inr"], stats.week_ids(trades["exit_time"]))
        else:
            p = None
        out[name] = {"trades": int(len(trades)), "luck_test_p": None if p is None else round(p, 4),
                     "mean": round(float(trades["net_inr"].mean()), 2) if len(trades) else None,
                     "without_best_3": stats.without_best(trades) if len(trades) else None}
    return out


def _stop_trades(rep: engine.Replay) -> pd.DataFrame:
    rows = [(t.net_inr, pd.Timestamp(t.exit_ts)) for t in rep.trades if t.status == "closed"]
    return pd.DataFrame(rows, columns=["net_inr", "exit_time"])


def _hedge_trades(rep: hedging.HedgeReplay) -> pd.DataFrame:
    rows = [(t.net_inr, t.exit_ts) for t in rep.trades if t.status == "closed"]
    return pd.DataFrame(rows, columns=["net_inr", "exit_time"])


# ================================================================================================================
# the trial file, the trades
# ================================================================================================================

def _record(v: Version, data: dict, facts, rules, n_runs: int, window, book) -> Path:
    o = v.out
    settings_used = {
        "entry": v.entry.id, "title": v.entry.title, "version_id": v.version_id, "readings": v.readings,
        "means": v.means, "timeframe": v.comp.tf, "form_sha256": v.comp.form.fingerprint(),
        "window": {"asked": list(window), "first_bar": str(book.first), "last_bar": str(book.last)},
        "management_hedging": {"table": hedging.TABLE, "rules": rules.as_dict(),
                               "rows": {m: what for m, what, _ in hedging.RULES}},
        "management_stop": "the tester (L3): a stop at the hedge point, the same target (M02's other way); a "
                           "trendline entry's stop does not move (the tester has fixed stops)",
        "closing_setups": {"which": facts.closers, "signals": facts.setup_signals, "filled": len(facts.setups)},
        "fill_checks": list(v.entry.fill_checks), "fill_checks_orders": o["fill_checks"]["orders"],
        "hedge_moves_with_the_line": o["moves"],
        "random_entries": {"runs": n_runs, "seed": o["seed"], "keeps": "month, direction, entry type, hedge and "
                           "target distances, cancel distance, waiting time, lots, a trendline hedge's slope; the "
                           "SAME twins in both managements", "changes": "the time (a random bar close in the same "
                           "month) and a limit's distance (drawn from his own); no checks while the order waits"},
        "news": f"Forex Factory red + orange, exact times: {len(facts.news)} moments in the window",
        "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
        "already_seen": o["referee"]["rows"],
        "code_sha256": trials.code_sha256([Path(__file__), Path(closers.__file__), Path(forms.__file__),
                                           Path(hedging.__file__), Path(engine.__file__)]),
        "signals_sha256": trials.sha256_of([s.model_dump(mode="json") for s in v.comp.signals]),
    }
    result = {**o["result"], "his_hedging": o["his_hedging"], "his_stop": o["his_stop"],
              "his_hedging_all_five": o["his_hedging_all_five"], "referee": o["referee"],
              "counts": {k: o[k] for k in COUNT_KEYS if k in o}, "fill_checks": o["fill_checks"],
              **({"second_gap_trades": o["second_gap_trades"]} if "second_gap_trades" in o else {})}
    path = trials.record(kind=KIND, idea_id=v.entry.id, version_id=v.version_id, parent_idea=None,
                         settings=settings_used, data=data, result=result,
                         note=f"L19b: his entry type {v.entry.id} ({v.means}) - one version, a try of this entry "
                              "family; already seen by the old project on 2016-2023")
    folder = OUT_DIR / v.entry.id
    folder.mkdir(parents=True, exist_ok=True)
    tag = v.version_id
    hedging.trades_table(o["replays"]["hedge"].trades).to_csv(folder / f"{tag}-hedged-trades.csv", index=False)
    from machine.tester.run import trades_table
    trades_table(o["replays"]["stop"].trades).to_csv(folder / f"{tag}-plain-stop-trades.csv", index=False)
    with (folder / f"{tag}-hedged-stories.jsonl").open("w", encoding="utf-8") as f:
        for t in o["replays"]["hedge"].trades:
            legs = [{"direction": x.direction, "opened": _when(book, x.open_i), "open_px": x.open_px,
                     "why_open": x.why_open, "closed": _when(book, x.close_i), "close_px": x.close_px,
                     "why_closed": x.why_closed, "net_inr": None if x.pnl is None else round(x.pnl.net_inr, 2)}
                    for x in t.legs]
            f.write(json.dumps({"signal": t.signal_id, "status": t.status, "net_inr": t.net_inr,
                                "story": [{**r, "ts": str(r["ts"])} for r in t.story], "legs": legs},
                               ensure_ascii=False) + "\n")
    return path


def _when(book, i):
    return None if i is None else str(pd.Timestamp(int(book.t[i])))


# ================================================================================================================
# the picture: the four boxes, all years (his picture rules: one chart, every named thing drawn, 5 seconds)
# ================================================================================================================

BOX_WORDS = ("his entries + his hedging", "his entries + a plain stop", "random entries + his hedging",
             "random entries + a plain stop")


def picture(v: Version, path: Path) -> dict:
    """Four bars: the 2x2's rupees after cost over all the years (random = the middle run). Returns what was drawn."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from machine.charts import draw as D
    from machine.report.build import check_words
    plt.rcParams["text.parse_math"] = False            # lesson B8: "$5" is five dollars, not maths
    r = v.out["result"]
    vals = [float(b["total"]) for b in r["boxes"]]
    trades = [b["trades"] for b in r["boxes"]]
    names = [f"{w}\n({t:g} trades{', middle of ' + str(r['runs']) + ' runs' if k >= 2 else ''})"
             for k, (w, t) in enumerate(zip(BOX_WORDS, trades))]
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.6), dpi=D.DPI)
    y = list(range(len(vals)))[::-1]
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    span = (hi - lo) or 1.0
    ax.barh(y, vals, height=0.58, color=[D.GREEN if x > 0 else D.RED for x in vals], zorder=2)
    ax.axvline(0, color=D.INK, linewidth=1.0, zorder=3)
    texts = []
    for yy, x in zip(y, vals):
        t = inr(x)
        ax.text(x + (0.02 * span if x >= 0 else -0.02 * span), yy, t, va="center", ha="left" if x >= 0 else "right",
                fontsize=D.F_NAME, color=D.INK, fontweight="bold")
        texts.append(t)
    ax.set_yticks(y, names, fontsize=D.F_AXIS, color=D.INK)
    texts += names
    ax.set_xlim(lo - 0.28 * span, hi + 0.28 * span)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: inr(x)))
    ax.tick_params(axis="x", labelsize=D.F_AXIS, colors=D.MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=D.GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    title = v.entry.title
    ys = r["years"][0] if r["years"][0] == r["years"][-1] else f"{r['years'][0]}–{r['years'][-1]}"
    sub = f"money after costs, {ys} · 0.01 lot · ₹40 a position, all-in"
    sub2 = "already seen by the old project: the answer can only be 'chance — needs fresh years'"
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=D.F_SUB + 2, color=D.INK, fontweight="bold")
    y0 = 0.9
    if v.readings:                                     # which reading of his open word, in words (his #113)
        fig.text(0.02, y0, v.plain, ha="left", fontsize=D.F_AXIS + 1, color=D.INK)
        texts.append(v.plain)
        y0 -= 0.045
    fig.text(0.02, y0, sub, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    fig.text(0.02, y0 - 0.045, sub2, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    texts += [title, sub, sub2]
    fig.subplots_adjust(left=0.33, right=0.97, top=y0 - 0.1, bottom=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", dpi=D.DPI)
    plt.close(fig)
    buf.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.open(buf).convert("RGB").save(tmp, format="JPEG", quality=92, optimize=True)
    tmp.replace(path)
    problems = [p for t in texts for p in check_words(t)]
    if problems:
        raise ValueError("the picture's words: " + "; ".join(problems))
    return {"path": path, "texts": texts}


# ================================================================================================================
# the reports (easy English - CLAUDE.md #113; check_words is his rule as a test)
# ================================================================================================================

def write_reports(versions: list[Version], closers_used: dict, window: tuple[str, str]) -> list[Path]:
    """The pictures, a report per entry type and the overview. closers_used: which of his entries closed legs (M10);
    window: the first and last 1-minute bar."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_entry: dict[str, list[Version]] = {}
    for v in versions:
        by_entry.setdefault(v.entry.id, []).append(v)
    paths = []
    for v in versions:
        if v.out.get("result"):
            p = OUT_DIR / v.entry.id / f"{v.version_id}-boxes.jpg"
            picture(v, p)
            v.out["picture"] = p.relative_to(OUT_DIR).as_posix()
            paths.append(p)
    for eid, vs in by_entry.items():
        p = OUT_DIR / f"{eid}.md"
        p.write_text(report_md(vs, closers_used, window), encoding="utf-8")
        paths.append(p)
    p = OUT_DIR / "README.md"
    p.write_text(overview_md(by_entry, closers_used, window), encoding="utf-8")
    paths.append(p)
    return paths


def load_versions(only: list[str] | None = None,
                  folder: Path | None = None) -> tuple[list[Version], dict, tuple[str, str]]:
    """Every version read back from its files - its counts file (plan/work/L19b/<entry>/<version>-counts.json) and
    its trial file (kind "hedging-2x2", the latest of that entry and version) - so the reports can be made again
    without running anything: python -m machine.entries --reports. Returns (versions, closers used, window)."""
    files = [f for f in trials.read_all(folder) if f.get("kind") == KIND]
    versions, closers_used, window = [], {}, None
    for e in ([forms.get(x) for x in only] if only else forms.ENTRY_TYPES.values()):
        counts = {}
        for p in sorted((OUT_DIR / e.id).glob("*-counts.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            counts[_key(d.get("readings", {}))] = d
        for rd in e.readings():
            d = counts.get(_key(rd))
            if d is None:
                raise FileNotFoundError(f"{e.id} [{_key(rd) or 'as written'}] has no counts file in {OUT_DIR / e.id}: "
                                        "python -m machine.entries --counts writes them (no money)")
            v = Version(e, rd, d["version_id"], None, dict(d))
            mine = [f for f in files if f["idea_id"] == e.id and f["version_id"] == v.version_id]
            if mine:
                t = mine[-1]                                   # read_all: oldest first
                r = t["result"]
                v.out.update({k: r[k] for k in ("his_hedging", "his_stop", "his_hedging_all_five", "referee",
                                                "fill_checks", "second_gap_trades") if k in r})
                v.out["result"] = {k: r[k] for k in ("years", "boxes", "adds", "runs", "his_beats_random",
                                                     "hedging_beats_stop_in_random_runs")}
                v.out["runs"] = r["runs"]
                v.out["trial"] = next(Path(config.TRIALS_DIR if folder is None else folder)
                                      .glob(f"{t['trial_id']}-*.json")).name
                if "second_gap_trades" not in v.out and "second_gap_ids" in v.out:
                    v.out["second_gap_trades"] = _second_gap_trades(v)
                closers_used = t["settings"]["closing_setups"]["which"]
                window = (t["settings"]["window"]["first_bar"], t["settings"]["window"]["last_bar"])
                pic = OUT_DIR / e.id / f"{v.version_id}-boxes.jpg"
                if pic.is_file():
                    v.out["picture"] = pic.relative_to(OUT_DIR).as_posix()
            versions.append(v)
    if window is None:
        raise FileNotFoundError("no trial file of kind 'hedging-2x2' for any entry type: run python -m machine.entries")
    return versions, closers_used, window


def _second_gap_trades(v: Version) -> dict:
    """From the trades this version's run saved: how many came from a second-gap order."""
    ids = set(v.out["second_gap_ids"])
    folder = OUT_DIR / v.entry.id
    out = {}
    for name, file in (("hedging", "hedged-trades"), ("stop", "plain-stop-trades")):
        t = pd.read_csv(folder / f"{v.version_id}-{file}.csv")
        out[name] = int(t["signal"].isin(ids).sum())
    return out


inr = costs.inr


def _n(k, word: str) -> str:
    """'1 trade', '15 trades' ('1 try', '9 tries')."""
    k = int(k)
    if word == "try":
        return f"{k} try" if k == 1 else f"{k} tries"
    return f"{k} {word}" + ("" if k == 1 else "s")


def _window(window) -> str:
    first, last = window
    return f"{pd.Timestamp(first):%d %b %Y} to {pd.Timestamp(last):%d %b %Y}"


def _seen_text(e: forms.EntryType, ref: dict) -> str:
    """Its own rows in the old list by name, the rest of the matching rows as one count."""
    own = [r for r in ref["rows"] if r["idea_key"] in e.seen_rows]
    rest = [r for r in ref["rows"] if r["idea_key"] not in e.seen_rows]
    rows = "; ".join(f"{r['what'] or r['idea_key']} ({_n(r['tries'], 'try')}, {r['years']})" for r in own)
    more = (f"; and {len(rest)} more rows of the old list on the same entry ({sum(r['tries'] for r in rest)} tries, "
            "some on 2016–2026 — all named in the trial file)") if rest else ""
    return (f"**Already seen.** The old project already tested this entry on these years: {rows}{more}. So "
            "2016–2023 is not fresh for it, and the referee's answer can only be **\"chance — needs fresh years\"**, "
            "whatever the rupees below say. Only new prices can judge it (the forward test, L17c). The years "
            "2021–2023 were not opened: the old project saw them too.")


def report_md(vs: list[Version], closers_used: dict, window: tuple[str, str]) -> str:
    e = vs[0].entry
    idea = e.idea()
    ref = next((v.out["referee"] for v in vs if v.out.get("referee")), None)
    L = [f"# {e.title}", "",
         f"> Made by `python -m machine.entries` (L19b) · {_window(window)} (MT5 server time) · 5-minute chart · 0.01 "
         "lot a leg · ₹40 a position, all-in · a trade counts in the year it closed · trial files: " +
         ", ".join(f"`{v.out['trial']}`" for v in vs if v.out.get("trial")), ""]
    if ref is not None:
        L += [_seen_text(e, ref), ""]
    L += ["## What it is", "", f"His words: *\"{idea.words}\"*", ""]
    for s in idea.steps:
        L.append(f"- **{s.id}**: {s.what}")
    w = idea.entry.max_wait_bars
    wait = f"for {w} five-minute candles" if isinstance(w, int) else "with no time limit"
    L.append(f"- **entry**: a limit at `{idea.entry.at}` (it waits {wait})")
    L.append(f"- **hedge point**: `{idea.exit.at}`")
    L.append("- **target**: " + ", ".join(f"`{t.at}`" if t.at else "the next liquidity" +
                                          (f" ({', '.join(t.kinds)})" if t.kinds else "") for t in idea.targets))
    for r in idea.do_not_enter:
        L.append(f"- **no trade when**: `{r.name}`" + (f" — {r.sources[0].ref}" if r.sources else ""))
    for r in idea.filters:
        L.append(f"- **only when**: `{r.name}` {dict(r.params)}" + (f" — {r.sources[0].ref}" if r.sources else ""))
    if e.fill_checks:
        L.append("- **while the order waits**: " + "; ".join(_CHECK_WORDS[c] for c in e.fill_checks))
    if e.his_count:
        L.append(f"- **how often he sees it**: {e.his_count}")
    L.append("")
    if e.open_word:
        u = idea.unclear[0]
        L += [f"**His open word: {e.open_word}.** Every reading is tested as its own version (a try). The answer waits "
              "for him (the weekly picture batch) — the machine never picks one.", "",
              "| reading | means | signals | a week |", "|---|---|---:|---:|"]
        for v in vs:
            rid = v.readings.get(u.id, "-")
            L.append(f"| {rid} | {next((x.means for x in u.readings if x.id == rid), 'as written')} | "
                     f"{v.out['signals']} | {v.out['signals_a_week']} |")
        L.append("")
    for v in vs:
        L += _version_md(v, len(vs) > 1)
    L += _guard_md(vs, ref)
    return "\n".join(L) + "\n"


_CHECK_WORDS = {
    "opposite_setup": "an opposite setup of his (grab + gap) forming before the fill cancels the order — #98: "
                      "\"it might be the sell trade, so it's not a good idea to enter at the retracement\"",
    "gap_excuse": "if it had an unfilled gap past its hedge, it was taken only because his liquidity waited on the "
                  "target side; when price takes all of that liquidity before the fill, the order is cancelled (#87)",
}


def _version_md(v: Version, several: bool) -> list[str]:
    o = v.out
    head = f"## {'Reading ' + v.label + ' — ' + v.means if several else 'Result'}"
    L = [head, ""]
    L.append(f"Signals 2016–2020: **{o['signals']}** ({o['buy']} buy, {o['sell']} sell) = {o['signals_a_week']} a "
             "week." + (f" Real zone breaks the machine sees: {o['real_breaks']} = {o['real_breaks_a_week']} a week "
                        "(before his hour and day rule and the target rule)." if "real_breaks" in o else ""))
    if "second_gap_orders" in o:
        L.append(f"Orders at the SECOND gap: **{o['second_gap_orders']}**. Every other signal is Type 1's own order at "
                 "the first gap (the same as entry-type1): the order waits at the first gap and moves to the second "
                 "only when the second is known, price has not come back to the first, and the trend is very strong "
                 "at that moment.")
    if o["breaks_his_rules"]:
        L.append("The machine's settings file lists his DEFAULT here, so it marks this version as a change from it: "
                 + " · ".join(o["breaks_his_rules"]) + (" The second gap is his own exception (#124); only his words "
                                                        "'very strong' are open." if "second_gap_orders" in o else ""))
    if o["dropped"]:
        L.append("Where setups stopped: " + " · ".join(f"{why} ({n:,})" for _, why, n in o["dropped"][:4]) + ".")
    if not o.get("result"):
        L += ["", "**No signal — nothing to test.**", ""]
        return L
    r, hh = o["result"], o["his_hedging"]
    ys = r["years"]
    L += ["", "### The four boxes (₹ after cost)", "",
          "| box | " + " | ".join(ys) + " | all years | trades | open at end | worst year |",
          "|---|" + "---:|" * (len(ys) + 4)]
    for b in r["boxes"]:
        L.append(f"| {b['arm']} | " + " | ".join(inr(b["per_year"][y]) for y in ys) +
                 f" | **{inr(b['total'])}** | {b['trades']:g} | {b['open_at_end']:g} | {b['worst_year']} |")
    n = r["runs"]
    if o.get("picture"):
        L += ["", f"![the four boxes, all years]({o['picture']})"]
    L += ["", f"Random boxes: the middle of {n} random runs (half did better, half worse). The band that holds 90 of "
          f"100 runs, all years: with hedging {inr(r['boxes'][2]['total_5_95'][0])} to "
          f"{inr(r['boxes'][2]['total_5_95'][1])} · with a stop {inr(r['boxes'][3]['total_5_95'][0])} to "
          f"{inr(r['boxes'][3]['total_5_95'][1])}.", "",
          "### What each part adds (₹ after cost; + means it helped)", "",
          "| | " + " | ".join(ys) + " | all years | years it helped | worst year |",
          "|---|" + "---:|" * (len(ys) + 3)]
    names = {"entry_with_hedging": "What HIS ENTRY adds, both with hedging",
             "entry_with_stop": "What HIS ENTRY adds, both with a plain stop",
             "hedging_for_his_entries": "What the HEDGING adds, for his entries",
             "hedging_for_random_entries": "What the HEDGING adds, for random entries (middle run)",
             "whole": "The WHOLE: his entries + hedging, against random entries + a stop"}
    for k, label in names.items():
        a = r["adds"][k]
        L.append(f"| {label} | " + " | ".join(inr(a["per_year"][y]) for y in ys) +
                 f" | **{inr(a['total'])}** | {a['years_positive']} of {len(ys)} | {a['worst_year']} |")
    hb = r["his_beats_random"]
    L += ["", f"- With his hedging, his entries made more than **{hb['with_hedging']} of {n}** random runs (all years). "
          "By year: " + ", ".join(f"{y} {x}" for y, x in hb["with_hedging_by_year"].items()) + ".",
          f"- With a plain stop: more than **{hb['with_stop']} of {n}**. By year: " +
          ", ".join(f"{y} {x}" for y, x in hb["with_stop_by_year"].items()) + ".",
          f"- His hedging did better than a stop in **{r['hedging_beats_stop_in_random_runs']} of {n}** random runs.",
          "", "### His entries with his hedging", "",
          f"{_n(hh['trades'], 'trade')} closed, {hh['open_at_end']} still open at the end (not counted) · "
          f"{hh['positions']} positions (₹{config.COST_INR_PER_POSITION} each) · before cost "
          f"{inr(hh['gross_inr_total'])}, cost {inr(hh['cost_inr_total'])} · hedged {hh['hedged_trades']} of "
          f"{hh['trades']} · days open: middle {hh['days_open_median']}, longest {hh['days_open_max']}.",
          f"With the trendline touch and the triangle false break also closing legs (at their loosest reading A — "
          f"their words are open): {inr(o['his_hedging_all_five']['net_inr_total'])} after cost "
          f"({_n(o['his_hedging_all_five']['trades'], 'trade')}), against {inr(hh['net_inr_total'])}.",
          ""]
    if "second_gap_trades" in o:
        g = o["second_gap_trades"]
        L.append(f"Trades that came from a second-gap order: {g['hedging']} in the hedging box, {g['stop']} in the "
                 "plain-stop box (every other trade is exactly one of Type 1's).")
    if o["moves"]:
        L.append(f"The hedge moved with the trendline in {hh.get('line_trades', 0)} of the hedged-box trades (M27).")
    fc = o["fill_checks"]
    if fc["orders"]:
        L.append(f"The check while the order waits cancelled {fc['cancelled_plain']} of his orders (plain-stop box) "
                 f"and {fc['cancelled_hedging']} (hedging box). {_n(fc['orders'], 'order')} had such a moment ahead; "
                 "the others filled first, or another rule cancelled them first (price reaching the nearest target "
                 "first cancels an order too, #119).")
    L.append(f"Every trade: `plan/work/L19b/{v.entry.id}/{v.version_id}-hedged-trades.csv` and "
             f"`-plain-stop-trades.csv`; each hedged trade step by step: `-hedged-stories.jsonl`.")
    L.append("")
    return L


def _guard_md(vs: list[Version], ref: dict | None) -> list[str]:
    """His judgement guard (#114), from the numbers above."""
    L = ["## The judgement guard (#114)", ""]
    done = [v for v in vs if v.out.get("result")]
    if not done:
        return L + ["Nothing traded - nothing to judge.", ""]
    n_tries = len(vs)
    for v in done:
        o, rf = v.out, v.out["referee"]
        hh, hs = o["his_hedging"], o["his_stop"]
        lab = f"Reading {v.label}: " if len(vs) > 1 else ""
        wb_h, wb_s = rf["hedging"]["without_best_3"], rf["stop"]["without_best_3"]
        same = (f"- {lab}**Same risk?** Each box above trades the same months, directions and distances; only one "
                "thing changes between two boxes side by side.")
        if hs["trades"] > 2 * max(hh["trades"], 1):
            same += (" But the two managements do not take the same NUMBER of trades: with his hedging "
                     f"{_n(hh['trades'], 'trade')}, with a plain stop {_n(hs['trades'], 'trade')} (one trade at a "
                     "time" + (f", and a hedged trade stays open a middle of {hh['days_open_median']} days"
                               if hh["trades"] else "") + "). So \"what the hedging adds\" is partly \"far fewer "
                     "trades, far less cost\" — the gap between the machine and his 4–5 trades a week (#126).")
        L.append(same)
        L.append(f"- {lab}**Bad years:** with hedging the worst year is {hh['worst_year']['year']} "
                 f"({inr(hh['worst_year']['net_inr'])}); with a stop {hs['worst_year']['year']} "
                 f"({inr(hs['worst_year']['net_inr'])}).")
        for name, wb, box in (("hedging", wb_h, hh), ("a plain stop", wb_s, hs)):
            if not wb:
                continue
            if box["trades"] <= config.WITHOUT_BEST:
                L.append(f"- {lab}**How much is 3 trades?** With {name}: all of it — only "
                         f"{_n(box['trades'], 'trade')} ({inr(box['net_inr_total'])}).")
            else:
                L.append(f"- {lab}**How much is 3 trades?** With {name}, the best 3 trades made "
                         f"{inr(sum(wb['best_inr']))}; without them the other {wb['trades']} made {inr(wb['net_inr'])}.")
        for name, box in (("with a plain stop", rf["stop"]), ("with his hedging", rf["hedging"])):
            if box["trades"] < config.MIN_TRADES:
                L.append(f"- {lab}**Luck test** ({name}): too few trades to ask ({box['trades']}; the referee needs "
                         f"{config.MIN_TRADES}).")
            elif box["luck_test_p"] is not None and (box["mean"] or 0) <= 0:
                L.append(f"- {lab}**Luck test** ({name}): not needed — it lost {inr(-box['mean'])} a trade; the test "
                         "only asks whether a PROFIT could be luck.")
            elif box["luck_test_p"] is not None:
                L.append(f"- {lab}**Luck test** ({name}): a world where this entry has no edge makes this much a trade "
                         f"{100 * box['luck_test_p']:.0f} times in 100 ({box['trades']} trades; under 5 would count).")
    L.append(f"- **Tries:** {n_tries} version{'s' if n_tries > 1 else ''} of this entry here" +
             (" (one per reading of his open word)" if n_tries > 1 else "") +
             ", plus the old project's tries on the same years" + (f": {ref['old_tries']}." if ref else "."))
    L.append("- **What would make it wrong:** a different reading of his open words, his other management readings "
             "(L19c), the zones or lines drawn a little differently, and above all new prices — these years were "
             "already seen.")
    L.append("")
    return L


def overview_md(by_entry: dict[str, list[Version]], closers_used: dict, window: tuple[str, str]) -> str:
    L = ["# L19b — his entry types, each as a form, through the tester, the hedging tester and the 2×2", "",
         f"> {_window(window)} (MT5 server time) · 5-minute chart · 0.01 lot a leg · ₹40 a position, all-in · "
         "made by `python -m machine.entries` · one report per entry type below.", "",
         "**Every one of these entries is already seen:** the old project tested each on 2016–2023 "
         "(plan/work/prior-tries.csv). So none can be \"works\" here; the referee's answer is at best **\"chance — "
         "needs fresh years\"**. Only new prices can judge them (L17c).", "",
         "His setups that close a hedged leg (M10): " +
         "; ".join(f"{forms.get(k).title.split(':')[0]} ({v} signals)" if isinstance(v, int) else
                   f"{forms.get(k).title.split(':')[0]} — {v}" for k, v in closers_used.items()) + ".", "",
         "| entry type | reading | signals | a week | his entries + hedging | his entries + a stop | random + hedging "
         "(middle) | random + a stop (middle) | his beat random (hedging / stop, of runs) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for eid, vs in by_entry.items():
        for v in vs:
            o = v.out
            name = f"[{v.entry.title}]({eid}.md)"
            if not o.get("result"):
                L.append(f"| {name} | {v.plain} | {o['signals']} | {o['signals_a_week']} | — | — | — | — | no signal |")
                continue
            b = {x["arm"]: x for x in o["result"]["boxes"]}
            hb = o["result"]["his_beats_random"]
            L.append(f"| {name} | {v.plain} | {o['signals']} | {o['signals_a_week']} | "
                     f"{inr(b[H2.ARMS[0]]['total'])} ({b[H2.ARMS[0]]['trades']:g}) | "
                     f"{inr(b[H2.ARMS[1]]['total'])} ({b[H2.ARMS[1]]['trades']:g}) | {inr(b[H2.ARMS[2]]['total'])} | "
                     f"{inr(b[H2.ARMS[3]]['total'])} | {hb['with_hedging']} / {hb['with_stop']} of "
                     f"{o['result']['runs']} |")
    L += ["", "The number in brackets: how many trades closed. Random = the middle one of the random runs.", ""]
    return "\n".join(L) + "\n"


# ================================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.entries", description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="entry ids, comma separated (default: all six)")
    ap.add_argument("--runs", type=int, default=config.RANDOM_RUNS)
    ap.add_argument("--start", default="2016")
    ap.add_argument("--end", default="2020")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--counts", action="store_true",
                    help="counts only: compile, no money; writes <entry>/<version>-counts.json")
    ap.add_argument("--reports", action="store_true",
                    help="make the reports and pictures again from the counts files and the trial files: runs nothing")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    only = [x.strip() for x in a.only.split(",")] if a.only else None
    if a.reports:
        versions, closers_used, window = load_versions(only)
        for p in write_reports(versions, closers_used, window):
            print(p.relative_to(config.ROOT).as_posix())
        return 0
    out = run(only, start=a.start, end=a.end, n_runs=a.runs, write=not a.no_write, counts_only=a.counts,
              log=lambda m: print(m, flush=True))
    if a.counts:
        for v in out["versions"]:
            print(json.dumps({k: v.out[k] for k in ("version_id", "readings", "signals", "signals_a_week", "dropped")},
                             ensure_ascii=False, default=str))
    return 0
