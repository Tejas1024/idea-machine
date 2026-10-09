"""An idea parked "for his hedge" through his hedge (L19d): the 2x2 of its chosen version on 2016-2020.

    .venv\\Scripts\\python -m machine.strategy parked idea-002

L10b's choice for an idea that lost with a plain stop (machine/intake/state.py choice, LEARNED #162): "parked, not
dropped ... When your hedge (instead of a stop) runs through the machine (L19c / L19d), its trades are tested with your
hedge too." This is that test:
  - the version the referee chose for the idea (its tune file) - the SAME version, so not a new try; compiled again on
    2016-2020, its setups must be exactly the tune's;
  - {his entries, random entries} x {his hedging, a plain stop} (L19a's 2x2: the same twins in both managements; his
    management version 1, legs closed by his settled entries, M10);
  - only 2016-2020, the years it was chosen on: its check years were used by the plain-stop judging (once per family),
    so they are not opened again for it;
  - ONE trial file (kind "hedging-2x2"), a page plan/work/L19d/<idea>-with-his-hedge.md (+ its picture), and a section
    added to the idea's decision.md: the machine's choice now that his hedge was tested (it tells, never asks).
The intake (machine/intake/state.py) runs this by itself for an idea it parked. A test that ran is never run again:
its trial file is read back (a crash after the file, or a session that ran it by hand first).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from machine import config, costs, settings, trials
from machine.ideas.compile import Market, compile_idea
from machine.ideas.form import Idea
from machine.referee import random_entries as RE
from machine.referee import tune
from machine.referee.replays import Segment
from machine.tester import engine, hedging
from machine.tester import hedging_2x2 as H2
from machine.tester.models import TF_MINUTES
from machine.tester.run import summarize as stop_summary
from machine.strategy import whole as W

KIND = "hedging-2x2"
inr = costs.inr


class ParkedError(ValueError):
    """The idea cannot be tested this way (not decided, no chosen version, its setups changed)."""


def choice(result: dict, hedged: dict) -> tuple[str, str]:
    """The machine's choice once his hedge was tested (Claude's reading of L10b's rule; README section 7): only an
    idea that made money WITH his hedge AND beat random entries with his hedge (95 of 100, the referee's line) can go
    on - and only new days can judge it now (its check years are used). Everything else stays kept on file."""
    runs = result["runs"]
    beat = result["his_beats_random"]["with_hedging"]
    if hedged["net_inr_total"] > 0 and beat >= 0.95 * runs:
        return ("FORWARD TEST", "with your hedge it made money and beat random entries with your hedge - but its "
                "check years are used, so only new days can judge it (a forward test, L17c, no money).")
    why = ("it lost with your hedge too" if hedged["net_inr_total"] <= 0 else
           f"with your hedge it made money, but random entries with your hedge did as well (it beat {beat} of {runs})")
    return ("KEPT ON FILE", f"{why}. It is kept, not dropped; nothing more runs for it.")


def run(idea: str, *, n_runs: int = config.RANDOM_RUNS, write: bool = True, log=print) -> dict:
    from machine.ideas import store
    state = store.read_state(idea)["state"]
    if state != "DECIDED":
        raise ParkedError(f"{idea} is {state}: only a decided idea is tested with his hedge here")
    t = tune.latest(idea)
    if t is None:
        raise ParkedError(f"{idea} has no tune file: no chosen version")
    vid = t["result"]["chosen"]
    row = t["result"]["versions"][vid]
    f = done(idea, vid)
    if write and f is not None:
        out = _from_file(idea, f)
        log(f"{idea} {vid}: tested with his hedge already ({out['trial']}) - not run again")
        page = _page_path(idea)
        out["page"] = page if page.is_file() else write_page(out)
        out["decision"] = add_to_decision(idea, out)
        return out
    form = Idea.load(config.IDEAS_DIR / idea / "form.json")
    if t.get("parent_idea") and not form.parent_idea:           # the family the research tools found (section D,
        form = form.model_copy(update={"parent_idea": t["parent_idea"]})   # L14.1): the tune's signals carry it
    t0 = time.time()
    years = config.TUNE_YEARS
    market = Market.load(str(years[0]), str(years[-1]), tfs=W.TFS)
    comp = compile_idea(form, market, readings=row.get("readings") or None, changes=row.get("changes") or None)
    if comp.version_id != vid:
        raise ParkedError(f"{idea}'s chosen readings now compile to {comp.version_id}, not {vid}")
    if row.get("signals_sha256") and tune.signals_sha256(comp.signals) != row["signals_sha256"]:
        raise ParkedError(f"{idea} {vid}: its setups on 2016-2020 are not the tune's any more - stop and find why")
    m1 = market.bars("1m")
    book = engine.bars_book(m1, comp.tf)
    fp0 = settings.fingerprint(settings.resolve({}))
    facts = hedging.MarketFacts.build(market, fp0, book)
    rules = hedging.Rules.now()
    log(f"{idea} {vid}: {len(comp.signals)} setups (the same as its tune); his setups that close a leg: "
        f"{len(facts.setups)} ({time.time() - t0:.0f} s)")
    moves = {sid: float(d["slope"]) * TF_MINUTES[comp.tf] / TF_MINUTES[d["tf"]]       # M27, as the research tools
             for sid, d in comp.hedge_lines.items()}
    his_h = hedging.replay(comp.signals, facts, comp.tf, rules, moves=moves)
    his_s = engine.replay(comp.signals, None, comp.tf, config.TRADES_AT_ONCE, book=book)
    ys = list(range(book.first.year, book.last.year + 1))
    seg = Segment(f"{years[0]}-{years[-1]}", tuple(ys), comp.tf, comp.signals, m1, book, m1["close"].to_numpy(float))
    seed = RE.seed_of([seg])
    rand = {"hedge": [], "stop": []}
    t1 = time.time()
    for r in range(n_runs):
        tw = H2.twins(seg, comp.signals, np.random.default_rng([seed, r]), r)
        rand["hedge"].append(H2._hedge_by_year(hedging.replay(tw, facts, comp.tf, rules), ys))
        rand["stop"].append(H2._stop_by_year(engine.replay(tw, None, comp.tf, config.TRADES_AT_ONCE, book=book), ys))
        if (r + 1) % 25 == 0:
            log(f"  random run {r + 1}/{n_runs} ({time.time() - t1:.0f} s)")
    result = H2._result(his_h, his_s, rand, ys)
    result["run_totals"] = {box: [round(float(sum(d.values())), 2) for d, _, _ in rand[box]] for box in rand}
    hh, hs = hedging.summarize(his_h), stop_summary(his_s)
    hh["trade_net_inr"] = [round(t.net_inr, 2) for t in his_h.trades if t.status == "closed"]   # L17c's forward plan
    head, means = choice(result, hh)                                                             # counts in them
    out = {"idea": idea, "version": vid, "about": row.get("about", ""), "result": result, "his_hedging": hh,
           "his_stop": hs, "choice": head, "means": means, "seconds": round(time.time() - t0)}
    if write:
        sett = {"idea_id": idea, "version_id": vid, "chosen_by": t["trial_id"], "readings": row.get("readings"),
                "changes": row.get("changes"), "timeframe": comp.tf,
                "window": {"years": f"{years[0]}-{years[-1]}", "first_bar": str(book.first), "last_bar": str(book.last)},
                "why": "L10b parked it 'for his hedge'; L19d: its chosen version through his hedge (the same version - "
                       "not a new try); 2016-2020 only (its check years were used once, with a plain stop)",
                "management_hedging": {"table": hedging.TABLE, "rules": rules.as_dict()},
                "management_stop": "the tester (L3): a stop at the hedge point, the same target",
                "random_entries": {"runs": n_runs, "seed": seed},
                "closing_setups": {"which": facts.closers, "signals": facts.setup_signals, "filled": len(facts.setups)},
                "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
                "code_sha256": trials.code_sha256([Path(__file__), Path(hedging.__file__), Path(engine.__file__),
                                                   Path(H2.__file__)]),
                "signals_sha256": tune.signals_sha256(comp.signals)}
        from machine.data import bars as BARS
        data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
                "gold_m1_content_sha256": BARS.m1_fingerprint()}
        path = trials.record(kind=KIND, idea_id=idea, version_id=vid, parent_idea=t.get("parent_idea"),
                             settings=sett, data=data,
                             result={**result, "his_hedging": hh, "his_stop": hs, "choice": head},
                             note=f"L19d: {idea} (parked for his hedge) - its chosen version through his hedge, the 2x2 "
                                  "on 2016-2020; the same version, not a new try")
        out["trial"] = path.name
        out["page"] = write_page(out)
        out["decision"] = add_to_decision(idea, out)
    return out


def write_page(out: dict) -> Path:
    from machine.strategy import report as RP
    r, hh = out["result"], out["his_hedging"]
    ys = r["years"]
    m = {"years": ys, "runs": r["runs"], "boxes": [], "windows": 1, "beats_by_year": r["his_beats_random"]["with_hedging_by_year"],
         "his_beats_random_hedging": r["his_beats_random"]["with_hedging"],
         "his_beats_random_stop": r["his_beats_random"]["with_stop"],
         "hedging_beats_stop_in_random_runs": r["hedging_beats_stop_in_random_runs"]}
    for b, words in zip(r["boxes"], RP.BOX_WORDS):
        m["boxes"].append({"words": words, "per_year": b["per_year"], "trades": b["trades"],
                           "open_at_end": b["open_at_end"], "random": "median_of_runs" in b, "total": b["total"]})
    folder = RP.out_dir()
    pic = folder / f"{out['idea']}-with-his-hedge.jpg"
    name = out["idea"].replace("idea-00", "idea #").replace("idea-0", "idea #")
    RP.picture_boxes(m, pic, f"{name.capitalize()} with his hedge", "the version the referee chose - not a new try",
                     seen=False)
    L = [f"# {name.capitalize()} with his hedge (L19d)", "",
         "> L10b parked it \"for his hedge\" (it lost with a plain stop). Here its chosen version runs through his hedge "
         f"on 2016–2020 · version `{out['version']}`: {out['about']} · the same version, not a new try · trial file "
         f"`{out['trial']}` · 0.01 lot a leg · ₹40 a position, all-in · MT5 server time.", "",
         f"**The machine's choice: {out['choice']}** - {out['means']}", "",
         f"With his hedge: {hh['trades']} trades closed ({hh['open_at_end']} still open at the end, not counted), "
         f"{inr(hh['net_inr_total'])} after cost; worst year {hh['worst_year']['year']} "
         f"({inr(hh['worst_year']['net_inr'])}); a hedged trade stays open {hh['days_open_median']} days (the middle "
         "one).", ""]
    L += RP._boxes_md(m, "## The four boxes (₹ after cost)")
    L += ["", f"![the four boxes]({pic.name})", "",
          "Only 2016–2020: its check years 2021–2023 were opened once already, for the plain-stop judging; they are not "
          "opened again.", ""]
    path = _page_path(out["idea"])
    text = "\n".join(L) + "\n"
    from machine.report.build import check_words
    problems = check_words(text)
    if problems:
        raise ParkedError("the page's words: " + "; ".join(problems[:3]))
    path.write_text(text, encoding="utf-8")
    return path


def add_to_decision(idea: str, out: dict) -> Path:
    """A section added to the idea's decision.md (append only): his hedge tested, the machine's choice."""
    path = config.IDEAS_DIR / idea / "decision.md"
    head = f"## L19d - tested with his hedge ({out['trial'][:15]})"
    if path.is_file() and head in path.read_text(encoding="utf-8"):
        return path
    hh, r = out["his_hedging"], out["result"]
    text = "\n".join(["", head, "",
                      f"Its chosen version ({out['version']}) through his hedge on 2016-2020 - the same version, not a "
                      f"new try (trial `{out['trial']}`, page `plan/work/L19d/{idea}-with-his-hedge.md`):", "",
                      f"- with his hedge: {hh['trades']} trades, {inr(hh['net_inr_total'])} after cost; random entries "
                      f"with his hedge (middle of {r['runs']}): {inr(r['boxes'][2]['total'])}; his beat "
                      f"{r['his_beats_random']['with_hedging']} of {r['runs']}",
                      f"- with a plain stop: {out['his_stop']['trades']} trades, "
                      f"{inr(out['his_stop']['net_inr_total'])}", "",
                      f"**The machine's choice: {out['choice']}** - {out['means']} (LEARNED #162: it tells, never "
                      "asks.)", ""])
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def _page_path(idea: str) -> Path:
    from machine.strategy import report as RP
    return RP.out_dir() / f"{idea}-with-his-hedge.md"


def done(idea: str, version: str | None = None) -> dict | None:
    """The idea's newest L19d trial file (of that version), or None: it was not tested with his hedge yet."""
    got = [f for f in trials.read_all() if f.get("kind") == KIND and f.get("idea_id") == idea
           and str(f.get("note", "")).startswith("L19d:") and version in (None, f.get("version_id"))]
    return got[-1] if got else None


def _from_file(idea: str, f: dict) -> dict:
    """run()'s answer, read back from its trial file (nothing runs)."""
    r = f["result"]
    about = tune.latest(idea)["result"]["versions"][f["version_id"]].get("about", "")
    head, means = choice(r, r["his_hedging"])
    trial = next(Path(config.TRIALS_DIR).glob(f"{f['trial_id']}-*.json")).name
    return {"idea": idea, "version": f["version_id"], "about": about, "result": r, "his_hedging": r["his_hedging"],
            "his_stop": r["his_stop"], "choice": head, "means": means, "trial": trial}


def page_again(idea: str) -> Path:
    """The page made again from the idea's L19d trial file (nothing runs) - after a change of the words only."""
    f = done(idea)
    if f is None:
        raise ParkedError(f"{idea} has no L19d trial file: python -m machine.strategy parked {idea}")
    return write_page(_from_file(idea, f))
