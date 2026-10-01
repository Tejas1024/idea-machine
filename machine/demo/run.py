"""The demo (L11): one made-up idea through every step of the machine, on made-up prices and made-up notes.

    .venv\\Scripts\\python -m machine.demo            the steps, about 2 minutes (short windows of made-up prices)
    .venv\\Scripts\\python -m machine.demo --full     + the referee's real answer: 2016-2020 to choose, 2021-2023
                                                     opened once, judged, the report (about 20 minutes)

Every step runs the REAL code of the machine - the glossary, the #129 search, the talker's checks, the compiler, the
tester, the referee, the guards, the report. What is made up (tests/data/demo/README.md):
  - the notes the search reads (tests/data/demo/notes/): invented textbook lines, NOT the trader's private notes;
  - the idea's words and the talker's answer (tests/data/demo/idea.json): the answer is written by hand in the exact
    shape of the talker's one Claude tool call, so no key and no call are needed; code checks it like a real one;
  - the trader's reply to the question (stand_in_answers in idea.json);
  - the prices: a random walk of 1-minute bars 2016-2023 (machine/agent/noise.py walk_m1, the recipe in
    tests/data/demo/prices.json). No minute says anything about the next one: there is NO edge to find, so the
    honest answer is "chance".
Nothing reaches plan/work/: every file the steps write (forms, trial files, the check-opened record, the report) goes
to a temporary folder (noise.sandbox), or to --out when it is given.
"""
from __future__ import annotations

import contextlib
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from machine import config

DEMO_DIR = Path(config.ROOT) / "tests" / "data" / "demo"
NOTES_DIR = DEMO_DIR / "notes"
IDEA_FILE = DEMO_DIR / "idea.json"
PRICES_FILE = DEMO_DIR / "prices.json"
IDEA_ID = "idea-950"
ASK_WINDOW = ("2019-01-01", "2019-03-31")       # step 4: do the readings of a question make different trades?
TEST_WINDOW = ("2019-01-01", "2019-12-31")      # step 5 (quick): every version on one year of made-up prices
EXAM_SEEDS = 100                                # step 6: the referee's noise exam (selftest.py's own sizes)
EXAM_PLANTED = 200                              # ... and ideas with a planted Rs 50 / Rs 100 edge


def _say(text: str = "") -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:                  # a Windows console without UTF-8
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.replace("₹", "Rs ").encode(enc, errors="replace").decode(enc), flush=True)


def _head(n: int, title: str, say: Callable) -> None:
    say("")
    say(f"{'=' * 4} {n}. {title} {'=' * max(4, 100 - len(title))}"[:110])


def _inr(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"-₹{-x:,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def load_idea() -> dict:
    return json.loads(IDEA_FILE.read_text(encoding="utf-8"))


def made_up_prices() -> tuple[pd.DataFrame, dict]:
    from machine.agent.noise import walk_m1
    recipe = json.loads(PRICES_FILE.read_text(encoding="utf-8"))
    args = {k: recipe[k] for k in ("seed", "start", "end", "price", "step")}
    return walk_m1(**args), recipe


def notes_reference() -> str:
    """What the talker's citation check reads instead of his private files: the made-up notes."""
    parts = []
    for p in sorted((NOTES_DIR / "strategy").glob("*.md")):
        parts.append(p.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


@contextlib.contextmanager
def demo_world(root: Path):
    """Every folder the machine writes into -> `root`; the talker's reference -> the made-up notes."""
    from machine.agent.noise import sandbox
    from machine.talker import translate as T
    old_ref = T.reference
    T.reference = notes_reference
    try:
        with sandbox(root):
            yield
    finally:
        T.reference = old_ref


# ================================================================================================= the steps
def step_intake(idea: dict, say: Callable):
    from machine.talker import glossary
    _head(1, "THE IDEA, IN PLAIN WORDS", say)
    say(f'"{idea["words"]}"')
    g = glossary.apply(idea["words"])
    if g.text != idea["words"]:
        say(f"the glossary (his voice-typing slips -> what he meant): {g.text}")
    return g


def step_search(g, say: Callable):
    """The #129 rule as code: every term is searched in the notes BEFORE anything is asked."""
    from machine.search import index as ix
    from machine.search import rank
    from machine.talker import translate as T
    _head(2, "SEARCH THE NOTES FIRST (never ask what is already written)", say)
    idx = ix.build(NOTES_DIR, embeddings="off", cache_dir=Path(tempfile.mkdtemp(prefix="demo-search-")))
    search = (lambda q: rank.answered(q, index=idx))
    searches = T.run_searches(g.text, search)
    say(f"{len(idx.pieces)} pieces of notes (made up) + the machine's own rules, searched by words (BM25)")
    for s in searches:
        top = (s.hits[0] if s.hits else s.closest) if s.found else (s.closest or (s.hits[0] if s.hits else None))
        quote = f'{top.id}: "{top.quote}"' if top else "none"
        say(f'  "{s.term.said}" -> {"FOUND" if s.found else "NOT FOUND, closest"}  {quote}')
    return searches, search


def step_talker(idea: dict, g, searches, search, say: Callable):
    """The talker: ONE Claude tool call writes a FORM (not code). Code checks every part of it."""
    from machine.talker import llm
    from machine.talker import translate as T
    _head(3, "THE AI WRITES A FORM, CODE CHECKS IT", say)
    answer = llm.Answer(copy.deepcopy(idea["talker_answer"]), "recorded answer (tests/data/demo/idea.json)",
                        llm.Tokens(), 0.0, False, False, "tool_use")
    d = T.build(idea["words"], answer, g, searches, IDEA_ID, search)
    f = d.form
    if f is None:
        raise RuntimeError("the demo's form did not load: " + "; ".join(d.problems))
    say(f"form: {f.title} · {f.timeframe} chart · {' and '.join(f.directions)}")
    for s in f.steps:
        say(f"  step {s.id:<6} {s.block:<20} {s.what}")
    say(f"  entry  limit order at {f.entry.at}   ·   exit: stop at {f.exit.at}   ·   target: the next liquidity")
    say(f"code's checks: status {d.status} · problems {len(d.problems)} · citations not among the quotes shown "
        f"{len(d.unknown_citations)} · sanity checks failed {sum(1 for c in d.sanity if not c.ok)}")
    for x in d.problems + d.unknown_citations + [c.message for c in d.sanity if not c.ok]:
        say(f"  !! {x[:200]}")
    for u in f.unclear:
        how = "ASK him (only he knows what he means)" if u.kind == "ask" else "VERSION (the machine tries each)"
        say(f"  unclear {u.id} '{u.part}': {how}")
        for r in u.readings:
            say(f"      {r.id}. {r.means}")
    # a form that changes one of his FIXED rules is refused, with the rule's own words
    bad = copy.deepcopy(idea["talker_answer"])
    bad["entry"]["at"] = "gap.far"
    worse = T.build(idea["words"], llm.Answer(bad, "edited", llm.Tokens(), 0.0, False, False, "tool_use"), g,
                    searches, IDEA_ID, search)
    say("a form that enters at the FAR edge of the gap instead: " + ("REFUSED" if worse.problems else "accepted (?)"))
    for line in (worse.problems[0].splitlines()[1:4] if worse.problems else []):
        say("    " + line.strip()[:200])
    return d


def _signal_keys(signals) -> set:
    return {(str(s.known_at), s.direction, None if s.entry_price is None else round(float(s.entry_price), 2))
            for s in signals}


def step_questions(d, idea: dict, m1: pd.DataFrame, say: Callable):
    """Ask only where the readings make DIFFERENT trades. One question here; none for a VERSION part."""
    from machine.ideas.compile import Market, compile_idea, needed_tfs
    _head(4, "ASK ONLY WHERE THE READINGS MAKE DIFFERENT TRADES", say)
    form = d.form
    part = m1.loc[ASK_WINDOW[0]:ASK_WINDOW[1]]
    market = Market.from_m1(part, tfs=needed_tfs(form))
    for u in form.unclear:
        if u.kind != "ask":
            continue
        got = {}
        for r in u.readings:
            rd = {x.id: (r.id if x.id == u.id else x.readings[0].id) for x in form.unclear}
            got[r.id] = _signal_keys(compile_idea(form, market, readings=rd).signals)
        a, b = (got[r.id] for r in u.readings[:2])
        say(f"{u.id} '{u.part}' on {ASK_WINDOW[0][:7]}..{ASK_WINDOW[1][:7]} (made-up prices): reading A {len(a)} "
            f"setups, reading B {len(b)}, the same in both {len(a & b)}")
        if a != b:
            q = next(x for x in d.questions if x.id == u.id)
            say("  -> different trades: the question goes into his weekly batch (with A / B pictures, no money shown)")
            say(f"     the #129 sentence it carries: {q.searched[:260]}")
        else:
            say("  -> the same trades: NOT asked (asking would waste his time)")
    answers = idea.get("stand_in_answers", {})
    if answers:
        say(f"(his reply - in the demo a stand-in: {', '.join(f'{k} = {v}' for k, v in answers.items())})")
    parts = [u.model_copy(update={"answer": answers[u.id]}) if u.id in answers else u for u in form.unclear]
    return form.model_copy(update={"unclear": tuple(parts), "status": "ANSWERED"})


def step_tester(form, m1: pd.DataFrame, say: Callable):
    """Every VERSION reading is a version the machine tries by itself - each one counted as a try."""
    from machine.ideas.compile import Market, compile_idea, needed_tfs
    from machine.ideas.form import reading_choices
    from machine.report.run import about
    from machine.tester import run as tester_run
    _head(5, "TEST EVERY VERSION, IN RUPEES AFTER COST", say)
    part = m1.loc[TEST_WINDOW[0]:TEST_WINDOW[1]]
    market = Market.from_m1(part, tfs=needed_tfs(form))
    choices = reading_choices(form)
    say(f"{len(choices)} versions ({len(choices)} of the {config.VERSION_BUDGET} tries an idea may use) on "
        f"{TEST_WINDOW[0][:4]} made-up prices · 0.01 lot · ₹{config.COST_INR_PER_POSITION} a position all-in")
    for rd in choices:
        c = compile_idea(form, market, readings=rd)
        r = tester_run(c.signals, tf=c.tf, m1=part, note="the demo (made-up prices)")
        s = r.summary
        say(f"  {c.version_id} {about(form, rd)}: {s['trades']} trades, won {s['win_pct']}%, "
            f"{_inr(s['net_inr_per_trade'])} a trade, {_inr(s['net_inr_total'])} in all "
            f"(cost {_inr(s['cost_inr_total'])})")
    say("each run wrote one trial file (settings, data fingerprint, code fingerprint, result) - here into a temporary "
        "folder; in the real machine into plan/work/trials/, so every try is counted")


def step_referee_exam(say: Callable):
    """The referee's rules are fixed before any result; its own exam shows what they reject."""
    from machine.referee import selftest, verdict
    _head(6, "THE REFEREE: RULES FIXED IN ADVANCE, AND ITS OWN EXAM", say)
    for line in verdict.RULES.splitlines()[1:]:
        say("  " + line)
    say(f"  (the rules' fingerprint {verdict.RULES_SHA256[:12]} is in every trial file; a test pins it)")
    t = time.time()
    cost = float(config.COST_INR_PER_POSITION)
    say(f"its exam on made-up trades ({selftest.N_TRADES} trades an idea, 20 versions each, the best version chosen "
        "on 2016-2020 and judged on fresh years):")
    for mean, name in ((-cost, f"no edge (they lose the ₹{cost:.0f} cost)"),
                       (0.0, "the hardest noise (exactly ₹0 a trade after cost)")):
        n = selftest.noise_exam(seeds=EXAM_SEEDS, noise_mean=mean, every_version=mean == 0.0)
        say(f"  {n['seeds']} ideas with {name}: 'works' {n['chosen_works']} of {n['seeds']}")
        if n["any_of_all"] is not None:
            say(f"    (if all 20 versions were judged on the same years and the best one kept: 'works' for "
                f"{n['any_of_all']} of {n['seeds']} - why the check years open only ONCE)")
    for edge in (50, 100):
        p = selftest.planted_exam(edge, sims=EXAM_PLANTED)
        say(f"  {p['sims']} ideas with a real ₹{edge} edge a trade: 'works' {p['caught']} of {p['sims']}")
    say(f"  ({time.time() - t:.0f} s)")


def step_guards(m1: pd.DataFrame, say: Callable):
    from machine import guards
    _head(7, "THE GUARDS (CODE, NOT PROMISES)", say)
    try:
        guards.check_dates(("2024-03-01", "2024-03-31"))
        say("2024 was opened (?)")
    except guards.SealedYearsError as err:
        say(f"ask for March 2024 -> REFUSED: {str(err)[:150]} ...")
    bars = m1.loc["2019-01-02"].resample("5min").agg({"high": "max", "low": "min", "close": "last"}).dropna()

    def peeks(df):                      # a "level" that uses the next hour's high: it looks at the future
        return df["high"].rolling(12).max().shift(-11)

    def fair(df):                       # the last hour's high: only what was known
        return df["high"].rolling(12).max()

    cuts = [bars.index[60], bars.index[150]]
    guards.assert_no_future(fair, bars, cuts)
    say("a level from the PAST hour's high -> passes the cut-off-the-future test")
    try:
        guards.assert_no_future(peeks, bars, cuts)
        say("a level from the NEXT hour's high -> passed (?)")
    except guards.FutureLeakError as err:
        say(f"a level from the NEXT hour's high -> CAUGHT: {str(err).splitlines()[0][:150]}")
    try:
        guards.check_opened(IDEA_ID, [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-02-01")])
        say("2021 was opened before the choice (?)")
    except guards.CheckYearsClosedError as err:
        say(f"test on 2021 before a version is chosen -> REFUSED: {str(err).split(f'{IDEA_ID} has no')[0].strip()}")


def step_full(form, m1: pd.DataFrame, out: Path | None, say: Callable, root: Path | None = None) -> dict:
    """The real path (L10): choose on 2016-2020, open 2021-2023 ONCE for the chosen version, judge, report."""
    from machine.ideas import store
    from machine.ideas.compile import Market, compile_idea, needed_tfs
    from machine.ideas.form import reading_choices
    from machine.referee import tune
    from machine.report import build as B
    from machine.report import run as R
    _head(8, "THE REFEREE'S ANSWER (2016-2020 to choose, 2021-2023 opened once)", say)
    store.create(form)
    R.lock(IDEA_ID)
    tune_m1 = m1.loc[:str(config.TUNE_YEARS[-1])]
    market = Market.from_m1(tune_m1, tfs=needed_tfs(form))
    versions, abouts, readings = {}, {}, {}
    for rd in reading_choices(form):
        t = time.time()
        c = compile_idea(form, market, readings=rd)
        versions[c.version_id], abouts[c.version_id], readings[c.version_id] = c.signals, R.about(form, rd), rd
        say(f"  {c.version_id}: {len(c.signals)} setups on 2016-2020 ({time.time() - t:.0f} s) - {abouts[c.version_id]}")
    tuned = tune.run_versions(versions, tf=form.timeframe, m1=tune_m1, about=abouts, readings=readings,
                              note="the demo: every version on 2016-2020 (made-up prices)")
    store.set_state(IDEA_ID, "TUNE RUN", f"the demo: {len(versions)} versions; chosen {tuned.chosen}")
    say(tuned.text())
    rep = R.check_idea(IDEA_ID, who="the demo", log=lambda x: say("  " + x), m1=m1)
    built = B.build(IDEA_ID)
    say("")
    say(built.text)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        page = Path(built.page_path).read_text(encoding="utf-8")
        if root is not None:            # the temporary folder's full path names this PC's user
            page = page.replace(str(root), "<demo folder>")
        (out / "report.html").write_text(page, encoding="utf-8", newline="\n")
        pic = store.folder(IDEA_ID) / B.PICTURE
        if pic.is_file():
            (out / pic.name).write_bytes(pic.read_bytes())
        say(f"\nthe report page and its picture: {out}")
    return {"word": rep.word}


def main_steps(full: bool = False, out: Path | None = None, keep: Path | None = None, say: Callable = _say) -> dict:
    t0 = time.time()
    idea = load_idea()
    m1, recipe = made_up_prices()
    say("THE IDEA MACHINE - a demo on MADE-UP data (made-up notes, a made-up idea, made-up prices with no edge)")
    say(f"prices: {len(m1):,} one-minute bars 2016-2023, a random walk (seed {recipe['seed']}) - "
        "the real gold prices and the trader's notes are private")
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="idea-machine-demo-"))
    if not keep:                        # the temporary folder's full path names this PC's user: shown short
        shown, plain = say, str(root)
        say = (lambda text="": shown(str(text).replace(plain, "<demo folder>")))
    result = {}
    with demo_world(root):
        g = step_intake(idea, say)
        searches, search = step_search(g, say)
        d = step_talker(idea, g, searches, search, say)
        form = step_questions(d, idea, m1, say)
        step_tester(form, m1, say)
        step_referee_exam(say)
        step_guards(m1, say)
        if full:
            result = step_full(form, m1, out, say, root)
        else:
            _head(8, "THE REPORT", say)
            say("the referee's real answer needs 2016-2020 to choose and 2021-2023 opened once: "
                "python -m machine.demo --full (about 20 minutes)")
    say(f"\ndone in {time.time() - t0:.0f} s · every file it wrote is in "
        f"{root if keep else 'a temporary folder (<demo folder>)'} - nothing in plan/work/")
    return result
