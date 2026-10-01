"""L7b: idea #1 as a form, every reading compiled on 2016-2020, the signals counted (NO money - lesson 16).

    .venv\\Scripts\\python -m tests.idea001_form             writes plan/work/ideas/idea-001/ (form.json, state.json,
                                                           readings.md) and prints the counts
    .venv\\Scripts\\python -m tests.idea001_form --no-write  the counts only

The form comes from plan/work/idea-001/words.md (his words, T0) and steps.md (his "Ok", T3), with his two T3
no-reasons (examples.md). Idea #1 is not the Fibonacci idea, so strategy-understanding 66bz part (4) (F1-F3) does not
apply; its unclear parts are U1-U3 below. It is NOT in plan/work/prior-tries.md: fresh on 2016-2023.
His answers to the machine-found examples are checked against every reading: does a reading take the trades he said
YES to and leave the ones he said NO to? Round 3 is counted (one 5-minute chart each); rounds 1-2 are shown apart.
Nothing is traded here: the entry candle of a signal is where price first comes back to its limit price before the
target (5-minute bid bars, like T3) - no money, no trial file (R14 is for test runs).
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, settings
from machine.blocks._base import first_hit
from machine.data.clock import to_ist
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import Idea, open_parts, reading_choices
from tests.idea001_acceptance import read_examples

YEARS = ("2016", "2020")                 # the tune years only: 2021-2023 open once per idea, through the referee (L5)
ROOT = config.ROOT
INBOX = "strategy/learned-inbox/"

# ---- his words, word for word (each is checked against its file by check_quotes) ----------------------------------
W21A = ("What are equally strong ones? Even if it is the opposite side, the target — that is ALSO strong. Price will "
        "go towards that eventually or whenever.")
W21B = ("the market eventually moves towards any of them. Equal highs — or if it goes to the equal highs first, then "
        "it will eventually move to the equal lows.")
W21C = "Even if it takes the equal lows, then it will go to the opposite, opposite."
W1 = "Later touch must NOT pass reference; exact strongest; 3>2. A magnet, timing unknown."
W90 = ("it should be exactly equal, equal highs or equal lows. So, like one point, very negligible point is okay. One "
       "pip or two pips are okay. If it is not equal, if more than that, then it is not a equal highs or equal lows.")
T3_BIG, T3_FAR, T3_SO_FAR = "3.too big", "16. No that is too far it moved", "it moves so far"
T3_NEAR = "the lows are not proper lows cause they are too near to each other"
QUOTES = [(W21A, "strategy/LEARNED.md"), (W21B, "strategy/LEARNED.md"), (W21C, "strategy/LEARNED.md"),
          (W1, "strategy/strategy.md"), (W90, "strategy/LEARNED.md"),
          (T3_BIG, INBOX + "2026-09-28-1101-session.md"), (T3_FAR, INBOX + "2026-09-28-1130-session.md"),
          (T3_SO_FAR, INBOX + "2026-09-28-1151-session.md"), (T3_NEAR, INBOX + "2026-09-28-1151-session.md")]


def _src(ref: str, words: str | None = None) -> dict:
    return {"ref": ref, **({"words": words} if words else {})}


FORM = dict(
    id="idea-001",
    title="equal highs and equal lows are a magnet",
    words="\n".join([W1, W21A, W21B, W21C, W90]),
    timeframe="5m",
    directions=["buy", "sell"],
    steps=[
        dict(id="take", block="sweep", tf="5m", params={"kinds": ["equal"]},
             what="price takes equal highs (a sell) / equal lows (a buy): two touches at most $0.06 apart, the second "
                  "not past the first, nothing past the first between them, any time apart, on the 5-minute, hour or "
                  "day chart; one wick past the first touch is enough; a level taken once is finished",
             sources=[_src("idea-001/steps.md steps 1 and 3 (his 'Ok', 28 Sep)"),
                      _src("strategy.md section 1", W1), _src("LEARNED #90 (15 Sep)", W90),
                      _src("LEARNED #88, #89, #95, #129; rules.py what_equal_highs_and_lows_are")]),
        dict(id="other", block="level_waiting", tf="5m", max_wait_bars=0,
             params={"kinds": ["equal"], "pick": "nearest"},
             what="the other side of the pair waits, not taken yet: equal lows below (a sell) / equal highs above (a "
                  "buy); if there are several, the nearest",
             sources=[_src("idea-001/steps.md step 2"), _src("LEARNED #21 (6 Sep)", W21B)]),
        dict(id="brk", block="break_of_structure", tf="5m", max_wait_bars=30, params={"after": "take"},
             what="price breaks structure the trade's way: a wick through his order block (a sell: below the low of "
                  "the most recent red candle in the rise); a new extreme after a small break = the grab goes on: "
                  "wait for the break again, the hedge point moves with it",
             sources=[_src("idea-001/steps.md step 4"), _src("LEARNED #56-#60; strategy-understanding section 66a "
                                                             "(rules.py a_grab_that_keeps_going_is_still_a_grab)"),
                      _src("settings state_timeout_bars = 30 (Claude's): candles from the take to the break")]),
        dict(id="gap", block="gap_of_the_move", tf="5m", max_wait_bars=6,
             what="the FIRST 5-minute gap after the level was taken, made by the breaking move (its 3rd candle at "
                  "most 6 candles after the break), any size",
             sources=[_src("idea-001/steps.md step 5"), _src("LEARNED #61, #62, #101, #131")]),
    ],
    entry=dict(how="limit", at="gap.near", max_wait_bars="no limit",
               sources=[_src("idea-001/steps.md step 6: sell at the gap's bottom edge on price's FIRST visit"),
                        _src("LEARNED #65, #68")]),
    exit=dict(kind="hedge", at="brk.extreme",
              sources=[_src("idea-001/steps.md step 7: hedge point = the highest price of the grab"),
                       _src("strategy.md section 9 (Type 1)")]),
    targets=[dict(kind="price", at="other.price",
                  sources=[_src("idea-001/steps.md step 7: target = the equal lows"), _src("LEARNED #21 (6 Sep)",
                                                                                           W21C)])],
    do_not_enter=[
        dict(name="target_already_reached", params={"since": "take"},
             words="If price already reached the equal lows before it came back, there is no trade.",
             sources=[_src("idea-001/steps.md step 6"), _src("LEARNED #119 ('T05')")]),
        dict(name="target_reached_first", sources=[_src("idea-001/steps.md step 6"), _src("LEARNED #119")]),
        dict(name="grab_ran_too_far", words=T3_SO_FAR,
             sources=[_src("inbox 2026-09-28-1101 (T3 round 1)", T3_BIG), _src("inbox 2026-09-28-1130 (round 2)",
                                                                               T3_FAR),
                      _src("inbox 2026-09-28-1151 (round 3, #35-#38)", T3_SO_FAR)]),
    ],
    slices=[
        dict(name="direction", dimension="direction", how="buy / sell",
             why="his #21: price goes to both sides of the pair"),
        dict(name="session", dimension="session",
             how="the signal's India time: Asia 4:30-10:30 / London 12:30-15:30 / New York 17:30-20:30 / in between",
             why="conditions.md C1"),
        dict(name="volatility", dimension="volatility",
             how="today's ADR against its own last 12 months: high / normal / low (thirds)", why="conditions.md C6"),
        dict(name="pair_chart", dimension="place", how="the chart of the equal pair price took: 5-minute / hour / day",
             why="his #90: 'the same on every chart'; conditions.md C12"),
        dict(name="adr_zone", dimension="place",
             how="where the entry sits in today's ADR zones: near the green / middle / near the red / beyond a zone",
             why="conditions.md C7 (#76)"),
        dict(name="recent", dimension="time", how="when the level taken was formed: today / yesterday / older",
             why="conditions.md C15 (#130)"),
        dict(name="hedge_side_gap", dimension="place",
             how="an unfilled gap on the hedge side at the signal: yes / no", why="conditions.md C13 (#87)"),
        dict(name="friday", dimension="time", how="Friday / other days", why="conditions.md C10"),
    ],
    unclear=[
        dict(id="U1", kind="version", part="how far the grab may run past the level it took before the break",
             words=T3_SO_FAR,
             readings=[dict(id="A", means="at most 0.20 of a day's range", changes={"settings.grab_ran_too_far_adr": 0.2}),
                       dict(id="B", means="at most 0.25 of a day's range", changes={"settings.grab_ran_too_far_adr": 0.25}),
                       dict(id="C", means="at most 0.30 of a day's range", changes={"settings.grab_ran_too_far_adr": 0.3})],
             searched="I searched strategy.md sections 1-2, LEARNED #56-#62, #95, #129, strategy-understanding 66a and "
                      "his T3 answers for how far a grab may run past the level it took and found no number - his "
                      "round-3 no's ran 0.27-1.32 of a day, his yes 0.00-0.22 (examples.md). A number, not a meaning: "
                      "the machine tries it (four-boxes Box 3), never asks.",
             sources=[_src("inbox 2026-09-28-1151", T3_SO_FAR), _src("plan/work/rules-missing.md R17")]),
        dict(id="U2", kind="version",
             part="how far apart in time the two touches of equal highs / lows must be (his 'too near')",
             words=T3_NEAR,
             readings=[dict(id="A", means="no minimum (his written rule: 'any time apart')",
                            changes={"settings.equal_levels_min_apart_bars": 0}),
                       dict(id="B", means="at least 5 five-minute candles apart (25 minutes) - an edge case",
                            changes={"settings.equal_levels_min_apart_bars": 5}),
                       dict(id="C", means="at least 7 five-minute candles apart (35 minutes) - his 'around 7 candles'",
                            changes={"settings.equal_levels_min_apart_bars": 7})],
             searched="His answer, 29 Sep 2026 (inbox 2026-09-29-1155): 'it would be better if it around 7 candles, "
                      "the edge cases and other probabilities it should only check for the best results. And should "
                      "Inform me.' A number, so a VERSION: 7 is his, 5 the edge case, 'any time apart' his written rule "
                      "(Claude's picks; 9 left out - it drops 6 of his 10 round-3 yes); the referee says which works "
                      "better. Never asked (the 2 questions of 29 Sep were taken back).",
             sources=[_src("inbox 2026-09-28-1151", T3_NEAR),
                      _src("plan/work/question-queue/L6.2-Q1-lows-too-near.md (queued by L6.2 for the weekly batch)"),
                      _src("inbox 2026-09-29-1155", "it would be better if it around 7 candles, the edge cases and "
                                                    "other probabilities it should only check for the best results. "
                                                    "And should Inform me.")]),
        dict(id="U3", kind="version", part="which equal lows (a sell) / highs (a buy) are the target when several wait",
             readings=[dict(id="A", means="the nearest one (steps.md step 2, his 'Ok')",
                            changes={"steps.other.params.pick": "nearest"}),
                       dict(id="B", means="the most recent first (#130: recent liquidity is taken first)",
                            changes={"steps.other.params.pick": "his_order"})],
             searched="steps.md step 2 (his 'Ok', 28 Sep) says the nearest; LEARNED #130 says recent liquidity is "
                      "taken first. On his examples both pick the same pair except #22 (not answered), #35 and #36 "
                      "(his no). The target is the idea's own number (likhith.md section D): the machine tries both.",
             sources=[_src("plan/work/rules-missing.md R16"), _src("plan/work/L6-misses/README.md")]),
    ],
    sources=[_src("plan/work/idea-001/words.md (T0, 27 Sep)"), _src("plan/work/idea-001/steps.md (T3, his 'Ok')"),
             _src("plan/work/idea-001/examples.md (T3: 10 yes, 5 no on round 3)")],
    status="FORM DRAFTED",
    notes="Already seen: NO - idea #1 is not in plan/work/prior-tries.md, so 2016-2023 are fresh for it. "
          "His other entry rules are NOT in this form yet, because the machine cannot see them until their blocks "
          "exist (Stage 2): no sell at discount / no buy at premium unless the hour and the day agree (#76, "
          "four-boxes B1-37 - L6b.3); no sell at the real all-time high (#72, B1-60 - L6b.3); a trendline in the "
          "way or a 1-minute trendline against the entry (#69, #111, B1-28, B1-30 - L6b.1, L6b.4). T3 measured the "
          "first two on its rough search: they left out 37 of 1,106 setups (3%). They join this form (a new version, "
          "parent_idea idea-001) when those prompts are done. Stage 1 tests the 7 steps he agreed.",
)


def _norm(text: str) -> str:
    text = re.sub(r"(?m)^\s*>\s?", "", text).replace("**", "")
    return " ".join(text.split())


def check_quotes() -> list[str]:
    """Every quote of his in the form, word for word in its file ('…' in a quote = a cut)."""
    bad = []
    for q, path in QUOTES:
        file = ROOT / path
        if not file.is_file() and path.startswith(INBOX):      # numbered by learned-merge (L9a): moved to merged/
            file = ROOT / INBOX / "merged" / Path(path).name
        body = _norm(file.read_text(encoding="utf-8"))
        for part in _norm(q).split(" … "):
            if part not in body:
                bad.append(f"{path}: {part!r}")
    return bad


# ======================================================================================================== the form

def the_form() -> Idea:
    return Idea.load(FORM)


def save(form: Idea) -> Path:
    """The form into plan/work/ideas/idea-001/ (whole, through a temp file); its state -> QUESTIONS QUEUED (U2 waits
    for his answer to L6.2-Q1 in the weekly batch). A locked idea is never changed (store refuses)."""
    if (store.folder(form.id) / store.FORM).is_file():
        old = store.load_form(form.id)
        store.save_form(form.model_copy(update={"status": old.status}))
    else:
        store.create(form)
    if store.read_state(form.id)["state"] == "FORM DRAFTED":
        store.set_state(form.id, "QUESTIONS QUEUED", "L7b: the form compiles; U2 ('too near') is an ASK part",
                        waits_for="Tejas's answer to L6.2-Q1-lows-too-near (U2) in the weekly picture batch (L9b)")
    return store.folder(form.id) / store.FORM


# ==================================================================================================== the readings

def entries(signals, m5: pd.DataFrame) -> dict[str, int]:
    """signal id -> the 5-minute candle price first comes back to its limit price, before the target (cancel_at);
    -1 = never (5-minute bid bars, like T3; the tester fills on the 1-minute ask - L3)."""
    if not signals:
        return {}
    h, lo = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
    i0 = m5.index.searchsorted(pd.DatetimeIndex([pd.Timestamp(s.known_at) for s in signals]))
    sell = np.array([s.direction == "sell" for s in signals])
    px = np.array([s.entry_price for s in signals])
    cancel = np.array([np.nan if s.cancel_at is None else s.cancel_at for s in signals])
    f = np.full(len(signals), -1)
    c = np.full(len(signals), -1)
    for side, arr_e, op_e, arr_c, op_c in ((True, h, ">=", lo, "<="), (False, lo, "<=", h, ">=")):
        m = sell == side
        if m.any():
            f[m] = first_hit(arr_e, i0[m], px[m], op_e)
            ok = m & ~np.isnan(cancel)
            if ok.any():
                c[ok] = first_hit(arr_c, i0[ok], cancel[ok], op_c)
    filled = (f >= 0) & ~((c >= 0) & (c <= f))
    return {s.id: int(f[k]) if filled[k] else -1 for k, s in enumerate(signals)}


def judge(out: C.Compiled, fill: dict[str, int], examples, m5: pd.DataFrame) -> dict[int, str]:
    """For each of his examples: 'takes it' (same direction, same entry candle, same limit price), '... aiming at
    another pair' (the target is not the pair he saw), or why not."""
    tol = 0.06 + 1e-9
    by_candle = {}
    for s in out.signals:
        if fill.get(s.id, -1) >= 0:
            by_candle.setdefault((s.direction, fill[s.id]), []).append(s)
    res = {}
    for e in examples:
        i = int(m5.index.get_indexer([e.entry])[0])
        got = [s for s in by_candle.get((e.direction, i), []) if abs(s.entry_price - round(e.near, 2)) < 0.005]
        if not got:
            res[e.k] = "not taken"
        elif any(abs(s.targets[0] - e.target_px) <= tol for s in got):
            res[e.k] = "takes it"
        else:
            res[e.k] = "takes it, aiming at another pair"
    return res


def run(write: bool = True) -> dict:
    bad = check_quotes()
    if bad:
        raise SystemExit("a quote in the form is not word for word:\n" + "\n".join(bad))
    form = the_form()
    choices = reading_choices(form)
    print(f"idea #1: {len(open_parts(form))} open parts ({', '.join(u.id for u in open_parts(form))}) -> "
          f"{len(choices)} readings to compile; market {YEARS[0]}-{YEARS[1]}")
    t0 = time.time()
    market = C.Market.load(*YEARS)
    m5 = market.bars("5m")
    examples = read_examples()
    rows = []
    for k, rd in enumerate(choices + [None]):
        t1 = time.time()
        if rd is None:        # not a reading: the same form with no "ran too far" line, to show what the line does
            rd = {"U1": "A", "U2": "A", "U3": "A"}
            out = C.compile_idea(form, market, readings=rd, changes={"settings.grab_ran_too_far_adr": None})
            name = "no line (NOT a reading)"
        else:
            out = C.compile_idea(form, market, readings=rd)
            name = " ".join(f"{p}={r}" for p, r in rd.items())
        fill = entries(out.signals, m5)
        j = judge(out, fill, examples, m5)
        years = Counter(pd.Timestamp(s.known_at).year for s in out.signals)
        row = dict(name=name, readings=rd, version=out.version_id, counts=out.counts(),
                   entries=sum(v >= 0 for v in fill.values()), years=dict(sorted(years.items())), judged=j,
                   regrabs=out.regrabs, funnel=out.funnel, dropped=out.dropped, reading_only=name.startswith("no line"))
        for tag, rnds in (("r3", (3,)), ("r12", (1, 2))):
            ys = [e.k for e in examples if e.rnd in rnds and e.answer == "yes"]
            ns = [e.k for e in examples if e.rnd in rnds and e.answer == "no"]
            row[f"{tag}_yes"] = (sum(j[x] != "not taken" for x in ys), len(ys))
            row[f"{tag}_no"] = (sum(j[x] != "not taken" for x in ns), len(ns))
        rows.append(row)
        c = row["counts"]
        print(f"  {name:28} {out.version_id}  signals {c['all']:>5,} ({c['buy']:,} buy / {c['sell']:,} sell)  "
              f"entries {row['entries']:>5,}  r3 yes {row['r3_yes'][0]}/{row['r3_yes'][1]}  r3 no taken "
              f"{row['r3_no'][0]}/{row['r3_no'][1]}  r1-2 yes {row['r12_yes'][0]}/{row['r12_yes'][1]}  no taken "
              f"{row['r12_no'][0]}/{row['r12_no'][1]}   {time.time() - t1:.0f} s")
    R = dict(form=form, rows=rows, examples=examples, seconds=time.time() - t0,
             text=C.compile_idea(form, market, readings={"U1": "B", "U2": "A", "U3": "A"}).text())
    print(R["text"])
    if write:
        save(form)
        write_readings(R)
    return R


# ======================================================================================================== the file

def _ist(t) -> str:
    return to_ist(pd.Timestamp(t)).strftime("%d %b %Y %H:%M")


def write_readings(R: dict) -> Path:
    form, rows = R["form"], R["rows"]
    real = [r for r in rows if not r["reading_only"]]
    ref = next(r for r in rows if r["reading_only"])
    years = sorted({y for r in rows for y in r["years"]})
    u = {x.id: x for x in form.unclear}
    L = [f"# Idea #1 — every reading compiled (L7b)", "",
         f"> Made by `python -m tests.idea001_form` on {pd.Timestamp.now():%d %b %Y}. The form: "
         "`plan/work/ideas/idea-001/form.json`. **Counts only — no money** (lesson 16: the meaning is chosen before "
         f"any money is seen). Years {YEARS[0]}–{YEARS[1]} only (2021–2023 open once, through the referee). "
         "Nothing was traded, so no trial file.", "",
         "**Already seen: NO.** Idea #1 is not in `plan/work/prior-tries.md` — 2016–2023 are fresh for it.", "",
         "## The unclear parts", "",
         "| part | kind | what is unclear | readings |", "|---|---|---|---|"]
    for x in form.unclear:
        rd = " · ".join(f"**{r.id}** {r.means}" for r in x.readings)
        L.append(f"| {x.id} | {'ASK (a question to him)' if x.kind == 'ask' else 'VERSION (the machine tries it)'} "
                 f"| {x.part} | {rd} |")
    L += ["", f"{len(real)} readings = 3 × 2 × 2. U2 is an ASK part: its question is already queued "
          "(`plan/work/question-queue/L6.2-Q1-lows-too-near.md`); once he answers, 6 readings are left — 6 of the "
          f"{config.VERSION_BUDGET} versions an idea may try.", "",
          "## Signals per reading", "",
          "signals = limit orders the form makes · entries = those price came back to before the target (5-minute "
          "bars) · his examples: how many of his answers the reading gets right — round 3 is counted; the two lines "
          "U1 and U2 were read off round 3 itself, so rounds 1–2 are the only answers they were not drawn from.", "",
          "| reading | version | signals | buy | sell | entries | " + " | ".join(str(y) for y in years)
          + " | round 3: his 10 yes taken | round 3: his 5 no taken | rounds 1–2: his 14 yes taken | his 2 no taken |",
          "|---|---|---|---|---|---|" + "---|" * len(years) + "---|---|---|---|"]
    for r in rows:
        c = r["counts"]
        name = r["name"] if r["reading_only"] else " · ".join(
            f"{p}={v} ({u[p].readings[[x.id for x in u[p].readings].index(v)].means.split(' (')[0]})"
            for p, v in r["readings"].items())
        L.append(f"| {name} | `{r['version']}` | {c['all']:,} | {c['buy']:,} | {c['sell']:,} | {r['entries']:,} | "
                 + " | ".join(f"{r['years'].get(y, 0):,}" for y in years)
                 + f" | {r['r3_yes'][0]} | {r['r3_no'][0]} | {r['r12_yes'][0]} | {r['r12_no'][0]} |")
    L += ["", "## His examples, reading by reading", "",
          "takes = the reading makes his trade (same direction, entry candle and price); **other pair** = it takes it "
          "but aims at a different pair than the one he saw; — = not taken.", "",
          "| # | round | buy / sell | entry (India time) | his answer | " + " | ".join(
              " ".join(f"{p}{v}" for p, v in r["readings"].items()) for r in real) + " | no line |",
          "|---|---|---|---|---|" + "---|" * (len(real) + 1)]
    short = {"takes it": "takes", "takes it, aiming at another pair": "other pair", "not taken": "—"}
    for e in R["examples"]:
        ans = e.answer or "(not answered)"
        L.append(f"| {e.k} | {e.rnd} | {e.direction} | {_ist(e.entry)} | {ans} | "
                 + " | ".join(short[r["judged"][e.k]] for r in real + [ref]) + " |")
    base = next(r for r in real if r["readings"] == {"U1": "B", "U2": "A", "U3": "A"})
    near = next(r for r in real if r["readings"] == {"U1": "B", "U2": "B", "U3": "A"})
    u3 = next(r for r in real if r["readings"] == {"U1": "B", "U2": "A", "U3": "B"})
    n_ex = len(R["examples"])
    lost = [e.k for e in R["examples"] if e.rnd == 3 and e.answer == "yes" and near["judged"][e.k] == "not taken"]
    r12_lost = [e.k for e in R["examples"] if e.rnd < 3 and e.answer == "yes" and base["judged"][e.k] == "not taken"]
    L += ["", "## One reading, step by step (U1=B · U2=A · U3=A)", "", "```text", R["text"], "```", "",
          "## What the counts say (Claude's reading — the choice between readings is made by the tests and his "
          "answers, not here)", "",
          f"- **The form compiles** in every reading: {min(r['counts']['all'] for r in real):,}–"
          f"{max(r['counts']['all'] for r in real):,} signals in 2016–2020 "
          f"({R['seconds'] / 60:.0f} minutes for all {len(rows)} compiles).",
          f"- **Without the line U1** the form takes {sum(v != 'not taken' for v in ref['judged'].values())} of his "
          f"{n_ex} examples, yes and no alike ({ref['counts']['all']:,} signals): it finds what T3 found. The line is "
          "what leaves out his no's.",
          f"- **With 0.25 (U1=B, U2=A, U3=A):** {base['counts']['all']:,} signals; round 3 — {base['r3_yes'][0]} of "
          f"his {base['r3_yes'][1]} yes, {base['r3_no'][0]} of his {base['r3_no'][1]} no. That proves little: the line "
          "was read off round 3, and one part of it was found on his #38 (below). Rounds 1–2, which the line was not "
          f"drawn from: {base['r12_yes'][0]} of {base['r12_yes'][1]} yes (#{', #'.join(map(str, r12_lost))} ran "
          f"further than 0.25), {base['r12_no'][0]} of {base['r12_no'][1]} no.",
          "- **The run counts from the FIRST level the grab took** (found on his #38): price took his equal highs at "
          "1207.17, ran on, and took a second pair at 1210.20 before the break. Measured from the second pair the "
          "run was 0.07 of a day, so the machine still took the trade he said no to. Now, when a grab took several "
          "levels that lead to the same entry, the farthest one decides.",
          f"- **U2=B ('too near' = 7 candles)** loses {len(lost)} of his round-3 yes (#{', #'.join(map(str, lost))}) "
          f"— as L6.2 found; his answer to L6.2-Q1 decides it.",
          f"- **U3** (nearest or most recent target) changes {abs(u3['counts']['all'] - base['counts']['all'])} "
          f"signal(s) of {base['counts']['all']:,} and none of his examples.",
          f"- **The grab kept going** {base['regrabs']:,} times in that reading: the break was looked for again and the "
          "hedge point moved (his fixed rule; built in L7b).",
          "- **Not in the form yet**, because the machine cannot see them until their blocks exist: "
          + form.notes.split("until their blocks exist (Stage 2): ")[1].split(" Stage 1")[0], ""]
    path = store.folder(form.id) / "readings.md"
    store._write(path, "\n".join(L) + "\n")
    print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    run(write=not ap.parse_args().no_write)
