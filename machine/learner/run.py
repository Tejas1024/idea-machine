"""One idea through the learner (L13.1): what was known at each setup -> condition tables -> the slice checks.

    .venv\\Scripts\\python -m machine.learner idea-001          one trial file (kind "learner"), then the text
    .venv\\Scripts\\python -m machine.report build idea-001     the report now has its "works best in / fails in" lines

What it reads: the idea's form (its slices, chosen before the test), the referee's tune file (the chosen version, its
readings and its trades on 2016-2020 - machine/referee/tune.py) and, when the check years were opened for that version,
the referee's judgement (its trades on 2021-2023). It compiles the chosen version again with the compiler's trace
(what each step found), refuses to go on unless its 2016-2020 signals are exactly the tune's, builds the features
(machine/learner/features.py) and joins them to the referee's trades by signal id. No trade is made or re-made here:
every rupee is the referee's.

The slice checks run on the 2016-2020 trades only (likhith.md section D names those years). The hidden years are SHOWN
next to each chosen slice when they are open for this version - they decide nothing (machine/learner/slices.py RULES).
A learner run is not a try: it splits the trades of a version that already has its trial files.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, guards, settings
from machine import trials as trial_files
from machine.learner import conditions as CN
from machine.learner import features as F
from machine.learner import slices as SL

CHECK_PREFIX = "the check years"                   # judge.judge_check's judged_on starts with this
LEARNER_FILES = ("features.py", "conditions.py", "slices.py", "run.py")
TFS = ("1m", "5m", "1h", "1d", "1w")               # the charts the features read (1m: the lines against the entry)


class LearnError(ValueError):
    """The learner refused: no tune file, signals that are not the tune's, a trade with no setup."""


def _inr(x) -> str:
    """Whole rupees with a sign (-₹1,234 · ₹790); a value that rounds to 0 is ₹0, never "₹-0"."""
    if x is None:
        return "no trades"
    r = round(x)
    return f"-₹{-r:,}" if r < 0 else f"₹{abs(r):,}"


def years_text(years) -> str:
    ys = sorted(int(y) for y in years)
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"


# ================================================================================================= the analysis

def trades_of(table: dict) -> pd.DataFrame:
    """The closed trades of a referee trades table ({"columns", "rows"}): index = signal id; net_inr, exit_time."""
    df = pd.DataFrame(table["rows"], columns=table["columns"])
    df = df[(df["status"] == "closed") & df["net_inr"].notna()]
    out = pd.DataFrame({"net_inr": df["net_inr"].astype(float).to_numpy(),
                        "exit_time": pd.to_datetime(df["exit_time"]).to_numpy()},
                       index=pd.Index(df["signal"].to_numpy(), name="signal"))
    if out.index.has_duplicates:
        raise LearnError("two trades share a signal id")
    return out


def analyse(form, trades: pd.DataFrame, feats: pd.DataFrame, *, check_trades: pd.DataFrame | None = None) -> dict:
    """The condition tables and the slice checks for one version's trades.

    form: the idea (its slices were chosen before the test). trades: net_inr + exit_time per closed trade on the tune
    years, index = signal id. feats: features.build's table (it must hold every trade's signal). check_trades: the
    same version's trades on the hidden years, when they are open - shown per chosen slice, never tested."""
    missing = [s for s in trades.index if s not in feats.index]
    if missing:
        raise LearnError(f"{len(missing)} trades have no setup in the feature table (the first: {missing[0]})")
    bad = sorted(set(pd.DatetimeIndex(trades["exit_time"]).year) - set(config.TUNE_YEARS))
    if bad:
        raise LearnError(f"the slice checks read the tune years {config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]} only "
                         f"(likhith.md section D); these trades closed in {bad}")
    f = feats.loc[trades.index]
    ids = list(trades.index)
    chosen, tests, names = [], {}, []
    for sl in form.slices:
        cond = CN.for_slice(sl)
        row = {"name": sl.name, "dimension": sl.dimension, "how": sl.how, "why": sl.why,
               "condition": None if cond is None else cond.name, "label": sl.name if cond is None else cond.label}
        if cond is None:
            row.update(measured=False, trades=0, not_known=len(trades), groups=[],
                       why_not=f"the machine has no condition named '{sl.name}' yet (machine/learner/conditions.py)")
            tests[sl.name] = {"tested": False, "why": "not measured", "p": 1.0}
        else:
            labels = cond.labels(f)
            seed = SL.seed_of(ids, cond.name)
            row.update(measured=True, source=cond.source, measured_how=cond.how,
                       **SL.table(trades, labels, cond.groups, seed=seed))
            tests[sl.name] = SL.test(trades, labels, cond.groups, seed=seed, slices_chosen=len(form.slices))
            if check_trades is not None and len(check_trades):
                lab_c = cond.labels(feats.loc[check_trades.index]).to_numpy()
                vc = check_trades["net_inr"].to_numpy(dtype=float)
                row["check_years"] = [{"group": g, **SL.numbers(vc[lab_c == g])} for g in cond.groups]
            names.append(cond.name)
        chosen.append(row)
    judged = SL.judge(tests)
    claims = []
    for row in chosen:
        row["test"] = judged[row["name"]]
        if row["test"].get("claim"):
            g = next(x for x in row["groups"] if x["group"] == row["test"]["part"])
            claims.append({"slice": row["name"], "label": row["label"], "claim": row["test"]["claim"],
                           "part": g["group"], "trades": g["trades"], "per_trade": g["per_trade"],
                           "rest_trades": g["rest_trades"], "rest_per_trade": g["rest_per_trade"], "gap": g["gap"]})
    looked, not_here = [], []
    for cond in CN.ALL:
        if cond.name in names:
            continue
        labels = cond.labels(f)
        if not labels.notna().any():
            not_here.append({"name": cond.name, "label": cond.label,
                             "why": "this idea has no such step" if cond.needs_facts else "not known on these trades"})
            continue
        parts = labels.dropna().unique()
        if len(parts) == 1:                                   # it splits nothing on this idea
            not_here.append({"name": cond.name, "label": cond.label,
                             "why": f"every trade is '{parts[0]}' - it splits nothing on this idea"})
            continue
        looked.append({"name": cond.name, "label": cond.label, "dimension": cond.dimension, "source": cond.source,
                       "measured_how": cond.how,
                       **SL.table(trades, labels, cond.groups, seed=SL.seed_of(ids, cond.name))})
    few = [r["label"] for r in chosen if r["measured"] and not r["test"]["tested"]]
    return {
        "trades": int(len(trades)), "years": list(config.TUNE_YEARS),
        "slices": chosen, "claims": claims, "looked_at": looked, "does_not_apply": not_here,
        "counts": {"slices_chosen": len(chosen), "slice_budget": config.SLICE_BUDGET,
                   "slices_measured": sum(r["measured"] for r in chosen),
                   "slices_tested": sum(bool(r["test"]["tested"]) for r in chosen),
                   "slices_passed": len(claims), "slices_too_few": few,
                   "looked_at": len(looked), "does_not_apply": len(not_here)},
        "check_years_shown": bool(check_trades is not None and len(check_trades)),
        "check_trades": 0 if check_trades is None else int(len(check_trades)),
    }


# ================================================================================================= the words

def works_line(r: dict) -> str:
    """The report's line (easy English, #113): what the chosen slices say, or that nothing passed."""
    c = r["counts"]
    if not c["slices_chosen"]:
        return ("Works best in / fails in: no slice was chosen before the test, so nothing is claimed "
                f"({c['looked_at']} conditions looked at, not tested)")
    head = (f"Works best in / fails in ({years_text(r['years'])}, the {c['slices_chosen']} slices chosen before the "
            f"test): ")
    if r["claims"]:
        said = "; ".join(
            f"{'WORKS BEST' if x['claim'] == SL.WORKS_BEST else 'FAILS'} when {x['label']} is "
            f"'{x['part']}' ({_inr(x['per_trade'])} a trade on {x['trades']} trades; the other {x['rest_trades']} "
            f"trades {_inr(x['rest_per_trade'])} a trade)"
            + (" - it still loses there" if x["claim"] == SL.WORKS_BEST and (x["per_trade"] or 0) <= 0 else "")
            + (" - the other trades lose too" if x["claim"] == SL.FAILS and (x["rest_per_trade"] or 0) <= 0 else "")
            for x in r["claims"])
        return head + said
    out = head + "nothing passed the 3 checks"
    seen = [s["test"]["could_see"] for s in r["slices"] if s["test"].get("could_see")]
    if seen:
        out += (f" - with these {r['trades']} trades a part must differ from the rest by about {_inr(min(seen))} a "
                "trade or more to be told from luck" if len(seen) == 1 or round(min(seen)) == round(max(seen)) else
                f" - with these {r['trades']} trades a part must differ from the rest by about {_inr(min(seen))} to "
                f"{_inr(max(seen))} a trade or more (by slice) to be told from luck")
    if c["slices_too_few"]:
        out += f"; {SL.FEW} to test: {', '.join(c['slices_too_few'])}"
    return out


def slice_lines(r: dict) -> list[str]:
    """One line per chosen slice: the trades in each part (* = too few trades to test that part)."""
    out = []
    for s in r["slices"]:
        if not s["measured"]:
            out.append(f"· {s['label']}: not measured ({s['why_not']})")
            continue
        parts = " · ".join(f"{g['group']} {g['trades']}{'' if g['enough'] else '*'}" for g in s["groups"]
                           if g["trades"])                    # a part with no trade at all: on the page only
        extra = f" · not known {s['not_known']}" if s["not_known"] else ""
        mark = f" → {s['test']['claim']} {s['test']['part']}" if s["test"].get("claim") else ""
        out.append(f"· {s['label']}: {parts}{extra}{mark}")
    return out


def text(r: dict, idea: str = "", version: str = "") -> str:
    """The learner's own print-out: every chosen slice with its numbers and checks, then the conditions looked at."""
    lines = [f"THE LEARNER · {idea} version {version} · {r['trades']} trades closed in {years_text(r['years'])} · "
             "₹ after the ₹40 cost, 0.01 lot · India time for sessions and hours, the broker's day for weekdays",
             works_line(r), "",
             f"THE {r['counts']['slices_chosen']} SLICES CHOSEN BEFORE THE TEST (the interval: the middle "
             f"{round(100 * config.SLICE_INTERVAL)} of 100 when the weeks are drawn again)"]
    for s in r["slices"]:
        lines.append(f"{s['label']} [{s['name']}]")
        if not s["measured"]:
            lines.append(f"    not measured: {s['why_not']}")
            continue
        lines += _table_lines(s, s.get("check_years"))
        lines.append("    " + check_text(s["test"]))
    lines += ["", f"LOOKED AT, NOT TESTED ({r['counts']['looked_at']} conditions - nothing here is a finding: with "
                  "this many tables some parts look good by luck alone)"]
    for c in r["looked_at"]:
        lines.append(f"{c['label']} [{c['name']}]")
        lines += _table_lines(c, None)
    if r["does_not_apply"]:
        lines.append("")
        lines.append("Not on this idea: " + "; ".join(f"{x['label']} ({x['why']})" for x in r["does_not_apply"]))
    return "\n".join(lines)


def _table_lines(t: dict, check: list | None) -> list[str]:
    hidden = {x["group"]: x for x in (check or [])}
    out = []
    for g in t["groups"]:
        if not g["trades"]:
            line = f"    {g['group']}: no trades"
        else:
            iv = g["interval"]
            line = (f"    {g['group']}: {g['trades']} trades · {_inr(g['per_trade'])} a trade"
                    + (f" (between {_inr(iv[0])} and {_inr(iv[1])})" if iv else "")
                    + ("" if g["enough"] else f" · {SL.FEW}"))
        h = hidden.get(g["group"])
        if h is not None:
            line += (f" · hidden years {years_text(config.CHECK_YEARS)}: {h['trades']} trades"
                     + (f", {_inr(h['per_trade'])} a trade" if h["trades"] else ""))
        out.append(line)
    if t.get("not_known"):
        out.append(f"    not known: {t['not_known']} trades")
    return out


def check_text(t: dict) -> str:
    """One slice's three checks in plain words."""
    if not t.get("tested"):
        return f"not tested: {t.get('why', SL.FEW)}"
    h, hs = t["holm"], t["holm_shuffle"]
    halves = " · ".join(f"{x['years']}: " + ("no trades on one side" if x["gap"] is None else
                                             f"{_inr(x['gap'])} a trade ({x['trades']} trades against "
                                             f"{x['rest_trades']})")
                        for x in t["halves"])
    return (f"the part that stands out most: {t['part']} ({_inr(t['gap'])} a trade against the rest). "
            f"1. luck test: luck alone gives a gap this big {100 * t['p']:.1f} times in 100 (this slice needs under "
            f"{100 * h['needs_under']:.2f}): {'yes' if h['passed'] else 'NO'}. "
            f"2. the same way in both halves ({halves}): {'yes' if t['same_way'] else 'NO'}. "
            f"3. shuffled labels give it {100 * t['p_shuffle']:.1f} times in 100 (this slice needs under "
            f"{100 * hs['needs_under']:.2f}): {'yes' if hs['passed'] else 'NO'}. "
            + (f"→ {t['claim']} {t['part']}" if t.get("claim") else "→ no finding")
            + (f" (a gap under about {_inr(t['could_see'])} a trade cannot be told from luck with these trades)"
               if t.get("could_see") and not t.get("claim") else ""))


# ================================================================================================= the run

@dataclass
class Learned:
    idea_id: str
    version_id: str
    result: dict
    feats: pd.DataFrame
    trial_path: Path | None

    def text(self) -> str:
        out = text(self.result, self.idea_id, self.version_id)
        if self.trial_path is not None:
            p = self.trial_path
            out += f"\n\nTrial file: {p.relative_to(config.ROOT).as_posix() if p.is_relative_to(config.ROOT) else p}"
        return out


def latest(idea: str, version: str | None = None, folder: Path | None = None) -> dict | None:
    """The newest learner trial file of this idea (and version), or None."""
    from machine.referee import trials
    files = [f for f in trials.load(folder) if f.get("kind") == config.LEARNER_KIND and f.get("idea_id") == idea
             and (version is None or f.get("version_id") == version)]
    return files[-1] if files else None


def code_sha256() -> str:
    here = Path(__file__).parent
    return trial_files.code_sha256([here / f for f in LEARNER_FILES])


def compiled_setups(form, tune_file: dict, market, *, news: pd.DataFrame | None = None, log=print):
    """The chosen version compiled again with the compiler's trace, and the features of every setup. Refuses unless
    its tune-year signals are exactly the tune's: the learner never describes setups the referee did not judge.
    Returns (the compile, the feature table)."""
    from machine.ideas.compile import compile_idea
    from machine.referee import tune
    chosen = tune_file["result"]["chosen"]
    row = tune_file["result"]["versions"][chosen]
    t0 = time.time()
    c = compile_idea(form, market, readings=row["readings"], changes=row.get("changes") or None, trace=True)
    if c.version_id != chosen:
        raise LearnError(f"the chosen readings compile to {c.version_id}, not {chosen}: the form or the code changed "
                         "since the tune")
    tune_years = [int(y) for y in tune_file["settings"]["years"]]
    last_tune = pd.Timestamp(f"{tune_years[-1] + 1}-01-01")
    before = [s for s in c.signals if pd.Timestamp(s.known_at) <= last_tune]
    if tune.signals_sha256(before) != row["signals_sha256"]:
        raise LearnError(f"the {years_text(tune_years)} signals of {chosen} compiled now are not the tune's "
                         f"({len(before)} now, {row['signals']} then): the blocks changed since the tune, or a signal "
                         "reads bars after its moment - the learner never describes setups the referee did not judge")
    log(f"compiled {len(c.signals)} signals with their steps in {time.time() - t0:.0f} s (the same as the tune's)")
    t1 = time.time()
    with settings.using(c.resolved):
        feats = F.build(F.setups_from(c, market), market, fp=settings.fingerprint(c.resolved), form=c.form, news=news)
    log(f"features of {len(feats)} setups in {time.time() - t1:.0f} s")
    return c, feats


def learn_with(form, tune_file: dict, market, *, judged: dict | None = None, news: pd.DataFrame | None = None,
               data: dict | None = None, record: bool = True, note: str | None = None, log=print) -> Learned:
    """The learner on a market already read. tune_file: the referee's tune trial (the chosen version, its readings,
    its signals' fingerprint, its trades). judged: the referee's judgement on the check years for that version (its
    trades are shown per chosen slice) - the market must then reach those years."""
    chosen = tune_file["result"]["chosen"]
    row = tune_file["result"]["versions"][chosen]
    tune_years = [int(y) for y in tune_file["settings"]["years"]]
    _, feats = compiled_setups(form, tune_file, market, news=news, log=log)
    trades = trades_of(tune_file["result"]["trades_table"])
    check = None
    if judged is not None:
        if judged.get("version_id") != chosen or judged.get("idea_id") != form.id:
            raise LearnError(f"the judgement given is for {judged.get('idea_id')} {judged.get('version_id')}, "
                             f"not {form.id} {chosen}")
        check = trades_of(judged["result"]["trades_table"])
        lost = [s for s in check.index if s not in feats.index]
        if lost:
            raise LearnError(f"{len(lost)} judged trades of the hidden years have no setup (the first: {lost[0]})")
    result = analyse(form, trades, feats, check_trades=check)
    result["tune_trial"] = tune_file["trial_id"]
    result["judged_trial"] = None if judged is None else judged["trial_id"]
    path = None
    if record:
        sets = {
            "years": tune_years, "timeframe": form.timeframe, "rules_sha256": SL.RULES_SHA256,
            "min_trades": config.SLICE_MIN_TRADES, "halves": [list(h) for h in config.SLICE_HALVES],
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES, "shuffles": config.SLICE_SHUFFLES,
            "interval": config.SLICE_INTERVAL, "p_limit": config.P_LIMIT,
            "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
            "clock": "MT5 server time in; India time for session and hour; the broker's day for weekday and today",
            "conditions": [c_.name for c_ in CN.ALL], "features": list(F.COLUMNS),
            "news": "Forex Factory red + orange US rows (T7); the next one only if scheduled",
            "signals_sha256": row["signals_sha256"], "code_sha256": code_sha256(),
            "a_try": "no - it splits the trades of a version the referee already judged",
        }
        path = trial_files.record(kind=config.LEARNER_KIND, idea_id=form.id, version_id=chosen,
                                  parent_idea=form.parent_idea, settings=sets,
                                  data=data or {"prices": "bars given to the learner (hand-made or cut)"},
                                  result=result, note=note)
    return Learned(form.id, chosen, result, feats, path)


def learn(idea_id: str, *, log=print, record: bool = True) -> Learned:
    """The learner for one idea's judged version, on the real prices."""
    from machine.data import bars
    from machine.ideas import store
    from machine.ideas.compile import Market, needed_tfs, prepare
    from machine.referee import splits, trials, tune
    form = store.with_chosen_slices(store.load_form(idea_id))     # + slices chosen before any money (L14.1)
    tf = tune.latest(idea_id)
    if tf is None:
        raise LearnError(f"{idea_id} has no tune file yet: python -m machine.report tune {idea_id}")
    chosen = tf["result"]["chosen"]
    if tf.get("parent_idea") and not form.parent_idea:           # the family the research tools found (L14.1): the
        form = form.model_copy(update={"parent_idea": tf["parent_idea"]})   # tune's signals carry it
    opening = splits.family_opening(idea_id, form.parent_idea)
    judged = None
    if opening is not None and (opening["idea_id"], opening.get("version_id")) == (idea_id, chosen):
        done = [f for f in trials.load() if f.get("kind") == "referee" and f.get("idea_id") == idea_id
                and f.get("version_id") == chosen
                and str(f["settings"].get("judged_on", "")).startswith(CHECK_PREFIX)]
        judged = done[0] if done else None                       # the FIRST look at the check years is the one
    start = str(config.TUNE_YEARS[0])
    end = str(config.CHECK_YEARS[-1] if judged is not None else config.TUNE_YEARS[-1])
    if judged is not None:
        guards.check_opened(idea_id, [start, f"{end}-12-31"], version=chosen)    # refuses any other version
    t0 = time.time()
    row = tf["result"]["versions"][chosen]                        # a version on another chart needs that chart too
    tfs = tuple(dict.fromkeys([*TFS, *needed_tfs(prepare(form, row["readings"], row.get("changes") or None)[0])]))
    market = Market.load(start, end, tfs=tfs)
    log(f"bars {start}-{end} read in {time.time() - t0:.0f} s"
        + ("" if judged is not None else " (the hidden years are not open for this version: not read)"))
    news = F.his_news(start, end)
    data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
            "gold_m1_content_sha256": bars.m1_fingerprint(),
            "news": "data/news/forexfactory (red + orange US rows with a minute)",
            "news_rows": int(len(news)),
            "news_sha256": hashlib.sha256(news.to_csv(index=False).encode("utf-8")).hexdigest()}
    return learn_with(form, tf, market, judged=judged, news=news, data=data, record=record, log=log,
                      note=f"L13.1: {idea_id} {chosen} - condition tables and the slice checks on "
                           f"{years_text(config.TUNE_YEARS)}")


# ================================================================================================= L13.2: the model, the pool

def latest_of(kind: str, idea: str, version: str | None = None, folder: Path | None = None) -> dict | None:
    """The newest trial file of this kind for this idea (and version), or None."""
    from machine.referee import trials
    files = [f for f in trials.load(folder) if f.get("kind") == kind and f.get("idea_id") == idea
             and (version is None or f.get("version_id") == version)]
    return files[-1] if files else None


def tune_market(idea_id: str, log=print) -> tuple:
    """The idea's form, its tune file, the market of the TUNE years only (the check years are never read here), his
    news on those years and the data's fingerprints."""
    from machine.data import bars
    from machine.ideas import store
    from machine.ideas.compile import Market, needed_tfs, prepare
    from machine.referee import tune
    form = store.with_chosen_slices(store.load_form(idea_id))     # the same form as learn() (L14.1)
    tf = tune.latest(idea_id)
    if tf is None:
        raise LearnError(f"{idea_id} has no tune file yet: python -m machine.report tune {idea_id}")
    if tf.get("parent_idea") and not form.parent_idea:           # the family the research tools found (L14.1)
        form = form.model_copy(update={"parent_idea": tf["parent_idea"]})
    start, end = str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1])
    row = tf["result"]["versions"][tf["result"]["chosen"]]       # a version on another chart needs that chart too
    tfs = tuple(dict.fromkeys([*TFS, *needed_tfs(prepare(form, row["readings"], row.get("changes") or None)[0])]))
    t0 = time.time()
    market = Market.load(start, end, tfs=tfs)
    news = F.his_news(start, end)
    log(f"bars {start}-{end} read in {time.time() - t0:.0f} s (the tune years only: 2021-2023 are not read)")
    data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
            "gold_m1_content_sha256": bars.m1_fingerprint(),
            "news": "data/news/forexfactory (red + orange US rows with a minute)", "news_rows": int(len(news)),
            "news_sha256": hashlib.sha256(news.to_csv(index=False).encode("utf-8")).hexdigest()}
    return form, tf, market, news, data


def learn_model(idea_id: str, *, record: bool = True, make_filter: bool = True, log=print) -> dict:
    """L13.2 part A on the real prices: the model (machine/learner/model.py) on the judged version's setups and
    trades of 2016-2020, then - only when its check passed - its take/skip filter as a new version, tried and judged
    (machine/learner/takeskip.py). One trial file of kind config.MODEL_KIND (not a try, not an answer)."""
    from machine.learner import model as MD
    from machine.learner import takeskip as TS
    form, tf, market, news, data = tune_market(idea_id, log)
    chosen = tf["result"]["chosen"]
    c, feats = compiled_setups(form, tf, market, news=news, log=log)
    trades = trades_of(tf["result"]["trades_table"])
    t0 = time.time()
    r = MD.analyse(feats, trades)
    log(f"the model: {config.MODEL_SHUFFLES} shuffles in {time.time() - t0:.0f} s - check "
        f"{'passed' if r['check']['passed'] else 'did not pass'} (luck {100 * r['check']['p']:.1f} in 100)")
    r["tune_trial"] = tf["trial_id"]
    r["version_id"] = chosen
    spec = MD.spec_sha256({"base": chosen, "features": r["encoder"]["columns"], "code": MD.code_sha256()})
    if not r["check"]["passed"]:
        r["filter"] = {"made": False, "why": "the model's check did not pass - a filter from a model that found "
                                             "nothing would spend one of the family's tries on noise"}
    elif not make_filter:
        r["filter"] = {"made": False, "why": "not asked for (--no-filter)"}
    else:
        try:
            r["filter"] = TS.make(form, c.signals, r["guesses"], base_version=chosen, spec=spec, tf=form.timeframe,
                                  record=record, log=log)
        except TS.FilterError as err:
            r["filter"] = {"made": False, "why": str(err)}
    path = None
    if record:
        sets = {**MD.settings_used(), "timeframe": form.timeframe, "spec_sha256": spec,
                "signals_sha256": tf["result"]["versions"][chosen]["signals_sha256"],
                "code_sha256": MD.code_sha256(), "features": list(F.COLUMNS),
                "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
                "a_try": "no - the model reads a judged version's trades; its take/skip filter, when made, is a "
                         "version with its own trial files (tester + referee)"}
        path = trial_files.record(kind=config.MODEL_KIND, idea_id=form.id, version_id=chosen,
                                  parent_idea=form.parent_idea, settings=sets, data=data, result=r,
                                  note=f"L13.2: {idea_id} {chosen} - LightGBM with a purged walk-forward on "
                                       f"{years_text(config.TUNE_YEARS)}, SHAP")
    r["trial_path"] = None if path is None else str(path)
    return r


def learn_pooled(idea_id: str, *, record: bool = True, others: bool = True, log=print) -> dict:
    """L13.2 part B on the real prices: this idea's judged trades, every other tested idea's trades and random entries
    of each, together (machine/learner/pooled.py). One trial file of kind config.POOLED_KIND."""
    from machine.learner import pooled as PL
    from machine.referee.replays import Segment
    from machine.tester import engine
    form, tf, market, news, data = tune_market(idea_id, log)
    chosen = tf["result"]["chosen"]
    c, feats = compiled_setups(form, tf, market, news=news, log=log)
    m1 = market.bars("1m")
    book = engine.bars_book(m1, "5m")
    years = tuple(config.TUNE_YEARS)
    table = tf["result"]["trades_table"]
    tt = pd.DataFrame(table["rows"], columns=table["columns"])
    tt = tt[(tt["status"] == "closed") & tt["net_inr"].notna()].copy()
    tt["known_at"] = pd.to_datetime(tt["known_at"])
    t0 = time.time()
    seg = Segment(years_text(years), years, form.timeframe, list(c.signals), m1, book, m1["close"].to_numpy(float))
    rsigs, rtr = PL.random_entries(seg, tt[["signal", "known_at"]], runs=config.POOLED_RANDOM_RUNS,
                                   cap=config.POOLED_RANDOM_MAX, seed=PL.seed_for(f"{idea_id}|{chosen}"), name=idea_id)
    with settings.using(c.resolved):
        rfe = F.build([PL.setup_of(rsigs[s], market) for s in rtr["signal"]], market,
                      fp=settings.fingerprint(c.resolved), form=None, news=news)
    log(f"{idea_id}: {len(rtr)} random entries and their features in {time.time() - t0:.0f} s")
    this = PL.Source(idea_id, f"this idea ({idea_id})", chosen, trades_of(table), feats, PL.trades_frame(rtr), rfe)
    keys = {(d, str(pd.Timestamp(t)), round(float(p), 2)) for d, t, p in zip(tt["direction"], tt["fill_time"],
                                                                                tt["fill_px"])}
    rest = other_sources(market, m1, book, news, form, keys, log) if others else []
    t1 = time.time()
    r = PL.analyse(this, rest)
    log(f"the pooled checks in {time.time() - t1:.0f} s: any entry {len(r['claims_any'])}, only this idea "
        f"{len(r['claims_only'])}")
    r["tune_trial"] = tf["trial_id"]
    r["version_id"] = chosen
    path = None
    if record:
        sets = {"years": list(years), "timeframe": form.timeframe, "rules_sha256": SL.RULES_SHA256,
                "random_runs": config.POOLED_RANDOM_RUNS, "random_max": config.POOLED_RANDOM_MAX,
                "conditions": [x.name for x in PL.MARKET], "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
                "shuffles": config.SLICE_SHUFFLES, "min_trades": config.SLICE_MIN_TRADES, "p_limit": config.P_LIMIT,
                "signals_sha256": tf["result"]["versions"][chosen]["signals_sha256"],
                "code_sha256": PL.code_sha256(),
                "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
                "a_try": "no - it splits trades already made; nothing is chosen"}
        path = trial_files.record(kind=config.POOLED_KIND, idea_id=form.id, version_id=chosen,
                                  parent_idea=form.parent_idea, settings=sets, data=data, result=r,
                                  note=f"L13.2: {idea_id} {chosen} - the pooled view on "
                                       f"{years_text(config.TUNE_YEARS)}")
    r["trial_path"] = None if path is None else str(path)
    return r


def other_sources(market, m1, book, news, form, keys: set, log=print) -> list:
    """Every other tested idea (a trial file of kind "hedging-2x2" outside this idea's family: L19b's entry types),
    made again with a plain stop at the hedge point exactly as L19b made it - the same trades as its trial file (count
    and rupees), else left out and said. A trade already in the pool (the same side, fill minute and fill price)
    counts once. Each with its random entries and the features of every setup."""
    from machine.entries import closers, forms
    from machine.ideas.compile import compile_idea
    from machine.learner import pooled as PL
    from machine.referee import trials as RT
    from machine.referee.replays import Segment
    from machine.tester import engine
    from machine.tester.run import summarize, trades_table
    files = RT.load()
    links = RT.parents(files)
    family = RT.root_of(form.id, links, form.parent_idea)
    wanted = [f for f in files if f.get("kind") == "hedging-2x2" and RT.root_of(f["idea_id"], links) != family]
    if not wanted:
        return []
    fp0 = settings.fingerprint(settings.resolve({}))
    type1 = compile_idea(forms.TYPE1.idea(), market).signals          # the opposite setups of #98 B (L19b)
    years = tuple(config.TUNE_YEARS)
    out = []
    for f in wanted:
        t0 = time.time()
        e = forms.get(f["settings"]["entry"])
        rd = f["settings"].get("readings") or {}
        c = compile_idea(e.idea(), market, readings=rd, trace=True)
        name = f"{e.id} {c.version_id}"
        if c.version_id != f["version_id"]:
            log(f"  {name}: left out - it compiles to {c.version_id}, not {f['version_id']} (the form changed)")
            continue
        cancel = closers.fill_checks(e, c.signals, market, fp0, type1)
        rep = engine.replay(c.signals, None, "5m", config.TRADES_AT_ONCE, book=book, cancel_by=cancel)
        s, want = summarize(rep), f["result"]["his_stop"]
        if s["trades"] != want["trades"] or abs(s["net_inr_total"] - want["net_inr_total"]) > 0.05:
            log(f"  {name}: left out - {s['trades']} trades, {s['net_inr_total']} now against {want['trades']}, "
                f"{want['net_inr_total']} in its trial file (the code changed since L19b)")
            continue
        t = trades_table(rep.trades)
        t = t[(t["status"] == "closed") & t["net_inr"].notna()].copy()
        k = [(d, str(pd.Timestamp(ft)), round(float(p), 2)) for d, ft, p in zip(t["direction"], t["fill_time"],
                                                                                   t["fill_px"])]
        new = np.array([x not in keys for x in k], dtype=bool)
        dup = int((~new).sum())
        keys.update(k)
        t = t[new]
        if not len(t):
            log(f"  {name}: every one of its trades is already in the pool")
            continue
        t["known_at"] = pd.to_datetime(t["known_at"])
        trades = PL.trades_frame(t)
        ids = set(trades.index)
        with settings.using(c.resolved):
            fp = settings.fingerprint(c.resolved)
            fe = F.build([x for x in F.setups_from(c, market) if x.id in ids], market, fp=fp, form=c.form, news=news)
            seg = Segment(years_text(years), years, "5m", list(c.signals), m1, book, m1["close"].to_numpy(float))
            rsigs, rtr = PL.random_entries(seg, t[["signal", "known_at"]], runs=config.POOLED_RANDOM_RUNS,
                                           cap=config.POOLED_RANDOM_MAX, seed=PL.seed_for(name),
                                           name=f"{e.id}.{c.version_id}")        # two readings: two sets of ids
            rfe = F.build([PL.setup_of(rsigs[x], market) for x in rtr["signal"]], market, fp=fp, form=None,
                          news=news)
        notes = [f"{dup} trades already in the pool left out"] if dup else []
        out.append(PL.Source(e.id, e.title + (f" ({', '.join(f'{a}={b}' for a, b in rd.items())})" if rd else ""),
                             c.version_id, trades, fe, PL.trades_frame(rtr), rfe, notes))
        log(f"  {name}: {len(trades)} trades ({dup} already in the pool), {len(rtr)} random entries, "
            f"{time.time() - t0:.0f} s")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.learner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("idea", nargs="?", help="the idea (idea-001); leave out with --conditions")
    ap.add_argument("--conditions", action="store_true", help="print every condition a slice may name, and the rules")
    ap.add_argument("--no-record", action="store_true", help="do not write a trial file (a dry run)")
    ap.add_argument("--model", action="store_true", help="L13.2 A: LightGBM + SHAP (and its take/skip filter version "
                                                         "when the model's check passes)")
    ap.add_argument("--no-filter", action="store_true", help="with --model: never make the filter version")
    ap.add_argument("--pooled", action="store_true", help="L13.2 B: every tested idea's trades + random entries")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.conditions:
        print(CN.describe())
        print()
        print(SL.RULES)
        return 0
    if not a.idea:
        ap.error("name the idea (idea-001)")
    if a.model or a.pooled:
        if a.model:
            from machine.learner import model as MD
            r = learn_model(a.idea, record=not a.no_record, make_filter=not a.no_filter)
            print(MD.text(r, a.idea, r["version_id"]))
            print("\nThe take/skip filter: " + (f"version {r['filter']['version_id']} - {r['filter']['answer']}"
                                                if r["filter"].get("made") else f"not made - {r['filter']['why']}"))
            if r.get("trial_path"):
                print(f"Trial file: {r['trial_path']}")
        if a.pooled:
            from machine.learner import pooled as PL
            r = learn_pooled(a.idea, record=not a.no_record)
            print(PL.text(r))
            if r.get("trial_path"):
                print(f"Trial file: {r['trial_path']}")
        return 0
    print(learn(a.idea, record=not a.no_record).text())
    return 0
