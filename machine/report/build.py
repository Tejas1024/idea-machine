"""The report (L10): one idea's answer in rupees, built ONLY from the referee's files.

    .venv\\Scripts\\python -m machine.report build idea-001            the text, the picture and report.html
    .venv\\Scripts\\python -m machine.report build idea-001 --send     ... and the Telegram lines into the outbox

Where every number comes from (nothing here adds up money of its own - the test pins it):
  - the judgement: the referee's trial file (kind "referee") - the check years 2021-2023, opened ONCE per idea family
    (the FIRST judgement of those years counts; a later one is a second look and is named as one), or walk-forward
    inside 2016-2020 for a later version;
  - the choice: the referee's tune file (kind "referee-tune") - every version on 2016-2020 and the one chosen;
  - the form (what the idea IS: its title, chart, hedge or stop, its slices chosen before), and the rules text
    (machine/referee/verdict.py: its fingerprint must match the one in the trial file);
  - "works best in / fails in": the learner's trial file (kind "learner", machine/learner/run.py, L13.1) for the judged
    version - the chosen slices' tables and checks on 2016-2020, the conditions only looked at. No learner file yet:
    the line says "not measured yet".
  - "what a model learned": the model's trial file (kind "learner-model", L13.2 A) for the judged version - its check,
    what it leaned on (SHAP), and its take/skip filter: the referee's walk-forward judgement of that version (its own
    trial file) when it was made. "Any entry or only this idea": the pooled view's trial file (kind "learner-pooled",
    L13.2 B). No file yet: "not measured yet".
Out: plan/work/ideas/<idea>/report.html (every trade, easy English), report-years.jpg (rupees by year, the hidden
years shaded), and a Telegram message (the picture, then the text). Every word passes check_words (#113: easy
English, no code words; a hard word only with its meaning in brackets).
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from machine import config  # noqa: E402
from machine.charts import draw as D  # noqa: E402
from machine.data import clock  # noqa: E402
from machine.ideas import store  # noqa: E402
from machine.ideas.form import Idea  # noqa: E402
from machine.learner import model as model_words  # noqa: E402
from machine.learner import pooled as pooled_words  # noqa: E402
from machine.learner import run as learner  # noqa: E402
from machine.learner import slices as slice_rules  # noqa: E402
from machine.referee import project, splits, trials, verdict  # noqa: E402
from machine.referee.tune import TUNE_KIND  # noqa: E402

plt.rcParams["text.parse_math"] = False           # lesson B8: "$5" is five dollars, not maths

PICTURE, PAGE = "report-years.jpg", "report.html"
CHECK_PREFIX = "the check years"                  # judge.judge_check's judged_on starts with this

# ---- easy English (#113, CLAUDE.md) --------------------------------------------------------------------------------
# Code words he never used (CLAUDE.md #113 rule 2; the same list as machine/charts/lint.py) and the maths shorthand
# his judgement guard bans ("t = 0.41", "p-value", "n = 25").
BANNED = ("leftover", "arm", "arms", "armed", "verdict", "veto", "causal", "placebo", "mirror", "baseline", "basket")
BANNED_PATTERNS = (
    (re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.IGNORECASE), "a code word he never used"),
    (re.compile(r"(?<![A-Za-z0-9_])R(?![A-Za-z0-9_&])"), "'R' - say the distance in rupees or dollars"),
    (re.compile(r"\bp[- ]?value\b|\bp\s*[<=>]\s*0", re.IGNORECASE), "a p-value - say 'luck gives this N times in 100'"),
    (re.compile(r"(?<![A-Za-z])[tn]\s*=\s*[-0-9]"), "'t =' / 'n =' - say it in words"),
)
# Hard words allowed only with their meaning in brackets right after them ("... (Deflated Sharpe = ...)" is not
# enough: the MEANING must come first, the word in brackets, or the word with its meaning in brackets right after).
HARD = re.compile(r"Deflated Sharpe|Sharpe|PBO|bootstrap|walk-forward|percentile|LightGBM|SHAP")


class ReportError(ValueError):
    """The report refused: no referee answer yet, the check years judged for another version, rules changed."""


def check_words(text: str) -> list[str]:
    """Every easy-English problem in a text: code words, maths shorthand, a hard word without its meaning."""
    out = []
    for pat, why in BANNED_PATTERNS:
        for m in pat.finditer(text):
            out.append(f"{why}: '{m.group(0)}' in '...{text[max(0, m.start() - 30):m.end() + 30]}...'")
    for m in HARD.finditer(text):
        before = text[:m.start()]
        inside_brackets = before.rfind("(") > before.rfind(")")          # "the meaning (Deflated Sharpe)"
        meaning_after = text[m.end():].startswith((" (", "("))            # "Deflated Sharpe (the meaning)"
        if not (inside_brackets or meaning_after):
            out.append(f"hard word '{m.group(0)}' without its meaning in brackets: "
                       f"'...{text[max(0, m.start() - 30):m.end() + 30]}...'")
    return out


# ================================================================================================================
# the facts: read from the referee's files, nothing else
# ================================================================================================================

@dataclass
class Facts:
    form: Idea
    judge: dict                    # the referee's trial file (kind "referee")
    tune: dict | None              # the referee's tune file for the judged version (kind "referee-tune")
    mode: str                      # "check" (2021-2023, opened once) or "walk-forward" (inside 2016-2020)
    opening: dict | None           # check-opened.json of the family
    warnings: list[str]
    judge_path: str
    tune_path: str | None
    project: dict                  # machine.referee.project.summary(): the whole project so far, from its trial files
    learner: dict | None = None    # the learner's trial file for the judged version (kind "learner", L13.1)
    model: dict | None = None      # the model's trial file for the judged version (kind "learner-model", L13.2 A)
    pooled: dict | None = None     # the pooled view's trial file (kind "learner-pooled", L13.2 B)
    filter_judge: dict | None = None   # the referee's judgement of the model's take/skip filter version (L13.2 A)

    @property
    def idea(self) -> str:
        return self.form.id

    @property
    def r(self) -> dict:
        return self.judge["result"]

    @property
    def hedged(self) -> bool:
        return self.form.exit.kind == "hedge"


def _path_of(body: dict, folder: Path) -> str:
    name = next((p for p in folder.glob(f"{body['trial_id']}-*.json")), None)
    if name is None:
        return f"plan/work/trials/{body['trial_id']}-..."
    try:
        return name.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(name)


def facts(idea_id: str) -> Facts:
    """The idea's report facts, read from its trial files. Refuses when the referee has not answered yet, or when
    the check years were judged for a version they were not opened for (CHECK once)."""
    form = store.with_chosen_slices(store.load_form(idea_id))     # + slices chosen before any money (L14.1)
    folder = Path(config.TRIALS_DIR)
    files = trials.load(folder)
    # a take/skip filter version (L13.2) has its own judgement: shown next to the idea's, never AS the idea's
    filters = {(f["result"].get("filter") or {}).get("version_id") for f in files
               if f.get("kind") == config.MODEL_KIND and f.get("idea_id") == idea_id} - {None}
    judged = [f for f in files if f.get("kind") == "referee" and f.get("idea_id") == idea_id
              and f.get("version_id") not in filters]
    if not judged:
        raise ReportError(f"{idea_id} has no referee answer yet (no trial file of kind 'referee'): run "
                          f"python -m machine.report tune {idea_id}, then check {idea_id}")
    warnings: list[str] = []
    opening = splits.family_opening(idea_id, form.parent_idea)
    checks = [f for f in judged if str(f["settings"].get("judged_on", "")).startswith(CHECK_PREFIX)]
    if checks:
        judge = checks[0]                                   # the FIRST look at the check years is the one that counts
        mode = "check"
        if opening is None:
            raise ReportError(f"{idea_id}: a judgement on the check years exists but no check-opened.json - the "
                              "check years must be opened (once) before they are judged")
        if (opening["idea_id"], opening.get("version_id")) != (idea_id, judge["version_id"]):
            raise ReportError(f"{idea_id}: the check years were opened for {opening['idea_id']} version "
                              f"{opening.get('version_id')}, but judged for version {judge['version_id']} - they open "
                              "ONCE, for one version")
        if len(checks) > 1:
            warnings.append(f"The hidden years were judged {len(checks)} times for this idea. Only the first "
                            f"({judge['made_at'][:16]}) counts here; the later ones were a second look.")
    else:
        judge = judged[-1]
        mode = "walk-forward"
    if judge["settings"].get("rules_sha256") != verdict.RULES_SHA256:
        raise ReportError(f"{idea_id}: the rules in machine/referee/verdict.py changed after this judgement "
                          f"({judge['made_at'][:10]}) - judge again under the rules printed now")
    tunes = [f for f in files if f.get("kind") == TUNE_KIND and f.get("idea_id") == idea_id
             and f["result"].get("chosen") == judge["version_id"]]
    tune = tunes[-1] if tunes else None
    if tune is None:
        warnings.append("No tune file for this version: the money on 2016-2020 is not shown.")
    elif mode == "check" and opening and str(tune["made_at"]) > str(opening.get("opened_at", "")):
        warnings.append("The version was chosen AFTER the hidden years were opened - the choice may have seen them.")
    learned = learner.latest(idea_id, judge["version_id"], folder)
    if learned is not None and learned["settings"].get("rules_sha256") != slice_rules.RULES_SHA256:
        warnings.append("The slice rules changed after the learner ran: run python -m machine.learner "
                        f"{idea_id} again. Its old result is not shown.")
        learned = None
    if learned is not None and tune is not None and learned["result"].get("tune_trial") != tune["trial_id"]:
        warnings.append("The learner ran on another tune of this idea: its result is not shown.")
        learned = None
    model = learner.latest_of(config.MODEL_KIND, idea_id, judge["version_id"], folder)
    pooled = learner.latest_of(config.POOLED_KIND, idea_id, judge["version_id"], folder)
    if pooled is not None and pooled["settings"].get("rules_sha256") != slice_rules.RULES_SHA256:
        warnings.append("The slice rules changed after the pooled view ran: run python -m machine.learner "
                        f"{idea_id} --pooled again. Its old result is not shown.")
        pooled = None
    for name, got in (("model", model), ("pooled view", pooled)):
        if got is not None and tune is not None and got["result"].get("tune_trial") != tune["trial_id"]:
            warnings.append(f"The {name} ran on another tune of this idea: its result is not shown.")
    model = None if model is None or (tune is not None and model["result"].get("tune_trial") != tune["trial_id"]) \
        else model
    pooled = None if pooled is None or (tune is not None and pooled["result"].get("tune_trial") != tune["trial_id"]) \
        else pooled
    filter_judge = None
    made = (model or {}).get("result", {}).get("filter") or {}
    if made.get("made"):
        fj = [f for f in files if f.get("kind") == "referee" and f.get("idea_id") == idea_id
              and f.get("version_id") == made["version_id"]]
        filter_judge = fj[0] if fj else None
        if filter_judge is None:
            warnings.append(f"The take/skip filter version {made['version_id']} has no referee file.")
    return Facts(form, judge, tune, mode, opening, warnings, _path_of(judge, folder),
                 None if tune is None else _path_of(tune, folder), project.summary(folder), learned, model, pooled,
                 filter_judge)


# ================================================================================================================
# words and numbers (formatting only)
# ================================================================================================================

def inr(x) -> str:
    """Whole rupees with a sign (-₹1,234 · ₹790); a value that rounds to 0 is ₹0, never "₹-0"."""
    if x is None:
        return "no trades"
    r = round(x)
    return f"-₹{-r:,}" if r < 0 else f"₹{abs(r):,}"


def idea_name(idea_id: str) -> str:
    """idea-001 -> Idea #1."""
    m = re.fullmatch(r"idea-(\d+)", idea_id)
    return f"Idea #{int(m.group(1))}" if m else idea_id


def family_line(f: Facts) -> str:
    """The report's "family of": the form's parent_idea, or the older idea the research tools found (likhith.md
    section D - the same blocks, chart and entry; its trial files carry it, the form does not). Named, never a code
    id (#113)."""
    parent = f.form.parent_idea or f.judge.get("parent_idea")
    if not parent:
        return "new (no older idea)"
    from machine.entries import forms
    e = forms.ENTRY_TYPES.get(parent)
    name = e.title if e is not None else idea_name(parent)
    if f.form.parent_idea:
        return name
    out = f"{name} (the same blocks, chart and entry: a version of it)"
    if f.mode == "walk-forward":
        out += (f" - the hidden years {years_text(config.CHECK_YEARS)} were opened once for that family, so this one "
                f"is judged inside {years_text(config.TUNE_YEARS)}")
    return out


def of100(a: int, b: int) -> str:
    return "no trades" if not b else f"{round(100 * a / b)} of 100"


def years_text(years) -> str:
    ys = sorted(int(y) for y in years)
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"


TF_WORDS = {"1m": "1-minute", "5m": "5-minute", "15m": "15-minute", "1h": "1-hour", "4h": "4-hour", "1d": "daily",
            "1w": "weekly"}


def to_him(text: str) -> str:
    """The readings were written about him ("his written rule"); the report speaks TO him ("your written rule")."""
    return re.sub(r"\bhis\b", "your", text)


def about_version(f: Facts) -> str:
    t = f.tune
    if t is None:
        return f"version {f.judge['version_id']}"
    return to_him(t["result"]["versions"][f.judge["version_id"]]["about"]) or f"version {f.judge['version_id']}"


def _judged_years(f: Facts) -> list[int]:
    return [int(y) for y in f.judge["settings"]["years"]]


def _where_judged(f: Facts) -> str:
    if f.mode == "check":
        return f"{years_text(_judged_years(f))}, hidden while choosing"
    return f"judged year by year inside {years_text(_judged_years(f))}, each year chosen only on the years before it"


def outcome_lines(f: Facts) -> list[str]:
    """How the trades ended. A hedged idea: closed in profit / at a loss / still open (never 'stop'); its hedge
    tested as a plain stop says so. A stop idea: hit the target before the stop."""
    o = f.r["outcomes"]
    filled = o["closed"] + o["open_at_end"]
    if f.hedged:
        return [f"Then (your hedge idea): closed in profit {of100(o['in_profit'], filled)} · closed at a loss "
                f"{of100(o['at_a_loss'], filled)} · still open at the end {of100(o['open_at_end'], filled)} "
                f"({o['in_profit']} · {o['at_a_loss']} · {o['open_at_end']} of {filled} trades)",
                f"  How it was tested: the hedge point was treated as a plain stop (every idea is judged this way; "
                f"your hedge is tested apart). Reached the target before the hedge point: "
                f"{of100(o['target_first'], o['closed'])}"
                f" ({o['target_first']} of {o['closed']} closed trades)."]
    return [f"Then: hit the target before the stop {of100(o['target_first'], o['closed'])} "
            f"({o['target_first']} of {o['closed']} closed trades; still open at the end, not counted: "
            f"{o['open_at_end']})"]


def learner_lines(f: Facts, counts: bool = True) -> list[str]:
    """"Works best in / fails in" (L13.1): the learner's line for the slices chosen before the test, then the trades
    in each part of each slice. counts=False leaves the part counts out (Telegram takes 4096 characters)."""
    if f.learner is None:
        return [f"Works best in / fails in: not measured yet (the learner has not run: python -m machine.learner "
                f"{f.idea})"]
    res = f.learner["result"]
    lines = [learner.works_line(res)]
    if res["slices"]:
        if counts:
            lines.append(f"Trades in each part of each slice, {learner.years_text(res['years'])} (* = too few trades "
                         "to test that part):")
            lines += learner.slice_lines(res)
        else:
            lines.append(f"Trades in each part of each slice: on the report page ({PAGE})")
    return lines


def looked_at(f: Facts) -> int:
    return 0 if f.learner is None else int(f.learner["result"]["counts"]["looked_at"])


MODEL_WORD = "What a model learned (LightGBM, a program that learns from tables"


def model_lines(f: Facts, short: bool = False) -> list[str]:
    """"What a model learned" (L13.2 A): the model's check, what it leaned on (SHAP), its take/skip filter.
    short=True: one line for the check (Telegram takes 4096 characters; the rest is on the page)."""
    if f.model is None:
        return [f"{MODEL_WORD}): not measured yet (python -m machine.learner {f.idea} --model)"]
    r = f.model["result"]
    c = r["check"]
    if short:
        found = "it found something" if c["passed"] else "it found nothing it could use"
        lines = [f"{MODEL_WORD}; each year guessed by a model made only from the years before): {found} - luck gives "
                 f"its gap {100 * c['p']:.1f} times in 100, it needs under {round(100 * config.P_LIMIT)} (details on "
                 f"{PAGE})"]
    else:
        lines = [f"{MODEL_WORD}) — {model_words.check_line(r)}", model_words.named_line(r)]
    return lines + [filter_line(f)]


def filter_line(f: Facts) -> str:
    """The model's take/skip filter: the referee's answer for that version, or why it was not made."""
    made = f.model["result"].get("filter") or {}
    if not made.get("made"):
        return f"No take/skip filter was made from it: {made.get('why', 'not asked for')}."
    j = f.filter_judge
    if j is None:
        return f"Its take/skip filter (version {made['version_id']}) has no judgement yet."
    s = j["result"]["summary"]
    return (f"Its take/skip filter as a new version ({made['version_id']}, try {made['try']} of {made['budget']}): "
            f"{j['result']['answer']['word'].upper()} — {inr(s['per_trade'])} a trade on {s['trades']} trades, judged "
            f"year by year on {years_text(j['settings']['years'])} (each year's setups chosen by a model made before "
            "it)")


def pooled_lines(f: Facts, short: bool = False) -> list[str]:
    """"Any entry or only this idea" (L13.2 B). short=True: the counts only (the claims are on the page)."""
    if f.pooled is None:
        return [f"Any entry or only this idea: not measured yet (python -m machine.learner {f.idea} --pooled)"]
    r = f.pooled["result"]
    if short:
        a, o = len(r["claims_any"]), len(r["claims_only"])
        return [f"Any entry or only this idea ({r['counts']['tested']} market conditions; your trades against "
                f"{r['random_trades']:,} random entries): helps or hurts ANY entry: "
                f"{f'{a} (on the page)' if a else 'nothing passed'} · only THIS idea: "
                f"{f'{o} (on the page)' if o else 'nothing passed'}"]
    return [pooled_words.head_line(r), pooled_words.any_line(r), pooled_words.only_line(r)]


def summary_lines(f: Facts, slice_counts: bool = True, short: bool = False) -> list[str]:
    """The report's lines, in the order of the plan (likhith.md L10) - the same in Telegram and in report.html."""
    r, form = f.r, f.form
    p, fk, rnd, wb, ex = r["path"], r["fake_levels"], r["random"], r["without_best"], r["extra_lines"]
    counts, proj, seen = r["counts"], f.project, r["already_seen"]
    parent = family_line(f)
    stop_word = "hedge point" if f.hedged else "stop"
    lines = [
        f"{idea_name(form.id)} — {form.title}",
        f"{TF_WORDS.get(form.timeframe, form.timeframe)} chart · {' and '.join(form.directions)} · family of: {parent}",
        f"Tested: {about_version(f)}",
        "₹ = after costs, 0.01 lot, ₹40 per position all-in (your #54, #115). Clock: MT5 server time.",
        "",
        f"ANSWER: {r['answer']['word'].upper()} — {r['answer']['why']}",
        "",
        f"Came to the level after the setup: {of100(p['came_back'], p['had_chance'])} times ({p['came_back']} of "
        f"{p['had_chance']} setups that had their full chance)",
        *outcome_lines(f),
    ]
    if fk.get("applies"):
        lines.append(f"Fake levels (the level moved up and down, the same distances to the target and the "
                     f"{stop_word}): reached the target first {of100(fk['target_first'], fk['trades'])} "
                     f"({fk['target_first']} of {fk['trades']} trades on {len(fk['steps'])} fake levels)")
    else:
        lines.append(f"Fake levels: do not apply — {fk.get('why', 'no level')}")
    n_versions = len(counts["family_versions"])
    if f.tune is not None:
        t = f.tune["result"]
        money = (f"Money after costs, 0.01 lot: {inr(t['a_year']['net_inr'])} a year "
                 f"({years_text(f.tune['settings']['years'])}, the best of {len(t['versions'])} versions there) · "
                 f"{inr(r['a_year']['net_inr'])} a year ({_where_judged(f)})")
    else:
        money = f"Money after costs, 0.01 lot: {inr(r['a_year']['net_inr'])} a year ({_where_judged(f)})"
    lines.append(money)
    if ex:
        lines.append(f"  With the broker's swap: {inr(ex['with_swap_a_year'])} a year · nights held "
                     f"{ex['nights_held']} · with the real spread instead of ₹40: "
                     f"{inr(ex['with_spread_a_year']) if ex['with_spread_a_year'] is not None else 'not known'} a year "
                     f"({years_text(_judged_years(f))})")
    wy = r["worst_year"]
    worst = f"Worst year: {inr(wy['net_inr'])} ({wy['year']}, {wy['trades']} trades)"
    if f.tune is not None:
        tw = f.tune["result"]["worst_year"]
        worst += f" · of {years_text(f.tune['settings']['years'])}: {inr(tw['net_inr'])} ({tw['year']})"
    lines += [
        f"{worst} · without the best {wb['left_out']} trades: {inr(wb['per_trade'])} a trade",
        f"Random entries: it beats {rnd['beaten']} of {rnd['runs_with_trades']} (random times, the same distances "
        f"and cost) · setups skipped (one trade at a time): {p['skipped_trade_on']}",
        f"Versions tried: {n_versions} of {counts['budget']} · slices chosen before: {len(form.slices)} of "
        f"{config.SLICE_BUDGET} · looked at, not tested: {looked_at(f)}",
        f"Smallest edge this test could see: {_smallest(r['power'])}",
        _seen_line(seen),
        f"Hidden years (2021–2023) opened in the whole project: {proj['check_opened']} "
        f"{'time' if proj['check_opened'] == 1 else 'times'}",
        *learner_lines(f, slice_counts),
        *model_lines(f, short),
        *pooled_lines(f, short),
        f"Project so far: {proj['families_judged']} {'idea' if proj['families_judged'] == 1 else 'ideas'} "
        f"judged, {proj['works']} \"works\", about {proj['false_works_expected']:g} \"works\" expected by luck alone",
    ]
    lines += ["", "The 6 checks (the rules were fixed before any result was seen):"]
    for c in r["answer"]["checks"]:
        lines.append(f"{c['number']}. {c['name']}: {'yes' if c['passed'] else 'NO'} — {c['shown']}")
    lines += [f"Note: {w}" for w in f.warnings]
    return lines


def _smallest(power: dict) -> str:
    s = power.get("smallest_edge_seen")
    n = power.get("own_n")
    if s is None:
        edges = [int(e) for e in next(iter(power["rows"].values()))] if power.get("rows") else [100]
        return f"more than ₹{max(edges)} a trade (with {n} trades)"
    return f"₹{s} a trade (caught at least half the time with {n} trades)"


def _seen_line(seen: dict) -> str:
    if seen["seen"]:
        detail = sorted(seen.get("rows_detail", []), key=lambda d: -d["tries"])
        rows = "; ".join(f"{d['what'] or d['idea_key']} ({d['tries']} tries, {d['years'][0]}–{d['years'][1]})"
                         for d in detail[:3])
        more = (f"; and {len(detail) - 3} more old tests of it" if len(detail) > 3 else "")
        return (f"Already seen by the old project: YES — {len(detail)} old tests, {seen['tries']} tries in all: "
                f"{rows}{more}. These years are not fresh: only new prices can judge it.")
    near = seen.get("closest") or []
    if not near:
        return "Already seen by the old project: no."
    rel = "; ".join(f"{d['what'] or d['idea_key']} ({d['tries']} tries, {d['years'][0]}–{d['years'][1]})"
                    for d in near)
    wider = seen.get("wider") or {}
    more = (f". In all, {wider['rows']} old tests ({wider['tries']} tries) used only parts of this idea"
            if wider.get("rows") else "")
    return f"Already seen by the old project: no. Nearest old tries (not the same idea): {rel}{more}"


def telegram_text(f: Facts) -> str:
    tail = ["", f"Every trade, every year and the full rules: plan/work/ideas/{f.idea}/{PAGE}"]
    # Telegram's limit is 4096 characters: first the part counts of each slice stay on the page, then the model's
    # and the pooled view's details (their answers stay in the text)
    for counts, short in ((True, False), (False, False), (False, True)):
        text = "\n".join(summary_lines(f, slice_counts=counts, short=short) + tail)
        if len(text) <= 4096:
            return text
    raise ReportError(f"the Telegram text is {len(text)} characters; Telegram takes 4096")


# ================================================================================================================
# the picture: rupees by year, the hidden years shaded
# ================================================================================================================

def year_rows(f: Facts) -> list[dict]:
    """Every year shown: the chosen version on the tune years (from the tune file) and the judged years."""
    rows = []
    if f.tune is not None:
        for y, v in f.tune["result"]["per_year"].items():
            rows.append({"year": int(y), "net_inr": v["net_inr"], "trades": v["trades"], "hidden": False})
    judged = {int(y): v for y, v in f.r["per_year"].items()}
    have = {r["year"] for r in rows}
    for y, v in judged.items():
        if y in have:
            continue                     # walk-forward years are inside the tune years: the judged numbers win below
        rows.append({"year": y, "net_inr": v["net_inr"], "trades": v["trades"], "hidden": f.mode == "check"})
    if f.mode == "walk-forward":
        for r in rows:
            if r["year"] in judged:
                r.update(net_inr=judged[r["year"]]["net_inr"], trades=judged[r["year"]]["trades"])
    return sorted(rows, key=lambda r: r["year"])


GREEN, RED, INK, MUTED, SHADE = D.GREEN, D.RED, D.INK, D.MUTED, "#e9eef6"


def picture(f: Facts, path: Path) -> dict:
    """One picture: rupees after cost for each year, the hidden years shaded. Returns what was drawn."""
    rows = year_rows(f)
    years = [r["year"] for r in rows]
    vals = [r["net_inr"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=D.DPI)
    texts = []

    def put(*a, **k):
        texts.append(str(a[2]) if len(a) > 2 else "")
        return ax.text(*a, **k)

    x = list(range(len(years)))
    top = max([0.0, *vals])
    bot = min([0.0, *vals])
    span = (top - bot) or 1.0
    hidden = [i for i, r in enumerate(rows) if r["hidden"]]
    if hidden:
        ax.axvspan(min(hidden) - 0.5, max(hidden) + 0.5, color=SHADE, zorder=0)
        put((min(hidden) + max(hidden)) / 2, top + 0.22 * span, "hidden while choosing", ha="center", va="bottom",
            fontsize=D.F_TAG, color=INK, fontweight="bold")
    chosen = [i for i, r in enumerate(rows) if not r["hidden"]]
    if chosen and f.tune is not None:
        put((min(chosen) + max(chosen)) / 2, top + 0.22 * span, "used to choose the version", ha="center",
            va="bottom", fontsize=D.F_TAG, color=MUTED)
    ax.bar(x, vals, width=0.62, color=[GREEN if v > 0 else RED for v in vals], zorder=2)
    ax.axhline(0, color=INK, linewidth=1.0, zorder=3)
    for i, r in enumerate(rows):
        v = r["net_inr"]
        put(i, v + (0.03 * span if v >= 0 else -0.03 * span), inr(v), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=D.F_NAME, color=INK, fontweight="bold")
    ax.set_xticks(x, [f"{r['year']}\n{r['trades']} trades" for r in rows], fontsize=D.F_AXIS, color=INK)
    texts += [f"{r['year']} {r['trades']} trades" for r in rows]
    ax.set_ylim(bot - 0.18 * span, top + 0.36 * span)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="y", labelsize=D.F_AXIS, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=D.GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    title = f"{idea_name(f.form.id)} · money after costs, each year"
    sub = (f"0.01 lot · ₹40 a position, all-in · {TF_WORDS.get(f.form.timeframe, f.form.timeframe)} chart · "
           f"a trade counts in the year it closed · answer: {f.r['answer']['word']}")
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=D.F_HEAD, color=INK, fontweight="bold")
    fig.text(0.06, 0.905, sub, ha="left", fontsize=D.F_SUB, color=MUTED)
    texts += [title, sub]
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.14)
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
        raise ReportError("the picture's words: " + "; ".join(problems))
    return {"path": path, "texts": texts, "pixel_sha256": D.pixel_sha256(path), "years": years,
            "hidden": [rows[i]["year"] for i in hidden]}


# ================================================================================================================
# report.html: easy English, every trade
# ================================================================================================================

HOW_CLOSED = {"target": "target", "stop": "stop", "past at fill": "closed at once (the fill was already past it)",
              "target+stop": "part at the target, the rest at the stop", "open at end": "still open at the end"}


def _closed_words(reason: str | None, hedged: bool) -> str:
    w = HOW_CLOSED.get(reason or "", reason or "")
    return w.replace("stop", "hedge point (tested as a stop)") if hedged else w


def _ist(ts: str | None) -> str:
    if not ts:
        return ""
    t = clock.to_ist(pd.Timestamp(ts))
    return t.strftime("%d %b %Y %H:%M")


def trade_table(table: dict, hedged: bool) -> str:
    cols = table["columns"]
    head = ("<tr><th>#</th><th>buy / sell</th><th>setup known (India time)</th><th>filled (India time)</th>"
            f"<th>fill price $</th><th>{'hedge point' if hedged else 'stop'} $</th><th>target $</th>"
            "<th>closed (India time)</th><th>close price $</th><th>how it closed</th><th>gold $ move</th>"
            "<th>₹ after cost</th></tr>")
    body = []
    for k, row in enumerate(table["rows"], 1):
        d = dict(zip(cols, row))
        net = d["net_inr"]
        cls = "" if net is None else (" class='win'" if net > 0 else " class='loss'")
        reason = d["closed_by"] if d["status"] == "closed" else "open at end"
        body.append(
            f"<tr><td>{k}</td><td>{d['direction']}</td><td>{_ist(d['known_at'])}</td><td>{_ist(d['fill_time'])}</td>"
            f"<td>{_num(d['fill_px'])}</td><td>{_num(d['stop'])}</td><td>{html.escape(str(d['targets'] or ''))}</td>"
            f"<td>{_ist(d['exit_time'])}</td><td>{_num(d['exit_px'])}</td>"
            f"<td>{html.escape(_closed_words(reason, hedged))}{' *' if d['ambiguous'] else ''}</td>"
            f"<td>{_usd(d['gold_usd'])}</td>"
            f"<td{cls}>{'not counted' if net is None else inr(net)}</td></tr>")
    return f"<table class='trades'>{head}{''.join(body)}</table>"


def _num(x) -> str:
    return "" if x is None else f"{x:,.2f}"


def _usd(x) -> str:
    return "" if x is None else (f"-${-x:,.2f}" if round(x, 2) < 0 else f"+${x:,.2f}")


def _condition_table(t: dict, hidden: list | None = None) -> str:
    """One condition's table (the learner's file): each part's trades, rupees a trade, the interval, the two halves
    of 2016-2020, and - for a chosen slice whose hidden years are open - the same part on 2021-2023 (shown only)."""
    esc = html.escape
    halves = [h["years"].replace("-", "–") for h in t["groups"][0]["halves"]] if t["groups"] else []
    shown = {x["group"]: x for x in (hidden or [])}
    head = ("<tr><th>part</th><th>trades</th><th>₹ a trade</th><th>when the weeks are drawn again: the middle "
            f"{round(100 * config.SLICE_INTERVAL)} of 100</th>"
            + "".join(f"<th>{h}: ₹ a trade (trades)</th>" for h in halves)
            + ("<th>hidden years 2021–2023: ₹ a trade (trades) — shown, not tested</th>" if shown else "") + "</tr>")
    rows = []
    for g in t["groups"]:
        iv = g["interval"]
        cells = [esc(g["group"]) + ("" if g["enough"] else " *"), str(g["trades"]),
                 inr(g["per_trade"]) if g["trades"] else "no trades",
                 f"{inr(iv[0])} to {inr(iv[1])}" if iv else "—"]
        cells += [f"{inr(h['per_trade'])} ({h['trades']})" if h["trades"] else "no trades" for h in g["halves"]]
        if shown:
            h = shown.get(g["group"])
            cells.append(f"{inr(h['per_trade'])} ({h['trades']})" if h and h["trades"] else "no trades")
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    if t.get("not_known"):
        rows.append(f"<tr><td>not known</td><td>{t['not_known']}</td>" + "<td></td>" * (2 + len(halves) + bool(shown))
                    + "</tr>")
    return f"<div class='wrap'><table>{head}{''.join(rows)}</table></div>"


def slices_html(f: Facts) -> str:
    """The report page's part about "works best in / fails in": the slices chosen before the test with their tables
    and checks, then every condition only looked at, then the slice rules."""
    esc = html.escape
    if f.learner is None:
        rows = "".join(f"<tr><td>{esc(s.how)}</td><td>{esc(s.why)}</td><td>not measured yet</td></tr>"
                       for s in f.form.slices)
        return (f"<h2>The {len(f.form.slices)} slices chosen before any result</h2>"
                f"<div class='wrap'><table><tr><th>split by</th><th>why</th><th>result</th></tr>{rows}</table></div>"
                "<p class='muted'>Works best in / fails in: not measured yet (the learner has not run).</p>")
    res = f.learner["result"]
    years = learner.years_text(res["years"])
    out = [f"<h2>Works best in / fails in — the {res['counts']['slices_chosen']} slices chosen before any result</h2>",
           f"<p>{esc(learner.works_line(res))}</p>",
           f"<p class='muted'>Split: the tested version's {res['trades']} trades closed in {years} (the years used to "
           "choose it), ₹ after cost at 0.01 lot. A part with * has too few trades to be tested. Sessions and hours "
           "are India time; weekdays and “today” are the broker's day.</p>"]
    for s in res["slices"]:
        out.append(f"<h3>{esc(s['label'])}</h3><p class='muted'>{esc(s['how'])} — {esc(s['why'])}</p>")
        if not s["measured"]:
            out.append(f"<p>Not measured: {esc(s['why_not'])}.</p>")
            continue
        out.append(_condition_table(s, s.get("check_years")))
        out.append(f"<p>{esc(learner.check_text(s['test']))}</p>")
    n = res["counts"]["looked_at"]
    out.append(f"<h2>Looked at, not tested — {n} more conditions</h2>")
    out.append("<p>These were not chosen before the test, so nothing here is a finding: with this many tables, some "
               f"parts look good by luck alone. Same trades ({years}). A condition worth testing becomes a slice of "
               "the next idea, chosen before its test.</p>")
    for c in res["looked_at"]:
        out.append(f"<details><summary>{esc(c['label'])} <span class='muted'>— {esc(c['measured_how'])} "
                   f"({esc(c['source'])})</span></summary>{_condition_table(c)}</details>")
    if res["does_not_apply"]:
        out.append("<p class='muted'>Not on this idea: "
                   + esc("; ".join(f"{x['label']} ({x['why']})" for x in res["does_not_apply"])) + ".</p>")
    out.append(f"<h2>The slice rules (fixed before any result was seen)</h2><pre>{esc(slice_rules.RULES)}</pre>")
    return "\n".join(out)


def model_html(f: Facts) -> str:
    """The report page's part about the model (L13.2 A): the check year by year, how each year's model was made,
    what it leaned on (SHAP) with each thing's parts, and its take/skip filter's judgement."""
    esc = html.escape
    head = "<h2>What a model learned — LightGBM (a program that learns from tables)</h2>"
    if f.model is None:
        return head + f"<p class='muted'>Not measured yet (python -m machine.learner {esc(f.idea)} --model).</p>"
    r = f.model["result"]
    c = r["check"]
    out = [head, f"<p>{esc(model_words.check_line(r))}</p>", f"<p>{esc(model_words.named_line(r))}</p>",
           f"<p class='muted'>The model guesses the rupees of each setup's trade from the things known at its moment "
           f"(the same {r['trades']} trades of {learner.years_text(r['years'])} as above). Each year from "
           f"{learner.years_text(r['judge_years'])} was guessed by a model made only from the trades that closed at "
           f"least {config.MODEL_PURGE_DAYS} days before that year began. The shuffles make the whole thing again on "
           "money shuffled inside each year, so the luck line knows every choice made here. Nothing here comes from "
           "2021–2023.</p>"]
    rows = "".join(f"<tr><td>{y['year']}</td><td>{y['top_trades']}</td><td>{inr(y['top_per_trade'])}</td>"
                   f"<td>{y['rest_trades']}</td><td>{inr(y['rest_per_trade'])}</td></tr>" for y in c["by_year"])
    out.append("<div class='wrap'><table><tr><th>year</th><th>trades it liked best</th><th>₹ a trade</th>"
               f"<th>the other trades</th><th>₹ a trade</th></tr>{rows}</table></div>")
    folds = "".join(f"<tr><td>{x['year']}</td><td>{x['learned_from']}</td><td>{x['purged']}</td>"
                    f"<td>{x['guessed']}</td><td>{x['traded']}</td></tr>" for x in r["folds"])
    out.append("<h3>How each year's model was made</h3><div class='wrap'><table><tr><th>the year guessed</th>"
               "<th>trades it learned from</th><th>left out: not closed a week before the year began</th>"
               f"<th>setups it guessed</th><th>of them traded</th></tr>{folds}</table></div>")
    srows = "".join(
        f"<tr><td>{esc(d.get('label', d['feature']))}</td><td>{inr(d['mean_push'])}</td>"
        f"<td>{inr(d['shuffled_push'])}</td><td>{100 * d['p']:.1f} in 100</td>"
        f"<td>{'NAMED' if d['named'] else ''}</td></tr>" for d in r["shap"])
    out.append("<h3>What it leaned on — SHAP (how much each thing pushed its guess, in rupees either way)</h3>"
               "<p class='muted'>A thing is named only when the check passed and luck gives its push less than 5 "
               "times in 100, counting every thing read (the biggest push of any thing in each shuffle).</p>"
               "<div class='wrap'><table><tr><th>thing known at the setup</th><th>its average push, ₹ a trade</th>"
               "<th>with the money shuffled</th><th>luck gives a push this big</th><th></th></tr>"
               f"{srows}</table></div>")
    for d in r["shap"]:
        if not d.get("parts"):
            continue
        prow = "".join(f"<tr><td>{esc(p['part'])}</td><td>{p['trades']}</td><td>{inr(p['mean_push'])}</td>"
                       f"<td>{inr(p['per_trade'])}</td></tr>" for p in d["parts"])
        out.append(f"<details><summary>{esc(d.get('label', d['feature']))} <span class='muted'>— "
                   f"{esc(to_him(d['what']))}</span></summary><div class='wrap'><table><tr><th>part</th><th>trades"
                   f"</th><th>it pushed the guess, ₹ a trade</th><th>the trades made, ₹ a trade</th></tr>{prow}"
                   "</table></div></details>")
    left = r["encoder"]["left_out"]
    if left:
        out.append("<p class='muted'>Not read by the model: "
                   + esc("; ".join(f"{model_words.LABELS.get(k, k)} ({to_him(v)})" for k, v in left.items()))
                   + ".</p>")
    out.append(f"<h3>Its take/skip filter</h3><p>{esc(filter_line(f))}</p>")
    j = f.filter_judge
    if j is not None:
        yr = "".join(f"<tr><td>{y}</td><td>{v['trades']}</td><td>{inr(v['net_inr'])}</td></tr>"
                     for y, v in j["result"]["per_year"].items())
        out.append(f"<p class='muted'>{esc((f.model['result'].get('filter') or {}).get('rule', ''))}. The referee "
                   f"judged it like any version: {esc(j['result']['answer']['why'])}.</p>"
                   "<div class='wrap'><table><tr><th>year</th><th>trades</th><th>₹ after cost</th></tr>"
                   f"{yr}</table></div>")
    p = f.model["settings"].get("params") or {}
    out.append("<p class='muted'>The model's settings (Claude's, the same for every idea, fixed before any idea's "
               f"model was made): {esc(', '.join(f'{k} {v}' for k, v in p.items()))}; "
               f"{f.model['settings'].get('rounds')} trees; {c['shuffles']} shuffles.</p>")
    return "\n".join(out)


def pooled_html(f: Facts) -> str:
    """The report page's part about the pooled view (L13.2 B): whose trades, then every market condition with this
    idea, its random entries, all random entries and the other tested ideas side by side, and the two checks."""
    esc = html.escape
    head = "<h2>Any entry or only this idea — every tested idea's trades and random entries together</h2>"
    if f.pooled is None:
        return head + f"<p class='muted'>Not measured yet (python -m machine.learner {esc(f.idea)} --pooled).</p>"
    r = f.pooled["result"]
    out = [head] + [f"<p>{esc(x)}</p>" for x in (pooled_words.head_line(r), pooled_words.any_line(r),
                                                 pooled_words.only_line(r))]
    srows = "".join(f"<tr><td>{esc(s['label'])}</td><td>{s['trades']}</td><td>{inr(s['per_trade'])}</td>"
                    f"<td>{s['random_trades']}</td><td>{inr(s['random_per_trade'])}</td>"
                    f"<td>{esc('; '.join(s['notes']))}</td></tr>" for s in r["sources"])
    out.append("<p class='muted'>Random entries keep each idea's side, entry type, distances and trades a month, at "
               "random times in the same month (like the referee's). ANY ENTRY asks whether random entries earn more "
               "or less in a part than in the rest; ONLY THIS IDEA asks whether this idea's gap there differs from "
               "the gap of its own random entries. Each has the slice rules' 3 checks and a correction for the "
               f"{r['counts']['tested']} conditions. The other ideas' numbers are shown, not tested.</p>"
               "<div class='wrap'><table><tr><th>idea</th><th>trades</th><th>₹ a trade</th><th>random entries</th>"
               f"<th>₹ a trade</th><th></th></tr>{srows}</table></div>")
    others = bool(r["other_ideas"])
    for c in r["conditions"]:
        rows = []
        for p in c["parts"]:
            cells = [esc(p["group"])]
            for key in ("this", "own_random", "random") + (("others",) if others else ()):
                x = p.get(key)
                cells.append("—" if not x or not x["trades"] else f"{inr(x['per_trade'])} ({x['trades']})")
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
        mark = (" — ANY ENTRY" if c["any"].get("claim") else "") + (" — ONLY THIS IDEA" if c["only"].get("claim")
                                                                   else "")
        out.append(f"<details><summary>{esc(c['label'])}{esc(mark)} <span class='muted'>— {esc(c['how'])}</span>"
                   "</summary><div class='wrap'><table><tr><th>part</th><th>this idea: ₹ a trade (trades)</th>"
                   "<th>its random entries</th><th>all random entries</th>"
                   + ("<th>the other tested ideas</th>" if others else "") + f"</tr>{''.join(rows)}</table></div>"
                   f"<p>Any entry: {esc(pooled_words.check_words(c['any'], False))}</p>"
                   f"<p>Only this idea: {esc(pooled_words.check_words(c['only'], True))}</p></details>")
    if r["not_here"]:
        out.append("<p class='muted'>Not in the pool: "
                   + esc("; ".join(f"{x['label']} ({x['why']})" for x in r["not_here"])) + ".</p>")
    return "\n".join(out)


def page(f: Facts, pic: dict | None) -> str:
    r = f.r
    esc = html.escape
    lines = summary_lines(f)
    img = ""
    if pic is not None:
        data = base64.b64encode(Path(pic["path"]).read_bytes()).decode("ascii")
        img = f"<img alt='money after costs, each year' src='data:image/jpeg;base64,{data}'>"
    # year by year
    yrows = "".join(
        f"<tr{' class=hidden' if y['hidden'] else ''}><td>{y['year']}{' (hidden while choosing)' if y['hidden'] else ''}"
        f"</td><td>{y['trades']}</td><td>{inr(y['net_inr'])}</td>"
        f"<td>{inr(y['net_inr'] / y['trades']) if y['trades'] else 'no trades'}</td></tr>" for y in year_rows(f))
    # versions
    vrows = ""
    if f.tune is not None:
        t = f.tune["result"]
        for v, d in t["versions"].items():
            mark = " class='chosen'" if v == t["chosen"] else ""
            vrows += (f"<tr{mark}><td>{esc(to_him(d['about']) or v)}{' ← chosen' if v == t['chosen'] else ''}</td>"
                      f"<td>{d['trades']}</td><td>{inr(d['per_trade'])}</td><td>{inr(d['net_inr'])}</td>"
                      f"<td>{round(100 * d['deflated_sharpe'])} in 100</td></tr>")
        pbo = t["choice"].get("pbo")
        pbo_line = ("" if not pbo else
                    f"<p>The chance that choosing on the past picked a lucky version (PBO): "
                    f"{round(100 * pbo['pbo'])} in 100 — measured over {pbo['ways']:,} ways of cutting the weeks in "
                    f"half. About 50 in 100 means the choice was luck.</p>")
    else:
        pbo_line = ""
    # power
    pw = r["power"]
    edges = list(next(iter(pw["rows"].values()))) if pw.get("rows") else []
    prow = "".join(f"<tr><td>{n}{' ← this idea' if int(n) == pw['own_n'] else ''}</td>"
                   + "".join(f"<td>{round(100 * s)} in 100</td>" for s in row.values()) + "</tr>"
                   for n, row in pw.get("rows", {}).items())
    phead = "<tr><th>trades</th>" + "".join(f"<th>₹{e} a trade</th>" for e in edges) + "</tr>"
    # fake levels
    fk = r["fake_levels"]
    frows = "".join(f"<tr><td>{'up' if x['step'] > 0 else 'down'} {abs(x['step']):g} times the "
                    f"{'hedge' if f.hedged else 'stop'} distance</td><td>{x['trades']}</td>"
                    f"<td>{inr(x['per_trade'])}</td><td>{of100(x['target_first'], x['trades'])}</td></tr>"
                    for x in fk.get("runs", []))
    # slices: "works best in / fails in" (the learner's file, L13.1)
    slices_part = slices_html(f) + "\n" + model_html(f) + "\n" + pooled_html(f)
    learner_file = "".join(f" and {_path_of(x, Path(config.TRIALS_DIR))}"
                           for x in (f.learner, f.model, f.filter_judge, f.pooled) if x is not None)
    judged_trades = trade_table(r["trades_table"], f.hedged)
    tune_trades = "" if f.tune is None else trade_table(f.tune["result"]["trades_table"], f.hedged)
    rules = esc(verdict.RULES)
    title = f"{f.form.id} report"
    tune_years = years_text(f.tune["settings"]["years"]) if f.tune else ""
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ --bg:#ffffff; --ink:#1f2328; --muted:#6e7781; --line:#d8dde3; --shade:#eef2f8; --win:#1b8a5a; --loss:#d0342c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15181c; --ink:#e6e8eb; --muted:#9aa4ae; --line:#353b42;
  --shade:#1e2a38; --win:#4cc38a; --loss:#f07068; }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px;
  padding:16px; }}
h1 {{ font-size:22px; margin:8px 0 4px; }} h2 {{ font-size:18px; margin:28px 0 8px; }}
h3 {{ font-size:15px; margin:18px 0 2px; }} summary {{ cursor:pointer; margin:6px 0; }}
pre {{ white-space:pre-wrap; background:var(--shade); padding:12px; border-radius:6px; font-size:14px; }}
img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; }}
table {{ border-collapse:collapse; margin:6px 0 12px; font-size:13px; }}
td, th {{ border-bottom:1px solid var(--line); padding:3px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }}
tr.hidden td {{ background:var(--shade); }} tr.chosen td {{ font-weight:700; }}
td.win {{ color:var(--win); }} td.loss {{ color:var(--loss); }}
.wrap {{ overflow-x:auto; }} .muted {{ color:var(--muted); }}
</style></head><body>
<h1>{esc(lines[0])}</h1>
<p class="muted">Made {dt.datetime.now().astimezone():%d %b %Y %H:%M} India time from the referee's files:
{esc(f.judge_path)}{'' if f.tune_path is None else ' and ' + esc(f.tune_path)}{esc(learner_file)}. Every number below
comes from them.</p>
<h2>The answer</h2>
<pre>{esc(chr(10).join(lines[1:]))}</pre>
<h2>Money after costs, each year</h2>
{img}
<div class="wrap"><table><tr><th>year</th><th>trades</th><th>₹ after cost</th><th>₹ a trade</th></tr>{yrows}</table></div>
<p class="muted">0.01 lot, ₹40 per position all-in (your #54, #115). A trade counts in the year it closed. The years
{tune_years} show the chosen version on the years used to choose it — the best of several, so they look better than
a fresh test would.</p>
<h2>Every version tried on {tune_years or 'the tune years'} (the machine chose one)</h2>
<div class="wrap"><table><tr><th>version</th><th>trades</th><th>₹ a trade</th><th>₹ in all</th>
<th>chance its edge is real, allowing for every version tried (Deflated Sharpe)</th></tr>{vrows}</table></div>
{pbo_line}
<h2>How big an edge this test could see</h2>
<p>An edge of this many rupees a trade was put into trades with no edge, {pw['sims']} times; how often the luck test
caught it:</p>
<div class="wrap"><table>{phead}{prow}</table></div>
<h2>Fake levels (the level moved up and down, everything else the same)</h2>
<div class="wrap"><table><tr><th>level moved</th><th>trades</th><th>₹ a trade</th><th>reached the target
first</th></tr>{frows}</table></div>
<p>The real level: {inr(r['summary']['per_trade'])} a trade on {r['summary']['trades']} trades. It beats {fk.get('beaten', 0)}
of its {fk.get('compared', 0)} fake levels.</p>
<h2>Random entries</h2>
<p>{r['random']['runs']} runs with the same distances, cost and trades per month, at random times and random levels:
the middle run made {inr(r['random']['per_trade_median'])} a trade. This idea made {inr(r['summary']['per_trade'])}
a trade and beats {r['random']['beaten']} of {r['random']['runs_with_trades']} runs.</p>
{slices_part}
<h2>Every trade — {esc(_where_judged(f))}</h2>
<p class="muted">India time. * = the order inside one minute was not known; the worse case was taken.</p>
<div class="wrap">{judged_trades}</div>
{'' if not tune_trades else f'<h2>Every trade of the chosen version — {tune_years} (used to choose it)</h2><div class="wrap">{tune_trades}</div>'}
<h2>The rules (fixed before any result was seen)</h2>
<pre>{rules}</pre>
</body></html>
"""
    problems = check_words(_visible(body))
    if problems:
        raise ReportError("report.html: " + "; ".join(problems[:5]))
    return body


def _visible(page_html: str) -> str:
    """The words a reader sees: no tags, no style, no picture data."""
    text = re.sub(r"<style.*?</style>", " ", page_html, flags=re.S)
    text = re.sub(r"src='data:[^']*'", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


# ================================================================================================================
# build and send
# ================================================================================================================

@dataclass
class Built:
    facts: Facts
    text: str
    picture: dict
    page_path: Path


def build(idea_id: str) -> Built:
    f = facts(idea_id)
    text = telegram_text(f)
    problems = check_words(text)
    if problems:
        raise ReportError("the Telegram text: " + "; ".join(problems))
    folder = store.folder(idea_id)
    pic = picture(f, folder / PICTURE)
    body = page(f, pic)
    out = folder / PAGE
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(out)
    return Built(f, text, pic, out)


def send(b: Built, *, outbox: Path | None = None, shown: Path | None = None) -> dict:
    """The Telegram lines into the outbox (the bot sends them: the picture, then the text). Never the same picture
    twice (#117)."""
    pic = Path(b.picture["path"]).resolve()
    rel = pic.relative_to(config.ROOT).as_posix() if pic.is_relative_to(config.ROOT) else str(pic)
    shown_path = Path(shown or config.SHOWN_PICTURES)
    if shown_path.is_file():
        for line in shown_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("pixel_sha256") == b.picture["pixel_sha256"]:
                raise ReportError(f"this exact picture was already sent ({rel}) - #117: never the same picture twice")
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    line = {"id": f"{b.facts.idea}-report-{stamp}", "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "to": "Tejas", "text": b.text, "photo": rel, "idea": b.facts.idea, "why": "L10 report",
            "shown": {"subject": f"L10 report {b.facts.idea} money by year", "kind": "report", "file": rel,
                      "pixel_sha256": b.picture["pixel_sha256"], "why": f"L10 report of {b.facts.idea}"}}
    path = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line
