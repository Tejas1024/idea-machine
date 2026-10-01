"""The research tools (L14.1): what the research agent (L14.2) may do, with the guards INSIDE each tool.

    from machine.mcp import tools as T
    T.start("idea-002")                            # one research run = ONE idea (the server: --idea idea-002)
    T.get_idea("idea-002")                         # his words, the steps, the unclear parts, the family, the budget
    T.propose_version("idea-002", "why", readings={"U1": "B", ...}, changes={"steps.gap.max_wait_bars": 8})
    T.run_version("idea-002", "v-...")             # 2016-2020 through the tester: one trial file
    T.write_report("idea-002")                     # the referee chooses ONE version; the check years opened once

machine/mcp/server.py hands exactly these tools to the agent over MCP. Why the guards live here and not in the agent's
instructions: an AI told "don't look at 2021-2023" still can, and none of the strategies an LLM found survived honest
judging (Gençay 2026, arXiv 2608.27734) - so every tool checks for itself, whatever it is asked:

  one idea per run           the server is started for ONE idea (its research target). Only that idea's versions can
                             be proposed, run, charted or reported; another idea can only be read, with no hidden year
  the sealed years 2024+     refused always (guards.check_dates; LEARNED #96 "No, don't touch those years", #104). No
                             tool unlocks them: the three uses Tejas said yes to (T14) are not research
  the check years 2021-2023  refused until the idea's check-opened.json exists and names the version asking
                             (guards.check_opened). Only write_report opens them - once per family, for the ONE version
                             the referee chose on 2016-2020 by the highest Deflated Sharpe, never the agent's pick
  money                      only after the meaning is locked AND his YES to the 3 setups the machine found (SHOWN,
                             L10b) - lesson 16: meaning is chosen before money is seen
  20 versions per family     propose_version and run_version (config.VERSION_BUDGET; machine/mcp/versions.py)
  8 slices per idea          choose_slices, only before any money was seen for the idea (config.SLICE_BUDGET); a slice
                             never comes in as a version (a filter, a direction) - versions.check_change refuses it
  his fixed rules            refused with his words (LEARNED #121), under any name or old alias
Every refusal is also written to data/mcp/refusals.jsonl (a trick is written down, not only refused).

What the tools do NOT guard: files. An agent that can also read data/ or plan/work/ on its own gets around every line
above - the research agent is given ONLY these tools (L14.2; machine/mcp/README.md).
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import ValidationError

from machine import config, guards, rules, settings
from machine import trials as trial_files
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import LOCKED_FROM, SLICE_DIMENSIONS, STATES, FormError, Idea, Slice, readable
from machine.mcp import versions as V
from machine.mcp.versions import Refused
from machine.referee import prior, splits, trials

TUNE = splits.years_text(config.TUNE_YEARS)
CHECK = splits.years_text(config.CHECK_YEARS)
MONEY_STATES = ("TUNE RUN", "CHECK OPENED", "REPORTED", "DECIDED")
COST = "Rs 40 per position per 0.01 lot, all-in (LEARNED #54, #115)"
LOCK = threading.RLock()        # one call at a time: settings.using() is ONE global (machine/settings.py)
_CODE = ("machine/ideas", "machine/blocks", "machine/settings.py", "machine/rules.py", "machine/tester",
         "machine/data/bars.py", "machine/data/clock.py")


# ================================================================================================= the run

@dataclass
class Session:
    """One research run: the idea it is about (its target) and who runs it (goes into every record it leaves)."""
    target: str | None = None
    who: str = "the research agent (L14.2) through the research tools (L14.1)"


SESSION = Session()


def start(idea: str | None, who: str | None = None) -> Session:
    """Bind this process to ONE idea (python -m machine.mcp --idea idea-002). None = read-only: get_idea, list_blocks,
    search_docs, list_trials (no hidden year shown)."""
    if idea is not None:
        V.idea_form(idea)                                  # only one of his ideas, with a form
    SESSION.target = idea
    if who:
        SESSION.who = who
    return SESSION


def _target(idea: str, tool: str) -> None:
    if SESSION.target is None:
        raise Refused(f"{tool}: this research run has no idea - start the server for ONE idea (python -m machine.mcp "
                      "--idea idea-NNN). Without one only get_idea, list_blocks, search_docs and list_trials work.")
    if idea != SESSION.target:
        raise Refused(f"{tool}: this research run is about {SESSION.target}, not {idea!r} - one run, one idea (another "
                      "idea's hidden years must not steer this one's versions)")


# ================================================================================================= the workbench

@dataclass
class Workbench:
    """What the tools call to reach prices and the referee: the real ones by default, hand-made ones in the tests.
    The tools check what comes back (the market's years, the signals' moments) - a workbench is not trusted."""
    load_market: Callable | None = None      # tfs -> a compile Market of the TUNE years
    compile: Callable | None = None          # C.compile_idea
    tester: Callable | None = None           # (signals, tf, note) -> machine.tester Result on the TUNE years
    walk_forward: Callable | None = None     # machine.referee.judge.judge_walk_forward
    tune: Callable | None = None             # machine.referee.tune.run_versions
    check: Callable | None = None            # machine.report.run.check_idea
    learn: Callable | None = None            # machine.learner.run.learn
    build: Callable | None = None            # machine.report.build.build
    send: Callable | None = None             # machine.report.build.send
    load_bars: Callable | None = None        # (tf, start, end) -> bars (machine.data.bars.load_bars)
    prices_hash: Callable | None = None      # () -> the 1-minute prices' content hash
    log: Callable = print
    _markets: dict = field(default_factory=dict, repr=False)

    def market(self, tfs) -> C.Market:
        key = tuple(sorted(tfs))
        if key not in self._markets:
            m = (self.load_market or _real_market)(tuple(tfs))
            lo, hi = m.window()
            inside_tune("the market", lo, hi)
            self._markets[key] = m
        return self._markets[key]

    def get(self, name: str) -> Callable:
        f = getattr(self, name)
        if f is not None:
            return f
        if name == "compile":
            return C.compile_idea
        if name == "tester":
            from machine.tester import run as T
            return lambda sigs, tf, note: T(sigs, tf=tf, start=str(config.TUNE_YEARS[0]),
                                            end=str(config.TUNE_YEARS[-1]), record=True, note=note)
        if name == "walk_forward":
            from machine.referee import judge
            return judge.judge_walk_forward
        if name == "tune":
            from machine.referee import tune
            return tune.run_versions
        if name == "check":
            from machine.report import run as R
            return R.check_idea
        if name == "learn":
            from machine.learner import run as L
            return L.learn
        if name in ("build", "send"):
            from machine.report import build as B
            return {"build": B.build, "send": B.send}[name]
        if name == "load_bars":
            from machine.data import bars
            return bars.load_bars
        if name == "prices_hash":
            from machine.data import bars
            return bars.m1_fingerprint
        raise KeyError(name)


def _real_market(tfs) -> C.Market:
    return C.Market.load(str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1]), tfs=tfs)


BENCH = Workbench()


def inside_tune(what: str, first, last) -> None:
    """Refuse anything that reaches outside 2016-2020: the sealed years with his words, the check years plainly."""
    guards.check_dates([pd.Timestamp(first), pd.Timestamp(last)])
    years = {pd.Timestamp(first).year, pd.Timestamp(last).year}
    if min(years) < config.TUNE_YEARS[0] or max(years) > config.TUNE_YEARS[-1]:
        raise Refused(f"{what} reaches {pd.Timestamp(first):%Y-%m-%d} to {pd.Timestamp(last):%Y-%m-%d}: versions are "
                      f"tried on {TUNE} only (the check years {CHECK} open once, through write_report)")


# ================================================================================================= refusals

def _log_refusal(tool: str, args: dict, why: str) -> None:
    line = {"at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "tool": tool,
            "target": SESSION.target, "args": {k: repr(v)[:200] for k, v in args.items()}, "why": why[:2000]}
    try:
        path = Path(config.MCP_DIR) / "refusals.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass                                               # the refusal itself still stands


REFUSALS = (Refused, guards.GuardError, store.StoreError, FormError, rules.FixedRuleError, settings.SettingError,
            C.CompileError)


def tool(fn: Callable) -> Callable:
    """One call at a time; every refusal written down and raised as Refused (its words kept)."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with LOCK:
            try:
                return fn(*args, **kwargs)
            except REFUSALS as err:
                why = str(err)
                if isinstance(err, guards.SealedYearsError):        # how unlocking works elsewhere is no hint here
                    why = (why.split(" Unlocking needs")[0] + " The research tools never open them: none of the "
                           "three uses Tejas said yes to (T14) is research.")
                named = dict(zip(fn.__code__.co_varnames, args)) | {k: v for k, v in kwargs.items() if k != "bench"}
                _log_refusal(fn.__name__, named, why)
                if isinstance(err, Refused):
                    raise
                raise Refused(why) from err
    return run


# ================================================================================================= small helpers

def money_allowed(idea: str) -> tuple[bool, str]:
    """(may money be seen for this idea, why): its meaning locked AND his YES to the 3 setups (SHOWN, L10b), or a
    state after it."""
    s = store.read_state(idea)
    state = s["state"]
    if state in MONEY_STATES:
        return True, f"{idea} is {state}"
    if state == "SHOWN":
        from machine.intake import shown
        rec = shown.load(idea) or {}
        if rec.get("status") == "yes":
            return True, f"{idea} is SHOWN and he said YES to the 3 setups the machine found"
        return False, (f"{idea} is SHOWN: it waits for his YES to the 3 setups the machine found (status "
                       f"{rec.get('status', 'not sent')!r})")
    return False, f"{idea} is {state}" + (f" - waits for {s['waits_for']}" if s.get("waits_for") else "")


def _money_gate(idea: str) -> None:
    ok, why = money_allowed(idea)
    if not ok:
        raise Refused(f"{why}. No money is seen for an idea before its meaning is locked and he said YES to the 3 "
                      "setups the machine found (lesson 16: meaning is chosen before money is seen; L10b SHOWN) - "
                      "else his NO could no longer reopen it (store.reopen).")


def check_open_for(idea: str, version: str | None) -> bool:
    """May the check years of this idea and version be SHOWN here? Only for this run's own idea, once its
    check-opened.json names this version (guards.check_opened)."""
    if SESSION.target != idea or version is None:
        return False
    try:
        guards.check_opened(idea, version=version)
    except guards.CheckYearsClosedError:
        return False
    return True


def _rel(path) -> str:
    return V.rel(Path(path))


def _plain(v: Any) -> Any:
    """A rule's value as JSON can carry it (his fixed rules keep theirs in read-only mappings and tuples)."""
    if isinstance(v, Mapping):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return v


def _key(rec: dict) -> prior.IdeaKey:
    k = rec["key"]
    return prior.IdeaKey(frozenset(k["blocks"]), frozenset(k["timeframes"]), k["entry"], frozenset(k.get("mgmt", ())))


def version_record(idea: str, version: str) -> dict:
    """A version of this idea: one proposed here (versions.jsonl), or one the idea's tune already tried (L10: its
    readings) - so the intake's versions can be run, charted and judged too. Anything else is refused."""
    for r in V.records(idea):
        if r["version_id"] == version:
            return r
    row = V.tuned_versions(idea).get(version)
    if row is None:
        V.record(idea, version)                          # raises: not a version of this idea
    form = V.idea_form(idea)
    parent, why = V.parent_of(idea, form)
    try:
        f = C.prepare(form, row["readings"] or None, row.get("changes") or None)[0]
    except (FormError, rules.FixedRuleError, settings.SettingError, C.CompileError) as err:
        raise Refused(f"{version} of the tune cannot be made again: {err}") from None
    return {"idea_id": idea, "version_id": version, "parent_idea": parent, "family": None, "parent_why": why,
            "readings": row["readings"], "changes": row.get("changes") or {}, "about": row.get("about", ""),
            "why": "a version the idea's tune tried (L10)", "key": prior.key_for_form(f).as_dict(),
            "timeframe": f.timeframe}


def _cache(idea: str, version: str) -> Path:
    return Path(config.MCP_DIR) / idea / f"{version}.json"


def _read_cache(idea: str, version: str) -> dict:
    p = _cache(idea, version)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _write_cache(idea: str, version: str, body: dict) -> None:
    p = _cache(idea, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


@functools.lru_cache(maxsize=1)
def _code_hash() -> str:
    files = []
    for part in _CODE:
        p = config.ROOT / part
        files += sorted(p.glob("*.py")) if p.is_dir() else [p]
    return trial_files.code_sha256(files)


def _cache_key(rec: dict, bench: Workbench) -> str:
    """The signals' fingerprint (lesson B19: a cache carries what made it): the rule, the family, the prices, the code."""
    body = {"version": rec["version_id"], "readings": rec.get("readings", {}), "changes": rec.get("changes", {}),
            "parent": rec.get("parent_idea"), "prices": bench.get("prices_hash")(), "code": _code_hash(),
            "years": list(config.TUNE_YEARS)}
    return trial_files.sha256_of(body)


def _form_for(idea: str, parent: str | None) -> Idea:
    form = store.load_form(idea)
    if parent and not form.parent_idea:                  # the family found by the section-D rule rides on the signals
        form = form.model_copy(update={"parent_idea": parent})
    return form


def signals_of(idea: str, rec: dict, bench: Workbench) -> list:
    """The version's signals on 2016-2020: from the cache when its fingerprint matches, else compiled (and checked)."""
    from machine.tester.models import Signal
    key = _cache_key(rec, bench)
    cached = _read_cache(idea, rec["version_id"])
    if cached.get("key") == key and "signals" in cached:
        return [Signal.model_validate(s) for s in cached["signals"]]
    form = _form_for(idea, rec.get("parent_idea"))
    f = C.prepare(form, rec.get("readings") or None, rec.get("changes") or None)[0]
    market = bench.market(C.needed_tfs(f))
    c = bench.get("compile")(form, market, readings=rec.get("readings") or None, changes=rec.get("changes") or None)
    if c.version_id != rec["version_id"]:
        raise Refused(f"{rec['version_id']} now compiles to {c.version_id}: the idea's form or the code changed since "
                      "it was proposed - propose it again")
    if c.signals:                                        # a signal known at the close of 2020's last bar (midnight
        known = [pd.Timestamp(s.known_at) for s in c.signals]   # on 1 Jan) belongs to 2020: one minute back
        inside_tune(f"{rec['version_id']}'s signals", min(known), max(known) - pd.Timedelta(minutes=1))
    _write_cache(idea, rec["version_id"], {"key": key, "version_id": rec["version_id"],
                                           "signals": [s.model_dump(mode="json") for s in c.signals]})
    return c.signals


def _years_of(f: dict) -> set[int]:
    """Every year a trial file's numbers cover (its settings' years, its window, its per-year table)."""
    out: set[int] = set()
    s, r = f.get("settings") or {}, f.get("result") or {}
    for y in s.get("years") or []:
        out.add(int(y))
    w = s.get("window") or {}
    for k in ("first_bar", "last_bar"):
        if w.get(k):
            out.add(pd.Timestamp(w[k]).year)
    for y in (r.get("per_year") or {}):
        if str(y).isdigit():
            out.add(int(y))
    if r.get("judged_trial"):                            # a learner run that also read the check years' trades
        out |= set(config.CHECK_YEARS)
    if len(out) >= 2:
        out |= set(range(min(out), max(out) + 1))
    return out


def _budget(fam: dict) -> dict:
    return {"family": fam["root"], "used": fam["used"], "budget": fam["budget"], "left": fam["left"],
            "tried": len(fam["tried"]), "proposed_not_run": len(fam["waiting"])}


def _tester_file(idea: str, version: str) -> dict | None:
    got = [f for f in trials.load() if f.get("kind") == "tester" and f.get("idea_id") == idea
           and f.get("version_id") == version]
    return got[-1] if got else None


def _numbers(summary: dict) -> dict:
    keep = ("signals", "trades", "won", "win_pct", "net_inr_total", "net_inr_per_trade", "gross_usd_per_trade",
            "per_year", "worst_year", "setups_skipped", "open_at_end", "closed_by", "not_filled", "closed_at_once",
            "ambiguous")
    out = {k: summary.get(k) for k in keep}
    extra = summary.get("extra_lines") or {}
    out["extra_lines"] = extra.get("text")
    return out


# ================================================================================================= 1. get_idea

@tool
def get_idea(idea: str) -> dict:
    """The idea as the machine holds it - his words, steps, entry, exit, targets, unclear parts and their readings,
    slices, state, family, version budget, check years - and NO money (no trial numbers: list_trials)."""
    form = V.idea_form(idea)
    s = store.read_state(idea)
    files = trials.load()
    parent, parent_why = V.parent_of(idea, form, files)
    fam = V.family(idea, parent, files)
    ok, money_why = money_allowed(idea)
    slices = [x.model_dump(mode="json") for x in store.with_chosen_slices(form).slices]
    recs = V.records(idea)
    opening = splits.family_opening(idea, parent)
    if opening is None:
        check = f"closed: the {fam['root']} family never opened {CHECK}"
    elif opening["idea_id"] == idea:
        check = (f"opened {str(opening.get('opened_at', '?'))[:10]} for {idea} version {opening.get('version_id')} - "
                 f"later versions are judged by walk-forward inside {TUNE}")
    else:
        check = (f"used by {opening['idea_id']} (same family {fam['root']}): {idea} is judged by walk-forward inside "
                 f"{TUNE}, never on {CHECK}")
    d = form.model_dump(mode="json")
    key = prior.key_for_form(form)
    return {
        "idea": idea, "title": form.title, "his_words": form.words, "timeframe": form.timeframe,
        "directions": list(form.directions), "state": s["state"], "since": s.get("since"),
        "waits_for": s.get("waits_for"), "needs": s.get("needs"),
        "locked": STATES.index(s["state"]) >= STATES.index(LOCKED_FROM),
        "money_allowed": ok, "money_why": money_why,
        "research_target": SESSION.target,
        "parent_idea": parent, "parent_why": parent_why, "family": fam["root"], "family_ideas": fam["ideas"],
        "steps": d["steps"], "entry": d["entry"], "exit": d["exit"], "targets": d["targets"],
        "do_not_enter": d["do_not_enter"], "filters": d["filters"], "settings": d["settings"],
        "unclear": [{"id": u.id, "kind": u.kind, "part": u.part, "his_words": u.words, "answer": u.answer,
                     "readings": [r.model_dump(mode="json") for r in u.readings]} for u in form.unclear],
        "slices": slices,
        "slice_budget": {"chosen": len(slices), "budget": config.SLICE_BUDGET,
                         "can_choose_more": not store.money_seen(idea) and len(slices) < config.SLICE_BUDGET},
        "versions": [{"version": r["version_id"], "about": r["about"], "why": r["why"],
                      "run": (idea, r["version_id"]) in fam["tried"]} for r in recs],
        "version_budget": _budget(fam),
        "check_years": check,
        "already_seen_by_the_old_project": prior.already_seen(key, config.TUNE_YEARS + config.CHECK_YEARS).text(),
        "how_versions_are_made": ("a version changes only the timeframe (dimension 1) or the idea's own numbers - "
                                  "level, stop, target, wait (dimension 10) - and chooses a reading for every VERSION "
                                  "part; see list_blocks"),
    }


# ================================================================================================= 2. list_blocks

@tool
def list_blocks() -> dict:
    """Every block, do-not-enter rule and filter a form may name; his fixed rules (never changed); the numbers a
    version may change; the conditions a slice may name."""
    from machine.learner import conditions
    fixed = [{"name": k, "value": _plain(r.value), "means": r.means, "taught": r.taught}
             for k, r in rules.FIXED.items()]
    numbers = [{"name": k, "value": s.value, "who": s.who, "means": s.means}
               for k, s in settings.ALL.items() if s.status == "test" and V.is_number(s.value)]
    return {
        "blocks": C.describe(),
        "his_fixed_rules": fixed,
        "settings_a_version_may_change": numbers,
        "version_rules": V.__doc__,
        "slice_dimensions": list(SLICE_DIMENSIONS),
        "slice_conditions": conditions.describe(),
        "budgets": {"versions_per_family": config.VERSION_BUDGET, "slices_per_idea": config.SLICE_BUDGET},
        "years": {"tune": TUNE, "check": f"{CHECK} - opened once per family, by write_report only",
                  "sealed": "2024 on - never (LEARNED #96, #104)"},
    }


# ================================================================================================= 3. search_docs

@tool
def search_docs(query: str, hits: int = 5) -> dict:
    """His #129 rule: search his documents (strategy.md, LEARNED.md, strategy-understanding.md, rules, settings, the
    lesson inbox, his answers, his pictures' lessons) before anything is asked. Words only - no prices, no trial files."""
    from machine.search import rank
    q = str(query or "").strip()
    if not q or len(q) > 500:
        raise Refused("search_docs needs a question of 1 to 500 characters")
    if not isinstance(hits, int) or isinstance(hits, bool) or not 1 <= hits <= 20:
        raise Refused("hits: a whole number from 1 to 20")
    a = rank.answered(q)
    return {"found": a.found, "answer": str(a), "how": a.how,
            "hits": [{"id": h.id, "source": h.source, "date": h.date,
                      "where": f"{h.path}:{h.line}" if h.line else h.path, "quote": h.quote, "passes": h.passes}
                     for h in a.hits[:hits]]}


# ================================================================================================= 4. propose_version

@tool
def propose_version(idea: str, why: str, readings: dict | None = None, changes: dict | None = None) -> dict:
    """Propose one version: a reading for every VERSION part + changes to the timeframe or the idea's own numbers.
    Refuses his fixed rules (his words), slices dressed as versions, anything that is what the idea IS, and a 21st
    version of the family. Nothing runs; the try counts from now on."""
    _target(idea, "propose_version")
    if (readings is not None and not isinstance(readings, dict)) or (changes is not None
                                                                     and not isinstance(changes, dict)):
        raise Refused("readings and changes are mappings: {\"U1\": \"B\"} and {\"steps.gap.max_wait_bars\": 8}")
    rec, new = V.propose(idea, readings=readings, changes=changes, why=why, who=SESSION.who)
    fam = V.family(idea, rec["parent_idea"])
    return {"version": rec["version_id"], "new": new, "about": rec["about"], "dimensions": rec["dimensions"],
            "readings": rec["readings"], "changes": rec["changes"], "timeframe": rec["timeframe"],
            "family": rec["family"], "parent_idea": rec["parent_idea"], "parent_why": rec["parent_why"],
            "version_budget": _budget(fam),
            "said": ("the same rule as an earlier proposal: the same version, not a new try" if not new else
                     f"proposed - it counts as a try now. Next: run_version('{idea}', '{rec['version_id']}')")}


# ================================================================================================= 5. run_version

@tool
def run_version(idea: str, version: str, how: str = "tune", bench: Workbench | None = None) -> dict:
    """Run a proposed version on 2016-2020 ("tune": the tester, one trial file) or judge it by walk-forward inside
    2016-2020 ("walk-forward": the referee, one trial file). Never the check years, never 2024+."""
    bench = bench or BENCH
    _target(idea, "run_version")
    if how not in ("tune", "walk-forward"):
        raise Refused(f"how = {how!r}: 'tune' runs the version on {TUNE}; 'walk-forward' judges it inside {TUNE}. The "
                      f"check years {CHECK} open only through write_report, once per family, for the one version the "
                      "referee chooses.")
    V.locked(idea)
    _money_gate(idea)
    rec = version_record(idea, version)
    fam = V.family(idea, rec["parent_idea"])
    if (idea, version) not in fam["tried"] and len(fam["tried"]) >= fam["budget"]:
        raise Refused(f"the {fam['root']} family has tried {len(fam['tried'])} versions - the budget is "
                      f"{fam['budget']} (config.VERSION_BUDGET)")
    if how == "walk-forward":
        return _walk_forward(idea, rec, bench)
    done = _tester_file(idea, version)
    cached = _read_cache(idea, version)
    if done is not None and cached.get("trades") is not None and cached.get("key") == _cache_key(rec, bench):
        return {"version": version, "already_run": True, "years": TUNE, "about": rec["about"],
                "numbers": _numbers(done["result"]), "trial": cached.get("trial"), "cost": COST,
                "version_budget": _budget(V.family(idea, rec["parent_idea"])),
                "said": "already run - the same rule gives the same trades; not run again"}
    sigs = signals_of(idea, rec, bench)
    if not sigs:
        raise Refused(f"{version} made no setup on {TUNE} - nothing to test (it still counts as proposed)")
    res = bench.get("tester")(sigs, rec["timeframe"], f"L14.1 research: {idea} {version} - {rec['about']} "
                                                      f"(why: {rec['why']})")
    lo, hi = res.replay.first_bar, res.replay.last_bar
    inside_tune(f"the test of {version}", lo, hi)
    body = _read_cache(idea, version)
    body.update(trades=[t.model_dump(mode="json") for t in res.replay.trades], summary=res.summary,
                trial=None if res.trial_path is None else _rel(res.trial_path), tf=rec["timeframe"])
    _write_cache(idea, version, body)
    seen = prior.already_seen(_key(rec), config.TUNE_YEARS)
    return {"version": version, "already_run": False, "years": TUNE, "about": rec["about"],
            "numbers": _numbers(res.summary), "trial": body["trial"], "cost": COST,
            "clock": "MT5 server time", "lots": "0.01",
            "already_seen_by_the_old_project": seen.text(),
            "version_budget": _budget(V.family(idea, rec["parent_idea"])),
            "said": (f"tried on {TUNE} only. These years chose nothing yet: write_report lets the referee choose ONE "
                     "version by the highest Deflated Sharpe (the chance its edge is real, allowing for every version "
                     "tried) and opens the check years once for it.")}


def _walk_forward(idea: str, rec: dict, bench: Workbench) -> dict:
    """The versions of this idea that were run (here or by its tune), plus this one, judged by walk-forward inside
    2016-2020 - for each judge year the referee picks the version, on the years before it only."""
    ran = {r["version_id"]: r for r in V.records(idea) if _tester_file(idea, r["version_id"]) is not None}
    for vid in V.tuned_versions(idea):
        ran.setdefault(vid, version_record(idea, vid))
    ran[rec["version_id"]] = rec
    sigs = {vid: signals_of(idea, r, bench) for vid, r in ran.items()}
    empty = [vid for vid, s in sigs.items() if not s]
    if empty:
        raise Refused(f"{', '.join(empty)} made no setup on {TUNE}: nothing to judge")
    rep = bench.get("walk_forward")(sigs, tf=rec["timeframe"], key=_key(rec), newest=rec["version_id"],
                                    tfs={vid: r["timeframe"] for vid, r in ran.items()},
                                    note=f"L14.1 research: {idea} {rec['version_id']} by walk-forward")
    return {"version": rec["version_id"], "how": "walk-forward", "answer": rep.word, "why": rep.answer.why,
            "judged_on": rep.judged_on, "text": rep.text(),
            "trial": None if rep.trial_path is None else _rel(rep.trial_path),
            "version_budget": _budget(V.family(idea, rec["parent_idea"]))}


# ================================================================================================= 6. list_trials

@tool
def list_trials(idea: str) -> dict:
    """Every trial file of the idea's family: what ran, when, on which years, its numbers. The check years' numbers
    only for this run's own idea once its check-opened.json names that version; 2024+ never."""
    form = V.idea_form(idea)
    files = trials.load()
    parent, _ = V.parent_of(idea, form, files)
    fam = V.family(idea, parent, files)
    members = set(fam["ideas"])
    rows, hidden = [], 0
    for f in files:
        if f.get("idea_id") not in members:
            continue
        row = _trial_row(idea, f)
        hidden += bool(row.get("hidden"))
        rows.append(row)
    waiting = [{"idea": i, "version": v, "said": "proposed, not run yet"} for i, v in fam["waiting"]]
    return {"idea": idea, "family": fam["root"], "family_ideas": sorted(members), "version_budget": _budget(fam),
            "trials": rows, "proposed_not_run": waiting, "hidden": hidden,
            "rule": (f"numbers on {TUNE} always; on {CHECK} only for this run's own idea, for the version its "
                     "check-opened.json names; 2024 on never")}


KNOWN_KINDS = ("tester", "referee-tune", "referee", "referee-control", config.LEARNER_KIND)


def _trial_row(idea: str, f: dict) -> dict:
    """One trial file as list_trials shows it. Its numbers - and its note, free text that may quote them - only when
    every year it covers is known and may be seen in this run."""
    years = _years_of(f)
    kind = f.get("kind")
    row = {"trial_id": f.get("trial_id"), "kind": kind, "idea": f.get("idea_id"),
           "version": f.get("version_id"), "made_at": f.get("made_at")}
    if any(y >= int(config.SEALED_FROM[:4]) for y in years):
        return {**row, "hidden": "it touches the sealed years - never shown (LEARNED #96, #104)"}
    if not years or kind not in KNOWN_KINDS:
        return {**row, "hidden": "a kind of run whose years these tools cannot read - not shown (it may reach the "
                                 "check years)"}
    check = sorted(y for y in years if y in config.CHECK_YEARS)
    if check and not (f.get("idea_id") == idea and check_open_for(idea, f.get("version_id"))):
        return {**row, "years": splits.years_text(years), "hidden": (
            f"its numbers reach the check years {splits.years_text(check)}, which are not open for "
            f"{f.get('idea_id')} {f.get('version_id')} in this run")}
    r = f.get("result") or {}
    row["years"] = splits.years_text(years)
    row["note"] = f.get("note")
    if kind == "tester":
        row["numbers"] = _numbers(r)
    elif kind == "referee-tune":
        row["chosen"] = r.get("chosen")
        row["versions"] = {v: {k: x.get(k) for k in ("about", "trades", "per_trade", "net_inr", "per_year",
                                                       "deflated_sharpe", "timeframe")}
                           for v, x in (r.get("versions") or {}).items()}
    elif kind in ("referee", "referee-control"):
        ans = r.get("answer") or {}
        row.update(answer=ans.get("word"), why=ans.get("why"), judged_on=(f.get("settings") or {}).get("judged_on"),
                   per_year=r.get("per_year"), summary=r.get("summary"))
    else:                                                  # the learner: its claims on 2016-2020 and its counts
        row.update(claims=r.get("claims"), counts=r.get("counts"))
    return row


# ================================================================================================= 7. draw_chart

@tool
def draw_chart(idea: str, version: str, trade: int | str | None = None, start: str | None = None,
               end: str | None = None, tf: str = "5m", bench: Workbench | None = None) -> dict:
    """ONE picture: a trade of a version that ran (trade = its number, 1 = the first, or its signal id), or a stretch
    of time (start, end: MT5 server time, no time zone) with the version's setups marked. 2016-2020 always; the check
    years only for this run's idea once opened for this version; 2024+ never."""
    bench = bench or BENCH
    _target(idea, "draw_chart")
    V.locked(idea)
    _money_gate(idea)
    rec = version_record(idea, version)                  # refused: not a version of this idea
    if (trade is None) == (start is None and end is None):
        raise Refused("draw_chart draws ONE thing: give trade (its number or signal id) OR start and end")
    if trade is not None:
        return _draw_trade(idea, version, trade, bench)
    return _draw_window(idea, rec, start, end, tf, bench)


def _when(text: Any, what: str) -> pd.Timestamp:
    """A time the agent typed: MT5 server time as text, no time zone, a real date inside the data."""
    if not isinstance(text, str) or not text.strip():
        raise Refused(f"{what}: give a date and time as text, like '2019-03-11 14:00' (MT5 server time)")
    try:
        t = pd.Timestamp(text.strip())
    except (ValueError, TypeError) as err:
        raise Refused(f"{what} = {text!r}: not a date ({err})") from None
    if t is pd.NaT or pd.isna(t):
        raise Refused(f"{what} = {text!r}: not a date")
    if t.tzinfo is not None:
        guards.check_dates(t)                            # a sealed moment on EITHER clock is refused with his words
        raise Refused(f"{what} = {text!r}: give MT5 server time without a time zone, like the price data")
    guards.check_dates(t)
    return t


def _allowed_end(idea: str, version: str) -> pd.Timestamp:
    last = config.CHECK_YEARS[-1] if check_open_for(idea, version) else config.TUNE_YEARS[-1]
    return pd.Timestamp(f"{last}-12-31 23:59")


def _chart_bars(bench: Workbench, tf: str, start: pd.Timestamp, end: pd.Timestamp, idea: str, version: str):
    lo = max(start, pd.Timestamp(f"{config.TUNE_YEARS[0]}-01-01"))
    hi = min(end, _allowed_end(idea, version))
    if hi <= lo:
        return None
    guards.check_dates([lo, hi])
    if not check_open_for(idea, version):
        inside_tune("the chart", lo, hi)
    else:
        guards.check_opened(idea, [lo, hi], version=version)
    b = bench.get("load_bars")(tf, lo, hi)
    guards.check_dates(b)
    return b


def _picture_path(idea: str, version: str, what: str) -> Path:
    safe = trial_files.SAFE.sub("_", what)
    return Path(config.MCP_DIR) / idea / "charts" / f"{version}-{safe}.jpg"


def _render(pic, path: Path) -> dict:
    from machine.charts import draw as D
    from machine.charts import lint
    path.parent.mkdir(parents=True, exist_ok=True)
    drawn = D.render(pic, path)
    problems = [str(p) for p in lint.check(drawn, shown=Path(config.MCP_DIR) / "no-pictures-shown.jsonl")]
    return {"picture": _rel(path), "picture_rules": problems or "all his picture rules pass",
            "shown_to_tejas": False}


def _draw_trade(idea: str, version: str, trade: int | str, bench: Workbench) -> dict:
    from machine.charts import draw as D
    from machine.tester.models import Trade
    cached = _read_cache(idea, version)
    pool = [Trade.model_validate(t) for t in cached.get("trades") or []]
    closed = [t for t in pool if t.status == "closed"]
    if not pool:
        raise Refused(f"{version} has no trades saved by run_version yet - run it first (run_version)")
    if isinstance(trade, bool) or not isinstance(trade, (int, str)):
        raise Refused("trade: its number (1 = the first closed trade) or its signal id")
    if isinstance(trade, int):
        if not 1 <= trade <= len(closed):
            raise Refused(f"trade {trade}: {version} has {len(closed)} closed trades on {TUNE} (1 to {len(closed)})")
        t = closed[trade - 1]
    else:
        t = next((x for x in closed if x.signal_id == trade), None)
        if t is None:
            raise Refused(f"no CLOSED trade of {version} has the signal id {trade!r} (a trade still open when the "
                          "test ended has no money to draw)")
    end = pd.Timestamp(t.exits[-1].ts)
    start = pd.Timestamp(t.fill_ts)
    inside_tune(f"trade {trade}", start, end)            # run_version keeps only its 2016-2020 trades
    m5 = _chart_bars(bench, "5m", start - pd.Timedelta(hours=6), end + pd.Timedelta(hours=3), idea, version)
    h1 = _chart_bars(bench, "1h", start - pd.Timedelta(days=4), end + pd.Timedelta(days=1), idea, version)
    if m5 is None or h1 is None or m5.empty or h1.empty:
        raise Refused(f"trade {trade}: no bars to draw it with inside the years this run may see")
    pic = D.trade_picture(t, m5, h1, subject=f"research {idea} {version} trade {t.signal_id}",
                          sub=f"{idea} · version {version} · a trade on {TUNE} · research tools (not sent to Tejas)")
    out = _render(pic, _picture_path(idea, version, f"trade-{t.signal_id}"))
    return {**out, "trade": {"signal": t.signal_id, "direction": t.direction, "fill": str(t.fill_ts),
                             "closed": str(end), "net_inr": t.net_inr, "status": t.status}}


def _draw_window(idea: str, rec: dict, start: Any, end: Any, tf: str, bench: Workbench) -> dict:
    from machine.charts import draw as D
    version = rec["version_id"]
    if tf not in config.MCP_CHART_DAYS:
        raise Refused(f"tf = {tf!r}: a chart is one of {', '.join(config.MCP_CHART_DAYS)}")
    t0, t1 = _when(start, "start"), _when(end, "end")
    if t1 <= t0:
        raise Refused("end must come after start")
    days = config.MCP_CHART_DAYS[tf]
    if t1 - t0 > pd.Timedelta(days=days):
        raise Refused(f"a {tf} chart shows at most {days} days (config.MCP_CHART_DAYS) - readable in 5 seconds")
    if t0.year < config.TUNE_YEARS[0]:
        raise Refused(f"the data starts in {config.TUNE_YEARS[0]}")
    guards.check_opened(idea, [t0, t1], version=version)  # the check years: only once opened for THIS version
    if not check_open_for(idea, version):                  # ... and only in this run's own idea
        inside_tune("the chart", t0, t1)
    b = _chart_bars(bench, tf, t0, t1, idea, version)
    if b is None or b.empty:
        raise Refused("no bars in that stretch")
    items: list = [D.Candles()]
    marked = 0
    if b.index[-1].year <= config.TUNE_YEARS[-1]:          # its setups: the 2016-2020 signals only
        for s in signals_of(idea, rec, bench):
            k = pd.Timestamp(s.known_at)
            if b.index[0] <= k <= b.index[-1] + pd.Timedelta(minutes=1):
                px = s.entry_price if s.entry_price is not None else float(b["close"].asof(k))
                items.append(D.Marker(k, px, "S" if s.direction == "sell" else "B", s.direction, "setup"))
                marked += 1
    panel = D.Panel(b, tf, items, f"{idea} {version} setups")
    pic = D.Picture(kind="trade", subject=f"research {idea} {version} {t0} {t1} {tf}",
                    heading=f"{idea} · {version}", rows=[[panel]],
                    sub=f"{marked} setups of this version in this stretch · research tools (not sent to Tejas)")
    out = _render(pic, _picture_path(idea, version, f"{tf}-{t0:%Y%m%d%H%M}-{t1:%Y%m%d%H%M}"))
    return {**out, "setups_marked": marked, "from": str(b.index[0]), "to": str(b.index[-1]), "tf": tf}


# ================================================================================================= 8. write_report

@tool
def write_report(idea: str, send: bool = False, bench: Workbench | None = None) -> dict:
    """The report, via L10: every version of the idea on 2016-2020 (a trial file each), the referee chooses ONE by the
    highest Deflated Sharpe, the check years 2021-2023 open ONCE for the family for that version and it is judged
    (a family that used them already: walk-forward inside 2016-2020). Then the report (report.html, the picture).
    send=True also puts it in the Telegram outbox (REPORTED)."""
    bench = bench or BENCH
    _target(idea, "write_report")
    form = V.locked(idea)
    _money_gate(idea)
    if not isinstance(send, bool):
        raise Refused("send: true or false")
    files = trials.load()
    parent, _ = V.parent_of(idea, form, files)
    pool: dict[str, dict] = {}                           # every version of the idea: its tune's and the proposed
    for vid in [*V.tuned_versions(idea), *(r["version_id"] for r in V.records(idea))]:
        pool[vid] = {**version_record(idea, vid), "parent_idea": parent}
    if not pool:
        raise Refused(f"{idea} has no version yet: propose_version, then run_version, then write_report")
    fam = V.family(idea, parent, files)
    if fam["used"] > fam["budget"]:
        raise Refused(f"the {fam['root']} family has {fam['used']} versions, over the budget {fam['budget']}")
    log: list[str] = []
    sigs = {vid: signals_of(idea, r, bench) for vid, r in pool.items()}
    empty = sorted(vid for vid, s in sigs.items() if not s)
    for vid in empty:
        log.append(f"{vid} made no setup on {TUNE}: left out of the tune")
        sigs.pop(vid)
    if not sigs:
        raise Refused(f"no version of {idea} made a setup on {TUNE}: nothing to judge")
    tf0 = form.timeframe
    tuned = bench.get("tune")(sigs, tf=tf0, about={v: pool[v]["about"] for v in sigs},
                              readings={v: pool[v]["readings"] for v in sigs},
                              changes={v: pool[v]["changes"] for v in sigs},
                              tfs={v: pool[v]["timeframe"] for v in sigs},
                              note=f"L14.1 research: {idea}, {len(sigs)} versions on {TUNE}")
    _advance(idea, "TUNE RUN", f"L14.1: {len(sigs)} versions on {TUNE}; chosen {tuned.chosen}",
             f"the check years {CHECK}, opened once for the chosen version")
    opening = splits.family_opening(idea, parent)
    judged: dict[str, Any]
    if opening is None or (opening["idea_id"], opening.get("version_id")) == (idea, tuned.chosen):
        done = [f for f in trials.load() if f.get("kind") == "referee" and f.get("idea_id") == idea
                and str((f.get("settings") or {}).get("judged_on", "")).startswith("the check years")]
        if done:
            judged = {"mode": "check", "said": f"the check years were judged already ({done[0]['made_at'][:16]}) - "
                                                "they are judged ONCE; the report is built again from the files"}
        else:
            rep = bench.get("check")(idea, who=SESSION.who, log=log.append)
            _advance(idea, "CHECK OPENED", f"L14.1: {CHECK} opened once for {tuned.chosen}",
                     "the report")
            judged = {"mode": "check", "answer": rep.word, "why": rep.answer.why, "judged_on": rep.judged_on,
                      "trial": None if rep.trial_path is None else _rel(rep.trial_path)}
    else:
        key = prior.key_for_form(C.prepare(form, pool[tuned.chosen]["readings"] or None,
                                           pool[tuned.chosen]["changes"] or None)[0])
        rep = bench.get("walk_forward")(sigs, tf=pool[tuned.chosen]["timeframe"], key=key, newest=tuned.chosen,
                                        tfs={v: pool[v]["timeframe"] for v in sigs},
                                        note=f"L14.1 research: {idea} by walk-forward (the family's check years went "
                                             f"to {opening['idea_id']} {opening.get('version_id')})")
        judged = {"mode": "walk-forward", "answer": rep.word, "why": rep.answer.why, "judged_on": rep.judged_on,
                  "text": rep.text(), "trial": None if rep.trial_path is None else _rel(rep.trial_path)}
    try:
        bench.get("learn")(idea, log=log.append)
    except Exception as err:                              # noqa: BLE001 - never blocks the report (like the intake)
        log.append(f"the learner did not run ({type(err).__name__}: {err}) - 'works best in / fails in: not "
                   "measured yet'")
    built = bench.get("build")(idea)
    out = {"idea": idea, "chosen": tuned.chosen, "versions_tuned": len(sigs), "judged": judged,
           "report_text": built.text, "page": _rel(built.page_path), "picture": _rel(built.picture["path"]),
           "tune_trial": None if tuned.trial_path is None else _rel(tuned.trial_path),
           "version_budget": _budget(V.family(idea, parent)), "log": log, "sent": False}
    if send:
        line = bench.get("send")(built)
        _advance(idea, "REPORTED", f"L14.1: report sent ({line['id']})", "T6: Tejas reads the report and decides")
        out["sent"] = line["id"]
    return out


def _advance(idea: str, state: str, note: str, waits_for: str | None) -> None:
    """Move the idea FORWARD only (an idea already REPORTED or DECIDED never goes back to TUNE RUN)."""
    now = store.read_state(idea)["state"]
    if STATES.index(state) > STATES.index(now):
        store.set_state(idea, state, note, waits_for=waits_for)


# ================================================================================================= 9. choose_slices

@tool
def choose_slices(idea: str, slices: list, why: str) -> dict:
    """Choose the slices (at most 8 per idea) BEFORE any money is seen: each names a condition (list_blocks) and its
    dimension. Refused once a trial file names the idea - a split chosen after looking is not a rule (lesson 18)."""
    from machine.learner import conditions
    _target(idea, "choose_slices")
    V.locked(idea)
    if not isinstance(slices, list) or not all(isinstance(s, dict) for s in slices):
        raise Refused("slices: a list of {name, dimension, how, why}")
    for s in slices:
        try:
            Slice.model_validate(s)
        except ValidationError as err:
            raise Refused(f"slice {s.get('name')!r}: {readable(err)}") from None
        cond = conditions.BY_NAME.get(s.get("name"))
        if cond is None:
            raise Refused(f"slice {s.get('name')!r}: not a condition the learner measures - one of "
                          f"{', '.join(conditions.names())} (list_blocks). An unknown name is never guessed (R13).")
        if s.get("dimension") != cond.dimension:
            raise Refused(f"slice {s['name']!r}: its dimension is {cond.dimension!r}, not {s.get('dimension')!r}")
    have = store.add_slices(idea, slices, who=SESSION.who, why=why)
    return {"idea": idea, "slices": [x.model_dump(mode="json") for x in have],
            "slice_budget": {"chosen": len(have), "budget": config.SLICE_BUDGET},
            "said": "chosen before any money was seen: only these can become 'works best in / fails in' (section D)"}


# the research agent's PLANNING tools (L14.2, `python -m machine.mcp --plan`): reading only - no tool that proposes,
# runs, draws, reports or shows a trial file
PLAN_TOOLS = ("get_idea", "list_blocks", "search_docs")

TOOLS = {f.__name__: f for f in (get_idea, list_blocks, search_docs, propose_version, run_version, list_trials,
                                  draw_chart, write_report, choose_slices)}
