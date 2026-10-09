"""The research loop for his WHOLE strategy (L19d): Claude plans, code checks and SAVES the plan, code runs exactly
the plan, the referee judges ONE version - with his hedge.

    .venv\\Scripts\\python -m machine.strategy plan        Claude writes the plan; code checks it and SAVES it
    .venv\\Scripts\\python -m machine.strategy run         the plan's versions on 2016-2020 (one trial file each = a try)
    .venv\\Scripts\\python -m machine.strategy judge --who "..."   the referee: ONE version chosen on 2016-2020, its
                                                          2x2 and slices there, the check years 2021-2023 opened ONCE
                                                          for it, judged there; then the reports
    .venv\\Scripts\\python -m machine.strategy reports     the reports again from the files (runs nothing)
    .venv\\Scripts\\python -m machine.strategy show        the plan and the log

Who does what - the same split as the research agent for one idea (L14.2, machine/agent/research.py):
  Claude    (his Claude Code login, no API key) reads his whole strategy with the PLANNING tools only - get_strategy,
            list_blocks, search_docs (`python -m machine.mcp --strategy-plan`; no built-in tool). It writes the PLAN:
            the hypothesis, at most RESEARCH_MAX_VERSIONS versions (what a version may change: machine/strategy/
            whole.py), at most 8 slices chosen now, the stop rule in its words, or "unclear".
  code      checks the plan (whole.version: every change; the budgets; his easy-English words); a refused plan goes back
            to Claude ONCE with the reasons; then SAVES it (research-plan.json + its sha256, research-log.md) before
            any version runs. Then runs EXACTLY the plan, in its order, once each (a trial file each, kind "strategy").
  the stop  code's rule (STOP_RULE): every planned version ran · the family's 20 are used · 3 planned versions in a row
            cannot run (a session looks). Never because of money.
  referee   ONE version chosen on 2016-2020 by the highest Deflated Sharpe of its hedged trades (every try counted);
            its 2x2 and the slices on 2016-2020; the check years 2021-2023 opened ONCE for the family his-strategy, for
            that version; judged there by the referee's 6 checks (machine/referee/verdict.py) WITH HIS HEDGE.
Files: plan/work/ideas/his-strategy/ (research-plan.json, research-log.md, versions.jsonl, check-opened.json,
research-final.md); plan/work/trials/ (kinds strategy, referee-tune, strategy-2x2, referee); the reports in
plan/work/L19d/ (machine/strategy/report.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from machine import config, costs, guards, trials as trial_files
from machine.entries import forms
from machine.mcp.versions import Refused
from machine.referee import power, prior, splits, stats, trials, verdict
from machine.strategy import whole as W

STRATEGY = W.STRATEGY
PLAN, LOG, FINAL, VERSIONS = "research-plan.json", "research-log.md", "research-final.md", "versions.jsonl"
WHO = "the research agent (L14.2) for his whole strategy (L19d)"
TUNE_KIND = "referee-tune"

STOP_RULE = (
    "the planned versions run in the plan's order, each once; a version that is not in the plan never runs",
    "it stops when every planned version ran",
    f"it stops when the family's {config.VERSION_BUDGET} versions are used",
    f"it stops when {config.RESEARCH_STUCK_IN_A_ROW} planned versions in a row cannot run - then a Claude session "
    "looks, because something is broken",
    "it never stops early and never adds a version because of money: a stop that looks at results is choosing by "
    "money (lesson 18)",
    "then the referee - never the agent - chooses ONE version and judges it",
)


class ResearchError(RuntimeError):
    """The research cannot go on by itself: a Claude session must look (the message says why)."""


class NoAnswer(ResearchError):
    """The planning call got no usable answer (his plan's usage limit, no network): try again later."""


@dataclass
class Planned:
    raw: dict | None
    meta: dict = field(default_factory=dict)
    error: str | None = None


class Planner(Protocol):
    def __call__(self, *, refused: list[str] | None = None, previous: dict | None = None) -> Planned: ...


# ================================================================================================= files

def folder() -> Path:
    return Path(config.IDEAS_DIR) / STRATEGY


def plan_path() -> Path:
    return folder() / PLAN


def log_path() -> Path:
    return folder() / LOG


def versions_path() -> Path:
    return folder() / VERSIONS


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, text: str) -> None:
    """A whole file through a temp file (never open(path, "w") on a tracked file - CLAUDE.md rule 7)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def log(heading: str, body: str = "") -> None:
    """One step into research-log.md (append only)."""
    path = log_path()
    first = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        if first:
            fh.write("# Research log - his whole strategy (L19d)\n\n> Written by code, oldest first, never rewritten. "
                     "The plan: research-plan.json (saved BEFORE any version ran). The AI writes rules; code judges."
                     "\n\n")
        fh.write(f"## {_now()[:16].replace('T', ' ')} · {heading}\n\n" + (body.rstrip() + "\n\n" if body else ""))


def sha_of(plan: dict) -> str:
    body = {k: v for k, v in plan.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def load_plan() -> dict | None:
    path = plan_path()
    if not path.is_file():
        return None
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("sha256") != sha_of(plan):
        raise ResearchError(f"{path} was changed after it was saved (its sha256 does not match) - a plan is never "
                            "edited; a session looks")
    return plan


def records() -> list[dict]:
    p = versions_path()
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def files(kind: str | None = None) -> list[dict]:
    """His whole strategy's trial files (oldest first), of one kind or all."""
    return [f for f in trials.load() if f.get("idea_id") == STRATEGY and (kind is None or f.get("kind") == kind)]


def budget() -> dict:
    """The family's tries: every version with a trial file, plus every version proposed and not run yet."""
    tried: list[str] = []
    for f in files():
        if f["version_id"] not in tried:
            tried.append(f["version_id"])
    waiting = [r["version_id"] for r in records() if r["version_id"] not in tried]
    used = len(tried) + len(waiting)
    left = config.VERSION_BUDGET - used
    return {"family": STRATEGY, "used": used, "tried": len(tried), "proposed_not_run": len(waiting),
            "budget": config.VERSION_BUDGET, "left": left, "cap": max(0, min(config.RESEARCH_MAX_VERSIONS, left))}


def meaning() -> str:
    """What the whole IS today (the entries in it and their forms): a new answer of his (an entry joins) = a new
    meaning = a new plan."""
    body = {"in": {e.id: e.form for e in W.entries_in()}, "waiting": sorted(W.waiting())}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


# ================================================================================================= the plan's shape

_TEXT = {"type": "string"}
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hypothesis": _TEXT,
        "versions": {"type": "array", "items": {"type": "object", "properties": {
            "name": _TEXT, "why": _TEXT,
            "changes": {"type": "array", "items": {"type": "object", "properties": {
                "path": _TEXT, "value": {"type": ["number", "string", "boolean", "null"]}},
                "required": ["path", "value"]}}},
            "required": ["name", "why", "changes"]}},
        "slices": {"type": "array", "items": {"type": "object", "properties": {
            "name": _TEXT, "dimension": _TEXT, "how": _TEXT, "why": _TEXT},
            "required": ["name", "dimension", "how", "why"]}},
        "stop_rule": _TEXT,
        "unclear": {"type": "array", "items": {"type": "object", "properties": {
            "what": _TEXT, "his_words": _TEXT, "readings": {"type": "array", "items": _TEXT},
            "why_trades_differ": _TEXT, "searched": _TEXT},
            "required": ["what", "his_words", "readings", "why_trades_differ", "searched"]}},
    },
    "required": ["hypothesis", "versions", "slices", "stop_rule", "unclear"],
}


def plan_schema() -> dict:
    """PLAN_SCHEMA with a slice's name and dimension held to the real ones, so Claude's answer cannot name a slice
    with a sentence - as the research agent's (machine/agent/research.py; idea #3's plan was refused twice for it in
    the 8 Oct rehearsal, L10b)."""
    from machine.ideas.form import SLICE_DIMENSIONS
    from machine.learner import conditions
    schema = json.loads(json.dumps(PLAN_SCHEMA))
    props = schema["properties"]["slices"]["items"]["properties"]
    props["name"] = {"type": "string", "enum": sorted(conditions.BY_NAME)}
    props["dimension"] = {"type": "string", "enum": list(SLICE_DIMENSIONS)}
    return schema


def _value(v: Any) -> Any:
    """A change's value as Claude typed it -> what it means ("3" -> 3, 3.0 -> 3, "true" -> True, "none" -> None)."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if s.lower() in ("none", "null"):
            return None
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?(\d+\.\d*|\.\d+)", s):
            return _value(float(s))
        return s
    return v


def _words_problems(text: str, what: str) -> list[str]:
    from machine.report.build import check_words
    return [f"{what}: {p}" for p in check_words(text)]


def check_plan(raw: dict) -> tuple[dict, list[str]]:
    """Claude's plan -> the plan code will run, and every problem (empty: it may be saved). Nothing is proposed or
    counted here."""
    from machine.ideas.form import Slice
    from machine.learner import conditions
    problems: list[str] = []
    b = budget()
    hyp = str(raw.get("hypothesis") or "").strip()
    stop = str(raw.get("stop_rule") or "").strip()
    unclear = [u for u in raw.get("unclear") or [] if str(u.get("what") or "").strip()]
    plan: dict[str, Any] = {"strategy": STRATEGY, "hypothesis": hyp, "versions": [], "slices": [],
                            "stop_rule": {"in_words": stop, "code": list(STOP_RULE)}, "unclear": unclear,
                            "entries_in": [e.id for e in W.entries_in()], "entries_waiting": W.waiting(),
                            "budget": b}
    if not hyp:
        problems.append("the hypothesis is empty: one sentence - what must be true on 2016-2020 if his strategy works")
    if len(hyp) > 600:
        problems.append(f"the hypothesis is {len(hyp)} characters - ONE sentence, at most 600")
    if not stop:
        problems.append("the stop rule is empty: say in your words when this research stops")
    problems += _words_problems(hyp, "the hypothesis") + _words_problems(stop, "the stop rule")
    if unclear:                                            # a question instead of versions: nothing runs (R13)
        for k, u in enumerate(unclear, 1):
            if len([r for r in u.get("readings") or [] if str(r).strip()]) < 2:
                problems.append(f"unclear {k} ({u['what']}): give 2 to 4 readings - a question needs choices")
        return plan, problems
    raw_versions = raw.get("versions") or []
    if not raw_versions:
        problems.append("no version: at least his strategy as written")
    if len(raw_versions) > b["cap"]:
        problems.append(f"{len(raw_versions)} versions: at most {b['cap']} (the family has {b['left']} of "
                        f"{b['budget']} left; a plan uses at most {config.RESEARCH_MAX_VERSIONS} - the rest stay for "
                        "his own change after the report)")
    tried = {f["version_id"] for f in files()}
    first: dict[str, int] = {}
    for k, v in enumerate(raw_versions, 1):
        name = str(v.get("name") or "").strip() or f"version {k}"
        why = str(v.get("why") or "").strip()
        changes = {str(c.get("path")): _value(c.get("value")) for c in v.get("changes") or []}
        if not why:
            problems.append(f"version {k} ({name}): no reason (why)")
        problems += _words_problems(f"{name}. {why}", f"version {k}")
        try:
            ver = W.version(changes)
        except Refused as err:
            problems.append(f"version {k} ({name}): REFUSED - {err}")
            continue
        if ver.version_id in first:
            problems.append(f"version {k} ({name}) is the same rule as version {first[ver.version_id]} "
                            f"({ver.version_id}) - one rule is one try")
            continue
        first[ver.version_id] = k
        plan["versions"].append({"k": k, "name": name, "why": why, "changes": ver.changes,
                                 "version_id": ver.version_id, "about": ver.about, "dimensions": ver.dimensions,
                                 "new_try": ver.version_id not in tried})
    if raw_versions and W.version({}).version_id not in first:
        problems.append("his strategy as written (no change) is not in the plan - version 1 is always his own way")
    new_tries = sum(v["new_try"] for v in plan["versions"])
    if new_tries > b["left"]:
        problems.append(f"{new_tries} new tries: the family has {b['left']} left")
    names: set[str] = set()
    raw_slices = raw.get("slices") or []
    if len(raw_slices) > config.SLICE_BUDGET:
        problems.append(f"{len(raw_slices)} slices: at most {config.SLICE_BUDGET}")
    for s in raw_slices:
        n = str(s.get("name"))
        try:
            Slice.model_validate(s)
        except Exception as err:                           # noqa: BLE001 - a pydantic error, said in words
            problems.append(f"slice {n!r}: {err}")
            continue
        cond = conditions.BY_NAME.get(n)
        if cond is None:                                   # said so that the fix-up call can fix it
            same = [c.name for c in conditions.ALL if c.dimension == s.get("dimension")]
            problems.append(f"slice {n!r}: not a condition the learner measures (list_blocks' slice_conditions). Its "
                            "\"name\" must be EXACTLY one of the short names there - the word before the colon, like "
                            "session, dial, friday"
                            + (f"; with the dimension {s.get('dimension')!r}: {', '.join(same)}" if same else "")
                            + ". Your own words go into \"how\" and \"why\"")
            continue
        if s.get("dimension") != cond.dimension:
            problems.append(f"slice {n!r}: its dimension is {cond.dimension!r}, not {s.get('dimension')!r}")
            continue
        if n in names:
            problems.append(f"slice {n!r} is chosen twice")
            continue
        names.add(n)
        problems += _words_problems(f"{s.get('how')}. {s.get('why')}", f"slice {n!r}")
        plan["slices"].append({k: s.get(k) for k in ("name", "dimension", "how", "why")})
    return plan, problems


# ================================================================================================= Claude plans

def planning_server() -> dict:
    return {"command": sys.executable, "args": ["-u", "-m", "machine.mcp", "--strategy-plan"],
            "env": {"PYTHONPATH": str(config.ROOT), "MLBOT_MCP_DIR": str(config.MCP_DIR),
                    "PYTHONIOENCODING": "utf-8"}}


def planner_system() -> str:
    banned = "leftover, arm, verdict, veto, causal, placebo, mirror, baseline, R, basket"
    stop = "\n".join(f"  - {x}" for x in STOP_RULE)
    return f"""You are the research agent of Tejas's idea machine (L14.2), planning research on his WHOLE strategy
(L19d). Tejas is an expert gold (XAUUSD) trader who never uses a stop loss: he hedges (opens the opposite trade of the
same size). The machine holds his entry types as forms; his whole strategy = his settled entries TOGETHER, one trade at
a time across all of them, each trade managed with his hedge and his management table (T12). You only plan. Code
checks your plan, SAVES it, and then runs exactly it - nothing else. Nothing runs and you see no money while you plan.
The AI writes rules; code judges.

Do this, with your tools:
1. get_strategy() - his entries in the whole (his words, steps, entry, hedge, target), the entries that WAIT for his
   answers (never yours), one trade at a time, his fixed management rows, the management rows a version may change
   ("management_a_version_may_change": name, path, version 1, values, needs), what a version may change, the budget.
2. list_blocks() - "settings_a_version_may_change" (numbers the machine chose), "slice_conditions" (a slice's name and
   its dimension), "his_fixed_rules" (never changed).
3. search_docs(query) - his rule #129: before you choose a value, search his documents for what he said about it (the
   hedge, room, Friday, news, the cap of hedges, the trend escape, double profit, a zone as the hedge point, ...).
4. Answer with the plan, in the answer's fixed shape.

The plan:
- hypothesis: ONE sentence in easy English - what must be true on 2016-2020 gold if his whole strategy works. No
  promise of profit.
- versions: at most the number the message gives. Each version is a different rule, and ONE TRY: every try makes it
  harder for the referee to call a real edge real, so fewer is fine.
  - changes: a list of {{path, value}}. Allowed paths: "exit.management.<name>" (his management's TEST rows, with a
    value from its list), "trades_at_once" = "one_per_entry" (one trade at a time per entry type - his rule is one
    across everything; he never approved this), "settings.<name>" (a number the machine chose), "<entry id>/<path>"
    (one entry's own wait or number, e.g. "entry-type1/steps.gap.max_wait_bars"). Never his fixed rules or his fixed
    management, never a timeframe, never an entry that waits for his answer, never a reading of his open words.
  - Version 1 = his strategy as written: NO change (changes = []). It must be in the plan.
  - Then one change at a time, most important first. Every value needs a reason from his words or his documents - say
    the place. Never a value "to see if it makes money": no money is known, and that is how luck is mistaken for an
    edge.
- slices: at most the number the message gives, chosen NOW, before any money - only these can later become "works best
  in / fails in". "name" is a condition's OWN name exactly as list_blocks' slice_conditions writes it (e.g. "session",
  "news_near", "dial", "friday", "zone_at_target") - never your own words; "dimension" is that condition's dimension;
  put your words in "how" and "why". Choose those his words or documents say matter (the session, news, the trend,
  the place ...). Fewer is fine: his hedged trades last weeks, so there are few.
- stop_rule: one or two sentences in easy English: when this research stops. Code's rule is fixed - say it in your
  words:
{stop}
- unclear: ONLY when a word of his changes which trades are taken or how a trade is managed, no value for it is in
  get_strategy, AND search_docs did not find his meaning. Then: what is unclear, his words, 2 to 4 readings, why they
  trade differently, what the search found. Never guess a meaning. The three entries that wait for his answers are
  NOT unclear here: they are already asked. Usually "unclear" is empty.

Easy English (his rule #113) in every sentence you write: short words, short sentences. Never these code words:
{banned}. No "p-value", "n =", "t =". Say money in rupees. Do not use the words Sharpe, bootstrap, walk-forward,
percentile, PBO, LightGBM or SHAP."""


def planner_user(refused: list[str] | None = None, previous: dict | None = None) -> str:
    b = budget()
    text = (f"Write the research plan for his whole strategy ({STRATEGY}).\n"
            f"- versions: at most {b['cap']} (the family has {b['left']} of {b['budget']} tries left; a plan uses at "
            f"most {config.RESEARCH_MAX_VERSIONS} - the rest stay for his own change after the report)\n"
            f"- slices: at most {config.SLICE_BUDGET}")
    if refused:
        text += ("\n\nCode REFUSED your plan. The reasons:\n" + "\n".join(f"- {p}" for p in refused) +
                 "\n\nYour plan was:\n" + json.dumps(previous, ensure_ascii=False) +
                 "\n\nWrite the whole plan again with these fixed. Change nothing else.")
    return text


def claude_planner(*, refused: list[str] | None = None, previous: dict | None = None) -> Planned:
    """One planning call: Claude Code with the planning tools only (machine/talker/llm.agent_call)."""
    from machine.mcp import tools as T
    from machine.talker import llm
    a = llm.agent_call(system=planner_system(), user=planner_user(refused, previous), schema=plan_schema(),
                       servers={"research": planning_server()},
                       allowed=tuple(f"mcp__research__{t}" for t in T.STRATEGY_PLAN_TOOLS),
                       model=config.RESEARCH_MODEL, effort=config.RESEARCH_EFFORT,
                       timeout=config.RESEARCH_PLAN_SECONDS)
    meta = {"model": a.model, "via": "Claude Code - his login (no API key, no bill per call)", "turns": a.turns,
            "seconds": a.seconds, "api_price_usd": a.api_price_usd,
            "tool_calls": [{"tool": c["tool"], "input": c["input"], "error": c["error"],
                            "said": (c["said"] or "")[:200]} for c in a.tool_calls]}
    return Planned(a.output, meta, a.error)


# ================================================================================================= make the plan

def make_plan(*, planner: Planner | None = None, say: Callable = print) -> dict:
    """The saved plan for the whole's current meaning - made now if there is none (Claude plans, code checks; ONE
    fix-up call with the reasons; then SAVED before anything runs)."""
    now = meaning()
    old = load_plan()
    if old is not None:
        if old["meaning"] == now:
            return old
        if records():
            raise ResearchError("his whole strategy changed (an entry joined or left) after versions of its plan were "
                                "proposed - a new meaning after money is a new family; a session looks")
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        plan_path().replace(folder() / f"research-plan-replaced-{stamp}.json")
        log("the plan was put aside (never run)", f"the whole's meaning changed. Kept as research-plan-replaced-"
                                                  f"{stamp}.json.")
    if files():
        raise ResearchError("money was already seen for his whole strategy (a trial file exists) - a research plan "
                            "comes BEFORE any money (lesson 18); a session looks")
    planner = planner or claude_planner
    calls = []
    got = planner()
    calls.append(got.meta)
    if got.raw is None:
        raise NoAnswer(f"the planning call got no answer: {got.error}")
    plan, problems = check_plan(got.raw)
    fixups = 0
    while problems and fixups < config.RESEARCH_FIXUPS:
        fixups += 1
        say(f"code refused the plan ({len(problems)} reasons) - one more call with the reasons")
        log(f"code refused the plan ({len(problems)} reasons) - Claude gets them once",
            "\n".join(f"- {p}" for p in problems))
        prev = got.raw
        got = planner(refused=problems, previous=prev)
        calls.append(got.meta)
        if got.raw is None:
            raise NoAnswer(f"the fix-up call got no answer: {got.error}")
        plan, problems = check_plan(got.raw)
    if problems:
        log("code refused the plan again - nothing runs", "\n".join(f"- {p}" for p in problems))
        raise ResearchError(f"code refused the plan {fixups + 1} times: " + "; ".join(problems)[:800])
    plan.update(made_at=_now(), meaning=now, money_seen_before=[], made_by={"agent": WHO, "calls": calls,
                                                                            "fixups": fixups}, raw=got.raw)
    if plan_path().exists():
        raise ResearchError("a saved plan exists already - it is never rewritten")
    plan["sha256"] = sha_of(plan)
    _write(plan_path(), json.dumps(plan, indent=1, ensure_ascii=False, default=str) + "\n")
    log("the plan was saved (before any version ran)", plan_text(plan))
    say(f"plan {plan['sha256'][:12]} saved - {len(plan['versions'])} versions, {len(plan['slices'])} slices"
        + (f", {len(plan['unclear'])} unclear" if plan["unclear"] else ""))
    return plan


def plan_text(plan: dict) -> str:
    out = [f"**What must be true if his whole strategy works:** {plan['hypothesis']}", "",
           f"**His entries in the whole:** {', '.join(plan['entries_in'])}. **Waiting for his words:** "
           + "; ".join(f"{k} - {v}" for k, v in plan["entries_waiting"].items()), ""]
    if plan["unclear"]:
        out.append("**Unclear - asked, never guessed (no version runs until it is answered):**")
        for u in plan["unclear"]:
            out.append(f"- {u['what']} - his words: \"{u['his_words']}\" - readings: " + " / ".join(u["readings"])
                       + f" - why they trade differently: {u['why_trades_differ']}")
        out.append("")
    b = plan["budget"]
    if plan["versions"]:
        out += [f"**Versions:** {len(plan['versions'])} (each one is a try; the {STRATEGY} family had used {b['used']} "
                f"of {b['budget']} before this plan)", "", "| # | version | what it is | why |", "|---|---|---|---|"]
        for v in plan["versions"]:
            out.append(f"| {v['k']} | {v['name']} (`{v['version_id']}`) | {v['about']} | {v['why']} |")
        out.append("")
    if plan["slices"]:
        out.append(f"**Slices chosen before any money ({len(plan['slices'])}):**")
        out += [f"- {s['name']} ({s['dimension']}): {s['how']} - {s['why']}" for s in plan["slices"]]
        out.append("")
    out.append(f"**When it stops:** {plan['stop_rule']['in_words']}")
    out.append("Code's rule: " + "; ".join(plan["stop_rule"]["code"]) + ".")
    for c in (plan.get("made_by") or {}).get("calls") or []:
        searched = [f"\"{(t.get('input') or {}).get('query')}\"" + (" (refused)" if t.get("error") else "")
                    for t in c.get("tool_calls", []) if t.get("tool") == "search_docs"]
        out.append(f"\n**Made by:** {c.get('model')} through {c.get('via', 'Claude Code - his login')}, "
                   f"{c.get('turns')} turns, {c.get('seconds')} s; tools: "
                   + (", ".join(t["tool"] for t in c.get("tool_calls", [])) or "none")
                   + (f"; it searched his documents for {', '.join(searched)}" if searched else ""))
    if plan.get("sha256"):
        out.append(f"\n**The plan's fingerprint (sha256):** `{plan['sha256']}` - saved {plan.get('made_at', '')}")
    return "\n".join(out)


# ================================================================================================= run the plan

def _data() -> dict:
    from machine.data import bars as BARS
    return {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
            "gold_m1_content_sha256": BARS.m1_fingerprint()}


def _code_sha() -> str:
    from machine.entries import closers
    from machine.tester import engine, hedging
    from machine.tester import management as MG
    return trial_files.code_sha256([Path(W.__file__), Path(__file__), Path(closers.__file__), Path(forms.__file__),
                                    Path(hedging.__file__), Path(engine.__file__), Path(MG.__file__)])


def propose(v: W.Version, plan: dict, k: int, name: str, why: str) -> dict:
    """A planned version into versions.jsonl (append only) - it counts as a try from now on."""
    for r in records():
        if r["version_id"] == v.version_id:
            return r
    rec = {"idea_id": STRATEGY, "version_id": v.version_id, "plan": plan["sha256"][:12], "k": k, "name": name,
           "why": why, **{x: y for x, y in v.as_dict().items() if x != "version_id"},
           "proposed_at": _now(), "who": WHO}
    with open(versions_path(), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def record_run(run_: W.Run, plan: dict, k: int, name: str, why: str) -> Path:
    """ONE trial file for a version run on 2016-2020 (kind "strategy") - the try."""
    s = W.run_summary(run_)
    v = run_.version
    sett = {**v.as_dict(), "plan": plan["sha256"], "k": k, "name": name, "why": why,
            "window": s.pop("window"), "timeframe": W.TF, "rules": s.pop("rules"),
            "management_table": "plan/work/management-decisions.md (T12)",
            "management_stop": "the tester (L3): a stop at the hedge point, the same target (M02's other way)",
            "what_it_is": W.WHAT_IT_IS, "entries_waiting": W.waiting(),
            "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
            "news": f"Forex Factory red + orange, exact times: {len(run_.facts.news)} moments in the window",
            "already_seen": prior.already_seen(W.SEEN_KEY, config.TUNE_YEARS).as_dict(),
            "code_sha256": _code_sha()}
    t = W.closed_hedged(run_.hedged.trades)
    s["hedged_trades"] = [[r.signal, str(r.exit_time), round(float(r.net_inr), 2)] for r in t.itertuples()]
    s["open_at_end_ids"] = s["hedged"].get("open_at_end_ids", [])
    return trial_files.record(kind=W.KIND, idea_id=STRATEGY, version_id=v.version_id, parent_idea=None, settings=sett,
                              data=_data(), result=s,
                              note=f"L19d: his whole strategy, research plan {plan['sha256'][:12]} version {k} "
                                   f"({name}): {v.about} - one try of the family {STRATEGY}; already seen by the old "
                                   "project on 2016-2023")


def _line(s: dict) -> str:
    h, p = s["hedged"], s["stop"]
    return (f"with his hedge {h['trades']} trades ({h['open_at_end']} open at the end), {costs.inr(h['net_inr_total'])}"
            f"; with a plain stop {p['trades']} trades, {costs.inr(p['net_inr_total'])}")


def run_plan(*, say: Callable = print, win: W.Window | None = None) -> dict:
    """Every planned version on 2016-2020, in the plan's order, once each (the stop rule)."""
    plan = load_plan()
    if plan is None:
        raise ResearchError("no saved plan: python -m machine.strategy plan")
    if plan["unclear"]:
        raise ResearchError("the plan has unclear meanings - they go to him first; nothing runs (R13)")
    done = {f["version_id"] for f in files(W.KIND)}
    stuck, ran, stopped = 0, [], None
    n = len(plan["versions"])
    for p in plan["versions"]:
        k, vid = p["k"], p["version_id"]
        if vid in done:
            ran.append(vid)
            say(f"version {k}/{n} {vid}: already run - not run again")
            continue
        try:
            v = W.version(p["changes"])
            if v.version_id != vid:
                raise Refused(f"it now is {v.version_id}, not the plan's {vid} - the strategy or the code changed")
            b = budget()
            if vid not in {r["version_id"] for r in records()} and b["left"] <= 0:
                stopped = f"the family's {config.VERSION_BUDGET} versions are used"
                break
            propose(v, plan, k, p["name"], p["why"])
            win = win or W.Window.tune(log=say)
            t0 = time.time()
            r = W.run(win, v, log=say)
            path = record_run(r, plan, k, p["name"], p["why"])
        except Refused as err:
            stuck += 1
            log(f"version {k} of {n} could not run", f"{p['name']} (`{vid}`): {err}")
            say(f"version {k}/{n} could not run - {err}")
            if stuck >= config.RESEARCH_STUCK_IN_A_ROW:
                stopped = f"{stuck} planned versions in a row could not run - a session looks"
                break
            continue
        stuck = 0
        ran.append(vid)
        s = json.loads(path.read_text(encoding="utf-8"))["result"]
        log(f"version {k} of {n} ran on 2016-2020", f"{p['name']} (`{vid}`): {_line(s)} ({time.time() - t0:.0f} s). "
                                                    f"Trial file: {path.name}")
        say(f"version {k}/{n} {vid} ({time.time() - t0:.0f} s): {_line(s)}")
    stopped = stopped or "every planned version ran"
    log(f"the versions stopped: {stopped}", f"{len(ran)} of {n} planned versions ran.")
    return {"ran": ran, "stopped": stopped, "window": win}


# ================================================================================================= the referee

def _tune_trades(f: dict) -> pd.DataFrame:
    rows = f["result"]["hedged_trades"]
    return pd.DataFrame({"net_inr": [float(x[2]) for x in rows], "exit_time": pd.to_datetime([x[1] for x in rows])})


def choose(*, say: Callable = print) -> dict:
    """ONE version on 2016-2020: the highest Deflated Sharpe of its hedged trades, allowing for every try of the
    family (stats.choose_version) - never the agent's pick. Written once (kind referee-tune)."""
    got = files(TUNE_KIND)
    if got:
        return got[-1]
    plan = load_plan()
    runs = {f["version_id"]: f for f in files(W.KIND)}
    planned = [p for p in plan["versions"] if p["version_id"] in runs]
    if not planned:
        raise ResearchError("no planned version has run yet: python -m machine.strategy run")
    vt = {p["version_id"]: _tune_trades(runs[p["version_id"]]) for p in planned}
    few = [vid for vid, t in vt.items() if len(t) < 2]
    for vid in few:
        vt.pop(vid)
    if not vt:
        raise ResearchError("no version closed 2 hedged trades on 2016-2020 - nothing to choose by")
    b = budget()
    ch = stats.choose_version(vt, n_tries=b["used"])
    about = {p["version_id"]: p for p in planned}
    rows = {vid: {**ch["versions"][vid], "k": about[vid]["k"], "name": about[vid]["name"],
                  "about": about[vid]["about"], "net_inr": runs[vid]["result"]["hedged"]["net_inr_total"],
                  "per_year": {y: d["net_inr"] for y, d in runs[vid]["result"]["hedged"]["per_year"].items()}}
            for vid in ch["versions"]}
    result = {"chosen": ch["chosen"], "versions": rows, "tries": ch["tries"], "pbo": ch["pbo"], "years": ch["years"],
              "left_out_too_few": few, "how": "the highest Deflated Sharpe of the hedged trades (the chance the edge is "
                                              "real, allowing for every try of the family)"}
    path = trial_files.record(kind=TUNE_KIND, idea_id=STRATEGY, version_id=ch["chosen"], parent_idea=None,
                              settings={"plan": plan["sha256"], "years": list(config.TUNE_YEARS),
                                        "judged_with": "his hedge (the hedging tester, L19a)",
                                        "verdict_rules_sha256": verdict.RULES_SHA256}, data=_data(), result=result,
                              note=f"L19d: the referee chose ONE version of his whole strategy on 2016-2020: "
                                   f"{ch['chosen']}")
    log("the referee chose ONE version on 2016-2020", f"`{ch['chosen']}` ({about[ch['chosen']]['name']}) by the "
        f"highest Deflated Sharpe of {len(vt)} versions ({b['used']} tries in the family). Trial file: {path.name}")
    say(f"chosen on 2016-2020: {ch['chosen']} - {about[ch['chosen']]['name']}")
    return json.loads(path.read_text(encoding="utf-8"))


def slices_of(win: W.Window, run_: W.Run, plan: dict, *, say: Callable = print) -> dict:
    """The slices the plan chose before any money, tested by the learner's rules (L13.1) on the chosen version's
    hedged trades of 2016-2020. Every other condition: looked at, not tested."""
    from machine import settings
    from machine.ideas.compile import compile_idea
    from machine.ideas.form import Slice
    from machine.learner import features as F
    from machine.learner import run as LR
    trades = W.closed_hedged(run_.hedged.trades).set_index("signal")[["net_inr", "exit_time"]]
    if not plan["slices"]:
        return {"slices": [], "said": "the plan chose no slice"}
    news = F.his_news()
    parts = []
    for e in run_.version.ins:
        c = compile_idea(e.idea(), win.market, changes=run_.version.entry_changes(e) or None, trace=True)
        with settings.using(c.resolved):
            parts.append(F.build(F.setups_from(c, win.market), win.market, fp=settings.fingerprint(c.resolved),
                                 form=c.form, news=news))
        say(f"  features for {e.id}: {len(parts[-1])} setups")
    feats = pd.concat(parts)
    form = SimpleNamespace(slices=[Slice.model_validate(s) for s in plan["slices"]])
    return LR.analyse(form, trades, feats)


def two_by_two_record(win: W.Window, run_: W.Run, chosen: dict, *, n_runs: int, say: Callable = print,
                      slices: dict | None = None) -> Path:
    tt = W.two_by_two(win, run_, n_runs, log=say)
    nets = tt.pop("random_hedged_nets")
    s = W.run_summary(run_)
    result = {"two_by_two": tt, "summary": {"hedged": s["hedged"], "stop": s["stop"], "by_entry": s["by_entry"]},
              "random_hedged_nets_count": len(nets), "slices": slices}
    sett = {"window": s["window"], "runs": n_runs, "seed": tt["seed"], "chosen_by": chosen["trial_id"],
            "random_entries": {"keeps": "month, direction, entry type, hedge and target distances, cancel distance, "
                                        "waiting time, lots; the SAME twins in both managements; one at a time the "
                                        "version's way", "changes": "the time (a random bar close in the same month) "
                                                                    "and a limit's distance (drawn from his own); no "
                                                                    "checks while the order waits"},
            "version": run_.version.as_dict(), "code_sha256": _code_sha()}
    path = trial_files.record(kind=W.KIND_2X2, idea_id=STRATEGY, version_id=run_.version.version_id,
                              parent_idea=None, settings=sett, data=_data(), result=result,
                              note=f"L19d: the 2x2 of his whole strategy's chosen version on {run_.window} (the same "
                                   "version - not a new try)")
    _trades_csv(run_)
    return path


def _trades_csv(run_: W.Run) -> None:
    from machine.tester import hedging
    from machine.tester.run import trades_table
    out = config.WORK_DIR / "L19d" / "trades"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{run_.version.version_id}-{run_.window}"
    hedging.trades_table(run_.hedged.trades).to_csv(out / f"{tag}-hedged.csv", index=False)
    trades_table(run_.stop.trades).to_csv(out / f"{tag}-plain-stop.csv", index=False)


def judge(*, who: str, n_runs: int = config.RANDOM_RUNS, say: Callable = print) -> dict:
    """The referee, start to end: choose ONE version (2016-2020), its 2x2 and slices on 2016-2020, the check years
    opened ONCE for it, compiled again on 2016-2023 (its 2016-2020 setups must be exactly the tune run's), judged on
    2021-2023 by the referee's 6 checks with his hedge, its 2x2 there; one trial file each. Then the reports."""
    if not str(who).strip():
        raise ResearchError("who opens the check years? (it goes into check-opened.json)")
    plan = load_plan()
    chosen = choose(say=say)
    vid = chosen["result"]["chosen"]
    p = next(x for x in plan["versions"] if x["version_id"] == vid)
    v = W.version(p["changes"])
    if v.version_id != vid:
        raise ResearchError(f"the chosen version now is {v.version_id}, not {vid}: the strategy or the code changed")
    tune_file = next(f for f in files(W.KIND) if f["version_id"] == vid)
    if not [f for f in files(W.KIND_2X2) if f["version_id"] == vid and f["settings"]["window"]["years"] == "2016-2020"]:
        win = W.Window.tune(log=say)
        r = W.run(win, v, log=say)
        if W.run_summary(r)["signals_sha256"] != tune_file["result"]["signals_sha256"]:
            raise ResearchError("the chosen version's setups on 2016-2020 differ from its own run's - stop and find why")
        sl = slices_of(win, r, plan, say=say)
        path = two_by_two_record(win, r, chosen, n_runs=n_runs, say=say, slices=sl)
        log("the chosen version's 2x2 and slices on 2016-2020", f"Trial file: {path.name}")
        del win, r
    done = [f for f in files("referee") if f["version_id"] == vid]
    if done:
        say(f"the check years were judged already ({done[0]['made_at'][:16]}) - they are judged ONCE")
        return done[-1]
    opening = splits.family_opening(STRATEGY)
    if opening is None:
        b = budget()
        rec = splits.open_check(STRATEGY, vid, None, who=who,
                                why=f"{vid} ({p['name']}) chosen on 2016-2020 by the highest Deflated Sharpe of its "
                                    f"hedged trades, {b['used']} tries in the family (trial {chosen['trial_id']})")
        log("the check years 2021-2023 were opened ONCE", f"for `{vid}` by {who} ({rec['opened_at']})")
        say(f"check years opened for {vid} by {who}")
    elif opening.get("version_id") != vid:
        raise ResearchError(f"the family opened its check years for {opening.get('version_id')}, not {vid}")
    win = W.Window.check(vid, log=say)
    tune_part = W.tune_part(win, v)
    if {e: W.signals_sha256(s) for e, s in tune_part.items()} != tune_file["result"]["signals_sha256"]:
        raise ResearchError("compiled on 2016-2023, the chosen version's 2016-2020 setups are not the tune run's: a "
                            "setup read bars after its moment - stop and find it")
    say("its 2016-2020 setups compiled on 2016-2023 are exactly the tune run's (nothing read the future)")
    r = W.run(win, v, log=say)
    return judged_record(win, r, chosen, tune_file, n_runs=n_runs, say=say)


def judged_record(win: W.Window, r: W.Run, chosen: dict, tune_file: dict, *, n_runs: int,
                  say: Callable = print) -> dict:
    """The chosen version's run on the check years -> its 2x2 there, the referee's 6 checks with his hedge, ONE trial
    file (kind "referee"), its trades, the log line."""
    v = r.version
    vid = v.version_id
    tt = W.two_by_two(win, r, n_runs, log=say)
    nets = tt.pop("random_hedged_nets")
    rep = _verdict(r, tt, nets)
    s = W.run_summary(r)
    result = {"answer": rep["answer"].as_dict(), "per_year": s["hedged"]["per_year"], "summary": s["hedged"],
              "stop": s["stop"], "by_entry": s["by_entry"], "two_by_two": tt, "power": rep["power"],
              "already_seen": rep["seen"].as_dict(), "luck_test_p": rep["p"], "without_best": rep["without_best"],
              "fake_levels": rep["fake"], "random_entries": tt["random_hedged"],
              "hedged_trades": [[x.signal, str(x.exit_time), round(float(x.net_inr), 2)]
                                for x in W.closed_hedged(r.hedged.trades).itertuples()]}
    sett = {"judged_on": f"the check years {config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]} (opened once for "
                         f"{STRATEGY} {vid})", "judged_with": "his hedge (the hedging tester, L19a) - his strategy",
            "version": v.as_dict(), "window": s["window"], "runs": n_runs, "seed": tt["seed"],
            "chosen_by": chosen["trial_id"], "verdict_rules_sha256": verdict.RULES_SHA256,
            "tune_signals_sha256": tune_file["result"]["signals_sha256"], "code_sha256": _code_sha()}
    path = trial_files.record(kind="referee", idea_id=STRATEGY, version_id=vid, parent_idea=None, settings=sett,
                              data=_data(), result=result,
                              note=f"L19d: his whole strategy's chosen version judged on the check years with his "
                                   f"hedge: {rep['answer'].word}")
    _trades_csv(r)
    log(f"the referee on 2021-2023: {rep['answer'].word}", f"{rep['answer'].why}. Trial file: {path.name}")
    say(f"the referee on 2021-2023: {rep['answer'].word} ({rep['answer'].why})")
    return json.loads(path.read_text(encoding="utf-8"))


FAKE_LEVELS = {"applies": False, "why": "his whole strategy has three entries, each with its own levels - there is "
                                        "no one level to move"}


def _verdict(r: W.Run, tt: dict, nets: list[float]) -> dict:
    """The referee's 6 checks (machine/referee/verdict.py) on the hedged trades of the check years."""
    t = W.closed_hedged(r.hedged.trades)
    per = float(t["net_inr"].mean()) if len(t) else None
    p = W.week_luck(t)
    weeks = max(1.0, (pd.Timestamp(r.hedged.last_bar) - pd.Timestamp(r.hedged.first_bar)) / pd.Timedelta(days=7))
    noise = nets if len(nets) >= 2 else (t["net_inr"].tolist() if len(t) >= 2 else [0.0, 0.0])
    pw = power.table(noise, len(t), len(t) / weeks, noise_from="the random twins with his hedge")
    wb = stats.without_best(t) if len(t) else {"left_out": config.WITHOUT_BEST, "trades": 0, "net_inr": 0.0,
                                               "per_trade": None, "best_inr": []}
    seen = prior.already_seen(W.SEEN_KEY, config.CHECK_YEARS)
    ev = verdict.Evidence(trades=len(t), per_trade=per, p=1.0 if p is None else p, random=tt["random_hedged"],
                          fake=FAKE_LEVELS, without_best=wb, seen=seen.seen,
                          needed_edge_caught=pw["needed_edge_caught"],
                          judged_on=f"{config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]}")
    return {"answer": verdict.decide(ev), "power": pw, "seen": seen, "p": p, "without_best": wb, "fake": FAKE_LEVELS}


DECISION = "decision.md"


def decide(*, say: Callable = print) -> str:
    """After the report the machine chooses what comes next by the plan's default and TELLS him - never asks (his
    words 1 Oct, LEARNED #162; machine.intake.state.choice is the same rule for an idea). Written once
    (plan/work/ideas/his-strategy/decision.md); returns the line to tell him."""
    from machine.intake.state import choice
    path = folder() / DECISION
    j = (files("referee") or [None])[-1]
    if j is None:
        raise ResearchError("no referee answer yet: python -m machine.strategy judge")
    a = j["result"]["answer"]
    failed = [c["name"] for c in a["checks"] if not c["passed"] and c["name"] != "not already seen"]
    head, means, _ = choice(a["word"], True, rest_passed=not failed)
    if head == "NEXT IDEA":                      # the whole is not an idea in the queue: it is kept, and waits
        head = "KEPT ON FILE"
        means = ("it lost with his hedge too, in the years it was chosen on and in the years it was judged on. "
                 "Nothing more runs for it on these years; only new prices could judge it.")
    s, b = j["result"]["summary"], budget()
    line = (f"His whole strategy (L19d) came out {a['word'].upper()}. The machine's choice: {head} - {means} "
            "You don't need to do anything. If you want something else, just say it.")
    if path.is_file():
        say("decision.md was written already - kept as it is")
        return line
    text = "\n".join([
        "# His whole strategy - the decision after the report (L19d)", "",
        f"**Date:** {_now()[:16].replace('T', ' ')} · **Report:** `plan/work/L19d/README.md` (answer "
        f"{a['word'].upper()}) · decided by the machine, by itself", "",
        "## Who decides", "",
        "His words, 1 Oct 2026, when a session asked what to do after idea #2's report (LEARNED #162):", "",
        "> I don't know why u asking these questions", "",
        "So after a report the machine chooses by the plan's default (machine.intake.state.choice - the same rule as "
        "for an idea) and only tells him. He can change it any time by saying so.", "",
        "## The choice", "", f"**{head}** - {means}", "",
        *([f"Not a forward test: the old project already used these years for it, AND it failed other checks too "
           f"({', '.join(failed)}). \"Needs fresh years\" is its answer at best (README section 7) - a forward test is "
           "for an idea that passed everything else (Claude's reading)."] if failed else []), "",
        "## What it means for the ideas parked \"for his hedge\"", "",
        "Idea #2 was parked until his hedge ran through the research loop. The answer here: his hedge loses LESS than "
        "a plain stop only because it trades far less (one trade at a time, open for weeks), and once a trade is "
        "hedged his entry does not add money. So idea #2 stays parked, kept on file. Its own trades with his hedge "
        "are a small separate run (plan/work/L19d-notes.md, next).", "",
        "## The numbers it rests on (the referee's, from its files - nothing new was run)", "",
        f"- Answer: {a['word'].upper()} - {a['why']}",
        f"- Judged on 2021-2023 with his hedge: {s['trades']} trades, {costs.inr(s['net_inr_total'])} after cost; "
        f"worst year {s['worst_year']['year']} {costs.inr(s['worst_year']['net_inr'])}",
        f"- Versions tried: {b['used']} of {b['budget']} (the family {STRATEGY}) · the referee's file: "
        f"`{j['trial_id']}`",
        "- Rupees at 0.01 lot a leg, ₹40 per position all-in (his #54, #115).", ""])
    _write(path, text)
    log(f"the machine decided: {head}", f"decision.md - {means}")
    say(f"decided: {head}")
    return line


def show() -> str:
    plan = load_plan()
    parts = [plan_text(plan) if plan else "his whole strategy: no research plan yet"]
    if log_path().is_file():
        parts.append(log_path().read_text(encoding="utf-8"))
    return "\n\n".join(parts)
