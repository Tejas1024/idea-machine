"""The research agent (L14.2): ONE of his ideas researched by itself - the plan first, then code runs exactly the plan.

    .venv\\Scripts\\python -m machine.agent plan idea-002     Claude writes the plan; code checks it and SAVES it
    .venv\\Scripts\\python -m machine.agent run idea-002      the plan (made first if missing) through the research tools,
                                                             the referee, the report, the final message
    .venv\\Scripts\\python -m machine.agent show idea-002     the plan and the log
    .venv\\Scripts\\python -m machine.agent noise             the noise exam (machine/agent/noise.py)

Who does what (the AI writes rules; code judges - CLAUDE.md):
  Claude    (his Claude Code login, no API key: machine/talker/llm.agent_call) reads the idea with the PLANNING tools
            only - get_idea, list_blocks, search_docs (`python -m machine.mcp --plan`): no tool that proposes, runs,
            draws or shows a trial file exists for it, and no built-in tool (`--tools ""`). It writes the PLAN: the
            hypothesis; at most RESEARCH_MAX_VERSIONS versions over dimension 1 (the timeframe) and dimension 10 (the
            idea's own numbers - level, stop, target, wait - and a reading of every VERSION part); at most 8 slices
            chosen in advance; the stop rule in its words. A meaning his words leave open -> "unclear", never a guess.
  code      checks the plan with the SAME rules the research tools use (versions.check_version - nothing is counted
            while checking); a refused plan goes back to Claude ONCE with the reasons (RESEARCH_FIXUPS). Then the plan
            is SAVED - research-plan.json with its sha256 + research-log.md - before any version runs. Then code runs
            EXACTLY the plan through the research tools (machine/mcp/tools.py, every guard of L14.1 inside):
            choose_slices, then propose_version + run_version for each planned version in the plan's order, once
            each, then write_report. A version that is not in the plan never runs.
  the stop  code's rule, the same for every plan (STOP_RULE): every planned version ran; or the family's 20 are used;
            or RESEARCH_STUCK_IN_A_ROW planned versions in a row could not run (a session looks). Never a stop or a
            new version because of money: a stop that looks at results is choosing by money (lesson 18).
  referee   write_report (L10 through L14.1) chooses ONE version on 2016-2020 by the highest Deflated Sharpe (the
            chance its edge is real, allowing for every version tried), opens 2021-2023 once per family for it, judges.
  unclear   the plan's "unclear" -> the #129 search first (search_docs): FOUND in his documents -> a session puts that
            meaning into the form (no question); NOT FOUND -> a question in the queue for the weekly batch
            (plan/work/question-queue/<idea>-research-<n>.md; a session draws its pictures first - L9b). Nothing runs.

Files (plan/work/ideas/<idea>/; on his PC they reach git through the nightly bot pull request, L9c):
  research-plan.json   the plan as saved, with its sha256 - never rewritten (a new meaning: the old plan is kept as
                       research-plan-replaced-<time>.json, only if none of its versions was proposed)
  research-log.md      what happened, oldest first (append only)
  research-final.md    the final message: the report + every version of the family from plan/work/trials/
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from machine import config
from machine.ideas import store
from machine.ideas.form import LOCKED_FROM, STATES, Slice, open_parts
from machine.mcp import tools as T
from machine.mcp import versions as V
from machine.mcp.versions import Refused
from machine.referee import trials

PLAN, LOG, FINAL = "research-plan.json", "research-log.md", "research-final.md"
WHO = "the research agent (L14.2)"
TELEGRAM_MAX = 4096

STOP_RULE = (
    "the planned versions run in the plan's order, each once; a version that is not in the plan never runs",
    "it stops when every planned version ran",
    f"it stops when the family's {config.VERSION_BUDGET} versions are used",
    f"it stops when {config.RESEARCH_STUCK_IN_A_ROW} planned versions in a row cannot run (no setup on 2016-2020, or "
    "refused) - then a Claude session looks, because something is broken",
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
    """One planning call's answer."""
    raw: dict | None                  # Claude's plan in the schema's shape (None: no usable answer)
    meta: dict = field(default_factory=dict)   # who made it: model, turns, seconds, the tools it called
    error: str | None = None


class Planner(Protocol):
    def __call__(self, idea: str, *, refused: list[str] | None = None, previous: dict | None = None) -> Planned: ...


@dataclass
class Outcome:
    idea: str
    status: str            # reported · plan saved · waits · question · needs a session · no answer
    said: str
    final: str | None = None
    answer: str | None = None
    plan_sha: str | None = None


# ================================================================================================= files

def plan_path(idea: str) -> Path:
    return store.folder(idea) / PLAN


def log_path(idea: str) -> Path:
    return store.folder(idea) / LOG


def final_path(idea: str) -> Path:
    return store.folder(idea) / FINAL


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, text: str) -> None:
    """A whole file through a temp file (never open(path, "w") on a tracked file - CLAUDE.md rule 7)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def log(idea: str, heading: str, body: str = "") -> None:
    """One step into research-log.md (append only; made with its header the first time)."""
    path = log_path(idea)
    first = not path.is_file()
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        if first:
            fh.write(f"# Research log - {idea} (the research agent, L14.2)\n\n> Written by code, oldest first, never "
                     "rewritten. The plan: research-plan.json (saved BEFORE any version ran). The AI writes rules; "
                     "code judges.\n\n")
        fh.write(f"## {_now()[:16].replace('T', ' ')} · {heading}\n\n" + (body.rstrip() + "\n\n" if body else ""))


def sha_of(plan: dict) -> str:
    body = {k: v for k, v in plan.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def load_plan(idea: str) -> dict | None:
    """The saved plan; refused when its file was changed after it was saved (its sha256 no longer matches)."""
    path = plan_path(idea)
    if not path.is_file():
        return None
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("sha256") != sha_of(plan):
        raise ResearchError(f"{V.rel(path)} was changed after it was saved (its sha256 does not match) - a plan is "
                            "never edited; a session looks")
    return plan


def _save_plan(idea: str, plan: dict) -> dict:
    if plan_path(idea).exists():
        raise ResearchError(f"{idea} has a saved plan already - it is never rewritten")
    plan["sha256"] = sha_of(plan)
    _write(plan_path(idea), json.dumps(plan, indent=1, ensure_ascii=False, default=str) + "\n")
    return plan


def _retire(idea: str, plan: dict, why: str) -> None:
    """A plan made for a meaning that changed before any of its versions was proposed: kept under another name."""
    proposed = {r["version_id"] for r in V.records(idea)}
    if proposed & {v["version_id"] for v in plan.get("versions", [])}:
        raise ResearchError(f"{idea}'s meaning changed after versions of its plan were proposed ({why}) - lesson 16: "
                            "a new meaning after money is a new idea; a session looks")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    plan_path(idea).replace(store.folder(idea) / f"research-plan-replaced-{stamp}.json")
    log(idea, "the plan was put aside (never run)", f"{why}. Kept as research-plan-replaced-{stamp}.json.")


# ================================================================================================= the gate

def gate(idea: str) -> tuple[bool, str]:
    """May this idea be researched now? Only a LOCKED meaning (lesson 16) with no ASK part open (R13)."""
    form = V.idea_form(idea)
    state = store.read_state(idea)["state"]
    if STATES.index(state) < STATES.index(LOCKED_FROM):
        return False, (f"{idea} is {state}: its meaning is not locked yet - the intake asks his questions first "
                       "(L9b); the research agent never guesses a meaning (R13)")
    asks = [u.id for u in open_parts(form) if u.kind == "ask"]
    if asks:
        return False, f"{idea}: {', '.join(asks)} wait for his answer (R13: never guess a meaning)"
    return True, f"{idea} is {state}"


# ================================================================================================= the plan's shape

_TEXT = {"type": "string"}
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hypothesis": _TEXT,
        "versions": {"type": "array", "items": {"type": "object", "properties": {
            "name": _TEXT, "why": _TEXT,
            "readings": {"type": "array", "items": {"type": "object", "properties": {
                "part": _TEXT, "reading": _TEXT}, "required": ["part", "reading"]}},
            "changes": {"type": "array", "items": {"type": "object", "properties": {
                "path": _TEXT, "value": {"type": ["number", "string"]}}, "required": ["path", "value"]}}},
            "required": ["name", "why", "readings", "changes"]}},
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


def _value(v: Any) -> Any:
    """A change's value as Claude typed it -> the number it means ("8" -> 8, 8.0 -> 8); words stay words."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?(\d+\.\d*|\.\d+)", s):
            return _value(float(s))
        return s
    return v


def _real_changes(form, readings: dict[str, str], changes: dict[str, Any]) -> dict[str, Any]:
    """The changes that change something: one that sets the value the readings already give is left out (the first
    real plan, 1 Oct, chose reading B = 0.04 AND the change 0.04 - the same rule, a confusing record)."""
    from machine.ideas.form import FormError, with_readings
    try:
        d = with_readings(form, readings)[0].model_dump(mode="json")
    except FormError:
        return changes                                     # check_version says why, in its words
    return {p: x for p, x in changes.items() if V._get(d, p) != x}


def _words_problems(text: str, what: str) -> list[str]:
    from machine.report.build import check_words
    return [f"{what}: {p}" for p in check_words(text)]


def budget(idea: str) -> dict:
    """What the plan may spend: the family's versions left, the plan's own cap, the slices left."""
    form = store.load_form(idea)
    files = trials.load()
    parent, _ = V.parent_of(idea, form, files)
    fam = V.family(idea, parent, files)
    slices = store.with_chosen_slices(form).slices
    return {"family": fam["root"], "used": fam["used"], "left": fam["left"], "budget": fam["budget"],
            "cap": max(0, min(config.RESEARCH_MAX_VERSIONS, fam["left"])),
            "slices_chosen": [s.name for s in slices], "slices_left": config.SLICE_BUDGET - len(slices)}


def check_plan(idea: str, raw: dict) -> tuple[dict, list[str]]:
    """Claude's plan -> the plan code will run, and every problem (empty: it may be saved). The versions are checked
    with the research tools' own rules (versions.check_version: a refused change, a fixed rule, a slice dressed as a
    version, what the idea IS) - nothing is proposed or counted here."""
    from machine.learner import conditions
    problems: list[str] = []
    form = V.locked(idea)
    b = budget(idea)
    hyp = str(raw.get("hypothesis") or "").strip()
    stop = str(raw.get("stop_rule") or "").strip()
    unclear = [u for u in raw.get("unclear") or [] if str(u.get("what") or "").strip()]
    plan: dict[str, Any] = {"idea": idea, "hypothesis": hyp, "versions": [], "slices": [],
                            "stop_rule": {"in_words": stop, "code": list(STOP_RULE)}, "unclear": unclear,
                            "family": b["family"], "budget": b}
    if not hyp:
        problems.append("the hypothesis is empty: one sentence - what must be true on 2016-2020 if his idea works")
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
        problems.append("no version: at least the idea as written")
    if len(raw_versions) > b["cap"]:
        problems.append(f"{len(raw_versions)} versions: at most {b['cap']} (the {b['family']} family has {b['left']} "
                        f"of {b['budget']} left; a plan uses at most {config.RESEARCH_MAX_VERSIONS} - the rest stay "
                        "for his own change after the report)")
    files, recs = trials.load(), V.all_records()
    tried = set(V.family(idea, V.parent_of(idea, form, files)[0], files, recs)["tried"])
    first: dict[str, int] = {}
    for k, v in enumerate(raw_versions, 1):
        name = str(v.get("name") or "").strip() or f"version {k}"
        why = str(v.get("why") or "").strip()
        readings = {str(r.get("part")): str(r.get("reading")) for r in v.get("readings") or []}
        changes = _real_changes(form, readings, {str(c.get("path")): _value(c.get("value"))
                                                 for c in v.get("changes") or []})
        if not why:
            problems.append(f"version {k} ({name}): no reason (why)")
        problems += _words_problems(f"{name}. {why}", f"version {k}")
        try:
            c = V.check_version(idea, readings=readings, changes=changes, files=files, recs=recs)
        except Refused as err:
            problems.append(f"version {k} ({name}): REFUSED - {err}")
            continue
        vid = c["version_id"]
        if vid in first:
            problems.append(f"version {k} ({name}) is the same rule as version {first[vid]} ({vid}) - one rule is "
                            "one try")
            continue
        first[vid] = k
        plan["versions"].append({"k": k, "name": name, "why": why, "readings": c["readings"],
                                 "changes": c["changes"], "version_id": vid, "about": c["about"],
                                 "dimensions": c["dimensions"], "timeframe": c["timeframe"],
                                 "new_try": not c["proposed_before"] and (idea, vid) not in tried})
    new_tries = sum(v["new_try"] for v in plan["versions"])
    if new_tries > b["left"]:
        problems.append(f"{new_tries} new tries: the family has {b['left']} left")
    have = set(b["slices_chosen"])
    raw_slices = raw.get("slices") or []
    new = [s for s in raw_slices if str(s.get("name")) not in have]
    if new and store.money_seen(idea):
        problems.append("money was already seen for this idea: no new slice (section D - chosen BEFORE running)")
    if len(new) > b["slices_left"]:
        problems.append(f"{len(new)} new slices: at most {b['slices_left']} more ({config.SLICE_BUDGET} per idea, "
                        f"chosen already: {', '.join(sorted(have)) or 'none'})")
    names: set[str] = set()
    for s in new:
        n = str(s.get("name"))
        try:
            Slice.model_validate(s)
        except Exception as err:                           # noqa: BLE001 - a pydantic error, said in words
            problems.append(f"slice {n!r}: {err}")
            continue
        cond = conditions.BY_NAME.get(n)
        if cond is None:
            problems.append(f"slice {n!r}: not a condition the learner measures (list_blocks' slice_conditions)")
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
    """The planning server for Claude Code: machine.mcp --plan (get_idea, list_blocks, search_docs only). The same
    Python as this program (pythonw in the intake: no window on his PC). Claude Code's MCP config has no working-
    folder field: PYTHONPATH tells the server where `machine` is (seen in L14.1)."""
    return {"command": sys.executable, "args": ["-u", "-m", "machine.mcp", "--plan"],
            "env": {"PYTHONPATH": str(config.ROOT), "MLBOT_MCP_DIR": str(config.MCP_DIR),
                    "PYTHONIOENCODING": "utf-8"}}


def planner_system() -> str:
    banned = "leftover, arm, verdict, veto, causal, placebo, mirror, baseline, R, basket"
    stop = "\n".join(f"  - {x}" for x in STOP_RULE)
    return f"""You are the research agent of Tejas's idea machine (L14.2). Tejas is an expert gold (XAUUSD) trader. He
says a trading idea in plain words; the machine has already turned it into a FORM (its steps, entry, exit, targets)
and asked him what was unclear. Your job: write the RESEARCH PLAN for ONE idea. You only plan. Code checks your plan,
SAVES it, and then runs exactly it - nothing else. Nothing runs and you see no money while you plan. The AI writes
rules; code judges.

Do this, with your tools:
1. get_idea(idea) - his words, the form, the unclear parts (kind "version": you choose a reading of each, per version;
   kind "ask": his answer - never yours), the slices already chosen, the version budget.
2. list_blocks() - "settings_a_version_may_change" and "version_rules" (what a version may change), "slice_conditions"
   (a slice's name and its dimension), "his_fixed_rules" (never changed).
3. search_docs(query) - his rule #129: before you decide what a word of his means, or which number fits his words,
   search his documents. Search every word of his that decides a reading or a number.
4. Answer with the plan, in the answer's fixed shape.

The plan:
- hypothesis: ONE sentence in easy English - what must be true on 2016-2020 gold if his idea works. Example: "Selling
  at the first gap after the London high is taken makes money after the Rs 40 cost of every trade, in most years."
  No promise of profit.
- versions: at most the number the message gives. Each version is a different rule, and ONE TRY: every try makes it
  harder for the referee to call a real edge real, so fewer is fine.
  - readings: one reading for EVERY "version" part of get_idea's "unclear" (part id + reading id).
  - changes: ONLY dimension 1 (the timeframe: "timeframe", "steps.<id>.tf") or dimension 10 (the idea's own numbers:
    "steps.<id>.max_wait_bars", "entry.max_wait_bars", "steps.<id>.params.<name>", "do_not_enter.<n>.params.<name>",
    "exit.beyond_usd", "targets.<n>.ratio", "settings.<name>" of a number the machine chose). Never his fixed rules,
    never a direction, a filter or a session (those are slices), never what the idea IS (its blocks, its entry).
  - Version 1 = the idea as written: the reading his words fit best (usually "A"), no change.
  - Then the readings his words leave open, then a few numbers near what his words or documents say. Order them most
    important first.
  - Every number needs a reason from his words or his documents - say the place. Never a number "to see if it makes
    money": no money is known, and that is how luck is mistaken for an edge.
- slices: at most the number the message gives, chosen NOW, before any money - only these can later become "works
  best in / fails in". Conditions from slice_conditions, each with ITS dimension. Choose those his words or documents
  say matter for this idea (the session, news, the trend, the place ...). Fewer is fine.
- stop_rule: one or two sentences in easy English: when this research stops. Code's rule is fixed - say it for this
  idea in your words:
{stop}
- unclear: ONLY when a word of his changes which trades are taken, the form has no reading for it, AND search_docs
  did not find his meaning. Then: what is unclear, his words, 2 to 4 readings, why they trade differently, what the
  search found. Never guess a meaning. When "unclear" is not empty the research stops there and the question goes to
  him; give no versions then. Usually "unclear" is empty: the form already holds his answers.

Easy English (his rule #113) in every sentence you write: short words, short sentences. Never these code words:
{banned}. No "p-value", "n =", "t =". Say money in rupees. Do not use the words Sharpe, bootstrap, walk-forward,
percentile, PBO, LightGBM or SHAP."""


def planner_user(idea: str, refused: list[str] | None = None, previous: dict | None = None) -> str:
    b = budget(idea)
    text = (f"Write the research plan for {idea}.\n"
            f"- versions: at most {b['cap']} (the {b['family']} family has {b['left']} of {b['budget']} tries left; "
            f"a plan uses at most {config.RESEARCH_MAX_VERSIONS} - the rest stay for his own change after the report)\n"
            f"- slices: at most {b['slices_left']} more (chosen already: {', '.join(b['slices_chosen']) or 'none'})")
    if refused:
        text += ("\n\nCode REFUSED your plan. The reasons:\n" + "\n".join(f"- {p}" for p in refused) +
                 "\n\nYour plan was:\n" + json.dumps(previous, ensure_ascii=False) +
                 "\n\nWrite the whole plan again with these fixed. Change nothing else.")
    return text


def claude_planner(idea: str, *, refused: list[str] | None = None, previous: dict | None = None) -> Planned:
    """One planning call: Claude Code with the planning tools only (machine/talker/llm.agent_call)."""
    from machine.talker import llm
    a = llm.agent_call(system=planner_system(), user=planner_user(idea, refused, previous), schema=PLAN_SCHEMA,
                       servers={"research": planning_server()},
                       allowed=tuple(f"mcp__research__{t}" for t in T.PLAN_TOOLS), model=config.RESEARCH_MODEL,
                       effort=config.RESEARCH_EFFORT, timeout=config.RESEARCH_PLAN_SECONDS)
    meta = {"model": a.model, "via": "Claude Code - his login (no API key, no bill per call)", "turns": a.turns,
            "seconds": a.seconds, "api_price_usd": a.api_price_usd,
            "tool_calls": [{"tool": c["tool"], "input": c["input"], "error": c["error"],
                            "said": (c["said"] or "")[:200]} for c in a.tool_calls]}
    return Planned(a.output, meta, a.error)


def saved_planner(plan: dict, versions: int | None = None) -> Planner:
    """A planner that answers with a saved plan's own words (its "raw"), cut to the first `versions` versions - the
    noise exam re-runs a real plan on made-up prices without asking Claude again."""
    def planner(idea: str, *, refused: list[str] | None = None, previous: dict | None = None) -> Planned:
        raw = json.loads(json.dumps(plan["raw"]))
        if versions is not None:
            raw["versions"] = raw["versions"][:versions]
        return Planned(raw, {"model": f"the saved plan {plan['sha256'][:12]} of {plan['idea']}", "turns": 0,
                             "seconds": 0, "tool_calls": []})
    return planner


# ================================================================================================= make the plan

def meaning_of(form) -> str:
    """The form's whole meaning: form.json without its status (every state change rewrites that). Unlike the rule's
    fingerprint it covers the unclear parts and their readings too: a new answer or a new reading = a new plan."""
    d = json.loads(form.to_json())
    d.pop("status", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def make_plan(idea: str, *, planner: Planner | None = None, say: Callable = print) -> dict:
    """The saved plan for the idea's current meaning - made now if there is none (Claude plans, code checks; ONE
    fix-up call with the reasons; then SAVED before anything runs)."""
    ok, why = gate(idea)
    if not ok:
        raise ResearchError(why)
    form = store.load_form(idea)
    now = meaning_of(form)
    old = load_plan(idea)
    if old is not None:
        if old["form_meaning"] == now:
            return old
        _retire(idea, old, f"the idea's form changed since the plan was made (its meaning {old['form_meaning'][:12]} "
                           f"-> {now[:12]})")
    seen = store.money_seen(idea)
    if seen:
        raise ResearchError(f"{idea}: money was already seen ({', '.join(seen[:2])}) - a research plan comes BEFORE "
                            "any money (lesson 18); a session looks")
    planner = planner or claude_planner
    calls = []
    got = planner(idea)
    calls.append(got.meta)
    if got.raw is None:
        raise NoAnswer(f"the planning call got no answer: {got.error}")
    plan, problems = check_plan(idea, got.raw)
    fixups = 0
    while problems and fixups < config.RESEARCH_FIXUPS:
        fixups += 1
        say(f"{idea}: code refused the plan ({len(problems)} reasons) - one more call with the reasons")
        log(idea, f"code refused the plan ({len(problems)} reasons) - Claude gets them once",
            "\n".join(f"- {p}" for p in problems))
        prev = got.raw
        got = planner(idea, refused=problems, previous=prev)
        calls.append(got.meta)
        if got.raw is None:
            raise NoAnswer(f"the fix-up call got no answer: {got.error}")
        plan, problems = check_plan(idea, got.raw)
    if problems:
        log(idea, "code refused the plan again - nothing runs", "\n".join(f"- {p}" for p in problems))
        raise ResearchError(f"code refused {idea}'s plan {fixups + 1} times: " + "; ".join(problems)[:600])
    state = store.read_state(idea)["state"]
    plan.update(made_at=_now(), state_when_made=state, form_meaning=now, money_seen_before=[],
                made_by={"agent": WHO, "calls": calls, "fixups": fixups}, raw=got.raw)
    plan = _save_plan(idea, plan)
    log(idea, "the plan was saved (before any version ran)", plan_text(plan))
    say(f"{idea}: plan {plan['sha256'][:12]} saved - {len(plan['versions'])} versions, {len(plan['slices'])} slices"
        + (f", {len(plan['unclear'])} unclear" if plan["unclear"] else ""))
    return plan


def replan(idea: str, why: str, *, planner: Planner | None = None, say: Callable = print) -> dict:
    """A session decided the plan must be made again (its unclear meaning was answered without a change of the form):
    the old plan is put aside - only if none of its versions was proposed - and a new one is made."""
    if not str(why).strip():
        raise ResearchError("say why the plan is made again (it goes into the log)")
    old = load_plan(idea)
    if old is not None:
        _retire(idea, old, f"a session: {why}")
    return make_plan(idea, planner=planner, say=say)


def plan_text(plan: dict) -> str:
    """The plan in easy English (the log, the session that reads it)."""
    out = [f"**What must be true if the idea works:** {plan['hypothesis']}", ""]
    if plan["unclear"]:
        out.append("**Unclear - asked, never guessed (no version runs until it is answered):**")
        for u in plan["unclear"]:
            out.append(f"- {u['what']} - his words: \"{u['his_words']}\" - readings: "
                       + " / ".join(u["readings"]) + f" - why they trade differently: {u['why_trades_differ']}")
        out.append("")
    b = plan["budget"]
    if plan["versions"]:
        out += [f"**Versions:** {len(plan['versions'])} (each one is a try; the {plan['family']} family had used "
                f"{b['used']} of {b['budget']} before this plan)", "",
                "| # | version | what it is | why |", "|---|---|---|---|"]
        for v in plan["versions"]:
            out.append(f"| {v['k']} | {v['name']} (`{v['version_id']}`) | {v['about']} | {v['why']} |")
        out.append("")
    if plan["slices"]:
        out.append(f"**Slices chosen before any money ({len(plan['slices'])}):**")
        out += [f"- {s['name']} ({s['dimension']}): {s['how']} - {s['why']}" for s in plan["slices"]]
        out.append("")
    out.append(f"**When it stops:** {plan['stop_rule']['in_words']}")
    out.append("Code's rule: " + "; ".join(plan["stop_rule"]["code"]) + ".")
    calls = (plan.get("made_by") or {}).get("calls") or []
    for c in calls:
        searched = [f"\"{(t.get('input') or {}).get('query')}\"" + (" (refused)" if t.get("error") else "")
                    for t in c.get("tool_calls", []) if t.get("tool") == "search_docs"]
        out.append(f"\n**Made by:** {c.get('model')} through {c.get('via', 'Claude Code - his login')}, "
                   f"{c.get('turns')} turns, {c.get('seconds')} s; tools: "
                   + (", ".join(t["tool"] for t in c.get("tool_calls", [])) or "none")
                   + (f"; it searched his documents for {', '.join(searched)}" if searched else ""))
    if plan.get("sha256"):
        out.append(f"\n**The plan's fingerprint (sha256):** `{plan['sha256']}` - saved {plan.get('made_at', '')}")
    return "\n".join(out)


# ================================================================================================= unclear -> asked

def queue_unclear(idea: str, plan: dict, *, tell: bool = False) -> list[str]:
    """Each unclear meaning of the plan: the #129 search first (search_docs). Found in his documents -> a session puts
    it into the form (no question); not found -> a question in the queue for the weekly batch (a session draws its
    pictures first - the batch lists it and does not send it before). Then the idea waits for a session."""
    from machine.intake import state as intake
    from machine.questions import queue as Q
    made, found = [], []
    folder = Q.queue_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for k, u in enumerate(plan["unclear"], 1):
        query = f"{u['his_words']} {u['what']}".strip()[:500]
        try:
            hit = T.search_docs(query)
        except Refused as err:
            hit = {"found": False, "answer": f"the search refused: {err}", "hits": []}
        tag = f"{idea}-research-{plan['sha256'][:8]}-{k}"
        if hit["found"]:
            found.append(f"{u['what']}: his documents answer it - {hit['answer'][:300]}")
            continue
        path = folder / f"{tag}.md"
        if not path.exists():
            _write(path, _question_md(idea, plan, k, u, hit))
        made.append(V.rel(path))
    lines = ([f"his documents answer it, a session puts it into the form: {x}" for x in found]
             + [f"a question queued for the weekly batch (a session draws its pictures first): {p}" for p in made])
    log(idea, "unclear meaning - nothing runs (R13: never a guess)", "\n".join(f"- {x}" for x in lines))
    intake.needs(idea, f"the research plan found {len(plan['unclear'])} unclear meaning(s): "
                       + "; ".join(u["what"] for u in plan["unclear"])[:300]
                       + f" - see plan/work/ideas/{idea}/{LOG}", tell=tell)
    return made


def _question_md(idea: str, plan: dict, k: int, u: dict, hit: dict) -> str:
    readings = "\n".join(f"| {chr(64 + i)} | {r} |" for i, r in enumerate(u["readings"], 1))
    closest = (hit.get("hits") or [{}])[0]
    near = f"{closest.get('where', '?')}: \"{str(closest.get('quote', ''))[:200]}\"" if closest else "nothing"
    return f"""# {idea} - research question {k}: {u['what']}

| field | value |
|---|---|
| status | waiting - made by the research agent (L14.2) on {_now()[:10]}. No picture yet: the batch lists it and does not send it until a session draws them |
| his words | "{u['his_words']}" |
| what waits for it | {idea}'s research plan `{plan['sha256'][:12]}` (plan/work/ideas/{idea}/{PLAN}) - no version runs before his answer |
| idea | {idea} |
| priority (L9b) | (2) the idea currently running |

## Why ask (the #129 search first)

I searched his documents for "{u['his_words']}" ({u['what']}) and found nothing - the closest line is {near}.
What the planning agent searched: {u['searched']}

## The readings

| his answer | the reading |
|---|---|
{readings}

## Why they trade differently

{u['why_trades_differ']}

## How to ask it (his picture rules)

ONE 5-minute chart per reading of the SAME moment, India time, every named thing drawn, no money; the question on
the picture. A session draws them (machine.questions), then this file's status says "waiting" with its pictures.
"""


# ================================================================================================= run the plan

def run(idea: str, *, planner: Planner | None = None, bench: T.Workbench | None = None, send: bool = False,
        tell: bool = False, say: Callable = print) -> Outcome:
    """Research ONE idea: the plan (made and saved first), then exactly the plan through the research tools, the
    referee's report, the final message. send=True: the report and the list of every version go to his Telegram
    (the outbox). tell=True: a step that needs a session tells him in one Telegram line (the intake)."""
    who = T.SESSION.target, T.SESSION.who
    T.start(idea, f"{WHO}, run by code")
    try:
        return _run(idea, planner=planner, bench=bench, send=send, tell=tell, say=say)
    finally:
        T.SESSION.target, T.SESSION.who = who


def _run(idea: str, *, planner, bench, send: bool, tell: bool, say: Callable) -> Outcome:
    if final_path(idea).is_file() and store.read_state(idea)["state"] in ("REPORTED", "DECIDED"):
        return Outcome(idea, "reported", "researched already", final_path(idea).read_text(encoding="utf-8"))
    ok, why = gate(idea)
    if not ok:
        return Outcome(idea, "waits", why)
    try:
        plan = make_plan(idea, planner=planner, say=say)
    except NoAnswer as err:
        return Outcome(idea, "no answer", str(err))
    except ResearchError as err:
        return Outcome(idea, "needs a session", str(err))
    sha = plan["sha256"]
    T.SESSION.who = f"{WHO}: the plan {sha[:12]}, run by code"   # into every version record and check-opened.json
    if plan["unclear"]:
        queue_unclear(idea, plan, tell=tell)
        return Outcome(idea, "question", f"{len(plan['unclear'])} unclear meaning(s) queued - nothing runs", plan_sha=sha)
    allowed, why = T.money_allowed(idea)
    if not allowed:
        return Outcome(idea, "plan saved", f"the plan {sha[:12]} is saved; it runs after: {why}", plan_sha=sha)
    have = {s.name for s in store.with_chosen_slices(store.load_form(idea)).slices}
    new = [s for s in plan["slices"] if s["name"] not in have]
    if new:
        T.choose_slices(idea, new, why=f"the research plan {sha[:12]}: chosen in advance, before any money")
        log(idea, f"{len(new)} slices chosen (before any money)", ", ".join(s["name"] for s in new))
    ran, stuck, stopped = [], 0, None
    n = len(plan["versions"])
    for v in plan["versions"]:
        k = v["k"]
        try:
            c = V.check_version(idea, readings=v["readings"], changes=v["changes"])
            if c["version_id"] != v["version_id"]:
                raise Refused(f"it now compiles to {c['version_id']}, not the plan's {v['version_id']} - the form or "
                              "the code changed since the plan was saved")
            p = T.propose_version(idea, f"research plan {sha[:12]}, version {k} of {n} ({v['name']}): {v['why']}",
                                  readings=v["readings"], changes=v["changes"])
            r = T.run_version(idea, p["version"], how="tune", bench=bench)
        except Refused as err:
            stuck += 1
            log(idea, f"version {k} of {n} could not run", f"{v['name']} (`{v['version_id']}`): {err}")
            say(f"{idea}: version {k}/{n} could not run - {str(err)[:200]}")
            if "the budget is" in str(err):
                stopped = f"the family's {config.VERSION_BUDGET} versions are used"
                break
            if stuck >= config.RESEARCH_STUCK_IN_A_ROW:
                stopped = f"{stuck} planned versions in a row could not run - a session looks"
                break
            continue
        stuck = 0
        ran.append(v["version_id"])
        log(idea, f"version {k} of {n} ran on 2016-2020", f"{v['name']} (`{v['version_id']}`): "
            + _numbers_line(r["numbers"]) + (" - already run, not run again" if r.get("already_run") else ""))
        say(f"{idea}: version {k}/{n} {v['version_id']} - {_numbers_line(r['numbers'])}")
    stopped = stopped or "every planned version ran"
    log(idea, f"the versions stopped: {stopped}", f"{len(ran)} of {n} planned versions ran.")
    if not ran:
        return Outcome(idea, "needs a session", f"no planned version could run ({stopped})", plan_sha=sha)
    if stopped.endswith("a session looks"):
        return Outcome(idea, "needs a session", stopped, plan_sha=sha)
    rep = T.write_report(idea, send=send, bench=bench)
    judged = rep["judged"]
    answer = judged.get("answer") or _judged_answer(idea)
    log(idea, f"the referee: {answer or judged.get('said', '?')}",
        f"chosen by the referee on 2016-2020: `{rep['chosen']}` of {rep['versions_tuned']} versions; judged "
        f"{judged.get('judged_on') or judged.get('mode')}. The report: {rep['page']}.")
    text = final_message(idea, plan, rep, stopped)
    _write(final_path(idea), text + "\n")
    log(idea, "the final message", f"{FINAL} - the report + every version from plan/work/trials/")
    if send:
        _send_versions(idea, plan, rep)
    return Outcome(idea, "reported", f"{answer}: the referee chose {rep['chosen']}", text, answer, sha)


def _judged_answer(idea: str) -> str | None:
    got = [f for f in trials.load() if f.get("kind") == "referee" and f.get("idea_id") == idea]
    return ((got[-1].get("result") or {}).get("answer") or {}).get("word") if got else None


# ================================================================================================= the final message

def _inr(x: float | None) -> str:
    if x is None:
        return "no trades"
    return f"-₹{-x:,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def _numbers_line(n: dict) -> str:
    w = n.get("worst_year") or {}
    worst = f", worst year {w.get('year')} {_inr(w.get('net_inr'))}" if w else ""
    return f"{n.get('trades') or 0} trades, {_inr(n.get('net_inr_per_trade'))} a trade{worst}"


def every_version(idea: str) -> list[dict]:
    """Every version of the idea's family with a tester trial file (plan/work/trials/): its numbers on 2016-2020 (the
    years every version may show - never the check years here), and the referee's choice."""
    form = store.load_form(idea)
    files = trials.load()
    parent, _ = V.parent_of(idea, form, files)
    fam = V.family(idea, parent, files)
    members = set(fam["ideas"])
    last: dict[tuple[str, str], dict] = {}
    for f in files:
        if f.get("kind") == "tester" and f.get("idea_id") in members:
            last[(f["idea_id"], f["version_id"])] = f
    chosen = {(f["idea_id"], (f.get("result") or {}).get("chosen")) for f in files
              if f.get("kind") == "referee-tune" and f.get("idea_id") in members}
    about = {r["version_id"]: r["about"] for i in members if i.startswith("idea-") for r in V.records(i)}
    out = []
    for (i, vid), f in last.items():
        r = f.get("result") or {}
        out.append({"idea": i, "version": vid, "about": about.get(vid, ""), "trades": r.get("trades"),
                    "per_trade": r.get("net_inr_per_trade"), "total": r.get("net_inr_total"),
                    "worst_year": r.get("worst_year"), "chosen": (i, vid) in chosen, "trial": f.get("trial_id")})
    return out


def versions_text(idea: str, plan: dict, *, short: bool = False) -> str:
    """Every version tried, in the plan's order first (2016-2020, rupees at 0.01 lot, Rs 40 per position all-in)."""
    rows = every_version(idea)
    order = {v["version_id"]: v for v in plan["versions"]}
    mine = sorted([r for r in rows if r["idea"] == idea and r["version"] in order],
                  key=lambda r: order[r["version"]]["k"])
    others = [r for r in rows if r not in mine]
    lines = [f"Every version tried for {idea} (2016-2020 only; rupees at 0.01 lot, ₹40 per position all-in; each "
             f"version is one try - {len(rows)} in the family):"]
    for r in mine:
        v = order[r["version"]]
        name = v["name"] if short else f"{v['name']} - {v['about']}"
        lines.append(f"{v['k']}. {name}: {r['trades'] or 0} trades, {_inr(r['per_trade'])} a trade, total "
                     f"{_inr(r['total'])}" + (f", worst year {r['worst_year']['year']} "
                                              f"{_inr(r['worst_year']['net_inr'])}" if r.get("worst_year") else "")
                     + (" - CHOSEN by the referee" if r["chosen"] else ""))
    for r in others:
        lines.append(f"- {r['idea']} {r['version']} (tried before this plan): {r['trades'] or 0} trades, "
                     f"{_inr(r['per_trade'])} a trade" + (" - chosen by the referee" if r["chosen"] else ""))
    planned = {v["version_id"] for v in plan["versions"]}
    missing = [v for v in plan["versions"] if v["version_id"] not in {r["version"] for r in mine}]
    for v in missing:
        lines.append(f"{v['k']}. {v['name']}: did not run (research-log.md says why)")
    if not planned:
        lines.append("(no planned version)")
    return "\n".join(lines)


def final_message(idea: str, plan: dict, rep: dict, stopped: str) -> str:
    """The research agent's final message: the referee's report + every version from plan/work/trials/."""
    head = (f"The research plan was saved {plan['made_at'][:16].replace('T', ' ')}, before any money "
            f"(fingerprint {plan['sha256'][:12]}). It said: {plan['hypothesis']}\n"
            f"It tried {len(plan['versions'])} versions; it stopped because {stopped}.")
    return "\n\n".join([rep["report_text"], head, versions_text(idea, plan)])


def _send_versions(idea: str, plan: dict, rep: dict) -> None:
    """The list of every version as ONE Telegram line after the report (the report itself is near 4,096 characters)."""
    from machine.intake import state as intake
    pick = next((f"version {v['k']}, {v['name']}" for v in plan["versions"] if v["version_id"] == rep["chosen"]),
                rep["chosen"])
    text = (f"{intake.label(idea).capitalize()} - every version the research tried. The referee chose ONE: {pick}."
            "\n\n" + versions_text(idea, plan, short=True))
    if len(text) > TELEGRAM_MAX:
        text = text[:TELEGRAM_MAX - 80].rsplit("\n", 1)[0] + f"\n... the rest: plan/work/ideas/{idea}/{FINAL}"
    intake._tell(f"{idea}-research-versions-{plan['sha256'][:8]}", text, idea=idea, why="L14.2 every version")


def show(idea: str) -> str:
    plan = load_plan(idea)
    parts = [plan_text(plan) if plan else f"{idea}: no research plan yet"]
    if log_path(idea).is_file():
        parts.append(log_path(idea).read_text(encoding="utf-8"))
    return "\n\n".join(parts)
