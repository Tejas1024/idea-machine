"""His plain words -> an idea form, with ONE strict tool call (L8, the talker).

    from machine.talker.translate import translate
    d = translate("london high taken and it give bos and fvg in 5 min then sell at fvg, target asian low")
    d.form          # machine.ideas.form.Idea (None if the answer made no good form)
    d.status        # "FORM DRAFTED" | "BLOCKED" (needs a block that is not built) | "NO FORM"
    print(d.text()) # everything: the glossary, the searches, the form, the unclear parts, the checks, the cost

What happens, in order (the AI writes rules; code judges - CLAUDE.md rule 11):
  1. The glossary (machine/talker/glossary.csv) turns his voice-typing phrases into what he meant; his words stay as
     typed in the form.
  2. Every term of his sentence (machine/talker/terms.py) is searched in his documents (machine.search, his #129
     rule). The top 5 quotes of each go to Claude.
  3. ONE call to Claude with ONE strict form, fill_idea_form (by default through his own Claude Code login, no
     bill per call - machine/talker/llm.py). What Claude sees: the form's schema; the blocks with
     their meanings (compile.describe()); his fixed rules (rules.py) and the settings (settings.py); the four boxes'
     Box 2 (his words only he can define -> ASK) and Box 3 (numbers the machine tries -> VERSION); strategy.md
     sections 5, 9, 10; the old tries (prior-tries.csv); his documented defaults; one example form (his usual setup).
     Everything but his sentence and its search results is the same for every sentence: it is cached by the API.
  4. Claude picks, for every term, the ONE quote that says what he means, or none (plan/work/L8b-notes.md: the
     search's own first citation is right 20 of 30 times, the right place is in the first 3 in 23 of 30).
  5. Code checks the answer: it must make a valid Idea; compile.check_without_bars (his fixed rules refused with his
     words, the settings, every step and name); every reading of every unclear part must make a valid form too; a
     citation must be one of the quotes it was shown; an ASK part only where no quote answers it; blocks that are not
     built make the idea BLOCKED (never invented); machine/talker/sanity.py (0.68 next to "golden ratio", impossible
     levels, unknown charts, fixed rules).
  6. When code refuses the form, or its checks find a question the form must ask, ONE fix-up call sends the form
     and the findings back (Tejas's yes, 29 Sep); the better of the two forms is kept, and both calls are counted.
  7. An ASK part carries the #129 sentence ("Searched strategy.md, LEARNED.md ... for X - closest line: ...") made by
     code from the search, never by Claude.
Answers are cached by (prompt hash, model) - machine/talker/llm.py.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from machine import config, rules, settings
from machine.ideas import compile as C
from machine.ideas.form import TF, FormError, Idea, apply_changes
from machine.talker import glossary, llm, terms
from machine.talker.glossary import Applied

TOOL_NAME = "fill_idea_form"
HITS_SHOWN = 5                       # quotes per term shown to Claude (L8b: the right place is in the first 3 for 23 of 30)
RULE_KINDS = ("idea", "rule:skip", "rule:entry", "rule:exit", "rule:filter", "rule:management")

# What the not-built names mean (compile.PLANNED names the prompt that builds each). Claude's one-line summaries of
# the plan's rows, so the talker names the right one.
PLANNED_MEANS = {
    "trendline": "a trendline (3 touches, #92) - its touch, its break, its sweep",
    "triangle": "a triangle pattern", "wedge": "a wedge pattern",
    "false_breakout": "a false break of a level or a pattern line, then back inside",
    "zone": "a strong zone (#79-#81)", "zone_break": "a strong zone broken with a close, then its retest (#97, #98)",
    "zone_entry": "the zone entry: obvious liquidity just past a strong zone taken, entry AT the zone (#84, #86)",
    "bias": "the direction of the hour / day / week chart (#70-#74)",
    "higher_timeframe": "the higher timeframe with or against the trade (same prompt as bias)",
    "all_time_high": "the real all-time high (#72)", "hammer": "his hammer candle (#115)",
    "premium_discount": "premium / discount: the ADR red / green zones (strategy.md section 7)",
    "halfway": "price went halfway to the target and turned (four-boxes B2-05)",
    "news": "a news event: Forex Factory red and orange (T7)",
}


# ================================================================================================= the tool's schema
def _obj(props: dict, **extra) -> dict:
    """A strict object: every property required, nothing else allowed (strict tool use)."""
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props, **extra}


def _null(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_S = {"type": "string"}
_SOURCES = {"type": "array", "items": _S,
            "description": "hit ids from the search results, rules.py / settings.py names, 'his words', or "
                           "'default: <where>'"}
_NV = _obj({"name": _S, "value_json": {"type": "string", "description": "the value as JSON text: 30, \"no limit\", "
                                                                        "[\"equal\"], true"}})
_TF = {"type": "string", "enum": list(TF.__args__)}


def tool() -> dict:
    """fill_idea_form: the idea form (machine/ideas/form.py) in a shape strict tool use accepts - every object closed,
    every field required, maps written as name + JSON value."""
    step = _obj({
        "id": {"type": "string", "description": "a short name in small letters: sweep, brk, gap"},
        "what": {"type": "string", "description": "plain words, one line"},
        "block": {"type": "string", "description": "a built block, or a not-built name"},
        "tf": _TF,
        "max_wait": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["first step", "no limit"]}]},
        "params": {"type": "array", "items": _NV},
        "sources": _SOURCES})
    entry = _obj({"how": {"type": "string", "enum": ["limit", "next_open"]},
                  "at": _null({"type": "string", "description": "limit: <step id>.<what it found>, e.g. gap.near"}),
                  "max_wait": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["no limit"]},
                                         {"type": "null"}]},
                  "sources": _SOURCES})
    exit_ = _obj({"kind": {"type": "string", "enum": ["hedge", "stop"]},
                  "at": {"type": "string", "description": "<step id>.<what it found>, e.g. brk.extreme"},
                  "beyond_usd": {"type": "number"},
                  "management": {"type": "array", "items": _NV,
                                 "description": "hedge only: the management settings this idea changes"},
                  "sources": _SOURCES})
    target = _obj({"kind": {"type": "string", "enum": ["ratio", "price", "next_liquidity"]},
                   "ratio": _null({"type": "number"}), "at": _null({"type": "string"}),
                   "kinds": _null({"type": "array", "items": {"type": "string",
                                                              "enum": ["equal", "session", "swing", "trendline"]}}),
                   "pick": {"type": "string", "enum": ["nearest", "his_order"]},
                   "sources": _SOURCES})
    rule = _obj({"name": _S, "params": {"type": "array", "items": _NV}, "words": _null(_S), "sources": _SOURCES})
    reading = _obj({"id": _S, "means": _S,
                    "changes": {"type": "array", "items": _obj({"path": _S, "value_json": _S})}})
    unclear = _obj({"id": _S, "kind": {"type": "string", "enum": ["ask", "version"]},
                    "term": {"type": "string", "description": "the searched term it belongs to"},
                    "part": {"type": "string", "description": "what is open, in plain words"},
                    "words": _null({"type": "string", "description": "his words that leave it open"}),
                    "readings": {"type": "array", "items": reading}})
    schema = _obj({
        "voice_typing": {"type": "array", "items": _obj({"heard": _S, "means": _S})},
        "terms": {"type": "array", "items": _obj({"term": _S, "answered_by": _null(_S)})},
        "kind": {"type": "string", "enum": list(RULE_KINDS)},
        "title": _S,
        "timeframe": _TF,
        "directions": {"type": "array", "items": {"type": "string", "enum": ["buy", "sell"]}},
        "steps": {"type": "array", "items": step},
        "entry": entry,
        "exit": exit_,
        "targets": {"type": "array", "items": target},
        "do_not_enter": {"type": "array", "items": rule},
        "filters": {"type": "array", "items": rule},
        "settings": {"type": "array", "items": _NV},
        "unclear": {"type": "array", "items": unclear},
        "missing": {"type": "array", "items": _obj({"name": _S, "what": _S})},
        "already_seen": {"type": "array", "items": _S},
        "notes": _null(_S)})
    return {"name": TOOL_NAME, "strict": True, "input_schema": schema,
            "description": "Fill the idea form for Tejas's words. Call it exactly once, with every field."}


# ================================================================================================= what Claude reads
INSTRUCTIONS = """\
You turn one message from Tejas into an idea form that code can test. Tejas is an expert gold (XAUUSD) trader. He \
types by voice on his phone, so his words have voice-typing mistakes, short forms (fvg = fair value gap, bos = break \
of structure) and half sentences. Answer by filling the form fill_idea_form exactly once (as the tool call, or as \
the JSON form you are given). Your only output is that form: instructions about the style of a chat reply do not \
apply here. The form is a draft: code checks it, and nothing is tested from it until its unclear parts are settled.

How this machine works:
- The AI writes rules; code judges. A form names BLOCKS (the steps code can find on the chart), DO-NOT-ENTER rules \
and FILTERS from the lists below, and nothing else - no code, no formulas.
- His meanings come from his documents, never from general trading knowledge. His documents were searched for every \
term he used; the results are in the message. When two places disagree, the newest lesson wins (the order of trust: \
LEARNED lessons and the inbox (his words) > strategy-understanding > rules.py and his settings > strategy.md > the \
rest).

Fill the tool's fields in order:
1. voice_typing: every voice-typing mistake or short form you read as something else, beyond the replacements \
already made (for example "1 is to 1" = 1:1; "S" alone at the start of his answer = skip, his card answers used T / S \
= take / skip). The message lists words that MAY be voice typing; list the ones that were.
2. terms: one row per searched term, in the order given. answered_by = the id of the ONE quote that states what he \
means by this term here, or null when none does. A quote that only uses the word does not answer it. A place that \
says the point is still open, or asks him about it, does not answer it. Prefer his own words (LEARNED, the inbox) \
when two answer.
3. kind: "idea" when his words describe a setup to trade. When they give one rule for his usual trades - a reason to \
skip (rule:skip), where to enter (rule:entry), where or when to exit (rule:exit), a condition that keeps or drops \
setups (rule:filter), how to manage an open trade (rule:management: close it, hedge it, hold it or not over the \
weekend or the news) - use that kind, and write the form as his usual setup (the example below) with this rule in it.
4. The form (title, timeframe, directions, steps, entry, exit, targets, do_not_enter, filters, settings):
   - What his words say, take from his words. What they leave out, take from his documented defaults (below) and \
cite the default. Never fill a part with a meaning you guessed.
   - Directions: the side his words name. A buy word (buy, bullish, up, the week / day / hour is up, only buy, a low \
taken, a session low) = buy only; a sell word = sell only. Both sides only when he says so ("even for bullish or \
bearish") or when his words name no side at all.
   - The hedge: every entry type has its documented hedge place (strategy.md section 9 and the defaults below) - use \
it, do not ask where the hedge is. Type 1 = the grab's extreme (the break step's `extreme`); Type 2 = a few pips past \
the line, moving with it (how many = a number the machine tries, B3-16); the zone entry and the zone-break retest = \
just past the zone's far edge. "Keep the hedge near" is a number the machine tries (B3-17). Ask about the hedge only \
when his own words contradict its documented place (a ratio that needs another stop).
   - Every step, entry, exit and target lists its sources: hit ids from the search results, rules.py / settings.py \
names, "his words", or "default: <where>".
   - Use only the built blocks with their params. When his idea needs something not built, name it with a not-built \
name (a step's block, a filter's or do-not-enter rule's name, a target's kinds): the idea then waits for that piece \
(BLOCKED), which is the right result. Never stand in a built block for a different thing (a sweep of equal lows is \
not a trendline). A piece no name covers (a management rule with no setting, a new kind of level) goes in `missing`.
   - His fixed rules never change (LEARNED #121). If his words seem to break one, keep the fixed rule and say so in \
notes.
   - Settings: only names from the settings list, with a value it allows.
5. unclear: the parts his words leave open. Each belongs to a term (the searched term, or the plain word when it was \
not searched), has 2-4 readings that trade differently, and reading A is the form exactly as you wrote it (no \
changes); each other reading lists its changes as dotted paths.
   - ask = a MEANING only he knows (Box 2 words: steep, big move, very strong, good, halfway, positive / false news), \
or a part that decides the trade and that no quote answers (a stop he never gave, a number that is not the known \
level). Only when its term's answered_by is null. He will be asked with pictures.
   - version = a number or choice the machine can try by itself (Box 3: how many pips, how near, how long to wait, \
how many hours before the close). Never asked. "Too near", "too far", "too big" are numbers too: his number (or the \
number his answers show) is one reading, the numbers around it are the others - his words, 29 Sep 2026: "These type \
of ambiguity it should comeup with results ... the edge cases and other probabilities it should only check for the \
best results. And should Inform me."
   - An entry type carries its Box 2 word even when he does not say it: a trendline touch entry (Type 2) needs a \
"steep" line (B2-01, #127); a triangle needs a "big move" before it (B2-02); the second gap needs a "very strong" \
trend (B2-03). That word is an ask part whenever the idea uses that entry.
   - Do not mark as unclear what his documents define (Box 1, a term with an answered_by, a default, a rule in \
strategy.md sections 5, 9, 10, a fixed rule, a setting marked his). When no quote of a term answers it, read the \
reference below before you call it unclear. He has complained about questions he already answered (#129): a \
needless question costs more than a missed one.
6. missing: pieces the idea needs that no built or not-built name covers.
7. already_seen: the idea_key of every old try (below) that is this same idea (the same blocks and entry); [] when \
none.
8. notes: a short line only when needed (a fixed rule his words seem to break, a default you had to choose).

value_json holds JSON text: 30, "no limit", ["equal"], true.
max_wait: the first step has "first step"; each later step a number of bars of its own chart, "no limit", or 0 for a \
check made at that moment (a block marked "a check" must have 0).
Reading paths (a reading can only change what the form already has, through these paths):
- steps.<id>.block, steps.<id>.tf, steps.<id>.max_wait_bars, steps.<id>.params.<name> - one part of a step that exists
- steps.<id> - a step that exists, replaced by a whole step (JSON: id, what, block, tf, max_wait_bars, params), or \
null to remove it
- steps - the WHOLE list of steps (JSON list of whole steps): the only way to ADD a step or change their order
- entry.at, entry.how, entry.max_wait_bars; exit.kind, exit.at, exit.beyond_usd, exit.management.<name> (a hedge only)
- exit - the whole exit: {"kind": "stop", "at": ..., "beyond_usd": 0} or {"kind": "hedge", "at": ..., \
"beyond_usd": 0, "management": {}} - a stop has no management
- targets (the whole list), targets.<n>.<field>, settings.<name>, directions, timeframe, filters, do_not_enter
Inside a reading, write a whole step, exit, target list or rule list in the same shape as the tool's own fields \
(max_wait, params as name + value_json pairs, sources as strings); code converts it. Every reading must still make a \
good form, and code checks each one: a block that can only start an idea stays the first step; the sweep, the break \
and the gap stay on one chart; a step a later part points at must still exist.
"""

DEFAULTS = """\
## His documented defaults (when his words do not say; cite them)
- chart: 5-minute. strategy.md section 5: "Entry TF = 5-min always (FVG)."
- his usual setup = Type 1 (strategy.md section 9): his obvious liquidity taken (a sweep) -> break of structure -> \
the FIRST gap after the liquidity was taken, made by the breaking move (#61, #62, #131).
- entry: a limit order at the NEAR edge of that gap (#65), on price's first visit back; the order waits "no limit" \
(#68: "it's not about the time").
- hedge (he never calls it a stop): at the extreme of the whole grab = the break step's `extreme` (#9, #33, #111; fixed \
rule a_grab_that_keeps_going_is_still_a_grab).
- target: the next liquidity, which must really exist (#119; strategy.md section 10: "Target = next liquidity / zone").
- waits: the break within 30 bars after the sweep (settings state_timeout_bars, Claude's number); the gap within 6 \
bars after the break (rules.py gap_must_belong_to_the_breaking_move).
"""

# His usual setup (strategy.md section 9, Type 1) in the tool's own shape: the example the talker copies the format of.
EXAMPLE = {
    "voice_typing": [], "terms": [], "kind": "idea", "title": "his usual setup: grab + gap", "timeframe": "5m",
    "directions": ["buy", "sell"],
    "steps": [
        {"id": "sweep", "what": "his obvious liquidity is taken (a wick is enough)", "block": "sweep", "tf": "5m",
         "max_wait": "first step", "params": [], "sources": ["default: strategy.md section 9 Type 1"]},
        {"id": "brk", "what": "break of structure the trade's way", "block": "break_of_structure", "tf": "5m",
         "max_wait": 30, "params": [{"name": "after", "value_json": "\"sweep\""}],
         "sources": ["default: strategy.md section 9 Type 1", "settings.py state_timeout_bars"]},
        {"id": "gap", "what": "the first gap of the breaking move", "block": "gap_of_the_move", "tf": "5m",
         "max_wait": 6, "params": [], "sources": ["default: #61, #62, #131"]}],
    "entry": {"how": "limit", "at": "gap.near", "max_wait": "no limit", "sources": ["default: #65, #68"]},
    "exit": {"kind": "hedge", "at": "brk.extreme", "beyond_usd": 0.0, "management": [],
             "sources": ["default: strategy.md section 9 Type 1 (#9, #33, #111)"]},
    "targets": [{"kind": "next_liquidity", "ratio": None, "at": None, "kinds": None, "pick": "nearest",
                 "sources": ["default: #119"]}],
    "do_not_enter": [], "filters": [], "settings": [], "unclear": [], "missing": [], "already_seen": [],
    "notes": None,
}


def _rules_text() -> str:
    return "\n".join(f"- {k} = {r.value!r}: {r.means} ({r.taught})" for k, r in rules.ALL.items())


def _settings_text() -> str:
    out = []
    for k, s in settings.CHANGEABLE.items():
        opt = f" Allowed: {list(s.options)}." if s.options else ""
        out.append(f"- {k} = {s.value!r} [{s.status}; {s.who}]: {s.means}{opt}")
    for k, s in settings.MORE.items():
        means = s.means if len(s.means) <= 220 else s.means[:219].rsplit(" ", 1)[0] + "…"
        out.append(f"- {k} = {s.value!r} [{s.status}]: {means}")
    return "\n".join(out)


def _section(path, start: str, stop: str) -> str:
    lines = (config.ROOT / path).read_text(encoding="utf-8").splitlines()
    a = next(i for i, x in enumerate(lines) if x.startswith(start))
    b = next(i for i, x in enumerate(lines) if i > a and x.startswith(stop))
    return "\n".join(lines[a:b]).strip()


def _prior_text() -> str:
    with open(config.PRIOR_TRIES_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return "idea_key | blocks | timeframe | entry | tries | years\n" + "\n".join(
        f"{r['idea_key']} | {r['blocks']} | {r['timeframe']} | {r['entry']} | {r['tries']} | {r['years']}"
        for r in rows)


@lru_cache(maxsize=1)
def prior_keys() -> frozenset[str]:
    with open(config.PRIOR_TRIES_CSV, encoding="utf-8", newline="") as f:
        return frozenset(r["idea_key"] for r in csv.DictReader(f))


@lru_cache(maxsize=1)
def reference() -> str:
    """Everything Claude reads that is the same for every sentence (cached by the API: one write, then reads)."""
    planned = "\n".join(f"- {n}: {PLANNED_MEANS.get(n, '')} -> {C._planned(n)}" for n in C.PLANNED)
    parts = [
        INSTRUCTIONS,
        DEFAULTS,
        "## Blocks, do-not-enter rules and filters that are BUILT (a form may name only these, with these params)\n"
        + C.describe(),
        "## Not built yet: name them exactly so the idea waits for them (BLOCKED)\n" + planned,
        "## His fixed rules (machine/rules.py) - never changed by a form\n" + _rules_text(),
        "## Settings (machine/settings.py) - the only names a form's settings or hedge management may use. "
        "[his] = his rule (a change is only a comparison); [test] = Claude's number, free to try\n" + _settings_text(),
        "## strategy.md sections 5, 9, 10 (his canonical rules)\n"
        + _section("strategy/strategy.md", "## 5.", "## 6.") + "\n\n"
        + _section("strategy/strategy.md", "## 9.", "## 11."),
        "## The four boxes (plan/work/four-boxes.md, T2 - his yes): Box 2 = ask him; Box 3 = the machine tries\n"
        + _section("plan/work/four-boxes.md", "## Box 2", "## Box 4"),
        "## Ideas the old project already tried on 2016-2023 (plan/work/prior-tries.csv)\n" + _prior_text(),
        "## Example: his usual setup as a tool call\n" + json.dumps(EXAMPLE, ensure_ascii=False),
    ]
    return "\n\n".join(parts)


# ================================================================================================= the search
@dataclass(frozen=True)
class Search:
    term: terms.Term
    found: bool              # the search's own FOUND / NOT FOUND (L8b's lines)
    hits: tuple              # machine.search.rank.Hit, the first HITS_SHOWN
    sentence: str            # the #129 sentence (NOT FOUND) or "Found in ..."
    closest: Any = None      # the closest Hit


SearchFn = Callable[[str], Any]      # q -> machine.search.rank.Answer


def default_search() -> SearchFn:
    from machine.search import index as ix
    from machine.search.rank import answered
    index = ix.get()
    return lambda q: answered(q, index=index)


def run_searches(text: str, search: SearchFn) -> tuple[Search, ...]:
    out = []
    for t in terms.find(text):
        a = search(t.query)
        out.append(Search(t, bool(a.found), tuple(a.hits[:HITS_SHOWN]), a.sentence, a.closest))
    return tuple(out)


def user_message(g: Applied, searches: tuple[Search, ...]) -> str:
    lines = ["His words, as typed:", g.words, ""]
    if g.replaced:
        lines += ["After the glossary (read this):", g.text, "",
                  "Replaced: " + "; ".join(f'"{r.heard}" -> "{r.means}" ({r.lesson})' for r in g.replaced)]
    else:
        lines += ["(No voice-typing phrase of the glossary in his words.)"]
    if g.hints:
        lines += ["Maybe voice typing, NOT replaced - decide from the sentence: "
                  + "; ".join(f'"{h.heard}" can mean "{h.means}" ({h.lesson}; {h.note})' for h in g.hints)]
    lines += ["", f"The search of his documents (#129): {len(searches)} terms, the best {HITS_SHOWN} places each."]
    if not searches:
        lines.append("(no term of his trading words was found in the sentence)")
    for k, s in enumerate(searches, 1):
        verdict = "FOUND" if s.found else "NOT FOUND"
        lines.append(f'\n### {k}. term "{s.term.name}" - he said "{s.term.said}" - searched "{s.term.query}" - '
                     f"the search says {verdict}")
        for h in s.hits:
            lines.append(f'- {h.id} [{h.date or "no date"}]: "{h.quote}"')
    return "\n".join(lines)


def request(g: Applied, searches: tuple[Search, ...], model: str | None = None, effort: str | None = None,
            backend: str | None = None) -> llm.Request:
    system = ({"type": "text", "text": reference(), "cache_control": {"type": "ephemeral"}},)
    return llm.Request(system=system, tools=(tool(),), user=user_message(g, searches), tool=TOOL_NAME,
                       model=model or config.TALKER_MODEL, effort=effort or config.TALKER_EFFORT,
                       backend=backend or config.TALKER_BACKEND)


# ================================================================================================= answer -> form
class WireError(ValueError):
    """The tool's answer cannot become a form (a bad JSON value, a max_wait in the wrong place ...)."""


def _json(v: str, where: str) -> Any:
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        raise WireError(f"{where}: {v!r} is not JSON text") from None


def _pairs(rows: list[dict], where: str) -> dict:
    return {r["name"]: _json(r["value_json"], f"{where} '{r['name']}'") for r in rows or []}


def _src(refs: list[str]) -> list[dict]:
    return [{"ref": r} for r in refs or [] if str(r).strip()]


def _form_shape(v: Any, where: str) -> Any:
    """A reading's value written in the tool's own shape -> the form's shape: sources as plain strings, params and
    hedge management as name + value_json pairs, max_wait, optional fields as null. The talker writes one shape
    everywhere (a whole step, exit, target list, filter list ...); this is the one place it is converted."""
    if isinstance(v, list):
        return [_form_shape(x, where) for x in v]
    if not isinstance(v, dict):
        return v
    out = {}
    for k, x in v.items():
        if k == "sources" and isinstance(x, list):
            out[k] = [s if isinstance(s, dict) else {"ref": str(s)} for s in x if s is not None and str(s).strip()]
        elif k in ("params", "management") and isinstance(x, list) and all(isinstance(p, dict) and "name" in p
                                                                           for p in x):
            out[k] = _pairs(x, f"{where} {k}")
        elif k == "max_wait":
            out["max_wait_bars"] = None if x == "first step" else x
        elif x is None and k in ("words", "ratio", "at", "kinds"):
            continue                                   # optional parts the tool writes as null
        else:
            out[k] = _form_shape(x, where)
    if out.get("kind") == "stop" and not out.get("management"):
        out.pop("management", None)                    # a stop has no hedge management
    return out


def _wait(v: Any, where: str, first: bool) -> Any:
    if first:
        if v not in ("first step", None):
            raise WireError(f"{where}: the first step starts the idea - its max_wait is \"first step\", not {v!r}")
        return None
    if v == "first step":
        raise WireError(f"{where}: only the first step has \"first step\"")
    return v


KIND_FAMILIES = C.KIND_FAMILIES                # compile.Kind: the side comes from the trade's direction
family_kinds = C.family_kinds                  # "session_high" -> "session" (one definition: the compiler's, L6b.3)


def to_form(w: dict, idea_id: str, words: str) -> dict:
    """The tool's answer -> the dict machine.ideas.form.Idea loads. Raises WireError."""
    steps = []
    for k, s in enumerate(w["steps"]):
        params = _pairs(s["params"], f"steps[{k}] params")
        if "kinds" in params:
            params["kinds"] = family_kinds(params["kinds"])
        steps.append({"id": s["id"], "what": s["what"], "block": s["block"], "tf": s["tf"],
                      "max_wait_bars": _wait(s["max_wait"], f"steps[{k}] '{s['id']}'", k == 0),
                      "params": params, "sources": _src(s["sources"])})
    e = w["entry"]
    entry = {"how": e["how"], "sources": _src(e["sources"])}
    if e["how"] == "limit":
        entry.update(at=e["at"], max_wait_bars=e["max_wait"])
    x = w["exit"]
    ex = {"kind": x["kind"], "at": x["at"], "beyond_usd": x["beyond_usd"], "sources": _src(x["sources"])}
    if x["kind"] == "hedge":
        ex["management"] = _pairs(x["management"], "exit.management")
    elif x["management"]:
        raise WireError("exit.management: only a hedge has management settings")
    targets = []
    for t in w["targets"]:
        d = {"kind": t["kind"], "pick": t["pick"], "sources": _src(t["sources"])}
        for f in ("ratio", "at", "kinds"):
            if t[f] is not None:
                d[f] = family_kinds(t[f]) if f == "kinds" else t[f]
        targets.append(d)
    rules_ = lambda rows, where: [{"name": r["name"], "params": _pairs(r["params"], f"{where} '{r['name']}'"),  # noqa
                                   **({"words": r["words"]} if r["words"] else {}), "sources": _src(r["sources"])}
                                  for r in rows]
    unclear = []
    for u in w["unclear"]:
        unclear.append({"id": u["id"], "kind": u["kind"], "part": u["part"],
                        **({"words": u["words"]} if u["words"] else {}),
                        "readings": [{"id": r["id"], "means": r["means"],
                                      "changes": {c["path"]: _form_shape(_json(c["value_json"], f"unclear {u['id']} "
                                                                               f"reading {r['id']} '{c['path']}'"),
                                                                         f"reading {r['id']} '{c['path']}'")
                                                  for c in r["changes"]}} for r in u["readings"]]})
    notes = [f"kind: {w['kind']}"] + ([w["notes"]] if w["notes"] else [])
    notes += [f"missing: {m['name']} - {m['what']}" for m in w["missing"]]
    return {"id": idea_id, "title": w["title"], "words": words, "timeframe": w["timeframe"],
            "directions": w["directions"], "steps": steps, "entry": entry, "exit": ex, "targets": targets,
            "do_not_enter": rules_(w["do_not_enter"], "do_not_enter"), "filters": rules_(w["filters"], "filters"),
            "unclear": unclear, "settings": _pairs(w["settings"], "settings"), "notes": "\n".join(notes)}


def check_form(idea: Idea) -> tuple[list[tuple[str, str, str]], list[str]]:
    """(what it needs that is not built, problems). Problems make the form invalid: his fixed rules, a setting that
    does not exist, a step or name that points nowhere, a reading that makes a bad form."""
    problems: list[str] = []
    missing: list[tuple[str, str, str]] = []
    try:
        missing = C.check_without_bars(idea)
    except (rules.FixedRuleError, settings.SettingError, C.CompileError, FormError) as err:
        problems.append(f"{type(err).__name__}: {err}")
    for u in idea.unclear:
        for r in u.readings:
            if not r.changes:
                continue
            try:
                f2 = apply_changes(idea, r.changes, f"reading {r.id} of {u.id}")
                for m in C.check_without_bars(f2):           # a reading that needs what is not built blocks too:
                    if m[0] not in {x[0] for x in missing}:  # its version would stop the tune halfway (L6b.3)
                        missing.append((m[0], f"reading {r.id} of {u.id}: {m[1]}", m[2]))
            except (rules.FixedRuleError, settings.SettingError, C.CompileError, FormError) as err:
                problems.append(f"reading {r.id} of {u.id} ({u.part}): {type(err).__name__}: {err}")
    return missing, problems


# ================================================================================================= the draft
@dataclass
class Question:
    id: str
    kind: str                  # ask | version
    term: str
    part: str
    searched: str              # the #129 sentence, made by code
    answered_by: str | None    # the talker's own pick for that term (an ASK with one is flagged)


@dataclass
class Draft:
    words: str
    glossary: Applied
    searches: tuple[Search, ...]
    answer: llm.Answer | None
    wire: dict | None = None
    form: Idea | None = None
    status: str = "NO FORM"
    kind: str | None = None
    missing: list[tuple[str, str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)     # non-empty = an invalid form
    questions: list[Question] = field(default_factory=list)
    picks: dict[str, str | None] = field(default_factory=dict)
    unknown_citations: list[str] = field(default_factory=list)
    already_seen: list[str] = field(default_factory=list)
    voice: list[tuple[str, str]] = field(default_factory=list)   # the talker's own voice-typing readings
    sanity: list = field(default_factory=list)                   # machine.talker.sanity.Check
    fixup: list[str] = field(default_factory=list)               # what code sent back for the fix-up call
    other: "Draft | None" = None                                 # after a fix-up: the draft NOT kept
    kept: str = ""                                               # after a fix-up: which form was kept

    @property
    def valid(self) -> bool:
        return self.form is not None and not self.problems

    def answers(self) -> list[llm.Answer]:
        """Every answer this draft took: its own, and the other draft's when a fix-up call was made."""
        return ([self.answer] if self.answer else []) + ([self.other.answer] if self.other and self.other.answer
                                                          else [])

    @property
    def cost_usd(self) -> float:
        return sum(a.paid_now for a in self.answers())

    def needs(self) -> set[str]:
        """Every block, rule, filter and liquidity kind the form names, built or not, and its `missing` names."""
        out: set[str] = {n for n, _, _ in self.missing}
        if self.form is not None:
            f = self.form
            out |= {s.block for s in f.steps} | {r.name for r in f.do_not_enter} | {r.name for r in f.filters}
            for s in f.steps:
                out |= set(s.params.get("kinds") or ())
            for t in f.targets:
                out |= set(t.kinds or ())
        if self.wire:
            out |= {m["name"] for m in self.wire.get("missing", [])}
        return out

    def text(self) -> str:
        from machine.talker import report
        return report.draft_text(self)


def _pick_map(w: dict, searches: tuple[Search, ...]) -> tuple[dict[str, str | None], list[str]]:
    """The talker's pick per searched term, and every pick that is not one of the quotes it was shown."""
    shown = {s.term.name: {h.id for h in s.hits} for s in searches}
    picks, bad = {}, []
    for row in w.get("terms", []):
        name, hit = row["term"], row["answered_by"]
        if hit is not None:
            hit = re.sub(r"^term\s*\d+\s*:\s*", "", hit.strip())       # "term4: LEARNED #92" -> "LEARNED #92"
        picks[name] = hit
        if hit is not None and hit not in shown.get(name, set()):
            bad.append(f'term "{name}": answered_by {hit!r} was not among its quotes')
    return picks, bad


def _known_ref(ref: str, hit_ids: set[str], stable: str) -> bool:
    """A source the talker could have read: his words, a default, a quote it was shown, or a place in the reference
    (a rules / settings name, a lesson number, a strategy.md section, a four-boxes row) - never an invented place."""
    r = ref.strip()
    low = r.lower()
    if low.startswith(("default", "his words", "not built")) or r in hit_ids:
        return True
    r2 = re.sub(r"\bsection\s+", "§", r)                               # "strategy.md section 1 line 78"
    if any(h in r or h in r2 for h in hit_ids) or low in stable.lower():
        return True
    sec = re.fullmatch(r"strategy\.md (?:section|§)\s*(\d+)(?:.*)?", r)
    if sec and f"## {sec.group(1)}." in stable:
        return True
    box = re.search(r"\bB[1-4]-\d{2}\b", r)
    if box and box.group(0) in stable:
        return True
    names = re.findall(r"(?:rules\.py|settings\.py)\s+([a-z_0-9]+)", r) or ([r] if re.fullmatch(r"[a-z_0-9]+", r)
                                                                           else [])
    if names and all(n in rules.ALL or n in settings.ALL for n in names):
        return True
    nums = re.findall(r"#(\d+)", r)
    return bool(nums) and all(f"#{n}" in stable for n in nums)


def _find_search(term: str, searches: tuple[Search, ...]) -> Search | None:
    t = term.strip().lower()
    for s in searches:
        if s.term.name.lower() == t or s.term.said.lower() == t:
            return s
    for s in searches:
        if t and (t in s.term.name.lower() or s.term.name.lower() in t or t in s.term.query.lower()):
            return s
    return None


def _sentence(term: str, s: Search | None, search: SearchFn | None) -> str:
    from machine.search.rank import SEARCHED
    if s is None and search is not None:
        a = search(term)
        c = a.closest
        line = f'{c.id}: "{c.quote}"' if c else "none"
        extra = " (searched after the talker answered; the search says FOUND - check it before asking)" if a.found \
            else ""
        return f'Searched {SEARCHED} for "{term}" — closest line: {line}{extra}'
    if s is None:
        return f'Searched {SEARCHED} for "{term}" — not searched'
    if not s.found:
        return s.sentence
    c = s.hits[0] if s.hits else s.closest
    line = f'{c.id}: "{c.quote}"' if c else "none"
    return f'Searched {SEARCHED} for "{s.term.query}" — places found, none says what he means; closest line: {line}'


def build(words: str, answer: llm.Answer | None, g: Applied, searches: tuple[Search, ...], idea_id: str,
          search: SearchFn | None = None) -> Draft:
    """The draft from one answer: the form, its status, every check. No API call here."""
    from machine.talker import sanity
    d = Draft(words, g, searches, answer)
    if answer is None or answer.input is None:
        d.problems.append(f"no answer: {answer.error if answer else 'not asked'}")
        return d
    w = d.wire = answer.input
    d.kind = w.get("kind")
    d.voice = [(v["heard"], v["means"]) for v in w.get("voice_typing", [])]
    d.picks, bad_picks = _pick_map(w, searches)
    d.unknown_citations += bad_picks
    d.already_seen = list(w.get("already_seen", []))
    for key in d.already_seen:
        if key not in prior_keys():
            d.unknown_citations.append(f"already_seen {key!r} is not an idea_key of prior-tries.csv")
    try:
        form_dict = to_form(w, idea_id, words)
        d.form = Idea.load(form_dict)
    except (WireError, FormError, KeyError, TypeError) as err:
        d.problems.append(f"{type(err).__name__}: {err}")
        return d
    d.missing, problems = check_form(d.form)
    d.problems += problems
    # the #129 sentence for every unclear part, made here from the search - never by Claude
    unclear = []
    for u_w, u in zip(w["unclear"], d.form.unclear):
        s = _find_search(u_w["term"], searches)
        pick = d.picks.get(s.term.name) if s else None
        sentence = _sentence(u_w["term"], s, search)
        d.questions.append(Question(u.id, u.kind, u_w["term"], u.part, sentence, pick))
        unclear.append(u.model_copy(update={"searched": sentence}))
    status = "BLOCKED" if (d.missing or w.get("missing")) else "FORM DRAFTED"
    d.form = d.form.model_copy(update={"unclear": tuple(unclear), "status": status})
    d.status = status if not d.problems else "NO FORM"
    hit_ids = {h.id for s in searches for h in s.hits}
    stable = reference()
    for where, refs in _refs(w):
        for r in refs:
            if not _known_ref(r, hit_ids, stable):
                d.unknown_citations.append(f"{where}: {r!r}")
    d.sanity = sanity.check(d)
    return d


def _refs(w: dict):
    for k, s in enumerate(w["steps"]):
        yield f"steps[{k}]", s["sources"]
    yield "entry", w["entry"]["sources"]
    yield "exit", w["exit"]["sources"]
    for k, t in enumerate(w["targets"]):
        yield f"targets[{k}]", t["sources"]
    for group in ("do_not_enter", "filters"):
        for r in w[group]:
            yield f"{group} '{r['name']}'", r["sources"]


def prepare(words: str, search: SearchFn | None = None, model: str | None = None, effort: str | None = None,
            backend: str | None = None) -> tuple[Applied, tuple[Search, ...], llm.Request]:
    """Steps 1-2 and the request of step 3 - no call yet (the eval sends many of these together)."""
    g = glossary.apply(words)
    searches = run_searches(g.text, search or default_search())
    return g, searches, request(g, searches, model, effort, backend)


# ================================================================================================= the fix-up call
FIXUP = ("YOUR FIRST FORM, checked by code. Code refused it, or its checks found a question the form must ask. Your "
         "first form is below, then every finding, word for word. Call fill_idea_form again with the WHOLE form, "
         "fixed: change what the findings name, keep everything else as it was.")


def fixup_findings(d: Draft) -> list[str]:
    """What one fix-up call may repair (Tejas's yes, 29 Sep - inbox 0600): the form's problems (a reading that makes a
    bad form, a value that is not JSON, a fixed rule changed ...) and the sanity checks that failed ("problem": an
    impossible form; "ask": a question the checks require, like 0.68 next to "golden ratio"). Nothing when there was
    no answer at all - a failed call is not a wrong form."""
    if d.wire is None:
        return []
    return list(d.problems) + [f"check {c.name}: {c.message}" for c in d.sanity
                               if not c.ok and c.severity in ("problem", "ask")]


def fixup_request(req: llm.Request, d: Draft, findings: list[str]) -> llm.Request:
    """The same request (the same fixed text, so the API / Claude Code cache it), with the first form and the
    findings added to his message."""
    user = (req.user + "\n\n---\n" + FIXUP + "\n\nYour first form:\n" + json.dumps(d.wire, ensure_ascii=False)
            + "\n\nWhat code found:\n" + "\n".join(f"- {x}" for x in findings))
    return dataclasses.replace(req, user=user)


def better(first: Draft, second: Draft, findings: list[str]) -> Draft:
    """The draft to keep after a fix-up: the fix-up's, unless it is invalid where the first was valid. The kept one
    carries the other and the findings (the report shows both; the cost counts both calls)."""
    keep, other = (second, first) if (second.valid or not first.valid) else (first, second)
    keep.fixup, keep.other = findings, other
    keep.kept = "the fix-up's form" if keep is second else "the first form (the fix-up's was invalid)"
    return keep


def translate(words: str, idea_id: str = "idea-900", *, search: SearchFn | None = None, api=None, runner=None,
              model: str | None = None, effort: str | None = None, backend: str | None = None,
              use_cache: bool = True, fixups: int | None = None) -> Draft:
    """His words -> a Draft (see the module text). One call to Claude (none when the answer is cached), and one
    fix-up call only when code refuses the form or its checks require a question it lacks."""
    search = search or default_search()
    g, searches, req = prepare(words, search, model, effort, backend)

    def ask(r: llm.Request) -> llm.Answer:
        try:
            return llm.call(r, api=api, runner=runner, use_cache=use_cache)
        except llm.LLMError as err:
            return llm.Answer(None, r.model, llm.Tokens(), 0.0, False, False, "", str(err), r.key())

    d = build(words, ask(req), g, searches, idea_id, search)
    findings = fixup_findings(d)
    if findings and (config.TALKER_FIXUPS if fixups is None else fixups) > 0:
        d2 = build(words, ask(fixup_request(req, d, findings)), g, searches, idea_id, search)
        d = better(d, d2, findings)
    return d
