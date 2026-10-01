"""The idea form (L7a): one trading idea of Tejas's, written so that code can test it.

    from machine.ideas.form import Idea
    idea = Idea.load("plan/work/ideas/idea-001/form.json")     # a readable FormError if it is not a good form

What a form holds (plan/likhith.md L7a; the fields golden-ideas.md, T4, marks for every sentence):
  id, parent_idea   "idea-001"; an idea with the same blocks, timeframe and entry as an older idea is a VERSION of it:
                    parent_idea names the older one (likhith.md section D)
  title, words      a short name; HIS WORDS, word for word (voice-typing mistakes left in)
  timeframe         the chart the idea is traded on: the tester's timeframe, and the entry's wait counts in its bars
  directions        "buy", "sell" or both. Steps are written the same for both: a sell's hedge side is ABOVE (its
                    highs), its target side BELOW (its lows); a buy is the mirror (steps.md: "a BUY is the same, with
                    highs and lows swapped")
  steps             the ORDERED steps. Each: id (later parts name it: "gap.near"), what (plain words), the block
                    that does it, its timeframe, and its MAXIMUM WAIT - how many bars of its timeframe after the step
                    before it became known the idea waits before it is dropped: a number, "no limit", or 0 for a check
                    made at that same moment. The first step starts the idea and has no wait.
                    e.g. "break of structure on 1h" -> "back to 0.618 within 20 bars"
  entry             limit at a price a step found ("gap.near"), waiting at most N bars of the timeframe (or "no
                    limit" - his #68: "it's not about the time"); or next_open (the next bar's open)
  exit              a STOP at a price a step found, or a HEDGE there with his management (settings.py / the T12
                    table, plan/work/management-decisions.md)
  targets           a fixed ratio of the stop distance ("1 is to 1" = 1.0), a price a step found, or the NEXT
                    LIQUIDITY from the blocks (#119: it must really exist)
  do_not_enter      rules that drop a setup (e.g. the move is already finished - #119 "T05")
  filters           rules that keep only some setups (e.g. only the London session)
  slices            at most 8 splits of the results, chosen BEFORE running (likhith.md section D)
  unclear           the parts his words leave open: ASK (a meaning only he knows - a question with pictures) or
                    VERSION (a number or choice the machine tries by itself, never asked); 2-4 readings each, and
                    what each reading changes in the form
  settings          changes to settings.py this idea makes (his FIXED rules can never be changed - compile refuses)
  sources, status   where each part comes from; where the idea is (the L10b states)

The form checks its own SHAPE here: types, order, the waits, the budgets, the names it points at. What the parts MEAN
- which blocks exist, his fixed rules, the settings - is checked by machine/ideas/compile.py, which refuses a form
that changes one of his fixed rules and quotes his words.
Readings and versions change a form through dotted paths (apply_changes): "steps.brk.block", "entry.at",
"steps.gap.params.move", "settings.order_block_max_age_bars", "targets.0.ratio".
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from machine import config

TF = Literal["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
Direction = Literal["buy", "sell"]
NO_LIMIT = "no limit"
Wait = Union[int, Literal["no limit"], None]

IDEA_ID = re.compile(r"idea-\d{3,}")
# One of HIS OWN entry types written as a form (L19b, machine/entries/forms.py): "entry-type1", "entry-zone-break".
# Not an idea from the phone: the store (plan/work/ideas/) and the intake never hold one.
ENTRY_ID = re.compile(r"entry-[a-z0-9]+(?:-[a-z0-9]+)*")
NAME = re.compile(r"[a-z][a-z0-9_]{0,23}")                  # step ids
REF = re.compile(r"([a-z][a-z0-9_]{0,23})\.([A-Za-z0-9_.]+)")  # "gap.near", "fib.level_0.618"

# The states of an idea (plan/likhith.md L10b). From "CONFIRMED & LOCKED" on, the rule may not change (lesson 16:
# meaning is chosen before money is seen) - machine/ideas/store.py refuses it.
STATES = ("RECEIVED", "SEARCHED", "FORM DRAFTED", "BLOCKED", "QUESTIONS QUEUED", "ANSWERED", "CONFIRMED & LOCKED",
          "SHOWN", "TUNE RUN", "CHECK OPENED", "REPORTED", "DECIDED")
LOCKED_FROM = "CONFIRMED & LOCKED"

# Dimensions 2-9 are slices; 1 (timeframe) and 10 (the idea's own numbers) make VERSIONS (plan/README.md section 5).
SLICE_DIMENSIONS = {"direction": 2, "session": 3, "news": 4, "higher_timeframe_trend": 5, "volatility": 6,
                    "market_type": 7, "place": 8, "time": 9}
VERSION_DIMENSIONS = {"timeframe": 1, "the_ideas_own_numbers": 10, "level": 10, "stop": 10, "target": 10, "wait": 10}
SECTION_D = ("likhith.md section D: a VERSION is a different rule (the timeframe, or the idea's own numbers: level, "
             "stop, target, wait) - it is a try, not a slice")


class FormError(ValueError):
    """A form that is not a good form. The message says where and what to do, one line each."""


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(v: str, what: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{what} must not be empty")
    return v


def _wait_text(w: Wait) -> str:
    return "not given" if w is None else ("no limit" if w == NO_LIMIT else f"{w} bars")


class Source(_M):
    """Where a part comes from: "LEARNED #135 (25 Sep)", "strategy.md section 1", "idea-001/steps.md step 6"."""
    ref: str
    words: str | None = None             # his words from there, word for word

    @field_validator("ref")
    @classmethod
    def _ref(cls, v: str) -> str:
        return _text(v, "a source's ref")


class Step(_M):
    id: str
    what: str                            # plain words: "price takes the equal highs"
    block: str                           # the block that does it: compile.BLOCKS lists them
    tf: TF
    max_wait_bars: Wait = None           # bars of tf after the step before it is known; "no limit"; 0 = a check then
    params: dict[str, Any] = Field(default_factory=dict)
    sources: tuple[Source, ...] = ()

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not NAME.fullmatch(v):
            raise ValueError(f"step id {v!r}: use a short name in small letters, like 'sweep' or 'gap' (later parts "
                             "name it: 'gap.near')")
        return v

    @field_validator("what", "block")
    @classmethod
    def _words(cls, v: str, info) -> str:
        return _text(v, f"a step's {info.field_name}")

    @field_validator("max_wait_bars")
    @classmethod
    def _wait(cls, v):
        if isinstance(v, int) and v < 0:
            raise ValueError("a wait cannot be below 0 bars")
        return v


class Entry(_M):
    how: Literal["limit", "next_open"]
    at: str | None = None                # limit: the price a step found, "gap.near"
    max_wait_bars: Wait = None           # limit: bars of the idea's timeframe the order waits, or "no limit"
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Entry":
        if self.how == "limit":
            if not self.at:
                raise ValueError("a limit entry needs 'at': the price a step found, like 'gap.near'")
            if self.max_wait_bars is None:
                raise ValueError("a limit entry must say its max_wait_bars: how many bars of the idea's timeframe the "
                                 "order waits - a number, or \"no limit\" (his #68: \"it's not about the time\")")
            if self.max_wait_bars == 0:
                raise ValueError("a limit entry must wait at least 1 bar (or \"no limit\")")
        elif self.at is not None or self.max_wait_bars is not None:
            raise ValueError("a next_open entry buys or sells at the next bar's open: it takes no 'at' and no wait")
        return self


class Stop(_M):
    kind: Literal["stop"]
    at: str                              # "brk.extreme", "fib.level_0.786"
    beyond_usd: float = 0.0              # this many dollars further away from the entry
    sources: tuple[Source, ...] = ()

    @field_validator("beyond_usd")
    @classmethod
    def _beyond(cls, v: float) -> float:
        return _not_negative(v, "beyond_usd")


class Hedge(_M):
    """His way: the other side opens at the hedge point instead of a stop. His management rules are settings.py /
    plan/work/management-decisions.md (T12); `management` holds only the ones this idea changes."""
    kind: Literal["hedge"]
    at: str
    beyond_usd: float = 0.0
    management: dict[str, Any] = Field(default_factory=dict)
    sources: tuple[Source, ...] = ()

    @field_validator("beyond_usd")
    @classmethod
    def _beyond(cls, v: float) -> float:
        return _not_negative(v, "beyond_usd")


def _not_negative(v: float, what: str) -> float:
    if v < 0:
        raise ValueError(f"{what} cannot be below 0")
    return v


Exit = Annotated[Union[Stop, Hedge], Field(discriminator="kind")]


class Target(_M):
    kind: Literal["ratio", "price", "next_liquidity"]
    ratio: float | None = None           # ratio: reward / the stop distance ("1 is to 1" = 1.0)
    at: str | None = None                # price: "fib.level_0.5"
    kinds: tuple[str, ...] | None = None  # next_liquidity: "equal", "session", "trendline", "swing" (None = his usual)
    # next_liquidity: the first price meets, his #130 order, or by his bias dial (#82: the hour AND the day with the
    # trade -> the next liquidity beyond the first; against -> the first, a recent 5-minute swing counts - #91)
    pick: Literal["nearest", "his_order", "by_the_dial"] = "nearest"
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Target":
        if self.kind == "ratio":
            if self.ratio is None or self.ratio <= 0 or self.at is not None or self.kinds is not None:
                raise ValueError("a ratio target takes only 'ratio', a number above 0 (1.0 = 1:1)")
        elif self.kind == "price":
            if not self.at or self.ratio is not None or self.kinds is not None:
                raise ValueError("a price target takes only 'at': the price a step found, like 'fib.level_0.5'")
        elif self.ratio is not None or self.at is not None:
            raise ValueError("a next_liquidity target takes no 'ratio' and no 'at' (it finds the level itself)")
        return self


class Rule(_M):
    """A do-not-enter rule or a filter: its name (compile.DO_NOT_ENTER / compile.FILTERS) and its numbers."""
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    words: str | None = None
    sources: tuple[Source, ...] = ()

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _text(v, "a rule's name")


class Slice(_M):
    name: str
    dimension: str                       # one of SLICE_DIMENSIONS (2-9)
    how: str                             # plain words: "by the entry's India time: Asia / London / New York"
    why: str | None = None

    @field_validator("dimension")
    @classmethod
    def _dim(cls, v: str) -> str:
        if v in VERSION_DIMENSIONS:
            raise ValueError(f"'{v}' is not a slice: {SECTION_D}")
        if v not in SLICE_DIMENSIONS:
            raise ValueError(f"'{v}' is not a dimension; a slice is one of {', '.join(SLICE_DIMENSIONS)}")
        return v

    @field_validator("name", "how")
    @classmethod
    def _words(cls, v: str, info) -> str:
        return _text(v, f"a slice's {info.field_name}")


class Reading(_M):
    id: str                              # "A"
    means: str                           # plain words
    changes: dict[str, Any] = Field(default_factory=dict)   # dotted path -> value; {} = the form as written

    @field_validator("id", "means")
    @classmethod
    def _words(cls, v: str, info) -> str:
        return _text(v, f"a reading's {info.field_name}")


class Unclear(_M):
    id: str                              # "F3"
    kind: Literal["ask", "version"]      # ask = a question to him (pictures, no money); version = the machine tries
    part: str                            # "which break counts"
    words: str | None = None             # his words that leave it open
    readings: tuple[Reading, ...]
    searched: str | None = None          # the #129 sentence: "Searched ... for X - closest line: ..."
    answer: str | None = None            # an ASK part he answered: the reading he chose
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Unclear":
        ids = [r.id for r in self.readings]
        if not 2 <= len(ids) <= 4:
            raise ValueError(f"unclear part {self.id}: give 2-4 readings (it has {len(ids)})")
        if len(set(ids)) != len(ids):
            raise ValueError(f"unclear part {self.id}: two readings share an id")
        if self.answer is not None:
            if self.kind == "version":
                raise ValueError(f"unclear part {self.id} is a VERSION: the machine tries each reading; it is never "
                                 "asked, so it has no answer")
            if self.answer not in ids:
                raise ValueError(f"unclear part {self.id}: the answer {self.answer!r} is not one of its readings "
                                 f"{ids}")
        return self


class Idea(_M):
    id: str
    parent_idea: str | None = None
    title: str
    words: str
    timeframe: TF
    directions: tuple[Direction, ...]
    steps: tuple[Step, ...]
    entry: Entry
    exit: Exit
    targets: tuple[Target, ...]
    do_not_enter: tuple[Rule, ...] = ()
    filters: tuple[Rule, ...] = ()
    slices: tuple[Slice, ...] = ()
    unclear: tuple[Unclear, ...] = ()
    settings: dict[str, Any] = Field(default_factory=dict)
    sources: tuple[Source, ...] = ()
    status: str = "FORM DRAFTED"
    notes: str | None = None

    @field_validator("id", "parent_idea")
    @classmethod
    def _idea_id(cls, v: str | None) -> str | None:
        if v is not None and not (IDEA_ID.fullmatch(v) or ENTRY_ID.fullmatch(v)):
            raise ValueError(f"{v!r} is not an idea id; use idea-001, idea-002 ...")
        return v

    @field_validator("title", "words")
    @classmethod
    def _words(cls, v: str, info) -> str:
        return _text(v, f"the idea's {info.field_name}")

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in STATES:
            raise ValueError(f"status {v!r} is not a state; one of: {', '.join(STATES)}")
        return v

    @model_validator(mode="after")
    def _check(self) -> "Idea":
        problems = []
        if self.parent_idea == self.id:
            problems.append(f"parent_idea: {self.id} cannot be its own parent")
        if not self.directions:
            problems.append("directions: give 'buy', 'sell' or both")
        elif len(set(self.directions)) != len(self.directions):
            problems.append("directions: a direction is given twice")
        if not self.steps:
            problems.append("steps: an idea needs at least one step - the one that starts it")
        ids = [s.id for s in self.steps]
        twice = sorted({i for i in ids if ids.count(i) > 1})
        if twice:
            problems.append(f"steps: two steps share the id {twice[0]!r}; each step needs its own")
        for k, s in enumerate(self.steps):
            if k == 0 and s.max_wait_bars is not None:
                problems.append(f"steps[0] '{s.id}': the first step STARTS the idea - there is nothing before it to "
                                "wait from, so it takes no max_wait_bars")
            if k > 0 and s.max_wait_bars is None:
                problems.append(f"steps[{k}] '{s.id}': say its max_wait_bars - how many {s.tf} bars after step "
                                f"'{self.steps[k - 1].id}' the idea waits before it is dropped: a number, \"no "
                                "limit\", or 0 for a check made at that moment")
        if not self.targets:
            problems.append("targets: give at least one (a ratio, a price, or the next liquidity)")
        for where, ref in self._refs():
            m = REF.fullmatch(ref)
            if not m:
                problems.append(f"{where}: {ref!r} must name a step and what it found, like 'gap.near'")
            elif m.group(1) not in ids:
                problems.append(f"{where}: {ref!r} names no step; the steps are {', '.join(ids)}")
        if len(self.slices) > config.SLICE_BUDGET:
            problems.append(f"slices: {len(self.slices)} given; at most {config.SLICE_BUDGET} slices per idea, chosen "
                            "before running (likhith.md section D) - the rest are only 'looked at, not tested'")
        names = [s.name for s in self.slices]
        if len(set(names)) != len(names):
            problems.append("slices: two slices share a name")
        u = [x.id for x in self.unclear]
        if len(set(u)) != len(u):
            problems.append("unclear: two unclear parts share an id")
        if problems:
            raise ValueError("\n".join(problems))
        return self

    def _refs(self):
        if self.entry.at:
            yield "entry.at", self.entry.at
        yield "exit.at", self.exit.at
        for k, t in enumerate(self.targets):
            if t.at:
                yield f"targets[{k}].at", t.at

    # ---- loading and saving ---------------------------------------------------------------------------------------

    @classmethod
    def load(cls, obj: "dict | str | Path") -> "Idea":
        """A form from a dict, a JSON text or a .json file. A bad form raises FormError, one readable line each."""
        if isinstance(obj, Path) or (isinstance(obj, str) and obj.strip().endswith(".json")
                                     and not obj.strip().startswith("{")):
            obj = Path(obj).read_text(encoding="utf-8")
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except ValueError as err:
                raise FormError(f"the form is not JSON: {err}") from None
        try:
            return cls.model_validate(obj)
        except ValidationError as err:
            raise FormError(readable(err)) from None

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False) + "\n"

    # ---- the rule, its fingerprint --------------------------------------------------------------------------------

    def rule(self) -> dict:
        """The parts that make the RULE - what decides which trades exist. Words, sources, notes, status, slices and
        the unclear parts (once applied) are not part of it: changing them never makes a new version."""
        d = self.model_dump(mode="json")
        drop_src = lambda x: {k: v for k, v in x.items() if k not in ("sources", "what", "words")}  # noqa: E731
        return {
            "timeframe": d["timeframe"], "directions": d["directions"],
            "steps": [drop_src(s) for s in d["steps"]], "entry": drop_src(d["entry"]), "exit": drop_src(d["exit"]),
            "targets": [drop_src(t) for t in d["targets"]],
            "do_not_enter": [drop_src(r) for r in d["do_not_enter"]], "filters": [drop_src(r) for r in d["filters"]],
            "settings": d["settings"],
        }

    def fingerprint(self) -> str:
        """The rule's hash: the same rule always gives the same version id; any change to the rule is a new try."""
        text = json.dumps(self.rule(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ================================================================================================= readable errors

_PLAIN = {
    "missing": "is missing",
    "extra_forbidden": "is not a field of the form here",
    "literal_error": "is not one of the allowed values",
    "string_type": "must be text",
    "int_type": "must be a whole number",
    "int_from_float": "must be a whole number",
    "float_type": "must be a number",
    "bool_type": "must be true or false",
    "tuple_type": "must be a list",
    "dict_type": "must be a mapping of names to values",
    "union_tag_invalid": "must say kind: \"stop\" or \"hedge\"",
    "union_tag_not_found": "must say kind: \"stop\" or \"hedge\"",
}


def readable(err: ValidationError) -> str:
    """pydantic's errors as plain lines: 'steps[1] (break) > max_wait_bars: ...'."""
    lines = []
    for e in err.errors(include_url=False):
        loc = _where(e["loc"], e.get("input"))
        msg = e["msg"]
        if e["type"] == "value_error":
            msg = msg.removeprefix("Value error, ")
        elif e["type"] in _PLAIN:
            msg = _PLAIN[e["type"]]
            if e["type"] == "literal_error":
                msg += f" ({e['ctx'].get('expected', '')})"
            elif e["type"] == "extra_forbidden":
                msg += " - remove it or check its spelling"
        for part in msg.split("\n"):
            lines.append(f"{loc}: {part}" if loc else part)
    return "The idea form has problems:\n" + "\n".join(f"  - {x}" for x in lines)


def _where(loc: tuple, _input) -> str:
    out = []
    for p in loc:
        if isinstance(p, int):
            out[-1:] = [f"{out[-1]}[{p}]"] if out else [f"[{p}]"]
        elif p in ("stop", "hedge") and out and out[-1] == "exit":
            continue                      # the discriminated union's branch name adds nothing
        else:
            out.append(str(p))
    return " > ".join(out)


# ================================================================================================ changes, readings

class ChangeError(FormError):
    """A reading or a version names a part of the form that does not exist."""


def apply_changes(idea: Idea, changes: dict[str, Any], who: str = "a change") -> Idea:
    """The form with `changes` applied: {dotted path: new value}. Paths:
        steps.<id>                     the whole step (a dict), or null to remove it
        steps.<id>.<field>             block, tf, max_wait_bars, what
        steps.<id>.params.<name>       one of the block's numbers or choices
        entry.<field> · exit.<field> · exit.management.<name>
        targets (the whole list) · targets.<n>.<field>
        settings.<name> · directions · timeframe · filters · do_not_enter
    The result is checked again as a whole form."""
    d = idea.model_dump(mode="json")
    for path, value in changes.items():
        parts = str(path).split(".")
        if parts[0] == "steps" and len(parts) >= 2:
            ids = [s["id"] for s in d["steps"]]
            if parts[1] not in ids:
                raise ChangeError(f"{who}: '{path}' names no step; the steps are {', '.join(ids)}")
            k = ids.index(parts[1])
            if len(parts) == 2:
                if value is None:
                    del d["steps"][k]
                else:
                    d["steps"][k] = copy.deepcopy(value)
                continue
            _set(d["steps"][k], parts[2:], value, path, who)
        else:
            _set(d, parts, value, path, who)
    try:
        return Idea.model_validate(d)
    except ValidationError as err:
        raise FormError(f"{who} makes a bad form:\n{readable(err)}") from None


def _set(d: Any, parts: list[str], value: Any, path: str, who: str) -> None:
    for k, p in enumerate(parts):
        last = k == len(parts) - 1
        if isinstance(d, list):
            if not p.isdigit() or int(p) >= len(d):
                raise ChangeError(f"{who}: '{path}' - {p!r} is not a place in a list of {len(d)}")
            p = int(p)
        elif not isinstance(d, dict):
            raise ChangeError(f"{who}: '{path}' goes inside a value that has no parts")
        elif not last and p not in d:
            raise ChangeError(f"{who}: '{path}' - the form has no {p!r} there")
        if last:
            d[p] = copy.deepcopy(value)
        else:
            d = d[p]


def open_parts(idea: Idea) -> list[Unclear]:
    """The unclear parts not settled yet: every VERSION part, and every ASK part he has not answered."""
    return [u for u in idea.unclear if u.answer is None]


def with_readings(idea: Idea, chosen: dict[str, str] | None = None) -> tuple[Idea, dict[str, str]]:
    """The form with one reading of every unclear part applied: his answer where he gave one, else the reading
    `chosen` names. Never guesses: an open part with no choice raises FormError (R13 - never guess a meaning).
    Returns (the form, {part: reading})."""
    chosen = dict(chosen or {})
    unknown = sorted(set(chosen) - {u.id for u in idea.unclear})
    if unknown:
        raise FormError(f"readings: {unknown} are not unclear parts of {idea.id} (it has "
                        f"{[u.id for u in idea.unclear] or 'none'})")
    used, changes = {}, {}
    missing = []
    for u in idea.unclear:
        rid = u.answer if u.answer is not None else chosen.get(u.id)
        if u.answer is not None and chosen.get(u.id, u.answer) != u.answer:
            raise FormError(f"{u.id}: Tejas answered {u.answer!r}; a compile cannot choose {chosen[u.id]!r}")
        if rid is None:
            missing.append(u)
            continue
        r = {x.id: x for x in u.readings}.get(rid)
        if r is None:
            raise FormError(f"{u.id}: {rid!r} is not one of its readings {[x.id for x in u.readings]}")
        used[u.id] = rid
        for path, v in r.changes.items():
            if path in changes and changes[path] != v:
                raise FormError(f"readings {sorted(used)} both change '{path}' in different ways")
            changes[path] = v
    if missing:
        lines = [f"{u.id} ({u.kind}, {u.part}): readings {[x.id for x in u.readings]}" for u in missing]
        raise FormError(f"{idea.id} has open parts; say which reading to use (it never guesses a meaning - R13):\n"
                        + "\n".join(f"  - {x}" for x in lines))
    return (apply_changes(idea, changes, "the readings " + ", ".join(f"{k}={v}" for k, v in used.items()))
            if changes else idea), used


def reading_choices(idea: Idea) -> list[dict[str, str]]:
    """Every combination of readings of the open parts (L7b: compile every reading). One {} if none is open."""
    parts = open_parts(idea)
    return [dict(zip([u.id for u in parts], combo))
            for combo in itertools.product(*[[r.id for r in u.readings] for u in parts])]
