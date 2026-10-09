"""Research versions (L14.1): what a version may change, which family it counts in, and the budget - plan/likhith.md
section D as code. The research tools (machine/mcp/tools.py) call these; nothing here reads a price.

A VERSION is a different rule. Only two kinds of change make one (section D; plan/README.md section 5):
  dimension 1   the timeframe: "timeframe" (the idea's chart) and "steps.<id>.tf" (one step's chart)
  dimension 10  the idea's own numbers - its level, stop, target and wait:
                  "steps.<id>.max_wait_bars", "entry.max_wait_bars"          a whole number, or "no limit"
                  "steps.<id>.params.<name>", "do_not_enter.<n>.params.<name>"  a number
                  "exit.beyond_usd"                                            dollars past the stop / hedge point
                  "targets.<n>.ratio"                                          the target, times the stop distance
                  "settings.<name>"                                            a NUMBER the machine chose (settings.py
                                                                               status "test") - never one of his
                  "exit.management.<name>"                                     his management's TEST rows (T12, L19c:
                                                                               how the hedge and the target are looked
                                                                               after) - a hedge only; the values and the
                                                                               reasons: machine/tester/management.py
and the readings of the idea's VERSION parts (unclear parts the machine tries; an ASK part is his answer, never a
choice). Everything else is refused, and the refusal says why:
  - his fixed rules (machine/rules.py), under any name or old alias, anywhere in the path: his words (LEARNED #121);
  - a setting that is his rule (status "his"): his rule, not a number to try;
  - directions, filters, slices, adding or removing a do-not-enter rule: a SLICE or a new rule, not a version
    (section D; lesson 18: a filter added after looking at results is not a rule) - slices are chosen BEFORE running
    (the tool choose_slices, at most 8);
  - anything else (a step's block or its kinds of level, entry.at, exit.at, parent_idea, his words ...): what the idea
    IS - that change is a new idea whose parent_idea is this one.

THE FAMILY (section D: "an idea with the same blocks, timeframe and entry as an older idea is a version OF that older
idea"): the form's parent_idea; else the oldest OLDER idea - one of his ideas (plan/work/ideas/) or one of his entry
types (L19b, machine/entries/forms.py) - with the same blocks, timeframes and entry (machine.referee.prior's "already
seen" key, the management left out). Older = seen first: an idea's RECEIVED time, an entry type's first trial file. A
version whose own key is an older idea of ANOTHER family is refused: it is a version of that idea.

THE BUDGET: at most config.VERSION_BUDGET versions per family - every version with a trial file (the referee's count,
machine/referee/trials.py) plus every version proposed here and not run yet. The same rule is the same version:
proposing it again returns the first record; the same rule under another idea of the family is refused (one rule, one
try).

Records: plan/work/ideas/<idea>/versions.jsonl, one JSON line per version, only ever added to (config.BOT_APPEND_ONLY).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import typing
from functools import lru_cache
from pathlib import Path
from typing import Any

from machine import config, rules, settings
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import LOCKED_FROM, NO_LIMIT, STATES, TF, FormError, Idea, with_readings
from machine.referee import prior, trials
from machine.tester import management as MG

TIMEFRAMES = typing.get_args(TF)
SECTION_D = "plan/likhith.md section D"


class Refused(ValueError):
    """A research tool said no. The message says which rule, and why, in plain words."""


# ================================================================================================= one change

def check_change(form: Idea, path: str, value: Any) -> str:
    """What one change of a version is ("the timeframe (dimension 1)", "a wait (dimension 10)" ...), or Refused.
    form: the idea with the version's readings applied (a reading may change its steps)."""
    if not isinstance(path, str) or not path.strip():
        raise Refused(f"a change needs a dotted path like 'steps.gap.max_wait_bars', not {path!r}")
    parts = path.split(".")
    for k, p in enumerate(parts):                        # his fixed rules first, anywhere, under any name (#121) -
        if parts[0] == "steps" and k == 1:               # but a step's own id is only a name ("hammer" is one of
            continue                                     # his rules' old names, and a fine step id)
        if rules.fixed_name(p) is not None:
            raise Refused(f"'{path}': {rules.refusal(p, value)}")
    head, n = parts[0], len(parts)
    step_ids = [s.id for s in form.steps]
    if path == "timeframe" or (head == "steps" and n == 3 and parts[2] == "tf"):
        _step(parts, step_ids, path)
        if value not in TIMEFRAMES:
            raise Refused(f"'{path}' = {value!r}: a timeframe is one of {', '.join(TIMEFRAMES)}")
        return "the timeframe (dimension 1)"
    if head == "steps" and n == 3 and parts[2] == "max_wait_bars":
        _step(parts, step_ids, path)
        if parts[1] == step_ids[0]:
            raise Refused(f"'{path}': the first step starts the idea - it has no wait")
        _wait(value, path, least=0)
        return "a wait (dimension 10)"
    if path == "entry.max_wait_bars":
        if form.entry.how != "limit":
            raise Refused(f"'{path}': this idea enters at the next bar's open - it has no wait")
        _wait(value, path, least=1)
        return "a wait (dimension 10)"
    if head == "steps" and n == 4 and parts[2] == "params":
        _step(parts, step_ids, path)
        if parts[3] == "kinds":
            raise Refused(f"'{path}': which kinds of level a step takes is what the idea IS (its blocks, {SECTION_D}) "
                          "- a new idea, not a version")
        _number(value, path)
        return "the idea's own number (dimension 10)"
    if head == "do_not_enter" and n == 4 and parts[2] == "params":
        if not parts[1].isdigit() or int(parts[1]) >= len(form.do_not_enter):
            raise Refused(f"'{path}': the idea has {len(form.do_not_enter)} do-not-enter rules (0 to "
                          f"{len(form.do_not_enter) - 1})")
        _number(value, path)
        return "the idea's own number (dimension 10)"
    if path == "exit.beyond_usd":
        _number(value, path, least=0.0)
        return "the stop (dimension 10)"
    if head == "targets" and n == 3 and parts[2] == "ratio":
        if not parts[1].isdigit() or int(parts[1]) >= len(form.targets) or form.targets[int(parts[1])].kind != "ratio":
            raise Refused(f"'{path}': only a target that is a ratio of the stop distance has a ratio to change")
        _number(value, path, above=0.0)
        return "the target (dimension 10)"
    if MG.name_of(path) is not None or (head == "settings" and n == 2 and parts[1] in MG.VERSIONS):
        try:                                             # his management (T12): his test rows only (L19c)
            return MG.check_value(form, MG.name_of(path) or parts[1], value, path)
        except MG.Refused as err:
            raise Refused(str(err)) from None
    if head == "settings" and n == 2:
        s = settings.ALL.get(parts[1])
        if s is None:
            raise Refused(f"'{path}': {parts[1]!r} is not a setting (machine/settings.py; use its own name)")
        if parts[1] in MG.RETIRED:
            raise Refused(f"'{path}': no code reads {parts[1]} any more - {MG.RETIRED[parts[1]]}. A version changing "
                          "it would give the same trades.")
        if s.status == "his":
            got = rules.managed_by(parts[1])
            row = f" - his FIXED management rule {got[0]}: '{got[1].rule}' (machine/rules.py MANAGEMENT)" if got else ""
            raise Refused(f"'{path}': {parts[1]} is HIS rule ({s.who}: {s.value!r}){row} - not a number to try. A "
                          "version changes the timeframe or the idea's own numbers only.")
        _number(value, path)
        if not is_number(s.value):
            raise Refused(f"'{path}': {parts[1]} is not a number ({s.value!r}) - a version changes numbers only")
        return "the machine's number (dimension 10)"
    if head == "directions":
        raise Refused(f"'{path}': buy or sell is a SLICE (dimension 2), not a version ({SECTION_D}). A slice is chosen "
                      "before running (choose_slices); the idea's trades are then split by it.")
    if head == "filters":
        raise Refused(f"'{path}': a filter keeps only some setups - a SLICE (session, news, trend ...), not a version "
                      f"({SECTION_D}; lesson 18: a filter added after looking at results is not a rule)")
    if head == "slices":
        raise Refused(f"'{path}': slices are not versions; choose them BEFORE running with choose_slices (at most "
                      f"{config.SLICE_BUDGET})")
    if head == "do_not_enter":
        raise Refused(f"'{path}': adding or removing a do-not-enter rule changes what the idea IS (lesson 18) - only "
                      "the numbers of its own rules make versions ('do_not_enter.<n>.params.<name>')")
    raise Refused(f"'{path}': a version changes only the timeframe (dimension 1) or the idea's own numbers - its "
                  f"level, stop, target, wait (dimension 10) - {SECTION_D}. '{path}' is what the idea IS: a new idea "
                  "whose parent_idea is this one.")


def _step(parts: list[str], ids: list[str], path: str) -> None:
    if parts[0] == "steps" and parts[1] not in ids:
        raise Refused(f"'{path}': the idea has no step {parts[1]!r}; its steps are {', '.join(ids)}")


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _number(v: Any, path: str, least: float | None = None, above: float | None = None) -> None:
    if not is_number(v):
        raise Refused(f"'{path}' = {v!r}: a version changes numbers here - give a number")
    if least is not None and v < least:
        raise Refused(f"'{path}' = {v!r}: it cannot be below {least:g}")
    if above is not None and v <= above:
        raise Refused(f"'{path}' = {v!r}: it must be above {above:g}")


def _wait(v: Any, path: str, least: int) -> None:
    if v == NO_LIMIT:
        return
    if not isinstance(v, int) or isinstance(v, bool) or v < least:
        raise Refused(f"'{path}' = {v!r}: a wait is a whole number of bars (at least {least}) or \"{NO_LIMIT}\"")


# ================================================================================================= the family

def idea_form(idea: str) -> Idea:
    """One of his ideas (plan/work/ideas/idea-NNN) with its form. Anything else is refused - the research tools work
    only on his ideas (never on a path, a belief or a test idea)."""
    try:
        store.folder(idea)
        return store.load_form(idea)
    except store.StoreError as err:
        raise Refused(str(err)) from None


def locked(idea: str) -> Idea:
    """The idea's form, only once its meaning is locked (lesson 16: meaning is chosen before money is seen)."""
    form = idea_form(idea)
    state = store.read_state(idea)["state"]
    if STATES.index(state) < STATES.index(LOCKED_FROM):
        raise Refused(f"{idea} is {state}: its meaning is not locked yet (every question answered, his YES) - versions "
                      "are tried only on a locked meaning (lesson 16: meaning is chosen before money is seen)")
    return form


def _key3(form: Idea) -> tuple:
    k = prior.key_for_form(form)
    return (k.blocks, k.timeframes, k.entry)


@lru_cache(maxsize=1)
def _entry_types() -> tuple[tuple[str, tuple], ...]:
    from machine.entries import forms
    return tuple((e.id, _key3(e.idea())) for e in forms.ENTRY_TYPES.values())


def _born(idea: str, files: list[dict]) -> str:
    """When an idea was first seen: an idea's first state (RECEIVED), an entry type's first trial file."""
    if idea.startswith("idea-"):
        try:
            return store.read_state(idea)["history"][0]["at"]
        except (store.StoreError, KeyError, IndexError):
            return "9999"
    made = [str(f.get("made_at", "")) for f in files if f.get("idea_id") == idea]
    return min(made) if made else "9999"


def _first_trial(idea: str, files: list[dict]) -> str:
    """The idea's first trial id: its clock has microseconds, so two ideas first tried in the same second keep their
    real order (L19b tried his six entry types within 2 seconds - Type 1 and "Type 1 with the second gap" in the same
    one). '~' = never tried (after every id)."""
    ids = [str(f.get("trial_id", "")) for f in files if f.get("idea_id") == idea]
    return min(ids) if ids else "~"


def _utc(stamp: str) -> str:
    """A time as UTC text, so 'older' compares times, not clocks ('9999' = never seen)."""
    try:
        t = dt.datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    if t.tzinfo is None:
        t = t.astimezone()
    return t.astimezone(dt.timezone.utc).isoformat()


def same_as_older(idea: str, form: Idea, files: list[dict] | None = None) -> tuple[str, str] | None:
    """(the older idea with the same blocks, timeframe and entry as this form, when it was seen) - or None."""
    files = trials.load() if files is None else files
    mine = _utc(_born(idea, files))
    key = _key3(form)
    found = []
    for other in store.ideas():
        if other == idea:
            continue
        try:
            k = _key3(store.load_form(other))
        except (store.StoreError, ValueError):
            continue                                     # no form yet / a block the old list has no name for
        if k == key:
            found.append((_utc(_born(other, files)), _first_trial(other, files), other))
    for other, k in _entry_types():
        if other != idea and k == key:
            found.append((_utc(_born(other, files)), _first_trial(other, files), other))
    older = sorted(x for x in found if x[0] < mine)       # the same second: the first trial decides, never the name
    return (older[0][2], older[0][0]) if older else None


def parent_of(idea: str, form: Idea | None = None, files: list[dict] | None = None) -> tuple[str | None, str]:
    """(the idea's parent, why): the form's parent_idea, else an older idea it is the same as (section D)."""
    form = form or idea_form(idea)
    if form.parent_idea:
        return form.parent_idea, f"its form names parent_idea {form.parent_idea}"
    same = same_as_older(idea, form, files)
    if same:
        return same[0], (f"the same blocks, timeframe and entry as {same[0]}, seen earlier ({same[1][:16]}): a version "
                         f"OF {same[0]} ({SECTION_D})")
    return None, "a new idea: no older idea has its blocks, timeframe and entry"


# ================================================================================================= the records

def records(idea: str) -> list[dict]:
    """Every version proposed for this idea, oldest first (versions.jsonl)."""
    path = store.folder(idea) / store.VERSIONS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(idea: str, version: str) -> dict:
    for r in records(idea):
        if r["version_id"] == version:
            return r
    raise Refused(f"{idea} has no version {version!r} - propose it first (propose_version); its versions: "
                  f"{', '.join(r['version_id'] for r in records(idea)) or 'none yet'}")


def all_records() -> list[dict]:
    out = []
    for idea in store.ideas():
        out += records(idea)
    return out


def links(files: list[dict], recs: list[dict]) -> dict[str, str]:
    """idea -> parent, from the trial files and the version records (two different parents: refused)."""
    out = trials.parents(files)
    for r in recs:
        i, p = r["idea_id"], r.get("parent_idea")
        if p and out.get(i, p) != p:
            raise Refused(f"{i} names two parents: {out[i]} (plan/work/trials/) and {p} (its versions.jsonl)")
        if p:
            out[i] = p
    return out


def family(idea: str, parent: str | None, files: list[dict] | None = None, recs: list[dict] | None = None) -> dict:
    """The family's root, its versions with a trial file ("tried"), its versions proposed and not run ("waiting"),
    the budget and what is left."""
    files = trials.load() if files is None else files
    recs = all_records() if recs is None else recs
    ln = links(files, recs)
    try:
        root = trials.root_of(idea, ln, parent)
    except ValueError as err:                            # two parents, or a loop: refused, never guessed
        raise Refused(str(err)) from None
    ln = {**ln, idea: parent} if parent else ln

    def fam(i: str) -> str:
        return trials.root_of(i, ln)

    tried: list[tuple[str, str]] = []
    for f in files:
        pair = (f["idea_id"], f["version_id"])
        if pair not in tried and fam(pair[0]) == root:
            tried.append(pair)
    waiting = []
    for r in recs:
        pair = (r["idea_id"], r["version_id"])
        if pair not in tried and pair not in waiting and fam(pair[0]) == root:
            waiting.append(pair)
    ideas = sorted({i for i, _ in tried + waiting} | {idea})
    used = len(tried) + len(waiting)
    return {"root": root, "ideas": ideas, "tried": tried, "waiting": waiting, "used": used,
            "budget": config.VERSION_BUDGET, "left": config.VERSION_BUDGET - used}


def about(form: Idea, readings: dict[str, str], changes: dict[str, Any]) -> str:
    """What a version is, in words: each reading's meaning, then each change with the value it replaces."""
    by_id = {u.id: u for u in form.unclear}
    parts = []
    for part, rid in readings.items():
        u = by_id.get(part)
        r = next((x for x in u.readings if x.id == rid), None) if u else None
        if r is not None and u.answer is None:
            parts.append(r.means)
    base, _ = with_readings(form, readings)
    d = base.model_dump(mode="json")
    for path, v in changes.items():
        name = MG.name_of(path)
        if name is not None and name in settings.ALL and _get(d, path) is None:   # his management: version 1's value
            parts.append(f"{path} = {v!r} (his management, T12 {MG.VERSIONS[name].row}: version 1 is "
                         f"{settings.ALL[name].value!r})" if name in MG.VERSIONS else f"{path} = {v!r}")
        else:
            parts.append(f"{path} = {v!r} (was {_get(d, path)!r})")
    return " · ".join(parts) or "the idea as written"


def _get(d: Any, path: str) -> Any:
    for p in path.split("."):
        if isinstance(d, list):
            if isinstance(d and d[0], dict) and "id" in d[0] and not p.isdigit():
                d = next((x for x in d if x.get("id") == p), None)
            else:
                d = d[int(p)] if p.isdigit() and int(p) < len(d) else None
        elif isinstance(d, dict):
            d = d.get(p)
        else:
            return None
    return d


def propose(idea: str, *, readings: dict[str, str] | None = None, changes: dict[str, Any] | None = None,
            why: str, who: str) -> tuple[dict, bool]:
    """Propose one version (see the module text). Returns (its record, True if new). Nothing is run and no money is
    seen here; the try is counted from now on (the budget holds proposed versions too)."""
    if not str(why).strip():
        raise Refused("a version needs its reason (why): it goes into the research log")
    c = check_version(idea, readings=readings, changes=changes)
    if c["proposed_before"]:
        return next(r for r in records(idea) if r["version_id"] == c["version_id"]), False
    if c["family_left"] <= 0:
        raise Refused(f"the {c['family']} family has {c['family_used']} versions ({c['family_tried']} tried, "
                      f"{c['family_used'] - c['family_tried']} proposed and not run) - the budget is "
                      f"{config.VERSION_BUDGET} per idea (config.VERSION_BUDGET, section D). A new version is not tried.")
    rec = {
        "idea_id": idea, "version_id": c["version_id"], "parent_idea": c["parent_idea"], "family": c["family"],
        "parent_why": c["parent_why"], "readings": c["readings"], "changes": c["changes"],
        "dimensions": c["dimensions"], "about": c["about"], "why": str(why), "key": c["key"],
        "timeframe": c["timeframe"], "proposed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "who": str(who), "budget_after": {"used": c["family_used"] + 1, "budget": config.VERSION_BUDGET},
    }
    path = store.folder(idea) / store.VERSIONS
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec, True


def check_version(idea: str, *, readings: dict[str, str] | None = None, changes: dict[str, Any] | None = None,
                  files: list[dict] | None = None, recs: list[dict] | None = None) -> dict:
    """Every check propose() makes, WITHOUT writing or counting anything: the version's record as it would be, plus
    "proposed_before" (the same rule is already a version of this idea: propose() returns that record) and the
    family's count before it ("family_used", "family_tried", "family_left"). The research agent (L14.2) checks its
    whole plan with it before the plan is saved. Refused with the same words as propose()."""
    readings = {str(k): str(v) for k, v in dict(readings or {}).items()}
    changes = dict(changes or {})
    form = locked(idea)
    try:
        base, used = with_readings(form, readings)
    except FormError as err:
        raise Refused(str(err)) from None
    asked = [u.id for u in form.unclear if u.kind == "ask" and u.id in readings and u.answer is None]
    if asked:
        raise Refused(f"{', '.join(asked)}: an ASK part is his to answer, never a version's choice (R13)")
    dims = {path: check_change(base, path, value) for path, value in changes.items()}
    try:
        MG.check_needs(base, changes)                    # a management number whose switch is off: the same trades
    except MG.Refused as err:
        raise Refused(str(err)) from None
    try:
        f, used, changes, _ = C.prepare(form, readings, changes)
        missing = C.check_without_bars(f)
    except (rules.FixedRuleError, settings.SettingError, FormError, C.CompileError) as err:
        raise Refused(str(err)) from None
    if missing:
        lines = "; ".join(f"'{n}' ({w}) - {p}" for n, w, p in missing)
        raise Refused(f"this version needs what is not built yet: {lines}")
    vid = "v-" + f.fingerprint()[:10]
    out = {"version_id": vid, "readings": used, "changes": changes, "dimensions": dims,
           "about": about(form, used, changes), "key": prior.key_for_form(f).as_dict(), "timeframe": f.timeframe}
    if any(r["version_id"] == vid for r in records(idea)):
        return {**out, "proposed_before": True}          # the same rule: the same version, not a new try
    files = trials.load() if files is None else files
    parent, parent_why = parent_of(idea, form, files)
    recs = all_records() if recs is None else recs
    fam = family(idea, parent, files, recs)
    elsewhere = [i for i, v in fam["tried"] + fam["waiting"] if v == vid and i != idea]
    if elsewhere:
        raise Refused(f"this exact rule is already {vid} of {elsewhere[0]} in the same family - one rule is one try "
                      "(list_trials shows its result)")
    same = same_as_older(idea, f, files)
    if same and same[0] not in fam["ideas"] and same[0] != parent:
        raise Refused(f"with these changes the idea has the same blocks, timeframe and entry as {same[0]}, seen "
                      f"earlier - it is a version OF {same[0]} ({SECTION_D}), and its try belongs to that family")
    return {**out, "proposed_before": False, "parent_idea": parent, "family": fam["root"], "parent_why": parent_why,
            "family_used": fam["used"], "family_tried": len(fam["tried"]), "family_left": fam["left"]}


def tuned_versions(idea: str) -> dict[str, dict]:
    """Versions of the idea from its referee tune files (L10: the readings of every version it tuned) - so a research
    report can put them next to the proposed ones. version -> {"readings", "changes", "about"}."""
    from machine.referee import tune
    out: dict[str, dict] = {}
    for f in trials.load():
        if f.get("kind") == tune.TUNE_KIND and f.get("idea_id") == idea:
            for vid, row in f["result"]["versions"].items():
                out.setdefault(vid, {"readings": row.get("readings", {}), "changes": row.get("changes", {}),
                                     "about": row.get("about", "")})
    return out


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
