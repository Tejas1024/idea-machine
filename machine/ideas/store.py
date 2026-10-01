"""Each idea's saved files (L7a): plan/work/ideas/idea-XXX/

    form.json          the idea form (machine/ideas/form.py) - always written whole through a temp file, never half
    state.json         where the idea is (the L10b states), since when, what it waits for, and every earlier state
    answers.md         Tejas's answers, word for word - APPEND ONLY (CLAUDE.md rule 7: never rewrite one of his files)
    questions/         one file per question to him (the weekly batch, L9b)
    report.html        the report (L10)
    check-opened.json  written by the referee once the check years are opened for the idea (L5, machine/guards.py)
    slices.jsonl       slices chosen AFTER the form was locked, before any money was seen (L14.1) - APPEND ONLY
    versions.jsonl     the research versions proposed through the research tools (L14.1, machine/mcp) - APPEND ONLY

    from machine.ideas import store
    store.receive("idea-002", "his words", source="Telegram message 912")   # RECEIVED, no form yet (L10b intake)
    store.create(form)                         # a new idea's folder (or the form of a RECEIVED one)
    store.set_state("idea-001", "QUESTIONS QUEUED", "F3 queued", waits_for="Tejas's answer to F3")
    store.add_answer("idea-001", "A", about="F3", source="strategy/learned-inbox/...")

Idea numbers: the next free idea-NNN over plan/work/ideas/ AND plan/work/ (idea-001 also lives at plan/work/idea-001/
from T0 and T3), so a new idea never takes a number already used.
Once an idea is CONFIRMED & LOCKED, what it means can no longer change - its rule, its readings and answers, its
slices, his words (lesson 16: meaning is chosen before money is seen; lesson 18: a filter added after looking is not
a rule). save_form refuses; the change is a new idea whose parent_idea is this one.
The one way back: his NO to the three trades the machine found (SHOWN, L10b) reopens the questions - reopen() - and
only while no money has been seen for the idea (no trial file names it, the check years are not opened).
Slices follow the same line (L14.1): the rule "chosen BEFORE running" (likhith.md section D) is kept when a slice is
chosen after the lock but before any money was seen - add_slices() refuses once money_seen() names anything, and
never more than config.SLICE_BUDGET in all. form.json is not rewritten: with_chosen_slices() adds them to the form the
learner and the report read.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path

from machine import config
from machine.ideas.form import IDEA_ID, LOCKED_FROM, STATES, Idea, Slice

FORM, STATE, ANSWERS, QUESTIONS, REPORT = "form.json", "state.json", "answers.md", "questions", "report.html"
SLICES, VERSIONS = "slices.jsonl", "versions.jsonl"
_NUMBER = re.compile(r"idea-(\d{3,})")


class StoreError(ValueError):
    """The store refused: a bad idea id, an idea that exists already, a locked idea changed, an unknown state."""


def folder(idea_id: str) -> Path:
    """plan/work/ideas/<idea_id>/ (not created). Only ids like idea-001."""
    if not isinstance(idea_id, str) or not IDEA_ID.fullmatch(idea_id):
        raise StoreError(f"not an idea id: {idea_id!r} (use idea-001, idea-002 ...)")
    return Path(config.IDEAS_DIR) / idea_id


def ideas() -> list[str]:
    """Every idea with a saved form or state (an idea the intake RECEIVED has no form yet), in number order."""
    root = Path(config.IDEAS_DIR)
    found = [p.name for p in root.glob("idea-*")
             if IDEA_ID.fullmatch(p.name) and ((p / FORM).is_file() or (p / STATE).is_file())]
    return sorted(found, key=lambda x: int(x.split("-")[1]))


def next_id() -> str:
    """The next free idea number, over plan/work/ideas/ and plan/work/ (the T0 / T3 folders)."""
    used = [0]
    for root in (Path(config.IDEAS_DIR), Path(config.WORK_DIR)):
        if root.is_dir():
            used += [int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := _NUMBER.fullmatch(p.name))]
    return f"idea-{max(used) + 1:03d}"


def receive(idea_id: str, words: str, *, source: str, key: str | None = None, at: str | None = None) -> dict:
    """A new idea from his words (L10b intake): its folder, words.md (his words as typed) and state.json - RECEIVED.
    No form yet (the talker writes it later; create() then keeps this history). key: the intake queue line it came
    from, so the same line never makes two ideas. Refuses an idea that exists already."""
    d = folder(idea_id)
    if (d / STATE).exists() or (d / FORM).exists():
        raise StoreError(f"{idea_id} already exists ({_show(d)}); take store.next_id()")
    if not str(words).strip() or not str(source).strip():
        raise StoreError("a new idea needs his words and where they came from (source)")
    d.mkdir(parents=True, exist_ok=True)
    now = _now()
    with open(d / "words.md", "x", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {idea_id} — his words, as typed\n\n"
                 + "\n".join(f"> {x}" if x else ">" for x in str(words).splitlines())
                 + f"\n\nsource: {source} · received {at or now}\n")
    s = {"idea_id": idea_id, "state": "RECEIVED", "since": now, "waits_for": None,
         "intake": {"words": str(words), "source": str(source), "key": key, "at": at or now},
         "history": [{"state": "RECEIVED", "at": now, "note": f"from {source}"}]}
    _write(d / STATE, _json(s))
    return s


def create(form: Idea) -> Path:
    """A new idea's folder: form.json, state.json (its first state = the form's status), answers.md, questions/.
    Refuses an idea that already has a form. An idea the intake RECEIVED keeps its state history (the form's status
    is added to it)."""
    d = folder(form.id)
    if (d / FORM).exists():
        raise StoreError(f"{form.id} already exists ({_show(d / FORM)}); load it, or take store.next_id()")
    (d / QUESTIONS).mkdir(parents=True, exist_ok=True)
    _write(d / FORM, form.to_json())
    now = _now()
    if (d / STATE).is_file():
        s = json.loads((d / STATE).read_text(encoding="utf-8"))
        s.update(state=form.status, since=now, waits_for=None)
        s["history"].append({"state": form.status, "at": now, "note": "the form is written"})
        _write(d / STATE, _json(s))
    else:
        _write(d / STATE, _json({"idea_id": form.id, "state": form.status, "since": now, "waits_for": None,
                                 "history": [{"state": form.status, "at": now, "note": "created"}]}))
    if not (d / ANSWERS).exists():
        with open(d / ANSWERS, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# Answers - {form.id}: {form.title}\n\nTejas's answers, word for word, newest at the bottom. "
                     "Append only (never rewritten).\n")
    return d


def load_form(idea_id: str) -> Idea:
    path = folder(idea_id) / FORM
    if not path.is_file():
        raise StoreError(f"{idea_id} has no form yet ({_show(path)})")
    return Idea.load(path)


def save_form(form: Idea) -> Path:
    """Write the form (whole, through a temp file). A LOCKED idea's meaning may not change: refused."""
    d = folder(form.id)
    if not (d / FORM).is_file():
        raise StoreError(f"{form.id} does not exist yet: store.create(form) first")
    old = load_form(form.id)
    if _locked(old.status) and _meaning(old) != _meaning(form):
        raise StoreError(
            f"{form.id} is {old.status}: its meaning was fixed before any money was seen (likhith.md lessons 16 and "
            f"18), so its rule, readings, answers, slices and his words cannot change now. Make the change a new "
            f"idea: store.next_id(), with parent_idea = {form.id!r}.")
    if _locked(old.status) and not _locked(form.status):
        raise StoreError(f"{form.id} is {old.status}; it cannot go back to {form.status}")
    _write(d / FORM, form.to_json())
    return d / FORM


def read_state(idea_id: str) -> dict:
    path = folder(idea_id) / STATE
    if not path.is_file():
        raise StoreError(f"{idea_id} has no state yet ({_show(path)})")
    return json.loads(path.read_text(encoding="utf-8"))


def set_state(idea_id: str, state: str, note: str = "", waits_for: str | None = None) -> dict:
    """Move the idea to `state` (one of the L10b states): state.json keeps every earlier state; the form's status
    follows. Which moves are allowed is L10b's job; a LOCKED idea never goes back before CONFIRMED & LOCKED."""
    if state not in STATES:
        raise StoreError(f"{state!r} is not a state; one of: {', '.join(STATES)}")
    s = read_state(idea_id)
    if _locked(s["state"]) and not _locked(state):
        raise StoreError(f"{idea_id} is {s['state']}; it cannot go back to {state}")
    return _move(idea_id, s, state, note, waits_for)


def _move(idea_id: str, s: dict, state: str, note: str, waits_for: str | None) -> dict:
    now = _now()
    s.pop("needs", None)                                    # what the old state waited a person for (L10b) is done
    s.pop("talker_failures", None)
    s.update(state=state, since=now, waits_for=waits_for)
    s["history"].append({"state": state, "at": now, "note": note})
    _write(folder(idea_id) / STATE, _json(s))
    if (folder(idea_id) / FORM).is_file():                  # RECEIVED / SEARCHED ideas have no form yet
        form = load_form(idea_id)
        if form.status != state:
            _write(folder(idea_id) / FORM, form.model_copy(update={"status": state}).to_json())
    return s


def set_waits_for(idea_id: str, waits_for: str | None, note: str = "") -> dict:
    """What the idea waits for changed, the state did not (e.g. the weekly batch sent its questions). Kept in history
    only with a note."""
    s = read_state(idea_id)
    s["waits_for"] = waits_for
    if note:
        s["history"].append({"state": s["state"], "at": _now(), "note": note})
    _write(folder(idea_id) / STATE, _json(s))
    return s


REOPEN_FROM = ("CONFIRMED & LOCKED", "SHOWN")


def money_seen(idea_id: str) -> list[str]:
    """Why money counts as seen for the idea: a trial file that names it (a tune or a judgement), or its check years
    opened. Empty = nothing seen yet (only signals and pictures without money)."""
    why = []
    trials = Path(config.TRIALS_DIR)
    if trials.is_dir():
        for p in sorted(trials.glob("*.json")):
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("idea_id") == idea_id:
                    why.append(_show(p))
            except (ValueError, OSError):
                continue
    if (folder(idea_id) / "check-opened.json").exists():
        why.append(_show(folder(idea_id) / "check-opened.json"))
    return why


def reopen(idea_id: str, note: str, waits_for: str) -> dict:
    """His NO to the three trades the machine found (L10b SHOWN): the idea goes back to QUESTIONS QUEUED, its meaning
    open again. Only from CONFIRMED & LOCKED or SHOWN, and only while no money has been seen for it (lesson 16: the
    meaning is chosen before money is seen - after that, a change is a new idea)."""
    s = read_state(idea_id)
    if s["state"] not in REOPEN_FROM:
        raise StoreError(f"{idea_id} is {s['state']}; only an idea {' or '.join(REOPEN_FROM)} can be reopened")
    seen = money_seen(idea_id)
    if seen:
        raise StoreError(f"{idea_id}: money was seen already ({', '.join(seen[:3])}) - its meaning cannot change now "
                         f"(lesson 16). Make the change a new idea: store.next_id(), with parent_idea = {idea_id!r}.")
    return _move(idea_id, s, "QUESTIONS QUEUED", note, waits_for)


# ---- slices chosen after the lock, before any money (L14.1) -----------------------------------------------------

def chosen_slices(idea_id: str) -> list[Slice]:
    """The slices added by add_slices(), oldest first (slices.jsonl; none = [])."""
    path = folder(idea_id) / SLICES
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out += [Slice.model_validate(s) for s in json.loads(line)["slices"]]
    return out


def with_chosen_slices(form: Idea) -> Idea:
    """The form with its chosen slices added after its own (what the learner tests and the report shows). The same
    form when none were chosen. Checked again as a whole form: at most config.SLICE_BUDGET, no name twice."""
    extra = chosen_slices(form.id)
    if not extra:
        return form
    d = form.model_dump(mode="json")
    d["slices"] = [*d["slices"], *(s.model_dump(mode="json") for s in extra)]
    return Idea.model_validate(d)


def add_slices(idea_id: str, slices: list[dict], *, who: str, why: str) -> list[Slice]:
    """Choose slices for an idea BEFORE its first result (likhith.md section D). Refused once money was seen for it
    (money_seen: a trial file names it, or its check years are open), past config.SLICE_BUDGET slices in all (the
    form's own + every earlier choice), for a name used twice, or for a dimension that makes a VERSION, not a slice
    (form.Slice checks it). Appends ONE line to slices.jsonl - never rewritten. Returns every slice the idea has now."""
    if not str(who).strip() or not str(why).strip():
        raise StoreError("choosing slices needs who and why (both go into slices.jsonl)")
    form = load_form(idea_id)
    seen = money_seen(idea_id)
    if seen:
        raise StoreError(f"{idea_id}: money was already seen ({', '.join(seen[:3])}) - slices are chosen BEFORE "
                         "running (likhith.md section D; lesson 18: a split chosen after looking is not a rule). Every "
                         "other split is 'looked at, not tested'.")
    if not slices:
        raise StoreError("no slices given")
    new = [Slice.model_validate(s) for s in slices]
    have = [*form.slices, *chosen_slices(idea_id)]
    if len(have) + len(new) > config.SLICE_BUDGET:
        raise StoreError(f"{idea_id} has {len(have)} slices; {len(new)} more would make {len(have) + len(new)} - at "
                         f"most {config.SLICE_BUDGET} slices per idea, chosen before running (config.SLICE_BUDGET, "
                         "likhith.md section D)")
    names = [s.name for s in [*have, *new]]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise StoreError(f"{idea_id}: the slice {twice[0]!r} is chosen twice")
    line = {"at": _now(), "who": str(who), "why": str(why), "slices": [s.model_dump(mode="json") for s in new]}
    path = folder(idea_id) / SLICES
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return [*have, *new]


def add_answer(idea_id: str, words: str, *, about: str, source: str, who: str = "Tejas") -> Path:
    """Append one answer of his, word for word (never rewrites the file). about: the question or unclear part
    ("F3", "L9b-2026-10-04-Q2"); source: where his words are (the inbox file, the Telegram message)."""
    if not str(words).strip() or not str(about).strip() or not str(source).strip():
        raise StoreError("an answer needs his words, what it is about, and where his words are (source)")
    path = folder(idea_id) / ANSWERS
    if not path.is_file():
        raise StoreError(f"{idea_id} has no {ANSWERS} ({_show(path)}): store.create(form) first")
    quoted = "\n".join(f"> {line}" if line else ">" for line in str(words).splitlines())
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"\n## {_now()} · {who} · {about}\n\n{quoted}\n\nsource: {source}\n")
    return path


def questions_dir(idea_id: str) -> Path:
    return folder(idea_id) / QUESTIONS


def report_path(idea_id: str) -> Path:
    return folder(idea_id) / REPORT


# ---- helpers ------------------------------------------------------------------------------------------------------

def _locked(state: str) -> bool:
    return STATES.index(state) >= STATES.index(LOCKED_FROM)


def _meaning(form: Idea) -> str:
    d = form.model_dump(mode="json", exclude={"status", "notes", "sources", "title"})
    return json.dumps(d, sort_keys=True, default=str)


def _write(path: Path, text: str) -> None:
    """Write a whole file through a temp file in the same folder, then move it into place (rule 7: never half)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _show(path: Path) -> str:
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
