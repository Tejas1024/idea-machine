"""The intake (L10b): one idea runs from his words to the report by itself. Only his answers are manual.

    .venv\\Scripts\\python -m machine.intake "sell when the asian high is taken ..."   queue an idea, then run
    .venv\\Scripts\\python -m machine.intake run                                     run what waits (the bot does this)
    .venv\\Scripts\\python -m machine.intake status                                  one line per idea
    .venv\\Scripts\\python -m machine.intake retry idea-002                          a session fixed it: try again

From Telegram: "IDEA ..." -> the bot (L9a) files it in plan/work/intake-queue.jsonl and starts `run` (a separate
program: a long test never stops the bot from reading his messages). One tracked state.json per idea
(plan/work/ideas/idea-XXX/, machine/ideas/store.py) - the states, in order:

  RECEIVED            his words saved (words.md), a number given
  SEARCHED            the #129 search of his documents for every term (search.json)
  FORM DRAFTED        the talker (L8) wrote the form; code checked it           | BLOCKED: a block is not built yet
  QUESTIONS QUEUED    the ASK parts' questions made (L9b); the weekly batch sends them
  ANSWERED            nothing left to ask (every part answered, or a VERSION part the machine tries)
  CONFIRMED & LOCKED  his YES locked every answer; the meaning can no longer change (lesson 16)
  SHOWN               ONE picture of 3 setups the machine found: "is this your idea?" (1 question of the batch)
  TUNE RUN            his YES -> every version on 2016-2020, one chosen (L10)
  CHECK OPENED        the check years 2021-2023 opened once for the family (a family that used them already is
                      judged by walk-forward inside 2016-2020 instead)
  REPORTED            the report sent (L10)
  DECIDED             his decision (T6) - a person's step

One idea runs at a time: the oldest idea that is not BLOCKED and not REPORTED / DECIDED. The next line of the queue
starts only when none is running. A BLOCKED idea steps aside; blocked.built() sends it back to FORM DRAFTED and it
runs again (L7b). A step never repeats: each one first looks for what it leaves behind (search.json, form.json, the
question records, shown.json, the tune file, the judgement, the report line) - so after a crash, `run` goes on
from where it stopped. When a step needs a person (a Claude session: the talker made no form, his documents answer a
question, his NO to the 3 setups ...) the idea waits with "needs" in its state.json; STATUS says why and
`retry` clears it.

After 10 answers in a row that are not "works" (machine/referee/project.py, README section 7), ONE line to Tejas:
keep going, change direction, or stop - his decision. New ideas wait until he answers.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from machine import config
from machine.ideas import blocked, store
from machine.ideas.form import LOCKED_FROM, STATES, Idea, open_parts

FINISHED = ("REPORTED", "DECIDED")
SESSION = "a Claude session"
BATCH_EVERY = dt.timedelta(days=7)          # the weekly batch (CLAUDE.md rule 16); the budget is config.QUESTION_BUDGET
TALKER_TRIES = 3                            # a talker call that got NO answer (his plan's limit, no network) is tried
                                            # again at the next run; after 3, a session looks
KEEP = re.compile(r"^\s*(keep going|keep|go on|continue|carry on|yes)\s*[.!]*\s*$", re.IGNORECASE)
STOP = re.compile(r"^\s*stop\s*[.!]*\s*$", re.IGNORECASE)
SHOWN_QID = re.compile(r"^(idea-\d{3,})-shown-(\d+)$")
DIRECTION_QID = re.compile(r"^project-direction-\d+$")
NEEDS_SESSION = ("needs a session", "answers differ", "confirm said no", "in his documents", "found", "not drawable")


# ================================================================================================= where things are

def queue_path() -> Path:
    """New ideas, one JSON line each: the bot appends ("IDEA ..."), `python -m machine.intake "<words>"` too. Never
    rewritten: an idea's state.json remembers the line it came from."""
    return Path(config.WORK_DIR) / "intake-queue.jsonl"


def work_dir() -> Path:
    """The intake's tracked files: direction.json (the 10-in-a-row question), sends.jsonl (the SHOWN pictures sent)."""
    return Path(config.WORK_DIR) / "intake"


def run_dir() -> Path:
    """Not in git: the lock (one run at a time) and the log."""
    return Path(config.DATA_DIR) / "intake"


def outbox_path() -> Path:
    return Path(config.WORK_DIR) / "telegram-outbox.jsonl"


def label(idea: str) -> str:
    return f"idea #{int(idea.split('-')[1])}"


def _now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def _iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# ================================================================================================= the queue

def add(words: str, *, source: str = "the PC (python -m machine.intake)", now: dt.datetime | None = None) -> dict:
    """Queue a new idea (the same line the bot writes for "IDEA ...")."""
    words = " ".join(str(words).split())
    if not words:
        raise ValueError("an idea needs words")
    line = {"at": _iso(now or _now()), "words": words, "from": source, "state": "RECEIVED"}
    _append_jsonl(queue_path(), line)
    return line


def key_of(line: dict) -> str:
    raw = json.dumps({k: line.get(k) for k in ("at", "words", "from")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def pending() -> list[dict]:
    """Queue lines no idea was made from yet, oldest first."""
    taken = set()
    for idea in store.ideas():
        try:
            taken.add((store.read_state(idea).get("intake") or {}).get("key"))
        except store.StoreError:
            continue
    out, seen = [], set()
    for line in _read_jsonl(queue_path()):
        k = key_of(line)
        if k not in taken and k not in seen and str(line.get("words", "")).strip():
            seen.add(k)
            out.append({**line, "key": k})
    return out


def take(line: dict) -> str:
    """A queue line becomes an idea: the next free number, RECEIVED."""
    idea = store.next_id()
    store.receive(idea, line["words"], source=line.get("from") or "the intake queue", key=line["key"],
                  at=line.get("at"))
    store.set_waits_for(idea, "the machine: the search of his documents (#129)")
    return idea


# ================================================================================================= who runs now

def state_of(idea: str) -> dict:
    return store.read_state(idea)


def active(states: dict[str, dict] | None = None) -> list[str]:
    """Ideas in the machine's hands or his (not BLOCKED, not finished), oldest first."""
    states = states if states is not None else {i: state_of(i) for i in store.ideas()}
    return [i for i, s in states.items() if s["state"] not in ("BLOCKED", *FINISHED)]


def current() -> str | None:
    """The idea that runs now: the oldest active one. Every other active idea waits for its turn."""
    a = active()
    return a[0] if a else None


def needs(idea: str, why: str, *, tell: bool = True) -> None:
    """The idea waits for a person (a Claude session on his PC). One Telegram line tells Tejas, once per reason."""
    s = state_of(idea)
    if (s.get("needs") or {}).get("why") == why:
        return
    s["needs"] = {"why": why, "since": _iso(_now())}
    s["waits_for"] = f"{SESSION}: {why}"
    s["history"].append({"state": s["state"], "at": _iso(_now()), "note": f"needs {SESSION}: {why}"})
    store._write(store.folder(idea) / store.STATE, store._json(s))
    if tell:
        _tell(f"{idea}-needs-{hashlib.sha256(why.encode()).hexdigest()[:8]}",
              f"{label(idea).capitalize()} waits for a Claude session on your PC: {why}\n"
              f"When you have time, start one and paste: \"L10b: {idea} needs a session - read "
              f"plan/work/ideas/{idea}/state.json\"", idea=idea, why="L10b needs a session")


def retry(idea: str) -> dict:
    """A session did what the idea waited for: the machine tries its step again."""
    s = state_of(idea)
    if not s.get("needs"):
        return s
    s["history"].append({"state": s["state"], "at": _iso(_now()), "note": f"retry after: {s['needs']['why']}"})
    s.pop("needs")
    s["waits_for"] = "the machine"
    s.pop("talker_failures", None)
    store._write(store.folder(idea) / store.STATE, store._json(s))
    return s


def _tell(line_id: str, text: str, **extra) -> bool:
    """One line to Tejas through the bot's outbox - once per id (a crash never sends a line twice)."""
    if any(ln.get("id") == line_id for ln in _read_jsonl(outbox_path())):
        return False
    _append_jsonl(outbox_path(), {"id": line_id, "at": _iso(_now()), "to": "Tejas", "text": text, **extra})
    return True


# ================================================================================================= the tools

@dataclass
class Tools:
    """What the steps call. The real ones by default; the tests give fakes (no price data, no Claude call)."""
    search: Callable | None = None                 # q -> machine.search.rank.Answer (the talker's #129 search)
    part_search: Callable | None = None            # the same, for one unclear part (machine.questions.ask)
    translate: Callable | None = None              # (words, idea_id, search=...) -> talker Draft
    load_market: Callable | None = None            # () -> compile Market of the TUNE years
    make_part: Callable | None = None              # machine.questions.ask.make_part
    compile: Callable | None = None                # machine.ideas.compile.compile_idea
    tune: Callable | None = None                   # machine.report.run.tune_idea
    check: Callable | None = None                  # machine.report.run.check_idea
    walk_forward: Callable | None = None           # (idea, log) -> judge.Report (a family that used the check years)
    learn: Callable | None = None                  # machine.learner.run.learn (L13.1: before the report)
    build: Callable | None = None                  # machine.report.build.build
    send: Callable | None = None                   # machine.report.build.send
    batch: Callable | None = None                  # machine.questions.batch.make
    research: Callable | None = None               # machine.agent.research.run (L14.2: after his YES at SHOWN)
    log: Callable = print
    _market: Any = field(default=None, repr=False)

    def market(self):
        if self._market is None:
            if self.load_market is None:
                from machine.ideas.compile import Market
                self._market = Market.load(str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1]))
            else:
                self._market = self.load_market()
        return self._market

    def get(self, name: str):
        f = getattr(self, name)
        if f is not None:
            return f
        if name == "search":
            from machine.talker.translate import default_search
            self.search = default_search()
            return self.search
        if name == "part_search":
            from machine.search.rank import answered
            return answered
        if name == "translate":
            from machine.talker.translate import translate
            return translate
        if name == "make_part":
            from machine.questions.ask import make_part
            return make_part
        if name == "compile":
            from machine.ideas.compile import compile_idea
            return compile_idea
        if name in ("tune", "check"):
            from machine.report import run as R
            return {"tune": R.tune_idea, "check": R.check_idea}[name]
        if name == "walk_forward":
            return _walk_forward
        if name == "learn":
            from machine.learner.run import learn
            return learn
        if name in ("build", "send"):
            from machine.report import build as B
            return {"build": B.build, "send": B.send}[name]
        if name == "batch":
            from machine.questions import batch
            return batch.make
        if name == "research":
            from machine.agent import research
            return research.run
        raise KeyError(name)


# ================================================================================================= the steps

def step(idea: str, tools: Tools) -> bool:
    """Do the idea's next step if it is the machine's. True = something changed."""
    s = state_of(idea)
    if s.get("needs"):
        return False
    fn = STEPS.get(s["state"])
    return bool(fn(idea, tools)) if fn else False


def _searched(idea: str, tools: Tools) -> bool:
    """RECEIVED -> SEARCHED: every term of his words searched in his documents (#129), saved in search.json."""
    from machine.talker import glossary
    from machine.talker.translate import run_searches
    s = state_of(idea)
    words = s["intake"]["words"]
    path = store.folder(idea) / "search.json"
    if not path.is_file():
        g = glossary.apply(words)
        found = run_searches(g.text, tools.get("search"))
        rec = {"words": words, "after_glossary": g.text, "terms": [
            {"term": x.term.name, "said": x.term.said, "searched": x.term.query, "found": x.found,
             "sentence": x.sentence, "places": [h.id for h in x.hits]} for x in found]}
        store._write(path, json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    rec = json.loads(path.read_text(encoding="utf-8"))
    n, f = len(rec["terms"]), sum(t["found"] for t in rec["terms"])
    store.set_state(idea, "SEARCHED", f"#129: {n} terms searched in his documents, {f} found there",
                    waits_for="the machine: the talker writes the form (L8)")
    return True


def _drafted(idea: str, tools: Tools) -> bool:
    """SEARCHED -> FORM DRAFTED (or BLOCKED): the talker writes the form from his words and the search."""
    from machine.talker.translate import Draft
    s = state_of(idea)
    if (store.folder(idea) / store.FORM).is_file():            # written before a crash: go on from the form
        form = store.load_form(idea)
        store.set_state(idea, "FORM DRAFTED", "the form was written already", waits_for="the machine")
        return True
    d: Draft = tools.get("translate")(s["intake"]["words"], idea, search=tools.get("search"))
    _save_draft(idea, d)
    if d.answer is None or d.answer.input is None:               # no answer at all: try again at the next run
        n = int(s.get("talker_failures", 0)) + 1
        s = state_of(idea)
        s["talker_failures"] = n
        why = d.answer.error if d.answer else "no answer"
        s["waits_for"] = f"the machine: the talker got no answer ({str(why)[:160]}); it tries again at the next run"
        s["history"].append({"state": s["state"], "at": _iso(_now()), "note": f"talker: no answer ({n}): {why}"})
        store._write(store.folder(idea) / store.STATE, store._json(s))
        if n >= TALKER_TRIES:
            needs(idea, f"the talker got no answer {n} times ({str(why)[:200]})")
        return False
    if not d.valid:
        needs(idea, "the talker could not turn your words into a form that code accepts ("
              + "; ".join(d.problems)[:300] + ") - see talker-draft.json")
        return False
    store.create(d.form)                                         # FORM DRAFTED or BLOCKED (the form's status)
    if d.status == "BLOCKED":
        return _block(idea, d.form, talker=(d.wire or {}).get("missing", []))
    store.set_waits_for(idea, "the machine: its questions (L9b)",
                        f"the talker: {len(d.form.steps)} steps, {len(d.form.unclear)} unclear parts")
    return True


def _save_draft(idea: str, d) -> None:
    """What the talker answered, for the session that reads it (talker-draft.json, as idea-001's)."""
    rec = {"words": d.words, "status": d.status, "valid": d.valid, "problems": d.problems,
           "missing": [list(m) for m in d.missing], "questions": [q.__dict__ for q in d.questions],
           "unknown_citations": d.unknown_citations, "already_seen": d.already_seen,
           "sanity": [getattr(c, "__dict__", str(c)) for c in d.sanity], "kept": d.kept,
           "answer_error": None if d.answer is None else d.answer.error,
           "from_cache": None if d.answer is None else d.answer.from_cache,
           "form": None if d.form is None else json.loads(d.form.to_json())}
    store._write(store.folder(idea) / "talker-draft.json", json.dumps(rec, indent=1, ensure_ascii=False,
                                                                      default=str) + "\n")


def _block(idea: str, form: Idea, talker: list[dict] | None = None) -> bool:
    """What code finds missing, plus what the talker said is missing (its "missing" list) that is really not built ->
    BLOCKED (L7b). Nothing really missing -> the idea goes on (FORM DRAFTED): the talker's list is only its view -
    the first real run (29 Sep) had "session_high" there, which is built ("session")."""
    from machine.ideas import compile as C
    from machine.talker.translate import family_kinds
    missing = list(C.missing_in_any_reading(form))
    known = set(C.BLOCKS) | set(C.DO_NOT_ENTER) | set(C.FILTERS) | set(C.TARGET_KINDS) | {n for n, _, _ in missing}
    for m in talker or []:
        name = str(m.get("name", "")).strip()
        if name and family_kinds([name])[0] not in known:
            missing.append((name, f"the talker: {str(m.get('what', ''))[:160]}", C._planned(name)))
            known.add(name)
    if missing:
        blocked.request(form, C.MissingBlockError(form.id, missing))
        return True
    if state_of(idea)["state"] == "BLOCKED":
        said = ", ".join(str(m.get("name")) for m in talker or []) or "a block"
        store.set_state(idea, "FORM DRAFTED", f"the talker said {said} is missing; code has it built - it goes on",
                        waits_for="the machine: its questions (L9b)")
        return True
    return False


def _questions(idea: str, tools: Tools) -> bool:
    """FORM DRAFTED -> QUESTIONS QUEUED (or ANSWERED): compile-check (a missing block -> BLOCKED), then the questions
    of every open ASK part (L9b: the #129 search first, the readings on 2016-2020, the pictures)."""
    from machine.ideas.compile import missing_in_any_reading
    from machine.questions import ask
    from machine.questions import queue as Q
    form = store.load_form(idea)
    try:
        missing = missing_in_any_reading(form)                 # the form and every reading (L6b.3)
    except ValueError as err:                                  # CompileError, FixedRuleError, SettingError ...
        needs(idea, f"its form does not pass code's check now ({type(err).__name__}: {str(err)[:300]})")
        return False
    if missing:
        return _block(idea, form)
    form = _numbers_are_versions(idea, form)
    asks = [u for u in open_parts(form) if u.kind == "ask"]
    if not asks:
        store.set_state(idea, "ANSWERED", "nothing to ask: every unclear part is a VERSION the machine tries",
                        waits_for="the machine: lock the meaning")
        return True
    made = []
    for u in asks:
        rec = Q.load(idea, u.id)
        if rec is None or rec.get("status") in ("new",):
            try:
                rec = tools.get("make_part")(idea, u.id, market=tools.market(), search=tools.get("part_search"),
                                             log=tools.log)
            except ask.AskError as err:
                needs(idea, f"its question {u.id} cannot be made: {err}")
                return False
        made.append((u, rec))
    agree = [u for u, rec in made if rec.get("status") == "readings agree"]
    if agree:
        form = _agree_are_versions(idea, form, agree)
        if not [u for u in open_parts(form) if u.kind == "ask"]:
            store.set_state(idea, "ANSWERED", "nothing to ask: the readings of every question trade the same "
                            "(95% or more) - the machine tries them as versions", waits_for="the machine")
            return True
    store.set_state(idea, "QUESTIONS QUEUED", f"{sum(len(r.get('questions', [])) for _, r in made)} questions "
                    f"made for {', '.join(u.id for u, _ in made)}", waits_for=_question_wait(idea))
    return True


def _numbers_are_versions(idea: str, form: Idea) -> Idea:
    """His words, 29 Sep (inbox 2026-09-29-1155): a number (how near, how far, how big) is never asked - the machine
    tries each. An ASK part whose readings only set one number becomes a VERSION part (before any money)."""
    from machine.questions.ask import is_a_number
    parts = [u.model_copy(update={"kind": "version"}) if u.kind == "ask" and u.answer is None and is_a_number(u)
             else u for u in form.unclear]
    changed = [u.id for u, v in zip(form.unclear, parts) if u.kind != v.kind]
    if not changed:
        return form
    new = Idea.load(json.loads(form.model_copy(update={"unclear": tuple(parts)}).to_json()))
    store.save_form(new)
    store.set_waits_for(idea, "the machine", f"{', '.join(changed)}: a number - a VERSION the machine tries, never "
                                              "asked (his words, inbox 2026-09-29-1155)")
    return new


def _agree_are_versions(idea: str, form: Idea, agree) -> Idea:
    """Readings that trade the same (95% or more of the signals, L9b) are not asked (R13: only where readings trade
    differently): the part becomes a VERSION part - the machine tries each (Claude's reading)."""
    ids = {u.id for u in agree}
    parts = [u.model_copy(update={"kind": "version"}) if u.id in ids else u for u in form.unclear]
    new = Idea.load(json.loads(form.model_copy(update={"unclear": tuple(parts)}).to_json()))
    store.save_form(new)
    store.set_waits_for(idea, "the machine", f"{', '.join(sorted(ids))}: the readings trade the same - not asked, "
                                              "tried as versions")
    return new


def _question_wait(idea: str) -> str:
    from machine.questions import queue as Q
    recs = [r for r in Q.records() if r["idea"] == idea]
    if any(r.get("status") == "waiting" for r in recs):
        due = next_batch()
        return ("the weekly batch: its questions go out " + (f"on {due:%d %b} (the last batch went on "
                f"{last_batch():%d %b})" if due else "at the next run"))
    if any(r.get("status") in ("sent", "confirm sent") for r in recs):
        return "Tejas: his answers to the questions on Telegram"
    return "Tejas: his answers"


def _answered(idea: str, tools: Tools) -> bool:
    """QUESTIONS QUEUED: every ASK part answered -> ANSWERED. A part a session must read -> needs a session. Else it
    waits (the batch sends the questions; the bot files his answers and locks them - L9b)."""
    from machine.questions import queue as Q
    form = store.load_form(idea)
    open_asks = [u for u in open_parts(form) if u.kind == "ask"]
    if not open_asks:
        store.set_state(idea, "ANSWERED", "every question answered", waits_for="the machine: lock the meaning")
        return True
    recs = {u.id: Q.load(idea, u.id) for u in open_asks}
    missing = [p for p, r in recs.items() if r is None]
    if missing:                                               # a part added after the questions were made
        store.set_state(idea, "FORM DRAFTED", f"no question yet for {', '.join(missing)}", waits_for="the machine")
        return True
    stuck = [(p, r["status"]) for p, r in recs.items() if r.get("status") in NEEDS_SESSION]
    if stuck:
        needs(idea, "; ".join(f"its question {p} is '{st}' (plan/work/question-queue/{idea}-{p}/part.md)"
                              for p, st in stuck))
        return False
    wait = _question_wait(idea)
    if wait != state_of(idea).get("waits_for"):
        store.set_waits_for(idea, wait)
    return False


def _lock(idea: str, tools: Tools) -> bool:
    """ANSWERED -> CONFIRMED & LOCKED (machine.report.run.lock: refused while a question is open)."""
    from machine.report import run as R
    R.lock(idea)
    store.set_waits_for(idea, "the machine: 3 setups it found, in one picture (SHOWN)")
    return True


def _show(idea: str, tools: Tools) -> bool:
    """CONFIRMED & LOCKED: the SHOWN picture made (sent by the weekly batch - batch_step)."""
    from machine.intake import shown
    rec = shown.load(idea)
    if rec is None or rec.get("status") in ("failed", "no", "withdrawn"):  # "no" / taken back: a NEW picture
        try:
            rec = shown.make(idea, tools.market(), compile_fn=tools.get("compile"), log=tools.log)
        except shown.ShownError as err:
            needs(idea, f"the picture of 3 setups could not be made: {err}")
            return False
        due = next_batch()
        store.set_waits_for(idea, "the weekly batch: the picture of 3 setups goes out "
                            + (f"on {due:%d %b}" if due else "at the next run"), "SHOWN picture made")
        return True
    return False


def _after_shown(idea: str, tools: Tools) -> bool:
    """SHOWN: his YES -> the research agent (L14.2) - or, with config.RESEARCH_AGENT off, L10's tune (TUNE RUN). His NO
    was handled when he replied (on_reply)."""
    from machine.intake import shown
    rec = shown.load(idea) or {}
    if rec.get("status") == "yes":
        return _research(idea, tools) if config.RESEARCH_AGENT else _tune(idea, tools)
    if rec.get("status") == "needs a session":
        needs(idea, "your reply to the picture of 3 setups was not a plain YES or NO - a session reads it",
              tell=False)
    return False


def _research(idea: str, tools: Tools) -> bool:
    """His YES -> the research agent (L14.2): its plan first (Claude plans, code checks it and SAVES it before any
    money), then exactly that plan through the research tools, the referee, the report and the list of every version
    to his Telegram (REPORTED). A planning call with no answer (his usage limit) is tried again at the next run; after
    config.RESEARCH_TRIES, or anything else that stops it (an unclear meaning, a refused plan), a session looks."""
    out = tools.get("research")(idea, send=True, tell=True, say=tools.log)
    tools.log(f"{idea}: the research agent - {out.status}: {out.said}")
    if out.status == "reported":
        if state_of(idea)["state"] != "REPORTED":           # write_report(send=True) moved it; a re-run that found
            store.set_state(idea, "REPORTED", "L14.2: researched and reported",      # it reported already: the same
                            waits_for="T6: Tejas reads the report and decides")
        direction_check()
        return True
    if out.status == "no answer":
        s = state_of(idea)
        n = int(s.get("research_failures", 0)) + 1
        s["research_failures"] = n
        s["waits_for"] = f"the machine: the research plan got no answer ({out.said[:160]}); it tries again"
        s["history"].append({"state": s["state"], "at": _iso(_now()), "note": f"research: no answer ({n})"})
        store._write(store.folder(idea) / store.STATE, store._json(s))
        if n >= config.RESEARCH_TRIES:
            needs(idea, f"the research plan got no answer {n} times ({out.said[:200]})")
        return False
    if out.status == "needs a session":
        needs(idea, f"the research agent stopped: {out.said[:300]}")
    return False                                     # "question": research.queue_unclear set needs already


def _tune(idea: str, tools: Tools) -> bool:
    from machine.referee import tune
    if tune.latest(idea) is None:
        t = tools.get("tune")(idea, log=tools.log)
        tools.log(t.text() if hasattr(t, "text") else str(t))
    else:
        tf = tune.latest(idea)
        chosen = tf["result"]["chosen"]
        store.set_state(idea, "TUNE RUN", f"the tune file was written already ({tf['trial_id']}); chosen {chosen}",
                        waits_for="the check years 2021-2023, opened once for the chosen version")
    if state_of(idea)["state"] != "TUNE RUN":
        store.set_state(idea, "TUNE RUN", "tuned", waits_for="the check years, opened once")
    return True


def _judge(idea: str, tools: Tools) -> bool:
    """TUNE RUN / CHECK OPENED -> judged, then REPORTED. An idea the research agent was running (its plan is saved)
    goes on through the agent: it resumes - a version that ran is not run again, the check years are judged once."""
    from machine.agent import research
    from machine.referee import splits
    if config.RESEARCH_AGENT and research.plan_path(idea).is_file():
        return _research(idea, tools)
    form = store.load_form(idea)
    if not _judged(idea):
        opening = splits.family_opening(idea, form.parent_idea)
        if opening is not None and opening["idea_id"] != idea:
            tools.log(f"{idea}: the {opening['family']} family used its check years for {opening['idea_id']} - "
                      "judged by walk-forward inside 2016-2020")
            tools.get("walk_forward")(idea, tools.log)
            store.set_waits_for(idea, "the machine: the report", f"judged by walk-forward inside 2016-2020 (the "
                                f"family's check years went to {opening['idea_id']})")
        else:
            who = ("the intake (L10b), by itself, after Tejas's YES to the 3 setups it found "
                   f"({(state_of(idea).get('shown_yes') or {}).get('source', 'Telegram')})")
            rep = tools.get("check")(idea, who=who, log=tools.log)
            tools.log(rep.text() if hasattr(rep, "text") else str(rep))
    return _report(idea, tools)


def _judged(idea: str) -> bool:
    from machine.referee import trials
    return any(f.get("kind") == "referee" and f.get("idea_id") == idea for f in trials.load())


def _report(idea: str, tools: Tools) -> bool:
    sent = [ln for ln in _read_jsonl(outbox_path()) if ln.get("idea") == idea and ln.get("why") == "L10 report"]
    if sent:
        line = sent[-1]
    else:
        _learn(idea, tools)
        b = tools.get("build")(idea)
        line = tools.get("send")(b)
    store.set_state(idea, "REPORTED", f"L10b: report sent ({line['id']})",
                    waits_for="T6: Tejas reads the report and decides")
    direction_check()
    return True


def _learn(idea: str, tools: Tools) -> None:
    """The learner (L13.1) before the report: the conditions of the judged version's trades, so the report has its
    "works best in / fails in" lines. It judges nothing and is not a try; if it cannot run, the report still goes out
    and says "not measured yet" - the reason goes to the log."""
    try:
        from machine.learner import run as learner
        from machine.referee import tune
        t = tune.latest(idea)
        if t is not None and learner.latest(idea, t["result"]["chosen"]) is None:
            tools.get("learn")(idea, log=tools.log)
    except Exception as err:                                  # noqa: BLE001 - never blocks the report
        tools.log(f"{idea}: the learner did not run ({type(err).__name__}: {err}) - the report says "
                  "'works best in / fails in: not measured yet'")


STEPS = {"RECEIVED": _searched, "SEARCHED": _drafted, "FORM DRAFTED": _questions, "QUESTIONS QUEUED": _answered,
         "ANSWERED": _lock, "CONFIRMED & LOCKED": _show, "SHOWN": _after_shown, "TUNE RUN": _judge,
         "CHECK OPENED": _judge}


def _walk_forward(idea: str, log=print):
    """A family whose check years were used by another idea: this idea's versions judged by walk-forward inside
    2016-2020 (machine/referee/judge.py) - every version compiled again on 2016-2020 from the tune file's readings
    (and a research version's own changes, L14.1)."""
    from machine.ideas.compile import Market, compile_idea, needed_tfs, prepare
    from machine.referee import judge, prior, tune
    form = store.load_form(idea)
    tf = tune.latest(idea)
    rows = tf["result"]["versions"]
    tfs = tuple(dict.fromkeys(t for row in rows.values()
                              for t in needed_tfs(prepare(form, row["readings"], row.get("changes") or None)[0])))
    market = Market.load(str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1]), tfs=tfs)
    versions = {vid: compile_idea(form, market, readings=row["readings"], changes=row.get("changes") or None).signals
                for vid, row in rows.items()}
    return judge.judge_walk_forward(versions, tf=form.timeframe, key=prior.key_for_form(form),
                                    newest=tf["result"]["chosen"], note=f"L10b: {idea} by walk-forward",
                                    tfs={vid: row.get("timeframe", form.timeframe) for vid, row in rows.items()})


# ================================================================================================= the weekly batch

def _parse_batch_date(name: str) -> dt.datetime | None:
    for fmt in ("%Y-%m-%d-%H%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(name, fmt).astimezone()
        except ValueError:
            continue
    return None


def last_batch() -> dt.datetime | None:
    """When the last batch that SENT something went out: the L9b batches (question-queue/batches/) and the SHOWN
    pictures (plan/work/intake/sends.jsonl)."""
    times = []
    folder = Path(config.QUESTION_QUEUE_DIR) / "batches"
    for f in folder.glob("*/batch.json") if folder.is_dir() else []:
        try:
            b = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        t = _parse_batch_date(str(b.get("date") or f.parent.name))
        if t and not b.get("dry_run") and b.get("chosen"):
            times.append(t)
    times += [dt.datetime.fromisoformat(x["at"]) for x in _read_jsonl(work_dir() / "sends.jsonl") if x.get("at")]
    return max(times) if times else None


def next_batch() -> dt.datetime | None:
    """When the next batch may go (None = now: no batch went in the last 7 days)."""
    last = last_batch()
    if last is None or _now() - last >= BATCH_EVERY:
        return None
    return last + BATCH_EVERY


def batch_step(tools: Tools, now: dt.datetime | None = None, early: str | None = None) -> bool:
    """Send the weekly batch when it is due and something waits: the L9b questions (machine.questions.batch) and the
    SHOWN picture (1 question of the budget). early: the batch goes NOW, before its day - only on his words (CLAUDE.md
    rule 16: questions go in one weekly batch); the words go into sends.jsonl, and the next batch is 7 days after it."""
    from machine.intake import shown
    from machine.questions import batch as QB
    now = now or _now()
    if early is not None and not str(early).strip():
        raise ValueError("sending before the weekly day needs his words (why)")
    if next_batch() is not None and early is None:
        return False
    made = [i for i in store.ideas() if (shown.load(i) or {}).get("status") == "made"
            and state_of(i)["state"] == "CONFIRMED & LOCKED"]
    l9b = QB.waiting()
    if not made and not l9b:
        return False
    budget = config.QUESTION_BUDGET - len(made)
    sent = []
    if l9b and budget > 0:
        out = tools.get("batch")(dry=False, budget=budget, now=now)
        sent += list(out.get("chosen", []))
        tools.log(f"weekly batch {out.get('date')}: {len(out.get('chosen', []))} questions")
    for idea in made:
        rec = shown.load(idea)
        pic = rec["picture"]
        words = state_of(idea).get("intake", {}).get("words") or store.load_form(idea).words
        short = words if len(words) <= 160 else words[:157] + "..."
        text = (f"Is this your idea? {label(idea).capitalize()}: \"{short}\"\n"
                "3 setups the machine found by itself (2016-2020). Each chart ends at its entry candle.\n"
                "Reply to this picture: YES - or NO with the number (1, 2 or 3) and a few words.\n"
                "(This picture counts as 1 question of this week.)")
        _tell(f"{rec['qid']}-{now:%Y-%m-%d-%H%M}", text, photo=pic["file"], qid=rec["qid"], idea=idea,
              why="L10b SHOWN",
              shown={"subject": pic["subject"], "pixel_sha256": pic["pixel_sha256"], "kind": "question",
                     "file": pic["file"], "why": f"L10b SHOWN {rec['qid']}"})
        rec.update(status="sent", sent=_iso(now))
        shown.save(rec)
        store.set_state(idea, "SHOWN", f"the picture of 3 setups sent ({rec['qid']})",
                        waits_for="Tejas: YES or NO to the picture of 3 setups")
        sent.append(rec["qid"])
    if made:
        _append_jsonl(work_dir() / "sends.jsonl", {"at": _iso(now), "shown": [shown.load(i)["qid"] for i in made],
                                                   "with_l9b": [q for q in sent if "-shown-" not in q],
                                                   **({"early": str(early)} if early is not None else {})})
    for idea in active():                                    # say when the next batch goes
        if state_of(idea)["state"] == "QUESTIONS QUEUED" and not state_of(idea).get("needs"):
            store.set_waits_for(idea, _question_wait(idea))
    return bool(sent)


# ================================================================================================= his replies

def on_reply(question: dict, words: str, has_photo: bool, *, source: str = "",
             now: dt.datetime | None = None) -> str | None:
    """The bot calls this for a reply to one of the intake's lines (the SHOWN picture, the 10-in-a-row question). It is
    filed already (inbox + answers.md). Returns the line for the bot's reply, or None when it is not the intake's."""
    qid = str(question.get("qid") or "")
    m = SHOWN_QID.match(qid)
    if m:
        return _shown_reply(m.group(1), qid, words, has_photo, source, now or _now())
    if DIRECTION_QID.match(qid):
        return _direction_reply(qid, words, has_photo, source, now or _now())
    return None


def _shown_reply(idea: str, qid: str, words: str, has_photo: bool, source: str, now: dt.datetime) -> str:
    from machine.intake import shown
    from machine.questions.interpret import read_confirm
    rec = shown.load(idea)
    if rec is None or rec.get("qid") != qid or rec.get("status") not in ("sent", "needs a session"):
        return "Saved. (That picture is not waiting for an answer any more.)"
    if rec["status"] == "needs a session":                    # a session gives its reading of his words (answer)
        retry(idea)
    kind = read_confirm(words, has_photo)
    rec["replies"].append({"words": words, "photo": has_photo, "source": source, "at": _iso(now), "kind": kind})
    if kind == "yes":
        rec["status"] = "yes"
        shown.save(rec)
        s = state_of(idea)
        s["shown_yes"] = {"words": words, "source": source, "at": _iso(now)}
        s["waits_for"] = "the machine: the test runs now (every version on 2016-2020, then the check years)"
        s["history"].append({"state": s["state"], "at": _iso(now), "note": f"his YES to the 3 setups ({source})"})
        store._write(store.folder(idea) / store.STATE, store._json(s))
        return (f"Thank you. The machine tests {label(idea)} now, by itself.\n"
                "The report comes on Telegram when it is done.")
    if kind == "no":
        rec["status"] = "no"
        shown.save(rec)
        store.reopen(idea, f"his NO to the 3 setups ({source}): \"{words[:200]}\"",
                     waits_for=f"{SESSION} reads his NO")
        needs(idea, f"you said NO to the picture of 3 setups: \"{words[:200]}\" - a session reads it and asks "
                    "again", tell=False)
        return ("Not tested. Nothing is changed by guessing.\n"
                "A Claude session reads your words and asks you again.")
    rec["status"] = "needs a session"
    shown.save(rec)
    needs(idea, f"your reply to the picture of 3 setups was not a plain YES or NO: \"{words[:200]}\"", tell=False)
    return ("Nothing changes: only a plain YES starts the test.\n"
            "A Claude session reads your words.")


# ================================================================================================= 10 in a row

def direction_path() -> Path:
    return work_dir() / "direction.json"


def direction() -> dict | None:
    p = direction_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def paused() -> str | None:
    """Why no new idea starts (None = new ideas may start)."""
    d = direction()
    if not d:
        return None
    if d["status"] == "asked":
        return (f"new ideas wait: {d['streak']} answers in a row were not \"works\"; Tejas decides - keep going, "
                "change direction, or stop (asked " + d["asked"][:10] + ")")
    if d["status"] == "stop":
        return "new ideas wait: Tejas said STOP after " + str(d["streak"]) + " answers in a row that were not \"works\""
    if d["status"] == "needs a session":
        return "new ideas wait: a Claude session reads his answer about the direction (plan/work/intake/direction.json)"
    return None


def direction_check(summary: dict | None = None) -> bool:
    """After a report: 10 or more answers in a row that are not "works" -> ONE line to Tejas, once per streak."""
    from machine.referee import project
    s = summary or project.summary()
    if not s.get("ask_tejas"):
        return False
    start = s["answers"] - s["not_works_in_a_row"]              # where this streak began
    d = direction()
    if d and d.get("streak_start") == start:
        return False
    qid = f"project-direction-{s['answers']}"
    text = (f"The last {s['not_works_in_a_row']} ideas all came out as \"chance\" or \"not enough trades\" - "
            "none as \"works\".\n"
            "This is your decision: keep going, change direction, or stop?\n"
            "Reply to this message: KEEP GOING, or STOP, or a few words about the new direction.\n"
            "New ideas wait until you answer.")
    rec = {"asked": _iso(_now()), "qid": qid, "streak": s["not_works_in_a_row"], "streak_start": start,
           "answers": s["answers"], "status": "asked", "reply": None}
    store._write(direction_path(), json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    _tell(qid, text, qid=qid, why="L10b 10 in a row")
    return True


def _direction_reply(qid: str, words: str, has_photo: bool, source: str, now: dt.datetime) -> str:
    d = direction()
    if not d or d.get("qid") != qid or d.get("status") != "asked":
        return "Saved. (That question is not waiting for an answer any more.)"
    text = (words or "").strip()
    kind = "keep going" if KEEP.match(text) and not has_photo else ("stop" if STOP.match(text) and not has_photo
                                                                     else "needs a session")
    d.update(status=kind, reply={"words": words, "photo": has_photo, "source": source, "at": _iso(now)})
    store._write(direction_path(), json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    if kind == "keep going":
        return "OK. The machine goes on with the next idea."
    if kind == "stop":
        return "Stopped. No new idea starts until you say so (a Claude session can start it again)."
    return "Saved. A Claude session reads your words about the new direction. New ideas wait until then."


# ================================================================================================= STATUS

def _since(iso: str, now: dt.datetime) -> str:
    try:
        t = dt.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return "?"
    days = (now.date() - t.date()).days
    return f"{t:%d %b}" + (f" ({days} day{'s' if days != 1 else ''})" if days else " (today)")


def status_lines(now: dt.datetime | None = None) -> list[str]:
    """One line per idea: its state, what it waits for, since when. A BLOCKED idea's line is blocked.status_lines()."""
    now = now or _now()
    lines = []
    b_lines = {ln.split(" ", 1)[0]: ln for ln in blocked.status_lines(now)}
    run_now = current()
    for idea in store.ideas():
        try:
            s = state_of(idea)
        except store.StoreError:
            continue
        if s["state"] == "BLOCKED" and idea in b_lines:
            lines.append(b_lines[idea])
            continue
        wait = s.get("waits_for") or ("nothing" if s["state"] in FINISHED else "the machine")
        if s["state"] not in ("BLOCKED", *FINISHED) and run_now and idea != run_now:
            wait = f"its turn ({run_now} runs now)"
        lines.append(f"{idea} {s['state']} since {_since(s.get('since', ''), now)}: waits for {wait}")
    q = pending()
    if q:
        lines.append(f"Queue: {len(q)} new idea{' waits its' if len(q) == 1 else 's wait their'} turn (the oldest "
                     f"since {_since(q[0].get('at', ''), now)})")
    p = paused()
    if p:
        lines.append(p[0].upper() + p[1:])
    return lines


# ================================================================================================= one run

@contextmanager
def one_run():
    """Only one run at a time on this PC. Yields False when another run holds the lock (then do nothing)."""
    path = run_dir() / "intake.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    got = True
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        got = False
    try:
        yield got
    finally:
        if got and os.name == "nt":
            try:
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        fh.close()


def run(tools: Tools | None = None, *, max_steps: int = 60) -> list[str]:
    """Everything the machine can do now: the running idea's steps, a new idea from the queue when none runs, the
    weekly batch when it is due. Returns what it did (one line each)."""
    tools = tools or Tools(log=_logger())
    did = []
    with one_run() as mine:
        if not mine:
            tools.log("another intake run is working - this one does nothing")
            return did
        for _ in range(max_steps):
            if batch_step(tools):
                did.append("weekly batch sent")
                tools.log(did[-1])
                continue
            idea = current()
            if idea is None:
                p = paused()
                q = pending()
                if p or not q:
                    if p and q:
                        tools.log(p)
                    break
                idea = take(q[0])
                did.append(f"{idea} RECEIVED from the queue")
                tools.log(did[-1])
                continue
            before = state_of(idea)["state"]
            try:
                moved = step(idea, tools)
            except Exception as err:                         # the state stays; the next run tries again
                tools.log(f"{idea}: the step after {before} stopped: {err}\n{traceback.format_exc()}")
                did.append(f"{idea}: stopped in {before} ({type(err).__name__}: {str(err)[:200]})")
                break
            after = state_of(idea)["state"]
            if moved:
                did.append(f"{idea}: {before} -> {after}")
                tools.log(did[-1])
            elif not batch_step(tools):
                break
        else:
            tools.log(f"stopped after {max_steps} steps")
    return did


def _logger() -> Callable[[str], None]:
    path = run_dir() / "intake.log"

    def log(text: str) -> None:
        line = f"{_now():%Y-%m-%d %H:%M:%S} {text}"
        try:
            print(line, flush=True)
        except (OSError, ValueError, UnicodeEncodeError):
            pass                                             # pythonw: no console
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
    return log


def spawn() -> bool:
    """Start `python -m machine.intake run` as its own program (the bot calls this: a long test never stops the bot
    from reading his messages). No window on his PC. A second run exits at once (the lock)."""
    exe = Path(sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        exe = w if w.is_file() else exe
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen([str(exe), "-u", "-m", "machine.intake", "run"], cwd=str(config.ROOT),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True)
    except OSError:
        return False
    return True
