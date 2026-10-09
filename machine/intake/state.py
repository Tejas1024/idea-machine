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
  DECIDED             what comes next, chosen by the machine by the plan's default and TOLD to him in one line
                      (decision.md) - his words 1 Oct, LEARNED #162: "I don't know why u asking these questions".
                      An idea that lost with a plain stop is parked for his hedge: the machine then tests its chosen
                      version with his hedge by itself (L19d, machine/strategy/parked.py) and tells him that answer too

One idea runs at a time: the oldest idea that is not BLOCKED and not REPORTED / DECIDED. The next line of the queue
starts only when none is running. A REPORTED idea is decided first (it never waits for him), then a parked idea gets
its test with his hedge (about 9 minutes on his PC, no Claude call). A BLOCKED idea steps
aside; blocked.built() sends it back to FORM DRAFTED and it
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
AT_ONCE = ("Tejas, 7 Oct 2026 (LEARNED #175): 'which ever thebtask that is blockage for other tasks' - a picture of 3 "
           "setups holds up its idea and every idea behind it, so it goes when it is made, not on the weekly day")
TALKER_TRIES = 3                            # a talker call that got NO answer (no network, a time-out) is tried again
                                            # at the next run; after 3, a session looks. His Claude plan's usage limit
                                            # is not counted: it resets by itself, the machine waits (_wait_for_limit)
AFTER_REPORT = "the machine: its choice after the report (LEARNED #162 - it decides and tells him)"
SHOWN_YES = re.compile(                                  # his plain YES to the 3 setups in his own words: idea #2's
    r"^\s*(?:(?:yes\s*[,.!]?\s*)?all\s*(?:3|three|of them)?\s*(?:setups?\s*)?(?:are\s*)?(?:correct|right|fine|good|"
    r"ok|okay|my idea|mine|yes)|yes\s*[,.!]?\s*(?:to\s*)?all\s*(?:3|three|of them)?(?:\s*setups?)?|(?:correct|right)"
    r"\s*[,.!]?\s*all\s*(?:3|three)?)\s*[.!]*\s*$",
    re.IGNORECASE)                                       # was "All 3 correct" (1 Oct); the picture says "YES = all 3
                                                         # are how you trade it". Only "all": "3 correct" may mean
                                                         # setup 3. Nothing else on the line. 8 Oct: "All three setups
                                                         # are good", "All yes" (idea #4's second picture).
PLAIN_YES = re.compile(r"^\s*(yes|yeah|yep|yup)\s*[.!]*\s*$", re.IGNORECASE)   # a YES typed on its own (plain_yes):
                                                         # never "ok" / "correct" alone - on its own that is often
                                                         # "got it" (to the day's report, say), not "yes, my idea"
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


def messages_path() -> Path:
    """The bot's log of every message he sent (machine/telegram/bot.py): read for his YES typed on its own."""
    return Path(config.TELEGRAM_DIR) / "messages.jsonl"


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
    plan = s["needs"]["why"].startswith(FORWARD_STUCK)        # its forward plan again, never a test with his hedge
    s.pop("needs")
    s["waits_for"] = FORWARD if plan else "the machine"
    s.pop("talker_failures", None)
    s.pop("research_failures", None)
    store._write(store.folder(idea) / store.STATE, store._json(s))
    return s


def _his_limit(why) -> bool:
    """The Claude call got no answer because his Claude plan's usage limit is reached (machine/talker/llm.py marks
    every such error with these words)."""
    return "usage limit" in str(why or "").lower()


def _wait_for_limit(idea: str, what: str, why) -> bool:
    """His Claude plan's usage limit: it resets by itself, and a session could only wait too. So the idea never asks
    for a session over it - the machine tries again at every run (config.INTAKE_EVERY_MINUTES) and goes on by itself.
    ONE Telegram line a day tells him why his idea is slow; the history gets one line per wait, not one per try."""
    s = state_of(idea)
    first = "limit_since" not in s
    s.setdefault("limit_since", _iso(_now()))
    s["waits_for"] = (f"the machine: your Claude plan's usage limit is reached ({what}); it tries again every "
                      f"{config.INTAKE_EVERY_MINUTES} minutes and goes on by itself")
    if first:
        s["history"].append({"state": s["state"], "at": _iso(_now()),
                             "note": f"{what}: his Claude plan's usage limit ({str(why)[:200]}) - the machine waits"})
    store._write(store.folder(idea) / store.STATE, store._json(s))
    _tell(f"{idea}-limit-{_now():%Y%m%d}",
          f"{label(idea).capitalize()} waits a little: your Claude plan has reached its usage limit.\n"
          f"The machine tries again every {config.INTAKE_EVERY_MINUTES} minutes and goes on by itself when the limit "
          "is over.\nYou don't need to do anything.", idea=idea, why="L10b waits for his usage limit")
    return False


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
    parked: Callable | None = None                 # machine.strategy.parked.run (L19d: a parked idea, his hedge)
    forward: Callable | None = None                # machine.forward.plan.write (L17c: a forward-test idea's plan)
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
        if name == "parked":
            from machine.strategy import parked
            return parked.run
        if name == "forward":
            from machine.forward import plan
            return plan.write
        raise KeyError(name)


# ================================================================================================= the steps

def step(idea: str, tools: Tools) -> bool:
    """Do the idea's next step if it is the machine's. True = something changed."""
    s = state_of(idea)
    if s.get("needs") and s["state"] == "SHOWN" and listed_reply(idea):   # his list of setup numbers, left for a
        s = state_of(idea)                                                  # session by the bot's older code
        if s.get("needs"):                                                  # a number left out: his NO, a session
            return True
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
        why = d.answer.error if d.answer else "no answer"
        if _his_limit(why):
            return _wait_for_limit(idea, "the talker writes the form", why)
        n = int(s.get("talker_failures", 0)) + 1
        s = state_of(idea)
        s["talker_failures"] = n
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
    """CONFIRMED & LOCKED: the SHOWN picture made (sent at once by batch_step - #175)."""
    from machine.intake import shown
    rec = shown.load(idea)
    if rec is None or rec.get("status") in ("failed", "no", "withdrawn") or shown.stale(rec):  # "no" / taken back:
        try:                                                     # a NEW picture; not sent, drawn by older code: again
            rec = shown.make(idea, tools.market(), compile_fn=tools.get("compile"), log=tools.log)
        except shown.ShownError as err:
            needs(idea, f"the picture of 3 setups could not be made: {err}")
            return False
        if shown.stale(rec):                                     # never drawn again and again in one run
            needs(idea, "the picture of 3 setups was drawn again, but its record still says an older drawing")
            return False
        store.set_waits_for(idea, "the machine: the picture of 3 setups goes out now (#175)", "SHOWN picture made")
        return True
    return False


def _after_shown(idea: str, tools: Tools) -> bool:
    """SHOWN: his YES -> the research agent (L14.2) - or, with config.RESEARCH_AGENT off, L10's tune (TUNE RUN). His NO
    was handled when he replied (on_reply). A YES he typed on its own, not as a reply to the picture, is found here
    (plain_yes)."""
    from machine.intake import shown
    rec = shown.load(idea) or {}
    if plain_yes(idea, rec):
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
    to his Telegram (REPORTED). His Claude plan's usage limit: the machine waits and tries again at every run, never
    a session (_wait_for_limit). A planning call with no answer for another reason (no network), or a plan code
    refused (nothing saved, no money seen - a NEW planning call may pass), is tried again at the next run; after
    config.RESEARCH_TRIES a session looks. Anything else that stops it (an unclear meaning his documents answer, a
    version that cannot run): a session looks at once."""
    out = tools.get("research")(idea, send=True, tell=True, say=tools.log)
    tools.log(f"{idea}: the research agent - {out.status}: {out.said}")
    if out.status == "reported":
        if state_of(idea)["state"] != "REPORTED":           # write_report(send=True) moved it; a re-run that found
            store.set_state(idea, "REPORTED", "L14.2: researched and reported",      # it reported already: the same
                            waits_for=AFTER_REPORT)
        direction_check()
        return True
    refused = out.status == "needs a session" and getattr(out, "again", False)
    if out.status == "no answer" or refused:
        if not refused and _his_limit(out.said):
            return _wait_for_limit(idea, "the research plan", out.said)
        what = "code refused the research plan" if refused else "the research plan got no answer"
        s = state_of(idea)
        n = int(s.get("research_failures", 0)) + 1
        s["research_failures"] = n
        s["waits_for"] = (f"the machine: {what} ({out.said[:160]}); it "
                          + ("asks for a new plan at the next run" if refused else "tries again"))
        s["history"].append({"state": s["state"], "at": _iso(_now()),
                             "note": f"research: {'the plan was refused' if refused else 'no answer'} ({n})"})
        store._write(store.folder(idea) / store.STATE, store._json(s))
        if n >= config.RESEARCH_TRIES:
            needs(idea, f"{what} {n} times ({out.said[:200]})")
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
    store.set_state(idea, "REPORTED", f"L10b: report sent ({line['id']})", waits_for=AFTER_REPORT)
    direction_check()
    return True


# ================================================================================================= after the report

DECISION = "decision.md"
PARK = "the machine: his hedge instead of a plain stop, on 2016-2020 (L19d, machine/strategy/parked.py)"
FORWARD = "L17c: its forward test on new days, no money - the machine writes its plan first (machine/forward/plan.py)"
PLANNED = "L17b: the pretend trades of its forward plan (forward-plan.md), no money (L17b is not built yet)"
FORWARD_STUCK = "its forward plan could not be written"


def choice(answer: str, hedged: bool, *, rest_passed: bool = True) -> tuple[str, str, str]:
    """What comes next after a report, by the plan's default (CLAUDE.md, LEARNED #162; Claude's reading of
    plan/README.md 1 and 7 for "works" and "needs fresh years"): (the choice in capitals, what it means for the idea,
    what the idea waits for). The machine goes on with the next idea in every case - one idea runs at a time.
    rest_passed: every check but "not already seen" passed. "Needs fresh years" is that answer AT BEST (README 7):
    only such an idea goes to a forward test; one that also failed money checks is a "chance" idea."""
    from machine.referee.verdict import FEW, FRESH, WORKS
    if answer == WORKS:
        return ("LIVE TRIAL", "it goes to a live trial with alerts on new days - no money (L17b), planned and locked "
                "before it starts. Real money only after the legal check and your yes (T11).",
                "L17b: its live trial with alerts, no money (L17b is not built yet)")
    if answer == FRESH and rest_passed:
        return ("FORWARD TEST", "the old project already used 2016-2023 for this idea, so only new days can judge it: "
                "a forward test on live prices, no money (L17c).", FORWARD)
    if answer == FEW:
        return ("NEXT IDEA", "it is kept, not dropped: 2016-2023 has too few of its trades to tell, and the years "
                "after 2023 are sealed.", "nothing (kept on file: too few trades to tell)")
    if hedged:
        return ("NEXT IDEA", "it lost with your hedge too. It is kept on file, not dropped; nothing more is run "
                "for it.",
                "nothing (kept on file: it lost with his hedge too)")
    return ("NEXT IDEA", "it is parked, not dropped: it lost with a plain stop. Now the machine tests the same rule "
            "with your hedge instead of a stop (2016-2020, not a new try), and tells you what it found.", PARK)


def undecided() -> list[str]:
    """REPORTED ideas the machine has not decided yet (a person's wait set aside), oldest first."""
    out = []
    for idea in store.ideas():
        try:
            s = state_of(idea)
        except store.StoreError:
            continue
        if s["state"] == "REPORTED" and not s.get("needs"):
            out.append(idea)
    return out


def unhedged() -> list[str]:
    """DECIDED ideas parked for his hedge whose test with his hedge has not run yet (a person's wait set aside), oldest
    first. "the machine" is what retry() leaves when a session fixed a test that could not run."""
    out = []
    for idea in store.ideas():
        try:
            s = state_of(idea)
        except store.StoreError:
            continue
        if s["state"] == "DECIDED" and s.get("waits_for") in (PARK, "the machine") and not s.get("needs"):
            out.append(idea)
    return out


def unplanned() -> list[str]:
    """DECIDED ideas the machine chose a FORWARD TEST for whose plan is not written yet (L17c), oldest first."""
    out = []
    for idea in store.ideas():
        try:
            s = state_of(idea)
        except store.StoreError:
            continue
        if s["state"] == "DECIDED" and s.get("waits_for") == FORWARD and not s.get("needs"):
            out.append(idea)
    return out


def _decide(idea: str, tools: Tools) -> bool:
    """REPORTED -> DECIDED: the machine chooses what comes next by the plan's default, writes decision.md, and TELLS
    him in one line - never asks (his words 1 Oct, LEARNED #162: "I don't know why u asking these questions"; as
    idea #1 on "What ever you think is better", 29 Sep). The numbers are the referee's, read from its files. A
    decision.md already written (a session, or a crash after writing) is kept as it is."""
    from machine.referee.verdict import FRESH
    from machine.report import build as RB
    path = store.folder(idea) / DECISION
    if path.is_file():
        store.set_state(idea, "DECIDED", "decision.md was written already - kept as it is", waits_for=None)
        return True
    try:
        f = RB.facts(idea)
    except (RB.ReportError, store.StoreError, OSError, ValueError, KeyError) as err:
        needs(idea, f"the machine cannot read its report to decide what comes next ({type(err).__name__}: "
                    f"{str(err)[:240]})")
        return False
    word, why = f.r["answer"]["word"], f.r["answer"]["why"]
    failed = ", ".join(c["name"] for c in f.r["answer"]["checks"]
                       if not c["passed"] and c["name"] != "not already seen")
    rest = not failed
    # How it was TESTED decides, not what the form says: the referee closes every trade with a plain stop today (a
    # hedge idea's hedge point as a stop - the report says so; idea #2's form says "hedge"). His hedge reaches the
    # referee in L19c / L19d.
    hedge_tested = False
    head, means, waits = choice(word, hedge_tested, rest_passed=rest)
    sent = [ln for ln in _read_jsonl(outbox_path()) if ln.get("idea") == idea and ln.get("why") == "L10 report"]
    report = f"sent {sent[-1]['at'][:16].replace('T', ' ')}" if sent else "sent"
    wy, counts = f.r["worst_year"], f.r["counts"]
    nxt = current()
    later = (f"The machine goes on with {label(nxt)} (now {state_of(nxt)['state']})." if nxt else
             "The machine starts the next idea in the queue." if pending() else
             "No other idea waits now: send the next one on Telegram, starting with IDEA.")
    p = paused()
    if p:
        later = f"New ideas wait for your answer to the other message ({p.split(';')[0]})."
    name = label(idea)
    text = "\n".join([
        f"# {name.capitalize()} — the decision after the report", "",
        f"**Date:** {_now():%d %b %Y %H:%M} · **Report:** `report.html` ({report}, answer {word.upper()}) · "
        "decided by the machine (L10b), by itself", "",
        "## Who decides", "",
        "His words, 1 Oct 2026, when a session asked what to do after idea #2's report (LEARNED #162):", "",
        "> I don't know why u asking these questions", "",
        "So after a report the machine chooses by the plan's default and only tells him. He can change it any time by "
        "saying so.", "",
        "## The choice (the line he got on Telegram says the same)", "",
        f"**{head}** - {name}: {means}", "",
        *([f"Not a forward test: the old project already used these years for it, AND it failed other checks too "
           f"({failed}). \"Needs fresh years\" is its answer at best (README section 7) - a forward test is for an "
           "idea that passed everything else (Claude's reading).", ""] if word == FRESH and not rest else []),
        f"{later}", "",
        "## The numbers it rests on (the referee's, from its files - nothing new was run)", "",
        f"- Answer: {word.upper()} - {why}",
        f"- Judged on: {RB._where_judged(f)} · {RB.inr(f.r['a_year']['net_inr'])} a year · worst year "
        f"{wy['year']}: {RB.inr(wy['net_inr'])} ({wy['trades']} trades)",
        f"- Versions tried: {len(counts['family_versions'])} of {counts['budget']} · closed by: a plain stop"
        + (" (his idea uses a hedge; the referee tested the hedge point as a plain stop)" if f.hedged else "")
        + f" · the referee's file: `{f.judge_path}`",
        "- Rupees at 0.01 lot, ₹40 per position all-in (his #54, #115).", ""])
    _tell(f"{idea}-decided", f"{name.capitalize()} came out {word.upper()}. The machine's choice: {head}.\n"
          f"{name.capitalize()}: {means}\n{later}\n"                # first the line (once per id: a crash after it
          "You don't need to do anything. If you want something else, just say it.",   # never sends it twice), then
          idea=idea, why="L10b decided")                                               # the file, then the state
    store._write(path, text)
    store.set_state(idea, "DECIDED", f"the machine decided (LEARNED #162): {head} - decision.md", waits_for=waits)
    return True


def _hedge(idea: str, tools: Tools) -> bool:
    """DECIDED and parked for his hedge -> its chosen version through his hedge on 2016-2020 (L19d), by itself: a
    section added to decision.md, ONE line that tells him (never asks, LEARNED #162), then what the idea waits for
    now. The same version - not a new try. A test that ran is never run again (parked.run reads its trial file back).
    A test that cannot run waits for a session: it never holds up the next idea."""
    from machine.costs import inr
    if idea not in unhedged():
        return False
    try:
        out = tools.get("parked")(idea, log=tools.log)
    except Exception as err:                                  # noqa: BLE001 - said in words; a session looks
        tools.log(f"{idea}: the test with his hedge stopped: {err}\n{traceback.format_exc()}")
        needs(idea, f"its test with your hedge could not run ({type(err).__name__}: {str(err)[:240]})")
        return False
    hh, r = out["his_hedging"], out["result"]
    wy = hh.get("worst_year") or {}
    name = label(idea).capitalize()
    _tell(f"{idea}-hedged", "\n".join([
        f"{name} with your hedge instead of a stop, on 2016-2020. The same rule the referee chose - not a new try. "
        "0.01 lot, ₹40 a position, all-in.",
        f"- Your entries with your hedge: {hh['trades']} trades, {inr(hh['net_inr_total'])} after cost"
        + (f". Worst year {wy['year']}: {inr(wy['net_inr'])}." if wy else "."),
        f"- Random entries with your hedge: {inr(r['boxes'][2]['total'])} (the middle of {r['runs']} tries). Yours did "
        f"better in {r['his_beats_random']['with_hedging']} of {r['runs']}.",
        f"The machine's choice: {out['choice']}. {out['means'][0].upper()}{out['means'][1:]}",
        "You don't need to do anything. If you want something else, just say it."]),
        idea=idea, why="L10b tested with his hedge")
    waits = FORWARD if out["choice"] == "FORWARD TEST" else "nothing (kept on file: tested with his hedge too)"
    store.set_state(idea, "DECIDED", f"tested with his hedge by itself (L19d, {out['trial']}): {out['choice']} - "
                                     "decision.md", waits_for=waits)
    return True


def _forward(idea: str, tools: Tools) -> bool:
    """DECIDED with a FORWARD TEST -> its forward plan written and locked by itself BEFORE any live bar (L17c,
    machine/forward/plan.py): forward-plan.json + .md in its folder (the nightly job puts them on main), ONE line that
    tells him (never asks, LEARNED #162), then the idea waits for its pretend trades (L17b). A plan that cannot be
    written waits for a session: it never holds up the next idea. A plan already there is never written again."""
    if idea not in unplanned():
        return False
    try:
        p = tools.get("forward")(idea, log=tools.log)
    except Exception as err:                                  # noqa: BLE001 - said in words; a session looks
        tools.log(f"{idea}: its forward plan could not be written: {err}\n{traceback.format_exc()}")
        needs(idea, f"{FORWARD_STUCK} ({type(err).__name__}: {str(err)[:240]})")
        return False
    from machine.forward import plan as FP
    _tell(f"{idea}-forward-plan", FP.telegram_text(p, label(idea).capitalize()), idea=idea,
          why="L17c forward plan")
    store.set_state(idea, "DECIDED", f"its forward plan is written and locked (L17c, {FP.PLAN_MD}): starts "
                                     f"{p['start']}, judged at {p['trades_needed'] or 'its end date'} trades or on "
                                     f"{p['end']}", waits_for=PLANNED)
    return True


def _after_decided(idea: str, tools: Tools) -> bool:
    """A decided idea: its test with his hedge (parked), or its forward plan (a forward test) - whichever it waits for."""
    return _hedge(idea, tools) or _forward(idea, tools)


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
         "CHECK OPENED": _judge, "REPORTED": _decide, "DECIDED": _after_decided}


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
    pictures (plan/work/intake/sends.jsonl) - except a picture sent at once when it was made (#175): it moves no
    weekly day."""
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
    times += [dt.datetime.fromisoformat(x["at"]) for x in _read_jsonl(work_dir() / "sends.jsonl")
              if x.get("at") and not x.get("at_once")]
    return max(times) if times else None


def next_batch() -> dt.datetime | None:
    """When the next batch may go (None = now: no batch went in the last 7 days)."""
    last = last_batch()
    if last is None or _now() - last >= BATCH_EVERY:
        return None
    return last + BATCH_EVERY


def batch_step(tools: Tools, now: dt.datetime | None = None, early: str | None = None) -> bool:
    """Send what waits for him: the SHOWN picture at once when it is made (his #175: it holds up its idea and every
    idea behind it - AT_ONCE), and the L9b questions (machine.questions.batch) in the weekly batch when it is due.
    early: the weekly batch goes NOW, before its day - only on his words (CLAUDE.md rule 16: questions go in one
    weekly batch); the words go into sends.jsonl, and the next batch is 7 days after it. A picture sent at once with
    no L9b question moves no weekly day (last_batch)."""
    from machine.intake import shown
    from machine.questions import batch as QB
    now = now or _now()
    if early is not None and not str(early).strip():
        raise ValueError("sending before the weekly day needs his words (why)")
    due = next_batch() is None or early is not None
    made = [i for i in store.ideas() if (shown.load(i) or {}).get("status") == "made"
            and not shown.stale(shown.load(i)) and state_of(i)["state"] == "CONFIRMED & LOCKED"]
    l9b = QB.waiting() if due else []
    if not made and not l9b:
        return False
    budget = config.QUESTION_BUDGET - len(made)
    sent = []
    if l9b and budget > 0:
        out = tools.get("batch")(dry=False, budget=budget, now=now)
        sent += list(out.get("chosen", []))
        tools.log(f"weekly batch {out.get('date')}: {len(out.get('chosen', []))} questions")
    with_l9b = list(sent)
    for idea in made:
        rec = shown.load(idea)
        pic = rec["picture"]
        words = state_of(idea).get("intake", {}).get("words") or store.load_form(idea).words
        short = words if len(words) <= 160 else words[:157] + "..."
        text = (f"Is this your idea? {label(idea).capitalize()}: \"{short}\"\n"
                "3 setups the machine found by itself (2016-2020). Each chart ends at its entry candle.\n"
                "Reply to this picture: YES - or NO with the number (1, 2 or 3) and a few words."
                + ("\n(This picture counts as 1 question of this week.)" if with_l9b else ""))
        _tell(f"{rec['qid']}-{now:%Y-%m-%d-%H%M}", text, photo=pic["file"], qid=rec["qid"], idea=idea,
              why="L10b SHOWN",
              shown={"subject": pic["subject"], "pixel_sha256": pic["pixel_sha256"], "kind": "question",
                     "file": pic["file"], "why": f"L10b SHOWN {rec['qid']}"})
        rec.update(status="sent", sent=_iso(now))
        shown.save(rec)
        store.set_state(idea, "SHOWN", f"the picture of 3 setups sent ({rec['qid']})",
                        waits_for="Tejas: YES or NO to the picture of 3 setups")
        tools.log(f"{idea}: the picture of 3 setups sent ({rec['qid']}, "
                  + ("with the weekly questions)" if with_l9b else
                     "early, on his words)" if early is not None else "at once when made, #175)"))
        sent.append(rec["qid"])
    if made:
        _append_jsonl(work_dir() / "sends.jsonl", {"at": _iso(now), "shown": [shown.load(i)["qid"] for i in made],
                                                   "with_l9b": with_l9b,
                                                   **({"early": str(early)} if early is not None else {}),
                                                   **({"at_once": AT_ONCE} if early is None and not with_l9b
                                                      else {})})
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
    rec = shown.load(idea)
    if rec is None or rec.get("qid") != qid or rec.get("status") not in ("sent", "needs a session"):
        return "Saved. (That picture is not waiting for an answer any more.)"
    if rec["status"] == "needs a session":                    # a session gives its reading of his words (answer)
        retry(idea)
    kind, got = _read_shown(words, has_photo, rec)
    rec["replies"].append({"words": words, "photo": has_photo, "source": source, "at": _iso(now), "kind": kind,
                           **got})
    return _shown_answer(idea, rec, kind, got, words, source, now)


def _read_shown(words: str, has_photo: bool, rec: dict) -> tuple[str, dict]:
    """His reply to the picture of 3 setups -> ("yes" / "no" / "words", {"listed": [...], "left_out": [...]} when it
    was his list of setup numbers, else {})."""
    from machine.questions.interpret import read_confirm
    if has_photo:
        return read_confirm(words, True), {}
    numbers = [int(x["n"]) for x in rec.get("setups") or [] if "n" in x]
    if SHOWN_YES.match(words or "") or listed_yes(words, numbers):
        return "yes", {}
    got = setup_list(words, numbers)
    if got is not None:
        return ("no" if got["left_out"] else "yes"), got
    return read_confirm(words, False), {}


def setup_list(words: str, numbers: list[int]) -> dict | None:
    """His way to answer a numbered picture, LEARNED #163 (1 Oct 2026, his zone charts): "Below are my zones, which I
    did not mentioned are not mine". A reply that is ONLY a list of the picture's setup numbers - "1,2,3", "1 2 3",
    "1.2.3.", any order - names the setups that are his; the rest are not. Read as his zone-chart lists are
    (machine.labels.replies.listed), the whole reply as one line, so a second line of words is never dropped.
    {"listed": [his numbers, his order], "left_out": [the picture's other numbers]}; None when the reply is more than
    such a list (other words, a yes / no word, a number the picture does not have): a session reads it."""
    from machine.labels import replies as LR
    mine = LR.listed(" ".join(str(words or "").split()))
    if not mine or not set(mine) <= set(numbers):
        return None
    return {"listed": mine, "left_out": [n for n in sorted(numbers) if n not in mine]}


_LIST_THEN_WORDS = re.compile(r"^\s*(\d[\d\s,.&]*?)\s*[-:]?\s*([A-Za-z].*)$")


def listed_yes(words: str, numbers: list[int]) -> bool:
    """His YES as he typed it to idea #4's second picture (8 Oct 2026, Telegram message 1247): "1,2,3" on one line and
    "All yes" on the next. True when every line is either the list of ALL the picture's setup numbers or a yes
    (SHOWN_YES, PLAIN_YES), and at least one line is a yes; "1,2,3 all yes" on one line counts too. A list that leaves
    a number out, a list with no yes (the zone charts take plain lists - never read for a picture), or any other word:
    False - a session reads it (R13)."""
    lines = [x.strip() for x in str(words or "").splitlines() if x.strip()]
    if len(lines) == 1 and (m := _LIST_THEN_WORDS.match(lines[0])):
        lines = [m.group(1), m.group(2)]
    yes = False
    for x in lines:
        if SHOWN_YES.match(x) or PLAIN_YES.match(x):
            yes = True
            continue
        got = setup_list(x, numbers)
        if got is None or got["left_out"]:
            return False
    return yes


def _setups(ns: list[int]) -> str:
    """[2] -> "setup 2" · [2, 3] -> "setups 2 and 3"."""
    if len(ns) == 1:
        return f"setup {ns[0]}"
    return "setups " + ", ".join(map(str, ns[:-1])) + f" and {ns[-1]}"


def _shown_answer(idea: str, rec: dict, kind: str, got: dict, words: str, source: str, now: dt.datetime) -> str:
    """What his reply does, read (_read_shown): YES -> the test runs; NO -> back to questions, a session reads it;
    else a session reads it. Returns the line for him."""
    from machine.intake import shown
    short = " ".join(str(words or "").split())[:40]
    if kind == "yes":
        rec["status"] = "yes"
        shown.save(rec)
        how = (f"his list \"{short}\" names all {len(got['listed'])} setups - his rule, LEARNED #163: the numbers he "
               "lists are his; " if got else "")
        s = state_of(idea)
        s["shown_yes"] = {"words": words, "source": source, "at": _iso(now),
                          **({"listed": got["listed"]} if got else {})}
        s["waits_for"] = "the machine: the test runs now (every version on 2016-2020, then the check years)"
        s["history"].append({"state": s["state"], "at": _iso(now), "note": f"his YES to the 3 setups ({how}{source})"})
        store._write(store.folder(idea) / store.STATE, store._json(s))
        return ((f"Your list \"{short}\" names all {len(got['listed'])} setups: that is YES (your rule: the numbers "
                 "you list are yours).\n" if got else "")
                + f"Thank you. The machine tests {label(idea)} now, by itself.\n"
                "The report comes on Telegram when it is done.")
    if kind == "no":
        rec["status"] = "no"
        shown.save(rec)
        out = _setups(got["left_out"]) if got else ""
        store.reopen(idea, f"his NO to the 3 setups ({source}): \"{words[:200]}\""
                     + (f" - his list leaves out {out} (LEARNED #163: the numbers he does not list are not his)"
                        if got else ""), waits_for=f"{SESSION} reads his NO")
        needs(idea, (f"your list \"{short}\" leaves out {out}: by your rule (the numbers you list are yours) that is "
                     f"a NO for {out} - a session reads it and asks again" if got else
                     f"you said NO to the picture of 3 setups: \"{words[:200]}\" - a session reads it and asks again"),
              tell=False)
        return ((f"Your list \"{short}\" leaves out {out}. By your rule (the numbers you list are yours), {out} "
                 f"{'is' if len(got['left_out']) == 1 else 'are'} not your idea: that is a NO.\n" if got else "")
                + "Not tested. Nothing is changed by guessing.\n"
                "A Claude session reads your words and asks you again.")
    rec["status"] = "needs a session"
    shown.save(rec)
    needs(idea, f"your reply to the picture of 3 setups was not a plain YES or NO: \"{words[:200]}\"", tell=False)
    return ("Nothing changes: only a plain YES starts the test.\n"
            "A Claude session reads your words.")


def listed_reply(idea: str, now: dt.datetime | None = None) -> bool:
    """The bot reads his replies in its own process, with the code it started with: before 7 Oct 2026 that code took
    his list of setup numbers ("1,2,3.", idea #3) for words only a session can read - the picture "needs a session".
    The intake (a new program every run) reads such a reply again the way _shown_reply reads it now - only when it is
    the picture's ONLY reply, words with no photo, and a list (setup_list): all its setup numbers = his YES, a number
    left out = his NO for that setup. ONE line tells him how it was read. A plain message is never read here: while
    zone charts wait, the bot takes his plain list of numbers for them. True = read."""
    from machine.intake import shown
    rec = shown.load(idea) or {}
    replies = rec.get("replies") or []
    if rec.get("status") != "needs a session" or len(replies) != 1 or replies[0].get("photo"):
        return False
    r = replies[0]
    kind, got = _read_shown(str(r.get("words") or ""), False, rec)
    if not got:
        return False
    now = now or _now()
    retry(idea)
    r.update(kind=kind, **got, read_again={"at": _iso(now), "by": "the intake: the bot's older code (before 7 Oct "
                                                                  "2026) had left his list for a session"})
    words, source = str(r["words"]), str(r.get("source") or "")
    line = _shown_answer(idea, rec, kind, got, words, f"{source}; his list read by the intake", now)
    try:
        at = f" at {dt.datetime.fromisoformat(str(r.get('at'))):%H:%M}"
    except (TypeError, ValueError):
        at = ""
    _tell(f"{rec['qid']}-list-read",
          f"{label(idea).capitalize()}, the picture of 3 setups: your reply{at} is read again now.\n"
          + ("The bot said \"only a plain YES starts the test\". That was wrong: its old code did not know your rule.\n"
             if kind == "yes" else "")
          + line, idea=idea, why="L10b SHOWN: his list of setup numbers, read by the intake")
    return True                                              # answers.md has his words already (the bot filed them)


def plain_yes(idea: str, rec: dict, now: dt.datetime | None = None) -> bool:
    """His YES to the picture of 3 setups typed on its own, NOT as a reply to it - his way on Telegram: none of his 9
    messages of 29 Sep - 4 Oct was a reply, and the label charts take his plain list the same way (message 915). The
    bot files such a message as a lesson (machine/telegram/route.py); this finds it in the bot's log and reads it as
    the reply it is. Taken only when ALL of it is a plain yes (SHOWN_YES, PLAIN_YES: "yes", "yeah" ..., or listed_yes:
    "1,2,3" + "All yes" - his 8 Oct answer), with no
    photo, not edited, sent after the picture and not after `now` - and only while nothing else waits for a yes
    (otherwise ONE line asks him to reply to the picture he means, and that message is never taken later). A plain NO
    stays a lesson: a NO needs a session anyway. True = the picture's answer is YES now."""
    from machine.intake import shown
    now = now or _now()
    if rec.get("status") != "sent":                          # answered already (or not sent): nothing to find
        return False
    try:
        since = dt.datetime.fromisoformat(rec["sent"])
    except (KeyError, TypeError, ValueError):
        return False
    unclear = set(rec.get("unclear_yes") or [])
    numbers = [int(x["n"]) for x in rec.get("setups") or [] if "n" in x]
    msg = None
    for m in _read_jsonl(messages_path()):
        if m.get("kind") != "lesson" or m.get("reply_to_id") is not None or m.get("photo_id") or m.get("edited") \
                or m.get("message_id") in unclear:
            continue
        try:
            at = dt.datetime.fromisoformat(str(m.get("sent")))
            when_ok = since <= at <= now
        except (TypeError, ValueError):                      # no time, or one with no time zone: never his answer
            continue
        words = str(m.get("words") or "").strip()
        if when_ok and (SHOWN_YES.match(words) or PLAIN_YES.match(words) or listed_yes(words, numbers)):
            msg = m
            break
    if msg is None:
        return False
    at, words = dt.datetime.fromisoformat(str(msg["sent"])), str(msg["words"]).strip()
    others = _others_waiting(idea)
    if others:
        rec.setdefault("unclear_yes", []).append(msg.get("message_id"))
        shown.save(rec)
        _tell(f"{rec['qid']}-plain-yes-unclear-{msg.get('message_id')}",
              f"Your message \"{words[:40]}\" ({at:%H:%M}) was not a reply to a picture, and more than one question "
              f"waits for a yes ({', '.join(others)}).\nPlease reply to the picture you mean: press on it, tap "
              "Reply, then type YES.", idea=idea, why="L10b SHOWN: his YES typed on its own, not clear")
        return False
    source = (f"Telegram message {msg.get('message_id')} at {at:%d %b %H:%M}, typed on its own (not a reply to the "
              "picture)" + (f"; the bot filed it as a lesson: {msg['inbox']}" if msg.get("inbox") else ""))
    line = _shown_reply(idea, rec["qid"], words, False, source, now)
    if shown_status(idea) != "yes":                         # a reply of his came first (the bot, the same minute)
        return False
    try:
        store.add_answer(idea, words, about=rec["qid"], source=source)
    except store.StoreError:
        pass                                                 # state.json and shown.json have it already
    lesson = Path(config.ROOT) / str(msg.get("inbox") or "")
    if msg.get("inbox") and lesson.is_file():               # its lesson file says how it was read (appended only)
        from machine.telegram import inbox
        inbox.add_to_inbox(lesson, words="", notes=[f"read by the intake ({_iso(now)}) as his answer to the picture of "
                                                    f"3 setups {rec['qid']} ({label(idea)}): a YES typed on its own, "
                                                    "not a reply to the picture"])
    _tell(f"{rec['qid']}-plain-yes",
          f"Your message \"{words[:40]}\" ({at:%H:%M}) is taken as your answer to the picture of 3 setups "
          f"({label(idea)}). It was not a reply to the picture, but no other picture waits for a YES.\n{line}",
          idea=idea, why="L10b SHOWN: his YES typed on its own")
    return (shown_status(idea) == "yes")


def shown_status(idea: str) -> str | None:
    from machine.intake import shown
    return (shown.load(idea) or {}).get("status")


def _others_waiting(idea: str) -> list[str]:
    """What else waits for his yes or no now - with any of these a YES typed on its own could mean that one: another
    idea's picture of 3 setups, an L9b question or its confirm picture, the 10-in-a-row question."""
    from machine.questions import queue as Q
    out = [f"the picture of 3 setups of {label(i)}" for i in store.ideas() if i != idea and shown_status(i) == "sent"]
    for r in Q.records():
        if r.get("status") in ("withdrawn", "locked"):
            continue
        try:
            if r.get("idea") and state_of(r["idea"])["state"] in FINISHED:
                continue                                     # an old question of a finished idea waits for nothing
        except (store.StoreError, OSError, ValueError, KeyError):
            pass
        if (r.get("confirm") or {}).get("status") == "sent" or \
                any(q.get("status") == "sent" for q in r.get("questions", [])):
            who = str(r.get("idea") or "")
            out.append(f"a question about {label(who)}" if store.IDEA_ID.fullmatch(who) else "a question")
    d = direction()
    if d and d.get("status") == "asked":
        out.append("the question about going on after 10 ideas in a row")
    return out


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
                did.append("sent what waited for him (the lines above say what)")
                tools.log(did[-1])
                continue
            late = undecided() or unhedged() or unplanned()  # a report first gets the machine's choice (#162),
                                                             # then a parked idea its test with his hedge (L19d),
                                                             # then a forward-test idea its plan (L17c)
            idea = late[0] if late else current()
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
            elif late:                                       # its choice waits for a session now (needs): the
                continue                                     # running idea goes on in the same run
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
