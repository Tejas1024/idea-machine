"""L10b - the intake: one idea from his words to the report by itself, only his answers by hand.

The prompt's tests: a fake idea RECEIVED -> REPORTED with a fake Telegram and a cached talker answer · BLOCKED resumes
after the block is added · a crash resumes and repeats no step · "no" at SHOWN goes back to questions. Also: the
SHOWN picture, STATUS, the 10-in-a-row line, one run at a time. No price data (hand-made bars, fake question records,
a hand-made tune and check through the real referee) except the last test (the real idea #1 SHOWN picture, 💻).
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from machine import config
from machine.charts import draw as D
from machine.charts import lint
from machine.ideas import blocked, store
from machine.ideas import compile as C
from machine.ideas.form import Idea
from machine.intake import shown
from machine.intake import state as S
from machine.questions import pictures as P
from machine.questions import queue as Q
from machine.referee import judge, prior, project, splits, tune
from machine.talker import glossary, llm
from machine.talker import translate as T
from machine.telegram import bot as B
from machine.telegram import inbox
from tests import chart_fakes as F
from tests.telegram_fakes import TEJAS, FakeTelegram
from tests.test_questions import _test_form, fake_part, sig
from tests.test_referee import signals_for, walk
from tests.test_talker import fake_search

WORDS = "sell when the london high is taken then the break and the first gap, sell at the gap"
WORDS2 = "buy when the asian low is taken then the break and the first gap, buy at the gap"
NOW = dt.datetime(2026, 10, 1, 10, 0).astimezone()
ASK = {"id": "Q1", "kind": "ask", "term": "session high / low", "part": "which session high counts",
       "words": "london high", "readings": [{"id": "A", "means": "London only", "changes": []},
                                            {"id": "B", "means": "any session", "changes": []}]}
ORDER = ["RECEIVED", "SEARCHED", "FORM DRAFTED", "QUESTIONS QUEUED", "ANSWERED", "CONFIRMED & LOCKED", "SHOWN",
         "TUNE RUN", "CHECK OPENED", "REPORTED"]


def wire(**changes) -> dict:
    w = copy.deepcopy(T.EXAMPLE)
    w.update(copy.deepcopy(changes))
    return w


def cache_answer(words: str, w: dict) -> None:
    """The talker's answer, cached as if Claude had answered it once (llm.py): translate() reads it, never calls."""
    _, _, req = T.prepare(words, fake_search())
    llm._save(req, llm.Answer(w, config.TALKER_MODEL, llm.Tokens(), 0.0, False, False, "tool_use"), None)


def boom(*a, **k):
    raise AssertionError("the talker must answer from the cache - no call to Claude")


class Boom:
    def __getattr__(self, name):
        boom()


# ================================================================================================= the fixtures

@pytest.fixture
def home(tmp_path, monkeypatch):
    work = tmp_path / "work"
    monkeypatch.setattr(config, "WORK_DIR", work)
    monkeypatch.setattr(config, "IDEAS_DIR", work / "ideas")
    monkeypatch.setattr(config, "QUESTION_QUEUE_DIR", work / "question-queue")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "SEARCH_FRESH_EXAM", tmp_path / "search-fresh.jsonl")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "talker-cache")
    monkeypatch.setattr(config, "RESEARCH_AGENT", False)    # L10's tune here; the research agent: tests/test_agent.py
    clock = SimpleNamespace(now=NOW)
    monkeypatch.setattr(S, "_now", lambda: clock.now)
    monkeypatch.setattr(store, "_now", lambda: clock.now.isoformat(timespec="seconds"))
    Image.new("RGB", (40, 30), "white").save(tmp_path / "pic.jpg", "JPEG")
    h = SimpleNamespace(tmp=tmp_path, work=work, clock=clock, calls=[], log=[], learned=[])

    def fake_make(idea, market, *, now=None, compile_fn=None, log=print):
        """The SHOWN picture without price data (the real one is tested below): a record like shown.make's."""
        rec = shown.load(idea)
        if rec is not None and rec.get("status") not in ("failed", "no", "withdrawn"):
            return rec
        n = 1
        if rec is not None and rec["status"] in ("no", "withdrawn"):
            shown.archive(idea, rec)
            n = rec["n"] + 1
        path = store.folder(idea) / shown.FILE
        shutil.copy(tmp_path / "pic.jpg", path)
        h.calls.append(f"shown {n}")
        rec = {"idea": idea, "n": n, "qid": shown.qid(idea, n), "status": "made", "replies": [],
               "picture": {"file": str(path), "subject": f"test SHOWN {idea} {n}", "pixel_sha256": f"sh{idea}{n}",
                           "texts": [shown.ASK, *shown.HOW]},
               "setups": [{"n": k, "signal": f"s{n}{k}"} for k in (1, 2, 3)]}
        shown.save(rec)
        return rec

    monkeypatch.setattr(shown, "make", fake_make)
    return h


def hand_tune(h, idea):
    """The tune on hand-made bars, through the real referee (one trial file per version + the tune file)."""
    h.calls.append("tune")
    m1 = walk("2019-03-11", days=2)
    versions = {"v1": signals_for(m1, idea, "v1"), "v2": signals_for(m1, idea, "v2", every=45)}
    return tune.run_versions(versions, tf="5m", m1=m1, about={"v1": "every 30 minutes", "v2": "every 45 minutes"},
                             readings={"v1": {}, "v2": {}}, note=f"test: {idea}")


def tools(h, **over) -> S.Tools:
    def make_part(idea, part, *, market=None, search=None, log=print, **kw):
        h.calls.append(f"questions {part}")
        return fake_part(Q.queue_dir(), idea, part, 2, S._iso(S._now()), tmp=h.tmp)

    def tune_fn(idea, log=print):
        t = hand_tune(h, idea)
        store.set_state(idea, "TUNE RUN", f"test tune: chosen {t.chosen}", waits_for="the check years")
        return t

    def check_fn(idea, *, who, log=print):
        h.calls.append("check")
        chosen = tune.latest(idea)["result"]["chosen"]
        if splits.family_opening(idea) is None:
            splits.open_check(idea, chosen, who=who, why="test: the tune's choice")
            store.set_state(idea, "CHECK OPENED", "test: opened once", waits_for="the judgement")
        m1c = walk("2021-03-08", days=4)
        return judge.judge_check(signals_for(m1c, idea, chosen), tf="5m",
                                 key=prior.key_for_form(store.load_form(idea)), m1=m1c, n_random=8)

    t = S.Tools(search=fake_search(), part_search=fake_search(),
                translate=lambda w, i, search: T.translate(w, i, search=search, runner=boom, api=Boom()),
                load_market=lambda: SimpleNamespace(), make_part=make_part, tune=tune_fn, check=check_fn,
                learn=lambda idea, log=print: h.learned.append(idea),      # the real learner reads the real prices
                log=h.log.append)
    for k, v in over.items():
        setattr(t, k, v)
    return t


def make_bot(h, tg, t: S.Tools) -> B.Bot:
    where = inbox.Where(inbox=h.tmp / "learned-inbox", pictures=h.tmp / "pictures", bot=h.tmp / "bot",
                        outbox=S.outbox_path(), intake=S.queue_path(), questions=Q.queue_dir(),
                        shown=config.SHOWN_PICTURES)
    return B.Bot(tg, TEJAS, where, echo=False, kick=lambda: S.run(t))


def states(idea: str) -> list[str]:
    out = []
    for x in store.read_state(idea)["history"]:
        if not out or out[-1] != x["state"]:
            out.append(x["state"])
    return out


def photos(tg) -> list[dict]:
    return [s for s in tg.sent if s.get("photo")]


def outbox() -> list[dict]:
    return S._read_jsonl(S.outbox_path())


# ================================================================================================= 1. start to end

def test_an_idea_runs_from_his_phone_to_the_report_and_only_his_answers_are_by_hand(home):
    """The prompt's first test: RECEIVED -> REPORTED through the bot, a fake Telegram and the talker's cached answer.
    What Tejas does: 1 idea, 2 letters, 1 yes, 1 YES. What the test does besides: moves the clock a week (the batch)."""
    cache_answer(WORDS, wire(unclear=[ASK]))
    tg = FakeTelegram()
    t = tools(home)
    bot = make_bot(home, tg, t)

    u = tg.text("IDEA " + WORDS)                                 # 1. his idea, from his phone
    bot.poll_once()
    idea = "idea-001"
    assert "The machine starts on it now, by itself." in tg.replies_to(u)[0]
    assert store.read_state(idea)["state"] == "QUESTIONS QUEUED"
    assert (store.folder(idea) / "search.json").is_file() and (store.folder(idea) / "talker-draft.json").is_file()
    q1, q2 = photos(tg)                                          # the weekly batch went at once (none this week)
    assert "idea-001-Q1" in q1["photo"] and any("This week's questions: 2" in (s.get("text") or "")
                                                for s in tg.sent)

    tg.text("A", reply_to=q1["message_id"])                      # 2. his answers
    tg.text("A", reply_to=q2["message_id"])
    bot.poll_once()
    confirm = [s for s in photos(tg) if s["photo"].endswith("confirm-A.jpg")]
    assert len(confirm) == 1
    tg.text("yes", reply_to=confirm[0]["message_id"])            # 3. his yes to the confirm picture
    bot.poll_once()
    s = store.read_state(idea)
    assert s["state"] == "CONFIRMED & LOCKED" and "the weekly batch" in s["waits_for"] and "08 Oct" in s["waits_for"]
    assert home.calls.count("shown 1") == 1 and len(photos(tg)) == 3          # made, not sent: one batch a week

    home.clock.now = NOW + dt.timedelta(days=7, minutes=1)       # a week later: the bot starts the intake by itself
    bot._kicked = 0
    bot.poll_once()
    shown_msg = photos(tg)[-1]
    assert shown_msg["photo"].endswith("shown.jpg") and "Is this your idea?" in shown_msg["caption"]
    assert store.read_state(idea)["state"] == "SHOWN"

    y = tg.text("YES", reply_to=shown_msg["message_id"])         # 4. his YES to the 3 setups
    bot.poll_once()
    assert "The machine tests idea #1 now, by itself." in tg.replies_to(y)[0]
    s = store.read_state(idea)
    assert s["state"] == "REPORTED" and s["waits_for"].startswith("T6")
    assert states(idea) == ORDER                                 # every state, in order, none twice
    assert home.calls == ["questions Q1", "shown 1", "tune", "check"]         # each step once
    assert home.learned == [idea]                                # the learner (L13.1) once, before the report
    report = [x for x in outbox() if x.get("why") == "L10 report"]
    assert len(report) == 1 and any(p["photo"].endswith("report-years.jpg") for p in photos(tg))
    assert (store.folder(idea) / "report.html").is_file()
    opened = json.loads((store.folder(idea) / "check-opened.json").read_text(encoding="utf-8"))
    assert opened["who"].startswith("the intake (L10b), by itself, after Tejas's YES")
    answers = (store.folder(idea) / "answers.md").read_text(encoding="utf-8")
    assert "LOCKED Q1 = A" in answers and "YES" in answers       # his words, word for word
    assert S.run(t) == [] and home.learned == [idea]             # nothing left: never a step twice


def test_a_learner_that_cannot_run_never_blocks_the_report(home):
    """The learner (L13.1) fails - no prices on this PC, say: the reason goes to the log, and the report still goes out
    (it says "works best in / fails in: not measured yet")."""
    hand_tune(home, "idea-901")

    def no_prices(idea, log=print):
        raise FileNotFoundError("data/gold_m1.parquet")

    S._learn("idea-901", tools(home, learn=no_prices))
    assert any("idea-901: the learner did not run (FileNotFoundError: data/gold_m1.parquet)" in x for x in home.log)
    S._learn("idea-902", tools(home))                              # no tune file yet: nothing to learn, no call
    assert home.learned == []


# ================================================================================================= 2. BLOCKED

def test_a_blocked_idea_steps_aside_and_resumes_when_its_block_is_built(home, monkeypatch):
    w = wire()
    w["filters"] = [{"name": "news", "params": [{"name": "only", "value_json": "[\"London\"]"}], "words": None,
                     "sources": ["his words"]}]
    cache_answer(WORDS, w)
    cache_answer(WORDS2, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    S.add(WORDS2)
    S.run(t)
    assert store.read_state("idea-001")["state"] == "BLOCKED"
    assert [r["block"] for r in blocked.read_requests()] == ["news"]
    assert store.read_state("idea-002")["state"] == "QUESTIONS QUEUED"     # the queue went on without it
    lines = S.status_lines()
    assert lines[0].startswith("idea-001 BLOCKED since") and "news (L12" in lines[0]     # blocked.status_lines
    assert lines[1].startswith("idea-002 QUESTIONS QUEUED since 01 Oct (today): waits for")

    monkeypatch.setitem(C.FILTERS, "news", C.FILTERS["session"])           # the block is built now (a stand-in)
    assert blocked.built("news", by="L12") == ["idea-001"]
    S.run(t)
    assert states("idea-001") == ["RECEIVED", "SEARCHED", "BLOCKED", "FORM DRAFTED", "ANSWERED",
                                  "CONFIRMED & LOCKED"]                     # nothing to ask: its picture is made
    assert "the weekly batch" in store.read_state("idea-001")["waits_for"]  # idea-002's questions went today
    assert "its turn (idea-001 runs now)" in S.status_lines()[1]            # one idea runs at a time


def test_the_talkers_own_missing_list_blocks_only_what_is_really_not_built(home):
    """The first real run (29 Sep): the talker listed "session_high" as missing - it is built ("session"). The idea
    went BLOCKED with no block to wait for, for ever. Now only a name that is really not built blocks it."""
    cache_answer(WORDS, wire(missing=[{"name": "session_high", "what": "the London session high"}]))
    cache_answer(WORDS2, wire(missing=[{"name": "close_before_weekend", "what": "close on Friday evening"}]))
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    assert states("idea-001")[:4] == ["RECEIVED", "SEARCHED", "BLOCKED", "FORM DRAFTED"] and not blocked.read_requests()
    assert store.read_state("idea-001")["state"] == "SHOWN"                 # it went on by itself
    store.set_state("idea-001", "REPORTED", "test: done")
    S.add(WORDS2)
    S.run(t)
    assert store.read_state("idea-002")["state"] == "BLOCKED"
    assert [(r["block"], r["idea"]) for r in blocked.read_requests()] == [("close_before_weekend", "idea-002")]


# ================================================================================================= 3. a crash

def test_a_crash_resumes_where_it_stopped_and_repeats_no_step(home, monkeypatch):
    cache_answer(WORDS, wire())
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    rec = shown.load(idea)
    assert store.read_state(idea)["state"] == "SHOWN" and rec["status"] == "sent"
    S.on_reply({"qid": rec["qid"]}, "Yes.", False, source="test")

    real_tune = t.tune

    def tune_then_the_pc_goes_off(i, log=print):
        hand_tune(home, i)                                   # the tune file is written (its tries are counted) ...
        raise RuntimeError("the PC restarted")               # ... and the PC goes off before the state moves

    t.tune = tune_then_the_pc_goes_off
    did = S.run(t)
    assert "stopped in SHOWN" in did[-1] and store.read_state(idea)["state"] == "SHOWN"
    t.tune = real_tune

    moves = []
    real_set = store.set_state

    def crash_on_report(i, state, note="", waits_for=None):              # the report is in the outbox, then a crash
        if state == "REPORTED" and not moves:
            moves.append(state)
            raise RuntimeError("the PC restarted again")
        return real_set(i, state, note, waits_for)

    monkeypatch.setattr(store, "set_state", crash_on_report)
    S.run(t)
    assert store.read_state(idea)["state"] == "CHECK OPENED"
    S.run(t)
    assert store.read_state(idea)["state"] == "REPORTED"
    assert home.calls.count("tune") == 1 and sum(c.startswith("check") for c in home.calls) == 1
    kinds = [json.loads(p.read_text(encoding="utf-8"))["kind"] for p in config.TRIALS_DIR.glob("*.json")]
    assert kinds.count("referee-tune") == 1 and kinds.count("referee") == 1
    assert len([x for x in outbox() if x.get("why") == "L10 report"]) == 1   # sent once
    assert S.run(t) == []


def test_one_run_at_a_time(home):
    t = tools(home)
    S.add(WORDS)
    with S.one_run() as mine:
        assert mine
        assert S.run(t) == [] and "another intake run is working" in home.log[-1]
    assert S.pending()


def test_a_talker_with_no_answer_is_tried_again_then_a_session_looks(home):
    def no_answer(words, idea, search):
        return T.Draft(words, glossary.apply(words), (), None)
    t = tools(home, translate=no_answer)
    S.add(WORDS)
    for n in range(1, S.TALKER_TRIES + 1):
        S.run(t)
        s = store.read_state("idea-001")
        assert s["state"] == "SEARCHED" and s["talker_failures"] == n
    assert "no answer 3 times" in s["needs"]["why"]
    assert any("waits for a Claude session" in x["text"] for x in outbox())
    S.retry("idea-001")
    assert "needs" not in store.read_state("idea-001")


# ================================================================================================= 4. "no" at SHOWN

def test_no_at_shown_goes_back_to_questions_and_a_new_picture_comes_after(home):
    cache_answer(WORDS, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    from machine.questions import interpret
    for q in Q.load(idea, "Q1")["questions"]:
        interpret.on_answer({"qid": q["qid"]}, "B", False, source="test")
    interpret.on_answer({"qid": f"{idea}-Q1-confirm"}, "yes", False, source="test")
    S.run(t)
    home.clock.now = NOW + dt.timedelta(days=8)
    S.run(t)
    rec = shown.load(idea)
    assert store.read_state(idea)["state"] == "SHOWN" and rec["status"] == "sent"

    said = S.on_reply({"qid": rec["qid"]}, "No, 2 is not my idea - the gap is far from the high", False,
                      source="test")
    assert said.startswith("Not tested.")
    s = store.read_state(idea)
    assert s["state"] == "QUESTIONS QUEUED" and "you said NO" in s["needs"]["why"]   # back to questions
    assert store.load_form(idea).status == "QUESTIONS QUEUED"                      # its meaning open again
    assert S.run(t) == []                                     # never guessed: a session reads his words first

    S.retry(idea)                                             # the session: here it changed nothing
    S.run(t)
    assert store.read_state(idea)["state"] == "CONFIRMED & LOCKED"
    assert (store.folder(idea) / "shown-1.json").is_file() and shown.load(idea)["qid"] == f"{idea}-shown-2"
    assert home.calls.count("shown 2") == 1                  # a NEW picture, never the same one again (#117)

    (config.TRIALS_DIR).mkdir(parents=True, exist_ok=True)   # once money is seen, NO cannot reopen it
    (config.TRIALS_DIR / "x.json").write_text(json.dumps({"idea_id": idea, "kind": "tester"}), encoding="utf-8")
    with pytest.raises(store.StoreError, match="money was seen"):
        store.reopen(idea, "test", "test")


def test_the_picture_of_3_setups_goes_before_its_day_only_on_his_words(home):
    cache_answer(WORDS, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    from machine.questions import interpret
    for q in Q.load(idea, "Q1")["questions"]:
        interpret.on_answer({"qid": q["qid"]}, "B", False, source="test")
    interpret.on_answer({"qid": f"{idea}-Q1-confirm"}, "yes", False, source="test")
    S.run(t)
    assert shown.load(idea)["status"] == "made" and S.next_batch() is not None    # made; the weekly day is later
    assert S.batch_step(t) is False                                               # not before its day ...
    with pytest.raises(ValueError, match="his words"):
        S.batch_step(t, early=" ")
    assert S.batch_step(t, early="Tejas, 1 Oct: 'So what you think is best'") is True   # ... unless he says so
    assert shown.load(idea)["status"] == "sent" and store.read_state(idea)["state"] == "SHOWN"
    line = S._read_jsonl(S.work_dir() / "sends.jsonl")[-1]
    assert line["early"].startswith("Tejas, 1 Oct") and line["shown"] == [f"{idea}-shown-1"]
    assert S.next_batch() == S.last_batch() + S.BATCH_EVERY and S.last_batch() == NOW   # the next one: 7 days on


def test_a_picture_he_could_not_judge_is_taken_back_and_a_new_one_made(home):
    """His words, 1 Oct (inbox 1747): not a NO about the idea - the meaning stays locked, no money is seen, the old
    picture is kept apart, a late reply to it changes nothing, and a new picture (a new question id) is made."""
    cache_answer(WORDS, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    from machine.questions import interpret
    for q in Q.load(idea, "Q1")["questions"]:
        interpret.on_answer({"qid": q["qid"]}, "B", False, source="test")
    interpret.on_answer({"qid": f"{idea}-Q1-confirm"}, "yes", False, source="test")
    S.run(t)
    S.batch_step(t, early="test: his words")
    assert shown.load(idea)["status"] == "sent"
    with pytest.raises(shown.ShownError, match="reason"):
        shown.withdraw(idea, " ")
    shown.withdraw(idea, "I can't see the reason to enter so I could not judge it")
    assert store.read_state(idea)["state"] == "CONFIRMED & LOCKED" and not store.money_seen(idea)
    assert S.on_reply({"qid": f"{idea}-shown-1"}, "YES", False, source="test").startswith("Saved.")  # too late
    S.run(t)
    rec = shown.load(idea)
    assert rec["qid"] == f"{idea}-shown-2" and rec["status"] == "made" and home.calls.count("shown 2") == 1
    old = json.loads((store.folder(idea) / "shown-1.json").read_text(encoding="utf-8"))
    assert old["status"] == "withdrawn" and "reason to enter" in old["withdrawn"]["why"]
    rec["status"] = "yes"                                       # once he answered, nothing is taken back
    shown.save(rec)
    with pytest.raises(shown.ShownError, match="nothing to take back"):
        shown.withdraw(idea, "x")


def test_a_reply_that_is_not_yes_or_no_waits_for_a_session(home):
    cache_answer(WORDS, wire())
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    rec = shown.load("idea-001")
    said = S.on_reply({"qid": rec["qid"]}, "yes but 3 is late", False, source="test")
    assert said.startswith("Nothing changes") and store.read_state("idea-001")["state"] == "SHOWN"
    assert S.run(t) == [] and home.calls.count("tune") == 0
    said = S.on_reply({"qid": rec["qid"]}, "YES", False, source="a session read 'yes but 3 is late' as yes")
    assert said.startswith("Thank you")                       # a session's reading, through the same door
    S.run(t)
    assert store.read_state("idea-001")["state"] == "REPORTED"


def test_a_family_that_used_its_check_years_is_judged_by_walk_forward_never_opened_again(home, monkeypatch):
    """CHECK OPENED is once per family: an idea of a family whose check years went to another idea is judged by
    walk-forward inside 2016-2020 (the routing; the walk-forward itself is L5's, tested in test_referee.py)."""
    cache_answer(WORDS, wire())
    walked = []
    t = tools(home, walk_forward=lambda idea, log: walked.append(idea))
    S.add(WORDS)
    S.run(t)
    S.on_reply({"qid": shown.load("idea-001")["qid"]}, "yes", False, source="test")
    monkeypatch.setattr(splits, "family_opening", lambda idea, parent=None: {
        "idea_id": "idea-000", "family": "idea-000", "version_id": "v-x", "opened_at": "2026-09-29"})
    did = S.run(t)
    assert walked == ["idea-001"] and "check" not in home.calls
    assert not (store.folder("idea-001") / "check-opened.json").exists()
    assert "stopped in TUNE RUN" in did[-1]                  # this fake judged nothing: no report without a referee


# ================================================================================================= STATUS

def test_status_one_line_per_idea_with_what_it_waits_for_and_since_when(home):
    cache_answer(WORDS, wire(unclear=[ASK]))
    tg = FakeTelegram()
    bot = make_bot(home, tg, tools(home))
    tg.text("IDEA " + WORDS)
    tg.text("IDEA " + WORDS2)
    bot.poll_once()
    st = tg.text("STATUS")
    bot.poll_once()
    said = tg.replies_to(st)[0]
    assert "idea-001 QUESTIONS QUEUED since 01 Oct (today): waits for Tejas: his answers to the questions" in said
    assert "Queue: 1 new idea waits its turn" in said


# ================================================================================================= 10 in a row

def test_ten_not_works_in_a_row_ask_him_once_and_new_ideas_wait(home, monkeypatch):
    summary = {"answers": 12, "not_works_in_a_row": 10, "ask_tejas": True}
    monkeypatch.setattr(project, "summary", lambda folder=None: dict(summary))
    assert S.direction_check() is True and S.direction_check() is False          # ONE line per streak
    [line] = [x for x in outbox() if x.get("why") == "L10b 10 in a row"]
    assert "keep going, change direction, or stop?" in line["text"] and line["qid"] == "project-direction-12"
    cache_answer(WORDS, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    assert S.run(t) == [] and S.pending()                                         # his decision first
    assert any(x.startswith("New ideas wait") for x in S.status_lines())
    assert S.on_reply({"qid": "project-direction-12"}, "Keep going", False) == \
        "OK. The machine goes on with the next idea."
    assert S.run(t)[0] == "idea-001 RECEIVED from the queue"
    summary.update(answers=25)                                                    # a new streak of 10 asks again
    assert S.direction_check() is True
    assert S.on_reply({"qid": "project-direction-25"}, "stop", False).startswith("Stopped.")
    assert S.paused().startswith("new ideas wait: Tejas said STOP")


# ================================================================================================= the SHOWN picture

def one_setup(m5: pd.DataFrame, i0: int, direction: str = "sell") -> P.Scene:
    """A hand-made setup of idea #1's steps at candle i0: equal highs taken, equal lows below, a gap, the entry."""
    I = m5.index
    w = m5.iloc[i0:i0 + 80]
    hi = float(w["high"].iloc[:30].max()) + 0.4
    lo = float(w["low"].iloc[:75].min()) - 0.4
    highs = {"level_id": f"eh{i0}", "event": "born", "kind": "equal_high", "tf": "5m", "side": "high", "price": hi,
             "first_bar": I[i0 + 5], "touch_bar": I[i0 + 11]}
    lows = {"level_id": f"el{i0}", "event": "born", "kind": "equal_low", "tf": "5m", "side": "low", "price": lo,
            "first_bar": I[i0 + 15], "touch_bar": I[i0 + 55]}
    gap_lo = float(m5["close"].iloc[i0 + 75]) + 0.1
    facts = {"take": {"level_id": f"eh{i0}", "bar": I[i0 + 65], "price": hi},
             "other": {"level_id": f"el{i0}", "price": lo},
             "brk": {"bar": I[i0 + 70], "block_bar": I[i0 + 66], "level": float(m5["low"].iloc[i0 + 66]),
                     "extreme": float(m5["high"].iloc[i0 + 65:i0 + 71].max())},
             "gap": {"lo": gap_lo, "hi": gap_lo + 0.4, "first_bar": I[i0 + 71], "near": gap_lo}}
    s = sig(str(I[i0 + 73]), direction=direction, entry=gap_lo, stop=hi + 1, target=lo, rid=str(i0))
    return P.Scene(end=I[i0 + 80], direction=direction, signals={"A": s}, facts={"A": facts},
                   books={"A": pd.DataFrame([highs, lows])})


def test_the_shown_picture_three_setups_each_chart_ends_at_its_own_entry(tmp_path):
    m5 = F.bars_5m(1000)
    form = _test_form()
    scenes = [one_setup(m5, i0) for i0 in (100, 400, 700)]
    pic = shown.picture(form, scenes, m5, subject="test SHOWN")
    drawn = D.render(pic, tmp_path / "shown.jpg")
    assert lint.check(drawn, shown=tmp_path / "none.jsonl") == []               # every picture rule
    assert len(pic.rows) == 3 and all(len(r) == 1 for r in pic.rows)            # one chart per setup
    for k, (row, sc) in enumerate(zip(pic.rows, scenes), 1):
        assert row[0].bars.index[-1] == sc.end == row[0].entry_at               # each ends at ITS entry candle
        assert row[0].title.startswith(f"{k} · SELL · ")
    assert pic.heading == "IDEA #1 · IS THIS YOUR IDEA?" and pic.question.ask == "Are these 3 your idea?"
    words = " ".join(t.text for t in drawn.texts)
    assert "equal highs · taken" in words and "equal lows · goes here" in words and "S1" in words
    assert not lint.MONEY_WORDS.search(words) and "$" not in words


def test_the_picture_shows_the_reason_to_enter(tmp_path):
    """His words, 1 Oct (inbox 1747): "I can't see the reason to enter so I could not judge it" - idea #2's picture drew
    what was taken, the zone and the gap, but not the break. Now each chart draws the break (a line at the price that
    broke, from his order block candle; an arrow for the breaking move) and the picture says the steps in order."""
    m5 = F.bars_5m(1000)
    form = _test_form()
    scenes = [one_setup(m5, i0) for i0 in (100, 400, 700)]
    pic = shown.picture(form, scenes, m5, subject="test SHOWN reason")
    drawn = D.render(pic, tmp_path / "reason.jpg")
    assert lint.check(drawn, shown=tmp_path / "none.jsonl") == []
    for row, sc in zip(pic.rows, scenes):
        f = sc.facts["A"]["brk"]
        [br] = [i for i in row[0].items if isinstance(i, D.Level) and i.name == shown.BREAK_NAME]
        assert br.price == f["level"] and br.start == f["block_bar"]          # from his order block candle
        [arrow] = [i for i in row[0].items if isinstance(i, D.Arrow)]
        assert arrow.points[-1] == (f["bar"], f["level"]) and arrow.points[0][1] == f["extreme"]
        assert row[0].bars.index[0] <= f["block_bar"]                         # the whole break is on the chart
    words = " ".join(t.text for t in drawn.texts)
    assert 'Why it enters: taken → break → first gap → entry · it aims at "goes here"' in words
    assert shown.BREAK_MEANS in words


def test_the_reason_line_follows_the_ideas_own_steps():
    d = json.loads(_test_form().to_json())
    d["steps"].append({"id": "adr", "what": "the gap sits at the top (red) zone", "block": "premium_discount",
                       "tf": "5m", "max_wait_bars": 0, "params": {}, "sources": [{"ref": "his words"}]})
    d["steps"].append({"id": "liq", "what": "a session level waits", "block": "level_waiting", "tf": "5m",
                       "max_wait_bars": 0, "params": {"where": "target_side", "kinds": ["session"]},
                       "sources": [{"ref": "his words"}]})
    line = shown.why_line(Idea.load(d))
    assert line == 'Why it enters: taken → break → first gap → at the zone → entry · it aims at "goes here"'


def test_the_zone_a_step_found_is_drawn_and_each_chart_starts_before_the_grab(tmp_path):
    """Idea #2's first picture (29 Sep): the day's zone - the heart of his idea - was not drawn, and one chart began
    after the grab. Now a step's zone is its box, and the chart starts before the first step."""
    m5 = F.bars_5m(1000)
    d = json.loads(_test_form().to_json())
    d["steps"].append({"id": "adr", "what": "the gap sits at the top (red) zone", "block": "premium_discount",
                       "tf": "5m", "max_wait_bars": 0, "params": {}, "sources": [{"ref": "his words"}]})
    form = Idea.load(d)
    scenes = []
    for i0 in (100, 400, 700):
        sc = one_setup(m5, i0)
        f = sc.facts["A"]
        f["take"]["bar"] = m5.index[i0 - 30]                    # the grab long before the rest of the story
        f["adr"] = {"zone_lo": f["gap"]["lo"] - 0.2, "zone_hi": f["gap"]["hi"] + 0.6, "known_at": m5.index[i0 - 20]}
        scenes.append(sc)
    pic = shown.picture(form, scenes, m5, subject="test SHOWN zone")
    drawn = D.render(pic, tmp_path / "zone.jpg")
    assert lint.check(drawn, shown=tmp_path / "none.jsonl") == []
    for row, sc in zip(pic.rows, scenes):
        panel = row[0]
        assert panel.bars.index[0] <= sc.facts["A"]["take"]["bar"]                  # the grab is on the chart
        [z] = [i for i in panel.items if isinstance(i, D.Zone)]
        assert z.name == "red zone · top of the day"                                # the whole day, on the chart:
        assert z.start == max(panel.bars.index[0], panel.bars.index[panel.bars.index >= sc.end.normalize()][0])
    assert "red zone · top of the day" in " ".join(t.text for t in drawn.texts)


def test_a_chart_that_goes_past_its_own_entry_breaks_the_rule(tmp_path):
    m5 = F.bars_5m(1000)
    scenes = [one_setup(m5, i0) for i0 in (100, 400, 700)]
    pic = shown.picture(_test_form(), scenes, m5, subject="test SHOWN bad")
    pic.rows[0][0].entry_at = scenes[0].end - pd.Timedelta(minutes=10)           # chart 1 shows 2 candles too many
    problems = lint.check(D.render(pic, tmp_path / "bad.jpg"), shown=tmp_path / "none.jsonl")
    assert problems and all(p.rule == "ends_at_entry" for p in problems)


def test_the_setups_are_spread_early_middle_late():
    sigs = [sig(str(pd.Timestamp("2016-01-04") + pd.Timedelta(days=7 * k)), rid=str(k)) for k in range(90)]
    order = shown.spread(sigs, n_tries=9)
    t = [pd.Timestamp(s.known_at) for s in (order[0], order[3], order[6])]      # the first try of each third
    third = [pd.Timestamp(sigs[k].known_at) for k in (30, 60)]
    assert t[0] < third[0] <= t[1] < third[1] <= t[2]                          # early, middle, late
    assert len(order) == 9 and len({s.id for s in order}) == 9


@pytest.mark.data
def test_the_real_shown_picture_of_idea_1(tmp_path, monkeypatch):
    """The real thing on 2016-2020 (💻): idea #1's form, compiled with its first readings, 3 setups, every rule."""
    monkeypatch.setattr(config, "IDEAS_DIR", tmp_path / "ideas")
    d = json.loads((config.ROOT / "plan/work/ideas/idea-001/form.json").read_text(encoding="utf-8"))
    form = Idea.load(d | {"status": "CONFIRMED & LOCKED"})
    store.create(form)
    market = C.Market.load("2016", "2020")
    rec = shown.make("idea-001", market, log=lambda s: None)
    assert rec["status"] == "made" and len(rec["setups"]) == 3
    assert len({(s["end"][:4], s["direction"]) for s in rec["setups"]}) == 3
    assert rec["picture"]["lint"].startswith("all ") and (tmp_path / "ideas/idea-001/shown.jpg").is_file()
    assert not any(y in s["end"] for s in rec["setups"] for y in ("2021", "2022", "2023"))   # tune years only
