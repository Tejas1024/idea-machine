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
from machine.referee import judge, prior, project, splits, trials, tune
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
         "TUNE RUN", "CHECK OPENED", "REPORTED", "DECIDED"]


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
    monkeypatch.setattr(config, "TELEGRAM_DIR", tmp_path / "bot")     # the bot's log, where make_bot writes it - never
                                                                      # the real data/telegram (his real messages)
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
        if rec is not None and rec.get("status") not in ("failed", "no", "withdrawn") and not shown.stale(rec):
            return rec
        n = 1
        if rec is not None and rec["status"] in ("no", "withdrawn"):
            shown.archive(idea, rec)
            n = rec["n"] + 1
        elif shown.stale(rec):                                  # never sent: the same question, drawn again
            n = rec["n"]
        path = store.folder(idea) / shown.FILE
        shutil.copy(tmp_path / "pic.jpg", path)
        h.calls.append(f"shown {n}")
        rec = {"idea": idea, "n": n, "qid": shown.qid(idea, n), "status": "made", "replies": [],
               "drawing": shown.DRAWING,
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


HEDGED = {"trades": 16, "net_inr_total": -11582.52, "worst_year": {"year": 2018, "net_inr": -8940.02, "trades": 7}}
HEDGED_R = {"runs": 200, "boxes": [{}, {}, {"total": -11766.0}, {}], "his_beats_random": {"with_hedging": 104}}


def fake_parked(h, fail: Exception | None = None):
    """The test with his hedge (machine.strategy.parked.run) without price data: what it returns, and its section
    added to decision.md. The numbers are idea #2's real ones (2 Oct); the real run reads 2016-2020 gold."""
    from machine.strategy import parked as PK

    def run(idea, log=print):
        h.calls.append(f"parked {idea}")
        if fail is not None:
            raise fail
        head, means = PK.choice(HEDGED_R, HEDGED)
        with open(store.folder(idea) / S.DECISION, "a", encoding="utf-8") as fh:
            fh.write(f"\n## L19d - tested with his hedge (test)\n\n**The machine's choice: {head}** - {means}\n")
        return {"idea": idea, "his_hedging": HEDGED, "result": HEDGED_R, "choice": head, "means": means,
                "his_stop": {"trades": 448, "net_inr_total": -21661.0}, "trial": "20261008-183000-test.json"}
    return run


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
                parked=fake_parked(h), log=h.log.append)
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
    """The prompt's first test: RECEIVED -> REPORTED -> DECIDED through the bot, a fake Telegram and the talker's cached
    answer. What Tejas does: 1 idea, 2 letters, 1 yes, and "All 3 correct" (his own words for idea #2's 3 setups,
    1 Oct). What the test does besides: moves the clock a week (the batch). After the report the machine decides and
    TELLS him (LEARNED #162) - no question. It lost with a plain stop (as ideas #1 and #2), so in the same run the
    machine tests it with his hedge (L19d) and tells him that too - no session (idea #2's needed one, 2 Oct)."""
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
    assert home.calls.count("shown 1") == 1 and len(photos(tg)) == 4          # made and sent at once (#175), though
    shown_msg = photos(tg)[-1]                                                 # a weekly batch went today
    assert shown_msg["photo"].endswith("shown.jpg") and "Is this your idea?" in shown_msg["caption"]
    assert "counts as 1 question" not in shown_msg["caption"]
    assert store.read_state(idea)["state"] == "SHOWN"
    assert S.next_batch() == NOW + S.BATCH_EVERY                 # the picture moved no weekly day

    y = tg.text("All 3 correct", reply_to=shown_msg["message_id"])   # 4. his YES to the 3 setups, his words
    bot.poll_once()
    assert "The machine tests idea #1 now, by itself." in tg.replies_to(y)[0]
    s = store.read_state(idea)
    [ans] = [f["result"]["answer"] for f in trials.load() if f.get("kind") == "referee"]
    word = ans["word"]
    rest = all(c["passed"] for c in ans["checks"] if c["name"] != "not already seen")
    head, _, waits = S.choice(word, hedged=False, rest_passed=rest)
    assert waits == S.PARK                                       # the plan's default for a plain-stop loss: parked
    assert s["state"] == "DECIDED" and s["waits_for"] == "nothing (kept on file: tested with his hedge too)"
    assert states(idea) == ORDER                                 # every state, in order, none twice
    assert [x["note"].split(":")[0] for x in s["history"][-2:]] == ["the machine decided (LEARNED #162)",
                                                                    "tested with his hedge by itself (L19d, 20261008-"
                                                                    "183000-test.json)"]
    [told] = [x for x in outbox() if x.get("why") == "L10b decided"]
    assert told["text"].startswith(f"Idea #1 came out {word.upper()}. The machine's choice: {head}.")
    assert "?" not in told["text"] and "You don't need to do anything." in told["text"]    # tells, never asks
    assert "Now the machine tests the same rule with your hedge" in told["text"]
    [hedged] = [x for x in outbox() if x.get("why") == "L10b tested with his hedge"]
    assert "KEPT ON FILE" in hedged["text"] and "?" not in hedged["text"]
    decision = (store.folder(idea) / S.DECISION).read_text(encoding="utf-8")
    assert "I don't know why u asking these questions" in decision and f"**{head}**" in decision
    assert "the referee tested the hedge point as a plain stop" in decision         # how it was tested decides
    assert f"-{idea}-" in decision.split("the referee's file:")[1]                   # its file named
    assert decision.count("## L19d - tested with his hedge") == 1
    print(decision, told["text"], hedged["text"], sep="\n----\n")
    assert home.calls == ["questions Q1", "shown 1", "tune", "check", f"parked {idea}"]     # each step once
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
                                  "CONFIRMED & LOCKED", "SHOWN"]            # nothing to ask: its picture made and
    assert "YES or NO" in store.read_state("idea-001")["waits_for"]         # sent at once, though idea-002's
    assert "its turn (idea-001 runs now)" in S.status_lines()[1]            # questions went today (#175)


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

    def crash_on_report(i, state, note="", waits_for=None):              # the report is in the outbox, then a crash;
        if state in ("REPORTED", "DECIDED") and state not in moves:      # the choice is told and written, then one
            moves.append(state)
            raise RuntimeError("the PC restarted again")
        return real_set(i, state, note, waits_for)

    monkeypatch.setattr(store, "set_state", crash_on_report)
    S.run(t)
    assert store.read_state(idea)["state"] == "CHECK OPENED"
    S.run(t)
    assert store.read_state(idea)["state"] == "REPORTED" and (store.folder(idea) / S.DECISION).is_file()
    S.run(t)
    assert store.read_state(idea)["state"] == "DECIDED" and moves == ["REPORTED", "DECIDED"]
    assert len([x for x in outbox() if x.get("why") == "L10b decided"]) == 1      # told once
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


def test_his_usage_limit_at_the_talker_waits_and_goes_on_by_itself(home):
    """His Claude plan's usage limit resets by itself: never 'a session looks' (the idea would stay stuck after the
    limit was over). ONE line tells him; the form is written at the first run after the limit."""
    limit = ("claude -p: You've hit your session limit · resets 12:10am - his plan's usage limit: run again when it "
             "resets (answers so far are cached)")

    def at_the_limit(words, idea, search):
        return T.Draft(words, glossary.apply(words), (),
                       llm.Answer(None, config.TALKER_MODEL, llm.Tokens(), 0.0, False, False, "error", limit))
    S.add(WORDS)
    t = tools(home, translate=at_the_limit)
    for _ in range(S.TALKER_TRIES * 3):
        S.run(t)
    s = store.read_state("idea-001")
    assert s["state"] == "SEARCHED" and "needs" not in s and "talker_failures" not in s
    assert "usage limit" in s["waits_for"] and "goes on by itself" in s["waits_for"]
    assert sum("usage limit" in h["note"] for h in s["history"]) == 1          # one line for the whole wait
    told = [x for x in outbox() if x.get("why") == "L10b waits for his usage limit"]
    assert len(told) == 1 and "Idea #1 waits a little" in told[0]["text"]
    assert not any("waits for a Claude session" in x["text"] for x in outbox())
    cache_answer(WORDS, wire())                                                 # the limit is over
    S.run(tools(home))
    s = store.read_state("idea-001")
    assert S.STATES.index(s["state"]) > S.STATES.index("SEARCHED") and "limit_since" not in s


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
    assert store.read_state(idea)["state"] == "SHOWN"         # the new picture went at once (#175)
    assert (store.folder(idea) / "shown-1.json").is_file() and shown.load(idea)["qid"] == f"{idea}-shown-2"
    assert home.calls.count("shown 2") == 1                  # a NEW picture, never the same one again (#117)

    (config.TRIALS_DIR).mkdir(parents=True, exist_ok=True)   # once money is seen, NO cannot reopen it
    (config.TRIALS_DIR / "x.json").write_text(json.dumps({"idea_id": idea, "kind": "tester"}), encoding="utf-8")
    with pytest.raises(store.StoreError, match="money was seen"):
        store.reopen(idea, "test", "test")


def test_the_picture_of_3_setups_goes_when_it_is_made_and_moves_no_weekly_day(home):
    """His words, 7 Oct (LEARNED #175): what holds up other tasks goes now. The picture of 3 setups holds up its idea
    and every idea behind it - 5 of 5 pictures before 8 Oct went early by hand (send-now)."""
    cache_answer(WORDS, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    week = S.next_batch()
    assert week == NOW + S.BATCH_EVERY                                            # the L9b questions went today
    from machine.questions import interpret
    for q in Q.load(idea, "Q1")["questions"]:
        interpret.on_answer({"qid": q["qid"]}, "B", False, source="test")
    interpret.on_answer({"qid": f"{idea}-Q1-confirm"}, "yes", False, source="test")
    home.clock.now = NOW + dt.timedelta(days=2)
    S.run(t)
    assert shown.load(idea)["status"] == "sent" and store.read_state(idea)["state"] == "SHOWN"   # not on its day:
    line = S._read_jsonl(S.work_dir() / "sends.jsonl")[-1]                                      # at once
    assert line["at_once"] == S.AT_ONCE and "early" not in line and line["with_l9b"] == []
    assert line["shown"] == [f"{idea}-shown-1"]
    [told] = [x for x in outbox() if x.get("qid") == f"{idea}-shown-1"]
    assert "counts as 1 question" not in told["text"]
    assert f"{idea}: the picture of 3 setups sent ({idea}-shown-1, at once when made, #175)" in home.log
    assert "weekly batch sent" not in home.log                                    # the log says what went
    assert S.next_batch() == week                                                 # the weekly day did not move
    assert S.batch_step(t) is False                                               # nothing else waits
    with pytest.raises(ValueError, match="his words"):
        S.batch_step(t, early=" ")


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
    assert shown.load(idea)["status"] == "sent"                 # sent at once (#175)
    with pytest.raises(shown.ShownError, match="reason"):
        shown.withdraw(idea, " ")
    shown.withdraw(idea, "I can't see the reason to enter so I could not judge it")
    assert store.read_state(idea)["state"] == "CONFIRMED & LOCKED" and not store.money_seen(idea)
    assert S.on_reply({"qid": f"{idea}-shown-1"}, "YES", False, source="test").startswith("Saved.")  # too late
    S.run(t)
    rec = shown.load(idea)
    assert rec["qid"] == f"{idea}-shown-2" and rec["status"] == "sent" and home.calls.count("shown 2") == 1
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
    assert store.read_state("idea-001")["state"] == "DECIDED"


@pytest.mark.parametrize("words", ["All 3 correct", "all correct", "Yes, all 3 are correct.", "all three right",
                                   "yes all 3", "Correct, all 3", "all are fine", "all 3 my idea", "Yes", "ok"])
def test_his_own_words_for_yes_to_the_3_setups(words):
    """His real reply to idea #2's 3 setups was "All 3 correct" (1 Oct; a session had to read it). The picture says
    "YES = all 3 are how you trade it": a line that says only that is his YES - read by code, no session."""
    from machine.questions.interpret import read_confirm
    assert S.SHOWN_YES.match(words) or read_confirm(words, False) == "yes"


@pytest.mark.parametrize("words", ["3 correct", "2 is correct", "all 3 correct but 2 is late", "not all correct",
                                   "all wrong", "all 3 no", "all", "correct 1 and 2", "yes but 3 is late"])
def test_words_that_are_not_a_plain_yes_wait_for_a_session(words):
    """"3 correct" may mean setup 3 only; anything more than the yes is read by a session, never guessed (R13)."""
    from machine.questions.interpret import read_confirm
    assert not S.SHOWN_YES.match(words) and read_confirm(words, False) != "yes"


# ================================================================================================= his YES on its own

def shown_and_sent(home) -> tuple[S.Tools, str, dict]:
    """An idea with nothing unclear: the first run makes its picture and the batch sends it at once (none this week)."""
    cache_answer(WORDS, wire())
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    rec = shown.load(idea)
    assert rec["status"] == "sent" and store.read_state(idea)["state"] == "SHOWN"
    return t, idea, rec


def his_message(words: str, *, at: dt.datetime, mid: int = 901, kind: str = "lesson", reply_to: int | None = None,
                photo: str | None = None, edited: bool = False) -> None:
    """One line of the bot's own log (machine/telegram/bot.py), as the bot writes it for a message he typed."""
    S._append_jsonl(S.messages_path(), {
        "update_id": 5000 + mid, "message_id": mid, "sent": at.isoformat(timespec="seconds"),
        "filed": at.isoformat(timespec="seconds"), "kind": kind, "why": "test", "command": None, "words": words,
        "photo_id": photo, "reply_to_id": reply_to, "edited": edited, "other": "", "question": None,
        "inbox": f"strategy/learned-inbox/test-m{mid}.md", "picture": None})


def test_his_yes_typed_on_its_own_is_his_answer_to_the_picture(home):
    """His way on Telegram: none of his 9 messages of 29 Sep - 4 Oct 2026 was a reply. The bot files a plain "Yes"
    as a lesson (as before); the intake's next run (the bot starts it every 30 minutes) finds it in the bot's log and
    reads it through the same door as a reply - the test runs, ONE line tells him how his message was read."""
    cache_answer(WORDS, wire())
    tg = FakeTelegram()
    t = tools(home)
    bot = make_bot(home, tg, t)
    tg.text("IDEA " + WORDS)
    bot.poll_once()
    idea = "idea-001"
    rec = shown.load(idea)
    assert rec["status"] == "sent" and store.read_state(idea)["state"] == "SHOWN"

    home.clock.now = NOW + dt.timedelta(minutes=20)
    y = tg._update({"text": "Yes", "date": int(home.clock.now.timestamp())})     # typed on its own: no reply
    bot.poll_once()
    assert tg.replies_to(y)[0].startswith("Saved as a lesson.")                 # the bot: nothing changed there
    assert shown.load(idea)["status"] == "sent"                                 # a lesson does not start the intake

    home.clock.now = NOW + dt.timedelta(minutes=30)
    bot._kicked = 0                                                              # the bot's 30-minute start
    bot.poll_once()
    rec = shown.load(idea)
    assert rec["status"] == "yes" and rec["replies"][-1]["kind"] == "yes"
    assert "typed on its own (not a reply to the picture)" in rec["replies"][-1]["source"]
    s = store.read_state(idea)
    assert s["shown_yes"]["words"] == "Yes" and s["state"] == "DECIDED"         # the test ran, by itself
    assert home.calls.count("tune") == 1
    [told] = [x for x in outbox() if x.get("why") == "L10b SHOWN: his YES typed on its own"]
    assert told["text"].startswith('Your message "Yes" (10:20) is taken as your answer to the picture of 3 setups '
                                   "(idea #1).")
    assert "Thank you. The machine tests idea #1 now, by itself." in told["text"] and "?" not in told["text"]
    answers = (store.folder(idea) / "answers.md").read_text(encoding="utf-8")
    assert "Yes" in answers and "typed on its own" in answers                   # his words, word for word
    [lesson] = [p for p in (home.tmp / "learned-inbox").rglob("*.md") if "> Yes" in p.read_text(encoding="utf-8")]
    said = lesson.read_text(encoding="utf-8")                                   # nothing lost, and it says how it
    assert said.startswith("# Lesson") and "read by the intake" in said and "idea-001-shown-1" in said  # was read
    assert S.run(t) == [] and len([x for x in outbox() if "plain-yes" in x["id"]]) == 1     # once
    size = (store.folder(idea) / "answers.md").stat().st_size
    assert S.plain_yes(idea, shown.load(idea)) is False                         # answered: never read twice
    assert (store.folder(idea) / "answers.md").stat().st_size == size


@pytest.mark.parametrize("words", ["All 3 ok", "all 3 are okay", "All 3 correct"])
def test_his_all_3_ok_typed_on_its_own_is_his_yes(home, words):
    """The bot files "All 3 ok" as a lesson even while label charts wait (labels.replies._ALL_N, tests/test_labels.py):
    the intake reads it as his YES to the picture of 3 setups."""
    t, idea, rec = shown_and_sent(home)
    home.clock.now = NOW + dt.timedelta(minutes=30)
    his_message(words, at=NOW + dt.timedelta(minutes=5))
    S.run(t)
    assert shown.load(idea)["status"] == "yes" and home.calls.count("tune") == 1


@pytest.mark.parametrize("words", ["1,2,3\nAll yes", "1,2,3 all yes", "1 2 3\nyes", "3,1,2\nAll three setups are good",
                                   "All three setups are good", "All yes", "yes to all 3"])
def test_his_list_of_all_the_setups_with_a_yes_typed_on_its_own_is_his_yes(home, words):
    """His answer to idea #4's second picture (8 Oct 2026, Telegram message 1247, typed on its own): "1,2,3" on one
    line and "All yes" on the next - every setup listed and a yes. Read by code, no session (L10b closes only on an
    idea that needs none)."""
    t, idea, rec = shown_and_sent(home)
    assert [int(x["n"]) for x in rec["setups"]] == [1, 2, 3]
    home.clock.now = NOW + dt.timedelta(minutes=30)
    his_message(words, at=NOW + dt.timedelta(minutes=5))
    S.run(t)
    assert shown.load(idea)["status"] == "yes" and home.calls.count("tune") == 1


@pytest.mark.parametrize("words", ["1,2,3", "1,2\nAll yes", "1,2,3\nno", "1,2,3\nAll yes but 2 is late", "1,2,3,4\nyes",
                                   "1,2 all yes", "all 3 no", "2\nyes"])
def test_a_list_that_leaves_one_out_or_has_no_yes_is_not_his_yes(words):
    """A plain list alone stays for the zone charts (never read for a picture); a list that leaves a setup out, a
    number the picture does not have, or more words than a yes: a session reads it (R13)."""
    assert not S.listed_yes(words, [1, 2, 3]) and not S.SHOWN_YES.match(words) and not S.PLAIN_YES.match(words)


@pytest.mark.parametrize("words", ["ok", "Okay", "correct", "Yes but 3 is late", "yes, test it", "No", "3 correct",
                                   "yes 2"])
def test_a_message_on_its_own_that_is_not_only_a_yes_stays_a_lesson(home, words):
    """"ok" or "correct" on its own is often "got it" (to the day's report, say) - only a yes-word, or his "all 3
    correct", is his YES. Anything more is read by a session, as for a reply."""
    t, idea, rec = shown_and_sent(home)
    home.clock.now = NOW + dt.timedelta(minutes=30)
    his_message(words, at=NOW + dt.timedelta(minutes=5))
    S.run(t)
    assert shown.load(idea)["status"] == "sent" and store.read_state(idea)["state"] == "SHOWN"
    assert not [x for x in outbox() if "plain-yes" in x["id"]] and home.calls.count("tune") == 0


def test_a_yes_that_is_a_photo_edited_a_reply_too_early_or_not_yet_is_never_taken(home):
    t, idea, rec = shown_and_sent(home)
    home.clock.now = NOW + dt.timedelta(minutes=30)
    his_message("Yes", at=NOW - dt.timedelta(minutes=1), mid=901)                # before the picture went
    his_message("Yes", at=NOW + dt.timedelta(minutes=2), mid=902, photo="p1")    # a picture with the word
    his_message("Yes", at=NOW + dt.timedelta(minutes=3), mid=903, edited=True)   # an edited message
    his_message("Yes", at=NOW + dt.timedelta(minutes=4), mid=904, kind="answer", reply_to=77)  # a reply: on_reply's
    his_message("Yes", at=NOW + dt.timedelta(minutes=40), mid=905)               # after the run's own time
    S._append_jsonl(S.messages_path(), {"message_id": 906, "kind": "lesson", "words": "Yes", "sent": "2026-10-01T10:05"})
    S.run(t)                                                                     # (906: a time with no time zone)
    assert shown.load(idea)["status"] == "sent" and not [x for x in outbox() if "plain-yes" in x["id"]]
    home.clock.now = NOW + dt.timedelta(minutes=41)                              # 905 is in the past now
    S.run(t)
    rec = shown.load(idea)
    assert rec["status"] == "yes" and "Telegram message 905" in rec["replies"][-1]["source"]


def test_a_yes_on_its_own_while_another_question_waits_asks_him_to_reply_and_is_never_taken_later(home):
    t, idea, rec = shown_and_sent(home)
    S.direction_path().parent.mkdir(parents=True, exist_ok=True)
    S.direction_path().write_text(json.dumps({"status": "asked", "streak": 10, "asked": S._iso(NOW)}),
                                  encoding="utf-8")                              # the 10-in-a-row question waits too
    home.clock.now = NOW + dt.timedelta(minutes=30)
    his_message("YES", at=NOW + dt.timedelta(minutes=5))
    S.run(t)
    S.run(t)
    assert shown.load(idea)["status"] == "sent" and shown.load(idea)["unclear_yes"] == [901]
    [told] = [x for x in outbox() if "plain-yes" in x["id"]]                     # ONE line, not one per run
    assert told["why"] == "L10b SHOWN: his YES typed on its own, not clear"
    assert "Please reply to the picture you mean: press on it, tap Reply, then type YES." in told["text"]
    assert "the question about going on after 10 ideas in a row" in told["text"]
    S.direction_path().unlink()                                                  # the other question is gone ...
    S.run(t)
    assert shown.load(idea)["status"] == "sent" and home.calls.count("tune") == 0   # ... his unclear YES stays unclear
    his_message("Yes", at=NOW + dt.timedelta(minutes=25), mid=902)              # a new YES, now clear
    S.run(t)
    assert shown.load(idea)["status"] == "yes" and home.calls.count("tune") == 1


# ================================================================================================= his list (#163)

@pytest.mark.parametrize("words, listed, left", [
    ("1,2,3.", [1, 2, 3], []), ("1,2,3", [1, 2, 3], []), ("1 2 3", [1, 2, 3], []), ("1.2.3.", [1, 2, 3], []),
    ("3,1,2", [3, 1, 2], []), ("2 3 1", [2, 3, 1], []), ("1, 2 and 3", [1, 2, 3], []), ("1\n2\n3", [1, 2, 3], []),
    ("1,3", [1, 3], [2]), ("3 1", [3, 1], [2]), ("2", [2], [1, 3])])
def test_his_list_of_setup_numbers(words, listed, left):
    """LEARNED #163, his zone charts: "Below are my zones, which I did not mentioned are not mine". His reply to idea
    #3's picture was "1,2,3." (7 Oct, message 1033; a session had to read it): the numbers he lists are his."""
    assert S.setup_list(words, [1, 2, 3]) == {"listed": listed, "left_out": left}


@pytest.mark.parametrize("words", ["1,2,3,4", "4", "1,2,3 but 2 is late", "1,2,3.\nbut 2 is late", "123", "1-3",
                                   "all 3", "All 3 ok", "0", "1 yes 2 yes 3 yes", "1,2,3 no", "No, 2", "yes", "",
                                   "setup 1 2 3 are mine"])
def test_more_than_a_list_of_its_setup_numbers_is_no_list(words):
    """A number the picture does not have, other words (on any line), a yes / no word: a session reads it."""
    assert S.setup_list(words, [1, 2, 3]) is None


def _bot_with_zone_charts(home, tg, t) -> tuple[B.Bot, inbox.Where]:
    """make_bot, with 3 zone charts of 1 Oct waiting for his answer (L15.1) - as on 7 Oct, when idea #3's picture
    went with the day's 10 zone charts."""
    from machine.labels import replies as LR
    where = inbox.Where(inbox=home.tmp / "learned-inbox", pictures=home.tmp / "pictures", bot=home.tmp / "bot",
                        outbox=S.outbox_path(), intake=S.queue_path(), questions=Q.queue_dir(),
                        shown=config.SHOWN_PICTURES, labels=home.tmp / "labels")
    path = LR.charts_file("strong_zone", where.labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"qid": LR.qid("strong_zone", dt.date(2026, 10, 1), k), "day": "2026-10-01",
                                        "n": k}) + "\n" for k in (1, 2, 3)), encoding="utf-8")
    assert LR.newest_waiting(where.labels) is not None
    return B.Bot(tg, TEJAS, where, echo=False, kick=lambda: S.run(t)), where


def test_his_reply_listing_all_3_setups_is_his_yes_through_the_bot(home):
    """His reply "1,2,3." to the picture: the bot (this code) reads it as YES and the test runs by itself - even while
    zone charts wait (a REPLY goes to its picture; only a plain list goes to the zone charts)."""
    from machine.labels import replies as LR
    cache_answer(WORDS, wire())
    tg = FakeTelegram()
    t = tools(home)
    bot, where = _bot_with_zone_charts(home, tg, t)
    tg.text("IDEA " + WORDS)
    bot.poll_once()
    idea = "idea-001"
    shown_msg = photos(tg)[-1]
    assert shown_msg["photo"].endswith("shown.jpg") and shown.load(idea)["status"] == "sent"

    y = tg.text("1,2,3.", reply_to=shown_msg["message_id"])
    bot.poll_once()
    said = tg.replies_to(y)[0]
    assert ('Your list "1,2,3." names all 3 setups: that is YES (your rule: the numbers you list are yours).\n'
            "Thank you. The machine tests idea #1 now, by itself.") in said
    rec = shown.load(idea)
    assert rec["status"] == "yes" and (rec["replies"][-1]["listed"], rec["replies"][-1]["left_out"]) == ([1, 2, 3], [])
    s = store.read_state(idea)
    assert s["state"] == "DECIDED" and s["shown_yes"]["listed"] == [1, 2, 3] and home.calls.count("tune") == 1
    assert any("LEARNED #163" in x["note"] for x in s["history"] if x["note"].startswith("his YES"))
    assert LR.read_replies("strong_zone", where.labels) == []          # the zone charts got nothing from it
    assert not [x for x in outbox() if x["id"].endswith("-list-read")]  # read by the bot: no second line


def test_a_plain_list_of_numbers_is_never_his_answer_to_the_picture(home):
    """Not a reply: while zone charts wait the bot takes it for them (bot._label_list); with none waiting it is a lesson
    - and the intake never reads it as the picture's answer (plain_yes takes only a plain yes)."""
    from machine.labels import replies as LR
    cache_answer(WORDS, wire())
    tg = FakeTelegram()
    t = tools(home)
    bot, where = _bot_with_zone_charts(home, tg, t)
    tg.text("IDEA " + WORDS)
    bot.poll_once()
    idea = "idea-001"
    home.clock.now = NOW + dt.timedelta(minutes=10)
    tg._update({"text": "1,2,3.", "date": int(home.clock.now.timestamp())})      # typed on its own
    bot.poll_once()
    assert [r["label"] for r in LR.read_replies("strong_zone", where.labels)] == ["yes", "yes", "yes"]
    home.clock.now = NOW + dt.timedelta(minutes=40)
    bot._kicked = 0
    bot.poll_once()
    assert shown.load(idea)["status"] == "sent" and home.calls.count("tune") == 0

    his_message("1 2 3", at=NOW + dt.timedelta(minutes=35), mid=950)             # a lesson (no zone chart waits)
    S.run(t)
    assert shown.load(idea)["status"] == "sent" and home.calls.count("tune") == 0
    assert not [x for x in outbox() if "plain-yes" in x["id"] or x["id"].endswith("-list-read")]


@pytest.mark.parametrize("words, out, verb", [("1,3", "setup 2", "is"), ("2", "setups 1 and 3", "are")])
def test_a_list_that_leaves_a_setup_out_is_his_no_for_it(home, words, out, verb):
    t, idea, rec = shown_and_sent(home)
    said = S.on_reply({"qid": rec["qid"]}, words, False, source="test")
    assert said.startswith(f'Your list "{words}" leaves out {out}. By your rule (the numbers you list are yours), '
                           f"{out} {verb} not your idea: that is a NO.\nNot tested.")
    s = store.read_state(idea)
    assert s["state"] == "QUESTIONS QUEUED" and f"leaves out {out}" in s["needs"]["why"]   # back to questions
    assert shown.load(idea)["status"] == "no" and shown.load(idea)["replies"][-1]["listed"] == [int(n) for n in
                                                                                             words.split(",")]
    assert S.run(t) == [] and home.calls.count("tune") == 0      # never guessed: a session reads it and asks again


def old_bot_reply(monkeypatch, rec: dict, words: str, photo: bool = False) -> None:
    """His reply read by the bot's OWN code - the one it started with (6 Oct 07:58, 19620a0), before this fix: a list
    of setup numbers is words a session reads ("needs a session"), as idea #3's "1,2,3." on 7 Oct 05:53."""
    with monkeypatch.context() as m:
        m.setattr(S, "setup_list", lambda *a, **k: None)
        said = S.on_reply({"qid": rec["qid"]}, words, photo, source="strategy/learned-inbox/test-m1033.md")
    assert said.startswith("Nothing changes: only a plain YES starts the test.")


def test_his_list_the_bots_older_code_left_for_a_session_is_read_by_the_intakes_next_run(home, monkeypatch):
    """The running bot keeps its old reading until it restarts; the intake is a new program every run (the bot starts
    one right after his answer), so it reads the picture's one reply again: all 3 listed = his YES, ONE line."""
    t, idea, rec = shown_and_sent(home)
    home.clock.now = NOW + dt.timedelta(minutes=5)
    old_bot_reply(monkeypatch, rec, "1,2,3.")
    s = store.read_state(idea)
    assert s["state"] == "SHOWN" and "not a plain YES or NO" in s["needs"]["why"]   # idea #3 on 7 Oct, 05:53
    home.clock.now = NOW + dt.timedelta(minutes=6)
    S.run(t)
    rec = shown.load(idea)
    assert rec["status"] == "yes" and len(rec["replies"]) == 1 and rec["replies"][0]["kind"] == "yes"
    assert rec["replies"][0]["read_again"]["by"].startswith("the intake") and rec["replies"][0]["listed"] == [1, 2, 3]
    s = store.read_state(idea)
    assert s["state"] == "DECIDED" and not s.get("needs") and home.calls.count("tune") == 1   # the test ran
    [told] = [x for x in outbox() if x["id"] == f"{rec['qid']}-list-read"]
    assert told["text"].startswith("Idea #1, the picture of 3 setups: your reply at 10:05 is read again now.\n"
                                   'The bot said "only a plain YES starts the test". That was wrong: its old code '
                                   "did not know your rule.\n"
                                   'Your list "1,2,3." names all 3 setups: that is YES')
    assert "Thank you. The machine tests idea #1 now, by itself." in told["text"] and "?" not in told["text"]
    assert S.run(t) == [] and len([x for x in outbox() if x["id"].endswith("-list-read")]) == 1   # once


def test_a_list_the_bots_older_code_left_that_leaves_a_setup_out_is_his_no(home, monkeypatch):
    t, idea, rec = shown_and_sent(home)
    old_bot_reply(monkeypatch, rec, "1 3")
    assert S.run(t) == ["idea-001: SHOWN -> QUESTIONS QUEUED"]
    s = store.read_state(idea)
    assert "leaves out setup 2" in s["needs"]["why"] and shown.load(idea)["status"] == "no"
    [told] = [x for x in outbox() if x["id"].endswith("-list-read")]
    assert "That was wrong" not in told["text"] and "setup 2 is not your idea: that is a NO." in told["text"]
    assert S.run(t) == [] and home.calls.count("tune") == 0


@pytest.mark.parametrize("replies, photo", [(["1,2,3,4"], False), (["yes but 3 is late"], False),
                                            (["1,2,3.", "wait, not 2"], False), (["1,2,3"], True)])
def test_the_intake_reads_again_only_a_pictures_one_reply_that_is_a_list_of_words(home, monkeypatch, replies, photo):
    """Two replies (he said more), a photo, words that are not only such a list: a session reads them, as before."""
    t, idea, rec = shown_and_sent(home)
    for w in replies:
        old_bot_reply(monkeypatch, shown.load(idea), w, photo)
    S.run(t)
    assert shown.load(idea)["status"] == "needs a session" and store.read_state(idea)["state"] == "SHOWN"
    assert store.read_state(idea).get("needs") and home.calls.count("tune") == 0
    assert not [x for x in outbox() if x["id"].endswith("-list-read")]


def test_after_the_report_the_machine_chooses_by_the_plans_default():
    """LEARNED #162: the machine decides and tells him. Plain stop + chance -> next idea, parked for his hedge (as
    idea #1 and #2); with his hedge -> next idea, kept on file; works -> live trial, no money; needs fresh years ->
    forward test; too few trades -> next idea, kept."""
    from machine.referee.verdict import CHANCE, FEW, FRESH, WORKS
    assert S.choice(CHANCE, hedged=False)[0] == "NEXT IDEA" and S.choice(CHANCE, hedged=False)[2] == S.PARK
    assert "lost with your hedge too" in S.choice(CHANCE, hedged=True)[1]
    assert S.choice(WORKS, hedged=False)[0] == "LIVE TRIAL" and "no money" in S.choice(WORKS, hedged=True)[1]
    assert S.choice(FRESH, hedged=False)[0] == "FORWARD TEST" and "L17c" in S.choice(FRESH, hedged=False)[2]
    assert S.choice(FRESH, hedged=False, rest_passed=False) == S.choice(CHANCE, hedged=False)   # "at best": lost too
    assert S.choice(FEW, hedged=False)[0] == "NEXT IDEA" and "too few" in S.choice(FEW, hedged=False)[1]
    for w in (WORKS, FRESH, FEW, CHANCE):
        for hedged in (False, True):
            assert "?" not in " ".join(S.choice(w, hedged))                 # never a question


def test_a_report_the_machine_cannot_read_waits_for_a_session_and_the_queue_goes_on(home):
    """A REPORTED idea with no referee file (only a hand-set state can do this): the machine cannot choose - a
    session looks (one line), and the next idea still runs."""
    cache_answer(WORDS, wire())
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    store.set_state("idea-001", "REPORTED", "test: reported by hand, no referee file")
    cache_answer(WORDS2, wire())
    S.add(WORDS2)
    S.run(t)
    s = store.read_state("idea-001")
    assert s["state"] == "REPORTED" and "cannot read its report" in s["needs"]["why"]
    s2 = store.read_state("idea-002")                                         # the queue went on: its picture is made
    assert s2["state"] == "SHOWN" and "YES or NO" in s2["waits_for"]          # and sent at once (#175)
    assert S.undecided() == []


def test_a_decision_a_session_wrote_is_kept_as_it_is(home):
    cache_answer(WORDS, wire())
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    store.set_state("idea-001", "REPORTED", "test")
    (store.folder("idea-001") / S.DECISION).write_text("# his own decision\n", encoding="utf-8")
    S.run(t)
    assert store.read_state("idea-001")["state"] == "DECIDED"
    assert (store.folder("idea-001") / S.DECISION).read_text(encoding="utf-8") == "# his own decision\n"
    assert not [x for x in outbox() if x.get("why") == "L10b decided"]
    assert home.calls.count("parked idea-001") == 0              # a session's decision: nothing is tested by itself


def parked_idea(h, t, idea="idea-001") -> None:
    """An idea the machine parked for his hedge (as _decide leaves it), and the next idea in the queue."""
    cache_answer(WORDS, wire())
    S.add(WORDS)
    S.run(t)
    store.set_state(idea, "DECIDED", "test: the machine decided - parked", waits_for=S.PARK)
    (store.folder(idea) / S.DECISION).write_text("# Idea #1 — the decision after the report\n", encoding="utf-8")
    cache_answer(WORDS2, wire())
    S.add(WORDS2)


def test_a_parked_idea_is_tested_with_his_hedge_by_itself_once_and_he_is_told(home):
    """Idea #2's test with his hedge was run by a session (2 Oct, `machine.strategy parked`). Now the next run does it
    by itself, before the next idea: one section in decision.md, ONE line that tells him (never asks, LEARNED #162),
    then the idea waits for nothing. The same rule - not a new try."""
    t = tools(home)
    parked_idea(home, t)
    assert S.unhedged() == ["idea-001"]
    assert "his hedge instead of a plain stop" in " ".join(S.status_lines())
    did = S.run(t)
    assert did[0] == "idea-001: DECIDED -> DECIDED" and "idea-002 RECEIVED from the queue" in did   # then the queue
    s = store.read_state("idea-001")
    assert s["waits_for"] == "nothing (kept on file: tested with his hedge too)" and not s.get("needs")
    [told] = [x for x in outbox() if x.get("why") == "L10b tested with his hedge"]
    text = told["text"]
    print(text)
    assert text.startswith("Idea #1 with your hedge instead of a stop, on 2016-2020. The same rule the referee chose "
                           "- not a new try. 0.01 lot, ₹40 a position, all-in.")
    assert "16 trades, -₹11,583 after cost. Worst year 2018: -₹8,940." in text
    assert "Random entries with your hedge: -₹11,766 (the middle of 200 tries). Yours did better in 104 of 200." in text
    assert "The machine's choice: KEPT ON FILE. It lost with your hedge too." in text
    assert "?" not in text and text.endswith("If you want something else, just say it.")
    from machine.report.build import check_words
    assert check_words(text) == []                               # his easy English (#113)
    assert store.read_state("idea-002")["state"] == "SHOWN"
    S.run(t)
    assert home.calls.count("parked idea-001") == 1 and S.unhedged() == []    # never twice
    assert len([x for x in outbox() if x.get("why") == "L10b tested with his hedge"]) == 1


def test_a_hedge_test_that_cannot_run_waits_for_a_session_and_the_queue_goes_on(home):
    """No prices on this PC, say: the idea waits for a session (one line), the next idea still runs; after the
    session's retry the machine tests it by itself."""
    t = tools(home, parked=fake_parked(home, fail=FileNotFoundError("data/gold_m1.parquet")))
    parked_idea(home, t)
    S.run(t)
    s = store.read_state("idea-001")
    assert s["needs"]["why"] == ("its test with your hedge could not run (FileNotFoundError: data/gold_m1.parquet)")
    assert store.read_state("idea-002")["state"] == "SHOWN" and S.unhedged() == []
    assert len([x for x in outbox() if x.get("why") == "L10b needs a session"]) == 1
    S.run(t)
    assert home.calls.count("parked idea-001") == 1              # it waits: not tried at every run
    S.retry("idea-001")
    t.parked = fake_parked(home)
    S.run(t)
    assert store.read_state("idea-001")["waits_for"] == "nothing (kept on file: tested with his hedge too)"
    assert home.calls.count("parked idea-001") == 2


def fake_forward(h, fail: Exception | None = None):
    """machine.forward.plan.write without a report: the plan it returns (idea #5's spread would need 105 trades)."""
    def write(idea, log=print):
        h.calls.append(f"forward {idea}")
        if fail is not None:
            raise fail
        return {"idea": idea, "trades_needed": 105, "power": {"caught": 0.51, "trades_a_week": 2.036}, "weeks_at_pace": 52,
                "start": "2026-10-12", "end": "2028-04-07"}
    return write


def forward_idea(h, t, idea="idea-001") -> None:
    """An idea the machine chose a FORWARD TEST for (as _decide or _hedge leaves it), and the next idea queued."""
    cache_answer(WORDS, wire())
    S.add(WORDS)
    S.run(t)
    store.set_state(idea, "DECIDED", "test: the machine decided - a forward test", waits_for=S.FORWARD)
    cache_answer(WORDS2, wire())
    S.add(WORDS2)


def test_a_forward_test_idea_gets_its_locked_plan_by_itself_once_and_he_is_told(home):
    """L17c: before 9 Oct a FORWARD TEST waited for "L17c (not built yet)" - a session. Now the next run writes its
    plan by itself, before the next idea, tells him in ONE line (never asks, #162), and the idea waits for its
    pretend trades (L17b). Never tested with his hedge, never planned twice."""
    t = tools(home, forward=fake_forward(home))
    forward_idea(home, t)
    assert S.unplanned() == ["idea-001"] and S.unhedged() == []
    did = S.run(t)
    assert did[0] == "idea-001: DECIDED -> DECIDED" and "idea-002 RECEIVED from the queue" in did
    s = store.read_state("idea-001")
    assert s["waits_for"] == S.PLANNED and not s.get("needs") and "starts 2026-10-12" in s["history"][-1]["note"]
    [told] = [x for x in outbox() if x.get("why") == "L17c forward plan"]
    assert told["text"].startswith("Idea #1: its forward test is planned and locked (L17c).")
    assert ("It needs 105 trades on new days - about 52 weeks (it made about 2.0 trades a week in the years "
            "it was tested on).") in told["text"]
    S.run(t)
    assert home.calls.count("forward idea-001") == 1 and S.unplanned() == []
    assert home.calls.count("parked idea-001") == 0
    assert len([x for x in outbox() if x.get("why") == "L17c forward plan"]) == 1


def test_a_forward_plan_that_cannot_be_written_waits_for_a_session_and_comes_back_to_its_plan(home):
    from machine.forward.plan import ForwardError
    t = tools(home, forward=fake_forward(home, fail=ForwardError("no report")))
    forward_idea(home, t)
    S.run(t)
    assert store.read_state("idea-001")["needs"]["why"] == "its forward plan could not be written (ForwardError: no report)"
    assert store.read_state("idea-002")["state"] == "SHOWN"                  # the queue goes on
    S.run(t)
    assert home.calls.count("forward idea-001") == 1                         # it waits: not tried at every run
    S.retry("idea-001")
    assert store.read_state("idea-001")["waits_for"] == S.FORWARD            # its plan again, not his hedge's test
    t.forward = fake_forward(home)
    S.run(t)
    assert store.read_state("idea-001")["waits_for"] == S.PLANNED
    assert home.calls.count("forward idea-001") == 2 and home.calls.count("parked idea-001") == 0


def test_an_idea_that_passes_with_his_hedge_gets_its_forward_plan_in_the_same_run(home):
    """The hedge road: made money with his hedge and beat 195 of 200 random entries with it -> FORWARD TEST
    (parked.choice) -> its plan, in the same run, by itself."""
    from machine.strategy import parked as PK
    good_h = {**HEDGED, "net_inr_total": 5200.0}
    good_r = {**HEDGED_R, "his_beats_random": {"with_hedging": 195}}

    def run(idea, log=print):
        home.calls.append(f"parked {idea}")
        head, means = PK.choice(good_r, good_h)
        return {"idea": idea, "his_hedging": good_h, "result": good_r, "choice": head, "means": means,
                "his_stop": {"trades": 448, "net_inr_total": -21661.0}, "trial": "20261009-213000-test.json"}

    t = tools(home, parked=run, forward=fake_forward(home))
    parked_idea(home, t)
    did = S.run(t)
    assert did[:2] == ["idea-001: DECIDED -> DECIDED"] * 2                  # his hedge's test, then its plan
    assert store.read_state("idea-001")["waits_for"] == S.PLANNED
    assert home.calls.count("parked idea-001") == 1 and home.calls.count("forward idea-001") == 1
    whys = [x.get("why") for x in outbox() if x.get("idea") == "idea-001"]
    assert whys.index("L10b tested with his hedge") < whys.index("L17c forward plan")


def test_a_hedge_test_that_ran_is_read_back_from_its_trial_file_and_never_runs_again(home, monkeypatch):
    """parked.run finds its own trial file (a crash after it, or a session ran it by hand first): nothing loads prices
    or compiles; the decision section comes from the file, once; no second trial file."""
    from machine import trials as TF
    from machine.strategy import parked as PK
    t = tools(home)
    parked_idea(home, t)
    chosen = hand_tune(home, "idea-001").chosen
    result = {**HEDGED_R, "his_hedging": HEDGED, "his_stop": {"trades": 448, "net_inr_total": -21661.0},
              "choice": "KEPT ON FILE"}
    TF.record(kind=PK.KIND, idea_id="idea-001", version_id=chosen, parent_idea=None, settings={},
              data={"prices": "test"}, result=result, note="L19d: idea-001 (parked for his hedge) - test")
    page = PK._page_path("idea-001")
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("# the page\n", encoding="utf-8")
    monkeypatch.setattr(PK, "Market", Boom())                    # no prices are read
    monkeypatch.setattr(PK, "compile_idea", boom)
    t.parked = PK.run                                            # the real one
    S.run(t)
    s = store.read_state("idea-001")
    assert s["waits_for"] == "nothing (kept on file: tested with his hedge too)"
    decision = (store.folder("idea-001") / S.DECISION).read_text(encoding="utf-8")
    assert decision.count("## L19d - tested with his hedge") == 1 and "-₹11,583 after cost" in decision
    assert page.read_text(encoding="utf-8") == "# the page\n"   # kept as it is
    assert any("tested with his hedge already" in x for x in home.log)
    PK.run("idea-001", log=lambda *_: None)                      # by hand again: still nothing new
    assert (store.folder("idea-001") / S.DECISION).read_text(encoding="utf-8") == decision
    assert len([f for f in TF.read_all() if f["kind"] == PK.KIND]) == 1


def test_a_picture_an_older_drawing_made_is_drawn_again_before_it_is_sent(home, monkeypatch):
    """Idea #3's picture (made 1 Oct, not sent) said "session high · taken" where his words say "the London high is
    taken". A picture made but not sent, by an older drawing, is drawn again - the same question, the same setups -
    and the batch never sends the old one."""
    cache_answer(WORDS, wire(unclear=[ASK]))
    t = tools(home)
    S.add(WORDS)
    S.run(t)
    idea = "idea-001"
    from machine.questions import interpret
    for q in Q.load(idea, "Q1")["questions"]:
        interpret.on_answer({"qid": q["qid"]}, "B", False, source="test")
    interpret.on_answer({"qid": f"{idea}-Q1-confirm"}, "yes", False, source="test")
    with monkeypatch.context() as m:                            # made and not sent - as idea #3's, before #175
        m.setattr(S, "batch_step", lambda *a, **k: False)
        S.run(t)
    rec = shown.load(idea)
    assert rec["status"] == "made"
    rec.pop("drawing")                                          # as idea #3's: made before the drawing had a number
    shown.save(rec)
    assert shown.stale(rec) and S.batch_step(t, early="test: his words") is False      # the old one never goes
    S.run(t)
    new = shown.load(idea)
    assert new["qid"] == rec["qid"] and new["drawing"] == shown.DRAWING and home.calls.count("shown 1") == 2
    assert not (store.folder(idea) / "shown-1.json").exists()   # never sent: nothing to keep apart
    assert new["status"] == "sent"                              # the new drawing went at once (#175)


def test_the_footer_names_only_the_marks_drawn():
    """His rule: a picture draws every thing it names. Idea #3's first picture (3 sells, no pair) said "B1 = the buy"
    and "1, 2 = two touches of a pair"."""
    t = pd.Timestamp("2019-03-11 10:00")
    sells = [SimpleNamespace(direction="sell")] * 3
    steps = SimpleNamespace(steps=[SimpleNamespace(block="sweep"), SimpleNamespace(block="break_of_structure")])
    no_pair = [[SimpleNamespace(items=[D.Level(1.0, "London high · taken")])]]
    assert shown.footer(steps, sells, no_pair) == f"S1 = the sell · {shown.BREAK_MEANS}"
    pair = [[SimpleNamespace(items=[D.Line((t, 1.0), (t + pd.Timedelta("1h"), 1.0), touches=(t,))])]]
    both = [*sells[:2], SimpleNamespace(direction="buy")]
    assert shown.footer(SimpleNamespace(steps=[]), both, pair) == \
        "S1 = the sell · B1 = the buy · 1, 2 = two touches of a pair"


def test_a_session_level_is_named_by_its_session():
    """'the London high is taken' (idea #3): the picture says "London high", not "session high"."""
    assert P._kind_words("session_high", "high", "London") == "London high"
    assert P._kind_words("session_low", "low", "Asia") == "Asia low"
    assert P._kind_words("session_high", "high", None) == "session high"
    assert P._kind_words("session_high", "high", float("nan")) == "session high"
    assert P._kind_words("equal_highs", "high", "London") == "equal highs"


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


def _with_step(sid: str, block: str, what: str) -> Idea:
    d = json.loads(_test_form().to_json())
    d["steps"].append({"id": sid, "what": what, "block": block, "tf": "5m", "max_wait_bars": 0, "params": {},
                       "sources": [{"ref": "his words"}]})
    return Idea.load(d)


def test_his_strong_zone_is_drawn_as_its_box_across_the_whole_chart(tmp_path):
    """9 Oct, his entry types as phone ideas: the zone entry (0 of 236 setups) and the zone breakout (0 of 54) had NO
    picture - each named a strong zone the chart did not draw (#113 rule 3), so such an idea waited for a session
    before he ever saw it. Now the step's zone is its box, from the chart's left edge (it was there days before)."""
    m5 = F.bars_5m(1000)
    form = _with_step("z", "zone", "price is at one of his strong zones")
    scenes = []
    for i0 in (100, 400, 700):
        sc = one_setup(m5, i0)
        g = sc.facts["A"]["gap"]
        sc.facts["A"]["z"] = {"lo": g["lo"] - 0.3, "hi": g["hi"] + 0.5, "zone_id": "week:2016-01-04:1200.00",
                              "known_at": m5.index[i0 - 200]}
        scenes.append(sc)
    pic = shown.picture(form, scenes, m5, subject="test SHOWN strong zone")
    drawn = D.render(pic, tmp_path / "strong.jpg")
    assert lint.check(drawn, shown=tmp_path / "none.jsonl") == []
    for row in pic.rows:
        [z] = [i for i in row[0].items if isinstance(i, D.Zone)]
        assert z.name == "strong zone · 5-min chart, 5 days" and z.start is None
    assert "Why it enters: taken → break → first gap → at a strong zone → entry" in pic.sub


def test_a_strong_zone_is_named_by_the_candles_it_was_counted_on():
    """Its zone_id starts with the window's name (settings zone_windows): "week" = the 5-minute candles of the last 5
    days - NOT the week chart (a first try of 9 Oct wrote "week chart")."""
    assert shown.zone_name({"zone_id": "week:2016-03-29:1214.59"}) == "strong zone · 5-min chart, 5 days"
    assert shown.zone_name({"zone_id": "hour:2016-03-16:1226.80"}) == "strong zone · hour chart, 20 days"
    assert shown.zone_name({"zone_id": "today:2016-03-16:1226.80"}) == "strong zone · 5-min chart, 1 day"
    assert shown.zone_name({}) == "strong zone"


def test_his_triangle_is_both_its_lines_and_his_trendline_has_its_3_touches():
    """9 Oct: his triangle false break (0 of 29 setups) and trendline touch (0 of 11) had NO picture - the chart did
    not draw the lines it named. Each line is rebuilt from the two wick ends its id names (#112)."""
    m5 = F.bars_5m(1000)
    I = m5.index
    t = lambda k: f"{I[k]:%Y-%m-%d %H:%M}"                                     # noqa: E731
    tri = f"tri:5m:line_high:5m:{t(100)}:{t(130)}|line_low:5m:{t(105)}:{t(140)}"
    form = _with_step("tri", "triangle", "a triangle after a big move")
    form = Idea.load({**json.loads(form.to_json()), "steps": [
        *json.loads(form.to_json())["steps"],
        {"id": "touch", "what": "the 3rd touch of a steep line", "block": "trendline_touch", "tf": "5m",
         "max_wait_bars": 0, "params": {}, "sources": [{"ref": "his words"}]}]})
    sc = one_setup(m5, 300)
    sc.facts["A"].update(tri={"tri_id": tri, "bar": I[150]}, fb={"level_id": tri, "bar": I[160]},
                         touch={"line_id": f"line_low:5m:{t(200)}:{t(240)}", "bar": I[290], "touch": 3})
    items, times = shown._lines(form, sc, m5)
    top, bottom, trend = items                                                 # in the form's order of steps
    assert trend.name == "trendline" and trend.touches == (I[200], I[240], I[290]) and trend.extend_to == sc.end
    assert trend.start == (I[200], float(m5["low"].iloc[200])) and trend.end == (I[240], float(m5["low"].iloc[240]))
    assert top.group == bottom.group == tri and top.name == bottom.name == "triangle"
    assert top.start == (I[100], float(m5["high"].iloc[100])) and bottom.end == (I[140], float(m5["low"].iloc[140]))
    assert top.extend_to == bottom.extend_to == I[160]                         # to where price pushed out of it
    assert min(times) == I[100]                                                # the chart starts before the shape
    sc.facts["A"]["touch"]["touch"] = 4                                         # where the 3rd was is not known:
    assert [i.name for i in shown._lines(form, sc, m5)[0]] == ["triangle", "triangle"]     # not drawn
    hour = shown.anchors(f"line_high:1h:{t(0)}:{t(24)}", m5)                   # an hour line: the hour's wick
    assert hour[0] == (I[0], float(m5["high"].iloc[0:12].max()))


def test_the_second_gap_is_said_named_and_found():
    """His second-gap entry (#124, L19b): about 130 of its 4,850 setups enter at the second gap - a picture spread by
    year alone showed 3 plain first gaps, under the words 'first gap' (9 Oct)."""
    d = json.loads(_test_form().to_json())
    d["settings"] = {"which_gap_of_the_move": "second"}
    assert "→ first gap, or the second when the trend is very strong →" in shown.why_line(Idea.load(d))
    m5 = F.bars_5m(1000)
    sc = one_setup(m5, 100)
    sc.facts["A"]["gap"]["gap_rank"] = 1
    [box] = [i for i in P.items_for(_test_form(), sc, {"A": "A"}, m5)[0] if isinstance(i, D.Box)]
    assert box.label.startswith("second gap ")
    trace = [{"signal": "s1", "facts": {"gap": {"gap_rank": 1}}}, {"signal": "s2", "facts": {"gap": {"gap_rank": 0}}}]
    assert shown.second_gaps(SimpleNamespace(trace=trace)) == {"s1"}


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


def weeks_of_bars(n_weeks: int = 5, per_week: int = 300) -> pd.DataFrame:
    """5-minute candles in n different weeks (a stretch from each Monday), so every setup is in a week of its own."""
    return pd.concat([F.bars_5m(per_week, start=str(pd.Timestamp("2019-03-11") + pd.Timedelta(weeks=k)), seed=7 + k)
                      for k in range(n_weeks)])


def choose_on(m5, monkeypatch, i0s=(100, 400, 700, 1000, 1300), **kw) -> tuple[list[P.Scene], list[P.Scene]]:
    """shown.choose over hand-made setups (one_setup), one a week: (the setups, the ones chosen)."""
    scenes = [one_setup(m5, i0) for i0 in i0s]
    by_sig = {sc.signals["A"].id: sc for sc in scenes}
    monkeypatch.setattr(P, "scene", lambda pick, comp, market: by_sig[pick.signals["A"].id])
    got = shown.choose(_test_form(), SimpleNamespace(signals=[sc.signals["A"] for sc in scenes]),
                       SimpleNamespace(bars=lambda tf: m5), log=lambda s: None, **kw)
    return scenes, got


def test_a_setup_whose_chart_shares_a_candle_with_a_chart_he_was_sent_is_never_chosen(monkeypatch):
    """His #117: "I don't want the same charts again". Idea #4's first picture (drawing 2, 7 Oct 2026) showed idea #3's
    sell of 12 Jun 2018 20:05 again - the chart he had answered that morning. Now a chart with ANY candle of a setup
    chart sent before, in any idea's picture, is never chosen (the zone charts' rule: labels/daily._clash)."""
    m5 = weeks_of_bars()
    scenes, free = choose_on(m5, monkeypatch)
    ends = [sc.end for sc in scenes]
    assert [sc.end for sc in free] == [ends[0], ends[2], ends[1]]                  # early, middle, late
    panel = shown.one_chart(_test_form(), scenes[1], m5)
    a, b = panel.bars.index[0], panel.bars.index[-1]
    _, got = choose_on(m5, monkeypatch, seen=[(b, b + pd.Timedelta(hours=1))])     # its last candle seen
    assert [sc.end for sc in got] == [ends[0], ends[2], ends[4]]                   # another late one instead
    _, got = choose_on(m5, monkeypatch, seen=[(a - pd.Timedelta(hours=1), a - pd.Timedelta(minutes=5))])
    assert [sc.end for sc in got] == [sc.end for sc in free]                       # no candle in common: as before
    _, got = choose_on(m5, monkeypatch, prefer=[sc.signals["A"].id for sc in free], seen=[(a, b)])
    assert [sc.end for sc in got] == [ends[0], ends[2], ends[4]]                   # never "the same setups again"


def test_the_charts_he_was_sent_are_every_ideas_sent_setups(tmp_path, monkeypatch):
    """seen_spans: every SENT picture of 3 setups of every idea, a kept one (shown-<n>.json) too; a record before 7 Oct
    has no first candle - the longest chart there can be stands in."""
    monkeypatch.setattr(config, "IDEAS_DIR", tmp_path / "ideas")

    def put(idea: str, name: str, rec: dict) -> None:
        d = tmp_path / "ideas" / idea
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text("{}", encoding="utf-8")
        (d / name).write_text(json.dumps(rec), encoding="utf-8")
    put("idea-002", "shown-1.json", {"status": "withdrawn", "sent": "2026-10-01T17:11:30+05:30",
                                     "setups": [{"n": 1, "end": "2016-12-05 04:40:00"}]})      # taken back: he saw it
    put("idea-003", "shown.json", {"status": "yes", "sent": "2026-10-07T00:16:53+05:30",
                                   "setups": [{"n": 2, "first": "2018-06-12 16:05:00", "end": "2018-06-12 17:35:00"}]})
    put("idea-004", "shown.json", {"status": "made", "setups": [{"n": 3, "end": "2018-06-12 17:35:00"}]})   # not sent
    end = pd.Timestamp("2016-12-05 04:40")
    assert shown.seen_spans() == [(end - pd.Timedelta(minutes=5 * (P.ONE_MAX - 1)), end),
                                  (pd.Timestamp("2018-06-12 16:05"), pd.Timestamp("2018-06-12 17:35"))]

    got = {}                                                    # make() chooses away from them
    (tmp_path / "ideas" / "idea-005").mkdir()
    monkeypatch.setattr(store, "load_form", lambda idea: _test_form())
    monkeypatch.setattr(shown, "choose", lambda *a, **k: got.update(k) or [])
    with pytest.raises(shown.ShownError, match="share no candle with a chart he was sent before"):
        shown.make("idea-005", SimpleNamespace(), compile_fn=lambda *a, **k: SimpleNamespace(version_id="v", signals=[]))
    assert got["seen"] == shown.seen_spans() and len(got["seen"]) == 2


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


@pytest.mark.data
def test_a_picture_shows_only_setups_the_tester_really_enters():
    """His #68 orders (first_visit_from): idea #4's buy of 17 Jun 2016 touched its entry on the chart at 21:10 India
    time (18:40 server), but the spread kept the buy limit from filling and price turned away - the tester calls it
    spent, so the picture of 3 setups never shows it (the chart alone would draw 21:10 as its entry). Setup 1 of 10 Aug
    2016 (his NO) is spent too; setup 2 of 15 Feb 2019 (his YES) is entered. An order without the check: no replay."""
    from machine.tester.models import Signal
    common = dict(direction="buy", entry_type="limit", levels="price", expiry_bars=1000000000, idea_id="idea-004",
                  version_id="v-b995436713")
    touch = Signal(id="a", known_at="2016-06-17 18:40", entry_price=1288.01, stop=1284.86, targets=(1295.97,),
                   cancel_at=1295.97, first_visit_from="2016-06-17 18:00", first_visit_to="2016-06-17 18:45", **common)
    one = Signal(id="b", known_at="2016-08-10 20:30", entry_price=1345.1, stop=1342.63, targets=(1357.11,),
                 cancel_at=1357.11, first_visit_from="2016-08-10 18:00", first_visit_to="2016-08-10 20:35", **common)
    two = Signal(id="c", known_at="2019-02-15 18:25", entry_price=1315.69, stop=1314.12, targets=(1319.9,),
                 cancel_at=1319.9, first_visit_from="2019-02-15 17:00", first_visit_to="2019-02-15 18:25", **common)
    assert not shown.the_test_enters(touch) and not shown.the_test_enters(one) and shown.the_test_enters(two)
    assert shown.the_test_enters(two.model_copy(update={"first_visit_from": None, "first_visit_to": None}))
