"""L9b: the question loop - disagree, the #129 search first, the pictures, the weekly batch, his reply, the lock.
No price data needed (hand-made bars, signals and records); the real run on idea #1 is in plan/work/question-queue/."""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from machine import config
from machine.charts import draw as D
from machine.charts import lint
from machine.ideas import store
from machine.ideas.form import Idea
from machine.questions import ask, batch, disagree, interpret, pictures
from machine.questions import queue as Q
from machine.telegram import bot as B
from machine.telegram import inbox
from machine.tester.models import Signal
from tests import chart_fakes as F
from tests.telegram_fakes import TEJAS, FakeTelegram

NOW = dt.datetime(2026, 9, 29, 10, 0).astimezone()
FORM = config.ROOT / "plan" / "work" / "ideas" / "idea-001" / "form.json"
MONEY = re.compile(r"₹|\brupees?\b|\bRs\b|\bINR\b|\bprofits?\b|\blosses\b|\bloss\b", re.IGNORECASE)


# A made-up ASK part for the tests (a MEANING, not a number): idea #1's real parts are all VERSION parts since his
# words of 29 Sep (inbox 2026-09-29-1155). Its sources hold the "too near" words, for the own-source search test.
TEST_ASK = {"id": "U2", "kind": "ask", "part": "which equal lows are the target when several wait (test part)",
            "words": "the lows are not proper lows cause they are too near to each other",
            "readings": [{"id": "A", "means": "the nearest one (his written rule: 'the nearest')",
                          "changes": {"steps.other.params.pick": "nearest"}},
                         {"id": "B", "means": "the most recent one first",
                          "changes": {"steps.other.params.pick": "his_order"}}],
            "searched": "test", "sources": [{"ref": "inbox 2026-09-28-1151", "words": "too near"}]}


def _test_form() -> Idea:
    d = json.loads(FORM.read_text(encoding="utf-8")) | {"status": "QUESTIONS QUEUED"}
    d["unclear"] = [u for u in d["unclear"] if u["id"] == "U1"] + [TEST_ASK]
    return Idea.load(d)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway plan/work/: idea #1's real form with a made-up open ASK part U2, the queue, the outbox, shown."""
    work = tmp_path / "work"
    monkeypatch.setattr(config, "WORK_DIR", work)
    monkeypatch.setattr(config, "IDEAS_DIR", work / "ideas")
    monkeypatch.setattr(config, "QUESTION_QUEUE_DIR", work / "question-queue")
    monkeypatch.setattr(config, "SHOWN_PICTURES", work / "pictures-shown.jsonl")
    monkeypatch.setattr(config, "SEARCH_FRESH_EXAM", tmp_path / "search-fresh.jsonl")
    form = _test_form()
    store.create(form)
    return SimpleNamespace(tmp=tmp_path, work=work, queue=work / "question-queue",
                           outbox=work / "telegram-outbox.jsonl", shown=work / "pictures-shown.jsonl", form=form)


def sig(t: str, direction="sell", entry=1300.0, stop=1301.0, target=1297.0, rid="x") -> Signal:
    return Signal(id=f"s-{rid}-{t}", known_at=pd.Timestamp(t), direction=direction, entry_type="limit",
                  entry_price=entry, stop=stop, targets=(target,), expiry_bars=100, idea_id="idea-001",
                  version_id=f"v-{rid}")


# ================================================================================================ 1. disagree

def test_readings_that_agree_95_percent_ask_nothing():
    same = [sig(f"2019-03-{d:02d} 10:00", rid=r) for d in range(1, 21) for r in ("A",)]
    a = same
    b = [sig(s.known_at.isoformat(), rid="B") for s in same]           # the same orders from another reading
    d = disagree.compare_signals({"A": a, "B": b})
    assert d.share_same == 1.0 and d.agree and d.moments == []
    b[-1] = sig("2019-03-20 10:00", entry=1299.5, rid="B")              # 1 different order of 21 = 95% same
    d = disagree.compare_signals({"A": a, "B": b})
    assert d.same == 19 and d.different == 2 and not d.agree            # 19 / 21 = 90%: they differ


def test_moments_are_ranked_by_how_different_the_trades_are():
    a = [sig("2019-03-01 10:00", rid="A"), sig("2019-03-05 10:00", entry=1300.0, rid="A")]
    b = [sig("2019-03-05 10:05", entry=1300.2, rid="B"), sig("2019-03-09 10:00", direction="buy", entry=1290,
                                                             stop=1289, target=1293, rid="B")]
    d = disagree.compare_signals({"A": a, "B": b})
    assert [m.score for m in d.moments] == [1.0, 1.0, pytest.approx(0.2)]
    assert d.moments[0].trading() == ["A"] and d.moments[1].trading() == ["B"]       # trade vs no trade first
    assert d.moments[2].trading() == ["A", "B"] and "different prices" in d.moments[2].why


def test_compare_refuses_the_check_years(home):
    market = SimpleNamespace(window=lambda: (pd.Timestamp("2016-01-04"), pd.Timestamp("2021-06-01")))
    with pytest.raises(ValueError, match="TUNE years"):
        disagree.compare(home.form, market, "U2")


# ================================================================================================ 2. search first

def hit(id_, quote="a line", see=()):
    return SimpleNamespace(id=id_, quote=quote, see=tuple(see))


def fake_search(*hits, found=True):
    return lambda q: SimpleNamespace(found=found, cites=tuple(hits), closest=hits[0] if hits else None,
                                     sentence=f"Found in {hits[0].id}" if hits else "Searched ...", how="fake")


def test_found_stops_the_question(home):
    rec = ask.make_part("idea-001", "U2", market=object(), search=fake_search(hit("LEARNED #88", "exactly equal")),
                        now=NOW, log=lambda s: None, base=home.queue)
    assert rec["status"] == "found" and rec["questions"] == [] and rec["search"]["stops"]
    assert "counts as an answer" in (home.queue / "idea-001-U2" / "part.md").read_text(encoding="utf-8")


def test_its_own_source_claudes_line_and_a_checked_place_do_not_stop_it(home, monkeypatch):
    agree = SimpleNamespace(text=lambda: "", held={}, years=(2016, 2020), counts={"A": 5, "B": 5}, same=5,
                            different=0, share_same=1.0, moments=[], agree=True)
    monkeypatch.setattr(disagree, "compare", lambda *a, **k: agree)
    ask.add_checked("idea-001", "U2", "understanding §66g", "about crossed pairs, not time apart", now=NOW,
                    base=home.queue)
    search = fake_search(hit("LEARNED #149", "too near", see=("inbox 2026-09-28-1151-session",)),
                         hit("settings.py x", "Who chose it: Claude (his T3 answers)"),
                         hit("understanding §66g", "crossed pairs"))
    rec = ask.make_part("idea-001", "U2", market=object(), search=search, now=NOW, log=lambda s: None,
                        base=home.queue)
    assert not rec["search"]["stops"] and rec["status"] == "readings agree"       # past the search: no question
    why = [c["why_not"] for c in rec["search"]["cites"]]
    assert why[0].startswith("its own source") and why[1].startswith("Claude's") and why[2].startswith("read by")


def test_a_version_part_is_never_asked(home):
    with pytest.raises(ask.AskError, match="VERSION"):
        ask.make_part("idea-001", "U1", market=object(), search=fake_search(), base=home.queue)


def test_a_number_is_never_asked_even_when_marked_ask(home):
    """His words, 29 Sep (inbox 2026-09-29-1155): 'too near' = around 7 candles; the machine tests the edge cases and
    tells him which works better. A part whose readings only set one number is never a question."""
    d = json.loads(home.form.to_json())
    d["unclear"].append({"id": "U9", "kind": "ask", "part": "how near is too near", "readings": [
        {"id": "A", "means": "any time apart", "changes": {}},
        {"id": "B", "means": "at least 7 candles apart", "changes": {"settings.equal_levels_min_apart_bars": 7}}]})
    store.save_form(Idea.load(d))
    with pytest.raises(ask.AskError, match="is a NUMBER .* never asked. His words, 29 Sep 2026"):
        ask.make_part("idea-001", "U9", market=object(), search=fake_search(), base=home.queue)
    assert not ask.is_a_number(next(u for u in store.load_form("idea-001").unclear if u.id == "U2"))   # a meaning


def test_idea001_has_no_question_left_and_18_versions():
    """His words, 29 Sep (inbox 2026-09-29-1155): U2 'too near' became versions - no minimum / 5 / 7 candles."""
    from machine.ideas.form import reading_choices
    form = Idea.load(FORM)
    u2 = next(u for u in form.unclear if u.id == "U2")
    assert u2.kind == "version" and [r.changes for r in u2.readings] == [
        {"settings.equal_levels_min_apart_bars": n} for n in (0, 5, 7)]
    assert all(u.kind == "version" for u in form.unclear)
    assert len(reading_choices(form)) == 18 <= config.VERSION_BUDGET


# ================================================================================================ 3. the pictures

def scene_and_form():
    """A hand-made sell of idea #1: equal highs 30 minutes apart taken, equal lows below, a gap, S1. Reading A's
    liquidity book has the highs as a pair; B's does not (too near) - B makes no trade there."""
    m5 = F.bars_5m(220)
    I = m5.index
    hi = float(m5["high"].iloc[95:125].max()) + 0.4
    lo = float(m5["low"].iloc[95:175].min()) - 0.4
    highs = {"level_id": "equal_high:5m:A", "event": "born", "kind": "equal_high", "tf": "5m", "side": "high",
             "price": hi, "first_bar": I[100], "touch_bar": I[106]}
    lows = {"level_id": "equal_low:5m:B", "event": "born", "kind": "equal_low", "tf": "5m", "side": "low",
            "price": lo, "first_bar": I[110], "touch_bar": I[150]}
    gap_lo = float(m5["close"].iloc[170]) + 0.1
    facts = {"take": {"level_id": "equal_high:5m:A", "bar": I[160], "price": hi},
             "other": {"level_id": "equal_low:5m:B", "price": lo},
             "brk": {"bar": I[165]},
             "gap": {"lo": gap_lo, "hi": gap_lo + 0.4, "first_bar": I[166], "near": gap_lo}}
    s = sig(str(I[168] + pd.Timedelta(minutes=5)), entry=gap_lo, stop=hi + 1, target=lo)
    sc = pictures.Scene(end=I[175], direction="sell", signals={"A": s, "B": None}, facts={"A": facts},
                        books={"A": pd.DataFrame([highs, lows]), "B": pd.DataFrame([lows])})
    form = _test_form()
    return sc, form, next(u for u in form.unclear if u.id == "U2"), m5


def all_words(drawn) -> str:
    return " ".join(t.text for t in drawn.texts)


def test_one_picture_per_question_marks_what_they_disagree_about_ends_at_entry_lint_no_money(tmp_path):
    """His words (inbox 2026-09-29-0912): 'Both images are same for each questions' -> ONE picture, the disputed
    thing marked with a ?, what A does and what B does both written on it."""
    sc, form, part, m5 = scene_and_form()
    pic = pictures.picture(form, part, sc, m5, subject="test U2", letters={"A": "A", "B": "B"}, number="QUESTION 1")
    drawn = D.render(pic, tmp_path / "q1.jpg")
    assert lint.check(drawn, shown=tmp_path / "none.jsonl") == []                     # every picture rule
    assert drawn.entry_at == sc.end and pic.rows[0][0].bars.index[-1] == sc.end       # ends at the entry candle
    assert pic.heading == "QUESTION 1 · A OR B?" and len(pic.rows) == 1 == len(pic.rows[0])   # ONE chart
    assert pic.question.ask == "Which one is how you trade it: A or B?"               # the question ON it
    how = " ".join(pic.question.how)
    assert "A: the nearest one" in how and "→ sell at S1" in how and "→ no sell here" in how
    assert "your written rule" in how                                                 # spoken TO him
    words = all_words(drawn)
    assert "equal highs? · taken · 30 min apart" in words                             # the ? = what they disagree on
    assert "equal lows · goes here" in words and "?" not in words.split("equal lows")[1][:3]
    assert "Look at the equal highs? (30 min apart): A and B read it differently." in pic.sub
    assert not MONEY.search(words) and "$" not in words


def test_the_confirm_picture_says_what_the_answer_does_there(tmp_path):
    sc, form, part, m5 = scene_and_form()
    pic = pictures.picture(form, part, sc, m5, subject="test U2 confirm B", letters={"A": "A", "B": "B"},
                           number="", confirm="B")
    assert pic.heading == "I READ YOUR ANSWER AS B" and pic.question.ask == "Is this how you trade it?"
    assert pic.question.how[0].endswith("→ no sell here") and pic.question.how[-1].startswith("Reply YES to lock")
    assert lint.check(D.render(pic, tmp_path / "c.jpg"), shown=tmp_path / "none.jsonl") == []


def test_the_entry_candle_is_the_first_fill_and_a_cancel_first_means_none():
    m5 = F.bars_5m(60)
    t = m5.index[10]
    above = float(m5["high"].iloc[11:].max()) + 5
    s = sig(str(t), entry=float(m5["high"].iloc[20]), stop=above, target=1000.0)
    got = pictures.entry_candle(s, m5)
    assert got is not None and got >= t and m5.loc[got, "high"] >= s.entry_price
    first = int(m5.index.get_loc(got))
    assert (m5["high"].iloc[10:first] < s.entry_price).all()
    cancelled = s.model_copy(update={"cancel_at": float(m5["low"].iloc[10]) + 0.01})
    assert pictures.entry_candle(cancelled.model_copy(update={"entry_price": above - 1}), m5) is None


# ================================================================================================ 4. the batch

def fake_part(queue: Path, idea: str, part: str, n: int, made: str, *, tmp: Path) -> dict:
    folder = queue / f"{idea}-{part}"
    folder.mkdir(parents=True, exist_ok=True)
    qs, confirm = [], {}
    for k in range(1, n + 1):
        f = folder / f"q{k}.jpg"
        shutil.copy(tmp / "pic.jpg", f)
        ask_ = "Which one is how you trade it: A or B?"
        pic = {"letter": None, "reading": None, "file": str(f), "subject": f"{idea} {part} {k}",
               "pixel_sha256": f"{idea}{part}{k}", "end": "2019-03-11 12:00:00", "ends_at": "2019-03-11 12:00:00",
               "last_candle": "2019-03-11 12:00:00", "question": ask_, "lint": "all 13 rules pass",
               "texts": [f"QUESTION {k} · A OR B?", ask_, "A: a pair counts → sell at S1", "B: too near → no sell here"]}
        qs.append({"qid": f"{idea}-{part}-{k}", "n": k, "direction": "sell", "end": "2019-03-11 12:00:00",
                   "end_ist": "11 Mar 2019 17:30", "score": 1.0, "why": "A trades here; B does not",
                   "trading": ["A"], "letters": ["A", "B"], "pictures": [pic], "status": "waiting", "answer": None})
    for letter in "AB":
        f = folder / f"confirm-{letter}.jpg"
        shutil.copy(tmp / "pic.jpg", f)
        confirm[letter] = {"letter": letter, "reading": letter, "file": str(f), "subject": f"{idea} {part} c {letter}",
                           "pixel_sha256": f"{idea}{part}c{letter}", "trades": letter == "A", "texts": []}
    rec = {"idea": idea, "part": part, "kind": "readings", "topic": "two touches close in time", "words": "too near",
           "readings": {"A": "a pair counts however near (his written rule)", "B": "at least 7 candles apart"},
           "letters": {"A": "A", "B": "B"}, "made_by": "test", "made": made, "status": "waiting", "checked": [],
           "search": {"query": "q", "found": False, "stops": False, "sentence": "Searched ...", "cites": [],
                      "closest": "LEARNED #89", "how": "fake"},
           "questions": qs, "confirm": {"qid": f"{idea}-{part}-confirm", "end": "x", "end_ist": "12 Mar 2019 10:00",
                                        "status": None, "proposed": None, "pictures": confirm},
           "history": []}
    Q.save(rec, queue)
    return rec


@pytest.fixture
def three_parts(home, monkeypatch):
    F.bars_5m(5)
    from PIL import Image
    Image.new("RGB", (40, 30), "white").save(home.tmp / "pic.jpg", "JPEG")
    fake_part(home.queue, "idea-003", "U1", 2, "2026-09-01T10:00:00+05:30", tmp=home.tmp)   # oldest, idle idea
    fake_part(home.queue, "idea-001", "U2", 2, "2026-09-20T10:00:00+05:30", tmp=home.tmp)   # the running idea
    fake_part(home.queue, "idea-002", "U5", 2, "2026-09-25T10:00:00+05:30", tmp=home.tmp)   # a BLOCKED idea
    states = {"idea-001": {"state": "QUESTIONS QUEUED", "since": "2026-09-28T16:00:00+05:30"},
              "idea-002": {"state": "BLOCKED", "since": "2026-09-29T09:00:00+05:30"},
              "idea-003": {"state": "QUESTIONS QUEUED", "since": "2026-09-02T09:00:00+05:30"}}
    monkeypatch.setattr(batch, "idea_states", lambda: states)
    return home


def test_the_budget_stops_the_extra_question_and_the_order_is_blocked_running_oldest(three_parts):
    h = three_parts
    out = batch.make(dry=False, budget=5, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)
    assert out["chosen"] == ["idea-002-U5-1", "idea-002-U5-2", "idea-001-U2-1", "idea-001-U2-2", "idea-003-U1-1"]
    assert out["later"] == ["idea-003-U1-2"]                                   # the 6th waits for next week
    lines = [json.loads(x) for x in h.outbox.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 + 5                                                # the intro + ONE picture a question
    assert {ln["qid"] for ln in lines[1:]} == set(out["chosen"])
    assert all(ln.get("shown") for ln in lines[1:]) and not h.shown.exists()    # shown only when the bot sends it
    assert Q.load("idea-003", "U1", h.queue)["questions"][1]["status"] == "waiting"
    assert "Next week (over the budget of 5):** idea-003-U1-2" in (h.queue / "batches" / "2026-09-29-1000" /
                                                                   "batch.md").read_text(encoding="utf-8")


def test_dry_run_writes_the_batch_and_sends_nothing(three_parts):
    h = three_parts
    out = batch.make(dry=True, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)
    assert len(out["chosen"]) == 6 and not h.outbox.exists()
    assert (h.queue / "batches" / "2026-09-29-1000" / "batch.md").read_text(encoding="utf-8").startswith(
        "# Question batch — 2026-09-29-1000 (DRY RUN: nothing sent)")
    assert all(q["status"] == "waiting" for r in Q.records(h.queue) for q in r["questions"])


def test_no_money_anywhere_in_a_question(three_parts):
    h = three_parts
    out = batch.make(dry=False, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)
    text = h.outbox.read_text(encoding="utf-8") + (h.queue / "batches" / "2026-09-29-1000" / "batch.md").read_text(
        encoding="utf-8")
    text += "".join(p.read_text(encoding="utf-8") for p in h.queue.glob("*/part.*"))
    assert out["chosen"] and not MONEY.search(text)


def test_a_question_that_fails_the_5_point_check_is_never_sent(three_parts):
    h = three_parts
    rec = Q.load("idea-002", "U5", h.queue)
    rec["questions"][0]["pictures"][0]["texts"].append("you win Rs 400")              # money on the picture
    rec["questions"][1]["pictures"][0]["last_candle"] = "2019-03-11 12:05:00"         # a candle after the entry
    Q.save(rec, h.queue)
    out = batch.make(dry=True, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)
    assert out["failing"] == {"idea-002-U5-1": {"no money": "money on picture 1"},
                              "idea-002-U5-2": {"ends at entry": "picture 1 does not end at the entry candle"}}
    assert "idea-002-U5-1" not in out["chosen"] and out["chosen"][0] == "idea-001-U2-1"
    assert all(all(v is None for v in p.values()) for p in out["five_points"].values())


def test_a_picture_already_shown_is_never_sent_again(three_parts):
    h = three_parts
    h.shown.write_text(json.dumps({"date": "2026-09-01", "subject": "idea-002 U5 1", "pixel_sha256": "zz"}) + "\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="shown already"):
        batch.make(dry=False, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)


# ================================================================================================ 5. his reply

@pytest.mark.parametrize("words,photo,kind,letter", [
    ("B", False, "letter", "B"), ("b.", False, "letter", "B"), ("Option A", False, "letter", "A"),
    ("B, because they are too near", False, "letter", "B"), ("A lot of times it works", False, "words", None),
    ("C", False, "words", None), ("It's in my documents", False, "documents", None),
    ("already told you this", False, "documents", None), ("", True, "drawing", None),
    ("B because", False, "words", None)])
def test_reading_his_reply_never_guesses(words, photo, kind, letter):
    assert interpret.read_reply(words, photo, ["A", "B"]) == (kind, letter)


@pytest.mark.parametrize("words,kind", [("yes", "yes"), ("Yes.", "yes"), ("ok", "yes"), ("yes but not always", "words"),
                                        ("No, too near", "no"), ("maybe", "words")])
def test_only_a_plain_yes_confirms(words, kind):
    assert interpret.read_confirm(words, False) == kind


def outbox_lines(h):
    return [json.loads(x) for x in h.outbox.read_text(encoding="utf-8").splitlines()] if h.outbox.exists() else []


def test_confirm_before_lock(three_parts):
    h = three_parts
    u2 = lambda: next(u for u in store.load_form("idea-001").unclear if u.id == "U2")   # noqa: E731
    said = interpret.on_answer({"qid": "idea-001-U2-1", "idea": "idea-001"}, "B", False, base=h.queue,
                               outbox=h.outbox, now=NOW)
    assert said.startswith("Saved. 1 more question") and u2().answer is None and outbox_lines(h) == []
    said = interpret.on_answer({"qid": "idea-001-U2-2", "idea": "idea-001"}, "b", False, base=h.queue,
                               outbox=h.outbox, now=NOW)
    assert said.startswith("Read as B") and u2().answer is None                 # NOT locked on the letter
    (line,) = outbox_lines(h)
    assert line["qid"] == "idea-001-U2-confirm" and line["photo"].endswith("confirm-B.jpg")
    assert line["text"].startswith("I read your answer as B: at least 7 candles apart.")
    assert not MONEY.search(line["text"])
    said = interpret.on_answer({"qid": "idea-001-U2-confirm"}, "yes but only in London", False, base=h.queue,
                               outbox=h.outbox, now=NOW)
    assert said.startswith("Not locked") and u2().answer is None                # only a plain YES locks
    rec = Q.load("idea-001", "U2", h.queue)
    rec["confirm"]["status"] = "sent"                                           # (a session sent it again)
    Q.save(rec, h.queue)
    said = interpret.on_answer({"qid": "idea-001-U2-confirm"}, "Yes", False, base=h.queue, outbox=h.outbox, now=NOW)
    assert said.startswith("Locked: B") and u2().answer == "B"
    assert store.read_state("idea-001")["state"] == "CONFIRMED & LOCKED"
    assert "LOCKED U2 = B" in (config.IDEAS_DIR / "idea-001" / "answers.md").read_text(encoding="utf-8")


def test_answers_that_differ_lock_nothing(three_parts):
    h = three_parts
    interpret.on_answer({"qid": "idea-001-U2-1"}, "A", False, base=h.queue, outbox=h.outbox, now=NOW)
    said = interpret.on_answer({"qid": "idea-001-U2-2"}, "B", False, base=h.queue, outbox=h.outbox, now=NOW)
    assert said.startswith("Your answers differ (A on question 1, B on question 2)")
    assert outbox_lines(h) == [] and Q.load("idea-001", "U2", h.queue)["status"] == "answers differ"


def test_withdrawn_questions_get_new_numbers_and_a_late_reply_changes_nothing(three_parts, monkeypatch):
    h = three_parts
    batch.make(dry=False, budget=10, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)
    ask.withdraw("idea-001", "U2", "he said the pictures look the same", now=NOW, base=h.queue)
    rec = Q.load("idea-001", "U2", h.queue)
    assert rec["status"] == "withdrawn" and rec["questions"] == [] and len(rec["withdrawn"]) == 2
    assert all("withdrawn-" in p["file"] and Path(p["file"]).is_file() for q in rec["withdrawn"]
               for p in q["pictures"])                                        # shown pictures are kept
    said = interpret.on_answer({"qid": "idea-001-U2-1"}, "B", False, base=h.queue, outbox=h.outbox, now=NOW)
    assert said.startswith("That picture was taken back") and outbox_lines(h)[-1]["qid"] != "idea-001-U2-confirm"
    agree = SimpleNamespace(text=lambda: "", held={}, years=(2016, 2020), counts={"A": 1}, same=0, different=2,
                            share_same=0.0, moments=["m1", "m2", "m3"], agree=False)
    monkeypatch.setattr(disagree, "compare", lambda *a, **k: agree)
    monkeypatch.setattr(ask, "choose", lambda *a, **k: [(SimpleNamespace(direction="sell", score=1.0, why="w"),
                                                         SimpleNamespace(end=pd.Timestamp(f"2019-0{i}-01 10:00"),
                                                                         signals={"A": 1, "B": None}))
                                                        for i in (1, 2, 3)])
    monkeypatch.setattr(ask, "_render", lambda *a, **k: {"file": "x.jpg", "letter": "A"})
    rec = ask.make_part("idea-001", "U2", market=SimpleNamespace(bars=lambda tf: None), search=fake_search(),
                        now=NOW, log=lambda s: None, base=h.queue)
    assert [q["qid"] for q in rec["questions"]] == ["idea-001-U2-3", "idea-001-U2-4"]   # never 1, 2 again


def test_in_my_documents_becomes_a_fresh_exam_row(three_parts):
    h = three_parts
    said = interpret.on_answer({"qid": "idea-001-U2-1"}, "it's in my documents", False, base=h.queue,
                               outbox=h.outbox, now=NOW)
    assert said.startswith("Sorry, I should have found it")
    row = json.loads(config.SEARCH_FRESH_EXAM.read_text(encoding="utf-8"))
    assert row["asked"] == "idea-001-U2-1" and row["cite"] is None and row["his_words"] == "it's in my documents"


def test_the_real_idea001_questions_were_taken_back_and_kept():
    """idea #1's U2 questions (in git): sent, then taken back on his words - first 'Both images are same' (inbox
    0912), then 'figure out by machine itself' (inbox 1155). Every shown picture is kept; nothing waits to be sent."""
    path = config.ROOT / "plan" / "work" / "question-queue" / "idea-001-U2" / "part.json"
    if not path.is_file():
        pytest.skip("idea-001-U2 not made on this copy")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["status"] == "withdrawn" and rec["questions"] == [] and len(rec["withdrawn"]) == 4
    for q in rec["withdrawn"]:
        assert all((config.ROOT / p["file"]).is_file() for p in q["pictures"])
    assert batch.five_points(rec["withdrawn"][-1]) == dict.fromkeys(batch.FIVE)       # the one-picture design
    assert next(u for u in Idea.load(FORM).unclear if u.id == "U2").kind == "version"   # never asked again


# ================================================================================================ the bot, end to end

def test_through_the_bot_a_letter_then_the_confirm_picture_then_yes_locks(three_parts):
    h = three_parts
    batch.make(dry=False, budget=4, now=NOW, base=h.queue, outbox=h.outbox, shown=h.shown)
    tg = FakeTelegram()
    where = inbox.Where(inbox=h.tmp / "learned-inbox", pictures=h.tmp / "pictures", bot=h.tmp / "bot",
                        outbox=h.outbox, intake=h.tmp / "intake.jsonl", questions=h.queue, shown=h.shown)
    bot = B.Bot(tg, TEJAS, where, echo=False)
    bot.poll_once()
    photos = [s for s in tg.sent if s.get("photo")]
    assert len(photos) == 4 and len(h.shown.read_text(encoding="utf-8").splitlines()) == 4   # shown = really sent
    q1 = next(s for s in photos if s["photo"].endswith("q1.jpg") and "idea-001-U2" in s["photo"])
    q2 = next(s for s in photos if s["photo"].endswith("q2.jpg") and "idea-001-U2" in s["photo"])
    tg.text("B", reply_to=q1["message_id"])
    tg.text("B", reply_to=q2["message_id"])
    bot.poll_once()
    confirm = [s for s in tg.sent if s.get("photo", "").endswith("confirm-B.jpg")]
    assert len(confirm) == 1 and next(u for u in store.load_form("idea-001").unclear if u.id == "U2").answer is None
    tg.text("yes", reply_to=confirm[0]["message_id"])
    bot.poll_once()
    assert next(u for u in store.load_form("idea-001").unclear if u.id == "U2").answer == "B"
    assert "Locked: B" in tg.sent[-1]["text"]
