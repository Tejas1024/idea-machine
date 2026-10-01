"""L7b step 1 - the MISSING BLOCK path (machine/ideas/blocked.py): a form that needs a block that does not exist ->
BLOCKED; a line in block-requests.md; one Telegram line; STATUS says how long it waits; an unclear meaning queues its
question at once; the block built -> the idea goes back to be compiled. Everything in a temporary plan/work/."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from machine import config
from machine.ideas import blocked, store
from machine.ideas import compile as C
from machine.ideas.form import Idea
from tests.test_ideas import fake_m1, type1

NOW = dt.datetime(2026, 9, 28, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
Q = ("Would you take this sell? Look at the line through the 3 touches: is the 3rd touch the entry?",
     "I searched strategy.md section 4, LEARNED #92, #94, #127 and strategy-understanding for the trendline touch "
     "entry and found how a line is drawn but not where the entry sits - the closest line is #127.")


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "IDEAS_DIR", tmp_path / "work" / "ideas")
    return tmp_path / "work"


@pytest.fixture(scope="module")
def market():
    return C.Market.from_m1(fake_m1(days=3))


def needs_two() -> Idea:
    """A form that needs a step no prompt builds yet (-> L6-new) and a rule L19c builds."""
    d = type1(id="idea-002", title="grab + gap + a trendline touch (a test form)")
    d["steps"] = d["steps"] + [dict(id="tl", what="price touches the trendline a 3rd time", block="trendline_retest",
                                    tf="5m", max_wait_bars=10,
                                    sources=[dict(ref="LEARNED #127", words="the third touch of the trendline")])]
    d["do_not_enter"] = d["do_not_enter"] + [dict(name="halfway", words="no sell at discount")]
    return Idea.load(d)


def blocked_error(form, market) -> C.MissingBlockError:
    with pytest.raises(C.MissingBlockError) as err:
        C.compile_idea(form, market)
    return err.value


@pytest.mark.private
def test_a_missing_block_blocks_the_idea(work, market):
    form = needs_two()
    err = blocked_error(form, market)
    new = blocked.request(form, err, unclear={"trendline_retest": Q}, now=NOW)

    # 1. BLOCKED, and what it waits for
    s = store.read_state("idea-002")
    assert s["state"] == "BLOCKED" and store.load_form("idea-002").status == "BLOCKED"
    assert "trendline_retest (L6-new, either)" in s["waits_for"]
    assert "halfway (L19c, Likhith)" in s["waits_for"]

    # 2. one line per block in plan/work/block-requests.md: block, idea, his words, the prompt, its owner, the date
    rows = blocked.read_requests()
    assert [(r["block"], r["built_by"], r["owner"]) for r in rows] == [
        ("trendline_retest", "L6-new", "either"), ("halfway", "L19c", "Likhith")] == \
        [(r["block"], r["built_by"], r["owner"]) for r in new]
    assert rows[0]["words"] == "the third touch of the trendline" and rows[1]["words"] == "no sell at discount"
    assert {r["asked"] for r in rows} == {"2026-09-28"} and {r["idea"] for r in rows} == {"idea-002"}

    # 3. one Telegram line each, easy English, no money, for the bot (L9a) to send
    lines = [json.loads(x) for x in blocked.outbox_path().read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and all(x["to"] == "Tejas" for x in lines)
    assert "idea #2" in lines[0]["text"] and "L6-new" in lines[0]["text"] and "question" in lines[0]["text"]
    assert "Nothing for you to do now" in lines[1]["text"]
    assert not any(w in x["text"] for x in lines for w in ("₹", "Rs", "rupee", "profit", "loss"))

    # 4. its meaning is unclear -> the question is queued at once, with the #129 search sentence
    q = (blocked.queue_dir() / f"{blocked.question_id('idea-002', 'trendline_retest')}.md").read_text(encoding="utf-8")
    assert Q[0] in q and Q[1] in q and "(1) it unblocks a BLOCKED idea" in q and "₹" not in q
    assert not (blocked.queue_dir() / "idea-002-block-halfway.md").exists()


def test_the_same_block_is_asked_for_once(work, market):
    form = needs_two()
    blocked.request(form, blocked_error(form, market), now=NOW)
    again = blocked.request(form, blocked_error(form, market), now=NOW + dt.timedelta(days=1))
    assert again == [] and len(blocked.read_requests()) == 2
    assert len(blocked.outbox_path().read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.private
def test_status_says_how_long_it_has_waited_and_built_unblocks(work, market):
    form = needs_two()
    blocked.request(form, blocked_error(form, market), now=NOW)
    later = NOW + dt.timedelta(days=3)
    assert [r["days"] for r in blocked.waiting(later)] == [3, 3]
    assert blocked.status_lines(later) == [
        "idea-002 BLOCKED since 2026-09-28 (3 days): waits for trendline_retest (L6-new, either); halfway "
        "(L19c, Likhith)"]
    # one of the two is built: still BLOCKED, waiting for the other
    assert blocked.built("halfway", by="L19c", now=later) == []
    s = store.read_state("idea-002")
    assert s["state"] == "BLOCKED" and "halfway" not in s["waits_for"]
    assert blocked.read_requests()[1]["status"] == "built 2026-10-01 by L19c (3 days)"
    # the other too: back to FORM DRAFTED, to be compiled again (L10b resumes it)
    assert blocked.built("trendline_retest", by="L6-new", now=later + dt.timedelta(days=1)) == ["idea-002"]
    assert store.read_state("idea-002")["state"] == "FORM DRAFTED" and blocked.waiting(later) == []
    assert blocked.status_lines(later) == []
    assert [h["state"] for h in store.read_state("idea-002")["history"]] == [
        "FORM DRAFTED", "BLOCKED", "BLOCKED", "FORM DRAFTED"]


def test_a_question_needs_the_search_first(work, market):
    form = needs_two()
    err = blocked_error(form, market)
    with pytest.raises(ValueError, match="#129"):
        blocked.request(form, err, unclear={"trendline_retest": (Q[0], " ")}, now=NOW)
    with pytest.raises(ValueError, match="not one of the missing"):
        blocked.request(form, err, unclear={"zone": Q}, now=NOW)
    assert not blocked.requests_path().exists()                    # nothing written when the request is refused


def test_words_with_a_bar_survive_the_table(work, market):
    d = needs_two().model_dump(mode="json")
    d["steps"][-1]["sources"] = [dict(ref="x", words="a | b")]
    form = Idea.load(d)
    blocked.request(form, blocked_error(form, market), now=NOW)
    assert blocked.read_requests()[0]["words"] == "a | b"


def test_a_block_built_under_another_name_rewrites_the_waiting_form(work, market):
    """L6-new, 30 Sep: idea #3's form named a made-up filter (london_session_high_only); the prompt built it as the
    sweep step's params.which_session. built(..., says=) rewrites the waiting form, writes it in the history, and the
    form compiles. A rewrite that leaves the made-up name, or touches his words, is refused and nothing is written."""
    d = type1(id="idea-003", title="the London high (a test form)", directions=["sell"],
              words="sell when the London high is taken, then a break and the first gap, sell at the gap",
              filters=[dict(name="london_session_high_only", words="sell when the London high is taken")])
    d["steps"][0] = {**d["steps"][0], "params": {"kinds": ["session"]}}
    form = Idea.load(d)
    blocked.request(form, blocked_error(form, market), now=NOW)
    assert blocked.read_requests()[0]["words"] == "sell when the London high is taken"
    later = NOW + dt.timedelta(days=1)

    with pytest.raises(ValueError, match="still needs 'london_session_high_only'"):
        blocked.built("london_session_high_only", by="L6-new", now=later,
                      says={"steps.sweep.params.which_session": ["London"]})
    with pytest.raises(ValueError, match="words or id: they are his"):
        blocked.built("london_session_high_only", by="L6-new", now=later, says={"words": "x", "filters": []})
    assert blocked.read_requests()[0]["status"] == "waiting"                    # nothing written
    assert store.load_form("idea-003").filters and store.read_state("idea-003")["state"] == "BLOCKED"

    says = {"steps.sweep.params.which_session": ["London"], "filters": []}
    assert blocked.built("london_session_high_only", by="L6-new", now=later, says=says) == ["idea-003"]
    new = store.load_form("idea-003")
    assert new.filters == () and new.steps[0].params == {"kinds": ["session"], "which_session": ["London"]}
    assert new.words == form.words and new.status == "FORM DRAFTED"
    s = store.read_state("idea-003")
    assert s["state"] == "FORM DRAFTED"
    assert 'steps.sweep.params.which_session = ["London"]' in s["history"][-1]["note"]
    assert blocked.read_requests()[0]["status"] == "built 2026-09-29 by L6-new (1 day)"
    assert C.missing_in_any_reading(new) == []
    C.compile_idea(new, market)                                                 # it compiles: nothing is missing
