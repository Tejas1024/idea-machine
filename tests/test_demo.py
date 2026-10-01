"""L11 - the demo: the made-up idea through the real steps, on made-up data, writing nothing into plan/work/.

The quick steps only (search, talker, questions, a short tester run, the guards). The referee's exam has its own
tests (tests/test_referee.py); `python -m machine.demo --full` is run by hand (plan/work/L11-notes.md).
"""
from __future__ import annotations

import json

import pytest

from machine import config
from machine.demo import run as D
from machine.talker import translate as T


@pytest.fixture(scope="module")
def m1():
    prices, _ = D.made_up_prices()
    return prices


@pytest.fixture
def world(tmp_path):
    out = []
    with D.demo_world(tmp_path):
        yield out.append, out


def test_the_demo_data_is_made_up_and_says_so():
    idea = json.loads(D.IDEA_FILE.read_text(encoding="utf-8"))
    assert "MADE-UP" in idea["about"] and "written by hand" in idea["about"]
    for p in (D.NOTES_DIR / "strategy").glob("*.md"):
        assert "MADE UP" in p.read_text(encoding="utf-8").splitlines()[0]
    assert "NO edge" in json.loads(D.PRICES_FILE.read_text(encoding="utf-8"))["about"]


def test_the_made_up_prices_cover_2016_2023_and_stop_before_the_sealed_years(m1):
    assert str(m1.index[0])[:4] == "2016" and str(m1.index[-1])[:4] == "2023"
    assert m1.index[-1] < __import__("pandas").Timestamp(config.SEALED_FROM)


def test_search_talker_and_question_run_on_the_real_code(world, m1):
    say, out = world
    idea = D.load_idea()
    g = D.step_intake(idea, say)
    searches, search = D.step_search(g, say)
    assert searches and all(s.hits or s.closest for s in searches)
    d = D.step_talker(idea, g, searches, search, say)
    assert d.status == "FORM DRAFTED" and not d.problems and not d.unknown_citations
    assert [u.kind for u in d.form.unclear] == ["ask", "version"]
    assert any("REFUSED" in x for x in out)                     # the far-edge form: his fixed rule
    assert any("FixedRuleError" in x or "FIXED" in x for x in out)
    form = D.step_questions(d, idea, m1, say)
    assert any("reading A" in x and "reading B" in x for x in out)
    assert any("different trades" in x for x in out)            # the two readings make different setups
    assert form.unclear[0].answer == "A" and form.status == "ANSWERED"


def test_the_talker_reference_is_the_made_up_notes_only_inside_the_demo(tmp_path):
    real = T.reference
    with D.demo_world(tmp_path):
        assert T.reference is D.notes_reference
        assert "MADE UP" in T.reference()
    assert T.reference is real


def test_the_tester_step_counts_versions_and_writes_trial_files_only_in_the_sandbox(world, m1, monkeypatch, tmp_path):
    say, out = world
    monkeypatch.setattr(D, "TEST_WINDOW", ("2019-01-01", "2019-01-31"))
    idea = D.load_idea()
    g = D.step_intake(idea, say)
    s, fn = D.step_search(g, say)
    form = D.step_questions(D.step_talker(idea, g, s, fn, say), idea, m1, say)
    D.step_tester(form, m1, say)
    assert any("2 versions (2 of the 20 tries" in x for x in out)
    assert sum(1 for x in out if " a trade, " in x) == 2
    trials = list((tmp_path / "work" / "trials").glob("*.json"))
    assert len(trials) == 2 and all(json.loads(p.read_text(encoding="utf-8"))["idea_id"] == D.IDEA_ID
                                    for p in trials)


def test_the_guards_refuse_and_catch(world, m1):
    say, out = world
    D.step_guards(m1, say)
    text = "\n".join(out)
    assert "March 2024 -> REFUSED" in text
    assert "PAST hour's high -> passes" in text and "NEXT hour's high -> CAUGHT" in text
    assert "2021 before a version is chosen -> REFUSED" in text
