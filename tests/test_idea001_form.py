"""L7b: idea #1 as a form (tests/idea001_form.py) - his words word for word, the saved form is the form, every reading
compiles, the signals do not peek on real prices, and the reading with his line takes his round-3 yes and leaves his
no. Counts only - no money (lesson 16)."""
from __future__ import annotations

import pytest

from machine import config, guards
from machine.ideas import compile as C
from machine.ideas.form import Idea, open_parts, reading_choices
from tests import idea001_form as F
from tests.test_ideas import signal_rows

SAVED = config.ROOT / "plan" / "work" / "ideas" / "idea-001"


def test_his_words_in_the_form_are_word_for_word():
    assert F.check_quotes() == []
    form = F.the_form()
    assert F.W21B in form.words and F.T3_NEAR == next(u for u in form.unclear if u.id == "U2").words


def test_the_form_is_his_seven_steps_and_the_saved_one():
    form = F.the_form()
    assert [s.block for s in form.steps] == ["sweep", "level_waiting", "break_of_structure", "gap_of_the_move"]
    assert form.entry.at == "gap.near" and form.exit.kind == "hedge" and form.exit.at == "brk.extreme"
    assert [t.at for t in form.targets] == ["other.price"]
    assert [r.name for r in form.do_not_enter] == ["target_already_reached", "target_reached_first",
                                                    "grab_ran_too_far"]
    assert [u.id for u in open_parts(form)] == ["U1", "U2", "U3"] and len(reading_choices(form)) == 18  # inbox 1155
    assert len(form.slices) == 8 and "Already seen: NO" in form.notes
    saved = Idea.load(SAVED / "form.json")
    assert saved.rule() == form.rule() and saved.unclear == form.unclear and saved.slices == form.slices
    assert (SAVED / "readings.md").is_file() and "₹" not in (SAVED / "readings.md").read_text(encoding="utf-8")


@pytest.mark.data
def test_every_reading_compiles_on_real_prices_without_peeking():
    from machine.data.bars import load_bars
    m1 = load_bars("1m", "2019-02-04", "2019-03-29")
    market = C.Market.from_m1(m1)
    form = F.the_form()
    seen = set()
    for rd in reading_choices(form):
        out = C.compile_idea(form, market, readings=rd)
        assert out.signals, rd
        seen.add(out.version_id)
    assert len(seen) == 18                                          # 18 readings = 18 different rules

    def signals_of(bars):
        return signal_rows(C.compile_idea(form, C.Market.from_m1(bars), readings={"U1": "C", "U2": "B",
                                                                                  "U3": "B"}).signals)

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2, (4 * n) // 5])


@pytest.mark.data
@pytest.mark.parametrize("u2,missed", [("A", []), ("C", [24, 25, 33])])
def test_the_line_takes_his_round3_yes_and_leaves_his_no(u2, missed):
    """U1=B (0.25 of a day), U3=A on 2016-2020: his 5 round-3 no are never taken; with no minimum between the two
    touches (U2=A) all his 10 yes are, with his 'around 7 candles' (U2=C, inbox 2026-09-29-1155) 3 are not - #24 and
    #33 (touches 3 and 4 candles apart) and #25 (6 apart)."""
    market = C.Market.load(*F.YEARS)
    out = C.compile_idea(F.the_form(), market, readings={"U1": "B", "U2": u2, "U3": "A"})
    m5 = market.bars("5m")
    ex = [e for e in F.read_examples() if e.rnd == 3]
    j = F.judge(out, F.entries(out.signals, m5), ex, m5)
    assert [e.k for e in ex if e.answer == "yes" and j[e.k] == "not taken"] == missed
    assert [e.k for e in ex if e.answer == "no" and j[e.k] != "not taken"] == []
