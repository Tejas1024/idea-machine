"""His fixed rules and the changeable settings (L6.1): a form, test or version that changes a FIXED rule is refused
with his words (LEARNED #121); every quote is his, word for word; the lists match the old knobs.py exactly."""
from __future__ import annotations

import dataclasses
import importlib.util
import re

import pytest

from machine import config, rules, settings
from machine.rules import FixedRuleError
from machine.settings import SettingError


def _old_knobs():
    spec = importlib.util.spec_from_file_location("old_knobs", config.ROOT / "old" / "strategy_lab" / "knobs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # tests may import old code to compare (likhith.md R2)
    return mod


def _norm(text: str) -> str:
    text = text.replace("**", "")
    text = re.sub(r"(?m)^> ?", "", text)
    return re.sub(r"\s+", " ", text)


# ---- the two lists are the old knobs.py lists ------------------------------------------------------------------

def test_19_fixed_and_31_changeable_the_same_names_as_knobs():
    knobs = _old_knobs()
    assert len(rules.FIXED) == 19 and set(rules.FIXED) == set(knobs.FIXED)
    assert len(settings.CHANGEABLE) == 31 and set(settings.CHANGEABLE) == set(knobs.CHANGEABLE)


def test_every_old_knob_location_is_named_in_its_new_home():
    knobs = _old_knobs()
    for name, k in knobs.FIXED.items():
        where = k["at"][-1]
        r = rules.FIXED[name]
        assert where in r.old_code or where in r.aliases or where.lower() in (a.lower() for a in r.aliases), name
    for name, k in knobs.CHANGEABLE.items():
        where = k["at"][-1]
        s = settings.CHANGEABLE[name]
        assert where in s.source or where in s.aliases or where == name, name


# ---- his words, word for word ----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(rules.ALL))
def test_every_quote_is_his_word_for_word(name):
    r = rules.ALL[name]
    if r.words is None:
        assert r.evidence, f"{name}: no words, so it must say what shows it is his"
        return
    texts = [_norm((config.ROOT / f).read_text(encoding="utf-8")) for f in r.words_from]
    assert texts, name
    for piece in r.words.split(" … "):
        assert any(_norm(piece).strip() in t for t in texts), f"{name}: not found word for word: {piece!r}"


def test_every_rule_names_its_lesson_old_code_and_who_chose_the_number():
    for name, r in rules.ALL.items():
        assert r.taught and r.old_code and r.number and r.means, name


# ---- a change to a fixed rule is refused, with his words -----------------------------------------------------

def test_entry_edge_far_is_refused_with_his_65_words():
    with pytest.raises(FixedRuleError) as e:
        settings.resolve({"entry_edge": "far"})
    msg = str(e.value)
    assert "where_in_the_gap_he_buys" in msg and "LEARNED #65" in msg
    assert "I don't wait till that deep of fvg . I'll enter at the beginning of fvg" in msg
    assert "LEARNED #121" in msg


@pytest.mark.parametrize("name", sorted(rules.ALL))
def test_every_fixed_rule_and_every_old_name_for_it_is_refused(name):
    for asked in (name, *rules.ALL[name].aliases):
        with pytest.raises(FixedRuleError, match=re.escape(name)):
            settings.resolve({asked: "anything"})
        with pytest.raises(FixedRuleError):                       # even the same value: a version never carries them
            rules.refuse_fixed({asked: rules.ALL[name].value})


def test_a_refusal_without_his_sentence_shows_his_marks_instead():
    msg = rules.refusal("REGRAB", "kill")
    assert "D5 A" in msg and "cannot be set to 'kill'" in msg


def test_fixed_rules_cannot_be_changed_in_place():
    with pytest.raises(TypeError):
        rules.FIXED["where_in_the_gap_he_buys"] = rules.FIXED["what_a_triangle_is"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.FIXED["where_in_the_gap_he_buys"].value = "far"
    with pytest.raises(TypeError):
        rules.get("what_a_hammer_is")["wick_min"] = 0.1
    with pytest.raises(TypeError):
        settings.CHANGEABLE["years"] = None


# ---- settings ----------------------------------------------------------------------------------------------------

def test_every_setting_says_who_chose_it():
    for name, s in settings.ALL.items():
        assert s.who and s.source and s.means, name
        assert s.status in ("his", "test"), name
        if s.status == "his":
            assert "Tejas" in s.who or "his" in s.who.lower(), name


def test_settings_agree_with_config():
    assert settings.get("cost_per_position") == config.COST_INR_PER_POSITION == 40
    assert settings.get("lot_size") == config.LOT_STEP
    assert settings.get("years") == (2016, 2023)


def test_his_newest_words_win_over_the_old_code():
    assert settings.get("which_leg_may_close_while_covered") == "any_rr"    # #134 over #105
    assert settings.get("where_profit_is_measured_from") == "current"       # #134
    assert settings.get("the_cover_follows_the_trendline") == "line"        # #69 (T12 M27)
    assert settings.get("which_gap_of_the_move") == "first"                 # #115, #124, L4 answer (inbox 0622)
    assert settings.get("first_visit_max_bars") is None                     # #68: time does not matter


def test_resolve_a_version():
    r = settings.resolve({"how_much_room": 2.0, "GAP_PICK": "second"})
    assert r["how_much_room"] == 2.0 and r["which_gap_of_the_move"] == "second"
    assert r.changed == ("how_much_room", "which_gap_of_the_move")
    assert len(r.breaks_his_rules) == 1 and "which_gap_of_the_move" in r.breaks_his_rules[0]
    assert settings.resolve().breaks_his_rules == ()
    assert settings.resolve()["how_much_room"] == 1.0                       # today's values untouched


def test_resolve_refuses_what_it_does_not_know_or_allow():
    with pytest.raises(SettingError, match="not a setting"):
        settings.resolve({"magic": 1})
    with pytest.raises(SettingError, match="cannot be"):
        settings.resolve({"which_gap_of_the_move": "third"})
    with pytest.raises(SettingError, match="don't touch those years"):
        settings.resolve({"years": [2016, 2024]})
    with pytest.raises(SettingError, match="0.01, 0.02"):
        settings.resolve({"lot_size": 0.015})
    with pytest.raises(SettingError, match="not entries"):
        settings.resolve({"entries_on": ["1", "X"]})
    assert settings.resolve({"lot_size": 0.02})["lot_size"] == 0.02


def test_no_name_is_both_a_setting_and_a_fixed_rule():
    for name, s in settings.ALL.items():
        for a in (name, *s.aliases):
            assert rules.fixed_name(a) is None, a
