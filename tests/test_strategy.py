"""L19d - his whole strategy through the research loop.

What the whole is (his settled entries together; the three open words wait, never chosen by money), its versions
(only his management's test rows, M18's test part, the machine's numbers and an entry's own numbers - his fixed rules,
a waiting entry and a timeframe refused), one trade at a time ACROSS his entries (a newer setup of another entry
cancels a waiting order; a setup while a trade is on is skipped), the plan saved before any money and exactly the plan
run, the referee's choice by the highest Deflated Sharpe of the hedged trades, the check years' answer "chance - needs
fresh years" (already seen), the reports in easy English with no money for a waiting entry.
Claude is never called here: every test gives the planner (tests/conftest.py refuses the real one). The market is
hand-made (tests/test_hedging.py's fake facts, the REAL hedging tester and tester); one test reads real gold (data).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from machine import config, guards
from machine.entries import forms
from machine.mcp import server as SRV
from machine.mcp import tools as T
from machine.mcp.versions import Refused
from machine.referee import power, prior, splits, stats, trials
from machine.report.build import check_words
from machine.strategy import report as RP
from machine.strategy import research as SR
from machine.strategy import whole as W
from machine.tester import engine
from tests.test_hedging import Market
from tests.test_mcp import lab  # noqa: F401 - a fixture

TYPE1, ZB, ZE = forms.TYPE1.id, forms.ZONE_BREAK.id, forms.ZONE_ENTRY.id
HYP = "His entries together, with his hedge, make money after the Rs 40 cost of every position, in most years."


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway plan/work/ (the trial files and ideas folders are already throwaway: tests/conftest.py); the prices'
    fingerprint of the hand-made market; a fast power table."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(SR, "_data", lambda: {"prices": "hand-made 1-minute bars (tests/test_strategy.py)"})
    monkeypatch.setattr(power, "caught_share", lambda *a, **k: 0.0)
    return tmp_path


# ================================================================================================= a hand-made market

def bars(t0: str, *points, tail: int = 60) -> pd.DataFrame:
    """1-minute bars through (minute, price) points, straight between them; spread $0.20."""
    mins = [m for m, _ in points]
    px = np.round(np.interp(np.arange(mins[-1] + tail + 2), mins, [p for _, p in points]), 2)
    o, c = px[:-1], px[1:]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "spread": 20,
                         "vol": 1}, index=pd.date_range(t0, periods=len(o), freq="1min", name="server_time"))


def sig(t0: str, entry: str, k: int, minute: int, direction: str, px: float, hedge: float, target: float):
    from machine.tester import Signal
    return Signal(id=f"{entry}.v-test.S{k}", known_at=pd.Timestamp(t0) + pd.Timedelta(minutes=minute),
                  direction=direction, entry_type="limit", entry_price=px, expiry_bars=10**6, stop=hedge,
                  targets=(target,), idea_id=entry, version_id="v-test")


def scene(t0: str = "2019-03-11 10:00") -> tuple[pd.DataFrame, dict]:
    """Price: flat 2000, down to 1990, up to 2012, flat, down to 2000. His setups:
    A Type 1 buy (known minute 5) waits at 1995 · B zone entry buy (minute 10) at 1996 - NEWER, so A's waiting order is
    cancelled (M39) · C zone break buy (minute 70) while B's trade is on - skipped (M18) · D Type 1 sell (minute 125)
    after B closed. No hedge is ever reached: every trade ends at its target."""
    m1 = bars(t0, (0, 2000), (30, 2000), (60, 1990), (120, 2012), (150, 2012), (220, 2000))
    s = {TYPE1: [sig(t0, TYPE1, 1, 5, "buy", 1995, 1980, 2008), sig(t0, TYPE1, 4, 125, "sell", 2011.5, 2018, 2003)],
         ZE: [sig(t0, ZE, 2, 10, "buy", 1996, 1985, 2009)],
         ZB: [sig(t0, ZB, 3, 70, "buy", 2000, 1988, 2010.5)]}
    return m1, s


def fake_window(m1: pd.DataFrame, label: str) -> W.Window:
    facts = Market(m1)
    facts.closers, facts.setup_signals = {}, 4
    win = W.Window(label, (pd.Timestamp(m1.index[0]).year,), None, m1, facts.book)
    win.fake_facts = facts
    return win


def fake_run(win: W.Window, v: W.Version, sigs: dict) -> W.Run:
    """The REAL replays (his hedge, a plain stop) of the version on the hand-made market."""
    facts = win.fake_facts
    h, s, R = W.replays(v, sigs, facts, {}, facts.book)
    return W.Run(v, win.label, list(win.years), sigs, h, s, facts, {}, R, 0.1)


def ids(trades) -> list[str]:
    return [t.signal_id.split(".")[-1] for t in trades]


# ================================================================================================= what the whole is

def test_the_whole_is_his_settled_entries_and_the_three_open_words_wait():
    assert [e.id for e in W.entries_in()] == [TYPE1, ZB, ZE]
    assert set(W.waiting()) == {forms.SECOND_GAP.id, forms.TRENDLINE_TOUCH.id, forms.TRIANGLE_FALSE_BREAK.id}
    why = W.waiting()                     # checked after the run: none of the three can change the answer
    assert "#154" in why[forms.TRENDLINE_TOUCH.id] and "#154" in why[forms.TRIANGLE_FALSE_BREAK.id]
    assert "same 15 trades" in why[forms.SECOND_GAP.id]
    for w in why.values():
        assert check_words(w) == []


def test_once_very_strong_is_answered_the_second_gap_takes_type_1s_place(monkeypatch):
    real = W.open_asks
    monkeypatch.setattr(W, "open_asks", lambda e: [] if e is forms.SECOND_GAP else real(e))
    got = [e.id for e in W.entries_in()]
    assert forms.SECOND_GAP.id in got and TYPE1 not in got          # never Type 1's first gap twice


def test_the_whole_is_already_seen_by_the_old_project_on_2016_2023():
    seen = prior.already_seen(W.SEEN_KEY, config.TUNE_YEARS + config.CHECK_YEARS)
    assert seen.seen and "combo_750" in {r.idea_key for r in seen.rows}
    assert prior.already_seen(W.SEEN_KEY, config.CHECK_YEARS).seen


# ================================================================================================= its versions

def test_version_1_is_his_strategy_as_written_and_a_change_to_version_1s_own_value_is_left_out():
    v1 = W.version({})
    assert v1.about.startswith("his strategy as written") and v1.trades_at_once == W.ONE and not v1.changes
    same = W.version({"exit.management.on_a_friday": "as_any_day", W.AT_ONCE: W.ONE,
                      "exit.management.how_many_covers_allowed": None})
    assert same.version_id == v1.version_id and not same.changes


def test_one_rule_is_one_version_id():
    v = W.version({"exit.management.how_many_covers_allowed": 3})
    assert v.management == {"how_many_covers_allowed": 3}
    assert W.version({"exit.management.how_many_covers_allowed": 3}).version_id == v.version_id
    assert len({v.version_id, W.version({}).version_id, W.version({"exit.management.how_many_covers_allowed": 4}).
                version_id, W.version({W.AT_ONCE: W.PER_ENTRY}).version_id}) == 4


def test_an_entrys_own_number_changes_only_that_entry():
    v = W.version({f"{TYPE1}/steps.gap.max_wait_bars": 8, "settings.order_block_max_age_bars": 60})
    assert v.entry_changes(forms.TYPE1) == {"settings.order_block_max_age_bars": 60, "steps.gap.max_wait_bars": 8}
    assert v.entry_changes(forms.ZONE_ENTRY) == {"settings.order_block_max_age_bars": 60}


@pytest.mark.parametrize("changes, match", [
    ({"exit.management.how_far_before_any_leg_may_close": 2.0}, "FIXED management rule M12"),
    ({"settings.what_a_triangle_is": 1}, "#121|his"),
    ({f"{forms.TRENDLINE_TOUCH.id}/steps.touch.max_wait_bars": 3}, "not in his whole strategy yet"),
    ({f"{forms.SECOND_GAP.id}/steps.gap.max_wait_bars": 3}, "very strong"),
    ({f"{TYPE1}/timeframe": "15m"}, "5-minute chart"),
    ({"timeframe": "15m"}, "5-minute chart"),
    ({"settings.trendline_hedge_pad_adr": 0.1}, "steep"),
    ({"exit.management.double_profit_zone_rank": 1}, "acts only when"),
    ({"settings.setup_near_level_adr": 0.1}, "no code reads"),
    ({"directions": ["buy"]}, "changes only his management"),
    ({W.AT_ONCE: "two at a time"}, "one_per_entry"),
    ({"exit.management.on_a_friday": "skip"}, "can be"),
    ({f"{TYPE1}/steps.sweep.params.kinds": 1}, "what the idea IS"),
    ({f"{TYPE1}/settings.order_block_max_age_bars": 60}, "a change of the whole"),
])
def test_a_version_is_refused_with_the_reason(changes, match):
    with pytest.raises(Refused, match=match):
        W.version(changes)


# ================================================================================================= one trade at a time

def test_one_trade_at_a_time_across_his_entries_a_newer_setup_cancels_a_waiting_order_and_one_on_skips_the_next():
    m1, s = scene()
    win = fake_window(m1, "2016-2020")
    r = fake_run(win, W.version({}), s)
    for rep in (r.hedged, r.stop):                                   # his hedge and the plain stop: the same rule
        assert ids(rep.trades) == ["S2", "S4"]
        assert (f"{TYPE1}.v-test.S1", "cancelled") in rep.not_filled  # B (zone entry) is newer: A's order goes (M39)
        assert f"{ZB}.v-test.S3" in rep.skipped                     # B's trade is on: C is skipped (M18, #55)
    for t in r.hedged.trades:
        assert t.status == "closed" and t.hedges == 0 and t.net_inr > 0
    spans = sorted((t.fill_ts, t.exit_ts) for t in r.hedged.trades)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))      # never two trades at once


def test_one_per_entry_type_is_m18s_test_part_each_entry_its_own_one_at_a_time():
    m1, s = scene()
    win = fake_window(m1, "2016-2020")
    r = fake_run(win, W.version({W.AT_ONCE: W.PER_ENTRY}), s)
    assert sorted(ids(r.hedged.trades)) == ["S1", "S2", "S3", "S4"]
    assert sorted(ids(r.stop.trades)) == ["S1", "S2", "S3", "S4"]
    assert not r.hedged.skipped and not r.hedged.not_filled


def test_each_entry_types_part_is_the_trades_it_opened():
    m1, s = scene()
    r = fake_run(fake_window(m1, "2016-2020"), W.version({}), s)
    be = W.by_entry(r)
    assert be[TYPE1]["hedged"]["trades"] == 1 and be[ZE]["hedged"]["trades"] == 1 and be[ZB]["hedged"]["trades"] == 0
    assert be[ZB]["skipped_hedged"] == 1
    total = sum(x["hedged"]["net_inr_total"] for x in be.values())
    assert total == pytest.approx(W.run_summary(r)["hedged"]["net_inr_total"])


def test_the_2x2_twins_keep_each_setups_entry_and_the_parts_add_up():
    m1, s = scene()
    win = fake_window(m1, "2016-2020")
    r = fake_run(win, W.version({}), s)
    tt = W.two_by_two(win, r, 6, log=lambda *_: None)
    assert set(tt["groups"]) == {"all", TYPE1, ZB, ZE}
    for g in tt["groups"].values():
        assert [b["arm"] for b in g["boxes"]] == list(W.H2.ARMS) and len(g["run_totals"]["hedge"]) == 6
    alls = np.array(tt["groups"]["all"]["run_totals"]["hedge"])
    parts = sum(np.array(tt["groups"][e]["run_totals"]["hedge"]) for e in (TYPE1, ZB, ZE))
    assert np.allclose(alls, parts)                                  # every random trade belongs to one entry
    assert tt["groups"]["all"]["boxes"][0]["total"] == pytest.approx(W.run_summary(r)["hedged"]["net_inr_total"])
    assert tt["random_hedged"]["runs"] == 6 and tt["seed"] == W.two_by_two(win, r, 1, log=lambda *_: None)["seed"]


# ================================================================================================= the plan first

def raw_plan(*versions, slices=None, unclear=None) -> dict:
    vs = versions or ([], [{"path": "exit.management.how_many_covers_allowed", "value": "3"}])
    return {"hypothesis": HYP,
            "versions": [{"name": f"version {k}", "why": "his table T12 lists it as a test row", "changes": list(c)}
                         for k, c in enumerate(vs, 1)],
            "slices": slices if slices is not None else [
                {"name": "session", "dimension": "session", "how": "Asia, London or New York by the entry time",
                 "why": "his sessions decide his levels (strategy.md section 5)"}],
            "stop_rule": "It stops when every planned version ran, or when three in a row cannot run.",
            "unclear": unclear or []}


def scripted(*plans):
    calls = []

    def planner(*, refused=None, previous=None):
        calls.append({"refused": refused, "previous": previous})
        raw = plans[min(len(calls), len(plans)) - 1]
        return SR.Planned(None if raw is None else json.loads(json.dumps(raw)),
                          {"model": "a test planner", "turns": 2, "seconds": 0.1,
                           "tool_calls": [{"tool": "get_strategy", "input": {}, "error": False, "said": ""}]},
                          None if raw is not None else "no answer (test)")
    planner.calls = calls
    return planner


def test_the_plan_is_saved_before_anything_runs(home):
    plan = SR.make_plan(planner=scripted(raw_plan()), say=lambda *_: None)
    assert SR.plan_path().is_file() and plan["sha256"] == SR.sha_of(plan) and not SR.files()
    assert [v["version_id"] for v in plan["versions"]] == [W.version({}).version_id,
                                                           W.version({"exit.management.how_many_covers_allowed": 3})
                                                           .version_id]
    assert plan["entries_in"] == [TYPE1, ZB, ZE] and set(plan["entries_waiting"]) == set(W.waiting())
    assert "the plan was saved (before any version ran)" in SR.log_path().read_text(encoding="utf-8")
    assert SR.make_plan(planner=scripted(None), say=lambda *_: None)["sha256"] == plan["sha256"]   # never made twice


def test_a_refused_plan_goes_back_to_claude_once_then_a_session_looks(home):
    bad = raw_plan([], [{"path": "exit.management.how_far_before_any_leg_may_close", "value": 2}])
    p = scripted(bad, bad)
    with pytest.raises(SR.ResearchError, match="refused the plan 2 times"):
        SR.make_plan(planner=p, say=lambda *_: None)
    assert len(p.calls) == 2 and any("M12" in x for x in p.calls[1]["refused"]) and not SR.plan_path().exists()
    good = scripted(bad, raw_plan())
    assert len(SR.make_plan(planner=good, say=lambda *_: None)["versions"]) == 2


@pytest.mark.parametrize("raw, match", [
    (raw_plan([{"path": "exit.management.how_many_covers_allowed", "value": 3}]), "as written"),
    (raw_plan([], [{"path": "exit.management.how_many_covers_allowed", "value": 3}],
              [{"path": "exit.management.how_many_covers_allowed", "value": "3"}]), "same rule"),
    (raw_plan(slices=[{"name": "moon", "dimension": "time", "how": "x", "why": "y"}]), "not a condition"),
    (raw_plan(slices=[{"name": "session", "dimension": "time", "how": "x", "why": "y"}]), "its dimension"),
    (raw_plan(*([[]] + [[{"path": "exit.management.how_many_covers_allowed", "value": k}] for k in range(1, 11)]
                + [[{"path": "exit.management.news_minutes_before", "value": k}] for k in (5, 10, 20, 30, 45, 60)])),
     "at most 16"),
])
def test_the_plan_is_checked_with_the_versions_own_rules(home, raw, match):
    _, problems = SR.check_plan(raw)
    assert any(match in p for p in problems), problems


def test_the_answers_shape_holds_a_slice_to_the_real_names_and_a_refusal_says_how_to_fix_it(home):
    """The research agent's fix (L10b's 8 Oct rehearsal: idea #3's slices named with sentences) for his whole strategy
    too: the answer's shape lists the real names, and code's reason names them for the fix-up call."""
    from machine.ideas.form import SLICE_DIMENSIONS
    from machine.learner import conditions
    item = SR.plan_schema()["properties"]["slices"]["items"]["properties"]
    assert item["name"]["enum"] == sorted(conditions.BY_NAME) and "friday" in item["name"]["enum"]
    assert item["dimension"]["enum"] == list(SLICE_DIMENSIONS)
    assert "enum" not in SR.PLAN_SCHEMA["properties"]["slices"]["items"]["properties"]["name"]   # the copy only
    sentence = [{"name": "The session the trade is entered in", "dimension": "session", "how": "Asia, London or New "
                 "York", "why": "his sessions decide his levels"}]
    plan, problems = SR.check_plan(raw_plan(slices=sentence))
    assert not plan["slices"] and len(problems) == 1
    assert "EXACTLY" in problems[0] and "session, new_york" in problems[0] and '"how" and "why"' in problems[0]


def test_a_plan_after_money_was_seen_is_refused(home):
    from machine import trials as trial_files
    trial_files.record(kind=W.KIND, idea_id=SR.STRATEGY, version_id="s-0000000000", parent_idea=None, settings={},
                       data={"prices": "test"}, result={})
    with pytest.raises(SR.ResearchError, match="money was already seen"):
        SR.make_plan(planner=scripted(raw_plan()), say=lambda *_: None)


def test_an_edited_plan_is_refused(home):
    SR.make_plan(planner=scripted(raw_plan()), say=lambda *_: None)
    d = json.loads(SR.plan_path().read_text(encoding="utf-8"))
    d["hypothesis"] = "something else"
    SR.plan_path().write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SR.ResearchError, match="changed after it was saved"):
        SR.load_plan()


def test_the_planning_server_has_only_the_three_reading_tools_and_shows_no_money(home):
    names = {t.name for t in SRV.build(strategy_plan=True)._tool_manager.list_tools()}
    assert names == set(T.STRATEGY_PLAN_TOOLS) == {"get_strategy", "list_blocks", "search_docs"}
    assert "get_strategy" not in {t.name for t in SRV.build(plan=True)._tool_manager.list_tools()}
    assert "get_strategy" not in {t.name for t in SRV.build()._tool_manager.list_tools()}
    d = T.get_strategy()
    text = json.dumps(d, ensure_ascii=False)
    assert "₹" not in text and "net_inr" not in text and "per_trade" not in text
    assert [e["id"] for e in d["entries_in"]] == [TYPE1, ZB, ZE]
    assert {e["id"] for e in d["entries_waiting"]} == set(W.waiting())
    assert "combo_750" in d["already_seen_by_the_old_project"]


# ================================================================================================= exactly the plan

@pytest.fixture
def planned(home, monkeypatch):
    """A saved plan; the tune window and W.run swapped for the hand-made market (the real replays)."""
    SR.make_plan(planner=scripted(raw_plan()), say=lambda *_: None)
    m1, s = scene()
    win = fake_window(m1, "2016-2020")
    ran = []

    def run(w, v, log=print):
        ran.append(v.version_id)
        return fake_run(win, v, s)
    monkeypatch.setattr(W.Window, "tune", classmethod(lambda cls, log=print: win))
    monkeypatch.setattr(W, "run", run)
    return SimpleNamespace(win=win, sigs=s, m1=m1, ran=ran)


def test_exactly_the_plan_runs_once_each_with_one_trial_file_each(planned):
    out = SR.run_plan(say=lambda *_: None)
    plan = SR.load_plan()
    assert out["stopped"] == "every planned version ran"
    assert planned.ran == [v["version_id"] for v in plan["versions"]]
    got = SR.files(W.KIND)
    assert [f["version_id"] for f in got] == planned.ran and SR.budget()["used"] == 2
    f = got[0]
    assert f["settings"]["plan"] == plan["sha256"] and f["result"]["hedged_trades"]
    assert f["settings"]["already_seen"]["seen"] and "combo_750" in f["settings"]["already_seen"]["rows"]
    SR.run_plan(say=lambda *_: None)                                  # a second run runs nothing again
    assert len(planned.ran) == 2 and len(SR.files(W.KIND)) == 2


def test_it_stops_after_three_planned_versions_in_a_row_that_cannot_run(home, monkeypatch):
    vs = [[]] + [[{"path": "exit.management.how_many_covers_allowed", "value": k}] for k in (1, 2, 3, 4)]
    SR.make_plan(planner=scripted(raw_plan(*vs)), say=lambda *_: None)
    real = W.version
    monkeypatch.setattr(W, "version", lambda ch: (_ for _ in ()).throw(Refused("broken on purpose")) if ch else
                        real(ch))
    monkeypatch.setattr(W.Window, "tune", classmethod(lambda cls, log=print: fake_window(scene()[0], "2016-2020")))
    m1, s = scene()
    monkeypatch.setattr(W, "run", lambda w, v, log=print: fake_run(fake_window(m1, "2016-2020"), v, s))
    out = SR.run_plan(say=lambda *_: None)
    assert out["stopped"].startswith("3 planned versions in a row") and len(SR.files(W.KIND)) == 1


def test_it_stops_when_the_familys_budget_is_used(planned, monkeypatch):
    monkeypatch.setattr(config, "VERSION_BUDGET", 1)
    out = SR.run_plan(say=lambda *_: None)
    assert out["stopped"] == "the family's 1 versions are used" and len(SR.files(W.KIND)) == 1


# ================================================================================================= the referee

def test_the_referee_chooses_by_the_highest_deflated_sharpe_of_the_hedged_trades_once(planned):
    SR.run_plan(say=lambda *_: None)
    ch = SR.choose(say=lambda *_: None)
    vt = {f["version_id"]: SR._tune_trades(f) for f in SR.files(W.KIND)}
    assert ch["kind"] == SR.TUNE_KIND and ch["result"]["chosen"] == stats.choose_version(vt, n_tries=2)["chosen"]
    assert SR.choose(say=lambda *_: None)["trial_id"] == ch["trial_id"]      # written once


def test_the_check_years_stay_closed_until_opened_for_the_chosen_version(planned):
    SR.run_plan(say=lambda *_: None)
    vid = SR.choose(say=lambda *_: None)["result"]["chosen"]
    with pytest.raises(guards.CheckYearsClosedError):
        W.Window.check(vid)
    splits.open_check(SR.STRATEGY, vid, None, who="test", why="chosen on 2016-2020 (test)")
    other = next(f["version_id"] for f in SR.files(W.KIND) if f["version_id"] != vid)
    with pytest.raises(guards.CheckYearsClosedError):
        W.Window.check(other)                                         # opened ONCE, for ONE version


def judged(planned):
    """The whole path on the hand-made market: run, choose, the tune 2x2, the check years opened, judged."""
    SR.run_plan(say=lambda *_: None)
    ch = SR.choose(say=lambda *_: None)
    vid = ch["result"]["chosen"]
    plan = SR.load_plan()
    v = W.version(next(p for p in plan["versions"] if p["version_id"] == vid)["changes"])
    SR.two_by_two_record(planned.win, fake_run(planned.win, v, planned.sigs), ch, n_runs=5, say=lambda *_: None)
    splits.open_check(SR.STRATEGY, vid, None, who="test", why="chosen on 2016-2020 (test)")
    m2, s2 = scene("2022-03-14 10:00")
    win2 = fake_window(m2, "2021-2023")
    tune_file = next(f for f in SR.files(W.KIND) if f["version_id"] == vid)
    return SR.judged_record(win2, fake_run(win2, v, s2), ch, tune_file, n_runs=5, say=lambda *_: None)


def test_already_seen_means_the_answer_is_chance_needs_fresh_years_with_his_hedge(planned):
    j = judged(planned)
    a = j["result"]["answer"]
    assert j["kind"] == "referee" and a["word"] == "chance — needs fresh years"
    assert [c["number"] for c in a["checks"]] == [1, 2, 3, 4, 5, 6]
    assert "does not apply" in next(c for c in a["checks"] if c["number"] == 3)["shown"]
    assert j["settings"]["judged_on"].startswith("the check years 2021-2023") and j["result"]["summary"]["trades"] == 2
    from machine.referee import project
    assert project.summary()["by_word"]["chance — needs fresh years"] == 1      # counted as a judged answer


def test_after_the_report_the_machine_decides_once_and_tells_never_asks(planned):
    judged(planned)
    line = SR.decide(say=lambda *_: None)
    text = (SR.folder() / SR.DECISION).read_text(encoding="utf-8")
    assert "KEPT ON FILE" in line and "You don't need to do anything" in line and "?" not in line   # LEARNED #162
    assert "Not a forward test" in text and "trades" in text and "I don't know why u asking these questions" in text
    (SR.folder() / SR.DECISION).write_text("his own change", encoding="utf-8")
    SR.decide(say=lambda *_: None)
    assert (SR.folder() / SR.DECISION).read_text(encoding="utf-8") == "his own change"     # written once, kept
    assert check_words(line) == []


def test_the_reports_are_made_from_the_files_in_easy_english(planned):
    judged(planned)
    paths = RP.write_all()
    out = RP.out_dir()
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[4].startswith("**The answer: chance — needs fresh years.**")   # the answer first
    assert check_words(readme.split("## The referee's rules")[0]) == []
    for name in ("whole-years.jpg", "whole-boxes.jpg", f"{TYPE1}-boxes.jpg", f"{ZE}-boxes.jpg"):
        assert (out / name).is_file()
    for e in (forms.TRENDLINE_TOUCH, forms.TRIANGLE_FALSE_BREAK, forms.SECOND_GAP):
        page = (out / f"{e.id}.md").read_text(encoding="utf-8")
        assert "Not in the whole:" in page and "₹" not in page.replace("₹40 a position", "")   # only the cost
    final = (SR.folder() / SR.FINAL).read_text(encoding="utf-8")
    assert sum("CHOSEN by the referee" in ln for ln in final.splitlines()) == 1 and check_words(final) == []
    assert all(p.exists() for p in paths)


# ================================================================================================= real gold

@pytest.mark.data
def test_on_real_gold_the_whole_never_holds_two_trades_and_its_setups_do_not_read_the_future():
    """Jan-Jun 2019, his strategy as written: the setups known by 31 Mar are the same compiled on Jan-Mar alone
    (nothing after a moment changes a setup known then), and no two hedged trades are open at once."""
    from machine.ideas.compile import Market
    v = W.version({})
    full = Market.load("2019-01-01", "2019-06-30", tfs=W.TFS)
    win = W.Window("2019H1", (2019,), full, full.bars("1m"), engine.bars_book(full.bars("1m"), W.TF))
    sigs = W.signals_of(win, v, log=lambda *_: None)
    short = Market.load("2019-01-01", "2019-03-31", tfs=W.TFS)
    win_s = W.Window("2019Q1", (2019,), short, short.bars("1m"), engine.bars_book(short.bars("1m"), W.TF))
    sigs_s = W.signals_of(win_s, v, log=lambda *_: None)
    cut = pd.Timestamp(win_s.book.last)
    for e in v.entry_ids:
        early = [s.model_dump() for s in sigs[e] if pd.Timestamp(s.known_at) <= cut]
        assert early and early == [s.model_dump() for s in sigs_s[e] if pd.Timestamp(s.known_at) <= cut], e
    r = W.run(win, v, log=lambda *_: None)
    spans = sorted((t.fill_ts, t.exit_ts or win.book.last) for t in r.hedged.trades)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert sum(len(x) for x in sigs.values()) == r.hedged.signals


# ================================================================================================= a parked idea

def test_a_parked_idea_goes_on_only_if_it_made_money_with_his_hedge_and_beat_random_with_his_hedge():
    from machine.strategy import parked as PK
    r = {"runs": 200, "his_beats_random": {"with_hedging": 199}}
    assert PK.choice(r, {"net_inr_total": 500.0})[0] == "FORWARD TEST"
    head, means = PK.choice({**r, "his_beats_random": {"with_hedging": 120}}, {"net_inr_total": 500.0})
    assert head == "KEPT ON FILE" and "120 of 200" in means
    head, means = PK.choice(r, {"net_inr_total": -1.0})
    assert head == "KEPT ON FILE" and "lost with your hedge too" in means and check_words(means) == []


def test_only_a_decided_idea_is_tested_with_his_hedge(lab):
    from machine.strategy import parked as PK
    from tests.test_mcp import make_idea
    make_idea("idea-101")
    with pytest.raises(PK.ParkedError, match="only a decided idea"):
        PK.run("idea-101", n_runs=1, write=False, log=lambda *_: None)


def test_a_version_of_an_older_idea_is_tested_with_his_hedge_on_the_setups_of_its_tune(lab, monkeypatch):
    """An idea with the same blocks, chart and entry as an older one is a version OF it (likhith.md section D): the
    research tools find the older idea and the setups carry it (parent_idea); the idea's own form does not. The test
    with his hedge compiles the chosen version again and must get the tune's exact setups - or the idea waits for a
    session after its report (found 8 Oct on made-up prices: "its setups on 2016-2020 are not the tune's any more")."""
    from machine.referee import tune
    from machine.strategy import parked as PK
    from tests.test_mcp import M1, READ, fake_compile, make_idea
    form = make_idea("idea-102", state="DECIDED")
    assert form.parent_idea is None
    comp = fake_compile(form.model_copy(update={"parent_idea": "idea-101"}), None, READ)
    tune.run_versions({comp.version_id: comp.signals}, tf="5m", readings={comp.version_id: READ},
                      changes={comp.version_id: {}}, m1=M1)
    assert tune.latest("idea-102")["parent_idea"] == "idea-101"

    class Reached(Exception):
        pass

    def bars(tf):
        raise Reached
    monkeypatch.setattr(PK, "compile_idea", fake_compile)
    monkeypatch.setattr(PK, "Market", SimpleNamespace(load=lambda *a, **k: SimpleNamespace(bars=bars)))
    with pytest.raises(Reached):              # past the check of its setups; the hedge itself is tested above
        PK.run("idea-102", n_runs=1, write=False, log=lambda *_: None)


def test_the_nightly_job_carries_the_page_of_a_test_with_his_hedge():
    """The intake tests a parked idea with his hedge by itself (L10b) and writes its page into plan/work/L19d/;
    decision.md links it. Found 9 Oct: ideas #3's and #5's pages stayed on his PC after the nightly pull request -
    the job's list (config.BOT_GIT_FILES) had no plan/work/L19d/."""
    from machine.strategy import parked as PK
    from machine.telegram import nightly
    page = PK._page_path("idea-005")
    assert page.parent == config.WORK_DIR / "L19d" and config.WORK_DIR == config.ROOT / "plan" / "work"
    for p in (page, page.with_suffix(".jpg")):
        assert nightly.matches(p.relative_to(config.ROOT).as_posix()), p.name
    assert not nightly.matches("plan/work/L19d/README.md")      # the session-made L19d pages stay a session's
