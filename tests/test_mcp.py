"""L14.1 - the research tools (MCP server) with the guards inside.

The prompt's tests: every guard; tricks to reach 2024+ or the check years early all fail. The tools run on hand-made
bars here (a fake compile, the REAL tester, the real referee for the tune and the check on hand-made bars) - only the
last test reads the real prices (💻). Every trick is also called through the real MCP protocol (the SDK's client, in
process), and one through a real server process over stdio, the way Claude Code starts it.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import functools
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from machine import config
from machine.data import bars as BARS
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import Idea
from machine.intake import shown
from machine.mcp import server as SRV
from machine.mcp import tools as T
from machine.mcp import versions as V
from machine.mcp.versions import Refused
from machine.referee import judge, prior, splits, trials, tune
from machine.tester import run as replay_on
from tests.test_referee import signals_for, walk

FORM = config.ROOT / "plan" / "work" / "ideas" / "idea-001" / "form.json"
READ = {"U1": "A", "U2": "A", "U3": "A"}                 # a reading for each of idea #1's three VERSION parts
M1 = walk("2019-03-11", days=3)                          # hand-made 2019 bars: the tune years
M1_CHECK = walk("2021-03-08", days=3)                    # hand-made 2021 bars: a check year


# ================================================================================================= the lab

@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A throwaway plan/work/ and data/mcp/; hand-made prices; this run's target reset after each test."""
    work = tmp_path / "work"
    monkeypatch.setattr(config, "WORK_DIR", work)
    monkeypatch.setattr(config, "IDEAS_DIR", work / "ideas")
    monkeypatch.setattr(config, "MCP_DIR", tmp_path / "mcp")
    monkeypatch.setattr(config, "SEALED_LOG", tmp_path / "sealed_access.log")
    clock = SimpleNamespace(t=dt.datetime(2026, 10, 1, 9, 0).astimezone())

    def tick() -> str:                                    # every saved state one minute later: "older" is real
        clock.t += dt.timedelta(minutes=1)
        return clock.t.isoformat(timespec="seconds")

    monkeypatch.setattr(store, "_now", tick)
    yield SimpleNamespace(tmp=tmp_path, work=work, bench=bench())
    T.SESSION.target, T.SESSION.who = None, T.Session().who


def make_idea(idea: str, *, state: str = "SHOWN", yes: bool = True, parent: str | None = None,
              slices: list | None = None) -> Idea:
    d = json.loads(FORM.read_text(encoding="utf-8")) | {"id": idea, "status": state}
    if parent:
        d["parent_idea"] = parent
    if slices is not None:
        d["slices"] = slices
    form = Idea.load(d)
    store.create(form)
    if state == "SHOWN":
        shown.save({"idea": idea, "n": 1, "qid": shown.qid(idea), "status": "yes" if yes else "sent"})
    return form


def fake_compile(form, market, readings=None, changes=None):
    """The compiler's version id for these readings and changes, and hand-made signals (no blocks are run)."""
    f = C.prepare(form, readings, changes)[0]
    vid = "v-" + f.fingerprint()[:10]
    every = 25 + int(vid[2:4], 16) % 20
    return SimpleNamespace(version_id=vid, tf=f.timeframe,
                           signals=signals_for(M1, form.id, vid, form.parent_idea, every=every))


def fake_bars(tf, lo, hi):
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    days = (hi.normalize() - lo.normalize()).days + 1
    m1 = walk(str(lo.date()), days=days)
    out = m1 if tf == "1m" else BARS.aggregate(m1, tf)
    return out.loc[lo:hi]


def bench(**over) -> T.Workbench:
    b = T.Workbench(load_market=lambda tfs: C.Market.from_m1(M1), compile=fake_compile,
                    tester=lambda sigs, tf, note: replay_on(sigs, tf=tf, m1=M1, note=note),
                    tune=functools.partial(tune.run_versions, m1=M1), load_bars=fake_bars,
                    prices_hash=lambda: "hand-made 2019", learn=lambda idea, log=print: None, log=lambda *_: None)
    for k, v in over.items():
        setattr(b, k, v)
    return b


def check_fn(idea, *, who, log=print):
    """The check step on hand-made 2021 bars through the real referee (opens the check years once, like L10)."""
    t = tune.latest(idea)
    chosen = t["result"]["chosen"]
    parent = t.get("parent_idea")
    if splits.family_opening(idea, parent) is None:
        splits.open_check(idea, chosen, parent, who=who, why="test: the tune's choice")
    return judge.judge_check(signals_for(M1_CHECK, idea, chosen, parent), tf="5m",
                             key=prior.key_for_form(store.load_form(idea)), m1=M1_CHECK, n_random=8)


def proposed(idea: str, n: int, start: int = 10) -> list[str]:
    """n versions of the idea, each a different wait for the break (dimension 10)."""
    return [T.propose_version(idea, f"test {k}", readings=READ, changes={"steps.brk.max_wait_bars": k})["version"]
            for k in range(start, start + n)]


def refused(fn, *args, match: str, **kwargs) -> str:
    with pytest.raises(Refused, match=match) as err:
        fn(*args, **kwargs)
    return str(err.value)


def via_mcp(idea: str | None, calls: list[tuple[str, dict]], b: T.Workbench | None = None) -> list:
    """Each call through the real MCP protocol (the SDK's own client, in process)."""
    from mcp import Client

    async def go():
        srv = SRV.build(idea, bench=b)
        async with Client(srv) as c:
            return [await c.call_tool(name, args) for name, args in calls]
    return asyncio.run(go())


def text(result) -> str:
    return " ".join(getattr(x, "text", "") for x in result.content)


# ================================================================================================= the server

def test_the_server_gives_exactly_the_research_tools(lab):
    from mcp import Client

    async def go():
        async with Client(SRV.build(None)) as c:
            return await c.list_tools()
    names = sorted(t.name for t in asyncio.run(go()).tools)
    assert names == sorted(["get_idea", "list_blocks", "search_docs", "propose_version", "run_version", "list_trials",
                            "draw_chart", "write_report", "choose_slices"])


def test_an_argument_a_tool_does_not_have_is_refused_and_written_down(lab):
    make_idea("idea-101")
    (r,) = via_mcp("idea-101", [("run_version", {"idea": "idea-101", "version": "v-x", "unlock": True,
                                                 "use": "final-exam", "who": "Tejas", "why": "final exam"})])
    assert r.is_error and "has no argument 'unlock', 'use', 'who', 'why'" in text(r)
    log = (lab.tmp / "mcp" / "refusals.jsonl").read_text(encoding="utf-8")
    assert "run_version" in log and "unlock" in log
    assert not config.SEALED_LOG.exists()                 # nothing was unlocked, so nothing was logged as an unlock


def test_one_run_is_about_one_idea(lab):
    make_idea("idea-101")
    make_idea("idea-102")
    refused(T.propose_version, "idea-101", "x", readings=READ, match="has no idea")      # read-only without --idea
    T.start("idea-101")
    for fn, args in ((T.propose_version, ("idea-102", "x")), (T.run_version, ("idea-102", "v-x")),
                     (T.draw_chart, ("idea-102", "v-x")), (T.write_report, ("idea-102",)),
                     (T.choose_slices, ("idea-102", [], "x"))):
        refused(fn, *args, match="this research run is about idea-101")
    assert T.get_idea("idea-102")["idea"] == "idea-102"   # another idea can be READ
    with pytest.raises(Refused, match="not an idea id"):
        T.start("../idea-101")
    refused(T.get_idea, "idea-101/../../trials", match="not an idea id")
    refused(T.get_idea, "belief-news-trap", match="not an idea id")


# ================================================================================================= 2024 and later

@pytest.mark.parametrize("start,end,why", [
    ("2024-01-02 10:00", "2024-01-02 12:00", "No, don't touch those years"),
    ("2023-12-31 22:00", "2024-01-01 02:00", "No, don't touch those years"),               # across the new year
    ("2023-12-31 23:30:00-05:00", "2023-12-31 23:50:00-05:00", "No, don't touch those years"),  # 2024 in UTC
    ("now", "now", "No, don't touch those years"),
    ("today", "tomorrow", "No, don't touch those years"),
    ("2019-03-11 10:00+05:30", "2019-03-11 12:00+05:30", "without a time zone"),
    (1704067200, 1704070800, "as text"),                                                    # 2024 as a number
    ("2019-02-30 10:00", "2019-03-01 10:00", "not a date"),
])
def test_the_sealed_years_are_refused_however_they_are_asked_for(lab, start, end, why):
    make_idea("idea-101")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    msg = refused(T.draw_chart, "idea-101", vid, start=start, end=end, bench=lab.bench, match=why)
    assert "unlock=True" not in msg                       # no hint how the sealed years open elsewhere
    assert not config.SEALED_LOG.exists()


def test_a_version_can_never_carry_the_years(lab):
    make_idea("idea-101")
    T.start("idea-101")
    for years in ([2016, 2024], [2016, 2023], (2021, 2023)):
        refused(T.propose_version, "idea-101", "x", readings=READ, changes={"settings.years": years},
                match="HIS rule")


def test_a_workbench_that_reaches_outside_2016_2020_is_not_trusted(lab):
    make_idea("idea-101")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    late = walk("2020-12-30", days=4)                     # into 2021
    b = bench(load_market=lambda tfs: C.Market.from_m1(late))
    refused(T.run_version, "idea-101", vid, bench=b, match="2021|check years")
    sealed = bench(load_market=lambda tfs: SimpleNamespace(window=lambda: (pd.Timestamp("2019-01-01"),
                                                                           pd.Timestamp("2024-02-01"))))
    refused(T.run_version, "idea-101", vid, bench=sealed, match="No, don't touch those years")

    def leaky_compile(form, market, readings=None, changes=None):
        c = fake_compile(form, market, readings, changes)
        c.signals = signals_for(M1_CHECK, form.id, c.version_id, form.parent_idea)
        return c
    refused(T.run_version, "idea-101", vid, bench=bench(compile=leaky_compile), match="2021")
    leaky_tester = bench(tester=lambda sigs, tf, note: replay_on(
        [s.model_copy(update={"known_at": s.known_at + dt.timedelta(days=730)}) for s in sigs], tf=tf,
        m1=walk("2021-03-10", days=3), record=False))
    with pytest.raises(Exception, match="2021|check years|hidden"):
        T.run_version("idea-101", vid, bench=leaky_tester)
    assert not config.SEALED_LOG.exists()


# ================================================================================================= 2021-2023, early

def test_the_check_years_are_shut_until_write_report_opens_them(lab):
    make_idea("idea-101")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    refused(T.run_version, "idea-101", vid, how="check", bench=lab.bench, match="open only through write_report")
    refused(T.run_version, "idea-101", vid, how="2021-2023", bench=lab.bench, match="open only through write_report")
    refused(T.draw_chart, "idea-101", vid, start="2021-03-08 10:00", end="2021-03-08 14:00", bench=lab.bench,
            match="stay hidden while choosing")
    refused(T.draw_chart, "idea-101", vid, start="2020-12-31 10:00", end="2021-01-01 05:00", bench=lab.bench,
            match="stay hidden while choosing")
    T.run_version("idea-101", vid, bench=lab.bench)
    assert not (config.IDEAS_DIR / "idea-101" / "check-opened.json").exists()


def test_write_report_opens_the_check_years_once_for_the_version_the_referee_chose(lab):
    make_idea("idea-101")
    T.start("idea-101", who="the test agent")
    vids = proposed("idea-101", 3)
    for v in vids:
        T.run_version("idea-101", v, bench=lab.bench)
    b = bench(check=check_fn)
    out = T.write_report("idea-101", bench=b)
    opened = json.loads((config.IDEAS_DIR / "idea-101" / "check-opened.json").read_text(encoding="utf-8"))
    assert opened["version_id"] == out["chosen"] in vids and opened["who"] == "the test agent"
    assert out["judged"]["mode"] == "check" and store.read_state("idea-101")["state"] == "CHECK OPENED"
    assert Path(out["page"]).is_file() and out["judged"]["answer"] == "chance"     # a random walk: no edge
    again = T.write_report("idea-101", bench=b)            # a second report: the check years are NOT judged again
    assert "judged ONCE" in again["judged"]["said"]
    assert splits.opened_count() == 1
    checks = [f for f in trials.load() if f["kind"] == "referee" and f["idea_id"] == "idea-101"]
    assert len(checks) == 1
    # now open for THIS version only: a chart of 2021 for it, never for another version
    T.draw_chart("idea-101", out["chosen"], start="2021-03-08 10:00", end="2021-03-08 14:00", bench=b)
    other = next(v for v in vids if v != out["chosen"])
    refused(T.draw_chart, "idea-101", other, start="2021-03-08 10:00", end="2021-03-08 14:00", bench=b,
            match="later version")


def test_a_family_that_used_its_check_years_is_judged_by_walk_forward(lab):
    make_idea("idea-101")
    make_idea("idea-102", parent="idea-101")
    T.start("idea-101")
    vids = proposed("idea-101", 2)
    for v in vids:
        T.run_version("idea-101", v, bench=lab.bench)
    T.write_report("idea-101", bench=bench(check=check_fn))
    T.start("idea-102")
    calls = []

    def wf(versions, **kw):
        calls.append((sorted(versions), kw))
        return SimpleNamespace(word="chance", answer=SimpleNamespace(why="test"), judged_on="walk-forward (test)",
                               text=lambda: "walk-forward text", trial_path=None)
    (v2,) = [T.propose_version("idea-102", "child", readings=READ, changes={"steps.gap.max_wait_bars": 4})["version"]]
    T.run_version("idea-102", v2, bench=lab.bench)
    b = bench(walk_forward=wf, check=lambda *a, **k: pytest.fail("the family's check years were used: no check"),
              build=lambda idea: SimpleNamespace(text="report", page_path=lab.tmp / "r.html",
                                                 picture={"path": lab.tmp / "r.jpg"}))
    out = T.write_report("idea-102", bench=b)
    assert out["judged"]["mode"] == "walk-forward" and calls and calls[0][0] == [v2]
    assert splits.opened_count() == 1 and not (config.IDEAS_DIR / "idea-102" / "check-opened.json").exists()
    # and the parent's 2021-2023 numbers stay hidden from the child's run
    rows = T.list_trials("idea-102")["trials"]
    hidden = [r for r in rows if r["idea"] == "idea-101" and r["kind"] == "referee"]
    assert hidden and all("hidden" in r for r in hidden)


def test_list_trials_shows_the_check_years_only_to_the_run_of_that_idea(lab):
    make_idea("idea-101")
    make_idea("idea-103")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    T.run_version("idea-101", vid, bench=lab.bench)
    T.write_report("idea-101", bench=bench(check=check_fn))
    mine = [r for r in T.list_trials("idea-101")["trials"] if r["kind"] == "referee"]
    assert mine and "hidden" not in mine[0] and mine[0]["answer"]
    T.start("idea-103")                                   # another run reads idea-101: its hidden years stay hidden
    theirs = [r for r in T.list_trials("idea-101")["trials"] if r["kind"] == "referee"]
    assert theirs and "hidden" in theirs[0] and "answer" not in theirs[0] and "note" not in theirs[0]
    T.start(None)
    assert all("hidden" in r for r in T.list_trials("idea-101")["trials"] if r["kind"] == "referee")
    # a kind whose years cannot be read is never shown; one that touches 2024 never either
    from machine import trials as tf
    tf.record(kind="money-plan", idea_id="idea-101", version_id=vid, parent_idea=None, settings={},
              data={"prices": "x"}, result={"total": 123})
    tf.record(kind="tester", idea_id="idea-101", version_id=vid, parent_idea=None,
              settings={"window": {"first_bar": "2023-06-01", "last_bar": "2024-02-01"}}, data={"prices": "x"},
              result={"per_year": {"2024": {"net_inr": 1}}})
    T.start("idea-101")
    rows = T.list_trials("idea-101")["trials"]
    assert all("hidden" in r for r in rows if r["kind"] == "money-plan")
    assert any("sealed" in r.get("hidden", "") for r in rows)


# ================================================================================================= money waits for him

@pytest.mark.parametrize("state,yes", [("CONFIRMED & LOCKED", False), ("SHOWN", False)])
def test_no_money_before_his_yes_to_the_three_setups(lab, state, yes):
    make_idea("idea-101", state=state, yes=yes)
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)                       # proposing sees no money: allowed on a locked idea
    for fn, args in ((T.run_version, ("idea-101", vid)), (T.write_report, ("idea-101",)),
                     (T.draw_chart, ("idea-101", vid))):
        refused(fn, *args, match="lesson 16")
    assert T.get_idea("idea-101")["money_allowed"] is False


def test_versions_need_a_locked_meaning(lab):
    make_idea("idea-101", state="QUESTIONS QUEUED")
    T.start("idea-101")
    refused(T.propose_version, "idea-101", "x", readings=READ, match="not locked")


# ================================================================================================= what a version is

@pytest.mark.parametrize("changes,why", [
    ({"settings.where_in_the_gap_he_buys": "far"}, "LEARNED #65"),       # his fixed rule, with his words
    ({"settings.entry_edge": "far"}, "FIXED"),                          # its old name
    ({"settings.ENTRY_EDGE": "far"}, "FIXED"),
    ({"steps.gap.params.fvg_shape": 1}, "FIXED"),
    ({"exit.management.sessions": ["London"]}, "FIXED"),
    ({"entry.edge": "far"}, "FIXED"),
    ({"directions": ["sell"]}, "SLICE"),
    ({"filters": [{"name": "session", "params": {"sessions": ["London"]}}]}, "SLICE|FIXED"),
    ({"slices": []}, "choose them BEFORE running"),
    ({"do_not_enter": []}, "what the idea IS"),
    ({"steps.take.block": "equal_levels"}, "what the idea IS"),
    ({"steps.take.params.kinds": ["session"]}, "what the idea IS"),
    ({"entry.at": "brk.extreme"}, "what the idea IS"),
    ({"exit.at": "gap.far"}, "what the idea IS"),
    ({"parent_idea": None}, "what the idea IS"),
    ({"words": "buy everything"}, "what the idea IS"),
    ({"settings.not_a_setting": 1}, "not a setting"),
    ({"settings.near_zone_adr": "0.1"}, "give a number"),
    ({"settings.grab_ran_too_far_adr": float("nan")}, "give a number"),
    ({"steps.brk.max_wait_bars": True}, "whole number"),
    ({"steps.brk.max_wait_bars": -1}, "whole number"),
    ({"steps.take.max_wait_bars": 5}, "first step"),
    ({"targets.0.ratio": 2.0}, "ratio"),
    ({"timeframe": "2m"}, "a timeframe is one of"),
    ({"steps.nope.max_wait_bars": 5}, "no step"),
])
def test_a_version_changes_only_the_timeframe_or_the_ideas_own_numbers(lab, changes, why):
    make_idea("idea-101")
    T.start("idea-101")
    refused(T.propose_version, "idea-101", "x", readings=READ, changes=changes, match=why)
    assert V.records("idea-101") == []                    # a refused proposal leaves no record and no try


def test_readings_are_never_guessed(lab):
    make_idea("idea-101")
    T.start("idea-101")
    refused(T.propose_version, "idea-101", "x", readings={"U1": "A"}, match="never guesses")
    refused(T.propose_version, "idea-101", "x", readings={**READ, "U9": "A"}, match="not unclear parts")
    refused(T.propose_version, "idea-101", "x", readings={**READ, "U1": "Z"}, match="not one of its readings")
    refused(T.propose_version, "idea-101", "", readings=READ, match="reason")


def test_good_versions_say_what_they_are(lab):
    make_idea("idea-101")
    T.start("idea-101")
    a = T.propose_version("idea-101", "a longer wait for the break", readings=READ,
                          changes={"steps.brk.max_wait_bars": 40, "timeframe": "15m",
                                   "settings.grab_ran_too_far_adr": 0.22, "entry.max_wait_bars": "no limit"})
    assert a["new"] and a["version"] == C.version_of(store.load_form("idea-101"), READ, a["changes"])
    assert a["dimensions"]["timeframe"] == "the timeframe (dimension 1)"
    assert a["dimensions"]["steps.brk.max_wait_bars"] == "a wait (dimension 10)"
    assert "steps.brk.max_wait_bars = 40 (was 30)" in a["about"] and a["timeframe"] == "15m"
    again = T.propose_version("idea-101", "the same rule, other words", readings=READ, changes=a["changes"])
    assert not again["new"] and again["version"] == a["version"] and len(V.records("idea-101")) == 1


# ================================================================================================= the budgets

def test_the_family_gets_20_versions_tried_or_proposed(lab):
    make_idea("idea-101")
    make_idea("idea-102", parent="idea-101")
    T.start("idea-101")
    proposed("idea-101", 15)
    T.start("idea-102")
    proposed("idea-102", 5, start=60)                     # the child counts in the same family
    assert T.get_idea("idea-102")["version_budget"]["left"] == 0
    refused(T.propose_version, "idea-102", "21st", readings=READ, changes={"steps.brk.max_wait_bars": 99},
            match="budget is 20")
    T.start("idea-101")
    refused(T.propose_version, "idea-101", "21st", readings=READ, changes={"steps.brk.max_wait_bars": 98},
            match="budget is 20")


def test_the_same_rule_under_another_idea_of_the_family_is_one_try(lab):
    make_idea("idea-101")
    make_idea("idea-102", parent="idea-101")
    T.start("idea-101")
    (v,) = proposed("idea-101", 1)
    T.start("idea-102")
    refused(T.propose_version, "idea-102", "same", readings=READ, changes={"steps.brk.max_wait_bars": 10},
            match="already .* of idea-101")


def test_an_idea_with_the_same_blocks_timeframe_and_entry_as_an_older_one_is_its_version(lab):
    make_idea("idea-101")
    make_idea("idea-104")                                  # no parent written - but the same idea, seen later
    parent, why = V.parent_of("idea-104")
    assert parent == "idea-101" and "section D" in why
    assert V.parent_of("idea-101")[0] is None
    T.start("idea-101")
    proposed("idea-101", 19)
    T.start("idea-104")
    proposed("idea-104", 1, start=80)
    refused(T.propose_version, "idea-104", "x", readings=READ, changes={"steps.brk.max_wait_bars": 90},
            match="budget is 20")
    (vid,) = [r["version_id"] for r in V.records("idea-104")]
    T.run_version("idea-104", vid, bench=lab.bench)        # its trial file carries the family link
    assert trials.parents(trials.load())["idea-104"] == "idea-101"


def test_run_version_never_passes_the_budget_and_never_runs_a_version_not_proposed(lab):
    make_idea("idea-101")
    T.start("idea-101")
    refused(T.run_version, "idea-101", "v-0000000000", bench=lab.bench, match="propose it first")
    for k in range(20):                                    # 20 versions tried elsewhere (their trial files)
        replay_on(signals_for(M1, "idea-101", f"v-elsewhere{k}"), tf="5m", m1=M1)
    refused(T.propose_version, "idea-101", "21st", readings=READ, changes={"steps.brk.max_wait_bars": 11},
            match="budget is 20")


def test_at_most_8_slices_and_only_before_any_money(lab):
    from machine.learner import conditions
    make_idea("idea-101", slices=[])
    T.start("idea-101")
    names = list(conditions.BY_NAME)
    six = [{"name": n, "dimension": conditions.BY_NAME[n].dimension, "how": "test", "why": "test"}
           for n in names[:6]]
    out = T.choose_slices("idea-101", six, "chosen before the test")
    assert out["slice_budget"]["chosen"] == 6
    three = [{"name": n, "dimension": conditions.BY_NAME[n].dimension, "how": "t"} for n in names[6:9]]
    refused(T.choose_slices, "idea-101", three, "x", match="at most 8")
    refused(T.choose_slices, "idea-101", [{"name": "made_up_split", "dimension": "time", "how": "t"}], "x",
            match="never guessed")
    n0 = names[6]
    wrong = next(d for d in ("direction", "session", "time") if d != conditions.BY_NAME[n0].dimension)
    refused(T.choose_slices, "idea-101", [{"name": n0, "dimension": wrong, "how": "t"}], "x", match="dimension")
    refused(T.choose_slices, "idea-101", [{"name": n0, "dimension": "timeframe", "how": "t"}], "x",
            match="not a slice")
    refused(T.choose_slices, "idea-101", [six[0]], "x", match="chosen twice")
    assert [s.name for s in store.with_chosen_slices(store.load_form("idea-101")).slices] == names[:6]
    (vid,) = proposed("idea-101", 1)
    T.run_version("idea-101", vid, bench=lab.bench)        # money seen: no slice may be chosen now
    refused(T.choose_slices, "idea-101", three[:1], "x", match="money was already seen")


# ================================================================================================= running, drawing

def test_run_version_tests_2016_2020_and_leaves_one_trial_file(lab):
    make_idea("idea-101")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    out = T.run_version("idea-101", vid, bench=lab.bench)
    assert not out["already_run"] and out["years"] == "2016-2020" and out["numbers"]["trades"] > 0
    files = [f for f in trials.load() if f["idea_id"] == "idea-101"]
    assert len(files) == 1 and files[0]["kind"] == "tester" and files[0]["version_id"] == vid
    assert "L14.1 research" in files[0]["note"] and "Rs 40" in out["cost"]
    again = T.run_version("idea-101", vid, bench=lab.bench)
    assert again["already_run"] and len([f for f in trials.load() if f["idea_id"] == "idea-101"]) == 1


def test_draw_chart_draws_a_trade_or_a_stretch_of_2016_2020(lab):
    make_idea("idea-101")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    refused(T.draw_chart, "idea-101", vid, trade=1, bench=lab.bench, match="run it first")
    T.run_version("idea-101", vid, bench=lab.bench)
    pic = T.draw_chart("idea-101", vid, trade=1, bench=lab.bench)
    path = lab.tmp / "mcp" / "idea-101" / "charts"
    assert any(path.glob("*.jpg")) and pic["shown_to_tejas"] is False
    assert open(next(path.glob("*.jpg")), "rb").read(3) == b"\xff\xd8\xff"          # a real JPEG (#106)
    refused(T.draw_chart, "idea-101", vid, trade=10_000, bench=lab.bench, match="closed trades")
    w = T.draw_chart("idea-101", vid, start="2019-03-11 08:00", end="2019-03-11 20:00", bench=lab.bench)
    assert w["setups_marked"] > 0
    refused(T.draw_chart, "idea-101", vid, start="2019-03-01 08:00", end="2019-03-20 20:00", bench=lab.bench,
            match="at most 3 days")
    refused(T.draw_chart, "idea-101", vid, trade=1, start="2019-03-11 08:00", end="2019-03-11 20:00",
            bench=lab.bench, match="ONE thing")
    refused(T.draw_chart, "idea-101", "v-notmine00", trade=1, bench=lab.bench, match="propose it first")


def test_his_fixed_rules_are_refused_with_his_words_through_mcp(lab):
    make_idea("idea-101")
    (r,) = via_mcp("idea-101", [("propose_version", {"idea": "idea-101", "why": "far edge",
                                                     "readings": READ,
                                                     "changes": {"settings.where_in_the_gap_he_buys": "far"}})])
    assert r.is_error and "REFUSED" in text(r) and "LEARNED #65" in text(r)


def test_the_tricks_fail_through_mcp_too(lab):
    make_idea("idea-101")
    T.start("idea-101")
    (vid,) = proposed("idea-101", 1)
    calls = [("draw_chart", {"idea": "idea-101", "version": vid, "start": "2024-03-01 10:00",
                             "end": "2024-03-01 12:00"}),
             ("draw_chart", {"idea": "idea-101", "version": vid, "start": "2022-03-01 10:00",
                             "end": "2022-03-01 12:00"}),
             ("run_version", {"idea": "idea-101", "version": vid, "how": "check"}),
             ("propose_version", {"idea": "idea-101", "why": "x", "readings": READ,
                                  "changes": {"settings.years": [2016, 2025]}}),
             ("write_report", {"idea": "idea-101", "send": True, "open_check_years": True}),
             ("run_version", {"idea": "idea-102", "version": vid}),
             ("list_trials", {"idea": "../trials"})]
    out = via_mcp("idea-101", calls, lab.bench)
    assert all(r.is_error for r in out), [text(r) for r in out if not r.is_error]
    assert "No, don't touch those years" in text(out[0]) and "stay hidden" in text(out[1])
    assert not config.SEALED_LOG.exists() and splits.opened_count() == 0
    lines = (lab.tmp / "mcp" / "refusals.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= len(calls)


@pytest.mark.private
def test_the_server_runs_as_its_own_process_over_stdio(lab):
    """The way Claude Code starts it (machine/mcp/README.md): a process speaking MCP on stdin / stdout."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    env = {"MLBOT_MCP_DIR": str(lab.tmp / "mcp-process"), "PYTHONIOENCODING": "utf-8"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "machine.mcp", "--idea", "idea-001"],
                                   cwd=str(config.ROOT), env=env)

    async def go():
        async with Client(params) as c:
            names = sorted(t.name for t in (await c.list_tools()).tools)
            g = await c.call_tool("get_idea", {"idea": "idea-001"})
            trick = await c.call_tool("draw_chart", {"idea": "idea-001", "version": "v-8cc96cf812",
                                                      "start": "2025-06-02 10:00", "end": "2025-06-02 12:00"})
            return names, g, trick
    names, g, trick = asyncio.run(go())
    assert len(names) == 9 and not g.is_error and g.structured_content["idea"] == "idea-001"
    assert trick.is_error and "No, don't touch those years" in text(trick)
    assert (lab.tmp / "mcp-process" / "refusals.jsonl").is_file()


# ================================================================================================= the key and the parts

def test_every_block_has_a_name_in_the_old_list_so_every_idea_has_a_key():
    assert set(C.BLOCKS) <= set(prior.FORM_BLOCKS)
    d = json.loads(FORM.read_text(encoding="utf-8"))
    d["steps"][2:2] = [{"id": "adr", "what": "at his ADR zone", "block": "premium_discount", "tf": "5m",
                        "max_wait_bars": 0}]
    k = prior.key_for_form(Idea.load(d))
    assert "adr" in k.blocks and k.entry == "gap:near"


def test_a_version_on_another_timeframe_is_tuned_on_its_own_chart(lab):
    """Dimension 1: the tune runs each version's tester on its own timeframe and keeps it."""
    m1 = M1
    a = signals_for(m1, "idea-x", "v-a")
    b = signals_for(m1, "idea-x", "v-b", every=45)
    t = tune.run_versions({"v-a": a, "v-b": b}, tf="5m", tfs={"v-b": "15m"}, m1=m1,
                          changes={"v-b": {"timeframe": "15m"}})
    assert t.versions["v-b"]["timeframe"] == "15m" and t.versions["v-b"]["changes"] == {"timeframe": "15m"}
    tester_files = {f["version_id"]: f["settings"]["timeframe"] for f in trials.load() if f["kind"] == "tester"}
    assert tester_files == {"v-a": "5m", "v-b": "15m"}


# ================================================================================================= the real prices (💻)

@pytest.mark.data
def test_a_real_version_of_idea_1_runs_on_2016_2020_and_is_drawn(lab):
    """Idea #1's family as it really is (its 18 versions' trial files copied into the test's folder): one more
    version through the real compiler and tester on 2016-2020, then one of its trades drawn. ~3 minutes."""
    shutil.copytree(config.ROOT / "plan" / "work" / "ideas" / "idea-001", config.IDEAS_DIR / "idea-001")
    config.TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    for p in (config.ROOT / "plan" / "work" / "trials").glob("*-idea-001-*.json"):
        shutil.copy(p, config.TRIALS_DIR / p.name)
    T.start("idea-001")
    g = T.get_idea("idea-001")
    assert g["version_budget"]["tried"] == 18 and g["version_budget"]["left"] == 2
    v = T.propose_version("idea-001", "the break may come later", readings={"U1": "A", "U2": "A", "U3": "B"},
                          changes={"steps.brk.max_wait_bars": 40})
    out = T.run_version("idea-001", v["version"])
    assert out["years"] == "2016-2020" and out["numbers"]["trades"] > 100
    assert set(out["numbers"]["per_year"]) == {"2016", "2017", "2018", "2019", "2020"}
    f = [x for x in trials.load() if x["version_id"] == v["version"]]
    assert len(f) == 1 and f[0]["settings"]["window"]["last_bar"] < "2021"
    assert T.get_idea("idea-001")["version_budget"]["left"] == 1
    pic = T.draw_chart("idea-001", v["version"], trade=1)
    assert pic["picture"].endswith(".jpg")
    refused(T.draw_chart, "idea-001", v["version"], start="2022-03-01 10:00", end="2022-03-01 12:00",
            match="later version")                       # opened for v-8cc96cf812, not this one
