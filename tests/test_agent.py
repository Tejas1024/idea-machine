"""L14.2 - the research agent: the plan first, unclear -> a queued question, exactly the plan, the final message.

The prompt's tests: the plan (hypothesis, <= 20 versions over dimensions 1 and 10, <= 8 slices chosen in advance, the
stop rule) is SAVED before anything runs; an unclear meaning becomes a queued question, never a guess; the final
message = the report + every version from plan/work/trials/; research-log.md; pure noise with a small budget ->
"chance". Claude is never called here: every test gives the agent a planner of its own (tests/conftest.py refuses the
real one). The prices are L14.1's hand-made ones (tests/test_mcp.py: a fake compile, the REAL tester, the real referee).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from machine import config
from machine.agent import noise
from machine.agent import research as R
from machine.ideas import store
from machine.intake import state as S
from machine.mcp import server as SRV
from machine.mcp import tools as T
from machine.mcp import versions as V
from machine.referee import splits, trials
from machine.talker import llm
from tests.test_mcp import bench, check_fn, lab, make_idea  # noqa: F401 - lab is a fixture

PARTS = ("U1", "U2", "U3")                         # idea #1's three VERSION parts (the test ideas copy its form)
HYP = "Selling or buying after the sweep, the break and the gap makes money after the Rs 40 cost, in most years."


def raw_plan(n: int = 3, start: int = 10, slices: list | None = None, unclear: list | None = None,
             changes: list | None = None) -> dict:
    """A plan in Claude's answer shape: n versions, each a different wait for the break (dimension 10)."""
    return {"hypothesis": HYP,
            "versions": [{"name": f"wait {k} bars for the break", "why": "his words give no wait; near the form's",
                          "readings": [{"part": p, "reading": "A"} for p in PARTS],
                          "changes": changes if changes is not None else
                          [{"path": "steps.brk.max_wait_bars", "value": k}]} for k in range(start, start + n)],
            "slices": slices if slices is not None else [
                {"name": "session", "dimension": "session", "how": "Asia / London / New York by the entry time",
                 "why": "his sessions decide his levels"}],
            "stop_rule": "It stops when every planned version ran, or when three in a row cannot run.",
            "unclear": unclear or []}


def scripted(*plans):
    """A planner that answers with these plans in turn (the last one again after that)."""
    calls = []

    def planner(idea, *, refused=None, previous=None):
        calls.append({"idea": idea, "refused": refused, "previous": previous})
        raw = plans[min(len(calls), len(plans)) - 1]
        return R.Planned(None if raw is None else json.loads(json.dumps(raw)),
                         {"model": "a test planner", "turns": 3, "seconds": 0.1,
                          "tool_calls": [{"tool": "get_idea", "input": {"idea": idea}, "error": False, "said": ""}]},
                         None if raw is not None else "no answer (test)")
    planner.calls = calls
    return planner


@pytest.fixture
def home(lab, monkeypatch):
    monkeypatch.setattr(config, "QUESTION_QUEUE_DIR", lab.work / "question-queue")
    return lab


# ================================================================================================= the plan first

def test_the_plan_is_saved_before_anything_runs_and_only_the_plan_runs(home):
    make_idea("idea-101", slices=[])
    out = R.run("idea-101", planner=scripted(raw_plan(3)), bench=bench(check=check_fn), say=lambda *_: None)
    assert out.status == "reported", out.said
    plan = R.load_plan("idea-101")
    assert plan["sha256"] == R.sha_of(plan) and plan["money_seen_before"] == []
    assert len(plan["versions"]) == 3 and plan["hypothesis"] == HYP and plan["stop_rule"]["code"] == list(R.STOP_RULE)
    recs = V.records("idea-101")
    assert [r["version_id"] for r in recs] == [v["version_id"] for v in plan["versions"]]   # exactly the plan
    assert all(plan["sha256"][:12] in r["why"] for r in recs)
    files = [f for f in trials.load() if f.get("idea_id") == "idea-101"]
    assert files and min(f["made_at"] for f in files) >= plan["made_at"]
    assert min(r["proposed_at"] for r in recs) >= plan["made_at"]
    log = R.log_path("idea-101").read_text(encoding="utf-8")
    assert log.index("· the plan was saved") < log.index("· version 1 of 3 ran") < log.index("· the referee:")
    assert [s.name for s in store.chosen_slices("idea-101")] == ["session"]       # chosen in advance, written first
    opened = json.loads((config.IDEAS_DIR / "idea-101" / "check-opened.json").read_text(encoding="utf-8"))
    assert plan["sha256"][:12] in opened["who"]                 # the check years opened ONCE, for the referee's pick


def test_the_planner_is_given_only_the_planning_tools():
    from mcp import Client
    import asyncio

    async def go():
        async with Client(SRV.build(None, plan=True)) as c:
            return sorted(t.name for t in (await c.list_tools()).tools)
    assert asyncio.run(go()) == ["get_idea", "list_blocks", "search_docs"]
    with pytest.raises(ValueError, match="read-only"):
        SRV.build("idea-101", plan=True)
    srv = R.planning_server()
    assert srv["args"][-2:] == ["machine.mcp", "--plan"] and srv["env"]["PYTHONPATH"] == str(config.ROOT)


def test_the_agent_call_has_no_built_in_tool_and_only_the_servers_given(monkeypatch, tmp_path):
    seen = {}
    stream = [
        {"type": "system", "subtype": "init", "mcp_servers": [{"name": "research", "status": "connected"}]},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name":
                                                       "mcp__research__get_idea", "input": {"idea": "idea-101"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{...}"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t2", "name": "StructuredOutput",
                                                       "input": {"x": 1}}]}},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "structured_output": {"x": 1},
         "usage": {"input_tokens": 10, "output_tokens": 5}, "modelUsage": {"claude-opus-5": {}}},
    ]

    def runner(args, **kw):
        seen["args"] = args
        cfg = Path(args[args.index("--mcp-config") + 1])
        seen["cfg"] = json.loads(cfg.read_text(encoding="utf-8"))
        seen["env"] = kw["env"]
        return SimpleNamespace(stdout="\n".join(json.dumps(x) for x in stream), stderr="", returncode=0)
    monkeypatch.setattr(llm, "claude_exe", lambda: "claude.exe")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-pass")
    a = llm.agent_call(system="s", user="u", schema={"type": "object"}, servers={"research": R.planning_server()},
                       allowed=("mcp__research__get_idea",), model="claude-opus-5", effort="high", runner=runner,
                       work=tmp_path)
    args = seen["args"]
    assert args[args.index("--tools") + 1] == "" and "--strict-mcp-config" in args
    assert args[args.index("--allowedTools") + 1] == "mcp__research__get_idea"
    assert list(seen["cfg"]["mcpServers"]) == ["research"] and "ANTHROPIC_API_KEY" not in seen["env"]
    assert a.output == {"x": 1} and a.error is None and a.turns == 3
    assert [c["tool"] for c in a.tool_calls] == ["get_idea"]          # the answer itself is not a tool call
    assert not list(tmp_path.glob("*.json"))                          # the config file is removed after the call
    stream[0]["mcp_servers"][0]["status"] = "failed"                  # L14.1: a server that did not start
    b = llm.read_agent_stream("\n".join(json.dumps(x) for x in stream), model="m", seconds=1, servers=("research",))
    assert b.output is None and "did not start" in b.error


def test_a_refused_plan_goes_back_once_then_a_session_looks(home):
    make_idea("idea-101", slices=[])
    bad = raw_plan(2, changes=[{"path": "directions", "value": "sell"}])          # a slice dressed as a version
    p = scripted(bad, bad)
    out = R.run("idea-101", planner=p, bench=bench(), say=lambda *_: None)
    assert out.status == "needs a session" and "SLICE" in out.said
    assert len(p.calls) == 2 and "SLICE" in " ".join(p.calls[1]["refused"])         # ONE fix-up with the reasons
    assert R.load_plan("idea-101") is None and V.records("idea-101") == []          # nothing saved, nothing counted
    assert not [f for f in trials.load() if f.get("idea_id") == "idea-101"]
    good = scripted(bad, raw_plan(2))
    plan = R.make_plan("idea-101", planner=good, say=lambda *_: None)
    assert plan["made_by"]["fixups"] == 1 and len(plan["versions"]) == 2


@pytest.mark.parametrize("raw,match", [
    (raw_plan(17), "at most 16"),                                                    # RESEARCH_MAX_VERSIONS
    (raw_plan(2, changes=[{"path": "settings.years", "value": 2024}]), "HIS rule"),  # his fixed rules
    (raw_plan(2, changes=[{"path": "steps.sweep.block", "value": "gap"}]), "what the idea IS"),
    ({**raw_plan(2), "hypothesis": ""}, "hypothesis is empty"),
    ({**raw_plan(1), "hypothesis": "It works with a p-value under 0.05 and the baseline."}, "code word|p-value"),
    (raw_plan(3, slices=[{"name": f"s{k}", "dimension": "time", "how": "x", "why": "y"} for k in range(9)]),
     "at most 8"),
    (raw_plan(2, slices=[{"name": "session", "dimension": "time", "how": "x", "why": "y"}]), "its dimension"),
])
def test_the_plan_is_checked_with_the_tools_own_rules(home, raw, match):
    make_idea("idea-101", slices=[])
    _, problems = R.check_plan("idea-101", raw)
    assert any(__import__("re").search(match, p) for p in problems), problems


def test_the_same_rule_twice_is_one_try(home):
    make_idea("idea-101", slices=[])
    raw = raw_plan(2)
    raw["versions"][1]["changes"] = raw["versions"][0]["changes"]
    _, problems = R.check_plan("idea-101", raw)
    assert any("same rule as version 1" in p for p in problems)


def test_a_plan_after_money_was_seen_is_refused(home):
    make_idea("idea-101", slices=[])
    T.start("idea-101")
    vid = T.propose_version("idea-101", "an earlier try", readings={p: "A" for p in PARTS})["version"]
    T.run_version("idea-101", vid, bench=bench())
    out = R.run("idea-101", planner=scripted(raw_plan(2)), bench=bench(), say=lambda *_: None)
    assert out.status == "needs a session" and "BEFORE any money" in out.said


def test_before_his_yes_the_plan_is_saved_and_nothing_runs(home):
    make_idea("idea-101", slices=[], state="CONFIRMED & LOCKED")
    out = R.run("idea-101", planner=scripted(raw_plan(2)), bench=bench(), say=lambda *_: None)
    assert out.status == "plan saved" and "runs after" in out.said
    assert R.load_plan("idea-101") is not None
    assert V.records("idea-101") == [] and store.chosen_slices("idea-101") == []   # nothing counted, nothing chosen


def test_not_locked_waits_and_never_plans(home):
    make_idea("idea-101", slices=[], state="QUESTIONS QUEUED")
    p = scripted(raw_plan(2))
    out = R.run("idea-101", planner=p, bench=bench(), say=lambda *_: None)
    assert out.status == "waits" and "never guesses" in out.said and p.calls == []


def test_an_edited_plan_is_refused(home):
    make_idea("idea-101", slices=[], state="CONFIRMED & LOCKED")
    R.run("idea-101", planner=scripted(raw_plan(2)), bench=bench(), say=lambda *_: None)
    path = R.plan_path("idea-101")
    d = json.loads(path.read_text(encoding="utf-8"))
    d["versions"] = d["versions"][:1]
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(R.ResearchError, match="changed after it was saved"):
        R.load_plan("idea-101")


def test_a_new_meaning_before_any_run_puts_the_old_plan_aside(home):
    make_idea("idea-101", slices=[], state="CONFIRMED & LOCKED")
    first = R.make_plan("idea-101", planner=scripted(raw_plan(2)), say=lambda *_: None)
    form = store.load_form("idea-101")
    d = json.loads(form.to_json())
    d["title"] = d["title"] + " (his answer changed it)"
    (store.folder("idea-101") / store.FORM).write_text(json.dumps(d), encoding="utf-8")
    second = R.make_plan("idea-101", planner=scripted(raw_plan(3)), say=lambda *_: None)
    assert second["sha256"] != first["sha256"] and len(second["versions"]) == 3
    assert list(store.folder("idea-101").glob("research-plan-replaced-*.json"))


# ================================================================================================= unclear -> asked

def test_an_unclear_meaning_is_a_queued_question_never_a_guess(home, monkeypatch):
    make_idea("idea-101", slices=[])
    monkeypatch.setattr(T, "search_docs", lambda q, hits=5: {"found": False, "answer": "NOT FOUND", "hits": [
        {"where": "strategy/strategy.md:12", "quote": "the closest line"}]})
    u = {"what": "which high counts as 'the top'", "his_words": "at the top", "readings": ["the day's high",
         "the week's high"], "why_trades_differ": "a week's high is touched far less often",
         "searched": "'top' in strategy.md: nothing"}
    p = scripted(raw_plan(3, unclear=[u]))
    out = R.run("idea-101", planner=p, bench=bench(), say=lambda *_: None)
    assert out.status == "question"
    assert V.records("idea-101") == [] and not trials.load()                     # nothing ran, nothing counted
    assert R.load_plan("idea-101")["versions"] == []                             # no version planned on a guess
    (q,) = list(Path(config.QUESTION_QUEUE_DIR).glob("idea-101-research-*.md"))
    text = q.read_text(encoding="utf-8")
    assert "| status | waiting" in text and "the week's high" in text and "the closest line" in text
    from machine.questions import queue as Q
    assert [r["qid"] for r in Q.hand_made()] == [q.stem]                          # the weekly batch lists it
    s = store.read_state("idea-101")
    assert "unclear meaning" in s["needs"]["why"]


def test_an_unclear_meaning_his_documents_answer_is_not_asked(home, monkeypatch):
    make_idea("idea-101", slices=[])
    monkeypatch.setattr(T, "search_docs", lambda q, hits=5: {"found": True, "answer": "FOUND: LEARNED #82 ...",
                                                               "hits": []})
    u = {"what": "x", "his_words": "y", "readings": ["a", "b"], "why_trades_differ": "z", "searched": "w"}
    out = R.run("idea-101", planner=scripted(raw_plan(1, unclear=[u])), bench=bench(), say=lambda *_: None)
    assert out.status == "question" and not list(Path(config.QUESTION_QUEUE_DIR).glob("*.md"))
    assert "his documents answer it" in R.log_path("idea-101").read_text(encoding="utf-8")


# ================================================================================================= the stop rule

def test_it_stops_after_three_planned_versions_that_cannot_run(home):
    from types import SimpleNamespace as NS
    make_idea("idea-101", slices=[])
    empty = bench(compile=lambda form, market, readings=None, changes=None: NS(
        version_id="v-" + __import__("machine.ideas.compile", fromlist=["prepare"]).prepare(
            form, readings, changes)[0].fingerprint()[:10], tf="5m", signals=[]))
    out = R.run("idea-101", planner=scripted(raw_plan(5)), bench=empty, say=lambda *_: None)
    assert out.status == "needs a session" and "3 planned versions in a row" in out.said
    assert len(V.records("idea-101")) == 3                                       # the 4th and 5th never proposed
    assert splits.opened_count() == 0 and store.read_state("idea-101")["state"] == "SHOWN"   # no check years used


def test_it_stops_when_the_family_budget_is_used(home, monkeypatch):
    make_idea("idea-101", slices=[])
    monkeypatch.setattr(config, "VERSION_BUDGET", 2)
    out = R.run("idea-101", planner=scripted(raw_plan(2)), bench=bench(check=check_fn), say=lambda *_: None)
    assert out.status == "reported"                                               # a plan fits the budget
    T.start("idea-101")
    with pytest.raises(V.Refused, match="budget"):                                 # the family's 2 are used
        T.propose_version("idea-101", "one more", readings={p: "A" for p in PARTS},
                          changes={"steps.brk.max_wait_bars": 30})


# ================================================================================================= noise -> chance

def test_pure_noise_with_a_small_budget_says_chance(home):
    """Hand-made random-walk prices (no edge), 3 versions: the referee's answer is "chance", and the final message is
    the report + every version from the trial files."""
    make_idea("idea-101", slices=[])
    out = R.run("idea-101", planner=scripted(raw_plan(3)), bench=bench(check=check_fn), say=lambda *_: None)
    assert out.status == "reported" and out.answer == "chance", out.said
    final = R.final_path("idea-101").read_text(encoding="utf-8")
    assert out.final.strip() == final.strip()
    plan = R.load_plan("idea-101")
    rows = R.every_version("idea-101")
    assert sorted(r["version"] for r in rows) == sorted(v["version_id"] for v in plan["versions"])
    for v in plan["versions"]:                                                     # every version, from its trial file
        row = next(r for r in rows if r["version"] == v["version_id"])
        line = next(ln for ln in final.splitlines() if ln.startswith(f"{v['k']}. {v['name']} - "))
        assert f": {row['trades']} trades" in line
    assert sum("CHOSEN by the referee" in ln for ln in final.splitlines()) == 1
    assert "before any money" in final and plan["sha256"][:12] in final
    from machine.report.build import check_words
    assert check_words(final) == []                                                # easy English (#113)
    assert "the referee: chance" in R.log_path("idea-101").read_text(encoding="utf-8")


def test_a_second_run_does_not_run_anything_again(home):
    make_idea("idea-101", slices=[])
    R.run("idea-101", planner=scripted(raw_plan(2)), bench=bench(check=check_fn), say=lambda *_: None)
    n = len(trials.load())
    again = R.run("idea-101", planner=scripted(raw_plan(5)), bench=bench(check=check_fn), say=lambda *_: None)
    assert again.status == "reported" and splits.opened_count() == 1
    assert len(V.records("idea-101")) == 2                                       # the saved plan, not a new one
    assert len(trials.load()) >= n


def test_send_puts_the_report_and_every_version_in_the_outbox(home):
    make_idea("idea-101", slices=[])
    out = R.run("idea-101", planner=scripted(raw_plan(2)), bench=bench(check=check_fn), send=True,
                say=lambda *_: None)
    assert out.status == "reported" and store.read_state("idea-101")["state"] == "REPORTED"
    lines = S._read_jsonl(S.outbox_path())
    whys = [ln.get("why") for ln in lines]
    assert whys == ["L10 report", "L14.2 every version"]
    assert len(lines[1]["text"]) <= R.TELEGRAM_MAX and "The referee chose ONE: version" in lines[1]["text"]


# ================================================================================================= the intake

def test_the_intake_runs_the_research_agent_after_his_yes(home, monkeypatch):
    make_idea("idea-101", slices=[])
    monkeypatch.setattr(config, "RESEARCH_AGENT", True)
    calls = []

    def research(idea, **kw):
        calls.append((idea, kw["send"], kw["tell"]))
        store.set_state(idea, "REPORTED", "test", waits_for="T6")
        return R.Outcome(idea, "reported", "chance: the referee chose v-x", "final", "chance", "sha")
    t = S.Tools(research=research, log=lambda *_: None)
    assert S.step("idea-101", t) is True
    assert calls == [("idea-101", True, True)] and S.state_of("idea-101")["state"] == "REPORTED"


def test_the_intake_tries_a_planning_call_with_no_answer_again_then_a_session_looks(home, monkeypatch):
    make_idea("idea-101", slices=[])
    monkeypatch.setattr(config, "RESEARCH_AGENT", True)
    t = S.Tools(research=lambda idea, **kw: R.Outcome(idea, "no answer", "his usage limit"), log=lambda *_: None)
    for _ in range(config.RESEARCH_TRIES - 1):
        assert S.step("idea-101", t) is False and not S.state_of("idea-101").get("needs")
    S.step("idea-101", t)
    assert "no answer 3 times" in S.state_of("idea-101")["needs"]["why"]


# ================================================================================================= the noise exam's parts

def test_the_noise_prices_are_weekdays_of_2016_2023_only():
    m1 = noise.walk_m1(seed=3, start="2023-12-28", end="2023-12-31 23:59")
    assert set(m1.index.dayofweek) <= {0, 1, 2, 3, 4} and m1.index[-1].year == 2023
    assert list(m1.columns) == ["open", "high", "low", "close", "vol", "spread"]
    assert (m1.high >= m1[["open", "close"]].max(axis=1)).all() and (m1.low <= m1[["open", "close"]].min(axis=1)).all()


def test_the_noise_sandbox_puts_every_folder_back(tmp_path):
    before = {k: getattr(config, k) for k in noise.FOLDERS}
    with noise.sandbox(tmp_path):
        assert all(str(getattr(config, k)).startswith(str(tmp_path)) for k in noise.FOLDERS)
    assert {k: getattr(config, k) for k in noise.FOLDERS} == before


def test_a_change_that_changes_nothing_is_left_out(home):
    make_idea("idea-101", slices=[])
    plan, problems = R.check_plan("idea-101", raw_plan(1, changes=[{"path": "steps.brk.max_wait_bars", "value": 30}]))
    assert problems == [] and plan["versions"][0]["changes"] == {}        # 30 is the form's own wait: the same rule
