"""L17c - the forward plan: written once, locked, BEFORE any live bar, for an idea only new days can judge."""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest

from machine import config
from machine.forward import plan as FP
from machine.ideas import store
from machine.referee import verdict as V

FAST = dict(sims=60, resamples=200)                          # the power cells, fewer draws: same test, faster


def _checks(failed=()):
    names = ["luck test", "random entries", "fake levels", "without the best 3", "trades", "not already seen"]
    return [{"number": i + 1, "name": n, "passed": n not in failed and n != "not already seen", "shown": ""}
            for i, n in enumerate(names)]


def _report(word=V.FRESH, failed=(), n=120, seed=1):
    rng = np.random.default_rng(seed)
    cols = ["signal", "status", "net_inr"]
    rows = [[f"s{i}", "closed", round(float(x), 2)] for i, x in enumerate(rng.normal(40, 600, n))]
    rows.append(["open", "open at end", None])                    # never counted
    r = {"answer": {"word": word, "why": "test", "checks": _checks(failed)},
         "trades_table": {"columns": cols, "rows": rows}, "power": {"trades_per_week": 2.0}}
    return SimpleNamespace(r=r, judge={"version_id": "v-1", "settings": {"timeframe": "5m"}},
                           judge_path="plan/work/trials/x-referee.json")


@pytest.fixture
def idea(monkeypatch):
    from machine.report import build as RB
    from machine.referee import tune
    from machine.strategy import parked
    folder = store.folder("idea-009")
    folder.mkdir(parents=True)
    (folder / "form.json").write_text('{"id": "idea-009"}\n', encoding="utf-8")
    box = SimpleNamespace(report=_report(), hedge=None)
    monkeypatch.setattr(RB, "facts", lambda i: box.report)
    monkeypatch.setattr(RB, "_where_judged", lambda f: "the check years 2021-2023 (opened once)")
    monkeypatch.setattr(tune, "latest", lambda i: {"result": {"chosen": "v-1", "versions": {"v-1": {
        "about": "the grab and the gap both in New York", "readings": {"U1": "B"}, "changes": {},
        "signals_sha256": "abc"}}}})
    monkeypatch.setattr(parked, "done", lambda i, v=None: box.hedge)
    return box


def test_only_an_idea_only_new_days_can_judge_that_passed_the_rest_gets_a_plan(idea):
    """The machine's own rule (L10b, README 7): "chance — needs fresh years" AND every other check passed. Its own
    closed trades are the noise; a trade open at the window end is never counted."""
    fx = FP.facts("idea-009")
    assert fx["road"] == FP.PLAIN and fx["version"] == "v-1" and len(fx["trades"]) == 120
    assert fx["trades_a_week"] == 2.0 and fx["signals_sha256"] == "abc" and fx["readings"] == {"U1": "B"}
    idea.report = _report(failed=("random entries",))
    with pytest.raises(FP.ForwardError, match="not a forward-test idea.*failed: random entries"):
        FP.facts("idea-009")
    idea.report = _report(word=V.CHANCE, failed=("luck test",))
    with pytest.raises(FP.ForwardError, match="not a forward-test idea"):
        FP.facts("idea-009")


def test_with_his_hedge_the_plan_counts_in_the_hedged_trades_and_needs_them_kept(idea, tmp_path):
    """The other road: lost with a stop, but with his hedge it made money and beat 95 of 100 random entries
    (parked.choice). A test with his hedge made before 9 Oct kept no rupees per trade: a session runs it again."""
    idea.report = _report(word=V.CHANCE, failed=("luck test",))
    hh = {"net_inr_total": 5000.0, "trades": 4}
    r = {"runs": 200, "his_beats_random": {"with_hedging": 195}, "his_hedging": hh}
    idea.hedge = {"trial_id": "20261009-210000-000000-aaaaaa", "version_id": "v-1", "result": r,
                  "settings": {"window": {"years": "2016-2020", "first_bar": "2016-01-04 01:00:00",
                                          "last_bar": "2020-12-31 23:00:00"}, "timeframe": "5m",
                               "management_hedging": {"table": "T", "rules": {"x": 1}}, "signals_sha256": "def",
                               "readings": {}, "changes": {}}}
    config.TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRIALS_DIR / "20261009-210000-000000-aaaaaa-idea-009-v-1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FP.ForwardError, match="kept no rupees per trade"):
        FP.facts("idea-009")
    hh["trade_net_inr"] = [3000.0, -1000.0, 2500.0, 500.0]
    fx = FP.facts("idea-009")
    assert fx["road"] == FP.HEDGE and fx["trades"] == [3000.0, -1000.0, 2500.0, 500.0]
    assert 0.015 < fx["trades_a_week"] < 0.016                    # 4 trades in 5 years
    assert fx["management"]["rules"] == {"table": "T", "rules": {"x": 1}} and fx["signals_sha256"] == "def"
    r["his_beats_random"]["with_hedging"] = 150                   # made money, but random entries did as well
    with pytest.raises(FP.ForwardError, match="not a forward-test idea"):
        FP.facts("idea-009")


def test_the_trades_needed_follow_the_referees_own_rule():
    """The fewest trades at which a Rs 100 edge is caught at least half the time (the power table's rule), never
    fewer than 30; a wider spread needs more; past the cap it says so."""
    rng = np.random.default_rng(3)
    calm = FP.trades_needed(rng.normal(0, 50, 300), 2.0, **FAST)
    assert calm["trades"] == config.MIN_TRADES and calm["caught"] >= 0.5
    mid = FP.trades_needed(rng.normal(0, 600, 300), 2.0, **FAST)
    assert config.MIN_TRADES < mid["trades"] < 400 and mid["caught"] >= 0.5
    assert all(v < 0.5 for k, v in mid["tried"].items() if int(k) < mid["trades"] - mid["trades"] // 10)
    wild = FP.trades_needed(rng.normal(0, 20000, 300), 2.0, cap=240, **FAST)
    assert wild["trades"] is None and wild["more_than"] == 240
    assert FP.trades_needed(rng.normal(0, 600, 300), 2.0, **FAST)["trades"] > calm["trades"]


def test_it_starts_on_a_monday_after_the_nightly_job_and_ends_on_a_friday():
    fri = FP.dates(dt.date(2026, 10, 9), 105, 2.036)              # Friday 9 Oct + 3 days = Monday 12 Oct
    assert fri == {"start": "2026-10-12", "weeks_at_pace": 52, "weeks_at_most": 78, "end": "2028-04-07"}
    assert dt.date.fromisoformat(fri["end"]).weekday() == 4
    sat = FP.dates(dt.date(2026, 10, 10), 30, 3.0)                # Saturday + 3 = Tuesday: the Monday after
    assert sat["start"] == "2026-10-19" and sat["weeks_at_pace"] == 10 and sat["weeks_at_most"] == 15


def test_the_plan_is_written_once_locked_and_never_rewritten(idea):
    now = dt.datetime(2026, 10, 9, 21, 30, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
    said = []
    p = FP.write("idea-009", now=now, log=said.append, **FAST)
    folder = store.folder("idea-009")
    on_disk = json.loads((folder / FP.PLAN_JSON).read_text(encoding="utf-8"))
    assert on_disk == p and p["kind"] == "forward-plan" and p["road"] == FP.PLAIN
    lk = p["lock"]
    assert lk["version"] == "v-1" and lk["signals_sha256_2016_2020"] == "abc" and lk["rules_sha256"] == V.RULES_SHA256
    assert len(lk["form_sha256"]) == 64 and len(lk["management_sha256"]) == 64
    assert p["trades_needed"] >= config.MIN_TRADES and p["start"] == "2026-10-12"
    md = (folder / FP.PLAN_MD).read_text(encoding="utf-8")
    for words in ("BEFORE any live bar", "Never rewritten", f"**{p['trades_needed']} trades.**",
                  "**Starts Mon 12 Oct 2026 at the earliest**", "git shows this plan on main AND its pretend trades", "Judged ONCE", V.RULES,
                  "Claude's, not Tejas's rules", "no money"):
        assert words in md, words
    before = (folder / FP.PLAN_JSON).read_bytes()
    again = FP.write("idea-009", now=now + dt.timedelta(days=30), log=said.append, **FAST)
    assert again["already"] and (folder / FP.PLAN_JSON).read_bytes() == before        # never rewritten
    (folder / FP.PLAN_MD).unlink()
    FP.write("idea-009", log=said.append)
    assert (folder / FP.PLAN_MD).read_text(encoding="utf-8") == md                    # its words, from the same plan
    assert "written already" in said[-1]


def test_his_line_tells_never_asks_in_his_easy_english(idea):
    from machine.report.build import check_words
    p = FP.write("idea-009", now=dt.datetime(2026, 10, 9, 21, 30).astimezone(), log=lambda *_: None, **FAST)
    text = FP.telegram_text(p, "Idea #9")
    assert text.startswith("Idea #9: its forward test is planned and locked (L17c).")
    assert f"It needs {p['trades_needed']:,} trades on new days" in text and "No money." in text
    assert "?" not in text and text.endswith("You don't need to do anything.") and check_words(text) == []
    assert "₹" not in text                                        # nothing was tested here: no money is shown
