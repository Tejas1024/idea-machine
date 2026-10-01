"""L5 — the referee (machine/referee). It must say "chance" whenever a result could be luck.

The three exams of the prompt (numbers printed — run pytest with -s to see them):
  noise, 100 seeds x 20 versions -> "works" at most 5 in 100 · planted edges Rs 50 and Rs 100 at 300 trades -> caught ·
  the Monday rule on real gold -> "chance" (needs the price data).
Plus: CHECK years open once per family; an already-seen idea can never reach "works"; the rules are pinned.
Hand-made bars are random walks, not gold. Every test writes into temporary folders (tests/conftest.py).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from machine import config, trials as trial_files
from machine.guards import CheckYearsClosedError
from machine.referee import (fake_levels, judge, power, prior, project, random_entries, selftest, splits, stats,
                             trials, verdict)
from machine.referee.replays import closed_trades, make_segment, replay
from machine.tester import Signal, run

# the rules' fingerprint: a change to any rule or number in verdict.RULES must change this line too (L5: "fixed in
# advance") — so it shows in git next to the change.
# L10, 29 Sep 2026, before idea #1 (the first real idea) was judged: WORDS only, no number or rule changed — "(p < 0.05)"
# taken out and "walk-forward" said in words, because the report shows this text to Tejas (#113 easy English).
# Was e6a71a0edd83711c6d287892dff6591d9e648dac95affb99cb8e390e7529ec31 (the Monday control run's trial file has that one).
RULES_SHA256 = "5b7ca6d6fb764a3ab2e2232905a98e12f173779b3a77f8ba41fd2c5f375a54c2"


# ================================================================================================================
# helpers
# ================================================================================================================

def show(text: str) -> None:
    """Print the exam numbers even on a Windows console that cannot show ₹ (pytest -s)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.replace("₹", "Rs ").encode(enc, errors="replace").decode(enc))


def walk(start: str, days: int = 3, seed: int = 7) -> pd.DataFrame:
    """A random walk of 1-min bars from start (whole days), with some 0 spreads."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(start, periods=n, freq="1min", name="server_time")
    steps = rng.normal(0, 0.3, n)
    c = 1800 + steps.cumsum()
    o = np.r_[1800, c[:-1]]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.15, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.15, n))
    return pd.DataFrame({"open": o.round(2), "high": h.round(2), "low": lo.round(2), "close": c.round(2),
                         "spread": rng.choice([0, 12, 18, 25], size=n), "vol": 1}, index=idx)


def signals_for(m1: pd.DataFrame, idea="idea-test", version="v1", parent=None, every=30) -> list[Signal]:
    """A signal every `every` minutes: half next-open (from-fill $1 stop, $1.5 target), half limits $0.5 away."""
    out = []
    for i, t in enumerate(m1.index[every - 1::every][:-2]):
        close = float(m1.close.loc[t])
        known = t + pd.Timedelta(minutes=1)
        base = dict(id=f"s{i}", known_at=known, idea_id=idea, version_id=version, parent_idea=parent)
        if i % 2 == 0:
            out.append(Signal(**base, direction="buy", levels="from_fill", stop=1.0, targets=(1.5,)))
        else:
            e = round(close + 0.5, 2)
            out.append(Signal(**base, direction="sell", entry_type="limit", entry_price=e, stop=round(e + 1, 2),
                              targets=(round(e - 1.5, 2),), expiry_bars=3))
    return out


def tried(idea, version, parent=None, start="2019-03-11"):
    """One tester run on the tune years: the version's trial file (a version must be tried before its check opens)."""
    m1 = walk(start, days=1)
    return run(signals_for(m1, idea, version, parent), tf="5m", m1=m1)


# ================================================================================================================
# 1. trials — versions per idea, family and project
# ================================================================================================================

def test_versions_are_counted_per_idea_family_and_project():
    tried("idea-a", "v1")
    tried("idea-a", "v2")
    tried("idea-a", "v2")                          # the same version again: still one version
    tried("idea-b", "v1", parent="idea-a")        # a version OF idea-a (section D)
    tried("idea-c", "v1")
    c = trials.count("idea-b")
    assert c.family == "idea-a"
    assert c.idea_versions == ["v1"]
    assert c.family_ideas == ["idea-a", "idea-b"]
    assert c.family_versions == [("idea-a", "v1"), ("idea-a", "v2"), ("idea-b", "v1")]
    assert (c.project_ideas, c.project_families, c.project_versions, c.trial_files) == (3, 2, 4, 5)
    assert "family idea-a: 3 versions of 2 ideas" in c.text()
    assert trials.count("idea-a", extra=("idea-a", "v3")).idea_versions == ["v1", "v2", "v3"]


def test_the_version_budget(monkeypatch):
    monkeypatch.setattr(config, "VERSION_BUDGET", 2)
    tried("idea-a", "v1")
    tried("idea-a", "v2")
    trials.check_budget("idea-a", "v2")           # an existing version: fine
    with pytest.raises(ValueError, match="budget"):
        trials.check_budget("idea-a", "v3")
    with pytest.raises(ValueError, match="budget"):
        trials.check_budget("idea-b", "v1", parent="idea-a")


def test_parent_chains_that_loop_or_disagree_are_refused():
    with pytest.raises(ValueError, match="loops"):
        trials.root_of("a", {"a": "b", "b": "a"})
    files = [{"idea_id": "a", "parent_idea": "b"}, {"idea_id": "a", "parent_idea": "c"}]
    with pytest.raises(ValueError, match="two parent"):
        trials.parents(files)


# ================================================================================================================
# 2. splits — the check years open ONCE per idea family
# ================================================================================================================

def test_check_years_open_once_per_family():
    tried("idea-a", "v1")
    tried("idea-a", "v2")
    tried("idea-b", "v1", parent="idea-a")
    assert splits.status("idea-a", "v1") == "closed"
    assert splits.opened_count() == 0
    rec = splits.open_check("idea-a", "v1", who="test", why="v1 chosen on 2016-2020")
    assert rec["family"] == "idea-a" and rec["version_id"] == "v1" and rec["family_versions_tried"] == 3
    assert json.loads((config.IDEAS_DIR / "idea-a" / "check-opened.json").read_text(encoding="utf-8"))["why"]
    assert splits.opened_count() == 1
    assert splits.status("idea-a", "v1") == "check"
    assert splits.status("idea-a", "v2") == "walk-forward"
    assert splits.status("idea-b", "v1", parent="idea-a") == "walk-forward"
    for idea, version in (("idea-a", "v1"), ("idea-a", "v2"), ("idea-b", "v1")):
        with pytest.raises(CheckYearsClosedError, match="ONCE per idea family"):
            splits.open_check(idea, version, who="test", why="again")
    assert splits.opened_count() == 1


def test_check_years_need_a_tried_version_and_who_and_why():
    with pytest.raises(ValueError, match="no trial file"):
        splits.open_check("idea-z", "v1", who="test", why="never tried")
    tried("idea-z", "v1")
    with pytest.raises(ValueError, match="who and why"):
        splits.open_check("idea-z", "v1", who="", why="x")
    assert splits.opened_count() == 0


def test_the_tester_lets_only_the_opened_version_into_the_check_years():
    tried("idea-a", "v1")
    tried("idea-a", "v2")
    m1 = walk("2021-03-08", days=1)
    with pytest.raises(CheckYearsClosedError):
        run(signals_for(m1, "idea-a", "v1"), tf="5m", m1=m1, record=False)        # not opened yet
    splits.open_check("idea-a", "v1", who="test", why="chosen")
    run(signals_for(m1, "idea-a", "v1"), tf="5m", m1=m1, record=False)            # now v1 may
    with pytest.raises(CheckYearsClosedError, match="later version"):
        run(signals_for(m1, "idea-a", "v2"), tf="5m", m1=m1, record=False)        # v2 never
    with pytest.raises(CheckYearsClosedError):
        run(signals_for(m1, "idea-b", "v1", "idea-a"), tf="5m", m1=m1, record=False)


def test_walk_forward_folds_choose_only_on_the_years_before():
    folds = splits.walk_forward_folds()
    assert [(f.choose_years, f.judge_year) for f in folds] == [
        ((2016, 2017), 2018), ((2016, 2017, 2018), 2019), ((2016, 2017, 2018, 2019), 2020)]
    assert all(max(f.choose_years) < f.judge_year <= 2020 for f in folds)


# ================================================================================================================
# 3. prior — "already seen"
# ================================================================================================================

TYPE1 = "liquidity_sweep+break_of_structure+fvg"


def test_the_same_blocks_timeframe_and_entry_kind_are_already_seen():
    s = prior.already_seen(prior.IdeaKey.make(TYPE1, "5m", "gap:near"), config.CHECK_YEARS)
    assert s.seen and "type1_grab_gap" in [r.idea_key for r in s.rows]
    far = prior.already_seen(prior.IdeaKey.make(TYPE1, "5m", "gap:far"), config.CHECK_YEARS)
    assert far.seen and "type1_grab_gap" in [r.idea_key for r in far.rows]     # near and far edge: one kind
    mg = prior.already_seen(prior.IdeaKey.make(TYPE1 + "+mgmt:hedge_close", "5m", "gap:near"), config.CHECK_YEARS)
    assert mg.seen                                                               # a management rule: still seen


def test_his_entries_rows_match_the_same_management_on_his_entry():
    s = prior.already_seen(prior.IdeaKey.make(TYPE1, "5m", "gap:near", mgmt=["no_close_zone"]), config.CHECK_YEARS)
    assert "no_close_zone" in [r.idea_key for r in s.rows]
    plain = prior.already_seen(prior.IdeaKey.make(TYPE1, "5m", "gap:near"), config.CHECK_YEARS)
    assert "no_close_zone" not in [r.idea_key for r in plain.rows]


def test_other_timeframe_other_blocks_or_other_years_are_not_seen():
    zone = "strong_zones+zone_entry+obvious_liquidity"
    assert prior.already_seen(prior.IdeaKey.make(zone, "5m", "zone:edge"), config.CHECK_YEARS).seen
    assert not prior.already_seen(prior.IdeaKey.make(zone, "4h", "zone:edge"), config.CHECK_YEARS).seen
    news = prior.IdeaKey.make("news_event+sessions", "1h", "market:break")
    assert prior.already_seen(news, config.CHECK_YEARS).seen
    assert not prior.already_seen(news, (2016, 2017)).seen                      # that row starts in 2018


def test_idea_1_equal_highs_lows_is_fresh():
    """plan/work/prior-tries.md: 'Idea #1 (equal highs and lows are a magnet) stays fresh.'"""
    s = prior.already_seen(prior.IdeaKey.make("equal_highs_lows", "5m", "line:touch"), config.CHECK_YEARS)
    assert not s.seen and "no" in s.text()


# ================================================================================================================
# 6. stats
# ================================================================================================================

def trades_df(values, start="2021-01-04", every_days=3):
    t = pd.date_range(start, periods=len(values), freq=f"{every_days}D") + pd.Timedelta(hours=12)
    return pd.DataFrame({"net_inr": np.asarray(values, dtype=float), "exit_time": t})


def test_per_year_shows_every_year_and_the_worst():
    tr = trades_df([100, -50, 30], start="2021-06-01")
    y = stats.per_year(tr, (2021, 2022, 2023))
    assert y[2021] == {"trades": 3, "net_inr": 80.0, "per_trade": 26.67}
    assert y[2022]["trades"] == 0 and y[2023]["net_inr"] == 0.0
    assert stats.worst_year(y)["year"] == 2022
    with pytest.raises(ValueError, match="outside"):
        stats.per_year(tr, (2022,))


def test_without_the_best_three():
    wb = stats.without_best(trades_df([1000, 900, 800, -10, -20]))
    assert wb["per_trade"] == -15.0 and wb["best_inr"] == [1000.0, 900.0, 800.0]


def test_the_luck_test_sees_a_big_edge_and_not_a_loss():
    rng = np.random.default_rng(3)
    good = trades_df(rng.normal(150, 400, 300))
    bad = trades_df(rng.normal(-40, 400, 300))
    w = stats.week_ids
    assert stats.bootstrap_p(good.net_inr, w(good.exit_time)) < 0.01
    assert stats.bootstrap_p(bad.net_inr, w(bad.exit_time)) > 0.5
    assert stats.bootstrap_p(good.net_inr, w(good.exit_time), seed=1) == stats.bootstrap_p(
        good.net_inr, w(good.exit_time), seed=1)


def test_the_luck_test_is_wrong_about_5_times_in_100():
    """With no edge at all, "p < 0.05" must come out about 5 times in 100 (its size), not more."""
    noise = np.random.default_rng(5).normal(0, 400, 20_000)
    size = power.caught_share(noise, 0, 300, 2.0, sims=600, seed=9)
    show(f"\nluck test with no edge (300 trades, 600 samples): p < 0.05 in {100 * size:.1f} of 100")
    assert size <= 0.075


def test_deflated_sharpe_falls_with_more_tries():
    x = np.random.default_rng(1).normal(40, 400, 300)
    one = stats.deflated_sharpe(x, 1, 0.0)
    many = stats.deflated_sharpe(x, 20, 0.004)
    assert 0 < many < one < 1
    assert stats.deflated_sharpe(np.random.default_rng(2).normal(300, 400, 300), 20, 0.004) > 0.99


def test_pbo_is_about_half_for_noise_and_low_for_a_real_edge():
    rng = np.random.default_rng(4)
    # one noise table's PBO swings a lot (whichever version is best overall by luck stays best): judge the middle one
    middle = np.median([stats.pbo_cscv(rng.normal(0, 100, (260, 12)))["pbo"] for _ in range(15)])
    assert 0.35 <= middle <= 0.65
    noise = rng.normal(0, 100, (260, 12))
    real = noise.copy()
    real[:, 0] += 60
    assert stats.pbo_cscv(real)["pbo"] < 0.1


def test_choosing_between_versions_is_only_on_the_tune_years():
    rng = np.random.default_rng(6)
    tune = {f"v{i}": trades_df(rng.normal(0, 400, 200), start="2016-01-04", every_days=9) for i in range(4)}
    tune["v4"] = trades_df(rng.normal(200, 400, 200), start="2016-01-04", every_days=9)
    pick = stats.choose_version(tune)
    assert pick["chosen"] == "v4" and pick["pbo"]["pbo"] < 0.5
    with pytest.raises(ValueError, match="ONLY on the tune years"):
        stats.choose_version({"v1": trades_df([1, 2, 3], start="2021-01-04")})
    with pytest.raises(ValueError, match="tune years"):
        stats.choose_version(tune, years=config.CHECK_YEARS)


# ================================================================================================================
# 7. power
# ================================================================================================================

def test_the_power_table_grows_with_the_edge_and_the_trades():
    noise = np.random.default_rng(8).normal(0, 400, 10_000)
    t = power.table(noise, own_n=60, trades_per_week=2.0, sims=120, resamples=300)
    rows = {int(n): {int(e): s for e, s in r.items()} for n, r in t["rows"].items()}
    assert set(rows) == {30, 60, 100, 300}
    assert rows[300][100] > 0.9 and rows[300][100] >= rows[100][100] >= rows[30][100]
    assert rows[300][100] >= rows[300][50] >= rows[300][10]
    assert t["needed_edge_caught"] == rows[60][100]
    assert power.caught_share(noise, 100, 29, 2.0) == 0.0                 # under 30 trades: never "works"


# ================================================================================================================
# 8. the answer — fixed in advance
# ================================================================================================================

def evidence(**kw) -> verdict.Evidence:
    base = dict(trades=300, per_trade=120.0, p=0.001,
                random={"runs": 200, "runs_with_trades": 200, "beaten": 199, "beaten_share": 0.995,
                        "per_trade_median": -30.0, "trades_median": 300},
                fake={"applies": True, "beaten": 8, "compared": 8, "beaten_share": 1.0},
                without_best={"left_out": 3, "per_trade": 100.0}, seen=False, needed_edge_caught=0.99)
    base.update(kw)
    return verdict.Evidence(**base)


def test_the_rules_are_pinned():
    show("\n" + verdict.RULES)
    assert verdict.RULES_SHA256 == RULES_SHA256, (
        f"verdict.RULES changed: its fingerprint is now {verdict.RULES_SHA256}. The rules are fixed in advance (L5) "
        "— if this change is meant, update RULES_SHA256 in this test in the same commit, with the reason.")
    for words in ("less than 5 times in 100", "95 of 100", "200 runs", "best 3", "At least 30 trades", "2021-2023", "₹100"):
        assert words in verdict.RULES, words


def test_works_needs_all_six():
    assert verdict.decide(evidence()).word == verdict.WORKS
    for kw in ({"p": 0.06}, {"per_trade": -1.0},
               {"random": {**evidence().random, "beaten_share": 0.94}},
               {"fake": {"applies": True, "beaten": 5, "compared": 8, "beaten_share": 0.625}},
               {"without_best": {"left_out": 3, "per_trade": -1.0}},
               {"trades": 29}):
        assert verdict.decide(evidence(**kw)).word != verdict.WORKS, kw


def test_already_seen_can_never_reach_works():
    """Even a huge edge on years the old project already saw is only "chance — needs fresh years"."""
    a = verdict.decide(evidence(per_trade=5000.0, p=0.0005, seen=True))
    assert a.word == verdict.FRESH and not a.checks[5].passed


def test_not_enough_trades_to_tell():
    a = verdict.decide(evidence(trades=12, p=0.3, needed_edge_caught=0.2))
    assert a.word == verdict.FEW
    assert verdict.decide(evidence(p=0.3, needed_edge_caught=0.8)).word == verdict.CHANCE


def test_no_level_to_move_does_not_block():
    ev = evidence(fake={"applies": False, "why": "no level to move"})
    a = verdict.decide(ev)
    assert a.word == verdict.WORKS and "does not apply" in a.checks[2].shown


# ================================================================================================================
# 4, 5. random entries and fake levels (hand-made bars)
# ================================================================================================================

def test_random_entries_keep_distances_direction_and_change_time_and_level():
    m1 = walk("2019-03-11")
    sigs = signals_for(m1)
    seg = make_segment("t", (2019,), "5m", sigs, m1, strict=True)
    lim = next(s for s in sigs if s.entry_type == "limit")
    t = random_entries.template(seg, lim)
    ref = seg.price_before(lim.known_at)
    assert t.offset == pytest.approx(lim.entry_price - ref)          # a sell limit above the price
    assert (t.stop_d, t.target_ds) == (pytest.approx(1.0), (pytest.approx(1.5),))
    when = seg.known_times()[500]
    r = random_entries.make_signal(seg, t, when, 0.8, "r1")
    assert r.direction == "sell" and r.known_at == when and r.expiry_bars == lim.expiry_bars
    assert r.entry_price == pytest.approx(round(seg.price_before(when) + 0.8, 2))
    assert r.stop - r.entry_price == pytest.approx(1.0) and r.entry_price - r.targets[0] == pytest.approx(1.5)
    nxt = random_entries.template(seg, sigs[0])
    assert nxt.offset is None and nxt.stop_d == 1.0


def test_a_random_entry_level_uses_no_later_price():
    """Cut off the future (R4): a random limit's price comes from the last 1-min close at or before its moment —
    the same with the later bars cut away."""
    m1 = walk("2019-03-11")
    sigs = signals_for(m1)
    full = make_segment("t", (2019,), "5m", sigs, m1, strict=True)
    when = full.known_times()[300]
    cut = make_segment("t", (2019,), "5m", [], m1.loc[:when], strict=True)
    t = random_entries.template(full, next(s for s in sigs if s.entry_type == "limit"))
    a = random_entries.make_signal(full, t, when, 0.5, "a")
    b = random_entries.make_signal(cut, t, when, 0.5, "b")
    assert a.entry_price == b.entry_price and a.stop == b.stop and a.targets == b.targets
    assert full.price_before(when) == float(m1.close[m1.index < when].iloc[-1])


def test_random_runs_match_the_trades_and_repeat_with_the_seed():
    m1 = walk("2019-03-11")
    seg = make_segment("t", (2019,), "5m", signals_for(m1), m1, strict=True)
    idea = closed_trades(replay(seg, seg.signals))
    rs, settings = random_entries.runs([seg], [idea], n_runs=6, seed=11)
    assert settings["seed"] == 11 and len(rs) == 6
    for r in rs:
        assert r.target_trades == len(idea)
        assert 0.8 * len(idea) <= r.trades <= 1.2 * len(idea)
    again, _ = random_entries.runs([seg], [idea], n_runs=6, seed=11)
    assert [r.net_inr for r in rs] == [r.net_inr for r in again]
    s = random_entries.summary(rs, idea.net_inr.mean())
    assert s["runs"] == 6 and s["trades_wanted"] == len(idea)


def test_fake_levels_move_the_level_and_keep_the_distances():
    m1 = walk("2019-03-11")
    sigs = signals_for(m1)
    lim = next(s for s in sigs if s.entry_type == "limit")
    up = fake_levels.moved(lim, 1.0)                                  # +1 stop distance ($1)
    assert up.entry_price == pytest.approx(lim.entry_price + 1.0)
    assert up.stop == pytest.approx(lim.stop + 1.0) and up.targets[0] == pytest.approx(lim.targets[0] + 1.0)
    assert fake_levels.moved(sigs[0], 1.0) is sigs[0]                 # next open: no level
    seg = make_segment("t", (2019,), "5m", sigs, m1, strict=True)
    fr = fake_levels.runs([seg])
    assert [f.step for f in fr] == list(config.FAKE_LEVEL_STEPS)
    only_next = make_segment("t", (2019,), "5m", [s for s in sigs if s.entry_type != "limit"], m1, strict=True)
    assert fake_levels.runs([only_next]) is None
    assert fake_levels.summary(None, 1.0)["applies"] is False


# ================================================================================================================
# the whole path on hand-made bars
# ================================================================================================================

def test_judge_on_the_check_years_writes_one_trial_file(_trials_in_tmp):
    tried("idea-a", "v1")
    tried("idea-a", "v2")
    m1 = walk("2021-03-08", days=4)
    key = prior.IdeaKey.make("equal_highs_lows", "5m", "line:touch")
    with pytest.raises(CheckYearsClosedError, match="still closed"):
        judge.judge_check(signals_for(m1, "idea-a", "v1"), tf="5m", key=key, m1=m1, n_random=5)
    splits.open_check("idea-a", "v1", who="test", why="chosen on the tune years")
    before = len(list(_trials_in_tmp.glob("*.json")))
    rep = judge.judge_check(signals_for(m1, "idea-a", "v1"), tf="5m", key=key, m1=m1, n_random=20)
    files = sorted(_trials_in_tmp.glob("*.json"))
    assert len(files) == before + 1
    body = json.loads(files[-1].read_text(encoding="utf-8"))
    assert body["kind"] == "referee" and body["settings"]["rules_sha256"] == verdict.RULES_SHA256
    assert body["result"]["answer"]["word"] in verdict.WORDS and body["result"]["check_opened"] == 1
    text = rep.text()
    for words in ("ANSWER:", "the check years 2021-2023", "CHECK years opened 1 times in the project",
                  "Worst year:", "The 6 checks:", "random entries", "fake levels", "THE RULES", "₹40 per position"):
        assert words in text, words
    assert rep.word != verdict.WORKS                                   # a random walk has no edge
    with pytest.raises(CheckYearsClosedError, match="walk-forward"):
        judge.judge_check(signals_for(m1, "idea-a", "v2"), tf="5m", key=key, m1=m1, n_random=5)
    s = project.summary()
    assert s["answers"] == 1 and s["check_opened"] == 1 and s["false_works_expected"] == 0.05


def test_a_control_run_is_recorded_but_is_not_an_answer(_trials_in_tmp):
    m1 = pd.concat([walk(f"{y}-03-{d:02d}", days=1, seed=y)                  # one Monday in each tune year
                    for y, d in ((2016, 7), (2017, 6), (2018, 5), (2019, 11), (2020, 9))])
    key = prior.IdeaKey.make("weekday_hour", "5m", "market:open")
    rep = judge.judge_walk_forward({"v1": signals_for(m1, "control-x")}, tf="5m", key=key, m1=m1, n_random=3,
                                   control=True)
    kinds = [json.loads(p.read_text(encoding="utf-8"))["kind"] for p in _trials_in_tmp.glob("*.json")]
    assert kinds == [judge.CONTROL_KIND]
    assert project.summary()["answers"] == 0 and "CONTROL" in rep.text()


def test_walk_forward_judges_one_family_only():
    m1 = walk("2019-03-11", days=1)
    key = prior.IdeaKey.make("equal_highs_lows", "5m", "line:touch")
    with pytest.raises(ValueError, match="ONE idea family"):
        judge.judge_walk_forward({"a": signals_for(m1, "idea-a"), "b": signals_for(m1, "idea-b")}, tf="5m", key=key,
                                 m1=m1, n_random=2)


def test_judge_refuses_bars_outside_its_years():
    tried("idea-a", "v1")
    splits.open_check("idea-a", "v1", who="test", why="chosen")
    m1 = walk("2020-12-31", days=2)                                    # reaches back into 2020
    with pytest.raises(ValueError, match="outside the judged years"):
        judge.judge_check(signals_for(m1.loc["2021"], "idea-a", "v1"), tf="5m",
                          key=prior.IdeaKey.make("x", "5m", "market:open"), m1=m1, n_random=2)


# ================================================================================================================
# THE THREE EXAMS
# ================================================================================================================

def test_exam_noise_never_works_more_than_5_in_100():
    """100 noise ideas x 20 versions: chosen on the tune years, judged once on fresh check years; and every one of
    the 2,000 versions judged on its own.

    NOISE = an idea with no edge: gold gives it nothing, so it loses the Rs 40 cost. The prompt's bar is for this one:
    "works" at most 5 in 100.
    THE HARDEST NOISE (Claude's extra case) = exactly Rs 0 a trade after cost, the very edge of "no edge". There a
    luck test built for 5 in 100 says "works" about 5 in 100 BY DESIGN. Measured 28 Sep 2026, before this bound was
    written: 99 of 2,000 versions (4.95%); 6 of 100 chosen versions. The bound below (6 in 100 over 2,000) is a guard
    set after that measurement: if a change pushes it higher, the luck test got looser. The rules were not changed to
    meet it (they are the plan's numbers).
    """
    real = selftest.noise_exam(noise_mean=-float(config.COST_INR_PER_POSITION), every_version=True)
    hard = selftest.noise_exam(noise_mean=0.0, every_version=True)
    show(f"\nNOISE (no edge, loses the ₹40 cost): \"works\" {real['chosen_works']} of 100 ideas; every version on "
         f"its own: {real['all_works']} of {real['all_judged']}")
    show(f"NOISE (hardest: exactly ₹0 a trade after cost): \"works\" {hard['chosen_works']} of 100 ideas; every "
         f"version on its own: {hard['all_works']} of {hard['all_judged']} ({100 * hard['all_rate']:.2f}%); "
         f"PBO median {hard['pbo_median']:.2f}")
    show(f"If all 20 versions were judged on the same years and the best kept: \"works\" for {hard['any_of_all']} "
         "of 100 hardest-noise ideas — why the check years open only once")
    assert real["chosen_works"] <= 5 and real["all_rate"] <= 0.05
    assert hard["all_rate"] <= 0.06
    assert hard["any_of_all"] > 5 * hard["chosen_works"] / 2          # judging all 20 would be far worse
    assert 0.35 <= hard["pbo_median"] <= 0.65                          # choosing among noise picks luck


def test_exam_planted_edges_are_caught():
    r50, r100 = selftest.planted_exam(50), selftest.planted_exam(100)
    show(f"\nPLANTED at 300 trades (about ₹400 a trade either way): ₹50 caught {r50['caught']} of {r50['sims']} "
          f"({100 * r50['rate']:.0f}%) · ₹100 caught {r100['caught']} of {r100['sims']} ({100 * r100['rate']:.0f}%)")
    assert r100["rate"] >= 0.9
    assert r50["rate"] >= 0.5


@pytest.mark.data
def test_exam_the_monday_rule_is_chance():
    """The L3 smoke rule (buy every Monday 10:00, stop $5, target $5): no edge. Real gold, walk-forward 2016-2020."""
    rep = selftest.monday_exam(record=False)
    show("\n" + "\n".join(rep.text().splitlines()[:16]))
    assert rep.word == verdict.CHANCE
    assert rep.numbers["summary"]["trades"] > 100
