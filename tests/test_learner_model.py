"""L13.2 - the learner's model (LightGBM + SHAP, a purged walk-forward on the tune years), its take/skip filter as a
new version, and the pooled view (every tested idea's trades + random entries: any entry, or only this idea).

What the prompt asks: a planted London-only edge is found; a noise condition is not. Around it: the model never sees
the money of the year it guesses (nor 2021-2023); SHAP is the plain Shapley sum; the filter is a new version - counted,
judged by walk-forward inside 2016-2020, and never shown AS the idea's judgement; the report has the lines in easy
English. Made-up trades have no edge unless one is planted (machine/learner/selftest.py); every file goes to temporary
folders (tests/conftest.py).
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from machine import config
from machine import trials as trial_files
from machine.ideas import store
from machine.learner import model as MD
from machine.learner import pooled as PL
from machine.learner import run as LR
from machine.learner import selftest
from machine.learner import takeskip as TS
from machine.learner.selftest import fake_pool, fake_setups, fake_trades
from machine.referee import project, splits, trials
from machine.referee.replays import closed_trades, make_segment, replay
from machine.report import build as B
from machine.tester.models import Signal
from tests.test_learner import with_learner
from tests.test_referee import signals_for, walk
from tests.test_report import IDEA, make_form, numbers_in


def planted(seed: int, gap: float = 0.0, n: int = 600):
    tr = fake_trades(n, seed=seed)
    f = fake_setups(tr, seed)
    if gap:
        tr.loc[f["session"] == "London", "net_inr"] += gap
    return tr, f


# ================================================================================================================
# 1. the model: a planted London edge is found, noise is not
# ================================================================================================================

@pytest.fixture(scope="module")
def london():
    tr, f = planted(21, 300.0)
    return tr, f, MD.analyse(f, tr, shuffles=60)


def test_a_planted_london_edge_is_found_and_nothing_else_is_named(london):
    """London trades earn Rs 300 more: the model's picks of 2018-2020 beat the rest past luck, the session is the
    thing it leans on most, London pushes its guesses up - and no noise column is named."""
    tr, f, r = london
    c = r["check"]
    assert c["passed"] and c["p"] < 0.05 and c["gap"] > 100
    assert c["p"] == pytest.approx(1 / 61, abs=1e-5)               # no shuffle of 60 came near it
    assert r["named"] == ["session"] and r["shap"][0]["feature"] == "session"
    parts = {p["part"]: p for p in r["shap"][0]["parts"]}
    assert parts["London"]["mean_push"] > 50 > max(parts[x]["mean_push"] for x in ("Asia", "New York", "in between"))
    assert selftest.model_found(r)
    assert sum(y["top_trades"] + y["rest_trades"] for y in c["by_year"]) == r["guessed_trades"]
    assert [x["year"] for x in r["folds"]] == [2018, 2019, 2020]
    line = MD.check_line(r)
    assert "it found something" in line and "trades it liked best made ₹" in line
    assert "It leans most on (SHAP, how much each thing pushed its guess): the session (India time) ('London'" in \
        MD.named_line(r)


@pytest.mark.parametrize("seed", [31, 32])
def test_pure_noise_passes_no_check_and_names_nothing(seed):
    """No edge anywhere (48 random things): the check does not pass and nothing is named. (The rate over many made-up
    ideas is python -m machine.learner.selftest --model: plan/work/L132-notes.md.)"""
    tr, f = planted(seed)
    r = MD.analyse(f, tr, shuffles=40)
    assert not r["check"]["passed"] and r["check"]["p"] > 0.05
    assert r["named"] == [] and "found nothing it could use" in MD.check_line(r)
    assert MD.named_line(r).startswith("Nothing is named")


def test_a_noise_condition_is_not_named_next_to_a_real_edge(london):
    """A column made to look useful - random yes / no - is read by the model but never named; its push stays near
    its push on shuffled money."""
    tr, f, r = london
    noise = next(d for d in r["shap"] if d["feature"] == "zone_at_entry")
    assert not noise["named"] and noise["p"] > 0.05 and noise["mean_push"] < 3 * noise["shuffled_push"]


def test_the_model_never_sees_the_money_of_the_year_it_guesses(london):
    """Cut off the future, for the model: change the money of every trade whose setup is in one year - the guesses of
    that year and every year before it do not move; only the later years' guesses move (their models learned it)."""
    tr, f, _ = london
    enc = MD.Encoder.fit(f)
    X = enc.transform(f)
    base = MD.walk_forward(X, f["known_at"], tr, seed=5, text_columns=enc.text_columns, shap=False).guesses
    year = pd.DatetimeIndex(f["known_at"]).year
    t_year = pd.DatetimeIndex(f.loc[tr.index, "known_at"]).year
    for changed in (2017, 2018, 2019, 2020):
        y = tr["net_inr"].copy()
        y[t_year == changed] = y[t_year == changed] * -3 + 777.0
        g = MD.walk_forward(X, f["known_at"], tr, seed=5, text_columns=enc.text_columns, y=y, shap=False).guesses
        for yy in (2018, 2019, 2020):
            m = year == yy
            if yy <= changed:
                assert np.allclose(g[m], base[m]), (changed, yy)
            elif yy == changed + 1:
                assert not np.allclose(g[m], base[m]), (changed, yy)      # the next year's model learned it


def test_the_walk_forward_is_purged(monkeypatch):
    """A trade teaches the model of year Y only if it closed at least 7 days before Y began; the model guesses only
    the setups of Y. The trades still open at the new year, or closing in its last week, are counted as left out."""
    tr, f = planted(41, n=400)
    tr.iloc[0, tr.columns.get_loc("exit_time")] = pd.Timestamp("2017-12-29 10:00")      # the last week of 2017
    f.iloc[0, f.columns.get_loc("known_at")] = pd.Timestamp("2017-12-28 10:00")
    tr.iloc[1, tr.columns.get_loc("exit_time")] = pd.Timestamp("2018-01-03 10:00")      # open at the new year
    f.iloc[1, f.columns.get_loc("known_at")] = pd.Timestamp("2017-12-30 10:00")
    seen = []
    real = MD.fit

    def spy(X, y, **k):
        seen.append(list(X.index))
        return real(X, y, **k)

    monkeypatch.setattr(MD, "fit", spy)
    enc = MD.Encoder.fit(f)
    wf = MD.walk_forward(enc.transform(f), f["known_at"], tr, seed=1, text_columns=enc.text_columns, shap=False)
    exits = pd.DatetimeIndex(tr["exit_time"])
    for fold, learned in zip(wf.folds, seen):
        start = pd.Timestamp(f"{fold['year']}-01-01")
        cut = start - pd.Timedelta(days=config.MODEL_PURGE_DAYS)
        assert pd.DatetimeIndex(tr.loc[learned, "exit_time"]).max() < cut
        assert fold["learned_from"] == len(learned) == int((exits < cut).sum())
        known = pd.DatetimeIndex(f.loc[tr.index, "known_at"])
        assert fold["purged"] == int(((known < start) & (exits >= cut)).sum())
    first = wf.folds[0]
    assert first["year"] == 2018 and first["purged"] >= 2
    assert tr.index[0] not in seen[0] and tr.index[1] not in seen[0] and tr.index[0] in seen[1]
    guessed = wf.guesses.dropna()
    assert set(pd.DatetimeIndex(f.loc[guessed.index, "known_at"]).year) == {2018, 2019, 2020}


def test_the_model_reads_the_tune_years_only():
    tr, f = planted(5, n=200)
    late = tr.copy()
    late.iloc[3, late.columns.get_loc("exit_time")] = pd.Timestamp("2021-01-05")
    with pytest.raises(MD.ModelError, match="tune years 2016-2020 only"):
        MD.analyse(f, late, shuffles=2)
    f2 = f.copy()
    f2.iloc[0, f2.columns.get_loc("known_at")] = pd.Timestamp("2022-03-01")
    with pytest.raises(MD.ModelError, match="tune years"):
        MD.analyse(f2, tr, shuffles=2)
    with pytest.raises(MD.ModelError, match="have no setup"):
        MD.analyse(f.iloc[3:], tr, shuffles=2)


# ================================================================================================================
# 2. the pieces: the table, the check, the shuffles, SHAP
# ================================================================================================================

def test_the_table_the_model_reads():
    f = pd.DataFrame({"known_at": pd.to_datetime(["2019-01-02"] * 4), "session": ["London", "Asia", None, "London"],
                      "at_ath": [True, False, True, None], "adr": [10.0, 12.5, np.nan, 11.0],
                      "level_kind": ["equal"] * 4, "reward": [None] * 4}, index=list("abcd"))
    enc = MD.Encoder.fit(f)
    assert enc.columns == ["session", "adr", "at_ath"] or set(enc.columns) == {"session", "adr", "at_ath"}
    assert enc.kinds["session"] == "text" and enc.categories["session"] == ["Asia", "London"]
    assert enc.kinds["at_ath"] == "number" and enc.kinds["adr"] == "number"
    assert enc.left_out["level_kind"] == "always 'equal' on these setups"
    assert enc.left_out["reward"] == "never known on these setups" and enc.left_out["direction"] == "not in the table"
    X = enc.transform(f)
    assert list(X["session"].cat.categories) == ["Asia", "London"] and pd.isna(X["session"].iloc[2])
    assert X["at_ath"].tolist()[:3] == [1.0, 0.0, 1.0] and math.isnan(X["at_ath"].iloc[3])
    assert set(MD.LABELS) == set(MD.F.COLUMNS)                    # every column has its short name for a report


def test_the_check_counts_the_trades_it_liked_best_in_each_year():
    guess = [5, 1, 3, 3, 9, 9, 9, 9]
    value = [100, -50, 20, 10, 7, 7, 7, 7]
    year = [2018] * 4 + [2019] * 4
    c = MD.check_gap(guess, value, year)
    assert (c["top_trades"], c["rest_trades"]) == (1, 3)                      # 2018: 5 is above the middle (3)
    assert c["gap"] == pytest.approx(100 - (-50 + 20 + 10) / 3, abs=0.01)
    assert c["by_year"][1]["gap"] is None                                      # 2019: every guess the same
    assert MD.check_gap([1, 1], [5, 6], [2018, 2018])["gap"] is None


def test_the_shuffles_keep_each_years_money():
    v = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=list("abcdef"))
    g = [2018, 2018, 2018, 2019, 2019, 2019]
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = MD.shuffled(v, g, rng)
        assert sorted(s[:3]) == [1.0, 2.0, 3.0] and sorted(s[3:]) == [10.0, 20.0, 30.0]


def _expect(node: dict, x: np.ndarray, S: set) -> float:
    """E[the tree's value | the features in S are x's], the path-dependent way (TreeSHAP's): a split on a feature
    outside S goes both ways, weighted by how many training rows went each way."""
    if "leaf_value" in node:
        return float(node["leaf_value"])
    j = node["split_feature"]
    left, right = node["left_child"], node["right_child"]
    if j in S:
        return _expect(left if x[j] <= node["threshold"] else right, x, S)
    cl = left.get("internal_count", left.get("leaf_count"))
    cr = right.get("internal_count", right.get("leaf_count"))
    return (cl * _expect(left, x, S) + cr * _expect(right, x, S)) / (cl + cr)


def test_shap_is_the_plain_shapley_sum_over_every_subset():
    """LightGBM's TreeSHAP (pred_contrib) = the Shapley value computed the long way: every subset of the 3 things,
    each weighted by |S|! (M - |S| - 1)! / M!; and the pushes plus the start value add up to the guess."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = 40 * (X["a"] > 0) + 25 * X["b"] * (X["c"] > 0.3) + rng.normal(0, 5, 300)
    params = {**config.MODEL_PARAMS, "min_data_in_leaf": 10, "bagging_fraction": 1.0, "bagging_freq": 0,
              "feature_fraction": 1.0}
    bst = MD.fit(X, y, seed=1, text_columns=[], params=params, rounds=12)
    trees = [t["tree_structure"] for t in bst.dump_model()["tree_info"]]
    contrib = bst.predict(X.iloc[:8], pred_contrib=True)
    M = 3

    def v(x, S):
        return sum(_expect(t, x, S) for t in trees)

    for k in range(8):
        x = X.iloc[k].to_numpy()
        assert contrib[k].sum() == pytest.approx(bst.predict(X.iloc[[k]])[0], abs=1e-8)
        assert contrib[k, M] == pytest.approx(v(x, set()), abs=1e-6)                 # the start value
        for j in range(M):
            others = [i for i in range(M) if i != j]
            phi = 0.0
            for r in range(M):
                for S in itertools.combinations(others, r):
                    w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                    phi += w * (v(x, set(S) | {j}) - v(x, set(S)))
            assert contrib[k, j] == pytest.approx(phi, abs=1e-6)


# ================================================================================================================
# 3. the take/skip filter: a new version, counted, judged by walk-forward
# ================================================================================================================

def _one_day_each_tune_year() -> pd.DataFrame:
    return pd.concat([walk(f"{y}-03-{d:02d}", days=1, seed=y)
                      for y, d in ((2016, 7), (2017, 6), (2018, 5), (2019, 11), (2020, 9))])


def test_the_filter_keeps_the_setups_guessed_above_the_line_and_renames_them():
    m1 = walk("2019-03-11", days=1)
    base = signals_for(m1, "idea-970", "v-base")
    guesses = {s.id: (50.0 if k % 3 == 0 else -20.0) for k, s in enumerate(base[:30])}
    vid = TS.version_id("v-base", "spec-1")
    sigs = TS.signals(base, guesses, vid)
    assert len(sigs) == 10 and {s.version_id for s in sigs} == {vid} and all(s.idea_id == "idea-970" for s in sigs)
    assert [s.known_at for s in sigs] == [s.known_at for k, s in enumerate(base[:30]) if k % 3 == 0]
    assert vid.startswith("v-") and len(vid) == 12 and TS.version_id("v-base", "spec-2") != vid
    assert TS.version_id("v-base", "spec-1") == vid                          # the same recipe, the same version
    assert "more than ₹0 after cost" in TS.rule_text()


def test_the_filter_waits_while_the_check_years_are_closed_and_takes_nothing_without_a_setup():
    form = make_form("idea-971")
    store.create(form)
    m1 = walk("2019-03-11", days=1)
    base = signals_for(m1, "idea-971", "v-base")
    with pytest.raises(TS.FilterError, match="takes no setup"):
        TS.make(form, base, {s.id: -1.0 for s in base}, base_version="v-base", spec="s", tf="5m", m1=m1)
    with pytest.raises(TS.FilterError, match="still closed"):
        TS.make(form, base, {s.id: 5.0 for s in base}, base_version="v-base", spec="s", tf="5m", m1=m1)


def test_the_filter_is_a_counted_try_judged_by_walk_forward(_trials_in_tmp):
    """The family opened its check years for the judged version: the filter version is tried on 2016-2020 (a tester
    file - a try), then judged by walk-forward (a referee file), and never opens 2021-2023."""
    form = make_form("idea-972")
    store.create(form)
    m1 = _one_day_each_tune_year()
    base = signals_for(m1, "idea-972", "v-base")
    from machine.tester.run import run as tester_run
    tester_run(base, tf="5m", m1=m1)                                        # the judged version: tried first
    splits.open_check("idea-972", "v-base", who="test", why="chosen on the tune years")
    late = [s for s in base if pd.Timestamp(s.known_at).year >= 2018]
    guesses = {s.id: (30.0 if k % 2 else -30.0) for k, s in enumerate(late)}
    out = TS.make(form, base, guesses, base_version="v-base", spec="spec", tf="5m", m1=m1, n_random=3,
                  log=lambda *_: None)
    vid = out["version_id"]
    assert out["made"] and out["setups_taken"] == len(late) // 2 and out["setups_guessed"] == len(late)
    assert trials.count("idea-972").idea_versions == ["v-base", vid] and out["try"] == 2
    kinds = sorted(json.loads(p.read_text(encoding="utf-8"))["kind"] for p in _trials_in_tmp.glob("*.json"))
    assert kinds == ["referee", "tester", "tester"]
    ref = next(json.loads(p.read_text(encoding="utf-8")) for p in _trials_in_tmp.glob("*.json")
               if json.loads(p.read_text(encoding="utf-8"))["kind"] == "referee")
    assert ref["version_id"] == vid and ref["settings"]["judged_on"].startswith("walk-forward inside 2016-2020")
    assert set(ref["settings"]["years"]) <= {2018, 2019, 2020}
    assert splits.opened_count() == 1                                        # the check years were not opened again
    assert project.summary()["answers"] == 1                                 # the filter's answer is an answer


# ================================================================================================================
# 4. the pooled view: any entry, or only this idea
# ================================================================================================================

def _claims(r, key):
    return [(c["condition"], c["part"], c["claim"]) for c in r[key]]


def test_a_london_edge_of_this_idea_only_is_only_this_idea():
    r = PL.analyse(fake_pool(1, gap_this=300.0), [], n_boot=1000, n_shuffle=1000)
    assert _claims(r, "claims_only") == [("session", "London", "more")] and r["claims_any"] == []
    c = r["claims_only"][0]
    assert c["gap_this"] > 250 and abs(c["gap_random"]) < 60 and c["gap"] == pytest.approx(
        c["gap_this"] - c["gap_random"], abs=0.02)
    assert "this idea does better there than any entry would" in PL.only_line(r)
    assert PL.any_line(r) == "Helps or hurts ANY entry: nothing passed the 3 checks"


def test_a_london_edge_for_every_entry_is_any_entry_not_this_idea():
    r = PL.analyse(fake_pool(2, gap_this=300.0, gap_random=300.0), [], n_boot=1000, n_shuffle=1000)
    assert ("session", "London", "more") in _claims(r, "claims_any")
    assert {c for c, _, _ in _claims(r, "claims_any")} <= {"session", "new_york"}   # New York = "not London" here
    assert r["claims_only"] == []
    assert "any entry earns more there (random entries ₹" in PL.any_line(r)


def test_a_noise_condition_is_not_found_in_the_pool():
    r = PL.analyse(fake_pool(3), [], n_boot=1000, n_shuffle=1000)
    assert r["claims_any"] == [] and r["claims_only"] == []
    assert r["counts"]["tested"] == len(PL.MARKET) == 36
    assert all(not c.needs_facts for c in PL.MARKET)


def test_the_difference_is_counted_by_hand():
    this = fake_pool(4, this=300, random=900)
    lab_a = pd.Series(np.where(np.arange(300) % 3 == 0, "London", "Asia"), index=this.trades.index, dtype=object)
    lab_b = pd.Series(np.where(np.arange(900) % 4 == 0, "London", "Asia"), index=this.random_trades.index,
                      dtype=object)
    t = PL.diff_test(this.trades, lab_a, this.random_trades, lab_b, ("Asia", "London"), seed=1, n_boot=200,
                     n_shuffle=200)
    va, vb = this.trades["net_inr"], this.random_trades["net_inr"]
    ga = va[lab_a == t["part"]].mean() - va[lab_a != t["part"]].mean()
    gb = vb[lab_b == t["part"]].mean() - vb[lab_b != t["part"]].mean()
    assert t["gap_this"] == pytest.approx(ga, abs=0.01) and t["gap_random"] == pytest.approx(gb, abs=0.01)
    assert t["gap"] == pytest.approx(ga - gb, abs=0.02) and t["tested"]
    few = PL.diff_test(this.trades.iloc[:40], lab_a.iloc[:40], this.random_trades, lab_b, ("Asia", "London"),
                       seed=1, n_boot=50, n_shuffle=50)
    assert few == {"tested": False, "why": "too few trades", "p": 1.0, "p_shuffle": 1.0, "eligible": []}


def test_the_pool_reads_the_tune_years_only_and_every_trade_needs_its_setup():
    s = fake_pool(5, this=200, random=400)
    s.trades.iloc[0, s.trades.columns.get_loc("exit_time")] = pd.Timestamp("2022-01-04")
    with pytest.raises(ValueError, match="tune years only"):
        PL.analyse(s, [], n_boot=50, n_shuffle=50)
    s2 = fake_pool(6, this=200, random=400)
    s2.random_feats = s2.random_feats.iloc[5:]
    with pytest.raises(ValueError, match="have no setup"):
        PL.analyse(s2, [], n_boot=50, n_shuffle=50)


def test_his_other_ideas_are_shown_next_to_this_one_not_tested():
    this = fake_pool(7, this=400, random=1200, gap_this=250.0)
    other = fake_pool(8, this=500, random=1000)
    other.name, other.label = "entry-other", "another tested idea"
    other.trades.index = pd.Index([f"o{i}" for i in range(500)], name="signal")
    other.feats.index = other.trades.index
    other.random_trades.index = pd.Index([f"or{i}" for i in range(1000)], name="signal")
    other.random_feats.index = other.random_trades.index
    r = PL.analyse(this, [other], n_boot=1000, n_shuffle=1000)
    assert r["other_ideas"] == ["entry-other"] and r["other_trades"] == 500 and r["random_trades"] == 2200
    s = next(c for c in r["conditions"] if c["name"] == "session")
    assert sum(p["others"]["trades"] for p in s["parts"]) == 500
    assert sum(p["random"]["trades"] for p in s["parts"]) == 2200               # every random entry together
    assert sum(p["own_random"]["trades"] for p in s["parts"]) == 1200            # this idea's own
    assert "2,200 random entries, 500 trades of 1 other tested idea" in PL.head_line(r)


def test_random_entries_keep_the_ideas_shape():
    """The pool's random entries are the referee's (the same side, distances, trades a month): each random signal is
    one of the idea's own templates at a random time in the same month, and its setup's prices follow."""
    from machine.ideas.compile import Market
    from machine.referee import random_entries as RE
    m1 = walk("2019-03-11", days=3)
    sigs = signals_for(m1, "idea-973", "v1")
    seg = make_segment("2019", (2019,), "5m", sigs, m1, strict=True)
    tr = closed_trades(replay(seg, sigs))
    rs, rt = PL.random_entries(seg, tr[["signal", "known_at"]], runs=3, cap=100_000, seed=11, name="idea-973")
    assert len(rt) and set(rt["signal"]) <= set(rs) and all(x.startswith("idea-973:rand") for x in rs)
    shapes = {(t.direction, t.entry_type, t.stop_d, t.target_ds) for t in
              (RE.template(seg, s) for s in sigs if s.id in set(tr["signal"]))}
    for s in rs.values():
        t = RE.template(seg, s)
        assert (t.direction, t.entry_type, round(t.stop_d, 2), tuple(round(x, 2) for x in t.target_ds)) in \
            {(a, b, round(c, 2), tuple(round(x, 2) for x in d)) for a, b, c, d in shapes}
    mk = Market.from_m1(m1, tfs=("5m", "1h", "1d"))
    for s in list(rs.values())[:20]:
        st = PL.setup_of(s, mk)
        if s.entry_type == "limit":
            assert st.entry == s.entry_price and st.stop == s.stop
        else:
            assert st.stop == pytest.approx(st.entry - s.stop) and st.target == pytest.approx(st.entry + 1.5)
    one = PL.random_entries(seg, tr[["signal", "known_at"]], runs=3, cap=1, seed=11, name="idea-973")[1]
    assert 0 < len(one) <= len(rt) and len(one) < len(rt)                      # whole runs: stops after the first


# ================================================================================================================
# 5. the report has the lines
# ================================================================================================================

_MADE: dict = {}


def _results() -> tuple[dict, dict]:
    """One model result (made-up trades, no edge) and one pooled result (a London edge of this idea only), made once
    for the report tests - each test writes its own copies into its own trial folder."""
    if not _MADE:
        tr, f = planted(51)
        _MADE["model"] = MD.analyse(f, tr, shuffles=20)
        _MADE["pooled"] = PL.analyse(fake_pool(52, this=400, random=1200, gap_this=300.0), [], n_boot=1000,
                                     n_shuffle=1000)
    return json.loads(json.dumps(_MADE["model"])), json.loads(json.dumps(_MADE["pooled"]))


def with_model(idea: str = IDEA, filt: dict | None = None):
    from machine.learner import slices as SL
    from machine.referee import tune
    made, _, _ = with_learner(idea)
    chosen = made["tune"].chosen
    tfile = tune.latest(idea)
    r, p = _results()
    r["tune_trial"] = p["tune_trial"] = tfile["trial_id"]
    r["filter"] = filt or {"made": False, "why": "the model's check did not pass - a filter from a model that found "
                                                "nothing would spend one of the family's tries on noise"}
    trial_files.record(kind=config.MODEL_KIND, idea_id=idea, version_id=chosen, parent_idea=None,
                       settings=MD.settings_used(), data={"prices": "hand-made"}, result=r)
    trial_files.record(kind=config.POOLED_KIND, idea_id=idea, version_id=chosen, parent_idea=None,
                       settings={"rules_sha256": SL.RULES_SHA256}, data={"prices": "hand-made"}, result=p)
    return made, r, p


def test_the_report_has_the_models_and_the_pooled_lines():
    _, r, p = with_model()
    b = B.build(IDEA)
    text = b.text
    assert len(text) <= 4096 and B.check_words(text) == []
    assert "What a model learned (LightGBM, a program that learns from tables" in text
    assert "No take/skip filter was made from it: the model's check did not pass" in text
    assert "Any entry or only this idea" in text and "not measured yet" not in text
    page = b.page_path.read_text(encoding="utf-8")
    assert B.check_words(B._visible(page)) == []
    for words in ("What a model learned — LightGBM (a program that learns from tables)",
                  "What it leaned on — SHAP (how much each thing pushed its guess",
                  "How each year's model was made", "Any entry or only this idea — every tested idea's trades",
                  "Only this idea: the part that stands out", "Its take/skip filter"):
        assert words in page, words


def test_every_rupee_of_the_new_lines_is_in_the_files():
    _, r, p = with_model()
    f = B.facts(IDEA)
    lines = "\n".join(B.model_lines(f) + B.pooled_lines(f))
    allowed = {B.inr(x) for x in numbers_in(f.model["result"]) + numbers_in(f.pooled["result"])} | {"₹0"}
    import re
    shown = re.findall(r"-?₹[\d,]+", lines)
    assert shown
    for amount in shown:
        assert amount in allowed, amount


def test_no_model_file_says_not_measured_yet():
    with_learner("idea-975")
    text = B.build("idea-975").text
    assert "What a model learned (LightGBM, a program that learns from tables): not measured yet" in text
    assert "Any entry or only this idea: not measured yet (python -m machine.learner idea-975 --pooled)" in text


def test_a_filter_version_is_shown_next_to_the_idea_never_as_its_judgement():
    made, _, _ = with_model("idea-976", filt={"made": True, "version_id": "v-filter0001", "try": 3, "budget": 20,
                                              "rule": TS.rule_text()})
    first = next(x for x in trials.load() if x["kind"] == "referee" and x["idea_id"] == "idea-976")
    copy = json.loads(json.dumps(first))
    copy["version_id"] = "v-filter0001"
    copy["settings"]["judged_on"] = "walk-forward inside 2016-2020: 2018 (...)"
    copy["settings"]["years"] = [2018, 2019, 2020]
    copy["result"]["per_year"] = {"2018": {"trades": 3, "net_inr": -120.0, "per_trade": -40.0}}
    trial_files.record(kind="referee", idea_id="idea-976", version_id="v-filter0001", parent_idea=None,
                       settings=copy["settings"], data=copy["data"], result=copy["result"])
    f = B.facts("idea-976")
    assert f.judge["version_id"] == made["tune"].chosen and f.mode == "check"
    assert f.filter_judge is not None and f.filter_judge["version_id"] == "v-filter0001"
    line = B.filter_line(f)
    assert line.startswith("Its take/skip filter as a new version (v-filter0001, try 3 of 20): ")
    assert "judged year by year on 2018–2020" in line
    page = B.build("idea-976").page_path.read_text(encoding="utf-8")
    assert "take a setup only when the model made before its year guesses more than ₹0" in page


def test_a_long_report_keeps_the_models_details_on_the_page(monkeypatch):
    with_model("idea-977")
    f = B.facts("idea-977")
    monkeypatch.setattr(B.pooled_words, "any_line", lambda r: "x" * 1800)
    text = B.telegram_text(f)
    assert len(text) <= 4096 and "x" * 100 not in text
    assert "(details on report.html)" in text and "helps or hurts ANY entry:" in text


# ================================================================================================================
# 6. the runner's pieces
# ================================================================================================================

def test_a_model_run_is_not_a_try_and_not_an_answer():
    with_model("idea-978")
    c = trials.count("idea-978")
    assert len(c.idea_versions) == 2                                        # the tune's two versions, nothing more
    assert project.summary()["answers"] == 1                                # the referee's one answer only
    assert LR.latest_of(config.MODEL_KIND, "idea-978") is not None
    assert LR.latest_of(config.POOLED_KIND, "idea-978", "v-nothing") is None


def test_the_filter_rule_is_written_before_any_money():
    assert config.FILTER_ABOVE_INR == 0.0 and config.MODEL_PURGE_DAYS == 7 and config.MODEL_SHUFFLES == 200
    assert config.WALK_FORWARD_JUDGE_YEARS == (2018, 2019, 2020)
    s = MD.settings_used()
    assert s["params"]["objective"] == "regression" and s["rounds"] == config.MODEL_ROUNDS
    assert MD.spec_sha256({"a": 1}) != MD.spec_sha256({"a": 2})


def test_signal_copies_stay_valid_signals():
    m1 = walk("2019-03-11", days=1)
    base = signals_for(m1, "idea-979", "v-b")
    out = TS.signals(base, {x.id: 1.0 for x in base[:6]}, "v-new0000001")
    assert len(out) == 6 and len({s.id for s in out}) == 6
    for s, b in zip(out, base[:6]):
        assert isinstance(s, Signal) and s.version_id == "v-new0000001" and s.id == f"v-new0000001.{b.id}"
        assert (s.known_at, s.direction, s.entry_price, s.stop, s.targets) == (b.known_at, b.direction, b.entry_price,
                                                                               b.stop, b.targets)
    named = Signal(**{**base[0].model_dump(), "id": "idea-979.v-b.S1"})
    assert TS.signals([named], {named.id: 1.0}, "v-x")[0].id == "idea-979.v-x.S1"
