"""His Box 4 beliefs as numbers (L13b): the world ends with the judged years, the race / reach / fair-comparison tools
on hand-made candles, planted differences found and noise not (both kinds of comparison), already seen can never
hold, the check years open once, the cut-off-the-future test for the events, and every Box 4 row has its belief and
its page.

Everything runs on made-up candles and made-up events (no price data), except the marked one at the end.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from machine import config, guards
from machine.beliefs import engine, events as EV, levels as LV, registry
from machine.beliefs.engine import CHANCE, FEW, FRESH, HOLDS, OPPOSITE, Belief
from machine.beliefs.world import BOTH, OPEN, UP, World
from machine.referee import splits


# ---- made-up candles -------------------------------------------------------------------------------------------------

def walk(start="2019-01-01", end="2019-03-31", seed=1, step=0.12) -> pd.DataFrame:
    """Made-up 1-minute candles: a random walk, Monday-Friday, a daily break before 01:00 server time."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end + " 23:59", freq="1min")
    t = t[(t.dayofweek < 5) & (t.hour >= 1)]
    close = 1300 + np.cumsum(rng.normal(0, step, len(t)))
    open_ = np.r_[1300, close[:-1]]
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.05, len(t)))
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.05, len(t)))
    return pd.DataFrame({"open": open_.round(2), "high": hi.round(2), "low": lo.round(2), "close": close.round(2),
                         "vol": 10, "spread": 20.0}, index=pd.DatetimeIndex(t, name="server_time"))


def flat(n_days=30, start="2019-01-07") -> pd.DataFrame:
    """Made-up candles that never move (1300.00), to place hand-made moves on."""
    t = pd.date_range(start, periods=n_days * 1440, freq="1min")
    t = t[t.dayofweek < 5]
    return pd.DataFrame({"open": 1300.0, "high": 1300.0, "low": 1300.0, "close": 1300.0, "vol": 10, "spread": 20.0},
                        index=pd.DatetimeIndex(t, name="server_time"))


@pytest.fixture(scope="module")
def noise_world():
    return World.from_m1(walk(), (2019,))


# ---- the rules -------------------------------------------------------------------------------------------------------

def test_the_rules_are_fixed_in_advance():
    """Changing a rule after seeing a result changes this fingerprint - and shows in git."""
    assert engine.RULES_SHA256 == "d57730b97fcf48c7c34c996a00a9552b0e481ebf3356a715cd9ce3d3b3fb5e4e"
    for fam, versions in registry.families().items():
        assert fam.startswith("belief-") and versions
        for b in versions:
            assert b.idea_id == fam and b.box.startswith("B4-") and b.his_words and b.source and b.counted
            assert b.window and b.compare and b.unit in ("share", "range", "inr") and b.code
            assert guards.IDEA_NAME.fullmatch(fam)


@pytest.mark.private
def test_every_box_4_row_has_a_belief_and_a_page():
    """B4-01 ... B4-22 each have a belief here (B4-07: L12b.2; B4-23: L20), and every belief its page, written
    before it ran."""
    boxes = {b.box for vs in registry.families().values() for b in vs}
    rows = {f"B4-{i:02d}" for i in range(1, 24)}
    assert boxes | set(registry.ELSEWHERE) == rows and not boxes & set(registry.ELSEWHERE)
    for fam in registry.families():
        page = config.ROOT / "plan" / "work" / "beliefs" / f"{fam.removeprefix('belief-')}.md"
        assert page.is_file(), f"{fam} has no page {page}"


# ---- the world and its tools ----------------------------------------------------------------------------------------

def test_the_world_ends_with_the_judged_years():
    m1 = walk("2019-12-20", "2020-01-10")
    with pytest.raises(ValueError, match="must end"):
        World.from_m1(m1, (2019,))
    w = World.from_m1(m1[m1.index < "2020-01-01"], (2019,))
    assert w.t[-1] < pd.Timestamp("2020-01-01")


def test_a_race_is_decided_by_the_first_candle_that_reaches_a_price():
    m1 = flat(3)
    t0 = m1.index[600]
    m1.loc[t0:t0 + pd.Timedelta("4min"), "high"] = 1302.0                  # one 5-minute candle reaches 1302
    t1 = m1.index[900]
    m1.loc[t1:t1 + pd.Timedelta("4min"), ["high", "low"]] = [1303.0, 1297.0]   # a candle that reaches both
    w = World.from_m1(m1, (2019,))
    i0 = int(w.at_or_after([t0])[0])
    ans, bar = w.race(np.array([0, 0, i0 + 1, 0]), np.array([1301.5, 1310.0, 1302.5, 1302.5]),
                      np.array([1290.0, 1290.0, 1297.5, 1299.0]), np.array([10_000, 10_000, 10_000, 5]))
    assert list(ans[:1]) == [UP] and bar[0] == i0
    assert ans[1] == OPEN                                                  # 1310 never comes, 1290 never comes
    assert ans[2] == BOTH                                                  # the candle that reaches both: unclear
    assert ans[3] == OPEN                                                  # the window was too short
    x = w.reach(np.array([0, 0, w.n - 3]), np.array([1302.0, 1305.0, 1305.0]), np.array(["high", "high", "high"]),
                np.array([10_000, w.n, 100]))
    assert x[0] == 1.0 and x[1] == 0.0 and np.isnan(x[2])                  # the years ended before the window did


def test_the_comparison_moments_are_the_same_year_weekday_and_hour_never_the_same_day(noise_world):
    w = noise_world
    pos = np.array([1000, 5000, 9000])
    c = w.controls(pos, 20, np.random.default_rng(0))
    assert (c >= 0).all()
    for p, row in zip(pos, c):
        assert (w.year[row] == w.year[p]).all() and (w.weekday[row] == w.weekday[p]).all()
        assert (w.hour[row] == w.hour[p]).all() and (w.day[row] != w.day[p]).all()
    again = w.controls(pos, 20, np.random.default_rng(0))
    assert (c == again).all()                                              # the same draws on every PC


def test_fake_levels_are_as_far_from_price_on_the_same_side(noise_world):
    w = noise_world
    birth = np.array([2000, 6000])
    dist = np.array([0.30, 0.05])
    c, fp = LV.fake_prices(w, birth, np.array(["high", "low"]), dist, np.random.default_rng(1))
    assert np.allclose((fp[0] - w.close[c[0]]) / w.ruler[c[0]], 0.30)
    assert np.allclose((w.close[c[1]] - fp[1]) / w.ruler[c[1]], 0.05)


def test_the_race_after_a_take_reads_the_take_candle_carefully():
    m1 = flat(3)
    w0 = World.from_m1(m1, (2019,))
    i = 700
    t = w0.t[i]
    m1.loc[t:t + pd.Timedelta("4min"), "high"] = 1305.0                    # the take candle runs 5 past the level
    w = World.from_m1(m1, (2019,))
    # a high at 1301, R = 2: the take candle already reached 1303 (the far price) -> the level did not turn
    assert LV.turn_race(w, np.array([i]), np.array([1301.0]), np.array([1301.0]), np.array(["high"]),
                        np.array([2.0]))[0] == 0.0
    m1.loc[t:t + pd.Timedelta("4min"), ["high", "low"]] = [1301.5, 1296.0]  # its low at the turn price: unclear
    w = World.from_m1(m1, (2019,))
    assert np.isnan(LV.turn_race(w, np.array([i]), np.array([1301.0]), np.array([1301.0]), np.array(["high"]),
                                 np.array([2.0]))[0])
    later = w.t[i + 30]
    m1.loc[later:later + pd.Timedelta("4min"), "low"] = 1298.0             # later, price falls to 1299 (the turn)
    m1.loc[t:t + pd.Timedelta("4min"), ["high", "low"]] = [1301.5, 1300.0]
    w = World.from_m1(m1, (2019,))
    assert LV.turn_race(w, np.array([i]), np.array([1301.0]), np.array([1301.0]), np.array(["high"]),
                        np.array([2.0]))[0] == 1.0


def test_a_level_alive_at_a_close_is_known_by_then_and_not_yet_taken(monkeypatch):
    w = SimpleNamespace(market=None)
    lf = pd.DataFrame({"kind": ["equal_low", "session_low", "swing_low", "equal_low"],
                       "target_when": ["always", "always", "higher_timeframe_against", "always"],
                       "start": [10, 10, 10, 30], "end": [50, 20, 50, 60], "price": [1290.0, 1295.0, 1298.0, 1292.0],
                       "slope": 0.0, "x0pos": 0, "side": "low", "level_id": ["a", "b", "c", "d"], "life_bars": 2880})
    monkeypatch.setattr(LV, "life", lambda _w: lf)
    al = LV.Alive(w)
    px, ids = al.nearest(np.array([15, 25, 29, 40]), np.array(["low"] * 4), np.array([1300.0] * 4))
    assert list(ids) == ["b", "a", "d", "d"]            # b taken on 20; d known at the close of 29 (start 30)
    assert px[0] == 1295.0 and px[1] == 1290.0          # the 5-minute swing is never "the next liquidity"


def test_a_band_is_touched_when_a_candle_enters_it_from_outside(noise_world):
    w = noise_world
    c = np.array([[100]])
    lo = np.array([[w.close[100] + 0.5]])
    hi = lo + 0.2
    j, came = LV.band_touches(w, c, lo, hi, 5000)
    if j[0, 0] >= 0:
        k = j[0, 0]
        assert (w.low[k] <= hi[0, 0]) and (w.high[k] >= lo[0, 0])
        assert w.high[k - 1] < lo[0, 0] or w.low[k - 1] > hi[0, 0]
        assert came[0, 0] == (1 if w.high[k - 1] < lo[0, 0] else -1)


# ---- the judgement on made-up events -------------------------------------------------------------------------------

def stub_world(years):
    return SimpleNamespace(years=tuple(years), fp="test")


def pool_belief(rows, **kw):
    return Belief("belief-test", kw.pop("version", "v1"), "B4-99", "a test belief", "words", "a test", "counted",
                  "window", "compare", kw.pop("unit", "share"), lambda w: rows, **kw)


def made_events(p_event=0.5, p_pool=0.5, years=(2016, 2017, 2018), per_year=80, seed=0) -> pd.DataFrame:
    """One event a few days apart: x = 1 with p_event; 20 comparison values each = 1 with p_pool."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([t for y in years for t in pd.date_range(f"{y}-01-03", periods=per_year, freq="4D")])
    out = pd.DataFrame({"x": (rng.random(len(idx)) < p_event).astype(float)}, index=idx)
    out["pool"] = [(rng.random(20) < p_pool).astype(float) for _ in range(len(idx))]
    out["side"] = np.where(np.arange(len(idx)) % 2, "high", "low")
    return out


def made_groups(mean_a=0.0, mean_b=0.0, years=(2016, 2017, 2018), n_a=60, n_b=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for g, n, m in (("A", n_a, mean_a), ("B", n_b, mean_b)):
            t = pd.date_range(f"{y}-01-03", f"{y}-12-20", periods=n)
            for tt, x in zip(t, rng.normal(m, 1.0, n)):
                rows.append((tt, x, g, f"{y}|all"))
    df = pd.DataFrame(rows, columns=["t", "x", "group", "stratum"]).set_index("t").sort_index()
    return df


def test_a_planted_difference_holds_and_the_opposite_is_found_too():
    years = (2016, 2017, 2018)
    b = pool_belief(made_events(0.62, 0.5))
    r = engine.judge(b, stub_world(years), record=False)
    assert r.word == HOLDS and r.way == 1
    b = pool_belief(made_events(0.38, 0.5, seed=1))
    r = engine.judge(b, stub_world(years), record=False)
    assert r.word == OPPOSITE and r.way == -1


def test_no_difference_is_chance_nearly_always():
    """40 worlds where the belief changes nothing, judged one way (as on the check years): "holds" at most 4 times."""
    words = [engine.judge(pool_belief(made_events(seed=s)), stub_world((2016, 2017, 2018)), way=1, record=False).word
             for s in range(40)]
    assert words.count(HOLDS) <= 4 and words.count(OPPOSITE) == 0 and words.count(CHANCE) >= 36


def test_two_groups_planted_found_noise_not():
    b = pool_belief(made_groups(0.5, 0.0), groups=True, unit="range")
    assert engine.judge(b, stub_world((2016, 2017, 2018)), record=False).word == HOLDS
    b = pool_belief(made_groups(-0.5, 0.0, seed=2), groups=True, unit="range")
    assert engine.judge(b, stub_world((2016, 2017, 2018)), record=False).word == OPPOSITE
    words = [engine.judge(pool_belief(made_groups(seed=s), groups=True, unit="range"), stub_world((2016, 2017, 2018)),
                          way=1, record=False).word for s in range(40)]
    assert words.count(HOLDS) <= 4 and words.count(CHANCE) >= 36


def test_two_groups_count_the_comparison_groups_own_luck():
    """A small comparison group is itself uncertain: the luck test must say so (the first build treated its average
    as exact and called noise rare)."""
    small = [engine.measure(pool_belief(made_groups(n_a=60, n_b=6, seed=s), groups=True, unit="range"),
                            made_groups(n_a=60, n_b=6, seed=s), (2016, 2017, 2018))[0] for s in range(40)]
    rare = sum(min(n["p_his"], n["p_opposite"]) < 0.05 for n in small)
    assert rare <= 8                                          # about 4 of 40 expected; the exact-average error gave ~20


def test_already_seen_can_never_hold():
    b = pool_belief(made_events(0.65, 0.5), seen_note="the old project tested it")
    r = engine.judge(b, stub_world((2016, 2017, 2018)), record=False)
    assert r.word == FRESH and "pass every other check" in r.why


def test_one_year_against_is_chance_and_the_worst_year_is_named():
    ev = pd.concat([made_events(0.66, 0.5, years=(2016, 2017)), made_events(0.35, 0.5, years=(2018,), seed=3)])
    r = engine.judge(pool_belief(ev), stub_world((2016, 2017, 2018)), record=False)
    assert r.word == CHANCE and "every year" in r.why and "2018" in r.checks[4].shown


def test_few_events_is_not_enough_to_tell():
    ev = made_events(0.9, 0.5, per_year=8)
    assert engine.judge(pool_belief(ev), stub_world((2016, 2017, 2018)), record=False).word == FEW


def test_three_helpful_events_cannot_carry_a_belief():
    ev = made_events(0.5, 0.5, seed=4).assign(x=0.0)
    ev["pool"] = [np.full(20, 0.1) for _ in range(len(ev))]
    ev.iloc[:3, ev.columns.get_loc("x")] = 50.0                            # three huge events, the rest below
    r = engine.judge(pool_belief(ev, unit="range"), stub_world((2016, 2017, 2018)), record=False)
    assert r.checks[2].passed is False and r.word != HOLDS


def test_an_event_outside_the_judged_years_is_refused():
    ev = made_events(years=(2016, 2021))
    with pytest.raises(ValueError, match="outside the judged years"):
        engine.measure(pool_belief(ev), ev, (2016,))


def test_the_check_years_open_once_and_only_for_the_chosen_version():
    b1 = pool_belief(made_events(0.6, 0.5, years=(2016, 2017)), version="v1")
    b2 = pool_belief(made_events(0.6, 0.5, years=(2016, 2017)), version="v2")
    chk = pool_belief(made_events(0.66, 0.5, years=(2021, 2022)), version="v1")
    with pytest.raises(guards.CheckYearsClosedError):
        engine.judge(chk, stub_world((2021, 2022)), way=1)
    with pytest.raises(guards.CheckYearsClosedError):
        World.load((2021, 2022, 2023))
    first = engine.judge(b1, stub_world((2016, 2017)))
    engine.judge(b2, stub_world((2016, 2017)))
    assert json.loads(first.trial_path.read_text(encoding="utf-8"))["kind"] == "belief"
    splits.open_check("belief-test", "v1", who="test", why="chosen")
    with pytest.raises(ValueError, match="ONE way"):
        engine.judge(chk, stub_world((2021, 2022)))
    final = engine.judge(chk, stub_world((2021, 2022)), way=1, final=True)
    assert final.final and final.word == HOLDS
    other = pool_belief(made_events(0.6, 0.5, years=(2021, 2022)), version="v2")
    with pytest.raises(guards.CheckYearsClosedError):
        engine.judge(other, stub_world((2021, 2022)), way=1)
    with pytest.raises(guards.CheckYearsClosedError):
        splits.open_check("belief-test", "v2", who="test", why="again")


def test_tune_chooses_one_version_per_belief_once():
    years = config.TUNE_YEARS
    fams = {"belief-test": [pool_belief(made_events(0.5, 0.5, years=years, seed=7), version="weak"),
                            pool_belief(made_events(0.64, 0.5, years=years, seed=8), version="strong")]}
    w = stub_world(years)
    reps = engine.tune(fams, w)
    chosen = engine.read_chosen()
    assert chosen["belief-test"]["version_id"] == "strong" and chosen["belief-test"]["way"] == 1
    assert len(reps["belief-test"]) == 2 and len(list(config.TRIALS_DIR.glob("*.json"))) == 2
    assert (config.BELIEFS_DIR / "test-tune.md").is_file()
    with pytest.raises(ValueError, match="already have a chosen version"):
        engine.tune(fams, w)


def test_a_run_again_after_a_fix_keeps_the_first_choice_and_the_same_versions():
    years = config.TUNE_YEARS
    fams = {"belief-test": [pool_belief(made_events(0.64, 0.5, years=years, seed=8), version="strong")]}
    w = stub_world(years)
    with pytest.raises(ValueError, match="no first choice"):
        engine.retune(fams, w, "a fix")
    engine.tune(fams, w)
    with pytest.raises(ValueError, match="needs its reason"):
        engine.retune(fams, w, " ")
    fixed = {"belief-test": [pool_belief(made_events(0.36, 0.5, years=years, seed=8), version="strong")]}
    engine.retune(fixed, w, "the events were read wrong")
    c = engine.read_chosen()["belief-test"]
    assert c["way"] == -1 and c["before_fix"]["way"] == 1 and c["fix"] == "the events were read wrong"
    assert len(list(config.TRIALS_DIR.glob("*.json"))) == 2                   # both runs' trial files stay
    new_version = {"belief-test": [pool_belief(made_events(years=years), version="another-reading")]}
    with pytest.raises(ValueError, match="new version"):
        engine.retune(new_version, w, "a fix")


# ---- the events do not peek -----------------------------------------------------------------------------------------

def _cut(m1, t):
    return World.from_m1(m1[m1.index < t], (2019,))


def test_the_events_are_the_same_when_the_future_is_cut_off():
    """An event's definition (where, which level, how far, R) uses nothing after its moment: cutting the candles later
    changes none of the events known before the cut (their outcomes may be unfinished in the cut world)."""
    m1 = walk("2019-01-01", "2019-03-31", seed=5)
    full = World.from_m1(m1, (2019,))
    cut_at = pd.Timestamp("2019-03-01")
    part = _cut(m1, cut_at)
    tf, tp = EV.takes(full), EV.takes(part)
    known = lambda w, t: LV.times(w, t["pos"]) < cut_at - pd.Timedelta(days=1)      # noqa: E731
    cols = ["pos", "level", "side", "kind", "dist_adr", "r_adr"]
    a = tf[known(full, tf) & tf["ok"]][cols].reset_index(drop=True)
    b = tp[known(part, tp) & tp["ok"]][cols].reset_index(drop=True)
    assert len(a) > 50
    pd.testing.assert_frame_equal(a, b)
    zf, zp = EV.zone_touches(full), EV.zone_touches(part)
    zc = ["pos", "zone_id", "touch", "side", "r_adr"]
    za = zf[LV.times(full, zf["pos"]) < cut_at - pd.Timedelta(days=1)][zc].reset_index(drop=True)
    zb = zp[LV.times(part, zp["pos"]) < cut_at - pd.Timedelta(days=1)][zc].reset_index(drop=True)
    pd.testing.assert_frame_equal(za, zb)
    ef, ep = EV.equal_births(full), EV.equal_births(part)
    ec = ["pos", "price", "side", "touches", "dist_adr", "gap_cents"]
    ea = ef[LV.times(full, ef["pos"]) < cut_at][ec].reset_index(drop=True)
    eb = ep[LV.times(part, ep["pos"]) < cut_at][ec].reset_index(drop=True)
    pd.testing.assert_frame_equal(ea, eb)


def test_a_zone_touch_needs_a_side(noise_world):
    """zones.lives lists a "?" when price already sat inside the band as the zone became known. It is no touch: the
    first check run counted it as the first touch, "from above" (fixed 30 Sep, plan/work/L13b-notes.md section 4)."""
    w = noise_world
    zl = w.market.zone_life(w.fp)
    t = EV.zone_touches(w)
    assert len(t) and set(t["role"]) <= {"res", "sup"}
    assert any(z and z[0][1] == "?" for z in zl["touches"])                # the made-up market has such zones too
    for _, g in t.groupby("zone_id"):
        assert list(g["touch"]) == list(range(1, len(g) + 1))
    first = {z.zone_id: next((k for k, r, _ in z.touches if r in ("res", "sup")), None) for z in zl.itertuples()}
    for zid, pos in t[t["touch"] == 1].set_index("zone_id")["pos"].items():
        assert pos == first[zid]
    assert (t["side"] == np.where(t["role"] == "res", "high", "low")).all()


@pytest.mark.news
def test_every_belief_runs_on_made_up_candles(noise_world):
    for fam, versions in registry.families().items():
        for b in versions:
            raw = b.rows(noise_world)
            n, _ = engine.measure(b, raw, noise_world.years)
            assert n["overall"]["n"] >= 0 and np.isfinite(n["stat"])


@pytest.mark.news
def test_rupee_trades_are_every_setup_on_its_own(noise_world):
    from machine.beliefs import trades as TR
    t = TR.type1(noise_world)
    comp, _ = TR.compiled(noise_world)
    assert len(t) > 50 and len(t) <= len(comp.signals)
    assert t.loc[t["status"] != "closed", "net_inr"].isna().all()           # open at the end: no money (lesson B6)
    assert set(t["family"]) <= {"equal", "session", "trendline", "swing"}
