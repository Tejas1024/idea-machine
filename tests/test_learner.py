"""L13.1 - the learner: features known at the signal, condition tables, the slice rules, the report's lines.

What the prompt asks: the feature builder passes the cut-off-the-future test · every condition of conditions.md (T7)
and every piece of COMPONENTS.md has a feature · only the slices chosen before the test can become "works best in /
fails in" (section D: Holm, the same sign in 2016-2018 and 2019-2020, a shuffled-label test) · the rest is "looked at,
not tested" · idea #1's report has the lines, with the trades in each part.
Hand-made bars are random walks, not gold; hand-made trades have no edge unless one is planted. Every file goes to
temporary folders (tests/conftest.py).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from machine import config, guards, settings
from machine import trials as trial_files
from machine.blocks import sessions as SES
from machine.data import clock
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import Idea
from machine.learner import conditions as CN
from machine.learner import features as F
from machine.learner import run as LR
from machine.learner import selftest
from machine.learner import slices as SL
from machine.learner.selftest import fake_features, fake_trades
from machine.referee import project, trials, tune
from machine.report import build as B
from tests.test_ideas import fake_m1, pair, type1
from tests.test_report import IDEA, judged_idea, make_form, numbers_in

NEWS = pd.DataFrame({
    "server": pd.to_datetime(["2019-01-10 15:30", "2019-01-15 17:00", "2019-01-16 21:00", "2019-01-22 15:30"]),
    "colour": ["red", "orange", "red", "red"],
    "name": ["Non-Farm Employment Change", "Core CPI m/m", "Fed Announcement", "FOMC Statement"],
    "kind": ["jobs", "inflation", "the Fed", "the Fed"],
    "scheduled": [True, True, False, True]})
SLICES = [dict(name="direction", dimension="direction", how="buy / sell", why="a test"),
          dict(name="session", dimension="session", how="by session", why="a test"),
          dict(name="banana", dimension="place", how="a condition the machine does not have", why="a test")]


# ================================================================================================ fakes

@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return fake_m1()


@pytest.fixture(scope="module")
def market(m1) -> C.Market:
    return C.Market.from_m1(m1, tfs=LR.TFS, ath_before=2000.0)


def features_of(form: Idea, market: C.Market, news=NEWS) -> pd.DataFrame:
    c = C.compile_idea(form, market, trace=True)
    with settings.using(c.resolved):
        return F.build(F.setups_from(c, market), market, fp=settings.fingerprint(c.resolved), form=c.form, news=news)


@pytest.fixture(scope="module")
def feats1(market) -> pd.DataFrame:
    return features_of(Idea.load(type1()), market)


def labelled(trades: pd.DataFrame, name: str, seed: int = 0):
    cond = CN.BY_NAME[name]
    return cond.labels(fake_features(trades.index, seed)), cond.groups


# ================================================================================================================
# 1. the features: nothing after the setup's moment
# ================================================================================================================

def _rows(feats: pd.DataFrame) -> pd.DataFrame:
    return feats.reset_index()


@pytest.mark.parametrize("make", [type1, pair], ids=["type1", "pair"])
def test_features_pass_the_cut_off_the_future_test(m1, make):
    """Cut the 1-minute bars off at a moment: every feature of every setup known by then must be exactly the same
    (R4, lesson B1) - the market columns and the columns about the idea's own steps."""
    form = Idea.load(make())

    def fn(bars):
        return _rows(features_of(form, C.Market.from_m1(bars, tfs=LR.TFS, ath_before=2000.0)))

    n = len(m1)
    guards.assert_no_future_events(fn, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])


def test_the_cut_off_test_catches_a_feature_that_peeks(m1, market, feats1):
    """The same test on a table with one more column that reads the next hour: caught."""
    form = Idea.load(type1())
    high = m1["high"]

    def peeks(bars):
        df = _rows(features_of(form, C.Market.from_m1(bars, tfs=LR.TFS, ath_before=2000.0)))
        h = bars["high"]
        df["next_hour_high"] = [float(h.loc[t:t + pd.Timedelta(minutes=60)].max()) if len(h.loc[t:]) else -1.0
                                for t in df["known_at"]]
        return df

    t = pd.Timestamp(feats1["known_at"].iloc[len(feats1) // 2])
    assert len(high.loc[t:])
    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(peeks, m1, 1, [int(m1.index.searchsorted(t)) + 3])


def test_every_setup_has_its_row_and_every_column(feats1, market):
    form = Idea.load(type1())
    c = C.compile_idea(form, market)
    assert list(feats1.index) == [s.id for s in c.signals] and len(feats1) > 50
    assert list(feats1.columns) == ["known_at", *F.COLUMNS]
    for col in ("adr_used", "recent_range", "hedge_adr", "room_adr", "grab_depth_adr", "gap_size_adr"):
        assert (feats1[col].dropna() >= 0).all() and feats1[col].notna().any(), col
    assert set(feats1["session"]) <= {"Asia", "London", "New York", "in between"}
    assert set(feats1["level_kind"].dropna()) <= {"equal", "session", "trendline", "swing"}
    assert feats1["break_to_gap"].dropna().between(-36, 6).all()      # the gap: from the take to 6 after the break


def test_market_columns_against_a_plain_count(feats1, market, m1):
    """A few columns counted again by hand from the bars, for setups spread over the walk."""
    m5 = market.bars("5m")
    ruler = market.ruler("5m", settings.fingerprint())
    sig = {s.id: s for s in C.compile_idea(Idea.load(type1()), market).signals}
    seen = 0
    for sid, r in list(feats1.iterrows())[::7]:
        t = pd.Timestamp(r["known_at"])
        i = market.closed_by("5m", t)
        if not np.isfinite(ruler[i]):
            assert pd.isna(r["adr_used"]) and pd.isna(r["adr"])
            continue
        day = m5.loc[m5.index[i].normalize():m5.index[i]]
        assert r["adr_used"] == pytest.approx((day["high"].max() - day["low"].min()) / ruler[i])
        hour = m5.iloc[i - 11:i + 1]
        assert r["recent_range"] == pytest.approx((hour["high"].max() - hour["low"].min()) / ruler[i])
        s = sig[sid]
        assert r["hedge_adr"] == pytest.approx(abs(s.entry_price - s.stop) / ruler[i])
        assert r["reward"] == pytest.approx(abs(r["room_adr"] / r["hedge_adr"]))
        assert r["ist_hour"] == clock.to_ist(t).hour
        assert r["weekday"] == (t - pd.Timedelta(minutes=1)).day_name()
        seen += 1
    assert seen >= 5


def test_the_session_is_the_session_filters_session(market):
    """One session rule: a setup's session here is exactly what the form's session filter keeps (the old project's two
    session rules agreed on 48% of setups)."""
    m5 = market.bars("5m")
    times = m5.index[288 * 12:288 * 14] + pd.Timedelta(minutes=5)             # every 5-minute close of two days
    px = m5["close"].to_numpy()[288 * 12:288 * 14]
    setups = [F.Setup(f"s{k}", t, "buy", float(p), float(p) - 1, float(p) + 1)
              for k, (t, p) in enumerate(zip(times, px))]
    got = F.build(setups, market, one_minute=False)["session"].to_numpy()
    run = C._Run(market, None, "buy", "")
    for name in SES.names():
        keep = np.array([C.SessionFilter.keep(run, C.SessionParams(only=(name,)), {}, t) for t in times])
        assert (keep == (got == name)).all(), name
    assert {"Asia", "London", "New York", "in between"} == set(got)
    london = [clock.to_ist(t).strftime("%H:%M") for t, s in zip(times, got) if s == "London"]
    assert min(london) == "12:35" and max(london) == "15:30"          # the candle that closes at 12:30 is not London


def test_the_next_news_is_only_a_scheduled_one_and_the_last_has_happened():
    t = pd.DatetimeIndex(["2019-01-15 16:00", "2019-01-16 20:00", "2019-01-16 21:00", "2019-01-16 21:05"])
    n = F.news_at(NEWS, t)
    assert list(n["news_next_min"]) == [60.0, 8370.0, 8310.0, 8305.0]      # never the unscheduled 16 Jan 21:00
    assert list(n["news_next_kind"]) == ["inflation", "the Fed", "the Fed", "the Fed"]
    assert n["news_last_min"].iloc[0] == 7230.0 and n["news_last_kind"].iloc[0] == "jobs"
    assert n["news_last_min"].iloc[2] == 0.0 and n["news_last_kind"].iloc[2] == "the Fed"   # out at that minute
    assert n["news_last_min"].iloc[3] == 5.0 and n["news_last_colour"].iloc[3] == "red"
    # an unscheduled release later changes nothing known now; a scheduled one does (it is on the calendar)
    more = pd.concat([NEWS, pd.DataFrame({"server": pd.to_datetime(["2019-01-15 16:30"]), "colour": ["red"],
                                          "name": ["Fed Announcement"], "kind": ["the Fed"], "scheduled": [False]})])
    more = more.sort_values("server").reset_index(drop=True)
    assert F.news_at(more, t[:1]).equals(n.iloc[:1])
    more.loc[more["name"] == "Fed Announcement", "scheduled"] = True
    assert F.news_at(more, t[:1])["news_next_min"].iloc[0] == 30.0
    empty = F.news_at(None, t)
    assert empty["news_next_min"].isna().all() and empty["news_last_kind"].isna().all()


@pytest.mark.parametrize("name, kind", [
    ("Non-Farm Employment Change", "jobs"), ("Unemployment Claims", "jobs"), ("Average Hourly Earnings m/m", "jobs"),
    ("ADP Non-Farm Employment Change", "jobs"), ("Core CPI m/m", "inflation"),
    ("Core PCE Price Index m/m", "inflation"),
    ("FOMC Statement", "the Fed"), ("Federal Funds Rate", "the Fed"), ("Fed Chair Powell Speaks", "the Fed"),
    ("FOMC Member Bullard Speaks", "the Fed"), ("Fed Announcement", "the Fed"), ("ISM Manufacturing PMI", "other news"),
    ("Crude Oil Inventories", "other news"), ("President Trump Speaks", "other news")])
def test_the_kind_of_a_forex_factory_row(name, kind):
    assert F.ff_kind(name) == kind


def test_todays_adr_against_its_last_12_months():
    days = pd.bdate_range("2019-01-01", periods=200).as_unit("ns").asi8
    ruler = np.r_[np.full(10, np.nan), np.arange(190, dtype=float)]           # it rises every day
    share = F._vol_share(days, ruler)
    assert np.isnan(share[:10 + F.VOL_MIN_DAYS]).all()                        # too few days with an ADR before it
    assert share[10 + F.VOL_MIN_DAYS] == 1.0 and (share[10 + F.VOL_MIN_DAYS:] == 1.0).all()   # the highest so far
    assert F._vol_share(days, ruler[::-1].copy())[100] == 0.0                 # falling: lower than every day before


def test_the_pair_ideas_own_steps(market):
    """The columns from what the idea's steps found (the pair form: equal highs taken, equal lows the target)."""
    form = Idea.load(pair())
    c = C.compile_idea(form, market, trace=True)
    f = features_of(form, market)
    assert len(f) == len(c.signals) > 5
    assert set(f["level_kind"]) == {"equal"} and set(f["target_kind"]) == {"equal"}
    assert set(f["level_chart"]) <= {"5m", "1h", "1d"} and (f["level_touches"] >= 2).all()
    assert set(f["level_day"]) <= {"today", "yesterday", "older"} and (f["level_age_days"] > 0).all()
    ok = f["adr"].notna()
    assert (f.loc[ok, "room_adr"] == f.loc[ok, "equal_target_adr"]).mean() > 0.5     # the target is the pair below
    facts = {r["signal"]: r["facts"] for r in c.trace if r.get("signal")}
    sid = f.index[0]
    i = market.closed_by("5m", f.loc[sid, "known_at"])
    r = market.ruler("5m", settings.fingerprint(c.resolved))[i]
    assert f.loc[sid, "gap_size_adr"] == pytest.approx(facts[sid]["gap"]["size"] / r)
    assert f.loc[sid, "grab_depth_adr"] == pytest.approx(
        abs(facts[sid]["brk"]["extreme"] - facts[sid]["sweep"]["price"]) / r)


# ================================================================================================================
# 2. the conditions: T7's list and COMPONENTS.md, each with a feature
# ================================================================================================================

def test_every_condition_gives_only_its_own_parts(feats1):
    for cond in CN.ALL:
        lab = cond.labels(feats1)
        assert len(lab) == len(feats1) and set(lab.dropna()) <= set(cond.groups), cond.name
        assert cond.dimension in ("direction", "session", "news", "higher_timeframe_trend", "volatility", "market_type",
                                  "place", "time"), cond.name
    known = [c.name for c in CN.ALL if CN.BY_NAME[c.name].labels(feats1).notna().any()]
    assert len(known) >= 40 and "volatility" not in known      # 25 days of bars: no 12 months of ADR yet
    rand = fake_features([f"s{k}" for k in range(300)])
    for cond in CN.ALL:                                         # every part of every condition can be reached
        lab = cond.labels(rand)
        assert lab.notna().all() and set(lab) <= set(cond.groups), cond.name
    with pytest.raises(ValueError, match="not in its list"):
        CN.Condition("x", "x", "time", "a test", "a test", ("a",), lambda f: ["b"] * len(f)).labels(feats1)


def test_every_condition_of_t7_has_a_condition_here():
    text = (config.WORK_DIR / "conditions.md").read_text(encoding="utf-8")
    his = sorted(set(re.findall(r"^\| (C\d+) \|", text, flags=re.M)), key=lambda c: int(c[1:]))
    assert his == [f"C{k}" for k in range(1, 16)]
    sources = " ; ".join(c.source for c in CN.ALL)
    for c in his:
        assert re.search(rf"conditions\.md {c}\b", sources), f"{c} has no condition"
    for name in ("direction", "hour", "weekday", "news_near", "news_kind", "adr_used", "recent_range", "liq_target",
                 "liq_hedge", "bias_day", "bias_week", "session"):       # the prompt's own list
        assert name in CN.BY_NAME


def test_every_piece_of_components_md_is_listed_with_its_feature():
    text = (config.STRATEGY_DIR / "COMPONENTS.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| \*\*(.+?)\*\* \|", text, flags=re.M)
    assert [int(n) for n, _ in rows] == list(range(1, 32)) == sorted(CN.COMPONENTS)
    for n, title in rows:
        mine, conds, note = CN.COMPONENTS[int(n)]
        assert mine == title, n
        assert conds or note.startswith("constant"), f"#{n} is covered by nothing and is not said to be constant"
        for c in conds:
            assert c in CN.BY_NAME and CN.BY_NAME[c].labels  # exists
    assert sorted(n for n, (_, c, _) in CN.COMPONENTS.items() if not c) == [7, 22, 23]


def test_idea_1s_eight_slices_each_name_a_condition():
    form = Idea.load(config.ROOT / "plan" / "work" / "ideas" / "idea-001" / "form.json")
    assert len(form.slices) == 8
    for sl in form.slices:
        cond = CN.for_slice(sl)
        assert cond is not None and cond.dimension == sl.dimension, sl.name
    assert CN.for_slice("banana") is None                       # never guessed (R13)
    assert "session: session (India time) - Asia / London / New York / in between" in CN.describe()


# ================================================================================================================
# 3. the slice rules
# ================================================================================================================

def test_the_slice_rules_are_pinned_and_in_easy_english():
    """Changing a rule after seeing a result shows in git: the fingerprint is pinned here."""
    assert SL.RULES_SHA256 == PINNED_RULES, SL.RULES
    assert B.check_words(SL.RULES) == []
    for words in ("chosen in the idea's form BEFORE the test (at most 8)",
                  "at least 30 trades in it and 30 in the rest",
                  "drawn again 2,000 times", "under 0.62 in 100", "2016-2018 and in 2019-2020",
                  "shuffled among the trades of the same year 2,000 times", "with the same correction as in check 1",
                  "looked at, not tested", "They decide nothing here"):
        assert words in SL.RULES, words


PINNED_RULES = "d614a249e638a487f44ed9b3394c71f3821421175bd205d0ebf512c88bf2c5c0"


def test_a_table_counts_every_trade_once():
    trades = fake_trades(600, seed=3)
    lab, groups = labelled(trades, "session", seed=3)
    lab.iloc[:7] = None                                          # not known on 7 trades
    t = SL.table(trades, lab, groups, seed=1)
    assert t["not_known"] == 7 and t["trades"] == 593 == sum(g["trades"] for g in t["groups"])
    v = trades["net_inr"].to_numpy()
    for g in t["groups"]:
        m = (lab == g["group"]).to_numpy()
        assert g["per_trade"] == pytest.approx(v[m].mean(), abs=0.01)
        assert g["net_inr"] == pytest.approx(v[m].sum(), abs=0.01)
        assert g["interval"][0] < g["per_trade"] < g["interval"][1]
        rest = (lab.notna() & (lab != g["group"])).to_numpy()
        assert g["rest_trades"] == rest.sum() and g["gap"] == pytest.approx(v[m].mean() - v[rest].mean(), abs=0.01)
        assert sum(h["trades"] for h in g["halves"]) == g["trades"] and g["enough"]
    assert [h["years"] for h in t["groups"][0]["halves"]] == ["2016-2018", "2019-2020"]


def test_the_interval_is_made_from_whole_weeks():
    weeks = np.repeat(np.arange(50), 4)
    same = np.tile([100.0, -50.0, -50.0, 0.0], 50)               # every week the same: the week total never moves
    assert SL.interval(same, weeks, seed=1) == [0.0, 0.0]
    by_week = np.repeat(np.where(np.arange(50) % 2 == 0, 200.0, -200.0), 4)   # the luck is in the WEEKS
    lo, hi = SL.interval(by_week, weeks, seed=1)
    trade_by_trade = SL.interval(by_week, np.arange(200), seed=1)
    assert hi - lo > 1.7 * (trade_by_trade[1] - trade_by_trade[0])            # trades of a week stay together
    assert SL.interval([5.0], [1], seed=1) is None and SL.interval([5.0, 6.0], [1, 1], seed=1) is None


_eight = selftest.eight          # idea #1's eight slices on made-up features: name -> (labels, parts)


def test_a_planted_london_edge_is_found_and_nothing_else():
    """Rs 300 a trade added to the London trades only: "works best in London"; the seven other slices say nothing."""
    trades = fake_trades(600, seed=11)
    f, sl = _eight(trades, 11)
    trades.loc[f["session"] == "London", "net_inr"] += 300.0
    out = SL.judge({n: SL.test(trades, lab, groups, seed=SL.seed_of(trades.index, n))
                    for n, (lab, groups) in sl.items()})
    s = out["session"]
    assert s["claim"] == SL.WORKS_BEST and s["part"] == "London" and s["gap"] > 150
    assert s["holm"]["passed"] and s["same_way"] and s["holm_shuffle"]["passed"]
    assert s["p"] < 0.00625 and s["p_shuffle"] < 0.00625
    assert 0 < s["could_see"] < s["gap"]                    # the gap it could see is smaller than the one it found
    assert [n for n, r in out.items() if r["claim"]] == ["session"]
    quiet = SL.test(fake_trades(600, seed=11), *sl["session"], seed=1)       # the same trades before the planting
    assert quiet["could_see"] == pytest.approx(s["could_see"], rel=0.35) and 80 < quiet["could_see"] < 400
    trades2 = fake_trades(600, seed=11)
    trades2.loc[f["session"] == "New York", "net_inr"] -= 300.0
    out2 = SL.judge({n: SL.test(trades2, lab, groups, seed=1) for n, (lab, groups) in sl.items()})
    assert out2["session"]["claim"] == SL.FAILS and out2["session"]["part"] == "New York"


def test_pure_noise_gives_almost_no_finding():
    """The slice rules' own exam, small (python -m machine.learner.selftest prints the big one: 400 no-edge ideas,
    a finding on 3.2 in 100). 60 made-up ideas with no edge, 8 slices each: the rules allow a finding on 5 ideas in
    100, so 60 ideas may give at most 6 (3 expected); and a single slice's two tests, with no correction, fire about
    5 times in 100 each - the first version of the luck test fired 6.3 times and let 7 no-edge ideas in 100 through."""
    n = selftest.noise(sets=60, trades=600, n_boot=600, n_shuffle=600)
    assert n["finding"] <= 6 and n["luck_with_holm"] <= 8 and n["shuffle_with_holm"] <= 8
    assert n["tests"] == 480 and n["luck_under_5"] <= 38 and n["shuffle_under_5"] <= 38    # 24 expected of 480


def test_a_planted_gap_is_found_when_it_is_big_enough():
    """London earns Rs 300 a trade more: found nearly every time; Rs 100 more: mostly not seen (the test says how
    small a gap it can see - 600 trades cannot show a Rs 100 gap)."""
    big = selftest.planted(300, sets=12, n_boot=600, n_shuffle=600)
    assert big["found"] >= 11 and big["another_slice_claimed"] <= 2
    small = selftest.planted(100, sets=12, n_boot=600, n_shuffle=600)
    assert small["found"] <= 6


def test_an_edge_in_one_half_only_is_not_a_finding():
    """Big in 2019-2020, the other way in 2016-2018: check 2 says no."""
    trades = fake_trades(800, seed=5)
    f, sl = _eight(trades, 5)
    late = pd.DatetimeIndex(trades["exit_time"]).year >= 2019
    london = (f["session"] == "London").to_numpy()
    trades.loc[london & late, "net_inr"] += 900.0
    trades.loc[london & ~late, "net_inr"] -= 120.0
    t = SL.test(trades, *sl["session"], seed=2)
    assert t["part"] == "London" and t["gap"] > 0 and not t["same_way"]
    assert [h["gap"] > 0 for h in t["halves"]] == [False, True]
    assert SL.judge({"session": t})["session"]["claim"] is None


def test_a_good_year_is_not_a_finding():
    """The label is just "the trade closed in 2020" and 2020 was a good year (like idea #1's 2016-2020). Shuffling the
    labels among the same year's trades cannot move them, so check 3 says no - and check 2 too (no such trade in
    2016-2018)."""
    trades = fake_trades(800, seed=6)
    y2020 = pd.DatetimeIndex(trades["exit_time"]).year == 2020
    trades.loc[y2020, "net_inr"] += 400.0
    lab = pd.Series(np.where(y2020, "high", "normal"), index=trades.index, dtype=object)
    t = SL.test(trades, lab, ("low", "normal", "high"), seed=3)
    assert t["part"] in ("high", "normal") and t["p"] < 0.01            # the luck test alone would call it
    assert t["p_shuffle"] == 1.0 and not t["same_way"]
    out = SL.judge({"volatility": t})["volatility"]
    assert out["holm"]["passed"] and not out["holm_shuffle"]["passed"] and out["claim"] is None


def test_too_few_trades_is_said_not_tested():
    trades = fake_trades(100, seed=7)
    lab = pd.Series(["Friday"] * 12 + ["other days"] * 88, index=trades.index, dtype=object)
    t = SL.test(trades, lab, ("Friday", "other days"), seed=1)
    assert t == {"tested": False, "why": SL.FEW, "p": 1.0, "p_shuffle": 1.0, "eligible": []}
    tab = SL.table(trades, lab, ("Friday", "other days"), seed=1)
    assert [g["enough"] for g in tab["groups"]] == [False, False]
    out = SL.judge({"friday": t})["friday"]
    assert out["claim"] is None and not out["passed"] and out["holm"]["needs_under"] == config.P_LIMIT


def test_holms_correction_for_the_slices_chosen():
    ps = {"a": 0.02, "b": 0.001, "c": 0.006, "d": 0.0001, "e": 1.0, "f": 0.5, "g": 0.3, "h": 0.2}
    h = SL.holm(ps)
    assert list(h) == list(ps)                                                   # the form's order kept
    assert [h[k]["place"] for k in "dbca"] == [1, 2, 3, 4]
    assert h["d"]["needs_under"] == pytest.approx(0.05 / 8) and h["b"]["needs_under"] == pytest.approx(0.05 / 7)
    assert h["d"]["passed"] and h["b"]["passed"] and h["c"]["passed"]           # 0.006 < 0.05 / 6
    assert not h["a"]["passed"]                                                  # 0.02 > 0.05 / 5 = 0.01
    assert not any(h[k]["passed"] for k in "efgh")
    two = SL.holm({"a": 0.03, "b": 0.011})                                       # 0.011 < 0.025, then 0.03 < 0.05
    assert two["b"]["passed"] and two["a"]["passed"]
    stop = SL.holm({"a": 0.04, "b": 0.03})                                       # the first fails: none passes
    assert not stop["b"]["passed"] and not stop["a"]["passed"]
    assert SL.holm({"only": 0.04})["only"]["passed"]                             # one slice chosen: no correction


# ================================================================================================================
# 4. the analysis: only the chosen slices can be a finding; the rest is looked at
# ================================================================================================================

def test_only_the_slices_chosen_before_can_become_a_finding():
    """The same planted London edge: with `session` chosen in the form it is a finding; with it not chosen it is one
    of the conditions looked at, and nothing is claimed."""
    trades = fake_trades(600, seed=11)
    f = fake_features(trades.index, 11)
    trades.loc[f["session"] == "London", "net_inr"] += 300.0
    chosen = make_form("idea-960")                                   # idea #1's 8 slices: session is one of them
    r = LR.analyse(chosen, trades, f)
    assert [(c["slice"], c["claim"], c["part"]) for c in r["claims"]] == [("session", SL.WORKS_BEST, "London")]
    assert r["counts"] == {**r["counts"], "slices_chosen": 8, "slices_measured": 8, "slices_tested": 8,
                           "slices_passed": 1, "looked_at": len(CN.ALL) - 8, "does_not_apply": 0}
    line = LR.works_line(r)
    assert line.startswith("Works best in / fails in (2016–2020, the 8 slices chosen before the test): WORKS BEST "
                           "when session (India time) is 'London' (₹")
    d = json.loads(chosen.to_json())
    d["slices"] = [s for s in d["slices"] if s["name"] != "session"]
    r2 = LR.analyse(Idea.load(d), trades, f)
    assert r2["claims"] == [] and "session" in [c["name"] for c in r2["looked_at"]]
    assert all("test" not in c for c in r2["looked_at"])              # looked at: a table, never a test
    assert r2["counts"]["looked_at"] == len(CN.ALL) - 7
    line = LR.works_line(r2)
    assert "nothing passed the 3 checks - with these 600 trades a part must differ from the rest by about ₹" in line
    assert "to be told from luck" in line                    # "nothing passed" never reads as "no difference"
    none = Idea.load({**d, "slices": []})
    r3 = LR.analyse(none, trades, f)
    assert r3["claims"] == [] and r3["counts"]["looked_at"] == len(CN.ALL)
    assert LR.works_line(r3).startswith("Works best in / fails in: no slice was chosen before the test")


def test_a_slice_the_machine_cannot_measure_is_said_and_still_counts_in_the_correction():
    trades = fake_trades(300, seed=2)
    f = fake_features(trades.index, 2)
    form = Idea.load(type1(id="idea-961", slices=SLICES))
    r = LR.analyse(form, trades, f)
    banana = r["slices"][2]
    assert not banana["measured"] and "no condition named 'banana'" in banana["why_not"]
    assert banana["test"] == {**banana["test"], "tested": False, "p": 1.0, "passed": False, "claim": None}
    assert r["slices"][0]["test"]["holm"]["needs_under"] in (0.05 / 3, 0.05 / 2)   # 3 slices chosen, not 2
    assert any("not measured" in ln for ln in LR.slice_lines(r))
    txt = LR.text(r, "idea-961", "v1")
    assert "LOOKED AT, NOT TESTED" in txt and "THE 3 SLICES CHOSEN BEFORE THE TEST" in txt


def test_a_condition_with_one_part_only_is_not_counted_as_looked_at():
    """Idea #1 always takes equal highs / lows: the kind of level taken splits nothing, so it is not one of the
    conditions 'looked at' - it is listed as not on this idea, with the reason."""
    trades = fake_trades(300, seed=8)
    f = fake_features(trades.index, 8)
    f["level_kind"] = "equal"
    r = LR.analyse(make_form("idea-964"), trades, f)
    assert "level_kind" not in [c["name"] for c in r["looked_at"]]
    why = next(x["why"] for x in r["does_not_apply"] if x["name"] == "level_kind")
    assert why == "every trade is 'equal highs / lows' - it splits nothing on this idea"
    assert r["counts"]["looked_at"] == len(CN.ALL) - 8 - 1


def test_the_analysis_refuses_other_years_and_trades_without_a_setup():
    trades = fake_trades(100, seed=1)
    f = fake_features(trades.index, 1)
    form = make_form("idea-962")
    with pytest.raises(LR.LearnError, match="have no setup"):
        LR.analyse(form, trades, f.iloc[5:])
    late = trades.copy()
    late.iloc[0, late.columns.get_loc("exit_time")] = pd.Timestamp("2021-02-01")
    with pytest.raises(LR.LearnError, match="tune years 2016-2020 only"):
        LR.analyse(form, late, f)


def test_the_hidden_years_are_shown_next_to_a_slice_and_decide_nothing():
    trades = fake_trades(600, seed=11)
    ids = list(trades.index) + [f"c{k}" for k in range(200)]
    f = fake_features(ids, 11)
    check = pd.DataFrame({"net_inr": np.full(200, -500.0), "exit_time": pd.Timestamp("2022-03-01")},
                         index=pd.Index(ids[600:], name="signal"))
    form = make_form("idea-963")
    a = LR.analyse(form, trades, f, check_trades=check)
    b = LR.analyse(form, trades, f)
    shown = a["slices"][1]["check_years"]
    assert sum(x["trades"] for x in shown) == 200 and {x["per_trade"] for x in shown} == {-500.0}
    assert a["check_years_shown"] and not b["check_years_shown"]
    for x, y in zip(a["slices"], b["slices"]):
        assert x["test"] == y["test"] and x["groups"] == y["groups"]


# ================================================================================================================
# 5. the whole path on hand-made bars: form -> tune -> learner -> one trial file
# ================================================================================================================

@pytest.fixture
def learned(m1, market, monkeypatch):
    monkeypatch.setattr(config, "SLICE_MIN_TRADES", 5)               # 25 days of a random walk: few trades
    form = Idea.load(type1(id="idea-950", slices=SLICES, status="ANSWERED"))
    store.create(form)
    c = C.compile_idea(form, market)
    t = tune.run_versions({c.version_id: c.signals}, tf="5m", m1=m1, readings={c.version_id: {}},
                          about={c.version_id: "the form as written"})
    return form, t, LR.learn_with(form, tune.latest("idea-950"), market, news=NEWS, log=lambda *_: None)


def test_the_learner_on_hand_made_bars(learned, _trials_in_tmp):
    form, t, L = learned
    r = L.result
    closed = [row for row in t.numbers["trades_table"]["rows"] if row[7] == "closed"]
    assert r["trades"] == len(closed) > 20 and L.version_id == t.chosen
    assert [s["name"] for s in r["slices"]] == ["direction", "session", "banana"]
    for s in r["slices"][:2]:
        assert s["trades"] + s["not_known"] == r["trades"] == sum(g["trades"] for g in s["groups"]) + s["not_known"]
    total = sum(row[13] for row in closed)
    assert sum(g["net_inr"] for g in r["slices"][0]["groups"]) == pytest.approx(total, abs=0.05)   # the referee's ₹
    assert r["slices"][0]["test"]["tested"] and r["counts"]["looked_at"] >= 25
    assert "volatility" in [x["name"] for x in r["does_not_apply"]]      # no 12 months of bars on a 25-day walk
    placed = ([s["condition"] for s in r["slices"] if s["condition"]] + [c["name"] for c in r["looked_at"]]
              + [x["name"] for x in r["does_not_apply"]])
    assert sorted(placed) == sorted(CN.names())                           # every condition in exactly one place
    body = json.loads(L.trial_path.read_text(encoding="utf-8"))
    assert body["kind"] == config.LEARNER_KIND == "learner" and body["version_id"] == t.chosen
    assert body["settings"]["rules_sha256"] == SL.RULES_SHA256 and body["result"]["tune_trial"]
    assert LR.latest("idea-950", t.chosen)["trial_id"] == body["trial_id"] and LR.latest("idea-950", "v-x") is None
    assert "THE LEARNER · idea-950" in L.text() and "Trial file:" in L.text()


def test_a_learner_run_is_not_a_try_and_not_an_answer(learned):
    form, t, L = learned
    c = trials.count("idea-950")
    assert c.idea_versions == [t.chosen]                                 # still one version
    assert project.summary()["answers"] == 0                             # the learner judges nothing


def test_the_learner_refuses_signals_that_are_not_the_tunes(learned, market):
    form, t, _ = learned
    other = json.loads(json.dumps(tune.latest("idea-950")))
    other["result"]["versions"][t.chosen]["signals_sha256"] = "0" * 64
    with pytest.raises(LR.LearnError, match="are not the tune's"):
        LR.learn_with(form, other, market, news=NEWS, record=False, log=lambda *_: None)


# ================================================================================================================
# 6. the report has the lines
# ================================================================================================================

def with_learner(idea: str = IDEA, planted: bool = False, seed: int = 4):
    """A judged idea (hand-made bars) and a learner file for it. The learner's trades are hand-made too (600 trades
    like idea #1's): the report reads the file, whatever made it."""
    made = judged_idea(idea)
    tf = tune.latest(idea)
    trades = fake_trades(600, seed=seed)
    f = fake_features(trades.index, seed)
    if planted:
        trades.loc[f["session"] == "London", "net_inr"] += 300.0
    r = LR.analyse(made["form"], trades, f)
    r["tune_trial"] = tf["trial_id"]
    path = trial_files.record(kind=config.LEARNER_KIND, idea_id=idea, version_id=tf["result"]["chosen"],
                              parent_idea=None, settings={"rules_sha256": SL.RULES_SHA256},
                              data={"prices": "hand-made"}, result=r)
    return made, r, path


def test_the_report_has_the_lines_with_the_trades_in_each_part():
    _, r, _ = with_learner()
    b = B.build(IDEA)
    text = b.text
    assert ("Works best in / fails in (2016–2020, the 8 slices chosen before the test): nothing passed the 3 checks"
            in text)
    assert f"slices chosen before: 8 of 8 · looked at, not tested: {len(CN.ALL) - 8}" in text
    assert "Trades in each part of each slice, 2016–2020 (* = too few trades to test that part):" in text
    for s in r["slices"]:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"· {s['label']}:"))
        for g in s["groups"]:
            assert (f"{g['group']} {g['trades']}" in line) == bool(g["trades"])   # a part with 0 trades: page only
    assert "Works best in / fails in: not measured yet" not in text          # (the model's lines may be: L13.2)
    assert len(text) <= 4096 and B.check_words(text) == []
    page = b.page_path.read_text(encoding="utf-8")
    assert B.check_words(B._visible(page)) == []
    for words in ("Works best in / fails in — the 8 slices chosen before any result", "Looked at, not tested —",
                  "THE SLICE RULES", "when the weeks are drawn again", "2016–2018: ₹ a trade (trades)",
                  "the part that stands out most"):
        assert words in page, words
    assert page.count("<details>") == len(CN.ALL) - 8


def test_a_finding_is_written_with_its_rupees_and_every_rupee_is_in_the_files():
    _, r, _ = with_learner(planted=True, seed=11)
    b = B.build(IDEA)
    line = next(ln for ln in b.text.splitlines() if ln.startswith("Works best in / fails in"))
    c = r["claims"][0]
    assert (f"WORKS BEST when session (India time) is 'London' ({B.inr(c['per_trade'])} a trade on {c['trades']} "
            "trades") in line
    assert "→ works best in London" in b.text
    f = b.facts
    allowed = {B.inr(x) for x in numbers_in(f.judge["result"]) + numbers_in(f.tune["result"])
               + numbers_in(f.learner["result"])} | {"₹40", "₹0", "₹100", "₹10", "₹25", "₹50"}
    for amount in re.findall(r"-?₹[\d,]+", b.text):
        assert amount in allowed, amount


@pytest.mark.parametrize("x, text", [(-0.3, "₹0"), (0.4, "₹0"), (-0.6, "-₹1"), (-1234.4, "-₹1,234"), (790.2, "₹790"),
                                     (None, "no trades")])
def test_rupees_never_show_minus_zero(x, text):
    """Idea #1's Friday interval ended at -₹0.3 and printed "₹-0" - now ₹0 (the report page and the learner's text)."""
    assert B.inr(x) == text and LR._inr(x) == text


def test_the_report_follows_the_learners_file():
    """Nothing is counted again: change a number in the learner's file and the report shows it."""
    _, _, path = with_learner()
    body = json.loads(path.read_text(encoding="utf-8"))
    body["result"]["slices"][0]["groups"][0]["trades"] = 4321
    body["result"]["counts"]["looked_at"] = 77
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    text = B.telegram_text(B.facts(IDEA))
    assert "buy 4321" in text and "looked at, not tested: 77" in text


def test_no_learner_file_or_old_slice_rules_say_not_measured_yet():
    judged_idea("idea-905")
    assert "Works best in / fails in: not measured yet (the learner has not run" in B.build("idea-905").text
    _, _, path = with_learner("idea-906")
    body = json.loads(path.read_text(encoding="utf-8"))
    body["settings"]["rules_sha256"] = "0" * 64
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    text = B.build("idea-906").text
    assert "not measured yet" in text and "The slice rules changed after the learner ran" in text
    assert "looked at, not tested: 0" in text


def test_a_long_report_keeps_the_part_counts_on_the_page(monkeypatch):
    with_learner("idea-907")
    f = B.facts("idea-907")
    long_line = ["· a very long slice line " + "x" * 400] * 8
    monkeypatch.setattr(B.learner, "slice_lines", lambda res: long_line)
    text = B.telegram_text(f)
    assert len(text) <= 4096 and "on the report page (report.html)" in text and "very long slice line" not in text


# ================================================================================================================
# 7. real prices (skipped without the data)
# ================================================================================================================

@pytest.mark.data
@pytest.mark.news
def test_his_news_is_forex_factorys_red_and_orange_with_the_unscheduled_ones_marked():
    news = F.his_news("2016", "2023")
    assert 4700 < len(news) < 4900 and set(news["colour"]) == {"red", "orange"}
    assert news["server"].is_monotonic_increasing and news["server"].dt.year.between(2016, 2023).all()
    off = news[~news["scheduled"]]
    assert 17 <= len(off) <= 25 and (off["kind"] == "the Fed").all() and off["server"].dt.year.min() == 2020
    nfp = news[news["name"] == "Non-Farm Employment Change"]
    assert len(nfp) == 96 and set(nfp["kind"]) == {"jobs"} and nfp["scheduled"].all()
    assert set(clock.to_ist(nfp["server"]).dt.strftime("%H:%M")) == {"18:00", "19:00"}     # 8:30 New York
    with pytest.raises(guards.SealedYearsError):
        F.his_news("2016", "2024")


@pytest.mark.data
@pytest.mark.news
def test_idea_1s_features_on_real_prices_do_not_peek():
    """Idea #1's chosen reading on real prices (Feb-Mar 2019, the 1-minute lines too): cut the bars off and every
    feature of every setup known by then is the same."""
    from machine.data.bars import load_bars
    from tests import idea001_form
    m1 = load_bars("1m", "2019-02-04", "2019-03-29")
    form = idea001_form.the_form()
    news = F.his_news("2019", "2019")
    rd = {"U1": "A", "U2": "A", "U3": "B"}

    def fn(bars):
        mk = C.Market.from_m1(bars, tfs=LR.TFS)
        c = C.compile_idea(form, mk, readings=rd, trace=True)
        with settings.using(c.resolved):
            return _rows(F.build(F.setups_from(c, mk), mk, fp=settings.fingerprint(c.resolved), form=c.form, news=news))

    full = fn(m1)
    assert len(full) >= 10 and set(full["level_kind"]) == {"equal"} and full["line_against"].notna().all()
    assert full["news_next_min"].notna().all() and (full["news_next_min"] >= 0).all()
    n = len(m1)
    guards.assert_no_future_events(fn, m1, 1, [n // 2, (4 * n) // 5])
