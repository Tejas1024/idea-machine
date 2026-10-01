"""The liquidity blocks (L6.2): each on hand-made candles with a known answer, and each through the cut-off-the-future
test (likhith.md R4) - on fake bars everywhere, and on real 2019 bars when the price data is here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import guards, settings
from machine.blocks import equal_levels as EQ
from machine.blocks import liquidity as LQ
from machine.blocks import sessions as SES
from machine.data.clock import to_ist
from tests.block_fakes import START, bars_from, random_bars

M5 = pd.Timedelta(minutes=5)
BASE = (100.7, 100.9, 100.6, 100.7)          # a quiet candle: no swing, a dip of only 0.4 from 101.0


def t(i: int, step=M5) -> pd.Timestamp:
    return START + i * step


def candles(n: int, points: dict[int, tuple]) -> list[tuple]:
    """n quiet candles, with some replaced: {bar: (high, low)}."""
    rows = [BASE] * n
    for i, (hi, lo) in points.items():
        rows[i] = (BASE[0], hi, lo, BASE[3])
    return rows


def mirror(rows: list[tuple]) -> list[tuple]:
    """The same candles upside down (price -> 200 - price): highs become lows."""
    return [(200 - o, 200 - lo, 200 - h, 200 - c) for o, h, lo, c in rows]


RULER = 10.0                                  # a day's range of $10: the dip must be at least 0.08 x 10 = $0.80
DIP = (100.9, 99.0)                           # a candle that dips to 99.0: 2.0 below the pair


def eq(rows, tf="5m", **kw):
    bars = bars_from(rows, {"5m": 5, "1h": 60}[tf])
    return EQ.equal_levels(bars, tf, ruler=np.full(len(bars), RULER), **kw)


# ---- equal highs / lows (#88, #89, #90) ---------------------------------------------------------------------------

def test_a_clean_pair_known_when_the_second_touch_is_known():
    e = eq(candles(14, {3: (101.0, 100.6), 6: DIP, 9: (100.97, 100.6)}))
    highs = e[e.side == "high"]
    assert len(highs) == 1
    r = highs.iloc[0]
    assert (r.price, r.first_bar, r.touch_bar, r.touches) == (101.0, t(3), t(9), 2)
    assert r.known_at == t(11) + M5                     # 2 candles after the 2nd touch, at that candle's close
    assert r.gap_usd == pytest.approx(0.03) and not r.exact
    assert r.bars_apart == 6 and r.dip == pytest.approx(0.2)


@pytest.mark.parametrize("second, is_pair", [
    (101.0, True),        # exact - "exact strongest"
    (100.94, True),       # 6 cents below: the edge of "one or two pips"
    (100.91, False),      # 9 cents: "not a very good equal low" (#89 D)
    (101.01, False),      # above the first: "second low should not be lower than the first low" (mirror, #88)
])
def test_how_equal_is_equal_in_dollars(second, is_pair):
    e = eq(candles(14, {3: (101.0, 100.6), 6: DIP, 9: (second, 100.6)}))
    assert (len(e[e.side == "high"]) == 1) == is_pair


def test_nothing_may_trade_past_the_first_between_them():
    # #88 A: "when something comes between the lows, then that is like already swept liquidity"
    e = eq(candles(16, {3: (101.0, 100.6), 6: (102.0, 99.0), 11: (100.97, 100.6)}))
    assert e[e.side == "high"].empty


def test_a_real_dip_between_them():
    assert eq(candles(14, {3: (101.0, 100.6), 9: (100.97, 100.6)}))[lambda d: d.side == "high"].empty   # 0.4 < 0.8
    assert len(eq(candles(14, {3: (101.0, 100.6), 6: (100.9, 100.2), 9: (100.97, 100.6)}))) >= 1        # 0.8 = 0.08


def test_three_touches_and_the_same_dip_is_not_a_new_touch():
    rows = candles(22, {3: (101.0, 100.6), 6: DIP, 9: (100.97, 100.6), 12: DIP, 15: (100.99, 100.6),
                        18: (101.0, 100.6)})
    e = eq(rows)
    highs = e[e.side == "high"]
    assert highs[["touch_bar", "touches", "prev_touch_bar"]].values.tolist() == [[t(9), 2, t(3)], [t(15), 3, t(9)]]
    assert set(highs["level_id"]) == {"equal_high:5m:2019-01-07 01:15"}
    # bar 18 touches again, but price did not dip since bar 15: the same dip, not a 4th touch


def test_equal_lows_are_the_mirror():
    e = eq(mirror(candles(14, {3: (101.0, 100.6), 6: DIP, 9: (100.97, 100.6)})))
    lows = e[e.side == "low"]
    assert len(lows) == 1
    assert lows.iloc[0].price == pytest.approx(99.0) and lows.iloc[0].touch_bar == t(9)
    assert e[e.side == "high"].empty


def test_min_apart_is_a_setting_default_none():
    rows = candles(14, {3: (101.0, 100.6), 6: DIP, 9: (100.97, 100.6)})          # 6 candles apart
    assert settings.get("equal_levels_min_apart_bars") == 0                        # his written rule: any time apart
    assert len(eq(rows, min_apart=6)) == 1
    assert eq(rows, min_apart=7)[lambda d: d.side == "high"].empty                 # the T3 reading, a version only


def test_two_weeks_on_the_5_minute_chart_no_limit_on_the_hour_chart():
    far = settings.get("equal_levels_max_apart_bars") + 10
    rows = candles(far + 20, {3: (101.0, 100.6), 6: DIP, far + 3: (100.97, 100.6)})
    assert eq(rows, "5m")[lambda d: d.side == "high"].empty
    got = eq(rows, "1h")                                                           # #89: "two days one week"
    assert len(got[got.side == "high"]) == 1
    assert got.iloc[0].tf == "1h"


def test_no_ruler_no_pair():
    bars = bars_from(candles(14, {3: (101.0, 100.6), 6: DIP, 9: (100.97, 100.6)}))
    assert EQ.equal_levels(bars, "5m", ruler=np.full(len(bars), np.nan)).empty     # no invented day's range


# ---- sweeps (#95, #129, #130) ----------------------------------------------------------------------------------

def level(level_id, side, price, known, kind=None, can_start=True, life=2880):
    kind = kind or f"equal_{side}"
    return dict(known_at=known, level_id=level_id, event="born", kind=kind, tf="5m", side=side, price=price,
                first_bar=known, touch_bar=pd.NaT, touches=2, rank=LQ.RANK[kind], can_start=can_start, target_when="always",
                life_bars=life, label=level_id)


def book_of(*rows):
    return LQ._frame(pd.DataFrame(list(rows)))      # flat levels: slope 0, no x0 (L6b.1's line columns)


def test_a_wick_one_cent_through_is_a_sweep_and_before_it_is_known_it_is_not():
    bars = bars_from(candles(12, {2: (101.5, 100.6), 5: (101.0, 100.6), 7: (101.01, 100.6)}))
    b = book_of(level("A", "high", 101.0, t(3) + M5))
    s = LQ.sweeps(bars, b)
    assert s[["level_id", "bar", "fresh"]].values.tolist() == [["A", t(7), True]]
    assert s.iloc[0].known_at == t(7) + M5
    # bar 2 went above before the level was known; bar 5 only reached it (not through): "even 1 point above" (#95)


def test_a_level_swept_once_is_finished_and_the_same_price_again_is_not_fresh():
    bars = bars_from(candles(40, {5: (101.05, 100.6), 20: (101.3, 100.6), 30: (103.5, 100.6), 33: (100.9, 99.0)}))
    b = book_of(level("A", "high", 101.0, t(1)), level("B", "high", 101.04, t(10)),
                level("E", "high", 101.2, t(10)), level("C", "high", 103.0, t(10)),
                level("D", "low", 99.1, t(10), kind="session_low"))
    s = LQ.sweeps(bars, b).set_index("level_id")
    assert s.loc["A", "bar"] == t(5) and s.loc["A", "fresh"]
    assert not s.loc["B", "fresh"] and s.loc["B", "same_as"] == "A"      # 4 cents from A: "really equal" (#129)
    assert s.loc["E", "fresh"]                                           # 20 cents: another level (#89: not equal)
    assert s.loc["C", "fresh"]                                           # $2 away: another level
    assert s.loc["D", "fresh"]                                           # a low is not the same stops as a high


def test_the_dead_window_is_two_days():
    dead = settings.get("how_long_a_swept_level_stays_dead")
    assert dead == 576                                                   # #130 "two days is correct"
    far = dead + 30
    bars = bars_from(candles(far + 10, {5: (101.05, 100.6), far: (101.3, 100.6)}))
    b = book_of(level("A", "high", 101.0, t(1)), level("B", "high", 101.04, t(10)))
    s = LQ.sweeps(bars, b).set_index("level_id")
    assert s.loc["B", "fresh"]                                           # more than two days later: liquidity again


def test_a_target_only_poke_does_not_use_up_a_price_and_life_ends_a_level():
    bars = bars_from(candles(40, {5: (101.05, 100.6), 20: (101.3, 100.6)}))
    b = book_of(level("S", "high", 101.0, t(1), kind="swing_high", can_start=False),
                level("B", "high", 101.04, t(10)), level("L", "high", 101.25, t(10), life=5))
    s = LQ.sweeps(bars, b).set_index("level_id")
    assert s.loc["B", "fresh"]                                           # a 5-minute swing is not entry liquidity (#91)
    assert "L" not in s.index                                            # expired 5 candles after it was known


def test_the_false_break_reading_and_off(monkeypatch):
    bars = bars_from(candles(40, {5: (101.05, 100.6), 20: (101.3, 100.6)}))
    b = book_of(level("A", "high", 101.0, t(1)), level("B", "high", 101.04, t(10)))
    real = settings.get

    def with_(mode):
        monkeypatch.setattr(settings, "get", lambda k: mode if k == "a_level_swept_once_is_finished" else real(k))
        return LQ.sweeps(bars, b).set_index("level_id")

    assert not with_("false_break").loc["B", "fresh"]    # candle 5 closed back below 101.0: a false break
    assert with_(False).loc["B", "fresh"]                # the old engine before 21 Sep: every sweep counts


# ---- swing roles (#91, #93) -------------------------------------------------------------------------------------

def test_swing_roles_his_jobs():
    r = LQ.swing_roles({"5m": random_bars(600, 1), "1h": random_bars(300, 2, 60), "1d": random_bars(200, 3, 1440)})
    by = r.groupby("tf").first()
    assert (by.loc["5m", "can_start"], by.loc["5m", "target_when"]) == (False, "higher_timeframe_against")   # #91
    assert by.loc["5m", "life_bars"] == settings.get("recent_swing_bars")
    assert (by.loc["1h", "can_start"], by.loc["1h", "target_when"]) == (False, "always")                     # #93
    assert (by.loc["1d", "can_start"], by.loc["1d", "target_when"]) == (True, "always")      # day / week start
    assert set(r["kind"]) == {"swing_high", "swing_low"}


# ---- recent first (#130) and the ranking ------------------------------------------------------------------------

def alive_rows(*rows):
    cols = ["level_id", "kind", "side", "price", "rank", "known_at", "can_start", "target_when", "strong"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


ROWS = alive_rows(
    ("old_near", "session_high", "high", 101.0, 1, t(1), True, "always", False),
    ("new_far", "session_high", "high", 103.0, 1, t(50), True, "always", False),
    ("equal", "equal_high", "high", 105.0, 3, t(10), True, "always", False),
    ("swing5", "swing_high", "high", 100.5, 1, t(60), False, "higher_timeframe_against", False),
    ("swing1h", "swing_high", "high", 100.8, 3, t(55), False, "always", False),
    ("below", "equal_low", "low", 99.0, 3, t(5), True, "always", False),
)


def test_recent_first_kind_then_most_recent_then_nearest():
    got = LQ.recent_first(ROWS, "high", 100.0)
    assert got["level_id"].tolist() == ["equal", "new_far", "old_near"]      # Equal > Session; today's first (#130)


def test_recent_first_off_is_kind_then_nearest(monkeypatch):
    real = settings.get
    monkeypatch.setattr(settings, "get",
                        lambda k: False if k == "the_most_recent_liquidity_comes_first" else real(k))
    assert LQ.recent_first(ROWS, "high", 100.0)["level_id"].tolist() == ["equal", "old_near", "new_far"]


def test_targets_only_when_asked_and_a_5_minute_swing_only_against_the_higher_timeframe():
    got = LQ.recent_first(ROWS, "high", 100.0, include_targets=True)["level_id"].tolist()
    assert got == ["swing1h", "equal", "new_far", "old_near"]              # the hour swing: a target (#93)
    got = LQ.recent_first(ROWS, "high", 100.0, include_targets=True, higher_timeframe_against=True)
    assert "swing5" in got["level_id"].tolist()                            # #91
    assert LQ.recent_first(ROWS, "low", 100.0)["level_id"].tolist() == ["below"]


def test_three_touches_rank_one_step_up():
    rows = ROWS.copy()
    rows.loc[rows.level_id == "old_near", ["strong", "kind", "rank"]] = [True, "equal_high", 3]
    assert LQ.recent_first(rows, "high", 100.0)["level_id"].tolist()[:2] == ["old_near", "equal"]   # "3>2"


# ---- the book, lifetimes, alive ----------------------------------------------------------------------------------

def test_obvious_liquidity_is_his_kinds_only():
    bars = random_bars(8000, seed=5)
    b = LQ.obvious_liquidity(bars)
    assert set(b["kind"]) <= set(settings.get("obvious_liquidity_kinds"))
    assert {"session_high", "session_low", "equal_high", "equal_low"} <= set(b["kind"])
    assert b["can_start"].all()
    s = b[b.kind == "session_high"]
    assert s["label"].str.endswith("session high").all()


def test_an_hour_pair_at_the_same_price_as_a_live_5_minute_pair_is_one_level():
    bars = bars_from(candles(30, {}))
    five = LQ._equal(pd.DataFrame([dict(known_at=t(2), level_id="five", tf="5m", side="high", price=101.0,
                                         first_bar=t(0), touch_bar=t(1), touches=2)]), "5m")
    hour = LQ._equal(pd.DataFrame([dict(known_at=t(10), level_id="hour", tf="1h", side="high", price=101.0,
                                         first_bar=t(0), touch_bar=t(8), touches=2),
                                    dict(known_at=t(10), level_id="hour2", tf="1h", side="high", price=101.5,
                                         first_bar=t(0), touch_bar=t(8), touches=2)]), "1h")
    kept = LQ._drop_same_level(pd.concat([five, hour], ignore_index=True), bars)
    assert kept["level_id"].tolist() == ["five", "hour2"]


def test_alive_is_known_before_the_candle_and_not_taken_by_its_end():
    bars = bars_from(candles(20, {8: (101.05, 100.6)}))
    b = book_of(level("A", "high", 101.0, t(3) + M5))
    s = LQ.sweeps(bars, b)
    life = LQ.lifetimes(bars, b, s)
    assert [i for i in range(20) if len(LQ.alive(life, i))] == [4, 5, 6, 7]   # known from 4; taken ON candle 8


# ---- "the London high" (L6-new: which session a session level comes from) -------------------------------------------

def test_the_london_high_is_the_london_windows_high_known_at_its_end_and_taken_after():
    # candle 0 opens 04:30 India time: Asia = candles 0-71, London (12:30-15:30) = 96-131, New York = 156-191
    bars = bars_from(candles(200, {10: (101.2, 100.6), 100: (101.5, 100.6), 150: (101.51, 100.6)}))
    b = LQ.obvious_liquidity(bars)
    lon = LQ.of_session(b, "London")
    assert set(lon["kind"]) == {"session_high", "session_low"} and set(lon["session"]) == {"London"}
    hi = lon[lon.side == "high"].iloc[0]
    assert (hi.price, hi.first_bar, hi.known_at) == (101.5, t(100), t(132))     # the window's high; known at its end
    assert (to_ist(hi.known_at).hour, to_ist(hi.known_at).minute) == (15, 30)
    sw = LQ.sweeps(bars, b)
    taken = LQ.of_session(sw, ["London"])
    taken = taken[taken.side == "high"]
    assert taken[["bar", "known_at", "price"]].values.tolist() == [[t(150), t(151), 101.5]]   # one cent through (#95)
    # the candle that MADE the London high took the Asia high: that is the Asia high taken, not the London high
    asia = LQ.of_session(sw, "Asian")
    assert asia[asia.side == "high"]["bar"].tolist() == [t(100)]
    assert LQ.of_session(sw, ["NY", "Asia"])["session"].isin(["New York", "Asia"]).all()


def test_only_a_session_level_has_a_session():
    bars = random_bars(8000, seed=5)
    b = LQ.book(bars, None, None, None)
    ses = b["kind"].str.startswith("session_")
    assert ses.any() and (~ses).any()
    assert b.loc[ses, "session"].isin(SES.names()).all() and b.loc[~ses, "session"].isna().all()
    assert (b.loc[ses, "label"] == b.loc[ses, "session"] + " session " + b.loc[ses, "side"]).all()
    sw = LQ.sweeps(bars, b)
    of = b.drop_duplicates("level_id").set_index("level_id")["session"]
    assert (sw["session"].fillna("-") == sw["level_id"].map(of).fillna("-")).all()     # a sweep says its level's
    assert set(LQ.of_session(sw, SES.names())["kind"]) <= {"session_high", "session_low"}


def test_of_session_reads_his_names_and_refuses_any_other():
    said = ("london", "London session", "Asian", "asia", "NY", "new_york", "New York")
    assert [SES.name_of(x) for x in said] == ["London", "London", "Asia", "Asia", "New York", "New York", "New York"]
    b = LQ.obvious_liquidity(random_bars(3000, seed=5))
    with pytest.raises(ValueError, match="Tokyo.*his sessions are Asia, London, New York"):
        LQ.of_session(b, ["London", "Tokyo"])
    with pytest.raises(ValueError, match="at least one session"):
        LQ.of_session(b, [])
    with pytest.raises(ValueError, match="no 'session' column"):
        LQ.of_session(b.drop(columns="session"), "London")


# ---- the cut-off-the-future test ----------------------------------------------------------------------------------

def _eq_fake(b):
    return EQ.equal_levels(b, "5m", ruler=np.full(len(b), 6.0))


def _book_fake(b):
    return LQ.book(b, None, None, None)


def _sweeps_fake(b):
    return LQ.sweeps(b, LQ.obvious_liquidity(b))


def _london_levels_fake(b):
    return LQ.of_session(LQ.obvious_liquidity(b), "London")


def _london_sweeps_fake(b):
    return LQ.of_session(_sweeps_fake(b), "London")


EVENT_BLOCKS = {"equal_levels": _eq_fake, "book": _book_fake, "sweeps": _sweeps_fake,
                "london_levels": _london_levels_fake, "london_sweeps": _london_sweeps_fake}


@pytest.mark.parametrize("name", sorted(EVENT_BLOCKS))
def test_no_future_liquidity_blocks_fake(name):
    bars = random_bars(8000, seed=9)
    guards.assert_no_future_events(EVENT_BLOCKS[name], bars, 5, [3500, 5000, 6500, 7999])


def test_the_future_test_catches_a_london_high_used_before_its_window_ends():
    """The London high is known at 15:30 India time. A block that says it one hour sooner is caught: cut the bars
    inside the London window (candles 96-131 of each day; 288 a day) and that day's row is not there yet."""
    def early(b):
        x = _london_levels_fake(b).copy()
        x["known_at"] = x["known_at"] - pd.Timedelta(hours=1)
        return x

    bars = random_bars(3000, seed=9)
    assert len(_london_sweeps_fake(bars)) > 0
    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(early, bars, 5, [288 * d + 126 for d in range(1, 9)])


def test_the_future_test_catches_a_pair_stamped_one_candle_early():
    def early(b):
        e = _eq_fake(b)
        e["known_at"] = e["known_at"] - M5
        return e

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(early, random_bars(4000, seed=9), 5, list(range(300, 4000, 5)))


def test_alive_uses_only_what_was_known():
    bars = random_bars(8000, seed=12)
    full_b = LQ.obvious_liquidity(bars)
    full = LQ.lifetimes(bars, full_b, LQ.sweeps(bars, full_b))
    for p in (4000, 6000, 7999):
        cut = bars.iloc[:p]
        cb = LQ.obvious_liquidity(cut)
        life = LQ.lifetimes(cut, cb, LQ.sweeps(cut, cb))
        for i in range(p - 400, p, 7):
            a = LQ.alive(full, i)[["level_id", "strong"]].sort_values("level_id").values.tolist()
            b = LQ.alive(life, i)[["level_id", "strong"]].sort_values("level_id").values.tolist()
            assert a == b, (p, i)


# ---- the same on real 2019 bars ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    from machine.data.bars import load_bars
    return {tf: load_bars(tf, "2019-02-01", "2019-03-31") for tf in ("5m", "1h", "1d")}


@pytest.mark.data
def test_no_future_equal_levels_real(real):
    for tf, step in (("5m", 5), ("1h", 60)):
        b = real[tf]
        n = len(b)
        guards.assert_no_future_events(lambda x: EQ.equal_levels(x, tf), b, step, [n // 2, 2 * n // 3, n - 1])


@pytest.mark.data
def test_no_future_book_and_sweeps_real(real):
    m5 = real["5m"]
    n = len(m5)
    cuts = [n // 2, 3 * n // 4, n - 1]
    guards.assert_no_future_events(lambda x: LQ.obvious_liquidity(x), m5, 5, cuts)
    guards.assert_no_future_events(lambda x: LQ.sweeps(x, LQ.obvious_liquidity(x)), m5, 5, cuts)
    guards.assert_no_future_events(lambda x: LQ.swing_roles({"5m": x}), m5, 5, cuts)


@pytest.mark.data
def test_no_future_the_london_high_real(real):
    m5 = real["5m"]
    n = len(m5)
    cuts = [n // 2, 3 * n // 4, n - 1]
    lon = LQ.of_session(LQ.sweeps(m5, LQ.obvious_liquidity(m5)), "London")
    assert len(lon[lon.side == "high"]) > 10 and len(lon[lon.side == "low"]) > 10
    guards.assert_no_future_events(_london_levels_fake, m5, 5, cuts)
    guards.assert_no_future_events(_london_sweeps_fake, m5, 5, cuts)
    # a London high is never taken before its own window has ended (15:30 India time)
    assert (lon["known_at"] > lon["level_known_at"]).all()
