"""L6b.2 - his strong zones (machine/blocks/zones.py) and their steps in the compiler.

His words tested here: strategy.md section 3 (TIME at price, the quiet zone, two tiers, weak after 2-3 touches, flips
when broken), #77 / #78 (a role is spent once used; "It will only if it breaks the zone, then it will actually
support"), #84 / #86 (the zone entry), #97 / #98 (the real break: 0.04 of the height "not broke", 0.437 "actual
break"), #99 (liquidity inside the zone), #87 (an unfilled gap past the hedge), sections 3 and 10 (double profit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import guards, rules, settings
from machine.blocks import fvg as FVG
from machine.blocks import liquidity as LQ
from machine.blocks import zones as Z
from machine.ideas import compile as C
from machine.ideas.form import Idea
from tests.block_fakes import START, bars_from, candles, random_bars
from tests.chart_fakes import to_1h
from tests.test_ideas import fake_m1, type1

M5 = pd.Timedelta(minutes=5)


def t(i: int) -> pd.Timestamp:
    return START + i * M5


def zone(lo: float, hi: float, known_at=START, life_bars: int = 100_000, tier: int = 1, rank: int = 0,
         zid: str = "z1", window: str = "today") -> pd.DataFrame:
    """One hand-made zone row (the columns strong_zones gives)."""
    return pd.DataFrame([{"known_at": pd.Timestamp(known_at), "zone_id": zid, "window": window, "tf": "5m",
                          "rank": rank, "lo": lo, "hi": hi, "bins": 3, "calm": 0.5, "tier": tier,
                          "src_first": pd.Timestamp(known_at) - pd.Timedelta(days=1),
                          "src_last": pd.Timestamp(known_at) - M5, "life_bars": life_bars, "ruler": 10.0,
                          "quiet_before": 0}])


def walk(*legs) -> pd.DataFrame:
    """legs = (candles, to, wick): price walks from leg to leg in straight steps."""
    closes, now = [], legs[0][1]
    wicks = []
    for n, to, wick in legs:
        closes += list(np.linspace(now, to, n + 1)[1:])
        wicks += [wick] * n
        now = to
    rows = candles(closes, 0.0)
    rows = [(o, h + w, lo - w, c) for (o, h, lo, c), w in zip(rows, wicks)]
    return bars_from(rows)


def ruler(b) -> np.ndarray:
    return np.full(len(b), 10.0)


def from_closes(closes, wick: float = 0.05) -> pd.DataFrame:
    return bars_from(candles(closes, wick))


# ---- his fixed rules are refused ----------------------------------------------------------------------------------

@pytest.mark.parametrize("name, value", [("ZB_BREAK_FRAC", 0.05), ("zone_calm_max", 0.9), ("ZONE_FLIP_RULE", "old"),
                                         ("what_a_real_zone_break_is", 0.1)])
def test_a_change_to_his_zone_rules_is_refused_with_his_words(name, value):
    with pytest.raises(rules.FixedRuleError):
        settings.resolve({name: value})


def test_his_numbers_are_where_the_blocks_read_them():
    assert rules.get("what_a_strong_zone_is")["calm_max"] == 0.74
    assert rules.get("what_a_strong_zone_is")["weak_after_touches"] == 3
    assert rules.get("what_a_real_zone_break_is") == 0.20
    assert Z.tiers_on() == (1, 2)                                        # #81: his main zone + his second choice
    with settings.using({"zone_tiers": "quiet"}):
        assert Z.tiers_on() == (1,)
    assert [w[0] for w in Z.windows()] == ["today", "week", "hour"]
    assert [w[4] for w in Z.windows()] == [0, 1, 2]                     # "1h stronger"


# ---- time at price, the quiet zone (section 3, #79, #80) ------------------------------------------------------------

def _parked_day() -> pd.DataFrame:
    """Day 1 (Mon 7 Jan 2019): price climbs 100 -> 110 in big candles, PARKS at 110 in tiny candles for 10 hours
    (bodies of 0.6 around 110), then runs to 118. Day 2: it wanders back down."""
    up = list(np.linspace(100.0, 110.0, 61)[1:])
    park = [110.0 + 0.3 * (-1) ** k for k in range(120)]
    run = list(np.linspace(110.0, 118.0, 97)[1:]) + list(np.linspace(118.0, 104.0, 289)[1:]) + [104.0] * 40
    rows = candles(up, 0.8)[:] + candles(park, 0.05) + candles(run, 0.8)
    for k in range(len(up), len(up) + len(park)):
        o, h, lo, c = rows[k]
        rows[k] = (o, max(o, c) + 0.05, min(o, c) - 0.05, c)
    return bars_from(rows)


def test_the_quiet_parked_band_is_his_main_zone_known_when_the_day_ends():
    m5 = _parked_day()
    z = Z.strong_zones(m5, to_1h(m5), ruler=ruler(m5), tiers=(1, 2, 3))
    today = z[z["window"] == "today"]
    first = today[today["known_at"] == pd.Timestamp("2019-01-08 00:00")]
    parked = first[(first["lo"] <= 110.0) & (first["hi"] >= 110.0)]
    assert len(parked) == 1 and (first.drop(parked.index)["tier"] > 1).all()   # the climb's bands are not quiet
    r = parked.iloc[0]
    assert r.lo <= 110.0 <= r.hi and r.hi - r.lo < 3.0                   # a tight band (the day ran $18)
    assert r.tier == 1 and r.calm <= 0.74                                # small candles: his main zone (#80)
    assert r.src_first == pd.Timestamp("2019-01-07 01:00") and r.src_last == pd.Timestamp("2019-01-07 23:55")
    assert (z["known_at"] >= pd.Timestamp("2019-01-08 00:00")).all()     # nothing before the first day ended


def test_a_day_not_finished_makes_no_zone():
    m5 = _parked_day().iloc[:200]                                        # day 1 cut at 17:35
    assert not len(Z.strong_zones(m5, to_1h(m5), ruler=ruler(m5), tiers=(1, 2, 3)))


def test_the_hour_window_counts_hour_candles_not_five_minute_ones():
    """strategy-understanding 47a: the old engine counted 5-minute bodies in its 'hour' window."""
    m5 = random_bars(9000, seed=21)
    h1 = to_1h(m5)
    with settings.using({"zone_windows": (("hour", "1h", 20, 20),)}):
        z = Z.strong_zones(m5, h1, ruler=ruler(m5), tiers=(1, 2, 3))
    assert len(z) and (z["tf"] == "1h").all()
    assert set(pd.DatetimeIndex(z["src_first"]).minute) == {0}          # hour candles, not 5-minute ones


# ---- a zone's life: touches, roles, the flip (#77, #78) --------------------------------------------------------------

def _two_tests(then_break: bool) -> pd.DataFrame:
    # below the zone 100-101 -> up into it and back (resistance used) -> up into it again -> back, or through
    cl = [99.0] * 5 + [100.5, 99.0, 99.0, 98.5, 99.0, 100.5] + ([102.0, 102.5, 102.5] if then_break else [99.0] * 3)
    return from_closes(cl + [cl[-1]] * 6)


def test_a_respected_side_is_spent_and_the_zone_ends_when_it_holds_again():
    b = _two_tests(then_break=False)
    L = Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))
    assert [(k, role) for k, role, _ in L["touches"].iat[0]] == [(5, "res"), (10, "res")]
    # candle 11 still reaches into the zone (its open is 100.5): the visit ends at 12, the first candle wholly outside
    assert L["dead_by"].iat[0] == "used up" and L["death"].iat[0] == 12          # when that visit ended (#78)
    with settings.using({"zone_spent_role_waits_for_the_break": False}):
        assert Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))["death"].iat[0] == 10   # the old engine: at once


def test_a_spent_side_that_breaks_hands_the_zone_the_other_role():
    """#78: "It will only if it breaks the zone, then it will actually support"."""
    b = _two_tests(then_break=True)
    L = Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))
    assert L["death"].iat[0] == len(b) and L["dead_by"].iat[0] == "expired"
    assert Z.roles_used(L.iloc[0], len(b)) == {"res"}


def test_the_third_touch_ends_the_zone():
    cl = [99.0] * 3 + [100.5, 99.0, 102.0, 102.0, 100.8, 102.5, 102.5, 100.9, 103.0] + [103.0] * 4
    b = from_closes(cl)
    L = Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))
    assert len(L["touches"].iat[0]) == 3 and L["dead_by"].iat[0] == "used up"


def test_the_flip_needs_a_touch_first_and_ends_the_zone_when_confirmed():
    far = from_closes([112.0] * 30)                                               # born far away, never touched
    assert Z.lives(far, zone(100.0, 101.0), ruler=ruler(far))["dead_by"].iat[0] == "expired"
    b = from_closes([99.0, 100.5, 99.0] + [88.0] * 20)                           # touched, then a day's range below
    L = Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))
    k = int(settings.get("zone_flip_confirm_bars"))
    assert L["dead_by"].iat[0] == "flip" and L["death"].iat[0] == L["flip_bar"].iat[0] + k


# ---- the real break (#97, #98) -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("close, broke", [(101.04, False), (101.19, False), (101.21, True), (101.44, True)])
def test_a_tiny_close_through_is_not_a_break_a_real_one_is(close, broke):
    """His #98 panel D: 0.040 of the zone's height 'NOT break of the zone', 0.437 'actual break'; the line: 0.20."""
    b = from_closes([99.0] * 5 + [100.5, close] + [close] * 5, wick=0.0)
    B = Z.breaks(b, Z.lives(b, zone(100.0, 101.0), ruler=ruler(b)))
    assert (len(B) == 1) == broke
    if broke:
        r = B.iloc[0]
        assert r.direction == "buy" and r.edge == 101.0 and r.hedge == pytest.approx(99.99)
        assert r.known_at == t(6) + M5                                   # known when that candle closed


def test_a_zone_that_holds_is_not_broken():
    """Price comes from ABOVE, dips into the zone and goes back up: the zone held (ZB_THROUGH, #98 / T06)."""
    b = from_closes([103.0] * 5 + [100.5, 101.5, 102.0] + [102.0] * 5, wick=0.0)
    assert not len(Z.breaks(b, Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))))


def test_a_tiny_poke_then_the_real_break_came_through_the_zone():
    """From below, a tiny close above (the zone 'drawn a bit tight', #98), back inside, then the real break."""
    b = from_closes([99.0] * 5 + [100.5, 101.05, 100.6, 101.5] + [101.5] * 5, wick=0.0)
    B = Z.breaks(b, Z.lives(b, zone(100.0, 101.0), ruler=ruler(b)))
    assert len(B) == 1 and B["bar"].iat[0] == t(8)


def test_one_candle_straight_through_from_the_other_side_is_a_break():
    b = from_closes([99.0] * 5 + [101.6] + [101.6] * 5, wick=0.0)
    B = Z.breaks(b, Z.lives(b, zone(100.0, 101.0), ruler=ruler(b)))
    assert len(B) == 1 and B["bar"].iat[0] == t(5)


def test_liquidity_inside_the_zone_is_taken_then_the_gap():
    """#99: wait for the level inside to be taken, then the 5-minute gap; the hedge past the take's extreme."""
    rows = [(101.5, 101.6, 101.4, 101.5)] * 3 + [(101.5, 101.5, 100.3, 100.4),       # 3: takes 100.5 inside the zone
                                                  (100.4, 100.6, 100.2, 100.55),       # 4
                                                  (100.55, 101.2, 100.5, 101.1),       # 5
                                                  (101.1, 101.8, 100.9, 101.7)]        # 6: low 100.9 > high of 4 (100.6)
    b = bars_from(rows + [(101.7, 101.8, 101.6, 101.7)] * 3)
    got = Z.after_inner_take(b, 0, "buy", 100.5, 10.0)
    assert got["take_bar"] == t(3) and got["bar"] == t(6)
    assert got["entry"] == 100.9 and got["hedge"] == pytest.approx(100.19)
    none = bars_from(rows[:4] + [(100.4, 100.6, 100.3, 100.5)] * 50)
    assert isinstance(Z.after_inner_take(none, 0, "buy", 100.5, 10.0), str)             # no gap, no entry (#98)


# ---- the zone entry (#84, #86) ------------------------------------------------------------------------------------------

def _sweeps(rows) -> pd.DataFrame:
    """Hand-made liquidity.sweeps rows: (candle, kind, side, price)."""
    return pd.DataFrame([{"known_at": t(k) + M5, "level_id": f"L{k}{kind}", "kind": kind, "tf": "5m", "side": side,
                          "price": px, "level_known_at": START, "bar": t(k), "can_start": True, "fresh": True,
                          "same_as": None} for k, kind, side, px in rows], columns=list(LQ.SWEEP_COLUMNS))


def test_a_take_just_above_a_zone_arms_a_buy_at_the_zone():
    b = from_closes([104.0] * 5 + [102.6, 102.2, 102.4, 102.5, 101.9, 101.2, 100.8, 102.0] + [102.0] * 5, wick=0.1)
    E = Z.entries(b, Z.lives(b, zone(100.0, 101.0), ruler=ruler(b)),
                  _sweeps([(6, "session_low", "low", 102.2), (8, "equal_low", "low", 102.4)]), ruler=ruler(b))
    r = E.iloc[0]
    assert (r.direction, r.bar, r.entry, r.hedge, r.level_kind) == ("buy", t(6), 101.0, 99.99, "session_low")
    assert r.known_at == t(6) + M5 and r.liq_gap_adr == pytest.approx(0.12)
    # the second take at the same zone arms again: a newer setup - the tester keeps one order (#145 "Cancel and take new")
    assert list(E["bar"]) == [t(6), t(8)] and E["entry"].iat[1] == 101.0


def test_no_zone_entry_when_the_take_reaches_the_zone_or_the_level_is_not_obvious_or_the_side_is_used():
    b = from_closes([104.0] * 5 + [102.6, 101.5, 100.7, 102.0] + [102.0] * 5, wick=0.1)
    L = Z.lives(b, zone(100.0, 101.0), ruler=ruler(b))
    assert not len(Z.entries(b, L, _sweeps([(7, "session_low", "low", 101.2)]), ruler=ruler(b)))   # reached it
    assert not len(Z.entries(b, L, _sweeps([(6, "swing_low", "low", 102.0)]), ruler=ruler(b)))    # #86 B
    used = from_closes([103.0] * 3 + [100.8, 103.0, 103.5, 102.6, 102.2, 102.5] + [102.5] * 4, wick=0.1)
    Lu = Z.lives(used, zone(100.0, 101.0), ruler=ruler(used))
    assert not len(Z.entries(used, Lu, _sweeps([(7, "session_low", "low", 102.2)]), ruler=ruler(used)))  # #78


# ---- double profit (sections 3, 10) and a zone as a target ----------------------------------------------------------------

def test_the_zone_before_the_target_and_where_he_gets_back_in():
    life = pd.concat([zone(105.0, 106.0, zid="ahead", rank=2), zone(99.5, 100.5, zid="here", rank=2),
                      zone(103.0, 104.0, zid="small", rank=0)], ignore_index=True)
    b = from_closes([100.0] * 10)
    L = Z.lives(b, life, ruler=ruler(b))
    got = Z.in_path(L, 5, "buy", 100.0, 110.0)
    assert got["zone_id"] == "ahead" and got["book_at"] == 105.0 and got["far"] == 106.0
    assert Z.in_path(L, 5, "buy", 100.0, 110.0, min_rank=0)["zone_id"] == "small"
    assert Z.in_path(L, 5, "buy", 100.0, 104.5) is None                 # the hour zone is past the target
    assert Z.reentry_price(got, "retest") == 105.0 and Z.reentry_price(got, "break") == 106.0
    assert Z.reentry_price(got, "setup") is None
    sell = Z.in_path(L, 5, "sell", 100.0, 90.0, min_rank=0)
    assert sell is None                                                  # standing inside 99.5-100.5 is not "before"
    assert Z.nearest_edge(L, 5, "high", 100.0)["price"] == 103.0


# ---- #87: an unfilled gap past the hedge --------------------------------------------------------------------------------

def test_an_unfilled_gap_above_a_sells_hedge_and_once_filled_it_is_gone():
    rows = [(108.0, 108.2, 107.8, 108.0), (108.0, 108.1, 107.0, 107.1), (107.1, 107.2, 105.9, 106.0),
            (106.0, 106.2, 104.0, 104.2)] + [(104.2, 104.4, 103.8, 104.0)] * 10
    b = bars_from(rows)
    g = FVG.unfilled_past(b, len(b) - 1, "sell", 105.0, 10.0)
    # three gaps above (104.4-105.9, 106.2-107.0, 107.2-107.8): the nearest, only its part above the hedge counts
    assert g is not None and g["lo"] == pytest.approx(105.0) and g["hi"] == pytest.approx(105.9)
    assert g["bar"] == t(4)
    filled = bars_from(rows + [(104.0, 108.0, 103.9, 104.0)])
    assert FVG.unfilled_past(filled, len(filled) - 1, "sell", 105.0, 10.0) is None
    assert FVG.unfilled_past(b, len(b) - 1, "buy", 103.0, 10.0) is None  # it is on the other side


# ---- nothing reads the future ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def walk9():
    return random_bars(9000, seed=14)


def _zones(b):
    return Z.strong_zones(b, to_1h(b))


def _touches(b):
    return Z.touch_events(Z.lives(b, _zones(b)), b)


def _breaks(b):
    return Z.breaks(b, Z.lives(b, _zones(b)))


def _entries(b):
    bk = LQ.obvious_liquidity(b)
    return Z.entries(b, Z.lives(b, _zones(b)), LQ.sweeps(b, bk))


EVENTS = {"strong_zones": _zones, "touches": _touches, "breaks": _breaks, "entries": _entries}


@pytest.mark.parametrize("name", sorted(EVENTS))
def test_no_future_fake(name, walk9):
    assert len(EVENTS[name](walk9)), name                                # the test compares something
    guards.assert_no_future_events(EVENTS[name], walk9, 5, [4000, 5500, 7100, 8999])


def test_alive_uses_only_what_was_known(walk9):
    full = Z.lives(walk9, _zones(walk9))
    for p in (5000, 8000):
        cut = walk9.iloc[:p]
        part = Z.lives(cut, _zones(cut))
        for i in range(p - 400, p, 9):
            a = sorted(Z.alive(full, i)["zone_id"])
            assert a == sorted(Z.alive(part, i)["zone_id"]), (p, i)
            for side, px in (("high", float(walk9["close"].iat[i])), ("low", float(walk9["close"].iat[i]))):
                assert Z.nearest_edge(full, i, side, px) == Z.nearest_edge(part, i, side, px)


@pytest.fixture(scope="module")
def real():
    from machine.data.bars import load_bars
    return load_bars("5m", "2019-01-01", "2019-04-30")


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(EVENTS))
def test_no_future_real(real, name):
    n = len(real)
    guards.assert_no_future_events(EVENTS[name], real, 5, [n // 2, 3 * n // 4, n - 1])


# ---- the compiler's zone steps ---------------------------------------------------------------------------------------------

def _form(fid: str, steps: list, entry_at: str, hedge_at: str, kinds=("equal", "session", "trendline"), dne=(),
          filters=()) -> dict:
    return dict(id=fid, title="a zone test form", words="a zone test form", timeframe="5m",
                directions=["buy", "sell"], steps=steps, entry=dict(how="limit", at=entry_at, max_wait_bars="no limit"),
                exit=dict(kind="hedge", at=hedge_at), targets=[dict(kind="next_liquidity", kinds=list(kinds))],
                do_not_enter=[dict(name="target_reached_first"), *dne], filters=list(filters))


ZE = _form("idea-931", [dict(id="z", what="his zone entry", block="zone_entry", tf="5m")], "z.entry", "z.hedge",
           dne=[dict(name="unfilled_gap_past_hedge")])
ZB = _form("idea-932", [dict(id="zb", what="his zone break retest", block="zone_break", tf="5m")], "zb.entry",
           "zb.hedge")
GZ = type1(id="idea-933", steps=type1()["steps"] + [dict(id="at", what="the gap is at a strong zone", block="zone",
                                                         tf="5m", max_wait_bars=0, params=dict(at="gap.near"))],
           targets=[dict(kind="next_liquidity", kinds=["equal", "session", "zone"])])


@pytest.fixture(scope="module")
def m1():
    return fake_m1(days=30, seed=3)


@pytest.fixture(scope="module")
def market(m1):
    return C.Market.from_m1(m1)


def test_the_zone_steps_and_the_zone_target_are_built(market):
    for d in (ZE, ZB, GZ):
        f = Idea.load(d)
        assert C.check_without_bars(f) == []                             # nothing waits for L6b.2 any more
        out = C.compile_idea(f, market)
        assert out.signals, f.id
    for s in C.compile_idea(Idea.load(ZE), market).signals:
        assert s.entry_type == "limit" and (s.stop < s.entry_price if s.direction == "buy" else s.stop > s.entry_price)


def test_a_zone_is_not_a_level_that_is_swept(market):
    bad = type1(id="idea-934", steps=[dict(type1()["steps"][0], params=dict(kinds=["zone"]))] + type1()["steps"][1:])
    with pytest.raises(C.CompileError) as err:
        C.compile_idea(Idea.load(bad), market)
    assert not isinstance(err.value, C.MissingBlockError)


def test_a_zone_target_is_the_nearest_zone_edge_when_it_is_nearer(market):
    plain = C.compile_idea(Idea.load(type1(id="idea-935")), market)
    with_zone = C.compile_idea(Idea.load(type1(id="idea-935", targets=[dict(kind="next_liquidity",
                                                                            kinds=["equal", "session", "swing",
                                                                                   "trendline", "zone"])])), market)
    by = {(s.known_at, s.direction): s for s in plain.signals}
    nearer = 0
    for s in with_zone.signals:
        p = by.get((s.known_at, s.direction))
        if p is not None and p.targets != s.targets:
            assert abs(s.targets[0] - s.entry_price) < abs(p.targets[0] - p.entry_price)
            nearer += 1
    assert nearer > 0


def test_compiled_zone_signals_do_not_peek(m1):
    ideas = [Idea.load(d) for d in (ZE, ZB, GZ)]

    def signals_of(bars):
        mk = C.Market.from_m1(bars)
        rows = [{"form": i.id, **s.model_dump(include={"known_at", "direction", "entry_price", "stop", "targets"})}
                for i in ideas for s in C.compile_idea(i, mk).signals]
        return pd.DataFrame(rows, columns=["form", "known_at", "direction", "entry_price", "stop", "targets"])

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])
