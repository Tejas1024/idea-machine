"""L19c - his management rules as versions: machine/tester/management.py, the TEST rows in the hedging tester,
the fixed rows in machine/rules.py MANAGEMENT, and the research tools that let a version try them.

Hand-made 1-minute bars and a hand-made market from tests/test_hedging.py (Monday 11 Mar 2019, spread $0.20, a day's
range $10). Every setting has a test that its other value changes the trade the way its words say - and that version 1
(its default) leaves every trade as it was before L19c.
"""
from __future__ import annotations

import functools
import json
import re

import pandas as pd
import pytest

from machine import config, rules, settings
from machine.ideas import compile as C
from machine.ideas.form import Idea
from machine.mcp import tools as T
from machine.mcp import versions as V
from machine.referee import trials
from machine.tester import hedging
from machine.tester import management as MG
from tests.test_hedging import Market, at, money, order, path, run, two_leg_story
from tests.test_mcp import FORM, M1, READ, bench, lab, make_idea, refused  # noqa: F401 - lab is a fixture

FRIDAY = pd.Timedelta(days=4)                       # Monday 11 Mar 2019 + 4 days = Friday 15 Mar 2019


class Zoned(Market):
    """The hand-made market, with the three questions only his TEST rows ask (zones, a trendline the trend's way)."""

    def __init__(self, m1, *a, edge=None, in_path=None, trend=False, **kw):
        super().__init__(m1, *a, **kw)
        self._edge, self._in_path, self._trend = edge, in_path, trend

    def zone_edge(self, side, t, price, max_away, min_rank):
        self.asked.append(("zone_edge", side, min_rank))
        z = self._edge
        if z is None or (side == "low" and z["price"] >= price) or (side == "high" and z["price"] <= price):
            return None
        return None if max_away is not None and abs(price - z["price"]) > max_away else dict(z)

    def zone_in_path(self, direction, t, entry, target, min_rank):
        self.asked.append(("zone_in_path", direction, min_rank))
        z = self._in_path
        if z is None:
            return None
        lo, hi = sorted((entry, target))
        return dict(z) if lo < z["book_at"] < hi else None

    def trend_line(self, direction, t):
        return self._trend(direction, t) if callable(self._trend) else self._trend


def with_(changes: dict, signals, market, tf="5m"):
    with settings.using(changes):
        return hedging.replay(list(signals), market, tf)


def legs(t):
    return [(x.direction, x.why_open, x.open_px, x.close_px) for x in t.legs]


# ================================================================================================================
# the table: every row of his management table is fixed, a version, or not yet - with its reason
# ================================================================================================================

def t12_rows() -> dict[str, str]:
    text = (config.ROOT / MG.TABLE).read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"^\| (M\d\d) \|.*\| ([^|]*) \|\s*$", text, re.M)}


def test_every_row_of_his_table_is_fixed_or_a_tested_setting_or_says_why_not():
    rows = t12_rows()
    assert len(rows) == 39
    version_rows = {v.row for v in MG.VERSIONS.values()} | {"M33"}      # M33 = strong_zone_as_hedge_point zone_break
    for row, last in rows.items():
        fixed, test = "**fixed**" in last, "**test**" in last or "Box 2" in last
        assert fixed or test, row
        if fixed:
            assert row in rules.MANAGEMENT, f"{row} is fixed in his table: machine/rules.py MANAGEMENT"
        if test:
            assert row in version_rows or row in MG.OTHER, f"{row} is a test row: a version, or why not (OTHER)"
        if row in MG.OTHER:
            assert len(MG.OTHER[row]) > 40                                  # the reason, in words
    assert set(rules.MANAGEMENT) <= set(rows) and set(MG.OTHER) <= set(rows)
    assert set(rules.MANAGEMENT).isdisjoint({r for r in MG.OTHER if r not in ("M18", "M33")})


def test_his_fixed_rows_are_carried_by_his_settings_and_refused_as_versions():
    for row, m in rules.MANAGEMENT.items():
        for name in m.carried_by:
            s = settings.ALL[name]
            assert s.status == "his", (row, name)
            assert rules.managed_by(name)[0] in rules.MANAGEMENT
        assert m.rule and m.taught and m.built


def test_every_version_setting_is_a_test_setting_of_his_table_with_version_1_as_its_default():
    for name, v in MG.VERSIONS.items():
        s = settings.ALL[name]
        assert s.status == "test" and s.t12 == v.row, name
        assert name not in MG.RETIRED and rules.managed_by(name) is None
        if v.values is not None:
            assert s.value in v.values and set(v.values) <= set(s.options or v.values), name
        for other in v.needs:
            assert other in MG.VERSIONS
        if s.value is not None or v.values is not None:                    # None = version 1, not a value to set
            assert MG.check_value(_hedge_form(), name, s.value, (MG.PATH if v.in_manager else "settings.") + name)


def test_the_retired_numbers_are_read_by_no_code():
    src = {p: p.read_text(encoding="utf-8") for p in (config.ROOT / "machine").rglob("*.py")
           if p.name not in ("settings.py", "management.py")}
    for name in MG.RETIRED:
        assert settings.ALL[name].status == "test"
        users = [p.name for p, s in src.items() if f'"{name}"' in s or f"'{name}'" in s]
        assert users == [], (name, users)


def test_version_1_of_every_setting_is_what_the_hedging_tester_did_before_L19c():
    m1, idea, setups = two_leg_story()
    before = run(idea, Zoned(m1, setups))
    for name, v in MG.VERSIONS.items():
        if v.in_manager:
            got = with_({name: settings.ALL[name].value}, idea, Zoned(m1, setups))
            assert [legs(t) for t in got.trades] == [legs(t) for t in before.trades], name
    r = hedging.Rules.now()
    assert (r.protection, r.friday, r.max_covers, r.escape, r.split, r.dbl, r.zone_hedge, r.zone_against,
            r.which_close, r.news_events) == ("never", "as_any_day", None, False, "first_target", "off", "off", "off",
                                              "any_rr", "forex_factory_red_and_orange")
    assert hedging.summarize(before)["test_rows_acted"] == {}


# ================================================================================================================
# M02 - a real stop instead of the hedge
# ================================================================================================================

def test_M02_always_a_real_stop_closes_at_the_hedge_point_and_the_trade_ends():
    m1, idea, setups = two_leg_story()
    (t,) = with_({"a_real_stop_instead_of_the_hedge": "always"}, idea, Zoned(m1, setups)).trades
    assert legs(t) == [("buy", "entry", 2000.0, 1998.0)] and t.status == "closed"
    assert t.real_stops == 1 and t.hedges == 0 and t.exit_reason == "real stop (M02)"
    assert t.net_inr == pytest.approx(money("buy", 2000.0, 1998.0, at(4), at(9)))


@pytest.mark.parametrize("ruler,stopped", [(10.0, False), (30.0, True)])
def test_M02_when_small_only_a_hedge_within_the_share_of_a_days_range_is_a_stop(ruler, stopped):
    """The hedge distance is $2.20 on MT5 prices (2000.20 -> 1998): small = at most 0.10 x the day's range."""
    m1, idea, setups = two_leg_story()
    (t,) = with_({"a_real_stop_instead_of_the_hedge": "when_small"}, idea, Zoned(m1, setups, ruler=ruler)).trades
    assert (t.real_stops, t.hedges > 0) == ((1, False) if stopped else (0, True))


def test_M02_a_trade_on_a_real_stop_gets_no_room_and_news_does_not_hedge_it():
    m1 = path((0, 2001), (5, 2000), (12, 1999.5), (20, 1997), (30, 1996))
    sig = [order("b", 0, "buy", 2000.20, 1998, 2010)]
    mk = dict(dial="with", levels=[(1990.0, "low", 0)], news=[at(25)])           # news acts at minute 10 (a loss)
    (v1,) = run(sig, Zoned(m1, **mk)).trades
    assert v1.room and v1.news_acts == 1                                         # version 1: room, then a news hedge
    (t,) = with_({"a_real_stop_instead_of_the_hedge": "always"}, sig, Zoned(m1, **mk)).trades
    assert not t.room and t.news_acts == 0 and legs(t) == [("buy", "entry", 2000.0, 1998.0)]


# ================================================================================================================
# M21 - the cap of hedges
# ================================================================================================================

def test_M21_at_the_cap_the_hedge_price_is_a_stop():
    """The story hedges twice (1998, then 1987.50). Cap 1: the 2nd hedge does not open - the buy left closes there."""
    m1, idea, setups = two_leg_story()
    (t,) = with_({"how_many_covers_allowed": 1}, idea, Zoned(m1, setups)).trades
    assert legs(t) == [("buy", "entry", 2000.0, 1987.5), ("sell", "hedge", 1998.0, 1988.5)]
    assert (t.hedges, t.cap_stops, t.positions) == (1, 1, 2) and t.exit_reason == "the cap of hedges, a stop (M21)"
    (t2,) = with_({"how_many_covers_allowed": 2}, idea, Zoned(m1, setups)).trades
    assert legs(t2) == legs(run(idea, Zoned(m1, setups)).trades[0]) and t2.cap_stops == 0


def test_M21_at_the_cap_news_leaves_a_losing_leg_as_it_is():
    """News acting at minute 32 (the buy left, bought at 1988.50, is at a loss): version 1 hedges it; at the cap of 1
    nothing happens there and the hedge price stops it at minute 34."""
    m1, idea, setups = two_leg_story()
    news = [at(47)]                                                              # acts 15 minutes before: minute 32
    (v1,) = run(idea, Zoned(m1, setups, news=news)).trades
    assert [x.why_open for x in v1.legs].count("news hedge") == 1
    (t,) = with_({"how_many_covers_allowed": 1}, idea, Zoned(m1, setups, news=news)).trades
    assert t.news_acts == 0 and t.cap_stops == 1 and t.legs[0].close_px == 1987.5


# ================================================================================================================
# M09 - Friday
# ================================================================================================================

def friday_trade(dial, changes, levels=((1996.0, "low", 0),)):
    m1 = path((0, 2001), (5, 2000), (12, 1999), (20, 1995), (30, 1994))
    m1 = m1.set_axis(m1.index + FRIDAY)
    sig = [order("b", 0, "buy", 2000.20, 1998, 2010).model_copy(update={"known_at": at(0) + FRIDAY})]
    with settings.using(changes):
        (t,) = hedging.replay(sig, Zoned(m1, dial=dial, levels=[(p, s, 0) for p, s, _ in levels]), "5m").trades
    return t


def test_M09_a_friday_entry_gets_room_or_no_room_by_the_setting():
    assert friday_trade("mixed", {}).legs[1].open_px == 1998.0                   # version 1: no room (split dial)
    t = friday_trade("mixed", {"on_a_friday": "room"})
    assert t.room and t.friday == 1 and t.legs[1].open_px == 1996.0              # room to the next liquidity
    assert friday_trade("with", {}).room                                         # version 1: the dial gives room
    t = friday_trade("with", {"on_a_friday": "hedge"})
    assert not t.room and t.friday == 1 and t.legs[1].open_px == 1998.0


def test_M09_only_fridays():
    m1, idea, setups = two_leg_story()                                           # a Monday
    (t,) = with_({"on_a_friday": "room"}, idea, Zoned(m1, setups)).trades
    assert t.friday == 0 and not t.room


# ================================================================================================================
# M24 - trailing when the hour and the day disagree; M35 trail_max_targets
# ================================================================================================================

def trail_trade(changes, dial="mixed"):
    m1 = path((0, 2001), (5, 2000), (20, 2010.5), (30, 2015.5), (40, 2012))
    sig = [order("b", 0, "buy", 2000.20, 1998, 2010)]
    with settings.using(changes):
        (t,) = hedging.replay(sig, Zoned(m1, dial=dial, levels=[(2015.0, "high", 0)]), "5m").trades
    return t


def test_M24_split_hour_and_day_first_target_or_hold():
    t = trail_trade({})
    assert t.legs[0].close_px == 2010.0 and t.trail_steps == 0
    t = trail_trade({"trail_when_hour_and_day_split": "hold"})
    assert t.legs[0].close_px == 2015.0 and t.trail_steps == 1 and t.split_holds == 1


def test_M35_trail_max_targets_caps_the_trail():
    assert trail_trade({}, dial="with").legs[0].close_px == 2015.0
    t = trail_trade({"trail_max_targets": 0}, dial="with")
    assert t.legs[0].close_px == 2010.0 and t.trail_steps == 0


# ================================================================================================================
# M07, M35 - room numbers
# ================================================================================================================

def room_price(changes, levels=()):
    m1 = path((0, 2001), (5, 2000), (30, 1980))
    sig = [order("b", 0, "buy", 2000.20, 1998, 2010)]
    with settings.using(changes):
        (t,) = hedging.replay(sig, Zoned(m1, dial="with", levels=list(levels)), "5m").trades
    return t.legs[1].open_px


def test_M07_how_much_room_when_no_liquidity_is_there():
    assert room_price({}) == 1988.0                                              # 1998 - 1.0 x $10
    assert room_price({"how_much_room": 0.5}) == 1993.0


def test_M35_next_liquidity_max_adr_how_far_the_next_liquidity_may_be():
    lv = [(1990.0, "low", 0)]
    assert room_price({}, lv) == 1990.0                                          # $8 away, inside 2.5 x $10
    assert room_price({"next_liquidity_max_adr": 0.5}, lv) == 1988.0             # $8 > $5: the room fallback


# ================================================================================================================
# M22 - the trend escape
# ================================================================================================================

def escape_story():
    """buy 2000.20 -> hedged at 1998 (R 2.20, 4R 8.80) -> a SELL setup fills at about 1996.6: inside the 1:4 zone of
    the buy (it points against the buy, which is at a loss)."""
    m1 = path((0, 2001), (5, 2000), (10, 1998), (14, 1996.6), (20, 1993), (30, 1985), (40, 1984))
    idea = [order("buy1", 0, "buy", 2000.20, 1998, 2010)]
    setups = [order("S1", 12, "sell", 1996.6, 1999.0, 1990.0, "idea-000")]
    return m1, idea, setups


def test_M22_the_escape_closes_the_losing_leg_inside_the_1_to_4_zone_only_with_the_trend():
    m1, idea, setups = escape_story()
    trend = dict(dial=lambda d, t: "with" if d == "sell" else "against", trend=True)
    (v1,) = run(idea, Zoned(m1, setups, **trend)).trades
    assert v1.legs[0].close_px is None or v1.legs[0].close_i > 30                # version 1: nothing closes there
    (t,) = with_({"close_the_losing_leg_when_a_trend_runs": True}, idea, Zoned(m1, setups, **trend)).trades
    buy = t.legs[0]
    assert t.escapes == 1 and buy.close_i is not None and buy.close_px < buy.open_px    # closed at a loss
    assert "trend escape" in buy.why_closed
    (no_line,) = with_({"close_the_losing_leg_when_a_trend_runs": True}, idea,
                       Zoned(m1, setups, dial=trend["dial"], trend=False)).trades
    assert no_line.escapes == 0                                                  # no trendline that way: no escape
    (no_dial,) = with_({"close_the_losing_leg_when_a_trend_runs": True}, idea,
                       Zoned(m1, setups, dial="mixed", trend=True)).trades
    assert no_dial.escapes == 0                                                  # the hour and the day not both


# ================================================================================================================
# M11 - #105 as a labelled comparison (his rule is #134)
# ================================================================================================================

def test_M11_the_105_reading_never_closes_a_leg_at_a_loss_and_is_labelled_not_his():
    m1, idea, setups = two_leg_story()
    with settings.using({"which_leg_may_close_while_covered": "profit"}) as r:
        (t,) = hedging.replay(idea, Zoned(m1, setups), "5m").trades
        assert r.breaks_his_rules and "not his rule" in r.breaks_his_rules[0]
    assert t.legs[0].close_i != 49                                               # #134 closed the buy there at a loss
    for x in t.legs:
        if x.close_i is not None and x.why_closed.startswith("a "):              # closed by a setup while hedged
            assert x.pnl.net_inr > 0


# ================================================================================================================
# M30 - double profit; M32 - a zone against you; M31 - a zone as the hedge point
# ================================================================================================================

ZONE = {"book_at": 2005.0, "far": 2006.0, "rank": 2, "tier": 1, "zone_id": "z1", "lo": 2005.0, "hi": 2006.0}


def dbl_trade(reading, points, setups=()):
    m1 = path(*points)
    sig = [order("b", 0, "buy", 2000.20, 1998, 2010)]
    with settings.using({"double_profit_reentry": reading} if reading else {}):
        (t,) = hedging.replay(sig, Zoned(m1, setups, in_path=ZONE), "5m").trades
    return t


BACK = ((0, 2001), (5, 2000), (10, 2005.5), (14, 2004.0), (24, 2010.5))


def test_M30_version_1_goes_to_the_target_in_one_leg():
    t = dbl_trade(None, BACK)
    assert legs(t) == [("buy", "entry", 2000.0, 2010.0)] and t.dbl_booked == 0


def test_M30_retest_books_at_the_zone_and_gets_back_in_at_that_price():
    t = dbl_trade("retest", BACK)
    assert legs(t) == [("buy", "entry", 2000.0, 2005.0), ("buy", "back in", 2004.8, 2010.0)]
    assert (t.dbl_booked, t.dbl_back, t.positions) == (1, 1, 2) and t.exit_reason == "target"


def test_M30_break_gets_back_in_past_the_far_edge():
    t = dbl_trade("break", BACK)
    assert legs(t)[0] == ("buy", "entry", 2000.0, 2005.0)
    assert legs(t)[1][1:] == ("back in", 2005.8, 2010.0)                         # the ask reached 2006


def test_M30_setup_gets_back_in_at_the_next_setup_of_his_that_way():
    s = order("S1", 11, "buy", 2004.6, 2003.0, 2012.0, "idea-000")
    t = dbl_trade("setup", BACK, setups=[s])
    assert t.dbl_back == 1 and "S1" in " ".join(r["words"] for r in t.story)


def test_M30_the_wait_ends_with_one_profit_when_the_target_comes_first():
    t = dbl_trade("retest", ((0, 2001), (5, 2000), (10, 2005.5), (20, 2010.5)))
    assert legs(t) == [("buy", "entry", 2000.0, 2005.0)] and (t.dbl_booked, t.dbl_back) == (1, 0)
    assert t.status == "closed" and t.exit_reason == "double profit, booked only (M30)"


def test_M30_once_hedged_there_is_no_booking():
    t = dbl_trade("retest", ((0, 2001), (5, 2000), (10, 1997.5), (30, 2008), (40, 2011)))
    assert t.dbl_booked == 0 and t.hedges == 1


def test_M32_small_cuts_the_target_back_to_the_zone():
    m1 = path(*BACK)
    sig = [order("b", 0, "buy", 2000.20, 1998, 2010)]
    (t,) = with_({"in_a_zone_against_you": "small"}, sig, Zoned(m1, in_path=ZONE)).trades
    assert legs(t) == [("buy", "entry", 2000.0, 2005.0)] and t.zone_small == 1 and t.exit_reason == "target"
    m = Zoned(m1, in_path=ZONE)
    with_({"in_a_zone_against_you": "small"}, sig, m)
    assert ("zone_in_path", "buy", 2) in m.asked                                 # the hour chart's zone (rank 2)


@pytest.mark.parametrize("reading,edge,want", [
    ("nearer", {"price": 1999.0, "far": 1998.5}, 1999.0),                        # the zone comes first
    ("nearer", {"price": 1997.0, "far": 1996.0}, 1998.0),                        # his hedge point comes first
    ("zone", {"price": 1997.0, "far": 1996.0}, 1997.0),
    ("zone_break", {"price": 1999.0, "far": 1998.5}, 1998.5),                    # the zone grid: on its break
])
def test_M31_a_strong_zone_as_the_hedge_point(reading, edge, want):
    m1, idea, setups = two_leg_story()
    z = {**edge, "rank": 2, "tier": 1, "zone_id": "z"}
    (t,) = with_({"strong_zone_as_hedge_point": reading}, idea, Zoned(m1, setups, edge=z)).trades
    assert t.legs[1].why_open == "hedge" and t.legs[1].open_px == want
    assert t.zone_hedges == (want != 1998.0)


def test_M31_the_zone_must_be_near_enough_and_strong_enough():
    m1, idea, setups = two_leg_story()
    z = {"price": 1999.0, "far": 1998.5, "rank": 2, "tier": 1, "zone_id": "z"}
    m = Zoned(m1, setups, edge=z)
    (t,) = with_({"strong_zone_as_hedge_point": "zone", "strong_zone_hedge_max_adr": 0.05}, idea, m).trades
    assert t.zone_hedges == 0                                                    # $1.00 away > 0.05 x $10
    with_({"strong_zone_as_hedge_point": "zone", "strong_zone_hedge_rank": 1}, idea, m)
    assert ("zone_edge", "low", 1) in m.asked


# ================================================================================================================
# M27 - how far past the line the trendline hedge sits (made by the compiler)
# ================================================================================================================

def test_M27_the_trendline_hedge_pad():
    from machine.blocks import trendlines as TL
    from tests.test_trendlines import RULER, falling, ruler, t
    b = falling()

    def hedge(changes):
        with settings.using(changes):
            e = TL.touch_entries(b, "5m", ruler(b))
        row = e[(e["touch"] == 3) & (e["first_bar"] == t(10))].iloc[0]
        return float(row["hedge"]) - float(row["line"])

    assert hedge({}) == pytest.approx(2.0 * 0.025 * RULER)                       # version 1: the line's break distance
    assert hedge({"trendline_hedge_pad_adr": 0.10}) == pytest.approx(0.10 * RULER)
    assert hedge({"trendline_hedge_pad_adr": 0.0}) == pytest.approx(0.0)


# ================================================================================================================
# M28 - red only, and the minutes before (the real calendar)
# ================================================================================================================

@pytest.mark.news
def test_M28_red_only_is_fewer_moments_than_red_and_orange_and_the_minutes_move_the_act():
    a, b = pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")
    both = hedging.news_moments(a, b, 15, "forex_factory_red_and_orange")
    red = hedging.news_moments(a, b, 15, "forex_factory_red")
    assert 0 < len(red) < len(both) and set(red[:, 1]) <= set(both[:, 1])
    lead30 = hedging.news_moments(a, b, 30, "forex_factory_red")
    assert (red[:, 1] - lead30[:, 0] == 30 * hedging.MINUTE_NS).all()
    m1 = path((0, 2001), (5, 2000))
    book = hedging.engine.bars_book(m1, "5m")
    f = hedging.MarketFacts(None, "fp", book, [], both[:0])
    assert f.with_news(hedging.Rules.now()) is f                                 # made under today's settings
    with settings.using({"news_events": "forex_factory_red", "news_minutes_before": 30}):
        g = f.with_news(hedging.Rules.now())
    assert g is not f and g.news_key == ("forex_factory_red", 30)


# ================================================================================================================
# the research tools: what a version may change in his management
# ================================================================================================================

def _hedge_form() -> Idea:
    return Idea.load(FORM)


def test_a_management_value_is_checked_with_its_reason():
    f = _hedge_form()
    assert "his management" in MG.check_value(f, "on_a_friday", "room", "exit.management.on_a_friday")
    cases = [
        ("on_a_friday", "skip", "exit.management.on_a_friday", "can be"),                 # a weekday slice
        ("in_a_zone_against_you", "avoid", "exit.management.in_a_zone_against_you", "can be"),
        ("how_many_covers_allowed", 0, "exit.management.how_many_covers_allowed", "can be"),
        ("how_much_room", -1.0, "exit.management.how_much_room", "above 0"),
        ("news_minutes_before", 7.5, "exit.management.news_minutes_before", "whole number"),
        ("trail_the_target", "off", "exit.management.trail_the_target", "FIXED management rule M23"),
        ("which_leg_may_close_while_covered", "profit", "exit.management.which_leg_may_close_while_covered", "M11"),
        ("close_one_side_early", True, "exit.management.close_one_side_early", "M10"),
        ("halfway_reversal", "off", "exit.management.halfway_reversal", "B2-05"),
        ("setup_near_level_adr", 0.2, "exit.management.setup_near_level_adr", "no code reads"),
        ("how_much_room", 2.0, "settings.how_much_room", "exit.management.how_much_room"),
        ("trendline_hedge_pad_adr", 0.1, "exit.management.trendline_hedge_pad_adr", "settings.trendline_hedge_pad_adr"),
        ("not_a_setting", 1, "exit.management.not_a_setting", "not one of his management"),
    ]
    for name, value, p, why in cases:
        with pytest.raises(MG.Refused, match=re.escape(why)):
            MG.check_value(f, name, value, p)
    stop = Idea.load(json.loads(FORM.read_text(encoding="utf-8")) | {"exit": {"kind": "stop", "at": "brk.extreme"}})
    with pytest.raises(MG.Refused, match="plain stop"):
        MG.check_value(stop, "on_a_friday", "room", "exit.management.on_a_friday")


def test_a_number_whose_switch_is_off_is_refused_it_would_give_the_same_trades():
    f = _hedge_form()
    with pytest.raises(MG.Refused, match="acts only when"):
        MG.check_needs(f, {"exit.management.a_small_hedge_is_within_adr": 0.2})
    MG.check_needs(f, {"exit.management.a_small_hedge_is_within_adr": 0.2,
                       "exit.management.a_real_stop_instead_of_the_hedge": "when_small"})
    with pytest.raises(MG.Refused, match="double_profit_reentry"):
        MG.check_needs(f, {"exit.management.double_profit_zone_rank": 1})


def test_resolved_puts_the_hedges_management_on_top_of_the_ideas_settings():
    from machine.ideas.form import apply_changes
    f = apply_changes(_hedge_form(), {"exit.management.on_a_friday": "room"}, "test")
    assert MG.resolved(f)["on_a_friday"] == "room" and MG.in_manager(
        {"exit.management.on_a_friday": "room", "settings.trendline_hedge_pad_adr": 0.1}) == {"on_a_friday": "room"}


def test_propose_version_takes_his_management_and_says_what_it_replaces(lab):
    make_idea("idea-101")
    T.start("idea-101")
    a = T.propose_version("idea-101", "his cap of hedges (B3-12)", readings=READ,
                          changes={"exit.management.how_many_covers_allowed": 3})
    assert a["new"] and "his management" in a["dimensions"]["exit.management.how_many_covers_allowed"]
    assert "version 1 is None" in a["about"] and "M21" in a["about"]
    b = T.propose_version("idea-101", "a Friday entry gets room", readings=READ,
                          changes={"exit.management.on_a_friday": "room"})
    assert b["version"] != a["version"]                                          # another rule: another version
    for changes, why in [({"exit.management.trail_the_target": "off"}, "FIXED management rule M23"),
                         ({"settings.trail_the_target": "off"}, "FIXED management rule M23"),
                         ({"settings.how_much_room": 2.0}, "exit.management.how_much_room"),
                         ({"settings.rearm_adr": 0.2}, "no code reads"),
                         ({"exit.management.strong_zone_hedge_rank": 1}, "acts only when")]:
        refused(T.propose_version, "idea-101", "x", readings=READ, changes=changes, match=re.escape(why))
    assert len(V.records("idea-101")) == 2                                       # refused proposals leave no record


def test_list_blocks_shows_the_management_versions_his_fixed_rows_and_the_rest():
    out = T.list_blocks()
    names = {m["name"] for m in out["management_a_version_may_change"]}
    assert names == set(MG.VERSIONS)
    assert {r["row"] for r in out["his_fixed_management"]} == set(rules.MANAGEMENT)
    assert set(out["management_not_a_version"]) == set(MG.OTHER)
    plain = {m["name"] for m in out["settings_a_version_may_change"]}
    assert plain.isdisjoint(MG.RETIRED) and plain.isdisjoint(MG.VERSIONS)


def closers_for(m1, every=37):
    """His setups that may close a hedged leg on the hand-made bars: a limit $0.50 away each way, by turns."""
    out = []
    for i, t in enumerate(m1.index[every - 1::every][:-2]):
        c, d = float(m1.close.loc[t]), ("buy", "sell")[i % 2]
        e = round(c - 0.5, 2) if d == "buy" else round(c + 0.5, 2)
        hedge, target = (e - 1, e + 1.5) if d == "buy" else (e + 1, e - 1.5)
        out.append(order(f"c{i}", 0, d, e, round(hedge, 2), round(target, 2), "idea-000").model_copy(
            update={"known_at": t + pd.Timedelta(minutes=1), "expiry_bars": 3}))
    return out


def hedged_bench(**over):
    facts = Market(M1, closers_for(M1), dial="mixed")                            # his hedge on the hand-made 2019 bars
    return bench(hedged=functools.partial(MG.research_run, facts=facts), **over)


def test_run_version_runs_a_management_version_with_his_hedge_beside_version_1(lab):
    make_idea("idea-101")
    T.start("idea-101")
    b = hedged_bench()
    vid = T.propose_version("idea-101", "the cap", readings=READ,
                            changes={"exit.management.how_many_covers_allowed": 1})["version"]
    out = T.run_version("idea-101", vid, bench=b)
    assert not out["already_run"] and "WITH HIS HEDGE" in out["how"] and out["management"] == {
        "how_many_covers_allowed": 1}
    v1 = out["version_1_of_his_management"]
    assert out["numbers"]["trades"] > 0 and v1["trades"] + v1["open_at_end"] > 0
    assert out["numbers"]["test_rows_acted"].get("cap_stops", 0) > 0 and not v1["test_rows_acted"]
    for y in out["by_year"].values():
        assert y["difference"] == pytest.approx(y["this"] - y["version_1"])
    files = [f for f in trials.load() if f["idea_id"] == "idea-101"]
    assert [f["kind"] for f in files] == ["hedging"] and files[0]["version_id"] == vid
    assert files[0]["settings"]["management"] == {"how_many_covers_allowed": 1}
    assert files[0]["data"]["gold_m1_content_sha256"] == "hand-made 2019"
    assert T.get_idea("idea-101")["version_budget"]["tried"] == 1                # it is a try of the family
    again = T.run_version("idea-101", vid, bench=b)
    assert again["already_run"] and len([f for f in trials.load() if f["idea_id"] == "idea-101"]) == 1
    rows = T.list_trials("idea-101")["trials"]
    assert rows[0]["kind"] == "hedging" and rows[0]["by_year"] and "hidden" not in rows[0]
    refused(T.run_version, "idea-101", vid, how="walk-forward", bench=b, match="WITH HIS HEDGE")


def test_write_report_leaves_management_versions_out_of_the_plain_stop_tune(lab):
    make_idea("idea-101")
    T.start("idea-101")
    T.propose_version("idea-101", "the cap", readings=READ, changes={"exit.management.how_many_covers_allowed": 1})
    refused(T.write_report, "idea-101", bench=hedged_bench(), match="every version of idea-101 changes his management")


# ================================================================================================================
# real gold (💻): the market questions his TEST rows ask, and every version runs on real setups
# ================================================================================================================

@pytest.mark.data
@pytest.mark.news
def test_on_real_gold_every_management_version_runs_and_acts():
    from machine.data import bars as BARS
    if not config.GOLD_M1.exists():
        pytest.skip("needs data/gold_m1.parquet")
    from machine.entries import forms
    from machine.tester import engine
    market = C.Market.load("2019-01-01", "2019-06-30", tfs=("1m", "5m", "1h", "1d", "1w"))
    book = engine.bars_book(market.bars("1m"), "5m")
    fp = settings.fingerprint(settings.resolve({}))
    comp = C.compile_idea(forms.TYPE1.idea(), market)
    facts = hedging.MarketFacts.build(market, fp, book, setups=hedging.setups_from(comp.signals, book))
    assert BARS.m1_fingerprint()                                                 # the real prices
    t = book.first + pd.Timedelta(days=40)
    i = market.closed_by("5m", t)
    px = float(market.bars("5m")["close"].iloc[i])
    z = facts.zone_edge("low", t, px, None, 0)
    assert z is None or (z["price"] < px and z["far"] <= z["price"])
    assert isinstance(facts.trend_line("sell", t), bool)
    sigs = comp.signals[:60]                         # each setup on its own: a hedged trade can last months (L19a)
    assert len(sigs) == 60

    def acted(changes) -> dict:
        out = dict.fromkeys(hedging.ACTED, 0)
        with settings.using(changes):
            for sg in sigs:
                for tr in hedging.replay([sg], facts, "5m").trades:
                    for k in hedging.ACTED:
                        out[k] += int(getattr(tr, k))
        return {k: v for k, v in out.items() if v}

    assert acted({}) == {}                                                       # version 1: none of them acts
    # measured on these 60 setups (Jan - Jun 2019): the stop, Friday, the cap, the split dial act often; his zones
    # stand between the entry and the target of 3 setups (any rank) and the hour AND the day were never both bearish
    # at any hourly sample - so the zone rules are run at rank 0 and the escape only has to run
    for changes, key in [({"a_real_stop_instead_of_the_hedge": "when_small"}, "real_stops"),
                         ({"on_a_friday": "room"}, "friday"), ({"how_many_covers_allowed": 1}, "cap_stops"),
                         ({"trail_when_hour_and_day_split": "hold"}, "split_holds"),
                         ({"double_profit_reentry": "retest", "double_profit_zone_rank": 0}, "dbl_booked"),
                         ({"strong_zone_as_hedge_point": "zone", "strong_zone_hedge_rank": 0}, "zone_hedges")]:
        assert acted(changes).get(key, 0) > 0, changes
    for changes in [{"close_the_losing_leg_when_a_trend_runs": True}, {"in_a_zone_against_you": "small"},
                    {"strong_zone_as_hedge_point": "nearer"}]:
        assert set(acted(changes)) <= set(hedging.ACTED)
