"""L19a - the hedging tester (machine/tester/hedging.py). Hand-made 1-minute bars and a hand-made market.

The bars start Monday 11 Mar 2019 10:00 server time (a TUNE year). `path` draws them through (minute, price) points
in straight lines, with no wicks: a price is touched exactly when the line crosses it. Spread 20 points ($0.20), so a
buy limit at L fills on the bid at L - 0.20 (the ask reaches L), and a sell's target T closes on the bid at T - 0.20.
Money is checked against machine.costs.pnl, the one money function (Rs 40 a position, all-in).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from machine import config, costs, settings
from machine.tester import Signal, engine, hedging
from machine.tester.hedging import MINUTE_NS

T0 = pd.Timestamp("2019-03-11 10:00")
SP = 0.20                                            # the bars' spread in dollars (20 points)
BIG = 10**6                                          # a limit that waits to the window's end


def at(minute: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minute)


def path(*points, tail: int = 30, spread: int = 20) -> pd.DataFrame:
    """1-min bars through (minute, price) points, straight between them; each bar opens where the last one closed.
    `tail` flat minutes after the last point."""
    mins = [m for m, _ in points]
    last = mins[-1] + tail
    px = np.interp(np.arange(last + 2), mins, [p for _, p in points])
    px = np.round(px, 2)
    o, c = px[:-1], px[1:]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "spread": spread, "vol": 1},
                        index=pd.date_range(T0, periods=len(o), freq="1min", name="server_time"))


def order(sid, minute, direction, entry, hedge, target, idea="idea-test") -> Signal:
    """A limit order known at `minute`: entry, its hedge point (the signal's stop) and one target."""
    return Signal(id=sid, known_at=at(minute), direction=direction, entry_type="limit", entry_price=entry,
                  expiry_bars=BIG, stop=hedge, targets=(target,), idea_id=idea, version_id="v1")


class Market:
    """A hand-made market for the manager: the dial, the liquidity levels, the ruler, the news, his setups."""

    def __init__(self, m1, setups=(), news=(), dial="mixed", levels=(), ruler=10.0, tf="5m", ath=False):
        self.book = engine.bars_book(m1, tf)
        self.setups = hedging.setups_from([s for s in setups if s.known_at <= m1.index[-1]], self.book)
        lead = int(settings.get("news_minutes_before")) * MINUTE_NS
        rel = np.array(sorted(pd.Timestamp(t).as_unit("ns").value for t in news), dtype=np.int64)
        self.news = np.stack([rel - lead, rel], axis=1) if len(rel) else np.zeros((0, 2), dtype=np.int64)
        self._dial, self.levels, self._ruler, self._ath = dial, list(levels), ruler, ath
        self.asked = []

    def dial(self, direction, t):
        self.asked.append(("dial", direction, t))
        return self._dial(direction, t) if callable(self._dial) else self._dial

    def next_level(self, side, beyond, t, max_away):
        got = [p for p, s, known in self.levels if s == side and at(known) <= t
               and ((p > beyond) if side == "high" else (p < beyond))
               and (max_away is None or abs(p - beyond) <= max_away)]
        return min(got, key=lambda p: abs(p - beyond)) if got else None

    def ruler(self, t):
        return self._ruler

    def at_all_time_high(self, t):
        return self._ath


def run(signals, market, tf="5m"):
    return hedging.replay(list(signals), market, tf)


def money(direction, a, b, ta, tb):
    return costs.pnl(direction, a, b, ta, tb, 0.01).net_inr


# ================================================================================================================
# the two-leg story: hedge, the no-close zone, a close at the confirmed price, the leg left, a close at a loss
# ================================================================================================================

def two_leg_story():
    """Distances are MT5 prices (a buy's price is the ask = the bid + $0.20); recorded prices are the bid.
    buy limit 2000.20 (fills: bid 2000.00; its current entry 2000.20; hedge 1998, target 2010) -> hedged: a sell at
    1998 (R = $2.20, 4R = $8.80) -> a buy setup at 1994.20 fills (3.80 past the sell's 1998: inside the zone, nothing)
    -> a buy setup at 1988.70 fills (9.30 past: the SELL closes, on the bid at 1988.50) -> the buy left is that setup's
    trade: current entry 1988.70, fixed hedge 1987.50, target max(1992, 1988.70 + 4 x 1.20 = 1993.50) -> hedged again
    at 1987.50 (R = 1.20, 4R = 4.80) -> a sell setup at 1995.50 fills (6.80 past 1988.70): the BUY closes there, at a
    LOSS against its 2000 open (#134) -> the sell left is that setup's trade (entry 1995.50, hedge 1996.50, target
    min(1990, 1995.50 - 4 x 1.00) = 1990) -> target 1990 (the ask): the sell closes on the bid at 1989.80."""
    m1 = path((0, 2001), (5, 2000), (10, 1998), (20, 1994), (30, 1988.5), (35, 1987.5), (50, 1995.5), (60, 1990),
              (65, 1989.5))
    idea = [order("buy1", 0, "buy", 2000.20, 1998, 2010)]
    setups = [order("S1", 12, "buy", 1994.20, 1993, 2000, "idea-000"),         # fills at 1994.00: inside the zone
              order("S2", 12, "buy", 1988.70, 1987.5, 1992, "idea-000"),       # fills at 1988.50: closes the sell
              order("S3", 36, "sell", 1995.50, 1996.5, 1990, "idea-000")]      # fills at 1995.50: closes the buy
    return m1, idea, setups


def test_the_two_leg_story_step_by_step():
    m1, idea, setups = two_leg_story()
    rep = run(idea, Market(m1, setups))
    (t,) = rep.trades
    assert t.status == "closed"
    legs = [(x.direction, x.why_open, x.open_px, x.close_px) for x in t.legs]
    assert legs == [("buy", "entry", 2000.0, 1995.5),            # closed by S3 at its fill price, at a loss
                    ("sell", "hedge", 1998.0, 1988.5),           # closed by S2 at its fill price (not S1: the zone)
                    ("sell", "hedge", 1987.5, 1989.8)]           # the 2nd hedge = the leg left: its target 1990
    assert [x.ref for x in t.legs] == [1988.7, 1998.0, 1995.5]   # each leg's current entry at its close (MT5)
    assert (t.hedges, t.closes_while_hedged, t.positions) == (2, 2, 3)
    assert [x.open_i for x in t.legs] == [4, 9, 34] and [x.close_i for x in t.legs] == [49, 29, 61]
    # the money: every position through costs.pnl, Rs 40 each, on the recorded (bid) prices
    want = (money("buy", 2000.0, 1995.5, at(4), at(49)) + money("sell", 1998.0, 1988.5, at(9), at(29))
            + money("sell", 1987.5, 1989.8, at(34), at(61)))
    assert t.net_inr == pytest.approx(want)
    assert t.cost_inr == 3 * config.COST_INR_PER_POSITION
    assert t.exit_reason == "target" and t.exit_ts == at(61)


def test_a_setup_inside_the_no_close_zone_closes_nothing():
    """S1 fills at 1994.20 (MT5), only $3.80 past the sell's 1998 (R = 2.20, 4R = 8.80): nothing closes (M12)."""
    m1, idea, setups = two_leg_story()
    rep = run(idea, Market(m1, setups[:1]))
    (t,) = rep.trades
    assert t.status == "open at end" and [x.close_i for x in t.legs] == [None, None]


def test_the_1_to_4_counts_from_the_current_entry_and_R_is_set_by_every_hedge():
    """After S2 the buy's current entry is 1988.70 (not 2000.20) and the new hedge sets R = 1.20: S3 at 1995.50 is
    6.80 past 1988.70 >= 4.80, so it closes the buy. Counted from the 2000.20 open it would be inside the zone."""
    m1, idea, setups = two_leg_story()
    (t,) = run(idea, Market(m1, setups)).trades
    story = [r for r in t.story if r["what"] in ("hedge", "close", "new trade")]
    assert [r["what"] for r in story] == ["hedge", "close", "new trade", "hedge", "close", "new trade", "close"]
    assert "4 x $2.20 = $8.80" in story[0]["words"] and "4 x $1.20 = $4.80" in story[3]["words"]


def test_R_is_measured_on_mt5_prices_so_a_spread_jump_does_not_shrink_it():
    """A sell's hedge is a BUY stop: it fires when the ASK reaches it. A $3.00 spread in one minute fires it while the
    bid is still far below - as in MT5 - but R stays the distance to the hedge price, not to the bid."""
    m1 = path((0, 1999), (5, 2000.2), (20, 2000.5), tail=20)
    m1.loc[at(10), "spread"] = 300
    (t,) = run([order("s", 0, "sell", 2000.0, 2002.0, 1990)], Market(m1)).trades
    hedge = t.legs[1]
    assert hedge.open_i == 10 and hedge.ref - hedge.open_px == pytest.approx(3.0)
    assert "4 x $2.00" not in [r for r in t.story if r["what"] == "hedge"][0]["words"]


def test_a_losing_leg_closes_on_the_setup_it_points_against():
    """#134: 'when the buy setup is formed the sell should have been closed right even if it is in loss'. The buy
    opened at 2000 closes at 1995.50: -$4.50 before cost."""
    m1, idea, setups = two_leg_story()
    (t,) = run(idea, Market(m1, setups)).trades
    buy = t.legs[0]
    assert buy.pnl.gross_usd == pytest.approx(-4.5) and buy.why_closed.startswith("a sell setup (S3)")


def test_the_leg_left_is_a_new_trade_with_that_setups_fixed_hedge_and_a_proper_target():
    """M15 / M16: after S2 closes the sell, the buy waits with S2's own hedge 1987.50 and a target of
    max(S2's 1992, 1988.70 + 4 x 1.20 = 1993.50), then the next liquidity at or past it (a level at 1996.00 here)."""
    m1, idea, setups = two_leg_story()
    mk = Market(m1, setups, levels=[(1996.0, "high", 0), (1993.0, "high", 0)])
    (t,) = run(idea, mk).trades
    new = [r for r in t.story if r["what"] == "new trade"][0]
    assert "new trade at 1988.70: its fixed hedge 1987.50, its target 1996.00" in new["words"]


# ================================================================================================================
# the confirmed price (lesson B20)
# ================================================================================================================

def test_a_closed_leg_is_credited_at_the_setups_fill_price_not_its_level():
    """The buy setup's limit is 1989.20 but price GAPS down past it: the minute opens at 1988.00. The sell leg closes
    at 1988.00 - the price really there (the tester's own fill) - never at 1989.20 or 1989.00."""
    m1 = path((0, 2001), (5, 2000), (10, 1998), (20, 1990), (30, 1990))
    m1.loc[at(25):, ["open", "high", "low", "close"]] = 1988.0          # a gap down at 10:25, then flat
    m1.loc[at(24), ["close"]] = 1990.0
    idea = [order("buy1", 0, "buy", 2000.20, 1998, 2010)]
    setup = order("S", 21, "buy", 1989.20, 1987.0, 1992, "idea-000")
    mk = Market(m1, [setup])
    fill = mk.book.fill(setup, pd.Timestamp(setup.known_at).as_unit("ns").value)
    assert (fill.i, fill.px, fill.at_open) == (25, 1988.0, True)
    (t,) = run(idea, mk).trades
    sell = t.legs[1]
    assert (sell.close_i, sell.close_px) == (25, 1988.0)
    assert sell.pnl.gross_usd == pytest.approx(10.0)                    # 1998 - 1988, not 1998 - 1989.20


def test_a_setup_known_before_the_hedge_opened_does_not_close_a_leg():
    """Only setups known from the hedge's minute on count (Claude's reading, as the old engine): S is known at
    10:02, before the hedge (10:09); it fills at 1989 while hedged, but closes nothing."""
    m1, idea, _ = two_leg_story()
    early = order("S", 2, "buy", 1989.20, 1987.5, 1992, "idea-000")
    (t,) = run(idea, Market(m1, [early])).trades
    assert t.closes_while_hedged == 0 and t.status == "open at end"


# ================================================================================================================
# the window end
# ================================================================================================================

def test_a_trade_open_at_the_window_end_is_reported_apart_and_never_counted():
    """Hedged at 1998 and never closed: open at the end - no money, even though nothing is missing."""
    m1 = path((0, 2001), (5, 2000), (10, 1998), (20, 1996))
    rep = run([order("buy1", 0, "buy", 2000.20, 1998, 2010)], Market(m1))
    (t,) = rep.trades
    assert t.status == "open at end" and t.net_inr is None and t.exit_ts is None
    s = hedging.summarize(rep)
    assert (s["trades"], s["open_at_end"], s["net_inr_total"]) == (0, 1, 0)
    assert s["open_at_end_ids"] == ["buy1"]


def test_legs_closed_inside_a_trade_open_at_the_end_are_not_counted_either():
    """The sell closed at 1989 (+$9) but the buy left is still open at the end: the TRADE is open - nothing counts."""
    m1, idea, setups = two_leg_story()
    rep = run(idea, Market(m1.loc[:at(40)], setups[:2]))
    (t,) = rep.trades
    assert t.status == "open at end" and t.legs[1].pnl is not None
    assert hedging.summarize(rep)["net_inr_total"] == 0
    assert "OPEN AT END (not counted in Rs): 1" in hedging.summary_text(hedging.summarize(rep))


def test_the_next_signal_waits_for_the_trade_to_end():
    """One trade at a time (M18): a signal known while the trade is on is skipped, even while it is hedged."""
    m1, idea, setups = two_leg_story()
    late = order("buy2", 40, "buy", 1990.20, 1985, 2005)
    rep = run(idea + [late], Market(m1, setups))
    assert rep.skipped == ["buy2"] and len(rep.trades) == 1


def test_a_waiting_limit_is_cancelled_by_a_newer_signal():
    """M39: buy1's limit 1990.20 still waits at 10:03 when buy2 comes: buy1 is cancelled, buy2 is taken."""
    m1 = path((0, 2001), (5, 2000), (15, 2010))
    a = order("buy1", 0, "buy", 1990.20, 1985, 2010)
    b = order("buy2", 3, "buy", 2000.20, 1998, 2005)
    rep = run([a, b], Market(m1))
    assert rep.not_filled == [("buy1", "cancelled")] and [t.signal_id for t in rep.trades] == ["buy2"]


# ================================================================================================================
# one leg: target, room, trailing, news, the same minute
# ================================================================================================================

def test_straight_to_the_target_is_one_position():
    m1 = path((0, 2001), (5, 2000), (15, 2010))
    (t,) = run([order("buy1", 0, "buy", 2000.20, 1998, 2008)], Market(m1)).trades
    assert (t.status, t.positions, t.hedges, t.legs[0].close_px) == ("closed", 1, 0, 2008.0)
    assert t.net_inr == pytest.approx(money("buy", 2000.0, 2008.0, at(4), at(12)))


def test_room_the_hour_and_day_with_the_trade_the_hedge_waits_at_the_next_liquidity():
    """M06: dial 'with' -> the hedge waits at the next liquidity below the hedge point (1995), not at 1998."""
    m1 = path((0, 2001), (5, 2000), (10, 1998), (12, 1997), (15, 2008))
    idea = [order("buy1", 0, "buy", 2000.20, 1998, 2008)]
    (t,) = run(idea, Market(m1, dial="with", levels=[(1995.0, "low", 0)])).trades
    assert t.room and t.hedges == 0 and t.status == "closed"
    (t2,) = run(idea, Market(m1, dial="mixed", levels=[(1995.0, "low", 0)])).trades
    assert not t2.room and t2.hedges == 1                         # without room it is hedged at 1998


def test_room_with_no_liquidity_there_is_one_days_range():
    """M07: no liquidity below: the hedge waits how_much_room (1.0) x a day's range (the ruler, $10) past 1998."""
    m1 = path((0, 2001), (5, 2000), (10, 1989), (15, 1987))
    (t,) = run([order("buy1", 0, "buy", 2000.20, 1998, 2008)], Market(m1, dial="with")).trades
    assert t.room and t.legs[1].open_px == 1988.0


def test_trailing_with_the_hour_and_day_holds_to_the_next_liquidity_then_an_opposite_setup_closes():
    """M23 / M25: at the target 2008 the dial is 'with': hold to 2012; an opposite (sell) setup known after the
    trail began fills at 2010.00: the buy closes there, at the setup's fill price."""
    m1 = path((0, 2001), (5, 2000), (15, 2008.5), (20, 2011), (30, 2009))
    setups = [order("S", 18, "sell", 2010.0, 2011.5, 2005, "idea-000")]
    (t,) = run([order("buy1", 0, "buy", 2000.20, 1998, 2008)],
               Market(m1, setups, dial=lambda d, t: "with" if t >= at(10) else "mixed",
                      levels=[(2012.0, "high", 0)])).trades
    assert t.trail_steps == 1 and t.exit_reason == "opposite setup after trailing"
    assert t.legs[0].close_px == 2010.0


def test_trailing_against_closes_at_the_first_target():
    m1 = path((0, 2001), (5, 2000), (15, 2008.5), (20, 2011))
    (t,) = run([order("buy1", 0, "buy", 2000.20, 1998, 2008)],
               Market(m1, dial=lambda d, t: "against" if t >= at(10) else "mixed",
                      levels=[(2012.0, "high", 0)])).trades
    assert t.trail_steps == 0 and t.legs[0].close_px == 2008.0


def test_news_closes_a_leg_in_profit_and_hedges_a_leg_in_a_loss():
    """M28: news at 10:40 -> the rule acts 15 minutes before, at the open of 10:25."""
    up = path((0, 2001), (5, 2000), (25, 2004), (40, 2006))
    (t,) = run([order("b", 0, "buy", 2000.20, 1990, 2010)], Market(up, news=[at(40)])).trades
    assert (t.news_acts, t.exit_reason, t.legs[0].close_i, t.legs[0].close_px) == (1, "news, in profit", 25, 2004.0)
    down = path((0, 2001), (5, 2000), (25, 1998), (40, 1997))
    (t,) = run([order("b", 0, "buy", 2000.20, 1990, 2010)], Market(down, news=[at(40)])).trades
    assert (t.news_acts, t.hedges) == (1, 1) and t.legs[1].why_open == "news hedge" and t.legs[1].open_px == 1998.0


def test_a_leg_up_less_than_its_cost_is_not_in_profit_at_news():
    """+$0.30 is less than Rs 40: not in profit - hedged, not closed (Claude's reading of 'in profit')."""
    m1 = path((0, 2001), (5, 2000), (25, 2000.3), (40, 2000.3))
    (t,) = run([order("b", 0, "buy", 2000.20, 1990, 2010)], Market(m1, news=[at(40)])).trades
    assert t.hedges == 1 and t.legs[1].open_px == 2000.3


def test_the_hedge_and_the_target_in_one_minute_the_hedge_counts():
    m1 = path((0, 2001), (5, 2000), (8, 2000), (40, 2000))
    m1.loc[at(8), ["high", "low"]] = [2009.0, 1997.0]
    (t,) = run([order("b", 0, "buy", 2000.20, 1998, 2008)], Market(m1)).trades
    assert t.hedges == 1 and t.ambiguous and "the hedge counted" in t.ambiguous[0]


def test_the_hedge_price_counts_in_the_minute_the_leg_opened():
    """The buy fills in minute 4 (after its open) and price goes on down to 1998 in that same minute: hedged there."""
    m1 = path((0, 2001), (4, 2001), (5, 1997), (15, 1996))
    (t,) = run([order("b", 0, "buy", 2000.20, 1998, 2008)], Market(m1)).trades
    assert t.legs[0].open_i == 4 and t.legs[1].open_i == 4 and t.legs[1].open_px == 1998.0


# ================================================================================================================
# the whole trade decides (M17)
# ================================================================================================================

def test_a_last_leg_ending_red_waits_while_the_whole_trade_is_red():
    """buy limit 2000.20 (bid 2000.00) hedged at 1999 (R = 1.20, 4R = 4.80): a buy setup at 1994.00 fills (5.00 past
    the sell: the sell closes on the bid at 1993.80, +$5.20). The buy left (opened 2000.00) is a new trade at 1994.00
    with its hedge 1993.70 and a target of max(1994.50, 1994.00 + 4 x 0.30) = 1995.20 (the level at 1998 is 2.80 away;
    'the next liquidity' reaches 2.5 x the $1.00 ruler). At 1995.20 the buy would end at -$4.80 and the whole trade
    red: NOT closed (M17). Its target moves to the next liquidity past the whole trade's break-even price: 1998."""
    m1 = path((0, 2001), (5, 2000), (8, 1999), (15, 1993.8), (25, 1996.5), (35, 1998.5))
    setups = [order("S", 10, "buy", 1994.00, 1993.7, 1994.5, "idea-000")]
    (t,) = run([order("b", 0, "buy", 2000.20, 1999, 2010)],
               Market(m1, setups, levels=[(1998.0, "high", 0)], ruler=1.0)).trades
    refused = [r for r in t.story if r["what"] == "not yet"]
    assert t.whole_trade_refusals == 1 and "target 1995.20 -> 1998.00" in refused[0]["words"]
    assert t.status == "closed" and t.legs[0].close_px == 1998.0
    assert t.legs[0].pnl.net_inr < 0 < t.net_inr                  # the leg ends red, the whole trade green


# ================================================================================================================
# cut off the future
# ================================================================================================================

def _random_world(seed: int, minutes: int = 6000):
    """Random 1-minute bars that swing around 2000 (so trades end), 100 random signals, 500 random setups, 4 news."""
    rng = np.random.default_rng(seed)
    x = np.zeros(minutes)
    for k in range(1, minutes):
        x[k] = 0.995 * x[k - 1] + rng.normal(0, 0.3)
    px = np.round(2000 + x, 2)
    o = np.r_[2000.0, px[:-1]]
    hi = np.maximum(o, px) + np.round(np.abs(rng.normal(0, 0.1, minutes)), 2)
    lo = np.minimum(o, px) - np.round(np.abs(rng.normal(0, 0.1, minutes)), 2)
    m1 = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": px, "spread": 20, "vol": 1},
                      index=pd.date_range(T0, periods=minutes, freq="1min", name="server_time"))

    def orders(n, prefix, idea):
        out = []
        for k in range(n):
            minute = int(rng.integers(5, minutes - 5))
            ref = float(m1["close"].iloc[minute - 1])
            d = "buy" if rng.random() < 0.5 else "sell"
            s = 1 if d == "buy" else -1
            e = round(ref - s * float(rng.uniform(0.2, 1.0)), 2)
            h = round(e - s * float(rng.uniform(0.3, 1.2)), 2)
            tg = round(e + s * float(rng.uniform(0.8, 3.0)), 2)
            out.append(order(f"{prefix}{k}", minute, d, e, h, tg, idea))
        return out
    return m1, orders(100, "i", "idea-test"), orders(500, "S", "idea-000"), [at(m) for m in (400, 1300, 2200, 4000)]


def _world_market(m1, setups, news):
    dial = lambda d, t: ("with", "against", "mixed")[t.minute % 3]            # changes with time, known at t
    levels = [(round(2000 + k / 2, 2), "high" if k > 0 else "low", 120 * abs(k)) for k in range(-16, 17) if k]
    return Market(m1, setups, news, dial, levels, ruler=1.0)


def _key(t: hedging.HedgedTrade):
    return (t.signal_id, t.status, t.fill_ts, [(x.direction, x.open_i, x.open_px, x.close_i, x.close_px)
                                               for x in t.legs], t.net_inr)


CUTS = (900, 1800, 2700, 3600, 4500, 5400)


@pytest.mark.parametrize("seed", [3, 5, 9, 14])
def test_cut_off_the_future_leaves_every_closed_trade_the_same(seed):
    """Replay the whole window, then only the bars up to each cut: every trade CLOSED before the cut is the same,
    leg by leg and in money; every trade still open at the cut was open (or closed later) in the whole run. The
    worlds were picked so the replays hedge, close legs while hedged, refuse a close (M17), trail and meet news."""
    m1, idea, setups, news = _random_world(seed)
    full = run(idea, _world_market(m1, setups, news))
    closed = [t for t in full.trades if t.status == "closed"]
    assert len(closed) >= 3 and sum(t.closes_while_hedged for t in closed) >= 4
    by_id = {t.signal_id: t for t in full.trades}
    compared = 0
    for cut in CUTS:
        part = m1.iloc[:cut]
        end = part.index[-1]
        got = run([s for s in idea if s.known_at <= end], _world_market(part, setups, news))
        for t in got.trades:
            if t.status == "closed":
                assert _key(t) == _key(by_id[t.signal_id]), f"cut {end}: {t.signal_id} changed"
                compared += 1
            else:
                f = by_id[t.signal_id]
                assert f.fill_ts == t.fill_ts and (f.exit_ts is None or f.exit_ts >= end)
        for f in full.trades:                            # every trade closed before the cut is there too
            if f.status == "closed" and f.exit_ts < end:
                assert f.signal_id in {t.signal_id for t in got.trades}
    assert compared > 0


def test_a_manager_that_peeks_is_caught(monkeypatch):
    """The comparison above catches a peek: a manager that credits every close at the window's LAST price changes
    trades closed long before a cut."""
    m1, idea, setups, news = _random_world(3)
    real = hedging._Manager.close

    def peeking(self, leg, m, px, why):
        return real(self, leg, m, float(self.b.o[self.b.n - 1]), why)

    monkeypatch.setattr(hedging._Manager, "close", peeking)
    full = {t.signal_id: _key(t) for t in run(idea, _world_market(m1, setups, news)).trades if t.status == "closed"}
    changed = 0
    for cut in CUTS:
        part = m1.iloc[:cut]
        end = part.index[-1]
        for t in run([s for s in idea if s.known_at <= end], _world_market(part, setups, news)).trades:
            if t.status == "closed" and full.get(t.signal_id) != _key(t):
                changed += 1
    assert changed > 0


# ================================================================================================================
# the real market (needs the price data)
# ================================================================================================================

@pytest.fixture(scope="module")
def march_2019():
    from machine.data.bars import load_bars
    from machine.ideas.compile import Market
    market = Market.load("2019-03", "2019-03")
    book = engine.bars_book(load_bars("1m", "2019-03", "2019-03"), "5m")
    fp = settings.fingerprint(settings.resolve({}))
    return hedging.MarketFacts.build(market, fp, book, setups=[], news=np.zeros((0, 2), dtype=np.int64))


@pytest.mark.data
@pytest.mark.parametrize("recent_first", [True, False])
def test_the_fast_liquidity_lookup_is_the_compilers_own(march_2019, recent_first):
    """MarketFacts.next_level (numpy) = compile.next_liquidity (pandas) on 600 random questions, March 2019."""
    from machine.ideas.compile import next_liquidity
    f = march_2019
    rng = np.random.default_rng(7)
    closes = f.market.bars("5m")["close"].to_numpy()
    times = f.market.bars("5m").index
    same = found = 0
    with settings.using({"the_most_recent_liquidity_comes_first": recent_first}):
        for _ in range(600):
            k = int(rng.integers(300, len(times) - 1))
            t = times[k] + pd.Timedelta(minutes=int(rng.integers(0, 30)))
            side = "high" if rng.random() < 0.5 else "low"
            beyond = float(closes[k]) + float(rng.normal(0, 3))
            away = None if rng.random() < 0.2 else float(rng.uniform(1, 30))
            want = next_liquidity(f.market, f.fp, t, side, beyond, away)
            got = f.next_level(side, beyond, t, away)
            same += want == got
            found += want is not None
    assert same == 600 and found > 300


@pytest.mark.data
def test_his_setups_on_real_bars_fill_after_they_are_known(march_2019):
    """His grab + gap setups (the moments a leg may close), March 2019: each fills only after it was known, on the
    tester's own fill rules; its hedge point is on the losing side of its fill and its target on the winning side."""
    from machine.ideas.compile import compile_idea
    from machine.ideas.form import Idea
    f = march_2019
    sig = compile_idea(Idea.load(hedging.HIS_GRAB_GAP), f.market).signals
    got = hedging.setups_from(sig, f.book)
    assert len(sig) > 30 and len(got) > 20
    for s in got:
        assert f.book.t[s.fill_i] >= s.known
        y = hedging.SIGN[s.direction]
        assert y * (s.target - s.fill_px) > 0


# ================================================================================================================
# the table is the only source
# ================================================================================================================

def test_every_row_of_his_management_table_is_listed():
    rows = re.findall(r"^\| (M\d\d) \|", (config.ROOT / hedging.TABLE).read_text(encoding="utf-8"), re.M)
    assert rows and [r for r, *_ in hedging.RULES] == rows


def test_a_management_version_it_cannot_run_is_refused_not_run_as_something_else():
    # 'any' (no 1:4 rule) was never his and is not built; #105 'profit' runs since L19c (a labelled comparison)
    with settings.using({"which_leg_may_close_while_covered": "any"}):
        with pytest.raises(NotImplementedError, match="management.py"):
            hedging.Rules.now()


def test_his_numbers_are_the_defaults():
    r = hedging.Rules.now()
    assert (r.min_rr, r.room, r.trail, r.whole, r.leftover_target, r.news) == (4.0, "hd", "hd", "trade", "his124",
                                                                              "profit_close")
