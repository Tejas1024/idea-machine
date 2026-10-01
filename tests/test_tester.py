"""L3 — the tester (machine/tester). Hand-made 1-minute bars: every price and every minute is written out.

Hand-made bars start Monday 11 Mar 2019 10:00 server time (a TUNE year). Spreads are in points (1 point = $0.01).
Money checks call machine.costs.pnl, the one money function, with the same prices and times.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from machine import config, costs, trials
from machine.data import bars
from machine.guards import CheckYearsClosedError, SealedYearsError, assert_no_future
from machine.tester import Signal, known_at_close, run
from machine.tester.engine import replay
from machine.tester.run import summarize

T0 = pd.Timestamp("2019-03-11 10:00")          # a Monday


def at(minute: int) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minute)


def mk(rows, start=T0, spread=20) -> pd.DataFrame:
    """1-min bars, one per minute from start. A row: (open, high, low, close) or (open, high, low, close, spread)."""
    recs = []
    for r in rows:
        o, h, lo, c, *sp = r
        recs.append({"open": o, "high": h, "low": lo, "close": c, "spread": sp[0] if sp else spread, "vol": 1})
    return pd.DataFrame(recs, index=pd.date_range(start, periods=len(rows), freq="1min", name="server_time"))


def flat(price: float, n: int, spread=20) -> list:
    """n quiet minutes around price: high +$0.05, low -$0.05."""
    return [(price, price + 0.05, price - 0.05, price, spread)] * n


def sig(**kw) -> Signal:
    base = dict(id="s1", known_at=at(0), direction="buy", idea_id="idea-test", version_id="v1")
    base.update(kw)
    return Signal(**base)


def play(signals, m1, tf="5m", k=1):
    return replay(list(signals), m1, tf, k)


# ================================================================================================================
# fills
# ================================================================================================================

def test_fills_only_from_the_next_bar():
    """The 5-min bar 10:00-10:04 makes the signal; it is known at its close, 10:05.
    The buy fills at the OPEN of 10:05 (2002) — not at the signal bar's close (2001), not inside the signal bar."""
    m1 = mk(flat(2000, 4) + [(2000, 2001, 1999.9, 2001)] + flat(2002, 10))
    known = known_at_close(T0, "5m")
    assert known == at(5)
    t = play([sig(known_at=known, stop=1990, targets=(2010,))], m1).trades[0]
    assert (t.fill_ts, t.fill_px, t.fill_at_open) == (at(5), 2002.0, True)


def test_a_limit_touched_only_inside_the_signal_bar_does_not_fill():
    """Buy limit 1995, known 10:05, waits 1 bar (10:05-10:09). Price touched 1990 at 10:02 — BEFORE the signal was
    known — and never again: not filled, "expired" (the 10:10 bar exists, so the wait really ended)."""
    rows = flat(2002, 15)
    rows[2] = (2002, 2002, 1990, 2002, 20)
    rep = play([sig(known_at=at(5), entry_type="limit", entry_price=1995, stop=1990, targets=(2010,),
                    expiry_bars=1)], mk(rows))
    assert rep.trades == [] and rep.not_filled == [("s1", "expired")]


def test_expiry_counts_bars_of_the_timeframe():
    """Buy limit 1999, known 10:05, expiry 2 bars of 5 min = 10:05-10:14. Price reaches it at 10:14 → filled.
    The same price at 10:15 (the third bar) → expired."""
    for minute, filled in ((14, True), (15, False)):
        rows = flat(2002, 25, spread=10)
        rows[minute] = (2002, 2002, 1998.5, 2002, 10)        # ask low = 1998.50 + 0.10 = 1998.60 <= 1999
        rep = play([sig(known_at=at(5), entry_type="limit", entry_price=1999, stop=1990, targets=(2010,),
                        expiry_bars=2)], mk(rows))
        assert bool(rep.trades) is filled, minute
        if not filled:
            assert rep.not_filled == [("s1", "expired")]


def test_gap_fills_at_the_open():
    """Buy limit 2000: the 10:01 bar OPENS at 1995 (a gap down from 2003) → filled at 1995, the open.
    Sell limit 2000: the 10:01 bar opens at 2005 (a gap up from 1997) → filled at 2005, the open."""
    buy = play([sig(entry_type="limit", entry_price=2000, stop=1980, targets=(2020,), expiry_bars=3)],
               mk([(2003, 2003.2, 2002.8, 2003)] + [(1995, 1996, 1994.5, 1995.5)] + flat(1995.5, 3)), tf="1m")
    assert (buy.trades[0].fill_ts, buy.trades[0].fill_px, buy.trades[0].fill_at_open) == (at(1), 1995.0, True)
    sell = play([sig(direction="sell", entry_type="limit", entry_price=2000, stop=2020, targets=(1980,),
                     expiry_bars=3)],
                mk([(1997, 1997.2, 1996.8, 1997)] + [(2005, 2005.5, 2004.5, 2005)] + flat(2005, 3)), tf="1m")
    assert (sell.trades[0].fill_ts, sell.trades[0].fill_px) == (at(1), 2005.0)


def test_buy_limit_fires_on_the_ask():
    """Buy limit 2000. The bid's low is 1999.90 at 10:01. The ask = bid + that minute's spread.
      spread 20 points ($0.20): ask low 2000.10 > 2000 → NOT filled
      spread 10 points ($0.10): ask low 2000.00 <= 2000 → filled, recorded at the bid then: 2000 - 0.10 = 1999.90
      spread 0 (the data says 0): the floor MIN_SPREAD_POINTS = 14 → ask low 2000.04 → NOT filled"""
    assert config.MIN_SPREAD_POINTS == 14.0
    for spread, fill in ((20, None), (10, 1999.90), (0, None)):
        m1 = mk([(2000.5, 2000.6, 2000.4, 2000.5, 20), (2000.5, 2000.6, 1999.9, 2000.2, spread)]
                + flat(2000.5, 5))
        rep = play([sig(entry_type="limit", entry_price=2000, stop=1990, targets=(2010,), expiry_bars=5)], m1,
                   tf="1m")
        got = rep.trades[0].fill_px if rep.trades else None
        assert got == (pytest.approx(fill) if fill else None), spread


def test_sell_limit_fires_on_the_bid():
    """Sell limit 2000: the bid's high reaches exactly 2000.00 at 10:01, spread 50 points — filled at 2000 (a sell
    sells at the bid, so the spread does not matter)."""
    m1 = mk([(1999.5, 1999.6, 1999.4, 1999.5), (1999.5, 2000.0, 1999.4, 1999.8, 50)] + flat(1999.8, 3))
    rep = play([sig(direction="sell", entry_type="limit", entry_price=2000, stop=2010, targets=(1990,),
                    expiry_bars=3)], m1, tf="1m")
    assert (rep.trades[0].fill_ts, rep.trades[0].fill_px, rep.trades[0].fill_at_open) == (at(1), 2000.0, False)


def test_a_sells_stop_fires_on_the_ask():
    """(Claude's reading: a sell is closed by BUYING, at the ask.) Sell at the 10:00 open, 2000; stop 2005.
    At 10:02 the bid's high is 2004.85, spread 20 points → ask 2005.05 >= 2005 → stopped, recorded at the bid
    then: 2005 - 0.20 = 2004.80. Gold: sold 2000, bought back 2004.80 on the bid scale = -$4.80."""
    m1 = mk(flat(2000, 2) + [(2001, 2004.85, 2000.9, 2003, 20)] + flat(2003, 3))
    t = play([sig(direction="sell", stop=2005, targets=(1990,))], m1, tf="1m").trades[0]
    assert (t.exit_reason, t.exit_ts, t.exit_px) == ("stop", at(2), pytest.approx(2004.80))
    assert t.gross_usd == pytest.approx(-4.80)


# ================================================================================================================
# money
# ================================================================================================================

def test_net_rupees_come_from_costs_py():
    """Buy at the 10:00 open 2000, target 2010 reached at 10:03 → +$10 at 0.01 lot.
    Net = costs.pnl(...).net_inr exactly: $10 x Rs/$ of March 2019 - Rs 40. At 0.02 lot the cost is Rs 80."""
    rows = flat(2000, 3) + [(2001, 2010.5, 2000.9, 2009)] + flat(2009, 2)
    for lots, cost in ((0.01, 40.0), (0.02, 80.0)):
        t = play([sig(stop=1990, targets=(2010,), lots=lots)], mk(rows), tf="1m").trades[0]
        want = costs.pnl("buy", 2000.0, 2010.0, at(0), at(3), lots)
        assert (t.fill_px, t.exit_px, t.exit_reason) == (2000.0, 2010.0, "target")
        assert t.cost_inr == cost == want.cost_inr
        assert t.net_inr == pytest.approx(want.net_inr)
        assert t.gross_usd == pytest.approx(10.0 * lots / 0.01)
        assert t.usd_inr == costs.usd_inr("2019-03")


def test_two_targets_split_the_lots():
    """0.02 lot, targets 2005 and 2010, stop 1995. 2005 is reached at 10:02 (0.01 lot closes there), then the stop at
    10:04 closes the other 0.01. Gold: +$5 + (-$5) = $0; cost 2 x Rs 40 = Rs 80 (one 0.02 position)."""
    rows = flat(2000, 2) + [(2001, 2005.2, 2000.9, 2004)] + [(2004, 2004.1, 2000, 2000)] + \
        [(2000, 2000.1, 1994, 1995)] + flat(1995, 2)
    t = play([sig(stop=1995, targets=(2010, 2005), lots=0.02)], mk(rows), tf="1m").trades[0]
    assert [(e.reason, e.px, e.lots, e.ts) for e in t.exits] == [("target", 2005.0, 0.01, at(2)),
                                                                ("stop", 1995.0, 0.01, at(4))]
    assert t.gross_usd == pytest.approx(0.0) and t.cost_inr == 80.0
    assert t.exit_reason == "target+stop"


# ================================================================================================================
# exits
# ================================================================================================================

def test_both_in_one_5min_bar_is_settled_by_its_1min_bars():
    """Buy at the 10:05 open, 2000; stop 1995, target 2005. The 5-min bar 10:05-10:09 touches BOTH (high 2005.5,
    low 1994.5). Its 1-min bars say which came first:
      target at 10:06, stop at 10:08 → the target (+$5)      stop at 10:06, target at 10:08 → the stop (-$5)
    Neither is ambiguous: the minutes tell the order."""
    up = (2001, 2005.5, 2000.9, 2003)
    down = (2001, 2001.1, 1994.5, 1996)
    for first, second, reason, gold in ((up, down, "target", 5.0), (down, up, "stop", -5.0)):
        rows = flat(2000, 5) + [(2000, 2000.1, 1999.9, 2001), first, (2001, 2001.1, 2000.9, 2001), second] + \
            flat(2001, 1)
        m1 = mk(rows)
        five = bars.aggregate(m1, "5m").loc[at(5)]
        assert five.high >= 2005 and five.low <= 1995                      # both inside ONE 5-min bar
        t = play([sig(known_at=at(5), stop=1995, targets=(2005,))], m1).trades[0]
        assert (t.exit_reason, t.gross_usd, t.ambiguous) == (reason, pytest.approx(gold), False)


def test_both_in_one_minute_is_the_stop_and_counted():
    """Buy at 2000; stop 1995, target 2005. The 10:02 minute touches both (high 2006, low 1994): the order inside
    one minute is unknown → the stop (the worse), counted "ambiguous"."""
    m1 = mk(flat(2000, 2) + [(2000, 2006, 1994, 2000)] + flat(2000, 2))
    rep = play([sig(stop=1995, targets=(2005,))], m1, tf="1m")
    t = rep.trades[0]
    assert (t.exit_reason, t.exit_px, t.ambiguous) == ("stop", 1995.0, True)
    assert "counted as the stop" in t.ambiguous_why
    assert summarize(rep)["ambiguous"] == 1


def test_fill_minute_of_a_limit_the_target_is_not_counted():
    """Buy limit 2000 (spread 10 points), filled inside the 10:01 minute (it opened at 2003, fell to 1999.5).
    That minute's high 2006 is past the target 2005 — but it may have come BEFORE the fill. Not counted;
    counted "ambiguous". The trade goes on and is stopped at 10:03 (1995)."""
    m1 = mk([(2003, 2003.1, 2002.9, 2003, 10), (2003, 2006, 1999.5, 2001, 10), (2001, 2001.1, 2000.9, 2001, 10),
             (2001, 2001.1, 1994, 1995, 10)] + flat(1995, 2))
    t = play([sig(entry_type="limit", entry_price=2000, stop=1995, targets=(2005,), expiry_bars=3)], m1,
             tf="1m").trades[0]
    assert (t.fill_ts, t.fill_px, t.fill_at_open) == (at(1), pytest.approx(1999.90), False)
    assert (t.exit_reason, t.exit_ts, t.ambiguous) == ("stop", at(3), True)
    assert "target was not counted" in t.ambiguous_why


def test_fill_minute_of_a_limit_the_stop_counts():
    """Buy limit 2000 filled inside 10:01 (opened 2003); the same minute falls to 1994, past the stop 1995. Price had
    to pass the fill on its way down, so the stop came after the fill: stopped in the fill minute, at 1995."""
    m1 = mk([(2003, 2003.1, 2002.9, 2003, 10), (2003, 2003.2, 1994, 1996, 10)] + flat(1996, 2))
    t = play([sig(entry_type="limit", entry_price=2000, stop=1995, targets=(2010,), expiry_bars=3)], m1,
             tf="1m").trades[0]
    assert (t.fill_ts, t.exit_ts, t.exit_reason, t.exit_px, t.ambiguous) == (at(1), at(1), "stop", 1995.0, False)


def test_a_fill_already_past_the_stop_closes_at_once():
    """Buy at the next open with its stop at 1995 — but the 10:00 bar opens at 1993 (a gap). Closed at once at the
    fill price: gold $0, cost Rs 40."""
    m1 = mk([(1993, 1993.5, 1992.5, 1993)] + flat(1993, 3))
    t = play([sig(stop=1995, targets=(2010,))], m1, tf="1m").trades[0]
    assert (t.exit_reason, t.exit_ts, t.gross_usd, t.cost_inr) == ("past at fill", at(0), 0.0, 40.0)


def test_a_gap_through_the_stop_closes_at_the_open():
    """Buy at 2000, stop 1995. The 10:02 bar opens at 1990 (a gap down): the stop fills at the open, 1990 — worse
    than the stop price, as a real stop order would."""
    m1 = mk(flat(2000, 2) + [(1990, 1991, 1989, 1990)] + flat(1990, 2))
    t = play([sig(stop=1995, targets=(2010,))], m1, tf="1m").trades[0]
    assert (t.exit_reason, t.exit_px, t.gross_usd) == ("stop", 1990.0, pytest.approx(-10.0))


def test_levels_from_the_fill():
    """levels="from_fill": stop 5 and target 5 are dollars away from the fill. Buy filled at 2000.37 →
    stop 1995.37, target 2005.37."""
    m1 = mk([(2000.37, 2000.5, 2000.2, 2000.4)] + [(2000.4, 2005.37, 2000.3, 2005)] + flat(2005, 2))
    t = play([sig(levels="from_fill", stop=5.0, targets=(5.0,))], m1, tf="1m").trades[0]
    assert (t.stop_px, t.target_pxs) == (1995.37, (2005.37,))
    assert (t.exit_reason, t.gross_usd) == ("target", pytest.approx(5.0))


# ================================================================================================================
# one trade at a time
# ================================================================================================================

def test_signals_during_an_open_trade_are_skipped_and_counted():
    """Signal A (known 10:00) buys at 2000; it is stopped during the 10:04 minute, so its place is free from 10:05.
    B (known 10:02) and C (known 10:04) come while A is on → skipped. D (known 10:05) is taken.
    With trades_at_once = 2 (a test version), B is taken too."""
    rows = flat(2000, 4) + [(2000, 2000.1, 1994, 1995)] + flat(1995, 10)
    signals = [sig(id=x, known_at=at(m), stop=1995, targets=(2020,)) for x, m in
               (("A", 0), ("B", 2), ("C", 4), ("D", 5))]
    rep = play(signals, mk(rows), tf="1m")
    assert [t.signal_id for t in rep.trades] == ["A", "D"]
    assert rep.skipped == ["B", "C"]
    s = summarize(rep)
    assert s["setups_skipped"] == 2
    two = play(signals, mk(rows), tf="1m", k=2)
    assert [t.signal_id for t in two.trades] == ["A", "B", "D"] and two.skipped == ["C"]


def test_a_newer_setup_cancels_a_waiting_limit_order():
    """Tejas, 28 Sep 2026: "Cancel and take new". A buy limit at 1990 (known 10:00) is still waiting at 10:01, when a
    new setup N comes: the limit is CANCELLED and N is taken (it buys at the 10:01 open, 2000). At 10:03 price falls
    to 1989 — the old limit would have filled there, but it no longer exists."""
    rows = flat(2000, 3) + [(2000, 2000.1, 1989, 1992)] + flat(1992, 3)
    signals = [sig(id="L", entry_type="limit", entry_price=1990, stop=1980, targets=(2010,), expiry_bars=5),
               sig(id="N", known_at=at(1), stop=1985, targets=(2010,))]
    rep = play(signals, mk(rows), tf="1m")
    assert rep.not_filled == [("L", "cancelled")] and rep.skipped == []
    assert [(t.signal_id, t.fill_ts, t.fill_px) for t in rep.trades] == [("N", at(1), 2000.0)]
    assert summarize(rep)["not_filled"] == {"expired": 0, "cancelled": 1, "target reached first": 0,
                                            "window ended": 0}


def test_a_filled_limit_is_a_trade_and_is_not_cancelled():
    """A buy limit at 2000 fills during the 10:01 minute (ask low 1999.80 + 0.10 = 1999.90). A setup known at 10:02
    comes during the TRADE → skipped, not a cancel. A setup known at 10:01 comes before that minute's fill → the
    order was still waiting → cancelled, and the new setup is taken."""
    rows = flat(2000.5, 1) + [(2000.5, 2000.6, 1999.8, 2000.2, 10)] + flat(2000.2, 4)
    limit = sig(id="L", entry_type="limit", entry_price=2000, stop=1990, targets=(2010,), expiry_bars=5)
    later = play([limit, sig(id="N", known_at=at(2), stop=1990, targets=(2010,))], mk(rows), tf="1m")
    assert ([t.signal_id for t in later.trades], later.skipped, later.not_filled) == (["L"], ["N"], [])
    same_minute = play([limit, sig(id="N", known_at=at(1), stop=1990, targets=(2010,))], mk(rows), tf="1m")
    assert same_minute.not_filled == [("L", "cancelled")]
    assert [t.signal_id for t in same_minute.trades] == ["N"]


def test_only_a_newer_setup_cancels():
    """Two setups known at the SAME moment: neither is newer. The first (in the order given) is taken, the second
    skipped — as before."""
    signals = [sig(id="L", entry_type="limit", entry_price=1990, stop=1980, targets=(2010,), expiry_bars=5),
               sig(id="N", stop=1980, targets=(2010,))]
    rep = play(signals, mk(flat(2000, 6)), tf="1m")
    assert rep.skipped == ["N"] and rep.not_filled == [("L", "expired")]


def test_a_limit_is_cancelled_when_price_reaches_the_target_first():
    """#119 (steps.md step 6): "If price already reached the equal lows before it came back, there is no trade."
    Sell limit 2000, cancel_at 1990 (the target). At 10:02 the bid low reaches 1990 → cancelled; at 10:04 price comes
    back to 2000 — the order no longer exists. Without cancel_at the same order fills at 10:04."""
    rows = flat(1995, 2) + [(1995, 1995.1, 1990, 1991)] + flat(1993, 1) + [(1993, 2000.5, 1992.9, 2000)] + flat(2000, 3)
    kw = dict(direction="sell", entry_type="limit", entry_price=2000, stop=2010, targets=(1990,), expiry_bars=10)
    rep = play([sig(cancel_at=1990, **kw)], mk(rows), tf="1m")
    assert rep.trades == [] and rep.not_filled == [("s1", "target reached first")]
    assert summarize(rep)["not_filled"]["target reached first"] == 1
    plain = play([sig(**kw)], mk(rows), tf="1m")
    assert [(t.fill_ts, t.fill_px) for t in plain.trades] == [(at(4), 2000.0)]


def test_a_fill_before_the_target_is_a_trade():
    """Buy limit 1995 fills at 10:01 (ask low 1994.8 + 0.2 = 1995.0); the target side 2005 is reached only at 10:03 —
    after the fill: a normal trade that closes at the target."""
    rows = flat(2000, 1) + [(2000, 2000.1, 1994.8, 1996)] + flat(1998, 1) + [(1998, 2006, 1997.9, 2005)] + flat(2005, 2)
    rep = play([sig(entry_type="limit", entry_price=1995, stop=1985, targets=(2005,), expiry_bars=10,
                    cancel_at=2005)], mk(rows), tf="1m")
    assert [(t.fill_ts, t.exit_reason) for t in rep.trades] == [(at(1), "target")] and rep.not_filled == []


def test_the_target_and_the_fill_in_one_minute_is_no_trade():
    """Sell limit 2000, cancel_at 1990: one minute touches BOTH 2000 and 1990. Which came first is unknown → no trade
    (T3's walk did the same: 'the target reached first = no trade', with ties counted as reached)."""
    rows = flat(1995, 1) + [(1995, 2000.2, 1989.5, 1994)] + flat(1994, 3)
    rep = play([sig(direction="sell", entry_type="limit", entry_price=2000, stop=2010, targets=(1990,),
                    expiry_bars=10, cancel_at=1990)], mk(rows), tf="1m")
    assert rep.trades == [] and rep.not_filled == [("s1", "target reached first")]


def test_summary_text_names_cancelled_orders():
    signals = [sig(id="L", entry_type="limit", entry_price=1990, stop=1980, targets=(2010,), expiry_bars=5),
               sig(id="N", known_at=at(1), stop=1980, targets=(2010,))]
    result = run(signals, tf="1m", m1=mk(flat(2000, 6)), record=False)
    assert "1 cancelled by a newer setup" in result.text()
    assert "Cancel and take new" in result.settings["waiting_orders"]


def test_summary_text_names_skipped_setups():
    rows = flat(2000, 4) + [(2000, 2000.1, 1994, 1995)] + flat(1995, 3)
    signals = [sig(id="A", stop=1995, targets=(2020,)), sig(id="B", known_at=at(2), stop=1995, targets=(2020,))]
    result = run(signals, tf="1m", m1=mk(rows), record=False)
    assert "setups skipped: 1" in result.text()
    assert "one trade at a time" in result.text()


# ================================================================================================================
# the window end
# ================================================================================================================

def test_a_trade_open_at_the_window_end_is_not_counted():
    """A closes at its target (+$10). B is still open when the bars end → "open at end": reported apart, no money,
    not in the Rs total."""
    rows = flat(2000, 2) + [(2001, 2010.5, 2000.9, 2003)] + flat(2003, 5)
    signals = [sig(id="A", stop=1990, targets=(2010,)), sig(id="B", known_at=at(4), stop=1990, targets=(2050,))]
    result = run(signals, tf="1m", m1=mk(rows), record=False)
    a, b = result.trades
    assert (a.status, b.status, b.net_inr, b.exit_ts) == ("closed", "open at end", None, None)
    s = result.summary
    assert (s["trades"], s["open_at_end"], s["open_at_end_ids"]) == (1, 1, ["B"])
    assert s["net_inr_total"] == pytest.approx(a.net_inr, abs=0.005)          # rounded to paise
    assert "OPEN AT END (still open when the test ended — not counted in ₹): 1" in result.text()
    with pytest.raises(ValueError, match="still open"):
        b.positions()


def test_a_limit_still_waiting_at_the_window_end():
    rep = play([sig(entry_type="limit", entry_price=1990, stop=1980, targets=(2010,), expiry_bars=50)],
               mk(flat(2000, 6)), tf="1m")
    assert rep.not_filled == [("s1", "window ended")]


@pytest.mark.data
def test_no_bar_after_the_window_is_loaded():
    """Real prices, window ending 2019-03-11 10:30. A buy at 10:00 with a far target and stop is still open at the
    end: open at end, not counted. The last bar the run saw is 10:30 — nothing after it."""
    s = sig(known_at=T0, stop=1000, targets=(3000,))
    result = run([s], tf="5m", start="2019-03-11 09:00", end="2019-03-11 10:30")
    assert result.settings["window"]["last_bar"] == "2019-03-11 10:30:00"
    assert result.replay.last_bar == pd.Timestamp("2019-03-11 10:30")
    assert result.summary["open_at_end"] == 1 and result.summary["trades"] == 0
    assert result.data["gold_m1_content_sha256"] == bars.m1_fingerprint()


def test_signals_outside_the_window_are_refused():
    with pytest.raises(ValueError, match="outside the window"):
        play([sig(known_at=at(30), stop=1990, targets=(2010,))], mk(flat(2000, 5)), tf="1m")


# ================================================================================================================
# cut off the future
# ================================================================================================================

def _random_bars(seed=7, days=3) -> pd.DataFrame:
    """A random walk of 1-min bars, Mon-Wed 11-13 Mar 2019, with a few gaps and some 0 spreads."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2019-03-11 00:00", periods=n, freq="1min", name="server_time")
    o = np.empty(n)
    c = np.empty(n)
    price = 1300.0
    for i in range(n):
        o[i] = price + (rng.normal(0, 1.0) if rng.random() < 0.02 else 0.0)
        c[i] = o[i] + rng.normal(0, 0.3)
        price = c[i]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.15, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.15, n))
    spread = rng.choice([0, 12, 18, 25, 30], size=n)
    return pd.DataFrame({"open": o.round(2), "high": h.round(2), "low": lo.round(2), "close": c.round(2),
                         "spread": spread, "vol": 1}, index=idx)


def _signals_for(m1: pd.DataFrame) -> list[Signal]:
    """Every 20 minutes a signal, made from the close of the bar before it (known at that bar's close)."""
    out = []
    for i, t in enumerate(m1.index[19::20]):
        close = float(m1.close.loc[t])
        known = t + pd.Timedelta(minutes=1)
        kind = i % 4
        if kind == 0:
            s = sig(id=f"s{i}", known_at=known, levels="from_fill", stop=1.0, targets=(1.5,))
        elif kind == 1:
            s = sig(id=f"s{i}", known_at=known, direction="sell", levels="from_fill", stop=1.0, targets=(1.5, 3.0),
                    lots=0.02)
        elif kind == 2:
            e = round(close - 0.5, 2)
            s = sig(id=f"s{i}", known_at=known, entry_type="limit", entry_price=e, stop=e - 1, targets=(e + 1.5,),
                    expiry_bars=6)            # 30 minutes: often still waiting when the next setup comes (a cancel)
        else:
            e = round(close + 0.5, 2)
            s = sig(id=f"s{i}", known_at=known, direction="sell", entry_type="limit", entry_price=e, stop=e + 1,
                    targets=(e - 1.5,), expiry_bars=2)            # 10 minutes: can expire
        out.append(s)
    return out


def test_cut_off_the_future():
    """Run the tester on all bars, and again on the bars up to a cut. Every trade that closed (and every fill) at or
    before the cut must be the same both ways — the tester never looks past its own minute (lesson B1)."""
    m1 = _random_bars()
    signals = _signals_for(m1)

    def rep_on(df):
        last = df.index[-1] + pd.Timedelta(minutes=1)
        return play([s for s in signals if pd.Timestamp(s.known_at) <= last], df)

    def closed(df):
        rows = {pd.Timestamp(t.exit_ts): {"fill": pd.Timestamp(t.fill_ts).value, "fill_px": t.fill_px,
                                          "exit_px": t.exit_px, "net_inr": t.net_inr, "ambiguous": int(t.ambiguous)}
                for t in rep_on(df).trades if t.status == "closed"}
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()

    def fills(df):
        rows = {pd.Timestamp(t.fill_ts): {"fill_px": t.fill_px, "signal": int(t.signal_id[1:])}
                for t in rep_on(df).trades}
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()

    full = rep_on(m1)
    assert len([t for t in full.trades if t.status == "closed"]) > 60 and full.skipped
    assert {"cancelled", "expired"} <= {why for _, why in full.not_filled}     # every path is exercised
    cuts = [700, 1500, 2222, 3001, 4000]
    assert_no_future(closed, m1, cuts)
    assert_no_future(fills, m1, cuts)


# ================================================================================================================
# trial files
# ================================================================================================================

def test_one_trial_file_per_run(_trials_in_tmp):
    """Each run writes exactly ONE JSON through machine.trials.record: settings, the data's hash, the git commit,
    the parent idea, the result. Two runs → two files; nothing is overwritten."""
    m1 = mk(flat(2000, 2) + [(2001, 2010.5, 2000.9, 2003)] + flat(2003, 3))
    signals = [sig(stop=1990, targets=(2010,), parent_idea="idea-000")]
    first = run(signals, tf="1m", m1=m1)
    assert sorted(_trials_in_tmp.glob("*.json")) == [first.trial_path]
    second = run(signals, tf="1m", m1=m1)
    files = sorted(_trials_in_tmp.glob("*.json"))
    assert len(files) == 2 and second.trial_path in files and first.trial_path != second.trial_path
    body = json.loads(first.trial_path.read_text(encoding="utf-8"))
    assert (body["kind"], body["idea_id"], body["version_id"], body["parent_idea"]) == \
        ("tester", "idea-test", "v1", "idea-000")
    assert len(body["git"]["commit"]) == 40
    assert len(body["data"]["frame_sha256"]) == 64
    assert body["settings"]["trades_at_once"] == 1 and body["settings"]["timeframe"] == "1m"
    assert body["result"]["trades"] == 1
    assert body["result"]["net_inr_total"] == pytest.approx(first.trades[0].net_inr, abs=0.005)   # rounded to paise
    second_id = json.loads(second.trial_path.read_text(encoding="utf-8"))["trial_id"]
    assert [b["trial_id"] for b in trials.read_all()] == [body["trial_id"], second_id]
    assert run(signals, tf="1m", m1=m1, record=False).trial_path is None
    assert len(list(_trials_in_tmp.glob("*.json"))) == 2


def test_trial_files_name_changed_files_exactly(monkeypatch):
    """git status --porcelain lines are "XY path"; the first line starts with a space (" M"). Once the first letter
    of the path was cut off ("achine/config.py")."""
    out = {("rev-parse", "HEAD"): "a" * 40,
           ("status", "--porcelain", "--untracked-files=no"): " M machine/config.py\nM  tests/test_x.py\n"}
    monkeypatch.setattr(trials, "_git", lambda *a, strip=True: out[a].strip() if strip else out[a])
    assert trials.git_state() == {"commit": "a" * 40, "changed_files": ["machine/config.py", "tests/test_x.py"]}


def test_code_hash_is_the_same_on_every_pc(tmp_path):
    """git writes Windows line endings (CRLF) on a PC with core.autocrlf, Unix ones (LF) on Linux: same code, same
    hash."""
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert trials.code_sha256([lf]) == trials.code_sha256([crlf])
    crlf.write_bytes(b"x = 1\r\ny = 3\r\n")
    assert trials.code_sha256([lf]) != trials.code_sha256([crlf])


def test_the_tests_never_write_the_real_trials_folder(_trials_in_tmp):
    assert config.TRIALS_DIR == _trials_in_tmp
    assert config.TRIALS_DIR != config.WORK_DIR / "trials"


# ================================================================================================================
# guards and refusals
# ================================================================================================================

def test_the_check_years_stay_hidden():
    m1 = mk(flat(2000, 5), start=pd.Timestamp("2021-03-08 10:00"))
    s = sig(known_at=pd.Timestamp("2021-03-08 10:00"), stop=1990, targets=(2010,))
    with pytest.raises(CheckYearsClosedError):
        run([s], tf="1m", m1=m1, record=False)


def test_the_sealed_years_are_refused():
    with pytest.raises(SealedYearsError):
        sig(known_at=pd.Timestamp("2024-01-02 10:00"), stop=1990, targets=(2010,))
    with pytest.raises(SealedYearsError):
        run([sig(stop=1990, targets=(2010,))], tf="5m", start="2023-12-01", end="2024-01-05", record=False)


@pytest.mark.parametrize("kw, words", [
    (dict(entry_type="limit", entry_price=1990, stop=1980, targets=(2010,)), "expiry_bars"),
    (dict(entry_type="limit", stop=1980, targets=(2010,), expiry_bars=3), "entry_price"),
    (dict(stop=1980, targets=(2010,), expiry_bars=3), "no entry_price, no expiry_bars"),
    (dict(stop=1980, targets=(2010,), cancel_at=2010), "no cancel_at"),
    (dict(entry_type="limit", entry_price=1990, stop=1980, targets=(2010,), expiry_bars=3, cancel_at=1985),
     "cancel_at must be above"),
    (dict(direction="sell", entry_type="limit", entry_price=2000, stop=2010, targets=(1990,), expiry_bars=3,
          cancel_at=2005), "cancel_at must be below"),
    (dict(stop=1980, targets=(2010,), lots=0.015), "0.01, 0.02"),
    (dict(stop=1980, targets=(2010, 2020), lots=0.01), "split equally"),
    (dict(entry_type="limit", entry_price=1990, stop=1995, targets=(2010,), expiry_bars=3), "below"),
    (dict(direction="sell", stop=1990, targets=(2010,)), "targets must be below"),
    (dict(), "a stop, a target, or both"),
    (dict(known_at=pd.Timestamp("2019-03-11 10:00", tz="UTC"), stop=1990, targets=(2010,)), "time zone"),
])
def test_bad_signals_are_refused(kw, words):
    with pytest.raises(ValidationError, match=words):
        sig(**kw)


def test_targets_are_kept_nearest_first():
    assert sig(stop=1990, targets=(2010, 2005), lots=0.02).targets == (2005, 2010)
    assert sig(direction="sell", stop=2010, targets=(1990, 1995), lots=0.02).targets == (1995, 1990)


def test_one_run_is_one_version_of_one_idea():
    with pytest.raises(ValueError, match="ONE version"):
        run([sig(stop=1990, targets=(2010,)), sig(id="s2", version_id="v2", stop=1990, targets=(2010,))], tf="1m",
            m1=mk(flat(2000, 5)), record=False)
