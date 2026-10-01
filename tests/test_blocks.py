"""The core blocks (L6.1): each on hand-made candles with a known answer, and each through the cut-off-the-future test
(likhith.md R4) - on fake bars everywhere, and on real 2019 bars when the price data is here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import guards
from machine.blocks import adr, fib, fvg, sessions, structure, swings
from machine.blocks._base import first_hit
from tests.block_fakes import START, bars_from, random_bars

M5 = pd.Timedelta(minutes=5)


def t(i: int, step=M5) -> pd.Timestamp:
    return START + i * step


def hl(highs, lows):
    return [((h + lo) / 2, h, lo, (h + lo) / 2) for h, lo in zip(highs, lows)]


# ---- first_hit ---------------------------------------------------------------------------------------------------

def test_first_hit_matches_a_plain_loop():
    rng = np.random.default_rng(3)
    arr = np.cumsum(rng.normal(0, 1, 5000))
    starts = rng.integers(0, 5000, 400)
    x = arr[starts] - rng.uniform(0, 30, 400)
    ends = np.minimum(starts + rng.integers(1, 3000, 400), 5000)
    for op, fn in (("<=", np.less_equal), ("<", np.less), (">=", np.greater_equal), (">", np.greater)):
        got = first_hit(arr, starts, x, op, ends=ends)
        for q in range(len(starts)):
            hits = np.flatnonzero(fn(arr[starts[q]:ends[q]], x[q]))
            assert got[q] == (starts[q] + hits[0] if len(hits) else -1)


# ---- swings -----------------------------------------------------------------------------------------------------

def test_swings_his_two_candle_rule_strict_and_known_two_candles_later():
    bars = bars_from(hl([10, 11, 15, 12, 11, 13, 13, 12, 11], [9, 10, 14, 11, 10, 12, 12, 11, 10]))
    s = swings.swings(bars, "5m")
    assert s[["bar", "side", "price"]].values.tolist() == [[t(2), "high", 15.0], [t(4), "low", 10.0]]
    # known when the 2nd candle after it has CLOSED
    assert s["known_at"].tolist() == [t(4) + M5, t(6) + M5]
    # the two equal highs of 13 (bars 5 and 6) are not a swing: "strictly" higher on both sides


# ---- order block + break of structure -------------------------------------------------------------------------

STRUCT = [
    (100.0, 101.2, 99.8, 101.0),    # 0 green
    (101.0, 101.3, 100.3, 100.5),   # 1 RED: the sell block (low 100.3)
    (100.5, 101.1, 100.4, 101.0),   # 2 green, not above 101.3 yet
    (101.0, 101.7, 100.9, 101.6),   # 3 green, above 101.3 -> block 1 confirmed here
    (101.6, 101.65, 101.1, 101.2),  # 4 red, but the next candle is red too: never confirmed
    (101.2, 101.25, 100.1, 100.2),  # 5 red, wick below 100.3 -> THE BREAK of block 1
    (100.2, 101.3, 100.15, 101.2),  # 6 green, above bar 5's high -> bar 5 is a block
]


def test_order_blocks_his_red_candle_in_the_rise():
    b = structure.order_blocks(bars_from(STRUCT), "5m")
    sell = b[b.side == "sell"]
    assert sell["block_bar"].tolist() == [t(1), t(5)]
    assert sell["confirmed_bar"].tolist() == [t(3), t(6)]
    assert sell["level"].tolist() == [100.3, 100.1]              # the block's low: a wick below it is the break
    assert sell["known_at"].tolist() == [t(3) + M5, t(6) + M5]
    buy = b[b.side == "buy"]                                     # green bar 3, the drop (bars 4-5) goes below its low
    assert buy[["block_bar", "confirmed_bar", "level"]].values.tolist() == [[t(3), t(5), 101.7]]


def test_breaks_a_wick_through_the_most_recent_block():
    br = structure.breaks(bars_from(STRUCT), "5m")
    assert br[["side", "bar", "level", "block_bar"]].values.tolist() == [["sell", t(5), 100.3, t(1)]]
    assert br["known_at"].tolist() == [t(5) + M5]


def test_first_break_after_the_sweep_and_the_block_age_limit():
    bars = bars_from(STRUCT)
    got = structure.first_break_after(bars, "5m", "sell", after=t(2))
    assert got == {"known_at": t(5) + M5, "bar": t(5), "level": 100.3, "block_bar": t(1)}
    assert structure.first_break_after(bars, "5m", "sell", after=t(2), max_age_bars=0) is None   # block 1 too old
    assert structure.first_break_after(bars, "5m", "buy", after=t(2)) is None


# ---- fair value gaps --------------------------------------------------------------------------------------------

GAPS = [
    (100.0, 100.5, 99.8, 100.4),    # 0
    (100.4, 101.8, 100.3, 101.7),   # 1 the big candle
    (101.7, 102.0, 101.0, 101.9),   # 2 low 101.0 > bar 0's high 100.5: BUY gap 100.5-101.0
    (101.9, 102.2, 101.5, 102.1),   # 3
    (102.1, 102.2, 101.2, 101.3),   # 4 comes down, not to 101.0 yet
    (101.3, 101.4, 100.9, 101.0),   # 5 reaches 101.0: the FIRST visit (not through: low 100.9 > 100.5)
    (101.0, 101.1, 100.2, 100.3),   # 6 a later visit: never counts
]


def test_gaps_three_candles_and_his_near_edge():
    g = fvg.gaps(bars_from(GAPS), "5m")
    buy = g[g.side == "buy"].iloc[0]
    assert (buy.first_bar, buy.bar, buy.lo, buy.hi) == (t(0), t(2), 100.5, 101.0)
    assert buy.near == 101.0 and buy.far == 100.5                 # #65: a buy enters at the TOP of the gap
    assert buy.known_at == t(2) + M5
    sell = g[g.side == "sell"]
    assert sell[["bar", "lo", "hi", "near"]].values.tolist() == [[t(5), 101.4, 101.5, 101.4],
                                                                 [t(6), 101.1, 101.2, 101.1]]   # a sell: the BOTTOM


def test_first_visit_only_the_first_counts_no_time_limit():
    bars = bars_from(GAPS)
    v = fvg.first_visits(bars, "5m")
    assert v[["side", "gap_bar", "bar", "near", "through"]].values.tolist() == [["buy", t(2), t(5), 101.0, False]]
    assert v["known_at"].tolist() == [t(5) + M5]
    # no time limit (#68): the same visit 500 candles later is still found
    far = bars_from(GAPS[:5] + [GAPS[4]] * 500 + GAPS[5:])
    v2 = fvg.first_visits(far, "5m")
    assert v2[v2.side == "buy"]["bar"].tolist() == [t(505)]


def test_gaps_of_the_move_after_the_sweep_and_at_most_6_after_the_break():
    bars = bars_from([(100, 100.5, 99.5, 100)] * 30)
    sweep, brk = 4, 8

    def gap(side, first, third):
        return dict(known_at=t(third) + M5, side=side, first_bar=t(first), bar=t(third), lo=1.0, hi=2.0,
                    near=2.0, far=1.0, size=1.0)

    table = pd.DataFrame([
        gap("buy", 3, 5),      # 1st candle BEFORE the sweep: "only valid if it forms after taking any liquidity" (#131)
        gap("buy", 4, 6),      # 1st candle = the sweep candle, before the break: counts (#62) -> rank 0
        gap("sell", 7, 9),     # the other side
        gap("buy", 12, 14),    # 3rd candle = break + 6: counts -> rank 1
        gap("buy", 13, 15),    # break + 7: not this move's gap (#101)
    ])
    m = fvg.gaps_of_move(bars, "5m", "buy", t(sweep), t(brk), gap_table=table)
    assert m["bar"].tolist() == [t(6), t(14)]
    assert m["rank"].tolist() == [0, 1]
    assert m["candles_after_break"].tolist() == [-2, 6]
    assert m["known_at"].tolist() == [t(brk) + M5, t(14) + M5]   # never known before the break is


# ---- Fibonacci -------------------------------------------------------------------------------------------------

def test_fib_plain_reading_higher_high_from_the_lowest_low_between():
    highs = [10, 11, 13, 12, 11, 10, 11, 12, 15, 14, 13]
    lows = [9, 10, 12, 11, 10, 8, 10, 11, 14, 13, 12]
    f = fib.fib_levels(bars_from(hl(highs, lows)), "5m", ratios=(0.5, 0.618))
    assert len(f) == 1
    r = f.iloc[0]
    assert (r.side, r.from_bar, r.to_bar, r.top, r.bottom) == ("buy", t(5), t(8), 15.0, 8.0)
    assert r["level_0.5"] == pytest.approx(11.5)
    assert r["level_0.618"] == pytest.approx(15 - 0.618 * 7)
    assert r.known_at == t(10) + M5                              # when the new high is known as a swing


# ---- sessions --------------------------------------------------------------------------------------------------

H1 = pd.Timedelta(hours=1)


def session_bars(n_hours=10):
    # 1-hour bars from 01:00 server = 04:30 IST (winter: IST = server + 3:30). Asia 04:30-10:30 IST = 01:00-07:00
    # server; London 12:30-15:30 IST = 09:00-12:00 server.
    highs = [5, 1, 2, 9, 3, 4, 8, 7, 6, 10][:n_hours]
    rows = [(100 + h - 0.5, 100 + h, 100 + h - 1, 100 + h - 0.5) for h in highs]
    return bars_from(rows, 60)


def test_sessions_his_windows_so_far_and_final():
    bars = session_bars()
    sf = sessions.so_far(bars, "1h")
    asia = sf[sf.session == "Asia"]
    assert asia.index.tolist() == [START + i * H1 for i in range(6)]         # 01:00 .. 06:00 server
    assert asia["high_so_far"].tolist() == [105, 105, 105, 109, 109, 109]
    assert asia["low_so_far"].tolist() == [104, 100, 100, 100, 100, 100]
    assert sf[sf.session == "London"].index.tolist() == [START + 8 * H1, START + 9 * H1]   # from 09:00 server
    fin = sessions.final(bars, "1h")
    assert fin["session"].tolist() == ["Asia"]                                # London has not ended in the data
    a = fin.iloc[0]
    assert (a.high, a.low, a.high_bar, a.low_bar, a.bars) == (109.0, 100.0, START + 3 * H1, START + H1, 6)
    assert a.known_at == START + 6 * H1                                       # 10:30 IST = 07:00 server


def test_session_not_final_before_its_window_ends():
    assert sessions.final(session_bars(5), "1h").empty                       # data stops at 06:00, Asia ends 07:00
    assert len(sessions.final(session_bars(6), "1h")) == 1                   # its last bar closes at 07:00
    with pytest.raises(ValueError, match="1 hour or less"):
        sessions.final(bars_from([(1, 2, 0, 1)] * 5, 240), "4h")


# ---- ADR -------------------------------------------------------------------------------------------------------

D1 = pd.Timedelta(days=1)


def test_adr_his_5_and_10_day_average_of_finished_days_only():
    ranges = list(range(1, 13))
    rows = [(100.5, 100 + r, 100, 100.5) for r in ranges]
    rows[11] = (100.5, 1000, 100, 100.5)                  # today's own range is never used for today
    d = adr.adr_days(bars_from(rows, 1440), "1d")
    assert d["adr5"].isna().sum() == 5 and d["adr10"].isna().sum() == 10
    assert d.loc[5, "adr5"] == pytest.approx(np.mean(ranges[0:5]))
    assert d.loc[10, "adr10"] == pytest.approx(np.mean(ranges[0:10]))
    assert d.loc[11, "adr5"] == pytest.approx(np.mean(ranges[6:11]))
    r = d.loc[11]
    lo, hi = min(r.adr5, r.adr10) / 2, max(r.adr5, r.adr10) / 2
    assert (r.up_lo, r.up_hi, r.dn_hi, r.dn_lo) == pytest.approx((100.5 + lo, 100.5 + hi, 100.5 - lo, 100.5 - hi))
    assert r.ruler == r.adr10
    assert d.loc[11, "known_at"] == START + 12 * D1       # the day's first bar has closed


# ---- the cut-off-the-future test, every block -------------------------------------------------------------------

EVENT_BLOCKS = {
    "swings": swings.swings,
    "order_blocks": structure.order_blocks,
    "breaks": structure.breaks,
    "gaps": fvg.gaps,
    "first_visits": fvg.first_visits,
    "fib_levels": fib.fib_levels,
    "sessions.final": sessions.final,
    "adr_days": adr.adr_days,
}
BAR_BLOCKS = {"sessions.so_far": sessions.so_far, "adr_bars": adr.adr_bars}


@pytest.mark.parametrize("name", sorted(EVENT_BLOCKS))
def test_no_future_event_blocks_fake(name):
    bars = random_bars(3000, seed=7)
    guards.assert_no_future_events(lambda b: EVENT_BLOCKS[name](b, "5m"), bars, 5, [300, 1111, 2000, 2999])


@pytest.mark.parametrize("name", sorted(BAR_BLOCKS))
def test_no_future_per_bar_blocks_fake(name):
    bars = random_bars(3000, seed=8)
    guards.assert_no_future(lambda b: BAR_BLOCKS[name](b, "5m"), bars, [300, 1111, 2000, 2998])


def test_the_future_test_catches_a_swing_stamped_one_candle_early():
    def early(b):
        s = swings.swings(b, "5m")
        s["known_at"] = s["known_at"] - M5                     # claims to know it at the 1st candle after, not the 2nd
        return s

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(early, random_bars(800, seed=2), 5, list(range(50, 800, 7)))


def test_the_future_test_catches_a_gap_that_reads_the_next_candle():
    def peek(b):
        g = fvg.gaps(b, "5m")
        nxt = b["close"].shift(-1).reindex(pd.DatetimeIndex(g["bar"])).to_numpy()
        g["size"] = g["size"] + np.nan_to_num(nxt) * 0 + (np.isnan(nxt) * 1.0)   # changes when the next bar is cut
        return g

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(peek, random_bars(800, seed=4), 5, list(range(50, 800, 3)))


def test_gaps_of_move_and_first_break_after_do_not_change_when_the_future_is_cut():
    bars = random_bars(3000, seed=11)
    br = structure.breaks(bars, "5m")
    done = 0
    for _, r in br.iloc[20:400:37].iterrows():
        side = r.side
        sweep = bars.index[max(bars.index.get_loc(r.bar) - 5, 0)]
        full = structure.first_break_after(bars, "5m", side, sweep)
        if full is None:
            continue
        stop = bars.index.get_loc(full["bar"]) + 1
        assert structure.first_break_after(bars.iloc[:stop], "5m", side, sweep) == full
        gside = "buy" if side == "buy" else "sell"
        cut = bars.iloc[:stop + 6]
        a = fvg.gaps_of_move(bars, "5m", gside, sweep, full["bar"])
        b = fvg.gaps_of_move(cut, "5m", gside, sweep, full["bar"])
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
        done += 1
    assert done >= 5


# ---- the same on real 2019 bars ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_5m():
    from machine.data.bars import load_bars
    return load_bars("5m", "2019-03-01", "2019-03-31")


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(EVENT_BLOCKS))
def test_no_future_event_blocks_real_march_2019(name, real_5m):
    n = len(real_5m)
    guards.assert_no_future_events(lambda b: EVENT_BLOCKS[name](b, "5m"), real_5m, 5,
                                   [n // 7, n // 3, n // 2, 2 * n // 3, n - 1])


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(BAR_BLOCKS))
def test_no_future_per_bar_blocks_real_march_2019(name, real_5m):
    n = len(real_5m)
    guards.assert_no_future(lambda b: BAR_BLOCKS[name](b, "5m"), real_5m, [n // 7, n // 2, n - 2])


@pytest.mark.data
def test_no_future_fib_and_sessions_on_real_hour_bars():
    from machine.data.bars import load_bars
    h1 = load_bars("1h", "2019-01", "2019-06")
    n = len(h1)
    for fn in (fib.fib_levels, sessions.final, swings.swings):
        guards.assert_no_future_events(lambda b: fn(b, "1h"), h1, 60, [n // 5, n // 2, n - 1])
