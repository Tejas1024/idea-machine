"""L6b.3 - his bias on the hour / day / week, the real all-time high, premium / discount, the hammer, halfway, and the
new steps, rules, filters and target of the idea compiler. Fake bars unless marked `data`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import guards, settings
from machine.blocks import bias as B
from machine.blocks import halfway as HW
from machine.blocks import hammer as HM
from machine.blocks import premium_discount as PD
from machine.ideas import compile as C
from machine.ideas.form import Idea
from tests.block_fakes import START, bars_from, random_bars
from tests.test_ideas import fake_m1, signal_rows, type1

D1 = pd.Timedelta(days=1)
FAR_ABOVE = 10_000.0                       # an all-time high far above the fake prices: never "at the highs"

# His daily trade, hand-made (the module text of machine/blocks/bias.py): a swing low (97, candle 2), a green order
# block (candle 6, high 100.4) confirmed by the drop (7), the GRAB (8: wick to 96.5, close back above 97), the BREAK
# (10: a wick above 100.4), the first GAP of the move (candles 8-10: 98.4 -> 99.4), price back to the gap (11: a doji,
# low 99.3), and the trade FAILS (12: a close below the grab's extreme 96.5).
DAILY = [(101, 102, 100, 101), (101, 101.5, 99, 99.5), (99.5, 100, 97, 98), (98, 100, 98.5, 99.5),
         (99.5, 101, 99, 100.5), (100.5, 101, 99.5, 100), (99.8, 100.4, 99.2, 100.2), (100.2, 100.3, 98.0, 98.2),
         (98.2, 98.4, 96.5, 97.8), (97.8, 99.5, 97.6, 99.3), (99.3, 101.0, 99.4, 100.8), (100.8, 101.2, 99.3, 100.9),
         (100.9, 101.0, 96.0, 96.2), (96.2, 96.5, 95.5, 95.8)]


def mirror(rows, k=200.0):
    """The same candles upside down: a sell's story instead of a buy's."""
    return [(k - o, k - lo, k - h, k - c) for o, h, lo, c in rows]


def daily_bias(rows, **changes) -> pd.DataFrame:
    with settings.using(changes):
        return B.htf_bias(bars_from(rows, 1440), "1d", ath_before=FAR_ABOVE)


# ================================================================================================= the bias

def test_his_daily_trade_neutral_until_the_gap_then_up_until_it_fails():
    r = daily_bias(DAILY)
    assert list(r["bias"]) == [0] * 10 + [1, 1, 0, 0]
    on = r.iloc[10]
    assert (on.swept, on.extreme, on.gap_near) == (97.0, 96.5, 99.4)
    assert on.sweep_bar == START + 8 * D1 and on.break_bar == START + 10 * D1
    assert on.why.startswith("up: a low was taken, then a break and a gap")
    assert "failed" in r.iloc[12].why and r.iloc[12].bias == 0
    assert r.iloc[9].why == "neutral: no trade on this chart yet"          # the grab waits for its break
    assert (r["known_at"] == r.index + D1).all()                           # known when the day closes


def test_the_mirror_is_a_sell():
    r = daily_bias(mirror(DAILY))
    assert list(r["bias"]) == [0] * 10 + [-1, -1, 0, 0]
    assert r.iloc[10].why.startswith("down: a high was taken")


def test_waiting_for_the_return_is_a_version():
    r = daily_bias(DAILY, bias_waits_for_the_return=True)
    assert list(r["bias"]) == [0] * 11 + [1, 0, 0]                          # candle 11 comes back to 99.4
    assert r.iloc[10].why.startswith("neutral: a grab, a break and a gap wait for price to come back")


def test_the_doji_is_a_version():
    assert B.is_doji(bars_from(DAILY, 1440))[11]
    assert list(daily_bias(DAILY, bias_doji="neutral")["bias"]) == [0] * 10 + [1, 0, 0, 0]
    assert list(daily_bias(DAILY)["bias"])[11] == 1                        # his #70 C: a doji changes nothing


def test_the_real_all_time_high_is_up_on_the_day_chart_only():
    bars = bars_from(DAILY, 1440)
    r = B.htf_bias(bars, "1d", ath_before=100.0)
    assert r["at_ath"].any() and (r.loc[r["at_ath"], "bias"] == 1).all()
    assert r.loc[r["at_ath"], "why"].str.contains("all-time high").all()
    hour = B.htf_bias(bars_from(DAILY, 60), "1h")                          # no all-time-high rule on the hour
    assert not hour["at_ath"].any() and list(hour["bias"]) == [0] * 10 + [1, 1, 0, 0]


def test_bias_at_reads_the_last_candle_that_closed():
    r = daily_bias(DAILY)
    day10 = START + 10 * D1
    got = B.bias_at(r, [day10 + D1 - pd.Timedelta(minutes=1), day10 + D1, day10 + 2 * D1 + pd.Timedelta(hours=5)])
    assert list(got) == [0, 1, 1]                   # candle 10 is known only when it closes
    assert B.bias_at(r, [START - D1])[0] == 0


@pytest.mark.parametrize("tf, minutes, seed, n", [("1h", 60, 5, 900), ("1d", 1440, 6, 900), ("1w", 10080, 7, 240)])
def test_no_future_bias_fake(tf, minutes, seed, n):
    bars = random_bars(n, seed=seed, tf_minutes=minutes)                  # 240 weeks from 2019: before the sealed years
    fn = lambda b: B.htf_bias(b, tf, ath_before=1295.0).reset_index()     # noqa: E731
    assert (fn(bars)["bias"] != 0).mean() > 0.1                           # it says something
    guards.assert_no_future_events(fn, bars, minutes, [n // 6, n // 3, n // 2, 5 * n // 6, n - 1])


def test_the_future_test_catches_a_bias_stamped_one_candle_early():
    bars = random_bars(900, seed=5, tf_minutes=60)

    def early(b):
        r = B.htf_bias(b, "1h").reset_index()
        r["known_at"] = r["known_at"] - pd.Timedelta(hours=1)
        return r

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(early, bars, 60, list(range(100, 900, 9)))


# ================================================================================================= the all-time high

def test_the_all_time_high_starts_from_the_2011_record_and_counts_new_highs():
    rows = [(1900, 1910, 1890, 1905), (1905, 1925, 1900, 1920), (1920, 1922, 1915, 1918), (1918, 1930, 1910, 1929)]
    first = pd.Timestamp("2016-01-04")
    a = B.all_time_high(bars_from(rows, 1440, first), "1d")
    assert settings.get("all_time_high_before_2016") == 1920.94
    assert list(a["ath"]) == [1920.94, 1925, 1925, 1930] and list(a["new"]) == [False, True, False, True]
    near = settings.get("near_all_time_high")
    assert list(a["at_ath"]) == [1905 >= near * 1920.94, True, True, True]
    assert list(B.new_highs(bars_from(rows, 1440, first), "1d")["record_before"]) == [1920.94, 1925]


def test_later_bars_must_say_the_high_before_them():
    with pytest.raises(ValueError, match="give `before`"):
        B.all_time_high(bars_from(DAILY, 1440), "1d")
    assert B.all_time_high(bars_from(DAILY, 1440), "1d", before=102.0)["new"].sum() == 0     # 102 = its highest
    assert B.all_time_high(bars_from(DAILY, 1440), "1d", before=101.5)["new"].sum() == 1


def test_no_future_all_time_high():
    bars = random_bars(900, seed=9, tf_minutes=1440)
    guards.assert_no_future_events(lambda b: B.all_time_high(b, "1d", before=1300.0).reset_index(), bars, 1440,
                                   [100, 450, 899])


# ================================================================================================= premium / discount

def zone_row(**kw):
    row = dict(day_open=100.0, up_lo=108.0, up_hi=110.0, dn_hi=92.0, dn_lo=90.0, premium_from=107.0,
               discount_to=93.0)
    row.update(kw)
    return row


def test_place_his_near_the_green_zone_and_past_it_is_the_same():
    r = zone_row()
    assert [PD.place(p, r) for p in (80, 90.5, 93, 93.01, 100, 106.99, 107, 115)] == [
        "discount", "discount", "discount", "middle", "middle", "middle", "premium", "premium"]
    assert PD.place(100, zone_row(premium_from=np.nan)) == "middle"         # no zones yet
    assert PD.share(90, r) == 0 and PD.share(110, r) == 1


def test_zones_are_his_adr_zones_with_the_two_measures():
    bars = random_bars(12 * 288, seed=3)
    z = PD.zones(bars, "5m")
    last = z.iloc[-1]
    pad = settings.get("near_zone_adr") * last.ruler
    assert last.premium_from == pytest.approx(last.up_lo - pad) and last.discount_to == pytest.approx(last.dn_hi + pad)
    s = PD.zones(bars, "5m", "share_of_the_day").iloc[-1]
    width = s.up_hi - s.dn_lo
    assert s.premium_from == pytest.approx(s.dn_lo + 0.85 * width)
    assert s.discount_to == pytest.approx(s.dn_lo + 0.15 * width)
    with pytest.raises(ValueError, match="measure"):
        PD.zones(bars, "5m", "halfway")


def test_arrivals_count_a_run_once_a_day():
    bars = random_bars(20 * 288, seed=4)
    for side in ("top", "bottom"):
        a = PD.arrivals(bars, "5m", side)
        assert len(a) > 3
        z = PD.zones(bars, "5m")
        pos = bars.index.get_indexer(pd.DatetimeIndex(a["bar"]))
        at = bars["high"].to_numpy() >= z["premium_from"].to_numpy() if side == "top" else \
            bars["low"].to_numpy() <= z["discount_to"].to_numpy()
        assert at[pos].all()
        prev_same = (pos > 0) & at[np.maximum(pos - 1, 0)] & (z["day"].to_numpy()[np.maximum(pos - 1, 0)]
                                                               == z["day"].to_numpy()[pos])
        assert not prev_same.any()


def test_no_future_zones_and_arrivals():
    bars = random_bars(14 * 288, seed=12)
    n = len(bars)
    guards.assert_no_future_events(lambda b: PD.zones(b, "5m").reset_index(), bars, 5, [n // 2, n - 7])
    for side in ("top", "bottom"):
        guards.assert_no_future_events(lambda b: PD.arrivals(b, "5m", side), bars, 5, [n // 3, n // 2, n - 7])


# ================================================================================================= the hammer

def test_his_hammer_a_long_wick_that_took_the_liquidity_and_snapped_back():
    rows = [(100, 100.3, 97.0, 100.2),          # a buy hammer: the long wick below, 3.0 >= 0.12 x 10
            (100, 100.03, 99.7, 100.02),         # the same shape, too small a dip for the day
            (100, 103.0, 99.8, 99.9),            # a sell hammer
            (100, 103, 99.9, 102.9),             # a big body: no hammer
            (100, 101.5, 98.5, 100.1)]           # wicks both ends: no hammer
    h = HM.hammers(bars_from(rows), "5m", ruler=np.full(5, 10.0))
    assert list(h["buy"]) == [True, False, False, False, False]
    assert list(h["sell"]) == [False, False, True, False, False]
    assert h["buy_dip"].iloc[0] == pytest.approx(0.3) and h["buy_wick_share"].iloc[0] == pytest.approx(3.0 / 3.3)


def test_no_future_hammer():
    bars = random_bars(15 * 288, seed=13)
    n = len(bars)
    guards.assert_no_future_events(lambda b: HM.hammers(b, "5m").reset_index(), bars, 5, [n // 2, n - 3])


# ================================================================================================= halfway

def test_halfway_both_readings():
    buy = bars_from([(100, 101, 99.5, 100.5), (100.5, 106, 100.4, 105.5), (105.5, 105.6, 94.5, 95)])
    got = HW.halfway(buy, "5m", "buy", 100, 110, 95, START, "target")
    assert got["bar"] == START + pd.Timedelta(minutes=10) and got["travelled"] == pytest.approx(0.6)
    short = bars_from([(100, 101, 99.5, 100.5), (100.5, 104, 100.4, 103.5), (103.5, 103.6, 94.5, 95)])
    assert HW.halfway(short, "5m", "buy", 100, 110, 95, START, "target") is None       # only 40% of the way
    back = bars_from([(100, 100.5, 97, 97.5), (97.5, 100.2, 97.4, 100.1)])
    got = HW.halfway(back, "5m", "buy", 100, 110, 95, START, "hedge")
    assert got["bar"] == START + pd.Timedelta(minutes=5) and got["travelled"] == pytest.approx(0.6)
    hit = bars_from([(100, 100.5, 94.9, 97.5), (97.5, 100.2, 97.4, 100.1)])
    assert HW.halfway(hit, "5m", "buy", 100, 110, 95, START, "hedge") is None         # the hedge point came first
    sell = bars_from(mirror([(100, 101, 99.5, 100.5), (100.5, 106, 100.4, 105.5), (105.5, 105.6, 94.5, 95)]))
    assert HW.halfway(sell, "5m", "sell", 100, 90, 105, START, "target")["travelled"] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="B2-05"):
        HW.halfway(buy, "5m", "buy", 100, 110, 95, START, "middle")


# ================================================================================================= the compiler

@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return fake_m1()


@pytest.fixture(scope="module")
def market(m1) -> C.Market:
    return C.Market.from_m1(m1, ath_before=1285.0)


def everything(**kw) -> dict:
    """His Type 1 with every L6b.3 piece a form can name, each as it is meant."""
    return type1(
        id="idea-910", title="grab + gap with his bias, zones and the dial (a test form)",
        filters=[dict(name="bias", params={"charts": ["1h"], "must": "not_against"})],
        do_not_enter=type1()["do_not_enter"] + [dict(name="premium_discount", params={"charts": ["1h"]}),
                                               dict(name="no_sell_at_the_all_time_high")],
        targets=[dict(kind="next_liquidity", pick="by_the_dial")], **kw)


def zone_start(**kw) -> dict:
    """Idea #2's reading U2-B shape: price at his ADR zone first, then the grab, the break, the gap."""
    d = type1(id="idea-911", title="the ADR zone, then the grab (a test form)")
    d["steps"] = [dict(id="adr", what="price comes to the top or bottom ADR zone", block="premium_discount", tf="5m")] + [
        {**d["steps"][0], "max_wait_bars": 12}] + d["steps"][1:]
    d.update(kw)
    return d


def kept(out) -> list[tuple]:
    """(signal, the facts its steps found) for every signal of a traced compile."""
    by = {s.id: s for s in out.signals}
    return [(by[r["signal"]], r["facts"]) for r in out.trace if r["signal"]]


def grab_is_hammer(market, fp, direction, facts) -> bool:
    j = market.bars("5m").index.get_loc(facts["sweep"]["bar"])
    return bool(market.hammers(fp)[direction].iat[j])


def test_every_new_piece_compiles_and_names_itself(market):
    out = C.compile_idea(Idea.load(everything()), market)
    base = C.compile_idea(Idea.load(type1()), market)
    assert out.signals and len(out.signals) < len(base.signals)
    text = C.describe()
    for name in ("bias", "premium_discount", "hammer", "all_time_high", "no_sell_at_the_all_time_high",
                 "higher_timeframe"):
        assert name in text
    assert set(C.PLANNED) & {"bias", "higher_timeframe", "all_time_high", "hammer", "premium_discount"} == set()


def test_the_bias_filter_keeps_only_what_his_bias_allows_or_a_hammer_grab(market):
    idea = Idea.load(type1(filters=[dict(name="bias", params={"charts": ["1h"]})]))
    out = C.compile_idea(idea, market, trace=True)
    base = C.compile_idea(Idea.load(type1()), market)
    assert out.signals and len(out.signals) < len(base.signals)
    fp = settings.fingerprint(out.resolved)
    hammers = 0
    for s, facts in kept(out):
        want = 1 if s.direction == "buy" else -1
        if market.bias_at("1h", fp, s.known_at) != want:
            assert grab_is_hammer(market, fp, s.direction, facts)       # his #115: a hammer grab passes
            hammers += 1
    with settings.using({"hammer_beats_the_higher_timeframe": False}):
        strict = C.compile_idea(idea, market)
    assert len(strict.signals) == len(out.signals) - hammers
    assert C.compile_idea(Idea.load(type1(filters=[dict(name="higher_timeframe", params={"charts": ["1h"]})])),
                          market).counts() == out.counts()


def test_the_zone_step_as_a_check_and_as_the_start(market):
    check = type1(id="idea-912")
    check["steps"] = check["steps"] + [dict(id="adr", what="the entry is at the ADR zone", block="premium_discount",
                                            tf="5m", max_wait_bars=0)]
    out = C.compile_idea(Idea.load(check), market)
    base = C.compile_idea(Idea.load(type1()), market)
    assert len(out.signals) < len(base.signals)
    fp = settings.fingerprint(out.resolved)
    z = market.adr_zones(fp, "near_the_zone")
    for s in out.signals:
        i = market.closed_by("5m", s.known_at)
        assert PD.place(s.entry_price, z.iloc[i]) == ("premium" if s.direction == "sell" else "discount")
    start = C.compile_idea(Idea.load(zone_start()), market)
    assert start.funnel["adr"]["buy"] > 0 and start.funnel["adr"]["sell"] > 0


def test_the_hammer_and_the_all_time_high_checks(market):
    ham = type1(id="idea-913")
    ham["steps"] = ham["steps"] + [dict(id="ham", what="the grab candle is a hammer", block="hammer", tf="5m",
                                        max_wait_bars=0)]
    out = C.compile_idea(Idea.load(ham), market, trace=True)
    fp = settings.fingerprint(out.resolved)
    assert out.dropped[("ham", "the candle that took the liquidity is not his hammer (#115)")] > 0
    for s, facts in kept(out):                      # few or none on the fake bars: a hammer needs a big wick
        assert grab_is_hammer(market, fp, s.direction, facts) and facts["ham"]["bar"] == facts["sweep"]["bar"]
    ath = type1(id="idea-914")
    ath["steps"] = ath["steps"] + [dict(id="top", what="price at the all-time high", block="all_time_high", tf="5m",
                                        max_wait_bars=0)]
    got = C.compile_idea(Idea.load(ath), market)
    a = market.ath5(settings.fingerprint(got.resolved))
    for s in got.signals:
        assert a["at_ath"].iat[market.closed_by("5m", s.known_at)]


def test_no_sell_at_the_all_time_high_unless_the_week_is_down(market):
    idea = Idea.load(type1(do_not_enter=type1()["do_not_enter"] + [dict(name="no_sell_at_the_all_time_high")]))
    out = C.compile_idea(idea, market)
    fp = settings.fingerprint(out.resolved)
    a = market.ath5(fp)
    for s in out.signals:
        if s.direction == "sell":
            at = a["at_ath"].iat[market.closed_by("5m", s.known_at)]
            assert not at or market.bias_at("1w", fp, s.known_at) == -1
    assert any("all-time high" in why for (_, why) in out.dropped)


def test_the_dial_target_his_82_and_91(market):
    """With the hour AND the day: the next liquidity beyond the first target (never nearer). Against: the first, or a
    recent 5-minute swing (never farther). Mixed: exactly the first."""
    near = C.compile_idea(Idea.load(type1()), market)
    dial = C.compile_idea(Idea.load(type1(targets=[dict(kind="next_liquidity", pick="by_the_dial")])), market)
    first = {(s.direction, s.known_at, s.entry_price): s.targets[0] for s in near.signals}
    fp = settings.fingerprint(dial.resolved)
    seen = {"with": 0, "against": 0, "mixed": 0}
    farther = 0
    for s in dial.signals:
        t0 = first.get((s.direction, s.known_at, s.entry_price))
        if t0 is None:
            continue
        how = C.dial(C._Run(market, dial.form, s.direction, fp), s.known_at)
        seen[how] += 1
        gain = (s.targets[0] - t0) * (1 if s.direction == "buy" else -1)       # > 0 = farther from the entry
        if how == "with":
            assert gain >= 0
            farther += gain > 0
        elif how == "against":
            assert gain <= 0
        else:
            assert gain == 0
    assert seen["mixed"] > 0 and seen["with"] + seen["against"] > 0


def test_a_chart_the_market_does_not_have_is_named(m1):
    small = C.Market.from_m1(m1, tfs=("5m",), ath_before=1285.0)
    with pytest.raises(C.CompileError, match="reads the 1h chart"):
        C.compile_idea(Idea.load(type1(filters=[dict(name="bias")])), small)
    with pytest.raises(C.CompileError, match="by_the_dial reads the 1h chart"):
        C.compile_idea(Idea.load(type1(targets=[dict(kind="next_liquidity", pick="by_the_dial")])), small)


@pytest.mark.parametrize("make", [everything, zone_start], ids=["everything", "zone_start"])
def test_compiled_signals_with_the_new_pieces_do_not_peek(m1, make):
    idea = Idea.load(make())

    def signals_of(bars):
        return signal_rows(C.compile_idea(idea, C.Market.from_m1(bars, ath_before=1285.0)).signals)

    n = len(m1)
    assert len(signals_of(m1)) > 5
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])


# ================================================================================================= real bars

@pytest.mark.data
def test_no_future_bias_real_hour_and_day():
    from machine.data.bars import load_bars
    for tf, start, end, step in (("1h", "2019-01", "2019-06", 60), ("1d", "2016-01", "2019-12", 1440)):
        bars = load_bars(tf, start, end)
        n = len(bars)
        guards.assert_no_future_events(lambda b: B.htf_bias(b, tf, ath_before=1920.94).reset_index(), bars, step,
                                       [n // 4, n // 2, 3 * n // 4, n - 1])
