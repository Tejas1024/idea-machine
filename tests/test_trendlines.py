"""L6b.1 - trendlines, his touch entry, a line in the path, triangles, false breakouts, wedges (machine/blocks/
trendlines.py, patterns.py; the liquidity book's trendlines; the compiler's new steps). Each rule on hand-made candles
with a known answer, and each block through the cut-off-the-future test (likhith.md R4) - fake bars everywhere, real
2019 bars when the price data is here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import guards, settings
from machine.blocks import liquidity as LQ
from machine.blocks import patterns as PT
from machine.blocks import trendlines as TL
from machine.ideas import compile as C
from machine.ideas.form import Idea
from tests.block_fakes import START, bars_from, fake_market, random_bars
from tests.test_ideas import fake_m1, type1

M5 = pd.Timedelta(minutes=5)
RULER = 10.0                        # a day's range of $10: tol 0.25, through 0.05, turn 2.5, 2 x tol short = 0.50
QUIET = (100.0, 100.2, 99.8, 100.0)


def t(i: int) -> pd.Timestamp:
    return START + i * M5


def chart(n: int, points: dict[int, tuple]) -> pd.DataFrame:
    """n quiet candles; points = {bar: (high, low)} or {bar: (open, high, low, close)}."""
    rows = [QUIET] * n
    for i, p in points.items():
        rows[i] = (QUIET[0], p[0], p[1], QUIET[3]) if len(p) == 2 else p
    return bars_from(rows)


def ruler(b) -> np.ndarray:
    return np.full(len(b), RULER)


# A falling line through three highs, 40 candles apart (3 hours and more), with lower lows between them: a clear
# downtrend. Touches at 10 (110.0), 50 (107.0), 90 (104.0) - the line falls $0.075 a candle.
FALL = {10: (110.0, 99.8), 30: (100.2, 97.0), 50: (107.0, 99.8), 70: (100.2, 95.0), 90: (104.0, 99.8)}


def falling(n: int = 200, extra: dict | None = None) -> pd.DataFrame:
    return chart(n, {**FALL, **(extra or {})})


# ---- the line (#92, #94, #112, #116) -----------------------------------------------------------------------------

def test_three_touches_through_the_wick_ends_is_a_trendline_known_2_candles_after_the_third():
    b = falling()
    L = TL.lines(b, "5m", ruler(b))
    mine = L[(L["a_bar"] == t(10)) & (L["b_bar"] == t(50))]
    assert mine["touches"].tolist() == [2, 3]
    two, three = mine.iloc[0], mine.iloc[1]
    assert two["known_at"] == t(52) + M5 and three["known_at"] == t(92) + M5     # 2 candles after the touch closes
    assert three["c_bar"] == t(90) and three["c_price"] == 104.0
    assert three["way"] == "falling" and three["slope"] == pytest.approx(-0.075)
    assert three["steepness"] == pytest.approx(0.075 * 288 / RULER)               # days of range per day


@pytest.mark.parametrize("third, is_line", [(104.0, True), (104.04, True), (104.06, False), (103.55, True),
                                            (103.45, False)])
def test_a_third_touch_through_the_line_even_slightly_is_not_a_trendline(third, is_line):
    """#94: at most 0.005 of a day's range through ($0.05 here); short of the line up to 2 x the touch width."""
    b = falling(extra={90: (third, 99.8)})
    L = TL.lines(b, "5m", ruler(b))
    got = L[(L["a_bar"] == t(10)) & (L["b_bar"] == t(50))]["touches"].tolist()
    assert got == ([2, 3] if is_line else [2])


def test_a_wick_through_the_line_between_its_touches_means_it_is_drawn_inside():
    """#92 D: 'forcefully drawn inside' - a candle poking through between the touches: no line at all."""
    b = falling(extra={30: (108.6, 97.0)})               # the line is at 108.5 there
    L = TL.lines(b, "5m", ruler(b))
    assert not len(L[(L["a_bar"] == t(10)) & (L["b_bar"] == t(50))])


def test_through_highs_a_line_falls_or_is_flat_and_through_lows_it_rises():
    rising_highs = chart(200, {10: (104.0, 99.8), 30: (100.2, 95.0), 50: (107.0, 99.8), 70: (100.2, 97.0),
                               90: (110.0, 99.8)})
    assert not len(TL.lines(rising_highs, "5m", ruler(rising_highs)).query("side == 'high' and touches == 3"))
    flat = chart(200, {10: (106.0, 99.8), 30: (100.2, 97.0), 50: (106.0, 99.8), 70: (100.2, 97.0),
                       90: (106.0, 99.8)})
    L = TL.lines(flat, "5m", ruler(flat))
    assert L.query("side == 'high' and touches == 3")["way"].tolist() == ["flat"]   # #116: flat is still a line


def test_the_4th_touch():
    b = falling(extra={110: (100.2, 93.0), 130: (101.0, 99.8)})    # the line is at 101.0 on candle 130
    four = TL.more_touches(b, "5m", ruler=ruler(b))
    assert four[["touch", "bar"]].values.tolist() == [[4, t(130)]]
    assert four["known_at"].iat[0] == t(132) + M5


# ---- his touch entry (Type 2, #127) -------------------------------------------------------------------------------

def test_the_touch_entry_of_a_falling_line_is_a_sell_with_a_limit_on_the_line():
    b = falling()
    e = TL.touch_entries(b, "5m", ruler(b))
    row = e[(e["touch"] == 3) & (e["first_bar"] == t(10))].iloc[0]
    assert row["direction"] == "sell" and row["known_at"] == t(92) + M5
    line_now = 107.0 - 0.075 * (92 - 50)
    assert row["line"] == pytest.approx(line_now)
    assert row["entry"] == pytest.approx(107.0 - 0.075 * (92 + 10 - 50))    # the worse fill 10 candles on (F10)
    assert row["hedge"] == pytest.approx(line_now + 2.0 * 0.025 * RULER)   # past the line
    assert row["extreme"] == 95.0                                           # the lowest low since the first touch
    assert bool(row["trend_visible"]) and bool(row["touches_apart"])


def test_touches_under_3_hours_apart_and_a_flat_line_are_not_touch_entries():
    close = chart(200, {10: (110.0, 99.8), 20: (100.2, 97.0), 30: (108.5, 99.8), 40: (100.2, 95.0),
                        50: (107.0, 99.8)})
    e = TL.touch_entries(close, "5m", ruler(close))
    assert len(e) and not e["touches_apart"].any()          # 20 candles apart: his fixed rule (36 = 3 hours)
    flat = chart(200, {10: (106.0, 99.8), 30: (100.2, 97.0), 50: (106.0, 99.8), 70: (100.2, 97.0),
                       90: (106.0, 99.8)})
    assert not len(TL.touch_entries(flat, "5m", ruler(flat)))


def test_no_clear_trend_is_marked():
    """#116: the lows must fall too for a sell - here the second low is HIGHER than the first."""
    b = falling(extra={30: (100.2, 95.0), 70: (100.2, 97.0)})
    e = TL.touch_entries(b, "5m", ruler(b))
    assert len(e) and not e["trend_visible"].any()


# ---- the line in the liquidity book, its sweep, and a line in the path (#69) ------------------------------------

def test_a_trendline_is_liquidity_with_a_moving_price_and_a_touch_is_not_a_take():
    b = falling(250, extra={120: (107.0 - 0.075 * 70 + 0.03, 99.8), 130: (107.0 - 0.075 * 80 + 0.10, 99.8)})
    lines = TL.lines(b, "5m", ruler(b))                       # a $10 day's range, by hand
    rows = LQ._trendlines(lines, b)
    assert rows["kind"].tolist() == ["trendline_res"] and rows["slope"].iat[0] == pytest.approx(-0.075)
    assert rows["beyond_usd"].iat[0] == pytest.approx(0.005 * RULER)
    sw = LQ.sweeps(b, rows)
    assert sw["bar"].tolist() == [t(130)]                     # 3 cents through at 120 = a touch; 10 cents = a take
    assert sw["price"].iat[0] == pytest.approx(107.0 - 0.075 * 80)
    life = LQ.lifetimes(b, rows, sw)
    for i in (100, 125):
        a = LQ.alive(life, i)
        assert a["price"].iat[0] == pytest.approx(107.0 - 0.075 * (i - 50))     # its price ON that candle
    assert not len(LQ.alive(life, 131))


def test_a_line_that_drifts_away_is_gone_not_taken():
    """The old engine's LINE_FAR_ADR: 2 days of range from the close - here the price drops far below the line."""
    rows = [QUIET] * 400
    for i, p in FALL.items():
        rows[i] = (QUIET[0], p[0], p[1], QUIET[3])
    for i in range(100, 400):
        rows[i] = (80.0, 80.2, 79.8, 80.0)
    b = bars_from(rows)
    r = LQ._trendlines(TL.lines(b, "5m", ruler(b)), b)
    assert not len(LQ.sweeps(b, r))
    life = LQ.lifetimes(b, r, LQ.sweeps(b, r))
    assert int(life["end"].iat[0]) < 250                      # gone once 20 dollars from the close


def test_the_line_in_the_path_is_found_between_the_entry_and_the_target():
    b = falling(250)
    L5 = TL.on_m5(TL.lines(b, "5m", ruler(b)).drop_duplicates("line_id"), b)
    line = 107.0 - 0.075 * (120 - 50)
    got = TL.in_path(L5, 120, 100.0, 103.0, "buy")
    assert got is not None and got[1] == pytest.approx(line)
    assert TL.in_path(L5, 120, 100.0, round(line, 2), "buy") is None     # a line AT the target is the target
    assert TL.in_path(L5, 120, 100.0, 101.0, "buy") is None              # beyond the target: not in the path


# ---- triangles, their false breaks, false breaks of liquidity, wedges ----------------------------------------------

TRI = {10: (110.0, 99.8), 20: (100.2, 90.0), 50: (107.0, 99.8), 60: (100.2, 92.0)}   # top falls 0.075, bottom rises 0.05


def test_a_triangle_is_known_when_its_lines_have_narrowed_to_60_percent():
    b = chart(300, TRI)
    tri = PT.triangles(b, "5m", ruler(b), need_big_move=False)
    r = tri.iloc[0]
    assert r["start_bar"] == t(62) and r["bar"] == t(107)                 # 14.0 wide at 62; 45 candles to narrow
    assert r["narrow"] <= 0.60 and r["known_at"] == t(107) + M5
    assert r["top"] == pytest.approx(107.0 - 0.075 * 57) and r["bottom"] == pytest.approx(92.0 + 0.05 * 47)
    assert not r["after_big_move"]                                        # flat before it: no big move
    with pytest.raises(ValueError, match="B2-02"):
        PT.triangles(b, "5m", ruler(b))                                   # his word is open: it must be given
    assert len(PT.triangles(b, "5m", ruler(b), big_move_adr=0.0))


def test_parallel_lines_are_never_a_triangle():
    b = chart(300, {10: (110.0, 99.8), 20: (100.2, 90.0), 50: (107.0, 99.8), 60: (100.2, 87.0)})
    assert not len(PT.triangles(b, "5m", ruler(b), need_big_move=False))


def test_a_poke_out_of_the_triangle_that_closes_back_is_a_false_break():
    top_115 = 107.0 - 0.075 * 65
    b = chart(300, {**TRI, 115: (100.0, top_115 + 0.4, 99.8, 100.0)})
    br = PT.triangle_breaks(b, "5m", ruler(b), need_big_move=False)
    r = br.iloc[0]
    assert (r["kind"], r["side"], r["direction"], r["bar"]) == ("false", "high", "sell", t(115))
    assert r["known_at"] == t(115) + M5 and r["price"] == pytest.approx(top_115)
    assert r["opposite"] == pytest.approx(92.0 + 0.05 * 55)                # P1's target: the other line


def test_a_wedge_is_a_held_line_and_a_close_out_of_the_other_side():
    b = falling(250, extra={100: (100.0, 100.2, 94.4, 94.5)})                 # the wedge's low was 95.0
    w = PT.wedges(b, "5m", ruler(b))
    r = w[w["start_bar"] == t(92)].iloc[0]
    assert (r["direction"], r["bar"], r["known_at"]) == ("sell", t(100), t(100) + M5)
    assert (r["wedge_low"], r["wedge_high"], r["level"]) == (95.0, 110.0, 95.0)
    broke = falling(250, extra={96: (100.0, 104.0, 99.8, 103.8), 100: (100.0, 100.2, 94.4, 94.5)})
    assert not len(PT.wedges(broke, "5m", ruler(broke)).query("start_bar == @t(92)"))   # the held side gave way first


def test_false_breaks_of_liquidity_say_which_way_it_went():
    b = random_bars(6000, seed=21)
    bk = LQ.obvious_liquidity(b)
    sw = LQ.sweeps(b, bk)
    fb = PT.false_breaks(b, bk, sw)
    assert set(fb["result"]) <= {"false", "real", "undecided"} and (fb["result"] == "false").any()
    c = b["close"]
    for r in fb[fb["result"] == "false"].head(40).itertuples():
        k = b.index.get_loc(r.decided_bar)
        assert (c.iat[k] < r.price) if r.side == "high" else (c.iat[k] > r.price)
        assert r.direction == ("sell" if r.side == "high" else "buy")


# ---- the cut-off-the-future test ---------------------------------------------------------------------------------

def _lines(b):
    return TL.lines(b, "5m")


def _touches(b):
    with settings.using({"trendline_swing_turn_adr": 0.0}):        # every swing: 4th touches are rare on a walk
        return TL.more_touches(b, "5m")


def _entries(b):
    return TL.touch_entries(b, "5m")


def _tris(b):
    return PT.triangles(b, "5m", need_big_move=False)


def _breaks(b):
    return PT.triangle_breaks(b, "5m", need_big_move=False)


def _wedges(b):
    return PT.wedges(b, "5m")


def _book(b):
    return LQ.obvious_liquidity(b)


def _sweeps(b):
    return LQ.sweeps(b, LQ.obvious_liquidity(b))


def _false(b):
    bk = LQ.obvious_liquidity(b)
    return PT.false_breaks(b, bk, LQ.sweeps(b, bk))


EVENT_BLOCKS = {"lines": _lines, "more_touches": _touches, "touch_entries": _entries, "triangles": _tris,
                "triangle_breaks": _breaks, "wedges": _wedges, "book": _book, "sweeps": _sweeps, "false_breaks": _false}


@pytest.fixture(scope="module")
def walk():
    return random_bars(9000, seed=14)


@pytest.mark.parametrize("name", sorted(EVENT_BLOCKS))
def test_no_future_fake(name, walk):
    guards.assert_no_future_events(EVENT_BLOCKS[name], walk, 5, [4000, 5500, 7100, 8999])


def test_alive_with_trendlines_uses_only_what_was_known(walk):
    full_b = LQ.obvious_liquidity(walk)
    full = LQ.lifetimes(walk, full_b, LQ.sweeps(walk, full_b))
    assert (full["kind"].str.startswith("trendline")).any()
    for p in (5000, 8000):
        cut = walk.iloc[:p]
        cb = LQ.obvious_liquidity(cut)
        life = LQ.lifetimes(cut, cb, LQ.sweeps(cut, cb))
        for i in range(p - 300, p, 11):
            a = LQ.alive(full, i)[["level_id", "price"]].sort_values("level_id").round(6).values.tolist()
            b = LQ.alive(life, i)[["level_id", "price"]].sort_values("level_id").round(6).values.tolist()
            assert a == b, (p, i)


def test_the_line_in_the_path_uses_only_what_was_known(walk):
    lines = TL.lines(walk, "5m").drop_duplicates("line_id")
    full = TL.on_m5(lines, walk)
    rng = np.random.default_rng(3)
    for p in (5000, 8000):
        cut = walk.iloc[:p]
        part = TL.on_m5(TL.lines(cut, "5m").drop_duplicates("line_id"), cut)
        for i in range(p - 300, p, 7):
            c = float(walk["close"].iat[i])
            for d in (1, -1):
                tgt = c + d * float(rng.uniform(1, 6))
                assert TL.in_path(full, i, c, tgt, "buy" if d > 0 else "sell") == \
                    TL.in_path(part, i, c, tgt, "buy" if d > 0 else "sell"), (p, i)


@pytest.fixture(scope="module")
def real():
    from machine.data.bars import load_bars
    return {tf: load_bars(tf, "2019-01-01", "2019-04-30") for tf in ("5m", "1h")}


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(set(EVENT_BLOCKS) - {"wedges"}))
def test_no_future_real(real, name):
    m5 = real["5m"]
    n = len(m5)
    guards.assert_no_future_events(EVENT_BLOCKS[name], m5, 5, [n // 2, 3 * n // 4, n - 1])


@pytest.mark.data
def test_no_future_wedges_real():
    """Wedges are rare (a few a year): the whole of 2019."""
    from machine.data.bars import load_bars
    m5 = load_bars("5m", "2019-01-01", "2019-12-31")
    n = len(m5)
    guards.assert_no_future_events(_wedges, m5, 5, [n // 2, 3 * n // 4, n - 1])


@pytest.mark.data
def test_no_future_hour_lines_real(real):
    h1 = real["1h"]
    n = len(h1)
    guards.assert_no_future_events(lambda x: TL.lines(x, "1h"), h1, 60, [n // 2, 3 * n // 4, n - 1])


# ---- the compiler's new steps ----------------------------------------------------------------------------------------

EVERY_SWING = {"trendline_swing_turn_adr": 0.0}     # a version: lines through every swing (more of them on fake bars)


@pytest.fixture(scope="module")
def m1():
    return fake_m1(days=30, seed=3)


@pytest.fixture(scope="module")
def market(m1):
    return fake_market(m1)


def touch_form(**kw) -> dict:
    d = dict(
        id="idea-920", title="the trendline touch (a test form)", words="sell at the 3rd touch of a steep falling line",
        timeframe="5m", directions=["buy", "sell"],
        steps=[dict(id="touch", what="the 3rd / 4th touch of a steep trendline", block="trendline_touch", tf="5m")],
        entry=dict(how="limit", at="touch.entry", max_wait_bars=10),
        exit=dict(kind="hedge", at="touch.hedge"),
        targets=[dict(kind="next_liquidity")],
        sources=[dict(ref="strategy.md section 9 (Type 2), #127")])
    d.update(kw)
    return d


def p1_form(**kw) -> dict:
    d = dict(
        id="idea-921", title="the triangle false break (a test form)", words="triangle, false break, break, gap",
        timeframe="5m", directions=["buy", "sell"],
        steps=[dict(id="tri", what="a triangle after a big move", block="triangle", tf="5m"),
               dict(id="fb", what="a false break of its boundary", block="false_breakout", tf="5m",
                    max_wait_bars="no limit", params={"of": "tri"}),
               dict(id="brk", what="structure breaks back", block="break_of_structure", tf="5m", max_wait_bars=30),
               dict(id="gap", what="the first gap", block="gap_of_the_move", tf="5m", max_wait_bars=6)],
        entry=dict(how="limit", at="gap.near", max_wait_bars="no limit"),
        exit=dict(kind="hedge", at="brk.extreme"),
        targets=[dict(kind="price", at="fb.opposite")],
        sources=[dict(ref="strategy.md section 4 (P1)")])
    d.update(kw)
    return d


def wedge_form(**kw) -> dict:
    return type1(id="idea-922", title="the wedge (a test form)", words="a line held, price out the other side, gap",
                 steps=[dict(id="wedge", what="a trendline held, price out the other side", block="wedge", tf="5m"),
                        dict(id="gap", what="the gap of the break", block="gap_of_the_move", tf="5m",
                             max_wait_bars=6)],
                 exit=dict(kind="hedge", at="wedge.line"), do_not_enter=[dict(name="target_reached_first")], **kw)


def test_the_touch_entry_waits_for_his_word_steep(market):
    """#127 + four-boxes B2-01: never guess the number - the idea is BLOCKED on 'steep' until he answers."""
    idea = Idea.load(touch_form())
    with pytest.raises(C.MissingBlockError, match="steep"):
        C.compile_idea(idea, market)
    assert [m[0] for m in C.check_without_bars(idea)] == ["steep"]
    assert "L9b" in C.check_without_bars(idea)[0][2]


def test_the_touch_entry_compiles_when_the_form_names_the_number(market):
    idea = Idea.load(touch_form(settings={"trendline_min_steepness_adr": 0.2}))
    out = C.compile_idea(idea, market, trace=True)
    fp = settings.fingerprint(out.resolved)
    e = market.touch_entries("5m", fp)
    ok = e[(e["steepness"] >= 0.2) & e["trend_visible"] & e["touches_apart"]]
    assert len(ok) and sum(out.funnel["touch"].values()) == len(ok)
    assert out.signals
    for s in out.signals:
        assert (s.stop > s.entry_price) if s.direction == "sell" else (s.stop < s.entry_price)


def test_the_triangle_waits_for_his_words_big_move_and_compiles_when_named(market):
    with pytest.raises(C.MissingBlockError, match="big_move"):
        C.compile_idea(Idea.load(p1_form()), market)
    out = C.compile_idea(Idea.load(p1_form(settings={**EVERY_SWING, "big_move_adr": 0.0})), market, trace=True)
    assert out.signals
    for r in out.trace:
        if r["signal"]:
            f = r["facts"]
            assert f["fb"]["known_at"] > f["tri"]["known_at"] and f["brk"]["known_at"] >= f["fb"]["known_at"]
            assert f["gap"]["first_bar"] >= f["fb"]["bar"]                     # the gap never before the poke


def test_the_wedge_and_a_false_break_of_his_liquidity_compile(market):
    out = C.compile_idea(Idea.load(wedge_form(settings=EVERY_SWING)), market, trace=True)
    assert out.signals
    for r in out.trace:
        if r["signal"]:
            f = r["facts"]
            assert f["gap"]["bar"] >= f["wedge"]["bar"]                        # its 3rd candle at or after the break
    p2 = type1(id="idea-923")
    p2["steps"] = [dict(id="fb", what="a false break of his liquidity", block="false_breakout", tf="5m")] + \
        p2["steps"][1:]
    p2["steps"][1] = {**p2["steps"][1], "params": {"after": "fb"}}
    p2["do_not_enter"] = [dict(name="target_reached_first")]
    out2 = C.compile_idea(Idea.load(p2), market)
    assert out2.signals and sum(out2.funnel["fb"].values()) > 0


def test_trendlines_are_a_kind_of_liquidity_now(market):
    """A sweep of a trendline starts his Type 1 like any obvious level (#69 'a trend line fake out ... liquidity')."""
    out = C.compile_idea(Idea.load(type1(settings=EVERY_SWING, steps=[{**type1()["steps"][0],
                                                                      "params": {"kinds": ["trendline"]}}]
                                         + type1()["steps"][1:])), market)
    assert sum(out.funnel["sweep"].values()) > 0


def test_the_line_in_the_path_on_a_hand_made_market(monkeypatch):
    """#69 on the falling line (at 101.75 on candle 120): with the hour AND the day against a buy, the line becomes the
    target when it is 1 hedge distance or more away, else no trade; with them not both against, nothing changes."""
    b = falling(250)
    mk = C.Market({"5m": b})
    monkeypatch.setattr(mk, "ruler", lambda tf, fp: ruler(b))
    fp = settings.fingerprint()
    run = C._Run(mk, Idea.load(type1()), "buy", fp)
    p = C.LineInPath.Params()
    at = t(120) + M5
    line = round(107.0 - 0.075 * 70, 2)
    monkeypatch.setattr(mk, "bias_at", lambda tf, fp, when: -1)                  # both DOWN: against a buy
    assert C.LineInPath.retarget(run, p, {}, at, 100.0, 99.0, [103.0, 105.0]) == [line, 105.0]
    assert "50-50" in C.LineInPath.retarget(run, p, {}, at, 101.5, 100.9, [103.0])
    assert C.LineInPath.retarget(run, p, {}, at, 100.0, 99.0, [101.0]) == [101.0]          # beyond the target
    monkeypatch.setattr(mk, "bias_at", lambda tf, fp, when: -1 if tf == "1h" else 0)   # the day not against
    assert C.LineInPath.retarget(run, p, {}, at, 100.0, 99.0, [103.0]) == [103.0]


def test_the_line_in_the_path_in_a_compile(market):
    base = C.compile_idea(Idea.load(type1()), market)
    out = C.compile_idea(Idea.load(type1(do_not_enter=type1()["do_not_enter"] + [dict(name="line_in_path")])),
                         market)
    fp = settings.fingerprint(out.resolved)
    before = {(s.direction, s.known_at): s for s in base.signals}
    for s in out.signals:
        b = before[(s.direction, s.known_at)]
        if s.targets != b.targets:
            against = 1 if s.direction == "sell" else -1          # a sell: the hour AND the day read UP
            assert all(market.bias_at(tf, fp, s.known_at) == against for tf in ("1h", "1d"))
            assert len(s.targets) == len(b.targets)                # the lots stay the same
    assert len(base.signals) - len(out.signals) == out.dropped.get(
        ("do not enter: line_in_path", "a trendline stands between the entry and the target and the hour and the day "
                                       "are both against the trade - 50-50, no trade (#69)"), 0)


def test_compiled_signals_with_the_new_steps_do_not_peek(m1):
    forms = [touch_form(settings={"trendline_min_steepness_adr": 0.2}),
             p1_form(settings={**EVERY_SWING, "big_move_adr": 0.0}), wedge_form(settings=EVERY_SWING),
             type1(do_not_enter=type1()["do_not_enter"] + [dict(name="line_in_path")])]
    ideas = [Idea.load(d) for d in forms]

    def signals_of(bars):
        mk = fake_market(bars)
        rows = [{"form": i.id, **s.model_dump(include={"known_at", "direction", "entry_price", "stop", "targets"})}
                for i in ideas for s in C.compile_idea(i, mk).signals]
        return pd.DataFrame(rows, columns=["form", "known_at", "direction", "entry_price", "stop", "targets"])

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])
