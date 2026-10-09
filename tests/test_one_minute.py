"""L6b.4 - the 1-minute layer (machine/blocks/one_minute.py; trendlines.py on the 1-minute chart; the compiler's step
one_minute). Each rule on hand-made 1-minute candles with a known answer, each table through the cut-off-the-future
test (likhith.md R4) on fake bars, compiled signals with the check cut at many moments, and his two 1-minute charts
(F04, Q98 C) on the real candles when the price data is here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import guards, settings
from machine.blocks import one_minute as OM
from machine.blocks import trendlines as TL
from machine.ideas import compile as C
from machine.ideas.form import Idea
from tests.block_fakes import START, bars_from, fake_market
from tests.test_ideas import fake_m1, type1

M1 = pd.Timedelta(minutes=1)
RULER = 10.0                        # a day's range of $10: through 0.05, tol 0.25, zig-zag turn 0.40 (0.04 x 10)
QUIET = (100.0, 100.2, 99.8, 100.0)


def t(i: int) -> pd.Timestamp:
    return START + i * M1


def chart(n: int, points: dict[int, tuple]) -> pd.DataFrame:
    """n quiet 1-minute candles; points = {candle: (high, low)} or {candle: (open, high, low, close)}."""
    rows = [QUIET] * n
    for i, p in points.items():
        rows[i] = (QUIET[0], p[0], p[1], QUIET[3]) if len(p) == 2 else p
    b = bars_from(rows, tf_minutes=1)
    b["vol"], b["spread"] = 1, 20.0
    return b


def ruler(b) -> np.ndarray:
    return np.full(len(b), RULER)


# His Q98 C shape: a rising line through the lows at 10 (98.0) and 70 (99.2) - $0.02 a candle - passing over two HIGHER
# lows between them (30: 99.0 where the line is 98.4; 50: 99.1 where it is 98.8). Swing highs between keep every low.
Q98 = {3: (101.0, 99.8), 10: (100.2, 98.0), 20: (101.0, 99.8), 30: (100.2, 99.0), 40: (101.0, 99.8),
       50: (100.2, 99.1), 60: (101.0, 99.8), 70: (100.2, 99.2), 80: (101.0, 99.8)}


def q98(n: int = 200, extra: dict | None = None) -> pd.DataFrame:
    return chart(n, {**Q98, **(extra or {})})


def the_line(L: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    return L[(L["a_bar"] == t(a)) & (L["b_bar"] == t(b))]


# ---- the 1-minute line (#100: the same rule as the 5-minute chart) ------------------------------------------------

def test_on_the_1_minute_chart_a_line_joins_two_lows_over_higher_ones():
    b = q98()
    L = OM.lines(b, ruler(b))
    mine = the_line(L, 10, 70)
    assert len(mine) >= 1 and mine["touches"].iat[0] == 2
    assert mine["known_at"].iat[0] == t(72) + M1                   # 2 candles after the touch closes
    assert mine["slope"].iat[0] == pytest.approx(0.02) and mine["way"].iat[0] == "rising"
    with settings.using({"trendline_pairs_1m": "neighbours"}):      # the old engine's way cannot draw it
        assert not len(the_line(OM.lines(b, ruler(b)), 10, 70))


def test_the_5_minute_chart_still_joins_only_neighbours():
    assert TL.rule("5m")["pairs"] == "neighbours" and TL.rule("1h")["pairs"] == "neighbours"
    assert TL.rule("1m")["pairs"] == "any" and TL.rule("1m")["turn"] == 0.04


def test_a_wick_through_the_line_between_its_touches_means_no_line():
    """#92 D on the 1-minute chart too: the line is at 98.6 on candle 40 - a wick to 98.5 goes through it."""
    b = q98(extra={40: (101.0, 98.5)})
    assert not len(the_line(OM.lines(b, ruler(b)), 10, 70))


def test_a_third_touch_makes_it_a_trendline():
    b = q98(extra={90: (101.0, 99.8), 95: (100.2, 99.75)})          # the line is at 99.70 on candle 95
    mine = the_line(OM.lines(b, ruler(b)), 10, 70)
    assert mine["touches"].tolist() == [2, 3] and mine["c_bar"].iat[1] == t(95)


# ---- its life on the 1-minute candles: the 1-minute trendline break --------------------------------------------------

def test_a_wick_through_it_takes_it_and_a_touch_does_not():
    """The line is at 99.6 on candle 90: a low of 99.56 is a touch (0.04 through, #94), 99.54 takes it (0.06)."""
    for low, taken in ((99.56, False), (99.54, True)):
        b = q98(extra={90: (100.2, low)})
        bk = OM.book(b, ruler=ruler(b))
        row = bk[bk["line_id"] == the_line(OM.lines(b, ruler(b)), 10, 70)["line_id"].iat[0]].iloc[0]
        assert bool(row["taken"] == 90) == taken
        assert row["start"] == 73 and row["x0"] == 70
        if not taken:                                  # the quiet lows (99.8) take it when the line is past 99.85
            assert row["taken"] == 103


def test_the_break_is_an_event_known_at_the_close_of_that_minute():
    b = q98(extra={90: (100.2, 99.3)})
    ev = OM.breaks(OM.book(b, ruler=ruler(b)), b)
    lid = the_line(OM.lines(b, ruler(b)), 10, 70)["line_id"].iat[0]
    row = ev[ev["line_id"] == lid].iloc[0]
    assert row["known_at"] == t(91) and row["bar"] == t(90)
    assert row["price"] == pytest.approx(98.0 + 0.02 * 80) and row["side"] == "low"


def test_a_line_older_than_its_life_is_gone_not_taken():
    b = q98(extra={90: (100.2, 99.3)})
    with settings.using({"one_minute_line_life_bars": 10}):
        bk = OM.book(b, ruler=ruler(b))
    row = bk[bk["line_id"] == the_line(OM.lines(b, ruler(b)), 10, 70)["line_id"].iat[0]].iloc[0]
    assert row["taken"] == -1 and row["gone"] == 83 and row["end"] == 83


# ---- the check's lookup ------------------------------------------------------------------------------------------------

def test_between_finds_the_line_on_the_hedge_side_between_the_entry_and_the_hedge():
    b = q98(extra={90: (100.2, 99.3)})
    bk = OM.book(b, ruler=ruler(b))
    lid = the_line(OM.lines(b, ruler(b)), 10, 70)["line_id"].iat[0]
    got = OM.between(bk, 80, 97.5, 99.9, "low")                   # a buy: entry 99.9, hedge 97.5
    assert lid in set(got["line_id"])
    assert got.set_index("line_id").loc[lid, "price"] == pytest.approx(99.4)
    assert lid not in set(OM.between(bk, 80, 99.5, 99.9, "low")["line_id"])      # past the hedge: not between
    assert lid not in set(OM.between(bk, 80, 97.5, 99.9, "high")["line_id"])     # a sell reads lines through highs
    assert lid not in set(OM.between(bk, 71, 97.5, 99.9, "low")["line_id"])      # not known yet (known at 72's close)
    assert lid in set(OM.between(bk, 72, 97.5, 99.9, "low")["line_id"])          # known at the close of 72
    assert lid not in set(OM.between(bk, 90, 97.5, 99.9, "low")["line_id"])      # taken on candle 90
    row, j = OM.first_take(OM.between(bk, 80, 97.5, 99.9, "low").query("line_id == @lid"), 80)
    assert j == 90 and row["line_id"] == lid


# ---- the step one_minute --------------------------------------------------------------------------------------------

def check_form(**kw) -> dict:
    d = type1(id="idea-940", title="grab + gap + the 1-minute check (a test form)")
    d["steps"] = d["steps"] + [dict(id="m1", what="no 1-minute line against the entry (#111)", block="one_minute",
                                    tf="1m", max_wait_bars="no limit", params=kw.pop("params", {}))]
    d.update(kw)
    return d


@pytest.fixture()
def hand_market(monkeypatch):
    b = q98(300, extra={90: (100.2, 99.3)})
    mk = fake_market(b, tfs=("1m", "5m"))
    monkeypatch.setattr(mk, "ruler", lambda tf, fp: np.full(len(mk.bars(tf)), RULER))
    return mk


def _next(mk, direction, facts, after, **params):
    facts = {"sweep": {}, **facts}
    p = C.OneMinuteParams(**{"entry": "gap.near", "hedge": "brk.extreme", "sweep": "sweep", "charts": ("1m",),
                             **params})
    run = C._Run(mk, Idea.load(check_form()), direction, settings.fingerprint())
    step = Idea.load(check_form()).steps[-1]
    return C.OneMinute().next(run, step, p, facts, after, None)


def test_the_check_waits_for_the_line_and_its_take_becomes_the_grab(hand_market):
    """After candle 80 two lines stand between the buy (99.9) and its hedge (97.5): 10 -> 70 (at 99.4) and the flat-ish
    30 -> 50 -> 70 (at 99.25). The first taken is 10 -> 70, on minute 90 (low 99.3, the line 99.6)."""
    facts = {"gap": {"near": 99.9}, "brk": {"extreme": 97.5}}
    got = _next(hand_market, "buy", facts, t(81))
    assert isinstance(got, C._Resweep) and got.step == "sweep"
    ev = got.event
    assert ev["level_id"] == f"line_low:1m:{t(10):%Y-%m-%d %H:%M}:{t(70):%Y-%m-%d %H:%M}"
    assert ev["known_at"] == t(95) and ev["bar"] == t(90)           # the 5-minute candle 90..94 that holds minute 90
    assert ev["price"] == pytest.approx(99.6) and ev["extreme"] == 99.3 and ev["level_tf"] == "1m"
    assert ev["kind"] == "trendline" and ev["waited_from"] == t(81)


def test_the_check_passes_at_once_when_no_line_is_against_the_entry(hand_market):
    pas = {"known_at": t(81), "lines": 0}
    assert _next(hand_market, "buy", {"gap": {"near": 99.9}, "brk": {"extreme": 99.5}}, t(81)) == pas   # lines past it
    assert _next(hand_market, "sell", {"gap": {"near": 99.9}, "brk": {"extreme": 100.9}}, t(81)) == pas  # other side
    assert _next(hand_market, "buy", {"gap": {"near": 99.9}, "brk": {"extreme": 97.5}}, t(25)) == \
        {"known_at": t(25), "lines": 0}                                                                  # none known


def test_after_a_take_only_the_lines_known_at_the_first_check_count(hand_market):
    """His F04 answer: the gap after the take is 'the actual entry' - a line born after the first check does not make
    it wait again (settings one_minute_new_lines_after_a_take = False). Every line here is known after t(31)."""
    facts = {"gap": {"near": 99.9}, "brk": {"extreme": 97.5}, "sweep": {"first_check": t(31)}}
    assert _next(hand_market, "buy", facts, t(81)) == {"known_at": t(81), "lines": 0}
    with settings.using({"one_minute_new_lines_after_a_take": True}):
        assert isinstance(_next(hand_market, "buy", facts, t(81)), C._Resweep)
    facts["sweep"]["first_check"] = t(81)                           # known by the first check: it still counts
    got = _next(hand_market, "buy", facts, t(81))
    assert isinstance(got, C._Resweep) and got.event["first_check"] == t(81)


def test_the_check_is_only_for_a_5_minute_gap_entry(hand_market):
    """strategy.md section 5: '1-min = never the entry chart; used as a check before a 5-min gap entry'."""
    bad = check_form(entry=dict(how="limit", at="brk.level", max_wait_bars="no limit"))
    with pytest.raises(C.CompileError, match="only for a 5-minute gap entry"):
        C.compile_idea(Idea.load(bad), hand_market)
    wedge = check_form(exit=dict(kind="hedge", at="wedge.line"))     # the take must replace a liquidity-taken step
    wedge["steps"] = [dict(id="wedge", what="a line held, out the other side", block="wedge", tf="5m")] + \
        wedge["steps"][2:]
    with pytest.raises(C.CompileError, match="needs a 'sweep' or 'false_breakout' step before it"):
        C.compile_idea(Idea.load(wedge), hand_market)
    with pytest.raises(C.CompileError, match="1m bars"):
        C.compile_idea(Idea.load(check_form()), fake_market(fake_m1(days=3), tfs=("5m",)))


def test_the_check_is_no_longer_a_missing_block():
    assert "one_minute" in C.BLOCKS and "one_minute" not in C.PLANNED
    assert C.check_without_bars(Idea.load(check_form())) == []


# ---- in a compile (fake 1-minute market) --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m1():
    return fake_m1(days=30, seed=3)


@pytest.fixture(scope="module")
def market(m1):
    return fake_market(m1, tfs=("1m", "5m", "1h", "1d", "1w"))


def test_in_a_compile_a_setup_either_passes_unchanged_or_waits_for_the_take(market):
    base = C.compile_idea(Idea.load(type1()), market)
    out = C.compile_idea(Idea.load(check_form()), market, trace=True)
    assert out.waits > 0 and "waited" in out.text()
    before = {(s.direction, s.known_at, s.entry_price, s.stop) for s in base.signals}
    waited = 0
    for r in out.trace:
        if not r["signal"]:
            continue
        s = next(x for x in out.signals if x.id == r["signal"])
        f = r["facts"]
        if r["waits"] == 0:
            assert (s.direction, s.known_at, s.entry_price, s.stop) in before      # the check changed nothing
            continue
        waited += 1
        sw = f["sweep"]
        assert sw["level_id"] == r["wait_for"][-1][0] and sw["known_at"] == r["wait_for"][-1][1]
        assert f["brk"]["known_at"] > sw["waited_from"]                             # the break came after the wait
        assert f["gap"]["first_bar"] >= sw["bar"]                                   # the gap after the take
        assert f["m1"]["lines"] == 0                                                 # and nothing left against it
    assert waited > 0


def test_compiled_signals_with_the_1_minute_check_do_not_peek(m1):
    idea = Idea.load(check_form())

    def signals_of(bars):
        mk = fake_market(bars, tfs=("1m", "5m", "1h", "1d", "1w"))
        rows = [s.model_dump(include={"known_at", "direction", "entry_price", "stop", "targets"})
                for s in C.compile_idea(idea, mk).signals]
        return pd.DataFrame(rows, columns=["known_at", "direction", "entry_price", "stop", "targets"])

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])


# ---- the zone break's liquidity inside the zone reads the 1-minute lines (#98 C, #99; L6b.2's step) -------------------

def test_a_zone_break_without_1_minute_bars_says_the_lines_were_not_read(m1, market):
    from tests.test_zones import ZB
    without = C.compile_idea(Idea.load(ZB), fake_market(m1))
    assert any("1-minute lines inside a zone" in n for n in without.notes)
    with_1m = C.compile_idea(Idea.load(ZB), market)
    assert not any("1-minute lines inside a zone" in n for n in with_1m.notes)
    off = C.compile_idea(Idea.load({**ZB, "settings": {"zone_break_inner_1m_lines": False}}), market)
    assert {(s.known_at, s.entry_price) for s in off.signals} == \
        {(s.known_at, s.entry_price) for s in without.signals}                 # off = L6b.2's own zone break


def test_zone_break_signals_with_the_1_minute_lines_do_not_peek(m1):
    from tests.test_zones import ZB
    idea = Idea.load(ZB)

    def signals_of(bars):
        mk = fake_market(bars, tfs=("1m", "5m", "1h", "1d", "1w"))
        rows = [s.model_dump(include={"known_at", "direction", "entry_price", "stop", "targets"})
                for s in C.compile_idea(idea, mk).signals]
        return pd.DataFrame(rows, columns=["known_at", "direction", "entry_price", "stop", "targets"])

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])


# ---- the cut-off-the-future test for every table -------------------------------------------------------------------

@pytest.fixture(scope="module")
def walk():
    return fake_m1(days=14, seed=11)


def _lines(b):
    return OM.lines(b)


def _breaks(b):
    return OM.breaks(OM.book(b), b)


def _tris(b):
    return OM.triangles(b)


EVENT_BLOCKS = {"lines": _lines, "breaks": _breaks, "triangles": _tris}


@pytest.mark.parametrize("name", sorted(EVENT_BLOCKS))
def test_no_future_fake(name, walk):
    n = len(walk)
    guards.assert_no_future_events(EVENT_BLOCKS[name], walk, 1, [n - 2000, n - 1100, n - 1])


def test_between_uses_only_what_was_known(walk):
    full = OM.book(walk)
    rng = np.random.default_rng(5)
    n = len(walk)
    seen = 0
    for p in (n - 1500, n - 400):
        part = OM.book(walk.iloc[:p])
        for i in range(p - 300, p, 7):
            c = float(walk["close"].iat[i])
            for side, d in (("low", -1), ("high", 1)):
                far = c + d * float(rng.uniform(1, 4))
                lo, hi = min(c, far), max(c, far)
                a = OM.between(full, i, lo, hi, side)[["line_id", "price"]].round(6).values.tolist()
                b = OM.between(part, i, lo, hi, side)[["line_id", "price"]].round(6).values.tolist()
                assert a == b, (p, i, side)
                seen += len(a)
    assert seen > 0


# ---- his two 1-minute charts on the real candles --------------------------------------------------------------------

def _srv(ist: str) -> pd.Timestamp:
    from machine.data import clock
    return clock.utc_to_server(pd.Timestamp(ist, tz="Asia/Kolkata"))


@pytest.mark.data
@pytest.mark.parametrize("name, first_day, a, b, known", [
    ("F04 #111", "2019-02-20", "2019-03-11 17:12", "2019-03-11 17:36", "2019-03-11 17:39"),
    ("Q98 C #98", "2017-05-20", "2017-06-09 20:46", "2017-06-09 21:35", "2017-06-09 21:38")])
def test_his_1_minute_lines_are_drawn(name, first_day, a, b, known):
    from machine.data.bars import load_bars
    m1 = load_bars("1m", first_day, _srv(known) + pd.Timedelta(hours=1))
    L = OM.lines(m1)
    mine = L[(L["a_bar"] == _srv(a)) & (L["b_bar"] == _srv(b))]
    assert len(mine) and mine["known_at"].min() == _srv(known), name


@pytest.mark.data
def test_his_f04_setup_waits_for_the_line_and_its_take():
    """F04 (#111): the buy at 1295.17 (the gap known 17:55 IST), hedge at the grab's low 1294.06. His line (17:12 ->
    17:36 lows) is between them: wait. It is taken at 17:58 (a low of 1294.70, the line 1294.81 there) - the grab is
    now the 5-minute candle 17:55-18:00, known at its close."""
    from machine.data.bars import load_bars
    m1 = load_bars("1m", "2019-02-20", "2019-03-11 16:00")
    mk = C.Market({"1m": m1, "5m": load_bars("5m", "2019-02-20", "2019-03-11 16:00")})
    got = _next(mk, "buy", {"gap": {"near": 1295.17}, "brk": {"extreme": 1294.06}}, _srv("2019-03-11 17:55"))
    assert isinstance(got, C._Resweep)
    ev = got.event
    assert ev["level_id"].startswith("line_low:1m:2019-03-11 13:42")          # touch a = 17:12 IST
    assert ev["known_at"] == _srv("2019-03-11 18:00") and ev["bar"] == _srv("2019-03-11 17:55")
    assert ev["price"] == pytest.approx(1294.06 + 0.39 / 24 * 46, abs=0.01) and ev["extreme"] == 1294.70


@pytest.mark.data
def test_his_f04_trade_moves_to_the_entry_he_named():
    """#111 F04, the whole way through the compiler: his Type 1 makes the buy he criticised (17:55 IST, 1295.17, hedge
    1294.06); with the 1-minute check it waits for his line, and the next gap - 18:35 IST, 1295.23-1295.61 - is the
    buy he called 'the actual entry' (hedge at the take's grab low 1293.51)."""
    mk = C.Market.load("2019-02-18", "2019-03-12", tfs=("1m", "5m", "1h", "1d", "1w"))

    def buys(form):
        out = C.compile_idea(Idea.load(form), mk)
        return {(s.known_at, s.entry_price, s.stop) for s in out.signals if s.direction == "buy"}

    assert (_srv("2019-03-11 17:55"), 1295.17, 1294.06) in buys(type1(directions=["buy"]))
    got = buys(check_form(directions=["buy"]))
    assert (_srv("2019-03-11 17:55"), 1295.17, 1294.06) not in got
    assert (_srv("2019-03-11 18:40"), 1295.61, 1293.51) in got
    literal = check_form(directions=["buy"], settings={"one_minute_new_lines_after_a_take": True})
    assert (_srv("2019-03-11 18:40"), 1295.61, 1293.51) not in buys(literal)   # the literal reading waits again


@pytest.mark.data
def test_his_q98c_zone_break_waits_for_his_1_minute_line():
    """#98 C: the zone broke at 21:50 IST (the machine's break). His line (20:46 -> 21:35 lows) lies inside the zone:
    wait for it to be taken (22:19, the line at 1268.39), then the first 5-minute gap - not the plain retest."""
    mk = C.Market.load("2017-04-20", "2017-06-12", tfs=("1m", "5m", "1h", "1d", "1w"))
    form = Idea.load(type1(directions=["buy"]))
    got = {}
    for on in (False, True):
        with settings.using({"zone_break_inner_1m_lines": on}):
            run = C._Run(mk, form, "buy", settings.fingerprint())
            t = mk.zone_breaks(run.fp)
            t = t[(t["direction"] == "buy") & (t["known_at"] == _srv("2017-06-09 21:50"))]
            got[on] = [C.ZoneBreak()._event(run, r) for r in t.itertuples()]
    assert got[False] and all(e["how"] == "retest" for e in got[False])
    for e in got[True]:
        assert e["how"] == "after_inner_take" and e["inner"] == pytest.approx(1268.39, abs=0.02)
        assert e["known_at"] > _srv("2017-06-09 22:19") and e["hedge"] < e["entry"]
