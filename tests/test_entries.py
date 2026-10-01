"""L19b - his entry types as forms (machine/entries/), and what the testers needed for them: a check made while an
order waits (engine.replay's cancel_by: #98 B, #87), the first hedge of a trendline entry moving with its line (T12
M27), his entries closing hedged legs (M10), the second gap of a move (#124), the grab's extreme of a false break.
Fake 1-minute bars unless marked `data` (tests/test_ideas.fake_m1); hand-made paths from tests/test_hedging."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import config, guards, settings
from machine.entries import closers, forms
from machine.ideas import compile as C
from machine.ideas.form import FormError, Idea, with_readings
from machine.referee import prior
from machine.tester import engine, hedging
from machine.tester.engine import CHECK_BEFORE_FILL, MINUTE_NS
from machine.tester.models import Signal
from tests.test_hedging import Market as HandMarket
from tests.test_hedging import at, order, path
from tests.test_ideas import fake_m1, type1


@pytest.fixture(scope="module")
def m1():
    return fake_m1(days=30, seed=3)


@pytest.fixture(scope="module")
def market(m1):
    return C.Market.from_m1(m1)


def ns(t) -> int:
    return pd.Timestamp(t).as_unit("ns").value


# ================================================================================================================
# the forms
# ================================================================================================================

def test_his_six_entry_types_are_forms_and_every_reading_compiles(market):
    assert list(forms.ENTRY_TYPES) == ["entry-type1", "entry-second-gap", "entry-trendline-touch", "entry-zone-break",
                                       "entry-triangle-false-break", "entry-zone-entry"]
    for e in forms.ENTRY_TYPES.values():
        idea = e.idea()
        assert idea.id == e.id and idea.words and idea.sources
        for rd in e.readings():
            assert C.check_without_bars(with_readings(idea, rd)[0]) == []   # every block of every reading is built
            out = C.compile_idea(idea, market, readings=rd)
            assert out.idea_id == e.id
            # only the second gap moves off one of his defaults (the first gap) - his own #124 exception, said in
            # its report; every other entry type is his rules as they stand
            got = {b.split(" = ")[0] for b in out.resolved.breaks_his_rules}
            if e is forms.SECOND_GAP:
                assert "which_gap_of_the_move" in got <= {"which_gap_of_the_move", "the_second_gap_needs_the_day"}
            else:
                assert not got


def test_type1_is_the_hedging_testers_own_grab_and_gap():
    assert forms.TYPE1.idea().rule() == Idea.load(hedging.HIS_GRAB_GAP).rule()


def test_an_entry_type_is_never_an_idea_of_the_store():
    """His entry types are forms (entry-...), not ideas from the phone: plan/work/ideas/ refuses them."""
    from machine.ideas import store
    with pytest.raises(store.StoreError):
        store.folder("entry-type1")


def test_his_open_words_are_readings_and_never_guessed(market):
    """R13: 'steep' (B2-01), 'big move' (B2-02), 'very strong' (B2-03) - a compile without a reading is refused; the
    readings are the numbers the queued questions show him."""
    for e, n, word in ((forms.TRENDLINE_TOUCH, 4, "B2-01"), (forms.TRIANGLE_FALSE_BREAK, 3, "B2-02"),
                       (forms.SECOND_GAP, 2, "B2-03")):
        assert len(e.readings()) == n and word in e.open_word
        with pytest.raises(FormError, match="never guesses"):
            C.compile_idea(e.idea(), market)
    steep = (config.WORK_DIR / "question-queue" / "B2-01-steep.md").read_text(encoding="utf-8")
    big = (config.WORK_DIR / "question-queue" / "B2-02-big-move.md").read_text(encoding="utf-8")
    for u, text in ((forms.TRENDLINE_TOUCH.idea().unclear[0], steep), (forms.TRIANGLE_FALSE_BREAK.idea().unclear[0],
                                                                      big)):
        for r in u.readings:
            v = next(iter(r.changes.values()))
            assert f"| {r.id} | {v:g}" in text, (r.id, v)            # the row of the question's own table


def test_every_entry_type_is_already_seen_by_the_old_project():
    """CLAUDE.md rule 18: each one is in plan/work/prior-tries.csv on 2016-2023, so it can never be 'works' here."""
    for e in forms.ENTRY_TYPES.values():
        seen = prior.already_seen(e.seen, config.TUNE_YEARS)
        assert seen.seen, e.id
        assert set(e.seen_rows) <= {r.idea_key for r in seen.rows}, e.id


# ================================================================================================================
# the tester: a check made while an order waits, and the faster fill search
# ================================================================================================================

def test_a_check_before_the_fill_cancels_a_waiting_order_then_and_only_then():
    m1 = path((0, 2001), (10, 2001), (20, 1999), (40, 2003))          # a buy limit at 2000.20 fills at minute 14
    s = order("b1", 0, "buy", 2000.20, 1995, 2010)
    fill = pd.Timestamp(engine.replay([s], m1, "5m").trades[0].fill_ts)
    assert fill == at(14)
    early = engine.replay([s], m1, "5m", cancel_by={"b1": ns(at(12))})
    assert early.trades == [] and early.not_filled == [("b1", CHECK_BEFORE_FILL)]
    late = engine.replay([s], m1, "5m", cancel_by={"b1": ns(at(30))})
    assert len(late.trades) == 1 and pd.Timestamp(late.trades[0].fill_ts) == fill
    # the minute that opens AT the check's moment cannot fill: the order is gone by then
    assert engine.replay([s], m1, "5m", cancel_by={"b1": ns(fill)}).trades == []
    assert len(engine.replay([s], m1, "5m", cancel_by={"b1": ns(fill) + MINUTE_NS}).trades) == 1
    # the place is free from that moment: a later signal is not skipped
    s2 = order("b2", 13, "buy", 1999.20, 1990, 2010)
    rep = engine.replay([s, s2], m1, "5m", cancel_by={"b1": ns(at(12))})
    assert [t.signal_id for t in rep.trades] == ["b2"]


def _full_scan_fill(book, s: Signal, t: int):
    """The fill search as it was before L19b (the whole window at once) - the chunked one must agree with it."""
    i0 = int(np.searchsorted(book.t, t, "left"))
    buy = s.direction == "buy"
    j = int(np.searchsorted(book.tf_open, t, "left"))
    deadline = book.tf_open[j + s.expiry_bars] if j + s.expiry_bars < len(book.tf_open) else engine.NEVER
    i1 = int(np.searchsorted(book.t, deadline, "left")) if deadline != engine.NEVER else book.n
    seg = slice(i0, i1)
    L = s.entry_price
    hit = (book.lo[seg] + book.sp[seg] <= L + engine.EPS) if buy else (book.h[seg] >= L - engine.EPS)
    if s.cancel_at is not None:
        gone = (book.h[seg] >= s.cancel_at - engine.EPS) if buy else (book.lo[seg] <= s.cancel_at + engine.EPS)
        if gone.any() and (not hit.any() or np.argmax(gone) <= np.argmax(hit)):
            return ("cancel", i0 + int(np.argmax(gone)))
    if not hit.any():
        return ("none", None)
    return ("fill", i0 + int(np.argmax(hit)))


def test_the_chunked_fill_search_finds_what_the_whole_scan_finds(m1):
    book = engine.bars_book(m1, "5m")
    rng = np.random.default_rng(11)
    close = m1["close"].to_numpy()
    for k in range(400):
        i = int(rng.integers(0, len(m1) - 10))
        buy = bool(rng.random() < 0.5)
        px = float(close[i])
        entry = round(px - rng.uniform(0, 3) if buy else px + rng.uniform(0, 3), 2)
        cancel = None if rng.random() < 0.3 else round(entry + rng.uniform(0.5, 4) * (1 if buy else -1), 2)
        s = Signal(id=f"s{k}", known_at=m1.index[i], direction="buy" if buy else "sell", entry_type="limit",
                   entry_price=entry, expiry_bars=int(rng.choice([1, 5, 40, 10**6])),
                   stop=round(entry - 5 if buy else entry + 5, 2), targets=(round(entry + 8 if buy else entry - 8, 2),),
                   cancel_at=cancel, idea_id="idea-test", version_id="v1")
        want = _full_scan_fill(book, s, ns(s.known_at))
        got = book.fill(s, ns(s.known_at))
        if want[0] == "fill":
            assert isinstance(got, engine.Filled) and got.i == want[1], s
        elif want[0] == "cancel":
            assert isinstance(got, engine.Missed) and got.why == "target reached first", s
            assert got.release == int(book.t[want[1]]) + MINUTE_NS
        else:
            assert isinstance(got, engine.Missed) and got.why in ("expired", "window ended"), s


# ================================================================================================================
# the hedging tester: the trendline hedge moves with its line (M27)
# ================================================================================================================

def _line_trade(slope=None, dial="mixed", levels=()):
    """A buy at 2000.20 (fills at minute 5, the bid at 2000.00), hedge point 1998, target 2010; price climbs to 2002 by
    minute 65, then falls to 1990 by minute 95."""
    m1 = path((0, 2001), (5, 2000), (65, 2002), (95, 1990), tail=20)
    mk = HandMarket(m1, dial=dial, levels=levels)
    s = order("t1", 0, "buy", 2000.20, 1998, 2010)
    rep = hedging.replay([s], mk, "5m", moves={} if slope is None else {"t1": slope})
    return rep.trades[0]


def test_the_first_hedge_of_a_trendline_entry_moves_with_the_line():
    """#69, strategy.md section 9: 'the hedge moves with the line' - a rising line (+$0.10 a 5-minute candle) lifts the
    buy's hedge from 1998 to about 1999.5 by the time price falls through it."""
    fixed = _line_trade()
    moved = _line_trade(slope=0.10)
    h_fixed = next(x for x in fixed.legs if x.why_open == "hedge")
    h_moved = next(x for x in moved.legs if x.why_open == "hedge")
    assert h_fixed.open_px == pytest.approx(1998.0, abs=0.011)
    assert 1999.3 <= h_moved.open_px <= 1999.6 and moved.line_moves > 0 and fixed.line_moves == 0
    assert h_moved.open_i < h_fixed.open_i                             # hedged sooner, higher


def test_the_line_only_tightens_the_hedge():
    """A falling line would move a buy's hedge DOWN - away from price: it never does (only tighter)."""
    t = _line_trade(slope=-0.10)
    assert next(x for x in t.legs if x.why_open == "hedge").open_px == pytest.approx(1998.0, abs=0.011)
    assert t.line_moves == 0


def test_a_trendline_entry_gets_no_room_its_hedge_stays_at_the_line():
    """M06 vs M27: with the hour and the day with the trade a buy gets room to the next liquidity - but a trendline
    entry keeps its hedge past the line. #69, the day and the hour both bullish: 'I will buy at the ... third touch
    ... keep the hedging below the trend line itself ... trailing like stop loss'."""
    lv = [(1995.0, "low", 0)]
    assert _line_trade(dial="with", levels=lv).room
    t = _line_trade(slope=0.10, dial="with", levels=lv)
    assert not t.room and any(r["what"] == "line" for r in t.story)


def test_the_compiler_hands_the_line_of_a_touch_entry_to_the_hedging_tester(market):
    out = C.compile_idea(forms.TRENDLINE_TOUCH.idea(), market, readings={"B2-01": "A"})
    assert set(out.hedge_lines) == {s.id for s in out.signals}
    for d in out.hedge_lines.values():
        assert d["tf"] == "5m" and d["slope"] != 0 and d["line_id"].startswith("line_")


# ================================================================================================================
# his entries close legs (M10), and the checks while an order waits
# ================================================================================================================

def test_the_entries_that_close_a_leg_follow_the_setting_and_wait_for_his_words():
    got, waits = closers.closer_entries()
    assert [e.id for e, _ in got] == ["entry-type1", "entry-zone-break", "entry-zone-entry"]
    assert set(waits) == {"entry-trendline-touch", "entry-triangle-false-break"}
    assert "B2-01" in waits["entry-trendline-touch"] and "B2-02" in waits["entry-triangle-false-break"]
    got5, waits5 = closers.closer_entries({"entry-trendline-touch": {"B2-01": "A"},
                                           "entry-triangle-false-break": {"B2-02": "B"}})
    assert len(got5) == 5 and waits5 == {}
    with settings.using({"which_entries_can_close_a_position": "type1"}):
        got1, _ = closers.closer_entries()
        assert [e.id for e, _ in got1] == ["entry-type1"]


def _sig(sid, minute, direction):
    return order(sid, minute, direction, 2000.0 if direction == "buy" else 2001.0,
                 1995.0 if direction == "buy" else 2006.0, 2010.0 if direction == "buy" else 1990.0)


def test_an_opposite_setup_after_the_order_cancels_it_then():
    """#98 B: the first OPPOSITE setup known strictly after the zone-break order - not one known before or with it."""
    zb = [_sig("zb", 60, "buy")]
    t1 = [_sig("s0", 30, "sell"), _sig("s1", 60, "sell"), _sig("b1", 70, "buy"), _sig("s2", 90, "sell"),
          _sig("s3", 120, "sell")]
    assert closers.opposite_setup(zb, t1) == {"zb": ns(at(90))}
    assert closers.opposite_setup(zb, t1[:3]) == {}
    with settings.using({"zone_break_opposite_setup_veto": False}):
        assert closers.fill_checks(forms.ZONE_BREAK, zb, None, "", t1) == {}
    assert closers.fill_checks(forms.ZONE_BREAK, zb, None, "", t1) == {"zb": ns(at(90))}
    assert closers.fill_checks(forms.TYPE1, zb, None, "", t1) == {}            # Type 1 has no such check


@pytest.mark.data
def test_the_gap_excuse_ends_at_a_candle_close_after_the_order():
    """#87 at the fill, real 2019 zone entries: every cancel moment is later than its order and on a 5-minute close."""
    mk = C.Market.load("2019-01", "2019-12")
    sig = C.compile_idea(forms.ZONE_ENTRY.idea(), mk).signals
    fp = settings.fingerprint(settings.resolve({}))
    got = closers.gap_excuse(sig, mk, fp)
    by = {s.id: s for s in sig}
    assert sig
    for sid, t in got.items():
        assert t > ns(by[sid].known_at) and t % (5 * MINUTE_NS) == 0


# ================================================================================================================
# the compiler: the second gap (#124), the grab's extreme of a false break
# ================================================================================================================

SECOND_ALWAYS = {"which_gap_of_the_move": "second", "the_second_gap_needs_the_day": False}


def test_the_second_gap_is_a_newer_order_in_place_of_the_first(market):
    """#124 'or else it should wait for the first': the order waits at the first gap; a second gap that forms later
    (the agreement always true here) is a NEWER signal at its own moment - the tester's cancel-and-take-new (M39)."""
    out = C.compile_idea(Idea.load(type1(settings=SECOND_ALWAYS)), market, trace=True)
    later = [r for r in out.trace if r.get("in_place_of") and r["signal"]]
    assert later
    firsts = {(r["direction"], r["start"]): r for r in out.trace if not r.get("in_place_of") and r["signal"]}
    by_id = {s.id: s for s in out.signals}
    checked = 0
    for r in later:
        f = firsts.get((r["direction"], r["start"]))
        if f is None:
            continue                                        # the first gap's own signal was dropped (e.g. a rule)
        a, b = by_id[f["signal"]], by_id[r["signal"]]
        assert b.known_at > a.known_at and b.entry_price != a.entry_price
        assert r["facts"]["gap"]["gap_rank"] == 1 and f["facts"]["gap"]["gap_rank"] == 0
        checked += 1
    assert checked
    plain = C.compile_idea(Idea.load(type1()), market, trace=True)
    assert not any(r.get("in_place_of") for r in plain.trace)
    assert {r["facts"]["gap"]["gap_rank"] for r in plain.trace if r["signal"]} == {0}


def test_second_gap_signals_do_not_peek(m1):
    idea = Idea.load(type1(settings=SECOND_ALWAYS))

    def signals_of(bars):
        rows = [s.model_dump(include={"known_at", "direction", "entry_price", "stop", "targets"})
                for s in C.compile_idea(idea, C.Market.from_m1(bars)).signals]
        return pd.DataFrame(rows, columns=["known_at", "direction", "entry_price", "stop", "targets"])

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])


def test_a_false_breaks_grab_extreme_counts_its_poke(market):
    """P1's hedge point is the grab's extreme: the poke out of the triangle is part of the grab, though the false
    break is known only when price closes back inside (#9, #33, #111)."""
    e = forms.TRIANGLE_FALSE_BREAK
    form = Idea.load({**e.form, "unclear": [], "settings": {"big_move_adr": 0.0, "trendline_swing_turn_adr": 0.0}})
    out = C.compile_idea(form, market, trace=True)
    done = [r for r in out.trace if r["signal"]]
    assert done
    for r in done:
        fb, brk = r["facts"]["fb"]["extreme"], r["facts"]["brk"]["extreme"]
        assert brk >= fb - 1e-9 if r["direction"] == "sell" else brk <= fb + 1e-9


# ================================================================================================================
# the runner and its reports
# ================================================================================================================

@pytest.mark.data
def test_the_runner_makes_the_2x2_and_a_report_in_easy_english(tmp_path, monkeypatch):
    """One month of Type 1, 2 random runs, written into a temporary folder (the trial file goes to conftest's)."""
    from machine.entries import run as R
    from machine.report.build import check_words
    monkeypatch.setattr(R, "OUT_DIR", tmp_path / "L19b")
    out = R.run(["entry-type1"], start="2019-03", end="2019-03", n_runs=2, write=True, log=lambda m: None)
    v = out["versions"][0]
    o = v.out
    assert o["signals"] > 20 and o["result"]["runs"] == 2 and len(o["result"]["boxes"]) == 4
    assert o["referee"]["seen"] and o["referee"]["answer"] == "chance — needs fresh years"
    assert "entry-zone-break" in out["facts"].closers and "entry-trendline-touch" in out["facts"].closers
    text = (R.OUT_DIR / "entry-type1.md").read_text(encoding="utf-8")
    assert check_words(text) == [], check_words(text)
    assert "Already seen" in text and "chance — needs fresh years" in text
    assert check_words((R.OUT_DIR / "README.md").read_text(encoding="utf-8")) == []
    folder = R.OUT_DIR / "entry-type1"
    assert (folder / f"{v.version_id}-boxes.jpg").stat().st_size > 10_000     # the four boxes, his easy English
    assert (folder / f"{v.version_id}-counts.json").is_file() and o["trial"].endswith(f"{v.version_id}.json")
    # the reports are made ONLY from the counts files and the trial files: read back, they give the same page
    versions, closers_used, window = R.load_versions(["entry-type1"])
    assert versions[0].comp is None and window == out["window"]
    assert R.report_md(versions, closers_used, window) == text
