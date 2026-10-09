"""L7a - the idea form, the compiler, the store (machine/ideas/).

The five checks the prompt asks for: readable errors · a deterministic compile · no peeking in compiled signals ·
his FIXED rules refused with his words · every step's max wait honoured. Fake 1-minute bars (a random walk on the
1-cent grid, January 2019) unless marked `data`. The test forms are idea-9xx: idea #1's own form is L7b's job.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from machine import config, guards, rules, settings
from machine.blocks import liquidity as LQ
from machine.data.clock import to_ist
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import FormError, Idea, apply_changes, reading_choices, with_readings
from machine.tester import run
from machine.tester.engine import replay


# ================================================================================================ fakes and forms

def fake_m1(days: int = 25, seed: int = 3, start: str = "2019-01-07 00:00") -> pd.DataFrame:
    """1-minute bars like load_bars('1m'): a random walk on the 1-cent grid, every day, spread 20 points."""
    n = days * 1440
    rng = np.random.default_rng(seed)
    close = 1290 + np.cumsum(rng.normal(0, 0.12, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.06, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.06, n))
    df = pd.DataFrame(np.round(np.c_[open_, high, low, close], 2), columns=["open", "high", "low", "close"],
                      index=pd.date_range(start, periods=n, freq="1min", name="server_time"))
    df["spread"] = 20.0
    df["vol"] = 1
    return df


@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return fake_m1()


@pytest.fixture(scope="module")
def market(m1) -> C.Market:
    return C.Market.from_m1(m1)


def type1(**kw) -> dict:
    """His Type 1 shape (strategy.md section 9): liquidity taken -> break -> the first gap -> the near edge."""
    d = dict(
        id="idea-900", title="grab + gap (a test form)", words="liquidity taken, break, first gap (a test form)",
        timeframe="5m", directions=["buy", "sell"],
        steps=[
            dict(id="sweep", what="price takes a level on the hedge side", block="sweep", tf="5m"),
            dict(id="brk", what="structure breaks the trade's way", block="break_of_structure", tf="5m",
                 max_wait_bars=30),
            dict(id="gap", what="the first gap of the breaking move", block="gap_of_the_move", tf="5m",
                 max_wait_bars=6),
        ],
        entry=dict(how="limit", at="gap.near", max_wait_bars="no limit"),
        exit=dict(kind="hedge", at="brk.extreme"),
        targets=[dict(kind="next_liquidity")],
        do_not_enter=[dict(name="target_already_reached", params={"since": "sweep"}),
                      dict(name="target_reached_first")],
        sources=[dict(ref="strategy.md section 9 (Type 1)")],
    )
    d.update(kw)
    return d


def fib(**kw) -> dict:
    """The prompt's own example: a break on the hour chart -> back to 0.618 within 20 bars."""
    d = dict(
        id="idea-901", title="fib 0.618 (a test form)", words="break of structure on 1h, back to 0.618 (a test form)",
        timeframe="1h", directions=["buy", "sell"],
        steps=[dict(id="fib", what="a higher high / lower low on the hour chart", block="fib_move", tf="1h")],
        entry=dict(how="limit", at="fib.level_0.618", max_wait_bars=20),
        exit=dict(kind="stop", at="fib.level_0.786"),
        targets=[dict(kind="price", at="fib.level_0.5")],
        do_not_enter=[dict(name="entry_already_reached")],
    )
    d.update(kw)
    return d


def comes_back(wait=12, **kw) -> dict:
    """A break -> price comes back to the broken block's level within `wait` bars (the max-wait tests)."""
    d = dict(
        id="idea-902", title="back to the broken level (a test form)", words="a test form", timeframe="5m",
        directions=["buy", "sell"],
        steps=[dict(id="brk", what="structure breaks", block="break_of_structure", tf="5m"),
               dict(id="back", what="price comes back to the broken level", block="comes_back_to", tf="5m",
                    max_wait_bars=wait, params={"at": "brk.level"})],
        entry=dict(how="next_open"),
        exit=dict(kind="stop", at="brk.extreme"),
        targets=[dict(kind="next_liquidity")],
    )
    d.update(kw)
    return d


def pair(**kw) -> dict:
    """Idea #1's SHAPE (plan/work/idea-001/steps.md) - equal highs form -> equal lows wait below -> the highs are taken
    -> break -> first gap. Only to test the blocks; idea #1's real form is L7b's."""
    d = type1(
        id="idea-903", title="the pair (a test form)",
        steps=[
            dict(id="pair", what="equal highs form", block="equal_levels", tf="5m"),
            dict(id="sweep", what="price takes them", block="sweep", tf="5m", max_wait_bars="no limit",
                 params={"of": "pair"}),
            dict(id="other", what="equal lows wait below, not taken", block="level_waiting", tf="5m",
                 max_wait_bars=0, params={"kinds": ["equal"]}),
            dict(id="brk", what="structure breaks down", block="break_of_structure", tf="5m", max_wait_bars=30,
                 params={"after": "sweep"}),
            dict(id="gap", what="the first gap of the breaking move", block="gap_of_the_move", tf="5m",
                 max_wait_bars=6),
        ],
        targets=[dict(kind="price", at="other.price")],
    )
    d.update(kw)
    return d


def london(sessions=("London",), **kw) -> dict:
    """Idea #3's SHAPE (L6-new): "sell when the London high is taken, then a break and the first gap" - his Type 1
    with the level taken fixed to one session's high. Only to test the sweep's params.which_session."""
    d = type1(id="idea-904", title="the London high (a test form)", directions=["sell"])
    d["steps"][0] = {**d["steps"][0], "what": "price takes the London session high",
                     "params": {"kinds": ["session"], "which_session": list(sessions)}}
    d.update(kw)
    return d


def signal_rows(signals) -> pd.DataFrame:
    rows = [{**s.model_dump(mode="python"), "targets": repr(s.targets)} for s in signals]
    cols = ["known_at", "id", "direction", "entry_type", "entry_price", "stop", "targets", "expiry_bars", "cancel_at",
            "lots", "first_visit_from", "first_visit_to"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("first_visit_from", "first_visit_to"):
        df[c] = pd.to_datetime(df[c]).astype("int64", errors="ignore") if df[c].notna().any() else -1
    df["known_at"] = pd.to_datetime(df["known_at"])
    return df


# ======================================================================================================== the form

def test_a_good_form_loads_and_round_trips():
    idea = Idea.load(type1())
    again = Idea.load(idea.to_json())
    assert again == idea
    assert [s.id for s in idea.steps] == ["sweep", "brk", "gap"] and idea.status == "FORM DRAFTED"
    assert idea.steps[0].max_wait_bars is None and idea.steps[1].max_wait_bars == 30


def test_a_form_loads_from_a_json_file(tmp_path):
    p = tmp_path / "form.json"
    p.write_text(json.dumps(fib()), encoding="utf-8")
    assert Idea.load(p).id == "idea-901" and Idea.load(str(p)).id == "idea-901"


def _without_wait(d):
    del d["steps"][1]["max_wait_bars"]
    return d


@pytest.mark.parametrize("make, words", [
    (_without_wait, ["steps[1] 'brk'", "say its max_wait_bars", "\"no limit\""]),
    (lambda d: d["steps"][0].update(max_wait_bars=5) or d, ["the first step STARTS the idea"]),
    (lambda d: d.update(timeframe="2h") or d, ["timeframe", "not one of the allowed values"]),
    (lambda d: d.update(slices=[dict(name=f"s{k}", dimension="session", how="by session") for k in range(9)]) or d,
     ["at most 8 slices", "section D"]),
    (lambda d: d.update(slices=[dict(name="tf", dimension="timeframe", how="5m vs 1h")]) or d,
     ["is not a slice", "a VERSION is a different rule"]),
    (lambda d: d["steps"][2].update(id="brk") or d, ["share the id 'brk'"]),
    (lambda d: d.update(entry=dict(how="limit", max_wait_bars=5)) or d, ["needs 'at'"]),
    (lambda d: d.update(entry=dict(how="limit", at="gap.near")) or d, ["must say its max_wait_bars", "#68"]),
    (lambda d: d.update(entry=dict(how="limit", at="nope.near", max_wait_bars=5)) or d,
     ["entry.at", "names no step", "sweep, brk, gap"]),
    (lambda d: d.update(entry_edge="far") or d, ["entry_edge", "is not a field of the form here"]),
    (lambda d: d.update(directions=[]) or d, ["give 'buy', 'sell' or both"]),
    (lambda d: d.update(unclear=[dict(id="F1", kind="ask", part="0.68 or 0.618",
                                      readings=[dict(id="A", means="0.618")])]) or d, ["give 2-4 readings"]),
    (lambda d: d.update(id="idea-9") or d, ["not an idea id"]),
])
def test_bad_forms_give_readable_errors(make, words):
    with pytest.raises(FormError) as err:
        Idea.load(make(type1()))
    text = str(err.value)
    assert text.startswith("The idea form has problems:")
    assert "https://" not in text and "pydantic" not in text
    for w in words:
        assert w in text, (w, text)


def test_changes_go_by_dotted_paths():
    idea = Idea.load(type1())
    got = apply_changes(idea, {"steps.brk.max_wait_bars": 12, "steps.gap.params.move": "brk",
                               "settings.order_block_max_age_bars": 60, "targets.0.pick": "his_order"})
    assert got.steps[1].max_wait_bars == 12 and got.steps[2].params == {"move": "brk"}
    assert got.settings == {"order_block_max_age_bars": 60} and got.targets[0].pick == "his_order"
    with pytest.raises(FormError, match="names no step; the steps are sweep, brk, gap"):
        apply_changes(idea, {"steps.gapp.tf": "1h"})
    with pytest.raises(FormError, match="not a place in a list of 1"):
        apply_changes(idea, {"targets.3.ratio": 1.0})
    with pytest.raises(FormError, match="makes a bad form"):
        apply_changes(idea, {"steps.brk.max_wait_bars": None})


def _with_unclear():
    return type1(unclear=[
        dict(id="F3", kind="ask", part="which break counts", words="it should break like it should higher high",
             readings=[dict(id="A", means="his documented break (sweep -> order-block break)"),
                       dict(id="B", means="the break within 12 bars", changes={"steps.brk.max_wait_bars": 12})]),
        dict(id="W", kind="version", part="how long the order waits",
             readings=[dict(id="1", means="no limit"),
                       dict(id="2", means="20 bars", changes={"entry.max_wait_bars": 20})]),
    ])


def test_an_open_part_is_never_guessed(market):
    idea = Idea.load(_with_unclear())
    with pytest.raises(FormError, match="never guesses a meaning") as err:
        C.compile_idea(idea, market)
    assert "F3 (ask, which break counts): readings ['A', 'B']" in str(err.value)
    assert len(reading_choices(idea)) == 4
    f, used = with_readings(idea, {"F3": "B", "W": "2"})
    assert (f.steps[1].max_wait_bars, f.entry.max_wait_bars, used) == (12, 20, {"F3": "B", "W": "2"})
    answered = Idea.load({**_with_unclear(), "unclear": [{**_with_unclear()["unclear"][0], "answer": "A"},
                                                         _with_unclear()["unclear"][1]]})
    assert reading_choices(answered) == [{"W": "1"}, {"W": "2"}]
    with pytest.raises(FormError, match="Tejas answered 'A'"):
        with_readings(answered, {"F3": "B", "W": "1"})
    with pytest.raises(FormError, match="not unclear parts"):
        with_readings(idea, {"F9": "A"})


def test_the_version_id_is_the_rule(market):
    a = C.compile_idea(Idea.load(type1()), market)
    words = C.compile_idea(Idea.load(type1(words="other words", title="x", sources=[], notes="n",
                                           slices=[dict(name="s", dimension="session", how="by session")])), market)
    wait = C.compile_idea(Idea.load(type1()), market, changes={"steps.brk.max_wait_bars": 29})
    assert a.version_id == words.version_id                 # the same rule = the same version = one try
    assert a.version_id != wait.version_id and a.version_id.startswith("v-")
    assert all(s.version_id == a.version_id and s.idea_id == "idea-900" for s in a.signals)


# ===================================================================================================== the compile

def test_the_compile_is_deterministic(m1, market):
    one = C.compile_idea(Idea.load(type1()), market)
    two = C.compile_idea(Idea.load(type1()), market)
    fresh = C.compile_idea(Idea.load(type1()), C.Market.from_m1(m1))
    dump = lambda c: [s.model_dump() for s in c.signals]   # noqa: E731
    assert one.signals and dump(one) == dump(two) == dump(fresh)
    assert one.funnel == fresh.funnel and one.dropped == fresh.dropped
    assert len({s.id for s in one.signals}) == len(one.signals)


def type1_68(**kw) -> dict:
    """Type 1 with G8 and his #68 check (the order fills only on price's first visit back into the setup's gaps)."""
    return type1(do_not_enter=[dict(name="target_reached_first"),
                               dict(name="price_comes_back_into_another_gap_first")], **kw)


@pytest.mark.parametrize("make", [type1, fib, comes_back, pair, london, type1_68],
                         ids=["type1", "fib", "comes_back", "pair", "london", "type1_68"])
def test_compiled_signals_do_not_peek(m1, make):
    """Cut the 1-minute bars off at a moment: every signal known by then must be exactly the same (R4, lesson B1)."""
    idea = Idea.load(make())

    def signals_of(bars):
        return signal_rows(C.compile_idea(idea, C.Market.from_m1(bars)).signals)

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2 + 7, (3 * n) // 4 + 131, n - 1441])


def test_the_68_check_stamps_each_order_with_its_own_grab_and_gap(market):
    """price_comes_back_into_another_gap_first: every order carries first_visit_from = the open of its grab's candle
    and first_visit_to = the close of its gap's 3rd candle or of the candle after the break, whichever is later (the
    old engine's arming candle); the rest of the order is the same as without the check. Only for a limit at a gap."""
    with_68 = C.compile_idea(Idea.load(type1_68()), market, trace=True)
    plain = C.compile_idea(Idea.load(type1(do_not_enter=[dict(name="target_reached_first")])), market)
    assert with_68.signals and len(with_68.signals) == len(plain.signals)
    five = pd.Timedelta(minutes=5)
    facts = {r["signal"]: r["facts"] for r in with_68.trace if r.get("signal")}
    for s, p in zip(with_68.signals, plain.signals):
        f = facts[s.id]
        assert pd.Timestamp(s.first_visit_from) == pd.Timestamp(f["sweep"].get("gap_from", f["sweep"]["bar"]))
        assert pd.Timestamp(s.first_visit_to) == max(pd.Timestamp(f["gap"]["bar"]), f["brk"]["bar"] + five) + five
        assert s.model_dump(exclude={"first_visit_from", "first_visit_to", "id", "version_id"}) == \
            p.model_dump(exclude={"id", "version_id"})
    bad = type1_68(entry=dict(how="limit", at="brk.extreme", max_wait_bars="no limit"))
    with pytest.raises(C.CompileError, match="only for a limit at a gap_of_the_move step"):
        C.compile_idea(Idea.load(bad), market)


def test_the_no_peek_test_catches_a_signal_stamped_early(m1, market):
    """The same test on signals that claim to be known one 5-minute candle too soon: cut 2 minutes before real
    signals, the stamped-early ones appear only with the future - caught."""
    idea = Idea.load(type1())

    def early(bars):
        df = signal_rows(C.compile_idea(idea, C.Market.from_m1(bars)).signals)
        df["known_at"] = df["known_at"] - pd.Timedelta(minutes=5)     # claims to know it one candle too soon
        return df

    real = C.compile_idea(idea, market).signals
    cuts = [int(m1.index.searchsorted(pd.Timestamp(s.known_at) - pd.Timedelta(minutes=3))) + 1 for s in real[10:15]]
    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future_events(early, m1, 1, cuts)


def test_the_max_wait_is_honoured(market):
    """A plain loop re-counts, for every break, how many 5-minute bars price took to come back to the broken level,
    and so which breaks pass step 'back' at each max wait. The compiler must agree break by break."""
    bars = market.bars("5m")
    h, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    idx = bars.index
    for wait in (1, 3, 12):
        out = C.compile_idea(Idea.load(comes_back(wait)), market, trace=True)
        for rec in out.trace:
            brk = next(r for r in C.BLOCKS["break_of_structure"].start(
                C._Run(market, out.form, rec["direction"], settings.fingerprint()), out.form.steps[0], None)
                if r["known_at"] == rec["start"])
            j0 = idx.searchsorted(rec["start"])                       # the first bar after the break is known
            came = np.flatnonzero((h[j0:] >= brk["level"]) if rec["direction"] == "sell" else (lo[j0:] <= brk["level"]))
            waited = int(came[0]) + 1 if len(came) else None           # bars t+1 ... t+N
            passed = "back" in rec["steps"]
            assert passed == (waited is not None and waited <= wait), (wait, rec, waited)
            if passed:
                assert rec["steps"]["back"] == idx[j0 + waited - 1] + pd.Timedelta(minutes=5)
        assert out.funnel["back"] and sum(out.dropped.values()) > 0


def test_the_max_wait_boundary_exactly(market):
    """One break that came back after exactly N bars: kept with max wait N, dropped with N - 1."""
    out = C.compile_idea(Idea.load(comes_back(40)), market, trace=True)
    idx = market.bars("5m").index
    rec = next(r for r in out.trace if "back" in r["steps"]
               and idx.searchsorted(r["steps"]["back"]) - idx.searchsorted(r["start"]) >= 3)
    n = int(idx.searchsorted(rec["steps"]["back"]) - idx.searchsorted(rec["start"]))
    for wait, kept in ((n, True), (n - 1, False)):
        t = C.compile_idea(Idea.load(comes_back(wait)), market, trace=True).trace
        same = next(r for r in t if r["start"] == rec["start"] and r["direction"] == rec["direction"])
        assert ("back" in same["steps"]) is kept, (wait, same)
        if not kept:
            assert same["dropped_at"] == "back" and "max wait" in same["why"]


def test_the_entry_wait_is_the_orders_expiry(m1, market):
    """The limit waits max_wait_bars bars of the idea's timeframe: the tester's expiry_bars, which it honours
    (tests/test_tester.py). "no limit" = the order waits to the end of the test."""
    f = C.compile_idea(Idea.load(fib()), market)
    assert f.signals and {s.expiry_bars for s in f.signals} == {20}
    t = C.compile_idea(Idea.load(type1()), market)
    assert {s.expiry_bars for s in t.signals} == {C.NO_LIMIT_BARS}
    rep = replay(f.signals, m1, "1h")
    assert len(rep.trades) + len(rep.not_filled) + len(rep.skipped) == len(f.signals)
    assert all(why in ("expired", "cancelled", "window ended") for _, why in rep.not_filled)


# ---- his fixed rules ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change, words", [
    ({"settings.entry_edge": "far"}, ["LEARNED #65", "I don't wait till that deep of fvg"]),
    ({"steps.gap.params.GAP_AFTER_BREAK_MAX": 10}, ["must belong to the breaking move", "#62"]),
    ({"entry.at": "gap.far"}, ["LEARNED #65", "I'll enter at the beginning of fvg", "Use 'gap.near'"]),
    ({"exit.management.REGRAB": "stop"}, ["a_grab_that_keeps_going_is_still_a_grab", "D5 A"]),
    ({"settings.fill_rule": "any_visit"}, ["LEARNED #68", "it's not about the time"]),
])
def test_his_fixed_rules_are_refused_with_his_words(market, change, words):
    idea = apply_changes(Idea.load(type1()), change) if "REGRAB" not in str(change) else \
        Idea.load(type1(exit=dict(kind="hedge", at="brk.extreme", management={"REGRAB": "stop"})))
    with pytest.raises(rules.FixedRuleError) as err:
        C.compile_idea(idea, market)
    for w in words:
        assert w in str(err.value), (w, str(err.value))
    assert "it is FIXED" in str(err.value) and "LEARNED #121" in str(err.value)


def test_a_reading_cannot_change_a_fixed_rule(market):
    idea = Idea.load(type1(unclear=[dict(id="E", kind="version", part="where in the gap", readings=[
        dict(id="A", means="near"), dict(id="B", means="far", changes={"settings.where_in_the_gap_he_buys": "far"})])]))
    assert C.compile_idea(idea, market, readings={"E": "A"}).signals
    with pytest.raises(rules.FixedRuleError, match="I don't wait till that deep of fvg"):
        C.compile_idea(idea, market, readings={"E": "B"})


def test_changing_his_setting_is_allowed_only_as_a_labelled_comparison(market):
    out = C.compile_idea(Idea.load(type1(settings={"which_gap_of_the_move": "deepest"})), market)
    assert out.resolved.breaks_his_rules and "NOT HIS RULE (a comparison only): which_gap_of_the_move" in out.text()


# ---- what is missing, and other readable refusals -----------------------------------------------------------------

@pytest.mark.parametrize("make, name, prompt", [
    (lambda: type1(steps=type1()["steps"] + [dict(id="tl", what="a trendline retest", block="trendline_retest",
                                                  tf="5m", max_wait_bars=10)]), "trendline_retest", "L6-new"),
    (lambda: type1(steps=[dict(id="tl", what="a trendline touch", block="trendline_touch", tf="5m")],
                   entry=dict(how="limit", at="tl.entry", max_wait_bars=10), exit=dict(kind="hedge", at="tl.hedge"),
                   do_not_enter=[]), "steep", "L9b"),
    (lambda: type1(filters=[dict(name="news", params={"only": "red"})]), "news", "L12"),
    (lambda: type1(steps=type1()["steps"] + [dict(id="hw", what="halfway to the target, then it turned",
                                                  block="halfway", tf="5m", max_wait_bars=10)]),
     "halfway", "L19c"),                                    # zones: built in L6b.2; one_minute: built in L6b.4
])                                                           # the second gap (#124): built in L19b
def test_a_missing_block_is_named(market, make, name, prompt):
    with pytest.raises(C.MissingBlockError) as err:
        C.compile_idea(Idea.load(make()), market)
    assert err.value.missing[0][0] == name and prompt in str(err.value) and "idea-900 needs" in str(err.value)


@pytest.mark.parametrize("make, words", [
    (lambda: type1(steps=type1()["steps"][:2] + [type1()["steps"][2] | {"params": {"move": "sweep"}}]),
     ["gap_of_the_move", "params.move = 'sweep' must be a 'break_of_structure' or 'wedge' step"]),
    (lambda: type1(steps=[type1()["steps"][1] | {"max_wait_bars": None}, type1()["steps"][2]],
                   do_not_enter=[]),
     ["needs a 'sweep' or 'false_breakout' or 'wedge' step before it", "#61, #62, #131"]),
    (lambda: comes_back(0), ["WAITS", "1 or more"]),
    (lambda: type1(steps=type1()["steps"] + [dict(id="tgt", what="a level below", block="level_waiting", tf="5m",
                                                  max_wait_bars=5)]), ["a CHECK", "must be 0"]),
    (lambda: type1(steps=type1()["steps"] + [dict(id="f", what="fib", block="fib_move", tf="5m", max_wait_bars=5)]),
     ["can only START an idea"]),
    (lambda: comes_back(steps=[comes_back()["steps"][0], {**comes_back()["steps"][1], "params": {"at": "back.price"}}]),
     ["params.at", "comes later or does not exist"]),
    (lambda: type1(exit=dict(kind="stop", at="gap.middle")), ["does not find 'middle'", "near, far"]),
    (lambda: type1(steps=[{**type1()["steps"][0], "params": {"level": 2}}] + type1()["steps"][1:]),
     ["'sweep' (sweep) params", "level", "is not a field"]),
    (lambda: fib(timeframe="4h", steps=[{**fib()["steps"][0], "tf": "4h"}]), ["needs 4h bars", "5m, 1h, 1d, 1w"]),
    (lambda: comes_back(do_not_enter=[dict(name="target_reached_first")]), ["only for a limit entry"]),
    (lambda: comes_back(targets=[dict(kind="ratio", ratio=1.0)]), ["a ratio target needs the entry price"]),
    (lambda: type1(filters=[dict(name="session", params={"only": ["Tokyo"]})]), ["Tokyo", "Asia, London, New York"]),
    (lambda: london(sessions=["Tokyo"]), ["'sweep' (sweep) params", "Tokyo", "his sessions are Asia, London, New York"]),
    (lambda: london(steps=[{**london()["steps"][0], "params": {"kinds": ["equal"], "which_session": ["London"]}}]
                    + london()["steps"][1:]), ["which_session names session highs / lows only", "kinds must be ['session']"]),
])
def test_meaning_errors_are_readable(market, make, words):
    with pytest.raises(C.CompileError) as err:
        C.compile_idea(Idea.load(make()), market)
    for w in words:
        assert w in str(err.value), (w, str(err.value))


# ---- the rules and filters do what they say ------------------------------------------------------------------------

def test_the_pair_blocks_link_up(market):
    """equal_levels -> sweep OF that level -> level_waiting on the other side: the sweep is of the very level the pair
    step found, and the target is an equal level on the other side, still untaken at the sweep (liquidity.alive)."""
    out = C.compile_idea(Idea.load(pair()), market, trace=True)
    L = market.liquidity(settings.fingerprint())
    f = out.funnel
    assert f["pair"]["sell"] >= f["sweep"]["sell"] >= f["other"]["sell"] > 0 and out.signals
    sw = L.sweeps.set_index("level_id")
    life = L.life.set_index("level_id")
    for s in out.signals:
        rec = next(r for r in out.trace if r["signal"] == s.id)
        assert rec["steps"]["sweep"] in set(sw["known_at"])
        m5 = market.bars("5m").index
        i = m5.searchsorted(rec["steps"]["sweep"]) - 1                      # the sweep candle
        tgt = life[(life["price"].round(2) == s.targets[0]) & life["kind"].str.startswith("equal_")]
        assert ((tgt["start"] <= i) & (i < tgt["end"])).any(), s.id        # alive through the sweep candle
    assert out.dropped.get(("other", "no level waiting on the target side"), 0) > 0


def test_the_grab_that_keeps_going_moves_the_hedge(market):
    """His fixed rule a_grab_that_keeps_going_is_still_a_grab (strategy-understanding 66a; steps.md step 4): after a
    small break, a candle past the grab's extreme before the gap is known sends the idea back to wait for the break
    again, and the hedge point moves with it. A plain re-check on the 5-minute bars of every signal."""
    out = C.compile_idea(Idea.load(type1()), market, trace=True)
    assert out.regrabs > 0 and "kept going" in out.text()
    bars = market.bars("5m")
    h, lo, idx = bars["high"].to_numpy(), bars["low"].to_numpy(), bars.index
    by_id = {s.id: s for s in out.signals}
    moved = 0
    for rec in out.trace:
        if rec["signal"] is None:
            continue
        s = by_id[rec["signal"]]
        sell = s.direction == "sell"
        i0 = idx.searchsorted(rec["steps"]["sweep"]) - 1                  # the sweep candle
        b = idx.searchsorted(rec["steps"]["brk"]) - 1                     # the (last) break candle
        g = idx.searchsorted(s.known_at) - 1                              # the candle that made the gap known
        ext = h[i0:b + 1].max() if sell else lo[i0:b + 1].min()
        assert s.stop == pytest.approx(round(ext, 2)), s.id              # the hedge point: the grab's extreme
        after = h[b + 1:g + 1] if sell else lo[b + 1:g + 1]
        assert not ((after > ext).any() if sell else (after < ext).any()), s.id   # nothing past it before the gap
        for at in rec["regrab_at"]:
            k = idx.searchsorted(at) - 1                                  # the candle that went past
            assert i0 < k < b
            assert (h[k] > h[i0:k].max()) if sell else (lo[k] < lo[i0:k].min()), (s.id, at)
        moved += bool(rec["regrab_at"])
    assert moved >= 5


def test_the_grab_ran_too_far_rule(market):
    """His T3 no-reasons ('it moves so far'): from the level taken to the grab's extreme more than the setting's share
    of a day's range = no trade. A plain re-check of every signal the rule keeps or drops."""
    rules_ = type1()["do_not_enter"]
    off = C.compile_idea(Idea.load(type1()), market, trace=True)
    on = C.compile_idea(Idea.load(type1(do_not_enter=rules_ + [dict(name="grab_ran_too_far")],
                                        settings={"grab_ran_too_far_adr": 0.1})), market)
    assert 0 < len(on.signals) < len(off.signals)
    assert any("it moves so far" in why for (_, why) in on.dropped)
    ruler, closes = market.ruler("5m", settings.fingerprint()), market.closes("5m")
    kept = {(s.direction, pd.Timestamp(s.known_at), s.entry_price) for s in on.signals}
    ok = {}
    for rec in off.trace:     # every start that reached an entry: a grab that took two levels gives the same entry
        if rec["signal"] is None and rec["why"] != "the same entry reached from an earlier start":
            continue
        t = pd.Timestamp(rec["steps"]["gap"])
        level, ext = rec["facts"]["sweep"]["price"], rec["facts"]["brk"]["extreme"]
        ran = ext - level if rec["direction"] == "sell" else level - ext
        r = ruler[int(np.searchsorted(closes, t.as_unit("ns").value, "right")) - 1]
        key = (rec["direction"], t, C._round(rec["facts"]["gap"]["near"]))
        ok[key] = ok.get(key, True) and bool(np.isfinite(r) and ran <= 0.1 * r)   # no day's range yet: dropped
    assert kept == {k for k, v in ok.items() if v}                  # the run counts from the FIRST level taken
    assert sum(not v for v in ok.values()) > 0 and any("earlier level" in why for (_, why) in on.dropped)
    no_line = C.compile_idea(Idea.load(type1(do_not_enter=rules_ + [dict(name="grab_ran_too_far")],
                                             settings={"grab_ran_too_far_adr": None})), market)
    assert [x.known_at for x in no_line.signals] == [x.known_at for x in off.signals]      # None = no line


@pytest.mark.parametrize("changes", [{"which_gap_of_the_move": "deepest"}, {}], ids=["deepest", "first+regrab"])
def test_no_peek_right_after_every_signal(m1, changes):
    """The hardest cut: one minute after each signal is known. which_gap_of_the_move = 'deepest' (Claude's, a version)
    can choose the deepest gap only when the gap's window has ended - L7b found it known too early (this test caught
    the L7a code at 13 Jan 2019 19:00); the grab that keeps going goes back to the break from the new extreme only."""
    idea = Idea.load(type1(settings=changes))

    def signals_of(bars):
        return signal_rows(C.compile_idea(idea, C.Market.from_m1(bars)).signals)

    sig = signals_of(m1)
    cuts = sorted(c for c in {int(m1.index.searchsorted(t)) + 1 for t in sig["known_at"]} if c < len(m1))
    assert len(cuts) > 30
    guards.assert_no_future_events(signals_of, m1, 1, cuts[::max(1, len(cuts) // 40)])


def test_the_stop_too_near_rule(market):
    near = C.compile_idea(Idea.load(type1(do_not_enter=[dict(name="stop_too_near")],
                                          settings={"min_hedge_distance_adr": 0.1})), market)
    everything = C.compile_idea(Idea.load(type1(do_not_enter=[])), market)
    assert 0 < len(near.signals) < len(everything.signals)
    ruler = market.ruler("5m", settings.fingerprint(near.resolved))
    closes = market.closes("5m")
    for s in near.signals:
        i = int(np.searchsorted(closes, pd.Timestamp(s.known_at).as_unit("ns").value, "right")) - 1
        assert abs(s.entry_price - s.stop) >= 0.1 * ruler[i] - 0.01


def test_the_target_reached_first_rule_goes_to_the_tester(market):
    out = C.compile_idea(Idea.load(type1()), market)
    assert out.signals and all(s.cancel_at == s.targets[0] for s in out.signals)
    none = C.compile_idea(Idea.load(type1(do_not_enter=[])), market)
    assert all(s.cancel_at is None for s in none.signals)


def test_a_finished_move_is_not_entered(market):
    """target_already_reached: between the sweep and the signal price never reached the target (a plain re-check)."""
    out = C.compile_idea(Idea.load(type1()), market, trace=True)
    bars = market.bars("5m")
    by_id = {s.id: s for s in out.signals}
    checked = 0
    for rec in out.trace:
        if rec["signal"] is None:
            continue
        s = by_id[rec["signal"]]
        seg = bars[(bars.index >= rec["steps"]["sweep"]) & (bars.index + pd.Timedelta(minutes=5) <= s.known_at)]
        tgt = s.targets[0]
        assert not ((seg["low"] <= tgt).any() if s.direction == "sell" else (seg["high"] >= tgt).any())
        checked += 1
    assert checked == len(out.signals) > 20


def test_the_entry_already_reached_rule(market):
    """A swing is known 2 hour-candles after its high: if price already went back to 0.618 in those 2 hours, the first
    visit is gone (#68) - dropped. A plain re-check on the 5-minute bars of those 2 hours."""
    with_rule = C.compile_idea(Idea.load(fib()), market)
    without = C.compile_idea(Idea.load(fib(do_not_enter=[])), market)
    assert 0 < len(with_rule.signals) < len(without.signals)
    b = market.bars("5m")
    kept = {(s.direction, s.known_at) for s in with_rule.signals}         # ids differ: two rules, two versions
    for s in without.signals:
        t = pd.Timestamp(s.known_at)
        seg = b[(b.index >= t - pd.Timedelta(hours=2)) & (b.index + pd.Timedelta(minutes=5) <= t)]
        reached = (seg["low"] <= s.entry_price).any() if s.direction == "buy" else (seg["high"] >= s.entry_price).any()
        assert ((s.direction, s.known_at) in kept) is (not reached), s.id


def test_the_session_filter_keeps_only_its_session(market):
    out = C.compile_idea(Idea.load(type1(filters=[dict(name="session", params={"only": ["London"]})])), market)
    everything = C.compile_idea(Idea.load(type1()), market)
    assert 0 < len(out.signals) < len(everything.signals)
    for s in out.signals:
        ist = to_ist(pd.Timestamp(s.known_at))
        assert 12 * 60 + 30 < ist.hour * 60 + ist.minute <= 15 * 60 + 30, ist


def test_the_london_high_only(market):
    """L6-new, idea #3 ("sell when the London high is taken"): with params.which_session the sweep step starts only
    from a fresh take of a LONDON session high - never the Asia or New York high, never another kind. The filter
    'session' is another thing (the clock at the setup's moment)."""
    out = C.compile_idea(Idea.load(london()), market, trace=True)
    every = C.compile_idea(Idea.load(london(steps=[{**london()["steps"][0], "params": {"kinds": ["session"]}}]
                                            + london()["steps"][1:])), market)
    sw = market.liquidity(settings.fingerprint()).sweeps
    lon = LQ.of_session(sw, "London")
    lon = lon[(lon["side"] == "high") & lon["fresh"] & lon["can_start"]]
    assert set(lon["kind"]) == {"session_high"} and set(lon["session"]) == {"London"}
    assert 0 < out.funnel["sweep"]["sell"] == len(lon) < every.funnel["sweep"]["sell"]
    assert 0 < len(out.signals) < len(every.signals)
    moments = set(lon["known_at"])
    for s in out.signals:
        assert s.direction == "sell"
        rec = next(r for r in out.trace if r["signal"] == s.id)
        assert rec["steps"]["sweep"] in moments, s.id          # its grab is a London high taken
    # he may say the names his way; a buy reads the session LOW
    said = C.SweepParams.model_validate({"which_session": ["london", "NY", "London"]})
    assert said.which_session == ("London", "New York")
    buys = C.compile_idea(Idea.load(london(directions=["buy"])), market)
    lows = LQ.of_session(sw, "London")
    assert buys.funnel["sweep"]["buy"] == int(((lows["side"] == "low") & lows["fresh"] & lows["can_start"]).sum()) > 0
    # the param is not called `sessions`: that is the old name of his FIXED rule (the windows), refused with his words
    d = london()
    d["steps"][0] = {**d["steps"][0], "params": {"sessions": ["London"]}}
    with pytest.raises(rules.FixedRuleError, match="what_a_session_high_or_low_is"):
        C.compile_idea(Idea.load(d), market)


def test_a_session_level_waiting(market):
    """level_waiting with params.which_session: "the Asian low waits below" - the target is an Asia session low."""
    d = london()
    d["steps"] = [d["steps"][0], dict(id="other", what="the Asian low waits below, not taken", block="level_waiting",
                                      tf="5m", max_wait_bars=0, params={"which_session": ["Asian"]})] + d["steps"][1:]
    d["targets"] = [dict(kind="price", at="other.price")]
    out = C.compile_idea(Idea.load(d), market)
    life = market.liquidity(settings.fingerprint()).life
    asia = set(life[(life["session"] == "Asia") & (life["side"] == "low")]["price"].round(2))
    assert out.signals and all(s.targets[0] in asia for s in out.signals)
    assert out.funnel["other"]["sell"] < out.funnel["sweep"]["sell"]      # not every time an Asian low waits


def test_a_versions_settings_reach_the_blocks(market):
    base = C.compile_idea(Idea.load(type1()), market)
    short = C.compile_idea(Idea.load(type1(settings={"order_block_max_age_bars": 2})), market)
    assert short.funnel["brk"] != base.funnel["brk"] and short.resolved.changed == ("order_block_max_age_bars",)
    assert settings.get("order_block_max_age_bars") == 40                   # today's value again after the compile
    assert short.version_id != base.version_id


def test_the_hedge_is_tested_as_a_stop_and_says_so(market):
    out = C.compile_idea(Idea.load(type1()), market)
    assert out.management == "hedge" and "tests it as a plain stop" in out.text()
    assert all(s.stop is not None for s in out.signals)
    stop = C.compile_idea(Idea.load(type1(exit=dict(kind="stop", at="brk.extreme", beyond_usd=0.5))), market)
    assert stop.management == "stop"
    one = {s.known_at: s for s in out.signals}
    for s in stop.signals:
        if s.known_at in one and s.direction == one[s.known_at].direction:
            assert s.stop == pytest.approx(one[s.known_at].stop + (0.5 if s.direction == "sell" else -0.5))


def test_the_compiled_signals_run_through_the_tester(m1, market):
    out = C.compile_idea(Idea.load(type1()), market)
    res = run(out.signals, tf=out.tf, m1=m1, record=False)
    assert res.summary["signals"] == len(out.signals)
    assert {t.version_id for t in res.trades} <= {out.version_id}
    assert "cancelled because price reached the target first" in res.text()


def test_the_text_says_no_money_and_counts_every_step(market):
    out = C.compile_idea(Idea.load(type1()), market)
    text = out.text()
    assert "no money here" in text and "₹" not in text and "Rs" not in text
    for s in ("sweep", "brk", "gap"):
        assert f" {s} " in text
    assert out.counts()["all"] == len(out.signals) == len(out.table())


def test_every_block_is_made_of_the_blocks():
    """Blocks only: every step block names the machine/blocks functions it is made of, and they exist; every rule and
    filter says what it means; describe() (what the talker, L8, reads) lists them all."""
    import importlib
    for x in C.BLOCKS.values():
        assert x.means and x.uses, x.name
        for u in x.uses:
            mod, fn = u.split(".", 1)
            assert hasattr(importlib.import_module(f"machine.blocks.{mod}"), fn), u
    for x in (*C.DO_NOT_ENTER.values(), *C.FILTERS.values()):
        assert x.means and x.uses, x.name
    text = C.describe()
    assert all(name in text for name in (*C.BLOCKS, *C.DO_NOT_ENTER, *C.FILTERS))


def test_settings_using_restores_todays_values():
    with settings.using({"order_block_max_age_bars": 7}) as r:
        assert settings.get("order_block_max_age_bars") == 7 and r.changed == ("order_block_max_age_bars",)
        fp_inside = settings.fingerprint()
    assert settings.get("order_block_max_age_bars") == 40 and settings.fingerprint() != fp_inside
    with pytest.raises(RuntimeError):
        with settings.using({"order_block_max_age_bars": 9}):
            raise RuntimeError("an error inside")
    assert settings.get("order_block_max_age_bars") == 40
    with pytest.raises(rules.FixedRuleError):
        with settings.using({"entry_edge": "far"}):
            pass


# ========================================================================================================= the store

@pytest.fixture
def ideas_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IDEAS_DIR", tmp_path / "work" / "ideas")
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    (tmp_path / "work" / "idea-001").mkdir(parents=True)          # the T0 / T3 folder
    return tmp_path / "work" / "ideas"


def test_the_store_makes_the_ideas_folder(ideas_dir):
    assert store.next_id() == "idea-002"                           # idea-001 is taken by plan/work/idea-001/
    idea = Idea.load(type1(id=store.next_id()))
    d = store.create(idea)
    assert sorted(p.name for p in d.iterdir()) == ["answers.md", "form.json", "questions", "state.json"]
    assert store.load_form("idea-002") == idea and store.ideas() == ["idea-002"]
    assert store.read_state("idea-002")["state"] == "FORM DRAFTED"
    assert store.next_id() == "idea-003"
    assert store.report_path("idea-002").name == "report.html" and store.questions_dir("idea-002").is_dir()
    with pytest.raises(store.StoreError, match="already exists"):
        store.create(idea)


def test_states_are_kept_with_their_history(ideas_dir):
    store.create(Idea.load(type1(id="idea-002")))
    store.set_state("idea-002", "QUESTIONS QUEUED", "F3 queued", waits_for="Tejas's answer to F3")
    s = store.set_state("idea-002", "ANSWERED", "F3 = A")
    assert [h["state"] for h in s["history"]] == ["FORM DRAFTED", "QUESTIONS QUEUED", "ANSWERED"]
    assert store.load_form("idea-002").status == "ANSWERED"
    with pytest.raises(store.StoreError, match="not a state"):
        store.set_state("idea-002", "DONE")


def test_a_locked_idea_cannot_change_its_meaning(ideas_dir):
    idea = Idea.load(type1(id="idea-002"))
    store.create(idea)
    store.save_form(idea.model_copy(update={"notes": "fine before the lock"}))
    changed = apply_changes(idea, {"steps.brk.max_wait_bars": 12})
    store.save_form(changed)                                       # before the lock: allowed
    store.set_state("idea-002", "CONFIRMED & LOCKED", "his yes")
    locked = store.load_form("idea-002")
    store.save_form(locked.model_copy(update={"notes": "a note is not the meaning"}))
    with pytest.raises(store.StoreError, match="new idea"):
        store.save_form(apply_changes(locked, {"steps.brk.max_wait_bars": 30}))
    with pytest.raises(store.StoreError, match="new idea"):
        store.save_form(locked.model_copy(update={"slices": (
            Idea.load(type1(slices=[dict(name="s", dimension="session", how="x")])).slices)}))
    with pytest.raises(store.StoreError, match="cannot go back"):
        store.set_state("idea-002", "FORM DRAFTED")


def test_answers_are_appended_never_rewritten(ideas_dir):
    store.create(Idea.load(type1(id="idea-002")))
    p = store.add_answer("idea-002", "A - my own break", about="F3", source="strategy/learned-inbox/x.md")
    first = p.read_text(encoding="utf-8")
    store.add_answer("idea-002", "yes\nthat one", about="SHOWN", source="telegram 123")
    text = p.read_text(encoding="utf-8")
    assert text.startswith(first) and "> yes\n> that one" in text and "source: telegram 123" in text
    with pytest.raises(store.StoreError, match="needs his words"):
        store.add_answer("idea-002", " ", about="F3", source="x")


@pytest.mark.parametrize("bad", ["../idea-001", "idea-1", "idea-001/..", "IDEA-001", ""])
def test_bad_idea_ids_are_refused(ideas_dir, bad):
    with pytest.raises(store.StoreError, match="not an idea id"):
        store.folder(bad)


# ====================================================================================================== real prices

@pytest.mark.data
def test_real_prices_compile_without_peeking():
    """March 2019, the real 1-minute bars: the Type 1 shape compiles, its signals do not peek, the tester runs them."""
    from machine.data.bars import load_bars
    m1 = load_bars("1m", "2019-03-04", "2019-03-29")
    idea = Idea.load(type1())
    out = C.compile_idea(idea, C.Market.from_m1(m1))
    assert len(out.signals) > 50 and out.funnel["sweep"]["buy"] > 0 and out.funnel["sweep"]["sell"] > 0

    def signals_of(bars):
        return signal_rows(C.compile_idea(idea, C.Market.from_m1(bars)).signals)

    n = len(m1)
    guards.assert_no_future_events(signals_of, m1, 1, [n // 2, (4 * n) // 5])
    res = run(out.signals, tf="5m", m1=m1, record=False)
    assert res.summary["signals"] == len(out.signals)
