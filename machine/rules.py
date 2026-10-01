"""Tejas's FIXED rules: what a thing IS. No test, form, version or research run may change one (LEARNED #121).

His words, 20 Sep 2026 (LEARNED #121, quoted in old/strategy_lab/knobs.py):
    "you differentiate the things in a way that predict which are changeable which are fixed - like triangle
    pattern is fixed right, our fair value gap is fixed ... once I teach it what is a fair value gap and how it
    should be traded, it should never change."

So the machine has two files:
    machine/rules.py     (this file)  WHAT A THING IS. The 19 fixed rules, ported from old/strategy_lab/knobs.py FIXED,
                         plus PARTS: the numbers inside his definitions that are his (or read off his own marked charts).
    machine/settings.py  WHAT TO DO ABOUT IT. The 31 changeable settings (knobs.py CHANGEABLE) and every number
                         Claude chose. Only these may differ between versions (plan/likhith.md section D).

Every rule carries: its value · what it means in plain words · the lesson that taught it · HIS WORDS, copied word for
word (tests/test_rules.py checks each quote against strategy/LEARNED.md and strategy/strategy.md) · where the old code
had it · who chose the NUMBER (his, read off his marked charts by Claude, or Claude's) · the names old code and forms
used for it (so a form that says entry_edge="far" is caught, not only one that says where_in_the_gap_he_buys="far").

A change to any of them is refused with his words: refuse_fixed(), which settings.resolve() calls first.
Blocks read their definitions from here and from machine/settings.py only (likhith.md R11).
Where his files and the old code disagree, his newest lesson wins (CLAUDE.md order of trust) and the difference is
listed in plan/work/rules-missing.md.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

LEARNED = "strategy/LEARNED.md"
STRATEGY = "strategy/strategy.md"


@dataclass(frozen=True)
class Rule:
    value: Any
    means: str                      # plain English
    taught: str                     # the lesson(s): "LEARNED #65"
    words: str | None               # HIS words, word for word; several pieces are joined by " … "
    words_from: tuple[str, ...]     # the file(s) the words are in (checked by tests/test_rules.py)
    old_code: str                   # where the old engine had it
    number: str                     # who chose the NUMBER: "his" · "Claude's, from his marked charts" · "Claude's" · "-"
    aliases: tuple[str, ...] = ()   # other names old code / forms used for the same rule
    evidence: str = ""              # when he gave no sentence: what shows it is his (his marks on charts)


class FixedRuleError(ValueError):
    """Something tried to change one of his fixed rules."""


_W = (LEARNED,)
_S = (STRATEGY,)
_WS = (LEARNED, STRATEGY)

# ------------------------------------------------------------------------------------------------ the 19 fixed rules
_FIXED = {
    # ---- the break, the gap, and where he buys -----------------------------------------------------------------
    "what_counts_as_the_break": Rule(
        value="order_block_wick",
        means="The break of structure is a wick through his order block. SELL: the most recent RED candle in the rise, "
              "after which price went above its high (a wick is enough); a wick below its low is the break. BUY: the "
              "mirror - the most recent GREEN candle in the drop; a wick above its high is the break.",
        taught="LEARNED #56, #57, #58, #59, #60 (his arrows on 9 charts)",
        words="the recent order block, that I would consider the low of the recent order block. That what I consider, "
              "it should be broken to consider as a makeup structure, break of structure. … It is the red candle in a "
              "uptrend, so not the green candle.",
        words_from=_W,
        old_code="old/ml/agent/setup_engine.py detect(bos_ref='ob_engulf') + ob_engulf_arrays()",
        number="-",
        aliases=("bos_ref", "bos_mode", "break_rule"),
    ),
    "which_gap_is_the_entry_gap": Rule(
        value="first_after_the_sweep",
        means="The entry gap is the FIRST gap after the liquidity is taken, any size. It may form just before the "
              "break, never before the sweep: the gap's first candle is the sweep candle or later.",
        taught="LEARNED #61, #62 (his arrows), #131",
        words="The entry gap is the FIRST gap after the liquidity is taken — any size; it may form just before the "
              "break, but never before the sweep … it only valid if it forms after taking any liquidity or else it's "
              "not correct.",
        words_from=_WS,
        old_code="old/ml/agent/setup_engine.py detect(gap_from='sweep'): the gap scan starts at sweep_bar + 2",
        number="-",
        aliases=("gap_from", "fvg_scan"),
    ),
    "gap_must_belong_to_the_breaking_move": Rule(
        value=6,
        means="The gap must be made by the move that breaks structure: its third candle closes at most 6 candles "
              "after the break candle. A later gap is a new move in chop, not this one.",
        taught="LEARNED #62 (chart C: gap 7 candles after the break, 'not going to take'), #101 (card G01212, gap "
               "17 candles after the break), #103",
        words="it must belong to the breaking move … there is no farewell gap there I don't know why it is taking "
              "entry there",
        words_from=_WS,
        old_code="old/ml/agent/setup_engine.py GAP_AFTER_BREAK_MAX = 6",
        number="Claude's, from his marked charts: his 12 marked gaps formed 1 candle before to 5 after the break",
        aliases=("GAP_AFTER_BREAK_MAX", "gap_after_break_max", "WEDGE_GAP_BARS"),
    ),
    "where_in_the_gap_he_buys": Rule(
        value="near",
        means="At the NEAR edge of the gap - the edge price meets first on the way back (buy: the top of the gap; "
              "sell: the bottom). He does not wait for the deep side.",
        taught="LEARNED #65",
        words="I don't wait till that deep of fvg . I'll enter at the beginning of fvg as I marked in the attached image",
        words_from=_W,
        old_code="old/ml/agent/setup_engine.py detect(entry_edge='near')",
        number="-",
        aliases=("entry_edge", "edge"),
    ),
    "only_the_first_visit_back_counts": Rule(
        value="first_visit",
        means="Only price's FIRST visit back into the gap is an entry. A gap price already came back into is spent. "
              "Time does not matter: no time limit (his words). A wait limit is the idea's own number (a version), "
              "never part of this rule.",
        taught="LEARNED #68",
        words="it's not about the time uh, it's about whether it is already filled or not … only price's FIRST visit "
              "back into the gap is an entry — time does not matter.",
        words_from=_WS,
        old_code="old/ml/agent/setup_engine.py first_visit() + manager.FILL_RULE = 'first_visit'. knobs.py listed "
                 "FVG_RETURN_BARS = 10 here: that is G8's retired 10-candle window (plan/work/rules-missing.md)",
        number="-",
        aliases=("FVG_RETURN_BARS", "fvg_return_bars", "fill_rule", "FILL_RULE", "entry_time_limit"),
    ),
    "a_grab_that_keeps_going_is_still_a_grab": Rule(
        value="continue",
        means="After a small break, price making a new extreme through the grab does NOT kill the grab: the grab is "
              "still going. Wait for the break again, and move the hedge to the new extreme.",
        taught="strategy-understanding section 66a (his marked charts)",
        words=None,
        words_from=(),
        old_code="old/ml/agent/setup_engine.py REGRAB = 'continue'",
        number="-",
        aliases=("REGRAB", "regrab"),
        evidence="No sentence: on his marked charts D5 A, Q59-A1c B and Q60-A5 C the grab made a new extreme after a "
                 "small break and he still marked it as the sweep (strategy-understanding section 66a).",
    ),
    "what_a_hammer_is": Rule(
        value=MappingProxyType(dict(wick_min=0.55, other_wick_max=0.40, body_max=0.40, dip_adr_min=0.12)),
        means="A long one-sided wick on the side the liquidity was taken - at least 55% of the candle - a small body "
              "(at most 40%), the other wick at most 40%, and the dip at least 0.12 of a day's range. A hammer that "
              "grabs liquidity and leaves a gap is a stronger setup; it may go against the higher timeframe.",
        taught="LEARNED #115 (his R05, 20 Sep 13:00). The plan and the old code say #117: it was BUILT in the #117 "
               "session",
        words="there is a very uh, hammer type of candle which is a signal that it took the liquidity and uh, buy "
              "bought suddenly and the fair value gap is formed that is uh, you know one powerful fair value gap … So "
              "the dip, huge dip, and the fair value gap will give the better trades than normal fair value gaps.",
        words_from=_W,
        old_code="old/ml/agent/setup_engine.py HAMMER_WICK_FRAC = 0.55, HAMMER_BODY_FRAC = 0.40, "
                 "HAMMER_DIP_ADR = 0.12, _is_hammer()",
        number="Claude's (the old engine, 20 Sep) - his words give the shape, no numbers (plan/work/rules-missing.md)",
        aliases=("HAMMER_WICK_FRAC", "HAMMER_BODY_FRAC", "HAMMER_DIP_ADR", "hammer"),
    ),
    # ---- the patterns -------------------------------------------------------------------------------------------
    "what_a_triangle_is": Rule(
        value=0.60,
        means="Two lines that clearly narrow toward each other, with price moving INSIDE them, then the break; after a "
              "big move. The lines must close to 60% of the gap they started with - wider than that looks parallel.",
        taught="LEARNED #116, #118",
        words="Triangle: two lines that clearly narrow toward each other, with price moving INSIDE them, then the "
              "break … the price action should be inside the triangle. Then only uh, it can we can validate.",
        words_from=_WS,
        old_code="old/ml/agent/entries.py TRI_CONVERGE = 0.60",
        number="Claude's, from his marked charts: the loosest number that removes both triangles he rejected (#118)",
        aliases=("TRI_CONVERGE", "tri_converge"),
    ),
    "how_long_a_triangle_must_last": Rule(
        value=24,
        means="At least 24 five-minute candles (2 hours).",
        taught="measured by Claude, strategy-understanding section 66bc",
        words=None,
        words_from=(),
        old_code="old/ml/agent/entries.py TRI_MIN_BARS = 24",
        number="Claude's (plan/work/rules-missing.md)",
        aliases=("TRI_MIN_BARS", "tri_min_bars"),
        evidence="In knobs.py FIXED since 20 Sep 2026. He gave no length; raising it did not remove the shape he "
                 "rejected (entries.py comment).",
    ),
    "what_a_trendline_is": Rule(
        value=MappingProxyType(dict(touches=3, through_max_adr=0.005)),
        means="Three touches, drawn through the wick ends, that respect the line. A touch through the line, even "
              "slightly, means it is not a trendline (at most 0.005 of a day's range, 'a touch AT the line to the "
              "eye'). Any angle can be drawn and is liquidity; the angle only decides the touch ENTRY.",
        taught="LEARNED #92, #94, #112, #116",
        words="you can see that uh, the three touches should be there … if the third touch is go beyond the um, trend "
              "line, even now slightest, it is not considered as a trend line.",
        words_from=_W,
        old_code="old/ml/agent/levels.py LINE_RULE = 'his', LINE_THROUGH_FRAC = 0.005",
        number="3 is his; 0.005 is Claude's, from his marked charts (#92 B 0.0035 yes; #94 0.007 and more no)",
        aliases=("LINE_RULE", "LINE_THROUGH_FRAC", "line_rule", "min_touches"),
    ),
    "two_touches_close_together_are_one_touch": Rule(
        value=36,
        means="Two touches of a line less than 36 five-minute candles (3 hours) apart count as ONE touch.",
        taught="LEARNED #115 (his R02: two of three touches 65 minutes apart), #116",
        words="It only touches twice, and you uh, why you are considering it as a trend line? That's a wrong trend line.",
        words_from=_W,
        old_code="old/ml/agent/entries.py TL_MIN_TOUCH_GAP = 36",
        number="Claude's, from his marked charts (his Q115 letters, #116)",
        aliases=("TL_MIN_TOUCH_GAP", "tl_min_touch_gap"),
    ),
    "what_equal_highs_and_lows_are": Rule(
        value=MappingProxyType(dict(tol_usd=0.06, bounce_adr_min=0.08, nothing_past_the_first=True,
                                    same_on_every_chart=("5m", "1h", "1d"))),
        means="Really equal - one or two pips (2-4 cents yes, 9 cents not good): at most $0.06 apart, in DOLLARS, the "
              "same on every chart (#90: not wider for bigger candles). The later one must not pass the first. "
              "Nothing may trade past the first one in between. Any time apart. A pair: after one side is taken, "
              "price goes to the other.",
        taught="LEARNED #88, #89, #90; #21 (6 Sep) for the pair",
        words="One pip or two pips are okay. If it is not equal, if more than that, then it is not a equal highs or "
              "equal lows. … equal loss should be something that shouldn't drawn yet no price uh, should never reach "
              "their particular two equals",
        words_from=_W,
        old_code="old/ml/agent/levels.py EQ_RULE = 'his', EQ_TOL_FRAC_HIS = 0.003, EQ_BOUNCE_FRAC = 0.08, "
                 "EQ_HTF = ('H1', 'D1')",
        number="'one or two pips' (2-4 cents yes, 9 not good) is his; $0.06 between them and the 0.08 bounce are "
               "Claude's, from his marked charts. The old code had 0.003 of a day's range: typical $0.04 (2019) to "
               "$0.07 (2023), up to $0.09 on wide days - near the same, but it grows with the day, which #90 says it "
               "must not (plan/work/rules-missing.md R6)",
        aliases=("EQ_RULE", "EQ_TOL_FRAC_HIS", "EQ_BOUNCE_FRAC", "EQ_HTF", "equal_tol", "eq_rule"),
    ),
    "what_a_strong_zone_is": Rule(
        value=MappingProxyType(dict(calm_max=0.74, weak_after_touches=3)),
        means="His main zone is the QUIET one: small candles parked in a tight range (calm at most 0.74). Two quiet "
              "stretches at one price are stronger. Strongest the first time; weak after 2-3 touches.",
        taught="LEARNED #77, #78, #79, #80, #81",
        words="B and d are the my kind of zones. … Once. It is touched and uh, respected, then that becomes weak.",
        words_from=_W,
        old_code="old/ml/agent/levels.py ZONE_CALM_MAX = 0.74, ZONE_MAX_TOUCHES = 3",
        number="0.74 is Claude's, from his marked charts (#80: B 0.54 and D 0.69 his, A 0.79 and C 0.98 not); "
               "'two to three times' is his",
        aliases=("ZONE_CALM_MAX", "ZONE_MAX_TOUCHES", "zone_calm_max"),
    ),
    "a_zone_only_flips_once_it_is_touched": Rule(
        value="touched",
        means="A zone flips (support becomes resistance) only after price has come back and touched it. Price that "
              "never came back cannot have accepted the other side.",
        taught="strategy.md section 3; strategy-understanding section 66aa",
        words="A zone is strongest the first time; once touched and respected (2–3 times) it gets weak; it flips "
              "when broken",
        words_from=_S,
        old_code="old/ml/agent/levels.py ZONE_FLIP_RULE = 'touched'",
        number="- (Claude's reading of 'flips when broken', found while building #97)",
        aliases=("ZONE_FLIP_RULE", "zone_flip_rule"),
    ),
    "what_a_real_zone_break_is": Rule(
        value=0.20,
        means="Price must CLOSE through the zone by 20% of the zone's own height. A poke or a tiny wick is not a break.",
        taught="LEARNED #98",
        words="tiny wicks are not considered as a break of that zone.",
        words_from=_W,
        old_code="old/ml/agent/entries.py ZB_BREAK_FRAC = 0.20",
        number="Claude's, from his marked charts (#98 panel D: 0.040 of the height 'not broke', 0.437 'actual break')",
        aliases=("ZB_BREAK_FRAC", "zb_break_frac"),
    ),
    "what_a_session_high_or_low_is": Rule(
        value=MappingProxyType(dict(extreme="raw", windows_ist=(("Asia", "04:30", "10:30"),
                                                                ("London", "12:30", "15:30"),
                                                                ("New York", "17:30", "20:30")))),
        means="The plain high and low of his exact session window (India time), with no age limit. Asia 04:30-10:30, "
              "London 12:30-15:30, New York 17:30-20:30.",
        taught="LEARNED #83, #85; strategy.md section 1",
        words="now let's just consider the high and low of this session range, whatever it is, as a liquidity … "
              "IST: Asian 4:30–10:30 · London 12:30–15:30 · NY 17:30–20:30",
        words_from=_WS,
        old_code="old/ml/agent/levels.py SESSION_EXTREME = 'raw', SESSIONS (the same windows written in UTC minutes)",
        number="his",
        aliases=("SESSION_EXTREME", "SESSIONS", "session_extreme", "sessions"),
    ),
    "a_recent_5_minute_swing_is_a_target_only": Rule(
        value=True,
        means="A recent 5-minute swing high / low is never an entry. It is only a quick target, and only when the "
              "higher timeframe is against the trade.",
        taught="LEARNED #91",
        words="I'll consider those swing high or low as liquidity for quick targets, when higher time frame is against "
              "my trade, but not to enter or any other reason",
        words_from=_W,
        old_code="old/ml/agent/levels.py RECENT_SWINGS_TGT = True",
        number="-",
        aliases=("RECENT_SWINGS_TGT", "recent_swings_tgt"),
    ),
    "an_hour_swing_is_a_target_not_an_entry": Rule(
        value=("1d", "1w"),
        means="A 1-hour swing high / low is a target (any size), not entry liquidity. Only day and week swings may "
              "start a setup.",
        taught="LEARNED #93",
        words="They can we can be considered them as a liquidity, but they are not a very good way to enter the trade.",
        words_from=_W,
        old_code="old/ml/agent/levels.py HTF_SWING_ENTRY_TF = ('D1', 'W1')",
        number="-",
        aliases=("HTF_SWING_ENTRY_TF", "htf_swing_entry_tf"),
    ),
    "broker_days_not_utc_days": Rule(
        value="server",
        means="Daily and weekly candles are cut at the broker's midnight (MT5 server time), like his own MT5 chart - "
              "not at UTC midnight.",
        taught="strategy-understanding section 65i",
        words=None,
        words_from=(),
        old_code="old/ml/agent/levels.py DAY_CLOCK = 'server'",
        number="-",
        aliases=("DAY_CLOCK", "day_clock"),
        evidence="His MT5 chart cuts days at the server's midnight. UTC days made ~400 fake Sunday candles "
                 "(likhith.md B4). machine.data.bars builds days this way (L1).",
    ),
}

FIXED: Mapping[str, Rule] = MappingProxyType(_FIXED)

# ------------------------------------------------------------------------------------ PARTS of his other definitions
# Numbers that are his (or his spec's, confirmed) and live outside knobs.py. Same protection as FIXED.
_PARTS = {
    "swing_lookback": Rule(
        value=2,
        means="A swing high is a candle whose high is higher than the 2 candles before AND the 2 after (strictly); "
              "a swing low the mirror. Known only 2 candles later.",
        taught="strategy-code.md section 3 and 13 (B3, confirmed); strategy.md glossary",
        words="Swing high/low** — local high/low with 2 lower/higher candles each side.",
        words_from=_S,
        old_code="old/ml/agent/levels.py PIV = 2, swings_of()",
        number="his",
        aliases=("PIV", "swing_lookback", "piv"),
    ),
    "adr_days": Rule(
        value=(5, 10),
        means="His ADR indicator: the average of the last 5 and the last 10 finished days' ranges (high - low), on "
              "daily candles. Bands = today's open +/- half of each.",
        taught="strategy-code.md section 16 (his Pine v2 study); strategy.md section 7",
        words="ADR (Avg Daily Range) zones = daily OPEN ± half of 5/10-day range",
        words_from=_S,
        old_code="old/ml/agent/levels.py adr_bands()",
        number="his",
        aliases=("adr_days",),
    ),
    "fvg_shape": Rule(
        value="3_candles",
        means="A fair value gap is a 3-candle gap. For a buy (a move up): the 1st candle's high is below the 3rd "
              "candle's low; the gap runs from that high to that low. For a sell: the mirror.",
        taught="strategy.md section 2 and glossary; strategy-code.md section 10",
        words="3-candle imbalance; bullish = 1st-high ↔ 3rd-low.",
        words_from=_S,
        old_code="old/ml/agent/setup_engine.py detect(): l[j] > h[j - 2] (buy), h[j] < l[j - 2] (sell)",
        number="his",
        aliases=("fvg_shape",),
    ),
    "a_sweep_is_a_wick": Rule(
        value="wick",
        means="Every sweep counts, even 1 point through, no close needed.",
        taught="strategy-code.md section 7 (B1, confirmed); LEARNED #95",
        words="Every sweep counts** — even 1 point through, no close needed.",
        words_from=_S,
        old_code="old/ml/agent/levels.py LevelBook (the first bar that wicks a level is its sweep)",
        number="his",
        aliases=("sweep_mode", "sweep_buffer"),
    ),
}

PARTS: Mapping[str, Rule] = MappingProxyType(_PARTS)

ALL: Mapping[str, Rule] = MappingProxyType({**_FIXED, **_PARTS})

_ALIAS: dict[str, str] = {}
for _name, _rule in ALL.items():
    for _a in (_name, *_rule.aliases):
        if _a.lower() in _ALIAS and _ALIAS[_a.lower()] != _name:
            raise RuntimeError(f"two fixed rules share the name {_a!r}")
        _ALIAS[_a.lower()] = _name


def get(name: str) -> Any:
    """The value of one fixed rule or part: rules.get('where_in_the_gap_he_buys') -> 'near'."""
    return ALL[name].value


def fixed_name(name: str) -> str | None:
    """The fixed rule a name (or an old code name like 'entry_edge') points at, or None if it is not fixed."""
    return _ALIAS.get(str(name).lower())


def refusal(name: str, asked: Any = None) -> str:
    """The sentence that refuses a change to one fixed rule, with his words and the lesson."""
    key = fixed_name(name)
    if key is None:
        raise KeyError(f"{name!r} is not a fixed rule")
    r = ALL[key]
    said = f'His words ({r.taught}): "{r.words}"' if r.words else f"Shown by his own marks ({r.taught}): {r.evidence}"
    what = f"'{name}'" if name == key else f"'{name}' (= his fixed rule '{key}')"
    to = "" if asked is None else f" cannot be set to {asked!r}:"
    return (f"{what}{to} it is FIXED - what a thing IS, not a rule to test (LEARNED #121). It stays {r.value!r}: "
            f"{r.means} {said}")


def refuse_fixed(changes: Mapping[str, Any]) -> None:
    """Raise FixedRuleError if `changes` (a form's, a version's or a test's settings) touches any fixed rule.

    Refused even when the value asked for is the same one: a version must not carry his definitions at all, so that
    no later edit of the version can quietly change them.
    """
    hits = [refusal(k, v) for k, v in changes.items() if fixed_name(k) is not None]
    if hits:
        raise FixedRuleError("\n".join(hits))
