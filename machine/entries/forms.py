"""His entry types as idea forms (L19b): strategy.md section 9, each written the way his ideas are written (L7a).

    from machine.entries import forms
    for e in forms.ENTRY_TYPES.values():
        idea = e.idea()                      # machine.ideas.form.Idea - compile it like any idea
        for rd in e.readings():              # his open word, one reading at a time (never guessed - R13)
            ...

What each one is (his words are in the form's `words` and `sources`, word for word):
  entry-type1            Type 1 - his obvious liquidity taken -> a break of structure -> the FIRST gap of the breaking
                         move -> a limit at its near edge; the hedge at the grab's extreme (#9, #33, #61, #65, #111)
  entry-second-gap       Type 1, but when one move leaves two gaps and the trend is "VERY STRONG" the second gap
                         (#124); otherwise the first, as always. "Very strong" is his open word (four-boxes B2-03)
  entry-trendline-touch  Type 2 - the 3rd / 4th touch of a steep line inside a clear trend, no gap; the hedge a few
                         pips past the line, moving with it (#51, #69, #127). "Steep" is open (B2-01)
  entry-zone-break       the zone breakout as he trades it (#97: "we will not directly enter when it breaks ... it
                         should come again for retesting, there we will enter"; #98, #99, #119, #126 - "per week only we
                         find 4 to 5 trades")
  entry-triangle-false-break  P1 - a push out of a triangle fails and closes back in: that boundary's liquidity is
                         taken; then his Type 1 (a break, the first gap) towards the other boundary. "Big move" is open
                         (B2-02)
  entry-zone-entry       a strong zone with obvious liquidity just beyond it: the liquidity is taken -> a limit AT the
                         zone, no gap; the hedge just past its far edge; the hour and the day with it (#84, #86, #87)

Not one of these is new to the old project: every one is in plan/work/prior-tries.csv on 2016-2023 (`seen` below), so
no result here can be "works" - the referee's answer is at best "chance - needs fresh years" (CLAUDE.md rule 18).

strategy.md section 9 has two zone-break shapes: Type 3 on line 120 ("strong move -> triangle -> breaks flipped zone +
FVG + liquidity taken", four-boxes B1-46, 'strong move' open as B2-02) and the zone break retest on line 122 (#97-#99).
The prompt's Type 3 names #97, #98 and #126 - the zone break's own lessons (#97: "we will not directly enter when it
breaks ... it should come again for retesting, there we will enter") - so entry-zone-break is the retest. The line-120
shape needs "strong move" and a block the machine does not have (a gap made by the move that breaks a flipped zone);
plan/work/L19b-notes.md lists it as not built, and who would build it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from machine.ideas.form import Idea, reading_choices
from machine.referee.prior import IdeaKey

# ---- the parts every gap entry of his shares (strategy.md sections 2 and 9, Type 1) ------------------------------------
_GRAB_GAP_STEPS = [
    dict(id="sweep", what="price takes his obvious liquidity on the hedge side (a wick is enough, #95)",
         block="sweep", tf="5m",
         sources=[dict(ref="strategy.md section 9 (Type 1); section 1: session, trendline, equal highs / lows")]),
    dict(id="brk", what="structure breaks the trade's way (a wick through his order block, #56-#60)",
         block="break_of_structure", tf="5m", max_wait_bars=30,
         sources=[dict(ref="strategy.md section 9 (Type 1): 'BOS mandatory' (section 2)")]),
    dict(id="gap", what="the FIRST gap of the breaking move", block="gap_of_the_move", tf="5m", max_wait_bars=6,
         sources=[dict(ref="strategy.md section 2; LEARNED #61, #62, #101, #131")]),
]
_NEAR_EDGE = dict(how="limit", at="gap.near", max_wait_bars="no limit",
                  sources=[dict(ref="LEARNED #65 (the near edge), #68 ('it's not about the time': only the first "
                                    "visit counts)")])
_GRAB_HEDGE = dict(kind="hedge", at="brk.extreme",
                   sources=[dict(ref="strategy.md section 9: the hedge point = the extreme of the whole grab, not the "
                                     "gap edge (#9, #33, #111)")])
_NEXT_LIQUIDITY = [dict(kind="next_liquidity", sources=[dict(ref="strategy.md section 10: 'Target = next liquidity'")])]
_NOT_FINISHED = [
    dict(name="target_already_reached", params={"since": "sweep"},
         sources=[dict(ref="LEARNED #119 (T05): the move is already finished")]),
    dict(name="target_reached_first",
         sources=[dict(ref="strategy.md section 2: 'If price took the target-side liquidity before coming back -> "
                           "no entry' (#68, #119)")]),
]


@dataclass(frozen=True)
class EntryType:
    id: str
    title: str                          # plain words, for the report
    form: dict                          # machine.ideas.form.Idea as a dict
    seen: IdeaKey                       # its row(s) in plan/work/prior-tries.csv, in the list's names
    seen_rows: tuple[str, ...]          # the rows it matches there (checked by tests/test_entries.py)
    closer: bool = False                # one of his entries that close a hedged leg (T12 M10, his5)
    needs_1m: bool = False              # reads the 1-minute chart (the 1-minute lines inside a zone, #98 C)
    his_count: str = ""                 # what he said about how often he sees it
    open_word: str = ""                 # his open word (four-boxes Box 2), if any
    fill_checks: tuple[str, ...] = ()   # the checks made while its order waits (closers.py)
    extra: dict = field(default_factory=dict)

    def idea(self) -> Idea:
        return Idea.load(copy.deepcopy(self.form))

    def readings(self) -> list[dict[str, str]]:
        """Every reading of its open word, one version each ({} when nothing is open)."""
        return reading_choices(self.idea())


def _form(id_: str, title: str, words: str, steps: list, entry: dict, exit_: dict, targets: list, dne: list = (),
          filters: list = (), settings: dict | None = None, unclear: list = (), sources: list = ()) -> dict:
    return dict(id=id_, title=title, words=words, timeframe="5m", directions=["buy", "sell"],
                steps=copy.deepcopy(steps), entry=copy.deepcopy(entry), exit=copy.deepcopy(exit_),
                targets=copy.deepcopy(targets), do_not_enter=copy.deepcopy(list(dne)),
                filters=copy.deepcopy(list(filters)), settings=dict(settings or {}),
                unclear=copy.deepcopy(list(unclear)), sources=list(sources), status="FORM DRAFTED",
                notes="L19b: one of his own entry types (machine/entries/forms.py), not an idea from the phone")


_HOUR_AND_DAY = dict(name="higher_timeframe", params={"charts": ["1h", "1d"], "must": "with"})

# ================================================================================================================
# the six
# ================================================================================================================

TYPE1 = EntryType(
    id="entry-type1", title="Type 1: liquidity taken, break, the first gap (grab + gap)",
    form=_form(
        "entry-type1", "Type 1: liquidity taken -> break -> the first gap",
        "Type 1 Liquidity-sweep → BOS → FVG (prediction). Hedge point (never called \"stop\") = the high (sell) / low "
        "(buy) of the swing where the liquidity was taken and the break formed — the extreme of the whole grab, not "
        "the gap edge (#9, #33, #111).",
        _GRAB_GAP_STEPS, _NEAR_EDGE, _GRAB_HEDGE, _NEXT_LIQUIDITY, _NOT_FINISHED,
        sources=[dict(ref="strategy.md section 9, Type 1 (line 118)"),
                 dict(ref="strategy.md section 2: 'The entry gap is the FIRST gap after the liquidity is taken ... "
                          "Enter at the NEAR edge ... target = next liquidity'")]),
    seen=IdeaKey.make("liquidity_sweep+break_of_structure+fvg", "5m", "gap:near"),
    seen_rows=("type1_grab_gap",), closer=True,
    his_count="#126 (for his trades overall): 'per week only we find 4 to 5 trades'")

SECOND_GAP = EntryType(
    id="entry-second-gap", title="Type 1 with the second gap when the trend is very strong (#124)",
    form=_form(
        "entry-second-gap", "Type 1, the second gap only with very strong agreement",
        "SECOND IS ONLY WHEN IT IS VERY STRONG BULLISH, OTHER CONFIDENCE ALSO AGREES, and only the second, OR ELSE IT "
        "SHOULD WAIT FOR THE FIRST.",
        _GRAB_GAP_STEPS, _NEAR_EDGE, _GRAB_HEDGE, _NEXT_LIQUIDITY, _NOT_FINISHED,
        settings={"which_gap_of_the_move": "second"},
        unclear=[dict(
            id="B2-03", kind="ask", part="how strong 'very strong' is before the second gap is taken",
            words="SECOND IS ONLY WHEN IT IS VERY STRONG BULLISH, OTHER CONFIDENCE ALSO AGREES",
            searched="Searched strategy.md sections 2 and 6, LEARNED #3, #115, #122-#124 and four-boxes B2-03 for "
                     "what 'very strong' is - no number or chart is named; the closest line is #124 itself.",
            readings=[
                dict(id="A", means="the hour AND the day with the trade and the week not against it (the old "
                                   "engine's reading)",
                     changes={"settings.the_second_gap_needs_the_day": "strong"}),
                dict(id="B", means="the hour, the day AND the week all with the trade",
                     changes={"settings.the_second_gap_needs_the_day": "all"}),
            ],
            sources=[dict(ref="four-boxes B2-03 (waits for the weekly picture batch)")])],
        sources=[dict(ref="LEARNED #124 (20 Sep)"),
                 dict(ref="strategy.md section 2 (line 83): 'Two stacked gaps: take the one nearer the hedge ... The "
                          "second gap only when the trend is very strong and other things agree'")]),
    seen=IdeaKey.make("liquidity_sweep+break_of_structure+fvg", "5m", "gap:second"),
    seen_rows=("second_gap", "type1_grab_gap"), open_word="B2-03 'very strong'")

TRENDLINE_TOUCH = EntryType(
    id="entry-trendline-touch", title="Type 2: the 3rd / 4th touch of a steep trendline (no gap)",
    form=_form(
        "entry-trendline-touch", "Type 2: the trendline touch",
        "Type 2 Trendline 3rd/4th touch, no gap needed. Only a steep line inside a clear trend (a falling line in a "
        "downtrend to sell, a rising line in an uptrend to buy). A flat line is never a touch entry — it will break; "
        "wait for the break or false break, then the gap. Hedge a few pips past the line; the hedge moves with the "
        "line. The higher timeframe must be with it.",
        [dict(id="touch", what="the 3rd (or 4th) touch of a steep falling line through highs (sell) or rising line "
                               "through lows (buy), a clear trend between its touches", block="trendline_touch",
              tf="5m", sources=[dict(ref="strategy.md section 9, Type 2 (line 119); LEARNED #51, #92, #116, #127")])],
        dict(how="limit", at="touch.entry", max_wait_bars=10,
             sources=[dict(ref="a limit on the line - the line's price 10 candles on, the worse fill of that wait "
                               "(settings trendline_touch_fill_bars, Claude's; four-boxes B3-16)")]),
        dict(kind="hedge", at="touch.hedge",
             sources=[dict(ref="strategy.md section 9: 'Hedge a few pips past the line; the hedge moves with the "
                               "line' (#69; T12 M27)")]),
        _NEXT_LIQUIDITY,
        [dict(name="target_reached_first", sources=[dict(ref="LEARNED #119")])],
        [dict(_HOUR_AND_DAY, sources=[dict(ref="strategy.md section 9: 'The higher timeframe must be with it'; #69: "
                                               "'day and hour, both'")])],
        unclear=[dict(
            id="B2-01", kind="ask", part="how steep the line must be for the touch entry",
            words="i trendline touch is real and the steep of the line decides that",
            searched="Searched strategy.md section 9, strategy-code.md section 14, LEARNED #51, #69, #92, #115, #116, "
                     "#118, #127 and the inbox for how steep - no number; the closest is #115 '45 degree ... not a "
                     "hard rule' (plan/work/question-queue/B2-01-steep.md).",
            readings=[
                dict(id="A", means="the line travels at least 0.15 of a day's range in a day",
                     changes={"settings.trendline_min_steepness_adr": 0.15}),
                dict(id="B", means="at least 0.53 of a day's range in a day",
                     changes={"settings.trendline_min_steepness_adr": 0.53}),
                dict(id="C", means="at least 0.93 of a day's range in a day",
                     changes={"settings.trendline_min_steepness_adr": 0.93}),
                dict(id="D", means="at least 1.31 of a day's range in a day",
                     changes={"settings.trendline_min_steepness_adr": 1.31}),
            ],
            sources=[dict(ref="four-boxes B2-01 (waits for the weekly picture batch)")])],
        sources=[dict(ref="LEARNED #127 (21 Sep): 'the trendline touch is real and the steep of the line decides "
                          "that'"),
                 dict(ref="LEARNED #69 (15 Sep): 'with the hedge of that breaking the trend line itself it's a "
                          "trailing'")]),
    seen=IdeaKey.make("trendlines", "5m", "line:touch"),
    seen_rows=("type2_trendline_touch", "type2_touch_rules", "trailing_cover"), closer=True,
    open_word="B2-01 'steep'")

ZONE_BREAK = EntryType(
    id="entry-zone-break", title="Type 3, the zone breakout as he trades it: a real break, then the retest",
    form=_form(
        "entry-zone-break", "the zone break: a real break, then the retest",
        "we will not directly enter when it breaks but it should break and close above and it should come come again "
        "for retesting there we will enter",
        [dict(id="zb", what="a strong zone breaks for real (a close 0.20 of its height through it, coming through "
                            "from the other side); liquidity inside it taken first, then the gap (#99)",
              block="zone_break", tf="5m",
              sources=[dict(ref="strategy.md section 9 (line 122); LEARNED #97, #98, #99, #119 (T06)")])],
        dict(how="limit", at="zb.entry", max_wait_bars="no limit",
             sources=[dict(ref="#97: 'it should come again for retesting, there we will enter'; #98: 'there is no "
                               "particular distance'")]),
        dict(kind="hedge", at="zb.hedge",
             sources=[dict(ref="just past the zone's far edge; at the all-time high the next strong zone (#97)")]),
        [dict(kind="next_liquidity", kinds=["equal", "session", "trendline"],
              sources=[dict(ref="#98: 'there should be a liquidity upside ... where I can target' - his obvious "
                                "liquidity (#84, #86)")])],
        [dict(name="target_reached_first",
              sources=[dict(ref="#98: 'when it is coming to retest, it shouldn't take the obvious liquidity'")])],
        [dict(_HOUR_AND_DAY, sources=[dict(ref="#97: 'first there should be a higher time frame confluence and very "
                                               "very good directional confirmation'")])],
        sources=[dict(ref="LEARNED #97 (19 Sep), #98 (19 Sep), #99, #119 (T05, T06), #126 (21 Sep)")]),
    seen=IdeaKey.make("strong_zones+zone_break", "5m", "zone:retest"),
    seen_rows=("type3_zone_breakout", "zone_break_retest_97"), closer=True, needs_1m=True,
    his_count="#126: 'No it's not that rare, per week only we find 4 to 5 trades'",
    fill_checks=("opposite_setup",))

TRIANGLE_FALSE_BREAK = EntryType(
    id="entry-triangle-false-break", title="P1: the triangle false break, then his Type 1",
    form=_form(
        "entry-triangle-false-break", "P1: the triangle false break",
        "Triangle: two lines that clearly narrow toward each other, with price moving INSIDE them, then the break; "
        "after a big move; both lines = liquidity (#116, #118). ⭐ False-breakout: first push fails → grabs "
        "liquidity, leaves FVG, reverses (P1).",
        [dict(id="tri", what="a triangle after a big move", block="triangle", tf="5m",
              sources=[dict(ref="strategy.md section 4 (line 92); rules.py what_a_triangle_is (#118)")]),
         dict(id="fb", what="a push out of one line that closes back inside: its liquidity is taken",
              block="false_breakout", tf="5m", max_wait_bars="no limit", params={"of": "tri"},
              sources=[dict(ref="strategy-code.md section 14: 'price wicks beyond a boundary but returns inside "
                                "within fakeout_lookback bars'")]),
         dict(id="brk", what="structure breaks back the other way", block="break_of_structure", tf="5m",
              max_wait_bars=30, params={"after": "fb"},
              sources=[dict(ref="strategy-code.md section 14: 'TRADE VIA ENTRY TYPE 1 FSM'")]),
         dict(id="gap", what="the first gap of that move", block="gap_of_the_move", tf="5m", max_wait_bars=6,
              params={"sweep": "fb", "move": "brk"}, sources=[dict(ref="LEARNED #61, #131")])],
        _NEAR_EDGE,
        dict(kind="hedge", at="brk.extreme",
             sources=[dict(ref="the extreme of the whole grab - the poke out of the triangle included (#9, #33, "
                               "#111)")]),
        [dict(kind="price", at="fb.opposite",
              sources=[dict(ref="strategy-code.md section 14: 'reversal to the OPPOSITE boundary'; strategy.md "
                                "section 4: 'both lines = liquidity'")])],
        [dict(name="target_already_reached", params={"since": "fb"}, sources=[dict(ref="LEARNED #119")]),
         dict(name="target_reached_first", sources=[dict(ref="LEARNED #119")])],
        unclear=[dict(
            id="B2-02", kind="ask", part="how big 'a big move' before the triangle is",
            words="after a big move",
            searched="Searched strategy.md sections 4 and 9, strategy-code.md section 14, LEARNED #116, #118, #119, "
                     "#126 and the inbox for the size of the move - nothing; the closest is strategy-code.md section "
                     "14 'following a prior impulse (big move)' (plan/work/question-queue/B2-02-big-move.md).",
            readings=[
                dict(id="A", means="the day before it moved at least half a day's range",
                     changes={"settings.big_move_adr": 0.5}),
                dict(id="B", means="at least one day's range", changes={"settings.big_move_adr": 1.0}),
                dict(id="C", means="at least one and a half days' range", changes={"settings.big_move_adr": 1.5}),
            ],
            sources=[dict(ref="four-boxes B2-02 (waits for the weekly picture batch)")])],
        sources=[dict(ref="strategy.md section 4 (P1); strategy-code.md section 14 (FALSE BREAKOUT)")]),
    seen=IdeaKey.make("triangles+false_breakout", "5m", "pattern:false_break"),
    seen_rows=("p1_triangle_false_break",), closer=True, open_word="B2-02 'big move'")

ZONE_ENTRY = EntryType(
    id="entry-zone-entry", title="The zone entry: liquidity just beyond a strong zone taken, enter AT the zone",
    form=_form(
        "entry-zone-entry", "the zone entry (no gap)",
        "when there is a strong zone and above that strong zone there is an obvious liquidity, when that liquidity "
        "took and it drops to the zone, then I enter there, because that will give me better risk to reward, so the "
        "zone low will be my hedge point, so I don't wait for fair value gap there",
        [dict(id="z", what="one of his obvious levels just beyond a live strong zone is taken; the candle stays "
                           "outside the zone; that side of the zone not used yet", block="zone_entry", tf="5m",
              sources=[dict(ref="strategy.md section 9 (line 121); LEARNED #84, #86, #78")])],
        dict(how="limit", at="z.entry", max_wait_bars="no limit",
             sources=[dict(ref="#84: 'then I enter there' - AT the zone's near edge")]),
        dict(kind="hedge", at="z.hedge", sources=[dict(ref="#84: 'the zone low will be my hedge point'")]),
        _NEXT_LIQUIDITY,
        [dict(name="target_reached_first", sources=[dict(ref="LEARNED #119")]),
         dict(name="unfilled_gap_past_hedge",
              sources=[dict(ref="#87: 'there was an unfilled gap above ... that will hit my hedge' - unless obvious "
                                "liquidity waits on the target side")])],
        [dict(_HOUR_AND_DAY, sources=[dict(ref="#84: 'even the higher time frame is not favoring me, then I would "
                                               "wait for fair value gap'; strategy.md section 9: 'The higher "
                                               "timeframe must be with it'")])],
        sources=[dict(ref="LEARNED #84 (15 Sep), #86 (15 Sep), #87")]),
    seen=IdeaKey.make("strong_zones+zone_entry+obvious_liquidity", "5m", "zone:edge"),
    seen_rows=("zone_entry_84",), closer=True, fill_checks=("gap_excuse",))

ENTRY_TYPES: dict[str, EntryType] = {e.id: e for e in (TYPE1, SECOND_GAP, TRENDLINE_TOUCH, ZONE_BREAK,
                                                      TRIANGLE_FALSE_BREAK, ZONE_ENTRY)}


def get(entry_id: str) -> EntryType:
    if entry_id not in ENTRY_TYPES:
        raise KeyError(f"{entry_id!r} is not one of his entry types: {', '.join(ENTRY_TYPES)}")
    return ENTRY_TYPES[entry_id]
