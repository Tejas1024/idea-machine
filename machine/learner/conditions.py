"""The conditions (L13.1): each one splits an idea's trades into named parts, from what was known at the setup.

    from machine.learner import conditions
    labels = conditions.BY_NAME["session"].labels(feats)      # "Asia" / "London" / "New York" / "in between" per setup
    conditions.for_slice(slice)                                # the condition a form's slice names, or None

Where they come from: plan/work/conditions.md (T7: C1-C15, "none missing" - his answer), the prompt's list (session,
weekday, hour, news, bias, ADR used, distance to liquidity, recent volatility), and strategy/COMPONENTS.md (COMPONENTS
below says which condition covers each of its 31 pieces). A condition's NAME is what a form's slice names
(plan/work/ideas/<idea>/form.json "slices"): the machine never guesses which condition a slice means - an unknown name
is "not measured" (R13).

Every cut point is Claude's (none is his rule), fixed before any condition table was seen (plan/work/L131-notes.md).
"A day's range" = his 10-day ADR. Only a slice chosen in the form BEFORE the test can become "works best in / fails
in" (machine/learner/slices.py); every other condition here is "looked at, not tested".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

NEWS_NEAR_MIN = 60             # "news near": a red / orange release within 60 minutes (COMPONENTS.md event_window_min)
NEWS_KIND_MIN = 120            # the kind of the nearest release within 2 hours (L12b.2: red news calm after ~2 hours)
DAY = "of a day's range"


@dataclass(frozen=True)
class Condition:
    name: str                  # what a form's slice names
    label: str                 # how a report names it (easy English)
    dimension: str             # likhith.md section D: direction, session, news, higher_timeframe_trend, volatility,
                               # market_type, place, time
    source: str                # his words' place: conditions.md C.., COMPONENTS #.., the prompt
    how: str                   # how it is measured
    groups: tuple[str, ...]    # its parts, in the order shown
    fn: Callable[[pd.DataFrame], list]
    needs_facts: bool = False  # True: it reads what the idea's own steps found (the level taken, the gap ...)

    def labels(self, feats: pd.DataFrame) -> pd.Series:
        """One part per setup (None = not known / does not apply to this idea)."""
        out = pd.Series(self.fn(feats), index=feats.index, dtype=object)
        bad = set(out.dropna().unique()) - set(self.groups)
        if bad:
            raise ValueError(f"condition {self.name}: parts {sorted(bad)} are not in its list {self.groups}")
        return out


def _known(v) -> bool:
    return v is not None and not pd.isna(v)


def _text(col: str, names: dict[str, str]) -> Callable:
    return lambda f: [names.get(v) if _known(v) else None for v in f[col]]


def _flag(col: str, yes: str, no: str) -> Callable:
    return lambda f: [(yes if bool(v) else no) if _known(v) else None for v in f[col]]


def _cut(col: str, edges: tuple[float, ...], names: tuple[str, ...], missing: str | None = None,
         when: str | None = None) -> Callable:
    """v < edges[0] -> names[0]; edges[0] <= v < edges[1] -> names[1]; ... A value not known -> `missing` (a part of
    its own, only on rows where column `when` is known), else None."""
    def fn(f: pd.DataFrame) -> list:
        out = []
        for k, v in enumerate(f[col]):
            if _known(v):
                out.append(names[int(np.searchsorted(edges, float(v), "right"))])
            elif missing is not None and (when is None or _known(f[when].iat[k])):
                out.append(missing)
            else:
                out.append(None)
        return out
    return fn


def _is(col: str, value: str, yes: str, no: str) -> Callable:
    return lambda f: [(yes if v == value else no) if _known(v) else None for v in f[col]]


def _news_near(f: pd.DataFrame) -> list:
    out = []
    for nxt, last in zip(f["news_next_min"], f["news_last_min"]):
        a = float(nxt) if _known(nxt) else np.inf
        b = float(last) if _known(last) else np.inf
        out.append("news in the next hour" if a <= NEWS_NEAR_MIN and a <= b else
                   "news in the last hour" if b <= NEWS_NEAR_MIN else "no news within an hour")
    return out


def _nearest(f: pd.DataFrame, what: str, within: float, none: str) -> list:
    out = []
    for nxt, last, vn, vl in zip(f["news_next_min"], f["news_last_min"], f[f"news_next_{what}"],
                                 f[f"news_last_{what}"]):
        a = float(nxt) if _known(nxt) else np.inf
        b = float(last) if _known(last) else np.inf
        out.append(none if min(a, b) > within else vn if a <= b else vl)
    return out


def _week_and_day(f: pd.DataFrame) -> list:
    out = []
    for w, d in zip(f["bias_week"], f["bias_day"]):
        if not (_known(w) and _known(d)):
            out.append(None)
        elif "neutral" in (w, d):
            out.append("one is neutral")
        else:
            out.append("they agree" if w == d else "they disagree")
    return out


def _trend_bias(f: pd.DataFrame) -> list:
    out = []
    for h, d, w in zip(f["bias_hour"], f["bias_day"], f["bias_week"]):
        if not (_known(h) and _known(d) and _known(w)):
            out.append(None)
        else:
            out.append("trending" if h == d == w and h != "neutral" else "sideways")
    return out


def _volatility(f: pd.DataFrame) -> list:
    return [("low" if v < 1 / 3 else "high" if v > 2 / 3 else "normal") if _known(v) else None
            for v in f["volatility_share"]]


def _touches(f: pd.DataFrame) -> list:
    return [("3 or more touches" if v >= 3 else "2 touches" if v == 2 else "1 touch") if _known(v) else None
            for v in f["level_touches"]]


SIDE = {"with": "with the trade", "against": "against the trade", "neutral": "neutral"}
SIDES = tuple(SIDE.values())
KIND = {"equal": "equal highs / lows", "trendline": "a trendline", "session": "a session high / low",
        "swing": "a day / week swing"}
CHART = {"5m": "5-minute", "1h": "hour", "1d": "day", "1w": "week", "1m": "1-minute"}
YES_NO = ("yes", "no")
SHARE3 = (f"under 0.25 {DAY}", "0.25 to 0.5", "0.5 or more")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
HOURS = tuple(f"{h:02d}:00" for h in range(24))

ALL: tuple[Condition, ...] = (
    # ---- direction, session, time (dimensions 2, 3, 9)
    Condition("direction", "buy or sell", "direction", "plan/README.md section 5 (the machine checks it by itself)",
              "the trade's side", ("buy", "sell"), _text("direction", {"buy": "buy", "sell": "sell"})),
    Condition("session", "session (India time)", "session", "conditions.md C1; COMPONENTS #20",
              "the India time of the setup's moment: Asia 4:30-10:30, London 12:30-15:30, New York 17:30-20:30, else "
              "in between (the same rule as the form's session filter)",
              ("Asia", "London", "New York", "in between"),
              _text("session", {k: k for k in ("Asia", "London", "New York", "in between")})),
    Condition("new_york", "New York hours", "session", "conditions.md C2", "New York 17:30-20:30 India time, or not",
              ("New York", "other hours"), _is("session", "New York", "New York", "other hours")),
    Condition("hour", "hour of the day (India time)", "time", "plan/README.md section 5; COMPONENTS #20",
              "the India-time hour of the setup's moment", HOURS,
              lambda f: [HOURS[int(v)] if _known(v) else None for v in f["ist_hour"]]),
    Condition("weekday", "day of the week", "time", "plan/README.md section 5",
              "the broker's day of the setup (the day of his daily candle)", WEEKDAYS,
              lambda f: [v if v in WEEKDAYS else None for v in f["weekday"]]),
    Condition("friday", "Friday", "time", "conditions.md C10", "the broker's Friday (the week's last day), or not",
              ("Friday", "other days"), _is("weekday", "Friday", "Friday", "other days")),
    Condition("monday", "Monday", "time", "conditions.md C11", "the broker's Monday, or not",
              ("Monday", "other days"), _is("weekday", "Monday", "Monday", "other days")),
    Condition("week_end", "hours left in the week", "time", "conditions.md C10",
              "hours from the setup to the end of the broker's week",
              ("under 12 hours left", "12 to 48 hours left", "48 hours or more left"),
              _cut("hours_to_week_end", (12.0, 48.0), ("under 12 hours left", "12 to 48 hours left",
                                                       "48 hours or more left"))),
    # ---- news (dimension 4): his news = Forex Factory red + orange (T7)
    Condition("news_near", "news within an hour", "news", "conditions.md C5; COMPONENTS #21",
              "a red or orange US release (Forex Factory) within 60 minutes: the next scheduled one, or the last one "
              "out; the nearer of the two decides",
              ("news in the next hour", "news in the last hour", "no news within an hour"), _news_near),
    Condition("news_kind", "kind of the nearest news", "news", "conditions.md C5",
              "the kind of the nearest red or orange release within 2 hours, before or after",
              ("jobs", "inflation", "the Fed", "other news", "no news within 2 hours"),
              lambda f: _nearest(f, "kind", NEWS_KIND_MIN, "no news within 2 hours")),
    Condition("news_colour", "colour of the nearest news", "news", "conditions.md C5 (four-boxes B3-29)",
              "the colour of the nearest release within 60 minutes, before or after",
              ("red", "orange", "no news within an hour"),
              lambda f: _nearest(f, "colour", NEWS_NEAR_MIN, "no news within an hour")),
    # ---- the higher charts (dimension 5)
    Condition("dial", "the hour AND the day", "higher_timeframe_trend", "conditions.md C3",
              "his bias on the hour chart and on the day chart, each read on its own candles: both with the trade, "
              "both against it, or mixed",
              ("both with the trade", "both against the trade", "mixed"),
              _text("dial", {"with": "both with the trade", "against": "both against the trade", "mixed": "mixed"})),
    Condition("bias_hour", "the hour chart", "higher_timeframe_trend", "COMPONENTS #29",
              "his bias on the hour chart against the trade", SIDES, _text("bias_hour", SIDE)),
    Condition("bias_day", "the day chart", "higher_timeframe_trend", "COMPONENTS #14",
              "his bias on the day chart against the trade", SIDES, _text("bias_day", SIDE)),
    Condition("bias_week", "the week chart", "higher_timeframe_trend", "conditions.md C4; COMPONENTS #15",
              "his bias on the week chart against the trade", SIDES, _text("bias_week", SIDE)),
    Condition("week_and_day", "the week and the day", "higher_timeframe_trend", "conditions.md C4",
              "the week chart's bias against the day chart's", ("they agree", "they disagree", "one is neutral"),
              _week_and_day),
    # ---- volatility (dimension 6)
    Condition("volatility", "today's ADR against its last 12 months", "volatility", "conditions.md C6",
              "today's ADR against the ADR of each day of the last 12 months: the lowest third, the middle, the "
              "highest third", ("low", "normal", "high"), _volatility),
    Condition("adr_used", "how much of a day's range today has used", "volatility", "the prompt (ADR used)",
              "today's high minus low so far, against a day's range",
              (f"under half {DAY}", "half to a full day's range", "a full day's range or more"),
              _cut("adr_used", (0.5, 1.0), (f"under half {DAY}", "half to a full day's range",
                                            "a full day's range or more"))),
    Condition("recent_range", "the last hour", "volatility", "the prompt (recent volatility)",
              "the high minus low of the last 12 five-minute candles, against a day's range",
              (f"quiet (under 0.15 {DAY})", "normal (0.15 to 0.30)", "busy (0.30 or more)"),
              _cut("recent_range", (0.15, 0.30), (f"quiet (under 0.15 {DAY})", "normal (0.15 to 0.30)",
                                                  "busy (0.30 or more)"))),
    # ---- sideways or trending (dimension 7): "a few trend measures" (four-boxes B3-08)
    Condition("trend_bias", "trend: the hour, day and week agree", "market_type", "conditions.md C8; COMPONENTS #26",
              "trending = his bias on the hour, day and week charts all point the same way; else sideways (built "
              "only from his own bias)", ("trending", "sideways"), _trend_bias),
    Condition("trend_days", "trend: the last 5 days", "market_type", "conditions.md C8",
              "the last 5 finished days: their net move against their whole range; half or more = trending",
              ("trending", "sideways"), _cut("trend_days", (0.5,), ("sideways", "trending"))),
    # ---- place (dimension 8)
    Condition("adr_zone", "the entry in today's ADR zones", "place", "conditions.md C7; COMPONENTS #11",
              "where the entry price sits in today's ADR zones (his green and red zones)",
              ("near the green", "middle", "near the red", "beyond a zone"),
              _text("adr_zone", {k: k for k in ("near the green", "middle", "near the red", "beyond a zone")})),
    Condition("zone_at_entry", "a strong zone at the entry", "place", "conditions.md C12; COMPONENTS #12",
              f"a live strong zone of his holds the entry price, or its edge is within 0.05 {DAY}",
              ("at a strong zone", "no zone at the entry"), _flag("zone_at_entry", "at a strong zone",
                                                                 "no zone at the entry")),
    Condition("zone_at_target", "a strong zone at the target", "place", "COMPONENTS #13",
              f"a live strong zone holds the target price, or its edge is within 0.05 {DAY}",
              ("a zone at the target", "no zone at the target"), _flag("zone_at_target", "a zone at the target",
                                                                       "no zone at the target")),
    Condition("at_ath", "the all-time high", "place", "conditions.md C9; COMPONENTS #17",
              "the last 5-minute close is at the real all-time high (the record starts from September 2011)",
              ("at the all-time high", "below it"), _flag("at_ath", "at the all-time high", "below it")),
    Condition("hedge_side_gap", "an unfilled gap on the hedge side", "place", "conditions.md C13",
              "an unfilled gap past the hedge point at the setup (his #87)", YES_NO,
              _flag("gap_past_hedge", "yes", "no")),
    Condition("line_in_path", "a trendline in the path", "place", "conditions.md C14; COMPONENTS #16",
              "a live line of the 5-minute or hour chart stands between the entry and the target", YES_NO,
              _flag("line_in_path", "yes", "no")),
    Condition("line_against", "a line against the entry", "place", "conditions.md C14 (#111)",
              "a live line of the 1-minute, 5-minute or hour chart on the hedge side, between the entry and the "
              "hedge point (his 1-minute check would wait)", YES_NO, _flag("line_against", "yes", "no")),
    Condition("line_retest", "a broken trendline at the entry", "place", "COMPONENTS #24",
              f"a trendline (3 touches) broken in the last 5 days is priced at the entry now (within 0.05 {DAY})",
              YES_NO, _flag("line_retest_at_entry", "yes", "no")),
    Condition("zone_flip", "a flipped zone at the entry", "place", "COMPONENTS #25",
              f"a strong zone that flipped in the last 10 days holds the entry (within 0.05 {DAY})", YES_NO,
              _flag("zone_flip_at_entry", "yes", "no")),
    Condition("liq_target", "the next liquidity on the target side", "place", "the prompt; COMPONENTS #18",
              "from the entry to the nearest live level of his past it on the target side", (*SHARE3, "none"),
              _cut("liq_target_adr", (0.25, 0.5), SHARE3, missing="none", when="adr")),
    Condition("liq_hedge", "the next liquidity on the hedge side", "place", "the prompt (distance to liquidity)",
              "from the entry to the nearest live level of his past it on the hedge side", (*SHARE3, "none"),
              _cut("liq_hedge_adr", (0.25, 0.5), SHARE3, missing="none", when="adr")),
    Condition("equal_other_side", "untaken equal highs / lows on the target side", "place", "COMPONENTS #30",
              "from the entry to the nearest live equal highs / lows past it on the target side", (*SHARE3, "none"),
              _cut("equal_target_adr", (0.25, 0.5), SHARE3, missing="none", when="adr")),
    Condition("hedge_size", "the hedge distance", "place", "COMPONENTS #19",
              "from the entry to the hedge point, against a day's range",
              (f"under 0.05 {DAY}", "0.05 to 0.10", "0.10 or more"),
              _cut("hedge_adr", (0.05, 0.10), (f"under 0.05 {DAY}", "0.05 to 0.10", "0.10 or more"))),
    Condition("room", "the room to the target", "place", "COMPONENTS #18",
              "from the entry to the nearest target, against a day's range", SHARE3,
              _cut("room_adr", (0.25, 0.5), SHARE3)),
    Condition("reward", "the target against the hedge", "place", "strategy.md section 10 ('1:5, 1:6+')",
              "the distance to the target divided by the distance to the hedge point",
              ("under 2 times", "2 to 5 times", "5 times or more"),
              _cut("reward", (2.0, 5.0), ("under 2 times", "2 to 5 times", "5 times or more"))),
    # ---- what the idea's own steps found (place / time)
    Condition("level_kind", "the liquidity taken", "place", "conditions.md C12; COMPONENTS #1-#4, #31",
              "the kind of level the setup took (his order: equal, then trendline, then session)",
              tuple(KIND.values()), _text("level_kind", KIND), needs_facts=True),
    Condition("pair_chart", "the chart of the level taken", "place", "conditions.md C12 (his #90)",
              "the chart the level taken was on", tuple(CHART.values()), _text("level_chart", CHART),
              needs_facts=True),
    Condition("level_touches", "touches of the level taken", "place", "COMPONENTS #1 ('3>2')",
              "how many touches the level had when it was taken", ("1 touch", "2 touches", "3 or more touches"),
              _touches, needs_facts=True),
    Condition("recent", "when the level taken was formed", "time", "conditions.md C15; COMPONENTS #28",
              "the broker day the level was formed on, against the day it was taken",
              ("today", "yesterday", "older"),
              _text("level_day", {k: k for k in ("today", "yesterday", "older")}), needs_facts=True),
    Condition("grab_depth", "how far the grab went past the level", "place", "COMPONENTS #5",
              "from the level to the grab's extreme, against a day's range",
              (f"under 0.05 {DAY}", "0.05 to 0.15", "0.15 or more"),
              _cut("grab_depth_adr", (0.05, 0.15), (f"under 0.05 {DAY}", "0.05 to 0.15", "0.15 or more")),
              needs_facts=True),
    Condition("break_size", "how far the break went", "place", "COMPONENTS #6",
              "how far the break candle's wick went past his order block, against a day's range",
              (f"under 0.02 {DAY}", "0.02 to 0.05", "0.05 or more"),
              _cut("break_beyond_adr", (0.02, 0.05), (f"under 0.02 {DAY}", "0.02 to 0.05", "0.05 or more")),
              needs_facts=True),
    Condition("gap_size", "the size of the gap", "place", "COMPONENTS #8 (four-boxes B4-10)",
              "the entry gap's size, against a day's range",
              (f"under 0.02 {DAY}", "0.02 to 0.05", "0.05 or more"),
              _cut("gap_size_adr", (0.02, 0.05), (f"under 0.02 {DAY}", "0.02 to 0.05", "0.05 or more")),
              needs_facts=True),
    Condition("break_to_gap", "when the gap came, against the break", "time", "COMPONENTS #9",
              "5-minute candles from the break candle to the gap's third candle (the gap may be made inside the "
              "breaking move, before the break candle itself)",
              ("at the break candle or before it", "the candle after the break", "2 candles or more after"),
              _cut("break_to_gap", (1.0, 2.0), ("at the break candle or before it", "the candle after the break",
                                                "2 candles or more after")), needs_facts=True),
    Condition("leg_position", "the entry inside the move", "place", "COMPONENTS #10",
              "the entry between the grab's extreme and the far end of the move so far",
              ("the third nearest the grab", "the middle third", "the third furthest from the grab"),
              _cut("leg_position", (1 / 3, 2 / 3), ("the third nearest the grab", "the middle third",
                                                    "the third furthest from the grab")), needs_facts=True),
    Condition("target_kind", "the kind of level the target is", "place", "COMPONENTS #27",
              "the kind of level the idea's target is (when the target is a level a step found)",
              tuple(KIND.values()), _text("target_kind", KIND), needs_facts=True),
)
BY_NAME: dict[str, Condition] = {c.name: c for c in ALL}
if len(BY_NAME) != len(ALL):
    raise RuntimeError("two conditions share a name")


def for_slice(sl) -> Condition | None:
    """The condition a form's slice names (its `name`), or None: an unknown name is never guessed (R13)."""
    return BY_NAME.get(getattr(sl, "name", sl))


def names() -> tuple[str, ...]:
    return tuple(BY_NAME)


def describe() -> str:
    """Every condition a slice may name - for the talker, the research agent (L14.2) and people."""
    out = ["THE CONDITIONS a slice may name (name: what it splits by - its parts)"]
    for c in ALL:
        out.append(f"  {c.name}: {c.label} - {' / '.join(c.groups)}  [{c.dimension}; {c.source}]")
    return "\n".join(out)


# ================================================================================================================
# strategy/COMPONENTS.md: each of its 31 pieces and the condition that covers it (the prompt: "list each and say which
# feature covers it"). "constant" = the same on every setup of an idea, so it cannot split that idea's trades; it is
# a VERSION of the idea instead (likhith.md section D), or a split across ideas (L13.2's pooled view).
# ================================================================================================================

COMPONENTS: dict[int, tuple[str, tuple[str, ...], str]] = {
    1: ("I swept EQUAL highs / lows", ("level_kind", "level_touches"), "the kind taken, and its touches ('3>2')"),
    2: ("I swept a TRENDLINE / wedge / triangle", ("level_kind",), ""),
    3: ("I swept a SESSION high / low", ("level_kind",), ""),
    4: ("I swept an H1 / DAILY swing", ("level_kind", "pair_chart"), "kind 'a day / week swing' and its chart; an "
        "hour swing never starts a setup (his #93)"),
    5: ("How DEEP the sweep went past the level", ("grab_depth",), ""),
    6: ("The break of structure, and how far it went beyond", ("break_size",), ""),
    7: ("A fair value gap is there", (), "constant: every setup of an idea with a gap step has one - 'do I need the "
        "gap?' is a version (an idea without the gap step), not a slice"),
    8: ("How WIDE the gap is", ("gap_size",), ""),
    9: ("How immediate it was - bars from the break to the gap", ("break_to_gap",), "the half known at the order; how "
        "fast price comes BACK is after the order (a fill rule, not a condition)"),
    10: ("The entry is in DISCOUNT inside the move", ("leg_position",), ""),
    11: ("Where the entry sits in TODAY'S ADR range", ("adr_zone",), "and the number adr_share in the feature table"),
    12: ("A strong zone AT the entry", ("zone_at_entry",), ""),
    13: ("A strong zone AS the target", ("zone_at_target",), ""),
    14: ("The DAILY bias agrees with me", ("bias_day", "dial"), ""),
    15: ("The WEEKLY bias agrees with me", ("bias_week", "week_and_day"), ""),
    16: ("A trendline standing in the way", ("line_in_path", "line_against"), "in the path to the target (#69), and "
         "against the entry on the hedge side (#111)"),
    17: ("We are at the 52-week / all-time highs", ("at_ath",), "the REAL all-time high only (his #72)"),
    18: ("ROOM - how far the next liquidity is", ("room", "liq_target"), ""),
    19: ("HEDGE DISTANCE - how far the hedge sits from the entry", ("hedge_size", "reward"), ""),
    20: ("Which SESSION and hour it is", ("session", "new_york", "hour"), "India time, not UTC"),
    21: ("Inside a NEWS window", ("news_near", "news_kind", "news_colour"), "his red + orange, every year 2016-2023 "
         "(the old calendar started in 2018)"),
    22: ("It is the SECOND gap of a two-gap move", (), "constant: which gap the idea takes is its setting "
         "which_gap_of_the_move (first / second / deepest) - a version (his #124), not a slice"),
    23: ("The liquidity was ALREADY taken by a false breakout", (), "constant: the take step keeps only a level not "
         "swept before (his #129, the sweeps' `fresh`) - the other reading is a version"),
    24: ("A broken trendline being RETESTED at my entry", ("line_retest",), ""),
    25: ("A zone that has FLIPPED - old resistance now support", ("zone_flip",), ""),
    26: ("The market is TRENDING, not sideways", ("trend_bias", "trend_days"), ""),
    27: ("My target is the high/low of the session I am trading inside", ("target_kind", "session"), "the target's "
         "kind (a session high / low or not) next to the session; an idea whose target is never a session level has "
         "one part only"),
    28: ("How RECENT the swept level is", ("recent",), "and the number level_age_days in the feature table"),
    29: ("The 1-HOUR bias agrees with me", ("bias_hour", "dial"), ""),
    30: ("There are untaken EQUAL highs / lows on the OTHER side", ("equal_other_side",), ""),
    31: ("WHICH RANK of liquidity I swept", ("level_kind",), "his order Equal > Trendline > Session is the order of "
         "its parts; the number level_rank in the feature table"),
}
for _n, (_title, _conds, _note) in COMPONENTS.items():
    for _c in _conds:
        if _c not in BY_NAME:
            raise RuntimeError(f"COMPONENTS #{_n} names the condition {_c!r}, which does not exist")
