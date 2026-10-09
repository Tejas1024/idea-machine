"""What versions may change: the 31 CHANGEABLE settings of old/strategy_lab/knobs.py, and every other number the old
engine used that is not one of his definitions (plan/likhith.md L6.1 steps 2 and 3).

His fixed rules (what a thing IS) are in machine/rules.py and can never be set here: resolve() refuses them first,
with his words (LEARNED #121).

Each setting says:
    value     what it is today: his newest word wins over the old code (CLAUDE.md order of trust; the management
              values follow plan/work/management-decisions.md, T12)
    who       who chose it: "Tejas #<lesson>" or "Claude"
    status    "his"  - his rule. A version MAY set something else, but only as a labelled comparison: the resolved
                       settings list it in `breaks_his_rules`, and every report must print that line.
              "test" - Claude's number or an open question: a version may change it freely (each version is a TRY).
    source    where the old code had it
    options   the values it may take (None = any value of the same type)
    t12       the row of the management table (plan/work/management-decisions.md), if any
    used_by   the prompt that builds what uses it

Years, lots and cost come from machine/config.py so the two can never disagree.
One version's settings reach the blocks through `with using(changes):` (every block reads settings.get()); caches
built under other settings are told apart by fingerprint() (L7a).
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from machine import config, rules
from machine.guards import SEALED_WORDS


@dataclass(frozen=True)
class Setting:
    value: Any
    who: str
    status: str                    # "his" | "test"
    means: str
    source: str
    options: tuple | None = None
    t12: str = ""
    used_by: str = ""
    aliases: tuple[str, ...] = ()


class SettingError(ValueError):
    """A form or version asked for a setting that does not exist, or a value it may not take."""


def _s(value, who, status, means, source, options=None, t12="", used_by="", aliases=()):
    assert status in ("his", "test"), status
    return Setting(value, who, status, means, source, tuple(options) if options is not None else None, t12, used_by,
                   tuple(aliases))


# ================================================================================== the 31 CHANGEABLE (knobs.py)
_CHANGEABLE = {
    # ---- which entries run, and which may close a position ------------------------------------------------------
    "entries_on": _s(
        ("1",), "Claude (the old engine's shipping list; his entries are all five, #113)", "test",
        "Which entries to trade. 1 = grab + gap (5-min) · M15 / H1 = grab + gap on the 15-min / 1-hour chart · "
        "2 = trendline 3rd touch · 3nt = zone breakout · P1 = triangle false break · Z = zone entry · "
        "ZB = zone break retest · P2 = false break of an obvious level, then its gap · 1v = grab + gap with falsely "
        "broken levels thrown away · W = the wedge.",
        "old/strategy_lab/run.py lab.entries_on",
        options=("1", "M15", "H1", "2", "3nt", "P1", "Z", "ZB", "P2", "1v", "W"), used_by="L7a, L19b"),
    "which_entries_can_close_a_position": _s(
        "his5", "Tejas #113", "his",
        "his5 = any of his five-minute entries may close a leg. type1 = only grab + gap (every number before 20 Sep "
        "2026). all = every entry.",
        "old/strategy_lab/run.py lab.exit_set", options=("type1", "his5", "all"), t12="M10", used_by="L19a",
        aliases=("exit_set",)),
    # ---- when to get in -----------------------------------------------------------------------------------------
    "when_to_enter": _s(
        "wait", "Tejas (strategy.md section 2; #68)", "his",
        "wait = his rule: price must come back into the gap. now = enter the moment the gap is finished. late = wait, "
        "and if price never came back, take it at the end of the wait (his #118: gaps that never come back are "
        "missed trades - 'we will build something').",
        "old/strategy_lab/run.py lab.when_to_enter", options=("wait", "now", "late"), used_by="L7a"),
    # ---- which gap of the move ----------------------------------------------------------------------------------
    "which_gap_of_the_move": _s(
        "first", "Tejas #3, #61, #115, #124; strategy.md section 2 (row 8); his L4 answer (inbox 2026-09-28-0622)",
        "his",
        "When one move leaves two gaps: first = the one nearer the hedge (the lower one for a buy) - his default. "
        "second = his #124 exception, only when the trend is 'very strong' (a word still open, four-boxes B2-03; "
        "then the_second_gap_needs_the_day decides). deepest = the biggest gap of the move. The old code shipped "
        "'second' + 'strong'; his newest words refuse a second gap until 'very strong' is settled. L19b built "
        "'second' in the compiler (the step gap_of_the_move): the order waits at the first gap; when the second gap "
        "is known and the agreement holds then (and the first was not reached), a newer order at the second gap "
        "takes its place.",
        "old/ml/agent/setup_engine.py GAP_PICK = 'second'", options=("first", "second", "deepest"),
        used_by="L7b, L19b",
        aliases=("GAP_PICK", "gap_pick")),
    "a_zone_break_must_go_through_the_zone": _s(
        "veto", "Tejas #119 (his T06)", "his",
        "A zone break goes THROUGH the zone, not out the side price came from. veto = refuse a break that does not "
        "(and a breakout whose cover would land on the wrong side of its entry).",
        "old/ml/agent/entries.py T3_THROUGH = 'veto'", options=("veto", True, False), used_by="L6b.2",
        aliases=("T3_THROUGH",)),
    "the_zone_breakout_checks_the_move_is_not_finished": _s(
        True, "Tejas #119 (his T05)", "his",
        "The zone breakout must not enter a move that already reached its target, and a next target must really "
        "exist.", "old/ml/agent/entries.py T3_T05 = True", options=(True, False), used_by="L6b.2",
        aliases=("T3_T05",)),
    "the_wedge_needs_the_hour_and_day": _s(
        True, "Tejas (strategy-code.md section 15: 'Entry Type 2 (wedge) DOES require HTF agreement')", "his",
        "The wedge entry only when the hour and the day agree.", "old/ml/agent/entries.py WEDGE_NEED_HD = True",
        options=(True, False), used_by="L6b.1", aliases=("WEDGE_NEED_HD",)),
    "the_most_recent_liquidity_comes_first": _s(
        True, "Tejas #130", "his",
        "Among levels of the same kind, the most recent one is picked before an older one that sits a little "
        "nearer: 'always it will look for a new session highs and lows'.",
        "old/ml/agent/levels.py RECENT_FIRST = True", options=(True, False), used_by="L6.2",
        aliases=("RECENT_FIRST",)),
    "a_level_swept_once_is_finished": _s(
        "swept", "Tejas #129", "his",
        "swept = any poke finishes a high or low: it is not liquidity again. false_break = only a poke that closed "
        "straight back. False = the old engine before 21 Sep 2026. Zones follow their own rule (rules.py).",
        "old/ml/agent/entries.py LIQ_ONCE = 'swept'", options=("swept", "false_break", False), used_by="L6.2",
        aliases=("LIQ_ONCE",)),
    "how_long_a_swept_level_stays_dead": _s(
        576, "Tejas #130 ('two days is correct')", "his",
        "5-minute candles a swept level stays finished: 288 = one day, 576 = two days (his), 864 = three days (his "
        "other number), 2016 = a week (not his).",
        "old/ml/agent/entries.py LIQ_ONCE_BARS = 576", options=(288, 576, 864, 2016), used_by="L6.2",
        aliases=("LIQ_ONCE_BARS", "P2_CONSUMED_BARS")),
    "the_second_gap_needs_the_day": _s(
        "strong", "Tejas #124; 'very strong' read by Claude (four-boxes B2-03, still open)", "his",
        "Only when a version takes the second gap: strong = the hour AND the day with the trade and the week not "
        "against it, else his first gap. all = the hour, the day AND the week all with the trade, else the first gap "
        "(the other reading of 'very strong', four-boxes B2-03; L19b). first = the day must agree, else the first "
        "gap. True = the day must agree, else skip. False = no condition.",
        "old/ml/agent/setup_engine.py TWO_GAP_NEEDS_HTF = 'strong'", options=(True, False, "first", "strong", "all"),
        used_by="L7b, L19b", aliases=("TWO_GAP_NEEDS_HTF",)),
    # ---- covering: when the other side opens --------------------------------------------------------------------
    "what_the_leg_left_over_aims_at_after_a_cover": _s(
        "his124", "Tejas #124, #125 ('Yes')", "his",
        "his124 = at least past the 1:4 no-close zone, then the next REAL liquidity. setup = whatever target the "
        "closing setup carried.", "old/ml/agent/manager.py DEFAULTS leftover_target", options=("setup", "his124"),
        t12="M16", used_by="L19a", aliases=("leftover_target",)),
    "the_cover_follows_the_trendline": _s(
        "line", "Tejas #69 (T12 M27: his words beat the old default 'off')", "his",
        "On a trendline entry the hedge price follows the line (only tighter, only the first cover).",
        "old/ml/agent/manager.py DEFAULTS trail_cover = 'off'", options=("off", "line"), t12="M27", used_by="L19a",
        aliases=("trail_cover",)),
    "give_it_room_before_covering": _s(
        "hd", "Tejas #111", "his",
        "hd = if the hour AND the day are both with the trade, the hedge waits at the next liquidity. off = always "
        "hedge at the hedge point.", "old/ml/agent/manager.py DEFAULTS room_tf", options=("off", "d", "hd"),
        t12="M06", used_by="L19a", aliases=("room_tf",)),
    "how_much_room": _s(
        1.0, "Claude", "test", "How far 'give it room' reaches when no liquidity is there, in days of normal range.",
        "old/ml/agent/manager.py DEFAULTS room_adr", t12="M07", used_by="L19a", aliases=("room_adr",)),
    "room_at_the_all_time_high": _s(
        True, "Tejas (strategy.md section 12)", "his", "A buy at the all-time high gets room, not an immediate hedge.",
        "old/ml/agent/manager.py DEFAULTS ath_room", options=(True, False), t12="M08", used_by="L19a",
        aliases=("ath_room",)),
    "how_many_covers_allowed": _s(
        None, "Claude (his spec: max_hedge_legs = TUNE, no number anywhere; None = no cap = what L19a ran - the old 4 "
        "only fed the escape, which is off)", "test",
        "How many times the other side may open in one trade. None = no cap (version 1). N = after N hedges, the next "
        "time price reaches the hedge price the open leg CLOSES there - a real stop (Claude's reading of 'max legs = "
        "capped': strategy.md section 11 gives no rule for what happens at the cap; four-boxes B3-12: 2 ... 10).",
        "old/ml/agent/manager.py DEFAULTS max_hedges = 4", options=(None, *range(1, 11)), t12="M21",
        used_by="L19a, L19c", aliases=("max_hedges",)),
    # ---- closing ------------------------------------------------------------------------------------------------
    "how_far_before_any_leg_may_close": _s(
        4.0, "Tejas #108", "his",
        "After a hedge, nothing closes until price is 4 times the hedge distance from the current entry (1:4).",
        "old/ml/agent/manager.py DEFAULTS hedge_min_rr", t12="M12", used_by="L19a", aliases=("hedge_min_rr",)),
    "which_leg_may_close_while_covered": _s(
        "any_rr", "Tejas #134 (newer than #105 and #112)", "his",
        "any_rr = the leg the new setup points against closes, even at a loss, but only 1:4 from the current entry. "
        "profit = only the leg in profit (#105, the old default). any = no distance rule (not his).",
        "old/ml/agent/manager.py DEFAULTS hedged_close = 'profit'", options=("profit", "any_rr", "any"), t12="M11",
        used_by="L19a", aliases=("hedged_close",)),
    "does_the_whole_trade_decide_a_close": _s(
        "trade", "Tejas #112", "his", "trade = a red leg may close if the WHOLE trade is green. leg = each leg alone.",
        "old/ml/agent/manager.py DEFAULTS hedged_whole", options=("trade", "leg"), t12="M17", used_by="L19a",
        aliases=("hedged_whole",)),
    "close_the_losing_leg_when_a_trend_runs": _s(
        False, "Tejas #19, #51 (parked: 'that will come later'); the trigger is his strategy-code.md section 18 'GRID "
        "ESCAPE'", "test",
        "Trend escape (strategy-code.md section 18: 'if HTF trending + trendline + proper liquidity taken -> close the "
        "LOSING leg at a loss, let the hedge run'). OFF = version 1 (parked until measured). True = while hedged, a full "
        "setup of his the trend's way (proper liquidity taken) closes the leg it points against EVEN INSIDE the 1:4 "
        "no-close zone - only when the hour AND the day are both that way (HTF trending) and a trendline of his on the "
        "5-minute chart runs that way (Claude's reading of '+ trendline'); the leg left is that setup's trade (M15).",
        "old/ml/agent/manager.py DEFAULTS escape", options=(True, False), t12="M22", used_by="L19c",
        aliases=("escape",)),
    "what_the_leg_left_over_aims_at": _s(
        "old", "Tejas #22 (6 Sep)", "his",
        "old = the leg becomes a new entry at the booking price and takes that target. entry / target = the two "
        "readings that locked trades open for years.", "old/ml/agent/manager.py DEFAULTS leftover_rule",
        options=("old", "entry", "target"), t12="M15", used_by="L19a", aliases=("leftover_rule",)),
    "a_floor_under_the_reward": _s(
        "none", "Tejas #53", "his", "No floor: no 'close everything N ranges away'.",
        "old/ml/agent/manager.py DEFAULTS rr_floor", options=("none", "first", "current"), t12="M20", used_by="L19a",
        aliases=("rr_floor",)),
    "where_profit_is_measured_from": _s(
        "current", "Tejas #134 ('1:4 from the current entry'; T12 M12)", "his",
        "current = from the leg's last booking (the current entry). opened = from where it first opened (the old "
        "default).", "old/ml/agent/manager.py DEFAULTS profit_ref = 'opened'", options=("opened", "current"),
        t12="M12", used_by="L19a", aliases=("profit_ref",)),
    # ---- targets ------------------------------------------------------------------------------------------------
    "trail_the_target": _s(
        "hd", "Tejas #82, #133", "his",
        "hd = with the hour AND the day on side, hold past the first target unless an opposite setup forms; "
        "against -> exit at the first target.", "old/ml/agent/manager.py DEFAULTS trail", options=("off", "d", "hd"),
        t12="M23", used_by="L19a", aliases=("trail",)),
    "the_target_moves": _s(
        True, "Tejas #20", "his", "The target keeps seeking the next liquidity; it is not a fixed number.",
        "old/ml/agent/manager.py DEFAULTS moving_target", options=(True, False), t12="M25", used_by="L19a",
        aliases=("moving_target",)),
    # ---- news ---------------------------------------------------------------------------------------------------
    "what_news_does": _s(
        "profit_close", "Tejas #110", "his",
        "In profit -> close; at break-even or any loss -> hedge. News never closes a losing trade.",
        "old/ml/agent/manager.py DEFAULTS news", options=("off", "profit_close", "hedge_or_close"), t12="M28",
        used_by="L19a", aliases=("news",)),
    # ---- money --------------------------------------------------------------------------------------------------
    "lot_size": _s(
        config.LOT_STEP, "Tejas #116", "his", "0.01 lot = one ounce. Lots only 0.01, 0.02, 0.03 ...",
        "old/ml/agent/manager.py DEFAULTS lot", t12="M05", used_by="L3, L20", aliases=("lot",)),
    "cost_per_position": _s(
        float(config.COST_INR_PER_POSITION), "Tejas #54, #115", "his",
        "Rupees per position per 0.01 lot, ALL-IN (Rs 20 in + Rs 20 out, includes the spread). No swap.",
        "old/ml/agent/manager.py DEFAULTS cost_rs_001", t12="M36", used_by="L2, L3", aliases=("cost_rs_001",)),
    "how_many_trades_at_once": _s(
        "one_across_everything", "Tejas #55", "his",
        "one_across_everything = one trade at a time in the whole account. one_per_entry = one per entry type (a "
        "test version he has NOT approved, #118).", "old/strategy_lab/run.py lab.at_once",
        options=("one_across_everything", "one_per_entry"), t12="M18", used_by="L3", aliases=("at_once",)),
    "years": _s(
        (config.TUNE_YEARS[0], config.CHECK_YEARS[-1]), "Tejas #96, #104 (2024-2026 sealed)", "his",
        "The years to test. 2016-2020 to choose, 2021-2023 hidden and opened once per idea family (L5).",
        "old/strategy_lab/run.py lab.years", used_by="L3, L5"),
}

CHANGEABLE: Mapping[str, Setting] = MappingProxyType(_CHANGEABLE)


# ================================================================== every other number of the old engine (step 3)
# (name, value, who, status, means, source, used_by). Retired constants and plumbing are NOT here: they are listed
# with the reason in plan/work/rules-missing.md.
def _table(rows):
    return {r[0]: _s(r[1], r[2], r[3], r[4], r[5], used_by=r[6], aliases=r[7] if len(r) > 7 else ())
            for r in rows}


_SE, _LV, _EN, _MG = ("old/ml/agent/setup_engine.py ", "old/ml/agent/levels.py ", "old/ml/agent/entries.py ",
                      "old/ml/agent/manager.py ")
_MORE = _table([
    # ---- the grab + gap setup (setup_engine.py) -----------------------------------------------------------------
    ("state_timeout_bars", 30, "Claude (his spec said 12, TUNE)", "test",
     "Candles allowed between the sweep and the break, and again between the break and the gap.",
     _SE + "STATE_TIMEOUT = 30", "L7b", ("STATE_TIMEOUT", "state_timeout")),
    ("order_block_max_age_bars", 40, "Claude", "test",
     "How many candles before the sweep his order block may sit.", _SE + "OB_BACK = 40", "L6.1, L7b", ("OB_BACK",)),
    ("second_setup_guard", "filled", "Claude (reading his #3 and #55)", "test",
     "Which earlier setup blocks a second one in the same direction: filled = only one that became a trade. "
     "live = any unresolved one. off = none.", _SE + "SECOND_GUARD = 'filled'", "L7b", ("SECOND_GUARD",)),
    ("second_setup_guard_bars", 10, "Claude (tied to the retired 10-candle window)", "test",
     "Candles a setup may still be filled in before it stops blocking the next one.",
     _SE + "SECOND_GUARD_WINDOW = FVG_RETURN_BARS", "L7b", ("SECOND_GUARD_WINDOW",)),
    ("first_visit_max_bars", None, "Tejas #68 ('it's not about the time')", "his",
     "How long the machine waits for the first visit. None = no limit (his rule). The old manager capped it at 288 "
     "(one day, Claude's). A version may set a limit: the idea's own 'max wait'.",
     _SE + "FIRST_VISIT_MAX = 288", "L7a", ("FIRST_VISIT_MAX",)),
    ("near_zone_adr", 0.08, "Claude", "test", "'The entry is AT a zone' = within 0.08 of a day's range. The same for "
     "his ADR zones (strategy.md section 7: 'red + green zones'; 'at discount (near the green zone)'): price at, inside "
     "or past the zone, or this close short of it (machine/blocks/premium_discount.py).",
     _SE + "NEAR_ZONE_FRAC = 0.08", "L6b.2, L6b.3", ("NEAR_ZONE_FRAC",)),
    ("adr_edge", 0.85, "Claude", "test", "Inside the outer 15% of today's ADR band = 'stretched for the day' (words "
     "only).", _SE + "ADR_EDGE_FRAC = 0.85", "L6b.3", ("ADR_EDGE_FRAC",)),
    ("min_hedge_distance_adr", 0.02, "Claude", "test",
     "A hedge closer than 0.02 of a day's range is a rounding error, not a hedge: the setup is dropped.",
     _SE + "MIN_R_FRAC = 0.02", "L7b", ("MIN_R_FRAC",)),
    ("near_all_time_high", 0.985, "Claude", "test", "Within 1.5% of the all-time high = 'at the highs'.",
     _SE + "ATH_NEAR_FRAC = 0.985 (= levels.ATH_NEAR_FRAC)", "L6b.3", ("ATH_NEAR_FRAC",)),
    ("premium_share", 0.85, "Claude (his #5, #76 give no number)", "test",
     "Buying above this share of today's ADR range is buying at premium (a wait unless the hour AND the day agree).",
     _SE + "PREMIUM_HARD = 0.85", "L6b.3", ("PREMIUM_HARD",)),
    ("discount_share", 0.15, "Claude (his #5, #76 give no number)", "test",
     "Selling below this share of today's ADR range is selling at discount.", _SE + "DISCOUNT_HARD = 0.15", "L6b.3",
     ("DISCOUNT_HARD",)),
    ("big_gap_usd", 8.0, "his old spec (strategy-code.md section 2, TUNE); not in strategy.md (T1)", "test",
     "A gap bigger than $8 is taken only when the higher timeframe agrees.", _SE + "BIG_GAP_USD = 8.0", "L7b",
     ("BIG_GAP_USD", "big_gap")),
    ("adr_factor_share", 0.20, "Claude", "test",
     "The ADR 'confluence factor': a buy in the bottom 20% of today's range, a sell in the top 20%.",
     _SE + "ADR_FACTOR_FRAC = 0.20", "L13b", ("ADR_FACTOR_FRAC",)),
    ("confluence_minimum", "OFF", "Tejas #15, #16, #17, #51, #84, #97 (strategy.md section 8: 'never a minimum count')",
     "his", "More confluence makes a trade better but is never a minimum count. OFF = no minimum.",
     _SE + "detect(confluence='OFF'), CONFLUENCE_READINGS", "L7b", ("confluence",)),
    ("line_in_path_min_reward", 1.0, "Claude (#69: A 0.63 skip, D 1.42 and B 2.36 take)", "test",
     "With the hour and the day against and a trendline before the target: under this many hedge distances of "
     "reward = skip ('50-50'), else the line is the target.", _SE + "PATH_MIN_R = 1.0", "L7b", ("PATH_MIN_R",)),
    ("line_taken_through_adr", 0.005, "Claude (#69: 'a trend line fake out is also a liquidity')", "test",
     "A candle more than this share of a day's range through a line since its first touch has taken it.",
     _SE + "LINE_THROUGH_ADR = 0.005", "L6b.1", ("LINE_THROUGH_ADR",)),
    ("htf_history_days", 1100, "Claude", "test",
     "Days of history the week bias needs before it can say anything (29 weekly candles cannot show structure).",
     _SE + "MIN_HTF_DAYS = 1100", "L6b.3", ("MIN_HTF_DAYS",)),
    ("fvg_target_reset", True, "Claude (section 64l)", "test",
     "The target is found again from the fill price when an order fills.", _MG + "FVG_TARGET_RESET = True", "L7b",
     ("FVG_TARGET_RESET",)),
    # ---- levels (levels.py) -------------------------------------------------------------------------------------
    ("line_touch_adr", 0.025, "his old spec (line_tol $0.50 at ADR $20, TUNE)", "test",
     "A swing this close to a line (share of a day's range) is a touch.", _LV + "LINE_TOL_FRAC = 0.025", "L6b.1",
     ("LINE_TOL_FRAC", "line_tol")),
    ("zone_bin_adr", 0.032, "his old spec (zone_bin $0.75 at ADR $20, TUNE) = the old engine's 0.0375 of its UTC-day "
     "ruler; the new ruler is 1.17x it (2016-2020), so 0.032 keeps the same dollars (L6b.2, like L6b.1's trendline "
     "0.25 -> 0.21). Read off his marked charts: at 0.0375 of the new ruler his Q86 D main zone (#86, 31 Mar 2016) is "
     "not drawn; at 0.032 all four quiet zones he said yes to in 2016-2020 (Q86 D, Q97 A, Q79 B, Q79 D) are the "
     "machine's main zone. A KNIFE EDGE: the price steps start at the window's lowest body, so a 3% change of this "
     "number swaps which of his 9 zones of 2016-2020 come out (0.032: 8 of 9, his Q97 B zone only as tier 3; 0.0375: "
     "8 of 9, Q86 D lost; only 0.033 gets 9, its neighbours do not - not chosen: plan/work/L6b2-notes.md)", "test",
     "Price step of the time-at-price count, as a share of a day's range.", _LV + "ZONE_BIN_FRAC = 0.0375", "L6b.2",
     ("ZONE_BIN_FRAC", "zone_bin")),
    ("zone_strong_top_percent", 20, "his old spec (zone_strong_pct, TUNE)", "test",
     "The top 20% of the time-at-price count is 'strong'.", _LV + "ZONE_STRONG_PCT = 20", "L6b.2",
     ("ZONE_STRONG_PCT", "zone_strong_pct")),
    ("zone_min_bins", 3, "his old spec (zone_min_bins, TUNE)", "test", "A band thinner than 3 steps is not a zone.",
     _LV + "ZONE_MIN_BINS = 3", "L6b.2", ("ZONE_MIN_BINS",)),
    ("zone_flip_confirm_bars", 12, "Claude", "test",
     "A close beyond a zone must still be beyond it this many candles later before the zone is called flipped.",
     _LV + "ZONE_FLIP_CONFIRM = 12", "L6b.2", ("ZONE_FLIP_CONFIRM",)),
    ("zone_second_priority_calm", 1.00, "Claude (his #81 'second priority' gave no number)", "test",
     "Calm above 0.74 and up to this = his second-priority zone.", _LV + "ZONE_CALM_2ND = 1.00", "L6b.2",
     ("ZONE_CALM_2ND",)),
    ("zone_tiers", "tiered", "Tejas #81", "his", "His main (quiet) zone first, then his second choice.",
     _LV + "ZONE_MODE = 'tiered'", "L6b.2", ("ZONE_MODE",)),
    ("zone_hour_chart_rank", 2, "Tejas (strategy.md section 3: '1h stronger')", "his",
     "The order of his zones: today = 0, this week = 1, the hour chart = 2 (strongest).",
     _MG + "Z_HOUR_CHART = 2", "L6b.2", ("Z_HOUR_CHART",)),
    ("equal_levels_max_apart_bars", 2880, "Claude (his #89: 'two days one week' - no limit given)", "test",
     "On 5-minute candles, how far back a pair of equal highs / lows is looked for (two weeks). On the hour and day "
     "charts: no limit (his #89).", _LV + "EQ_LOOKBACK_HIS = 2880, EQ_LOOKBACK_HTF = None", "L6.2",
     ("EQ_LOOKBACK_HIS", "EQ_LOOKBACK_HTF")),
    ("equal_levels_min_apart_bars", 0, "Claude (his T3 answers: 'the lows are not proper lows cause they are too near "
     "to each other', inbox 2026-09-28-1151 - no number; question queued, plan/work/question-queue/)", "test",
     "The two touches of an equal high / low at least this many candles apart; 0 = no minimum (his written rule: 'any "
     "time apart'). His T3 answers split at 7 (every 'no' 3-6 apart) but his #24, #25 (3 and 6 apart) were 'yes' and "
     "#33 (4) 'yes, partly no' - so 7 is a version to compare until he answers.", "new (L6.2)", "L6.2, L7b"),
    ("same_level_usd", 0.06, "Claude: his 'really equal' ($0.06, rules.py what_equal_highs_and_lows_are) - DECISIONS "
     "D3 'reuses his equal-highs tolerance'; the old engine used 0.05 of a day (about $0.50), which made 48% of equal-"
     "level sweeps 'already swept' in 2016-2020 (L6.2)", "test",
     "A level this close (dollars) to a level of the same side already swept is the SAME level - finished (#129).",
     _EN + "liq_once(): abs(bp - lv) <= 0.05 * a", "L6.2", ("consumed_tol_frac",)),
    ("grab_ran_too_far_adr", 0.25, "Claude (his T3 no-reasons 'it moves so far', 'too big', 'too far it moved' - inbox "
     "2026-09-28-1101, -1130, -1151 - give no number: on round 3 every yes ran <= 0.22 of a day, every no >= 0.27; his "
     "round-1 yes #12 (0.27) and #15 (0.36) sit above the line)", "test",
     "The do-not-enter rule grab_ran_too_far: the grab may run at most this share of a day's range past the level it "
     "took (the level to the hedge point, when the setup is made); more = no trade. None = no line. Idea #1 tries "
     "0.20 / 0.25 / 0.30 as versions. The old engine ended a take at 0.5 (its 'room' line, never ported).",
     _SE + "room = 0.5 * adr (its section 7)", "L7b", ("room",)),
    ("level_life_bars", 2880, "Claude", "test", "5-minute candles a level is tracked before it is dropped (10 days).",
     _LV + "MAX_LIFE = 2880", "L6.2", ("MAX_LIFE",)),
    ("line_far_adr", 2.0, "Claude", "test",
     "A trendline whose projection has drifted this many days of range from price is dead.",
     _LV + "LINE_FAR_ADR = 2.0", "L6b.1", ("LINE_FAR_ADR",)),
    ("recent_swing_bars", 288, "Claude (his #91 'recent' gave no number)", "test",
     "'Recent' 5-minute swing = from the last day (288 candles).", _LV + "RECENT_SWING_BARS = 288", "L6.2",
     ("RECENT_SWING_BARS",)),
    ("all_time_high_before_2016", 1920.94, "Claude: gold's spot record of 6 Sep 2011, quoted $1,920.85 - $1,921.17 by "
     "source (Investing.com, 24 Jul 2020, 'record high of $1,920.85'; AP / Spokesman-Review, 24 Jul 2020, '$1,921.17'; "
     "Gulf News, 'on September 6, 2011 ... $1,921'); 1,920.94 sits inside that range - the broker's own 2011 price is "
     "not in the data (it starts in 2016). Comex futures printed $1,923.70 (not spot)", "test",
     "The data starts in 2016, so the running all-time high starts from the 2011 record (never reached again until "
     "July 2020).",
     _LV + "ATH_BEFORE_DATA = 1920.94", "L6b.3", ("ATH_BEFORE_DATA",)),
    ("bias_method", "trade", "Tejas #70, #71, #72, #73, #74 ('perfect')", "his",
     "The daily bias = his 5-minute entry read on the daily chart: a daily level taken -> a break -> a gap -> price "
     "comes BACK to the gap. Neutral until then. The same on the hour (#69).",
     _LV + "BIAS_METHOD = 'trade', BIAS_TRADE_KW = dict(same_detectors=True, at_target='hold')", "L6b.3",
     ("BIAS_METHOD", "BIAS_TRADE_KW")),
    ("all_time_high_is_the_real_one", "all", "Tejas #72", "his",
     "'Only all time high means only all time high': the REAL all-time high, not a 52-week high.",
     _SE + "ATH_WINDOW = 'all'", "L6b.3", ("ATH_WINDOW",)),
    # ---- patterns and his other entries (entries.py) ------------------------------------------------------------
    ("wedge_touch", 3, "Tejas (strategy.md section 9: 'Trendline 3rd/4th touch')", "his",
     "The wedge / trendline entry is on the 3rd touch, up to the 4th.", _EN + "WEDGE_TOUCH = 3", "L6b.1",
     ("WEDGE_TOUCH", "wedge_entry_touch")),
    ("triangle_min_swings", 2, "his old spec (triangle_min_swings, TUNE)", "test",
     "Swings per triangle line.", _EN + "TRI_MIN_SWINGS = 2", "L6b.1", ("TRI_MIN_SWINGS",)),
    ("fakeout_lookback_bars", 20, "his old spec (fakeout_lookback, TUNE)", "test",
     "Candles within which price must come back inside for a false breakout.", _EN + "FAKEOUT_LOOKBACK = 20",
     "L6b.1", ("FAKEOUT_LOOKBACK",)),
    ("line_break_tolerance", 2.0, "Claude", "test",
     "'A break back beyond the line', in units of the touch tolerance.", _EN + "BREAK_TOL_MULT = 2.0", "L6b.1",
     ("BREAK_TOL_MULT",)),
    ("big_move_adr", None, "open: 'big move' is a word still open (four-boxes B2-02). The old engine's 1.0 (its "
     "75th percentile, Claude's) is one candidate reading; L6.1 ported it as the value, L6b.1 made it wait (R13)",
     "test",
     "'After a big move' = the day before the pattern moved at least this many days of range. None = no rule yet: a "
     "triangle step waits for his answer, unless the form names the number (a reading).",
     _EN + "IMPULSE_ADR = 1.0", "L6b.1", ("IMPULSE_ADR",)),
    ("big_move_bars", 288, "Claude", "test", "... measured over the 288 candles before the pattern.",
     _EN + "IMPULSE_BARS = 288", "L6b.1", ("IMPULSE_BARS",)),
    ("triangle_max_height_adr", 2.0, "Claude", "test", "A 'triangle' taller than two days of range is not a pattern.",
     _EN + "TRI_MAX_ADR = 2.0", "L6b.1", ("TRI_MAX_ADR",)),
    ("triangle_inside_share", 0.0, "Claude (off)", "test", "Share of candles that must stay inside (0 = off).",
     _EN + "TRI_INSIDE_FRAC = 0.0", "L6b.1", ("TRI_INSIDE_FRAC",)),
    ("triangle_min_touch", 0, "Claude (off)", "test", "Touches each line needs beyond its two swings (0 = off).",
     _EN + "TRI_MIN_TOUCH = 0", "L6b.1", ("TRI_MIN_TOUCH",)),
    ("move_finished_share", 0.10, "Claude (his #46, #47, #119 give no number)", "test",
     "The first target counts as taken when price came within 10% of the way to it.",
     _EN + "T3_TOOK_FRAC = 0.10 (= detect()'s own 10%)", "L7b", ("T3_TOOK_FRAC",)),
    ("zone_side_lookback_bars", 576, "Claude", "test",
     "Two days back to find which side of a zone price came from.", _EN + "T3_FROM_BARS = 576, ZB_FROM_BARS = 576",
     "L6b.2", ("T3_FROM_BARS", "ZB_FROM_BARS")),
    ("zone_return_bars", 576, "Claude", "test", "Two days for price to come back and test a flipped zone.",
     _EN + "ZONE_RETURN_BARS = 576", "L6b.2", ("ZONE_RETURN_BARS",)),
    ("liquidity_taken_bars", 96, "Claude", "test", "'Relevant liquidity already taken' = within 96 candles (8 h).",
     _EN + "LIQ_TAKEN_BARS = 96", "L6b.2", ("LIQ_TAKEN_BARS",)),
    ("hammer_beats_the_higher_timeframe", True, "Tejas #115", "his",
     "A hammer grab may pass the hour + day gate: 'even if the higher time frame is not ... aligning you can enter'.",
     _EN + "HAMMER_OVERRIDES_HTF = True", "L6b.3", ("HAMMER_OVERRIDES_HTF",)),
    ("trendline_min_steepness_adr", None, "open: 'steep' is a word still open (four-boxes B2-01; #127 'the steep of "
     "the line decides that'; queued: plan/work/question-queue/B2-01-steep.md)", "test",
     "How steep a line must be for the touch ENTRY, in days of range the line travels in one day. None = his word is "
     "open: the step trendline_touch waits for his answer, unless the form names the number (a reading).",
     _EN + "TL_MIN_SLOPE_ADR = None", "L6b.1", ("TL_MIN_SLOPE_ADR",)),
    ("trendline_needs_trend", True, "Tejas #116 ('There should be clear uptrend or low trend visible while you are "
     "drawing the trend line. If you want to enter at the touch'); the TEST is Claude's (lower highs AND lower lows "
     "between the first and the last touch - his B2-01 pictures were drawn with it). L6.1 ported the old engine's None",
     "his", "True = the touch entry needs a clear trend the line's way between its first and last touch (#116).",
     _EN + "TL_NEED_TREND = None", "L6b.1", ("TL_NEED_TREND",)),
    ("zone_entry_liquidity_adr", 0.50, "Claude (#86: the widest he was shown was 0.45)", "test",
     "The zone entry's obvious liquidity may sit up to this share of a day's range beyond the zone.",
     _EN + "ZONE_LIQ_ADR = 0.50", "L6b.2", ("ZONE_LIQ_ADR",)),
    ("obvious_liquidity_kinds", ("session_high", "session_low", "trendline_res", "trendline_sup", "equal_high",
                                 "equal_low"), "Tejas #84, #86", "his",
     "His obvious liquidity: session highs / lows, trendlines, equal highs / lows.",
     _EN + "OBVIOUS_KINDS (= ZONE_LIQ_KINDS = P2_KINDS)", "L6.2", ("OBVIOUS_KINDS", "ZONE_LIQ_KINDS", "P2_KINDS")),
    ("unfilled_gap_is_a_magnet", True, "Tejas #87", "his",
     "An unfilled gap on the hedge side -> skip, unless recent obvious liquidity waits on the target side; checked "
     "again when the order fills.", _EN + "ZONE_GAP_RULE = True, ZONE_GAP_WATCH = True", "L7b",
     ("ZONE_GAP_RULE", "ZONE_GAP_WATCH")),
    ("unfilled_gap_lookback_bars", 576, "Claude", "test", "Two days back for unfilled gaps.",
     _EN + "ZONE_GAP_LOOKBACK = 576", "L7b", ("ZONE_GAP_LOOKBACK",)),
    ("unfilled_gap_beyond_adr", 1.0, "Claude", "test", "How far past the hedge an unfilled gap still counts.",
     _EN + "ZONE_GAP_BEYOND_ADR = 1.0", "L7b", ("ZONE_GAP_BEYOND_ADR",)),
    ("unfilled_gap_min_adr", 0.01, "Claude (#62: a tiny gap is still a gap)", "test",
     "The smallest gap the unfilled-gap check looks at (share of a day's range).", _EN + "ZONE_GAP_MIN_ADR = 0.01",
     "L7b", ("ZONE_GAP_MIN_ADR", "ZB_INNER_MIN_ADR")),
    ("open_gap_min_adr", 0.03, "Claude", "test",
     "A jump this big from one candle's close to the next open = his 'gap up / gap down'.",
     _EN + "OPEN_GAP_MIN_ADR = 0.03", "L7b", ("OPEN_GAP_MIN_ADR",)),
    ("zone_break_no_distance_rule", 0.0, "Tejas #98 ('there is no particular distance')", "his",
     "How far price must move away before the retest counts: 0 = no rule.", _EN + "ZB_EXT_ZONE = 0.0", "L6b.2",
     ("ZB_EXT_ZONE",)),
    ("zone_break_move_bars", 288, "Claude", "test", "One day for the move away after a zone break.",
     _EN + "ZB_EXT_BARS = 288", "L6b.2", ("ZB_EXT_BARS",)),
    ("zone_break_is_through", True, "Tejas #98 (his T06, #119)", "his",
     "A zone break goes through the zone, not out the side it came from.", _EN + "ZB_THROUGH = True", "L6b.2",
     ("ZB_THROUGH",)),
    ("zone_break_no_reentry", False, "Claude (off: his own named take fails it)", "test",
     "Price may not touch the band again until the move away is complete.", _EN + "ZB_NO_REENTRY = False", "L6b.2",
     ("ZB_NO_REENTRY",)),
    ("zone_break_needs_hour_and_day", True, "Tejas #97, #98", "his",
     "The zone break retest needs the hour AND the day with it.", _EN + "ZB_NEED_HD = True", "L6b.2",
     ("ZB_NEED_HD",)),
    ("zone_break_hedge_at_ath", True, "Tejas #97, #98", "his",
     "At the all-time high the hedge goes at the next real strong zone, not at the broken one.",
     _EN + "ZB_ATH_HEDGE = True", "L6b.2", ("ZB_ATH_HEDGE",)),
    ("zone_break_ath_max_hedge_adr", 1.0, "Claude ('should not be so much loss' gave no number)", "test",
     "That next zone may be at most one day's range away.", _EN + "ZB_ATH_MAX_R_ADR = 1.0", "L6b.2",
     ("ZB_ATH_MAX_R_ADR",)),
    ("zone_break_needs_obvious_target", True, "Tejas #98", "his",
     "'There should be liquidity there where I can target.'", _EN + "ZB_TARGET_OBVIOUS = True", "L6b.2",
     ("ZB_TARGET_OBVIOUS",)),
    ("zone_break_inner_liquidity", "gap", "Tejas #98, #99", "his",
     "Liquidity inside the zone: wait for it to be taken, then the 5-minute gap (hedge just past it).",
     _EN + "ZB_INNER_LIQ = 'gap', ZB_INNER_NEED_GAP = True", "L6b.2", ("ZB_INNER_LIQ", "ZB_INNER_NEED_GAP")),
    ("zone_break_inner_gap_bars", 36, "Claude", "test", "3 hours for that 5-minute gap to form after the sweep.",
     _EN + "ZB_INNER_GAP_BARS = 36", "L6b.2", ("ZB_INNER_GAP_BARS",)),
    ("zone_break_opposite_setup_veto", True, "Tejas #98 (Q98 B)", "his",
     "An opposite 5-minute setup on the way back makes the retest a target, not an entry.",
     _EN + "ZB_OPP_VETO = True", "L6b.2", ("ZB_OPP_VETO",)),
    ("zone_break_retest_liquidity", "target", "Tejas #98 (Claude's reading of 'it shouldn't take the obvious "
     "liquidity')", "his", "The retest must not have taken the obvious liquidity on the target side.",
     _EN + "ZB_RETEST_LIQ = 'target'", "L6b.2", ("ZB_RETEST_LIQ",)),
    ("false_break_close_back_bars", 12, "Claude", "test", "One hour for a false break to be given up.",
     _EN + "P2_CLOSE_BACK = 12", "L6b.1", ("P2_CLOSE_BACK",)),
    ("false_break_needs_hour_and_day", True, "Tejas (strategy.md section 4: 'Zones + HTF spot these')", "his",
     "The hour AND the day must agree with the way back.", _EN + "P2_NEED_HD = True", "L6b.1", ("P2_NEED_HD",)),
    ("wedge_held_line_no_close", True, "Tejas (strategy.md section 4: the wedge line holds)", "his",
     "No CLOSE beyond the held line while the wedge lives.", _EN + "WEDGE_HOLD_CLOSES = True", "L6b.1",
     ("WEDGE_HOLD_CLOSES",)),
    ("wedge_break_bars", 288, "Claude", "test", "One day for the other side to give way after touch 3.",
     _EN + "WEDGE_BREAK_BARS = 288", "L6b.1", ("WEDGE_BREAK_BARS",)),
    # ---- the other management defaults (manager.py DEFAULTS) - his table: plan/work/management-decisions.md -----
    ("slippage_share", 0.25, "Claude", "test", "The old spread model's slippage (share of the spread). Only for the "
     "labelled 'real spread' stress line, never on top of Rs 40.", _MG + "DEFAULTS slip_frac", "L2",
     ("slip_frac",)),
    ("setup_near_level_adr", 0.15, "Claude (T12 M35)", "test", "How near a closing setup must be to the level.",
     _MG + "DEFAULTS setup_tol_adr", "L19a", ("setup_tol_adr",)),
    ("setup_wait_bars", 48, "Claude (the old bars-and-level test; off while hedge_wait = 0)", "test",
     "4 hours for a closing setup in the old test.", _MG + "DEFAULTS setup_wait", "L19a", ("setup_wait",)),
    ("rearm_adr", 0.10, "Claude", "test", "Distance before a hedge may be re-armed.", _MG + "DEFAULTS rearm_adr",
     "L19a", ("rearm_adr",)),
    ("runner_tolerance_adr", 2.0, "Claude ('his rule, my number')", "test",
     "How near the escaped runner an opposite setup must arm to count.", _MG + "DEFAULTS runner_tol_adr", "L19c",
     ("runner_tol_adr",)),
    ("no_timeout", 0, "Tejas #19, #20, #104", "his", "No timeout: a trade ends when price ends it (0 = none).",
     _MG + "DEFAULTS max_bars", "L19a", ("max_bars",)),
    ("leg_left_is_a_new_entry", "new_entry", "Tejas #22 (6 Sep), #104", "his",
     "The leg left open after a close is a NEW trade at that price.", _MG + "DEFAULTS exit_mode", "L19a",
     ("exit_mode",)),
    ("closed_hedge_price", "close", "Claude (section 61; likhith.md B20)", "test",
     "A closed hedge is credited at the close of the candle that confirmed the setup, not at the level.",
     _MG + "DEFAULTS book_px", "L19a", ("book_px",)),
    ("hedge_closes_on", "fill", "Tejas #47", "his",
     "A hedge closes when price comes back to the opposite setup's gap, at that price.", _MG + "DEFAULTS book_on",
     "L19a", ("book_on",)),
    ("hedge_after_a_close", "entry", "Tejas #104", "his",
     "'There shouldn't be a single position without the fixed hedge': the leg left gets that setup's own hedge.",
     _MG + "DEFAULTS rehedge", "L19a", ("rehedge",)),
    ("target_once_hedged", "none", "Tejas #51", "his", "Once hedged, the first trade's target is gone.",
     _MG + "DEFAULTS hedged_target", "L19a", ("hedged_target",)),
    ("hedged_setup_wait_bars", 0, "Tejas #20 (no clock)", "his",
     "How long a hedged trade waits for a proper setup: 0 = no clock.", _MG + "DEFAULTS hedge_wait", "L19a",
     ("hedge_wait",)),
    ("carried_leg_hedge", "fixed", "Tejas #104", "his", "The carried leg waits with the flipping setup's own hedge.",
     _MG + "DEFAULTS carried_hedge", "L19a", ("carried_hedge",)),
    ("a_new_setup_resolves_the_hedge", "any", "Tejas #55", "his",
     "While hedged, the first new setup either way that price comes back to resolves it.",
     _MG + "DEFAULTS hedge_resolve", "L19a", ("hedge_resolve",)),
    ("carried_leg_close", "any", "Tejas #53", "his", "The carried leg may close at any price on a proper setup.",
     _MG + "DEFAULTS carried_close", "L19a", ("carried_close",)),
    ("leg_left_close", "any", "Tejas #22 (6 Sep)", "his", "The leg left may end at a loss (the whole trade decides).",
     _MG + "DEFAULTS leftover_close", "L19a", ("leftover_close",)),
    ("trail_only_when_not_hedged", "unhedged", "Tejas #113", "his", "Trailing only when not hedged.",
     _MG + "DEFAULTS trail_exit", "L19a", ("trail_exit",)),
    ("fill_at_the_exact_price", True, "Claude", "test", "Limit fills at the order's price, not a better one.",
     _MG + "DEFAULTS fill_exact", "L3", ("fill_exact",)),
    ("news_tiny_loss_adr", 0.05, "Claude (only for the retired #52 reading)", "test",
     "'A tiny loss' at news, as a share of a day's range.", _MG + "DEFAULTS news_tiny_adr", "L19a",
     ("news_tiny_adr",)),
    ("decide_at_entry", True, "Tejas #111, #17", "his", "Hedge-or-room is decided at entry.",
     _MG + "DEFAULTS decide_at_entry", "L19a", ("decide_at_entry",)),
    ("floor", "off", "Tejas #53", "his", "No floor (the numbers below it are unused while it is off).",
     _MG + "DEFAULTS floor, floor_n = 4, floor_adr = 4.0", "L19a", ("floor", "floor_n", "floor_adr")),
    ("zone_flip_in_management", "keep", "Claude", "test", "What the manager does with a flipped zone. Read by no "
     "code: the zones' flip is the block's own rule (L6b.2, #77, #78) - machine/tester/management.py RETIRED.",
     _MG + "DEFAULTS zone_flip", "L19c", ("zone_flip",)),
    ("exit_on_opposite_setup_unhedged", "off", "Claude", "test", "An opposite setup closes an unhedged trade. Read by "
     "no code: his rule for it is the trailing one (M23, 'unless an opposite setup forms there') - "
     "machine/tester/management.py RETIRED.", _MG + "DEFAULTS opp_exit", "L19c", ("opp_exit",)),
    # ---- the new machine's own -----------------------------------------------------------------------------------
    ("adr_ruler_days", 10, "Claude", "test",
     "'A day's range' - the ruler for every number written as a share of a day - is his 10-day ADR on broker days "
     "(the old engine's ruler used UTC days: plan/work/rules-missing.md).",
     _LV + "adr_bands() 'adr' (UTC-day 10-day average)", "L6.1"),
    ("fib_ratios", (0.5, 0.618, 0.68, 0.705, 0.786),
     "Tejas #135 (0.68 entry, 0.5 target); 0.618 = the golden ratio (his F1 question open); 0.705 and 0.786 Claude's",
     "test", "The retracement levels the fib block draws: price = top - ratio x (top - bottom) for a buy.",
     "old/build/fib_135.py LEVELS", "L6.1, L7b", ()),
])
# ---- L6b.3: his bias on the hour / day / week charts (machine/blocks/bias.py) -----------------------------------------
_MORE.update({
    "bias_wait_bars": _s(
        12, "Claude (the old bias: levels.htf_bias_trade(horizon=12))", "test",
        "On the hour / day / week chart: candles allowed from the grab to his break. The gap then follows his fixed "
        "rule (its 3rd candle at most 6 candles after the break).", _LV + "htf_bias_trade(horizon=12)",
        used_by="L6b.3"),
    "bias_waits_for_the_return": _s(
        False, "Tejas: #74 ('reading it perfectly' on the reading that does not wait) and strategy.md line 104 ('price "
        "comes back to the gap') differ - both are kept, a version", "test",
        "False = a chart's trade is ON when his break and its first gap exist (the reading drawn and confirmed, #74). "
        "True = it also waits for price to come back to the gap's near edge (strategy.md section 6, #70 B).",
        _LV + "htf_bias_trade(need_return=False)", options=(False, True), used_by="L6b.3"),
    "bias_doji": _s(
        "no_change", "Tejas #70 C (a doji candle read '75% bullish', scored as up); four-boxes B3-06 lists both",
        "test", "no_change = a doji changes nothing. neutral = a doji as the last closed candle makes that chart "
        "NEUTRAL ('a doji weakens it').", "new (L6b.3): the old bias never built the doji",
        options=("no_change", "neutral"), used_by="L6b.3"),
    "doji_body_share": _s(
        0.10, "Claude (#70 C 'indecision candle' gave no number)", "test",
        "A doji = a candle whose body is at most this share of its range.", "new (L6b.3)", used_by="L6b.3"),
})
# ---- L6b.1: trendlines, triangles, wedges, false breakouts (machine/blocks/trendlines.py, patterns.py) ---------------
_MORE.update({
    "trendline_swing_turn_adr": _s(
        0.21, "Claude, from his marked charts: the old engine's 0.25 (levels.PROM_FRAC, structural swings - its lines "
        "matched his 8 trendline calls, #92 / #94) was a share of its UTC-day ruler; the new ruler is 1.17x it (median "
        "2016-2020), so 0.25 / 1.17 = 0.21 keeps the same dollars. On the new ruler every line he answered about in "
        "2016-2020 is drawn as he saw it for 0.20-0.23: Q68 A and B (#69, 2-touch lines in the path), Q92 A and C and "
        "Q94 A (#92, #94: 2 touches, never a trendline); at 0.25 his Q68 B line is lost (plan/work/L6b1-notes.md)",
        "test",
        "A line is drawn through his 2-candle swings (wick ends, #112) that TURNED at least this share of a day's range "
        "from the last one kept (a zig-zag); 0 = through every swing.", _LV + "PROM_FRAC = 0.25, structural_swings()",
        used_by="L6b.1"),
    "trendline_touch_span_bars": _s(
        (6, 300), "Claude (the old engine)", "test",
        "On the 5-minute chart: two touches of a line are at least 6 and at most 300 candles apart (the third touch "
        "too, from the second).", _LV + "trendlines(): 6 <= bi - ai <= 300, third within 300", used_by="L6b.1"),
    "trendline_touch_span_bars_other": _s(
        (3, 200), "Claude (the old engine)", "test",
        "The same on the hour (and any other) chart: 3 to 200 candles.",
        _LV + "htf_trendlines(): 3 <= bi - ai <= 200, third within 200", used_by="L6b.1"),
    "trendline_touch_fill_bars": _s(
        10, "Claude (the old engine; AMBIGUITY F10 - where a limit on a moving line fills is a version, four-boxes "
        "B3-16)", "test",
        "The touch entry's limit sits on the line; the line moves, so the price is the line this many candles after "
        "the touch is known - the worse fill of that wait.", _EN + "type2: entry = the line at arm + FVG_RETURN_BARS",
        used_by="L6b.1"),
})
# ---- L6b.2: strong zones, the zone break, the zone entry, the zone before the target (machine/blocks/zones.py) ----------
_MORE.update({
    "zone_windows": _s(
        (("today", "5m", 1, 3), ("week", "5m", 5, 5), ("hour", "1h", 20, 20)),
        "Claude (the old engine's three windows: levels.build + build/zones_cache.py; his words are '5-minute and 1-hour' "
        "(#78, #80) and 'as far back as needed' (#78) - the number of days is a version, four-boxes B3-03, B3-25)",
        "test",
        "The windows his zones are counted in: (name, chart, broker days counted, days the zone lives). today = the "
        "5-minute candles of the last day; week = the 5-minute candles of the last 5 days; hour = the HOUR candles of "
        "the last 20 days (strategy-understanding 47a: the old engine counted 5-minute candles there).",
        _LV + "build(): tpo_zones(window_days=1 / 5 / 20, life_days=None (3) / 5 / 20)", used_by="L6b.2"),
    "zone_spent_role_waits_for_the_break": _s(
        True, "Tejas #78 (Claude's reading of 'It will only if it breaks the zone, then it will actually support')",
        "his",
        "True = price coming back to a zone whose role on that side is already used (#78: 'it will not be a resistance "
        "next time') ends the zone only if that visit does not break it; a visit that breaks it hands it the other "
        "role (#78: 'if it breaks the zone, then it will actually [be] support'). False = the old engine: the zone "
        "ends at the start of that visit.", _LV + "_mark_zone_touches(): role in spent -> died at the touch",
        options=(True, False), used_by="L6b.2"),
    "zone_quiet_stretch_days": _s(
        20, "Claude (#79: '14th to 16th of March ... around March 2 to March 5, in the same zone' - two weeks apart; "
        "no number)", "test",
        "A measure only (no decision reads it - four-boxes B4-13 tests it): how many EARLIER quiet zones, built from "
        "other candles, sat at the same price in this many broker days before ('two quiet stretches at the same "
        "price make it stronger', strategy.md section 3).", "new (L6b.2): the old engine never counted it (65s)",
        used_by="L6b.2, L13b"),
    "double_profit_zone_rank": _s(
        2, "Claude (the old manager's dbl_min = the hour chart - his strategy.md section 3 '1h stronger')", "test",
        "The double-profit re-entry (strategy.md sections 3, 10) books at a zone of at least this rank standing "
        "between the entry and the target: 0 = any window, 1 = week or hour, 2 = the hour chart only.",
        _MG + "DEFAULTS dbl_min = 2", options=(0, 1, 2), t12="M30", used_by="L6b.2, L19c", aliases=("dbl_min",)),
})
# ---- L6b.4: the 1-minute layer (machine/blocks/one_minute.py) - his #100 "the same apply for one minute" -------------
_MORE.update({
    "trendline_swing_turn_adr_1m": _s(
        0.04, "Claude, read off his two 1-minute lines (plan/work/L6b4-notes.md): F04 (#111, 11 Mar 2019, the lows "
        "17:12 and 17:36 IST) is drawn for 0 to 0.053, Q98 C (#98, 9 Jun 2017, the lows 20:46 and 21:35 IST) for 0 to "
        "0.042; 0.04 = the fewest lines that still draw both", "test",
        "The 1-minute chart's zig-zag: a line is drawn through the 1-minute swings that turned at least this share of "
        "a day's range from the last one kept (the 5-minute chart's trendline_swing_turn_adr, for 1-minute candles).",
        "new (L6b.4): the old engine had no 1-minute lines", used_by="L6b.4"),
    "trendline_touch_span_bars_1m": _s(
        (6, 300), "Claude: the 5-minute chart's numbers in 1-minute candles (#100 'the same apply for one minute'); "
        "his two 1-minute lines are 24 and 49 candles long", "test",
        "On the 1-minute chart: two touches of a line are at least 6 and at most 300 candles apart (the third touch "
        "too, from the second).", "new (L6b.4)", used_by="L6b.4"),
    "trendline_pairs_1m": _s(
        "any", "Claude, from his Q98 C line (#98): it joins the 20:46 low to the 21:35 low and passes over three higher "
        "lows between them - joining only neighbouring swings (the old engine's way, kept for the 5-minute and hour "
        "charts) cannot draw it", "test",
        "any = on the 1-minute chart a line may join ANY two swings, as long as nothing goes through it between them "
        "(his rule: 'drawn through the wick ends; a touch through the line = not a trendline', #92, #94, #112). "
        "neighbours = only two swings next to each other (the 5-minute and hour charts).",
        "new (L6b.4)", options=("any", "neighbours"), used_by="L6b.4"),
    "one_minute_line_life_bars": _s(
        2880, "Claude: settings level_life_bars (2,880 five-minute candles) counted in 1-minute candles, like the "
        "spans", "test",
        "1-minute candles a 1-minute line is tracked before it is dropped (2 days).", "new (L6b.4)", used_by="L6b.4"),
    "one_minute_new_lines_after_a_take": _s(
        False, "Claude's reading of his F04 answer (#111, #112): after his line was taken, he named the next gap (18:35 "
        "IST, 1295.23-1295.61) 'that is actual entry' - on the picture he saw, a new line already ran from the "
        "take's low (18:08 -> 18:22 lows). The machine's re-walk finds that same gap; checking it afresh would wait "
        "again and skip it", "test",
        "False = once a setup has waited, only the lines known at its FIRST check can make it wait again (all of them "
        "must be taken first - #99 'wait for that to take out first'); a line born from the new grab does not. "
        "True = every gap after a take is checked afresh.", "new (L6b.4)", options=(False, True), used_by="L6b.4"),
    "zone_break_inner_1m_lines": _s(
        True, "Tejas #98 C ('if you look into the one minute candle there is definitely a triangle pattern ... I want "
        "price to break that trend line and I will wait for the fair value gap in five minutes'), #99 ('trend line "
        "type of thing and triangle type of thing ... inside that zone'), #111 ('that same thing applies for zone')",
        "his",
        "The zone break's liquidity inside the zone (settings zone_break_inner_liquidity) also counts the 1-minute "
        "lines inside it - a triangle's lines among them - when the market has 1-minute bars; the first one taken "
        "(flat level or line) is the take the 5-minute gap follows. False = his 5-minute levels only (L6b.2).",
        "new (L6b.4)", options=(True, False), used_by="L6b.4"),
})

# ---- L19c: his management rules that are a TEST in his table (plan/work/management-decisions.md, T12) --------------
# Each one is run by the hedging tester (machine/tester/hedging.py) in every value listed, and a research version may
# set it in exit.management (machine/tester/management.py VERSIONS says which, and why). Version 1 = the first value
# = what the hedging tester did before L19c, so every number made before stays the same.
_MORE.update({
    "a_real_stop_instead_of_the_hedge": _s(
        "never", "Tejas #17 ('sometimes a real stop is better ... the model should define when'): version 1 = his rule, "
        "the hedge (M01); the other readings = four-boxes B3-11 ('stop vs hedge, per kind of setup')", "test",
        "never = the hedge is the protection (his rule, M01). when_small = a trade whose hedge distance at entry is at "
        "most a_small_hedge_is_within_adr of a day's range gets a REAL stop at its hedge point instead (#17: 'a very "
        "small stop'). always = every trade of this kind (this idea) gets a real stop there. A real stop is never moved "
        "(no room); a trade on a real stop is not hedged at news either - it closes in profit, else the stop protects.",
        "new (L19c): the old manager had no real stop", options=("never", "when_small", "always"), t12="M02",
        used_by="L19c"),
    "a_small_hedge_is_within_adr": _s(
        0.10, "Claude (#17 'a very small stop' gave no number)", "test",
        "For a_real_stop_instead_of_the_hedge = when_small: 'small' = a hedge distance at entry of at most this share "
        "of a day's range.", "new (L19c)", t12="M02", used_by="L19c"),
    "on_a_friday": _s(
        "as_any_day", "Tejas strategy.md section 11 ('Decide hedge-or-not AT ENTRY (esp. Friday)'); the readings are "
        "four-boxes B3-13 ('on Friday: room / hedge / skip')", "test",
        "as_any_day = a Friday entry is decided like any other (version 1). room = a trade filled on a Friday (MT5 "
        "server time) always gets room - the hedge waits at the next liquidity. hedge = it never gets room - the hedge "
        "waits at the hedge point. ('skip' is not a value: not trading on Fridays keeps only some setups - a weekday "
        "SLICE, dimension 9, chosen before running.)",
        "new (L19c)", options=("as_any_day", "room", "hedge"), t12="M09", used_by="L19c"),
    "trail_when_hour_and_day_split": _s(
        "first_target", "Claude (his #82 / #133 say 'with' and 'against' only; the old code took the first target)",
        "test",
        "Trailing when the hour and the day DISAGREE: first_target = close at the first target (the safer side, version "
        "1). hold = trail as if they were with the trade (to the next liquidity, unless an opposite setup forms there).",
        "old/ml/agent/manager.py: trail 'hd', split -> the first target", options=("first_target", "hold"), t12="M24",
        used_by="L19c"),
    "news_minutes_before": _s(
        15, "Claude (T12 M28: how many minutes before = test; four-boxes B3-22)", "test",
        "Minutes before the news when the news rule acts.", _MG + "DEFAULTS news_lead_min", t12="M28",
        used_by="L19a, L19c", aliases=("news_lead_min",)),
    "news_events": _s(
        "forex_factory_red_and_orange", "Tejas (T7, inbox 2026-09-28-0017 and 0020: red AND orange = version 1); his "
        "#142: 'I want you to machine to verify that ... come up with the best thing' - red only is the version to "
        "compare (four-boxes B3-29)", "test",
        "Which news counts: Forex Factory RED and ORANGE (version 1), or RED only. The old engine used NFP + FOMC only.",
        _MG + "news events = NFP + FOMC", options=("forex_factory_red_and_orange", "forex_factory_red"), t12="M28",
        used_by="L12, L19a, L19c"),
    "halfway_reversal": _s(
        "off", "open: 'halfway' is a word still open (four-boxes B2-05, waiting in the weekly picture batch)", "test",
        "Halfway reversal -> no hedge, re-enter lower. Off until the word is settled: the old project's two readings "
        "(noh / reenter, at halfway to the target or to the hedge point) become versions after his answer.",
        _MG + "DEFAULTS halfway, halfway_frac = 0.50, halfway_once = True", options=("off",), t12="M29",
        used_by="L19c (after B2-05)", aliases=("halfway", "halfway_frac", "halfway_once")),
    "trendline_hedge_pad_adr": _s(
        None, "Claude (T12 M27 / four-boxes B3-16: 'a few pips' past the line = test). None = where the line counts as "
        "broken (line_break_tolerance x line_touch_adr = 0.05 of a day's range) - what L6b.1 built; the old manager's "
        "pad was 0.02", "test",
        "How far past the line the trendline touch entry's hedge sits (it then moves with the line, M27), as a share of "
        "a day's range. None = the line's own break distance (version 1).",
        _MG + "DEFAULTS trail_cover_pad_adr = 0.02", t12="M27", used_by="L6b.1, L19c", aliases=("trail_cover_pad_adr",)),
    "trail_max_targets": _s(
        6, "Claude (T12 M35; four-boxes B3-26)", "test", "How many next targets trailing may walk.",
        _MG + "DEFAULTS trail_max", t12="M35", used_by="L19a, L19c", aliases=("trail_max",)),
    "next_liquidity_max_adr": _s(
        2.5, "Claude (T12 M35; four-boxes B3-26)", "test",
        "How far ahead 'the next liquidity' may be (room, trailing, the leg left's target), in days of range.",
        _MG + "DEFAULTS next_liq_adr", t12="M35", used_by="L19a, L19c", aliases=("next_liq_adr",)),
    "close_one_side_early": _s(
        False, "not in his words (his old EA, path 5; T12 M34)", "test",
        "Close one side at a nearer level while the hedge stays open. NOT a version: a close at a level on its own is "
        "what his M10 forbids ('never at the next level on its own', #47, #113) - machine/tester/management.py.",
        _MG + "DEFAULTS paired", options=(False,), t12="M34", used_by="L19c", aliases=("paired",)),
    "strong_zone_as_hedge_point": _s(
        "off", "Tejas strategy.md section 3 ('hedge point' is one of a zone's uses) - the readings are Claude's (the old "
        "manager's two) + the zone grid's (strategy-code.md section 17, T12 M33 / four-boxes B3-10)", "test",
        "The FIRST hedge at a strong zone against the trade (live at the fill, at least strong_zone_hedge_rank, its edge "
        "within strong_zone_hedge_max_adr of a day's range): off = his hedge point only (version 1). nearer = whichever "
        "price comes first - the hedge point (after room) or the zone's near edge. zone = the zone's near edge is the "
        "hedge price. zone_break = the zone's FAR edge: the hedge opens when price breaks through the zone (the zone "
        "grid, M33).",
        _MG + "DEFAULTS hedge_zone", options=("off", "nearer", "zone", "zone_break"), t12="M31", used_by="L19c",
        aliases=("hedge_zone",)),
    "strong_zone_hedge_max_adr": _s(
        1.5, "Claude (the old manager's hedge_zone_adr)", "test",
        "For strong_zone_as_hedge_point: the zone's edge at most this many days of range from the entry.",
        _MG + "DEFAULTS hedge_zone_adr = 1.5", t12="M31", used_by="L19c", aliases=("hedge_zone_adr",)),
    "strong_zone_hedge_rank": _s(
        2, "Tejas strategy.md section 3 ('1h stronger': the hour chart = 2, his 'strong'); the old manager's "
        "hedge_zone_min", "test",
        "For strong_zone_as_hedge_point: the zone's rank at least this (0 = any window, 1 = week or hour, 2 = the hour "
        "chart).", _MG + "DEFAULTS hedge_zone_min = 2", options=(0, 1, 2), t12="M31", used_by="L19c",
        aliases=("hedge_zone_min",)),
    "in_a_zone_against_you": _s(
        "off", "Tejas strategy.md section 6 ('in a HTF zone against you -> aim small / avoid'; T12 M32, four-boxes "
        "B3-04)", "test",
        "off = version 1. small = when an hour-chart zone (settings zone_hour_chart_rank) stands between the entry and the "
        "target at the fill, the target is cut back to that zone's near edge (he books in front of it). ('avoid' is not "
        "a value: not taking the trade keeps only some setups - a place SLICE, dimension 8, chosen before running.)",
        _MG + "DEFAULTS zone_block", options=("off", "small"), t12="M32", used_by="L19c", aliases=("zone_block",)),
    "double_profit_reentry": _s(
        "off", "Tejas strategy.md sections 3, 10 ('book profit at a zone before the target; double-profit re-entry'; "
        "#82 'we will backtest later'; T12 M30, four-boxes B3-09) - WHERE he gets back in is not in his words: the "
        "three readings are the old manager's", "test",
        "off = version 1. Otherwise, while one leg is open and not hedged, a zone of at least double_profit_zone_rank "
        "standing between the entry and the target (zones.in_path, at the fill) is booked at: the leg closes at its near "
        "edge (the first profit). Back in, in the same direction, once: retest = when price comes back to that price; "
        "break = when price goes through the zone's far edge; setup = at the fill of the next setup of his that way. "
        "The leg back in keeps the trade's target and hedge price. If price reaches the target or the hedge price first, "
        "the trade ends with the one profit (Claude's: his words give no end to the wait).",
        _MG + "DEFAULTS dbl, dbl_min = 2, dbl_wait = 0", options=("off", "retest", "break", "setup"), t12="M30",
        used_by="L19c", aliases=("dbl", "dbl_wait")),
})
# ---- L6-new: the break of his ADR zone (idea #4, #76; machine/blocks/premium_discount.py breaks) ----------------------
_MORE.update({
    "adr_break_beyond_adr": _s(
        0.0, "Claude (#76 'if it breaks the lower range of ADR, lower zone, it will go further low' gives no "
        "distance; 'keeps going' is the machine's to find, #154)", "test",
        "A break of his ADR zone = a 5-minute candle CLOSES beyond the zone's far edge (below the green zone's bottom, "
        "above the red zone's top) by at least this share of a day's range. 0 = any close beyond; a wick is never a "
        "break (his zone rule, rules.py what_a_real_zone_break_is: 'tiny wicks are not considered').",
        "new (L6-new): the old engine had no ADR zone break", used_by="L6-new"),
})

MORE: Mapping[str, Setting] = MappingProxyType(_MORE)
ALL: Mapping[str, Setting] = MappingProxyType({**_CHANGEABLE, **_MORE})

_ALIAS: dict[str, str] = {}
for _name, _set in ALL.items():
    for _a in (_name, *_set.aliases):
        if rules.fixed_name(_a) is not None:
            raise RuntimeError(f"{_a!r} is both a setting and a fixed rule")
        if _a.lower() in _ALIAS and _ALIAS[_a.lower()] != _name:
            raise RuntimeError(f"two settings share the name {_a!r}")
        _ALIAS[_a.lower()] = _name


_ACTIVE: "Resolved | None" = None      # the version's settings while using() runs, else None (today's values)


def get(name: str) -> Any:
    """Today's value of one setting: settings.get('how_long_a_swept_level_stays_dead') -> 576.
    Inside `with using(changes):` it is that version's value (every block reads its settings through here)."""
    if _ACTIVE is not None:
        return _ACTIVE.values[name]
    return ALL[name].value


@dataclass(frozen=True)
class Resolved:
    """The settings one version runs with: today's values with the version's changes on top."""
    values: Mapping[str, Any]
    changed: tuple[str, ...]              # the settings this version changed
    breaks_his_rules: tuple[str, ...]     # one line per changed setting that is HIS rule - every report prints them

    def __getitem__(self, name: str) -> Any:
        return self.values[name]


def resolve(changes: Mapping[str, Any] | None = None) -> Resolved:
    """Today's settings with a form's / version's / test's changes applied.

    Refuses (FixedRuleError, with his words) any change that touches a fixed rule, by its name or an old code name.
    Refuses (SettingError) a name that is no setting, a value outside the options, and years that reach the sealed
    ones. A change to a setting that is HIS rule is allowed only as a comparison and is listed in breaks_his_rules.
    """
    changes = dict(changes or {})
    rules.refuse_fixed(changes)
    values = {k: s.value for k, s in ALL.items()}
    changed, breaks = [], []
    for asked, v in changes.items():
        name = _ALIAS.get(str(asked).lower())
        if name is None:
            raise SettingError(f"'{asked}' is not a setting of this machine. Fixed rules: machine/rules.py; "
                               "settings: machine/settings.py.")
        s = ALL[name]
        _check_value(name, s, v)
        values[name] = tuple(v) if isinstance(v, list) else v
        changed.append(name)
        if s.status == "his" and values[name] != s.value:
            breaks.append(f"{name} = {values[name]!r} is not his rule ({s.who}: {s.value!r}). A comparison only.")
    return Resolved(MappingProxyType(values), tuple(changed), tuple(breaks))


@contextmanager
def using(changes: "Mapping[str, Any] | Resolved | None" = None) -> Iterator[Resolved]:
    """Run blocks with ONE version's settings (the idea compiler, L7a): inside the with-block settings.get() gives
    the version's values; after it, today's again (even after an error).

        with settings.using({"order_block_max_age_bars": 60}) as r:
            ...                       # every block now reads 60; r.breaks_his_rules lists changes to his rules

    The changes go on top of TODAY's values (resolve() refuses his fixed rules with his words first) - not on top of
    an outer using(): one version at a time.
    """
    global _ACTIVE
    r = changes if isinstance(changes, Resolved) else resolve(changes)
    before = _ACTIVE
    _ACTIVE = r
    try:
        yield r
    finally:
        _ACTIVE = before


def fingerprint(r: Resolved | None = None) -> str:
    """A short hash of every setting value in force (today's, or one version's): a cache made under other settings
    is never reused (likhith.md B19)."""
    values = (r or (_ACTIVE if _ACTIVE is not None else resolve())).values
    text = repr(sorted((k, repr(v)) for k, v in values.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _check_value(name: str, s: Setting, v: Any) -> None:
    if name == "entries_on":
        vals = [v] if isinstance(v, str) else list(v)
        bad = [e for e in vals if e not in s.options]
        if bad:
            raise SettingError(f"entries_on: {bad} are not entries of this machine. It has {list(s.options)}.")
        return
    if name == "years":
        y0, y1 = (int(x) for x in v)
        if y1 >= int(config.SEALED_FROM[:4]):
            raise SettingError(f"years {y0}-{y1} reach the sealed years. {SEALED_WORDS}.")
        if y0 < config.TUNE_YEARS[0] or y0 > y1:
            raise SettingError(f"years {y0}-{y1}: the data covers {config.TUNE_YEARS[0]}-{config.CHECK_YEARS[-1]}.")
        return
    if name == "lot_size":
        steps = round(float(v) / config.LOT_STEP, 9)
        if v <= 0 or steps != int(steps):
            raise SettingError(f"lot_size {v}: lots are only 0.01, 0.02, 0.03 ... (LEARNED #116).")
        return
    if s.options is not None and not any(v == o and type(v) is type(o) for o in s.options):
        raise SettingError(f"'{name}' cannot be {v!r}. It can be one of {list(s.options)}.")
