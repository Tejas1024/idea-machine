"""The hedging tester (L19a): Tejas's own method - a HEDGE instead of a stop - replayed on 1-minute bars.

    from machine.tester import hedging
    facts = hedging.MarketFacts.build(market, fp, book)      # his setups, bias, liquidity, news - read once
    rep = hedging.replay(signals, facts)                     # one trade at a time, like the tester
    print(hedging.summary_text(hedging.summarize(rep)))

The rules come ONLY from his management table, plan/work/management-decisions.md (T12, 28 Sep 2026). His words are
version 1. Every number that is Claude's is a setting in machine/settings.py, so L19c can test it as a version. RULES
(below) lists every row of the table: what this file does with it and the lessons it rests on. A row that needs a
block not built yet says which prompt builds it. It is never guessed.

Words used here:
  leg        one open position (0.01 lot, M05). A trade has one leg, or two while hedged.
  hedge      the opposite leg, the same size, opened instead of a stop (M01).
  current entry   the price a leg counts from now: its open price, or, after the other leg was closed, the fill
             price of the setup that closed it. The leg left "is a NEW trade at that price" (M15); #134 counts the
             1:4 "from the current entry".
  R          the hedge distance: from the current entry of the leg that was hedged to the price the hedge opened at.
  MT5 prices every distance of his (R, the 1:4, the current entry) is measured on the prices MT5 shows for the orders:
             a buy fills on the ask, a sell on the bid - the same prices his chart and his order tickets show. So a
             spread jump that fires a hedge (a buy stop on the ask) does not shrink R. The recorded prices and the
             money stay on the bid, with Rs 40 all-in (the tester's rule, lesson B14).
  his setups his own entries (today: grab + gap; M10 names five), used as the moments a leg may close.

How a trade runs (each step has a test in tests/test_hedging.py):
  1. FILL: the signal's order fills exactly as in the tester (engine fill: the next bar only, a buy on the ask ...).
  2. DECIDE AT ENTRY (M06, #111): if the hour AND the day are both with the trade, the hedge waits at the next
     liquidity past the hedge point ("give it room"). If no liquidity is there, it waits `how_much_room` days of
     range past it (M07, Claude's 1.0). Otherwise the hedge waits at the hedge point (M03: the grab's extreme, which
     the signal carries as its `stop`).
  3. ONE LEG OPEN - the first of these:
       - price reaches the hedge price against the leg: the hedge opens there (after a gap: at the minute's open).
       - price reaches the target: if the hour AND the day are with the trade, hold on to the next liquidity (M23,
         M25), up to `trail_max_targets` steps (Claude's 6). If they are against, or split, close there (M24: split
         = the first target, Claude's). While held past a target, the first opposite setup of his that price comes
         back to closes the leg at that setup's fill price ("unless an opposite setup forms there", M23).
       - Forex Factory red or orange news in `news_minutes_before` minutes (M28, #110; Claude's 15). If the leg is in
         profit after its Rs 40: close at the price then. At break-even or a loss: hedge at the price then.
  4. TWO LEGS OPEN (hedged): no target (M14). Only one of his setups closes a leg (M10, M13), and only when price
     comes back to its gap. The leg closes AT THAT FILL PRICE: the price really there when the close was confirmed
     (lesson B20, #47), never the level. The setup closes the leg it points against, even at a loss (M11, #134), but
     only if its fill price is at least 4 x R past that leg's current entry, on that leg's winning side (M12,
     #107/#108/#111/#134). Inside that distance nothing closes (the no-close zone). Only setups known from the
     hedge's minute on count (Claude's reading, the old engine's too). News does nothing to a hedged trade.
  5. AFTER A CLOSE: the leg left is that setup's own trade (M15, #104). Its current entry = the fill price. Its FIXED
     hedge = that setup's own hedge point (no room: "fixed"). Its target = the farther of that setup's target and
     4 x R' past the current entry (R' = from the current entry to that hedge), then the next real liquidity at or
     past it (M16, #124, #125). Back to step 3.
  6. THE END: the last leg closes (step 3). If the trade already closed a leg before, and this last leg would end at
     a loss, it may close only when the WHOLE trade (every position, Rs 40 each) is in profit (M17, #112). If not,
     its target moves out to the next liquidity past the price where the whole trade breaks even.
  - Inside one minute the order is unknown: the hedge comes before a target or a close (the worse case), counted
    "ambiguous". In the minute a leg opens after the minute's open, its hedge price counts (price had to pass the
    entry to reach it); its target does not.
  - One trade at a time (M18, #55). A limit order still waiting is cancelled by a newer setup (M39), as the tester.
  - No timeout and no floor (M19, M20). A trade still open at the window end is reported apart, never counted in Rs.
  - Money: every leg is one position at Rs 40 all-in (M36, #54, #115), through machine.costs.pnl. A trade counts in
    the year its last leg closed.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from machine import config, costs, settings
from machine.tester import engine
from machine.tester.engine import EPS, MINUTE_NS, NEVER, Filled, Missed
from machine.tester.models import Signal

SIGN = {"buy": 1, "sell": -1}
OTHER = {"buy": "sell", "sell": "buy"}
TABLE = "plan/work/management-decisions.md"

# Every row of his table (T12): (row, what this file does, his lessons). "waits for" = a block not built yet.
RULES: tuple[tuple[str, str, str], ...] = (
    ("M01", "built: no stop; the hedge is the protection", "#9, #17; strategy.md 11"),
    ("M02", "a version, not here: the plain stop is the other arm of the 2x2 (B3-11)", "#17"),
    ("M03", "built: the hedge point = the signal's hedge point (the grab's extreme)", "#9, #33, #111"),
    ("M04", "a test, off (parked: 'we will backtest it later')", "#112"),
    ("M05", "built: 0.01 lot on every leg", "strategy.md 11; #116"),
    ("M06", "built: the hour AND the day with the trade -> the hedge waits at the next liquidity. 'Setup good' "
            "(B2-04, open) = any valid entry of his: every signal here is one (reading A; for idea #1 reading B "
            "is the same; C = a version for L19c)", "#111; strategy.md 11"),
    ("M07", "built: no liquidity there -> how_much_room days of range (Claude's 1.0, a test)", "-"),
    ("M08", "not built: a buy at the all-time high is only counted, not given the room to the next strong zone (the "
            "zones exist since L6b.2; the zone break entry carries its own all-time-high hedge, #97)",
     "strategy.md 12; #97"),
    ("M09", "a test, off (the Friday part)", "#17"),
    ("M10", "built (L19b): any of his entries closes a leg - grab + gap, zone entry, zone break (machine/entries); the "
            "trendline touch and the triangle false break join the day their open words are set (B2-01 'steep', "
            "B2-02 'big move': his answer, or a version that names the number)", "#47, #113"),
    ("M11", "built: the leg the setup points against closes, even at a loss", "#134"),
    ("M12", "built: only 4 x R past that leg's current entry, after every hedge", "#107, #108, #111, #134"),
    ("M13", "built: only a full sweep -> break -> gap setup closes (his setups)", "#95, #47"),
    ("M14", "built: once hedged, the target is gone", "#51"),
    ("M15", "built: the leg left is a new trade at the fill price, with that setup's FIXED hedge", "#22, #104"),
    ("M16", "built: its target = past the 1:4 zone, then the next real liquidity", "#111, #112, #124, #125"),
    ("M17", "built: a last leg that ends red needs the whole trade green, else its target moves out", "#112"),
    ("M18", "built: one trade at a time; a new setup while a trade is on is skipped", "#55"),
    ("M19", "built: no timeout", "#19, #20, #104"),
    ("M20", "built: no floor", "#53"),
    ("M21", "a test: no cap here (the old cap of 4 only fed the escape, which is off)", "strategy.md 11"),
    ("M22", "a test, off (trend escape)", "#19, #51"),
    ("M23", "built: trailing only when not hedged, by the hour AND the day", "#82, #133"),
    ("M24", "built: hour and day split -> the first target (Claude's, a test)", "-"),
    ("M25", "built: the target moves to the next liquidity while trailing", "#20"),
    ("M26", "built: every close is the whole leg", "strategy.md 10"),
    ("M27", "built (L19b): on a trendline touch entry the first hedge moves with the line - only tighter, only before "
            "the first hedge, never onto a price the market is already past; such a trade gets no room (M06): #69 is "
            "the day AND the hour bullish and still 'keep the hedging below the trend line itself ... trailing'",
     "#69; strategy.md section 9"),
    ("M28", "built: news in profit -> close; break-even or a loss -> hedge. Forex Factory red + orange; "
            "news_minutes_before = Claude's 15 (a test)", "#110, #17; T7"),
    ("M29", "Box 2 (B2-05 'halfway'), off", "strategy.md 1"),
    ("M30", "a test, off (book at a zone, double profit)", "strategy.md 3, 10"),
    ("M31", "a test, off (a strong zone as the hedge point)", "strategy.md 3"),
    ("M32", "a test, off (a zone against you)", "strategy.md 6"),
    ("M33", "a test (zone grid vs gap grid): gap grid only", "strategy.md 11"),
    ("M34", "a test, off (close one side early)", "legacy EA"),
    ("M35", "Claude's numbers, settings: next_liquidity_max_adr 2.5, trail_max_targets 6", "-"),
    ("M36", "built: Rs 40 per position, all-in, no swap", "#54, #115"),
    ("M37", "a test (prediction + probability)", "strategy.md 8"),
    ("M38", "L20 (money plan), not here: 0.01 lot every leg", "#115, #116"),
    ("M39", "built: a waiting limit order is cancelled by a newer setup", "inbox 2026-09-28-0203"),
)


# ================================================================================================================
# the rules' settings (machine/settings.py) - read once per replay
# ================================================================================================================

@dataclass(frozen=True)
class Rules:
    min_rr: float           # M12 how_far_before_any_leg_may_close (his 4.0, #108)
    room: str               # M06 give_it_room_before_covering ("hd" his, "off")
    room_adr: float         # M07 how_much_room (Claude's 1.0)
    next_liq_adr: float     # M35 next_liquidity_max_adr (Claude's 2.5)
    trail: str              # M23 trail_the_target ("hd" his, "off")
    trail_max: int          # M35 trail_max_targets (Claude's 6)
    whole: str              # M17 does_the_whole_trade_decide_a_close ("trade" his)
    leftover_target: str    # M16 what_the_leg_left_over_aims_at_after_a_cover ("his124" his, "setup")
    news: str               # M28 what_news_does ("profit_close" his, "off")
    news_lead_min: int      # M28 news_minutes_before (Claude's 15)
    lot: float              # M05 lot_size
    follow: str = "line"    # M27 the_cover_follows_the_trendline ("line" his, "off")

    @classmethod
    def now(cls) -> "Rules":
        """Today's settings (or the active `settings.using(...)` version's). A value this file cannot run yet is
        refused with the prompt that builds it - never run as something else."""
        g = settings.get
        need = {
            "which_leg_may_close_while_covered": ("any_rr",),      # M11 #134 (the #105 'profit' reading: L19c)
            "where_profit_is_measured_from": ("current",),          # M12 #134 (the same rule as any_rr)
            "close_the_losing_leg_when_a_trend_runs": (False,),     # M22 escape: L19c
            "the_target_moves": (True,),                            # M25
            "give_it_room_before_covering": ("hd", "off"),
            "trail_the_target": ("hd", "off"),
            "does_the_whole_trade_decide_a_close": ("trade",),
            "what_the_leg_left_over_aims_at_after_a_cover": ("his124", "setup"),
            "what_news_does": ("profit_close", "off"),
            "which_entries_can_close_a_position": ("his5", "type1"),
            "halfway_reversal": ("off",),
            "double_profit_reentry": ("off",),
            "close_one_side_early": (False,),
            "strong_zone_as_hedge_point": ("off",),
            "in_a_zone_against_you": ("off",),
            "floor": ("off",),
            "no_timeout": (0,),
            "the_cover_follows_the_trendline": ("line", "off"),     # M27
        }
        for name, ok in need.items():
            if g(name) not in ok:
                raise NotImplementedError(f"the hedging tester runs {name} = {' / '.join(map(repr, ok))} only; "
                                          f"{g(name)!r} is a version for L19c ({TABLE})")
        lot = float(g("lot_size"))
        if costs.lot_steps(lot) != 1:
            raise NotImplementedError("the hedging tester runs 0.01 lot per leg (M05); lot sizes are L20")
        return cls(float(g("how_far_before_any_leg_may_close")), g("give_it_room_before_covering"),
                   float(g("how_much_room")), float(g("next_liquidity_max_adr")), g("trail_the_target"),
                   int(g("trail_max_targets")), g("does_the_whole_trade_decide_a_close"),
                   g("what_the_leg_left_over_aims_at_after_a_cover"), g("what_news_does"),
                   int(g("news_minutes_before")), lot, g("the_cover_follows_the_trendline"))

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ================================================================================================================
# what the market says: his setups, his bias, his liquidity, the news
# ================================================================================================================

@dataclass(frozen=True)
class Setup:
    """One of his setups as a moment a leg may close (M10): it acts only when its own order FILLS."""
    id: str
    direction: str
    known: int                 # ns: when it was known (its signal's known_at)
    fill_i: int                # the 1-minute bar its order filled in (in the facts' book)
    fill_px: float             # bid scale: the price really there then (lesson B20)
    at_open: bool              # filled at that minute's open
    order_px: float            # its order's own price (the gap's near edge) - what the old engine credited
    hedge: float               # its own hedge point (M15: the leg left takes it)
    target: float              # its own target (M16 starts from it)


class Facts(Protocol):
    """What the manager may ask the market. Every answer uses only what was known at moment t."""
    book: engine._Bars
    setups: list[Setup]        # sorted by fill_i
    news: np.ndarray           # int64 ns, shape (k, 2): [the moment the news rule acts, the release], sorted

    def dial(self, direction: str, t: pd.Timestamp) -> str: ...
    def next_level(self, side: str, beyond: float, t: pd.Timestamp, max_away: float | None) -> float | None: ...
    def ruler(self, t: pd.Timestamp) -> float: ...
    def at_all_time_high(self, t: pd.Timestamp) -> bool: ...


def setups_from(signals: list[Signal], book: engine._Bars, cancel_by: dict[str, int] | None = None) -> list[Setup]:
    """His setups -> the moments they fill (the tester's own fill rules; a setup that never fills is left out).
    cancel_by: the checks before the fill of his entry types (engine.replay's cancel_by, L19b)."""
    out = []
    cancel_by = cancel_by or {}
    for s in signals:
        if s.stop is None or len(s.targets) != 1 or s.levels != "price":
            raise ValueError(f"setup {s.id}: a closing setup needs its hedge point (stop) and one target, as prices")
        got = book.fill(s, _ns(s.known_at), cancel_by.get(s.id))
        if isinstance(got, Filled):
            out.append(Setup(s.id, s.direction, _ns(s.known_at), got.i, got.px, got.at_open,
                             float(s.entry_price if s.entry_price is not None else got.px), float(s.stop),
                             float(s.targets[0])))
    return sorted(out, key=lambda x: (x.fill_i, x.known, x.id))


def news_moments(first: pd.Timestamp, last: pd.Timestamp, lead_min: int) -> np.ndarray:
    """His news (Forex Factory red + orange, T7; settings news_events) between first and last, server time, as
    [act at = release - lead_min, release] in ns. Only rows with an exact time; a release not known in advance (the
    L12 calendar's unscheduled rows: the Fed's emergency moves in March 2020) is left out."""
    from machine.data import clock
    from machine.news import forexfactory as FF
    if settings.get("news_events") != "forex_factory_red_and_orange":
        raise NotImplementedError("the hedging tester reads Forex Factory red + orange (T7); other news sets: L19c")
    ff = FF.load()
    ff = ff[ff["colour"].isin(["red", "orange"]) & ~ff["timeMasked"].astype(bool)]
    ev = pd.read_parquet(config.NEWS_DIR / "events.parquet", columns=["utc", "scheduled"])
    unsched = set(pd.DatetimeIndex(ev.loc[~ev["scheduled"].astype(bool), "utc"]).floor("min"))
    utc = pd.DatetimeIndex(ff["utc"]).floor("min")
    utc = utc[~utc.isin(list(unsched))].unique().sort_values()
    server = pd.DatetimeIndex(clock.utc_to_server(utc))
    server = server[(server >= pd.Timestamp(first)) & (server <= pd.Timestamp(last) + pd.Timedelta(minutes=1))]
    rel = server.as_unit("ns").asi8
    act = rel - int(lead_min) * MINUTE_NS
    return np.stack([act, rel], axis=1) if len(rel) else np.zeros((0, 2), dtype=np.int64)


# Tejas's grab + gap entry (strategy.md section 9, Type 1) - used ONLY to close legs (M10). Not an idea: it is never
# traded, recorded or counted as a try. The same shape as the L7a test form.
HIS_GRAB_GAP = dict(
    id="idea-000", title="his grab + gap entry, used only to close a leg (T12 M10)",
    words="his obvious liquidity taken, a break of structure, the first gap of the breaking move, price back to its "
          "near edge (strategy.md section 9, Type 1) - the moments a hedged leg may close",
    timeframe="5m", directions=["buy", "sell"],
    steps=[
        dict(id="sweep", what="price takes his obvious liquidity on the hedge side", block="sweep", tf="5m"),
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
    sources=[dict(ref="strategy.md section 9 (Type 1)"), dict(ref="T12 M10, M13; LEARNED #47, #113")],
)


@dataclass
class MarketFacts:
    """The facts from the real market (machine.ideas.compile.Market), under the settings with fingerprint fp."""
    market: Any
    fp: str
    book: engine._Bars
    setups: list[Setup]
    news: np.ndarray
    setup_signals: int = 0
    closers: dict = field(default_factory=dict)   # which of his entries close a leg (M10): entry -> signals, or why not

    @classmethod
    def build(cls, market, fp: str, book: engine._Bars, *, news: np.ndarray | None = None,
              setups: list[Setup] | None = None) -> "MarketFacts":
        """setups None: his entries that may close a leg, by settings which_entries_can_close_a_position (M10):
        "type1" = grab + gap only; "his5" = every entry of his whose words are settled (machine.entries.closers)."""
        n_sig, which = 0, {}
        if setups is None:
            from machine.entries import closers
            got = closers.closing_setups(market, book)
            setups, n_sig, which = got.setups, got.signals, got.which
        if news is None:
            news = news_moments(book.first, book.last, int(settings.get("news_minutes_before")))
        return cls(market, fp, book, setups, news, n_sig, which)

    def dial(self, direction: str, t) -> str:
        from machine.ideas.compile import DIAL_CHARTS
        d = SIGN[direction]
        got = [self.market.bias_at(tf, self.fp, t) for tf in DIAL_CHARTS]
        return "with" if all(v == d for v in got) else "against" if all(v == -d for v in got) else "mixed"

    def next_level(self, side: str, beyond: float, t, max_away: float | None) -> float | None:
        """machine.ideas.compile.next_liquidity, from numpy arrays of the same table (90% of a replay's time went
        into the pandas version). tests/test_hedging.py checks the two give the same level on real bars."""
        i = self.market.closed_by("5m", t)
        if i < 0:
            return None
        L = self._levels()
        price = L["price"] + L["slope"] * (i - L["x0pos"])            # a trendline's price on candle i (L6b.1);
        m = (L["start"] <= i) & (i < L["end"]) & (L["high"] == (side == "high"))   # a flat level's slope is 0
        m &= (price > beyond) if side == "high" else (price < beyond)
        if max_away is not None:
            m &= np.abs(price - beyond) <= max_away
        k = np.flatnonzero(m)
        if not len(k):
            return None
        dist = np.abs(price[k] - beyond)
        if settings.get("the_most_recent_liquidity_comes_first"):
            rank = L["rank"][k] + (L["strong_from"][k] <= i)          # liquidity.recent_first: kind (+ strong), then
            best = np.lexsort((dist, -L["known"][k], -rank))[0]      # the most recent, then the nearest
        else:
            best = int(np.argmin(dist))
        return float(price[k[best]])

    def _levels(self) -> dict[str, np.ndarray]:
        if getattr(self, "_lv", None) is None:
            from machine.ideas.compile import BUILT_KINDS
            life = self.market.liquidity(self.fp).life
            fam = life["kind"].str.split("_", n=1).str[0]
            life = life[fam.isin(BUILT_KINDS) & (life["target_when"] != "higher_timeframe_against")]
            self._lv = {"start": life["start"].to_numpy(np.int64), "end": life["end"].to_numpy(np.int64),
                        "high": (life["side"] == "high").to_numpy(), "price": life["price"].to_numpy(float),
                        "slope": life["slope"].to_numpy(float), "x0pos": life["x0pos"].to_numpy(np.int64),
                        "rank": life["rank"].to_numpy(np.int64), "strong_from": life["strong_from"].to_numpy(np.int64),
                        "known": pd.DatetimeIndex(life["known_at"]).as_unit("ns").asi8}
        return self._lv

    def ruler(self, t) -> float:
        i = self.market.closed_by("5m", t)
        return float(self.market.ruler("5m", self.fp)[i]) if i >= 0 else float("nan")

    def at_all_time_high(self, t) -> bool:
        i = self.market.closed_by("5m", t)
        return bool(self.market.ath5(self.fp)["at_ath"].iloc[i]) if i >= 0 else False


# ================================================================================================================
# one trade
# ================================================================================================================

@dataclass
class Leg:
    direction: str
    open_i: int
    open_px: float
    why_open: str               # "entry" | "hedge" | "news hedge"
    ref: float                  # its current entry (M15, #134), as MT5 shows it: a buy's is the ask (step 4)
    close_i: int | None = None
    close_px: float | None = None
    why_closed: str | None = None
    pnl: costs.PnL | None = None


@dataclass
class HedgedTrade:
    """One filled signal, managed his way. Money only when status is "closed" (the window end: reported apart)."""
    signal_id: str
    idea_id: str
    version_id: str
    parent_idea: str | None
    direction: str
    known_at: pd.Timestamp
    fill_ts: pd.Timestamp
    fill_px: float
    status: str                               # "closed" | "open at end"
    legs: list[Leg]
    story: list[dict]                         # one row per thing that happened: ts, what, price, words
    room: bool = False                        # M06: the first hedge waited at the next liquidity
    at_all_time_high: bool = False            # a buy at the all-time high (M08 waits for L6b.2)
    hedges: int = 0
    closes_while_hedged: int = 0
    trail_steps: int = 0
    news_acts: int = 0
    whole_trade_refusals: int = 0             # M17: a last leg's close refused (the whole trade was red)
    line_moves: int = 0                       # M27: candles the first hedge moved with the trendline
    ambiguous: list[str] = field(default_factory=list)
    exit_ts: pd.Timestamp | None = None
    exit_reason: str | None = None
    gross_usd: float | None = None
    gross_inr: float | None = None
    cost_inr: float | None = None
    net_inr: float | None = None

    @property
    def positions(self) -> int:
        return len(self.legs)

    @property
    def days_open(self) -> float | None:
        return None if self.exit_ts is None else (self.exit_ts - self.fill_ts) / pd.Timedelta(days=1)


class _Manager:
    """One trade's life from its fill (the numbered steps in the module text)."""

    def __init__(self, facts: Facts, s: Signal, f: Filled, R: Rules, fills: np.ndarray, slope: float | None = None):
        self.fills = fills                                          # his setups' fill minutes, sorted
        if s.stop is None or len(s.targets) != 1:
            raise ValueError(f"signal {s.id}: the hedging tester needs a hedge point (the signal's stop) and exactly "
                             "one target (M26: every close is the whole leg)")
        if s.lots != R.lot:
            raise ValueError(f"signal {s.id}: {s.lots} lot; every leg is {R.lot} lot (M05)")
        self.f, self.b, self.s, self.R = facts, facts.book, s, R
        sign = SIGN[s.direction]
        if s.levels == "from_fill":
            hedge_at, target = f.px - sign * s.stop, f.px + sign * s.targets[0]
        else:
            hedge_at, target = float(s.stop), float(s.targets[0])
        self.legs = [Leg(s.direction, f.i, f.px, "entry", self._mt5(s.direction, f.px, f.i))]
        self.t = HedgedTrade(s.id, s.idea_id, s.version_id, s.parent_idea, s.direction, pd.Timestamp(s.known_at),
                             self.ts(f.i), f.px, "open at end", self.legs, [])
        self.H, self.T = hedge_at, target
        self.R_dist = 0.0
        self.hedged_since = 0
        self.had_close = False
        self.trailing, self.trail_since = False, 0
        self.fill = f
        # M27 (#69): a trendline touch entry - its first hedge moves with the line, `slope` price per candle of the
        # book's chart, from the hedge price measured on the candle whose close made the signal known
        self.slope = float(slope) if slope is not None and R.follow == "line" and s.levels == "price" else None
        self.H0 = hedge_at
        self.k0 = int(np.searchsorted(self.b.tf_open, _ns(s.known_at), "left")) - 1
        self.say(f.i, "entry", f.px, f"{s.direction} filled at {f.px:.2f}; hedge point {hedge_at:.2f}, "
                                     f"target {target:.2f}")

    # ---- small helpers ----------------------------------------------------------------------------------------
    def ts(self, m: int) -> pd.Timestamp:
        return pd.Timestamp(int(self.b.t[m]))

    def _mt5(self, direction: str, px: float, m: int) -> float:
        """The price MT5 shows for an order of `direction` recorded at the bid px in minute m: a buy's is the ask."""
        return round(px + float(self.b.sp[m]), 6) if direction == "buy" else px

    def say(self, m: int, what: str, price: float | None, words: str) -> None:
        self.t.story.append({"ts": self.ts(m), "what": what, "price": None if price is None else round(price, 2),
                             "words": words})

    def open_legs(self) -> list[Leg]:
        return [x for x in self.legs if x.close_i is None]

    def money(self, leg: Leg, m: int, px: float) -> costs.PnL:
        return costs.pnl(leg.direction, leg.open_px, px, self.ts(leg.open_i), self.ts(m), self.R.lot,
                         entry_spread_pts=float(self.b.raw_spread[leg.open_i]),
                         exit_spread_pts=float(self.b.raw_spread[m]))

    def booked(self) -> float:
        return sum(x.pnl.net_inr for x in self.legs if x.pnl is not None)

    def close(self, leg: Leg, m: int, px: float, why: str) -> None:
        leg.close_i, leg.close_px, leg.why_closed = m, round(float(px), 6), why
        leg.pnl = self.money(leg, m, leg.close_px)

    def room_for(self, t, r: float) -> float | None:
        return self.R.next_liq_adr * r if math.isfinite(r) else None

    # ---- the whole life ---------------------------------------------------------------------------------------
    def run(self) -> tuple[HedgedTrade, int]:
        f, s, R = self.fill, self.s, self.R
        t0 = self.ts(f.i)
        side = "low" if s.direction == "buy" else "high"            # where a buy's hedge waits: below
        if self.slope is not None:
            self.say(f.i, "line", self.H, "a trendline entry: the hedge sits a few pips past the line and moves with it "
                                          "(M27; strategy.md section 9) - no room (M06): #69, with the day and the "
                                          "hour both with it, 'keep the hedging below the trend line itself'")
        elif R.room == "hd" and self.f.dial(s.direction, t0) == "with":
            r = self.f.ruler(t0)
            lv = self.f.next_level(side, self.H, t0, self.room_for(t0, r))
            why = f"the hour and the day are with the trade: the hedge waits at the next liquidity {lv or 0:.2f}"
            if lv is None and math.isfinite(r):
                lv = self.H - SIGN[s.direction] * R.room_adr * r
                why = (f"the hour and the day are with the trade, no liquidity there: the hedge waits "
                       f"{R.room_adr:g} day's range past the hedge point, at {lv:.2f}")
            if lv is not None:
                self.t.room = True
                self.H = round(float(lv), 6)
                self.say(f.i, "room", self.H, why + " (M06, #111)")
        if s.direction == "buy" and self.f.at_all_time_high(t0):
            self.t.at_all_time_high = True                           # M08 waits for strong zones (L6b.2)
        state, p, first_minute = "one", f.i, not f.at_open
        while True:
            if state == "one":
                state, p, first_minute = self._one(p, first_minute)
            elif state == "two":
                state, p, first_minute = self._two(p)
            else:
                break
        if state == "open":
            self.say(self.b.n - 1, "open at end", None, "the window ended with the trade still open: not counted")
            return self.t, NEVER
        return self._finish_trade(), int(self.b.t[p]) + MINUTE_NS

    # ---- step 3: one leg open ---------------------------------------------------------------------------------
    def _one(self, p: int, first_minute: bool) -> tuple[str, int, bool]:
        b, (leg,) = self.b, self.open_legs()
        buy = leg.direction == "buy"
        if first_minute:
            # the minute the leg opened (after its open): the hedge price counts, a target does not
            if b._stop_hit(p, p + 1, self.H, buy).any():
                if b._target_hit(p, p + 1, self.T, buy).any():
                    self.t.ambiguous.append("the leg opened and its hedge and target were reached in one minute: "
                                            "the hedge counted")
                return self._hedge(p, b._stop_px(p, self.H, buy), "hedge")
            p += 1
        while True:
            if p >= b.n:
                return "open", b.n, False
            lim = b.n
            if self._following():
                lim = self._follow(p, leg)                      # M27: this candle's hedge price, until its end
            m_px = b._first_event(p, self.H, self.T, buy, lim)
            m_news = self._next_news(p)
            m_set, st = self._next_trail_setup(p, leg) if self.trailing else (b.n, None)
            m = min(m_px, m_news, m_set)
            if m >= lim and lim < b.n:
                p = lim                                         # nothing in this candle: the line moves on
                continue
            if m >= b.n:
                return "open", b.n, False
            if m == m_news:                                     # acts at that minute's open, before anything in it
                got = self._news(m, leg)
                if got is not None:
                    return got
                p = m + 1
                continue
            hh = m == m_px and bool(b._stop_hit(m, m + 1, self.H, buy).any())
            th = m == m_px and bool(b._target_hit(m, m + 1, self.T, buy).any())
            ch = m == m_set
            if hh:
                if th or ch:
                    self.t.ambiguous.append("the hedge price and a target / close in one minute: the hedge counted")
                return self._hedge(m, b._stop_px(m, self.H, buy), "hedge")
            if th:
                if ch:
                    self.t.ambiguous.append("the target and an opposite setup in one minute: the target counted")
                got = self._at_target(m, leg)
            else:
                got = self._last_close(m, leg, st.fill_px, "an opposite setup formed past the target "
                                                           f"({st.id}) - closed at its fill price (M23)")
            if got is not None:
                return got
            p = m + 1

    def _following(self) -> bool:
        """M27: the first hedge still follows the line - a trendline entry, no hedge yet, no leg closed yet."""
        return self.slope is not None and self.t.hedges == 0 and not self.had_close

    def _follow(self, p: int, leg: Leg) -> int:
        """The hedge price for the candle minute p is in: the line's price there, the same distance past it as at the
        entry - only tighter, and never onto a price the market is already past at that minute's open (the old
        manager's rule). Returns the first minute of the next candle (where it is looked at again)."""
        b = self.b
        k = int(np.searchsorted(b.tf_open, b.t[p], "right")) - 1
        new = round(self.H0 + self.slope * (k - self.k0), 6)
        if leg.direction == "buy":
            ok = new > self.H + EPS and float(b.o[p]) > new + EPS          # a buy's hedge SELLS: the bid
        else:
            ok = new < self.H - EPS and float(b.o[p] + b.sp[p]) < new - EPS  # a sell's hedge BUYS: the ask
        if ok:
            if not self.t.line_moves:
                self.say(p, "line", new, f"the hedge moves with the line: {self.H:.2f} -> {new:.2f} (M27)")
            self.t.line_moves += 1
            self.H = new
        return int(np.searchsorted(b.t, b.tf_open[k + 1], "left")) if k + 1 < len(b.tf_open) else b.n

    def _next_news(self, p: int) -> int:
        b, news = self.b, self.f.news
        if self.R.news == "off" or not len(news):
            return b.n
        k = int(np.searchsorted(news[:, 0], b.t[p], "left"))
        while k < len(news):
            m = int(np.searchsorted(b.t, news[k, 0], "left"))
            if m >= b.n:
                return b.n
            if b.t[m] < news[k, 1]:                             # a bar before the release: act there
                return m
            k += 1
        return b.n

    def _next_trail_setup(self, p: int, leg: Leg) -> tuple[int, Setup | None]:
        want = OTHER[leg.direction]
        ss = self.f.setups
        for k in range(self._first_setup(p), len(ss)):
            st = ss[k]
            if st.direction == want and st.known >= self.trail_since:
                return st.fill_i, st
        return self.b.n, None

    def _first_setup(self, p: int) -> int:
        """The first of his setups that fills in minute p or later."""
        return int(np.searchsorted(self.fills, p, "left"))

    def _news(self, m: int, leg: Leg) -> tuple[str, int, bool] | None:
        px = float(self.b.o[m])
        self.t.news_acts += 1
        net = self.money(leg, m, px).net_inr
        if net > 0:                                     # a leg in profit never meets M17's "ends red"
            return self._last_close(m, leg, px, f"news in {self.R.news_lead_min} minutes and the leg is in profit "
                                                f"(Rs {net:,.0f} after its cost): closed where it is (M28, #110)")
        return self._hedge(m, px, "news hedge")

    def _hedge(self, m: int, px: float, why: str) -> tuple[str, int, bool]:
        (leg,) = self.open_legs()
        d = OTHER[leg.direction]
        h = Leg(d, m, round(float(px), 6), why, self._mt5(d, round(float(px), 6), m))
        self.legs.append(h)
        self.R_dist = abs(leg.ref - h.ref)
        self.hedged_since = int(self.b.t[m])
        self.t.hedges += 1
        self.trailing = False
        words = (f"news in {self.R.news_lead_min} minutes and the leg is not in profit: hedged at the price now"
                 if why == "news hedge" else "price reached the hedge price: the other side opened")
        self.say(m, why, h.open_px, f"{words}; nothing closes until price is {self.R.min_rr:g} x "
                                    f"${self.R_dist:.2f} = ${self.R.min_rr * self.R_dist:.2f} past a leg's current "
                                    f"entry (M12)")
        return "two", m + 1, False

    def _at_target(self, m: int, leg: Leg) -> tuple[str, int, bool] | None:
        b, R, buy = self.b, self.R, leg.direction == "buy"
        t = self.ts(m)
        if R.trail == "hd" and self.t.trail_steps < R.trail_max and self.f.dial(leg.direction, t) == "with":
            nx = self.f.next_level("high" if buy else "low", self.T, t, self.room_for(t, self.f.ruler(t)))
            if nx is not None:
                self.say(m, "trail", self.T, f"target {self.T:.2f} reached with the hour and the day on side: "
                                             f"holding on to the next liquidity {nx:.2f} (M23, M25)")
                self.T = round(float(nx), 6)
                self.t.trail_steps += 1
                if not self.trailing:
                    self.trailing, self.trail_since = True, int(b.t[m])
                return None
        return self._last_close(m, leg, b._target_px(m, self.T, buy), f"target {self.T:.2f} reached")

    def _last_close(self, m: int, leg: Leg, px: float, why: str) -> tuple[str, int, bool] | None:
        """Close the last leg - unless M17 says no: it would end red and the whole trade is red too."""
        if self.had_close and self.R.whole == "trade":
            leg_net = self.money(leg, m, px).net_inr
            whole = self.booked() + leg_net
            if leg_net < 0 and whole < 0:
                t = self.ts(m)
                y = SIGN[leg.direction]
                be = leg.open_px + y * (config.COST_INR_PER_POSITION - self.booked()) / costs.usd_inr(t)
                be = (math.ceil(be * 100 - 1e-9) if y > 0 else math.floor(be * 100 + 1e-9)) / 100   # a whole cent
                nx = self.f.next_level("high" if y > 0 else "low", be - y * 0.001, t,
                                       self.room_for(t, self.f.ruler(t)))
                self.t.whole_trade_refusals += 1
                old = self.T
                self.T = round(float(nx if nx is not None else be), 6)
                self.say(m, "not yet", px, f"{why}, but this leg would end at Rs {leg_net:,.0f} and the whole trade "
                                           f"at Rs {whole:,.0f}: not closed (M17, #112); target {old:.2f} -> "
                                           f"{self.T:.2f} (the whole trade breaks even at {be:.2f})")
                return None
        self.close(leg, m, px, why)
        self.say(m, "close", leg.close_px, f"{why}: the {leg.direction} closed, Rs {leg.pnl.net_inr:,.0f}")
        return "end", m, False

    # ---- step 4: two legs open --------------------------------------------------------------------------------
    def _two(self, p: int) -> tuple[str, int, bool]:
        R, ss = self.R, self.f.setups
        legs = {x.direction: x for x in self.open_legs()}
        need = R.min_rr * self.R_dist
        for k in range(self._first_setup(p), len(ss)):
            st = ss[k]
            if st.known < self.hedged_since:
                continue
            shut = legs[OTHER[st.direction]]                   # the leg the setup points against (M11)
            if SIGN[shut.direction] * (self._mt5(st.direction, st.fill_px, st.fill_i) - shut.ref) < need - EPS:
                continue                                        # inside the no-close zone (M12)
            return self._after_close(st, shut, legs[st.direction])
        return "open", self.b.n, False

    def _after_close(self, st: Setup, shut: Leg, left: Leg) -> tuple[str, int, bool]:
        R, m = self.R, st.fill_i
        at = self._mt5(st.direction, st.fill_px, m)                 # the setup's fill as MT5 shows it
        moved = SIGN[shut.direction] * (at - shut.ref)
        self.close(shut, m, st.fill_px, f"a {st.direction} setup ({st.id}) filled {moved:.2f} past this leg's "
                                        f"current entry {shut.ref:.2f} (needed {R.min_rr * self.R_dist:.2f})")
        self.had_close = True
        self.t.closes_while_hedged += 1
        self.say(m, "close", st.fill_px, f"a {st.direction} setup filled: the {shut.direction} closed at its fill "
                                         f"price, Rs {shut.pnl.net_inr:,.0f} (M10, M11, M12)")
        # step 5: the leg left is that setup's own trade (M15), with a proper target (M16)
        y, t = SIGN[left.direction], self.ts(m)
        left.ref = at
        self.H = st.hedge
        if R.leftover_target == "his124":
            r2 = abs(at - st.hedge)
            far = max(st.target, at + y * R.min_rr * r2, key=lambda v: y * v)
            nx = self.f.next_level("high" if y > 0 else "low", far - y * 0.001, t,
                                   self.room_for(t, self.f.ruler(t)))
            self.T = round(float(nx if nx is not None else far), 6)
        else:
            self.T = st.target
        self.say(m, "new trade", st.fill_px, f"the {left.direction} left is a new trade at {at:.2f}: its "
                                             f"fixed hedge {self.H:.2f}, its target {self.T:.2f} (M15, M16)")
        if not st.at_open:
            return "one", m, True                               # its hedge price may come in this same minute
        return "one", m, False

    # ---- the end ----------------------------------------------------------------------------------------------
    def _finish_trade(self) -> HedgedTrade:
        t = self.t
        last = max(self.legs, key=lambda x: x.close_i)
        pn = [x.pnl for x in self.legs]
        t.status = "closed"
        t.exit_ts = self.ts(last.close_i)
        t.exit_reason = _reason(last.why_closed)
        t.gross_usd = sum(x.gross_usd for x in pn)
        t.gross_inr = sum(x.gross_inr for x in pn)
        t.cost_inr = sum(x.cost_inr for x in pn)
        t.net_inr = sum(x.net_inr for x in pn)
        return t


def _reason(why: str) -> str:
    if why.startswith("target"):
        return "target"
    if why.startswith("news"):
        return "news, in profit"
    if why.startswith("an opposite setup"):
        return "opposite setup after trailing"
    return "other"


def _ns(t) -> int:
    return pd.Timestamp(t).as_unit("ns").value


# ================================================================================================================
# many trades: one at a time
# ================================================================================================================

@dataclass
class HedgeReplay:
    trades: list[HedgedTrade]
    skipped: list[str]                          # signals that came while a trade was on (M18)
    not_filled: list[tuple[str, str]]           # (signal id, "expired" | "cancelled" | "target reached first" | ...)
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    tf: str
    rules: Rules
    signals: int = 0
    setups: int = 0                             # his setups that could close a leg (filled ones)


def replay(signals: list[Signal], facts: Facts, tf: str = "5m", rules: Rules | None = None, *,
           moves: dict[str, float] | None = None, cancel_by: dict[str, int] | None = None) -> HedgeReplay:
    """Every signal, in known_at order, one trade at a time (M18) - a waiting limit is cancelled by a newer setup
    (M39) - each filled one managed his way. The window is the facts' book: nothing after its last bar exists.
    moves: {signal id: the trendline's slope, price per candle of tf} - its first hedge moves with the line (M27).
    cancel_by: {signal id: moment ns} - a check before the fill cancels the order then (engine.replay's, L19b)."""
    R = rules or Rules.now()
    moves, cancel_by = moves or {}, cancel_by or {}
    b = facts.book
    if b.tf != tf:
        raise ValueError(f"the bars were read for {b.tf} bars, not {tf}")
    ids = [s.id for s in signals]
    if len(set(ids)) != len(ids):
        raise ValueError("two signals share one id")
    out = HedgeReplay([], [], [], b.first, b.last, tf, R, len(signals), len(facts.setups))
    fills = np.array([x.fill_i for x in facts.setups], dtype=np.int64)
    if len(fills) > 1 and (np.diff(fills) < 0).any():
        raise ValueError("facts.setups must be sorted by the minute they fill (setups_from does it)")
    rows: list[list] = []                       # [signal, trade or None, why not filled]
    held = None                                 # (known ns, release ns, waiting until ns or None, row)
    for s in sorted(signals, key=lambda s: s.known_at):
        t = _ns(s.known_at)
        if t < b.t[0] or t > b.t[-1] + MINUTE_NS:
            raise ValueError(f"signal {s.id} is known at {s.known_at}, outside the window {b.first} .. {b.last}")
        if held is not None and held[1] > t:
            if held[2] is not None and t < held[2] and held[0] < t:
                rows[held[3]][1:] = [None, "cancelled"]        # "Cancel and take new" (M39)
                held = None
            else:
                out.skipped.append(s.id)
                continue
        got = b.fill(s, t, cancel_by.get(s.id))
        if isinstance(got, Missed):
            held = (t, got.release, got.wait_until, len(rows))
            rows.append([s, None, got.why])
            continue
        trade, release = _Manager(facts, s, got, R, fills, moves.get(s.id)).run()
        held = (t, release, got.wait_until, len(rows))
        rows.append([s, trade, ""])
    for s, trade, why in rows:
        if trade is not None:
            out.trades.append(trade)
        else:
            out.not_filled.append((s.id, why))
    return out


# ================================================================================================================
# the summary and the table
# ================================================================================================================

def summarize(rep: HedgeReplay) -> dict:
    closed = [t for t in rep.trades if t.status == "closed"]
    open_end = [t for t in rep.trades if t.status == "open at end"]
    n = len(closed)
    years = range(rep.first_bar.year, rep.last_bar.year + 1)
    per_year = {y: {"trades": 0, "net_inr": 0.0, "positions": 0} for y in years}
    for t in closed:
        y = per_year[t.exit_ts.year]
        y["trades"] += 1
        y["net_inr"] += t.net_inr
        y["positions"] += t.positions
    worst = min(per_year, key=lambda y: per_year[y]["net_inr"])
    net = sum(t.net_inr for t in closed)
    days = np.array([t.days_open for t in closed]) if closed else np.array([])
    unreal = []
    for t in open_end:                           # what the open trades would show at the last bar: NOT counted
        booked = sum(x.pnl.net_inr for x in t.legs if x.pnl is not None)
        unreal.append(booked)
    missed = Counter(why for _, why in rep.not_filled)
    return {
        "signals": rep.signals, "his_setups_that_can_close": rep.setups,
        "trades": n, "won": sum(t.net_inr > 0 for t in closed),
        "win_pct": round(100 * sum(t.net_inr > 0 for t in closed) / n, 1) if n else None,
        "net_inr_total": round(net, 2), "net_inr_per_trade": round(net / n, 2) if n else None,
        "gross_inr_total": round(sum(t.gross_inr for t in closed), 2),
        "cost_inr_total": round(sum(t.cost_inr for t in closed), 2),
        "positions": sum(t.positions for t in closed),
        "per_year": {str(y): {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}
                     for y, d in per_year.items()},
        "worst_year": {"year": worst, "net_inr": round(per_year[worst]["net_inr"], 2),
                       "trades": per_year[worst]["trades"]},
        "hedged_trades": sum(t.hedges > 0 for t in closed),
        "hedges": sum(t.hedges for t in closed),
        "closes_while_hedged": sum(t.closes_while_hedged for t in closed),
        "room_given": sum(t.room for t in closed),
        "trail_steps": sum(t.trail_steps for t in closed),
        "news_acts": sum(t.news_acts for t in closed),
        "whole_trade_refusals": sum(t.whole_trade_refusals for t in closed),
        "line_trades": sum(t.line_moves > 0 for t in closed),
        "ended_by": dict(Counter(t.exit_reason for t in closed)),
        "ambiguous": sum(bool(t.ambiguous) for t in closed),
        "at_all_time_high_buys": sum(t.at_all_time_high for t in rep.trades),
        "days_open_median": round(float(np.median(days)), 1) if n else None,
        "days_open_max": round(float(days.max()), 1) if n else None,
        "setups_skipped": len(rep.skipped),
        "not_filled": dict(missed),
        "open_at_end": len(open_end),
        "open_at_end_ids": [t.signal_id for t in open_end],
        "open_at_end_booked_inr_not_counted": round(sum(unreal), 2),
    }


def summary_text(s: dict, label: str = "") -> str:
    inr = costs.inr
    lines = [f"HEDGING {label}".rstrip(),
             "  his management, version 1 (T12, plan/work/management-decisions.md) · 0.01 lot a leg · "
             "Rs 40 per position all-in · bid prices · MT5 server time"]
    if s["trades"]:
        lines.append(f"Trades: {s['trades']} · won {s['won']} ({s['win_pct']:.0f}%) · after cost "
                     f"{inr(s['net_inr_total'])} · {inr(s['net_inr_per_trade'])} a trade · positions "
                     f"{s['positions']} (cost {inr(s['cost_inr_total'])})")
    else:
        lines.append("Trades: 0 - nothing to count")
    years = " · ".join(f"{y} {inr(v['net_inr'])} ({v['trades']})" for y, v in s["per_year"].items())
    wy = s["worst_year"]
    lines += [
        f"Year by year (trades in brackets; a trade counts in the year its last leg closed): {years}",
        f"Worst year: {wy['year']} {inr(wy['net_inr'])} ({wy['trades']} trades)",
        f"Hedged: {s['hedged_trades']} of {s['trades']} trades · {s['hedges']} hedges · {s['closes_while_hedged']} "
        f"legs closed by his setups while hedged · room given {s['room_given']} · trailing steps {s['trail_steps']} "
        f"· news acted {s['news_acts']} · last closes refused by the whole-trade rule {s['whole_trade_refusals']}"
        + (f" · trades whose hedge moved with the trendline {s['line_trades']}" if s.get("line_trades") else ""),
        f"Ended by: {' · '.join(f'{k} {v}' for k, v in s['ended_by'].items()) or '-'} · days open: median "
        f"{s['days_open_median']}, longest {s['days_open_max']}",
        f"Setups skipped (a trade was on): {s['setups_skipped']} · not filled: {s['not_filled']} · "
        f"same-minute cases (the worse counted): {s['ambiguous']}",
        f"OPEN AT END (not counted in Rs): {s['open_at_end']} · buys at the all-time high (M08, counted only - not "
        f"built): {s['at_all_time_high_buys']}",
    ]
    return "\n".join(lines)


def trades_table(trades: list[HedgedTrade]) -> pd.DataFrame:
    rows = [{
        "signal": t.signal_id, "direction": t.direction, "known_at": t.known_at, "fill_time": t.fill_ts,
        "fill_px": t.fill_px, "status": t.status, "exit_time": t.exit_ts, "ended_by": t.exit_reason,
        "positions": t.positions, "hedges": t.hedges, "closes_while_hedged": t.closes_while_hedged,
        "room": t.room, "trail_steps": t.trail_steps, "news_acts": t.news_acts, "line_moves": t.line_moves,
        "days_open": None if t.days_open is None else round(t.days_open, 2),
        "gold_usd": t.gross_usd, "inr_before_cost": t.gross_inr, "cost_inr": t.cost_inr, "net_inr": t.net_inr,
        "ambiguous": "; ".join(t.ambiguous),
    } for t in trades]
    return pd.DataFrame(rows)
