"""The compiler (L7a): an idea form -> Signals for the tester, made from the blocks only.

    from machine.ideas.compile import Market, compile_idea
    market = Market.load("2016", "2020")            # the bars (5m, 1h, 1d, 1w), read once; every block made once
    out = compile_idea(form, market, readings={"F3": "A"})
    out.signals                                     # machine.tester Signals: one version of one idea
    print(out.text())                               # how many setups reached each step, and why the rest were dropped

"Blocks only" (the AI writes rules; code judges - CLAUDE.md rule 11): a form names blocks from BLOCKS, rules from
DO_NOT_ENTER and filters from FILTERS, and nothing else - no code, no formula. Each of them is made of the blocks in
machine/blocks/ (its `uses` names them). A name that does not exist yet stops the compile with MissingBlockError (L7b
turns that into BLOCKED and a block request).

How a form becomes signals, for each direction it trades:
  1. The first step's block lists every place the idea could start (every sweep, every higher high ...).
  2. From each one, the next steps are looked for IN ORDER. A step may be found only on its own timeframe's bars that
     come after the step before it became known, and at most its max wait later (max_wait_bars bars of its timeframe:
     bars t+1 ... t+N, the tester's own count). Not found in time -> the idea is dropped there, and the reason is
     counted. A step with wait 0 is a check made at that same moment.
  3. When the last step is known, the signal is made AT THAT MOMENT, from what the steps found: the entry (a limit at
     a price a step found, waiting max_wait_bars bars of the idea's timeframe; or the next bar's open), the stop or
     hedge point, the targets (a ratio, a price, or the next liquidity alive then), then the do-not-enter rules and
     the filters. Nothing after that moment is read. Prices are rounded to the chart's 1-cent step (config.POINT), as
     MT5 does with an order price.
  4. Two starts that end in the same entry (same direction, moment and price) are one signal, the earlier start's.
  The 1-minute check (step one_minute, L6b.4) may answer "wait": a line against the entry must be taken first; that
  take becomes the idea's liquidity grab, and the steps after the sweep step are looked for again from it (#111).
Every signal passes the cut-off-the-future test (tests/test_ideas.py): cutting the bars off at any moment leaves every
signal known by then exactly the same.

What it refuses, before anything runs:
  - a change to one of his FIXED rules (machine/rules.py), anywhere in the form - its settings, a step's numbers, the
    hedge management - and an entry at a gap's far side (#65): FixedRuleError, with his words;
  - a name that is no block / rule / filter (MissingBlockError), a setting that does not exist (SettingError);
  - a part that points at a step that comes later, or at something that step does not find (CompileError).
A HEDGE is tested by this tester (L3) as a plain stop at the hedge point, and says so; his management rules are
carried in the result for the hedging tester (L19a).
"""
from __future__ import annotations

import math
import re
import typing
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from machine import config, guards, rules, settings
from machine.blocks import adr as ADR
from machine.blocks import bias as HB
from machine.blocks import fib as FIB
from machine.blocks import fvg as FVG
from machine.blocks import hammer as HM
from machine.blocks import liquidity as LQ
from machine.blocks import one_minute as OM
from machine.blocks import patterns as PT
from machine.blocks import premium_discount as PD
from machine.blocks import sessions as SES
from machine.blocks import structure as ST
from machine.blocks import trendlines as TL
from machine.blocks import zones as ZN
from machine.blocks._base import TF_MINUTES, bar_close, check_bars, empty, first_hit
from machine.data import bars as BARS
from machine.data import clock
from machine.ideas.form import NO_LIMIT, REF, TF, Idea, apply_changes, readable, with_readings
from machine.tester.models import Signal

NO_LIMIT_BARS = 10**9            # "no limit": the tester's deadline falls past the window, so the order waits to its end


class CompileError(ValueError):
    """The form cannot be compiled. The message says where and why, one line each."""


class MissingBlockError(CompileError):
    """The form needs a block, rule or filter that does not exist yet (L7b: the idea becomes BLOCKED)."""

    def __init__(self, idea_id: str, missing: list[tuple[str, str, str]]):
        self.idea_id = idea_id
        self.missing = missing                   # (name, where in the form, the prompt that builds it)
        lines = [f"'{n}' ({w}) - {p}" for n, w, p in missing]
        super().__init__(f"{idea_id} needs what does not exist yet:\n" + "\n".join(f"  - {x}" for x in lines))


# Names his ideas use that later prompts build (plan/likhith.md). Anything else: the reusable prompt L6-new.
# L6b.3 built bias, higher_timeframe, all_time_high, hammer and premium_discount (steps, rules and filters below).
# L6b.1 built trendlines (a kind of liquidity), trendline_touch, triangle, false_breakout, wedge and line_in_path. Two
# of their words are still his to answer; a step that needs one waits for the weekly picture batch (L9b), unless the
# form itself names the number (a reading or a version) - the machine never picks it (likhith.md R13).
# L6b.2 built his strong zones: the steps zone_entry, zone_break and zone, the target kind 'zone' and the rule
# unfilled_gap_past_hedge (#87). A zone is not swept like a level (strategy.md section 1: "zones follow their own rule").
# L6b.4 built the step one_minute (the 1-minute check before a 5-minute gap entry, #111).
# L6-new (30 Sep, idea #3 "sell when the London high is taken") built WHICH session a session level comes from: the
# param `which_session` of the steps sweep and level_waiting (liquidity.of_session). The filter 'session' is another
# thing: it reads the clock at the setup's own moment, not the level. (The param cannot be called `sessions`: that
# is the old name of his FIXED rule what_a_session_high_or_low_is, and a form that writes it is refused.)
# L19b built the second gap of a move (#124: "second is only when it is VERY STRONG ..., or else it should wait for
# the first"): settings which_gap_of_the_move = 'second' with the_second_gap_needs_the_day (the step gap_of_the_move).
# L6-new (2 Oct, idea #4 "sell when price breaks below the green ADR zone") built the step adr_zone_break: a 5-minute
# CLOSE past his ADR zone the trade's way (#76). premium_discount is price AT the zone; a talker without this step had
# written "breaks below" as "at the zone" (a dry run, 2 Oct) - a different trade, and the old band filter's key.
PLANNED = {
    "trendline": "L6b.1 built it under other names - the form must name one: kinds ['trendline'] (a sweep of a "
                 "trendline, a target on one), the step trendline_touch (his Type 2), the steps triangle / wedge / "
                 "false_breakout, the rule line_in_path",
    "steep": "L9b (the weekly picture batch: his answer to the word 'steep', four-boxes B2-01 - "
             "plan/work/question-queue/B2-01-steep.md; settings trendline_min_steepness_adr waits for it)",
    "big_move": "L9b (the weekly picture batch: his answer to the words 'big move', four-boxes B2-02 - "
                "plan/work/question-queue/B2-02-big-move.md; settings big_move_adr waits for it)",
    "halfway": "L19c (the halfway reversal is management - no hedge, re-enter lower - for the hedging tester; its "
               "detector is machine/blocks/halfway.py, L6b.3; the word waits for his answer, four-boxes B2-05)",
    "news": "L12 (the news calendar)",
}


def _planned(name: str) -> str:
    return PLANNED.get(name, "no prompt builds it yet: the reusable prompt L6-new (add a missing block)")


# ================================================================================================== the market

@dataclass
class _Liquidity:
    book: pd.DataFrame
    sweeps: pd.DataFrame
    life: pd.DataFrame
    by_level: dict


class Market:
    """The bars one compile reads (server clock), and each block's table, made once per settings."""

    def __init__(self, bars: Mapping[str, pd.DataFrame], ath_before: float | None = None):
        """ath_before: the highest price before these bars (the all-time high starts from it). None = gold's
        September 2011 record for bars that start in January 2016, else read from the 2016 daily bars before them."""
        if not bars:
            raise ValueError("a market needs bars of at least one timeframe")
        self._bars: dict[str, pd.DataFrame] = {}
        for tf, b in bars.items():
            self._bars[tf] = check_bars(b, tf)              # refuses the sealed years (LEARNED #96)
        self._cache: dict = {}
        self._ath_before = ath_before
        self._earlier_high: float | None = None

    @classmethod
    def load(cls, start=None, end=None, tfs: Iterable[str] = ("5m", "1h", "1d", "1w")) -> "Market":
        """Real bars from start to end (machine.data.bars.load_bars; the sealed years are refused)."""
        return cls({tf: BARS.load_bars(tf, start, end) for tf in tfs})

    @classmethod
    def from_m1(cls, m1: pd.DataFrame, tfs: Iterable[str] = ("5m", "1h", "1d", "1w"),
                ath_before: float | None = None) -> "Market":
        """Bars made from 1-minute bars (hand-made, or cut at a moment for the no-future test): the MT5 way
        (machine.data.bars.aggregate), with the utc column the session block needs."""
        guards.check_dates(m1)
        out = {}
        for tf in tfs:
            b = m1.copy() if tf == "1m" else BARS.aggregate(m1, tf)
            b["utc"] = clock.server_to_utc(pd.DatetimeIndex(b.index))
            out[tf] = b
        return cls(out, ath_before)

    @property
    def tfs(self) -> tuple[str, ...]:
        return tuple(self._bars)

    def has(self, tf: str) -> bool:
        return tf in self._bars

    def bars(self, tf: str) -> pd.DataFrame:
        if tf not in self._bars:
            raise CompileError(f"this needs {tf} bars; the market has {', '.join(self._bars)}")
        return self._bars[tf]

    def finest(self) -> str:
        return min(self._bars, key=lambda tf: TF_MINUTES[tf])

    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        b = self._bars[self.finest()]
        return b.index[0], b.index[-1]

    def cached(self, key: tuple, fp: str, make):
        """A block's table under the settings with fingerprint fp - never reused under other settings (B19)."""
        k = (key, fp)
        if k not in self._cache:
            self._cache[k] = make()
        return self._cache[k]

    # ---- shared tables ----------------------------------------------------------------------------------------
    def opens(self, tf: str) -> np.ndarray:
        return self.cached(("opens", tf), "", lambda: self.bars(tf).index.as_unit("ns").asi8)

    def closes(self, tf: str) -> np.ndarray:
        return self.cached(("closes", tf), "", lambda: bar_close(self.bars(tf).index, tf).as_unit("ns").asi8)

    def closed_by(self, tf: str, t) -> int:
        """The position of the last bar of tf that has CLOSED by moment t (-1 if none)."""
        return int(np.searchsorted(self.closes(tf), _ns(t), "right")) - 1

    def opens_from(self, tf: str, t) -> int:
        """The position of the first bar of tf that opens at or after moment t (len if none)."""
        return int(np.searchsorted(self.opens(tf), _ns(t), "left"))

    def arrays(self, tf: str) -> dict[str, np.ndarray]:
        return self.cached(("arrays", tf), "", lambda: {c: self.bars(tf)[c].to_numpy(dtype=float)
                                                       for c in ("open", "high", "low", "close")})

    def ruler(self, tf: str, fp: str) -> np.ndarray:
        """A day's range on every bar of tf (his 10-day ADR on broker days - machine/blocks/adr.py)."""
        return self.cached(("ruler", tf), fp, lambda: ADR.adr_bars(self.bars(tf), tf)["ruler"].to_numpy(float))

    def liquidity(self, fp: str) -> _Liquidity:
        """His liquidity on the 5-minute chart (machine/blocks/liquidity.py): the book (obvious liquidity + swing
        roles, with the hour / day / week charts the market has), the sweeps, the lifetimes."""
        def make():
            m5 = self.bars("5m")
            book = LQ.book(m5, self._bars.get("1h"), self._bars.get("1d"), self._bars.get("1w"))
            sw = LQ.sweeps(m5, book)
            life = LQ.lifetimes(m5, book, sw)
            return _Liquidity(book, sw, life, {lid: k for k, lid in enumerate(sw["level_id"])})
        return self.cached(("liquidity",), fp, make)

    # ---- L6b.3: his bias, the all-time high, the ADR zones, the hammer ------------------------------------------
    def ath_before(self) -> float:
        """The highest price before this market's bars: gold's September 2011 record (settings
        "all_time_high_before_2016"), and the 2016 daily bars before a later start (read once)."""
        seed = float(settings.get("all_time_high_before_2016"))
        if self._ath_before is not None:
            return float(self._ath_before)
        first = self.bars(self.finest()).index[0]
        if first <= HB.FIRST_DAY:
            return seed
        if self._earlier_high is None:
            earlier = BARS.load_bars("1d", "2016-01-01", first - pd.Timedelta(days=1))
            self._earlier_high = float(earlier["high"].max()) if len(earlier) else float("-inf")
        return max(seed, self._earlier_high)

    def bias(self, tf: str, fp: str) -> pd.DataFrame:
        """His bias on the tf chart's own candles (machine/blocks/bias.py): one row per candle, known at its close."""
        return self.cached(("bias", tf), fp, lambda: HB.htf_bias(
            self.bars(tf), tf, ath_before=self.ath_before() if tf in HB.ATH_CHARTS else None))

    def bias_at(self, tf: str, fp: str, t) -> int:
        """The tf chart's bias known at moment t (its last candle closed by then): +1 up, -1 down, 0 neutral."""
        def make():
            tab = self.bias(tf, fp)
            return pd.DatetimeIndex(tab["known_at"]).as_unit("ns").asi8, tab["bias"].to_numpy()
        known, vals = self.cached(("bias_arrays", tf), fp, make)
        k = int(np.searchsorted(known, _ns(t), "right")) - 1
        return int(vals[k]) if k >= 0 else HB.NEUTRAL

    def ath5(self, fp: str) -> pd.DataFrame:
        """The real all-time high through every 5-minute bar (machine/blocks/bias.py all_time_high)."""
        return self.cached(("ath", "5m"), fp, lambda: HB.all_time_high(self.bars("5m"), "5m", self.ath_before()))

    def adr_zones(self, fp: str, measure: str) -> pd.DataFrame:
        """His ADR zones and where premium / discount start, on every 5-minute bar (premium_discount.zones)."""
        return self.cached(("adr_zones", measure), fp, lambda: PD.zones(self.bars("5m"), "5m", measure))

    def hammers(self, fp: str) -> pd.DataFrame:
        """His hammer on every 5-minute bar (machine/blocks/hammer.py)."""
        return self.cached(("hammers", "5m"), fp, lambda: HM.hammers(self.bars("5m"), "5m",
                                                                       self.ruler("5m", fp)))

    # ---- L6b.1: trendlines, triangles, wedges, false breakouts ----------------------------------------------------
    def lines(self, tf: str, fp: str) -> pd.DataFrame:
        """Every 2-touch line and trendline of the tf chart (machine/blocks/trendlines.py)."""
        return self.cached(("lines", tf), fp, lambda: TL.lines(self.bars(tf), tf, self.ruler(tf, fp)))

    def lines5(self, fp: str) -> pd.DataFrame:
        """Every line of the 5-minute and hour charts the market has (each once, from when it has 2 touches), on the
        5-minute candles (trendlines.on_m5) - what a line in the path (#69) is looked for in."""
        def make():
            parts = [self.lines(tf, fp) for tf in LQ.TRENDLINE_CHARTS if self.has(tf)]
            t = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
                else empty(TL.COLUMNS)
            t = t.sort_values("known_at", kind="stable").drop_duplicates("line_id")
            return TL.on_m5(t, self.bars("5m"))
        return self.cached(("lines5",), fp, make)

    def touch_entries(self, tf: str, fp: str) -> pd.DataFrame:
        return self.cached(("touch_entries", tf), fp, lambda: TL.touch_entries(
            self.bars(tf), tf, self.ruler(tf, fp), self.lines(tf, fp)))

    def triangles(self, tf: str, fp: str, big_move: float) -> pd.DataFrame:
        return self.cached(("triangles", tf, big_move), fp, lambda: PT.triangles(
            self.bars(tf), tf, self.ruler(tf, fp), self.lines(tf, fp), big_move_adr=float(big_move)))

    def triangle_breaks(self, tf: str, fp: str, big_move: float) -> pd.DataFrame:
        return self.cached(("triangle_breaks", tf, big_move), fp, lambda: PT.triangle_breaks(
            self.bars(tf), tf, self.ruler(tf, fp), self.triangles(tf, fp, big_move), self.lines(tf, fp)))

    def wedges(self, tf: str, fp: str) -> pd.DataFrame:
        return self.cached(("wedges", tf), fp, lambda: PT.wedges(self.bars(tf), tf, self.ruler(tf, fp),
                                                                 self.lines(tf, fp)))

    def false_breaks(self, fp: str) -> pd.DataFrame:
        def make():
            L = self.liquidity(fp)
            return PT.false_breaks(self.bars("5m"), L.book, L.sweeps)
        return self.cached(("false_breaks",), fp, make)

    # ---- L6b.2: his strong zones ------------------------------------------------------------------------------------
    def zones(self, fp: str) -> pd.DataFrame:
        """His strong zones (machine/blocks/zones.py): time at price on the 5-minute and hour candles, the tiers
        settings zone_tiers hands on."""
        return self.cached(("zones",), fp, lambda: ZN.strong_zones(self.bars("5m"), self.bars("1h")))

    def zone_life(self, fp: str) -> pd.DataFrame:
        """Each zone's life on the 5-minute chart (touches, roles, the end) - read only through zones.alive()."""
        return self.cached(("zone_life",), fp, lambda: ZN.lives(self.bars("5m"), self.zones(fp)))

    def zone_breaks(self, fp: str) -> pd.DataFrame:
        return self.cached(("zone_breaks",), fp, lambda: ZN.breaks(self.bars("5m"), self.zone_life(fp)))

    def zone_entries(self, fp: str) -> pd.DataFrame:
        """His zone entry (zones.entries): every take arms an order; the tester keeps one at a time (#145)."""
        return self.cached(("zone_entries",), fp, lambda: ZN.entries(self.bars("5m"), self.zone_life(fp),
                                                                     self.liquidity(fp).sweeps))

    # ---- L6b.4: the 1-minute layer ----------------------------------------------------------------------------------
    def book1(self, fp: str) -> pd.DataFrame:
        """Every 1-minute line on the 1-minute candles (machine/blocks/one_minute.py book) - what the 1-minute check
        reads. Made once per settings for the whole market (2019: about 66,000 lines, 10 seconds)."""
        return self.cached(("book1",), fp, lambda: OM.book(self.bars("1m"), self.lines("1m", fp),
                                                           self.ruler("1m", fp)))


def _ns(t) -> int:
    return pd.Timestamp(t).as_unit("ns").value


# ================================================================================================== one run

@dataclass
class _Run:
    """One direction of one compile."""
    market: Market
    form: Idea
    direction: str
    fp: str
    steps: dict[str, Any] = field(default_factory=dict)     # step id -> its checked params

    @property
    def sell(self) -> bool:
        return self.direction == "sell"

    @property
    def hedge_side(self) -> str:                   # a sell's hedge side is above (its highs); a buy's below
        return "high" if self.sell else "low"

    @property
    def target_side(self) -> str:
        return "low" if self.sell else "high"

    def deadline(self, tf: str, after: pd.Timestamp, wait) -> pd.Timestamp | None:
        """The last moment a step may become known: the close of bar t+N of tf after `after` (N = its max wait).
        None = no limit (or past the end of the bars: nothing after them exists here). Wait 0 = `after` itself."""
        if wait is None or wait == NO_LIMIT:
            return None
        if wait == 0:
            return after
        j = self.market.opens_from(tf, after) + int(wait) - 1
        idx = self.market.bars(tf).index
        return None if j >= len(idx) else idx[j] + pd.Timedelta(minutes=TF_MINUTES[tf])


def _price(facts: dict, ref: str) -> float | None:
    step, name = REF.fullmatch(ref).groups()
    v = facts[step].get(name)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(px: float) -> float:
    return round(round(px / config.POINT) * config.POINT, 2)


# ================================================================================================== the blocks

class _P(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Side = Literal["hedge_side", "target_side"]
Kind = Literal["equal", "session", "swing", "trendline"]
BUILT_KINDS = ("equal", "session", "swing", "trendline")     # trendlines: L6b.1 (5-minute and hour lines, 3 touches)
TARGET_KINDS = BUILT_KINDS + ("zone",)   # L6b.2: "Target = next liquidity / zone" (strategy.md section 10) - a zone is
                                         # a target (its near edge), never a level that is swept


def _kinds_col(df: pd.DataFrame) -> pd.Series:
    return df["kind"].str.split("_", n=1).str[0]


KIND_FAMILIES = typing.get_args(Kind)             # the side comes from the trade's direction, never from the name


def family_kinds(kinds):
    """"session_high", "equal_lows", "swing_low" -> "session", "equal", "swing": a kind is only its family (a sell's
    hedge side is its highs, a buy's its lows). The talker writes the side into the name - found on the first real
    intake run (L10b, 29 Sep) in a form, and again in a READING's changes (L6b.3, idea #2's U2-B), so the compiler
    reads it this way everywhere. Unknown names stay as they are (a missing block is still missing)."""
    if not isinstance(kinds, (list, tuple)):
        return kinds
    out = []
    for k in kinds:
        head = str(k).split("_", 1)[0]
        k2 = head if isinstance(k, str) and k not in KIND_FAMILIES and head in KIND_FAMILIES else k
        if k2 not in out:
            out.append(k2)
    return out


def _families(f: Idea) -> Idea:
    """The form with every kinds list read as its families (family_kinds); the same form when nothing changes."""
    d = f.model_dump(mode="json")
    changed = False
    for part in [s["params"] for s in d["steps"]] + d["targets"]:
        ks = part.get("kinds")
        if ks and family_kinds(ks) != list(ks):
            part["kinds"] = family_kinds(ks)
            changed = True
    return Idea.load(d) if changed else f


def _is(block: str, want) -> bool:
    """Is a step of `block` what a step-param wants (None = any; a name; a tuple of names)?"""
    return want is None or (block in want if isinstance(want, tuple) else block == want)


def _names(want) -> str:
    return "' or '".join(want) if isinstance(want, tuple) else str(want)


def _his_sessions(v) -> tuple[str, ...]:
    """Session names as machine/rules.py writes them (Asia, London, New York), read the way he says them ("Asian",
    "NY" - sessions.name_of). A name that is not his is refused: the machine never guesses a session."""
    out, bad = [], []
    for x in v or ():
        try:
            name = SES.name_of(x)
        except ValueError:
            bad.append(x)
            continue
        if name not in out:
            out.append(name)
    if bad or not out:
        raise ValueError(f"sessions {bad or '(none)'}: his sessions are {', '.join(SES.names())}")
    return tuple(out)


class _SessionLevels(_P):
    """For a step that reads levels. `which_session` = only the high / low of the session windows named ("the
    London high" = which_session ["London"]; L6-new). It names session levels, so kinds must be ['session'] or
    left out."""

    @field_validator("which_session", check_fields=False)
    @classmethod
    def _session_names(cls, v):
        return None if v is None else _his_sessions(v)

    @model_validator(mode="after")
    def _session_levels_only(self):
        if getattr(self, "which_session", None) and getattr(self, "kinds", None) not in (None, ("session",)):
            raise ValueError("which_session names session highs / lows only: with it, kinds must be ['session'] "
                             f"(or left out), not {list(self.kinds)}")
        return self


def _his_obvious() -> tuple[str, ...]:
    """His obvious liquidity (settings "obvious_liquidity_kinds", #84, #86) as kinds: session, trendline, equal."""
    return tuple(dict.fromkeys(k.split("_", 1)[0] for k in settings.get("obvious_liquidity_kinds")))


class Block:
    """One kind of step. `start` lists where an idea can start; `next` finds the step after the one before it."""
    name = ""
    means = ""
    uses: tuple[str, ...] = ()                 # the machine/blocks functions it is made of
    can_start = False
    waits = True                               # False: a check at the moment the step before is known (wait 0)
    tfs: tuple[str, ...] | None = None         # the timeframes it works on (None = any)
    Params: type[_P] = _P
    gives_: tuple[str, ...] = ("known_at", "bar")
    step_params: dict[str, str | tuple[str, ...] | None] = {}   # param -> the block(s) the named step must have
                                                                  # (None = any step)
    ref_params: tuple[str, ...] = ()           # params that name a price a step found ("fib.level_0.618")

    def gives(self, p) -> tuple[str, ...]:
        return self.gives_

    def charts_needed(self, p) -> tuple[str, ...]:
        """The charts it reads besides its own step's (the bias reads the hour / day / week)."""
        return ()

    def default_step(self, param: str, before: list) -> str | None:
        """The step a step-param names when the form leaves it out: the nearest earlier step with the right block."""
        want = self.step_params[param]
        for s in reversed(before):
            if _is(s.block, want):
                return s.id
        return None

    def start(self, run: _Run, step, p) -> Iterable[dict]:
        raise CompileError(f"{self.name} cannot start an idea")

    def next(self, run: _Run, step, p, facts: dict, after: pd.Timestamp, deadline) -> dict | str:
        raise CompileError(f"{self.name} can only start an idea")

    @staticmethod
    def after_deadline(t, deadline) -> bool:
        return deadline is not None and pd.Timestamp(t) > deadline


# ---- equal highs / lows form --------------------------------------------------------------------------------------
class EqualParams(_P):
    where: Side = "hedge_side"


class EqualLevels(Block):
    name = "equal_levels"
    means = ("equal highs (a sell's hedge side) or equal lows form: two highs at most $0.06 apart, nothing past the "
             "first between them (#88-#90); known 2 candles after the second touch")
    uses = ("equal_levels.equal_levels", "liquidity.book")
    can_start = True
    tfs = ("5m", "1h", "1d")
    Params = EqualParams
    gives_ = ("known_at", "bar", "first_bar", "price", "level_id", "touches")

    def _table(self, run: _Run, step, p) -> pd.DataFrame:
        side = run.hedge_side if p.where == "hedge_side" else run.target_side
        def make():
            b = run.market.liquidity(run.fp).book
            m = (b["event"] == "born") & (b["kind"] == f"equal_{side}") & (b["tf"] == step.tf)
            return b[m].sort_values("known_at", kind="stable").reset_index(drop=True)
        return run.market.cached(("equal_levels", step.tf, side), run.fp, make)

    @staticmethod
    def _event(r) -> dict:
        return {"known_at": r.known_at, "bar": r.touch_bar, "first_bar": r.first_bar, "price": float(r.price),
                "level_id": r.level_id, "touches": int(r.touches)}

    def start(self, run, step, p):
        for r in self._table(run, step, p).itertuples():
            yield self._event(r)

    def next(self, run, step, p, facts, after, deadline):
        t = self._table(run, step, p)
        k = int(t["known_at"].searchsorted(after, "right"))
        if k >= len(t) or self.after_deadline(t["known_at"].iat[k], deadline):
            return "no equal highs / lows formed within the max wait"
        return self._event(next(t.iloc[[k]].itertuples()))


# ---- the sweep -----------------------------------------------------------------------------------------------------
class SweepParams(_SessionLevels):
    of: str | None = None                      # the step that found the level (equal_levels / level_waiting)
    kinds: tuple[Kind, ...] | None = None      # without `of`: which liquidity (None = his obvious liquidity)
    charts: tuple[TF, ...] | None = None       # ... on which charts (None = all)
    which_session: tuple[str, ...] | None = None   # only the high / low of these sessions: Asia, London, New York


class Sweep(Block):
    name = "sweep"
    means = ("price takes a level on the hedge side: one wick past it is enough, no close needed (#95); the level is "
             "finished after it (#129: the same level swept again within two days is not fresh - not used). "
             "params.which_session = only the high / low of the session windows named: 'the London high is taken' "
             "= which_session ['London'] (his windows in India time, #85; the level is known when its window ends, "
             "so it is taken after that; any such level still alive counts, the same life every session level has)")
    uses = ("liquidity.book", "liquidity.sweeps", "liquidity.of_session")
    can_start = True
    tfs = ("5m",)
    Params = SweepParams
    gives_ = ("known_at", "bar", "price", "extreme", "level_id", "kind", "level_tf", "session")
    step_params = {"of": None}

    def default_step(self, param, before):
        return None                             # without `of`: any level of the kinds asked

    def _table(self, run: _Run, p) -> pd.DataFrame:
        def make():
            sw = run.market.liquidity(run.fp).sweeps
            if p.which_session:                 # "the London high": session levels of those windows only
                sw = LQ.of_session(sw, p.which_session)
            m = (sw["side"] == run.hedge_side) & sw["can_start"] & sw["fresh"]
            m &= _kinds_col(sw).isin(("session",) if p.which_session else (p.kinds or _his_obvious()))
            if p.charts:
                m &= sw["tf"].isin(p.charts)
            return sw[m].reset_index(drop=True)
        return run.market.cached(("sweeps", run.hedge_side, p.kinds, p.charts, p.which_session), run.fp, make)

    def _event(self, run: _Run, r) -> dict:
        a = run.market.arrays("5m")
        i = run.market.closed_by("5m", r.known_at)
        ext = a["high"][i] if run.sell else a["low"][i]
        return {"known_at": r.known_at, "bar": r.bar, "price": float(r.price), "extreme": float(ext),
                "level_id": r.level_id, "kind": r.kind, "level_tf": r.tf, "session": r.session}

    def start(self, run, step, p):
        for r in self._table(run, p).itertuples():
            yield self._event(run, r)

    def next(self, run, step, p, facts, after, deadline):
        if p.of is None:
            t = self._table(run, p)
            k = int(t["known_at"].searchsorted(after, "right"))
            if k >= len(t) or self.after_deadline(t["known_at"].iat[k], deadline):
                return "no level taken within the max wait"
            return self._event(run, next(t.iloc[[k]].itertuples()))
        L = run.market.liquidity(run.fp)
        k = L.by_level.get(facts[p.of].get("level_id"))
        if k is None:
            return "the level was never taken"
        r = next(L.sweeps.iloc[[k]].itertuples())
        if r.side != run.hedge_side:
            return "the level is not on the hedge side"
        if r.known_at <= after:
            return "the level was taken before the step before this one"
        if self.after_deadline(r.known_at, deadline):
            return "the level was not taken within the max wait"
        if not r.fresh:
            return "the same level was already swept (#129: it is finished)"
        if p.which_session and r.session not in p.which_session:
            return f"the level is not a {' / '.join(p.which_session)} session high or low"
        return self._event(run, r)


# ---- a level waiting on one side ------------------------------------------------------------------------------------
class WaitingParams(_SessionLevels):
    where: Side = "target_side"
    kinds: tuple[Kind, ...] | None = None      # None = his obvious liquidity
    charts: tuple[TF, ...] | None = None
    pick: Literal["nearest", "his_order"] = "nearest"
    beyond: str | None = None                  # a price the level must be past (default: the candle's close)
    which_session: tuple[str, ...] | None = None   # only the high / low of these sessions: Asia, London, New York


class LevelWaiting(Block):
    name = "level_waiting"
    means = ("a level not taken yet waits on the target side (or the hedge side) at this moment - the nearest one, "
             "or the first in his order (strongest kind, then the most recent, #130). A check: wait 0. "
             "params.which_session = only the high / low of the session windows named ('the Asian low waits "
             "below' = which_session ['Asia'])")
    uses = ("liquidity.lifetimes", "liquidity.alive", "liquidity.recent_first", "liquidity.of_session")
    waits = False
    tfs = ("5m",)
    Params = WaitingParams
    gives_ = ("known_at", "price", "level_id", "kind", "level_tf", "level_known_at")
    ref_params = ("beyond",)

    def next(self, run, step, p, facts, after, deadline):
        i = run.market.closed_by("5m", after)
        if i < 0:
            return "no candle has closed yet"
        side = run.target_side if p.where == "target_side" else run.hedge_side
        ref = _price(facts, p.beyond) if p.beyond else run.market.arrays("5m")["close"][i]
        if ref is None:
            return f"the price {p.beyond} is missing"
        kinds = ("session",) if p.which_session else (p.kinds or _his_obvious())
        got = _pick_level(run, i, side, ref, kinds, p.charts, p.pick, sessions=p.which_session)
        if got is None:
            return f"no level waiting on the {p.where.replace('_', ' ')}"
        return {"known_at": after, "price": float(got.price), "level_id": got.level_id, "kind": got.kind,
                "level_tf": got.tf, "level_known_at": got.known_at}


def _pick_level(run: _Run, i: int, side: str, ref: float, kinds, charts, pick, max_away: float | None = None,
                five_minute_swings: bool = False, sessions=None):
    """The alive level (liquidity.alive: known before candle i opened, not taken by its end) on `side` past ref.
    five_minute_swings: a recent 5-minute swing counts too - only when the hour AND the day are against the trade
    (#91: "for quick targets, when higher time frame is against my trade"; the target pick by_the_dial).
    sessions: only the high / low of the session windows named (liquidity.of_session)."""
    rows = LQ.alive(run.market.liquidity(run.fp).life, i)
    if sessions:
        rows = LQ.of_session(rows, sessions)
    rows = rows[(rows["side"] == side) & _kinds_col(rows).isin(kinds)]
    rows = rows[(rows["price"] > ref) if side == "high" else (rows["price"] < ref)]
    if not five_minute_swings:
        rows = rows[rows["target_when"] != "higher_timeframe_against"]   # a 5-minute swing: only then (#91)
    if charts:
        rows = rows[rows["tf"].isin(charts)]
    if max_away is not None:
        rows = rows[(rows["price"] - ref).abs() <= max_away]
    if not len(rows):
        return None
    if pick == "his_order":
        rows = LQ.recent_first(rows, side, ref, include_targets=True)
        return next(rows.iloc[[0]].itertuples()) if len(rows) else None
    d = (rows["price"] - ref).abs().to_numpy()
    return next(rows.iloc[[int(np.argmin(d))]].itertuples())


def next_liquidity(market: Market, fp: str, t, side: str, beyond: float, max_away: float | None = None) -> float | None:
    """The next liquidity alive at moment t on `side` ("high" above, "low" below) past the price `beyond`, or None.
    For the hedging tester (L19a): where room ends (T12 M06), the proper target (M16) and the trailing target (M23,
    M25). His real liquidity only: equal highs / lows, session highs / lows, trendlines (L6b.1: 5-minute and hour, at
    their price on that candle), hour and day swings - never a 5-minute swing, "the next small high or low" (#91,
    #124). In his order (#130: the_most_recent_liquidity_comes_first) or else the nearest. Read at the last 5-minute
    candle CLOSED by t: nothing later is used."""
    i = market.closed_by("5m", t)
    if i < 0:
        return None
    pick = "his_order" if settings.get("the_most_recent_liquidity_comes_first") else "nearest"
    lv = _pick_level(_Run(market, None, "buy", fp), i, side, float(beyond), BUILT_KINDS, None, pick, max_away)
    return None if lv is None else float(lv.price)


# ---- the break of structure -----------------------------------------------------------------------------------------
class BreakParams(_P):
    after: str | None = None                   # the step the break must come after (default: the one before it)


class BreakOfStructure(Block):
    name = "break_of_structure"
    means = ("price breaks structure the trade's way: a wick through his order block - for a sell, the most recent "
             "RED candle in the rise; a wick below the LOWEST wick from it to the candle that went above its high is "
             "the break (#56-#60; his 7 Oct 2026 words: the green candle right after the red one counts when it wicks "
             "lower). extreme = the highest price from the step before it to the break (a sell) - the grab's extreme")
    uses = ("structure.breaks", "structure.BreakSearch")
    can_start = True
    can_resume = True                          # the grab kept going: searched again from the new extreme (_Regrab)
    Params = BreakParams
    gives_ = ("known_at", "bar", "level", "block_bar", "extreme")
    step_params = {"after": None}

    def start(self, run, step, p):
        tf = step.tf
        br = run.market.cached(("breaks", tf), run.fp, lambda: ST.breaks(run.market.bars(tf), tf))
        br = br[br["side"] == run.direction]
        idx = run.market.bars(tf).index
        a = run.market.arrays(tf)
        for r in br.itertuples():
            b, k = idx.get_loc(r.bar), idx.get_loc(r.block_bar)
            ext = a["high"][k:b + 1].max() if run.sell else a["low"][k:b + 1].min()
            yield {"known_at": r.known_at, "bar": r.bar, "level": float(r.level), "block_bar": r.block_bar,
                   "extreme": float(ext)}

    def next(self, run, step, p, facts, after, deadline, resume=None):
        """resume: the moment the grab made a new extreme after an earlier break (_Regrab) - the break is looked for
        again from the candle after it, within the same max wait; the extreme is still counted from the step before."""
        tf = step.tf
        s = run.market.closed_by(tf, facts[p.after]["known_at"])
        if s < 0:
            return "no candle before the break"
        until = None if deadline is None else run.market.closed_by(tf, deadline)
        search = run.market.cached(("break_search", tf, run.direction), run.fp,
                                   lambda: ST.BreakSearch(run.market.bars(tf), tf, run.direction))
        r = search.after(s, until, None if resume is None else run.market.opens_from(tf, resume))
        if r is None:
            return "no break of structure within the max wait"
        b = run.market.bars(tf).index.get_loc(r["bar"])
        a = run.market.arrays(tf)
        # the grab's extreme counts from the grab itself: a false break is known when price closes back, candles after
        # the poke that took the liquidity - the poke's own high / low is part of the grab (L19b, P1; #9, #33, #111)
        e0 = s
        before = next((x for x in run.form.steps if x.id == p.after), None)
        if before is not None and before.block == "false_breakout" and before.tf == tf:
            j = int(run.market.bars(tf).index.get_indexer([pd.Timestamp(facts[p.after]["bar"])])[0])
            if 0 <= j < s:
                e0 = j
        ext = a["high"][e0:b + 1].max() if run.sell else a["low"][e0:b + 1].min()
        return {**r, "extreme": float(ext)}


# ---- the gap of the breaking move ------------------------------------------------------------------------------------
class GapParams(_P):
    sweep: str | None = None                   # the step where the liquidity was taken (default: the nearest sweep)
    move: str | None = None                    # the break whose move made the gap (default: the nearest break)


@dataclass(frozen=True)
class _Regrab:
    """A step's answer "the grab kept going" (his fixed rule a_grab_that_keeps_going_is_still_a_grab): the break step
    `step` is looked for again from moment `at` - the close of the candle that went past the grab's extreme."""
    step: str
    at: pd.Timestamp


class GapOfTheMove(Block):
    name = "gap_of_the_move"
    means = ("the entry gap: the FIRST gap after the liquidity is taken, made by the breaking move - its 3rd candle "
             "at most 6 candles after the break, any size (#61, #62, #101, #131); settings which_gap_of_the_move. "
             "A gap price already came back to before it was known is spent (#68). near = his entry edge (#65). "
             "The grab keeps going (his fixed rule): a candle past the grab's extreme after the break, before the gap "
             "is known, sends the idea back to wait for the break again - the hedge point moves with it. The "
             "liquidity may also be taken by a false_breakout step, and a wedge step may be both (its gap: the 3rd "
             "candle at or after the break candle). The SECOND gap (L19b; #124 'second is only when it is VERY "
             "STRONG bullish, other confidence also agrees ... or else it should wait for the first'): settings "
             "which_gap_of_the_move = 'second' - the order waits at the first gap as always; when the move's second "
             "gap is known (within the same window), the first was not reached yet, the grab has not kept going, "
             "and settings the_second_gap_needs_the_day agrees AT THAT MOMENT ('all' = the hour, the day AND the week "
             "read the trade's way; 'strong' = the hour and the day do and the week is not against; 'first' = the "
             "day does; False = always), a newer order at the second gap takes its place (a second signal; the tester "
             "cancels the waiting one, M39). Both gaps known at once: the choice is made then (True = the day must "
             "agree, else no trade). Only when the gap is the form's last step. His words 'very strong' are open "
             "(four-boxes B2-03)")
    uses = ("fvg.gaps", "fvg.gaps_of_move", "bias.htf_bias")
    Params = GapParams
    gives_ = ("known_at", "bar", "first_bar", "lo", "hi", "near", "far", "size", "gap_rank")
    step_params = {"sweep": ("sweep", "false_breakout", "wedge"), "move": ("break_of_structure", "wedge")}
    SECOND_CHARTS = {"all": ("1h", "1d", "1w"), "strong": ("1h", "1d", "1w"), "first": ("1d",), True: ("1d",)}

    def charts_needed(self, p):
        if settings.get("which_gap_of_the_move") != "second":
            return ()
        return self.SECOND_CHARTS.get(settings.get("the_second_gap_needs_the_day"), ())

    @staticmethod
    def _agrees(run: _Run, need, t) -> bool:
        """Does the higher timeframe agree enough for the second gap at moment t (settings
        the_second_gap_needs_the_day)? Every chart's bias is what was known at t (bias.bias_at)."""
        if need is False:
            return True
        d = HB.UP if run.direction == "buy" else HB.DOWN
        at = lambda tf: run.market.bias_at(tf, run.fp, t)                       # noqa: E731
        if need == "all":
            return all(at(tf) == d for tf in ("1h", "1d", "1w"))
        if need == "strong":
            return at("1h") == d and at("1d") == d and at("1w") != -d
        return at("1d") == d                                                    # "first" and True: the day agrees

    def next(self, run, step, p, facts, after, deadline):
        tf = step.tf
        bars = run.market.bars(tf)
        g_all = run.market.cached(("gaps", tf), run.fp, lambda: FVG.gaps(bars, tf))
        sweep_bar, break_bar = facts[p.sweep].get("gap_from", facts[p.sweep]["bar"]), facts[p.move]["bar"]
        b = bars.index.get_loc(break_bar)
        end = min(b + int(rules.get("gap_must_belong_to_the_breaking_move")), len(bars) - 1)
        col = g_all["bar"]
        g = g_all.iloc[int(col.searchsorted(sweep_bar, "left")):int(col.searchsorted(bars.index[end], "right"))]
        gm = FVG.gaps_of_move(bars, tf, run.direction, sweep_bar, break_bar, gap_table=g)
        gm = gm.assign(look=np.maximum(bars.index.get_indexer(pd.DatetimeIndex(gm["bar"])), b))  # known at its close
        win = end if deadline is None else min(end, run.market.closed_by(tf, deadline))
        a = run.market.arrays(tf)
        above = a["high"] if run.sell else a["low"]            # the side a sell's grab goes past (a buy's: below)
        past = (lambda x, y: x > y) if run.sell else (lambda x, y: x < y)
        # the grab keeps going: the first candle after the break past the grab's extreme, inside the gap's window
        k = None
        ext = facts[p.move].get("extreme")
        if ext is not None and rules.get("a_grab_that_keeps_going_is_still_a_grab") == "continue" and win > b:
            seg = past(above[b + 1:win + 1], ext)
            if seg.any():
                k = b + 1 + int(np.argmax(seg))
        which = settings.get("which_gap_of_the_move")
        late_visit = False
        if which == "deepest":
            # the deepest gap of the window is known only when the window ends: the choice is made at that close
            decide = win
            live = gm[(gm["rank"] >= 0) & (gm["look"] <= decide)]
            if len(live):
                gone = first_hit(above, live["look"].to_numpy() + 1, live["far"].to_numpy(), ">=" if run.sell else "<=",
                                 ends=np.full(len(live), decide + 1)) >= 0       # filled after it was known
                live = live[~gone]
            rows = live.iloc[[int(np.argmax(live["size"].to_numpy()))]] if len(live) else live
        else:
            rows = gm[gm["rank"] == 0]                         # his first gap: the one nearer the hedge (#61, #115)
            decide = int(rows["look"].iat[0]) if len(rows) else None
        if k is not None and (not len(rows) or k <= decide):
            return _Regrab(p.move, bars.index[k] + pd.Timedelta(minutes=TF_MINUTES[tf]))
        if not len(rows):
            return "the breaking move left no gap"
        then = None
        if which == "second":
            # #124: the second gap only with strong agreement, "or else it should wait for the first". Nothing later
            # than a decision's own moment is read, and a gap price already came back to is spent (#68).
            need = settings.get("the_second_gap_needs_the_day")
            close = lambda j: bars.index[j] + pd.Timedelta(minutes=TF_MINUTES[tf])          # noqa: E731
            second = gm[(gm["rank"] == 1) & (gm["look"] <= win) & ~gm["visited_before"].astype(bool)]
            if len(second) and int(second["look"].iat[0]) <= decide:
                # both gaps known at the same moment: the choice is made then
                if self._agrees(run, need, close(decide)):
                    rows = second
                elif need is True:
                    return "two gaps, and the day is not with the trade (settings the_second_gap_needs_the_day)"
            elif len(second):
                # the second gap forms later: the order waits at the first gap (his rule) until then; when the second
                # is known, if the first was not reached yet, the grab has not kept going and the trend is very strong
                # THEN, a newer order at the second gap takes its place (the tester's "cancel and take new", M39)
                look2 = int(second["look"].iat[0])
                seg = above[decide + 1:look2 + 1]
                r1 = rows.iloc[0]
                reached = bool((seg >= r1["near"]).any() if run.sell else (seg <= r1["near"]).any())
                if not reached and (k is None or k > look2) and self._agrees(run, need, close(look2)):
                    r2 = second.iloc[0]
                    then = {"known_at": close(look2), "bar": r2["bar"], "first_bar": r2["first_bar"],
                            "lo": float(r2["lo"]), "hi": float(r2["hi"]), "near": float(r2["near"]),
                            "far": float(r2["far"]), "size": float(r2["size"]), "gap_rank": int(r2["rank"])}
        r = next(rows.iloc[[0]].itertuples())
        known = pd.Timestamp(r.known_at)
        if which == "deepest":
            seg = above[int(r.look) + 1:decide + 1]
            late_visit = bool((seg >= r.near).any() if run.sell else (seg <= r.near).any())
            known = max(known, bars.index[decide] + pd.Timedelta(minutes=TF_MINUTES[tf]))
        if r.visited_before or late_visit:
            return "price came back to the gap before it was known (#68: spent)"
        if self.after_deadline(known, deadline):
            return "the gap was not known within the max wait"
        out = {"known_at": known, "bar": r.bar, "first_bar": r.first_bar, "lo": float(r.lo), "hi": float(r.hi),
               "near": float(r.near), "far": float(r.far), "size": float(r.size), "gap_rank": int(r.rank)}
        if then is not None:
            out["then"] = then                  # a later order in its place (_walk makes it a newer signal)
        return out


# ---- price comes back to a level ----------------------------------------------------------------------------------
class TouchParams(_P):
    at: str                                    # the price a step found: "fib.level_0.618"


class ComesBackTo(Block):
    name = "comes_back_to"
    means = ("price comes back to a price a step found, against the trade: a buy's low reaches down to it, a sell's "
             "high up to it (a wick is enough); known at that candle's close")
    uses = ("_base.first_hit",)
    Params = TouchParams
    gives_ = ("known_at", "bar", "price")
    ref_params = ("at",)

    def next(self, run, step, p, facts, after, deadline):
        level = _price(facts, p.at)
        if level is None:
            return f"the price {p.at} is missing"
        tf = step.tf
        a = run.market.arrays(tf)
        start = run.market.opens_from(tf, after)
        end = len(a["low"]) if deadline is None else run.market.closed_by(tf, deadline) + 1
        if start >= end:
            return "price did not come back within the max wait"
        arr, op = (a["high"], ">=") if run.sell else (a["low"], "<=")
        j = int(first_hit(arr, [start], [level], op, ends=[end])[0])
        if j < 0:
            return "price did not come back within the max wait"
        idx = run.market.bars(tf).index
        return {"known_at": idx[j] + pd.Timedelta(minutes=TF_MINUTES[tf]), "bar": idx[j], "price": level}


# ---- a one-sided move and its Fibonacci levels ----------------------------------------------------------------------
class FibMove(Block):
    name = "fib_move"
    means = ("a one-sided move: a higher high (a buy) or a lower low (a sell) of his 2-candle swings, and its "
             "Fibonacci levels (settings fib_ratios): level_0.618 = top - 0.618 x (top - bottom) for a buy. The plain "
             "reading of #135; which break counts is his open question F3")
    uses = ("fib.fib_levels", "swings.swings")
    can_start = True

    def gives(self, p):
        return ("known_at", "bar", "from_bar", "top", "bottom",
                *[FIB.level_name(r) for r in settings.get("fib_ratios")])

    def start(self, run, step, p):
        tf = step.tf
        f = run.market.cached(("fib", tf), run.fp, lambda: FIB.fib_levels(run.market.bars(tf), tf))
        cols = [c for c in f.columns if c.startswith("level_")]
        for r in f[f["side"] == run.direction].to_dict("records"):
            yield {"known_at": r["known_at"], "bar": r["to_bar"], "from_bar": r["from_bar"], "top": r["top"],
                   "bottom": r["bottom"], **{c: r[c] for c in cols}}


# ---- L6b.3: his bias on the hour / day / week (#69-#74) --------------------------------------------------------------
HTF = Literal["15m", "1h", "4h", "1d", "1w"]


class BiasParams(_P):
    charts: tuple[HTF, ...] = ("1h", "1d")     # his "the hour AND the day" (#69, #76, #82)
    must: Literal["with", "not_against"] = "with"

    @field_validator("charts")
    @classmethod
    def _some(cls, v):
        if not v:
            raise ValueError("name at least one chart: 1h, 1d, 1w ...")
        return v


def _grab_is_hammer(run: _Run, facts: dict) -> bool:
    """The candle of the idea's first sweep step is his hammer, the trade's way (#115)."""
    sid = next((s.id for s in run.form.steps if s.block == "sweep" and s.id in facts), None)
    if sid is None or facts[sid].get("bar") is None:
        return False
    j = run.market.bars("5m").index.get_indexer([pd.Timestamp(facts[sid]["bar"])])[0]
    return bool(j >= 0 and run.market.hammers(run.fp)[run.direction].iat[j])


def _bias_why_not(run: _Run, p, t, facts: dict) -> str | None:
    """None = the bias passes at moment t; else why not, in plain words."""
    d = HB.UP if run.direction == "buy" else HB.DOWN
    got = {tf: run.market.bias_at(tf, run.fp, t) for tf in p.charts}
    ok = all(v == d for v in got.values()) if p.must == "with" else not all(v == -d for v in got.values())
    if ok or (settings.get("hammer_beats_the_higher_timeframe") and _grab_is_hammer(run, facts)):
        return None
    now = ", ".join(f"{tf} {HB.WORD[v]}" for tf, v in got.items())
    if p.must == "with":
        return f"the higher timeframe is not with the trade ({now})"
    return f"the higher timeframe is against the trade ({now})"


class BiasCheck(Block):
    name = "bias"
    means = ("his direction on the hour / day / week chart (#69-#74): his 5-minute entry run on that chart's own "
             "candles - up / down once a level is taken there, then his break and a gap; neutral while it waits; up at "
             "the real all-time high (day, week). must 'with' = every chart named reads the trade's way; "
             "'not_against' = not all of them read against it. A hammer grab passes (#115, settings "
             "hammer_beats_the_higher_timeframe). A check at that moment; the filter 'bias' is the same check")
    uses = ("bias.htf_bias", "bias.bias_at", "hammer.hammers")
    waits = False
    Params = BiasParams
    gives_ = ("known_at",)

    def charts_needed(self, p):
        return tuple(p.charts)

    def next(self, run, step, p, facts, after, deadline):
        why = _bias_why_not(run, p, after, facts)
        return why if why else {"known_at": after}


# ---- L6b.3: premium / discount - his ADR zones (strategy.md section 7, #76) ------------------------------------------
class ZoneParams(_P):
    side: Literal["sell_top_buy_bottom", "sell_bottom_buy_top"] = "sell_top_buy_bottom"
    measure: Literal["near_the_zone", "share_of_the_day"] = "near_the_zone"
    price: str | None = None                   # a price a step found (default: the entry price, else the close)


class PremiumDiscount(Block):
    name = "premium_discount"
    means = ("price is at his ADR zone for this trade (strategy.md section 7; #76): side 'sell_top_buy_bottom' = a sell "
             "at the top (red, premium) zone, a buy at the bottom (green, discount); 'sell_bottom_buy_top' = the other "
             "way. 'At' (measure near_the_zone) = at, inside or past the zone, or settings near_zone_adr of a day's "
             "range short of it; measure share_of_the_day = the old engine's share of the day's ADR range (settings "
             "premium_share / discount_share). As a check it reads `price` (default: the entry price when a step "
             "before it found it, else the candle's close). As the first step: each time price comes to that zone "
             "(a run of candles counts once a day)")
    uses = ("adr.adr_bars", "premium_discount.zones", "premium_discount.arrivals")
    can_start = True
    waits = False
    tfs = ("5m",)
    Params = ZoneParams
    gives_ = ("known_at", "bar", "price", "zone_lo", "zone_hi")
    ref_params = ("price",)

    @staticmethod
    def _top(run: _Run, p) -> bool:
        return run.sell == (p.side == "sell_top_buy_bottom")

    def _event(self, run: _Run, p, i: int, price: float, bar) -> dict:
        z = run.market.adr_zones(run.fp, p.measure).iloc[i]
        lo, hi = (z["up_lo"], z["up_hi"]) if self._top(run, p) else (z["dn_lo"], z["dn_hi"])
        return {"known_at": z["known_at"], "bar": bar, "price": float(price), "zone_lo": float(lo),
                "zone_hi": float(hi)}

    def start(self, run, step, p):
        top = self._top(run, p)
        arr = run.market.cached(("adr_arrivals", top, p.measure), run.fp, lambda: PD.arrivals(
            run.market.bars("5m"), "5m", "top" if top else "bottom", p.measure))
        idx = run.market.bars("5m").index
        for r in arr.itertuples():
            yield self._event(run, p, int(idx.get_loc(r.bar)), r.price, r.bar)

    def next(self, run, step, p, facts, after, deadline):
        i = run.market.closed_by("5m", after)
        if i < 0:
            return "no candle has closed yet"
        ref = p.price
        if ref is None and run.form.entry.how == "limit" and REF.fullmatch(run.form.entry.at).group(1) in facts:
            ref = run.form.entry.at
        px = _price(facts, ref) if ref else float(run.market.arrays("5m")["close"][i])
        if px is None:
            return f"the price {ref} is missing"
        z = run.market.adr_zones(run.fp, p.measure).iloc[i]
        want = PD.PREMIUM if self._top(run, p) else PD.DISCOUNT
        got = PD.place(px, z)
        if got != want:
            where = "top (red)" if want == PD.PREMIUM else "bottom (green)"
            return f"the price is not at the {where} ADR zone (it is at {got})"
        ev = self._event(run, p, i, px, None)
        return {**ev, "known_at": after}


# ---- L6-new: the break of his ADR zone (idea #4; strategy.md section 7, #76) ------------------------------------------
class AdrZoneBreak(Block):
    name = "adr_zone_break"
    means = ("price BREAKS OUT of his ADR zone the trade's way (strategy.md section 7; #76 'if it breaks the lower range "
             "of ADR, lower zone, it will go further low'): a sell = a 5-minute candle CLOSES below the green (bottom) "
             "zone's bottom, a buy = it closes above the red (top) zone's top - by at least settings "
             "adr_break_beyond_adr of a day's range (0 = any close beyond). A wick is never a break. Not the same as "
             "premium_discount, which is price AT a zone (a turn there). As the first step: each new close beyond "
             "(a run of closes beyond counts once a day); later: the first one after the step before. Known at that "
             "candle's close. It does NOT replace his grab: an entry gap after it still needs his liquidity taken (a "
             "sweep step) and his break of structure (rules.py which_gap_is_the_entry_gap, #61, #62, #131; #76 itself: "
             "'if it took the liquidity and fair value is good') - his Type 1 after the zone break, so its hedge is "
             "Type 1's (the grab's extreme)")
    uses = ("adr.adr_bars", "premium_discount.breaks")
    can_start = True
    tfs = ("5m",)
    gives_ = ("known_at", "bar", "price", "zone_lo", "zone_hi")

    @staticmethod
    def _table(run: _Run) -> pd.DataFrame:
        side = "bottom" if run.sell else "top"
        return run.market.cached(("adr_breaks", side), run.fp, lambda: PD.breaks(run.market.bars("5m"), "5m", side))

    @staticmethod
    def _event(r) -> dict:
        return {"known_at": r.known_at, "bar": r.bar, "price": float(r.price), "zone_lo": float(r.zone_lo),
                "zone_hi": float(r.zone_hi)}

    def start(self, run, step, p):
        for r in self._table(run).itertuples():
            yield self._event(r)

    def next(self, run, step, p, facts, after, deadline):
        t = self._table(run)
        idx = run.market.bars("5m").index
        start = run.market.opens_from("5m", after)
        if start >= len(idx):
            return "no candle after the step before"
        rows = t[pd.DatetimeIndex(t["bar"]) >= idx[start]]
        if deadline is not None:
            rows = rows[pd.DatetimeIndex(rows["known_at"]) <= pd.Timestamp(deadline)]
        if rows.empty:
            where = "below the green" if run.sell else "above the red"
            return f"no 5-minute close {where} ADR zone within the max wait"
        return self._event(next(rows.itertuples()))


# ---- L6b.3: his hammer (#115) ----------------------------------------------------------------------------------------
class HammerParams(_P):
    of: str | None = None                      # the sweep step whose candle took the liquidity (default: the nearest)


class HammerCheck(Block):
    name = "hammer"
    means = ("the candle that took the liquidity is his hammer (#115, rules.py what_a_hammer_is): the wick on the side "
             "it took is at least 55% of the candle and 0.12 of a day's range, the body and the other wick at most 40% "
             "- it took the stops and snapped back. A check at that moment. His hammer + gap is a stronger setup and "
             "passes the higher-timeframe check (settings hammer_beats_the_higher_timeframe)")
    uses = ("hammer.hammers",)
    waits = False
    Params = HammerParams
    gives_ = ("known_at", "bar", "wick_share", "dip")
    step_params = {"of": "sweep"}

    def next(self, run, step, p, facts, after, deadline):
        bar = facts[p.of].get("bar")
        j = run.market.bars("5m").index.get_indexer([pd.Timestamp(bar)])[0] if bar is not None else -1
        if j < 0:
            return "the candle that took the liquidity is missing"
        hm = run.market.hammers(run.fp)
        side = run.direction
        if not hm[side].iat[j]:
            return "the candle that took the liquidity is not his hammer (#115)"
        return {"known_at": after, "bar": bar, "wick_share": float(hm[f"{side}_wick_share"].iat[j]),
                "dip": float(hm[f"{side}_dip"].iat[j])}


# ---- L6b.3: the real all-time high (#72) -----------------------------------------------------------------------------
class AllTimeHigh(Block):
    name = "all_time_high"
    means = ("price is at the REAL all-time high (#72: 'only all time high means only all time high', not a 52-week "
             "high): the 5-minute close is within settings near_all_time_high of the highest price gold ever printed, "
             "counted from its September 2011 record (settings all_time_high_before_2016). A check at that moment")
    uses = ("bias.all_time_high",)
    waits = False
    gives_ = ("known_at", "price", "ath")

    def next(self, run, step, p, facts, after, deadline):
        i = run.market.closed_by("5m", after)
        if i < 0:
            return "no candle has closed yet"
        a = run.market.ath5(run.fp)
        if not a["at_ath"].iat[i]:
            return "price is not at the real all-time high"
        return {"known_at": after, "price": float(run.market.arrays("5m")["close"][i]), "ath": float(a["ath"].iat[i])}


# ---- L6b.1: his trendline touch entry (Type 2, #127) ---------------------------------------------------------------
class TrendlineTouchParams(_P):
    touches: tuple[Literal[3, 4], ...] = (3, 4)      # settings wedge_touch: "the 3rd touch, up to the 4th"


class TrendlineTouch(Block):
    name = "trendline_touch"
    means = ("his Type 2 (strategy.md section 9; #127 'the trendline touch is real and the steep of the line decides "
             "that'): the 3rd (or 4th) touch of a FALLING trendline through highs = a sell, of a RISING one through "
             "lows = a buy; no gap needed. Only a STEEP line (settings trendline_min_steepness_adr, days of range the "
             "line travels in a day - his word 'steep' is OPEN, four-boxes B2-01: the step waits for his answer unless "
             "the form names the number), a clear trend visible between its touches (#116, settings "
             "trendline_needs_trend), touches at least 3 hours apart (his fixed rule). Gives entry (a limit on the line "
             "- the worse fill of settings trendline_touch_fill_bars), hedge (past the line - it then moves with the "
             "line: management, L19a), extreme (the lowest low / highest high since the first touch: his 'target = "
             "wedge low'). A flat line is never a touch entry (#116). The higher timeframe must be with it: add the "
             "filter 'bias'")
    uses = ("trendlines.lines", "trendlines.more_touches", "trendlines.touch_entries")
    can_start = True
    tfs = ("5m", "1h")
    Params = TrendlineTouchParams
    gives_ = ("known_at", "bar", "line", "entry", "hedge", "extreme", "steepness", "touch", "first_bar")

    def start(self, run, step, p):
        steep = settings.get("trendline_min_steepness_adr")
        if steep is None:
            raise CompileError("trendline_touch: 'steep' is still open (four-boxes B2-01) - the compiler refuses it "
                               "before this point")
        t = run.market.touch_entries(step.tf, run.fp)
        t = t[(t["direction"] == run.direction) & t["touch"].isin(p.touches) & t["touches_apart"]
              & (t["steepness"] >= float(steep))]
        if settings.get("trendline_needs_trend"):
            t = t[t["trend_visible"]]
        for r in t.itertuples():
            yield {"known_at": r.known_at, "bar": r.bar, "line": float(r.line), "entry": float(r.entry),
                   "hedge": float(r.hedge), "extreme": float(r.extreme), "steepness": float(r.steepness),
                   "touch": int(r.touch), "first_bar": r.first_bar, "line_id": r.line_id, "slope": float(r.slope)}


# ---- L6b.1: triangles and their false breaks (strategy.md section 4) -----------------------------------------------
class Triangle(Block):
    name = "triangle"
    means = ("a triangle (strategy.md section 4; #116, #118): a line through highs and a line through lows, both alive, "
             "that narrow to 60% of the gap they started with (rules.py what_a_triangle_is), at least 2 hours long, no "
             "taller than two days of range, after a big move (settings big_move_adr - his words 'big move' are OPEN, "
             "four-boxes B2-02: the step waits for his answer unless the form names the number). Known at the close of "
             "the candle where it has narrowed enough. top / bottom = the two lines' prices then")
    uses = ("trendlines.lines", "patterns.triangles")
    can_start = True
    tfs = ("5m", "1h")
    gives_ = ("known_at", "bar", "top", "bottom", "tri_id", "big_move")

    def start(self, run, step, p):
        t = run.market.triangles(step.tf, run.fp, _big_move())
        for r in t.itertuples():                    # a triangle is both ways: the break decides the direction
            yield {"known_at": r.known_at, "bar": r.bar, "top": float(r.top), "bottom": float(r.bottom),
                   "tri_id": r.tri_id, "big_move": float(r.big_move)}


def _big_move() -> float:
    v = settings.get("big_move_adr")
    if v is None:
        raise CompileError("triangle: 'big move' is still open (four-boxes B2-02) - the compiler refuses it before "
                           "this point")
    return float(v)


class FalseParams(_P):
    of: str | None = None                      # a triangle step (its boundary); None = his obvious liquidity
    kinds: tuple[Kind, ...] | None = None      # without `of`: which liquidity (None = his obvious liquidity)
    charts: tuple[TF, ...] | None = None


class FalseBreakout(Block):
    name = "false_breakout"
    means = ("a false breakout (strategy.md sections 4, 14; P1 / P2): price pokes past a level on the hedge side and "
             "comes back - the liquidity is CONSUMED and price goes the other way. of = a triangle step: its boundary, "
             "a close back inside within settings fakeout_lookback_bars (his old spec, 20); without `of`: his obvious "
             "liquidity (a fresh sweep, #129) whose first decisive close is back on the side price came from (the old "
             "engine's reading, #128), within settings false_break_close_back_bars. Known at that close. price = the "
             "level at the poke; extreme = the farthest the poke went; opposite = the triangle's other line (P1's "
             "target)")
    uses = ("patterns.triangle_breaks", "patterns.false_breaks", "liquidity.sweeps")
    can_start = True
    tfs = ("5m", "1h")
    Params = FalseParams
    gives_ = ("known_at", "bar", "price", "extreme", "opposite", "level_id", "kind")
    step_params = {"of": None}                 # optional; when named it must be a triangle step (_check)

    def default_step(self, param, before):
        return None                             # without `of`: any of his obvious liquidity

    def _levels(self, run: _Run, step, p) -> pd.DataFrame:
        if step.tf != "5m":
            raise CompileError("false_breakout without 'of' reads his liquidity on the 5-minute chart")

        def make():
            fb = run.market.false_breaks(run.fp)
            m = (fb["result"] == "false") & (fb["side"] == run.hedge_side) & fb["fresh"]
            L = run.market.liquidity(run.fp)
            can = fb["level_id"].map(L.sweeps.set_index("level_id")["can_start"]).fillna(False).astype(bool)
            m &= can & _kinds_col(fb).isin(p.kinds or _his_obvious())
            if p.charts:
                m &= fb["tf"].isin(p.charts)
            return fb[m].reset_index(drop=True)
        return run.market.cached(("false_levels", run.hedge_side, p.kinds, p.charts), run.fp, make)

    @staticmethod
    def _event(r) -> dict:
        return {"known_at": r.known_at, "bar": r.bar, "price": float(r.price), "extreme": float(r.extreme),
                "opposite": math.nan, "level_id": r.level_id, "kind": r.kind}

    def start(self, run, step, p):
        if p.of is not None:
            raise CompileError("false_breakout with 'of' cannot start an idea: put the triangle step first")
        for r in self._levels(run, step, p).itertuples():
            yield self._event(r)

    def next(self, run, step, p, facts, after, deadline):
        if p.of is None:
            t = self._levels(run, step, p)
            k = int(t["known_at"].searchsorted(after, "right"))
            if k >= len(t) or self.after_deadline(t["known_at"].iat[k], deadline):
                return "no false breakout within the max wait"
            return self._event(next(t.iloc[[k]].itertuples()))
        tri = facts[p.of]
        b = run.market.triangle_breaks(step.tf, run.fp, _big_move())
        b = b[b["tri_id"] == tri["tri_id"]]
        if not len(b):
            return "price never left the triangle"
        r = next(b.iloc[[0]].itertuples())
        if r.kind != "false":
            return "the triangle broke for real (no close back inside)"
        if r.direction != run.direction:
            return "the false break points the other way"
        if r.known_at <= after:
            return "the false break came before the step before this one"
        if self.after_deadline(r.known_at, deadline):
            return "no false breakout within the max wait"
        ev = {"known_at": r.known_at, "bar": r.bar, "price": float(r.price), "extreme": float(r.extreme),
              "opposite": float(r.opposite), "level_id": r.tri_id, "kind": "triangle"}
        return ev


# ---- L6b.1: the wedge (strategy.md section 4) --------------------------------------------------------------------------
class Wedge(Block):
    name = "wedge"
    means = ("a wedge (strategy.md section 4: 'single line holds one side while price breaks the other'): a trendline "
             "(3 touches) that no candle closed beyond while price CLOSED past the other side of the wedge (its lowest "
             "low of settings wedge_break_bars candles before the line, for a line through highs) - the trade goes out "
             "that side (a line through highs held = a sell). Known at that close. level = the edge it broke; "
             "wedge_high / wedge_low = the wedge's extremes (the hedge goes back beyond the wedge); line = the held "
             "line's price then. The gap after it: gap_of_the_move with sweep and move = this step (a gap whose 3rd "
             "candle is the break candle or later, the old engine's). settings the_wedge_needs_the_hour_and_day: add "
             "the filter 'bias'")
    uses = ("trendlines.lines", "patterns.wedges")
    can_start = True
    tfs = ("5m", "1h")
    gives_ = ("known_at", "bar", "level", "wedge_high", "wedge_low", "line", "gap_from")

    def start(self, run, step, p):
        w = run.market.wedges(step.tf, run.fp)
        idx = run.market.bars(step.tf).index
        for r in w[w["direction"] == run.direction].itertuples():
            k = idx.get_loc(r.bar)
            yield {"known_at": r.known_at, "bar": r.bar, "level": float(r.level), "wedge_high": float(r.wedge_high),
                   "wedge_low": float(r.wedge_low), "line": float(r.line), "gap_from": idx[max(0, k - 2)]}


# ---- L6b.2: his strong zones (strategy.md sections 3, 9, 10) -----------------------------------------------------------
class ZoneEntry(Block):
    name = "zone_entry"
    means = ("his zone entry, no gap (strategy.md section 9; #84, #86): a candle takes one of his obvious levels "
             "(session high / low, trendline, equal highs / lows - a fresh sweep, #129) with a live strong zone just "
             "beyond it (its near edge at most settings zone_entry_liquidity_adr of a day's range away; his main, "
             "quiet zone first, #81), the candle stays outside the zone, and the zone has not played that role yet "
             "(#78). Known at that candle's close. entry = the zone's near edge (a limit - 'I enter there'); hedge = "
             "just past its far edge ('the zone low will be my hedge point'). A later take at the same zone is a newer "
             "setup (the tester: #145 'Cancel and take new'). His 'the higher time frame must be with me' (#84, #86): add the filter 'higher_timeframe'; his unfilled gap "
             "(#87): add the rule 'unfilled_gap_past_hedge'")
    uses = ("zones.strong_zones", "zones.lives", "zones.entries", "liquidity.sweeps")
    can_start = True
    tfs = ("5m",)
    gives_ = ("known_at", "bar", "entry", "hedge", "lo", "hi", "tier", "zone_id", "level_price", "level_id", "kind")

    def charts_needed(self, p):
        return ("1h",)

    def start(self, run, step, p):
        t = run.market.zone_entries(run.fp)
        for r in t[t["direction"] == run.direction].itertuples():
            yield {"known_at": r.known_at, "bar": r.bar, "entry": float(r.entry), "hedge": float(r.hedge),
                   "lo": float(r.lo), "hi": float(r.hi), "tier": int(r.tier), "zone_id": r.zone_id,
                   "level_price": float(r.level_price), "level_id": r.level_id, "kind": r.level_kind}


class ZoneBreak(Block):
    name = "zone_break"
    means = ("his zone break retest (strategy.md section 9; #97, #98, #99): a candle CLOSES through a strong zone by a "
             "real amount (rules.py what_a_real_zone_break_is: 0.20 of the zone's height - 'tiny wicks are not "
             "considered'), having come through it from the other side; no distance rule (#98). entry = the broken "
             "edge (the retest, the S/R flip); hedge = just past the far edge - at the all-time high the NEXT strong "
             "zone beyond it, at most settings zone_break_ath_max_hedge_adr of a day's range away, else no trade (#97; "
             "settings zone_break_hedge_at_ath). His obvious liquidity INSIDE the zone (#99, settings "
             "zone_break_inner_liquidity): 'gap' = wait for it to be taken, then the first 5-minute gap the trade's way "
             "within settings zone_break_inner_gap_bars (entry its near edge, hedge past the take's extreme); 'wait' = "
             "no trade; 'off' = the plain retest. The 1-minute lines inside the zone count too (L6b.4, #98 C - his "
             "1-minute triangle; settings zone_break_inner_1m_lines, with 1-minute bars): the first one taken, flat "
             "level or line, is the take. Known at the break's close (or the gap's). His 'higher time frame "
             "confluence': the filter 'higher_timeframe'; 'there should be liquidity to target': the target kinds; 'it "
             "shouldn't take the obvious liquidity' on the way back: the rule 'target_reached_first'")
    uses = ("zones.strong_zones", "zones.lives", "zones.breaks", "zones.after_inner_take", "liquidity.alive",
            "bias.all_time_high", "one_minute.between", "one_minute.first_take")
    can_start = True
    tfs = ("5m",)
    gives_ = ("known_at", "bar", "entry", "hedge", "edge", "lo", "hi", "tier", "zone_id", "how", "inner")

    def charts_needed(self, p):
        return ("1h",)

    @staticmethod
    def _lines_inside(run: _Run, r) -> pd.DataFrame:
        """L6b.4 (#98 C, #99): the 1-minute lines on the hedge side inside the zone when the break is known - only with
        settings zone_break_inner_1m_lines and 1-minute bars in the market (else none: _check writes a note)."""
        if not settings.get("zone_break_inner_1m_lines") or not run.market.has("1m"):
            return pd.DataFrame({"line_id": pd.Series(dtype="object")})
        return OM.between(run.market.book1(run.fp), run.market.closed_by("1m", r.known_at), float(r.lo), float(r.hi),
                          run.hedge_side)

    def _event(self, run: _Run, r) -> dict | None:
        m5 = run.market.bars("5m")
        i = int(m5.index.get_loc(r.bar))
        ruler = run.market.ruler("5m", run.fp)[i]
        if not math.isfinite(ruler):
            return None
        buy = run.direction == "buy"
        entry, hedge, how, inner, known, bar = float(r.edge), float(r.hedge), "retest", math.nan, r.known_at, r.bar
        mode = settings.get("zone_break_inner_liquidity")
        if mode != "off":
            rows = LQ.alive(run.market.liquidity(run.fp).life, i)
            rows = rows[(rows["side"] == run.hedge_side) & rows["kind"].isin(settings.get("obvious_liquidity_kinds"))]
            px = rows["price"].to_numpy(float)
            px = px[(px > r.lo) & (px < r.hi)]
            lines = self._lines_inside(run, r)                     # L6b.4: the 1-minute lines inside it (#98 C)
            if len(px) or len(lines):
                if mode == "wait":
                    return None                    # #99 as a wait: no plain retest when liquidity sits inside
                # the take must come while the zone's days last (Claude's reading: its life, known when it was born)
                z = run.market.cached(("zone_days",), run.fp, lambda: {
                    q.zone_id: int(q.born + q.life_bars) for q in run.market.zone_life(run.fp).itertuples()})
                d = z.get(r.zone_id)
                end = None if d is None else d - 1
                start, inner = i, float(px.max() if buy else px.min()) if len(px) else math.nan
                first = OM.first_take(lines, run.market.closed_by("1m", r.known_at)) if len(lines) else None
                if first is not None:              # a line taken before the level (or with no level): its take first
                    row, j = first
                    t1 = run.market.bars("1m").index[j]
                    k5 = int(m5.index.searchsorted(t1, "right")) - 1
                    a5 = run.market.arrays("5m")
                    seg = a5["low"][i + 1:k5] if buy else a5["high"][i + 1:k5]
                    level_first = len(px) and bool((seg <= inner).any() if buy else (seg >= inner).any())
                    if (end is None or k5 <= end) and not level_first:
                        start, inner = k5 - 1, float(row["b_price"] + row["slope"] * (j - row["x0"]))
                elif not len(px):
                    return None                    # the lines inside are never taken: no trade (#99 'wait')
                got = ZN.after_inner_take(m5, start, run.direction, inner, float(ruler), end=end)
                if isinstance(got, str):
                    return None
                entry, hedge, how, known, bar = got["entry"], got["hedge"], "after_inner_take", got["known_at"], \
                    got["bar"]
        if how == "retest" and settings.get("zone_break_hedge_at_ath"):
            ath = run.market.ath5(run.fp)["ath"].iat[i]
            if math.isfinite(ath) and entry >= float(settings.get("near_all_time_high")) * ath:
                z = ZN.alive(run.market.zone_life(run.fp), i)
                far = r.lo if buy else r.hi
                z = z[(z["hi"] < far) if buy else (z["lo"] > far)]
                if not len(z):
                    return None                    # "there should be a proper good real strong zone ... or disaster"
                edge = z["hi"].to_numpy(float) if buy else z["lo"].to_numpy(float)
                k = int(np.argmin(np.abs(edge - far)))
                hedge = float(z["lo"].iat[k] - ZN.PAD if buy else z["hi"].iat[k] + ZN.PAD)
                if abs(entry - hedge) > float(settings.get("zone_break_ath_max_hedge_adr")) * ruler:
                    return None
        return {"known_at": known, "bar": bar, "entry": float(entry), "hedge": float(hedge), "edge": float(r.edge),
                "lo": float(r.lo), "hi": float(r.hi), "tier": int(r.tier), "zone_id": r.zone_id, "how": how,
                "inner": inner}

    def start(self, run, step, p):
        t = run.market.zone_breaks(run.fp)
        out = [e for r in t[t["direction"] == run.direction].itertuples() if (e := self._event(run, r)) is not None]
        yield from sorted(out, key=lambda e: pd.Timestamp(e["known_at"]))


class ZoneCheckParams(_P):
    at: str | None = None                      # a price a step found (default: the entry price, else the close)
    min_rank: int = 0                          # 0 any window · 1 week or hour · 2 the hour chart ("1h stronger")
    tiers: tuple[Literal[1, 2, 3], ...] | None = None   # None = what settings zone_tiers hands on


class ZoneCheck(Block):
    name = "zone"
    means = ("the price is AT one of his live strong zones (strategy.md section 3: 'high-prob sweep+FVG at a zone'): "
             "inside it, or within settings near_zone_adr of a day's range of it. Main zone first (#81), then the "
             "hour chart ('1h stronger'). A check at that moment; reads `at` (default: the entry price when a step "
             "before it found it, else the candle's close)")
    uses = ("zones.strong_zones", "zones.lives", "zones.alive")
    waits = False
    tfs = ("5m",)
    Params = ZoneCheckParams
    gives_ = ("known_at", "lo", "hi", "tier", "rank", "zone_id")
    ref_params = ("at",)

    def charts_needed(self, p):
        return ("1h",)

    def next(self, run, step, p, facts, after, deadline):
        i = run.market.closed_by("5m", after)
        if i < 0:
            return "no candle has closed yet"
        ref = p.at
        if ref is None and run.form.entry.how == "limit" and REF.fullmatch(run.form.entry.at).group(1) in facts:
            ref = run.form.entry.at
        px = _price(facts, ref) if ref else float(run.market.arrays("5m")["close"][i])
        if px is None:
            return f"the price {ref} is missing"
        z = ZN.alive(run.market.zone_life(run.fp), i, p.min_rank)
        if p.tiers:
            z = z[z["tier"].isin(p.tiers)]
        pad = float(settings.get("near_zone_adr")) * run.market.ruler("5m", run.fp)[i]
        z = z[(z["lo"] - pad <= px) & (px <= z["hi"] + pad)] if math.isfinite(pad) else z.iloc[:0]
        if not len(z):
            return "the price is not at a strong zone"
        r = z.iloc[0]
        return {"known_at": after, "lo": float(r.lo), "hi": float(r.hi), "tier": int(r.tier), "rank": int(r["rank"]),
                "zone_id": r.zone_id}


# ---- L6b.4: the 1-minute check before a 5-minute gap entry (strategy.md sections 5, 14; #100, #111) ----------------
@dataclass(frozen=True)
class _Resweep:
    """The 1-minute check's answer "wait": a line against the entry was taken at event["known_at"]; that take is the
    idea's liquidity grab now - the step `step` (where the liquidity was taken) holds it, and every step after it is
    looked for again from it (#111 'the actual entry should be once after it sweep the trend line')."""
    step: str
    event: dict


LINE_CHARTS = ("1m", "5m", "1h")
ALSO_KINDS = ("equal", "session")


class OneMinuteParams(_P):
    entry: str | None = None                   # the entry price (default: the form's entry.at - a 5-minute gap)
    hedge: str | None = None                   # the hedge price (default: the form's exit.at)
    sweep: str | None = None                   # the step where the liquidity was taken (default: the nearest one)
    charts: tuple[Literal["1m", "5m", "1h"], ...] = LINE_CHARTS     # whose lines count (1m: #111; 5m / 1h: section 4)
    also: tuple[Literal["equal", "session"], ...] = ()               # #99: his other liquidity between them counts too


class OneMinute(Block):
    name = "one_minute"
    means = ("the 1-minute check before a 5-minute gap entry (strategy.md sections 5 and 14: '1-min = never the entry "
             "chart; a 1-minute trendline or triangle against the entry means wait', #100, #111; his F04 and Q98 C "
             "charts). At the moment the gap is known: a line (2 or 3 touches - 'or about to get its 3rd', #111) of the "
             "1-minute chart - or the 5-minute / hour chart (strategy.md section 4: 'a trendline against the entry') - "
             "on the hedge side, BETWEEN the entry and the hedge, not taken yet: WAIT (a triangle is two lines: its "
             "lines count). None: the check passes at once. Else the idea waits for the first of them to be taken (a "
             "wick through it, on the 1-minute candles - #69 'a trend line fake out is a liquidity taken'), and that "
             "take becomes the idea's liquidity grab: the steps after the sweep step - the break, the first gap of its "
             "move - are looked for again from it, so the hedge moves to the new grab's extreme (#99 'so our hedging "
             "distance will be less'). The check then runs again, but only on the lines known at its FIRST check (his "
             "F04: the gap after the take is 'the actual entry' - settings one_minute_new_lines_after_a_take). also: "
             "his other liquidity between them counts too (#99 for zone entries: equal "
             "highs / lows, session highs / lows). max_wait_bars: 1-minute candles to wait for the take. Only after a "
             "5-minute gap step: the zone entries (#99, #111) join when L6b.2 builds them")
    uses = ("trendlines.lines", "one_minute.book", "one_minute.between", "one_minute.first_take", "trendlines.on_m5",
            "one_minute.between5", "liquidity.alive")
    tfs = ("1m",)
    Params = OneMinuteParams
    gives_ = ("known_at", "lines")
    step_params = {"sweep": ("sweep", "false_breakout")}
    ref_params = ("entry", "hedge")

    def charts_needed(self, p) -> tuple[str, ...]:
        return ("5m",) + (("1h",) if "1h" in p.charts else ())

    def next(self, run, step, p, facts, after, deadline):
        e, h = _price(facts, p.entry), _price(facts, p.hedge)
        if e is None or h is None:
            return f"the price {p.entry if e is None else p.hedge} is missing"
        mk, side = run.market, run.hedge_side
        lo, hi = min(e, h), max(e, h)
        i1, i5 = mk.closed_by("1m", after), mk.closed_by("5m", after)
        if i1 < 0 or i5 < 0:
            return "no candle has closed yet"
        # a check after a take: only the lines known at the FIRST check count (settings
        # one_minute_new_lines_after_a_take - his F04 answer: the gap after the take is the entry)
        first = None if settings.get("one_minute_new_lines_after_a_take") else facts[p.sweep].get("first_check")
        seen = (lambda rows: rows) if first is None else (lambda rows: rows[rows["known_at"] <= first])
        m5 = mk.bars("5m").index
        found, takes = 0, []                     # takes: (moment, order, 5-minute candle, price, level id, kind, chart)
        if "1m" in p.charts:
            rows = seen(OM.between(mk.book1(run.fp), i1, lo, hi, side))
            found += len(rows)
            got = OM.first_take(rows, i1)
            if got is not None:
                r, j = got
                t1 = mk.bars("1m").index[j]
                takes.append((t1 + pd.Timedelta(minutes=1), 0, int(m5.searchsorted(t1, "right")) - 1,
                              float(r["b_price"] + r["slope"] * (j - r["x0"])), r["line_id"], "trendline", "1m"))
        five = tuple(c for c in p.charts if c != "1m")
        if five:
            l5 = mk.lines5(run.fp)
            rows = seen(OM.between5(l5[l5["tf"].isin(five)], i5, lo, hi, side))
            found += len(rows)
            got = OM.first_take(rows, i5)
            if got is not None:
                r, j = got
                px = float(r["b_price"] + r["slope5"] * (j - r["x0"]))
                takes.append((m5[j] + pd.Timedelta(minutes=5), 1, j, px, r["line_id"], "trendline", r["tf"]))
        if p.also:
            life = mk.liquidity(run.fp).life
            rows = LQ.alive(life, i5)
            rows = rows[(rows["side"] == side) & _kinds_col(rows).isin(p.also)]
            rows = seen(rows[(rows["price"] > lo + OM.INSIDE) & (rows["price"] < hi - OM.INSIDE)])
            found += len(rows)
            tk = rows["taken"].to_numpy()
            if (tk > i5).any():
                k = int(np.flatnonzero(tk > i5)[np.argmin(tk[tk > i5])])
                r, j = rows.iloc[k], int(tk[k])
                takes.append((m5[j] + pd.Timedelta(minutes=5), 2, j, float(r["price"]), r["level_id"], r["kind"],
                              r["tf"]))
        if not found:
            return {"known_at": after, "lines": 0}
        if not takes:
            return "wait: a line against the entry was never taken (#111: wait for it to be swept)"
        moment, _, k5, px, lid, kind, tf = min(takes, key=lambda x: (x[0], x[1]))
        if self.after_deadline(moment, deadline):
            return "wait: a line against the entry was not taken within the max wait (#111)"
        a5 = mk.arrays("5m")
        ext = a5["high"][k5] if run.sell else a5["low"][k5]
        return _Resweep(p.sweep, {"known_at": m5[k5] + pd.Timedelta(minutes=5), "bar": m5[k5], "price": px,
                                  "extreme": float(ext), "opposite": math.nan, "level_id": lid, "kind": kind,
                                  "level_tf": tf, "session": None, "waited_from": after,
                                  "first_check": facts[p.sweep].get("first_check", after)})


BLOCKS: dict[str, Block] = {b.name: b for b in (EqualLevels(), Sweep(), LevelWaiting(), BreakOfStructure(),
                                                  GapOfTheMove(), ComesBackTo(), FibMove(), BiasCheck(),
                                                  PremiumDiscount(), AdrZoneBreak(), HammerCheck(), AllTimeHigh(),
                                                  TrendlineTouch(), Triangle(), FalseBreakout(), Wedge(), ZoneEntry(),
                                                  ZoneBreak(), ZoneCheck(), OneMinute())}


# ================================================================================ do-not-enter rules and filters

class _Rule:
    name = ""
    means = ""
    uses: tuple[str, ...] = ()
    Params: type[_P] = _P
    step_params: dict[str, str | None] = {}
    ref_params: tuple[str, ...] = ()           # params that name a price a step found ("sweep.price")
    limit_only = False

    @staticmethod
    def default_since(f: Idea) -> str:
        return f.steps[0].id

    @staticmethod
    def default_refs(f: Idea) -> dict[str, str | None]:
        """The prices a rule reads when the form leaves them out."""
        return {}

    @staticmethod
    def charts_needed(p) -> tuple[str, ...]:
        """The charts it reads (the bias reads the hour / day / week)."""
        return ()


class _SinceParams(_P):
    since: str | None = None                   # the step from which it is checked (default: see each rule)


class EntryAlreadyReached(_Rule):
    name = "entry_already_reached"
    means = ("price already reached the entry price after the step named 'since' (default: the step that found the "
             "entry price) and before the signal: its first visit is gone - no trade (#68: only the first visit "
             "counts). A buy: the low reached it; a sell: the high")
    uses = ("the finest bars in the market",)
    Params = _SinceParams
    step_params = {"since": None}
    limit_only = True

    @staticmethod
    def default_since(f: Idea) -> str:
        return REF.fullmatch(f.entry.at).group(1)

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        tf = run.market.finest()
        a = run.market.arrays(tf)
        i0 = run.market.opens_from(tf, _after_found(run, facts, p.since))
        i1 = run.market.closed_by(tf, t) + 1
        seg = a["high"][i0:i1] >= entry if run.sell else a["low"][i0:i1] <= entry
        return "price already reached the entry before the signal (#68: the first visit is gone)" if seg.any() else None


def _after_found(run: _Run, facts: dict, sid: str) -> pd.Timestamp:
    """The close of the candle a step was found on (a swing's own candle, a gap's 3rd candle, a sweep candle) - what
    came after it counts, even the candles before the step became KNOWN (they are closed by the signal). A check
    step (no candle of its own): its moment."""
    ev = facts[sid]
    if ev.get("bar") is None:
        return pd.Timestamp(ev["known_at"])
    tf = next(s.tf for s in run.form.steps if s.id == sid)
    return pd.Timestamp(ev["bar"]) + pd.Timedelta(minutes=TF_MINUTES[tf])


class TargetAlreadyReached(_Rule):
    name = "target_already_reached"
    means = ("the move is already finished: price reached the (nearest) target after the step named 'since' "
             "(default: the first step) and before the signal - no trade (#119 'T05'; steps.md step 6)")
    uses = ("the finest bars in the market",)
    Params = _SinceParams
    step_params = {"since": None}

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        tf = run.market.finest()
        a = run.market.arrays(tf)
        i0 = run.market.opens_from(tf, _after_found(run, facts, p.since))
        i1 = run.market.closed_by(tf, t) + 1
        near = max(targets) if run.sell else min(targets)
        seg = a["low"][i0:i1] <= near if run.sell else a["high"][i0:i1] >= near
        return "the move is already finished: price reached the target before the signal (#119)" if seg.any() else None


class TargetReachedFirst(_Rule):
    name = "target_reached_first"
    means = ("while the limit order waits, price reaching the (nearest) target first cancels it: no trade (#119; "
             "steps.md step 6: 'If price already reached the equal lows before it came back, there is no trade'). "
             "The tester does it (Signal.cancel_at)")
    uses = ("tester: Signal.cancel_at",)
    limit_only = True


class ComesBackIntoAnotherGapFirst(_Rule):
    name = "price_comes_back_into_another_gap_first"
    means = ("his #68 (strategy-understanding §65e; his 7 Oct 2026 words on idea #4's setup 1: 'if you would have "
             "entered that in the first time only where it has reached the second fvg in that moment if it reached the "
             "first fvg that's the good entry'): the order at the gap is good only on price's FIRST visit back into "
             "the gaps the setup's move left - every 5-minute gap the trade's way from the sweep candle to one candle "
             "after the break (the old engine's arming candle) or the entry gap, whichever is later. If price comes "
             "back into one of them and turns away without reaching the entry, the order is cancelled: spent. Only "
             "for a limit at a gap_of_the_move step. The tester does it (Signal.first_visit_from / first_visit_to, "
             "fvg.first_visit)")
    uses = ("tester: Signal.first_visit_from / first_visit_to", "fvg.setup_gaps", "fvg.first_visit")
    limit_only = True

    @staticmethod
    def window(run: _Run, params: dict, facts: dict) -> dict | str:
        """The two times the tester needs, from what the gap step found (all known when the order is made)."""
        gid = REF.fullmatch(run.form.entry.at).group(1)
        step = next(s for s in run.form.steps if s.id == gid)
        if step.block != "gap_of_the_move":
            return "the entry is not at a gap_of_the_move step"
        p = params[gid]
        sweep = facts[p.sweep]
        frm = pd.Timestamp(sweep.get("gap_from", sweep["bar"]))
        last = max(pd.Timestamp(facts[gid]["bar"]), pd.Timestamp(facts[p.move]["bar"]) + pd.Timedelta(minutes=5))
        return {"first_visit_from": frm, "first_visit_to": last + pd.Timedelta(minutes=5)}


class StopTooNear(_Rule):
    name = "stop_too_near"
    means = ("a stop or hedge nearer to the entry than settings min_hedge_distance_adr of a day's range (Claude's "
             "0.02) is a rounding error, not a stop: the setup is dropped")
    uses = ("adr.adr_bars",)

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        tf = "5m" if run.market.has("5m") else run.market.finest()
        i = run.market.closed_by(tf, t)
        r = run.market.ruler(tf, run.fp)[i]
        if not math.isfinite(r):
            return "no day's range yet (the first 10 days of the bars)"
        if abs(entry - stop) < float(settings.get("min_hedge_distance_adr")) * r:
            return "the stop is too near the entry (settings min_hedge_distance_adr)"
        return None


class _RanParams(_P):
    level: str | None = None                   # the level taken (default: the first sweep step's price)
    extreme: str | None = None                 # the grab's extreme (default: the exit's price - the hedge point)


class GrabRanTooFar(_Rule):
    name = "grab_ran_too_far"
    means = ("the grab ran too far past the level it took: from the level to the grab's extreme (the hedge point) is "
             "more than settings grab_ran_too_far_adr of a day's range when the setup is made - no trade (his T3 "
             "no-reasons 'it moves so far', 'too big', 'too far it moved' - idea-001/examples.md; the number is "
             "Claude's; None = no line). Counted from the FIRST level the grab took: when a grab took several levels "
             "that lead to the same entry, the one farthest from the extreme decides (his #38: equal highs taken, "
             "then a second pair higher up)")
    uses = ("adr.adr_bars",)
    Params = _RanParams
    ref_params = ("level", "extreme")

    @staticmethod
    def default_refs(f: Idea) -> dict[str, str | None]:
        sweep = next((s.id for s in f.steps if s.block == "sweep"), None)
        return {"level": f"{sweep}.price" if sweep else None, "extreme": f.exit.at}

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        line = settings.get("grab_ran_too_far_adr")
        if line is None:
            return None
        level, ext = _price(facts, p.level), _price(facts, p.extreme)
        if level is None or ext is None:
            return f"the price {p.level if level is None else p.extreme} is missing"
        tf = "5m" if run.market.has("5m") else run.market.finest()
        r = run.market.ruler(tf, run.fp)[run.market.closed_by(tf, t)]
        if not math.isfinite(r):
            return "no day's range yet (the first 10 days of the bars)"
        ran = ext - level if run.sell else level - ext
        if ran > float(line) * r:
            return "the grab ran too far past the level it took (his 'it moves so far'; settings grab_ran_too_far_adr)"
        return None


TOO_FAR = GrabRanTooFar.name


class _ZoneRuleParams(_P):
    measure: Literal["near_the_zone", "share_of_the_day"] = "near_the_zone"
    charts: tuple[HTF, ...] = ("1h", "1d")


class PremiumDiscountRule(_Rule):
    name = "premium_discount"
    means = ("his #76: a sell at DISCOUNT (at, in or past the green ADR zone - measured as the block premium_discount) "
             "only when the hour AND the day are both bearish; a buy at PREMIUM (the red zone) only when both are "
             "bullish - else no trade ('I would avoid this'). Past the zone is the same rule, not a harder one (#76 C). "
             "Read on the entry price (a limit) or the close (next_open), at the signal")
    uses = ("premium_discount.zones", "bias.htf_bias")
    Params = _ZoneRuleParams

    @staticmethod
    def charts_needed(p) -> tuple[str, ...]:
        return tuple(p.charts)

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        i = run.market.closed_by("5m", t)
        wrong = PD.DISCOUNT if run.sell else PD.PREMIUM
        if i < 0 or PD.place(entry, run.market.adr_zones(run.fp, p.measure).iloc[i]) != wrong:
            return None
        d = HB.UP if run.direction == "buy" else HB.DOWN
        got = {tf: run.market.bias_at(tf, run.fp, t) for tf in p.charts}
        if all(v == d for v in got.values()):
            return None
        now = ", ".join(f"{tf} {HB.WORD[v]}" for tf, v in got.items())
        return f"a {run.direction} at {wrong} with the hour and the day not both with it (#76; {now})"


class NoSellAtTheAllTimeHigh(_Rule):
    name = "no_sell_at_the_all_time_high"
    means = ("strategy.md section 12 (#72): at the REAL all-time high, NO sells until the WEEKLY gives a break down + a "
             "gap (the week chart's bias is down). A buy is never stopped by it. Read at the signal on the 5-minute "
             "close (as the block all_time_high)")
    uses = ("bias.all_time_high", "bias.htf_bias")

    @staticmethod
    def charts_needed(p) -> tuple[str, ...]:
        return ("5m", "1w")

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        if not run.sell:
            return None
        i = run.market.closed_by("5m", t)
        if i < 0 or not run.market.ath5(run.fp)["at_ath"].iat[i]:
            return None
        if run.market.bias_at("1w", run.fp, t) == HB.DOWN:
            return None
        return "a sell at the real all-time high while the week chart is not down (strategy.md section 12, #72)"


class _PathParams(_P):
    charts: tuple[HTF, ...] = ("1h", "1d")     # his "day and hour, both" (#69)


class LineInPath(_Rule):
    name = "line_in_path"
    means = ("a trendline standing in the path (#69, strategy.md section 4): a line (2 or 3 touches, 5-minute or hour "
             "chart) not taken yet (#69 'a trend line fake out is ... a liquidity taken out') between the entry and "
             "the nearest target. It matters ONLY when the hour AND the day are both against the trade: then the line "
             "is the target ('I would exit at the trend line itself'), and if it is nearer than settings "
             "line_in_path_min_reward hedge distances ('50-50', his A: 0.63 skip, B 2.36 and D 1.42 take - the number "
             "is Claude's) there is no trade. With the hour and the day not both against, it changes nothing (his C, "
             "Q63 D)")
    uses = ("trendlines.lines", "trendlines.on_m5", "trendlines.in_path", "bias.htf_bias")
    Params = _PathParams

    @staticmethod
    def charts_needed(p) -> tuple[str, ...]:
        return ("5m",) + tuple(p.charts)

    @staticmethod
    def retarget(run: _Run, p, facts, t, entry, stop, targets) -> list | str:
        if not targets:
            return targets
        against = HB.DOWN if run.direction == "buy" else HB.UP
        if not all(run.market.bias_at(tf, run.fp, t) == against for tf in p.charts):
            return targets
        near = max(targets) if run.sell else min(targets)
        i = run.market.closed_by("5m", t)
        got = TL.in_path(run.market.lines5(run.fp), i, entry, near, run.direction)
        if got is None:
            return targets
        _, line = got
        risk = abs(entry - stop)
        if risk <= 0 or abs(line - entry) / risk < float(settings.get("line_in_path_min_reward")):
            return ("a trendline stands between the entry and the target and the hour and the day are both against "
                    "the trade - 50-50, no trade (#69)")
        k = targets.index(near)                  # the line takes the nearest target's place (the lots stay the same)
        return targets[:k] + [_round(line)] + targets[k + 1:]


class SessionParams(_P):
    only: tuple[str, ...]                      # his session names: Asia, London, New York
    at: str | None = None                      # the step whose moment is checked (default: the signal's)

    @field_validator("only")
    @classmethod
    def _names(cls, v):
        return _his_sessions(v)


class SessionFilter(_Rule):
    name = "session"
    means = ("keep a setup only when its moment (the close of the candle that makes it) falls in one of his session "
             "windows, India time: Asia 04:30-10:30, London 12:30-15:30, New York 17:30-20:30 (rules.py). It reads "
             "the CLOCK, not the level: 'the London high is taken' is the step sweep with which_session ['London']")
    uses = ("sessions.windows",)
    Params = SessionParams
    step_params = {"at": None}

    @staticmethod
    def keep(run: _Run, p, facts, t) -> bool:
        moment = facts[p.at]["known_at"] if p.at else t
        ist = clock.to_ist(pd.Timestamp(moment))
        m = ist.hour * 60 + ist.minute
        for name, a, b in SES.windows():
            if name in p.only and a.hour * 60 + a.minute < m <= b.hour * 60 + b.minute:
                return True
        return False


class BiasFilterParams(BiasParams):
    at: str | None = None                      # the step whose moment is checked (default: the signal's)


class BiasFilter(_Rule):
    name = "bias"
    means = ("keep a setup only when his bias agrees (the same check as the step 'bias', #69-#74): must 'with' = every "
             "chart named (default the hour AND the day) reads the trade's way; 'not_against' = not all of them read "
             "against it (#69: 'if the day and hour are both bullish, avoid the sell'). A hammer grab passes (#115). "
             "At the signal's moment, or at the step 'at'")
    uses = ("bias.htf_bias", "bias.bias_at", "hammer.hammers")
    Params = BiasFilterParams
    step_params = {"at": None}

    @staticmethod
    def charts_needed(p) -> tuple[str, ...]:
        return tuple(p.charts)

    @staticmethod
    def keep(run: _Run, p, facts, t) -> bool:
        moment = facts[p.at]["known_at"] if p.at else t
        return _bias_why_not(run, p, moment, facts) is None


class HigherTimeframeFilter(BiasFilter):
    name = "higher_timeframe"
    means = "the same filter as 'bias' (his words say both: 'the higher time frame', 'the bias')"


class UnfilledGapPastHedge(_Rule):
    name = "unfilled_gap_past_hedge"
    means = ("his #87 (strategy.md section 2: 'An unfilled gap on the hedge side is a reason to skip, unless recent "
             "obvious liquidity waits on the target side'): an unfilled gap past the hedge - a 3-candle gap or a gap "
             "up / down at the open (fvg.unfilled_past, Claude's numbers in settings unfilled_gap_*) - and none of his "
             "obvious liquidity (session high / low, trendline, equal highs / lows) waiting past the entry on the "
             "target side -> no trade: 'it will go to that unfilled gap, that will hit my hedge'. Off when settings "
             "unfilled_gap_is_a_magnet is False. Checked when the order is made; for his zone entry it is checked "
             "again while the order waits (L19b: machine/entries/closers.py gap_excuse - the tester's cancel_by)")
    uses = ("fvg.unfilled_past", "liquidity.alive")

    @staticmethod
    def check(run: _Run, p, facts, t, entry, stop, targets) -> str | None:
        if not settings.get("unfilled_gap_is_a_magnet"):
            return None
        i = run.market.closed_by("5m", t)
        r = run.market.ruler("5m", run.fp)[i]
        if i < 0 or not math.isfinite(r):
            return "no day's range yet (the first 10 days of the bars)"
        g = FVG.unfilled_past(run.market.bars("5m"), i, run.direction, stop, float(r))
        if g is None:
            return None
        far = float(settings.get("next_liquidity_max_adr")) * r
        if _pick_level(run, i, run.target_side, entry, _his_obvious(), None, "nearest", far) is not None:
            return None                                     # his D: "the market always moves towards the recent liquidity"
        return (f"an unfilled gap {g['lo']:.2f}-{g['hi']:.2f} past the hedge and no obvious liquidity past the entry "
                "- price can go to the gap and hit the hedge (#87)")


DO_NOT_ENTER: dict[str, _Rule] = {r.name: r for r in (EntryAlreadyReached(), TargetAlreadyReached(),
                                                        TargetReachedFirst(), ComesBackIntoAnotherGapFirst(),
                                                        StopTooNear(), GrabRanTooFar(),
                                                        PremiumDiscountRule(), NoSellAtTheAllTimeHigh(),
                                                        LineInPath(), UnfilledGapPastHedge())}
FILTERS: dict[str, _Rule] = {r.name: r for r in (SessionFilter(), BiasFilter(), HigherTimeframeFilter())}


def describe() -> str:
    """Every block, do-not-enter rule and filter a form may name - for the talker (L8) and for people."""
    def params(model) -> str:
        return ", ".join(f"{k} (required)" if v.is_required() else f"{k}={v.default!r}"
                         for k, v in model.model_fields.items()) or "-"

    out = ["BLOCKS (steps)"]
    for b in BLOCKS.values():
        kind = " · ".join(x for x in ("starts an idea" if b.can_start else "",
                                       "waits" if b.waits else "a check at that moment (wait 0)") if x)
        out.append(f"  {b.name}: {b.means}\n      {kind} · charts {', '.join(b.tfs) if b.tfs else 'any'} · "
                   f"params: {params(b.Params)} · gives: {', '.join(b.gives(None))}")
    for title, reg in (("DO NOT ENTER", DO_NOT_ENTER), ("FILTERS", FILTERS)):
        out.append(title)
        for r in reg.values():
            out.append(f"  {r.name}: {r.means}\n      params: {params(r.Params)}")
    return "\n".join(out)


# ================================================================================================== the result

@dataclass
class Compiled:
    idea_id: str
    version_id: str
    parent_idea: str | None
    tf: str                                    # the tester's timeframe (the idea's chart)
    form: Idea                                 # the form as compiled: readings and changes applied
    readings: dict[str, str]
    changes: dict[str, Any]
    management: str                            # "stop" | "hedge"
    resolved: settings.Resolved
    signals: list[Signal]
    funnel: dict[str, dict[str, int]]          # step id -> {"buy": n, "sell": n}: setups that reached it
    dropped: dict[tuple[str, str], int]        # (where, why) -> how many
    notes: list[str]
    window: tuple[pd.Timestamp, pd.Timestamp]
    trace: list[dict] | None = None            # trace=True: one row per start - the steps it reached (and what
                                               # each found: "facts"), where it was dropped and why, its signal, the
                                               # moments the grab kept going (L7b: why an example was missed)
    regrabs: int = 0                           # times a grab kept going and the break was looked for again
    waits: int = 0                             # times the 1-minute check waited for a line and the grab moved to its take
    hedge_lines: dict[str, dict] = field(default_factory=dict)  # signal id -> the trendline its hedge stands past
                                               # (a trendline_touch step whose hedge is the exit): line_id, slope (price
                                               # per candle of its chart), tf - the hedging tester moves the first hedge
                                               # with it (T12 M27, #69)

    def counts(self) -> dict[str, int]:
        c = Counter(s.direction for s in self.signals)
        return {"buy": c.get("buy", 0), "sell": c.get("sell", 0), "all": len(self.signals)}

    def table(self) -> pd.DataFrame:
        """Every signal, one row each (no money: lesson 16)."""
        cols = ["id", "known_at", "direction", "entry_type", "entry_price", "stop", "targets", "expiry_bars",
                "cancel_at", "lots"]
        rows = [{**s.model_dump(include=set(cols)), "targets": " / ".join(f"{x:.2f}" for x in s.targets)}
                for s in self.signals]
        return pd.DataFrame(rows, columns=cols)

    def text(self) -> str:
        f = self.form
        c = self.counts()
        rd = ", ".join(f"{k}={v}" for k, v in self.readings.items()) or "none open"
        lines = [
            f"COMPILED  {self.idea_id} · version {self.version_id} · {self.tf} chart · "
            f"{' and '.join(f.directions)} · readings: {rd} · {self.window[0]:%Y-%m-%d} to {self.window[1]:%Y-%m-%d} "
            f"(MT5 server time)",
            "  counts only - no money here (lesson 16: meaning is chosen before money is seen)",
            "Setups that reached each step (buy / sell):",
        ]
        for k, s in enumerate(f.steps, 1):
            n = self.funnel.get(s.id, {})
            lines.append(f"  {k}. {s.id:<10} {s.what[:60]:<60} {sum(n.values()):>7,} ({n.get('buy', 0):,} / "
                         f"{n.get('sell', 0):,})")
        lines.append(f"  signals    {'made at the last step, after the rules and filters':<60} {c['all']:>7,} "
                     f"({c['buy']:,} / {c['sell']:,})")
        if self.dropped:
            lines.append("Dropped:")
            for (where, why), n in sorted(self.dropped.items(), key=lambda x: -x[1]):
                lines.append(f"  {where}: {why} ... {n:,}")
        if self.regrabs:
            lines.append(f"The grab kept going {self.regrabs:,} times: the break was looked for again after the new "
                         "extreme, and the hedge point moved with it (his fixed rule a_grab_that_keeps_going_is_still_a_grab)")
        if self.waits:
            lines.append(f"The 1-minute check waited {self.waits:,} times for a line against the entry to be taken; "
                         "each take became the grab, and the steps after it were looked for again from it (#111)")
        ex = f.exit
        if self.management == "hedge":
            lines.append(f"Exit: a HEDGE at {ex.at} - this tester (L3) tests it as a plain stop there; his management "
                         "(settings.py, the T12 table) is carried for the hedging tester (L19a).")
        else:
            lines.append(f"Exit: a stop at {ex.at}" + (f", ${ex.beyond_usd:g} further" if ex.beyond_usd else ""))
        for b in self.resolved.breaks_his_rules:
            lines.append(f"NOT HIS RULE (a comparison only): {b}")
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


# ================================================================================================== compile

def compile_idea(form: Idea, market: Market, *, readings: dict[str, str] | None = None,
                 changes: dict[str, Any] | None = None, version_id: str | None = None,
                 trace: bool = False) -> Compiled:
    """The form -> signals (see the module text). readings: {unclear part: reading} for every open part (it never
    guesses - R13). changes: a version's own changes to the form (dotted paths, like a reading's). version_id:
    default = "v-" + the rule's fingerprint, so the same rule is always the same version (and one try)."""
    f, used, changes, resolved = prepare(form, readings, changes)
    vid = version_id or ("v-" + f.fingerprint()[:10])
    with settings.using(resolved):
        fp = settings.fingerprint(resolved)
        params, notes = _check(f, market)
        signals, funnel, dropped, traced, regrabs, waits, lines = _walk(f, market, params, fp, vid, trace)
    if f.exit.kind == "hedge":
        notes.append("the hedge is tested as a plain stop at the hedge point until the hedging tester (L19a) exists")
    return Compiled(f.id, vid, f.parent_idea, f.timeframe, f, used, changes, f.exit.kind, resolved, signals, funnel,
                    dropped, notes, market.window(), traced, regrabs, waits, lines)


def prepare(form: Idea, readings: dict[str, str] | None = None, changes: dict[str, Any] | None = None
            ) -> tuple[Idea, dict[str, str], dict[str, Any], settings.Resolved]:
    """What compile_idea does before it reads a bar: the readings (never guessed - R13) and the version's changes
    applied, level kinds as families, his fixed rules refused with his words, a missing block named, the settings
    resolved. Returns (the form as compiled, the readings used, the changes, its settings). version_of() = its id."""
    if not isinstance(form, Idea):
        raise TypeError("compile_idea takes an Idea (machine.ideas.form.Idea.load(...))")
    f, used = with_readings(form, readings)
    changes = dict(changes or {})
    if changes:
        f = apply_changes(f, changes, "the version's changes")
    f = _families(f)
    _refuse_fixed(f)
    _refuse_missing(f)
    resolved = settings.resolve(f.settings)
    if f.exit.kind == "hedge":
        settings.resolve(f.exit.management)            # his management: only real settings, never a fixed rule
    return f, used, changes, resolved


def version_of(form: Idea, readings: dict[str, str] | None = None, changes: dict[str, Any] | None = None) -> str:
    """The version id compile_idea gives these readings and changes ("v-" + the rule's fingerprint) - without bars."""
    return "v-" + prepare(form, readings, changes)[0].fingerprint()[:10]


MARKET_TFS = ("5m", "1h", "1d", "1w")            # Market.load's charts: the blocks' own charts (bias, zones, days)


def needed_tfs(form: Idea) -> tuple[str, ...]:
    """The charts a compile of this form reads: Market.load's own, the idea's chart, every step's chart, and the
    1-minute chart for the 1-minute check (L6b.4). A version on another timeframe (dimension 1) needs its chart."""
    extra = [form.timeframe, *(s.tf for s in form.steps)]
    if any(s.block == "one_minute" for s in form.steps):
        extra.append("1m")
    return tuple(dict.fromkeys([*MARKET_TFS, *extra]))


class _EveryChart:
    """A market that has every chart and no bars: for checking a form before anything is read."""
    tfs = tuple(typing.get_args(TF))

    @staticmethod
    def has(tf: str) -> bool:
        return True


def check_without_bars(form: Idea) -> list[tuple[str, str, str]]:
    """Everything compile_idea checks before it reads a bar, with no bars at all (the talker's drafts, L8): his fixed
    rules (FixedRuleError, with his words), the settings (SettingError), every step, name and price a part points at
    (CompileError). Returns what the form needs that is not built yet - (name, where, the prompt that builds it); a form
    that needs something is BLOCKED, and its steps are not checked further (their blocks do not exist)."""
    form = _families(form)
    _refuse_fixed(form)
    resolved = settings.resolve(form.settings)
    if form.exit.kind == "hedge":
        settings.resolve(form.exit.management)
    try:
        _refuse_missing(form)
    except MissingBlockError as err:
        return list(err.missing)
    with settings.using(resolved):
        _check(form, _EveryChart())
    return []


def missing_in_any_reading(form: Idea) -> list[tuple[str, str, str]]:
    """check_without_bars on the form AND on each reading of its unclear parts: what ANY version of the idea would need
    that is not built yet. A reading that needs a missing block would stop the tune halfway (found on idea #2's U2-B,
    L6b.3), so the intake blocks the idea on it before anything runs."""
    out = list(check_without_bars(form))
    for u in form.unclear:
        for r in u.readings:
            if not r.changes:
                continue
            for name, where, prompt in check_without_bars(apply_changes(form, r.changes, f"reading {r.id} of {u.id}")):
                if name not in {x[0] for x in out}:
                    out.append((name, f"reading {r.id} of {u.id}: {where}", prompt))
    return out


# ---- the refusals ------------------------------------------------------------------------------------------------

def _keys(obj: Any, where: str) -> Iterable[tuple[str, str, Any]]:
    """Every (key, where, value) inside nested dicts / lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), where, v
            yield from _keys(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for k, v in enumerate(obj):
            yield from _keys(v, f"{where}[{k}]")


def _refuse_fixed(f: Idea) -> None:
    """Refuse (with his words) any part of the form that changes one of his fixed rules (LEARNED #121)."""
    hits = []
    places = [(f.settings, "settings")]
    if f.exit.kind == "hedge":
        places.append((f.exit.management, "exit.management"))
    places += [(s.params, f"steps '{s.id}'.params") for s in f.steps]
    places += [(r.params, f"do_not_enter '{r.name}'.params") for r in f.do_not_enter]
    places += [(r.params, f"filters '{r.name}'.params") for r in f.filters]
    for obj, where in places:
        for k, w, v in _keys(obj, where):
            if rules.fixed_name(k) is not None:
                hits.append(f"{w}: {rules.refusal(k, v)}")
    # his #65: he enters at the NEAR edge of the gap - an entry at any other edge changes it
    gap_steps = {s.id for s in f.steps if s.block == "gap_of_the_move"}
    if f.entry.at:
        step, name = REF.fullmatch(f.entry.at).groups()
        if step in gap_steps and name != "near":
            hits.append(f"entry.at: {rules.refusal('where_in_the_gap_he_buys', f.entry.at)} Use '{step}.near'.")
    if hits:
        raise rules.FixedRuleError(f"{f.id} changes his fixed rules - refused:\n" + "\n".join(f"  - {h}" for h in hits))


def _refuse_missing(f: Idea) -> None:
    missing = []
    for s in f.steps:
        if s.block not in BLOCKS:
            missing.append((s.block, f"step '{s.id}'", _planned(s.block)))
        for k in ("kinds",):
            for kind in s.params.get(k) or ():
                if kind == "zone":
                    continue                    # built, but not a level that is swept: the step's own check says so
                if kind not in BUILT_KINDS:
                    missing.append((kind, f"step '{s.id}' params.kinds", _planned(kind)))
    for t in f.targets:
        for kind in t.kinds or ():
            if kind not in TARGET_KINDS:
                missing.append((kind, "targets kinds", _planned(kind)))
    for r in f.do_not_enter:
        if r.name not in DO_NOT_ENTER:
            missing.append((r.name, "do_not_enter", _planned(r.name)))
    for r in f.filters:
        if r.name not in FILTERS:
            missing.append((r.name, "filters", _planned(r.name)))
    s2 = settings.resolve(f.settings)
    # his words still open (four-boxes Box 2): the step waits for his answer, unless the form names the number
    for s in f.steps:
        if s.block == "trendline_touch" and s2["trendline_min_steepness_adr"] is None:
            missing.append(("steep", f"step '{s.id}' (trendline_touch): how steep a line must be for the touch "
                                     "entry - settings trendline_min_steepness_adr (#127 'the steep of the line "
                                     "decides')", _planned("steep")))
        if s.block == "triangle" or (s.block == "false_breakout" and s.params.get("of")):
            if s2["big_move_adr"] is None and not any(m[0] == "big_move" for m in missing):
                missing.append(("big_move", f"step '{s.id}' ({s.block}): how big 'a big move' before a triangle is - "
                                            "settings big_move_adr (strategy.md section 4)", _planned("big_move")))
    if missing:
        raise MissingBlockError(f.id, missing)


# ---- the checks of meaning ----------------------------------------------------------------------------------------

def _check(f: Idea, market: Market) -> tuple[dict[str, Any], list[str]]:
    """Every step's params, every name a part points at, every timeframe. All problems at once."""
    problems, notes = [], []
    params: dict[str, Any] = {}
    gives: dict[str, tuple[str, ...]] = {}
    for k, s in enumerate(f.steps):
        b = BLOCKS[s.block]
        where = f"steps[{k}] '{s.id}' ({s.block})"
        before = list(f.steps[:k])
        if k == 0 and not b.can_start:
            problems.append(f"{where}: this block cannot start an idea; blocks that can: "
                            f"{', '.join(n for n, x in BLOCKS.items() if x.can_start)}")
        if k > 0 and type(b).next is Block.next:
            problems.append(f"{where}: this block can only START an idea (be the first step)")
        if b.tfs and s.tf not in b.tfs:
            problems.append(f"{where}: works on {', '.join(b.tfs)} charts, not {s.tf}")
        if not market.has(s.tf):
            problems.append(f"{where}: needs {s.tf} bars; the market has {', '.join(market.tfs)}")
        if k > 0:
            if b.waits and s.max_wait_bars == 0:
                problems.append(f"{where}: this step WAITS for something to happen - give max_wait_bars 1 or more, "
                                "or \"no limit\"")
            if not b.waits and s.max_wait_bars != 0:
                problems.append(f"{where}: this step is a CHECK made at the moment the step before is known - its "
                                "max_wait_bars must be 0")
        try:
            p = b.Params.model_validate(s.params)
        except ValidationError as err:
            problems.append(f"{where} params: " + readable(err).split("\n", 1)[1].strip().replace("\n  - ", "; ")
                            .lstrip("- "))
            continue
        # step-params: fill the default, check the named step is earlier (and of the right block)
        fill = {}
        for name, want in b.step_params.items():
            sid = getattr(p, name)
            if sid is None:
                sid = b.default_step(name, before)
                if sid is None and want is not None:
                    problems.append(f"{where}: needs a '{_names(want)}' step before it (params.{name})"
                                    + (" - the entry gap is the FIRST gap after the liquidity is taken (#61, #62, "
                                       "#131)" if _is("sweep", want) else ""))
                    continue
                if sid is None and name == "after" and before:
                    sid = before[-1].id
                fill[name] = sid
            elif sid not in [x.id for x in before]:
                problems.append(f"{where}: params.{name} = {sid!r} must name an EARLIER step "
                                f"({', '.join(x.id for x in before) or 'there is none'})")
            elif not _is({x.id: x for x in before}[sid].block, want):
                problems.append(f"{where}: params.{name} = {sid!r} must be a '{_names(want)}' step")
        if fill:
            p = p.model_copy(update=fill)
        for tf in b.charts_needed(p):
            if not market.has(tf):
                problems.append(f"{where}: reads the {tf} chart; the market has {', '.join(market.tfs)}")
        if s.block == "sweep" and p.of is not None and "level_id" not in gives.get(p.of, ()):
            problems.append(f"{where}: params.of = {p.of!r} must be a step that finds a level (equal_levels, "
                            "level_waiting)")
        if s.block == "false_breakout" and p.of is not None and {x.id: x for x in before}.get(p.of) is not None                 and {x.id: x for x in before}[p.of].block != "triangle":
            problems.append(f"{where}: params.of = {p.of!r} must be a 'triangle' step (without 'of': his liquidity)")
        if s.block == "gap_of_the_move":
            for name in ("sweep", "move"):
                sid = getattr(p, name)
                if sid and {x.id: x for x in before}[sid].tf != s.tf:
                    problems.append(f"{where}: the gap, the sweep and the break must be on the same chart "
                                    f"(step {sid!r} is {({x.id: x for x in before}[sid]).tf}, this is {s.tf})")
        if s.block == "one_minute":
            p = p.model_copy(update={n: v for n, v in (("entry", f.entry.at), ("hedge", f.exit.at))
                                     if getattr(p, n) is None})
            m = REF.fullmatch(p.entry) if p.entry else None
            gap = {x.id: x for x in before}.get(m.group(1)) if m else None
            if gap is None or gap.block != "gap_of_the_move" or gap.tf != "5m":
                problems.append(f"{where}: the 1-minute check is only for a 5-minute gap entry (strategy.md section 5: "
                                "'1-min = never the entry chart; used as a check before a 5-min gap entry') - its "
                                f"entry ({p.entry or 'none: a next_open entry'}) must be a price of an EARLIER "
                                "5-minute gap_of_the_move step. The zone entries (#99, #111) join when L6b.2 builds "
                                "them")
            if p.hedge is None:
                problems.append(f"{where}: the check reads the hedge price (exit.at)")
        for name in b.ref_params:
            ref = getattr(p, name)
            if ref:
                problems += _check_ref(ref, f"{where} params.{name}", gives, [x.id for x in before])
        params[s.id] = p
        gives[s.id] = b.gives(p)
    all_ids = [s.id for s in f.steps]
    for where, ref in f._refs():
        problems += _check_ref(ref, where, gives, all_ids)
    for k, t in enumerate(f.targets):
        if t.kind == "ratio" and f.entry.how != "limit":
            problems.append(f"targets[{k}]: a ratio target needs the entry price known in advance (a limit entry); "
                            "a next_open entry fills at a price nobody knows yet")
        if t.kind == "next_liquidity" and not market.has("5m"):
            problems.append(f"targets[{k}]: the next liquidity needs 5m bars (the liquidity blocks are on 5m)")
        if t.kind == "next_liquidity" and t.pick == "by_the_dial":
            for tf in DIAL_CHARTS:
                if not market.has(tf):
                    problems.append(f"targets[{k}]: pick by_the_dial reads the {tf} chart; the market has "
                                    f"{', '.join(market.tfs)}")
    for group, reg, rows in (("do_not_enter", DO_NOT_ENTER, f.do_not_enter), ("filters", FILTERS, f.filters)):
        for k, r in enumerate(rows):
            rule = reg[r.name]
            where = f"{group}[{k}] '{r.name}'"
            if rule.limit_only and f.entry.how != "limit":
                problems.append(f"{where}: only for a limit entry (a next_open entry fills at once - nothing waits)")
            elif r.name == ComesBackIntoAnotherGapFirst.name and \
                    next((s.block for s in f.steps if s.id == REF.fullmatch(f.entry.at).group(1)), None) \
                    != "gap_of_the_move":
                problems.append(f"{where}: only for a limit at a gap_of_the_move step (the setup's gaps, #68)")
            try:
                p = rule.Params.model_validate(r.params)
            except ValidationError as err:
                problems.append(f"{where} params: " + readable(err).split("\n", 1)[1].strip().replace("\n  - ", "; ")
                                .lstrip("- "))
                continue
            for name in rule.step_params:
                sid = getattr(p, name)
                if sid is not None and sid not in all_ids:
                    problems.append(f"{where}: params.{name} = {sid!r} names no step ({', '.join(all_ids)})")
            for tf in rule.charts_needed(p):
                if not market.has(tf):
                    problems.append(f"{where}: reads the {tf} chart; the market has {', '.join(market.tfs)}")
            if "since" in rule.step_params and p.since is None and not problems:
                p = p.model_copy(update={"since": rule.default_since(f)})
            fill = {n: v for n, v in rule.default_refs(f).items() if getattr(p, n) is None}
            if fill:
                p = p.model_copy(update=fill)
            for name in rule.ref_params:
                ref = getattr(p, name)
                if ref is None:
                    problems.append(f"{where}: give params.{name} - the price a step found, like 'sweep.price'")
                else:
                    problems += _check_ref(ref, f"{where} params.{name}", gives, all_ids)
            params[f"{group}:{k}"] = p
    if problems:
        raise CompileError(f"{f.id} cannot be compiled:\n" + "\n".join(f"  - {x}" for x in problems))
    used = {s.block for s in f.steps}
    if used & {"sweep", "level_waiting"} or any(t.kind == "next_liquidity" for t in f.targets):
        notes.append("his obvious liquidity here = session highs / lows + equal highs / lows + trendlines (5-minute and "
                     "hour, 3 touches; a line's price moves - a target on it is its price when the signal is made)")
    if "zone_break" in used and settings.get("zone_break_inner_1m_lines") and not market.has("1m"):
        notes.append("the 1-minute lines inside a zone (#98 C, L6b.4) were NOT read: the market has no 1-minute bars - "
                     "load them (Market.load(..., tfs=('1m', ...))) for his rule")
    return params, notes


def _check_ref(ref: str, where: str, gives: dict, allowed: list[str]) -> list[str]:
    m = REF.fullmatch(ref)
    if not m:
        return [f"{where}: {ref!r} must name a step and what it found, like 'gap.near'"]
    step, name = m.groups()
    if step not in allowed:
        return [f"{where}: {ref!r} - step {step!r} comes later or does not exist; it may name "
                f"{', '.join(allowed) or 'no step'}"]
    if step in gives and name not in gives[step]:
        return [f"{where}: step {step!r} does not find {name!r}; it finds {', '.join(gives[step])}"]
    return []


# ---- the walk ------------------------------------------------------------------------------------------------------

def _walk(f: Idea, market: Market, params: dict, fp: str, vid: str, trace: bool = False):
    funnel: dict[str, dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0})
    dropped: Counter = Counter()
    traced: list[dict] | None = [] if trace else None
    found = []                                            # (known_at, direction order, start order, facts, trace)
    regrabs = waits = 0
    ids = [s.id for s in f.steps]
    for d_order, direction in enumerate(sorted(f.directions)):
        run = _Run(market, f, direction, fp)
        first = f.steps[0]
        for n_start, ev in enumerate(BLOCKS[first.block].start(run, first, params[first.id])):
            funnel[first.id][direction] += 1
            facts, after, ok = {first.id: ev}, pd.Timestamp(ev["known_at"]), True
            searched_from = {}                            # step id -> the moment it was looked for from
            reached = {first.id}
            rec = None
            if trace:
                rec = {"direction": direction, "start": after, "steps": {first.id: after}, "dropped_at": None,
                       "why": None, "signal": None, "regrabs": 0, "regrab_at": [], "waits": 0, "wait_for": []}
                traced.append(rec)
            k = 1
            while k < len(f.steps):
                step = f.steps[k]
                dl = run.deadline(step.tf, after, step.max_wait_bars)
                searched_from[step.id] = after
                got = BLOCKS[step.block].next(run, step, params[step.id], facts, after, dl)
                if isinstance(got, _Resweep):
                    # the 1-minute check (#111): a line against the entry was taken - that take is the grab now, and
                    # every step after the sweep step is looked for again from it (the hedge moves with the new grab)
                    t = pd.Timestamp(got.event["known_at"])
                    if t <= after:
                        raise RuntimeError(f"block {step.block} waited for a take at {t}, not after {after}")
                    k = ids.index(got.step)
                    for sid in ids[k + 1:]:
                        facts.pop(sid, None)
                    facts[got.step], after = got.event, t
                    waits += 1
                    if rec is not None:
                        rec["waits"] += 1
                        rec["wait_for"].append((got.event["level_id"], t))
                        rec["steps"][got.step] = t
                    k += 1
                    continue
                if isinstance(got, _Regrab):
                    # the grab kept going (his fixed rule): the break is looked for again after the new extreme,
                    # within its own max wait from the step before it; the steps after it are found again
                    k = ids.index(got.step)
                    step = f.steps[k]
                    if not getattr(BLOCKS[step.block], "can_resume", False):
                        raise RuntimeError(f"step {step.id!r} ({step.block}) cannot be looked for again")
                    after = searched_from[step.id]
                    dl = run.deadline(step.tf, after, step.max_wait_bars)
                    at = got.at
                    # a new grab: a 1-minute check after it is a FIRST check again - "only the lines known at the
                    # first check" (settings one_minute_new_lines_after_a_take) is for the gap right after a take, not
                    # for a grab that went on for hours (his F04, #111: a line against the entry means wait). Found
                    # 7 Oct 2026: with his break level the 17:55 buy he criticised came back through such a grab.
                    for sid, fx in list(facts.items()):
                        if isinstance(fx, dict) and "first_check" in fx:
                            facts[sid] = {a: b for a, b in fx.items() if a != "first_check"}
                    got = BLOCKS[step.block].next(run, step, params[step.id], facts, after, dl, resume=at)
                    regrabs += 1
                    if rec is not None:
                        rec["regrabs"] += 1
                        rec["regrab_at"].append(at)
                    if isinstance(got, str):
                        got = f"the grab kept going, then {got}"
                if isinstance(got, str):
                    dropped[(step.id, got)] += 1
                    if rec is not None:
                        rec.update(dropped_at=step.id, why=got)
                    ok = False
                    break
                t = pd.Timestamp(got["known_at"])
                if t < after or (dl is not None and t > dl):
                    raise RuntimeError(f"block {step.block} returned a step known at {t}, outside {after} .. {dl}")
                facts[step.id], after = got, t
                if step.id not in reached:
                    reached.add(step.id)
                    funnel[step.id][direction] += 1
                if rec is not None:
                    rec["steps"][step.id] = t
                k += 1
            if rec is not None:
                rec["facts"] = facts
            if ok:
                found.append((after, d_order, n_start, direction, facts, rec))
                then = facts[ids[-1]].get("then")
                if then is not None:
                    # the last step gave a later order in its place (the second gap, #124): a newer signal, made at
                    # its own moment from what was known then - the tester cancels the waiting older one (M39)
                    f2 = {**facts, ids[-1]: {k: v for k, v in then.items() if k != "then"}}
                    rec2 = None
                    if rec is not None:
                        rec2 = {**rec, "steps": {**rec["steps"], ids[-1]: pd.Timestamp(then["known_at"])},
                                "facts": f2, "signal": None, "in_place_of": True}
                        traced.append(rec2)
                    found.append((pd.Timestamp(then["known_at"]), d_order, n_start, direction, f2, rec2))
    found.sort(key=lambda x: x[:3])
    made = []
    for t, _, _, direction, facts, rec in found:
        run = _Run(market, f, direction, fp)
        made.append((t, direction, facts, rec, _signal(run, params, facts, t), _entry_key(run, facts, t)))
    # the grab's run counts from the FIRST level it took: when several levels taken in one grab lead to the same entry
    # and one of them is too far from the grab's extreme, the entry is not taken from a later, nearer level either
    ran_far = {key for *_, got, key in made if isinstance(got, tuple) and got[0] == f"do not enter: {TOO_FAR}"}
    touch_steps = [s.id for s in f.steps if s.block == "trendline_touch"]
    signals, seen, per_moment, lines = [], set(), Counter(), {}
    for t, direction, facts, rec, got, key in made:
        if not isinstance(got, tuple) and key in ran_far:
            got = (f"do not enter: {TOO_FAR}", "the same grab took an earlier level and ran too far past it")
        if isinstance(got, tuple):
            dropped[got] += 1
            if rec is not None:
                rec.update(dropped_at=got[0], why=got[1])
            continue
        if key in seen:
            dropped[("signal", "the same entry reached from an earlier start (one signal)")] += 1
            if rec is not None:
                rec.update(dropped_at="signal", why="the same entry reached from an earlier start")
            continue
        seen.add(key)
        k = per_moment[(direction, t)]
        per_moment[(direction, t)] += 1
        sid = f"{f.id}.{vid}.{direction[0].upper()}{t:%Y%m%d-%H%M}" + (f"-{k}" if k else "")
        try:
            signals.append(Signal(id=sid, idea_id=f.id, version_id=vid, parent_idea=f.parent_idea, **got))
        except ValidationError as err:
            why = err.errors(include_url=False)[0]["msg"].removeprefix("Value error, ")
            dropped[("signal", "not a valid order: " + re.sub(r" -?[\d.]+$", "", why))] += 1   # one line per kind
            if rec is not None:
                rec.update(dropped_at="signal", why=f"not a valid order: {why}")
            continue
        if rec is not None:
            rec["signal"] = sid
        if touch_steps and f.exit.at == f"{touch_steps[0]}.hedge":   # its hedge point is past the line: it moves
            ev = facts[touch_steps[0]]                                  # with it (M27)
            lines[sid] = {"line_id": ev["line_id"], "slope": ev["slope"],
                          "tf": next(s.tf for s in f.steps if s.id == touch_steps[0])}
    return signals, dict(funnel), dict(dropped), traced, regrabs, waits, lines


def _entry_key(run: _Run, facts: dict, t: pd.Timestamp) -> tuple:
    """(direction, moment, entry type, entry price): two starts with the same key make one signal."""
    if run.form.entry.how != "limit":
        return (run.direction, t, "next_open", None)
    e = _price(facts, run.form.entry.at)
    return (run.direction, t, "limit", None if e is None else _round(e))


DIAL_CHARTS = ("1h", "1d")


def dial(run: _Run, t) -> str:
    """His bias dial (#82, strategy.md section 6 'a DIAL, not a gate'): 'with' = the hour AND the day both read the
    trade's way at moment t; 'against' = both read the other way; else 'mixed'."""
    d = HB.UP if run.direction == "buy" else HB.DOWN
    got = [run.market.bias_at(tf, run.fp, t) for tf in DIAL_CHARTS]
    return "with" if all(v == d for v in got) else "against" if all(v == -d for v in got) else "mixed"


def _dial_target(run: _Run, i: int, t, ref: float, kinds, far: float | None):
    """The target by the dial (#82; strategy.md section 10 trailing and trend extension):
      with     don't exit at the first target - the next liquidity beyond it (the tester's targets are fixed: the
               'unless an opposite setup forms there' and the trailing itself need the hedging tester, L19a);
      against  the first target, and a recent 5-minute swing counts as a quick one (#91);
      mixed    the first target (the nearest)."""
    how = dial(run, t)
    lv = _pick_level(run, i, run.target_side, ref, kinds, None, "nearest", far, five_minute_swings=how == "against")
    if lv is None or how != "with":
        return lv
    rest = None if far is None else far - abs(float(lv.price) - ref)
    nxt = _pick_level(run, i, run.target_side, float(lv.price), kinds, None, "nearest", rest)
    return nxt if nxt is not None else lv


def _signal(run: _Run, params: dict, facts: dict, t: pd.Timestamp) -> dict | tuple[str, str]:
    """The order made at moment t from what the steps found, or (where, why) it is not made."""
    f, sell = run.form, run.sell
    out: dict[str, Any] = {"known_at": t, "direction": run.direction}
    if f.entry.how == "limit":
        e = _price(facts, f.entry.at)
        if e is None:
            return ("entry", f"the price {f.entry.at} is missing")
        e = _round(e)
        w = f.entry.max_wait_bars
        out.update(entry_type="limit", entry_price=e, expiry_bars=NO_LIMIT_BARS if w == NO_LIMIT else int(w))
        ref = e
    else:
        tf = run.market.finest()
        ref = float(run.market.arrays(tf)["close"][run.market.closed_by(tf, t)])
        e = ref                                           # only to measure distances: the fill is the next open
        out.update(entry_type="next_open")
    raw = _price(facts, f.exit.at)
    if raw is None:
        return ("exit", f"the price {f.exit.at} is missing")
    stop = _round(raw + f.exit.beyond_usd if sell else raw - f.exit.beyond_usd)
    targets = []
    for k, tg in enumerate(f.targets):
        if tg.kind == "ratio":
            risk = abs(e - stop)
            px = e - tg.ratio * risk if sell else e + tg.ratio * risk
        elif tg.kind == "price":
            px = _price(facts, tg.at)
            if px is None:
                return ("targets", f"the price {tg.at} is missing")
        else:
            i = run.market.closed_by("5m", t)
            r = run.market.ruler("5m", run.fp)[i]
            far = float(settings.get("next_liquidity_max_adr")) * r if math.isfinite(r) else None
            kinds = tg.kinds or BUILT_KINDS
            lv_kinds = tuple(k for k in kinds if k != "zone")
            lv = None
            if lv_kinds and tg.pick == "by_the_dial":
                lv = _dial_target(run, i, t, ref, lv_kinds, far)
            elif lv_kinds:
                lv = _pick_level(run, i, run.target_side, ref, lv_kinds, None, tg.pick, far)
            px = None if lv is None else float(lv.price)
            # L6b.2 "Target = next liquidity / zone": a live zone's near edge is one more candidate for the nearest
            # target - not when the dial says 'with' (#82: then he does not exit at the first target)
            if "zone" in kinds and not (tg.pick == "by_the_dial" and dial(run, t) == "with"):
                ze = ZN.nearest_edge(run.market.zone_life(run.fp), i, run.target_side, ref, far)
                if ze is not None and abs(_round(ze["price"]) - _round(ref)) < config.POINT / 2:
                    ze = None                  # under a cent past the entry on his chart: not a target
                if ze is not None and (px is None or abs(ze["price"] - ref) < abs(px - ref)):
                    px = ze["price"]
            if px is None:
                return ("targets", "no next liquidity on the target side - a next target must really exist (#119)")
        targets.append(_round(px))
    # a rule that may MOVE the target (a line in the path, #69) runs first, so every other rule reads the final targets
    for k, rule in enumerate(f.do_not_enter):
        r = DO_NOT_ENTER[rule.name]
        if hasattr(r, "retarget"):
            got = r.retarget(run, params[f"do_not_enter:{k}"], facts, t, e, stop, targets)
            if isinstance(got, str):
                return (f"do not enter: {rule.name}", got)
            targets = got
    for k, rule in enumerate(f.do_not_enter):
        r = DO_NOT_ENTER[rule.name]
        if rule.name == "target_reached_first":
            out["cancel_at"] = max(targets) if sell else min(targets)
            continue
        if rule.name == ComesBackIntoAnotherGapFirst.name:
            got = ComesBackIntoAnotherGapFirst.window(run, params, facts)
            if isinstance(got, str):
                return (f"do not enter: {rule.name}", got)
            out.update(got)
            continue
        if hasattr(r, "retarget"):
            continue
        why = r.check(run, params[f"do_not_enter:{k}"], facts, t, e, stop, targets)
        if why:
            return (f"do not enter: {rule.name}", why)
    for k, flt in enumerate(f.filters):
        if not FILTERS[flt.name].keep(run, params[f"filters:{k}"], facts, t):
            return (f"filter: {flt.name}", "outside what the filter keeps")
    out.update(stop=stop, targets=tuple(targets), levels="price",
               lots=round(float(settings.get("lot_size")) * len(targets), 2))
    return out
