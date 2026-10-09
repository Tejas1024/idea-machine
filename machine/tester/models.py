"""What goes into the tester and what comes out (L3): a Signal ("buy / sell here") and a Trade.

Prices are CHART prices: the BID scale (the price data is bid; his MT5 chart shows the bid).
Times are MT5 SERVER time without a time zone, like load_bars' index.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_serializer, model_validator

from machine import costs
from machine.guards import check_dates

Direction = Literal["buy", "sell"]
EntryType = Literal["next_open", "limit"]
Levels = Literal["price", "from_fill"]
ExitReason = Literal["stop", "target", "past at fill"]

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}


def known_at_close(bar_open, tf: str) -> pd.Timestamp:
    """The moment a signal made by a bar is KNOWN: the bar's close = its open time + the timeframe.
    A 5-min bar that opens 09:55 is known at 10:00, so its signal fills at the 10:00 open at the earliest."""
    if tf not in TF_MINUTES:
        raise ValueError(f"timeframe {tf!r} unknown; use one of {', '.join(TF_MINUTES)}.")
    return pd.Timestamp(bar_open) + pd.Timedelta(minutes=TF_MINUTES[tf])


def _server_time(value) -> dt.datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        raise ValueError(f"{value!r} has a time zone; give MT5 server time without one (like load_bars' index)")
    check_dates(ts)                                  # the sealed years: LEARNED #96, #104
    return ts.to_pydatetime()


class Signal(BaseModel):
    """One setup: "buy / sell here", known at one moment in the past.

    known_at     when the signal is KNOWN: the CLOSE of the bar that made it (use known_at_close(bar_open, tf)).
                 It fills only on a 1-min bar that opens at or after known_at — never on its own bar (lesson B2).
    entry_type   "next_open": at the open of the next bar.
                 "limit": waits at entry_price (a price on his chart) for up to expiry_bars bars, then is cancelled.
    stop         the price that closes the trade at a loss. None = no stop (then at least one target).
    targets      the prices that close it in profit. The lots are split equally among them (0.02 lot, 2 targets =
                 0.01 at each); they are kept nearest first.
    levels       "price": stop and targets are prices on his chart.
                 "from_fill": they are gold dollars away from the fill price (stop 5 = $5 below the fill of a buy).
    expiry_bars  bars of the test's timeframe a limit order waits (bars t+1 ... t+N). Needed for a limit.
    cancel_at    a limit still WAITING is cancelled when price reaches this chart price before it fills - the
                 do-not-enter rule "if price already reached the target before it came back, there is no trade"
                 (#119; plan/work/idea-001/steps.md step 6). A buy's is above its entry (the bid high reaches it), a
                 sell's below (the bid low). A cancel and a fill in the same minute: no trade (the order inside the
                 minute is unknown; T3's walk did the same). Only for a limit (L7a).
    first_visit_from, first_visit_to   his #68 on a limit at a gap (the do-not-enter rule
                 "price_comes_back_into_another_gap_first"; machine/blocks/fvg.first_visit): the open of the candle
                 where the setup's grab began, and the close of the last candle whose gaps count. Every 5-minute gap
                 the trade's way whose 3rd candle is 2 candles after the first ... the last is one of the setup's gaps;
                 if price comes back into one of them and turns away without reaching the entry, the waiting order is
                 cancelled (spent). Both are known when the order is made: the last candle closes at most one candle
                 after it (the old engine's arming candle, L6-differences item 6), and its gap is used only from the
                 candle after. None = not checked (left out of a dump).
    lots         0.01, 0.02 ... only (LEARNED #116).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    known_at: dt.datetime
    direction: Direction
    entry_type: EntryType = "next_open"
    entry_price: float | None = None
    stop: float | None = None
    targets: tuple[float, ...] = ()
    levels: Levels = "price"
    expiry_bars: int | None = None
    cancel_at: float | None = None
    first_visit_from: dt.datetime | None = None
    first_visit_to: dt.datetime | None = None
    lots: float = 0.01
    idea_id: str
    version_id: str
    parent_idea: str | None = None

    @field_validator("known_at", mode="before")
    @classmethod
    def _known_at(cls, v):
        return _server_time(v)

    @field_validator("first_visit_from", "first_visit_to", mode="before")
    @classmethod
    def _first_visit_times(cls, v):
        return None if v is None else _server_time(v)

    @model_serializer(mode="wrap")
    def _dump(self, handler):
        # the #68 check's times only when set: every signal made without it dumps (and hashes) as before
        d = handler(self)
        if isinstance(d, dict):
            for k in ("first_visit_from", "first_visit_to"):
                if k in d and d[k] is None:
                    del d[k]
        return d

    @field_validator("id", "idea_id", "version_id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _check(self) -> "Signal":
        for name, x in (("entry_price", self.entry_price), ("stop", self.stop), ("cancel_at", self.cancel_at),
                        *(("target", t) for t in self.targets)):
            if x is not None and (not math.isfinite(x) or x <= 0):
                raise ValueError(f"{name}={x!r}: must be a number above 0")
        try:
            steps = costs.lot_steps(self.lots)
        except costs.CostError as err:
            raise ValueError(str(err)) from None
        if self.targets and steps % len(self.targets):
            raise ValueError(f"lots={self.lots} cannot be split equally among {len(self.targets)} targets in "
                             "0.01 steps (LEARNED #116)")
        if self.stop is None and not self.targets:
            raise ValueError("a signal needs a stop, a target, or both — else the trade can never close")
        if self.entry_type == "limit":
            if self.entry_price is None:
                raise ValueError("a limit order needs entry_price")
            if self.expiry_bars is None or self.expiry_bars < 1:
                raise ValueError("a limit order needs expiry_bars ≥ 1 (the old live orders that never expired: "
                                 "old/CLAUDE-old-phase.md §33e)")
        elif self.entry_price is not None or self.expiry_bars is not None or self.cancel_at is not None \
                or self.first_visit_from is not None or self.first_visit_to is not None:
            raise ValueError("a next_open signal takes no entry_price, no expiry_bars, no cancel_at and no first "
                             "visit check")
        if (self.first_visit_from is None) != (self.first_visit_to is None):
            raise ValueError("first_visit_from and first_visit_to come together")
        if self.first_visit_from is not None:
            k, a, b = (pd.Timestamp(x) for x in (self.known_at, self.first_visit_from, self.first_visit_to))
            if not (a < k and a < b <= k + pd.Timedelta(minutes=5)):
                raise ValueError("first_visit_from must be before known_at and first_visit_to, and first_visit_to at "
                                 "most one 5-minute candle after known_at")

        up = self.direction == "buy"
        if self.cancel_at is not None and ((self.cancel_at <= self.entry_price) if up
                                           else (self.cancel_at >= self.entry_price)):
            raise ValueError(f"a {self.direction} limit's cancel_at must be {'above' if up else 'below'} its entry "
                             f"price {self.entry_price}: it is the target side price reaches without coming back")
        if self.levels == "price":
            ordered = tuple(sorted(self.targets, reverse=not up))           # nearest first
            if self.stop is not None and any((t <= self.stop) if up else (t >= self.stop) for t in ordered):
                raise ValueError(f"a {self.direction}'s targets must be {'above' if up else 'below'} its stop")
            if self.entry_price is not None:
                e = self.entry_price
                if self.stop is not None and ((self.stop >= e) if up else (self.stop <= e)):
                    raise ValueError(f"a {self.direction} limit's stop must be {'below' if up else 'above'} "
                                     f"its entry price {e}")
                if any((t <= e) if up else (t >= e) for t in ordered):
                    raise ValueError(f"a {self.direction} limit's targets must be {'above' if up else 'below'} "
                                     f"its entry price {e}")
        else:
            ordered = tuple(sorted(self.targets))                           # dollars away: nearest first
        object.__setattr__(self, "targets", ordered)
        return self


class Exit(BaseModel):
    """One close (a whole position, or its share when there are several targets)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: dt.datetime              # the 1-min bar it happened in (its open time, server clock)
    px: float                    # bid scale: the bid when the order fired
    lots: float
    reason: ExitReason           # "past at fill": the fill price was already past the stop or target
    spread_pts: float            # the data's spread in that minute (points) — only for the "real spread" extra line


class Trade(BaseModel):
    """One filled signal. Money only when status is "closed": a trade open at the window end is never counted."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str
    idea_id: str
    version_id: str
    parent_idea: str | None
    direction: Direction
    entry_type: EntryType
    known_at: dt.datetime
    fill_ts: dt.datetime         # the 1-min bar of the fill (its open time)
    fill_px: float               # bid scale
    fill_at_open: bool           # True: filled at that minute's open (next_open, or a bar that opened past the limit)
    fill_spread_pts: float
    stop_px: float | None        # the stop as a price (from_fill levels turned into prices)
    target_pxs: tuple[float, ...]
    lots: float
    status: Literal["closed", "open at end"]
    exits: tuple[Exit, ...]      # an open-at-end trade may already have closed a share at a target
    ambiguous: bool = False      # the order inside one minute could not be known; the worse case was taken
    ambiguous_why: str | None = None
    gross_usd: float | None = None
    gross_inr: float | None = None
    cost_inr: float | None = None
    net_inr: float | None = None
    usd_inr: float | None = None  # the rupee rate of the (last) close month

    @property
    def exit_ts(self) -> dt.datetime | None:
        return self.exits[-1].ts if self.status == "closed" else None

    @property
    def exit_px(self) -> float | None:
        """The average close price, weighted by lots."""
        if self.status != "closed":
            return None
        return round(sum(e.px * e.lots for e in self.exits) / sum(e.lots for e in self.exits), 6)

    @property
    def exit_reason(self) -> str:
        if self.status != "closed":
            return "open at end"
        return "+".join(dict.fromkeys(e.reason for e in self.exits))

    def positions(self) -> list[costs.PnL]:
        """The money of each close, from machine.costs (the one money function). Refused for an open trade."""
        if self.status != "closed":
            raise ValueError(f"trade {self.signal_id} is still open at the window end; it has no money")
        return [costs.pnl(self.direction, self.fill_px, e.px, self.fill_ts, e.ts, e.lots,
                          entry_spread_pts=self.fill_spread_pts, exit_spread_pts=e.spread_pts)
                for e in self.exits]
