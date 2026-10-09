"""The tester's engine (L3): replays signals over 1-minute bars, honestly. No files here — run.py loads and records.

How a signal becomes a trade (every rule has a test in tests/test_tester.py):
  - FILL only from the next bar: the first 1-min bar that opens at or after known_at (lesson B2).
    "next_open" fills at that bar's open. A "limit" waits up to expiry_bars bars of the test's timeframe.
  - Prices on his chart are BID prices. An order that BUYS fires when the ASK reaches its price; the ask = the bid +
    that minute's spread (config.MIN_SPREAD_POINTS when the data says 0). An order that SELLS fires on the bid.
      buy limit  (buys):  bid low + spread <= limit      sell limit (sells): bid high >= limit
      a buy's stop and targets close it by SELLING: the bid.
      a sell's stop and targets close it by BUYING: the ask (Claude's reading: MT5 closes a sell at the ask, the same
      rule as the buy limit — without it a sell's target would be credited when only the bid touched it).
    Every price is RECORDED on the bid scale: the bid when the order fired (a buy limit at 2000 with a $0.20 spread is
    recorded at 1999.80). The money cost is Rs 40 only — never the spread on top (lesson B14).
  - A bar that opens past the order's price fills at that bar's open (a gap).
  - A limit with cancel_at (L7a) is cancelled, not filled, when the bid reaches cancel_at first — a buy's high up to
    it, a sell's low down to it: "if price already reached the target before it came back, there is no trade"
    (#119). Both in the same minute: cancelled (the order inside the minute is unknown).
  - A check before the fill (L19b, his entry types): replay(..., cancel_by={signal id: moment}) cancels a limit still
    waiting at that moment - e.g. an opposite setup of his became known (#98 B), or the liquidity that excused an
    unfilled gap was all taken (#87). The moment comes from what was known then; the Signal itself carries nothing
    from after its known_at, so the cut-off-the-future test on signals still holds.
  - EXITS are checked minute by minute from the fill on (so "both in one 5-min bar" is always settled by its 1-min
    bars). Stop and target in ONE minute -> the stop, counted "ambiguous".
    In the fill minute of a limit filled after the open: the stop counts (price had to pass the fill to reach it);
    a target does not (it may have come before the fill) — counted "ambiguous" when a target was touched.
  - A fill already past its stop or target (a gap, or a stop inside the spread) closes at once at the fill price.
  - trades_at_once (default 1, LEARNED #55): while that many are on, a new signal is SKIPPED if they are all trades.
    If one is a limit order still WAITING (not filled), it is CANCELLED and the new signal is taken — Tejas, 28 Sep
    2026: "Cancel and take new" (strategy/learned-inbox/2026-09-28-0203-session.md; management table M39). Only a
    NEWER signal cancels (not one known at the same moment); the oldest waiting order goes first.
  - A trade still open at the window end is "open at end": reported apart, never counted in rupees. Nothing after the
    window's last bar exists here.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from machine import config
from machine.data.bars import bar_open_times
from machine.tester.models import Exit, Signal, Trade

EPS = 1e-6                    # price comparisons (a millionth of a dollar; prices sit on a 1-cent grid)
MINUTE_NS = 60 * 10**9
FIVE_NS = 5 * MINUTE_NS
FIRST_VISIT_SPENT = "spent: came back into the setup's gaps and turned away (#68)"
NEVER = np.iinfo(np.int64).max
REQUIRED = ("open", "high", "low", "close", "spread")
CHECK_BEFORE_FILL = "a check before the fill"   # why a limit was cancelled by replay(cancel_by=...)


@dataclass
class Replay:
    """What happened to every signal."""
    trades: list[Trade]                                   # filled: "closed" and "open at end"
    skipped: list[str]                                    # signal ids: trades_at_once trades already on
    not_filled: list[tuple[str, str]]                     # (signal id, "expired" | "cancelled" | "target reached
                                                          #  first" | "window ended" | CHECK_BEFORE_FILL)
    first_bar: pd.Timestamp                               # the window: first and last 1-min bar (server time)
    last_bar: pd.Timestamp
    tf: str
    trades_at_once: int
    signals: int = 0
    notes: list[str] = field(default_factory=list)


def replay(signals: list[Signal], m1: pd.DataFrame | None, tf: str, trades_at_once: int | None = None, *,
           book: "_Bars | None" = None, cancel_by: Mapping[str, int] | None = None) -> Replay:
    """Replay the signals over m1 (1-min bars: server-time index; open high low close in $ bid; spread in points).

    tf is the timeframe expiry_bars counts in. Signals are taken in known_at order (ties: the order given).
    A signal known before the first bar, or after the last bar has closed, is refused.
    book: the same bars already read by bars_book(m1, tf) — the referee replays one window hundreds of times (its
    random entries, L5) and reads the bars once; then m1 is not needed.
    cancel_by: {signal id: moment in ns} - a check before the fill (the module text): the limit is cancelled if it has
    not filled in a minute that opens before that moment.
    """
    cancel_by = cancel_by or {}
    k = config.TRADES_AT_ONCE if trades_at_once is None else trades_at_once
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"trades_at_once={k!r}: must be 1 or more (LEARNED #55: one trade at a time)")
    if book is None:
        book = _Bars(m1, tf)
    elif book.tf != tf:
        raise ValueError(f"the bars were read for {book.tf} bars, not {tf}")
    ids = [s.id for s in signals]
    if len(set(ids)) != len(ids):
        raise ValueError("two signals share one id")

    out = Replay([], [], [], book.first, book.last, tf, k, len(signals))
    taken: list[list] = []                                # [signal, trade or None, why not filled], in order taken
    held: list[_Held] = []                                # the signals on right now: a trade or a waiting order
    for s in sorted(signals, key=lambda s: s.known_at):   # sorted() is stable: ties keep the order given
        t = pd.Timestamp(s.known_at).as_unit("ns").value
        if t < book.t[0] or t > book.t[-1] + MINUTE_NS:
            raise ValueError(f"signal {s.id} is known at {s.known_at}, outside the window "
                             f"{book.first} .. {book.last} (+1 minute). Make signals from the window's own bars.")
        held = [h for h in held if h.release > t]
        if len(held) >= k:
            # Tejas, 28 Sep 2026 (inbox 0203): "Cancel and take new" — a limit order still waiting is cancelled by a
            # NEWER setup (known later, not at the same moment); a filled trade is never cancelled.
            waiting = [h for h in held if h.wait_until is not None and t < h.wait_until and h.known < t]
            if not waiting:
                out.skipped.append(s.id)
                continue
            oldest = min(waiting, key=lambda h: h.known)
            held.remove(oldest)
            taken[oldest.row][1:] = [None, "cancelled"]  # the fill it would have had comes after t: never happened
        release, trade, missed, wait_until = book.play(s, t, cancel_by.get(s.id))
        held.append(_Held(t, release, wait_until, len(taken)))
        taken.append([s, trade, missed])
    for s, trade, missed in taken:
        if trade is not None:
            out.trades.append(trade)
        else:
            out.not_filled.append((s.id, missed))
    return out


def bars_book(m1: pd.DataFrame, tf: str) -> "_Bars":
    """The window's bars read once, for many replays of the same window (replay(..., book=))."""
    return _Bars(m1, tf)


@dataclass(frozen=True)
class Filled:
    """An order filled: in 1-min bar i, at px (bid scale), at that minute's open or not."""
    i: int
    px: float
    at_open: bool
    wait_until: int | None      # a limit: still waiting before this time (ns); None for a next_open order


@dataclass(frozen=True)
class Missed:
    """An order that did not fill: its place is free from `release` (ns) on."""
    release: int
    why: str                    # "expired" | "target reached first" | "window ended"
    wait_until: int | None


@dataclass
class _Held:
    known: int                  # known_at (ns)
    release: int                # its place is free from this time on (ns)
    wait_until: int | None      # still waiting (not filled) before this time; None = cannot be cancelled
    row: int                    # its row in `taken`


class _Bars:
    """The window's 1-min bars as arrays, and one signal's life on them."""

    def __init__(self, m1: pd.DataFrame, tf: str):
        missing = [c for c in REQUIRED if c not in m1.columns]
        if missing:
            raise ValueError(f"m1 needs the columns {', '.join(REQUIRED)}; missing {', '.join(missing)}")
        idx = pd.DatetimeIndex(m1.index)
        if idx.tz is not None or not idx.is_monotonic_increasing or not idx.is_unique or len(idx) == 0:
            raise ValueError("m1 needs a sorted server-time index (no time zone), no repeats, at least one bar")
        if (idx != idx.floor("1min")).any():
            raise ValueError("m1 must be 1-minute bars: every time on a whole minute")
        self.first, self.last = idx[0], idx[-1]
        self.tf = tf
        self.t = idx.as_unit("ns").asi8
        self.o = m1["open"].to_numpy(dtype="float64")
        self.h = m1["high"].to_numpy(dtype="float64")
        self.lo = m1["low"].to_numpy(dtype="float64")
        raw = m1["spread"].to_numpy(dtype="float64")
        self.raw_spread = raw                                               # points, as in the data
        floor = np.where(np.isnan(raw) | (raw <= 0), config.MIN_SPREAD_POINTS, raw)
        self.sp = floor * config.POINT                                      # dollars; the ask = bid + sp
        self.n = len(idx)
        self.tf_open = bar_open_times(idx, tf).unique().as_unit("ns").asi8  # the tf bars, from the window only

    # ---- one signal -------------------------------------------------------------------------------------------

    def play(self, s: Signal, t: int, cancel_by: int | None = None) -> tuple[int, Trade | None, str, int | None]:
        """(release time ns, the trade or None, why not filled, waiting until).

        waiting until (ns, not included): a limit order is still WAITING — not filled — at any time before it, so a
        newer setup can still cancel it. None for a next_open order (it fills at once; nothing to cancel).
        cancel_by: a check before the fill cancels the limit at that moment (replay's cancel_by).
        """
        got = self.fill(s, t, cancel_by)
        if isinstance(got, Missed):
            return got.release, None, got.why, got.wait_until
        release, trade = self._after_fill(s, got.i, got.px, got.at_open)
        return release, trade, "", got.wait_until

    def fill(self, s: Signal, t: int, cancel_by: int | None = None) -> "Filled | Missed":
        """Where and at what price the signal's order fills (the rules in the module text), or why it does not.
        The hedging tester (L19a) fills its orders here too, so both testers fill the same way.
        cancel_by (ns): a limit not filled in a minute that opens before this moment is cancelled then (L19b)."""
        i0 = int(np.searchsorted(self.t, t, "left"))       # the first 1-min bar opening at or after known_at
        if i0 >= self.n:
            return Missed(NEVER, "window ended", None)
        buy = s.direction == "buy"
        if s.entry_type == "next_open":
            i, px, at_open = i0, self.o[i0], True
        else:
            j = int(np.searchsorted(self.tf_open, t, "left"))
            deadline = self.tf_open[j + s.expiry_bars] if j + s.expiry_bars < len(self.tf_open) else NEVER
            checked = cancel_by is not None and cancel_by < deadline
            end = int(cancel_by) if checked else deadline
            spent = self._spent_at(s, t, end) if s.first_visit_from is not None else None
            if spent is not None:
                end, checked = spent, False
            i1 = int(np.searchsorted(self.t, end, "left")) if end != NEVER else self.n
            L = s.entry_price
            ih, ig = self._first_fill(i0, i1, L, s.cancel_at, buy)
            if ig >= 0 and (ih < 0 or ig <= ih):
                # price reached the target side first (or in the fill's own minute): no trade (#119)
                c = int(self.t[ig])
                return Missed(c + MINUTE_NS, "target reached first", c + 1)
            if ih < 0:
                if spent is not None:
                    return Missed(end, FIRST_VISIT_SPENT, end)
                if checked:
                    return Missed(end, CHECK_BEFORE_FILL, end)
                return Missed(deadline, "window ended" if deadline == NEVER else "expired", deadline)
            i = ih
            if buy:
                at_open = self.o[i] + self.sp[i] <= L + EPS
                px = self.o[i] if at_open else L - self.sp[i]
            else:
                at_open = self.o[i] >= L - EPS
                px = self.o[i] if at_open else L
        # a limit fills during minute i: before that minute starts it is still waiting
        wait = None if s.entry_type == "next_open" else int(self.t[i]) + 1
        return Filled(i, round(float(px), 6), bool(at_open), wait)

    def _five(self):
        """The window's 5-minute candles made from its 1-minute bars (his gap chart, bid prices): their opens (ns),
        highs, lows and lowest ask - read once."""
        if getattr(self, "_m5", None) is None:
            g = bar_open_times(pd.DatetimeIndex(self.t), "5m").as_unit("ns").asi8
            st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            self._m5 = (g[st], np.maximum.reduceat(self.h, st), np.minimum.reduceat(self.lo, st),
                        np.minimum.reduceat(self.lo + self.sp, st))
        return self._m5

    def _spent_at(self, s: Signal, t: int, end: int) -> int | None:
        """His #68 (fvg.first_visit): the moment (ns) the waiting limit is spent - price came back into the setup's
        gaps and a 5-minute candle turned away without reaching the entry, known at that candle's close - if it comes
        before `end`; else None. Only candles that open at or after known_at, and a gap only from the candle after its
        3rd, are read. "Reached the entry" is the order's own fill test (a buy: the ask): the order may fill only during
        price's first visit - when the spread kept a buy limit from filling while the chart touched it, and price
        turned away, a fill on a later return would be his second visit (7 Oct 2026: one such idea #4 buy filled 3
        days later). The pictures draw the chart touch as the entry, as before (questions/pictures.entry_candle)."""
        from machine.blocks import fvg                    # here: machine.blocks imports the tester's models
        opens, h5, lo5, ask_lo5 = self._five()
        buy, L = s.direction == "buy", s.entry_price
        frm = pd.Timestamp(s.first_visit_from).as_unit("ns").value
        to = pd.Timestamp(s.first_visit_to).as_unit("ns").value
        j_from = int(np.searchsorted(opens, frm, "left")) + 2
        j_to = int(np.searchsorted(opens, to - FIVE_NS, "right")) - 1      # the last candle closing by first_visit_to
        gaps = fvg.setup_gaps(h5, lo5, j_from, j_to, s.direction)
        k0 = int(np.searchsorted(opens, t, "left"))
        k1 = len(opens) if end == NEVER else int(np.searchsorted(opens, end, "left"))

        class _Reach:                     # the fill test of _first_fill, on a 5-minute candle
            def __getitem__(self, sl):
                return (ask_lo5[sl] <= L + EPS) if buy else (h5[sl] >= L - EPS)

        what, k = fvg.first_visit(h5, lo5, _Reach(), gaps, k0, k1, s.direction)
        if what != "spent":
            return None
        moment = int(opens[k]) + FIVE_NS
        return moment if moment < end else None

    def _after_fill(self, s: Signal, i: int, px: float, at_open: bool) -> tuple[int, Trade]:
        """The stop and targets from the fill on: (release time ns, the trade)."""
        buy = s.direction == "buy"
        if s.levels == "from_fill":
            sign = 1 if buy else -1
            stop = None if s.stop is None else round(px - sign * s.stop, 6)
            targets = tuple(round(px + sign * d, 6) for d in s.targets)
        else:
            stop, targets = s.stop, s.targets
        each = round(s.lots / len(targets), 2) if targets else s.lots
        left = list(targets)
        exits: list[Exit] = []
        why: list[str] = []

        def close(m: int, price: float, lots: float, reason: str) -> None:
            exits.append(Exit(ts=self._ts(m), px=round(float(price), 6), lots=lots, reason=reason,
                              spread_pts=float(self.raw_spread[m])))

        # already past the stop or a target at the fill: close at once at the fill price
        ask = px + self.sp[i]
        past = ((stop is not None and (px <= stop + EPS if buy else ask >= stop - EPS)) or
                (bool(left) and (px >= left[0] - EPS if buy else ask <= left[0] + EPS)))
        if past:
            close(i, px, s.lots, "past at fill")
            return self._done(s, i, px, at_open, stop, targets, exits, why)

        p = i
        if not at_open:
            # the fill minute: the stop counts, a target does not (it may have come before the fill)
            if left and self._target_hit(i, i + 1, left[0], buy).any():
                why.append("the fill and a target in the same minute: the target was not counted")
            if stop is not None and self._stop_hit(i, i + 1, stop, buy).any():
                close(i, self._stop_px(i, stop, buy), s.lots, "stop")
                return self._done(s, i, px, at_open, stop, targets, exits, why)
            p = i + 1

        while True:
            m = self._first_event(p, stop, left[0] if left else None, buy)
            if m >= self.n:
                return self._done(s, i, px, at_open, stop, targets, exits, why, open_at_end=True)
            hit_targets = [T for T in left if self._target_hit(m, m + 1, T, buy).any()]
            if stop is not None and self._stop_hit(m, m + 1, stop, buy).any():
                if hit_targets:
                    why.append("the stop and a target in the same minute: counted as the stop")
                close(m, self._stop_px(m, stop, buy), round(s.lots - sum(e.lots for e in exits), 2), "stop")
                return self._done(s, i, px, at_open, stop, targets, exits, why)
            for T in hit_targets:
                close(m, self._target_px(m, T, buy), each, "target")
                left.remove(T)
            if not left:
                return self._done(s, i, px, at_open, stop, targets, exits, why)
            p = m + 1

    # ---- the checks, on bars [a, b) ---------------------------------------------------------------------------

    def _stop_hit(self, a: int, b: int, stop: float, buy: bool) -> np.ndarray:
        # a buy's stop SELLS (the bid); a sell's stop BUYS (the ask = bid + spread)
        return self.lo[a:b] <= stop + EPS if buy else self.h[a:b] + self.sp[a:b] >= stop - EPS

    def _target_hit(self, a: int, b: int, target: float, buy: bool) -> np.ndarray:
        return self.h[a:b] >= target - EPS if buy else self.lo[a:b] + self.sp[a:b] <= target + EPS

    def _stop_px(self, m: int, stop: float, buy: bool) -> float:
        if buy:
            return self.o[m] if self.o[m] <= stop + EPS else stop
        return self.o[m] if self.o[m] + self.sp[m] >= stop - EPS else stop - self.sp[m]

    def _target_px(self, m: int, target: float, buy: bool) -> float:
        if buy:
            return self.o[m] if self.o[m] >= target - EPS else target
        return self.o[m] if self.o[m] + self.sp[m] <= target + EPS else target - self.sp[m]

    def _first_fill(self, a: int, b: int, limit: float, cancel_at: float | None, buy: bool) -> tuple[int, int]:
        """(the first bar in [a, b) where the limit fills, the first where cancel_at is reached), -1 = none. Searched
        in growing chunks and stopped at the first chunk that has either, so an order that fills soon never scans the
        whole window (the same answer as scanning it all: a later chunk can only hold later bars)."""
        size = 256
        while a < b:
            q = min(b, a + size)
            hit = (self.lo[a:q] + self.sp[a:q] <= limit + EPS) if buy else (self.h[a:q] >= limit - EPS)
            ih = a + int(np.argmax(hit)) if hit.any() else -1
            ig = -1
            if cancel_at is not None:
                gone = (self.h[a:q] >= cancel_at - EPS) if buy else (self.lo[a:q] <= cancel_at + EPS)
                ig = a + int(np.argmax(gone)) if gone.any() else -1
            if ih >= 0 or ig >= 0:
                return ih, ig
            a, size = q, min(size * 4, 1 << 20)
        return -1, -1

    def _first_event(self, p: int, stop: float | None, target: float | None, buy: bool, end: int | None = None) -> int:
        """The first bar from p on (before `end`, default the window's end) where the stop or the nearest target is
        reached (self.n if none). Searched in growing chunks, so a short trade never scans the whole window."""
        size = 256
        stop_at = self.n if end is None else min(self.n, end)
        while p < stop_at:
            q = min(stop_at, p + size)
            hit = np.zeros(q - p, dtype=bool)
            if stop is not None:
                hit |= self._stop_hit(p, q, stop, buy)
            if target is not None:
                hit |= self._target_hit(p, q, target, buy)
            if hit.any():
                return p + int(np.argmax(hit))
            p, size = q, min(size * 4, 1 << 20)
        return self.n

    # ---- the result -------------------------------------------------------------------------------------------

    def _ts(self, m: int) -> pd.Timestamp:
        return pd.Timestamp(self.t[m])

    def _done(self, s: Signal, i: int, px: float, at_open: bool, stop, targets, exits: list[Exit], why: list[str],
              open_at_end: bool = False) -> tuple[int, Trade]:
        fields = dict(
            signal_id=s.id, idea_id=s.idea_id, version_id=s.version_id, parent_idea=s.parent_idea,
            direction=s.direction, entry_type=s.entry_type, known_at=s.known_at,
            fill_ts=self._ts(i), fill_px=px, fill_at_open=bool(at_open), fill_spread_pts=float(self.raw_spread[i]),
            stop_px=stop, target_pxs=tuple(targets), lots=s.lots, exits=tuple(exits),
            ambiguous=bool(why), ambiguous_why="; ".join(why) or None,
        )
        if open_at_end:
            return NEVER, Trade(status="open at end", **fields)
        trade = Trade(status="closed", **fields)
        legs = trade.positions()
        trade = trade.model_copy(update=dict(
            gross_usd=sum(p.gross_usd for p in legs), gross_inr=sum(p.gross_inr for p in legs),
            cost_inr=sum(p.cost_inr for p in legs), net_inr=sum(p.net_inr for p in legs), usd_inr=legs[-1].usd_inr))
        # the slot is free only once the minute of the last close is over
        return int(pd.Timestamp(exits[-1].ts).as_unit("ns").value) + MINUTE_NS, trade
