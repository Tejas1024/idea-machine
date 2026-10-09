"""One stretch of judged years, read once and replayed many times (the idea, its random entries, its fake levels).

Every replay goes through the tester's own engine (machine/tester/engine.py): the same fills, exits, one trade at a
time and Rs 40 cost as the idea itself. Nothing here writes a trial file — the referee writes ONE for the whole
judgement (machine.trials.record, kind "referee").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from machine import config
from machine.tester import engine
from machine.tester.models import TF_MINUTES, Signal
from machine.tester.run import trades_table

MINUTE = pd.Timedelta(minutes=1)


@dataclass
class Segment:
    label: str                          # "2021-2023", or one walk-forward judge year "2019"
    years: tuple[int, ...]
    tf: str
    signals: list[Signal]
    m1: pd.DataFrame
    book: object                        # engine.bars_book(m1, tf)
    close: np.ndarray                   # 1-min closes, for "the price when the signal was known"

    def price_before(self, known_at) -> float:
        """The close of the last 1-min bar that opened before known_at (the price when the signal is known)."""
        i = int(np.searchsorted(self.book.t, pd.Timestamp(known_at).as_unit("ns").value, "left")) - 1
        return float(self.close[max(i, 0)])

    def known_times(self) -> pd.DatetimeIndex:
        """Every moment a signal of this timeframe can be known in the window: each bar's close."""
        opens = pd.DatetimeIndex(self.book.tf_open)
        known = opens + pd.Timedelta(minutes=TF_MINUTES[self.tf])
        first, last = pd.Timestamp(self.book.first), pd.Timestamp(self.book.last) + MINUTE
        return known[(known >= first) & (known <= last)]


def make_segment(label: str, years, tf: str, signals: list[Signal], m1: pd.DataFrame, *, strict: bool) -> Segment:
    """strict: every signal must be inside the window (the check years); else only the window's signals are kept
    (a walk-forward year cut out of 2016-2020)."""
    if m1.empty:
        raise ValueError(f"no bars for {label}")
    first, last = m1.index[0], m1.index[-1] + MINUTE
    inside = [s for s in signals if first <= pd.Timestamp(s.known_at) <= last]
    if strict and len(inside) != len(signals):
        out = next(s for s in signals if s not in inside)
        raise ValueError(f"signal {out.id} is known at {out.known_at}, outside {label} ({first} .. {last})")
    return Segment(label, tuple(sorted(years)), tf, inside, m1, engine.bars_book(m1, tf),
                   m1["close"].to_numpy(dtype=float))


def replay(seg: Segment, signals: list[Signal]) -> engine.Replay:
    return engine.replay(signals, None, seg.tf, config.TRADES_AT_ONCE, book=seg.book)


def closed_trades(rep: engine.Replay) -> pd.DataFrame:
    """The closed trades: net_inr, exit_time, known_at, closed_by (open-at-end trades have no money — L3)."""
    t = trades_table(rep.trades)
    t = t[t["status"] == "closed"]
    return pd.DataFrame({"net_inr": t["net_inr"].astype(float).to_numpy(),
                         "exit_time": pd.to_datetime(t["exit_time"]).to_numpy(),
                         "known_at": pd.to_datetime(t["known_at"]).to_numpy(),
                         "signal": t["signal"].to_numpy(),
                         "closed_by": t["closed_by"].astype(str).to_numpy()})


def pooled(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame({"net_inr": pd.Series(dtype=float), "exit_time": pd.Series(dtype="datetime64[ns]"),
                             "known_at": pd.Series(dtype="datetime64[ns]"), "signal": pd.Series(dtype=object),
                             "closed_by": pd.Series(dtype=object)})
    return pd.concat(frames, ignore_index=True)


# ---- what happened to every setup (the report's "came to the level", "hit the target first" - L10) ---------------

# A setup "had its full chance" when nothing but the market decided whether price came back to its level: it was not
# skipped (a trade was already on), not cancelled by a newer setup, and the test did not end while it waited.
FULL_CHANCE_MISSED = ("expired", "target reached first", engine.FIRST_VISIT_SPENT)   # spent: the market decided (#68)


def path_counts(reps: list[engine.Replay]) -> dict:
    """What happened to the setups of these replays: filled, never came back, skipped, cancelled, still waiting.

    came_back / had_chance = "came to the level after the setup": a limit that filled, out of the limits that had
    their full chance (filled + waited out its time + cancelled because price reached the target first). Skipped
    setups, setups cancelled by a newer one and orders still waiting at the end are counted apart - nobody knows
    whether price would have come back to them. A next-open signal has no level to come back to: not counted here.
    """
    filled = [t for r in reps for t in r.trades]
    missed = [why for r in reps for _, why in r.not_filled]
    limits_filled = sum(t.entry_type == "limit" for t in filled)
    never = {w: missed.count(w) for w in FULL_CHANCE_MISSED}
    return {
        "signals": sum(r.signals for r in reps),
        "filled": len(filled),
        "limits_filled": limits_filled,
        "next_open_filled": len(filled) - limits_filled,
        "had_chance": limits_filled + sum(never.values()),
        "came_back": limits_filled,
        "never_came_back": never,
        "skipped_trade_on": sum(len(r.skipped) for r in reps),
        "cancelled_by_newer": missed.count("cancelled"),
        "still_waiting_at_end": missed.count("window ended"),
    }


def outcome_counts(reps: list[engine.Replay]) -> dict:
    """How the filled trades ended. Money only for closed trades; a trade still open at the end is never a profit
    (lesson B6: profit was once counted on still-open hedges)."""
    trades = [t for r in reps for t in r.trades]
    closed = [t for t in trades if t.status == "closed"]
    return {
        "closed": len(closed),
        "in_profit": sum(t.net_inr > 0 for t in closed),
        "at_a_loss": sum(t.net_inr <= 0 for t in closed),
        "open_at_end": len(trades) - len(closed),
        "target_first": sum(t.exit_reason == "target" for t in closed),
        "stop_first": sum(t.exit_reason.startswith("stop") for t in closed),
        "at_once": sum("past at fill" in t.exit_reason for t in closed),
        "target_then_stop": sum(t.exit_reason == "target+stop" for t in closed),
        "ambiguous": sum(t.ambiguous for t in closed),
    }


TRADE_COLUMNS = ("signal", "direction", "known_at", "fill_time", "fill_px", "stop", "targets", "status", "exit_time",
                 "exit_px", "closed_by", "gold_usd", "cost_inr", "net_inr", "usd_inr", "ambiguous")


def trade_rows(reps: list[engine.Replay]) -> dict:
    """Every trade (open-at-end ones too, with no money), as {"columns": [...], "rows": [[...], ...]} - compact, so
    the trial file can hold every trade (his #107 / #114: show every trade, not averages)."""
    rows = []
    for r in reps:
        t = trades_table(r.trades)
        for rec in t.to_dict("records"):
            row = []
            for c in TRADE_COLUMNS:
                v = rec[c]
                if isinstance(v, pd.Timestamp) or hasattr(v, "isoformat"):
                    v = None if pd.isna(v) else pd.Timestamp(v).isoformat()
                elif isinstance(v, float):
                    v = None if np.isnan(v) else round(v, 6 if c in ("fill_px", "exit_px", "stop") else 2)
                elif v is None or (not isinstance(v, str) and pd.isna(v)):
                    v = None
                row.append(v)
            rows.append(row)
    rows.sort(key=lambda r: r[2] or "")
    return {"columns": list(TRADE_COLUMNS), "rows": rows}


def per_trade(trades: pd.DataFrame) -> float | None:
    return float(trades["net_inr"].mean()) if len(trades) else None
