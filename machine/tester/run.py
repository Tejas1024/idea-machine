"""The tester (L3): replay an idea's signals over the past, then a trades table, a summary, and ONE trial file.

    from machine.tester import Signal, run
    result = run(signals, tf="5m", start="2016", end="2020")
    print(result.text())          # the summary in easy English
    result.table()                # every trade, one row each

Only the window's bars are loaded — nothing after `end` exists for the run. The sealed years are refused; the check
years (2021-2023) are refused until the idea's check-opened.json exists (machine/guards.py).
The rules of a fill and an exit: machine/tester/engine.py.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from machine import config, costs, guards, trials
from machine.data import bars
from machine.tester.engine import CHECK_BEFORE_FILL, FIRST_VISIT_SPENT, Replay, replay
from machine.tester.models import Signal, Trade

CODE_FILES = ("engine.py", "models.py", "run.py")


@dataclass
class Result:
    replay: Replay
    settings: dict
    data: dict
    summary: dict
    trial_path: Path | None

    @property
    def trades(self) -> list[Trade]:
        return self.replay.trades

    def table(self) -> pd.DataFrame:
        return trades_table(self.replay.trades)

    def text(self) -> str:
        return summary_text(self.summary, self.settings)


def run(signals: list[Signal], *, tf: str, start=None, end=None, m1: pd.DataFrame | None = None,
        trades_at_once: int | None = None, record: bool = True, note: str | None = None) -> Result:
    """Replay the signals and write one trial file (record=False only for a run that is part of a bigger trial).

    Real prices: give start / end (like load_bars: "2016", "2020-06", "2019-03-11 10:30"; both ends included).
    Hand-made prices (tests): give m1 — 1-min bars with open high low close spread.
    """
    signals = list(signals)
    if not signals:
        raise ValueError("no signals to test")
    who = {(s.idea_id, s.version_id, s.parent_idea) for s in signals}
    if len(who) != 1:
        raise ValueError(f"one run tests ONE version of ONE idea; these signals come from {sorted(map(str, who))}")
    idea_id, version_id, parent_idea = who.pop()
    if tf not in bars.TIMEFRAMES:
        raise ValueError(f"timeframe {tf!r} unknown; use one of {', '.join(bars.TIMEFRAMES)}.")

    if m1 is None:
        m1 = bars.load_bars("1m", start, end)             # refuses the sealed years; nothing after `end`
        data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
                "gold_m1_content_sha256": bars.m1_fingerprint()}
    else:
        guards.check_dates(m1)
        data = {"prices": "bars given to run() (hand-made or cut)", "frame_sha256": _frame_hash(m1)}
    if m1.empty:
        raise ValueError(f"no bars between {start} and {end}")
    # the check years stay hidden until opened, and then only for the version they were opened for (L5)
    guards.check_opened(idea_id, [m1.index[0], m1.index[-1]], version=version_id)

    k = config.TRADES_AT_ONCE if trades_at_once is None else trades_at_once
    rep = replay(signals, m1, tf, k)
    settings = _settings(rep, signals, start, end)
    summary = summarize(rep)
    path = None
    if record:
        path = trials.record(kind="tester", idea_id=idea_id, version_id=version_id, parent_idea=parent_idea,
                             settings=settings, data=data, result=summary, note=note)
    return Result(rep, settings, data, summary, path)


# ================================================================================================================
# the summary
# ================================================================================================================

def summarize(rep: Replay) -> dict:
    closed = [t for t in rep.trades if t.status == "closed"]
    open_end = [t for t in rep.trades if t.status == "open at end"]
    n = len(closed)
    net = sum(t.net_inr for t in closed)
    years = range(rep.first_bar.year, rep.last_bar.year + 1)
    per_year = {y: {"trades": 0, "net_inr": 0.0} for y in years}
    for t in closed:                                       # a trade's year = the year it CLOSED (when money is made)
        y = per_year[pd.Timestamp(t.exit_ts).year]
        y["trades"] += 1
        y["net_inr"] += t.net_inr
    worst = min(per_year, key=lambda y: per_year[y]["net_inr"])
    reasons = Counter(t.exit_reason for t in closed)
    missed = Counter(why for _, why in rep.not_filled)
    extra = costs.extra_lines([p for t in closed for p in t.positions()]) if closed else None
    return {
        "signals": rep.signals,
        "trades": n,
        "won": sum(t.net_inr > 0 for t in closed),
        "win_pct": round(100 * sum(t.net_inr > 0 for t in closed) / n, 1) if n else None,
        "net_inr_total": round(net, 2),
        "net_inr_per_trade": round(net / n, 2) if n else None,
        "gross_usd_per_trade": round(sum(t.gross_usd for t in closed) / n, 4) if n else None,
        "cost_inr_total": round(sum(t.cost_inr for t in closed), 2),
        "per_year": {str(y): {"trades": v["trades"], "net_inr": round(v["net_inr"], 2)} for y, v in per_year.items()},
        "worst_year": {"year": worst, "net_inr": round(per_year[worst]["net_inr"], 2),
                       "trades": per_year[worst]["trades"]},
        "setups_skipped": len(rep.skipped),
        "not_filled": {"expired": missed.get("expired", 0), "cancelled": missed.get("cancelled", 0),
                       "target reached first": missed.get("target reached first", 0),
                       "window ended": missed.get("window ended", 0),
                       **({CHECK_BEFORE_FILL: missed[CHECK_BEFORE_FILL]} if missed.get(CHECK_BEFORE_FILL) else {}),
                       **({FIRST_VISIT_SPENT: missed[FIRST_VISIT_SPENT]} if missed.get(FIRST_VISIT_SPENT) else {})},
        "closed_by": dict(reasons),
        "closed_at_once": sum("past at fill" in t.exit_reason for t in closed),
        "ambiguous": sum(t.ambiguous for t in closed),
        "open_at_end": len(open_end),
        "open_at_end_ids": [t.signal_id for t in open_end],
        "extra_lines": None if extra is None else {
            "with_swap_inr": round(extra.with_swap_inr, 2), "nights_held": extra.nights_held,
            "nights_charged": extra.nights_charged, "with_spread_inr": round(extra.with_spread_inr, 2),
            "swap_reading": extra.swap_reading, "text": extra.text},
    }


def summary_text(s: dict, settings: dict) -> str:
    """The summary in easy English, with its settings named."""
    inr = costs.inr
    w = settings["window"]
    lots = " / ".join(f"{x:.2f}" for x in settings["lots"])
    lines = [
        f"TEST  {settings['idea_id']} · version {settings['version_id']} · {settings['timeframe']} bars · "
        f"{w['first_bar'][:10]} to {w['last_bar'][:10]} (server time) · {lots} lot · "
        f"{'one trade at a time' if settings['trades_at_once'] == 1 else str(settings['trades_at_once']) + ' trades at once (a TEST version)'}",
        f"  cost: ₹40 per position per 0.01 lot, all-in (LEARNED #54, #115) · prices: bid · clock: MT5 server time",
    ]
    if s["trades"]:
        lines += [
            f"Trades: {s['trades']} · won {s['won']} ({s['win_pct']:.0f}%) · after cost {inr(s['net_inr_total'])} · "
            f"{inr(s['net_inr_per_trade'])} a trade (gold {costs.usd(s['gross_usd_per_trade'])} a trade before cost)",
            f"  {s['extra_lines']['text']}",
        ]
    else:
        lines.append("Trades: 0 — nothing to count")
    years = " · ".join(f"{y} {inr(v['net_inr'])} ({v['trades']})" for y, v in s["per_year"].items())
    wy = s["worst_year"]
    closed_by = " · ".join(f"{k} {v}" for k, v in s["closed_by"].items()) or "—"
    lines += [
        f"Year by year (trades in brackets; a trade counts in the year it closed): {years}",
        f"Worst year: {wy['year']} {inr(wy['net_inr'])} ({wy['trades']} trades)",
        f"Closed by: {closed_by}",
        f"setups skipped: {s['setups_skipped']} (a trade was already on)",
        f"Limit orders not filled: {s['not_filled']['expired']} expired · {s['not_filled']['cancelled']} cancelled "
        f"by a newer setup · {s['not_filled'].get('target reached first', 0)} cancelled because price reached the "
        f"target first · {s['not_filled']['window ended']} still waiting when the test ended",
        f"Closed at once (the fill was already past its stop or target): {s['closed_at_once']}",
        f"Stop and target in the same minute (counted as the stop), or fill and target in the same minute "
        f"(target not counted): {s['ambiguous']}",
        f"OPEN AT END (still open when the test ended — not counted in ₹): {s['open_at_end']}",
    ]
    return "\n".join(lines)


def trades_table(trades: list[Trade]) -> pd.DataFrame:
    """Every trade, one row each (open-at-end trades too, with no money)."""
    rows = [{
        "signal": t.signal_id, "direction": t.direction, "entry": t.entry_type, "known_at": t.known_at,
        "fill_time": t.fill_ts, "fill_px": t.fill_px, "stop": t.stop_px,
        "targets": " / ".join(f"{x:.2f}" for x in t.target_pxs), "lots": t.lots, "status": t.status,
        "exit_time": t.exit_ts, "exit_px": t.exit_px, "closed_by": t.exit_reason,
        "gold_usd": t.gross_usd, "inr_before_cost": t.gross_inr, "cost_inr": t.cost_inr, "net_inr": t.net_inr,
        "usd_inr": t.usd_inr, "ambiguous": t.ambiguous_why or "",
    } for t in trades]
    return pd.DataFrame(rows, columns=[
        "signal", "direction", "entry", "known_at", "fill_time", "fill_px", "stop", "targets", "lots", "status",
        "exit_time", "exit_px", "closed_by", "gold_usd", "inr_before_cost", "cost_inr", "net_inr", "usd_inr",
        "ambiguous"])


# ================================================================================================================
# settings and fingerprints
# ================================================================================================================

def _settings(rep: Replay, signals: list[Signal], start, end) -> dict:
    s0 = signals[0]
    return {
        "idea_id": s0.idea_id,
        "version_id": s0.version_id,
        "parent_idea": s0.parent_idea,
        "timeframe": rep.tf,
        "window": {"asked": [None if start is None else str(start), None if end is None else str(end)],
                   "first_bar": str(rep.first_bar), "last_bar": str(rep.last_bar)},
        "trades_at_once": rep.trades_at_once,
        "trades_at_once_source": "Tejas, LEARNED #55: one trade at a time (fixed, T12 M18); more = a test version",
        "waiting_orders": "a limit order still waiting is cancelled by a newer setup, which is taken (Tejas, inbox "
                          "2026-09-28-0203: \"Cancel and take new\"; M39); a filled trade is never cancelled",
        "lots": sorted({s.lots for s in signals}),
        "entry_types": sorted({s.entry_type for s in signals}),
        "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
        "prices": "chart = bid; an order that buys fires when the ask (bid + that minute's spread) reaches its price; "
                  "every price recorded at the bid",
        "min_spread_points": config.MIN_SPREAD_POINTS,
        "same_minute": "stop and target in one minute -> the stop (ambiguous); in the fill minute of a limit the "
                       "target is not counted (ambiguous)",
        "window_end": "a trade still open at the window end is reported apart, never counted in Rs",
        "usd_inr": "monthly average of the month a position closed (data/usd_inr.csv)",
        "clock": "MT5 server time",
        "signals_sha256": trials.sha256_of([s.model_dump(mode="json") for s in signals]),
        "code_sha256": _code_hash(),
    }


def _code_hash() -> str:
    here = Path(__file__).parent
    return trials.code_sha256([here / name for name in CODE_FILES])


def _frame_hash(m1: pd.DataFrame) -> str:
    cols = [c for c in ("open", "high", "low", "close", "spread") if c in m1.columns]
    return hashlib.sha256(pd.util.hash_pandas_object(m1[cols], index=True).to_numpy().tobytes()).hexdigest()
