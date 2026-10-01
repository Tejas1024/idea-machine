"""Fake levels (L5, step 5): the idea's level moved to its neighbours.

If the level itself matters, the real one must earn more than the same trades placed at prices a little above and
below it. Each fake copy moves every limit order's price — and its stop and targets with it, so the distances stay the
same — by config.FAKE_LEVEL_STEPS times that trade's own stop distance (its first target distance when it has no
stop): -2, -1.5, -1, -0.5, +0.5, +1, +1.5, +2. Claude's reading of "the level moved to its neighbours".
The times stay the same; the same tester and cost. A limit's cancel price (#119: price reached the target before it
came back) moves with its target (L10).

An idea with no level (every entry at the next open, like "buy every Monday at 10:00") has nothing to move: the check
"does not apply" and says so.
"""
from __future__ import annotations

from dataclasses import dataclass

from machine import config
from machine.referee.replays import Segment, closed_trades, pooled, replay
from machine.tester.models import Signal


@dataclass
class FakeRun:
    step: float
    trades: int
    net_inr: float
    per_trade: float | None
    target_first: int = 0          # closed trades that reached their target before the stop (the report, L10)


def moved(s: Signal, step: float) -> Signal:
    """The same signal with its limit price (and its stop and targets, when they are prices) moved by step x its stop
    distance. A next-open signal comes back unchanged."""
    if s.entry_type != "limit":
        return s
    e = s.entry_price
    if s.levels == "from_fill":
        dist = s.stop if s.stop is not None else s.targets[0]
    else:
        dist = abs(e - s.stop) if s.stop is not None else abs(s.targets[0] - e)
    shift = round(step * dist, 2)
    data = s.model_dump()
    data["id"] = f"{s.id}~{step:+g}"
    data["entry_price"] = round(e + shift, 2)
    if s.levels == "price":
        data["stop"] = None if s.stop is None else round(s.stop + shift, 2)
        data["targets"] = tuple(round(t + shift, 2) for t in s.targets)
    if s.cancel_at is not None:
        # "price reached the target before it came back: no trade" (#119) - the target moved, so its cancel price
        # moves with it (L10: before this, a moved buy limit could sit above its own cancel price and be refused)
        data["cancel_at"] = round(s.cancel_at + shift, 2)
    return Signal(**data)


def has_level(segments: list[Segment]) -> bool:
    return any(s.entry_type == "limit" for seg in segments for s in seg.signals)


def runs(segments: list[Segment], steps=config.FAKE_LEVEL_STEPS) -> list[FakeRun] | None:
    """One run per step over all segments; None when the idea has no level to move."""
    if not has_level(segments):
        return None
    out = []
    for step in steps:
        tr = pooled([closed_trades(replay(seg, [moved(s, step) for s in seg.signals])) for seg in segments])
        n = len(tr)
        total = float(tr["net_inr"].sum()) if n else 0.0
        out.append(FakeRun(float(step), n, round(total, 2), round(total / n, 2) if n else None,
                           int((tr["closed_by"] == "target").sum()) if n else 0))
    return out


def summary(fr: list[FakeRun] | None, idea_per_trade: float | None) -> dict:
    if fr is None:
        return {"applies": False, "why": "no level to move: every entry is at the next open"}
    rated = [f for f in fr if f.per_trade is not None]
    beaten = sum(idea_per_trade is not None and f.per_trade < idea_per_trade for f in rated)
    trades = sum(f.trades for f in fr)
    return {"applies": True, "steps": [f.step for f in fr], "beaten": int(beaten), "compared": len(rated),
            "beaten_share": round(beaten / len(rated), 4) if rated else 0.0,
            "trades": int(trades), "target_first": int(sum(f.target_first for f in fr)),
            "runs": [{"step": f.step, "trades": f.trades, "per_trade": f.per_trade, "net_inr": f.net_inr,
                      "target_first": f.target_first} for f in fr]}
