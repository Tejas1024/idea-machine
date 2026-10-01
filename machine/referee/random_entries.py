"""Random entries (L5, step 4): the idea's trades with the TIMING and the LEVEL taken away.

Each random run keeps, from the idea's own trades: the direction, the entry type (next open / limit), the stop and
target distances, a limit's cancel price distance (#119: "price reached the target before it came back: no trade" -
L10), the limit's waiting time, the lots, the cost (the same tester), and the number of trades in each
month. It changes: the TIME (a random bar close in the same month) and, for a limit order, the LEVEL (the limit is
put at a distance from the price drawn from the idea's own limit distances). config.RANDOM_RUNS runs (200).

Trades per month: each month gets as many random signals as the idea had trades; a month that ends up short (a
random signal skipped because a trade was already on, or a limit that never filled) gets more random signals, up to
MAX_PASSES times. Each run reports its trade count next to the idea's.
The direction is kept (Claude's reading): the random runs then carry the same exposure to gold's trend as the idea,
so an idea that only buys in a rising market does not "beat" random entries just by buying.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from machine import config, trials
from machine.referee.replays import Segment, closed_trades, replay
from machine.tester.models import Signal

MAX_PASSES = 4


@dataclass(frozen=True)
class Template:
    """One of the idea's traded signals, as distances (no time, no level)."""
    direction: str
    entry_type: str
    offset: float | None          # limit: how far the limit sat from the price when known (+ = a better price)
    stop_d: float | None          # dollars from the entry to the stop
    target_ds: tuple[float, ...]  # dollars from the entry to each target
    cancel_d: float | None        # a limit's cancel price (#119, target reached first), dollars from the entry (L10)
    expiry_bars: int | None
    lots: float
    idea_id: str
    version_id: str
    parent_idea: str | None


@dataclass
class Run:
    trades: int
    target_trades: int
    net_inr: float
    per_trade: float | None
    nets: np.ndarray
    exit_times: np.ndarray


def template(seg: Segment, s: Signal) -> Template:
    buy = s.direction == "buy"
    ref = seg.price_before(s.known_at)
    if s.entry_type == "limit":
        entry = s.entry_price
        offset = (ref - entry) if buy else (entry - ref)
    else:
        entry, offset = ref, None
    if s.levels == "from_fill":
        stop_d, target_ds = s.stop, tuple(s.targets)
    else:
        stop_d = None if s.stop is None else abs(entry - s.stop)
        target_ds = tuple(abs(t - entry) for t in s.targets)
    cancel_d = None if s.cancel_at is None else abs(s.cancel_at - entry)
    return Template(s.direction, s.entry_type, offset, stop_d, target_ds, cancel_d, s.expiry_bars, s.lots,
                    s.idea_id, s.version_id, s.parent_idea)


def make_signal(seg: Segment, t: Template, known_at: pd.Timestamp, offset: float | None, sid: str) -> Signal:
    common = dict(id=sid, known_at=known_at, direction=t.direction, entry_type=t.entry_type, lots=t.lots,
                  idea_id=t.idea_id, version_id=t.version_id, parent_idea=t.parent_idea)
    if t.entry_type == "limit":
        buy = t.direction == "buy"
        ref = seg.price_before(known_at)
        entry = round(ref - offset if buy else ref + offset, 2)
        sign = 1 if buy else -1
        return Signal(**common, entry_price=entry, expiry_bars=t.expiry_bars, levels="price",
                      stop=None if t.stop_d is None else round(entry - sign * t.stop_d, 2),
                      targets=tuple(round(entry + sign * d, 2) for d in t.target_ds),
                      cancel_at=None if t.cancel_d is None else round(entry + sign * t.cancel_d, 2))
    return Signal(**common, levels="from_fill", stop=t.stop_d, targets=t.target_ds)


def seed_of(segments: list[Segment]) -> int:
    """The same idea gives the same random runs on every PC (recorded in the trial file)."""
    sigs = [s.model_dump(mode="json") for seg in segments for s in seg.signals]
    return int(trials.sha256_of(sigs)[:12], 16)


def plan(seg: Segment, idea_trades: pd.DataFrame) -> tuple[dict, dict]:
    """(templates by month, trades wanted by month) from the idea's own closed trades in this segment."""
    by_id = {s.id: s for s in seg.signals}
    months = pd.DatetimeIndex(idea_trades["known_at"]).to_period("M")
    temps: dict = defaultdict(list)
    for sid, m in zip(idea_trades["signal"], months):
        temps[m].append(template(seg, by_id[sid]))
    return dict(temps), dict(Counter(months))


def runs(segments: list[Segment], idea_trades: list[pd.DataFrame], n_runs: int = config.RANDOM_RUNS,
         seed: int | None = None) -> tuple[list[Run], dict]:
    """n_runs random runs over the same segments. Returns (runs, settings used)."""
    seed = seed_of(segments) if seed is None else seed
    plans = [plan(seg, tr) for seg, tr in zip(segments, idea_trades)]
    offsets = np.array([t.offset for temps, _ in plans for ts in temps.values() for t in ts if t.offset is not None])
    times = [seg.known_times() for seg in segments]
    by_month = []
    for tt in times:
        per = tt.to_period("M")
        by_month.append({m: np.flatnonzero(per == m) for m in per.unique()})
    out = []
    for r in range(n_runs):
        rng = np.random.default_rng([seed, r])
        nets, exits, wanted = [], [], 0
        for k, (seg, (temps, want)) in enumerate(zip(segments, plans)):
            sigs: list[Signal] = []
            count = 0

            def draw(month, how_many: int) -> None:
                nonlocal count
                pool = by_month[k].get(month)
                if pool is None or not len(pool):
                    return
                for _ in range(how_many):
                    t = temps[month][rng.integers(len(temps[month]))]
                    when = times[k][pool[rng.integers(len(pool))]]
                    off = float(offsets[rng.integers(len(offsets))]) if t.entry_type == "limit" else None
                    sigs.append(make_signal(seg, t, when, off, f"rand{r}-{k}-{count}"))
                    count += 1

            for m, n in want.items():
                draw(m, n)
            tr = None
            for p in range(MAX_PASSES):
                tr = closed_trades(replay(seg, sigs))
                got = Counter(pd.DatetimeIndex(tr["known_at"]).to_period("M"))
                short = {m: n - got.get(m, 0) for m, n in want.items() if got.get(m, 0) < n}
                if not short or p == MAX_PASSES - 1:
                    break
                for m, n in short.items():
                    draw(m, n)
            wanted += sum(want.values())
            nets.append(tr["net_inr"].to_numpy(dtype=float))
            exits.append(tr["exit_time"].to_numpy())
        x = np.concatenate(nets) if nets else np.array([])
        out.append(Run(int(x.size), wanted, float(x.sum()), float(x.mean()) if x.size else None, x,
                       np.concatenate(exits) if exits else np.array([], dtype="datetime64[ns]")))
    settings = {"runs": n_runs, "seed": seed, "keeps": "direction, entry type, stop and target distances, limit "
                "waiting time, lots, cost, trades per month", "changes": "the time (a random bar close in the same "
                "month) and a limit's level (its distance from the price drawn from the idea's own)",
                "limit_distances_pool": int(offsets.size), "max_passes": MAX_PASSES}
    return out, settings


def summary(rs: list[Run], idea_per_trade: float | None) -> dict:
    pts = np.array([r.per_trade for r in rs if r.per_trade is not None])
    n = np.array([r.trades for r in rs])
    beaten = int((pts < idea_per_trade).sum()) if idea_per_trade is not None and pts.size else 0
    return {
        "runs": len(rs), "runs_with_trades": int(pts.size),
        "beaten": beaten, "beaten_share": round(beaten / pts.size, 4) if pts.size else 0.0,
        "per_trade_median": round(float(np.median(pts)), 2) if pts.size else None,
        "per_trade_5_95": [round(float(np.percentile(pts, 5)), 2), round(float(np.percentile(pts, 95)), 2)]
        if pts.size else None,
        "trades_median": int(np.median(n)) if n.size else 0, "trades_wanted": rs[0].target_trades if rs else 0,
    }


def pooled_noise(rs: list[Run]) -> np.ndarray:
    """Every random trade of every run: no-edge trades with the idea's own exits (the power table's noise)."""
    return np.concatenate([r.nets for r in rs]) if rs else np.array([])
