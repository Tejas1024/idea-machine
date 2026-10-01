"""How big an edge could this test even see? (L5, step 7)

Plant an edge of Rs 10 / 25 / 50 / 100 a trade into trades with NO edge, at 30 / 100 / 300 trades (and at the idea's
own trade count), and count the share the luck test catches. "Caught" = at least config.MIN_TRADES trades AND the
weekly bootstrap p below config.P_LIMIT — the same test the idea itself must pass (stats.bootstrap_p).

The no-edge trades ("noise") are the idea's random-entry runs (same exits, distances and cost, no edge) moved to a
mean of exactly Rs 0, so the table shows the idea's own spread of results. With no random runs, its own trades.
The planted trades are spread over weeks like the idea's own trades (the same trades per week).
"""
from __future__ import annotations

import math

import numpy as np

from machine import config
from machine.referee.stats import bootstrap_p


def caught_share(noise, edge: float, n: int, trades_per_week: float, *, sims: int = config.POWER_SIMS,
                 resamples: int = config.POWER_RESAMPLES, seed: int = 0) -> float:
    """The share of `sims` planted samples (n trades, `edge` rupees a trade) the luck test catches."""
    if n < config.MIN_TRADES:
        return 0.0                                  # fewer than MIN_TRADES trades can never be "works"
    x = np.asarray(noise, dtype=float)
    if x.size < 2:
        raise ValueError("the power table needs at least 2 no-edge trades to plant into")
    x = x - x.mean()
    rng = np.random.default_rng(seed)
    n_weeks = max(1, math.ceil(n / max(trades_per_week, 1e-9)))
    hit = 0
    for i in range(sims):
        sample = rng.choice(x, size=n, replace=True) + edge
        weeks = rng.integers(0, n_weeks, size=n)
        if bootstrap_p(sample, weeks, resamples, seed=rng.integers(1 << 62)) < config.P_LIMIT:
            hit += 1
    return hit / sims


def table(noise, own_n: int, trades_per_week: float, *, edges=config.POWER_EDGES_INR, counts=config.POWER_TRADES,
          sims: int = config.POWER_SIMS, resamples: int = config.POWER_RESAMPLES, seed: int = 0,
          noise_from: str = "") -> dict:
    """The power table: {trades: {edge: share caught}}, with the idea's own trade count as one of the rows."""
    rows = sorted({*counts, int(own_n)})
    out = {n: {e: caught_share(noise, e, n, trades_per_week, sims=sims, resamples=resamples,
                               seed=seed + 7919 * n + e) for e in edges}
           for n in rows}
    own = out[int(own_n)]
    seen = [e for e in edges if own[e] >= config.POWER_NEEDED_SHARE]
    return {
        "rows": {str(n): {str(e): round(s, 3) for e, s in r.items()} for n, r in out.items()},
        "own_n": int(own_n),
        "needed_edge_caught": round(own.get(config.POWER_NEEDED_EDGE,
                                            caught_share(noise, config.POWER_NEEDED_EDGE, own_n, trades_per_week,
                                                         sims=sims, resamples=resamples, seed=seed)), 3),
        "smallest_edge_seen": min(seen) if seen else None,
        "noise_sd_inr": round(float(np.std(np.asarray(noise, dtype=float), ddof=1)), 2) if len(noise) > 1 else None,
        "noise_from": noise_from,
        "trades_per_week": round(float(trades_per_week), 3),
        "sims": sims, "resamples": resamples,
    }


def text(t: dict) -> list[str]:
    edges = list(next(iter(t["rows"].values())))
    head = "  trades | " + " | ".join(f"₹{e} a trade" for e in edges)
    lines = [f"How often an edge of this size would be caught (planted {t['sims']} times into trades with no edge; "
             f"their spread: ₹{t['noise_sd_inr']:,.0f} a trade, from {t['noise_from']}):", head]
    for n, r in t["rows"].items():
        mark = "  ← this idea" if int(n) == t["own_n"] else ""
        lines.append(f"  {int(n):>6} | " + " | ".join(f"{100 * s:>5.0f} in 100" for s in r.values()) + mark)
    smallest = (f"₹{t['smallest_edge_seen']} a trade" if t["smallest_edge_seen"] is not None
                else f"more than ₹{max(int(e) for e in edges)} a trade")
    lines.append(f"The smallest edge it would catch at least half the time with {t['own_n']} trades: {smallest}")
    return lines
