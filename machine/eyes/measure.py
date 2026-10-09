"""How right a guesser is on his held-out answers (L15.2) - the same measures for both models, his written rule and
"always no".

    from machine.eyes import measure
    s = measure.score("model A", y, said_yes, chance, years)     # right / n, the 90% interval, both kinds found, by year
    measure.pair(right_a, right_b)                               # the two differ on how many charts, and is it luck?

    accuracy        right answers / answers, with its 90% interval (Wilson score: right for small counts too)
    yes found       of the charts he said yes to, how many it said yes to; no found the same for his no
    yes ranks higher   take one chart he said yes to and one he said no to: how often its chance is higher for the yes
                    one (0.5 = a coin, 1 = always; ties count half) - with a 90% interval from resampled answers
    pair            McNemar's exact test: only the charts where exactly one of the two is right count; with no real
                    difference each such chart is a coin toss, so "beyond luck" = fewer than 10 in 100 coin tosses
                    split at least that unevenly
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from machine import config


def wilson(k: int, n: int, level: float | None = None) -> tuple[float, float]:
    """The interval around k right of n (Wilson score)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + (config.EYES_LEVEL if level is None else level) / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def pair(right_a: np.ndarray, right_b: np.ndarray) -> dict:
    """A and B on the same charts: how many only A got right, only B, and the two-sided McNemar exact chance."""
    a, b = np.asarray(right_a, bool), np.asarray(right_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    k = only_a + only_b
    p = 1.0 if k == 0 else float(min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), k, 0.5)))
    return {"only_a": only_a, "only_b": only_b, "p": p, "beyond_luck": p < 1 - config.EYES_LEVEL}


def ranks_higher(y: np.ndarray, chance: np.ndarray) -> float:
    """How often a yes chart's chance is above a no chart's (ties half) = the area under the ROC curve."""
    y, s = np.asarray(y, int), np.asarray(chance, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ranks_interval(y: np.ndarray, chance: np.ndarray, runs: int | None = None, seed: int = 0) -> tuple[float, float]:
    y, s = np.asarray(y, int), np.asarray(chance, float)
    rng = np.random.default_rng(seed)
    got = []
    for _ in range(int(runs or config.EYES_BOOT)):
        i = rng.integers(0, len(y), len(y))
        v = ranks_higher(y[i], s[i])
        if np.isfinite(v):
            got.append(v)
    lo_q = (1 - config.EYES_LEVEL) / 2
    return (float(np.quantile(got, lo_q)), float(np.quantile(got, 1 - lo_q))) if got else (np.nan, np.nan)


@dataclass
class Score:
    name: str
    n: int
    right: int
    accuracy: float
    lo: float
    hi: float
    yes_found: int
    yes_all: int
    no_found: int
    no_all: int
    table: dict                                     # his answer x its answer
    by_year: dict = field(default_factory=dict)     # year -> [right, n]
    ranks: float = float("nan")
    ranks_lo: float = float("nan")
    ranks_hi: float = float("nan")

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def score(name: str, y: np.ndarray, said_yes: np.ndarray, chance: np.ndarray | None, years: np.ndarray) -> Score:
    y, said = np.asarray(y, int), np.asarray(said_yes, int)
    right = y == said
    k, n = int(right.sum()), len(y)
    lo, hi = wilson(k, n)
    table = {"he yes, it yes": int(((y == 1) & (said == 1)).sum()), "he yes, it no": int(((y == 1) & (said == 0)).sum()),
             "he no, it yes": int(((y == 0) & (said == 1)).sum()), "he no, it no": int(((y == 0) & (said == 0)).sum())}
    by_year = {int(yr): [int(right[years == yr].sum()), int((years == yr).sum())] for yr in sorted(set(years))}
    s = Score(name, n, k, k / n if n else float("nan"), lo, hi, table["he yes, it yes"], int((y == 1).sum()),
              table["he no, it no"], int((y == 0).sum()), table, by_year)
    if chance is not None:
        s.ranks = ranks_higher(y, chance)
        s.ranks_lo, s.ranks_hi = ranks_interval(y, chance)
    return s


def half_width(n: int, p: float = 0.75, level: float | None = None) -> float:
    """How wide the interval is (+/- points) at n answers - for the "how sure" line."""
    lo, hi = wilson(int(round(p * n)), n, level)
    return (hi - lo) / 2
