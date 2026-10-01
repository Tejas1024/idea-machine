"""Condition tables and the slice rules (L13.1; plan/likhith.md section D).

A CONDITION TABLE: an idea's trades split into the parts of one condition (machine/learner/conditions.py) - trades,
rupees a trade after the Rs 40 cost, and an interval: the weeks are drawn again with replacement (a "weekly block
bootstrap", like the referee's luck test - trades of one week stay together), and the middle 90 of 100 draws is shown.

A SLICE is a condition the idea's form chose BEFORE the test (at most config.SLICE_BUDGET = 8). Only a slice can become
"works best in ..." or "fails in ...", and only when it passes the three checks of RULES below. Every other condition
is "looked at, not tested": its table is shown and nothing in it is a finding.

The rules are FIXED IN ADVANCE: RULES is printed with every result and its fingerprint (RULES_SHA256) goes into every
learner trial file; tests/test_learner.py pins it, so a change after seeing a result shows in git. The numbers are in
machine/config.py (Claude's, not Tejas's rules).

How the three checks are counted (one slice, its trades with a known part):
  - a part is ELIGIBLE with at least SLICE_MIN_TRADES trades in it and as many in the rest;
  - gap(part) = rupees a trade in the part - rupees a trade in the rest of the slice's trades;
    size(part) = |gap| / the gap's own spread (the usual one for two groups of trades: the square root of each
    group's variance over its count, added) - so a part with few trades, or wild ones, needs a bigger gap;
    THE PART = the eligible part with the biggest size (a tie: the part with fewer trades).
  - check 1, the luck test: the weeks are drawn again BOOTSTRAP_RESAMPLES times. In each draw every eligible part's
    |gap in the draw - real gap| is divided by ITS spread IN THAT DRAW, and the biggest over the parts is kept (the
    real part was also picked as the biggest). p = the share of draws at least as big as the real size.
  - check 2: the gap of THE PART on the trades closed in each of config.SLICE_HALVES has the sign of the whole gap.
  - check 3, the shuffled-label test: the slice's labels are shuffled among the trades closed in the same year
    SLICE_SHUFFLES times (every part keeps its size in every year); the biggest size over the eligible parts is
    counted on each shuffle the same way. p_shuffle = the share of shuffles at least as big as the real size.
  - Holm's correction over the m slices chosen, for check 1 and again for check 3: the smallest p needs under
    P_LIMIT / m, the next under P_LIMIT / (m - 1) ...; after the first slice that fails, none passes.

Why the spread is measured inside every draw, and why check 3 carries the correction too: MEASURED on made-up trades
with no edge, shaped like idea #1's (1 trade in 7 wins big). With one fixed spread per part, 7 of 100 no-edge ideas
got a slice through check 1 (a small part that happens to hold few winners looks calm, so its gap looks surer than it
is), and 6.5 of 100 got a finding. The numbers after the change are in plan/work/L131-notes.md; the test
tests/test_learner.py::test_pure_noise_gives_almost_no_finding keeps watch.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from machine import config
from machine.referee import stats

WORKS_BEST = "works best in"
FAILS = "fails in"
FEW = "too few trades"


def _years(ys) -> str:
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}-{ys[-1]}"


HALVES_TEXT = " and in ".join(_years(h) for h in config.SLICE_HALVES)
RULES = f"""THE SLICE RULES (fixed before any condition table was seen; the same for every idea)
A slice is one way of splitting an idea's trades, chosen in the idea's form BEFORE the test (at most {config.SLICE_BUDGET}).
The trades split: the judged version's trades on {config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]}, after the ₹{config.COST_INR_PER_POSITION} cost, 0.01 lot.
A part of a slice (say London) is compared with the rest of that slice's trades: ₹ a trade in the part minus ₹ a
trade in the rest. A part is tested only with at least {config.SLICE_MIN_TRADES} trades in it and {config.SLICE_MIN_TRADES} in the rest - else "{FEW}".
How big a gap is: the gap divided by its own spread - a part with few trades, or wild ones, needs a bigger gap.
A part becomes "{WORKS_BEST} ..." (it earns more a trade than the rest) or "{FAILS} ..." (it earns less) only with ALL of:
  1. The luck test: the weeks are drawn again {config.BOOTSTRAP_RESAMPLES:,} times in a world where the split makes no difference. A gap
     this big, between any part and its rest, must show up less than {round(100 * config.P_LIMIT)} times in 100 - after a correction for the
     number of slices chosen (Holm): with {config.SLICE_BUDGET} slices the clearest one needs under {100 * config.P_LIMIT / config.SLICE_BUDGET:.2f} in 100, the next under
     {100 * config.P_LIMIT / (config.SLICE_BUDGET - 1):.2f} in 100, and so on; after the first slice that fails, none passes.
  2. The gap points the same way in {HALVES_TEXT}.
  3. The shuffled-label test: the slice's labels are shuffled among the trades of the same year {config.SLICE_SHUFFLES:,} times.
     A gap this big must show up less than {round(100 * config.P_LIMIT)} times in 100 - with the same correction as in check 1.
Every other condition is "looked at, not tested": its table is shown, and nothing in it is a finding.
The hidden years {config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]} are shown next to a slice once they were opened for this version. They decide nothing here."""
RULES_SHA256 = hashlib.sha256(RULES.encode("utf-8")).hexdigest()


# ================================================================================================= helpers

def seed_of(ids, name: str = "") -> int:
    """A seed from the trades themselves and the condition's name: the same trades give the same draws on every PC."""
    text = "|".join(sorted(str(i) for i in ids)) + "#" + name
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _arrays(trades: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = trades["net_inr"].to_numpy(dtype=float)
    t = pd.DatetimeIndex(pd.to_datetime(trades["exit_time"]))
    return v, stats.week_ids(t), t.year.to_numpy()


def interval(values, weeks, *, seed: int, n_boot: int | None = None, share: float | None = None) -> list | None:
    """The middle `share` of rupees a trade when the weeks are drawn again with replacement (None: fewer than 2 trades
    or 1 week - nothing to draw)."""
    n_boot = config.BOOTSTRAP_RESAMPLES if n_boot is None else n_boot
    share = config.SLICE_INTERVAL if share is None else share
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    _, inv = np.unique(np.asarray(weeks), return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv).astype(float)
    if sums.size < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sums.size, size=(n_boot, sums.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - share) / 2, 1 - (1 - share) / 2])
    return [round(float(lo), 2), round(float(hi), 2)]


def numbers(v: np.ndarray) -> dict:
    """Trades, rupees, rupees a trade and how many won, for one part's trades."""
    n = int(v.size)
    return {"trades": n, "net_inr": round(float(v.sum()), 2) if n else 0.0,
            "per_trade": round(float(v.mean()), 2) if n else None, "won": int((v > 0).sum())}


# ================================================================================================= a condition table

def table(trades: pd.DataFrame, labels: pd.Series, groups, *, seed: int) -> dict:
    """One condition's table. trades: net_inr + exit_time, one row per closed trade; labels: each trade's part (None =
    not known), in the same order. Per part: trades, rupees, rupees a trade, won, the interval, the rest's rupees a
    trade and the gap, each half of the tune years, and `enough` (it may be tested)."""
    lab = np.asarray(labels, dtype=object)
    if len(lab) != len(trades):
        raise ValueError("one label per trade")
    known = ~pd.isna(lab)
    v, w, y = _arrays(trades)
    v, w, y, lab = v[known], w[known], y[known], lab[known]
    n = len(v)
    rows = []
    for k, g in enumerate(groups):
        m = lab == g
        row = {"group": g, **numbers(v[m])}
        row["interval"] = interval(v[m], w[m], seed=seed + k) if row["trades"] else None
        rest = v[~m]
        row["rest_trades"] = int(rest.size)
        row["rest_per_trade"] = round(float(rest.mean()), 2) if rest.size else None
        row["gap"] = (round(float(v[m].mean() - rest.mean()), 2) if row["trades"] and rest.size else None)
        row["enough"] = bool(row["trades"] >= config.SLICE_MIN_TRADES and n - row["trades"] >= config.SLICE_MIN_TRADES)
        row["halves"] = [{"years": _years(h), **numbers(v[m & np.isin(y, h)])} for h in config.SLICE_HALVES]
        rows.append(row)
    return {"trades": int(n), "not_known": int((~known).sum()), "groups": rows}


# ================================================================================================= the three checks

def _gap_and_spread(s1, s2, n, t1, t2, tn):
    """A part (the sum of its rupees s1, of their squares s2, its count n) against the rest of the total (t1, t2,
    tn): the gap in rupees a trade, and the gap's spread = sqrt(var part / n + var rest / count rest). NaN where a
    side has fewer than 2 trades or no spread at all."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r1, r2, rn = t1 - s1, t2 - s2, tn - n
        gap = s1 / n - r1 / rn
        var = (s2 - s1 * s1 / n) / (n - 1) / n + (r2 - r1 * r1 / rn) / (rn - 1) / rn
        spread = np.sqrt(np.where(var > 1e-12, var, np.nan))
    bad = (np.asarray(n) < 2) | (np.asarray(rn) < 2)
    return np.where(bad, np.nan, gap), np.where(bad, np.nan, spread)


def test(trades: pd.DataFrame, labels: pd.Series, groups, *, seed: int, n_boot: int | None = None,
         n_shuffle: int | None = None, slices_chosen: int | None = None) -> dict:
    """The three checks for one slice (see the module text), without the Holm step (judge() does it over all slices).
    Returns tested False with p = p_shuffle = 1 when no part has enough trades.
    could_see = the smallest gap (rupees a trade, THE PART against the rest) that would reach check 1's strictest
    line with these trades (P_LIMIT / slices_chosen; default the budget of 8): a smaller real gap cannot be told from
    luck here - so "nothing passed" never means "no difference"."""
    n_boot = config.BOOTSTRAP_RESAMPLES if n_boot is None else n_boot
    n_shuffle = config.SLICE_SHUFFLES if n_shuffle is None else n_shuffle
    lab = np.asarray(labels, dtype=object)
    known = ~pd.isna(lab)
    v, w, y = _arrays(trades)
    v, w, y, lab = v[known], w[known], y[known], lab[known]
    n = len(v)
    masks = {g: lab == g for g in groups}
    size = {g: int(m.sum()) for g, m in masks.items()}
    eligible = [g for g in groups if size[g] >= config.SLICE_MIN_TRADES and n - size[g] >= config.SLICE_MIN_TRADES]
    if not eligible:
        return {"tested": False, "why": FEW, "p": 1.0, "p_shuffle": 1.0, "eligible": []}
    v2 = v * v
    t1, t2 = float(v.sum()), float(v2.sum())
    gap, big, spread = {}, {}, {}
    for g in eligible:
        m = masks[g]
        d, sp = _gap_and_spread(float(v[m].sum()), float(v2[m].sum()), size[g], t1, t2, n)
        gap[g], spread[g] = float(d), float(sp)
        big[g] = abs(float(d)) / float(sp) if np.isfinite(sp) else 0.0
    top = max(big.values())
    part = min((g for g in eligible if big[g] >= top - 1e-9), key=lambda g: (size[g], eligible.index(g)))
    # ---- check 1: the weeks drawn again; every draw measures its own spread
    _, inv = np.unique(w, return_inverse=True)
    n_weeks = int(inv.max()) + 1
    draws = np.random.default_rng(seed).integers(0, n_weeks, size=(n_boot, n_weeks))

    def drawn(x: np.ndarray) -> np.ndarray:
        return np.bincount(inv, weights=x, minlength=n_weeks)[draws].sum(axis=1)

    a1, a2, an = drawn(v), drawn(v2), drawn(np.ones(n))
    worst = np.full(n_boot, -np.inf)
    for g in eligible:
        m = masks[g].astype(float)
        d, sp = _gap_and_spread(drawn(v * m), drawn(v2 * m), drawn(m), a1, a2, an)
        with np.errstate(invalid="ignore"):
            z = np.abs(d - gap[g]) / sp
        worst = np.where(np.isfinite(z), np.maximum(worst, z), worst)
    has = np.isfinite(worst)
    p = float((1 + np.count_nonzero(worst[has] >= top - 1e-9)) / (1 + int(has.sum()))) if top > 0 else 1.0
    m_slices = config.SLICE_BUDGET if slices_chosen is None else max(1, int(slices_chosen))
    could_see = None
    if has.any() and np.isfinite(spread[part]):
        could_see = round(float(np.quantile(worst[has], 1 - config.P_LIMIT / m_slices) * spread[part]), 2)
    # ---- check 2: the same way in both halves
    m = masks[part]
    halves = []
    for h in config.SLICE_HALVES:
        inside = np.isin(y, h)
        a, b = v[m & inside], v[~m & inside]
        halves.append({"years": _years(h), "trades": int(a.size), "rest_trades": int(b.size),
                       "gap": round(float(a.mean() - b.mean()), 2) if a.size and b.size else None})
    same = all(x["gap"] is not None and x["gap"] * gap[part] > 0 for x in halves)
    # ---- check 3: the labels shuffled among the same year's trades
    codes = np.full(n, -1, dtype=np.int16)
    for k, g in enumerate(eligible):
        codes[masks[g]] = k
    by_year = np.argsort(y, kind="stable")
    keys = np.random.default_rng(seed + 1).random((n_shuffle, n))
    order = np.argsort(y[None, :] * 2.0 + keys, axis=1)               # each year's trades, in a random order
    shuffled = np.empty((n_shuffle, n), dtype=np.int16)
    np.put_along_axis(shuffled, order, np.broadcast_to(codes[by_year], (n_shuffle, n)), axis=1)
    worst_s = np.zeros(n_shuffle)
    for k, g in enumerate(eligible):
        pick = shuffled == k
        d, sp = _gap_and_spread(pick @ v, pick @ v2, float(size[g]), t1, t2, float(n))
        with np.errstate(invalid="ignore"):
            z = np.abs(d) / sp
        worst_s = np.where(np.isfinite(z), np.maximum(worst_s, z), worst_s)
    p_shuffle = float((1 + np.count_nonzero(worst_s >= top - 1e-9)) / (1 + n_shuffle)) if top > 0 else 1.0
    return {"tested": True, "eligible": eligible, "part": part, "trades": size[part], "rest_trades": n - size[part],
            "gap": round(gap[part], 2), "size": round(float(top), 4), "p": round(p, 5), "halves": halves,
            "same_way": bool(same), "p_shuffle": round(p_shuffle, 5), "draws": int(has.sum()),
            "shuffles": int(n_shuffle), "could_see": could_see}


def holm(ps: dict[str, float], alpha: float | None = None) -> dict[str, dict]:
    """Holm's correction over the slices chosen: the smallest p needs under alpha / m, the next under
    alpha / (m - 1) ...; after the first that fails, none passes. ps keeps the form's order (ties go in that order)."""
    alpha = config.P_LIMIT if alpha is None else alpha
    m = len(ps)
    out, failed = {}, False
    for rank, name in enumerate(sorted(ps, key=lambda k: ps[k])):
        needs = alpha / (m - rank)
        passed = (not failed) and ps[name] < needs
        failed = failed or not passed
        out[name] = {"place": rank + 1, "needs_under": needs, "passed": bool(passed)}
    return {name: out[name] for name in ps}


def judge(tests: dict[str, dict]) -> dict[str, dict]:
    """Every chosen slice's answer: the three checks, and its claim ("works best in" / "fails in" / None). tests:
    slice name -> test(), in the form's order (a slice that could not be measured: {"tested": False, "p": 1})."""
    luck = holm({name: float(t.get("p", 1.0)) for name, t in tests.items()})
    shuffle = holm({name: float(t.get("p_shuffle", 1.0)) for name, t in tests.items()})
    out = {}
    for name, t in tests.items():
        r = {**t, "holm": luck[name], "holm_shuffle": shuffle[name]}
        if not t.get("tested"):
            r.update(passed=False, claim=None)
        else:
            ok = luck[name]["passed"] and t["same_way"] and shuffle[name]["passed"]
            r.update(passed=bool(ok), claim=(WORKS_BEST if t["gap"] > 0 else FAILS) if ok else None)
        out[name] = r
    return out
