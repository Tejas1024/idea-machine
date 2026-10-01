"""All tested ideas' trades and random-entry trades together (L13.2, part B): which conditions help ANY entry, and
which help only THIS idea.

    from machine.learner import pooled
    r = pooled.analyse(this, others)          # Source: an idea's trades, their setups' features, its random entries
    print(pooled.text(r))

THE TRADES (the tune years 2016-2020; rupees after the Rs 40 cost, 0.01 lot):
  - THIS idea: the judged version's trades (the referee's tune file) - the trades its slices split (L13.1);
  - HIS OTHER TESTED ENTRIES: every other idea judged on 2016-2020 in the trial files (L19b's entry types, kind
    "hedging-2x2"), made again by the same code with a plain stop at the hedge point - they must be the same trades
    as the trial file (count and rupees), else that idea is left out and the reason said; a trade already in the pool
    (the same side, fill minute and fill price) counts once;
  - RANDOM ENTRIES for each of them, made like the referee's (machine/referee/random_entries.py: the same side, entry
    type, distances, lots and trades per month; a random time in the same month; a limit's distance drawn from its
    own): config.POOLED_RANDOM_RUNS runs, at most config.POOLED_RANDOM_MAX random trades each.
THE CONDITIONS: the learner's conditions that read the market (machine/learner/conditions.py, needs_facts False) -
random entries have no steps of their own. The same parts and cut points as L13.1.
THE TWO QUESTIONS, each with Holm's correction for the number of conditions tested:
  ANY ENTRY - a part where the random entries (all of them together) earn more, or less, than in the rest of their
    trades: the three checks of the slice rules (machine/learner/slices.py).
  ONLY THIS IDEA - a part where THIS idea's gap (the part against the rest of its trades) differs from the gap of its
    OWN random entries (the same shapes: sides, distances, months): the same three checks made for the difference -
    the weeks drawn again for both at once (the spread measured in every draw), the same way in 2016-2018 and
    2019-2020, the labels shuffled among the same year's trades of each.
His other entries' numbers are shown next to them: looked at, not tested. Every number is Claude's, fixed before any
pooled table was made (plan/work/L132-notes.md).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config
from machine import trials as trial_files
from machine.learner import conditions as CN
from machine.learner import features as F
from machine.learner import slices as SL

MARKET: tuple[CN.Condition, ...] = tuple(c for c in CN.ALL if not c.needs_facts)
ANY, ONLY = "any entry", "only this idea"
CODE_FILES = ("pooled.py", "slices.py", "conditions.py", "features.py")


def _inr(x) -> str:
    if x is None:
        return "no trades"
    r = round(x)
    return f"-₹{-r:,}" if r < 0 else f"₹{abs(r):,}"


@dataclass
class Source:
    """One tested idea in the pool: its trades (net_inr, exit_time; index = signal id), the features of their setups
    (the same index), and its random entries (the same two tables)."""
    name: str
    label: str
    version_id: str
    trades: pd.DataFrame
    feats: pd.DataFrame
    random_trades: pd.DataFrame
    random_feats: pd.DataFrame
    notes: list[str] = field(default_factory=list)


# ================================================================================================================
# the difference test (ONLY THIS IDEA)
# ================================================================================================================

def _shuffled_codes(codes: np.ndarray, years: np.ndarray, n_shuffle: int, rng) -> np.ndarray:
    """The codes shuffled among the trades of the same year, n_shuffle times (every part keeps its size each year)."""
    n = codes.size
    by_year = np.argsort(years, kind="stable")
    keys = rng.random((n_shuffle, n))
    order = np.argsort(years[None, :] * 2.0 + keys, axis=1)
    out = np.empty((n_shuffle, n), dtype=np.int16)
    np.put_along_axis(out, order, np.broadcast_to(codes[by_year], (n_shuffle, n)), axis=1)
    return out


def diff_test(a: pd.DataFrame, labels_a, b: pd.DataFrame, labels_b, groups, *, seed: int,
              n_boot: int | None = None, n_shuffle: int | None = None, tested: int | None = None) -> dict:
    """ONLY THIS IDEA for one condition: the part whose gap in `a` (this idea's trades) differs most from its gap in
    `b` (its own random entries), and the three checks for that difference (see the module text). Holm is applied by
    judge() over the conditions. tested: how many conditions are tested (for could_see; default all of MARKET)."""
    n_boot = config.BOOTSTRAP_RESAMPLES if n_boot is None else n_boot
    n_shuffle = config.SLICE_SHUFFLES if n_shuffle is None else n_shuffle
    la, lb = np.asarray(labels_a, dtype=object), np.asarray(labels_b, dtype=object)
    ka, kb = ~pd.isna(la), ~pd.isna(lb)
    va, wa, ya = SL._arrays(a)
    vb, wb, yb = SL._arrays(b)
    va, wa, ya, la = va[ka], wa[ka], ya[ka], la[ka]
    vb, wb, yb, lb = vb[kb], wb[kb], yb[kb], lb[kb]
    na, nb = len(va), len(vb)
    ma = {g: la == g for g in groups}
    mb = {g: lb == g for g in groups}
    sa = {g: int(m.sum()) for g, m in ma.items()}
    sb = {g: int(m.sum()) for g, m in mb.items()}
    low = config.SLICE_MIN_TRADES
    eligible = [g for g in groups if min(sa[g], na - sa[g], sb[g], nb - sb[g]) >= low]
    if not eligible:
        return {"tested": False, "why": SL.FEW, "p": 1.0, "p_shuffle": 1.0, "eligible": []}
    va2, vb2 = va * va, vb * vb
    ta = (float(va.sum()), float(va2.sum()), float(na))
    tb = (float(vb.sum()), float(vb2.sum()), float(nb))
    d, spread, big, gap_a, gap_b = {}, {}, {}, {}, {}
    for g in eligible:
        x, sx = SL._gap_and_spread(float(va[ma[g]].sum()), float(va2[ma[g]].sum()), sa[g], *ta)
        y, sy = SL._gap_and_spread(float(vb[mb[g]].sum()), float(vb2[mb[g]].sum()), sb[g], *tb)
        gap_a[g], gap_b[g] = float(x), float(y)
        d[g] = float(x - y)
        spread[g] = float(np.sqrt(sx * sx + sy * sy))
        big[g] = abs(d[g]) / spread[g] if np.isfinite(spread[g]) and spread[g] > 0 else 0.0
    top = max(big.values())
    part = min((g for g in eligible if big[g] >= top - 1e-9), key=lambda g: (sa[g], eligible.index(g)))
    # ---- check 1: the weeks drawn again, the same draw for both sets of trades
    weeks = np.unique(np.concatenate([wa, wb]))
    ia, ib = np.searchsorted(weeks, wa), np.searchsorted(weeks, wb)
    n_weeks = len(weeks)
    draws = np.random.default_rng(seed).integers(0, n_weeks, size=(n_boot, n_weeks))

    def drawn(inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.bincount(inv, weights=x, minlength=n_weeks)[draws].sum(axis=1)

    A = (drawn(ia, va), drawn(ia, va2), drawn(ia, np.ones(na)))
    B = (drawn(ib, vb), drawn(ib, vb2), drawn(ib, np.ones(nb)))
    worst = np.full(n_boot, -np.inf)
    for g in eligible:
        xa, xb = ma[g].astype(float), mb[g].astype(float)
        x, sx = SL._gap_and_spread(drawn(ia, va * xa), drawn(ia, va2 * xa), drawn(ia, xa), *A)
        y, sy = SL._gap_and_spread(drawn(ib, vb * xb), drawn(ib, vb2 * xb), drawn(ib, xb), *B)
        with np.errstate(invalid="ignore"):
            z = np.abs((x - y) - d[g]) / np.sqrt(sx * sx + sy * sy)
        worst = np.where(np.isfinite(z), np.maximum(worst, z), worst)
    has = np.isfinite(worst)
    p = float((1 + np.count_nonzero(worst[has] >= top - 1e-9)) / (1 + int(has.sum()))) if top > 0 else 1.0
    m_tests = len(MARKET) if tested is None else max(1, int(tested))
    could_see = None
    if has.any() and np.isfinite(spread[part]):
        could_see = round(float(np.quantile(worst[has], 1 - config.P_LIMIT / m_tests) * spread[part]), 2)
    # ---- check 2: the same way in both halves
    halves = []
    for h in config.SLICE_HALVES:
        in_a, in_b = np.isin(ya, h), np.isin(yb, h)
        a1, a0 = va[ma[part] & in_a], va[~ma[part] & in_a]
        b1, b0 = vb[mb[part] & in_b], vb[~mb[part] & in_b]
        ok = min(a1.size, a0.size, b1.size, b0.size) > 0
        halves.append({"years": SL._years(h), "trades": int(a1.size), "random_trades": int(b1.size),
                       "gap": round(float((a1.mean() - a0.mean()) - (b1.mean() - b0.mean())), 2) if ok else None})
    same = all(x["gap"] is not None and x["gap"] * d[part] > 0 for x in halves)
    # ---- check 3: the labels shuffled among the same year's trades, in each set
    rng = np.random.default_rng(seed + 1)
    codes_a, codes_b = np.full(na, -1, np.int16), np.full(nb, -1, np.int16)
    for k, g in enumerate(eligible):
        codes_a[ma[g]], codes_b[mb[g]] = k, k
    sh_a = _shuffled_codes(codes_a, ya, n_shuffle, rng)
    sh_b = _shuffled_codes(codes_b, yb, n_shuffle, rng)
    worst_s = np.zeros(n_shuffle)
    for k, g in enumerate(eligible):
        pa, pb = sh_a == k, sh_b == k
        x, sx = SL._gap_and_spread(pa @ va, pa @ va2, float(sa[g]), *ta)
        y, sy = SL._gap_and_spread(pb @ vb, pb @ vb2, float(sb[g]), *tb)
        with np.errstate(invalid="ignore"):
            z = np.abs(x - y) / np.sqrt(sx * sx + sy * sy)
        worst_s = np.where(np.isfinite(z), np.maximum(worst_s, z), worst_s)
    p_shuffle = float((1 + np.count_nonzero(worst_s >= top - 1e-9)) / (1 + n_shuffle)) if top > 0 else 1.0
    return {"tested": True, "eligible": eligible, "part": part, "trades": sa[part], "rest_trades": na - sa[part],
            "random_trades": sb[part], "random_rest_trades": nb - sb[part],
            "gap_this": round(gap_a[part], 2), "gap_random": round(gap_b[part], 2), "gap": round(d[part], 2),
            "size": round(float(top), 4), "p": round(p, 5), "halves": halves, "same_way": bool(same),
            "p_shuffle": round(p_shuffle, 5), "draws": int(has.sum()), "shuffles": int(n_shuffle),
            "could_see": could_see}


def judge(tests: dict[str, dict], gap_key: str = "gap") -> dict[str, dict]:
    """Holm over the conditions tested, for the luck test and again for the shuffled labels; a claim needs both and
    the same way in both halves. The claim's word: the gap's sign ("more" / "less")."""
    luck = SL.holm({n: float(t.get("p", 1.0)) for n, t in tests.items()})
    shuf = SL.holm({n: float(t.get("p_shuffle", 1.0)) for n, t in tests.items()})
    out = {}
    for n, t in tests.items():
        r = {**t, "holm": luck[n], "holm_shuffle": shuf[n]}
        ok = bool(t.get("tested") and luck[n]["passed"] and t["same_way"] and shuf[n]["passed"])
        r.update(passed=ok, claim=("more" if t[gap_key] > 0 else "less") if ok else None)
        out[n] = r
    return out


# ================================================================================================================
# the analysis
# ================================================================================================================

def _numbers(v: np.ndarray) -> dict:
    return SL.numbers(v)


def _part_rows(cond: CN.Condition, sets: dict[str, tuple[pd.DataFrame, pd.Series]]) -> list[dict]:
    rows = []
    for g in cond.groups:
        row = {"group": g}
        for key, (tr, lab) in sets.items():
            v = tr["net_inr"].to_numpy(dtype=float)
            m = np.asarray(lab, dtype=object) == g
            row[key] = _numbers(v[m])
        if any(row[k]["trades"] for k in sets):
            rows.append(row)
    return rows


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames) if frames else pd.DataFrame(columns=["net_inr", "exit_time"])


def analyse(this: Source, others: list[Source], *, n_boot: int | None = None, n_shuffle: int | None = None) -> dict:
    """The pooled view for THIS idea (see the module text)."""
    for s in (this, *others):
        for tr, fe, what in ((s.trades, s.feats, "trades"), (s.random_trades, s.random_feats, "random trades")):
            lost = [i for i in tr.index if i not in fe.index]
            if lost:
                raise ValueError(f"{s.name}: {len(lost)} {what} have no setup in the feature table")
            bad = sorted(set(pd.DatetimeIndex(pd.to_datetime(tr["exit_time"])).year) - set(config.TUNE_YEARS))
            if bad:
                raise ValueError(f"the pooled view reads the tune years only; {s.name}'s {what} closed in {bad}")
    rand_t = _concat([this.random_trades] + [o.random_trades for o in others])
    rand_f = _concat([this.random_feats] + [o.random_feats for o in others])
    oth_t = _concat([o.trades for o in others])
    oth_f = _concat([o.feats for o in others])
    if rand_t.index.has_duplicates or oth_t.index.has_duplicates:
        raise ValueError("two trades of the pool share a signal id")
    conds, tests_any, tests_only, not_here = [], {}, {}, []
    n_tests = len(MARKET)
    ids = list(this.trades.index)
    for cond in MARKET:
        lab_this = cond.labels(this.feats.loc[this.trades.index])
        lab_own = cond.labels(this.random_feats.loc[this.random_trades.index])
        lab_rand = cond.labels(rand_f.loc[rand_t.index])
        sets = {"this": (this.trades, lab_this), "own_random": (this.random_trades, lab_own),
                "random": (rand_t, lab_rand)}
        if len(oth_t):
            sets["others"] = (oth_t, cond.labels(oth_f.loc[oth_t.index]))
        parts = set(lab_rand.dropna()) | set(lab_this.dropna())
        if len(parts) < 2:
            not_here.append({"name": cond.name, "label": cond.label,
                             "why": "never known here" if not parts else f"every trade is '{next(iter(parts))}'"})
            continue
        seed = SL.seed_of(ids, "pooled:" + cond.name)
        tests_any[cond.name] = SL.test(rand_t, lab_rand, cond.groups, seed=seed, n_boot=n_boot, n_shuffle=n_shuffle,
                                       slices_chosen=n_tests)
        tests_only[cond.name] = diff_test(this.trades, lab_this, this.random_trades, lab_own, cond.groups,
                                          seed=seed + 7, n_boot=n_boot, n_shuffle=n_shuffle, tested=n_tests)
        conds.append({"name": cond.name, "label": cond.label, "dimension": cond.dimension, "source": cond.source,
                      "how": cond.how, "parts": _part_rows(cond, sets),
                      "not_known": {k: int(pd.isna(np.asarray(lab, dtype=object)).sum())
                                    for k, (_, lab) in sets.items()}})
    any_j, only_j = judge(tests_any, "gap"), judge(tests_only, "gap")
    claims_any, claims_only = [], []
    for c in conds:
        c["any"], c["only"] = any_j[c["name"]], only_j[c["name"]]
        a, o = c["any"], c["only"]
        if a["claim"]:
            g = next(x for x in c["parts"] if x["group"] == a["part"])
            rest = _rest(c["parts"], a["part"], "random")
            claims_any.append({"condition": c["name"], "label": c["label"], "part": a["part"], "claim": a["claim"],
                               "gap": a["gap"], "trades": g["random"]["trades"],
                               "per_trade": g["random"]["per_trade"], "rest_per_trade": rest})
        if o["claim"]:
            g = next(x for x in c["parts"] if x["group"] == o["part"])
            claims_only.append({"condition": c["name"], "label": c["label"], "part": o["part"], "claim": o["claim"],
                                "gap": o["gap"], "gap_this": o["gap_this"], "gap_random": o["gap_random"],
                                "trades": g["this"]["trades"], "per_trade": g["this"]["per_trade"],
                                "rest_per_trade": _rest(c["parts"], o["part"], "this"),
                                "random_per_trade": g["own_random"]["per_trade"],
                                "random_rest_per_trade": _rest(c["parts"], o["part"], "own_random")})
    src = [{"name": s.name, "label": s.label, "version_id": s.version_id, "this": s is this,
            "trades": int(len(s.trades)), **_money(s.trades), "random_trades": int(len(s.random_trades)),
            "random_per_trade": _money(s.random_trades)["per_trade"], "notes": list(s.notes)}
           for s in (this, *others)]
    return {
        "years": list(config.TUNE_YEARS), "this": this.name, "sources": src,
        "trades": int(len(this.trades)), "own_random_trades": int(len(this.random_trades)),
        "random_trades": int(len(rand_t)), "other_trades": int(len(oth_t)),
        "other_ideas": [o.name for o in others],
        "conditions": conds, "not_here": not_here, "claims_any": claims_any, "claims_only": claims_only,
        "counts": {"conditions": len(MARKET), "tested": len(conds), "not_here": len(not_here),
                   "any": len(claims_any), "only": len(claims_only)},
    }


def _money(tr: pd.DataFrame) -> dict:
    v = tr["net_inr"].to_numpy(dtype=float) if len(tr) else np.array([])
    return {"net_inr": round(float(v.sum()), 2), "per_trade": round(float(v.mean()), 2) if v.size else None}


def _rest(parts: list[dict], part: str, key: str) -> float | None:
    n = sum(x[key]["trades"] for x in parts if x["group"] != part)
    s = sum(x[key]["net_inr"] for x in parts if x["group"] != part)
    return round(s / n, 2) if n else None


# ================================================================================================================
# the words
# ================================================================================================================

def any_words(c: dict) -> str:
    return (f"{c['label']} '{c['part']}': any entry earns {c['claim']} there (random entries {_inr(c['per_trade'])} "
            f"a trade there, {_inr(c['rest_per_trade'])} elsewhere)")


def only_words(c: dict) -> str:
    better = "better" if c["claim"] == "more" else "worse"
    return (f"{c['label']} '{c['part']}': this idea does {better} there than any entry would (this idea "
            f"{_inr(c['per_trade'])} a trade there, {_inr(c['rest_per_trade'])} elsewhere; random entries of its "
            f"shape {_inr(c['random_per_trade'])} there, {_inr(c['random_rest_per_trade'])} elsewhere)")


def head_line(r: dict) -> str:
    others = (f", {r['other_trades']:,} trades of {len(r['other_ideas'])} other tested "
              f"{'idea' if len(r['other_ideas']) == 1 else 'ideas'}" if r["other_ideas"] else "")
    return (f"Any entry or only this idea ({years_text(r['years'])}; this idea's {r['trades']} trades, "
            f"{r['random_trades']:,} random entries{others}; {r['counts']['tested']} market conditions, each with "
            "the 3 checks and a correction for how many were tested)")


def any_line(r: dict) -> str:
    if not r["claims_any"]:
        return "Helps or hurts ANY entry: nothing passed the 3 checks"
    return "Helps or hurts ANY entry: " + "; ".join(any_words(c) for c in r["claims_any"])


def only_line(r: dict) -> str:
    if not r["claims_only"]:
        return "Only THIS idea (better or worse than any entry would do there): nothing passed the 3 checks"
    return "Only THIS idea: " + "; ".join(only_words(c) for c in r["claims_only"])


def years_text(years) -> str:
    ys = sorted(int(y) for y in years)
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"


def check_words(t: dict, only: bool) -> str:
    if not t.get("tested"):
        return f"not tested: {t.get('why', SL.FEW)}"
    h, hs = t["holm"], t["holm_shuffle"]
    what = (f"this idea's gap {_inr(t['gap_this'])} against its random entries' {_inr(t['gap_random'])}: a "
            f"difference of {_inr(t['gap'])}" if only else f"random entries: a gap of {_inr(t['gap'])} a trade")
    halves = " · ".join(f"{x['years']}: " + ("no trades on one side" if x["gap"] is None else _inr(x["gap"]))
                        for x in t["halves"])
    return (f"the part that stands out: {t['part']} ({what}). 1. luck gives it {100 * t['p']:.1f} times in 100 (needs "
            f"under {100 * h['needs_under']:.2f}): {'yes' if h['passed'] else 'NO'}. 2. the same way in both halves "
            f"({halves}): {'yes' if t['same_way'] else 'NO'}. 3. shuffled labels give it {100 * t['p_shuffle']:.1f} "
            f"times in 100 (needs under {100 * hs['needs_under']:.2f}): {'yes' if hs['passed'] else 'NO'}.")


def text(r: dict) -> str:
    lines = [f"THE POOLED VIEW · {r['this']} · ₹ after the ₹40 cost, 0.01 lot", head_line(r), any_line(r),
             only_line(r), "", "THE TRADES IN THE POOL:"]
    for s in r["sources"]:
        lines.append(f"  {s['label']} ({s['name']} {s['version_id']}): {s['trades']} trades, "
                     f"{_inr(s['per_trade'])} a trade · {s['random_trades']} random entries, "
                     f"{_inr(s['random_per_trade'])} a trade" + ("".join(f" · {n}" for n in s["notes"])))
    for c in r["conditions"]:
        lines += ["", f"{c['label']} [{c['name']}]"]
        for p in c["parts"]:
            cells = [f"this idea {p['this']['trades']} × {_inr(p['this']['per_trade'])}",
                     f"its random {p['own_random']['trades']} × {_inr(p['own_random']['per_trade'])}",
                     f"all random {p['random']['trades']} × {_inr(p['random']['per_trade'])}"]
            if "others" in p:
                cells.append(f"other ideas {p['others']['trades']} × {_inr(p['others']['per_trade'])}")
            lines.append(f"    {p['group']}: " + " · ".join(cells))
        lines.append("    ANY ENTRY: " + check_words(c["any"], False))
        lines.append("    ONLY THIS IDEA: " + check_words(c["only"], True))
    if r["not_here"]:
        lines += ["", "Not in the pool: " + "; ".join(f"{x['label']} ({x['why']})" for x in r["not_here"])]
    return "\n".join(lines)


# ================================================================================================================
# the pool from the real trades (the runner: machine/learner/run.py learn_pooled)
# ================================================================================================================

def setup_of(s, market, tf: str = "5m") -> F.Setup:
    """A random entry's setup: its moment, side, the entry (a limit's price, else the close of its candle), the stop
    and the nearest target as prices (a from-fill distance is put at that entry)."""
    t = pd.Timestamp(s.known_at)
    closes = market.arrays(tf)["close"]
    entry = float(s.entry_price) if s.entry_type == "limit" else float(closes[market.closed_by(tf, t)])
    sign = 1.0 if s.direction == "buy" else -1.0
    if s.levels == "price":
        stop = None if s.stop is None else float(s.stop)
        target = (float(max(s.targets) if s.direction == "sell" else min(s.targets))) if s.targets else None
    else:
        stop = None if s.stop is None else entry - sign * float(s.stop)
        target = entry + sign * float(min(s.targets)) if s.targets else None
    return F.Setup(s.id, t, s.direction, entry, stop, target, {})


def random_entries(seg, trades: pd.DataFrame, *, runs: int, cap: int, seed: int, name: str):
    """Random entries of one idea's shape (machine/referee/random_entries.py, the same keeps and changes), run by run:
    `runs` runs, whole runs only, stopping before a run that would pass `cap` closed trades (at least one run).
    trades: the idea's closed trades (signal, known_at). Returns (the random signals by id, their closed trades:
    net_inr, exit_time, known_at, signal)."""
    from machine.referee import random_entries as RE
    from machine.referee.replays import closed_trades, replay
    temps, want = RE.plan(seg, trades)
    offsets = np.array([t.offset for ts in temps.values() for t in ts if t.offset is not None])
    times = seg.known_times()
    per = times.to_period("M")
    by_month = {m: np.flatnonzero(per == m) for m in per.unique()}
    sigs_all, frames = {}, []
    got = 0
    for r in range(runs):
        rng = np.random.default_rng([seed, r])
        sigs, count = [], 0

        def draw(month, how_many: int) -> None:
            nonlocal count
            pool = by_month.get(month)
            if pool is None or not len(pool):
                return
            for _ in range(how_many):
                t = temps[month][rng.integers(len(temps[month]))]
                when = times[pool[rng.integers(len(pool))]]
                off = float(offsets[rng.integers(len(offsets))]) if t.entry_type == "limit" else None
                sigs.append(RE.make_signal(seg, t, when, off, f"{name}:rand{r}-{count}"))
                count += 1

        for m, n in want.items():
            draw(m, n)
        tr = None
        for p in range(RE.MAX_PASSES):
            tr = closed_trades(replay(seg, sigs))
            have = pd.DatetimeIndex(tr["known_at"]).to_period("M").value_counts()
            short = {m: n - int(have.get(m, 0)) for m, n in want.items() if int(have.get(m, 0)) < n}
            if not short or p == RE.MAX_PASSES - 1:
                break
            for m, n in short.items():
                draw(m, n)
        if frames and got + len(tr) > cap:                   # whole runs only: the next would pass the cap
            break
        by_id = {s.id: s for s in sigs}
        sigs_all.update({sid: by_id[sid] for sid in tr["signal"]})
        frames.append(tr)
        got += len(tr)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["signal"])
    return sigs_all, out


def trades_frame(tr: pd.DataFrame) -> pd.DataFrame:
    """closed_trades' table -> the pool's (index = signal id; net_inr, exit_time)."""
    return pd.DataFrame({"net_inr": tr["net_inr"].astype(float).to_numpy(),
                         "exit_time": pd.to_datetime(tr["exit_time"]).to_numpy()},
                        index=pd.Index(tr["signal"].astype(str).to_numpy(), name="signal"))


def seed_for(name: str) -> int:
    return int(hashlib.sha256(f"pooled-random|{name}".encode("utf-8")).hexdigest()[:12], 16)


def code_sha256() -> str:
    here = Path(__file__).parent
    return trial_files.code_sha256([here / f for f in CODE_FILES])
