"""His Box 4 beliefs as numbers, judged the referee's way (L13b) - the same judge L12b.2 built for his news beliefs
(machine/news/beliefs.py), for any event: a level, a touch, a trade, a week.

A belief is a FAMILY (plan/work/four-boxes.md, one row, or one half of a row that says two things). Each family has one
or more VERSIONS (readings of his words, or the windows his measure names). Every version is tried on the tune years
2016-2020 (a trial file each); ONE is chosen there (the smallest luck-test number, his way or the opposite); the check
years 2021-2023 are opened ONCE for it (splits.open_check -> plan/work/ideas/<belief>/check-opened.json) and judge it
one way: the way it pointed on the tune years.

What is judged: on each EVENT a number x (a yes / no as 1 / 0, a share of a day's range, or rupees), next to its own
FAIR COMPARISON: a pool of values measured the same way where the belief says nothing should happen (fake levels at
the same distance, the same weekday and hour, the other kind of level - each belief's page says which). d = x - the
pool's average. The referee's two checks that need trades of an idea (random entries, fake levels) become one that
needs none: 200 sets of fake events, each event replaced by one of its own comparison values.

    python -m machine.beliefs rules            # the rules, as printed in every report
    python -m machine.beliefs tune             # every version on 2016-2020, one chosen per belief
    python -m machine.beliefs check            # the check years, once per belief
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, guards
from machine import trials as trial_files
from machine.data import bars as BARS
from machine.referee import prior, splits, stats, trials

KIND = "belief"
MIN_EVENTS = 30                 # fewer counted events than this can never be "holds"
MIN_MATCH = 5                   # an event needs at least 5 comparison values, else it is not counted
MIN_YEAR_EVENTS = 10            # a year with fewer counted events is shown but not judged on its own
FAKE_RUNS = 200                 # sets of fake events (like the referee's 200 random-entry runs)
FAKE_BEAT_SHARE = 0.95
WITHOUT_BEST = 3
P_LIMIT = config.P_LIMIT

HOLDS = "holds"
OPPOSITE = "the opposite holds"
FRESH = "chance — needs fresh years"
FEW = "not enough events to tell"
CHANCE = "chance"
WORDS = (HOLDS, OPPOSITE, FRESH, FEW, CHANCE)

RULES = f"""THE RULES FOR A BELIEF (L13b; fixed before any result was seen; the same for every belief)
A belief is judged on the check years {config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]} (opened once per belief), one way only: the way its numbers pointed on
the tune years {config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]}. "{HOLDS}" (his way) or "{OPPOSITE}" (the other way) needs ALL of these:
  1. The luck test: the weeks are drawn again {config.BOOTSTRAP_RESAMPLES:,} times in a world where the belief changes nothing; that
     world gives a difference this big less than {round(100 * P_LIMIT)} times in 100. (Two groups of events: the events of both
     groups in a drawn week come together.)
  2. The difference is bigger than in at least {round(100 * FAKE_BEAT_SHARE)} of 100 sets of fake events ({FAKE_RUNS} sets: each event is replaced by
     one of its own comparison values, measured against the rest of them. Two groups: the two groups' names are
     shuffled inside each stratum - the same year and side).
  3. It still points the same way without its {WITHOUT_BEST} most helpful events.
  4. At least {MIN_EVENTS} events counted.
  5. Every judged year with {MIN_YEAR_EVENTS} or more events points the same way (the worst year is named).
  6. Not already seen: the old project did not test this belief on these years (plan/work/prior-tries.csv), and no
     earlier test of this machine used these years for it.
"{FRESH}": these years were already looked at for this belief. Only new prices can judge it.
"{FEW}": fewer than {MIN_EVENTS} events counted.
"{CHANCE}": everything else. It could be luck.
Each event is compared with its OWN fair comparison (the belief's page says which), needing at least {MIN_MATCH} comparison
values (two groups: at least {MIN_MATCH} events of the other group in its stratum); an event without them is not counted."""
RULES_SHA256 = hashlib.sha256(RULES.encode("utf-8")).hexdigest()


# ===================================================================================================== a belief

@dataclass(frozen=True)
class Belief:
    idea_id: str                    # the belief, a family of readings: "belief-<name>"
    version_id: str                 # one reading of his words
    box: str                        # its row in plan/work/four-boxes.md: "B4-03"
    title: str                      # his belief, in plain words
    his_words: str
    source: str
    counted: str                    # what ONE event is, and what is counted on it
    window: str                     # how long it may take
    compare: str                    # the fair comparison, in plain words
    unit: str                       # "share" (a yes / no), "range" (a share of a day's range) or "inr" (rupees)
    rows: Callable = field(compare=False, repr=False)    # World -> the events (see measure)
    noun: str = "events"            # what an event is called in the report: "levels", "touches", "trades", "weeks"
    slices: tuple[str, ...] = ()    # columns of the rows looked at (chosen before running), never tested
    seen_key: tuple | None = None   # (blocks, timeframes, entry) in the names of plan/work/prior-tries.csv
    seen_note: str = ""             # why these years are not fresh for it (the old project, or this machine's idea)
    code: tuple[str, ...] = ()      # the files its numbers come from (their fingerprint goes into the trial file)
    groups: bool = False            # True: two groups of events (group "A" = his kind, "B" = the comparison), compared
                                    # inside each `stratum` (the same year and side) - see measure()


# ================================================================================================= the numbers

def counted(rows: pd.DataFrame) -> pd.DataFrame:
    """The events that count: a number x, and at least MIN_MATCH comparison values. The rest are counted by reason in
    .attrs["not_counted"]."""
    n0 = len(rows)
    x = pd.to_numeric(rows["x"], errors="coerce")
    pools = [None if v is None else _finite(v) for v in rows["pool"]]
    size = pd.Series([0 if v is None else len(v) for v in pools], index=rows.index)
    has_x = x.notna() & np.isfinite(x.fillna(0))
    keep = has_x & (size >= MIN_MATCH)
    out = rows[keep].copy()
    out["x"] = x[keep].astype(float)
    out["pool"] = [v for v, k in zip(pools, keep) if k]
    out["comp"] = [float(v.mean()) for v in out["pool"]]
    out["comp_n"] = [len(v) for v in out["pool"]]
    out["d"] = out["x"] - out["comp"]
    out["year"] = pd.DatetimeIndex(out.index).year
    out["week"] = stats.week_ids(out.index) if len(out) else np.array([], dtype=np.int64)
    out.attrs["not_counted"] = {"no answer (unclear, unfinished, or the years ended)": int((~has_x).sum()),
                                f"fewer than {MIN_MATCH} comparison values": int((has_x & (size < MIN_MATCH)).sum())}
    out.attrs["events"] = n0
    return out


def _finite(v) -> np.ndarray:
    """The comparison values that are numbers. A pool shared by many events (the other kind's values) is kept as the
    same array, not copied for each event."""
    a = np.asarray(v, dtype=float)
    return a if np.isfinite(a).all() else a[np.isfinite(a)]


def fake_sets(rows: pd.DataFrame, n_runs: int, seed: int) -> np.ndarray:
    """The difference (the average of d) in n_runs sets of fake events: each event replaced by one of its own
    comparison values, measured against the rest of them."""
    rng = np.random.default_rng(seed)
    if not len(rows):
        return np.zeros(n_runs)
    total = np.zeros(n_runs)
    for v in rows["pool"]:
        c = v[rng.integers(0, len(v), size=n_runs)]
        total += c - (v.sum() - c) / (len(v) - 1)
    return total / len(rows)


def counted_groups(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two groups (a belief with groups=True): the A events that count - a number x, and at least MIN_MATCH B events
    with a number in their stratum - with comp = the B events' average there; and every event used (A counted, and the
    B events of those strata), for the luck test and the fake sets."""
    n0 = int((rows["group"] == "A").sum())
    x = pd.to_numeric(rows["x"], errors="coerce")
    ok = x.notna() & np.isfinite(x.fillna(0)) & rows["group"].isin(["A", "B"])
    r = rows[ok].copy()
    r["x"] = x[ok].astype(float)
    b = r[r["group"] == "B"].groupby("stratum")["x"].agg(["mean", "size"])
    enough = b.index[b["size"] >= MIN_MATCH]
    used = r[r["stratum"].isin(enough)].copy()
    a = used[used["group"] == "A"].copy()
    a["comp"] = a["stratum"].map(b["mean"]).astype(float)
    a["comp_n"] = a["stratum"].map(b["size"]).astype(int)
    a["d"] = a["x"] - a["comp"]
    for df in (a, used):
        df["year"] = pd.DatetimeIndex(df.index).year
        df["week"] = stats.week_ids(df.index) if len(df) else np.array([], dtype=np.int64)
    no_x = n0 - int(((rows["group"] == "A") & ok).sum())
    a.attrs["not_counted"] = {"no answer (unclear, unfinished, or the years ended)": no_x,
                              f"fewer than {MIN_MATCH} comparison events in its stratum":
                                  int(((r["group"] == "A") & ~r["stratum"].isin(enough)).sum())}
    a.attrs["events"] = n0
    return a, used


def _stratified(x: np.ndarray, is_a: np.ndarray, code: np.ndarray, k: int) -> float:
    """sum over strata of n_A x (mean of A - mean of B) / sum of n_A, over strata that have both."""
    na = np.bincount(code, weights=is_a, minlength=k)
    nb = np.bincount(code, weights=~is_a, minlength=k)
    sa = np.bincount(code, weights=x * is_a, minlength=k)
    sb = np.bincount(code, weights=x * ~is_a, minlength=k)
    m = (na > 0) & (nb > 0)
    return float((na[m] * (sa[m] / na[m] - sb[m] / nb[m])).sum() / na[m].sum()) if m.any() else 0.0


def group_tests(used: pd.DataFrame, n_boot: int, n_runs: int, seed: int) -> tuple[float, float, np.ndarray]:
    """The luck test and the fake sets for two groups. Luck: the weeks are drawn again (A and B events of a week come
    together), in a world with no difference - A moved, inside each stratum, to B's average - and the stratified
    difference measured on each draw. Fake sets: the A / B labels shuffled inside each stratum. Returns (p his way,
    p the opposite way, the fake sets' differences)."""
    if not len(used) or not (used["group"] == "A").any():
        return 1.0, 1.0, np.zeros(n_runs)
    rng = np.random.default_rng(seed)
    code, strata = pd.factorize(used["stratum"])
    k = len(strata)
    is_a = (used["group"] == "A").to_numpy()
    x = used["x"].to_numpy(float)
    stat = _stratified(x, is_a, code, k)
    na = np.bincount(code, weights=is_a, minlength=k)
    nb = np.bincount(code, weights=~is_a, minlength=k)
    mean_a = np.where(na > 0, np.bincount(code, weights=x * is_a, minlength=k) / np.maximum(na, 1), 0)
    mean_b = np.where(nb > 0, np.bincount(code, weights=x * ~is_a, minlength=k) / np.maximum(nb, 1), 0)
    shift = mean_a - mean_b
    x0 = x - np.where(is_a, shift[code], 0.0)
    wk, week = np.unique(used["week"].to_numpy(), return_inverse=True)
    W = len(wk)
    cell = week * k + code
    size = W * k

    def per(weights):
        return np.bincount(cell, weights=weights, minlength=size).reshape(W, k)
    SA, SB, NA, NB = per(x0 * is_a), per(x0 * ~is_a), per(is_a.astype(float)), per((~is_a).astype(float))
    draws = rng.integers(0, W, size=(n_boot, W))
    C = np.zeros((n_boot, W))
    np.add.at(C, (np.repeat(np.arange(n_boot), W), draws.ravel()), 1.0)
    sa, sb, nA, nB = C @ SA, C @ SB, C @ NA, C @ NB
    ok = (nA > 0) & (nB > 0)
    diff = np.where(ok, sa / np.where(nA > 0, nA, 1) - sb / np.where(nB > 0, nB, 1), 0.0)
    wa = np.where(ok, nA, 0.0)
    t = (wa * diff).sum(axis=1) / np.maximum(wa.sum(axis=1), 1e-12)
    p_his = float((1 + np.count_nonzero(t >= stat - 1e-12)) / (n_boot + 1))
    p_opp = float((1 + np.count_nonzero(t <= stat + 1e-12)) / (n_boot + 1))
    fakes = np.zeros(n_runs)
    order = np.argsort(code, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(code, minlength=k))]
    lab = is_a[order]
    xs = x[order]
    cs = code[order]
    for r in range(n_runs):
        perm = lab.copy()
        for j in range(k):
            a, b = bounds[j], bounds[j + 1]
            perm[a:b] = rng.permutation(perm[a:b])
        fakes[r] = _stratified(xs, perm, cs, k)
    return p_his, p_opp, fakes


def _part(rows: pd.DataFrame) -> dict:
    n = len(rows)
    return {"n": n, "event": float(rows["x"].mean()) if n else None,
            "comparison": float(rows["comp"].mean()) if n else None, "diff": float(rows["d"].mean()) if n else None}


def _seed(b: Belief, years) -> int:
    return int(hashlib.sha256(f"{b.idea_id}|{b.version_id}|{sorted(years)}".encode()).hexdigest()[:12], 16)


def measure(b: Belief, raw: pd.DataFrame, years) -> tuple[dict, pd.DataFrame]:
    """Every number of one belief on one set of years (no answer yet), and the counted events."""
    years = tuple(sorted(int(y) for y in years))
    idx = pd.DatetimeIndex(raw.index)
    outside = sorted(set(idx.year) - set(years))
    if outside:
        raise ValueError(f"{b.idea_id} {b.version_id}: events in {outside}, outside the judged years {years}")
    seed = _seed(b, years)
    if b.groups:
        rows, used = counted_groups(raw)
        d = rows["d"].to_numpy()
        p_his, p_opp, fakes = group_tests(used, config.BOOTSTRAP_RESAMPLES, FAKE_RUNS, seed)
    else:
        rows = counted(raw)
        d, weeks = rows["d"].to_numpy(), rows["week"].to_numpy()
        p_his = stats.bootstrap_p(d, weeks, seed=seed) if len(d) else 1.0
        p_opp = stats.bootstrap_p(-d, weeks, seed=seed) if len(d) else 1.0
        fakes = fake_sets(rows, FAKE_RUNS, seed)
    stat = float(d.mean()) if len(d) else 0.0
    per_year = {y: _part(rows[rows["year"] == y]) for y in years}
    looked = {}
    for col in b.slices:
        if col in rows:
            looked[col] = {str(k): {**_part(g), "per_year": {y: _part(g[g["year"] == y]) for y in years}}
                           for k, g in rows.groupby(col, sort=True)}
    n = {
        "years": list(years), "overall": _part(rows), "stat": stat, "p_his": round(p_his, 5),
        "p_opposite": round(p_opp, 5), "events": rows.attrs["events"], "not_counted": rows.attrs["not_counted"],
        "fake": {"runs": FAKE_RUNS, "below": int((fakes < stat).sum()), "above": int((fakes > stat).sum()),
                 "middle": float(np.median(fakes)), "low": float(np.quantile(fakes, 0.025)),
                 "high": float(np.quantile(fakes, 0.975))},
        "without_best": {"his": float(np.sort(d)[:-WITHOUT_BEST].mean()) if len(d) > WITHOUT_BEST else None,
                         "opposite": float(np.sort(d)[WITHOUT_BEST:].mean()) if len(d) > WITHOUT_BEST else None},
        "per_year": per_year, "looked_at": looked, "comparison_values": int(rows["comp_n"].sum()),
        "bootstrap_seed": seed,
    }
    return n, rows


# ==================================================================================================== the judgement

@dataclass
class Check:
    number: int
    name: str
    passed: bool
    shown: str


def fmt(b: Belief, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if b.unit == "share":
        return f"{100 * v:.0f} in 100"
    if b.unit == "inr":
        return f"₹{v:,.0f} a trade"
    return f"{v:.2f} of a day's range"


def short(b: Belief, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100 * v:.0f}" if b.unit == "share" else (f"{v:,.0f}" if b.unit == "inr" else f"{v:.2f}")


def diff(b: Belief, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if b.unit == "share":
        return f"{100 * v:+.0f} in 100"
    if b.unit == "inr":
        return f"{'+' if v >= 0 else '-'}₹{abs(v):,.0f} a trade"
    return f"{v:+.2f} of a day's range"


def checks(b: Belief, n: dict, way: int, seen: bool) -> list[Check]:
    """The 6 checks, for the numbers pointing `way` (+1 his way, -1 the opposite)."""
    o, stat = n["overall"], n["stat"]
    p = n["p_his"] if way > 0 else n["p_opposite"]
    beaten = (n["fake"]["below"] if way > 0 else n["fake"]["above"]) / n["fake"]["runs"]
    wb = n["without_best"]["his" if way > 0 else "opposite"]
    judged = {y: v for y, v in n["per_year"].items() if v["n"] >= MIN_YEAR_EVENTS}
    against = [y for y, v in judged.items() if not way * v["diff"] > 0]
    worst = min(judged, key=lambda y: way * judged[y]["diff"]) if judged else None
    side = "his way" if way > 0 else "the opposite way"
    return [
        Check(1, "luck test", way * stat > 0 and p < P_LIMIT,
              f"the difference is {diff(b, stat)} ({side}); a world where it changes nothing gives this much "
              f"{100 * p:.1f} times in 100 (needs under {round(100 * P_LIMIT)})"),
        Check(2, "fake events", beaten >= FAKE_BEAT_SHARE,
              f"bigger than {100 * beaten:.0f} of 100 fake sets (needs {round(100 * FAKE_BEAT_SHARE)}); the fake sets' "
              f"middle: {diff(b, n['fake']['middle'])}"),
        Check(3, f"without the {WITHOUT_BEST} most helpful", wb is not None and way * wb > 0,
              f"{diff(b, wb)} without them (needs to stay {side})"),
        Check(4, b.noun, o["n"] >= MIN_EVENTS, f"{o['n']:,} {b.noun} counted (needs {MIN_EVENTS})"),
        Check(5, "every year", bool(judged) and not against,
              ("every judged year points " + side if not against else
               f"{', '.join(map(str, against))} point{'s' if len(against) == 1 else ''} the other way")
              + (f"; the worst year: {worst} ({diff(b, judged[worst]['diff'])}, {judged[worst]['n']} {b.noun})"
                 if worst is not None else "; no year has enough events")),
        Check(6, "not already seen", not seen,
              "these years were already looked at for this belief" if seen else "not in the old project's list"),
    ]


def decide(b: Belief, n: dict, way: int, seen: bool) -> tuple[str, str, list[Check]]:
    cs = checks(b, n, way, seen)
    side = "his way" if way > 0 else "the opposite way"
    if all(c.passed for c in cs):
        return (HOLDS if way > 0 else OPPOSITE), "all 6 checks passed", cs
    if seen:
        rest = [c for c in cs if c.number != 6 and not c.passed]
        return FRESH, ("these years were already looked at for this belief; the numbers point " + side
                       + (" and pass every other check" if not rest else
                          "; failed too: " + ", ".join(c.name for c in rest))), cs
    if n["overall"]["n"] < MIN_EVENTS:
        return FEW, f"{n['overall']['n']} {b.noun} counted", cs
    return CHANCE, "failed: " + ", ".join(c.name for c in cs if not c.passed), cs


@dataclass
class Report:
    belief: Belief
    judged_on: str
    years: tuple[int, ...]
    word: str
    why: str
    way: int                        # +1 = judged his way, -1 = the opposite way
    checks: list[Check]
    numbers: dict
    seen: prior.Seen | None
    counts: trials.Counts | None = None
    trial_path: Path | None = None
    final: bool = False

    def text(self) -> str:
        return "\n".join(_lines(self))


def seen_of(b: Belief, years) -> tuple[bool, prior.Seen | None]:
    s = prior.already_seen(prior.IdeaKey.make(*b.seen_key), years) if b.seen_key else None
    return bool(s and s.seen) or bool(b.seen_note), s


def judge(b: Belief, world, *, way: int | None = None, final: bool = False, record: bool = True,
          note: str | None = None) -> Report:
    """One belief on the world's years. way: None = the way the numbers point (the tune years); +1 / -1 = judged that
    way only (the check years: the way chosen on the tune years)."""
    years = tuple(world.years)
    hidden = sorted(set(years) & set(config.CHECK_YEARS))
    if hidden:
        guards.check_opened(b.idea_id, version=b.version_id)
        if way is None:
            raise ValueError("the check years judge ONE way: the way chosen on the tune years")
    raw = b.rows(world)
    n, rows = measure(b, raw, years)
    if way is None:
        way = 1 if n["stat"] >= 0 else -1
    is_seen, seen = seen_of(b, years)
    word, why, cs = decide(b, n, way, is_seen)
    judged_on = (f"the check years {splits.years_text(years)} (opened once for this belief)" if final else
                 f"the tune years {splits.years_text(years)} (used for choosing - not the answer)")
    counts = trials.count(b.idea_id, None, extra=(b.idea_id, b.version_id))
    rep = Report(b, judged_on, years, word, why, way, cs, n, seen, counts, None, final)
    if record:
        settings_ = {
            "judged_on": judged_on, "years": list(years), "box": b.box, "title": b.title, "his_words": b.his_words,
            "source": b.source, "counted": b.counted, "window": b.window, "compare": b.compare, "unit": b.unit,
            "way_judged": way, "rules_sha256": RULES_SHA256, "fake_runs": FAKE_RUNS,
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES, "bootstrap_seed": n["bootstrap_seed"],
            "min_match": MIN_MATCH, "slices": list(b.slices), "code_sha256": code_hash(b),
            "settings_fingerprint": world.fp, "clock": "MT5 server time",
        }
        result = {"answer": {"word": word, "why": why, "rules_sha256": RULES_SHA256, "final": final,
                             "checks": [c.__dict__ for c in cs]},
                  **n, "already_seen": {"seen": is_seen, "note": b.seen_note,
                                        "rows": [r.idea_key for r in seen.rows] if seen else []},
                  "counts": counts.as_dict(), "check_opened": splits.opened_count()}
        rep.trial_path = trial_files.record(kind=KIND, idea_id=b.idea_id, version_id=b.version_id, parent_idea=None,
                                            settings=settings_, data=_data_print(rows), result=result, note=note)
    return rep


def _worst(b: Belief, v: dict, way: int) -> str:
    years = {y: p for y, p in v.get("per_year", {}).items() if p["n"]}
    if not years:
        return ""
    y = min(years, key=lambda k: way * years[k]["diff"])
    return f"worst {y}: {diff(b, years[y]['diff'])} ({years[y]['n']})"


def _lines(r: Report) -> list[str]:
    b, n = r.belief, r.numbers
    o = n["overall"]
    out = [
        f"THE REFEREE, FOR A BELIEF · {b.box} · {b.title} · 5-minute candles · MT5 server clock",
        f"His words: \"{b.his_words}\" ({b.source})",
        f"What is counted: {b.counted}",
        f"The window: {b.window}",
        f"The fair comparison: {b.compare}",
        f"Judged on: {r.judged_on}",
        f"ANSWER: {r.word.upper()}  ({r.why})",
        "",
        f"The {b.noun}: {fmt(b, o['event'])} ({o['n']:,} {b.noun} counted) · the comparison: {fmt(b, o['comparison'])} "
        f"· difference: {diff(b, o['diff'])}",
        f"Year by year (the {b.noun} / the comparison / how many{', in 100' if b.unit == 'share' else ''}): "
        + " · ".join(f"{y}: {short(b, v['event'])} / {short(b, v['comparison'])} ({v['n']})"
                     for y, v in n["per_year"].items()),
        "",
        f"The 6 checks (judged {'his way' if r.way > 0 else 'the OPPOSITE way'}):",
        *[f"  {c.number}. {c.name}: {'yes' if c.passed else 'NO'} — {c.shown}" for c in r.checks],
    ]
    for col, groups in n["looked_at"].items():
        out += ["", f"Looked at, NOT tested - by {col.replace('_', ' ')} (chosen before running; the {b.noun} vs the "
                    f"comparison, how many, the year furthest {'below' if r.way > 0 else 'above'} it):"]
        out += [f"  {k:<28} {fmt(b, v['event']):>24} vs {fmt(b, v['comparison']):<24} ({v['n']:>5,}) "
                f"{_worst(b, v, r.way)}" for k, v in groups.items()]
    nc = "; ".join(f"{why}: {k:,}" for why, k in n["not_counted"].items() if k)
    out.append(f"\nEvents found: {n['events']:,} · counted: {o['n']:,}" + (f" · not counted - {nc}" if nc else ""))
    if b.seen_note:
        out += ["", f"Already seen: YES — {b.seen_note}. These years are not fresh for this belief."]
    elif r.seen is not None:
        out += ["", r.seen.text()]
    if r.counts is not None:
        out += ["", r.counts.text(), f"CHECK years opened {splits.opened_count()} times in the project"]
    out += ["", RULES]
    if r.trial_path is not None:
        out.append(f"\nTrial file: {_rel(r.trial_path)}")
    return out


# =========================================================================================== tune, choose, check

CHOSEN_FILE = "chosen.json"


def chosen_path() -> Path:
    return Path(config.BELIEFS_DIR) / CHOSEN_FILE


def read_chosen() -> dict:
    p = chosen_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def clear_of_fakes(n: dict) -> float:
    """How far the difference stands from the fake sets' middle, in the fake sets' own spread (the tie-break)."""
    spread = (n["fake"]["high"] - n["fake"]["low"]) / 3.92
    return abs(n["stat"] - n["fake"]["middle"]) / spread if spread > 0 else 0.0


def choose(reps: list[Report]) -> Report:
    """ONE version per belief on the tune years: the smallest luck-test number, his way or the opposite; a tie ->
    the one standing furthest from its fake sets."""
    return min(reps, key=lambda r: (min(r.numbers["p_his"], r.numbers["p_opposite"]), -clear_of_fakes(r.numbers)))


def tune(families: dict[str, list[Belief]], world, record: bool = True) -> dict[str, list[Report]]:
    """Every version of these beliefs on the tune years; one chosen per belief (written to plan/work/beliefs/
    chosen.json). A belief that already has its chosen version is refused: the tune years chose once."""
    if set(world.years) - set(config.TUNE_YEARS):
        raise ValueError(f"the tune years are {config.TUNE_YEARS}, not {world.years}")
    chosen = read_chosen()
    done = [f for f in families if f in chosen]
    if done:
        raise ValueError(f"{done} already have a chosen version: the tune years chose once. A new reading is a new "
                         "version - judge it inside 2016-2020 (walk-forward).")
    out = {}
    for fam, versions in families.items():
        reps = [judge(b, world, record=record) for b in versions]
        chosen[fam] = _choice(reps)
        out[fam] = reps
        if record:
            save_page(fam, reps, "tune")
            _write_json(chosen_path(), chosen)
    return out


def _choice(reps: list[Report]) -> dict:
    best = choose(reps)
    return {"version_id": best.belief.version_id, "way": best.way, "box": best.belief.box,
            "why": f"the smallest luck-test number on 2016-2020 among {len(reps)} version(s), his way or the opposite",
            "tune": {r.belief.version_id: {"word": r.word, "p_his": r.numbers["p_his"],
                                           "p_opposite": r.numbers["p_opposite"],
                                           "diff": r.numbers["stat"], "n": r.numbers["overall"]["n"],
                                           "clear_of_fake_sets": round(clear_of_fakes(r.numbers), 2),
                                           "trial": _rel(r.trial_path) if r.trial_path else None}
                     for r in reps}}


def retune(families: dict[str, list[Belief]], world, why: str, record: bool = True) -> dict[str, list[Report]]:
    """After a FIX to how a belief's events are read - never a new reading, never a new version: the same versions
    judged again on the tune years and chosen again by the same rule. chosen.json keeps the first choice under
    "before_fix" with the reason; every trial file of both runs stays in plan/work/trials/."""
    if set(world.years) - set(config.TUNE_YEARS):
        raise ValueError(f"the tune years are {config.TUNE_YEARS}, not {world.years}")
    if not str(why).strip():
        raise ValueError("a run again after a fix needs its reason (it goes into chosen.json and the trial files)")
    chosen = read_chosen()
    missing = [f for f in families if f not in chosen]
    if missing:
        raise ValueError(f"{missing} have no first choice: that is `tune`, not a run again")
    out = {}
    for fam, versions in families.items():
        known = {v.version_id for v in versions}
        if set(chosen[fam]["tune"]) != known:
            raise ValueError(f"{fam}: the versions changed ({sorted(chosen[fam]['tune'])} -> {sorted(known)}) - a new "
                             "reading is a new version, judged by walk-forward, not a run again after a fix")
        reps = [judge(b, world, record=record, note=f"the tune years again after a fix: {why}") for b in versions]
        first = {k: v for k, v in chosen[fam].items() if k not in ("before_fix", "fix")}
        chosen[fam] = {**_choice(reps), "fix": why, "before_fix": first}
        out[fam] = reps
        if record:
            save_page(fam, reps, "tune", note=f"Run again after a fix (the first run's choice is in chosen.json, "
                                              f"\"before_fix\"): {why}")
            _write_json(chosen_path(), chosen)
    return out


def open_all(families: dict[str, list[Belief]], who: str) -> None:
    """Open the check years once for each belief's chosen version (a belief already opened is left as it is)."""
    chosen = read_chosen()
    missing = [f for f in families if f not in chosen]
    if missing:
        raise ValueError(f"no version chosen yet for {missing}: run `python -m machine.beliefs tune` first")
    for fam in families:
        if splits.family_opening(fam) is None:
            c = chosen[fam]
            splits.open_check(fam, c["version_id"], who=who,
                              why=f"chosen on 2016-2020: {c['why']} (judged {'his way' if c['way'] > 0 else 'the opposite way'})")


def check(families: dict[str, list[Belief]], world, record: bool = True, note: str | None = None
          ) -> dict[str, Report]:
    """The check years, once per belief, for its chosen version, judged the chosen way. open_all() first. note: a run
    again after a fix says so (in the trial file and on the page); the check years stay opened once, for this same
    version (splits.open_check refuses a second opening)."""
    chosen = read_chosen()
    out = {}
    for fam, versions in families.items():
        c = chosen[fam]
        b = next(v for v in versions if v.version_id == c["version_id"])
        out[fam] = judge(b, world, way=int(c["way"]), final=True, record=record, note=note)
        if record:
            save_page(fam, [out[fam]], "check", note=note)
    return out


def save_page(fam: str, reps: list[Report], stage: str, note: str | None = None) -> Path:
    """plan/work/beliefs/<belief>-<stage>.md: every report of this run, as printed."""
    path = Path(config.BELIEFS_DIR) / f"{fam.removeprefix('belief-')}-{stage}.md"
    title = {"tune": "the tune years 2016-2020 (for choosing - not the answer)",
             "check": "the check years 2021-2023 (opened once - the answer)"}[stage]
    body = [f"# {reps[0].belief.box} {fam} — {title}", "",
            f"Made by `python -m machine.beliefs {stage}` (L13b). The belief's own page: "
            f"`{fam.removeprefix('belief-')}.md`.", ""]
    if note:
        body += [f"**{note}**", ""]
    for r in reps:
        body += [f"## {r.belief.version_id}", "", "```text", r.text(), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(body), encoding="utf-8")
    tmp.replace(path)
    return path


# ============================================================================================================ helpers

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def code_hash(b: Belief) -> str:
    here = Path(__file__).parent
    files = [here / "engine.py", here / "world.py", here.parent / "referee" / "stats.py",
             *(config.ROOT / f for f in b.code)]
    return trial_files.code_sha256(sorted(set(files)))


def _data_print(rows: pd.DataFrame) -> dict:
    out = {"events_sha256": hashlib.sha256(
        pd.util.hash_pandas_object(rows[["x", "comp"]], index=True).to_numpy().tobytes()).hexdigest()
        if len(rows) else None}
    try:
        out["gold_m1_content_sha256"] = BARS.m1_fingerprint()
    except (OSError, ValueError):
        out["prices"] = "bars given by hand"
    return out


def _rel(path: Path) -> str:
    try:
        return Path(path).relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
