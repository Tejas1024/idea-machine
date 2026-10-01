"""His news beliefs as numbers, judged the referee's way (L12b.2).

    python -m machine.news.beliefs tune     # every version on 2016-2020 (a trial file each); ONE version per belief chosen
    python -m machine.news.beliefs check    # opens 2021-2023 ONCE per belief, for the chosen version: the answer
    python -m machine.news.beliefs rules    # the rules, as printed in every report
Each step also saves its reports as plan/work/beliefs/<belief>-tune.md / -check.md.

A belief is not a trade: nothing is bought, so there are no rupees here. What is judged is a COUNT on each release
(did the first side come back? how far did gold move?) against THE FAIR COMPARISON: the same New York clock minute on
the same weekday in the same year, on days with no release within 60 minutes (machine/news/reaction.py controls). A
belief about good or bad news has no such comparison (a quiet day has no news to be good or bad): it is judged
against a coin.

The referee's pieces are used as they are (machine/referee/): the tune years 2016-2020 choose, the check years
2021-2023 open ONCE per belief for ONE version (splits.open_check -> plan/work/ideas/<belief>/check-opened.json), the
weekly luck test (stats.bootstrap_p), "already seen" (prior.already_seen on plan/work/prior-tries.csv), one trial file
per run (kind "belief" - counted as tries, never as an idea's answer). The referee's two checks that need trades
(random entries, fake levels) become ONE check that needs none: fake news moments drawn from the comparison days.

THE BELIEFS (his words, what is counted) are in BELIEFS below; all of it was written before any reaction was looked at
(plan/work/L12b2-notes.md, "fixed before looking").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
MIN_EVENTS = 30                 # fewer counted releases than this can never be "holds"
MIN_MATCH = 5                   # a release needs at least 5 comparison moments (same weekday; else any weekday)
MIN_YEAR_EVENTS = 10            # a year with fewer counted releases is shown but not judged on its own
FAKE_RUNS = 200                 # fake sets of news moments (like the referee's 200 random-entry runs)
FAKE_BEAT_SHARE = 0.95
WITHOUT_BEST = 3
P_LIMIT = config.P_LIMIT
LOOKED_MIN = 20                 # a report is printed in the "looked at" table from 20 counted releases

HOLDS = "holds"
OPPOSITE = "the opposite holds"
FRESH = "chance — needs fresh years"
FEW = "not enough events to tell"
CHANCE = "chance"
WORDS = (HOLDS, OPPOSITE, FRESH, FEW, CHANCE)

RULES = f"""THE RULES FOR A BELIEF (fixed before any result was seen; the same for every belief)
A belief is judged on the check years {config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]} (opened once per belief), one way only: the way its numbers pointed on
the tune years {config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]}. "{HOLDS}" (his way) or "{OPPOSITE}" (the other way) needs ALL of these:
  1. The luck test: the weeks are drawn again {config.BOOTSTRAP_RESAMPLES:,} times in a world where news changes nothing; that world
     gives a difference this big less than {round(100 * P_LIMIT)} times in 100.
  2. The difference is bigger than in at least {round(100 * FAKE_BEAT_SHARE)} of 100 sets of fake news moments ({FAKE_RUNS} sets: for each release
     one comparison moment is drawn in its place; for a belief judged against a coin, the news is called good or bad
     by a coin).
  3. It still points the same way without its {WITHOUT_BEST} most helpful releases.
  4. At least {MIN_EVENTS} releases counted.
  5. Every judged year with {MIN_YEAR_EVENTS} or more releases points the same way (the worst year is named).
  6. The old project did not already test this belief on these years (plan/work/prior-tries.csv).
"{FRESH}": the old project already looked at these years for this belief. Only new prices can judge it.
"{FEW}": fewer than {MIN_EVENTS} releases counted.
"{CHANCE}": everything else. It could be luck.
The fair comparison: the same New York clock minute on the same weekday in the same year, on days with no release
within 60 minutes (at least {MIN_MATCH} such moments; else any weekday at that clock minute; else the release is not counted)."""
RULES_SHA256 = hashlib.sha256(RULES.encode("utf-8")).hexdigest()


# ===================================================================================================== the beliefs

@dataclass(frozen=True)
class Belief:
    idea_id: str                    # the belief (a family of readings)
    version_id: str                 # one reading of his words
    title: str
    his_words: str
    source: str
    counted: str                    # what is counted on each release, in plain words
    unit: str                       # "share" (a yes/no per release) or "range" (a share of a day's range)
    compare: str                    # "same clock" (the fair comparison) or "coin"
    value: Callable[[pd.DataFrame], pd.Series] = field(compare=False, repr=False)   # NaN = not counted
    flips: bool = False             # coin beliefs: a fake set flips each release's yes/no with a coin
    seen_key: tuple | None = None   # (blocks, timeframes, entry) in the names of plan/work/prior-tries.csv
    seen_note: str = ""


def _his(df: pd.DataFrame) -> pd.Series:
    return df["his_news"] & df["has_bars"].fillna(False) if "his_news" in df else df["has_bars"].fillna(False)


def _race(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    def value(df: pd.DataFrame) -> pd.Series:
        x = df[col].map({"back": 1.0, "on": 0.0})
        return x.where(_his(df))
    return value


def _range(df: pd.DataFrame) -> pd.Series:
    return df["range_60_adr"].where(_his(df))


def _way(minutes: int) -> Callable[[pd.DataFrame], pd.Series]:
    def value(df: pd.DataFrame) -> pd.Series:
        move = np.sign(df[f"move_{minutes}"])
        x = pd.Series(np.where(move == 0, 0.5, (move == df["gold_way"]).astype(float)), index=df.index)
        return x.where(_his(df) & (df["gold_way"] != 0))
    return value


def _against(df: pd.DataFrame) -> pd.Series:
    side = df["hr_side"].map({"low": 1.0, "high": -1.0})       # +1: the LOW of the hour before broke first
    x = (side == df["gold_way"]).astype(float).where(side.notna())
    return x.where(_his(df) & (df["gold_way"] != 0))


_TRAP_WORDS = "News trap: positive → sweep low first then up; false → sweep high first then down."
_TRAP_SRC = "strategy.md section 1, line 73 (four-boxes B4-07)"
_TRAP_SEEN = ("the old project measured this belief on 117 jobs reports and Fed decisions of 2018-2023 "
              "(strategy-understanding section 48f): the first side went ON 68 times in 100")

BELIEFS: tuple[Belief, ...] = (
    Belief("belief-news-moves-gold", "red-and-orange",
           "Forex Factory red and orange news moves gold",
           "I used to watch for orange and red color signals of the news, so those are the high impact.",
           "LEARNED #141, #142 (T7); four-boxes B3-29; plan/work/news.md NQ1",
           "how far gold travelled in the first hour after the release (its highest price minus its lowest), as a "
           "share of a day's range",
           "range", "same clock", _range),
    Belief("belief-news-direction", "first-15-minutes",
           "News that is bad for the dollar sends gold up (and good for the dollar, down)",
           "positive → ... then up; false → ... then down",
           "strategy.md section 1, line 73; plan/work/news.md NQ2 (positive = good for gold; which numbers are good "
           "for gold is measured)",
           "releases with a BIG surprise (bigger than that number's usual miss): is gold, 15 minutes later, on the "
           "side the surprise says (up when it is bad for the dollar)?",
           "share", "coin", _way(15), flips=True),
    Belief("belief-news-trap", "hour-range",
           "News trap: the first side taken after news is the wrong side (the hour before is the range)",
           _TRAP_WORDS, _TRAP_SRC,
           "the high and low of the hour before the release are the range; the first side price breaks in the hour "
           "after is the first side; from that edge, does price come BACK to the far side of the range before it "
           "goes the same distance further on?",
           "share", "same clock", _race("hr_race"),
           seen_key=("news_event+sessions", "1h", "market:break"), seen_note=_TRAP_SEEN),
    Belief("belief-news-trap", "his-levels",
           "News trap: the first of HIS levels taken after news is the wrong side",
           _TRAP_WORDS, _TRAP_SRC,
           "his nearest liquidity level above and below the price at the release (session highs / lows, equal highs "
           "/ lows, trendlines, day and week swings); the one a wick takes first in the hour after is the first "
           "side; from that level, does price come BACK by the size of the hour-before range before it goes that "
           "far further on?",
           "share", "same clock", _race("lv_race"),
           seen_key=("news_event+sessions", "1h", "market:break"), seen_note=_TRAP_SEEN),
    Belief("belief-news-trap", "good-or-bad-news",
           "News trap: good news for gold takes the LOW first, bad news takes the HIGH first",
           _TRAP_WORDS, _TRAP_SRC,
           "releases with a BIG surprise: is the first side of the hour-before range that breaks the side AGAINST "
           "the news (the low when the news is good for gold, the high when it is bad)?",
           "share", "coin", _against, flips=True,
           seen_key=("news_event+sessions", "1h", "market:break"), seen_note=_TRAP_SEEN),
)
# the colours are chosen before running (2 slices per belief: likhith.md section D)
SLICES = (("red", lambda df: df["colour"] == "red"), ("orange", lambda df: df["colour"] == "orange"))
FAMILIES = tuple(dict.fromkeys(b.idea_id for b in BELIEFS))
CHOSEN_FILE = "news-chosen.json"


def versions(idea_id: str) -> list[Belief]:
    return [b for b in BELIEFS if b.idea_id == idea_id]


def get(idea_id: str, version_id: str) -> Belief:
    for b in BELIEFS:
        if (b.idea_id, b.version_id) == (idea_id, version_id):
            return b
    raise KeyError(f"no belief {idea_id} version {version_id}")


# ================================================================================================= the comparison

def compare(b: Belief, ev: pd.DataFrame, ct: pd.DataFrame | None) -> pd.DataFrame:
    """One row per COUNTED release: x (its value), comp (its comparison), d = x - comp, the week, and for the fake
    sets the comparison moments' values (`pool`). Releases with no fair comparison are left out (their count is in
    .attrs["no_comparison"])."""
    x = b.value(ev)
    rows = ev.loc[x.notna(), ["year", "ny_clock", "weekday", "colour", "his_reports"]].copy()
    rows["x"] = x[x.notna()]
    if b.compare == "coin":
        rows["comp"], rows["comp_n"], rows["match"], rows["pool"] = 0.5, 0, "coin", None
        dropped = 0
    else:
        c = ct.assign(x=b.value(ct.assign(his_news=True))).dropna(subset=["x"])
        by_wd = {k: g.to_numpy() for k, g in c.groupby(["year", "ny_clock", "weekday"])["x"]}
        by_clock = {k: g.to_numpy() for k, g in c.groupby(["year", "ny_clock"])["x"]}
        found = []
        for y, clk, wd in zip(rows["year"], rows["ny_clock"], rows["weekday"]):
            v, m = by_wd.get((y, clk, wd)), "same weekday"
            if v is None or len(v) < MIN_MATCH:
                v, m = by_clock.get((y, clk)), "any weekday"
            ok = v is not None and len(v) >= MIN_MATCH
            found.append((float(v.mean()), len(v), m, v) if ok else (np.nan, 0, "none", None))
        rows["comp"] = [f[0] for f in found]
        rows["comp_n"] = [f[1] for f in found]
        rows["match"] = [f[2] for f in found]
        rows["pool"] = pd.Series([f[3] for f in found], index=rows.index, dtype=object)
        dropped = int(rows["comp"].isna().sum())
        rows = rows[rows["comp"].notna()]
    rows["d"] = rows["x"] - rows["comp"]
    rows["week"] = stats.week_ids(rows.index) if len(rows) else np.array([], dtype=np.int64)
    rows.attrs["no_comparison"] = dropped
    return rows


def fake_sets(rows: pd.DataFrame, b: Belief, n_runs: int, seed: int) -> np.ndarray:
    """The difference (the average of d) in n_runs sets of fake news moments. Same clock: each release is replaced by
    one of its own comparison moments, measured against the rest of them. Coin: each release's yes / no is flipped
    by a coin (the news called good or bad at random)."""
    rng = np.random.default_rng(seed)
    if not len(rows):
        return np.zeros(n_runs)
    if b.flips:
        x = rows["x"].to_numpy()
        flip = rng.integers(0, 2, size=(n_runs, len(x))).astype(bool)
        return np.where(flip, 1.0 - x[None, :], x[None, :]).mean(axis=1) - 0.5
    total = np.zeros(n_runs)
    for v in rows["pool"]:
        c = rng.choice(v, size=n_runs)
        total += c - (v.sum() - c) / (len(v) - 1)
    return total / len(rows)


# ==================================================================================================== the judgement

@dataclass
class Check:
    number: int
    name: str
    passed: bool
    shown: str


@dataclass
class Report:
    belief: Belief
    judged_on: str
    years: tuple[int, ...]
    word: str
    why: str
    way: int                        # +1 = the numbers were judged his way, -1 = the opposite way
    checks: list[Check]
    numbers: dict
    seen: prior.Seen | None
    counts: trials.Counts | None = None
    trial_path: Path | None = None
    final: bool = False

    def text(self) -> str:
        return "\n".join(_lines(self))


def _seed(b: Belief, years) -> int:
    return int(hashlib.sha256(f"{b.idea_id}|{b.version_id}|{sorted(years)}".encode()).hexdigest()[:12], 16)


def _fmt(b: Belief, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100 * v:.0f} in 100" if b.unit == "share" else f"{v:.2f} of a day's range"


def _short(b: Belief, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100 * v:.0f}" if b.unit == "share" else f"{v:.2f}"


def _diff(b: Belief, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100 * v:+.0f} in 100" if b.unit == "share" else f"{v:+.2f} of a day's range"


def _worst(b: Belief, v: dict, way: int) -> str:
    """The year of one report that points least the judged way (years with any counted release)."""
    years = {y: p for y, p in v.get("per_year", {}).items() if p["n"]}
    if not years:
        return ""
    y = min(years, key=lambda k: way * years[k]["diff"])
    return f"worst {y}: {_diff(b, years[y]['diff'])} ({years[y]['n']})"


def _part(rows: pd.DataFrame) -> dict:
    n = len(rows)
    return {"n": n, "event": float(rows["x"].mean()) if n else None, "comparison": float(rows["comp"].mean()) if n else None,
            "diff": float(rows["d"].mean()) if n else None}


def measure(b: Belief, ev: pd.DataFrame, ct: pd.DataFrame | None, years) -> dict:
    """Every number of one belief on one set of years (no answer yet)."""
    years = tuple(sorted(int(y) for y in years))
    ev = ev[ev["year"].isin(years)]
    ct = None if ct is None else ct[ct["year"].isin(years)]
    rows = compare(b, ev, ct)
    seed = _seed(b, years)
    d, weeks = rows["d"].to_numpy(), rows["week"].to_numpy()
    p_his = stats.bootstrap_p(d, weeks, seed=seed) if len(d) else 1.0
    p_opp = stats.bootstrap_p(-d, weeks, seed=seed) if len(d) else 1.0
    fakes = fake_sets(rows, b, FAKE_RUNS, seed)
    stat = float(d.mean()) if len(d) else 0.0
    per_year = {int(y): _part(rows[rows["year"] == y]) for y in years}
    slices = {}
    for name, mask in SLICES:
        r = rows[mask(rows)]
        rd, rw = r["d"].to_numpy(), r["week"].to_numpy()
        slices[name] = {**_part(r), "p_his": stats.bootstrap_p(rd, rw, seed=seed) if len(rd) > 1 else 1.0,
                        "p_opposite": stats.bootstrap_p(-rd, rw, seed=seed) if len(rd) > 1 else 1.0}
    looked = {}
    e = rows.assign(report=rows["his_reports"].str.split(" | ", regex=False)).explode("report")
    for rep, g in e.groupby("report"):
        if len(g) >= LOOKED_MIN:
            looked[rep] = {**_part(g), "per_year": {int(y): _part(g[g["year"] == y]) for y in years}}
    old = e[e["report"].isin(("Jobs report", "FOMC decision"))]
    old = old[~old.index.duplicated()]
    his = int(_his(ev).sum())
    return {
        "years": list(years), "overall": _part(rows), "stat": stat, "p_his": round(p_his, 5),
        "p_opposite": round(p_opp, 5),
        "his_releases": his, "not_counted": his - len(rows) - rows.attrs["no_comparison"],
        "fake": {"runs": FAKE_RUNS, "below": int((fakes < stat).sum()), "above": int((fakes > stat).sum()),
                 "middle": float(np.median(fakes)), "low": float(np.quantile(fakes, 0.025)),
                 "high": float(np.quantile(fakes, 0.975))},
        "without_best": {"his": float(np.sort(d)[:-WITHOUT_BEST].mean()) if len(d) > WITHOUT_BEST else None,
                         "opposite": float(np.sort(d)[WITHOUT_BEST:].mean()) if len(d) > WITHOUT_BEST else None},
        "per_year": per_year, "slices": slices, "looked_at": looked,
        "old_project_events": _part(old),
        "match": rows["match"].value_counts().to_dict(), "no_comparison": rows.attrs["no_comparison"],
        "bootstrap_seed": seed,
    }


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
              f"the difference is {_diff(b, stat)} ({side}); a world where news changes nothing gives this much "
              f"{100 * p:.1f} times in 100 (needs under {round(100 * P_LIMIT)})"),
        Check(2, "fake news moments", beaten >= FAKE_BEAT_SHARE,
              f"bigger than {100 * beaten:.0f} of 100 fake sets (needs {round(100 * FAKE_BEAT_SHARE)}); the fake sets' "
              f"middle: {_diff(b, n['fake']['middle'])}"),
        Check(3, f"without the {WITHOUT_BEST} most helpful", wb is not None and way * wb > 0,
              f"{_diff(b, wb)} without them (needs to stay {side})"),
        Check(4, "releases", o["n"] >= MIN_EVENTS, f"{o['n']:,} releases counted (needs {MIN_EVENTS})"),
        Check(5, "every year", bool(judged) and not against,
              ("every judged year points " + side if not against else
               f"{', '.join(map(str, against))} point{'s' if len(against) == 1 else ''} the other way")
              + (f"; the worst year: {worst} ({_diff(b, judged[worst]['diff'])}, {judged[worst]['n']} releases)"
                 if worst is not None else "; no year has enough releases")),
        Check(6, "not already seen", not seen,
              "the old project already tested this belief on these years" if seen else "not in the old project's list"),
    ]


def decide(b: Belief, n: dict, way: int, seen: bool) -> tuple[str, str, list[Check]]:
    cs = checks(b, n, way, seen)
    side = "his way" if way > 0 else "the opposite way"
    if all(c.passed for c in cs):
        return (HOLDS if way > 0 else OPPOSITE), "all 6 checks passed", cs
    if seen:
        rest = [c for c in cs if c.number != 6 and not c.passed]
        return FRESH, ("the old project already looked at these years for this belief; the numbers point " + side
                       + (" and pass every other check" if not rest else
                          "; failed too: " + ", ".join(c.name for c in rest))), cs
    if n["overall"]["n"] < MIN_EVENTS:
        return FEW, f"{n['overall']['n']} releases counted", cs
    return CHANCE, "failed: " + ", ".join(c.name for c in cs if not c.passed), cs


def judge(b: Belief, ev: pd.DataFrame, ct: pd.DataFrame | None, years, *, way: int | None = None,
          final: bool = False, record: bool = True, note: str | None = None) -> Report:
    """One belief on one set of years. way: None = the way the numbers point on these years (the tune years);
    +1 / -1 = judged that way only (the check years: the way chosen on the tune years)."""
    years = tuple(sorted(int(y) for y in years))
    hidden = sorted(set(years) & set(config.CHECK_YEARS))
    if hidden:
        guards.check_opened(b.idea_id, [f"{hidden[0]}-01-01"], version=b.version_id)
        if way is None:
            raise ValueError("the check years judge ONE way: the way chosen on the tune years")
    n = measure(b, ev, ct, years)
    if way is None:
        way = 1 if n["stat"] >= 0 else -1
    seen = prior.already_seen(prior.IdeaKey.make(*b.seen_key), years) if b.seen_key else None
    is_seen = bool(seen and seen.seen) or bool(b.seen_note)
    word, why, cs = decide(b, n, way, is_seen)
    judged_on = (f"the check years {splits.years_text(years)} (opened once for this version)" if final else
                 f"the tune years {splits.years_text(years)} (used for choosing - not the answer)")
    counts = trials.count(b.idea_id, None, extra=(b.idea_id, b.version_id))
    rep = Report(b, judged_on, years, word, why, way, cs, n, seen, counts, None, final)
    if record:
        settings_ = {
            "judged_on": judged_on, "years": list(years), "title": b.title, "his_words": b.his_words,
            "source": b.source, "counted": b.counted, "compare": b.compare, "unit": b.unit, "way_judged": way,
            "rules_sha256": RULES_SHA256, "fake_runs": FAKE_RUNS, "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": n["bootstrap_seed"], "min_match": MIN_MATCH, "slices": [s for s, _ in SLICES],
            "his_news": "Forex Factory red + orange (T7)", "clock": "MT5 server time; the comparison by New York's",
            "code_sha256": _code_hash(), "calendar": _calendar_fingerprint(),
        }
        result = {"answer": {"word": word, "why": why, "rules_sha256": RULES_SHA256, "final": final,
                             "checks": [c.__dict__ for c in cs]},
                  **n, "already_seen": {"seen": is_seen, "note": b.seen_note,
                                        "rows": [r.idea_key for r in seen.rows] if seen else []},
                  "counts": counts.as_dict(), "check_opened": splits.opened_count()}
        rep.trial_path = trial_files.record(kind=KIND, idea_id=b.idea_id, version_id=b.version_id, parent_idea=None,
                                            settings=settings_, data=_data_print(ev, ct), result=result, note=note)
    return rep


def _lines(r: Report) -> list[str]:
    b, n = r.belief, r.numbers
    o = n["overall"]
    out = [
        f"THE REFEREE, FOR A BELIEF · {b.title} · 1-minute bars · MT5 server clock",
        f"His words: \"{b.his_words}\" ({b.source})",
        f"What is counted: {b.counted}",
        ("The fair comparison: the same New York clock minute on the same weekday in the same year, on days with no "
         "release within 60 minutes." if b.compare == "same clock" else
         "The comparison: a coin (50 in 100) - a day with no news has no news to be good or bad."),
        f"His news = Forex Factory red + orange. Judged on: {r.judged_on}",
        f"ANSWER: {r.word.upper()}  ({r.why})",
        "",
        f"After his news: {_fmt(b, o['event'])} ({o['n']:,} releases counted) · the comparison: {_fmt(b, o['comparison'])} "
        f"· difference: {_diff(b, o['diff'])}",
        f"Year by year (after news / the comparison / releases{', in 100' if b.unit == 'share' else ''}): "
        + " · ".join(f"{y}: {_short(b, v['event'])} / {_short(b, v['comparison'])} ({v['n']})"
                     for y, v in n["per_year"].items()),
        "",
        f"The 6 checks (judged {'his way' if r.way > 0 else 'the OPPOSITE way'}):",
        *[f"  {c.number}. {c.name}: {'yes' if c.passed else 'NO'} — {c.shown}" for c in r.checks],
        "",
        "By colour (chosen before running): "
        + " · ".join(f"{k}: {_fmt(b, v['event'])} vs {_fmt(b, v['comparison'])} ({v['n']:,} releases; luck test "
                     f"{100 * (v['p_his'] if r.way > 0 else v['p_opposite']):.1f} in 100)"
                     for k, v in n["slices"].items()),
        f"The old project's events (jobs reports + Fed decisions): {_fmt(b, n['old_project_events']['event'])} vs "
        f"{_fmt(b, n['old_project_events']['comparison'])} ({n['old_project_events']['n']} releases)",
        f"By report ({len(n['looked_at'])} looked at, NOT tested - a report with {LOOKED_MIN} or more counted releases; "
        f"after news vs the comparison, the releases, and the year furthest {'below' if r.way > 0 else 'above'} it):",
        *[f"  {rep:<24} {_fmt(b, v['event']):>26} vs {_fmt(b, v['comparison']):<26} ({v['n']:>4}) {_worst(b, v, r.way)}"
          for rep, v in sorted(n["looked_at"].items(), key=lambda kv: -abs(kv[1]["diff"]))],
        f"His releases with prices: {n['his_releases']:,} · counted: {o['n']:,} · not counted (what is counted did not "
        f"happen, e.g. no side broke, or no big surprise): {n['not_counted']:,}",
    ]
    if b.compare == "same clock":
        m = n["match"]
        out.append(f"Comparison found on the same weekday for {m.get('same weekday', 0):,} releases, on any weekday for "
                   f"{m.get('any weekday', 0):,}; none for {n['no_comparison']:,} (not counted).")
    if b.seen_note:
        out += ["", f"Already seen: YES — {b.seen_note}. These years are not fresh for this belief."]
    if r.counts is not None:
        out += ["", r.counts.text(), f"CHECK years opened {splits.opened_count()} times in the project"]
    out += ["", RULES]
    if r.trial_path is not None:
        out.append(f"\nTrial file: {_rel(r.trial_path)}")
    return out


# =========================================================================================== tune, choose, check

def chosen_path() -> Path:
    return Path(config.BELIEFS_DIR) / CHOSEN_FILE


def read_chosen() -> dict:
    p = chosen_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def tune(record: bool = True) -> dict[str, list[Report]]:
    """Every version on the tune years; per belief ONE version is chosen: the smallest luck-test number, either way
    (his way or the opposite). The choice and its way are written to plan/work/beliefs/news-chosen.json."""
    from machine.news import reaction
    ev, ct = reaction.load("events"), reaction.load("controls")
    out, chosen = {}, read_chosen()
    for fam in FAMILIES:
        if fam in chosen:
            raise ValueError(f"{fam} already has its chosen version ({chosen[fam]['version_id']}): the tune years chose "
                             "once. A new reading is a new version - add it and judge it inside 2016-2020.")
        reps = [judge(b, ev, ct, config.TUNE_YEARS, record=record) for b in versions(fam)]
        best = min(reps, key=lambda r: (min(r.numbers["p_his"], r.numbers["p_opposite"]), -_clear(r.numbers)))
        chosen[fam] = {"version_id": best.belief.version_id, "way": best.way,
                       "why": "the smallest luck-test number on 2016-2020 among "
                              f"{len(reps)} version(s), his way or the opposite",
                       "tune": {r.belief.version_id: {"p_his": r.numbers["p_his"], "p_opposite": r.numbers["p_opposite"],
                                                      "diff": r.numbers["stat"], "n": r.numbers["overall"]["n"],
                                                      "clear_of_fake_sets": round(_clear(r.numbers), 2),
                                                      "trial": _rel(r.trial_path) if r.trial_path else None}
                                for r in reps}}
        out[fam] = reps
        if record:
            save_page(fam, reps, "tune")
    if record:
        _write_json(chosen_path(), chosen)
    return out


def _clear(n: dict) -> float:
    """How far the difference stands from the fake sets' middle, in the fake sets' own spread (the tie-break when two
    versions both sit at the luck test's smallest number)."""
    spread = (n["fake"]["high"] - n["fake"]["low"]) / 3.92
    return abs(n["stat"] - n["fake"]["middle"]) / spread if spread > 0 else 0.0


def save_page(fam: str, reps: list[Report], stage: str) -> Path:
    """plan/work/beliefs/<belief>-<stage>.md: every report of this run, as printed."""
    path = Path(config.BELIEFS_DIR) / f"{fam.removeprefix('belief-')}-{stage}.md"
    title = {"tune": "the tune years 2016-2020 (for choosing - not the answer)",
             "check": "the check years 2021-2023 (opened once - the answer)"}[stage]
    body = [f"# {fam} — {title}", "", "Made by `python -m machine.news.beliefs " + stage + "` (L12b.2). "
            "No rupees: a belief is a count, nothing is bought.", ""]
    for r in reps:
        body += [f"## {r.belief.version_id}", "", "```text", r.text(), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(body), encoding="utf-8")
    tmp.replace(path)
    return path


def require_open() -> None:
    """Refuse the check years of the reactions until every news belief has opened them (once each)."""
    closed = [fam for fam in FAMILIES if splits.family_opening(fam) is None]
    if closed:
        raise guards.CheckYearsClosedError(
            f"The reactions of {config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]} stay hidden until every news belief has "
            f"opened its check years (python -m machine.news.beliefs check). Still closed: {', '.join(closed)}.")


def check(who: str, record: bool = True) -> dict[str, Report]:
    """Open the check years ONCE per belief for its chosen version, measure 2021-2023, and give the answer."""
    from machine.news import reaction
    chosen = read_chosen()
    missing = [f for f in FAMILIES if f not in chosen]
    if missing:
        raise ValueError(f"no version chosen yet for {missing}: run `python -m machine.news.beliefs tune` first")
    for fam in FAMILIES:
        if splits.family_opening(fam) is None:
            c = chosen[fam]
            splits.open_check(fam, c["version_id"], who=who,
                              why=f"chosen on 2016-2020: {c['why']} (judged {'his way' if c['way'] > 0 else 'the opposite way'})")
    reaction.build(config.TUNE_YEARS + config.CHECK_YEARS)
    out = {}
    for fam in FAMILIES:
        c = chosen[fam]
        pair = (fam, c["version_id"])
        ev = reaction.load("events", years=config.CHECK_YEARS, check_for=pair)
        ct = reaction.load("controls", years=config.CHECK_YEARS, check_for=pair)
        out[fam] = judge(get(*pair), ev, ct, config.CHECK_YEARS, way=int(c["way"]), final=True, record=record)
        if record:
            save_page(fam, [out[fam]], "check")
    return out


# ============================================================================================================ helpers

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _code_hash() -> str:
    here = Path(__file__).parent
    return trial_files.code_sha256([here / "beliefs.py", here / "reaction.py", here / "surprise.py",
                                    here.parent / "referee" / "stats.py"])


def _calendar_fingerprint() -> str | None:
    try:
        from machine.news import calendar
        return calendar.fingerprint()
    except (OSError, ValueError, KeyError):
        return None


def _data_print(ev: pd.DataFrame, ct: pd.DataFrame | None) -> dict:
    def h(df: pd.DataFrame | None) -> str | None:
        if df is None or not len(df):
            return None
        cols = [c for c in df.columns if df[c].dtype.kind in "fiub"]
        return hashlib.sha256(pd.util.hash_pandas_object(df[cols], index=True).to_numpy().tobytes()).hexdigest()
    out = {"reactions_sha256": h(ev), "controls_sha256": h(ct)}
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


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.news.beliefs", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=("tune", "check", "rules"))
    ap.add_argument("--who", default="Tejas (with Claude)", help="who opens the check years (goes into the file)")
    a = ap.parse_args(argv)
    if a.step == "rules":
        print(RULES)
        return 0
    if a.step == "tune":
        for fam, reps in tune().items():
            for r in reps:
                print(r.text(), "\n" + "=" * 118)
        print("CHOSEN (one version per belief, for the check years):")
        for fam, c in read_chosen().items():
            print(f"  {fam}: {c['version_id']} · judged {'his way' if c['way'] > 0 else 'the OPPOSITE way'} · {c['why']}")
    else:
        for fam, r in check(a.who).items():
            print(r.text(), "\n" + "=" * 118)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
