"""'The last 10 times' (L12b.3): for one release of his news, the 10 past releases of the same news MOST LIKE IT, and
what gold did then - next to what gold did on ALL releases of that news.     python -m machine.news.similar pages

A LOOKUP, NOT A TRAINED PREDICTION MODEL. Nothing here learns from gold's reactions or predicts one: the matching rule
below was written (plan/work/L12b3-notes.md section 1) before any lookup ran, and it is never tuned on what gold did -
tuning it on the outcomes would turn it into a trained model. Every page and every summary line says so.

    from machine.news import similar
    tab = similar.releases()                                   # one row per release of his news, 2016-2023
    q = similar.query_of(tab, "Jobs report", "2023-10-06 15:30", when="after")   # a past release as the question
    res = similar.lookup(tab, q)                               # the 10 most like it: all released AND done before it
    for line in res.lines: print(line.text)

THE RELEASES (releases()): one row per minute in which a report's HEADLINE number came out (surprise.py: NFP = the
payrolls, CPI = core CPI m/m, the Fed = its rate decision - so the press conference 30 minutes later is not a second
release), for every report of his colours (Forex Factory red + orange, T7) with 1-minute prices around it
(reaction.py), 2016-2023 only (T14: the lookup shows gold's reaction from 2016-2023, never a later year). Each row
carries the report's OWN surprise (reaction.py keeps only the minute's main number) and everything reaction.py
measured. `done_at` = the release + 240 minutes: from then on every measure used here is known.

"MOST LIKE IT" (Claude's rule, fixed before any lookup ran; none of it is Tejas's rule):
  the same report; released and DONE before the question's minute (no future). After the number is out
  (when="after", news with a forecast): only the same surprise group of L12b.1 (big good for the dollar / small / big
  bad for the dollar). Then the smallest sum of STEPS:
    surprise           1 step = 1 usual miss of that number (L12b.1's z, each capped at 3 either way) - after only
    session            his session window differs (Asia / London / New York / outside) = 1 step
    day bias           up vs down = 1 step, up or down vs neutral = 1/2 (his bias on the day chart)
    week bias          the same on the week chart
    level above        1 step = half a day's range between the two distances to his nearest level above (a distance
                       over one day's range counts as one day's range; no level alive = one day's range)
    level below        the same below
    day's range used   1 step = half a day's range (how far gold had already moved that day before the release)
    2 hours before     1 step = a quarter of a day's range (the move into the release, up or down)
    gold's day range   1 step = 1.5 times bigger or smaller (his 10-day ADR in dollars: quiet years vs busy years)
  A value the question lacks is left out for all; a value a past release lacks counts 1 step. Ties: the more recent
  release first. Before the number is out (when="before" - and always for the Fed, whose surprise is known only the
  next day, and for news with no forecast number): no surprise part and no surprise group.

WHAT IS COUNTED (each summary line, next to the same count on ALL earlier releases of that news, on those with the
same kind of surprise when the number is out - the fair comparison for what the situation adds - and on the last 3
years of them, and the warning that 10 is few):
  first 5 minutes / after 1 hour   price 5 / 60 minutes after the release, above or below the last price before it
  the first hour                   from its high to its low: dollars, and rupees on 0.01 lot at that month's rate
  swept first                      which of his two nearest levels (above / below) a wick took first in the first hour
  after the sweep                  from that level: back through it by the size of the hour before, first ("turned
                                   back" - his news trap), or that far on first ("went on"), or neither in 4 hours
  calm again                       minutes until three 5-minute candles in a row are back to that clock time's usual
                                   size (not calm within 4 hours = 240)
  luck                             how often 10 releases picked at random from the ones the 10 were chosen from (the
                                   same kind of surprise, or all earlier ones) give that count or one further from
                                   the usual - the exact count of such a draw; no test, only a sense of size

python -m machine.news.similar pages     -> plan/work/news/<news>.md, one per report with 20+ releases, + pictures
python -m machine.news.similar look nfp 2023-10-06 [--before]     one lookup, printed
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machine import config, costs, guards, settings
from machine.news import reaction

K = 10                                  # the prompt: the 10 nearest past releases
MIN_RELEASES = 20                       # a page per report with at least 20 releases (reaction.type_table's 20)
RECENT_YEARS = 3                        # "8 years, and the last 3" (L12b2-notes.md 3a: CPI changed in 2021-2023)
DONE_MIN = reaction.WINDOW_AFTER        # a release is DONE 240 minutes after it: every measure used here is known
Z_CAP = 3.0                             # a surprise of 5 usual misses is as far from 1 as one of 3
LEVEL_FAR = 1.0                         # a level further than one day's range away (or none alive) = one day's range
CALM_NOT = float(reaction.WINDOW_AFTER)  # not calm within 4 hours counts as 240 minutes (reaction.type_table's way)
WHEN = ("before", "after")
SURPRISE_GROUPS = ("big good for the dollar", "small", "big bad for the dollar")

# one step of each part (Claude's, fixed before any lookup ran - plan/work/L12b3-notes.md section 1)
STEPS = {
    "surprise": (1.0, "1 usual miss of that number (L12b.1), each capped at 3 either way - only once the number is out"),
    "session": (1.0, "a different session window of yours (Asia / London / New York / outside them)"),
    "bias_d": (2.0, "your bias on the day chart: up vs down (neutral vs up or down = half a step)"),
    "bias_w": (2.0, "your bias on the week chart: the same"),
    "level_up": (0.5, "half a day's range between the distances to your nearest level above"),
    "level_dn": (0.5, "half a day's range between the distances to your nearest level below"),
    "adr_used": (0.5, "half a day's range of how far gold had already moved that day"),
    "before_120": (0.25, "a quarter of a day's range of gold's move in the 2 hours before"),
    "adr": (math.log(1.5), "gold's day range (ADR) 1.5 times bigger or smaller"),
}
SITUATION = ["session", "bias_d", "bias_w", "up_level_kind", "up_level_adr", "dn_level_kind", "dn_level_adr",
             "adr_used", "before_120_adr", "adr"]

# the page for each report: (file name, the short name he uses, what it is in plain words)
NAMES = {
    "Jobs report": ("nfp", "NFP", "the US jobs report (non-farm payrolls), first Friday of the month"),
    "CPI": ("cpi", "CPI", "US inflation (consumer prices); the number used is core CPI m/m"),
    "FOMC decision": ("fed-decision", "Fed decision", "the Fed's interest rate decision"),
    "FOMC minutes": ("fed-minutes", "Fed minutes", "the minutes of the Fed's last meeting"),
    "Fed Chair": ("fed-chair", "Fed Chair", "the Fed Chair speaking"),
    "Fed speaker": ("fed-speakers", "Fed speaker", "other Fed members speaking"),
    "Jobless claims": ("jobless-claims", "jobless claims", "new US jobless claims, every Thursday"),
    "Oil stocks": ("oil-stocks", "oil stocks", "US crude oil stocks, every Wednesday"),
    "Michigan survey": ("michigan", "Michigan", "the University of Michigan consumer survey"),
    "ADP report": ("adp", "ADP", "the ADP private jobs report, two days before NFP"),
    "ISM manufacturing": ("ism-manufacturing", "ISM manufacturing", "the ISM survey of factories"),
    "ISM services": ("ism-services", "ISM services", "the ISM survey of services"),
    "CB confidence": ("cb-confidence", "CB confidence", "the Conference Board consumer confidence survey"),
    "Retail sales": ("retail-sales", "retail sales", "US retail sales"),
    "Philly Fed": ("philly-fed", "Philly Fed", "the Philadelphia Fed factory survey"),
    "GDP": ("gdp", "GDP", "US growth (GDP q/q)"),
    "PPI": ("ppi", "PPI", "US producer prices"),
    "PCE": ("pce", "PCE", "the Fed's own inflation number (core PCE)"),
    "Durable goods": ("durable-goods", "durable goods", "orders of long-lasting goods"),
    "Chicago PMI": ("chicago-pmi", "Chicago PMI", "the Chicago business survey"),
    "Pending home sales": ("pending-home-sales", "pending home sales", "US homes under contract"),
    "Empire State": ("empire-state", "Empire State", "the New York Fed factory survey"),
    "Housing starts": ("housing-starts", "housing starts", "new US homes started"),
    "S&P Global PMI": ("sp-global-pmi", "S&P Global PMI", "the S&P Global business surveys"),
    "Existing home sales": ("existing-home-sales", "existing home sales", "sales of existing US homes"),
    "JOLTS": ("jolts", "JOLTS", "US job openings"),
    "Industrial production": ("industrial-production", "industrial production", "US factory output"),
    "Bond auction": ("bond-auction", "bond auction", "US Treasury bond auctions"),
    "New home sales": ("new-home-sales", "new home sales", "sales of new US homes"),
    "Import prices": ("import-prices", "import prices", "US import prices"),
    "Factory orders": ("factory-orders", "factory orders", "US factory orders"),
    "Trade balance": ("trade-balance", "trade balance", "the US trade balance"),
    "Employment cost": ("employment-cost", "employment cost", "the US employment cost index"),
}
# The 3 real past lookups of the prompt's "done when", chosen before any lookup ran: the three reports that move gold
# most in 2021-2023 (L12b2-notes.md 3a). Their pages draw the day and the 3 releases most like it; the other pages
# draw only their day (a picture is ~165 KB in git; the table lists all 10).
SHOW = ("Jobs report", "CPI", "FOMC decision")
GROUP_WORDS = {
    "big good for the dollar": "a big surprise, good for the dollar (bad for gold)",
    "small": "close to the forecast",
    "big bad for the dollar": "a big surprise, bad for the dollar (good for gold)",
    "not scored": "no surprise measured",
}
SESSION_WORDS = {"Asia": "Asia (04:30-10:30 India time)", "London": "London (12:30-15:30 India time)",
                 "New York": "New York (17:30-20:30 India time)", "outside": "outside your session windows"}
LOOKUP_WORDS = ("A LOOKUP of past releases — not a trained prediction model. It finds the past releases most like one "
                "release and shows what gold did then, next to what gold did on ALL releases of that news. It learns "
                "nothing and predicts nothing.")


def name_of(report: str) -> tuple[str, str, str]:
    if report in NAMES:
        return NAMES[report]
    slug = "".join(c if c.isalnum() else "-" for c in report.lower().replace("&", "and")).strip("-")
    return slug, report, report


def report_of(name: str) -> str:
    """'nfp' / 'NFP' / 'Jobs report' -> 'Jobs report'."""
    for rep, (slug, short, _) in NAMES.items():
        if name.lower() in (slug, short.lower(), rep.lower()):
            return rep
    return name


# =================================================================================================== the releases

def releases() -> pd.DataFrame:
    """One row per release of a report of his colours with prices around it, 2016-2023 (a LOOKUP: reaction.load(...,
    lookup=True) - never to test or choose). Columns: time (server), report, the report's own surprise (event, z,
    group, method, known_from, own_colour, number), done_at, usd_inr, and every column of reaction.py."""
    from machine.news import forexfactory, surprise
    years = config.TUNE_YEARS + config.CHECK_YEARS
    ev = reaction.load("events", years=years, lookup=True)
    x = reaction.by_report(ev[ev["his_news"] & ev["has_bars"].fillna(False)])
    x = x.drop(columns=["event", "kind", "z", "group", "method", "z_known_from"]).reset_index()
    s = surprise.load(str(min(years)), str(max(years)))
    rank = s["ff_colour"].map(reaction.COLOUR_RANK).fillna(0)
    own = s.assign(_c=rank).groupby(["server", "report"])["_c"].max().map(
        {v: k for k, v in reaction.COLOUR_RANK.items()}).rename("own_colour")
    h = s[s["headline"]].assign(_z=s["z"].notna())
    h = h.sort_values(["server", "report", "_z"], ascending=[True, True, False], kind="stable")
    h = h.drop_duplicates(["server", "report"])                  # PCE has one release written twice
    ff = forexfactory.load()[["id", "actual", "forecast"]].rename(columns={"id": "ff_id", "actual": "ff_a",
                                                                            "forecast": "ff_f"})
    h = h.merge(ff, on="ff_id", how="left")
    h["number"] = [_number(r) for r in h.itertuples()]
    h = h[["server", "report", "event", "z", "group", "method", "known_from", "number"]]
    t = x.merge(h, on=["server", "report"], how="inner")         # only the minute its HEADLINE number came out
    t = t.merge(own.reset_index(), on=["server", "report"], how="left")
    t = t.rename(columns={"server": "time"}).sort_values(["time", "report"], kind="stable", ignore_index=True)
    t["done_at"] = t["time"] + pd.Timedelta(minutes=DONE_MIN)
    months = t["time"].dt.strftime("%Y-%m-01")
    rate = {m: costs.usd_inr(m) for m in months.unique()}
    t["usd_inr"] = months.map(rate)
    _only_before_sealed(t)
    return t


def _number(r) -> str:
    """'336K (forecast 171K)' in Forex Factory's own writing; '' when there is no number to compare."""
    if r.method == "flagged":
        return ""
    a = r.ff_a if isinstance(r.ff_a, str) and r.ff_a else (f"{r.ff_actual:g}" if pd.notna(r.ff_actual) else "")
    f = r.ff_f if isinstance(r.ff_f, str) and r.ff_f else (f"{r.ff_forecast:g}" if pd.notna(r.ff_forecast) else "")
    if not a:
        return ""
    return f"{a} (forecast {f})" if f else f"{a} (no forecast)"


def _only_before_sealed(t: pd.DataFrame) -> None:
    """T14: the lookup shows gold's reaction from 2016-2023 only - never a sealed year, whatever the question's date."""
    if len(t) and t["time"].max() >= pd.Timestamp(config.SEALED_FROM):
        raise guards.SealedYearsError(f"a release at {t['time'].max()} is in the sealed years: the lookup shows "
                                      "2016-2023 only (plan/work/T14-decisions.md, use 2)")


# ===================================================================================================== the question

@dataclass(frozen=True)
class Query:
    """The release asked about. situation = the SITUATION columns known before it (reaction.CONTEXT)."""
    report: str
    time: pd.Timestamp                            # the release minute, MT5 server time
    when: str = "before"                          # "before" the number is out, or "after"
    z: float = float("nan")                       # the report's own surprise (after only)
    group: str | None = None
    situation: dict = field(default_factory=dict)
    number: str = ""                              # '336K (forecast 171K)'
    note: str = ""                                # why an "after" question had no number to match on

    def __post_init__(self):
        if self.when not in WHEN:
            raise ValueError(f"when is one of {WHEN}, not {self.when!r}")

    @property
    def has_number(self) -> bool:
        return self.when == "after" and self.group in SURPRISE_GROUPS and bool(np.isfinite(self.z))


def query_of(tab: pd.DataFrame, report: str, time: Any, when: str = "after") -> Query:
    """A past release as the question: what was known at its minute. "after": its own surprise, when the number is
    known at the release (a forecast number); the Fed's (the 2-year yield of the whole day) is known only the next
    day, and speeches have none - then the question matches on the situation only, and `note` says why."""
    from machine.data import clock
    t = pd.Timestamp(time)
    r = tab[(tab["report"] == report) & (tab["time"] == t)]
    if r.empty:
        raise KeyError(f"no release of {report!r} at {t} (MT5 server time)")
    r = r.iloc[0]
    sit = {c: r[c] for c in SITUATION}
    if when == "before":
        return Query(report, t, "before", situation=sit, number="")
    at = clock.server_to_utc(pd.DatetimeIndex([t]))[0]
    if r["method"] == "forecast" and pd.notna(r["z"]) and r["known_from"] <= at:
        return Query(report, t, "after", z=float(r["z"]), group=r["group"], situation=sit, number=r["number"])
    note = ("its surprise is the 2-year yield of the whole day, known only the next day" if r["method"] == "2y yield"
            else "this news has no forecast number" if r["method"] == "flagged"
            else "its surprise could not be measured (too few past forecasts)")
    return Query(report, t, "before", situation=sit, number=r["number"], note=note)


def _known(v: Any) -> bool:
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and not pd.isna(v)
    except (TypeError, ValueError):
        return True


def _level_adr(kind: pd.Series, adr_dist: pd.Series) -> pd.Series:
    """The distance to his nearest level on one side, in days' ranges: capped at LEVEL_FAR; no level = LEVEL_FAR;
    a level with no day's range to measure it = unknown (NaN)."""
    far = pd.Series(LEVEL_FAR, index=kind.index)
    return far.where(kind.isna(), np.minimum(adr_dist, LEVEL_FAR))


def steps(pool: pd.DataFrame, q: Query) -> pd.DataFrame:
    """One column per part of STEPS (in steps), named step_<part>, and their sum `steps`, for each row of pool."""
    s = q.situation
    idx = pool.index
    out: dict[str, pd.Series] = {}
    if q.has_number:
        out["surprise"] = (pool["z"].clip(-Z_CAP, Z_CAP) - float(np.clip(q.z, -Z_CAP, Z_CAP))).abs()
    if isinstance(s.get("session"), str):
        out["session"] = (pool["session"] != s["session"]).astype(float).where(pool["session"].notna())
    for c in ("bias_d", "bias_w"):
        if _known(s.get(c)):
            out[c] = (pool[c] - float(s[c])).abs() / STEPS[c][0]
    for side, part in (("up", "level_up"), ("dn", "level_dn")):
        mine = _level_adr(pd.Series([s.get(f"{side}_level_kind")]), pd.Series([s.get(f"{side}_level_adr")],
                                                                              dtype=float)).iat[0]
        if _known(mine):
            out[part] = (_level_adr(pool[f"{side}_level_kind"], pool[f"{side}_level_adr"]) - mine).abs() \
                / STEPS[part][0]
    if _known(s.get("adr_used")):
        out["adr_used"] = (pool["adr_used"] - float(s["adr_used"])).abs() / STEPS["adr_used"][0]
    if _known(s.get("before_120_adr")):
        out["before_120"] = (pool["before_120_adr"] - float(s["before_120_adr"])).abs() / STEPS["before_120"][0]
    if _known(s.get("adr")) and float(s["adr"]) > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["adr"] = np.log(pool["adr"].where(pool["adr"] > 0) / float(s["adr"])).abs() / STEPS["adr"][0]
    parts = pd.DataFrame({f"step_{k}": v.reindex(idx) for k, v in out.items()}, index=idx).astype(float).fillna(1.0)
    parts["steps"] = parts.sum(axis=1)
    return parts


# ======================================================================================================= the lookup

@dataclass(frozen=True)
class Line:
    what: str                                     # "first 5 minutes"
    text: str                                     # the sentence for him: the 10, the base rates, the warning


@dataclass
class Result:
    query: Query
    near: pd.DataFrame                            # the k most like it (their rows + one column per step + `steps`)
    base: pd.DataFrame                            # every release of the report released AND done before the question
    recent: pd.DataFrame                          # the same, in the last RECENT_YEARS years before it
    candidates: int                               # how many it chose from (after: the same surprise group)
    lines: list[Line]


def lookup(tab: pd.DataFrame, q: Query, k: int = K) -> Result:
    """The k releases of q.report most like q - only those released and DONE (all 4 measured hours over) before q's
    minute - with the summary lines. A LOOKUP: nothing is learnt, nothing is predicted."""
    _only_before_sealed(tab)
    base = tab[(tab["report"] == q.report) & (tab["done_at"] <= q.time)]
    recent = base[base["time"] >= q.time - pd.DateOffset(years=RECENT_YEARS)]
    cands = base[base["group"] == q.group] if q.has_number else base
    parts = steps(cands, q)
    order = parts.assign(_t=cands["time"]).sort_values(["steps", "_t"], ascending=[True, False], kind="stable")
    pick = order.index[:k]
    near = cands.loc[pick].join(parts.loc[pick])
    label = name_of(q.report)[1]
    kind = cands if q.has_number else None
    return Result(q, near, base, recent, len(cands),
                  summary_lines(near, base, recent, label, before_it=True, same_kind=kind))


# ============================================================================================ what gold did: counts

def counts(rows: pd.DataFrame) -> dict:
    """What gold did on some releases - the numbers behind every summary line and table."""
    d: dict[str, Any] = {"n": len(rows)}
    for c in ("move_5", "move_15", "move_60"):
        d[f"{c}_down"] = int((rows[c] < 0).sum())
        d[f"{c}_up"] = int((rows[c] > 0).sum())
    d["range_usd"] = _median(rows["range_60"])
    d["range_inr"] = _median(rows["range_60"] * rows["usd_inr"])
    side = rows["lv_side"]
    d["above"], d["below"] = int((side == "high").sum()), int((side == "low").sum())
    d["both"], d["neither"] = int((side == "both").sum()), int(side.isna().sum())
    race = rows.loc[side.isin(["high", "low"]) & rows["lv_race"].notna(), "lv_race"]
    d["swept"] = len(race)
    d["back"], d["on"] = int((race == "back").sum()), int((race == "on").sum())
    d["no_race"] = d["swept"] - d["back"] - d["on"]
    known = rows[rows["normal_5m"].notna()]
    calm = known["calm_normal"].fillna(CALM_NOT)
    d["calm_n"] = len(known)
    d["calm"] = _median(calm)
    d["calm_late"] = float(calm.quantile(0.75)) if len(calm) else float("nan")
    return d


def _median(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.median()) if len(s) else float("nan")


def _of(x: int, n: int) -> str:
    """'52 of 100'; with fewer than 20, the plain count '9 of 15'."""
    if n == 0:
        return "none to count"
    return f"{round(100 * x / n)} of 100" if n >= 20 else f"{x} of {n}"


def luck(x: int, n: int, hits: int, total: int) -> float:
    """How often n releases picked at random (without putting back) from `total` releases, `hits` of which went this
    way, give x or one further from the usual - x or more when x/n is at least hits/total, else x or fewer. The
    exact count for such a draw (hypergeometric)."""
    if n == 0 or total == 0 or n > total:
        return float("nan")
    den = math.comb(total, n)
    pr = [math.comb(hits, i) * math.comb(total - hits, n - i) / den for i in range(n + 1)]
    return float(sum(pr[x:]) if x / n >= hits / total else sum(pr[:x + 1]))


def _luck_words(x: int, n: int, hits: int, total: int, pool_words: str) -> str:
    tail = luck(x, n, hits, total)
    if not np.isfinite(tail):
        return ""
    way = "or more" if x / n >= hits / total else "or fewer"
    pct = round(100 * tail)
    often = "less than 1 time in 100" if pct < 1 else f"about {pct} times in 100"
    return f": by luck alone ({n} picked at random {pool_words}), {x} {way} of {n} happens {often}"


def _warning(n: int) -> str:
    return f"{n} events is few — treat as a hint"


def summary_lines(near: pd.DataFrame, base: pd.DataFrame, recent: pd.DataFrame, label: str,
                  before_it: bool = True, same_kind: pd.DataFrame | None = None) -> list[Line]:
    """The summary lines: each count of the 10 next to the same count on ALL earlier releases of that news, on those
    with the same kind of surprise (after the number is out - the fair comparison for what the situation adds;
    added after the first real lookup, plan/work/L12b3-notes.md section 3) and on the last 3 years of them; then the
    warning that 10 is few. The luck sentence picks its 10 from the releases the 10 were chosen from."""
    n = len(near)
    if n == 0:
        return [Line("none", f"No earlier {label} release to compare with.")]
    c = counts(near)
    groups = [(f"on all {len(base)} {label} days" + (" before it" if before_it else ""), counts(base))]
    if same_kind is not None:
        groups.append((f"with the same kind of surprise ({len(same_kind)})", counts(same_kind)))
    if len(recent):
        groups.append((f"the last {RECENT_YEARS} years ({len(recent)} days)", counts(recent)))
    pool_words, pool = ((f"from those {len(same_kind)}", groups[1][1]) if same_kind is not None
                        else (f"from all {len(base)}", groups[0][1]))

    def cmp(fmt) -> str:
        return " · ".join(f"{w}: {fmt(d)}" for w, d in groups)

    lines: list[Line] = []
    for col, what in (("move_5", "in the first 5 minutes"), ("move_60", "after 1 hour")):
        dn, up = c[f"{col}_down"], c[f"{col}_up"]
        key, word, x = ("down", "dropped", dn) if dn >= up else ("up", "rose", up)
        lines.append(Line(what, f"{x} of {n} {word} {what} ({cmp(lambda d: _of(d[f'{col}_{key}'], d['n']))}). "
                                f"{_warning(n)}{_luck_words(x, n, pool[f'{col}_{key}'], pool['n'], pool_words)}."))
    lines.append(Line("the first hour", f"In the middle case of the {n}, the first hour went ${c['range_usd']:.2f} "
                                        f"from its high to its low — {costs.inr(c['range_inr'])} on 0.01 lot "
                                        f"({cmp(lambda d: '$' + format(d['range_usd'], '.2f'))}). {_warning(n)}."))

    def swept(d: dict) -> str:
        m = d["n"]
        s = f"above {_of(d['above'], m)}, below {_of(d['below'], m)}, " + \
            (f"both {_of(d['both'], m)}, " if d["both"] else "") + f"neither {_of(d['neither'], m)}"
        return s.replace(" of 100,", ",") if m >= 20 else s

    both = f", both in the same minute in {c['both']}" if c["both"] else ""
    lines.append(Line("swept first", f"Your nearest level ABOVE was swept first in {c['above']} of {n}, the one BELOW "
                                     f"in {c['below']}{both}, neither in the first hour in {c['neither']} "
                                     f"({cmp(swept)}). {_warning(n)}."))
    if c["swept"] == 0:
        lines.append(Line("after the sweep", f"None of the {n} swept one of your levels in the first hour, so there "
                                             "is no turn back to count."))
    else:
        def race(d: dict) -> str:
            m = d["swept"]
            if m == 0:
                return "none swept"
            s = f"turned back {_of(d['back'], m)}, went on {_of(d['on'], m)}, neither {_of(d['no_race'], m)}"
            return (s.replace(" of 100,", ",") + " swept") if m >= 20 else s + " swept"

        lines.append(Line("after the sweep", f"After that first sweep, price turned back through the level (your news "
                                             f"trap) in {c['back']} of those {c['swept']}, went on in {c['on']}, "
                                             f"neither within 4 hours in {c['no_race']} ({cmp(race)}). "
                                             f"{_warning(c['swept'])}."))
    if c["calm_n"] < max(3, n // 2):
        lines.append(Line("calm again", f"Calm again: the usual size of a 5-minute candle at that time is known for "
                                        f"only {c['calm_n']} of the {n} — not counted."))
    else:
        lines.append(Line("calm again", f"In the middle case of the {c['calm_n']}, gold was calm again after "
                                        f"{c['calm']:.0f} minutes ({cmp(lambda d: format(d['calm'], '.0f') + ' minutes')}"
                                        f"). {_warning(c['calm_n'])}."))
    return lines


# ====================================================================================================== the picture

class Levels:
    """His liquidity levels (machine/blocks/liquidity.py - the rows that may start a setup, the same rows reaction.py
    measured the nearest level from), so a picture can draw the two nearest at a release by what they are."""

    def __init__(self, market):
        life = market.liquidity(settings.fingerprint()).life
        self.life = life[life["can_start"]].sort_values("start", kind="stable").reset_index(drop=True)
        self.opens5 = market.opens("5m")
        self._start = self.life["start"].to_numpy(np.int64)
        self._end = self.life["end"].to_numpy(np.int64)
        self._price = self.life["price"].to_numpy(float)
        self._slope = self.life["slope"].to_numpy(float)
        self._x0 = self.life["x0pos"].to_numpy(np.int64)
        self._kind = self.life["kind"].to_numpy(object)

    def pos5(self, t: Any) -> int:
        return int(np.searchsorted(self.opens5, pd.Timestamp(t).value, "right")) - 1

    def find(self, t0: Any, price: float, kind: str) -> pd.Series | None:
        """The level of that kind, alive at the 5-minute candle holding t0, at that price there."""
        i5 = self.pos5(t0)
        n = int(np.searchsorted(self._start, i5, "right"))
        pr = self._price[:n] + self._slope[:n] * (i5 - self._x0[:n])
        ok = (self._end[:n] >= i5) & (self._kind[:n] == kind) & (np.abs(pr - price) < 1e-6)
        j = np.flatnonzero(ok)
        return self.life.iloc[j[-1]] if len(j) else None

    def price_at(self, row: pd.Series, t: Any) -> float:
        return float(row["price"] + row["slope"] * (self.pos5(t) - row["x0pos"]))


BEFORE_BARS = 12                                  # the picture: 60 minutes before the release ...
AFTER_BARS = 48                                   # ... to 240 after (the measured window)
MAX_BEFORE_BARS = 100                             # reach back at most ~8 hours for a level's touches (one chart,
                                                  # at most ~150 candles - his T3 answer, "one chart")


def _ist(t: Any) -> pd.Timestamp:
    from machine.data import clock
    return clock.to_ist(pd.DatetimeIndex([pd.Timestamp(t)]))[0]


NEAR_MISS = 0.25                                  # a level price came this close to (dollars) is named "missed by"
# The news move began in the 1-minute candle stamped BEFORE the release (the broker's clock a few seconds behind).
# Found on NFP 5 Aug 2016: 1360.57 -> 1355.23 in the 15:29 candle, closing at its low, then 1339.25 in the release
# minute (15:30:00). Its mark: that candle is 5 times the usual size of the 30 before it and at least $1; it MOVED
# (its body at least 0.6 of its range) and closed in the last quarter of its range the way it moved; the release
# candle went on at least 50 cents past that close. (A spike that came back - a broker widening its spread before
# the news - is not it: 20 of the 31 big candles before a release, 2016-2023.) 11 of 2,831 release minutes.
EARLY_TIMES, EARLY_USD, EARLY_BODY, EARLY_END, EARLY_ON = 5.0, 1.0, 0.6, 0.25, 0.5


def early_starts(m1: pd.DataFrame, times) -> set:
    """The release minutes whose move began in the candle stamped before them. For the pages ONLY (a mark and a note:
    reaction.py's last price before the release already holds the start of the move there) - never used to match."""
    T = pd.DatetimeIndex(m1.index).as_unit("ns").asi8
    O, H, L, C = (m1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    minute = 60 * 10 ** 9
    out = set()
    for t in pd.DatetimeIndex(times):
        t0 = t.as_unit("ns").value
        i0 = int(np.searchsorted(T, t0, "left"))
        if i0 < 31 or i0 >= len(T) or T[i0] != t0 or T[i0 - 1] != t0 - minute:
            continue
        b = i0 - 1
        usual = float(np.median(H[b - 30:b] - L[b - 30:b]))
        size = H[b] - L[b]
        if size < EARLY_TIMES * max(usual, 0.01) or size < EARLY_USD or abs(C[b] - O[b]) < EARLY_BODY * size:
            continue
        down = C[b] < O[b]
        at_end = (C[b] - L[b] <= EARLY_END * size) if down else (H[b] - C[b] <= EARLY_END * size)
        goes_on = (L[i0] <= C[b] - EARLY_ON) if down else (H[i0] >= C[b] + EARLY_ON)
        if at_end and goes_on:
            out.add(pd.Timestamp(t))
    return out


def _second_words(above: bool, price: float, first_hour: pd.DataFrame | None) -> str:
    """For a level NOT swept first: what the eye sees in the first hour - swept second, or missed by a few cents
    (a wick that stops 3 cents short looks like a touch on a 5-minute chart; it is not a sweep: no wick past it)."""
    if first_hour is None or first_hour.empty:
        return ""
    past = first_hour["high"] > price if above else first_hour["low"] < price
    if past.any():
        return f"swept second, {_ist(past.idxmax()):%H:%M}"
    gap = price - float(first_hour["high"].max()) if above else float(first_hour["low"].min()) - price
    if gap < 0.005:
        return "touched, not past"
    return f"missed by {round(gap * 100)} cents" if gap <= NEAR_MISS else ""


def _level_items(row: pd.Series, lv: Levels | None, t0: pd.Timestamp, first_ok: pd.Timestamp, t_end: pd.Timestamp,
                 first_hour: pd.DataFrame | None = None):
    """The two nearest levels at the release, drawn by what they are - each runs to where it was swept first, or to
    the chart's end. Returns (items, the earliest time they need on the chart)."""
    from machine.charts import draw as D
    out = []
    need = t0
    rel_day = _ist(t0).date()
    for side, above in (("up", True), ("dn", False)):
        kind = row.get(f"{side}_level_kind")
        dist = row.get(f"{side}_level_dist")
        if not isinstance(kind, str) or not _known(dist):
            continue
        price = row["p0"] + dist if above else row["p0"] - dist
        first = row.get("lv_side") in (("high" if above else "low"), "both")
        swept_at = t0 + pd.Timedelta(minutes=float(row["lv_min"])) if first and _known(row.get("lv_min")) else None
        words = f"swept {_ist(swept_at):%H:%M}" if swept_at is not None else _second_words(above, price, first_hour)
        if row.get("lv_side") == "both":            # both taken inside one 1-minute candle: no first, no second
            past = (first_hour["high"] > price if above else first_hour["low"] < price) if first_hour is not None \
                else pd.Series(dtype=bool)
            swept_at = past.idxmax() if past.any() else t0
            words = f"both swept at {_ist(swept_at):%H:%M}"
        tail = "\n" + words if words else ""            # on a second line: the margin stays as wide as the name
        L = lv.find(t0, price, kind) if lv is not None else None
        style = dict(color=D.PURPLE)
        if L is None:
            out.append(D.Level(price, kind.replace("_", " ") + tail, end=swept_at, style="-" if first else "--",
                               width=2.2 if first else 1.5, name_below=not above, **style))
            continue
        made = pd.Timestamp(L["first_bar"])
        made_ist = _ist(made)
        hour = "hour-chart " if L["tf"] == "1h" else ""
        word = "high" if above else "low"
        if kind.startswith(("equal", "trendline")):
            touches = [pd.Timestamp(L["first_bar"])]
            if kind.startswith("trendline") and pd.notna(L.get("x0")):
                touches.append(pd.Timestamp(L["x0"]))
            if pd.notna(L.get("touch_bar")):
                touches.append(pd.Timestamp(L["touch_bar"]))
            touches = sorted(set(touches))
            on_chart = touches[0] >= first_ok
            if kind.startswith("equal"):
                if on_chart:
                    need = min(need, touches[0])
                    out.append(D.Line((touches[0], price), (touches[-1], price), touches=tuple(touches),
                                      name=f"{hour}equal {word}s{tail}", extend_to=swept_at or t_end, **style))
                else:
                    out.append(D.Level(price, f"{word} of {made_ist:%d %b %H:%M}{tail}", end=swept_at,
                                       style="-" if first else "--", width=2.2 if first else 1.5,
                                       name_below=not above, **style))
                continue
            way = "falling" if L["slope"] < 0 else "rising" if L["slope"] > 0 else "flat"
            if on_chart and len(touches) >= 2:
                need = min(need, touches[0])
                a, b = touches[0], touches[-1]
                out.append(D.Line((a, lv.price_at(L, a)), (b, lv.price_at(L, b)), touches=tuple(touches),
                                  name=f"{hour}{way} trendline{tail}", extend_to=swept_at or t_end, **style))
            else:
                a = first_ok
                b = swept_at or t_end
                out.append(D.Line((a, lv.price_at(L, a)), (b, lv.price_at(L, b)),
                                  name=f"{way} line from {made_ist:%d %b}{tail}", **style))
            continue
        if kind.startswith("swing"):
            name = f"{'daily' if L['tf'] == '1d' else 'weekly'} swing {word} of {made_ist:%d %b}"
        else:
            name = str(L["label"]) + (f" of {made_ist:%d %b}" if made_ist.date() != rel_day else "")
        out.append(D.Level(price, name + tail, end=swept_at, style="-" if first else "--",
                           width=2.2 if first else 1.5, name_below=not above, **style))
    return out, need


def gold_words(row: pd.Series) -> str:
    rng = row["range_60"]
    return (f"Gold: {costs.usd(row['move_5'])} at 5 minutes · {costs.usd(row['move_60'])} at 1 hour · the first hour "
            f"${rng:.2f} high to low ({costs.inr(rng * row['usd_inr'])} on 0.01 lot)")


def number_words(row: pd.Series) -> str:
    grp = GROUP_WORDS.get(row["group"], "")
    if row["method"] == "flagged" or not row["number"]:
        return f"{row['event']}: no forecast number, so no surprise to measure"
    if row["method"] == "2y yield":
        return f"{row['event']} {row['number']} · measured the next day: {grp}"
    return f"{row['event']} {row['number']}: {grp}"


def picture(row: pd.Series, m5: pd.DataFrame, lv: Levels | None, *, subject: str, heading: str | None = None,
            sub: str | None = None, early: bool = False):
    """One release on ONE 5-minute chart (India time): 60 minutes before to 240 after, the release as a line, his two
    nearest levels at the release - the one swept first solid, ending where it was swept. early: the move began in
    the candle stamped before the release (early_starts) - said on the picture, so the eye is not fooled."""
    from machine.charts import draw as D
    t0 = pd.Timestamp(row["time"])
    i = int(m5.index.searchsorted(t0, "left"))              # the candle holding the release (or the next one)
    if i >= len(m5) or m5.index[i] - t0 >= pd.Timedelta(minutes=5):
        raise ValueError(f"no 5-minute candle at the release {t0}")
    j_far = max(0, i - MAX_BEFORE_BARS)
    t_end = m5.index[min(len(m5), i + AFTER_BARS) - 1]
    first_hour = m5.iloc[i:i + 12] if m5.index[i] == t0 else None   # 5-minute candles = whole minutes only
    items, need = _level_items(row, lv, t0, m5.index[j_far], t_end, first_hour)
    i0 = min(i - BEFORE_BARS, int(m5.index.searchsorted(need, "right")) - 1 - 3)
    w = m5.iloc[max(0, i0):min(len(m5), i + AFTER_BARS)]
    short = name_of(row["report"])[1]
    ist = _ist(t0)
    its = [D.Candles(), D.Moment(t0, f"{short} out {ist:%H:%M}"), *items]
    num = number_words(row)
    if len(num) > 80:                               # a long report name: the surprise words on their own line (#124)
        num = num.replace(": ", ":\n", 1)
    note = (f"\nThe move began in the {_ist(t0 - pd.Timedelta(minutes=1)):%H:%M} minute: the broker's clock ran a "
            "few seconds behind." if early else "")
    return D.Picture(
        kind="news", subject=subject, heading=heading or f"{short} · {ist:%a %d %b %Y}",
        sub=sub if sub is not None else f"{num}\n{gold_words(row)}{note}",
        rows=[[D.Panel(w, "5m", its, "the release, and 4 hours after", role="main", clock="ist")]],
        footer_left="purple = your nearest levels above and below when the news came · solid = swept first",
        footer_right="a lookup of the past, not a prediction")


# ======================================================================================================== the pages

def example(same: pd.DataFrame) -> pd.Series:
    """The page's real day, by a rule fixed before any lookup ran: the last release of 2023 whose number was a BIG
    surprise known at the release; else the last release of 2023; else the last release there is."""
    from machine.data import clock
    y = same[same["time"].dt.year == max(config.CHECK_YEARS)]
    at = pd.Series(clock.server_to_utc(pd.DatetimeIndex(y["time"])), index=y.index)
    big = y[(y["method"] == "forecast") & y["z"].abs().ge(1) & (y["known_from"] <= at)]
    pick = big if len(big) else y if len(y) else same
    return pick.iloc[-1]


def _day(t: Any) -> str:
    return f"{_ist(t):%a %d %b %Y, %H:%M}"


def level_names(row: pd.Series, lv: Levels | None) -> dict:
    """{"up": "London session high", "dn": ...}: what his two nearest levels at the release were, in his words."""
    out = {}
    for side, above in (("up", True), ("dn", False)):
        kind, dist = row.get(f"{side}_level_kind"), row.get(f"{side}_level_dist")
        if not isinstance(kind, str):
            continue
        L = lv.find(row["time"], row["p0"] + dist if above else row["p0"] - dist, kind) if lv is not None else None
        word = "high" if above else "low"
        if L is None:
            out[side] = kind.replace("_", " ")
        elif kind.startswith("equal"):
            out[side] = ("hour-chart " if L["tf"] == "1h" else "") + f"equal {word}s"
        elif kind.startswith("trendline"):
            out[side] = str(L["label"]).replace("1h ", "hour-chart ")
        elif kind.startswith("swing"):
            out[side] = f"{'daily' if L['tf'] == '1d' else 'weekly'} swing {word}"
        else:
            out[side] = str(L["label"])
    return out


def _situation_words(s: dict, names: dict | None = None) -> list[str]:
    def bias(v):
        return {1.0: "up", -1.0: "down", 0.0: "neutral"}.get(float(v), "not known") if _known(v) else "not known"

    names = names or {}
    out = [f"session: {SESSION_WORDS.get(s.get('session'), 'not known')}",
           f"your bias: day {bias(s.get('bias_d'))} · week {bias(s.get('bias_w'))}"]
    for side, word in (("up", "above"), ("dn", "below")):
        kind = s.get(f"{side}_level_kind")
        v = s.get(f"{side}_level_adr")
        if isinstance(kind, str):
            usd = f"${v * s['adr']:.2f}, " if _known(v) and _known(s.get("adr")) else ""
            far = f"{usd}{v:.2f} of a day's range away" if _known(v) else "distance not known"
            out.append(f"your nearest level {word}: {names.get(side, kind.replace('_', ' '))}, {far}")
        else:
            out.append(f"your nearest level {word}: none alive")
    if _known(s.get("adr")):
        used = f" · today had already moved {s['adr_used']:.2f} of it" if _known(s.get("adr_used")) else ""
        out.append(f"gold's day range (your 10-day ADR): ${s['adr']:.2f}{used}")
    if _known(s.get("before_120_adr")) and _known(s.get("adr")):
        mv = s["before_120_adr"] * s["adr"]
        out.append(f"the 2 hours before: gold {'rose' if mv > 0 else 'fell'} ${abs(mv):.2f}")
    return out


def _side_word(v: Any) -> str:
    return {"high": "above", "low": "below", "both": "both"}.get(v, "neither") if isinstance(v, str) else "neither"


def _race_word(v: Any) -> str:
    return {"back": "turned back", "on": "went on"}.get(v, "neither") if isinstance(v, str) else ""


def _surprise_cell(r: pd.Series) -> str:
    """The surprise in usual misses; past 3 it is written 'over +3' (the 2020 lockdown months reach +143)."""
    if r["method"] != "forecast" or not _known(r["z"]):
        return "-"
    z = float(r["z"])
    size = f"over {'+' if z > 0 else '-'}{Z_CAP:g}" if abs(z) > Z_CAP else f"{z:+.1f}"
    return f"{size} ({'big' if abs(z) >= 1 else 'small'})"


EARLY_WORDS = ("† the move began in the minute stamped before the release (the broker's clock ran a few seconds "
               "behind), so its first minutes here start after part of the move — its 5-minute move is smaller here "
               "than it really was.")


def _dagger(t: Any, early: set) -> str:
    return " †" if pd.Timestamp(t) in early else ""


def _near_table(near: pd.DataFrame, early: set = frozenset()) -> list[str]:
    L = ["| # | day (India time) | the number | surprise | day bias · 2 hours before | 5 min | 1 hour | first hour | "
         "swept first · then | how different |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (_, r) in enumerate(near.iterrows(), start=1):
        bias = {1.0: "up", -1.0: "down", 0.0: "neutral"}.get(r["bias_d"], "?") if _known(r["bias_d"]) else "?"
        mv = r["before_120"] if _known(r["before_120"]) else float("nan")
        mv_w = costs.usd(mv) if _known(mv) else "?"
        race = _race_word(r["lv_race"])
        L.append(f"| {k} | {_day(r['time'])}{_dagger(r['time'], early)} | {r['number'] or '-'} | "
                 f"{_surprise_cell(r)} | {bias} · {mv_w} | "
                 f"{costs.usd(r['move_5'])} | {costs.usd(r['move_60'])} | ${r['range_60']:.2f} | "
                 f"{_side_word(r['lv_side'])}{' · ' + race if race else ''} | {r['steps']:.1f} steps |")
    if any(pd.Timestamp(t) in early for t in near["time"]):
        L += ["", EARLY_WORDS]
    return L


def _all_table(tab_all: pd.DataFrame, label: str) -> list[str]:
    first, last = int(tab_all["time"].dt.year.min()), int(tab_all["time"].dt.year.max())
    old = tab_all
    new = tab_all[tab_all["time"].dt.year >= max(config.CHECK_YEARS) - RECENT_YEARS + 1]
    a, b = counts(old), counts(new)

    def pair(d, key_dn, key_up):
        return f"down {_of(d[key_dn], d['n'])} · up {_of(d[key_up], d['n'])}"

    def swept(d):
        both = f" · both {_of(d['both'], d['n'])}" if d["both"] else ""
        return (f"above {_of(d['above'], d['n'])} · below {_of(d['below'], d['n'])}{both} · neither "
                f"{_of(d['neither'], d['n'])}")

    def race(d):
        return (f"turned back {_of(d['back'], d['swept'])} · went on {_of(d['on'], d['swept'])} · neither "
                f"{_of(d['no_race'], d['swept'])}")

    y0 = max(config.CHECK_YEARS) - RECENT_YEARS + 1
    rows = [
        ("the first 5 minutes", pair(a, "move_5_down", "move_5_up"), pair(b, "move_5_down", "move_5_up")),
        ("after 1 hour", pair(a, "move_60_down", "move_60_up"), pair(b, "move_60_down", "move_60_up")),
        ("the first hour, high to low (middle case)", f"${a['range_usd']:.2f} ({costs.inr(a['range_inr'])})",
         f"${b['range_usd']:.2f} ({costs.inr(b['range_inr'])})" if b["n"] else "-"),
        ("your nearest level swept first (first hour)", swept(a), swept(b) if b["n"] else "-"),
        (f"after that first sweep (of the {a['swept']} / {b['swept']} with one)", race(a) if a["swept"] else "-",
         race(b) if b["swept"] else "-"),
        ("calm again after (middle case / 3 of 4 by)", f"{a['calm']:.0f} / {a['calm_late']:.0f} minutes",
         f"{b['calm']:.0f} / {b['calm_late']:.0f} minutes" if b["calm_n"] else "-"),
    ]
    L = [f"| what gold did | {first}–{last} ({a['n']} releases) | {y0}–{last} ({b['n']}) |", "|---|---|---|"]
    L += [f"| {w} | {x} | {y} |" for w, x, y in rows]
    return L


def _group_table(same: pd.DataFrame) -> list[str]:
    L = ["| the number | releases | gold UP 15 minutes later | gold UP 1 hour later | the first hour, high to low "
         "(middle case) |", "|---|---|---|---|---|"]
    for g in ("big bad for the dollar", "small", "big good for the dollar", "not scored"):
        x = same[same["group"] == g]
        if x.empty:
            continue
        c = counts(x)
        L.append(f"| {GROUP_WORDS[g]} | {c['n']} | {_of(c['move_15_up'], c['n'])} | {_of(c['move_60_up'], c['n'])} | "
                 f"${c['range_usd']:.2f} |")
    return L


def _last_table(same: pd.DataFrame, k: int = K, early: set = frozenset()) -> list[str]:
    L = ["| day (India time) | the number | surprise | 5 min | 1 hour | first hour | swept first · then | calm "
         "after |", "|---|---|---|---|---|---|---|---|"]
    last = same.tail(k).iloc[::-1]
    for _, r in last.iterrows():
        race = _race_word(r["lv_race"])
        calm = ("-" if not _known(r["normal_5m"]) else f"{r['calm_normal']:.0f} min" if _known(r["calm_normal"])
                else "not in 4 h")
        L.append(f"| {_day(r['time'])}{_dagger(r['time'], early)} | {r['number'] or '-'} | {_surprise_cell(r)} | "
                 f"{costs.usd(r['move_5'])} | "
                 f"{costs.usd(r['move_60'])} | ${r['range_60']:.2f} | {_side_word(r['lv_side'])}"
                 f"{' · ' + race if race else ''} | {calm} |")
    if any(pd.Timestamp(t) in early for t in last["time"]):
        L += ["", EARLY_WORDS]
    return L


STEP_TABLE = ["| part | 1 step = |", "|---|---|"] + [f"| {p} | {w} |" for p, w in (
    ("the news", "the same news only (no steps: a different news is never picked)"),
    ("the surprise", STEPS["surprise"][1] + "; and only the same kind of surprise (big good for the dollar / "
                                            "close to the forecast / big bad for the dollar)"),
    ("session", STEPS["session"][1]), ("day bias", STEPS["bias_d"][1]), ("week bias", STEPS["bias_w"][1]),
    ("level above", STEPS["level_up"][1] + " (further than one day's range, or none = one day's range)"),
    ("level below", STEPS["level_dn"][1]), ("day's range used", STEPS["adr_used"][1]),
    ("2 hours before", STEPS["before_120"][1]), ("gold's day range", STEPS["adr"][1]))]


def page(tab: pd.DataFrame, report: str, m5: pd.DataFrame, lv: Levels | None, folder: Path,
         check_pictures: bool = True, early: set = frozenset()) -> tuple[Path, list, Result]:
    """One page: plan/work/news/<news>.md and its pictures. Returns (page, drawn pictures, the lookup).
    early: the release minutes whose move began in the minute before (early_starts) - marked, never matched on."""
    from machine.charts import draw as D, lint
    slug, short, plain = name_of(report)
    same = tab[tab["report"] == report].sort_values("time", kind="stable")
    ex = example(same)
    q = query_of(tab, report, ex["time"], "after")
    res = lookup(tab, q)
    pics_dir = folder / "pictures"
    drawn = []

    def draw(r: pd.Series, why: str) -> str:
        ist = _ist(r["time"])
        path = pics_dir / f"{slug}-{ist:%Y-%m-%d-%H%M}.jpg"
        d = D.render(picture(r, m5, lv, subject=f"L12b.3 {short} {ist:%Y-%m-%d %H:%M} ({why})",
                             early=pd.Timestamp(r["time"]) in early), path)
        if check_pictures:
            lint.passed(d)
        drawn.append(d)
        return f"pictures/{path.name}"

    first, last = int(same["time"].dt.year.min()), int(same["time"].dt.year.max())
    y0 = max(config.CHECK_YEARS) - RECENT_YEARS + 1
    clocks = same["time"].map(lambda t: f"{_ist(t):%H:%M}").value_counts()
    usual = " or ".join(clocks.index[:2]) if len(clocks) else "-"
    L = [f"# {short} — \"the last {K} times\"", "",
         f"> **{LOOKUP_WORDS}**",
         f"> Gold: 1-minute and 5-minute candles, {first}–{last} only (no year after 2023). Your news = Forex Factory "
         "red and orange. Nothing here was bought or sold. How it is made: `plan/work/L12b3-notes.md`.", "",
         f"**{short}** = {plain}. **{len(same)} releases with prices, {first}–{last}** "
         f"({int((same['time'].dt.year >= y0).sum())} in {y0}–{last}). Usually at {usual} India time.", ""]

    # ---- 1. the lookup on a real day
    L += [f"## 1. The lookup on a real day: {_day(q.time)} India time", ""]
    if q.has_number:
        size = ("The number was the same as the forecast." if abs(q.z) < 0.05 else
                f"The miss was {abs(q.z):.1f} times the usual miss of this number.")
        L.append(f"**Known at {_ist(q.time):%H:%M}:** {number_words(ex)}. {size}")
    else:
        L.append(f"**Known at {_ist(q.time):%H:%M}:** only the situation — {q.note}. So the lookup matches on the "
                 "situation only.")
    L += ["", "**The situation just before it:**", ""]
    L += [f"- {w}" for w in _situation_words(q.situation, level_names(ex, lv))] + [""]
    kind = (f"the same kind of surprise ({GROUP_WORDS[q.group].split(' (')[0]}), then the closest situation"
            if q.has_number else "the closest situation")
    L.append(f"**The {len(res.near)} past {short} releases most like it** — all released and done before it; "
             f"chosen from {res.candidates} by {kind}:")
    if len(res.near) < K:
        L.append(f"*Only {res.candidates} earlier {short} releases fit, so only {len(res.near)} are shown.*")
    L += [""] + [f"- {ln.text}" for ln in res.lines] + [""]
    L += _near_table(res.near, early) + [""]
    L += [f"**What really happened on {_ist(q.time):%d %b %Y}** (the day itself is not one of the {len(res.near)}; "
          "shown to compare — one day proves nothing either way):", "",
          f"{gold_words(ex)}.", "", f"![{short} {_ist(q.time):%d %b %Y}]({draw(ex, 'the day looked up')})", ""]
    n_pics = 3 if report in SHOW else 0            # ~165 KB a picture: the other pages draw only their day
    for k, (_, r) in enumerate(res.near.head(n_pics).iterrows(), start=1):
        L += [f"**Number {k} most like it: {_day(r['time'])}** — {r['steps']:.1f} steps different.", "",
              f"![{short} {_ist(r['time']):%d %b %Y}]({draw(r, f'number {k} most like {_ist(q.time):%Y-%m-%d}')})",
              ""]

    # ---- 2. all days
    L += [f"## 2. On all {short} days — 8 years, and the last 3", "",
          "The same counts on every release, with no lookup — what the lines above are compared with.", ""]
    L += _all_table(same, short) + [""]

    # ---- 3. by the surprise
    L += ["## 3. By the surprise (how far the number landed from the forecast)", ""]
    if (same["method"] == "flagged").all():
        L += ["This news has no forecast number, so there is no surprise to split by.", ""]
    else:
        if (same["method"] == "2y yield").any():
            L += ["The Fed has no forecast number: its surprise is the move of the 2-year yield over the whole day, "
                  "**known only the next day**. So this table describes the past; it cannot be known at the "
                  "release, and the lookup above does not use it.", ""]
        L += _group_table(same) + [""]

    # ---- 4. the last 10
    L += [f"## 4. The last {K} {short} releases (before 2024)", ""] + _last_table(same, early=early) + [""]

    # ---- 5. the rule
    L += ["## 5. How \"most like it\" is chosen", "",
          "Claude's rule, written before any lookup ran, the same for every news. It was not tuned on what gold did "
          "— tuning it on gold's moves would turn this lookup into a trained model. The past releases with the "
          "fewest steps are the most like it; a tie goes to the more recent one. Only releases whose 4 measured "
          "hours were over before the day asked about are used (no future).", ""] + STEP_TABLE + [""]
    L += ["**The words.** \"Swept\" = a wick went past your level (session high or low, equal highs or lows, a "
          "trendline, a day or week swing) in the first hour. \"Turned back\" = after that, price came back through "
          "the level as far as the whole hour before the news had moved, before it went that far on (your news "
          "trap). \"Calm again\" = three 5-minute candles in a row no bigger than usual at that clock time. "
          "\"With the same kind of surprise\" = the fair comparison once the number is out: it shows what the "
          "surprise alone does, so the lookup's 10 are special only if they differ from THAT. \"By luck alone\" = "
          "how often 10 releases picked at random from the ones the 10 were chosen from give that count or one "
          "further from the usual. 6 things are counted on the same 10 releases, so one of them standing out by "
          "luck is normal.", ""]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}.md"
    _write(path, "\n".join(L))
    return path, drawn, res


def _write(path: Path, text: str) -> None:
    """Temp file + move (lesson B10: never open a file with "w" to rewrite it in place)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def index_page(tab: pd.DataFrame, made: list[tuple[str, Path]], folder: Path) -> Path:
    y0 = max(config.CHECK_YEARS) - RECENT_YEARS + 1
    L = ["# Your news — \"the last 10 times\", one page per news (L12b.3)", "",
         f"> **{LOOKUP_WORDS}**",
         "> Gold 2016–2023 only. Your news = Forex Factory red and orange. Every page: the lookup on its last big day "
         "of 2023, what gold did on all its days (8 years and the last 3), by the surprise, and its last 10 "
         "releases. How it is made: `plan/work/L12b3-notes.md`.", "",
         f"| news | releases 2016–2023 ({y0}–2023) | the first hour, high to low, middle case: 2016–2023 / "
         f"{y0}–2023 |", "|---|---|---|"]
    rows = []
    for rep, path in made:
        same = tab[tab["report"] == rep]
        new = same[same["time"].dt.year >= y0]
        a, b = counts(same), counts(new)
        rows.append((a["range_usd"], f"| [{name_of(rep)[1]}]({path.name}) — {name_of(rep)[2]} | {a['n']} ({b['n']}) | "
                                      f"${a['range_usd']:.2f} / " + (f"${b['range_usd']:.2f}" if b["n"] else "-")
                                      + " |"))
    L += [r for _, r in sorted(rows, key=lambda x: -x[0])]
    path = folder / "README.md"
    _write(path, "\n".join(L))
    return path


def pages(only: list[str] | None = None, folder: Path | None = None) -> list[Path]:
    from machine.data import bars as BARS
    from machine.ideas.compile import Market
    folder = Path(folder or config.WORK_DIR / "news")
    tab = releases()
    n = tab.groupby("report").size()
    reports = [r for r in n[n >= MIN_RELEASES].sort_values(ascending=False).index]
    if only:
        want = {report_of(o) for o in only}
        reports = [r for r in reports if r in want]
    first, last = str(min(config.TUNE_YEARS)), str(max(config.CHECK_YEARS))
    m5 = BARS.load_bars("5m", first, last)
    early = early_starts(BARS.load_bars("1m", first, last), tab["time"].unique())
    print(f"the move began in the minute before the release: {len(early)} of {tab['time'].nunique()} release minutes")
    lv = Levels(Market.load(first, last))
    made = []
    for rep in reports:
        path, drawn, res = page(tab, rep, m5, lv, folder, early=early)
        made.append((rep, path))
        print(f"{path.relative_to(config.ROOT).as_posix()}: {len(drawn)} pictures · lookup on "
              f"{_day(res.query.time)} ({res.query.when}) from {res.candidates}")
    if not only:
        made_all = [(r, folder / f"{name_of(r)[0]}.md") for r in reports]
        index_page(tab, made_all, folder)
    return [p for _, p in made]


# ============================================================================================================ main

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.news.similar", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages", help="write plan/work/news/<news>.md + pictures (one per news with 20+ releases)")
    p.add_argument("--only", nargs="*", help="only these news (nfp, cpi, fed-decision ...)")
    look = sub.add_parser("look", help="one lookup on a past release, printed")
    look.add_argument("news", help="nfp, cpi, fed-decision ... or the report name")
    look.add_argument("date", help="the release day in India time, YYYY-MM-DD")
    look.add_argument("--before", action="store_true", help="before the number is out (situation only)")
    a = ap.parse_args(argv)
    if a.cmd == "pages":
        pages(a.only)
        return 0
    tab = releases()
    rep = report_of(a.news)
    same = tab[(tab["report"] == rep)]
    day = same[same["time"].map(lambda t: f"{_ist(t):%Y-%m-%d}") == a.date]
    if day.empty:
        print(f"no {rep} release on {a.date} (India time) in 2016-2023")
        return 1
    q = query_of(tab, rep, day.iloc[0]["time"], "before" if a.before else "after")
    res = lookup(tab, q)
    print(f"{name_of(rep)[1]} · {_day(q.time)} India time · {q.when} the number" + (f" ({q.note})" if q.note else ""))
    print(LOOKUP_WORDS)
    for ln in res.lines:
        print("-", ln.text)
    print("\n".join(_near_table(res.near)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
