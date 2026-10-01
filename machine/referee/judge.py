"""The referee (L5): every result passes through here. It says "chance" whenever the result could be luck.

The path of one idea:
  1. TUNE years 2016-2020: try versions (each run leaves a trial file); choose one with stats.choose_version
     (Deflated Sharpe + PBO — used only here, only on these years).
  2. Open the check years ONCE for the chosen version:
        splits.open_check("idea-001", "v3", who="Tejas", why="v3 chosen on 2016-2020 by Deflated Sharpe")
  3. Judge it on 2021-2023:
        rep = judge_check(signals_2021_2023, tf="5m", key=prior.IdeaKey.make("equal_highs_lows", "5m", "line:touch"))
        print(rep.text())
  Later versions of the same family never see the check years; they are judged by walk-forward inside 2016-2020:
        rep = judge_walk_forward({"v3": sigs_v3, "v4": sigs_v4}, tf="5m", key=key)   # signals over 2016-2020

Each judgement runs the idea, config.RANDOM_RUNS random-entry runs and its fake levels through the tester's own
engine, then writes ONE trial file (kind "referee") with the answer, every number, and the rules' fingerprint.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, costs, guards
from machine import trials as trial_files
from machine.data import bars
from machine.referee import fake_levels, power, prior, project, random_entries, splits, stats, trials, verdict
from machine.referee.replays import (Segment, closed_trades, make_segment, outcome_counts, path_counts, pooled, replay,
                                     trade_rows)
from machine.tester.models import Signal
from machine.tester.run import summarize

# A CONTROL run (an idea with no edge on purpose, like the Monday rule) is recorded like any run (R14), under this
# kind, so project.py never counts it as an idea's answer.
CONTROL_KIND = "referee-control"
REFEREE_FILES = ("judge.py", "replays.py", "random_entries.py", "fake_levels.py", "stats.py", "power.py",
                 "verdict.py", "prior.py", "splits.py", "trials.py", "project.py")
TESTER_FILES = ("engine.py", "models.py", "run.py")


@dataclass
class Report:
    idea_id: str
    version_id: str
    tf: str
    judged_on: str
    answer: verdict.Answer
    numbers: dict
    counts: trials.Counts
    seen: prior.Seen
    power_table: dict
    choices: list[dict] = field(default_factory=list)
    project: dict = field(default_factory=dict)
    trial_path: Path | None = None
    control: bool = False

    @property
    def word(self) -> str:
        return self.answer.word

    def text(self) -> str:
        inr = costs.inr
        n = self.numbers
        s = n["summary"]
        lots = " / ".join(f"{x:.2f}" for x in n["lots"])
        lines = [
            f"THE REFEREE · {self.idea_id} version {self.version_id} · {self.tf} bars · MT5 server time"
            + (" · a CONTROL run (no edge on purpose; not counted as an idea's answer)" if self.control else ""),
            f"Judged on: {self.judged_on}",
            f"ANSWER: {self.answer.word.upper()}  ({self.answer.why})",
            "",
        ]
        if s["trades"]:
            lines.append(f"Trades: {s['trades']} · won {s['won']} ({100 * s['won'] / s['trades']:.0f}%) · after cost "
                         f"{inr(s['net_inr'])} · {inr(s['per_trade'])} a trade · {lots} lot · ₹40 per position per "
                         "0.01 lot, all-in (LEARNED #54, #115)")
        else:
            lines.append("Trades: 0")
        years = " · ".join(f"{y} {inr(v['net_inr'])} ({v['trades']})" for y, v in n["per_year"].items())
        wy = n["worst_year"]
        lines += [f"Year by year (trades in brackets; a trade counts in the year it closed): {years}",
                  f"Worst year: {wy['year']} {inr(wy['net_inr'])} ({wy['trades']} trades)",
                  f"Setups skipped (a trade was already on): {n['skipped']} · still open at the end (not counted): "
                  f"{n['open_at_end']}",
                  "", "The 6 checks:"]
        lines += [f"  {c.number}. {c.name}: {'yes' if c.passed else 'NO'} — {c.shown}" for c in self.answer.checks]
        lines += ["", *power.text(self.power_table), "", self.seen.text()]
        for c in self.choices:
            vs = " · ".join(f"{k} {v['deflated_sharpe']:.2f}" for k, v in c["versions"].items())
            pbo = "" if c["pbo"] is None else f"; chance the choice was luck (PBO): {100 * c['pbo']['pbo']:.0f} in 100"
            lines.append(f"Year {c['judge_year']}: chosen on {c['years'][0]}-{c['years'][-1]} — {c['chosen']} "
                         f"(Deflated Sharpe = the chance its edge is real, allowing for {c['tries']} tries: {vs}{pbo})")
        lines += ["", self.counts.text(), *project.text(self.project), "", verdict.RULES]
        if self.trial_path is not None:
            lines.append(f"\nTrial file: {_rel(self.trial_path)}")
        return "\n".join(lines)


# ================================================================================================================
# the two ways in
# ================================================================================================================

def judge_check(signals: list[Signal], *, tf: str, key: prior.IdeaKey, m1: pd.DataFrame | None = None,
                n_random: int | None = None, record: bool = True, note: str | None = None,
                control: bool = False) -> Report:
    """Judge ONE version on the check years 2021-2023 — only the version the family's check years were opened for."""
    idea, version, parent = _one_version(signals)
    st = splits.status(idea, version, parent)
    if st != "check":
        why = ("they are still closed: choose the version on 2016-2020 first, then splits.open_check()"
               if st == "closed" else
               "they were opened for another version of this family: judge this one by walk-forward inside "
               "2016-2020 (judge_walk_forward)")
        raise guards.CheckYearsClosedError(f"{idea} {version} cannot be judged on the check years — {why}.")
    years = config.CHECK_YEARS
    m1, data = _bars(m1, years)
    guards.check_opened(idea, [m1.index[0], m1.index[-1]], version=version)
    seg = make_segment(splits.years_text(years), years, tf, list(signals), m1, strict=True)
    opened = splits.family_opening(idea, parent)
    judged_on = (f"the check years {splits.years_text(years)} (opened once, {str(opened.get('opened_at', '?'))[:10]}, "
                 f"for this version)")
    return _judge([seg], tf=tf, key=key, idea=idea, version=version, parent=parent, judged_on=judged_on,
                  n_random=n_random, choices=[], data=data, record=record, note=note, control=control)


def judge_walk_forward(versions: dict[str, list[Signal]], *, tf: str, key: prior.IdeaKey, newest: str | None = None,
                       m1: pd.DataFrame | None = None, n_random: int | None = None, record: bool = True,
                       note: str | None = None, control: bool = False, tfs: dict[str, str] | None = None) -> Report:
    """Judge the versions of one family by walk-forward inside 2016-2020.

    versions: label -> that version's signals over the tune years. For each judge year (2018, 2019, 2020) the version
    with the best Deflated Sharpe on the years before it is chosen, and its signals of that year are judged. The
    trial file is written for `newest` (the version that asked for the judgement; the last one given by default).
    tfs: label -> that version's chart when it is not `tf` (a version on another timeframe, L14.1).
    """
    if not versions:
        raise ValueError("no versions to judge")
    newest = newest or list(versions)[-1]
    tf_of = {label: (tfs or {}).get(label, tf) for label in versions}
    ids = {label: _one_version(sigs) for label, sigs in versions.items()}
    idea, version, parent = ids[newest]
    links = trials.parents(trials.load())
    families = {trials.root_of(i, links, p) for i, _, p in ids.values()}
    if len(families) != 1:
        raise ValueError(f"walk-forward judges the versions of ONE idea family, not {sorted(families)}")
    m1, data = _bars(m1, config.TUNE_YEARS)
    n_tries = len(trials.count(idea, parent, extra=(idea, version)).family_versions)
    segments, choices = [], []
    for fold in splits.walk_forward_folds():
        choose_m1 = m1.loc[f"{fold.choose_years[0]}":f"{fold.choose_years[-1]}"]
        judge_m1 = m1.loc[f"{fold.judge_year}":f"{fold.judge_year}"]
        if choose_m1.empty or judge_m1.empty:
            raise ValueError(f"walk-forward needs bars in {fold.choose_years} and {fold.judge_year}")
        tried = {}
        for label, sigs in versions.items():
            seg = make_segment("choose", fold.choose_years, tf_of[label], sigs, choose_m1, strict=False)
            tried[label] = closed_trades(replay(seg, seg.signals))
        pick = stats.choose_version(tried, n_tries=n_tries, years=fold.choose_years)
        choices.append({"judge_year": fold.judge_year, **pick})
        segments.append(make_segment(str(fold.judge_year), (fold.judge_year,), tf_of[pick["chosen"]],
                                     versions[pick["chosen"]], judge_m1, strict=False))
    judged_on = (f"walk-forward inside {splits.years_text(config.TUNE_YEARS)}: "
                 + ", ".join(f"{c['judge_year']} (version {c['chosen']}, chosen on {c['years'][0]}-{c['years'][-1]})"
                             for c in choices)
                 + f" — the check years {splits.years_text(config.CHECK_YEARS)} are not used "
                 f"({splits.status(idea, version, parent)})")
    return _judge(segments, tf=tf, key=key, idea=idea, version=version, parent=parent, judged_on=judged_on,
                  n_random=n_random, choices=choices, data=data, record=record, note=note, control=control)


# ================================================================================================================
# the judgement
# ================================================================================================================

def _judge(segments: list[Segment], *, tf, key, idea, version, parent, judged_on, n_random, choices, data, record,
           note, control) -> Report:
    n_random = config.RANDOM_RUNS if n_random is None else n_random
    years = sorted({y for seg in segments for y in seg.years})
    reps = [replay(seg, seg.signals) for seg in segments]
    per_seg = [closed_trades(r) for r in reps]
    trades = pooled(per_seg)
    summ = stats.summary(trades)
    by_year = stats.per_year(trades, years)
    wb = stats.without_best(trades)
    seed = random_entries.seed_of(segments)
    p = stats.bootstrap_p(trades["net_inr"], stats.week_ids(trades["exit_time"]), seed=seed) if len(trades) else 1.0

    rand, rand_settings = random_entries.runs(segments, per_seg, n_random, seed=seed)
    rand_sum = random_entries.summary(rand, summ["per_trade"])
    fake = fake_levels.runs(segments)
    fake_sum = fake_levels.summary(fake, summ["per_trade"])

    noise = random_entries.pooled_noise(rand)
    noise_from = f"the {n_random} random-entry runs ({noise.size:,} trades)"
    if noise.size < config.MIN_TRADES:
        noise, noise_from = trades["net_inr"].to_numpy(dtype=float), "the idea's own trades"
    weeks = len(np.unique(stats.week_ids(trades["exit_time"]))) if len(trades) else 1
    if noise.size >= 2:
        ptable = power.table(noise, summ["trades"], summ["trades"] / max(weeks, 1), seed=seed % (1 << 31),
                             noise_from=noise_from)
    else:
        ptable = {"rows": {}, "own_n": summ["trades"], "needed_edge_caught": 0.0, "smallest_edge_seen": None,
                  "noise_sd_inr": None, "noise_from": "nothing: no trades", "trades_per_week": 0, "sims": 0,
                  "resamples": 0}
    seen = prior.already_seen(key, years)
    ev = verdict.Evidence(trades=summ["trades"], per_trade=summ["per_trade"], p=p, random=rand_sum, fake=fake_sum,
                          without_best=wb, seen=seen.seen, needed_edge_caught=ptable["needed_edge_caught"],
                          judged_on=judged_on)
    answer = verdict.decide(ev)
    counts = trials.count(idea, parent, extra=(idea, version))
    tester_sums = [summarize(r) for r in reps]
    numbers = {
        "summary": summ, "per_year": by_year, "worst_year": stats.worst_year(by_year), "without_best": wb,
        "a_year": stats.a_year(summ, years),
        "p": round(p, 5), "random": rand_sum, "fake_levels": fake_sum,
        "lots": sorted({s.lots for seg in segments for s in seg.signals}),
        "skipped": sum(t["setups_skipped"] for t in tester_sums),
        "open_at_end": sum(t["open_at_end"] for t in tester_sums),
        "ambiguous": sum(t["ambiguous"] for t in tester_sums),
        "path": path_counts(reps),                 # the report (L10): came to the level, skipped, cancelled
        "outcomes": outcome_counts(reps),          # ... in profit / at a loss / still open; target or stop first
        "extra_lines": stats.extra_lines(reps, years),
        "trades_table": trade_rows(reps),          # every trade (his #107 / #114)
        "segments": [{"label": seg.label, "signals": len(seg.signals), "first_bar": str(seg.book.first),
                      "last_bar": str(seg.book.last), "tester": t} for seg, t in zip(segments, tester_sums)],
    }
    path = None
    if record:
        settings = {
            "judged_on": judged_on, "timeframe": tf, "years": years, "idea_key": key.as_dict(),
            "rules_sha256": verdict.RULES_SHA256, "random_entries": rand_settings,
            "fake_level_steps": list(config.FAKE_LEVEL_STEPS), "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": seed, "trades_at_once": config.TRADES_AT_ONCE,
            "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
            "clock": "MT5 server time",
            "signals_sha256": trial_files.sha256_of([s.model_dump(mode="json") for seg in segments
                                                     for s in seg.signals]),
            "code_sha256": _code_hash(),
        }
        near = [{"idea_key": r.idea_key, "what": prior.describe(r.idea_key), "tries": r.tries,
                 "years": list(r.years), "blocks": sorted(r.blocks), "entry": r.entry, "source": r.source}
                for r in prior.closest(key, seen)]
        hits = [{"idea_key": r.idea_key, "what": prior.describe(r.idea_key), "tries": r.tries,
                 "years": list(r.years)} for r in seen.rows]
        wider = [r for r in seen.relatives if r.blocks <= key.blocks]
        result = {"answer": answer.as_dict(), **numbers, "power": ptable,
                  "already_seen": {**seen.as_dict(), "rows_detail": hits, "closest": near,
                                   "wider": {"rows": len({r.idea_key for r in wider}),
                                             "tries": sum({r.idea_key: r.tries for r in wider}.values())}},
                  "choices": choices, "counts": counts.as_dict(), "check_opened": splits.opened_count()}
        path = trial_files.record(kind=CONTROL_KIND if control else "referee", idea_id=idea, version_id=version, parent_idea=parent,
                                  settings=settings, data=data, result=result, note=note)
    return Report(idea, version, tf, judged_on, answer, numbers, counts, seen, ptable, choices,
                  project.summary(), path, control)


# ================================================================================================================
# helpers
# ================================================================================================================

def _one_version(signals: list[Signal]) -> tuple[str, str, str | None]:
    signals = list(signals)
    if not signals:
        raise ValueError("no signals to judge")
    who = {(s.idea_id, s.version_id, s.parent_idea) for s in signals}
    if len(who) != 1:
        raise ValueError(f"one judgement is ONE version of ONE idea; these signals come from {sorted(map(str, who))}")
    return who.pop()


def _bars(m1: pd.DataFrame | None, years) -> tuple[pd.DataFrame, dict]:
    if m1 is None:
        start, end = splits.window(years)
        m1 = bars.load_bars("1m", start, end)
        data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
                "gold_m1_content_sha256": bars.m1_fingerprint()}
    else:
        guards.check_dates(m1)
        bad = sorted(set(pd.DatetimeIndex(m1.index).year) - set(years))
        if bad:
            raise ValueError(f"the bars given reach {bad}, outside the judged years {list(years)}")
        cols = [c for c in ("open", "high", "low", "close", "spread") if c in m1.columns]
        data = {"prices": "bars given to the referee (hand-made or cut)",
                "frame_sha256": hashlib.sha256(pd.util.hash_pandas_object(m1[cols], index=True)
                                               .to_numpy().tobytes()).hexdigest()}
    if m1.empty:
        raise ValueError(f"no bars in {list(years)}")
    return m1, data


def _code_hash() -> str:
    here = Path(__file__).parent
    tester = here.parent / "tester"
    return trial_files.code_sha256([here / f for f in REFEREE_FILES] + [tester / f for f in TESTER_FILES])


def _rel(path: Path) -> str:
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
