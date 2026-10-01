"""Choosing ONE version on the tune years (L10; step 1 of the path in judge.py).

    from machine.referee import tune
    t = tune.run_versions({"v-abc": sigs_a, "v-def": sigs_b}, tf="5m", about={"v-abc": "U1=A ...", ...})
    t.chosen                       # the version with the best Deflated Sharpe on 2016-2020
    print(t.text())

Every version runs through the tester on 2016-2020 - one trial file each (kind "tester", likhith.md R14), so every
version is a counted try. Then the referee chooses with stats.choose_version: the highest Deflated Sharpe (the chance
a version's edge is real, allowing for every version the family tried) wins; PBO (the chance that choosing on the past
picked luck) is reported next to it. ONE more trial file, kind "referee-tune", holds every version's numbers, the
choice, and the chosen version's full numbers on 2016-2020 - every trade too (his #107 / #114) - for the report
(machine/report/build.py reads only the referee's files). The check years (2021-2023) are never read here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from machine import config
from machine import trials as trial_files
from machine.referee import splits, stats, trials
from machine.referee.judge import _bars, _code_hash, _one_version
from machine.referee.replays import closed_trades, outcome_counts, path_counts, trade_rows
from machine.tester import run as tester_run
from machine.tester.models import Signal

TUNE_KIND = "referee-tune"


@dataclass
class Tuned:
    idea_id: str
    parent_idea: str | None
    tf: str
    chosen: str
    choice: dict                      # stats.choose_version's answer
    versions: dict[str, dict]         # every version: what it is, its numbers on the tune years
    numbers: dict                     # the chosen version's full numbers on the tune years
    trial_path: Path | None

    def text(self) -> str:
        inr = _inr
        lines = [f"TUNE  {self.idea_id} · {len(self.versions)} versions on {splits.years_text(config.TUNE_YEARS)} · "
                 f"{self.tf} bars · MT5 server time · ₹40 per position per 0.01 lot, all-in (LEARNED #54, #115)",
                 "  version | what it is | trades | ₹ a trade | ₹ after cost | chance its edge is real (Deflated "
                 "Sharpe, allowing for every version tried)"]
        for v, d in self.versions.items():
            mark = "  <- chosen" if v == self.chosen else ""
            lines.append(f"  {v} | {d['about']} | {d['trades']} | {inr(d['per_trade'])} | {inr(d['net_inr'])} | "
                         f"{d['deflated_sharpe']:.2f}{mark}")
        pbo = self.choice.get("pbo")
        if pbo:
            lines.append(f"The chance the choice was luck (PBO): {100 * pbo['pbo']:.0f} in 100 ({pbo['ways']} ways of "
                         f"cutting the weeks in half)")
        if self.trial_path is not None:
            lines.append(f"Trial file: {self.trial_path.relative_to(config.ROOT).as_posix()}"
                         if self.trial_path.is_relative_to(config.ROOT) else f"Trial file: {self.trial_path}")
        return "\n".join(lines)


def _inr(x) -> str:
    if x is None:
        return "no trades"
    return f"-₹{-x:,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def signals_sha256(signals: list[Signal]) -> str:
    """A fingerprint of a version's signals that does not depend on their order (sorted by the moment known, id)."""
    ordered = sorted(signals, key=lambda s: (s.known_at, s.id))
    return trial_files.sha256_of([s.model_dump(mode="json") for s in ordered])


def run_versions(versions: dict[str, list[Signal]], *, tf: str, about: dict[str, str] | None = None,
                 readings: dict[str, dict] | None = None, changes: dict[str, dict] | None = None,
                 tfs: dict[str, str] | None = None, m1: pd.DataFrame | None = None, record: bool = True,
                 note: str | None = None) -> Tuned:
    """Test every version of ONE idea on the tune years, choose one, and record the choice (see the module text).

    versions: version id -> that version's signals, all known inside 2016-2020. about: version id -> what it is
    ("at most 0.25 of a day's range · ..."), shown in the report. readings: version id -> its readings of the idea's
    unclear parts ({"U1": "B", ...}), kept so the chosen one can be compiled again on the check years. changes: version
    id -> the version's own changes to the form (dotted paths: a research version's timeframe or numbers, L14.1), kept
    for the same reason; none = the readings alone make the version. tfs: version id -> its chart when it is not `tf`
    (a version on another timeframe, dimension 1). m1: hand-made bars (tests); real bars otherwise.
    """
    if not versions:
        raise ValueError("no versions to tune")
    about, readings, changes = dict(about or {}), dict(readings or {}), dict(changes or {})
    tf_of = {label: (tfs or {}).get(label, tf) for label in versions}
    ids = {label: _one_version(sigs) for label, sigs in versions.items()}
    ideas = {(i, p) for i, _, p in ids.values()}
    if len(ideas) != 1:
        raise ValueError(f"the tune step chooses between versions of ONE idea, not {sorted(map(str, ideas))}")
    idea, parent = ideas.pop()
    for label, (_, v, _) in ids.items():
        if v != label:
            raise ValueError(f"version {label!r}: its signals say version {v!r}")
    years = config.TUNE_YEARS
    m1, data = _bars(m1, years)                        # refuses bars outside 2016-2020 (the check years stay shut)
    first, last = m1.index[0], m1.index[-1] + pd.Timedelta(minutes=1)
    for label, sigs in versions.items():
        out = [s for s in sigs if not first <= pd.Timestamp(s.known_at) <= last]
        if out:
            raise ValueError(f"version {label}: signal {out[0].id} is known at {out[0].known_at}, outside the tune "
                             f"years {splits.years_text(years)}")
        trials.check_budget(idea, label, parent)       # a new version past the family's budget is not tried

    real = data.get("prices") == config.GOLD_M1.relative_to(config.ROOT).as_posix()
    results, tried = {}, {}
    for label, sigs in versions.items():
        res = (tester_run(sigs, tf=tf_of[label], start=str(years[0]), end=str(years[-1]), record=record,
                          note=f"L10 tune: {about.get(label, label)}") if real
               else tester_run(sigs, tf=tf_of[label], m1=m1, record=record,
                               note=f"L10 tune: {about.get(label, label)}"))
        results[label] = res
        tried[label] = closed_trades(res.replay)
    counts = trials.count(idea, parent)
    choice = stats.choose_version(tried, n_tries=len(counts.family_versions), years=years)
    chosen = choice["chosen"]

    rows = {}
    for label, res in results.items():
        s = res.summary
        rows[label] = {"about": about.get(label, ""), "readings": readings.get(label, {}),
                       "changes": changes.get(label, {}), "timeframe": tf_of[label],
                       "signals": s["signals"], "signals_sha256": signals_sha256(versions[label]),
                       "trades": s["trades"],
                       "net_inr": s["net_inr_total"], "per_trade": s["net_inr_per_trade"],
                       "per_year": s["per_year"], "worst_year": s["worst_year"],
                       "deflated_sharpe": choice["versions"][label]["deflated_sharpe"],
                       "sharpe": choice["versions"][label]["sharpe"],
                       "tester_trial": None if res.trial_path is None else _rel(res.trial_path)}
    rep = results[chosen].replay
    ct = tried[chosen]
    summ = stats.summary(ct)
    by_year = stats.per_year(ct, years)
    numbers = {
        "summary": summ, "per_year": by_year, "worst_year": stats.worst_year(by_year),
        "without_best": stats.without_best(ct), "a_year": stats.a_year(summ, years),
        "path": path_counts([rep]), "outcomes": outcome_counts([rep]),
        "extra_lines": stats.extra_lines([rep], years), "skipped": len(rep.skipped),
        "open_at_end": sum(t.status != "closed" for t in rep.trades),
        "trades_table": trade_rows([rep]),
    }
    path = None
    if record:
        settings = {"years": list(years), "timeframe": tf_of[chosen],
                    "chosen_by": "the highest Deflated Sharpe on the tune years "
                    "(machine.referee.stats.choose_version); PBO reported", "tries_counted": len(counts.family_versions),
                    "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
                    "clock": "MT5 server time", "trades_at_once": config.TRADES_AT_ONCE,
                    "code_sha256": _code_hash()}
        result = {"chosen": chosen, "choice": choice, "versions": rows, **numbers, "counts": counts.as_dict()}
        path = trial_files.record(kind=TUNE_KIND, idea_id=idea, version_id=chosen, parent_idea=parent,
                                  settings=settings, data=data, result=result, note=note)
    return Tuned(idea, parent, tf_of[chosen], chosen, choice, rows, numbers, path)


def latest(idea: str, folder: Path | None = None) -> dict | None:
    """The newest tune trial file of this idea, or None."""
    files = [f for f in trials.load(folder) if f.get("kind") == TUNE_KIND and f.get("idea_id") == idea]
    return files[-1] if files else None


def _rel(path: Path) -> str:
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
