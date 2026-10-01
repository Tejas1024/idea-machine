"""The model's take/skip filter as a new VERSION of the idea (L13.2, part A; likhith.md section D).

    from machine.learner import takeskip
    out = takeskip.make(form, base_signals, guesses, base_version="v-8cc96cf812", spec=spec, tf="5m")

THE RULE: take a setup only when the model made BEFORE its year guesses it will make more than
config.FILTER_ABOVE_INR rupees after the Rs 40 cost; skip the rest. Only the setups of the walk-forward years
(2018-2020) have such a model - there is none for 2016-2017 (nothing before them to learn from), so the version has no
setup there.

It is a different rule, so it is a new VERSION of the same idea: its own version id (from the model's settings, the
rule and the base version), counted against the family's budget (config.VERSION_BUDGET - refused past it), run
through the tester on 2016-2020 (one trial file: a TRY) and judged by the referee by walk-forward inside 2016-2020
(machine/referee/judge.py judge_walk_forward) - its setups of each year were chosen by a model that saw only the
years before, so each judged year is fresh for it. Only when the family's check years were already opened (for the
judged version): while they are closed the idea has not been judged yet, and its filter waits.

It is made ONLY when the model's check passed (machine/learner/run.py): a filter from a model that found nothing
would spend one of the family's tries on noise.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from machine import config
from machine.referee import judge, prior, splits, trials
from machine.tester.models import Signal
from machine.tester.run import run as tester_run


class FilterError(ValueError):
    """No filter version: the check years still closed, the budget used up, no setup taken."""


def version_id(base_version: str, spec: str) -> str:
    """The filter version's id: the base version and the model's whole recipe (settings, rule, code)."""
    return "v-" + hashlib.sha256(f"take-skip|{base_version}|{spec}".encode("utf-8")).hexdigest()[:10]


def rule_text() -> str:
    return (f"take a setup only when the model made before its year guesses more than ₹{config.FILTER_ABOVE_INR:g} "
            "after cost; skip the rest (no model before 2018: no setup in 2016-2017)")


def kept(base_signals: list[Signal], guesses: dict[str, float]) -> list[Signal]:
    """The base version's setups the filter takes (their guess above the line), in their order."""
    return [s for s in base_signals if s.id in guesses and guesses[s.id] > config.FILTER_ABOVE_INR]


def signals(base_signals: list[Signal], guesses: dict[str, float], vid: str) -> list[Signal]:
    """The filter version's signals: the kept setups, each renamed to the new version (nothing else changes)."""
    out = []
    for s in kept(base_signals, guesses):
        new_id = s.id.replace(s.version_id, vid) if s.version_id in s.id else f"{vid}.{s.id}"
        out.append(Signal(**{**s.model_dump(), "id": new_id, "version_id": vid}))
    return out


def make(form, base_signals: list[Signal], guesses: dict[str, float], *, base_version: str, spec: str, tf: str,
         m1: pd.DataFrame | None = None, n_random: int | None = None, record: bool = True, log=print) -> dict:
    """The filter version: its signals, the tester (a try), the referee's walk-forward judgement. Returns what was
    made (the ids of both trial files, the answer) - or refuses with FilterError."""
    idea, parent = form.id, form.parent_idea
    vid = version_id(base_version, spec)
    guessed = [s for s in base_signals if s.id in guesses]
    sigs = signals(base_signals, guesses, vid)
    if not sigs:
        raise FilterError(f"the model guessed a loss for every one of the {len(guessed)} setups of "
                          f"{splits.years_text(config.WALK_FORWARD_JUDGE_YEARS)}: the filter takes no setup")
    status = splits.status(idea, vid, parent)
    if status == "closed":
        raise FilterError(f"the {idea} family's check years are still closed: the idea is judged first (L10), then "
                          "its filter by walk-forward")
    c = trials.check_budget(idea, vid, parent)                # past the family's budget: refused, not tried
    first, last = config.TUNE_YEARS[0], config.TUNE_YEARS[-1]
    note = (f"L13.2: the model's take/skip filter of {idea} {base_version} - {rule_text()}; {len(sigs)} of the "
            f"{len(guessed)} setups of {splits.years_text(config.WALK_FORWARD_JUDGE_YEARS)} taken")
    res = (tester_run(sigs, tf=tf, m1=m1, record=record, note=note) if m1 is not None
           else tester_run(sigs, tf=tf, start=str(first), end=str(last), record=record, note=note))
    log(f"filter version {vid}: {len(sigs)} of {len(guessed)} setups taken; the tester: {res.summary['trades']} "
        f"trades, try {len(c.family_versions) + 1} of {c.budget} in the {c.family} family")
    rep = judge.judge_walk_forward({vid: sigs}, tf=tf, key=prior.key_for_form(form), newest=vid, m1=m1,
                                   n_random=n_random, record=record, note=note)
    log(f"filter version {vid} judged by walk-forward inside {first}-{last}: {rep.word}")
    s = rep.numbers["summary"]
    return {"made": True, "version_id": vid, "rule": rule_text(), "base_version": base_version,
            "setups_guessed": len(guessed), "setups_taken": len(sigs),
            "tester_trial": None if res.trial_path is None else res.trial_path.name,
            "tester_trades": res.summary["trades"],
            "referee_trial": None if rep.trial_path is None else rep.trial_path.name,
            "answer": rep.word, "judged_on": rep.judged_on, "trades": s["trades"], "per_trade": s["per_trade"],
            "net_inr": s["net_inr"], "try": len(trials.count(idea, parent).family_versions), "budget": c.budget}
