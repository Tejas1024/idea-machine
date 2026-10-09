"""The time split (L15.2): each test year is guessed by models that learned ONLY from the charts before it.

    from machine.eyes import split
    for f in split.folds(lab):        # f.year, f.train (row positions), f.test (row positions)
        ...

A chart belongs to the year of its first candle. Year Y's models learn from the charts whose LAST candle is
config.EYES_PURGE_DAYS days or more before the first candle of Y's first chart - so a chart that runs over the new
year, or ends a few days before it, teaches nothing to the year it touches (lesson B1: nothing from the future).
config.EYES_TEST_YEARS = 2018, 2019, 2020 (the referee's own years): 2016-2017 only teach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from machine import config


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray            # row positions in the label table
    test: np.ndarray
    learn_until: pd.Timestamp    # the last candle of a teaching chart is before this

    def text(self) -> str:
        return f"{self.year}: learns from {len(self.train)} answers (charts ending before " \
               f"{self.learn_until:%d %b %Y}), guesses {len(self.test)}"


def folds(lab: pd.DataFrame, years: tuple[int, ...] | None = None, purge_days: int | None = None) -> list[Fold]:
    years = tuple(years or config.EYES_TEST_YEARS)
    purge = pd.Timedelta(days=config.EYES_PURGE_DAYS if purge_days is None else purge_days)
    first = pd.to_datetime(lab["first_candle"])
    last = pd.to_datetime(lab["last_candle"])
    out = []
    for y in years:
        test = np.flatnonzero(first.dt.year.to_numpy() == y)
        if test.size == 0:
            raise ValueError(f"no chart starts in {y}: nothing to guess")
        until = first.iloc[test].min() - purge
        train = np.flatnonzero((last < until).to_numpy())
        if np.intersect1d(train, test).size:
            raise AssertionError("a chart both teaches and is guessed")      # cannot happen: kept as the guard
        if train.size == 0:
            raise ValueError(f"nothing to learn from before {y}")
        out.append(Fold(int(y), train, test, until))
    return out


def tested(fs: list[Fold]) -> np.ndarray:
    """Every row guessed, once each, in row order."""
    t = np.concatenate([f.test for f in fs])
    if len(np.unique(t)) != len(t):
        raise AssertionError("a chart guessed twice")
    return np.sort(t)
