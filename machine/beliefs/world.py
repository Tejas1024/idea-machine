"""The judged years, read once, for the beliefs (L13b): the 5-minute candles, a day's range, the clock, and the three
things every belief is built from.

    from machine.beliefs.world import World
    w = World.load((2016, 2017, 2018, 2019, 2020))      # the tune years; the blocks come from w.market (compile.Market)
    w.race(start, up, down, max_bars)                   # which of two prices comes first
    w.reach(start, price, side, max_bars)               # does price come to a price in time
    w.controls(pos, k, rng)                             # the fair comparison's moments: same year, weekday and hour

The world ENDS at the last minute of the last judged year: nothing after it exists, so an event of 2020 can never read a
candle of 2021 (an event too near the end is "unfinished" and not counted). It may START earlier (a warm-up: his levels,
zones and day's range need the weeks before), but events and comparison moments come only from the judged years.
The check years (2021-2023) are refused unless the belief's check-opened.json exists (machine.guards.check_opened) -
machine/beliefs/engine.py opens them, once per belief.

Words used on every page (Claude's, the same for every belief):
  a day          288 five-minute candles (24 hours), the machine's count (settings "level_life_bars" = 2,880 = 10 days)
  a day's range  his 10-day ADR on broker days, at that candle (machine/blocks/adr.py, the "ruler")
  a race         from a candle on: the first candle whose high reaches the upper price or whose low reaches the lower one
                 decides. Both on the same candle = "unclear" (which came first inside it is unknown); neither before
                 the window or the years end = "unfinished". Neither is counted.
  the same weekday and hour   a 5-minute candle of the same year, the same weekday and the same hour of the MT5 server
                 clock (it moves with New York's summer time, like the market's busy hours), never on the same day
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from machine import config, guards, settings
from machine.blocks._base import first_hit
from machine.ideas.compile import Market

DAY = 288                                 # five-minute candles in a day (24 hours)
UP, DOWN, OPEN, BOTH = 1, -1, 0, 2        # race answers: the upper price first, the lower first, neither, both at once
WARMUP_DAYS = 120                         # the weeks before the first judged year that the blocks may read (check years)


class World:
    """The candles of one set of judged years (see the module text)."""

    def __init__(self, market: Market, years):
        self.years = tuple(sorted(int(y) for y in years))
        self.market = market
        self.fp = settings.fingerprint()
        m5 = market.bars("5m")
        last = pd.Timestamp(f"{self.years[-1]}-12-31 23:59")
        if len(m5) and m5.index[-1] > last:
            raise ValueError(f"the world of {self.years} must end by {last}: nothing after the judged years may exist")
        self.m5 = m5
        self.t = m5.index
        self.n = len(m5)
        self.open = m5["open"].to_numpy(float)
        self.high = m5["high"].to_numpy(float)
        self.low = m5["low"].to_numpy(float)
        self.close = m5["close"].to_numpy(float)
        self.ruler = market.ruler("5m", self.fp)
        self.year = self.t.year.to_numpy()
        self.weekday = self.t.dayofweek.to_numpy()
        self.hour = self.t.hour.to_numpy()
        self.day = (self.t.normalize().as_unit("s").asi8 // 86_400).astype(np.int64)
        self.judged = np.isin(self.year, self.years)
        self.first = int(np.argmax(self.judged)) if self.judged.any() else self.n
        ok = self.judged & np.isfinite(self.ruler) & (self.ruler > 0)
        keys = self._key(np.arange(self.n))
        order = np.argsort(keys[ok], kind="stable")
        pos = np.flatnonzero(ok)[order]
        k = keys[pos]
        cut = np.flatnonzero(np.diff(k)) + 1
        self._groups = {int(g[0]): p for g, p in zip(np.split(k, cut), np.split(pos, cut)) if len(g)}
        self._memo: dict = {}

    def memo(self, key, make):
        """A table several beliefs share (the takes and their fake takes ...), made once per world."""
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def rng(self, name: str) -> np.random.Generator:
        """A generator seeded by a name and the judged years: the same draws on every PC."""
        h = hashlib.sha256(f"{name}|{list(self.years)}".encode()).hexdigest()
        return np.random.default_rng(int(h[:12], 16))

    # ---- making one ------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, years, *, opened: list[tuple[str, str]] | None = None) -> "World":
        """Real candles for these years (and a warm-up before them). A check year is read only for beliefs that opened
        it: `opened` = [(belief, version)], each with its check-opened.json naming that version - else
        guards.CheckYearsClosedError."""
        years = tuple(sorted(int(y) for y in years))
        hidden = sorted(set(years) & set(config.CHECK_YEARS))
        if hidden:
            if not opened:
                raise guards.CheckYearsClosedError("the check years are read only for beliefs that opened them")
            for idea, version in opened:
                guards.check_opened(idea, version=version)
        start = pd.Timestamp(f"{years[0]}-01-01") - pd.Timedelta(days=WARMUP_DAYS)
        start = max(start, pd.Timestamp(f"{config.TUNE_YEARS[0]}-01-01"))
        return cls(Market.load(start, f"{years[-1]}-12-31 23:59"), years)

    @classmethod
    def from_m1(cls, m1: pd.DataFrame, years) -> "World":
        """Made-up or cut 1-minute candles (tests): the bars the MT5 way (compile.Market.from_m1), the 1-minute ones
        kept for the tester."""
        return cls(Market.from_m1(m1, tfs=("1m", "5m", "1h", "1d", "1w")), years)

    # ---- the clock -------------------------------------------------------------------------------------------------
    def _key(self, pos) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.int64)
        return (self.year[pos].astype(np.int64) * 1000 + self.weekday[pos] * 100 + self.hour[pos]).astype(np.int64)

    def at_or_after(self, times) -> np.ndarray:
        """The candle that opens at or after each moment (n = none): where anything known at that moment can start."""
        return np.searchsorted(self.t.as_unit("ns").asi8, pd.DatetimeIndex(pd.to_datetime(times)).as_unit("ns").asi8,
                               "left")

    def in_years(self, pos) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.int64)
        return (pos >= 0) & (pos < self.n) & self.judged[np.clip(pos, 0, max(self.n - 1, 0))]

    # ---- the three tools -------------------------------------------------------------------------------------------
    def race(self, start, up, down, max_bars) -> tuple[np.ndarray, np.ndarray]:
        """For each query: which price comes first from candle `start` on (start included), within max_bars candles:
        UP (a high reaches `up`), DOWN (a low reaches `down`), BOTH (on the same candle - unclear) or OPEN (neither in
        time, or the years ended first). Returns (answer, the deciding candle or -1)."""
        start = np.asarray(start, dtype=np.int64)
        up = np.broadcast_to(np.asarray(up, dtype=float), start.shape)
        down = np.broadcast_to(np.asarray(down, dtype=float), start.shape)
        end = np.minimum(start + np.broadcast_to(np.asarray(max_bars, dtype=np.int64), start.shape), self.n)
        u = first_hit(self.high, start, up, ">=", ends=end)
        d = first_hit(self.low, start, down, "<=", ends=end)
        ans = np.full(start.shape, OPEN, np.int64)
        bar = np.full(start.shape, -1, np.int64)
        only_u = (u >= 0) & ((d < 0) | (u < d))
        only_d = (d >= 0) & ((u < 0) | (d < u))
        both = (u >= 0) & (u == d)
        ans[only_u], bar[only_u] = UP, u[only_u]
        ans[only_d], bar[only_d] = DOWN, d[only_d]
        ans[both], bar[both] = BOTH, u[both]
        return ans, bar

    def reach(self, start, price, side, max_bars) -> np.ndarray:
        """Does price come to `price` within max_bars candles from `start` (included)? 1.0 yes, 0.0 no (the whole window
        was there), NaN (the years ended before the window did - not counted). side "high": a high at or above it;
        "low": a low at or below it."""
        start = np.asarray(start, dtype=np.int64)
        price = np.broadcast_to(np.asarray(price, dtype=float), start.shape)
        high = np.broadcast_to(np.asarray(side) == "high", start.shape)
        max_bars = np.broadcast_to(np.asarray(max_bars, dtype=np.int64), start.shape)
        end = start + max_bars
        out = np.full(start.shape, np.nan)
        for m, arr, op in ((high, self.high, ">="), (~high, self.low, "<=")):
            if m.any():
                hit = first_hit(arr, start[m], price[m], op, ends=np.minimum(end[m], self.n))
                out[m] = np.where(hit >= 0, 1.0, np.where(end[m] <= self.n, 0.0, np.nan))
        return out

    def controls(self, pos, k: int, rng: np.random.Generator) -> np.ndarray:
        """k comparison candles for each candle in `pos`: the same year, weekday and hour, never the same day (-1 when
        that group has too few other days). Drawn with a seeded generator, so every PC draws the same ones."""
        pos = np.asarray(pos, dtype=np.int64)
        out = np.full((len(pos), k), -1, np.int64)
        keys = self._key(pos)
        for q, (p, key) in enumerate(zip(pos, keys)):
            g = self._groups.get(int(key))
            if g is None:
                continue
            other = g[self.day[g] != self.day[p]]
            if len(other) >= k:
                out[q] = other[rng.choice(len(other), size=k, replace=False)]
            elif len(other):
                out[q, :len(other)] = other
        return out
