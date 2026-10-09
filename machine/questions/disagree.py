"""Where the readings of one unclear part TRADE differently (L9b step 1). ClarifyGPT's idea (arXiv 2310.10996): run
every reading, and ask only where their answers differ.

    from machine.questions import disagree
    d = disagree.compare(form, market, "U2")          # TUNE years, signals only - no money, no fills, no exits
    d.agree                                           # True: 95% or more of the signals are the same -> no question
    d.moments[:4]                                     # the moments where they differ, most different first

How:
  1. Each reading of the part is compiled on the TUNE years (config.TUNE_YEARS); every OTHER open part is held at its
     first reading (said in the question). Only SIGNALS: the orders the rule makes. Never the tester, never money
     (lesson B16: meaning is chosen before money is seen). Outcomes are never looked at either.
  2. Two readings trade the same at a signal when it is the same order: direction, moment, entry, stop, targets.
     share_same = signals the same in every reading / all different signals of all readings. At or above
     config.READINGS_AGREE_SHARE (0.95, the prompt's number) the readings agree: no question.
  3. The signals that are not the same in every reading are joined into MOMENTS: the same direction, known within
     GROUP_BARS candles of the moment's first signal, at most one signal of each reading.
  4. How different a moment's trades are (its score): 1.0 when one reading trades there and another does not; else
     the biggest distance between two readings' entry, stop and targets, as a share of the stop distance (at most 1);
     0.1 when only the moment differs. Most different first; then the older first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from machine import config
from machine.ideas.form import FormError, Idea, open_parts
from machine.tester.models import Signal

GROUP_BARS = 12            # 5-minute candles (one hour): signals this close in time are one moment (Claude's number)


@dataclass
class Moment:
    direction: str
    at: pd.Timestamp                        # the first signal's known_at (server time)
    signals: dict[str, Signal | None]       # reading -> its signal in this moment (None = no trade there)
    score: float
    why: str

    def trading(self) -> list[str]:
        return [r for r, s in self.signals.items() if s is not None]


@dataclass
class Disagreement:
    idea: str
    part: str
    readings: list[str]
    held: dict[str, str]                    # the other open parts, held at these readings
    counts: dict[str, int]                  # signals per reading
    same: int
    different: int
    moments: list[Moment]
    years: tuple[int, ...]
    compiled: dict = field(default_factory=dict, repr=False)   # reading -> machine.ideas.compile.Compiled (traced)

    @property
    def share_same(self) -> float:
        total = self.same + self.different
        return 1.0 if total == 0 else self.same / total

    @property
    def agree(self) -> bool:
        return self.share_same >= config.READINGS_AGREE_SHARE

    def text(self) -> str:
        held = ", ".join(f"{k}={v}" for k, v in self.held.items()) or "none"
        lines = [f"{self.idea} {self.part}: readings {', '.join(self.readings)} on {self.years[0]}-{self.years[-1]} "
                 f"(signals only, no money; other parts held at {held})",
                 "  signals: " + ", ".join(f"{r} {n:,}" for r, n in self.counts.items()),
                 f"  the same in every reading: {self.same:,} of {self.same + self.different:,} = "
                 f"{self.share_same:.0%} -> " + ("the readings AGREE: no question" if self.agree
                                                  else f"they differ at {len(self.moments):,} moments")]
        return "\n".join(lines)


def key(s: Signal) -> tuple:
    """The order itself: two readings trade the same when this is the same."""
    return (s.direction, pd.Timestamp(s.known_at), s.entry_type, s.entry_price, s.stop, tuple(s.targets),
            s.cancel_at, s.first_visit_from, s.first_visit_to)        # his #68 window changes which fills count


def compare_signals(by_reading: dict[str, list[Signal]], *, idea: str = "", part: str = "",
                    held: dict[str, str] | None = None, years: tuple[int, ...] = config.TUNE_YEARS) -> Disagreement:
    """Steps 2-4 on signals already made (the tests give hand-made ones)."""
    readings = list(by_reading)
    keys = {r: {key(s): s for s in sigs} for r, sigs in by_reading.items()}
    everyone = set.intersection(*(set(k) for k in keys.values())) if keys else set()
    rest = sorted(((r, s) for r in readings for k, s in keys[r].items() if k not in everyone),
                  key=lambda x: (pd.Timestamp(x[1].known_at), readings.index(x[0])))
    different = len({key(s) for _, s in rest})
    groups: list[dict] = []
    for r, s in rest:
        t = pd.Timestamp(s.known_at)
        home = None
        for g in reversed(groups):
            if t - g["at"] > pd.Timedelta(minutes=5 * GROUP_BARS):
                break
            if g["direction"] == s.direction and r not in g["signals"]:
                home = g
                break
        if home is None:
            home = {"direction": s.direction, "at": t, "signals": {}}
            groups.append(home)
        home["signals"][r] = s
    moments = []
    for g in groups:
        sigs = {r: g["signals"].get(r) for r in readings}
        score, why = _score(sigs)
        moments.append(Moment(g["direction"], g["at"], sigs, score, why))
    moments.sort(key=lambda m: (-m.score, m.at))
    return Disagreement(idea, part, readings, dict(held or {}), {r: len(v) for r, v in by_reading.items()},
                        len(everyone), different, moments, tuple(years))


def _score(sigs: dict[str, Signal | None]) -> tuple[float, str]:
    have = {r: s for r, s in sigs.items() if s is not None}
    if len(have) < len(sigs):
        no = [r for r in sigs if r not in have]
        return 1.0, f"{', '.join(have)} trade{'s' if len(have) == 1 else ''} here; {', '.join(no)} do{'es' if len(no) == 1 else ''} not"
    best, why = 0.0, "only the moment differs"
    for (ra, a), (rb, b) in combinations(have.items(), 2):
        risk = max(abs((a.entry_price or 0) - (a.stop or a.entry_price or 0)),
                   abs((b.entry_price or 0) - (b.stop or b.entry_price or 0)), 0.01)
        gap = abs((a.entry_price or 0) - (b.entry_price or 0)) + abs((a.stop or 0) - (b.stop or 0))
        if a.targets and b.targets:
            gap += abs(a.targets[0] - b.targets[0])
        d = min(1.0, gap / risk)
        if d > best:
            best, why = d, f"{ra} and {rb} trade here with different prices"
    return (best or 0.1), why


def compare(form: Idea, market, part: str, *, years: tuple[int, ...] = config.TUNE_YEARS) -> Disagreement:
    """Compile every reading of one unclear part (the others held at their first reading) and compare the signals.
    `market` = machine.ideas.compile.Market for the TUNE years only (the caller loads it)."""
    from machine.ideas.compile import compile_idea

    lo, hi = market.window()
    if lo.year < min(years) or hi.year > max(years):
        raise ValueError(f"readings are compared on the TUNE years {years[0]}-{years[-1]} only, not "
                         f"{lo:%Y-%m-%d}..{hi:%Y-%m-%d}")
    parts = {u.id: u for u in open_parts(form)}
    if part not in parts:
        raise FormError(f"{form.id}: {part} is not an open unclear part (open: {sorted(parts) or 'none'})")
    held = {k: u.readings[0].id for k, u in parts.items() if k != part}
    compiled = {}
    for r in parts[part].readings:
        compiled[r.id] = compile_idea(form, market, readings={**held, part: r.id}, trace=True)
    d = compare_signals({r: c.signals for r, c in compiled.items()}, idea=form.id, part=part, held=held, years=years)
    d.compiled = compiled
    return d
