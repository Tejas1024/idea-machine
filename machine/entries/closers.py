"""Which of his entries close a hedged leg (T12 M10), and the checks made while an order of his waits (L19b).

    got = closing_setups(market, book)          # his setups that may close a leg, by settings (M10)
    cancel = fill_checks(entry, signals, market, fp, type1_signals)   # {signal id: the moment its order is cancelled}

M10 (#113: "it can be a triangle false break as well, even zone break as well, trend line entry as well ... whatever
the rules I gave, it can be any of that"): settings which_entries_can_close_a_position = "his5" -> every entry of his
in machine/entries/forms.py marked `closer` (grab + gap, the trendline touch, the zone break, the triangle false break,
the zone entry). Two of them need a word he has not answered yet ("steep" B2-01, "big move" B2-02): they close legs
only when a reading is given (`readings`, e.g. a version that names the number) - never guessed (R13). "type1" = grab
+ gap only (every number before 20 Sep 2026, and L19a's).

The checks while an order waits (the tester has no other way to see them - engine.replay's cancel_by):
  opposite_setup  #98 (Q98 B): "in between ... there is another BOS and fair value gap. It might be the sell trade, so
                  it's not a good idea to enter at the retracement of this zone ... we can exit at this zone retest
                  as an initial target." A zone break's retest order is cancelled the moment an OPPOSITE setup of his
                  Type 1 becomes known before it fills (settings zone_break_opposite_setup_veto, his).
  gap_excuse      #87 at the fill (settings unfilled_gap_is_a_magnet: "checked again when the order fills"): a zone entry
                  that had an unfilled gap past its hedge was taken only because obvious liquidity waited on the
                  target side ("the market always moves towards the recent liquidity"). If price takes EVERY one of
                  those levels before the order fills, the excuse is gone: the order is cancelled when the last one
                  is taken (the old engine's gap_excuse_watch).
Each moment is when the thing became known (a setup's known_at; the close of the candle that took the last level):
nothing later is read, and the Signal itself is not changed (the cut-off-the-future test on signals still holds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from machine import settings
from machine.blocks import fvg as FVG
from machine.blocks import liquidity as LQ
from machine.blocks._base import TF_MINUTES
from machine.entries import forms
from machine.tester.models import Signal

CHECKS = ("opposite_setup", "gap_excuse")


@dataclass
class ClosingSetups:
    setups: list                        # hedging.Setup, sorted by the minute they fill
    signals: int                        # his signals they came from
    which: dict = field(default_factory=dict)       # entry id -> signals used, or why it does not close legs (yet)
    by_entry: dict = field(default_factory=dict)    # entry id -> its signals (for fill checks and the reports)


def closer_entries(readings: dict[str, dict[str, str]] | None = None) -> tuple[list, dict]:
    """([(entry, its reading)], {entry id: why it waits}) for settings which_entries_can_close_a_position."""
    readings = readings or {}
    which = settings.get("which_entries_can_close_a_position")
    if which == "type1":
        return [(forms.TYPE1, {})], {e.id: "settings which_entries_can_close_a_position = 'type1'"
                                     for e in forms.ENTRY_TYPES.values() if e.closer and e is not forms.TYPE1}
    out, waits = [], {}
    for e in forms.ENTRY_TYPES.values():
        if not e.closer:
            continue
        open_parts = [u.id for u in e.idea().unclear if u.answer is None]
        rd = readings.get(e.id, {})
        missing = [p for p in open_parts if p not in rd]
        if missing:
            waits[e.id] = f"waits for his answer to {e.open_word or ', '.join(missing)} - or a version that names it"
            continue
        out.append((e, rd))
    return out, waits


def closing_setups(market, book, *, readings: dict[str, dict[str, str]] | None = None,
                   signals: dict[str, list[Signal]] | None = None) -> ClosingSetups:
    """His setups that may close a hedged leg (M10) on this market, filled by the tester's own rules (with the checks
    their orders get while they wait). signals: already-compiled signals by entry id (not compiled again)."""
    from machine.ideas.compile import compile_idea
    from machine.tester import hedging
    got, waits = closer_entries(readings)
    by_entry = dict(signals or {})
    for e, rd in got:
        if e.id not in by_entry:
            by_entry[e.id] = compile_idea(e.idea(), market, readings=rd).signals
    fp = settings.fingerprint(settings.resolve({}))
    type1 = by_entry.get(forms.TYPE1.id)
    if type1 is None:
        type1 = compile_idea(forms.TYPE1.idea(), market).signals
    sig, cancel = [], {}
    for e, _ in got:
        sig += by_entry[e.id]
        cancel.update(fill_checks(e, by_entry[e.id], market, fp, type1))
    setups = hedging.setups_from(_unique(sig), book, cancel)
    setups.sort(key=lambda x: (x.fill_i, x.known, x.id))
    which = {e.id: len(by_entry[e.id]) for e, _ in got}
    which.update(waits)
    return ClosingSetups(setups, len(sig), which, {e.id: by_entry[e.id] for e, _ in got})


def _unique(sig: list[Signal]) -> list[Signal]:
    seen, out = set(), []
    for s in sig:
        if s.id not in seen:
            seen.add(s.id)
            out.append(s)
    return out


# ================================================================================================================
# the checks while an order waits
# ================================================================================================================

def _ns(t) -> int:
    return pd.Timestamp(t).as_unit("ns").value


def fill_checks(entry: forms.EntryType, signals: list[Signal], market, fp: str,
                type1: list[Signal] | None = None) -> dict[str, int]:
    """{signal id: the moment (ns) its waiting order is cancelled} for the checks this entry type has."""
    out: dict[str, int] = {}
    if "opposite_setup" in entry.fill_checks and settings.get("zone_break_opposite_setup_veto"):
        out.update(opposite_setup(signals, type1 or []))
    if "gap_excuse" in entry.fill_checks and settings.get("unfilled_gap_is_a_magnet"):
        out.update(gap_excuse(signals, market, fp))
    return out


def opposite_setup(signals: list[Signal], others: list[Signal]) -> dict[str, int]:
    """#98 B: each limit is cancelled when the first of `others` (his Type 1 setups) pointing the OTHER way becomes
    known after it (strictly later)."""
    times = {d: np.sort(np.array([_ns(s.known_at) for s in others if s.direction == d], dtype=np.int64))
             for d in ("buy", "sell")}
    out = {}
    for s in signals:
        if s.entry_type != "limit":
            continue
        t = times["sell" if s.direction == "buy" else "buy"]
        k = int(np.searchsorted(t, _ns(s.known_at), "right"))
        if k < len(t):
            out[s.id] = int(t[k])
    return out


def gap_excuse(signals: list[Signal], market, fp: str) -> dict[str, int]:
    """#87 at the fill: a setup with an unfilled gap past its hedge, kept because his obvious liquidity waited past
    the entry on the target side (within settings next_liquidity_max_adr - the same look as the do-not-enter rule
    unfilled_gap_past_hedge), is cancelled at the close of the candle that takes the LAST of those levels."""
    m5 = market.bars("5m")
    life = market.liquidity(fp).life
    obvious = tuple(dict.fromkeys(k.split("_", 1)[0] for k in settings.get("obvious_liquidity_kinds")))
    fam = life["kind"].str.split("_", n=1).str[0]
    ruler = market.ruler("5m", fp)
    step = pd.Timedelta(minutes=TF_MINUTES["5m"])
    out = {}
    for s in signals:
        if s.entry_type != "limit" or s.stop is None:
            continue
        i = market.closed_by("5m", s.known_at)
        r = ruler[i] if i >= 0 else math.nan
        if not math.isfinite(r):
            continue
        g = FVG.unfilled_past(m5, i, s.direction, float(s.stop), float(r))
        if g is None:
            continue
        side = "high" if s.direction == "buy" else "low"
        far = float(settings.get("next_liquidity_max_adr")) * r
        rows = LQ.alive(life[fam.isin(obvious)], i)
        rows = rows[(rows["side"] == side) & (rows["target_when"] != "higher_timeframe_against")]
        px = rows["price"].to_numpy(float)
        ok = (px > s.entry_price) if side == "high" else (px < s.entry_price)
        ok &= np.abs(px - s.entry_price) <= far
        rows = rows[ok]
        if not len(rows):
            continue                                    # no excuse: the do-not-enter rule already dropped it
        taken = rows["taken"].to_numpy(np.int64)
        if (taken < 0).any():
            continue                                    # one of them is never taken: the excuse stands
        last = int(taken.max())
        out[s.id] = _ns(m5.index[last] + step)          # the close of the candle that took the last one
    return out
