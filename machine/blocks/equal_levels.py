"""Equal highs and equal lows (L6.2) - machine/rules.py "what_equal_highs_and_lows_are".

His words:
  strategy.md section 1: "Type 3 Equal highs/lows ⭐ #1 (L2). Later touch must NOT pass reference; exact strongest;
      3>2. A magnet, timing unknown."
  LEARNED #88: "when something comes between the lows, then that is like already swept liquidity ... equal loss should
      be something that shouldn't drawn yet no price uh, should never reach their particular two equals" - and "second
      low should not be lower than the first low".
  LEARNED #89: "it is perfect if it is the exact price of both equal highs or equal lows uh, but slight one or two
      point is okay" - "it can be like two days one week in that level" - "the same applies for uh, daily chat as well
      and hourly chat for every chat".
  LEARNED #90: "One pip or two pips are okay. If it is not equal, if more than that, then it is not a equal highs or
      equal lows." - the SAME few cents on the hour and day charts: bigger candles do not widen it.
  LEARNED #21 (6 Sep): "if it goes to the equal highs first, then it will eventually move to the equal lows" - a PAIR.

The rule (for highs; lows are the mirror):
  - two swing highs (machine/blocks/swings.py: 2 candles each side, strictly) - the FIRST is the reference;
  - the later one is not above the reference, and at most tol_usd ($0.06) below it (rules.py - "2-4 cents yes, 9 cents
    not good"; $0.06 between them is Claude's);
  - nothing trades above the reference between them (#88 A);
  - a dip between them of at least bounce_adr_min (0.08) of a day's range (#89 C bounced 0.09 and is "the perfect
    one"; 0.08 is Claude's, just under it). The day's range is settings "adr_ruler_days" (his ADR, broker days),
    read at the bar that makes the pair known;
  - any time apart (#89); on 5-minute candles the partner is looked for at most settings
    "equal_levels_max_apart_bars" back (two weeks, Claude's), on hour and day candles with no limit (#89);
  - settings "equal_levels_min_apart_bars" (0 = none): the two touches at least this many candles apart. His T3 answers
    (inbox 2026-09-28-1151, "the lows are not proper lows cause they are too near to each other") may mean a minimum
    in time; his #24, #25 (highs 3 and 6 candles apart: yes) and #33 ("yes, partly no") say it is not a hard rule on
    their own. OPEN - a question is queued (plan/work/question-queue/); until he answers, his written rule ("any time
    apart") stands and the minimum is only a version to compare.

One LEVEL per reference: a later swing joins the OLDEST reference it is valid for (its price is the highest of them,
for highs - "the stops sit above both"). The pair is known when the later swing is known (2 candles after it).
A later valid swing is another TOUCH of the same level ("3>2") when price dipped the same amount again since the last
touch - else it is the same dip, not a new touch. Each touch is its own row, so every row is known when it is written.

  equal_levels(bars, tf)  one row per touch from the 2nd on: known_at, level_id, side ("high" / "low"), price (the
                          reference), first_bar, touch_bar, touches (2, 3 ...), prev_touch_bar, bars_apart (candles
                          since the previous touch), gap_usd (reference - this touch, for highs), dip (the dip since
                          the previous touch, as a share of the day's range), exact, ruler.
Taking a level (the sweep) and what is alive at a moment: machine/blocks/liquidity.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import adr as ADR
from machine.blocks._base import bar_close, check_bars, empty
from machine.blocks.swings import swings

COLUMNS = {"known_at": "datetime64[ns]", "level_id": "object", "tf": "object", "side": "object", "price": "float64",
           "first_bar": "datetime64[ns]", "touch_bar": "datetime64[ns]", "touches": "int64",
           "prev_touch_bar": "datetime64[ns]", "bars_apart": "int64", "gap_usd": "float64", "dip": "float64",
           "exact": "bool", "ruler": "float64"}
EPS = 1e-9


def rule() -> dict:
    """His rule's numbers (rules.py) and the two settings that bound the search."""
    r = rules.get("what_equal_highs_and_lows_are")
    return dict(tol=float(r["tol_usd"]), bounce=float(r["bounce_adr_min"]),
                max_apart=int(settings.get("equal_levels_max_apart_bars")),
                min_apart=int(settings.get("equal_levels_min_apart_bars") or 0))


def equal_levels(bars: pd.DataFrame, tf: str, ruler: pd.Series | np.ndarray | None = None,
                 min_apart: int | None = None) -> pd.DataFrame:
    """Every equal high and equal low (see the module text). `ruler` = a day's range on every bar (default: his ADR
    from these bars, machine/blocks/adr.py). `min_apart` overrides settings "equal_levels_min_apart_bars" (a version)."""
    check_bars(bars, tf, ("high", "low"))
    R = rule()
    if min_apart is not None:
        R["min_apart"] = int(min_apart)
    max_apart = R["max_apart"] if tf == "5m" else None          # #89: no limit on the hour and day charts
    if ruler is None:
        ruler = ADR.adr_bars(bars, tf)["ruler"]
    ruler = np.asarray(ruler, dtype=float)
    if len(ruler) != len(bars):
        raise ValueError("the ruler must have one value per bar")
    piv = int(rules.get("swing_lookback"))
    sw = swings(bars, tf)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    closes = bar_close(bars.index, tf)
    parts = []
    for side in ("high", "low"):
        s = sw[sw["side"] == side]
        pos = np.sort(bars.index.get_indexer(pd.DatetimeIndex(s["bar"])))
        # lows are the mirror: work on (-low, -high) and every test reads like the highs one
        up, dn = (h, lo) if side == "high" else (-lo, -h)
        rows = _pairs(pos, up, dn, ruler, piv, R, max_apart)
        if not rows:
            continue
        a, b, prev, touches, dip = (np.array(x) for x in zip(*rows))
        sign = 1.0 if side == "high" else -1.0
        ref, px = sign * up[a], sign * up[b]
        parts.append(pd.DataFrame({
            "known_at": closes[b + piv], "level_id": [f"equal_{side}:{tf}:{t:%Y-%m-%d %H:%M}" for t in bars.index[a]],
            "tf": tf, "side": side, "price": ref, "first_bar": bars.index[a], "touch_bar": bars.index[b],
            "touches": touches.astype("int64"), "prev_touch_bar": bars.index[prev],
            "bars_apart": (b - prev).astype("int64"), "gap_usd": np.round(np.abs(ref - px), 9), "dip": dip,
            "exact": np.abs(ref - px) < EPS, "ruler": ruler[b + piv]}))
    if not parts:
        return empty(COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "side", "first_bar"], kind="stable", ignore_index=True)


def _pairs(pos: np.ndarray, up: np.ndarray, dn: np.ndarray, ruler: np.ndarray, piv: int, R: dict,
           max_apart: int | None) -> list[tuple]:
    """(reference, touch, previous touch, touches, dip) for every touch from the 2nd on, highs-shaped arrays.

    For each later swing b: walk back over earlier swings a, folding in the candles between them - `past` = the highest
    price between a and b (b's own included), `away` = the lowest. An older a can only be valid if it is at least as
    high as `past`, so the walk stops as soon as `past` is more than the tolerance above b.
    """
    tol, bounce, min_apart = R["tol"], R["bounce"], R["min_apart"]
    n = len(up)
    level_of: dict[int, dict] = {}          # reference position -> {"last": last touch, "touches": n}
    out = []
    for k in range(1, len(pos)):
        b = int(pos[k])
        if b + piv >= n or not np.isfinite(ruler[b + piv]):
            continue
        need = bounce * ruler[b + piv]
        vb = up[b]
        past, away, c0 = vb, np.inf, b
        best = None
        for j in range(k - 1, -1, -1):
            a = int(pos[j])
            if max_apart is not None and b - a > max_apart:
                break
            if a + 1 < c0:                                  # the candles between this candidate and the one after it
                past = max(past, float(up[a + 1:c0].max()))
                away = min(away, float(dn[a + 1:c0].min()))
                c0 = a + 1
            if past > vb + tol + EPS:
                break                                       # price went above anything that could still pair
            va = up[a]
            if va + EPS >= past and va - vb <= tol + EPS and va - away + EPS >= need:
                best = a                                    # keep walking: the OLDEST valid reference wins
        if best is None:
            continue
        L = level_of.get(best)
        prev = best if L is None else L["last"]
        dip = (up[best] - float(dn[prev + 1:b].min())) / ruler[b + piv] if b > prev + 1 else 0.0
        if dip + EPS < bounce or b - prev < min_apart:
            continue                                        # the same dip (or too near): not a new touch
        touches = 2 if L is None else L["touches"] + 1
        level_of[best] = {"last": b, "touches": touches}
        out.append((best, b, prev, touches, float(dip)))
    return out
