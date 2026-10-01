"""The halfway reversal (L6b.3) - strategy.md section 1 line 78: "Halfway reversal -> no hedge, re-enter on lower
sweep->BOS->FVG."

"Halfway" is one of his Box 2 words (plan/work/four-boxes.md B2-05): nothing from him says halfway to WHAT. The two
readings written there are both built - never one guessed (R13). Which he means waits for the weekly picture batch:
the question loop (L9b) asks it when an idea uses it, on that idea's own trades (it asks only where the readings trade
differently). What a halfway reversal DOES to a trade (no hedge, re-enter lower) is management: the hedging tester
(L19a) and settings "halfway_reversal" (L19c, off until the word is settled).

  reading "target"  (the old engine's, strategy-understanding section 40e - measured there: +Rs 5 / +Rs 48 a trade on
                    about 45 changed trades) price travelled at least HALF the way from the entry to the target, then
                    turned and came back to the hedge point: that turn is a pullback inside his own move, not the failure
                    a hedge is for. The moment = the candle that reaches the hedge point.
  reading "hedge"   price went at least HALF the way from the entry to the hedge point, then turned and came back to the
                    entry without touching the hedge point (Claude's reading of "then turned" - back to where it
                    started). The moment = the candle back at the entry.
"Half" = 0.5 - his word, not a setting.

  halfway(bars, tf, direction, entry, target, hedge, start, reading)
      -> {known_at, bar, travelled} (travelled = the share of the way to the target, or to the hedge point, price had
         gone when it turned), or None. start = the first candle to read (the one after the fill). Known at the close of
         the candle it names; the candles after it are never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.blocks._base import bar_close, check_bars

HALF = 0.5
READINGS = ("target", "hedge")


def halfway(bars: pd.DataFrame, tf: str, direction: str, entry: float, target: float, hedge: float, start,
            reading: str = "target") -> dict | None:
    """See the module text. direction "buy" / "sell"."""
    if reading not in READINGS:
        raise ValueError(f"reading {reading!r}: one of {', '.join(READINGS)} (four-boxes B2-05)")
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', not {direction!r}")
    check_bars(bars, tf, ("high", "low"))
    up = direction == "buy"
    if (target - entry) * (1 if up else -1) <= 0 or (entry - hedge) * (1 if up else -1) <= 0:
        raise ValueError("a buy needs hedge < entry < target; a sell the mirror")
    s = int(bars.index.searchsorted(pd.Timestamp(start), "left"))
    h = bars["high"].to_numpy(dtype=float)[s:]
    lo = bars["low"].to_numpy(dtype=float)[s:]
    if not len(h):
        return None
    fav, adv = (h, lo) if up else (-lo, -h)              # a sell is the mirror: work on negated prices
    sign = 1.0 if up else -1.0
    e, t, hd = sign * entry, sign * target, sign * hedge
    best = np.maximum.accumulate(fav)                    # the furthest toward the target so far (this candle included)
    worst = np.minimum.accumulate(adv)                   # the furthest toward the hedge point so far
    hit = np.flatnonzero(adv <= hd)                      # candles that reach the hedge point
    first_hit = int(hit[0]) if len(hit) else None
    if reading == "target":
        if first_hit is None:
            return None
        k = first_hit
        # what it had done BEFORE this candle (inside one candle the order of high and low is unknown - assume the
        # worse: the hedge point first, so this candle's own high does not count)
        gone = (best[k - 1] - e) / (t - e) if k > 0 else 0.0
        if gone < HALF:
            return None
        travelled = gone
    else:
        deep = (e - worst) / (e - hd)                    # the share of the way to the hedge point, so far
        back = fav >= e
        ok = np.flatnonzero((np.r_[0.0, deep[:-1]] >= HALF) & back)
        if not len(ok):
            return None
        k = int(ok[0])
        if first_hit is not None and first_hit <= k:
            return None                                  # it reached the hedge point first: a plain hedge
        travelled = float(np.r_[0.0, deep[:-1]][k])
    j = s + k
    return {"known_at": bar_close(bars.index[[j]], tf)[0], "bar": bars.index[j], "travelled": float(travelled)}
