"""Fair value gaps (L6.1): the gap, where he buys in it, its first visit, and the gaps of the breaking move.

The gap (machine/rules.py PARTS "fvg_shape"; strategy.md section 2): "3-candle imbalance; bullish = 1st-high ↔
3rd-low." A BUY gap is left by a move up: the 1st candle's high is below the 3rd candle's low; the gap runs from that
high (lo) to that low (hi). A SELL gap is the mirror. Any size (#62): a tiny gap is still a gap.
Known when the 3rd candle closes.

His fixed rules about it (machine/rules.py), each used here:
  where_in_the_gap_he_buys (#65)  "I'll enter at the beginning of fvg": the NEAR edge - the edge price meets first on
                                  the way back. Buy: the top of the gap (hi). Sell: the bottom (lo).
  only_the_first_visit_back_counts (#68)  "it's not about the time uh, it's about whether it is already filled or
                                  not". first_visits() finds the first candle after the gap that comes back to its near
                                  edge. No time limit. A later visit never counts.
  which_gap_is_the_entry_gap (#61, #62, #131)  the FIRST gap after the liquidity is taken, any size, but "it only valid
                                  if it forms after taking any liquidity": its 1st candle is the sweep candle or later.
  gap_must_belong_to_the_breaking_move (#62 C, #101)  its 3rd candle closes at most 6 candles after the break.
gaps_of_move() applies the last two; which of the move's gaps is traded is settings "which_gap_of_the_move".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import rules
from machine.blocks._base import bar_close, check_bars, empty, first_hit

GAP_COLUMNS = {"known_at": "datetime64[ns]", "side": "object", "first_bar": "datetime64[ns]",
               "bar": "datetime64[ns]", "lo": "float64", "hi": "float64", "near": "float64", "far": "float64",
               "size": "float64"}
VISIT_COLUMNS = {"known_at": "datetime64[ns]", "side": "object", "gap_bar": "datetime64[ns]", "bar": "datetime64[ns]",
                 "near": "float64", "through": "bool"}


def _near_far(side: str, lo, hi):
    near_edge = rules.get("where_in_the_gap_he_buys")
    if near_edge != "near":                      # his #65; rules.py can never hold anything else
        raise RuntimeError(f"rules.py where_in_the_gap_he_buys = {near_edge!r}")
    return (hi, lo) if side == "buy" else (lo, hi)


def gaps(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Every 3-candle gap: side ("buy" / "sell"), first_bar (1st candle), bar (3rd candle), lo, hi, near (his entry
    edge), far, size ($). known_at = the 3rd candle's close."""
    check_bars(bars, tf, ("high", "low"))
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    n = len(h)
    if n < 3:
        return empty(GAP_COLUMNS)
    closes = bar_close(bars.index, tf)
    j = np.arange(2, n)
    parts = []
    for side, mask, g_lo, g_hi in (("buy", h[j - 2] < lo[j], h[j - 2], lo[j]),
                                   ("sell", lo[j - 2] > h[j], h[j], lo[j - 2])):
        jj, a, b = j[mask], g_lo[mask], g_hi[mask]
        near, far = _near_far(side, a, b)
        parts.append(pd.DataFrame({"known_at": closes[jj], "side": side, "first_bar": bars.index[jj - 2],
                                   "bar": bars.index[jj], "lo": a, "hi": b, "near": near, "far": far,
                                   "size": b - a}))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "side"], kind="stable", ignore_index=True)


def first_visits(bars: pd.DataFrame, tf: str, gap_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each gap, price's FIRST visit back: the first candle after the gap's 3rd candle that reaches its near edge
    (buy: low <= near; sell: high >= near). through = that candle also reached the far edge (the whole gap filled).
    known_at = that candle's close. A gap never visited has no row. No time limit (#68)."""
    check_bars(bars, tf, ("high", "low"))
    g = gaps(bars, tf) if gap_table is None else gap_table
    if not len(g):
        return empty(VISIT_COLUMNS)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    pos = bars.index.get_indexer(pd.DatetimeIndex(g["bar"]))
    if (pos < 0).any():
        raise ValueError("a gap's bar is not one of these bars")
    closes = bar_close(bars.index, tf)
    parts = []
    for side in ("buy", "sell"):
        m = (g["side"] == side).to_numpy()
        if not m.any():
            continue
        near, far, start = g["near"].to_numpy()[m], g["far"].to_numpy()[m], pos[m] + 1
        if side == "buy":
            k = first_hit(lo, start, near, "<=")
        else:
            k = first_hit(h, start, near, ">=")
        ok = k >= 0
        kk = k[ok]
        through = (lo[kk] <= far[ok]) if side == "buy" else (h[kk] >= far[ok])
        parts.append(pd.DataFrame({"known_at": closes[kk], "side": side,
                                   "gap_bar": g["bar"].to_numpy()[m][ok], "bar": bars.index[kk],
                                   "near": near[ok], "through": through}))
    if not parts:
        return empty(VISIT_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "side", "gap_bar"], kind="stable", ignore_index=True)


def gaps_of_move(bars: pd.DataFrame, tf: str, side: str, sweep_bar, break_bar,
                 gap_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """The gaps of ONE move, in order, for a setup whose liquidity was taken on the candle `sweep_bar` and whose
    structure broke on the candle `break_bar` (open times):
      - same side as the setup;
      - the gap's 1st candle is the sweep candle or later ("only valid if it forms after taking any liquidity", #131);
      - its 3rd candle is at most rules "gap_must_belong_to_the_breaking_move" candles after the break (#101).
    It may form before the break (#62). Each row is looked at when it is known: at the close of the break candle, or
    of its own 3rd candle if that comes later (known_at). Two facts about the candles between the gap and that moment:
      filled_before   price already went all the way through it, to the far edge: the gap is gone ("it's about
                      whether it is already filled or not", #68). It gets no rank.
      visited_before  price already came back to its near edge: the setup is spent (#68) - the setup decides (L7b).
    rank = 0 for the first gap not filled before (his entry gap, #61), 1 for the next ...; -1 = filled before.
    Which one a version trades is settings "which_gap_of_the_move".
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', not {side!r}")
    check_bars(bars, tf, ("high", "low"))
    g = gaps(bars, tf) if gap_table is None else gap_table
    s, b = bars.index.get_indexer([pd.Timestamp(sweep_bar), pd.Timestamp(break_bar)])
    if s < 0 or b < 0:
        raise KeyError("sweep_bar and break_bar must be bars of these bars")
    if b < s:
        raise ValueError("the break cannot come before the sweep")
    max_after = int(rules.get("gap_must_belong_to_the_breaking_move"))
    first = bars.index.get_indexer(pd.DatetimeIndex(g["first_bar"]))
    third = bars.index.get_indexer(pd.DatetimeIndex(g["bar"]))
    keep = (g["side"].to_numpy() == side) & (first >= s) & (third <= b + max_after)
    out = g.loc[keep].copy()
    out = out.sort_values("bar", kind="stable", ignore_index=True)
    j = bars.index.get_indexer(pd.DatetimeIndex(out["bar"]))
    look = np.maximum(j, b)                           # the candle whose close makes this row known
    closes = bar_close(bars.index, tf)
    out["known_at"] = closes[look]
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    near, far = out["near"].to_numpy(), out["far"].to_numpy()
    if side == "buy":
        filled = first_hit(lo, j + 1, far, "<=", ends=look + 1) >= 0
        visited = first_hit(lo, j + 1, near, "<=", ends=look + 1) >= 0
    else:
        filled = first_hit(h, j + 1, far, ">=", ends=look + 1) >= 0
        visited = first_hit(h, j + 1, near, ">=", ends=look + 1) >= 0
    out["filled_before"] = filled
    out["visited_before"] = visited
    out["rank"] = np.where(filled, -1, np.cumsum(~filled) - 1)
    out["candles_after_break"] = j - b
    return out


def unfilled_past(bars: pd.DataFrame, i: int, direction: str, hedge: float, ruler: float) -> dict | None:
    """His #87: "market always move towards the unfilled gaps ... even when the market is give a gap up, eventually it
    will come back to that gap up or gap down" - "there was an unfilled gap above ... if I enter there, there is a
    possibility that it will go to that unfilled gap, that will hit my hedge". The nearest gap price has NOT filled by
    candle i that lies past the hedge: above a sell's hedge, below a buy's. Both kinds he named: the 3-candle gap, and
    the gap up / down between one candle's close and the next one's open (settings "open_gap_min_adr" of a day's
    range). Only the part still unfilled past the hedge counts, at least settings "unfilled_gap_min_adr" of a day's
    range, starting within settings "unfilled_gap_beyond_adr" of it, formed in the last settings
    "unfilled_gap_lookback_bars" candles (the old engine's entries.gap_past_hedge, Claude's numbers). Candles up to i
    only. Returns {lo, hi, bar} or None."""
    from machine import settings
    o, h, lo, c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    look = int(settings.get("unfilled_gap_lookback_bars"))
    beyond = float(settings.get("unfilled_gap_beyond_adr")) * ruler
    min_gap = float(settings.get("unfilled_gap_min_adr")) * ruler
    open_gap = float(settings.get("open_gap_min_adr")) * ruler
    k0 = max(2, i - look)
    if i < k0:
        return None
    k = np.arange(k0, i + 1)
    sell = direction == "sell"
    # the highest high (a sell's gap above must be reached from below) / lowest low since each candle, up to i
    seg = h[k0:i + 1] if sell else lo[k0:i + 1]
    since = (np.maximum.accumulate(seg[::-1])[::-1] if sell else np.minimum.accumulate(seg[::-1])[::-1])
    cand = []
    if sell:
        three = h[k] < lo[k - 2]                                   # left by a move down: the candles after k fill it
        cand += [(h[x], lo[x - 2], x + 1, x) for x in k[three]]
        jump = (c[k - 1] - o[k]) >= open_gap                       # a gap down at the open: candle k can start filling
        cand += [(o[x], c[x - 1], x, x) for x in k[jump]]
    else:
        three = lo[k] > h[k - 2]
        cand += [(h[x - 2], lo[x], x + 1, x) for x in k[three]]
        jump = (o[k] - c[k - 1]) >= open_gap
        cand += [(c[x - 1], o[x], x, x) for x in k[jump]]
    best = None
    for g_lo, g_hi, f0, x in cand:
        if sell:
            g_lo = max(g_lo, hedge, since[f0 - k0] if f0 <= i else g_lo)
            away = g_lo - hedge
        else:
            g_hi = min(g_hi, hedge, since[f0 - k0] if f0 <= i else g_hi)
            away = hedge - g_hi
        if g_hi - g_lo >= min_gap and away <= beyond and (best is None or away < best[0]):
            best = (away, float(g_lo), float(g_hi), int(x))
    if best is None:
        return None
    return {"lo": best[1], "hi": best[2], "bar": bars.index[best[3]]}
