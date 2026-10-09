"""Trendlines (L6b.1): the line, its touches, his touch ENTRY (Type 2), and a line standing in the path (#69).

His words:
  strategy.md section 4: "Trendline: 3 touches, drawn through the wick ends; a touch through the line (even slightly)
      = not a trendline. Any angle can be drawn and is liquidity; the angle only decides the touch entry (section 9
      Type 2) (#92, #94, #112, #116)."
  LEARNED #92: "the three touches should be there. The third touch itself is going above the trend line, that is not a
      good thing" · (D) "now it is like forcefully drawn inside" - a line with candles poking through it between its
      touches is drawn in the wrong place.
  LEARNED #94: "if the third touch is go beyond the um, trend line, even now slightest, it is not considered as a
      trend line."
  LEARNED #112: "you should consider the uh, end points of the wick itself. That, that is the proper trend line, not
      the closing of anything."
  LEARNED #116: the angle does not matter for DRAWING a line - "if it's a straight line top of trend line that it
      would break definitely ... We will wait for it to break or false break and then farewell gap" - and "There should
      be clear uptrend or low trend visible while you are drawing the trend line. If you want to enter at the touch".
  LEARNED #127: "i trendline touch is real and the steep of the line decides that".
  strategy.md section 9, Type 2: "Trendline 3rd/4th touch, no gap needed. Only a steep line inside a clear trend (a
      falling line in a downtrend to sell, a rising line in an uptrend to buy). A flat line is never a touch entry ...
      Hedge a few pips past the line; the hedge moves with the line. The higher timeframe must be with it."
  LEARNED #69 (a line in the path): "if any of the two, like uh, day and hour, both are uh, bullish, then I would avoid
      this trade" · "I would exit at the trend line itself" · "trend line fake out that is also a liquidity".

The line (machine/rules.py "what_a_trendline_is": 3 touches, at most 0.005 of a day's range through):
  - its touches are swing highs (a line through highs: liquidity ABOVE it) or swing lows (liquidity BELOW), his
    2-candle swings (swings.py), drawn through the WICK ENDS (#112). On the 5-minute chart only the swings that
    turned at least settings "trendline_swing_turn_adr" of a day's range (the old engine's zig-zag, Claude's 0.25;
    0 = every swing); on the hour and other charts every swing (the old engine's htf_trendlines);
  - two consecutive such swings a, b (settings "trendline_touch_span_bars": 6-300 candles apart on the 5-minute chart,
    3-200 on the others - Claude's, the old engine's) make a 2-touch line. On the 1-minute chart (L6b.4) ANY two of its
    swings may (settings "trendline_pairs_1m", "trendline_touch_span_bars_1m", "trendline_swing_turn_adr_1m"): his
    Q98 C line joins two lows over three higher ones, and nothing between them goes through it. Through highs it FALLS or is flat, through
    lows it RISES or is flat - his spec (strategy-code.md section 6: "descending line: recent confirmed swing highs ...
    ascending line = mirror (rising swing lows)"); any steepness, flat included (#116: the angle "is relevant ... to
    predict", "a straight line ... it would break definitely" - so a flat line IS a line, and liquidity). The old
    engine refused flat pairs (strictly falling highs); measured on 2019, a RISING line through highs is almost
    always a chain of one move's new highs (249 of the 287 "any direction" 3-touch lines through highs), not a line
    anyone draws. No candle between the two touches closes more than settings "line_touch_adr" beyond it, and no wick
    goes more than the fixed 0.005 of a day through it (#92 D);
  - the THIRD touch is the next such swing within the span that comes to the line - from 2 x line_touch_adr short of
    it to 0.005 of a day through it (#94) - before any candle closes beyond it or pokes through it (#92: "the touch
    itself included");
  - the day's range (the ruler, machine/blocks/adr.py) is read where the 2-touch line is known.
  A 2-touch line is known when swing b is known (2 candles after it); a trendline when its 3rd touch is (2 candles
  after it). A 2-touch line is NOT liquidity (#92); it only matters as a line in the path (#69, #111 "about to get
  its 3rd").

  lines(bars, tf)          one row per line when it is known: touches 2, and a second row with touches 3 (the same
                           line_id) when its third touch is known. a_*, b_*, c_* = the touches; slope = price per
                           candle of that chart; way = falling / rising / flat; steepness = how many days of range the
                           line travels in one day (the old engine's measure, #115).
  more_touches(bars, tf)   the 4th, 5th ... touch of every trendline (the same test as the 3rd), until it is broken.
  on_m5(lines, m5)         every line on the 5-minute candles: where it starts, its price on any candle, the candle that
                           TAKES it (a wick more than settings "line_taken_through_adr" through - #69 "a trend line fake
                           out that is also a liquidity"), the candle it drifts too far from price (settings
                           "line_far_adr") or expires (settings "level_life_bars"). READS AHEAD - like
                           liquidity.lifetimes(), only lookups at a candle may use it.
  touch_entries(bars, tf)  his Type 2 candidates: the 3rd (and 4th, settings "wedge_touch") touch of a FALLING line
                           through highs (a sell) or a RISING line through lows (a buy); entry = a limit on the line,
                           hedge past it. Each row carries its steepness, whether a clear trend is visible and whether
                           its touches are 3 hours apart (his fixed rule "two_touches_close_together_are_one_touch") -
                           the compiler's step trendline_touch decides with settings "trendline_min_steepness_adr"
                           (his word "steep" is still OPEN: four-boxes B2-01) and "trendline_needs_trend".
  in_path(...)             the untaken line nearest the entry between the entry and the target, at a 5-minute candle.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import adr as ADR
from machine.blocks._base import TF_MINUTES, bar_close, check_bars, empty
from machine.blocks.swings import swings

COLUMNS = {"known_at": "datetime64[ns]", "line_id": "object", "tf": "object", "side": "object", "touches": "int64",
           "a_bar": "datetime64[ns]", "a_price": "float64", "b_bar": "datetime64[ns]", "b_price": "float64",
           "c_bar": "datetime64[ns]", "c_price": "float64", "slope": "float64", "way": "object",
           "steepness": "float64", "ruler": "float64"}
TOUCH_COLUMNS = {"known_at": "datetime64[ns]", "line_id": "object", "tf": "object", "touch": "int64",
                 "bar": "datetime64[ns]", "price": "float64", "line": "float64"}
ENTRY_COLUMNS = {"known_at": "datetime64[ns]", "line_id": "object", "tf": "object", "direction": "object",
                 "touch": "int64", "bar": "datetime64[ns]", "line": "float64", "entry": "float64", "hedge": "float64",
                 "extreme": "float64", "steepness": "float64", "trend_visible": "bool", "touches_apart": "bool",
                 "first_bar": "datetime64[ns]", "slope": "float64", "ruler": "float64"}
EPS = 1e-9


# ============================================================================================== the rule's numbers

def rule(tf: str) -> dict:
    """His rule's numbers (rules.py) and the settings that bound the search, for one chart. pairs: which two swings a
    line may join - "neighbours" (the old engine's, 5-minute and hour) or "any" (the 1-minute chart, L6b.4)."""
    r = rules.get("what_a_trendline_is")
    if tf == "5m":                                    # the old engine thinned only the 5-minute swings ("an HTF swing
        span = settings.get("trendline_touch_span_bars")                              # is already structural")
        turn, pairs = float(settings.get("trendline_swing_turn_adr")), "neighbours"
    elif tf == "1m":
        span = settings.get("trendline_touch_span_bars_1m")
        turn, pairs = float(settings.get("trendline_swing_turn_adr_1m")), str(settings.get("trendline_pairs_1m"))
    else:
        span, turn, pairs = settings.get("trendline_touch_span_bars_other"), 0.0, "neighbours"
    return dict(touches=int(r["touches"]), through=float(r["through_max_adr"]),
                tol=float(settings.get("line_touch_adr")), turn=turn, pairs=pairs,
                min_span=int(span[0]), max_span=int(span[1]))


def per_day(tf: str) -> float:
    """Candles of this chart in one day (the steepness is 'days of range travelled in one day')."""
    return 1440.0 / TF_MINUTES[tf]


# ============================================================================================== the touch points

def touch_points(bars: pd.DataFrame, tf: str, ruler: np.ndarray, turn: float) -> pd.DataFrame:
    """The swings a line may be drawn through: his 2-candle swings (wick ends), thinned to the ones that TURNED at least
    `turn` x the day's range from the last one kept (the old engine's structural_swings, walked in time order: two of
    the same side in a row keep the more extreme; a turn smaller than that is not structure). turn 0 = every swing.
    Each is known when the swing is (2 candles after it); the choice uses only swings before it."""
    sw = swings(bars, tf)
    pos = bars.index.get_indexer(pd.DatetimeIndex(sw["bar"]))
    sw = sw.assign(pos=pos)
    if turn <= 0 or not len(sw):
        return sw.sort_values(["pos", "side"], kind="stable", ignore_index=True)
    order = np.lexsort(((sw["side"] == "low").to_numpy(), pos))       # the old order: a high before a low on one bar
    keep = np.zeros(len(sw), bool)
    last_side, last_p = None, math.nan
    side = sw["side"].to_numpy()
    price = sw["price"].to_numpy(float)
    for k in order:
        s, p, i = side[k], price[k], pos[k]
        r = ruler[i] if np.isfinite(ruler[i]) else math.nan
        if last_side is None:
            last_side, last_p = s, p
        elif s == last_side:
            if (s == "high" and p > last_p) or (s == "low" and p < last_p):
                last_side, last_p = s, p                              # a new extreme: a level, not a turn
            else:
                continue                                              # an inside swing: not structure
        else:
            if not np.isfinite(r) or abs(p - last_p) < turn * r:
                continue                                              # the turn was too small to be structure
            last_side, last_p = s, p
        keep[k] = True
    return sw[keep].sort_values(["pos", "side"], kind="stable", ignore_index=True)


# ================================================================================================== the lines

def _pairs(P: np.ndarray, V: np.ndarray, sgn: float, R: dict, ruler: np.ndarray, piv: int):
    """The (anchor, partner) touch points a line may join, as indexes into P (positions) / V (prices) of one side.
    neighbours: each touch point and the next one (the old engine's). any: each touch point and every later one within
    the span that no touch point between them goes through - a quick sieve (it keeps every pair lines() can accept:
    the widest `through` of the stretch is used), the exact test on every candle is lines()' own."""
    if R["pairs"] == "neighbours":
        for k in range(1, len(P)):
            yield k - 1, k
        return
    if R["pairs"] != "any":
        raise ValueError(f"pairs {R['pairs']!r}: 'neighbours' or 'any'")
    n = len(ruler)
    for ka in range(len(P) - 1):
        a = int(P[ka])
        hi = int(np.searchsorted(P, a + R["max_span"], "right"))
        if hi <= ka + 1:
            continue
        idx = np.arange(ka + 1, hi)
        dt = (P[idx] - a).astype(float)
        r = ruler[a:min(n, int(P[hi - 1]) + piv + 1)]
        r = r[np.isfinite(r)]
        if not len(r):
            continue
        thr = R["through"] * float(r.max()) + EPS
        away = -sgn * (V[idx] - V[ka])                  # how far each later touch point sits on the line's inner side
        slope = away / dt                                # the line's slope to it (>= 0: rising lows / falling highs)
        room = np.minimum.accumulate(np.r_[np.inf, ((away + thr) / dt)[:-1]])   # the most any touch point before it
        ok = (slope >= -EPS) & (slope <= room + EPS) & (dt >= R["min_span"])     # lets the line rise
        for kb in idx[ok]:
            yield ka, int(kb)


def lines(bars: pd.DataFrame, tf: str, ruler: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Every 2-touch line and every trendline (see the module text). `ruler` = a day's range on every bar (default:
    his ADR from these bars)."""
    check_bars(bars, tf, ("high", "low", "close"))
    if ruler is None:
        ruler = ADR.adr_bars(bars, tf)["ruler"]
    ruler = np.asarray(ruler, dtype=float)
    if len(ruler) != len(bars):
        raise ValueError("the ruler must have one value per bar")
    R = rule(tf)
    if R["touches"] != 3:
        raise RuntimeError("rules.py what_a_trendline_is: three touches (#92)")
    piv = int(rules.get("swing_lookback"))
    pts = touch_points(bars, tf, ruler, R["turn"])
    h, lo, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    closes = bar_close(bars.index, tf)
    n = len(bars)
    rows = []
    for side in ("high", "low"):
        sgn = 1.0 if side == "high" else -1.0
        ext = h if side == "high" else lo                      # the side of the candle that can poke THROUGH the line
        p = pts[pts["side"] == side]
        P = p["pos"].to_numpy()
        V = p["price"].to_numpy(float)
        for ka, kb in _pairs(P, V, sgn, R, ruler, piv):
            a, b = int(P[ka]), int(P[kb])
            if not R["min_span"] <= b - a <= R["max_span"]:
                continue
            pa, pb = float(V[ka]), float(V[kb])
            if sgn * (pb - pa) > 0:
                continue                                       # through highs it falls (or is flat); through lows it rises
            fit = b + piv
            if fit >= n or not np.isfinite(ruler[fit]):
                continue
            slope = (pb - pa) / (b - a)
            tol, thr = R["tol"] * ruler[fit], R["through"] * ruler[fit]
            seg = pa + slope * (np.arange(a, b + 1) - a)
            if (sgn * (c[a:b + 1] - seg) > tol).any() or (sgn * (ext[a:b + 1] - seg) > thr + EPS).any():
                continue                                       # closed beyond it, or "forcefully drawn inside" (#92 D)
            third = None
            for j in range(kb + 1, len(P)):
                ci = int(P[j])
                if ci - b > R["max_span"]:
                    break
                proj = pb + slope * (np.arange(b + 1, ci + 1) - b)
                if (sgn * (c[b + 1:ci + 1] - proj) > tol).any() or (sgn * (ext[b + 1:ci + 1] - proj) > thr + EPS).any():
                    break                                      # broken before a third touch - the touch itself too (#92)
                lc = pb + slope * (ci - b)
                if -2.0 * tol - EPS <= sgn * (float(V[j]) - lc) <= thr + EPS:
                    third = (ci, float(V[j]))
                    break
            lid = f"line_{side}:{tf}:{bars.index[a]:%Y-%m-%d %H:%M}:{bars.index[b]:%Y-%m-%d %H:%M}"
            base = dict(line_id=lid, tf=tf, side=side, a_bar=bars.index[a], a_price=pa, b_bar=bars.index[b],
                        b_price=pb, slope=slope, way="rising" if slope > 0 else "falling" if slope < 0 else "flat",
                        steepness=abs(slope) * per_day(tf) / ruler[fit], ruler=ruler[fit])
            rows.append({**base, "known_at": closes[fit], "touches": 2, "c_bar": pd.NaT, "c_price": np.nan})
            if third is not None and third[0] + piv < n:
                rows.append({**base, "known_at": closes[third[0] + piv], "touches": 3,
                             "c_bar": bars.index[third[0]], "c_price": third[1]})
    if not rows:
        return empty(COLUMNS)
    out = pd.DataFrame(rows)[list(COLUMNS)]
    out["c_bar"] = pd.to_datetime(out["c_bar"])
    return out.sort_values(["known_at", "side", "a_bar"], kind="stable", ignore_index=True)


def price_at(row, pos: int | np.ndarray, index: pd.DatetimeIndex) -> float | np.ndarray:
    """A line's price on candle position(s) `pos` of the chart it was drawn on (index = that chart's bar times)."""
    b = index.get_loc(pd.Timestamp(row["b_bar"]))
    return float(row["b_price"]) + float(row["slope"]) * (np.asarray(pos) - b)


def more_touches(bars: pd.DataFrame, tf: str, table: pd.DataFrame | None = None,
                 ruler: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """The 4th, 5th ... touch of every trendline: the next touch point that comes to the line exactly as the 3rd did,
    before any candle closes beyond it or pokes through it. Known 2 candles after the touch."""
    check_bars(bars, tf, ("high", "low", "close"))
    if ruler is None:
        ruler = ADR.adr_bars(bars, tf)["ruler"]
    ruler = np.asarray(ruler, dtype=float)
    table = lines(bars, tf, ruler) if table is None else table
    R = rule(tf)
    piv = int(rules.get("swing_lookback"))
    pts = touch_points(bars, tf, ruler, R["turn"])
    h, lo, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    closes = bar_close(bars.index, tf)
    n = len(bars)
    rows = []
    for r in table[table["touches"] == 3].itertuples():
        sgn = 1.0 if r.side == "high" else -1.0
        ext = h if r.side == "high" else lo
        tol, thr = R["tol"] * r.ruler, R["through"] * r.ruler
        b = bars.index.get_loc(r.b_bar)
        last = bars.index.get_loc(r.c_bar)
        p = pts[(pts["side"] == r.side) & (pts["pos"] > last)]
        touch = 3
        for ci, v in zip(p["pos"].to_numpy(), p["price"].to_numpy(float)):
            ci = int(ci)
            if ci - last > R["max_span"]:
                break
            proj = r.b_price + r.slope * (np.arange(last + 1, ci + 1) - b)
            if (sgn * (c[last + 1:ci + 1] - proj) > tol).any() or (sgn * (ext[last + 1:ci + 1] - proj) > thr + EPS).any():
                break
            lc = r.b_price + r.slope * (ci - b)
            if -2.0 * tol - EPS <= sgn * (v - lc) <= thr + EPS and ci + piv < n:
                touch += 1
                rows.append({"known_at": closes[ci + piv], "line_id": r.line_id, "tf": tf, "touch": touch,
                             "bar": bars.index[ci], "price": float(v), "line": float(lc)})
                last = ci
    if not rows:
        return empty(TOUCH_COLUMNS)
    return pd.DataFrame(rows)[list(TOUCH_COLUMNS)].sort_values(["known_at", "line_id"], kind="stable",
                                                               ignore_index=True)


# ======================================================================================== on the 5-minute candles

def on_m5(table: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """Every line of `table` (any chart) on the 5-minute candles. Adds, as 5-minute candle numbers:
      x0      the candle of touch b (the line's price there is b_price)   slope5  price per 5-minute candle
      a5      the candle of touch a                                         start   the first candle after it is known
      taken   the first candle from touch a on whose wick goes more than settings "line_taken_through_adr" x its
              ruler through the line (#69: a fake out takes it; -1 = never). A trendline taken before it was known
              (between its 3rd touch and the 2 candles that confirm it) is dead on arrival.
      gone    the candle it stops being a line without being taken: its projection drifted more than settings
              "line_far_adr" x its ruler from the close, or settings "level_life_bars" candles passed.
      end     min(taken, gone) from start on (the candle it is taken on, or where it is gone).
    READS AHEAD: only a lookup at a candle (what was known before it opened, not ended by it) may use it.
    An hour line's slope per 5-minute candle is measured between the 5-minute candles of its two touches (weekends and
    holidays need no calendar - the old engine's htf_trendlines)."""
    check_bars(m5, "5m", ("high", "low", "close"))
    out = table.copy()
    if not len(out):
        for col in ("x0", "a5", "start", "taken", "gone", "end"):
            out[col] = pd.Series(dtype="int64")
        out["slope5"] = pd.Series(dtype="float64")
        return out
    idx = m5.index
    h, lo, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(m5)
    a5 = idx.searchsorted(pd.DatetimeIndex(out["a_bar"]), "left")
    x0 = idx.searchsorted(pd.DatetimeIndex(out["b_bar"]), "left")
    start = idx.searchsorted(pd.DatetimeIndex(out["known_at"]), "left")
    span = np.maximum(x0 - a5, 1)
    slope5 = np.where(out["tf"].to_numpy() == "5m", out["slope"].to_numpy(float),
                      (out["b_price"].to_numpy(float) - out["a_price"].to_numpy(float)) / span)
    through = float(settings.get("line_taken_through_adr"))
    far = float(settings.get("line_far_adr"))
    life = int(settings.get("level_life_bars"))
    taken = np.full(len(out), -1, np.int64)
    gone = np.full(len(out), n, np.int64)
    for k, (side, pb, r) in enumerate(zip(out["side"], out["b_price"].to_numpy(float), out["ruler"].to_numpy(float))):
        s0, e0 = int(a5[k]), int(min(n, start[k] + life))
        if s0 >= e0:
            gone[k] = min(n, start[k])
            continue
        j = np.arange(s0, e0)
        line = pb + slope5[k] * (j - x0[k])
        poke = (h[s0:e0] - line) if side == "high" else (line - lo[s0:e0])
        hit = np.flatnonzero(poke > through * r + EPS)
        if len(hit):
            taken[k] = s0 + int(hit[0])
        st = int(start[k])
        drift = np.flatnonzero((np.abs(c[s0:e0] - line) > far * r) & (j >= st))
        gone[k] = s0 + int(drift[0]) if len(drift) else e0
    out["x0"], out["a5"], out["start"], out["slope5"] = x0.astype(np.int64), a5.astype(np.int64), start, slope5
    out["taken"], out["gone"] = taken, gone
    t_eff = np.where(taken >= 0, taken, np.iinfo(np.int64).max)
    out["end"] = np.minimum(np.maximum(t_eff, start), gone)
    return out


def price5(lines5: pd.DataFrame, i: int | np.ndarray) -> np.ndarray:
    """The price of every line of an on_m5() table on 5-minute candle i."""
    return lines5["b_price"].to_numpy(float) + lines5["slope5"].to_numpy(float) * (np.asarray(i) - lines5["x0"].to_numpy())


# ================================================================================ his touch entry (Type 2)

def _trend_visible(h: np.ndarray, lo: np.ndarray, first: int, last: int, falling: bool) -> bool:
    """Is a clear trend visible between the first touch and the last (#116)? Lower highs AND lower lows for a falling
    line (a sell), higher highs AND higher lows for a rising one: the two halves of the stretch compared (the old
    engine's _trending; the test is Claude's, his B2-01 pictures were drawn with it). Fewer than 12 candles: no."""
    if last - first < 12:
        return False
    mid = (first + last) // 2
    h1, h2 = float(h[first:mid + 1].max()), float(h[mid:last + 1].max())
    l1, l2 = float(lo[first:mid + 1].min()), float(lo[mid:last + 1].min())
    return (h2 < h1 and l2 < l1) if falling else (h2 > h1 and l2 > l1)


def touch_gap_bars(tf: str) -> int:
    """His fixed rule "two_touches_close_together_are_one_touch" (36 five-minute candles = 3 hours) on this chart."""
    return max(1, math.ceil(int(rules.get("two_touches_close_together_are_one_touch")) * 5 / TF_MINUTES[tf]))


def touch_entries(bars: pd.DataFrame, tf: str, ruler: pd.Series | np.ndarray | None = None,
                  table: pd.DataFrame | None = None, touches4: pd.DataFrame | None = None) -> pd.DataFrame:
    """His Type 2 candidates (see the module text): one row per 3rd touch (when the line becomes a trendline) and per
    4th touch (settings "wedge_touch": the 3rd, up to the 4th) of a FALLING line through highs (sell) or a RISING line
    through lows (buy). A flat line is never a touch entry (#116: "it will break") - it stays liquidity.
      entry    a limit ON the line. The line moves, so it is the line's price settings "trendline_touch_fill_bars"
               candles after the moment it is known - the worse fill of that wait (the old engine's, AMBIGUITY F10;
               where it fills is a version, four-boxes B3-16)
      hedge    past the line by settings "trendline_hedge_pad_adr" of a day's range, measured at the moment it is
               known; None (version 1) = where the line counts as broken, "line_break_tolerance" x "line_touch_adr"
               ("a few pips past the line" is a version, B3-16, L19c; that it then moves with the line is management -
               settings the_cover_follows_the_trendline, L19a)
      extreme  the lowest low (a sell) / highest high (a buy) since the first touch - his "target = wedge low"
      trend_visible   #116 (see _trend_visible); touches_apart  every two touches at least 3 hours apart (rules.py)."""
    check_bars(bars, tf, ("high", "low", "close"))
    if ruler is None:
        ruler = ADR.adr_bars(bars, tf)["ruler"]
    ruler = np.asarray(ruler, dtype=float)
    table = lines(bars, tf, ruler) if table is None else table
    touches4 = more_touches(bars, tf, table, ruler) if touches4 is None else touches4
    first_touch = int(settings.get("wedge_touch"))
    fill = int(settings.get("trendline_touch_fill_bars"))
    pad = settings.get("trendline_hedge_pad_adr")       # M27 (L19c): None = the line's own break distance
    pad = float(settings.get("line_break_tolerance")) * float(settings.get("line_touch_adr")) if pad is None else float(pad)
    gap = touch_gap_bars(tf)
    idx = bars.index
    h, lo = bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    four = touches4[touches4["touch"] == first_touch + 1].set_index("line_id") if len(touches4) else None
    rows = []
    for r in table[table["touches"] == 3].itertuples():
        falling_high = r.side == "high" and r.slope < 0
        rising_low = r.side == "low" and r.slope > 0
        if not (falling_high or rising_low):
            continue
        sell = falling_high
        b = idx.get_loc(r.b_bar)
        t = [idx.get_loc(r.a_bar), b, idx.get_loc(r.c_bar)]
        arms = [(3, r.known_at, r.c_bar, list(t))]
        if four is not None and r.line_id in four.index:
            f = four.loc[r.line_id]
            arms.append((4, f["known_at"], f["bar"], t + [idx.get_loc(f["bar"])]))
        for touch, known, bar, tt in arms:
            if touch < first_touch or touch > first_touch + 1:
                continue
            i = int(idx.searchsorted(known, "left")) - 1          # the candle whose close made it known
            line_now = r.b_price + r.slope * (i - b)
            entry = r.b_price + r.slope * (i + fill - b)
            hedge = line_now + pad * r.ruler if sell else line_now - pad * r.ruler
            ext = float(lo[t[0]:i + 1].min()) if sell else float(h[t[0]:i + 1].max())
            apart = min(np.diff(sorted(tt))) >= gap
            rows.append({"known_at": known, "line_id": r.line_id, "tf": tf, "direction": "sell" if sell else "buy",
                         "touch": touch, "bar": bar, "line": line_now, "entry": entry, "hedge": hedge,
                         "extreme": ext, "steepness": r.steepness,
                         "trend_visible": _trend_visible(h, lo, min(tt), max(tt), sell), "touches_apart": bool(apart),
                         "first_bar": r.a_bar, "slope": r.slope, "ruler": r.ruler})
    if not rows:
        return empty(ENTRY_COLUMNS)
    return pd.DataFrame(rows)[list(ENTRY_COLUMNS)].sort_values(["known_at", "line_id", "touch"], kind="stable",
                                                               ignore_index=True)


# ============================================================================================ a line in the path

def in_path(lines5: pd.DataFrame, i: int, entry: float, target: float, direction: str) -> tuple[str, float] | None:
    """The line (2 or 3 touches, any chart put on the 5-minute candles by on_m5) standing between the entry and the
    target at 5-minute candle i, nearest the entry: known before candle i opened, not taken (a wick through it since its
    first touch - #69 "a trend line fake out ... is considered as a liquidity taken out"), not gone. A line priced at
    the target (within a cent) is the target itself. (line_id, its price at candle i) or None."""
    if not len(lines5):
        return None
    m = (lines5["start"].to_numpy() <= i) & (i < lines5["gone"].to_numpy())
    tk = lines5["taken"].to_numpy()
    m &= (tk < 0) | (tk > i)
    if not m.any():
        return None
    sub = lines5[m]
    px = price5(sub, i)
    lo_, hi_ = min(entry, target), max(entry, target)
    ok = (px > lo_) & (px < hi_) & (np.abs(px - target) > 0.011)
    if not ok.any():
        return None
    k = np.flatnonzero(ok)[int(np.argmin(np.abs(px[ok] - entry)))]
    return str(sub["line_id"].iat[k]), float(px[k])
