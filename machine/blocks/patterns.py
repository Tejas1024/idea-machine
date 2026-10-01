"""Triangles, wedges and false breakouts (L6b.1) - strategy.md section 4, drawn from the lines of trendlines.py.

His words:
  strategy.md section 4: "Triangle: two lines that clearly narrow toward each other, with price moving INSIDE them,
      then the break; after a big move; both lines = liquidity (#116, #118)." · "False-breakout: first push fails ->
      grabs liquidity, leaves FVG, reverses (P1)." · "Wedge: single line holds one side while price breaks the other
      (falling/rising)." · "False-liquidity (P2): a false breakout consumes the liquidity -> price goes the other way,
      returns days later."
  strategy.md section 14: "False-liquidity: a false breakout consumes the liquidity -> continuation the other way (FVG
      confirms). Once a high or low is swept it is finished - not used as liquidity a second time (#129)."
  LEARNED #116: "I want to see the price action inside the triangle, then the break."
  LEARNED #118: "Here, this is NOT a proper triangle pattern ... it's just a third touch" (T01, 28 Feb 2018; T02, 09 Oct
      2019: "BOTH are not proper triangles").
  strategy-code.md section 14 (his old spec): "FALSE BREAKOUT: price wicks beyond a boundary (sweep) but returns inside
      within fakeout_lookback bars -> that boundary's liquidity is CONSUMED" · the wedge: "a descending resistance line
      held over 3-4 touches" · his hand-off, 20 Sep: "a trendline with 3 respected touches that price has NOT broken,
      and a break of structure on the OPPOSITE side -> the gap after that break."

  triangles(bars, tf)          a line through highs and a line through lows (2 touches or more - settings
                               "triangle_min_swings" = 2, his old spec) that are both alive; from the moment both are
                               known (the triangle's START) they must CLOSE to rules.py "what_a_triangle_is" (0.60) of
                               the gap between them at the start, not crossed, at least rules.py
                               "how_long_a_triangle_must_last" (24) candles from the start, no taller than settings
                               "triangle_max_height_adr" at the start, price inside (settings "triangle_inside_share",
                               0 = off) - and after a big move (the largest move into the start over settings
                               "big_move_bars" is at least settings "big_move_adr" of a day's range; "big move" is his
                               word still OPEN - four-boxes B2-02 - so every row carries the move it measured and
                               `after_big_move`, and need_big_move=False lists them all). KNOWN at the close of the
                               first candle where all of that is true. A line lives until a candle CLOSES beyond it,
                               drifts settings "line_far_adr" from the close, or settings "level_life_bars" pass.
  triangle_breaks(bars, tf)    the first wick past either line after the triangle is known (a sweep of that boundary,
                               #95), and what came of it: FALSE = a candle closes back inside within settings
                               "fakeout_lookback_bars" (his old spec, 20) - known at that close, the trade goes the
                               other way; REAL = no close back inside in that window - known at its end.
  false_breaks(m5, levels, s)  every sweep of his liquidity (liquidity.sweeps) and what came of it, the old engine's
                               reading (entries._false_break, #128): the first candle from the sweep on that closes
                               back on the side price came from = FALSE (the sweep failed: the liquidity is consumed,
                               price goes the other way); closes beyond first = REAL; neither within settings
                               "false_break_close_back_bars" = undecided. Known at that candle's close. The level is
                               finished either way (#129: liquidity.sweeps).
  wedges(bars, tf)             a trendline (3 touches) HELD - no candle closes beyond it (settings
                               "wedge_held_line_no_close") - while price CLOSES beyond the other side of the wedge (the
                               lowest low of settings "wedge_break_bars" candles before the line was known, for a line
                               through highs) within that many candles after it: the trade goes out the other side
                               (a line through highs held -> a sell). Known at that close.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import adr as ADR
from machine.blocks import trendlines as TL
from machine.blocks._base import bar_close, check_bars, empty

TRIANGLE_COLUMNS = {"known_at": "datetime64[ns]", "tri_id": "object", "tf": "object", "top_id": "object",
                    "bottom_id": "object", "start_bar": "datetime64[ns]", "bar": "datetime64[ns]", "top": "float64",
                    "bottom": "float64", "top_start": "float64", "bottom_start": "float64", "narrow": "float64",
                    "height_adr": "float64", "big_move": "float64", "after_big_move": "bool",
                    "inside_share": "float64", "ruler": "float64"}
BREAK_COLUMNS = {"known_at": "datetime64[ns]", "tri_id": "object", "tf": "object", "side": "object",
                 "direction": "object", "kind": "object", "bar": "datetime64[ns]", "price": "float64",
                 "extreme": "float64", "opposite": "float64", "back_bar": "datetime64[ns]"}
FALSE_COLUMNS = {"known_at": "datetime64[ns]", "level_id": "object", "kind": "object", "tf": "object",
                 "side": "object", "direction": "object", "result": "object", "bar": "datetime64[ns]",
                 "price": "float64", "extreme": "float64", "decided_bar": "datetime64[ns]", "fresh": "bool"}
WEDGE_COLUMNS = {"known_at": "datetime64[ns]", "line_id": "object", "tf": "object", "direction": "object",
                 "bar": "datetime64[ns]", "level": "float64", "wedge_high": "float64", "wedge_low": "float64",
                 "line": "float64", "start_bar": "datetime64[ns]", "ruler": "float64"}
INSIDE_SLACK = 0.02            # the old engine's: a wick within 2 cents of a line still counts as inside (entries.py)
EPS = 1e-9


def _ruler(bars: pd.DataFrame, tf: str, ruler) -> np.ndarray:
    r = ADR.adr_bars(bars, tf)["ruler"] if ruler is None else ruler
    r = np.asarray(r, dtype=float)
    if len(r) != len(bars):
        raise ValueError("the ruler must have one value per bar")
    return r


# ================================================================================================ one line's life

def _lives(table: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Each line (its first row: the moment it is known with 2 touches) on its own chart: born (the candle whose close
    made it known), x0 (touch b), death (the first candle after born that CLOSES beyond it, drifts settings
    "line_far_adr" of its day's range from the close, or settings "level_life_bars" later - the old engine's
    mark_deaths). Reads ahead: only a question asked AT a candle (alive through it?) may use death."""
    first = table.sort_values("known_at", kind="stable").drop_duplicates("line_id").reset_index(drop=True)
    if not len(first):
        return first.assign(born=pd.Series(dtype="int64"), x0=pd.Series(dtype="int64"),
                            death=pd.Series(dtype="int64"))
    idx = bars.index
    c = bars["close"].to_numpy(float)
    n = len(bars)
    life = int(settings.get("level_life_bars"))
    far = float(settings.get("line_far_adr"))
    born = idx.searchsorted(pd.DatetimeIndex(first["known_at"]), "left") - 1
    x0 = idx.get_indexer(pd.DatetimeIndex(first["b_bar"]))
    death = np.empty(len(first), np.int64)
    for k, (side, pb, s, r) in enumerate(zip(first["side"], first["b_price"].to_numpy(float),
                                             first["slope"].to_numpy(float), first["ruler"].to_numpy(float))):
        s0, e0 = int(born[k]) + 1, int(min(n, born[k] + 1 + life))
        if s0 >= e0:
            death[k] = min(n, s0)
            continue
        line = pb + s * (np.arange(s0, e0) - x0[k])
        beyond = (c[s0:e0] > line) if side == "high" else (c[s0:e0] < line)
        hit = np.flatnonzero(beyond | (np.abs(line - c[s0:e0]) > far * r))
        death[k] = s0 + int(hit[0]) if len(hit) else e0
    return first.assign(born=born.astype(np.int64), x0=x0.astype(np.int64), death=death)


def _at(L, pos) -> np.ndarray | float:
    return L.b_price + L.slope * (np.asarray(pos) - L.x0)


# ==================================================================================================== triangles

def triangles(bars: pd.DataFrame, tf: str = "5m", ruler=None, table: pd.DataFrame | None = None,
              need_big_move: bool = True, big_move_adr: float | None = None) -> pd.DataFrame:
    """Every triangle when it is known (see the module text). big_move_adr: how big 'a big move' is (default settings
    "big_move_adr" - None while his word is open, B2-02: then give it, or need_big_move=False = every shape, each
    with the move it measured in `big_move`)."""
    check_bars(bars, tf, ("high", "low", "close"))
    ruler = _ruler(bars, tf, ruler)
    table = TL.lines(bars, tf, ruler) if table is None else table
    if int(settings.get("triangle_min_swings")) != 2:
        raise NotImplementedError("triangle_min_swings: the lines have 2 or 3 touches (trendlines.py)")
    narrow_to = float(rules.get("what_a_triangle_is"))
    min_bars = int(rules.get("how_long_a_triangle_must_last"))
    max_h = float(settings.get("triangle_max_height_adr"))
    big = settings.get("big_move_adr") if big_move_adr is None else big_move_adr
    if big is None and need_big_move:
        raise ValueError("'a big move' is his word still open (four-boxes B2-02): give big_move_adr, or "
                         "need_big_move=False for every shape with the move it measured")
    big_adr, big_bars = (math.inf if big is None else float(big)), int(settings.get("big_move_bars"))
    inside_min = float(settings.get("triangle_inside_share"))
    if int(settings.get("triangle_min_touch")):
        raise NotImplementedError("triangle_min_touch: off in the old engine (0) and not ported")
    lives = _lives(table, bars)
    h, lo, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    idx = bars.index
    closes = bar_close(idx, tf)
    per = min(int(TL.per_day(tf)), 288)         # lines alive at the same time share a bucket of this many candles
    tops = lives[lives["side"] == "high"]
    bots = lives[lives["side"] == "low"]
    by_day: dict[int, list] = {}
    for S in bots.itertuples():
        for q in range(int(S.born) // per, int(S.death) // per + 1):
            by_day.setdefault(q, []).append(S)
    rows, seen = [], set()
    for R in tops.itertuples():
        for q in range(int(R.born) // per, int(R.death) // per + 1):
            for S in by_day.get(q, ()):
                if (R.line_id, S.line_id) in seen:
                    continue
                seen.add((R.line_id, S.line_id))
                b = max(int(R.born), int(S.born))
                e = min(int(R.death), int(S.death))
                if b < 0 or e - b < min_bars or not np.isfinite(ruler[b]):
                    continue
                top_b, bot_b = float(_at(R, b)), float(_at(S, b))
                w = top_b - bot_b
                if w <= 0 or w > max_h * ruler[b]:
                    continue                            # crossed already, or a market, not a pattern
                shrink = S.slope - R.slope              # how fast the gap closes per candle
                if shrink <= 0:
                    continue                            # parallel or opening: never narrows
                t = max(b + min_bars - 1, b + math.ceil((1.0 - narrow_to) * w / shrink - EPS))
                if t >= e:
                    continue                            # a line broke before the two had narrowed
                top_t, bot_t = float(_at(R, t)), float(_at(S, t))
                if top_t <= bot_t:
                    continue                            # an apex, not a triangle
                k0 = max(0, b - big_bars)
                move = float(np.max(np.abs(c[b] - c[k0:b + 1]))) / ruler[b]
                seg = np.arange(b, t + 1)
                inside = float(((h[b:t + 1] <= _at(R, seg) + INSIDE_SLACK)
                                & (lo[b:t + 1] >= _at(S, seg) - INSIDE_SLACK)).mean())
                if inside < inside_min:
                    continue
                after = move >= big_adr
                if need_big_move and not after:
                    continue
                rows.append({"known_at": closes[t], "tri_id": f"tri:{tf}:{R.line_id}|{S.line_id}", "tf": tf,
                             "top_id": R.line_id, "bottom_id": S.line_id, "start_bar": idx[b], "bar": idx[t],
                             "top": top_t, "bottom": bot_t, "top_start": top_b, "bottom_start": bot_b,
                             "narrow": (top_t - bot_t) / w, "height_adr": w / ruler[b], "big_move": move,
                             "after_big_move": after, "inside_share": inside, "ruler": ruler[b]})
    if not rows:
        return empty(TRIANGLE_COLUMNS)
    return pd.DataFrame(rows)[list(TRIANGLE_COLUMNS)].sort_values(["known_at", "tri_id"], kind="stable",
                                                                  ignore_index=True)


def triangle_breaks(bars: pd.DataFrame, tf: str = "5m", ruler=None, tris: pd.DataFrame | None = None,
                    table: pd.DataFrame | None = None, need_big_move: bool = True,
                    big_move_adr: float | None = None) -> pd.DataFrame:
    """The first wick out of each triangle after it is known, false or real (see the module text)."""
    check_bars(bars, tf, ("high", "low", "close"))
    ruler = _ruler(bars, tf, ruler)
    table = TL.lines(bars, tf, ruler) if table is None else table
    tris = triangles(bars, tf, ruler, table, need_big_move, big_move_adr) if tris is None else tris
    window = int(settings.get("fakeout_lookback_bars"))
    lives = _lives(table, bars).set_index("line_id")
    h, lo, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    idx = bars.index
    closes = bar_close(idx, tf)
    n = len(bars)
    life = int(settings.get("level_life_bars"))
    rows = []
    for T in tris.itertuples():
        R, S = lives.loc[T.top_id], lives.loc[T.bottom_id]
        t = idx.get_loc(T.bar)
        s0, e0 = t + 1, min(n, int(min(R["death"], S["death"])) + 1, t + 1 + life)
        if s0 >= e0:
            continue
        j = np.arange(s0, e0)
        top = R["b_price"] + R["slope"] * (j - R["x0"])
        bot = S["b_price"] + S["slope"] * (j - S["x0"])
        up = np.flatnonzero(h[s0:e0] > top + EPS)
        dn = np.flatnonzero(lo[s0:e0] < bot - EPS)
        if not len(up) and not len(dn):
            continue
        ju = int(up[0]) if len(up) else e0
        jd = int(dn[0]) if len(dn) else e0
        side = "high" if ju <= jd else "low"            # a candle out of both sides at once: the top (rare)
        k = s0 + min(ju, jd)
        own = (lambda q: R["b_price"] + R["slope"] * (q - R["x0"])) if side == "high" else \
            (lambda q: S["b_price"] + S["slope"] * (q - S["x0"]))
        other = (lambda q: S["b_price"] + S["slope"] * (q - S["x0"])) if side == "high" else \
            (lambda q: R["b_price"] + R["slope"] * (q - R["x0"]))
        back = None
        for q in range(k, min(n, k + window + 1)):
            if (c[q] < own(q)) if side == "high" else (c[q] > own(q)):
                back = q
                break
        end = back if back is not None else k + window
        if end >= n:
            continue                                     # the window is not over yet: not known
        ext = float(h[k:end + 1].max()) if side == "high" else float(lo[k:end + 1].min())
        rows.append({"known_at": closes[end], "tri_id": T.tri_id, "tf": tf, "side": side,
                     "direction": ("sell" if side == "high" else "buy") if back is not None else
                     ("buy" if side == "high" else "sell"),
                     "kind": "false" if back is not None else "real", "bar": idx[k], "price": float(own(k)),
                     "extreme": ext, "opposite": float(other(end)),
                     "back_bar": idx[back] if back is not None else pd.NaT})
    if not rows:
        return empty(BREAK_COLUMNS)
    out = pd.DataFrame(rows)[list(BREAK_COLUMNS)]
    out["back_bar"] = pd.to_datetime(out["back_bar"])
    return out.sort_values(["known_at", "tri_id"], kind="stable", ignore_index=True)


# ======================================================================================== false breaks of liquidity

def false_breaks(m5: pd.DataFrame, levels: pd.DataFrame, sweep_table: pd.DataFrame) -> pd.DataFrame:
    """Every sweep of his liquidity and what came of it (see the module text). levels = liquidity.book rows (a line's
    price moves with the candle: slope, x0); sweep_table = liquidity.sweeps."""
    check_bars(m5, "5m", ("high", "low", "close"))
    if not len(sweep_table):
        return empty(FALSE_COLUMNS)
    window = int(settings.get("false_break_close_back_bars"))
    h, lo, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    idx = m5.index
    n = len(m5)
    born = levels[levels["event"] == "born"].drop_duplicates("level_id").set_index("level_id")
    rows = []
    slopes = born["slope"] if "slope" in born else pd.Series(0.0, index=born.index)
    for s in sweep_table.itertuples():
        j = int(idx.get_loc(s.bar))
        slope = float(slopes.get(s.level_id, 0.0))
        level = lambda q: s.price + slope * (q - j)      # the sweep row's price is the line's price at the sweep
        res, dec = "undecided", min(n - 1, j + window)
        for q in range(j, min(n, j + window + 1)):
            lv = level(q)
            if (c[q] > lv) if s.side == "high" else (c[q] < lv):
                res, dec = "real", q
                break
            if (c[q] < lv) if s.side == "high" else (c[q] > lv):
                res, dec = "false", q
                break
        if res == "undecided" and j + window >= n:
            continue                                     # the window is not over yet
        ext = float(h[j:dec + 1].max()) if s.side == "high" else float(lo[j:dec + 1].min())
        rows.append({"known_at": bar_close(idx[[dec]], "5m")[0], "level_id": s.level_id, "kind": s.kind, "tf": s.tf,
                     "side": s.side, "direction": ("sell" if s.side == "high" else "buy") if res == "false" else
                     ("buy" if s.side == "high" else "sell") if res == "real" else "",
                     "result": res, "bar": s.bar, "price": float(s.price), "extreme": ext, "decided_bar": idx[dec],
                     "fresh": bool(s.fresh)})
    if not rows:
        return empty(FALSE_COLUMNS)
    return pd.DataFrame(rows)[list(FALSE_COLUMNS)].sort_values(["known_at", "level_id"], kind="stable",
                                                               ignore_index=True)


# ======================================================================================================= wedges

def wedges(bars: pd.DataFrame, tf: str = "5m", ruler=None, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every wedge break (see the module text): a trendline held, price out the other side."""
    check_bars(bars, tf, ("high", "low", "close"))
    ruler = _ruler(bars, tf, ruler)
    table = TL.lines(bars, tf, ruler) if table is None else table
    hold = bool(settings.get("wedge_held_line_no_close"))
    span = int(settings.get("wedge_break_bars"))
    far = float(settings.get("line_far_adr"))
    h, lo, c = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    idx = bars.index
    closes = bar_close(idx, tf)
    n = len(bars)
    rows = []
    for r in table[table["touches"] == 3].itertuples():
        b0 = int(idx.searchsorted(r.known_at, "left")) - 1      # the candle whose close made it a trendline
        cpos = idx.get_loc(r.c_bar)
        x0 = idx.get_loc(r.b_bar)
        s0, e0 = b0 + 1, min(n, b0 + 1 + span)
        if s0 >= e0:
            continue
        held_high = r.side == "high"
        lo_w, hi_w = float(lo[max(0, b0 - span):b0 + 1].min()), float(h[max(0, b0 - span):b0 + 1].max())
        edge = lo_w if held_high else hi_w
        brk = np.flatnonzero((c[s0:e0] < edge) if held_high else (c[s0:e0] > edge))
        if not len(brk):
            continue
        k = s0 + int(brk[0])
        q = np.arange(cpos + 1, k + 1)                  # from the 3rd touch to the break
        line = r.b_price + r.slope * (q - x0)
        if hold and ((c[cpos + 1:k + 1] > line) if held_high else (c[cpos + 1:k + 1] < line)).any():
            continue                                    # the held side gave way first
        if (np.abs(line - c[cpos + 1:k + 1]) > far * r.ruler).any():
            continue                                    # the line had drifted away: not a line any more
        rows.append({"known_at": closes[k], "line_id": r.line_id, "tf": tf,
                     "direction": "sell" if held_high else "buy", "bar": idx[k], "level": edge,
                     "wedge_high": hi_w, "wedge_low": lo_w, "line": float(r.b_price + r.slope * (k - x0)),
                     "start_bar": idx[b0], "ruler": r.ruler})
    if not rows:
        return empty(WEDGE_COLUMNS)
    return pd.DataFrame(rows)[list(WEDGE_COLUMNS)].sort_values(["known_at", "line_id"], kind="stable",
                                                               ignore_index=True)
