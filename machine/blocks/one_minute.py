"""The 1-minute layer (L6b.4): the blocks on 1-minute bars his lessons name.

His words:
  strategy.md section 5 (row 29): "1-min = never the entry chart; used as a check before a 5-min gap entry - a
      1-minute trendline or triangle against the entry means wait (#111)."
  strategy.md section 14: "1-min: never the entry chart (strategy is 5-min based); used as a check before a 5-min gap
      entry - a 1-minute trendline or triangle against the entry means wait (#100, #111)."
  strategy.md section 4: "A trendline against the entry (3 touches, or about to get its 3rd): wait for it to be swept,
      then enter the gap after it ... A trendline fake-out counts as liquidity taken (#69, #111)."
  LEARNED #100: "once we build everything in five minutes I can simply tell the same apply for one minute".
  LEARNED #111 (his F04 chart): "the actual entry should be once after it sweep the trend line. And then it formed the
      fair value gap ... kindly check is there any one minute ... trend line ... before entering the five minute fair
      value gap ... Now is the time to build that one minute ... trend line and triangle ... so we can avoid a lot of
      unwanted hedges and ... refine the entry better ... that same thing applies for zone as well".
  LEARNED #98 (his Q98 C chart): "it won't look like a triangle in five minute ... but if you look into the one minute
      candle there is definitely a triangle pattern ... I want price to break that trend line and I will wait for the
      fair value gap in five minutes only, then I will enter there".
  LEARNED #99: "Whenever there is any sort of liquidity that you can see - equal lows ... trend line type of thing and
      triangle type of thing ... a session low inside that zone - then I would wait for that to take out first, then I
      would wait for the five[-minute] gap ... So our hedging distance will be less".

The blocks (1-minute candles, server clock; a "day's range" = his 10-day ADR, machine/blocks/adr.py):
  lines(m1)          the 1-minute lines: trendlines.lines on the 1-minute chart - HIS line rule, the same as on the
                     5-minute chart (#100): through his 2-candle swings, drawn through the wick ends (#112), nothing
                     through the line between its touches (#92 D), 3 touches = a trendline (#92, #94). The chart's own
                     numbers are settings: the zig-zag "trendline_swing_turn_adr_1m" (0.04 - read off his two 1-minute
                     lines, F04 and Q98 C), the spans "trendline_touch_span_bars_1m", and "trendline_pairs_1m" = any: a
                     line may join ANY two swings, not only neighbours (his Q98 C line passes over three higher lows).
  book(m1)           every 1-minute line once, from when it has 2 touches ("3 touches, or about to get its 3rd",
                     #111; both his lines had 2 when he pointed at them), on the 1-minute candles:
                       x0       the candle of touch b (the line's price there is b_price); slope = price per candle
                       start    the first candle after it is known
                       taken    the first candle after touch b whose wick goes more than settings
                                "line_taken_through_adr" x its day's range through it - THE 1-MINUTE TRENDLINE BREAK
                                (a fake-out takes it too, #69); -1 = never. Taken before it was known = never seen.
                       gone     where it stops being a line untaken: its price drifted settings "line_far_adr" from the
                                close, or settings "one_minute_line_life_bars" candles passed
                       end      taken, else gone
                     READS AHEAD (like liquidity.lifetimes): only the lookups at a moment below may use it.
  breaks(bk, m1)     the 1-minute trendline break as events: one row per line taken, known at the close of that minute.
  triangles(m1)      the 1-minute triangle: patterns.triangles on the 1-minute chart (his fixed rules: two lines that
                     narrow to 60%, for at least 24 candles). "After a big move" is his word still open (four-boxes
                     B2-02): every shape, with the move it measured. A triangle is two lines and both are liquidity
                     (strategy.md section 4) - so the check reads every line, and holds every triangle's lines whatever
                     "big move" turns out to mean. Made for a stretch of candles (pictures, a later idea), not for years.
  between(bk, i, lo, hi, side)   THE CHECK's lookup: the lines alive at the close of 1-minute candle i - known by then,
                     not taken, not gone - on `side` ("low": lines through lows, their liquidity under price) whose price
                     on candle i is strictly between lo and hi. For a gap entry: between the entry and the hedge (F04).
                     For a zone entry the zone lies between them too, so it is "liquidity inside the zone" (#99).
  between5(l5, i5, lo, hi, side)  the same for 5-minute / hour lines on the 5-minute candles (trendlines.on_m5).
  first_take(rows)   which of those lines is taken first, and on which candle (None: none is ever taken).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from machine import settings
from machine.blocks import adr as ADR
from machine.blocks import patterns as PT
from machine.blocks import trendlines as TL
from machine.blocks._base import bar_close, check_bars, empty

TF = "1m"
BOOK_EXTRA = {"x0": "int64", "start": "int64", "taken": "int64", "gone": "int64", "end": "int64"}
BREAK_COLUMNS = {"known_at": "datetime64[ns]", "line_id": "object", "side": "object", "touches": "int64",
                 "bar": "datetime64[ns]", "price": "float64", "line_known_at": "datetime64[ns]"}
INSIDE = 0.011                  # a line within a cent of the entry or the hedge is that price itself (trendlines.in_path)
EPS = 1e-9
_W = 64


def _ruler(m1: pd.DataFrame, ruler) -> np.ndarray:
    r = ADR.adr_bars(m1, TF)["ruler"] if ruler is None else ruler
    r = np.asarray(r, dtype=float)
    if len(r) != len(m1):
        raise ValueError("the ruler must have one value per bar")
    return r


# ================================================================================================== the lines

def lines(m1: pd.DataFrame, ruler=None) -> pd.DataFrame:
    """Every 1-minute 2-touch line and trendline (trendlines.lines on the 1-minute chart - see the module text)."""
    check_bars(m1, TF, ("high", "low", "close"))
    return TL.lines(m1, TF, _ruler(m1, ruler))


def _first_cross(arr: np.ndarray, starts, ends, base, slope, x0, thr, sign) -> np.ndarray:
    """For each line q: the first candle i, starts[q] <= i < ends[q], where sign[q] * (arr[i] - its price on i) is more
    than thr[q] (its price on i = base[q] + slope[q] * (i - x0[q])); -1 if none. The next 64 candles of every line at
    once, the few still open in growing chunks (like _base.first_hit, for a price that moves)."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    starts = np.clip(np.asarray(starts, dtype=np.int64), 0, n)
    ends = np.minimum(np.asarray(ends, dtype=np.int64), n)
    base, slope, thr, sign = (np.broadcast_to(np.asarray(v, dtype=float), starts.shape) for v in (base, slope, thr, sign))
    x0 = np.broadcast_to(np.asarray(x0, dtype=np.int64), starts.shape)
    out = np.full(len(starts), -1, np.int64)
    if not len(starts) or not n:
        return out
    win = sliding_window_view(np.concatenate([arr, np.full(_W, np.nan)]), _W)   # NaN past the end: never a hit
    k = np.arange(_W)
    for b0 in range(0, len(starts), 8192):
        sl = slice(b0, b0 + 8192)
        s = starts[sl]
        pos = s[:, None] + k[None, :]
        line = base[sl, None] + slope[sl, None] * (pos - x0[sl, None])
        hit = (sign[sl, None] * (win[s] - line) > thr[sl, None] + EPS) & (pos < ends[sl, None])
        has = hit.any(axis=1)
        out[sl][has] = (s + hit.argmax(axis=1))[has]
    for q in np.flatnonzero((out < 0) & (starts + _W < ends)):
        pos, size, end = int(starts[q]) + _W, 256, int(ends[q])
        while pos < end:
            e = min(pos + size, end)
            d = sign[q] * (arr[pos:e] - (base[q] + slope[q] * (np.arange(pos, e) - x0[q])))
            h = np.flatnonzero(d > thr[q] + EPS)
            if len(h):
                out[q] = pos + int(h[0])
                break
            pos, size = e, size * 4
    return out


def book(m1: pd.DataFrame, table: pd.DataFrame | None = None, ruler=None) -> pd.DataFrame:
    """Every 1-minute line once, on the 1-minute candles (see the module text). READS AHEAD."""
    check_bars(m1, TF, ("high", "low", "close"))
    ruler = _ruler(m1, ruler)
    table = lines(m1, ruler) if table is None else table
    first = table.sort_values("known_at", kind="stable").drop_duplicates("line_id").reset_index(drop=True)
    if not len(first):
        out = first.copy()
        for c, t in BOOK_EXTRA.items():
            out[c] = pd.Series(dtype=t)
        return out
    idx = m1.index
    n = len(m1)
    h, lo, c = (m1[x].to_numpy(float) for x in ("high", "low", "close"))
    x0 = idx.get_indexer(pd.DatetimeIndex(first["b_bar"])).astype(np.int64)
    start = idx.searchsorted(pd.DatetimeIndex(first["known_at"]), "left").astype(np.int64)
    life = np.minimum(start + int(settings.get("one_minute_line_life_bars")), n)
    base, slope, r = (first[x].to_numpy(float) for x in ("b_price", "slope", "ruler"))
    high = (first["side"] == "high").to_numpy()
    through = float(settings.get("line_taken_through_adr")) * r
    far = float(settings.get("line_far_adr")) * r
    taken = np.full(len(first), -1, np.int64)
    for m, arr, sg in ((high, h, 1.0), (~high, lo, -1.0)):          # a wick THROUGH it: above a line through highs
        if m.any():
            taken[m] = _first_cross(arr, x0[m] + 1, life[m], base[m], slope[m], x0[m], through[m], sg)
    up = _first_cross(c, start, life, base, slope, x0, far, 1.0)     # the close drifted far above it ...
    dn = _first_cross(c, start, life, base, slope, x0, far, -1.0)    # ... or far below it
    gone = np.minimum(np.where(up >= 0, up, life), np.where(dn >= 0, dn, life))
    taken = np.where((taken >= 0) & (taken > gone), -1, taken)       # gone first: no longer a line when it was poked
    end = np.where(taken >= 0, taken, gone)                          # taken before it was known: never alive
    out = first.assign(x0=x0, start=start, taken=taken, gone=gone, end=end)
    out.attrs["longest"] = int(max(0, (end - start).max()))          # between() looks only this far back
    return out


def breaks(bk: pd.DataFrame, m1: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 1-minute trendline break (strategy.md section 14): one row per line taken after it was known, known at the
    close of the minute that took it. price = the line's price on that minute; touches = 3 if its third touch was known
    by then (table = lines(); None = count 2 for every line)."""
    check_bars(m1, TF, ("high", "low", "close"))
    t = bk[(bk["taken"] >= 0) & (bk["taken"] >= bk["start"])]
    if not len(t):
        return empty(BREAK_COLUMNS)
    j = t["taken"].to_numpy()
    known = bar_close(m1.index[j], TF)
    touches = np.full(len(t), 2, np.int64)
    if table is not None and len(table):
        third = table[table["touches"] == 3].drop_duplicates("line_id").set_index("line_id")["known_at"]
        k3 = pd.to_datetime(t["line_id"].map(third))
        touches = np.where(k3.notna().to_numpy() & (k3.to_numpy() <= known.to_numpy()), 3, 2)
    out = pd.DataFrame({"known_at": known, "line_id": t["line_id"].to_numpy(), "side": t["side"].to_numpy(),
                        "touches": touches, "bar": m1.index[j],
                        "price": t["b_price"].to_numpy(float) + t["slope"].to_numpy(float) * (j - t["x0"].to_numpy()),
                        "line_known_at": t["known_at"].to_numpy()})
    return out.sort_values(["known_at", "line_id"], kind="stable", ignore_index=True)[list(BREAK_COLUMNS)]


def triangles(m1: pd.DataFrame, ruler=None, table: pd.DataFrame | None = None,
              big_move_adr: float | None = None) -> pd.DataFrame:
    """The 1-minute triangle (see the module text). big_move_adr None while "big move" is open (B2-02): every shape,
    each with the move it measured (`big_move`, `after_big_move`)."""
    check_bars(m1, TF, ("high", "low", "close"))
    ruler = _ruler(m1, ruler)
    big = settings.get("big_move_adr") if big_move_adr is None else big_move_adr
    return PT.triangles(m1, TF, ruler, lines(m1, ruler) if table is None else table,
                        need_big_move=big is not None, big_move_adr=big)


# ================================================================================================== the check

def _alive(start, end, side, want_side, i) -> np.ndarray:
    return (np.asarray(start) <= i + 1) & (np.asarray(end) > i) & (np.asarray(side) == want_side)


def between(bk: pd.DataFrame, i: int, lo: float, hi: float, side: str) -> pd.DataFrame:
    """The 1-minute lines alive at the close of candle i - known by then (start <= i + 1), not taken or gone on it or
    before - on `side`, priced strictly between lo and hi on candle i (a cent inside each). Adds `price`; nearest to
    hi first for side "low" (the first one price meets coming down), nearest to lo first for side "high"."""
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', not {side!r}")
    if not len(bk):
        return bk.assign(price=pd.Series(dtype="float64"))
    if "longest" in bk.attrs:                         # a book() in start order: only lines that can still be alive
        st = bk["start"].to_numpy()
        bk = bk.iloc[int(np.searchsorted(st, i - bk.attrs["longest"], "left")):
                     int(np.searchsorted(st, i + 1, "right"))]
    sub = bk[_alive(bk["start"], bk["end"], bk["side"], side, i)]
    px = sub["b_price"].to_numpy(float) + sub["slope"].to_numpy(float) * (i - sub["x0"].to_numpy())
    ok = (px > lo + INSIDE) & (px < hi - INSIDE)
    out = sub[ok].assign(price=px[ok])
    return out.sort_values("price", ascending=side == "high", kind="stable")


def between5(l5: pd.DataFrame, i5: int, lo: float, hi: float, side: str) -> pd.DataFrame:
    """The same for lines of the 5-minute / hour chart put on the 5-minute candles (trendlines.on_m5): alive at the
    close of 5-minute candle i5, priced between lo and hi on it. Adds `price` and `taken5` / `end5` (5-minute candles)."""
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', not {side!r}")
    if not len(l5):
        return l5.assign(price=pd.Series(dtype="float64"))
    tk = l5["taken"].to_numpy()
    end = np.minimum(np.where(tk >= 0, tk, np.iinfo(np.int64).max), l5["gone"].to_numpy())
    sub = l5[_alive(l5["start"], end, l5["side"], side, i5)]
    px = TL.price5(sub, i5)
    ok = (px > lo + INSIDE) & (px < hi - INSIDE)
    out = sub[ok].assign(price=px[ok])
    return out.sort_values("price", ascending=side == "high", kind="stable")


def first_take(rows: pd.DataFrame, after: int) -> tuple[pd.Series, int] | None:
    """Of these lines (a between() / between5() answer at candle `after`), the one taken first after that candle, and
    the candle that took it. Two taken on the same candle: the first in `rows` (the one price meets first). None: none
    of them is taken before it is gone."""
    if not len(rows):
        return None
    tk = rows["taken"].to_numpy()
    ok = tk > after
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[np.argmin(tk[ok])])
    return rows.iloc[k], int(tk[k])
