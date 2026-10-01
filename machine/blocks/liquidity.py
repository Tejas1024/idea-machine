"""Liquidity (L6.2): which levels are liquidity, what each swing is for, when a level is taken, and which comes first.

His words:
  strategy.md section 1: "levels where stops cluster; grabbed first, then the real move. Recent > older" · "Ranking:
      Equal > Trendline > Session." · "Every sweep counts - even 1 point through, no close needed. Once a high or low is
      swept, it is finished: it is not used as liquidity a second time (zones follow their own rule)" · "Swing highs /
      lows are TARGETS, not entry liquidity".
  LEARNED #86 (the zone entry): "swing high I would not consider as a liquidity" - his three takes were a session low,
      a trendline and equal highs. His OBVIOUS liquidity = settings "obvious_liquidity_kinds" (#84, #86).
  LEARNED #91: "I'll consider those swing high or low as liquidity for quick targets, when higher time frame is against
      my trade, but not to enter or any other reason".
  LEARNED #93: "They can we can be considered them as a liquidity, but they are not a very good way to enter the trade."
  LEARNED #95: "Even 1 point above the liquidity is considered as liquidity grab."
  LEARNED #129: "once it poke poke on the any uh, high or low then that's closed that is that's done that that is
      invalid data".
  LEARNED #130: "two days is correct ... Always it will look for a new session highs and lows".

The blocks:
  obvious_liquidity()  his obvious kinds: session highs / lows (machine/blocks/sessions.py, the final high and low of
                       each window, known at the window's end), equal highs / lows (machine/blocks/equal_levels.py)
                       on the 5-minute, hour and day charts (#90: "the same on every chart"), and trendlines
                       (machine/blocks/trendlines.py, L6b.1: three touches, any angle) on the 5-minute and hour charts
                       (#9: "his lines are drawn on the HOUR chart too" - the old engine's two). A trendline's price
                       MOVES: its row carries `slope` (price per 5-minute candle) and `x0` (the 5-minute candle where
                       its price is `price`); a flat level has slope 0 and no x0. A line is taken by a wick more than
                       `beyond_usd` through it (settings "line_taken_through_adr" - a touch AT the line is not a take,
                       #94) and is gone, not taken, once it drifts `far_usd` from the close (settings "line_far_adr").
                       A trendline poked through between its 3rd touch and the moment it is known is never in the book.
                       An hour / day pair at the same price as a live 5-minute pair is the same level: dropped.
  swing_roles()        every swing (2 candles each side) with its JOB (machine/rules.py):
                         5-minute  a target only, and only when the higher timeframe is against the trade (#91);
                                   "recent" = settings "recent_swing_bars" (one day, Claude's)
                         1-hour    a target (any size), never an entry (#93)
                         day, week may start a setup, and a target ("only day and week swings start a setup")
  book()               both together: one row per level when it is born, and one more row each time an equal level
                       gets another touch ("3>2"). Every row is known when it is written.
  sweeps()             the FIRST 5-minute candle that pokes past each level (a wick, no close needed - #95), within
                       settings "level_life_bars" of its birth. The level is finished after it (#129). `fresh` = no
                       level of the same kind at the same price (settings "same_level_usd") was swept in the
                       settings "how_long_a_swept_level_stays_dead" candles before (#129 "you can't consider the same
                       level twice", #130 "two days is correct"); only levels that may START a setup count as having
                       used a price. Settings "a_level_swept_once_is_finished": swept (his) / false_break / False.
  lifetimes(), alive() what is alive through a 5-minute candle: known before it opened, not taken by its end, not
                       expired. lifetimes() reads ahead to make every lookup fast; alive(i) uses only what was known by
                       candle i (tests/test_liquidity.py checks it against the same lookup on cut-off bars).
  recent_first()       the order he looks at levels in: the strongest KIND first (Equal > Trendline > Session; a
                       level with 3 or more touches one step up - "3>2"), then the MOST RECENT (#130, settings
                       "the_most_recent_liquidity_comes_first"), then the nearest.
  of_session()         "the London high" (L6-new, idea #3: "sell when the London high is taken"): the book rows or the
                       sweeps whose level is the high / low of the session windows NAMED - machine/rules.py
                       "what_a_session_high_or_low_is" (#85: "the high and low of this session range, whatever it is,
                       as a liquidity"; strategy.md section 1: "London 12:30–15:30" India time). Every session level
                       carries its window's name in the column `session` (None on every other kind). The level is known
                       when its window ends, so "the London high is taken" can only happen after 15:30 India time; it is
                       any London high still alive (settings "level_life_bars"), the same life every session level has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import sessions as SES
from machine.blocks import trendlines as TL
from machine.blocks._base import bar_close, check_bars, empty, first_hit
from machine.blocks.equal_levels import equal_levels
from machine.blocks.swings import swings

# The order he ranks liquidity in (strategy.md section 1: "Ranking: Equal > Trendline > Session"). The swing ranks are
# the old engine's (old/ml/agent/levels.py LevelBook.RANK: an hour / day / week swing 2 + "strong" 1, a 5-minute
# swing 1) - Claude's, not his; swings are only ever targets or day / week starts.
RANK = {"equal_high": 3, "equal_low": 3, "trendline_res": 2, "trendline_sup": 2, "session_high": 1,
        "session_low": 1, "swing_high": 1, "swing_low": 1}
HTF_SWING_RANK = 3
STRONG_TOUCHES = 3                         # strategy.md "3>2": a level touched 3 times ranks one step up

BOOK_COLUMNS = {"known_at": "datetime64[ns]", "level_id": "object", "event": "object", "kind": "object",
                "tf": "object", "side": "object", "price": "float64", "first_bar": "datetime64[ns]",
                "touch_bar": "datetime64[ns]", "touches": "int64", "rank": "int64", "can_start": "bool",
                "target_when": "object", "life_bars": "int64", "label": "object", "slope": "float64",
                "x0": "datetime64[ns]", "beyond_usd": "float64", "far_usd": "float64", "session": "object"}
FLAT = {"slope": 0.0, "x0": pd.NaT, "beyond_usd": 0.0, "far_usd": np.nan}      # a level whose price does not move
SWEEP_COLUMNS = {"known_at": "datetime64[ns]", "level_id": "object", "kind": "object", "tf": "object",
                 "side": "object", "price": "float64", "level_known_at": "datetime64[ns]", "bar": "datetime64[ns]",
                 "can_start": "bool", "fresh": "bool", "same_as": "object", "session": "object"}


def _life() -> int:
    return int(settings.get("level_life_bars"))


def _frame(rows: dict | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not len(df):
        return empty(BOOK_COLUMNS)
    for col, v in FLAT.items():
        if col not in df:
            df[col] = v
    if "session" not in df:
        df["session"] = None                   # only a session high / low has a session window
    df["x0"] = pd.to_datetime(df["x0"])
    return df[list(BOOK_COLUMNS)]


TRENDLINE_CHARTS = ("5m", "1h")            # the old engine's two (levels.trendlines + htf_trendlines 'H1'), his #9


# ------------------------------------------------------------------------------------------- obvious liquidity (#86)
def obvious_liquidity(m5: pd.DataFrame, h1: pd.DataFrame | None = None, d1: pd.DataFrame | None = None,
                      min_apart: int | None = None) -> pd.DataFrame:
    """His obvious liquidity (settings "obvious_liquidity_kinds"): session highs / lows, equal highs / lows and
    trendlines on the 5-minute chart; equal highs / lows on the hour and day charts and trendlines on the hour chart,
    when their bars are given. Book rows. min_apart: a version of settings "equal_levels_min_apart_bars" (None = the
    setting)."""
    check_bars(m5, "5m", ("high", "low", "utc"))
    kinds = set(settings.get("obvious_liquidity_kinds"))
    parts = []
    if {"session_high", "session_low"} & kinds:
        parts.append(_sessions(SES.final(m5, "5m"), kinds))
    eq = [_equal(equal_levels(m5, "5m", min_apart=min_apart), "5m")]
    for bars, tf in ((h1, "1h"), (d1, "1d")):
        if bars is not None and tf in rules.get("what_equal_highs_and_lows_are")["same_on_every_chart"]:
            eq.append(_equal(equal_levels(bars, tf, min_apart=min_apart), tf))
    eq = pd.concat(eq, ignore_index=True)
    eq = eq[eq["kind"].isin(kinds)]
    parts.append(_drop_same_level(eq, m5))
    if {"trendline_res", "trendline_sup"} & kinds:
        for bars, tf in ((m5, "5m"), (h1, "1h")):
            if bars is not None and tf in TRENDLINE_CHARTS:
                t = _trendlines(TL.lines(bars, tf), m5)
                parts.append(t[t["kind"].isin(kinds)])
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else empty(
        BOOK_COLUMNS)
    return out.sort_values(["known_at", "kind", "first_bar"], kind="stable", ignore_index=True)


def _trendlines(lines: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """The trendlines (3 touches) of one chart as book rows. A line whose wick-through (settings
    "line_taken_through_adr") came between its 3rd touch and the moment it is known was taken before anyone could see
    it: it never enters the book."""
    t3 = lines[lines["touches"] == 3]
    if not len(t3):
        return empty(BOOK_COLUMNS)
    on = TL.on_m5(t3, m5)
    on = on[(on["taken"] < 0) | (on["taken"] >= on["start"])]
    if not len(on):
        return empty(BOOK_COLUMNS)
    through, far = float(settings.get("line_taken_through_adr")), float(settings.get("line_far_adr"))
    kind = np.where(on["side"] == "high", "trendline_res", "trendline_sup")
    pre = [("" if tf == "5m" else f"{tf} ") for tf in on["tf"]]
    x0 = m5.index[np.minimum(on["x0"].to_numpy(), len(m5) - 1)]
    return _frame({
        "known_at": on["known_at"].to_numpy(), "level_id": on["line_id"].to_numpy(), "event": "born", "kind": kind,
        "tf": on["tf"].to_numpy(), "side": on["side"].to_numpy(), "price": on["b_price"].to_numpy(float),
        "first_bar": on["a_bar"].to_numpy(), "touch_bar": on["c_bar"].to_numpy(), "touches": 3,
        "rank": [RANK[k] for k in kind], "can_start": True, "target_when": "always", "life_bars": _life(),
        "label": [f"{p}{w} trendline" for p, w in zip(pre, on["way"])], "slope": on["slope5"].to_numpy(float),
        "x0": x0, "beyond_usd": through * on["ruler"].to_numpy(float), "far_usd": far * on["ruler"].to_numpy(float)})


def _sessions(fin: pd.DataFrame, kinds: set) -> pd.DataFrame:
    rows = []
    for side, col, bar in (("high", "high", "high_bar"), ("low", "low", "low_bar")):
        kind = f"session_{side}"
        if kind not in kinds or not len(fin):
            continue
        rows.append(pd.DataFrame({
            "known_at": fin["known_at"],
            "level_id": [f"{kind}:{s}:{d}" for s, d in zip(fin["session"], fin["ist_date"])],
            "event": "born", "kind": kind, "tf": "5m", "side": side, "price": fin[col].astype(float),
            "first_bar": fin[bar], "touch_bar": pd.NaT, "touches": 1, "rank": RANK[kind], "can_start": True,
            "target_when": "always",
            "life_bars": _life(), "label": [f"{s} session {side}" for s in fin["session"]],
            "session": fin["session"]}))
    return _frame(pd.concat(rows, ignore_index=True)) if rows else empty(BOOK_COLUMNS)


def _equal(e: pd.DataFrame, tf: str) -> pd.DataFrame:
    if not len(e):
        return empty(BOOK_COLUMNS)
    kind = "equal_" + e["side"]
    pre = "" if tf == "5m" else f"{tf} "
    return _frame({
        "known_at": e["known_at"], "level_id": e["level_id"],
        "event": np.where(e["touches"] == 2, "born", "touch"), "kind": kind, "tf": tf, "side": e["side"],
        "price": e["price"], "first_bar": e["first_bar"], "touch_bar": e["touch_bar"], "touches": e["touches"],
        "rank": RANK["equal_high"],
        "can_start": True, "target_when": "always", "life_bars": _life(),
        "label": [f"{pre}equal {s}s x{t}" for s, t in zip(e["side"], e["touches"])]})


def _drop_same_level(eq: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """An hour / day equal level at the same side and price (to the cent) as a 5-minute equal level that is alive
    when it is born is the same two lows: one level, the 5-minute one (it is known first). The old engine's
    _drop_same_equal."""
    if not len(eq) or (eq["tf"] == "5m").all():
        return eq
    born = eq[eq["event"] == "born"]
    five = born[born["tf"] == "5m"]
    start = m5.index.searchsorted(pd.DatetimeIndex(five["known_at"]), side="left")
    end = np.minimum(start + five["life_bars"].to_numpy(), len(m5))
    h, lo = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
    hit = np.where(five["side"] == "high", first_hit(h, start, five["price"].to_numpy(), ">", ends=end),
                   first_hit(lo, start, five["price"].to_numpy(), "<", ends=end))
    gone = np.where(hit >= 0, hit, end)                    # the 5-minute candle it was taken on (or expired at)
    by = {}
    for s, p, a, g in zip(five["side"], five["price"].round(2), start, gone):
        by.setdefault((s, p), []).append((a, g))
    drop = set()
    for _, r in born[born["tf"] != "5m"].iterrows():
        i = int(m5.index.searchsorted(r["known_at"], side="left"))
        if any(a <= i < g for a, g in by.get((r["side"], round(r["price"], 2)), ())):
            drop.add(r["level_id"])
    return eq[~eq["level_id"].isin(drop)]


# ------------------------------------------------------------------------------------------------- swing roles (#91, #93)
def swing_roles(bars_by_tf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every swing of the charts given ({"5m": ..., "1h": ..., "1d": ..., "1w": ...}) with its job. Book rows:
    can_start = may start a setup; target_when = "always" or "higher_timeframe_against" (#91)."""
    start_tfs = tuple(rules.get("an_hour_swing_is_a_target_not_an_entry"))
    five_target_only = bool(rules.get("a_recent_5_minute_swing_is_a_target_only"))
    parts = []
    for tf, bars in bars_by_tf.items():
        if bars is None:
            continue
        s = swings(bars, tf)
        if not len(s):
            continue
        if tf == "5m":
            if not five_target_only:
                raise RuntimeError("rules.py a_recent_5_minute_swing_is_a_target_only must be True (#91)")
            can, when, life, rank = False, "higher_timeframe_against", int(settings.get("recent_swing_bars")), 1
        elif tf in start_tfs:
            can, when, life, rank = True, "always", _life(), HTF_SWING_RANK
        else:                                             # the hour chart (#93) - and any chart he never named
            can, when, life, rank = False, "always", _life(), HTF_SWING_RANK
        kind = "swing_" + s["side"]
        parts.append(_frame({
            "known_at": s["known_at"], "level_id": [f"{k}:{tf}:{t:%Y-%m-%d %H:%M}" for k, t in zip(kind, s["bar"])],
            "event": "born", "kind": kind, "tf": tf, "side": s["side"], "price": s["price"], "first_bar": s["bar"],
            "touch_bar": pd.NaT, "touches": 1, "rank": rank, "can_start": can, "target_when": when, "life_bars": life,
            "label": [f"{tf} swing {x}" for x in s["side"]]}))
    if not parts:
        return empty(BOOK_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["known_at", "tf", "side"], kind="stable", ignore_index=True)


def book(m5: pd.DataFrame, h1=None, d1=None, w1=None, min_apart: int | None = None) -> pd.DataFrame:
    """His obvious liquidity + every swing with its job, in the order they became known."""
    out = pd.concat([obvious_liquidity(m5, h1, d1, min_apart), swing_roles({"5m": m5, "1h": h1, "1d": d1, "1w": w1})],
                    ignore_index=True)
    return out.sort_values(["known_at", "kind", "first_bar"], kind="stable", ignore_index=True)


# ------------------------------------------------------------------------------------------ sweeps (#95, #129, #130)
def _born(levels: pd.DataFrame) -> pd.DataFrame:
    return levels[levels["event"] == "born"].drop_duplicates("level_id")


def _paths(m5: pd.DataFrame, b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every born level: start (the first 5-minute candle after it is known), hit (the candle that takes it, -1 =
    none), end (the candle it ends: taken, or expired / drifted away) and x0 (a line's anchor candle; 0 for a flat
    level). A flat level is taken by any poke past it (#95). A line is taken by a wick more than beyond_usd through
    it; it ends without a take once its projection is more than far_usd from the close (the old engine's LINE_FAR_ADR:
    a drifting projection is not liquidity any more). Reads ahead - sweeps() and lifetimes() only."""
    h, lo, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(m5)
    start = m5.index.searchsorted(pd.DatetimeIndex(b["known_at"]), side="left")   # the first candle after it is known
    end = np.minimum(start + b["life_bars"].to_numpy(np.int64), n)
    price = b["price"].to_numpy(float)
    slope = b["slope"].to_numpy(float) if "slope" in b else np.zeros(len(b))
    x0t = pd.to_datetime(b["x0"]) if "x0" in b else pd.Series(pd.NaT, index=b.index)
    line = x0t.notna().to_numpy()
    x0 = np.zeros(len(b), np.int64)
    if line.any():
        x0[line] = m5.index.searchsorted(pd.DatetimeIndex(x0t[line]), "left")
    high = (b["side"] == "high").to_numpy()
    hit = np.full(len(b), -1, np.int64)
    f = ~line
    hit[f & high] = first_hit(h, start[f & high], price[f & high], ">", ends=end[f & high])   # "even 1 point" (#95)
    hit[f & ~high] = first_hit(lo, start[f & ~high], price[f & ~high], "<", ends=end[f & ~high])
    beyond = b["beyond_usd"].to_numpy(float) if "beyond_usd" in b else np.zeros(len(b))
    far = b["far_usd"].to_numpy(float) if "far_usd" in b else np.full(len(b), np.nan)
    for k in np.flatnonzero(line):
        s0, e0 = int(start[k]), int(end[k])
        if s0 >= e0:
            continue
        j = np.arange(s0, e0)
        at = price[k] + slope[k] * (j - x0[k])
        poke = (h[s0:e0] - at) if high[k] else (at - lo[s0:e0])
        t = np.flatnonzero(poke > beyond[k] + 1e-9)
        g = np.flatnonzero(np.abs(c[s0:e0] - at) > far[k]) if np.isfinite(far[k]) else np.array([], int)
        tj = int(t[0]) if len(t) else e0 - s0
        gj = int(g[0]) if len(g) else e0 - s0
        if tj < e0 - s0 and tj <= gj:
            hit[k] = s0 + tj                    # taken (on the candle it drifted too, the poke comes first)
        elif gj < e0 - s0:
            end[k] = s0 + gj                    # gone before anything took it
    end = np.where(hit >= 0, hit, end)
    return start, hit, end, x0


def sweeps(m5: pd.DataFrame, levels: pd.DataFrame, same_kind: bool = True) -> pd.DataFrame:
    """The candle that takes each level (see the module text). One row per level taken, known at that candle's close.
    same_kind=False only measures the other reading of "the same level" (any kind at that price)."""
    check_bars(m5, "5m")
    b = _born(levels)
    if not len(b):
        return empty(SWEEP_COLUMNS)
    h, lo, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    start, hit, _, x0 = _paths(m5, b)
    ok = hit >= 0
    s = b[ok]
    j = hit[ok]
    px = s["price"].to_numpy(float) + s["slope"].to_numpy(float) * (j - x0[ok])     # a line: its price on that candle
    out = pd.DataFrame({"known_at": bar_close(m5.index[j], "5m"), "level_id": s["level_id"].to_numpy(),
                        "kind": s["kind"].to_numpy(), "tf": s["tf"].to_numpy(), "side": s["side"].to_numpy(),
                        "price": px, "level_known_at": s["known_at"].to_numpy(),
                        "bar": m5.index[j], "can_start": s["can_start"].to_numpy(),
                        "session": s["session"].to_numpy() if "session" in s else None, "_j": j})
    out = out.sort_values(["_j", "level_id"], kind="stable", ignore_index=True)
    fresh, same = _fresh(out, h, lo, c, same_kind)
    out["fresh"], out["same_as"] = fresh, same
    return out.drop(columns="_j")[list(SWEEP_COLUMNS)]


def _fresh(sw: pd.DataFrame, h, lo, c, same_kind: bool = True) -> tuple[np.ndarray, list]:
    """#129: a sweep is not fresh when a level of the same KIND (and side) that may start a setup was swept at the
    same price (settings same_level_usd) in the dead window before it - strictly earlier candles only.

    Same kind is the old engine's liq_once (it kept a list per kind), and his T3 answer #31 keeps it: an equal-lows
    pair that formed right at a session low price had just poked was a "yes" (inbox 2026-09-28-1151). same_kind=False
    (any kind) is only for measuring that (plan/work/L6-differences.md)."""
    mode = settings.get("a_level_swept_once_is_finished")
    dead_for = int(settings.get("how_long_a_swept_level_stays_dead"))
    tol = float(settings.get("same_level_usd"))
    window = int(settings.get("false_break_close_back_bars"))
    n = len(sw)
    fresh = np.ones(n, bool)
    same: list = [None] * n
    if not mode or not n:
        return fresh, same
    j = sw["_j"].to_numpy()
    price = sw["price"].to_numpy()
    groups = [(sw["side"] == s).to_numpy() for s in ("high", "low")]
    if same_kind:
        groups = [(sw["kind"] == k).to_numpy() for k in sw["kind"].unique()]
    for m in groups:
        if not m.any():
            continue
        used = np.flatnonzero(m & sw["can_start"].to_numpy())
        side = sw["side"].to_numpy()[m][0]
        # the candle from which an earlier sweep counts as having used its price: its own candle ("swept"), or the
        # candle that closed back ("false_break") - never later than the sweep it is judged against (no peeking)
        if mode == "false_break":
            ready = np.array([_false_break(h, lo, c, int(j[u]), side, float(price[u]), window) for u in used],
                             dtype=np.int64)
        else:
            ready = j[used]
        uj, up = j[used], price[used]
        for q in np.flatnonzero(m):
            a, z = np.searchsorted(uj, j[q] - dead_for, "left"), np.searchsorted(uj, j[q], "left")
            if a >= z:
                continue
            close = np.flatnonzero((np.abs(up[a:z] - price[q]) <= tol + 1e-9)
                                   & (ready[a:z] >= 0) & (ready[a:z] < j[q]))
            if len(close):
                fresh[q] = False
                same[q] = sw["level_id"].iat[int(used[a + close[-1]])]
    return fresh, same


def _false_break(h, lo, c, j, side, level, window) -> int:
    """The old engine's reading (entries._false_break): the candle, within `window` candles, that CLOSES back on the
    side price came from before any candle closes beyond the level; -1 if none (a real break, or no close back)."""
    for k in range(j, min(len(c), j + window + 1)):
        if side == "high":
            if c[k] > level:
                return -1
            if c[k] < level:
                return k
        else:
            if c[k] < level:
                return -1
            if c[k] > level:
                return k
    return -1


# ------------------------------------------------------------------------------------ "the London high" (L6-new)
def of_session(rows: pd.DataFrame, only) -> pd.DataFrame:
    """The rows (book rows, sweeps, alive levels) whose level is the high or low of one of his session windows named
    in `only` ("London"; "Asian", "NY" are read as he writes them - sessions.name_of; any other name is refused).
    Nothing is added to a row and nothing later is read: a row is kept or dropped by its own level alone."""
    if isinstance(only, str):
        only = (only,)
    want = {SES.name_of(x) for x in only}
    if not want:
        raise ValueError(f"name at least one session: {', '.join(SES.names())}")
    if "session" not in rows:
        raise ValueError("these rows do not say which session a level came from (no 'session' column): make them "
                         "with liquidity.book / obvious_liquidity / sweeps")
    return rows[rows["session"].isin(want)]


# ------------------------------------------------------------------------------------------- alive + recent first
def lifetimes(m5: pd.DataFrame, levels: pd.DataFrame, sweep_table: pd.DataFrame) -> pd.DataFrame:
    """One row per level: start (the first 5-minute candle after it is known), end (the candle that takes it, or where
    it expires - a line also where it drifted too far from price) as candle numbers of m5, x0pos (a line's anchor
    candle), and the touch history for "3>2". READS AHEAD - only alive() may use it."""
    b = _born(levels).copy()
    start, _, gone, x0 = _paths(m5, b)
    sw = sweep_table.set_index("level_id")

    def candle(times: pd.Series, missing: int) -> np.ndarray:
        t = pd.to_datetime(times)
        out = np.full(len(t), missing, np.int64)
        has = t.notna().to_numpy()
        if has.any():
            out[has] = m5.index.searchsorted(pd.DatetimeIndex(t[has]), "left")
        return out

    tj = candle(b["level_id"].map(sw["bar"].to_dict()), -1)
    b["start"] = start
    b["taken"] = tj
    b["end"] = np.where(tj >= 0, tj, gone)
    b["x0pos"] = x0
    b["fresh"] = b["level_id"].map(sw["fresh"]).fillna(True).astype(bool) if len(sw) else True
    t = levels[levels["event"] == "touch"]
    strong = t[t["touches"] >= STRONG_TOUCHES].drop_duplicates("level_id").set_index("level_id")["known_at"]
    b["strong_from"] = candle(b["level_id"].map(strong.to_dict()), np.iinfo(np.int64).max)
    return b.reset_index(drop=True)


def alive(life: pd.DataFrame, i: int) -> pd.DataFrame:
    """The levels alive THROUGH 5-minute candle i: known before it opened, not taken on it or before, not expired.
    strong = 3 or more touches known before candle i opened. `price` is each level's price ON candle i (a trendline's
    moves with the candle; a flat level's is its price)."""
    m = (life["start"].to_numpy() <= i) & (i < life["end"].to_numpy())
    out = life[m].copy()
    out["strong"] = out["strong_from"].to_numpy() <= i
    if "slope" in out and len(out):
        s = out["slope"].to_numpy(float)
        if (s != 0).any():
            out["price"] = out["price"].to_numpy(float) + s * (i - out["x0pos"].to_numpy())
    return out


def recent_first(rows: pd.DataFrame, side: str, price: float, *, kinds=None, include_targets: bool = False,
                 higher_timeframe_against: bool = False, max_n: int | None = None) -> pd.DataFrame:
    """The alive levels ABOVE price (side "high": highs above) or BELOW it (side "low"), in his order: strongest kind
    first (a strong level one step up), then the most recent, then the nearest (settings
    "the_most_recent_liquidity_comes_first"; False = kind then nearest). Levels that cannot start a setup are left out
    unless include_targets; a 5-minute swing only when higher_timeframe_against (#91). One level per price (cents)."""
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', not {side!r}")
    r = rows[rows["side"] == side]
    r = r[(r["price"] > price) if side == "high" else (r["price"] < price)]
    if kinds is not None:
        r = r[r["kind"].isin(kinds)]
    if not include_targets:
        r = r[r["can_start"]]
    elif not higher_timeframe_against:
        r = r[r["target_when"] != "higher_timeframe_against"]
    strong = r["strong"].astype(int) if "strong" in r else 0
    r = r.assign(_rank=r["rank"] + strong, _dist=(r["price"] - price).abs())
    keys, asc = ["_rank"], [False]
    if settings.get("the_most_recent_liquidity_comes_first"):
        keys, asc = keys + ["known_at"], asc + [False]
    r = r.sort_values(keys + ["_dist"], ascending=asc + [True], kind="stable")
    r = r[~r["price"].round(2).duplicated()]
    return (r if max_n is None else r.head(max_n)).drop(columns=["_rank", "_dist"])
