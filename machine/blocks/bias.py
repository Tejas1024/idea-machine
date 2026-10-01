"""His direction on the hour, day and week charts (L6b.3): the bias, the real all-time high, and the doji.

His words:
  LEARNED #70 (Q69, 15 Sep): "we will wait to another day to check whether it will form a fair value gap or not ...
      then it will become the bullish or else it is a neutral" · "the same rule as ... five minutes, like fair value
      gap, all the things, order block, everything apply here as well."
  LEARNED #72: "only all time high means only all time high" · "once after you take the liquidity, what is our target
      is the nearest next liquidity ... is it moving to [the] nearest target of liquidity in higher time frame like
      daily candle? ... So that can be the rule to be biased."
  LEARNED #73: "when it is analyzing the daily candle, then five minute won't come into play. Only after daily bias is
      fixed, then only it will go to the five minutes" - the bias is read on that chart's own candles only.
  LEARNED #74: the machine's daily reading drawn on 4 new charts - "Yes, reading it perfectly."
  LEARNED #69: "if you want to know the bias in one hour we can apply the same thing" (the hour chart too).
  LEARNED #132: the flat-bias fallback - "if it's good then ok, or else no need" -> NO fallback (nothing beat a coin,
      strategy-understanding section 66bq): a chart with no trade on it is NEUTRAL, never "up because it rose".
  strategy.md section 6 (line 104, T1): "my 5-min entry read on the daily chart alone - a daily level taken -> a break
      -> a gap -> price comes back to the gap = bias that way. Until price comes back: neutral. ... At the REAL all-time
      high (not a 52-week high) the daily is up."

The rule, on ONE chart's candles (hour, day or week):
  1. its liquidity = its untaken swing highs / lows (machine/blocks/swings.py - his glossary swing, 2 candles each
     side, on every chart: plan/work/rules-missing.md R14) and its equal highs / lows (equal_levels.py, his rule, the
     same on the 5-minute, hour and day charts - #90). Trendlines join in L6b.1 (not built: a trendline taken, or
     broken and retested - #69, #70 D - is not seen yet).
  2. a GRAB: a candle's wick takes one or more of them and the candle closes back inside the deepest one it took (the
     version he confirmed, #74). Every level it touched is used up (#129).
  3. his BREAK of structure (machine/blocks/structure.py, #56-#60) within settings "bias_wait_bars" candles.
  4. the FIRST gap of the breaking move (machine/blocks/fvg.py gaps_of_move rank 0: #61, #62, #101, #131).
  5. the trade is ON and the chart's bias is its way. settings "bias_waits_for_the_return": False = the reading drawn
     and confirmed (#74); True = it also waits for price to come back to the gap's near edge (strategy.md line 104,
     #70 B). Two of his sources, so a version, not a guess.
  6. the bias HOLDS while the trade travels to that chart's next liquidity and past it - the target trails (#82;
     strategy.md section 10 "hold until a proper opposite reversal"). It ends only when
       - a candle CLOSES back beyond the grab's extreme (the trade failed) -> NEUTRAL, or
       - a new trade the other way turns on.
     A new grab + break still waiting for its gap is NEUTRAL, whichever side it is on (#70 A, B: "now it is neutral").
  7. on the day and week charts, a close within settings "near_all_time_high" of the REAL all-time high = UP (#71, #72,
     strategy.md section 12). The hour chart has no such rule (the old engine's too).
  8. settings "bias_doji" (four-boxes B3-06 - both readings, a version): no_change (default; his #70 C read a doji
     candle "75% bullish") or neutral (a doji as the last closed candle makes the chart NEUTRAL: "a doji weakens it").

  htf_bias(bars, tf)       one row per candle of that chart, known at its close: bias (+1 up, -1 down, 0 neutral), why
                           (plain words), and the trade behind it (the level taken, the grab's extreme, the break, the
                           gap's near edge, the next target) - what a picture of "the daily trade" draws (#74).
  bias_at(table, times)    the bias known at each moment: the last candle of that chart CLOSED by then (#73).
  all_time_high(bars, tf)  the running all-time high through each candle, from gold's September 2011 record (settings
                           "all_time_high_before_2016", cited there), and whether the close is AT it.
  new_highs(bars, tf)      the candles that made a new all-time high (few: none from 2016 to July 2020).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import fvg as FVG
from machine.blocks._base import bar_close, check_bars
from machine.blocks.equal_levels import equal_levels
from machine.blocks.structure import BreakSearch
from machine.blocks.swings import swings

UP, DOWN, NEUTRAL = 1, -1, 0
WORD = {UP: "up", DOWN: "down", NEUTRAL: "neutral"}
ATH_CHARTS = ("1d", "1w")                 # rule 7: the day and week charts (#72 was asked on the daily)
FIRST_DAY = pd.Timestamp("2016-01-10")    # bars that start later need the highest price before them (all_time_high)


# ============================================================================================== the all-time high

def all_time_high(bars: pd.DataFrame, tf: str, before: float | None = None) -> pd.DataFrame:
    """The REAL all-time high through each candle (#72: "only all time high means only all time high").

    before: the highest price before these bars. None = gold's September 2011 record (settings
    "all_time_high_before_2016") - right only for bars that start in the first days of 2016; later bars must say it.
    Per candle: ath (the highest price up to and including this candle), near (settings "near_all_time_high" x ath),
    at_ath (the close is at or above `near`), new (this candle's high made a new record), known_at (its close)."""
    check_bars(bars, tf, ("high", "close"))
    if before is None:
        if len(bars) and bars.index[0] > FIRST_DAY:
            raise ValueError(f"the bars start {bars.index[0]:%d %b %Y}: give `before` - the highest price before them "
                             "(the September 2011 record is only right for bars that start in January 2016)")
        before = float(settings.get("all_time_high_before_2016"))
    h = bars["high"].to_numpy(dtype=float)
    ath = np.maximum(np.maximum.accumulate(h), float(before)) if len(h) else np.array([])
    prev = np.r_[float(before), ath[:-1]] if len(h) else np.array([])
    near = float(settings.get("near_all_time_high")) * ath
    out = pd.DataFrame({"known_at": bar_close(bars.index, tf), "ath": ath, "near": near,
                        "at_ath": bars["close"].to_numpy(dtype=float) >= near, "new": h > prev}, index=bars.index)
    return out


def new_highs(bars: pd.DataFrame, tf: str, before: float | None = None) -> pd.DataFrame:
    """The candles whose high made a new all-time high: bar, high, the record before it, known_at."""
    a = all_time_high(bars, tf, before)
    k = np.flatnonzero(a["new"].to_numpy())
    prev = np.r_[a["ath"].iloc[0] if len(a) else np.nan, a["ath"].to_numpy()[:-1]]
    if len(a):
        prev[0] = float(settings.get("all_time_high_before_2016")) if before is None else float(before)
    return pd.DataFrame({"known_at": a["known_at"].to_numpy()[k], "bar": bars.index[k],
                         "high": bars["high"].to_numpy(dtype=float)[k], "record_before": prev[k]})


# ============================================================================================== the doji

def is_doji(bars: pd.DataFrame) -> np.ndarray:
    """A doji (#70 C, "indecision candle"): a body at most settings "doji_body_share" of the candle's range."""
    o, h, lo, c = (bars[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    rng = h - lo
    return (rng > 0) & (np.abs(c - o) <= float(settings.get("doji_body_share")) * rng)


# ============================================================================================== the bias

TRADE_COLUMNS = ("side", "swept", "extreme", "sweep_bar", "break_bar", "gap_near", "target")


def _level_adds(bars: pd.DataFrame, tf: str, ruler) -> dict[int, list[tuple[int, float]]]:
    """{candle position: [(side, price), ...]} - the levels usable FROM that candle: known before it opened."""
    parts = [swings(bars, tf)[["known_at", "side", "price"]]]
    if tf in rules.get("what_equal_highs_and_lows_are")["same_on_every_chart"]:
        eq = equal_levels(bars, tf, ruler)
        parts.append(eq.loc[eq["touches"] == 2, ["known_at", "side", "price"]])
    lv = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else None
    adds: dict[int, list[tuple[int, float]]] = {}
    if lv is None:
        return adds
    pos = bars.index.searchsorted(pd.DatetimeIndex(lv["known_at"]), side="left")
    for p, s, px in zip(pos, lv["side"], lv["price"].astype(float)):
        if p < len(bars):
            adds.setdefault(int(p), []).append((UP if s == "high" else DOWN, float(px)))
    return adds


def htf_bias(bars: pd.DataFrame, tf: str, ruler=None, ath_before: float | None = None) -> pd.DataFrame:
    """His bias on this chart's candles (see the module text). One row per candle, index = its open time; known_at =
    its close. ruler: a day's range on every candle (for his equal highs / lows; default his ADR from these bars).
    ath_before: the highest price before these bars (all_time_high), day and week charts only."""
    check_bars(bars, tf)
    n = len(bars)
    idx = bars.index
    o, h, lo, c = (bars[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    cols = {"known_at": bar_close(idx, tf), "bias": np.zeros(n, np.int8), "why": np.full(n, "", object)}
    trade_cols = {k: np.full(n, np.nan) for k in ("side", "swept", "extreme", "gap_near", "target")}
    trade_cols["sweep_bar"] = np.full(n, np.datetime64("NaT", "ns"))
    trade_cols["break_bar"] = np.full(n, np.datetime64("NaT", "ns"))
    if n < 5:
        cols["why"][:] = "neutral: too few candles"
        return pd.DataFrame({**cols, **trade_cols, "at_ath": np.zeros(n, bool)}, index=idx)

    wait = int(settings.get("bias_wait_bars"))
    need_return = bool(settings.get("bias_waits_for_the_return"))
    doji_rule = settings.get("bias_doji")
    belong = int(rules.get("gap_must_belong_to_the_breaking_move"))
    adds = _level_adds(bars, tf, ruler)
    search = {UP: BreakSearch(bars, tf, "buy"), DOWN: BreakSearch(bars, tf, "sell")}
    g_all = FVG.gaps(bars, tf)
    g_first = idx.get_indexer(pd.DatetimeIndex(g_all["first_bar"])) if len(g_all) else np.array([], int)
    at_ath = np.zeros(n, bool)
    if tf in ATH_CHARTS:
        at_ath = all_time_high(bars, tf, ath_before)["at_ath"].to_numpy()
    doji = is_doji(bars) if doji_rule == "neutral" else np.zeros(n, bool)

    lows: list[float] = []                  # untaken lows (sorted), untaken highs (sorted)
    highs: list[float] = []
    pend: dict | None = None                # a grab waiting for its break and gap
    trade: dict | None = None               # the trade that is ON
    ended = ""                              # why the last trade ended (words for `why`)

    def nearest(side: int, px: float) -> float:
        if side == UP:
            k = bisect_right(highs, px)
            return highs[k] if k < len(highs) else np.nan
        k = bisect_left(lows, px) - 1
        return lows[k] if k >= 0 else np.nan

    for i in range(n):
        for side, px in adds.get(i, ()):
            insort(highs if side == UP else lows, px)
        # the levels this candle's wick took (#129: used up, whatever the close)
        took_lo: list[float] = []
        if lows and lows[-1] > lo[i]:
            k = bisect_right(lows, lo[i])
            took_lo = lows[k:]
            del lows[k:]
        took_hi: list[float] = []
        if highs and highs[0] < h[i]:
            k = bisect_left(highs, h[i])
            took_hi = highs[:k]
            del highs[:k]

        # the trade that is on: failed, or its target taken (the target trails - #82)
        if trade is not None:
            s = trade["side"]
            if (s == UP and c[i] < trade["extreme"]) or (s == DOWN and c[i] > trade["extreme"]):
                trade, ended = None, f"no trade (the last one failed {idx[i]:%d %b %Y}: a close back past its grab)"
            elif np.isfinite(trade["target"]) and ((s == UP and h[i] >= trade["target"]) or
                                                   (s == DOWN and lo[i] <= trade["target"])):
                trade["target"] = nearest(s, c[i])

        # a new grab (the high side wins when one candle took both - the old engine's order)
        for side, took, ok, ext in ((UP, took_lo, bool(took_lo) and c[i] > min(took_lo or [0.0]), lo[i]),
                                    (DOWN, took_hi, bool(took_hi) and c[i] < max(took_hi or [0.0]), h[i])):
            if ok:
                brk = search[side].after(i, until=i + wait)
                pend = {"side": side, "i": i, "extreme": float(ext),
                        "swept": float(min(took) if side == UP else max(took)),
                        "brk": None if brk is None else int(idx.get_loc(brk["bar"])), "broke": False, "rows": None}

        # the grab waiting: its break, then its first gap
        if pend is not None and i > pend["i"]:
            s = pend["side"]
            failed = (s == UP and c[i] < pend["extreme"]) or (s == DOWN and c[i] > pend["extreme"])
            late = (pend["brk"] is None and i > pend["i"] + wait) or (
                pend["brk"] is not None and i > pend["brk"] + belong and not pend.get("gap"))
            if failed or late:
                pend = None
            else:
                if pend["brk"] is not None and i >= pend["brk"] and not pend["broke"]:
                    pend["broke"] = True
                    a0 = int(np.searchsorted(g_first, pend["i"], "left"))
                    a1 = int(np.searchsorted(g_first, pend["brk"] + belong + 1, "left"))
                    rows = FVG.gaps_of_move(bars, tf, "buy" if s == UP else "sell", idx[pend["i"]],
                                            idx[pend["brk"]], gap_table=g_all.iloc[a0:a1])
                    look = np.maximum(idx.get_indexer(pd.DatetimeIndex(rows["bar"])), pend["brk"])
                    pend["rows"] = list(zip(look, rows["filled_before"], rows["near"].astype(float)))
                if pend["broke"] and not pend.get("gap"):
                    for lk, filled, near in pend["rows"]:
                        if lk > i:
                            break                      # the next gap of the move is not known yet
                        if not filled:
                            pend["gap"] = (int(lk), float(near))
                            break
                if pend.get("gap"):
                    lk, near = pend["gap"]
                    back = i > lk and ((s == UP and lo[i] <= near) or (s == DOWN and h[i] >= near))
                    if back or not need_return:
                        trade = {"side": s, "extreme": pend["extreme"], "swept": pend["swept"],
                                 "sweep_bar": idx[pend["i"]], "break_bar": idx[pend["brk"]], "gap_near": near,
                                 "target": nearest(s, c[i])}
                        pend, ended = None, ""

        waiting = pend is not None and pend["broke"]
        if at_ath[i]:
            b, why = UP, "up: the close is at the real all-time high (#72)"
        elif doji[i]:
            b, why = NEUTRAL, "neutral: a doji (settings bias_doji = neutral)"
        elif waiting:
            b, why = NEUTRAL, (f"neutral: a grab, a break and a gap wait for price to come back to the gap "
                               f"({WORD[pend['side']]})" if pend.get("gap") else
                               f"neutral: a grab and a break wait for their gap ({WORD[pend['side']]})")
        elif trade is None:
            b, why = NEUTRAL, "neutral: " + (ended or "no trade on this chart yet")
        else:
            b, why = trade["side"], (f"{WORD[trade['side']]}: a {'low' if trade['side'] == UP else 'high'} was "
                                     "taken, then a break and a gap - the trade is on")
        cols["bias"][i], cols["why"][i] = b, why
        if trade is not None:
            for key in ("side", "swept", "extreme", "gap_near", "target"):
                trade_cols[key][i] = trade[key]
            trade_cols["sweep_bar"][i] = pd.Timestamp(trade["sweep_bar"]).to_datetime64()
            trade_cols["break_bar"][i] = pd.Timestamp(trade["break_bar"]).to_datetime64()
    return pd.DataFrame({**cols, **trade_cols, "at_ath": at_ath}, index=idx)


def bias_at(table: pd.DataFrame, times) -> np.ndarray:
    """The bias known at each moment: the row of the last candle of that chart that CLOSED by then (0 before any)."""
    t = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(times))).as_unit("ns").asi8
    known = pd.DatetimeIndex(table["known_at"]).as_unit("ns").asi8
    k = np.searchsorted(known, t, side="right") - 1
    vals = table["bias"].to_numpy()
    return np.where(k >= 0, vals[np.maximum(k, 0)], NEUTRAL).astype(np.int8)
