"""Strong zones (L6b.2) - strategy.md section 3, and the three things he does with them: the zone entry (#84, #86), the
break of a zone and its retest (#97, #98, #99), and the zone before the target (double profit, sections 3 and 10).

His words:
  strategy.md section 3: "TIME-at-price (Market Profile/TPO), NOT volume. Hand-drawn 5m (1h stronger). Strong cluster
      only; don't force. Uses: high-prob sweep+FVG at a zone; marks where reversal is likely; target/book-profit;
      double-profit re-entry; hedge point. Break -> S/R flip." · "My main zone is the QUIET one: small candles parked in
      a tight range, recent, that nobody else sees. Rare - that is what makes it special. Two quiet stretches at the
      same price make it stronger. A classic support / resistance touched many times is second priority. Same rule on
      5-min and 1-hour; look back as far as needed. A zone is strongest the first time; once touched and respected
      (2-3 times) it gets weak; it flips when broken (#77, #78, #79, #80, #81)."
  LEARNED #77: "Once. It is touched and uh, respected, then that becomes weak."
  LEARNED #78: "if the price is below that zone, then it will be a resistance and if price goes above or break it, then
      it will become support ... It will only if it breaks the zone, then it will actually support." · "I usually
      consider five minutes and uh, one hour as well."
  LEARNED #80: "B and d are the my kind of zones." (calm 0.54 and 0.69 yes, 0.79 and 0.98 no -> rules.py 0.74)
  LEARNED #81: "If there are two zones exist, this one and that one, I would consider this as you know my main zone."
  strategy.md section 9: "Zone entry (no gap): a strong zone with an obvious liquidity (session high / low, trendline,
      equal highs / lows) just beyond it. Price takes that liquidity and comes back into the zone -> enter AT the zone;
      hedge just past the zone's far edge. The higher timeframe must be with it. Rare (#84, #86)." · "Zone break retest:
      ... Price must really break a strong zone and close through it (tiny wicks don't count), then come back to retest
      it -> enter at the retest. No distance rule. ... If liquidity sits inside the zone, wait for it to be taken, then
      the 5-min gap. Used at the all-time high; there the hedge waits at the next real strong zone (#97, #98, #99)."
  strategy.md section 10: "Zone: book profit at a zone before the target; double-profit re-entry."
  LEARNED #82: "when the first target sometimes not always the next liquidity it can be the strongest zone ... I will
      simply exit there and uh, wait for some bounce back and I can uh, again enter around ... if it is back again and it
      is not giving the clear sign of falling from there then I would enter again for the same trade".

The blocks (definitions from machine/rules.py and machine/settings.py only, likhith.md R11):
  strong_zones(m5, h1)   TIME at price: in each window of settings "zone_windows" (today = the 5-minute candles of the
                         last broker day, week = of the last 5, hour = the HOUR candles of the last 20) every candle's
                         BODY adds one to each price step (settings "zone_bin_adr" of a day's range) it covers; the steps
                         in the top settings "zone_strong_top_percent" merge into bands; a band thinner than settings
                         "zone_min_bins" steps is not a zone. calm = the median range of the hour candles whose body sits
                         in the band / the median range of all the window's hour candles (#79: the candles are SMALL; the
                         measure works on the hour candles he reads). tier 1 = his main, quiet zone (calm at most
                         rules.py what_a_strong_zone_is 0.74); tier 2 = his second priority (at most settings
                         "zone_second_priority_calm"); tier 3 = the rest (settings "zone_tiers" says which are handed on).
                         rank = his order (strategy.md "1h stronger"): today 0, week 1, hour settings
                         "zone_hour_chart_rank". KNOWN when the window's last broker day ends (server midnight).
                         quiet_before = how many EARLIER quiet zones, built from other candles, sat at the same price
                         (a measure, B4-13).
  lives(m5, zones)       each zone's life on the 5-minute chart: born = the first candle after it is known; touches (a
                         candle whose range enters the band after one wholly outside it - sitting inside is one touch);
                         the ROLE of a touch (#78: resistance when price comes from below, support from above); a role
                         used once is spent; the zone ends at the rules.py weak_after_touches-th touch (#77 "two to
                         three times"), when a spent role is used again (settings "zone_spent_role_waits_for_the_break":
                         only if that visit does not break it, #78), on a real flip (a close a full day's range beyond it
                         that is still beyond settings "zone_flip_confirm_bars" candles later - only after price has
                         touched it: rules.py a_zone_only_flips_once_it_is_touched), or when its days run out. READS
                         AHEAD - only alive() and the event tables use it.
  alive(life, i)         the zones alive through 5-minute candle i: known before it opened, not ended on it or before.
  touch_events(life, m5) one row per touch, known at the close of the touch's first candle.
  breaks(m5, life)       the REAL break of a zone (#97, #98): the first 5-minute candle that CLOSES beyond the band by
                         rules.py what_a_real_zone_break_is (0.20 of the zone's own height) having come THROUGH it
                         (settings "zone_break_is_through"): the last close before it that was not inside the band or a
                         tiny bit past it this way (#98: "we draw the zone bit tight") was on the OTHER side, within
                         settings "zone_side_lookback_bars" - not out the side it came from (the zone holding). One each
                         way per zone, while it lives. Known at that close. entry = the broken edge (the retest, the S/R
                         flip); hedge = just past the far edge.
  after_inner_take(...)  #99: liquidity inside the zone - wait for it to be taken, then the first 5-minute gap the
                         trade's way (settings "zone_break_inner_gap_bars"); the hedge just past the low / high the take
                         made (the grab's extreme, his #9 / #33).
  entries(m5, life, sw)  the zone entry (#84, #86): a candle takes one of his obvious levels (a fresh sweep, #129) with
                         a live zone just beyond it - its near edge at most settings "zone_entry_liquidity_adr" of a day's
                         range away (#86: the distance is not his rule), main zone first (#81), then nearest - while the
                         candle stays outside the zone and the zone's role on that side is not used yet (#78). Every
                         such take arms an order: a later one at the same zone is "a newer setup", and the tester keeps
                         one order at a time - his #145 "Cancel and take new". entry = the zone's near edge; hedge = just
                         past its far edge ("the zone low will be my hedge point"). The hour + day (#84) and the
                         unfilled gap (#87) are the form's filter and rule.
  in_path(...)           the zone between the entry and the target (double profit): he books at its near edge and gets
                         back in - at the price he booked (retest) or past its far edge (break). A zone the entry already
                         stands in is not "before the target" (strategy-understanding 48b: the swapped edges once faked
                         +Rs 10 a trade).
  nearest_edge(...)      the nearest live zone edge on one side of a price ("Target = next liquidity / zone", B1-49).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import adr as ADR
from machine.blocks._base import bar_close, check_bars, empty

ZONE_COLUMNS = {"known_at": "datetime64[ns]", "zone_id": "object", "window": "object", "tf": "object",
                "rank": "int64", "lo": "float64", "hi": "float64", "bins": "int64", "calm": "float64",
                "tier": "int64", "src_first": "datetime64[ns]", "src_last": "datetime64[ns]", "life_bars": "int64",
                "ruler": "float64", "quiet_before": "int64"}
TOUCH_COLUMNS = {"known_at": "datetime64[ns]", "zone_id": "object", "bar": "datetime64[ns]", "n": "int64",
                 "role": "object"}
BREAK_COLUMNS = {"known_at": "datetime64[ns]", "zone_id": "object", "direction": "object", "bar": "datetime64[ns]",
                 "edge": "float64", "hedge": "float64", "lo": "float64", "hi": "float64", "height": "float64",
                 "depth": "float64", "came_from": "datetime64[ns]", "tier": "int64", "rank": "int64",
                 "window": "object", "ruler": "float64"}
ENTRY_COLUMNS = {"known_at": "datetime64[ns]", "zone_id": "object", "direction": "object", "bar": "datetime64[ns]",
                 "level_id": "object", "level_kind": "object", "level_tf": "object", "level_price": "float64",
                 "entry": "float64", "hedge": "float64", "lo": "float64", "hi": "float64", "tier": "int64",
                 "rank": "int64", "window": "object", "liq_gap_adr": "float64", "ruler": "float64"}
DAY = pd.Timedelta(days=1)
PAD = 0.01                                # "just past" the far edge: one cent (the old engine's lo - 0.01)
MIN_BIN_USD = 0.05                        # the old engine's floor on a price step (levels.tpo_zones)
MAX_BINS = 4000                           # ... and on the number of steps (a window this wide is not one market)
MIN_WINDOW_BARS = 20                      # ... and a window needs at least 20 candles
WINDOW_RANK = {"today": 0, "week": 1}     # the hour chart's rank: settings "zone_hour_chart_rank" (2)
FIVE = pd.Timedelta(minutes=5)


def windows() -> list[tuple[str, str, int, int, int]]:
    """settings "zone_windows" as (name, chart, days counted, days it lives, rank)."""
    out = []
    for name, tf, days, life in settings.get("zone_windows"):
        rank = int(settings.get("zone_hour_chart_rank")) if name == "hour" else WINDOW_RANK.get(name, 0)
        out.append((str(name), str(tf), int(days), int(life), rank))
    return out


def tiers_on() -> tuple[int, ...]:
    """The tiers handed on (settings "zone_tiers", #81): tiered = his main zone and his second choice; quiet = his main
    zone only; all = every zone (the backtest he asked for, #79)."""
    mode = settings.get("zone_tiers")
    return {"tiered": (1, 2), "quiet": (1,), "all": (1, 2, 3)}[mode]


def _tier(calm: float) -> int:
    if math.isfinite(calm) and calm <= float(rules.get("what_a_strong_zone_is")["calm_max"]):
        return 1
    if math.isfinite(calm) and calm <= float(settings.get("zone_second_priority_calm")):
        return 2
    return 3


def _complete_days(m5: pd.DataFrame) -> pd.DatetimeIndex:
    """The broker days whose 5-minute candles are final: every day but the last, and the last once a candle closes at
    its end (server midnight). A day is known complete at its end."""
    days = m5.index.normalize().unique()
    if len(days) and m5.index[-1] + FIVE < days[-1] + DAY:
        days = days[:-1]
    return days


def _day_ruler(m5: pd.DataFrame) -> pd.Series:
    """The ruler a zone born after day d carries: his ADR of the last settings "adr_ruler_days" finished days, d
    included (machine/blocks/adr.py gives the same number to the day after d)."""
    key = m5.index.normalize()
    rng = m5.groupby(key)["high"].max() - m5.groupby(key)["low"].min()
    return rng.rolling(int(settings.get("adr_ruler_days"))).mean()


def _bands(a: np.ndarray, b: np.ndarray, ruler: float) -> list[tuple[float, float, int]]:
    """Time at price over one window (the old engine's tpo_zones, strategy-code.md section 11): bodies [a, b] vote on
    price steps; the top share of steps merge into bands of at least zone_min_bins steps. [(lo, hi, steps)]."""
    lo, hi = float(a.min()), float(b.max())
    binw = max(float(settings.get("zone_bin_adr")) * ruler, MIN_BIN_USD)
    nb = int((hi - lo) / binw) + 1
    min_bins = int(settings.get("zone_min_bins"))
    if nb < min_bins * 2 or nb > MAX_BINS:
        return []
    i0 = np.clip(((a - lo) / binw).astype(int), 0, nb - 1)
    i1 = np.clip(((b - lo) / binw).astype(int), 0, nb - 1)
    diff = np.zeros(nb + 1)
    np.add.at(diff, i0, 1)
    np.add.at(diff, i1 + 1, -1)
    cnt = np.cumsum(diff)[:nb]
    pos = cnt[cnt > 0]
    thr = np.percentile(pos, 100 - float(settings.get("zone_strong_top_percent"))) if len(pos) else 0.0
    strong = cnt >= max(thr, 1)
    out, i = [], 0
    while i < nb:
        if strong[i]:
            j = i
            while j + 1 < nb and strong[j + 1]:
                j += 1
            if j - i + 1 >= min_bins:
                out.append((lo + i * binw, lo + (j + 1) * binw, j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def strong_zones(m5: pd.DataFrame, h1: pd.DataFrame, ruler=None, tiers: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Every strong zone when it is known (see the module text). h1: the hour candles (they measure calm, and build the
    hour window). ruler: one value per 5-minute candle (the old engine's, to compare) - the value on the window's last
    candle is used; None = his ADR on broker days. tiers: which tiers to keep (None = settings "zone_tiers")."""
    check_bars(m5, "5m")
    check_bars(h1, "1h")
    tiers = tiers_on() if tiers is None else tuple(tiers)
    days = _complete_days(m5)
    rows = []
    if not len(days):
        return empty(ZONE_COLUMNS)
    day_r = _day_ruler(m5) if ruler is None else None
    r5 = None if ruler is None else np.asarray(ruler, dtype=float)
    if r5 is not None and len(r5) != len(m5):
        raise ValueError("the ruler must have one value per 5-minute candle")
    hk = h1.index.normalize()
    ho, hc = h1["open"].to_numpy(float), h1["close"].to_numpy(float)
    h_lo, h_hi = np.minimum(ho, hc), np.maximum(ho, hc)
    h_rng = (h1["high"] - h1["low"]).to_numpy(float)
    src = {"5m": (m5, m5.index.normalize()), "1h": (h1, hk)}
    for name, tf, ndays, life, rank in windows():
        bars, key = src[tf]
        o, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
        body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
        for k in range(ndays - 1, len(days)):
            first, last = days[k - ndays + 1], days[k]
            s, e = key.searchsorted(first, "left"), key.searchsorted(last + DAY, "left")
            if e - s < MIN_WINDOW_BARS:
                continue
            if r5 is None:
                r = float(day_r.get(last, np.nan))
            else:
                r = float(r5[m5.index.searchsorted(last + DAY, "left") - 1])
            if not math.isfinite(r) or r <= 0:
                continue
            hs, he = hk.searchsorted(first, "left"), hk.searchsorted(last + DAY, "left")
            all_rng = np.median(h_rng[hs:he]) if he > hs else np.nan
            for zlo, zhi, nb in _bands(body_lo[s:e], body_hi[s:e], r):
                inb = (h_hi[hs:he] >= zlo) & (h_lo[hs:he] <= zhi)
                calm = float(np.median(h_rng[hs:he][inb]) / all_rng) \
                    if inb.any() and math.isfinite(all_rng) and all_rng > 0 else math.nan
                rows.append({"known_at": last + DAY, "zone_id": f"{name}:{last.date()}:{zlo:.2f}", "window": name,
                             "tf": tf, "rank": rank, "lo": zlo, "hi": zhi, "bins": nb, "calm": calm,
                             "tier": _tier(calm), "src_first": bars.index[s], "src_last": bars.index[e - 1],
                             "life_bars": life * 288, "ruler": r})
    if not rows:
        return empty(ZONE_COLUMNS)
    z = pd.DataFrame(rows)
    z["quiet_before"] = _quiet_before(z)
    z = z[z["tier"].isin(tiers)]
    z = z.sort_values(["known_at", "rank", "lo"], kind="stable", ignore_index=True)
    return z[list(ZONE_COLUMNS)].astype({k: v for k, v in ZONE_COLUMNS.items() if v in ("int64", "float64")})


def _quiet_before(z: pd.DataFrame) -> np.ndarray:
    """For each zone: the EARLIER quiet (tier 1) zones known within settings "zone_quiet_stretch_days" before it, built
    from candles that end before this zone's candles start, whose band overlaps this one (#79: two separate quiet
    stretches at one price). Only zones known by then are counted."""
    days = pd.Timedelta(days=int(settings.get("zone_quiet_stretch_days")))
    t = pd.DatetimeIndex(z["known_at"]).as_unit("ns").asi8
    order = np.argsort(t, kind="stable")
    ts = t[order]
    lo, hi = z["lo"].to_numpy(float)[order], z["hi"].to_numpy(float)[order]
    quiet = (z["tier"].to_numpy() == 1)[order]
    last = pd.DatetimeIndex(z["src_last"]).as_unit("ns").asi8[order]
    first = pd.DatetimeIndex(z["src_first"]).as_unit("ns").asi8[order]
    out = np.zeros(len(z), np.int64)
    span = days.value
    for q in range(len(z)):
        a = np.searchsorted(ts, ts[q] - span, "left")
        b = np.searchsorted(ts, ts[q], "left")          # known strictly before this one
        if a >= b:
            continue
        m = quiet[a:b] & (last[a:b] < first[q]) & (hi[a:b] >= lo[q]) & (lo[a:b] <= hi[q])
        out[order[q]] = int(m.sum())
    return out


# ================================================================================================ a zone's life

def lives(m5: pd.DataFrame, zones: pd.DataFrame, ruler=None) -> pd.DataFrame:
    """One row per zone with its life on the 5-minute chart (see the module text): born, death (the candle it is no
    longer alive on), dead_by (expired / flip / used up), flip_bar, touches [(candle, role, broke)]. READS AHEAD."""
    check_bars(m5, "5m")
    z = zones.reset_index(drop=True).copy()
    n = len(m5)
    h, lo_, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    r = (ADR.adr_bars(m5, "5m")["ruler"].to_numpy(float) if ruler is None else np.asarray(ruler, dtype=float))
    born = m5.index.searchsorted(pd.DatetimeIndex(z["known_at"]), "left") if len(z) else np.array([], np.int64)
    k_conf = int(settings.get("zone_flip_confirm_bars"))
    max_t = int(rules.get("what_a_strong_zone_is")["weak_after_touches"])
    if rules.get("a_zone_only_flips_once_it_is_touched") != "touched":
        raise RuntimeError("rules.py a_zone_only_flips_once_it_is_touched must be 'touched'")
    waits = bool(settings.get("zone_spent_role_waits_for_the_break"))
    death = np.empty(len(z), np.int64)
    dead_by, flip_bar, touches = [], np.full(len(z), -1, np.int64), []
    for q, (zlo, zhi, lb) in enumerate(zip(z["lo"].to_numpy(float), z["hi"].to_numpy(float),
                                          z["life_bars"].to_numpy(np.int64))):
        b = int(born[q])
        e = min(n, b + int(lb))
        why = "expired"
        if b >= e:
            death[q] = min(n, max(b, e))
            dead_by.append(why)
            touches.append([])
            continue
        inside = (h[b:e] >= zlo) & (lo_[b:e] <= zhi)
        # the flip (strategy.md "Break -> S/R flip" read as price ACCEPTING the other side): a close a full day's range
        # beyond the band still beyond it k_conf candles later, counted from the first touch; it ends the zone at the
        # CONFIRMATION candle (the break itself cannot know price will stay there)
        if inside.any() and e - b > k_conf:
            seg, a = c[b:e], r[b:e]
            above, below = seg > zhi + a, seg < zlo - a
            conf = (above[:-k_conf] & above[k_conf:]) | (below[:-k_conf] & below[k_conf:])
            conf[:int(np.argmax(inside))] = False
            if conf.any():
                j = int(np.argmax(conf))
                e, why = b + j + k_conf, "flip"
                flip_bar[q] = b + j
                inside = inside[:e - b]
        starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]]) + b
        got, spent = [], set()
        for k in starts:
            k = int(k)
            before = c[k - 1] if k > 0 else c[k]
            role = "res" if before < zlo else "sup" if before > zhi else None
            j = k
            while j + 1 < e and h[j + 1] >= zlo and lo_[j + 1] <= zhi:
                j += 1                                         # the whole visit
            after = c[min(j + 1, n - 1)]
            broke = (role == "res" and after > zhi) or (role == "sup" and after < zlo)
            got.append((k, role or "?", bool(broke)))
            if role is None:
                continue
            if len(got) >= max_t:                              # his "two to three times": the zone is over
                e, why = k, "used up"
                break
            if role in spent:
                if not waits:
                    e, why = k, "used up"                      # the old engine: over at once
                    break
                if not broke:                                  # #78: it held again - it is over when the visit ends
                    e, why = min(j + 1, n), "used up"
                    break
                continue                                       # #78: it broke - "then it will actually [be] support"
            spent.add(role)
        death[q] = e
        dead_by.append(why)
        touches.append(got)
    z["born"] = born.astype(np.int64) if len(z) else np.array([], np.int64)
    z["death"] = death
    z["dead_by"] = dead_by
    z["flip_bar"] = flip_bar
    z["touches"] = touches
    return z


def alive(life: pd.DataFrame, i: int, min_rank: int = 0) -> pd.DataFrame:
    """The zones alive THROUGH 5-minute candle i: known before it opened, not ended on it or before. Main zone first
    (#81), then the higher rank ("1h stronger"), then the order they were born."""
    m = (life["born"].to_numpy() <= i) & (i < life["death"].to_numpy()) & (life["rank"].to_numpy() >= min_rank)
    return life[m].sort_values(["tier", "rank", "known_at"], ascending=[True, False, True], kind="stable")


def roles_used(row, i: int) -> set[str]:
    """The roles a zone has used by touches that started before candle i ("res" / "sup")."""
    return {role for k, role, _ in row.touches if k < i and role in ("res", "sup")}


def touch_events(life: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """One row per touch of a live zone, known at the close of the touch's first candle (its role reads the close
    before it)."""
    rows = []
    for r in life.itertuples():
        for n, (k, role, _) in enumerate(r.touches, 1):
            if k < r.death or (r.dead_by == "used up" and k == r.death):
                rows.append({"known_at": m5.index[k] + FIVE, "zone_id": r.zone_id, "bar": m5.index[k], "n": n,
                             "role": role})
    if not rows:
        return empty(TOUCH_COLUMNS)
    return pd.DataFrame(rows).sort_values(["known_at", "zone_id"], kind="stable", ignore_index=True)


# ================================================================================================ the real break

def breaks(m5: pd.DataFrame, life: pd.DataFrame) -> pd.DataFrame:
    """The real break of each zone, each way (see the module text). Known at the breaking candle's close."""
    check_bars(m5, "5m")
    c = m5["close"].to_numpy(float)
    n = len(c)
    frac = float(rules.get("what_a_real_zone_break_is"))
    back = int(settings.get("zone_side_lookback_bars"))
    through = settings.get("zone_break_is_through")
    rows = []
    for r in life.itertuples():
        b, e = int(r.born), min(int(r.death), n)
        height = r.hi - r.lo
        if e <= b or height <= 0:
            continue
        s0 = max(0, b - back)
        seg = c[s0:e]
        at = np.arange(s0, e)
        for dr, edge, far in ((1, r.hi, r.lo), (-1, r.lo, r.hi)):
            deep = (seg - r.hi) if dr == 1 else (r.lo - seg)
            real = deep >= frac * height
            cand = np.flatnonzero(real[b - s0:]) + b                           # candles that close through, deep
            if not len(cand):
                continue
            # where price came from: the last close before the break that is either on the OTHER side of the band
            # (it came through) or already a real break this way (it did not come through now). Closes inside the band
            # and a tiny bit past it this way are skipped - #98: "it doesn't make that zone broken, it just made that
            # we draw the zone bit tight".
            other = (seg < r.lo) if dr == 1 else (seg > r.hi)
            code = np.where(other, -1, np.where(real, 1, 0))
            last_pos = np.maximum.accumulate(np.where(code != 0, at, -1))
            prev = cand - 1 - s0
            ok = prev >= 0
            cpos = np.where(ok, last_pos[np.clip(prev, 0, None)], -1)
            if through:
                came = np.where(cpos >= 0, code[np.clip(cpos - s0, 0, None)], 0)
                ok &= (came == -1) & (cand - cpos <= back)
            if not ok.any():
                continue
            i = int(cand[np.argmax(ok)])
            cp = int(cpos[np.argmax(ok)])
            rows.append({"known_at": m5.index[i] + FIVE, "zone_id": r.zone_id,
                         "direction": "buy" if dr == 1 else "sell", "bar": m5.index[i], "edge": float(edge),
                         "hedge": float(far - PAD if dr == 1 else far + PAD), "lo": float(r.lo), "hi": float(r.hi),
                         "height": float(height), "depth": float(deep[i - s0] / height),
                         "came_from": m5.index[cp] if cp >= 0 else pd.NaT, "tier": int(r.tier), "rank": int(r.rank),
                         "window": r.window, "ruler": float(r.ruler)})
    if not rows:
        return empty(BREAK_COLUMNS)
    return pd.DataFrame(rows).sort_values(["known_at", "zone_id"], kind="stable", ignore_index=True)


def after_inner_take(m5: pd.DataFrame, i: int, direction: str, level: float, ruler: float,
                     end: int | None = None) -> dict | str:
    """#99 after a zone break known at candle i: "if there is a session low INSIDE that zone ... I would wait for that
    to take out first, then I would wait for [the] five[-minute] gap". The first candle after i that takes `level`
    (a buy's low reaches it), then the first 5-minute gap the trade's way within settings "zone_break_inner_gap_bars"
    candles of the take (a gap of at least settings "unfilled_gap_min_adr" of a day's range - #62: a tiny gap is a gap).
    Entry = the gap's near edge (#65); hedge = just past the farthest price from the take to the gap (the grab's
    extreme, his hedge point #9 / #33). end: the last candle it may look at (None = the end of the bars).
    Returns the event {known_at, bar, take_bar, entry, hedge, gap_lo, gap_hi} or why there is none."""
    h, lo, = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
    n = len(h) if end is None else min(len(h), end + 1)
    buy = direction == "buy"
    took = next((k for k in range(i + 1, n) if (lo[k] <= level if buy else h[k] >= level)), None)
    if took is None:
        return "the liquidity inside the zone was not taken"
    gap_bars = int(settings.get("zone_break_inner_gap_bars"))
    min_gap = float(settings.get("unfilled_gap_min_adr")) * ruler
    ext = lo[took] if buy else h[took]
    for k in range(max(took, 2), min(n, took + gap_bars + 1)):
        ext = min(ext, lo[k]) if buy else max(ext, h[k])
        if buy and lo[k] > h[k - 2] and lo[k] - h[k - 2] >= min_gap:
            g_lo, g_hi, entry, hedge = h[k - 2], lo[k], lo[k], ext - PAD
        elif not buy and h[k] < lo[k - 2] and lo[k - 2] - h[k] >= min_gap:
            g_lo, g_hi, entry, hedge = h[k], lo[k - 2], h[k], ext + PAD
        else:
            continue
        if (entry - hedge) * (1 if buy else -1) <= 0:
            return "the hedge would sit on the wrong side of the entry"
        return {"known_at": m5.index[k] + FIVE, "bar": m5.index[k], "take_bar": m5.index[took], "entry": float(entry),
                "hedge": float(hedge), "gap_lo": float(g_lo), "gap_hi": float(g_hi)}
    return "no 5-minute gap after the take (#98: 'if there is no fair value gap I won't enter')"


# ================================================================================================ the zone entry

def entries(m5: pd.DataFrame, life: pd.DataFrame, sweep_table: pd.DataFrame, ruler=None) -> pd.DataFrame:
    """His zone entry, one row per armed order (see the module text). sweep_table: liquidity.sweeps(); only fresh
    sweeps of his obvious kinds (settings "obvious_liquidity_kinds") that may start a setup arm it. Known at the close
    of the candle that took the level."""
    check_bars(m5, "5m")
    h, lo, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(m5)
    r = (ADR.adr_bars(m5, "5m")["ruler"].to_numpy(float) if ruler is None else np.asarray(ruler, dtype=float))
    kinds = set(settings.get("obvious_liquidity_kinds"))
    max_gap = float(settings.get("zone_entry_liquidity_adr"))
    min_r = float(settings.get("min_hedge_distance_adr"))
    sw = sweep_table[sweep_table["kind"].isin(kinds) & sweep_table["can_start"] & sweep_table["fresh"]]
    if not len(sw) or not len(life):
        return empty(ENTRY_COLUMNS)
    from machine.blocks.liquidity import RANK
    sw = sw.assign(_j=m5.index.get_indexer(pd.DatetimeIndex(sw["bar"])), _rank=sw["kind"].map(RANK).fillna(0))
    sw = sw.sort_values(["_j", "_rank"], ascending=[True, False], kind="stable")
    born, death = life["born"].to_numpy(), life["death"].to_numpy()
    zlo, zhi = life["lo"].to_numpy(float), life["hi"].to_numpy(float)
    tier, rank = life["tier"].to_numpy(), life["rank"].to_numpy()
    rows = []
    for j, grp in sw.groupby("_j", sort=True):
        j = int(j)
        if j < 0 or j >= n or not math.isfinite(r[j]) or r[j] <= 0:
            continue
        live = np.flatnonzero((born <= j) & (j < death))
        done = set()
        for s in grp.itertuples():
            direction = "buy" if s.side == "low" else "sell"   # a low taken -> buy at the support zone below it
            if direction in done or not len(live):
                continue
            px = float(s.price)
            gap = ((px - zhi[live]) if direction == "buy" else (zlo[live] - px)) / r[j]
            near = live[(gap >= 0) & (gap <= max_gap)]
            if not len(near):
                continue
            g = ((px - zhi[near]) if direction == "buy" else (zlo[near] - px)) / r[j]
            q = int(near[np.lexsort((g, tier[near]))[0]])      # main zone first (#81), then nearest
            z = life.iloc[q]
            role = "sup" if direction == "buy" else "res"
            if role in roles_used(z, j):
                continue                                        # #78: that side of the zone is used
            if direction == "buy" and not (lo[j] > zhi[q] and c[j] > zhi[q]):
                continue                                        # the take reached the zone in one go: no order waited
            if direction == "sell" and not (h[j] < zlo[q] and c[j] < zlo[q]):
                continue
            entry = float(zhi[q] if direction == "buy" else zlo[q])
            hedge = float(zlo[q] - PAD if direction == "buy" else zhi[q] + PAD)
            if abs(entry - hedge) < min_r * r[j]:
                continue
            rows.append({"known_at": m5.index[j] + FIVE, "zone_id": z.zone_id, "direction": direction,
                         "bar": m5.index[j], "level_id": s.level_id, "level_kind": s.kind, "level_tf": s.tf,
                         "level_price": px, "entry": entry, "hedge": hedge, "lo": float(zlo[q]), "hi": float(zhi[q]),
                         "tier": int(tier[q]), "rank": int(rank[q]), "window": z.window,
                         "liq_gap_adr": float(((px - zhi[q]) if direction == "buy" else (zlo[q] - px)) / r[j]),
                         "ruler": float(r[j])})
            done.add(direction)
    if not rows:
        return empty(ENTRY_COLUMNS)
    return pd.DataFrame(rows)[list(ENTRY_COLUMNS)]


# ================================================================================ the zone before the target, and edges

def in_path(life: pd.DataFrame, i: int, direction: str, entry: float, target: float,
            min_rank: int | None = None) -> dict | None:
    """The zone standing between the entry and the target at candle i (double profit, strategy.md sections 3, 10): the
    nearest live zone of at least settings "double_profit_zone_rank" whose NEAR edge lies strictly between the entry
    and the target. He books at that edge (book_at) and gets back in at the price he booked (the retest reading) or
    past its far edge (the break reading) - settings "double_profit_reentry" (off / retest / break / setup) is the
    manager's (L19c). A zone the entry stands inside is not in front of him: None."""
    min_rank = int(settings.get("double_profit_zone_rank")) if min_rank is None else int(min_rank)
    z = alive(life, i, min_rank)
    if not len(z):
        return None
    up = direction == "buy"
    near = z["lo"].to_numpy(float) if up else z["hi"].to_numpy(float)
    far = z["hi"].to_numpy(float) if up else z["lo"].to_numpy(float)
    inside = (z["lo"].to_numpy(float) <= entry) & (entry <= z["hi"].to_numpy(float))
    d = (near - entry) * (1 if up else -1)
    ahead = ~inside & (d > 0) & ((target - near) * (1 if up else -1) > 0)
    if not ahead.any():
        return None
    k = int(np.argmin(np.where(ahead, d, np.inf)))
    row = z.iloc[k]
    return {"zone_id": row.zone_id, "book_at": float(near[k]), "far": float(far[k]), "tier": int(row.tier),
            "rank": int(row["rank"]), "lo": float(row.lo), "hi": float(row.hi)}


def reentry_price(zone: dict, reading: str) -> float | None:
    """Where he gets back in after booking at `zone` (in_path): retest = the price he booked at; break = past the far
    edge; setup = only on a full setup of his (the manager finds it, L19c) -> None; off -> None."""
    if reading == "retest":
        return float(zone["book_at"])
    if reading == "break":
        return float(zone["far"])
    return None


def nearest_edge(life: pd.DataFrame, i: int, side: str, price: float, max_away: float | None = None) -> dict | None:
    """The nearest edge of a live zone on `side` of price ("high" = above: a zone's LOW edge is where price meets it;
    "low" = below: its HIGH edge) - "Target = next liquidity / zone" (B1-49). A zone price stands in is not ahead."""
    z = alive(life, i)
    if not len(z):
        return None
    up = side == "high"
    edge = z["lo"].to_numpy(float) if up else z["hi"].to_numpy(float)
    d = (edge - price) if up else (price - edge)
    ok = d > 0
    if max_away is not None:
        ok &= d <= max_away
    if not ok.any():
        return None
    k = int(np.argmin(np.where(ok, d, np.inf)))
    row = z.iloc[k]
    return {"zone_id": row.zone_id, "price": float(edge[k]), "tier": int(row.tier), "rank": int(row["rank"])}
