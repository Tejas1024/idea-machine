"""How gold reacted to each news release (L12b.2).     python -m machine.news.reaction
    -> data/news/reactions.parquet            one row per release MINUTE (several numbers can come out in one minute)
    -> data/news/reaction_controls.parquet    the same measures at the same New York clock minute on days with NO
                                              release near it: the fair comparison for every news belief

Everything is measured on the 1-minute bars, 60 minutes before the release to 240 minutes after (server clock, bid).
p0 = the last price BEFORE the release: the close of the last 1-minute bar that opened before the release minute.

THE REACTION (what happened after - never usable as a signal at the release):
  move_1 / _5 / _15 / _60 / _240   price that many minutes after the release minus p0, in dollars (move_5 = the close of
                                   the 5th minute). `_adr` = the same as a share of a day's range (his 10-day ADR).
  move_close                       the broker day's last price minus p0 ("the close": 24:00 server = 17:00 New York).
  up_60 / dn_60 (+ their minute)   the biggest move up and down from p0 in the first hour; range_60 = high - low.
  first side, hour range (hr_*)    the high and low of the hour before the release are the range. hr_side = the side
                                   price broke FIRST in the hour after ("high" / "low" / "both" in one minute / none);
                                   hr_race = from that edge, did price come BACK to the far side of the range first
                                   ("back"), go the same distance further ON first ("on"), do neither in 4 hours
                                   ("open"), or both inside one 1-minute bar ("unclear")? The old project's reading of
                                   his news trap (strategy-understanding section 48f): equal distances, so a coin = 50%.
  first side, his levels (lv_*)    the same with HIS liquidity: the nearest level above and below p0 that was alive at
                                   the release (session highs / lows, equal highs / lows, trendlines, day and week
                                   swings - machine/blocks/liquidity.py, the rows that may start a setup). lv_side = the
                                   one a wick took first in the hour after; lv_race = from that level, back or on by
                                   the same distance as above (the hour range before the release).
  calm_pre / calm_normal           minutes from the release until gold was calm again (his G11, "news done"): the
                                   first three 5-minute bars in a row whose average range is no bigger than
                                   (pre) the average 5-minute range in the hour before the release - "back to pre-news
                                   size"; (normal) the usual 5-minute range at that New York clock time on the last 20
                                   days with no release near it (news.md NQ4). Empty = not calm within 4 hours.

THE SITUATION KNOWN BEFORE (CONTEXT - each passes the cut-off-the-future test, tests/test_reaction.py):
  session          his session window the release falls in (India time), or "outside"
  bias_h/_d/_w     his bias on the hour / day / week chart at the release (+1 up, -1 down, 0 neutral - blocks/bias.py)
  up_level_* / dn_level_*   the nearest of his levels above and below: distance in dollars and in a day's range, kind
  adr, adr_used, day_pos    a day's range (his 10-day ADR); how much of it today has already used (high - low so
                   far); where price sits from today's open (both as a share of a day's range)
  before_120       the move in the 2 hours before (dollars; _adr as a share)
  pre_hi / pre_lo / pre_range / pre_5m   the hour before: high, low, their distance, the average 5-minute range
  normal_5m        the usual 5-minute range at this clock time on the last 20 quiet days (past days only)

The numbers WINDOW_*, OPEN_GAP_MIN, QUIET_MIN, CALM_BARS, NORMAL_* are Claude's, fixed before any reaction was
looked at (plan/work/L12b2-notes.md); none is Tejas's rule.

The check years (2021-2023) are measured only once every news belief's check years are open
(machine/news/beliefs.py); load() gives the tune years unless asked otherwise.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd

from machine import config, guards, settings
from machine.blocks import sessions as SES
from machine.data import bars as BARS
from machine.data import clock

NY = "America/New_York"
MIN = 60 * 10 ** 9                  # one minute in nanoseconds
DAY = 24 * 60 * MIN
WINDOW_BEFORE = 60                  # minutes before the release that are read (the prompt)
WINDOW_AFTER = 240                  # minutes after
FIRST_HOUR = 60
HORIZONS = (1, 5, 15, 60, 240)      # "first move at +1 / +5 / +15 min ... price at +1 h, +4 h"
OPEN_GAP_MIN = 5                    # no 1-minute bar within 5 minutes on a side of the release = the market is closed
QUIET_MIN = 60                      # a comparison moment has no release within 60 minutes on either side
CALM_BARS = 3                       # calm = three 5-minute bars in a row (15 minutes) at or under the size
NORMAL_DAYS = 20                    # "the usual range at this clock time": the last 20 quiet days ...
NORMAL_MIN_DAYS = 10                # ... at least 10 of them, looked for in the last ...
NORMAL_LOOKBACK = 60                # ... 60 calendar days
PRE_MIN_BARS = 30                   # the hour before needs 30 of its 60 one-minute bars
COLOUR_RANK = {"red": 3, "orange": 2, "yellow": 1}
HIS_COLOURS = ("red", "orange")     # Tejas, T7 (LEARNED #141): his news = Forex Factory red and orange
BIG = 1.0                           # a big surprise: |z| >= 1 (machine/news/surprise.py BIG)

CONTEXT = ["p0", "adr", "session", "bias_h", "bias_d", "bias_w", "up_level_dist", "up_level_kind", "dn_level_dist",
           "dn_level_kind", "up_level_adr", "dn_level_adr", "adr_used", "day_pos", "before_120", "before_120_adr",
           "pre_hi", "pre_lo", "pre_range", "pre_5m", "normal_5m"]
REACTION = ["has_bars", *[f"move_{k}" for k in HORIZONS], "move_close", *[f"move_{k}_adr" for k in HORIZONS],
            "move_close_adr", "open_240", "up_60", "dn_60", "up_60_min", "dn_60_min", "range_60", "up_60_adr",
            "dn_60_adr", "range_60_adr", "hr_side", "hr_min", "hr_race", "lv_side", "lv_min", "lv_kind", "lv_dist",
            "lv_race", "calm_pre", "calm_normal"]
_TEXT = {"session", "up_level_kind", "dn_level_kind", "hr_side", "hr_race", "lv_side", "lv_kind", "lv_race"}


# ================================================================================================= release minutes

def moments(table: pd.DataFrame) -> pd.DataFrame:
    """One row per release MINUTE from the surprise table (machine.news.surprise.load, with `scheduled`): index =
    server time. Several numbers often come out in one minute (the jobs report: payrolls + unemployment + earnings).

      colour / his_news   the strongest Forex Factory colour in that minute (red > orange > yellow > none); his news =
                          red or orange (T7)
      reports             each report's name in that minute, strongest first, joined by " | "
      his_reports         the same, only the reports whose OWN row is red or orange ("" when none)
      report, event, kind, z, group, method, z_known_from   the MAIN number: the strongest colour, then a report's
                          own number (headline), then MT5 "high", then the biggest surprise
      gold_way            +1 = the news was good for gold, -1 = bad for gold, 0 = small or not scored. From the
                          surprises of the strongest colour's headline numbers with a forecast: a BIG surprise
                          (|z| >= 1) that is bad for the dollar counts as good for gold (the usual reading; whether
                          gold really moves that way is what beliefs.py measures). Two big ones that disagree ->
                          0 and mixed = True.
      gold_way_fed        the same for the Fed's decisions (the 2-year yield, a DAILY number known only the next day)
    """
    t = table[table["server"].notna()].copy()
    t["_c"] = t["ff_colour"].map(COLOUR_RANK).fillna(0).astype(int)
    t["_h"] = t["headline"].astype(int)
    t["_i"] = (t["importance"] == "high").astype(int)
    t["_z"] = t["z"].abs().fillna(-1.0)
    t = t.sort_values(["server", "_c", "_h", "_i", "_z"], ascending=[True, False, False, False, False], kind="stable")
    main = t.drop_duplicates("server", keep="first").set_index("server")
    top = t[t["_c"] == t.groupby("server")["_c"].transform("max")]
    big = top[(top["_h"] == 1) & (top["z"].abs() >= BIG)]
    big = big.assign(way=-np.sign(big["z"]))

    def consensus(method: str) -> tuple[pd.Series, pd.Series]:
        w = big[big["method"] == method].groupby("server")["way"].agg(["min", "max"])
        same = w["min"] == w["max"]
        return w["min"].where(same, 0.0), ~same

    g_fc, mixed = consensus("forecast")
    g_fed, _ = consensus("2y yield")
    g = t.groupby("server", sort=True)
    out = pd.DataFrame({
        "utc": main["utc"], "ist": main["ist"],
        "colour": main["_c"].map({v: k for k, v in COLOUR_RANK.items()}).fillna("none"),
        "importance": g["_i"].max().map({1: "high", 0: "moderate"}),
        "numbers": g.size(),
        "reports": g["report"].agg(lambda x: " | ".join(dict.fromkeys(x))),
        "his_reports": t[t["_c"] >= COLOUR_RANK["orange"]].groupby("server")["report"].agg(
            lambda x: " | ".join(dict.fromkeys(x))),
        "report": main["report"], "event": main["event"], "kind": main["kind"], "method": main["method"],
        "z": main["z"], "group": main["group"], "z_known_from": main["known_from"],
        "scheduled": g["scheduled"].all() if "scheduled" in t else True,
    })
    out["his_news"] = out["colour"].isin(HIS_COLOURS)
    out["his_reports"] = out["his_reports"].fillna("")
    out["gold_way"] = g_fc.reindex(out.index).fillna(0.0).astype(int)
    out["mixed"] = mixed.reindex(out.index).fillna(False).astype(bool)
    out["gold_way_fed"] = g_fed.reindex(out.index).fillna(0.0).astype(int)
    return _with_clock(out)


def _with_clock(out: pd.DataFrame) -> pd.DataFrame:
    """ny_clock ("08:30"), weekday (0 = Monday, New York's) and year (the server clock's) of each moment."""
    idx = pd.DatetimeIndex(out.index)
    ny = clock.server_to_utc(idx).tz_convert(NY)
    out["ny_clock"] = ny.strftime("%H:%M")
    out["weekday"] = ny.dayofweek
    out["year"] = idx.year
    out.index.name = "server"
    return out


def busy_times(start: Any, end: Any) -> np.ndarray:
    """Every moment with a release, sorted (server clock, nanoseconds): all rows of the L12 calendar (MT5 high +
    moderate) and every Forex Factory red or orange US row (463 of those are not in the calendar - L12b1-notes.md).
    A comparison moment must have none of these within QUIET_MIN minutes."""
    from machine.news import forexfactory
    guards.check_dates([x for x in (start, end) if x is not None] or ["2016"])
    ev = pd.read_parquet(config.NEWS_EVENTS, columns=["server"])
    cal = pd.DatetimeIndex(ev["server"].dropna())
    ff = forexfactory.load()
    ff = ff[ff["colour"].isin(HIS_COLOURS) & ff["utc"].notna()]
    ffs = clock.utc_to_server(pd.DatetimeIndex(pd.to_datetime(ff["utc"], utc=True)))
    both = cal.append(ffs).sort_values().unique()
    lo = pd.Timestamp(str(start)) - pd.Timedelta(days=2)
    hi = pd.Timestamp(f"{pd.Timestamp(str(end)).year}-12-31") + pd.Timedelta(days=3)
    both = both[(both >= lo) & (both <= min(hi, pd.Timestamp(config.SEALED_FROM) - pd.Timedelta(minutes=1)))]
    return both.as_unit("ns").asi8


def control_times(event_moments: pd.DataFrame, busy: np.ndarray, years) -> pd.DataFrame:
    """The comparison moments: every Monday-Friday of each year at every New York clock minute where a coloured
    release (red, orange or yellow) came out IN THAT YEAR - kept only when NO release lies within QUIET_MIN minutes
    on either side. Per year, so a year's comparison moments never depend on which other years are measured.
    Index = server time; ny_clock, weekday, year like moments()."""
    coloured = event_moments[event_moments["colour"] != "none"]
    parts = []
    for y in sorted(int(v) for v in years):
        days = pd.bdate_range(f"{y}-01-01", f"{y}-12-31")
        for c in sorted(set(coloured.loc[coloured["year"] == y, "ny_clock"])):
            h, m = (int(x) for x in c.split(":"))
            ny = (days + pd.Timedelta(hours=h, minutes=m)).tz_localize(NY, nonexistent="NaT", ambiguous="NaT")
            server = ny[~ny.isna()].tz_convert(clock.SERVER_ZONE).tz_localize(None)
            parts.append(server[server.year == y])
    if not parts:
        return _with_clock(pd.DataFrame(index=pd.DatetimeIndex([])))
    t = parts[0].append(parts[1:]).unique().sort_values()
    ns = t.as_unit("ns").asi8
    k = np.searchsorted(busy, ns - QUIET_MIN * MIN, "left")
    quiet = (k >= len(busy)) | (busy[np.minimum(k, len(busy) - 1)] > ns + QUIET_MIN * MIN)
    return _with_clock(pd.DataFrame(index=t[quiet]))


# ======================================================================================================= measuring

class Measurer:
    """Measures any moments on one set of 1-minute bars. market: a machine.ideas.compile.Market over the same span
    (5m, 1h, 1d, 1w) for his bias, his levels and his day's range; None = those columns stay empty.
    busy: busy_times() - only for normal_5m (which past days were quiet)."""

    def __init__(self, m1: pd.DataFrame, market=None, busy: np.ndarray | None = None):
        guards.check_dates(m1)
        self.T = pd.DatetimeIndex(m1.index).as_unit("ns").asi8
        self.O, self.H, self.L, self.C = (m1[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
        self.market = market
        self.fp = settings.fingerprint()
        self.busy = np.array([], dtype=np.int64) if busy is None else np.asarray(busy, dtype=np.int64)
        self._normal = None
        self._mk = None

    # ---- the usual 5-minute range at each New York clock time (past days only) -----------------------------------
    def _normal_table(self):
        """Per New York calendar day and 5-minute clock slot: the average 5-minute range over the next 60 minutes,
        and whether the slot is quiet (no release within QUIET_MIN minutes either side - the comparison moments'
        rule). Read only for days BEFORE the moment's own day."""
        if self._normal is None:
            idx = pd.DatetimeIndex(self.T)
            ny = clock.server_to_utc(idx).tz_convert(NY).tz_localize(None)
            d0 = ny[0].normalize()
            n_days = (ny[-1].normalize() - d0).days + 1
            flat = ((ny.normalize() - d0).days * 288 + (ny.hour * 60 + ny.minute) // 5).to_numpy()
            hi = np.full(n_days * 288, -np.inf)
            lo = np.full(n_days * 288, np.inf)
            np.maximum.at(hi, flat, self.H)
            np.minimum.at(lo, flat, self.L)
            has = hi > -np.inf
            rng = np.where(has, hi - lo, 0.0)
            cs = np.concatenate([[0.0], np.cumsum(rng)])
            cn = np.concatenate([[0], np.cumsum(has)])
            k = np.arange(n_days * 288)
            e = np.minimum(k + 12, n_days * 288)
            cnt = cn[e] - cn[k]
            fwd = np.where(cnt >= 6, (cs[e] - cs[k]) / np.maximum(cnt, 1), np.nan)
            quiet = np.ones(n_days * 288, dtype=bool)
            if len(self.busy):
                b = clock.server_to_utc(pd.DatetimeIndex(self.busy)).tz_convert(NY).tz_localize(None)
                f = ((b.normalize() - d0).days * 288 + (b.hour * 60 + b.minute) // 5).to_numpy()
                f = f[(f >= 0) & (f < n_days * 288)]
                step = QUIET_MIN // 5
                for s in range(-step, step + 1):             # a release within QUIET_MIN minutes either side
                    g = f + s
                    quiet[g[(g >= 0) & (g < n_days * 288)]] = False
            self._normal = (d0, n_days, fwd.reshape(n_days, 288), quiet.reshape(n_days, 288))
        return self._normal

    def _normal_5m(self, ny_naive: pd.Timestamp) -> float:
        d0, n_days, fwd, quiet = self._normal_table()
        d = (ny_naive.normalize() - d0).days
        if d <= 0 or d > n_days:
            return np.nan
        slot = (ny_naive.hour * 60 + ny_naive.minute) // 5
        a = max(0, d - NORMAL_LOOKBACK)
        v = fwd[a:d, slot]
        v = v[quiet[a:d, slot] & np.isfinite(v)]
        return float(np.median(v[-NORMAL_DAYS:])) if len(v) >= NORMAL_MIN_DAYS else np.nan

    # ---- his levels, bias and day's range ------------------------------------------------------------------------
    def _market_arrays(self):
        if self._mk is None and self.market is not None:
            m = self.market
            life = m.liquidity(self.fp).life
            life = life[life["can_start"]].sort_values("start", kind="stable")
            self._mk = {
                "opens5": m.opens("5m"), "ruler": m.ruler("5m", self.fp),
                "start": life["start"].to_numpy(np.int64), "end": life["end"].to_numpy(np.int64),
                "price": life["price"].to_numpy(float), "slope": life["slope"].to_numpy(float),
                "x0": life["x0pos"].to_numpy(np.int64), "high": (life["side"] == "high").to_numpy(),
                "kind": life["kind"].to_numpy(object),
            }
        return self._mk

    def _levels(self, t0: int, i0: int, p0: float):
        """The nearest of his levels above and below p0 that is alive at the release: known before the 5-minute
        candle holding the release opened, and not taken before the release minute. A trendline's price is read on
        that candle. Returns (up_price, up_kind, dn_price, dn_kind, a day's range)."""
        mk = self._market_arrays()
        if mk is None:
            return np.nan, None, np.nan, None, np.nan
        i5 = int(np.searchsorted(mk["opens5"], t0, "right")) - 1
        if i5 < 0:
            return np.nan, None, np.nan, None, np.nan
        n = int(np.searchsorted(mk["start"], i5, "right"))
        ok = mk["end"][:n] >= i5
        price = mk["price"][:n] + mk["slope"][:n] * (i5 - mk["x0"][:n])
        high = mk["high"][:n]
        j5 = int(np.searchsorted(self.T, mk["opens5"][i5], "left"))
        if j5 < i0:                                    # the part of that candle before the release minute
            ok &= np.where(high, price >= self.H[j5:i0].max(), price <= self.L[j5:i0].min())
        up = ok & high & (price > p0)
        dn = ok & ~high & (price < p0)
        kinds = mk["kind"][:n]
        ui = np.flatnonzero(up)
        di = np.flatnonzero(dn)
        u = ui[np.argmin(price[ui])] if len(ui) else -1
        d = di[np.argmax(price[di])] if len(di) else -1
        return (price[u] if u >= 0 else np.nan, kinds[u] if u >= 0 else None,
                price[d] if d >= 0 else np.nan, kinds[d] if d >= 0 else None, float(mk["ruler"][i5]))

    # ---- one race ------------------------------------------------------------------------------------------------
    def _race(self, j: int, edge: float, side_high: bool, dist: float, end: int) -> str | None:
        """Price went through `edge` on 1-minute bar j. From the edge: `dist` back through it first ("back"), or
        `dist` further on first ("on")? Inside bar j itself only what is sure counts: the far price in the break's
        direction came after the break; a close at the back price means it came back."""
        if not dist > 0:
            return None
        s = 1.0 if side_high else -1.0
        on_p, back_p = edge + s * dist, edge - s * dist
        H, L, C = self.H, self.L, self.C
        if (H[j] >= on_p) if side_high else (L[j] <= on_p):
            return "on"
        if (C[j] <= back_p) if side_high else (C[j] >= back_p):
            return "back"
        a = j + 1
        if a >= end:
            return "open"
        on_hit = (H[a:end] >= on_p) if side_high else (L[a:end] <= on_p)
        back_hit = (L[a:end] <= back_p) if side_high else (H[a:end] >= back_p)
        io = int(on_hit.argmax()) if on_hit.any() else -1
        ib = int(back_hit.argmax()) if back_hit.any() else -1
        if io < 0 and ib < 0:
            return "open"
        if ib < 0 or 0 <= io < ib:
            return "on"
        if io < 0 or ib < io:
            return "back"
        return "unclear"

    def _first_side(self, a: int, b: int, up: float, dn: float):
        """In bars a..b-1: which of the two prices is passed first? (side, bar) - side None when neither."""
        hu = self.H[a:b] > up if np.isfinite(up) else np.zeros(b - a, bool)
        hd = self.L[a:b] < dn if np.isfinite(dn) else np.zeros(b - a, bool)
        iu = int(hu.argmax()) if hu.any() else -1
        idn = int(hd.argmax()) if hd.any() else -1
        if iu < 0 and idn < 0:
            return None, -1
        if idn < 0 or 0 <= iu < idn:
            return "high", a + iu
        if iu < 0 or idn < iu:
            return "low", a + idn
        return "both", a + iu

    # ---- all moments ---------------------------------------------------------------------------------------------
    def measure(self, times, context_only: bool = False) -> pd.DataFrame:
        """CONTEXT (+ REACTION unless context_only) for each moment (server clock). A moment with no 1-minute bar in
        the OPEN_GAP_MIN minutes before it has no context either (the market is closed)."""
        idx = pd.DatetimeIndex(times)
        ns = idx.as_unit("ns").asi8
        utc = clock.server_to_utc(idx)
        ny = utc.tz_convert(NY).tz_localize(None)
        ist_min = (utc.tz_convert(clock.IST_ZONE).hour * 60 + utc.tz_convert(clock.IST_ZONE).minute).to_numpy()
        wins = [(name, a.hour * 60 + a.minute, b.hour * 60 + b.minute) for name, a, b in SES.windows()]
        cols = CONTEXT if context_only else CONTEXT + REACTION
        out = {c: (np.full(len(idx), None, dtype=object) if c in _TEXT else np.full(len(idx), np.nan)) for c in cols}
        T, H, L, C, O = self.T, self.H, self.L, self.C, self.O
        n = len(T)
        for q, t0 in enumerate(ns):
            i0 = int(np.searchsorted(T, t0, "left"))
            if i0 == 0 or t0 - T[i0 - 1] > OPEN_GAP_MIN * MIN:
                continue
            p0 = C[i0 - 1]
            out["p0"][q] = p0
            out["session"][q] = next((w for w, a, b in wins if a <= ist_min[q] < b), "outside")
            # ---- the hour before, the 2 hours before, today so far
            a0 = int(np.searchsorted(T, t0 - WINDOW_BEFORE * MIN, "left"))
            pre_ok = i0 - a0 >= PRE_MIN_BARS
            if pre_ok:
                out["pre_hi"][q], out["pre_lo"][q] = H[a0:i0].max(), L[a0:i0].min()
                out["pre_range"][q] = out["pre_hi"][q] - out["pre_lo"][q]
            j2 = int(np.searchsorted(T, t0 - 120 * MIN, "left")) - 1
            if j2 >= 0 and (t0 - 120 * MIN) - T[j2] <= 30 * MIN:
                out["before_120"][q] = p0 - C[j2]
            d0 = int(np.searchsorted(T, t0 - t0 % DAY, "left"))
            up_p, up_k, dn_p, dn_k, adr = self._levels(int(t0), i0, p0)
            out["adr"][q] = adr
            out["up_level_dist"][q], out["up_level_kind"][q] = up_p - p0, up_k
            out["dn_level_dist"][q], out["dn_level_kind"][q] = p0 - dn_p, dn_k
            if d0 < i0:
                out["adr_used"][q] = (H[d0:i0].max() - L[d0:i0].min()) / adr
                out["day_pos"][q] = (p0 - O[d0]) / adr
            if self.market is not None:
                for tf, c in (("1h", "bias_h"), ("1d", "bias_d"), ("1w", "bias_w")):
                    if self.market.has(tf):
                        out[c][q] = self.market.bias_at(tf, self.fp, idx[q])
            # ---- 5-minute ranges: 12 before the release, 48 after
            a = int(np.searchsorted(T, t0 - 60 * MIN, "left"))
            b = i0 if context_only else int(np.searchsorted(T, t0 + WINDOW_AFTER * MIN, "left"))
            bk = (T[a:b] - t0) // (5 * MIN) + 12
            hi5 = np.full(60, -np.inf)
            lo5 = np.full(60, np.inf)
            np.maximum.at(hi5, bk, H[a:b])
            np.minimum.at(lo5, bk, L[a:b])
            r5 = np.where(hi5 > -np.inf, hi5 - lo5, np.nan)
            pre = r5[:12]
            if np.isfinite(pre).sum() >= 6:
                out["pre_5m"][q] = np.nanmean(pre)
            out["normal_5m"][q] = self._normal_5m(ny[q])
            if context_only:
                continue
            # ---- the reaction
            if i0 >= n or T[i0] - t0 > OPEN_GAP_MIN * MIN:
                out["has_bars"][q] = 0.0
                continue
            out["has_bars"][q] = 1.0
            for k in HORIZONS:
                j = int(np.searchsorted(T, t0 + k * MIN, "left")) - 1
                out[f"move_{k}"][q] = C[max(j, i0 - 1)] - p0
                if k == WINDOW_AFTER:
                    out["open_240"][q] = float((t0 + k * MIN) - T[j] <= (OPEN_GAP_MIN + 1) * MIN)
            jc = int(np.searchsorted(T, t0 - t0 % DAY + DAY, "left")) - 1
            out["move_close"][q] = C[jc] - p0
            b60 = int(np.searchsorted(T, t0 + FIRST_HOUR * MIN, "left"))
            hh, ll = H[i0:b60], L[i0:b60]
            out["up_60"][q], out["dn_60"][q] = max(hh.max() - p0, 0.0), max(p0 - ll.min(), 0.0)
            out["up_60_min"][q] = (T[i0 + int(hh.argmax())] - t0) / MIN
            out["dn_60_min"][q] = (T[i0 + int(ll.argmin())] - t0) / MIN
            out["range_60"][q] = hh.max() - ll.min()
            end = int(np.searchsorted(T, t0 + WINDOW_AFTER * MIN, "left"))
            w = out["pre_range"][q]
            if pre_ok:
                side, j = self._first_side(i0, b60, out["pre_hi"][q], out["pre_lo"][q])
                out["hr_side"][q] = side
                if side in ("high", "low"):
                    out["hr_min"][q] = (T[j] - t0) / MIN
                    edge = out["pre_hi"][q] if side == "high" else out["pre_lo"][q]
                    out["hr_race"][q] = self._race(j, edge, side == "high", w, end)
            side, j = self._first_side(i0, b60, up_p, dn_p)
            out["lv_side"][q] = side
            if side in ("high", "low"):
                out["lv_min"][q] = (T[j] - t0) / MIN
                out["lv_kind"][q] = up_k if side == "high" else dn_k
                out["lv_dist"][q] = (up_p - p0) if side == "high" else (p0 - dn_p)
                if pre_ok:
                    out["lv_race"][q] = self._race(j, up_p if side == "high" else dn_p, side == "high", w, end)
            post = r5[12:]
            run = (post[:-2] + post[1:-1] + post[2:]) / CALM_BARS         # NaN when a bar of the three has no price
            for name, size in (("calm_pre", out["pre_5m"][q]), ("calm_normal", out["normal_5m"][q])):
                if np.isfinite(size):
                    hit = np.flatnonzero(run <= size)
                    if len(hit):
                        out[name][q] = 5.0 * hit[0]
        df = pd.DataFrame(out, index=idx)
        for c in ("up_level", "dn_level"):
            df[f"{c}_adr"] = df[f"{c}_dist"] / df["adr"]
        df["before_120_adr"] = df["before_120"] / df["adr"]
        if not context_only:
            for c in [f"move_{k}" for k in HORIZONS] + ["move_close", "up_60", "dn_60", "range_60"]:
                df[f"{c}_adr"] = df[c] / df["adr"]
            df["has_bars"] = df["has_bars"] == 1.0
            df["open_240"] = df["open_240"] == 1.0
        df.index.name = "server"
        return df[cols]


# ============================================================================================ build, load, report

def event_moments(start: Any, end: Any) -> pd.DataFrame:
    """moments() of the surprise table from start to end, with each row's `scheduled` from the calendar."""
    from machine.news import surprise
    table = surprise.load(start, end)
    ev = pd.read_parquet(config.NEWS_EVENTS, columns=["value_id", "scheduled"])
    table = table.merge(ev, on="value_id", how="left", validate="one_to_one")
    table["scheduled"] = table["scheduled"].fillna(True).astype(bool)
    return moments(table)


def build(years=config.TUNE_YEARS, write: bool = True, market=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(reactions, controls) for the years given - always measured from 2016 on, so his bias and levels have their
    history. Years inside 2021-2023 only when every news belief's check years are open (beliefs.require_open)."""
    years = tuple(sorted(int(y) for y in years))
    if set(years) & set(config.CHECK_YEARS):
        from machine.news import beliefs
        beliefs.require_open()
    start, end = str(min(config.TUNE_YEARS[0], years[0])), str(years[-1])
    guards.check_dates([start, end])
    from machine.ideas.compile import Market
    m1 = BARS.load_bars("1m", start, end)
    market = market or Market.load(start, end)
    busy = busy_times(start, end)
    ev = event_moments(str(years[0]), end)
    ev = ev[ev["year"].isin(years)]
    ct = control_times(ev, busy, years)
    meas = Measurer(m1, market, busy)
    ev = ev.join(meas.measure(ev.index))
    ct = ct.join(meas.measure(ct.index))
    ct = ct[ct["has_bars"]]
    if write:
        for df, path in ((ev, config.NEWS_REACTIONS), (ct, config.NEWS_REACTION_CONTROLS)):
            tmp = path.with_suffix(".tmp")
            df.reset_index().to_parquet(tmp, index=False)
            tmp.replace(path)
    return ev, ct


def load(which: str = "events", *, years=None, check_for: tuple[str, str] | None = None,
         lookup: bool = False) -> pd.DataFrame:
    """The reactions ("events") or the comparison moments ("controls"), index = server time.

    years: default the tune years 2016-2020. The check years 2021-2023 only with
      check_for=(belief idea, version)  the belief whose check years are open for that version (guards.check_opened), or
      lookup=True                       a LOOKUP of past releases to show him (L12b.3, L17a) - never to test or choose.
    """
    path = {"events": config.NEWS_REACTIONS, "controls": config.NEWS_REACTION_CONTROLS}[which]
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing - make it with `python -m machine.news.reaction`")
    df = pd.read_parquet(path).set_index("server")
    guards.check_dates(df.index)
    years = tuple(config.TUNE_YEARS if years is None else years)
    hidden = sorted(set(years) & set(config.CHECK_YEARS))
    if hidden and not lookup:
        if check_for is None:
            raise guards.CheckYearsClosedError(
                f"the reactions of {hidden[0]}-{hidden[-1]} are the check years: pass check_for=(belief, version) "
                "for a belief whose check years are open, or lookup=True for a lookup that tests nothing")
        guards.check_opened(check_for[0], [f"{hidden[0]}-01-01"], version=check_for[1])
    return df[df["year"].isin(years)]


def by_report(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (release minute, report of his colours in it): for tables per event type. `alone` = the only
    report of his colours in that minute (the jobs report and the trade balance sometimes share 08:30)."""
    x = ev[ev["his_reports"] != ""]
    x = x.assign(report=x["his_reports"].str.split(" | ", regex=False)).explode("report")
    x["alone"] = x["his_reports"].str.count(r"\|") == 0
    return x


def _med(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.median()) if len(s) else np.nan


def type_table(ev: pd.DataFrame, ct: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Per report of his colours (at least min_n releases that had prices): how far gold moved in the first hour and
    how long until calm, next to the same clock minute on days with no release (the middle value of each); the same
    for the releases that were ALONE in their minute. calm = calm_normal; `not_calm_4h` = the share not calm within 4
    hours, among the releases whose usual range is known. Sorted by the first-hour range against the comparison."""
    e = by_report(ev[ev["his_news"] & ev["has_bars"].fillna(False)])
    quiet = ct.groupby("ny_clock").agg(c_range=("range_60_adr", "median"), c_calm=("calm_normal", "median"))
    rows = []
    for rep, g in e.groupby("report"):
        if len(g) < min_n:
            continue
        q = quiet.reindex(g["ny_clock"])
        alone = g[g["alone"]]
        known = g[g["normal_5m"].notna()]
        rows.append({"report": rep, "releases": len(g), "colour": g["colour"].mode().iat[0],
                     "clock": g["ny_clock"].mode().iat[0],
                     "move_5_usd": _med(g["move_5"].abs()), "range_60_usd": _med(g["range_60"]),
                     "range_60_adr": _med(g["range_60_adr"]), "quiet_range_60_adr": _med(q["c_range"]),
                     "times_quiet": _med(g["range_60_adr"]) / _med(q["c_range"]),
                     "alone": len(alone),
                     "alone_times_quiet": (_med(alone["range_60_adr"]) / _med(quiet.reindex(alone["ny_clock"])["c_range"])
                                           if len(alone) >= 10 else np.nan),
                     "calm_normal_min": _med(known["calm_normal"].fillna(WINDOW_AFTER)),
                     "calm_late_min": float(known["calm_normal"].fillna(WINDOW_AFTER).quantile(0.75)) if len(known) else np.nan,
                     "quiet_calm_min": _med(q["c_calm"]),
                     "not_calm_4h": float(known["calm_normal"].isna().mean()) if len(known) else np.nan})
    return pd.DataFrame(rows).sort_values("times_quiet", ascending=False, ignore_index=True)


def report(ev: pd.DataFrame, ct: pd.DataFrame) -> None:
    years = f"{ev['year'].min()}-{ev['year'].max()}"
    ok = ev["has_bars"].fillna(False)
    print(f"REACTIONS {years} · 1-minute bars, MT5 server clock, bid prices · a day's range = his 10-day ADR")
    print(f"release minutes: {len(ev):,} · with prices around them: {int(ok.sum()):,} · the market was closed: "
          f"{int((~ok).sum()):,}")
    print("by the strongest Forex Factory colour in the minute: "
          + " · ".join(f"{c} {int((ev['colour'] == c).sum()):,}" for c in ("red", "orange", "yellow", "none")))
    print(f"comparison moments (the same New York clock minute, no release within {QUIET_MIN} minutes): {len(ct):,}")
    t = type_table(ev, ct)
    print("\nHis news (red + orange), per report with 20 or more releases - the MIDDLE release of each.")
    print("x quiet = its first-hour range against the same clock minute on days with no release; alone = the releases "
          "with no other report of his colours in the same minute; calm = minutes until the 5-minute range is back to "
          "that clock time's usual size (240 = not within 4 hours); late = 3 of 4 releases are calm by then.")
    print(f"{'report':<24}{'n':>5} {'colour':<7}{'NY':>6}{'5 min $':>9}{'1st hour $':>11}{'of a day':>9}"
          f"{'quiet day':>10}{'x quiet':>8}{'alone':>6}{'x quiet':>8}{'calm':>6}{'late':>6}{'quiet':>6}{'not calm':>9}")
    for r in t.itertuples():
        print(f"{r.report:<24}{r.releases:>5} {r.colour:<7}{r.clock:>6}{r.move_5_usd:>9.2f}{r.range_60_usd:>11.2f}"
              f"{r.range_60_adr:>9.2f}{r.quiet_range_60_adr:>10.2f}{r.times_quiet:>8.2f}{r.alone:>6}"
              f"{r.alone_times_quiet:>8.2f}{r.calm_normal_min:>6.0f}{r.calm_late_min:>6.0f}{r.quiet_calm_min:>6.0f}"
              f"{100 * r.not_calm_4h:>8.0f}%")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.news.reaction", description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="2016-2023 (only once the news beliefs' check years are open)")
    ap.add_argument("--show", action="store_true", help="only print the table that is already built (tune years)")
    a = ap.parse_args(argv)
    if a.show:
        ev, ct = load("events"), load("controls")
    else:
        ev, ct = build(config.TUNE_YEARS + config.CHECK_YEARS if a.all else config.TUNE_YEARS)
        ev, ct = (d[d["year"].isin(config.TUNE_YEARS)] for d in (ev, ct))
    report(ev, ct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
