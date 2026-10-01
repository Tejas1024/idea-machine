"""The events several beliefs share (L13b), each made once per world (World.memo). Every one carries its race (or
what a magnet needs) and its fake comparison, so a belief is only a choice of which events and which numbers.

  takes(w)          every fresh take of his levels that may start a setup (liquidity.sweeps: fresh, can_start - #95,
                    #129), the race after it to the next liquidity the other way (levels.turn_race), and 20 fake takes:
                    fake levels born at the same weekday and hour as the real level, at the same distance, on the same
                    side, taken by their first poke within the real level's life, then the same race, as far (in
                    day's ranges) as the real one
  false_breaks(w)   the takes his false breakout reads (patterns.false_breaks, "false": a candle closed back before one
                    closed beyond, within settings "false_break_close_back_bars"), with fake false breaks
  equal_births(w)   every equal high / low when it is known (touches 2) and every 3rd touch, with 20 fake levels
  zone_touches(w)   every touch of his strong zones (zones.lives), the race after it, and 20 fake bands for each
                    zone's first touch: the same height, born at the same weekday and hour, as far from price
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine import settings
from machine.blocks import equal_levels as EQ
from machine.beliefs import levels as LV
from machine.beliefs.world import DAY

AWAY_ADR = 0.25                  # "price goes the other way": at least this far from the level (B4-14b) - Claude's


def _level_at(lf: pd.DataFrame, ids, pos) -> np.ndarray:
    """His level's price on candle pos (a trendline moves: price + slope x candles from its anchor)."""
    r = lf.loc[list(ids)]
    return (r["price"].to_numpy(float)
            + r["slope"].to_numpy(float) * (np.asarray(pos, dtype=np.int64) - r["x0pos"].to_numpy(np.int64)))


def fake_take_points(w, c, fp, side, life_bars) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each fake level's take (its first poke within life_bars), flattened: (take candle, price, side)."""
    k = c.shape[1]
    j = LV.fake_takes(w, c, fp, side, life_bars).ravel()
    return j, fp.ravel(), np.repeat(np.asarray(side), k)


def fake_races(w, j, price, side, r_adr, k, include_j=True) -> np.ndarray:
    """The same race from each fake (j >= 0), as far as its real level's (r_adr day's ranges at the fake's candle);
    (n, k) with NaN where a fake has no event."""
    ok = j >= 0
    r = np.repeat(np.asarray(r_adr, dtype=float), k) * w.ruler[np.where(ok, j, 0)]
    x = np.full(j.shape, np.nan)
    if ok.any():
        x[ok] = LV.turn_race(w, j[ok], price[ok], price[ok], side[ok], r[ok], include_j=include_j)
    return x.reshape(-1, k)


# ================================================================================================ his takes

def takes(w) -> pd.DataFrame:
    """Every fresh take of his levels that may start a setup, with its race and 20 fake takes (the module text)."""
    def make():
        sw = LV.sweeps(w)
        sw = sw[sw["fresh"] & sw["can_start"]].reset_index(drop=True)
        lf = LV.life(w).drop_duplicates("level_id").set_index("level_id")
        j = w.at_or_after(sw["bar"])
        birth = w.at_or_after(sw["level_known_at"]) - 1
        side = sw["side"].to_numpy()
        sg = LV.sign(side)
        lv_birth = _level_at(lf, sw["level_id"], birth)
        lv = sw["price"].to_numpy(float)
        ok = (birth >= 0) & (j < w.n)
        bb = np.clip(birth, 0, w.n - 1)
        dist = np.where(ok, sg * (lv_birth - w.close[bb]) / w.ruler[bb], np.nan)
        alive = LV.Alive(w)
        turn_side = np.where(side == "high", "low", "high")
        r = LV.next_liquidity_r(w, alive, np.clip(j, 0, w.n - 1), lv, turn_side)
        ok &= np.isfinite(dist) & (dist > 0) & np.isfinite(r)
        x = np.full(len(sw), np.nan)
        x[ok] = LV.turn_race(w, j[ok], lv[ok], lv[ok], side[ok], r[ok])
        r_adr = r / w.ruler[np.clip(j, 0, w.n - 1)]
        pool = np.full((len(sw), LV.K_FAKES), np.nan)
        if ok.any():
            c, fp = LV.fake_prices(w, birth[ok], side[ok], dist[ok], w.rng("takes"))
            life_bars = lf.loc[sw["level_id"][ok], "life_bars"].to_numpy(np.int64)
            jf, f, s = fake_take_points(w, c, fp, side[ok], life_bars)
            pool[ok] = fake_races(w, jf, f, s, r_adr[ok], LV.K_FAKES)
        fam = sw["kind"].str.split("_", n=1).str[0].to_numpy()
        return pd.DataFrame({"pos": j, "birth": birth, "x": x, "pool": list(pool), "kind": sw["kind"].to_numpy(),
                             "family": fam, "tf": sw["tf"].to_numpy(), "side": side, "level": lv, "r_adr": r_adr,
                             "dist_adr": dist, "level_id": sw["level_id"].to_numpy(), "ok": ok})
    return w.memo("takes", make)


def _closes_back(w, j, level, side, window) -> np.ndarray:
    """His false breakout, the old engine's reading (liquidity._false_break): from the take candle j, the first candle
    within `window` that CLOSES back on the side price came from - before any candle closes beyond. -1 = a real break
    or undecided."""
    out = np.full(len(j), -1, np.int64)
    for q in np.flatnonzero(np.asarray(j) >= 0):
        a = int(j[q])
        cl = w.close[a:min(a + window + 1, w.n)]
        hi = side[q] == "high"
        beyond = np.flatnonzero(cl > level[q]) if hi else np.flatnonzero(cl < level[q])
        back = np.flatnonzero(cl < level[q]) if hi else np.flatnonzero(cl > level[q])
        if len(back) and (not len(beyond) or back[0] < beyond[0]):
            out[q] = a + int(back[0])
    return out


def false_breaks(w) -> pd.DataFrame:
    """The takes that were his false breakout (a close back first), the race after the close back, the away-and-back
    count (B4-14b), and 20 fake false breaks each."""
    def make():
        t = takes(w)
        window = int(settings.get("false_break_close_back_bars"))
        side = t["side"].to_numpy()
        lv = t["level"].to_numpy(float)
        k = np.where(t["ok"], _closes_back(w, t["pos"].to_numpy(), lv, side, window), -1)
        fb = t[k >= 0].copy()
        fb["decided"] = k[k >= 0]
        alive = LV.Alive(w)
        s = fb["side"].to_numpy()
        turn_side = np.where(s == "high", "low", "high")
        r = LV.next_liquidity_r(w, alive, fb["decided"].to_numpy(), fb["level"].to_numpy(float), turn_side)
        fb["r_adr"] = r / w.ruler[fb["decided"].to_numpy()]
        fb["x_other_way"] = LV.turn_race(w, fb["decided"].to_numpy(), fb["level"].to_numpy(float),
                                         fb["level"].to_numpy(float), s, r, include_j=False)
        fb["x_back"] = away_and_back(w, fb["decided"].to_numpy(), fb["level"].to_numpy(float), s)
        c, fp = LV.fake_prices(w, fb["birth"].to_numpy(), s, fb["dist_adr"].to_numpy(float), w.rng("false_breaks"))
        life_bars = LV.life(w).drop_duplicates("level_id").set_index("level_id").loc[fb["level_id"], "life_bars"]
        jf, f, ss = fake_take_points(w, c, fp, s, life_bars.to_numpy(np.int64))
        kf = _closes_back(w, jf, f, ss, window)
        fb["pool_other_way"] = list(fake_races(w, kf, f, ss, fb["r_adr"].to_numpy(float), LV.K_FAKES, include_j=False))
        back = np.full(kf.shape, np.nan)
        m = kf >= 0
        back[m] = away_and_back(w, kf[m], f[m], ss[m])
        fb["pool_back"] = list(back.reshape(-1, LV.K_FAKES))
        return fb.reset_index(drop=True)
    return w.memo("false_breaks", make)


def away_and_back(w, k, level, side, days: int = 3) -> np.ndarray:
    """B4-14b: after the close back at candle k, does price first go AWAY_ADR of a day's range the other way, then come
    back to the level - both within `days` days? 1.0 / 0.0 (the whole window was there) / NaN (the years ended)."""
    k = np.asarray(k, dtype=np.int64)
    level = np.asarray(level, dtype=float)
    sg = LV.sign(side)
    start = k + 1
    span = days * DAY
    away_px = level - sg * AWAY_ADR * w.ruler[np.clip(k, 0, w.n - 1)]
    away = LV.first_poke(w, start, away_px, np.where(sg > 0, "low", "high"), span)
    out = np.where(start + span <= w.n, 0.0, np.nan)
    got = away >= 0
    if got.any():
        rest = start[got] + span - away[got] - 1
        back = w.reach(away[got] + 1, level[got], np.asarray(side)[got], np.maximum(rest, 0))
        out[got] = np.where(back == 1.0, 1.0, out[got])
    return out


# ======================================================================================================== equal levels

def equal_births(w) -> pd.DataFrame:
    """Every equal high / low when it is known (touches 2) and every 3rd touch: the candle, the distance from price in
    day's ranges, how far apart the two touches were in cents (exact = 0-2 cents), and 20 fake levels (candles and
    prices; each belief measures its own window)."""
    def make():
        bk = LV.book(w)
        eq = bk[bk["kind"].isin(["equal_high", "equal_low"]) & bk["touches"].isin([2, 3])].reset_index(drop=True)
        pos = w.at_or_after(eq["known_at"]) - 1
        side = eq["side"].to_numpy()
        pp = np.clip(pos, 0, w.n - 1)
        dist = LV.sign(side) * (eq["price"].to_numpy(float) - w.close[pp]) / w.ruler[pp]
        ok = (pos >= 0) & np.isfinite(dist) & (dist > 0)
        gap = _equal_gaps(w, eq)
        c = np.full((len(eq), LV.K_FAKES), -1, np.int64)
        fp = np.full((len(eq), LV.K_FAKES), np.nan)
        if ok.any():
            c[ok], fp[ok] = LV.fake_prices(w, pos[ok], side[ok], dist[ok], w.rng("equal_births"))
        return pd.DataFrame({"pos": pos, "price": eq["price"].to_numpy(float), "side": side, "tf": eq["tf"].to_numpy(),
                             "touches": eq["touches"].to_numpy(), "level_id": eq["level_id"].to_numpy(),
                             "dist_adr": dist, "gap_cents": gap, "ok": ok, "fake_c": list(c), "fake_price": list(fp)})
    return w.memo("equal_births", make)


def _equal_gaps(w, eq: pd.DataFrame) -> np.ndarray:
    """How far the touch that made each row was from the first touch, in cents (equal_levels.gap_usd)."""
    out = np.full(len(eq), np.nan)
    for tf in eq["tf"].unique():
        e = EQ.equal_levels(w.market.bars(tf), tf, w.market.ruler(tf, w.fp))
        key = dict(zip(zip(e["level_id"], e["touches"]), np.round(100 * e["gap_usd"].to_numpy(float))))
        m = (eq["tf"] == tf).to_numpy()
        out[m] = [key.get((lid, t), np.nan) for lid, t in zip(eq["level_id"][m], eq["touches"][m])]
    return out


def magnet(w, eb: pd.DataFrame, days: float) -> tuple[np.ndarray, list]:
    """For equal_births rows: reached within `days` days after being known (x), and each fake level the same (pool)."""
    bars = int(round(days * DAY))
    pos = eb["pos"].to_numpy()
    x = np.where(eb["ok"], w.reach(np.clip(pos, 0, w.n - 1) + 1, eb["price"].to_numpy(float), eb["side"].to_numpy(),
                                   bars), np.nan)
    c = np.stack(eb["fake_c"].to_numpy())
    fp = np.stack(eb["fake_price"].to_numpy())
    return x, LV.pools(LV.fake_reach(w, c, fp, eb["side"].to_numpy(), bars))


# ================================================================================================ his strong zones

def zone_touches(w) -> pd.DataFrame:
    """Every touch of every strong zone (zones.lives: a candle entering the band after one wholly outside; role res =
    price came from below, sup = from above; a listed "?" - price already inside when the zone became known - is no
    touch), the race after it, and for each zone's FIRST touch 20 fake bands (the module text)."""
    def make():
        zl = w.market.zone_life(w.fp)
        alive = LV.Alive(w)
        rows = []
        for z in zl.itertuples():
            # only a touch that ENTERS the band from below ("res") or from above ("sup"); zones.lives also lists a
            # "?" when price already sat inside the band as the zone became known - no side, not a touch (fixed
            # 30 Sep after the first check run: it had been counted as the first touch, "from above")
            proper = [t for t in z.touches if t[1] in ("res", "sup")]
            for n_touch, (cand, role, broke) in enumerate(proper, start=1):
                rows.append((z.zone_id, z.window, z.tf, int(z.tier), int(z.quiet_before), float(z.lo), float(z.hi),
                             int(z.born) - 1, int(z.life_bars), int(cand), role, bool(broke), n_touch))
        t = pd.DataFrame(rows, columns=["zone_id", "window", "tf", "tier", "quiet_before", "lo", "hi", "birth",
                                        "life_bars", "pos", "role", "broke", "touch"])
        side = np.where(t["role"] == "res", "high", "low")
        near = np.where(side == "high", t["lo"], t["hi"])
        far = np.where(side == "high", t["hi"], t["lo"])
        turn_side = np.where(side == "high", "low", "high")
        pos = t["pos"].to_numpy()
        r = LV.next_liquidity_r(w, alive, pos, near, turn_side)
        t["side"] = side
        t["r_adr"] = r / w.ruler[pos]
        t["x"] = LV.turn_race(w, pos, near, far, side, r)
        # the fake bands, for first touches: same height, the centre as far from the birth candle's close
        first = (t["touch"] == 1).to_numpy() & np.isfinite(t["x"].to_numpy())
        pool = np.full((len(t), LV.K_FAKES), np.nan)
        if first.any():
            f = t[first]
            b = f["birth"].to_numpy()
            ruler_b = w.ruler[b]
            centre = ((f["lo"] + f["hi"]) / 2).to_numpy(float)
            off = (centre - w.close[b]) / ruler_b
            height = (f["hi"] - f["lo"]).to_numpy(float) / ruler_b
            c = w.controls(b, LV.K_FAKES, w.rng("zone_touches"))
            cc = np.where(c >= 0, c, 0)
            mid = w.close[cc] + off[:, None] * w.ruler[cc]
            half = height[:, None] * w.ruler[cc] / 2
            lo, hi = np.where(c >= 0, mid - half, np.nan), np.where(c >= 0, mid + half, np.nan)
            life_bars = f["life_bars"].to_numpy()
            jf, came = LV.band_touches(w, c, lo, hi, life_bars)
            ok = jf >= 0
            s = np.where(came > 0, "high", "low")
            nr = np.where(came > 0, lo, hi)
            fr = np.where(came > 0, hi, lo)
            rr = f["r_adr"].to_numpy()[:, None] * w.ruler[np.where(ok, jf, 0)]
            xf = np.full(c.shape, np.nan)
            xf[ok] = LV.turn_race(w, jf[ok], nr[ok], fr[ok], s[ok], rr[ok])
            pool[first] = xf
        t["pool"] = list(pool)
        return t
    return w.memo("zone_touches", make)
