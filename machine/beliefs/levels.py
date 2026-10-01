"""The tools several beliefs share (L13b): his liquidity alive at a moment, the race after a take or a touch, and the
three fair comparisons - fake levels at the same distance, fake bands of the same size, a random moment at the same
weekday and hour.

All of them read the world's 5-minute candles (machine/beliefs/world.py) and his blocks through its market (the tables
machine/ideas/compile.py Market makes once). Claude's numbers, the same for every belief that uses them:

  K_FAKES        20     fake levels (or moments) drawn for each event
  NEXT_MIN_ADR   0.10   "the next liquidity" must be at least this far, in day's ranges (a race of a few cents is noise)
  NEXT_MAX_ADR   2.5    and at most this far (four-boxes B3-26, E3: "next liquidity up to 2.5 day's ranges")

THE NEXT LIQUIDITY (the target of "the real move", "a turn", "a bounce"): the NEAREST of his levels alive at the close
of the candle - known by then and not taken by then - past the price, on that side. His real liquidity only: equal
highs / lows, session highs / lows, trendlines (at their price on that candle, then held flat), hour, day and week
swings; never a 5-minute swing (#91: a target only when the higher timeframe is against the trade, not known here).
The nearest, not his #130 order: "the next" is the first one price meets (Claude's reading).

A RACE AFTER A TAKE OR A TOUCH (at candle j, of a level or a band edge, `ref`): the turn price is R back from `ref`, the
far price R past the level's far side. The candle j itself: its extreme past the far price counts (that came after the
take); its other extreme at the turn price is UNCLEAR (the candle may have come from there before it took the level),
so the event is not counted. Else the race goes on from candle j + 1 (world.race).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.blocks._base import first_hit
from machine.beliefs.world import BOTH, DAY, DOWN, OPEN, UP

K_FAKES = 20
NEXT_MIN_ADR = 0.10
NEXT_MAX_ADR = 2.5
HIS_KINDS = ("equal", "session", "trendline", "swing")


def sign(side) -> np.ndarray:
    """+1 for a high (above price, taken upward), -1 for a low."""
    return np.where(np.asarray(side) == "high", 1.0, -1.0)


# ================================================================================================ his liquidity

def life(w) -> pd.DataFrame:
    """His levels with their lives on the world's 5-minute candles (liquidity.lifetimes via the market)."""
    return w.market.liquidity(w.fp).life


def book(w) -> pd.DataFrame:
    return w.market.liquidity(w.fp).book


def sweeps(w) -> pd.DataFrame:
    return w.market.liquidity(w.fp).sweeps


class Alive:
    """His levels, sorted by the candle they start, for "what is alive at the close of candle p" lookups."""

    def __init__(self, w, only_kinds=HIS_KINDS):
        lf = life(w)
        fam = lf["kind"].str.split("_", n=1).str[0]
        keep = fam.isin(only_kinds) & (lf["target_when"] != "higher_timeframe_against")
        lf = lf[keep].sort_values("start", kind="stable")
        self.start = lf["start"].to_numpy(np.int64)
        self.end = lf["end"].to_numpy(np.int64)
        self.price = lf["price"].to_numpy(float)
        self.slope = lf["slope"].to_numpy(float) if "slope" in lf else np.zeros(len(lf))
        self.x0 = lf["x0pos"].to_numpy(np.int64)
        self.high = (lf["side"] == "high").to_numpy()
        self.fam = fam[keep].reindex(lf.index).to_numpy()
        self.ids = lf["level_id"].to_numpy()
        self.longest = int(lf["life_bars"].max()) + 1 if len(lf) else 0

    def nearest(self, pos, side, beyond, *, kinds=None, skip_ids=None, known_by=None) -> tuple[np.ndarray, np.ndarray]:
        """For each query: the nearest level alive at the CLOSE of candle pos - known by then (start <= pos + 1), not
        taken or gone by then (end > pos) - on `side` of `beyond` ("high": above it). kinds: only these families
        ("equal" ...). known_by: only levels already known when candle known_by opened (an OLDER level). Returns
        (price, level id); NaN / None when there is none."""
        pos = np.asarray(pos, dtype=np.int64)
        side = np.broadcast_to(np.asarray(side), pos.shape)
        beyond = np.broadcast_to(np.asarray(beyond, dtype=float), pos.shape)
        price = np.full(pos.shape, np.nan)
        ids = np.full(pos.shape, None, dtype=object)
        lo_idx = np.searchsorted(self.start, pos + 1 - self.longest, "left")
        last = pos + 1 if known_by is None else np.minimum(pos + 1, np.asarray(known_by, dtype=np.int64))
        hi_idx = np.searchsorted(self.start, last, "right")
        for q in range(len(pos)):
            a, b = lo_idx[q], hi_idx[q]
            if a >= b:
                continue
            p = pos[q]
            m = self.end[a:b] > p
            want_high = side[q] == "high"
            m &= self.high[a:b] == want_high
            if kinds is not None:
                m &= np.isin(self.fam[a:b], kinds)
            if skip_ids is not None and skip_ids[q] is not None:
                m &= self.ids[a:b] != skip_ids[q]
            if not m.any():
                continue
            px = self.price[a:b][m] + self.slope[a:b][m] * (p - self.x0[a:b][m])
            ok = (px > beyond[q]) if want_high else (px < beyond[q])
            if not ok.any():
                continue
            px = px[ok]
            k = int(np.argmin(np.abs(px - beyond[q])))
            price[q] = px[k]
            ids[q] = self.ids[a:b][m][ok][k]
        return price, ids


def next_liquidity_r(w, alive: Alive, pos, ref, turn_side) -> np.ndarray:
    """R for a turn from `ref` at the close of candle pos: the distance to the next liquidity on turn_side ("low":
    below ref), in dollars; NaN when there is none, or it is nearer than NEXT_MIN_ADR or farther than NEXT_MAX_ADR of
    a day's range."""
    px, _ = alive.nearest(pos, turn_side, ref)
    r = np.abs(np.asarray(ref, dtype=float) - px)
    ruler = w.ruler[np.asarray(pos, dtype=np.int64)]
    ok = (r >= NEXT_MIN_ADR * ruler) & (r <= NEXT_MAX_ADR * ruler)
    return np.where(ok, r, np.nan)


# ======================================================================================================== races

def turn_race(w, j, near, far, side, r, max_bars=5 * DAY, include_j: bool = True) -> np.ndarray:
    """The race after a take / touch at candle j (see the module text). side: the side the level is on ("high": the
    level is above, price came up to it, the turn is DOWN). near: the price the turn is measured from (the level, or
    the band edge price met); far: the far side (the level itself, or the band's far edge). 1.0 = the turn price
    (near -/+ r) came first, 0.0 = the far price (far +/- r) came first, NaN = unclear or unfinished.
    include_j=False: candle j is only the moment it starts from (its close is known - a close back), the race is the
    candles after it."""
    j = np.asarray(j, dtype=np.int64)
    sg = sign(side)
    near = np.asarray(near, dtype=float)
    far = np.asarray(far, dtype=float)
    r = np.asarray(r, dtype=float)
    turn = near - sg * r
    past = far + sg * r
    out = np.full(j.shape, np.nan)
    ok = (j >= 0) & (j < w.n) & np.isfinite(r)
    jj = np.clip(j, 0, w.n - 1)
    hi_side = sg > 0
    if include_j:
        past_now = np.where(hi_side, w.high[jj] >= past, w.low[jj] <= past)
        turn_now = np.where(hi_side, w.low[jj] <= turn, w.high[jj] >= turn)
    else:
        past_now = turn_now = np.zeros(j.shape, bool)
    out[ok & ~turn_now & past_now] = 0.0
    go = ok & ~turn_now & ~past_now
    if go.any():
        up = np.where(hi_side, past, turn)[go]
        down = np.where(hi_side, turn, past)[go]
        ans, _ = w.race(j[go] + 1, up, down, max_bars)
        x = np.full(ans.shape, np.nan)
        x[ans == UP] = np.where(hi_side[go], 0.0, 1.0)[ans == UP]
        x[ans == DOWN] = np.where(hi_side[go], 1.0, 0.0)[ans == DOWN]
        out[go] = x
    return out


def move_race(w, start, from_price, r, way, max_bars=5 * DAY) -> np.ndarray:
    """From the close of candle start - 1, does price go r its `way` (+1 up, -1 down) before r the other way? 1.0 / 0.0
    / NaN (unclear or unfinished). The race for a random moment and for the ADR beliefs."""
    start = np.asarray(start, dtype=np.int64)
    from_price = np.asarray(from_price, dtype=float)
    r = np.asarray(r, dtype=float)
    way = np.broadcast_to(np.asarray(way, dtype=float), start.shape)
    ans, _ = w.race(start, from_price + r, from_price - r, max_bars)
    out = np.full(start.shape, np.nan)
    out[ans == UP] = (way[ans == UP] > 0).astype(float)
    out[ans == DOWN] = (way[ans == DOWN] < 0).astype(float)
    out[~np.isfinite(r) | (start >= w.n)] = np.nan
    return out


# ============================================================================================== fair comparisons

def fake_prices(w, birth, side, dist_adr, rng, k=K_FAKES) -> tuple[np.ndarray, np.ndarray]:
    """k fake levels for each real one: born on a candle of the same year, weekday and hour as the real level's birth
    candle (never the same day), on the same side of price, as far from that candle's close (in day's ranges) as the
    real level was from its birth candle's close. Returns (fake birth candles (n, k), fake prices (n, k)); -1 / NaN
    where no candle was found."""
    c = w.controls(birth, k, rng)
    ok = c >= 0
    cc = np.where(ok, c, 0)
    px = w.close[cc] + sign(side)[:, None] * np.asarray(dist_adr, dtype=float)[:, None] * w.ruler[cc]
    return c, np.where(ok, px, np.nan)


def fake_reach(w, c, price, side, max_bars) -> np.ndarray:
    """Did price come to each fake level within max_bars candles after its birth candle? (n, k) of 1 / 0 / NaN."""
    n, k = c.shape
    s = np.repeat(np.asarray(side), k)
    out = w.reach(c.ravel() + 1, price.ravel(), s, max_bars).reshape(n, k)
    return np.where(c >= 0, out, np.nan)


def first_poke(w, start, price, side, max_bars) -> np.ndarray:
    """The first candle from `start` on whose wick goes PAST price (a high above it, a low below it - "even 1 point",
    #95), within max_bars; -1 if none."""
    start = np.asarray(start, dtype=np.int64)
    price = np.asarray(price, dtype=float)
    hi = np.asarray(side) == "high"
    out = np.full(start.shape, -1, np.int64)
    end = np.minimum(start + max_bars, w.n)
    for m, arr, op in ((hi, w.high, ">"), (~hi, w.low, "<")):
        if m.any():
            out[m] = first_hit(arr, start[m], price[m], op, ends=end[m])
    return out


def fake_takes(w, c, price, side, life_bars) -> np.ndarray:
    """The candle that takes each fake level (its first poke within life_bars after its birth), -1 if none."""
    n, k = c.shape
    s = np.repeat(np.asarray(side), k)
    j = first_poke(w, c.ravel() + 1, price.ravel(), s, np.repeat(np.asarray(life_bars), k) if np.ndim(life_bars)
                   else life_bars)
    return np.where(c.ravel() >= 0, j, -1).reshape(n, k)


def pools(values: np.ndarray) -> list[np.ndarray]:
    """(n, k) comparison values -> one array per event, the missing ones left out."""
    return [row[np.isfinite(row)] for row in np.asarray(values, dtype=float)]


def same_group_pools(values: np.ndarray, groups: pd.DataFrame, want: np.ndarray, keys: list[str]) -> list:
    """Two kinds of event compared (equal highs vs session highs; a first touch vs a later one): for each event of the
    wanted kind, its pool = the values of the OTHER kind with the same keys (e.g. year and side). values: one per event;
    want: True = the belief's kind, False = the comparison kind. Returns one pool per event (None for the comparison
    kind's own rows)."""
    g = groups[keys].astype(str).agg("|".join, axis=1).to_numpy()
    v = np.asarray(values, dtype=float)
    other = {}
    for key in np.unique(g):
        m = (g == key) & ~want & np.isfinite(v)
        other[key] = v[m]
    return [other.get(key) if w_ else None for key, w_ in zip(g, want)]


def band_touches(w, c, lo, hi, max_bars) -> tuple[np.ndarray, np.ndarray]:
    """The first touch of each band [lo, hi] after its birth candle c: the first candle whose range enters the band
    after a candle wholly outside it (the birth candle counts as the one before), within max_bars (one per row of c,
    or one for all). Returns (touch candle, came from: +1 below / -1 above / 0 none), each shaped like c."""
    shape = c.shape
    span = np.broadcast_to(np.asarray(max_bars, dtype=np.int64).reshape(-1, *([1] * (c.ndim - 1))), shape).ravel() \
        if np.ndim(max_bars) else np.full(c.size, int(max_bars), np.int64)
    c = c.ravel()
    lo, hi = np.asarray(lo, dtype=float).ravel(), np.asarray(hi, dtype=float).ravel()
    j = np.full(c.shape, -1, np.int64)
    came = np.zeros(c.shape, np.int64)
    for q in np.flatnonzero((c >= 0) & np.isfinite(lo)):
        a, b = int(c[q]), min(int(c[q]) + int(span[q]) + 1, w.n)
        h, l_ = w.high[a:b], w.low[a:b]
        inside = (l_ <= hi[q]) & (h >= lo[q])
        enter = np.flatnonzero(inside[1:] & ~inside[:-1])
        if not len(enter):
            continue
        k = int(enter[0]) + 1
        j[q] = a + k
        came[q] = 1 if h[k - 1] < lo[q] else -1
    return j.reshape(shape), came.reshape(shape)


def news_times(w) -> np.ndarray:
    """His news (Forex Factory red + orange, T7) released inside the world, server time in ns (sorted)."""
    from machine.tester.hedging import news_moments

    def make():
        return news_moments(w.t[0], w.t[-1], 0)[:, 1]
    return w.memo("news_times", make)


def news_between(w, start_ns, end_ns) -> np.ndarray:
    """Is there a release of his news at or after start and before end (each in ns)?"""
    rel = news_times(w)
    return np.searchsorted(rel, np.asarray(end_ns), "left") > np.searchsorted(rel, np.asarray(start_ns), "left")


def year_of(w, pos) -> np.ndarray:
    return w.year[np.clip(np.asarray(pos, dtype=np.int64), 0, w.n - 1)]


def times(w, pos) -> pd.DatetimeIndex:
    """The moment an event is KNOWN: the close of its candle (its open + 5 minutes)."""
    return pd.DatetimeIndex(w.t[np.asarray(pos, dtype=np.int64)]) + pd.Timedelta(minutes=5)


def frame(w, pos, x, pool, **cols) -> pd.DataFrame:
    """The engine's rows: one per event, indexed by the moment it is known, only events of the judged years."""
    pos = np.asarray(pos, dtype=np.int64)
    keep = w.in_years(pos)
    keep[keep] &= np.isin(times(w, pos[keep]).year, w.years)     # a candle closing at midnight on 31 December
    idx = times(w, pos[keep]).rename("known_at")
    out = pd.DataFrame({"x": np.asarray(x, dtype=float)[keep]}, index=idx)
    out["pool"] = pd.Series([p for p, k in zip(pool, keep) if k], index=idx, dtype=object)
    for name, v in cols.items():
        out[name] = np.asarray(v)[keep]
    return out.sort_index(kind="stable")


def group_frame(w, pos, x, is_a, stratum, **cols) -> pd.DataFrame:
    """The engine's rows for a TWO-GROUP belief (Belief.groups): every event of both groups - "A" = his kind, "B" =
    the comparison - with its stratum (the groups are compared inside it: the same year and side), only events of the
    judged years."""
    pos = np.asarray(pos, dtype=np.int64)
    keep = w.in_years(pos)
    keep[keep] &= np.isin(times(w, pos[keep]).year, w.years)
    idx = times(w, pos[keep]).rename("known_at")
    out = pd.DataFrame({"x": np.asarray(x, dtype=float)[keep],
                        "group": np.where(np.asarray(is_a, dtype=bool)[keep], "A", "B"),
                        "stratum": np.asarray(stratum, dtype=object)[keep]}, index=idx)
    for name, v in cols.items():
        out[name] = np.asarray(v)[keep]
    return out.sort_index(kind="stable")


def stratum(w, pos, side) -> np.ndarray:
    """"2019|high": the year of the event's candle and its side - what two groups are compared inside."""
    return np.array([f"{y}|{s}" for y, s in zip(year_of(w, pos), np.asarray(side))], dtype=object)


__all__ = ["group_frame", "stratum", "K_FAKES", "NEXT_MIN_ADR", "NEXT_MAX_ADR", "Alive", "next_liquidity_r", "turn_race",
           "move_race",
           "fake_prices", "fake_reach", "first_poke", "fake_takes", "pools", "same_group_pools", "band_touches",
           "frame", "life", "book", "sweeps", "sign", "year_of", "times", "news_times", "news_between", "BOTH",
           "OPEN"]
