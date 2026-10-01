"""What was known at each setup's moment (L13.1): the feature builder.

    with settings.using(compiled.resolved):
        feats = features.build(features.setups_from(compiled, market), market, form=form, news=features.his_news())

One row per setup (index = its signal id). Every value uses only what was known at the setup's moment `known_at` (the
close of the candle that made it): the candles closed by then, the levels / zones / lines alive through the last
5-minute candle closed by then, the news releases SCHEDULED in advance (the next one) or already out (the last one).
tests/test_learner.py runs guards.assert_no_future_events on the whole table (R4).

A setup = one signal of an idea: known_at, direction, entry (a limit's price; for a next-open order the close of its
candle - the fill comes later), stop (the hedge point), target (the nearest), and `facts` - what each step of the idea
found (the compiler's trace, machine/ideas/compile.py). The market columns need no facts; the columns about the level
taken, the grab, the break and the gap need them (an idea without such a step leaves them empty).

Clocks: MT5 server time in; India time for the session and the hour (his windows, rules.py); the BROKER day (the server
calendar day, like his daily candles - likhith.md B4) for the weekday, "today" and the day's range.
The session is read the way the idea form's session filter reads a moment (compile.SessionFilter): inside a window
when start < its India time <= end - the candle that makes it lies inside the window. One session rule for filters and
slices: the old project's two session rules agreed on 48% of setups (strategy-understanding section 29).
Every number below is Claude's, fixed before any condition table was seen (plan/work/L131-notes.md).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from machine import config, guards, settings
from machine.blocks import adr as ADR
from machine.blocks import bias as HB
from machine.blocks import fvg as FVG
from machine.blocks import liquidity as LQ
from machine.blocks import one_minute as OM
from machine.blocks import premium_discount as PD
from machine.blocks import sessions as SES
from machine.blocks import trendlines as TL
from machine.blocks import zones as ZN
from machine.data import clock
from machine.ideas import compile as C

DAY_CANDLES = 288                         # 5-minute candles in a day
NEAR_ADR = 0.05                           # "at" a zone / a broken line: within 0.05 of a day's range (the old engine's
                                          # retest_tol_frac, COMPONENTS.md - Claude's)
RETEST_DAYS = 5                           # a broken trendline's retest: within 5 days (AMBIGUITY D4, four-boxes B3-25)
FLIP_DAYS = 10                            # a flipped zone counts for 10 days (AMBIGUITY D5, B3-25)
VOL_LOOKBACK_DAYS = 365                   # C6: today's ADR against its own last 12 months ...
VOL_MIN_DAYS = 60                         # ... once 60 broker days of it exist (early 2016: not known yet)
TREND_DAYS = 5                            # C8, the second trend measure: the last 5 finished broker days
RECENT_CANDLES = 12                       # "recent volatility": the last hour = 12 five-minute candles
TAKE_BLOCKS = ("sweep", "false_breakout")
LINE_CHARTS_AGAINST = ("5m", "1h")        # the 1-minute check reads these charts' lines too (compile.OneMinute)


# ================================================================================================= his news (T7)

# The kind of a Forex Factory row, by its name - Claude's grouping (calendar.KINDS is written for MT5's names: it
# puts Forex Factory's "Unemployment Claims" under growth). The first match wins.
FF_KINDS: tuple[tuple[str, str], ...] = (
    ("the Fed", r"^(FOMC |Federal Funds Rate|Fed Chair|Fed Announcement|Fed Monetary Policy Report|Jackson Hole"
                r"|Fed Governor|Fed Vice Chair)"),
    ("jobs", r"Non-Farm|Nonfarm|Unemployment|Average Hourly Earnings|JOLTS|Employment Cost|Employment Change"
             r"|Challenger"),
    ("inflation", r"CPI|PPI|PCE Price|Price Index|Inflation Expectations|Unit Labor Costs"),
)
_FF_KIND_RE = tuple((k, re.compile(p)) for k, p in FF_KINDS)
OTHER_NEWS = "other news"
HIS_COLOURS = ("red", "orange")           # T7 (LEARNED #141): his news = Forex Factory red and orange


def ff_kind(name: str) -> str:
    """'Non-Farm Employment Change' -> jobs, 'Core CPI m/m' -> inflation, 'FOMC Statement' -> the Fed, else other."""
    for kind, pat in _FF_KIND_RE:
        if pat.search(str(name)):
            return kind
    return OTHER_NEWS


def his_news(start: Any = None, end: Any = None) -> pd.DataFrame:
    """His news - Forex Factory red + orange US rows (T7) with an exact minute: server (naive server time), colour,
    name, kind (FF_KINDS), scheduled. NOT scheduled = known only once it happened: Forex Factory's "Fed Announcement"
    rows (the Fed's unscheduled statements 2020-2023, L12b1-notes.md section 5) and every row at the minute of an
    unscheduled row of the L12 calendar (the Fed's emergency cuts of 3 and 15 March 2020). "All Day" rows (elections)
    have no minute and are left out. start / end: years or dates (server clock)."""
    from machine.news import forexfactory
    asked = [x for x in (start, end) if x is not None]
    if asked:
        guards.check_dates(asked)              # a request reaching 2024 is refused before anything is read (#96)
    ff = forexfactory.load()
    ff = ff[ff["colour"].isin(HIS_COLOURS) & ~ff["timeMasked"].fillna(False).astype(bool) & ff["utc"].notna()]
    server = clock.utc_to_server(pd.DatetimeIndex(pd.to_datetime(ff["utc"], utc=True)))
    out = pd.DataFrame({"server": server, "colour": ff["colour"].to_numpy(), "name": ff["name"].to_numpy()})
    out["kind"] = out["name"].map(ff_kind)
    unscheduled: set = set()
    if config.NEWS_EVENTS.is_file():
        ev = pd.read_parquet(config.NEWS_EVENTS, columns=["server", "scheduled"])
        unscheduled = set(pd.DatetimeIndex(ev.loc[~ev["scheduled"].astype(bool), "server"].dropna()))
    out["scheduled"] = ~(out["name"].eq("Fed Announcement") | out["server"].isin(unscheduled))
    if start is not None:
        out = out[out["server"] >= pd.Timestamp(str(start))]
    if end is not None:
        out = out[out["server"] < pd.Timestamp(f"{pd.Timestamp(str(end)).year + 1}-01-01")]
    guards.check_dates(out[["server"]])
    return out.sort_values(["server", "name"], kind="stable").reset_index(drop=True)


def news_at(news: pd.DataFrame | None, times) -> pd.DataFrame:
    """For each moment: minutes to the next SCHEDULED release at or after it, its kind and colour; minutes since the
    last release at or before it (scheduled or not - it has happened), its kind and colour. NaN / None = none."""
    idx = pd.DatetimeIndex(times)
    n = len(idx)
    out = {"news_next_min": np.full(n, np.nan), "news_next_kind": np.full(n, None, object),
           "news_next_colour": np.full(n, None, object), "news_last_min": np.full(n, np.nan),
           "news_last_kind": np.full(n, None, object), "news_last_colour": np.full(n, None, object)}
    if news is not None and len(news) and n:
        t = idx.as_unit("ns").asi8
        sch = news[news["scheduled"].astype(bool)]
        at = pd.DatetimeIndex(sch["server"]).as_unit("ns").asi8
        k = np.searchsorted(at, t, "left")
        ok = k < len(at)
        out["news_next_min"][ok] = (at[k[ok]] - t[ok]) / 6e10
        out["news_next_kind"][ok] = sch["kind"].to_numpy()[k[ok]]
        out["news_next_colour"][ok] = sch["colour"].to_numpy()[k[ok]]
        at2 = pd.DatetimeIndex(news["server"]).as_unit("ns").asi8
        k2 = np.searchsorted(at2, t, "right") - 1
        ok2 = k2 >= 0
        out["news_last_min"][ok2] = (t[ok2] - at2[k2[ok2]]) / 6e10
        out["news_last_kind"][ok2] = news["kind"].to_numpy()[k2[ok2]]
        out["news_last_colour"][ok2] = news["colour"].to_numpy()[k2[ok2]]
    return pd.DataFrame(out, index=idx)


# ================================================================================================= the setups

@dataclass
class Setup:
    """One signal and what its steps found. entry: a limit's price, or the close of the signal's candle."""
    id: str
    known_at: pd.Timestamp
    direction: str
    entry: float
    stop: float | None
    target: float | None
    facts: dict = field(default_factory=dict)


def setups_from(compiled, market) -> list[Setup]:
    """The setups of a compile made with trace=True (machine.ideas.compile.compile_idea): each signal with the facts of
    the start that made it."""
    if compiled.trace is None:
        raise ValueError("the learner needs the compile's trace: compile_idea(..., trace=True)")
    facts = {rec["signal"]: rec.get("facts") or {} for rec in compiled.trace if rec.get("signal")}
    tf = market.finest()
    closes = market.arrays(tf)["close"]
    out = []
    for s in compiled.signals:
        t = pd.Timestamp(s.known_at)
        entry = float(s.entry_price) if s.entry_type == "limit" else float(closes[market.closed_by(tf, t)])
        target = None
        if s.targets and s.levels == "price":
            target = float(max(s.targets) if s.direction == "sell" else min(s.targets))
        out.append(Setup(s.id, t, s.direction, entry, None if s.stop is None else float(s.stop), target,
                         facts.get(s.id, {})))
    return out


# ================================================================================================= the columns

COLUMNS: dict[str, str] = {
    "direction": "buy or sell",
    "ist_hour": "the hour of the moment, India time (0-23)",
    "session": "his session window the moment falls in, India time (Asia / London / New York / in between)",
    "weekday": "the broker's day of the week (Monday ... Friday)",
    "hours_to_week_end": "hours from the moment to the end of the broker's week (Saturday 00:00 server time)",
    "adr": "a day's range: his 10-day ADR on broker days (machine/blocks/adr.py), gold dollars",
    "volatility_share": "today's ADR against the last 12 months' days (at least 60): the share with a smaller one",
    "adr_used": "today's high - low so far, as a share of a day's range",
    "recent_range": "the last hour's high - low (12 five-minute candles), as a share of a day's range",
    "adr_zone": "where the entry sits in today's ADR zones: near the green / middle / near the red / beyond a zone",
    "adr_share": "the entry's place from the bottom of the green zone (0) to the top of the red zone (1)",
    "bias_hour": "his bias on the hour chart, against the trade: with / against / neutral (blocks/bias.py)",
    "bias_day": "the same on the day chart",
    "bias_week": "the same on the week chart",
    "dial": "the hour AND the day (his dial, compile.dial): with (both) / against (both) / mixed",
    "trend_days": "the last 5 finished broker days: the net move divided by their whole range (1 = a straight line)",
    "at_ath": "the last 5-minute close is at the real all-time high (blocks/bias.py all_time_high)",
    "zone_at_entry": "a live strong zone holds the entry, or its edge is within 0.05 of a day's range",
    "zone_at_target": "the same at the target",
    "zone_flip_at_entry": "a zone that flipped in the last 10 days holds the entry (within 0.05 of a day's range)",
    "line_retest_at_entry": "a trendline (3 touches, 5-minute / hour chart) broken in the last 5 days is priced at the "
                            "entry now (within 0.05 of a day's range)",
    "line_in_path": "a live line (5-minute / hour chart) stands between the entry and the target (#69)",
    "line_against": "a live line (1-minute, 5-minute or hour chart) on the hedge side between the entry and the hedge: "
                    "the 1-minute check would wait (#111)",
    "gap_past_hedge": "an unfilled gap past the hedge (fvg.unfilled_past, #87)",
    "liq_target_adr": "the nearest live level of his (equal, session, trendline, day / week swing) past the entry on "
                      "the target side, as a share of a day's range",
    "liq_hedge_adr": "the same on the hedge side",
    "equal_target_adr": "the nearest live equal highs / lows past the entry on the target side, a share of a day",
    "hedge_adr": "the hedge distance (entry to stop), as a share of a day's range",
    "room_adr": "the room (entry to the nearest target), as a share of a day's range",
    "reward": "the room divided by the hedge distance",
    "level_kind": "the kind of liquidity the idea's take step took (equal / trendline / session / swing)",
    "level_chart": "the chart that level was on (5m / 1h / 1d / 1w; 1m = a 1-minute line the check waited for)",
    "level_rank": "his order of that kind: 3 equal, 2 trendline, 1 session / swing (+1 with 3 or more touches)",
    "level_touches": "its touches known when it was taken",
    "level_day": "when it was formed, against the broker day it was taken: today / yesterday / older",
    "level_age_days": "days from when it was formed to when it was taken",
    "grab_depth_adr": "how far the grab went past the level (to its extreme), as a share of a day's range",
    "break_beyond_adr": "how far the break candle's wick went past his order block's level, a share of a day",
    "gap_size_adr": "the entry gap's size, as a share of a day's range",
    "break_to_gap": "5-minute candles from the break candle to the gap's third candle (0 or less: the gap was made "
                    "inside the breaking move, at the break candle or before it)",
    "leg_position": "the entry between the grab's extreme (0) and the far end of the move so far (1)",
    "target_kind": "the kind of level the target is (when the target is a price a step found)",
    "news_next_min": "minutes to the next scheduled red / orange release (his news, Forex Factory)",
    "news_next_kind": "its kind (jobs / inflation / the Fed / other news)",
    "news_next_colour": "its colour",
    "news_last_min": "minutes since the last red / orange release",
    "news_last_kind": "its kind",
    "news_last_colour": "its colour",
}
NEWS_COLUMNS = tuple(k for k in COLUMNS if k.startswith("news_"))


# ================================================================================================= the builder

def build(setups: list[Setup], market, *, fp: str | None = None, form=None, news: pd.DataFrame | None = None,
          one_minute: bool | None = None) -> pd.DataFrame:
    """The feature table (see the module text): index = signal id; known_at + COLUMNS. fp: the fingerprint of the
    settings the idea was compiled under (the market's cached tables) - call it inside settings.using(<those
    settings>). form: the idea (which step took the level, broke, made the gap, holds the target). news: his_news().
    one_minute: read the 1-minute lines (the market needs 1-minute bars; None = when it has them)."""
    fp = settings.fingerprint() if fp is None else fp
    one_minute = market.has("1m") if one_minute is None else one_minute
    roles = _roles(form)
    ctx = _Context(market, fp, one_minute, {s.facts[roles.take]["level_id"] for s in setups
                                            if roles.take and roles.take in (s.facts or {})})
    rows = [_one(s, ctx, roles) for s in setups]
    df = pd.DataFrame(rows, columns=["signal", "known_at", *COLUMNS])
    df["known_at"] = pd.to_datetime(df["known_at"])
    nw = news_at(news, pd.DatetimeIndex(df["known_at"]))
    for c in NEWS_COLUMNS:
        df[c] = nw[c].to_numpy()
    return df.set_index("signal")


@dataclass
class _Roles:
    take: str | None = None                # the step that took the liquidity (sweep / false breakout)
    brk: str | None = None                 # the break of structure
    gap: str | None = None                 # the gap of the move
    target: str | None = None              # the step whose price is the target


def _roles(form) -> _Roles:
    r = _Roles()
    if form is None:
        return r
    for s in form.steps:
        if r.take is None and s.block in TAKE_BLOCKS:
            r.take = s.id
        elif r.brk is None and s.block == "break_of_structure":
            r.brk = s.id
        elif r.gap is None and s.block == "gap_of_the_move":
            r.gap = s.id
    tg = form.targets[0] if form.targets else None
    if tg is not None and tg.kind == "price" and tg.at:
        r.target = str(tg.at).split(".", 1)[0]
    return r


class _Context:
    """The market's tables, read once (every one is made from the market's bars by its own block)."""

    def __init__(self, market, fp: str, one_minute: bool, take_levels: set):
        self.m, self.fp = market, fp
        self.b5 = market.bars("5m")
        a = market.arrays("5m")
        self.h, self.lo = a["high"], a["low"]
        self.ruler = market.ruler("5m", fp)
        day = self.b5.index.normalize()
        self.day_ns = day.as_unit("ns").asi8
        self.day_start = np.searchsorted(self.day_ns, self.day_ns, "left")   # each candle's day's first candle
        self.days = day.unique()                                             # the broker days with candles
        self.zones_pd = market.adr_zones(fp, "near_the_zone")
        liq = market.liquidity(fp)
        self.sweeps, self.by_level = liq.sweeps, liq.by_level
        b = liq.book[liq.book["level_id"].isin(take_levels)]
        self.touches = {lid: (pd.DatetimeIndex(g["known_at"]).as_unit("ns").asi8, g["touches"].to_numpy())
                        for lid, g in b.groupby("level_id", sort=False)}
        self.zone_life = market.zone_life(fp)
        self.lines5 = market.lines5(fp)                  # every line (2 or 3 touches) of the 5-minute and hour charts
        self.lines_against = (self.lines5[self.lines5["tf"].isin(LINE_CHARTS_AGAINST)] if len(self.lines5)
                              else self.lines5)
        self.broken = _broken_trendlines(market, fp, self.b5)
        self.ath = market.ath5(fp)["at_ath"].to_numpy()
        self.one_minute = one_minute
        self.book1 = market.book1(fp) if one_minute else None
        d = ADR.adr_days(self.b5, "5m")
        self.adr_days = pd.DatetimeIndex(d["day"]).as_unit("ns").asi8
        self.vol_share = _vol_share(self.adr_days, d["ruler"].to_numpy(float))
        self.d1 = market.bars("1d") if market.has("1d") else None


def _broken_trendlines(market, fp: str, b5: pd.DataFrame) -> dict | None:
    """His trendlines (3 touches, #92 / #94; the 5-minute and hour charts, liquidity.TRENDLINE_CHARTS) that were
    broken after they were known: the 5-minute candle whose wick went through each (`taken`), and what gives its
    price on any later candle. A 2-touch line is not a trendline and is not here."""
    parts = [market.lines(tf, fp) for tf in LQ.TRENDLINE_CHARTS if market.has(tf)]
    parts = [p[p["touches"] == 3] for p in parts if len(p)]
    if not any(len(p) for p in parts):
        return None
    on = TL.on_m5(pd.concat(parts, ignore_index=True), b5)
    on = on[(on["taken"] >= 0) & (on["taken"] >= on["start"])]
    if not len(on):
        return None
    return {"taken": on["taken"].to_numpy(np.int64), "x0": on["x0"].to_numpy(np.int64),
            "slope5": on["slope5"].to_numpy(float), "b_price": on["b_price"].to_numpy(float)}


def _vol_share(days_ns: np.ndarray, ruler: np.ndarray) -> np.ndarray:
    """Per broker day: the share of the days of the last 12 months (strictly before it; at least VOL_MIN_DAYS of them
    with an ADR) whose ADR was smaller than this day's. NaN when there are too few."""
    out = np.full(len(days_ns), np.nan)
    back = pd.Timedelta(days=VOL_LOOKBACK_DAYS).value
    for k in range(len(days_ns)):
        if not math.isfinite(ruler[k]):
            continue
        past = ruler[int(np.searchsorted(days_ns, days_ns[k] - back, "left")):k]
        past = past[np.isfinite(past)]
        if len(past) >= VOL_MIN_DAYS:
            out[k] = float((past < ruler[k]).mean())
    return out


def _side_word(bias: int, direction: str) -> str:
    want = HB.UP if direction == "buy" else HB.DOWN
    return "with" if bias == want else "against" if bias == -want else "neutral"


def _one(s: Setup, x: _Context, roles: _Roles) -> list:
    m, t = x.m, pd.Timestamp(s.known_at)
    ist = clock.to_ist(t)
    ist_min = ist.hour * 60 + ist.minute
    session = next((w for w, a, b in SES.windows() if a.hour * 60 + a.minute < ist_min <= b.hour * 60 + b.minute),
                   "in between")
    bday = (t - pd.Timedelta(minutes=1)).normalize()             # the broker day of the candle that made it
    week_end = bday + pd.Timedelta(days=5 - bday.dayofweek)      # Saturday 00:00 server time
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row.update(direction=s.direction, ist_hour=int(ist.hour), session=session, weekday=bday.day_name(),
               hours_to_week_end=(week_end - t) / pd.Timedelta(hours=1))
    i = m.closed_by("5m", t)
    if i >= 0:
        _market(row, s, x, i, t, bday)
        _from_facts(row, s, x, roles, i)
    return [s.id, t, *(row[k] for k in COLUMNS)]


def _market(row: dict, s: Setup, x: _Context, i: int, t: pd.Timestamp, bday: pd.Timestamp) -> None:
    m, sell = x.m, s.direction == "sell"
    r = float(x.ruler[i])
    ok_r = math.isfinite(r) and r > 0
    e, stop, tg = s.entry, s.stop, s.target
    # ---- the day so far, the last hour, today's ADR against its last 12 months
    if ok_r:
        row["adr"] = r
        d0 = int(x.day_start[i])
        row["adr_used"] = (x.h[d0:i + 1].max() - x.lo[d0:i + 1].min()) / r
        a = max(i - RECENT_CANDLES + 1, 0)
        row["recent_range"] = (x.h[a:i + 1].max() - x.lo[a:i + 1].min()) / r
    k = int(np.searchsorted(x.adr_days, x.day_ns[i], "left"))
    if k < len(x.adr_days) and x.adr_days[k] == x.day_ns[i] and math.isfinite(x.vol_share[k]):
        row["volatility_share"] = float(x.vol_share[k])
    zrow = x.zones_pd.iloc[i]
    if math.isfinite(float(zrow["up_hi"])) and math.isfinite(float(zrow["dn_lo"])):
        place = PD.place(e, zrow)
        row["adr_zone"] = ("beyond a zone" if e > float(zrow["up_hi"]) or e < float(zrow["dn_lo"])
                           else "near the red" if place == PD.PREMIUM else "near the green" if place == PD.DISCOUNT
                           else "middle")
        row["adr_share"] = PD.share(e, zrow)
    # ---- his bias on the hour, day and week, the dial, the last 5 days
    for tf, col in (("1h", "bias_hour"), ("1d", "bias_day"), ("1w", "bias_week")):
        if m.has(tf):
            row[col] = _side_word(m.bias_at(tf, x.fp, t), s.direction)
    run = C._Run(m, None, s.direction, x.fp)
    if m.has("1h") and m.has("1d"):
        row["dial"] = C.dial(run, t)
    if x.d1 is not None:
        kd = int(x.d1.index.searchsorted(bday, "left"))            # the days before today are finished
        if kd >= TREND_DAYS:
            seg = x.d1.iloc[kd - TREND_DAYS:kd]
            rng = float(seg["high"].max() - seg["low"].min())
            if rng > 0:
                row["trend_days"] = abs(float(seg["close"].iloc[-1] - seg["open"].iloc[0])) / rng
    row["at_ath"] = bool(x.ath[i])
    # ---- zones and lines at the entry, the target and the hedge
    tol = NEAR_ADR * r if ok_r else 0.0
    z = ZN.alive(x.zone_life, i)
    zl, zh = z["lo"].to_numpy(float), z["hi"].to_numpy(float)
    row["zone_at_entry"] = bool(((zl - tol <= e) & (e <= zh + tol)).any())
    if tg is not None:
        row["zone_at_target"] = bool(((zl - tol <= tg) & (tg <= zh + tol)).any())
    fl = x.zone_life
    death = fl["death"].to_numpy() if len(fl) else np.array([], np.int64)
    f = fl[(fl["dead_by"].to_numpy() == "flip") & (death <= i) & (death > i - FLIP_DAYS * DAY_CANDLES)] if len(fl) \
        else fl
    flo, fhi = f["lo"].to_numpy(float), f["hi"].to_numpy(float)
    row["zone_flip_at_entry"] = bool(((flo - tol <= e) & (e <= fhi + tol)).any())
    l5, br = x.lines5, x.broken
    row["line_retest_at_entry"] = False
    if br is not None:
        recent = (br["taken"] < i) & (br["taken"] > i - RETEST_DAYS * DAY_CANDLES)     # broken on an earlier candle
        price = br["b_price"][recent] + br["slope5"][recent] * (i - br["x0"][recent])  # where the line is now
        row["line_retest_at_entry"] = bool((np.abs(price - e) <= tol).any())
    if tg is not None:
        row["line_in_path"] = TL.in_path(l5, i, e, tg, s.direction) is not None
    if stop is not None:
        side = "high" if sell else "low"
        lo_, hi_ = min(e, stop), max(e, stop)
        found = len(OM.between5(x.lines_against, i, lo_, hi_, side)) if len(l5) else 0
        if x.one_minute:
            i1 = m.closed_by("1m", t)
            if i1 >= 0:
                found += len(OM.between(x.book1, i1, lo_, hi_, side))
            row["line_against"] = found > 0
        if ok_r:
            row["gap_past_hedge"] = FVG.unfilled_past(x.b5, i, s.direction, stop, r) is not None
            row["hedge_adr"] = abs(e - stop) / r
    # ---- his liquidity around the entry
    if ok_r:
        for col, side_, kinds in (("liq_target_adr", run.target_side, C.BUILT_KINDS),
                                  ("liq_hedge_adr", run.hedge_side, C.BUILT_KINDS),
                                  ("equal_target_adr", run.target_side, ("equal",))):
            lv = C._pick_level(run, i, side_, e, kinds, None, "nearest")
            row[col] = None if lv is None else abs(float(lv.price) - e) / r
        if tg is not None:
            row["room_adr"] = abs(tg - e) / r
    if tg is not None and stop is not None and abs(e - stop) > 0:
        row["reward"] = abs(tg - e) / abs(e - stop)


def _family(kind) -> str | None:
    return None if kind is None else str(kind).split("_", 1)[0]


def _broker_day(t: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(t) - pd.Timedelta(minutes=1)).normalize()


def _from_facts(row: dict, s: Setup, x: _Context, roles: _Roles, i: int) -> None:
    """The columns about what the idea's own steps found: the level taken, the grab, the break, the gap, the target."""
    f = s.facts or {}
    sell = s.direction == "sell"
    r = float(x.ruler[i])
    r = r if math.isfinite(r) and r > 0 else None
    take = f.get(roles.take) if roles.take else None
    brk = f.get(roles.brk) if roles.brk else None
    gap = f.get(roles.gap) if roles.gap else None
    if take:
        kind = take.get("kind")
        row["level_kind"] = _family(kind)
        row["level_chart"] = take.get("level_tf")
        rank = LQ.RANK.get(kind)
        t_take = pd.Timestamp(take["known_at"])
        k = x.by_level.get(take.get("level_id"))
        if k is not None:
            born = pd.Timestamp(x.sweeps["level_known_at"].iat[k])
            tt = x.touches.get(take.get("level_id"))
            touches = None
            if tt is not None and (tt[0] <= t_take.value).any():
                touches = int(tt[1][tt[0] <= t_take.value].max())
            row["level_touches"] = touches
            if rank is not None:
                row["level_rank"] = rank + int(bool(touches and touches >= LQ.STRONG_TOUCHES))
            row["level_age_days"] = (t_take - born) / pd.Timedelta(days=1)
            apart = int(x.days.searchsorted(_broker_day(t_take))) - int(x.days.searchsorted(_broker_day(born)))
            row["level_day"] = "today" if apart <= 0 else "yesterday" if apart == 1 else "older"
        elif rank is not None:
            row["level_rank"] = rank
        ext = (brk or {}).get("extreme", take.get("extreme"))
        if ext is not None and math.isfinite(float(ext)):
            if r and take.get("price") is not None:
                row["grab_depth_adr"] = abs(float(ext) - float(take["price"])) / r
            a = x.m.closed_by("5m", t_take)                     # the candle that took the level
            if 0 <= a <= i:
                far = x.lo[a:i + 1].min() if sell else x.h[a:i + 1].max()
                span = abs(float(ext) - float(far))
                if span > 0:
                    row["leg_position"] = abs(float(ext) - s.entry) / span
    if brk and r:
        b = int(x.b5.index.get_loc(pd.Timestamp(brk["bar"])))
        lvl = float(brk["level"])
        row["break_beyond_adr"] = max(0.0, (lvl - x.lo[b]) if sell else (x.h[b] - lvl)) / r
        if gap:
            row["break_to_gap"] = int(x.b5.index.get_loc(pd.Timestamp(gap["bar"]))) - b
    if gap and r and gap.get("size") is not None:
        row["gap_size_adr"] = float(gap["size"]) / r
    if roles.target and f.get(roles.target):
        row["target_kind"] = _family(f[roles.target].get("kind"))
