"""His money plan (L20): money for 0.01 lot = 2 x Monday's ADR, the Rs 100-of-chocolate rule, one trade at a time,
a burner account of ~10% with the other 90% in reserve, profit taken out at each double, blow-ups refilled.

    .venv\\Scripts\\python -m machine.money.plan                 # idea #1's plain trades 2016-2023 -> money-plan.md
    .venv\\Scripts\\python -m machine.money.plan --runs 50 --no-write

What every word means - BLOW-UP, REFILL, a cycle, a double, your money - is plan/work/L20-notes.md section 1, fixed
and committed before the plan ran on a single real trade. In short (version 1):
  money needed for 0.01 lot  2 x that week's Monday ADR x 1 ounce x the rupee rate of that Monday's month; the ADR =
                             the larger of the 5- and 10-day average day range (strategy-code.md section 16 USE 3,
                             machine/blocks/adr.py); kept for the week (strategy.md sections 7 and 12; #127, #133)
  lots                       the account's money / the money needed, rounded DOWN to a whole 0.01 step (#115, #116)
  one trade at a time        the trades come in order; each opens after the last one closed (strategy.md line 151)
  the account, the reserve   your money Rs 200,000: 10% in the account at the broker (Rs 20,000 - his number, #114,
                             #115), 90% in reserve (capital-plan.md)
  a double                   after a trade closes, the account at 2 x the cycle's starting money: everything above
                             the starting money is taken out (capital-plan "withdraw at each double / ~75% gain")
  BLOW-UP                    the account below the money needed for 0.01 lot while no trade is open (checked after
                             each close and before each open)
  REFILL                     from the reserve, back up to the account's size (or to the money needed for 0.01 lot when
                             that is more); a reserve too small to reach the money needed for 0.01 lot = the plan
                             stops (the reserve is used up)
  THE STOP-OUT               inside a trade, the account at Rs 0 (the account when the trade opened + the lots x the
                             trade's money at that minute): the broker closes every leg there - a full blow-up, the
                             refill follows. No money goes in while a trade is open (his burner account, strategy.md
                             section 12 line 147) - except row 10, which tops the account up from the reserve inside a
                             trade (plan/work/L20-notes.md section 3)
Any trade table works - plain trades, and his hedged trades (L19b, machine/money/hedged.py): one row per trade, one
trade at a time, with its open, close, the price move in rupees at 0.01 lot (gross_inr), the Rs 40 per position
(cost_inr) and the lowest point inside the trade at 0.01 lot (low_inr, the Rs 40 counted from the start). A hedged
trade also brings its money minute by minute (Inside): where the account reaches Rs 0, and when.

Many account paths run at once (numpy, one row per path): the real order is one path, the 1,000 reshuffles of whole
calendar months are 1,000 paths through the same code. Nothing here writes a file; main() does.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from machine import config, costs, guards
from machine.blocks import adr as ADR

EPOCH = pd.Timestamp("2016-01-01")               # the plan's clock: days since this moment (server time)
DAY_NS = 86_400 * 10**9
RESHUFFLES = 1000                                # the prompt: 1,000 reshuffles
SEED = 20260930                                  # fixed in the code before any run on real trades (L20-notes 1d)
TRADE_COLUMNS = ("open", "close", "gross_inr", "cost_inr", "low_inr")


# ================================================================================================================
# the plan's settings
# ================================================================================================================

@dataclass(frozen=True)
class Plan:
    """One setting of his money plan. Version 1 = his words (L20-notes 1a, 1b); every extra row changes ONE field."""
    name: str = "his plan, version 1"
    change: str = ""                     # the one change from version 1, in words ("" = version 1)
    your_money: float = 200_000.0        # the account + the reserve at the start
    account_share: float = 0.10          # the burner account: ~10% of capital (capital-plan.md)
    capital: str = "2xADR"               # "2xADR": 2 x that Monday's ADR | "sheet": sheet_inr every week
    sheet_inr: float = 7_100.0           # capital-plan.md sheet: "Initial 7100 ... 0.01"
    sizing: str = "chocolate"            # "chocolate": the account / the money needed, whole steps | "flat": 0.01
    withdraw_at: float | None = 2.0      # take the profit out at this x the cycle's start (None = never)
    refill_to: str = "size"              # "size": back to the account's size | "need": only up to 0.01 lot's money
    blowup_line: str = "need"            # "need": below the money for 0.01 lot | "zero": at Rs 0 (it still trades 0.01)
    cost: bool = True                    # False = before the Rs 40 - NOT REAL, only to show what the cost takes
    inside: str = "none"                 # an open trade: "none" = no money goes in (the broker stops it out at Rs 0)
                                         # | "refill" = below the line inside a trade, the reserve tops it up (row 10)

    def __post_init__(self):
        for name, value, allowed in (("capital", self.capital, ("2xADR", "sheet")),
                                     ("sizing", self.sizing, ("chocolate", "flat")),
                                     ("refill_to", self.refill_to, ("size", "need")),
                                     ("blowup_line", self.blowup_line, ("need", "zero")),
                                     ("inside", self.inside, ("none", "refill"))):
            if value not in allowed:
                raise ValueError(f"Plan.{name}={value!r}: use one of {allowed}")
        if not 0 < self.account_share <= 1 or self.your_money <= 0:
            raise ValueError("the account must be a share of your money above 0 and at most all of it")
        if self.withdraw_at is not None and self.withdraw_at <= 1:
            raise ValueError(f"withdraw_at={self.withdraw_at}: the profit is taken out above the cycle's start (> 1)")

    @property
    def account_size(self) -> float:
        return self.your_money * self.account_share

    @property
    def reserve_size(self) -> float:
        return self.your_money - self.account_size

    @property
    def double_at(self) -> float:
        """The x of the cycle's start that ends a cycle in profit: the withdrawal (2 = a double when never taken
        out)."""
        return self.withdraw_at if self.withdraw_at is not None else 2.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


VERSION_1 = Plan()
EXTRA_ROWS: tuple[Plan, ...] = (
    Plan(name="row 1", change="take the profit out at +75%, not at a double", withdraw_at=1.75),
    Plan(name="row 2", change="the account 5% of your money (Rs 10,000)", account_share=0.05),
    Plan(name="row 3", change="the account 20% of your money (Rs 40,000)", account_share=0.20),
    Plan(name="row 4", change="never take the profit out", withdraw_at=None),
    Plan(name="row 5", change="no chocolate rule: 0.01 lot always", sizing="flat"),
    Plan(name="row 6", change="his sheet's number: Rs 7,100 for 0.01 lot every week", capital="sheet"),
    Plan(name="row 7", change="refill only up to the money needed for 0.01 lot", refill_to="need"),
    Plan(name="row 8", change="a blow-up only at Rs 0 (below the line it still trades 0.01)", blowup_line="zero"),
    Plan(name="row 9", change="NOT REAL: before the Rs 40 cost", cost=False),
)
ROW_10 = Plan(name="row 10", change="the reserve keeps an open trade alive: below the line inside a trade, it tops "
                                    "the account up to Rs 20,000", inside="refill")
HEDGED_ROWS: tuple[Plan, ...] = (*EXTRA_ROWS, ROW_10)   # row 10 needs each trade's money minute by minute (hedged)


# ================================================================================================================
# money needed for 0.01 lot: 2 x that Monday's ADR
# ================================================================================================================

def weekly_need(m5: pd.DataFrame, multiple: float = 2.0) -> pd.DataFrame:
    """One row per broker week: its first broker day (a Monday, or a Tuesday after a holiday), known_at (when that
    day's first 5-minute bar closed), the ADR in gold dollars (the larger of adr5 and adr10 - strategy-code.md section
    16 USE 3: the top of the red zone to the bottom of the green zone), the rupee rate of that day's month, and
    need_inr = multiple x ADR x 1 ounce x the rate. Weeks without both averages yet (the first two of 2016) are left
    out. Only finished days go into a Monday's ADR (machine/blocks/adr.py), so nothing here sees the future."""
    days = ADR.adr_days(m5, "5m")
    week = pd.DatetimeIndex(days["day"]).to_period("W-SUN")
    first = days.groupby(week, sort=True).head(1).copy()
    first = first[first["adr5"].notna() & first["adr10"].notna()]
    first["adr_usd"] = first[["adr5", "adr10"]].max(axis=1)
    first["usd_inr"] = [costs.usd_inr(d) for d in first["day"]]
    first["need_inr"] = multiple * first["adr_usd"] * config.LOT_STEP * config.OUNCES_PER_LOT * first["usd_inr"]
    out = first[["day", "known_at", "adr5", "adr10", "adr_usd", "usd_inr", "need_inr"]].reset_index(drop=True)
    return guards.check_dates(out) if len(out) else out


def attach_need(trades: pd.DataFrame, needs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Each trade gets the money needed for 0.01 lot of the newest week whose Monday number was KNOWN at or before the
    trade opened. Trades before the first known number are dropped (returned: how many)."""
    t = trades.sort_values("open", kind="stable").reset_index(drop=True)
    n = needs.sort_values("known_at")[["known_at", "day", "adr_usd", "need_inr"]].rename(
        columns={"day": "monday", "known_at": "need_known_at"})
    t["open"], n["need_known_at"] = t["open"].astype("datetime64[ns]"), n["need_known_at"].astype("datetime64[ns]")
    got = pd.merge_asof(t, n, left_on="open", right_on="need_known_at", direction="backward")
    dropped = int(got["need_inr"].isna().sum())
    got = got[got["need_inr"].notna()].reset_index(drop=True)
    if len(got) and (got["need_known_at"] > got["open"]).any():
        raise AssertionError("a trade was given a Monday number known after it opened")
    return got, dropped


# ================================================================================================================
# the trades and the paths through them
# ================================================================================================================

@dataclass(frozen=True)
class Inside:
    """A hedged trade's money inside it at 0.01 lot a leg, kept only at the minutes where it falls to a new low
    (L20-notes 3b): days from the trade's open, and the money - with the Rs 40 of every open leg owed from its open
    (t, v), and before any cost (t0, v0: row 9, NOT REAL). The account can only reach Rs 0 or fall below the line at a
    new low, so these minutes are all the plan needs."""
    t: np.ndarray
    v: np.ndarray
    t0: np.ndarray
    v0: np.ndarray

    @classmethod
    def from_minutes(cls, days: np.ndarray, inr: np.ndarray, inr0: np.ndarray) -> "Inside":
        return cls(*new_lows(days, inr), *new_lows(days, inr0))

    def lows(self, cost: bool = True) -> tuple[np.ndarray, np.ndarray]:
        return (self.t, self.v) if cost else (self.t0, self.v0)


def new_lows(days: np.ndarray, money: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The minutes where the money falls below everything before it (the first minute always counts)."""
    days, money = np.asarray(days, float), np.asarray(money, float)
    run = np.minimum.accumulate(money)
    new = np.ones(len(money), bool)
    new[1:] = run[1:] < run[:-1]
    return days[new], money[new]


def first_at_or_below(t: np.ndarray, v: np.ndarray, level: np.ndarray) -> np.ndarray:
    """For each level: the days from the trade's open to the first new low at or below it (inf: never). v falls
    strictly, so -v rises and a binary search finds it."""
    j = np.searchsorted(-v, -np.asarray(level, float), side="left")
    return np.where(j < len(v), t[np.minimum(j, len(v) - 1)], np.inf)


@dataclass
class TradeSet:
    """The trades as arrays (per 0.01 lot, in rupees), sorted by open. Everything at N x 0.01 lot is N x these."""
    frame: pd.DataFrame
    open_d: np.ndarray                    # days since EPOCH (server time)
    close_d: np.ndarray
    gross: np.ndarray                     # the price move in Rs at 0.01 lot
    cost: np.ndarray                      # Rs 40 x positions at 0.01 lot
    low: np.ndarray                       # the lowest point inside the trade at 0.01 lot (Rs 40 included), <= 0
    need: np.ndarray                      # money needed for 0.01 lot, the trade's week
    month: np.ndarray                     # calendar month of the open: 0 = Jan 2016
    n_months: int
    low0: np.ndarray | None = None        # the lowest point before any cost (row 9); plain trades: low + cost
    inside: list[Inside] | None = None    # hedged trades: each one's new lows (None: plain trades)

    @classmethod
    def build(cls, frame: pd.DataFrame, last_month: str | None = None,
              inside: dict[str, Inside] | None = None) -> "TradeSet":
        """inside: {signal: its money minute by minute} for hedged trades - every trade needs one, and its lowest
        points must be the table's."""
        missing = [c for c in (*TRADE_COLUMNS, "need_inr") if c not in frame.columns]
        if missing:
            raise ValueError(f"the trade table needs the columns {', '.join(missing)}")
        f = frame.sort_values("open", kind="stable").reset_index(drop=True)
        guards.check_dates(f[["open", "close"]])
        o, c = pd.DatetimeIndex(f["open"]), pd.DatetimeIndex(f["close"])
        if (c < o).any():
            raise ValueError("a trade closes before it opens")
        if len(f) > 1 and (o[1:] < c[:-1]).any():
            k = int(np.flatnonzero(o[1:] < c[:-1])[0])
            raise ValueError(f"two trades overlap ({o[k + 1]} opens before {c[k]}): the plan takes one trade at a time")
        if f["need_inr"].isna().any() or (f["need_inr"] <= 0).any():
            raise ValueError("every trade needs its money-needed number above 0")
        if (f["low_inr"] > 0).any():
            raise ValueError("the lowest point inside a trade cannot be above 0 (the Rs 40 is owed from the start)")
        month = np.asarray((o.year - EPOCH.year) * 12 + (o.month - 1), int)
        if (month < 0).any():
            raise ValueError(f"a trade opens before {EPOCH.date()}, the plan's clock")
        last = pd.Period(last_month, "M") if last_month else o[-1].to_period("M")
        n_months = (last.year - EPOCH.year) * 12 + last.month
        if int(month.max()) >= n_months:
            raise ValueError(f"a trade opens after {last}")
        low, cost = f["low_inr"].to_numpy(float), f["cost_inr"].to_numpy(float)
        low0 = f["low0_inr"].to_numpy(float) if "low0_inr" in f.columns else low + cost
        ins = None
        if inside is not None:
            if "signal" not in f.columns or any(s not in inside for s in f["signal"]):
                raise ValueError("a hedged trade has no money minute by minute (Inside)")
            ins = [inside[s] for s in f["signal"]]
            days = _days(c) - _days(o)
            for j, x in enumerate(ins):
                if abs(x.v[-1] - low[j]) > 1e-6 or abs(x.v0[-1] - low0[j]) > 1e-6:
                    raise ValueError(f"{f['signal'][j]}: its minutes' lowest point is not the table's")
                if (np.diff(x.v) >= 0).any() or (np.diff(x.t) < 0).any() or x.t[-1] > days[j] + 1e-9:
                    raise ValueError(f"{f['signal'][j]}: its new lows must fall, in time order, inside the trade")
        return cls(f, _days(o), _days(c), f["gross_inr"].to_numpy(float), cost, low, f["need_inr"].to_numpy(float),
                   month, n_months, low0, ins)

    def __len__(self) -> int:
        return len(self.open_d)

    def month_starts(self) -> tuple[np.ndarray, np.ndarray]:
        """(start, length) of every calendar month from Jan 2016, in days."""
        d = _days(pd.date_range(EPOCH, periods=self.n_months + 1, freq="MS"))
        return d[:-1], np.diff(d)


def _days(t) -> np.ndarray:
    return (pd.DatetimeIndex(t).as_unit("ns").asi8 - EPOCH.as_unit("ns").value) / DAY_NS


def when(days: float) -> pd.Timestamp:
    """A time on the real clock from days since EPOCH."""
    return EPOCH + pd.Timedelta(days=float(days))


@dataclass
class Paths:
    """P orders of the same trades. order[p, i] = the trade at step i of path p; times in days since EPOCH on that
    path's clock; year[p, i] = the path-year of that step (its month's position // 12). t_open / t_close: every trade
    run to its end; raw: the open before a longer trade pushed it - simulate() pushes again from each trade's real end
    (a stop-out ends a trade early)."""
    order: np.ndarray
    t_open: np.ndarray
    t_close: np.ndarray
    year: np.ndarray
    n_years: int
    months: np.ndarray | None = None      # (P, n_months): the month at each position (reshuffles only)
    seed: int | None = None
    raw: np.ndarray | None = None         # (P, N): each step's open before any push (None: = t_open)

    @property
    def P(self) -> int:
        return self.order.shape[0]

    def one(self, p: int) -> "Paths":
        return Paths(self.order[p:p + 1], self.t_open[p:p + 1], self.t_close[p:p + 1], self.year[p:p + 1],
                     self.n_years, None if self.months is None else self.months[p:p + 1], self.seed,
                     None if self.raw is None else self.raw[p:p + 1])


def real_order(ts: TradeSet) -> Paths:
    """The trades as they happened: path-year = the calendar year the trade opened (0 = 2016)."""
    return Paths(np.arange(len(ts))[None, :], ts.open_d[None, :].copy(), ts.close_d[None, :].copy(),
                 (ts.month // 12)[None, :], (ts.n_months + 11) // 12, raw=ts.open_d[None, :].copy())


def reshuffled(ts: TradeSet, runs: int = RESHUFFLES, seed: int = SEED) -> Paths:
    """`runs` orders of whole calendar months, each month used once (L20-notes 1d). A trade keeps its place inside its
    month and its own week's numbers; time runs month after month; a trade that runs past its month's end pushes the
    next trade's open to its close (one trade at a time)."""
    rng = np.random.default_rng(seed)
    M = ts.n_months
    start, length = ts.month_starts()
    perm = np.argsort(rng.random((runs, M)), axis=1)             # perm[p, j] = the month at position j
    pos = np.argsort(perm, axis=1)                                # pos[p, m] = the position of month m
    blen = length[perm]
    bstart = np.cumsum(blen, axis=1) - blen                       # the start of position j on the path's clock
    rows = np.arange(runs)[:, None]
    tpos = pos[rows, ts.month[None, :]]                           # (P, N): each trade's position
    raw = bstart[rows, tpos] + (ts.open_d - start[ts.month])[None, :]
    order = np.argsort(raw, axis=1, kind="stable")
    dur = ts.close_d - ts.open_d
    before_push = np.take_along_axis(raw, order, axis=1)
    t_open = before_push.copy()
    t_close = np.empty_like(t_open)
    prev = np.full(runs, -np.inf)
    for i in range(order.shape[1]):
        t_open[:, i] = np.maximum(t_open[:, i], prev)
        t_close[:, i] = t_open[:, i] + dur[order[:, i]]
        prev = t_close[:, i]
    year = np.take_along_axis(tpos, order, axis=1) // 12
    return Paths(order, t_open, t_close, year, (M + 11) // 12, perm, seed, before_push)


# ================================================================================================================
# one run of the plan (many paths at once)
# ================================================================================================================

@dataclass
class Sim:
    plan: Plan
    paths: Paths
    account: np.ndarray                   # (P,) at the end
    reserve: np.ndarray
    out: np.ndarray                       # taken out, all years
    added: np.ndarray                     # added back after blow-ups, all years
    alive: np.ndarray                     # False: the reserve was used up
    stop_year: np.ndarray                 # the path-year it was used up (-1 = never)
    stop_t: np.ndarray                    # when (days on the path's clock; nan = never)
    taken: np.ndarray                     # trades taken
    pnl: np.ndarray                       # (P, Y) the trades' rupees at the plan's lots = the change in your money
    trades: np.ndarray                    # (P, Y)
    blowups: np.ndarray                   # (P, Y)
    doubles: np.ndarray                   # (P, Y)
    out_y: np.ndarray                     # (P, Y)
    added_y: np.ndarray                   # (P, Y)
    low: np.ndarray                       # (P, Y) the account's lowest point inside a trade (inf: no trade)
    acc_end: np.ndarray                   # (P, Y) the account after the year's last step (nan: no trade)
    dbl_path: np.ndarray                  # every double: its path, path-year, days from the cycle's start
    dbl_year: np.ndarray
    dbl_days: np.ndarray
    steps: np.ndarray | None = None       # (Y, S) trades at each lot step - one path only
    trace: pd.DataFrame | None = None     # one row per trade taken - one path only
    events: list[dict] | None = None      # every blow-up, double, stop-out and the stop - one path only
    stopouts: np.ndarray | None = None    # (P, Y) the broker closed a trade at Rs 0 (each is also a blow-up)
    under: np.ndarray | None = None       # (P, Y) below the line inside a trade, but no blow-up (version 1)
    cut_days: np.ndarray | None = None    # (P,) the days of the stopped trades that never happened
    put_in: np.ndarray | None = None      # (P,) money the reserve put into open trades (row 10; inside `added`)

    @property
    def your_money_end(self) -> np.ndarray:
        return self.reserve + self.account + self.out

    @property
    def result(self) -> np.ndarray:
        return self.your_money_end - self.plan.your_money

    def your_money_start(self) -> np.ndarray:
        """(P, Y): your money at the start of each path-year (only trades change it)."""
        return self.plan.your_money + np.cumsum(self.pnl, axis=1) - self.pnl

    def growth(self) -> np.ndarray:
        """(P, Y): the year's result / your money at its start."""
        return self.pnl / self.your_money_start()


def inside_trade(t: np.ndarray, v: np.ndarray, acc: np.ndarray, steps: np.ndarray, reserve: np.ndarray,
                 need: np.ndarray, target: np.ndarray, top_up: bool) -> dict:
    """One trade's inside for several accounts at once (L20-notes 3c, 3d). t, v: the trade's new lows (Inside) at
    0.01 lot a leg; the account's money at a minute = acc + what was put in so far + steps x v.
      top_up False (version 1): no money goes in; the first new low where the account is at Rs 0 or below is the
        STOP-OUT (the broker closes every leg).
      top_up True (row 10): the first new low BELOW the line (the money needed for 0.01 lot) gets the reserve's money up
        to `target`, again and again; a minute that is already at Rs 0 or below when it first falls under the line is a
        stop-out (no time to add money); once the reserve cannot reach the line, only the stop-out is left.
    Returns: put_in, reserve (left), stopped, t_stop (days from the open; inf: none), lowest (the account's lowest
    point inside; 0 when stopped), tops = [(account index, day, account before, amount), ...] in time order."""
    n, K = len(acc), len(v)
    put, res = np.zeros(n), reserve.astype(float).copy()
    stopped, t_stop = np.zeros(n, bool), np.full(n, np.inf)
    no_more = np.full(n, not top_up) | (res <= 1e-9)
    lowest = np.full(n, np.inf)
    tops: list[tuple[int, float, float, float]] = []
    while True:
        j_line = np.searchsorted(-v, -((need - 1e-9 - acc - put) / steps), side="right")    # first v below the line
        j_zero = np.searchsorted(-v, -((1e-9 - acc - put) / steps), side="left")           # first v at/below Rs 0
        top = ~stopped & ~no_more & (j_line < j_zero)
        stop = ~stopped & ~top & (j_zero < K)
        if not (top.any() or stop.any()):
            break
        stopped[stop], t_stop[stop] = True, t[j_zero[stop]]
        idx = np.flatnonzero(top)
        if len(idx):
            e = acc[idx] + put[idx] + steps[idx] * v[j_line[idx]]
            amt = np.minimum(np.maximum(target[idx] - e, 0.0), res[idx])
            put[idx] += amt
            res[idx] -= amt
            lowest[idx] = np.minimum(lowest[idx], e)
            no_more[idx[(e + amt < need[idx] - 1e-9) | (res[idx] <= 1e-9)]] = True    # the reserve cannot do more
            tops += [(int(a), float(t[j]), float(b), float(c)) for a, j, b, c in zip(idx, j_line[idx], e, amt)]
    lowest = np.where(stopped, 0.0, np.minimum(lowest, acc + put + steps * v[-1]))
    tops.sort(key=lambda x: x[1])
    return {"put_in": put, "reserve": res, "stopped": stopped, "t_stop": t_stop, "lowest": lowest, "tops": tops}


def simulate(ts: TradeSet, paths: Paths, plan: Plan = VERSION_1, *, trace: bool = False) -> Sim:
    """Run the plan along every path (L20-notes 1a, 1b, 3c, 3d). trace=True (one path only) keeps every trade's lots
    and money, and every blow-up, double, stop-out and stop, for the report's tables."""
    P, N = paths.order.shape
    Y = paths.n_years
    if trace and P != 1:
        raise ValueError("trace works on one path")
    if plan.inside == "refill" and ts.inside is None:
        raise ValueError(f"{plan.name}: topping the account up inside a trade needs each trade's money minute by "
                         "minute (hedged trades)")
    size = plan.account_size
    acc, res = np.full(P, size), np.full(P, plan.reserve_size)
    out, added, base = np.zeros(P), np.zeros(P), np.full(P, size)
    raw = paths.raw if paths.raw is not None else paths.t_open
    cyc0 = raw[:, 0].astype(float).copy()
    prev_end = np.full(P, -np.inf)
    alive, stop_year, stop_t = np.ones(P, bool), np.full(P, -1), np.full(P, np.nan)
    taken = np.zeros(P, int)
    pnl_y, trades_y, blow_y, dbl_y, out_y, added_y, stops_y, under_y = (np.zeros((P, Y)) for _ in range(8))
    low_y, acc_end = np.full((P, Y), np.inf), np.full((P, Y), np.nan)
    cut_days, put_all = np.zeros(P), np.zeros(P)
    dbl_p, dbl_yr, dbl_d = [], [], []
    rows = np.arange(P)
    net = ts.gross - (ts.cost if plan.cost else 0.0)
    low001 = ts.low if plan.cost else (ts.low0 if ts.low0 is not None else ts.low + ts.cost)
    length = ts.close_d - ts.open_d
    steps_hist = np.zeros((Y, 2)) if trace else None
    tr: list[dict] | None = [] if trace else None
    events: list[dict] | None = [] if trace else None

    def below(a, need):
        return a < need - 1e-9 if plan.blowup_line == "need" else a <= 1e-9

    def note(t, what, why, before, moved, after, need, reserve=None):
        events.append({"t": float(t), "what": what, "why": why, "account_before": float(before),
                       "money_in_out": float(moved), "account_after": float(after), "need_inr": float(need),
                       "reserve_after": float(res[0] if reserve is None else reserve),
                       "taken_out_so_far": float(out[0])})

    def blow(mask, t, y, need, why):
        nonlocal acc, res, added, base, cyc0
        if not mask.any():
            return
        np.add.at(blow_y, (rows[mask], y[mask]), 1)
        before = acc
        target = need if plan.refill_to == "need" else np.maximum(size, need)   # at least enough for 0.01 lot
        amt = np.where(mask, np.minimum(np.maximum(target - acc, 0.0), res), 0.0)
        acc, res, added = acc + amt, res - amt, added + amt
        np.add.at(added_y, (rows, y), amt)
        dead = mask & below(acc, need)
        alive[dead], stop_year[dead], stop_t[dead] = False, y[dead], t[dead]
        ok = mask & ~dead
        base, cyc0 = np.where(ok, acc, base), np.where(ok, t, cyc0)
        if events is not None and mask[0]:
            note(t[0], "blow-up", why, before[0], amt[0], acc[0], need[0])
            if dead[0]:
                note(t[0], "the reserve is used up", "the reserve cannot bring the account back to the line",
                     acc[0], 0.0, acc[0], need[0])

    for i in range(N):
        k, y = paths.order[:, i], paths.year[:, i]
        t0 = np.maximum(raw[:, i], prev_end)                  # one trade at a time: after the last one really ended
        need = np.full(P, plan.sheet_inr) if plan.capital == "sheet" else ts.need[k]
        blow(alive & below(acc, need), t0, y, need, "before a trade opened")      # a Monday number may have risen
        live = alive.copy()
        if plan.sizing == "chocolate":
            steps = np.floor(acc / need + 1e-9)
            if plan.blowup_line == "zero":
                steps = np.maximum(steps, 1.0)                                       # #114: "20 only fine"
        else:
            steps = np.ones(P)
        steps = np.where(live, steps, 0.0)
        if (steps[live] < 1).any():
            raise AssertionError("a live account had no lot to trade - the blow-up check missed it")
        start_acc = acc
        # inside the open trade (L20-notes 3c): the broker's stop-out at Rs 0; row 10 tops the account up
        lowpt = acc + steps * low001[k]
        put = np.zeros(P)
        stopped = live & (lowpt <= 1e-9)
        t_in = length[k].copy()                                # days from the open to the trade's end
        tops: list[tuple[int, float, float, float]] = []
        if ts.inside is not None:
            groups = live & (stopped | (plan.inside == "refill"))
            for kk in np.unique(k[groups]):
                m = np.flatnonzero(groups & (k == kk))
                tt, vv = ts.inside[kk].lows(plan.cost)
                target = need[m] if plan.refill_to == "need" else np.maximum(size, need[m])
                got = inside_trade(tt, vv, acc[m], steps[m], res[m], need[m], target, plan.inside == "refill")
                stopped[m], lowpt[m], put[m], res[m] = got["stopped"], got["lowest"], got["put_in"], got["reserve"]
                t_in[m] = np.where(got["stopped"], got["t_stop"], t_in[m])
                tops += [(int(m[a]), d, b, c) for a, d, b, c in got["tops"]]
        lowpt = np.where(stopped, 0.0, lowpt)
        pnl = np.where(stopped, -(acc + put), steps * net[k])                       # stopped: the whole account
        end = t0 + t_in
        cut_days += np.where(stopped, length[k] - t_in, 0.0)
        reserve0 = res[0] + put[0]                                                    # path 0 before the top-ups
        for a, d, before, amt in sorted(tops, key=lambda x: x[1]):                    # row 10: each top-up is a
            ta = t0[a] + d                                                            # blow-up and a refill
            blow_y[a, y[a]] += 1
            added[a] += amt
            added_y[a, y[a]] += amt
            put_all[a] += amt
            if before + amt >= need[a] - 1e-9:                                        # a new cycle starts
                base[a], cyc0[a] = before + amt, ta
            if events is not None and a == 0:
                reserve0 -= amt
                note(ta, "blow-up", "inside a trade: below the line, the reserve topped the account up (row 10)",
                     before, amt, before + amt, need[0], reserve0)
        acc = acc + put + pnl
        after_trade = acc
        taken += live
        np.add.at(pnl_y, (rows, y), pnl)
        np.add.at(trades_y, (rows, y), live.astype(float))
        np.add.at(stops_y, (rows, y), stopped.astype(float))
        np.add.at(under_y, (rows, y), (live & ~stopped & (put == 0) & (lowpt < need - 1e-9)).astype(float))
        np.minimum.at(low_y, (rows, y), np.where(live, lowpt, np.inf))
        if events is not None and stopped[0]:
            note(end[0], "stop-out", "the account reached Rs 0 inside the trade: the broker closed every leg",
                 start_acc[0] + put[0], 0.0, 0.0, need[0])
        if steps_hist is not None and live[0]:
            s = int(steps[0])
            if s >= steps_hist.shape[1]:
                steps_hist = np.pad(steps_hist, ((0, 0), (0, s + 1 - steps_hist.shape[1])))
            steps_hist[y[0], s] += 1
        # after the close: a double (the profit is taken out), else a blow-up
        d = live & (acc >= plan.double_at * base - 1e-9)
        taken_out = np.zeros(P)
        if d.any():
            days = end - cyc0
            dbl_p.append(rows[d]), dbl_yr.append(y[d]), dbl_d.append(days[d])
            np.add.at(dbl_y, (rows[d], y[d]), 1)
            if plan.withdraw_at is not None:
                taken_out = np.where(d, acc - base, 0.0)
                out = out + taken_out
                np.add.at(out_y, (rows, y), taken_out)
                acc = np.where(d, base, acc)
            else:
                base = np.where(d, acc, base)
            cyc0 = np.where(d, end, cyc0)
            if events is not None and d[0]:
                what = "a double" if plan.double_at == 2 else f"+{plan.double_at - 1:.0%}"
                note(end[0], what, f"{days[0]:.1f} days after the cycle started", after_trade[0], -taken_out[0],
                     acc[0], need[0])
        if tr is not None and live[0]:
            tr.append({"step": i, "trade": int(k[0]), "need_inr": float(need[0]), "lots": round(steps[0] * 0.01, 2),
                       "inr": float(pnl[0]), "lowest_inside_inr": float(lowpt[0] - start_acc[0]),
                       "account_after_trade": float(after_trade[0]), "taken_out": float(taken_out[0]),
                       "account_at_open": float(start_acc[0]), "account_lowest_inside": float(lowpt[0]),
                       "stopped_out": bool(stopped[0]), "put_in_inside": float(put[0]),
                       "opened_d": float(t0[0]), "ended_d": float(end[0]),
                       "days_cut": float(length[k[0]] - t_in[0]) if stopped[0] else 0.0})
        blow(live & ~d & below(acc, need), end, y, need,
             "after the broker closed the trade at Rs 0" if stopped[0] else "after a trade closed")
        acc_end[rows[live], y[live]] = acc[live]
        prev_end = np.where(live, end, prev_end)

    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype)      # noqa: E731
    return Sim(plan, paths, acc, res, out, added, alive, stop_year, stop_t, taken, pnl_y, trades_y, blow_y, dbl_y,
               out_y, added_y, low_y, acc_end, cat(dbl_p, int), cat(dbl_yr, int), cat(dbl_d, float),
               steps_hist, pd.DataFrame(tr) if tr is not None else None, events,
               stopouts=stops_y, under=under_y, cut_days=cut_days, put_in=put_all)


# ================================================================================================================
# what the report reads
# ================================================================================================================

def pct(a, q) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def days_text(days) -> str:
    """Days per double: the middle and the slow 10% (90 of 100 doubles were faster)."""
    d = np.asarray(days, float)
    if not len(d):
        return "no double"
    if len(d) == 1:
        return f"{d[0]:.0f} days (1 double)"
    return f"middle {pct(d, 50):.0f} days, slow 10% {pct(d, 90):.0f} days ({len(d)} doubles)"


def lots_text(row) -> str:
    """How many trades at each lot size - never an average lot (#107, #116)."""
    parts = [f"{s * 0.01:.2f} x{int(n)}" for s, n in enumerate(row) if n > 0 and s > 0]
    return " · ".join(parts) if parts else "-"


def year_rows(sim: Sim, ts: TradeSet, first_year: int = EPOCH.year) -> list[dict]:
    """The real order, calendar year by calendar year (a trade counts in the year it opened)."""
    if sim.paths.P != 1 or sim.trace is None:
        raise ValueError("year_rows reads the real order: one path, simulated with trace=True")
    tr = sim.trace
    ym, growth = sim.your_money_start()[0], sim.growth()[0]
    trade_year = ts.month[tr["trade"].to_numpy(int)] // 12 if len(tr) else np.zeros(0, int)
    out, prev_end = [], sim.plan.account_size
    for y in range(sim.paths.n_years):
        in_year = tr[trade_year == y]
        end = sim.acc_end[0, y]
        out.append({
            "year": first_year + y, "trades": int(sim.trades[0, y]),
            "lots": lots_text(sim.steps[y]) if sim.steps is not None else "",
            "need_lo": float(in_year["need_inr"].min()) if len(in_year) else float("nan"),
            "need_hi": float(in_year["need_inr"].max()) if len(in_year) else float("nan"),
            "account_start": prev_end, "account_end": float(end) if np.isfinite(end) else prev_end,
            "taken_out": float(sim.out_y[0, y]), "added_back": float(sim.added_y[0, y]),
            "result": float(sim.pnl[0, y]), "your_money_start": float(ym[y]), "growth": float(growth[y]),
            "doubles": int(sim.doubles[0, y]), "days": days_text(sim.dbl_days[sim.dbl_year == y]),
            "blowups": int(sim.blowups[0, y]), "stopouts": int(sim.stopouts[0, y]),
            "under_inside": int(sim.under[0, y]),
            "lowest_inside": float(sim.low[0, y]) if np.isfinite(sim.low[0, y]) else float("nan"),
        })
        if np.isfinite(end):
            prev_end = float(end)
    return out


def spread(sim: Sim) -> dict:
    """The reshuffles: the middle and the bad 10% (L20-notes 1d)."""
    g, b, so = sim.growth(), sim.blowups, sim.stopouts
    per_year = []
    for y in range(sim.paths.n_years):
        dd = sim.dbl_days[sim.dbl_year == y]
        per_year.append({"year": y + 1, "growth_mid": pct(g[:, y], 50), "growth_bad": pct(g[:, y], 10),
                         "blowups_mid": pct(b[:, y], 50), "blowups_bad": pct(b[:, y], 90),
                         "stopouts_mid": pct(so[:, y], 50), "stopouts_bad": pct(so[:, y], 90),
                         "doubles_mid": pct(sim.doubles[:, y], 50), "days_mid": pct(dd, 50),
                         "days_slow": pct(dd, 90), "n_doubles": int(len(dd)),
                         "used_up_by_now": int(((sim.stop_year >= 0) & (sim.stop_year <= y)).sum())})
    worst = int(np.argmin(sim.your_money_end))
    stopped = sim.stop_year >= 0
    total_b, total_so = b.sum(axis=1), so.sum(axis=1)
    return {
        "runs": sim.paths.P, "seed": sim.paths.seed, "per_year": per_year,
        "any_year": {"growth_mid": pct(g, 50), "growth_bad": pct(g, 10), "blowups_mid": pct(b, 50),
                     "blowups_bad": pct(b, 90), "days_mid": pct(sim.dbl_days, 50),
                     "days_slow": pct(sim.dbl_days, 90), "n_doubles": int(len(sim.dbl_days))},
        "all_years": {"result_mid": pct(sim.result, 50), "result_bad": pct(sim.result, 10),
                      "result_good": pct(sim.result, 90), "out_mid": pct(sim.out, 50), "out_bad": pct(sim.out, 10),
                      "added_mid": pct(sim.added, 50), "added_bad": pct(sim.added, 90),
                      "blowups_mid": pct(total_b, 50), "blowups_bad": pct(total_b, 90),
                      "stopouts_mid": pct(total_so, 50), "stopouts_bad": pct(total_so, 90),
                      "runs_with_a_stopout": int((total_so > 0).sum()),
                      "put_in_mid": pct(sim.put_in, 50), "put_in_bad": pct(sim.put_in, 90),
                      "doubles_mid": pct(sim.doubles.sum(axis=1), 50),
                      "runs_reserve_used_up": int(stopped.sum()),
                      "used_up_year_mid": pct(sim.stop_year[stopped] + 1, 50) if stopped.any() else None,
                      "runs_no_double": int((sim.doubles.sum(axis=1) == 0).sum())},
        "worst": {"run": worst, "your_money_end": float(sim.your_money_end[worst]),
                  "result": float(sim.result[worst]), "out": float(sim.out[worst]), "added": float(sim.added[worst]),
                  "blowups": int(total_b[worst]), "doubles": int(sim.doubles[worst].sum()),
                  "stopouts": int(total_so[worst]),
                  "used_up_year": int(sim.stop_year[worst]) + 1 if sim.stop_year[worst] >= 0 else None,
                  "taken": int(sim.taken[worst])},
    }


def headline(sim: Sim, p: int = 0) -> dict:
    """One path's whole run (the real order: p = 0)."""
    mine = sim.dbl_path == p
    stop = when(sim.stop_t[p]).strftime("%b %Y") if sim.stop_year[p] >= 0 else None
    return {"result": float(sim.result[p]), "your_money_end": float(sim.your_money_end[p]),
            "out": float(sim.out[p]), "added": float(sim.added[p]), "account_end": float(sim.account[p]),
            "reserve_end": float(sim.reserve[p]), "blowups": int(sim.blowups[p].sum()),
            "doubles": int(sim.doubles[p].sum()), "days": days_text(sim.dbl_days[mine]),
            "days_mid": pct(sim.dbl_days[mine], 50), "reserve_used_up": stop, "taken": int(sim.taken[p]),
            "lowest_inside": float(np.min(sim.low[p])), "stopouts": int(sim.stopouts[p].sum()),
            "under_inside": int(sim.under[p].sum()), "days_cut": float(sim.cut_days[p]),
            "put_in_inside": float(sim.put_in[p])}


# ================================================================================================================
# the trades it runs on now: idea #1's plain trades (L20-notes 1c)
# ================================================================================================================

IDEA, VERSION, READINGS = "idea-001", "v-8cc96cf812", {"U1": "A", "U2": "A", "U3": "B"}
TUNE_SIGNALS_SHA256 = "724244052366cbe289a6cdedd782a74660cb4653f1eee3576c864f031ecb7605"
RECORDED = {                                         # what the machine reported for these trades (L10)
    "2016-2020": {"trades": 612, "net_inr": -2786.75, "trial": "20260929-074446-734232-2eb4c4"},
    "2021-2023": {"trades": 260, "net_inr": -30980.82, "trial": "20260929-075859-444575-fa68b5"},
}
LABEL = "PLAIN TRADES — NOT HIS HEDGING"


def trade_lows(book, trades: list) -> list[float]:
    """The lowest point inside each trade at 0.01 lot, in rupees, the Rs 40 counted from the start: 1-minute bid lows
    (a buy) / highs (a sell) from the fill minute (all of it - the worse case, lesson B3) up to the minute before the
    close, and the close itself."""
    out = []
    for t in trades:
        lots = sum(e.lots for e in t.exits)
        if abs(lots - config.LOT_STEP) > 1e-9:
            raise ValueError(f"{t.signal_id}: {lots} lot - the plan's rows are per 0.01 lot")
        i0 = int(np.searchsorted(book.t, pd.Timestamp(t.fill_ts).as_unit("ns").value, "left"))
        i1 = int(np.searchsorted(book.t, pd.Timestamp(t.exit_ts).as_unit("ns").value, "left"))
        worst = 0.0
        if i1 > i0:
            worst = (book.lo[i0:i1].min() - t.fill_px) if t.direction == "buy" else (t.fill_px - book.h[i0:i1].max())
        out.append(min(min(worst, 0.0) * t.usd_inr - t.cost_inr, t.net_inr, -t.cost_inr))
    return out


def idea_trades(log=print):
    """Idea #1's plain trades 2016-2023, made again with today's code; stops if they are not the recorded ones.
    Returns (the trades, what was checked, the 5-minute bars 2016-2023 for the Monday numbers)."""
    from machine.data import bars as BARS
    from machine.ideas.compile import Market, compile_idea
    from machine.ideas.form import Idea
    from machine.referee import tune
    from machine.tester import engine

    t0 = time.time()
    form = Idea.load(config.IDEAS_DIR / IDEA / "form.json")
    market = Market.load(str(config.TUNE_YEARS[0]), str(config.CHECK_YEARS[-1]))
    comp = compile_idea(form, market, readings=READINGS)
    if comp.version_id != VERSION:
        raise RuntimeError(f"idea #1's readings compile to {comp.version_id}, not {VERSION}: the form or code changed")
    last_tune = market.bars("5m").loc[:str(config.TUNE_YEARS[-1])].index[-1] + pd.Timedelta(minutes=5)
    before = [s for s in comp.signals if pd.Timestamp(s.known_at) <= last_tune]
    after = [s for s in comp.signals if pd.Timestamp(s.known_at) > last_tune]
    if tune.signals_sha256(before) != TUNE_SIGNALS_SHA256:
        raise RuntimeError("idea #1's 2016-2020 signals are not the ones L10 tested (the fingerprint differs)")
    log(f"{IDEA} {VERSION}: {len(before)} signals 2016-2020 (the same as L10) + {len(after)} in 2021-2023 "
        f"({time.time() - t0:.0f} s)")
    rows, info = [], {"signals": {"2016-2020": len(before), "2021-2023": len(after)}, "segments": {}}
    for label, sigs, (a, b) in (("2016-2020", before, (str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1]))),
                                ("2021-2023", after, (str(config.CHECK_YEARS[0]), str(config.CHECK_YEARS[-1])))):
        m1 = BARS.load_bars("1m", a, b)
        guards.check_opened(IDEA, [m1.index[0], m1.index[-1]], version=VERSION)
        book = engine.bars_book(m1, comp.tf)
        rep = engine.replay(sigs, None, comp.tf, config.TRADES_AT_ONCE, book=book)
        closed = [t for t in rep.trades if t.status == "closed"]
        total = round(sum(t.net_inr for t in closed), 2)
        want = RECORDED[label]
        if (len(closed), total) != (want["trades"], want["net_inr"]):
            raise RuntimeError(f"{label}: {len(closed)} trades {total} - the recorded run (trial {want['trial']}) "
                               f"had {want['trades']} trades {want['net_inr']}: stop and find why")
        for t, lo in zip(closed, trade_lows(book, closed)):
            rows.append({"signal": t.signal_id, "direction": t.direction, "known_at": pd.Timestamp(t.known_at),
                         "open": pd.Timestamp(t.fill_ts), "close": pd.Timestamp(t.exit_ts), "fill_px": t.fill_px,
                         "exit_px": t.exit_px, "closed_by": t.exit_reason, "gross_inr": t.gross_inr,
                         "cost_inr": t.cost_inr, "net_inr": t.net_inr, "usd_inr": t.usd_inr, "low_inr": lo,
                         "segment": label})
        info["segments"][label] = {"trades": len(closed), "net_inr": total,
                                   "open_at_end": len(rep.trades) - len(closed), "recorded_trial": want["trial"]}
        log(f"  {label}: {len(closed)} trades, {costs.inr(total)} at 0.01 lot - the same as trial {want['trial']}")
    info["seconds"] = round(time.time() - t0)
    return pd.DataFrame(rows), info, market.bars("5m")


# ================================================================================================================
# what plain trades cannot show: how far a HEDGED trade dips before it closes (L19a's recorded trades)
# ================================================================================================================

HEDGED_STORIES = config.WORK_DIR / "L19a" / "idea-001-hedged-stories.jsonl"
HEDGED_TRIAL = "20260929-222235-794384-f21105"


@dataclass
class MinuteBook:
    """The 1-minute bars as the hedged measures read them: times, bid lows and highs, and the rupee rate of each
    minute's month."""
    idx: pd.DatetimeIndex
    t: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    rate: np.ndarray

    @classmethod
    def build(cls, m1: pd.DataFrame) -> "MinuteBook":
        idx = pd.DatetimeIndex(m1.index).as_unit("ns")
        month = np.asarray(idx.year * 12 + idx.month)                                  # Jan 2016 = 2016 * 12 + 1
        months = np.unique(month)
        rates = np.array([costs.usd_inr(pd.Timestamp(year=(m - 1) // 12, month=(m - 1) % 12 + 1, day=1))
                          for m in months])
        return cls(idx, idx.asi8, m1["low"].to_numpy(float), m1["high"].to_numpy(float),
                   rates[np.searchsorted(months, month)])

    def at(self, when) -> int:
        return int(np.searchsorted(self.t, pd.Timestamp(when).as_unit("ns").value, "left"))


def trade_minutes(story: dict, book: MinuteBook) -> dict:
    """One HEDGED trade minute by minute, every leg at 0.01 lot, in rupees: the legs already closed at their recorded
    money (after their Rs 40), plus every open leg at that minute's worse price for the whole trade (all legs at the
    bid low, or all at the bid high, whichever leaves less), each open leg's Rs 40 counted from its open; the rupee
    rate of that minute's month (L20-notes 2e, 3b). From the first leg's open minute to the last leg's close minute
    (a trade still open: to the last bar). Returns a0 (its first minute in the book), inr, inr0 (the same before any
    cost: every leg opened so far gets its Rs 40 back), n_open, still_open. A closed trade's legs must add up to it."""
    legs = story["legs"]
    t = book.t
    still_open = any(not leg["closed"] for leg in legs)
    a0 = min(book.at(leg["opened"]) for leg in legs)
    b0 = len(t) if still_open else max(book.at(leg["closed"]) for leg in legs) + 1
    n = b0 - a0
    pos, base, n_open, done, opened = (np.zeros(n + 1) for _ in range(5))
    for leg in legs:
        d = 1.0 if leg["direction"] == "buy" else -1.0
        a = book.at(leg["opened"]) - a0
        b = book.at(leg["closed"]) - a0 if leg["closed"] else n
        pos[a] += d
        pos[b] -= d
        base[a] -= d * leg["open_px"]
        base[b] += d * leg["open_px"]
        n_open[a] += 1
        n_open[b] -= 1
        opened[a] += 1
        if leg["closed"]:
            done[b] += leg["net_inr"]
    pos, base, n_open, done, opened = (np.cumsum(x)[:n] for x in (pos, base, n_open, done, opened))
    usd = np.minimum(pos * book.lo[a0:b0] + base, pos * book.hi[a0:b0] + base)
    inr = done + usd * book.rate[a0:b0] - config.COST_INR_PER_POSITION * n_open
    if not still_open and abs(inr[-1] - story["net_inr"]) > 0.01 + 0.005 * len(legs):     # legs are kept to paise
        raise AssertionError(f"{story['signal']}: the legs add up to {inr[-1]:.2f}, the trade says "
                             f"{story['net_inr']:.2f}")
    return {"a0": a0, "b0": b0, "inr": inr, "inr0": inr + config.COST_INR_PER_POSITION * opened,
            "n_open": n_open, "still_open": still_open}


def hedged_dips(stories: list[dict], m1: pd.DataFrame | MinuteBook, needs: pd.DataFrame,
                account: float = VERSION_1.account_size) -> pd.DataFrame:
    """The lowest point inside each HEDGED trade, every leg at 0.01 lot, in rupees - minute by minute (trade_minutes).

    One row per trade: when it opened and ended, its positions, its money at the end, its lowest point and when, the
    money needed for 0.01 lot the week it opened, the lowest point as a multiple of that money, and what the dip is at
    the plan's lots from a full account (account / money needed, whole steps): every leg would be that many steps."""
    book = m1 if isinstance(m1, MinuteBook) else MinuteBook.build(m1)
    idx, t = book.idx, book.t
    rows = []
    for s in stories:
        got = trade_minutes(s, book)
        a0, b0, inr, still_open = got["a0"], got["b0"], got["inr"], got["still_open"]
        legs = s["legs"]
        j = int(np.argmin(inr))
        rows.append({"signal": s["signal"], "direction": legs[0]["direction"], "open": idx[a0],
                     "ended": None if still_open else idx[b0 - 1], "still_open": still_open,
                     "days": (t[b0 - 1] - t[a0]) / DAY_NS, "positions": len(legs),
                     "most_open_at_once": int(got["n_open"].max()),
                     "net_inr": None if still_open else float(s["net_inr"]),
                     "low_inr": float(inr[j]), "low_when": idx[a0 + j],
                     "days_below_zero": float((inr < 0).sum() / 1440)})
    out, dropped = attach_need(pd.DataFrame(rows), needs)
    if dropped:
        raise AssertionError("a hedged trade opened before the first Monday number")
    out["low_in_needs"] = -out["low_inr"] / out["need_inr"]
    out["plan_steps"] = np.floor(account / out["need_inr"] + 1e-9)
    out["dip_at_plan_lots"] = out["plan_steps"] * out["low_inr"]
    out["account_at_low"] = account + out["dip_at_plan_lots"]
    return out


# ================================================================================================================
# the command
# ================================================================================================================

def main(argv=None) -> int:
    import argparse
    import sys

    from machine.money import report

    ap = argparse.ArgumentParser(prog="python -m machine.money.plan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=RESHUFFLES, help="how many reshuffles of whole months (1,000)")
    ap.add_argument("--no-write", action="store_true", help="a look only: no report, no pictures, no trial file")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = report.run(n_runs=args.runs, write=not args.no_write)
    print(out["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
