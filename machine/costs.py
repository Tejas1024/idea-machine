"""Money in rupees (L2): what one position makes after Tejas's cost, plus two labelled extra numbers.

Tejas's decisions — the ONLY cost in the main number:
  - 0.01 lot = 1 ounce: a $1 move is $1 at 0.01 lot.
  - Rs 40 per position per 0.01 lot, ALL-IN: Rs 20 in + Rs 20 out, and it already includes the spread
    (LEARNED #54, #115). "Per one trade open and close, it will cost forty rupees."
  - No swap on his account (#54). Lots only in 0.01 steps (#116). Never "Rs 1,000 a trade" (#105).
Prices are on the BID scale (the price data is bid; the tester records fills on the bid scale). The Rs 40 REPLACES
the spread — it is never added on top of it (likhith.md B14; the old manager.py did the same).

Two extra numbers, clearly labelled, on the same line as the main number and never mixed into it:
  "with the broker's swap"  = the main number + the swap the broker would charge (data/swap.csv, read from MT5)
  "with the real spread"    = the price move minus the data's real spread, INSTEAD of the Rs 40
Dollars become rupees at the RBI/FBIL monthly average (data/usd_inr.csv) of the month the position CLOSED.
Times are MT5 SERVER time without a time zone, like load_bars' index. The broker charges swap at server midnight.
What each cost is, its value and its source: machine/costs.md (or  python -m machine.costs).
"""
from __future__ import annotations

import functools
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import pandas as pd

from machine import config
from machine.guards import check_dates

COST_WORDS = "Tejas, LEARNED #54 and #115: Rs 20 in + Rs 20 out per 0.01 lot, all-in (it includes the spread)"
LOT_WORDS = 'Tejas, LEARNED #116: "It only round off like 0.01 or 0.02 or 0.03"'
MT5_DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")  # swap_rollover3days
SWAP_POINTS, SWAP_DISABLED = 1, 0                         # MT5 swap_mode values this file knows how to price


class CostError(ValueError):
    """A money input that would give a wrong number."""


# ================================================================================================================
# lots
# ================================================================================================================

def round_lot(x: float) -> float:
    """The nearest 0.01 step, half going up (his "round off", #116): 0.014 → 0.01, 0.015 → 0.02, 0.054 → 0.05.
    Below half a step it is 0.0 — no position; the caller decides whether to skip or use 0.01."""
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise CostError(f"round_lot({x!r}): a lot size must be a number, 0 or more")
    step = Decimal(str(config.LOT_STEP))
    return float((Decimal(str(x)) / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step)


def lot_steps(lots: float) -> int:
    """How many 0.01 steps: 0.01 → 1, 0.02 → 2. Refuses 0, negative sizes, and anything between steps (0.015)."""
    n = round(lots / config.LOT_STEP) if isinstance(lots, (int, float)) and math.isfinite(lots) else 0
    if n < 1 or abs(lots - n * config.LOT_STEP) > 1e-9:
        raise CostError(f"lots={lots!r} is not 0.01, 0.02, 0.03 ... {LOT_WORDS}. Use round_lot() first.")
    return n


# ================================================================================================================
# rupees per dollar
# ================================================================================================================

@functools.lru_cache(maxsize=4)
def _rates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, dtype={"month": str, "source": str, "edition": str})   # line 1 = sources
    return df.set_index("month")


def usd_inr_row(date: Any) -> dict:
    """The month's row of data/usd_inr.csv: {month, usd_inr, source, edition}. Refuses the sealed years."""
    t = pd.Timestamp(date)
    check_dates(t)
    month = f"{t.year:04d}-{t.month:02d}"
    df = _rates(str(config.USD_INR_CSV))
    if month not in df.index:
        raise CostError(f"no rupee rate for {month}: {config.USD_INR_CSV.name} has {df.index[0]} to {df.index[-1]}")
    r = df.loc[month]
    return {"month": month, "usd_inr": float(r.usd_inr), "source": r.source, "edition": r.edition}


def usd_inr(date: Any) -> float:
    """Rupees per US dollar in the month of `date`: the RBI reference rate until June 2018, both in July 2018 (RBI
    1–9 July, FBIL from 10 July), the FBIL reference rate after — monthly averages (data/usd_inr.csv)."""
    return usd_inr_row(date)["usd_inr"]


# ================================================================================================================
# one position
# ================================================================================================================

@dataclass(frozen=True)
class PnL:
    """One position (one leg). Rupees are floats; round only when printing."""
    direction: str                  # "buy" or "sell"
    entry_px: float                 # bid scale
    exit_px: float
    entry_ts: pd.Timestamp          # server time
    exit_ts: pd.Timestamp
    lots: float
    gross_usd: float                # the price move in gold dollars
    gross_inr: float
    cost_inr: float                 # Rs 40 x (lots / 0.01)
    net_inr: float                  # gross_inr - cost_inr: THE number
    usd_inr: float                  # the rate used
    settings: dict = field(default_factory=dict)
    entry_spread_pts: float | None = None   # the data's spread at the entry minute (points) — extra line only
    exit_spread_pts: float | None = None    # at the exit minute

    @property
    def ounces(self) -> float:
        return self.lots * config.OUNCES_PER_LOT


def _direction(d: Any) -> tuple[str, int]:
    word = {"buy": "buy", "long": "buy", "1": "buy", "sell": "sell", "short": "sell", "-1": "sell"}.get(
        str(d).strip().lower())
    if word is None:
        raise CostError(f"direction={d!r}: use 'buy' or 'sell'")
    return word, (1 if word == "buy" else -1)


def _server_time(t: Any, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        raise CostError(f"{name}={t!r} has a time zone; give MT5 server time without one (like load_bars' index) — "
                        "the swap night is the server's midnight")
    return ts


def _price(p: Any, name: str) -> float:
    if not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
        raise CostError(f"{name}={p!r}: a price must be a number above 0 (a 0.00 price once came from a dead feed)")
    return float(p)


def pnl(direction: Any, entry_px: float, exit_px: float, entry_ts: Any, exit_ts: Any, lots: float, *,
        usd_inr_rate: float | None = None, entry_spread_pts: float | None = None,
        exit_spread_pts: float | None = None) -> PnL:
    """The money of one position, after Tejas's Rs 40 all-in cost.

    gross_usd = (exit - entry) x ounces, the other way round for a sell; ounces = lots x 100 (0.01 lot = 1 ounce)
    gross_inr = gross_usd x rupees per dollar (the month the position closed; usd_inr_rate= overrides it)
    cost_inr  = 40 x (lots / 0.01)
    net_inr   = gross_inr - cost_inr
    entry_spread_pts / exit_spread_pts (the data's spread, in points) are kept only for the "real spread" extra line.
    """
    word, sign = _direction(direction)
    steps = lot_steps(lots)
    t0, t1 = _server_time(entry_ts, "entry_ts"), _server_time(exit_ts, "exit_ts")
    if t1 < t0:
        raise CostError(f"the exit {t1} is before the entry {t0}")
    check_dates([t0, t1])
    p0, p1 = _price(entry_px, "entry_px"), _price(exit_px, "exit_px")

    ounces = steps * (config.LOT_STEP * config.OUNCES_PER_LOT)
    gross_usd = sign * (p1 - p0) * ounces
    if usd_inr_rate is None:
        row = usd_inr_row(t1)
        rate, rate_source = row["usd_inr"], f"{row['source']}, monthly average {row['month']} ({config.USD_INR_CSV.name})"
    else:
        if not isinstance(usd_inr_rate, (int, float)) or not math.isfinite(usd_inr_rate) or usd_inr_rate <= 0:
            raise CostError(f"usd_inr_rate={usd_inr_rate!r} must be above 0")
        rate, rate_source = float(usd_inr_rate), "given by the caller"
    gross_inr = gross_usd * rate
    cost_inr = float(config.COST_INR_PER_POSITION * steps)
    settings = {
        "lots": round(steps * config.LOT_STEP, 2),
        "ounces": ounces,
        "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
        "swap": "none: his account has no swap (LEARNED #54)",
        "usd_inr": rate,
        "usd_inr_source": rate_source,
        "prices": "bid scale",
        "clock": "MT5 server time",
    }
    return PnL(word, p0, p1, t0, t1, settings["lots"], gross_usd, gross_inr, cost_inr, gross_inr - cost_inr, rate,
               settings, entry_spread_pts, exit_spread_pts)


# ================================================================================================================
# the two extra lines
# ================================================================================================================

@functools.lru_cache(maxsize=4)
def _swap_rows(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"read_on": str})


def broker_swap() -> dict:
    """The newest reading in data/swap.csv (python -m machine.data.swap_read adds one)."""
    if not config.SWAP_CSV.is_file():
        raise CostError(f"{config.SWAP_CSV.name} is missing: run  python -m machine.data.swap_read  on the MT5 PC")
    df = _swap_rows(str(config.SWAP_CSV))
    return df.sort_values("read_on", kind="stable").iloc[-1].to_dict()


def swap_nights(entry_ts: Any, exit_ts: Any, triple_day: int = 3) -> tuple[int, int]:
    """(nights held, nights the broker charges) for ONE position.

    A night = a server midnight passed while the position is open (entry < midnight < exit).
    The broker charges for the trading day that the midnight ends: Monday to Friday once each, the triple day 3 times
    (MT5 swap_rollover3days; 3 = Wednesday — it pays for the weekend), Saturday and Sunday nothing.
    Monday 10:00 → Thursday 10:00: held 3 (Mon, Tue, Wed nights), charged 1 + 1 + 3 = 5.
    Friday 10:00 → Monday 10:00: held 3 (Fri, Sat, Sun nights), charged 1.
    """
    t0, t1 = _server_time(entry_ts, "entry_ts"), _server_time(exit_ts, "exit_ts")
    held = charged = 0
    for midnight in pd.date_range(t0.normalize() + pd.Timedelta(days=1), t1, freq="D"):
        if midnight >= t1:
            break
        day = (midnight - pd.Timedelta(days=1)).weekday()    # the day that ended: pandas Monday = 0
        mt5_day = (day + 1) % 7                               # MT5: Sunday = 0, Monday = 1 ... Saturday = 6
        held += 1
        if mt5_day not in (0, 6):
            charged += 3 if mt5_day == triple_day else 1
    return held, charged


def real_spread_points(pts: float | None) -> float | None:
    """The data's spread in points; a spread of 0 (about 1.5% of 1-min bars) counts as config.MIN_SPREAD_POINTS."""
    if pts is None:
        return None
    if not isinstance(pts, (int, float)) or math.isnan(pts) or pts <= 0:
        return float(config.MIN_SPREAD_POINTS)
    return float(pts)


@dataclass(frozen=True)
class Extras:
    """The two extra numbers for one or more positions (a hedged trade = several legs)."""
    net_inr: float                     # the main number, all legs
    swap_inr: float                    # the swap alone (minus = paid)
    with_swap_inr: float               # net_inr + swap_inr
    nights_held: int                   # leg-nights: each position counts each night it was open
    nights_charged: int                # what the broker counts: triple day x3, Saturday and Sunday 0
    spread_inr: float | None           # the real spread paid (None when a spread was not given)
    with_spread_inr: float | None      # gross - real spread, INSTEAD of Rs 40
    swap_reading: str                  # which reading: server, account kind, date read
    text: str


def extra_lines(legs: PnL | Iterable[PnL], swap: dict | None = None) -> Extras:
    """The swap line and the real-spread line for one position or all legs of a trade.

    Swap (MT5 swap mode POINTS): points per lot per night x point x 100 oz x lots x nights charged, then rupees.
    Real spread: a buy pays the spread when it opens (buys at the ask), a sell when it closes (buys back at the ask);
    with_spread_inr = the price move in rupees - that spread in rupees, and NO Rs 40.
    swap= overrides data/swap.csv (tests use a fixed reading).
    """
    legs = [legs] if isinstance(legs, PnL) else list(legs)
    if not legs:
        raise CostError("extra_lines needs at least one position")
    sw = broker_swap() if swap is None else swap
    mode = int(sw["swap_mode"])
    if mode not in (SWAP_POINTS, SWAP_DISABLED):
        raise CostError(f"swap mode {mode} ({sw.get('swap_mode_name', '?')}) is not priced here; only POINTS or "
                        "DISABLED — add it to machine/costs.py before using this reading")
    triple = int(sw["swap_rollover3days"])

    swap_inr = spread_inr = 0.0
    held = charged = 0
    spread_known = True
    for leg in legs:
        h, c = swap_nights(leg.entry_ts, leg.exit_ts, triple)
        held, charged = held + h, charged + c
        if mode == SWAP_POINTS:
            pts = float(sw["swap_long"] if leg.direction == "buy" else sw["swap_short"])
            swap_inr += pts * float(sw["point"]) * float(sw["contract_size"]) * leg.lots * c * leg.usd_inr
        sp = real_spread_points(leg.entry_spread_pts if leg.direction == "buy" else leg.exit_spread_pts)
        if sp is None:
            spread_known = False
        else:
            spread_inr += sp * config.POINT * leg.ounces * leg.usd_inr

    net = sum(leg.net_inr for leg in legs)
    gross = sum(leg.gross_inr for leg in legs)
    with_spread = gross - spread_inr if spread_known else None
    nights = f"nights held {held}" + (f" · the broker counts {charged}" if charged != held else "")
    spread_text = inr(with_spread) if spread_known else "not given"
    text = f"(with the broker's swap: {inr(net + swap_inr)} · {nights})  (with the real spread: {spread_text})"
    reading = f"{sw.get('server', '?')} ({sw.get('account', '?')}), read {sw.get('read_on', '?')}"
    return Extras(net, swap_inr, net + swap_inr, held, charged, -spread_inr if spread_known else None,
                  with_spread, reading, text)


def inr(x: float) -> str:
    """Whole rupees with a sign: -₹1,234 · ₹790."""
    return f"-₹{-x:,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def usd(x: float, decimals: int = 2) -> str:
    """Gold dollars with a sign: +$10.00 · -$0.126."""
    return f"-${-x:,.{decimals}f}" if round(x, decimals) < 0 else f"+${x:,.{decimals}f}"


def result_line(legs: PnL | Iterable[PnL], swap: dict | None = None) -> str:
    """The main number and its settings, with the two extra numbers on the SAME line."""
    legs = [legs] if isinstance(legs, PnL) else list(legs)
    ex = extra_lines(legs, swap)
    lots = round(sum(leg.lots for leg in legs), 2)
    who = (f"{legs[0].direction.upper()} {legs[0].lots:.2f} lot" if len(legs) == 1
           else f"{len(legs)} positions ({lots:.2f} lot in all)")
    lo, hi = min(leg.usd_inr for leg in legs), max(leg.usd_inr for leg in legs)
    rate = f"$1 = ₹{lo:.2f}" if hi - lo < 0.005 else f"$1 = ₹{lo:.2f} to ₹{hi:.2f}"
    gross_usd = sum(leg.gross_usd for leg in legs)
    gross_inr = sum(leg.gross_inr for leg in legs)
    cost = sum(leg.cost_inr for leg in legs)
    return (f"{who} · gold {usd(gross_usd)} = {inr(gross_inr)} · cost {inr(-cost)} (₹40 per 0.01 lot, all-in) · "
            f"after cost {inr(ex.net_inr)} ({rate})  {ex.text}")


# ================================================================================================================
# the cost table (machine/costs.md shows the same rows; tests/test_costs.py checks they agree)
# ================================================================================================================

def cost_table() -> list[tuple[str, str, str]]:
    rates = _rates(str(config.USD_INR_CSV))
    lo, hi = rates.usd_inr.idxmin(), rates.usd_inr.idxmax()
    sw = broker_swap()
    per_night = {k: float(sw[k]) * float(sw["point"]) * float(sw["contract_size"]) * config.LOT_STEP
                 for k in ("swap_long", "swap_short")}
    return [
        ("Cost of one position (open + close)",
         f"₹{config.COST_INR_PER_POSITION} per 0.01 lot (₹20 in + ₹20 out), all-in: it includes the spread",
         "Tejas, LEARNED #54 (14 Sep 2026), #115 (20 Sep 2026)"),
        ("Swap on his account", "none", "Tejas, LEARNED #54"),
        ("Lot sizes", "0.01, 0.02, 0.03 ... only (round_lot)", "Tejas, LEARNED #116"),
        ("Gold in one lot", f"{config.OUNCES_PER_LOT} ounces: 0.01 lot = 1 ounce, so a $1 move = $1",
         "MT5 symbol_info trade_contract_size, MetaQuotes-Demo, 28 Sep 2026"),
        ("Rupees per dollar",
         f"monthly average of the month the position closed: ₹{rates.usd_inr[lo]:.2f} ({lo}) to "
         f"₹{rates.usd_inr[hi]:.2f} ({hi})",
         "RBI reference rate to Jun 2018, FBIL from 10 Jul 2018 (RBI Handbook table); data/usd_inr.csv"),
        ("Broker's swap (extra line only)",
         f"buy {float(sw['swap_long']):g} points a lot a night, sell {float(sw['swap_short']):g}; "
         f"{MT5_DAYS[int(sw['swap_rollover3days'])]} night counts 3, Saturday and Sunday nights 0; at 0.01 lot: "
         f"buy {usd(per_night['swap_long'], 3)} a night, sell {usd(per_night['swap_short'], 3)}",
         f"MT5 symbol_info on {sw['server']} ({sw['account']}), read {sw['read_on']}; data/swap.csv"),
        ("Real spread (extra line only)",
         "the data's spread at the minute of the fill, INSTEAD of ₹40: a buy pays it when it opens, a sell when it "
         f"closes; a spread of 0 counts as {config.MIN_SPREAD_POINTS:g} points",
         "the spread column of data/gold_m1.parquet; config.MIN_SPREAD_POINTS (Claude's pick)"),
    ]


def _main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("COSTS (L2) — cost · value · source\n")
    for cost, value, source in cost_table():
        print(f"  {cost}\n      {value}\n      source: {source}")
    print("\nEXAMPLES (0.01 lot; real rupee rate of the month; the swap and spread lines are extras only)\n")
    buy = pnl("buy", 2000.0, 2010.0, "2023-11-13 10:00", "2023-11-16 10:00", 0.01,
              entry_spread_pts=19, exit_spread_pts=19)
    sell = pnl("sell", 1300.0, 1290.0, "2018-05-11 10:00", "2018-05-14 10:00", 0.01,
               entry_spread_pts=19, exit_spread_pts=0)
    pair = [pnl("buy", 1800.0, 1797.0, "2021-06-07 10:00", "2021-06-10 10:00", 0.01, entry_spread_pts=20,
                exit_spread_pts=20),
            pnl("sell", 1797.0, 1797.0, "2021-06-08 10:00", "2021-06-10 10:00", 0.01, entry_spread_pts=20,
                exit_spread_pts=20)]
    for name, legs in (("buy, Monday to Thursday", buy), ("sell, Friday to Monday", sell),
                       ("a hedged pair, 2 positions", pair)):
        print(f"  {name}:\n    {result_line(legs)}")


if __name__ == "__main__":
    _main()
