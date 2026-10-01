"""L2 — cost and money in rupees (machine/costs.py). Runs everywhere: data/usd_inr.csv and data/swap.csv are in git.

Every money test shows its hand arithmetic in the docstring. The rate is given (usd_inr_rate=83) where the test is
about arithmetic, and the swap reading is fixed (SWAP) where the test is about swap, so a new reading in
data/swap.csv cannot change these numbers.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from machine import config, costs
from machine.costs import CostError, extra_lines, lot_steps, pnl, result_line, round_lot, swap_nights, usd_inr
from machine.guards import SealedYearsError

# the reading of 28 Sep 2026 (MetaQuotes-Demo): points per lot per night, Wednesday = 3
SWAP = {"read_on": "2026-09-28", "server": "MetaQuotes-Demo", "account": "demo", "swap_mode": 1,
        "swap_mode_name": "POINTS", "swap_long": -12.6, "swap_short": -4.6, "swap_rollover3days": 3,
        "point": 0.01, "contract_size": 100.0}
MON, THU = "2023-11-13 10:00", "2023-11-16 10:00"     # a Monday and the Thursday after


# ================================================================================================================
# the main number — hand arithmetic
# ================================================================================================================

def test_buy_by_hand():
    """Buy at 2000, sell at 2010, 0.01 lot = 1 ounce, ₹83 a dollar:
    gold dollars = (2010 - 2000) x 1 oz = $10;  rupees = 10 x 83 = ₹830;  cost = ₹40;  after cost = 830 - 40 = ₹790."""
    p = pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0)
    assert p.gross_usd == pytest.approx(10.0)
    assert p.gross_inr == pytest.approx(830.0)
    assert p.cost_inr == 40.0
    assert p.net_inr == pytest.approx(790.0)
    assert p.settings["lots"] == 0.01 and p.settings["usd_inr"] == 83.0
    assert "LEARNED #54" in p.settings["cost"] and "given by the caller" in p.settings["usd_inr_source"]


def test_sell_by_hand():
    """A sell makes money when price falls. Sell at 2010, buy back at 2000, 0.01 lot, ₹83:
    (2010 - 2000) x 1 oz = $10 → ₹830 → after ₹40 = ₹790.
    The same sell when price RISES 2000 → 2010: -$10 → -₹830 → after ₹40 = -₹870 (the cost is paid on a loss too)."""
    win = pnl("sell", 2010.0, 2000.0, MON, THU, 0.01, usd_inr_rate=83.0)
    assert (win.gross_usd, win.net_inr) == (pytest.approx(10.0), pytest.approx(790.0))
    loss = pnl("sell", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0)
    assert (loss.gross_usd, loss.gross_inr, loss.net_inr) == (pytest.approx(-10.0), pytest.approx(-830.0),
                                                              pytest.approx(-870.0))


def test_two_lots_cost_80():
    """0.02 lot = 2 ounces. Buy 2000 → 2010, ₹83: $10 x 2 = $20 → ₹1,660; cost = 40 x (0.02 / 0.01) = ₹80;
    after cost = 1,660 - 80 = ₹1,580."""
    p = pnl("buy", 2000.0, 2010.0, MON, THU, 0.02, usd_inr_rate=83.0)
    assert p.cost_inr == 80.0
    assert p.gross_usd == pytest.approx(20.0)
    assert p.net_inr == pytest.approx(1580.0)
    assert pnl("buy", 2000.0, 2010.0, MON, THU, 0.10, usd_inr_rate=83.0).cost_inr == 400.0   # 0.10 lot = 10 x ₹40


def test_the_rate_is_the_month_the_position_closed():
    """No rate given → data/usd_inr.csv, the month of the EXIT. Nov 2023 = ₹83.2980 (FBIL):
    $10 x 83.2980 = ₹832.98; after ₹40 = ₹792.98."""
    p = pnl("buy", 2000.0, 2010.0, "2023-10-31 10:00", "2023-11-01 10:00", 0.01)
    assert p.usd_inr == pytest.approx(83.2980)
    assert p.net_inr == pytest.approx(792.98)
    assert "FBIL" in p.settings["usd_inr_source"] and "2023-11" in p.settings["usd_inr_source"]


# ================================================================================================================
# lots (#116)
# ================================================================================================================

@pytest.mark.parametrize("x, want", [(0.01, 0.01), (0.014, 0.01), (0.015, 0.02), (0.054, 0.05), (0.012, 0.01),
                                     (0.004, 0.0), (0.005, 0.01), (0.1, 0.1), (1, 1.0), (2.345, 2.35), (0, 0.0)])
def test_round_lot(x, want):
    assert round_lot(x) == want


def test_lots_only_in_steps():
    assert lot_steps(0.03) == 3 and lot_steps(1.0) == 100
    for bad in (0.015, 0.012, 0.0, -0.01, float("nan")):
        with pytest.raises(CostError, match="#116"):
            lot_steps(bad)
    with pytest.raises(CostError, match="#116"):
        pnl("buy", 2000.0, 2010.0, MON, THU, 0.054, usd_inr_rate=83.0)
    with pytest.raises(CostError):
        round_lot(-0.01)


# ================================================================================================================
# swap nights — "a 3-night hold shows leg-nights"
# ================================================================================================================

def test_three_night_hold_shows_leg_nights():
    """Buy 0.01 lot Monday 10:00 → Thursday 10:00: 3 nights held (Mon, Tue, Wed).
    The broker counts Mon 1 + Tue 1 + Wed 3 = 5.  Swap = -12.6 points x $0.01 x 1 oz x 5 = -$0.63 → x ₹83 = -₹52.29.
    Main ₹790 → with the broker's swap ₹737.71.
    A hedged pair — a buy and a sell, both open the same 3 nights — shows leg-nights 6 (each leg counts its nights):
    the broker counts 10; swap = (-12.6 - 4.6) x $0.01 x 1 oz x 5 = -$0.86 → -₹71.38."""
    buy = pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0)
    ex = extra_lines(buy, swap=SWAP)
    assert (ex.nights_held, ex.nights_charged) == (3, 5)
    assert ex.swap_inr == pytest.approx(-52.29)
    assert ex.with_swap_inr == pytest.approx(737.71)
    assert "nights held 3" in ex.text and "the broker counts 5" in ex.text

    sell = pnl("sell", 2010.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0)
    pair = extra_lines([buy, sell], swap=SWAP)
    assert (pair.nights_held, pair.nights_charged) == (6, 10)
    assert pair.swap_inr == pytest.approx(-71.38)
    assert "nights held 6" in pair.text


@pytest.mark.parametrize("t0, t1, held, charged, why", [
    ("2023-11-17 10:00", "2023-11-20 10:00", 3, 1, "Friday → Monday: Sat and Sun nights are free"),
    ("2023-11-15 10:00", "2023-11-16 10:00", 1, 3, "Wednesday night alone counts 3"),
    ("2023-11-13 09:00", "2023-11-13 22:00", 0, 0, "closed the same day: no night"),
    ("2023-11-13 10:00", "2023-11-14 00:00", 0, 0, "closed exactly at midnight: not held past it"),
    ("2023-11-14 00:00", "2023-11-14 10:00", 0, 0, "opened exactly at midnight: that midnight is not passed"),
    ("2023-11-13 10:00", "2023-11-20 10:00", 7, 7, "one full week: Mon Tue Wed(3) Thu Fri = 7, weekend 0"),
])
def test_swap_nights(t0, t1, held, charged, why):
    assert swap_nights(t0, t1) == (held, charged), why


def test_triple_day_comes_from_the_broker():
    """Some brokers charge the triple on Friday (swap_rollover3days = 5). Monday → Thursday is then 1 + 1 + 1 = 3."""
    assert swap_nights(MON, THU, triple_day=5) == (3, 3)
    assert swap_nights("2023-11-17 10:00", "2023-11-20 10:00", triple_day=5) == (3, 3)


def test_sell_swap_and_unknown_swap_mode():
    """Sell 0.01 lot Friday → Monday: counted 1 night. -4.6 x $0.01 x 1 oz x 1 = -$0.046 → x ₹83 = -₹3.818."""
    s = pnl("sell", 2010.0, 2000.0, "2023-11-17 10:00", "2023-11-20 10:00", 0.01, usd_inr_rate=83.0)
    assert extra_lines(s, swap=SWAP).swap_inr == pytest.approx(-3.818)
    assert extra_lines(s, swap={**SWAP, "swap_mode": 0}).swap_inr == 0.0          # swap switched off
    with pytest.raises(CostError, match="not priced here"):
        extra_lines(s, swap={**SWAP, "swap_mode": 5, "swap_mode_name": "INTEREST_CURRENT"})


# ================================================================================================================
# the real spread — INSTEAD of ₹40, never on top (likhith.md B14)
# ================================================================================================================

def test_real_spread_line_by_hand():
    """Buy 2000 → 2010, 0.01 lot, ₹83. A buy opens at the ask, so it pays the ENTRY spread: 19 points = $0.19.
    With the real spread = $10 - $0.19 = $9.81 → x 83 = ₹814.23 — and NO ₹40.
    A sell closes at the ask, so it pays the EXIT spread: 25 points → $10 - $0.25 = $9.75 → ₹809.25.
    A spread of 0 counts as 14 points: $10 - $0.14 = $9.86 → ₹818.38."""
    buy = pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0, entry_spread_pts=19, exit_spread_pts=99)
    assert extra_lines(buy, swap=SWAP).with_spread_inr == pytest.approx(814.23)
    sell = pnl("sell", 2010.0, 2000.0, MON, THU, 0.01, usd_inr_rate=83.0, entry_spread_pts=99, exit_spread_pts=25)
    assert extra_lines(sell, swap=SWAP).with_spread_inr == pytest.approx(809.25)
    zero = pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0, entry_spread_pts=0, exit_spread_pts=0)
    assert config.MIN_SPREAD_POINTS == 14.0
    assert extra_lines(zero, swap=SWAP).with_spread_inr == pytest.approx(818.38)
    # the main number never changes with the spread
    assert buy.net_inr == sell.net_inr == zero.net_inr == pytest.approx(790.0)


def test_real_spread_not_given():
    p = pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0)
    ex = extra_lines(p, swap=SWAP)
    assert ex.with_spread_inr is None and "with the real spread: not given" in ex.text


def test_everything_on_the_same_line():
    """The main number, its settings and both extra numbers — one line, clearly labelled."""
    p = pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0, entry_spread_pts=19)
    line = result_line(p, swap=SWAP)
    print("\n" + line)
    assert "\n" not in line
    assert re.search(r"after cost ₹790 \(\$1 = ₹83\.00\)", line)
    assert "(with the broker's swap: ₹738 · nights held 3 · the broker counts 5)" in line
    assert "(with the real spread: ₹814)" in line
    assert "₹40 per 0.01 lot, all-in" in line and "BUY 0.01 lot" in line and "+$10.00" in line
    assert "₹1,000" not in line                                                    # #105
    loss = pnl("sell", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=83.0)
    assert "after cost -₹870" in result_line(loss, swap=SWAP)


# ================================================================================================================
# refusals
# ================================================================================================================

def test_refusals():
    ok = dict(entry_ts=MON, exit_ts=THU, lots=0.01, usd_inr_rate=83.0)
    with pytest.raises(SealedYearsError):
        pnl("buy", 2000.0, 2010.0, "2023-12-29 10:00", "2024-01-02 10:00", 0.01)        # #96
    with pytest.raises(CostError, match="before the entry"):
        pnl("buy", 2000.0, 2010.0, THU, MON, 0.01, usd_inr_rate=83.0)
    with pytest.raises(CostError, match="time zone"):
        pnl("buy", 2000.0, 2010.0, pd.Timestamp(MON, tz="UTC"), pd.Timestamp(THU, tz="UTC"), 0.01, usd_inr_rate=83.0)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(CostError, match="price"):
            pnl("buy", bad, 2010.0, **ok)
    with pytest.raises(CostError, match="direction"):
        pnl("up", 2000.0, 2010.0, **ok)
    assert pnl("long", 2000.0, 2010.0, **ok).direction == "buy" and pnl(-1, 2000.0, 2010.0, **ok).direction == "sell"
    with pytest.raises(CostError):
        pnl("buy", 2000.0, 2010.0, MON, THU, 0.01, usd_inr_rate=0.0)


# ================================================================================================================
# data/usd_inr.csv — RBI until July 2018, FBIL after
# ================================================================================================================

# The RBI's calendar-year average, ₹ per $ (Handbook Table 132, 2025-26 edition — a different table from the monthly one)
RBI_YEARLY = {2016: 67.1953, 2017: 65.1216, 2018: 68.3895, 2019: 70.4203, 2020: 74.0996, 2021: 73.9180,
              2022: 78.6045, 2023: 82.5993}


def _rates() -> pd.DataFrame:
    return pd.read_csv(config.USD_INR_CSV, skiprows=1, dtype={"month": str, "edition": str})


def test_usd_inr_file():
    first = config.USD_INR_CSV.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# sources:") and "rbi.org.in/Scripts/PublicationsView.aspx?id=" in first
    d = _rates()
    months = [str(p) for p in pd.period_range("2016-01", "2023-12", freq="M")]
    assert d.month.tolist() == months                                              # 96 months, none missing
    assert d.usd_inr.between(60, 90).all()
    assert (d.usd_inr.pct_change().abs().dropna() < 0.05).all()                    # biggest real move: 4.0%, Mar 2020
    assert set(d.source[d.month < "2018-07"]) == {"RBI reference rate"}
    assert "FBIL" in d.source[d.month == "2018-07"].item() and "RBI" in d.source[d.month == "2018-07"].item()
    assert set(d.source[d.month > "2018-07"]) == {"FBIL reference rate"}


def test_usd_inr_matches_the_rbi_yearly_average():
    """A wrong column or a month shifted by one cannot give the RBI's own yearly average."""
    d = _rates()
    d["year"] = d.month.str[:4].astype(int)
    for year, mean in d.groupby("year").usd_inr.mean().items():
        assert mean == pytest.approx(RBI_YEARLY[year], abs=0.0001), year


def test_usd_inr_lookup():
    assert usd_inr("2016-01-15") == pytest.approx(67.2523)
    assert usd_inr(pd.Timestamp("2023-12-29 23:56")) == pytest.approx(83.2787)
    assert costs.usd_inr_row("2018-07-20")["source"].startswith("RBI reference rate 1-9 Jul")
    with pytest.raises(SealedYearsError):
        usd_inr("2024-01-02")
    with pytest.raises(CostError, match="2015-06"):
        usd_inr("2015-06-01")


# ================================================================================================================
# data/swap.csv and machine/costs.md
# ================================================================================================================

def test_swap_file():
    sw = costs.broker_swap()
    for k in ("read_on", "server", "account", "swap_mode", "swap_long", "swap_short", "swap_rollover3days", "point",
              "contract_size"):
        assert k in sw, k
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", sw["read_on"])
    assert sw["contract_size"] == config.OUNCES_PER_LOT
    assert 0 <= int(sw["swap_rollover3days"]) <= 6


def test_costs_md_matches_the_code():
    """machine/costs.md shows the same table the code prints (python -m machine.costs)."""
    md = (config.ROOT / "machine" / "costs.md").read_text(encoding="utf-8")
    for cost, value, source in costs.cost_table():
        assert f"| {cost} | {value} | {source} |" in md, cost


@pytest.mark.mt5
def test_swap_read_from_mt5():
    """🪟 Reads the swap settings only (no price, nothing written)."""
    from machine.data import swap_read
    try:
        r = swap_read.read()
    except RuntimeError as e:
        pytest.skip(f"MT5 is installed but did not start: {e}")
    assert set(r) == set(swap_read.COLUMNS)
    assert not {"bid", "ask", "last"} & set(r)
