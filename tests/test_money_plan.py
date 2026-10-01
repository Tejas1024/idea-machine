"""L20 — his money plan (machine/money/plan.py). Made-up trades only: every money test shows its hand arithmetic.

What the words mean (BLOW-UP, REFILL, a double, a cycle, your money) is plan/work/L20-notes.md section 1.
Version 1 everywhere unless a test names a row: Rs 200,000 of your money, Rs 20,000 in the account, Rs 180,000 in
reserve; lots = the account / the money needed for 0.01 lot, rounded down to a whole step.
One test needs the price data (marked data): the Monday numbers on the real bars.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from machine import costs, guards
from machine.guards import SealedYearsError
from machine.money import plan as MP
from machine.money.plan import EXTRA_ROWS, VERSION_1, Plan, TradeSet

MON = pd.Timestamp("2019-01-07 10:00")            # a Monday
ROW = {p.name: p for p in EXTRA_ROWS}


def table(*rows, need=5_000.0) -> pd.DataFrame:
    """rows = (open, close, net Rs at 0.01 lot[, money needed for 0.01 lot[, lowest point inside]]).
    Cost Rs 40, so the price move = net + 40; the lowest point inside = the loss (a losing trade) or -40 (a winner:
    only the cost was ever owed) unless given."""
    out = []
    for r in rows:
        net = float(r[2])
        out.append({"open": pd.Timestamp(r[0]), "close": pd.Timestamp(r[1]), "gross_inr": net + 40.0,
                    "cost_inr": 40.0, "need_inr": float(r[3]) if len(r) > 3 else need,
                    "low_inr": float(r[4]) if len(r) > 4 else min(net, -40.0)})
    return pd.DataFrame(out)


def daily(nets, need=5_000.0, start=MON) -> pd.DataFrame:
    """One trade a day from `start`, 10:00 to 11:00, each with its net at 0.01 lot (a pair = (net, money needed))."""
    rows = []
    for j, n in enumerate(nets):
        o = start + pd.Timedelta(days=j)
        net, nd = (n if isinstance(n, tuple) else (n, need))
        rows.append((o, o + pd.Timedelta(hours=1), net, nd))
    return table(*rows)


def run(frame, plan=VERSION_1):
    ts = TradeSet.build(frame)
    return MP.simulate(ts, MP.real_order(ts), plan, trace=True)


def money(sim) -> tuple[float, float, float, float]:
    """(the account, the reserve, taken out, added back) at the end."""
    return tuple(round(float(x[0]), 6) for x in (sim.account, sim.reserve, sim.out, sim.added))


# ================================================================================================================
# the plan's settings
# ================================================================================================================

def test_version_1_is_his_numbers():
    """Rs 200,000 of your money: 10% = Rs 20,000 in the account (#114, #115), Rs 180,000 in reserve; the profit is
    taken out at a double (x2); the blow-up line is the money needed for 0.01 lot; the refill goes back to Rs 20,000."""
    p = VERSION_1
    assert (p.your_money, p.account_size, p.reserve_size) == (200_000.0, 20_000.0, 180_000.0)
    assert (p.capital, p.sizing, p.withdraw_at, p.refill_to, p.blowup_line, p.cost) == \
        ("2xADR", "chocolate", 2.0, "size", "need", True)
    assert p.change == ""


def test_every_extra_row_changes_exactly_one_thing():
    """'Extra rows with one change each': apart from its name and the words of the change, a row differs from version
    1 in ONE field."""
    base = dataclasses.asdict(VERSION_1)
    assert len(EXTRA_ROWS) == 9 and len({p.name for p in EXTRA_ROWS}) == 9
    for p in EXTRA_ROWS:
        d = dataclasses.asdict(p)
        changed = [k for k in base if k not in ("name", "change") and d[k] != base[k]]
        assert len(changed) == 1, (p.name, changed)
        assert p.change


def test_a_plan_with_a_wrong_word_is_refused():
    for bad in (dict(capital="4xADR"), dict(sizing="average"), dict(refill_to="top"), dict(blowup_line="half"),
                dict(withdraw_at=1.0), dict(account_share=0.0), dict(your_money=0.0)):
        with pytest.raises(ValueError):
            Plan(**bad)


# ================================================================================================================
# lots: the Rs 100-of-chocolate rule (#115), whole steps only (#116)
# ================================================================================================================

def test_lots_are_the_account_divided_by_the_money_needed():
    """His own example (#114): Rs 20,000 in the account and about Rs 7,500 needed for 0.01 lot -> "zero point zero
    two lot": 20,000 / 7,500 = 2.67 -> 2 whole steps = 0.02. A small range buys more: 20,000 / 3,000 = 6.67 -> 0.06.
    Exactly enough for one: 20,000 / 20,000 = 1 -> 0.01. Every trade here ends at Rs 0 after its cost, so the account
    stays Rs 20,000."""
    sim = run(daily([(0, 7_500), (0, 3_000), (0, 20_000)]))
    assert sim.trace["lots"].tolist() == [0.02, 0.06, 0.01]
    assert money(sim) == (20_000.0, 180_000.0, 0.0, 0.0)
    assert MP.lots_text(sim.steps[3]) == "0.01 x1 · 0.02 x1 · 0.06 x1"          # 2019 = year 3 from 2016


def test_lots_follow_the_account_not_the_start():
    """The account, not the first Rs 20,000, is divided: Rs 5,000 needed -> 0.04. A loss of Rs 1,000 at 0.01 lot is
    Rs 4,000 at 0.04 -> the account is Rs 16,000 -> 16,000 / 5,000 = 3.2 -> 0.03 next. (16,000 is above the Rs 5,000
    line: no blow-up.) The second trade loses 3 x 1,000 -> Rs 13,000 -> 0.02."""
    sim = run(daily([-1_000, -1_000, 0]))
    assert sim.trace["lots"].tolist() == [0.04, 0.03, 0.02]
    assert sim.trace["inr"].tolist() == [-4_000.0, -3_000.0, 0.0]
    assert money(sim) == (13_000.0, 180_000.0, 0.0, 0.0)
    assert sim.blowups.sum() == 0


def test_the_cost_grows_with_the_lot():
    """Rs 40 per position per 0.01 lot (#115): a trade whose price did not move loses Rs 40 at 0.01 lot and
    4 x 40 = Rs 160 at 0.04 lot."""
    frame = table((MON, MON + pd.Timedelta(hours=1), -40.0))
    assert run(frame).trace["inr"].tolist() == [-160.0]
    assert run(frame, ROW["row 9"]).trace["inr"].tolist() == [0.0]              # NOT REAL: before the Rs 40


# ================================================================================================================
# a double: the profit is taken out
# ================================================================================================================

def test_a_double_takes_the_profit_out():
    """Rs 20,000, Rs 5,000 needed -> 0.04 lot. The trade makes Rs 5,000 at 0.01 lot = Rs 20,000 at 0.04: the account
    is Rs 40,000 = 2 x 20,000 -> a double. Rs 20,000 is taken out and the account is back at Rs 20,000.
    Opened Monday 10:00, closed Wednesday 10:00 = 2.0 days. Your money = 180,000 + 20,000 + 20,000 = Rs 220,000."""
    sim = run(table((MON, MON + pd.Timedelta(days=2), 5_000.0)))
    assert money(sim) == (20_000.0, 180_000.0, 20_000.0, 0.0)
    assert sim.doubles.sum() == 1 and sim.dbl_days.tolist() == [2.0]
    assert sim.your_money_end[0] == 220_000.0 and sim.result[0] == 20_000.0
    e = sim.events[0]
    assert (e["what"], e["account_before"], e["money_in_out"], e["account_after"]) == \
        ("a double", 40_000.0, -20_000.0, 20_000.0)
    assert abs(MP.when(e["t"]) - (MON + pd.Timedelta(days=2))) < pd.Timedelta(seconds=1)


def test_not_yet_a_double_takes_nothing_out():
    """+Rs 3,750 at 0.01 lot = +15,000 at 0.04 -> Rs 35,000: under 2 x 20,000, so nothing is taken out in version 1.
    Row 1 takes the profit out at +75%: 35,000 = 1.75 x 20,000 -> Rs 15,000 out, the account back at Rs 20,000."""
    frame = daily([3_750])
    assert money(run(frame)) == (35_000.0, 180_000.0, 0.0, 0.0)
    sim = run(frame, ROW["row 1"])
    assert money(sim) == (20_000.0, 180_000.0, 15_000.0, 0.0)
    assert sim.events[0]["what"] == "+75%"


def test_the_days_of_a_double_count_from_the_cycle_start():
    """Three trades, a day apart: +2,500 at 0.01 lot = +10,000 at 0.04 (Rs 30,000, lots now 0.06), then +2,000 x 6 =
    +12,000 -> Rs 42,000 >= 40,000: a double after the second trade, 1 day + 1 hour after the first trade opened
    = 1.0417 days; Rs 22,000 out. The next cycle starts at that close: the third trade (+5,000 x 4 = +20,000) doubles
    it again 1.0 day later (Wednesday 11:00 - Tuesday 11:00)."""
    sim = run(daily([2_500, 2_000, 5_000]))
    assert sim.trace["lots"].tolist() == [0.04, 0.06, 0.04]
    assert sim.doubles.sum() == 2
    assert sim.dbl_days.tolist() == pytest.approx([1 + 1 / 24, 1.0])
    assert money(sim) == (20_000.0, 180_000.0, 42_000.0, 0.0)


def test_never_taking_out_only_starts_a_new_cycle():
    """Row 4: +5,000 at 0.01 lot = +20,000 -> Rs 40,000: a double is counted, nothing is taken out, and the next cycle
    starts from Rs 40,000 - the lots are 40,000 / 5,000 = 0.08. A second +5,000 x 8 = +40,000 -> Rs 80,000 = 2 x 40,000:
    the second double."""
    sim = run(daily([5_000, 5_000]), ROW["row 4"])
    assert sim.trace["lots"].tolist() == [0.04, 0.08]
    assert money(sim) == (80_000.0, 180_000.0, 0.0, 0.0)
    assert sim.doubles.sum() == 2


# ================================================================================================================
# BLOW-UP and REFILL
# ================================================================================================================

def test_blow_up_and_refill_by_hand():
    """Rs 20,000, Rs 5,000 needed -> 0.04. The trade loses Rs 4,000 at 0.01 lot = Rs 16,000 -> the account is Rs 4,000,
    below the Rs 5,000 needed for 0.01 lot: a BLOW-UP. REFILL: 20,000 - 4,000 = Rs 16,000 comes from the reserve
    (180,000 -> 164,000) and the account is Rs 20,000 again. Your money = 164,000 + 20,000 = Rs 184,000
    = 200,000 - 16,000."""
    sim = run(daily([-4_000, 0]))
    assert money(sim) == (20_000.0, 164_000.0, 0.0, 16_000.0)
    assert sim.blowups.sum() == 1 and sim.alive[0]
    assert sim.your_money_end[0] == 184_000.0
    e = sim.events[0]
    assert (e["what"], e["why"], e["account_before"], e["money_in_out"], e["account_after"]) == \
        ("blow-up", "after a trade closed", 4_000.0, 16_000.0, 20_000.0)
    assert sim.trace["lots"].tolist() == [0.04, 0.04]                          # back at full size after the refill


def test_exactly_on_the_line_is_not_a_blow_up():
    """-3,750 at 0.01 lot = -15,000 at 0.04 -> the account is Rs 5,000 = exactly the money needed for 0.01 lot: it can
    still trade 0.01, so no blow-up. One rupee less (-3,750.25 x 4 = -15,001) is one."""
    assert run(daily([-3_750])).blowups.sum() == 0
    assert run(daily([-3_750.25])).blowups.sum() == 1


def test_a_monday_number_that_rises_is_a_blow_up_before_the_trade():
    """Trade 1: -3,500 at 0.01 = -14,000 at 0.04 -> Rs 6,000, above that week's Rs 5,000: no blow-up. Trade 2's week
    needs Rs 7,000 for 0.01 lot: 6,000 < 7,000 -> a blow-up BEFORE it opens; Rs 14,000 comes in; then
    20,000 / 7,000 = 2.86 -> 0.02."""
    sim = run(daily([(-3_500, 5_000), (0, 7_000)]))
    assert sim.blowups.sum() == 1
    assert sim.events[0]["why"] == "before a trade opened"
    assert sim.trace["lots"].tolist() == [0.04, 0.02]
    assert money(sim) == (20_000.0, 166_000.0, 0.0, 14_000.0)


def test_the_refill_reaches_the_money_needed_when_that_is_above_the_account_size():
    """A week that needs Rs 25,000 for 0.01 lot with Rs 20,000 in the account: below the line -> a blow-up, and the
    refill goes up to Rs 25,000 (the account must be able to trade 0.01), not only to Rs 20,000: Rs 5,000 comes in.
    Row 8 (a blow-up only at Rs 0) follows his #114 words instead - "even if it requires 25,000, then 20 only fine":
    no blow-up, 0.01 lot."""
    sim = run(daily([(0, 25_000)]))
    assert sim.blowups.sum() == 1 and sim.alive[0]
    assert sim.trace["lots"].tolist() == [0.01]
    assert money(sim) == (25_000.0, 175_000.0, 0.0, 5_000.0)
    sim8 = run(daily([(0, 25_000)]), ROW["row 8"])
    assert sim8.blowups.sum() == 0 and sim8.trace["lots"].tolist() == [0.01]
    assert money(sim8) == (20_000.0, 180_000.0, 0.0, 0.0)


def test_the_reserve_used_up_stops_the_plan():
    """Your money Rs 40,000: Rs 10,000 in the account (25%), Rs 30,000 in reserve; Rs 5,000 needed -> 0.02; every
    trade loses Rs 4,000 at 0.01 lot.
      trade 1: -8,000 -> 2,000: blow-up 1, refill 8,000 (reserve 22,000) -> 10,000
      trade 2: -8,000 -> 2,000: blow-up 2, refill 8,000 (reserve 14,000) -> 10,000
      trade 3: -8,000 -> 2,000: blow-up 3, refill 8,000 (reserve 6,000)  -> 10,000
      trade 4: -8,000 -> 2,000: blow-up 4, refill 6,000 (reserve 0)      ->  8,000  (still above the line: 0.01 lot)
      trade 5: -4,000 -> 4,000: blow-up 5, nothing left to add -> below the line: THE RESERVE IS USED UP, the plan stops
      trade 6: not taken.
    Your money at the end = the Rs 4,000 still in the account = 40,000 - 36,000 lost."""
    sim = run(daily([-4_000] * 6), Plan(your_money=40_000.0, account_share=0.25))
    assert sim.trace["lots"].tolist() == [0.02, 0.02, 0.02, 0.02, 0.01]
    assert money(sim) == (4_000.0, 0.0, 0.0, 30_000.0)
    assert sim.blowups.sum() == 5 and not sim.alive[0] and sim.taken[0] == 5
    assert sim.your_money_end[0] == 4_000.0
    assert sim.events[-1]["what"] == "the reserve is used up"
    assert MP.headline(sim)["reserve_used_up"] == "Jan 2019"


def test_refilling_only_to_the_line_makes_every_next_loss_a_blow_up():
    """Row 7 (the prompt's words read literally). Trade 1: -4,000 x 4 = -16,000 -> Rs 4,000: a blow-up; only Rs 1,000
    comes in (up to the Rs 5,000 needed for 0.01 lot). Trades 2 and 3 lose Rs 100 each at 0.01 lot: 4,900 < 5,000 ->
    a blow-up each time, Rs 100 in each time. 3 blow-ups.
    Version 1 on the same trades: 1 blow-up (after it the account is Rs 20,000: -100 x 4 -> 19,600, then
    19,600 / 5,000 = 3.92 -> 0.03: -300 -> 19,300)."""
    frame = daily([-4_000, -100, -100])
    sim7 = run(frame, ROW["row 7"])
    assert sim7.blowups.sum() == 3
    assert money(sim7) == (5_000.0, 178_800.0, 0.0, 1_200.0)
    sim1 = run(frame)
    assert sim1.blowups.sum() == 1
    assert money(sim1) == (19_300.0, 164_000.0, 0.0, 16_000.0)


def test_the_broker_closes_a_trade_where_the_account_ran_out_in_every_row():
    """Rs 20,000, Rs 5,000 needed -> 0.04. The trade ends +Rs 1,000 at 0.01 lot, but inside it the lowest point was
    -Rs 6,000 at 0.01 lot = -24,000 at 0.04: the account would be at -4,000, so the broker closes it when the account
    reaches Rs 0 - the win never comes. A STOP-OUT and a blow-up; Rs 20,000 comes in (180,000 -> 160,000).
    This is the broker, not a setting (L20-notes 3c point 6): version 1 and row 8 alike. (Before 1 Oct 2026 version 1
    let this plain trade go on below Rs 0 and end at +4,000: a plain trade has a stop, so it never happened on real
    trades - the plain run's numbers are the same to the paisa, L20-notes 3g.) A plain trade has no minute-by-minute
    money, so the stop-out is placed at its close."""
    frame = table((MON, MON + pd.Timedelta(hours=1), 1_000.0, 5_000.0, -6_000.0))
    for plan in (VERSION_1, ROW["row 8"]):
        sim = run(frame, plan)
        assert sim.trace["inr"].tolist() == [-20_000.0], plan.name
        assert sim.blowups.sum() == 1 and sim.stopouts.sum() == 1 and money(sim) == (20_000.0, 160_000.0, 0.0,
                                                                                     20_000.0)
        assert sim.trace["stopped_out"].tolist() == [True] and sim.low[0, 3] == 0.0
        assert sim.trace["lowest_inside_inr"].tolist() == [-20_000.0]
        assert [e["what"] for e in sim.events] == ["stop-out", "blow-up"]
        assert sim.events[1]["why"] == "after the broker closed the trade at Rs 0"
        assert abs(MP.when(sim.events[0]["t"]) - (MON + pd.Timedelta(hours=1))) < pd.Timedelta(seconds=1)
    # one rupee short of Rs 0 is not a stop-out: -4,999.75 x 4 = -19,999 -> the account touched Rs 1 and came back
    sim = run(table((MON, MON + pd.Timedelta(hours=1), 1_000.0, 5_000.0, -4_999.75)))
    assert sim.stopouts.sum() == 0 and sim.blowups.sum() == 0 and money(sim) == (24_000.0, 180_000.0, 0.0, 0.0)
    assert sim.under.sum() == 1                                   # below the Rs 5,000 line inside: counted, no blow-up


def test_below_the_line_row_8_still_trades_the_smallest_lot():
    """Row 8: -3,900 x 4 = -15,600 -> Rs 4,400, under the Rs 5,000 line but above Rs 0: no blow-up, and the next trade
    is 0.01 lot (#114). Version 1 calls the same moment a blow-up."""
    frame = daily([-3_900, 0])
    sim8 = run(frame, ROW["row 8"])
    assert sim8.blowups.sum() == 0 and sim8.trace["lots"].tolist() == [0.04, 0.01]
    assert run(frame).blowups.sum() == 1


# ================================================================================================================
# the other rows
# ================================================================================================================

def test_no_chocolate_rule_is_always_the_smallest_lot():
    """Row 5: 0.01 lot whatever the account holds: +5,000 -> Rs 25,000, then +5,000 -> Rs 30,000."""
    sim = run(daily([5_000, 5_000]), ROW["row 5"])
    assert sim.trace["lots"].tolist() == [0.01, 0.01]
    assert money(sim) == (30_000.0, 180_000.0, 0.0, 0.0)


def test_his_sheet_number_replaces_the_monday_number():
    """Row 6: Rs 7,100 for 0.01 lot every week (capital-plan's sheet), whatever that Monday's ADR says:
    20,000 / 7,100 = 2.8 -> 0.02 even in a week whose own number is Rs 3,000 (version 1: 0.06). The blow-up line is
    Rs 7,100 too: -3,300 x 2 = -6,600 -> Rs 13,400: fine; then 13,400 / 7,100 = 1.9 -> 0.01."""
    frame = daily([(-3_300, 3_000), (0, 3_000)])
    sim = run(frame, ROW["row 6"])
    assert sim.trace["lots"].tolist() == [0.02, 0.01]
    assert sim.trace["need_inr"].tolist() == [7_100.0, 7_100.0]
    assert run(frame).trace["lots"].tolist() == [0.06, 0.06]      # 6 x -3,300 = -19,800 -> 200: a blow-up, refilled


def test_the_account_share_sets_the_account_and_the_reserve():
    """Rows 2 and 3: the account 5% (Rs 10,000, reserve Rs 190,000) and 20% (Rs 40,000, reserve Rs 160,000).
    Rs 5,000 needed: 0.02 and 0.08."""
    s2, s3 = run(daily([0]), ROW["row 2"]), run(daily([0]), ROW["row 3"])
    assert money(s2) == (10_000.0, 190_000.0, 0.0, 0.0) and s2.trace["lots"].tolist() == [0.02]
    assert money(s3) == (40_000.0, 160_000.0, 0.0, 0.0) and s3.trace["lots"].tolist() == [0.08]


# ================================================================================================================
# the books always balance
# ================================================================================================================

def _random_table(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, t = [], pd.Timestamp("2016-02-01 09:00")
    for _ in range(n):
        t = t + pd.Timedelta(hours=int(rng.integers(2, 120)))
        win = rng.random() < 0.2
        net = float(rng.uniform(300, 2_000)) if win else -float(rng.uniform(60, 260))
        low = -float(rng.uniform(40, 200)) if win else net
        rows.append((t, t + pd.Timedelta(hours=1), net, float(rng.uniform(1_500, 5_000)), min(low, -40.0)))
    return table(*rows)


@pytest.mark.parametrize("plan", [VERSION_1, *EXTRA_ROWS], ids=lambda p: p.name)
def test_your_money_changes_only_by_trades(plan):
    """The reserve + the account + everything taken out - the start = the sum of every trade's rupees at the plan's
    lots. Withdrawals and refills only move money between your own pockets. And the per-year numbers add up."""
    frame = _random_table()
    sim = run(frame, plan)
    assert sim.result[0] == pytest.approx(sim.trace["inr"].sum(), abs=1e-6)
    assert sim.pnl.sum() == pytest.approx(sim.trace["inr"].sum(), abs=1e-6)
    assert sim.out_y.sum() == pytest.approx(sim.out[0]) and sim.added_y.sum() == pytest.approx(sim.added[0])
    assert sim.trades.sum() == sim.taken[0] == len(sim.trace)
    assert plan.reserve_size - sim.reserve[0] == pytest.approx(sim.added[0])
    steps = sim.trace["lots"].to_numpy() * 100
    assert np.allclose(steps, np.round(steps)) and steps.min() >= 1            # whole 0.01 steps only (#116)
    assert sim.steps.sum() == len(sim.trace)
    rows = MP.year_rows(sim, TradeSet.build(frame))
    assert sum(r["result"] for r in rows) == pytest.approx(sim.result[0], abs=1e-6)
    assert sum(r["blowups"] for r in rows) == sim.blowups.sum()
    assert rows[0]["account_start"] == plan.account_size


def test_the_lots_never_use_a_later_trade():
    """Cut off the future: the lots and the money of the first 150 trades are the same whether the plan is run on
    150 trades or on all 400."""
    frame = _random_table()
    full, part = run(frame), run(frame.iloc[:150])
    cols = ["lots", "inr", "account_after_trade", "taken_out"]
    pd.testing.assert_frame_equal(full.trace.iloc[:150][cols], part.trace[cols])


# ================================================================================================================
# the reshuffles: whole months, each used once
# ================================================================================================================

def _two_years() -> TradeSet:
    rows = []
    for m in range(24):
        first = pd.Timestamp(2016 + m // 12, m % 12 + 1, 1)
        for j, day in enumerate((3, 12, 21)):
            o = first + pd.Timedelta(days=day, hours=9)
            rows.append((o, o + pd.Timedelta(hours=5), 100.0 * m + j, 2_000.0 + m))     # the money names its month
    return TradeSet.build(table(*rows), last_month="2017-12")


def test_a_reshuffle_uses_every_month_once_and_keeps_each_month_whole():
    ts = _two_years()
    paths = MP.reshuffled(ts, runs=40, seed=3)
    assert paths.order.shape == (40, 72) and paths.n_years == 2
    months_in_order = ts.month[paths.order]
    for p in range(40):
        assert sorted(paths.order[p]) == list(range(72))                        # every trade exactly once
        assert sorted(paths.months[p]) == list(range(24))                       # every month exactly once
        seen = months_in_order[p].reshape(24, 3)
        assert (seen == seen[:, :1]).all()                                      # a month's 3 trades stay together
        assert seen[:, 0].tolist() == paths.months[p].tolist()                  # ... in the order of the reshuffle
        assert (np.diff(paths.order[p].reshape(24, 3), axis=1) == 1).all()      # ... and in their own order
        assert (paths.year[p] == np.repeat(np.arange(24) // 12, 3)).all()       # a path-year = 12 months
    assert (paths.t_open[:, 1:] >= paths.t_close[:, :-1]).all()                 # one trade at a time
    assert np.allclose(paths.t_close - paths.t_open, 5 / 24)                    # every trade keeps its length
    assert len({tuple(m) for m in paths.months}) > 30                           # the orders really differ


def test_a_reshuffle_keeps_each_trade_at_its_place_in_its_month():
    """A trade that opened on the 4th day of its month at 09:00 opens 3 days 9 hours after its month starts on the
    path's clock too. The whole path is as long as the 24 months (731 days: 2016 is a leap year)."""
    ts = _two_years()
    paths = MP.reshuffled(ts, runs=5, seed=1)
    start, length = ts.month_starts()
    assert length.sum() == 731
    for p in range(5):
        blen = length[paths.months[p]]
        bstart = np.cumsum(blen) - blen
        first_of_block = paths.t_open[p].reshape(24, 3)[:, 0]
        assert np.allclose(first_of_block - bstart, 3 + 9 / 24)


def test_the_same_seed_gives_the_same_reshuffles():
    ts = _two_years()
    a, b, c = MP.reshuffled(ts, 10, seed=5), MP.reshuffled(ts, 10, seed=5), MP.reshuffled(ts, 10, seed=6)
    assert (a.order == b.order).all() and (a.t_open == b.t_open).all()
    assert (a.order != c.order).any()
    assert MP.SEED == 20260930 and MP.RESHUFFLES == 1000


def test_a_long_trade_pushes_the_next_one_back():
    """A trade that runs 40 days past its month's end: on a reshuffled path the next trade cannot open before it
    closed (one trade at a time) - its open is pushed to that close."""
    rows = [(pd.Timestamp("2016-01-20 10:00"), pd.Timestamp("2016-03-10 10:00"), 50.0)]
    rows += [(pd.Timestamp(2016, m, 5, 10), pd.Timestamp(2016, m, 5, 12), 10.0) for m in range(4, 13)]
    ts = TradeSet.build(table(*rows), last_month="2016-12")
    paths = MP.reshuffled(ts, runs=60, seed=2)
    assert (paths.t_open[:, 1:] >= paths.t_close[:, :-1] - 1e-12).all()
    pushed = (np.take_along_axis(np.broadcast_to(ts.close_d - ts.open_d, paths.order.shape), paths.order, 1))
    assert np.allclose(paths.t_close - paths.t_open, pushed)


def test_many_paths_at_once_give_the_same_as_one_at_a_time():
    """The 1,000 reshuffles run together in one go; each must come out exactly as if it were run alone."""
    ts = TradeSet.build(_random_table(300, seed=11))
    paths = MP.reshuffled(ts, runs=25, seed=9)
    for plan in (VERSION_1, ROW["row 4"], ROW["row 7"], ROW["row 8"]):
        together = MP.simulate(ts, paths, plan)
        for p in (0, 7, 24):
            alone = MP.simulate(ts, paths.one(p), plan, trace=True)
            assert together.your_money_end[p] == pytest.approx(alone.your_money_end[0], abs=1e-6)
            assert together.blowups[p].tolist() == alone.blowups[0].tolist()
            assert together.doubles[p].tolist() == alone.doubles[0].tolist()
            assert together.out[p] == pytest.approx(alone.out[0]) and together.added[p] == pytest.approx(alone.added[0])
            assert np.allclose(together.dbl_days[together.dbl_path == p], alone.dbl_days)


def test_the_order_does_not_matter_when_the_lot_never_changes():
    """A comparison that must NOT move (lesson B5 the other way round): 0.01 lot always, a very large account, nothing
    taken out - every reshuffle ends with exactly the same money, the sum of the trades. Only the lot rule, the
    blow-ups and the withdrawals make the order matter."""
    ts = TradeSet.build(_random_table(300, seed=11))
    plan = Plan(sizing="flat", your_money=10_000_000.0, account_share=1.0, withdraw_at=None)
    sim = MP.simulate(ts, MP.reshuffled(ts, runs=30, seed=4), plan)
    total = float((ts.gross - ts.cost).sum())
    assert np.allclose(sim.result, total) and sim.blowups.sum() == 0
    moved = MP.simulate(ts, MP.reshuffled(ts, runs=30, seed=4), VERSION_1)
    assert np.ptp(moved.result) > 1_000                                          # version 1: the order matters


def test_the_spread_reads_the_middle_and_the_bad_tenth():
    ts = TradeSet.build(_random_table(300, seed=11))
    sim = MP.simulate(ts, MP.reshuffled(ts, runs=200, seed=4), VERSION_1)
    s = MP.spread(sim)
    assert s["runs"] == 200 and s["seed"] == 4 and len(s["per_year"]) == sim.paths.n_years
    a = s["all_years"]
    assert a["result_bad"] <= a["result_mid"] <= a["result_good"]
    assert a["added_mid"] <= a["added_bad"] and a["blowups_mid"] <= a["blowups_bad"]
    w = s["worst"]
    assert w["your_money_end"] == pytest.approx(float(sim.your_money_end.min()))
    assert w["result"] == pytest.approx(w["your_money_end"] - 200_000.0)
    if s["any_year"]["n_doubles"]:
        assert s["any_year"]["days_mid"] <= s["any_year"]["days_slow"]


# ================================================================================================================
# the trade table
# ================================================================================================================

def test_the_trade_table_is_checked():
    good = daily([10, 20])
    with pytest.raises(ValueError, match="overlap"):                            # one trade at a time
        TradeSet.build(table((MON, MON + pd.Timedelta(days=2), 1.0), (MON + pd.Timedelta(days=1),
                                                                         MON + pd.Timedelta(days=3), 1.0)))
    with pytest.raises(ValueError, match="columns"):
        TradeSet.build(good.drop(columns=["low_inr"]))
    with pytest.raises(ValueError, match="above 0"):
        TradeSet.build(good.assign(need_inr=[5_000.0, 0.0]))
    with pytest.raises(ValueError, match="lowest point"):
        TradeSet.build(good.assign(low_inr=[-40.0, 5.0]))
    with pytest.raises(SealedYearsError):                                       # 2024+ is sealed (#96)
        TradeSet.build(table((pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-02 11:00"), 1.0)))


def test_lowest_point_inside_a_trade_by_hand():
    """At 0.01 lot = 1 ounce, Rs 83 a dollar, Rs 40 owed from the start.
    A buy filled at 2000.00 at 10:00, closed at 10:03. Bid lows 10:00 1999.00, 10:01 1998.50, 10:02 1999.50; the close
    minute (low 1990) is not read - the close itself is.
      stopped at 1998.00: money -2 x 83 - 40 = -206. Lowest before = -1.5 x 83 - 40 = -164.5 -> the close is lower: -206
      a winner closed at 2010.00 (+790): the lowest point was -1.5 x 83 - 40 = -164.5
    A sell filled at 2000.00: bid highs 2000.40, 2001.20, 2000.10 -> -1.2 x 83 - 40 = -139.6.
    Closed in its fill minute (+790): only the cost was ever owed: -40."""
    t = pd.DatetimeIndex(pd.date_range("2019-01-07 10:00", periods=5, freq="1min")).as_unit("ns").asi8
    book = SimpleNamespace(t=t, lo=np.array([1999.0, 1998.5, 1999.5, 1990.0, 2000.0]),
                           h=np.array([2000.4, 2001.2, 2000.1, 2012.0, 2000.0]))
    at = lambda m: pd.Timestamp("2019-01-07 10:00") + pd.Timedelta(minutes=m)      # noqa: E731

    def trade(direction, close_min, net):
        return SimpleNamespace(signal_id="t", direction=direction, fill_ts=at(0), exit_ts=at(close_min), fill_px=2000.0,
                               usd_inr=83.0, cost_inr=40.0, net_inr=net, exits=[SimpleNamespace(lots=0.01)])

    lows = MP.trade_lows(book, [trade("buy", 3, -206.0), trade("buy", 3, 790.0), trade("sell", 3, 100.0),
                                trade("buy", 0, 790.0)])
    assert lows == pytest.approx([-206.0, -164.5, -139.6, -40.0])
    with pytest.raises(ValueError, match="per 0.01 lot"):
        MP.trade_lows(book, [SimpleNamespace(signal_id="t", exits=[SimpleNamespace(lots=0.02)])])


# ================================================================================================================
# the dips inside a HEDGED trade (what plain trades cannot show)
# ================================================================================================================

def _hedged_case(buy_closes: bool):
    """Seven 1-minute bars from Monday 7 Jan 2019 10:00, and one hedged trade: a buy at 2000.00 (10:00) and its
    hedge, a sell at 1998.00 (10:02). The sell closes at 1996.00 (10:04); the buy closes at 2001.00 (10:06) or not at
    all. The week needs Rs 500 for 0.01 lot."""
    idx = pd.DatetimeIndex(pd.date_range("2019-01-07 10:00", periods=7, freq="1min"), name="server_time").as_unit("ns")
    m1 = pd.DataFrame({"low": [1999.5, 1998.2, 1997.5, 1997.0, 1995.5, 1996.0, 2000.5],
                       "high": [2000.5, 1999.9, 1998.4, 1998.0, 1996.5, 1997.0, 2001.5]}, index=idx)
    r = costs.usd_inr("2019-01-07")
    sell = {"direction": "sell", "opened": "2019-01-07 10:02:00", "open_px": 1998.0, "closed": "2019-01-07 10:04:00",
            "close_px": 1996.0, "net_inr": 2 * r - 40}
    buy = {"direction": "buy", "opened": "2019-01-07 10:00:00", "open_px": 2000.0,
           "closed": "2019-01-07 10:06:00" if buy_closes else None, "close_px": 2001.0 if buy_closes else None,
           "net_inr": (1 * r - 40) if buy_closes else None}
    story = {"signal": "made-up", "status": "closed" if buy_closes else "open at end", "legs": [buy, sell],
             "net_inr": (3 * r - 80) if buy_closes else None}
    needs = pd.DataFrame({"day": [pd.Timestamp("2019-01-07")], "known_at": [pd.Timestamp("2019-01-07 01:10")],
                          "adr_usd": [250 / r], "need_inr": [500.0]})
    return story, m1, needs, r


def test_the_lowest_point_inside_a_hedged_trade_by_hand():
    """Every leg 0.01 lot = 1 ounce; r = the rupee rate of January 2019; each open leg owes its Rs 40.
      10:00  the buy alone, at the minute's low 1999.50: -0.5 r - 40
      10:01  low 1998.20: -1.8 r - 40
      10:02  the sell opens at 1998.00: locked at 1998 - 2000 = -2 -> -2 r - 80 (both legs open: price cannot move it)
      10:03  the same: -2 r - 80
      10:04  the sell closes at 1996.00: it made 2 r - 40. The buy alone at the low 1995.50: -4.5 r - 40.
             Together: -2.5 r - 80   <- THE LOWEST POINT
      10:05  low 1996.00: 2 r - 40 - 4 r - 40 = -2 r - 80
      10:06  the buy closes at 2001.00: 2 r - 40 + r - 40 = 3 r - 80 = the trade's money.
    The week needs Rs 500 for 0.01 lot: the lowest point is (2.5 r + 80) / 500 of it. A full Rs 20,000 account trades
    20,000 / 500 = 40 steps: the dip is 40 x the lowest point."""
    story, m1, needs, r = _hedged_case(buy_closes=True)
    d = MP.hedged_dips([story], m1, needs).iloc[0]
    low = -2.5 * r - 80
    assert d["low_inr"] == pytest.approx(low) and d["low_when"] == pd.Timestamp("2019-01-07 10:04")
    assert d["net_inr"] == pytest.approx(3 * r - 80) and not d["still_open"]
    assert (d["positions"], d["most_open_at_once"]) == (2, 2)
    assert d["days"] == pytest.approx(6 / 1440) and d["days_below_zero"] == pytest.approx(6 / 1440)
    assert d["low_in_needs"] == pytest.approx(-low / 500)
    assert d["plan_steps"] == 40 and d["dip_at_plan_lots"] == pytest.approx(40 * low)
    assert d["account_at_low"] == pytest.approx(20_000 + 40 * low)


def test_a_hedged_trade_still_open_is_read_to_the_last_bar():
    """The buy never closes: the trade is still open at the window's end. Its lowest point is the same 10:04 moment;
    it has no money at the end (an open trade is never counted - lesson B6)."""
    story, m1, needs, r = _hedged_case(buy_closes=False)
    d = MP.hedged_dips([story], m1, needs).iloc[0]
    assert d["still_open"]
    assert pd.isna(d["net_inr"]) and pd.isna(d["ended"])
    assert d["low_inr"] == pytest.approx(-2.5 * r - 80)
    assert d["days"] == pytest.approx(6 / 1440)


def test_hedged_legs_that_do_not_add_up_are_refused():
    story, m1, needs, _ = _hedged_case(buy_closes=True)
    story["net_inr"] += 5.0
    with pytest.raises(AssertionError, match="add up"):
        MP.hedged_dips([story], m1, needs)


def test_a_hedged_trades_minutes_before_any_cost_by_hand():
    """The same trade as above, before any cost (row 9, NOT REAL): every leg opened so far gets its Rs 40 back.
      10:00 -0.5 r (one leg opened) · 10:02 -2 r (two) · 10:04 -2.5 r · 10:06 3 r = the trade's money + 2 x 40."""
    story, m1, _, r = _hedged_case(buy_closes=True)
    got = MP.trade_minutes(story, MP.MinuteBook.build(m1))
    assert got["inr0"].tolist() == pytest.approx([-0.5 * r, -1.8 * r, -2 * r, -2 * r, -2.5 * r, -2 * r, 3 * r])
    assert got["inr"][-1] == pytest.approx(3 * r - 80) and not got["still_open"]
    days = np.arange(7) / 1440
    ins = MP.Inside.from_minutes(days, got["inr"], got["inr0"])
    assert ins.v.tolist() == pytest.approx([-0.5 * r - 40, -1.8 * r - 40, -2 * r - 80, -2.5 * r - 80])
    assert ins.t.tolist() == pytest.approx(days[[0, 1, 2, 4]].tolist())         # only the minutes of a new low


# ================================================================================================================
# INSIDE an open trade: the broker's stop-out at Rs 0, and row 10 (L20-notes 3c, 3d)
# ================================================================================================================

def hedged(*rows, need=5_000.0):
    """Made-up hedged trades. rows = (open, close, net at 0.01 lot a leg, new lows [(days from the open, money at
    0.01 lot)...]); before any cost every new low is Rs 40 higher (one position). Returns (the trade table, Inside)."""
    out, inside = [], {}
    for j, (o, c, net, lows) in enumerate(rows):
        t = np.array([x[0] for x in lows], float)
        v = np.array([x[1] for x in lows], float)
        sig = f"h{j}"
        inside[sig] = MP.Inside(t, v, t, v + 40.0)
        out.append({"signal": sig, "open": pd.Timestamp(o), "close": pd.Timestamp(c), "gross_inr": net + 40.0,
                    "cost_inr": 40.0, "need_inr": need, "low_inr": v[-1], "low0_inr": v[-1] + 40.0})
    return pd.DataFrame(out), inside


def run_h(frame, inside, plan=VERSION_1):
    ts = TradeSet.build(frame, inside=inside)
    return MP.simulate(ts, MP.real_order(ts), plan, trace=True)


DIP = [(0.0, -40.0), (0.5, -2_000.0), (1.0, -4_000.0), (2.0, -6_000.0)]      # a trade's new lows at 0.01 lot a leg


def test_the_stop_out_by_hand():
    """Rs 20,000, Rs 5,000 needed -> every leg 0.04 lot (4 steps). The trade's new lows at 0.01 lot a leg: -40 (day 0),
    -2,000 (day 0.5), -4,000 (day 1), -6,000 (day 2); it would end +1,000 on day 3. The account: 20,000 + 4 x ... =
    19,840, 12,000, 4,000 (below the Rs 5,000 line, above Rs 0: the trade goes on), then -4,000: the broker closes every
    leg ON DAY 2 - a STOP-OUT: the whole Rs 20,000 is lost, the +1,000 never comes. Then the refill (Rs 20,000 from the
    reserve). The day the trade never had: 1. The next trade (day 4, price did not move: -40 x 4) is not moved."""
    frame, inside = hedged((MON, MON + pd.Timedelta(days=3), 1_000.0, DIP),
                           (MON + pd.Timedelta(days=4), MON + pd.Timedelta(days=5), -40.0, [(0.0, -40.0)]))
    sim = run_h(frame, inside)
    assert sim.trace["inr"].tolist() == [-20_000.0, -160.0]
    assert sim.trace["stopped_out"].tolist() == [True, False]
    assert sim.stopouts.sum() == 1 and sim.blowups.sum() == 1 and sim.under.sum() == 0
    assert money(sim) == (19_840.0, 160_000.0, 0.0, 20_000.0)
    assert sim.cut_days[0] == pytest.approx(1.0)
    e = sim.events
    assert [x["what"] for x in e] == ["stop-out", "blow-up"]
    assert abs(MP.when(e[0]["t"]) - (MON + pd.Timedelta(days=2))) < pd.Timedelta(seconds=1)
    assert (e[0]["account_before"], e[0]["account_after"]) == (20_000.0, 0.0)
    assert (e[1]["why"], e[1]["money_in_out"], e[1]["account_after"]) == \
        ("after the broker closed the trade at Rs 0", 20_000.0, 20_000.0)
    assert MP.when(sim.trace["opened_d"].iloc[1]) - (MON + pd.Timedelta(days=4)) < pd.Timedelta(seconds=1)


def test_below_the_line_inside_a_trade_is_not_a_blow_up():
    """The same dip, but the trade's worst is -4,000 at 0.01 lot a leg: the account falls to 20,000 - 16,000 = 4,000,
    under the Rs 5,000 line but above Rs 0. No money goes in while a trade is open (his burner account): the trade goes
    on and ends +1,000 x 4 = +4,000 -> Rs 24,000. Counted as "below the line inside a trade" - not a blow-up."""
    frame, inside = hedged((MON, MON + pd.Timedelta(days=3), 1_000.0, DIP[:3]))
    sim = run_h(frame, inside)
    assert sim.blowups.sum() == 0 and sim.stopouts.sum() == 0 and sim.under.sum() == 1
    assert money(sim) == (24_000.0, 180_000.0, 0.0, 0.0)
    assert sim.trace["account_lowest_inside"].tolist() == [4_000.0]


def test_row_10_tops_the_account_up_inside_a_trade_by_hand():
    """Row 10, the same trade (4 steps). Day 1: the account is 20,000 - 16,000 = 4,000, below the Rs 5,000 line: a
    BLOW-UP inside the trade - the reserve puts in 16,000 (180,000 -> 164,000) and the account is back at Rs 20,000; a
    new cycle starts there. Day 2 (-6,000): 20,000 + 16,000 - 24,000 = 12,000, above the line. The trade goes on to
    its end: +1,000 x 4 = +4,000 -> 20,000 + 16,000 + 4,000 = Rs 40,000 = 2 x the cycle's Rs 20,000: a DOUBLE, 2 days
    after the top-up; Rs 20,000 is taken out. Your money = 164,000 + 20,000 + 20,000 = 204,000 = 200,000 + the trade's
    4,000 (the top-up only moved money between your own pockets)."""
    frame, inside = hedged((MON, MON + pd.Timedelta(days=3), 1_000.0, DIP))
    sim = run_h(frame, inside, MP.ROW_10)
    assert sim.trace["inr"].tolist() == [4_000.0]
    assert sim.blowups.sum() == 1 and sim.stopouts.sum() == 0 and sim.doubles.sum() == 1
    assert money(sim) == (20_000.0, 164_000.0, 20_000.0, 16_000.0)
    assert sim.put_in[0] == 16_000.0 and sim.dbl_days.tolist() == pytest.approx([2.0])
    assert sim.your_money_end[0] == 204_000.0
    e = sim.events
    assert [x["what"] for x in e] == ["blow-up", "a double"]
    assert (e[0]["account_before"], e[0]["money_in_out"], e[0]["account_after"], e[0]["reserve_after"]) == \
        (4_000.0, 16_000.0, 20_000.0, 164_000.0)
    assert abs(MP.when(e[0]["t"]) - (MON + pd.Timedelta(days=1))) < pd.Timedelta(seconds=1)
    assert sim.trace["account_lowest_inside"].tolist() == [4_000.0]


def test_row_10_until_the_reserve_is_used_up_inside_a_trade():
    """Your money Rs 40,000: Rs 20,000 in the account (50%), Rs 20,000 in reserve; 4 steps. New lows -4,000 (day 1),
    -8,000 (day 2), -12,000 (day 3).
      day 1: 20,000 - 16,000 = 4,000 < 5,000 -> top-up 16,000 (reserve 4,000) -> 20,000
      day 2: 20,000 + 16,000 - 32,000 = 4,000 -> top-up of the last 4,000 (reserve 0) -> 8,000
      day 3: 20,000 + 20,000 - 48,000 = -8,000 -> Rs 0: THE STOP-OUT; everything in the account is lost (Rs 40,000)
    Then nothing is left to refill: the reserve is used up, the plan stops. Blow-ups: the 2 top-ups + the one after the
    stop-out. Your money at the end: 0."""
    lows = [(0.0, -40.0), (1.0, -4_000.0), (2.0, -8_000.0), (3.0, -12_000.0)]
    frame, inside = hedged((MON, MON + pd.Timedelta(days=4), -1_000.0, lows),
                           (MON + pd.Timedelta(days=5), MON + pd.Timedelta(days=6), 0.0, [(0.0, -40.0)]))
    plan = dataclasses.replace(MP.ROW_10, your_money=40_000.0, account_share=0.5)
    sim = run_h(frame, inside, plan)
    assert sim.trace["inr"].tolist() == [-40_000.0] and sim.taken[0] == 1
    assert sim.stopouts.sum() == 1 and sim.blowups.sum() == 3 and not sim.alive[0]
    assert money(sim) == (0.0, 0.0, 0.0, 20_000.0) and sim.your_money_end[0] == 0.0
    assert [x["what"] for x in sim.events] == ["blow-up", "blow-up", "stop-out", "blow-up", "the reserve is used up"]
    assert [x["money_in_out"] for x in sim.events[:2]] == [16_000.0, 4_000.0]
    # version 1 on the same money: no top-up; the account reaches Rs 0 on day 2 (20,000 - 32,000 < 0)
    v1 = run_h(frame, inside, dataclasses.replace(VERSION_1, your_money=40_000.0, account_share=0.5))
    assert v1.trace["inr"].tolist()[0] == -20_000.0 and v1.stopouts.sum() == 1
    assert MP.when(v1.events[0]["t"]) - (MON + pd.Timedelta(days=2)) < pd.Timedelta(seconds=1)


def test_a_minute_already_at_zero_when_it_crosses_the_line_is_a_stop_out_not_a_top_up():
    """Row 10: the first new low under the line is already -6,000 x 4 = -24,000 -> the account at -4,000: the broker
    closes it in that minute - there is no time to add money."""
    frame, inside = hedged((MON, MON + pd.Timedelta(days=3), 1_000.0, [(0.0, -40.0), (1.0, -6_000.0)]))
    sim = run_h(frame, inside, MP.ROW_10)
    assert sim.stopouts.sum() == 1 and sim.put_in[0] == 0.0 and sim.blowups.sum() == 1
    assert money(sim) == (20_000.0, 160_000.0, 0.0, 20_000.0)


def test_a_trade_that_ends_early_lets_the_next_one_open_on_its_own_time():
    """One trade at a time, pushed from the trade's REAL end: a 30-day trade stopped out on day 2 frees the account on
    day 2; the next trade (its own time: day 10) opens on day 10, not on day 30. Without the stop-out it waits for day 30.
    (Two steps of a made-up order - on a reshuffled path the next month can start inside a long trade.)"""
    frame, inside = hedged((MON, MON + pd.Timedelta(days=30), 1_000.0, DIP),
                           (MON + pd.Timedelta(days=40), MON + pd.Timedelta(days=41), 0.0, [(0.0, -40.0)]))
    ts = TradeSet.build(frame, inside=inside)
    d0 = ts.open_d[0]
    paths = MP.Paths(np.array([[0, 1]]), np.array([[d0, d0 + 30]]), np.array([[d0 + 30, d0 + 31]]),
                     np.zeros((1, 2), int), 1, raw=np.array([[d0, d0 + 10]]))
    stopped = MP.simulate(ts, paths, VERSION_1, trace=True)
    assert stopped.trace["opened_d"].tolist() == pytest.approx([d0, d0 + 10])
    assert stopped.trace["ended_d"].tolist() == pytest.approx([d0 + 2, d0 + 11])
    rich = MP.simulate(ts, paths, dataclasses.replace(VERSION_1, your_money=1_000_000.0, account_share=0.5),
                       trace=True)                                    # 500,000 / 5,000 = 100 steps: -6,000 x 100 < 0
    assert rich.stopouts.sum() == 1
    far = MP.simulate(ts, paths, dataclasses.replace(VERSION_1, sizing="flat"), trace=True)   # 0.01: no stop-out
    assert far.stopouts.sum() == 0 and far.trace["opened_d"].tolist() == pytest.approx([d0, d0 + 30])


def test_row_10_is_one_change_and_needs_minutes():
    base = dataclasses.asdict(VERSION_1)
    d = dataclasses.asdict(MP.ROW_10)
    assert [k for k in base if k not in ("name", "change") and d[k] != base[k]] == ["inside"]
    assert MP.HEDGED_ROWS == (*EXTRA_ROWS, MP.ROW_10)
    with pytest.raises(ValueError, match="minute by minute"):
        run(daily([10]), MP.ROW_10)                                   # a plain trade has no minutes
    with pytest.raises(ValueError):
        Plan(inside="sometimes")


def test_a_hedged_table_must_bring_its_minutes():
    frame, inside = hedged((MON, MON + pd.Timedelta(days=3), 1_000.0, DIP))
    with pytest.raises(ValueError, match="minute by minute"):
        TradeSet.build(frame, inside={})
    with pytest.raises(ValueError, match="lowest point"):
        TradeSet.build(frame.assign(low_inr=-5_000.0), inside=inside)
    bad = {"h0": MP.Inside(np.array([0.0, 1.0]), np.array([-40.0, -10.0]), np.array([0.0]), np.array([0.0]))}
    with pytest.raises(ValueError):
        TradeSet.build(frame.assign(low_inr=-10.0, low0_inr=0.0), inside=bad)     # "new lows" that rise


def _random_hedged(n=120, seed=5):
    """Made-up hedged trades: a few days each, new lows that sink by random steps, a random end."""
    rng = np.random.default_rng(seed)
    rows, t = [], pd.Timestamp("2016-02-01 09:00")
    for _ in range(n):
        t = t + pd.Timedelta(days=int(rng.integers(3, 20)))
        days = float(rng.uniform(1, 12))
        k = int(rng.integers(1, 12))
        lows = [(0.0, -40.0)]
        for x in np.sort(rng.uniform(0, days, k)):
            lows.append((float(x), lows[-1][1] - float(rng.uniform(50, 1_500))))
        end = float(rng.uniform(lows[-1][1], 3_000))
        rows.append((t, t + pd.Timedelta(days=days), end, lows))
        t = t + pd.Timedelta(days=days)
    frame, inside = hedged(*rows)
    frame["need_inr"] = rng.uniform(1_500, 5_000, len(frame))
    return frame, inside


@pytest.mark.parametrize("plan", [VERSION_1, MP.ROW_10, ROW["row 8"], ROW["row 9"]], ids=lambda p: p.name)
def test_hedged_many_paths_at_once_give_the_same_as_one_at_a_time(plan):
    """With stop-outs and top-ups inside trades, each reshuffled path must come out as if run alone - and the books
    balance: your money changes only by the trades."""
    frame, inside = _random_hedged()
    ts = TradeSet.build(frame, inside=inside)
    paths = MP.reshuffled(ts, runs=30, seed=8)
    together = MP.simulate(ts, paths, plan)
    assert together.stopouts.sum() > 0                                   # the made-up trades do reach Rs 0
    for p in (0, 11, 29):
        alone = MP.simulate(ts, paths.one(p), plan, trace=True)
        assert together.your_money_end[p] == pytest.approx(alone.your_money_end[0], abs=1e-6)
        assert together.blowups[p].tolist() == alone.blowups[0].tolist()
        assert together.stopouts[p].tolist() == alone.stopouts[0].tolist()
        assert together.put_in[p] == pytest.approx(alone.put_in[0])
        assert np.allclose(together.dbl_days[together.dbl_path == p], alone.dbl_days)
        assert alone.result[0] == pytest.approx(alone.trace["inr"].sum(), abs=1e-6)
        assert plan.reserve_size - alone.reserve[0] == pytest.approx(alone.added[0])
        t = alone.trace
        assert (t["opened_d"].to_numpy()[1:] >= t["ended_d"].to_numpy()[:-1] - 1e-9).all()   # one at a time


# ================================================================================================================
# the money needed for 0.01 lot: 2 x that Monday's ADR
# ================================================================================================================

def day_bars(ranges: dict) -> pd.DataFrame:
    """Three 5-minute bars a day (01:00, 01:05, 01:10 server time); the day's high - low = the given range."""
    rows, idx = [], []
    for day, r in ranges.items():
        for j, (hi, lo) in enumerate(((1300.0, 1300.0), (1300.0 + r / 2, 1300.0), (1300.0, 1300.0 - r / 2))):
            idx.append(pd.Timestamp(day) + pd.Timedelta(hours=1, minutes=5 * j))
            rows.append((1300.0, hi, lo, 1300.0))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.DatetimeIndex(idx, name="server_time").as_unit("ns"))


def four_weeks(skip=()) -> pd.DataFrame:
    """Mon 7 Jan 2019 to Fri 1 Feb 2019: the first 10 days range $10, the next 10 days $20."""
    days = pd.bdate_range("2019-01-07", periods=20)
    return day_bars({d: (10.0 if i < 10 else 20.0) for i, d in enumerate(days) if str(d.date()) not in skip})


def test_the_money_needed_is_two_times_that_mondays_adr_by_hand():
    """Monday 21 Jan: the 5 finished days before it ranged $10 each and so did the 10 before it -> ADR $10 ->
    money needed = 2 x 10 x 1 ounce x the rupee rate of January 2019.
    Monday 28 Jan: the last 5 finished days ranged $20, the last 10 average $15 -> the LARGER, $20 (the top of his red
    zone to the bottom of his green zone) -> 2 x 20 x the rate.
    Mondays 7 and 14 Jan have fewer than 10 finished days: no number yet. Each number is known when the Monday's first
    5-minute bar has closed (01:05)."""
    need = MP.weekly_need(four_weeks())
    rate = costs.usd_inr("2019-01-21")
    assert need["day"].tolist() == [pd.Timestamp("2019-01-21"), pd.Timestamp("2019-01-28")]
    assert need["known_at"].tolist() == [pd.Timestamp("2019-01-21 01:05"), pd.Timestamp("2019-01-28 01:05")]
    assert need["adr_usd"].tolist() == pytest.approx([10.0, 20.0])
    assert need["adr5"].tolist() == pytest.approx([10.0, 20.0])
    assert need["adr10"].tolist() == pytest.approx([10.0, 15.0])
    assert need["need_inr"].tolist() == pytest.approx([2 * 10 * rate, 2 * 20 * rate])
    assert 60 < rate < 80                                                       # a real rate from data/usd_inr.csv


def test_a_week_that_starts_on_a_tuesday_uses_that_tuesday():
    """Monday 28 Jan is a holiday (no bars): the week's first broker day is Tuesday 29 Jan, and its number is known
    when that day's first bar closes."""
    need = MP.weekly_need(four_weeks(skip=("2019-01-28",)))
    assert need["day"].tolist()[-1] == pd.Timestamp("2019-01-29")
    assert need["known_at"].tolist()[-1] == pd.Timestamp("2019-01-29 01:05")
    assert need["adr_usd"].tolist()[-1] == pytest.approx(20.0)


def test_a_mondays_number_never_uses_a_later_bar():
    """Cut off the future (R4): with the bars cut right after a Monday's first bar, that Monday's row is the same as
    with every bar - so Monday's own range, and the rest of the week, are not in it."""
    bars = four_weeks()
    guards.assert_no_future_events(MP.weekly_need, bars, 5, cut_points=[31, 46, 50])
    cut = MP.weekly_need(bars.iloc[:46])                                        # ... up to Monday 28 Jan's first bar
    assert cut["day"].tolist()[-1] == pd.Timestamp("2019-01-28") and cut["adr_usd"].tolist()[-1] == pytest.approx(20.0)


def test_a_trade_gets_the_number_known_before_it_opened():
    """A trade that opens on Monday 28 Jan at 01:02 - before that Monday's first bar has closed (01:05) - still uses
    the week before's number. At 01:05 it uses the new one. A trade before the first number is left out (counted)."""
    need = MP.weekly_need(four_weeks())
    trades = table(("2019-01-18 10:00", "2019-01-18 11:00", 1.0), ("2019-01-23 10:00", "2019-01-23 11:00", 2.0),
                   ("2019-01-28 01:02", "2019-01-28 01:03", 3.0), ("2019-01-28 01:05", "2019-01-28 02:00", 4.0))
    got, dropped = MP.attach_need(trades.drop(columns=["need_inr"]), need)
    assert dropped == 1 and len(got) == 3
    assert got["monday"].tolist() == [pd.Timestamp("2019-01-21"), pd.Timestamp("2019-01-21"),
                                      pd.Timestamp("2019-01-28")]
    assert got["need_inr"].tolist() == pytest.approx(need["need_inr"].iloc[[0, 0, 1]].tolist())
    assert (got["need_known_at"] <= got["open"]).all()
    # a trade table that has its own known_at column (the signal's) keeps it: the Monday's is need_known_at
    with_own = trades.drop(columns=["need_inr"]).assign(known_at=pd.Timestamp("2019-01-01"))
    assert MP.attach_need(with_own, need)[0]["known_at"].eq(pd.Timestamp("2019-01-01")).all()


@pytest.mark.data
def test_the_monday_numbers_on_the_real_bars():
    """2016-2023: a number for every week from Monday 18 Jan 2016 (the first with 10 finished days), 7 of them on a
    Tuesday after a holiday, every one known on its own day; none uses a later bar (three cuts)."""
    from machine.data import bars as BARS
    m5 = BARS.load_bars("5m", "2016", "2023")
    need = MP.weekly_need(m5)
    assert need["day"].iloc[0] == pd.Timestamp("2016-01-18") and len(need) == 415
    assert (pd.DatetimeIndex(need["day"]).dayofweek == 1).sum() == 7
    assert (pd.DatetimeIndex(need["known_at"]).normalize() == pd.DatetimeIndex(need["day"])).all()
    assert need["need_inr"].between(500, 20_000).all()
    assert (need["adr_usd"] == need[["adr5", "adr10"]].max(axis=1)).all()
    for day in ("2017-06-05", "2020-03-16", "2023-10-09"):
        cut = m5.loc[:pd.Timestamp(day) + pd.Timedelta(hours=1, minutes=5)]
        a = MP.weekly_need(cut).set_index("day").loc[pd.Timestamp(day)]
        b = need.set_index("day").loc[pd.Timestamp(day)]
        assert a["need_inr"] == pytest.approx(b["need_inr"]) and a["known_at"] == b["known_at"]
