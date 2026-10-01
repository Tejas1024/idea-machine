"""L6b.3's blocks see what the old engine saw on 2016-2020, except where plan/work/L6b3-differences.md says why.
Needs the price data (a few minutes: the old engine's hour bias and a hammer check on every candle)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def R():
    from tests.compare_old_bias import compare
    return compare()


def test_the_all_time_high_hammer_and_adr_position_are_the_old_ones(R):
    assert R["ath"]["same"] == R["ath"]["days"] and R["ath"]["first_new"] == "2020-07-27"
    for side in ("buy", "sell"):
        hm = R[f"hammer_{side}"]
        assert hm["same"] >= R["bars"] - 2 and hm["old"] > 100          # the same rule on the same ruler
        assert hm["new_his_ruler"] < hm["old"]                           # his broker-day ruler is bigger
    assert R["adr_share"]["same"] == R["adr_share"]["checked"] > 1000


def test_the_bias_agrees_where_both_say_up_or_down(R):
    for tf, least in (("1h", 0.85), ("1d", 0.95)):
        b = R[f"bias_{tf}"]
        assert b["only_old"] == b["only_new"] == 0
        assert b["both_said_same"] >= least * b["both_said"] > 0


def test_his_daily_answers_in_the_tune_years_at_least_as_the_old_engine(R):
    old = sum(r[5] for r in R["answers"])
    new = sum(r[6] for r in R["answers"])
    assert len(R["answers"]) == 8 and new >= old
