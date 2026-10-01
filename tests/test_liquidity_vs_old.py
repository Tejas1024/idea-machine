"""The L6.2 liquidity blocks see what the old engine saw on March 2019, except where plan/work/L6-differences.md
(its L6.2 part) says why. Needs the price data."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def R():
    from tests.compare_old_liquidity import compare
    return compare()


def test_every_equal_pair_difference_is_explained(R):
    assert R["eq_tally"].get("A. the same pair", 0) > 20
    assert all(k[:2] in ("A.", "B.", "C.", "D.", "E.", "F.", "G.") for k in R["eq_tally"]), R["eq_tally"]


def test_every_matched_level_is_taken_on_the_same_candle(R):
    t = R["sweep_tally"]
    assert t.get("session: taken on the same candle", 0) > 100
    assert t.get("equal: taken on the same candle", 0) > 20
    assert not any("different candle" in k for k in t), t


def test_the_order_is_the_old_one_on_the_old_levels(R):
    total, same = R["recent_order"]
    assert total > 500 and same == total


def test_swing_jobs_his_rules(R):
    roles = R["roles"]
    assert (roles["5m"][2], roles["5m"][3]) == (False, "higher_timeframe_against")   # #91
    assert (roles["1h"][2], roles["1h"][3]) == (False, "always")                     # #93
    assert roles["1d"][2] and roles["1w"][2]                                          # day / week may start
    assert roles["1d"][:2] == (roles["1d"][0], roles["1d"][0])                        # the same daily swings


def test_the_london_levels_are_the_ones_the_old_engine_labelled_london(R):
    """L6-new: liquidity.of_session(.., "London") = the old engine's levels labelled "London session high / low" -
    the same price, known at the same moment, taken on the same candle."""
    t = R["london"]
    assert sum(t.values()) >= 40 and R["london_new_only"] == 0, (t, R["london_new_only"])
    assert all("the same level, taken on the same candle" in k or "never taken in the data, in both" in k
               for k in t), t
    assert any(k.startswith("London high") for k in t) and any(k.startswith("London low") for k in t)
