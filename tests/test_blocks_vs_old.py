"""The new core blocks see what the old engine saw on March 2019, except where plan/work/L6-differences.md says why
(L6.1 step 5). Needs the price data."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def R():
    from tests.compare_old_blocks import compare
    return compare()


def test_swings_blocks_breaks_sessions_and_adr_bands_are_the_old_ones(R):
    for side in ("high", "low"):
        old, new, same = R[f"swing_{side}"]
        assert old == new == same > 0
    assert R["swing_known_right"][0] == R["swing_known_right"][1]
    for side in ("sell", "buy"):
        bars, same, _ = R[f"block_{side}"]
        assert bars == same
        old, new, same = R[f"break_{side}"]
        assert old == new == same > 0
    old, new, same = R["sessions"]
    assert old == new == same > 0
    for col in ("day_open", "adr5", "adr10", "up_lo", "up_hi", "dn_hi", "dn_lo"):
        assert R[f"adr_{col}"] == R["bars_month"], col
    assert R["first_visit_same"][0] == R["first_visit_same"][1] > 0


def test_every_old_setup_is_explained(R):
    t = R["setup_tally"]
    assert R["setups"] > 30
    assert not any("not explained" in k or "not among" in k or "a different number" in k or "entry differs" in k
                   for k in t), t
    assert t["C. entry = his near edge"] == R["setups"]


def test_the_ruler_moved_to_broker_days(R):
    # his broker days make the 10-day ADR bigger than the old engine's UTC-day ruler (no Sunday stubs)
    assert 1.0 < R["adr_ruler_ratio"][0] < 1.5
