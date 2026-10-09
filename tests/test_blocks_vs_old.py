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
        assert old >= same > 0 and new >= same          # his 7 Oct wick (L6-differences 8) moves some breaks:
        his, new2, same2 = R[f"break_{side}_his_wick"]  # the old loop with his level gives exactly the new ones,
        assert his == new2 == same2 == new > 0
        differ, moved = R[f"break_{side}_differs"]      # and every block that breaks elsewhere had its level moved
        assert differ == moved > 0
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


def test_the_first_visit_rule_is_the_old_engines():
    """His #68 (§65e): fvg.setup_gaps + fvg.first_visit give the old engine's first_visit answer - filled, spent or
    never, on the same candle - for every old setup of March 2019 whose story starts at its arming (a visit already
    under way before the old engine armed is the old engine's own pre-arming walk, not compared)."""
    from machine.blocks import fvg
    from machine.data.bars import load_bars
    from tests.compare_old_blocks import MONTH, WARM, _in_month, _old, _old_frame

    OLV, OSE = _old()
    bars = load_bars("5m", WARM, MONTH[1])
    df = _old_frame(bars)
    h, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    month = _in_month(bars.index)
    book, A, bias = OLV.build(df)
    compared, statuses = 0, set()
    for s in (s for s in OSE.detect(df, book, A, bias) if month[s["arm_bar"]]):
        side, ab, e = ("buy" if s["dir"] == 1 else "sell"), int(s["arm_bar"]), float(s["entry"])
        gaps = fvg.setup_gaps(h, lo, int(s["sweep_bar"]) + 2, ab, side)
        into = lambda k: (lo[k] <= e if side == "buy" else h[k] >= e) or any(            # noqa: E731
            (lo[k] <= g_hi if side == "buy" else h[k] >= g_lo) for j, g_lo, g_hi in gaps if j < k)
        if any(into(k) for k in range(int(s["fvg_bar"]) + 1, ab + 1)):
            continue
        old = OSE.first_visit(h, lo, dict(s, target=None))
        reach = (lo <= e) if side == "buy" else (h >= e)
        new = fvg.first_visit(h, lo, reach, gaps, ab + 1, min(len(h), ab + 1 + OSE.FIRST_VISIT_MAX), side)
        assert (old[0], old[1]) == (("never" if new[0] == "open" else new[0]), (None if new[0] == "open" else new[1]))
        compared += 1
        statuses.add(old[0])
    assert compared > 30 and {"fill", "spent"} <= statuses, (compared, statuses)
