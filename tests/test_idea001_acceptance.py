"""L6.2 acceptance: idea #1's examples (T3) against the new liquidity blocks - every "yes" found, and every "no" that
still fires is a known miss with its question queued (plan/work/L6-misses/README.md). Needs the price data."""
from __future__ import annotations

import pytest

from machine import config

pytestmark = pytest.mark.data

MISSES_KNOWN = {3, 16, 34, 35, 36, 37, 38}     # his "no" - every one says the grab ran too far (L7b) or the lows


@pytest.fixture(scope="module")
def R():
    from tests.idea001_acceptance import run
    return run(write=False)


def test_every_yes_is_found(R):
    yes = [e for e in R["examples"] if e.answer == "yes"]
    assert len(yes) == 24                                                   # 10 in round 3 + 14 in round 1
    missed = [e.k for e in yes if not (e.notes["fires"] and e.notes["same_target"])]
    assert missed == [], missed


def test_the_no_that_still_fire_are_the_known_misses_and_their_question_is_queued(R):
    fired = {e.k for e in R["examples"] if e.answer == "no" and e.notes["fires"]}
    assert fired <= MISSES_KNOWN, fired - MISSES_KNOWN
    q = config.WORK_DIR / "question-queue" / "L6.2-Q1-lows-too-near.md"
    assert q.is_file() and q.with_suffix(".jpg").is_file()
    assert "₹" not in q.read_text(encoding="utf-8")                        # lesson B16: no money in a question


def test_too_near_as_a_rule_would_lose_his_yes(R):
    r = {x["reading"].split(" = ")[0]: x for x in R["readings"]}
    near = next(v for k, v in r.items() if "too near" in k)
    assert near["r3_yes"][0] < near["r3_yes"][1]                            # #24, #25, #33 lost
    assert near["r3_no"][0] > 0                                             # and his no's still fire
