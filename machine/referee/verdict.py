"""The answer (L5, step 8): works / chance / chance — needs fresh years / not enough trades to tell.

The rules are FIXED IN ADVANCE: RULES is printed in every report and its fingerprint (RULES_SHA256) goes into every
referee trial file. tests/test_referee.py pins the fingerprint, so changing a rule after seeing a result shows in git.
The numbers come from machine/config.py (the plan's numbers, likhith.md L5 — Claude's, not Tejas's rules).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from machine import config

WORKS = "works"
FRESH = "chance — needs fresh years"
FEW = "not enough trades to tell"
CHANCE = "chance"
WORDS = (WORKS, FRESH, FEW, CHANCE)

RULES = f"""THE RULES (fixed before any result was seen; the same for every idea)
"{WORKS}" needs ALL of these, on the check years {config.CHECK_YEARS[0]}-{config.CHECK_YEARS[-1]} (opened once per idea):
  1. The luck test: the idea's weeks are drawn again {config.BOOTSTRAP_RESAMPLES:,} times in a world with no edge;
     that world gives this many rupees a trade less than {round(100 * config.P_LIMIT)} times in 100.
  2. It earns more a trade than at least {round(100 * config.RANDOM_BEAT_SHARE)} of 100 random-entry runs ({config.RANDOM_RUNS} runs: same exits,
     distances, cost and trades per month; random times and random limit levels).
  3. It earns more a trade than at least {round(100 * config.FAKE_LEVELS_BEAT_SHARE)} of 100 of its fake levels (its level moved up and down by
     {min(abs(s) for s in config.FAKE_LEVEL_STEPS):g} to {max(abs(s) for s in config.FAKE_LEVEL_STEPS):g} stop distances). An idea with no level: does not apply.
  4. It still earns more than ₹0 a trade without its best {config.WITHOUT_BEST} trades.
  5. At least {config.MIN_TRADES} trades.
  6. The old project did not already test this idea on these years (plan/work/prior-tries.csv).
"{FRESH}": the old project already looked at these years for this idea. Only new prices can judge it (a forward test).
"{FEW}": even an edge of ₹{config.POWER_NEEDED_EDGE} a trade would be caught less than {round(100 * config.POWER_NEEDED_SHARE)} times in 100 with this many trades.
"{CHANCE}": everything else. It could be luck.
Later versions of an idea whose check years were already opened are judged the same way, year by year inside
{config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]}: each judged year's version is chosen only on the years before it."""

RULES_SHA256 = hashlib.sha256(RULES.encode("utf-8")).hexdigest()


@dataclass
class Evidence:
    trades: int
    per_trade: float | None
    p: float                                 # the luck test's one-sided p
    random: dict                             # random_entries.summary
    fake: dict                               # fake_levels.summary
    without_best: dict                       # stats.without_best
    seen: bool
    needed_edge_caught: float                # power: share of Rs 100 edges caught at this trade count
    judged_on: str = ""


@dataclass
class Check:
    number: int
    name: str
    passed: bool
    shown: str


@dataclass
class Answer:
    word: str
    checks: list[Check] = field(default_factory=list)
    why: str = ""
    rules_sha256: str = RULES_SHA256

    def as_dict(self) -> dict:
        return {"word": self.word, "why": self.why, "rules_sha256": self.rules_sha256,
                "checks": [{"number": c.number, "name": c.name, "passed": c.passed, "shown": c.shown}
                           for c in self.checks]}


def _inr(x: float | None) -> str:
    if x is None:
        return "no trades"
    return f"-₹{-x:,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def checks(ev: Evidence) -> list[Check]:
    r, f, wb = ev.random, ev.fake, ev.without_best
    luck_ok = ev.per_trade is not None and ev.per_trade > 0 and ev.p < config.P_LIMIT
    rand_ok = (r["runs_with_trades"] >= max(1, r["runs"] // 2)
               and r["beaten_share"] >= config.RANDOM_BEAT_SHARE and ev.per_trade is not None)
    if f.get("applies"):
        fake_ok = f["compared"] > 0 and f["beaten_share"] >= config.FAKE_LEVELS_BEAT_SHARE
        fake_shown = (f"it beats {f['beaten']} of its {f['compared']} fake levels "
                      f"(needs {round(100 * config.FAKE_LEVELS_BEAT_SHARE)} in 100)")
    else:
        fake_ok, fake_shown = True, f"does not apply — {f.get('why', 'no level')}"
    best_ok = wb["per_trade"] is not None and wb["per_trade"] > 0
    return [
        Check(1, "luck test", luck_ok,
              f"{_inr(ev.per_trade)} a trade; a world with no edge gives this much {100 * ev.p:.1f} times in 100 "
              f"(needs under {round(100 * config.P_LIMIT)}, and above ₹0)"),
        Check(2, "random entries", rand_ok,
              f"it beats {100 * r['beaten_share']:.0f} of 100 random runs (needs {round(100 * config.RANDOM_BEAT_SHARE)}); "
              f"random runs: middle {_inr(r['per_trade_median'])} a trade, {r['trades_median']} trades "
              f"(this idea {ev.trades})"),
        Check(3, "fake levels", fake_ok, fake_shown),
        Check(4, f"without the best {wb['left_out']}", best_ok,
              f"{_inr(wb['per_trade'])} a trade without its best {wb['left_out']} trades (needs above ₹0)"),
        Check(5, "trades", ev.trades >= config.MIN_TRADES, f"{ev.trades} trades (needs {config.MIN_TRADES})"),
        Check(6, "not already seen", not ev.seen,
              "the old project already tested this idea on these years" if ev.seen else "not in the old project's list"),
    ]


def decide(ev: Evidence) -> Answer:
    """The one answer, in this order: works (all 6 checks) · needs fresh years (already seen) · not enough trades
    (a Rs 100 edge caught less than half the time) · chance."""
    cs = checks(ev)
    if all(c.passed for c in cs):
        return Answer(WORKS, cs, "all 6 checks passed")
    if ev.seen:
        return Answer(FRESH, cs, "the old project already looked at these years for this idea")
    if ev.needed_edge_caught < config.POWER_NEEDED_SHARE:
        return Answer(FEW, cs, f"a ₹{config.POWER_NEEDED_EDGE} edge would be caught only "
                               f"{100 * ev.needed_edge_caught:.0f} times in 100 with {ev.trades} trades")
    failed = ", ".join(c.name for c in cs if not c.passed)
    return Answer(CHANCE, cs, f"failed: {failed}")
