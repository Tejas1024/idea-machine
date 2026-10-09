"""His management rules as versions (L19c): which rows of his table may be tried, how, and the hedged research run.

His table is plan/work/management-decisions.md (T12, 28 Sep 2026). Every row is ONE of:
  fixed       his rule - machine/rules.py MANAGEMENT says which settings (status "his") or code carry it. No version
              may change it (machine/mcp/versions.py refuses it with its row).
  a version   a TEST row: a setting in machine/settings.py (status "test") that the hedging tester runs in every value
              listed here, and that a research version may set (VERSIONS). The first value is version 1 = what the
              hedging tester did before L19c, so every number made before stays the same.
  not yet     a TEST row that cannot be a version today, with the reason (OTHER): a word of his still open, or a rule
              that would break one of his fixed rows.
A version changes his management through the idea's hedge, not its entry:
    exit.management.<name> = <value>          e.g. {"exit.management.how_many_covers_allowed": 3}
The signals are the same as without it (the compiler never reads exit.management); only the hedging tester does. So a
management version is judged WITH HIS HEDGE, beside version 1 of his management on the same signals (research_run).
One exception: how far past the line the trendline touch entry's hedge sits (M27, trendline_hedge_pad_adr) is made by
the compiler (the signal's own hedge point) - it is an ordinary number of the idea: settings.trendline_hedge_pad_adr.

    from machine.tester import management as MG
    MG.check_changes(form, {"exit.management.on_a_friday": "room"})     # refused with the reason, or what it is
    MG.resolved(form)                                                   # the settings the hedging tester runs with
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from machine import config, rules, settings, trials

TABLE = "plan/work/management-decisions.md"
PATH = "exit.management."
DIMENSION = "his management (dimension 10: the stop and the target are looked after his way - T12 {row})"


class Refused(ValueError):
    """A management change a version may not make. The message says why, in plain words."""


@dataclass(frozen=True)
class Version:
    row: str                                  # the T12 row
    box: str                                  # the four-boxes row that lists the versions to try
    values: tuple | None = None               # the values a version may take (None: a number, see `number`)
    number: str = ""                          # "share" (> 0) · "share0" (>= 0) · "whole" (an integer >= 0)
    needs: dict = field(default_factory=dict)  # {other setting: values it must have} - else this one does nothing
    in_manager: bool = True                   # False: the compiler makes it (an ordinary number of the idea)


VERSIONS: dict[str, Version] = {
    "a_real_stop_instead_of_the_hedge": Version("M02", "B3-11", ("never", "when_small", "always")),
    "a_small_hedge_is_within_adr": Version("M02", "B3-11", number="share",
                                           needs={"a_real_stop_instead_of_the_hedge": ("when_small",)}),
    "how_much_room": Version("M07", "B3-26", number="share"),
    "on_a_friday": Version("M09", "B3-13", ("as_any_day", "room", "hedge")),
    "how_many_covers_allowed": Version("M21", "B3-12", (None, *range(1, 11))),
    "close_the_losing_leg_when_a_trend_runs": Version("M22", "B3-14", (False, True)),
    "trail_when_hour_and_day_split": Version("M24", "-", ("first_target", "hold")),
    "trendline_hedge_pad_adr": Version("M27", "B3-16", number="share0", in_manager=False),
    "news_events": Version("M28", "B3-29", ("forex_factory_red_and_orange", "forex_factory_red")),
    "news_minutes_before": Version("M28", "B3-22", number="whole"),
    "double_profit_reentry": Version("M30", "B3-09", ("off", "retest", "break", "setup")),
    "double_profit_zone_rank": Version("M30", "B3-09", (0, 1, 2),
                                       needs={"double_profit_reentry": ("retest", "break", "setup")}),
    "strong_zone_as_hedge_point": Version("M31", "B3-10", ("off", "nearer", "zone", "zone_break")),
    "strong_zone_hedge_max_adr": Version("M31", "B3-10", number="share",
                                         needs={"strong_zone_as_hedge_point": ("nearer", "zone", "zone_break")}),
    "strong_zone_hedge_rank": Version("M31", "B3-10", (0, 1, 2),
                                      needs={"strong_zone_as_hedge_point": ("nearer", "zone", "zone_break")}),
    "in_a_zone_against_you": Version("M32", "B3-04", ("off", "small")),
    "next_liquidity_max_adr": Version("M35", "B3-26", number="share"),
    "trail_max_targets": Version("M35", "B3-26", number="whole"),
}

# Test rows that are not a version (yet) - each with the reason. Never guessed (R13).
OTHER: dict[str, str] = {
    "M04": "waits for his word 'very strong' (four-boxes B2-03): the hedge at the gap candle (#112, #115) acts only on an "
           "order at a LATER gap of the move, and a later gap is his rule only when the trend is 'very strong' (#124) - "
           "which_gap_of_the_move stays 'first' until he answers",
    "M18": "'one trade per entry type' only means something across his entry types together - the whole strategy, L19d "
           "(one idea is one entry type: one trade at a time already)",
    "M29": "waits for his word 'halfway' (four-boxes B2-05, in the weekly picture batch): after his answer the old "
           "project's two readings (no hedge / re-enter lower) become versions of halfway_reversal (B3-28: once or more)",
    "M33": "the zone grid's hedge is a version: strong_zone_as_hedge_point = 'zone_break' (hedge when price breaks the "
           "zone). Its other half - closing the hedge at the next zone - is not: his newer #47, #113, #134 (M10, fixed) "
           "close a leg only on a full opposite setup, never at a level on its own",
    "M34": "not a version: closing one side 'at a nearer level' while the hedge stays open is a close at a level on its "
           "own, which his fixed M10 forbids (#47, #113). It is not in his words either (his old EA, path 5)",
    "M37": "the research loop itself: a version may now change his entry's numbers AND his management together (this "
           "file) - 'the best combinations are found by testing' (B3-01). L19d runs his whole strategy through it",
}

# Numbers of the old manager that no code reads any more: a version changing one would give the same trades and waste
# a try. Refused, with the rule that replaced what they tuned.
RETIRED: dict[str, str] = {
    "setup_near_level_adr": "a leg closes at the closing setup's own fill price (#47, M10) - there is no 'setup near the "
                            "level' test",
    "setup_wait_bars": "no clock waits for a closing setup (#20, M19); it closes at the setup's own fill (#47)",
    "rearm_adr": "nothing is re-armed: the leg left gets that setup's own fixed hedge (#104, M15)",
    "runner_tolerance_adr": "the old escape's runner; the escape (M22) now hands the leg left to the setup that closed "
                            "the other one (M15)",
    "news_tiny_loss_adr": "'a tiny loss' at news was his #52, replaced by #110 (break-even or any loss -> hedge)",
    "closed_hedge_price": "a closed leg is credited at the closing setup's fill price (lesson B20, M10)",
    "zone_flip_in_management": "the zones' flip is the zone block's own rule (L6b.2, #77, #78); the manager reads it",
    "exit_on_opposite_setup_unhedged": "his rule for it is the trailing one (M23: 'unless an opposite setup forms there')",
}


# ================================================================================================= the checks

def name_of(path: str) -> str | None:
    """The management setting a change path names ("exit.management.on_a_friday" -> "on_a_friday"), else None."""
    if path.startswith(PATH) and path.count(".") == 2:
        return path[len(PATH):]
    return None


def is_hedge(form) -> bool:
    return getattr(form.exit, "kind", None) == "hedge"


def check_value(form, name: str, value: Any, path: str) -> str:
    """One change to his management: what it is (its dimension text), or Refused with the reason."""
    s = settings.ALL.get(name)
    if s is None:
        raise Refused(f"'{path}': {name!r} is not one of his management settings ({TABLE}; machine/settings.py)")
    if name in RETIRED:
        raise Refused(f"'{path}': no code reads {name} any more - {RETIRED[name]}. A version changing it would give the "
                      "same trades.")
    got = rules.managed_by(name)
    if got is not None:
        row, m = got
        raise Refused(f"'{path}': {name} carries his FIXED management rule {row} ({m.taught}): '{m.rule}' - not a "
                      "version (machine/rules.py MANAGEMENT; LEARNED #121)")
    v = VERSIONS.get(name)
    if v is None:
        rows = [r for r, why in OTHER.items() if s.t12 == r]
        why = f" - {OTHER[rows[0]]}" if rows else ""
        raise Refused(f"'{path}': {name} is not a management version{why}")
    if v.in_manager and not path.startswith(PATH):
        raise Refused(f"'{path}': {name} is his management - a version sets it on the hedge: "
                      f"'{PATH}{name}' (the hedging tester reads it; the plain tester never would)")
    if not v.in_manager and path.startswith(PATH):
        raise Refused(f"'{path}': {name} moves the signal's own hedge point, made by the compiler - a version sets it "
                      f"as 'settings.{name}'")
    if v.in_manager and not is_hedge(form):
        raise Refused(f"'{path}': this idea exits with a plain stop - his management (T12) runs only with a hedge")
    _check_kind(v, name, value, path)
    return DIMENSION.format(row=v.row)


def _check_kind(v: Version, name: str, value: Any, path: str) -> None:
    if v.values is not None:
        if not any(value == o and type(value) is type(o) for o in v.values):
            raise Refused(f"'{path}' = {value!r}: {name} can be {', '.join(map(repr, v.values))}")
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise Refused(f"'{path}' = {value!r}: {name} is a number")
    if v.number == "whole" and (not isinstance(value, int) or value < 0):
        raise Refused(f"'{path}' = {value!r}: {name} is a whole number, 0 or more")
    if v.number == "share" and value <= 0:
        raise Refused(f"'{path}' = {value!r}: {name} is a share of a day's range above 0")
    if v.number == "share0" and value < 0:
        raise Refused(f"'{path}' = {value!r}: {name} is a share of a day's range, 0 or more")


def changes_of(changes: dict[str, Any]) -> dict[str, Any]:
    """{setting: value} of the management changes among a version's changes (exit.management.* and the compiler's
    management number settings.trendline_hedge_pad_adr)."""
    out = {}
    for path, value in changes.items():
        n = name_of(path)
        if n is not None:
            out[n] = value
        elif path.startswith("settings.") and VERSIONS.get(path[len("settings."):]) is not None:
            out[path[len("settings."):]] = value
    return out


def in_manager(changes: dict[str, Any]) -> dict[str, Any]:
    """The changes only the hedging tester reads: a version with any is judged WITH HIS HEDGE."""
    return {n: v for n, v in changes_of(changes).items() if VERSIONS[n].in_manager}


def check_needs(form, changes: dict[str, Any]) -> None:
    """A number that acts only when another setting has a certain value is refused without it (the same trades)."""
    have = {**dict(getattr(form.exit, "management", {}) or {}), **changes_of(changes)}
    for name in changes_of(changes):
        for other, ok in VERSIONS[name].needs.items():
            now = have.get(other, settings.ALL[other].value)
            if now not in ok:
                raise Refused(f"{name} acts only when {other} is {' or '.join(map(repr, ok))} (it is {now!r}) - "
                              f"without that this version gives the same trades")


def check_changes(form, changes: dict[str, Any]) -> dict[str, str]:
    """Every management change of a version checked, then their needs. {path: what it is}."""
    out = {}
    for path, value in changes.items():
        n = name_of(path)
        if n is not None or (path.startswith("settings.") and path[len("settings."):] in VERSIONS):
            out[path] = check_value(form, n or path[len("settings."):], value, path)
    check_needs(form, changes)
    return out


def resolved(form) -> settings.Resolved:
    """The settings the hedging tester runs a (compiled) form with: its settings, then its hedge's management."""
    extra = dict(form.exit.management) if is_hedge(form) else {}
    return settings.resolve({**form.settings, **extra})


def describe() -> list[dict]:
    """Every management version a research version may try (the research tools' list_blocks)."""
    out = []
    for name, v in VERSIONS.items():
        s = settings.ALL[name]
        out.append({"name": name, "path": (PATH if v.in_manager else "settings.") + name, "row": v.row,
                    "four_boxes": v.box, "version_1": s.value,
                    "values": list(v.values) if v.values is not None else f"a number ({v.number})",
                    "needs": {k: list(x) for k, x in v.needs.items()}, "who": s.who, "means": s.means})
    return out


# ================================================================================================= the hedged run

@dataclass
class HedgedRun:
    summary: dict                     # this version's management
    version_1: dict                   # version 1 of his management, on the SAME signals
    by_year: dict                     # year -> {"this": Rs, "version_1": Rs, "difference": Rs}
    trial_path: Path | None
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    rules: dict
    acted: dict                       # how often each management rule acted (this version)


_FACTS: dict = {}


def tune_facts(start: str | None = None, end: str | None = None, log=print):
    """His closing setups (every entry of his whose words are settled, M10), his bias, his liquidity, his zones and the
    news on the tune years - made once per process (minutes: his entries are compiled), under today's settings."""
    from machine.ideas.compile import Market
    from machine.tester import engine, hedging
    start = start or str(config.TUNE_YEARS[0])
    end = end or str(config.TUNE_YEARS[-1])
    fp0 = settings.fingerprint(settings.resolve({}))
    key = (start, end, fp0)
    if key not in _FACTS:
        t0 = time.time()
        market = Market.load(start, end, tfs=("1m", "5m", "1h", "1d", "1w"))
        book = engine.bars_book(market.bars("1m"), "5m")
        _FACTS.clear()
        _FACTS[key] = hedging.MarketFacts.build(market, fp0, book)
        log(f"his management's market facts {start}-{end}: {len(_FACTS[key].setups)} closing setups, "
            f"{len(_FACTS[key].news)} news moments, {time.time() - t0:.0f} s")
    return _FACTS[key]


def research_run(signals: list, tf: str, management: dict[str, Any], *, idea_id: str, version_id: str,
                 parent_idea: str | None, note: str, moves: dict | None = None, facts=None, record: bool = True,
                 data: dict | None = None, log=print) -> HedgedRun:
    """One research version WITH HIS HEDGE on the tune years: its signals through the hedging tester under its
    management, and the SAME signals under version 1 of his management (the fair comparison: only the management
    differs). Writes ONE trial file (kind "hedging") - the try. The check years and 2024 on are never read here.
    data: the prices' record for the trial file (default: the real 1-minute prices' content hash)."""
    from machine.tester import hedging
    facts = facts or tune_facts(log=log)
    guards_years(facts)
    base = hedging.replay(signals, facts, tf, hedging.Rules.now(), moves=moves)
    with settings.using(management) as res:
        R = hedging.Rules.now()
        rep = hedging.replay(signals, facts, tf, R, moves=moves)
    this, v1 = hedging.summarize(rep), hedging.summarize(base)
    years = sorted(set(this["per_year"]) | set(v1["per_year"]))
    by_year = {y: {"this": this["per_year"].get(y, {}).get("net_inr", 0.0),
                   "version_1": v1["per_year"].get(y, {}).get("net_inr", 0.0)} for y in years}
    for y in by_year.values():
        y["difference"] = round(y["this"] - y["version_1"], 2)
    did = this.get("test_rows_acted", {})
    path = None
    if record:
        sett = {"idea_id": idea_id, "version_id": version_id, "timeframe": tf,
                "window": {"first_bar": str(facts.book.first), "last_bar": str(facts.book.last)},
                "management": dict(management), "breaks_his_rules": list(res.breaks_his_rules),
                "rules": R.as_dict(), "version_1_rules": hedging.Rules.now().as_dict(), "table": TABLE,
                "closing_setups": {"filled": len(facts.setups), "which": getattr(facts, "closers", {})},
                "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
                "code_sha256": trials.code_sha256([Path(hedging.__file__), Path(__file__)]),
                "signals_sha256": trials.sha256_of([s.model_dump(mode="json") for s in signals])}
        if data is None:
            from machine.data import bars as BARS
            data = {"prices": config.GOLD_M1.relative_to(config.ROOT).as_posix(),
                    "gold_m1_content_sha256": BARS.m1_fingerprint()}
        path = trials.record(kind="hedging", idea_id=idea_id, version_id=version_id, parent_idea=parent_idea,
                             settings=sett, data=data,
                             result={**this, "version_1_of_his_management": v1, "by_year": by_year, "acted": did},
                             note=note)
    return HedgedRun(this, v1, by_year, path, facts.book.first, facts.book.last, R.as_dict(), did)


def guards_years(facts) -> None:
    """The hedged research run reads the tune years only: refused otherwise (the check years open through the report)."""
    lo, hi = pd.Timestamp(facts.book.first), pd.Timestamp(facts.book.last)
    if lo.year < config.TUNE_YEARS[0] or hi.year > config.TUNE_YEARS[-1]:
        raise Refused(f"his management's research run reads {config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]} only; these "
                      f"facts reach {lo} .. {hi}")
