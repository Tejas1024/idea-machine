"""Ideas the old project already tested (L5, step 3): reads plan/work/prior-tries.csv (built in L5-prior).

An idea is "already seen" on the judged years when some row of the CSV (Claude's reading of the match that L5-prior
suggested — plan/work/prior-tries.md, "How the referee reads"):
  - has the same set of blocks, leaving out the management names (mgmt:...), and
  - shares a timeframe, and
  - has the same kind of entry (gap:near and gap:far are both "gap"; a row with entry "any" matches every kind), and
  - looked at one of the judged years.
A row whose only block is `his_entries` (the old test ran a management rule on all of his entries) matches an idea
that uses the same management rule on an entry the list already holds.
Rows that share the entry kind and a timeframe and differ by blocks are printed as "close relatives" — not seen.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from machine import config

HIS_ENTRIES = "his_entries"


@dataclass(frozen=True)
class IdeaKey:
    """What an idea IS, in the block names of prior-tries.csv (L6.1 block names)."""
    blocks: frozenset[str]
    timeframes: frozenset[str]
    entry: str                                  # "gap:near", "line:touch", "market:open", "none", ...
    mgmt: frozenset[str] = frozenset()          # management rule names, without "mgmt:"

    @classmethod
    def make(cls, blocks, timeframes, entry: str, mgmt=()) -> "IdeaKey":
        def parts(value, sep: str) -> set[str]:
            items = [value] if isinstance(value, str) else list(value)
            return {p.strip() for item in items for p in item.split(sep) if p.strip()}

        names = parts(blocks, "+")
        mg = {b[5:] for b in names if b.startswith("mgmt:")} | {m.removeprefix("mgmt:") for m in mgmt}
        if not entry or not entry.strip():
            raise ValueError("an idea key needs its entry kind (e.g. 'gap:near', 'market:open')")
        return cls(frozenset(b for b in names if not b.startswith("mgmt:")), frozenset(parts(timeframes, "|")),
                   entry.strip(), frozenset(mg))

    def as_dict(self) -> dict:
        return {"blocks": sorted(self.blocks), "timeframes": sorted(self.timeframes), "entry": self.entry,
                "mgmt": sorted(self.mgmt)}


@dataclass(frozen=True)
class Row:
    idea_key: str
    blocks: frozenset[str]
    mgmt: frozenset[str]
    timeframes: frozenset[str]
    entry: str
    tries: int
    years: tuple[int, int]
    source: str


@dataclass
class Seen:
    seen: bool
    rows: list[Row] = field(default_factory=list)          # the rows that make it seen
    relatives: list[Row] = field(default_factory=list)     # same entry kind and timeframe, other blocks — not seen
    judged_years: tuple[int, ...] = ()

    @property
    def tries(self) -> int:
        return sum(r.tries for r in self.rows)

    def text(self) -> str:
        if not self.seen:
            rel = (f" Close relatives (other blocks, not the same idea): {', '.join(r.idea_key for r in self.relatives)}."
                   if self.relatives else "")
            return "Already seen by the old project: no — no row of plan/work/prior-tries.csv is this idea." + rel
        rows = "; ".join(f"{r.idea_key} ({r.tries} tries, {r.years[0]}-{r.years[1]})" for r in self.rows)
        return (f"Already seen by the old project: YES — {rows}. These years are not fresh for this idea; only new "
                "prices can judge it (the forward test, L17c).")

    def as_dict(self) -> dict:
        return {"seen": self.seen, "rows": [r.idea_key for r in self.rows], "tries": self.tries,
                "relatives": [r.idea_key for r in self.relatives], "judged_years": list(self.judged_years)}


def entry_kind(entry: str) -> str:
    return entry.split(":", 1)[0].strip()


# ---- an idea form -> its key in the old list's names (L10) ---------------------------------------------------------
# The compiler's block names (machine/ideas/compile.py BLOCKS) in the names of prior-tries.csv (prior-tries.md, "How
# the referee reads"). Claude's reading of which old name each block is. A block that is several old names at once
# lists them all (the zone entry = a strong zone + the entry + the obvious level taken beyond it, as its old rows
# name it). Blocks L6b.1-L6b.4 built were added by L14.1 (1 Oct 2026): idea #2's ADR-zone step had no name here, so
# its key - and the check step that reads it - failed. `all_time_high` and `one_minute` have no old row: their own
# new names, so an idea that uses them is never "seen" by a row that never tested them.
FORM_BLOCKS = {
    "sweep": "liquidity_sweep",                 # price takes a liquidity level with a wick
    "break_of_structure": "break_of_structure",
    "gap_of_the_move": "fvg",
    "fib_move": "fib_levels",
    "equal_levels": "equal_highs_lows",
    "level_waiting": None,                      # the target side: named by the kinds of level it waits for (below)
    "comes_back_to": None,                      # a check on a level another step found: adds no block of its own
    "bias": "htf_bias",                         # his direction on the hour / day / week chart (L6b.3)
    "hammer": "hammer",
    "premium_discount": "adr",                  # his ADR zones (strategy.md section 7) - the old list's ADR band
    # L6-new (idea #4): a CLOSE beyond the ADR zone, then the trade. No old row traded that: `band_pass_through_filter`
    # (understanding §33g) only DROPPED setups whose fill candle closed beyond the band - it stays a close relative.
    "adr_zone_break": ("adr", "adr_zone_break"),
    "all_time_high": "all_time_high",
    "one_minute": "one_minute",                 # the 1-minute check before a 5-minute gap entry (L6b.4)
    "trendline_touch": "trendlines",            # his Type 2 (L19b's key for it: trendlines, line:touch)
    "triangle": "triangles",
    "false_breakout": "false_breakout",
    "wedge": "wedges",
    "zone": "strong_zones",                     # a check: price is at one of his strong zones
    "zone_break": ("strong_zones", "zone_break"),
    "zone_entry": ("strong_zones", "zone_entry", "obvious_liquidity"),
}
KIND_BLOCKS = {"equal": "equal_highs_lows", "session": "sessions"}
# A limit order at the price one of these steps found: the entry's kind in the old list (anything else: line:touch).
ENTRY_KINDS = {"fib_move": "fib:limit", "zone_entry": "zone:edge", "zone_break": "zone:retest",
               "false_breakout": "pattern:false_break", "wedge": "pattern:wedge"}


def key_for_form(form) -> IdeaKey:
    """The idea's key for the "already seen" check, from its form (machine.ideas.form.Idea): its steps' blocks (a
    step that only takes EQUAL highs / lows, or waits for them, adds equal_highs_lows), its timeframes, its entry."""
    blocks: set[str] = set()
    for s in form.steps:
        if s.block not in FORM_BLOCKS:
            raise ValueError(f"step {s.id}: no old-list name for the block {s.block!r} - add it to "
                             "machine/referee/prior.py FORM_BLOCKS")
        names = FORM_BLOCKS[s.block]
        blocks |= {names} if isinstance(names, str) else set(names or ())
        kinds = (s.params or {}).get("kinds") or []
        blocks |= {KIND_BLOCKS[k] for k in kinds if k in KIND_BLOCKS}
    by_ref = {s.id: s.block for s in form.steps}
    e = form.entry
    if e.how == "limit":
        step, _, part = str(e.at).partition(".")
        block = by_ref.get(step)
        entry = f"gap:{part or 'near'}" if block == "gap_of_the_move" else ENTRY_KINDS.get(block, "line:touch")
    else:
        entry = "market:open"
    tfs = {form.timeframe} | {s.tf for s in form.steps}
    mgmt = {"hedge"} if form.exit.kind == "hedge" else set()
    return IdeaKey(frozenset(blocks), frozenset(tfs), entry, frozenset(mgmt))


def describe(idea_key: str, path: Path | None = None) -> str:
    """The plain words of an old try (the second column of its row in prior-tries.md), for the report - never the
    short name alone (#113: no invented short names). '' when the row is not found."""
    md = Path(path or config.PRIOR_TRIES_MD)
    if not md.is_file():
        return ""
    for line in md.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 2 and cells[0] == f"`{idea_key}`":
            return cells[1]
    return ""


def closest(key: IdeaKey, seen: "Seen", n: int = 3) -> list[Row]:
    """The old tries most like this idea (the report's "close relatives"): first those whose blocks are all blocks
    of this idea (an old idea that is a WIDER version of this one), then by how many blocks they share, then the same
    entry detail (gap:near = gap:near), then the entry itself before its management tests, then by tries. Relatives
    only: a row that makes the idea "seen" is not here."""
    def rank(r: Row):
        return (r.blocks <= key.blocks, len(r.blocks & key.blocks), r.entry == key.entry, not r.mgmt, r.tries)

    out, keys = [], set()
    for r in sorted(seen.relatives, key=rank, reverse=True):
        if r.idea_key not in keys:
            keys.add(r.idea_key)
            out.append(r)
    return out[:n]


def load_rows(path: Path | None = None) -> list[Row]:
    return list(_load(str(path or config.PRIOR_TRIES_CSV)))


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[Row, ...]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            parts = [b for b in r["blocks"].split("+") if b]
            first, last = (int(y) for y in re.fullmatch(r"(\d{4})-(\d{4})", r["years"]).groups())
            rows.append(Row(
                idea_key=r["idea_key"],
                blocks=frozenset(b for b in parts if not b.startswith("mgmt:")),
                mgmt=frozenset(b[5:] for b in parts if b.startswith("mgmt:")),
                timeframes=frozenset(r["timeframe"].split("|")),
                entry=r["entry"], tries=int(r["tries"]), years=(first, last), source=r["source"]))
    return tuple(rows)


def already_seen(key: IdeaKey, judged_years, path: Path | None = None) -> Seen:
    """Is this idea in the old project's list, on any of the judged years?"""
    judged = tuple(sorted(int(y) for y in judged_years))
    rows = load_rows(path)
    kind = entry_kind(key.entry)

    def same_entry(r: Row) -> bool:
        return r.entry == "any" or entry_kind(r.entry) == kind

    def shares_tf(r: Row) -> bool:
        return bool(r.timeframes & key.timeframes)

    def on_years(r: Row) -> bool:
        return any(r.years[0] <= y <= r.years[1] for y in judged)

    his_entry_sets = {r.blocks for r in rows if HIS_ENTRIES not in r.blocks}
    hits, relatives = [], []
    for r in rows:
        if not (shares_tf(r) and same_entry(r) and on_years(r)):
            continue
        if r.blocks == key.blocks:
            hits.append(r)
        elif r.blocks == {HIS_ENTRIES} and r.mgmt and r.mgmt <= key.mgmt and key.blocks in his_entry_sets:
            hits.append(r)
        elif r.blocks & key.blocks:
            relatives.append(r)
    return Seen(bool(hits), hits, relatives, judged)
