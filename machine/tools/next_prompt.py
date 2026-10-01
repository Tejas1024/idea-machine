"""What can I do now? Reads plan/PROGRESS.md and prints the prompts ready for one person, then the blocked ones.

    python -m machine.tools.next_prompt --who Likhith

Ready = status ⬜, every hard need ✅, and the owner is that person or "either" (the first name in the Owner cell is
the owner). Needs in "(soft: …)" may be missing: the prompt can start and only that step waits (likhith.md R0).
Best first: lowest stage, then table order.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from machine import config

DONE, TODO, TAKEN = "✅", "⬜", "🟡"
PEOPLE = ("Tejas", "Likhith")
ID_RE = re.compile(r"\b[LT]\d+[A-Za-z0-9.\-]*")
SOFT_RE = re.compile(r"\(soft:([^)]*)\)")


@dataclass
class Row:
    order: int
    stage: str
    id: str
    owner: str  # "Tejas", "Likhith", or "either"
    what: str
    needs: list[str]
    soft: list[str]
    runs_on: str
    status: str
    note: str

    @property
    def rank(self) -> tuple[int, int]:
        return (int(self.stage) if self.stage.isdigit() else 99, self.order)


@dataclass
class Item:
    row: Row
    waits: list[str] = field(default_factory=list)  # hard needs not ✅ yet
    soft_missing: list[str] = field(default_factory=list)


@dataclass
class Plan:
    who: str
    ready: list[Item]
    blocked: list[Item]
    taken: list[Item]


def parse_progress(text: str) -> list[Row]:
    """The rows of the first table whose header has Stage, ID, Owner, Needs and Status."""
    lines = text.splitlines()
    for start, line in enumerate(lines):
        head = _cells(line)
        if {"stage", "id", "owner", "needs", "status"} <= {h.lower() for h in head}:
            break
    else:
        raise ValueError("no progress table found (a header with Stage | ID | Owner | … | Needs | … | Status).")
    col = {h.lower(): i for i, h in enumerate(head)}

    def cell(cells: list[str], name: str) -> str:
        i = col.get(name)
        return cells[i] if i is not None and i < len(cells) else ""

    rows: list[Row] = []
    for line in lines[start + 1:]:
        if not line.strip().startswith("|"):
            break
        cells = _cells(line)
        if all(set(c) <= set("-: ") for c in cells):  # the |---|---| line
            continue
        needs_cell = cell(cells, "needs")
        soft = SOFT_RE.search(needs_cell)
        rows.append(Row(
            order=len(rows),
            stage=cell(cells, "stage"),
            id=cell(cells, "id").replace("⭐", "").strip(),
            owner=_owner(cell(cells, "owner")),
            what=cell(cells, "what"),
            needs=ID_RE.findall(SOFT_RE.sub("", needs_cell)),
            soft=ID_RE.findall(soft.group(1)) if soft else [],
            runs_on=cell(cells, "runs on"),
            status=cell(cells, "status"),
            note=cell(cells, "note"),
        ))
    return rows


def _cells(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    return [c.strip() for c in line.strip("|").split("|")]


def _owner(cell: str) -> str:
    first = re.match(r"[A-Za-z]+", cell.replace("*", "").strip())
    if not first:
        return ""
    name = first.group(0)
    return "either" if name.lower() == "either" else name.capitalize()


def plan_for(rows: list[Row], who: str) -> Plan:
    by_id = {r.id: r for r in rows}

    def missing(ids: list[str]) -> list[str]:
        return [n for n in ids if n not in by_id or by_id[n].status != DONE]

    plan = Plan(who, [], [], [])
    for r in sorted((r for r in rows if r.owner in (who, "either")), key=lambda r: r.rank):
        if r.status == TAKEN:
            plan.taken.append(Item(r))
        elif r.status == TODO:
            item = Item(r, missing(r.needs), missing(r.soft))
            (plan.blocked if item.waits else plan.ready).append(item)
    return plan


def render(plan: Plan, rows: list[Row]) -> str:
    by_id = {r.id: r for r in rows}
    width = max([len(i.row.id) for i in plan.ready + plan.blocked + plan.taken] + [2])

    def need(n: str) -> str:
        r = by_id.get(n)
        return f"{n} (not in the table)" if r is None else f"{n} ({r.owner}, {r.status})"

    def line(item: Item, extra: str = "") -> str:
        r = item.row
        text = f"  {r.id:<{width}}  {r.what}"
        if r.runs_on not in ("", "—"):
            text += f"  · runs on {r.runs_on}"
        return text + extra

    out = []
    if plan.ready:
        out.append(f"Ready for {plan.who}, best first. Paste {plan.ready[0].row.id}:")
        for item in plan.ready:
            extra = ""
            if item.soft_missing:
                extra += f"  · soft need {', '.join(item.soft_missing)} not done: that step waits"
            if item.row.note:
                extra += f"  · note: {item.row.note}"
            out.append(line(item, extra))
    else:
        out.append(f"Nothing is ready for {plan.who} right now.")
    if plan.blocked:
        out += ["", "Blocked, waiting for:"]
        out += [line(item, "  · waits for " + ", ".join(need(n) for n in item.waits)) for item in plan.blocked]
    if plan.taken:
        out += ["", f"Taken now {TAKEN} (already started):"]
        out += [line(item) for item in plan.taken]
    return "\n".join(out)


def _person(name: str) -> str:
    for person in PEOPLE:
        if name.strip().lower() == person.lower():
            return person
    raise argparse.ArgumentTypeError(f"--who must be one of {', '.join(PEOPLE)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m machine.tools.next_prompt", description=__doc__.splitlines()[0])
    parser.add_argument("--who", required=True, type=_person, help="Tejas or Likhith")
    parser.add_argument("--file", type=Path, default=config.PROGRESS_FILE, help="default: plan/PROGRESS.md")
    args = parser.parse_args(argv)
    rows = parse_progress(args.file.read_text(encoding="utf-8"))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ⬜ ✅ ₹ on a Windows pipe
    print(render(plan_for(rows, args.who), rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
