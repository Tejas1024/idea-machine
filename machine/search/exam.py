"""The search exam (L8b): tests/evals/search.jsonl - 30 questions whose answers ARE in his documents, built from real
re-asks he complained about, and 10 that are not there.

A "found" question passes when answered() says FOUND and its FIRST citation is one of the places that hold the answer
(the talker quotes the first one). A picture counts when it is listed, or when a LEARNED entry that names it is (the
picture is cited through that entry). "Any of the cited places" is printed beside it. A "not found" question passes
when answered() says NOT FOUND; saying FOUND there is a false "found".

    python -m machine.search.exam                         # the exam, with the default embeddings
    python -m machine.search.exam tests/evals/search-dev.jsonl --embeddings off
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from machine import config
from machine.search import index as ix
from machine.search.rank import Answer, Hit, answered

EXAM = config.ROOT / "tests" / "evals" / "search.jsonl"
DEV = config.ROOT / "tests" / "evals" / "search-dev.jsonl"
PASS_FOUND = 27        # the plan's Done-when: >= 27 of 30 found with the right citation ...
PASS_FALSE_FOUND = 2   # ... and <= 2 of 10 false "found"
# The last recorded run of the exam with the local model (plan/work/L8b-notes.md). tests/test_search.py fails if the
# search gets WORSE than this. Raise it when the search gets better - in the same commit, with the new run's notes.
RECORDED = {"found_right_first": 20, "false_found": 0,
            "run": "28 Sep 2026, run 2 of 2 (run 1, the clean one: 19 of 30; run 2 came after a change it motivated)"}


@dataclass(frozen=True)
class Result:
    row: dict
    answer: Answer

    @property
    def expect_found(self) -> bool:
        return self.row["expect"] == "found"

    def _right(self, h: Hit) -> bool:
        expected = self.row.get("cite", [])
        return h.id in expected or any(s in expected for s in h.see)

    @property
    def first_right(self) -> bool:
        return self.answer.found and self._right(self.answer.cites[0])

    @property
    def any_right(self) -> bool:
        return self.answer.found and any(self._right(h) for h in self.answer.cites)

    @property
    def ok(self) -> bool:
        return self.first_right if self.expect_found else not self.answer.found


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run(path: Path = EXAM, index: ix.Index | None = None) -> list[Result]:
    index = index or ix.get()
    return [Result(row, answered(row["q"], index=index)) for row in load(path)]


def score(results: list[Result]) -> dict:
    found = [r for r in results if r.expect_found]
    absent = [r for r in results if not r.expect_found]
    return {"found_right_first": sum(r.first_right for r in found), "found_right_any": sum(r.any_right for r in found),
            "said_found": sum(r.answer.found for r in found), "found_total": len(found),
            "false_found": sum(r.answer.found for r in absent), "absent_total": len(absent)}


def passed(s: dict) -> bool:
    return s["found_right_first"] >= PASS_FOUND and s["false_found"] <= PASS_FALSE_FOUND


def report(results: list[Result], how: str) -> str:
    out = [f"search: {how}", ""]
    for r in results:
        a = r.answer
        mark = "ok  " if r.ok else "MISS"
        if a.found:
            first = a.cites[0]
            got = f"FOUND {first.id} (meaning {first.meaning:.2f}, covered {first.covered:.2f})"
            if r.expect_found and not r.first_right:
                got += f"  | also cited: {', '.join(h.id for h in a.cites[1:]) or '-'}"
        else:
            c = a.closest
            got = f"NOT FOUND (closest {c.id}, meaning {c.meaning:.2f}, covered {c.covered:.2f})" if c else "NOT FOUND"
        out.append(f"{mark} {r.row['id']:<4} {r.row['expect']:<9} {got}")
    s = score(results)
    out += ["", f"found with the right FIRST citation: {s['found_right_first']} of {s['found_total']} "
                f"(right among the cited: {s['found_right_any']}; said FOUND: {s['said_found']})",
            f"false 'found': {s['false_found']} of {s['absent_total']}"]
    if (s["found_total"], s["absent_total"]) == (30, 10):
        out.append(f"pass mark (plan L8b): >= {PASS_FOUND} of 30 and <= {PASS_FALSE_FOUND} of 10 -> "
                   f"{'PASS' if passed(s) else 'NOT YET'}")
    else:
        out.append("(not the exam: no pass mark)")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ with Claude's pick (L8)
@dataclass(frozen=True)
class Picked:
    """One question with Claude's pick among the search's best 5 quotes (machine/talker/pick.py): FOUND = it picked a
    place, the citation = that place."""
    row: dict
    answer: Answer
    pick: str | None
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def expect_found(self) -> bool:
        return self.row["expect"] == "found"

    @property
    def first_right(self) -> bool:
        h = next((h for h in self.answer.hits if h.id == self.pick), None)
        return h is not None and Result(self.row, self.answer)._right(h)

    @property
    def ok(self) -> bool:
        return self.first_right if self.expect_found else self.pick is None


def run_pick(path: Path = EXAM, index: ix.Index | None = None, *, model: str | None = None, batch: bool = False,
             api=None) -> list[Picked]:
    from machine.talker import llm, pick
    index = index or ix.get()
    rows = load(path)
    searched = [(row, answered(row["q"], index=index)) for row in rows]
    reqs = {row["id"]: pick.request(row["q"], a.hits, model) for row, a in searched}
    if batch:
        answers = llm.call_batch(reqs, api=api)
    else:
        answers = {}
        for cid, req in reqs.items():
            try:
                answers[cid] = llm.call(req, api=api)
            except llm.LLMError as err:
                answers[cid] = llm.Answer(None, req.model, llm.Tokens(), 0.0, False, False, "", str(err), req.key())
    return [Picked(row, a, pick.chosen(answers[row["id"]]), answers[row["id"]].paid_now, answers[row["id"]].error)
            for row, a in searched]


def report_pick(results: list[Picked], how: str) -> str:
    out = [f"search: {how}; the citation: Claude's pick among the best 5 (machine/talker/pick.py)",
           "NOTE: this exam is SEEN - the search was run on it 3 times in L8b (plan/work/L8b-notes.md)", ""]
    for r in results:
        mark = "ok  " if r.ok else "MISS"
        got = r.error or (f"picked {r.pick}" if r.pick else "picked none")
        first = r.answer.hits[0].id if r.answer.hits else "-"
        out.append(f"{mark} {r.row['id']:<4} {r.row['expect']:<9} {got}  (the search's first: {first})")
    found = [r for r in results if r.expect_found]
    absent = [r for r in results if not r.expect_found]
    right, false = sum(r.first_right for r in found), sum(r.pick is not None for r in absent)
    out += ["", f"found with the right citation: {right} of {len(found)} (said FOUND: "
                f"{sum(r.pick is not None for r in found)})",
            f"false 'found': {false} of {len(absent)}",
            f"API cost of this run: ${sum(r.cost_usd for r in results):.2f}"]
    if (len(found), len(absent)) == (30, 10):
        out.append(f"pass mark (plan L8b): >= {PASS_FOUND} of 30 and <= {PASS_FALSE_FOUND} of 10 -> "
                   f"{'PASS' if right >= PASS_FOUND and false <= PASS_FALSE_FOUND else 'NOT YET'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default=str(EXAM))
    ap.add_argument("--embeddings", default="auto", choices=("auto", "off", "local", "voyage"))
    ap.add_argument("--pick", action="store_true", help="Claude picks the citation among the best 5 (L8; API)")
    ap.add_argument("--batch", action="store_true", help="with --pick: the Message Batches API (half the price)")
    ap.add_argument("--model", default=None, help="with --pick: the model (default config.TALKER_MODEL)")
    args = ap.parse_args(argv)
    index = ix.get(embeddings=args.embeddings)
    if args.pick:
        picked = run_pick(Path(args.file), index, model=args.model, batch=args.batch)
        print(report_pick(picked, index.how))
        found = [r for r in picked if r.expect_found]
        absent = [r for r in picked if not r.expect_found]
        return 0 if (sum(r.first_right for r in found) >= PASS_FOUND
                     and sum(r.pick is not None for r in absent) <= PASS_FALSE_FOUND) else 1
    results = run(Path(args.file), index)
    print(report(results, index.how))
    return 0 if passed(score(results)) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
