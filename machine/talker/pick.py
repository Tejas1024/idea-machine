"""Claude picks the citation among the search's best quotes (L8; the plan in plan/work/L8b-notes.md).

The search alone: FOUND / NOT FOUND is right 38 of 40 times, but its FIRST citation is right only 20 of 30 - the right
place is among its first 3 in 23 of 30. So Claude reads the question and the best 5 quotes and picks the one that
answers it, or says none answers. The talker does this for every term inside its one call (translate.py, `terms`);
this module is the same step alone, for the search exam:

    python -m machine.search.exam --pick            # the exam with Claude's pick (tests/evals/search.jsonl is SEEN)
"""
from __future__ import annotations

from machine import config
from machine.talker import llm

TOOL_NAME = "pick_citation"
SHOWN = 5

SYSTEM = """\
You check whether Tejas's own documents already answer a question, before anyone asks him (his rule #129: a question \
he already answered costs his trust). Tejas is an expert gold trader; the places are his lessons (LEARNED, his exact \
words), the notes written from them, his rules and settings. You get one question and the best places a search \
found. Call pick_citation once: answer = the id of the ONE place that answers the question, or "none" when none \
does. A place that only uses the same words does not answer it; a place that says the point is still open, or asks \
him about it, does not answer it. When two places answer, prefer his own words (LEARNED, the inbox), and the newer \
lesson when they disagree."""


def tool(ids: list[str]) -> dict:
    return {"name": TOOL_NAME, "strict": True,
            "description": "The place that answers the question, or none.",
            "input_schema": {"type": "object", "additionalProperties": False, "required": ["answer", "why"],
                             "properties": {"answer": {"type": "string", "enum": [*ids, "none"]},
                                            "why": {"type": "string", "description": "one short line"}}}}


def request(question: str, hits, model: str | None = None, effort: str = "low") -> llm.Request:
    shown = list(hits)[:SHOWN]
    lines = [f"Question: {question}", "", "Places:"]
    lines += [f'- {h.id} [{h.date or "no date"}]: "{h.quote}"' for h in shown]
    return llm.Request(system=({"type": "text", "text": SYSTEM},), tools=(tool([h.id for h in shown]),),
                       user="\n".join(lines), tool=TOOL_NAME, model=model or config.TALKER_MODEL, effort=effort)


def chosen(ans: llm.Answer) -> str | None:
    if ans.input is None:
        return None
    a = ans.input.get("answer")
    return None if a in (None, "none") else a
