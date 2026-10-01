"""The weekly batch (L9b step 4 + the dry run, step 6).

    python -m machine.questions batch --dry-run      write this week's batch; send NOTHING
    python -m machine.questions batch                write it and put its lines in the Telegram outbox (the bot sends)

At most config.QUESTION_BUDGET questions (10; Claude's number, a setting), in this order:
  (1) questions that unblock a BLOCKED idea;
  (2) questions of the idea currently running (the idea whose state changed last, not BLOCKED and not finished);
  (3) the oldest waiting.
Inside one priority: the oldest part first, its questions in their order. The rest wait for next week, and the batch
page says so. A question is one moment: its pictures (A, B, C) go together. Labelling (L15.1) is separate.

Sending = lines appended to plan/work/telegram-outbox.jsonl; the bot (L9a) sends each once. A picture counts as SHOWN
(#117) only when the bot has sent it (the line carries "shown"). Questions made by hand without a part.json are
listed, never sent (a session draws their pictures first).
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

from machine import config
from machine.charts import lint
from machine.ideas import store
from machine.questions import ask
from machine.questions import queue as Q

FINISHED = ("REPORTED", "DECIDED")
PRIORITY_WORDS = {1: "(1) it unblocks a BLOCKED idea", 2: "(2) the idea currently running", 3: "(3) the oldest waiting"}


@dataclass
class Item:
    rec: dict
    q: dict
    priority: int

    @property
    def qid(self) -> str:
        return self.q["qid"]


def idea_states() -> dict[str, dict]:
    out = {}
    for idea in store.ideas():
        try:
            out[idea] = store.read_state(idea)
        except Exception:
            out[idea] = {"state": "?", "since": ""}
    return out


def running(states: dict[str, dict]) -> str | None:
    """The idea currently running: not BLOCKED, not finished, its state changed last."""
    live = [(s.get("since", ""), i) for i, s in states.items() if s.get("state") not in ("BLOCKED", *FINISHED)]
    return max(live)[1] if live else None


def waiting(base: Path | None = None, states: dict | None = None) -> list[Item]:
    """Every question not sent yet, in the batch's order."""
    states = idea_states() if states is None else states
    now_running = running(states)
    items = []
    for rec in Q.records(base):
        if rec.get("status") != "waiting":
            continue
        state = (states.get(rec["idea"]) or {}).get("state")
        p = 1 if state == "BLOCKED" or rec.get("unblocks") else (2 if rec["idea"] == now_running else 3)
        for q in rec.get("questions", []):
            if q.get("status") == "waiting":
                items.append(Item(rec, q, p))
    items.sort(key=lambda it: (it.priority, it.rec["made"], it.rec["idea"], it.rec["part"], it.q["n"]))
    return items


MONEY = re.compile(r"₹|\brupees?\b|\bRs\b|\bINR\b|\bprofits?\b|\bloss(es)?\b", re.IGNORECASE)
FIVE = ("same moment", "ends at entry", "question on picture", "lint", "no money")


def five_points(q: dict) -> dict[str, str | None]:
    """The prompt's 5-point check for one question: {check: None when it passes, else what is wrong}. A question is
    ONE picture (his words, inbox 2026-09-29-0912) that shows every reading at the same moment."""
    pics = q["pictures"]
    letters = q.get("letters") or []
    out: dict[str, str | None] = {}
    texts = " ".join(t for p in pics for t in p.get("texts", []))
    missing = [x for x in letters if f"{x}:" not in texts]
    ends = {p.get("end") for p in pics}
    out["same moment"] = (None if len(pics) == 1 and ends == {q["end"]} and len(letters) >= 2 and not missing
                          else f"one picture must show {', '.join(letters) or 'A, B'} at one moment "
                               f"({len(pics)} pictures; missing {', '.join(missing) or 'none'})")
    bad = [str(k) for k, p in enumerate(pics, 1) if not (p.get("ends_at") == p.get("last_candle") == q["end"])]
    out["ends at entry"] = None if not bad else f"picture {', '.join(bad)} does not end at the entry candle"
    bad = [str(k) for k, p in enumerate(pics, 1)
           if not (p.get("question", "").endswith("?") and p["question"] in p.get("texts", []))]
    out["question on picture"] = None if not bad else f"no question written on picture {', '.join(bad)}"
    bad = [str(k) for k, p in enumerate(pics, 1) if not str(p.get("lint", "")).startswith("all ")]
    out["lint"] = None if not bad else f"picture {', '.join(bad)} was not checked by lint"
    bad = [str(k) for k, p in enumerate(pics, 1) if MONEY.search(" ".join(p.get("texts", [])))]
    out["no money"] = None if not bad else f"money on picture {', '.join(bad)}"
    return out


def make(*, dry: bool, budget: int | None = None, now: dt.datetime | None = None, base: Path | None = None,
         outbox: Path | None = None, shown: Path | None = None) -> dict:
    """This week's batch. Writes batches/<date>/batch.md + batch.json; unless dry, the outbox lines and the
    questions' status. Returns what it chose. A question that fails the 5-point check is never sent."""
    now = now or dt.datetime.now().astimezone()
    budget = config.QUESTION_BUDGET if budget is None else budget
    failing = {}
    items = []
    for it in waiting(base):
        problems = {k: v for k, v in five_points(it.q).items() if v}
        if problems:
            failing[it.qid] = problems
        else:
            items.append(it)
    chosen, later = items[:budget], items[budget:]
    for it in chosen:                                    # never the same picture twice (#117): checked again now
        for p in it.q["pictures"]:
            seen = [r for r in lint._shown(Path(shown or config.SHOWN_PICTURES))
                    if r.get("pixel_sha256") == p["pixel_sha256"] or r.get("subject") == p["subject"]]
            if seen:
                raise ValueError(f"{p['file']} was shown already on {seen[0].get('date')} (#117)")
    date = now.strftime("%Y-%m-%d-%H%M")                # one folder and one set of line ids per batch
    folder = Q.queue_dir(base) / "batches" / date
    lines = outbox_lines(chosen, date) if chosen else []
    out = {"date": date, "dry_run": dry, "budget": budget, "chosen": [it.qid for it in chosen],
           "later": [it.qid for it in later], "failing": failing, "hand_made": Q.hand_made(base), "lines": lines,
           "five_points": {it.qid: five_points(it.q) for it in chosen}}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "batch.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                       newline="\n")
    (folder / "batch.md").write_text(page(out, chosen, later), encoding="utf-8", newline="\n")
    if dry or not chosen:
        return out
    path = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
    touched = {}
    for it in chosen:
        it.q["status"] = "sent"
        it.q["batch"] = date
        touched[(it.rec["idea"], it.rec["part"])] = it.rec
    for rec in touched.values():
        Q.set_status(rec, "sent", now)
        Q.note(rec, f"sent in the batch of {date}: " + ", ".join(q["qid"] for q in rec["questions"]
                                                                  if q.get("batch") == date), now)
        ask._save(rec, base)
    return out


def outbox_lines(chosen: list[Item], date: str) -> list[dict]:
    n = len(chosen)
    intro = (f"This week's questions: {n}. ONE picture each.\n"
             "On each picture: the thing marked with ? is what A and B read differently, and what A does and what B "
             "does are written on it.\n"
             "Reply to the picture with A or B. Or draw on it. Or write a few words.\n"
             "If it is already in your documents, reply: it's in my documents.\n"
             "Nothing is locked until you say YES to one more picture.")
    lines = [{"id": f"batch-{date}-intro", "text": intro, "why": "L9b weekly batch"}]
    first_of_part = set()
    for k, it in enumerate(chosen, 1):
        letters = it.q.get("letters") or ["A", "B"]
        (p,) = it.q["pictures"]
        text = f"Question {k} of {n}: reply {' or '.join(letters)} to this picture."
        key = (it.rec["idea"], it.rec["part"])
        if key not in first_of_part:
            first_of_part.add(key)
            text += "\n" + ask.why_ask(it.rec)
        lines.append({"id": f"{it.qid}-{date}", "text": text, "photo": p["file"], "qid": it.qid,
                      "idea": it.rec["idea"], "why": "L9b weekly batch",
                      "shown": {"subject": p["subject"], "pixel_sha256": p["pixel_sha256"], "kind": "question",
                                "file": p["file"], "why": f"L9b batch {date}, {it.qid}"}})
    return lines


def page(out: dict, chosen: list[Item], later: list[Item]) -> str:
    L = [f"# Question batch — {out['date']}" + (" (DRY RUN: nothing sent)" if out["dry_run"] else ""), "",
         f"> At most {out['budget']} questions (config.QUESTION_BUDGET, Claude's number). Priority: "
         + " · ".join(PRIORITY_WORDS.values()) + ". No money in any question (lesson B16).", ""]
    if chosen:
        L += ["| # | question | priority | moment (India time) | pictures | why asked |", "|---|---|---|---|---|---|"]
        for k, it in enumerate(chosen, 1):
            pics = " · ".join(p["file"] for p in it.q["pictures"])
            L.append(f"| {k} | {it.qid} — {it.rec['topic']} | {PRIORITY_WORDS[it.priority]} | {it.q['end_ist']} | "
                     f"{pics} | {ask.why_ask(it.rec)} |")
        L += ["", "**The 5-point check** (every picture of every question):", "",
              "| question | " + " | ".join(FIVE) + " |", "|---|" + "---|" * len(FIVE)]
        for it in chosen:
            L.append(f"| {it.qid} | " + " | ".join("✅" if not out["five_points"][it.qid][k] else "❌"
                                                   for k in FIVE) + " |")
    else:
        L.append("Nothing to ask this week.")
    if out.get("failing"):
        L += ["", "**Not sent - they fail the 5-point check:** " + "; ".join(
            f"{q}: {', '.join(f'{k} ({v})' for k, v in p.items())}" for q, p in out["failing"].items())]
    if later:
        L += ["", f"**Next week (over the budget of {out['budget']}):** " + ", ".join(it.qid for it in later)]
    if out["hand_made"]:
        L += ["", "**Made by hand, not sent (a session draws their pictures first):** "
              + ", ".join(f"{h['qid']} ({h['status'] or 'no status'})" for h in out["hand_made"])]
    L += ["", "## The Telegram lines" + (" it WOULD send" if out["dry_run"] else " (in the outbox; the bot sends)"),
          ""]
    for ln in out["lines"]:
        L.append(f"- `{ln['id']}`" + (f" + picture `{ln['photo']}`" if ln.get("photo") else "") + ": "
                 + ln["text"].replace("\n", " / "))
    return "\n".join(L) + "\n"
