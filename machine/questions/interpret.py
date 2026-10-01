"""His reply to a question -> a proposed reading -> ONE confirm picture -> the lock, only on his YES (L9b step 5).

The bot (L9a) files every reply first (the inbox, the idea's answers.md), then calls on_answer(); a T5 session on the
PC calls it too (`python -m machine.questions answer <qid> "<his words>"`). Code reads only what code can read
without guessing (R13: never guess a meaning - queue it):

  "A" / "b" / "Option C" / "B, because ..."   -> that reading (a letter the question has)
  "it's in my documents" / "already told"     -> no lock; the question + his words -> a fresh search-exam row
  a drawing, or other words                   -> "needs a session": a Claude session reads it (T5 on the PC)
On the CONFIRM picture:
  exactly "yes" (yes / yeah / yep / correct / ok, a full stop allowed)  -> LOCK: the reading goes into form.json
  "no ..."                                    -> not locked; his words wait for a session
  anything else ("yes but ...")               -> not locked; a session reads it

The confirm: when EVERY question of the part has his letter and they are the same letter, ONE confirm picture goes out:
that reading on another day ("I read your answer as B - is this how you trade it?"). Letters that differ -> no
confirm, no lock: a session asks what is different. When every ASK part of the idea is locked, the idea is CONFIRMED &
LOCKED (machine/ideas/store.py: its meaning can no longer change).
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from machine import config
from machine.ideas import store
from machine.questions import ask
from machine.questions.pictures import for_him
from machine.questions import queue as Q

LETTER_ALONE = re.compile(r"^\s*(?:option|picture|answer|reading)?\s*([A-Da-d])\s*[.!]?\s*$", re.IGNORECASE)
LETTER_THEN_WORDS = re.compile(r"^\s*(?:option|picture|answer)?\s*([A-D])\s*[,.:;!)\-–—]\s*(\S.*)$",
                               re.IGNORECASE | re.DOTALL)
YES = re.compile(r"^\s*(yes|yeah|yep|yup|correct|ok|okay)\s*[.!]*\s*$", re.IGNORECASE)
NO = re.compile(r"^\s*no\b", re.IGNORECASE)
DOCUMENTS = re.compile(r"in my (documents|docs|files)|already (told|taught|said|given|explained)|i (told|taught) "
                       r"(you|this)|it'?s in my", re.IGNORECASE)


def read_reply(words: str, has_photo: bool, letters: list[str]) -> tuple[str, str | None]:
    """(kind, letter): kind = letter / documents / drawing / words. Only a letter the question has counts."""
    text = (words or "").strip()
    if DOCUMENTS.search(text):
        return "documents", None
    m = LETTER_ALONE.match(text)
    if m is None:
        m2 = LETTER_THEN_WORDS.match(text)
        m = m2 if m2 and m2.group(1).isupper() else None
    if m and m.group(1).upper() in letters:
        return "letter", m.group(1).upper()
    if has_photo:
        return "drawing", None
    return "words", None


def read_confirm(words: str, has_photo: bool) -> str:
    """yes / no / words. Only a clean yes locks."""
    text = (words or "").strip()
    if has_photo:
        return "words"
    if YES.match(text):
        return "yes"
    if NO.match(text):
        return "no"
    return "words"


def on_answer(question: dict, words: str, has_photo: bool, *, source: str = "", now: dt.datetime | None = None,
              base: Path | None = None, outbox: Path | None = None) -> str | None:
    """Called after his reply is filed. question = the sent record ({"qid", "idea", ...}). Returns one more line for
    the bot's reply to him, or None when the question is not one of L9b's."""
    now = now or dt.datetime.now().astimezone()
    qid = question.get("qid") or question.get("id")
    found = Q.find(qid, base) if qid else None
    if not found:
        return None
    rec, q = found
    answer = {"words": words, "photo": has_photo, "source": source, "at": now.isoformat(timespec="seconds")}
    if q is None:
        return _confirm_reply(rec, answer, now, base)
    if q.get("status") == "withdrawn":
        q.setdefault("late_replies", []).append(answer)
        ask._save(rec, base)
        return ("That picture was taken back (it was not clear). Nothing changes. Please answer the new picture of "
                "this question.")
    letters = q.get("letters") or list(rec["letters"].values())
    kind, letter = read_reply(words, has_photo, letters)
    answer.update(kind=kind, letter=letter)
    q["answer"] = answer
    q["status"] = "answered"
    Q.note(rec, f"{qid}: his reply read as {letter or kind}", now)
    if kind == "documents":
        _fresh_exam_row(rec, q, words, now)
        Q.set_status(rec, "in his documents", now)
        ask._save(rec, base)
        return ("Sorry, I should have found it. I will look again in your documents (a Claude session). "
                "Nothing to do for this question.")
    if kind in ("drawing", "words"):
        Q.set_status(rec, "needs a session", now)
        ask._save(rec, base)
        return ("I cannot read this as a letter by code, and I never guess. A Claude session reads it (T5 on the "
                "PC) and sends you one picture to say yes to.")
    return _after_letter(rec, now, base, outbox)


def _after_letter(rec: dict, now, base, outbox) -> str:
    qs = rec["questions"]
    left = [q for q in qs if not q.get("answer")]
    if left:
        ask._save(rec, base)
        return f"Saved. {len(left)} more question{'s' if len(left) != 1 else ''} about this wait{'s' if len(left) == 1 else ''} for your answer."
    kinds = {q["answer"].get("kind") for q in qs}
    if kinds != {"letter"}:
        Q.set_status(rec, "needs a session", now)
        ask._save(rec, base)
        return "Saved. One of your answers needs a Claude session to read it; it sends you one picture after."
    chosen = {q["answer"]["letter"] for q in qs}
    if len(chosen) > 1:
        Q.set_status(rec, "answers differ", now)
        ask._save(rec, base)
        said = ", ".join(f"{q['answer']['letter']} on question {q['n']}" for q in qs)
        return (f"Your answers differ ({said}). Nothing is locked. A Claude session asks you what is different "
                "between those days.")
    letter = chosen.pop()
    reading = next(r for r, x in rec["letters"].items() if x == letter)
    c = rec["confirm"]
    pic = c["pictures"][reading]
    c.update(proposed=reading, status="sent", sent_at=now.isoformat(timespec="seconds"))
    means = for_him(rec["readings"][reading])
    text = (f"I read your answer as {letter}: {means}.\n"
            "Here it is on another day. Is this how you trade it?\n"
            "Reply YES to lock it. Or NO and a few words why.")
    line = {"id": f"{c['qid']}-{letter}", "text": text, "photo": pic["file"], "qid": c["qid"], "idea": rec["idea"],
            "why": "L9b confirm before lock",
            "shown": {"subject": pic["subject"], "pixel_sha256": pic["pixel_sha256"], "kind": "question",
                      "file": pic["file"], "why": f"L9b confirm {c['qid']}"}}
    path = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    Q.set_status(rec, "confirm sent", now)
    ask._save(rec, base)
    return f"Read as {letter}: {means}. One more picture comes now: say YES to lock it."


def _confirm_reply(rec: dict, answer: dict, now, base) -> str:
    c = rec["confirm"]
    kind = read_confirm(answer["words"], answer["photo"])
    answer["kind"] = kind
    c.setdefault("replies", []).append(answer)
    if c.get("status") != "sent" or not c.get("proposed"):
        ask._save(rec, base)
        return "Saved. (That picture is not waiting for a yes any more.)"
    if kind != "yes":
        c["status"] = "said no" if kind == "no" else "needs a session"
        Q.set_status(rec, "confirm said no" if kind == "no" else "needs a session", now)
        ask._save(rec, base)
        return ("Not locked. A Claude session reads your words and asks again." if kind == "no" else
                "Not locked: only a plain YES locks it. A Claude session reads your words.")
    reading = c["proposed"]
    lock(rec["idea"], rec["part"], reading, now=now, source=answer.get("source", ""))
    c["status"] = "yes"
    Q.set_status(rec, "locked", now)
    ask._save(rec, base)
    letter = rec["letters"][reading]
    return f"Locked: {letter} - {for_him(rec['readings'][reading])}. It will not change for this idea."


def lock(idea: str, part: str, reading: str, *, now: dt.datetime, source: str = "") -> None:
    """His YES: the reading goes into form.json (the part's answer). Every ASK part locked -> CONFIRMED & LOCKED."""
    form = store.load_form(idea)
    parts = [u.model_copy(update={"answer": reading}) if u.id == part else u for u in form.unclear]
    new = form.model_copy(update={"unclear": tuple(parts)})
    store.save_form(type(form).load(json.loads(new.to_json())))
    store.add_answer(idea, f"LOCKED {part} = {reading} (his YES to the confirm picture)", about=f"{part} lock",
                     source=source or "machine.questions.interpret", who="the machine")
    if all(u.answer is not None for u in store.load_form(idea).unclear if u.kind == "ask"):
        store.set_state(idea, "ANSWERED", f"his YES locked {part} = {reading}; every question answered")
        store.set_state(idea, "CONFIRMED & LOCKED", "every answer has his YES on a picture: the meaning is locked "
                        "before any money is seen (lesson 16)",
                        waits_for="the machine: 3 setups it found, in one picture (SHOWN, L10b)")


def _fresh_exam_row(rec: dict, q: dict, words: str, now) -> None:
    """plan/work/L8b-notes.md plan step 2: a question he says is already in his documents becomes a held-out row
    for the search exam. The place he means is filled in by the session that finds it ("cite": null until then)."""
    path = Path(config.SEARCH_FRESH_EXAM)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"q": rec["topic"], "asked": q["qid"], "his_words": words, "date": now.date().isoformat(), "cite": None,
           "search_said": rec.get("search", {}).get("sentence"), "note": "L9b: he said it is in his documents"}
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
