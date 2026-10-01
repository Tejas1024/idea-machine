"""The question queue on disk (L9b): plan/work/question-queue/

    <idea>-<part>/part.json     one unclear part: its readings, the #129 search, the numbers, its questions (one per
                                moment, each with one picture per reading), the confirm pictures, every change
    <idea>-<part>/part.md       the same for a person to read
    <idea>-<part>/q1-A.jpg ...  the pictures; confirm-A.jpg ... the confirm pictures (another moment, never asked)
    batches/<date>/batch.md     what one weekly batch sent (or would send: --dry-run) and what waits
    <name>.md (+ .jpg)          a question made by hand or by L7b's blocked.py (no part.json): listed, never sent
                                until a session draws its pictures

A part's status: waiting (questions made, not sent) · found (the #129 search found an answer: no question) ·
readings agree (no question) · not drawable · sent · answers differ · confirm sent · locked · confirm said no ·
needs a session · in his documents. part.json is the machine's own record (written whole through a temp file);
his words live in the inbox and the idea's answers.md (append only).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

from machine import config

PART = "part.json"


def queue_dir(base: Path | None = None) -> Path:
    return Path(base or config.QUESTION_QUEUE_DIR)


def folder(idea: str, part: str, base: Path | None = None) -> Path:
    return queue_dir(base) / f"{idea}-{part}"


def load(idea: str, part: str, base: Path | None = None) -> dict | None:
    path = folder(idea, part, base) / PART
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save(rec: dict, base: Path | None = None) -> Path:
    path = folder(rec["idea"], rec["part"], base) / PART
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)
    return path


def records(base: Path | None = None) -> list[dict]:
    root = queue_dir(base)
    out = []
    for p in sorted(root.glob(f"*/{PART}")) if root.is_dir() else []:
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def find(qid: str, base: Path | None = None) -> tuple[dict, dict | None] | None:
    """(the part record, the question) for a question id; (record, None) for the part's confirm question."""
    for rec in records(base):
        if (rec.get("confirm") or {}).get("qid") == qid:
            return rec, None
        for q in rec.get("questions", []) + rec.get("withdrawn", []):
            if q["qid"] == qid:
                return rec, q
    return None


def note(rec: dict, what: str, now: dt.datetime | None = None) -> None:
    rec.setdefault("history", []).append({"at": _iso(now), "what": what})


def set_status(rec: dict, status: str, now: dt.datetime | None = None) -> None:
    if rec.get("status") != status:
        rec["status"] = status
        note(rec, f"status: {status}", now)


def hand_made(base: Path | None = None) -> list[dict]:
    """Questions queued as a .md without a part.json (made by hand or by blocked.py). Never sent by the batch: their
    pictures are drawn by a session first. A .md whose status line says 'replaced' is left out."""
    root = queue_dir(base)
    out = []
    for p in sorted(root.glob("*.md")) if root.is_dir() else []:
        if p.name.lower() == "readme.md":
            continue
        text = p.read_text(encoding="utf-8")
        status = next((ln.split("|")[2].strip() for ln in text.splitlines() if ln.startswith("| status |")), "")
        if "replaced" in status.lower():
            continue
        out.append({"qid": p.stem, "file": p.as_posix(), "status": status,
                    "blocked": "(1)" in text and "BLOCKED" in text,
                    "picture": any(p.with_suffix(s).is_file() for s in (".jpg", ".png"))})
    return out


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _iso(now: dt.datetime | None) -> str:
    return (now or dt.datetime.now().astimezone()).isoformat(timespec="seconds")
