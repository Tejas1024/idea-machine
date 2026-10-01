"""Where the bot files what Tejas sends (L9a). Nothing here rewrites a file of his: new files are created with "x"
(never over an old one), additions are appended.

    strategy/learned-inbox/<YYYY-MM-DD-HHMM>-telegram.md    one lesson or answer, his words verbatim (quoted)
                           <...>-telegram-2.md              a second one in the same minute
    data/telegram/photos/<YYYY-MM-DD-HHMM>-m<id>.jpg        the photo as it came (Tejas's PC only, not in git)
    strategy/pictures/<idea>-<qid>-his-answer.jpg           the cropped copy of a photo that answers a question
    strategy/pictures/<YYYY-MM-DD-HHMM>-telegram-m<id>-his-picture.jpg   ... of any other photo
    plan/work/labels/<word>/replies.jsonl                   his yes / no / not sure on a label chart (L15.1)

The crop: a phone screenshot (at least config.PHONE_SHAPE_MIN times taller than wide) loses the status bar at the top
(the clock, the notification icons, the battery) and the phone's buttons at the bottom - the shares were measured on
his own screenshots (config.py). Any other picture (a chart he drew on and sent as it is) is copied uncut.

Where(): the real places. Where.test(): everything under data/telegram-test/ - Likhith's test bot never writes into
his lesson inbox or his pictures.
"""
from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from machine import config

HEADING_END = "— waiting for a number"


@dataclass
class Where:
    inbox: Path          # lessons and answers wait here for their number
    pictures: Path       # the cropped copies
    bot: Path            # the bot's own files: state.json, messages.jsonl, sent.jsonl, photos/, bot.log
    outbox: Path         # lines TO Tejas (read only): plan/work/telegram-outbox.jsonl
    intake: Path         # new ideas wait here for the intake (L10b)
    answers: Path | None = None  # None = each idea's own answers.md (machine.ideas.store); else one folder of files
    questions: Path | None = None  # the L9b question records (plan/work/question-queue/); None = none here
    shown: Path | None = None      # every picture he was shown (#117): a line when the bot SENDS one (L9b)
    labels: Path | None = None     # his yes / no / not sure on label charts: <word>/replies.jsonl (L15.1); None = none

    @classmethod
    def real(cls) -> "Where":
        return cls(inbox=config.INBOX_DIR, pictures=config.PICTURES_DIR, bot=config.TELEGRAM_DIR,
                   outbox=Path(config.WORK_DIR) / "telegram-outbox.jsonl",
                   intake=Path(config.WORK_DIR) / "intake-queue.jsonl",
                   questions=Path(config.QUESTION_QUEUE_DIR), shown=Path(config.SHOWN_PICTURES),
                   labels=Path(config.LABELS_WORK_DIR))

    @classmethod
    def test(cls, base: Path | None = None) -> "Where":
        """Likhith's test bot: reads the real outbox, writes only under data/telegram-test/."""
        base = Path(base or config.TELEGRAM_TEST_DIR)
        return cls(inbox=base / "learned-inbox", pictures=base / "pictures", bot=base,
                   outbox=Path(config.WORK_DIR) / "telegram-outbox.jsonl", intake=base / "intake-queue.jsonl",
                   answers=base / "answers", questions=None, shown=base / "pictures-shown.jsonl",
                   labels=base / "labels")

    @property
    def photos(self) -> Path:
        return self.bot / "photos"


def show(path: Path) -> str:
    """A path as the repo writes it (strategy/pictures/x.jpg); outside the repo, as it is."""
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def stamp(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%d-%H%M")


def new_file(folder: Path, stem: str, suffix: str) -> Path:
    """A path in folder that does not exist yet: stem + suffix, then stem-2, stem-3 ... (never over an old file)."""
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = folder / f"{stem}{'' if n == 1 else f'-{n}'}{suffix}"
        if not path.exists():
            return path
        n += 1


def _write_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def quote(words: str) -> list[str]:
    """His words, verbatim, as a markdown quote (an empty line stays an empty quote line)."""
    return [f"> {line}" if line.strip() else ">" for line in words.splitlines()] or [">"]


# ======================================================================================================= the inbox

def write_inbox(where: Where, *, kind: str, when: dt.datetime, words: str, notes: list[str]) -> Path:
    """One new inbox file: '# Lesson (Telegram, 29 Sep 2026 14:02) — waiting for a number', his words quoted, then
    one line per note (the picture, what he replied to, the Telegram message). kind: 'Lesson' or 'Answer ...'."""
    path = new_file(where.inbox, f"{stamp(when)}-telegram", ".md")
    head = f"# {kind} (Telegram, {when.day} {when.strftime('%b %Y %H:%M')}) {HEADING_END}"
    body = quote(words) if words else ["> (a picture, no words)"]
    _write_new(path, "\n".join([head, "", *body, "", *[f"- {n}" for n in notes], ""]))
    return path


def add_to_inbox(path: Path, *, words: str, notes: list[str]) -> None:
    """Another photo of the same album goes into the same file (appended, never rewritten)."""
    lines = [""] + (quote(words) + [""] if words else []) + [f"- {n}" for n in notes] + [""]
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# ====================================================================================================== the photos

def crop_phone(im: Image.Image) -> tuple[Image.Image, bool]:
    """Cut the phone's status bar (top) and buttons (bottom) off a screenshot. (picture, was it cut)."""
    w, h = im.size
    if h < config.PHONE_SHAPE_MIN * w:
        return im, False
    top = round(h * config.PHONE_STATUS_BAR_SHARE)
    bottom = h - round(h * config.PHONE_NAV_BAR_SHARE)
    return im.crop((0, top, w, bottom)), True


def save_photo(where: Where, *, when: dt.datetime, message_id: int, data: bytes, ext: str,
               picture_stem: str) -> dict:
    """The photo as it came -> where.photos; a cropped JPEG copy -> where.pictures/<picture_stem>.jpg (-2, -3 ... if
    that name is taken). Returns {"original", "copy", "cut"} as repo paths. A picture that cannot be read is still
    kept as it came, and the copy is None."""
    where.photos.mkdir(parents=True, exist_ok=True)
    original = new_file(where.photos, f"{stamp(when)}-m{message_id}", ext if ext.startswith(".") else "." + ext)
    with open(original, "xb") as fh:
        fh.write(data)
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        cut_im, cut = crop_phone(im)
        copy = new_file(where.pictures, picture_stem, ".jpg")
        buf = io.BytesIO()
        cut_im.convert("RGB").save(buf, "JPEG", quality=92)
        with open(copy, "xb") as fh:
            fh.write(buf.getvalue())
        return {"original": show(original), "copy": show(copy), "cut": cut}
    except Exception as e:                          # a broken picture must not cost him his words
        return {"original": show(original), "copy": None, "cut": False, "error": str(e)[:120]}


def picture_note(saved: dict) -> str:
    if not saved.get("copy"):
        return (f"picture: kept as it came at `{saved['original']}` (Tejas's PC only) - it could not be opened to "
                f"make a copy ({saved.get('error', '')})")
    how = "the phone's top bar and buttons cut off" if saved["cut"] else "not a phone screenshot, so not cut"
    return f"picture: `{saved['copy']}` ({how}); as it came: `{saved['original']}` (Tejas's PC only)"


# ================================================================================================ ideas and answers

def add_idea(where: Where, *, when: dt.datetime, words: str, message_id: int) -> int:
    """A new idea from Telegram waits in the intake queue (L10b runs it). Returns its place in the queue."""
    where.intake.parent.mkdir(parents=True, exist_ok=True)
    line = {"at": when.isoformat(timespec="seconds"), "words": words, "from": f"Telegram message {message_id}",
            "state": "RECEIVED"}
    with open(where.intake, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return sum(1 for ln in where.intake.read_text(encoding="utf-8").splitlines() if ln.strip())


def add_answer(where: Where, *, question: dict, words: str, source: str) -> str | None:
    """His answer, word for word, into the idea's answers.md (append only). Returns where it went, or None when the
    question names no idea with an answers file (the inbox file still has it)."""
    idea, qid = question.get("idea"), question.get("qid") or question.get("id")
    if where.answers is not None:
        where.answers.mkdir(parents=True, exist_ok=True)
        path = where.answers / f"{idea or 'no-idea'}.md"
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(["", f"## {qid}", "", *quote(words), "", f"source: {source}", ""]))
        return show(path)
    if not idea:
        return None
    from machine.ideas import store
    try:
        return show(store.add_answer(idea, words, about=str(qid), source=source))
    except store.StoreError:
        return None
