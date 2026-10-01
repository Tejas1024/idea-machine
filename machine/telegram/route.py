"""The ROUTING TABLE (L9a): where one message from Tejas goes. Checked top to bottom, the first row that fits wins:

    1. a reply to one of the bot's QUESTIONS        -> an answer   (the question loop reads it, L9b; on a LABEL chart
                                                                    his yes / no / not sure -> the label file, L15.1)
    2. a photo, or an edited message                -> a lesson    (a photo never starts a command or an idea)
    3. exactly STOP / START / STATUS                -> a command
    4. starts with "IDEA" and has words after it    -> the intake  (L10b runs it)
    5. anything else                                -> a lesson in strategy/learned-inbox/
    no words and no picture (a voice note, a sticker) -> unreadable: the bot says it reads only words and pictures

"Exactly" = the whole message is the one word; capitals and one full stop or "!" at the end do not matter ("Stop."
from his phone's voice typing is STOP; "stop sending charts" is a lesson). "IDEA" the same: "Idea: buy when ..." is
an idea; "Ideally ..." is not. Both are Claude's reading of the prompt's "exactly" and "starting". A question is a
message the bot sent from an outbox line that had a "qid" (data/telegram/sent.jsonl remembers its message id).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

ANSWER, IDEA, COMMAND, LESSON, UNREADABLE = "answer", "idea", "command", "lesson", "unreadable"
COMMANDS = ("STOP", "START", "STATUS")
_COMMAND = re.compile(r"\s*(stop|start|status)\s*[.!]?\s*", re.IGNORECASE)
_IDEA = re.compile(r"\s*idea\b[\s:.,;\-–—]*", re.IGNORECASE)


@dataclass
class Message:
    """What the bot needs from one Telegram update."""
    update_id: int
    message_id: int
    chat_id: str
    date: dt.datetime                  # when he SENT it, on this PC's clock (India time on his PC)
    words: str                         # the text, or the photo's caption ("" if none)
    photo_id: str | None = None        # the biggest size of a photo, or an image sent as a file
    media_group_id: str | None = None  # photos sent together share this
    reply_to_id: int | None = None     # the message he replied to
    reply_to_words: str = ""           # ... and what it said
    edited: bool = False
    other: str = ""                    # "voice note", "sticker", ... when there are no words and no picture


@dataclass
class Route:
    kind: str
    why: str
    command: str | None = None         # STOP / START / STATUS
    words: str = ""                    # the idea's words, without "IDEA"
    question: dict | None = None       # the sent question he answered: {"id", "qid", "idea", ...}


def parse(update: dict) -> Message | None:
    """One Telegram update -> a Message (None for an update that is not a message, e.g. a channel post)."""
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    photo_id = None
    if msg.get("photo"):
        photo_id = msg["photo"][-1]["file_id"]                        # the last size is the biggest
    elif (msg.get("document") or {}).get("mime_type", "").startswith("image/"):
        photo_id = msg["document"]["file_id"]                        # a picture sent "as a file", uncompressed
    other = ""
    if not photo_id and not (msg.get("text") or msg.get("caption")):
        other = next((name for key, name in (("voice", "voice note"), ("audio", "audio"), ("video", "video"),
                                             ("video_note", "video"), ("sticker", "sticker"),
                                             ("document", "file"), ("location", "location"),
                                             ("contact", "contact")) if key in msg), "message with no words")
    reply = msg.get("reply_to_message") or {}
    return Message(
        update_id=int(update["update_id"]), message_id=int(msg["message_id"]),
        chat_id=str((msg.get("chat") or {}).get("id", "")),
        date=dt.datetime.fromtimestamp(int(msg.get("date", 0))).astimezone(),
        words=(msg.get("text") or msg.get("caption") or "").strip(),
        photo_id=photo_id, media_group_id=msg.get("media_group_id"),
        reply_to_id=reply.get("message_id"), reply_to_words=(reply.get("text") or reply.get("caption") or ""),
        edited="edited_message" in update and "message" not in update, other=other)


def route(msg: Message, questions: dict[int, dict]) -> Route:
    """questions: {Telegram message id of a question the bot sent: its sent-log record}."""
    if not msg.words and not msg.photo_id:
        return Route(UNREADABLE, f"a {msg.other or 'message'} has no words or picture to read")
    if msg.reply_to_id is not None and msg.reply_to_id in questions:
        q = questions[msg.reply_to_id]
        return Route(ANSWER, f"a reply to question {q.get('qid')}", question=q)
    if msg.photo_id:
        return Route(LESSON, "a photo never starts a command or an idea")
    if msg.edited:
        return Route(LESSON, "an edited message never starts a command or an idea")
    m = _COMMAND.fullmatch(msg.words)
    if m:
        return Route(COMMAND, "the whole message is a command word", command=m.group(1).upper())
    m = _IDEA.match(msg.words)
    if m and msg.words[m.end():].strip():
        return Route(IDEA, 'it starts with "IDEA"', words=msg.words[m.end():].strip())
    return Route(LESSON, "not a reply to a question, not a command, not an idea")
