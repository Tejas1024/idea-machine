"""The ONE Telegram bot (L9a). It runs only on Tejas's PC and is the only program that reads his chat (likhith.md
B12: two readers steal each other's messages; Telegram deletes what one of them has read).

    .venv\\Scripts\\python -m machine.telegram.bot                  run it (leave it running)
    .venv\\Scripts\\python -m machine.telegram.bot --once           file what waits, send the outbox, stop
    .venv\\Scripts\\python -m machine.telegram.bot --test           Likhith's own test bot (TELEGRAM_TEST_BOT_TOKEN +
                                                                   TELEGRAM_TEST_CHAT_ID); writes only in data/telegram-test/
    .venv\\Scripts\\python -m machine.telegram.bot --copy-old-keys  copy the old bot's key + chat id into .env (never shown)
    .venv\\Scripts\\python -m machine.telegram.bot --install        start it by itself at logon (Task Scheduler)

Every message from his chat id (any other chat is ignored) goes through the routing table (machine/telegram/route.py)
and is FILED FIRST: the lesson inbox, his answers, the intake queue (machine/telegram/inbox.py). Only then does the
bot move past it on Telegram (a crash before that means Telegram gives the message again - never lost), and then it
REPLIES to his message saying where it was filed. A message it has filed once is never filed twice.

Lines TO him: each line of plan/work/telegram-outbox.jsonl ({"id", "text"}, optional "photo", "qid", "idea") whose id
is not in data/telegram/sent.jsonl yet is sent, then its id is added there. A line with a "qid" is a QUESTION: a reply
to it is his answer. A LABEL chart's question (qid LBL-..., L15.1): his yes / no / not sure goes to
plan/work/labels/<word>/replies.jsonl, not the lesson inbox; anything more he writes goes to the inbox as well.
STOP pauses these lines (the bot still files everything he types); START sends them again.

The bot's own files (data/telegram/, not in git): state.json (where it is on Telegram, STOP or not), messages.jsonl
(every message it filed), sent.jsonl, photos/, failed/ (an update the bot could not file, kept whole), bot.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from machine import config
from machine.telegram import inbox
from machine.telegram import route as R
from machine.telegram.api import TelegramAPI, TelegramError

OLD_KEYS = Path("old") / "ml" / "agent" / "agent_config.json"   # the old bot's key + chat id (read, never imported)
STOP_KINDS = ("conflict", "unauthorized")                        # errors that stop the bot; the rest are retried


class Bot:
    def __init__(self, tg, chat_id: str, where: inbox.Where, *, echo: bool = True, kick=None):
        """kick: starts the intake (L10b) as its own program - machine.intake.state.spawn for the real bot; None =
        never (the test bot, the tests)."""
        if not str(chat_id).strip():
            raise ValueError("no chat id: TELEGRAM_CHAT_ID in .env (TELEGRAM_TEST_CHAT_ID with --test)")
        self.tg, self.chat_id, self.where, self.echo = tg, str(chat_id).strip(), where, echo
        self.kick, self._kicked = kick, 0.0
        where.bot.mkdir(parents=True, exist_ok=True)
        self.state = {"offset": 0, "sending": True, "groups": {}}
        if self._state_path.is_file():
            self.state.update(json.loads(self._state_path.read_text(encoding="utf-8")))
        self.done = {r.get("update_id") for r in _read_jsonl(self._messages_path)}
        self._bad_lines: set[str] = set()

    # ------------------------------------------------------------------------------------------------ one poll
    def poll_once(self, timeout: int = 0) -> int:
        """Send the outbox, read what waits on Telegram (waiting up to `timeout` seconds), file each message, send the
        outbox again (a line never waits behind a long listen)."""
        self.send_outbox()
        updates = self.tg.get_updates(int(self.state["offset"]), timeout)
        for update in updates:
            self.handle(update)
        if time.time() - self._kicked >= config.INTAKE_EVERY_MINUTES * 60:
            self._kick()                                 # a run that stopped, the batch day: picked up by itself
        self.send_outbox()
        return len(updates)

    def _kick(self) -> None:
        """Start the intake (L10b) as its own program; it does nothing when another run is on (its lock)."""
        if self.kick is None:
            return
        self._kicked = time.time()
        try:
            self.kick()
        except Exception:
            self.log(f"could not start the intake:\n{traceback.format_exc()}")

    def handle(self, update: dict) -> R.Route | None:
        uid = int(update["update_id"])
        try:
            msg = R.parse(update)
            if msg is None or msg.chat_id != self.chat_id:
                if msg is not None:
                    self.log(f"ignored a message from another chat (update {uid})")
                self._advance(uid)
                return None
            if uid in self.done:                         # Telegram gave it again after a crash: filed already
                self._advance(uid)
                return None
            r = R.route(msg, self.questions())
            filed, reply = self._file(msg, r)
        except Exception:
            self._keep_failed(uid, update)
            return None
        _append_jsonl(self._messages_path, {
            "update_id": uid, "message_id": msg.message_id, "sent": msg.date.isoformat(timespec="seconds"),
            "filed": _now(), "kind": r.kind, "why": r.why, "command": r.command, "words": msg.words,
            "photo_id": msg.photo_id, "reply_to_id": msg.reply_to_id, "edited": msg.edited, "other": msg.other,
            "question": r.question, **filed})
        self.done.add(uid)
        self._advance(uid)                               # only now: it is on disk
        self.log(f"{r.kind}: {msg.words[:60]!r}" + (" + photo" if msg.photo_id else "") + f" -> {filed}")
        self._reply(msg, reply)
        if r.kind in (R.IDEA, R.ANSWER):
            self._kick()                                 # a new idea, or an answer that may let an idea go on
        return r

    # ------------------------------------------------------------------------------------------------ filing
    def _file(self, msg: R.Message, r: R.Route) -> tuple[dict, str]:
        """Files the message; returns (what was written where, the reply to him)."""
        if r.kind == R.COMMAND:
            return {}, self._command(r.command)
        if r.kind == R.UNREADABLE:
            return {}, (f"I can read only words and pictures, so this {msg.other} was not saved.\n"
                        "Please type it (your phone's voice typing works).")
        if r.kind == R.IDEA:
            n = inbox.add_idea(self.where, when=msg.date, words=r.words, message_id=msg.message_id)
            place = inbox.show(self.where.intake)
            waiting = self._ideas_ahead()
            return {"intake": place, "place": n}, (
                f"Saved as a new idea.\nFile: {place}\n"
                + (f"It starts when the idea running now is done ({waiting})." if waiting else
                   "The machine starts on it now, by itself.")
                + "\nIt tells you when it needs you. Type STATUS to see where it is.")
        if r.kind == R.ANSWER:
            from machine.labels import replies as LR
            if LR.is_label_question(r.question) and self.where.labels is not None:
                return self._label(msg, r)
            return self._answer(msg, r)
        return self._lesson(msg, r)

    def _lesson(self, msg: R.Message, r: R.Route) -> tuple[dict, str]:
        notes, saved = [], None
        # the same words without the photo would have been a command or an idea
        looks_like = bool(msg.photo_id) and R.route(R.Message(**{**msg.__dict__, "photo_id": None}),
                                                    {}).kind in (R.COMMAND, R.IDEA)
        if msg.photo_id:
            saved = self._photo(msg, f"{inbox.stamp(msg.date)}-telegram-m{msg.message_id}-his-picture")
            notes.append(inbox.picture_note(saved))
        if looks_like:
            notes.append("its words look like a command or an idea, but a photo never starts one: filed as a lesson")
        if msg.edited:
            notes.append(f"he edited an earlier message (Telegram message {msg.message_id}); this is the new text")
        if msg.reply_to_id is not None:
            notes.append(f'a reply to the bot\'s message: "{_short(msg.reply_to_words)}"')
        notes.append(_message_note(msg))
        group = self.state["groups"].get(msg.media_group_id) if msg.media_group_id else None
        if group and Path(group).is_file():
            path = Path(group)
            inbox.add_to_inbox(path, words=msg.words, notes=notes)
            first = "Added this picture to the same lesson."
        else:
            path = inbox.write_inbox(self.where, kind="Lesson", when=msg.date, words=msg.words, notes=notes)
            if msg.media_group_id:
                self._remember_group(msg.media_group_id, path)
            first = "Saved as a lesson" + (", with your picture." if msg.photo_id else ".")
        reply = [first, f"File: {inbox.show(path)}"]
        if saved:
            reply.append(_picture_line(saved))
        if looks_like:
            reply.append("A photo never starts a command or an idea, so this is a lesson.")
        reply.append("It gets its lesson number when it reaches GitHub.")
        return {"inbox": inbox.show(path), "picture": saved}, "\n".join(reply)

    def _answer(self, msg: R.Message, r: R.Route) -> tuple[dict, str]:
        q = r.question or {}
        qid, idea = q.get("qid") or q.get("id"), q.get("idea")
        notes, saved = [], None
        if msg.photo_id:
            saved = self._photo(msg, f"{idea}-{qid}-his-answer" if idea else f"{qid}-his-answer")
            notes.append(inbox.picture_note(saved))
        notes.append(f'his answer to question {qid}{f" ({idea})" if idea else ""}: '
                     f'"{_short(msg.reply_to_words)}"')
        notes.append(_message_note(msg))
        kind = f"Answer to question {qid}" + (f", {idea}" if idea else "")
        path = inbox.write_inbox(self.where, kind=kind, when=msg.date, words=msg.words, notes=notes)
        source = inbox.show(path) + (f"; picture `{saved['copy']}`" if saved and saved.get("copy") else "")
        answers = inbox.add_answer(self.where, question=q, words=msg.words or "(a picture, no words)", source=source)
        reply = [f"Saved as your answer to question {qid}" + (f" ({_idea_label(idea)})." if idea else "."),
                 f"File: {inbox.show(path)}"]
        if answers:
            reply.append(f"Also in: {answers}")
        if saved:
            reply.append(_picture_line(saved))
        read = self._interpret(q, msg, source)
        reply.append(read or "Nothing changes until you say yes to one picture that shows how I read your answer.")
        return {"inbox": inbox.show(path), "answers": answers, "picture": saved}, "\n".join(reply)

    def _label(self, msg: R.Message, r: R.Route) -> tuple[dict, str]:
        """A reply to a label chart (L15.1). "yes" / "no" / "not sure" (or "1 yes 2 no ...") goes to the label file,
        NOT the lesson inbox - a label is not a lesson. Anything more - a reason, a drawing, words it cannot read as a
        label - ALSO goes to the inbox like any answer, so nothing he says is lost."""
        from machine.labels import replies as LR
        q = r.question or {}
        qid = str(q.get("qid"))
        got = LR.read(msg.words)
        filed: dict = {}
        lines = []
        if got:
            saved = LR.record(self.where.labels, question=qid, got=got, words=msg.words, message_id=msg.message_id,
                              when=msg.date)
            filed["labels"] = saved["file"]
            lines.append(saved["said"])
        if msg.photo_id or not got or LR.more_than_labels(msg.words):
            notes, pic = [], None
            if msg.photo_id:
                pic = self._photo(msg, f"{qid}-his-answer")
                notes.append(inbox.picture_note(pic))
            notes.append(f'his answer to the label chart {qid}: "{_short(msg.reply_to_words)}"')
            notes.append(_message_note(msg))
            path = inbox.write_inbox(self.where, kind=f"Answer to label chart {qid}", when=msg.date, words=msg.words,
                                     notes=notes)
            filed.update({"inbox": inbox.show(path), "picture": pic})
            lines.append(("Your words are also saved as a lesson." if got else
                          "I could not find yes / no / not sure in this, so the chart stays unanswered. "
                          "Saved as a lesson.") + f"\nFile: {inbox.show(path)}")
            if pic:
                lines.append(_picture_line(pic))
        return filed, "\n".join(lines)

    def _interpret(self, q: dict, msg: R.Message, source: str) -> str | None:
        """L9b: read his reply (a letter, a yes to the confirm picture ...); L10b: his YES / NO to the 3 setups, his
        answer to the 10-in-a-row question. It is filed already, so a bug here only costs the extra line; it is
        logged."""
        if self.where.questions is None:
            return None
        try:
            from machine.intake import state as intake
            line = intake.on_reply(q, msg.words, bool(msg.photo_id), source=source)
            if line is not None:
                return line
        except Exception:
            self.log(f"could not read the answer to {q.get('qid')} (it is filed):\n{traceback.format_exc()}")
            return None
        try:
            from machine.questions import interpret
            return interpret.on_answer(q, msg.words, bool(msg.photo_id), source=source, base=self.where.questions,
                                       outbox=self.where.outbox)
        except Exception:
            self.log(f"could not read the answer to {q.get('qid')} (it is filed):\n{traceback.format_exc()}")
            return None

    def _photo(self, msg: R.Message, stem: str) -> dict:
        try:
            data, ext = self.tg.download(msg.photo_id)
        except TelegramError as e:
            return {"original": None, "copy": None, "cut": False, "error": f"download failed: {e}"}
        return inbox.save_photo(self.where, when=msg.date, message_id=msg.message_id, data=data, ext=ext,
                                picture_stem=stem)

    # ------------------------------------------------------------------------------------------------ commands
    def _command(self, word: str) -> str:
        if word == "STOP":
            self.state["sending"] = False
            self._save_state()
            return ("Stopped. I send you nothing until you type START.\n"
                    "I still save everything you type, and tell you where.")
        if word == "START":
            self.state["sending"] = True
            self._save_state()
            n = len(self.unsent())
            return f"Sending again. {n} message{'s' if n != 1 else ''} waiting."
        return self.status_text()

    def status_text(self) -> str:
        lines = ["STATUS",
                 f"Sending to you: {'ON' if self.state['sending'] else 'PAUSED (type START)'}. "
                 f"Waiting to be sent: {len(self.unsent())}.",
                 f"Lessons and answers waiting for a number: {len(_inbox_files(self.where.inbox))}."]
        from machine.telegram import nightly
        lines.append(nightly.status_line(self.where.bot))
        if self.where.labels is not None and Path(self.where.labels).is_dir():
            try:
                from machine.labels.replies import status_line
                lines.append(status_line(base=self.where.labels))
            except Exception as e:
                lines.append(f"(could not read the label charts: {str(e)[:100]})")
        try:
            from machine.intake import state as intake
            lines += intake.status_lines() or ["No idea yet. Send one: IDEA and your words."]
        except Exception as e:
            lines.append(f"(could not read the ideas: {str(e)[:100]})")
        return "\n".join(lines)

    def _ideas_ahead(self) -> str:
        """The idea running now, when a new idea must wait for it (the intake runs one at a time)."""
        try:
            from machine.intake import state as intake
            now = intake.current()
            return f"{intake.label(now)} runs now" if now else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------------------------------------ lines TO him
    def outbox_lines(self) -> list[dict]:
        out = []
        for n, raw in enumerate(_lines(self.where.outbox), 1):
            try:
                line = json.loads(raw)
                if not (isinstance(line, dict) and str(line.get("id", "")).strip() and str(line.get("text", ""))):
                    raise ValueError("needs an id and a text")
                out.append(line)
            except ValueError as e:
                if raw not in self._bad_lines:
                    self._bad_lines.add(raw)
                    self.log(f"outbox line {n} skipped ({e}): {raw[:80]}")
        return out

    def unsent(self) -> list[dict]:
        sent = {r.get("id") for r in _read_jsonl(self._sent_path)}
        return [ln for ln in self.outbox_lines() if ln["id"] not in sent]

    def send_outbox(self) -> int:
        """Sends each outbox line not sent yet, then records it (sent first: a crash sends it twice, never zero)."""
        if not self.state["sending"]:
            return 0
        n = 0
        for line in self.unsent():
            try:
                ids = self._send_line(line)
            except TelegramError as e:
                if e.kind in STOP_KINDS:
                    raise
                self.log(f"could not send outbox line {line['id']} ({e}); trying again next time")
                break
            rec = {"id": line["id"], "at": _now(), "message_id": ids[0], "message_ids": ids}
            rec.update({k: line[k] for k in ("qid", "idea") if line.get(k)})
            _append_jsonl(self._sent_path, rec)
            shown = line.get("shown")
            if isinstance(shown, dict) and self.where.shown is not None:     # #117: shown = really in front of him
                _append_jsonl(self.where.shown, {"date": dt.datetime.now().isoformat(timespec="seconds"),
                                                 "subject": shown.get("subject"), "kind": shown.get("kind"),
                                                 "file": shown.get("file"), "pixel_sha256": shown.get("pixel_sha256"),
                                                 "why": shown.get("why") or f"sent on Telegram ({line['id']})"})
            n += 1
        return n

    def _send_line(self, line: dict) -> list[int]:
        text = str(line["text"])
        photo = line.get("photo")
        if photo:
            path = Path(photo) if Path(photo).is_absolute() else config.ROOT / photo
            if not path.is_file():
                raise TelegramError("api", f"the picture {photo} is missing")   # not sent, not marked: retried
            if len(text) <= 1024:                                                # Telegram's caption limit
                return [self.tg.send_photo(self.chat_id, path, caption=text)["message_id"]]
            first = self.tg.send_photo(self.chat_id, path)["message_id"]
            return [first, self.tg.send_message(self.chat_id, text[:4096])["message_id"]]
        return [self.tg.send_message(self.chat_id, text[:4096])["message_id"]]

    def questions(self) -> dict[int, dict]:
        """{Telegram message id: the sent record} for every QUESTION sent (an outbox line with a qid)."""
        out = {}
        for rec in _read_jsonl(self._sent_path):
            if rec.get("qid"):
                for mid in rec.get("message_ids") or [rec.get("message_id")]:
                    out[int(mid)] = rec
        return out

    # ------------------------------------------------------------------------------------------------ plumbing
    @property
    def _state_path(self) -> Path:
        return self.where.bot / "state.json"

    @property
    def _messages_path(self) -> Path:
        return self.where.bot / "messages.jsonl"

    @property
    def _sent_path(self) -> Path:
        return self.where.bot / "sent.jsonl"

    def _advance(self, update_id: int) -> None:
        if update_id + 1 > int(self.state["offset"]):
            self.state["offset"] = update_id + 1
            self._save_state()

    def _remember_group(self, group: str, path: Path) -> None:
        groups = self.state["groups"]
        groups[group] = str(path)
        for old in list(groups)[:-50]:                  # only the newest 50 albums are remembered
            groups.pop(old)
        self._save_state()

    def _save_state(self) -> None:
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self._state_path)

    def _reply(self, msg: R.Message, text: str) -> None:
        if not text:
            return
        try:
            self.tg.send_message(self.chat_id, text[:4096], reply_to=msg.message_id)
        except TelegramError as e:
            self.log(f"could not reply to message {msg.message_id} ({e}); it is filed anyway")

    def _keep_failed(self, uid: int, update: dict) -> None:
        """A bug must neither lose his message nor stop the bot on it forever: the update is kept whole, the bot moves
        on, and tells him."""
        folder = self.where.bot / "failed"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uid}.json"
        path.write_text(json.dumps({"update": update, "error": traceback.format_exc()}, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
        self.log(f"could not file update {uid}: kept at {inbox.show(path)}\n{traceback.format_exc()}")
        self._advance(uid)
        mid = (update.get("message") or update.get("edited_message") or {}).get("message_id")
        chat = str(((update.get("message") or update.get("edited_message") or {}).get("chat") or {}).get("id", ""))
        if mid and chat == self.chat_id:
            try:
                self.tg.send_message(self.chat_id, "I could not file this message (a bug on my side). It is kept "
                                     f"whole at {inbox.show(path)} on your PC.", reply_to=mid)
            except TelegramError:
                pass

    def log(self, text: str) -> None:
        line = f"{_now()} {text}"
        if self.echo:
            print(line, flush=True)
        with open(self.where.bot / "bot.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ==================================================================================================== running it

@contextmanager
def only_one(lock_path: Path):
    """A second copy of the bot on this PC stops at once (the lock goes away by itself when the bot stops)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"Another copy of the bot is already running on this PC ({inbox.show(lock_path)} is "
                         "locked). Only ONE may read Telegram.") from None
    try:
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        fh.close()


def run(bot: Bot, *, once: bool = False, sleep=time.sleep) -> None:
    delay = 5
    bot.log(f"bot started ({'once' if once else 'running'}); files: {inbox.show(bot.where.bot)}")
    while True:
        try:
            bot.poll_once(0 if once else config.TELEGRAM_POLL_SECONDS)
            delay = 5
        except TelegramError as e:
            if e.kind in STOP_KINDS:
                why = ("another program reads this bot (only ONE may - is the old bot or a second copy running?)"
                       if e.kind == "conflict" else "Telegram refused the key (check TELEGRAM_BOT_TOKEN in .env)")
                bot.log(f"STOPPED: {why} [{e}]")
                raise SystemExit(f"Bot stopped: {why}") from None
            bot.log(f"Telegram not reachable ({e}); trying again in {delay} s")
            if once:
                raise SystemExit(1) from None
            sleep(delay)
            delay = min(delay * 2, 300)
        if once:
            return


def copy_old_keys(root: Path | None = None) -> list[str]:
    """The old bot's key and chat id -> .env (appended; a name already in .env is left alone). Never shows a value."""
    root = Path(root or config.ROOT)
    src, dotenv = root / OLD_KEYS, root / ".env"
    if not src.is_file():
        raise SystemExit(f"{OLD_KEYS.as_posix()} is not on this PC - the keys live only on Tejas's PC")
    if (root / ".git").exists():
        ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", ".env"]).returncode == 0
        if not ignored:
            raise SystemExit(".env is not ignored by git here - refusing to write a key into it")
    old = json.loads(src.read_text(encoding="utf-8"))
    have = {}
    if dotenv.is_file():
        from dotenv import dotenv_values
        have = dotenv_values(dotenv)
    said, add = [], []
    for name, key in (("TELEGRAM_BOT_TOKEN", "telegram_token"), ("TELEGRAM_CHAT_ID", "telegram_chat_id")):
        value = str(old.get(key) or "").strip()
        if have.get(name):
            said.append(f"{name}: already in .env, left as it is")
        elif not value:
            said.append(f"{name}: not in the old file")
        else:
            add.append(f"{name}={value}")
            said.append(f"{name}: copied ({len(value)} characters, not shown)")
    if add:
        text = dotenv.read_text(encoding="utf-8") if dotenv.is_file() else ""
        with open(dotenv, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(("" if not text or text.endswith("\n") else "\n") + "\n".join(add) + "\n")
    return said


LOGON_TASK = "ML Bot - the Telegram bot (starts at logon)"


def install_at_logon(*, remove: bool = False, root: Path | None = None) -> str:
    """The PC restarted at 10:54 on 29 Sep 2026 and the bot stayed off until a session saw it (bot.log): now Task
    Scheduler starts it at logon (pythonw = no window; restarted up to 3 times if it stops with an error)."""
    from machine.telegram.nightly import _ps, _run
    if os.name != "nt":
        raise SystemExit("Task Scheduler is on Windows only")
    if remove:
        _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
              f"Unregister-ScheduledTask -TaskName {_ps(LOGON_TASK)} -Confirm:$false"], check=False)
        return f"removed '{LOGON_TASK}'"
    root = Path(root or config.ROOT)
    exe = root / ".venv" / "Scripts" / "pythonw.exe"
    if not exe.is_file():
        raise SystemExit(f"{exe} is missing")
    ps = (f"$a = New-ScheduledTaskAction -Execute {_ps(str(exe))} -Argument '-u -m machine.telegram.bot' "
          f"-WorkingDirectory {_ps(str(root))};"
          "$t = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME;"
          "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
          "-ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) "
          "-MultipleInstances IgnoreNew;"
          f"Register-ScheduledTask -TaskName {_ps(LOGON_TASK)} -Action $a -Trigger $t -Settings $s -Force "
          f"-Description {_ps('The one Telegram bot of the idea machine (plan/likhith.md L9a). A second copy refuses.')}"
          " | Out-Null")
    done = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=False)
    if done.returncode:
        raise SystemExit("Task Scheduler said no: " + (done.stderr or b"").decode("utf-8", "replace")[:300])
    return f"Task Scheduler: '{LOGON_TASK}' starts the bot when {os.environ.get('USERNAME', 'you')} logs in"


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.telegram.bot", description=__doc__.split("\n\n")[0])
    ap.add_argument("--once", action="store_true", help="file what waits, send the outbox, then stop")
    ap.add_argument("--test", action="store_true", help="the test bot (TELEGRAM_TEST_*): writes only in data/telegram-test/")
    ap.add_argument("--copy-old-keys", action="store_true", help="copy the old bot's key + chat id into .env")
    ap.add_argument("--install", action="store_true", help="start the bot by itself when Tejas logs in (Task "
                                                           "Scheduler); a second copy still refuses to run")
    ap.add_argument("--uninstall", action="store_true", help="remove that Task Scheduler entry")
    args = ap.parse_args(argv)
    if args.install or args.uninstall:
        print(install_at_logon(remove=args.uninstall))
        return
    if args.copy_old_keys:
        for line in copy_old_keys():
            print(line)
        return
    names = ("TELEGRAM_TEST_BOT_TOKEN", "TELEGRAM_TEST_CHAT_ID") if args.test else ("TELEGRAM_BOT_TOKEN",
                                                                                 "TELEGRAM_CHAT_ID")
    key, chat = (config.env(n) for n in names)
    missing = [n for n, v in zip(names, (key, chat)) if not v]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)} (see .env.example; on Tejas's PC: --copy-old-keys)")
    where = inbox.Where.test() if args.test else inbox.Where.real()
    kick = None
    if not args.test:
        from machine.intake.state import spawn as kick     # the intake runs as its own program (L10b)
    with only_one(where.bot / "bot.lock"):
        try:
            run(Bot(TelegramAPI(key), chat, where, kick=kick), once=args.once)
        except KeyboardInterrupt:
            print("bot stopped (Ctrl+C)")


# ==================================================================================================== small helpers

def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _message_note(msg: R.Message) -> str:
    return (f"Telegram message {msg.message_id}, sent {msg.date.strftime('%Y-%m-%d %H:%M:%S %z')}, "
            "filed by the bot (L9a)")


def _picture_line(saved: dict) -> str:
    if not saved.get("copy"):
        return "Picture: kept as it came on your PC (a copy could not be made)."
    return f"Picture: {saved['copy']}" + (" (the phone's top bar and buttons cut off)" if saved["cut"] else "")


def _idea_label(idea: str) -> str:
    try:
        return f"idea #{int(idea.split('-')[1])}"
    except (IndexError, ValueError):
        return idea


def _inbox_files(folder: Path) -> list[Path]:
    return [p for p in Path(folder).glob("*.md") if p.name.lower() != "readme.md"] if Path(folder).is_dir() else []


def _lines(path: Path) -> list[str]:
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()] if Path(
        path).is_file() else []


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in _lines(path):
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
