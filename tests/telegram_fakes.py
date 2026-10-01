"""A fake Telegram (L9a): no key, no internet, no phone. Tejas "types" with .text() / .photo() / .voice(); what the
bot sends lands in .sent. It behaves like the real getUpdates queue: reading with offset N deletes every update before
N (the reason there is only one poller). .fail = "conflict" / "network" makes the next get_updates fail that way."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from machine.telegram.api import TelegramError

TEJAS = "111222333"
WHEN = 1790000000                                  # a fixed Unix time for every message (Jan 2026)


def phone_screenshot(w: int = 600, h: int = 1335) -> bytes:
    """A picture shaped like his phone's screenshots (1200 x 2670 halved): a RED status bar on top, BLUE buttons at
    the bottom, white in between - so a test can see what the crop cut."""
    im = Image.new("RGB", (w, h), "white")
    im.paste((255, 0, 0), (0, 0, w, round(h * 0.045)))
    im.paste((0, 0, 255), (0, h - round(h * 0.05), w, h))
    return _jpeg(im)


def chart_picture(w: int = 800, h: int = 500) -> bytes:
    """A wide picture (a chart he drew on): never cut. A red line along the top edge proves it."""
    im = Image.new("RGB", (w, h), "white")
    im.paste((255, 0, 0), (0, 0, w, 10))
    return _jpeg(im)


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue()


class FakeTelegram:
    def __init__(self, chat_id: str = TEJAS):
        self.chat_id = chat_id
        self.queue: list[dict] = []
        self.sent: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.fail: str | None = None
        self._uid, self._mid, self._fid = 1000, 500, 0

    # ---- Tejas's phone
    def _update(self, fields: dict, *, chat=None, reply_to=None, edited=False, group=None) -> dict:
        self._uid += 1
        self._mid += 1
        msg = {"message_id": self._mid, "date": WHEN, "chat": {"id": int(chat or self.chat_id)}, **fields}
        if reply_to is not None:
            said = next((s for s in self.sent if s["message_id"] == reply_to), {})
            msg["reply_to_message"] = {"message_id": reply_to, "text": said.get("text") or said.get("caption") or ""}
        if group:
            msg["media_group_id"] = group
        update = {"update_id": self._uid, ("edited_message" if edited else "message"): msg}
        self.queue.append(update)
        return update

    def text(self, words: str, **kw) -> dict:
        return self._update({"text": words}, **kw)

    def photo(self, data: bytes, caption: str | None = None, *, as_file: bool = False, **kw) -> dict:
        self._fid += 1
        fid = f"file-{self._fid}"
        self.files[fid] = data
        if as_file:
            fields = {"document": {"file_id": fid, "mime_type": "image/jpeg"}}
        else:
            fields = {"photo": [{"file_id": fid + "-small", "width": 90}, {"file_id": fid, "width": 1280}]}
            self.files[fid + "-small"] = data
        if caption is not None:
            fields["caption"] = caption
        return self._update(fields, **kw)

    def voice(self, **kw) -> dict:
        return self._update({"voice": {"file_id": "v1", "duration": 4}}, **kw)

    # ---- the five calls the bot uses (same as machine.telegram.api.TelegramAPI)
    def get_updates(self, offset: int, timeout: int) -> list[dict]:
        if self.fail:
            kind, self.fail = self.fail, None
            raise TelegramError(kind, "fake failure")
        self.queue = [u for u in self.queue if u["update_id"] >= offset]     # Telegram forgets what was read
        return list(self.queue)

    def send_message(self, chat_id: str, text: str, reply_to: int | None = None) -> dict:
        return self._send({"chat_id": str(chat_id), "text": text, "reply_to": reply_to})

    def send_photo(self, chat_id: str, path: Path, caption: str | None = None, reply_to: int | None = None) -> dict:
        return self._send({"chat_id": str(chat_id), "photo": str(path), "caption": caption, "reply_to": reply_to})

    def download(self, file_id: str) -> tuple[bytes, str]:
        return self.files[file_id], ".jpg"

    def _send(self, rec: dict) -> dict:
        self._mid += 1
        rec["message_id"] = self._mid
        self.sent.append(rec)
        return {"message_id": self._mid}

    # ---- helpers for the tests
    def replies_to(self, update: dict) -> list[str]:
        mid = (update.get("message") or update.get("edited_message"))["message_id"]
        return [s["text"] for s in self.sent if s.get("reply_to") == mid]
