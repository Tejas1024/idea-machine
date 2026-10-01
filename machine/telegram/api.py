"""The real Telegram wire (Bot API over https, standard library only). The bot talks to it through five calls, so the
tests swap in a fake Telegram (tests/telegram_fakes.py) with no key, no internet and no phone:

    get_updates(offset, timeout) -> [update, ...]     long poll; reading with offset N deletes every update before N
    send_message(chat_id, text, reply_to=None) -> the sent message (its message_id)
    send_photo(chat_id, path, caption=None, reply_to=None) -> the sent message
    download(file_id) -> (bytes, ".jpg")

The key is never printed: every error text has it replaced by "<key>" (the URL holds it).
"""
from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

API = "https://api.telegram.org/bot{key}/{method}"
FILE = "https://api.telegram.org/file/bot{key}/{path}"


class TelegramError(RuntimeError):
    """kind: "conflict" (409: another program polls this bot - only ONE poller, likhith.md B12), "unauthorized" (401:
    a wrong key), "network" (no internet; try again), "api" (Telegram said no)."""

    def __init__(self, kind: str, text: str):
        super().__init__(f"{kind}: {text}")
        self.kind = kind


class TelegramAPI:
    def __init__(self, key: str):
        if not key:
            raise ValueError("no bot key: TELEGRAM_BOT_TOKEN in .env (or TELEGRAM_TEST_BOT_TOKEN with --test)")
        self._key = key

    # ---- the five calls
    def get_updates(self, offset: int, timeout: int) -> list[dict]:
        return self._call("getUpdates", {"offset": offset, "timeout": timeout,
                                         "allowed_updates": json.dumps(["message", "edited_message"])},
                          wait=timeout + 15)

    def send_message(self, chat_id: str, text: str, reply_to: int | None = None) -> dict:
        data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
        if reply_to:
            data["reply_to_message_id"] = reply_to
            data["allow_sending_without_reply"] = "true"
        return self._call("sendMessage", data)

    def send_photo(self, chat_id: str, path: Path, caption: str | None = None, reply_to: int | None = None) -> dict:
        fields = {"chat_id": str(chat_id)}
        if caption:
            fields["caption"] = caption
        if reply_to:
            fields["reply_to_message_id"] = str(reply_to)
            fields["allow_sending_without_reply"] = "true"
        body, ctype = _multipart(fields, "photo", Path(path))
        return self._call("sendPhoto", body=body, content_type=ctype, wait=60)

    def download(self, file_id: str) -> tuple[bytes, str]:
        info = self._call("getFile", {"file_id": file_id})
        path = info["file_path"]
        try:
            with urllib.request.urlopen(FILE.format(key=self._key, path=path), timeout=60) as r:
                return r.read(), (Path(path).suffix or ".jpg")
        except (urllib.error.URLError, OSError) as e:
            raise TelegramError("network", self._hide(str(e))) from None

    # ---- plumbing
    def _call(self, method: str, data: dict | None = None, *, body: bytes | None = None,
              content_type: str | None = None, wait: int = 20):
        url = API.format(key=self._key, method=method)
        if body is None:
            body = urllib.parse.urlencode(data or {}).encode()
            content_type = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=wait) as r:
                out = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                said = json.loads(e.read().decode("utf-8")).get("description", "")
            except Exception:
                said = ""
            kind = {409: "conflict", 401: "unauthorized", 404: "unauthorized"}.get(e.code, "api")
            raise TelegramError(kind, self._hide(f"{method}: HTTP {e.code} {said}".strip())) from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise TelegramError("network", self._hide(f"{method}: {e}")) from None
        if not out.get("ok"):
            raise TelegramError("api", self._hide(f"{method}: {out.get('description', out)}"))
        return out["result"]

    def _hide(self, text: str) -> str:
        return str(text).replace(self._key, "<key>")


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{path.name}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
