"""Messages to Tejas from the daily run and the heartbeat (L16.1).

Two ways, on purpose:
    alert(text)            straight to Telegram with the bot's key - the alert may be that the BOT is not running, so it
                           cannot wait in the bot's outbox. Sending is not polling: it never disturbs the one bot (B12;
                           the nightly job sends the same way). No network yet (right after a power cut the router is
                           still starting)? tried again a few times, then kept in unsent.jsonl for the next heartbeat.
    to_outbox(id, text)    a line in the bot's outbox (plan/work/telegram-outbox.jsonl): the daily summary, when the bot
                           runs. The bot sends it once (an id is never sent twice) and the nightly job puts it in git.

Both respect his STOP: while the bot's state says "not sending" (he typed STOP), nothing is sent - only logged.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from machine import config

TRIES = 3
WAIT_SECONDS = 20
INDIA = dt.timezone(dt.timedelta(hours=5, minutes=30))   # every time Tejas reads is India time (no summer clock)


def now() -> dt.datetime:
    return dt.datetime.now(INDIA)


def india(t: dt.datetime) -> dt.datetime:
    return t.astimezone(INDIA)


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def log(name: str, text: str) -> None:
    """One line in data/host/<name>.log."""
    path = Path(config.HOST_DIR) / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for i, line in enumerate(str(text).splitlines() or [""]):
            fh.write(f"{now():%Y-%m-%d %H:%M:%S} {'  ' if i else ''}{line}\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def paused() -> bool:
    """He typed STOP: the bot's state.json says "sending": false."""
    path = Path(config.TELEGRAM_DIR) / "state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("sending", True) is False
    except (OSError, ValueError):
        return False


def telegram():
    """A function that sends one message with the real bot key, or None when .env has no key."""
    from machine.telegram.nightly import telegram_tell
    return telegram_tell()


def alert(text: str, *, tell=None, sleep=time.sleep, kind: str = "alert") -> bool:
    """Send `text` to Tejas now. True = sent. False = paused (STOP), no key, or no network (kept for the next beat)."""
    text = str(text)[:4096]
    if paused():
        log("alerts", f"NOT SENT (he typed STOP): {text}")
        return False
    tell = tell or telegram()
    if tell is None:
        log("alerts", f"NOT SENT (no TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env): {text}")
        return False
    last = ""
    for i in range(TRIES):
        try:
            tell(text)
            log("alerts", f"sent ({kind}): {text}")
            return True
        except Exception as e:                         # no network, Telegram busy: wait a little, try again
            last = f"{type(e).__name__}: {e}"
            if i + 1 < TRIES:
                sleep(WAIT_SECONDS)
    append_jsonl(Path(config.HOST_DIR) / "unsent.jsonl", {"at": iso(now()), "kind": kind, "text": text, "why": last})
    log("alerts", f"NOT SENT YET ({last}) - kept for the next heartbeat: {text}")
    return False


def resend(*, tell=None, sleep=time.sleep) -> int:
    """Send what an earlier run could not (no network then). Returns how many went now."""
    path = Path(config.HOST_DIR) / "unsent.jsonl"
    waiting = read_jsonl(path)
    if not waiting or paused():
        return 0
    tell = tell or telegram()
    if tell is None:
        return 0
    sent, left = 0, []
    for w in waiting:
        try:
            tell(f"(from {w['at'][:16].replace('T', ' ')}, sent late: the PC had no network then)\n{w['text']}"[:4096])
            sent += 1
        except Exception:
            left.append(w)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(w, ensure_ascii=False) + "\n" for w in left), encoding="utf-8")
    tmp.replace(path)                                  # written whole, then moved into place (hard rule 7)
    if sent:
        log("alerts", f"sent {sent} late message{'s' if sent != 1 else ''}")
    return sent


def to_outbox(line_id: str, text: str) -> bool:
    """One line for the bot to send. False = this id is in the outbox already (sent once, never twice)."""
    box = Path(config.WORK_DIR) / "telegram-outbox.jsonl"
    if any(line.get("id") == line_id for line in read_jsonl(box)):
        return False
    append_jsonl(box, {"id": line_id, "text": str(text)[:4096], "at": iso(now()), "from": "machine.host (L16.1)"})
    return True
