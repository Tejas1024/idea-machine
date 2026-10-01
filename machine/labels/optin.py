"""His T9 answer to the daily label charts (L15.1): plan/work/labels/opt-in.md. Light on purpose - the ONE bot reads it
for STATUS (no pandas, no pictures here).

The file is only ever APPENDED to (lesson B10): each entry is "## <date time> — opt in" or "## <date time> — no", his
words quoted under it, word for word. The NEWEST entry counts.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from machine import config

_ENTRY = re.compile(r"^## (?P<when>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) — (?P<answer>opt in|no)\s*$", re.MULTILINE)


def opt_in_state(path: Path | None = None) -> dict:
    """{"answer": "opt in" / "no" / None, "when"}: the NEWEST entry of plan/work/labels/opt-in.md counts."""
    path = Path(path or config.LABELS_OPT_IN)
    if not path.is_file():
        return {"answer": None, "when": None}
    found = list(_ENTRY.finditer(path.read_text(encoding="utf-8")))
    return {"answer": found[-1]["answer"], "when": found[-1]["when"]} if found else {"answer": None, "when": None}


def opt_in(answer: str, words: str, *, who: str, path: Path | None = None, now: dt.datetime | None = None) -> Path:
    """Add his T9 answer (his words, word for word) to plan/work/labels/opt-in.md - appended, never rewritten (B10)."""
    if answer not in ("opt in", "no"):
        raise ValueError('his answer is "opt in" or "no"')
    if not str(words).strip():
        raise ValueError("his own words are needed (T9) - not a summary")
    path = Path(path or config.LABELS_OPT_IN)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = now or dt.datetime.now()
    head = "" if path.is_file() else (
        "# Label charts - his answer (T9)\n\n> The NEWEST entry counts: \"opt in\" = the bot sends 10 zone charts a "
        "day (plan/likhith.md L15.1); \"no\" = nothing is sent. Written by `python -m machine.labels opt-in`.\n")
    quoted = "\n".join(f"> {ln}" if ln.strip() else ">" for ln in str(words).splitlines())
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{head}\n## {now:%Y-%m-%d %H:%M} — {answer}\n\n{quoted}\n\n(who: {who})\n")
    return path
