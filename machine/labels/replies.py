"""His replies to label charts (L15.1): yes / no / not sure, read from his words, kept in
plan/work/labels/<word>/replies.jsonl (append only - lesson B10; the nightly job carries it to git, L9c).

A chart's question id: LBL-<word>-<YYYYMMDD>-<NN> (the day it was sent, its number that day). The bot routes a reply
to such a picture here (machine/telegram/bot.py), not to the lesson inbox: "yes" on a chart is a label, not a lesson.
What is MORE than a label still goes to the inbox as well (more_than_labels), so nothing he teaches is lost.

What counts (Claude's reading of the prompt's "yes / no / not sure"; his phone's voice typing adds capitals and
full stops): yes = yes, yeah, yep, yup, ya, yea, y, haan · no = no, nope, nah, nahi, n · not sure = not sure, notsure,
unsure, don't know, not certain, can't say, maybe. One reply may carry numbers: "1 yes 2 no 3 not sure" answers
charts 1, 2 and 3 of the SAME day as the picture he replied to. A label with no number answers the picture itself.
Anything else (a letter, a drawing with no words) labels nothing; it is filed as a lesson and the chart stays open.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from machine import config

QID = re.compile(r"^LBL-(?P<word>[a-z_]+)-(?P<day>\d{8})-(?P<n>\d{2})$")
_SAYS = {
    "not sure": ("not sure", "notsure", "unsure", "don't know", "dont know", "do not know", "not certain",
                 "can't say", "cant say", "maybe"),
    "yes": ("yes", "yeah", "yep", "yup", "yea", "ya", "y", "haan"),
    "no": ("no", "nope", "nah", "nahi", "n"),
}
_WORD_OF = {w: label for label, words in _SAYS.items() for w in words}
_ONE = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted(_WORD_OF, key=len, reverse=True))
_PAIR = re.compile(rf"(?<![\w'])(?:(?P<n>\d{{1,2}})\s*[-:.)=]?\s*)?(?P<say>{_ONE})(?![\w'-])", re.IGNORECASE)
# words that say nothing new next to a label ("yes this is my kind of zone")
_FILLER = {"this", "is", "it", "its", "it's", "my", "kind", "of", "zone", "zones", "the", "a", "one", "that", "i",
           "think", "and", "also", "chart", "picture", "ok", "okay", "for", "to", "so", "all", "are", "these", "number"}
MORE_WORDS = 3                    # this many other words = he is saying something: filed in the lesson inbox too


def is_label_question(question: dict | None) -> bool:
    return bool(question) and bool(QID.match(str(question.get("qid") or "")))


def qid(word: str, day: dt.date, n: int) -> str:
    return f"LBL-{word}-{day:%Y%m%d}-{n:02d}"


def parts(q: str) -> tuple[str, str, int]:
    m = QID.match(q)
    if not m:
        raise ValueError(f"{q!r} is not a label chart's id (LBL-<word>-<YYYYMMDD>-<NN>)")
    return m["word"], m["day"], int(m["n"])


def read(words: str) -> list[tuple[int | None, str]]:
    """His words -> [(chart number or None, "yes" / "no" / "not sure")], in the order he wrote them."""
    text = " ".join(str(words or "").replace("’", "'").split())
    return [(int(m["n"]) if m["n"] else None, _WORD_OF[" ".join(m["say"].lower().split())])
            for m in _PAIR.finditer(text)]


def more_than_labels(words: str) -> bool:
    """Does the reply say more than its labels (a reason, a correction)? Then it is ALSO a lesson."""
    rest = _PAIR.sub(" ", " ".join(str(words or "").replace("’", "'").split()))
    left = [w for w in re.findall(r"[a-z']+", rest.lower()) if w not in _FILLER]
    return len(left) >= MORE_WORDS


def replies_file(word: str, base: Path | None = None) -> Path:
    return Path(base or config.LABELS_WORK_DIR) / word / "replies.jsonl"


def record(base: Path | None, *, question: str, got: list[tuple[int | None, str]], words: str, message_id: int,
           when: dt.datetime) -> dict:
    """Append one line per chart he answered. A number answers chart N of the same day as `question`; no number
    answers `question` itself; a number with no chart that day is not saved (and he is told). Returns {"file",
    "rows", "missing", "said"} (said = the line the bot replies with)."""
    word, day, _n = parts(question)
    sent = {c["qid"] for c in read_charts(word, base)}
    rows, missing = [], []
    for n, label in got:
        target = question if n is None else f"LBL-{word}-{day}-{n:02d}"
        if target != question and target not in sent:
            missing.append(n)
            continue
        rows.append({"qid": target, "label": label, "words": words, "replied_to": question,
                     "message_id": message_id, "at": when.isoformat(timespec="seconds")})
    path = replies_file(word, base)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    answered = {r["qid"] for r in read_replies(word, base)}
    if rows and all(r["qid"] == question for r in rows):
        said = f"Saved: {rows[-1]['label']}."
    else:
        said = "Saved: " + (", ".join(f"chart {parts(r['qid'])[2]} {r['label']}" for r in rows) or "nothing") + "."
    if missing:
        said += f" There is no chart {', '.join(map(str, missing))} that day - not saved."
    said += f" Charts answered so far: {len(answered)}."
    return {"file": _show(path), "rows": rows, "missing": missing, "said": said}


def charts_file(word: str, base: Path | None = None) -> Path:
    return Path(base or config.LABELS_WORK_DIR) / word / "charts.jsonl"


def read_charts(word: str, base: Path | None = None) -> list[dict]:
    """Every label chart sent (machine/labels/daily.py writes one line per chart)."""
    return _jsonl(charts_file(word, base))


def read_replies(word: str, base: Path | None = None) -> list[dict]:
    return _jsonl(replies_file(word, base))


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def status_line(word: str = "strong_zone", base: Path | None = None) -> str:
    """One line for the bot's STATUS (light: the bot imports no pandas or pictures for it). base: plan/work/labels/."""
    from machine.labels import drawings, optin, words
    sent = read_charts(word, base)
    answered = latest(word, base)
    state = optin.opt_in_state(Path(base) / "opt-in.md" if base else None)["answer"] or "not asked yet (T9)"
    return (f"Label charts ({words.get(word).says}): {len(drawings.labels(word))} from your drawings; {len(sent)} sent, "
            f"{sum(1 for c in sent if c['qid'] in answered)} answered, "
            f"{sum(1 for c in sent if c['qid'] not in answered)} waiting. Daily charts: {state}.")


def latest(word: str, base: Path | None = None) -> dict[str, dict]:
    """{chart id: his LAST reply to it} - he may change his mind; the newest line wins (all lines stay)."""
    out: dict[str, dict] = {}
    for r in read_replies(word, base):
        out[r["qid"]] = {**r, "replies": out.get(r["qid"], {}).get("replies", 0) + 1}
    return out


def _show(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
