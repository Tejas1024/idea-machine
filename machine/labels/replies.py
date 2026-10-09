"""His replies to label charts (L15.1): yes / no / not sure, read from his words, kept in
plan/work/labels/<word>/replies.jsonl (append only - lesson B10; the nightly job carries it to git, L9c).

A chart's question id: LBL-<word>-<YYYYMMDD>-<NN> (the day it was sent, its number that day). The bot routes a reply
to such a picture here (machine/telegram/bot.py), not to the lesson inbox: "yes" on a chart is a label, not a lesson.
What is MORE than a label still goes to the inbox as well (more_than_labels), so nothing he teaches is lost.

What counts (Claude's reading of the prompt's "yes / no / not sure"; his phone's voice typing adds capitals and
full stops): yes = yes, yeah, yep, yup, ya, yea, y, haan · no = no, nope, nah, nahi, n · not sure = not sure, notsure,
unsure, don't know, not certain, can't say, maybe. One reply may carry numbers: "1 yes 2 no 3 not sure" answers
charts 1, 2 and 3 of the SAME day as the picture he replied to. A label with no number answers the picture itself.
His own way (1 Oct 2026): a line of chart numbers only ("1.3.7.", "my zones: 1 3 7") = his zones; every other chart
of that day = no (listed / read_all); a line saying none is his ("none", "no zones today") = every chart of that day no
(none_listed) - as a reply to a picture, or as a plain message while the newest day's charts still wait (the bot then
takes it for that day). Anything else (a letter, a drawing with no words) labels nothing; it is filed as a lesson and the chart stays open.
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


# His own way (1 Oct 2026, Telegram message 915, NOT a reply to a picture): "Below are my zones, which I did not
# mentioned are not mine ... 1.3.7." A line that carries only chart numbers (filler words allowed: "my zones: 1, 3 and
# 7") lists HIS zones = yes; every other chart of that day = no. Only when the message has no yes / no / not sure.
_LIST_LINE = re.compile(r"\d{1,2}(?:\s*[.,;&+/]?\s*\d{1,2})*\s*[.,;!]?")
# "All 3 ok", "All 10" (6 Oct 2026, Claude's reading): "all" right before a line's ONE number counts things - all three
# (the picture of 3 setups says "YES = all 3 are how you trade it"), all ten - it names no chart. Such a line is no list:
# it stays a lesson (the intake reads "All 3 ok" as his YES to that picture), and chart 3 is never his zone by it.
_ALL_N = re.compile(r"\ball\s+(?:(?:the|of|these|those)\s+)?\d{1,2}\b", re.IGNORECASE)


def listed(words: str) -> list[int]:
    """The chart numbers on his list line, in his order, each once. [] when no line is only numbers, when a number is
    not a chart number (0, 74 ...: "0.74" is no list), when the message says yes / no itself (read() wins), or when a
    line's one number follows "all" ("All 3 ok" = all three: _ALL_N)."""
    if read(words):
        return []
    for line in str(words or "").replace("’", "'").splitlines():
        core = re.sub(r"[A-Za-z']+", lambda m: "" if m.group(0).lower() in _FILLER else m.group(0), line)
        core = core.strip(" \t:-–—")
        if not core or not _LIST_LINE.fullmatch(core):
            continue
        nums = [int(n) for n in re.findall(r"\d{1,2}", core)]
        if len(nums) == 1 and _ALL_N.search(line):
            continue
        if all(1 <= n <= config.LABELS_PER_DAY for n in nums):
            return list(dict.fromkeys(nums))
    return []


# His list with NO number (2 Oct 2026, Claude's reading of the same rule, "which I did not mentioned are not mine"): a
# line that says only that none is his - "none", "none of them are mine", "nothing", "zero", "0", "no zones today",
# "all no", "not even one" - makes every chart of that day no. "no" alone, "no zone" and "not mine" are not it: they
# answer one picture, as before.
_NONE = {"none", "nothing", "zero", "nil", "0"}
_NONE_OK = _FILLER | _NONE | {"no", "not", "mine", "them", "today", "even", "any", "yours", "ones", "here"}


def _none_line(line: str) -> bool:
    toks = re.findall(r"[a-z']+|\d+", line.lower())
    if not toks or not all(t in _NONE_OK for t in toks):
        return False
    pairs = set(zip(toks, toks[1:]))
    return bool(_NONE & set(toks)) or bool(pairs & {("no", "zones"), ("all", "no"), ("even", "one")})


def none_listed(words: str) -> bool:
    """A line saying none of that day's charts is his zone (_none_line) - not when he listed numbers or answered a
    numbered chart himself ("1 yes"): what he said wins."""
    if listed(words) or any(n is not None for n, _ in read(words)):
        return False
    return any(_none_line(line) for line in str(words or "").replace("’", "'").splitlines())


def is_list(words: str) -> bool:
    """His own way to answer a whole day at once: his zones' numbers (listed), or none (none_listed)."""
    return bool(listed(words)) or none_listed(words)


def read_all(words: str, of: list[int]) -> list[tuple[int | None, str]]:
    """read(); or, when he wrote no yes / no but listed chart numbers (listed), his way: each chart in `of` (that
    day's chart numbers) is yes if listed, else no. A listed number with no chart that day stays in, so record()
    tells him. none_listed: every chart in `of` is no (checked first: "no zones today" holds a "no")."""
    if none_listed(words):
        return [(n, "no") for n in of]
    got = read(words)
    if got:
        return got
    mine = listed(words)
    if not mine:
        return []
    return [(n, "yes" if n in mine else "no") for n in of] + [(n, "yes") for n in mine if n not in of]


def day_numbers(question: str, base: Path | None = None) -> list[int]:
    """The chart numbers sent on the same day as the chart `question` (LBL-<word>-<YYYYMMDD>-<NN>)."""
    word, day, _n = parts(question)
    return sorted(int(c["n"]) for c in read_charts(word, base) if str(c.get("day", "")).replace("-", "") == day)


def newest_waiting(base: Path | None = None) -> str | None:
    """The first chart of the NEWEST day's label charts (any word), if one of that day's charts still waits for his
    answer; else None. A list of numbers that is not a reply goes to this day (bot._label_list)."""
    best: tuple[str, str] | None = None
    root = Path(base or config.LABELS_WORK_DIR)
    for path in sorted(root.glob("*/charts.jsonl")) if root.is_dir() else []:
        word = path.parent.name
        charts = read_charts(word, base)
        if not charts:
            continue
        day = max(str(c.get("day", "")) for c in charts)
        todays = sorted((c for c in charts if str(c.get("day", "")) == day), key=lambda c: int(c.get("n", 0)))
        answered = latest(word, base)
        if any(c["qid"] not in answered for c in todays) and (best is None or day > best[0]):
            best = (day, todays[0]["qid"])
    return best[1] if best else None


def more_than_labels(words: str) -> bool:
    """Does the reply say more than its labels (a reason, a correction)? Then it is ALSO a lesson."""
    text = str(words or "").replace("’", "'")
    if none_listed(text):                              # "none of them are mine" says only its labels
        text = "\n".join(ln for ln in text.splitlines() if not _none_line(ln))
    rest = _PAIR.sub(" ", " ".join(text.split()))
    left = [w for w in re.findall(r"[a-z']+", rest.lower()) if w not in _FILLER]
    return len(left) >= MORE_WORDS


def replies_file(word: str, base: Path | None = None) -> Path:
    return Path(base or config.LABELS_WORK_DIR) / word / "replies.jsonl"


def record(base: Path | None, *, question: str, got: list[tuple[int | None, str]], words: str, message_id: int,
           when: dt.datetime, as_list: bool = False, reply: bool = True) -> dict:
    """Append one line per chart he answered. A number answers chart N of the same day as `question`; no number
    answers `question` itself; a number with no chart that day is not saved (and he is told). as_list: `got` came
    from his list of numbers (read_all), so the bot says it back as "your zones ... / not yours ...". reply=False: his
    message was no reply to a picture (replied_to = None). Returns {"file", "rows", "missing", "said"} (said = the
    line the bot replies with)."""
    word, day, _n = parts(question)
    sent = {c["qid"] for c in read_charts(word, base)}
    rows, missing = [], []
    for n, label in got:
        target = question if n is None else f"LBL-{word}-{day}-{n:02d}"
        if target != question and target not in sent:
            missing.append(n)
            continue
        rows.append({"qid": target, "label": label, "words": words, "replied_to": question if reply else None,
                     "message_id": message_id, "at": when.isoformat(timespec="seconds")})
    path = replies_file(word, base)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    answered = {r["qid"] for r in read_replies(word, base)}
    if as_list and rows:
        def nums(label: str) -> str:
            return ", ".join(str(parts(r["qid"])[2]) for r in rows if r["label"] == label) or "none"
        said = (f"Saved for the charts of {dt.datetime.strptime(day, '%Y%m%d'):%d %b}: your zones {nums('yes')}; "
                f"not yours {nums('no')}.")
    elif rows and all(r["qid"] == question for r in rows):
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
