"""Everything Tejas already taught, cut into pieces the search can find and cite (L8b: his #129 rule as code).

His words, LEARNED #129 (21 Sep 2026): "anytime when you have a doubt you should go through the strategy.md and
learnings ... all these things are the source of truth that you should have check first ... read after reading every
line then only you should ask me".

Each piece has:
    id      the citation a reply can quote: "LEARNED #129" · "LEARNED #21 (6 Sep)" (the two #21-#24) ·
            "understanding §66bt" · "strategy.md §1 line 73" · "strategy-code §7" · "DECISIONS C2" ·
            "AMBIGUITY-REGISTER A2" · "EVERYTHING-YOU-TAUGHT line 45" · "rules.py <name>" · "settings.py <name>" ·
            "inbox 2026-09-29-1402-telegram" (not numbered yet; once numbered, its LEARNED entry keeps that id as a
            second name in `see`) · "idea-001 answers" · "picture Q98-D8-break-distance"
    tier    the order of trust (CLAUDE.md, newest wins): 1 his words (LEARNED, the Telegram archive - filed under its
            LEARNED number -, the inbox, idea answers, his pictures) · 2 strategy-understanding · 3 rules.py and his
            settings · 4 strategy.md · 5 strategy-code, DECISIONS, AMBIGUITY-REGISTER, EVERYTHING-YOU-TAUGHT and
            Claude's settings. old/ is never searched.
    date    the day it was said, when the file says it (newest first inside a tier)
    segs    the lines it came from, (line number, text), so a quote names its line

Two ways of finding: BM25 on the words (always), and embeddings (meaning, not only the same words): a local
sentence-transformers model by default, or Voyage AI when VOYAGE_API_KEY is set. The embeddings are cached in
data/cache/search/ under a file named by the model and its settings, one vector per piece TEXT, so an edited file
re-embeds only its changed pieces and a different model never reads another's vectors (likhith.md B19).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from machine import config

# ------------------------------------------------------------------------------------------------ the sources
LEARNED_MD = Path("strategy/LEARNED.md")
UNDERSTANDING_MD = Path("strategy/strategy-understanding.md")
STRATEGY_MD = Path("strategy/strategy.md")
STRATEGY_CODE_MD = Path("strategy/strategy-code.md")
EVERYTHING_MD = Path("strategy/EVERYTHING-YOU-TAUGHT.md")
DECISIONS_MD = Path("strategy/DECISIONS.md")
AMBIGUITY_MD = Path("strategy/AMBIGUITY-REGISTER.md")
TELEGRAM_JSONL = Path("strategy/telegram-archive.jsonl")
INBOX_DIR = Path("strategy/learned-inbox")
IDEAS_DIR = Path("plan/work/ideas")
RULES_PY = Path("machine/rules.py")
SETTINGS_PY = Path("machine/settings.py")
# Where his pictures sit (a picture is found through the LEARNED entry that names it). old/ is read only for the
# FILE NAMES of pictures here, never searched.
PICTURE_DIRS = (Path("strategy/pictures"), Path("old/ml/agent/logs"))

TIER = {"LEARNED": 1, "inbox": 1, "answers": 1, "picture": 1, "understanding": 2, "rules.py": 3,
        "settings.py (his)": 3, "strategy.md": 4, "strategy-code": 5, "DECISIONS": 5, "AMBIGUITY-REGISTER": 5,
        "EVERYTHING-YOU-TAUGHT": 5, "settings.py": 5}

MAX_WORDS = 160          # a piece is at most about this many words (Claude's number: one idea, fits the model)
MONTHS = ("", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class Piece:
    id: str
    source: str                          # a key of TIER
    tier: int
    date: str                            # "2026-09-21" or ""
    path: str                            # repo-relative, forward slashes
    line: int                            # first line of the piece in its file (1-based)
    title: str                           # the heading it sits under
    segs: tuple[tuple[int, str], ...]    # (line number, text) - what is searched and quoted
    see: tuple[str, ...] = ()            # a picture: the LEARNED entries that name it

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.segs)

    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}"[:2000]


# ------------------------------------------------------------------------------------------------ cutting text up
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long(no: int, text: str, max_words: int) -> list[tuple[int, str]]:
    """One long line (his voice notes are one line of 400 words) as pieces of at most max_words words."""
    out, cur, n = [], [], 0
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        while len(words) > max_words:                      # a sentence with no full stop in 400 words
            if cur:
                out.append((no, " ".join(cur)))
                cur, n = [], 0
            out.append((no, " ".join(words[:max_words])))
            words = words[max_words:]
        if n + len(words) > max_words and cur:
            out.append((no, " ".join(cur)))
            cur, n = [], 0
        cur.extend(words)
        n += len(words)
    if cur:
        out.append((no, " ".join(cur)))
    return out


def _passages(body: list[tuple[int, str]], max_words: int = MAX_WORDS) -> list[tuple[tuple[int, str], ...]]:
    """Lines -> paragraphs (a blank line ends one) -> pieces of at most max_words words, never cutting a paragraph
    that fits."""
    paragraphs: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for no, raw in body:
        text = raw.strip()
        if not text or text == "---" or re.fullmatch(r"\|?[\s:|-]+\|?", text):
            if cur:
                paragraphs.append(cur)
                cur = []
            continue
        cur.append((no, text))
    if cur:
        paragraphs.append(cur)

    out: list[tuple[tuple[int, str], ...]] = []
    pack: list[tuple[int, str]] = []
    n = 0
    for para in paragraphs:
        size = sum(len(t.split()) for _, t in para)
        if size > max_words:                               # a long paragraph: its own pieces ...
            chunk: list[tuple[int, str]] = []
            m = 0
            if pack and n <= max_words // 4:               # ... a short heading before it rides along
                chunk, m = pack, n
            elif pack:
                out.append(tuple(pack))
            pack, n = [], 0
            segs: list[tuple[int, str]] = []
            for no, t in para:
                segs.extend(_split_long(no, t, max_words))
            for no, t in segs:
                w = len(t.split())
                if m + w > max_words and chunk:
                    out.append(tuple(chunk))
                    chunk, m = [], 0
                chunk.append((no, t))
                m += w
            if chunk:
                out.append(tuple(chunk))
            continue
        if n + size > max_words and pack:
            out.append(tuple(pack))
            pack, n = [], 0
        pack.extend(para)
        n += size
    if pack:
        out.append(tuple(pack))
    return out


def _read_lines(root: Path, rel: Path) -> list[str]:
    path = root / rel
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _slug(title: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", title)
    return "-".join(w.lower() for w in words[:2]) or "top"


def _date_in(text: str) -> str:
    m = re.search(r"\b(20\d\d-\d\d-\d\d)\b", text)
    return m.group(1) if m else ""


def _day(date: str) -> str:
    """'2026-09-06' -> '6 Sep' (how the plan cites the two #21-#24)."""
    return f"{int(date[8:10])} {MONTHS[int(date[5:7])]}"


def _pieces(pid: str, source: str, date: str, rel: Path, title: str, body: list[tuple[int, str]],
            see: tuple[str, ...] = ()) -> list[Piece]:
    return [Piece(pid, source, TIER[source], date, rel.as_posix(), segs[0][0], title, segs, see)
            for segs in _passages(body)]


# ------------------------------------------------------------------------------------------------ LEARNED.md
LEARNED_HEAD = re.compile(r"^## #(\d+)\s+(?:·|—)\s+(.*)$")


@dataclass
class _Entry:
    n: int
    date: str
    line: int
    title: str
    body: list[tuple[int, str]] = field(default_factory=list)


def _learned_entries(lines: list[str]) -> list[_Entry]:
    entries: list[_Entry] = []
    for no, line in enumerate(lines, 1):
        m = LEARNED_HEAD.match(line)
        if m:
            rest = m.group(2)
            title = re.sub(r"\d{4}-\d\d-\d\d( \d\d:\d\d)?", "", rest).strip(" ·.—,")
            entries.append(_Entry(int(m.group(1)), _date_in(rest), no, title))
        elif entries:                                      # the file's own preamble (before #1) is not a lesson
            entries[-1].body.append((no, line))
    return entries


def learned_id(n: int, date: str, twice: set[int]) -> str:
    """'LEARNED #129'; a number used twice (#21-#24) carries its day: 'LEARNED #21 (6 Sep)'."""
    return f"LEARNED #{n} ({_day(date)})" if n in twice and date else f"LEARNED #{n}"


def _numbers_used_twice(entries: list[_Entry]) -> set[int]:
    count = Counter(e.n for e in entries)
    return {n for n, c in count.items() if c > 1}


# An entry numbered by learned-merge (L9a) ends with the inbox file it came from: its old id "inbox <stem>" stays a
# second name for it (Piece.see), so a citation made while it waited still counts.
_MERGED_FROM = re.compile(r"\(from `strategy/learned-inbox/merged/([^`/]+)\.md`")


def learned(root: Path) -> list[Piece]:
    entries = _learned_entries(_read_lines(root, LEARNED_MD))
    twice = _numbers_used_twice(entries)
    out: list[Piece] = []
    for e in entries:
        pid = learned_id(e.n, e.date, twice)
        was = tuple(f"inbox {m.group(1)}" for _, t in e.body if (m := _MERGED_FROM.search(t)))
        out += _pieces(pid, "LEARNED", e.date, LEARNED_MD, f"{pid} — {e.title}", [(e.line, e.title)] + e.body,
                       see=was)
    return out


# ------------------------------------------------------------------------------------------------ pictures
_CARD = re.compile(r"on card \*\*([^*]+)\*\*(?:\s*\(([^)]*)\))?")
_IMAGE = re.compile(r"([\w.-]+)\.(?:png|jpe?g)\b", re.IGNORECASE)


def _picture_files(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for d in PICTURE_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for dirpath, _, files in os.walk(base):
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() in (".png", ".jpg", ".jpeg"):
                    found.setdefault(stem, (Path(dirpath) / f).relative_to(root).as_posix())
    return found


def pictures(root: Path) -> list[Piece]:
    """A picture is found through the LEARNED entries that name it: the card it was sent on ("on card
    **Q98-D8-break-distance**") or its file ("Q64-his-answer.png"). The piece holds the picture's name in words, what
    it asked, and the titles of the entries - his words themselves stay in the LEARNED piece it points to."""
    entries = _learned_entries(_read_lines(root, LEARNED_MD))
    twice = _numbers_used_twice(entries)
    files = _picture_files(root)
    seen: dict[str, dict] = {}
    for e in entries:
        pid = learned_id(e.n, e.date, twice)
        text = "\n".join([e.title] + [t for _, t in e.body])
        names = [(m.group(1).strip(), m.group(2) or "") for m in _CARD.finditer(text)]
        names += [(m.group(1), "") for m in _IMAGE.finditer(text)]
        for stem, about in names:
            p = seen.setdefault(stem, {"see": [], "about": [], "titles": [], "date": "", "line": e.line})
            if pid not in p["see"]:
                p["see"].append(pid)
                p["titles"].append(e.title)
            if about and about not in p["about"]:
                p["about"].append(about)
            p["date"] = max(p["date"], e.date)
    out = []
    for stem, p in seen.items():
        words = " ".join(re.split(r"[-_]+", stem))
        path = files.get(stem, "")
        segs = [(p["line"], f"picture {words}.")] + [(p["line"], a) for a in p["about"]]
        segs += [(p["line"], t) for t in p["titles"]]
        segs.append((p["line"], "See " + ", ".join(p["see"]) + (f". File: {path}" if path else ". File not kept.")))
        body_words = 0
        kept = []
        for s in segs:                                     # one piece per picture
            body_words += len(s[1].split())
            if body_words > MAX_WORDS * 2:
                break
            kept.append(s)
        out.append(Piece(f"picture {stem}", "picture", TIER["picture"], p["date"],
                         path or LEARNED_MD.as_posix(), p["line"] if not path else 0, f"picture {stem}",
                         tuple(kept), tuple(p["see"])))
    return out


# ------------------------------------------------------------------------------------------------ markdown by section
def _sections(root: Path, rel: Path, label: str, head: re.Pattern, key_prefix: str = "",
              other_level: int = 2) -> list[Piece]:
    """Cut a markdown file at its numbered headings (head: group 1 = key, group 2 = title). Any other heading of
    level <= other_level starts a section named by its first two words; deeper ones stay inside. Headings inside
    ``` blocks are code comments, not headings."""
    lines = _read_lines(root, rel)
    sections: list[list] = [["top", 1, "", []]]
    in_fence = False
    for no, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            m = head.match(line)
            if m:
                sections.append([key_prefix + m.group(1), no, m.group(2).strip(), []])
                continue
            h = re.match(r"^(#{1,6})\s+(.*)$", line)
            if h and len(h.group(1)) <= other_level and no > 1:
                sections.append([_slug(h.group(2)), no, h.group(2).strip(), []])
                continue
        sections[-1][3].append((no, line))
    used = Counter(s[0] for s in sections)
    out: list[Piece] = []
    for key, no, title, body in sections:
        pid = f"{label} {key}" if used[key] == 1 else f"{label} {key} (line {no})"
        out += _pieces(pid, label, _date_in(title), rel, f"{pid} — {title}".rstrip(" —"),
                       ([(no, title)] if title else []) + body)
    return out


def understanding(root: Path) -> list[Piece]:
    return _sections(root, UNDERSTANDING_MD, "understanding", re.compile(r"^#{2,4}\s+§?(\d+[a-z]*)\.\s*(.*)$"),
                     key_prefix="§", other_level=1)


def strategy_code(root: Path) -> list[Piece]:
    return _sections(root, STRATEGY_CODE_MD, "strategy-code", re.compile(r"^##\s+(\d+)\.\s*(.*)$"), key_prefix="§")


def decisions(root: Path) -> list[Piece]:
    return _sections(root, DECISIONS_MD, "DECISIONS", re.compile(r"^##\s+([A-Z]\d+[a-z]?)\s*·\s*(.*)$"))


def ambiguity(root: Path) -> list[Piece]:
    return _sections(root, AMBIGUITY_MD, "AMBIGUITY-REGISTER",
                     re.compile(r"^###\s+([A-Z]\d+[a-z]?)\b[^·]*·\s*(.*)$"))


# ------------------------------------------------------------------------------------------------ markdown by line
def _by_line(root: Path, rel: Path, label: str, with_section: bool) -> list[Piece]:
    """strategy.md and EVERYTHING-YOU-TAUGHT.md: every line is one rule, cited by its line number."""
    out: list[Piece] = []
    section, title = "top", ""
    for no, line in enumerate(_read_lines(root, rel), 1):
        h = re.match(r"^(#{1,3})\s+(.*)$", line)
        if h:
            title = h.group(2).strip()
            m = re.match(r"(\d+)\.", title)
            section = f"§{m.group(1)}" if m else _slug(title)
            continue
        text = line.strip()
        if not text or text == "---" or re.fullmatch(r"\|?[\s:|-]+\|?", text):
            continue
        pid = f"{label} {section} line {no}" if with_section else f"{label} line {no}"
        for segs in _passages([(no, text)]):
            out.append(Piece(pid, label, TIER[label], "", rel.as_posix(), no, title, segs))
    return out


def strategy_md(root: Path) -> list[Piece]:
    return _by_line(root, STRATEGY_MD, "strategy.md", with_section=True)


def everything(root: Path) -> list[Piece]:
    return _by_line(root, EVERYTHING_MD, "EVERYTHING-YOU-TAUGHT", with_section=False)


# ------------------------------------------------------------------------------------------------ rules.py, settings.py
def _line_of(lines: list[str], *needles: str) -> int:
    for no, line in enumerate(lines, 1):
        if any(n in line for n in needles):
            return no
    return 1


def rules_and_settings(root: Path) -> list[Piece]:
    """His fixed rules (tier 3) and the settings: his (tier 3) or Claude's numbers (tier 5). Read from the modules
    of THIS checkout, so the index always matches the code the machine runs."""
    from machine import rules, settings

    out: list[Piece] = []
    rl = _read_lines(root, RULES_PY)
    for name, r in rules.ALL.items():
        words = f" His words: {r.words}" if r.words else ""
        text = (f"{name.replace('_', ' ')}: {r.means}{words} Taught: {r.taught}. Value: {r.value}. "
                f"Who chose the number: {r.number}.")
        no = _line_of(rl, f'"{name}": Rule(')
        out.append(Piece(f"rules.py {name}", "rules.py", TIER["rules.py"], "", RULES_PY.as_posix(), no,
                         f"his fixed rule: {name}", tuple((no, t) for t in _chunk_words(text))))
    sl = _read_lines(root, SETTINGS_PY)
    for name, s in settings.ALL.items():
        text = f"{name.replace('_', ' ')}: {s.means} Value: {s.value}. Who chose it: {s.who}."
        no = _line_of(sl, f'"{name}": _s(', f'("{name}",')
        source = "settings.py (his)" if s.status == "his" else "settings.py"
        out.append(Piece(f"settings.py {name}", source, TIER[source], "", SETTINGS_PY.as_posix(), no,
                         f"setting: {name}", tuple((no, t) for t in _chunk_words(text))))
    return out


def _chunk_words(text: str) -> list[str]:
    return [t for _, t in _split_long(0, text, MAX_WORDS * 2)]


# ------------------------------------------------------------------------------------------------ his raw words
def telegram(root: Path) -> list[Piece]:
    """strategy/telegram-archive.jsonl: everything he typed, filed under the LEARNED number it became (row n = #n;
    the two #21-#24 told apart by their day), so a hit cites the lesson, and the quote names the archive row."""
    path = root / TELEGRAM_JSONL
    if not path.is_file():
        return []
    twice = _numbers_used_twice(_learned_entries(_read_lines(root, LEARNED_MD)))
    out: list[Piece] = []
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        date = str(row.get("ts", ""))[:10]
        pid = learned_id(int(row["n"]), date, twice)
        title = f"{pid} — Telegram archive row {no} ({row.get('kind', '')})"
        out += _pieces(pid, "LEARNED", date, TELEGRAM_JSONL, title, [(no, str(row.get("text", "")))])
    return out


def inbox(root: Path) -> list[Piece]:
    """strategy/learned-inbox/*.md: his newest words, not numbered yet (only main numbers them, L9a)."""
    out: list[Piece] = []
    base = root / INBOX_DIR
    if not base.is_dir():
        return out
    for f in sorted(base.glob("*.md")):
        if f.name.lower() == "readme.md":
            continue
        rel = f.relative_to(root)
        lines = f.read_text(encoding="utf-8").splitlines()
        title = next((l.lstrip("# ").strip() for l in lines if l.startswith("#")), f.stem)
        body = [(no, l) for no, l in enumerate(lines, 1) if not l.startswith("# ")]
        pid = f"inbox {f.stem}"
        out += _pieces(pid, "inbox", _date_in(f.stem), rel, f"{pid} — {title}", body)
    return out


def answers(root: Path) -> list[Piece]:
    """plan/work/ideas/<idea>/answers.md: his answers to an idea's questions, word for word (L9b appends them)."""
    out: list[Piece] = []
    base = root / IDEAS_DIR
    if not base.is_dir():
        return out
    for f in sorted(base.glob("*/answers.md")):
        rel = f.relative_to(root)
        idea = f.parent.name
        lines = f.read_text(encoding="utf-8").splitlines()
        body = [(no, l) for no, l in enumerate(lines, 1)
                if not l.startswith("# ") and not l.startswith("Tejas's answers, word for word")]
        if not any(l.strip() for _, l in body):
            continue
        sections: list[tuple[str, int, list]] = [("", 1, [])]
        for no, l in body:
            if l.startswith("## "):
                sections.append((l[3:].strip(), no, []))
            else:
                sections[-1][2].append((no, l))
        for title, no, sec in sections:
            if not any(l.strip() for _, l in sec):
                continue
            pid = f"{idea} answers" + (f" — {title}" if title else "")
            out += _pieces(pid, "answers", _date_in(title), rel, pid, sec)
    return out


SOURCES = (learned, telegram, inbox, answers, pictures, understanding, rules_and_settings, strategy_md,
           strategy_code, decisions, ambiguity, everything)


def all_pieces(root: Path | None = None) -> list[Piece]:
    root = Path(root or config.ROOT)
    out: list[Piece] = []
    for source in SOURCES:
        out += source(root)
    return out


# ------------------------------------------------------------------------------------------------ words (BM25)
STOP = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing done down during each few for from further had has have having he her here him his how i
if in into is it its itself just me more most my no nor not now of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your yours uh um okay ok yeah like actually
also mean means see thing things one get got go going
""".split())
# The same thing written two ways in his documents. Both sides of a search read the same word.
SAME = {
    "fvg": "fair value gap", "fvgs": "fair value gap", "bos": "break structure", "ob": "order block",
    "ath": "all time high", "htf": "higher timeframe", "ltf": "lower timeframe", "tf": "timeframe",
    "sl": "stop loss", "tp": "target", "eqh": "equal high", "eql": "equal low", "tl": "trendline",
    "h1": "hour", "1h": "hour", "m5": "5 min", "5m": "5 min", "m1": "1 min", "rs": "rupee", "inr": "rupee",
    "adr": "adr average daily range", "rr": "risk reward", "ny": "new york", "ist": "india time",
}
PHRASES = ((re.compile(r"time[\s-]+frames?"), "timeframe"), (re.compile(r"trend[\s-]+lines?"), "trendline"),
           (re.compile(r"₹"), " rupee "), (re.compile(r"(\d+)-min(ute)?s?\b"), r"\1 min"),
           (re.compile(r"\bfive[\s-]minutes?\b"), "5 min"), (re.compile(r"\bone[\s-]hour\b"), "hour"))
IRREGULAR = {"swept": "sweep", "took": "take", "taken": "take", "broke": "break", "broken": "break",
             "went": "go", "gone": "go", "lows": "low", "highs": "high", "best": "good", "better": "good"}
_WORD = re.compile(r"[a-z0-9]+")


def stem(w: str) -> str:
    w = IRREGULAR.get(w, w)
    if len(w) <= 3 or w.isdigit():
        return w
    for suffix in ("ings", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suffix) and not w.endswith("ss") and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def terms(text: str) -> list[str]:
    """Words -> search terms: lower case, the same thing written two ways made one, filler dropped, light stems."""
    t = text.lower()
    for pattern, repl in PHRASES:
        t = pattern.sub(repl, t)
    out: list[str] = []
    for w in _WORD.findall(t):
        for x in SAME.get(w, w).split():
            if x not in STOP:
                out.append(stem(x))
    return out


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75: the usual values, Claude's choice)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg or 1.0))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                s[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        return s


# ------------------------------------------------------------------------------------------------ embeddings
class Embedder(Protocol):
    recipe: str                                            # model + settings: names the cache file

    def documents(self, texts: list[str]) -> np.ndarray: ...
    def query(self, text: str) -> np.ndarray: ...


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class LocalEmbedder:
    """A sentence-transformers model on this PC's CPU (downloads once into the Hugging Face cache)."""

    def __init__(self, model: str = config.SEARCH_LOCAL_MODEL, query_prefix: str = config.SEARCH_LOCAL_QUERY_PREFIX):
        self.model_name, self.query_prefix = model, query_prefix
        self.recipe = f"local|{model}|{query_prefix}|unit"
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def documents(self, texts: list[str]) -> np.ndarray:
        return _unit(self._load().encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True))

    def query(self, text: str) -> np.ndarray:
        return _unit(self._load().encode([self.query_prefix + text], normalize_embeddings=True))[0]


class VoyageEmbedder:
    """Voyage AI over its HTTP API (no extra package). Needs VOYAGE_API_KEY; never sends anything else."""

    URL = "https://api.voyageai.com/v1/embeddings"
    BATCH = 128

    def __init__(self, key: str, model: str = config.SEARCH_VOYAGE_MODEL):
        self.key, self.model = key, model
        self.recipe = f"voyage|{model}|input_type|unit"

    def _post(self, texts: list[str], input_type: str) -> np.ndarray:
        import urllib.request

        body = json.dumps({"input": texts, "model": self.model, "input_type": input_type}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())["data"]
        return _unit([d["embedding"] for d in sorted(data, key=lambda d: d["index"])])

    def documents(self, texts: list[str]) -> np.ndarray:
        parts = [self._post(texts[i:i + self.BATCH], "document") for i in range(0, len(texts), self.BATCH)]
        return np.vstack(parts) if parts else np.zeros((0, 1), np.float32)

    def query(self, text: str) -> np.ndarray:
        return self._post([text], "query")[0]


def choose_embedder(mode: str = "auto") -> tuple[Embedder | None, str]:
    """auto: SEARCH_EMBEDDINGS=off -> none; VOYAGE_API_KEY -> Voyage; else the local model if installed.
    Returns the embedder (or None = words only) and one line saying which, for every result."""
    if mode == "off" or (mode == "auto" and (config.env("SEARCH_EMBEDDINGS") or "").lower() == "off"):
        return None, "words only (BM25): embeddings switched off"
    key = config.env("VOYAGE_API_KEY") if mode in ("auto", "voyage") else None
    if key:
        return VoyageEmbedder(key), f"words (BM25) + Voyage AI {config.SEARCH_VOYAGE_MODEL}"
    if mode == "voyage":
        return None, "words only (BM25): VOYAGE_API_KEY is not set"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None, ("words only (BM25): sentence-transformers is not installed "
                      "(python -m pip install -r requirements-search.txt)")
    return LocalEmbedder(), f"words (BM25) + local model {config.SEARCH_LOCAL_MODEL}"


def _cache_file(recipe: str, cache_dir: Path) -> Path:
    return cache_dir / f"emb-{hashlib.sha256(recipe.encode()).hexdigest()[:16]}.npz"


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def embed_pieces(pieces: list[Piece], embedder: Embedder, cache_dir: Path | None = None) -> np.ndarray:
    """One vector per piece. Vectors are kept per piece TEXT in a file named by the embedder's recipe: a changed
    piece is embedded again, an unchanged one is read back, and another model's vectors are never mixed in."""
    cache_dir = Path(cache_dir or config.SEARCH_CACHE_DIR)
    file = _cache_file(embedder.recipe, cache_dir)
    known: dict[str, np.ndarray] = {}
    if file.is_file():
        with np.load(file, allow_pickle=False) as z:
            if str(z["recipe"]) == embedder.recipe:
                known = dict(zip(z["keys"].tolist(), z["vecs"]))
    texts = [p.embed_text() for p in pieces]
    keys = [_key(t) for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in known}.items())
    if missing:
        vecs = embedder.documents([t for _, t in missing])
        known.update({k: v for (k, _), v in zip(missing, vecs)})
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp.npz")
        live = set(keys)
        kept = sorted(k for k in known if k in live)
        np.savez(tmp, recipe=np.array(embedder.recipe), keys=np.array(kept),
                 vecs=np.array([known[k] for k in kept], dtype=np.float32))
        os.replace(tmp, file)                              # never a half-written cache
    return np.array([known[k] for k in keys], dtype=np.float32)


# ------------------------------------------------------------------------------------------------ the index
@dataclass
class Index:
    pieces: list[Piece]
    bm25: BM25
    vectors: np.ndarray | None
    embedder: Embedder | None
    how: str                                               # one line: which search ran


def build(root: Path | None = None, embeddings: str = "auto", cache_dir: Path | None = None,
          embedder: Embedder | None = None) -> Index:
    pieces = all_pieces(root)
    bm25 = BM25([terms(p.embed_text()) for p in pieces])
    how = "words (BM25) + the given embedder"
    if embedder is None:
        embedder, how = choose_embedder(embeddings)
    vectors = embed_pieces(pieces, embedder, cache_dir) if embedder is not None and pieces else None
    return Index(pieces, bm25, vectors, embedder, how)


_BUILT: dict[tuple, Index] = {}


def fingerprint(root: Path | None = None) -> str:
    """Every searched file's size and time, so a changed file rebuilds the index in a running process."""
    root = Path(root or config.ROOT)
    h = hashlib.sha256()
    files: Iterable[Path] = [LEARNED_MD, UNDERSTANDING_MD, STRATEGY_MD, STRATEGY_CODE_MD, EVERYTHING_MD, DECISIONS_MD,
                             AMBIGUITY_MD, TELEGRAM_JSONL, RULES_PY, SETTINGS_PY]
    files = list(files) + sorted(p.relative_to(root) for p in (root / INBOX_DIR).glob("*.md"))
    files += sorted(p.relative_to(root) for p in (root / IDEAS_DIR).glob("*/answers.md"))
    for rel in files:
        f = root / rel
        st = f.stat() if f.is_file() else None
        h.update(f"{rel.as_posix()}|{st.st_size if st else -1}|{st.st_mtime_ns if st else -1}\n".encode())
    return h.hexdigest()[:16]


def get(root: Path | None = None, embeddings: str = "auto") -> Index:
    """The index for this checkout, built once per process and again when a searched file changes."""
    key = (str(Path(root or config.ROOT)), embeddings, fingerprint(root))
    if key not in _BUILT:
        _BUILT.clear()
        _BUILT[key] = build(root, embeddings)
    return _BUILT[key]
