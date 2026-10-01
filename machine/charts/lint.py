"""One check per rule of Tejas's, run on every picture BEFORE anyone sees it (L4).

    from machine.charts import draw as D, lint
    drawn = D.render(pic, "x.jpg")
    problems = lint.check(drawn)         # [] = it may go to him; else every broken rule, in plain words
    lint.passed(drawn)                   # the same, but raises PictureRejected
    lint.mark_shown(drawn, "sent on Telegram")   # ONLY after he has really been shown it (#117)

Each check reads what draw.py really drew (the pixel boxes of every word, candle, line and gap, taken from the
rendered picture), and one check reads the file itself. `python -m machine.charts.lint` lists the rules.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image

from machine import config
from machine.charts.draw import CLOCK_WORDS, TF_WORDS, Drawn, ItemOnPicture, TextOnPicture
from machine.tester.models import TF_MINUTES


@dataclass(frozen=True)
class Problem:
    rule: str
    says: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.says}"


class PictureRejected(Exception):
    pass


RULES: dict[str, tuple[Callable[..., list[Problem]], str]] = {}

CHART_WORDS = {"tag", "name", "touch", "number", "span"}     # words drawn INSIDE a chart
LINE_KINDS = {"Level", "Line", "Fib", "Position", "Arrow", "Moment"}


def rule(name: str, his: str):
    def add(fn):
        RULES[name] = (fn, his)
        return fn
    return add


def check(drawn: Drawn, *, shown: Path | None = None, again_because: str | None = None) -> list[Problem]:
    """Every rule on one picture. `shown`: the list of pictures he has seen (default plan/work/pictures-shown.jsonl).
    `again_because`: the same trade may be drawn again only with a reason (his own "fix them and send them
    again", #110) - never the same pixels."""
    out: list[Problem] = []
    for name, (fn, _his) in RULES.items():
        if name == "never_twice":
            out += fn(drawn, shown=shown, again_because=again_because)
        else:
            out += fn(drawn)
    return out


def passed(drawn: Drawn, **kw) -> Drawn:
    problems = check(drawn, **kw)
    if problems:
        raise PictureRejected(f"{drawn.path.name}: " + "; ".join(map(str, problems)))
    return drawn


def _words(drawn: Drawn) -> list[TextOnPicture]:
    return [t for t in drawn.texts if t.role != "tick"]


def _items_for(drawn: Drawn, text: TextOnPicture) -> list[ItemOnPicture]:
    """A word inside a chart must be drawn in THAT chart; a word in the heading, question or footer, in any.
    A point's keyword ("B1 gap") may also be drawn in that point's own 5-minute close-up - that is where he checks
    "the fair value gap actually was there or not" (#121)."""
    panels = {text.panel}
    if text.role == "tag" and text.names is not None:
        m = drawn.items[text.names]
        panels |= {i.panel for i in drawn.items
                   if i.kind == "Marker" and drawn.panels[i.panel].role == "point"
                   and i.info.get("tag") == m.info.get("tag") and i.info.get("time") == m.info.get("time")}
    return [i for i in drawn.items if text.panel is None or i.panel in panels]


def _short(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= 40 else s[:37] + "..."


# =================================================================================================================
# the rules
# =================================================================================================================

@rule("label_overlap", "#115 labels never cover the line they name · #116 nothing on the price action, "
                       "words in the empty space · #110 nothing covers the gap")
def label_overlap(drawn: Drawn) -> list[Problem]:
    out = []
    texts = drawn.texts
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if a.box.overlaps(b.box):
                out.append(Problem("label_overlap", f"'{_short(a.text)}' sits on '{_short(b.text)}'"))
    for t in texts:
        if t.role not in CHART_WORDS or t.panel is None:
            continue
        panel = drawn.panels[t.panel]
        if any(t.box.overlaps(c) for c in panel.candles):
            out.append(Problem("label_overlap", f"'{_short(t.text)}' sits on the candles"))
        for k, item in enumerate(drawn.items):
            if item.panel != t.panel:
                continue
            if item.kind == "Box" and any(t.box.overlaps(r) for r in item.boxes):
                out.append(Problem("label_overlap", f"'{_short(t.text)}' covers the gap"))
            if item.kind in LINE_KINDS and any(t.box.holds(x, y) for path in item.paths for x, y in path):
                what = "the line it names" if t.names == k else f"the {item.kind.lower()} '{_short(item.name)}'"
                out.append(Problem("label_overlap", f"'{_short(t.text)}' covers {what}"))
    for c in drawn.crowded:
        out.append(Problem("label_overlap", f"no empty room for '{c}' - the picture is too crowded"))
    return out


@rule("words_fit", "#124 a line of words ran off the right edge - every word stays on the picture")
def words_fit(drawn: Drawn) -> list[Problem]:
    w, h = drawn.size
    return [Problem("words_fit", f"'{_short(t.text)}' runs off the picture")
            for t in drawn.texts if t.box.x0 < -1 or t.box.y0 < -1 or t.box.x1 > w + 1 or t.box.y1 > h + 1]


def _drawn_boxes(kind: str):
    return lambda its: [i for i in its if i.kind == kind and i.visible]


def _numbered_lines(its: list[ItemOnPicture]) -> list[ItemOnPicture]:
    return [i for i in its if i.kind == "Line" and i.visible and i.info.get("touches", 0) >= 2
            and i.info.get("numbered", 0) == i.info.get("touches")]


def _triangles(its: list[ItemOnPicture]) -> list[ItemOnPicture]:
    groups: dict[str, list[ItemOnPicture]] = {}
    for i in its:
        if i.kind == "Line" and i.visible and i.info.get("group"):
            groups.setdefault(i.info["group"], []).append(i)
    return [g[0] for g in groups.values() if len(g) == 2 and (g[0].info.get("inside_share") or 0) >= 0.8]


def _fibs(its: list[ItemOnPicture]) -> list[ItemOnPicture]:
    return [i for i in its if i.kind == "Fib" and i.visible and len(i.info.get("numbers", [])) >= 2]


def _levels_named(word: str):
    return lambda its: [i for i in its if i.kind == "Level" and i.visible and word in i.name.lower()]


# a word on the picture -> what must be drawn for it (#113 rule 3, his words on M01-M05 and R01-R04)
NEEDS: list[tuple[str, str, Callable[[list[ItemOnPicture]], list], str]] = [
    ("gap", r"\bgaps?\b", _drawn_boxes("Box"), "the gap box"),
    ("zone", r"(?<!no-close )\bzones?\b", _drawn_boxes("Zone"), "the zone's box"),
    ("trendline", r"\btrend ?lines?\b", _numbered_lines, "the line with its touches numbered"),
    ("triangle", r"\btriangles?\b", _triangles, "both lines, with the price inside them"),
    ("Fibonacci", r"\bfib(onacci)?\b", _fibs, "its lines, each with its number"),
    ("ADR", r"\badr\b|moves in a day", _drawn_boxes("ADRBand"), "the ADR band"),
    ("equal highs / lows", r"\bequal (highs|lows)\b", _numbered_lines, "the line through them, touches numbered"),
    ("target", r"\btargets?\b", _levels_named("target"), "the target line"),
    ("stop", r"\bstop\b", _levels_named("stop"), "the stop line"),
]


@rule("named_not_drawn", "#113 rule 3: a picture must DRAW every thing it names (triangle = both lines with price "
                         "inside; zone = the box; trendline = the line with numbered touches; Fibonacci = its "
                         "lines with numbers; ADR = its band)")
def named_not_drawn(drawn: Drawn) -> list[Problem]:
    out = []
    for t in _words(drawn):
        for word, pattern, found, what in NEEDS:
            if re.search(pattern, t.text, flags=re.IGNORECASE) and not found(_items_for(drawn, t)):
                where = "this chart" if t.panel is not None else "the picture"
                out.append(Problem("named_not_drawn", f"'{_short(t.text)}' names the {word}, but {where} does not "
                                                      f"draw {what}"))
    for t in drawn.texts:                         # a name written for a thing that is off the chart
        if t.names is not None and not drawn.items[t.names].visible:
            out.append(Problem("named_not_drawn", f"'{_short(t.text)}' is written, but its "
                                                  f"{drawn.items[t.names].kind.lower()} is not on the chart"))
    return out


BANNED = ["leftover", "arm", "arms", "armed", "verdict", "veto", "causal", "placebo", "mirror", "baseline", "basket"]


@rule("banned_words", "#113 rule 2 (CLAUDE.md): code words he never used - leftover, arm, verdict, veto, causal, "
                      "placebo, mirror, baseline, R, basket")
def banned_words(drawn: Drawn) -> list[Problem]:
    out = []
    words = re.compile(r"\b(" + "|".join(BANNED) + r")\b", flags=re.IGNORECASE)
    r_unit = re.compile(r"(?<![A-Za-z])(\d+(\.\d+)?)?R(?![A-Za-z])")
    for t in _words(drawn):
        for m in words.finditer(t.text):
            out.append(Problem("banned_words", f"'{m.group(0)}' in '{_short(t.text)}' - say what it means"))
        if r_unit.search(t.text):
            out.append(Problem("banned_words", f"'R' in '{_short(t.text)}' - say the distance in dollars"))
    return out


TAG = re.compile(r"^[A-Z]{1,2}\d{1,2}( C)?$")
MONEY = re.compile(r"₹|\$|\brupees?\b|\bRs\.?(?=[\s\d]|$)|\bINR\b|\d", flags=re.IGNORECASE)


@rule("his_labels", "#115 1-2 letter labels (B1, S1, H1, 'B1 C') · #121 one keyword each · #116 no money on a "
                    "point, no word 'hedge' on it")
def his_labels(drawn: Drawn) -> list[Problem]:
    out = []
    for t in drawn.texts:
        if t.role != "tag":
            continue
        lines = t.text.split("\n")
        if not TAG.match(lines[0]):
            out.append(Problem("his_labels", f"'{_short(lines[0])}' is not his label - use B1, S1, H1, 'B1 C'"))
        word = " ".join(lines[1:]).strip()
        if len(word.split()) > 2:
            out.append(Problem("his_labels", f"'{_short(word)}' - one keyword under a label, not a sentence"))
        if MONEY.search(word):
            out.append(Problem("his_labels", f"'{_short(word)}' - no money or numbers on a point"))
        if re.search(r"\bhedg", word, flags=re.IGNORECASE):
            out.append(Problem("his_labels", f"'{_short(word)}' - no word 'hedge' on a point"))
    return out


@rule("question_on_picture", "#132 every question is written ON the picture, not in a message beside it")
def question_on_picture(drawn: Drawn) -> list[Problem]:
    if drawn.kind != "question":
        return []
    q = drawn.question
    if q is None or not q.ask.strip():
        return [Problem("question_on_picture", "a question picture with no question on it")]
    out = []
    if not q.ask.rstrip().endswith("?"):
        out.append(Problem("question_on_picture", f"'{_short(q.ask)}' is not a question (no '?')"))
    if not any(t.role == "question" and t.text == q.ask for t in drawn.texts):
        out.append(Problem("question_on_picture", "the question is not written on the picture"))
    if not q.how:
        out.append(Problem("question_on_picture", "no line saying how to answer (YES / NO, or what to say)"))
    return out


MONEY_WORDS = re.compile(r"₹|\brupees?\b|\bRs\.?(?=[\s\d]|$)|\bINR\b|\bprofits?\b|\bloss(es)?\b|\bwon\b|\blost\b",
                         flags=re.IGNORECASE)


@rule("no_money_in_question", "lesson B16: meaning is chosen BEFORE money is seen - no ₹ in a question")
def no_money_in_question(drawn: Drawn) -> list[Problem]:
    if drawn.kind != "question":
        return []
    return [Problem("no_money_in_question", f"'{_short(t.text)}' shows money on a question picture")
            for t in _words(drawn) if MONEY_WORDS.search(t.text)]


@rule("ends_at_entry", "lesson B16: a question picture ENDS at the entry candle - nothing after it is shown")
def ends_at_entry(drawn: Drawn) -> list[Problem]:
    if drawn.kind != "question":
        return []
    if drawn.entry_at is None:
        return [Problem("ends_at_entry", "a question picture must say its entry candle (entry_at)")]
    step = pd.Timedelta(minutes=TF_MINUTES[drawn.entry_tf])
    entry_of = {p.index: (p.entry_at if p.entry_at is not None else drawn.entry_at) for p in drawn.panels}
    out = []
    for p in drawn.panels:                        # a chart with its own entry (several setups, L10b) ends at it
        entry = entry_of[p.index]
        close = entry + step
        p_close = p.last_bar + pd.Timedelta(minutes=TF_MINUTES[p.tf])
        if p_close > close:
            out.append(Problem("ends_at_entry", f"the {TF_WORDS[p.tf]} chart shows a candle closing at {p_close}, "
                                                f"after the entry candle closed ({close})"))
        if p.role == "main" and p.tf == drawn.entry_tf and p.last_bar != entry:
            out.append(Problem("ends_at_entry", f"the {TF_WORDS[p.tf]} chart ends at {p.last_bar}, not at the "
                                                f"entry candle {entry}"))
    for i in drawn.items:
        close = entry_of.get(i.panel, drawn.entry_at) + step
        late = [t for t in i.info.get("times", []) if t is not None and t >= close]
        if late:
            out.append(Problem("ends_at_entry", f"the {i.kind.lower()} '{_short(i.name)}' reaches {max(late)}, "
                                                f"after the entry candle"))
    return out


def _shown(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


@rule("never_twice", "#117 never send the same picture twice (and #106: 'not the same that you already sent')")
def never_twice(drawn: Drawn, *, shown: Path | None = None, again_because: str | None = None) -> list[Problem]:
    path = config.SHOWN_PICTURES if shown is None else Path(shown)
    out = []
    for row in _shown(path):
        if row.get("pixel_sha256") == drawn.pixel_sha256:
            out.append(Problem("never_twice", f"this exact picture was already shown on {row.get('date')} "
                                              f"({row.get('file')})"))
        elif row.get("subject") == drawn.subject and not again_because:
            out.append(Problem("never_twice", f"'{drawn.subject}' was already shown on {row.get('date')} - draw "
                                              "another one, or say why again (again_because=...)"))
    return out


def mark_shown(drawn: Drawn, why: str, *, shown: Path | None = None) -> None:
    """Add the picture to the list of pictures he has seen - ONLY once it is really in front of him (#117).
    Append-only (lesson B10: never rewrite a file of his)."""
    path = config.SHOWN_PICTURES if shown is None else Path(shown)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = drawn.path.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        rel = str(drawn.path)
    row = {"date": dt.datetime.now().isoformat(timespec="seconds"), "subject": drawn.subject, "kind": drawn.kind,
           "file": rel, "pixel_sha256": drawn.pixel_sha256, "why": why}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@rule("real_file_type", "#106 the pictures stuck loading on Telegram: JPEGs were sent labelled PNG - a .jpg must "
                        "be a real JPEG")
def real_file_type(drawn: Drawn) -> list[Problem]:
    want = {".jpg": ("JPEG", b"\xff\xd8\xff"), ".jpeg": ("JPEG", b"\xff\xd8\xff"),
            ".png": ("PNG", b"\x89PNG\r\n\x1a\n")}.get(drawn.path.suffix.lower())
    if want is None:
        return [Problem("real_file_type", f"{drawn.path.name}: a picture is .jpg or .png")]
    if not drawn.path.is_file():
        return [Problem("real_file_type", f"{drawn.path.name} is not on disk")]
    head = drawn.path.read_bytes()[:8]
    try:
        with Image.open(drawn.path) as im:
            fmt = im.format
    except Exception:                             # noqa: BLE001 - any unreadable file is the finding
        fmt = None
    if not head.startswith(want[1]) or fmt != want[0]:
        return [Problem("real_file_type", f"{drawn.path.name} is really {fmt or 'not a picture'}, "
                                          f"not {want[0]}")]
    return []


@rule("title_timeframe_clock", "#110 'which timeframe chart is this?' - the timeframe named · lesson B24 the clock "
                               "named (India time / server time / UTC)")
def title_timeframe_clock(drawn: Drawn) -> list[Problem]:
    out = []
    for p in drawn.panels:
        if TF_WORDS[p.tf] not in p.title:
            out.append(Problem("title_timeframe_clock", f"chart {p.index + 1} ('{_short(p.title)}') does not say it "
                                                        f"is a {TF_WORDS[p.tf]} chart"))
        if CLOCK_WORDS[p.clock] not in p.title:
            out.append(Problem("title_timeframe_clock", f"chart {p.index + 1} ('{_short(p.title)}') does not say "
                                                        f"its clock ({CLOCK_WORDS[p.clock]})"))
        if p.bar_minutes is not None and TF_MINUTES[p.tf] < 1440 and abs(p.bar_minutes - TF_MINUTES[p.tf]) > 0.5:
            out.append(Problem("title_timeframe_clock", f"chart {p.index + 1} says {TF_WORDS[p.tf]}, but its candles "
                                                        f"are {p.bar_minutes:g} minutes apart"))
    return out


@rule("every_point_own_panel", "#117 every entry and exit has its own 5-minute panel, with the hour chart above")
def every_point_own_panel(drawn: Drawn) -> list[Problem]:
    if drawn.kind != "whole_trade":
        return []
    if not drawn.panels or drawn.panels[0].tf != "1h":
        return [Problem("every_point_own_panel", "the whole trade is not on a 1-hour chart at the top")]
    top = drawn.panels[0]
    out = []
    points = [i for i in drawn.items if i.panel == 0 and i.kind == "Marker"]
    for m in points:
        ok = any(i.kind == "Marker" and i.info.get("tag") == m.info.get("tag")
                 and i.info.get("time") == m.info.get("time") and i.visible
                 and drawn.panels[i.panel].tf == "5m" and drawn.panels[i.panel].role == "point"
                 and drawn.panels[i.panel].area.y1 <= top.area.y0
                 for i in drawn.items if i.panel != 0)
        if not ok:
            out.append(Problem("every_point_own_panel", f"{m.info.get('tag')} at {m.info.get('time')} has no "
                                                        "5-minute panel of its own under the hour chart"))
    return out


@rule("parse_math_off", "lesson B8: matplotlib reads '$...$' as maths - text.parse_math must be off")
def parse_math_off(drawn: Drawn) -> list[Problem]:
    return [Problem("parse_math_off", "text.parse_math was on: '$' would be read as maths")] if drawn.parse_math \
        else []


if __name__ == "__main__":
    for name, (_fn, his) in RULES.items():
        print(f"{name:22s} {his}")
