"""The voice-typing glossary (L8): what his phone's speech-to-text writes, and what he meant.

    from machine.talker import glossary
    g = glossary.apply("the farewell gap after the heads point")
    g.text          # "the fair value gap after the hedge point"  - what the talker reads
    g.replaced      # every replacement, with its lesson: the report shows them
    g.hints         # words that MAY be voice typing ("loss" = lows) - shown to the talker, never replaced

machine/talker/glossary.csv holds every "Speech-to-text:" note in strategy/LEARNED.md (#59 ... #103) and "the liquid"
= Likhith (#136). Two kinds of row (the kind is Claude's call; tests/test_talker.py checks every note has a row):
    always   a phrase that is never a normal word ("farewell gap", "heads point"): replaced before anything reads it
    hint     a word that is ALSO a normal word ("loss" = lows, but "stop loss"; "swap" = pass, but the broker's swap;
             "trial" = trail, but the live trial): never replaced - the talker is told "his voice typing sometimes
             writes X for Y" and decides from the sentence, and says so in its answer
His words stay as he typed them in the form (Idea.words, word for word); only the text the talker reads is changed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GLOSSARY = Path(__file__).with_name("glossary.csv")


@dataclass(frozen=True)
class Row:
    heard: str          # what the voice typing wrote (lower case)
    means: str          # what he meant
    apply: str          # "always" | "hint"
    lesson: str         # where the note is: "LEARNED #84"
    note: str = ""


@dataclass(frozen=True)
class Replacement:
    heard: str          # as it was in his words
    means: str
    lesson: str


@dataclass(frozen=True)
class Applied:
    words: str                          # his words as typed
    text: str                           # after the "always" rows
    replaced: tuple[Replacement, ...]
    hints: tuple[Row, ...]              # "hint" rows whose word is in his words

    def lines(self) -> list[str]:
        out = [f'replaced "{r.heard}" -> "{r.means}" ({r.lesson})' for r in self.replaced]
        out += [f'maybe "{h.heard}" = "{h.means}" ({h.lesson}; not replaced: {h.note or "a normal word too"})'
                for h in self.hints]
        return out


@lru_cache(maxsize=4)
def load(path: Path = GLOSSARY) -> tuple[Row, ...]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = tuple(Row(r["heard"].strip().lower(), r["means"].strip(), r["apply"].strip(), r["lesson"].strip(),
                         (r.get("note") or "").strip()) for r in csv.DictReader(f))
    bad = [r.heard for r in rows if r.apply not in ("always", "hint")]
    if bad:
        raise ValueError(f"{path.name}: 'apply' must be always or hint - {bad}")
    return rows


def _pattern(heard: str) -> str:
    """A phrase as a whole-word pattern; spaces match any run of spaces ("the liquid" never matches "the liquidity")."""
    return r"(?<![\w])" + r"\s+".join(re.escape(w) for w in heard.split()) + r"(?![\w])"


def apply(words: str, rows: tuple[Row, ...] | None = None) -> Applied:
    rows = load() if rows is None else rows
    always = sorted((r for r in rows if r.apply == "always"), key=lambda r: -len(r.heard))
    replaced: list[Replacement] = []
    text = words
    if always:
        rx = re.compile("|".join(f"({_pattern(r.heard)})" for r in always), re.IGNORECASE)

        def swap(m: re.Match) -> str:
            row = always[m.lastindex - 1]
            replaced.append(Replacement(m.group(0), row.means, row.lesson))
            return row.means

        text = rx.sub(swap, words)
    hints = tuple(r for r in rows if r.apply == "hint" and re.search(_pattern(r.heard), text, re.IGNORECASE))
    return Applied(words, text, tuple(replaced), hints)
