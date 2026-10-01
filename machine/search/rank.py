"""search(q) and answered(q): the #129 rule as code (L8b).

His rule, LEARNED #129: search his documents before asking him anything. CLAUDE.md turns it into one sentence that
must start any question whose answer is not there:
    "Searched strategy.md, LEARNED.md, strategy-understanding.md, rules, pictures for X — closest line: …"

How a search runs:
1. Two rankings of every piece: by words (BM25) and by meaning (embeddings, when on). They are joined by reciprocal
   rank fusion (1 / (60 + place) summed - the usual 60), so neither scale has to be trusted on its own.
2. Pieces of one place (one LEARNED entry, one § ...) count as one hit: its best piece.
3. The hits that are about the same point - the most relevant hit that passes the FOUND lines, plus the others of
   the best BAND_N that pass and are no further than `band` below it in meaning - are put in the ORDER OF TRUST
   (CLAUDE.md): his words first, then strategy-understanding, rules, strategy.md, the rest; inside one source, the
   more relevant first. Hits further down keep their place.
   "Newest first" is NOT a sort key: the band cannot tell "the same point" from "a nearby point", and on the exam's
   first run it put LEARNED #72 (all-time high) above #21 (6 Sep) (equal highs) for an equal-highs question. The newer
   lesson is still cited when it is in the band; which one wins on a real conflict is the reader's call (the dates
   are on every hit).
4. answered(q): FOUND when at least one of those hits passes the lines - it cites up to config.SEARCH_CITE_MAX of
   them, most trusted first. Otherwise NOT FOUND, with the sentence above and the closest line.

The FOUND lines are Claude's numbers, chosen on tests/evals/search-dev.jsonl (30 questions, never the exam), and
they belong to one embedding model: a model with another scale gets the words-only lines until it is tuned.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from machine import config
from machine.search import index as ix

RRF_K = 60            # reciprocal rank fusion's usual constant
CANDIDATES = 80       # pieces kept from each ranking before joining
BAND_N = 5            # the trust order re-orders only inside the best 5 hits

# The FOUND lines, per way of searching. meaning = cosine of the best piece; covered = share of the question's words
# (weighted by how rare each word is in his documents) found in that piece; words = its BM25 score.
# The trust-order band ("the same point"): band = how far below the most relevant passing hit's meaning another hit
# may be; band_words (words only) = the share of its BM25 score another hit must reach.
# Chosen on tests/evals/search-dev.jsonl (30 questions) by Claude, 28 Sep 2026 - see plan/work/L8b-notes.md.
LINES = {
    "local|BAAI/bge-small-en-v1.5": {"meaning": 0.68, "covered": 0.45, "band": 0.01},
    "words": {"words": 5.0, "covered": 0.55, "band_words": 1.0},
}


def _lines_for(index: ix.Index) -> tuple[str, dict]:
    if index.embedder is not None:
        for key, lines in LINES.items():
            if key != "words" and index.embedder.recipe.startswith(key):
                return key, lines
    return "words", LINES["words"]


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    tier: int
    date: str
    path: str
    line: int
    quote: str
    words: float            # BM25 score of its best piece
    meaning: float          # cosine of its best piece (nan when embeddings are off)
    covered: float          # share of the question's words (rarity-weighted) in its best piece
    rank: int               # 1 = most relevant, before the trust order
    passes: bool            # passes the FOUND lines on its own
    see: tuple[str, ...] = ()

    def cite(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        extra = (f" ({'named in' if self.source == 'picture' else 'was'} {', '.join(self.see)})" if self.see
                 else "")
        return f'{self.id}{extra} [{where}]: "{self.quote}"'


@dataclass(frozen=True)
class Answer:
    question: str
    found: bool
    cites: tuple[Hit, ...]
    closest: Hit | None
    sentence: str
    how: str
    hits: tuple[Hit, ...] = field(default=(), repr=False)

    def __str__(self) -> str:
        if not self.found:
            return self.sentence
        return "FOUND\n" + "\n".join(f"  {h.cite()}" for h in self.cites)


# ------------------------------------------------------------------------------------------------ pieces of a search
def _coverage(qterms: list[str], piece_terms: set[str], index: ix.Index) -> float:
    """Rarity-weighted share of the question's words that are in the piece. A word his documents never use weighs
    the most (the idf of a word seen 0 times), so a question about something he never talked about stays low."""
    distinct = set(qterms)
    if not distinct:
        return 0.0
    unseen = math.log(1 + (index.bm25.n + 0.5) / 0.5)
    weight = {t: index.bm25.idf.get(t, unseen) for t in distinct}
    total = sum(weight.values())
    return sum(w for t, w in weight.items() if t in piece_terms) / total if total else 0.0


def _quote(piece: ix.Piece, qterms: set[str], width: int = 240) -> tuple[int, str]:
    """The sentence of the piece with most of the question's words, and its line."""
    best, best_score = (piece.segs[0][0], piece.segs[0][1]), -1.0
    for no, text in piece.segs:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            score = len(qterms & set(ix.terms(sentence)))
            if score > best_score:
                best, best_score = (no, sentence), score
    no, s = best
    s = re.sub(r"\s+", " ", s.strip().lstrip(">").strip())
    if len(s) > width:
        s = s[: width - 1].rsplit(" ", 1)[0] + "…"
    return no, s


def _passes(lines: dict, key: str, meaning: float, words: float, covered: float) -> bool:
    if key == "words":
        return words >= lines["words"] and covered >= lines["covered"]
    return meaning >= lines["meaning"] and covered >= lines["covered"]


def search(q: str, k: int = 10, index: ix.Index | None = None) -> list[Hit]:
    """The places in his documents that answer q: ranked by relevance, the close ones put in the order of trust."""
    index = index or ix.get()
    qterms = ix.terms(q)
    n = len(index.pieces)
    if n == 0:
        return []
    words = index.bm25.scores(qterms)
    order_w = np.argsort(-words, kind="stable")
    place_w = np.empty(n, dtype=int)
    place_w[order_w] = np.arange(1, n + 1)
    place_w[words <= 0] = n + RRF_K * 100                  # no shared word: last
    if index.vectors is not None:
        meaning = index.vectors @ index.embedder.query(q)
        order_m = np.argsort(-meaning, kind="stable")
        place_m = np.empty(n, dtype=int)
        place_m[order_m] = np.arange(1, n + 1)
        pool = set(order_w[:CANDIDATES].tolist()) | set(order_m[:CANDIDATES].tolist())
        fused = {i: 1 / (RRF_K + place_w[i]) + 1 / (RRF_K + place_m[i]) for i in pool}
    else:
        meaning = np.full(n, np.nan)
        pool = set(order_w[:CANDIDATES].tolist())
        fused = {i: 1 / (RRF_K + place_w[i]) for i in pool}

    best: dict[str, int] = {}                              # one hit per place: its best piece
    for i in sorted(pool, key=lambda i: -fused[i]):
        best.setdefault(index.pieces[i].id, i)
    key, lines = _lines_for(index)
    qset = set(qterms)
    hits: list[Hit] = []
    for rank, (pid, i) in enumerate(sorted(best.items(), key=lambda kv: -fused[kv[1]])[:k], 1):
        p = index.pieces[i]
        covered = _coverage(qterms, set(ix.terms(p.embed_text())), index)
        no, quote = _quote(p, qset)
        m = float(meaning[i])
        hits.append(Hit(p.id, p.source, p.tier, p.date, p.path, no if p.line else 0, quote, float(words[i]), m,
                        covered, rank, _passes(lines, key, m, float(words[i]), covered), p.see))
    return _trust_order(hits, lines)


def _trust_order(hits: list[Hit], lines: dict) -> list[Hit]:
    """The band = the most relevant hit that passes, plus every other passing hit of the best BAND_N that is nearly
    as relevant (meaning at most `band` below it; words-only: BM25 at least `band_words` times its score). Inside the
    band: the order of trust, then relevance. Everything else keeps its relevance order."""
    top = hits[:BAND_N]
    passing = [h for h in top if h.passes]
    if not passing:
        return hits
    anchor = passing[0]

    def same_point(h: Hit) -> bool:
        if h is anchor:
            return True
        if "band_words" in lines:
            return h.words >= lines["band_words"] * anchor.words
        return h.meaning >= anchor.meaning - lines["band"]

    band = [h for h in passing if same_point(h)]
    rest = [h for h in hits if h not in band]
    # A picture holds Claude's description of it; his words are in the LEARNED entry that names it - that goes first.
    band.sort(key=lambda h: (h.tier, h.source == "picture", h.rank))
    return band + rest


# ------------------------------------------------------------------------------------------------ answered
SEARCHED = "strategy.md, LEARNED.md, strategy-understanding.md, rules, pictures"


def answered(q: str, index: ix.Index | None = None) -> Answer:
    """FOUND (with citations, most trusted first) or NOT FOUND (with the #129 sentence and the closest line)."""
    index = index or ix.get()
    hits = search(q, index=index)
    cites = tuple(h for h in hits[:BAND_N] if h.passes)[: config.SEARCH_CITE_MAX]
    closest = min(hits, key=lambda h: h.rank) if hits else None
    if cites:
        first = cites[0]
        sentence = f'Found in {first.id}: "{first.quote}"'
        return Answer(q, True, cites, closest, sentence, index.how, tuple(hits))
    x = q.strip().rstrip("?").strip()
    line = f'{closest.id}: "{closest.quote}"' if closest else "none"
    sentence = f'Searched {SEARCHED} for "{x}" — closest line: {line}'
    return Answer(q, False, (), closest, sentence, index.how, tuple(hits))
