"""Which words of his sentence are searched in his documents before the talker reads it (L8, his #129 rule).

    from machine.talker.terms import find
    find("london high taken and it give bos and fvg in 5 min then sell at fvg, target asian low")
    # -> [Term("session high / low", "london high", query="london high taken and it give bos ..."), ...]

Every trading word he uses, and every vague word (Box 2 of plan/work/four-boxes.md: steep, big move, very strong,
good, halfway, positive / false news; and the Box 3 kind: near, few pips, eventually, evening ...), is a TERM. A term
found in his words is searched with the words around it (his own context), so the search finds what HE means by it
here. The list is Claude's (from plan/likhith.md section A, strategy.md sections 1-13 and the four boxes), not his.
At most MAX_TERMS per sentence (the vague words first: they decide the questions), in the order he said them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_TERMS = 10          # Claude's number: 10 terms x 5 quotes keeps one sentence's search under ~4,000 tokens
WINDOW = 6              # words of his sentence kept on each side of the term in its search

# (name, pattern, vague). A vague word is one only he can define (Box 2) or a number the machine tries (Box 3).
TERMS: tuple[tuple[str, str, bool], ...] = (
    # ---- his vague words -----------------------------------------------------------------------------------------
    ("steep", r"\bsteep\w*", True),
    ("big move", r"\b(big|strong|huge)\s+(move|movement|impulse)\b", True),
    ("very strong / strong trend", r"\bvery strong\b|\bstrong\s+(trend|uptrend|downtrend|bullish|bearish)\b", True),
    ("good setup", r"\b(setup|set up) (is|was) good\b|\bgood (setup|entry)\b", True),
    ("halfway", r"\bhalf\s?way\b|\bhalf of the\b", True),
    ("positive / false news", r"\b(positive|false|negative) news\b", True),
    ("few pips", r"\bfew pips\b|\b(minimal|little|small) pips\b", True),
    ("near", r"\bnear\b|\bclose to\b", True),
    ("eventually", r"\beventually\b|\bwhenever\b|\bsooner or later\b", True),
    ("friday / weekend", r"\bfriday\b|\bweekend\b|\bevening\b", True),
    ("too far / too big", r"\btoo (far|big|early|late)\b", True),
    # ---- his trading words ---------------------------------------------------------------------------------------
    ("golden ratio / fibonacci", r"\bgolden ratio\b|\bfib\w*|\b0\.\d{1,3}\b", False),
    ("one-sided move", r"\bone (up )?one side\b|\bone side\w* move\w*|\bone.sided\b", False),
    ("higher high / lower low", r"\bhigher highs?\b|\blower lows?\b", False),
    ("break of structure", r"\bbreak of structure\b|\bbos\b|\bbreaks? (the )?structure\b", False),
    ("order block", r"\border blocks?\b", False),
    ("first gap", r"\b(1st|first|1et)\s+(fair value gap|fvg|gap)s?\b", False),
    ("second gap", r"\b(2nd|second)\s+(fair value gap|fvg|gap)s?\b|\bthe second\b", False),
    ("fair value gap", r"\bfair value gaps?\b|\bfvgs?\b", False),
    ("where in the gap", r"\bbeginning of (the )?fvg\b|\bdeep of (the )?fvg\b|\b(near|far) edge\b", False),
    ("liquidity taken / sweep", r"\bsweep\w*|\b(took|taken|take out|takes out|grab\w*)\b", False),
    ("liquidity", r"\bliquidity\b", False),
    ("obvious liquidity", r"\bobvious\b", False),
    ("equal highs / lows", r"\bequal (highs?|lows?)\b", False),
    ("session high / low", r"\b(london|asian?|new york|ny|session)\s+(high|low)s?\b", False),
    ("swing high / low", r"\bswing (high|low)s?\b|\bswings?\b", False),
    ("trendline", r"\btrend\s?lines?\b", False),
    ("third touch", r"\b(3rd|third|4th|fourth)\s+touch\b|\btouches\b", False),
    ("triangle", r"\btriangles?\b", False),
    ("wedge", r"\bwedges?\b", False),
    ("false break", r"\bfalse(ly)?\s+break\w*", False),
    ("strong zone", r"\b(strong )?zones?\b", False),
    ("retest", r"\bretest\w*", False),
    ("all-time high", r"\ball\s?time high\b|\bath\b", False),
    ("adr green / red zone", r"\badr\b|\bgreen zone\b|\bred zone\b|\bdiscount\b|\bpremium\b", False),
    ("higher timeframe direction", r"\bhigher time\s?frames?\b|\bhtf\b|\bconfluence\b", False),
    ("week / day / hour direction", r"\b(weekly|daily|hourly|week|day|hour)\s+(is\s+|are\s+)?(up|down|bullish|bearish)\b"
                                    r"|\b(bullish|bearish|uptrend|downtrend)\b", False),
    ("news", r"\bnews\b", False),
    ("hedge", r"\bhedg\w*|\bhedge point\b", False),
    ("stop", r"\bstop( loss)?\b", False),
    ("risk to reward", r"\b1\s*is to\s*\d\b|\b1\s*:\s*\d+\b|\brisk to reward\b", False),
    ("target", r"\btargets?\b", False),
    ("trail", r"\btrail\w*|\btrial\b", False),
    ("skip", r"^\s*s\b|\bskip\w*|\bavoid\w*", False),
    ("come back / first visit", r"\bcome\w* back\b|\bback to\b|\bdrops? to\b", False),
    ("doji / hammer", r"\bdoji\b|\bhammer\b", False),
    ("close the trade", r"\bclose (the )?(trade|position)\b|\bdon'?t hold\b", False),
)
_COMPILED = tuple((name, re.compile(p, re.IGNORECASE), vague) for name, p, vague in TERMS)


@dataclass(frozen=True)
class Term:
    name: str           # the term, as the talker and the #129 sentence name it
    said: str           # his words that matched
    query: str          # what is searched: his words around it
    vague: bool
    at: int             # where in his words (for the order)


def find(text: str, max_terms: int = MAX_TERMS) -> list[Term]:
    """The terms of his sentence, each once (its first place), vague words first, then in his order."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    found: dict[str, Term] = {}
    for name, rx, vague in _COMPILED:
        m = rx.search(text)
        if not m or name in found:
            continue
        i = next((k for k, (a, b) in enumerate(words) if b > m.start()), 0)
        j = next((k for k, (a, b) in enumerate(words) if b >= m.end()), len(words) - 1)
        lo, hi = max(0, i - WINDOW), min(len(words), j + WINDOW + 1)
        query = " ".join(text[a:b] for a, b in words[lo:hi]).strip(" ,.;…")
        found[name] = Term(name, m.group(0).strip(), query, vague, m.start())
    chosen = sorted(found.values(), key=lambda t: (not t.vague, t.at))[:max_terms]
    return sorted(chosen, key=lambda t: t.at)
