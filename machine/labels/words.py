"""The word his eyes judge that the machine learns to see (L15.1). ONE word for now: "strong zone".

Why this word (plan/work/L15.1-notes.md has the long form):
  - The prompt's own example is "strong zone", and plan/README.md section 4 gives strategy.md section 3 (strong zones)
    to L15.1-.2. T2 later moved "strong zone" to Box 1 (his #77-#81 define it: plan/work/four-boxes.md) - but none of
    the six Box 2 words (steep, big move, very strong, good setup, halfway, positive / false news) has one drawn answer
    of his in strategy/pictures/ (for "steep" he only ever said no, on pictures that are not there - #118, #127), so
    none of them can give the "first labels from his drawings".
  - Of his drawn answers, three words have yes AND no marks: strong zone, equal lows (Q88, Q89), trendline touches
    (Q92, Q94). Equal lows and trendline touches are decided by cents (#89: 2 cents yes, 9 cents not good; #94: 0.0035
    of a day's range yes, 0.007 no) - a picture cannot show that, the code already measures it. His zone is what a
    picture shows: "the candles were very small ... they simply stay there" (#79).
  - What "yes" means comes from his cards: the Q79 card asked "Which of these is YOUR kind of zone?" and he answered
    "B and d are the my kind of zones" (#80); the quiet one is his main zone (#81). A classic support / resistance is
    "also strong zone ... but I won't [trade] that" (#78) - so it is a NO here: the question is his KIND of zone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    slug: str                     # file names: data/labels/<slug>.parquet, plan/work/labels/<slug>/
    says: str                     # the word, his way
    ask: str                      # the question ON the picture (#132)
    how: tuple[str, ...]          # how to answer, on the picture
    heading: str
    sub: str                      # one plain sentence under the heading
    band_name: str                # the name drawn next to the band
    yes_means: str                # plain words, with his lessons
    lessons: tuple[str, ...]


STRONG_ZONE = Word(
    slug="strong_zone",
    says="strong zone",
    ask="Is the brown band YOUR kind of zone - the quiet one you trade?",
    how=("Reply to this picture: yes / no / not sure",),
    heading="YOUR KIND OF ZONE?",
    sub="The brown band = where price sat. The chart ends when the band is finished - nothing after it.",
    band_name="the band",
    yes_means=("his MAIN zone: small candles parked in a tight range, that nobody else sees (#79, #80, #81). A "
               "classic support / resistance touched many times is his second choice - that is a NO here (#78, #79)."),
    lessons=("LEARNED #77", "LEARNED #78", "LEARNED #79", "LEARNED #80", "LEARNED #81"),
)

WORDS: dict[str, Word] = {STRONG_ZONE.slug: STRONG_ZONE}
LABELS = ("yes", "no", "not sure")


def get(slug: str) -> Word:
    if slug not in WORDS:
        raise KeyError(f"no word {slug!r} to label; the words: {', '.join(WORDS)}")
    return WORDS[slug]
