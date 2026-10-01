"""His drawn answers (strategy/pictures/, 43 files) and the labels they give (L15.1).

PICTURES = every file in strategy/pictures/, read with its LEARNED entry: what the card asked, what he marked, and which
word it could label. A test checks the list against the folder (a new picture must be read and added here).

LABELS = the first labels of the word "strong zone" (machine/labels/words.py), one per band he judged on one chart:
  Q78 A yes, Q78 B no   the same two bands as Q76 panel D (#78: "That is the one I will consider most ..." / "I will not
                        consider that as a um, strong zone"), redrawn on the 1-hour chart and asked again (#79)
  Q79 B yes, Q79 D yes  "B and d are the my kind of zones" (#80) - so A and C (the card asked "Which of these is YOUR
  Q79 A no,  Q79 C no   kind of zone?") are not; A (2022) and C (2021) are the hidden years: kept, never drawn
Not labels here, and why (each is in PICTURES): Q76 A ("the zone that you have drawn are correct" - which of its 4 bands,
and his kind or not, he did not say; his kind was first asked on Q78 / Q79), Q76 B and C (hidden years; B "a lot of
zones ... it's not like that"), Q86 (take / skip of the zone ENTRY - he also trades his second-choice zones, #81), Q97 and
Q98 (the BREAK of a zone and its retest). His words are quoted word for word (a test checks each piece in LEARNED.md).

Where each chart's candles and band come from: the card's own bar numbers and bands in old/build/concept_chart.py
(C1B, C1C) and the old engine's bar times (old/ml/agent ... fastbt bars.npz, UTC) - READ ONCE while building this
file (L15.1, 30 Sep 2026), typed here, never imported. The Q79 bands are the old engine's zones, rebuilt exactly by
machine/blocks/zones.py on the old ruler and price step (B 1293.78-1294.79 calm 0.54, D 1254.78-1257.10 calm 0.69 - the
calms the card was built on). The Q78 bands are the card's typed ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from machine import config
from machine.data import clock
from machine.labels import words as W

HIDDEN_FROM = pd.Timestamp(f"{config.CHECK_YEARS[0]}-01-01")      # 2021: the check years, hidden (plan/likhith.md A)


@dataclass(frozen=True)
class Drawing:
    """One file in strategy/pictures/."""
    file: str
    card: str             # the question card it belongs to ("" = not a question)
    lessons: str          # the LEARNED entries that answer it
    asks: str             # what the card asked, in plain words
    his: str              # what he marked or said
    word: str             # the word it could label ("" = none)


PICTURES: tuple[Drawing, ...] = (
    Drawing("L106-telegram-not-loading.png", "", "#106", "his phone: the charts stuck loading on Telegram",
            "not a chart answer", ""),
    Drawing("L107-telegram-still-not-loading.png", "", "#107", "his phone: still not loading",
            "not a chart answer", ""),
    Drawing("Q56-order-block.png", "Q56-order-block", "#57", "which candle is the order block", "see his drawing", ""),
    Drawing("Q56-his-answer.png", "Q56-order-block", "#57", "his drawing on Q56",
            "arrows at the RED candle: A 23:25, B C D 23:45 India time", "order block"),
    Drawing("Q57-A1-order-block-red.png", "Q57-A1-order-block-red", "#58", "which RED candle is the order block "
            "(4-5 Oct 2016)", "see his drawing", "order block"),
    Drawing("Q57-A1-his-answer.png", "Q57-A1-order-block-red", "#58", "his drawing on Q57",
            "all four panels: the 23:25 red candle", "order block"),
    Drawing("Q58-A1b-order-block-4charts.png", "Q58-A1b-order-block-4charts", "#59", "his order-block rule on 4 charts",
            "see his drawing", "order block"),
    Drawing("Q58-A1b-his-answer.png", "Q58-A1b-order-block-4charts", "#59", "his drawing on Q58",
            "A 5 May 2022 18:10 · B 3 Jun 2020 17:10 · C 8 Oct 2020 22:40 · D 27 Mar 2018 12:25", "order block"),
    Drawing("Q59-A1c-order-block-corrected.png", "Q59-A1c-order-block-corrected", "#60",
            "order block, corrected rule, 4 new charts", "see his drawing", "order block"),
    Drawing("Q59-A1c-his-answer.png", "Q59-A1c-order-block-corrected", "#60", "his drawing on Q59",
            "A 9 Jan 2023 20:25 · B 2 Feb 2017 18:55 · C 27 Jul 2022 10:05 · D 15 Nov 2018 23:15", "order block"),
    Drawing("Q60-A5-which-gap.png", "Q60-A5-which-gap", "#61", "which gap after the break is the entry",
            "see his drawing", "which gap"),
    Drawing("Q60-A5-his-answer.png", "Q60-A5-which-gap", "#61", "his drawing on Q60", "gap 1 on all four (the FIRST gap)",
            "which gap"),
    Drawing("Q61-A5b-first-gap.png", "Q61-A5b-first-gap", "#62, #63", "the first gap: before the break / tiny / late",
            "A gap 1 · B gap 1 (tiny) · C NO TRADE (the move had gone too far) · D gap 1", "which gap"),
    Drawing("Q61-A5b-his-answer.png", "Q61-A5b-first-gap", "#62", "his drawing on Q61", "as the card's row above",
            "which gap"),
    Drawing("Q63-breaks-before-gap.png", "Q63-breaks-before-gap", "#64", "how many breaks before the gap (take / skip)",
            "A skip · B take · C take · D skip (\"too far to hedge\")", "take or skip"),
    Drawing("Q64-hedge-distance.png", "Q64-hedge-distance", "#65, #66", "too far to hedge? (take / skip)",
            "all four TAKE; the entry at the gap's NEAR edge", "take or skip"),
    Drawing("Q64-his-answer.png", "Q64-hedge-distance", "#65", "his drawing on Q64",
            "arrow tips on the gap edge price meets first", "entry price"),
    Drawing("Q66-skip-vs-take.png", "Q66-skip-vs-take", "#67", "why skip one and take the other",
            "he takes both once entry, hedge and target were drawn", "take or skip"),
    Drawing("Q67-A7-return-delay.png", "Q67-A7-return-delay", "#68", "still enter when price comes back this late?",
            "A yes · B yes · C no · D no (only the FIRST visit back)", "first visit"),
    Drawing("Q67-his-answer.png", "Q67-A7-return-delay", "#68", "his drawing on Q67",
            "arrows at the earlier visits on C and D", "first visit"),
    Drawing("Q68-D5-trendlines.png", "Q68-D5-trendlines", "#69", "a 2-touch line between entry and target",
            "A 50-50 · B take · C take · D take", "take or skip"),
    Drawing("Q68-explain-first-visit.png", "", "", "Claude's picture explaining #68 to him (not a question)",
            "not answered", ""),
    Drawing("Q69-C3-daily-bias.png", "Q69-C3-daily-bias", "#70", "the daily bias: up / down / neutral",
            "A neutral · B neutral · C 75% up · D up", "daily bias"),
    Drawing("Q69-his-answer.png", "Q69-C3-daily-bias", "#70", "his drawing on Q69",
            "a falling trendline on D, broken and retested", "daily bias"),
    Drawing("Q70-C3b-daily-bias.png", "Q70-C3b-daily-bias", "#71, #72, #73", "the daily bias, second set",
            "A up · B up · C neutral · D up", "daily bias"),
    Drawing("Q73-C3c-daily-trade.png", "Q73-C3c-daily-trade", "#74", "the machine's daily reading drawn: right / wrong",
            "all four right", "daily bias"),
    Drawing("Q74-C2-premium-discount.png", "Q74-C2-premium-discount", "#75, #76", "premium / discount (take / skip)",
            "he could not read the first draw; then all four are proper setups if the hour and day agree",
            "take or skip"),
    Drawing("Q76-C1-strong-zones.png", "Q76-C1-strong-zones", "#77, #78", "are these the zones YOU would draw?",
            "A: the zones are correct (which band not said) · B (2022): too many zones · C (2021): not said · "
            "D: the 22-candle band at 1,230 is the one he trades; the 31-candle band at 1,252-1,257 he does not",
            "strong zone"),
    Drawing("Q78-C1b-which-zone.png", "Q78-C1b-which-zone", "#79", "what makes A a zone you trade and B not?",
            "A: he trades it (recent, a clear range, the candles form one range) · B: a typical support and "
            "resistance - valid, not what he trades", "strong zone"),
    Drawing("Q79-C1c-quiet-bar.png", "Q79-C1c-quiet-bar", "#80, #81", "which of these is YOUR kind of zone?",
            "B yes · D yes · A no · C no (A and C are 2022 and 2021, the hidden years)", "strong zone"),
    Drawing("Q81-C4-bias-dial.png", "Q81-C4-bias-dial", "#82", "with the hour + day: next liquidity or 1st target?",
            "not answered picture by picture (the target trails)", ""),
    Drawing("Q83-B1-session-liquidity.png", "Q83-B1-session-liquidity", "#83", "session high / low as liquidity",
            "all four are liquidity", "liquidity"),
    Drawing("Q85-B1c-session-start.png", "Q85-B1c-session-start", "#85", "the session's start high / low or its swing",
            "the grey line (the session's own high / low) is the one to count", "liquidity"),
    Drawing("Q86-D6-zone-entry.png", "Q86-D6-zone-entry", "#86, #87", "the zone entry with no gap: take / skip",
            "A take · B skip (swing high is not liquidity; an unfilled gap on the hedge side) · C take · D take",
            "zone entry"),
    Drawing("Q88-B2-equal-lows.png", "Q88-B2-equal-lows", "#88", "equal lows: yes / no",
            "A no · B no · C yes · D no", "equal lows"),
    Drawing("Q89-B2b-equal-lows-numbers.png", "Q89-B2b-equal-lows-numbers", "#89", "equal lows: how close",
            "A no · B no · C yes (\"the perfect one\") · D no", "equal lows"),
    Drawing("Q91-B4-recent-swing.png", "Q91-B4-recent-swing", "#91", "a recent 5-minute swing as liquidity",
            "no for entries (a target only)", "liquidity"),
    Drawing("Q92-B3-trendline-touches.png", "Q92-B3-trendline-touches", "#92", "trendline liquidity: how many touches",
            "A no · B yes · C no · D no (three touches, none through the line)", "trendline"),
    Drawing("Q93-B6-hour-swing.png", "Q93-B6-hour-swing", "#93", "a 1-hour swing alone as liquidity",
            "all four: a target, not an entry", "liquidity"),
    Drawing("Q94-B3b-touch-through.png", "Q94-B3b-touch-through", "#94", "how far a touch may go through the line",
            "all four no", "trendline"),
    Drawing("Q95-B5-proper-sweep.png", "Q95-B5-proper-sweep", "#95, #96", "a proper grab or an accidental one",
            "all four are proper grabs", "sweep"),
    Drawing("Q97-D7-zone-break.png", "Q97-D7-zone-break", "#97", "the break of a strong zone",
            "never at the break: close through, move away, enter on the retest (he named B's retest)", "zone break"),
    Drawing("Q98-D8-break-distance.png", "Q98-D8-break-distance", "#98, #99", "how far the break must go",
            "all four valid - no distance rule", "zone break"),
)


@dataclass(frozen=True)
class DrawnLabel:
    """One band he judged on one chart: a label of the word."""
    key: str                          # "Q79-B"
    word: str                         # machine/labels/words.py slug
    label: str                        # yes / no / not sure
    picture: str                      # the file in strategy/pictures/
    panel: str                        # the panel letter on that picture
    lessons: tuple[str, ...]
    his_words: str                    # word for word; several pieces joined by " … " (each piece is in LEARNED.md)
    first_utc: str                    # the first 5-minute candle he saw (UTC, the card's own bar)
    last_utc: str                     # the last one
    lo: float                         # the band he saw
    hi: float | None                  # None = not read (the hidden years)
    same_as: str = ""                 # another picture where he judged the same band
    note: str = ""

    @property
    def hidden(self) -> bool:
        """The chart is in the check years (2021-2023): kept as his words, never drawn (like L6b.1-L6b.3)."""
        return pd.Timestamp(self.first_utc) >= HIDDEN_FROM or pd.Timestamp(self.last_utc) >= HIDDEN_FROM

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """His first and last 5-minute candle on MT5's server clock (the card grouped exactly these into hours, so the
        picture does the same: machine/labels/picture.hour_candles)."""
        return (clock.utc_to_server(pd.Timestamp(self.first_utc, tz="UTC")),
                clock.utc_to_server(pd.Timestamp(self.last_utc, tz="UTC")))


_Q78 = dict(picture="Q78-C1b-which-zone.png", first_utc="2016-02-26 14:15", last_utc="2016-03-29 23:15",
            same_as="Q76-C1-strong-zones.png panel D (the same bands, 4-hour candles)")

LABELS: tuple[DrawnLabel, ...] = (
    DrawnLabel("Q78-A", "strong_zone", "yes", panel="A", lessons=("LEARNED #78", "LEARNED #79"), lo=1230.61, hi=1233.74,
               his_words="That is the one I will consider most and I will only trade at those. … option A I will "
                         "trade because it, it is a recent one and it will give you a clear range",
               note="he added: 'even option A is also like not my typical uh, zone' - an example, not his best (#79)",
               **_Q78),
    DrawnLabel("Q78-B", "strong_zone", "no", panel="B", lessons=("LEARNED #78", "LEARNED #79"), lo=1251.91, hi=1257.00,
               his_words="But I won't read that. I will not consider that as a um, strong zone. … It's a typical "
                         "support and resistance",
               note="'won't read' = won't trade (his phone's voice typing); valid, his second choice (#79, #81)",
               **_Q78),
    DrawnLabel("Q79-B", "strong_zone", "yes", "Q79-C1c-quiet-bar.png", "B", ("LEARNED #80",),
               "B and d are the my kind of zones.", "2017-12-26 08:10", "2018-01-03 21:50", 1293.78, 1294.79),
    DrawnLabel("Q79-D", "strong_zone", "yes", "Q79-C1c-quiet-bar.png", "D", ("LEARNED #80",),
               "B and d are the my kind of zones.", "2017-05-01 15:05", "2017-05-08 20:50", 1254.78, 1257.10),
    DrawnLabel("Q79-A", "strong_zone", "no", "Q79-C1c-quiet-bar.png", "A", ("LEARNED #80",),
               "B and d are the my kind of zones.", "2021-12-29 13:10", "2022-01-05 21:50", 1813.56, None,
               note="asked 'which is YOUR kind'; he named B and D only. 2022 = the hidden years: not drawn"),
    DrawnLabel("Q79-C", "strong_zone", "no", "Q79-C1c-quiet-bar.png", "C", ("LEARNED #80",),
               "B and d are the my kind of zones.", "2021-02-26 16:05", "2021-03-05 21:50", 1694.48, None,
               note="asked 'which is YOUR kind'; he named B and D only. 2021 = the hidden years: not drawn"),
)


def labels(word: str) -> list[DrawnLabel]:
    return [x for x in LABELS if x.word == word]


def page() -> str:
    """plan/work/labels/his-drawings.md: every picture of his, read with its lesson, and the labels taken from them."""
    words: dict[str, int] = {}
    for d in PICTURES:
        if d.word:
            words[d.word] = words.get(d.word, 0) + 1
    L = ["# His drawn answers - read one by one (L15.1)", "",
         f"> Made by `python -m machine.labels catalogue` from `machine/labels/drawings.py`. All {len(PICTURES)} files "
         "in `strategy/pictures/`, each read with its LEARNED entry: what the card asked and what he marked. A test "
         "checks this list against the folder.", "",
         "## The labels of \"strong zone\" taken from them", "",
         f"The question on every label chart: *{W.STRONG_ZONE.ask}*", "",
         "| label | answer | his words (word for word) | lesson | candles he saw (UTC) | band | drawn? |",
         "|---|---|---|---|---|---|---|"]
    for x in LABELS:
        band = f"{x.lo:.2f}-{x.hi:.2f}" if x.hi is not None else f"from {x.lo:.2f} (top not read)"
        L.append(f"| {x.key} | **{x.label}** | {x.his_words} | {', '.join(x.lessons)} | {x.first_utc} to "
                 f"{x.last_utc} | {band} | " + ("no - the hidden years (2021-2023)" if x.hidden else "yes") + " |")
    L += ["", "Q78 A and B are the same two bands as Q76 panel D, where he first said it (#78); Q78 asked again on the "
          "1-hour chart (#79). Not labels, and why: Q76 A (\"the zone that you have drawn are correct\" - which of "
          "its 4 bands, and his KIND or not, he did not say), Q76 B and C (the hidden years), Q86 (take / skip of the "
          "zone ENTRY - he also trades his second-choice zones, #81), Q97 and Q98 (the BREAK of a zone).", "",
          "## Every picture", "",
          "| file | card | lesson | the card asked | what he marked | the word it could label |",
          "|---|---|---|---|---|---|"]
    for d in PICTURES:
        L.append(f"| `{d.file}` | {d.card or '-'} | {d.lessons or '-'} | {d.asks} | {d.his} | {d.word or '-'} |")
    L += ["", "Pictures per word: " + ", ".join(
        f"{w} ({n} picture{'' if n == 1 else 's'})" for w, n in sorted(words.items(), key=lambda kv: -kv[1])) + ".",
          ""]
    return "\n".join(L)
