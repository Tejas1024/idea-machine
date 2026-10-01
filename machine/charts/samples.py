"""The L4 sample pictures, from real 2019 gold (💻):  python -m machine.charts.samples [--mark-shown]

    plan/work/samples/1-trade.jpg              a short trade: a small 1-hour picture on top, the 5-minute chart under
                                               it - the grammar of A01, the first chart he did not reject (#120)
    plan/work/samples/2-question.jpg           a question: written ON the picture (#132); the charts END at the
                                               candle it asks about (lesson B16); no money
    plan/work/samples/3-whole-trade.jpg        a longer trade: THE WHOLE TRADE on the 1-hour chart, then a 5-minute
                                               panel for every entry and every exit (#117, #123)
    plan/work/samples/3-whole-trade-arrows.jpg the same trade in his blue arrows, no candles (#122; #123 "both images")

WHAT IS DRAWN IS REAL: real 2019 prices, and trades made by the tester (machine/tester, L3) with its real fills,
exits and money (Rs 40 a position, all-in). WHAT IS NOT HIS: the rule that made the trades. It exists only so there
is something to draw - Claude's rule, not an idea under test, and not a result:
    a 5-minute gap (three candles: the 1st candle's high below the 3rd's low, or the mirror) at least 50 cents wide;
    a limit order at the gap's NEAR edge (his #65), waiting 12 candles (1 hour);
    stop 1 dollar past the gap's far edge; target 3 times as far from the entry as the stop.
It enters ANY gap, so it takes second gaps he would never enter (Tejas, 28 Sep, on these samples: "it's a second gap,
I would not enter there" - four-boxes B1-13). Never show its trades as examples of his trading.
The pictures are chosen by fixed tests written BEFORE looking (never by result): the first 2019 trade closed within
2 hours of its fill; the first that lasted a day or more (else 8 hours or more).
The question uses his settled equal-highs rule (four-boxes B1-07: 0-4 cents apart, the later high does not pass the
first, nothing trades past the first between them), found by a quick sample search - the real block is L6.2.
Nothing here is asked of him: the question is a sample of how a question will look (the real ones come in the weekly
batch, L9b / T3).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from machine import config
from machine.charts import draw as D
from machine.charts import lint
from machine.data import bars
from machine.data.clock import to_ist
from machine.tester import Signal, known_at_close, run

YEAR = "2019"
IDEA = "L4-picture-sample"
MIN_GAP = 0.50
STOP_PAST = 1.00
TARGET_X = 3.0
WAIT_BARS = 12
RULE_WORDS = ("a sample rule of Claude's (not yours), only to have a trade to draw:\n"
              "{side} at a 5-minute gap · stop 1 dollar past it · target 3 times as far")


def gap_signals(m5: pd.DataFrame) -> list[Signal]:
    """Every 5-minute gap of at least MIN_GAP, as a limit order at its near edge. Known at the 3rd candle's close."""
    h, lo, idx = m5["high"].to_numpy(float), m5["low"].to_numpy(float), m5.index
    out = []
    for i in range(2, len(m5)):
        if lo[i] - h[i - 2] >= MIN_GAP:                       # price jumped UP: buy when it comes back down
            side, near, far = "buy", lo[i], h[i - 2]
        elif lo[i - 2] - h[i] >= MIN_GAP:                     # price jumped DOWN: sell when it comes back up
            side, near, far = "sell", h[i], lo[i - 2]
        else:
            continue
        sign = 1 if side == "buy" else -1
        stop = far - sign * STOP_PAST
        target = near + sign * TARGET_X * abs(near - stop)
        out.append(Signal(id=f"gap-{idx[i]:%Y%m%d-%H%M}", known_at=known_at_close(idx[i], "5m"), direction=side,
                          entry_type="limit", entry_price=round(near, 2), stop=round(stop, 2),
                          targets=(round(target, 2),), expiry_bars=WAIT_BARS, idea_id=IDEA, version_id="v1"))
    return out


def gap_of(sig: Signal, m5: pd.DataFrame) -> D.Box:
    """The gap a signal came from, as the blue box: from its 1st candle to where price came back (the fill)."""
    i3 = int(m5.index.searchsorted(pd.Timestamp(sig.known_at))) - 1
    first = m5.index[i3 - 2]
    near = sig.entry_price
    far = sig.stop + (STOP_PAST if sig.direction == "buy" else -STOP_PAST)
    return D.Box(first, first, min(near, far), max(near, far))


def adr_band(day: pd.Timestamp) -> D.ADRBand:
    """The day's ADR zones from the 10 broker days BEFORE it and its own open (strategy-code.md section 16) - a
    sample calculation; the real ADR block is L6.1."""
    d1 = bars.load_bars("1d", day - pd.Timedelta(days=30), day)
    before = d1[d1.index < day].iloc[-10:]
    rng = (before["high"] - before["low"]).to_numpy(float)
    return D.ADRBand(day, day + pd.Timedelta(days=1), float(d1.loc[day, "open"]), adr5=float(rng[-5:].mean()),
                     adr10=float(rng.mean()))


def equal_highs(m5: pd.DataFrame, apart: float = 0.04, min_bars: int = 6, max_bars: int = 72):
    """The first pair of 5-minute swing highs that are his equal highs (B1-07). A swing high is higher than the 2
    candles on each side and is KNOWN 2 candles later. Returns (first bar, second bar, the candle it is known on)."""
    h = m5["high"].to_numpy(float)
    swings = [i for i in range(2, len(h) - 2) if h[i] > h[i - 2:i].max() and h[i] > h[i + 1:i + 3].max()]
    for a in swings:
        for b in swings:
            if not (a + min_bars <= b <= a + max_bars):
                continue
            if h[a] - apart <= h[b] <= h[a] and h[a + 1:b].max() <= h[a]:
                return a, b, b + 2
    raise LookupError("no equal highs found")


def _day_words(t) -> str:
    return to_ist(pd.Timestamp(t)).strftime("%d %b %Y")


def _held(a, b) -> str:
    hours = (pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 3600
    return f"{hours * 60:.0f} minutes" if hours < 1 else (f"{hours:.0f} hours" if hours < 48 else
                                                          f"{hours / 24:.1f} days")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mark-shown", action="store_true", help="add them to plan/work/pictures-shown.jsonl (#117) - "
                                                               "only when he is really shown them")
    args = ap.parse_args(argv)
    out_dir = config.SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    m5 = bars.load_bars("5m", YEAR, YEAR)
    h1 = bars.load_bars("1h", YEAR, YEAR)
    sigs = gap_signals(m5)
    by_id = {s.id: s for s in sigs}
    res = run(sigs, tf="5m", start=YEAR, end=YEAR,
              note="L4 sample pictures only: Claude's rule to have real trades to draw. Not an idea, not a result.")
    closed = [t for t in res.trades if t.status == "closed"]
    print(f"sample rule, {YEAR}: {len(sigs)} gaps, {len(closed)} trades closed (trial file: "
          f"{res.trial_path.relative_to(config.ROOT).as_posix() if res.trial_path else '-'})")

    def lasted(t) -> float:
        return (pd.Timestamp(t.exit_ts) - pd.Timestamp(t.fill_ts)).total_seconds() / 3600

    short = next(t for t in closed if lasted(t) <= 2)
    long_ = next((t for t in closed if lasted(t) >= 24), None) or next(t for t in closed if lasted(t) >= 8)
    made: list[D.Drawn] = []

    # ---- 1. a short trade
    sig = by_id[short.signal_id]
    gap = gap_of(sig, m5)
    gap = D.Box(gap.start, short.fill_ts, gap.lo, gap.hi)
    day = pd.Timestamp(short.fill_ts).normalize()
    stop = D.Level(short.stop_px, "stop", start=short.fill_ts, color=D.INK)
    target = D.Level(short.target_pxs[0], "target", start=short.fill_ts, color=D.GREEN)
    pic = D.trade_picture(
        short, m5, h1, subject=f"L4 sample · trade · {short.direction} {short.fill_ts:%Y-%m-%d %H:%M} server",
        sub=f"{_day_words(short.fill_ts)} · held {_held(short.fill_ts, short.exit_ts)}\n"
            + RULE_WORDS.format(side=short.direction),
        items_5m=[gap, stop, target], items_1h=[adr_band(day)])
    made.append(D.render(pic, out_dir / "1-trade.jpg"))

    # ---- 2. a question: his equal highs, the charts end at the candle the pair is known on
    a, b, k = equal_highs(m5)
    ta, tb, tk = m5.index[a], m5.index[b], m5.index[k]
    pa, pb = float(m5["high"].iloc[a]), float(m5["high"].iloc[b])
    cents = round((pa - pb) * 100)
    line = D.Line((ta, pa), (tb, pb), touches=(ta, tb), name="equal highs", extend_to=tk)
    q = D.Question("Are 1 and 2 the equal highs your idea #1 means?",
                   ("Answer YES, or NO and say what is different.",))
    pic = D.question_picture(
        m5, h1, entry_at=tk, question=q, subject=f"L4 sample · question · equal highs {ta:%Y-%m-%d %H:%M} server",
        heading="IDEA #1 · A QUESTION",
        sub=f"{_day_words(tk)} · two highs {cents} cent{'' if cents == 1 else 's'} apart; "
            "the chart stops where both are known",
        items_5m=[line], bars_5m=max(40, k - a + 24),
        footer_left="a sample of a question picture - no need to answer it now",
        footer_right="1, 2 = the two highs")
    made.append(D.render(pic, out_dir / "2-question.jpg"))

    # ---- 3. a longer trade: the whole trade on the hour chart + a 5-minute panel for every entry and exit,
    #         and the same trade in his arrows
    sig = by_id[long_.signal_id]
    gap = gap_of(sig, m5)
    gap = D.Box(gap.start, long_.fill_ts, gap.lo, gap.hi)
    stop = D.Level(long_.stop_px, "stop", start=long_.fill_ts, color=D.INK)
    target = D.Level(long_.target_pxs[0], "target", start=long_.fill_ts, color=D.GREEN)
    out_word = {"target": target, "stop": stop}.get(long_.exit_reason)
    tag = "B1" if long_.direction == "buy" else "S1"
    sub = (f"{_day_words(long_.fill_ts)} to {_day_words(long_.exit_ts)} · held {_held(long_.fill_ts, long_.exit_ts)}"
           f"\n" + RULE_WORDS.format(side=long_.direction))
    subject = f"L4 sample · whole trade · {long_.direction} {long_.fill_ts:%Y-%m-%d %H:%M} server"
    pic = D.whole_trade_picture(long_, m5, h1, subject=subject, sub=sub, items_1h=[stop, target],
                                point_items={f"{tag} in": [gap],
                                             f"{tag} C out": [x for x in (out_word,) if x is not None]})
    made.append(D.render(pic, out_dir / "3-whole-trade.jpg"))
    adr_now = adr_band(pd.Timestamp(long_.fill_ts).normalize())
    pic = D.arrows_picture(long_, h1, subject=subject + " · arrows", sub=sub,
                           min_move=0.22 * (adr_now.adr10 or 10.0),
                           items=[D.Box(gap.start, gap.end, gap.lo, gap.hi, named=False), stop, target])
    made.append(D.render(pic, out_dir / "3-whole-trade-arrows.jpg"))

    bad = 0
    for d in made:
        problems = lint.check(d)
        print(f"{d.path.relative_to(config.ROOT).as_posix():45s} {d.size[0]}x{d.size[1]}  "
              f"{'lint: all ' + str(len(lint.RULES)) + ' rules pass' if not problems else 'LINT FAILS:'}")
        for p in problems:
            print("    ", p)
        bad += bool(problems)
    print(f"short trade: {short.signal_id} {short.direction} filled {short.fill_ts} closed {short.exit_ts} "
          f"({short.exit_reason}) net {short.net_inr:+.0f} Rs")
    print(f"long trade:  {long_.signal_id} {long_.direction} filled {long_.fill_ts} closed {long_.exit_ts} "
          f"({long_.exit_reason}) net {long_.net_inr:+.0f} Rs")
    print(f"question:    highs {ta} ({pa:.2f}) and {tb} ({pb:.2f}), known at {tk} (server time)")
    if args.mark_shown and not bad:
        for d in made:
            lint.mark_shown(d, "L4 samples - SHOW TEJAS")
        print(f"marked as shown in {config.SHOWN_PICTURES.relative_to(config.ROOT).as_posix()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
