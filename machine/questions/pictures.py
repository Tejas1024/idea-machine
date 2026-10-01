"""The question pictures (L9b step 3): ONE picture per question, and the ONE confirm picture ("I read your answer as
this - yes?").

Why one picture, not one per reading: Tejas, 29 Sep 2026, on the first batch (two pictures of the same moment, A and
B): "Both images are same for each questions" (strategy/learned-inbox/2026-09-29-0912-session.md). Two charts of the
same candles look the same on a phone; the difference was a colour or one line, and nothing pointed at what was asked.
So now: the whole story on ONE 5-minute chart (his T3 round-3 rule), the thing the readings disagree about MARKED on it
("equal lows? · 25 min apart" - the "?" is the question), and what each reading does written on the same picture
("A: ... -> buy at B1", "B: ... -> no buy here"). He still replies A or B.

His other picture rules (machine/charts/draw.py lists them): every thing named by its job, the question ON the
picture (#132), the chart ENDS at the entry candle and no money anywhere (lesson B16). The chart ends at the FIRST
entry of any reading there (a later one would show him what happened after it).

What is drawn comes from the compile's own trace (what each step found), for every reading that trades there:
  a level with an id in the liquidity book  -> an equal pair: a line with its 2 touches numbered (5-minute pairs);
                                               any other level: a price line - "taken" or "goes here"
  a gap (lo, hi, first_bar)                 -> the blue box
  the entry                                 -> S1 / B1 with one keyword
A level that is in one reading's liquidity book and not in another's is what they disagree about: it gets the "?".
A job that only some readings give a level carries their letter ("A: goes here").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from machine import settings
from machine.charts import draw as D
from machine.data.clock import to_ist
from machine.tester.models import Signal

ONE_MAX = 150              # candles: the whole story on ONE 5-minute chart (as T3 round 3)
BEFORE = 10                # candles shown before the first thing drawn
LOOK_AHEAD = 4000          # 5-minute candles a waiting order is followed to find its entry (about 2 weeks)
TARGET_COLOR = D.FIBC
WHY_ONE = "strategy/learned-inbox/2026-09-29-0912-session.md"


def entry_candle(s: Signal, m5: pd.DataFrame) -> pd.Timestamp | None:
    """The 5-minute candle where the order is filled (on the chart's bid prices): the next candle for next_open; for a
    limit, the first candle from known_at that reaches the price - None when price reaches cancel_at first (#119),
    the order runs out, or nothing happens within LOOK_AHEAD candles. Only to END a picture there: no money, no exit."""
    j = int(m5.index.searchsorted(pd.Timestamp(s.known_at)))
    if j >= len(m5):
        return None
    if s.entry_type == "next_open":
        return m5.index[j]
    n = min(len(m5) - j, LOOK_AHEAD, (s.expiry_bars or LOOK_AHEAD))
    hi, lo = m5["high"].to_numpy()[j:j + n], m5["low"].to_numpy()[j:j + n]
    sell = s.direction == "sell"
    hit = hi >= s.entry_price if sell else lo <= s.entry_price
    cancel = (lo <= s.cancel_at if sell else hi >= s.cancel_at) if s.cancel_at is not None else np.zeros(n, bool)
    h = int(np.argmax(hit)) if hit.any() else None
    c = int(np.argmax(cancel)) if cancel.any() else None
    if h is None or (c is not None and c <= h):
        return None
    return m5.index[j + h]


@dataclass
class Scene:
    """One moment, ready to draw: where the chart ends; for each reading its signal that enters there (None = no
    trade here), what its steps found (only readings that trade), and its own liquidity book."""
    end: pd.Timestamp
    direction: str
    signals: dict[str, Signal | None]
    facts: dict[str, dict]
    books: dict[str, pd.DataFrame]


def scene(moment, compiled: dict, market) -> Scene | None:
    """The moment's end candle and what to draw; None when no reading's order is ever filled there."""
    m5 = market.bars("5m")
    entries = {r: (s, entry_candle(s, m5)) for r, s in moment.signals.items() if s is not None}
    entries = {r: (s, t) for r, (s, t) in entries.items() if t is not None}
    if not entries:
        return None
    end = min(t for _, t in entries.values())
    signals = {r: (entries[r][0] if r in entries and entries[r][1] == end else None) for r in moment.signals}
    facts = {}
    for r, s in signals.items():
        if s is None:
            continue
        rec = next((x for x in compiled[r].trace or [] if x.get("signal") == s.id), None)
        if rec is None:
            return None
        facts[r] = rec["facts"]
    books = {r: market.liquidity(settings.fingerprint(compiled[r].resolved)).book for r in signals}
    return Scene(end, moment.direction, signals, facts, books)


# ============================================================================================== words

def for_him(text: str) -> str:
    """A form's words are written ABOUT him ("his written rule"); on his picture they speak TO him."""
    for a, b in ((r"\bhis\b", "your"), (r"\bHis\b", "Your"), (r"\bhim\b", "you"), (r"\bhe\b", "you"),
                 (r"\bHe\b", "You")):
        text = re.sub(a, b, text)
    return text


def _apart(a: pd.Timestamp, b: pd.Timestamp) -> str:
    m = int(round(abs((b - a).total_seconds()) / 60))
    if m < 120:
        return f"{m} min apart"
    if m < 48 * 60:
        return f"{round(m / 60)} hours apart"
    return f"{round(m / 1440)} days apart"


def _kind_words(kind: str, side: str) -> str:
    if kind.startswith("equal_"):
        return f"equal {side}s"
    if kind.startswith("session_"):
        return f"session {side}"
    return kind.replace("_", " ")


# ============================================================================================== what gets drawn

@dataclass
class _Level:
    key: tuple
    row: pd.Series
    jobs: dict           # reading -> "taken" / "goes here" / ""
    sweep: pd.Timestamp | None
    disputed: bool = False


_BORN: dict[int, tuple] = {}              # id(book) -> (the book, its born rows, their keys): made once per book


def _born_and_keys(book: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    hit = _BORN.get(id(book))
    if hit is not None and hit[0] is book:
        return hit[1], hit[2]
    born = book[book["event"] == "born"].drop_duplicates("level_id").set_index("level_id")
    touch = [pd.Timestamp(t) if pd.notna(t) else None for t in born["touch_bar"]]
    keys = set(zip(born["kind"], born["tf"], (round(float(p), 2) for p in born["price"]),
                   (pd.Timestamp(t) for t in born["first_bar"]), touch))
    _BORN[id(book)] = (book, born, keys)
    return born, keys


def _born(book: pd.DataFrame) -> pd.DataFrame:
    return _born_and_keys(book)[0]


def _key(row: pd.Series) -> tuple:
    return (row["kind"], row["tf"], round(float(row["price"]), 2), pd.Timestamp(row["first_bar"]),
            pd.Timestamp(row["touch_bar"]) if pd.notna(row["touch_bar"]) else None)


def levels(form, sc: Scene) -> list[_Level]:
    """Every level a trading reading's steps used, once each, with the job each reading gives it, and whether the
    readings disagree about it (in one reading's liquidity book, not in another's)."""
    target_step = next((t.at.split(".")[0] for t in form.targets if getattr(t, "at", None)), None)
    first_step = form.steps[0].id
    keys_in = {r: _born_and_keys(b)[1] for r, b in sc.books.items()}
    out: dict[tuple, _Level] = {}
    for r, facts in sc.facts.items():
        born = _born(sc.books[r])
        for step in form.steps:
            f = facts.get(step.id) or {}
            lid = f.get("level_id")
            if not lid or lid not in born.index:
                continue
            row = born.loc[lid]
            k = _key(row)
            job = "taken" if step.id == first_step else ("goes here" if step.id == target_step else "")
            lv = out.setdefault(k, _Level(k, row, {}, None))
            lv.jobs[r] = job
            if job == "taken" and "bar" in f:
                lv.sweep = pd.Timestamp(f["bar"])
    for lv in out.values():
        lv.disputed = not all(lv.key in keys_in[r] for r in sc.books)
    return list(out.values())


def items_for(form, sc: Scene, letters: dict[str, str], m5: pd.DataFrame) -> tuple[list, list[pd.Timestamp], list]:
    """The things to draw for the whole question (every reading at once), every time they need on the chart, and the
    disputed levels."""
    trading = [r for r, s in sc.signals.items() if s is not None]
    out, times, disputed = [], [], []
    for lv in levels(form, sc):
        row = lv.row
        price, side = float(row["price"]), str(row["side"])
        what = _kind_words(str(row["kind"]), side) + ("?" if lv.disputed else "")
        jobs = {j for j in lv.jobs.values()}
        if len(jobs) == 1 and set(lv.jobs) == set(trading):
            job = jobs.pop()
        else:
            job = " ".join(f"{letters[r]}: {j}" for r, j in sorted(lv.jobs.items(), key=lambda x: letters[x[0]]) if j)
        color = TARGET_COLOR if "goes here" in job else D.PURPLE
        if str(row["kind"]).startswith("equal_") and row["tf"] == "5m" and pd.notna(row["touch_bar"]):
            a, b = pd.Timestamp(row["first_bar"]), pd.Timestamp(row["touch_bar"])
            name = " · ".join(x for x in (what, job, _apart(a, b)) if x)
            stop = lv.sweep - pd.Timedelta(minutes=5) if lv.sweep is not None and "taken" in job else sc.end
            out.append(D.Line((a, price), (b, price), touches=(a, b), name=name, extend_to=max(stop, b), color=color))
            times += [a, b]
            if lv.disputed:
                disputed.append(f"the {what} ({_apart(a, b)})")
        else:
            tf = "" if row["tf"] == "5m" else f"{row['tf']} "
            out.append(D.Level(price, " · ".join(x for x in (f"{tf}{what}", job) if x), color=color))
            if lv.disputed:
                disputed.append(f"the {tf}{what}")
    gaps = {}
    for r in trading:
        f = next((sc.facts[r].get(s.id) for s in form.steps if {"lo", "hi", "first_bar"} <= set(sc.facts[r].get(s.id)
                                                                                                or {})), None)
        if f:
            gaps.setdefault((pd.Timestamp(f["first_bar"]), float(f["lo"]), float(f["hi"])), []).append(r)
    for (start, lo, hi), rs in gaps.items():
        name = "gap" if len(gaps) == 1 else f"{'/'.join(letters[r] for r in rs)} gap"
        out.append(D.Box(start, sc.end, lo, hi, name=name))
        times.append(start)
    at = (form.entry.at or "").split(".")[0] if form.entry.how == "limit" else ""
    block = next((x.block for x in form.steps if x.id == at), "")
    word = "gap" if "gap" in block else ("fib" if "fib" in block else "")
    prices = {}
    for r in trading:
        s = sc.signals[r]
        prices.setdefault(float(s.entry_price if s.entry_price is not None else m5.loc[sc.end, "open"]), []).append(r)
    tag0 = "S" if sc.direction == "sell" else "B"
    for k, (px, rs) in enumerate(sorted(prices.items()), 1):
        w = word if len(prices) == 1 else "/".join(letters[r] for r in rs)
        out.append(D.Marker(sc.end, px, f"{tag0}{k}", sc.direction, w))
    return out, times, disputed


def outcome(sc: Scene, reading: str, letters: dict[str, str]) -> str:
    """What one reading does at this moment, in a few words: 'buy at B1' / 'no buy here' / '... goes to the lows
    marked A'."""
    verb = "sell" if sc.direction == "sell" else "buy"
    s = sc.signals[reading]
    if s is None:
        return f"no {verb} here"
    trading = [r for r, x in sc.signals.items() if x is not None]
    prices = sorted({x.entry_price for x in (sc.signals[r] for r in trading)})
    tag = f"{'S' if sc.direction == 'sell' else 'B'}{prices.index(s.entry_price) + 1}"
    text = f"{verb} at {tag}"
    if len({tuple(sc.signals[r].targets) for r in trading}) > 1:
        side = "lows" if sc.direction == "sell" else "highs"
        text += f", aims at the {side} marked {letters[reading]}"
    return text


def _lines(letter: str, means: str, what: str) -> tuple[str, ...]:
    one = f"{letter}: {means} → {what}"
    return (one,) if len(one) <= 100 else (f"{letter}: {means}", f"      → {what}")


def picture(form, part, sc: Scene, m5: pd.DataFrame, *, subject: str, letters: dict[str, str], number: str,
            confirm: str | None = None) -> D.Picture | None:
    """The question's ONE picture (confirm=None), or the confirm picture of reading `confirm`. None when the story
    does not fit one chart."""
    items, times, disputed = items_for(form, sc, letters, m5)
    e = int(m5.index.get_loc(sc.end))
    start = max(0, min([int(m5.index.searchsorted(t)) for t in times] + [e]) - BEFORE)
    if e - start + 1 > ONE_MAX:
        return None
    bars = m5.iloc[start:e + 1]
    side = "SELL" if sc.direction == "sell" else "BUY"
    means = {r.id: for_him(r.means) for r in part.readings}
    order = sorted(letters, key=lambda r: letters[r])
    when = to_ist(sc.end).strftime("%d %b %Y, %H:%M")
    label = f"IDEA #{int(form.id.split('-')[1])}"
    look = "Look at " + " and ".join(disputed) + ": A and B read it differently." if disputed else \
        "A and B trade this moment differently."
    if confirm is not None:
        letter = letters[confirm]
        heading = f"I READ YOUR ANSWER AS {letter}"
        sub = f"{label} · another day: {when} India time\n{look}"
        q = D.Question("Is this how you trade it?",
                       _lines(letter, means[confirm], outcome(sc, confirm, letters))
                       + ("Reply YES to lock it. Or NO and a few words why.",))
    else:
        heading = f"{number} · A OR B?"
        sub = f"{label} · {when} India time\n{look}"
        how = tuple(x for r in order for x in _lines(letters[r], means[r], outcome(sc, r, letters)))
        q = D.Question(f"Which one is how you trade it: {' or '.join(letters[r] for r in order)}?",
                       how + (f"Reply {' or '.join(letters[r] for r in order)} - or draw your own on the picture.",))
    tag = "S1" if sc.direction == "sell" else "B1"
    footer = f"{tag} = the {side.lower()} · 1, 2 = the two touches of a pair · ? = what A and B read differently"
    return D.Picture(
        kind="question", subject=subject, heading=heading, heading_color=D.side_colour(sc.direction), sub=sub,
        rows=[[D.Panel(bars, "5m", [D.Candles(), *items], "the whole story · ends at the entry candle", role="main")]],
        question=q, footer_left=footer, footer_right="the chart ends at the entry candle",
        entry_at=sc.end, entry_tf="5m")
