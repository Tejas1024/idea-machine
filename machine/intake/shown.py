"""SHOWN (L10b): before any money, the machine shows Tejas 3 setups it found by itself, in ONE picture: "is this your
idea?" It counts as ONE question of the weekly batch (config.QUESTION_BUDGET).

    from machine.intake import shown
    rec = shown.make("idea-002", market)          # plan/work/ideas/idea-002/shown.jpg + shown.json (nothing sent)

What is drawn (the same drawing as the L9b question pictures, machine/questions/pictures.py):
  - the idea compiled on the TUNE years (2016-2020 - never the check years) with his answers; a VERSION part (a number
    the machine tries) at its FIRST reading - the words he wrote, when a reading says so (Claude's reading: any one
    version shows the idea; the numbers are what the tune tries);
  - 3 setups spread over the years (one early, one in the middle, one late), in 3 different weeks, a sell and a buy
    when the idea trades both; each one's whole story on ONE 5-minute chart that ENDS at its entry candle (lesson B16:
    nothing after the entry, so no result and no money can be seen);
  - one chart per setup, numbered 1, 2, 3 in its title, each thing named by its job (his T3 rule: "taken",
    "goes here", the gap, S1 / B1), the question ON the picture (#132).
Why three charts in one picture: the plan's words ("3 machine-found trades in one picture ... counts as 1 question").
His T3 complaint about three charts was about ONE trade drawn three times ("which chart I need to see?"); here each
chart is a different trade, and its number is in its title (Claude's reading - SHOW TEJAS on the first one).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from machine import config
from machine.charts import draw as D
from machine.charts import lint
from machine.data.clock import to_ist
from machine.ideas import store
from machine.ideas.form import Idea, open_parts
from machine.questions import pictures as P

SETUPS = 3
FILE, RECORD = "shown.jpg", "shown.json"
DRAWING = 4       # the drawing's version (lesson 19: a saved thing carries what made it). A picture MADE but not sent
                  # yet that an older drawing made is drawn again - the same setups - before the batch sends it.
                  # 2 (1 Oct): a session level is named by its session ("London high · taken", his words for idea
                  # #3), not "session high"; the footer names only the marks drawn.
                  # 3 (7 Oct): no setup's chart shares a candle with a setup chart SENT before in any idea's picture
                  # (seen_spans; his #117 "I don't want the same charts again") - idea #4's drawing 2 showed idea
                  # #3's sell of 12 Jun 2018 20:05 again, the chart he had answered that morning.
                  # 4 (9 Oct): his strong zones drawn as their box, a triangle as both its lines, a trendline as the
                  # line with its touches numbered; the second gap named and one chart of it chosen - before, an
                  # idea of his zone entry, zone breakout, triangle or trendline had NO picture (0 of 236 / 54 / 29 /
                  # 11 setups passed the picture rules: each named a thing the chart did not draw).
ASK ="Are these 3 your idea?"
HOW = ("YES = all 3 are how you trade it. Then the machine tests it by itself.",
       "NO = say which number (1, 2 or 3) and a few words why.")


class ShownError(ValueError):
    """The three setups cannot be made (not enough setups fit one chart, or the idea is not locked)."""


def record_path(idea: str) -> Path:
    return store.folder(idea) / RECORD


def load(idea: str) -> dict | None:
    p = record_path(idea)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def save(rec: dict) -> None:
    store._write(record_path(rec["idea"]), json.dumps(rec, indent=1, ensure_ascii=False) + "\n")


def stale(rec: dict | None) -> bool:
    """Made, not sent, and drawn by an older drawing (DRAWING): it is drawn again before it goes out."""
    return bool(rec) and rec.get("status") == "made" and int(rec.get("drawing", 1)) < DRAWING


def qid(idea: str, n: int = 1) -> str:
    return f"{idea}-shown-{n}"


# ================================================================================================= which setups

def first_readings(form: Idea) -> dict[str, str]:
    """Every part still open (a VERSION part - ASK parts are answered before this step) at its first reading."""
    return {u.id: u.readings[0].id for u in open_parts(form)}


def spread(signals: list, n_tries: int = 60) -> list:
    """The signals to try, in order: evenly spread over the years (early, middle, late ...), then the rest."""
    sigs = sorted(signals, key=lambda s: pd.Timestamp(s.known_at))
    if len(sigs) <= n_tries:
        picks = list(range(len(sigs)))
    else:
        picks = [round(k * (len(sigs) - 1) / (n_tries - 1)) for k in range(n_tries)]
    order = []
    for band in range(SETUPS):                     # 3 bands; inside each, from its middle outwards
        lo, hi = band * len(picks) // SETUPS, (band + 1) * len(picks) // SETUPS
        mid = (lo + hi) // 2
        idx = sorted(range(lo, hi), key=lambda k: (abs(k - mid), k))
        order += [sigs[picks[k]] for k in idx]
    return order


def seen_spans() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Every setup chart he was SENT in a picture of 3 setups, of any idea (each idea's shown.json and the kept
    shown-<n>.json - one taken back counts: he saw it): (its first candle, its entry candle), MT5 server time. A record
    made before 7 Oct 2026 keeps no first candle: the longest chart there can be (P.ONE_MAX candles before its entry)
    stands in - never too short."""
    out = []
    for idea in store.ideas():
        d = store.folder(idea)
        for path in [d / RECORD, *sorted(d.glob("shown-*.json"))]:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not rec.get("sent"):
                continue
            for s in rec.get("setups") or []:
                try:
                    end = pd.Timestamp(s["end"])
                    first = (pd.Timestamp(s["first"]) if s.get("first") else
                             end - pd.Timedelta(minutes=5 * (P.ONE_MAX - 1)))
                except (KeyError, TypeError, ValueError):
                    continue
                out.append((first, end))
    return out


def shown_before(panel: D.Panel, seen: list[tuple[pd.Timestamp, pd.Timestamp]] | None) -> bool:
    """Does this chart share ANY candle with a setup chart he was sent before (seen_spans)? The zone charts' rule too
    (machine/labels/daily._clash): every chart a stretch of price he has never seen - his #117."""
    a, b = panel.bars.index[0], panel.bars.index[-1]
    return any(a <= end and first <= b for first, end in seen or [])


def the_test_enters(s) -> bool:
    """An order with his #68 check (first_visit_from): does the tester really enter it? The chart draws a touch of the
    entry as the entry, but when the spread kept a buy limit from filling and price turned away, the tester calls it
    spent (fvg.first_visit) - such a setup is never shown as the idea's (7 Oct 2026). One replay on the 1-minute bars
    of its own days (the tune years only)."""
    if s.entry_type != "limit" or s.first_visit_from is None:
        return True
    from machine.data.bars import load_bars
    from machine.tester import engine
    a = pd.Timestamp(s.first_visit_from) - pd.Timedelta(hours=1)
    b = min(pd.Timestamp(s.known_at) + pd.Timedelta(minutes=5 * P.LOOK_AHEAD),
            pd.Timestamp(f"{config.TUNE_YEARS[-1]}-12-31 23:59"))
    m1 = load_bars("1m", str(a.date()), str(b.date()))
    return bool(engine.replay([s], m1, "5m", 1).trades)


def choose(form: Idea, compiled, market, *, n: int = SETUPS, avoid: set[str] | None = None,
           prefer: list[str] | None = None, seen: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
           log=print) -> list[P.Scene]:
    """n scenes of different weeks (a sell and a buy when both exist), each fitting ONE 5-minute chart and passing
    every picture rule. Fewer when the idea does not have them. avoid: signal ids shown before (never twice, #117).
    seen: the setup charts sent before in any idea's picture (seen_spans) - a chart with a candle of one is never
    chosen (#117). prefer: the setups of a picture taken back before he judged it (withdraw) - drawn again when all
    still pass."""
    m5 = market.bars("5m")
    comp = {"A": compiled}
    chosen, weeks, again_seen = [], set(), 0
    tmp = Path(tempfile.mkdtemp(prefix="l10b-"))
    try:
        if prefer:
            by_id = {s.id: s for s in compiled.signals}
            again = []
            for sid in prefer:
                s = by_id.get(sid)
                sc = None if s is None else P.scene(SimpleNamespace(signals={"A": s}, direction=s.direction), comp,
                                                    market)
                panel = None if sc is None else one_chart(form, sc, m5)
                if panel is None or shown_before(panel, seen) or lint.check(
                        D.render(picture(form, [sc], m5, subject="trial"), tmp / "trial.jpg"), shown=tmp / "n.jsonl"):
                    break
                again.append(sc)
            if len(again) == len(prefer):
                log(f"{form.id}: the same {len(again)} setups drawn again (he never judged them)")
                return again
        by_band = spread([s for s in compiled.signals if s.id not in (avoid or set())])
        band_size = max(1, len(by_band) // SETUPS)
        second = second_gaps(compiled)
        for band in range(SETUPS):
            want_dir = None
            if band == SETUPS - 1 and chosen and len({s.direction for s in compiled.signals}) > 1:
                have = {sc.direction for sc in chosen}
                want_dir = next((d for d in ("sell", "buy") if d not in have), None)
            pool = by_band[band * band_size:(band + 1) * band_size] if band < SETUPS - 1 else \
                by_band[band * band_size:]
            if want_dir:
                pool = [s for s in pool if s.direction == want_dir] + [s for s in pool if s.direction != want_dir]
            if band == 0 and second:                     # the part that makes it HIS second-gap idea, on one chart -
                pool = [s for s in by_band if s.id in second] + pool    # else all 3 can be plain first gaps (9 Oct)
            for s in pool:
                week = pd.Timestamp(s.known_at).isocalendar()[:2]
                if week in weeks:
                    continue
                sc = P.scene(SimpleNamespace(signals={"A": s}, direction=s.direction), comp, market)
                if sc is not None and not the_test_enters(s):
                    continue
                panel = None if sc is None else one_chart(form, sc, m5)
                if panel is None:
                    continue
                if shown_before(panel, seen):
                    again_seen += 1
                    continue
                if lint.check(D.render(picture(form, [sc], m5, subject="trial"), tmp / "trial.jpg"),
                              shown=tmp / "none.jsonl"):
                    continue
                chosen.append(sc)
                weeks.add(week)
                break
            if len(chosen) >= n:
                break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"{form.id}: {len(chosen)} setups chosen for the SHOWN picture from {len(compiled.signals)} signals"
        + (f" ({again_seen} skipped: a candle of a chart he was sent before, #117)" if again_seen else ""))
    return chosen


# ================================================================================================= the picture

ZONE_NAMES = {True: ("red zone · top of the day", D.ADR_TOP), False: ("green zone · bottom of the day", D.ADR_BOT)}


def _facts(sc: P.Scene) -> dict:
    return sc.facts.get("A") or next(iter(sc.facts.values()), {})


def _story(form: Idea, sc: P.Scene) -> tuple[list, list[pd.Timestamp]]:
    """What the question pictures (L9b) do not draw, and the idea needs: every step's moment on the time axis (the
    chart starts before the FIRST step - the grab - so the whole story is on it), and a zone a step found (his
    day's red / green zone, premium_discount - L6b.3) as its box. Found on idea #2's first picture (29 Sep): its
    zone - the heart of his idea - was not drawn, and one chart began after the grab. The box is named by its colour
    in his Pine study, not "ADR" (the picture rule for "ADR" wants his whole two-band drawing)."""
    facts = _facts(sc)
    items, times = [], []
    for step in form.steps:
        f = facts.get(step.id) or {}
        times += [pd.Timestamp(f[k]) for k in ("bar", "first_bar") if f.get(k) is not None]
        if f.get("zone_lo") is not None and f.get("zone_hi") is not None:
            side = (step.params or {}).get("side", "sell_top_buy_bottom")
            if step.block == "adr_zone_break":            # L6-new: a sell breaks out BELOW the green zone (#76)
                side = "sell_bottom_buy_top"
            name, color = ZONE_NAMES[(sc.direction == "sell") == (side == "sell_top_buy_bottom")]
            known = pd.Timestamp(f["known_at"]) if f.get("known_at") is not None else None
            items.append((known, D.Zone(float(f["zone_lo"]), float(f["zone_hi"]), name=name, color=color)))
        elif step.block in STRONG_ZONE_STEPS and f.get("lo") is not None and f.get("hi") is not None:
            known = pd.Timestamp(f["known_at"]) if f.get("known_at") is not None else None    # L6b.2: his strong
            items.append((known, D.Zone(float(f["lo"]), float(f["hi"]), name=zone_name(f))))  # zone, its box
    return items, [t for t in times if t <= sc.end]


STRONG_ZONE = "strong zone"
STRONG_ZONE_STEPS = ("zone_entry", "zone_break", "zone")


def zone_name(f: dict) -> str:
    """'strong zone · 5-min chart, 5 days': the candles it was counted on - its zone_id starts with the window's name
    (settings zone_windows, L6b.2: "week" = the 5-minute candles of the last 5 days, NOT the week chart)."""
    from machine.blocks import zones
    name = str(f.get("zone_id") or "").split(":", 1)[0]
    for w, tf, days, _, _ in zones.windows():
        if w == name:
            chart = "hour chart" if tf == "1h" else "5-min chart" if tf == "5m" else f"{tf} chart"
            return f"{STRONG_ZONE} · {chart}, {days} day{'s' if days != 1 else ''}"
    return STRONG_ZONE


LINE_MAX = 180   # candles: a chart with one of his LINES (a triangle, a trendline) may be one hour longer than
                 # P.ONE_MAX - the line's first touch must be on it. Claude's number (his #110: "whichever you think
                 # visually good for me"): 9 Oct, of his triangle false breaks on 2016-2020, 6 of 25 too-long
                 # stories were 151-179 candles; the rest 215 to 2,026 - one 5-minute chart cannot hold those.
LINE_ID = re.compile(r"(line_high|line_low):(\w+):(\d{4}-\d\d-\d\d \d\d:\d\d):(\d{4}-\d\d-\d\d \d\d:\d\d)")


def anchors(line_id: str, m5: pd.DataFrame) -> tuple[tuple[pd.Timestamp, float], ...] | None:
    """His line rebuilt from the two wick ends it was drawn through (#112; its id names them, L6b.1): the highs of a
    line_high, the lows of a line_low, each at its own candle's time (an hour line's wick = the hour's highest 5-minute
    high). None when the id is not a line's or a candle is missing."""
    from machine.tester.models import TF_MINUTES
    m = LINE_ID.fullmatch(str(line_id))
    if m is None or m.group(2) not in TF_MINUTES:
        return None
    kind, tf, a, b = m.groups()
    width = pd.Timedelta(minutes=TF_MINUTES[tf] - 5)
    out = []
    for t in (pd.Timestamp(a), pd.Timestamp(b)):
        part = m5.loc[t:t + width, "high" if kind == "line_high" else "low"]
        if part.empty:
            return None
        out.append((t, float(part.max() if kind == "line_high" else part.min())))
    return tuple(out)


def _lines(form: Idea, sc: P.Scene, m5: pd.DataFrame) -> tuple[list, list[pd.Timestamp]]:
    """His lines a step found, drawn the way the picture rules want them (#113 rule 3): a trendline (his Type 2, the
    3rd touch) as the line with touches 1, 2, 3 numbered; a triangle as BOTH its lines, one shape, drawn to where price
    pushed out of it. A 4th-touch trendline is not drawn (where the 3rd touch was is not known here): that setup is
    not chosen for the picture."""
    facts = _facts(sc)
    items, times = [], []
    for step in form.steps:
        f = facts.get(step.id) or {}
        if step.block == "trendline_touch" and f.get("line_id") and f.get("bar") is not None and f.get("touch") == 3:
            got = anchors(f["line_id"], m5)
            if got is None:
                continue
            (t1, p1), (t2, p2) = got
            touch = pd.Timestamp(f["bar"])
            items.append(D.Line((t1, p1), (t2, p2), touches=(t1, t2, touch), name="trendline", extend_to=sc.end))
            times += [t1, t2, touch]
        elif step.block == "triangle" and f.get("tri_id") and f.get("bar") is not None:
            ids = str(f["tri_id"]).split(":", 2)[-1].split("|")              # "tri:<chart>:<line>|<line>"
            got = [anchors(i, m5) for i in ids]
            if len(got) != 2 or None in got:
                continue
            upto = max([pd.Timestamp(f["bar"]), *[pd.Timestamp(g["bar"]) for g in facts.values()
                                                  if g.get("level_id") == f["tri_id"] and g.get("bar") is not None]])
            for (t1, p1), (t2, p2) in got:
                items.append(D.Line((t1, p1), (t2, p2), touches=(t1, t2), name="triangle", extend_to=upto,
                                    group=str(f["tri_id"])))
                times += [t1, t2]
    return items, [t for t in times if t <= sc.end]


BREAK_NAME = "break"                          # short: a long name in the right margin squeezed the time axis
BREAK_MEANS = "break = price went past the last candle of the other colour"


def _break(form: Idea, sc: P.Scene, m5: pd.DataFrame) -> tuple[list, list[pd.Timestamp]]:
    """THE REASON TO ENTER, drawn (his words 1 Oct, inbox 1747: "I can't see the reason to enter so I could not judge
    it" - the first idea #2 picture drew what was taken, the zone and the gap, but not the break): each break of
    structure as a short line at the price that broke - his order block, the last red candle in the rise for a sell
    (its low), the last green candle in the fall for a buy (its high), #56-#60 - from that candle to the candle that
    broke it, named; and one of his blue arrows (#122) for the breaking move: from the grab's extreme to the break."""
    facts = _facts(sc)
    items, times = [], []
    for k, step in enumerate(form.steps):
        f = facts.get(step.id) or {}
        if step.block != "break_of_structure" or f.get("level") is None or f.get("block_bar") is None:
            continue
        bar, block = pd.Timestamp(f["bar"]), pd.Timestamp(f["block_bar"])
        if not block <= bar <= sc.end:
            continue
        items.append(D.Level(float(f["level"]), BREAK_NAME, start=block, color=D.INK,
                             style="-", width=2.0, name_below=sc.direction == "sell"))   # its name in the right
                                                                          # margin: on the chart it sat on candles
        times.append(block)
        before = facts.get((step.params or {}).get("after") or (form.steps[k - 1].id if k else "")) or {}
        t0 = pd.Timestamp(before["bar"]) if before.get("bar") is not None else block
        part = m5.loc[min(t0, block):bar]
        if f.get("extreme") is not None and len(part) > 1:
            t_ext = part["high"].idxmax() if sc.direction == "sell" else part["low"].idxmin()
            if t_ext < bar:
                items.append(D.Arrow(((t_ext, float(f["extreme"])), (bar, float(f["level"])))))
                times.append(t_ext)
    return items, times


STEP_WORDS = {"sweep": "taken", "false_breakout": "false break", "break_of_structure": "break",
              "gap_of_the_move": "first gap", "premium_discount": "at the zone",
              "adr_zone_break": "closed past the zone",
              "zone_entry": "a level just outside a strong zone taken → at the zone",
              "zone_break": "a strong zone broken for real", "zone": "at a strong zone",
              "trendline_touch": "3rd touch of the trendline", "triangle": "a triangle"}
GAP_WORDS = {"second": "first gap, or the second when the trend is very strong", "deepest": "biggest gap of the move"}


def why_line(form: Idea) -> str:
    """The reason to enter in his order, from the idea's own steps: the words each chart draws (inbox 1747)."""
    parts, target = [], False
    for s in form.steps:
        if s.block == "level_waiting" and (s.params or {}).get("where", "target_side") == "target_side":
            target = True                                # its default side is the target's (compile.py)
            continue
        short = s.what if len(s.what) <= 24 else s.block.replace("_", " ")
        word = STEP_WORDS.get(s.block, short)
        if s.block == "gap_of_the_move":                 # settings which_gap_of_the_move (L19b, his #124)
            word = GAP_WORDS.get(str((form.settings or {}).get("which_gap_of_the_move", "first")), word)
        parts.append(word)
    text = "Why it enters: " + " → ".join(parts) + " → entry"
    return text + (" · it aims at \"goes here\"" if target else "")


def second_gaps(compiled) -> set[str]:
    """The setups that entered at the move's SECOND gap (settings which_gap_of_the_move = 'second', L19b): with a
    plain first-gap idea, none. About 130 of 4,850 for his second-gap entry on 2016-2020 - a picture spread by year
    alone showed only first gaps."""
    return {x.get("signal") for x in getattr(compiled, "trace", None) or []
            if any(isinstance(v, dict) and v.get("gap_rank") == 1 for v in (x.get("facts") or {}).values())}


def one_chart(form: Idea, sc: P.Scene, m5: pd.DataFrame) -> D.Panel | None:
    """One setup's chart: its whole story on ONE 5-minute chart that ends at its entry candle (None: too long)."""
    items, times, _ = P.items_for(form, sc, {"A": "A"}, m5)
    brk, brk_times = _break(form, sc, m5)
    shapes, shape_times = _lines(form, sc, m5)
    items, times = [*brk, *shapes, *items], [*times, *brk_times, *shape_times]
    zones, step_times = _story(form, sc)
    e = int(m5.index.get_loc(sc.end))
    start = max(0, min([int(m5.index.searchsorted(t)) for t in times + step_times] + [e]) - P.BEFORE)
    if e - start + 1 > (LINE_MAX if shapes else P.ONE_MAX):
        return None
    bars = m5.iloc[start:e + 1]
    day0 = pd.Timestamp(sc.end).normalize()       # the zones are the broker day's (server clock): its first candle on
    for _, z in zones:                            # the chart, never a day before it
        s = bars.index[0] if day0 <= bars.index[0] else bars.index[min(int(bars.index.searchsorted(day0)),
                                                                       len(bars) - 1)]
        items.append(D.Zone(z.lo, z.hi, name=z.name, start=None if z.name.startswith(STRONG_ZONE) else s,
                            color=z.color))              # a strong zone was there days before: the whole chart
    return D.Panel(bars, "5m", [D.Candles(), *items], "", role="main", entry_at=sc.end)


def footer(form: Idea, scenes: list[P.Scene], rows: list) -> str:
    """What the marks mean - only the marks this picture draws (his rule: a picture draws every thing it names; idea
    #3's first picture said "B1 = the buy" and "two touches of a pair" over 3 sells with no pair)."""
    sides = {sc.direction for sc in scenes}
    pair = any(isinstance(x, D.Line) and x.touches for row in rows for p in row for x in p.items)
    parts = [*(["S1 = the sell"] if "sell" in sides else []), *(["B1 = the buy"] if "buy" in sides else []),
             *(["1, 2 = two touches of a pair"] if pair else []),
             *([BREAK_MEANS] if any(x.block == "break_of_structure" for x in form.steps) else [])]
    return " · ".join(parts)


def picture(form: Idea, scenes: list[P.Scene], m5: pd.DataFrame, *, subject: str) -> D.Picture:
    rows = []
    for k, sc in enumerate(scenes, 1):
        panel = one_chart(form, sc, m5)
        if panel is None:
            raise ShownError(f"setup {k} does not fit one chart")
        side = "SELL" if sc.direction == "sell" else "BUY"
        panel.title = f"{k} · {side} · {to_ist(sc.end).strftime('%d %b %Y, %H:%M')}"
        rows.append([panel])
    label = f"IDEA #{int(form.id.split('-')[1])}"
    n = len(scenes)
    return D.Picture(
        kind="question", subject=subject, heading=f"{label} · IS THIS YOUR IDEA?", heading_color=D.INK,
        sub=f"{n} setup{'s' if n != 1 else ''} the machine found by itself in 2016-2020. Each chart ends at its "
            f"entry candle.\n{why_line(form)}",
        rows=rows, question=D.Question(ASK if n == SETUPS else f"Are these {n} your idea?", HOW),
        footer_left=footer(form, scenes, rows),
        footer_right="",                          # said at the top; the footer needs the room (the break)
        entry_at=max(sc.end for sc in scenes), entry_tf="5m")


# ================================================================================================= make it

def archive(idea: str, rec: dict) -> None:
    """A picture he said NO to stays, as shown-<n>.json / .jpg (it was shown: kept, never sent again - #117)."""
    d = store.folder(idea)
    n = rec.get("n", 1)
    for name, suffix in ((FILE, ".jpg"), (RECORD, ".json")):
        if (d / name).is_file():
            os.replace(d / name, d / f"shown-{n}{suffix}")
    if rec.get("picture"):
        rec["picture"]["file"] = _rel(d / f"shown-{n}.jpg")
    store._write(d / f"shown-{n}.json", json.dumps(rec, indent=1, ensure_ascii=False) + "\n")


def withdraw(idea: str, why: str, *, now: dt.datetime | None = None) -> dict:
    """Take back a picture he could not judge (his words 1 Oct, inbox 1747: "I can't see the reason to enter so I could
    not judge it"): not a NO about the idea - the meaning stays locked. The idea goes back to CONFIRMED & LOCKED, a NEW
    picture of the same setups is made (make), and a late reply to the old one changes nothing. Only a picture that
    is made or sent and not answered can be taken back."""
    rec = load(idea)
    if rec is None or rec.get("status") not in ("made", "sent"):
        raise ShownError(f"{idea}: no picture of 3 setups waits for his answer (status "
                         f"{(rec or {}).get('status')!r}) - nothing to take back")
    if not str(why).strip():
        raise ShownError("taking a picture back needs the reason (his words)")
    now = now or dt.datetime.now().astimezone()
    rec.update(status="withdrawn", withdrawn={"at": now.isoformat(timespec="seconds"), "why": str(why)})
    save(rec)
    store.set_state(idea, "CONFIRMED & LOCKED", f"the picture of 3 setups ({rec['qid']}) taken back: {why}"[:300],
                    waits_for="the machine: a new picture of the same 3 setups")
    return rec


def make(idea: str, market, *, now: dt.datetime | None = None, compile_fn=None, log=print) -> dict:
    """The SHOWN picture of a CONFIRMED & LOCKED idea, saved with its record (nothing sent - the weekly batch sends
    it). A record already made is returned as it is (never drawn twice). After his NO the old one is kept apart and a
    new one is made from other setups."""
    rec = load(idea)
    if rec is not None and rec.get("status") not in ("failed", "no", "withdrawn") and not stale(rec):
        return rec
    n, avoid, prefer = 1, set(), None
    if rec is not None:
        avoid = {x["signal"] for x in rec.get("setups", [])} | set(rec.get("avoid", []))
        n = rec.get("n", 1)
        if rec["status"] == "withdrawn" or stale(rec):   # taken back before he judged it, or never sent: the same
            prefer, avoid = [x["signal"] for x in rec.get("setups", [])], set(rec.get("avoid", []))   # setups anew
            log(f"{idea}: the picture of 3 setups is drawn again ({rec['status']}, drawing "
                f"{rec.get('drawing', 1)} -> {DRAWING})")
        if rec["status"] in ("no", "withdrawn"):
            archive(idea, rec)
            n += 1
    now = now or dt.datetime.now().astimezone()
    form = store.load_form(idea)
    if compile_fn is None:
        from machine.ideas.compile import compile_idea as compile_fn
    readings = first_readings(form)
    c = compile_fn(form, market, readings=readings, trace=True)
    scenes = choose(form, c, market, avoid=avoid, prefer=prefer, seen=seen_spans(), log=log)
    rec = {"idea": idea, "n": n, "qid": qid(idea, n), "made": now.isoformat(timespec="seconds"), "drawing": DRAWING,
           "readings": readings, "version_id": c.version_id, "signals": len(c.signals), "setups": [],
           "avoid": sorted(avoid), "status": "failed", "replies": []}
    if len(scenes) < SETUPS:
        rec["why"] = (f"only {len(scenes)} of {len(c.signals)} setups fit one 5-minute chart, pass every picture "
                      "rule and share no candle with a chart he was sent before - a Claude session looks at it")
        save(rec)
        raise ShownError(rec["why"])
    path = store.folder(idea) / FILE
    pic = picture(form, scenes, market.bars("5m"), subject=f"L10b {qid(idea, n)} (3 setups the machine found)")
    drawn = D.render(pic, path)
    problems = lint.check(drawn)
    if problems:
        rec["why"] = "the picture breaks a picture rule: " + "; ".join(str(p) for p in problems)
        save(rec)
        raise ShownError(rec["why"])
    rec.update(status="made", picture={"file": _rel(path), "subject": pic.subject, "pixel_sha256": drawn.pixel_sha256,
                                       "lint": f"all {len(lint.RULES)} rules pass",
                                       "texts": [t.text for t in drawn.texts]},
               setups=[{"n": k, "direction": sc.direction, "first": str(row[0].bars.index[0]), "end": str(sc.end),
                        "end_ist": to_ist(sc.end).strftime("%d %b %Y %H:%M"),
                        "signal": next(s.id for s in sc.signals.values() if s is not None)}
                       for k, (sc, row) in enumerate(zip(scenes, pic.rows), 1)])
    save(rec)
    log(f"{idea}: SHOWN picture made -> {_rel(path)}")
    return rec


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
