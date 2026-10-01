"""L6.2 acceptance: do the liquidity blocks find every example Tejas said YES to, and never one he said NO to?

    .venv\\Scripts\\python -m tests.idea001_acceptance             table, misses drawn, the question queued (writes)
    .venv\\Scripts\\python -m tests.idea001_acceptance --no-write  the table only

His answers are in plan/work/idea-001/examples.md (T3, 28 Sep 2026): round 3 is COUNTED (10 yes, 5 no, one 5-minute
chart each); rounds 1-2 are shown apart and not counted (three charts per picture confused him - inbox 1130).

The setups are idea #1's 7 agreed steps (plan/work/idea-001/steps.md) with the NEW liquidity blocks in the place of
T3's rough search: the equal highs / lows (machine/blocks/equal_levels.py, on the 5-minute, hour and day charts), the
sweep and "a level swept once is finished" (machine/blocks/liquidity.py sweeps, #129), and what is alive at the sweep
(liquidity.alive). The part after the sweep - the break, the first gap, the first visit - is T3's own walk
(find_examples._walk, built on the L6.1 blocks), so any difference here comes from the liquidity blocks alone. The
real form of idea #1 is L7b's job. 2016-2020 only; nothing after an entry candle is read; no money.

Writes plan/work/L6-misses/ (README.md + one picture per miss, each ending at its entry) and the ONE question the
misses raise into plan/work/question-queue/ (for the weekly batch, L9b), on a new picture that isolates it.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from machine import config, rules, settings
from machine.blocks import adr as ADR
from machine.blocks import fvg as FVG
from machine.blocks import liquidity as LQ
from machine.blocks import structure as ST
from machine.data.bars import load_bars
from machine.data.clock import to_ist, utc_to_server

IDEA = config.WORK_DIR / "idea-001"
MISSES = config.WORK_DIR / "L6-misses"
QUEUE = config.WORK_DIR / "question-queue"
YEARS = ("2016", "2020")
RUN_LINE = 0.25          # Claude's line between his 0.22 yes and 0.27 no (examples.md) - L7b builds and tests it
NEAR_LINE = 7            # Claude's reading of "too near to each other" (examples.md) - a version, not built as a rule
SEED = 20260928 + 4      # the question's example: a fixed shuffle, like T3's rounds (seed + 1 .. + 3 used there)
QUESTION_ID = "L6.2-Q1-lows-too-near"


def _find_examples():
    if str(IDEA) not in sys.path:
        sys.path.insert(0, str(IDEA))
    import find_examples as FE        # noqa: E402 - T3's walk (break, first gap, first visit) + its settled-rule check
    return FE


# ------------------------------------------------------------------------------------------ his answers (examples.md)
@dataclass
class Example:
    k: int
    rnd: int
    entry: pd.Timestamp               # server time
    direction: str
    taken_t1: pd.Timestamp
    taken_t2: pd.Timestamp
    taken_px: float
    target_t1: pd.Timestamp
    target_t2: pd.Timestamp
    target_px: float
    near: float
    hedge: float
    answer: str                       # "yes" / "no" / "" (not answered)
    reason: str
    notes: dict = field(default_factory=dict)


_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d\d \w\w\w \d{4} \d\d:\d\d)\s*\|\s*(buy|sell)\s*\|(.*)$")
_PAIR = re.compile(r"(\d\d \w\w\w \d\d:\d\d) \+ (\d\d \w\w\w \d\d:\d\d) at ([\d.]+)")


def _server(ist: pd.Timestamp) -> pd.Timestamp:
    return utc_to_server(ist.tz_localize("Asia/Kolkata"))


def _stamp(s: str, entry_ist: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(f"{s[:6]} {entry_ist.year} {s[7:]}")
    if t > entry_ist:
        t = t - pd.DateOffset(years=1)
    return _server(t)


def read_examples() -> list[Example]:
    """Every numbered example row of examples.md, with his answer. Round = 1 (#1-#15), 2 (#16-#23), 3 (#24-#38)."""
    out = []
    for line in (IDEA / "examples.md").read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        k, ist = int(m.group(1)), pd.Timestamp(m.group(2))
        cells = [c.strip() for c in m.group(4).split("|")]
        pairs = [_PAIR.search(c) for c in cells]
        at = [i for i, p in enumerate(pairs) if p]
        tk, tg = pairs[at[0]], pairs[at[1]]
        near, hedge = float(cells[at[1] + 1]), float(cells[at[1] + 2])
        answer = cells[at[1] + 4].strip("* ").lower()
        answer = "yes" if answer.startswith("yes") else "no" if answer.startswith("no") else ""
        out.append(Example(k, 1 if k <= 15 else 2 if k <= 23 else 3, _server(ist), m.group(3),
                           _stamp(tk.group(1), ist), _stamp(tk.group(2), ist), float(tk.group(3)),
                           _stamp(tg.group(1), ist), _stamp(tg.group(2), ist), float(tg.group(3)),
                           near, hedge, answer, cells[at[1] + 5].strip("* ")))
    return out


# ------------------------------------------------------------------------------ the setups, with the new blocks
@dataclass
class Setup:
    direction: str
    taken: pd.Series                  # the taken level's "born" book row
    target: pd.Series | None          # the nearest alive equal level on the other side (steps.md step 2)
    target_recent: pd.Series | None   # what recent_first (#130) would pick instead
    sweep: int
    fresh: bool
    walk: tuple | None                # find_examples._walk: (brk, blk, gap_first, gap_third, lo, hi, near, entry, ext)

    @property
    def entry(self) -> int:
        return -1 if self.walk is None else int(self.walk[7])


@dataclass
class Data:
    m5: pd.DataFrame
    ruler: np.ndarray
    book: pd.DataFrame
    sweeps: pd.DataFrame
    life: pd.DataFrame


def load(min_apart: int | None = None, bars: tuple | None = None) -> Data:
    m5, h1, d1 = bars or (load_bars("5m", *YEARS), load_bars("1h", *YEARS), load_bars("1d", *YEARS))
    ruler = ADR.adr_bars(m5, "5m")["ruler"].to_numpy(float)
    book = LQ.obvious_liquidity(m5, h1, d1, min_apart=min_apart)
    sw = LQ.sweeps(m5, book)
    return Data(m5, ruler, book, sw, LQ.lifetimes(m5, book, sw))


def setups(D: Data) -> list[Setup]:
    """Idea #1 (steps.md): an equal high (low) taken -> the equal lows (highs) on the other side, alive through the
    sweep candle, the NEAREST one (steps.md step 2) -> T3's walk. One setup per level taken (fresh or not)."""
    FE = _find_examples()
    m5 = D.m5
    o, h, lo, c = (m5[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    ob_age = int(settings.get("order_block_max_age_bars"))
    timeout = int(settings.get("state_timeout_bars"))
    after = int(rules.get("gap_must_belong_to_the_breaking_move"))
    blocks = ST._block_arrays(o, h, lo, c)
    cur = {d: ST._current(n, *blocks[d]) for d in ("sell", "buy")}
    gaps = FVG.gaps(m5, "5m")
    gi = {d: g.assign(first=m5.index.get_indexer(pd.DatetimeIndex(g["first_bar"])),
                      third=m5.index.get_indexer(pd.DatetimeIndex(g["bar"]))).sort_values("third")
          for d, g in gaps.groupby("side")}
    born = D.book[D.book["event"] == "born"].drop_duplicates("level_id").set_index("level_id", drop=False)
    eq = D.life[D.life["kind"].isin(["equal_high", "equal_low"])]
    starts, ends, sides, prices = (eq[x].to_numpy() for x in ("start", "end", "side", "price"))
    out = []
    for r in D.sweeps[D.sweeps["kind"].isin(["equal_high", "equal_low"])].itertuples():
        s = int(m5.index.get_loc(r.bar))
        d = "sell" if r.side == "high" else "buy"
        other = "low" if d == "sell" else "high"
        m = (starts <= s) & (s < ends) & (sides == other) & ((prices < r.price) if d == "sell" else (prices > r.price))
        cand = eq[m]
        if not len(cand):
            out.append(Setup(d, born.loc[r.level_id], None, None, s, bool(r.fresh), None))
            continue
        tgt = cand.iloc[int(np.argmax(cand["price"].to_numpy()) if d == "sell" else np.argmin(cand["price"]))]
        rf = LQ.recent_first(LQ.alive(D.life, s)[lambda x: x["kind"] == f"equal_{other}"], other, r.price, max_n=1)
        walk = FE._walk(d, 1 if d == "sell" else -1, s, float(tgt["price"]), o, h, lo, c, cur[d], ob_age, timeout,
                        after, gi[d])
        out.append(Setup(d, born.loc[r.level_id], tgt, rf.iloc[0] if len(rf) else None, s, bool(r.fresh), walk))
    return out


# ------------------------------------------------------------------------------------------------ judging
def touches(level: pd.Series, book: pd.DataFrame) -> list[pd.Timestamp]:
    rows = book[book["level_id"] == level["level_id"]]
    return sorted({pd.Timestamp(level["first_bar"])} | set(pd.DatetimeIndex(rows["touch_bar"].dropna())))


def _same_level(level: pd.Series, t1: pd.Timestamp, t2: pd.Timestamp, px: float, book: pd.DataFrame) -> bool:
    """The pair he saw (touch 1, touch 2, price) is this level: within his 'really equal' ($0.06, rules.py) of its
    price, and one of his two touches is a touch of this level. (T3's rough search made a level for EVERY valid first
    touch, so it could draw a pair whose 1st touch is, in the new block, only part of an older level's top - #33.)"""
    if abs(level["price"] - px) > float(rules.get("what_equal_highs_and_lows_are")["tol_usd"]) + 1e-9:
        return False
    t = touches(level, book)
    return t1 in t or t2 in t


def judge(ex: Example, found: list[Setup], D: Data) -> dict:
    """fires = the machine makes this trade: same direction, same taken level (fresh), same entry candle.
    same_target = and it aims at the same pair he saw. A YES is FOUND when both hold; a NO is missed when it fires."""
    i = int(D.m5.index.get_indexer([ex.entry])[0])
    z = next((z for z in found if z.direction == ex.direction
              and _same_level(z.taken, ex.taken_t1, ex.taken_t2, ex.taken_px, D.book)), None)
    if z is None:
        return dict(fires=False, same_target=False, why="the pair it took is not an equal level here", setup=None)
    if not z.fresh:
        return dict(fires=False, same_target=False, why="that price was already swept (#129)", setup=z)
    if z.target is None:
        return dict(fires=False, same_target=False, why="no equal level on the other side", setup=z)
    if z.walk is None or z.entry != i:
        return dict(fires=False, same_target=False, why="no entry at that candle", setup=z)
    same = _same_level(z.target, ex.target_t1, ex.target_t2, ex.target_px, D.book)
    return dict(fires=True, same_target=same, why="takes it" if same else "takes it, aiming at another pair", setup=z)


def run_past(z: Setup, D: Data) -> float:
    """His "it moves so far" (examples.md): how far the grab ran past the level it took before the entry, as a share
    of the day's range at the entry (Claude's measure; L7b builds the line)."""
    return abs(float(z.walk[8]) - float(z.taken["price"])) / D.ruler[z.entry]


def apart(level: pd.Series, book: pd.DataFrame, m5: pd.DataFrame) -> int:
    """5-minute candles between a level's first touch and the touch that made it a pair."""
    t = touches(level, book)
    return int(m5.index.get_loc(t[1]) - m5.index.get_loc(t[0])) if len(t) > 1 else 0


# ------------------------------------------------------------------------------------------------ the run
def run(write: bool = True) -> dict:
    ex = read_examples()
    D = load()
    found = setups(D)
    D7 = load(NEAR_LINE, (D.m5, load_bars("1h", *YEARS), load_bars("1d", *YEARS)))
    found7 = setups(D7)
    for e in ex:
        j = judge(e, found, D)
        j7 = judge(e, found7, D7)
        e.notes = dict(j, near7=j7, run=abs(e.hedge - e.taken_px) / D.ruler[D.m5.index.get_loc(e.entry)],
                       taken_apart=int(D.m5.index.get_loc(e.taken_t2) - D.m5.index.get_loc(e.taken_t1)),
                       target_apart=int(D.m5.index.get_loc(e.target_t2) - D.m5.index.get_loc(e.target_t1)))
        z = j["setup"]
        e.notes["recent_first_same"] = (None if z is None or z.target is None or z.target_recent is None
                                        else z.target_recent["level_id"] == z.target["level_id"])
    entered = [z for z in found if z.walk is not None and z.fresh]
    R = dict(examples=ex, D=D, found=found, entered=entered,
             counts=dict(entered=len(entered), entered_near7=sum(z.walk is not None and z.fresh for z in found7),
                         equal_sweeps=int(D.sweeps["kind"].str.startswith("equal").sum()),
                         not_fresh=int((~D.sweeps.loc[D.sweeps["kind"].str.startswith("equal"), "fresh"]).sum()),
                         not_fresh_entries=sum(z.walk is not None and not z.fresh for z in found),
                         recent_differs=sum(z.target_recent is not None and z.target_recent["level_id"]
                                            != z.target["level_id"] for z in entered)))
    R["readings"] = readings(ex)
    print_table(R)
    if write:
        MISSES.mkdir(parents=True, exist_ok=True)
        QUEUE.mkdir(parents=True, exist_ok=True)
        R["question"] = draw_question(R)
        R["miss_pictures"] = draw_misses(R)
        write_readme(R)
        write_queue(R)
    return R


def readings(ex: list[Example]) -> list[dict]:
    """How many of his answers each reading gets right. The two lines were read off these same answers (round 3), so
    getting round 3 right by them proves nothing new - rounds 1-2 are the only answers they were not drawn from."""
    out = []
    for name, fires in (
            ("the L6.2 blocks, his written rules", lambda e: e.notes["fires"]),
            (f"+ 'too near' = touches at least {NEAR_LINE} candles apart (a version)", lambda e: e.notes["near7"]["fires"]),
            (f"+ 'ran too far' = at most {RUN_LINE} of a day past the level (L7b's line)",
             lambda e: e.notes["fires"] and e.notes["run"] <= RUN_LINE),
            ("+ both", lambda e: e.notes["near7"]["fires"] and e.notes["run"] <= RUN_LINE)):
        row = {"reading": name}
        for tag, rnds in (("r3", (3,)), ("r12", (1, 2))):
            ys = [e for e in ex if e.rnd in rnds and e.answer == "yes"]
            ns = [e for e in ex if e.rnd in rnds and e.answer == "no"]
            row[f"{tag}_yes"] = (sum(fires(e) for e in ys), len(ys))
            row[f"{tag}_no"] = (sum(fires(e) for e in ns), len(ns))
        out.append(row)
    return out


def print_table(R: dict) -> None:
    c = R["counts"]
    print(f"idea #1 with the new blocks, 2016-2020: {c['entered']:,} entries (not counting {c['not_fresh_entries']} "
          f"whose level's price was already swept - #129); equal-level sweeps {c['equal_sweeps']:,}, "
          f"{c['not_fresh']} not fresh; recent-first picks another target on {c['recent_differs']}; "
          f"with 'too near' = {NEAR_LINE}: {c['entered_near7']:,} entries")
    print(f"{'#':>3} {'rnd':>3} {'dir':4} {'his':4} {'new blocks':36} {'run':>5} {'taken':>5} {'target':>6}  near7")
    for e in R["examples"]:
        n = e.notes
        print(f"{e.k:>3} {e.rnd:>3} {e.direction:4} {e.answer or '-':4} {n['why'][:36]:36} {n['run']:5.2f} "
              f"{n['taken_apart']:>5} {n['target_apart']:>6}  {n['near7']['why']}")
    for r in R["readings"]:
        print(f"{r['reading'][:70]:70} r3 yes {r['r3_yes'][0]}/{r['r3_yes'][1]} no fired {r['r3_no'][0]}/"
              f"{r['r3_no'][1]} | r1-2 yes {r['r12_yes'][0]}/{r['r12_yes'][1]} no fired {r['r12_no'][0]}/"
              f"{r['r12_no'][1]}")


# ------------------------------------------------------------------------------------------------ pictures
def _ist(t) -> str:
    return to_ist(pd.Timestamp(t)).strftime("%d %b %Y, %H:%M")


def _pair(level: pd.Series, book: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    t = touches(level, book)
    return t[0], t[1] if len(t) > 1 else t[0]


def _picture(z: Setup, D: Data, *, subject: str, heading: str, sub: str, ask: str, how: tuple[str, ...]):
    """One 5-minute chart with the whole story, ending at the entry candle (his one-chart rule, inbox 1130): the pair
    price took (purple 1, 2), the pair it should go to (orange 1, 2), the first gap, S1 / B1."""
    from machine.charts import draw as DR
    m5, I = D.m5, D.m5.index
    sell = z.direction == "sell"
    tk_word, tg_word = ("equal highs", "equal lows") if sell else ("equal lows", "equal highs")
    tag = "S1" if sell else "B1"
    a1, a2 = _pair(z.taken, D.book)
    b1, b2 = _pair(z.target, D.book)
    entry = I[z.entry]
    s0 = max(0, min(I.get_loc(a1), I.get_loc(b1)) - 10)
    brk, blk, gfirst, gthird, glo, ghi, near, _, _ = z.walk
    items = [DR.Candles(),
             DR.Line((a1, z.taken["price"]), (a2, z.taken["price"]), touches=(a1, a2), name=f"{tk_word} · taken",
                     extend_to=I[z.sweep - 1]),
             DR.Line((b1, z.target["price"]), (b2, z.target["price"]), touches=(b1, b2),
                     name=f"{tg_word} · goes here", extend_to=entry, color=DR.FIBC),
             DR.Box(I[int(gfirst)], entry, float(glo), float(ghi)),
             DR.Marker(entry, float(near), tag, z.direction, "gap")]
    return DR.Picture(
        kind="question", subject=subject, heading=heading, heading_color=DR.side_colour(z.direction), sub=sub,
        rows=[[DR.Panel(m5.iloc[s0:z.entry + 1], "5m", items, f"the whole story · ends at {tag}", role="main")]],
        question=DR.Question(ask, how), footer_left=f"{tag} = the {'sell' if sell else 'buy'} · 1, 2 = the two "
                                                    "touches of a pair",
        footer_right="the chart ends at the entry candle", entry_at=entry, entry_tf="5m")


def _render(pic, path) -> list:
    from machine.charts import draw as DR
    from machine.charts import lint
    return lint.check(DR.render(pic, path))


Q_ASK = "Would you take this sell for idea #1?"
Q_HOW = ("Look at the equal lows it should go to (orange 1, 2): only {mins} minutes apart.",
         "Price ran only a little past the equal highs (purple 1, 2) before the break.",
         "Answer YES, or NO and a few words why.")
MISS_ASK = "Are two lows only 15 to 30 minutes apart still equal lows that price goes to?"


def pick_question_setup(R: dict) -> list[Setup]:
    """The question's example, by a fixed rule written BEFORE looking (never by result; nothing after the entry is
    read): a SELL whose target equal lows are 3-5 candles apart (his "too near": his no's were 3-6), whose grab ran
    less than 0.10 of a day past the equal highs (like his yes #24-#28) and whose taken highs are 7+ candles apart (so
    only ONE thing is "too near"), with a move toward the target before the entry under 0.10 of a day (as round 3), the
    whole story on one chart (140 candles), no weekend folded into the chart, keeping every settled rule T3 checked,
    and not one of the 38 he has seen. (Tried and dropped after looking: "a gap of at least 10 cents" - it left only
    pictures across a weekend, a second thing to look at; a tiny gap is his own rule, "any size" #62, #131, he took
    4-cent gaps in #25 and #30, and the picture prints the gap's size.)
    Shuffled with SEED; the first whose picture passes every picture rule is used."""
    FE = _find_examples()
    D = R["D"]
    m5 = D.m5
    adr = ADR.adr_bars(m5, "5m")
    ath = np.maximum.accumulate(np.maximum(m5["high"].to_numpy(float), FE.OLD_ALL_TIME_HIGH))
    seen = {e.entry for e in R["examples"]}
    h, lo = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
    pool = []
    for z in R["entered"]:
        if z.direction != "sell" or m5.index[z.entry] in seen:
            continue
        if not 3 <= apart(z.target, D.book, m5) <= 5 or apart(z.taken, D.book, m5) < 7 or run_past(z, D) >= 0.10:
            continue
        pre = (float(z.walk[6]) - lo[int(z.walk[0]):z.entry].min()) / D.ruler[z.entry]
        if pre >= 0.10:
            continue
        first = min(m5.index.get_loc(_pair(z.taken, D.book)[0]), m5.index.get_loc(_pair(z.target, D.book)[0]))
        if z.entry - first > 140:
            continue
        if (np.diff(m5.index[max(0, first - 10):z.entry + 1]) > pd.Timedelta(hours=3)).any():
            continue                                           # no weekend (or holiday) folded into the one chart
        a1, a2 = _pair(z.taken, D.book)
        b1, b2 = _pair(z.target, D.book)
        rough = FE.Setup("sell", FE.Level("high", float(z.taken["price"]), m5.index.get_loc(a1),
                                          m5.index.get_loc(a2), 0, z.sweep),
                         FE.Level("low", float(z.target["price"]), m5.index.get_loc(b1), m5.index.get_loc(b2), 0, -1),
                         z.sweep, *[int(x) if k < 4 or k == 7 else float(x) for k, x in enumerate(z.walk[:8])],
                         float(z.walk[8]))
        if FE.allowed(rough, m5, adr, ath):
            continue
        pool.append(z)
    rng = np.random.default_rng(SEED)
    return [pool[k] for k in rng.permutation(len(pool))]


def draw_question(R: dict) -> dict | None:
    D = R["D"]
    for z in pick_question_setup(R):
        mins = 5 * apart(z.target, D.book, D.m5)
        path = QUEUE / f"{QUESTION_ID}.jpg"
        pic = _picture(z, D, subject=f"L6.2 question 1 (lows too near) · sell {D.m5.index[z.entry]:%Y-%m-%d %H:%M} server",
                       heading="IDEA #1 · ONE QUESTION · SELL?",
                       sub=f"{_ist(D.m5.index[z.entry])} India time · found by the machine",
                       ask=Q_ASK, how=tuple(h.format(mins=mins) for h in Q_HOW))
        problems = _render(pic, path)
        if not problems:
            return dict(setup=z, path=path, mins=mins, run=run_past(z, D), taken_apart=apart(z.taken, D.book, D.m5),
                        ask=Q_ASK, how=tuple(h.format(mins=mins) for h in Q_HOW))
        path.unlink(missing_ok=True)
    return None


def draw_misses(R: dict) -> dict:
    """One picture per miss (a YES not found, a NO that fires), ending at its entry. These are records for the
    builders: he has seen each setup already, so they are NOT sent again - the one question they raise is queued on
    a new picture (draw_question)."""
    D = R["D"]
    out = {}
    for e in R["examples"]:
        n = e.notes
        miss = (e.answer == "yes" and not (n["fires"] and n["same_target"])) or (e.answer == "no" and n["fires"])
        if not miss or n["setup"] is None or n["setup"].walk is None:
            continue
        z = n["setup"]
        lows = n["target_apart" if e.direction == "sell" else "taken_apart"]
        path = MISSES / f"{e.k:02d}-{to_ist(e.entry):%Y-%m-%d}-{e.direction}.jpg"
        pic = _picture(z, D, subject=f"L6.2 miss · idea #1 example {e.k} (record, not sent)",
                       heading=f"L6.2 MISS · EXAMPLE {e.k} · {e.direction.upper()} · YOU SAID {e.answer.upper()}",
                       sub=f"{_ist(e.entry)} India time · the new blocks still take it\n"
                           f"the grab ran {run_past(z, D):.2f} of a day past the level · the two lows are "
                           f"{5 * lows} minutes apart",
                       ask=MISS_ASK,
                       how=("A record for the builders - not sent again (he answered this setup in T3).",
                            f"The question is queued once, on a new picture: {QUESTION_ID}."))
        out[e.k] = (path, _render(pic, path))
    return out


# ------------------------------------------------------------------------------------------------ writing
def _cell(x) -> str:
    return f"{x[0]} of {x[1]}"


def write_readme(R: dict) -> None:
    c = R["counts"]
    ex = R["examples"]
    q = R.get("question")
    L = ["# L6.2 — idea #1's examples against the new liquidity blocks (found / missed)", "",
         "> Made by `python -m tests.idea001_acceptance` (L6.2 acceptance). His answers: "
         "`plan/work/idea-001/examples.md` (T3). **Round 3 is counted** (one 5-minute chart each); rounds 1–2 are "
         "shown apart (three charts per picture confused him, inbox 2026-09-28-1130). 2016–2020 only. No money.", "",
         "The setups are idea #1's 7 agreed steps with the NEW blocks for everything about liquidity: the equal "
         "highs / lows (5-minute, hour and day charts), the sweep, \"a level swept once is finished\" (#129), what is "
         "alive at the sweep. After the sweep it is T3's own walk (break, first gap, first visit), so every "
         f"difference comes from the liquidity blocks. **{c['entered']:,} entries 2016–2020** (T3's rough search: "
         f"1,106); {c['not_fresh_entries']} more setups are dropped because their level's price was already swept in "
         f"the two days before (#129; {c['not_fresh']} of {c['equal_sweeps']:,} equal-level sweeps).", "",
         "## The answer", "",
         "| his examples | found by the new blocks | what that means |", "|---|---|---|"]
    r0 = R["readings"][0]
    L += [f"| round 3 — his **yes** | **{_cell(r0['r3_yes'])}** | every yes is found: the same level taken, the "
          "same pair to go to, the same entry candle |",
          f"| round 3 — his **no** | **{_cell(r0['r3_no'])} still taken** | his reasons are not liquidity rules yet "
          "(below) — these are the misses |",
          f"| rounds 1–2 (not counted) — yes | {_cell(r0['r12_yes'])} | |",
          f"| rounds 1–2 (not counted) — no | {_cell(r0['r12_no'])} still taken | #3 \"too big\", #16 \"too far it "
          "moved\" |", ""]
    L += ["## Every example", "",
          "run = how far the grab ran past the level it took before the entry, as a share of the day's range (his "
          "\"it moves so far\"); apart = 5-minute candles between the two touches of a pair (his \"too near to each "
          "other\"). Both are Claude's measures of his words.", "",
          "| # | round | buy / sell | his answer | his reason | new blocks | run | taken pair apart | target pair "
          "apart | with 'too near' = 7 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in ex:
        n = e.notes
        L.append(f"| {e.k} | {e.rnd} | {e.direction} | {e.answer or '—'} | {e.reason or '—'} | {n['why']} | "
                 f"{n['run']:.2f} | {n['taken_apart']} | {n['target_apart']} | {n['near7']['why']} |")
    L += ["", "## His two no-reasons, as readings (no money — only which examples each reading takes)", "",
          "| reading | round 3: yes found | round 3: no taken | rounds 1–2: yes found | rounds 1–2: no taken |",
          "|---|---|---|---|---|"]
    for r in R["readings"]:
        L.append(f"| {r['reading']} | {_cell(r['r3_yes'])} | {_cell(r['r3_no'])} | {_cell(r['r12_yes'])} | "
                 f"{_cell(r['r12_no'])} |")
    L += ["", "**How many tries:** 4 readings on the same 30 answers. The two lines were read off round 3 itself "
          "(examples.md), so round 3 cannot test them; rounds 1–2 are the only answers they were not drawn from — "
          "and he said he may have answered those wrongly.", "",
          "What the table says (Claude's reading of it — the rule is his to make):",
          "1. **\"Too near\" is not what makes his no.** As a rule for every pair it loses 3 of his 10 round-3 yes "
          "(#24 highs 3 candles apart, #25 highs 6, #33 lows 4 — \"yes, partly no\") and still takes 4 of the 5 "
          "no's, only aiming at a farther pair. It is kept as a version (`settings.equal_levels_min_apart_bars`, "
          "default 0 = his written \"any time apart\").",
          f"2. **\"Ran too far\" separates round 3 completely** (yes ≤ 0.22, no ≥ 0.27) — but it was drawn from "
          "round 3. On rounds 1–2 it also drops two of his yes: #12 (0.27) and #15 (0.36). It is not a liquidity "
          "rule: L7b builds it as a do-not-enter setting and tests 0.20 / 0.25 / 0.30.",
          "3. **What is still open:** in round 3 every \"lows too near\" came with a big run, except #33 (small run, "
          "the taken lows too near: \"yes, partly no\"). No example has a small run with the TARGET lows too near. "
          "That one picture tells the two readings apart → the question below.", ""]
    L += ["## The misses (each drawn, ending at its entry)", "",
          "Every miss is a NO the new blocks still take. He has seen each of these setups, so the pictures are "
          "records for the builders and are not sent again (his rule: never re-show what he answered). They all "
          "raise the same open point, so **one** question is queued, on a new picture that isolates it.", "",
          "| # | round | his words | picture | why it is still taken | its question |", "|---|---|---|---|---|---|"]
    for k, (path, problems) in sorted(R["miss_pictures"].items()):
        e = next(x for x in ex if x.k == k)
        lint_note = "" if not problems else f" (lint: {len(problems)} problems)"
        lows = "near" in e.reason or "same reason" in e.reason
        ask = (f"`{QUESTION_ID}` for \"lows too near\"; \"moves so far\" → L7b's line, no question" if lows else
               "none — his words say it (the grab ran too far) → L7b's line")
        L.append(f"| {k} | {e.rnd} | \"{e.reason.strip(chr(34))}\" | `{path.name}`{lint_note} | nothing in the "
                 f"liquidity blocks stops it: its pairs are real equal levels by his rule; ran {e.notes['run']:.2f} "
                 f"of a day past the level | {ask} |")
    L += ["", "## The question queued (weekly batch, L9b)", ""]
    if q:
        z = q["setup"]
        L += [f"`plan/work/question-queue/{QUESTION_ID}.md` + `.jpg` — a sell on "
              f"{_ist(R['D'].m5.index[z.entry])} India time, never shown before: the target equal lows "
              f"{q['mins']} minutes apart, the grab ran {q['run']:.2f} of a day past the equal highs (small), the taken "
              f"highs {q['taken_apart']} candles apart. On the picture: *\"{q['ask']} {q['how'][0]}\"*", ""]
    else:
        L += ["No example passed every picture rule — the question is not queued yet.", ""]
    differ = [e for e in ex if e.notes.get("recent_first_same") is False]
    L += ["## Also found (for L7b — not questions)", "",
          f"- **The target: nearest or most recent?** steps.md step 2 (his \"Ok\") says the nearest; #130 says the "
          f"most recent first. On his examples they pick the same pair everywhere except "
          + ", ".join(f"#{e.k} ({'his ' + e.answer if e.answer else 'not answered'})" for e in differ)
          + f"; over all {c['entered']:,} entries they differ on {c['recent_differs']}. The idea's own target choice "
          "→ L7b runs both as versions.",
          "- **One level per first touch:** the new block joins a later touch to the OLDEST level it fits (its price "
          "is the one the stops sit behind). T3's rough search made a level for every first touch, so on #33 it drew "
          "the target as 12:35 + 15:00 at 1321.89 where the new block has 11:45 + 15:00 at 1321.92 — the same "
          "stops, 3 cents apart. Counted as the same pair.",
          "- **\"The same level\" (#129) is the same KIND at the same price ($0.06).** The first build used any kind "
          "at 0.05 of a day (about $0.50): it called 48% of equal-level sweeps \"already swept\", and any kind killed "
          "his yes #31 (equal lows formed right at yesterday's London low price, just poked). The old engine kept a "
          "list per kind; his #31 yes keeps it that way.", ""]
    (MISSES / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_queue(R: dict) -> None:
    q = R.get("question")
    readme = QUEUE / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Questions waiting for the weekly batch\n\n"
            "> Each question is one `.md` + its picture. The question loop (L9b) sends at most `QUESTION_BUDGET` (10) "
            "a week, in its priority order, and marks a picture shown only once it is in front of Tejas "
            "(`machine.charts.lint.mark_shown`). No money in any question (lesson B16). His answer goes to "
            "`strategy/learned-inbox/` word for word, then ONE confirm picture before it is locked (T5).\n",
            encoding="utf-8")
    if not q:
        return
    D = R["D"]
    z = q["setup"]
    c = R["counts"]
    L = [f"# {QUESTION_ID}", "",
         "| field | value |", "|---|---|",
         "| status | waiting for the weekly batch |",
         "| made by | L6.2 (28 Sep 2026), from the misses in `plan/work/L6-misses/` |",
         "| idea | idea #1 — equal highs / lows are a magnet |",
         "| priority (L9b) | (2) the idea currently running |",
         f"| picture | `{QUESTION_ID}.jpg` — one 5-minute chart, ends at the entry, lint clean, never shown |",
         f"| moment | sell, {_ist(D.m5.index[z.entry])} India time ({D.m5.index[z.entry]} server) |", "",
         "## On the picture", "", f"**{q['ask']}**", "", *[f"- {h}" for h in q["how"]], "",
         "## Why ask (the #129 search first)", "",
         "I searched strategy.md §1, strategy-code.md §5, LEARNED #88–#90, strategy-understanding §66h–§66j and his "
         "T3 answers for how far apart the two touches of equal lows must be, and found nothing about pairs CLOSE in "
         "time — the closest lines are strategy.md §1 \"Any time apart — two days or a week\" and #89 \"there is no "
         "reason that it should be in a very short duration\" (both about pairs far apart). His T3 words \"the lows "
         "are not proper lows cause they are too near to each other\" (#33–#38) always came with a big run past the "
         "level, except #33 (the TAKEN lows 20 minutes apart, small run: \"yes, partly no\").", "",
         "## The two readings, and what each would change (no money)", "",
         "| his answer | reading | what the machine does |", "|---|---|---|",
         "| NO (\"too near\") | two touches of equal lows only 15–25 minutes apart are ONE low, not a pair | equal "
         "levels need about 7 candles between touches (`settings.equal_levels_min_apart_bars` becomes his rule in "
         f"rules.py); idea #1 entries 2016–2020: {c['entered']:,} → {c['entered_near7']:,}; and ask whether highs are "
         "the same (his #24 and #25 highs, 3 and 6 candles apart, were yes) |",
         "| YES | \"too near\" is a weaker pair, not a no; his no's came from the run past the level | nothing changes "
         "in the equal-levels block; L7b's run line decides |", "",
         "The misses it explains: " + ", ".join(f"#{k}" for k in sorted(R["miss_pictures"])) + ".", ""]
    (QUEUE / f"{QUESTION_ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
