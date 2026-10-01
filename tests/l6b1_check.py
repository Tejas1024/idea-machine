"""L6b.1 side by side: his marked trendline charts (strategy/pictures/, via their LEARNED entries) next to what the
new blocks draw on the SAME candles. 2016-2020 only (the tune years; his 2021-2023 charts are the hidden years and are
not read - like L6b.3). No money.

    .venv\\Scripts\\python -m tests.l6b1_check        writes plan/work/L6b-check/*.jpg and README.md

Left = his picture (a crop of the panel he answered about). Right = the machine's chart of the same stretch, India time:
its 2-touch line (grey, not liquidity - #92) or trendline (purple, 3 touches), and for #69 the setup and what the rule
"a line in the path" does with it. His words and the machine's reading are written under each.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from machine import config, settings
from machine.blocks import patterns as PT
from machine.blocks import trendlines as TL
from machine.charts import draw as D
from machine.charts import lint
from machine.data.bars import load_bars
from machine.ideas import compile as C
from machine.ideas.form import Idea

OUT = config.WORK_DIR / "L6b-check"
PICS = config.ROOT / "strategy" / "pictures"
GREY = "#8a939c"
Q_TOP_LEFT, Q_TOP_RIGHT, Q_BOTTOM_LEFT = (0, 85, 650, 610), (650, 85, 1300, 610), (0, 695, 650, 1220)


def ist_index(bars: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(bars["utc"]).tz_convert("Asia/Kolkata").tz_localize(None)


def at_ist(bars: pd.DataFrame, day: str, hhmm: str) -> pd.Timestamp:
    """The server time of the 5-minute candle that opens at `hhmm` India time on `day` (his pictures' clock)."""
    k = np.flatnonzero(ist_index(bars) == pd.Timestamp(f"{day} {hhmm}"))
    if not len(k):
        raise KeyError(f"no candle at {day} {hhmm} India time")
    return bars.index[int(k[0])]


@dataclass
class Case:
    key: str
    lesson: str
    picture: str
    crop: tuple
    day: str
    window: tuple[str, str]           # India time on `day`
    side: str                         # the side of his line: high / low
    a: str                            # his line's touches, India time
    b: str
    his: str                          # his answer, his words
    tf: str = "5m"
    notes: list = field(default_factory=list)


CASES = [
    Case("01-Q92A-2016-11-09", "LEARNED #92", "Q92-B3-trendline-touches.png", Q_TOP_LEFT, "2016-11-09",
         ("11:35", "13:55"), "high", "12:05", "12:50",
         "NO - \"the three touches should be there. The third touch itself is going above the trend line\""),
    Case("02-Q92C-2020-01-08", "LEARNED #92", "Q92-B3-trendline-touches.png", Q_BOTTOM_LEFT, "2020-01-08",
         ("19:30", "22:05"), "high", "20:00", "20:50",
         "NO - \"it only touched two touches correctly, third touch is not respecting\""),
    Case("03-Q94A-2016-09-08", "LEARNED #94", "Q94-B3b-touch-through.png", Q_TOP_LEFT, "2016-09-08",
         ("17:45", "21:20"), "high", "18:15", "19:05",
         "NO - \"if the third touch is go beyond the ... trend line, even now slightest, it is not considered as a "
         "trend line\""),
    Case("04-Q68A-2017-02-24", "LEARNED #69", "Q68-D5-trendlines.png", Q_TOP_LEFT, "2017-02-24",
         ("20:05", "23:15"), "low", "20:15", "21:35",
         "\"there is a trend line so this is like 50 50 ... if day and hour both are bullish, then I would avoid this "
         "trade\""),
    Case("05-Q68B-2017-12-27", "LEARNED #69", "Q68-D5-trendlines.png", Q_TOP_RIGHT, "2017-12-27",
         ("18:10", "21:55"), "low", "18:20", "20:30",
         "TAKE - \"if market is bullish in higher time frames ... I would exit at the trend line itself\""),
]


def machine_lines(bars: pd.DataFrame, lines: pd.DataFrame, c: Case, lo: pd.Timestamp, hi: pd.Timestamp):
    """Items for the machine's chart and one sentence on his line: the machine's line through his two touches (grey =
    2 touches, purple = a trendline), and every other trendline of the machine inside the window."""
    ta, tb = at_ist(bars, c.day, c.a), at_ist(bars, c.day, c.b)
    mine = lines[(lines["side"] == c.side) & (lines["a_bar"] == ta) & (lines["b_bar"] == tb)]
    items, said = [], ""
    if len(mine):
        r = mine.sort_values("touches").iloc[-1]
        if r["touches"] == 3:
            items.append(D.Line((ta, r["a_price"]), (r["c_bar"], r["c_price"]), touches=(ta, tb, r["c_bar"]),
                                name="trendline", extend_to=hi))
            said = "the machine: a TRENDLINE (3 touches)"
        else:
            items.append(D.Line((ta, r["a_price"]), (tb, r["b_price"]), touches=(ta, tb), name="2 touches only",
                                extend_to=hi, color=GREY))
            said = "the machine: 2 touches only - not a trendline, not liquidity (#92)"
    else:
        said = "the machine does not draw a line through these two touches"
    others = lines[(lines["touches"] == 3) & (lines["c_bar"] >= lo) & (lines["a_bar"] >= lo - pd.Timedelta(hours=6))
                   & (lines["c_bar"] <= hi) & ~((lines["a_bar"] == ta) & (lines["b_bar"] == tb))]
    for r in others.itertuples():
        items.append(D.Line((r.a_bar, r.a_price), (r.c_bar, r.c_price), touches=(r.a_bar, r.b_bar, r.c_bar),
                            name="trendline", extend_to=hi))
    return items, said, len(mine) > 0, (int(mine["touches"].max()) if len(mine) else 0), len(others)


def machine_picture(c: Case, path: Path) -> dict:
    d = pd.Timestamp(c.day)
    bars = load_bars("5m", d - pd.Timedelta(days=40), d + pd.Timedelta(days=1))
    lines = TL.lines(bars, "5m")
    lo, hi = at_ist(bars, c.day, c.window[0]), at_ist(bars, c.day, c.window[1])
    items, said, found, touches, others = machine_lines(bars, lines, c, lo, hi)
    extra, sub2 = [], ""
    if c.lesson == "LEARNED #69":
        extra, sub2 = path_rule(c, bars, lo, hi)
    win = bars.loc[lo:hi]
    panel = D.Panel(win, "5m", [D.Candles(), *items, *extra], title=f"the machine · {d:%d %b %Y}", clock="ist")
    sub = said + (f"\n{sub2}" if sub2 else "")
    pic = D.Picture("trade", f"L6b.1 check {c.key}", f"THE MACHINE · {c.key[3:]}", [[panel]], sub=wrap(sub),
                    footer_left="purple = a trendline (3 touches) · grey = a line with 2 touches (not liquidity)",
                    footer_right="L6b.1")
    drawn = D.render(pic, path)
    return dict(found=found, touches=touches, others=others, said=said, sub2=sub2, lint=lint.check(drawn))


def path_rule(c: Case, bars: pd.DataFrame, lo, hi) -> tuple[list, str]:
    """#69: his Type 1 sell found by the compiler in the window, and what 'a line in the path' does with it."""
    d = pd.Timestamp(c.day)
    mk = C.Market.load(str((d - pd.Timedelta(days=400)).date()), str((d + pd.Timedelta(days=1)).date()))
    sys.path.insert(0, str(config.ROOT))
    from tests.test_ideas import type1
    out = C.compile_idea(Idea.load(type1(directions=["sell"], do_not_enter=[])), mk)
    fp = settings.fingerprint(out.resolved)
    sig = [s for s in out.signals if lo <= s.known_at - pd.Timedelta(minutes=5) <= hi]
    if not sig:
        return [], "his Type 1 as the machine builds it makes no sell here (so the rule has nothing to read)"
    s = sig[-1]
    i = mk.closed_by("5m", s.known_at)
    near = max(s.targets)
    got = TL.in_path(mk.lines5(fp), i, s.entry_price, near, "sell")
    hour, day = (mk.bias_at(tf, fp, s.known_at) for tf in ("1h", "1d"))
    words = {1: "up", -1: "down", 0: "neutral"}
    items = [D.Level(s.entry_price, "sell entry", start=s.known_at - pd.Timedelta(minutes=5)),
             D.Level(s.stop, "hedge", start=s.known_at - pd.Timedelta(minutes=5), color=D.RED),
             D.Level(near, "target", start=s.known_at - pd.Timedelta(minutes=5), color=D.GREEN)]
    if got is None:
        return items, "no line stands between this sell and its target"
    reward = abs(got[1] - s.entry_price) / abs(s.entry_price - s.stop)
    both = hour == 1 and day == 1
    what = ("the line becomes the target" if reward >= float(settings.get("line_in_path_min_reward"))
            else "no trade (50-50)") if both else "nothing changes"
    return items, (f"the line is {reward:.1f} hedge distances below the entry · hour {words[hour]}, day {words[day]}"
                   f" -> {what} (#69: it matters only when the hour AND the day are both against)")


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def side_by_side(his: Path, crop: tuple, machine: Path, out: Path, top: str, bottom: str) -> None:
    right = Image.open(machine).convert("RGB")
    left = Image.open(his).convert("RGB").crop(crop)
    h = right.height
    left = left.resize((int(left.width * h / left.height), h))
    pad = 80
    im = Image.new("RGB", (left.width + right.width + 30, h + 2 * pad + 20), "white")
    im.paste(left, (0, pad))
    im.paste(right, (left.width + 30, pad))
    dr = ImageDraw.Draw(im)
    dr.text((20, 20), top, fill="#1f2328", font=font(34))
    dr.text((20, h + pad + 20), bottom, fill="#1f2328", font=font(28))
    im.save(out, "JPEG", quality=90)


def daily_case() -> dict:
    """His Q69 panel D (#70): a daily chart to 31 Aug 2018 with the falling trendline he drew - "breakout of trend line
    and now it is at the retesting ... we can consider that as a bullish"."""
    d1 = load_bars("1d", "2018-01-01", "2018-08-31")
    L = TL.lines(d1, "1d")
    lo = pd.Timestamp("2018-06-01")
    L = L[L["b_bar"] >= lo]
    items = []
    for r in L.sort_values("touches").drop_duplicates("line_id", keep="last").itertuples():
        if r.side != "high":
            continue
        pts = (r.a_bar, r.b_bar) + ((r.c_bar,) if r.touches == 3 else ())
        items.append(D.Line((r.a_bar, r.a_price), (pts[-1], r.c_price if r.touches == 3 else r.b_price), touches=pts,
                            name="trendline" if r.touches == 3 else "2 touches only", extend_to=d1.index[-1],
                            color=D.PURPLE if r.touches == 3 else GREY))
    win = d1.loc[lo:]
    last = win.iloc[-1]
    above = [r for r in L.itertuples() if r.side == "high"
             and last["close"] > r.b_price + r.slope * (d1.index.get_loc(win.index[-1]) - d1.index.get_loc(r.b_bar))]
    panel = D.Panel(win, "1d", [D.Candles(), *items], title="the machine · daily chart to 31 Aug 2018", clock="server")
    sub = (f"the machine's falling lines on the daily chart: {sum(r.side == 'high' for r in L.itertuples())} "
           f"(3 touches: {int(((L['side'] == 'high') & (L['touches'] == 3)).sum())}); the last close is above "
           f"{len(above)} of them")
    pic = D.Picture("trade", "L6b.1 check Q69 D daily", "THE MACHINE · Q69 D · daily", [[panel]], sub=wrap(sub),
                    footer_left="purple = a trendline (3 touches) · grey = 2 touches", footer_right="L6b.1")
    path = OUT / "_m-06-Q69D.jpg"
    drawn = D.render(pic, path)
    return dict(path=path, sub=sub, lint=lint.check(drawn), lines=len(L))


TRIANGLE_DAYS = [
    ("07-T01-2018-02-28", "2018-02-28", ("18:30", "23:30"), "his-T01-2018-02-28.jpg"),
    ("08-T02-2019-10-09", "2019-10-09", ("04:00", "13:30"), "his-T02-2019-10-09.jpg"),
]


def triangle_case(key: str, day: str, window: tuple[str, str], his: str) -> dict:
    """His two rejected triangles (#118, the T01 / T02 pictures of the new117 pack: "BOTH are not proper triangles").
    The machine's 5-minute chart of that day with every triangle it knows then - with NO big-move rule at all (his
    word is open, B2-02), so nothing is hidden by a number of Claude's."""
    d = pd.Timestamp(day)
    m5 = load_bars("5m", d - pd.Timedelta(days=40), d + pd.Timedelta(days=1))
    lines = TL.lines(m5, "5m")
    tri = PT.triangles(m5, "5m", table=lines, need_big_move=False)
    tri = tri[tri["bar"].dt.normalize() == d]
    lo, hi = at_ist(m5, day, window[0]), at_ist(m5, day, window[1])
    items, said = [], []
    first = lines.drop_duplicates("line_id").set_index("line_id")
    for k, r in enumerate(tri.itertuples()):
        for lid in (r.top_id, r.bottom_id):
            L = first.loc[lid]
            items.append(D.Line((L["a_bar"], L["a_price"]), (L["b_bar"], L["b_price"]),
                                touches=(L["a_bar"], L["b_bar"]), name="triangle", extend_to=r.bar,
                                group=f"tri{k}"))
        br = PT.triangle_breaks(m5, "5m", tris=tri.iloc[[k]], table=lines)
        brk = (f"; then a {br['kind'].iat[0]} break {'up' if br['side'].iat[0] == 'high' else 'down'} at "
               f"{ist_index(m5)[m5.index.get_loc(br['bar'].iat[0])]:%H:%M}") if len(br) else ""
        said.append(f"a triangle {ist_index(m5)[m5.index.get_loc(r.start_bar)]:%H:%M}-"
                    f"{ist_index(m5)[m5.index.get_loc(r.bar)]:%H:%M} India time on 5-minute lines, after a move of "
                    f"{r.big_move:.2f} days of range{brk}")
    # nothing the picture names may be left undrawn (#113): with no shape, the words do not name one
    sub = "\n".join(said) if said else "no two lines that narrow together that day (with any move before them)"
    starts = [first.loc[lid, "a_bar"] for r in tri.itertuples() for lid in (r.top_id, r.bottom_id)]
    lo = min([lo, *starts]) - 3 * pd.Timedelta(minutes=5)       # a line is drawn from its first touch, on the chart
    panel = D.Panel(m5.loc[lo:hi], "5m", [D.Candles(), *items], title=f"the machine · {d:%d %b %Y}", clock="ist")
    pic = D.Picture("trade", f"L6b.1 check {key}", f"THE MACHINE · {key[3:]}", [[panel]], sub=wrap(sub),
                    footer_left="purple = the two lines of a triangle" if said else "", footer_right="L6b.1")
    mpath = OUT / f"_m-{key}.jpg"
    drawn = D.render(pic, mpath)
    problems = lint.check(drawn)
    side_by_side(OUT / his, (0, 0, 1210, 950), mpath, OUT / f"{key}.jpg",
                 f"LEFT: the triangle he rejected ({his}, LEARNED #118)   RIGHT: the machine, the same day (L6b.1)",
                 "HIS ANSWER: NO - \"this is NOT a proper triangle pattern ... BOTH are not proper triangles\"")
    mpath.unlink()
    return dict(key=key, sub=sub, lint=problems, n=len(tri))


def wrap(text: str, width: int = 105) -> str:
    import textwrap
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def main() -> int:
    plt_rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        m = OUT / f"_m-{c.key}.jpg"
        got = machine_picture(c, m)
        final = OUT / f"{c.key}.jpg"
        side_by_side(PICS / c.picture, c.crop, m, final,
                     f"LEFT: his picture ({c.picture}, {c.lesson})   RIGHT: the machine, the same candles (L6b.1)",
                     f"HIS ANSWER: {c.his[:150]}")
        m.unlink()
        plt_rows.append((c, got, final))
    daily = daily_case()
    side_by_side(PICS / "Q69-his-answer.png", (0, 1370, 1200, 2270), daily["path"], OUT / "06-Q69D-2018-08-31.jpg",
                 "LEFT: his drawing on Q69 panel D (LEARNED #70)   RIGHT: the machine's daily lines (L6b.1)",
                 "HIS ANSWER: BULLISH - \"it gave breakout of trend line and now it is at the retesting of the trend "
                 "line\"")
    daily["path"].unlink()
    tris = [triangle_case(*t) for t in TRIANGLE_DAYS]
    lines = ["# L6b.1 — his marked trendline charts, side by side with the machine", "",
             "> Made by `python -m tests.l6b1_check`. LEFT = his picture (the panel he answered about); RIGHT = what "
             "the new blocks draw on the same candles. 2016–2020 only: his other trendline charts (Q92 B, Q92 D, Q94 "
             "B–D, Q68 D) are 2021–2023, the hidden years — not read (like L6b.3). No money.", "",
             "| picture | his answer | the machine | lint |", "|---|---|---|---|"]
    for c, got, final in plt_rows:
        machine = got["said"] + (f"; {got['sub2']}" if got["sub2"] else "")
        lines.append(f"| `{final.name}` | {c.his} | {machine} | {'clean' if not got['lint'] else len(got['lint'])} |")
    lines.append(f"| `06-Q69D-2018-08-31.jpg` | BULLISH — \"breakout of trend line and now it is at the retesting\" "
                 f"(#70) | {daily['sub']} | {'clean' if not daily['lint'] else len(daily['lint'])} |")
    for t in tris:
        lines.append(f"| `{t['key']}.jpg` | NO — \"not a proper triangle\" (#118; his picture: the new117 pack, "
                     f"copied here) | {t['sub'].replace(chr(10), '; ')} | "
                     f"{'clean' if not t['lint'] else len(t['lint'])} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
