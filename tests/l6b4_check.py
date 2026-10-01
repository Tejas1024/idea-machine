"""L6b.4 check: his two 1-minute charts drawn by the machine, his F04 trade through the compiler, and how often the
1-minute check waits on his ideas. 2016-2020 only (the tune years). Counts only - no money, so no trial file.

    .venv\\Scripts\\python -m tests.l6b4_check     writes plan/work/L6b-check/09-*.jpg, 10-*.jpg and
                                                  plan/work/L6b4-check.md

F04 (#111, 11 Mar 2019): the machine's 1-minute chart - his line, the buy the check stops, the take, the gap after it.
Q98 C (#98, 9 Jun 2017): his panel next to the machine's 1-minute chart of the same candles - his line (20:46 -> 21:35
lows), the band his card called the strong zone, the take.
Ideas: idea #1 (its chosen version, U1=A U2=A U3=B) and idea #2 (A/A/A), each with the step one_minute added right
after its gap step - a VERSION of the idea, never written into its locked form.
His T3 examples (plan/work/idea-001/examples.md, 30 setups he judged on 5-minute pictures): which would wait.
"""
from __future__ import annotations

import copy
import sys
import textwrap
import time
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from machine import config, settings
from machine.blocks import one_minute as OM
from machine.charts import draw as D
from machine.charts import lint
from machine.data import clock
from machine.data.bars import load_bars
from machine.ideas import compile as C
from machine.ideas import store
from machine.ideas.form import Idea

sys.path.insert(0, str(config.ROOT))
from tests.l6b1_check import GREY, side_by_side  # noqa: E402
from tests.test_ideas import type1  # noqa: E402

OUT = config.WORK_DIR / "L6b-check"
REPORT = config.WORK_DIR / "L6b4-check.md"
PICS = config.ROOT / "strategy" / "pictures"
M1, M5 = pd.Timedelta(minutes=1), pd.Timedelta(minutes=5)


def srv(ist: str) -> pd.Timestamp:
    return clock.utc_to_server(pd.Timestamp(ist, tz="Asia/Kolkata"))


def ist(t) -> str:
    return clock.to_ist(pd.Timestamp(t)).strftime("%H:%M")


def wrap(text: str, width: int = 105) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def with_check(form: Idea, **params) -> Idea:
    """The idea with the step one_minute right after its gap step: a VERSION of the same idea (its own version id,
    from its fingerprint) - measured here, never written into the locked form."""
    d = form.model_dump(mode="json")
    k = next(i for i, s in enumerate(d["steps"]) if s["block"] == "gap_of_the_move")
    d["steps"].insert(k + 1, dict(id="m1", what="no 1-minute line against the entry (#111)", block="one_minute",
                                  tf="1m", max_wait_bars="no limit", params=params,
                                  sources=[dict(ref="strategy.md section 5, #100, #111 (L6b.4)")]))
    return Idea.load(d)


def check_form(**kw) -> dict:
    d = type1(id="idea-940", title="grab + gap + the 1-minute check")
    d["steps"] = d["steps"] + [dict(id="m1", what="no 1-minute line against the entry (#111)", block="one_minute",
                                    tf="1m", max_wait_bars="no limit")]
    d.update(kw)
    return d


# ======================================================================================================== F04

def f04() -> dict:
    mk = C.Market.load("2019-02-18", "2019-03-12", tfs=("1m", "5m", "1h", "1d", "1w"))
    base = C.compile_idea(Idea.load(type1(directions=["buy"])), mk)
    chk = C.compile_idea(Idea.load(check_form(directions=["buy"])), mk)
    first = next(s for s in base.signals if s.known_at == srv("2019-03-11 17:55"))
    after = next(s for s in chk.signals if srv("2019-03-11 17:55") <= s.known_at <= srv("2019-03-11 19:00"))
    fp = settings.fingerprint(chk.resolved)
    bk = mk.book1(fp)
    line = bk[(bk["a_bar"] == srv("2019-03-11 17:12")) & (bk["b_bar"] == srv("2019-03-11 17:36"))].iloc[0]
    m1 = mk.bars("1m")
    take = m1.index[int(line["taken"])]
    take_px = float(line["b_price"] + line["slope"] * (int(line["taken"]) - int(line["x0"])))
    g_lo, g_hi = 1295.23, 1295.61
    lo, hi = srv("2019-03-11 16:55"), srv("2019-03-11 18:50")
    items = [D.Candles(),
             D.Line((line["a_bar"], line["a_price"]), (line["b_bar"], line["b_price"]),
                    touches=(line["a_bar"], line["b_bar"]), name="the line (2 touches)", extend_to=take, color=GREY),
             D.Level(first.entry_price, "1st buy - the check stops it", start=first.known_at),
             D.Level(first.stop, "its hedge", start=first.known_at, color=D.RED),
             D.Box(srv("2019-03-11 18:35"), after.known_at, g_lo, g_hi, name="gap"),
             D.Level(after.entry_price, "buy after the line is taken", start=after.known_at, color=D.GREEN),
             D.Level(after.stop, "new hedge", start=after.known_at, color=D.RED)]
    panel = D.Panel(m1.loc[lo:hi], "1m", items, title="the machine · 11 Mar 2019 (his F04)", clock="ist")
    sub = (f"the line ({ist(line['a_bar'])} -> {ist(line['b_bar'])} lows) is between the {ist(first.known_at)} buy "
           f"({first.entry_price}) and its hedge ({first.stop}): wait. Taken at {ist(take)} (the line at "
           f"{take_px:.2f}).\nThen the break and the first gap ({ist(srv('2019-03-11 18:35'))}, {g_lo}-{g_hi}): buy "
           f"{after.entry_price} at {ist(after.known_at)}, hedge {after.stop} - his words: \"that is actual entry\"")
    pic = D.Picture("trade", "L6b.4 check F04 2019-03-11", "THE MACHINE · F04 · 1-minute check", [[panel]],
                    sub=wrap(sub, 88), footer_left="grey = a line with 2 touches on the 1-minute chart",
                    footer_right="L6b.4")
    path = OUT / "09-F04-2019-03-11.jpg"
    drawn = D.render(pic, path)
    return dict(path=path, lint=lint.check(drawn), first=first, after=after, take=take, take_px=take_px,
                line=line, waits=chk.waits)


# ======================================================================================================== Q98 C

def q98c() -> dict:
    m1 = load_bars("1m", "2017-05-20", srv("2017-06-09 22:40"))
    ruler = OM._ruler(m1, None)
    L = OM.lines(m1, ruler)
    bk = OM.book(m1, L, ruler)
    a, b = srv("2017-06-09 20:46"), srv("2017-06-09 21:35")
    line = bk[(bk["a_bar"] == a) & (bk["b_bar"] == b)].iloc[0]
    third = L[(L["line_id"] == line["line_id"]) & (L["touches"] == 3)]
    take = m1.index[int(line["taken"])] if line["taken"] >= 0 else None
    zone_lo, zone_hi = 1265.59, 1268.21                   # his card's band (the old engine's zone - L6b.2 builds zones)
    arm = srv("2017-06-09 21:40")                         # the 21:35 candle closed through the zone
    i1 = int(m1.index.searchsorted(arm, "left")) - 1
    inside = OM.between(bk, i1, zone_lo, zone_hi, "low")
    day = m1.loc[srv("2017-06-07 00:00"):]
    tri = OM.triangles(day, ruler[-len(day):], L[L["b_bar"] >= day.index[0]])
    tri = tri[(tri["bar"] >= srv("2017-06-09 20:30")) & (tri["bar"] <= srv("2017-06-09 22:30"))]
    first_tri = tri[tri["known_at"] <= arm].sort_values("known_at").head(1)     # known before the zone broke
    touches = (a, b) + ((third["c_bar"].iat[0],) if len(third) else ())
    items = [D.Candles(), D.Zone(zone_lo, zone_hi, "strong zone (his card)"),
             D.Line((a, line["a_price"]), (b, line["b_price"]), touches=touches,
                    name="his line" + (" (3rd touch 22:12)" if len(third) else ""), extend_to=take or m1.index[-1],
                    color=D.PURPLE if len(third) else GREY)]
    tri_words = "no 1-minute triangle known before the zone broke"
    starts = []
    if len(first_tri):
        T = first_tri.iloc[0]
        one = L.drop_duplicates("line_id").set_index("line_id")
        for lid in (T["top_id"], T["bottom_id"]):
            r = one.loc[lid]
            items.append(D.Line((r["a_bar"], r["a_price"]), (r["b_bar"], r["b_price"]), touches=(r["a_bar"], r["b_bar"]),
                                name="triangle", extend_to=T["bar"], group="tri", color=D.BLUE))
            starts.append(r["a_bar"])
        tri_words = (f"the machine's 1-minute triangle (blue), known {ist(T['known_at'])}: "
                     f"{T['bottom']:.2f}-{T['top']:.2f} then - inside the zone")
    lo, hi = min([srv("2017-06-09 20:30"), *starts]) - 3 * M1, srv("2017-06-09 22:25")
    panel = D.Panel(m1.loc[lo:hi], "1m", items, title="the machine · 9 Jun 2017 (his Q98 C)", clock="ist")
    px = float(inside.set_index("line_id").loc[line["line_id"], "price"]) if line["line_id"] in set(
        inside["line_id"]) else float("nan")
    sub = (f"his line: the {ist(a)} low ({line['a_price']}) -> the {ist(b)} low ({line['b_price']}), known "
           f"{ist(line['known_at'])}; 3rd touch {ist(third['c_bar'].iat[0]) if len(third) else '-'}.\n"
           f"{tri_words}.\nWhen the 21:35 candle closed through the zone, his line was inside it at {px:.2f}: wait. "
           f"Taken at {ist(take) if take is not None else '-'}.")
    pic = D.Picture("trade", "L6b.4 check Q98 C 2017-06-09", "THE MACHINE · Q98 C · 1-minute chart", [[panel]],
                    sub=wrap(sub, 95), footer_left="purple = a trendline (3 touches) · blue = a triangle's two lines · "
                    "brown band = the strong zone on his card", footer_right="L6b.4")
    mpath = OUT / "_m-10-Q98C.jpg"
    drawn = D.render(pic, mpath)
    problems = lint.check(drawn)
    final = OUT / "10-Q98C-2017-06-09.jpg"
    side_by_side(PICS / "Q98-D8-break-distance.png", (0, 700, 650, 1240), mpath, final,
                 "LEFT: his panel C (Q98-D8-break-distance.png, LEARNED #98)   RIGHT: the machine's 1-minute chart (L6b.4)",
                 "HIS WORDS: \"if you look into the one minute candle there is definitely a triangle ... I want price to "
                 "break that trend line\"")
    mpath.unlink()
    zb = {}                                             # L6b.2's zone break at this chart, with and without the lines
    mk = C.Market.load("2017-04-20", "2017-06-12", tfs=("1m", "5m", "1h", "1d", "1w"))
    form = Idea.load(type1(directions=["buy"]))
    for on in (False, True):
        with settings.using({"zone_break_inner_1m_lines": on}):
            run = C._Run(mk, form, "buy", settings.fingerprint())
            t = mk.zone_breaks(run.fp)
            t = t[(t["direction"] == "buy") & (t["known_at"] >= srv("2017-06-09 21:00"))
                  & (t["known_at"] <= srv("2017-06-09 22:30"))]
            zb[on] = [(r, C.ZoneBreak()._event(run, r)) for r in t.itertuples()]
    return dict(path=final, lint=problems, line=line, third=third, take=take, inside=len(inside), px=px,
                tri=len(tri), tri_words=tri_words, sub=sub, zb=zb)


# ======================================================================================================== the ideas

T3 = [  # plan/work/idea-001/examples.md: (number, entry candle IST, direction, entry, hedge point, his answer, counted)
    (1, "2016-08-23 04:20", "buy", 1337.73, 1335.96, "yes", False),
    (2, "2016-08-01 22:00", "sell", 1352.52, 1354.02, "yes", False),
    (3, "2016-06-22 15:35", "buy", 1263.55, 1261.13, "no: too big", False),
    (4, "2017-12-01 18:35", "buy", 1275.85, 1274.34, "yes", False),
    (5, "2017-07-31 12:50", "sell", 1266.83, 1269.28, "yes", False),
    (6, "2017-08-30 23:00", "buy", 1308.19, 1307.36, "yes", False),
    (7, "2018-08-07 01:05", "buy", 1207.70, 1207.39, "yes", False),
    (8, "2018-12-18 00:10", "sell", 1247.25, 1248.06, "yes", False),
    (9, "2018-06-22 19:00", "buy", 1268.46, 1266.85, "yes", False),
    (10, "2019-06-11 12:55", "buy", 1325.67, 1324.50, "yes", False),
    (11, "2019-02-22 14:20", "sell", 1326.19, 1326.99, "yes", False),
    (12, "2019-01-07 20:05", "sell", 1292.99, 1295.03, "yes", False),
    (13, "2020-10-16 21:50", "buy", 1900.05, 1897.76, "yes", False),
    (14, "2020-11-05 17:45", "sell", 1919.03, 1920.77, "yes", False),
    (15, "2020-06-24 16:15", "sell", 1778.26, 1779.43, "yes", False),
    (24, "2016-02-16 22:45", "sell", 1210.81, 1216.11, "yes", True),
    (25, "2018-09-06 12:05", "buy", 1197.00, 1195.74, "yes", True),
    (26, "2018-03-28 07:45", "sell", 1345.82, 1346.99, "yes", True),
    (27, "2018-02-06 00:20", "buy", 1335.13, 1333.32, "yes", True),
    (28, "2019-12-17 05:50", "buy", 1475.95, 1474.75, "yes", True),
    (29, "2020-05-14 01:40", "sell", 1715.59, 1718.08, "yes", True),
    (30, "2017-06-01 23:45", "sell", 1266.27, 1267.98, "yes", True),
    (31, "2018-12-13 19:45", "buy", 1243.25, 1240.34, "yes", True),
    (32, "2018-08-24 01:05", "buy", 1185.09, 1183.80, "yes", True),
    (33, "2016-09-29 19:00", "buy", 1318.52, 1316.33, "yes (partly no)", True),
    (34, "2017-08-01 20:25", "sell", 1272.08, 1274.04, "no: lows too near", True),
    (35, "2019-01-10 10:10", "sell", 1296.05, 1297.03, "no: too far + too near", True),
    (36, "2017-08-10 20:10", "sell", 1286.07, 1287.76, "no: same as 35", True),
    (37, "2018-05-15 19:40", "buy", 1294.79, 1291.88, "no: same as 35", True),
    (38, "2017-03-13 13:05", "sell", 1209.32, 1211.11, "no: same as 35", True),
]


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def compare(form: Idea, readings: dict, mk: C.Market) -> dict:
    t0 = time.time()
    base = C.compile_idea(form, mk, readings=readings, trace=True)
    say(f"{form.id}: without the check {len(base.signals)} signals ({time.time() - t0:.0f} s)")
    chk = C.compile_idea(with_check(form), mk, readings=readings, trace=True)
    say(f"{form.id}: with the check {len(chk.signals)} signals ({time.time() - t0:.0f} s)")
    lit = C.compile_idea(with_check(form), mk, readings=readings,
                         changes={"settings.one_minute_new_lines_after_a_take": True})
    say(f"{form.id}: literal {len(lit.signals)} signals ({time.time() - t0:.0f} s)")
    key = lambda s: (s.direction, s.known_at, s.entry_price, s.stop)
    b_keys, c_keys = {key(s) for s in base.signals}, {key(s) for s in chk.signals}
    waited = [r for r in chk.trace if r["waits"]]
    got_new = [r for r in waited if r["signal"]]
    by_id = {s.id: s for s in chk.signals}
    b_by_id = {s.id: s for s in base.signals}
    dist_new = [abs(by_id[r["signal"]].entry_price - by_id[r["signal"]].stop) for r in got_new]
    starts = {(r["direction"], r["start"]): r for r in base.trace if r["signal"]}
    dist_old = [abs(b_by_id[starts[(r["direction"], r["start"])]["signal"]].entry_price
                    - b_by_id[starts[(r["direction"], r["start"])]["signal"]].stop)
                for r in waited if (r["direction"], r["start"]) in starts]
    years = Counter(s.known_at.year for s in base.signals), Counter(s.known_at.year for s in chk.signals)
    return dict(base=base, chk=chk, lit=lit, kept=len(b_keys & c_keys), gone=len(b_keys - c_keys),
                new=len(c_keys - b_keys), waited_starts=len(waited), waited_new=len(got_new),
                waited_starts_with_base_signal=sum((r["direction"], r["start"]) in starts for r in waited),
                dist_new=float(np.median(dist_new)) if dist_new else float("nan"),
                dist_old=float(np.median(dist_old)) if dist_old else float("nan"), years=years,
                seconds=time.time() - t0)


def t3_examples(res: dict) -> list[dict]:
    base, chk = res["base"], res["chk"]
    b_by_id = {s.id: s for s in base.signals}
    c_by_id = {s.id: s for s in chk.signals}
    base_rec = {r["signal"]: r for r in base.trace if r["signal"]}
    chk_rec = {(r["direction"], r["start"]): r for r in chk.trace}
    rows = []
    for n, when, d, entry, hedge, his, counted in T3:
        fill = srv(when)
        match = [s for s in base.signals if s.direction == d and abs(s.entry_price - entry) < 0.015
                 and abs(s.stop - hedge) < 0.015 and fill - pd.Timedelta(days=3) <= s.known_at <= fill + M5]
        if not match:
            rows.append(dict(n=n, when=when, d=d, his=his, counted=counted, found="not made by the compiler"))
            continue
        s = match[-1]
        r = base_rec[s.id]
        c = chk_rec.get((r["direction"], r["start"]))
        if c is None or not c["waits"]:
            what = "no wait - the same entry"
        else:
            lid, t = c["wait_for"][0]
            side, tf, rest = lid.split(":", 2)                   # line_low:1m:<touch a>:<touch b> (server time)
            ta, tb = pd.Timestamp(rest[:16]), pd.Timestamp(rest[17:])
            what = (f"WAITS for a {tf} line through the {side.split('_')[1]}s {ist(ta)} -> {ist(tb)} "
                    f"({(tb - ta) // M1} min), taken {ist(t)} -> ")
            what += (f"new entry {c_by_id[c['signal']].entry_price} at {ist(c_by_id[c['signal']].known_at)}"
                     if c["signal"] else f"no entry ({c['why']})")
        rows.append(dict(n=n, when=when, d=d, his=his, counted=counted, found=what))
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--pictures" in sys.argv:                      # only the two pictures (the counts take several minutes)
        f, q = f04(), q98c()
        print(f"F04 lint {f['lint'] or 'clean'} · Q98 C lint {q['lint'] or 'clean'}")
        return 0
    out = ["# L6b.4 — the 1-minute layer, checked", "",
           "> Made by `python -m tests.l6b4_check`. 2016–2020 only (the tune years). Counts only — no money, so no "
           "trial file. India time unless said.", ""]
    f = f04()
    out += ["## F04 (#111, 11 Mar 2019) — his trade through the compiler", "",
            f"- His Type 1 (grab → break → first gap) makes the buy he criticised: **{ist(f['first'].known_at)}, "
            f"{f['first'].entry_price}, hedge {f['first'].stop}**.",
            f"- His line ({ist(f['line']['a_bar'])} → {ist(f['line']['b_bar'])} lows, known "
            f"{ist(f['line']['known_at'])}) stands between that buy and its hedge → **wait**. It is taken at "
            f"**{ist(f['take'])}** (the line at {f['take_px']:.2f}).",
            f"- The take becomes the grab; the break and the first gap after it: **buy {f['after'].entry_price} at "
            f"{ist(f['after'].known_at)}, hedge {f['after'].stop}** — the 18:35 gap (1295.23–1295.61) he called "
            "\"that is actual entry\".",
            f"- Picture: `L6b-check/{f['path'].name}` — lint {'clean' if not f['lint'] else f['lint']}.", ""]
    q = q98c()
    out += ["## Q98 C (#98, 9 Jun 2017) — his 1-minute triangle", "",
            f"- His line: the 20:46 low ({q['line']['a_price']}) → the 21:35 low ({q['line']['b_price']}), drawn by "
            f"the machine, known {ist(q['line']['known_at'])}"
            + (f"; its 3rd touch {ist(q['third']['c_bar'].iat[0])} (the candle his chart ended on)." if len(q['third'])
               else "."),
            f"- When the 21:35 candle closed through the zone, the line lay inside the zone at {q['px']:.2f} → "
            f"**wait** ({q['inside']} line(s) inside). Taken at **{ist(q['take']) if q['take'] is not None else '-'}**.",
            f"- 1-minute triangles the machine knows between 20:30 and 22:30 (with ANY move before them — 'big move' "
            f"is open, B2-02): **{q['tri']}**. The first: {q['tri_words']} — his \"triangle inside the zone, on the "
            "one-minute chart\". Its lower line is not the one he drew (his runs 20:46 → 21:35; the triangle's runs "
            "into the 20:46 low from earlier); both are lines, and the check reads every line.",
            f"- Picture: `L6b-check/{q['path'].name}` — lint {'clean' if not q['lint'] else q['lint']}.", "",
            "L6b.2's zone break at this chart (each strong zone the break closed through), without and with the "
            "1-minute lines inside the zone (settings `zone_break_inner_1m_lines`):", ""]
    for on in (False, True):
        for r, e in q["zb"][on]:
            out.append(f"- {'with' if on else 'without'} the 1-minute lines: zone {r.lo:.2f}–{r.hi:.2f}, break "
                       f"{ist(r.known_at)} → " + ("no trade" if e is None else
                                                   f"{e['how']}: buy {e['entry']:.2f}, hedge {e['hedge']:.2f}, known "
                                                   f"{ist(e['known_at'])}" + (f" (after the take at {e['inner']:.2f})"
                                                                              if e['how'] != 'retest' else "")))
    out.append("")
    say("pictures done")
    t0 = time.time()
    mk = C.Market.load("2016", "2020", tfs=("1m", "5m", "1h", "1d", "1w"))
    fp = settings.fingerprint()
    bk = mk.book1(fp)
    say(f"1-minute book 2016-2020: {len(bk):,} lines ({time.time() - t0:.0f} s)")
    out += ["## The 1-minute lines, 2016–2020", "",
            f"- {len(bk):,} lines ({int((bk['taken'] >= 0).sum()):,} taken on 1-minute - the 1-minute trendline "
            f"break); built in {time.time() - t0:.0f} s with the bars.", ""]
    out += ["## How often the check waits on his ideas (2016–2020, counts only)", "",
            "Each idea with the step `one_minute` added right after its gap step — a **version** of the idea, not "
            "written into its locked form. Default = the lines known at the first check; literal "
            "= every gap after a take checked again (settings `one_minute_new_lines_after_a_take`).", "",
            "| idea | signals without | with the check | literal | kept the same | gone | new | starts that "
            "waited | … then made a signal | hedge distance, median (without → after the wait) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for idea, rd in (("idea-001", {"U1": "A", "U2": "A", "U3": "B"}), ("idea-002", {"U1": "A", "U2": "A", "U3": "A"})):
        form = store.load_form(idea)
        r = compare(form, rd, mk)
        res[idea] = r
        out.append(f"| {idea} ({', '.join(f'{k}={v}' for k, v in rd.items())}) | {len(r['base'].signals):,} | "
                   f"{len(r['chk'].signals):,} | {len(r['lit'].signals):,} | {r['kept']:,} | {r['gone']:,} | "
                   f"{r['new']:,} | {r['waited_starts']:,} | {r['waited_new']:,} | ${r['dist_old']:.2f} → "
                   f"${r['dist_new']:.2f} |")
    # the open question (plan/work/question-queue/L6b4-Q1-1-minute-lines.md): a 2-touch line only when its touches are
    # 20+ minutes apart (his two lines: 24 and 49) - reading B, measured on idea #1
    r1 = res["idea-001"]
    b_form = with_check(store.load_form("idea-001"))
    rb = C.compile_idea(b_form, mk, readings={"U1": "A", "U2": "A", "U3": "B"}, trace=True,
                        changes={"settings.trendline_touch_span_bars_1m": [20, 300]})
    say(f"idea-001 reading B: {len(rb.signals)} signals")
    key = lambda s: (s.direction, s.known_at, s.entry_price, s.stop)
    kept_b = len({key(s) for s in r1["base"].signals} & {key(s) for s in rb.signals})
    rows_b = t3_examples({"base": r1["base"], "chk": rb})
    out += ["", "Signals by year (without / with the check):", ""]
    for idea, r in res.items():
        yrs = sorted(set(r["years"][0]) | set(r["years"][1]))
        out.append(f"- {idea}: " + " · ".join(f"{y} {r['years'][0][y]} / {r['years'][1][y]}" for y in yrs))
    rows = t3_examples(res["idea-001"])
    out += ["", "## His T3 examples (idea #1, judged on 5-minute pictures — round 3 counted, round 1 kept)", "",
            "He judged these without a 1-minute chart, so they cannot set any number here; this only shows which "
            "of them the check would change.", "",
            "| # | entry candle | buy / sell | his answer | counted | the 1-minute check |", "|---|---|---|---|---|---|"]
    for x in rows:
        out.append(f"| {x['n']} | {x['when']} | {x['d']} | {x['his']} | {'yes' if x['counted'] else 'no'} | "
                   f"{x['found']} |")
    yes = [x for x in rows if x["counted"] and x["his"].startswith("yes")]
    no = [x for x in rows if x["counted"] and x["his"].startswith("no")]
    waits = lambda xs: sum(x["found"].startswith("WAITS") for x in xs)
    made = lambda xs: sum(not x["found"].startswith("not made") for x in xs)
    yes_b = [x for x in rows_b if x["counted"] and x["his"].startswith("yes")]
    out += ["", f"Counted round: his YES {len(yes)} (the compiler makes {made(yes)}; the check waits on "
            f"{waits(yes)}: {', '.join(str(x['n']) for x in yes if x['found'].startswith('WAITS'))}) · his NO "
            f"{len(no)} (made {made(no)}; waits on {waits(no)}).", "",
            "## The open question: which 1-minute lines count (idea #1, 2016–2020)", "",
            "| reading | signals with the check | kept the same (of the 868) | his counted YES that wait |",
            "|---|---|---|---|",
            f"| A (built): 2 touches 6+ minutes apart | {len(r1['chk'].signals)} | {r1['kept']} | {waits(yes)} |",
            f"| B: 2 touches 20+ minutes apart (his lines: 24, 49) | {len(rb.signals)} | {kept_b} | {waits(yes_b)} |",
            "", "Queued for him (pictures still to draw: his five YES setups that wait, and his F04): "
            "`plan/work/question-queue/L6b4-Q1-1-minute-lines.md`.", ""]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")          # the Windows console is cp1252: the report has arrows
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
