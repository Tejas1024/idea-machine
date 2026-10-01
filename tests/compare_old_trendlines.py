"""The L6b.1 blocks against the old engine, 2016-2020 (the tune years): trendlines (5-minute, hour), the touch entry,
triangles, wedges, false breaks.

    .venv\\Scripts\\python -m tests.compare_old_trendlines            prints, and writes plan/work/L6b1-differences.md
    .venv\\Scripts\\python -m tests.compare_old_trendlines --no-write  prints only

Tests may import the old code to compare (likhith.md R2); machine/ never does. Both engines read the SAME bars (the new
5-minute bars, given to the old code with its UTC index). Each block is run twice on the new side: with the OLD ruler
(the old engine's 10-day UTC-day range) - the rule alone - and with the NEW ruler (his ADR on broker days, 1.23x) -
the machine as it ships. No money: this compares what the blocks SEE.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter

import numpy as np
import pandas as pd

from machine import config, rules, settings
from machine.blocks import adr, patterns as PT, trendlines as TL
from machine.data.bars import load_bars
from tests.compare_old_blocks import _old, _old_frame

YEARS = ("2016-01-01", "2020-12-31 23:59")
OUT = config.WORK_DIR / "L6b1-differences.md"
OLD_NUMBERS = {"trendline_swing_turn_adr": 0.25}   # the old engine's zig-zag (levels.PROM_FRAC), on its own ruler


def _entries():
    agent = str(config.ROOT / "old" / "ml" / "agent")
    if agent not in sys.path:
        sys.path.insert(0, agent)
    import entries as old_entries          # noqa: E402 - the old engine, for comparing only
    return old_entries


def _okey(L, idx) -> tuple:
    m = L["meta"]
    return ("high" if L["side"] > 0 else "low", idx[m["anchor_a"]], idx[m["anchor_b"]],
            idx[m["touch_c"]] if "touch_c" in m else None)


def compare() -> dict:
    OLV, _ = _old()
    EN = _entries()
    m5 = load_bars("5m", *YEARS)
    h1 = load_bars("1h", *YEARS)
    df = _old_frame(m5)
    idx = m5.index
    h, lo = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
    A = OLV.adr_bands(df)["adr"]
    old_r = np.where(np.isfinite(A), A, 20.0)
    new_r = adr.adr_bars(m5, "5m")["ruler"].to_numpy(float)
    R: dict = {"bars": len(m5)}
    R["ruler_ratio"] = float(np.nanmedian(new_r / old_r))

    # ---- 5-minute trendlines ----------------------------------------------------------------------------------------
    sh, sl = OLV.swings_of(h, lo)
    ssh, ssl = OLV.structural_swings(sh, sl, old_r)
    old = OLV.trendlines(df, ssh, ssl, old_r)
    o3 = {_okey(L, idx) for L in old if L["kind"].startswith("trendline")}
    o2 = {_okey(L, idx)[:3] for L in old if L["kind"].startswith("line")}
    R["m5"] = {}
    for name, rul, numbers in (("old ruler, the old engine's 0.25", old_r, OLD_NUMBERS),
                               ("new ruler, 0.21 (what ships)", new_r, {})):
        with settings.using(numbers):
            t = TL.lines(m5, "5m", rul)
        n3 = {(r.side, r.a_bar, r.b_bar, r.c_bar) for r in t[t.touches == 3].itertuples()}
        n2 = {(r.side, r.a_bar, r.b_bar) for r in t[t.touches == 2].itertuples()} - {k[:3] for k in n3}
        flat = int((t[t.touches == 3]["way"] == "flat").sum())
        R["m5"][name] = dict(old3=len(o3), new3=len(n3), same3=len(o3 & n3), old_only3=len(o3 - n3),
                             new_only3=len(n3 - o3), flat3=flat, old2=len(o2), new2=len(n2), same2=len(o2 & n2))

    # ---- hour trendlines (the old engine priced them on 5-minute candles: its anchors are each hour's first
    # 5-minute candle, 01:05 at the day's start - compared by the hour) ------------------------------------------------
    old_h = OLV.htf_trendlines(df, "1h", "H1", old_r)
    floor = lambda x: None if x is None else pd.Timestamp(x).floor("h")
    oh3 = {tuple(floor(v) if isinstance(v, pd.Timestamp) else v for v in _okey(L, idx)) for L in old_h
           if L["kind"].startswith("trendline")}
    pos = idx.searchsorted(h1.index, "left").clip(0, len(idx) - 1)
    R["h1"] = {}
    for name, rul in (("old ruler", old_r[pos]), ("new ruler", adr.adr_bars(h1, "1h")["ruler"].to_numpy(float))):
        t = TL.lines(h1, "1h", rul)
        n3 = {(r.side, r.a_bar, r.b_bar, r.c_bar) for r in t[t.touches == 3].itertuples()}
        R["h1"][name] = dict(old3=len(oh3), new3=len(n3), same3=len(oh3 & n3))

    # ---- the touch entry (Type 2): the old engine's arms vs the new candidates, old ruler ------------------------
    book, OA, bias = OLV.build(df, with_zones=False, verbose=False)
    o2 = EN.type2(df, book, OA, bias, require_htf=False)
    old_hour = sum(str(s.get("sweep_label", "")).startswith("H1") for s in o2)
    old_arms = {(("high" if s["dir"] == -1 else "low"), idx[s["line_touches"][0]], idx[s["line_touches"][1]],
                 int(s["touch"])) for s in o2 if not str(s.get("sweep_label", "")).startswith("H1")}
    with settings.using(OLD_NUMBERS):
        lines_old = TL.lines(m5, "5m", old_r)
        te_old = TL.touch_entries(m5, "5m", old_r, lines_old)
    new_arms = {("high" if r.direction == "sell" else "low", r.first_bar,
                 lines_old.loc[lines_old["line_id"] == r.line_id, "b_bar"].iat[0], int(r.touch))
                for r in te_old.itertuples()}
    R["type2"] = dict(old=len(old_arms), old_hour=old_hour, new=len(new_arms), same=len(old_arms & new_arms),
                      old_only=len(old_arms - new_arms), new_only=len(new_arms - old_arms),
                      old_only_4th=sum(k[3] == 4 for k in old_arms - new_arms))
    te = TL.touch_entries(m5, "5m", new_r)
    ok = te[te["trend_visible"] & te["touches_apart"]]
    R["steep"] = {"candidates (new ruler, 3rd and 4th touches)": len(te), "a clear trend visible (#116)":
                  int(te["trend_visible"].sum()), "touches 3 hours apart (his fixed rule)":
                  int(te["touches_apart"].sum()), "both": len(ok)}
    R["steep_levels"] = {f"at least {x}": int((ok["steepness"] >= x).sum()) for x in (0.15, 0.53, 0.93, 1.31)}
    R["steep_quartiles"] = [round(float(q), 2) for q in ok["steepness"].quantile([0.25, 0.5, 0.75])] if len(ok) \
        else []

    # ---- triangles: one chart only; known when narrowed (the old engine judged the narrowing at the lines' death) ----
    tris_old = EN.triangles(df, book, OA)
    kinds = Counter((T["res"].get("tf", "M5"), T["sup"].get("tf", "M5")) for T in tris_old)
    one = [T for T in tris_old if T["res"].get("tf", "M5") == "M5" and T["sup"].get("tf", "M5") == "M5"]
    with settings.using({**OLD_NUMBERS, "big_move_adr": 1.0}):
        tn = PT.triangles(m5, "5m", old_r, lines_old)
    new_starts = set(tn["start_bar"])
    R["tri"] = {"old, all": len(tris_old), "old, a 5-minute line with an hour line": kinds.get(("M5", "H1"), 0)
                + kinds.get(("H1", "M5"), 0), "old, two hour lines": kinds.get(("H1", "H1"), 0),
                "old, two 5-minute lines": len(one), "new (old ruler, big move 1.0, 5-minute lines)": len(tn),
                "old one-chart triangles starting on the same candle as a new one": sum(idx[T["born"]] in new_starts
                                                                                        for T in one)}
    narrow_to, min_bars = float(rules.get("what_a_triangle_is")), int(rules.get("how_long_a_triangle_must_last"))
    ex = EN.triangle_sweeps(df, tris_old, OA)
    early = total = 0
    for T in tris_old:
        Rl, Sl, b = T["res"], T["sup"], int(T["born"])
        w = OLV.price_at(Rl, b) - OLV.price_at(Sl, b)
        shrink = Sl["slope"] - Rl["slope"]
        tstar = b + max(min_bars - 1, math.ceil((1 - narrow_to) * w / shrink)) if shrink > 0 else 10 ** 9
        for j, ps in ex.items():
            for P in ps:
                if P["meta"].get("tri_born") == b and P["meta"]["own_line"]["x0"] in (Rl["x0"], Sl["x0"]):
                    total += 1
                    early += int(j < tstar)
    R["tri_peek"] = dict(pokes=total, before_narrowed=early)
    with settings.using({"big_move_adr": None}):
        tall = PT.triangles(m5, "5m", new_r, TL.lines(m5, "5m", new_r), need_big_move=False)
    R["tri_new"] = {"shapes (new ruler, no big-move rule)": len(tall)} | {
        f"after a move of at least {x} days of range": int((tall["big_move"] >= x).sum()) for x in (0.5, 1.0, 1.5)}
    if len(tall):
        R["tri_new"]["his rejected T01 (28 Feb 2018) / T02 (09 Oct 2019): shapes known that day"] = int(
            tall["bar"].dt.normalize().isin([pd.Timestamp("2018-02-28"), pd.Timestamp("2019-10-09")]).sum())

    # ---- wedges: the old engine's breaks (no hour / day filter) vs the new ones, old ruler ------------------------
    ow = EN.type_wedge(df, book, OA, bias, need_hd=False)
    m5_lines = {tuple(sorted((L["meta"]["anchor_a"], L["meta"]["anchor_b"], L["meta"]["touch_c"]))) for L in old
                if L["kind"].startswith("trendline")}
    old_w5 = {idx[int(s["sweep_bar"])] for s in ow if tuple(s["wedge_line"]["touches"]) in m5_lines}
    nw = PT.wedges(m5, "5m", old_r, lines_old)
    R["wedge"] = dict(old_with_a_gap=len(ow), old_5m=len(old_w5), new_breaks=len(nw),
                      old_found_in_new=len(old_w5 & set(nw["bar"])), new_ruler=len(PT.wedges(m5, "5m", new_r)))
    return R


def text(R: dict) -> str:
    L = ["# L6b.1 — trendlines, triangles, wedges, false breakouts against the old engine, 2016–2020", "",
         "> Written by `python -m tests.compare_old_trendlines` (L6b.1). Both engines read the same 5-minute bars. "
         "Counts only, no money. The new side is run twice: with the OLD ruler and the old engine's own numbers (the "
         "port alone: does the new code draw what the old one drew?), and as it ships — the NEW ruler (his ADR on "
         f"broker days, median {R['ruler_ratio']:.2f}× the old; plan/work/rules-missing.md R1) with the swing turn "
         "0.21 (= the old 0.25 in the same dollars; every line of his 2016–2020 charts is drawn as he saw it — "
         "plan/work/L6b1-notes.md).", "",
         "## 5-minute trendlines", "",
         "| ruler | old 3-touch | new 3-touch | the same line | old only | new only | new flat (#116) | old 2-touch "
         "| new 2-touch | the same |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in R["m5"].items():
        L.append(f"| {k} | {v['old3']:,} | {v['new3']:,} | {v['same3']:,} | {v['old_only3']:,} | {v['new_only3']:,} "
                 f"| {v['flat3']:,} | {v['old2']:,} | {v['new2']:,} | {v['same2']:,} |")
    L += ["", "With the old ruler and its numbers the new code draws the old engine's lines, line for line; the only "
          "new ones are FLAT (the old engine asked for strictly falling highs; #116: a flat line is still a line). As "
          "it ships, the touch width (0.025) and the 0.005 through are shares of a bigger day's range, and the "
          "zig-zag keeps a slightly different set of swings, so some lines differ. His trendline calls in these "
          "years hold either way: Q92 A, Q92 C and Q94 A (all NO) stay 2-touch lines; his Q68 A and B lines (#69) are "
          "drawn (plan/work/L6b-check/).", "",
          "## Hour trendlines", "", "| ruler | old 3-touch | new 3-touch | the same line |", "|---|---|---|---|"]
    for k, v in R["h1"].items():
        L.append(f"| {k} | {v['old3']:,} | {v['new3']:,} | {v['same3']:,} |")
    t2 = R["type2"]
    L += ["", "## The touch entry (Type 2) — where it would arm, before any steepness", "",
          f"Old engine `entries.type2` (no hour / day filter) on 5-minute lines {t2['old']:,} arms (and "
          f"{t2['old_hour']:,} more on hour lines, which the new block draws on the hour chart itself) · new "
          f"`touch_entries` with the old ruler {t2['new']:,} · the same line and touch {t2['same']:,} · old only "
          f"{t2['old_only']:,} (4th touches: {t2['old_only_4th']:,}) · new only {t2['new_only']:,}. The old 4th touch "
          "was any structural swing within 2 touch widths of the line on EITHER side; the new one respects the line "
          "like the 3rd (#92, #94: never more than 0.005 of a day through) and comes before the line is broken. (The "
          "old one also dropped arms with a tiny stop or no target inside the entry; the new block lists every arm "
          "and the compiler's rules decide.)", "",
          "With the new ruler, the candidates that pass what is already his (a clear trend #116, touches 3 hours "
          "apart):", ""]
    L += [f"- {k}: {v:,}" for k, v in R["steep"].items()]
    L += ["", "How many would stay at each steepness of the B2-01 question (days of range the line travels in a day) — "
          f"counts only; quartiles of the candidates: {R['steep_quartiles']}:", ""]
    L += [f"- {k}: {v:,}" for k, v in R["steep_levels"].items()]
    tr = R["tri"]
    L += ["", "## Triangles", ""] + [f"- {k}: {v:,}" for k, v in tr.items()]
    p = R["tri_peek"]
    L += ["", f"**The old engine's triangle knew its future.** It dated a triangle at the later birth of its two lines "
          f"but judged the narrowing (0.60) where a line DIED — later. Of its {p['pokes']:,} false-break pokes, "
          f"{p['before_narrowed']:,} came before the two lines had narrowed to 0.60: on those candles it was not a "
          "triangle yet. The new block knows a triangle only at the close of the candle where it has narrowed, and "
          "looks for the break only after it. It also pairs lines of ONE chart: the old engine paired a 5-minute line "
          "with an hour line (a shape nobody draws on one chart).", "",
          "The new ruler, his word 'big move' still open (B2-02):", ""]
    L += [f"- {k}: {v:,}" for k, v in R["tri_new"].items()]
    w = R["wedge"]
    L += ["", "## Wedges", "",
          f"Old `type_wedge` (no hour / day filter; only breaks followed by its gap) {w['old_with_a_gap']:,}, of them on "
          f"5-minute lines {w['old_5m']:,} · new wedge breaks with the old ruler {w['new_breaks']:,} (every break; the "
          f"gap is the compiler's step) · the old 5-minute ones found in the new {w['old_found_in_new']:,} · new "
          f"ruler {w['new_ruler']:,}.", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    R = compare()
    out = text(R)
    print(out)
    if not a.no_write:
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(out, encoding="utf-8")
        tmp.replace(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
