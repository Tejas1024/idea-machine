"""The new core blocks against the old engine on one month of 2019 (L6.1 step 5).

    .venv\\Scripts\\python -m tests.compare_old_blocks            prints the table, writes plan/work/L6-differences.md
    .venv\\Scripts\\python -m tests.compare_old_blocks --no-write  prints only

Tests may import the old code to compare (likhith.md R2); machine/ never does. Both engines read the SAME bars: the
new 5-minute bars (built from MT5's 1-minute bars, identical to MT5's own 5-minute bars, L1), given to the old code
with its UTC index. The month compared is March 2019; November 2018 - February 2019 is warm-up (the ADR needs 10
days, the old engine's hour / day bias needs history).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

from machine import config, settings
from machine.blocks import adr, fib, fvg, sessions, structure, swings
from machine.data.bars import load_bars

MONTH = ("2019-03-01", "2019-03-31 23:59")
WARM = "2018-11-01"
OUT = config.WORK_DIR / "L6-differences.md"
L62_MARK = "<!-- L6.2: tests.compare_old_liquidity writes below this line -->"


def _old():
    agent = str(config.ROOT / "old" / "ml" / "agent")
    if agent not in sys.path:
        sys.path.insert(0, agent)
    import levels as old_levels            # noqa: E402 - the old engine, for comparing only
    import setup_engine as old_setup       # noqa: E402
    return old_levels, old_setup


def _old_frame(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars[["open", "high", "low", "close", "vol", "spread"]].copy()
    df.index = pd.DatetimeIndex(bars["utc"])
    return df


def _in_month(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return np.asarray((t >= pd.Timestamp(MONTH[0])) & (t <= pd.Timestamp(MONTH[1])))


def compare() -> dict:
    OLV, OSE = _old()
    bars = load_bars("5m", WARM, MONTH[1])
    df = _old_frame(bars)
    o, h, lo, c = (bars[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(bars)
    idx = bars.index
    month = _in_month(idx)
    R: dict = {"bars_month": int(month.sum()), "first": str(idx[month][0]), "last": str(idx[month][-1])}

    # ---- swings ---------------------------------------------------------------------------------------------
    sh, sl = OLV.swings_of(h, lo)
    new = swings.swings(bars, "5m")
    pos = idx.get_indexer(pd.DatetimeIndex(new["bar"]))
    for side, arr in (("high", sh), ("low", sl)):
        old_set = {int(k) for k in np.flatnonzero(np.isfinite(arr)) if month[k]}
        new_set = {int(p) for p, s in zip(pos, new["side"]) if s == side and month[p]}
        R[f"swing_{side}"] = (len(old_set), len(new_set), len(old_set & new_set))
    ok_pos = pos + 2 < n
    stamped = pd.DatetimeIndex(new["known_at"])[ok_pos] == (idx[pos[ok_pos] + 2] + pd.Timedelta(minutes=5))
    R["swing_known_right"] = (int(ok_pos.sum()), int(stamped.sum()))

    # ---- order blocks: the block known AT each bar -----------------------------------------------------------
    old_dn, old_up = OSE.ob_engulf_arrays(o, h, lo, c)
    ob = structure.order_blocks(bars, "5m")
    for side, old_arr in (("sell", old_dn), ("buy", old_up)):
        s = ob[ob.side == side]
        cur = structure._current(n, idx.get_indexer(pd.DatetimeIndex(s["block_bar"])),
                                 idx.get_indexer(pd.DatetimeIndex(s["confirmed_bar"])))
        diff = np.flatnonzero((cur != old_arr) & month)
        R[f"block_{side}"] = (int(month.sum()), int(month.sum() - len(diff)), [str(idx[d]) for d in diff[:3]])
        R[f"blocks_{side}_count"] = int(_in_month(s["confirmed_bar"]).sum())

    # ---- breaks: the first wick through each block while it is the most recent (old arrays, bar by bar) ------
    br = structure.breaks(bars, "5m")
    for side, old_arr in (("sell", old_dn), ("buy", old_up)):
        old_first = {}
        for i in range(n):
            k = int(old_arr[i])
            if k < 0 or k in old_first:
                continue
            if (side == "sell" and lo[i] < lo[k]) or (side == "buy" and h[i] > h[k]):
                old_first[k] = i
        old_set = {(k, i) for k, i in old_first.items() if month[i]}
        s = br[br.side == side]
        new_set = {(int(k), int(i)) for k, i in zip(idx.get_indexer(pd.DatetimeIndex(s["block_bar"])),
                                                    idx.get_indexer(pd.DatetimeIndex(s["bar"]))) if month[i]}
        R[f"break_{side}"] = (len(old_set), len(new_set), len(old_set & new_set))

    # ---- the old setups (grab + gap) against the new break + gap blocks ------------------------------------
    book, A, bias = OLV.build(df)
    S = [s for s in OSE.detect(df, book, A, bias) if month[s["arm_bar"]]]
    gt = fvg.gaps(bars, "5m")
    tally = Counter()
    examples = {}
    for s in S:
        side = "buy" if s["dir"] == 1 else "sell"
        sweep, bos = idx[s["sweep_bar"]], idx[s["bos_bar"]]
        nb = structure.first_break_after(bars, "5m", side, sweep)
        if nb is not None and nb["bar"] == bos and nb["block_bar"] == idx[s["ob_bar"]] \
                and abs(nb["level"] - s["bos_level"]) < 1e-9:
            tally["A. break: the same candle, block and level"] += 1
        elif nb is not None and nb["bar"] < bos and _grab_kept_going(h, lo, s, idx.get_loc(nb["bar"])):
            tally["A. break: new one earlier - price then made a new extreme through the grab (old REGRAB)"] += 1
            examples.setdefault("grab kept going", (str(sweep), str(nb["bar"]), str(bos)))
        else:
            tally["A. break: differs, not explained"] += 1
            examples.setdefault("break differs", (str(sweep), None if nb is None else str(nb["bar"]), str(bos)))
        mv = fvg.gaps_of_move(bars, "5m", side, sweep, bos, gap_table=gt)
        key = (idx[s["fvg_bar"]], round(s["band_lo"], 2), round(s["band_hi"], 2))
        got = [(r.bar, round(r.lo, 2), round(r.hi, 2)) for r in mv.itertuples()]
        if key in got:
            row = mv.iloc[got.index(key)]
            if row["rank"] == s["gap_rank"]:
                tally["B. gap: the old gap has the same number among the move's unfilled gaps"] += 1
            elif _filled_on_next_candle(mv, h, lo, idx.get_loc(bos) + 1, side, below=row["rank"]):
                tally["B. gap: an earlier gap was filled on the candle AFTER the break - the old engine looked one "
                      "candle late and skipped it; the new block decides at the break's close"] += 1
                examples.setdefault("one candle late", (str(sweep), str(bos), str(row["bar"])))
            else:
                tally["B. gap: a different number"] += 1
                examples.setdefault("gap number differs", (str(sweep), str(bos), str(row["bar"]),
                                                            int(row["rank"]), int(s["gap_rank"])))
            tally["C. entry = his near edge" if abs(row["near"] - s["entry"]) < 1e-9 else "C. entry differs"] += 1
        else:
            tally["B. gap: not among the move's gaps"] += 1
            examples.setdefault("gap not among the move's gaps", (str(sweep), str(bos), str(idx[s["fvg_bar"]])))
        tally[f"D. the old engine bought gap number {s['gap_rank'] + 1} of the move"] += 1
    R["setups"] = len(S)
    R["setup_tally"] = dict(sorted(tally.items()))
    R["setup_examples"] = examples

    # ---- first visits: the old rule (a plain loop) ----------------------------------------------------------
    fv = fvg.first_visits(bars, "5m", gt)
    gm = gt[_in_month(gt["bar"])]
    same = 0
    for r in gm.itertuples():
        j = idx.get_loc(r.bar)
        k = next((k for k in range(j + 1, n) if (lo[k] <= r.near if r.side == "buy" else h[k] >= r.near)), None)
        row = fv[(fv.gap_bar == r.bar) & (fv.side == r.side)]
        same += (k is None and row.empty) or (k is not None and len(row) == 1 and row.iloc[0]["bar"] == idx[k])
    R["gaps_month"] = (len(gm), int((gm.side == "buy").sum()), int((gm.side == "sell").sum()))
    R["first_visit_same"] = (len(gm), int(same))

    # ---- sessions -----------------------------------------------------------------------------------------
    old_s = {}
    for L in OLV.session_levels(df):
        name = L["label"].split(" session ")[0].replace("NY", "New York")
        old_s[(name, int(L["born"]), L["kind"])] = L["price"]
    fin = sessions.final(bars, "5m")
    fin = fin[_in_month(fin["last_bar"])]
    match = total = 0
    for r in fin.itertuples():
        b = idx.get_loc(r.last_bar)
        for kind, price in (("session_high", r.high), ("session_low", r.low)):
            total += 1
            match += abs(old_s.get((r.session, b, kind), np.nan) - price) < 1e-9
    old_month = sum(1 for (nm, b, k) in old_s if month[b])
    R["sessions"] = (old_month, total, int(match))
    R["session_known_after_last_bar_min"] = sorted({int((r.known_at - r.last_bar) / pd.Timedelta(minutes=1))
                                                    for r in fin.itertuples()})

    # ---- ADR ------------------------------------------------------------------------------------------------
    A_old = OLV.adr_bands(df)
    nb_ = adr.adr_bars(bars, "5m")
    for col in ("day_open", "adr5", "adr10", "up_lo", "up_hi", "dn_hi", "dn_lo"):
        a, b = np.asarray(A_old[col], float)[month], nb_[col].to_numpy()[month]
        R[f"adr_{col}"] = int(np.sum(np.isclose(a, b, equal_nan=True)))
    ratio = nb_["ruler"].to_numpy()[month] / np.asarray(A_old["adr"], float)[month]
    R["adr_ruler_ratio"] = (float(np.nanmedian(ratio)), float(np.nanmin(ratio)), float(np.nanmax(ratio)))
    R["adr_ruler_values"] = (float(np.nanmedian(nb_["ruler"].to_numpy()[month])),
                             float(np.nanmedian(np.asarray(A_old["adr"], float)[month])))

    # ---- Fibonacci: old/build/fib_135.py's reading, on the same hour bars ----------------------------------
    h1 = load_bars("1h", WARM, MONTH[1])
    old_f = _fib_135_events(h1)
    new_f = fib.fib_levels(h1, "1h")
    m_old = {(e["side"], round(e["top"], 2), round(e["bot"], 2)) for e in old_f if _in_month([e["known"]])[0]}
    m_new = {(r.side, round(r.top, 2), round(r.bottom, 2)) for r in new_f.itertuples() if _in_month([r.known_at])[0]}
    R["fib"] = (len(m_old), len(m_new), len(m_old & m_new))
    return R


def _filled_on_next_candle(mv: pd.DataFrame, h, lo, nxt: int, side: str, below: int) -> bool:
    """Was one of the move's gaps ranked before `below` filled through (to its far edge) on the candle `nxt`?"""
    if nxt >= len(h):
        return False
    early = mv[(mv["rank"] >= 0) & (mv["rank"] < below)]
    if early.empty:
        return False
    return bool((lo[nxt] <= early["far"]).any() if side == "buy" else (h[nxt] >= early["far"]).any())


def _grab_kept_going(h, lo, s, new_break: int) -> bool:
    """Between the new block's first break and the old engine's break, did price make a new extreme through the grab
    (lower than every low since the sweep, for a buy)? That is when the old engine went back to waiting (REGRAB)."""
    a, b = int(s["sweep_bar"]), int(s["bos_bar"])
    if s["dir"] == 1:
        ext = lo[a:new_break + 1].min()
        return bool((lo[new_break + 1:b] < ext).any())
    ext = h[a:new_break + 1].max()
    return bool((h[new_break + 1:b] > ext).any())


def _fib_135_events(df: pd.DataFrame) -> list[dict]:
    """old/build/fib_135.py swings() and events(), copied (that script runs on import): its swing takes the high
    STRICTLY above the 2 before but only AT LEAST the 2 after."""
    H, L = df.high.to_numpy(), df.low.to_numpy()
    n = len(H)
    sh = [i for i in range(2, n - 2) if H[i] > H[i - 1] and H[i] > H[i - 2] and H[i] >= H[i + 1] and H[i] >= H[i + 2]]
    sl = [i for i in range(2, n - 2) if L[i] < L[i - 1] and L[i] < L[i - 2] and L[i] <= L[i + 1] and L[i] <= L[i + 2]]
    out = []
    for side, pts in (("buy", sh), ("sell", sl)):
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a < 2 or b + 2 >= n:
                continue
            if side == "buy":
                if H[b] <= H[a]:
                    continue
                top, bot = H[b], L[a + 1:b].min()
            else:
                if L[b] >= L[a]:
                    continue
                bot, top = L[b], H[a + 1:b].max()
            if top - bot <= 0:
                continue
            out.append(dict(side=side, top=top, bot=bot, known=df.index[b + 2] + pd.Timedelta(hours=1)))
    return out


def _pct(a: int, b: int) -> str:
    return f"{a} of {b}" + ("" if b == 0 else f" ({100 * a / b:.0f}%)")


def markdown(R: dict) -> str:
    t = R["setup_tally"]
    lines = [
        "# L6.1 — the new core blocks against the old engine, March 2019",
        "",
        "> Made by `python -m tests.compare_old_blocks` (L6.1 step 5). Both engines read the SAME bars: the new 5-minute",
        f"> bars (MT5 server clock), {R['bars_month']:,} of them from {R['first']} to {R['last']}; Nov 2018 – Feb 2019 is",
        "> warm-up only. The old code is `old/ml/agent/levels.py` and `setup_engine.py` as they last shipped; tests may import",
        "> it, `machine/` never does. **No money here** — this compares what the blocks SEE, not what trades earn.",
        "",
        "## Summary",
        "",
        "| block | old engine | new block | the same | why they differ |",
        "|---|---|---|---|---|",
        f"| swing highs | {R['swing_high'][0]} | {R['swing_high'][1]} | {R['swing_high'][2]} | — (same rule: 2 candles each "
        f"side, strictly). Stamped known at the close of the 2nd candle after: "
        f"{_pct(R['swing_known_right'][1], R['swing_known_right'][0])} (the whole Nov–Mar run) |",
        f"| swing lows | {R['swing_low'][0]} | {R['swing_low'][1]} | {R['swing_low'][2]} | — |",
        f"| sell order block known at each bar | {R['block_sell'][0]:,} bars | | {R['block_sell'][1]:,} bars | "
        + ("—" if R['block_sell'][0] == R['block_sell'][1] else f"first differences: {R['block_sell'][2]}") + " |",
        f"| buy order block known at each bar | {R['block_buy'][0]:,} bars | | {R['block_buy'][1]:,} bars | "
        + ("—" if R['block_buy'][0] == R['block_buy'][1] else f"first differences: {R['block_buy'][2]}") + " |",
        f"| first break of each sell block | {R['break_sell'][0]} | {R['break_sell'][1]} | {R['break_sell'][2]} | — |",
        f"| first break of each buy block | {R['break_buy'][0]} | {R['break_buy'][1]} | {R['break_buy'][2]} | — |",
        f"| gaps (3-candle) in the month | | {R['gaps_month'][0]} ({R['gaps_month'][1]} buy, {R['gaps_month'][2]} sell) "
        "| | the same formula as the old `detect()` |",
        f"| first visit back to the near edge | a plain loop | | {_pct(R['first_visit_same'][1], R['first_visit_same'][0])}"
        " | — |",
        f"| session highs / lows (final) | {R['sessions'][0]} | {R['sessions'][1]} | {R['sessions'][2]} | — (same windows;"
        f" the new one is known {R['session_known_after_last_bar_min']} minutes after its last candle OPENS = at the "
        "window's end) |",
    ]
    nbar = R["bars_month"]
    adr_same = min(R[f"adr_{c}"] for c in ("day_open", "adr5", "adr10", "up_lo", "up_hi", "dn_hi", "dn_lo"))
    lines += [
        f"| ADR: day open, 5-day, 10-day, both bands | per bar | per bar | {adr_same:,} of {nbar:,} bars | "
        + ("— (both on broker days)" if adr_same == nbar else "see below") + " |",
        f"| ADR as the RULER ('a share of a day's range') | UTC-day 10-day average, median ${R['adr_ruler_values'][1]:.2f}"
        f" | broker-day 10-day ADR, median ${R['adr_ruler_values'][0]:.2f} | | **new / old = "
        f"{R['adr_ruler_ratio'][0]:.3f} (median; {R['adr_ruler_ratio'][1]:.3f}–{R['adr_ruler_ratio'][2]:.3f})** — see 1 |",
        f"| Fibonacci moves (1-hour, known in March) | {R['fib'][0]} (fib_135.py) | {R['fib'][1]} | {R['fib'][2]} | see 2 |",
        "",
        "## The old setups (grab + gap, 5-minute) read with the new blocks",
        "",
        f"The old engine made **{R['setups']} setups** that armed in March 2019. For each one, the new blocks were asked the",
        "same questions from the old setup's own sweep candle: *which candle breaks structure?* "
        "(`structure.first_break_after`) and *which gaps belong to that move?* (`fvg.gaps_of_move`).",
        "",
        "| check | setups |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in t.items()]
    lines += [
        "",
        "Examples - the sweep candle first; then for 'grab kept going': the new first break, the old break; for 'one candle late': the break, the gap the old engine took instead:",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in R["setup_examples"].items()]
    lines += [
        "",
        "## What the differences are, and which one is his",
        "",
        "1. **The ruler.** The old engine kept its 'share of a day's range' ruler on **UTC** days after it moved his ADR "
        "to broker days (old `levels.adr_bands`, section 65i), so its tuned numbers kept their meaning. His fixed rule "
        "is broker days (`rules.py` broker_days_not_utc_days), so the new ruler is his 10-day ADR on broker days "
        f"(`settings.adr_ruler_days`, Claude's pick). In March 2019 it is **{R['adr_ruler_ratio'][0]:.2f}×** the old "
        "one. Every number written as a share of a day (the hammer's dip 0.12, equal highs 0.003, the trendline 0.005 …) "
        "therefore means a slightly bigger dollar amount in the new machine. Listed in `plan/work/rules-missing.md` "
        "(item R1).",
        "2. **Fibonacci.** Two readings of 'swing': fib_135.py (25 Sep) took a high strictly above the 2 before but only "
        "*at least* the 2 after; his glossary and `rules.py` say strictly on both sides. Equal highs next to each other "
        "count there and not here. The new block follows his glossary. "
        + (f"In March 2019 this changed nothing ({R['fib'][0]} = {R['fib'][1]} moves). " if R['fib'][0] == R['fib'][1]
           == R['fib'][2] else f"In March 2019: {R['fib'][2]} moves the same of {R['fib'][0]} / {R['fib'][1]}. ")
        + "Fibonacci itself is new to his files; F1–F3 are still open (strategy-understanding §66bz).",
        "3. **The grab that keeps going.** When price makes a new extreme after a small break, the old engine throws "
        "that break away and waits for the next one (`REGRAB = 'continue'`, his fixed rule "
        "a_grab_that_keeps_going_is_still_a_grab). `first_break_after` finds the first break only; the regrab walk "
        "belongs to the setup (L7b), which calls it again from the new extreme. Those setups show as 'break earlier'.",
        "4. **Which gap.** The old engine shipped `GAP_PICK = 'second'` (with the 'strong' check) and bought the 2nd gap "
        f"of the move in {t.get('D. the old engine bought gap number 2 of the move', 0)} of these setups. His newest "
        "words take the FIRST gap (four-boxes B1-18, his L4 answer, inbox 2026-09-28-0622): "
        "`settings.which_gap_of_the_move` = 'first', so those setups buy gap 1 in the new machine. The 2nd gap waits for "
        "his 'very strong' (four-boxes B2-03). `gaps_of_move` returns every gap of the move in order, so both readings "
        "can be run and compared.",
        "5. **Unknown ADR.** The old engine used $20 where the ADR was unknown (the first 10 days). The new block leaves "
        "it empty: no invented number.",
        "6. **One candle late.** The old engine chose the gap one candle AFTER the break (its loop reached the break's "
        "next candle first) and dropped a gap price went through on that candle. The new `gaps_of_move` decides at the "
        "break's own close - when a real order at the near edge would already be waiting, so that candle FILLS it "
        "(the tester fills from the next bar, L3). Where this happens the new machine has a trade the old one skipped. "
        "A gap filled through BEFORE that close is gone (`filled_before`, rank -1: his #68 'whether it is already "
        "filled'); one only visited is marked `visited_before` for the setup (L7b) to call spent.",
        "7. **No-future stamps.** Every new row carries `known_at` (the close of the last candle it needs). The old engine "
        "carried the same lag in its loops; the new one is checked by `guards.assert_no_future_events` on every block "
        "(tests/test_blocks.py), which also catches a row stamped one candle early.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    R = compare()
    text = markdown(R)
    print(text)
    if not a.no_write:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        keep = old[old.index(L62_MARK):] if L62_MARK in old else ""      # the L6.2 part (tests.compare_old_liquidity)
        tmp = OUT.with_suffix(".md.tmp")
        tmp.write_text(text.rstrip("\n") + "\n\n" + keep if keep else text, encoding="utf-8")
        tmp.replace(OUT)
        print(f"\nwritten: {OUT.relative_to(config.ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
