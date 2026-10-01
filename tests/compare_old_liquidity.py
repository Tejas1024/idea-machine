"""The L6.2 liquidity blocks against the old engine on one month of 2019 (the same test as L6.1).

    .venv\\Scripts\\python -m tests.compare_old_liquidity            prints, and writes the L6.2 part of
                                                                   plan/work/L6-differences.md
    .venv\\Scripts\\python -m tests.compare_old_liquidity --no-write  prints only

Tests may import the old code to compare (likhith.md R2); machine/ never does. Both engines read the SAME bars (the new
5-minute bars, given to the old code with its UTC index). The month is March 2019; November 2018 - February 2019 is
warm-up. No money - this compares what the blocks SEE.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

from machine import rules, settings
from machine.blocks import adr, equal_levels as EQ, liquidity as LQ
from machine.data.bars import load_bars
from tests.compare_old_blocks import L62_MARK, MONTH, OUT, WARM, _in_month, _old, _old_frame

OBVIOUS = ("equal_high", "equal_low", "session_high", "session_low")


def compare() -> dict:
    OLV, _ = _old()
    m5 = load_bars("5m", WARM, MONTH[1])
    h1 = load_bars("1h", WARM, MONTH[1])
    d1 = load_bars("1d", WARM, MONTH[1])
    w1 = load_bars("1w", "2018-01-01", MONTH[1])
    df = _old_frame(m5)
    idx = m5.index
    h, lo = m5["high"].to_numpy(float), m5["low"].to_numpy(float)
    month = _in_month(idx)
    old_ruler = np.where(np.isfinite(OLV.adr_bands(df)["adr"]), OLV.adr_bands(df)["adr"], 20.0)
    new_ruler = adr.adr_bars(m5, "5m")["ruler"].to_numpy(float)
    R: dict = {"bars_month": int(month.sum())}

    # ---- equal highs / lows on the 5-minute chart ------------------------------------------------------------
    sh, sl = OLV.swings_of(h, lo)
    old_eq = OLV.equal_levels_his(df, sh, sl, old_ruler)
    new_eq = EQ.equal_levels(m5, "5m")
    piv = int(rules.get("swing_lookback"))
    old_pairs = {(L["side"], int(L["born"]) - piv): int(L["meta"]["first_bar"]) for L in old_eq
                 if month[int(L["born"]) - piv]}
    new_pairs = {}
    for r in new_eq.itertuples():
        b = idx.get_loc(r.touch_bar)
        if month[b]:
            new_pairs[(+1 if r.side == "high" else -1, b)] = idx.get_loc(r.first_bar)
    tol = float(rules.get("what_equal_highs_and_lows_are")["tol_usd"])
    bounce = float(rules.get("what_equal_highs_and_lows_are")["bounce_adr_min"])
    tally, ex = Counter(), {}
    for key in sorted(set(old_pairs) | set(new_pairs)):
        side, b = key
        a_old, a_new = old_pairs.get(key), new_pairs.get(key)
        up = h if side > 0 else -lo
        dn = lo if side > 0 else -h
        if a_old is not None and a_new is not None:
            why = "A. the same pair" if a_old == a_new else \
                "B. the same touch; the new block names the OLDER level it belongs to (one level per first touch)"
        elif a_old is not None:
            gap = up[a_old] - up[b]
            dip = (up[a_old] - float(dn[a_old + 1:b].min())) if b > a_old + 1 else 0.0
            if gap > tol + 1e-9:
                why = f"C. old only: {gap:.2f} apart - more than his $0.06 (the old 0.003 of a day was wider that day)"
            elif dip + 1e-9 < bounce * new_ruler[b + piv]:
                why = "D. old only: the dip is under 0.08 of the NEW day's range (broker days, 1.23x the old)"
            else:
                why = "E. old only: the same dip as the level's last touch - not a new touch (3>2 counts real touches)"
        else:
            gap = up[a_new] - up[b]
            why = (f"F. new only: {gap:.2f} apart - inside his $0.06, outside 0.003 of the old day"
                   if gap > 0.003 * old_ruler[b + piv] + 1e-9 else "G. new only: the old engine paired it with a "
                                                                   "NEWER first touch that the new block counts as "
                                                                   "the same level")
        tally[why] += 1
        ex.setdefault(why, (str(idx[a_old if a_old is not None else a_new]), str(idx[b])))
    R["eq_old"], R["eq_new"] = len(old_pairs), len(new_pairs)
    R["eq_tally"], R["eq_examples"] = dict(sorted(tally.items())), ex

    # ---- the book: sessions + equal levels, their sweeps --------------------------------------------------------
    book, A, bias = OLV.build(df)
    new_book = LQ.obvious_liquidity(m5, h1, d1)
    new_sw = LQ.sweeps(m5, new_book)
    life = LQ.lifetimes(m5, new_book, new_sw)
    sw_by = new_sw.set_index("level_id")
    born_new = new_book[new_book["event"] == "born"].drop_duplicates("level_id")
    tally = Counter()
    for L in book.levels:
        if L["kind"] not in OBVIOUS or L.get("tf", "M5") != "M5" or not month[min(int(L["born"]), len(idx) - 1)]:
            continue
        if L["kind"].startswith("session"):
            cand = born_new[(born_new["kind"] == L["kind"]) & ((born_new["price"] - L["price"]).abs() < 1e-9)
                            & (born_new["known_at"] == idx[int(L["born"])] + pd.Timedelta(minutes=5))]
        else:
            cand = born_new[(born_new["kind"] == L["kind"]) & (born_new["tf"] == "5m")
                            & (born_new["first_bar"] == idx[int(L["meta"]["first_bar"])])]
        if not len(cand):
            tally[f"{L['kind'].split('_')[0]}: not in the new book (see the equal pairs above)"] += 1
            continue
        lid = cand.iloc[0]["level_id"]
        old_taken = int(L["died"]) if L.get("dead_by") == "swept" else None
        new_taken = idx.get_loc(sw_by.loc[lid, "bar"]) if lid in sw_by.index else None
        same = old_taken == new_taken
        tally[f"{L['kind'].split('_')[0]}: taken on the same candle" if same else
              f"{L['kind'].split('_')[0]}: taken on a different candle"] += 1
    R["sweep_tally"] = dict(sorted(tally.items()))

    # ---- "the London high" (L6-new): the old engine's levels LABELLED London against liquidity.of_session --------
    lon_new = LQ.of_session(born_new, "London")
    tally, seen = Counter(), set()
    for L in book.levels:
        if not (L["kind"].startswith("session_") and str(L["label"]).startswith("London session")
                and month[min(int(L["born"]), len(idx) - 1)]):
            continue
        side = "high" if L["side"] > 0 else "low"
        cand = lon_new[(lon_new["side"] == side) & ((lon_new["price"] - L["price"]).abs() < 1e-9)
                       & (lon_new["known_at"] == idx[int(L["born"])] + pd.Timedelta(minutes=5))]
        if not len(cand):
            tally[f"London {side}: the old engine's level is not a London level in the new book"] += 1
            continue
        lid = cand.iloc[0]["level_id"]
        seen.add(lid)
        old_taken = int(L["died"]) if L.get("dead_by") == "swept" else None
        new_taken = idx.get_loc(sw_by.loc[lid, "bar"]) if lid in sw_by.index else None
        how = "taken on the same candle" if old_taken is not None else "never taken in the data, in both"
        tally[f"London {side}: the same level, {how}" if old_taken == new_taken else
              f"London {side}: the same level, taken on a different candle"] += 1
    extra = lon_new[_in_month(lon_new["known_at"] - pd.Timedelta(minutes=5)) & ~lon_new["level_id"].isin(seen)]
    R["london"] = dict(sorted(tally.items()))
    R["london_new_only"] = int(len(extra))

    # ---- a level swept once is finished (#129): three readings on the same sweeps -------------------------------
    m = _in_month(new_sw["bar"]) & new_sw["kind"].isin(OBVIOUS)
    any_kind = LQ.sweeps(m5, new_book, same_kind=False).set_index("level_id")
    old_rule = _old_liq_once(new_sw, idx, old_ruler)
    R["fresh"] = {"sweeps of his obvious levels in the month": int(m.sum()),
                  "not fresh - new: same kind, $0.06 (his 'really equal')": int((~new_sw.loc[m, "fresh"]).sum()),
                  "not fresh - old liq_once: same kind, 0.05 of the old day (~$0.50)": int((~old_rule[m]).sum()),
                  "not fresh - any kind, $0.06 (measured only)":
                      int((~any_kind.loc[new_sw.loc[m, "level_id"], "fresh"]).sum())}

    # ---- recent first (#130) vs the old LevelBook.nearest -------------------------------------------------------
    same = total = same_order = total_order = 0
    diff = Counter()
    book.reset()
    starts, ends = life["start"].to_numpy(), life["end"].to_numpy()
    for i in np.flatnonzero(month)[::12]:
        book.advance(int(i))
        price = float(m5["close"].iat[i])
        cap = 3.0 * old_ruler[i]
        rows = life[(starts <= i + 1) & (ends > i)].assign(strong=lambda x: x["strong_from"] <= i + 1)
        olds = _as_rows(book.active(int(i), OBVIOUS), idx)
        for s_old, s_new in ((+1, "high"), (-1, "low")):
            o = book.nearest(int(i), s_old, price, max_n=1, kinds=OBVIOUS)
            n = LQ.recent_first(rows[(rows["price"] - price).abs() <= cap], s_new, price, max_n=1)
            po = round(o[0]["price"], 2) if o else None
            # the ORDER alone: recent_first on the old engine's own live levels must pick what nearest() picked
            oo = LQ.recent_first(olds[(olds["price"] - price).abs() <= cap], s_new, price, max_n=1) if len(olds) \
                else olds
            if o or len(oo):
                total_order += 1
                same_order += po == (round(float(oo.iloc[0]["price"]), 2) if len(oo) else None)
            if not o and not len(n):
                continue
            total += 1
            pn = round(float(n.iloc[0]["price"]), 2) if len(n) else None
            if po == pn:
                same += 1
            else:
                ko = o[0]["kind"] if o else "none"
                kn = n.iloc[0]["kind"] if len(n) else "none"
                diff[f"old {ko} / new {kn}"] += 1
    R["recent"] = (total, same, dict(diff.most_common(6)))
    R["recent_order"] = (total_order, same_order)

    # ---- swing roles (#91, #93) -------------------------------------------------------------------------------
    roles = LQ.swing_roles({"5m": m5, "1h": h1, "1d": d1, "1w": w1})
    old_n = Counter()
    for L in book.levels:
        if L["kind"] in ("swing_high", "swing_low") and month[min(int(L["born"]), len(idx) - 1)]:
            old_n["5m"] += 1
        elif L["kind"] in ("htf_high", "htf_low") and month[min(int(L["born"]), len(idx) - 1)]:
            old_n[{"H1": "1h", "D1": "1d", "W1": "1w"}[L["tf"]]] += 1
    rn = roles[_in_month(roles["known_at"])]
    R["roles"] = {tf: (old_n.get(tf, 0), int((rn["tf"] == tf).sum()),
                       rn.loc[rn["tf"] == tf, "can_start"].iloc[0] if (rn["tf"] == tf).any() else None,
                       rn.loc[rn["tf"] == tf, "target_when"].iloc[0] if (rn["tf"] == tf).any() else None)
                  for tf in ("5m", "1h", "1d", "1w")}
    return R


def _as_rows(levels: list[dict], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """The old engine's live levels as recent_first rows (its rank, its 'strong', born = when it became known)."""
    if not levels:
        return pd.DataFrame(columns=["level_id", "kind", "side", "price", "rank", "known_at", "can_start",
                                     "target_when", "strong"])
    return pd.DataFrame({"level_id": [str(id(L)) for L in levels], "kind": [L["kind"] for L in levels],
                         "side": ["high" if L["side"] > 0 else "low" for L in levels],
                         "price": [float(L["price"]) for L in levels],
                         "rank": [LQ.RANK[L["kind"]] for L in levels],
                         "known_at": [idx[min(int(L["born"]), len(idx) - 1)] for L in levels],
                         "can_start": True, "target_when": "always",
                         "strong": ["strong" in L["strength"] for L in levels]})


def _old_liq_once(sw: pd.DataFrame, idx: pd.DatetimeIndex, old_ruler: np.ndarray) -> np.ndarray:
    """The old engine's #129 rule (entries.liq_once) on the new sweeps: same kind, an earlier sweep within 576 candles
    at within 0.05 of the OLD day's range."""
    j = idx.get_indexer(pd.DatetimeIndex(sw["bar"]))
    fresh = np.ones(len(sw), bool)
    dead = int(settings.get("how_long_a_swept_level_stays_dead"))
    for kind in sw["kind"].unique():
        k = np.flatnonzero((sw["kind"] == kind).to_numpy() & sw["can_start"].to_numpy())
        kj, kp = j[k], sw["price"].to_numpy()[k]
        for q, jq, pq in zip(k, kj, kp):
            w = (kj < jq) & (kj >= jq - dead) & (np.abs(kp - pq) <= 0.05 * old_ruler[jq])
            fresh[q] = not w.any()
    return fresh


def markdown(R: dict) -> str:
    L = [L62_MARK, "", "# L6.2 — the liquidity blocks against the old engine, March 2019", "",
         "> Made by `python -m tests.compare_old_liquidity` (L6.2, the same test as L6.1). The same bars, warm-up "
         "Nov 2018 – Feb 2019. The old code: `old/ml/agent/levels.py` and `entries.py` as they last shipped. No money.",
         "", "## Equal highs / lows (5-minute chart), by the touch that makes the pair", "",
         f"Old engine {R['eq_old']} pairs, new block {R['eq_new']} in the month.", "",
         "| what | pairs | example (first touch, touch) |", "|---|---|---|"]
    for k, v in R["eq_tally"].items():
        L.append(f"| {k} | {v} | {R['eq_examples'][k][0]} → {R['eq_examples'][k][1]} |")
    L += ["", "## When each level is taken (the sweep, #95)", "", "| level | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in R["sweep_tally"].items()]
    L += ["", "## \"The London high\" (L6-new, 30 Sep 2026 — idea #3: \"sell when the London high is taken\")", "",
          "The old engine had no way to ask for one session's level; it only wrote the session's name into each "
          "level's label (\"London session high\"). The new block `liquidity.of_session(rows, \"London\")` keeps the "
          "levels whose window is London (12:30–15:30 India time, rules.py `what_a_session_high_or_low_is`, #85). "
          "The same levels, by the old label:", "", "| London level of the month | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in R["london"].items()]
    L += [f"| a London level only the new block has | {R['london_new_only']} |"]
    L += ["", "## A level swept once is finished (#129) — three readings on the same sweeps", "",
          "| reading | sweeps |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in R["fresh"].items()]
    t, s, d = R["recent"]
    to, so = R["recent_order"]
    L += ["", "## Which level comes first (#130) — the new recent_first vs the old LevelBook.nearest", "",
          f"Asked every 12th candle of the month, above and below the close, his obvious kinds only (equal + "
          f"session; trendlines come in L6b.1), within 3 days of range.", "",
          f"- **The order itself:** recent_first on the OLD engine's own live levels picks what the old nearest() "
          f"picked **{so} of {to}** times — the ranking (kind, then most recent, then nearest; \"3>2\") is the same.",
          f"- **With the new book:** {s} of {t} the same level. Every other one comes from the equal levels above "
          f"(a pair the new block has and the old did not, or the other way round), or - since L6b.1 put trendlines "
          f"in the new book - from a trendline (this count still asks the old engine for equal + session only):", "",
          "| old / new first level | times |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in d.items()]
    L += ["", "## Swing roles (#91, #93)", "",
          "| chart | old engine (its swings) | new block (his 2-candle swings) | may start a setup | a target |",
          "|---|---|---|---|---|"]
    for tf, (o, n, can, when) in R["roles"].items():
        L.append(f"| {tf} | {o} | {n} | {'yes' if can else 'no'} | "
                 f"{'only with the hour and day against (#91)' if when == 'higher_timeframe_against' else 'yes'} |")
    L += ["", "## What the differences are, and which one is his", "",
          "1. **How close is \"equal\".** His $0.06 in dollars (#90: the same on every chart) against the old 0.003 "
          "of a day. March 2019 was a narrow month (a day's range about $10-12), so 0.003 of a day was only about "
          "3-4 cents and the new block finds more pairs 3-6 cents apart. On wide days (2020, $30+) it is the other "
          "way round. His rule is the dollars.",
          "2. **One level per first touch.** The old engine made a level for every later touch, paired with its most "
          "recent partner; the new block joins a later touch to the OLDEST level it fits (the price the stops sit "
          "behind) and counts it as another touch (\"3>2\"). Same stops, fewer duplicate levels.",
          "3. **The ruler.** The dip of 0.08 of a day is measured with his ADR on broker days (L6.1, 1.23x the old UTC "
          "ruler), so a few small dips no longer count.",
          "4. **\"The same level\" (#129).** Same kind, as the old engine, but \"the same price\" is his really-equal "
          "$0.06, not 0.05 of a day (about $0.50): at $0.50 it called 48% of equal-level sweeps 2016–2020 already "
          "swept. Any kind was measured and dropped: it killed his yes #31 (L6-misses/README.md).",
          "5. **Swings.** The old engine's 5-minute target swings and its hour / day / week swings were zig-zag "
          "(\"structural\", 0.25 of a day) swings; the new block uses his glossary swing (2 candles each side), as "
          "L6.1 does. Their jobs are the same: 5-minute = a quick target only with the hour and day against; 1-hour = a "
          "target; day / week = may start a setup.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    R = compare()
    text = markdown(R)
    print(text)
    if not a.no_write:
        old = OUT.read_text(encoding="utf-8")
        head = old[:old.index(L62_MARK)] if L62_MARK in old else old.rstrip("\n") + "\n\n"
        tmp = OUT.with_suffix(".md.tmp")
        tmp.write_text(head + text, encoding="utf-8")
        tmp.replace(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
