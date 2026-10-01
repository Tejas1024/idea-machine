"""L6b.3's blocks against the old engine, on the SAME bars, 2016-2020 (the tune years - the hidden years are not read).

    .venv\\Scripts\\python -m tests.compare_old_bias            prints the tables, writes plan/work/L6b3-differences.md
    .venv\\Scripts\\python -m tests.compare_old_bias --no-write  prints only

Tests may import the old code to compare (likhith.md R2); machine/ never does. The old engine gets the new 5-minute
bars with its UTC index (as tests/compare_old_blocks.py does). Compared:
  - the bias on the hour, day and week: old levels.htf_bias_trade(same_detectors=True, at_target="hold") - the reading
    he confirmed (#74) - against machine/blocks/bias.py, candle by candle, and against his dated daily answers that fall
    in 2016-2020 (8 of his 12: the other 4 are 2021-2022 charts, in the hidden years - not opened here);
  - the real all-time high (old levels.all_time_high, the same 2011 seed);
  - the hammer (old setup_engine._is_hammer) on every 5-minute candle, with the old ruler and with the new one;
  - where a price sits in the day's ADR range (old setup_engine.adr_position vs premium_discount.share).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from machine import config
from machine.blocks import bias as B
from machine.blocks import hammer as HM
from machine.blocks import premium_discount as PD
from machine.data.bars import load_bars

YEARS = ("2016-01-01", "2020-12-31 23:59")
OUT = config.WORK_DIR / "L6b3-differences.md"
UP, DN, NEU = 1, -1, 0
# his dated daily answers (old/build/verify_daily_bias.py) that fall in 2016-2020: (question, panel, day, his word, blind)
ANSWERS = [("Q69", "A", "2017-12-13", NEU, True), ("Q69", "B", "2020-06-12", NEU, True),
           ("Q69", "D", "2018-08-31", UP, True), ("Q70", "A", "2020-02-20", UP, True),
           ("Q70", "C", "2019-08-13", NEU, True), ("Q73", "A", "2019-01-31", UP, False),
           ("Q73", "B", "2018-06-21", DN, False), ("Q73", "C", "2020-12-24", UP, False)]
HIDDEN = [("Q69", "C", "2021-05-10"), ("Q70", "B", "2022-03-23"), ("Q70", "D", "2021-11-12"),
          ("Q73", "D", "2022-06-16")]
WORD = {UP: "UP", DN: "DOWN", NEU: "NEUTRAL"}


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


def _server(idx, OLV) -> pd.DatetimeIndex:
    """The old engine's candle labels (UTC) -> server time without a zone, like the new bars."""
    return pd.DatetimeIndex(idx).tz_convert(OLV.SERVER_TZ).tz_localize(None)


def compare() -> dict:
    OLV, OSE = _old()
    m5 = load_bars("5m", *YEARS)
    df = _old_frame(m5)
    A = OLV.adr_bands(df)
    adr_old = np.where(np.isfinite(A["adr"]), A["adr"], 20.0)
    R: dict = {"bars": len(m5)}

    # ---- the bias, chart by chart ----------------------------------------------------------------------------
    for tf, rule in (("1h", "1h"), ("1d", "1D"), ("1w", "1W")):
        r, old = OLV.htf_bias_trade(df, rule, adr_old, **OLV.BIAS_TRADE_KW)
        old = pd.Series(np.asarray(old, int), index=_server(r.index, OLV))
        new_bars = load_bars(tf, *YEARS)
        new = B.htf_bias(new_bars, tf)["bias"].astype(int)
        both = old.index.intersection(new.index)
        o, n = old.reindex(both).to_numpy(), new.reindex(both).to_numpy()
        said = (o != 0) & (n != 0)
        R[f"bias_{tf}"] = dict(
            candles=len(both), only_old=len(old.index.difference(new.index)), only_new=len(new.index.difference(old.index)),
            same=int((o == n).sum()), both_said=int(said.sum()), both_said_same=int((o[said] == n[said]).sum()),
            old_share={WORD[k]: round(float(np.mean(o == k)), 2) for k in (UP, DN, NEU)},
            new_share={WORD[k]: round(float(np.mean(n == k)), 2) for k in (UP, DN, NEU)})
        if tf == "1d":
            day = {d.strftime("%Y-%m-%d"): (int(old.get(d, 0)), int(new.get(d, 0))) for d in both}
            rows = []
            for q, p, d, his, blind in ANSWERS:
                ov, nv = day[d]
                rows.append((f"{q} {p}", d, WORD[his] + ("" if blind else " (drawn)"), WORD[ov], WORD[nv],
                             ov == his, nv == his))
            R["answers"] = rows

    # ---- the real all-time high (daily) ---------------------------------------------------------------------
    d1 = load_bars("1d", *YEARS)
    old_ath = OLV.all_time_high(d1["high"].to_numpy())
    new_ath = B.all_time_high(d1, "1d")["ath"].to_numpy()
    R["ath"] = dict(days=len(d1), same=int(np.isclose(old_ath, new_ath).sum()),
                    new_high_days=len(B.new_highs(d1, "1d")),
                    first_new=str(B.new_highs(d1, "1d")["bar"].iloc[0].date()) if len(B.new_highs(d1, "1d")) else "-",
                    at_ath_days=int(B.all_time_high(d1, "1d")["at_ath"].sum()))

    # ---- the hammer ------------------------------------------------------------------------------------------
    o, h, lo, c = (m5[x].to_numpy() for x in ("open", "high", "low", "close"))
    new_ruler = HM.hammers(m5, "5m")
    new_old_ruler = HM.hammers(m5, "5m", ruler=adr_old)
    for side, dr in (("buy", 1), ("sell", -1)):
        old_h = np.array([OSE._is_hammer(o, h, lo, c, j, dr, adr_old[j]) for j in range(len(m5))])
        R[f"hammer_{side}"] = dict(old=int(old_h.sum()), new_same_ruler=int(new_old_ruler[side].sum()),
                                   same=int((old_h == new_old_ruler[side].to_numpy()).sum()),
                                   new_his_ruler=int(new_ruler[side].sum()))

    # ---- where a price sits in the day's ADR range ----------------------------------------------------------
    z = PD.zones(m5, "5m", "share_of_the_day")
    rng = np.random.default_rng(1)
    pick = rng.choice(np.flatnonzero(np.isfinite(z["premium_from"].to_numpy())), 2000, replace=False)
    diffs = []
    for i in pick:
        px = float(c[i])
        f_old, _ = OSE.adr_position(A, int(i), px)
        f_new = PD.share(px, z.iloc[i])
        if np.isfinite(f_old) and np.isfinite(f_new):
            diffs.append(abs(f_old - f_new))
    R["adr_share"] = dict(checked=len(diffs), same=int(np.sum(np.array(diffs) < 1e-9)),
                          max_diff=float(np.max(diffs)) if diffs else float("nan"))
    return R


def markdown(R: dict) -> str:
    L = ["# L6b.3 — the new blocks against the old engine (2016–2020)", "",
         "> Written by `python -m tests.compare_old_bias` (L6b.3). The same 5-minute bars go to both engines (the old one "
         "with its UTC index). The tune years only: 2021–2023 are not read. Counts, no money.", "",
         "## 1. The bias, candle by candle", "",
         "| chart | candles | the same | both say up / down, and agree | old: up · down · neutral | new: up · down · "
         "neutral |", "|---|---|---|---|---|---|"]
    for tf in ("1h", "1d", "1w"):
        b = R[f"bias_{tf}"]
        L.append(f"| {tf} | {b['candles']:,} | {b['same']:,} ({100 * b['same'] / max(b['candles'], 1):.0f}%) | "
                 f"{b['both_said_same']:,} of {b['both_said']:,} "
                 f"({100 * b['both_said_same'] / max(b['both_said'], 1):.0f}%) | "
                 f"{b['old_share']['UP']:.2f} · {b['old_share']['DOWN']:.2f} · {b['old_share']['NEUTRAL']:.2f} | "
                 f"{b['new_share']['UP']:.2f} · {b['new_share']['DOWN']:.2f} · {b['new_share']['NEUTRAL']:.2f} |")
    ok_old = sum(r[5] for r in R["answers"])
    ok_new = sum(r[6] for r in R["answers"])
    L += ["", "## 2. His daily answers in 2016–2020", "",
          "| chart | the last daily candle | his answer | old engine | new block |", "|---|---|---|---|---|"]
    for name, day, his, ov, nv, oo, no in R["answers"]:
        L.append(f"| {name} | {day} | {his} | {ov} {'✓' if oo else '✗'} | {nv} {'✓' if no else '✗'} |")
    L += ["", f"**Old engine {ok_old} of {len(R['answers'])} · new block {ok_new} of {len(R['answers'])}.** "
          f"His other 4 answers ({', '.join(f'{q} {p} {d}' for q, p, d in HIDDEN)}) are 2021–2022 charts, in the "
          "hidden years: not read here (the old project scored all 12: old default 9 of 12).", "",
          "## 3. The all-time high, the hammer, the ADR position", "",
          f"- **All-time high (daily):** {R['ath']['same']:,} of {R['ath']['days']:,} days the same as the old engine "
          f"(the same 2011 seed). New highs 2016–2020: **{R['ath']['new_high_days']} days**, the first "
          f"{R['ath']['first_new']}; days that close within 1.5% of it: {R['ath']['at_ath_days']}.",
          ]
    for side in ("buy", "sell"):
        hm = R[f"hammer_{side}"]
        L.append(f"- **Hammer ({side}):** with the old ruler {hm['same']:,} of {R['bars']:,} candles the same "
                 f"(old {hm['old']:,}, new {hm['new_same_ruler']:,}); with his broker-day ruler: {hm['new_his_ruler']:,} "
                 "hammers (the ruler is bigger, so fewer wicks reach 0.12 of it - L6-differences.md note 1).")
    s = R["adr_share"]
    L.append(f"- **Where a price sits in the day's ADR range:** {s['same']:,} of {s['checked']:,} random candles the "
             f"same as the old `adr_position` (largest difference {s['max_diff']:.2e}).")
    L += ["", "## 4. Why the bias differs", "",
          "1. **Swings.** The old bias used zig-zag swings (a turn of 0.25 of a day's range); the new one uses his "
          "glossary swing (2 candles each side) on every chart (`plan/work/rules-missing.md` R14) - more levels to take.",
          "2. **Trendlines.** The old bias also took 3-touch trendlines and a trendline break + retest (#69, #70 D); "
          "trendlines come in L6b.1, so the new bias does not see them yet.",
          "3. **The gap.** The old bias took any gap after the grab; the new one takes the first gap of the breaking "
          "move - his fixed rules (at most 6 candles after the break, #62 C, #101).",
          "4. **The day.** Both read broker days (the old engine's `DAY_CLOCK = 'server'`).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    R = compare()
    text = markdown(R)
    print(text)
    if not a.no_write:
        tmp = OUT.with_suffix(".md.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(OUT)
        print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
