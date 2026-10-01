"""The L6b.2 blocks against the old engine, 2016-2020 (the tune years): strong zones (today / week on 5-minute candles,
the hour chart), their tiers, their lives (touches, roles, flips), the real zone break, and the zone entry (2019).

    .venv\\Scripts\\python -m tests.compare_old_zones            prints, and writes plan/work/L6b2-differences.md
    .venv\\Scripts\\python -m tests.compare_old_zones --no-write  prints only

Tests may import the old code to compare (likhith.md R2); machine/ never does. Both engines read the SAME bars (the new
5-minute bars, given to the old code with its UTC index). The zones are compared twice: with the OLD ruler (the old
engine's 10-day UTC-day range - the rule alone) and with the NEW ruler (his ADR on broker days - the machine as it
ships). No money: this compares what the blocks SEE.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

from machine import config, settings
from machine.blocks import adr, liquidity as LQ, zones as Z
from machine.data.bars import load_bars
from tests.compare_old_blocks import _old, _old_frame

YEARS = ("2016-01-01", "2020-12-31 23:59")
ENTRY_YEAR = ("2018-11-01", "2019-12-31 23:59")      # the zone entry needs the old engine's whole book: one year
OUT = config.WORK_DIR / "L6b2-differences.md"
OLD_NUMBERS = {"zone_bin_adr": 0.0375}          # the old engine's price step, on its own ruler (ours: the same dollars)
OLD_WAY = {"zone_spent_role_waits_for_the_break": False, **OLD_NUMBERS}


def _entries():
    agent = str(config.ROOT / "old" / "ml" / "agent")
    if agent not in sys.path:
        sys.path.insert(0, agent)
    import entries as old_entries          # noqa: E402 - the old engine, for comparing only
    return old_entries


def _key(window, day, lo, hi) -> tuple:
    return (window, pd.Timestamp(day).normalize(), round(float(lo), 2), round(float(hi), 2))


def _old_zones(OLV, df, h_df, old_r, idx):
    """The old engine's three windows: today and week exactly as levels.build makes them; the hour window as
    build/zones_cache.py makes it (on HOUR candles - levels.build counted 5-minute ones there, 47a)."""
    days = OLV.server_days(df.index)
    out = []
    for tag, wd, life in (("today", 1, None), ("this week", 5, 5)):
        for Zd in OLV.tpo_zones(df, old_r, window_days=wd, tf="M5", tag=tag, life_days=life):
            Zd["_window"] = "today" if tag == "today" else "week"
            Zd["_day"] = days[int(Zd["born"]) - 1].tz_convert(OLV.SERVER_TZ).tz_localize(None)
            out.append(Zd)
    pos = np.clip(np.searchsorted(df.index.values, h_df.index.values), 0, len(df) - 1)
    hdays = OLV.server_days(h_df.index)
    hz = OLV.tpo_zones(h_df, old_r[pos], window_days=20, tf="H1", tag="1H month", life_days=20)
    for Zd in hz:
        Zd["_window"] = "hour"
        Zd["_day"] = hdays[int(Zd["born"]) - 1].tz_convert(OLV.SERVER_TZ).tz_localize(None)
        out.append(Zd)
    return out


def _old_break(c, n, Zd, ZB_FROM_BARS=576, frac=0.20):
    """The old engine's zone break, pass 1 of entries.type_zone_break (ZB_EXT_ZONE = 0, ZB_THROUGH = True), copied so it
    runs on one zone: [(candle, direction)]."""
    lo, hi = float(Zd["meta"]["band_lo"]), float(Zd["meta"]["band_hi"])
    height = hi - lo
    b0, b1 = int(Zd["born"]) + 1, min(int(Zd["died"]), n)
    out = []
    if height <= 0 or b1 - b0 < 3:
        return out
    for dr in (1, -1):
        inside = (c[b0 - 1:b1 - 1] >= lo) & (c[b0 - 1:b1 - 1] <= hi)
        thru = (c[b0:b1] > hi) if dr == 1 else (c[b0:b1] < lo)
        ok = np.flatnonzero(inside & thru)
        if not ok.size:
            continue
        b = int(b0 + ok[0])
        w = c[max(0, b - ZB_FROM_BARS):b]
        o = np.flatnonzero((w > hi) | (w < lo))
        if not o.size or ((w[o[-1]] > hi) if dr == 1 else (w[o[-1]] < lo)):
            continue
        deep = (c[b] - hi) if dr == 1 else (lo - c[b])
        while deep < frac * height:
            nxt = np.flatnonzero(thru[b - b0 + 1:])
            if not nxt.size:
                b = None
                break
            b = int(b + 1 + nxt[0])
            deep = (c[b] - hi) if dr == 1 else (lo - c[b])
        if b is not None:
            out.append((b, dr))
    return out


def compare() -> dict:
    OLV, _ = _old()
    m5 = load_bars("5m", *YEARS)
    h1 = load_bars("1h", *YEARS)
    df = _old_frame(m5)
    idx = m5.index
    n = len(m5)
    c = m5["close"].to_numpy(float)
    A = OLV.adr_bands(df)["adr"]
    old_r = np.where(np.isfinite(A), A, 20.0)
    new_r = adr.adr_bars(m5, "5m")["ruler"].to_numpy(float)
    # the old engine reads the ruler on the zone's first candle (the next day's); ours on the window's last candle
    old_r_next = np.r_[old_r[1:], old_r[-1]]
    h_df = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    R: dict = {"bars": n, "ruler_ratio": float(np.nanmedian(new_r / old_r))}

    # ---- the zones ----------------------------------------------------------------------------------------------
    old = _old_zones(OLV, df, h_df, old_r, idx)
    okeys = {}
    for Zd in old:
        okeys[_key(Zd["_window"], Zd["_day"], Zd["meta"]["band_lo"], Zd["meta"]["band_hi"])] = Zd
    with settings.using(OLD_NUMBERS):
        new_old_r = Z.strong_zones(m5, h1, ruler=old_r_next, tiers=(1, 2, 3))
    new_ship = Z.strong_zones(m5, h1, tiers=(1, 2, 3))
    R["zones"] = {}
    for name, t in (("old ruler (the rule alone)", new_old_r), ("new ruler (what ships)", new_ship)):
        nkeys = {_key(r.window, r.known_at - pd.Timedelta(days=1), r.lo, r.hi): r for r in t.itertuples()}
        per = {}
        for w in ("today", "week", "hour"):
            o = {k for k in okeys if k[0] == w}
            nw = {k for k in nkeys if k[0] == w}
            same = o & nw
            tier_same = sum(1 for k in same if int(okeys[k]["meta"]["tier"]) == int(nkeys[k].tier))
            calm_diff = [abs(float(okeys[k]["meta"]["calm"]) - float(nkeys[k].calm)) for k in same
                         if np.isfinite(okeys[k]["meta"]["calm"]) and np.isfinite(nkeys[k].calm)]
            per[w] = dict(old=len(o), new=len(nw), same=len(same), tier_same=tier_same,
                          calm_max_diff=float(max(calm_diff)) if calm_diff else 0.0)
        R["zones"][name] = per
    tiers = new_ship.groupby(["window", "tier"]).size()
    R["ship_tiers"] = {f"{w} tier {t}": int(v) for (w, t), v in tiers.items()}
    months = (idx[-1] - idx[0]).days / 30.44
    R["tier1_a_month"] = float((new_ship["tier"] == 1).sum() / months)
    R["old_tier_counts"] = dict(Counter(f"{Zd['_window']} tier {Zd['meta']['tier']}" for Zd in old))

    # ---- lives: touches, roles, flips (the 5-minute windows, the old ruler, the old way) ------------------------------
    m5_old = [Zd for Zd in old if Zd["_window"] in ("today", "week")]
    OLV.mark_zone_deaths(m5_old, df, old_r)
    nk = {_key(r.window, r.known_at - pd.Timedelta(days=1), r.lo, r.hi): r.zone_id for r in new_old_r.itertuples()}
    matched = [(Zd, nk[k]) for Zd in m5_old if (k := _key(Zd["_window"], Zd["_day"], Zd["meta"]["band_lo"],
                                                          Zd["meta"]["band_hi"])) in nk]
    sub = new_old_r[new_old_r["zone_id"].isin({z for _, z in matched})]
    with settings.using(OLD_WAY):
        life_old_way = Z.lives(m5, sub, ruler=old_r).set_index("zone_id")
    life_ship = Z.lives(m5, sub, ruler=old_r).set_index("zone_id")
    tally, ex = Counter(), {}
    for Zd, zid in matched:
        L = life_old_way.loc[zid]
        od, nd = int(Zd["died"]), int(L["death"])
        if od == nd and Zd["dead_by"] == L["dead_by"]:
            tally["the same end, the same reason"] += 1
            continue
        first_new = L["touches"][0][0] if L["touches"] else None
        if first_new == int(L["born"]):
            why = "a touch on the zone's first candle (the old engine looked from its second candle)"
        elif Zd["dead_by"] == L["dead_by"] == "expired" and od - nd == 1:
            why = "its days ran out one candle apart (the old engine counted them from its second candle)"
        else:
            why = f"other: old {Zd['dead_by']} / new {L['dead_by']}"
        tally[why] += 1
        ex.setdefault(why, (str(idx[min(int(L['born']), n - 1)]), Zd["dead_by"], L["dead_by"]))
    R["lives_matched"] = len(matched)
    R["lives_tally"], R["lives_ex"] = dict(tally), ex
    R["dead_by_old"] = dict(Counter(Zd["dead_by"] for Zd, _ in matched))
    R["dead_by_old_way"] = dict(Counter(life_old_way["dead_by"]))
    R["dead_by_ship"] = dict(Counter(life_ship["dead_by"]))
    moved = (life_ship["death"] != life_old_way["death"])
    R["his78_changed"] = int(moved.sum())
    R["his78_later"] = int((life_ship["death"] > life_old_way["death"]).sum())

    # ---- the real break (the same zones, the same lives) -------------------------------------------------------------
    brk_new = Z.breaks(m5, life_old_way.reset_index())
    newb = {(r.zone_id, 1 if r.direction == "buy" else -1): idx.get_loc(r.bar) for r in brk_new.itertuples()}
    oldb = {}
    for Zd, zid in matched:
        for b, dr in _old_break(c, n, Zd):
            oldb[(zid, dr)] = b
    tally, ex = Counter(), {}
    for k in sorted(set(oldb) | set(newb)):
        o_, n_ = oldb.get(k), newb.get(k)
        if o_ == n_:
            why = "the same candle"
        elif o_ is None:
            why = "new only"
        elif n_ is None:
            why = "old only"
        else:
            why = "both, another candle"
        tally[why] += 1
        ex.setdefault(why, (k[0], "buy" if k[1] == 1 else "sell", str(idx[o_]) if o_ is not None else "-",
                            str(idx[n_]) if n_ is not None else "-"))
    R["breaks_tally"], R["breaks_ex"] = dict(tally), ex
    R["breaks_old"], R["breaks_new"] = len(oldb), len(newb)
    # why they differ: replay the old engine's first step on each difference
    reasons, back = Counter(), int(settings.get("zone_side_lookback_bars"))
    for k in sorted(set(oldb) | set(newb)):
        o_, n_ = oldb.get(k), newb.get(k)
        if o_ == n_:
            continue
        zid, dr = k
        L = life_old_way.loc[zid]
        lo_, hi_, b = float(L["lo"]), float(L["hi"]), int(L["born"])
        if n_ is None:
            first = L["touches"][0][0] if L["touches"] else None
            reasons["old only: the zone ended sooner - a touch on its first candle (the old engine looked from its "
                    "second)" if first == b else "old only: other"] += 1
            continue
        if n_ == b:
            reasons["new: the break is the zone's first candle (the old engine looked from its second)"] += 1
            continue
        e = min(int(L["death"]), n)
        ins = (c[b:e - 1] >= lo_) & (c[b:e - 1] <= hi_)
        thru = (c[b + 1:e] > hi_) if dr == 1 else (c[b + 1:e] < lo_)
        ok = np.flatnonzero(ins & thru)
        if not ok.size:
            reasons["new: one candle went from the other side straight through (the old engine needed a close "
                    "inside the zone first)"] += 1
            continue
        x = int(b + 1 + ok[0])
        w = c[max(0, x - back):x]
        out = np.flatnonzero((w > hi_) | (w < lo_))
        if out.size and ((w[out[-1]] > hi_) if dr == 1 else (w[out[-1]] < lo_)):
            reasons["new: the old engine's first crossing was the zone HOLDING, and it never looked again that way; "
                    "a later break through is found"] += 1
        else:
            reasons["new: other"] += 1
    R["breaks_new_only_why"] = dict(reasons)

    # ---- the zone entry: 2019 against old entries.type_zone (hour + day and the unfilled-gap rule off on both) --------
    R["entry"] = _compare_entries(OLV)
    return R


def _compare_entries(OLV) -> dict:
    EN = _entries()
    m5 = load_bars("5m", *ENTRY_YEAR)
    h1 = load_bars("1h", *ENTRY_YEAR)
    d1 = load_bars("1d", *ENTRY_YEAR)
    df = _old_frame(m5)
    idx = m5.index
    in_year = np.asarray(idx >= pd.Timestamp("2019-01-01"))
    book, A, bias = OLV.build(df)
    old = [s for s in EN.type_zone(df, book, A, bias, need_hd=False, gap_rule=False) if in_year[int(s["sweep_bar"])]]
    ok = {(int(s["sweep_bar"]), "buy" if s["dir"] == 1 else "sell"): s for s in old}
    oband = {(round(z["meta"]["band_lo"], 2), round(z["meta"]["band_hi"], 2)) for z in book.zones}
    bk = LQ.book(m5, h1, d1)
    sw = LQ.sweeps(m5, bk)
    out = {"old": len(ok)}
    # the rule alone: the old engine's own zones (its ruler, its lives), our sweeps; then everything as it ships
    old_r = np.asarray(A["adr"], dtype=float)
    old_r = np.where(np.isfinite(old_r), old_r, 20.0)
    with settings.using(OLD_WAY):
        zs_old = Z.strong_zones(m5, h1, ruler=np.r_[old_r[1:], old_r[-1]])
        life_old = Z.lives(m5, zs_old, ruler=old_r)
        rule_alone = Z.entries(m5, life_old, sw, ruler=old_r)
    ship = Z.entries(m5, Z.lives(m5, Z.strong_zones(m5, h1)), sw)
    zband = {(round(a, 2), round(b, 2)) for a, b in zip(zs_old["lo"], zs_old["hi"])}
    for name, new in (("rule alone (old zones and ruler)", rule_alone), ("as it ships", ship)):
        new = new[np.asarray(pd.DatetimeIndex(new["bar"]) >= pd.Timestamp("2019-01-01"))]
        nk = {(idx.get_loc(r.bar), r.direction): r for r in new.itertuples()}
        same = set(ok) & set(nk)
        band = sum(1 for k in same if abs(ok[k]["zone_lo"] - nk[k].lo) < 0.01
                   and abs(ok[k]["zone_hi"] - nk[k].hi) < 0.01)
        why = Counter()
        for k in set(ok) - set(nk):
            s = ok[k]
            lv = s.get("sweep_kind", "")
            hit = sw[(sw["bar"] == idx[k[0]]) & (sw["kind"] == lv)]
            if s.get("zone_tag") == "1H month" and (round(s["zone_lo"], 2), round(s["zone_hi"], 2)) not in zband:
                why["old only: an 'hour' zone the old engine counted on 5-minute candles (47a) - ours count hour "
                    "candles"] += 1
            elif not len(hit):
                why["old only: no sweep of that kind on that candle in the new book (L6.2's levels)"] += 1
            elif not bool(hit["fresh"].any()):
                why["old only: the level was already swept in the last two days (#129 - not fresh)"] += 1
            elif any(d == k[1] and j2 < k[0] and abs(r.lo - s["zone_lo"]) < 0.01 for (j2, d), r in nk.items()):
                why["old only: the same zone was armed earlier on another take"] += 1
            else:
                why["old only: other"] += 1
        for k in set(nk) - set(ok):
            r = nk[k]
            if any(d == k[1] and abs(s["zone_lo"] - r.lo) < 0.01 for (j2, d), s in ok.items()):
                why["new only: the same zone, armed on another take (every take arms; the tester keeps one order, #145 - the old engine skipped a take while its first order waited)"] += 1
            elif r.window == "hour" and (round(r.lo, 2), round(r.hi, 2)) not in oband:
                why["new only: an hour zone the old engine never had (ours count hour candles, 47a)"] += 1
            elif not any(abs(z0 - r.lo) < 0.01 for z0, _ in zband) or name == "as it ships":
                why["new only: a zone or take the old engine did not have"] += 1
            else:
                why["new only: other"] += 1
        out[name] = dict(new=len(nk), same_candle=len(same), same_band=band, why=dict(why),
                         old_only=[str(idx[k[0]]) + " " + k[1] for k in sorted(set(ok) - set(nk))[:4]],
                         new_only=[str(idx[k[0]]) + " " + k[1] for k in sorted(set(nk) - set(ok))[:4]],
                         tiers=dict(Counter(int(t) for t in new["tier"])), kinds=dict(Counter(new["level_kind"])))
    return out


def write(R: dict) -> str:
    L = ["# L6b.2 - strong zones against the old engine (2016-2020, the tune years)", "",
         "> Made by `tests/compare_old_zones.py` (both engines read the same 5-minute bars). No money - this compares "
         "what the blocks SEE. The hidden years 2021-2023 were not read.", "",
         f"The new ruler (his ADR on broker days) is **{R['ruler_ratio']:.2f}x** the old one (UTC days), so every zone "
         "is compared twice: on the OLD ruler (the rule alone) and on the NEW one (the machine as it ships).", "",
         "## 1. The zones (time at price)", "",
         "| ruler | window | old | new | the same band on the same day | same tier | biggest calm difference |",
         "|---|---|---|---|---|---|---|"]
    for name, per in R["zones"].items():
        for w, v in per.items():
            L.append(f"| {name} | {w} | {v['old']:,} | {v['new']:,} | {v['same']:,} | {v['tier_same']:,} | "
                     f"{v['calm_max_diff']:.3f} |")
    L += ["", f"As it ships (all tiers): {', '.join(f'{k} {v:,}' for k, v in R['ship_tiers'].items())}. "
          f"His main, quiet zone: **{R['tier1_a_month']:.1f} a month** (his #81: about 5 a month, 'that is what makes "
          "it special').", "", f"The old engine's tiers: {', '.join(f'{k} {v:,}' for k, v in sorted(R['old_tier_counts'].items()))}.",
          "", "## 2. Their lives (5-minute windows, old ruler)", "",
          f"{R['lives_matched']:,} zones both engines built. With the old way (settings "
          "zone_spent_role_waits_for_the_break = False):", ""]
    for k, v in sorted(R["lives_tally"].items(), key=lambda x: -x[1]):
        ex = R["lives_ex"].get(k)
        L.append(f"- {k}: **{v:,}**" + (f" (e.g. born {ex[0]}: old {ex[1]}, new {ex[2]})" if ex else ""))
    L += ["", f"How they end - old: {R['dead_by_old']}; new, the old way: {R['dead_by_old_way']}; "
          f"new, as it ships: {R['dead_by_ship']}.", "",
          f"His #78 ('It will only if it breaks the zone, then it will actually support'): a zone whose used side is "
          f"tested again now ends only if price does not break it. That moves the end of **{R['his78_changed']:,}** "
          f"of these zones ({R['his78_later']:,} later).", "",
          "## 3. The real break (#97, #98) - the same zones, the same lives", "",
          f"Old {R['breaks_old']:,}, new {R['breaks_new']:,}.", ""]
    for k, v in sorted(R["breaks_tally"].items(), key=lambda x: -x[1]):
        ex = R["breaks_ex"].get(k)
        L.append(f"- {k}: **{v:,}**" + (f" (e.g. {ex[0]} {ex[1]}: old {ex[2]}, new {ex[3]})" if ex else ""))
    if R["breaks_new_only_why"]:
        L += ["", "Why they differ (the old engine's first step replayed on each):"]
        L += [f"- {k}: {v:,}" for k, v in R["breaks_new_only_why"].items()]
    e = R["entry"]
    L += ["", "## 4. The zone entry (#84, #86) - 2019, hour + day and the unfilled gap off on both", "",
          f"The old engine armed **{e['old']:,}**.", "",
          "| new side | armed | the same candle and way | of those, the same zone |", "|---|---|---|---|"]
    for name in ("rule alone (old zones and ruler)", "as it ships"):
        v = e[name]
        L.append(f"| {name} | {v['new']:,} | {v['same_candle']:,} | {v['same_band']:,} |")
    for name in ("rule alone (old zones and ruler)", "as it ships"):
        v = e[name]
        L += ["", f"**{name}:** " + "; ".join(f"{k} {n:,}" for k, n in sorted(v["why"].items(), key=lambda x: -x[1]))
              + f". By tier {v['tiers']}; by the level taken {v['kinds']}. Old only e.g. {', '.join(v['old_only'])}; "
              f"new only e.g. {', '.join(v['new_only'])}."]
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    R = compare()
    text = write(R)
    print(text)
    if not a.no_write:
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(OUT)
        print(f"written: {OUT}")


if __name__ == "__main__":
    main()
