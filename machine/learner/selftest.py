"""The slice rules' own exams (L13.1), the model's and the pooled view's (L13.2). tests/test_learner.py and
tests/test_learner_model.py run them small, with asserts; this prints the numbers.

    python -m machine.learner.selftest              # 400 no-edge ideas and the planted gaps (about 10 minutes)
    python -m machine.learner.selftest --sets 100   # quicker, rougher
    python -m machine.learner.selftest --model      # L13.2 A: the model's check and SHAP on made-up setups
    python -m machine.learner.selftest --pooled     # L13.2 B: any entry / only this idea on made-up trades

1. NOISE: made-up ideas with NO edge and 8 slices each. The rules (machine/learner/slices.py) may call "works best in
   / fails in" on at most 5 ideas in 100; each of the two corrected checks alone on at most 5 in 100.
2. PLANTED GAPS: the London trades get Rs 100 ... Rs 400 a trade more. How often "works best in London" is found, and
   how often another slice is claimed by mistake.

The made-up trades are shaped like idea #1's on 2016-2020 (612 trades, 82 winners): about 1 trade in 7 wins big
(about Rs 900), the rest lose small (about Rs 150); Rs 0 a trade on average; closed on random working days of
2016-2020. Such lopsided trades are the hard case for a test of a gap: a small part that happens to hold few winners
looks calm, so its gap looks surer than it is (what the first version of check 1 got wrong - slices.py).
The made-up features are random and know nothing of the money: every slice is pure noise unless a gap is planted.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from machine.learner import conditions as CN
from machine.learner import features as F
from machine.learner import slices as SL

EIGHT = ("session", "direction", "volatility", "pair_chart", "adr_zone", "recent", "hedge_side_gap",
         "friday")                                        # idea #1's eight slices


def fake_trades(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """n trades with no edge (see the module text): net_inr + exit_time, index = signal ids s0, s1 ..."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2020-12-30")
    t = pd.DatetimeIndex(rng.choice(days, n)) + pd.to_timedelta(rng.integers(0, 1380, n), unit="min")
    win = rng.random(n) < 0.14
    net = np.where(win, rng.normal(900, 500, n).clip(100), -rng.normal(150, 60, n).clip(20))
    return pd.DataFrame({"net_inr": net.round(2), "exit_time": t},
                        index=pd.Index([f"s{k}" for k in range(n)], name="signal"))


def fake_features(ids, seed: int = 0) -> pd.DataFrame:
    """A feature table with every column of features.COLUMNS, random and unrelated to any money."""
    rng = np.random.default_rng(seed + 1000)
    n = len(ids)

    def pick(xs, p=None):
        return rng.choice(np.array(xs, dtype=object), n, p=p)

    def u(a, b):
        return rng.uniform(a, b, n)

    side = ["with", "against", "neutral"]
    kinds = ["jobs", "inflation", "the Fed", "other news"]
    d = {
        "direction": pick(["buy", "sell"]), "ist_hour": rng.integers(0, 24, n),
        "session": pick(["Asia", "London", "New York", "in between"], [0.25, 0.25, 0.2, 0.3]),
        "weekday": pick(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
        "hours_to_week_end": u(0, 120), "adr": u(15, 30), "volatility_share": u(0, 1), "adr_used": u(0, 1.6),
        "recent_range": u(0, 0.5), "adr_zone": pick(["near the green", "middle", "near the red", "beyond a zone"]),
        "adr_share": u(-0.2, 1.2), "bias_hour": pick(side), "bias_day": pick(side), "bias_week": pick(side),
        "dial": pick(["with", "against", "mixed"]), "trend_days": u(0, 1), "at_ath": rng.random(n) < 0.1,
        "zone_at_entry": rng.random(n) < 0.3, "zone_at_target": rng.random(n) < 0.3,
        "zone_flip_at_entry": rng.random(n) < 0.2, "line_retest_at_entry": rng.random(n) < 0.3,
        "line_in_path": rng.random(n) < 0.3, "line_against": rng.random(n) < 0.5, "gap_past_hedge": rng.random(n) < 0.4,
        "liq_target_adr": u(0, 1), "liq_hedge_adr": u(0, 1), "equal_target_adr": u(0, 1), "hedge_adr": u(0.01, 0.2),
        "room_adr": u(0.05, 1), "reward": u(0.5, 9), "level_kind": pick(["equal", "trendline", "session", "swing"]),
        "level_chart": pick(["5m", "1h", "1d"], [0.6, 0.3, 0.1]), "level_rank": rng.integers(1, 5, n),
        "level_touches": rng.integers(1, 4, n), "level_day": pick(["today", "yesterday", "older"]),
        "level_age_days": u(0, 5), "grab_depth_adr": u(0, 0.3), "break_beyond_adr": u(0, 0.1),
        "gap_size_adr": u(0, 0.1), "break_to_gap": rng.integers(-2, 7, n), "leg_position": u(0, 1),
        "target_kind": pick(["equal", "session"]), "news_next_min": u(0, 600), "news_next_kind": pick(kinds),
        "news_next_colour": pick(["red", "orange"]), "news_last_min": u(0, 600), "news_last_kind": pick(kinds),
        "news_last_colour": pick(["red", "orange"]),
    }
    if set(d) != set(F.COLUMNS):
        raise RuntimeError(f"fake_features is out of step with features.COLUMNS: {sorted(set(d) ^ set(F.COLUMNS))}")
    return pd.DataFrame(d, index=pd.Index(list(ids), name="signal"))[list(F.COLUMNS)]


def eight(trades: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """The made-up features of these trades, and idea #1's eight slices on them: name -> (labels, parts)."""
    f = fake_features(trades.index, seed)
    return f, {n: (CN.BY_NAME[n].labels(f), CN.BY_NAME[n].groups) for n in EIGHT}


def noise(sets: int = 400, trades: int = 600, n_boot: int = 1000, n_shuffle: int = 1000,
          first_seed: int = 5000) -> dict:
    """`sets` no-edge ideas with 8 slices each: how many get a finding, and how many pass each corrected check."""
    out = {"sets": sets, "trades": trades, "finding": 0, "luck_with_holm": 0, "shuffle_with_holm": 0,
           "luck_under_5": 0, "shuffle_under_5": 0, "tests": 0}
    for k in range(sets):
        tr = fake_trades(trades, seed=first_seed + k)
        _, sl = eight(tr, first_seed + k)
        tests = {n: SL.test(tr, lab, groups, seed=k, n_boot=n_boot, n_shuffle=n_shuffle)
                 for n, (lab, groups) in sl.items()}
        judged = SL.judge(tests)
        out["finding"] += any(r["claim"] for r in judged.values())
        out["luck_with_holm"] += any(r["holm"]["passed"] for r in judged.values() if r["tested"])
        out["shuffle_with_holm"] += any(r["holm_shuffle"]["passed"] for r in judged.values() if r["tested"])
        out["luck_under_5"] += sum(t["p"] < 0.05 for t in tests.values())
        out["shuffle_under_5"] += sum(t["p_shuffle"] < 0.05 for t in tests.values())
        out["tests"] += len(tests)
    return out


def planted(gap: float, sets: int = 100, trades: int = 600, n_boot: int = 1000, n_shuffle: int = 1000,
            first_seed: int = 9000) -> dict:
    """The London trades get `gap` rupees a trade more: how often "works best in London" is found, and how often
    another slice is claimed by mistake."""
    found = wrong = 0
    for k in range(sets):
        tr = fake_trades(trades, seed=first_seed + k)
        f, sl = eight(tr, first_seed + k)
        tr.loc[f["session"] == "London", "net_inr"] += gap
        judged = SL.judge({n: SL.test(tr, lab, groups, seed=k, n_boot=n_boot, n_shuffle=n_shuffle)
                           for n, (lab, groups) in sl.items()})
        s = judged["session"]
        found += bool(s["claim"] == SL.WORKS_BEST and s["part"] == "London")
        wrong += any(r["claim"] for n, r in judged.items() if n != "session")
    return {"gap": gap, "sets": sets, "trades": trades, "found": found, "another_slice_claimed": wrong}


# ================================================================================================================
# L13.2 A: the model on made-up setups
# ================================================================================================================

def fake_setups(trades: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """fake_features with each setup's moment: 1 to 30 hours before its trade closed (every one inside 2016-2020)."""
    f = fake_features(trades.index, seed)
    rng = np.random.default_rng(seed + 77)
    t = pd.DatetimeIndex(trades["exit_time"]) - pd.to_timedelta(rng.integers(60, 1800, len(trades)), unit="min")
    f.insert(0, "known_at", t.where(t.year >= 2016, pd.Timestamp("2016-01-04 01:00")))
    return f


def model_found(r: dict) -> bool:
    """The planted London edge found: the check passed, the session is named, London pushed the guesses up."""
    if not (r["check"]["passed"] and "session" in r["named"]):
        return False
    s = next(d for d in r["shap"] if d["feature"] == "session")
    london = next((p for p in s.get("parts") or [] if p["part"] == "London"), None)
    return bool(london and london["mean_push"] > 0)


def model_exam(gap: float, sets: int, trades: int = 600, shuffles: int = 100, first_seed: int = 7000) -> dict:
    """`sets` made-up ideas; the London trades get `gap` rupees a trade more (0 = pure noise). How often the check
    passes, the London edge is found, and a noise thing is named."""
    from machine.learner import model as MD
    out = {"gap": gap, "sets": sets, "trades": trades, "shuffles": shuffles, "passed": 0, "found": 0,
           "noise_named": 0}
    for k in range(sets):
        tr = fake_trades(trades, seed=first_seed + k)
        f = fake_setups(tr, first_seed + k)
        if gap:
            tr.loc[f["session"] == "London", "net_inr"] += gap
        r = MD.analyse(f, tr, shuffles=shuffles)
        out["passed"] += r["check"]["passed"]
        out["found"] += model_found(r)
        out["noise_named"] += any(n not in ("session", "ist_hour") for n in r["named"])
    return out


# ================================================================================================================
# L13.2 B: the pooled view on made-up trades
# ================================================================================================================

def fake_pool(seed: int, *, this: int = 600, random: int = 3000, gap_this: float = 0.0, gap_random: float = 0.0):
    """This idea's made-up trades and its made-up random entries (features random, unrelated to money); the London
    trades of each get their gap."""
    from machine.learner import pooled as PL
    tr = fake_trades(this, seed=seed)
    f = fake_features(tr.index, seed)
    rt = fake_trades(random, seed=seed + 50_000)
    rt.index = pd.Index([f"r{i}" for i in range(random)], name="signal")
    rf = fake_features(rt.index, seed + 50_000)
    if gap_this:
        tr.loc[f["session"] == "London", "net_inr"] += gap_this
    if gap_random:
        rt.loc[rf["session"] == "London", "net_inr"] += gap_random
    return PL.Source("idea-fake", "a made-up idea", "v1", tr, f, rt, rf)


def pooled_exam(sets: int, *, gap_this: float = 0.0, gap_random: float = 0.0, n_boot: int = 1000,
                n_shuffle: int = 1000, first_seed: int = 3000) -> dict:
    """How often each question claims London, and how often it claims anything else."""
    from machine.learner import pooled as PL
    out = {"sets": sets, "gap_this": gap_this, "gap_random": gap_random, "any_london": 0, "only_london": 0,
           "any_other": 0, "only_other": 0}
    for k in range(sets):
        r = PL.analyse(fake_pool(first_seed + k, gap_this=gap_this, gap_random=gap_random), [], n_boot=n_boot,
                       n_shuffle=n_shuffle)
        out["any_london"] += any(c["condition"] == "session" and c["part"] == "London" for c in r["claims_any"])
        out["any_other"] += any(c["condition"] not in ("session", "new_york") for c in r["claims_any"])
        out["only_london"] += any(c["condition"] == "session" and c["part"] == "London" for c in r["claims_only"])
        out["only_other"] += any(c["condition"] not in ("session", "new_york") for c in r["claims_only"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.learner.selftest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", type=int, default=400, help="no-edge ideas in the noise exam (planted gaps: a quarter)")
    ap.add_argument("--model", action="store_true", help="L13.2 A: the model's own exam")
    ap.add_argument("--pooled", action="store_true", help="L13.2 B: the pooled view's own exam")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    if a.model:
        n = min(a.sets, 60)
        z = model_exam(0.0, n)
        print(f"MODEL, NOISE: {n} made-up ideas with no edge, 600 trades each, {z['shuffles']} shuffles: the check "
              f"passed {z['passed']} ({100 * z['passed'] / n:.1f} in 100; the rules allow 5) · a thing named "
              f"{z['noise_named']} ({time.time() - t0:.0f} s)", flush=True)
        for gap in (150, 200, 300):
            m = max(10, n // 3)
            p = model_exam(float(gap), m, first_seed=8000)
            print(f"MODEL, PLANTED: London +₹{gap} a trade: the check passed {p['passed']} of {m} · London found "
                  f"{p['found']} · a noise thing named {p['noise_named']} ({time.time() - t0:.0f} s)", flush=True)
        return 0
    if a.pooled:
        n = min(a.sets, 60)
        z = pooled_exam(n)
        print(f"POOLED, NOISE: {n} made-up pools (600 trades + 3,000 random entries): ANY ENTRY claimed something "
              f"{z['any_london'] + z['any_other']} times, ONLY THIS IDEA {z['only_london'] + z['only_other']} "
              f"({time.time() - t0:.0f} s)", flush=True)
        for gi, gr, what in ((300.0, 0.0, "this idea's London +₹300 only"),
                             (300.0, 300.0, "London +₹300 for this idea AND the random entries"),
                             (0.0, 100.0, "the random entries' London +₹100 only")):
            m = max(10, n // 3)
            p = pooled_exam(m, gap_this=gi, gap_random=gr, first_seed=4000)
            print(f"POOLED, PLANTED ({what}), {m} pools: ANY ENTRY says London {p['any_london']}, something else "
                  f"{p['any_other']} · ONLY THIS IDEA says London {p['only_london']}, something else "
                  f"{p['only_other']} ({time.time() - t0:.0f} s)", flush=True)
        return 0
    n = noise(a.sets)
    pct = lambda x, of: f"{100 * x / of:.1f} in 100"          # noqa: E731
    print(f"NOISE: {n['sets']} made-up ideas with no edge, {n['trades']} trades and 8 slices each "
          f"({time.time() - t0:.0f} s)")
    print(f"  a finding (all 3 checks): {n['finding']} = {pct(n['finding'], n['sets'])} (the rules allow 5)")
    print(f"  a slice through check 1 (luck test, corrected): {n['luck_with_holm']} = "
          f"{pct(n['luck_with_holm'], n['sets'])} (5 allowed)")
    print(f"  a slice through check 3 (shuffled labels, corrected): {n['shuffle_with_holm']} = "
          f"{pct(n['shuffle_with_holm'], n['sets'])} (5 allowed)")
    print(f"  one slice alone, no correction: luck test {pct(n['luck_under_5'], n['tests'])}, shuffled labels "
          f"{pct(n['shuffle_under_5'], n['tests'])} (5 each when the tests are honest)")
    print("PLANTED GAPS: the London trades (about 150 of 600) earn this much more a trade")
    for gap in (100, 150, 200, 300, 400):
        p = planted(gap, max(25, a.sets // 4))
        print(f"  ₹{gap}: found {p['found']} of {p['sets']} · another slice claimed by mistake "
              f"{p['another_slice_claimed']} of {p['sets']}   ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
