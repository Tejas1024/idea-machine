"""The referee's own exams (L5). tests/test_referee.py runs them with asserts; this prints the numbers.

    python -m machine.referee.selftest            # noise and planted edges (synthetic trades, about a minute)
    python -m machine.referee.selftest --monday   # + the Monday rule on real gold 2016-2020 (needs the price data)

1. NOISE: 100 seeds, each an idea with 20 versions and NO edge. Each seed goes the real way: the 20 versions are
   tried on the tune years, one is chosen (Deflated Sharpe), the check years open once for it, it is judged.
   "works" must come out at most 5 times in 100. Also every one of the 2,000 versions judged on its own.
   Two kinds of noise: the realistic one (no edge, so it loses the Rs 40 cost) and the hardest one (exactly Rs 0 a
   trade after cost — the edge of "no edge").
2. PLANTED EDGES: Rs 50 and Rs 100 a trade planted at 300 trades; the share caught.
3. THE MONDAY RULE (real gold): buy every Monday 10:00, stop $5, target $5 (the L3 smoke run) — no edge; the answer
   must be "chance".

The synthetic trades look like the Monday rule's: about $5 won or lost a trade at 0.01 lot (about Rs 400 either way
at Rs 80 a dollar), in random weeks of the judged years. The random-entry runs and fake levels are drawn from the
same no-edge trades, because a noise idea IS no better than random entries.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from machine import config
from machine.referee import power, stats, verdict
from machine.referee.fake_levels import FakeRun
from machine.referee.fake_levels import summary as fake_summary
from machine.referee.random_entries import Run
from machine.referee.random_entries import summary as random_summary

INR_PER_USD = 80.0
N_TRADES = 300


def fake_trades(n: int, mean: float, rng: np.random.Generator, years) -> pd.DataFrame:
    """n trades: won or lost $4-$6 at 0.01 lot (x Rs 80) on a fair coin, plus `mean` rupees a trade (the planted
    edge, or no edge), closing at random times. The TRUE mean is `mean`; each sample's own mean varies around it
    (forcing the sample mean to exactly `mean` once made every planted edge "caught" — no luck left to test)."""
    size = rng.uniform(4.0, 6.0, n) * INR_PER_USD
    side = rng.choice([-1.0, 1.0], n)
    x = side * size + mean
    start = pd.Timestamp(f"{min(years)}-01-01").value
    end = pd.Timestamp(f"{max(years)}-12-31 23:00").value
    times = pd.to_datetime(np.sort(rng.integers(start, end, n)))
    return pd.DataFrame({"net_inr": x, "exit_time": times})


def fake_evidence(trades: pd.DataFrame, rng: np.random.Generator, noise_mean: float, needed_edge_caught: float,
                  runs: int = config.RANDOM_RUNS) -> verdict.Evidence:
    """The evidence for a synthetic idea: its random runs and fake levels are no-edge trades of the same kind."""
    summ = stats.summary(trades)
    rand = []
    for _ in range(runs):
        x = rng.choice([-1.0, 1.0], len(trades)) * rng.uniform(4.0, 6.0, len(trades)) * INR_PER_USD + noise_mean
        rand.append(Run(len(x), len(x), float(x.sum()), float(x.mean()), x, np.array([])))
    fakes = []
    for step in config.FAKE_LEVEL_STEPS:
        x = rng.choice([-1.0, 1.0], len(trades)) * rng.uniform(4.0, 6.0, len(trades)) * INR_PER_USD + noise_mean
        fakes.append(FakeRun(step, len(x), float(x.sum()), float(x.mean())))
    p = stats.bootstrap_p(trades["net_inr"], stats.week_ids(trades["exit_time"]), seed=rng.integers(1 << 62))
    return verdict.Evidence(trades=summ["trades"], per_trade=summ["per_trade"], p=p,
                            random=random_summary(rand, summ["per_trade"]), fake=fake_summary(fakes, summ["per_trade"]),
                            without_best=stats.without_best(trades), seen=False,
                            needed_edge_caught=needed_edge_caught)


def needed_power(noise_mean: float = 0.0, n: int = N_TRADES) -> float:
    rng = np.random.default_rng(12345)
    noise = fake_trades(20_000, noise_mean, rng, config.CHECK_YEARS)["net_inr"].to_numpy()
    return power.caught_share(noise, config.POWER_NEEDED_EDGE, n, n / 156)


def noise_exam(seeds: int = 100, versions: int = 20, noise_mean: float = 0.0, every_version: bool = True) -> dict:
    """The noise exam. Returns the "works" counts: for the chosen version of each seed, and for every version."""
    need = needed_power(noise_mean)
    chosen_works, all_works, all_n, words = 0, 0, 0, {w: 0 for w in verdict.WORDS}
    any_of_all = 0                  # seeds where at least one of the versions says "works" (if all were judged)
    pbos = []
    for seed in range(seeds):
        rng = np.random.default_rng([2026, 9, 28, seed])
        tune = {f"v{i + 1}": fake_trades(N_TRADES, noise_mean, rng, config.TUNE_YEARS) for i in range(versions)}
        pick = stats.choose_version(tune, n_tries=versions)
        pbos.append(pick["pbo"]["pbo"])
        check = {k: fake_trades(N_TRADES, noise_mean, rng, config.CHECK_YEARS) for k in tune}   # fresh years
        this_seed = 0
        for k, tr in check.items():
            if k != pick["chosen"] and not every_version:
                continue
            ans = verdict.decide(fake_evidence(tr, rng, noise_mean, need))
            if k == pick["chosen"]:
                chosen_works += ans.word == verdict.WORKS
                words[ans.word] += 1
            all_works += ans.word == verdict.WORKS
            this_seed += ans.word == verdict.WORKS
            all_n += 1
        any_of_all += this_seed > 0
    return {"seeds": seeds, "versions": versions, "noise_mean": noise_mean, "chosen_works": chosen_works,
            "chosen_rate": chosen_works / seeds, "all_works": all_works, "all_judged": all_n,
            "all_rate": all_works / all_n if all_n else None,
            "any_of_all": any_of_all if every_version else None, "words": words,
            "pbo_median": float(np.median(pbos)), "needed_edge_caught": need}


def planted_exam(edge: float, n: int = N_TRADES, sims: int = 200) -> dict:
    """An edge of `edge` rupees a trade after cost, planted into n trades; random runs and fake levels have none."""
    need = needed_power(0.0, n)
    caught = 0
    for i in range(sims):
        rng = np.random.default_rng([2026, 9, 28, 7, int(edge), i])
        tr = fake_trades(n, float(edge), rng, config.CHECK_YEARS)
        caught += verdict.decide(fake_evidence(tr, rng, 0.0, need)).word == verdict.WORKS
    return {"edge": edge, "trades": n, "sims": sims, "caught": caught, "rate": caught / sims}


def monday_exam(record: bool = False):
    """The L3 smoke rule through the referee (walk-forward inside 2016-2020; the check years are not opened)."""
    from machine.data.bars import load_bars
    from machine.referee import judge, prior
    from machine.tester.smoke import monday_signals

    start, end = str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1])
    signals = monday_signals(load_bars("5m", start, end))
    key = prior.IdeaKey.make("weekday_hour", "5m", "market:open")
    return judge.judge_walk_forward({"v1": signals}, tf="5m", key=key, record=record, control=True,
                                    note="L5 control: the Monday rule (buy every Monday 10:00, stop $5, target $5) — "
                                         "no edge; must be 'chance'")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--monday", action="store_true", help="also judge the Monday rule on real gold")
    ap.add_argument("--record", action="store_true", help="write the Monday judgement's trial file")
    args = ap.parse_args()
    t0 = time.time()
    for mean, name in ((-float(config.COST_INR_PER_POSITION), "no edge (loses the ₹40 cost)"),
                       (0.0, "the hardest noise (exactly ₹0 a trade after cost)")):
        r = noise_exam(noise_mean=mean)
        print(f"NOISE — {name}: {r['seeds']} ideas x {r['versions']} versions, {N_TRADES} trades each")
        print(f"  chosen version judged on fresh years: \"works\" {r['chosen_works']} of {r['seeds']} "
              f"({100 * r['chosen_rate']:.1f}%) · answers: {r['words']}")
        print(f"  every version judged on its own: \"works\" {r['all_works']} of {r['all_judged']} "
              f"({100 * r['all_rate']:.2f}%) · PBO median {r['pbo_median']:.2f} (0.5 = choosing picks luck)")
        print(f"  if all {r['versions']} versions of an idea were judged on the same years and the best kept: "
              f"\"works\" for {r['any_of_all']} of {r['seeds']} ideas — why the check years open only once")
    for edge in (50, 100):
        r = planted_exam(edge)
        print(f"PLANTED ₹{edge} a trade at {r['trades']} trades: caught {r['caught']} of {r['sims']} "
              f"({100 * r['rate']:.0f}%)")
    print(f"({time.time() - t0:.0f} s)")
    if args.monday:
        rep = monday_exam(record=args.record)
        print()
        print(rep.text())


if __name__ == "__main__":
    main()
