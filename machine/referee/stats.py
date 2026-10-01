"""The numbers the referee judges with (L5, step 6).

Trades come in as a table with at least `net_inr` (rupees after the Rs 40 cost) and `exit_time` (server time). A trade
counts in the year and week it CLOSED, like the tester (L3).

- per_year / worst_year           rupees, trades and rupees a trade for every judged year (a year with no trade is shown)
- without_best                    the same without the best 3 trades
- bootstrap_p                     the luck test: weeks drawn again with replacement (a "weekly block bootstrap": trades
                                  of one week stay together, because they share that week's market); one-sided p for
                                  "rupees a trade > 0"
- deflated_sharpe, pbo_cscv       used ONLY at TUNE time to choose between versions (choose_version refuses any other
                                  year): Bailey & Lopez de Prado's Deflated Sharpe Ratio (2014) and the Probability of
                                  Backtest Overfitting by CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2015), written here
                                  (no pypbo).
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from machine import config, costs

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


# ---- the plain numbers -------------------------------------------------------------------------------------------

def closed(table: pd.DataFrame) -> pd.DataFrame:
    """The closed trades of a tester table (open-at-end trades have no money — L3), as net_inr + exit_time."""
    t = table[table["status"] == "closed"] if "status" in table.columns else table
    return pd.DataFrame({"net_inr": t["net_inr"].astype(float).to_numpy(),
                         "exit_time": pd.to_datetime(t["exit_time"]).to_numpy()})


def per_year(trades: pd.DataFrame, years) -> dict[int, dict]:
    out = {int(y): {"trades": 0, "net_inr": 0.0, "per_trade": None} for y in sorted(years)}
    if len(trades):
        by = trades.groupby(pd.DatetimeIndex(trades["exit_time"]).year)["net_inr"]
        for y, s in by:
            if int(y) not in out:
                raise ValueError(f"a trade closed in {y}, outside the judged years {sorted(out)}")
            out[int(y)] = {"trades": int(s.size), "net_inr": round(float(s.sum()), 2),
                           "per_trade": round(float(s.mean()), 2)}
    return out


def worst_year(years: dict[int, dict]) -> dict:
    y = min(years, key=lambda k: years[k]["net_inr"])
    return {"year": y, **years[y]}


def summary(trades: pd.DataFrame) -> dict:
    n = len(trades)
    total = float(trades["net_inr"].sum()) if n else 0.0
    return {"trades": n, "net_inr": round(total, 2), "per_trade": round(total / n, 2) if n else None,
            "won": int((trades["net_inr"] > 0).sum()) if n else 0}


def a_year(summ: dict, years) -> dict:
    """Rupees a year: the judged years' total after cost divided by how many years were judged (a year with no trade
    counts as a year)."""
    n = len(tuple(years))
    return {"years": n, "net_inr": round(summ["net_inr"] / n, 2) if n else None}


def extra_lines(reps, years) -> dict | None:
    """The two labelled extra lines (L2) for every closed trade of these replays: the broker's swap and the real
    spread INSTEAD of the Rs 40. Totals, and a year (divided like a_year). None when no trade closed."""
    legs = [p for r in reps for t in r.trades if t.status == "closed" for p in t.positions()]
    if not legs:
        return None
    x = costs.extra_lines(legs)
    n = len(tuple(years))
    return {"with_swap_inr": round(x.with_swap_inr, 2), "with_swap_a_year": round(x.with_swap_inr / n, 2),
            "nights_held": x.nights_held, "nights_charged": x.nights_charged,
            "with_spread_inr": None if x.with_spread_inr is None else round(x.with_spread_inr, 2),
            "with_spread_a_year": None if x.with_spread_inr is None else round(x.with_spread_inr / n, 2),
            "swap_reading": x.swap_reading}


def without_best(trades: pd.DataFrame, k: int = config.WITHOUT_BEST) -> dict:
    """Rupees a trade after taking out the k best trades (None when k or fewer trades)."""
    x = np.sort(trades["net_inr"].to_numpy(dtype=float))
    rest = x[:-k] if k else x
    best = x[-k:] if k else x[:0]
    return {"left_out": k, "trades": int(rest.size), "net_inr": round(float(rest.sum()), 2),
            "per_trade": round(float(rest.mean()), 2) if rest.size else None,
            "best_inr": [round(float(v), 2) for v in best[::-1]]}


# ---- the luck test -----------------------------------------------------------------------------------------------

def week_ids(times) -> np.ndarray:
    """The Monday (as a day number) of each time's week: trades of one week share one block."""
    t = pd.DatetimeIndex(pd.to_datetime(times))
    monday = t.normalize() - pd.to_timedelta(t.dayofweek, unit="D")
    return (monday.as_unit("s").asi8 // 86_400).astype(np.int64)


def bootstrap_p(values, weeks, n_boot: int = config.BOOTSTRAP_RESAMPLES, seed=0) -> float:
    """One-sided p for "rupees a trade > 0", by a weekly block bootstrap under the null.

    The trades are first moved so their mean is exactly 0 (the world where the idea has no edge). Then the weeks that
    have trades are drawn again with replacement, as many weeks as there are, n_boot times; each draw gives a mean
    rupees a trade (its trades' total / its trade count). p = the share of draws at least as good as the real mean
    (with the usual +1 so p is never 0). Small p: luck alone rarely gives this much.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return 1.0
    w = np.asarray(weeks)
    if w.shape != v.shape:
        raise ValueError("values and weeks must have the same length")
    observed = v.mean()
    _, inv = np.unique(w, return_inverse=True)
    sums = np.bincount(inv, weights=v - observed)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sums.size, size=(n_boot, sums.size))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float((1 + np.count_nonzero(means >= observed - 1e-9)) / (n_boot + 1))


# ---- choosing between versions: TUNE years only --------------------------------------------------------------------

def sharpe(values) -> float:
    """Rupees a trade divided by their spread (per trade, not per year). 0 when it cannot be measured."""
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def deflated_sharpe(values, n_tries: int, sharpe_variance: float) -> float:
    """The Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the probability that this version's true Sharpe is
    above the best one pure luck would give among n_tries versions, allowing for skew, fat tails and the trade count.

    sharpe_variance: the variance of the Sharpe ratios of all the versions tried (0 with one version: then the bar is
    Sharpe 0, the plain "Probabilistic Sharpe Ratio").
    """
    x = np.asarray(values, dtype=float)
    t = x.size
    if t < 3:
        return 0.0
    sr = sharpe(x)
    if n_tries > 1 and sharpe_variance > 0:
        sr0 = math.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_tries)
                                            + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_tries * math.e)))
    else:
        sr0 = 0.0
    sd = x.std(ddof=0)
    skew = float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    kurt = float(((x - x.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(max(denom, 1e-12))
    return float(_N.cdf(z))


def pbo_cscv(matrix, blocks: int = config.CSCV_BLOCKS) -> dict:
    """The Probability of Backtest Overfitting by CSCV (combinatorially symmetric cross-validation).

    matrix: rows = periods in time order (weeks), columns = versions (rupees per week; 0 in a week with no trade).
    The rows are cut into `blocks` equal parts. For every way of taking half the parts as the "choosing" half: the
    version with the best Sharpe there is found, and its rank among all versions on the OTHER half. PBO = the share
    of ways where it lands in the bottom half. About 0.5 = choosing on the past picks luck; near 0 = the best version
    stays good.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("PBO needs a periods x versions table with at least 2 versions")
    if blocks % 2 or blocks < 2 or m.shape[0] < blocks * 2:
        raise ValueError(f"PBO needs an even number of blocks and at least 2 periods per block ({m.shape[0]} periods)")
    parts = np.array_split(np.arange(m.shape[0]), blocks)
    s1 = np.array([m[p].sum(axis=0) for p in parts])              # blocks x versions
    s2 = np.array([(m[p] ** 2).sum(axis=0) for p in parts])
    cnt = np.array([len(p) for p in parts], dtype=float)
    combos = np.array(list(combinations(range(blocks), blocks // 2)))
    pick = np.zeros((len(combos), blocks))
    pick[np.arange(len(combos))[:, None], combos] = 1.0

    def sharpe_of(sel: np.ndarray) -> np.ndarray:
        n = (sel @ cnt)[:, None]
        mean = (sel @ s1) / n
        var = ((sel @ s2) - n * mean ** 2) / np.maximum(n - 1, 1)
        sd = np.sqrt(np.maximum(var, 0))
        return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 1e-12)

    is_sr, oos_sr = sharpe_of(pick), sharpe_of(1.0 - pick)
    best = is_sr.argmax(axis=1)
    rows = np.arange(len(combos))
    rank = (oos_sr < oos_sr[rows, best][:, None]).sum(axis=1) + 1          # 1 = worst on the other half
    omega = rank / (m.shape[1] + 1)
    logits = np.log(omega / (1 - omega))
    return {"pbo": float(np.mean(logits <= 0)), "ways": int(len(combos)), "blocks": blocks,
            "versions": int(m.shape[1]), "median_logit": float(np.median(logits))}


def weekly_matrix(version_trades: dict[str, pd.DataFrame], years) -> pd.DataFrame:
    """Rupees per week for each version over every week of the given years (0 in a week with no trade)."""
    start, end = pd.Timestamp(f"{min(years)}-01-01"), pd.Timestamp(f"{max(years)}-12-31")
    all_weeks = np.unique(week_ids(pd.date_range(start, end, freq="D")))
    out = pd.DataFrame(0.0, index=all_weeks, columns=list(version_trades))
    for name, tr in version_trades.items():
        if len(tr):
            s = pd.Series(tr["net_inr"].to_numpy(dtype=float)).groupby(week_ids(tr["exit_time"])).sum()
            out.loc[s.index, name] = s.to_numpy()
    return out


def choose_version(version_trades: dict[str, pd.DataFrame], n_tries: int | None = None, years=None) -> dict:
    """Choose between versions on the TUNE years (2016-2020) only: the highest Deflated Sharpe wins; PBO is reported.

    n_tries: every version the family has tried (from the trial files), when more than the ones given here.
    Refuses any trade that closed outside the tune years — these two numbers never see the check years (L5).
    """
    years = tuple(years or config.TUNE_YEARS)
    if not set(years) <= set(config.TUNE_YEARS):
        raise ValueError(f"Deflated Sharpe and PBO choose between versions on the tune years {config.TUNE_YEARS} "
                         f"only (L5), not {years}")
    if not version_trades:
        raise ValueError("no versions to choose from")
    for name, tr in version_trades.items():
        bad = sorted(set(pd.DatetimeIndex(tr["exit_time"]).year) - set(years))
        if bad:
            raise ValueError(f"version {name} has trades in {bad}: Deflated Sharpe and PBO are used ONLY on the tune "
                             f"years {years[0]}-{years[-1]} to choose between versions (L5)")
    n = max(n_tries or 0, len(version_trades))
    srs = {k: sharpe(v["net_inr"]) for k, v in version_trades.items()}
    var = float(np.var(list(srs.values()), ddof=1)) if len(srs) > 1 else 0.0
    rows = {k: {"trades": len(v), "per_trade": summary(v)["per_trade"], "sharpe": round(srs[k], 4),
                "deflated_sharpe": round(deflated_sharpe(v["net_inr"], n, var), 4)}
            for k, v in version_trades.items()}
    chosen = max(rows, key=lambda k: (rows[k]["deflated_sharpe"], rows[k]["sharpe"]))
    pbo = None
    if len(version_trades) > 1:
        wk = weekly_matrix(version_trades, years)
        blocks = config.CSCV_BLOCKS
        while blocks > 2 and len(wk) < blocks * 2:
            blocks -= 2
        pbo = pbo_cscv(wk.to_numpy(), blocks)
    return {"years": list(years), "tries": n, "versions": rows, "chosen": chosen, "pbo": pbo}
