"""What a model learns from an idea's setups (L13.2, part A): LightGBM (a program that learns from tables) with a
PURGED WALK-FORWARD on the tune years, and SHAP (how much each thing known at a setup pushed the model's guess).

    from machine.learner import model
    r = model.analyse(feats, trades)       # feats: features.build of every 2016-2020 setup; trades: the traded ones
    print(model.text(r))

THE MODEL: its guess for a setup = the rupees it expects that trade to make after the Rs 40 cost (least-squares
regression trees, config.MODEL_PARAMS). It reads the things known at the setup's moment (machine/learner/features.py,
the L13.1 table) - a text column is a LightGBM category (its names only, never money), yes / no is 1 / 0, and a
column with one value on these setups (or none) is left out and said so.

THE WALK-FORWARD (the referee's own years, config.WALK_FORWARD_JUDGE_YEARS): the model for 2018 learns from the trades
of 2016-2017 and guesses every setup known in 2018; the model for 2019 learns from 2016-2018; the model for 2020 from
2016-2019. PURGED: a trade teaches the model of year Y only if it CLOSED at least config.MODEL_PURGE_DAYS days before
Y began - a trade still open at the new year (its money not known yet), or closing in the last days, is left out. The
check years 2021-2023 are never read: setups and trades outside the tune years are refused.

THE CHECK (does the model know anything?): in each guessed year, the trades whose setups it guessed ABOVE that year's
middle guess, against the rest - rupees a trade, all three years together. Luck: the money is shuffled among the
trades of the same year (the year of the setup) config.MODEL_SHUFFLES times, and the WHOLE walk-forward is made again
on each shuffle - so the luck line knows every choice made here. The check passes when fewer than config.P_LIMIT of
the shuffles give a gap at least this big.

SHAP: LightGBM's own TreeSHAP (predict(..., pred_contrib=True); tests/test_learner_model.py checks it against the
plain Shapley sum over every subset of things). Each trade of 2018-2020 is explained by the model of its year: how
many rupees each thing added to or took from its guess. A thing is NAMED ("the model leans on it") only when the check
passes AND its average push (rupees, either way) against its own average in the shuffles is bigger than the biggest
such ratio of ANY thing in 95 of 100 shuffles - so naming one of ~45 things by luck stays under 5 in 100.

Every number is Claude's, fixed before any idea's model was made, and measured on made-up trades first
(python -m machine.learner.selftest --model; plan/work/L132-notes.md).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config
from machine import trials as trial_files
from machine.learner import features as F

CODE_FILES = ("model.py", "features.py")

# Each feature column's short name in a report (easy English, #113; the long meaning is features.COLUMNS).
LABELS: dict[str, str] = {
    "direction": "buy or sell", "ist_hour": "the hour (India time)", "session": "the session (India time)",
    "weekday": "the day of the week", "hours_to_week_end": "hours left in the week",
    "adr": "a day's range in dollars", "volatility_share": "today's ADR against its last 12 months",
    "adr_used": "how much of a day's range today has used", "recent_range": "the last hour's range",
    "adr_zone": "the entry in today's ADR zones", "adr_share": "the entry's place in today's ADR zones",
    "bias_hour": "the hour chart", "bias_day": "the day chart", "bias_week": "the week chart",
    "dial": "the hour AND the day", "trend_days": "trend: the last 5 days", "at_ath": "the all-time high",
    "zone_at_entry": "a strong zone at the entry", "zone_at_target": "a strong zone at the target",
    "zone_flip_at_entry": "a flipped zone at the entry", "line_retest_at_entry": "a broken trendline at the entry",
    "line_in_path": "a trendline in the path", "line_against": "a line against the entry",
    "gap_past_hedge": "an unfilled gap on the hedge side", "liq_target_adr": "the next liquidity on the target side",
    "liq_hedge_adr": "the next liquidity on the hedge side",
    "equal_target_adr": "untaken equal highs / lows on the target side", "hedge_adr": "the hedge distance",
    "room_adr": "the room to the target", "reward": "the target against the hedge",
    "level_kind": "the liquidity taken", "level_chart": "the chart of the level taken",
    "level_rank": "the rank of the liquidity taken", "level_touches": "touches of the level taken",
    "level_day": "when the level taken was formed", "level_age_days": "the age of the level taken (days)",
    "grab_depth_adr": "how far the grab went past the level", "break_beyond_adr": "how far the break went",
    "gap_size_adr": "the size of the gap", "break_to_gap": "when the gap came, against the break",
    "leg_position": "the entry inside the move", "target_kind": "the kind of level the target is",
    "news_next_min": "minutes to the next news", "news_next_kind": "the kind of the next news",
    "news_next_colour": "the colour of the next news", "news_last_min": "minutes since the last news",
    "news_last_kind": "the kind of the last news", "news_last_colour": "the colour of the last news",
}


def _lgb():
    import lightgbm                     # only the model needs it (requirements.txt)
    return lightgbm


def _inr(x) -> str:
    """Whole rupees with a sign (-₹1,234 · ₹790); a value that rounds to 0 is ₹0."""
    if x is None:
        return "no trades"
    r = round(x)
    return f"-₹{-r:,}" if r < 0 else f"₹{abs(r):,}"


def _missing(v) -> bool:
    return v is None or (not isinstance(v, str) and pd.isna(v))


def _is_number(v) -> bool:
    return isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating))


class ModelError(ValueError):
    """The model refused: years outside the tune years, trades without a setup."""


# ================================================================================================================
# the table the model reads
# ================================================================================================================

@dataclass
class Encoder:
    """Which feature columns the model reads and how. Fitted on the setups' table (names and values only, no money)."""
    columns: list[str]
    kinds: dict[str, str]                     # column -> "text" / "number"
    categories: dict[str, list[str]]          # a text column's names (sorted) - the same in every model
    left_out: dict[str, str]                  # column -> why the model does not read it

    @classmethod
    def fit(cls, feats: pd.DataFrame) -> "Encoder":
        cols, kinds, cats, out = [], {}, {}, {}
        for c in F.COLUMNS:
            if c not in feats.columns:
                out[c] = "not in the table"
                continue
            vals = [v for v in feats[c].tolist() if not _missing(v)]
            if not vals:
                out[c] = "never known on these setups"
                continue
            if all(_is_number(v) for v in vals):
                x = {float(v) for v in vals}
                if len(x) < 2:
                    out[c] = f"always {next(iter(x)):g} on these setups"
                    continue
                kinds[c] = "number"
            else:
                names = sorted({str(v) for v in vals})
                if len(names) < 2:
                    out[c] = f"always '{names[0]}' on these setups"
                    continue
                kinds[c] = "text"
                cats[c] = names
            cols.append(c)
        return cls(cols, kinds, cats, out)

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.columns:
            s = feats[c].tolist()
            if self.kinds[c] == "text":
                out[c] = pd.Categorical([None if _missing(v) else str(v) for v in s], categories=self.categories[c])
            else:
                out[c] = np.array([np.nan if _missing(v) else float(v) for v in s], dtype=float)
        return pd.DataFrame(out, index=feats.index)

    @property
    def text_columns(self) -> list[str]:
        return [c for c in self.columns if self.kinds[c] == "text"]

    def as_dict(self) -> dict:
        return {"columns": self.columns, "kinds": self.kinds, "categories": self.categories, "left_out": self.left_out}


# ================================================================================================================
# the walk-forward
# ================================================================================================================

@dataclass
class WalkForward:
    guesses: pd.Series            # every setup of the guessed years: the model of its year's guess (NaN elsewhere)
    contrib: pd.DataFrame         # the traded setups of the guessed years: SHAP per column + "base" (the start value)
    folds: list[dict]


def fit(X: pd.DataFrame, y, *, seed: int, text_columns: list[str], params: dict | None = None,
        rounds: int | None = None):
    """One LightGBM model (config.MODEL_PARAMS, config.MODEL_ROUNDS trees)."""
    lgb = _lgb()
    p = {**(config.MODEL_PARAMS if params is None else params), "seed": int(seed) % (2 ** 31)}
    data = lgb.Dataset(X, label=np.asarray(y, dtype=float), categorical_feature=text_columns or "auto",
                       free_raw_data=False)
    return lgb.train(p, data, num_boost_round=config.MODEL_ROUNDS if rounds is None else rounds)


def years_of(known_at) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(known_at)).year.to_numpy()


def walk_forward(X: pd.DataFrame, known_at: pd.Series, trades: pd.DataFrame, *, seed: int,
                 text_columns: list[str], y: pd.Series | None = None, shap: bool = True,
                 params: dict | None = None, rounds: int | None = None) -> WalkForward:
    """The models of the guessed years (see the module text). X: the encoded table of EVERY setup (index = signal id);
    known_at: each setup's moment; trades: net_inr + exit_time of the traded setups; y: the money the models learn
    (default the trades' own - the shuffles pass theirs)."""
    y = trades["net_inr"].astype(float) if y is None else y.loc[trades.index].astype(float)
    known = pd.Series(pd.to_datetime(known_at).to_numpy(), index=X.index)
    setup_year = years_of(known)
    exits = pd.DatetimeIndex(pd.to_datetime(trades["exit_time"]))
    t_known = pd.DatetimeIndex(known.loc[trades.index])
    guesses = pd.Series(np.nan, index=X.index, dtype=float)
    cols = [*X.columns, "base"]
    parts, folds = [], []
    for year in config.WALK_FORWARD_JUDGE_YEARS:
        start = pd.Timestamp(f"{year}-01-01")
        cut = start - pd.Timedelta(days=config.MODEL_PURGE_DAYS)
        learn = trades.index[exits < cut]
        purged = trades.index[(t_known < start) & (exits >= cut)]
        guess = X.index[setup_year == year]
        traded = [i for i in guess if i in trades.index]
        info = {"year": int(year), "learn_before": str(cut), "learned_from": int(len(learn)),
                "purged": int(len(purged)), "guessed": int(len(guess)), "traded": int(len(traded))}
        folds.append(info)
        if len(learn) < 2 or not len(guess):
            continue
        booster = fit(X.loc[learn], y.loc[learn], seed=seed, text_columns=text_columns, params=params, rounds=rounds)
        guesses.loc[guess] = booster.predict(X.loc[guess])
        if shap and traded:
            c = booster.predict(X.loc[traded], pred_contrib=True)
            parts.append(pd.DataFrame(np.asarray(c, dtype=float), index=pd.Index(traded, name="signal"), columns=cols))
    contrib = pd.concat(parts) if parts else pd.DataFrame(columns=cols, dtype=float)
    return WalkForward(guesses, contrib, folds)


# ================================================================================================================
# the check and the shuffles
# ================================================================================================================

def check_gap(guess, value, year) -> dict:
    """The trades whose setups the model guessed above their year's middle guess, against the rest (rupees a trade,
    every guessed year together). A year where every guess is the same says nothing and is left out."""
    g, v, yr = (np.asarray(x) for x in (guess, value, year))
    g, v = g.astype(float), v.astype(float)
    top = np.zeros(len(g), bool)
    used = np.zeros(len(g), bool)
    by_year = []
    for y in np.unique(yr):
        m = yr == y
        ok = m & np.isfinite(g)
        if not ok.any():
            continue
        med = float(np.median(g[ok]))
        t = ok & (g > med + 1e-9)
        if t.any() and (ok & ~t).any():
            top |= t
            used |= ok
            a, b = v[t], v[ok & ~t]
            by_year.append({"year": int(y), "top_trades": int(a.size), "top_per_trade": round(float(a.mean()), 2),
                            "rest_trades": int(b.size), "rest_per_trade": round(float(b.mean()), 2),
                            "gap": round(float(a.mean() - b.mean()), 2)})
        else:
            by_year.append({"year": int(y), "top_trades": 0, "top_per_trade": None, "rest_trades": int(ok.sum()),
                            "rest_per_trade": round(float(v[ok].mean()), 2), "gap": None})
    a, b = v[top], v[used & ~top]
    gap = float(a.mean() - b.mean()) if a.size and b.size else None
    return {"gap": None if gap is None else round(gap, 2), "top_trades": int(a.size),
            "top_per_trade": round(float(a.mean()), 2) if a.size else None, "rest_trades": int(b.size),
            "rest_per_trade": round(float(b.mean()), 2) if b.size else None, "by_year": by_year}


def shuffled(values: pd.Series, groups, rng) -> pd.Series:
    """The money moved among the trades of the same group (the setup's year): each year keeps its own money."""
    v = values.to_numpy(dtype=float).copy()
    g = np.asarray(groups)
    for key in np.unique(g):
        idx = np.flatnonzero(g == key)
        v[idx] = v[rng.permutation(idx)]
    return pd.Series(v, index=values.index)


def seed_of(ids, name: str = "") -> int:
    text = "|".join(sorted(str(i) for i in ids)) + "#model#" + name
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _importance(contrib: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not len(contrib):
        return np.zeros(len(columns))
    return contrib[columns].abs().mean(axis=0).to_numpy(dtype=float)


def analyse(feats: pd.DataFrame, trades: pd.DataFrame, *, shuffles: int | None = None, seed: int | None = None,
            params: dict | None = None, rounds: int | None = None) -> dict:
    """The walk-forward, the check, SHAP and the named things for one version's setups (see the module text).

    feats: features.build's table of EVERY setup of the version on the tune years (index = signal id, with known_at).
    trades: net_inr + exit_time of its closed trades (index = signal id), the referee's."""
    shuffles = config.MODEL_SHUFFLES if shuffles is None else int(shuffles)
    missing = [s for s in trades.index if s not in feats.index]
    if missing:
        raise ModelError(f"{len(missing)} trades have no setup in the feature table (the first: {missing[0]})")
    tune = set(config.TUNE_YEARS)
    bad = sorted(set(pd.DatetimeIndex(pd.to_datetime(trades["exit_time"])).year) - tune)
    bad += sorted(set(years_of(feats["known_at"])) - tune - set(bad))
    if bad:
        raise ModelError(f"the model reads the tune years {config.TUNE_YEARS[0]}-{config.TUNE_YEARS[-1]} only "
                         f"(likhith.md L13.2); these setups or trades are in {bad}")
    enc = Encoder.fit(feats)
    X = enc.transform(feats)
    seed = seed_of(trades.index) if seed is None else int(seed)
    known = feats["known_at"]
    wf = walk_forward(X, known, trades, seed=seed, text_columns=enc.text_columns, params=params, rounds=rounds)
    guessed = [i for i in trades.index if np.isfinite(wf.guesses.get(i, np.nan))]
    value = trades.loc[guessed, "net_inr"].astype(float)
    t_year = years_of(known.loc[guessed])
    real = check_gap(wf.guesses.loc[guessed], value, t_year)

    # ---- the shuffles: the whole walk-forward again on money shuffled inside each year
    rng = np.random.default_rng(seed)
    group = years_of(known.loc[trades.index])
    null_gap = np.full(shuffles, np.nan)
    null_imp = np.zeros((shuffles, len(enc.columns)))
    for r in range(shuffles):
        y_r = shuffled(trades["net_inr"].astype(float), group, rng)
        w = walk_forward(X, known, trades, seed=seed, text_columns=enc.text_columns, y=y_r, params=params,
                         rounds=rounds)
        g = check_gap(w.guesses.loc[guessed], y_r.loc[guessed], t_year)["gap"]
        null_gap[r] = np.nan if g is None else g
        null_imp[r] = _importance(w.contrib, enc.columns)
    ok = np.isfinite(null_gap)
    if real["gap"] is None:
        p = 1.0
    else:
        p = float((1 + np.count_nonzero(null_gap[ok] >= real["gap"] - 1e-9)) / (1 + shuffles))
    passed = bool(real["gap"] is not None and p < config.P_LIMIT)

    # ---- SHAP: what it leans on
    imp = _importance(wf.contrib, enc.columns)
    scale = null_imp.mean(axis=0) if shuffles else np.ones(len(enc.columns))
    floor = 0.01 * float(scale.mean()) if scale.size and scale.mean() > 0 else 1e-9
    scale = np.maximum(scale, max(floor, 1e-9))
    ratio = imp / scale
    top_null = (null_imp / scale).max(axis=1) if shuffles else np.zeros(0)
    rows = []
    for k, c in enumerate(enc.columns):
        pj = float((1 + np.count_nonzero(top_null >= ratio[k] - 1e-12)) / (1 + shuffles))
        rows.append({"feature": c, "label": LABELS.get(c, c), "what": F.COLUMNS[c], "kind": enc.kinds[c],
                     "mean_push": round(float(imp[k]), 2), "shuffled_push": round(float(scale[k]), 2),
                     "ratio": round(float(ratio[k]), 3), "p": round(pj, 5),
                     "named": bool(passed and pj < config.P_LIMIT)})
    rows.sort(key=lambda d: -d["mean_push"])
    named = [d["feature"] for d in rows if d["named"]]
    for d in rows[:12]:
        d["parts"] = parts_of(feats.loc[wf.contrib.index, d["feature"]], wf.contrib[d["feature"]],
                              trades.loc[wf.contrib.index, "net_inr"], enc.kinds[d["feature"]])
    base = float(wf.contrib["base"].mean()) if len(wf.contrib) else None
    return {
        "years": list(config.TUNE_YEARS), "judge_years": list(config.WALK_FORWARD_JUDGE_YEARS),
        "setups": int(len(feats)), "trades": int(len(trades)), "guessed_trades": int(len(guessed)),
        "encoder": enc.as_dict(), "folds": wf.folds,
        "check": {**real, "p": round(p, 5), "passed": passed, "shuffles": shuffles,
                  "shuffled_gap_middle": None if not ok.any() else round(float(np.median(null_gap[ok])), 2),
                  "shuffled_gap_95": None if not ok.any() else round(float(np.quantile(null_gap[ok], 0.95)), 2)},
        "shap": rows, "named": named, "base_guess": None if base is None else round(base, 2),
        "guesses": {str(k): round(float(v), 2) for k, v in wf.guesses.dropna().items()},
        "seed": seed,
    }


def parts_of(values: pd.Series, push: pd.Series, money: pd.Series, kind: str) -> list[dict]:
    """How one thing pushed the guesses, part by part: a text column's names; a number column's thirds of these
    trades (a description, not a cut point that decides anything); "not known" apart."""
    vals = values.tolist()
    push = push.to_numpy(dtype=float)
    money = money.to_numpy(dtype=float)
    known = np.array([not _missing(v) for v in vals])
    out = []
    if kind == "text":
        names = pd.Series([str(v) if k else None for v, k in zip(vals, known)])
        for name, n in names[known].value_counts().items():
            m = (names == name).to_numpy()
            out.append({"part": str(name), "trades": int(n), "mean_push": round(float(push[m].mean()), 2),
                        "per_trade": round(float(money[m].mean()), 2)})
    else:
        x = np.array([float(v) if k else np.nan for v, k in zip(vals, known)])
        xs = x[known]
        flags = all(isinstance(v, (bool, np.bool_)) for v, k in zip(vals, known) if k)
        if xs.size and (flags or len(np.unique(xs)) <= 5):          # yes / no, or a few values: each one its own part
            for u in np.unique(xs):
                m = known & (x == u)
                name = ("yes" if u else "no") if flags else f"{u:g}"
                out.append({"part": name, "trades": int(m.sum()), "mean_push": round(float(push[m].mean()), 2),
                            "per_trade": round(float(money[m].mean()), 2)})
        elif xs.size:
            q1, q2 = np.quantile(xs, [1 / 3, 2 / 3])
            for name, m in ((f"lowest third (up to {q1:.3g})", known & (x <= q1)),
                            (f"middle third ({q1:.3g} to {q2:.3g})", known & (x > q1) & (x <= q2)),
                            (f"highest third (above {q2:.3g})", known & (x > q2))):
                if m.any():
                    out.append({"part": name, "trades": int(m.sum()), "mean_push": round(float(push[m].mean()), 2),
                                "per_trade": round(float(money[m].mean()), 2)})
    if (~known).any():
        m = ~known
        out.append({"part": "not known", "trades": int(m.sum()), "mean_push": round(float(push[m].mean()), 2),
                    "per_trade": round(float(money[m].mean()), 2)})
    return out


# ================================================================================================================
# the words
# ================================================================================================================

def years_text(years) -> str:
    ys = sorted(int(y) for y in years)
    return str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"


def check_line(r: dict) -> str:
    """The model check in one line (easy English, #113)."""
    c = r["check"]
    judged = years_text(r["judge_years"])
    if c["gap"] is None:
        return (f"The model's check ({judged}, each year guessed by a model made only from the years before): it gave "
                "every setup of a year the same guess - it found nothing it could use.")
    verdict = "it found something" if c["passed"] else "it found nothing it could use"
    return (f"The model's check ({judged}, each year guessed by a model made only from the years before): the "
            f"{c['top_trades']} trades it liked best made {_inr(c['top_per_trade'])} a trade, the other "
            f"{c['rest_trades']} {_inr(c['rest_per_trade'])} (a gap of {_inr(c['gap'])}). The money shuffled inside "
            f"each year gives a gap this big {100 * c['p']:.1f} times in 100 ({c['shuffles']} shuffles; it needs "
            f"under {round(100 * config.P_LIMIT)}): {verdict}.")


def named_line(r: dict) -> str:
    """What the model leans on (only the named things), with the part that pushed most."""
    if not r["named"]:
        if r["check"]["passed"]:
            return "It leans on nothing more than luck would: no one thing stands out."
        return "Nothing is named: the check did not pass, so what it leaned on is shown on the page only."
    out = []
    for d in r["shap"]:
        if not d["named"]:
            continue
        parts = sorted(d.get("parts") or [], key=lambda p: -abs(p["mean_push"]))
        best = parts[0] if parts else None
        out.append(d.get("label", d["feature"]) + (f" ('{best['part']}' pushed the guess {_inr(best['mean_push'])} a "
                                                   f"trade, on {best['trades']} trades)" if best else ""))
    return ("It leans most on (SHAP, how much each thing pushed its guess): " + "; ".join(out))


def text(r: dict, idea: str = "", version: str = "") -> str:
    enc = r["encoder"]
    lines = [f"THE MODEL (LightGBM, a program that learns from tables) · {idea} version {version} · "
             f"{r['trades']} trades of {r['setups']} setups on {years_text(r['years'])} · ₹ after the ₹40 cost, "
             "0.01 lot", check_line(r), named_line(r), "",
             "How each year's model was made (a trade teaches it only if it closed at least "
             f"{config.MODEL_PURGE_DAYS} days before the year began):"]
    for f in r["folds"]:
        lines.append(f"  {f['year']}: learned from {f['learned_from']} trades, {f['purged']} left out (not closed by "
                     f"then) · guessed {f['guessed']} setups, {f['traded']} of them traded")
    lines += ["", "By year (the half it liked best against the rest):"]
    for y in r["check"]["by_year"]:
        lines.append(f"  {y['year']}: {y['top_trades']} trades {_inr(y['top_per_trade'])} a trade · "
                     f"{y['rest_trades']} trades {_inr(y['rest_per_trade'])} a trade")
    lines += ["", f"WHAT IT LEANED ON (SHAP, how much each thing pushed its guess; {len(enc['columns'])} things read"
                  + (f", {len(enc['left_out'])} left out" if enc["left_out"] else "") + "):"]
    for d in r["shap"][:15]:
        lines.append(f"  {d['feature']}: {_inr(d['mean_push'])} a trade either way (shuffled money: "
                     f"{_inr(d['shuffled_push'])}) · luck {100 * d['p']:.1f} in 100"
                     + (" · NAMED" if d["named"] else ""))
        for p in d.get("parts") or []:
            lines.append(f"      {p['part']}: {p['trades']} trades, pushed {_inr(p['mean_push'])} · made "
                         f"{_inr(p['per_trade'])} a trade")
    if enc["left_out"]:
        lines.append("Left out: " + "; ".join(f"{k} ({v})" for k, v in enc["left_out"].items()))
    return "\n".join(lines)


# ================================================================================================================
# the settings recorded with a result
# ================================================================================================================

def code_sha256() -> str:
    here = Path(__file__).parent
    return trial_files.code_sha256([here / f for f in CODE_FILES])


def settings_used() -> dict:
    return {"params": dict(config.MODEL_PARAMS), "rounds": config.MODEL_ROUNDS,
            "purge_days": config.MODEL_PURGE_DAYS, "shuffles": config.MODEL_SHUFFLES, "p_limit": config.P_LIMIT,
            "judge_years": list(config.WALK_FORWARD_JUDGE_YEARS), "tune_years": list(config.TUNE_YEARS),
            "filter_above_inr": config.FILTER_ABOVE_INR}


def spec_sha256(extra: dict | None = None) -> str:
    body = {**settings_used(), **(extra or {})}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
