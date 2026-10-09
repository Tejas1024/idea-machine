"""One measurement (L15.2): the time split, both models, his written rule and "always no" - on the same held-out
charts. The exams run it on made-up answers; `python -m machine.eyes run` runs it ONCE on his answers.

    from machine.eyes import run
    data = run.prepare()                          # the answers, model A's numbers, model B's pictures
    r = run.measure(data, data.lab["y"].to_numpy())
    r["scores"]["model B"]["accuracy"], r["winner"]
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from machine import config
from machine.eyes import features as FE
from machine.eyes import labels as LB
from machine.eyes import measure as M
from machine.eyes import model_a, model_b, split

GUESSERS = ("always no", "his rule", "model A", "model B")
NAMES = {"always no": "always 'no'", "his rule": "his written rule (calm at most 0.74)",
         "model A": "model A - numbers (LightGBM)", "model B": "model B - pictures (ResNet18)"}


@dataclass
class Data:
    lab: pd.DataFrame          # labels.load(), oldest chart first
    X: pd.DataFrame            # model A's numbers, same row order
    imgs: np.ndarray           # model B's pictures, same row order


def prepare(lab: pd.DataFrame | None = None, m5: pd.DataFrame | None = None, imgs: bool = True) -> Data:
    lab = LB.load() if lab is None else lab
    if m5 is None:
        from machine.data.bars import load_bars
        m5 = load_bars("5m", str(min(config.TUNE_YEARS)), str(max(config.TUNE_YEARS)))
    X = FE.build(lab, m5).loc[lab["label_id"]]
    return Data(lab, X, model_b.stack(lab["image"]) if imgs else np.zeros((len(lab), 1, 1, 3), np.uint8))


def measure(data: Data, y: np.ndarray, *, models: tuple[str, ...] = ("model A", "model B"),
            seeds: tuple[int, ...] | None = None, pretrained: bool = True, say=None) -> dict:
    """Every guesser on the held-out charts of config.EYES_TEST_YEARS. y: the answers (his, or made-up ones)."""
    t0 = time.time()
    lab, y = data.lab, np.asarray(y, int)
    fs = split.folds(lab)
    rows = split.tested(fs)
    years = lab["year"].to_numpy()[rows]
    yt = y[rows]
    said = {"always no": np.zeros(len(rows), int), "his rule": lab["rule_yes"].to_numpy()[rows]}
    chance = {"always no": None, "his rule": -lab["calm_on_chart"].to_numpy(float)[rows]}
    extra: dict = {}
    if "model A" in models:
        pa, used = model_a.walk(data.X, y, fs)
        chance["model A"], said["model A"] = pa[rows], (pa[rows] >= config.EYES_THRESHOLD).astype(int)
        extra["model A uses"] = used
    if "model B" in models:
        pb, by_seed = model_b.walk(data.imgs, y, fs, seeds, pretrained=pretrained, say=say)
        chance["model B"], said["model B"] = pb[rows], (pb[rows] >= config.EYES_THRESHOLD).astype(int)
        extra["model B each start"] = {
            int(s): round(float(((p[rows] >= config.EYES_THRESHOLD).astype(int) == yt).mean()), 4)
            for s, p in by_seed.items()}
    names = [g for g in GUESSERS if g in said]
    scores = {g: M.score(g, yt, said[g], chance[g], years) for g in names}
    right = {g: said[g] == yt for g in names}
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs[f"{a} vs {b}"] = M.pair(right[a], right[b])
    return {
        "folds": [f.text() for f in fs], "n": int(len(rows)), "years": sorted(int(v) for v in set(years)),
        "yes": int(yt.sum()), "no": int(len(yt) - yt.sum()),
        "scores": {g: s.as_dict() for g, s in scores.items()}, "pairs": pairs, "winner": winner(scores, pairs),
        "guesses": pd.DataFrame({"label_id": lab["label_id"].to_numpy()[rows], "year": years, "his": yt,
                                 **{f"{g} chance": chance[g] for g in names if chance[g] is not None},
                                 **{f"{g} said": said[g] for g in names}}),
        **extra, "seconds": round(time.time() - t0, 1),
    }


def winner(scores: dict, pairs: dict) -> dict:
    """Between the two models: more right answers on the held-out charts wins; beyond luck only if the pair test says
    so. Without model B (an exam of model A alone) there is no winner to name."""
    if "model A" not in scores or "model B" not in scores:
        return {"name": None, "said": "one model only"}
    a, b = scores["model A"].right, scores["model B"].right
    p = pairs["model A vs model B"]
    if a == b:
        return {"name": None, "said": f"a tie: both right on {a} of {scores['model A'].n}", **p}
    name = "model A" if a > b else "model B"
    return {"name": name, "by": abs(a - b), "said": f"{NAMES[name]}: {max(a, b)} right against {min(a, b)}", **p}
