"""Both models taught on ALL his answers (L15.2), for later prompts - data/eyes/ (not in git: model B is ~45 MB).

    python -m machine.eyes final       -> data/eyes/model_a.txt, model_b_start0.pt ..., about.json
    from machine.eyes import final
    final.chance_a(X), final.chance_b(imgs)        # their chance of his yes on new charts

The held-out numbers in plan/work/eyes-model.md are the models' measure; these copies learned from every answer
(2016-2020), the plan's settings unchanged. about.json names the answers, settings and code they came from - a copy
whose answers or code differ from today's is refused when loaded (lesson B19).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from machine import config
from machine import trials as trial_files


def make(folder: Path | None = None, say=print) -> dict:
    from machine.eyes import labels as LB
    from machine.eyes import model_a, model_b, plan
    from machine.eyes import run as RUN
    folder = Path(folder or config.EYES_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    data = RUN.prepare()
    y = data.lab["y"].to_numpy()
    a = model_a.fit(data.X, y)
    a.save_model(str(folder / "model_a.txt"))
    say("model A taught on all answers")
    import torch
    for s in config.EYES_SEEDS:
        m = model_b.fit(data.imgs, y, s)
        torch.save(m.state_dict(), folder / f"model_b_start{s}.pt")
        say(f"model B, start {s}: taught on all answers")
    about = {"answers": LB.fingerprint(data.lab), "n": int(len(y)), "yes": int(y.sum()),
             "settings_sha256": trial_files.sha256_of(plan.settings()), "code_sha256": plan.code_sha256(),
             "columns": list(data.X.columns), "seeds": list(config.EYES_SEEDS)}
    tmp = folder / "about.json.tmp"
    tmp.write_text(json.dumps(about, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, folder / "about.json")
    return about


def _about(folder: Path) -> dict:
    from machine.eyes import plan
    about = json.loads((folder / "about.json").read_text(encoding="utf-8"))
    if about["code_sha256"] != plan.code_sha256() or about["settings_sha256"] != trial_files.sha256_of(plan.settings()):
        raise RuntimeError("data/eyes/ was made by other code or settings - run: python -m machine.eyes final")
    return about


def chance_a(X, folder: Path | None = None) -> np.ndarray:
    import lightgbm as lgb
    folder = Path(folder or config.EYES_DIR)
    about = _about(folder)
    m = lgb.Booster(model_file=str(folder / "model_a.txt"))
    return np.asarray(m.predict(X[about["columns"]].to_numpy(float)), dtype=float)


def chance_b(imgs: np.ndarray, folder: Path | None = None) -> np.ndarray:
    import torch
    from machine.eyes import model_b
    folder = Path(folder or config.EYES_DIR)
    about = _about(folder)
    out = []
    for s in about["seeds"]:
        m, _ = model_b.net(pretrained=False)
        m.load_state_dict(torch.load(folder / f"model_b_start{s}.pt", map_location="cpu"))
        out.append(model_b.predict(m, imgs))
    return np.mean(out, axis=0)
