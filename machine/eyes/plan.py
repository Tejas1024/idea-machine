"""The plan, fixed BEFORE his answers are scored (L15.2) - plan/work/eyes/plan.json (+ plan.md in easy English).

    python -m machine.eyes plan       writes it (after the exams); then COMMIT AND PUSH it
    python -m machine.eyes run        check() first: refuses unless the plan is committed and pushed, and the
                                      settings, the measuring code and his answers are exactly the plan's

So the order is in git for anyone to see: the exams, then the plan, then ONE score of his held-out answers. A second
score needs `--again "<why>"` and is written next to the first (runs.jsonl) - never instead of it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from machine import config
from machine import trials as trial_files

# The files that decide the numbers (the report's wording and pictures are not among them).
CODE = ("labels.py", "features.py", "split.py", "model_a.py", "model_b.py", "measure.py", "run.py")


class PlanError(RuntimeError):
    pass


def settings() -> dict:
    return {
        "word": config.EYES_WORD, "test_years": list(config.EYES_TEST_YEARS), "purge_days": config.EYES_PURGE_DAYS,
        "level": config.EYES_LEVEL, "threshold": config.EYES_THRESHOLD, "lgbm_params": config.EYES_LGBM_PARAMS,
        "lgbm_rounds": config.EYES_LGBM_ROUNDS, "crop": list(config.EYES_CROP), "image_hw": list(config.EYES_IMAGE_HW),
        "net": config.EYES_NET, "seeds": list(config.EYES_SEEDS), "boot": config.EYES_BOOT,
    }


def code_sha256() -> str:
    here = Path(__file__).resolve().parent
    return trial_files.code_sha256([here / f for f in CODE])


def path(folder: Path | None = None) -> Path:
    return Path(folder or config.EYES_WORK_DIR) / "plan.json"


def make(lab, exams: dict | None, *, versions: dict | None = None) -> dict:
    from machine.eyes import labels as LB
    from machine.eyes import split
    fs = split.folds(lab)
    n = int(sum(len(f.test) for f in fs))
    return {
        "prompt": "L15.2", "made_by": "python -m machine.eyes plan",
        "labels": {"fingerprint": LB.fingerprint(lab), "n": int(len(lab)), "yes": int(lab["y"].sum()),
                   "per_year": {int(k): v for k, v in LB.counts(lab).to_dict(orient="index").items()}},
        "split": [f.text() for f in fs], "held_out": n,
        "settings": settings(), "settings_sha256": trial_files.sha256_of(settings()),
        "code_sha256": code_sha256(), "code_files": [f"machine/eyes/{f}" for f in CODE],
        "versions": versions or {},
        "reported": ["every guesser's accuracy on the held-out charts with its 90% interval (Wilson)",
                     "his yes found / his no found, the table of his answer against its answer",
                     "year by year, the worst year", "how often a yes chart gets a higher chance than a no chart",
                     "the pair test between the two models, and each model against his rule and 'always no'"],
        "winner_rule": ("the model with more right answers on the held-out charts; 'beyond luck' only when McNemar's "
                        "exact two-sided chance is below 10 in 100; equal counts = a tie. Not beating model A, his "
                        "rule or 'always no' is an honest result."),
        "exams": (exams or {}).get("summary"),
    }


def write(plan: dict, folder: Path | None = None) -> tuple[Path, Path]:
    p = path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    _put(p, json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    md = p.with_name("plan.md")
    _put(md, page(plan))
    return p, md


def page(plan: dict) -> str:
    s = plan["settings"]
    L = ["# L15.2 - the plan, fixed before his answers were scored", "",
         f"**Answers:** {plan['labels']['n']} with a picture ({plan['labels']['yes']} yes). "
         f"**Held out:** {plan['held_out']} - each year guessed by models that learned only from the charts before it:",
         *[f"- {x}" for x in plan["split"]], "",
         "**Model A** - LightGBM on 14 numbers that describe the chart (machine/eyes/features.py), "
         f"{s['lgbm_rounds']} small trees (at most {s['lgbm_params']['num_leaves']} leaves each).",
         f"**Model B** - ResNet18 with its ImageNet weights; the candle area of the picture at "
         f"{s['image_hw'][0]} x {s['image_hw'][1]}; layers from {s['net']['train_from']} on learn for "
         f"{s['net']['epochs']} rounds; taught {len(s['seeds'])} times, its answer = the average.",
         f"**A model says yes** when its chance of his yes is {s['threshold']} or more.",
         f"**Next to them:** 'always no' and his written rule on the same chart (calm at most 0.74).", "",
         "**Reported:** " + "; ".join(plan["reported"]) + ".", "",
         f"**The winner:** {plan['winner_rule']}", "",
         f"Settings fingerprint `{plan['settings_sha256'][:16]}`, measuring code `{plan['code_sha256'][:16]}`, "
         f"answers `{plan['labels']['fingerprint'][:16]}` - `python -m machine.eyes run` refuses if any differs."]
    return "\n".join(L) + "\n"


def check(lab, folder: Path | None = None, *, git=None) -> dict:
    """Refuse the real run unless the plan is committed AND pushed, and everything it fixed is still the same."""
    from machine.eyes import labels as LB
    git = git or _git
    p = path(folder)
    if not p.is_file():
        raise PlanError("no plan: run `python -m machine.eyes exams`, then `python -m machine.eyes plan`, commit and "
                        "push it - his answers are scored only after that")
    plan = json.loads(p.read_text(encoding="utf-8"))
    try:
        rel = p.resolve().relative_to(Path(config.ROOT).resolve()).as_posix()
    except ValueError:                                  # a plan outside the repo (the tests' own folder)
        rel = p.resolve().as_posix()
    if git("ls-files", "--error-unmatch", rel) is None:
        raise PlanError(f"{rel} is not committed - commit and push it first")
    if git("diff", "--quiet", "HEAD", "--", rel, ok_codes=(0,)) is None:
        raise PlanError(f"{rel} differs from the commit - commit and push it first")
    changed = git("status", "--porcelain", "--", "machine/eyes", "machine/config.py") or ""
    if changed.strip():
        raise PlanError("the measuring code differs from the commit:\n" + changed)
    commit = git("log", "-1", "--format=%H", "--", rel)
    if not commit:
        raise PlanError(f"{rel} has no commit")
    git("fetch", "-q", "origin", "main")
    if git("merge-base", "--is-ancestor", commit.strip(), "origin/main", ok_codes=(0,)) is None:
        raise PlanError(f"the plan's commit {commit.strip()[:10]} is not on GitHub's main yet - push it first")
    if plan["settings_sha256"] != trial_files.sha256_of(settings()):
        raise PlanError("the settings differ from the plan's")
    if plan["code_sha256"] != code_sha256():
        raise PlanError("the measuring code differs from the plan's")
    if plan["labels"]["fingerprint"] != LB.fingerprint(lab):
        raise PlanError("his answers (or their pictures) differ from the plan's")
    return {**plan, "plan_commit": commit.strip()}


def _git(*args: str, ok_codes: tuple[int, ...] = (0,)) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode in ok_codes else None


def _put(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
