"""MLflow logs every trial file (L16.1 step 4). The trial files stay the count.

Every JSON in plan/work/trials/ (R14: one per test run) becomes ONE MLflow run in a local store (data/mlflow/: sqlite +
the files), in the experiment "trials-<kind>" (trials-tester, trials-belief, trials-money-plan, ...):
    params    the trial's settings (nested ones as a.b.c)
    metrics   every number in its result (nested ones as a.b.c)
    tags      the trial file and its SHA-256, the trial id, kind, idea, version, parent idea, who, the git commit,
              the price data's content hash; its note is the run's description
    artifact  the trial file itself, unchanged
The referee counts FILES (machine/referee), never MLflow runs: MLflow is a window onto the files. A file already in
MLflow with the same SHA-256 is not logged again; a file whose bytes changed after it was logged (trial files are never
rewritten - R14) is logged again and named in the summary.

    python -m machine.host mlflow-sync      log what is missing now
    python -m machine.host mlflow-ui        see them in the browser (MLflow's own page, http://127.0.0.1:5000)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path

from machine import config

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # MLflow's own hint line, before any mlflow import

MAX_PARAM = 6000                      # MLflow's longest param value
MAX_KEY = 250                         # MLflow's longest key
MAX_METRICS = 1000                    # one log_batch call takes at most 1000 metrics
BAD_KEY = re.compile(r"[^A-Za-z0-9_\-. /]+")


def uri(folder: Path | None = None) -> str:
    folder = Path(folder or config.MLFLOW_DIR)
    return "sqlite:///" + (folder / "mlflow.db").as_posix()


def client(folder: Path | None = None):
    for name in ("mlflow", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    from mlflow.tracking import MlflowClient
    folder = Path(folder or config.MLFLOW_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=uri(folder))


def flat(d, prefix: str = "", depth: int = 0) -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}, three levels deep; lists stay whole (as text in a param)."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and depth < 3:
            out.update(flat(v, key + ".", depth + 1))
        else:
            out[key] = v
    return out


def _key(k: str) -> str:
    return BAD_KEY.sub("_", str(k))[:MAX_KEY] or "_"


def params_of(trial: dict) -> dict[str, str]:
    out = {}
    for k, v in flat(trial.get("settings")).items():
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
        out[_key(k)] = text[:MAX_PARAM]
    return out


def metrics_of(trial: dict) -> dict[str, float]:
    out = {}
    for k, v in flat(trial.get("result")).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if math.isfinite(float(v)):
            out[_key(k)] = float(v)
    return dict(list(out.items())[:MAX_METRICS])


def tags_of(trial: dict, rel: str, sha: str) -> dict[str, str]:
    data = trial.get("data") or {}
    data_hash = next((str(v) for k, v in data.items() if "sha256" in k or "content" in k), "")
    tags = {"trial_file": rel, "trial_sha256": sha, "trial_id": trial.get("trial_id", ""),
            "kind": trial.get("kind", ""), "idea_id": trial.get("idea_id", ""),
            "version_id": trial.get("version_id", ""), "parent_idea": trial.get("parent_idea") or "",
            "who": trial.get("who", ""), "git_commit": (trial.get("git") or {}).get("commit", ""),
            "data_sha256": data_hash, "mlflow.note.content": trial.get("note") or ""}
    return {k: str(v)[:MAX_PARAM] for k, v in tags.items()}


def _start_ms(trial: dict) -> int | None:
    try:
        return int(dt.datetime.fromisoformat(trial["made_at"]).timestamp() * 1000)
    except (KeyError, ValueError, TypeError):
        return None


def logged(c) -> dict[str, list[str]]:
    """trial file -> the SHA-256 values already logged for it."""
    out: dict[str, list[str]] = {}
    for e in c.search_experiments():
        token = None
        while True:
            page = c.search_runs([e.experiment_id], max_results=1000, page_token=token)
            for r in page:
                f = r.data.tags.get("trial_file")
                if f:
                    out.setdefault(f, []).append(r.data.tags.get("trial_sha256", ""))
            token = page.token
            if not token:
                break
    return out


def sync(trials_dir: Path | None = None, folder: Path | None = None) -> dict:
    """Log every trial file not in MLflow yet. Returns the counts (the files are the count)."""
    from mlflow.entities import Metric, Param
    trials_dir = Path(trials_dir or config.TRIALS_DIR)
    folder = Path(folder or config.MLFLOW_DIR)
    c = client(folder)
    have = logged(c)
    experiments: dict[str, str] = {}
    files = sorted(trials_dir.glob("*.json"))
    new, changed = [], []
    for f in files:
        rel = _rel(f)
        raw = f.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha in have.get(rel, []):
            continue
        if rel in have:
            changed.append(rel)
        trial = json.loads(raw)
        kind = str(trial.get("kind") or "unknown")
        if kind not in experiments:
            name = f"{config.MLFLOW_EXPERIMENT_PREFIX}{kind}"
            got = c.get_experiment_by_name(name)
            experiments[kind] = got.experiment_id if got else c.create_experiment(
                name, artifact_location=(folder / "artifacts" / name).as_uri())
        start = _start_ms(trial)
        run = c.create_run(experiments[kind], start_time=start, tags=tags_of(trial, rel, sha),
                           run_name=str(trial.get("trial_id") or f.stem))
        rid = run.info.run_id
        params = [Param(k, v) for k, v in params_of(trial).items()]
        for i in range(0, len(params), 100):
            c.log_batch(rid, params=params[i:i + 100])
        metrics = [Metric(k, v, start or 0, 0) for k, v in metrics_of(trial).items()]
        if metrics:
            c.log_batch(rid, metrics=metrics)
        c.log_artifact(rid, str(f))
        c.set_terminated(rid, status="FINISHED", end_time=start)
        new.append(rel)
        have.setdefault(rel, []).append(sha)
    names = {_rel(f) for f in files}
    return {"files": len(files), "in_mlflow": sum(1 for n in names if n in have), "new": len(new),
            "changed": changed, "only_in_mlflow": sorted(set(have) - names), "store": uri(folder)}


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(config.ROOT).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def line(out: dict) -> str:
    if out["in_mlflow"] == out["files"]:
        s = f"Tests recorded: {out['files']} trial files, all of them in MLflow"
    else:
        s = f"Tests recorded: {out['files']} trial files, only {out['in_mlflow']} in MLflow"
    s += f" ({out['new']} new today)." if out["new"] else " (nothing new)."
    if out["changed"]:
        s += f" {len(out['changed'])} trial file(s) CHANGED after they were logged: {', '.join(out['changed'][:3])}."
    return s
