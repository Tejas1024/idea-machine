"""Trial files (likhith.md R14): every test run leaves ONE tracked JSON in plan/work/trials/.

record() is the ONE function in the whole project that writes one. The referee (L5) counts versions per idea from
these files, never from memory — so Tejas and Likhith see the same count of tries on both computers.
Never delete a file here. A run that is only a PART of a bigger recorded trial (the referee's 200 random-entry runs)
passes record=False to the tester; the bigger trial's own file says so.

Each file holds: when, who, the idea / version / parent idea, the git commit (and any changed tracked files), the
data's content hash, every setting, and the result.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machine import config

SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
NEEDED = ("kind", "idea_id", "version_id", "settings", "data", "result")


def record(*, kind: str, idea_id: str, version_id: str, parent_idea: str | None, settings: dict, data: dict,
           result: dict, note: str | None = None) -> Path:
    """Write one trial file and return its path. Refuses to overwrite anything."""
    for name, value in (("kind", kind), ("idea_id", idea_id), ("version_id", version_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"a trial needs {name}")
    if not data:
        raise ValueError("a trial needs its data fingerprint (the content hash of the prices it ran on)")
    now = dt.datetime.now(dt.timezone.utc)
    trial_id = f"{now:%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex[:6]}"
    body = {
        "trial_id": trial_id,
        "kind": kind,
        "made_at": now.astimezone().isoformat(timespec="seconds"),
        "who": _git("config", "user.name") or os.environ.get("USERNAME") or os.environ.get("USER") or "?",
        "idea_id": idea_id,
        "version_id": version_id,
        "parent_idea": parent_idea,
        "git": git_state(),
        "data": data,
        "settings": settings,
        "result": result,
        "note": note,
    }
    folder = Path(config.TRIALS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{trial_id}-{SAFE.sub('_', idea_id)}-{SAFE.sub('_', version_id)}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("x", encoding="utf-8") as f:                 # "x": never write over an existing file
        json.dump(body, f, indent=2, ensure_ascii=False, default=_plain)
        f.write("\n")
    if path.exists():
        tmp.unlink()
        raise FileExistsError(f"{path} already exists; a trial file is never overwritten")
    os.replace(tmp, path)
    return path


def read_all(folder: Path | None = None) -> list[dict]:
    """Every trial file, oldest first (the referee's counts start here)."""
    folder = Path(folder or config.TRIALS_DIR)
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def git_state() -> dict:
    """The commit the code ran at, and the tracked files that differed from it (untracked files are ignored)."""
    commit = _git("rev-parse", "HEAD")
    changed = _git("status", "--porcelain", "--untracked-files=no", strip=False)   # "XY path": keep the spaces
    return {"commit": commit or "unknown (no git)",
            "changed_files": [line[3:] for line in (changed or "").splitlines() if line.strip()]}


def code_sha256(paths: list[Path]) -> str:
    """A fingerprint of source files that is the same on every PC: Windows line endings (CRLF — git writes them on
    a PC with core.autocrlf) count as Unix ones (LF), so the same code gives the same hash everywhere."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def sha256_of(obj: Any) -> str:
    """A fingerprint of any JSON-able thing (signals, settings)."""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_plain)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args: str, strip: bool = True) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() if strip else out.stdout


def _plain(x: Any) -> Any:
    if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)):
        return x.isoformat()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if math.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    if isinstance(x, Path):
        return x.as_posix()
    raise TypeError(f"a trial file cannot hold a {type(x).__name__}")
