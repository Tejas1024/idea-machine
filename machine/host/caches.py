"""Caches (L16.1 step 1): ready before the machine needs them; old ones removed.

    bars      every timeframe built for TODAY's data and code (data/cache/bars_<tf>_<data>_<code>.parquet), so the day's
              first idea run does not wait; a file of other data or older code is removed once it is OLD_AFTER_DAYS old
              (a fresher one may belong to a Claude session's worktree working on other code - it uses data/cache too)
    search    the document search (L8b) re-reads his files; only pieces that changed are embedded again
    talker    his ideas' talker answers (data/cache/talker): kept - each one saves a Claude call; only counted
"""
from __future__ import annotations

import time
from pathlib import Path

from machine import config

OLD_AFTER_DAYS = 3


def tidy(now: float | None = None) -> dict:
    """Returns what was built, removed and kept (counts and sizes)."""
    from machine.data import bars
    now = now or time.time()
    folder = Path(config.BARS_CACHE_DIR)
    content = bars.m1_fingerprint()
    code = bars._code_fingerprint()                        # the same name bars.py gives its cache files
    want = {f"bars_{tf}_{content[:16]}_{code[:12]}.parquet" for tf in bars.TIMEFRAMES if tf != "1m"}
    built = []
    for tf in bars.TIMEFRAMES:
        if tf == "1m":                                     # read straight from data/gold_m1.parquet: no cache
            continue
        if not (folder / f"bars_{tf}_{content[:16]}_{code[:12]}.parquet").is_file():
            bars._all_bars(tf, content)                    # builds and writes it
            built.append(tf)
    removed, freed, kept_old = [], 0, 0
    for f in sorted(folder.glob("bars_*.parquet*")) if folder.is_dir() else []:
        if f.name in want:
            continue
        age_days = (now - f.stat().st_mtime) / 86400
        if age_days < OLD_AFTER_DAYS and not f.name.endswith(".tmp"):
            kept_old += 1
            continue
        size = f.stat().st_size
        try:
            f.unlink()
        except OSError:                                    # open in another program right now: next time
            kept_old += 1
            continue
        removed.append(f.name)
        freed += size
    out = {"bars_built": built, "bars_ready": len(want), "removed": removed, "freed_mb": round(freed / 1e6, 1),
           "old_kept": kept_old, "search": _search(), "talker": _count(folder / "talker", "*.json")}
    out["size_mb"] = round(sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6, 1) \
        if folder.is_dir() else 0.0
    return out


def _search() -> str:
    try:
        from machine.search import index
        t0 = time.time()
        idx = index.build()
        return f"{len(idx.pieces):,} pieces of his files, {idx.how} ({time.time() - t0:.0f} s)"
    except Exception as e:                                 # the search is not the daily run's job: say it, go on
        return f"could not be refreshed ({type(e).__name__}: {str(e)[:120]})"


def _count(folder: Path, pattern: str) -> dict:
    files = list(folder.glob(pattern)) if folder.is_dir() else []
    return {"files": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}


def line(out: dict) -> str:
    """One line for the daily summary, in plain words."""
    parts = [f"{out['bars_ready']} price tables ready"]
    if out["bars_built"]:
        parts.append(f"{len(out['bars_built'])} built new ({', '.join(out['bars_built'])})")
    if out["removed"]:
        parts.append(f"{len(out['removed'])} old ones removed ({out['freed_mb']} MB)")
    return "Caches: " + ", ".join(parts) + f"; {out['size_mb']} MB in all."
