"""Which years judge what (L5, step 2).

- TUNE years 2016-2020: used for choosing between versions (config.TUNE_YEARS).
- CHECK years 2021-2023: hidden while choosing; opened ONCE per idea FAMILY, for ONE version — the one chosen on the
  tune years. open_check() writes plan/work/ideas/<idea>/check-opened.json (tracked in git) and refuses a second
  opening anywhere in the family. The tester's guard (machine.guards.check_opened) then lets only that version in.
- LATER versions (after the family's check years were opened) are judged by WALK-FORWARD inside 2016-2020: for each
  judge year (2018, 2019, 2020) the version is chosen on the years before it only, then judged on that year.
- Every report says "CHECK years opened __ times in the project" (opened_count()).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

from machine import config, guards
from machine import trials as trial_files
from machine.referee import trials


@dataclass(frozen=True)
class Fold:
    choose_years: tuple[int, ...]       # the version is chosen on these (all tune years before the judge year)
    judge_year: int                     # and judged on this one


def years_text(years) -> str:
    years = sorted(years)
    return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"


def window(years) -> tuple[str, str]:
    """(start, end) for load_bars: the first day of the first year to the last minute of the last year."""
    years = sorted(years)
    return f"{years[0]}-01-01", f"{years[-1]}-12-31 23:59"


def walk_forward_folds() -> list[Fold]:
    """Choose on the years before, judge the next: (2016-2017 -> 2018), (2016-2018 -> 2019), (2016-2019 -> 2020)."""
    folds = []
    for y in config.WALK_FORWARD_JUDGE_YEARS:
        before = tuple(t for t in config.TUNE_YEARS if t < y)
        if y not in config.TUNE_YEARS or len(before) < 2:
            raise ValueError(f"walk-forward judge year {y} needs 2 tune years before it, inside {config.TUNE_YEARS}")
        folds.append(Fold(before, y))
    return folds


# ---- the check years, opened once per family ---------------------------------------------------------------------

def check_files() -> list[tuple[Path, dict]]:
    """Every plan/work/ideas/*/check-opened.json, with its contents (an unreadable file is refused, not skipped)."""
    out = []
    for path in sorted(Path(config.IDEAS_DIR).glob("*/check-opened.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise ValueError(f"{path} cannot be read as JSON") from err
        out.append((path, record if isinstance(record, dict) else {}))
    return out


def opened_count() -> int:
    """How many times the check years were opened in the whole project (each file = one opening)."""
    return len(check_files())


def family_opening(idea: str, parent: str | None = None) -> dict | None:
    """The check-opened record of this idea's family, or None if the family never opened its check years."""
    links = trials.parents(trials.load())
    family = trials.root_of(idea, links, parent)
    for path, record in check_files():
        owner = record.get("idea_id") or path.parent.name
        fam = record.get("family") or trials.root_of(owner, links)
        if fam == family:
            return {**record, "idea_id": owner, "family": fam, "file": _rel(path)}
    return None


def status(idea: str, version: str, parent: str | None = None) -> str:
    """"check" (the check years were opened for exactly this idea and version), "walk-forward" (the family opened
    them for another version: judge this one inside 2016-2020), or "closed" (the family never opened them)."""
    rec = family_opening(idea, parent)
    if rec is None:
        return "closed"
    if rec["idea_id"] == idea and rec.get("version_id") == version:
        return "check"
    return "walk-forward"


def open_check(idea: str, version: str, parent: str | None = None, *, who: str, why: str) -> dict:
    """Open the check years ONCE for this idea family, for this version. Returns the record written.

    Refused when: the family already opened them (any idea, any version); this version has no trial file yet (it
    must have been tried on the tune years first); the family is over its version budget; who or why is missing.
    """
    if not str(who).strip() or not str(why).strip():
        raise ValueError("opening the check years needs who and why (they go into check-opened.json)")
    if not guards.IDEA_NAME.fullmatch(idea) or ".." in idea:
        raise ValueError(f"not an idea name: {idea!r}")
    earlier = family_opening(idea, parent)
    if earlier is not None:
        raise guards.CheckYearsClosedError(
            f"The check years ({years_text(config.CHECK_YEARS)}) open ONCE per idea family. The {earlier['family']} "
            f"family opened them on {earlier.get('opened_at', '?')} for {earlier['idea_id']} version "
            f"{earlier.get('version_id', '?')} ({earlier['file']}). {idea} {version} is judged by walk-forward "
            "inside 2016-2020 (machine.referee.judge.judge_walk_forward).")
    c = trials.count(idea, parent)
    if version not in c.idea_versions:
        raise ValueError(f"{idea} version {version} has no trial file in plan/work/trials/: a version is tried on the "
                         f"tune years ({years_text(config.TUNE_YEARS)}) before its check years open")
    if len(c.family_versions) > c.budget:
        raise ValueError(f"the {c.family} family has {len(c.family_versions)} versions, over the budget {c.budget}")
    record = {
        "idea_id": idea,
        "version_id": version,
        "parent_idea": parent,
        "family": c.family,
        "check_years": list(config.CHECK_YEARS),
        "opened_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "who": str(who),
        "why": str(why),
        "family_versions_tried": len(c.family_versions),
        "git": trial_files.git_state(),
        "rule": "opened ONCE per idea family (plan/likhith.md L5); later versions: walk-forward inside 2016-2020",
    }
    folder = Path(config.IDEAS_DIR) / idea
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "check-opened.json"
    tmp = folder / "check-opened.json.tmp"
    with tmp.open("x", encoding="utf-8") as f:                      # never over an existing file
        json.dump(record, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if path.exists():
        tmp.unlink()
        raise FileExistsError(f"{path} already exists; the check years open once")
    os.replace(tmp, path)
    return record


def _rel(path: Path) -> str:
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
