"""The guards every loader, block and tool must pass. Each refusal says why, in Tejas's words where he gave them.

- check_dates       the sealed years (2024, 2025, 2026) are never touched for testing (LEARNED #96, #104); opened only
                    for the three uses Tejas said yes to in T14 (SEALED_USES).
- assert_no_future  the "cut off the future" test: a value may use only what was known at its own time (R4).
- assert_no_future_events  the same for event tables (a row per swing, gap, break ...): a row known at t may use
                    only the bars that had closed by t (L6.1).
- check_opened      the check years (2021-2023) stay hidden until the idea's check-opened.json exists.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machine import config

SEALED_WORDS = "Tejas, LEARNED #96: \"No, don't touch those years\" (also #104)"
IDEA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class GuardError(Exception):
    """A guard refused."""


class SealedYearsError(GuardError):
    """Something touched 2024 or later."""


class CheckYearsClosedError(GuardError):
    """Something touched 2021-2023 before this idea's check years were opened."""


class FutureLeakError(GuardError, AssertionError):
    """A past value changed when the future was cut off: the function peeks."""


# ---- the sealed years -------------------------------------------------------------------------------------------

# The only uses the sealed years can ever be opened for (T14, 29 Sep 2026). Each opens only when
# plan/work/T14-decisions.md has the line "use: <name> = yes". Never for testing, tuning or choosing (#96, and T14
# question 3: "I hope like it won't learning and all for these years").
SEALED_USES = {
    "live-trial": "live prices from today onward (+ the recent days an alert needs for its levels): alerts and paper "
                  "trades only (T14 question 1: \"Yes\")",
    "news-values": "news numbers from 2024 onward, read only by the live news alerts (T14 question 2: Claude's "
                   "choice on his \"Do whatever you think is best.\")",
    "final-exam": "2024-2026 prices ONE time, a final exam for ideas that passed everything; needs his yes again on "
                  "that day in plan/work/final-exam-yes.md (T14 question 3)",
}
_USE_LINE = re.compile(r"^\s*use:\s*([a-z-]+)\s*=\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)


def check_dates(obj: Any, unlock: bool = False, who: str | None = None, why: str | None = None,
                use: str | None = None) -> Any:
    """Refuse anything that touches a time on or after config.SEALED_FROM. Returns obj unchanged when it passes.

    obj: a DataFrame / Series / Index (its index and every date column are checked), a single time (Timestamp,
    datetime, date, "2024-01-02"), or a list / tuple of those — e.g. a (start, end) request.
    Naive times are compared on their own clock (the raw data is MT5 server time). Times with a time zone are
    refused if EITHER their own clock or UTC is in the sealed years (the safe side).
    An object with no dates at all is refused too: a loader that forgot to parse its times must not slip past.

    unlock=True opens the sealed years only with who, why and a `use` from SEALED_USES that Tejas said yes to in
    plan/work/T14-decisions.md ("use: <name> = yes"; the file merely existing opens nothing). "final-exam" also needs
    his yes again on that day (config.FINAL_EXAM_YES). Every unlock appends one line to data/sealed_access.log.
    """
    latest = _latest(obj)
    if latest is None or latest < pd.Timestamp(config.SEALED_FROM):
        return obj
    if not unlock:
        raise SealedYearsError(
            f"This touches {latest}; the sealed years start {config.SEALED_FROM}. {SEALED_WORDS}. "
            f"Unlocking needs a use Tejas said yes to in {_show(config.T14_DECISIONS)}, then unlock=True with who, "
            f"why and use."
        )
    if not str(who or "").strip() or not str(why or "").strip():
        raise SealedYearsError(f"unlock=True needs who and why; both go into {_show(config.SEALED_LOG)}.")
    if use not in SEALED_USES:
        raise SealedYearsError(
            f"unlock=True needs use = one of {', '.join(SEALED_USES)} (T14); got {use!r}. Testing, tuning and "
            f"choosing are never a use. {SEALED_WORDS}."
        )
    if not config.T14_DECISIONS.is_file():
        raise SealedYearsError(
            f"unlock=True refused: {_show(config.T14_DECISIONS)} does not exist. Until Tejas writes his yes there "
            f"(prompt T14), the answer is no. {SEALED_WORDS}."
        )
    if allowed_uses().get(use) is not True:
        raise SealedYearsError(
            f"unlock=True refused: {_show(config.T14_DECISIONS)} has no line \"use: {use} = yes\". Only Tejas's "
            f"words can add it. {SEALED_WORDS}."
        )
    if use == "final-exam":
        _check_final_exam_yes()
    _log_unlock(latest, str(who), str(why), use)
    return obj


def allowed_uses(path: Path | None = None) -> dict[str, bool]:
    """The "use: <name> = yes|no" lines of the T14 file. A name written twice: the LAST line wins (his newest word)."""
    path = path or config.T14_DECISIONS
    if not path.is_file():
        return {}
    found = {name.lower(): answer.lower() == "yes"
             for name, answer in _USE_LINE.findall(path.read_text(encoding="utf-8"))}
    unknown = sorted(set(found) - set(SEALED_USES))
    if unknown:
        raise SealedYearsError(f"{_show(path)} names a use the lock does not know: {', '.join(unknown)}.")
    return found


def _check_final_exam_yes() -> None:
    """The final exam opens only with his yes AGAIN on that day, naming the ideas (T14 question 3)."""
    path = config.FINAL_EXAM_YES
    text = path.read_text(encoding="utf-8") if path.is_file() else ""

    def line(label: str) -> str:
        hit = re.search(rf"^\s*{label}:[ \t]*(.*)$", text, re.IGNORECASE | re.MULTILINE)
        return hit.group(1).strip() if hit else ""

    if not line("his yes") or not line("ideas"):
        raise SealedYearsError(
            f"The final exam needs Tejas's yes AGAIN on that day: {_show(path)} with a \"his yes:\" line (his words) "
            f"and an \"ideas:\" line (which ideas). T14 question 3."
        )


def _log_unlock(latest: pd.Timestamp, who: str, why: str, use: str) -> None:
    def one_line(text: str) -> str:
        return " ".join(text.split())

    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{stamp}\twho={one_line(who)}\twhy={one_line(why)}\tuse={use}\ttouches up to {latest}\n"
    config.SEALED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with config.SEALED_LOG.open("a", encoding="utf-8") as log:
        log.write(line)


def _latest(obj: Any) -> pd.Timestamp | None:
    found = _times(obj, "check_dates")
    latest = [idx.max() for idx in found if len(idx)]
    return max(latest) if latest else None


def _times(obj: Any, caller: str) -> list[pd.DatetimeIndex]:
    found: list[pd.DatetimeIndex] = []
    _collect(obj, found)
    if not found:
        raise TypeError(f"{caller} found no dates in this {type(obj).__name__}; refusing.")
    return found


def _collect(obj: Any, found: list[pd.DatetimeIndex]) -> None:
    if isinstance(obj, pd.DataFrame):
        _collect(obj.index, found)
        for name in obj.columns:
            if pd.api.types.is_datetime64_any_dtype(obj[name]):
                _add(pd.DatetimeIndex(obj[name]), found)
    elif isinstance(obj, pd.Series):
        _collect(obj.index, found)
        if pd.api.types.is_datetime64_any_dtype(obj):
            _add(pd.DatetimeIndex(obj), found)
    elif isinstance(obj, pd.MultiIndex):
        for level in range(obj.nlevels):
            _collect(obj.get_level_values(level), found)
    elif isinstance(obj, pd.Index):
        if pd.api.types.is_datetime64_any_dtype(obj):
            _add(pd.DatetimeIndex(obj), found)
    elif isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.datetime64):
        _add(pd.DatetimeIndex(obj.ravel()), found)
    elif isinstance(obj, (str, dt.date, np.datetime64)):  # dt.datetime and pd.Timestamp are dt.date too
        _add(pd.DatetimeIndex([pd.Timestamp(obj)]), found)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _collect(item, found)
    else:
        raise TypeError(f"the guards cannot read dates from a {type(obj).__name__}.")


def _add(idx: pd.DatetimeIndex, found: list[pd.DatetimeIndex]) -> None:
    idx = idx[~idx.isna()]
    if idx.tz is None:
        found.append(idx)
    else:
        found.append(idx.tz_localize(None))                     # its own clock
        found.append(idx.tz_convert("UTC").tz_localize(None))   # and UTC


# ---- the cut-off-the-future test --------------------------------------------------------------------------------

def assert_no_future(
    fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    df: pd.DataFrame,
    cut_points: Sequence[Any],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """The "cut off the future" test (likhith.md R4, lesson B1).

    For each cut point t: run fn on all rows, and again on only the rows up to t. Every value fn gives for a time
    at or before t must be the same both ways. If cutting off the future changes a past value (or makes it appear
    or vanish), fn peeks at the future: FutureLeakError says which value, where, and both numbers.

    df needs a sorted index with no repeats. A cut point is an index label, or a position (int) when the index is
    made of times. fn gets a copy each time, so changing its input in place cannot hide a leak.
    """
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("assert_no_future needs a df with a sorted index and no repeated rows.")
    if len(cut_points) == 0:
        raise ValueError("assert_no_future needs at least one cut point.")

    full = _as_frame(fn(df.copy()), "all rows")
    compared = 0
    for cut in cut_points:
        t = df.index[cut] if isinstance(cut, (int, np.integer)) and isinstance(df.index, pd.DatetimeIndex) else cut
        part = df.loc[:t]
        if part.empty or len(part) == len(df):
            raise ValueError(f"cut point {t} must leave rows both before and after it.")
        cut_out = _as_frame(fn(part.copy()), f"rows up to {t}")
        want = full.loc[:t]
        _same_past(want, cut_out.loc[:t], t, rtol, atol)
        compared += want.size
    if compared == 0:
        raise ValueError("fn gave no values at or before any cut point, so nothing was compared.")


def assert_no_future_events(
    fn: Callable[[pd.DataFrame], pd.DataFrame],
    bars: pd.DataFrame,
    bar_minutes: int,
    cut_points: Sequence[int],
    *,
    known: str = "known_at",
    decimals: int = 9,
) -> None:
    """The "cut off the future" test for EVENT tables (one row per thing found, each stamped `known` = the moment it
    became known). Stricter than assert_no_future: a row known at time t may use only the bars that had CLOSED by t.

    For each cut point p (a bar position): t = the close of bar p-1 (its open + bar_minutes). fn runs on all bars and
    again on bars[:p] only. The rows known at or before t must be the same both ways - none added, none missing, none
    changed. A row stamped too early (known before the bar it needs has closed) fails here even when it is only one
    bar early.
    """
    if not isinstance(bars.index, pd.DatetimeIndex) or not bars.index.is_monotonic_increasing \
            or not bars.index.is_unique:
        raise ValueError("assert_no_future_events needs bars with a sorted time index and no repeated bar.")
    if len(cut_points) == 0:
        raise ValueError("assert_no_future_events needs at least one cut point.")
    step = pd.Timedelta(minutes=int(bar_minutes))
    full = fn(bars.copy())
    if known not in full.columns:
        raise ValueError(f"the event table has no {known!r} column.")
    compared = 0
    for p in cut_points:
        p = int(p)
        if not 0 < p < len(bars):
            raise ValueError(f"cut point {p} must leave bars both before and after it.")
        t = bars.index[p - 1] + step
        want = _rows_known_by(full, known, t, decimals)
        got = _rows_known_by(fn(bars.iloc[:p].copy()), known, t, decimals)
        if want != got:
            extra = sorted(set(want) - set(got)) or sorted(set(got) - set(want))
            where = "with" if set(want) - set(got) else "without"
            raise FutureLeakError(f"cut at {t} (bars up to {bars.index[p - 1]}): a row known by then appears only "
                                  f"{where} the future: {extra[0]}. It peeks at the future (likhith.md B1).")
        compared += len(want)
    if compared == 0:
        raise ValueError("no row was known by any cut point, so nothing was compared.")


def _rows_known_by(table: pd.DataFrame, known: str, t: Any, decimals: int) -> list[tuple]:
    rows = table[pd.to_datetime(table[known]) <= t]
    out = []
    for rec in rows.itertuples(index=False, name=None):
        out.append(tuple(("nan" if v != v else round(v, decimals)) if isinstance(v, float) else v for v in rec))
    return sorted(out, key=repr)


def _as_frame(out: Any, which: str) -> pd.DataFrame:
    if isinstance(out, pd.Series):
        out = out.to_frame()
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"fn must return a Series or DataFrame (on {which} it gave a {type(out).__name__}).")
    return out.sort_index()


def _same_past(want: pd.DataFrame, got: pd.DataFrame, t: Any, rtol: float, atol: float) -> None:
    if list(want.columns) != list(got.columns):
        raise FutureLeakError(f"cut at {t}: fn gives columns {list(want.columns)} with the future and "
                              f"{list(got.columns)} without it.")
    only_full = want.index.difference(got.index)
    only_cut = got.index.difference(want.index)
    if len(only_full) or len(only_cut):
        where = only_full[0] if len(only_full) else only_cut[0]
        raise FutureLeakError(f"cut at {t}: at {where} fn gives a value only "
                              f"{'with' if len(only_full) else 'without'} the future. It peeks.")
    got = got.reindex(want.index)
    for col in want.columns:
        a, b = want[col], got[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            bad = ~np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=rtol, atol=atol, equal_nan=True)
        else:
            bad = ~((a == b) | (a.isna() & b.isna())).to_numpy(dtype=bool)
        if bad.any():
            i = int(np.argmax(bad))
            raise FutureLeakError(f"cut at {t}: {col!r} at {want.index[i]} is {a.iloc[i]} with the future and "
                                  f"{b.iloc[i]} without it. It peeks at the future (likhith.md B1).")


# ---- the hidden check years -------------------------------------------------------------------------------------

def check_opened(idea: str, obj: Any = None, version: str | None = None) -> dict | None:
    """Refuse the check years (config.CHECK_YEARS, 2021-2023) for this idea until
    plan/work/ideas/<idea>/check-opened.json exists (the referee, L5, writes it when the check is opened ONCE).

    Without obj: refuses whenever the file is missing — call it right before opening the check years.
    With obj (data, a time, a (start, end) request): refuses only if obj touches a check year, so data from the
    tune years alone passes without the file (returns None then).
    version: the version asking. The check years open ONCE per idea family, for ONE version (L5): if the file names
    another version, this one is refused — later versions are judged by walk-forward inside 2016-2020.
    Returns the file's contents when it was read.
    """
    if not isinstance(idea, str) or not IDEA_NAME.fullmatch(idea) or ".." in idea:
        raise ValueError(f"not an idea name: {idea!r} (expected something like 'idea-001').")
    if obj is not None:
        years = {int(y) for idx in _times(obj, "check_opened") for y in idx.year}
        touched = sorted(years & set(config.CHECK_YEARS))
        if not touched:
            return None
    else:
        touched = list(config.CHECK_YEARS)
    path = config.IDEAS_DIR / idea / "check-opened.json"
    span = str(touched[0]) if len(touched) == 1 else f"{touched[0]}-{touched[-1]}"
    why = f"The check years ({span}) stay hidden while choosing and open ONCE per idea (plan/README.md section 7)."
    if not path.is_file():
        raise CheckYearsClosedError(f"{why} {idea} has no {_show(path)} yet.")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise CheckYearsClosedError(f"{why} {_show(path)} cannot be read as JSON.") from err
    if not isinstance(record, dict):
        raise CheckYearsClosedError(f"{why} {_show(path)} must hold a JSON object.")
    opened_for = record.get("version_id")
    if version is not None and opened_for is not None and opened_for != version:
        raise CheckYearsClosedError(
            f"{why} They were opened for {idea} version {opened_for} on {record.get('opened_at', '?')}; "
            f"version {version} is a later version: it is judged by walk-forward inside 2016-2020 "
            "(machine/referee/splits.py), never on the check years.")
    return record


def _show(path: Path) -> str:
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return str(path)
