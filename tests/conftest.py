"""Shared test setup: tests marked @pytest.mark.data or @pytest.mark.mt5 skip themselves when their need is missing.

    data  the 3 price files in old/build/data/ (💻) — not in git, copied by USB or drive
    mt5   Windows + the MetaTrader5 package (🪟) — requirements-live.txt
    search_model  the document search's local model (requirements-search.txt) — the search exam (L8b)
    news  the news calendar data/news/events.parquet (🪟 made from MT5 on Tejas's PC, L12)

The public copy (L11, machine/tools/publish.py) leaves out his private folders (PRIVATE-PARTS.md at its root names
them). There, a test that fails ONLY because it needs a file in one of those folders is reported as skipped, with the
file it needs; marked @pytest.mark.private, a test that needs them in another way skips too. In this repo the marker
file does not exist, so nothing here is ever skipped that way.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from machine import config as mconfig

# The public copy's marker file and the folders it leaves out (machine/tools/publish.py writes both lists).
PRIVATE_NOTE = Path(mconfig.ROOT) / "PRIVATE-PARTS.md"
PRIVATE_ROOTS = ("strategy", "old", "data", "plan", ".github")


def is_public_copy() -> bool:
    return PRIVATE_NOTE.is_file()


def _private_rel(name: str) -> str | None:
    """`name` as a path inside this repo under PRIVATE_ROOTS that does not exist, else None."""
    p = Path(name)
    p = p if p.is_absolute() else Path(mconfig.ROOT) / p
    try:
        rel = p.resolve().relative_to(Path(mconfig.ROOT).resolve()).as_posix()
    except (ValueError, OSError):
        return None
    if not p.exists() and any(rel == r or rel.startswith(r + "/") for r in PRIVATE_ROOTS):
        return rel
    return None


def private_file_missing(exc: BaseException | None) -> str | None:
    """The private file a failure needs (a path under PRIVATE_ROOTS that does not exist), in the public copy only.
    The path is the error's filename, or a path inside this repo named in its message (the price loader says
    "<root>/data/gold_m1.parquet is missing - run ...")."""
    if not is_public_copy():
        return None
    root = str(Path(mconfig.ROOT).resolve())
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, FileNotFoundError):
            names = [str(exc.filename)] if exc.filename else []
            names += re.findall(re.escape(root) + r"[\\/][^\s'\"]+", str(exc))
            for name in names:
                rel = _private_rel(name)
                if rel:
                    return rel
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        rel = private_file_missing(call.excinfo.value)
        if rel:
            rep.outcome = "skipped"
            rep.longrepr = (str(item.path), item.location[1] or 0,
                            f"private: needs {rel}, which is not in the public copy (PRIVATE-PARTS.md)")


def have_price_data() -> bool:
    return all(p.is_file() for p in (mconfig.RAW_M1, mconfig.RAW_M5, mconfig.RAW_DUKA))


def have_mt5() -> bool:
    return sys.platform == "win32" and importlib.util.find_spec("MetaTrader5") is not None


def have_news() -> bool:
    return mconfig.NEWS_EVENTS.is_file()


def have_search_model() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


@pytest.fixture(autouse=True)
def _trials_in_tmp(tmp_path, monkeypatch):
    """No test ever writes into the tracked plan/work/trials/ — the count of tries must stay the real one (R14)."""
    folder = tmp_path / "trials"
    monkeypatch.setattr(mconfig, "TRIALS_DIR", folder)
    # ... and never into the tracked list of pictures Tejas has seen (#117, L4)
    monkeypatch.setattr(mconfig, "SHOWN_PICTURES", tmp_path / "pictures-shown.jsonl")
    # ... and never opens the real check years: check-opened.json files go to a temporary ideas folder (L5)
    monkeypatch.setattr(mconfig, "IDEAS_DIR", tmp_path / "ideas")
    # ... and never writes a belief's chosen version or its pages into the tracked plan/work/beliefs/ (L12b.2)
    monkeypatch.setattr(mconfig, "BELIEFS_DIR", tmp_path / "beliefs")
    # ... and never touches his label charts, his replies, his T9 answer or the label file (L15.1)
    monkeypatch.setattr(mconfig, "LABELS_DIR", tmp_path / "labels-data")
    monkeypatch.setattr(mconfig, "LABELS_WORK_DIR", tmp_path / "labels")
    monkeypatch.setattr(mconfig, "LABELS_OPT_IN", tmp_path / "labels" / "opt-in.md")
    return folder


@pytest.fixture(autouse=True)
def _no_research_planner_call(monkeypatch):
    """No test reaches Claude through Tejas's login by accident (his plan's usage, L14.2): the research agent's own
    planner refuses - a test gives the agent a planner of its own."""
    from machine.agent import research

    def refuse(*a, **k):
        raise AssertionError("a test reached Claude (the research agent's planner) - give the agent a planner")
    monkeypatch.setattr(research, "claude_planner", refuse)


def pytest_collection_modifyitems(config, items):
    skip_data = pytest.mark.skip(reason="needs the 3 price files in old/build/data/ (copy them by USB or drive)")
    skip_mt5 = pytest.mark.skip(reason="needs Windows + MetaTrader5 (pip install -r requirements-live.txt)")
    skip_model = pytest.mark.skip(reason="needs the search model (pip install -r requirements-search.txt)")
    skip_news = pytest.mark.skip(reason="needs data/news/events.parquet (python -m machine.news.build, L12)")
    skip_private = pytest.mark.skip(reason="private: needs his notes or the old project, which are not in the public "
                                           "copy (PRIVATE-PARTS.md)")
    data_ok, mt5_ok, model_ok, news_ok = have_price_data(), have_mt5(), have_search_model(), have_news()
    public = is_public_copy()
    for item in items:
        if item.get_closest_marker("private") and public:
            item.add_marker(skip_private)
        if item.get_closest_marker("data") and not data_ok:
            item.add_marker(skip_data)
        if item.get_closest_marker("mt5") and not mt5_ok:
            item.add_marker(skip_mt5)
        if item.get_closest_marker("search_model") and not model_ok:
            item.add_marker(skip_model)
        if item.get_closest_marker("news") and not news_ok:
            item.add_marker(skip_news)
