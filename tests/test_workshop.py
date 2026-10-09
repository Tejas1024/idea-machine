"""The workshop itself: no imports from old/, a docstring in every package, empty keys in .env.example, the markers."""
import importlib
import re

import pytest

from machine import config

PACKAGES = ["data", "blocks", "tester", "referee", "ideas", "talker", "search", "questions", "report", "intake",
            "learner", "news", "beliefs", "charts", "telegram", "live", "money", "tools", "agent", "mcp", "entries",
            "dashboard"]
KEYS = ["ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_TEST_BOT_TOKEN",
        "TELEGRAM_TEST_CHAT_ID", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "HEARTBEAT_PING_URL"]


def test_machine_never_imports_old():
    """likhith.md R2: new code in machine/ never imports from old/."""
    bad = []
    for path in (config.ROOT / "machine").rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import)\s+old\b", line) or "sys.path" in line and "old" in line:
                bad.append(f"{path.relative_to(config.ROOT)}:{n}")
    assert not bad


@pytest.mark.parametrize("name", PACKAGES)
def test_every_package_has_a_one_line_docstring(name):
    doc = importlib.import_module(f"machine.{name}").__doc__
    assert doc and doc.strip() and "\n" not in doc.strip()


def test_env_example_has_every_key_and_no_values():
    lines = [ln for ln in (config.ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    pairs = dict(ln.split("=", 1) for ln in lines)
    assert list(pairs) == KEYS
    assert all(v.strip() == "" for v in pairs.values())


def test_config_settings():
    assert config.COST_INR_PER_POSITION == 40 and config.LOT_STEP == 0.01
    assert config.SEALED_FROM == "2024-01-01"
    assert config.TUNE_YEARS == (2016, 2017, 2018, 2019, 2020) and config.CHECK_YEARS == (2021, 2022, 2023)
    assert (config.VERSION_BUDGET, config.SLICE_BUDGET, config.QUESTION_BUDGET) == (20, 8, 10)
    assert config.MIN_SPREAD_POINTS > 0


@pytest.mark.data
def test_the_price_files_are_here():
    """Only looks at the files' sizes; never opens them (L0 reads no price data)."""
    for path in (config.RAW_M1, config.RAW_M5, config.RAW_DUKA):
        assert path.stat().st_size > 1_000_000, path


@pytest.mark.mt5
def test_metatrader5_imports():
    mt5 = importlib.import_module("MetaTrader5")
    assert hasattr(mt5, "initialize")
