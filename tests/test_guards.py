"""The guards: the sealed years, the cut-off-the-future test, the hidden check years. Made-up prices only."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from machine import config
from machine.guards import (
    CheckYearsClosedError,
    FutureLeakError,
    SealedYearsError,
    allowed_uses,
    assert_no_future,
    check_dates,
    check_opened,
)


@pytest.fixture(autouse=True)
def tmp_files(tmp_path, monkeypatch):
    """Every guard file points into a temporary folder: no test touches the real log or plan/work/."""
    monkeypatch.setattr(config, "SEALED_LOG", tmp_path / "data" / "sealed_access.log")
    monkeypatch.setattr(config, "T14_DECISIONS", tmp_path / "T14-decisions.md")
    monkeypatch.setattr(config, "FINAL_EXAM_YES", tmp_path / "final-exam-yes.md")
    monkeypatch.setattr(config, "IDEAS_DIR", tmp_path / "ideas")
    return tmp_path


def bars(start, periods, freq="1min"):
    """Made-up 1-minute closes (a random walk), not gold data."""
    idx = pd.date_range(start, periods=periods, freq=freq)
    close = 1800 + np.random.default_rng(7).normal(0, 0.5, periods).cumsum()
    return pd.DataFrame({"close": close}, index=idx)


# ---- sealed years -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("obj", [
    pd.Timestamp("2024-01-01 00:00"),
    "2024-03-05",
    dt.date(2025, 6, 1),
    ("2016-01-04", "2024-06-01"),                                   # a (start, end) request
    bars("2023-12-29 23:00", 3000),                                 # runs into 1 Jan 2024
    pd.DataFrame({"entry": pd.to_datetime(["2023-12-28 10:00"]),    # a trade that closes in 2024
                  "exit": pd.to_datetime(["2024-01-02 09:00"])}),
    pd.Timestamp("2024-01-01 03:00", tz="Asia/Kolkata"),            # 2023 in UTC, 2024 on its own clock
])
def test_2024_raises(obj):
    with pytest.raises(SealedYearsError, match="don't touch those years"):
        check_dates(obj)


def test_last_minute_of_2023_passes():
    assert check_dates(pd.Timestamp("2023-12-31 23:59")) == pd.Timestamp("2023-12-31 23:59")
    df = bars("2023-12-31 23:00", 60)
    assert df.index[-1] == pd.Timestamp("2023-12-31 23:59")
    assert check_dates(df) is df


@settings(max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.datetimes(min_value=dt.datetime(2016, 1, 1), max_value=dt.datetime(2030, 12, 31)))
def test_sealed_exactly_from_2024(when):
    if when >= dt.datetime(2024, 1, 1):
        with pytest.raises(SealedYearsError):
            check_dates(when)
    else:
        check_dates(when)


def test_no_dates_is_refused():
    with pytest.raises(TypeError, match="no dates"):
        check_dates(pd.DataFrame({"close": [1.0, 2.0]}))
    with pytest.raises(TypeError):
        check_dates(42)


def test_unlock_logs_one_line(tmp_files):
    config.T14_DECISIONS.write_text("made up for this test\nuse: live-trial = yes\n", encoding="utf-8")
    ts = pd.Timestamp("2024-02-01 10:00")
    assert check_dates(ts, unlock=True, who="Likhith", why="unit test\nwith a second line", use="live-trial") == ts
    lines = config.SEALED_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "who=Likhith" in lines[0] and "why=unit test with a second line" in lines[0] and "2024-02-01" in lines[0]
    assert "use=live-trial" in lines[0]
    check_dates(ts, unlock=True, who="Likhith", why="again", use="live-trial")
    assert len(config.SEALED_LOG.read_text(encoding="utf-8").splitlines()) == 2   # appended, never rewritten


@pytest.mark.parametrize("text", [
    "no\n",                                                      # the file exists but says nothing the lock reads
    "use: live-trial = no\n",                                    # his no
    "use: news-values = yes\n",                                  # a yes to a DIFFERENT use
    "use: live-trial = yes\nuse: live-trial = no\n",             # he changed it later: the last line wins
])
def test_the_file_existing_opens_nothing(tmp_files, text):
    """Before T14 the lock opened when the file merely existed, so even a written "no" would have opened it."""
    config.T14_DECISIONS.write_text(text, encoding="utf-8")
    with pytest.raises(SealedYearsError, match="no line"):
        check_dates(pd.Timestamp("2024-02-01"), unlock=True, who="Likhith", why="test", use="live-trial")
    assert not config.SEALED_LOG.exists()


@pytest.mark.parametrize("use", [None, "testing", "tune", "LIVE-TRIAL "])
def test_unlock_needs_a_known_use(tmp_files, use):
    config.T14_DECISIONS.write_text("use: live-trial = yes\n", encoding="utf-8")
    with pytest.raises(SealedYearsError, match="use = one of"):
        check_dates(pd.Timestamp("2024-02-01"), unlock=True, who="Likhith", why="test", use=use)


def test_an_unknown_use_line_is_refused(tmp_files):
    config.T14_DECISIONS.write_text("use: live-trial = yes\nuse: tuning = yes\n", encoding="utf-8")
    with pytest.raises(SealedYearsError, match="does not know: tuning"):
        check_dates(pd.Timestamp("2024-02-01"), unlock=True, who="Likhith", why="test", use="live-trial")


def test_final_exam_needs_his_yes_again_on_the_day(tmp_files):
    config.T14_DECISIONS.write_text("use: final-exam = yes\n", encoding="utf-8")
    ts = pd.Timestamp("2025-06-01")
    with pytest.raises(SealedYearsError, match="AGAIN"):                                   # no second file
        check_dates(ts, unlock=True, who="Likhith", why="the final exam", use="final-exam")
    config.FINAL_EXAM_YES.write_text("his yes: \nideas: idea-001\n", encoding="utf-8")     # his words missing
    with pytest.raises(SealedYearsError, match="AGAIN"):
        check_dates(ts, unlock=True, who="Likhith", why="the final exam", use="final-exam")
    config.FINAL_EXAM_YES.write_text("his yes: yes, open it (made up)\n", encoding="utf-8")  # no ideas named
    with pytest.raises(SealedYearsError, match="AGAIN"):
        check_dates(ts, unlock=True, who="Likhith", why="the final exam", use="final-exam")
    assert not config.SEALED_LOG.exists()
    config.FINAL_EXAM_YES.write_text("his yes: yes, open it (made up)\nideas: idea-001\n", encoding="utf-8")
    assert check_dates(ts, unlock=True, who="Likhith", why="the final exam", use="final-exam") == ts
    assert "use=final-exam" in config.SEALED_LOG.read_text(encoding="utf-8")


@pytest.mark.private
def test_the_real_T14_file_says_what_he_answered():
    """plan/work/T14-decisions.md, 29 Sep 2026: yes / yes (Claude's choice on his words) / yes with his yes again."""
    real = config.ROOT / "plan" / "work" / "T14-decisions.md"
    assert allowed_uses(real) == {"live-trial": True, "news-values": True, "final-exam": True}
    text = real.read_text(encoding="utf-8")
    for his_words in ("> Yes", "> Do whatever you think is best.", "> If you think it's best then fine."):
        assert his_words in text


def test_unlock_refused_without_his_yes_or_without_who_and_why(tmp_files):
    ts = pd.Timestamp("2024-02-01")
    with pytest.raises(SealedYearsError, match="does not exist"):
        check_dates(ts, unlock=True, who="Likhith", why="test", use="live-trial")
    config.T14_DECISIONS.write_text("yes\n", encoding="utf-8")
    with pytest.raises(SealedYearsError, match="who and why"):
        check_dates(ts, unlock=True, who="Likhith", why="  ")
    with pytest.raises(SealedYearsError, match="who and why"):
        check_dates(ts, unlock=True, why="test")
    assert not config.SEALED_LOG.exists()


# ---- cut off the future -----------------------------------------------------------------------------------------

DF = bars("2020-03-02 01:00", 2000)
CUTS = [300, 777, 1500]


def lagged_mean(d):
    return d["close"].rolling(20).mean().shift(1)       # uses only the 20 bars BEFORE each bar


def centred_mean(d):
    return d["close"].rolling(21, center=True).mean()   # uses 10 bars AFTER each bar


def test_lagged_mean_passes():
    assert_no_future(lagged_mean, DF, CUTS)
    assert_no_future(lagged_mean, DF, [DF.index[400], pd.Timestamp("2020-03-02 20:13")])


def test_centred_mean_fails():
    with pytest.raises(FutureLeakError, match="peeks"):
        assert_no_future(centred_mean, DF, CUTS)


@pytest.mark.parametrize("fn", [
    lambda d: d["close"].shift(-1),                      # the next bar's close
    lambda d: d["close"] - d["close"].mean(),            # the whole file's average
    lambda d: centred_mean(d).dropna(),                  # a value that vanishes when the future is cut
])
def test_other_peeks_fail(fn):
    with pytest.raises(FutureLeakError):
        assert_no_future(fn, DF, CUTS)


def test_frame_output_passes():
    def two_columns(d):
        return pd.DataFrame({"prev_high": d["close"].shift(1).cummax(), "up": d["close"].diff() > 0})
    assert_no_future(two_columns, DF, CUTS)


def test_bad_cut_points_are_refused():
    with pytest.raises(ValueError):
        assert_no_future(lagged_mean, DF, [])
    with pytest.raises(ValueError):
        assert_no_future(lagged_mean, DF, [pd.Timestamp("2030-01-01")])   # nothing is cut off


# ---- hidden check years -----------------------------------------------------------------------------------------

def test_check_opened_refuses_until_the_file_exists(tmp_files):
    with pytest.raises(CheckYearsClosedError, match="2021-2023"):
        check_opened("idea-001")
    with pytest.raises(CheckYearsClosedError, match="2022"):
        check_opened("idea-001", bars("2022-05-02", 100))
    with pytest.raises(CheckYearsClosedError):
        check_opened("idea-001", ("2019-01-01", "2021-06-30"))

    folder = config.IDEAS_DIR / "idea-001"
    folder.mkdir(parents=True)
    (folder / "check-opened.json").write_text(json.dumps({"opened": "2026-09-28", "by": "test"}), encoding="utf-8")
    assert check_opened("idea-001") == {"opened": "2026-09-28", "by": "test"}
    assert check_opened("idea-001", bars("2022-05-02", 100))["by"] == "test"


def test_tune_years_pass_without_the_file():
    assert check_opened("idea-001", bars("2019-06-03", 100)) is None


def test_broken_file_is_refused(tmp_files):
    folder = config.IDEAS_DIR / "idea-002"
    folder.mkdir(parents=True)
    (folder / "check-opened.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckYearsClosedError, match="JSON"):
        check_opened("idea-002")


@pytest.mark.parametrize("name", ["../idea-001", "ideas/idea-001", "", ".hidden", "idea..001"])
def test_bad_idea_names_are_refused(name):
    with pytest.raises(ValueError):
        check_opened(name)
