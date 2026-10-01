"""L1 — the price data and its clock.

Tests marked `data` need the 3 raw CSVs in old/build/data/ and `python -m machine.data.convert` run once (they skip
when the CSVs are missing). See their tables with:  python -m pytest tests/test_data.py -s
The unmarked tests run everywhere (CI): the clock rule, and the bar rules on a small made-up file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import config
from machine.data import bars, clock, convert, hashes, measure_clock
from machine.guards import SealedYearsError

YEARS = range(2016, 2024)
UTC = "UTC"


# ================================================================================================================
# the clock — runs everywhere
# ================================================================================================================

@pytest.mark.parametrize("server, utc, why", [
    ("2021-01-13 12:00", "2021-01-13 10:00", "winter: +2"),
    ("2021-07-14 12:00", "2021-07-14 09:00", "summer: +3"),
    ("2021-03-17 12:00", "2021-03-17 10:00", "US already on summer time, Europe not yet: +2 (EU rule)"),
    ("2021-11-03 12:00", "2021-11-03 10:00", "Europe back on winter time, US not yet: +2 (EU rule)"),
    ("2021-03-26 23:00", "2021-03-26 21:00", "Friday before Europe's switch (28 Mar): +2"),
    ("2021-03-29 01:05", "2021-03-28 22:05", "Monday after Europe's switch: +3"),
    ("2021-10-29 23:00", "2021-10-29 20:00", "Friday before Europe's switch back (31 Oct): +3"),
    ("2021-11-01 01:05", "2021-10-31 23:05", "Monday after Europe's switch back: +2"),
    ("2016-03-16 12:00", "2016-03-16 10:00", "2016 uses the rule 2019-2023 shows (US/EU-differ week)"),
])
def test_server_to_utc(server, utc, why):
    assert clock.server_to_utc(pd.Timestamp(server)) == pd.Timestamp(utc, tz=UTC), why


def test_to_ist_and_back():
    assert clock.to_ist("2021-07-14 12:00") == pd.Timestamp("2021-07-14 14:30", tz="Asia/Kolkata")
    assert clock.to_ist(pd.Timestamp("2021-07-14 09:00", tz=UTC)) == pd.Timestamp("2021-07-14 14:30",
                                                                                   tz="Asia/Kolkata")
    # every 5 minutes of 2 years, Monday-Friday (the clock never changes then): server -> UTC -> server is the same
    idx = pd.date_range("2020-01-01", "2021-12-31 23:55", freq="5min")
    idx = idx[idx.dayofweek < 5]
    assert (clock.utc_to_server(clock.server_to_utc(idx)) == idx).all()
    assert set(clock.server_offset_hours(idx)) == {2, 3}
    s = pd.Series(idx[:3], name="t")
    assert list(clock.server_to_utc(s)) == list(clock.server_to_utc(idx[:3]))


def test_clock_refuses_wrong_input():
    with pytest.raises(ValueError, match="clock change"):
        clock.server_to_utc("2021-03-28 03:30")         # this server hour does not exist (Sunday, gold closed)
    with pytest.raises(ValueError, match="SERVER time"):
        clock.server_to_utc(pd.Timestamp("2021-07-14 12:00", tz=UTC))
    with pytest.raises(ValueError, match="time zone"):
        clock.utc_to_server(pd.Timestamp("2021-07-14 12:00"))


# ================================================================================================================
# the bar rules on a small made-up file — runs everywhere
# ================================================================================================================

def _fake_m1(start: str, minutes: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=minutes, freq="1min", name="time")
    close = 1300 + np.cumsum(rng.integers(-30, 31, minutes)) / 100
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.integers(0, 20, minutes) / 100
    low = np.minimum(open_, close) - rng.integers(0, 20, minutes) / 100
    return pd.DataFrame({"open": open_.round(2), "high": high.round(2), "low": low.round(2), "close": close.round(2),
                         "vol": rng.integers(1, 90, minutes), "spread": rng.integers(0, 60, minutes)}, index=t)


def _install(tmp_path, monkeypatch, m1: pd.DataFrame) -> str:
    """Point config at tmp_path and write m1 there the way convert.py does. Returns its content hash."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "GOLD_M1", tmp_path / "data" / "gold_m1.parquet")
    monkeypatch.setattr(config, "DATA_HASHES", tmp_path / "data" / "DATA_HASHES.txt")
    monkeypatch.setattr(config, "BARS_CACHE_DIR", tmp_path / "data" / "cache")
    config.DATA_DIR.mkdir(exist_ok=True)
    content = hashes.content_hash(m1, convert.MT5_DECIMALS)
    convert.write_parquet(m1, config.GOLD_M1, {"content_sha256": content})
    config.DATA_HASHES.write_text(f"clean  data/gold_m1.parquet  content={content}\n", encoding="utf-8")
    bars._all_bars.cache_clear()
    return content


def test_aggregate_the_mt5_way():
    m1 = _fake_m1("2019-03-11 01:00", 10)
    five = bars.aggregate(m1, "5m")
    first = m1.iloc[:5]
    assert list(five.index) == [pd.Timestamp("2019-03-11 01:00"), pd.Timestamp("2019-03-11 01:05")]
    row = five.iloc[0]
    assert row.open == first.open.iloc[0] and row.close == first.close.iloc[-1]
    assert row.high == first.high.max() and row.low == first.low.min()
    assert row.vol == first.vol.sum() and row.spread == first.spread.min() and row.n_m1 == 5


def test_broker_day_4h_day_and_week():
    m1 = _fake_m1("2019-03-11 22:00", 60 * 60)          # Monday 22:00 server .. Thursday 10:00
    four = bars.aggregate(m1, "4h")
    assert set(four.index.hour) <= {0, 4, 8, 12, 16, 20}
    assert four.index[0] == pd.Timestamp("2019-03-11 20:00") and four.n_m1.iloc[0] == 120
    day = bars.aggregate(m1, "1d")
    assert (day.index == day.index.normalize()).all()   # a day = one SERVER calendar day
    assert day.n_m1.iloc[0] == 120 and day.n_m1.iloc[1] == 1440
    week = bars.aggregate(m1, "1w")
    assert list(week.index) == [pd.Timestamp("2019-03-11")] and week.n_m1.iloc[0] == len(m1)


def test_load_bars_refuses_2024_before_reading_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GOLD_M1", tmp_path / "not-there.parquet")   # nothing to read: refusal comes first
    for start, end in [("2023-12-01", "2024-01-05"), ("2024-02-01", None), (None, "2025")]:
        with pytest.raises(SealedYearsError, match="don't touch those years"):
            bars.load_bars("1h", start, end)
    with pytest.raises(ValueError, match="timeframe"):
        bars.load_bars("2h")


def test_cache_is_keyed_by_content_and_code(tmp_path, monkeypatch):
    content = _install(tmp_path, monkeypatch, _fake_m1("2019-03-11 01:00", 600))
    first = bars.load_bars("15m")
    cached = list(config.BARS_CACHE_DIR.glob("bars_15m_*.parquet"))
    assert len(cached) == 1 and content[:16] in cached[0].name
    assert "utc" in first.columns and first.index.name == "server_time"

    bars._all_bars.cache_clear()
    monkeypatch.setattr(bars, "aggregate", lambda *a: pytest.fail("should read the cache, not rebuild"))
    again = bars.load_bars("15m")
    pd.testing.assert_frame_equal(first, again)

    _install(tmp_path, monkeypatch, _fake_m1("2019-03-11 01:00", 600, seed=2))   # new data -> new content hash
    with pytest.raises(pytest.fail.Exception, match="should read the cache"):
        bars.load_bars("15m")                              # proves it did NOT reuse the old cache
    bars._all_bars.cache_clear()


def test_load_bars_notices_a_replaced_file(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, _fake_m1("2019-03-11 01:00", 60))
    config.DATA_HASHES.write_text("clean  data/gold_m1.parquet  content=0000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        bars.load_bars("5m")
    bars._all_bars.cache_clear()


def test_content_hash_ignores_row_order_and_refuses_off_grid():
    m1 = _fake_m1("2019-03-11 01:00", 30)
    assert hashes.content_hash(m1, convert.MT5_DECIMALS) == hashes.content_hash(m1.iloc[::-1],
                                                                                 convert.MT5_DECIMALS)
    changed = m1.copy()
    changed.iloc[3, 0] += 0.01
    assert hashes.content_hash(changed, convert.MT5_DECIMALS) != hashes.content_hash(m1, convert.MT5_DECIMALS)
    changed.iloc[3, 0] += 0.001
    with pytest.raises(ValueError, match="grid"):
        hashes.content_hash(changed, convert.MT5_DECIMALS)


# ================================================================================================================
# the real data (💻) — needs the 3 CSVs + python -m machine.data.convert
# ================================================================================================================

@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return bars.load_bars("1m")


def _read(path) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("time")
    return df


@pytest.mark.data
def test_hashes_match_the_files():
    recorded = hashes.read_data_hashes()
    for src in convert.SOURCES:
        rel = lambda p: p.relative_to(config.ROOT).as_posix()  # noqa: E731
        assert hashes.file_sha256(src.raw) == recorded[rel(src.raw)]["sha256"], f"{src.raw.name} changed: reconvert"
        content = hashes.content_hash(_read(src.out), src.decimals)
        assert content == recorded[rel(src.out)]["content"] == hashes.parquet_meta(src.out)["content_sha256"]
        assert hashes.file_sha256(src.sealed) == recorded[rel(src.sealed)]["sha256"]   # bytes only, never parsed
    fp = hashes.data_fingerprint()
    assert fp["data/gold_m1.parquet"] == bars.m1_fingerprint()
    assert not any("sealed" in name for name in fp)


@pytest.mark.data
def test_same_data_as_expected():
    """This PC has exactly the prices in git's record (machine/data/EXPECTED_HASHES.txt) — so both PCs agree."""
    assert hashes.differences_from_expected() == []


@pytest.mark.data
def test_rows_per_year_and_nothing_sealed(m1):
    per_year = m1.index.year.value_counts().sort_index()
    print("\n1-min rows per year (server clock): " + "  ".join(f"{y} {n:,}" for y, n in per_year.items()))
    assert list(per_year.index) == list(YEARS)
    assert per_year.between(330_000, 360_000).all()
    for src in convert.SOURCES:
        last = _read(src.out).index.max()
        assert last < (pd.Timestamp(config.SEALED_FROM, tz=UTC) if src.clock == "utc"
                       else pd.Timestamp(config.SEALED_FROM))
    recorded = hashes.read_data_hashes()
    print("sealed files (row count only): " + "  ".join(
        f"{name} {int(row['rows']):,}" for name, row in recorded.items() if row["kind"] == "sealed"))


@pytest.mark.data
def test_no_duplicates():
    for src in convert.SOURCES:
        df = _read(src.out)
        assert df.index.is_monotonic_increasing, src.name
        assert not df.index.duplicated().any(), f"{src.name}: a time appears twice"


@pytest.mark.data
def test_ohlc_sane_and_point_size(m1):
    assert config.POINT == 0.01 and config.POINT_SOURCE
    for name, d in [("m1", m1), ("m5 raw", _read(config.GOLD_M5_MT5))]:
        assert (d.high >= d[["open", "close"]].max(axis=1)).all(), name
        assert (d.low <= d[["open", "close"]].min(axis=1)).all(), name
        assert (d.low > 0).all() and (d.spread >= 0).all() and (d.vol > 0).all(), name
    for col in ("open", "high", "low", "close"):                  # the point check: every price is whole cents
        cents = m1[col].to_numpy() / config.POINT
        assert np.abs(cents - np.rint(cents)).max() < 1e-6, col
    assert (m1.index.second == 0).all()

    size = (m1.high - m1.low).sort_values(ascending=False)
    minute = m1.index.to_series().diff() == pd.Timedelta(minutes=1)
    jump = (m1.open - m1.close.shift()).abs()[minute].sort_values(ascending=False)
    print("\nbiggest 1-min candles (high-low), server clock: " + "  ".join(
        f"{t:%Y-%m-%d %H:%M} ${v:.2f}" for t, v in size.head(5).items()))
    print("biggest jumps from one minute's close to the next minute's open: " + "  ".join(
        f"{t:%Y-%m-%d %H:%M} ${v:.2f}" for t, v in jump.head(5).items()))
    assert size.iloc[0] < 50 and jump.iloc[0] < 30                # a $50 1-min candle in 2016-2023 = broken data
    print(f"spread: 0 on {(m1.spread == 0).mean():.2%} of 1-min bars (config.MIN_SPREAD_POINTS = "
          f"{config.MIN_SPREAD_POINTS:g} is the floor for those); median {m1.spread.median():g} points")


@pytest.mark.data
def test_no_saturday_or_sunday_bars(m1):
    for name, d in [("m1", m1), ("m5 raw", _read(config.GOLD_M5_MT5)), ("1d", bars.load_bars("1d"))]:
        days = pd.DatetimeIndex(d.index).dayofweek
        assert not (days == 5).any(), f"{name}: Saturday bars on the server clock"
        assert not (days == 6).any(), f"{name}: Sunday bars on the server clock"


@pytest.mark.data
def test_ten_biggest_weekday_gaps(m1):
    t = m1.index.to_series()
    week = t.dt.normalize() - pd.to_timedelta(t.dt.dayofweek, unit="D")
    gap = t.diff()
    inside_week = week == week.shift()                            # both bars in the same Mon-Fri: no weekend
    top = gap[inside_week].sort_values(ascending=False).head(10)
    print("\n10 biggest gaps between 1-min bars inside a week (server clock; the weekend is not a gap):")
    for end, g in top.items():
        start = t.shift()[end]
        print(f"  {start:%a %Y-%m-%d %H:%M} -> {end:%a %Y-%m-%d %H:%M}   {g}")
    assert top.iloc[0] < pd.Timedelta(days=3)


@pytest.mark.data
def test_5min_from_1min_matches_raw_mt5_5min():
    built = bars.load_bars("5m")
    raw = _read(config.GOLD_M5_MT5)
    assert built.index.equals(raw.index), "different 5-min bar times"
    close_enough = np.ones(len(raw), dtype=bool)
    for col in ("open", "high", "low", "close"):
        close_enough &= (built[col] - raw[col]).abs().to_numpy() <= 0.02 + 1e-9
    same_spread = (built.spread == raw.spread).mean()
    same_vol = (built.vol == raw.vol).mean()
    print(f"\n5-min bars built from 1-min vs MT5's own 5-min file: {len(raw):,} bars; all four prices within $0.02: "
          f"{close_enough.mean():.4%}; spread = smallest 1-min spread: {same_spread:.4%}; volume = sum: "
          f"{same_vol:.4%}")
    assert close_enough.mean() >= 0.999


@pytest.mark.data
def test_daily_candles_per_year():
    day = bars.load_bars("1d")
    per_year = day.index.year.value_counts().sort_index()
    print("\ndaily candles per year (broker day): " + "  ".join(f"{y} {n}" for y, n in per_year.items()))
    assert list(per_year.index) == list(YEARS)
    assert per_year.between(255, 262).all()
    four = bars.load_bars("4h")
    assert set(four.index.hour) <= {0, 4, 8, 12, 16, 20}
    assert (bars.load_bars("1w").index.dayofweek == 0).all()


@pytest.mark.data
def test_2024_refused_on_real_data():
    with pytest.raises(SealedYearsError):
        bars.load_bars("1m", "2023-12-28", "2024-01-03")
    last = bars.load_bars("1d", "2023-12")
    assert last.index.max() < pd.Timestamp(config.SEALED_FROM)


@pytest.mark.data
def test_clock_matches_the_measurement():
    table = pd.read_csv(config.SERVER_OFFSET_CSV, parse_dates=["week_start_server"], index_col=0)
    print("\n" + measure_clock.summary(table))
    priced = table[table.measured_offset_h.notna()]
    assert priced.clear.all(), "a week where the two feeds did not clearly agree on one hour"
    assert len(priced) >= 260
    wednesday_noon = priced.index + pd.Timedelta(days=2, hours=12)
    assert (priced.measured_offset_h.astype(int).to_numpy() == clock.server_offset_hours(wednesday_noon)).all()
    differ = table.us_eu_differ.astype(bool)
    empty_23h = table.bars_at_23h == 0
    assert (empty_23h == differ).all(), "the daily stop does not sit where the EU rule puts it"
    assert differ[table.index.year <= 2018].sum() == 9      # 2016-2018: 9 US/EU-differ weeks checked this way
