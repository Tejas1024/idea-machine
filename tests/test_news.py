"""The news calendar (L12): the clock trap, what is known when, and the real table's checks.

Tests with no mark run anywhere on a small made-up table. Tests marked `news` read data/news/events.parquet (made on
Tejas's PC from MT5: python -m machine.news.mt5_export, then python -m machine.news.build) and skip without it.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from machine import config, guards, settings
from machine.data import clock
from machine.news import build, calendar, official

NY = "America/New_York"


# ---- a made-up table ----------------------------------------------------------------------------------------------

def make_events(rows: list[tuple]) -> pd.DataFrame:
    """rows: (server time, event, importance, actual, forecast, scheduled)."""
    server = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    utc = clock.server_to_utc(server)
    ev = pd.DataFrame({
        "utc": utc, "server": server, "ist": utc.tz_convert(clock.IST_ZONE), "day": server.normalize(),
        "event": [r[1] for r in rows], "kind": [calendar.event_kind(r[1]) for r in rows],
        "importance": [r[2] for r in rows], "mt5_importance": [3 if r[2] == "high" else 2 for r in rows],
        "actual": [r[3] for r in rows], "forecast": [r[4] for r in rows], "previous": np.nan,
        "revised_previous": np.nan, "has_actual": [r[3] == r[3] for r in rows], "exact_time": True,
        "scheduled": [r[5] for r in rows],
    })
    return ev.sort_values("server").reset_index(drop=True)


EVENTS = make_events([
    ("2019-06-05 15:15", "ADP Nonfarm Employment Change", "high", 27.0, 180.0, True),
    ("2019-06-07 15:30", "Nonfarm Payrolls", "high", 75.0, 185.0, True),
    ("2019-06-07 15:30", "Unemployment Rate", "high", 3.6, 3.6, True),
    ("2019-06-10 17:00", "Fed Interest Rate Decision", "high", 2.25, np.nan, False),   # a made-up emergency cut
    ("2019-06-12 15:30", "CPI m/m", "high", 0.1, 0.1, True),
    ("2019-07-05 15:30", "Nonfarm Payrolls", "high", 224.0, 160.0, True),
])


# ---- the clock trap -----------------------------------------------------------------------------------------------

def test_mt5_times_carry_the_export_days_offset():
    """MT5 gave winter NFP at 16:30 and summer NFP at 15:30 (exported in summer, +3 h): both are 8:30 New York."""
    mt5 = pd.Series(pd.to_datetime(["2016-01-08 16:30", "2016-07-08 15:30", "2020-03-15 00:00"]))
    utc = build.mt5_to_utc(mt5, pd.Timedelta(hours=3))
    ny = utc.dt.tz_convert(NY)
    assert list(ny.dt.strftime("%H:%M")) == ["08:30", "08:30", "17:00"]      # 15 Mar 2020: the Fed's Sunday cut
    assert list(utc.dt.tz_convert(clock.IST_ZONE).dt.strftime("%H:%M")[:2]) == ["19:00", "18:00"]
    # the same rows read with each date's own offset would put the winter NFP an hour late - the trap
    own = clock.server_to_utc(mt5[:2]).dt.tz_convert(NY).dt.strftime("%H:%M")
    assert list(own) == ["09:30", "08:30"]


def test_to_events_keeps_red_and_orange_and_explains_missing_forecasts():
    raw = pd.DataFrame({
        "value_id": [1, 2, 3, 4, 5, 6], "event_id": [10, 10, 20, 30, 40, 50],
        "event_name": ["Nonfarm Payrolls", "Nonfarm Payrolls", "FOMC Member Williams Speech", "Jackson Hole "
                       "Economic Symposium", "Redbook y/y", "Nonfarm Payrolls"],
        "importance": [3, 3, 2, 2, 1, 3], "time_mode": [0, 0, 0, 1, 0, 0],
        "mt5_time": pd.to_datetime(["2016-01-08 16:30", "2018-01-05 16:30", "2018-01-09 20:00", "2018-08-23 00:00",
                                    "2018-01-09 15:55", "2018-02-02 16:30"]),
        "actual_x1e6": [292e6, 148e6, np.nan, np.nan, 5e6, 200e6],
        "forecast_x1e6": [np.nan, 190e6, np.nan, np.nan, np.nan, 180e6],
        "prev_x1e6": np.nan, "revised_prev_x1e6": np.nan, "unit": [4, 4, 0, 0, 1, 4],
        "multiplier": [1, 1, 0, 0, 0, 1], "period": "2015.12.01 00:00:00", "revision": 0,
    })
    ev = build.to_events(raw, pd.Timedelta(hours=3), {"server": "MetaQuotes-Demo"})
    assert "Redbook y/y" not in set(ev["event"])                          # low = yellow: not his news
    nfp = ev[ev["event"] == "Nonfarm Payrolls"]
    assert list(nfp["actual"]) == [292.0, 148.0, 200.0] and set(nfp["unit"]) == {"jobs"}
    assert set(nfp["multiplier"]) == {"thousands"}
    assert list(nfp["no_forecast_why"]) == ["MT5 has none in 2016 to mid-2017", "", ""]
    speech = ev[ev["kind"] == "fed_speaker"].iloc[0]
    assert speech["no_forecast_why"] == "none for this kind" and not speech["has_forecast"]
    hole = ev[ev["event"].str.startswith("Jackson")].iloc[0]
    assert not hole["exact_time"] and pd.isna(hole["utc"]) and hole["day"] == pd.Timestamp("2018-08-23")
    assert (nfp["utc"].dt.tz_convert(NY).dt.strftime("%H:%M") == "08:30").all()
    assert (nfp["server"] == clock.utc_to_server(nfp["utc"])).all()


def test_the_build_never_reads_the_sealed_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NEWS_SEALED_DIR", tmp_path / "sealed")
    (tmp_path / "sealed").mkdir()
    with pytest.raises(guards.SealedYearsError):
        build.read_raw(tmp_path / "sealed" / "mt5_calendar_US_2024plus.tsv")


def test_load_events_refuses_the_sealed_years():
    with pytest.raises(guards.SealedYearsError):
        calendar.load_events("2023-06-01", "2024-02-01")


def test_kinds():
    want = {"Nonfarm Payrolls": "jobs", "Nonfarm Productivity q/q": "growth", "Initial Jobless Claims": "jobs",
            "CPI m/m": "inflation", "Core PCE Price Index m/m": "inflation", "Real PCE q/q": "growth",
            "FOMC Statement": "fed", "Fed Interest Rate Decision": "fed", "Fed Chair Powell Speech": "fed",
            "FOMC Member Williams Speech": "fed_speaker", "Fed Industrial Production m/m": "growth",
            "ISM Manufacturing Employment": "jobs", "ISM Manufacturing PMI": "growth", "10-Year Note Auction":
            "auction", "EIA Crude Oil Stocks Change": "energy", "Existing Home Sales": "housing",
            "CFTC Gold Non-Commercial Net Positions": "positions", "Federal Budget Balance": "other"}
    assert {name: calendar.event_kind(name) for name in want} == want


def test_his_setting_picks_the_colours():
    assert calendar.importance_for(settings.get("news_events")) == ("high", "moderate")   # T7: red AND orange
    with settings.using({"news_events": "forex_factory_red"}):
        assert calendar.importance_for(settings.get("news_events")) == ("high",)


# ---- minutes to / since, news days, what was released -----------------------------------------------------------

def test_minutes_to_next_and_since_last():
    t = pd.DatetimeIndex(["2019-06-07 15:00", "2019-06-07 15:30", "2019-06-07 16:00", "2019-06-10 17:30"])
    to = calendar.minutes_to_next(t, EVENTS)
    assert list(to[:3]) == [30.0, 0.0, (pd.Timestamp("2019-06-12 15:30") - pd.Timestamp("2019-06-07 16:00"))
                            / pd.Timedelta(minutes=1)]
    since = calendar.minutes_since_last(t, EVENTS)
    assert since.iloc[1] == 0.0 and since.iloc[2] == 30.0
    assert since.iloc[3] == 30.0                                   # the emergency cut counts once it happened


def test_an_unscheduled_release_is_never_seen_coming():
    before = pd.DatetimeIndex(["2019-06-10 16:00"])
    to = calendar.minutes_to_next(before, EVENTS)
    assert to.iloc[0] == (pd.Timestamp("2019-06-12 15:30") - before[0]) / pd.Timedelta(minutes=1)   # not 60
    assert not calendar.is_news_day(pd.DatetimeIndex(["2019-06-10"]), EVENTS).iloc[0]
    assert calendar.is_news_day(pd.DatetimeIndex(["2019-06-07 03:00", "2019-06-12"]), EVENTS).all()


def test_times_with_a_zone_are_converted():
    utc = pd.DatetimeIndex(["2019-06-07 12:00"]).tz_localize("UTC")      # = 15:00 server (summer, +3)
    assert calendar.minutes_to_next(utc, EVENTS).iloc[0] == 30.0


def test_next_event_names_the_most_important_at_that_minute():
    nxt = calendar.next_event(pd.DatetimeIndex(["2019-06-06 12:00"]), EVENTS)
    assert nxt["event"].iloc[0] in {"Nonfarm Payrolls", "Unemployment Rate"} and nxt["kind"].iloc[0] == "jobs"
    assert nxt["forecast"].iloc[0] in (185.0, 3.6)


def test_last_released_actual():
    t = pd.DatetimeIndex(["2019-06-07 15:29", "2019-06-07 15:30", "2019-07-05 15:29", "2019-07-05 15:31"])
    got = calendar.last_released(t, "Nonfarm Payrolls", events=EVENTS)
    assert np.isnan(got.iloc[0]) and list(got.iloc[1:]) == [75.0, 75.0, 224.0]


def test_as_known_at_hides_actuals_and_surprises():
    at = pd.Timestamp("2019-06-07 15:29")                                   # server time, one minute before NFP
    known = calendar.as_known_at(EVENTS, at)
    nfp = known[known["event"] == "Nonfarm Payrolls"]
    assert nfp["actual"].isna().all() and list(nfp["forecast"]) == [185.0, 160.0]   # the forecast IS known
    assert "Fed Interest Rate Decision" not in set(known["event"])            # the surprise: not yet
    later = calendar.as_known_at(EVENTS, "2019-06-10 17:00")
    assert later.loc[later["event"] == "Fed Interest Rate Decision", "actual"].iloc[0] == 2.25


# ---- the "cut off the future" test (R4) ---------------------------------------------------------------------------

def _features(bars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    known = calendar.as_known_at(events, bars.index[-1])
    return pd.DataFrame({
        "to": calendar.minutes_to_next(bars.index, known).to_numpy(),
        "since": calendar.minutes_since_last(bars.index, known).to_numpy(),
        "nfp": calendar.last_released(bars.index, "Nonfarm Payrolls", events=known).to_numpy(),
        "news_day": calendar.is_news_day(bars.index, known).to_numpy(),
    }, index=bars.index)


def _minute_bars(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1min")
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


def test_the_calendar_passes_the_cut_off_the_future_test():
    bars = _minute_bars("2019-06-04", "2019-07-06")
    cuts = [bars.index.get_loc(pd.Timestamp(t)) for t in
            ("2019-06-07 15:29", "2019-06-07 15:30", "2019-06-10 16:59", "2019-06-10 17:00", "2019-07-05 15:29")]
    guards.assert_no_future(lambda b: _features(b, EVENTS), bars, cuts)


def test_the_cut_off_test_catches_a_peek():
    """The test gives each function the table as known at its last bar. Two ways to peek, both caught: seeing the
    emergency cut coming, and reading the next NFP's actual number before it is out."""
    bars = _minute_bars("2019-06-04", "2019-07-06")
    cut = [bars.index.get_loc(pd.Timestamp("2019-06-10 16:00"))]

    def sees_surprises(b):
        known = calendar.as_known_at(EVENTS, b.index[-1])
        return calendar.minutes_to_next(b.index, known.assign(scheduled=True)).to_frame()

    def reads_the_next_actual(b):
        known = calendar.as_known_at(EVENTS, b.index[-1])
        nfp = known[known["event"] == "Nonfarm Payrolls"]
        nxt = np.searchsorted(nfp["server"].to_numpy(), b.index.to_numpy(), side="left")
        return pd.DataFrame({"next_nfp": np.append(nfp["actual"].to_numpy(), np.nan)[nxt]}, index=b.index)

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future(sees_surprises, bars, cut)
    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future(reads_the_next_actual, bars, cut)


# ---- the real table (data/news/events.parquet) ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    return pd.read_parquet(config.NEWS_EVENTS)


@pytest.mark.news
def test_real_nfp_count_and_times(real):
    nfp = real[(real["event"] == "Nonfarm Payrolls") & (real["day"] >= "2016-01-01")]
    assert len(nfp) == 96 and nfp.groupby(nfp["day"].dt.year).size().eq(12).all()
    assert (nfp["utc"].dt.tz_convert(NY).dt.strftime("%H:%M") == "08:30").all()
    ist = nfp["ist"]
    assert set(ist[ist.dt.month == 1].dt.strftime("%H:%M")) == {"19:00"}      # the prompt: about 19:00 in January
    assert set(ist[ist.dt.month == 7].dt.strftime("%H:%M")) == {"18:00"}      # and 18:00 in July
    assert nfp["has_actual"].all()


@pytest.mark.news
def test_real_table_is_2015_to_2023_on_the_price_clock(real):
    guards.check_dates(real[["utc", "server", "ist", "day"]])
    exact = real[real["exact_time"]]
    assert (exact["server"] == clock.utc_to_server(exact["utc"])).all()
    assert set(real["importance"]) == {"high", "moderate"}
    assert real["source"].str.startswith(("MT5 calendar", "Federal Reserve statement page")).all()


@pytest.mark.news
def test_real_surprise_fed_releases_are_not_scheduled(real):
    """The Fed labels them "(unscheduled)" / "(notation vote)": 3 and 15 Mar 2020 (decision, statement, press
    conference) and the 23 Mar 2020 8:00 statement - that one MT5 lacks, so its row comes from the Fed's page."""
    surprise = real[~real["scheduled"]]
    assert set(surprise["utc"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d %H:%M")) == {
        "2020-03-03 10:00", "2020-03-03 11:00", "2020-03-15 17:00", "2020-03-15 18:30", "2020-03-23 08:00"}
    added = real[real["source"].str.startswith("Federal Reserve")]
    assert list(added["event"]) == ["FOMC Statement"] and not added["scheduled"].any()


@pytest.mark.news
def test_real_calendar_passes_the_cut_off_the_future_test(real):
    ev = real[real["day"].between("2020-02-01", "2020-04-30")]
    bars = _minute_bars("2020-02-24", "2020-03-20")
    bars = bars[bars.index.dayofweek < 5]
    cuts = [bars.index.get_loc(pd.Timestamp(t)) for t in
            ("2020-03-03 16:59", "2020-03-06 15:29", "2020-03-06 15:30", "2020-03-13 23:00")]
    guards.assert_no_future(lambda b: _features(b, ev), bars, cuts)


@pytest.mark.news
def test_real_official_check_agrees():
    if not config.NEWS_OFFICIAL_CHECK.is_file():
        pytest.skip("no official check yet (python -m machine.news.build)")
    check = pd.read_csv(config.NEWS_OFFICIAL_CHECK)
    for event in ("Nonfarm Payrolls", "Unemployment Rate"):
        rows = check[(check["event"] == event) & check["in_official"]]
        assert len(rows) == 96 and rows["in_mt5"].all()
        assert rows["time_same"].astype(str).eq("True").all() and rows["number_same"].astype(str).eq("True").all()
    cpi = check[check["event"] == "CPI m/m"]
    compared = cpi[cpi["number_same"].notna()]
    assert len(cpi) == 96 and compared["number_same"].astype(str).eq("True").all() and len(compared) >= 95


# ---- the official pages' parsers, on made-up pages (no network) ---------------------------------------------------

def _page(body: str) -> str:
    return "<html><body>" + body + " " + "x" * 2000 + "</body></html>"


@pytest.mark.parametrize("sentence, number, rounding", [
    ("Total nonfarm payroll employment rose by 292,000 in December", 292.0, 0.5),
    ("Total nonfarm payroll employment rose by 2.5 million in May", 2500.0, 50.0),
    ("Total nonfarm payroll employment fell by 20.5 million in April", -20500.0, 50.0),
    ("Total nonfarm payroll employment changed little in May (+38,000)", 38.0, 0.5),
    ("Total nonfarm payroll employment declined by 140,000 in December", -140.0, 0.5),
])
def test_bls_jobs_headline(monkeypatch, sentence, number, rounding):
    text = ("Transmission of material in this release is embargoed until USDL-16-0001 8:30 a.m. (EST) Friday, "
            f"January 8, 2016 THE EMPLOYMENT SITUATION {sentence}, and the unemployment rate declined by 0.3 "
            "percentage point to 4.7 percent.")
    monkeypatch.setattr(official, "fetch", lambda url, name: _page(text))
    got = official.bls_release("empsit", dt.date(2016, 1, 8))
    assert got["nfp_thousands"] == number and got["nfp_thousands_rounding"] == rounding
    assert got["unemployment_rate"] == 4.7                  # the rate, not the 0.3 change (a real bug, fixed)
    assert got["official_ny"] == pd.Timestamp("2016-01-08 08:30", tz=NY)


@pytest.mark.parametrize("sentence, number", [
    ("The Consumer Price Index for All Urban Consumers (CPI-U) declined 0.1 percent in December", -0.1),
    ("The Consumer Price Index for All Urban Consumers (CPI-U) was unchanged in March", 0.0),
    ("The Consumer Price Index for All Urban Consumers (CPI-U) rose 0.4 percent in May", 0.4),
])
def test_bls_cpi_headline(monkeypatch, sentence, number):
    text = f"embargoed until 8:30 a.m. (EDT) Thursday, June 16, 2016 {sentence} on a seasonally adjusted basis."
    monkeypatch.setattr(official, "fetch", lambda url, name: _page(text))
    got = official.bls_release("cpi", dt.date(2016, 6, 16))
    assert got["cpi_mm"] == number and got["official_ny"] == pd.Timestamp("2016-06-16 08:30", tz=NY)


def test_an_empty_bls_page_is_said_not_guessed(monkeypatch):
    def empty(url, name):
        raise ConnectionError("0 bytes")
    monkeypatch.setattr(official, "fetch", empty)
    got = official.bls_release("cpi", dt.date(2016, 6, 16))
    assert pd.isna(got["official_ny"]) and "empty" in got["note"] and "cpi_mm" not in got


def test_the_feds_own_label_says_what_was_a_surprise():
    assert official._not_known_in_advance("Statement Minutes March 15 (unscheduled) Meeting - 2020 Press")
    assert official._not_known_in_advance("March 17-18 (cancelled) Meeting - 2020 March 23 (notation vote) - 2020")
    assert not official._not_known_in_advance("March 23 (notation vote) - 2020 Statement April 28-29 Meeting - 2020")
    assert not official._not_known_in_advance("January 26-27 Meeting - 2016 Beige Book Statement")


def test_bea_rows_carry_their_time():
    html = ('<td class="views-field views-field-title"><a href="/news/2018/gdp-3rd-quarter-2018-advance-estimate" '
            'hreflang="en">Gross Domestic Product, 3rd quarter 2018 (advance estimate)</a>        </td>\n'
            '<td headers="view-created-table-column" class="views-field views-field-created">'
            '<time datetime="2018-10-26T08:30:00-04:00">October 26, 2018</time>')
    [(link, title, when)] = official._BEA_ROW.findall(html)
    assert link.endswith("advance-estimate") and "advance estimate" in title
    assert pd.Timestamp(when).tz_convert(NY) == pd.Timestamp("2018-10-26 08:30", tz=NY)
