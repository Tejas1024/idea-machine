"""News surprise (L12b.1): signs.csv, the surprise the right way round, the 2-year yield for the Fed, the size against
PAST surprises only (the cut-off-the-future test), and the groups.

Tests with no mark run anywhere on made-up tables. Tests marked `news` read the real tables (data/news/, made on
Tejas's PC) and skip without them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import config, guards
from machine.data import clock
from machine.news import calendar, forexfactory, surprise, yields

NY = "America/New_York"
K = 1000.0


def ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def make_events(rows: list[tuple]) -> pd.DataFrame:
    """rows: (New York time, MT5 event, scheduled)."""
    utc = pd.DatetimeIndex([ny(r[0]) for r in rows])
    server = clock.utc_to_server(utc)
    ev = pd.DataFrame({
        "utc": utc, "server": server, "ist": utc.tz_convert(clock.IST_ZONE), "day": server.normalize(),
        "event": [r[1] for r in rows], "kind": [calendar.event_kind(r[1]) for r in rows], "importance": "high",
        "value_id": range(len(rows)), "actual": np.nan, "forecast": np.nan, "revised_previous": np.nan,
        "multiplier": "", "has_actual": False, "exact_time": True, "scheduled": [r[2] for r in rows],
    })
    return ev.sort_values("utc", kind="stable").reset_index(drop=True)


def make_ff(rows: list[tuple]) -> pd.DataFrame:
    """rows: (New York time, Forex Factory name, actual, forecast)."""
    ff = pd.DataFrame({"utc": [ny(r[0]) for r in rows], "name": [r[1] for r in rows],
                       "actual_value": [r[2] for r in rows], "forecast_value": [r[3] for r in rows],
                       "colour": "red"})
    ff["actual"] = ff["actual_value"].astype(str)
    ff["better_worse"] = "same or none"
    return ff


# the jobs report on the first Friday: forecast 100K every time; the misses (thousands of jobs) below. Release 11 is a
# freak month (1,000K off). Plus: one unemployment rate, a day with ONLY the unemployment rate, a JOLTS with no Forex
# Factory row, a speech, two Fed decisions (one on a Sunday, not known in advance).
MISSES = [10, -10, 20, -20, 10, -10, 20, -20, 10, -10, 1000, 20, -20, 30]
NFP_DAYS = pd.date_range("2019-01-04", periods=len(MISSES), freq="WOM-1FRI")
ROWS, FF_ROWS = [], []
for day, miss in zip(NFP_DAYS, MISSES):
    ROWS.append((f"{day:%Y-%m-%d} 08:30", "Nonfarm Payrolls", True))
    FF_ROWS.append((f"{day:%Y-%m-%d} 08:30", "Non-Farm Employment Change", (100 + miss) * K, 100 * K))
ROWS += [(f"{NFP_DAYS[0]:%Y-%m-%d} 08:30", "Unemployment Rate", True), ("2019-03-12 08:30", "Unemployment Rate", True),
         ("2019-03-13 10:00", "JOLTS Job Openings", True), ("2019-03-14 12:00", "Fed Chair Powell Speech", True),
         ("2019-06-19 14:00", "Fed Interest Rate Decision", True), ("2019-06-19 14:00", "FOMC Statement", True),
         ("2019-06-19 14:30", "FOMC Press Conference", True), ("2019-06-23 17:00", "Fed Interest Rate Decision", False)]
FF_ROWS += [(f"{NFP_DAYS[0]:%Y-%m-%d} 08:30", "Unemployment Rate", 3.5, 3.7),
            ("2019-03-12 08:30", "Unemployment Rate", 3.9, 3.7)]
EVENTS = make_events(ROWS)
FF = make_ff(FF_ROWS)
QUOTES = {"2019-06-18": 1.85, "2019-06-19": 1.75, "2019-06-21": 1.70, "2019-06-24": 1.60}
DAYS = pd.bdate_range("2018-12-03", "2020-03-31")
DGS2 = yields.with_moments(pd.DataFrame({"date": DAYS, "dgs2": [QUOTES.get(f"{d:%Y-%m-%d}", 2.0) for d in DAYS]}))


@pytest.fixture(scope="module")
def signs():
    return surprise.load_signs()


@pytest.fixture(scope="module")
def table(signs):
    return surprise.score(EVENTS, FF, DGS2, signs)


# ---- signs.csv ------------------------------------------------------------------------------------------------------

def test_signs_file_is_valid_and_turns_the_right_ones_round(signs):
    sign = signs.set_index("event")["sign"]
    for ev in ("Initial Jobless Claims", "Continuing Jobless Claims", "Unemployment Rate", "U6 Unemployment Rate"):
        assert sign[ev] == -1, ev                                        # the prompt: these are inverted
    assert (sign[["Nonfarm Payrolls", "Core CPI m/m", "CPI m/m", "GDP q/q", "Retail Sales m/m"]] == 1).all()
    first = signs[signs["rank"] == 1].set_index("report")["event"]
    assert first["Jobs report"] == "Nonfarm Payrolls"                    # NFP = the headline payrolls
    assert first["CPI"] == "Core CPI m/m"                                # CPI: core, with its why
    assert "food and fuel" in signs.set_index("event").at["Core CPI m/m", "why"]
    fed = signs.set_index("event").loc["Fed Interest Rate Decision"]
    assert fed["method"] == "2y yield" and "AFTER the release" in fed["why"]
    assert (signs.loc[signs["method"] == "flagged", "sign"] == 0).all()


@pytest.mark.parametrize("change, message", [
    (lambda s: s.assign(rank=s["rank"].where(s["event"] != "Unemployment Rate", 1)), "share a rank"),
    (lambda s: s.assign(sign=s["sign"].where(s["event"] != "FOMC Minutes", 1)), "flagged <-> sign 0"),
    (lambda s: s.assign(why=s["why"].where(s["event"] != "JOLTS Job Openings", " ")), "without its why"),
])
def test_signs_file_refuses_a_bad_line(tmp_path, signs, change, message):
    path = tmp_path / "signs.csv"
    change(signs.copy()).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        surprise.load_signs(path)


def test_every_event_needs_a_line(signs):
    with pytest.raises(ValueError, match="no line for"):
        surprise.score(make_events([("2019-01-04 08:30", "A Made Up Event", True)]), FF, DGS2, signs)


# ---- the surprise, the right way round -----------------------------------------------------------------------------

def test_surprise_is_actual_minus_forecast_turned_round_for_jobless_numbers(table):
    nfp = table[table["event"] == "Nonfarm Payrolls"].reset_index(drop=True)
    assert list(nfp["dollar_surprise"] / K) == MISSES                   # more jobs than expected = good for the dollar
    ur = table[table["event"] == "Unemployment Rate"].reset_index(drop=True)
    assert ur["dollar_surprise"].round(6).tolist() == [0.2, -0.2]        # 3.5 vs 3.7 = good; 3.9 vs 3.7 = bad


def test_the_report_number_is_rank_one_or_the_next_there(table):
    first_friday = table[table["ny_date"] == NFP_DAYS[0]].set_index("event")["headline"]
    assert first_friday["Nonfarm Payrolls"] and not first_friday["Unemployment Rate"]
    alone = table[(table["event"] == "Unemployment Rate") & (table["ny_date"] == "2019-03-12")]
    assert alone["headline"].all()                                        # no NFP that day: the rate stands for it


def test_size_uses_only_past_surprises_and_caps_a_freak_month(table):
    nfp = table[table["event"] == "Nonfarm Payrolls"].reset_index(drop=True)
    assert nfp["z"].iloc[:8].isna().all() and (nfp["past_n"].iloc[:9].tolist() == list(range(9)))
    assert nfp.at[8, "usual_miss"] == pytest.approx(15 * K)               # past 10,10,20,20,10,10,20,20 -> 15
    assert nfp.at[8, "z"] == pytest.approx(10 / 15) and nfp.at[8, "group"] == "small"
    # after the 1,000K freak month: past = 10 x6, 20 x4, 1000 -> middle 10 -> capped at 40 -> (60+80+40)/11
    assert nfp.at[11, "usual_miss"] == pytest.approx((60 + 80 + 40) / 11 * K)
    assert nfp.at[11, "z"] == pytest.approx(20 / ((60 + 80 + 40) / 11)) and nfp.at[11, "group"] == \
        "big good for the dollar"
    assert nfp.at[10, "group"] == "big good for the dollar" and nfp.at[12, "group"] == "big bad for the dollar"


def test_the_feds_decision_uses_the_2_year_yield_after_minus_before(table):
    june19 = table[table["ny_date"] == "2019-06-19"].set_index("event")
    assert june19["surprise"].round(6).eq(-0.10).all()                     # 1.85 (18 Jun) -> 1.75 (19 Jun, 3:30 pm)
    assert june19["measured"].str.startswith("daily, measured after the release").all()
    assert june19.at["Fed Interest Rate Decision", "headline"] and not june19.at["FOMC Press Conference", "headline"]
    assert june19.at["Fed Interest Rate Decision", "known_from"] == ny("2019-06-20 16:15")   # the next quote day
    sunday = table[table["ny_date"] == "2019-06-23"].iloc[0]              # Sunday 17:00: Fri 21 Jun -> Mon 24 Jun
    assert (sunday["yield_before"], sunday["yield_after"]) == (1.70, 1.60)
    assert sunday["known_from"] == ny("2019-06-25 16:15")


def test_not_scored_says_why(table):
    why = table.set_index("event")["why_not_scored"]
    assert why["JOLTS Job Openings"] == "no Forex Factory row for this release"
    assert why["Fed Chair Powell Speech"].startswith("flagged")
    assert table.loc[table["event"] == "Fed Chair Powell Speech", "group"].eq("not scored").all()
    assert table[table["event"] == "Nonfarm Payrolls"]["why_not_scored"].iloc[0] == \
        f"only 0 known past surprises (needs {surprise.MIN_PAST})"


def test_forex_factory_numbers():
    assert forexfactory.number("-22000K") == (-22_000_000.0, "K")
    assert forexfactory.number("0.3%") == (0.3, "%")
    assert forexfactory.number("<0.25%") == (0.25, "%")
    assert forexfactory.number("-48.1B") == (-48_100_000_000.0, "B")
    assert np.isnan(forexfactory.number("")[0])
    assert forexfactory.half_step("-19.4") == pytest.approx(0.05)          # -19.37 rounds to -19.4: the same number
    assert forexfactory.half_step("0.2M") == pytest.approx(50_000)
    assert forexfactory.half_step("225K") == pytest.approx(500)


def test_no_forex_factory_week_reaches_2024():
    weeks = forexfactory.sundays()
    assert weeks[-1] + pd.Timedelta(days=6) < pd.Timestamp("2024-01-01").date()
    with pytest.raises(guards.SealedYearsError):
        forexfactory.sundays(last=pd.Timestamp("2023-12-31").date())


# ---- the "cut off the future" test (R4) -----------------------------------------------------------------------------

def _known(bars: pd.DataFrame, events=EVENTS, ff=FF, dgs2=DGS2, signs=None):
    at = bars.index[-1]
    return surprise.score(calendar.as_known_at(events, at), forexfactory.as_known_at(ff, at),
                          yields.known_at(dgs2, at), signs if signs is not None else surprise.load_signs())


def _features(bars: pd.DataFrame, **kw) -> pd.DataFrame:
    t = _known(bars, **kw)
    return pd.DataFrame({"jobs z": surprise.latest(bars.index, t, "Jobs report").to_numpy(),
                         "jobs group": surprise.latest(bars.index, t, "Jobs report", "dollar_surprise").to_numpy(),
                         "fed": surprise.latest(bars.index, t, "FOMC decision", "surprise").to_numpy()},
                        index=bars.index)


def _hour_bars(start: str, end: str) -> pd.DataFrame:
    """Hourly bars Monday-Friday (server time). No Sundays: the clock change hour does not exist then."""
    idx = pd.date_range(start, end, freq="1h")
    idx = idx[idx.dayofweek < 5]
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


def _cuts(bars, times):
    return [bars.index.get_loc(clock.utc_to_server(pd.DatetimeIndex([ny(t)]))[0].floor("h")) for t in times]


def test_the_surprise_table_passes_the_cut_off_the_future_test():
    bars = _hour_bars("2019-01-01", "2020-03-10")
    cuts = _cuts(bars, [f"{NFP_DAYS[8]:%Y-%m-%d} 07:00", f"{NFP_DAYS[10]:%Y-%m-%d} 10:00", "2019-06-19 15:00",
                        "2019-06-20 17:00", "2019-06-24 12:00", f"{NFP_DAYS[12]:%Y-%m-%d} 09:00"])
    guards.assert_no_future(_features, bars, cuts)


def test_the_cut_off_test_catches_a_peek():
    """A usual miss taken over ALL releases (the future ones too) changes the past when the future is cut off."""
    bars = _hour_bars("2019-01-01", "2020-03-10")
    cut = _cuts(bars, [f"{NFP_DAYS[9]:%Y-%m-%d} 12:00"])

    def whole_sample(b):
        t = _known(b)
        t["z"] = t["dollar_surprise"] / t.groupby("event")["dollar_surprise"].transform(lambda s: s.abs().mean())
        return surprise.latest(b.index, t, "Jobs report").to_frame()

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future(whole_sample, bars, cut)


def test_the_feds_number_is_not_known_on_the_day():
    t = surprise.score(EVENTS, FF, DGS2, surprise.load_signs())
    at = surprise.latest(pd.DatetimeIndex([ny("2019-06-19 15:00"), ny("2019-06-20 16:14"), ny("2019-06-20 16:15")]),
                         t, "FOMC decision", "surprise")
    assert np.isnan(at.iloc[0]) and np.isnan(at.iloc[1]) and at.iloc[2] == pytest.approx(-0.10)


# ---- the real tables (data/news/) -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    if not any(config.NEWS_FF_DIR.glob("*.json")) or not config.NEWS_DGS2.is_file():
        pytest.skip("needs data/news/forexfactory/ (python -m machine.news.forexfactory) and data/news/dgs2.csv")
    events = calendar.load_events(importance=("high", "moderate"))
    return events, forexfactory.load(), yields.load()


@pytest.mark.news
def test_real_signs_cover_every_calendar_event(signs):
    names = set(pd.read_parquet(config.NEWS_EVENTS)["event"])
    assert names <= set(signs["event"])


@pytest.mark.news
def test_real_surprise_table(real, signs):
    events, ff, dgs2 = real
    t = surprise.score(events, ff, dgs2, signs)
    guards.check_dates(t[["utc", "server"]])
    nfp = t[(t["event"] == "Nonfarm Payrolls") & (t["ny_date"] >= "2016-01-01")]
    assert len(nfp) == 96 and (nfp["ff_match"] == "same minute").sum() >= 94
    may = nfp[nfp["ny_date"] == "2020-05-08"].iloc[0]
    assert may["ff_forecast"] == -22_000_000                               # MT5 said +139K; the consensus was -22M
    assert may["dollar_surprise"] > 0                                       # -20.5M was better than feared
    cut = t[(t["report"] == "FOMC decision") & (t["ny_date"] == "2020-03-15")].iloc[0]
    assert (cut["yield_before"], cut["yield_after"]) == (0.49, 0.36)        # Fri 13 Mar -> Mon 16 Mar (FRED DGS2)
    assert cut["known_from"] == ny("2020-03-17 16:15")


@pytest.mark.news
def test_real_surprise_passes_the_cut_off_the_future_test(real, signs):
    events, ff, dgs2 = real
    ev = events[(events["day"] >= "2018-01-01") & (events["day"] <= "2020-06-30")]
    bars = _hour_bars("2020-02-24", "2020-04-10")
    cuts = _cuts(bars, ["2020-03-03 09:00", "2020-03-06 09:00", "2020-03-16 12:00", "2020-03-17 17:00",
                        "2020-04-03 10:00"])
    guards.assert_no_future(lambda b: _features(b, events=ev, ff=ff, dgs2=dgs2, signs=signs), bars, cuts)
