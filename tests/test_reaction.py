"""How gold reacted to news (L12b.2): the release minutes, the measures on 1-minute bars, the comparison moments, and
the cut-off-the-future test for everything called "known before".

Tests with no mark run anywhere on made-up bars. Tests marked `data` + `news` read the real tables and skip without.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import config, guards, settings
from machine.data import clock
from machine.ideas.compile import Market
from machine.news import reaction
from machine.news.reaction import MIN, Measurer
from tests.block_fakes import random_bars

DAY = "2019-03-05"                       # a Tuesday in winter: server = UTC + 2, New York = UTC - 5
T0 = pd.Timestamp(f"{DAY} 15:30")        # 08:30 New York


def day_bars(after: list[tuple], t0: pd.Timestamp = T0, start: str = "00:05", end: str = "23:55",
             skip: tuple[str, str] | None = None) -> pd.DataFrame:
    """One day of 1-minute bars. Before the release every bar is open 1300.25, high 1300.50, low 1300.00, close
    1300.25 (so the hour before has high 1300.50, low 1300.00 and every 5 minutes a range of 0.50). From the release
    on: the bars in `after`, then quiet bars 0.25 either side of the last close. skip: (from, to) minutes removed."""
    idx = pd.date_range(f"{t0:%Y-%m-%d} {start}", f"{t0:%Y-%m-%d} {end}", freq="1min")
    rows = []
    last = after[-1][3] if after else 1300.25
    for t in idx:
        k = int((t - t0) / pd.Timedelta(minutes=1))
        if k < 0:
            rows.append((1300.25, 1300.50, 1300.00, 1300.25))
        elif k < len(after):
            rows.append(after[k])
        else:
            rows.append((last, last + 0.25, last - 0.25, last))
    m1 = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx.rename("server_time"))
    if skip:
        a, b = (pd.Timestamp(f"{t0:%Y-%m-%d} {x}") for x in skip)
        m1 = m1[(m1.index < a) | (m1.index >= b)]
    return m1


def one(m1: pd.DataFrame, t0: pd.Timestamp = T0) -> pd.Series:
    return Measurer(m1).measure([t0]).iloc[0]


# ---- the measures on hand-made bars -------------------------------------------------------------------------------

def test_first_moves_and_the_first_hour():
    r = one(day_bars([(1300.25, 1300.80, 1300.20, 1300.70), (1300.70, 1300.70, 1299.90, 1300.00)]))
    assert r["has_bars"] and r["p0"] == 1300.25
    assert r["move_1"] == pytest.approx(0.45)               # the close of the first minute
    assert r["move_5"] == pytest.approx(-0.25)              # the close of the fifth minute (quiet at 1300.00)
    assert r["move_60"] == pytest.approx(-0.25) and r["move_240"] == pytest.approx(-0.25)
    assert r["move_close"] == pytest.approx(-0.25)          # the broker day's last price
    assert r["up_60"] == pytest.approx(0.55) and r["up_60_min"] == 0
    assert r["dn_60"] == pytest.approx(0.50)                # quiet lows at 1299.75
    assert r["range_60"] == pytest.approx(1.05)
    assert (r["pre_hi"], r["pre_lo"], r["pre_range"], r["pre_5m"]) == (1300.50, 1300.00, 0.50, 0.50)
    assert r["before_120"] == 0.0 and r["session"] == "New York"      # 19:00 India time


def test_the_first_side_comes_back():
    """The high of the hour before breaks first; price then reaches the far side (1300.00) before 1301.00."""
    r = one(day_bars([(1300.25, 1300.80, 1300.20, 1300.70), (1300.70, 1300.70, 1299.90, 1300.00)]))
    assert (r["hr_side"], r["hr_min"], r["hr_race"]) == ("high", 0, "back")


def test_the_first_side_goes_on():
    """The low breaks first (minute 2) and price falls the range's size (0.50) further before it comes back."""
    r = one(day_bars([(1300.25, 1300.40, 1300.10, 1300.20), (1300.20, 1300.30, 1300.05, 1300.10),
                      (1300.10, 1300.10, 1299.80, 1299.85), (1299.85, 1299.90, 1299.40, 1299.45)]))
    assert (r["hr_side"], r["hr_min"], r["hr_race"]) == ("low", 2, "on")


def test_on_inside_the_break_minute_itself():
    r = one(day_bars([(1300.25, 1301.20, 1300.20, 1301.10)]))
    assert (r["hr_side"], r["hr_race"]) == ("high", "on")    # the far price came after the break: sure


def test_back_inside_the_break_minute_needs_the_close():
    """A low at the back price inside the break minute may have come BEFORE the break: only the close is sure."""
    r = one(day_bars([(1300.25, 1300.60, 1299.95, 1300.55), (1300.55, 1300.60, 1300.50, 1300.55)]))
    assert r["hr_side"] == "both"                            # high and low of the range in one minute: no side
    r = one(day_bars([(1300.25, 1300.60, 1300.00, 1300.00)]))
    assert (r["hr_side"], r["hr_race"]) == ("high", "back")  # closed at the far side: it came back


def test_no_side_breaks_and_a_race_left_open():
    r = one(day_bars([(1300.25, 1300.45, 1300.05, 1300.25)]))
    assert pd.isna(r["hr_side"]) and pd.isna(r["hr_race"]) and np.isnan(r["hr_min"])
    r = one(day_bars([(1300.25, 1300.70, 1300.20, 1300.60)]))     # broke the high, then sat between the two prices
    assert (r["hr_side"], r["hr_race"]) == ("high", "open")


def test_both_prices_inside_one_later_minute_is_unclear():
    r = one(day_bars([(1300.25, 1300.70, 1300.20, 1300.60), (1300.60, 1301.10, 1299.90, 1300.50)]))
    assert (r["hr_side"], r["hr_race"]) == ("high", "unclear")


def test_calm_again():
    """5-minute ranges after the release: 1.05, then 0.50 (the size before the release) -> calm from minute 5."""
    r = one(day_bars([(1300.25, 1300.80, 1300.20, 1300.70), (1300.70, 1300.70, 1299.90, 1300.00)]))
    assert r["calm_pre"] == 5
    wild = [(1300.25, 1301.25, 1299.25, 1300.25)] * 47            # 47 minutes of 2-dollar bars, then quiet
    assert one(day_bars(wild))["calm_pre"] == 50                  # minutes 45-49 still hold two wild bars
    assert np.isnan(one(day_bars([(1300.25, 1301.25, 1299.25, 1300.25)] * 300))["calm_pre"])   # never in 4 hours


def test_the_market_is_closed():
    r = one(day_bars([], skip=("15:20", "15:45")))               # no bar 10 minutes before the release
    assert np.isnan(r["p0"]) and not r["has_bars"]
    r = one(day_bars([], skip=("15:30", "15:50")))               # bars before, none for 20 minutes after
    assert r["p0"] == 1300.25 and not r["has_bars"] and np.isnan(r["move_5"])


def test_four_hours_later_the_market_is_closed():
    r = one(day_bars([(1300.25, 1300.45, 1300.05, 1300.25)], end="18:00"))
    assert r["has_bars"] and not r["open_240"]


def test_the_race_alone():
    m = Measurer(day_bars([(1300.25, 1300.70, 1300.20, 1300.60), (1300.60, 1300.65, 1300.55, 1300.60)]))
    j = int(np.searchsorted(m.T, T0.value))
    assert m._race(j, 1300.50, True, 0.0, len(m.T)) is None       # no distance, no race
    assert m._race(j, 1300.50, True, 0.10, j + 1) == "on"         # 1300.60 reached inside the break minute
    assert m._race(j, 1300.50, True, 5.0, len(m.T)) == "open"


# ---- the release minutes ------------------------------------------------------------------------------------------

def surprise_rows(rows: list[dict]) -> pd.DataFrame:
    base = {"ff_colour": None, "headline": True, "importance": "high", "z": np.nan, "method": "forecast",
            "group": "small", "kind": "jobs", "scheduled": True}
    t = pd.DataFrame([{**base, **r} for r in rows])
    t["server"] = pd.to_datetime(t["server"])
    t["utc"] = clock.server_to_utc(pd.DatetimeIndex(t["server"]))
    t["ist"] = t["utc"].dt.tz_convert(clock.IST_ZONE)
    t["known_from"] = t["utc"]
    t["event"] = t.get("event", t["report"])
    return t


def test_one_row_per_release_minute():
    t = surprise_rows([
        {"server": f"{DAY} 15:30", "report": "Jobs report", "ff_colour": "red", "z": 2.0},
        {"server": f"{DAY} 15:30", "report": "Jobs report", "ff_colour": "red", "z": -0.2, "headline": False},
        {"server": f"{DAY} 15:30", "report": "Trade balance", "ff_colour": "yellow", "z": -3.0},
        {"server": f"{DAY} 17:00", "report": "ISM services", "ff_colour": "orange", "z": -1.5},
        {"server": f"{DAY} 17:00", "report": "Factory orders", "ff_colour": "orange", "z": 1.2},
        {"server": f"{DAY} 18:00", "report": "Bond auction", "importance": "moderate", "method": "flagged"},
        {"server": f"{DAY} 21:00", "report": "FOMC decision", "ff_colour": "red", "z": -2.0, "method": "2y yield"},
    ])
    m = reaction.moments(t)
    assert list(m.index) == [pd.Timestamp(f"{DAY} {h}") for h in ("15:30", "17:00", "18:00", "21:00")]
    jobs = m.iloc[0]
    assert (jobs["colour"], jobs["report"], jobs["numbers"], jobs["his_news"]) == ("red", "Jobs report", 3, True)
    assert jobs["reports"] == "Jobs report | Trade balance"
    assert jobs["gold_way"] == -1 and not jobs["mixed"]      # good for the dollar = bad for gold; the yellow row is
    #                                                          not the strongest colour, so it has no say
    assert m.iloc[1]["gold_way"] == 0 and m.iloc[1]["mixed"]  # two big orange surprises that disagree
    assert m.iloc[2]["colour"] == "none" and not m.iloc[2]["his_news"] and m.iloc[2]["importance"] == "moderate"
    assert m.iloc[3]["gold_way"] == 0 and m.iloc[3]["gold_way_fed"] == 1     # the Fed: a daily number, kept apart
    assert list(m["ny_clock"]) == ["08:30", "10:00", "11:00", "14:00"] and set(m["weekday"]) == {1}


def test_comparison_moments_are_quiet_weekdays_at_the_same_clock():
    ev = reaction.moments(surprise_rows([
        {"server": "2019-03-05 15:30", "report": "Jobs report", "ff_colour": "red"},
        {"server": "2019-03-06 17:00", "report": "Bond auction", "importance": "moderate", "method": "flagged"},
    ]))
    busy = pd.DatetimeIndex(["2019-03-05 15:30", "2019-03-06 17:00", "2019-03-07 16:31", "2019-03-08 14:30"]).as_unit("ns").asi8
    ct = reaction.control_times(ev, busy, (2019,))
    assert set(ct["ny_clock"]) == {"08:30"}                  # the clock of a release with no colour is not used
    days = set(ct.index.normalize())
    assert pd.Timestamp("2019-03-05") not in days            # the release itself
    assert pd.Timestamp("2019-03-07") in days                # a release 61 minutes later: still quiet ...
    assert pd.Timestamp("2019-03-08") not in days            # ... one exactly 60 minutes before: not quiet
    assert pd.Timestamp("2019-03-06") in days and pd.Timestamp("2019-03-09") not in days     # Saturday
    summer = ct[ct.index.normalize() == pd.Timestamp("2019-07-09")].index[0]
    assert summer == pd.Timestamp("2019-07-09 15:30")        # 08:30 New York all year (both clocks move)
    gap = ct[ct.index.normalize() == pd.Timestamp("2019-03-12")].index[0]
    assert gap == pd.Timestamp("2019-03-12 14:30")           # the weeks the US is on summer time and Europe is not
    assert len(ct) == 261 - 2 and (ct["year"] == 2019).all()


# ---- the cut-off-the-future test for everything "known before" ----------------------------------------------------

def _context(times: pd.DatetimeIndex, busy: np.ndarray):
    def fn(m1: pd.DataFrame) -> pd.DataFrame:
        market = Market.from_m1(m1)
        t = times[times <= m1.index[-1]]
        return Measurer(m1, market, busy).measure(t, context_only=True).drop(columns=["session", "up_level_kind",
                                                                                     "dn_level_kind"])
    return fn


def _kinds(times: pd.DatetimeIndex, busy: np.ndarray):
    def fn(m1: pd.DataFrame) -> pd.DataFrame:
        t = times[times <= m1.index[-1]]
        out = Measurer(m1, Market.from_m1(m1), busy).measure(t, context_only=True)
        return out[["session", "up_level_kind", "dn_level_kind"]].fillna("none")
    return fn


@pytest.fixture(scope="module")
def walk():
    m1 = random_bars(n=26 * 1440, seed=7, tf_minutes=1)       # 26 days of 1-minute bars, a random walk
    m1 = m1.drop(columns=["utc"]).assign(vol=1, spread=20)    # Market.from_m1 builds the other charts from these
    times = pd.date_range("2019-01-21 15:30", "2019-02-01 15:30", freq="1D").append(
        pd.date_range("2019-01-22 17:00", "2019-01-31 17:00", freq="3D"))
    busy = pd.DatetimeIndex(["2019-01-15 15:30", "2019-01-16 17:00"]).as_unit("ns").asi8
    return m1, times.sort_values(), busy


def test_the_situation_before_passes_the_cut_off_the_future_test(walk):
    m1, times, busy = walk
    cuts = [int(m1.index.get_loc(t)) for t in (times[3], times[8])] + [int(m1.index.get_loc(times[11])) + 7]
    guards.assert_no_future(_context(times, busy), m1, cuts)
    guards.assert_no_future(_kinds(times, busy), m1, cuts)
    full = Measurer(m1, Market.from_m1(m1), busy).measure(times, context_only=True)
    assert full["adr"].notna().all() and full["normal_5m"].notna().any() and full["bias_d"].notna().all()
    assert (full["up_level_dist"].dropna() > 0).all() and (full["dn_level_dist"].dropna() > 0).all()
    assert full["up_level_dist"].notna().sum() >= 5          # the walk has levels above it most days


def test_the_cut_off_test_catches_a_reaction_called_context(walk):
    """A measure taken AFTER the release, passed off as known before, is caught."""
    m1, times, busy = walk

    def peeks(bars: pd.DataFrame) -> pd.DataFrame:
        t = times[times <= bars.index[-1]]
        return Measurer(bars, None, busy).measure(t)[["move_60"]].fillna(-999.0)

    with pytest.raises(guards.FutureLeakError):
        guards.assert_no_future(peeks, m1, [int(m1.index.get_loc(times[3])) + 10])


def test_his_levels_at_the_release_are_the_ones_not_taken_before_it(walk):
    """The nearest level each side = the nearest among the levels known before the release candle opened and not
    taken before it (checked against a plain search of the lifetimes table)."""
    m1, times, busy = walk
    market = Market.from_m1(m1)
    meas = Measurer(m1, market, busy)
    out = meas.measure(times, context_only=True)
    life = market.liquidity(settings.fingerprint()).life
    life = life[life["can_start"]]
    opens5 = market.opens("5m")
    for t, row in out.iterrows():
        i5 = int(np.searchsorted(opens5, t.value, "right")) - 1
        alive = life[(life["start"] <= i5) & (life["end"] >= i5)]
        price = alive["price"] + alive["slope"] * (i5 - alive["x0pos"])
        up = price[(alive["side"] == "high") & (price > row["p0"])]
        dn = price[(alive["side"] == "low") & (price < row["p0"])]
        assert (np.isnan(row["up_level_dist"]) and up.empty) or row["up_level_dist"] == pytest.approx(up.min() - row["p0"])
        assert (np.isnan(row["dn_level_dist"]) and dn.empty) or row["dn_level_dist"] == pytest.approx(row["p0"] - dn.max())


def test_the_first_of_his_levels_taken(walk):
    """lv_side / lv_min are read off the 1-minute bars after the release, from the two nearest levels."""
    m1, times, busy = walk
    meas = Measurer(m1, Market.from_m1(m1), busy)
    out = meas.measure(times)
    assert out["has_bars"].all()
    for t, r in out.iterrows():
        hour = m1.loc[t:t + pd.Timedelta(minutes=59)]
        up = hour.index[hour["high"] > r["p0"] + r["up_level_dist"]] if np.isfinite(r["up_level_dist"]) else []
        dn = hour.index[hour["low"] < r["p0"] - r["dn_level_dist"]] if np.isfinite(r["dn_level_dist"]) else []
        first = min([*up[:1], *dn[:1]], default=None)
        if first is None:
            assert pd.isna(r["lv_side"])
        else:
            assert r["lv_min"] == (first - t) / pd.Timedelta(minutes=1)
            both = len(up) and len(dn) and up[0] == dn[0]
            assert r["lv_side"] == ("both" if both else "high" if len(up) and up[0] == first else "low")
    assert out["lv_side"].notna().sum() >= 3


# ---- the check years stay hidden -----------------------------------------------------------------------------------

def test_the_check_years_of_the_reactions_stay_hidden(tmp_path, monkeypatch):
    path = tmp_path / "reactions.parquet"
    idx = pd.DatetimeIndex(["2019-03-05 15:30", "2022-03-04 15:30"], name="server")
    pd.DataFrame({"year": [2019, 2022], "move_5": [1.0, -2.0]}, index=idx).reset_index().to_parquet(path)
    monkeypatch.setattr(config, "NEWS_REACTIONS", path)
    assert list(reaction.load("events")["year"]) == [2019]            # the tune years unless asked
    with pytest.raises(guards.CheckYearsClosedError):
        reaction.load("events", years=config.CHECK_YEARS)
    with pytest.raises(guards.CheckYearsClosedError):                 # a belief that never opened them
        reaction.load("events", years=config.CHECK_YEARS, check_for=("belief-news-trap", "hour-range"))
    assert list(reaction.load("events", years=config.CHECK_YEARS, lookup=True)["year"]) == [2022]
    with pytest.raises(guards.CheckYearsClosedError):                 # building them needs every belief's opening
        reaction.build(config.CHECK_YEARS, write=False)


# ---- the real tables ------------------------------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.news
def test_real_jobs_report_march_2019():
    """8 March 2019, 08:30 New York = 15:30 server: the measures agree with the 1-minute bars read by hand."""
    from machine.data.bars import load_bars
    m1 = load_bars("1m", "2019-03-07", "2019-03-08")
    t0 = pd.Timestamp("2019-03-08 15:30")
    r = Measurer(m1).measure([t0]).iloc[0]
    assert r["has_bars"] and r["p0"] == m1["close"].loc[:t0 - pd.Timedelta(minutes=1)].iloc[-1]
    hour = m1.loc[t0:t0 + pd.Timedelta(minutes=59)]
    assert r["range_60"] == pytest.approx(hour["high"].max() - hour["low"].min())
    assert r["move_5"] == pytest.approx(m1["close"].loc[t0 + pd.Timedelta(minutes=4)] - r["p0"])
    assert r["range_60"] > 3 * r["pre_range"]                 # a jobs report: the hour after dwarfs the hour before
    ev = reaction.event_moments("2019-03-08", "2019-03-08")
    assert ev.loc[t0, "colour"] == "red" and "Jobs report" in ev.loc[t0, "reports"] and ev.loc[t0, "ny_clock"] == "08:30"


@pytest.mark.data
@pytest.mark.news
def test_real_situation_before_passes_the_cut_off_the_future_test():
    from machine.data.bars import load_bars
    m1 = load_bars("1m", "2019-02-01", "2019-03-15")
    busy = reaction.busy_times("2019-02-01", "2019-03-15")
    ev = reaction.event_moments("2019-03-04", "2019-03-15")
    times = pd.DatetimeIndex(ev.index[ev["his_news"]])
    assert len(times) >= 10
    cuts = [int(m1.index.searchsorted(t)) for t in (times[2], times[len(times) // 2], times[-2])]
    guards.assert_no_future(_context(times, busy), m1, cuts)
    guards.assert_no_future(_kinds(times, busy), m1, cuts)
