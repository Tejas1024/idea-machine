"""'The last 10 times' (L12b.3): the lookup on made-up tables (the rule, no future, the sealed years, the lines) and the
news picture on made-up candles. Tests marked `data` + `news` read the real tables and skip without.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from machine import config, guards
from machine.charts import draw as D
from machine.charts import lint
from machine.data import clock
from machine.news import similar as S
from tests import chart_fakes as F

T0 = pd.Timestamp("2019-06-07 15:30")                 # a Friday 08:30 New York (server = UTC + 3 in June)


def rows(n: int = 30, report: str = "Jobs report", end: pd.Timestamp = T0, every: str = "7D", **cols) -> pd.DataFrame:
    """n made-up releases, the last at `end`; every value the same unless given (a value or a list of n)."""
    times = pd.date_range(end=end, periods=n, freq=every)
    base = dict(time=times, report=report, event="Nonfarm Payrolls", z=0.2, group="small", method="forecast",
                number="100K (forecast 90K)", own_colour="red", session="New York", bias_d=0.0, bias_w=0.0,
                up_level_kind="session_high", up_level_adr=0.3, dn_level_kind="session_low", dn_level_adr=0.3,
                adr_used=0.5, before_120_adr=0.0, adr=20.0, p0=1300.0, up_level_dist=6.0, dn_level_dist=6.0,
                move_5=-1.0, move_15=-1.0, move_60=-2.0, range_60=5.0, lv_side=None, lv_min=np.nan, lv_race=None,
                normal_5m=1.0, calm_normal=30.0, usd_inr=70.0)
    base.update(cols)
    t = pd.DataFrame(base)
    t["known_from"] = clock.server_to_utc(pd.DatetimeIndex(t["time"]))
    t["done_at"] = t["time"] + pd.Timedelta(minutes=S.DONE_MIN)
    return t


def ask(tab, t=T0, when="after", report="Jobs report"):
    return S.lookup(tab, S.query_of(tab, report, t, when))


# ---- which releases may be looked at ------------------------------------------------------------------------------

def test_only_the_same_news_released_and_done_before_the_question():
    tab = rows(20)
    extra = pd.concat([
        rows(1, end=T0 - pd.Timedelta(hours=3)),                           # released before, but not done (240 min)
        rows(1, end=T0 - pd.Timedelta(hours=5)),                           # done before
        rows(1, end=T0 + pd.Timedelta(days=1)),                            # after the question
        rows(1, end=T0 - pd.Timedelta(days=3), report="CPI"),              # another news
    ])
    tab = pd.concat([tab, extra], ignore_index=True).sort_values("time", ignore_index=True)
    res = ask(tab)
    assert (res.base["report"] == "Jobs report").all()
    assert (res.base["done_at"] <= T0).all() and T0 - pd.Timedelta(hours=3) not in set(res.base["time"])
    assert T0 - pd.Timedelta(hours=5) in set(res.base["time"])
    assert len(res.base) == 19 + 1 and len(res.near) == S.K                # the question itself is not done yet


def test_after_the_number_only_the_same_surprise_group_counts_before_it_all():
    grp = ["big good for the dollar"] * 12 + ["small"] * 17 + ["big good for the dollar"]
    z = [1.5] * 12 + [0.1] * 17 + [1.4]
    tab = rows(30, group=grp, z=z)
    after = ask(tab, when="after")
    assert after.candidates == 12 and (after.near["group"] == "big good for the dollar").all()
    before = ask(tab, when="before")
    assert before.candidates == 29 and "step_surprise" not in before.near


def test_one_step_of_each_part():
    q = S.Query("Jobs report", T0, "after", z=1.0, group="big good for the dollar",
                situation=dict(session="New York", bias_d=1.0, bias_w=1.0, up_level_kind="session_high",
                               up_level_adr=0.2, dn_level_kind="session_low", dn_level_adr=0.2, adr_used=0.5,
                               before_120_adr=0.1, adr=20.0))
    same = dict(z=1.0, session="New York", bias_d=1.0, bias_w=1.0, up_level_kind="session_high", up_level_adr=0.2,
                dn_level_kind="session_low", dn_level_adr=0.2, adr_used=0.5, before_120_adr=0.1, adr=20.0)
    one = [dict(z=2.0), dict(session="London"), dict(bias_d=-1.0), dict(bias_w=-1.0), dict(up_level_adr=0.7),
           dict(dn_level_adr=0.7), dict(adr_used=1.0), dict(before_120_adr=-0.15), dict(adr=30.0)]
    half = [dict(bias_d=0.0), dict(bias_w=0.0)]
    pool = pd.DataFrame([{**same, **d} for d in [{}] + one + half])
    got = S.steps(pool, q)["steps"].round(9).tolist()
    assert got == [0.0] + [1.0] * len(one) + [0.5, 0.5]


def test_caps_missing_values_and_no_level():
    q = S.Query("Jobs report", T0, "after", z=5.0, group="big good for the dollar",
                situation=dict(session="New York", up_level_kind=None, up_level_adr=np.nan, adr=20.0))
    pool = pd.DataFrame([
        dict(z=3.0, session="New York", up_level_kind=None, up_level_adr=np.nan, adr=20.0),      # z capped at 3
        dict(z=3.0, session=None, up_level_kind="equal_high", up_level_adr=2.5, adr=np.nan),      # far = 1 day
    ]).assign(bias_d=np.nan, bias_w=np.nan, dn_level_kind=None, dn_level_adr=np.nan, adr_used=np.nan,
              before_120_adr=np.nan)
    p = S.steps(pool, q)
    assert p.loc[0, "steps"] == 0.0                     # parts the question lacks are left out for all
    assert p.loc[1, "step_session"] == 1.0 and p.loc[1, "step_adr"] == 1.0   # the past release lacks it: 1 step
    assert p.loc[1, "step_level_up"] == 0.0             # a level over one day's range away = none alive
    assert "step_bias_d" not in p and "step_level_dn" in p       # no level below in the question = far, compared


def test_ties_go_to_the_more_recent_and_k_is_10_or_fewer():
    res = ask(rows(30))
    assert len(res.near) == 10 and (res.near["steps"] == 0).all()
    assert list(res.near["time"]) == sorted(res.near["time"], reverse=True)
    assert res.near["time"].iloc[0] == T0 - pd.Timedelta(days=7)
    few = ask(rows(4))
    assert len(few.near) == 3


def test_the_feds_number_is_not_known_at_the_release():
    tab = rows(20, report="FOMC decision", method="2y yield", z=-2.0, group="big bad for the dollar",
               event="Fed Interest Rate Decision")
    tab["known_from"] = tab["known_from"] + pd.Timedelta(days=1)        # the yield of the whole day: next day
    q = S.query_of(tab, "FOMC decision", T0, "after")
    assert q.when == "before" and not q.has_number and "next day" in q.note
    res = S.lookup(tab, q)
    assert res.candidates == 19 and "step_surprise" not in res.near


def test_no_future_the_lookup_never_reads_a_later_release():
    rng = np.random.default_rng(3)
    n = 60
    tab = rows(n, z=rng.normal(0, 1.2, n), bias_d=rng.choice([-1.0, 0.0, 1.0], n), adr=rng.uniform(10, 30, n),
               before_120_adr=rng.normal(0, 0.2, n), move_5=rng.normal(0, 1, n), move_60=rng.normal(0, 2, n))
    tab["group"] = np.where(tab["z"] >= 1, "big good for the dollar",
                            np.where(tab["z"] <= -1, "big bad for the dollar", "small"))

    def mean_of_the_ten(df: pd.DataFrame) -> pd.Series:
        t = df.reset_index()
        out = {}
        for when in t["time"]:
            res = S.lookup(t, S.query_of(t, "Jobs report", when, "after"))
            out[when] = res.near["move_5"].mean() if len(res.near) else np.nan
        return pd.Series(out).rename("ten")

    guards.assert_no_future(mean_of_the_ten, tab.set_index("time"), cut_points=[20, 40])
    # ... and scrambling what gold did AFTER the question changes nothing in the answer
    res = ask(tab, t=tab["time"].iloc[40])
    later = tab.copy()
    later.loc[41:, ["move_5", "move_60", "range_60"]] = 99.0
    again = ask(later, t=tab["time"].iloc[40])
    assert [x.text for x in res.lines] == [x.text for x in again.lines]


def test_a_sealed_year_is_refused():
    tab = rows(10, end=pd.Timestamp("2024-01-05 15:30"))
    with pytest.raises(guards.SealedYearsError):
        S.lookup(tab, S.query_of(tab, "Jobs report", tab["time"].iloc[-1], "before"))


# ---- the summary lines ----------------------------------------------------------------------------------------------

def test_luck_is_the_count_of_a_draw_without_putting_back():
    assert S.luck(2, 2, 2, 4) == pytest.approx(1 / 6)                   # both of the 2 that went this way
    assert S.luck(0, 2, 2, 4) == pytest.approx(1 / 6)                   # none of them: 0 or fewer
    assert S.luck(1, 2, 2, 4) == pytest.approx(5 / 6)                   # 1 = the usual: 1 or more
    assert S.luck(10, 10, 12, 12) == pytest.approx(1.0)                 # all went this way: luck gives it always
    big = S.luck(7, 10, 5000, 10000)                                    # a huge pool: the binomial count
    assert big == pytest.approx(sum(math.comb(10, i) for i in range(7, 11)) / 1024, rel=1e-3)
    assert math.isnan(S.luck(3, 10, 5, 8))                              # 10 cannot be picked from 8


def test_each_line_has_the_base_rate_the_last_3_years_and_the_warning():
    moves = [-1.0] * 7 + [1.0] * 3
    tab = rows(40, move_5=[-1.0, 1.0] * 15 + moves)
    res = ask(tab, when="before")
    first = res.lines[0].text
    assert first.startswith("7 of 10 dropped in the first 5 minutes (on all 39 NFP days before it: ")
    assert "the last 3 years (39 days): " in first and "same kind" not in first     # before the number: no kind
    assert ("10 events is few — treat as a hint: by luck alone (10 picked at random from all 39), 7 or more of 10 "
            "happens about") in first
    assert all("is few — treat as a hint" in ln.text or "not counted" in ln.text or "nothing" in ln.text
               or "no turn back" in ln.text for ln in res.lines)
    assert [ln.what for ln in res.lines] == ["in the first 5 minutes", "after 1 hour", "the first hour",
                                             "swept first", "after the sweep", "calm again"]
    assert "₹350 on 0.01 lot" in res.lines[2].text                      # $5.00 at ₹70


def test_after_the_number_the_lines_also_compare_with_the_same_kind_of_surprise():
    """The fair comparison for what the situation adds: releases with the same kind of surprise. The luck count
    picks its 10 from them (all of them drop here, so 10 of 10 dropping is what luck gives every time)."""
    grp = ["big good for the dollar"] * 12 + ["small"] * 17 + ["big good for the dollar"]
    moves = [-1.0] * 12 + [1.0] * 17 + [-1.0]
    res = ask(rows(30, group=grp, z=[1.5] * 12 + [0.1] * 17 + [1.4], move_5=moves), when="after")
    first = res.lines[0].text
    assert first.startswith("10 of 10 dropped in the first 5 minutes (on all 29 NFP days before it: 41 of 100 · "
                            "with the same kind of surprise (12): 12 of 12 · the last 3 years (29 days): 41 of 100)")
    assert "(10 picked at random from those 12), 10 or more of 10 happens about 100 times in 100" in first


def test_the_sweep_lines_count_his_levels():
    side = ["high"] * 6 + ["low"] * 2 + [None] * 2
    race = ["back"] * 2 + ["on"] * 4 + ["open", "back"] + [None] * 2
    tab = rows(10, lv_side=side, lv_race=race)
    tab = pd.concat([tab, rows(1, end=T0 + pd.Timedelta(days=7))], ignore_index=True)
    res = ask(tab, t=T0 + pd.Timedelta(days=7), when="before")
    swept = next(ln.text for ln in res.lines if ln.what == "swept first")
    assert swept.startswith("Your nearest level ABOVE was swept first in 6 of 10, the one BELOW in 2, neither in "
                            "the first hour in 2")
    after = next(ln.text for ln in res.lines if ln.what == "after the sweep")
    assert "turned back through the level (your news trap) in 3 of those 8, went on in 4" in after


def test_every_page_and_line_says_it_is_a_lookup_not_a_model():
    assert "not a trained prediction model" in S.LOOKUP_WORDS
    assert "not a trained prediction model" in S.__doc__.replace("\n", " ").lower()


# ---- the picture ----------------------------------------------------------------------------------------------------

def release_row(m5: pd.DataFrame, i: int, **cols) -> pd.Series:
    t0 = m5.index[i]
    p0 = float(m5["close"].iloc[i - 1])
    r = dict(time=t0, report="Jobs report", event="Nonfarm Payrolls", number="336K (forecast 171K)",
             group="big good for the dollar", method="forecast", p0=p0,
             up_level_kind="session_high", up_level_dist=float(m5["high"].iloc[i:i + 6].max() - p0 - 0.01),
             dn_level_kind="session_low", dn_level_dist=float(p0 - m5["low"].iloc[i - 30:i].min() + 1.5),
             lv_side="high", lv_min=float(m5["high"].iloc[i:i + 6].to_numpy().argmax() * 5), lv_race="on",
             move_5=-1.2, move_60=2.5, range_60=6.1, usd_inr=70.0)
    r.update(cols)
    return pd.Series(r)


def test_the_news_picture_draws_the_release_and_the_levels_and_passes_every_rule(tmp_path):
    m5 = F.bars_5m(300)
    row = release_row(m5, 150)
    pic = S.picture(row, m5, None, subject="test news picture 1")
    drawn = D.render(pic, tmp_path / "news.jpg")
    assert lint.check(drawn, shown=tmp_path / "shown.jsonl") == []
    assert drawn.kind == "news" and len(drawn.panels) == 1 and drawn.panels[0].tf == "5m"
    assert drawn.panels[0].n_bars == S.BEFORE_BARS + S.AFTER_BARS            # 60 minutes before, 240 after
    moment = next(i for i in drawn.items if i.kind == "Moment")
    assert moment.visible and moment.name == f"NFP out {S._ist(row['time']):%H:%M}"
    assert any(t.text == moment.name for t in drawn.texts)
    levels = [i for i in drawn.items if i.kind == "Level"]
    assert len(levels) == 2 and all(i.visible for i in levels)
    swept = next(i for i in levels if i.name.endswith(f"\nswept {S._ist(row['time'] + pd.Timedelta(minutes=row['lv_min'])):%H:%M}"))
    assert swept.info["last_time"] == row["time"] + pd.Timedelta(minutes=row["lv_min"])   # it ends where it was swept


def test_a_level_not_swept_first_says_what_the_eye_sees():
    """CPI 12 Dec 2023: the 19:00 wick stopped 3 cents under the New York high - a touch to the eye, not a sweep."""
    hour = pd.DataFrame({"high": [1996.68, 1996.44], "low": [1985.67, 1993.51]},
                        index=pd.DatetimeIndex(["2023-12-12 15:30", "2023-12-12 15:35"]))
    assert S._second_words(True, 1996.71, hour) == "missed by 3 cents"
    assert S._second_words(True, 1996.68, hour) == "touched, not past"
    assert S._second_words(True, 1996.50, hour) == "swept second, 19:00"
    assert S._second_words(True, 1999.00, hour) == ""                  # far away: nothing to say
    assert S._second_words(False, 1985.70, hour) == "swept second, 19:00"


def test_both_sides_swept_in_one_minute_is_named_so(tmp_path):
    """reaction.py has no sweep minute when both levels were passed inside one 1-minute candle: neither is second."""
    m5 = F.bars_5m(300)
    i = 150
    p0 = float(m5["close"].iloc[i - 1])
    row = release_row(m5, i, lv_side="both", lv_min=np.nan,
                      up_level_dist=float(m5["high"].iloc[i] - p0 - 0.01), dn_level_dist=float(p0 - m5["low"].iloc[i] - 0.01))
    drawn = D.render(S.picture(row, m5, None, subject="test news picture 5"), tmp_path / "both.jpg")
    names = [i.name for i in drawn.items if i.kind == "Level"]
    assert len(names) == 2 and all(n.endswith(f"\nboth swept at {S._ist(m5.index[150]):%H:%M}") for n in names)
    assert lint.check(drawn, shown=tmp_path / "shown.jsonl") == []


def test_a_move_that_began_in_the_minute_before_is_marked_not_matched():
    """NFP 5 Aug 2016: 1360.57 -> 1355.23 in the 15:29 candle, the release at 15:30:00 (the broker's clock a few
    seconds behind). Marked on the pages; a normal release and an ordinary busy minute are not."""
    idx = pd.date_range("2016-08-05 14:50", "2016-08-05 15:40", freq="1min")
    m1 = pd.DataFrame({"open": 1360.8, "high": 1361.0, "low": 1360.6, "close": 1360.8}, index=idx)
    ohlc = ["open", "high", "low", "close"]
    m1.loc["2016-08-05 15:29", ohlc] = (1360.57, 1360.82, 1355.23, 1355.23)   # the move began here (the real one) ...
    m1.loc["2016-08-05 15:30", ohlc] = (1355.22, 1355.29, 1339.25, 1352.51)   # ... and went on in the release minute
    t0 = pd.Timestamp("2016-08-05 15:30")
    assert S.early_starts(m1, [t0]) == {t0}

    def changed(**bar):
        x = m1.copy()
        x.loc["2016-08-05 15:29", list(bar)] = list(bar.values())
        return S.early_starts(x, [t0])

    assert changed(high=1361.0, low=1360.6, close=1360.8, open=1360.8) == set()   # a quiet minute before
    assert changed(high=1360.9, low=1360.0, close=1360.0) == set()                 # 5x the usual but under $1
    assert changed(close=1360.50) == set()           # a spike that came back (a spread widening before the news)
    assert "early" not in " ".join(S.STEPS) and not any("early" in c for c in S.SITUATION)   # never matched on


def test_the_release_line_sits_between_two_candles(tmp_path):
    m5 = F.bars_5m(300)
    pic = S.picture(release_row(m5, 150), m5, None, subject="test news picture 2")
    drawn = D.render(pic, tmp_path / "news.png")
    p = drawn.panels[0]
    moment = next(i for i in drawn.items if i.kind == "Moment")
    x = float(moment.paths[0][0][0])
    k = S.BEFORE_BARS                                    # the release candle is the 13th on the chart
    left, right = p.candles[k - 1], p.candles[k]
    assert left.x1 < x < right.x0                        # on no candle (#116)


def test_a_word_on_the_release_line_is_caught(tmp_path):
    """The release line is a line like any other: a name written over it fails label_overlap."""
    m5 = F.bars_5m(300)
    t0 = m5.index[150]
    mid = float(m5["close"].iloc[138:198].median())
    pic = D.Picture(kind="news", subject="test news picture 3", heading="NFP", rows=[[D.Panel(
        m5.iloc[138:198], "5m", [D.Candles(), D.Moment(t0, "NFP out"),
                                 D.Level(mid, "a name over the release line", start=m5.index[149],
                                         end=m5.index[160])], "the release", clock="ist")]])
    problems = lint.check(D.render(pic, tmp_path / "x.png"), shown=tmp_path / "shown.jsonl")
    assert any(p.rule == "label_overlap" and "moment" in p.says for p in problems)


def test_the_release_name_stays_joined_to_its_line_when_slid_aside(tmp_path):
    """Two releases one candle apart: the second name moves aside (a dashed leader joins it to its line) and no word
    sits on either line."""
    m5 = F.bars_5m(300)
    pic = D.Picture(kind="news", subject="test news picture 4", heading="two releases", rows=[[D.Panel(
        m5.iloc[138:198], "5m", [D.Candles(), D.Moment(m5.index[150], "NFP out"),
                                 D.Moment(m5.index[151], "claims out")], "the releases", clock="ist")]])
    drawn = D.render(pic, tmp_path / "x.png")
    xs = [float(i.paths[0][0][0]) for i in drawn.items if i.kind == "Moment"]
    boxes = [next(t.box for t in drawn.texts if t.text == w) for w in ("NFP out", "claims out")]
    assert sum(b.x0 < x < b.x1 for b, x in zip(boxes, xs)) <= 1            # one of them had to move aside ...
    assert lint.check(drawn, shown=tmp_path / "shown.jsonl") == []          # ... and nothing overlaps


# ---- the real tables (Tejas's PC) -----------------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.news
def test_the_real_releases_are_2016_to_2023_headline_minutes_with_their_own_surprise():
    if not config.NEWS_REACTIONS.is_file():
        pytest.skip("data/news/reactions.parquet is missing (python -m machine.news.reaction)")
    tab = S.releases()
    assert tab["time"].min().year == 2016 and tab["time"].max().year == 2023
    fed = tab[tab["report"] == "FOMC decision"]
    assert len(fed) == 65 and not fed["time"].dt.date.duplicated().any()      # one release a meeting day ...
    assert "14:30" not in set(fed["ny_clock"])                                 # ... never the press conference
    assert set(fed["event"]) == {"Fed Interest Rate Decision", "FOMC Statement"}   # 23 Mar 2020: the statement only
    assert not tab.duplicated(["time", "report"]).any()
    nfp = tab[(tab["report"] == "Jobs report") & (tab["time"] == pd.Timestamp("2023-10-06 15:30"))].iloc[0]
    assert nfp["number"] == "336K (forecast 171K)" and nfp["group"] == "big good for the dollar"
    res = S.lookup(tab, S.query_of(tab, "Jobs report", nfp["time"], "after"))
    assert len(res.near) == 10 and (res.near["group"] == nfp["group"]).all()
    assert (res.near["done_at"] <= nfp["time"]).all() and (res.base["time"] < nfp["time"]).all()
    cut = tab[tab["time"] < nfp["time"]]
    again = S.lookup(pd.concat([cut, tab[tab["time"] == nfp["time"]]]),
                     S.query_of(tab, "Jobs report", nfp["time"], "after"))
    assert [x.text for x in res.lines] == [x.text for x in again.lines]     # later years change nothing
