"""His news beliefs, judged the referee's way (L12b.2): the fair comparison, the rules fixed in advance, planted
differences caught, no difference -> "chance", already seen can never be "holds", the check years opened once.

Everything here runs on made-up tables (no price data, no calendar).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from machine import config, guards
from machine.news import beliefs, reaction
from machine.news.beliefs import CHANCE, FEW, FRESH, HOLDS, OPPOSITE
from machine.referee import splits

GOLD = beliefs.get("belief-news-moves-gold", "red-and-orange")
WAY = beliefs.get("belief-news-direction", "first-15-minutes")
TRAP = beliefs.get("belief-news-trap", "hour-range")
LEVELS = beliefs.get("belief-news-trap", "his-levels")
AGAINST = beliefs.get("belief-news-trap", "good-or-bad-news")


def world(seed: int = 0, years=(2016, 2017, 2018), per_year: int = 50, controls: int = 40, back_ev: float = 0.5,
          back_ct: float = 0.5, lv_back_ev: float = 0.5, range_ev: float = 0.30, range_ct: float = 0.30,
          with_news: float = 0.5, against: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Made-up releases (one a week, Fridays 08:30 New York, red and orange in turn) and comparison moments (the
    same clock and weekday). back_*: the share of first sides that come back; range_*: the middle first-hour range;
    with_news: the share of releases where gold goes the way the surprise says; against: the share where the first
    side that breaks is the side against the news."""
    rng = np.random.default_rng(seed)

    def frame(times: pd.DatetimeIndex, year: int, back: float, lv_back: float, rng_mid: float) -> pd.DataFrame:
        n = len(times)
        return pd.DataFrame({
            "year": year, "ny_clock": "08:30", "weekday": 4, "has_bars": True,
            "hr_race": np.where(rng.random(n) < back, "back", "on"),
            "lv_race": np.where(rng.random(n) < lv_back, "back", "on"),
            "range_60_adr": rng_mid * np.exp(rng.normal(0, 0.3, n)),
        }, index=times.rename("server"))

    ev, ct = [], []
    for y in years:
        e = frame(pd.date_range(f"{y}-01-01", periods=per_year, freq="W-FRI") + pd.Timedelta("15:30:00"), y,
                  back_ev, lv_back_ev, range_ev)
        e["his_news"] = True
        e["colour"] = np.where(np.arange(per_year) % 2 == 0, "red", "orange")
        e["his_reports"] = np.where(e["colour"] == "red", "Jobs report", "ISM services")
        e["gold_way"] = rng.choice([-1, 1], per_year)
        e["move_15"] = e["gold_way"] * np.where(rng.random(per_year) < with_news, 1.0, -1.0)
        low_first = (e["gold_way"] == 1) == (rng.random(per_year) < against)
        e["hr_side"] = np.where(low_first, "low", "high")
        ev.append(e)
        ct.append(frame(pd.date_range(f"{y}-01-01", periods=controls, freq="W-THU") + pd.Timedelta("15:30:00"), y,
                        back_ct, back_ct, range_ct))
    return pd.concat(ev), pd.concat(ct)


def answer(b, ev, ct, years=(2016, 2017, 2018), way=None):
    return beliefs.judge(b, ev, ct, years, way=way, record=False)


# ---- the rules -------------------------------------------------------------------------------------------------------

def test_the_rules_are_fixed_in_advance():
    """Changing a rule after seeing a result changes this fingerprint - and shows in git."""
    assert beliefs.RULES_SHA256 == "c8bae6e2c5a49deb416b0b19577f74f8c504d8751f25d5742f7c0b424f52b9c5"
    assert beliefs.FAMILIES == ("belief-news-moves-gold", "belief-news-direction", "belief-news-trap")
    assert [b.version_id for b in beliefs.versions("belief-news-trap")] == ["hour-range", "his-levels",
                                                                           "good-or-bad-news"]
    for b in beliefs.BELIEFS:
        assert b.his_words and b.source and b.counted and b.compare in ("same clock", "coin")
        assert bool(b.seen_note) == (b.idea_id == "belief-news-trap")       # the news trap is "already seen"


# ---- the fair comparison ------------------------------------------------------------------------------------------

def test_the_comparison_is_the_same_clock_weekday_and_year():
    ev, ct = world(per_year=6, controls=10)
    ct.loc[ct["year"] == 2016, "range_60_adr"] = 0.2
    ct.loc[ct["year"] == 2017, "range_60_adr"] = 0.4
    rows = beliefs.compare(GOLD, ev, ct)
    assert set(rows.loc[rows["year"] == 2016, "comp"].round(9)) == {0.2}
    assert set(rows.loc[rows["year"] == 2017, "comp"].round(9)) == {0.4}
    assert (rows["match"] == "same weekday").all() and (rows["comp_n"] == 10).all()
    assert np.allclose(rows["d"], rows["x"] - rows["comp"]) and rows.attrs["no_comparison"] == 0


def test_too_few_on_the_weekday_falls_back_to_any_weekday_then_to_not_counted():
    ev, ct = world(per_year=6, controls=10)
    ct16 = ct["year"] == 2016
    ct.loc[ct16 & (np.arange(len(ct)) >= 4), "weekday"] = 2             # 2016: only 4 on the weekday, 6 on another
    ct = ct[~((ct["year"] == 2017) & (np.arange(len(ct)) % 10 >= 3))]   # 2017: 3 in all
    rows = beliefs.compare(GOLD, ev, ct)
    assert set(rows.loc[rows["year"] == 2016, "match"]) == {"any weekday"}
    assert set(rows.loc[rows["year"] == 2016, "comp_n"]) == {10}
    assert 2017 not in set(rows["year"]) and rows.attrs["no_comparison"] == 6
    assert set(rows.loc[rows["year"] == 2018, "match"]) == {"same weekday"}


def test_a_release_with_no_prices_or_not_his_colour_is_not_counted():
    ev, ct = world(per_year=6, controls=10)
    ev.iloc[0, ev.columns.get_loc("has_bars")] = False
    ev.iloc[1, ev.columns.get_loc("his_news")] = False
    ev.iloc[2, ev.columns.get_loc("hr_race")] = "open"
    assert len(beliefs.compare(GOLD, ev, ct)) == 16 and len(beliefs.compare(TRAP, ev, ct)) == 15


def test_a_coin_belief_counts_only_big_surprises():
    ev, ct = world(per_year=6)
    ev.iloc[:4, ev.columns.get_loc("gold_way")] = 0
    ev.iloc[4, ev.columns.get_loc("move_15")] = 0.0                      # no move: half a yes
    rows = beliefs.compare(WAY, ev, None)
    assert len(rows) == 14 and (rows["comp"] == 0.5).all() and rows["x"].iloc[0] == 0.5


def test_fake_sets_sit_around_no_difference():
    ev, ct = world(range_ev=0.6, with_news=0.8)
    same = beliefs.fake_sets(beliefs.compare(GOLD, ev, ct), GOLD, 200, seed=1)
    coin = beliefs.fake_sets(beliefs.compare(WAY, ev, None), WAY, 200, seed=1)
    assert abs(np.median(same)) < 0.02 and abs(np.median(coin)) < 0.02      # whatever the real releases did


# ---- the answers -----------------------------------------------------------------------------------------------------

def test_a_planted_difference_holds():
    r = answer(GOLD, *world(range_ev=0.45, range_ct=0.30))
    assert r.word == HOLDS and r.way == 1 and all(c.passed for c in r.checks)
    assert r.numbers["overall"]["n"] == 150 and r.numbers["overall"]["diff"] > 0.1
    assert set(r.numbers["slices"]) == {"red", "orange"} and r.numbers["slices"]["red"]["n"] == 75
    assert "ANSWER: HOLDS" in r.text() and "THE RULES FOR A BELIEF" in r.text()


def test_the_opposite_is_found_too():
    r = answer(GOLD, *world(range_ev=0.20, range_ct=0.30))
    assert r.word == OPPOSITE and r.way == -1


def test_a_coin_belief_holds_when_gold_follows_the_news():
    ev, ct = world(with_news=0.72)
    r = answer(WAY, ev, None)
    assert r.word == HOLDS and 0.6 < r.numbers["overall"]["event"] < 0.85
    assert answer(WAY, *world(with_news=0.5)[:1], None).word == CHANCE


def test_no_difference_is_chance_nearly_always():
    """40 worlds where news changes nothing, judged one way (as on the check years): "holds" at most 4 times."""
    words = [answer(GOLD, *world(seed=s), way=1).word for s in range(40)]
    assert words.count(HOLDS) <= 4 and words.count(OPPOSITE) == 0 and words.count(CHANCE) >= 36


def test_already_seen_can_never_hold():
    """The news trap, with a planted difference as strong as the old project's (32 in 100 come back, 50 on quiet
    days): every other check passes the opposite way, and the answer is still "needs fresh years"."""
    ev, ct = world(back_ev=0.32, back_ct=0.50)
    r = answer(TRAP, ev, ct)
    assert r.word == FRESH and r.way == -1 and "pass every other check" in r.why
    assert [c.passed for c in r.checks] == [True, True, True, True, True, False]
    assert answer(TRAP, *world(back_ev=0.5, back_ct=0.5)).word == FRESH        # and with nothing in it: the same word
    assert "Already seen: YES" in r.text()


def test_one_year_against_is_chance_and_the_worst_year_is_named():
    ev, ct = world(range_ev=0.45, range_ct=0.30)
    ev.loc[ev["year"] == 2017, "range_60_adr"] = 0.25
    r = answer(GOLD, ev, ct)
    year_check = r.checks[4]
    assert r.word == CHANCE and not year_check.passed and "2017" in year_check.shown and "every year" in r.why


def test_few_releases_is_not_enough_to_tell():
    r = answer(GOLD, *world(per_year=8, range_ev=0.6))
    assert r.word == FEW and r.numbers["overall"]["n"] == 24


def test_three_helpful_releases_cannot_carry_a_belief():
    ev, ct = world(seed=3, range_ev=0.26, range_ct=0.30)              # the rest points the other way
    ev.iloc[[0, 60, 120], ev.columns.get_loc("range_60_adr")] = 40.0          # three freak releases
    r = answer(GOLD, ev, ct, way=1)
    assert r.word == CHANCE and not r.checks[2].passed


# ---- the check years: opened once, for one version, judged one way ---------------------------------------------------

def test_the_check_years_open_once_for_one_version():
    ev, ct = world(years=(2016, 2017, 2021, 2022), back_ev=0.3)
    with pytest.raises(guards.CheckYearsClosedError):                      # closed until opened
        beliefs.judge(TRAP, ev, ct, (2021, 2022), way=-1)
    tuned = beliefs.judge(TRAP, ev, ct, (2016, 2017))                      # a trial file on the tune years first
    beliefs.judge(LEVELS, ev, ct, (2016, 2017))
    assert json.loads(tuned.trial_path.read_text(encoding="utf-8"))["kind"] == "belief"
    splits.open_check(TRAP.idea_id, TRAP.version_id, who="test", why="chosen on the tune years")
    with pytest.raises(ValueError, match="ONE way"):
        beliefs.judge(TRAP, ev, ct, (2021, 2022))
    final = beliefs.judge(TRAP, ev, ct, (2021, 2022), way=-1, final=True)
    assert final.final and "check years 2021-2022" in final.judged_on and final.word == FRESH
    with pytest.raises(guards.CheckYearsClosedError):                      # another version of the same belief
        beliefs.judge(LEVELS, ev, ct, (2021, 2022), way=-1)
    with pytest.raises(guards.CheckYearsClosedError):                      # and the family cannot open them again
        splits.open_check(LEVELS.idea_id, LEVELS.version_id, who="test", why="again")


def test_tune_chooses_one_version_per_belief_once(monkeypatch):
    ev, ct = world(years=config.TUNE_YEARS, per_year=40, back_ev=0.45, lv_back_ev=0.25, range_ev=0.5,
                   with_news=0.7)
    monkeypatch.setattr(reaction, "load", lambda which, **kw: ev if which == "events" else ct)
    with pytest.raises(guards.CheckYearsClosedError):
        beliefs.require_open()
    reps = beliefs.tune()
    chosen = beliefs.read_chosen()
    assert set(chosen) == set(beliefs.FAMILIES) and [len(v) for v in reps.values()] == [1, 1, 3]
    assert chosen["belief-news-trap"]["version_id"] == "his-levels" and chosen["belief-news-trap"]["way"] == -1
    assert chosen["belief-news-moves-gold"]["way"] == 1 and chosen["belief-news-direction"]["way"] == 1
    assert len(list(config.TRIALS_DIR.glob("*.json"))) == 5                   # one trial file per version
    assert (config.BELIEFS_DIR / "news-trap-tune.md").is_file()
    with pytest.raises(ValueError, match="already has its chosen version"):
        beliefs.tune()
    for fam, c in chosen.items():
        splits.open_check(fam, c["version_id"], who="test", why="chosen")
    beliefs.require_open()                                                    # every belief opened: no refusal
    assert splits.opened_count() == 3
