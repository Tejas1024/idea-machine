"""L20 — the money plan's report (machine/money/report.py) on MADE-UP trades: the page, its two pictures, its tables
and its one trial file. No price data needed (the rupee rates are in git). The trial file goes to a temporary folder
(tests/conftest.py), never into plan/work/trials/.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from machine import config, costs
from machine.money import plan as MP
from machine.money import report as R
from machine.report.build import check_words

QUIET = lambda *a, **k: None                                                     # noqa: E731


def made_up_trades(seed: int = 5, n: int = 700, win_gross=(300.0, 900.0), loss_gross=(40.0, 100.0),
                   wins: float = 0.13) -> pd.DataFrame:
    """Trades shaped like idea #1's: about 1 in 8 wins, a win is several times a loss, and the Rs 40 cost turns a
    small gain in price into a loss. One every 20-160 hours from Feb 2016 (they end in 2023)."""
    rng = np.random.default_rng(seed)
    t, rows = pd.Timestamp("2016-02-01 09:00"), []
    for k in range(n):
        t = t + pd.Timedelta(hours=int(rng.integers(20, 160)))
        won = rng.random() < wins
        gross = float(rng.uniform(*win_gross)) if won else -float(rng.uniform(*loss_gross))
        net = gross - 40.0
        rows.append({"signal": f"made-up-{k}", "direction": "buy" if k % 2 else "sell", "open": t,
                     "close": t + pd.Timedelta(hours=3), "closed_by": "target" if won else "stop",
                     "gross_inr": gross, "cost_inr": 40.0, "net_inr": net, "usd_inr": 70.0,
                     "low_inr": min(net, -40.0 - float(rng.uniform(0, 30))),
                     "segment": "2016-2020" if t.year <= 2020 else "2021-2023"})
    return pd.DataFrame(rows)


def made_up_needs() -> pd.DataFrame:
    """A Monday number for every week from 18 Jan 2016: between Rs 1,500 and Rs 4,500."""
    day = pd.date_range("2016-01-18", "2023-12-25", freq="W-MON")
    need = 3_000.0 + 1_500.0 * np.sin(np.arange(len(day)) / 20.0)
    return pd.DataFrame({"day": day, "known_at": day + pd.Timedelta(hours=1, minutes=10), "adr5": need / 140.0,
                         "adr10": need / 140.0, "adr_usd": need / 140.0, "usd_inr": 70.0, "need_inr": need})


def made_up_hedged() -> tuple[list[dict], pd.DataFrame]:
    """Two hedged trades on two hours of 1-minute bars (Monday 7 Jan 2019): one dips a little, one dips far below
    the money needed for 0.01 lot before it ends in profit."""
    idx = pd.DatetimeIndex(pd.date_range("2019-01-07 10:00", periods=120, freq="1min"),
                           name="server_time").as_unit("ns")
    low, high = np.full(120, 1299.5), np.full(120, 1300.5)
    low[40], low[80] = 1285.0, 1200.0                                            # 10:40 and 11:20
    m1 = pd.DataFrame({"low": low, "high": high}, index=idx)
    r = costs.usd_inr("2019-01-07")
    at = lambda m: str(idx[m])                                                    # noqa: E731

    def leg(direction, a, open_px, b, close_px):
        d = 1 if direction == "buy" else -1
        return {"direction": direction, "opened": at(a), "open_px": open_px, "closed": at(b), "close_px": close_px,
                "net_inr": d * (close_px - open_px) * r - 40}

    one = [leg("buy", 0, 1300.0, 50, 1301.0), leg("sell", 10, 1295.0, 30, 1290.0)]
    two = [leg("buy", 60, 1300.0, 90, 1302.0)]
    stories = [{"signal": f"hedged-{k}", "status": "closed", "legs": legs, "net_inr": sum(x["net_inr"] for x in legs)}
               for k, legs in enumerate((one, two))]
    return stories, m1


def info_for(frame: pd.DataFrame) -> dict:
    seg = {}
    for label, part in frame.groupby("segment"):
        seg[label] = {"trades": int(len(part)), "net_inr": round(float(part["net_inr"].sum()), 2), "open_at_end": 0,
                      "recorded_trial": "made-up"}
    return {"segments": seg}


@pytest.fixture(scope="module")
def built():
    frame = made_up_trades()
    stories, m1 = made_up_hedged()
    return R.build(frame, info_for(frame), made_up_needs(), stories, m1, n_runs=60, log=QUIET,
                   data={"prices": "made-up trades (tests/test_money_report.py)"})


def test_the_made_up_trades_are_like_the_real_ones(built):
    """Guard on the fixture itself: the price moves are a little positive, the Rs 40 turns them into a loss, and the
    plan loses more than 0.01 lot always - the case the report's words were written for."""
    fx, v1 = built["facts"], built["rows"][0]
    assert fx["flat_gross"] > 0 > fx["flat_total"]
    assert v1["head"]["result"] < fx["flat_total"] < 0
    assert fx["trades"] == 700 and fx["dropped"] == 0
    assert len(built["rows"]) == 10 and built["n_runs"] == 60


def test_the_report_is_written_in_easy_english(built, tmp_path):
    R.publish(built, md_path=tmp_path / "money-plan.md", folder=tmp_path / "money-plan", record=True, log=QUIET)
    md = (tmp_path / "money-plan.md").read_text(encoding="utf-8")
    assert check_words(md) == []                                                # no code words, no "R", no p-value
    for must in ("# Your money plan, measured (L20)", "PLAIN TRADES — NOT YOUR HEDGING", "**Already seen:**",
                 "## The answer", "## Year by year, the real order", "## The same months in 60 other orders",
                 "## One change at a time (shown, not recommended)", "## Two ways to read this",
                 "The decision is yours.", "## What would make this wrong", "**Tries:** 10 settings",
                 "**Your hedging, measured apart (2 hedged trades", "More often it blew up first",
                 "orders ended above the start. The best one ended at", "each time) |",
                 "![The account in the real order](money-plan/account.jpg)",
                 "![Where your money ended in each order](money-plan/orders.jpg)"):
        assert must in md, must
    for year in range(2016, 2024):                                              # every year has its row
        assert f"| {year} |" in md
    assert not re.search(r"(?<!\d)(2024|2025)(?!\d)", md)                       # the sealed years are never named
    assert md.count("| row ") == 9 and "| **the plan** |" in md
    h = built["rows"][0]["head"]
    assert R.inr(h["your_money_end"]) in md and R.inr(h["out"]) in md and R.inr(h["added"]) in md
    trial = built["trial"]
    assert f"`{trial}`" in md


def test_the_pictures_are_real_jpegs_with_their_words_kept(built, tmp_path):
    folder = tmp_path / "pics"
    a = R.account_picture(built, folder / "account.jpg")
    o = R.orders_picture(built, folder / "orders.jpg")
    for pic, name in ((a, "account.jpg"), (o, "orders.jpg")):
        with Image.open(folder / name) as im:
            assert im.format == "JPEG" and im.size[0] > 1200                    # a real JPEG (#106), wide enough
        assert all(check_words(t) == [] for t in pic["texts"]) and len(pic["pixel_sha256"]) == 64
    words = " | ".join(a["texts"])
    for named in ("the money in the account", "a double", "a blow-up", "money needed", "the full account"):
        assert named in words                                                   # each is drawn (the legend / margin)
    words = " | ".join(o["texts"])
    for named in ("the middle order", "the real order", "0.01 lot always", "the start", "grey bars"):
        assert named in words
    assert "NOT your hedging" in " ".join(a["texts"]) and "NOT your hedging" in " ".join(o["texts"])


def test_the_tables_hold_every_trade_and_every_event(built, tmp_path):
    folder = tmp_path / "money-plan"
    R.publish(built, md_path=tmp_path / "m.md", folder=folder, record=False, log=QUIET)
    one = built["rows"][0]["one"]
    trades = pd.read_csv(folder / "trades.csv")
    assert len(trades) == int(one.taken[0])
    assert trades["inr_at_that_lot"].sum() == pytest.approx(one.result[0], abs=5)      # the file keeps paise
    steps = trades["lots"] * 100
    assert np.allclose(steps, steps.round()) and steps.min() >= 1               # whole 0.01 steps (#116)
    events = pd.read_csv(folder / "events.csv")
    assert (events["what"] == "blow-up").sum() == int(one.blowups.sum())
    assert events.loc[events["what"] == "blow-up", "money_in_out"].sum() == pytest.approx(one.added[0], abs=1)
    assert -events.loc[events["what"] == "a double", "money_in_out"].sum() == pytest.approx(one.out[0], abs=1)
    lots = pd.read_csv(folder / "lots.csv")                                     # counts at each lot, never an average
    assert lots["trades"].sum() == len(trades) and set(lots["year"]) <= set(range(2016, 2024))
    assert lots.groupby("lot")["trades"].sum().to_dict() == trades["lots"].round(2).value_counts().to_dict()
    assert len(pd.read_csv(folder / "mondays.csv")) == len(built["needs"])
    dips = pd.read_csv(folder / "hedged-dips.csv")
    assert len(dips) == 2 and (dips["low_in_needs"] >= 1).sum() == 1            # the second one dipped past the line


def test_one_trial_file_is_recorded(built, tmp_path):
    R.publish(built, md_path=tmp_path / "m.md", folder=tmp_path / "money-plan", record=True, log=QUIET)
    files = sorted(config.TRIALS_DIR.glob("*.json"))
    assert len(files) == 1 and files[0].name == built["trial"]
    t = json.loads(files[0].read_text(encoding="utf-8"))
    assert (t["kind"], t["idea_id"], t["version_id"]) == ("money-plan", MP.IDEA, MP.VERSION)
    s, r = t["settings"], t["result"]
    assert s["trades"]["label"] == "PLAIN TRADES — NOT HIS HEDGING"
    assert s["version_1"]["your_money"] == 200_000 and s["version_1"]["account_share"] == 0.1
    assert len(s["extra_rows"]) == 9 and s["reshuffles"] == {"runs": 60, "seed": MP.SEED, "months": 96,
                                                            "blocks": "calendar months, each used once"}
    assert set(s["already_seen"]) == {"account_doubles_7_8_days", "capital_adr_sizing"}
    assert len(r["rows"]) == 9 and len(r["version_1"]["real_order"]["years"]) == 8
    assert r["hedged_dips"]["trades"] == 2 and r["hedged_dips"]["at_or_past_1x_the_money_needed"] == 1
    assert r["version_1"]["real_order"]["result"] == pytest.approx(built["rows"][0]["head"]["result"], abs=0.01)
    R.publish(built, md_path=tmp_path / "m2.md", folder=tmp_path / "mp2", record=False, log=QUIET)
    assert len(list(config.TRIALS_DIR.glob("*.json"))) == 1                     # record=False writes none


def test_the_two_readings_refuse_trades_they_were_not_written_for(tmp_path):
    """The report's 'two ways to read this' was written for trades that LOSE after cost. With trades that make money
    the sentences would be false, so the report refuses and says which check failed - it never prints them anyway."""
    frame = made_up_trades(seed=6, win_gross=(900.0, 1_800.0))                   # wins big enough to pay the cost
    stories, m1 = made_up_hedged()
    out = R.build(frame, info_for(frame), made_up_needs(), stories, m1, n_runs=40, log=QUIET)
    assert out["facts"]["flat_total"] > 0
    with pytest.raises(R.MoneyReportError, match="two ways to read this"):
        R.publish(out, md_path=tmp_path / "m.md", folder=tmp_path / "money-plan", record=False, log=QUIET)
    assert not (tmp_path / "m.md").exists()


def test_no_hedged_trades_leaves_that_part_out(tmp_path):
    frame = made_up_trades()
    _, m1 = made_up_hedged()
    out = R.build(frame, info_for(frame), made_up_needs(), [], m1, n_runs=40, log=QUIET)
    R.publish(out, md_path=tmp_path / "m.md", folder=tmp_path / "money-plan", record=False, log=QUIET)
    md = (tmp_path / "m.md").read_text(encoding="utf-8")
    assert "## Your hedging" not in md and "For your hedging" not in md
    assert not (tmp_path / "money-plan" / "hedged-dips.csv").exists()


def test_small_helpers():
    assert R.signed(1234.4) == "+₹1,234" and R.signed(-1234.4) == "-₹1,234" and R.signed(0.2) == "₹0"
    assert R.pc(0.0342) == "+3.4%" and R.pc(-0.15) == "-15.0%"
    assert R.days_list([]) == "none" and R.days_list([13.9]) == "14 days"
    assert R.days_list([27.6, 13.9, 97.9, 18.1]) == "14, 18, 28 and 98 days"
    assert R.num(float("nan")) == "none" and R.num(41.4) == "41"
    assert R._lot_range(np.array([0, 3, 0, 2])) == "0.01 to 0.03" and R._lot_range(np.array([0, 0, 5])) == "0.02"
    assert R._years_text([2019, 2017, 2018]) == "2017–2019" and R._years_text([2019]) == "2019"
    assert R._years_text([2016, 2018]) == "2016 and 2018" and R._years_text([2016, 2018, 2021]) == "2016, 2018 and 2021"
    # the years the rule buys the most lots in are called "chosen on" only when they all are tune years
    assert R._quiet_years({"quiet_years": [2017, 2018, 2019]}) == ("2017–2019",
                                                                   " years this version of idea #1 was chosen on")
    assert R._quiet_years({"quiet_years": [2019, 2020, 2021]}) == ("2019–2021", "")
