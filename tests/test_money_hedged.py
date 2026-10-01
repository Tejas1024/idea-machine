"""L20, the second run — his money plan on his HEDGED trades (machine/money/hedged.py).

Made-up trades for every money test (hand arithmetic in each docstring). The tests marked "recorded" read only L19b's
tracked files (plan/work/L19b, plan/work/trials): every stream must load and equal its trial file.
What the words mean (THE STOP-OUT, row 10): plan/work/L20-notes.md section 3.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from machine import config, costs
from machine.money import hedged as H
from machine.money import plan as MP

MON = pd.Timestamp("2019-01-07 10:00")


def book_of(prices: list[tuple[float, float]], start=MON) -> MP.MinuteBook:
    """1-minute bars from `start`: (bid low, bid high) each."""
    idx = pd.DatetimeIndex(pd.date_range(start, periods=len(prices), freq="1min"), name="server_time")
    return MP.MinuteBook.build(pd.DataFrame({"low": [p[0] for p in prices], "high": [p[1] for p in prices]},
                                            index=idx))


def needs_of(*days, need=500.0) -> pd.DataFrame:
    return pd.DataFrame({"day": [pd.Timestamp(d) for d in days],
                         "known_at": [pd.Timestamp(d) + pd.Timedelta(hours=1, minutes=5) for d in days],
                         "adr_usd": [3.0] * len(days), "need_inr": [need] * len(days)})


def leg(direction, opened, open_px, closed=None, close_px=None, net=None):
    return {"direction": direction, "opened": str(opened), "open_px": open_px, "closed": None if closed is None
            else str(closed), "close_px": close_px, "net_inr": net}


def minute(m: int) -> pd.Timestamp:
    return MON + pd.Timedelta(minutes=m)


def two_trades():
    """A closed hedged trade (minutes 0-6) and one still open at the end (from minute 8). r = January 2019's rate.
      trade 1: a buy at 2000.00 (minute 0), its hedge a sell at 1998.00 (minute 2) closed at 1996.00 (minute 4) for
               2 r - 40; the buy closed at 2001.00 (minute 6) for r - 40. Lows: 1999.5, 1998.2, 1997.5, 1997, 1995.5,
               1996, 2000.5 -> its lowest point: minute 4, the buy alone at 1995.50 + the sell's 2 r - 40:
               -4.5 r - 40 + 2 r - 40 = -2.5 r - 80.
      trade 2: a sell at 2001.00 (minute 8), still open; highs 2001.5, 2003.0 -> its lowest point -2 r - 40 (minute 9);
               on the last minute -2 r - 40."""
    r = costs.usd_inr("2019-01-07")
    lows = [1999.5, 1998.2, 1997.5, 1997.0, 1995.5, 1996.0, 2000.5, 2000.0, 2000.8, 2001.0]
    highs = [2000.5, 1999.9, 1998.4, 1998.0, 1996.5, 1997.0, 2001.5, 2000.6, 2001.5, 2003.0]
    book = book_of(list(zip(lows, highs)))
    t1 = {"signal": "made-up.B1", "status": "closed", "net_inr": 3 * r - 80,
          "legs": [leg("buy", minute(0), 2000.0, minute(6), 2001.0, r - 40),
                   leg("sell", minute(2), 1998.0, minute(4), 1996.0, 2 * r - 40)]}
    t2 = {"signal": "made-up.S2", "status": "open at end", "net_inr": None,
          "legs": [leg("sell", minute(8), 2001.0)]}
    return [t1, t2], book, r


def test_a_stream_is_its_closed_trades_minute_by_minute():
    stories, book, r = two_trades()
    got = H.stream_trades(H.stream("type1"), book, needs_of("2019-01-07"), stories=stories, recorded={})
    f = got["frame"]
    assert f["signal"].tolist() == ["made-up.B1"]                      # the still-open trade is not in the plan (B6)
    row = f.iloc[0]
    assert row["net_inr"] == pytest.approx(3 * r - 80) and row["cost_inr"] == 80.0
    assert row["gross_inr"] == pytest.approx(3 * r) and row["positions"] == 2
    assert row["low_inr"] == pytest.approx(-2.5 * r - 80) and row["low0_inr"] == pytest.approx(-2.5 * r)
    assert row["open"] == minute(0) and row["close"] == minute(6) and row["need_inr"] == 500.0
    ins = got["inside"]["made-up.B1"]
    assert ins.v[-1] == row["low_inr"] and ins.t[-1] == pytest.approx(4 / 1440)
    s = got["still"].iloc[0]
    assert s["signal"] == "made-up.S2" and s["inr_at_the_end"] == pytest.approx(-2 * r - 40)
    assert s["low_inr"] == pytest.approx(-2 * r - 40) and s["low_when"] == minute(9)


def test_a_trade_before_the_first_monday_number_is_left_out_and_counted():
    stories, book, _ = two_trades()
    got = H.stream_trades(H.stream("type1"), book, needs_of("2019-01-14"), stories=stories, recorded={})
    assert len(got["frame"]) == 0 and got["dropped"] == 1 and got["dropped_signals"] == ["made-up.B1"]


def test_the_plan_runs_on_a_made_up_stream():
    """Rs 20,000, Rs 500 needed -> 40 steps: the lowest point -2.5 r - 80 (about -Rs 290) x 40 = about -Rs 11,600:
    the account stays above Rs 0, no stop-out; the trade ends (3 r - 80) x 40."""
    stories, book, r = two_trades()
    got = H.stream_trades(H.stream("type1"), book, needs_of("2019-01-07"), stories=stories, recorded={})
    res = H.run_stream(got, n_runs=5, log=lambda *a: None)
    assert [p.name for p in (x["plan"] for x in res["rows"])] == [MP.VERSION_1.name] + \
        [p.name for p in MP.HEDGED_ROWS]
    one = res["rows"][0]["one"]
    assert one.trace["lots"].tolist() == [0.40] and one.stopouts.sum() == 0
    assert one.trace["inr"].tolist() == pytest.approx([40 * (3 * r - 80)])
    line = H.still_open_line(res)[0]
    acc = float(one.account[0])
    assert line["account"] == pytest.approx(acc)
    assert line["lots"] == pytest.approx(np.floor(acc / 500 + 1e-9) * 0.01)
    only_v1 = H.run_stream({**got, "stream": H.stream("zone-entry")}, n_runs=5, log=lambda *a: None)
    assert len(only_v1["rows"]) == 1                                   # every other stream: version 1 only


def test_a_stream_whose_stories_differ_from_its_trial_file_is_refused(tmp_path, monkeypatch):
    stories, _, _ = two_trades()
    st = H.Stream("x", "entry-x", "v-x", "20260101-000000-000000-000000", "made up")
    (tmp_path / "L19b" / "entry-x").mkdir(parents=True)
    (tmp_path / "trials").mkdir()
    st_path = tmp_path / "L19b" / "entry-x" / "v-x-hedged-stories.jsonl"
    st_path.write_text("\n".join(json.dumps(s) for s in stories) + "\n", encoding="utf-8")
    monkeypatch.setattr(H, "L19B", tmp_path / "L19b")
    monkeypatch.setattr(H, "TRIALS", tmp_path / "trials")
    trial = {"idea_id": "entry-x", "version_id": "v-x",
             "result": {"his_hedging": {"trades": 1, "net_inr_total": round(stories[0]["net_inr"], 2)}}}
    (tmp_path / "trials" / f"{st.trial}-entry-x-v-x.json").write_text(json.dumps(trial), encoding="utf-8")
    got, rec = H.load(st)
    assert len(got) == 2 and rec["open_at_end"] == 1 and rec["trades"] == 1
    trial["result"]["his_hedging"]["net_inr_total"] += 0.02                  # two paise off
    (tmp_path / "trials" / f"{st.trial}-entry-x-v-x.json").write_text(json.dumps(trial), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stop and find why"):
        H.load(st)


def _made_up_report_input(nets=(-300.0, 250.0, -500.0), need=2_000.0):
    """Three one-position trades over three days of made-up minutes (a random walk around $2,000)."""
    n = 3 * 1440
    rng = np.random.default_rng(1)
    mid = 2000 + np.cumsum(rng.normal(0, 0.3, n))
    book = book_of(list(zip(mid - 0.2, mid + 0.2)))

    def one_leg(sig, d, a, b, net):
        return {"signal": sig, "status": "closed", "net_inr": net,
                "legs": [leg(d, minute(a), float(mid[a]), minute(b), float(mid[b]), net)]}

    stories = [one_leg("made-up.B1", "buy", 0, 500, nets[0]), one_leg("made-up.S2", "sell", 600, 1500, nets[1]),
               one_leg("made-up.B3", "buy", 1600, 4000, nets[2])]
    rec = {"trial": "made-up", "trades": 3, "net_inr": float(sum(nets)), "open_at_end": 0}
    return book, needs_of("2019-01-07", need=need), {"type1": (stories, rec)}


def test_the_report_on_made_up_trades(tmp_path):
    """The whole report on made-up trades: its words pass the easy-English check (the pictures' words too), the
    pictures and tables are written, and the two readings say which one the numbers fit (these lose at 0.01 lot a
    position: reading 2)."""
    book, needs, stories = _made_up_report_input()
    out = H.build([H.stream("type1")], book, needs, n_runs=20, log=lambda *a: None, stories=stories)
    md = H.report_md(out)
    assert "## Two ways to read this" in md and "The decision is yours." in md
    assert "**The numbers fit reading 2.**" in md or "**The numbers fit both.**" in md
    assert "YOUR HEDGING" in md and "Already seen" in md
    H.publish(out, md_path=tmp_path / "r.md", folder=tmp_path / "f", record=False, log=lambda *a: None)
    assert "## Inside a hedged trade" in (tmp_path / "r.md").read_text(encoding="utf-8")
    for name in ("account.jpg", "orders.jpg", "trades.csv", "events.csv", "lots.csv", "mondays.csv", "dips.csv",
                 "streams.csv"):
        assert (tmp_path / "f" / name).exists(), name
    t = pd.read_csv(tmp_path / "f" / "trades.csv")
    assert len(t) == 3 and t["inr_at_0.01_lot"].round(2).tolist() == [-300.0, 250.0, -500.0]


def test_the_two_readings_refuse_when_the_numbers_fit_neither():
    """Trades that make money at 0.01 lot a position and never reach Rs 0 fit neither reading (L20-notes 3j): the
    report refuses instead of printing a reading the numbers do not support."""
    from machine.money.report import MoneyReportError
    book, needs, stories = _made_up_report_input(nets=(300.0, 250.0, 500.0), need=200_000.0)
    out = H.build([H.stream("type1")], book, needs, n_runs=20, log=lambda *a: None, stories=stories)
    with pytest.raises(MoneyReportError, match="neither reading"):
        H.report_md(out)


# ================================================================================================================
# the recorded streams (L19b's tracked files; no price data needed)
# ================================================================================================================

def test_every_l19b_stream_loads_and_equals_its_trial_file():
    """L20-notes 3a: the table of streams, each equal to its L19b trial file to the paisa."""
    want = {"type1": (15, 1), "zone-break": (10, 1), "zone-entry": (20, 1), "p1-a": (9, 1), "p1-b": (4, 0),
            "p1-c": (1, 0), "touch-a": (3, 0), "touch-b": (2, 0)}
    assert [s.key for s in H.STREAMS] == list(want)
    for st in H.STREAMS:
        stories, rec = H.load(st)
        assert (rec["trades"], rec["open_at_end"]) == want[st.key], st.key
        assert all(len(s["legs"]) >= 1 for s in stories)


def test_the_second_gap_made_the_same_trades_as_type1():
    """L19b-notes point 4: both readings of the second gap took Type 1's 16 trades to the rupee - not a stream."""
    t1 = [(s["signal"].split(".")[-1], s["status"], s["net_inr"]) for s in H.read_stories(H.stream("type1").stories)]
    for entry, version, trial in H.SAME_AS_TYPE1:
        other = H.read_stories(H.L19B / entry / f"{version}-hedged-stories.jsonl")
        assert [(s["signal"].split(".")[-1], s["status"], s["net_inr"]) for s in other] == t1
        assert (H.TRIALS / f"{trial}-{entry}-{version}.json").exists()


def test_the_streams_are_his_cost_and_his_window():
    for st in H.STREAMS:
        stories, _ = H.load(st)
        for s in stories:
            for x in s["legs"]:
                assert pd.Timestamp(x["opened"]) < pd.Timestamp("2021-01-01")          # 2016-2020 only
                if x["closed"] and x["net_inr"] is not None:
                    assert x["net_inr"] < 1e6
    assert config.COST_INR_PER_POSITION == 40
