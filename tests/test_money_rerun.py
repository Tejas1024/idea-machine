"""L20, the second run: Type 1's real order made again after each stop-out (machine/money/rerun.py, L20-notes 3c
point 7).

Made-up trades only, with hand arithmetic in each docstring. The hedging tester is replaced by `Universe.again`: the
same one-trade-at-a-time rule as hedging.replay (a signal known while a trade is on is skipped, M18) on four made-up
signals, so the loop can be checked by hand. The real set-up (rerun.Type1Hedging) is checked on L19b's trades by the
run itself: its replay from the start must equal L19b's recorded stories, leg by leg, or nothing runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from machine import costs
from machine.money import hedged as H
from machine.money import plan as MP
from machine.money import rerun as RR

MON = pd.Timestamp("2019-01-07 10:00")
NEED = 500.0                                       # money needed for 0.01 lot: Rs 20,000 buys 40 steps


def minute(m: int) -> pd.Timestamp:
    return MON + pd.Timedelta(minutes=m)


def rate() -> float:
    return costs.usd_inr("2019-01-07")


def needs() -> pd.DataFrame:
    return pd.DataFrame({"day": [pd.Timestamp("2019-01-07")], "known_at": [pd.Timestamp("2019-01-07 01:05")],
                         "adr_usd": [3.0], "need_inr": [NEED]})


def book(spike_b: bool = False) -> MP.MinuteBook:
    """400 minutes at bid low 2000.00 / high 2000.40; minute 50: low 1993.00 (trade A's dip); with spike_b, minute 80:
    high 2007.00 (trade B's dip)."""
    lo, hi = np.full(400, 2000.0), np.full(400, 2000.4)
    lo[50] = 1993.0
    if spike_b:
        hi[80] = 2007.0
    idx = pd.DatetimeIndex(pd.date_range(MON, periods=400, freq="1min"), name="server_time")
    return MP.MinuteBook.build(pd.DataFrame({"low": lo, "high": hi}, index=idx))


def one_leg(sig: str, d: str, a: int, b: int, close_px: float) -> dict:
    r = rate()
    net = ((close_px - 2000.0) if d == "buy" else (2000.0 - close_px)) * r - 40
    return {"signal": sig, "status": "closed", "net_inr": net,
            "legs": [{"direction": d, "opened": str(minute(a)), "open_px": 2000.0, "closed": str(minute(b)),
                      "close_px": close_px, "net_inr": net}]}


class Universe:
    """Four made-up signals (known at, its trade): A known 0, a buy 0-300; B known 55, a sell 60-100; C known 200, a buy
    210-250; D known 320, a sell 330-350. again(bar) takes them one at a time from the signals known AFTER bar - as the
    hedging tester does when the trade it held ended at bar."""

    def __init__(self, d_close: float = 1999.0):
        self.trades = [(0, one_leg("A", "buy", 0, 300, 2001.0)), (55, one_leg("B", "sell", 60, 100, 1999.0)),
                       (200, one_leg("C", "buy", 210, 250, 2002.0)), (320, one_leg("D", "sell", 330, 350, d_close))]
        self.asked: list[pd.Timestamp] = []

    def take(self, after: int) -> list[dict]:
        out, free = [], after
        for known, s in self.trades:
            if known <= after or known < free:
                continue
            out.append(s)
            free = int((pd.Timestamp(s["legs"][0]["closed"]) - MON) / pd.Timedelta(minutes=1))
        return out

    def first(self) -> list[dict]:
        return self.take(-1)

    def again(self, bar: pd.Timestamp) -> tuple[list[dict], dict]:
        self.asked.append(pd.Timestamp(bar))
        got = self.take(int((pd.Timestamp(bar) - MON) / pd.Timedelta(minutes=1)))
        return got, {s["signal"]: "made up" for s in got}


def run(u: Universe, spike_b: bool = False) -> dict:
    return RR.made_again(H.stream("type1"), book(spike_b), needs(), u.first(), u.again, log=lambda *a: None)


def test_the_first_list_is_one_trade_at_a_time():
    """A runs 0-300, so B (55) and C (200) are skipped; D (320) comes after A."""
    assert [s["signal"] for s in Universe().first()] == ["A", "D"]


def test_a_stop_out_is_made_again_from_that_minute():
    """Rs 20,000 / Rs 500 = 40 steps. A at minute 50: (1993 - 2000) r - 40 a step, x 40 = -280 r - 1,600 (about
    -Rs 21,400): the account reaches Rs 0 -> the broker closes A at minute 50. Made again from minute 50: B (known 55),
    C (200, after B's close at 100), D (320). Nothing else is closed by the broker: one round."""
    u = Universe()
    res = run(u)
    assert u.asked == [minute(50)]
    assert [s["signal"] for s in res["stories"]] == ["A", "B", "C", "D"]
    assert res["cuts"] == [("A", minute(50))]
    r0 = res["rounds"][0]
    assert (r0["signal"], r0["stopped_at"], r0["trades_left_out"], r0["trades_made_again"], r0["the_same_trades"]) == \
        ("A", minute(50), 1, 3, 1)
    assert res["new_signals"] == ["B", "C"]
    tr = res["trace"]
    assert tr["signal"].tolist() == ["A", "B", "C", "D"]
    assert tr["stopped_out"].tolist() == [True, False, False, False]
    h = res["head"]
    assert (h["stopouts"], h["blowups"], h["added"]) == (1, 1, 20_000.0)


def test_the_money_after_a_stop_out_by_hand():
    """After A: the account Rs 0, refilled to Rs 20,000 from the reserve (Rs 160,000 left). B: 40 steps x (r - 40);
    C: the account / 500 rounded down, x (2 r - 40); D: again, x (r - 40). Your money = Rs 200,000 - Rs 20,000 + all
    three."""
    r = rate()
    res = run(Universe())
    acc, lots = 20_000.0, [0.40]                       # A: 40 steps, then the broker closes it
    for gain in (r - 40, 2 * r - 40, r - 40):
        steps = np.floor(acc / NEED + 1e-9)
        lots.append(steps / 100)
        acc += steps * gain
    assert res["one"].account[0] == pytest.approx(acc)
    assert res["one"].reserve[0] == pytest.approx(160_000.0)
    assert res["head"]["result"] == pytest.approx(-20_000.0 + acc - 20_000.0)
    assert res["trace"]["lots"].tolist() == pytest.approx(lots)


def test_two_stop_outs_are_two_rounds():
    """With B's spike (high 2007.00 at minute 80): B, a sell at 2000.00 with 40 steps, is at -7 r - 40 a step there ->
    the broker closes B at minute 80 too. Round 2 makes the trades again from minute 80: C and D - the same as the
    list's (the paths meet again)."""
    u = Universe()
    res = run(u, spike_b=True)
    assert u.asked == [minute(50), minute(80)]
    assert res["cuts"] == [("A", minute(50)), ("B", minute(80))]
    assert [r["the_same_trades"] for r in res["rounds"]] == [1, 2]
    assert res["trace"]["stopped_out"].tolist() == [True, True, False, False]
    assert (res["head"]["stopouts"], res["head"]["blowups"]) == (2, 2)
    assert len(res["heads"]) == 3                       # the list as given, then after each round


def test_no_stop_out_asks_for_nothing():
    """With Rs 2,000 needed for 0.01 lot the account trades 10 steps: A's dip -7 r - 40 a step x 10 is about
    -Rs 5,350 - no stop-out, nothing is made again."""
    u = Universe()
    n = needs().assign(need_inr=2_000.0)
    res = RR.made_again(H.stream("type1"), book(), n, u.first(), u.again, log=lambda *a: None)
    assert u.asked == [] and res["rounds"] == [] and res["cuts"] == []
    assert res["heads"][0] == res["head"]


def test_the_first_round_is_the_plan_on_the_list_as_given():
    u = Universe()
    res = run(u)
    got = H.stream_trades(H.stream("type1"), book(), needs(), stories=u.first(), recorded={}, ended={})
    ts = MP.TradeSet.build(got["frame"], last_month=H.LAST_MONTH, inside=got["inside"])
    alone = MP.headline(MP.simulate(ts, MP.real_order(ts), MP.VERSION_1, trace=True))
    assert res["first_head"].keys() == alone.keys()
    for k, v in alone.items():                         # "days per double" is nan in both: no double
        w = res["first_head"][k]
        assert w == v or (isinstance(v, float) and np.isnan(v) and np.isnan(w)), k


def test_a_cut_trade_ends_at_its_stop_out_minute_by_hand():
    """A cut at minute 50: closed then; its money then -7 r - 40 (its lowest point too); before the cost -7 r; one
    position, Rs 40; 50 minutes long."""
    r = rate()
    u = Universe()
    got = H.stream_trades(H.stream("type1"), book(), needs(), stories=u.first(), recorded={}, ended={})
    c = RR.cut(got, "A", minute(50))
    a = c["frame"].set_index("signal").loc["A"]
    assert a["close"] == minute(50) and a["closed_by"] == RR.STOPPED_BY
    assert a["net_inr"] == pytest.approx(-7 * r - 40) and a["low_inr"] == pytest.approx(-7 * r - 40)
    assert a["low0_inr"] == pytest.approx(-7 * r) and a["cost_inr"] == 40 and a["positions"] == 1
    assert a["days"] == pytest.approx(50 / 1440) and a["gross_inr"] == pytest.approx(-7 * r)
    assert c["inside"]["A"].t[-1] == pytest.approx(50 / 1440)
    assert got["frame"].set_index("signal").loc["A", "close"] == minute(300)          # the input is not changed


def test_a_cut_refuses_a_minute_that_is_not_a_new_low():
    """The account can only reach Rs 0 at a new low: minute 49 is not one (A sits at -Rs 40 from minute 0)."""
    u = Universe()
    got = H.stream_trades(H.stream("type1"), book(), needs(), stories=u.first(), recorded={}, ended={})
    with pytest.raises(AssertionError, match="not one of its new lows"):
        RR.cut(got, "A", minute(49))


def test_a_trade_made_again_must_open_after_the_stop_out():
    u = Universe()

    def wrong(bar):
        return [one_leg("B", "sell", 40, 100, 1999.0)], {}

    with pytest.raises(AssertionError, match="opens at or before"):
        RR.made_again(H.stream("type1"), book(), needs(), u.first(), wrong, log=lambda *a: None)


def test_same_trades_finds_every_difference():
    a = [{"signal": "x", "net_inr": 1.0}, {"signal": "y", "net_inr": 2.0}]
    assert RR.same_trades(a, [dict(x) for x in a]) == []
    assert RR.same_trades(a, [a[0], {"signal": "y", "net_inr": 2.01}]) == [1]
    assert RR.same_trades(a, a[:1]) == [0, 1]


def test_only_type1_is_made_again():
    with pytest.raises(ValueError, match="only for Type 1"):
        RR.Type1Hedging(H.stream("zone-break"))


def test_stop_bar_is_on_the_minute():
    assert RR.stop_bar(minute(0), 50 / 1440 + 1e-12) == minute(50)


def test_after_the_reserve_is_empty_no_trade_is_counted():
    """Your money Rs 40,000, half of it in the account: one refill of Rs 20,000 empties the reserve. The broker closes
    A (minute 50) and B (its spike, minute 80): after B nothing can refill the account - the plan stops, and C and D
    (found again) are not taken. The report counts only the trades taken."""
    small = MP.Plan(name="made up", change="your money Rs 40,000, the account Rs 20,000", your_money=40_000.0,
                    account_share=0.5)
    u = Universe()
    res = RR.made_again(H.stream("type1"), book(spike_b=True), needs(), u.first(), u.again, plan=small,
                        log=lambda *a: None)
    g = H.again_facts(res)
    assert (g["trades"], g["new"], g["not_taken"]) == (2, 1, 2)
    assert res["head"]["reserve_used_up"] == "Jan 2019"
    assert H._each(g["refills"]) == " (nothing at the last one: the reserve was empty)"


def test_each_refill_in_words():
    assert H._each([20_000.0]) == ""
    assert H._each([20_000.0, 20_000.0]) == " (₹20,000 each time)"
    assert H._each([20_000.0, 14_000.0]) == " (₹14,000 to ₹20,000 each time)"
    assert H._each([20_000.0, 20_000.0, 0.0]) == " (₹20,000 each time; nothing at the last one: the reserve was empty)"


def test_a_stopped_trade_counts_only_the_positions_it_opened():
    """A with a hedge (a sell at 2000.00) opened at minute 100 - after the broker closed A at minute 50: in the plan A
    had 1 position, not 2 (L19b's list counts both)."""
    a = one_leg("A", "buy", 0, 300, 2001.0)
    hedge = {"direction": "sell", "opened": str(minute(100)), "open_px": 2000.0, "closed": str(minute(200)),
             "close_px": 2000.0, "net_inr": -40.0}
    a2 = {**a, "net_inr": a["net_inr"] - 40.0, "legs": a["legs"] + [hedge]}
    got = H.stream_trades(H.stream("type1"), book(), needs(), stories=[a2], recorded={}, ended={})
    tr = H.trace_table(H.run_stream(got, n_runs=5, log=lambda *x: None))
    assert tr["stopped_out"].tolist() == [True]
    assert tr["positions"].tolist() == [2] and tr["positions_in_the_plan"].tolist() == [1]


def test_the_report_shows_the_trades_found_again(tmp_path):
    """The whole report with a stop-out made again: the answer gets its second column, the words pass the easy-English
    check, and the tables of the second column are written. D closes at 2001.00 (a loss) so the trades lose at 0.01
    lot a position (reading 2 fits; L20-notes 3j)."""
    u = Universe(d_close=2001.0)
    first = u.first()
    rec = {"trial": "made-up", "trades": len(first), "net_inr": float(sum(s["net_inr"] for s in first)),
           "open_at_end": 0}
    out = H.build([H.stream("type1")], book(), needs(), n_runs=20, log=lambda *a: None,
                  stories={"type1": (first, rec)}, again=u.again)
    assert out["again"]["rounds"] and u.asked[0] == minute(50)
    md = H.report_md(out)
    assert "the same, with your trades found again after each stop-out" in md and "**The second column.**" in md
    H.publish(out, md_path=tmp_path / "r.md", folder=tmp_path / "f", record=False, log=lambda *a: None)
    for name in ("trades-found-again.csv", "events-found-again.csv", "found-again-rounds.csv",
                 "type1-found-again-stories.jsonl"):
        assert (tmp_path / "f" / name).exists(), name
    t = pd.read_csv(tmp_path / "f" / "trades-found-again.csv")
    assert t["signal"].tolist() == ["A", "B", "C", "D"] and t["found_again"].tolist() == [False, True, True, False]
