"""The live alerts (L17a): this week's calendar, the lock on 2024+ news numbers and live prices, the surprise, the
windows, the outbox, STOP; the live prices (closed bars, the tape, the daily check), the big-move rule (no future, his
ADR, the news near it, the quiet hours), gold now / since the number, the situation; the rehearsals on real weeks."""
from __future__ import annotations

import datetime as dt
import json
import re

import numpy as np
import pandas as pd
import pytest

from machine import config, guards
from machine.host import prices as HP
from machine.live import alerts as LA
from machine.live import calendar as LC
from machine.live import check as LK
from machine.live import move as LM
from machine.live import opened as LO
from machine.live import prices as LP
from machine.live import run as LR
from machine.live import situation as LSit
from machine.live import surprise as LS
from machine.live import tasks as LT
from machine.news import similar

UTC = dt.timezone.utc
REAL_LIVE = LP.live


def row(i, name, when_utc, colour="red", actual="", forecast="", previous="", masked=False):
    cls = {"red": "icon--ff-impact-red", "orange": "icon--ff-impact-ora", "yellow": "icon--ff-impact-yel"}[colour]
    return dict(id=i, ebaseId=i, name=name, prefixedName=f"US {name}", impactName=colour, impactClass=cls,
                dateline=int(pd.Timestamp(when_utc, tz="UTC").timestamp()), timeLabel="", actual=actual,
                forecast=forecast, previous=previous, timeMasked=masked, revision="", actualBetterWorse=0,
                revisionBetterWorse=0, notice="", date="")


NFP_T = "2023-10-06 12:30"
ROWS = [row(1, "Non-Farm Employment Change", NFP_T, "red", "336K", "171K", "187K"),
        row(2, "Unemployment Rate", NFP_T, "red", "3.8%", "3.7%", "3.8%"),
        row(3, "ISM Services PMI", "2023-10-04 14:00", "orange", "53.6", "53.5", "54.5"),
        row(4, "Crude Oil Inventories", "2023-10-04 14:30", "yellow", "-2.2M", "-0.2M", "-2.2M"),
        row(5, "Bank Holiday", "2023-10-09 00:00", "red", masked=True),
        row(6, "A Speech Nobody Knows", "2023-10-05 16:00", "orange")]


@pytest.fixture(autouse=True)
def _fresh_lock_memory(monkeypatch):
    monkeypatch.setattr(LO, "_opened", set())


@pytest.fixture(autouse=True)
def _no_real_mt5(monkeypatch):
    """No test reaches the real MT5 by accident: a pass with no feed of its own gets no prices (as when MT5 is off)."""
    monkeypatch.setattr(LP, "live", lambda now=None, **kw: (None, "no MT5 in tests"))


# ===================================================================================================== the calendar

def test_the_week_is_known_by_its_sunday():
    assert LC.sunday_of(dt.datetime(2026, 10, 5, 6, 0, tzinfo=UTC)) == dt.date(2026, 10, 4)      # a Monday
    assert LC.sunday_of(dt.datetime(2026, 10, 4, 6, 0, tzinfo=UTC)) == dt.date(2026, 10, 4)      # the Sunday itself
    assert LC.sunday_of(dt.datetime(2026, 10, 10, 23, 0, tzinfo=UTC)) == dt.date(2026, 10, 4)    # Saturday night


def test_his_news_only_with_its_report_and_the_names_forex_factory_used():
    ev = LC.events(ROWS)
    assert set(ev["name"]) == {"Non-Farm Employment Change", "Unemployment Rate", "ISM Services PMI",
                               "A Speech Nobody Knows"}                # no yellow row, no row with no exact time
    nfp = ev[ev["name"] == "Non-Farm Employment Change"].iloc[0]
    assert (nfp["report"], nfp["rank"], nfp["method"], nfp["sign"]) == ("Jobs report", 1, "forecast", 1)
    assert ev[ev["name"] == "ISM Services PMI"].iloc[0]["report"] == "ISM services"    # signs.csv "a|b" names
    odd = ev[ev["name"] == "A Speech Nobody Knows"].iloc[0]
    assert (odd["report"], odd["method"]) == ("A Speech Nobody Knows", "flagged")
    assert nfp["ist"].strftime("%H:%M") == "18:00" and nfp["server"] == pd.Timestamp("2023-10-06 15:30")


def test_one_release_per_minute_and_report_with_the_headline_number():
    rels = LC.releases(LC.events(ROWS))
    jobs = [r for r in rels if r["report"] == "Jobs report"]
    assert len(jobs) == 1 and jobs[0]["head"]["name"] == "Non-Farm Employment Change"     # rank 1, not the rate
    assert len(jobs[0]["numbers"]) == 2 and jobs[0]["colour"] == "red" and LC.out_now(jobs[0])


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_a_2026_week_opens_the_lock_once_per_run_and_logs_it():
    week = [row(9, "ISM Services PMI", "2026-10-05 14:00", "orange", "", "55.1", "55.4")]
    with pytest.raises(guards.SealedYearsError):
        guards.check_dates(pd.DatetimeIndex([pd.Timestamp("2026-10-05 14:00", tz="UTC")]))   # shut without a use
    LC._open(week, dt.date(2026, 10, 4))
    LC._open(week, dt.date(2026, 10, 4))
    LO._opened.clear()                                                      # the next pass: a new program
    LC._open(week, dt.date(2026, 10, 4))
    lines = config.SEALED_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "use=news-values" in lines[0] and "live news alerts" in lines[0]   # once per week


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_a_no_in_t14_stops_the_alerts_even_after_the_week_was_opened(monkeypatch):
    week = [row(9, "ISM Services PMI", "2026-10-05 14:00", "orange", "", "55.1", "55.4")]
    LC._open(week, dt.date(2026, 10, 4))
    LO._opened.clear()
    monkeypatch.setattr(guards, "allowed_uses", lambda path=None: {"news-values": False})
    with pytest.raises(guards.SealedYearsError, match="no longer says"):
        LC._open(week, dt.date(2026, 10, 4))


def _page(rows) -> str:
    return "<script>\n  days: " + json.dumps([{"events": rows}]) + ",\n</script>"


def test_the_week_is_saved_reused_for_an_hour_and_kept_when_a_fetch_fails():
    calls = []

    def get(url):
        calls.append(url)
        return _page(ROWS)

    t0 = dt.datetime(2023, 10, 2, 8, 0, tzinfo=UTC)
    rows, fetched = LC.week(t0, get=get)
    assert len(rows) == len(ROWS) and fetched == t0 and calls == ["https://www.forexfactory.com/calendar?week=oct1.2023"]
    assert LC.week_file(dt.date(2023, 10, 1)).is_file()
    LC.week(t0 + dt.timedelta(minutes=59), get=get)
    assert len(calls) == 1                                                  # younger than an hour: the saved copy

    def broken(url):
        raise ConnectionError("blocked")
    rows, fetched = LC.week(t0 + dt.timedelta(minutes=61), get=broken)
    assert fetched == t0 and len(rows) == len(ROWS)                        # a failed fetch keeps the saved copy


# ===================================================================================================== the surprise

def _table(n: int, size: float = 50_000.0) -> pd.DataFrame:
    t = pd.date_range("2021-01-08", periods=n, freq="MS", tz="UTC")
    return pd.DataFrame(dict(event="Nonfarm Payrolls", method="forecast", known_from=t,
                             dollar_surprise=[size * (1 if i % 2 else -1) for i in range(n)]))


def test_the_live_surprise_follows_the_history_rule():
    head = dict(method="forecast", event="Nonfarm Payrolls", sign=1, actual="336K", forecast="171K")
    s = LS.score(head, _table(30))
    assert s["past_n"] == 24 and s["usual_miss"] == pytest.approx(50_000.0)
    assert s["z"] == pytest.approx(165_000 / 50_000) and s["group"] == "big good for the dollar"
    assert LS.score({**head, "actual": "180K"}, _table(30))["group"] == "small"
    assert LS.score({**head, "sign": -1}, _table(30))["group"] == "big bad for the dollar"
    assert LS.score(head, _table(7))["group"] == "not scored"              # fewer than 8 past surprises
    assert "known only tomorrow" in LS.score({**head, "method": "2y yield"}, _table(30))["why"]
    assert LS.score({**head, "actual": ""}, _table(30))["why"] == "the number is not out yet"


# ====================================================================================================== the windows

def _rel(actual="", method="forecast", forecast="171K"):
    rows = [row(1, "Non-Farm Employment Change", NFP_T, "red", actual, forecast, "187K")]
    rel = LC.releases(LC.events(rows))[0]
    rel["head"]["method"] = method
    return rel


def test_before_in_the_last_30_minutes_after_when_the_number_is_out_and_never_late():
    t = dt.datetime(2023, 10, 6, 12, 30, tzinfo=UTC)
    rel, out = _rel(""), _rel("336K")
    m = dt.timedelta(minutes=1)
    assert LR.due(rel, t - 31 * m, set()) == []
    assert LR.due(rel, t - 30 * m, set()) == ["before"] and LR.due(rel, t - m, set()) == ["before"]
    assert LR.due(rel, t, set()) == []                                    # its number is not out yet
    assert LR.due(out, t + m, set()) == ["after"]
    assert LR.due(rel, t + 29 * m, set()) == ["after"]                    # the last minute: without a number
    assert LR.due(out, t + 31 * m, set()) == []                           # too late: never sent late
    assert LR.due(out, t + m, {("after", out["key"])}) == []              # once
    speech = _rel("", method="flagged", forecast="")
    assert LR.due(speech, t - 10 * m, set()) == ["before"] and LR.due(speech, t + m, set()) == []


@pytest.fixture()
def no_lookup(monkeypatch):
    """A lookup with nothing to compare with (the real one needs data/news): the alert's own parts are tested."""
    def empty(tab, q, k=10):
        return similar.Result(q, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, [])
    monkeypatch.setattr(LR.similar, "lookup", empty)


def test_once_queues_each_alert_once_and_nothing_while_he_said_stop(tmp_path, no_lookup):
    box = tmp_path / "outbox.jsonl"
    t = dt.datetime(2023, 10, 6, 12, 10, tzinfo=UTC)
    rows = [row(1, "Non-Farm Employment Change", NFP_T, "red", "", "171K", "187K")]
    done = LR.once(t, rows=rows, outbox=box, tab=pd.DataFrame(), paused=lambda: False)
    assert [d["kind"] for d in done] == ["before"] and done[0]["sent"] == "queued"
    assert LR.once(t + dt.timedelta(minutes=1), rows=rows, outbox=box, tab=pd.DataFrame(),
                   paused=lambda: False) == []                             # once per release
    line = json.loads(box.read_text(encoding="utf-8").splitlines()[0])
    assert line["id"].startswith("live-before-") and "18:00 India time" in line["text"] and "#110" in line["text"]
    after = [row(1, "Non-Farm Employment Change", NFP_T, "red", "336K", "171K", "187K")]
    done = LR.once(t + dt.timedelta(minutes=21), rows=after, outbox=box, tab=pd.DataFrame(), paused=lambda: True)
    assert [d["sent"] for d in done] == ["not sent: STOP"]
    assert len(box.read_text(encoding="utf-8").splitlines()) == 1         # STOP: nothing reaches the outbox
    assert LR.once(t + dt.timedelta(minutes=22), rows=after, outbox=box, tab=pd.DataFrame(),
                   paused=lambda: False) == []                             # and it is not sent late after START
    state = json.loads((config.LIVE_DIR / "state.json").read_text(encoding="utf-8"))
    assert state["releases_this_week"] == 1


def test_a_pass_holds_a_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(LR, "once", lambda **kw: ["ran"])
    assert LR.locked_once() == ["ran"]
    lock = config.LIVE_DIR / "once.lock"
    lock.write_text("someone")
    assert LR.locked_once() is None                                         # another pass holds it
    assert not (config.LIVE_DIR / "sent.jsonl").exists()


def test_the_words_say_when_his_rule_and_the_years():
    rel = _rel("336K")
    q = LA.question(rel, {"group": "big good for the dollar", "z": 1.7})
    assert q.when == "after" and q.has_number and q.number == "336K (forecast 171K)"
    empty = similar.Result(q, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0, [])
    before = LA.before_text(_rel(""), empty)
    assert "18:00 India time" in before and "#110" in before and "2016–2023" in before
    after = LA.after_text(rel, {"group": "big good for the dollar", "z": 1.7}, empty)
    assert "336K (forecast 171K)" in after and "good for the dollar" in after


def test_the_task_runs_once_every_minute():
    s = LT.script()
    assert config.LIVE_TASK_NAME in s and "-m machine.live once" in s and "New-TimeSpan -Minutes 1" in s
    assert "IgnoreNew" in s and "AtLogOn" in s


# ==================================================================================================== the rehearsal

@pytest.mark.data
@pytest.mark.news
def test_a_real_2023_week_replayed_sends_before_and_after_and_never_the_number_too_early():
    if not config.NEWS_FF_DIR.joinpath("2023-10-01.json").is_file():
        pytest.skip("needs data/news/forexfactory/2023-10-01.json (python -m machine.news.forexfactory)")
    done = LR.rehearse(dt.date(2023, 10, 1), prices=False)
    jobs = {d["kind"]: d for d in done if d["report"] == "Jobs report"}
    assert set(jobs) == {"before", "after"}
    assert jobs["before"]["at"].startswith("2023-10-06T12:00") and jobs["after"]["at"].startswith("2023-10-06T12:31")
    assert "336K" not in jobs["before"]["text"] and "336K (forecast 171K)" in jobs["after"]["text"]
    assert "The last 10 times like this (2016–2023)" in jobs["after"]["text"] and "Gold" not in jobs["after"]["text"]
    assert all(d["report"] not in ("Fed speaker", "Fed Chair") or d["kind"] == "before" for d in done)
    lit = {d["kind"]: d["text"] for d in LR.rehearse(dt.date(2023, 10, 1)) if d.get("report") == "Jobs report"}
    assert "Gold now: " in lit["before"] and "Gold since the number: " in lit["after"]     # the week's bars as the feed
    assert "Your bias now: hour " in lit["before"] and "Your nearest levels: above, " in lit["before"]
    assert "The 10 times most like this (2016–2023):" in lit["after"] and "your day and week bias" in lit["after"]
    with pytest.raises(guards.SealedYearsError):
        LR.rehearse(dt.date(2023, 12, 31))                                # a week reaching 2024: refused


# ===================================================================================================== live prices

RATE = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                 ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])
NOW = dt.datetime(2026, 10, 5, 9, 30, 20, tzinfo=UTC)          # a Monday, 12:30:20 on the server (UTC+3 in October)


def rates(first: str, n: int, price: float = 2000.0) -> np.ndarray:
    """n made-up 1-minute bars from server time `first`, each 0.20 from high to low."""
    t0 = int((pd.Timestamp(first) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
    out = np.zeros(n, dtype=RATE)
    for i in range(n):
        out[i] = (t0 + 60 * i, price, price + 0.1, price - 0.1, price, 5, 20, 0)
    return out


class FakeMT5:
    TIMEFRAME_M1 = 1

    def __init__(self, r, server="MetaQuotes-Demo", usd_inr=88.5):
        self.r, self.server, self.usd_inr, self.down = r, server, usd_inr, False

    def initialize(self, **kw):
        return True

    def shutdown(self):
        self.down = True

    def last_error(self):
        return (1, "Success")

    def terminal_info(self):
        return type("T", (), {"connected": True})()

    def account_info(self):
        return type("A", (), {"server": self.server, "company": "MetaQuotes Ltd.", "trade_mode": 0})()

    def symbol_select(self, symbol, on):
        return True

    def symbol_info_tick(self, symbol):
        return type("K", (), {"bid": self.usd_inr})()

    def copy_rates_range(self, symbol, tf, lo, hi):
        return self.r[(self.r["time"] >= lo) & (self.r["time"] <= hi)]


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_live_prices_are_closed_bars_on_the_server_clock_and_open_the_lock_once_a_week():
    assert LP.server_now(NOW) == pd.Timestamp("2026-10-05 12:30:20")
    fake = FakeMT5(rates("2026-10-05 11:30", 61))                    # 11:30 ... 12:30 (12:30 is still forming)
    p, why = REAL_LIVE(NOW, mt5=fake, days=1)
    assert why == "" and fake.down and len(p.m1) == 60 and p.m1.index[-1] == pd.Timestamp("2026-10-05 12:29")
    assert p.m1.index.tz is None and "utc" in p.m1 and p.fresh()
    assert p.m1.index.unit == "ns"                                    # nanoseconds, as load_bars ...
    later = dt.datetime(2026, 10, 5, 9, 30, 20, 123456, tzinfo=UTC)
    p2, _ = REAL_LIVE(later, mt5=FakeMT5(rates("2026-10-05 11:30", 61)), days=1)
    assert LSit.at(p2.m1, p2.at)["p0"] == 2000.0                  # ... so "now" with microseconds can be looked up
    assert p.usd_inr == 88.5 and p.rupees(-10.0) == pytest.approx(885.0)    # 0.01 lot = 1 ounce, at today's rate
    REAL_LIVE(NOW, mt5=FakeMT5(rates("2026-10-05 11:30", 61)), days=1)
    LO._opened.clear()                                                # the next pass: a new program
    REAL_LIVE(NOW, mt5=FakeMT5(rates("2026-10-05 11:30", 61)), days=1)
    lines = config.SEALED_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "use=live-trial" in lines[0] and "live alerts" in lines[0]      # one line a week


def test_another_server_or_a_no_in_t14_switches_the_prices_off_never_the_alerts(monkeypatch):
    p, why = REAL_LIVE(NOW, mt5=FakeMT5(rates("2026-10-05 11:30", 61), server="XMGlobal-MT5 5"), days=1)
    assert p is None and "MetaQuotes-Demo" in why
    monkeypatch.setattr(guards, "allowed_uses", lambda path=None: {"news-values": True, "live-trial": False})
    p, why = LR.prices_now(NOW, lambda now: REAL_LIVE(now, mt5=FakeMT5(rates("2026-10-05 11:30", 61)), days=1))
    assert p is None and "stopped by the lock" in why
    assert not config.SEALED_LOG.exists()                             # nothing was opened


def test_the_tape_keeps_only_the_bars_seen_live_in_l1s_own_text(tmp_path):
    r = rates("2026-10-05 12:00", 31)                                 # 12:00 ... 12:30
    at = pd.Timestamp("2026-10-05 12:30:20")
    closed = r[r["time"] + 60 <= LP._sec(at)]
    assert LP.tape(closed, at, tmp_path) == 5                         # 12:25 ... 12:29: closed in the last 5 minutes
    assert LP.tape(closed, at, tmp_path) == 0                         # never twice
    text = (tmp_path / "2026-10-05.csv").read_text(encoding="ascii").splitlines()
    assert text[0] == HP.HEADER and text[1:] == HP.lines(closed[-5:])
    later = pd.Timestamp("2026-10-05 12:31:20")
    assert LP.tape(r[r["time"] + 60 <= LP._sec(later)], later, tmp_path) == 1     # the next pass adds 12:30


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_the_daily_check_compares_text_and_tells_only_when_prices_differ(tmp_path):
    kept, tape, folder = tmp_path / "kept", tmp_path / "tape", tmp_path / "live"
    kept.mkdir()
    tape.mkdir()
    (kept / "2026-10-05.csv").write_bytes(HP.day_text(rates("2026-10-05 01:00", 6)))        # 01:00 ... 01:05
    seen = rates("2026-10-05 00:59", 5)                               # 00:59 (not in the history) ... 01:03
    seen["close"][3] += 0.5                                           # 01:02: another price
    seen["spread"][4] = 99                                            # 01:03: only another spread
    (tape / "2026-10-05.csv").write_bytes(HP.day_text(seen))
    told = []
    rec, = LK.run(folder=folder, tape_dir=tape, kept_dir=kept, tell=told.append)
    assert (rec["history"], rec["seen"], rec["same"], rec["other"], rec["minor"], rec["missed"], rec["extra"]) == \
        (6, 5, 2, 1, 1, 2, 1)
    assert rec["first_other"] == ["01:02"] and rec["first_missed"] == ["01:04", "01:05"]
    assert len(told) == 1 and "01:02" in told[0] and "2000" not in told[0]    # no price in the words
    assert LK.words(rec).startswith("Mon 05 Oct: 4 of 6 minutes seen live, 1 with OTHER prices")
    (kept / "2026-10-02.csv").write_bytes(HP.day_text(rates("2026-10-02 01:00", 3)))
    assert LK.run(folder=folder, tape_dir=tape, kept_dir=kept, tell=told.append) == []   # once; before the tape: no


MON = dict(day="2026-10-05", history=1320, seen=672, same=672, other=0, minor=0, missed=648, extra=0, first_other=[],
           first_missed=["01:00"])


def test_the_days_report_says_what_the_server_day_did_with_no_price(tmp_path):
    """6 Oct 2026: he cannot keep a Claude chat open for the daily checks - the machine says them itself, once a day."""
    folder, tape, K = config.LIVE_DIR, config.LIVE_TAPE_DIR, "20261005-1400-146608"
    tape.mkdir(parents=True)
    for d in ("2026-10-05", "2026-10-06"):
        (tape / f"{d}.csv").write_text(HP.HEADER + "\n")
    sent = [dict(kind="before", key="x", id="live-before-x", at="2026-10-04T13:30:02+00:00", sent="queued",
                 report="ISM services"),                                       # the server day before
            dict(kind="before", key=K, id=f"live-before-{K}", at="2026-10-05T13:30:02+00:00", sent="queued",
                 report="ISM services"),
            dict(kind="after", key=K, id=f"live-after-{K}", at="2026-10-05T14:01:02+00:00", sent="queued",
                 report="ISM services"),
            dict(kind="move", key="20261005-2114", id="live-move-20261005-2114", at="2026-10-05T18:15:02+00:00",
                 sent="queued", bar="2026-10-05T21:14:00"),
            dict(kind="before", key="y", id="live-before-y", at="2026-10-05T21:30:02+00:00", sent="queued",
                 report="FOMC minutes")]                                       # 00:30 on the server's 6 Oct
    (folder / "sent.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sent), encoding="utf-8")
    bot = tmp_path / "bot-sent.jsonl"
    bot.write_text("".join(json.dumps({"id": f"live-{k}-{K}"}) + "\n" for k in ("before", "after")), encoding="utf-8")
    (folder / "live.log").write_text(
        "2026-10-05T20:58:02+05:30 no live prices: MT5 did not start ((-6, 'Terminal: Authorization failed'))\n"
        "2026-10-05T19:00:05+05:30 before ISM services 20261005-1400-146608: queued\n"
        "2026-10-04T20:00:00+05:30 preview X 1: FAILED boom\n", encoding="utf-8")   # the server day before
    (folder / "state.json").write_text(json.dumps({"next": {"report": "FOMC minutes", "colour": "red",
                                                             "utc": "2026-10-07T18:00:00+00:00"}}), encoding="utf-8")
    text = LK.day_report(MON, folder, bot_sent=bot)
    assert text.splitlines()[0] == "📋 Live alerts, Mon 05 Oct - the daily check"
    assert "• News alerts: 2 - ISM services: before and after." in text
    assert "• Big moves with no news: 1, at 23:44 India time." in text      # server 21:14 (UTC+3 till 25 Oct)
    assert "⚠️ 1 alert(s) queued but NOT sent by the bot." in text          # the big move
    assert "the alerts saw 672 of 1,320 minutes live - every one the same as MT5's own history." in text
    assert "• Minutes not seen live: 648" in text and "• MT5 did not answer: 1 time(s)" in text
    assert "failed line" not in text                                          # the FAILED line was the day before
    assert "Trading days the live alerts have run: 1 (since Mon 05 Oct)." in text
    assert "Next news alert: Fed minutes (red), Wed 07 Oct 23:30." in text
    assert text.endswith("open a Claude chat and paste your L17a prompt.")   # an alert the bot did not send
    assert not re.search(r"\d{4}\.\d|\d,\d{3}\.\d|2000", text)               # no price
    bot.write_text(bot.read_text(encoding="utf-8") + json.dumps({"id": "live-move-20261005-2114"}) + "\n",
                   encoding="utf-8")
    calm = LK.day_report(MON, folder, bot_sent=bot)
    assert "• The bot sent every one." in calm and calm.endswith("Nothing for you to do.")
    with open(folder / "live.log", "a", encoding="utf-8") as fh:
        fh.write("2026-10-05T23:50:07+05:30 preview Fed minutes y: FAILED KeyError: 'x'\n")
    other = {**MON, "other": 2, "first_other": ["18:27"]}
    text = LK.day_report(other, folder, bot_sent=bot)
    assert "⚠️ Prices: on 2 minute(s) the alerts saw OTHER prices than MT5's own history (first at 20:57 India time)." \
        in text and "⚠️ 1 failed line(s) in the live log, first: preview Fed minutes y: FAILED" in text
    (folder / "sent.jsonl").write_text("", encoding="utf-8")
    assert "• News alerts: none (none of your news that day)." in LK.day_report(MON, folder, bot_sent=bot)


def test_minutes_mt5_lost_with_no_alert_near_them_are_said_calmly_not_as_a_warning(tmp_path):
    """8 Oct 2026: on Tue 6 and Wed 7 Oct MT5 lost its login for a few minutes; the day report said ⚠️ "open a Claude
    chat" and a session had to say by hand that no alert read those minutes. Now the machine says it - and still warns
    when an alert came soon after such a minute, when MT5 never said it lost its connection, or when not every such
    minute is known (a check written before 8 Oct)."""
    folder = config.LIVE_DIR
    folder.mkdir(parents=True)
    bot = tmp_path / "bot-sent.jsonl"
    bot.write_text(json.dumps({"id": "live-before-k"}) + "\n", encoding="utf-8")
    log = "2026-10-07T19:12:01+05:30 no live prices: MT5 did not start ((-6, 'Terminal: Authorization failed'))\n"
    (folder / "live.log").write_text(log, encoding="utf-8")
    late = dict(kind="before", key="k", id="live-before-k", at="2026-10-07T16:30:02+00:00", sent="queued",
                report="FOMC minutes")                     # 19:30 server: 2 h 48 min after the last such minute
    (folder / "sent.jsonl").write_text(json.dumps(late) + "\n", encoding="utf-8")
    wed = dict(day="2026-10-07", history=1320, seen=1289, same=1282, other=3, minor=2, missed=31, extra=0,
               first_other=["13:02", "15:31", "16:42"], first_missed=["10:32"], other_at=["13:02", "15:31", "16:42"])
    assert LK.explained(wed, folder)
    text = LK.day_report(wed, folder, bot_sent=bot)
    assert "⚠️" not in text and text.endswith("Nothing for you to do.")
    assert ("• Prices: 1,286 of the 1,289 minutes seen live had the same prices as MT5's own history; 3 were not "
            "complete (first at 15:32, 18:01, 19:12 India time) - MT5 lost its connection that day") in text
    assert "• MT5 did not answer: 1 time(s)" in text
    # an alert 40 minutes after a minute MT5 lost: it may have read it -> the warning, as before
    soon = {**late, "key": "s", "id": "live-before-s", "at": "2026-10-07T14:22:02+00:00"}      # 16:42 server + 40 min
    (folder / "sent.jsonl").write_text(json.dumps(late) + "\n" + json.dumps(soon) + "\n", encoding="utf-8")
    assert not LK.explained(wed, folder)
    assert "⚠️ Prices: on 3 minute(s) the alerts saw OTHER prices" in LK.day_report(wed, folder, bot_sent=bot)
    (folder / "sent.jsonl").write_text(json.dumps(late) + "\n", encoding="utf-8")
    # MT5 never said it lost its connection that day -> the warning
    (folder / "live.log").write_text("", encoding="utf-8")
    assert not LK.explained(wed, folder)
    (folder / "live.log").write_text(log, encoding="utf-8")
    # a check from before 8 Oct: 5 such minutes, only the first 3 known -> the warning
    old = {k: v for k, v in wed.items() if k != "other_at"} | {"other": 5}
    assert not LK.explained(old, folder)
    assert LK.day_report(old, folder, bot_sent=bot).endswith("open a Claude chat and paste your L17a prompt.")
    assert LK.explained({**wed, "other": 0, "other_at": []}, folder)               # nothing to explain


def _shifted(r: np.ndarray, first: str) -> np.ndarray:
    out = r.copy()
    t0 = int((pd.Timestamp(first) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
    out["time"] = t0 + 60 * np.arange(len(r))
    return out


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_the_check_keeps_every_other_minute_and_tells_only_when_not_explained(tmp_path):
    kept, tape, folder = tmp_path / "kept", tmp_path / "tape", tmp_path / "live"
    kept.mkdir()
    tape.mkdir()
    folder.mkdir()
    (kept / "2026-10-07.csv").write_bytes(HP.day_text(rates("2026-10-07 13:00", 8)))
    seen = rates("2026-10-07 13:00", 8)
    for i in (1, 3, 5, 7):                                    # 13:01 13:03 13:05 13:07: the last ticks never came
        seen["close"][i] += 0.05                              # inside the minute's high and low
        seen["tick_volume"][i] -= 2                           # 3 of its 5 ticks
    (tape / "2026-10-07.csv").write_bytes(HP.day_text(seen))
    told = []
    rec, = LK.run(folder=folder, tape_dir=tape, kept_dir=kept, tell=told.append)
    assert rec["other"] == 4 and rec["partial"] == 4 and rec["other_at"] == ["13:01", "13:03", "13:05", "13:07"]
    assert rec["first_other"] == ["13:01", "13:03", "13:05"]
    assert told == []                  # explained: the day's report says it, no ⚠️ message - with no MT5 line in live.log
    (kept / "2026-10-08.csv").write_bytes(HP.day_text(rates("2026-10-08 13:00", 8)))
    whole = _shifted(seen, "2026-10-08 13:00")
    whole["tick_volume"] = 5                                  # every tick, other prices: not part of the same minute
    (tape / "2026-10-08.csv").write_bytes(HP.day_text(whole))
    rec2, = LK.run(folder=folder, tape_dir=tape, kept_dir=kept, tell=told.append)
    assert (rec2["other"], rec2["partial"]) == (4, 0) and len(told) == 1


def test_a_minute_the_live_feed_got_part_of_is_said_calmly_with_its_true_cause(tmp_path):
    """9 Oct 2026: Thursday's report said its one incomplete minute (21:47 India time) came because "MT5 lost its
    connection that day" - but MT5's only login loss was 19:57; the live bar had 931 of the busy minute's 936 ticks.
    The check now tests the minute itself: fewer ticks, high and low inside the history's = part of the minute."""
    folder = config.LIVE_DIR
    folder.mkdir(parents=True)
    seen = rates("2026-10-08 19:15", 4)
    seen["close"][2] += 0.05
    seen["tick_volume"][2] -= 1                               # 19:17: 4 of its 5 ticks
    seen["high"][3] += 0.5                                    # 19:18: a high the history never had
    seen["tick_volume"][3] -= 1
    a, b = (HP.day_text(r).decode("ascii").splitlines()[1:] for r in (seen, rates("2026-10-08 19:15", 4)))
    assert [LK._part_of(x, y) for x, y in zip(a, b)] == [False, False, True, False]
    bot = tmp_path / "bot-sent.jsonl"
    bot.write_text("", encoding="utf-8")
    thu = dict(day="2026-10-08", history=1320, seen=1285, same=1283, other=1, partial=1, minor=1, missed=35, extra=0,
               first_other=["19:17"], first_missed=["05:43"], other_at=["19:17"])
    assert LK.explained(thu, folder)                          # no MT5 line in live.log: not needed any more
    text = LK.day_report(thu, folder, bot_sent=bot)
    assert ("• Prices: 1,284 of the 1,285 minutes seen live had the same prices as MT5's own history; 1 was not "
            "complete (first at 21:47 India time) - the live feed got only part of its ticks, and no alert was made "
            "within 60 minutes after it, so no alert read it.") in text
    assert "MT5 lost" not in text and text.endswith("Nothing for you to do.")
    two = {**thu, "other": 2, "other_at": ["19:17", "19:18"], "first_other": ["19:17", "19:18"]}
    assert not LK.explained(two, folder)                      # 19:18 is not part of the minute -> the warning
    assert "⚠️ Prices: on 2 minute(s)" in LK.day_report(two, folder, bot_sent=bot)
    soon = dict(kind="move", key="m", id="live-move-m", at="2026-10-08T16:40:02+00:00", sent="queued",
                bar="2026-10-08T19:39:00")                    # 19:40 server: 23 minutes after 19:17
    (folder / "sent.jsonl").write_text(json.dumps(soon) + "\n", encoding="utf-8")
    assert not LK.explained(thu, folder)                      # an alert may have read it -> the warning


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_the_pass_sends_the_days_report_once_after_its_check_and_stop_holds_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SEALED_MT5_DIR", tmp_path / "mt5")       # made-up kept days, never the real ones
    monkeypatch.setattr(config, "TELEGRAM_DIR", tmp_path / "telegram")
    kept, tape = tmp_path / "mt5" / "M1", config.LIVE_TAPE_DIR
    kept.mkdir(parents=True)
    tape.mkdir(parents=True)
    for d in ("2026-10-05", "2026-10-06"):
        (tape / f"{d}.csv").write_bytes(HP.day_text(rates(f"{d} 01:00", 6)))
    (kept / "2026-10-05.csv").write_bytes(HP.day_text(rates("2026-10-05 01:00", 6)))
    monkeypatch.setattr(LR, "once", lambda **kw: [])
    box = tmp_path / "outbox.jsonl"
    assert LR.locked_once(outbox=box, paused=lambda: False) == []
    line, = [json.loads(x) for x in box.read_text(encoding="utf-8").splitlines()]
    assert line["id"] == "live-day-2026-10-05" and "6 of 6 minutes live - every one the same" in line["text"]
    assert line["text"].endswith("Nothing for you to do.")
    LR.locked_once(outbox=box, paused=lambda: False)
    assert len(box.read_text(encoding="utf-8").splitlines()) == 1             # a day is checked and told once
    (kept / "2026-10-06.csv").write_bytes(HP.day_text(rates("2026-10-06 01:00", 6)))
    LR.locked_once(outbox=box, paused=lambda: True)                           # he typed STOP
    assert len(box.read_text(encoding="utf-8").splitlines()) == 1
    days = [r for r in LR.HA.read_jsonl(config.LIVE_DIR / "sent.jsonl") if r["kind"] == "day"]
    assert [(r["key"], r["sent"]) for r in days] == [("2026-10-05", "queued"), ("2026-10-06", "not sent: STOP")]


@pytest.mark.private                                    # opens the lock through his yes in plan/work/T14-decisions.md
def test_a_failing_report_never_stops_the_pass_or_the_check(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SEALED_MT5_DIR", tmp_path / "mt5")
    kept, tape = tmp_path / "mt5" / "M1", config.LIVE_TAPE_DIR
    kept.mkdir(parents=True)
    tape.mkdir(parents=True)
    (kept / "2026-10-05.csv").write_bytes(HP.day_text(rates("2026-10-05 01:00", 6)))
    (tape / "2026-10-05.csv").write_bytes(HP.day_text(rates("2026-10-05 01:00", 6)))
    monkeypatch.setattr(LR, "once", lambda **kw: ["ran"])
    monkeypatch.setattr(LK, "day_report", lambda *a, **k: 1 / 0)
    assert LR.locked_once(outbox=tmp_path / "outbox.jsonl", paused=lambda: False) == ["ran"]
    assert [r["day"] for r in LR.HA.read_jsonl(config.LIVE_DIR / "checks.jsonl")] == ["2026-10-05"]
    assert "the day's report failed: ZeroDivisionError" in (config.LIVE_DIR / "live.log").read_text(encoding="utf-8")


def test_the_commands_print_a_big_move_and_a_days_report(monkeypatch, capsys):
    from machine.live import __main__ as cli
    monkeypatch.setattr(LR, "locked_once", lambda: [{"kind": "move", "key": "20261005-2114", "sent": "queued"}])
    assert cli.main(["once"]) == 0 and "move 20261005-2114: queued" in capsys.readouterr().out   # no report: no error
    config.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (config.LIVE_DIR / "checks.jsonl").write_text(json.dumps(MON) + "\n", encoding="utf-8")
    assert cli.main(["report", "2026-10-05"]) == 0 and "📋 Live alerts, Mon 05 Oct" in capsys.readouterr().out
    assert cli.main(["report", "2026-10-06"]) == 1 and "not checked yet" in capsys.readouterr().out


# ===================================================================================================== the big move

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2019-06-03", periods=26)]     # server = UTC+3, NY = UTC-4


def flat(days=DAYS, first="12:00", n=120, price=2000.0) -> pd.DataFrame:
    """Made-up 1-minute bars, server 12:00-14:00 on each day: the price flat, every bar 0.20 from high to low."""
    from machine.data import clock
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d} {first}") + pd.Timedelta(minutes=i) for d in days for i in range(n)],
                           name="server_time")
    m1 = pd.DataFrame({"open": price, "high": price + 0.1, "low": price - 0.1, "close": price}, index=idx)
    m1["vol"], m1["spread"] = 5, 20
    m1["utc"] = clock.server_to_utc(idx)
    return m1


def climbed(m1: pd.DataFrame, at: str, up: float = 10.0) -> pd.DataFrame:
    """The same bars with gold climbing `up` dollars over the 5 bars from `at`, then staying there."""
    m = m1.copy()
    step = ((m.index - pd.Timestamp(at)) / pd.Timedelta(minutes=5)).to_numpy().clip(0, 1) * up
    for c in ("open", "high", "low", "close"):
        m[c] = m[c] + step
    return m


def test_six_times_the_usual_and_a_third_of_a_day_is_big_and_a_flat_day_never():
    jump = pd.Timestamp(f"{DAYS[-1]} 13:05")
    mv = LM.measure(climbed(flat(), f"{DAYS[-1]} 13:00"))
    row_ = mv.loc[jump]
    assert row_["usual"] == pytest.approx(0.2) and row_["adr"] == pytest.approx(0.2)
    assert row_["move"] == pytest.approx(10.2)
    assert LM.big(mv).loc[jump] and not LM.big(mv.loc[:f"{DAYS[-1]} 12:59"]).any()
    wide = flat()
    wide.loc[wide.index.strftime("%H:%M") == "12:00", "high"] += 5.0       # every day 5.20 from high to low, so a
    mv = LM.measure(climbed(wide, f"{DAYS[-1]} 13:00", up=1.2))           # 1.40 move: 7 x usual, 0.27 of a day
    row_ = mv.loc[jump]
    assert row_["times_usual"] == pytest.approx(7.0) and row_["share_of_day"] == pytest.approx(1.4 / 5.2)
    assert not LM.big(mv).loc[jump]


def test_the_move_measure_cuts_off_the_future():
    m1 = climbed(flat(), f"{DAYS[-1]} 13:00")
    guards.assert_no_future(LM.measure, m1, [130, 1500, len(m1) - 62, len(m1) - 55, len(m1) - 2])


def test_the_adr_is_his_adr_blocks_ruler():
    from machine.blocks import adr as ADR
    m1 = flat(DAYS[:16])
    m1["high"] += (m1.index.day % 4) * 0.5                            # days of different sizes
    pd.testing.assert_series_equal(LM.adr(m1), ADR.adr_bars(m1, "1m")["ruler"], check_names=False)


def test_his_news_near_the_move_and_the_quiet_hours():
    t = pd.DatetimeIndex(["2019-07-08 15:00"])

    def news(at, colour):
        return pd.DataFrame({"server": [pd.Timestamp(at)], "colour": [colour]})
    assert LM.near_news(t, news("2019-07-08 13:01", "red"))[0]        # red: 120 minutes after it
    assert not LM.near_news(t, news("2019-07-08 12:59", "red"))[0]
    assert not LM.near_news(t, news("2019-07-08 14:19", "orange"))[0]  # orange: only 40
    assert LM.near_news(t, news("2019-07-08 15:05", "orange"))[0]     # and from 5 minutes before it
    assert not LM.near_news(t, news("2019-07-08 15:06", "red"))[0]
    got = LM.quiet_times(pd.DatetimeIndex(["2019-07-08 10:00", "2019-07-08 13:59", "2019-07-08 14:00",
                                           "2019-07-08 14:30", "2019-07-08 18:00"]))
    assert list(got.strftime("%H:%M")) == ["10:00", "14:00", "18:00"]      # 4 quiet hours after each alert


def feed_of(m1: pd.DataFrame, usd_inr: float = 70.0):
    """A pass's prices from made-up bars: only the bars closed by `now`."""
    def feed(now):
        at = LP.server_now(now)
        return LP.Prices(m1[m1.index + pd.Timedelta(minutes=1) <= at], at, usd_inr, "test"), ""
    return feed


def utc_of(server: str) -> dt.datetime:
    from machine.data import clock
    return clock.server_to_utc(pd.DatetimeIndex([server]))[0].to_pydatetime()


def test_a_big_move_is_sent_once_with_his_words_and_then_quiet_for_4_hours(tmp_path):
    box = tmp_path / "outbox.jsonl"
    m1 = climbed(climbed(flat(), f"{DAYS[-1]} 13:00"), f"{DAYS[-1]} 13:40", up=-20.0)    # up, then down 40 min later
    feed = feed_of(m1)
    done = LR.once(utc_of(f"{DAYS[-1]} 13:06"), rows=[], outbox=box, tab=pd.DataFrame(), paused=lambda: False,
                   feed=feed)
    move, = [d for d in done if d["kind"] == "move"]
    text = move["text"]
    assert text.startswith("⚡ Gold moved UP $10.20 in 15 minutes") and "1,999.90 → 2,010.10" in text
    assert "51 times the usual" in text and "Your analysis may change." in text and "about once a week" in text
    assert "On 0.01 lot: ₹714." in text                               # $10.20 x 1 ounce x 70
    assert LR.once(utc_of(f"{DAYS[-1]} 13:46"), rows=[], outbox=box, tab=pd.DataFrame(), paused=lambda: False,
                   feed=feed) == []                                   # big again, but inside the 4 quiet hours
    assert len(box.read_text(encoding="utf-8").splitlines()) == 1
    news = [row(7, "Non-Farm Employment Change", str(utc_of(f"{DAYS[-1]} 12:30"))[:16], "red", "1K", "2K")]
    assert LR.once(utc_of(f"{DAYS[-1]} 13:06"), rows=news, folder=tmp_path / "other", outbox=tmp_path / "b2.jsonl",
                   tab=pd.DataFrame(), paused=lambda: False, feed=feed) == []      # his red news 35 min before: none


def test_gold_now_before_and_gold_since_the_number_after(tmp_path, no_lookup):
    box = tmp_path / "outbox.jsonl"
    m1 = climbed(flat(), f"{DAYS[-1]} 13:30", up=4.0)                 # the number at 13:30 server, gold climbs
    rel_utc = str(utc_of(f"{DAYS[-1]} 13:30"))[:16]
    before = [row(1, "Non-Farm Employment Change", rel_utc, "red", "", "171K", "187K")]
    done = LR.once(utc_of(f"{DAYS[-1]} 13:00"), rows=before, outbox=box, tab=pd.DataFrame(), paused=lambda: False,
                   feed=feed_of(m1))
    assert "Gold now: 2,000.00." in done[0]["text"]
    after = [row(1, "Non-Farm Employment Change", rel_utc, "red", "336K", "171K", "187K")]
    done = LR.once(utc_of(f"{DAYS[-1]} 13:33"), rows=after, outbox=box, tab=pd.DataFrame(), paused=lambda: False,
                   feed=feed_of(m1))
    assert "Gold since the number: UP $1.60 in 3 minutes (2,000.00 → 2,001.60)." in done[0]["text"]


@pytest.mark.data
@pytest.mark.news
def test_the_situation_is_measured_as_every_past_release_was():
    """machine/live/situation.py against reaction.py's own table (2019 releases): the session, the ADR, the day so far
    and the 2 hours before are the same numbers."""
    from machine.data import bars as BARS
    from machine.news import reaction
    ev = reaction.load("events", years=(2019,))
    ev = ev[ev["his_news"] & ev["has_bars"].fillna(False)].iloc[::12]
    m1 = BARS.load_bars("1m", "2018-11-01", "2019-12-31")
    for t, r in ev.iterrows():
        s = LSit.at(m1, t)
        if not s:
            assert pd.isna(r["p0"]), t
            continue
        assert s["session"] == r["session"] and s["p0"] == r["p0"], t
        for k in ("adr", "adr_used", "before_120_adr"):
            assert (pd.isna(r[k]) and k not in s) or s[k] == pytest.approx(r[k], rel=1e-9), (t, k)


@pytest.mark.data
@pytest.mark.news
def test_a_2020_week_replayed_sends_the_big_move_of_the_vaccine_news():
    """9 Nov 2020: the vaccine news (about 17:15 India time) - no news of his near it - found by the rule, sent at the
    pass right after its bar closed."""
    from machine.data import clock
    done = LR.rehearse(dt.date(2020, 11, 8))
    moves = [d for d in done if d["kind"] == "move" and d["bar"].startswith("2020-11-09")]
    assert moves and "DOWN" in moves[0]["text"]
    bar_close = pd.Timestamp(moves[0]["bar"]) + pd.Timedelta(minutes=1)
    assert pd.Timestamp(moves[0]["at"]) == clock.server_to_utc(pd.DatetimeIndex([bar_close]))[0]


# ======================================================================================= step 3: his bias and levels

@pytest.mark.private                                    # reads his yes in plan/work/T14-decisions.md
def test_the_live_trial_opening_is_shut_outside_and_needs_t14s_yes(monkeypatch):
    live = pd.DatetimeIndex(["2026-10-05 12:00"])
    with pytest.raises(guards.SealedYearsError):
        guards.check_dates(live)
    with guards.live_trial("a test", "his bias on live bars"):
        assert guards.check_dates(live) is live                       # his blocks may read live bars in here
    with pytest.raises(guards.SealedYearsError):
        guards.check_dates(live)                                      # and shut again after
    with pytest.raises(guards.SealedYearsError, match="who and why"):
        with guards.live_trial("", "x"):
            pass
    monkeypatch.setattr(guards, "allowed_uses", lambda path=None: {"live-trial": False})
    with pytest.raises(guards.SealedYearsError, match="live-trial = yes"):
        with guards.live_trial("a test", "his bias on live bars"):
            pass


def test_only_the_live_alerts_open_it():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "machine"
    users = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                   if re.search(r"live_trial\(", p.read_text(encoding="utf-8")) and p.name != "guards.py")
    assert users == ["live/situation.py"]


def test_without_his_charts_the_alert_leaves_his_bias_and_levels_out():
    m1 = flat()
    p = LP.Prices(m1, m1.index[-1] + pd.Timedelta(minutes=1), 70.0, "test")      # no charts_fn: no charts
    s = LSit.whole(p, m1.index[-1] + pd.Timedelta(minutes=1))
    assert s["charts_why"] == "no charts" and "bias_d" not in s and "up_level_kind" not in s
    assert not any(k.startswith(("bias", "up_level", "dn_level")) for k in LSit.for_lookup(s))
    lines = LA.now_lines(s)
    assert lines[0].startswith("Gold now: 2,000.00.") and not any("bias" in x or "levels" in x for x in lines)
    words = LA.now_lines({**s, "bias_h": 1.0, "bias_d": -1.0, "bias_w": 0.0, "up_level_kind": "session_high",
                          "up_level_dist": 4.2, "dn_level_kind": None, "level_names": {"up": "London session high"}})
    assert words[1] == "Your bias now: hour up · day down · week neutral."
    assert words[2] == "Your nearest levels: above, London session high ($4.20 away) · below, no level of yours left."


def test_the_next_release_is_built_ahead_once_per_code_and_a_failure_tells_him_once(monkeypatch, no_lookup):
    m1 = flat()
    now = utc_of(f"{DAYS[-1]} 13:00")
    p, _ = feed_of(m1)(now)
    rels = LC.releases(LC.events([row(1, "Non-Farm Employment Change", str(utc_of(f"{DAYS[-1]} 16:00"))[:16], "red",
                                      "", "171K", "187K")]))
    told, minute = [], dt.timedelta(minutes=1)
    rec = LR.preview(now, rels, p, None, tab=pd.DataFrame(), tell=told.append)
    assert rec["ok"] and rec["with_prices"] and rec["report"] == "Jobs report" and told == []
    assert LR.preview(now + minute, rels, p, rec, tab=pd.DataFrame(), tell=told.append) is None   # done, same code
    assert LR.preview(now + 9 * 60 * minute, rels, p, rec, tab=pd.DataFrame(), tell=told.append) is None   # passed
    monkeypatch.setattr(LR, "code_stamp", lambda: "a new version")
    assert LR.preview(now + minute, rels, p, rec, tab=pd.DataFrame(), tell=told.append)["ok"]     # new code: again

    def boom(*a, **k):
        raise ValueError("Cannot losslessly convert units")
    monkeypatch.setattr(LA, "before_text", boom)
    bad = LR.preview(now + 2 * minute, rels, p, {**rec, "code": "old"}, tab=pd.DataFrame(), tell=told.append)
    assert not bad["ok"] and "losslessly" in bad["error"]
    assert len(told) == 1 and "NFP" in told[0] and "will probably FAIL" in told[0] and "2000" not in told[0]
    again = LR.preview(now + 3 * minute, rels, p, {**bad, "code": "older"}, tab=pd.DataFrame(), tell=told.append)
    assert not again["ok"] and len(told) == 1                         # still failing: told once, not every pass


def test_a_pass_keeps_the_preview_in_its_state_and_a_rehearsal_never_builds_one(tmp_path, no_lookup):
    m1 = flat()
    rows = [row(1, "Non-Farm Employment Change", str(utc_of(f"{DAYS[-1]} 16:00"))[:16], "red", "", "171K", "187K")]
    told = []
    LR.once(utc_of(f"{DAYS[-1]} 13:00"), rows=rows, outbox=tmp_path / "o.jsonl", tab=pd.DataFrame(),
            paused=lambda: False, feed=feed_of(m1), tell=told.append, ahead=True)
    state = json.loads((config.LIVE_DIR / "state.json").read_text(encoding="utf-8"))
    assert state["preview"]["ok"] and state["preview"]["report"] == "Jobs report"
    LR.once(utc_of(f"{DAYS[-1]} 13:01"), rows=rows, outbox=tmp_path / "o.jsonl", tab=pd.DataFrame(),
            paused=lambda: False, feed=feed_of(m1), tell=told.append, ahead=True)
    assert json.loads((config.LIVE_DIR / "state.json").read_text(encoding="utf-8"))["preview"] == state["preview"]
    assert any(line.startswith("Built ahead at ") for line in LR.status_lines())
    LR.once(utc_of(f"{DAYS[-1]} 13:00"), rows=rows, folder=tmp_path / "r", outbox=tmp_path / "r.jsonl",
            tab=pd.DataFrame(), paused=lambda: False, feed=feed_of(m1))     # rows given: a rehearsal - none
    assert json.loads((tmp_path / "r" / "state.json").read_text(encoding="utf-8"))["preview"] is None


def test_the_real_pass_builds_ahead_by_default(tmp_path, no_lookup, monkeypatch):
    """Exactly the way Task Scheduler runs it: the page is READ (no rows) and no tell is given - the preview is on.
    (5 Oct: it was decided after `rows` had been filled from the page, so on the PC it never ran; a test that gave
    its own tell missed it.) Telegram's sender is replaced, never alert() itself, so this tests the real default."""
    told = []
    monkeypatch.setattr(LR.HA, "telegram", lambda: told.append)
    m1 = flat()
    rows = [row(1, "Non-Farm Employment Change", str(utc_of(f"{DAYS[-1]} 16:00"))[:16], "red", "", "171K", "187K")]
    LR.once(utc_of(f"{DAYS[-1]} 13:00"), get=lambda url: _page(rows), outbox=tmp_path / "o.jsonl",
            tab=pd.DataFrame(), paused=lambda: False, feed=feed_of(m1))
    state = json.loads((config.LIVE_DIR / "state.json").read_text(encoding="utf-8"))
    assert state["fetched"] and state["preview"]["ok"] and state["preview"]["report"] == "Jobs report" and told == []


def test_a_missing_surprise_table_leaves_the_size_out_not_the_alert(monkeypatch):
    monkeypatch.setattr(config, "NEWS_SURPRISES", config.DATA_DIR / "no-such-table.parquet")
    monkeypatch.setattr(LS, "_table", [])
    s = LS.score(dict(method="forecast", event="Nonfarm Payrolls", sign=1, actual="336K", forecast="171K"))
    assert s["group"] == "not scored" and "no 2016-2023 surprise table" in s["why"]


@pytest.mark.data
@pytest.mark.news
def test_his_bias_and_levels_on_live_cut_charts_are_the_tables():
    """chart_parts on the charts as the after-alert has them (1 minute after the number) = reaction.py's table, 2019."""
    from machine.news import reaction
    ev = reaction.load("events", years=(2019,))
    ev = ev[ev["his_news"] & ev["has_bars"].fillna(False) & ev["p0"].notna()].iloc[::24]
    for t, r in ev.iterrows():
        got = LSit.chart_parts(LP.history_charts(t + pd.Timedelta(minutes=1)), t)
        for k in ("bias_h", "bias_d", "bias_w"):
            assert got[k] == r[k], (t, k)
        for side in ("up", "dn"):
            kind = r[f"{side}_level_kind"]
            assert got[f"{side}_level_kind"] == (kind if isinstance(kind, str) else None), (t, side)
            assert (pd.isna(r[f"{side}_level_dist"]) and pd.isna(got[f"{side}_level_dist"])) or \
                got[f"{side}_level_dist"] == pytest.approx(r[f"{side}_level_dist"], rel=1e-9), (t, side)


@pytest.mark.data
@pytest.mark.news
def test_the_live_after_alert_picks_the_same_10_as_the_history_lookup():
    """With the table's own surprise, the situation the live after-alert measures (whole -> for_lookup) picks the same
    10 past releases, in the same order, as L12b.3's lookup of that release."""
    from machine.data import clock
    tab = similar.releases()
    for rep, t in (("Jobs report", "2023-10-06 15:30"), ("FOMC decision", "2023-12-13 21:00")):
        t = pd.Timestamp(t)
        qt = similar.query_of(tab, rep, t, "after")
        feed, _ = LR.history_feed(t - pd.Timedelta(days=60), t + pd.Timedelta(days=1))
        p, _ = feed(clock.server_to_utc(pd.DatetimeIndex([t + pd.Timedelta(minutes=1)]))[0].to_pydatetime())
        q = similar.Query(rep, t, qt.when, z=qt.z, group=qt.group, situation=LSit.for_lookup(LSit.whole(p, t)))
        assert list(similar.lookup(tab, q).near["time"]) == list(similar.lookup(tab, qt).near["time"]), rep
