"""L16.1 - the host PC runs every day: the daily run, the heartbeat, the sealed prices, MLflow, the two Task Scheduler
jobs. Every test works in a temporary folder with a fake Telegram, a fake MT5 and a fake Windows log: nothing here
sends a real message, reads a real price or asks the real Windows anything."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pytest

from machine import config
from machine.host import alert, daily, heartbeat, lock, power, prices, procs, tasks

UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
RATE = np.dtype([("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                 ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])   # MetaTrader5's own dtype
FAKE_PRICE = 1234.56                                   # must never show up in anything the machine writes or says


@pytest.fixture
def host(tmp_path, monkeypatch):
    for name, sub in (("HOST_DIR", "data/host"), ("SEALED_DIR", "data/sealed"), ("SEALED_MT5_DIR", "data/sealed/mt5"),
                      ("WORK_DIR", "plan/work"), ("TELEGRAM_DIR", "data/telegram"), ("MLFLOW_DIR", "data/mlflow"),
                      ("TRIALS_DIR", "plan/work/trials")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "env", lambda name, default=None: default)       # no key, no login, no ping URL
    monkeypatch.setattr(alert, "telegram", lambda: None)                        # never the real bot
    return tmp_path


class Tell:
    """A fake Telegram: keeps what was sent; `fail` times it raises first (no network)."""
    def __init__(self, fail: int = 0):
        self.sent, self.fail = [], fail

    def __call__(self, text):
        if self.fail:
            self.fail -= 1
            raise OSError("no network")
        self.sent.append(text)


def no_sleep(_):
    return None


# ================================================================================================= the lock

def test_one_run_at_a_time(tmp_path):
    with lock.one_run(tmp_path / "x.lock") as first:
        with lock.one_run(tmp_path / "x.lock") as second:
            assert first is True and second is False
    with lock.one_run(tmp_path / "x.lock") as again:
        assert again is True                           # let go when the first run ended


# ================================================================================================= the sealed prices

def secs(t: str) -> int:
    return int((dt.datetime.fromisoformat(t) - dt.datetime(1970, 1, 1)).total_seconds())


def bars(day: str, n: int, step_min: int = 1, price: float = FAKE_PRICE) -> np.ndarray:
    start = secs(f"{day} 01:00:00")
    out = np.zeros(n, dtype=RATE)
    for i in range(n):
        out[i] = (start + i * 60 * step_min, price, price + 0.1, price - 0.1, price, 7, 30, 0)
    return out


class FakeMT5:
    TIMEFRAME_M1, TIMEFRAME_M5 = 1, 5

    def __init__(self, data: dict, server="MetaQuotes-Demo", connected=True, changing=False):
        self.data, self.server, self.connected, self.changing = data, server, connected, changing
        self.asked, self.down = 0, False

    def initialize(self, **kw):
        return True

    def shutdown(self):
        self.down = True

    def last_error(self):
        return (1, "Success")

    def terminal_info(self):
        return type("T", (), {"connected": self.connected})()

    def account_info(self):
        return type("A", (), {"server": self.server, "company": "MetaQuotes Ltd.", "trade_mode": 0})()

    def symbol_select(self, symbol, on):
        return True

    def copy_rates_from_pos(self, symbol, tf, start, count):
        return self.data[tf][-1:]

    def copy_rates_range(self, symbol, tf, lo, hi):
        self.asked += 1
        r = self.data[tf]
        r = r[(r["time"] >= lo) & (r["time"] <= hi)]
        if self.changing:                               # still loading: every answer is longer than the last
            return r[: max(1, len(r) - 10 + self.asked)]
        return r


def l1_sealed(host: Path, last: str = "2026-08-14 22:59:00"):
    for name in prices.L1_SEALED.values():
        f = host / "data" / "sealed" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"{prices.HEADER}\n{last},{FAKE_PRICE},{FAKE_PRICE},{FAKE_PRICE},{FAKE_PRICE},1,1\n")


def gold(days: list[str], m1: int = 30) -> dict:
    return {1: np.concatenate([bars(d, m1) for d in days]),
            5: np.concatenate([bars(d, m1 // 5, 5) for d in days])}


SAT_EVENING = dt.datetime(2026, 10, 3, 19, 0, tzinfo=UTC)      # 22:00 on the server's clock (UTC+3 till 25 Oct)


def test_the_text_is_l1s_own_format():
    r = np.zeros(1, dtype=RATE)
    r[0] = (secs("2016-01-05 10:01:00"), 1076.53, 1076.6, 1076.47, 1076.56, 116, 36, 0)
    assert prices.lines(r) == ["2016-01-05 10:01:00,1076.53,1076.6,1076.47,1076.56,116,36"]  # a line of L1's raw file
    assert prices.day_text(r).startswith(b"time,open,high,low,close,vol,spread\n")


def test_keeps_complete_days_after_l1_and_never_today(host):
    l1_sealed(host)
    mt5 = FakeMT5(gold(["2026-08-17", "2026-08-18", "2026-10-02", "2026-10-05"]))
    out = prices.store(SAT_EVENING, mt5=mt5, sleep=no_sleep)
    m1 = out["timeframes"]["M1"]
    assert m1["from_day"] == "2026-08-15"               # the day after L1's last line
    assert m1["to_day"] == "2026-10-02"                 # never server-today (3 Oct); 5 Oct is "newer" but not complete
    assert [d["day"] for d in m1["days"]] == ["2026-08-17", "2026-08-18", "2026-10-02"]
    assert (host / "data/sealed/mt5/M1/2026-08-17.csv").is_file() and mt5.down
    assert not (host / "data/sealed/mt5/M1/2026-10-05.csv").exists()
    assert prices.kept_until("M1") == dt.date(2026, 10, 2)


def test_a_day_waits_until_mt5_has_a_later_day(host):
    l1_sealed(host, "2026-10-01 22:59:00")
    mt5 = FakeMT5(gold(["2026-10-02"]))                # the newest bar is ON 2 Oct: it may still be loading
    out = prices.store(SAT_EVENING, mt5=mt5, sleep=no_sleep)
    assert out["timeframes"]["M1"]["days"] == []
    assert not (host / "data/sealed/mt5/M1/2026-10-02.csv").exists()


def test_nothing_new_twice_and_a_kept_day_is_never_written_again(host):
    l1_sealed(host)
    data = gold(["2026-08-17", "2026-08-18"])
    prices.store(SAT_EVENING, mt5=FakeMT5(data), sleep=no_sleep)
    again = prices.store(SAT_EVENING, mt5=FakeMT5(data), sleep=no_sleep)
    assert again["timeframes"]["M1"]["to_day"] is None and again["timeframes"]["M1"]["days"] == []
    f = host / "data/sealed/mt5/M1/2026-08-17.csv"
    before = f.read_bytes()
    with pytest.raises(prices.PriceError, match="other M1 bars"):
        prices._keep("M1", bars("2026-08-17", 5, price=999.0), dt.date(2026, 8, 17), dt.date(2026, 8, 17), {})
    assert f.read_bytes() == before                     # nothing overwritten


class JustStarted(FakeMT5):
    """MT5 the moment Windows started after a power cut: its newest bar is still the evening before for the first
    `late` asks (None = never loads); the bars of today come after."""
    def __init__(self, data: dict, today: str, late: int | None):
        super().__init__(data)
        self.cut, self.late, self.newest_asks = secs(f"{today} 00:00:00"), late, 0

    def copy_rates_from_pos(self, symbol, tf, start, count):
        self.newest_asks += 1
        r = self.data[tf]
        if self.late is None or self.newest_asks <= self.late:
            r = r[r["time"] < self.cut]
        return r[-1:]


class Sleeps:
    def __init__(self):
        self.seconds = 0

    def __call__(self, s):
        self.seconds += s


TUE_AFTER_THE_CUT = dt.datetime(2026, 10, 6, 2, 32, 35, tzinfo=UTC)    # 08:02 India time = 05:32 on the server's clock


def test_after_a_power_cut_it_waits_for_mt5_to_load_and_keeps_yesterday(host):
    """6 Oct 2026: the PC was off from 23:47 to 07:57; the 07:00 run caught up at 08:02 with MT5 just opened, saw
    Monday's bar as the newest and kept nothing - Monday's daily check of the live feed waited a day."""
    l1_sealed(host, "2026-10-02 22:59:00")
    mt5, sleeps = JustStarted(gold(["2026-10-02", "2026-10-05", "2026-10-06"]), "2026-10-06", late=4), Sleeps()
    out = prices.store(TUE_AFTER_THE_CUT, mt5=mt5, sleep=sleeps)
    m1 = out["timeframes"]["M1"]
    assert [d["day"] for d in m1["days"]] == ["2026-10-05"] and m1["no_bar_today"] is False
    assert m1["waited_s"] == 4 * prices.LOAD_ASK_SECONDS
    assert (host / "data/sealed/mt5/M1/2026-10-05.csv").is_file()
    assert daily.prices_line(out) == ("Prices: 1 day kept (Mon 05 Oct), 30 one-minute rows - sealed, nobody opens "
                                      "them. MT5 was still loading: waited 40 seconds for it.")


def test_mt5_never_loads_today_waits_at_most_the_limit_and_says_so(host):
    l1_sealed(host, "2026-10-02 22:59:00")
    mt5, sleeps = JustStarted(gold(["2026-10-02", "2026-10-05", "2026-10-06"]), "2026-10-06", late=None), Sleeps()
    out = prices.store(TUE_AFTER_THE_CUT, mt5=mt5, sleep=sleeps)
    m1 = out["timeframes"]["M1"]
    assert m1["days"] == [] and m1["no_bar_today"] is True
    assert m1["waited_s"] == out["timeframes"]["M5"]["waited_s"] == config.MT5_LOAD_WAIT_SECONDS   # no longer
    assert not (host / "data/sealed/mt5/M1/2026-10-05.csv").exists()   # nothing kept half
    assert "still had no bar of today after 5 minutes" in daily.prices_line(out)


def test_no_wait_on_a_weekend_or_before_gold_opens(host):
    l1_sealed(host, "2026-10-01 22:59:00")
    for now in (SAT_EVENING, dt.datetime(2026, 10, 5, 22, 5, tzinfo=UTC)):  # Sat; Mon 22:05 UTC = Tue 01:05 server
        mt5, sleeps = JustStarted(gold(["2026-10-02"]), "2026-10-03", late=None), Sleeps()
        out = prices.store(now, mt5=mt5, sleep=sleeps)
        assert sleeps.seconds == 0 and out["timeframes"]["M1"]["waited_s"] == 0
        assert out["timeframes"]["M1"]["no_bar_today"] is False
    assert prices.should_have_today(TUE_AFTER_THE_CUT) and not prices.should_have_today(SAT_EVENING)


def test_refuses_another_server_and_a_terminal_without_connection(host):
    l1_sealed(host)
    with pytest.raises(prices.PriceError, match="not MetaQuotes-Demo"):
        prices.store(SAT_EVENING, mt5=FakeMT5(gold(["2026-08-17"]), server="XMGlobal-MT5"), sleep=no_sleep)
    with pytest.raises(prices.PriceError, match="not connected"):
        prices.store(SAT_EVENING, mt5=FakeMT5(gold(["2026-08-17"]), connected=False), sleep=no_sleep)
    assert not (host / "data/sealed/mt5/M1").exists()


def test_mt5_still_loading_keeps_nothing(host):
    l1_sealed(host)
    with pytest.raises(prices.PriceError, match="still loading"):
        prices.store(SAT_EVENING, mt5=FakeMT5(gold(["2026-08-17", "2026-08-18"]), changing=True), sleep=no_sleep)
    assert not list((host / "data/sealed/mt5").rglob("*.csv"))


def test_no_price_in_what_it_says_or_records(host):
    l1_sealed(host)
    out = prices.store(SAT_EVENING, mt5=FakeMT5(gold(["2026-08-17", "2026-08-18"])), sleep=no_sleep)
    said = json.dumps(out) + (host / "data/sealed/mt5/manifest.jsonl").read_text() + daily.prices_line(out)
    assert "1234." not in said                          # no price (a hash has no dot)
    assert daily.prices_line(out).startswith("Prices: 1 day kept (Mon 17 Aug)")   # 18 Aug waits for a later day


def test_only_prices_py_touches_the_sealed_mt5_folder():
    """The sealed days are never read for anything (hard rule 1): no other code in machine/ uses their folder - but
    the daily check of the live feed (L17a, T14 use 1 "live-trial"), which compares a kept day with what the live
    alerts saw, as text, after the lock opened that week (machine/live/check.py)."""
    root = Path(__file__).resolve().parent.parent / "machine"
    users = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
             if re.search(r"\bSEALED_MT5_DIR\b", p.read_text(encoding="utf-8"))]
    assert sorted(users) == ["config.py", "host/prices.py", "live/check.py"]
    text = (root / "live" / "check.py").read_text(encoding="utf-8")
    assert text.index("opened.week(") < text.index("compare(tape_dir")          # the lock first, then the reading


# ================================================================================================= Windows' log

def ev(provider: str, eid: int, at: str, data: dict | None = None, binary: str = "") -> str:
    d = "".join(f"<Data Name='{k}'>{v}</Data>" for k, v in (data or {}).items())
    b = f"<Binary>{binary}</Binary>" if binary else ""
    return (f"<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='{provider}'/>"
            f"<EventID>{eid}</EventID><TimeCreated SystemTime='{at}'/></System><EventData>{d}{b}</EventData></Event>")


K41 = {"BugcheckCode": "0", "SleepInProgress": "0", "PowerButtonTimestamp": "0", "LongPowerButtonPressDetected": "false"}
# his PC's real 6008 Binary of 3 Oct 2026: 22:50:22 India time, 17:20:22 UTC on 2 Oct
BIN_6008 = "EA070A00050002001600320016001F01EA070A00050002001100140016001F01600900003C000000"


def boot(at: str, start: str) -> str:
    return ev("Microsoft-Windows-Kernel-General", 12, at, {"StartTime": start})


def test_a_power_cut_is_named_with_its_times():
    xml = "<Events>" + boot("2026-10-03T07:36:58.85Z", "2026-10-03T07:36:56.5000000Z") \
        + ev("Microsoft-Windows-Kernel-Power", 41, "2026-10-03T07:37:00.66Z", K41) \
        + ev("EventLog", 6008, "2026-10-03T07:37:30.36Z", {}, BIN_6008) + "</Events>"
    [o] = power.periods(power.parse(xml))
    assert o.why == "the power went off - no proper shutdown"
    assert o.off_at == dt.datetime(2026, 10, 2, 17, 20, 22, 287000, tzinfo=UTC)
    assert o.on_at == dt.datetime(2026, 10, 3, 7, 36, 56, 500000, tzinfo=UTC)
    assert round(o.minutes()) == 857


@pytest.mark.parametrize("flags, why", [({"BugcheckCode": "209"}, "Windows crashed - a blue screen"),
                                        ({"PowerButtonTimestamp": "133"}, "the power button was held down"),
                                        ({"LongPowerButtonPressDetected": "true"}, "the power button was held down")])
def test_a_crash_and_the_power_button_are_told_apart(flags, why):
    xml = boot("2026-10-03T07:36:58Z", "2026-10-03T07:36:56Z") + ev(
        "Microsoft-Windows-Kernel-Power", 41, "2026-10-03T07:37:00Z", {**K41, **flags})
    assert power.periods(power.parse(xml))[0].why == why


def test_a_proper_restart_by_windows_update():
    xml = ev("User32", 1074, "2026-10-01T20:00:00Z", {"param1": r"C:\Windows\servicing\TrustedInstaller.exe (PC)",
                                                       "param5": "restart"}) \
        + ev("EventLog", 6006, "2026-10-01T20:00:40Z") + boot("2026-10-01T20:01:30Z", "2026-10-01T20:01:28Z")
    [o] = power.periods(power.parse(xml))
    assert o.why == "Windows Update restarted it" and o.off_at == dt.datetime(2026, 10, 1, 20, 0, 40, tzinfo=UTC)


def test_shut_down_with_fast_startup_and_plain_sleep():
    """His PC on 30 Sep 2026: 1074 "power off" by winlogon, 42 with TargetState 6, then back (107, Troubleshooter 1)."""
    xml = ev("User32", 1074, "2026-09-30T10:29:16Z", {"param1": r"C:\WINDOWS\system32\winlogon.exe (PC)",
                                                       "param5": "power off"}) \
        + ev("Microsoft-Windows-Kernel-Power", 42, "2026-09-30T10:29:20Z", {"TargetState": "6"}) \
        + ev("Microsoft-Windows-Kernel-Power", 107, "2026-09-30T10:29:39Z") \
        + ev("Microsoft-Windows-Power-Troubleshooter", 1, "2026-09-30T10:31:42Z",
             {"SleepTime": "2026-09-30T10:29:17Z", "WakeTime": "2026-09-30T10:31:37.9293863Z"}) \
        + ev("Microsoft-Windows-Kernel-Power", 42, "2026-09-30T12:00:00Z", {"TargetState": "4"}) \
        + ev("Microsoft-Windows-Kernel-Power", 107, "2026-09-30T12:30:00Z")
    a, b = power.periods(power.parse(xml))
    assert a.why == "it was shut down from Windows (Start, Power), with fast startup"
    assert a.on_at == dt.datetime(2026, 9, 30, 10, 31, 37, 929386, tzinfo=UTC)
    assert b.why == "it went to sleep" and round(b.minutes()) == 30


def test_the_query_names_only_the_events_it_reads():
    q = power.xpath(dt.datetime(2026, 10, 3, 12, 0, tzinfo=IST))
    assert "@SystemTime>='2026-10-03T06:30:00.000Z'" in q
    for eid in ("EventID=12", "EventID=41", "EventID=6008", "EventID=1074"):
        assert eid in q


# ================================================================================================= the programs

def proc(pid, parent, command, started="2026-10-03T07:37:34+00:00"):
    return procs.Proc(pid, parent, command, dt.datetime.fromisoformat(started))


def test_the_venv_launcher_and_its_python_are_one_program():
    ps = [proc(6732, 1972, '"C:\\R\\.venv\\Scripts\\pythonw.exe" -u -m machine.telegram.bot'),
          proc(11880, 6732, '"C:\\Python313\\pythonw.exe" -u -m machine.telegram.bot'),
          proc(500, 1, '"C:\\R\\.venv\\Scripts\\python.exe" -m machine.telegram.bot --test')]
    assert [p.pid for p in procs.bot(ps)] == [6732]
    assert procs.intake(ps) == []


# ================================================================================================= the heartbeat

NO_BOT = proc(5, 0, "pythonw.exe -m machine.host heartbeat")    # what Windows lists when only the beat itself runs


def beat(host, now, *, ps=None, offs=(), boot_at=None, free=100.0, tell=None):
    ps = [proc(1, 0, "pythonw.exe -m machine.telegram.bot"), NO_BOT] if ps is None else ps
    return heartbeat.beat(now, tell=tell or Tell(), sleep=no_sleep, procs_now=ps, offs=list(offs),
                          boot=boot_at or now - dt.timedelta(days=1), free_gb=free)


T0 = dt.datetime(2026, 10, 4, 10, 15, tzinfo=IST)


def test_windows_not_answering_is_never_a_stopped_bot(host):
    tell = Tell()
    rec = beat(host, T0, ps=[], tell=tell)              # the process question failed: an empty answer
    assert rec["bot"] is None and rec["problems"] == [] and tell.sent == []
    assert procs.bot([]) is None and procs.intake([]) is None


def test_first_beat_and_an_on_time_beat_send_nothing(host):
    tell = Tell()
    assert beat(host, T0, tell=tell)["first"] is True
    rec = beat(host, T0 + dt.timedelta(minutes=60), tell=tell)
    assert "missed" not in rec and rec["problems"] == [] and tell.sent == []


def test_a_missed_heartbeat_alerts_with_what_windows_says(host):
    tell = Tell()
    beat(host, T0, tell=tell)
    cut = power.Off(dt.datetime(2026, 10, 4, 5, 20, tzinfo=UTC), dt.datetime(2026, 10, 4, 7, 36, tzinfo=UTC),
                    "the power went off - no proper shutdown")
    rec = beat(host, T0 + dt.timedelta(hours=3, minutes=2), offs=[cut], tell=tell)
    assert rec["missed"]["minutes"] == 182 and rec["sent"] == ["missed"]
    [text] = tell.sent
    assert "heartbeat was missed" in text and "Sun 10:15 to Sun 13:17" in text
    assert "the power went off" in text and "Sun 10:50 to 13:06" in text


def test_missed_without_a_restart_says_the_heartbeat_itself_did_not_run(host):
    tell = Tell()
    beat(host, T0, tell=tell)
    beat(host, T0 + dt.timedelta(minutes=95), tell=tell)
    assert "did NOT stop" in tell.sent[0]


def test_a_stopped_bot_alerts_once_then_once_more_when_it_runs_again(host):
    tell = Tell()
    beat(host, T0, ps=[NO_BOT], tell=tell)
    beat(host, T0 + dt.timedelta(hours=1), ps=[NO_BOT], tell=tell)
    assert len(tell.sent) == 1 and "bot is not running" in tell.sent[0]
    beat(host, T0 + dt.timedelta(hours=2), tell=tell)
    assert len(tell.sent) == 2 and "running again" in tell.sent[1]


def test_he_typed_stop_so_nothing_is_sent(host):
    (host / "data/telegram").mkdir(parents=True)
    (host / "data/telegram/state.json").write_text(json.dumps({"sending": False}))
    tell = Tell()
    beat(host, T0, ps=[NO_BOT], tell=tell)
    assert tell.sent == [] and "he typed STOP" in (host / "data/host/alerts.log").read_text(encoding="utf-8")


def test_no_network_keeps_the_alert_for_the_next_beat(host):
    beat(host, T0, ps=[NO_BOT], tell=Tell(fail=3))
    assert len(alert.read_jsonl(host / "data/host/unsent.jsonl")) == 1
    later = Tell()
    beat(host, T0 + dt.timedelta(hours=1), ps=[NO_BOT], tell=later)
    assert len(later.sent) == 1 and "sent late" in later.sent[0] and "bot is not running" in later.sent[0]
    assert alert.read_jsonl(host / "data/host/unsent.jsonl") == []


def test_the_daily_run_not_run_today_is_a_problem_only_when_it_is_late(host):
    alert.append_jsonl(host / "data/host/daily.jsonl", {"at": "2026-10-03T07:00:05+05:30", "steps": []})
    at_8 = dt.datetime(2026, 10, 4, 8, 15, tzinfo=IST)
    assert "daily" not in beat(host, at_8)["problems"]                  # 07:00 + 2 hours not reached
    at_10 = dt.datetime(2026, 10, 4, 10, 15, tzinfo=IST)
    assert "daily" not in beat(host, at_10, boot_at=at_10 - dt.timedelta(minutes=10))["problems"]  # PC just started
    tell = Tell()
    assert "daily" in beat(host, at_10 + dt.timedelta(minutes=1), tell=tell)["problems"]
    assert "Today's daily run has not run" in tell.sent[0]


def test_disk_and_a_stuck_idea_run(host):
    old = (T0 - dt.timedelta(hours=7)).astimezone(UTC).isoformat()
    ps = [proc(1, 0, "pythonw.exe -m machine.telegram.bot"), proc(9, 0, "pythonw.exe -u -m machine.intake run", old)]
    tell = Tell()
    rec = beat(host, T0, ps=ps, free=2.0, tell=tell)
    assert rec["problems"] == ["disk", "intake"] and len(tell.sent) == 2


def test_the_live_alerts_not_running_alerts_once_then_once_more_when_they_run_again(host):
    """L17a: the live news alerts write data/live/state.json every minute; no pass for 10 minutes = one alert."""
    state = config.LIVE_DIR / "state.json"
    hour = dt.timedelta(hours=1)
    tell = Tell()
    assert "live" not in beat(host, T0, tell=tell)["problems"]                  # never installed: nothing to say
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"at": (T0 + hour - dt.timedelta(minutes=5)).isoformat()}))
    assert "live" not in beat(host, T0 + hour, tell=tell)["problems"]           # a pass 5 minutes ago
    assert "live" in beat(host, T0 + 2 * hour, tell=tell)["problems"]           # none for 65 minutes
    assert len(tell.sent) == 1 and "live news alerts have not run since Sun 11:10" in tell.sent[0]
    assert heartbeat.live_problem(T0 + 2 * hour, T0 + 2 * hour - dt.timedelta(minutes=10)) is None   # PC just on
    state.write_text(json.dumps({"at": (T0 + 3 * hour - dt.timedelta(minutes=1)).isoformat()}))
    assert "live" not in beat(host, T0 + 3 * hour, tell=tell)["problems"]
    assert len(tell.sent) == 2 and "live news alerts run again" in tell.sent[1]


def test_the_outside_watcher_gets_fail_while_a_problem_is_open(host, monkeypatch):
    monkeypatch.setattr(config, "env", lambda name, default=None: "https://hc-ping.com/abc" if name ==
                        "HEARTBEAT_PING_URL" else default)
    seen = []

    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(url, timeout):
        seen.append(url)
        return R()

    assert heartbeat.ping(False, opener=opener) == "ok (200)"
    assert heartbeat.ping(True, opener=opener) == "ok (200) /fail"
    assert seen == ["https://hc-ping.com/abc", "https://hc-ping.com/abc/fail"]


# ================================================================================================= the daily run

def fake_steps(fail: str | None = None) -> dict:
    def make(name):
        def fn(now):
            if name == fail:
                raise RuntimeError(f"{name} broke")
            return f"{daily.TITLES[name]}: fine.", {}
        return fn
    return {n: make(n) for n in daily.STEPS}


@pytest.fixture
def quiet_pc(monkeypatch):
    monkeypatch.setattr(power, "offs_since", lambda since: [])
    monkeypatch.setattr(power, "boot_time", lambda now=None: dt.datetime(2026, 10, 3, 7, 37, tzinfo=UTC))


def test_a_good_day_sends_one_summary_through_the_bots_outbox(host, quiet_pc, monkeypatch):
    monkeypatch.setattr(procs, "bot", lambda ps=None: [proc(1, 0, "bot")])
    tell = Tell()
    rec = daily.run(dt.datetime(2026, 10, 4, 7, 0, tzinfo=IST), tell=tell, functions=fake_steps(), sleep=no_sleep)
    assert rec["ok"] and tell.sent == [] and rec["summary"] == {"how": "outbox", "id": "daily-2026-10-04"}
    [line] = alert.read_jsonl(host / "plan/work/telegram-outbox.jsonl")
    assert line["text"].startswith("Daily run, Sun 04 Oct 07:00: all 5 steps worked ✅")
    assert "Bot: running." in line["text"] and "Heartbeat: not running yet." in line["text"]
    again = daily.run(dt.datetime(2026, 10, 4, 9, 30, tzinfo=IST), tell=tell, functions=fake_steps(), sleep=no_sleep)
    assert again["summary"]["id"] == "daily-2026-10-04-093000"        # the same id is never sent twice


def test_a_failed_step_alerts_at_once_and_the_rest_still_run(host, quiet_pc, monkeypatch):
    monkeypatch.setattr(procs, "bot", lambda ps=None: [])
    tell = Tell()
    rec = daily.run(dt.datetime(2026, 10, 4, 7, 0, tzinfo=IST), tell=tell, functions=fake_steps("caches"),
                    sleep=no_sleep)
    assert not rec["ok"] and [s["ok"] for s in rec["steps"]] == [True, False, True, True, True]
    alert_text, summary = tell.sent                     # the bot is not running: the summary goes straight too
    assert 'failed in step "Caches": caches broke' in alert_text
    assert "1 of 5 steps FAILED ❌" in summary and "❌ Caches: FAILED - caches broke" in summary
    assert "the bot is not running" in summary


def test_a_broken_summary_still_leaves_the_record_and_tells_him(host, quiet_pc, monkeypatch):
    monkeypatch.setattr(procs, "bot", lambda ps=None: [proc(1, 0, "bot")])
    monkeypatch.setattr(daily, "pc_line", lambda now, before: 1 / 0)
    tell = Tell()
    rec = daily.run(dt.datetime(2026, 10, 4, 7, 0, tzinfo=IST), tell=tell, functions=fake_steps(), sleep=no_sleep)
    assert rec["summary"]["how"] == "failed" and "summary could not be made" in tell.sent[0]
    assert alert.read_jsonl(host / "data/host/daily.jsonl")[-1]["ok"] is True


def test_windows_not_answering_sends_the_summary_the_usual_way(host, quiet_pc, monkeypatch):
    monkeypatch.setattr(procs, "bot", lambda ps=None: None)
    tell = Tell()
    rec = daily.run(dt.datetime(2026, 10, 4, 7, 0, tzinfo=IST), tell=tell, functions=fake_steps(), sleep=no_sleep)
    assert rec["summary"]["how"] == "outbox" and tell.sent == []
    [line] = alert.read_jsonl(host / "plan/work/telegram-outbox.jsonl")
    assert "Bot: could not check (Windows did not answer)." in line["text"]


def test_a_forced_failure_says_it_is_a_test(host, quiet_pc, monkeypatch):
    monkeypatch.setattr(procs, "bot", lambda ps=None: [proc(1, 0, "bot")])
    tell = Tell()
    rec = daily.run(dt.datetime(2026, 10, 4, 7, 0, tzinfo=IST), force_fail="mlflow", summary=False, tell=tell,
                    functions=fake_steps(), sleep=no_sleep)
    assert rec["forced"] == "mlflow" and rec["steps"][-1]["ok"] is False
    assert tell.sent[0].startswith("TEST - nothing is wrong.") and '"Tests recorded"' in tell.sent[0]
    with pytest.raises(ValueError):
        daily.run(force_fail="nonsense", functions=fake_steps())


def test_a_second_daily_run_at_the_same_time_does_nothing(host, quiet_pc):
    with lock.one_run(host / "data/host/daily.lock"):
        assert daily.run(functions=fake_steps(), tell=Tell(), sleep=no_sleep) is None


def test_the_pc_line_names_each_time_it_was_off(host, monkeypatch):
    monkeypatch.setattr(power, "boot_time", lambda now=None: dt.datetime(2026, 10, 3, 7, 37, tzinfo=UTC))
    monkeypatch.setattr(power, "offs_since", lambda since: [
        power.Off(dt.datetime(2026, 10, 2, 9, 40, tzinfo=UTC), dt.datetime(2026, 10, 2, 10, 0, tzinfo=UTC), "the power went off - no proper shutdown"),
        power.Off(dt.datetime(2026, 10, 2, 17, 20, tzinfo=UTC), dt.datetime(2026, 10, 3, 7, 36, tzinfo=UTC), "the power went off - no proper shutdown")])
    line = daily.pc_line(dt.datetime(2026, 10, 4, 7, 0, tzinfo=IST), None)
    assert line.startswith("PC: on since Sat 13:07. Off 2 times since the last daily run: Fri 15:10 to 15:30")
    assert "Fri 22:50 to Sat 13:06" in line


# ================================================================================================= MLflow

def trial(host: Path, name: str, **extra) -> Path:
    body = {"trial_id": name, "kind": "tester", "made_at": "2026-10-01T13:57:53+05:30", "who": "t", "idea_id": "idea-001",
            "version_id": "v-1", "parent_idea": None, "git": {"commit": "abc"},
            "data": {"prices": "data/gold_m1.parquet", "gold_m1_content_sha256": "416d"},
            "settings": {"stop": 2.5, "window": {"from": "2016", "to": "2020"}},
            "result": {"trades": 41, "net_inr_per_trade": -34.5, "by_year": {"2016": -12.0}, "ok": True},
            "note": "a test trial"}
    body.update(extra)
    f = host / "plan/work/trials" / f"{name}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(body))
    return f


def test_mlflow_has_one_run_per_trial_file_and_the_files_stay_the_count(host):
    pytest.importorskip("mlflow")
    from machine.host import mlflow_sync
    trial(host, "t1")
    trial(host, "t2", kind="belief")
    out = mlflow_sync.sync()
    assert (out["files"], out["in_mlflow"], out["new"]) == (2, 2, 2)
    assert mlflow_sync.sync()["new"] == 0                # nothing logged twice
    c = mlflow_sync.client()
    names = sorted(e.name for e in c.search_experiments())
    assert names == ["Default", "trials-belief", "trials-tester"]
    [run] = c.search_runs([c.get_experiment_by_name("trials-tester").experiment_id])
    assert run.data.params["window.from"] == "2016" and run.data.metrics["by_year.2016"] == -12.0
    assert "ok" not in run.data.metrics                  # a yes/no is not a number
    assert run.data.tags["trial_file"] == "plan/work/trials/t1.json" and run.data.tags["git_commit"] == "abc"
    assert run.data.tags["data_sha256"] == "416d"
    trial(host, "t1", note="changed")                    # trial files are never rewritten - if one is, it is named
    again = mlflow_sync.sync()
    assert again["changed"] == ["plan/work/trials/t1.json"] and "CHANGED" in mlflow_sync.line(again)


# ================================================================================================= Task Scheduler

def test_the_two_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    s = tasks.script()
    assert "'ML Bot - daily run (L16.1)'" in s and "'ML Bot - heartbeat (L16.1)'" in s
    assert "-Argument '-m machine.host daily'" in s and "-Argument '-m machine.host heartbeat'" in s
    assert f"-Daily -At '{config.DAILY_AT}'" in s and "-RepetitionInterval (New-TimeSpan -Minutes 60)" in s
    assert "-AtLogOn" in s and "$t2.Delay = 'PT2M'" in s
    assert s.count("-StartWhenAvailable -WakeToRun") == 2 and s.count("-MultipleInstances IgnoreNew") == 2
    assert str(tmp_path / ".venv" / "Scripts" / "pythonw.exe") in s


def test_no_console_under_task_scheduler_gets_a_log_file(host, monkeypatch):
    """pythonw from Task Scheduler: sys.stdout and sys.stderr are None, and a progress bar on them is an error."""
    import sys

    from machine.host import __main__ as cli
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    cli.console()
    try:
        sys.stderr.write("a progress bar\n")
        print("a line")
        assert (host / "data/host/console.log").read_text(encoding="utf-8") == "a progress bar\na line\n"
    finally:
        sys.stdout.close()
