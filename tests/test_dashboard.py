"""L16.2 - the read-only dashboard: it opens only its list of files, writes nothing, answers GET only, and never shows a
2024+ price (hard rule 1); compose.yaml and .dockerignore keep the data and the keys out of Docker.

Most tests build a fake machine: his real ideas #1 and #2 with every real trial file (so the numbers are the reports'
own), a made-up idea #3, made-up host / intake / bot files - and planted "live prices" in every text the page reads.
Next to them lie poison files the page must never open: the live alerts' folder, the sealed prices, the bot's outbox
and sent messages, his messages, .env.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from machine import config
from machine.dashboard import mask, page, read

REPO = Path(config.ROOT)
REAL_WORK = REPO / "plan" / "work"
T = dt.datetime(2026, 10, 6, 13, 30, tzinfo=read.INDIA)
PLANTED = ("4,123.45", "4123.45", "2,650.10", "2650.5", "4,321.99", "3999.75")     # "live prices", every shape
HIS_WORDS = "PRIVATE-WORDS-not-for-the-page"


# ============================================================================================================ helpers

def _iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def _log(t: dt.datetime) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def at_root(root: Path, monkeypatch) -> Path:
    """Point the machine at a fake folder (what the dashboard reads comes from config, read at call time)."""
    data, work = root / "data", root / "plan" / "work"
    for name, value in (("ROOT", root), ("DATA_DIR", data), ("PLAN_DIR", root / "plan"), ("WORK_DIR", work),
                        ("IDEAS_DIR", work / "ideas"), ("TRIALS_DIR", work / "trials"), ("HOST_DIR", data / "host"),
                        ("TELEGRAM_DIR", data / "telegram"), ("LIVE_DIR", data / "live"),
                        ("LIVE_TAPE_DIR", data / "live" / "tape"), ("SEALED_DIR", data / "sealed")):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(read, "_cache", {})
    (work / "ideas").mkdir(parents=True, exist_ok=True)
    return root


def poison(root: Path) -> None:
    """Files the page must never open: they hold live prices, his messages, the bot's key."""
    line = "Gold now: 4,123.45 (live); 4123.45; " + HIS_WORDS
    _write(root / "data" / "live" / "state.json", json.dumps({"at": _iso(T), "prices": {"close": 4123.45}}))
    _jsonl(root / "data" / "live" / "sent.jsonl", [{"kind": "after", "text": line}])
    _write(root / "data" / "live" / "live.log", f"{_log(T)} {line}\n")
    _write(root / "data" / "live" / "tape" / "2026-10-05.csv", "time,open,high,low,close\n2026.10.05 13:01,4123.45,"
           "4124.0,4122.5,4123.45\n")
    _write(root / "data" / "sealed" / "mt5" / "M1" / "2026-10-05.csv", "2026.10.05 13:01,4123.45,4124.0\n")
    _jsonl(root / "plan" / "work" / "telegram-outbox.jsonl", [{"id": "live-before-x", "text": line}])
    _jsonl(root / "data" / "telegram" / "sent.jsonl", [{"text": line}])
    _jsonl(root / "data" / "telegram" / "messages.jsonl", [{"text": HIS_WORDS}])
    _write(root / ".env", "# the bot's key would be here: " + HIS_WORDS + "\n")


def ideas_and_trials(root: Path) -> None:
    """His real ideas #1 and #2 and his whole strategy's folder, every real trial file, and a made-up idea #3."""
    ideas = root / "plan" / "work" / "ideas"
    for name in ("idea-001", "idea-002", "his-strategy"):
        shutil.copytree(REAL_WORK / "ideas" / name, ideas / name)
    shutil.copytree(REAL_WORK / "trials", root / "plan" / "work" / "trials")
    form = json.loads((ideas / "idea-002" / "form.json").read_text(encoding="utf-8"))
    form["words"] = form["words"] + " - sell at 4,123.45 or 2050"            # his words may hold a price
    _write(ideas / "idea-002" / "form.json", json.dumps(form, indent=1, ensure_ascii=False))
    with open(ideas / "idea-002" / "decision.md", "a", encoding="utf-8") as fh:
        fh.write("\n**The machine's choice: KEPT ON FILE at 2,650.10**\n")
    _write(ideas / "idea-003" / "state.json", json.dumps({
        "idea_id": "idea-003", "state": "RECEIVED", "since": _iso(T - dt.timedelta(hours=5)),
        "waits_for": "the machine - gold was 4,321.99 then", "history": [],
        "intake": {"words": "buy when gold breaks 3999.75 " + "x" * 10, "source": "Telegram message 1", "at": _iso(T)}}))


def host_files(root: Path, *, plant: bool = True) -> None:
    """A day of the host's own files: hourly beats with one missed stretch, the daily runs, alerts, the intake, the
    bot, the nightly copy - with planted prices in every free text."""
    p = (lambda s: s) if plant else (lambda s: re.sub(r"\d,?\d{3}\.\d+", "", s))
    host, tg = root / "data" / "host", root / "data" / "telegram"
    beats, t = [], T.replace(minute=15) - dt.timedelta(hours=30)
    while t < T:
        if not (T - dt.timedelta(hours=14) < t < T - dt.timedelta(hours=6)):       # the PC was off at night
            rec = {"at": _iso(t), "bot": True, "disk_free_gb": 130.5, "problems": [], "ping": p("failed (4,123.45)"),
                   "sent": []}
            if beats and (t - dt.datetime.fromisoformat(beats[-1]["at"])) > dt.timedelta(hours=2):
                rec["missed"] = {"since": beats[-1]["at"], "minutes": 480, "offs": [
                    {"off": _iso(t - dt.timedelta(hours=7)), "on": _iso(t - dt.timedelta(minutes=20)),
                     "why": p("the power went off - no proper shutdown (4123.45)")}]}
                rec["sent"] = ["missed"]
            beats.append(rec)
        t += dt.timedelta(hours=1)
    _jsonl(host / "heartbeat.jsonl", beats)
    steps = ["prices", "caches", "ideas", "questions", "mlflow"]
    ok = [{"step": s, "ok": True, "line": f"{s}: fine, 1,320 one-minute rows", "seconds": 1.0} for s in steps]
    bad = [dict(x) for x in ok]
    bad[0] = {"step": "prices", "ok": False, "line": p("Prices: FAILED - MT5 said close 4,123.45"), "seconds": 2.0}
    _jsonl(host / "daily.jsonl", [
        {"at": _iso(T - dt.timedelta(days=1, hours=6)), "ok": True, "forced": None, "steps": ok},
        {"at": _iso(T - dt.timedelta(hours=6)), "ok": False, "forced": None, "steps": bad},
        {"at": _iso(T - dt.timedelta(hours=5)), "ok": False, "forced": "mlflow", "steps": ok}])
    _write(host / "alerts.json", json.dumps({"open": {"live": {"since": _iso(T - dt.timedelta(hours=2)),
                                                                 "sent": True}}}))
    m = T - dt.timedelta(hours=5, minutes=30)
    _write(host / "alerts.log", "".join(f"{_log(m)} {x}\n" for x in (
        "sent (missed): ⚠️ The heartbeat was missed: no beat from Mon 23:15 to Tue 08:00 (8 hours 45 min).",
        "  Windows says the PC was off: Mon 23:47 to Tue 07:57, the power went off.",
        p("sent (live-preview): ⚠️ The live alert for ISM services at 19:30 India time will probably FAIL "
          "(KeyError: close 4,123.45 and 4123.45)"),
        p("NOT SENT (he typed STOP): Gold now: 2,650.10 (made-up example), since the number +12.34"))))
    _jsonl(host / "unsent.jsonl", [{"at": _iso(T), "kind": "x", "text": p("Gold 4,321.99 now"), "why": "URLError"}])
    _write(root / "data" / "intake" / "intake.log", "".join(f"{_log(T - dt.timedelta(hours=h))} {x}\n" for h, x in (
        (4, "idea-003 RECEIVED from the queue"),
        (3, p("idea-003: the step after RECEIVED stopped: ValueError: level 2650.5 not found\nTraceback (most recent "
              "call last):\n  File x\nValueError: level 2650.5")))))
    _write(tg / "state.json", json.dumps({"offset": 7, "sending": True}))
    _jsonl(tg / "nightly.jsonl", [
        {"at": _iso(T - dt.timedelta(days=1, hours=14)), "branch": "bot/2026-10-04", "files": [{"path": "a"}],
         "pr": None, "link": "https://github.com/x/y/compare/a", "error": None, "dry": False},
        {"at": _iso(T - dt.timedelta(hours=14)), "branch": "bot/2026-10-05", "files": [], "pr": None, "link": None,
         "error": p("push refused: 4,123.45"), "dry": False}])
    _write(tg / "bot.log", "".join(f"{_iso(T - dt.timedelta(hours=h))} {x}\n" for h, x in (
        (6, "bot started (running); files: data/telegram"),
        (3, "Telegram not reachable (network: getUpdates: timed out); trying again in 5 s"),
        (2, f"idea: '{HIS_WORDS} buy at 4,123.45' -> {{'filed': 'x'}}"))))


def fake_machine(tmp_path: Path, monkeypatch, *, plant: bool = True) -> Path:
    root = at_root(tmp_path / "machine", monkeypatch)
    ideas_and_trials(root)
    host_files(root, plant=plant)
    poison(root)
    return root


_OPENED: list | None = None
_HOOKED = False


def _hook(event, args):
    if _OPENED is not None and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.fsdecode(args[0]))


@contextmanager
def opened_files():
    """Every file opened inside the block (Python's audit hook: it sees every open, whoever asks)."""
    global _OPENED, _HOOKED
    if not _HOOKED:
        sys.addaudithook(_hook)
        _HOOKED = True
    seen: list[str] = []
    _OPENED = seen
    try:
        yield seen
    finally:
        _OPENED = None


def everything(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir")
            for p in sorted(root.rglob("*"))}


def visible(doc: str) -> str:
    """The page's words, without its style sheet and tags."""
    doc = re.sub(r"<style>.*?</style>", " ", doc, flags=re.S)
    return re.sub(r"<[^>]+>", " ", doc)


def planted_in(text: str) -> list[str]:
    found = [x for x in PLANTED if re.search(r"(?<![\d,.])" + re.escape(x) + r"(?![\d])", text)]
    return found + (["his 2050"] if re.search(r"(?<![\d,.])2050(?!\d)", text) else [])


# ============================================================================================================ the masks

def test_every_price_shape_is_hidden_and_the_rest_stays():
    for s in ("4,123.45", "4123.45", "2,650", "2650.5", "4123", "1999", "12,345.6"):
        assert mask.prices(f"gold {s} now") == "gold #### now", s
    kept = ("total -₹21,661, worst year 2020 -₹12,003", "on 2016-2020 and 2021-23", "v-a0708fb6bd v-8331245570",
            "idea-003", "1,320 one-minute rows", "34.3 MB", "182 trial files, 1,610 passed", "932.10", "13:05")
    for s in kept:
        assert mask.prices(s) == s, s
    assert mask.prices("buy at 2050 or 2020", years=False) == "buy at #### or ####"     # his words: no year kept
    assert mask.prices("in 2024 at 4,123.45") == "in 2024 at ####"


def test_every_digit_goes_except_clock_times():
    out = mask.every_digit("no beat from Mon 23:15 to Tue 07:57, gold 4,123.45 at 19:30; +12.34; 1:4; 2026-10-06")
    assert out == "no beat from Mon 23:15 to Tue 07:57, gold #,###.## at 19:30; +##.##; #:#; ####-##-##"
    assert mask.every_digit("23:15", times=False) == "##:##"


# ============================================================================================================ the page

@pytest.mark.private
def test_the_page_shows_no_planted_price_and_nothing_private(tmp_path, monkeypatch):
    fake_machine(tmp_path, monkeypatch)
    snap = read.snapshot(T)
    doc = page.html_page(snap)
    words = visible(doc)
    for text in (words, json.dumps(snap, ensure_ascii=False)):
        assert planted_in(text) == []
        assert HIS_WORDS not in text
    # ... and it still says what it must (so the test above is not empty)
    for need in ("idea #1", "idea #2", "idea #3", "DECIDED", "RECEIVED", "18 of 20", "7 of 20", "CHANCE",
                 "KEPT ON FILE", "his whole strategy", "1,320 one-minute rows", "19:30", "####"):
        assert need in words, need
    assert "Then: KEPT ON FILE at ####" in words                     # decision.md, masked
    assert "sell at #### or ####" in words                           # his words: a plain 2050 is hidden too


@pytest.mark.private
def test_it_opens_only_its_list_of_files(tmp_path, monkeypatch):
    root = fake_machine(tmp_path, monkeypatch)
    with opened_files() as seen:
        page.html_page(read.snapshot(T))
    mine = [Path(p).resolve() for p in seen if Path(p).resolve().is_relative_to(root.resolve())]
    assert mine, "the page read nothing of the machine"
    assert [str(p) for p in mine if not read.allowed(p)] == []
    names = {p.relative_to(root.resolve()).as_posix() for p in mine}
    for never in ("data/live/", "data/sealed/", "plan/work/telegram-outbox.jsonl", "data/telegram/sent.jsonl",
                  "data/telegram/messages.jsonl", ".env"):
        assert not any(n.startswith(never) for n in names), never
    assert {"data/host/heartbeat.jsonl", "data/host/daily.jsonl", "data/host/alerts.log",
            "plan/work/ideas/idea-002/state.json"} <= names


@pytest.mark.private
def test_it_writes_nothing(tmp_path, monkeypatch):
    root = fake_machine(tmp_path, monkeypatch)
    before = everything(root)
    page.html_page(read.snapshot(T))
    with serving() as base:
        for path in ("/", "/status.json", "/health"):
            urllib.request.urlopen(base + path, timeout=30).read()
    assert everything(root) == before


@pytest.mark.private
def test_the_six_boxes_say_what_happened(tmp_path, monkeypatch):
    fake_machine(tmp_path, monkeypatch)
    boxes = {x["name"]: x for x in read.snapshot(T)["now"]}
    assert boxes["Heartbeat"]["state"] == "good" and "13:15" in boxes["Heartbeat"]["value"]
    assert boxes["Telegram bot"]["state"] == "good"
    assert boxes["Daily run"]["state"] == "bad" and "Prices kept" in boxes["Daily run"]["value"]
    assert boxes["Live news alerts"]["state"] == "bad"                # the heartbeat's open problem "live"
    assert boxes["Nightly copy to GitHub"]["state"] == "bad"
    assert boxes["Open problems"]["state"] == "bad"
    snap = read.snapshot(T)
    assert snap["heartbeat"]["missed"] and snap["heartbeat"]["missed"][0]["minutes"] == 480
    kinds = [c["kind"] for c in snap["heartbeat"]["strip"]]
    assert len(kinds) == read.STRIP_HOURS and "none" in kinds and kinds[-1] == "beat"
    assert any(e["where"] == "message sent to you" and "19:30" in e["text"] for e in snap["errors"])
    assert any(e["where"] == "ideas" and "stopped" in e["text"] for e in snap["errors"])


@pytest.mark.private
def test_one_broken_idea_file_spoils_only_its_own_row(tmp_path, monkeypatch):
    root = at_root(tmp_path / "m", monkeypatch)
    ideas_and_trials(root)
    _write(root / "plan" / "work" / "ideas" / "idea-003" / "state.json", "{ half a file")
    snap = read.snapshot(T)
    rows = {r["id"]: r for r in snap["ideas"]["rows"]}
    assert rows["idea-003"]["state"] == "?" and "cannot be read" in rows["idea-003"]["waits"]
    assert rows["idea-001"]["state"] == "DECIDED" and rows["idea-002"]["state"] == "DECIDED"
    assert [r["tried"] for r in snap["versions"]["rows"]][:2] == [18, 7]
    assert "Could not be read" not in page.html_page(snap)


def test_an_empty_machine_says_so_and_does_not_break(tmp_path, monkeypatch):
    at_root(tmp_path / "empty", monkeypatch)
    snap = read.snapshot(T)
    assert {x["state"] for x in snap["now"]} == {"unknown"}
    doc = page.html_page(snap)
    for need in ("No heartbeat on this computer", "No daily run on this computer", "No idea yet"):
        assert need in doc, need


def test_the_server_shows_the_last_reading_at_once_and_reads_again_behind(tmp_path, monkeypatch):
    at_root(tmp_path / "m", monkeypatch)
    monkeypatch.setenv("DASHBOARD_FILES", "C:/ML Bot")
    first = read.latest()                                     # none yet: read now
    assert 'from the files in "C:/ML Bot"' in page.html_page(first)
    assert read.latest() is first                             # fresh: the same reading
    read._cache["at"] -= read.CACHE_SECONDS + 1               # older: shown at once, a new one made behind
    assert read.latest() is first
    with read._making:                                        # held until the reading behind is made
        pass
    second = read.latest()
    assert second is not first
    read._cache["at"] -= read.TOO_OLD_SECONDS + 1             # nobody looked for 5 minutes: read now
    assert read.latest() is not second


def test_a_late_heartbeat_a_stopped_bot_and_stop_are_told(tmp_path, monkeypatch):
    root = at_root(tmp_path / "m", monkeypatch)
    _jsonl(root / "data" / "host" / "heartbeat.jsonl", [{"at": _iso(T - dt.timedelta(hours=3)), "bot": False,
                                                         "problems": ["bot"], "sent": []}])
    boxes = {x["name"]: x for x in read.snapshot(T)["now"]}
    assert boxes["Heartbeat"]["state"] == "bad" and "LATE" in boxes["Heartbeat"]["value"]
    assert boxes["Telegram bot"]["state"] == "bad"
    _jsonl(root / "data" / "host" / "heartbeat.jsonl", [{"at": _iso(T - dt.timedelta(minutes=20)), "bot": True}])
    _write(root / "data" / "telegram" / "state.json", json.dumps({"sending": False}))
    boxes = {x["name"]: x for x in read.snapshot(T)["now"]}
    assert boxes["Telegram bot"]["state"] == "warning" and "STOP" in boxes["Telegram bot"]["line"]


def test_the_daily_box_waits_until_its_time_then_says_not_run(tmp_path, monkeypatch):
    root = at_root(tmp_path / "m", monkeypatch)
    yesterday = {"at": _iso(T - dt.timedelta(days=1)), "ok": True, "forced": None,
                 "steps": [{"step": "prices", "ok": True, "line": "Prices: fine"}]}
    _jsonl(root / "data" / "host" / "daily.jsonl", [yesterday])
    early = T.replace(hour=8, minute=0)
    assert {x["name"]: x for x in read.snapshot(early)["now"]}["Daily run"]["state"] == "waiting"
    assert {x["name"]: x for x in read.snapshot(T)["now"]}["Daily run"]["state"] == "bad"     # 13:30, not run


def test_the_page_is_in_easy_english(tmp_path, monkeypatch):
    """LEARNED #113: code words he never used are banned on what he reads."""
    at_root(tmp_path / "m", monkeypatch)
    host_files(tmp_path / "m", plant=False)
    words = visible(page.html_page(read.snapshot(T)))
    for banned in ("leftover", "arm", "verdict", "veto", "causal", "placebo", "mirror", "baseline", "basket", "R"):
        assert not re.search(rf"\b{banned}\b", words), banned


# ============================================================================================================ the server

@contextmanager
def serving():
    from machine.dashboard.__main__ import server
    srv = server("127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _status(url: str, method: str = "GET") -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, method=method, data=b"x" if method in ("POST", "PUT", "PATCH") else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def test_the_server_answers_get_only_and_serves_no_file(tmp_path, monkeypatch):
    at_root(tmp_path / "m", monkeypatch)
    with serving() as base:
        code, head, body = _status(base + "/")
        assert code == 200 and head["Content-Type"].startswith("text/html")
        assert "default-src 'none'" in head["Content-Security-Policy"] and head["X-Frame-Options"] == "DENY"
        assert b"<script" not in body
        code, head, body = _status(base + "/status.json")
        assert code == 200 and "now" in json.loads(body)
        assert _status(base + "/health")[2] == b"ok\n"
        code, _, body = _status(base + "/", "HEAD")
        assert code == 200 and body == b""
        for path in ("/x", "/.env", "/../.env", "/plan/work/telegram-outbox.jsonl", "/data/live/state.json"):
            assert _status(base + path)[0] == 404, path
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            code, head, _ = _status(base + "/", method)
            assert code == 405 and head.get("Allow") == "GET, HEAD", method


# ============================================================================================================ Docker

def _root_file(name: str) -> Path:
    path = REPO / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this copy")
    return path


def test_compose_gives_the_dashboard_only_its_folders_read_only():
    yaml = pytest.importorskip("yaml")
    c = yaml.safe_load(_root_file("compose.yaml").read_text(encoding="utf-8"))
    s = c["services"]
    d = s["dashboard"]
    mounts = [v.replace("${MACHINE_HOME:-.}/", "") for v in d["volumes"]]
    assert mounts == ["plan/work:/app/plan/work:ro", "data/host:/app/data/host:ro", "data/intake:/app/data/intake:ro",
                      "data/telegram:/app/data/telegram:ro"]
    assert d["read_only"] is True and d["cap_drop"] == ["ALL"] and "no-new-privileges:true" in d["security_opt"]
    assert all(p.startswith("127.0.0.1:") for p in d["ports"])
    assert s["tests"]["network_mode"] == "none" and s["demo"]["network_mode"] == "none"
    text = _root_file("compose.yaml").read_text(encoding="utf-8")
    vols = "\n".join(d["volumes"])
    for never in ("data/live", "data/sealed", ".env", "old/build", "data/cache", "mlflow"):
        assert never not in vols, never
    assert "telegram.bot" not in text and "machine.live" not in text and "machine.host" not in text


@pytest.mark.private                     # the public copy keeps other files at its root (machine/tools/publish.py)
def test_dockerignore_lets_in_only_what_git_keeps():
    lines = [x.strip() for x in _root_file(".dockerignore").read_text(encoding="utf-8").splitlines()]
    rules = [x for x in lines if x and not x.startswith("#")]
    assert rules[0] == "*"                                          # an allow-list: everything out first
    let_in = [x[1:].rstrip("/") for x in rules if x.startswith("!")]
    assert "data" not in let_in and not any(x.startswith("data/") and not x.endswith(".csv") for x in let_in)
    for out in ("**/.env", "**/*.env", "old/build/data", "old/ml/agent/agent_config.json", "old/ml/agent/state.json",
                "old/ml/agent/clock.json"):
        assert out in rules, out
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no git here")
    tops = {p.split("/")[0] for p in tracked.splitlines() if p and not p.startswith("data/")}
    tops |= {p for p in tracked.splitlines() if p.startswith("data/")}
    assert tops <= set(let_in), sorted(tops - set(let_in))         # every tracked file is let in
    assert set(let_in) - tops - {"Dockerfile", "compose.yaml", ".dockerignore", "PRIVATE-PARTS.md"} == set()
    # ... and nothing else (PRIVATE-PARTS.md: the public copy's marker - its tests skip what it leaves out)


@pytest.mark.private
def test_the_real_page_shows_no_price_shaped_number(monkeypatch):
    """This copy's own files (on Tejas's PC: the running machine's): nothing left on the page that looks like a gold
    price - every number there is rupees, a year, a count, a size or a time (the mask has nothing left to hide)."""
    for name, value in (("IDEAS_DIR", REAL_WORK / "ideas"), ("TRIALS_DIR", REAL_WORK / "trials")):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(read, "_cache", {})
    words = visible(page.html_page(read.snapshot()))
    words = words.replace("&quot;", '"').replace("&#x27;", "'").replace("&amp;", "&")
    assert mask.prices(words) == words
