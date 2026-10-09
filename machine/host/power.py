"""What Windows says about restarts (L16.1): when the PC went off, when it came back, and why - in plain words.

Read from the System event log (wevtutil, no extra package), only these events:
    Kernel-General 12      the PC started (its StartTime)
    Kernel-Power 41        it started without a proper shutdown first: a crash (BugcheckCode), the power button held
                           (PowerButtonTimestamp / LongPowerButtonPressDetected), else the power went off
    EventLog 6008          "the previous shutdown was unexpected": its Binary holds the last moment the PC was alive,
                           as two SYSTEMTIMEs (India time, then UTC) - measured 4 Oct 2026 on his PC: 22:50:22 / 17:20:22
    EventLog 6006          a proper shutdown;  User32 1074: who asked for it ("power off" / "restart", which program)
    Kernel-Power 42        going to sleep - TargetState 4 sleep, 5 hibernate, 6 shut down with fast startup (his PC on
                           30 Sep: 1074 "power off" by winlogon, then 42 with TargetState 6)
    Kernel-Power 107, Power-Troubleshooter 1   back from sleep (1 has SleepTime and WakeTime)
"""
from __future__ import annotations

import datetime as dt
import os
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
WANTED = (("Microsoft-Windows-Kernel-General", (12,)),
          ("Microsoft-Windows-Kernel-Power", (41, 42, 107)),
          ("Microsoft-Windows-Power-Troubleshooter", (1,)),
          ("User32", (1074,)),
          ("EventLog", (6006, 6008)))
AFTER_BOOT = dt.timedelta(minutes=10)      # 41 and 6008 are written while the PC starts: this soon after it
BEFORE_DOWN = dt.timedelta(minutes=10)     # 1074 (who asked) comes this soon before the shutdown or sleep
UPDATE_PROGRAMS = ("trustedinstaller.exe", "mousocoreworker.exe", "usoclient.exe", "wuauclt.exe")


@dataclass
class Event:
    provider: str
    id: int
    at: dt.datetime                         # UTC
    data: dict[str, str] = field(default_factory=dict)
    binary: bytes = b""


@dataclass
class Off:
    """One time the PC was not running: from off_at (None = Windows did not say) to on_at, and why."""
    off_at: dt.datetime | None
    on_at: dt.datetime
    why: str

    def minutes(self) -> float | None:
        return None if self.off_at is None else (self.on_at - self.off_at).total_seconds() / 60


# ================================================================================================= reading the log

def xpath(since: dt.datetime) -> str:
    t = since.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    kinds = " or ".join(f"(Provider[@Name='{p}'] and ({' or '.join(f'EventID={i}' for i in ids)}))"
                        for p, ids in WANTED)
    return f"*[System[TimeCreated[@SystemTime>='{t}']] and System[{kinds}]]"


def query(since: dt.datetime) -> str:
    """The events since `since` as one XML text ('' off Windows or when wevtutil fails)."""
    if os.name != "nt":
        return ""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(["wevtutil", "qe", "System", f"/q:{xpath(since)}", "/f:xml", "/e:Events"],
                              capture_output=True, timeout=60, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if done.returncode:
        return ""
    raw = done.stdout
    for enc in ("utf-8", "utf-16", "mbcs"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def parse(text: str) -> list[Event]:
    if not text.strip():
        return []
    text = text.lstrip("﻿")
    if not text.lstrip().startswith("<Events"):
        text = f"<Events>{text}</Events>"
    root = ET.fromstring(text)
    out = []
    for ev in root.iter(f"{{{NS['e']}}}Event"):
        sysn = ev.find("e:System", NS)
        if sysn is None:
            continue
        prov = sysn.find("e:Provider", NS)
        eid = sysn.find("e:EventID", NS)
        tc = sysn.find("e:TimeCreated", NS)
        if prov is None or eid is None or tc is None:
            continue
        data, binary = {}, b""
        ed = ev.find("e:EventData", NS)
        if ed is not None:
            for i, d in enumerate(ed.findall("e:Data", NS)):
                data[d.get("Name") or f"#{i}"] = (d.text or "").strip()
            b = ed.find("e:Binary", NS)
            if b is not None and b.text:
                try:
                    binary = bytes.fromhex(b.text.strip())
                except ValueError:
                    binary = b""
        out.append(Event(prov.get("Name", ""), int(eid.text), utc(tc.get("SystemTime", "")), data, binary))
    return sorted(out, key=lambda e: e.at)


def utc(text: str) -> dt.datetime:
    """'2026-10-03T07:37:30.3669430Z' -> an aware UTC time (Windows writes 7 decimals; Python reads 6)."""
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?", text)
    if not m:
        raise ValueError(f"not an event time: {text!r}")
    frac = (m.group(2) or "0")[:6].ljust(6, "0")
    return dt.datetime.fromisoformat(f"{m.group(1)}.{frac}").replace(tzinfo=dt.timezone.utc)


def systemtime(raw: bytes, at: int = 0) -> dt.datetime | None:
    """A Windows SYSTEMTIME (8 little-endian 16-bit numbers) at byte `at`, as a naive time."""
    if len(raw) < at + 16:
        return None
    y, mo, _, d, h, mi, s, ms = struct.unpack_from("<8H", raw, at)
    try:
        return dt.datetime(y, mo, d, h, mi, s, ms * 1000)
    except ValueError:
        return None


def last_alive(ev: Event) -> dt.datetime | None:
    """6008: the last moment the PC was alive - the SECOND SYSTEMTIME is UTC."""
    t = systemtime(ev.binary, 16)
    return t.replace(tzinfo=dt.timezone.utc) if t else None


# ================================================================================================= the story

def periods(events: list[Event]) -> list[Off]:
    """Each time the PC stopped, as Off(off_at, on_at, why), oldest first."""
    ev = sorted(events, key=lambda e: e.at)
    out: list[Off] = []
    boots = [e for e in ev if (e.provider, e.id) == ("Microsoft-Windows-Kernel-General", 12)]
    prev_boot = None
    for b in boots:
        on = utc(b.data["StartTime"]) if b.data.get("StartTime") else b.at
        near = [e for e in ev if b.at <= e.at <= b.at + AFTER_BOOT]
        k41 = next((e for e in near if (e.provider, e.id) == ("Microsoft-Windows-Kernel-Power", 41)), None)
        e6008 = next((e for e in near if (e.provider, e.id) == ("EventLog", 6008)), None)
        if k41 or e6008:
            out.append(Off(last_alive(e6008) if e6008 else None, on, _unexpected(k41)))
        else:
            down = [e for e in ev if (e.provider, e.id) == ("EventLog", 6006) and e.at < b.at
                    and (prev_boot is None or e.at > prev_boot.at)]
            if down:
                d = down[-1]
                out.append(Off(d.at, on, _asked(ev, d.at, "shut down")))
            else:
                out.append(Off(None, on, "it restarted - Windows did not say why"))
        prev_boot = b
    sleeps = [e for e in ev if (e.provider, e.id) == ("Microsoft-Windows-Kernel-Power", 42)]
    for n, s in enumerate(sleeps):
        until = sleeps[n + 1].at if n + 1 < len(sleeps) else None      # a wake belongs to THIS sleep only
        nxt = [e for e in ev if e.at > s.at and (until is None or e.at < until)
               and (e.provider, e.id) in (("Microsoft-Windows-Kernel-Power", 107),
                                          ("Microsoft-Windows-Power-Troubleshooter", 1),
                                          ("Microsoft-Windows-Kernel-General", 12))]
        if not nxt or (nxt[0].provider, nxt[0].id) == ("Microsoft-Windows-Kernel-General", 12):
            continue                                   # no wake before the next start: that start tells the story
        wake = next((e for e in nxt if e.provider == "Microsoft-Windows-Power-Troubleshooter"), None)
        on = utc(wake.data["WakeTime"]) if wake and wake.data.get("WakeTime") else nxt[0].at
        target = s.data.get("TargetState", "")
        if target == "6":
            why = _asked(ev, s.at, "shut down") + ", with fast startup"
        elif target == "5":
            why = "it hibernated"
        else:
            why = "it went to sleep"
        out.append(Off(s.at, on, why))
    return sorted(out, key=lambda o: o.on_at)


def _unexpected(k41: Event | None) -> str:
    if k41 is None:
        return "it stopped without a proper shutdown"
    d = k41.data
    if d.get("BugcheckCode", "0") not in ("0", "", "0x0"):
        return "Windows crashed - a blue screen"
    if d.get("PowerButtonTimestamp", "0") not in ("0", "") or d.get("LongPowerButtonPressDetected") == "true":
        return "the power button was held down"
    if d.get("SleepInProgress", "0") not in ("0", ""):
        return "the power went off while it was asleep"
    return "the power went off - no proper shutdown"


def _asked(ev: list[Event], down_at: dt.datetime, what: str) -> str:
    """Who asked for a shutdown / restart (1074 just before it)."""
    asks = [e for e in ev if (e.provider, e.id) == ("User32", 1074) and down_at - BEFORE_DOWN <= e.at <= down_at]
    if not asks:
        return f"it was {what} properly"
    a = asks[-1].data
    kind = (a.get("param5") or "").lower()
    prog = re.sub(r"\s*\(.*\)\s*$", "", a.get("param1") or "").replace("/", "\\").split("\\")[-1].lower()
    verb = "restarted" if "restart" in kind else "shut down"
    if prog in UPDATE_PROGRAMS:
        return f"Windows Update {verb} it"
    if prog == "winlogon.exe":
        return f"it was {verb} from Windows (Start, Power)"
    return f"it was {verb} by {prog or 'a program'}"


def offs_since(since: dt.datetime) -> list[Off]:
    """Every time the PC stopped after `since` ([] off Windows, or when the log cannot be read)."""
    try:
        return [o for o in periods(parse(query(since - AFTER_BOOT))) if o.on_at > since]
    except (ET.ParseError, ValueError, KeyError):
        return []


def boot_time(now: dt.datetime | None = None) -> dt.datetime | None:
    """When Windows last started (None off Windows)."""
    if os.name != "nt":
        return None
    import ctypes
    now = now or dt.datetime.now(dt.timezone.utc)
    ms = ctypes.windll.kernel32.GetTickCount64
    ms.restype = ctypes.c_ulonglong
    return now - dt.timedelta(milliseconds=int(ms()))
