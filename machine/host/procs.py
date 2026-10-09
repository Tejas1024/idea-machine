"""Which of the machine's programs are running (L16.1): the Telegram bot, intake runs.

One question to Windows (Get-CimInstance Win32_Process) for every python program with its command line and start time.
The .venv's pythonw.exe starts the real Python as its child with the same command line (seen 4 Oct 2026: the bot was
pids 6732 -> 11880), so one program can be two processes: a child whose parent has the same command line is not
counted again.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from dataclasses import dataclass

PS = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
      "Select-Object ProcessId, ParentProcessId, CommandLine, "
      "@{n='Created';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress")


@dataclass
class Proc:
    pid: int
    parent: int
    command: str
    started: dt.datetime | None


def processes() -> list[Proc] | None:
    """Every python program now. None = Windows could not be asked (off Windows, a time-out): then nobody may say "the
    bot is not running". A real answer is never empty - the program asking is a python program itself."""
    if os.name != "nt":
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS],
                              capture_output=True, timeout=60, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = done.stdout.decode("utf-8", "replace").strip()
    try:
        rows = json.loads(text) if text else None
    except ValueError:
        rows = None
    if not rows:
        return None
    rows = rows if isinstance(rows, list) else [rows]
    out = []
    for r in rows:
        started = None
        if r.get("Created"):
            try:
                started = dt.datetime.fromisoformat(str(r["Created"])[:26].rstrip("Z")).replace(tzinfo=dt.timezone.utc)
            except ValueError:
                started = None
        out.append(Proc(int(r.get("ProcessId") or 0), int(r.get("ParentProcessId") or 0),
                        str(r.get("CommandLine") or ""), started))
    return out


def programs(procs: list[Proc], marker: str, *, without: str | None = None) -> list[Proc]:
    """The programs whose command line holds `marker` (one per launcher + child pair), oldest first."""
    found = [p for p in procs if marker in p.command and not (without and without in p.command)]
    by_pid = {p.pid: p for p in found}
    roots = [p for p in found if not (p.parent in by_pid and _same(by_pid[p.parent].command, p.command))]
    return sorted(roots, key=lambda p: p.started or dt.datetime.max.replace(tzinfo=dt.timezone.utc))


def _same(a: str, b: str) -> bool:
    """The same command apart from the program's path (the launcher's path differs from the real Python's)."""
    def rest(c: str) -> str:
        c = c.strip()
        if c.startswith('"'):
            return c[c.find('"', 1) + 1:].strip()
        return c.split(" ", 1)[1].strip() if " " in c else ""
    return rest(a) == rest(b)


BOT = "-m machine.telegram.bot"
INTAKE = "-m machine.intake run"


def bot(procs: list[Proc] | None = None) -> list[Proc] | None:
    """The real bot (never the test bot: --test). None = Windows could not be asked."""
    procs = processes() if procs is None else procs
    return None if not procs else programs(procs, BOT, without="--test")


def intake(procs: list[Proc] | None = None) -> list[Proc] | None:
    procs = processes() if procs is None else procs
    return None if not procs else programs(procs, INTAKE)
