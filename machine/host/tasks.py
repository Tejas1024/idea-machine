"""The two Task Scheduler jobs of L16.1 (Windows only), like the nightly job's (L9c) and the label job's (L15.1):

    "ML Bot - daily run (L16.1)"    every day at config.DAILY_AT; a day missed while the PC was off runs as soon as it
                                    is on again (StartWhenAvailable); it may wake a sleeping PC (WakeToRun)
    "ML Bot - heartbeat (L16.1)"    every hour at :15, and 2 minutes after he logs in (a power cut ends with a restart)

Both run pythonw (no window) from the repo's .venv, in the repo's folder, as him, when he is logged in - the same as the
bot's own task. One copy at a time (IgnoreNew; the jobs' own locks too).

    python -m machine.host install      add (or replace) both
    python -m machine.host commands     print the PowerShell lines that do the same (to paste by hand)
    python -m machine.host uninstall    remove both
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from machine import config


class TaskError(RuntimeError):
    pass


def _ps(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def script(root: Path | None = None) -> str:
    """The PowerShell that registers both jobs (what install() runs)."""
    root = Path(root or config.ROOT)
    exe = root / ".venv" / "Scripts" / "pythonw.exe"
    if not re.fullmatch(r"\d{1,2}:\d{2}", config.DAILY_AT):
        raise TaskError(f"config.DAILY_AT wants a time like 07:00, not {config.DAILY_AT!r}")
    delay = f"PT{config.HEARTBEAT_LOGON_DELAY_MINUTES}M"
    every = config.HEARTBEAT_EVERY_MINUTES
    return "\n".join([
        f"$exe = {_ps(exe)}; $dir = {_ps(root)}",
        # the daily run
        "$a = New-ScheduledTaskAction -Execute $exe -Argument '-m machine.host daily' -WorkingDirectory $dir",
        f"$t = New-ScheduledTaskTrigger -Daily -At {_ps(config.DAILY_AT)}",
        "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)",
        f"Register-ScheduledTask -TaskName {_ps(config.DAILY_TASK_NAME)} -Action $a -Trigger $t -Settings $s -Force "
        f"-Description {_ps('The daily run: prices (sealed), caches, ideas, weekly questions, MLflow, a summary (plan/likhith.md L16.1).')} | Out-Null",
        # the heartbeat: every hour at :15 from the next :15, and at logon (+ a delay for the network)
        "$at = (Get-Date).Date.AddHours((Get-Date).Hour).AddMinutes(15); if ($at -lt (Get-Date)) { $at = $at.AddHours(1) }",
        "$a = New-ScheduledTaskAction -Execute $exe -Argument '-m machine.host heartbeat' -WorkingDirectory $dir",
        f"$t1 = New-ScheduledTaskTrigger -Once -At $at -RepetitionInterval (New-TimeSpan -Minutes {every})",
        f"$t2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME; $t2.Delay = {_ps(delay)}",
        "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15)",
        f"Register-ScheduledTask -TaskName {_ps(config.HEARTBEAT_TASK_NAME)} -Action $a -Trigger @($t1, $t2) "
        f"-Settings $s -Force -Description {_ps('The hourly heartbeat: a missed one, a stopped bot or a failed daily run alerts Tejas (plan/likhith.md L16.1).')} | Out-Null",
        f"foreach ($n in @({_ps(config.DAILY_TASK_NAME)}, {_ps(config.HEARTBEAT_TASK_NAME)})) "
        "{ $i = Get-ScheduledTask -TaskName $n | Get-ScheduledTaskInfo; \"$n next runs \" + $i.NextRunTime.ToString('yyyy-MM-dd HH:mm') }",
    ])


def install(root: Path | None = None) -> str:
    """Adds (or replaces) both jobs. Returns their next run times."""
    if os.name != "nt":
        raise TaskError("Task Scheduler is on Windows only")
    exe = Path(root or config.ROOT) / ".venv" / "Scripts" / "pythonw.exe"
    if not exe.is_file():
        raise TaskError(f"{exe} is missing (make the .venv first: docs/SETUP.md)")
    done = _powershell(script(root))
    if done.returncode:
        raise TaskError("Task Scheduler said no: " + _first(done))
    return done.stdout.decode("utf-8", "replace").strip()


def uninstall() -> None:
    for name in (config.DAILY_TASK_NAME, config.HEARTBEAT_TASK_NAME):
        _powershell(f"Unregister-ScheduledTask -TaskName {_ps(name)} -Confirm:$false -ErrorAction SilentlyContinue")


def status() -> list[str]:
    """One line per job: its state, last run and result, next run (or that it is not installed)."""
    if os.name != "nt":
        return ["Task Scheduler is on Windows only"]
    out = []
    for name in (config.DAILY_TASK_NAME, config.HEARTBEAT_TASK_NAME):
        ps = (f"$t = Get-ScheduledTask -TaskName {_ps(name)} -ErrorAction SilentlyContinue; "
              "if ($t) { $i = $t | Get-ScheduledTaskInfo; "
              "'{0}: {1}, last {2} (result {3}), next {4}' -f $t.TaskName, $t.State, $i.LastRunTime, $i.LastTaskResult, "
              f"$i.NextRunTime }} else {{ {_ps(name + ': NOT installed')} }}")
        done = _powershell(ps)
        out.append(done.stdout.decode("utf-8", "replace").strip() or _first(done))
    return out


def _powershell(command: str) -> subprocess.CompletedProcess:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                          capture_output=True, creationflags=flags)


def _first(done: subprocess.CompletedProcess) -> str:
    text = (done.stderr or b"").decode("utf-8", "replace") + (done.stdout or b"").decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[0] if lines else f"exit code {done.returncode}")[:300]
