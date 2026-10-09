"""The Task Scheduler job of the live news alerts (L17a, Windows only), like L16.1's (machine/host/tasks.py):

    "ML Bot - live news alerts (L17a)"   `pythonw -m machine.live once` every config.LIVE_EVERY_MINUTES minute(s) from
                                         the next minute, and 2 minutes after he logs in; one copy at a time
                                         (IgnoreNew; once.lock too); a pass that runs longer than 5 minutes is stopped

    python -m machine.live install      add (or replace) it
    python -m machine.live commands     print the PowerShell lines that do the same (to paste by hand)
    python -m machine.live uninstall    remove it
"""
from __future__ import annotations

import os
from pathlib import Path

from machine import config
from machine.host.tasks import TaskError, _first, _powershell, _ps


def script(root: Path | None = None) -> str:
    root = Path(root or config.ROOT)
    exe = root / ".venv" / "Scripts" / "pythonw.exe"
    return "\n".join([
        f"$exe = {_ps(exe)}; $dir = {_ps(root)}",
        "$a = New-ScheduledTaskAction -Execute $exe -Argument '-m machine.live once' -WorkingDirectory $dir",
        "$at = (Get-Date).AddMinutes(1); $at = $at.Date.AddHours($at.Hour).AddMinutes($at.Minute)",
        f"$t1 = New-ScheduledTaskTrigger -Once -At $at -RepetitionInterval (New-TimeSpan -Minutes {config.LIVE_EVERY_MINUTES})",
        "$t2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME; $t2.Delay = 'PT2M'",
        "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)",
        f"Register-ScheduledTask -TaskName {_ps(config.LIVE_TASK_NAME)} -Action $a -Trigger @($t1, $t2) -Settings $s "
        f"-Force -Description {_ps('The live news alerts: 30 min before his red/orange news and when its number is out (plan/likhith.md L17a).')} | Out-Null",
        f"$i = Get-ScheduledTask -TaskName {_ps(config.LIVE_TASK_NAME)} | Get-ScheduledTaskInfo; "
        f"{_ps(config.LIVE_TASK_NAME + ' next runs ')} + $i.NextRunTime.ToString('yyyy-MM-dd HH:mm')",
    ])


def install(root: Path | None = None) -> str:
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
    _powershell(f"Unregister-ScheduledTask -TaskName {_ps(config.LIVE_TASK_NAME)} -Confirm:$false "
                "-ErrorAction SilentlyContinue")


def status() -> str:
    if os.name != "nt":
        return "Task Scheduler is on Windows only"
    ps = (f"$t = Get-ScheduledTask -TaskName {_ps(config.LIVE_TASK_NAME)} -ErrorAction SilentlyContinue; "
          "if ($t) { $i = $t | Get-ScheduledTaskInfo; '{0}: {1}, last {2} (result {3}), next {4}' -f $t.TaskName, "
          f"$t.State, $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime }} else {{ {_ps(config.LIVE_TASK_NAME + ': NOT installed')} }}")
    done = _powershell(ps)
    return done.stdout.decode("utf-8", "replace").strip() or _first(done)
