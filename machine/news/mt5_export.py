"""Export MT5's economic calendar (US) into data/news/ (L12).   Run:  python -m machine.news.mt5_export

🪟 Windows + MT5 only. The session may open MT5 and MetaEditor itself (Tejas, 28 Sep 2026, LEARNED #144).
READ ONLY: the script sends no order and reads no price (hard rules 1 and 10).

Steps (each printed):
 1. copy machine/news/mql5/L12_CalendarExport.mq5 into the terminal's MQL5/Scripts/ML-Bot/ and compile it with
    MetaEditor (0 errors needed);
 2. MT5 runs a script unattended only when it STARTS with a startup file, so a running terminal is closed politely
    (a normal close, never a forced kill) and started again with the script; it stays open afterwards;
 3. wait for the script's info file (written last);
 4. copy the 2015-2023 file to data/news/raw/ as it came, and the 2024+ file to data/news/sealed/ as BYTES - never
    parsed; only its line count and SHA-256 are printed (LEARNED #96: "No, don't touch those years");
 5. delete the terminal's own copies.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
from pathlib import Path

from machine import config

SCRIPT_SRC = Path(__file__).resolve().parent / "mql5" / "L12_CalendarExport.mq5"
SCRIPT_NAME = "L12_CalendarExport"
FOLDER = "ML-Bot"                     # the script's folder inside MQL5/Scripts and MQL5/Files
OPEN_FILE = "mt5_calendar_US_2015_2023.tsv"
INFO_FILE = "mt5_calendar_US_info.txt"
SEALED_FILE = "mt5_calendar_US_2024plus.tsv"
WAIT_SECONDS = 300


def data_folder(terminal: Path) -> Path:
    """The terminal's data folder: the one in %APPDATA%/MetaQuotes/Terminal whose origin.txt names its install."""
    root = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    for origin in root.glob("*/origin.txt"):
        raw = origin.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "ignore")
        if Path(text.strip().strip("﻿")).resolve() == terminal.parent.resolve():
            return origin.parent
    raise FileNotFoundError(f"no MT5 data folder under {root} belongs to {terminal.parent}")


def compile_script(terminal: Path, mql5: Path) -> Path:
    dest = mql5 / "Scripts" / FOLDER
    dest.mkdir(parents=True, exist_ok=True)
    src = dest / SCRIPT_SRC.name
    shutil.copyfile(SCRIPT_SRC, src)
    ex5 = src.with_suffix(".ex5")
    ex5.unlink(missing_ok=True)
    log = config.NEWS_RAW_DIR / "compile.log"
    editor = terminal.parent / "MetaEditor64.exe"
    log.unlink(missing_ok=True)
    # MetaEditor reads /log:"path" - quotes AFTER the colon (the repo path has a space: "C:\ML Bot").
    subprocess.run(f'"{editor}" /compile:"{src}" /log:"{log}"', check=False)
    result = [ln for ln in log.read_text(encoding="utf-16").splitlines() if ln.startswith("Result:")]
    print(f"1. compiled {src.name}: {result[-1] if result else 'no result line'}")
    if not ex5.is_file() or not result or not result[-1].startswith("Result: 0 errors"):
        raise RuntimeError(f"MetaEditor did not compile {src}; see {log}")
    return ex5


def running_terminals(terminal: Path) -> list[int]:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-Process terminal64 -ErrorAction SilentlyContinue | "
                          "ForEach-Object { \"$($_.Id)`t$($_.Path)\" }"],
                         capture_output=True, text=True, check=False).stdout
    pids = []
    for line in out.splitlines():
        pid, _, path = line.partition("\t")
        if path and Path(path).resolve() == terminal.resolve():
            pids.append(int(pid))
    return pids


def close_politely(terminal: Path) -> None:
    pids = running_terminals(terminal)
    if not pids:
        print("2. MT5 was not running")
        return
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True, check=False)   # a normal close, no /F
    for _ in range(60):
        if not running_terminals(terminal):
            print(f"2. MT5 closed normally (process {', '.join(map(str, pids))})")
            return
        time.sleep(1)
    raise RuntimeError("MT5 did not close within 60 s. Close it by hand, then run this again (never a forced kill: "
                       "a forced kill can lose the terminal's settings).")


def start_with_script(terminal: Path) -> None:
    ini = config.NEWS_RAW_DIR / "mt5_startup.ini"
    # A chart must be open for a script; EURUSD, not gold: no gold price is shown or read (hard rule 1).
    ini.write_text("[StartUp]\n"
                   f"Script={FOLDER}\\{SCRIPT_NAME}\n"
                   "Symbol=EURUSD\n"
                   "Period=H1\n"
                   "ShutdownTerminal=0\n", encoding="utf-16")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(f'"{terminal}" /config:"{ini}"', creationflags=flags, close_fds=True)
    print("2. MT5 started with the script (it stays open afterwards)")


def wait_for(info: Path) -> None:
    for _ in range(WAIT_SECONDS):
        if info.is_file() and "done=1" in info.read_text(encoding="utf-8", errors="ignore"):
            print("3. the script finished")
            return
        time.sleep(1)
    raise TimeoutError(f"no {info.name} after {WAIT_SECONDS} s - see the terminal's Experts tab / MQL5/Logs")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect(files: Path) -> None:
    config.NEWS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.NEWS_SEALED_DIR.mkdir(parents=True, exist_ok=True)
    moves = ((files / "calendar_US_open.tsv", config.NEWS_RAW_DIR / OPEN_FILE),
             (files / "calendar_US_info.txt", config.NEWS_RAW_DIR / INFO_FILE),
             (files / "calendar_US_sealed.tsv", config.NEWS_SEALED_DIR / SEALED_FILE))
    for src, dst in moves:
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(src, tmp)           # bytes only: nothing is parsed
        tmp.replace(dst)
        src.unlink()
    sealed = config.NEWS_SEALED_DIR / SEALED_FILE
    with sealed.open("rb") as f:
        lines = sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))
    print(f"4. {config.NEWS_RAW_DIR / OPEN_FILE}")
    print(f"   {config.NEWS_RAW_DIR / INFO_FILE}:")
    for line in (config.NEWS_RAW_DIR / INFO_FILE).read_text(encoding="utf-8").splitlines():
        print(f"     {line}")
    print(f"   sealed, NOT read: {sealed} - {lines - 1} rows, sha256 {sha256(sealed)}")
    print(f"5. the terminal's copies deleted from {files}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--terminal", type=Path, default=config.MT5_TERMINAL, help="terminal64.exe to use")
    a = ap.parse_args()
    config.NEWS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    mql5 = data_folder(a.terminal) / "MQL5"
    compile_script(a.terminal, mql5)
    files = mql5 / "Files" / FOLDER
    (files / "calendar_US_info.txt").unlink(missing_ok=True)
    close_politely(a.terminal)
    start_with_script(a.terminal)
    wait_for(files / "calendar_US_info.txt")
    collect(files)


if __name__ == "__main__":
    main()
