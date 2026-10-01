"""Read the broker's swap for gold from MT5 and add one row to data/swap.csv (L2).  Run:  python -m machine.data.swap_read

🪟 Windows + MT5 only. The session may open MT5 itself (Tejas, 28 Sep 2026, strategy/learned-inbox/2026-09-28-0119-session.md).
READ ONLY: no order is sent, and no price is read or printed — only the swap settings (hard rules 1 and 10).

The swap is shown ONLY as a labelled extra line. Tejas's account has no swap (LEARNED #54): "There is no swapping
down ... whatever the things that you are telling about swap, it won't apply."
Each reading is ADDED as a new row with the date it was read; machine/costs.py uses the newest row. The server and
company are recorded because the swap belongs to that server — the terminal on Tejas's PC is on MetaQuotes-Demo (the
server the price data came from), not XM.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt

from machine import config

COLUMNS = ("read_on", "server", "company", "account", "symbol", "swap_mode", "swap_mode_name", "swap_long",
           "swap_short", "swap_rollover3days", "point", "contract_size", "currency_profit")
SWAP_MODES = {0: "DISABLED", 1: "POINTS", 2: "CURRENCY_SYMBOL", 3: "CURRENCY_MARGIN", 4: "CURRENCY_DEPOSIT",
              5: "INTEREST_CURRENT", 6: "INTEREST_OPEN", 7: "REOPEN_CURRENT", 8: "REOPEN_BID"}
ACCOUNT_KIND = {0: "demo", 1: "contest", 2: "real"}


def read(symbol: str = "XAUUSD", path: str | None = None) -> dict:
    import MetaTrader5 as mt5

    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 did not start: {mt5.last_error()}")
    try:
        acc = mt5.account_info()
        info = mt5.symbol_info(symbol)
        if acc is None or info is None:
            raise RuntimeError(f"MT5 gave no account or no {symbol}: {mt5.last_error()}")
        return {
            "read_on": dt.date.today().isoformat(),
            "server": acc.server,
            "company": acc.company,
            "account": ACCOUNT_KIND.get(acc.trade_mode, str(acc.trade_mode)),
            "symbol": symbol,
            "swap_mode": int(info.swap_mode),
            "swap_mode_name": SWAP_MODES.get(int(info.swap_mode), "unknown"),
            "swap_long": float(info.swap_long),
            "swap_short": float(info.swap_short),
            "swap_rollover3days": int(info.swap_rollover3days),   # 0 = Sunday ... 3 = Wednesday ... 6 = Saturday
            "point": float(info.point),
            "contract_size": float(info.trade_contract_size),
            "currency_profit": info.currency_profit,
        }
    finally:
        mt5.shutdown()


def append(row: dict) -> None:
    """Add one row; the header is written only when the file is new. Never rewrites old rows."""
    new = not config.SWAP_CSV.is_file()
    config.SWAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SWAP_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", help="terminal64.exe of the MT5 to read (default: the one MT5 finds itself)")
    ap.add_argument("--symbol", default="XAUUSD")
    a = ap.parse_args()
    r = read(a.symbol, a.path)
    append(r)
    print(f"added to {config.SWAP_CSV.relative_to(config.ROOT)}:")
    for k in COLUMNS:
        print(f"  {k:20s} {r[k]}")
