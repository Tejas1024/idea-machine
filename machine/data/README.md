# machine/data — gold prices 2016–2023 on the right clock (L1)

```python
from machine.data.bars import load_bars
from machine.data.clock import server_to_utc, to_ist

d = load_bars("1h", "2016", "2020")   # 1m 5m 15m 1h 4h 1d 1w; both ends included, like df.loc
```

Every frame: index `server_time` = the bar's **open** time on the broker's clock (MT5 server time — what an MT5 chart
shows); columns `open high low close` (BID prices, $), `vol` (ticks), `spread` (points; 1 point = $0.01), `n_m1`
(1-min bars inside), `utc` (open time in UTC). India time: `to_ist(d.index)`.

## Sources

| Raw file (in `old/build/data/`, not in git) | What | Clean file (in `data/`, not in git) |
|---|---|---|
| `mt5_xauusd_m1.csv` | MT5 XAUUSD 1-min bars, 2016-01-04 → 2026-08-14. Server time, BID prices, spread in points, vol = ticks. The old notes say the MetaQuotes-Demo server. | `data/gold_m1.parquet` — **the** price data |
| `mt5_xauusd_m5.csv` | MT5 5-min bars, same clock | `data/gold_m5_mt5.parquet` — only to check the 5-min bars built from M1 |
| `gold_5m_dukascopy.csv` | Dukascopy 5-min bars, UTC, 2018-12-31 → | `data/gold_m5_dukascopy.parquet` — only to measure the clock (volume dropped) |

Make the clean files (about 20 seconds): `python -m machine.data.convert`. It also measures the clock and compares
the result with `EXPECTED_HASHES.txt`.

**Point size:** `config.POINT = 0.01`, **read from MT5**: `symbol_info("XAUUSD")` on MetaQuotes-Demo (the server this
data came from) gave point 0.01, digits 2, 100 oz per lot, lot min and step 0.01 (28 Sep 2026). The old code
(`old/ml/agent/skill.py`) has the same value. It is also **checked against the data**: every 2016–2023 M1 price is a
whole number of cents. Only these settings were read, not the live price (live prices wait for T14).

## The clock — measured

**Server time = UTC+2 in winter, UTC+3 in summer, on the EUROPEAN summer-time dates** (last Sunday of March → last
Sunday of October). It is *not* the US dates, which is what the L1 prompt expected. The old engine's
`Europe/Bucharest` was right. In code, `clock.SERVER_ZONE = "Europe/Athens"` (same rule).

| Check | Years | Result |
|---|---|---|
| MT5 5-min closes vs Dukascopy (UTC); every week, shifts −1…+5 h; best = smallest typical gap between the two feeds | 2019–2023 | 261 of 261 weeks clear; every one matches the EU rule. Right hour: 1–22 cents. Next-best hour: ≥ $0.55 (≥ 10×). |
| The weeks when the US and Europe differ (2–3 in March, 1 around the end of Oct / start of Nov) | 2019–2023 | 17 of 17 measured +2. The US rule says +3 in all 17 → wrong every time. |
| Gold's daily stop (17:00 New York) leaves server hour 23 empty — only in those differ weeks under the EU rule | 2016–2023 | Empty in 26 of 26 differ weeks, 0 of 391 other weeks. 2016–2018: 9 of 9, 0 of 148. |

**2016–2018** has no second feed, so the rule 2019–2023 shows is **applied** there. The daily-stop check is the only
evidence for those years, and it agrees.

Per-week table: `data/server_offset.csv` (`python -m machine.data.measure_clock` prints the summary).
**This is the clock of this data file only.** Another broker's server (XM, for the live trial) must be measured again.
The old live path was once 3 hours out (`old/CLAUDE-old-phase.md`).
The clock changes at 03:00–04:00 server time on a Sunday, when gold is closed. `server_to_utc` refuses a time inside
that hour instead of guessing.

## How bars are built (the MT5 way)

open = first, high = max, low = min, close = last, vol = sum, **spread = the smallest 1-min spread in the bar**.
Checked on all 561,728 raw MT5 5-min bars: 100% identical prices, spreads and volumes. That is expected, not "too
good": MT5 builds its own 5-min bars from the same 1-min bars.
Days follow the **broker day**: a `1d` bar is one server calendar day; `4h` bars start at 00/04/08/12/16/20 server
time; a `1w` bar runs Monday → Sunday (labelled Monday; MT5 labels the same week by its Sunday). There are no Saturday
or Sunday bars on the server clock. There are 257–259 daily candles a year.

**Cache:** `data/cache/bars_<tf>_<M1 content hash>_<code hash>.parquet`. New data or changed code (`bars.py`,
`clock.py`) makes a new file. An old cache is never reused (lesson B19). Deleting `data/cache/` is always safe.

## The sealed years (2024, 2025, 2026)

Tejas, LEARNED #96: *"No, don't touch those years"* (also #104).
- `convert.py` sorts each raw line by the date text at its start. Lines from 2024-01-01 on (each file's own clock) are
  **copied as raw text** to `data/sealed/<name>_2024plus.csv`. They are never parsed: no price is read, no statistic
  is made, only the row count is printed.
- `gold_m1.parquet` ends at 2023-12-29 23:56 server time. `load_bars` runs `guards.check_dates` on the request (before
  reading anything) and on the result. Any time ≥ 2024-01-01 raises `SealedYearsError`.
- Opening them needs Tejas's written yes in `plan/work/T14-decisions.md` (prompt T14) and a new loader. Until then, no.

## Hashes — which data a result was made on

`data/DATA_HASHES.txt` (written by `convert.py`) has one line per file:
- **raw**: SHA-256 of the CSV bytes.
- **clean**: a **content hash** — SHA-256 of the rows' values sorted by time, with prices as whole cents (definition
  in `hashes.py`). Parquet bytes change between library versions; the values do not.
- **sealed**: SHA-256 of the copied lines.

Trial files (`plan/work/trials/`, R14) record `hashes.data_fingerprint()` (or `bars.m1_fingerprint()` for the M1
content hash alone).
`EXPECTED_HASHES.txt` (in git) is the record of the data the project was built on. `tests/test_data.py` fails if this
PC's `DATA_HASHES.txt` differs. If you change the data on purpose, copy the new lines into it and commit.

## Copying the CSVs to another PC

1. From Tejas's PC, copy the 3 files by USB or Google Drive (≈ 300 MB; too big for git):
   `old/build/data/mt5_xauusd_m1.csv`, `mt5_xauusd_m5.csv`, `gold_5m_dukascopy.csv`
   → the same folder `old/build/data/` in the other PC's repo.
2. Check the copy (Windows): `certutil -hashfile old\build\data\mt5_xauusd_m1.csv SHA256`. It must equal the `raw`
   line in `EXPECTED_HASHES.txt` (Mac/Linux: `shasum -a 256 <file>`).
3. `python -m machine.data.convert` → it prints `same data as machine/data/EXPECTED_HASHES.txt`.
4. `python -m pytest tests/test_data.py -s` → all pass, with the tables printed.

## What the data looks like (2016–2023, from `pytest tests/test_data.py -s`)

- 1-min rows a year: 349,050–352,900. Daily candles a year: 257–259.
- The 10 biggest gaps inside a week are holidays: Christmas Eve → 26 Dec and New Year's Eve → 2 Jan (about 33
  hours), and the early closes on US holidays (about 5–7 hours).
- The biggest 1-min candle: $37.21 on 2020-03-24 (the COVID crash). The biggest jump from one minute to the next:
  $27.26 on 2021-08-09 01:58 server (the flash crash). Dukascopy's biggest 5-min candle is at 01:55 server (22:55 UTC)
  that same night. The two feeds agree on the clock.
- Spread is 0 on 1.53% of 1-min bars (`config.MIN_SPREAD_POINTS` is the floor for those); the median is 19 points
  ($0.19).

## The two money files (L2) — small, and IN git

| File | Made by | What |
|---|---|---|
| `data/usd_inr.csv` | `python -m machine.data.usd_inr_fetch` (needs the internet) | ₹ per $, monthly average, 2016-01 → 2023-12: RBI reference rate, then FBIL from 10 Jul 2018. Line 1 = the source pages. |
| `data/swap.csv` | `python -m machine.data.swap_read` (🪟 MT5; reads swap settings only, no price) | the broker's swap, one row added per reading, with the date and server read |

`.gitignore` hides everything in `data/` except these two, so every PC and CI has the same values without running
anything. How they are used, and their checks: `machine/costs.md`.
