"""Turn the 3 raw price CSVs into clean files in data/ (L1).  Run:  python -m machine.data.convert

For each raw file, every line is sorted by the date at its start, on that file's own clock:
  before config.SEALED_FROM (2024-01-01) -> parsed, checked, written to data/<name>.parquet
  on or after it                          -> copied as raw text to data/sealed/<name>_2024plus.csv. NEVER parsed:
                                             no price is read, no statistic is made, only the row count is printed.
                                             Tejas, LEARNED #96: "No, don't touch those years" (also #104).
Parquet column statistics are switched off for the sealed years' sake too (they are not written for any output).
data/DATA_HASHES.txt: SHA-256 of each raw CSV, a CONTENT hash of each Parquet output (machine/data/hashes.py), and
SHA-256 of each sealed file. Then the server clock is measured (machine/data/measure_clock.py).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from machine import config
from machine.data import hashes

DATE_AT_START = re.compile(rb"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Source:
    name: str                    # output name: data/<name>.parquet
    raw: Path
    clock: str                   # "server" (MT5) or "utc" (Dukascopy)
    time_col: str
    columns: dict[str, str]      # raw column -> dtype
    decimals: dict[str, int]     # price columns -> decimals on the file's grid (for the content hash)
    note: str

    @property
    def out(self) -> Path:
        return config.DATA_DIR / f"{self.name}.parquet"

    @property
    def sealed(self) -> Path:
        return config.SEALED_DIR / f"{self.name}_2024plus.csv"


MT5_COLUMNS = {"open": "float64", "high": "float64", "low": "float64", "close": "float64",
               "vol": "int64", "spread": "int64"}
MT5_DECIMALS = {"open": 2, "high": 2, "low": 2, "close": 2}

SOURCES = (
    Source("gold_m1", config.RAW_M1, "server", "time", MT5_COLUMNS, MT5_DECIMALS,
           "MT5 XAUUSD 1-min bars (MetaQuotes-Demo), SERVER time, BID prices, spread in points, vol = ticks"),
    Source("gold_m5_mt5", config.RAW_M5, "server", "time", MT5_COLUMNS, MT5_DECIMALS,
           "MT5 XAUUSD 5-min bars, same clock as gold_m1; only used to check the 5-min bars made from gold_m1"),
    Source("gold_m5_dukascopy", config.RAW_DUKA, "utc", "timestamp",
           {"open": "float64", "high": "float64", "low": "float64", "close": "float64"},
           {"open": 3, "high": 3, "low": 3, "close": 3},
           "Dukascopy XAUUSD 5-min bars, UTC; only used to measure the MT5 server clock (volume not kept)"),
)


def split_lines(raw: Path) -> tuple[bytes, list[bytes], list[bytes]]:
    """(header, lines before SEALED_FROM, lines from SEALED_FROM on). Compares the date text only."""
    sealed_from = config.SEALED_FROM.encode()
    header, *lines = raw.read_bytes().splitlines()
    before, sealed = [], []
    for n, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        if not DATE_AT_START.match(line):
            raise ValueError(f"{raw.name} line {n} does not start with a date (YYYY-MM-DD); stopping.")
        (sealed if line[:10] >= sealed_from else before).append(line)
    return header, before, sealed


def parse(src: Source, header: bytes, lines: list[bytes]) -> pd.DataFrame:
    """The rows before 2024 as a frame indexed by time (naive server time, or UTC for Dukascopy)."""
    text = io.BytesIO(header + b"\n" + b"\n".join(lines))
    df = pd.read_csv(text, dtype={c: t for c, t in src.columns.items()}, usecols=[src.time_col, *src.columns])
    if src.clock == "utc":
        t = pd.to_datetime(df[src.time_col], format="%Y-%m-%d %H:%M:%S%z", utc=True)
    else:
        t = pd.to_datetime(df[src.time_col], format="%Y-%m-%d %H:%M:%S")
    df.index = pd.DatetimeIndex(t).as_unit("ns").rename("time")
    df = df[list(src.columns)]
    if df.isna().any().any():
        raise ValueError(f"{src.raw.name}: blank values before {config.SEALED_FROM}; stopping (nothing is filled in).")
    return df


def write_parquet(df: pd.DataFrame, path: Path, meta: dict) -> None:
    """Write through a temp file, then move it into place (never a half-written file)."""
    table = pa.Table.from_pandas(df.reset_index(), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), hashes.META_KEY: json.dumps(meta)})
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, write_statistics=False)
    os.replace(tmp, path)


def write_sealed(header: bytes, lines: list[bytes], path: Path) -> str:
    """Copy the sealed lines as they are; return the file's SHA-256. Nothing in them is parsed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("wb") as f:
        f.write(header + b"\n")
        for line in lines:
            f.write(line + b"\n")
    os.replace(tmp, path)
    return hashes.file_sha256(path)


def convert_one(src: Source) -> list[str]:
    """Convert one raw file; return its lines for DATA_HASHES.txt."""
    if not src.raw.is_file():
        raise FileNotFoundError(f"{src.raw} is missing — copy the 3 CSVs first (machine/data/README.md).")
    raw_sha = hashes.file_sha256(src.raw)
    header, before, sealed = split_lines(src.raw)
    df = parse(src, header, before)
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{src.raw.name}: rows are not in time order before {config.SEALED_FROM}; stopping.")
    content = hashes.content_hash(df, src.decimals)
    meta = {"name": src.name, "clock": src.clock, "source": src.raw.relative_to(config.ROOT).as_posix(),
            "source_sha256": raw_sha, "content_sha256": content, "rows": len(df),
            "before": config.SEALED_FROM, "decimals": src.decimals, "note": src.note}
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(df, src.out, meta)
    sealed_sha = write_sealed(header, sealed, src.sealed)

    per_year = df.index.year.value_counts().sort_index()
    print(f"\n{src.name}  ({src.note})")
    print(f"  raw      {src.raw.relative_to(config.ROOT).as_posix()}  sha256 {raw_sha[:16]}")
    print(f"  clean    {src.out.relative_to(config.ROOT).as_posix()}  {len(df):,} rows "
          f"({df.index[0]} to {df.index[-1]}, {src.clock} clock)  content {content[:16]}")
    print("  rows per year: " + "  ".join(f"{y} {n:,}" for y, n in per_year.items()))
    print(f"  sealed   {src.sealed.relative_to(config.ROOT).as_posix()}  {len(sealed):,} rows (row count only)")

    rel = lambda p: p.relative_to(config.ROOT).as_posix()  # noqa: E731
    return [
        f"raw     {rel(src.raw)}  sha256={raw_sha}  bytes={src.raw.stat().st_size}",
        f"clean   {rel(src.out)}  content={content}  rows={len(df)}  clock={src.clock}  from={rel(src.raw)}",
        f"sealed  {rel(src.sealed)}  sha256={sealed_sha}  rows={len(sealed)}  from={rel(src.raw)}",
    ]


def main() -> None:
    lines = [
        f"# L1 price data hashes — machine/data/convert.py, {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "# raw: SHA-256 of the file bytes.  clean: CONTENT hash of the rows' values (machine/data/hashes.py) — the",
        "# number trial files record.  sealed: SHA-256 of the 2024+ raw lines, copied unread.",
    ]
    for src in SOURCES:
        lines += convert_one(src)
    tmp = config.DATA_HASHES.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, config.DATA_HASHES)
    print(f"\nwrote {config.DATA_HASHES.relative_to(config.ROOT).as_posix()}")
    expected = config.EXPECTED_HASHES.relative_to(config.ROOT).as_posix()
    if not config.EXPECTED_HASHES.is_file():
        print(f"no {expected} yet: copy the raw/clean/sealed lines of DATA_HASHES.txt into it and commit it")
    elif diff := hashes.differences_from_expected():
        print(f"DIFFERENT from {expected} (the data in git's record) - a bad copy, or new data on purpose "
              "(then copy DATA_HASHES.txt's lines into it and commit):\n  " + "\n  ".join(diff))
    else:
        print(f"same data as {expected}: this PC has exactly the prices the project was built on")

    from machine.data import measure_clock
    measure_clock.main()


if __name__ == "__main__":
    main()
