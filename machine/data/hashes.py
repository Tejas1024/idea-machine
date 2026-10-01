"""Fingerprints of the price data (L1).

- A raw CSV: SHA-256 of its bytes.
- A clean output (Parquet): a CONTENT hash — SHA-256 of the rows' values, sorted by time. Parquet bytes change
  between library versions; the values do not, so the content hash is the one trial files record (likhith.md R14).
- A sealed file (2024+ raw text lines, never parsed): SHA-256 of its bytes.

The content hash, exactly (so another PC or another language can rebuild it):
  rows sorted by time; then for the time and for each column, in the frame's column order:
    a header line  b"<name>:<kind>\\n"
    the values as little-endian int64 bytes
  time              -> nanoseconds since 1970-01-01 on the file's own clock      kind "time-ns"
  a price column    -> round(value * 10**d), exact on that grid or refused         kind "int-e<d>"
  an integer column -> the integer                                                 kind "int"
data/DATA_HASHES.txt holds every hash; each Parquet file also carries its own in its metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from machine import config

META_KEY = b"idea_machine"          # Parquet metadata key: JSON with content_sha256, clock, source, ...
CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def content_hash(df: pd.DataFrame, decimals: dict[str, int]) -> str:
    """The content hash of a frame indexed by time (see the module text). decimals: price column -> decimals."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("content_hash needs a frame indexed by time.")
    df = df.sort_index(kind="stable")
    h = hashlib.sha256()
    idx = df.index.tz_convert("UTC").tz_localize(None) if df.index.tz is not None else df.index
    h.update(b"time:time-ns\n")
    h.update(idx.as_unit("ns").asi8.astype("<i8").tobytes())
    for name in df.columns:
        col = df[name]
        if name in decimals:
            d = decimals[name]
            scaled = col.to_numpy(dtype="float64") * 10.0 ** d
            ints = np.rint(scaled)
            off = np.abs(scaled - ints)
            if np.isnan(scaled).any() or (off > 1e-6).any():
                raise ValueError(f"column {name!r} is not on a 10^-{d} grid (or has blanks); refusing to hash it.")
            h.update(f"{name}:int-e{d}\n".encode())
            h.update(ints.astype("<i8").tobytes())
        elif pd.api.types.is_integer_dtype(col):
            h.update(f"{name}:int\n".encode())
            h.update(col.to_numpy(dtype="int64").astype("<i8").tobytes())
        else:
            raise TypeError(f"content_hash: column {name!r} is {col.dtype}; give its decimals or make it integer.")
    return h.hexdigest()


def parquet_meta(path: Path) -> dict:
    """The idea_machine metadata a converted Parquet file carries (content_sha256, clock, source, ...)."""
    meta = pq.read_schema(path).metadata or {}
    if META_KEY not in meta:
        raise ValueError(f"{path} has no {META_KEY.decode()} metadata — rebuild it: python -m machine.data.convert")
    return json.loads(meta[META_KEY])


def read_data_hashes(path: Path | None = None) -> dict[str, dict[str, str]]:
    """data/DATA_HASHES.txt as {name: {field: value}} (one line per file; '#' lines are notes)."""
    path = Path(path or config.DATA_HASHES)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing — run: python -m machine.data.convert")
    out: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind, name, *fields = line.split()
        out[name] = {"kind": kind, **dict(f.split("=", 1) for f in fields)}
    return out


COMPARED = ("sha256", "content", "rows")


def differences_from_expected() -> list[str]:
    """What differs between data/DATA_HASHES.txt (this PC) and machine/data/EXPECTED_HASHES.txt (in git).

    Empty = this PC has exactly the prices the project was built on. Not empty = different raw files: either a bad
    copy, or new data on purpose — then copy the DATA_HASHES.txt lines into EXPECTED_HASHES.txt and commit it, so
    both PCs see the change.
    """
    here, expected = read_data_hashes(), read_data_hashes(config.EXPECTED_HASHES)
    out = [f"{name}: missing on this PC" for name in expected if name not in here]
    out += [f"{name}: not in {config.EXPECTED_HASHES.name}" for name in here if name not in expected]
    for name in expected.keys() & here.keys():
        for field in COMPARED:
            if expected[name].get(field) != here[name].get(field):
                out.append(f"{name}: {field} {here[name].get(field)} here, {expected[name].get(field)} expected")
    return sorted(out)


def data_fingerprint() -> dict[str, str]:
    """What a trial file records about the data it ran on: {output name: content hash}, plus the raw SHA-256s.

    Sealed files are left out on purpose: a trial never reads them.
    """
    found = read_data_hashes()
    return {name: row.get("content") or row["sha256"] for name, row in found.items() if row["kind"] != "sealed"}
