"""prior-tries.csv (read by the referee, L5) loads, every row has a source, and it agrees with prior-tries.md."""
import csv
import re

from machine import config

COLUMNS = ["idea_key", "blocks", "timeframe", "entry", "tries", "years", "source"]
TIMEFRAMES = {"5m", "15m", "1h", "4h", "1d", "1w", "none"}
ENTRY_KINDS = {"gap", "line", "zone", "fib", "pattern", "stop", "market", "random", "any", "none"}


def _csv_rows():
    with open(config.PRIOR_TRIES_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def _md_rows():
    """The data rows of prior-tries.md: 7 cells, the first a `key`, the fourth a whole number (tries)."""
    rows = {}
    for line in config.PRIOR_TRIES_MD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) != 7 or not cells[3].isdigit():
            continue
        key = cells[0].strip("`")
        assert key not in rows, f"{key} is twice in prior-tries.md"
        rows[key] = {"tries": cells[3], "years": cells[4], "source": cells[6]}
    return rows


def test_csv_loads_with_its_columns_and_every_row_has_a_source():
    rows = _csv_rows()
    assert len(rows) >= 37  # the 27 Sep draft had 17 rows; L5-prior expected about 20 more
    keys = [r["idea_key"] for r in rows]
    assert len(keys) == len(set(keys)), "an idea_key is used twice"
    for r in rows:
        assert re.fullmatch(r"[a-z0-9_]+", r["idea_key"]), r["idea_key"]
        assert r["source"].strip(), f"{r['idea_key']} has no source"
        assert r["blocks"].strip() and " " not in r["blocks"], r["idea_key"]
        assert set(r["timeframe"].split("|")) <= TIMEFRAMES, r["idea_key"]
        assert r["entry"].split(":")[0] in ENTRY_KINDS, r["idea_key"]
        assert int(r["tries"]) >= 1, r["idea_key"]
        first, last = (int(y) for y in re.fullmatch(r"(\d{4})-(\d{4})", r["years"]).groups())
        assert 2015 <= first <= last <= 2026, r["idea_key"]


def test_csv_and_md_have_the_same_rows_tries_and_years():
    csv_rows = {r["idea_key"]: r for r in _csv_rows()}
    md_rows = _md_rows()
    assert set(csv_rows) == set(md_rows), (
        f"only in the CSV: {sorted(set(csv_rows) - set(md_rows))}; only in the md: {sorted(set(md_rows) - set(csv_rows))}")
    for key, md in md_rows.items():
        assert md["source"], f"{key} has no source in prior-tries.md"
        assert csv_rows[key]["tries"] == md["tries"], key
        assert csv_rows[key]["years"] == md["years"], key


def test_rows_that_saw_the_sealed_years_are_marked():
    """Before LEARNED #96 (15 Sep 2026) the old project used 2024-2026 as test years; those rows must say so."""
    rows = {r["idea_key"]: r for r in _csv_rows()}
    for key in ("early_sweep_126_combos", "type1_race_causal", "donchian_trend", "ml_direction_six"):
        assert int(rows[key]["years"][-4:]) >= 2024, key
    assert rows["fib_0618"]["years"] == "2016-2023"  # §66bz: 2024-26 sealed, never loaded
