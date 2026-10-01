"""Build data/usd_inr.csv — the rupees per US dollar for every month 2016–2023 (L2).  Run:  python -m machine.data.usd_inr_fetch

Source: the RBI's "Handbook of Statistics on the Indian Economy", the table "Exchange Rate of the Indian Rupee
vis-à-vis the SDR, US Dollar, Pound Sterling, Euro and Japanese Yen (Monthly Average and end-Month Rates)" — the US
Dollar MONTHLY AVERAGE column. The table's own note: RBI reference rates from May 2012; FBIL (Financial Benchmarks
India) reference rates from 10 July 2018. So 2016 → June 2018 is the RBI reference rate, July 2018 is both (RBI
1–9 July, FBIL 10–31 July), and August 2018 on is the FBIL reference rate.

Each edition shows about 4½ years, so 6 editions are read. Where editions overlap, the NEWEST edition wins (RBI
corrects old months in later editions). Two checks before anything is written:
  1. every overlapping month is compared; disagreements are printed. Run of 28 Sep 2026: 84 months overlap, 5 differ
     — four by ₹0.0001 (rounding), and Sep 2020 by ₹0.0104 in the 2022 edition only (the 2021, 2023 and 2024
     editions all say 73.4817, and the newest edition wins, so 73.4817 is used);
  2. for each year, the mean of the 12 monthly averages must equal the RBI's separate calendar-year average table
     (Table 132) within ₹0.0001 — a wrong column or a shifted month cannot pass this. All 8 years pass.
Only months before config.SEALED_FROM are kept, and nothing later is printed.
The file is small and in git (data/usd_inr.csv), so other PCs and CI do not need to run this.
"""
from __future__ import annotations

import html
import re
import urllib.request
from dataclasses import dataclass

import pandas as pd

from machine import config

PAGE = "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id={}"

# (edition, page id) — the monthly table in each Handbook edition. Oldest first; the newest edition wins a month.
EDITIONS = (
    ("2019", 19191),       # published 15 Sep 2019: Jan 2015 – Jul 2019
    ("2021", 20602),       # 15 Sep 2021: Jan 2017 – Jul 2021
    ("2022", 21294),       # 15 Sep 2022: Jan 2018 – Jul 2022 (gives 2018; its Sep 2020 is the one typo, not used)
    ("2023", 22006),       # 15 Sep 2023: Jan 2019 – Jul 2023
    ("2024", 22674),       # 13 Sep 2024: Jan 2020 – ...
    ("2025", 23378),       # 29 Aug 2025: Jan 2021 – ...
    ("2025-26", 24030),    # 31 Jul 2026: Jan 2022 – ...
)
ANNUAL_TABLE = 23957       # Table 132, calendar-year annual average (2025-26 edition): the second check
FIRST_MONTH, LAST_MONTH = "2016-01", "2023-12"
MONTHS = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
FBIL_FROM = "2018-07"      # FBIL took over on 10 July 2018


@dataclass(frozen=True)
class Table:
    edition: str
    url: str
    rows: dict[str, float]  # "YYYY-MM" -> US Dollar monthly average, ₹ per $


def fetch(page_id: int) -> str:
    req = urllib.request.Request(PAGE.format(page_id), headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def cells_by_row(page: str) -> list[list[str]]:
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)]
        if any(cells):
            out.append(cells)
    return out


def monthly_table(edition: str, page_id: int) -> Table:
    page = fetch(page_id)
    if "Exchange Rate of the Indian Rupee" not in page or "Monthly" not in page:
        raise ValueError(f"page {page_id} ({edition}) is not the monthly exchange-rate table")
    rows = cells_by_row(page)
    head = next(r for r in rows if "US Dollar" in r)
    # top header: Year/Month, SDR, US Dollar, ... ; each currency has 2 cells below it: Average, End-month
    col = 1 + 2 * (head.index("US Dollar") - 1)
    out, year = {}, None
    for r in rows:
        m = re.fullmatch(r"(\d{4})(?:-\d{2})?", r[0])
        if m and all(c == "" for c in r[1:]):
            year = int(m.group(1))
            continue
        if year and r[0][:3] in MONTHS and len(r) > col:
            key = f"{year}-{MONTHS[r[0][:3]]:02d}"
            if key < config.SEALED_FROM[:7]:
                out[key] = float(r[col])
    return Table(edition, PAGE.format(page_id), out)


def annual_averages() -> dict[int, float]:
    rows = cells_by_row(fetch(ANNUAL_TABLE))
    head = next(r for r in rows if "US Dollar" in r)
    col = head.index("US Dollar")       # one cell per currency in this table
    return {int(r[0]): float(r[col]) for r in rows if re.fullmatch(r"\d{4}", r[0]) and len(r) > col
            and int(r[0]) < int(config.SEALED_FROM[:4])}


def source_of(month: str) -> str:
    if month < FBIL_FROM:
        return "RBI reference rate"
    if month == FBIL_FROM:
        return "RBI reference rate 1-9 Jul + FBIL reference rate 10-31 Jul"
    return "FBIL reference rate"


def build() -> pd.DataFrame:
    tables = [monthly_table(ed, i) for ed, i in EDITIONS]
    for t in tables:
        print(f"  {t.edition:8s} {t.url}  months {min(t.rows)} -> {max(t.rows)}")

    # check 1: editions that overlap
    seen: dict[str, dict[str, float]] = {}
    for t in tables:
        for k, v in t.rows.items():
            seen.setdefault(k, {})[t.edition] = v
    overlap = {k: v for k, v in seen.items() if len(v) > 1}
    differ = {k: v for k, v in overlap.items() if max(v.values()) - min(v.values()) > 1e-9}
    print(f"  months in 2+ editions: {len(overlap)}; they differ in {len(differ)}:")
    for k, v in sorted(differ.items()):
        print(f"    {k}: " + ", ".join(f"{ed} edition {x:.4f}" for ed, x in v.items()))

    months = [str(p) for p in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")]
    rows = []
    for m in months:
        newest = next((t for t in reversed(tables) if m in t.rows), None)
        if newest is None:
            raise ValueError(f"no edition has {m}")
        rows.append({"month": m, "usd_inr": newest.rows[m], "source": source_of(m), "edition": newest.edition})
    df = pd.DataFrame(rows)

    # check 2: the mean of the 12 months = the RBI's own calendar-year average
    ann = annual_averages()
    df["year"] = df.month.str[:4].astype(int)
    for y, g in df.groupby("year"):
        mean = g.usd_inr.mean()
        ok = abs(mean - ann[y]) < 0.00006
        print(f"  {y}: mean of 12 months {mean:.4f} | RBI calendar-year average {ann[y]:.4f} | {'same' if ok else 'DIFFERENT'}")
        if not ok:
            raise ValueError(f"{y}: the monthly rows do not give the RBI's annual average — wrong column or month?")
    return df.drop(columns="year")


def write(df: pd.DataFrame) -> None:
    urls = " | ".join(f"{ed} edition {PAGE.format(i)}" for ed, i in EDITIONS if ed in set(df.edition))
    first = ("# sources: RBI Handbook of Statistics on the Indian Economy, table 'Exchange Rate of the Indian Rupee "
             "... (Monthly Average and end-Month Rates)', US Dollar monthly average, Rs per $ | " + urls +
             f" | checked against the calendar-year average table {PAGE.format(ANNUAL_TABLE)}"
             " | made by python -m machine.data.usd_inr_fetch\n")
    path = config.USD_INR_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(first)
        df.to_csv(f, index=False, float_format="%.4f", lineterminator="\n")
    tmp.replace(path)
    print(f"  wrote {path.relative_to(config.ROOT)}: {len(df)} months, {df.month.iloc[0]} -> {df.month.iloc[-1]}, "
          f"Rs {df.usd_inr.min():.2f} - {df.usd_inr.max():.2f} per $")


if __name__ == "__main__":
    write(build())
