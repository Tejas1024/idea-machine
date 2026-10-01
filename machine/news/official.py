"""The official release calendars (L12 step 2): BLS, the Federal Reserve, BEA.

They CHECK the MT5 calendar - its dates, its New York times, and for the jobs report and CPI its first-released
number. MT5 stays the source of every row (it has every actual number 2016-2023, so nothing needs filling).

  BLS   Employment Situation (NFP + unemployment rate) and CPI: every archived news release 2016-2023 - its
        "embargoed until 8:30 a.m. (EST) Friday, January 8, 2016" line and its first sentence.
  Fed   FOMC statements 2016-2023 from the Fed's meeting pages ("(unscheduled)" meetings marked there), each
        statement's "For release at 2:00 p.m. EST" line and title.
  BEA   GDP release dates 2016-2023 from BEA's news archive (BEA's pages give no time).

Only 2016-2023 is ever requested: a release dated 2024 or later is skipped by the date in its link, never fetched
(LEARNED #96). Pages are kept in data/news/official/ (not in git), so a second run reads no network.
The agencies refuse scripts that look like scripts (BLS answers "Access Denied"), so the requests carry a normal
browser's headers and wait DELAY_SECONDS between pages.
"""
from __future__ import annotations

import datetime as dt
import re
import time
import urllib.request

import pandas as pd

from machine import config

FIRST_YEAR, LAST_YEAR = 2016, 2023
NY = "America/New_York"
DELAY_SECONDS = 3.0
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Dest": "document",
}
BLS_INDEX = "https://www.bls.gov/bls/news-release/{kind}.htm"
BLS_RELEASE = "https://www.bls.gov/news.release/archives/{kind}_{mmddyyyy}.htm"
FED_HISTORICAL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"   # 2016-2020
FED_CALENDARS = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"           # 2021 on
FED_PAGE = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{yyyymmdd}a.htm"
BEA_ARCHIVE = ("https://www.bea.gov/news/archive?field_related_product_target_id=451&created_1={year}"
               "&title=&page={page}")

_last_fetch = [0.0]


def fetch(url: str, name: str) -> str:
    """The page's text, from data/news/official/<name> when it is there, else from the network (then saved)."""
    path = config.NEWS_OFFICIAL_DIR / name
    if path.is_file() and path.stat().st_size >= 2000:
        return path.read_bytes().decode("utf-8", errors="replace")   # BLS pages hold a few Windows-1252 bytes
    wait = DELAY_SECONDS - (time.monotonic() - _last_fetch[0])
    if wait > 0:
        time.sleep(wait)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        raw = r.read()
    _last_fetch[0] = time.monotonic()
    body = raw.decode("utf-8", errors="replace")
    if len(raw) < 2000:
        raise ConnectionError(f"{url} gave {len(raw)} bytes - not saved; run again")
    if "<title>Access Denied</title>" in body:
        raise PermissionError(f"{url} answered 'Access Denied' - wait a few minutes and run again (pages already "
                              f"saved are not fetched again)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)                     # the page exactly as it came
    tmp.replace(path)
    return body


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).replace("&nbsp;", " ").replace("&amp;", "&")


def _in_years(day: dt.date) -> bool:
    return FIRST_YEAR <= day.year <= LAST_YEAR


def _ny(day_text: str, clock: str, ampm: str) -> pd.Timestamp:
    """'January 8, 2016', '8:30', 'a' -> that moment in New York (EST or EDT decided by the date, as the agency's)."""
    t = pd.Timestamp(f"{day_text} {clock} {ampm.upper()}M")
    return t.tz_localize(NY)


# ---- BLS ----------------------------------------------------------------------------------------------------------

def bls_release_days(kind: str) -> list[dt.date]:
    """Release days 2016-2023 of a BLS news release ('empsit' = the jobs report, 'cpi'), from BLS's own archive list."""
    html = fetch(BLS_INDEX.format(kind=kind), f"bls_{kind}_index.html")
    days = {dt.datetime.strptime(m, "%m%d%Y").date() for m in re.findall(rf"archives/{kind}_(\d{{8}})\.htm", html)}
    return sorted(d for d in days if _in_years(d))


_EMBARGO = re.compile(r"embargoed until.{0,40}?(\d{1,2}:\d{2})\s*([ap])\.m\.\s*\((?:E[SD]T|ET)\),?\s*"
                      r"(?:\w+day,\s*)?(\w+ \d{1,2},\s*\d{4})", re.IGNORECASE)
_NFP = re.compile(r"Total nonfarm payroll employment (.{0,90}?)(?:by ([\d,.]+)( million)?|\(([+\-−]?[\d,.]+)\))")
# "the unemployment rate declined by 0.3 percentage point to 4.7 percent": the rate is the number before "percent",
# not the change before "percentage point"
# (a "." inside a number like 0.3 does not end the sentence - else the search skips the headline and finds a later
# rate, e.g. adult men's 4.3 percent)
_UNEMPLOYMENT = re.compile(r"unemployment rate(?:[^.]|\.(?=\d)){0,90}?(\d{1,2}\.\d) percent(?!age)")
_CPI = re.compile(r"Consumer Price Index for All Urban Consumers \(CPI-U\) (.{0,40}?)(?:(\d+\.\d) percent| in )")
_DOWN = re.compile(r"declin|fell|decreas|down|lost|drop", re.IGNORECASE)


def bls_release(kind: str, day: dt.date) -> dict:
    """One archived release: its official time (New York) and its first-released headline number(s).
    One page is empty on BLS's own server (CPI, 16 Jun 2016: 200 OK, 20 bytes): its row says so and has no numbers."""
    try:
        html = fetch(BLS_RELEASE.format(kind=kind, mmddyyyy=day.strftime("%m%d%Y")), f"bls_{kind}_{day:%Y%m%d}.html")
    except ConnectionError:
        return {"agency": "BLS", "release": kind, "day": day, "official_ny": pd.NaT,
                "note": "BLS's archived page is empty"}
    text = _text(html)
    m = _EMBARGO.search(text)
    out: dict = {"agency": "BLS", "release": kind, "day": day,
                 "official_ny": _ny(m.group(3), m.group(1), m.group(2)) if m else pd.NaT}
    if kind == "empsit":
        n = _NFP.search(text)
        if n:
            verb, by, million, paren = n.groups()
            if by:
                value = float(by.replace(",", "")) * (1000.0 if million else 0.001)   # -> thousands
                out["nfp_thousands"] = -value if _DOWN.search(verb) else value
            else:
                out["nfp_thousands"] = float(paren.replace(",", "").replace("−", "-")) / 1000.0
            # "rose by 2.5 million" is rounded to 0.1 million: the exact first number (2,509) is within 50 thousand
            out["nfp_thousands_rounding"] = 50.0 if million else 0.5
        u = _UNEMPLOYMENT.search(text)
        out["unemployment_rate"] = float(u.group(1)) if u else float("nan")
    elif kind == "cpi":
        c = _CPI.search(text)
        if c:
            verb, pct = c.groups()
            if pct is None:
                out["cpi_mm"] = 0.0 if "unchanged" in verb else float("nan")
            else:
                out["cpi_mm"] = -float(pct) if _DOWN.search(verb) else float(pct)
    return out


# ---- the Federal Reserve ------------------------------------------------------------------------------------------

_STATEMENT = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm")
_FED_TIME = re.compile(r"For release at (\d{1,2}:\d{2})\s*([ap])\.m\.\s*(?:E[SD]T|ET)", re.IGNORECASE)
_FED_DAY = re.compile(r"(?:Release Date|For release on)[: ]*\s*(\w+ \d{1,2},\s*\d{4})", re.IGNORECASE)


def _fed_year_pages() -> list[tuple[int, str]]:
    pages = [(y, fetch(FED_HISTORICAL.format(year=y), f"fed_fomchistorical{y}.html")) for y in range(FIRST_YEAR, 2021)]
    cal = fetch(FED_CALENDARS, "fed_fomccalendars.html")
    starts = [(int(m.group(1)), m.start()) for m in re.finditer(r"(\d{4}) FOMC Meetings", cal)]
    for i, (year, pos) in enumerate(starts):
        if 2021 <= year <= LAST_YEAR:      # later sections are skipped, not read
            end = starts[i + 1][1] if i + 1 < len(starts) else len(cal)
            pages.append((year, cal[pos:end]))
    return pages


_FED_HEADING = re.compile(r"\w+ \d{1,2}(?:-\d{1,2})?\s*(?:\((unscheduled|notation vote|cancelled)\))?\s*"
                         r"(?:Meeting\s*)?-\s*\d{4}", re.IGNORECASE)


def _not_known_in_advance(window: str) -> bool:
    """The Fed's own label on the meeting a statement belongs to: '(unscheduled)' or '(notation vote)' = nobody knew
    it was coming. The label sits in the heading right before the statement's link (2016-2020 pages); the 2021-2023
    page has no such headings - there, the words just before the link."""
    headings = list(_FED_HEADING.finditer(window))
    if headings:
        return (headings[-1].group(1) or "").lower() in ("unscheduled", "notation vote")
    tail = window[-400:].lower()
    return "unscheduled" in tail or "notation vote" in tail


def fomc_statements() -> pd.DataFrame:
    """Every FOMC statement 2016-2023 linked from the Fed's meeting pages: day, unscheduled (the Fed's own label:
    "(unscheduled)" or "(notation vote)"), official New York time and title (from the statement's page)."""
    rows = []
    for year, html in _fed_year_pages():
        prev = 0
        for m in _STATEMENT.finditer(html):
            day = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            window = _text(html[prev:m.start()])
            prev = m.end()
            if not _in_years(day) or day.year != year or any(r["day"] == day for r in rows):
                continue
            raw = fetch(FED_PAGE.format(yyyymmdd=m.group(1)), f"fed_monetary{m.group(1)}a.html")
            page = _text(raw)
            t, d = _FED_TIME.search(page), _FED_DAY.search(page)
            title = re.search(r"<title>\s*Federal Reserve Board - ([^<]+?)\s*</title>", raw)
            rows.append({"agency": "Fed", "release": "fomc", "day": day,
                         "unscheduled": _not_known_in_advance(window),
                         "official_ny": _ny(d.group(1) if d else day.strftime("%B %d, %Y"), t.group(1), t.group(2))
                         if t else pd.NaT,
                         "title": title.group(1).strip() if title else ""})
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


# ---- BEA ----------------------------------------------------------------------------------------------------------

_BEA_ROW = re.compile(r'<a href="(/news/\d{4}/[^"]+)"[^>]*>([^<]*)</a>\s*</td>\s*<td[^>]*views-field-created[^>]*>'
                      r'\s*<time datetime="([^"]+)"')


def _bea_year_codes() -> dict[int, str]:
    """BEA's archive filters years by a code (1 = 2024, 2 = 2023 ...), read from its form. The form is fetched with
    an invalid year, so BEA shows the form and no release at all."""
    html = fetch(BEA_ARCHIVE.format(year="0", page=0), "bea_gdp_form.html")
    select = html[html.find('name="created_1"'):]
    select = select[:select.find("</select>")]
    return {int(year): code for code, year in re.findall(r'<option value="(\d+)">\s*(\d{4})\s*</option>', select)}


def bea_gdp_days() -> pd.DataFrame:
    """GDP releases 2016-2023 (advance / second / third estimate) from BEA's news archive, year by year: the day
    and the time BEA lists (its <time datetime=...> tag), the title and the release's link."""
    codes = _bea_year_codes()
    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        for page in range(4):
            html = fetch(BEA_ARCHIVE.format(year=codes[year], page=page), f"bea_gdp_{year}_p{page}.html")
            found = 0
            for link, title, when in _BEA_ROW.findall(html):
                t = pd.Timestamp(when).tz_convert(NY)
                title = " ".join(title.split())
                if t.year == year and re.search(r"GDP|Gross Domestic Product", title) and "estimate" in title.lower():
                    found += 1
                    rows.append({"agency": "BEA", "release": "gdp", "day": t.date(), "official_ny": t,
                                 "title": title, "link": "https://www.bea.gov" + link})
            if found == 0 or f"page={page + 1}" not in html:
                break
    out = pd.DataFrame(rows).drop_duplicates("day").sort_values("day").reset_index(drop=True)
    return out


_BEA_EMBARGO = re.compile(r"EMBARGOED UNTIL RELEASE AT (\d{1,2}:\d{2})\s*([AP])\.M\.\s*E[SD]T,\s*(?:\w+day,\s*)?"
                          r"(\w+ \d{1,2},\s*\d{4})", re.IGNORECASE)


def bea_release_time(link: str) -> pd.Timestamp:
    """The time a BEA release gives ITSELF ("EMBARGOED UNTIL RELEASE AT 8:30 A.M. EDT, Friday, October 26, 2018").
    Used when BEA's archive list disagrees with MT5: the list was a day early once (25 Oct 2018 for the 26th)."""
    name = "bea_release_" + re.sub(r"[^A-Za-z0-9]+", "_", link.rsplit("/", 2)[-2] + "_" + link.rsplit("/", 1)[-1])
    m = _BEA_EMBARGO.search(_text(fetch(link, name[:150] + ".html")))
    return _ny(m.group(3), m.group(1), m.group(2)) if m else pd.NaT


def all_releases() -> dict[str, pd.DataFrame]:
    """Every official list, fetched (or read from data/news/official/)."""
    config.NEWS_OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "empsit": pd.DataFrame([bls_release("empsit", d) for d in bls_release_days("empsit")]),
        "cpi": pd.DataFrame([bls_release("cpi", d) for d in bls_release_days("cpi")]),
        "fomc": fomc_statements(),
        "gdp": bea_gdp_days(),
    }
