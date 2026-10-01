# machine/news — the news calendar (L12)

```python
from machine.news.calendar import load_events, minutes_to_next, minutes_since_last, next_event, is_news_day

ev = load_events("2019", "2020")              # Tejas's news: red + orange (settings news_events) -> MT5 high + moderate
t = bars.index + pd.Timedelta(minutes=5)      # the moment you DECIDE at: a bar's close, not its open
minutes_to_next(t)                            # scheduled releases only (known in advance)
minutes_since_last(t)                         # every release that has happened
is_news_day(bars.index)                       # the broker's day (clock_name="ist" for India's)
```

Naive times are **server time** (the price data's clock, like every bar index). The table `data/news/events.parquet`
has `utc`, `server` and `ist` for every row.

## Make it (🪟 the PC with MT5 — Tejas's; Likhith's with a free MT5 demo works the same)

```
python -m machine.news.mt5_export     # MT5: compile + run L12_CalendarExport.mq5 (restarts the terminal politely)
python -m machine.news.build          # -> data/news/events.parquet + official_check.csv; prints every check
```

| File (in `data/news/`, not in git) | What |
|---|---|
| `raw/mt5_calendar_US_2015_2023.tsv` | every US calendar row MT5 has, 2015–2023, all importances, as exported |
| `raw/mt5_calendar_US_info.txt` | the server, the terminal build, the export time and **its offset** (needed: see the trap) |
| `sealed/mt5_calendar_US_2024plus.tsv` | 2024 on — copied as bytes, **never parsed** (#96). Only L17a may read it (T14 `news-values`) |
| `official/` | the BLS / Fed / BEA pages, as downloaded (the check) |
| `events.parquet` | **the calendar**: high + moderate, 2015–2023 |
| `official_check.csv` | one row per official release 2016–2023 against MT5's row |

## The clock trap (measured 29 Sep 2026)

MT5's calendar gives every **old** release in server time converted with the offset of **the day it was exported**
(+3 h in EU summer), not the offset on the release's own date. So `utc = MT5 time − the export's offset` (from the
info file). Read with each date's own offset, a winter NFP lands at 9:30 New York — one hour late.

Proof (printed by every build): with the export's offset, 89 of 142 red / orange event kinds sit at one single New
York time each (every NFP at 8:30, every FOMC statement at 14:00 except the two emergency cuts); with each date's own
offset, 3 of 142. Gold agrees: its tallest 1-minute candle sits at the release minute for most NFP / CPI / Fed
releases (the build prints the counts for both readings). The build refuses to write if any NFP is not at 8:30 New
York.

## What is known when (the no-future rule)

- Known in advance: the time, event, kind, importance, **forecast** and previous number of a scheduled release.
- Known from the release minute on: the **actual** number and the revised previous one.
- Not known before it happens: an **unscheduled** release. The Fed labels three FOMC statements of 2016–2023
  "(unscheduled)" or "(notation vote)": 3 Mar 2020 (10:00 New York), Sunday 15 Mar 2020 (17:00) and 23 Mar 2020
  (8:00). Their decision, statement and press conference rows have `scheduled = False`. MT5 has no row for 23 Mar
  2020, so that one comes from the Fed's page (`source` says so) — the only row not from MT5.

`as_known_at(events, t)` gives the table as the world knew it at `t`. `tests/test_news.py` runs every function
through `guards.assert_no_future` with the table as known at the last bar, and shows the test catches two peeks.

## What is in it — and what is not

- **Actual numbers:** MT5 has one for every number-type row 2016–2023, and it is the **first-released** number (it
  equals the next release's "previous", not its "revised previous", on 99.9% of rows). So nothing is filled from
  ALFRED; the official pages check them instead (`official.py`): BLS's archived news releases (NFP, the
  unemployment rate, CPI: day, time, first number), the Fed's meeting and statement pages (FOMC: day, time, the
  "(unscheduled)" / "(notation vote)" labels), BEA's news archive (GDP: day, time). The results: `official_check.csv`
  and `plan/work/L12-notes.md` section 4.
- **Forecasts: MT5 has none in 2016 and few in the first half of 2017** (0% in 2016, 57% in 2017, 95–100% after).
  ALFRED cannot fill this (it has no forecasts). `no_forecast_why` says which: "MT5 has none in 2016 to mid-2017",
  "none for this kind" (FOMC, speeches, auctions — nothing to forecast), or "MT5 has none for this release".
- 🔴 **MT5's forecast is NOT the economists' consensus** (found in L12b.1, 30 Sep 2026). 8 May 2020 jobs report:
  MT5 +139K, the consensus −22,000K; Sep–Nov 2020 MT5 "forecast" −74,433K, +78,356K, −76,324K jobs. Never compute a
  surprise from this table's `forecast`. The surprise is `surprise.py` → `data/news/surprises.parquet` (Forex
  Factory's forecast — Tejas's own calendar — and the 2-year yield for the Fed). How often the two agree: see the
  section below and `plan/work/L12b1-notes.md`.
- **Importance** is MT5's rating. Tejas's news is Forex Factory red + orange (T7); MT5 high + moderate stands in.
  MT5's "moderate" is wider than Forex Factory's orange (bond auctions, oil rig counts, CFTC positions, many Fed
  speakers). Which kinds move gold is measured, not assumed (plan/work/news.md NQ1; L12b.2).
- `kind` is Claude's grouping of MT5's names (`calendar.KINDS`): jobs, inflation, fed, fed_speaker, growth, housing,
  energy, auction, positions, other.

## The surprise (L12b.1)

```
python -m machine.news.forexfactory   # Forex Factory's US weeks 2015-2023 -> data/news/forexfactory/ (once, ~40 min)
python -m machine.news.yields         # the 2-year yield 2015-2023 -> data/news/dgs2.csv (in git; = FRED DGS2)
python -m machine.news.surprise       # -> data/news/surprises.parquet + every check printed
```

```python
from machine.news import surprise
s = surprise.load("2019", "2020")                        # one row per calendar row
nfp = s[(s["report"] == "Jobs report") & s["headline"]]   # the report's number: the headline payrolls
surprise.latest(bars.index + pd.Timedelta(minutes=5), s, "Jobs report")   # the last jobs z KNOWN at each moment
```

| column | meaning |
|---|---|
| `report`, `rank`, `headline` | which report the number comes in; `headline` = THE number used for it that day (`signs.csv`) |
| `method` | `forecast` (actual − Forex Factory's forecast) · `2y yield` (the Fed: 2-year yield after − before, daily, measured AFTER the release) · `flagged` (not scored) |
| `dollar_surprise` | sign × surprise: above 0 = good for the dollar (claims and unemployment turned round) |
| `usual_miss`, `past_n`, `z` | the size against the same event's last 24 surprises known BEFORE it (each capped at 4 × the middle one) |
| `group` | big good for the dollar (z ≥ 1) · small · big bad for the dollar (z ≤ −1) · not scored (`why_not_scored`) |
| `ff_colour` | Forex Factory's colour for that release — his red / orange (T7) |
| `known_from` | from when the surprise is known: the release minute; the Fed's number the next quote day 16:15 New York |

`data/news/signs.csv` (in git) has one line per MT5 event with its `why`. No gold price is read here (that is L12b.2).

## The reaction (L12b.2)

```
python -m machine.news.reaction          # 2016-2020 -> data/news/reactions.parquet + reaction_controls.parquet (1 minute)
python -m machine.news.reaction --show   # the table per report again, from the files
python -m machine.news.reaction --all    # 2016-2023 - refused until every news belief has opened its check years
```

One row per release **minute** (the jobs report's numbers are one row), measured on the 1-minute bars from 60 minutes
before to 240 after. `reaction_controls.parquet` holds the same measures at the same New York clock minute on days with
**no release within 60 minutes either side** — the fair comparison.

```python
from machine.news import reaction
ev = reaction.load("events")                       # the tune years 2016-2020 (the default)
ct = reaction.load("controls")
ev = reaction.load("events", years=(2016, ..., 2023), lookup=True)   # L12b.3 / L17a: a LOOKUP to show him, never a test
ev = reaction.load("events", years=config.CHECK_YEARS, check_for=("belief-news-trap", "good-or-bad-news"))
reaction.type_table(ev, ct)                        # per report: first-hour range against a quiet day, minutes to calm
```

| columns | meaning |
|---|---|
| `colour`, `his_news`, `reports`, `his_reports`, `report`, `z`, `group` | the strongest Forex Factory colour in the minute (his news = red or orange); every report in it; the ones whose own row is red / orange; the main one and its surprise |
| `gold_way` | +1 the news was good for gold (a BIG surprise, bad for the dollar), −1 bad for gold, 0 small / not scored / two big ones that disagree (`mixed`). `gold_way_fed`: the Fed's days (the 2-year yield, known only the next day) |
| **known before** (`reaction.CONTEXT`) | `p0` (the last price before), `session`, `bias_h` / `bias_d` / `bias_w`, `up_level_*` / `dn_level_*` (his nearest level each side: dollars, share of a day, kind), `adr`, `adr_used`, `day_pos`, `before_120`, `pre_hi` / `pre_lo` / `pre_range` / `pre_5m`, `normal_5m` |
| **the reaction** (`reaction.REACTION`) | `move_1` / `_5` / `_15` / `_60` / `_240` / `move_close` (+ `_adr`), `up_60` / `dn_60` (+ their minute), `range_60`, `hr_side` / `hr_min` / `hr_race` (the hour before is the range: the first side broken, and back or on), `lv_side` / `lv_min` / `lv_kind` / `lv_dist` / `lv_race` (the same with his levels), `calm_pre` / `calm_normal` (minutes until calm; empty = not within 4 hours), `has_bars`, `open_240` |

Everything under "known before" passes the cut-off-the-future test (`tests/test_reaction.py`, made-up and real bars);
the reaction columns are what happened AFTER and are never a signal at the release. Adding 2021–2023 changed no value
of 2016–2020 (checked on the whole table, `plan/work/L12b2-notes.md`).

## The beliefs (L12b.2)

```
python -m machine.news.beliefs tune      # every version on 2016-2020, one trial file each; one version per belief chosen
python -m machine.news.beliefs check     # opens 2021-2023 ONCE per belief for the chosen version: the answer
python -m machine.news.beliefs rules     # the rules, as printed in every report
```

`beliefs.BELIEFS` holds each belief: his words, what is counted on each release, and the comparison (the same clock
minute, weekday and year with no release — or a coin, for a belief about good or bad news). The answer is one of
**holds · the opposite holds · chance — needs fresh years · not enough events to tell · chance**, by six rules fixed in
advance (`beliefs.RULES`, fingerprint pinned by `tests/test_news_beliefs.py`). No rupees: a belief is a count. Reports:
`plan/work/beliefs/`; trial files of kind `belief`; the chosen versions: `plan/work/beliefs/news-chosen.json`. A new
reading of a belief whose check years are open is a new version judged inside 2016–2020 only.

## "The last 10 times" (L12b.3)

```
python -m machine.news.similar pages               # plan/work/news/<news>.md + pictures, one per news with 20+ releases
python -m machine.news.similar look nfp 2023-10-06 # one lookup printed (--before: the number not out yet)
```

```python
from machine.news import similar
tab = similar.releases()                                           # a release = the minute a report's HEADLINE came out
q = similar.query_of(tab, "Jobs report", "2023-10-06 15:30", when="after")   # its own surprise + the situation before
res = similar.lookup(tab, q)                                       # the 10 most like it, released AND done before it
res.lines                                                          # each with all days, the same kind of surprise,
                                                                   # the last 3 years, "10 is few", and luck
```

**A LOOKUP, not a trained prediction model** — the rule was fixed before any lookup ran and is never tuned on what gold
did (`plan/work/L12b3-notes.md` section 1). Same report; after the number is out, the same surprise group of L12b.1;
then the fewest steps (surprise, session, day and week bias, his nearest levels, the day's range used, the 2 hours
before, gold's day range). Only 2016–2023 (T14): `lookup` refuses a table with a 2024+ release, whatever the question's
date — so L17a can ask about a live minute and still see only 2016–2023. `similar.picture` draws one release on one
5-minute chart (the release as a `draw.Moment` line, his two nearest levels, the one swept first). `early_starts`
marks the 11 release minutes whose move began in the minute stamped before them (the broker's clock a few seconds
behind) — shown on the pages, never matched on; reaction.py's p0 already holds that move's start there.

## For the live alerts later (L17a)

`mql5/L17a_CalendarService.mq5` — an MT5 **service** (no chart), written and compiled in L12, **not run**:
`CalendarValueLast(change_id, ...)` every few seconds appends each new or changed value (an actual number coming
out) to `Common\Files\ML-Bot\calendar_live.tsv`; every hour it rewrites the next 7 days to `calendar_week.tsv`
(temp file + move, so the bot never reads half a file); `heartbeat.txt` holds the last write time and the server's
offset now. Every line carries `server_minus_gmt_seconds`, so the bot turns it into UTC the same way as here. The
bot reads 2024+ news only through `guards.check_dates(..., unlock=True, use="news-values")` (T14). A stale heartbeat
means a dead feed (likhith.md B21). XM's server clock must be measured again (machine/data/README.md).
