# What this public copy leaves out

This repo is the public copy of a private working repo, made by `machine/tools/publish.py` (an allow-list: a file
not on its list never comes here). Left out on purpose:

| Path | What it is |
|---|---|
| `strategy/` | the trader's own notes: strategy.md, the numbered lessons (LEARNED.md), his marked chart pictures and the Telegram archive - his private knowledge |
| `old/` | the old system (before 27 Sep 2026), kept for reference only |
| `data/` | the gold prices (1-minute bars 2016-2023, and the sealed 2024+ rows), the news calendar, the bot's files - only four small public tables are in this copy |
| `plan/work/` | his ideas, his answers, the question queue, every trial file (one per test run), the notes of each build step - only prior-tries.csv and demo-script.md are in this copy |
| `plan/PROGRESS.md, plan/tejas.md, plan/likhith.md` | the working plan of the two people who build it |

**The tests:** `tests/conftest.py` sees this file. A test that fails only because it needs one of the files above is
reported as **skipped**, with the file it needs (`pytest -rs` lists them). Tests marked `private` skip here too.
Tests that need the real gold prices skip as before (`data` marker). Everything else runs on made-up data.

**The demo:** `python -m machine.demo` runs every step on made-up prices and made-up notes in `tests/data/demo/`.
