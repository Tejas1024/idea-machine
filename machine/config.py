"""Every path and setting the machine uses. Change a value here, never inside a block.

Each setting says who chose it: Tejas (with the LEARNED number) or Claude.
Tejas's fixed trading rules are NOT here — they live in machine/rules.py (L6.1).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- paths ------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                          # not in git (.gitignore: /data/*), except usd_inr.csv and swap.csv
SEALED_DIR = DATA_DIR / "sealed"                  # 2024+ rows, kept apart and never read (L1)
SEALED_LOG = DATA_DIR / "sealed_access.log"       # one line per unlock of the sealed years

PLAN_DIR = ROOT / "plan"
PROGRESS_FILE = PLAN_DIR / "PROGRESS.md"
WORK_DIR = PLAN_DIR / "work"
IDEAS_DIR = WORK_DIR / "ideas"                    # one folder per idea: plan/work/ideas/<idea>/
TRIALS_DIR = WORK_DIR / "trials"                  # one tracked JSON per test run (likhith.md R14)
T14_DECISIONS = WORK_DIR / "T14-decisions.md"     # Tejas's written yes on sealed years / live prices (T14)
# The final exam on 2024-2026 opens only with his yes AGAIN on that day, naming the ideas (T14 question 3, 29 Sep 2026:
# "If you think it's best then fine"). The file needs a "his yes:" line and an "ideas:" line.
FINAL_EXAM_YES = WORK_DIR / "final-exam-yes.md"
# Every picture Tejas has been shown, one JSON line each (append-only): "never the same picture twice" (#117).
# Both brothers' copies merge line by line (.gitattributes). machine.charts.lint.mark_shown adds a line.
SHOWN_PICTURES = WORK_DIR / "pictures-shown.jsonl"
SAMPLES_DIR = WORK_DIR / "samples"                # the L4 sample pictures
# Ideas the old project already tested (L5-prior): one CSV line per row of prior-tries.md. The referee reads the CSV.
PRIOR_TRIES_MD = WORK_DIR / "prior-tries.md"
PRIOR_TRIES_CSV = WORK_DIR / "prior-tries.csv"

# The 3 raw price files. Not in git — copy them by USB or drive (plan/README.md section 8).
RAW_DIR = ROOT / "old" / "build" / "data"
RAW_M1 = RAW_DIR / "mt5_xauusd_m1.csv"            # MT5 server time, bid prices, spread in points
RAW_M5 = RAW_DIR / "mt5_xauusd_m5.csv"            # same clock as RAW_M1
RAW_DUKA = RAW_DIR / "gold_5m_dukascopy.csv"      # UTC, from 2018-12-31

# The clean price data, made from the raw files by `python -m machine.data.convert` (L1). Only rows before
# SEALED_FROM; the later rows sit in SEALED_DIR as raw text, never parsed. Read bars with machine.data.bars.load_bars.
GOLD_M1 = DATA_DIR / "gold_m1.parquet"                # MT5 1-min bars, server clock, bid prices: THE price data
GOLD_M5_MT5 = DATA_DIR / "gold_m5_mt5.parquet"        # MT5 5-min bars: only to check the 5-min bars made from M1
GOLD_M5_DUKA = DATA_DIR / "gold_m5_dukascopy.parquet" # Dukascopy 5-min bars, UTC: only to measure the server clock
DATA_HASHES = DATA_DIR / "DATA_HASHES.txt"            # raw SHA-256 + output content hashes (trial files record them)
EXPECTED_HASHES = ROOT / "machine" / "data" / "EXPECTED_HASHES.txt"  # in git: the data every PC must have
SERVER_OFFSET_CSV = DATA_DIR / "server_offset.csv"    # the measured server clock, week by week (L1)
BARS_CACHE_DIR = DATA_DIR / "cache"                   # built bars, keyed by the data's content hash + the code

# Money (L2). Both files are small and IN git (.gitignore lets these two through), so every PC and CI has them.
USD_INR_CSV = DATA_DIR / "usd_inr.csv"    # Rs per $, monthly average 2016-2023 (RBI, then FBIL): python -m machine.data.usd_inr_fetch
SWAP_CSV = DATA_DIR / "swap.csv"          # the broker's swap, one row per reading from MT5: python -m machine.data.swap_read

# The news calendar (L12). Not in git (data/). machine/news/README.md says how each file is made.
# MT5 on Tejas's PC: the Exness-branded terminal, logged into MetaQuotes-Demo = the server the price data came from, so
# the calendar's server times are on the price data's measured clock (machine/data/clock.py). Found 28 Sep 2026.
MT5_TERMINAL = Path(r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")
NEWS_DIR = DATA_DIR / "news"
NEWS_RAW_DIR = NEWS_DIR / "raw"                   # what MT5 exported, 2015-2023, as it came (+ its info file)
NEWS_SEALED_DIR = NEWS_DIR / "sealed"             # the 2024+ rows, copied as bytes, never parsed (#96; T14 use 2 later)
NEWS_EVENTS = NEWS_DIR / "events.parquet"         # THE calendar: python -m machine.news.build
NEWS_OFFICIAL_DIR = NEWS_DIR / "official"         # BLS / Fed / BEA pages as downloaded (the check of MT5's rows)
NEWS_OFFICIAL_CHECK = NEWS_DIR / "official_check.csv"  # one row per release compared: MT5 vs the agency
NEWS_FF_DIR = NEWS_DIR / "forexfactory"           # Forex Factory's US events, one JSON per week 2015-2023 (L12b.1)
NEWS_SIGNS = NEWS_DIR / "signs.csv"               # per event: which way is good for the dollar (L12b.1; in git)
NEWS_DGS2 = NEWS_DIR / "dgs2.csv"                 # the 2-year Treasury yield, daily 2015-2023 = FRED DGS2 (in git)
NEWS_YIELDS_DIR = NEWS_DIR / "yields"             # the Treasury's yearly files + FRED's DGS2 file, as downloaded
NEWS_SURPRISES = NEWS_DIR / "surprises.parquet"   # THE surprise table: python -m machine.news.surprise
NEWS_REACTIONS = NEWS_DIR / "reactions.parquet"   # how gold reacted, per release minute: python -m machine.news.reaction
NEWS_REACTION_CONTROLS = NEWS_DIR / "reaction_controls.parquet"  # the same at that clock minute on days with no release
BELIEFS_DIR = WORK_DIR / "beliefs"                # one page per belief: what is counted, the comparison, the result

# Price step of the data: 1 point = $0.01. READ from MT5 on 28 Sep 2026: symbol_info("XAUUSD") on MetaQuotes-Demo
# (the server the price data came from; Tejas's PC, the Exness-branded terminal) gave point 0.01, digits 2, contract
# 100 oz per lot, lot min 0.01, step 0.01. Same as the old code (old/ml/agent/skill.py POINT = 0.01), and CHECKED
# against the data: every 2016-2023 M1 price is a whole number of cents (tests/test_data.py). The live trial reads
# the point from its own broker again (likhith.md B21).
POINT = 0.01
POINT_SOURCE = ("MT5 symbol_info('XAUUSD').point on MetaQuotes-Demo, 28 Sep 2026 = 0.01 (digits 2, 100 oz/lot); "
                "checked: every 2016-2023 M1 price is on the 0.01 grid")

# ---- years ------------------------------------------------------------------------------------------------------
# Tejas, LEARNED #96: "No, don't touch those years" (also #104). Compared on the data's own clock.
SEALED_FROM = "2024-01-01"
TUNE_YEARS = (2016, 2017, 2018, 2019, 2020)       # used for choosing
CHECK_YEARS = (2021, 2022, 2023)                  # hidden while choosing; opened ONCE per idea (check-opened.json)

# ---- money ------------------------------------------------------------------------------------------------------
# Tejas, LEARNED #54 and #115: Rs 20 in + Rs 20 out per 0.01 lot. ALL-IN: it already includes the spread. No swap.
COST_INR_PER_POSITION = 40
LOT_STEP = 0.01                                   # Tejas, LEARNED #116: lots only 0.01, 0.02, 0.03 ...
# 0.01 lot = 1 ounce, so a $1 move = $1 at 0.01 lot (Tejas's decision, likhith.md section A). READ from MT5 on
# 28 Sep 2026: symbol_info("XAUUSD").trade_contract_size = 100 on MetaQuotes-Demo.
OUNCES_PER_LOT = 100
# Floor for the spread when a bar says 0 (about 3% of MT5 bars do). Used only for the ask-side trigger (L3) and the
# labelled stress column — never added on top of the Rs 40. Claude's pick: the old engine's newest value
# (old/ml/agent/manager.py DEFAULTS; old/build/fastbt.py used 8). 1 point = $0.01.
MIN_SPREAD_POINTS = 14.0

# ---- the tester (L3) ---------------------------------------------------------------------------------------------
# Tejas, LEARNED #55 (14 Sep 2026): one trade at a time — a FIXED rule (T12 table, M18). A number above 1 is only
# ever a TEST version ("one per entry type"), never his rule. A limit order still WAITING (not filled) is cancelled by
# a newer setup, which is taken — Tejas, 28 Sep 2026: "Cancel and take new" (learned-inbox 0203; management M39).
TRADES_AT_ONCE = 1

# ---- the referee (L5) --------------------------------------------------------------------------------------------
# The numbers of the rules that decide "works" (machine/referee/verdict.py prints them in every report). They come from
# the plan (plan/likhith.md L5) — Claude's numbers, not Tejas's rules. Changing one changes verdict.RULES, and a test
# pins its fingerprint, so a change after seeing results shows in git.
P_LIMIT = 0.05                  # one-sided weekly-block bootstrap: p below this for "Rs per trade > 0"
RANDOM_RUNS = 200               # random-entry runs per judged idea
RANDOM_BEAT_SHARE = 0.95        # the idea's Rs per trade must be above this share of the random runs
MIN_TRADES = 30                 # fewer trades than this can never be "works"
WITHOUT_BEST = 3                # still above Rs 0 a trade without the best 3 trades
BOOTSTRAP_RESAMPLES = 2000      # weeks drawn again this many times
# Fake levels: the idea's level (and its stop and targets with it) moved by these many times the trade's stop
# distance, up and down. Claude's reading of "the level moved to its neighbours" (L5): a neighbour is a price a
# stop-distance or so away, where a real level would not be. "Beats its fake levels" = above this share of them.
FAKE_LEVEL_STEPS = (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)
FAKE_LEVELS_BEAT_SHARE = 0.75   # Claude's number: above 6 of the 8 moved copies
# The power table: edges of these many rupees a trade planted in the idea's own noise, at these many trades.
POWER_EDGES_INR = (10, 25, 50, 100)
POWER_TRADES = (30, 100, 300)
POWER_SIMS = 200                # planted samples per cell
POWER_RESAMPLES = 500           # bootstrap draws per planted sample (fewer than 2000 only for speed; same test)
POWER_NEEDED_EDGE = 100         # "not enough trades to tell" when a Rs 100 edge would be caught less than ...
POWER_NEEDED_SHARE = 0.5        # ... half the time
# Choosing between versions at TUNE time only: PBO by CSCV (the chance the best version on half the weeks is below
# the middle on the other half) splits the weeks into this many blocks.
CSCV_BLOCKS = 16
# Later versions are judged by walk-forward inside the tune years: choose on the years before, judge the next year.
WALK_FORWARD_JUDGE_YEARS = (2018, 2019, 2020)

# ---- the learner (L13.1): conditions and slices ----------------------------------------------------------------
# The numbers of the slice rules (likhith.md section D; machine/learner/slices.py prints them in every result and a test
# pins their fingerprint). Claude's numbers, not Tejas's rules; fixed before any condition table was seen
# (plan/work/L131-notes.md).
SLICE_HALVES = ((2016, 2017, 2018), (2019, 2020))  # "the same sign in 2016-2018 and 2019-2020" (section D)
SLICE_MIN_TRADES = 30          # a part of a slice (London) and the rest each need this many trades to be tested
SLICE_SHUFFLES = 2000          # the shuffled-label test: labels shuffled among the same year's trades this many times
SLICE_INTERVAL = 0.90          # the interval shown for Rs a trade: the middle 90 of 100 weekly redraws
LEARNER_KIND = "learner"       # the learner's trial files (plan/work/trials/): never counted as an idea's answer

# ---- the learner's model and the pooled view (L13.2) ------------------------------------------------------------
# Claude's numbers, not Tejas's rules; fixed before any idea's model or pooled table was made, each measured first on
# made-up trades (python -m machine.learner.selftest --model / --pooled; plan/work/L132-notes.md).
MODEL_KIND = "learner-model"    # the model's trial files: never an idea's answer, never a try
POOLED_KIND = "learner-pooled"  # the pooled view's trial files: never an idea's answer, never a try
MODEL_PURGE_DAYS = 7            # a trade teaches the model of year Y only if it CLOSED 7 days or more before Y began
MODEL_SHUFFLES = 200            # the model check: the money shuffled inside each year this many times
MODEL_ROUNDS = 100              # LightGBM: trees per model
MODEL_PARAMS = {                # LightGBM's settings (the same for every idea; no tuning on any idea's trades)
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 4, "max_depth": 2, "min_data_in_leaf": 30,
    "min_data_per_group": 30, "max_cat_to_onehot": 8, "cat_smooth": 10.0, "lambda_l2": 10.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "deterministic": True, "force_col_wise": True, "num_threads": 1, "verbose": -1,
}
FILTER_ABOVE_INR = 0.0          # the take/skip filter takes a setup when its model guesses more than this, after cost
POOLED_RANDOM_RUNS = 10         # the pooled view: random-entry runs per tested idea ...
POOLED_RANDOM_MAX = 6500        # ... but at most this many random trades per tested idea (whole runs)

# ---- the document search (L8b): his #129 rule as code ------------------------------------------------------------
STRATEGY_DIR = ROOT / "strategy"
SEARCH_CACHE_DIR = DATA_DIR / "cache" / "search"  # the pieces' embeddings, one file per model (not in git)
# The embeddings. Default: a local sentence-transformers model (downloads ~130 MB once, then works offline). With
# VOYAGE_API_KEY set, Voyage AI instead (Anthropic has no embeddings API; Anthropic's docs point to Voyage).
# SEARCH_EMBEDDINGS=off in the environment = words only (BM25). All Claude's choices, not Tejas's.
SEARCH_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
SEARCH_LOCAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # the model's own advice
SEARCH_VOYAGE_MODEL = "voyage-3.5"
# The FOUND / NOT FOUND lines are in machine/search/rank.py (one set per embedding model: each model has its own
# scale). They are chosen on tests/evals/search-dev.jsonl, NEVER on the exam (tests/evals/search.jsonl).
SEARCH_CITE_MAX = 3                # at most this many places cited in one FOUND answer

# ---- the talker (L8): his words -> an idea form, one Claude tool call -----------------------------------------
# HOW it reaches Claude - Tejas, 28 Sep 2026: no money on the Claude API (inbox 2026-09-28-2020); "You can use the
# claude itself right the one I'm using now?" -> "claude-code": his own Claude Code login (Claude Pro) through
# `claude -p` on his PC - no key, no bill per call, his plan's usage allowance. "api" = ANTHROPIC_API_KEY, paid per
# token (a PC or CI with a key). The model is Claude's choice, not Tejas's (the claude-api skill's default, Claude
# Opus 5; his Pro plan runs it - measured 28 Sep).
TALKER_BACKEND = "claude-code"
TALKER_MODEL = "claude-opus-5"
# Tejas, 29 Sep 2026 (inbox 0600): "Yes" to effort high and one fix-up call. Before: medium, no fix-up (runs 2 and 4 of
# the exam: too many small slips, not steady - plan/work/L8-notes.md).
TALKER_EFFORT = "high"
TALKER_FIXUPS = 1              # one fix-up call when code refuses the form or its checks require a question

# ---- the Telegram bot (L9a): the only poller, on Tejas's PC -----------------------------------------------------
# Keys in .env (never committed): TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID on Tejas's PC (copied from the old bot's
# old/ml/agent/agent_config.json by `python -m machine.telegram.bot --copy-old-keys`); TELEGRAM_TEST_BOT_TOKEN +
# TELEGRAM_TEST_CHAT_ID on Likhith's PC (his own test bot - `--test` writes only into TELEGRAM_TEST_DIR).
LEARNED_MD = STRATEGY_DIR / "LEARNED.md"          # add-only; numbered entries come only from learned_merge on main
INBOX_DIR = STRATEGY_DIR / "learned-inbox"        # his new words wait here for a number (merged/ = numbered)
PICTURES_DIR = STRATEGY_DIR / "pictures"          # his pictures, cropped (the phone's top bar and buttons cut off)
TELEGRAM_DIR = DATA_DIR / "telegram"              # the bot's own files: photos/, state.json, messages.jsonl, sent.jsonl
TELEGRAM_TEST_DIR = DATA_DIR / "telegram-test"    # --test: EVERYTHING the test bot writes goes here, nothing tracked
TELEGRAM_POLL_SECONDS = 25                        # one long poll waits this long for a message (Claude's number)
# His phone's screenshots (every one in strategy/pictures/ is 1200 x 2670, measured 29 Sep 2026): the status bar's
# icons (clock, notification icons, battery) sit on rows 39-80, so the bar is the top 120 rows = 4.5%; the phone's
# three buttons sit on rows 2581-2624, so that bar is the last 134 rows = 5%. A picture at least 1.7 times taller
# than wide is a phone screenshot; any other picture is copied uncut. Claude's numbers, from his screenshots.
PHONE_SHAPE_MIN = 1.7
PHONE_STATUS_BAR_SHARE = 0.045
PHONE_NAV_BAR_SHARE = 0.05

# ---- the bot's files reach git every night (L9c) -----------------------------------------------------------------
# The bot writes into THIS clone (where it runs). A Task Scheduler job copies those files into a SEPARATE clone that
# only the job uses, commits them to the branch bot/<date>, pushes it and opens a pull request (gh) or sends Tejas the
# link on Telegram. Only files matching these patterns are ever copied (one * = one name, never a folder); data/ and
# .env never (checked again in machine/telegram/nightly.py). Claude's list, from what L9a and L9b write.
BOT_CLONE = ROOT.parent / f"{ROOT.name}-bot"      # c:\ML Bot-bot (the prompt's name); the job makes it the first time
BOT_BRANCH_PREFIX = "bot/"
NIGHTLY_AT = "23:30"                              # India time on his PC; a missed night runs when the PC is back on
NIGHTLY_TASK_NAME = "ML Bot - bot files to GitHub (L9c)"
BOT_GIT_FILES = (
    "strategy/learned-inbox/*-telegram*.md",      # his lessons and answers from Telegram (not merged/: the Action's)
    "strategy/pictures/*-his-answer*.jpg",        # his drawn answers, cut
    "strategy/pictures/*-telegram-m*-his-picture*.jpg",  # any other picture he sent, cut
    "plan/work/ideas/*/answers.md",               # his answers, word for word (append only)
    "plan/work/ideas/*/state.json",               # where each idea is
    "plan/work/ideas/*/form.json",                # a reading locked on his yes (L9b)
    "plan/work/ideas/*/questions/*",
    "plan/work/ideas/*/questions/*/*",
    "plan/work/question-queue/*",                 # the weekly questions and batches (L9b)
    "plan/work/question-queue/*/*",
    "plan/work/question-queue/*/*/*",
    "plan/work/trials/*.json",                    # one per test run (R14)
    "plan/work/intake-queue.jsonl",               # new ideas typed as "IDEA ..." (L10b runs them)
    "plan/work/telegram-outbox.jsonl",            # lines TO him
    "plan/work/pictures-shown.jsonl",             # every picture he was shown (#117)
    "plan/work/ideas/*/*",                        # the intake (L10b): each idea's words, search, form, pictures, report
    "plan/work/intake/*",                         # the intake's own files (the 10-in-a-row question, SHOWN sends)
    "plan/work/block-requests.md",                # blocks an idea waits for (L7b, written by the intake too)
    "plan/work/labels/*/*.jsonl",                 # every label chart sent + his yes / no / not sure (L15.1)
)
# Files only ever ADDED to (never rewritten): when his PC and GitHub both added lines, the job keeps both.
BOT_APPEND_ONLY = (
    "strategy/learned-inbox/*-telegram*.md",
    "plan/work/ideas/*/answers.md",
    "plan/work/intake-queue.jsonl",
    "plan/work/telegram-outbox.jsonl",
    "plan/work/pictures-shown.jsonl",
    "plan/work/intake/sends.jsonl",
    "plan/work/labels/*/*.jsonl",
    "plan/work/ideas/*/slices.jsonl",             # slices chosen before any money, research versions (L14.1)
    "plan/work/ideas/*/versions.jsonl",
    "plan/work/ideas/*/research-log.md",          # what the research agent did, in order (L14.2)
)

# ---- the intake (L10b) -------------------------------------------------------------------------------------------
# The bot starts `python -m machine.intake run` after an IDEA or an answer, and at least this often (a run that
# stopped goes on; the weekly batch goes on its day). Claude's number.
INTAKE_EVERY_MINUTES = 30

# ---- budgets (Claude's numbers, settings — not Tejas's rules) --------------------------------------------------
VERSION_BUDGET = 20                               # at most 20 versions (different rules) per idea
SLICE_BUDGET = 8                                  # at most 8 slices per idea, chosen BEFORE running
QUESTION_BUDGET = 10                              # at most 10 picture questions to Tejas per weekly batch

# ---- the research tools (L14.1): machine/mcp, the MCP server the research agent (L14.2) calls ------------------
# Not in git: each version's signals and trades (fingerprinted - lesson B19), the charts it drew, and one line per
# refused call (a trick to reach 2024+ or the check years early is written down, not only refused). MLBOT_MCP_DIR in the
# server's environment moves it (the tests' server process writes into a temporary folder, never into data/).
MCP_DIR = Path(os.environ.get("MLBOT_MCP_DIR") or DATA_DIR / "mcp")
# A chart of a stretch of time: at most this many days on each chart (Claude's numbers: readable in 5 seconds).
MCP_CHART_DAYS = {"5m": 3, "15m": 7, "1h": 20, "4h": 60, "1d": 250}

# ---- the research agent (L14.2): machine/agent ------------------------------------------------------------------
# Claude PLANS (his Claude Code login, the planning tools only); code checks the plan, saves it, then runs exactly it.
# All Claude's numbers, not Tejas's rules.
RESEARCH_AGENT = True              # the intake (L10b) researches an idea with the agent after his YES to the 3 setups
                                   # (False: L10's tune of the readings only, as before L14.2)
RESEARCH_MODEL = "claude-opus-5"   # the talker's model (L8: measured on his Pro plan, 28 Sep)
RESEARCH_EFFORT = "high"
RESEARCH_FIXUPS = 1                # code refused the plan -> ONE more call with the reasons (the talker's rule, L8)
RESEARCH_TRIES = 3                 # a planning call that got NO answer (his usage limit, no network) is tried again at
                                   # the intake's next run; after 3, a session looks (the talker's TALKER_TRIES)
RESEARCH_PLAN_SECONDS = 1800       # one planning call: Claude reads the idea and searches his documents (minutes)
RESEARCH_MAX_VERSIONS = 16         # a plan tries at most 16 of the family's 20: 4 stay for his own "change one thing"
                                   # after the report (a changed idea is a child - the same family, the same 20)
RESEARCH_STUCK_IN_A_ROW = 3        # the stop rule: this many planned versions in a row cannot run -> stop, a session
                                   # looks (something is broken; never a stop because of money)

# ---- the question loop (L9b) -------------------------------------------------------------------------------------
QUESTION_QUEUE_DIR = WORK_DIR / "question-queue"  # one folder per unclear part; batches/<date>/ per weekly batch
# The readings of an unclear part AGREE when this share of their signals (TUNE years) is the same order in every
# reading: then there is no question (the prompt's number: ">= 95% same -> no question").
READINGS_AGREE_SHARE = 0.95
# Moments asked per unclear part: one per reading, at least 2, at most 4 (the prompt's "2-4 per unclear part"; how
# many inside that range is Claude's pick: a reading gets a moment where it is the one that differs). Plus ONE more
# moment, never asked, for the confirm picture ("I read your answer as this - yes?" on another day: #117, never the
# same picture twice).
QUESTION_MOMENTS_MIN = 2
QUESTION_MOMENTS_MAX = 4
# His replies that mean "you already have this" (T5: reply "it's in my documents"): no lock; the question and his
# words become a fresh row for the search exam (plan/work/L8b-notes.md plan step 2).
SEARCH_FRESH_EXAM = ROOT / "tests" / "evals" / "search-fresh.jsonl"

# ---- labels for "his eyes" (L15.1) -------------------------------------------------------------------------------
# One of his words, answered yes / no / not sure on charts. The first labels come from his drawn answers in
# strategy/pictures/ (machine/labels/drawings.py); then - ONLY after his T9 opt-in - 10 new charts a day on Telegram.
LABELS_DIR = DATA_DIR / "labels"                  # <word>.parquet (the labels, images + times) and <word>/ (the charts)
LABELS_WORK_DIR = WORK_DIR / "labels"             # in git: <word>/charts.jsonl (every chart sent), replies.jsonl (his)
LABELS_OPT_IN = LABELS_WORK_DIR / "opt-in.md"     # T9: his words "opt in"; without this file no chart is ever sent
LABELS_PER_DAY = 10                               # the prompt's number (plan/likhith.md L15.1)
LABELS_AT = "21:00"                               # India time on his PC, after the New York session (Claude's; T9 may move it)
LABELS_TASK_NAME = "ML Bot - label charts (L15.1)"
LABELS_YEARS = TUNE_YEARS                         # the daily charts come from 2016-2020 only: the hidden years stay hidden
LABELS_WINDOWS = ("week",)                        # the zone block's 5-day window, drawn as 5 days of hour candles (Q79)
# The day's charts by how quiet the band's candles are (zones.py "calm"): the steps start at his rule's number
# (rules.py what_a_strong_zone_is, 0.74) and go on at these edges; how many charts from each step. Claude's numbers:
# most charts near his line, where a yes and a no are both likely, and some far from it on both sides.
LABELS_CALM_EDGES = (0.85, 1.00)
LABELS_PER_STEP = (3, 3, 2, 2)                    # calm up to 0.74 · to 0.85 · to 1.00 · above 1.00 (sums to 10)
LABELS_WAITING_MAX = 20                           # no new charts while this many wait for his answer (Claude's number)


def env(name: str, default: str | None = None) -> str | None:
    """A key or login from the environment.

    On a PC: the environment, then .env in the repo root (never committed).
    In a cloud session: only the environment (its secrets settings) — no file is read there.
    """
    if name in os.environ:
        return os.environ[name]
    dotenv_file = ROOT / ".env"
    if dotenv_file.is_file():
        from dotenv import dotenv_values

        value = dotenv_values(dotenv_file).get(name)
        if value:
            return value
    return default
