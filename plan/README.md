# The plan (trimmed for the public copy)

The working plan lives in the private repo: two people, about 60 build prompts, a progress table, and the notes of
each step. This page keeps only the shape of it.

## The end product

1. The trader sends an idea in plain words on Telegram.
2. It searches everything he already taught and never asks again what is already answered.
3. It asks only what is still unclear, with pictures, only where readings make different trades, in one weekly
   batch, and never with money shown. Before testing, it shows him 3 trades it found: "is this your idea?"
4. It tests the idea by itself on 2016–2023 gold, with his real cost.
5. It answers in rupees: *works / chance / chance — needs fresh years / not enough trades to tell*, with the worst
   year, the versions tried, the smallest edge it could have seen, and where it works best or fails (only for slices
   chosen before the run).
6. News memory: "the last 10 times this news came out like this, gold did …".
7. Ideas that pass, or need fresh years, go to a live trial with alerts and no money, planned and locked before the
   trial starts. Real money only after a legal check and his yes.

**Done even if every idea says "chance":** the machine works and the reports are honest. Its job is the truth, not
a promise of profit.

## The stages

| Stage | What it adds | Status (1 Oct 2026) |
|---|---|---|
| 0. Clean start | price data on the measured server clock; cost in rupees | done |
| 1. One idea, start to end (v1) | tester, pictures, referee, idea form + compiler, search, talker, Telegram bot, question loop, report, intake, **this public repo** | done (the intake waits for the trader's first phone idea to reach a report) |
| 2. All of his strategy + news | his hedging tester; trendlines, zones, bias, the 1-minute layer; news calendar, surprise, reaction, "last 10 times"; learner (condition tables, LightGBM + SHAP); his beliefs as numbers; his entry types as forms | done |
| 3. It researches by itself | research tools over MCP with the guards inside; the research agent (≤ 20 versions, ≤ 8 slices, the plan saved first) | tools done; the agent built, its first real run waits for the trader's yes |
| 4. His eyes | labels from his drawings; a PyTorch image model vs LightGBM, measured on ≥ 400 charts it never trained on | labels started |
| 5. Every day + live | the PC runs daily (heartbeat, MLflow); Docker + dashboard; live feed + news alerts; paper trades; forward tests; his money plan measured | money plan running |
| 6. Real money | tiny real trades with hard limits | locked: needs a legal check and his written yes |

## The rules every step follows

- 2024–2026 are sealed: never opened for testing, tuning or choosing.
- No peeking: every block, signal and feature passes the cut-off-the-future test.
- Money in rupees at 0.01 lot, ₹40 per position all-in, year by year, the worst year, the number of tries.
- If a number looks too good, hunt the bug before reporting it.
- Every test run leaves one trial file, so every try is counted.
- The trader's fixed rules cannot be changed by any test, form or research run.
- Never guess what he means: queue a question.
- The AI writes rules; code judges.

The design behind these rules: [docs/DESIGN.md](../docs/DESIGN.md). The demo: [work/demo-script.md](work/demo-script.md).
