# Design: the AI writes rules, code judges

This document explains the seven decisions that shape the idea machine, and where each one is enforced in code.
The short version: a language model is good at reading a trader's messy words and bad at being an honest judge of
its own work. So the model is only allowed to fill in a strict form. Everything after that (compiling, testing,
counting, judging) is ordinary, tested Python whose rules were written down before any result was seen.

| # | Decision | Enforced in |
|---|---|---|
| 1 | A form, not free code | `machine/ideas/form.py`, `machine/ideas/compile.py`, `machine/talker/translate.py` |
| 2 | Search before asking; ask only where readings disagree | `machine/search/`, `machine/questions/`, `machine/talker/translate.py` |
| 3 | The referee's rules are fixed in advance | `machine/referee/verdict.py` (+ its pinned fingerprint in `tests/test_referee.py`) |
| 4 | Every try is counted | `machine/trials.py`, `machine/referee/tune.py`, `machine/mcp/versions.py`, `plan/work/prior-tries.csv` |
| 5 | Sealed years, and check years opened once | `machine/guards.py` (`check_dates`, `check_opened`), `machine/referee/splits.py` |
| 6 | No peeking at the future, tested | `machine/guards.py` (`assert_no_future`), `machine/tester/engine.py` |
| 7 | The AI plans, code runs exactly the plan | `machine/agent/research.py`, `machine/mcp/` |

---

## 1. A form, not free code

The talker (`machine/talker/translate.py`) makes **one** Claude call with **one** strict tool, `fill_idea_form`. The
model never writes Python. It fills in fields: the ordered steps (each a named building block such as `sweep`,
`break_of_structure`, `gap_of_the_move`), how long each step may wait, the entry, the exit, the targets, filters,
"do not enter" rules, and the parts the words leave open.

Code then checks the answer before anything else happens:

- it must load as a valid `Idea` (types, order, waits, names it points at);
- `compile.check_without_bars` checks the meaning: every block exists, every parameter is known, and **none of the
  trader's 19 fixed rules is changed** (`machine/rules.py`). A form that, say, enters at the far edge of a gap instead
  of the near edge is refused, and the refusal quotes the rule's source;
- every reading of every unclear part must also make a valid form;
- every citation must be one of the quotes the search actually showed the model (no invented sources);
- a form that needs a block that does not exist yet is marked BLOCKED and a build request is written. The model is
  never allowed to approximate a missing block;
- `machine/talker/sanity.py` catches what a model gets quietly wrong: a 0.68 next to "golden ratio", a target on the
  wrong side of the entry, a chart timeframe the data does not have.

When code refuses a form, **one** fix-up call sends the findings back; both calls are counted and the better form is
kept. The talker is measured by an exam (`machine/talker/evals.py`, `tests/evals/golden.jsonl`): on its last run,
fields right 93%, unclear parts found 17 of 17; not met: 1 invalid form, 5 questions that were not needed.

Why a form: a form can be compiled, diffed, hashed, versioned and tested. Two forms with the same blocks, timeframe
and entry are the same idea (a version of it), which is what makes counting tries possible (decision 4).

## 2. Search before asking; ask only where readings disagree

The trader's rule: never ask him something his notes already answer. `machine/search/` indexes everything he wrote
(numbered lessons, notes, rules, the Telegram archive) and ranks pieces two ways, BM25 on words and embeddings on
meaning (a local `bge-small` model, or Voyage), joined by reciprocal rank fusion. Results are re-ordered by an
explicit order of trust (his own newest words first). `answered(q)` returns FOUND with citations, or NOT FOUND with a
sentence naming what was searched and the closest line. That sentence is built by code, never by the model. The
search has its own exam: asks-or-not right 38 of 40; first citation right 19 of 30 (the mark of 27 was not met, and
the README says so).

The form splits whatever is still open into two kinds:

- **ASK**: a meaning only the trader knows ("which high counts: any session's, or only London's?");
- **VERSION**: a number or choice the machine can try by itself ("the break within 12 or 30 bars"). Versions are
  never asked.

An ASK part becomes a question only if its readings **make different trades**. Each reading is compiled on real
prices and the setups are compared. If they are the same, asking would waste his time. Questions go out in one
weekly batch (at most `QUESTION_BUDGET` = 10), as pictures, with no money shown, so an answer cannot be steered by
which reading made more. A reply is read by code only when it is unambiguous ("B"); anything else waits for a human
session. The machine never guesses a meaning. A confirmed answer locks the idea: after that its meaning cannot
change (`machine/ideas/store.py` refuses).

## 3. The referee's rules are fixed in advance

`machine/referee/verdict.py` holds the rules as one text block, printed in every report. Its SHA-256 goes into every
referee trial file, and a test pins the fingerprint, so changing a rule after seeing a result shows up in git. An idea
"works" only if, on the check years 2021–2023, **all** of these hold:

1. **Luck test:** a one-sided weekly-block bootstrap (2,000 redraws); a world with no edge gives this many rupees a
   trade less than 5 times in 100.
2. **Random entries:** it beats at least 95 of 100 of 200 random-entry runs with the same exits, distances, cost
   and trades per month.
3. **Fake levels:** it beats at least 75 of 100 copies of itself with its level moved 0.5–2 stop distances up and
   down. A real level should matter.
4. **Not three lucky trades:** still above ₹0 a trade without its best 3 trades.
5. **Enough trades:** at least 30.
6. **Fresh years:** the old project did not already test this idea on these years.

Otherwise the answer is *chance*, *chance — needs fresh years* (only a forward test on new prices can judge it), or
*not enough trades to tell* (a power table shows a ₹100 edge would be caught less than half the time at this count).

The referee is itself examined (`python -m machine.referee.selftest`). On made-up trades with no edge it says
"works" 0 of 100 times. On the hardest noise (exactly ₹0 a trade after cost) it says "works" about 5 times in 100,
which is the error rate rule 1 allows. It catches a planted ₹100 edge 99% of the time and a ₹50 edge 71% of the
time at 300 trades. The demo runs this exam.

## 4. Every try is counted

Try 20 variations of anything and one will look good by luck. So:

- every test run writes **one tracked JSON trial file** (`machine/trials.py`): the settings, a content hash of the
  price data, the git commit and code hash, the parent idea, the result;
- an idea family may use at most `VERSION_BUDGET` = 20 versions. A version is a different rule (timeframe, or the
  idea's own numbers); slicing the same trades by session or direction is not a version, and at most 8 slices are
  chosen **before** running;
- the referee chooses one version on 2016–2020 by the **Deflated Sharpe ratio**, which discounts for the number of
  versions tried. It reports **PBO** (the probability that the chosen version is overfit, by combinatorially
  symmetric cross-validation);
- `plan/work/prior-tries.csv` lists 78 ideas the earlier project already tested, with their try counts; those years
  are not fresh for them (rule 6);
- after 10 answers in a row that are not "works", the machine stops and asks whether to go on.

## 5. Sealed years, and check years opened once

- **2024 onwards is sealed.** The price loader refuses any timestamp on or after 2024-01-01
  (`guards.check_dates`). It opens only for three uses written in a decisions file (live alerts, live news values, one
  final exam with a fresh yes on the day), never for testing, tuning or choosing, and every opening is logged.
- **2021–2023 are the check years.** They stay hidden while versions are compared on 2016–2020, and open **once
  per idea family**, for the one chosen version (`check-opened.json`, written by `referee/splits.py`, with who and
  why). Opening them again for the same family is refused. Later versions are judged by walk-forward inside
  2016–2020: each judged year's version is chosen only on the years before it.
- When the check years open, the chosen version is compiled again on 2016–2023, and its 2016–2020 signals must be
  **byte-identical** to the tune signals. If a later bar changed an earlier signal, the code read the future, and the
  run stops.

## 6. No peeking at the future, tested

`guards.assert_no_future(fn, df, cut_points)` runs a function on all the data and again on the data cut at each
point. Every value at or before the cut must be identical. Every block, signal and feature has such a test. Signals
carry `known_at` (the close of the bar that made them) and may fill only on a later 1-minute bar. Limit orders fire on
the ask (bid + that minute's spread); a stop and a target in the same minute count as the stop.

Things these tests have caught in this project: a choice of "the deepest gap" made before its window ended (caught
by a test that cuts one minute after every signal); a zone-touch bug found after a first check run (fixed, and both
runs kept on record). The earlier project, without these guards, reported three profits that turned out to be bugs.

## 7. The AI plans, code runs exactly the plan

The research agent (`machine/agent/research.py`) lets Claude design the research for one idea. It sees only three
planning tools: `get_idea`, `list_blocks`, `search_docs` (served over MCP by `machine/mcp/`). It writes a plan: a
hypothesis, at most 16 versions, at most 8 slices chosen in advance, a stop rule. Code checks the plan with the same
rules the tools enforce and **saves it, with its hash, before any version runs**. Then code runs exactly that plan
through the research tools, where every guard above lives inside the tool, not in the prompt. A version that is not
in the plan never runs, and nothing stops early because of money (a stop that looks at results is choosing by
results). On made-up prices with no edge, a real saved plan's first 3 versions came out *chance*.

## What this design does not protect against

- **Wrong meaning, honestly locked.** If the trader confirms a reading that is not what he does, the machine tests
  the wrong idea correctly.
- **Small edges.** At 300 trades a ₹50 edge is caught 71% of the time; smaller edges are mostly invisible.
- **One market, one data source, one clock.** Gold 1-minute bars from one MT5 server, 2016–2023. Another broker's
  prices, clock and costs must be measured again.
- **The cost is one trader's.** ₹40 per position per 0.01 lot, all-in (spread included). A stress column shows the
  real spread and swap.
- **A model in the loop.** The talker and the planner are language models. Their output is checked, but a check can
  only refuse what it was written to see.
