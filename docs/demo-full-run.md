# The full demo run, as it printed (1 Oct 2026)

`python -m machine.demo --full --out demo-output` on made-up prices (a random walk of 1-minute bars 2016-2023, seed
7, no edge). 1,225 seconds on one Windows PC. Only two local folder paths are replaced (`<demo folder>`,
`demo-output/`); nothing else is edited.

```text
THE IDEA MACHINE - a demo on MADE-UP data (made-up notes, a made-up idea, made-up prices with no edge)
prices: 3,002,400 one-minute bars 2016-2023, a random walk (seed 7) - the real gold prices and the trader's notes are private

==== 1. THE IDEA, IN PLAIN WORDS ============================================================================
"sell when the high is taken, then the break soon after and the first gap, sell at the gap"

==== 2. SEARCH THE NOTES FIRST (never ask what is already written) ==========================================
198 pieces of notes (made up) + the machine's own rules, searched by words (BM25)
  "taken" -> NOT FOUND, closest  rules.py what_counts_as_the_break: "SELL: the most recent RED candle in the rise, after which price went above its high (a wick is enough); a wick below its low is the break."
  "first gap" -> NOT FOUND, closest  LEARNED #2: ""I only take the first gap after the break."

==== 3. THE AI WRITES A FORM, CODE CHECKS IT ================================================================
form: sell after a high is taken, the break and the first gap · 5m chart · sell
  step sweep  sweep                a session high is taken (a wick is enough)
  step brk    break_of_structure   break of structure down, soon after
  step gap    gap_of_the_move      the first gap of the breaking move
  entry  limit order at gap.near   ·   exit: stop at brk.extreme   ·   target: the next liquidity
code's checks: status FORM DRAFTED · problems 0 · citations not among the quotes shown 0 · sanity checks failed 0
  unclear U1 'which high counts': ASK him (only he knows what he means)
      A. the high of any session (Asia, London or New York)
      B. only the London session's high
  unclear U2 'how soon the break must come': VERSION (the machine tries each)
      A. the break within 30 five-minute bars (2.5 hours)
      B. the break within 12 five-minute bars (1 hour)
a form that enters at the FAR edge of the gap instead: REFUSED
    - entry.at: 'where_in_the_gap_he_buys' cannot be set to 'gap.far': it is FIXED - what a thing IS, not a rule to test (LEARNED #121). It stays 'near': At the NEAR edge of the gap - the edge price meets

==== 4. ASK ONLY WHERE THE READINGS MAKE DIFFERENT TRADES ===================================================
U1 'which high counts' on 2019-01..2019-03 (made-up prices): reading A 96 setups, reading B 42, the same in both 42
  -> different trades: the question goes into his weekly batch (with A / B pictures, no money shown)
     the #129 sentence it carries: Searched strategy.md, LEARNED.md, strategy-understanding.md, rules, pictures for "sell when the high is taken, then the break soon after and" — closest line: rules.py what_counts_as_the_break: "SELL: the most recent RED candle in the rise, after which price we
(his reply - in the demo a stand-in: U1 = A)

==== 5. TEST EVERY VERSION, IN RUPEES AFTER COST ============================================================
2 versions (2 of the 20 tries an idea may use) on 2019 made-up prices · 0.01 lot · ₹40 a position all-in
  v-c9a68d636f the break within 30 five-minute bars (2.5 hours): 355 trades, won 15.2%, -₹45 a trade, -₹16,080 in all (cost ₹14,200)
  v-3d23dff06e the break within 12 five-minute bars (1 hour): 257 trades, won 15.6%, -₹46 a trade, -₹11,935 in all (cost ₹10,280)
each run wrote one trial file (settings, data fingerprint, code fingerprint, result) - here into a temporary folder; in the real machine into plan/work/trials/, so every try is counted

==== 6. THE REFEREE: RULES FIXED IN ADVANCE, AND ITS OWN EXAM ===============================================
  "works" needs ALL of these, on the check years 2021-2023 (opened once per idea):
    1. The luck test: the idea's weeks are drawn again 2,000 times in a world with no edge;
       that world gives this many rupees a trade less than 5 times in 100.
    2. It earns more a trade than at least 95 of 100 random-entry runs (200 runs: same exits,
       distances, cost and trades per month; random times and random limit levels).
    3. It earns more a trade than at least 75 of 100 of its fake levels (its level moved up and down by
       0.5 to 2 stop distances). An idea with no level: does not apply.
    4. It still earns more than ₹0 a trade without its best 3 trades.
    5. At least 30 trades.
    6. The old project did not already test this idea on these years (plan/work/prior-tries.csv).
  "chance — needs fresh years": the old project already looked at these years for this idea. Only new prices can judge it (a forward test).
  "not enough trades to tell": even an edge of ₹100 a trade would be caught less than 50 times in 100 with this many trades.
  "chance": everything else. It could be luck.
  Later versions of an idea whose check years were already opened are judged the same way, year by year inside
  2016-2020: each judged year's version is chosen only on the years before it.
  (the rules' fingerprint 5b7ca6d6fb76 is in every trial file; a test pins it)
its exam on made-up trades (300 trades an idea, 20 versions each, the best version chosen on 2016-2020 and judged on fresh years):
  100 ideas with no edge (they lose the ₹40 cost): 'works' 0 of 100
  100 ideas with the hardest noise (exactly ₹0 a trade after cost): 'works' 6 of 100
    (if all 20 versions were judged on the same years and the best one kept: 'works' for 64 of 100 - why the check years open only ONCE)
  200 ideas with a real ₹50 edge a trade: 'works' 142 of 200
  200 ideas with a real ₹100 edge a trade: 'works' 198 of 200
  (64 s)

==== 7. THE GUARDS (CODE, NOT PROMISES) =====================================================================
ask for March 2024 -> REFUSED: This touches 2024-03-31 00:00:00; the sealed years start 2024-01-01. Tejas, LEARNED #96: "No, don't touch those years" (also #104). Unlocking needs a  ...
a level from the PAST hour's high -> passes the cut-off-the-future test
a level from the NEXT hour's high -> CAUGHT: cut at 2019-01-02 05:00:00: 'high' at 2019-01-02 04:10:00 is 1042.44 with the future and nan without it. It peeks at the future (likhith.md B1).
test on 2021 before a version is chosen -> REFUSED: The check years (2021) stay hidden while choosing and open ONCE per idea (plan/README.md section 7).

==== 8. THE REFEREE'S ANSWER (2016-2020 to choose, 2021-2023 opened once) ===================================
  v-c9a68d636f: 2061 setups on 2016-2020 (108 s) - the break within 30 five-minute bars (2.5 hours)
  v-3d23dff06e: 1477 setups on 2016-2020 (58 s) - the break within 12 five-minute bars (1 hour)
TUNE  idea-950 · 2 versions on 2016-2020 · 5m bars · MT5 server time · ₹40 per position per 0.01 lot, all-in (LEARNED #54, #115)
  version | what it is | trades | ₹ a trade | ₹ after cost | chance its edge is real (Deflated Sharpe, allowing for every version tried)
  v-c9a68d636f | the break within 30 five-minute bars (2.5 hours) | 1828 | -₹36 | -₹66,140 | 0.00  <- chosen
  v-3d23dff06e | the break within 12 five-minute bars (1 hour) | 1326 | -₹37 | -₹48,703 | 0.00
The chance the choice was luck (PBO): 31 in 100 (12870 ways of cutting the weeks in half)
Trial file: <demo folder>\work\trials\20261001-124829-376937-3a223b-idea-950-v-c9a68d636f.json
  check years opened: 2026-10-01T18:18:29+05:30 for v-c9a68d636f by the demo
  compiled 2016-2023 in 226 s: 2061 signals up to 2020 (the same as the tune), 1219 in 2021-2023
  judged in 724 s: chance

Idea #950 — sell after a high is taken, the break and the first gap
5-minute chart · sell · family of: new (no older idea)
Tested: the break within 30 five-minute bars (2.5 hours)
₹ = after costs, 0.01 lot, ₹40 per position all-in (your #54, #115). Clock: MT5 server time.

ANSWER: CHANCE — failed: luck test, random entries, fake levels, without the best 3

Came to the level after the setup: 100 of 100 times (1099 of 1099 setups that had their full chance)
Then: hit the target before the stop 17 of 100 (183 of 1099 closed trades; still open at the end, not counted: 0)
Fake levels (the level moved up and down, the same distances to the target and the stop): reached the target first 9 of 100 (721 of 8024 trades on 8 fake levels)
Money after costs, 0.01 lot: -₹13,228 a year (2016–2020, the best of 2 versions there) · -₹14,745 a year (2021–2023, hidden while choosing)
  With the broker's swap: -₹14,914 a year · nights held 138 · with the real spread instead of ₹40: -₹6,955 a year (2021–2023)
Worst year: -₹18,316 (2023, 385 trades) · of 2016–2020: -₹16,448 (2019) · without the best 3 trades: -₹42 a trade
Random entries: it beats 128 of 200 (random times, the same distances and cost) · setups skipped (one trade at a time): 36
Versions tried: 2 of 20 · slices chosen before: 0 of 8 · looked at, not tested: 0
Smallest edge this test could see: ₹10 a trade (caught at least half the time with 1099 trades)
Already seen by the old project: no. Nearest old tries (not the same idea): Type 1: liquidity grab → break → gap (5-minute) (29 tries, 2016–2023); A minimum gap size (5 tries, 2016–2023); Where the leg left open after a hedge closes should aim (#111, #112, #124) (11 tries, 2016–2023). In all, 27 old tests (461 tries) used only parts of this idea
Hidden years (2021–2023) opened in the whole project: 1 time
Works best in / fails in: not measured yet (the learner has not run: python -m machine.learner idea-950)
What a model learned (LightGBM, a program that learns from tables): not measured yet (python -m machine.learner idea-950 --model)
Any entry or only this idea: not measured yet (python -m machine.learner idea-950 --pooled)
Project so far: 1 idea judged, 0 "works", about 0.05 "works" expected by luck alone

The 6 checks (the rules were fixed before any result was seen):
1. luck test: NO — -₹40 a trade; a world with no edge gives this much 100.0 times in 100 (needs under 5, and above ₹0)
2. random entries: NO — it beats 64 of 100 random runs (needs 95); random runs: middle -₹42 a trade, 1090 trades (this idea 1099)
3. fake levels: NO — it beats 5 of its 8 fake levels (needs 75 in 100)
4. without the best 3: NO — -₹42 a trade without its best 3 trades (needs above ₹0)
5. trades: yes — 1099 trades (needs 30)
6. not already seen: yes — not in the old project's list

Every trade, every year and the full rules: plan/work/ideas/idea-950/report.html

the report page and its picture: demo-output/

done in 1225 s · every file it wrote is in a temporary folder (idea-machine-demo-vagafuiu) - nothing in plan/work/
```
