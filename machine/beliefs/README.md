# machine/beliefs — his Box 4 beliefs as numbers (L13b)

    python -m machine.beliefs rules                 # the rules for an answer (printed in every report)
    python -m machine.beliefs counts --years 2019   # how many events each belief finds - no outcome is printed
    python -m machine.beliefs tune                  # every version on 2016-2020, one chosen per belief (trial files)
    python -m machine.beliefs open --who "..."      # the check years, opened ONCE per belief
    python -m machine.beliefs check                 # 2021-2023 for each chosen version (trial files)

| file | what |
|---|---|
| `engine.py` | a belief, the six rules, the numbers (one comparison pool per event, or two groups), tune / open / check, the trial files and the reports (`plan/work/beliefs/<belief>-tune.md`, `-check.md`) |
| `world.py` | the judged years read once (ending at the last judged minute), the race, the reach, the same weekday and hour |
| `levels.py` | his liquidity alive at a close, the race after a take, fake levels / bands, random moments, the rows |
| `events.py` | the events several beliefs share: takes, false breaks, equal-level births, zone touches (each with its fakes) |
| `trades.py` | his grab + gap setups, each its own trade with a plain stop at his hedge point (the rupee beliefs) |
| `registry.py` | every Box 4 belief in order; B4-07 (L12b.2) and B4-23 (L20) are answered elsewhere |
| one module per belief | its events, versions, his words and source - the plan is `plan/work/beliefs/<belief>.md` |

Tests: `tests/test_beliefs.py`. Notes: `plan/work/L13b-notes.md`. The news beliefs of L12b.2 keep their own judge
(`machine/news/beliefs.py`, the same rules).
