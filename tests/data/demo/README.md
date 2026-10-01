# tests/data/demo — the made-up data of `python -m machine.demo`

Everything here is invented for the demo (L11). None of it is the trader's private notes, his ideas or real prices.

| File | What it is |
|---|---|
| `notes/strategy/strategy.md` | made-up trader notes in the same format as the private ones (numbered sections, one rule a line). The demo's search (`machine/search`, words only) reads them instead of the private notes. |
| `notes/strategy/LEARNED.md` | three made-up "lessons" in the format of the private lessons file. |
| `idea.json` | the made-up idea's words; the talker's answer **written by hand** in the exact shape of its one Claude tool call (`fill_idea_form`), so the demo needs no key and makes no call (code checks it like a real answer); and `stand_in_answers`, standing in for the trader's reply to the question. |
| `prices.json` | the recipe of the made-up prices: a random walk of 1-minute bars 2016–2023 (`machine/agent/noise.py` `walk_m1`), about 3 million bars made in a second. No minute says anything about the next, so there is **no edge**: the honest answer for any idea is *chance*. |

The demo writes every file it makes (the form, trial files, the check-years record, the report) into a temporary
folder, never into `plan/work/`.
