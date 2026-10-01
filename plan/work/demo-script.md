# The 2-minute demo — script and recording (L11)

> For Tejas (to record it in his own voice, if he wants) and for anyone who wants to make the recording again.
> Everything shown is made-up data: made-up notes, a made-up idea, made-up prices with no edge. Nothing private.

## The recording that already exists

`docs/demo.gif` (in the public repo; made 1 Oct 2026). It is the real output of `python -m machine.demo`, drawn as a
terminal at reading pace, about 2 minutes. Make it again with one command:

```
.venv\Scripts\python -m machine.demo.record docs/demo.gif
```

The run itself takes about 100 seconds. The long wait in step 6 (the referee's exam, about 70 seconds) is shortened
in the picture, and the picture says so. The clock in the top corner is the picture's own time.

## The script — 8 steps, 2 minutes

Run in a terminal: `python -m machine.demo`. Say the line under each step while it shows.

| Time | On screen | Say (plain words) |
|---|---|---|
| 0:00 | the title line | "This is the idea machine. A trader says an idea in plain words. An AI turns it into a rule. Code tests it and judges it. Everything here is made-up data." |
| 0:10 | **1. The idea** — "sell when the high is taken, then the break soon after and the first gap" | "This is how a trader really talks. Short, and some parts are left open." |
| 0:20 | **2. Search the notes first** — FOUND / NOT FOUND lines | "Before anything is asked, every term is searched in his notes. If the notes already say it, he is never asked again." |
| 0:32 | **3. The AI writes a form** — the 3 steps, the entry, the exit; "code's checks: FORM DRAFTED"; the two unclear parts; a far-edge form REFUSED | "Claude does not write code. It fills one strict form. Code checks every part. Look — a form that changes one of his fixed rules is refused, and the refusal says why." |
| 0:50 | **4. Ask only where readings disagree** — reading A 96 setups, reading B 42 | "Two readings of 'the high'. They make different trades, so this one is worth asking. If they made the same trades, he would never be asked." |
| 1:02 | **5. Test every version** — two versions, rupees a trade after cost | "'Soon' is a number the machine tries by itself: two versions. Each one is a try, and every try is written down." |
| 1:15 | **6. The referee** — the rules, then its exam: no edge "works" 0 of 100; a real ₹100 edge caught 198 of 200 | "The rules for 'works' were fixed before any result. And the referee is tested too: on noise it almost never says 'works'; on a real edge it almost always does." |
| 1:40 | **7. The guards** — 2024 REFUSED; a peeking level CAUGHT; 2021 REFUSED | "These are guards in code, not promises. The newest years are sealed. A rule that looks at the future is caught. The check years open once." |
| 1:52 | **8. The report** — and "done" | "The full run gives the answer in rupees, year by year. On made-up prices with no edge, the honest answer is 'chance'." |

## To record it with your voice (optional, about 10 minutes)

1. Open a terminal, make the font big (Ctrl and +), and make the window wide.
2. Windows 11: press **Win + Shift + R** (Snipping Tool's screen recorder), choose the terminal window, and press
   Start. Your microphone can be turned on in the same bar.
3. Type `.venv\Scripts\python -m machine.demo` and press Enter. Read the lines above as each step shows.
4. Press Stop. Trim the start and the end in Clipchamp (it is on this PC). Save as MP4.
5. Upload the MP4 to the public repo's README (drag it into the README editor on GitHub), or to YouTube as
   "unlisted", and put the link in the README.

## What not to show

- Never the private repo, `strategy/`, the Telegram chat, or real money results with the prices behind them.
- No API key on screen (the demo needs none).
