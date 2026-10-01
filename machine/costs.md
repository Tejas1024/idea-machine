# Costs — what one position costs, in rupees (L2)

Code: `machine/costs.py`. Print this table from the code: `python -m machine.costs`.
`tests/test_costs.py` checks that every value below is the same as the code's.

## The table: cost · value · source

| Cost | Value | Source |
|---|---|---|
| Cost of one position (open + close) | ₹40 per 0.01 lot (₹20 in + ₹20 out), all-in: it includes the spread | Tejas, LEARNED #54 (14 Sep 2026), #115 (20 Sep 2026) |
| Swap on his account | none | Tejas, LEARNED #54 |
| Lot sizes | 0.01, 0.02, 0.03 ... only (round_lot) | Tejas, LEARNED #116 |
| Gold in one lot | 100 ounces: 0.01 lot = 1 ounce, so a $1 move = $1 | MT5 symbol_info trade_contract_size, MetaQuotes-Demo, 28 Sep 2026 |
| Rupees per dollar | monthly average of the month the position closed: ₹63.64 (2018-01) to ₹83.30 (2023-11) | RBI reference rate to Jun 2018, FBIL from 10 Jul 2018 (RBI Handbook table); data/usd_inr.csv |
| Broker's swap (extra line only) | buy -12.6 points a lot a night, sell -4.6; Wednesday night counts 3, Saturday and Sunday nights 0; at 0.01 lot: buy -$0.126 a night, sell -$0.046 | MT5 symbol_info on MetaQuotes-Demo (demo), read 2026-09-28; data/swap.csv |
| Real spread (extra line only) | the data's spread at the minute of the fill, INSTEAD of ₹40: a buy pays it when it opens, a sell when it closes; a spread of 0 counts as 14 points | the spread column of data/gold_m1.parquet; config.MIN_SPREAD_POINTS (Claude's pick) |

His words: *"always twenty per entry, twenty per exit. That's it. Per one trade open and close, it will cost forty
rupees ... So whatever the things that you are telling about swap, it won't apply."* (#54).
*"per entry no 0.01 lot entry 20 rupees exit 20 rupees"* (#115). Never "₹1,000 a trade" (#105): money is shown at
0.01 lot.

## The main number

One position = one leg. A hedged trade has several legs; each leg pays its own ₹40.

```
gold dollars   = (exit price − entry price) × ounces      (the other way round for a sell)
ounces         = lots × 100                               (0.01 lot = 1 ounce)
rupees         = gold dollars × rupees per dollar         (the month the position closed)
cost           = ₹40 × (lots ÷ 0.01)                      (0.01 lot → ₹40, 0.02 lot → ₹80)
after cost     = rupees − cost                            ← THE number
```

Worked by hand (a test checks it): buy at 2000, sell at 2010, 0.01 lot, ₹83 a dollar →
gold +$10 = ₹830 → cost ₹40 → **₹790**.

- Prices are **bid** prices, like the price data. The ₹40 **replaces** the spread. Adding the spread on top would
  count it twice (lesson B14 in `plan/likhith.md`). The old `manager.py` also used the ₹40 instead of the spread.
- Lots must be 0.01, 0.02, 0.03 … `pnl()` refuses 0.015. `round_lot()` rounds to the nearest step, half going up
  (0.015 → 0.02). Below half a step it gives 0.0, and the caller decides (Claude's reading of "round off", #116).
- Times are **MT5 server time** with no time zone, like `load_bars`. 2024 and later are refused (sealed years, #96).

## The two extra numbers — same line, clearly labelled, never inside the main number

```
BUY 0.01 lot · gold +$10.00 = ₹833 · cost -₹40 (₹40 per 0.01 lot, all-in) · after cost ₹793 ($1 = ₹83.30)  (with the broker's swap: ₹741 · nights held 3 · the broker counts 5)  (with the real spread: ₹817)
```

**With the broker's swap** = the main number + the swap this broker would charge. His account has no swap (#54),
so this line only shows what a swap account would pay.
- A night = a server midnight passed while the position is open.
- The broker charges Monday to Friday nights once, Wednesday night 3 times (it pays for the weekend), and
  Saturday and Sunday nights nothing.
- Monday 10:00 → Thursday 10:00 = nights held 3, the broker counts 5. Friday → Monday = held 3, counts 1.
- For several legs, each leg counts its own nights. Two legs open 3 nights = **nights held 6** (leg-nights).
- ₹ = swap points × $0.01 × 100 ounces × lots × nights counted × rupees per dollar.

**With the real spread** = the price move minus the spread really paid, **instead of** ₹40 (Claude's reading: ₹40
already includes the spread, so the only honest extra is to swap one for the other).
- A buy opens at the ask (bid + spread), so it pays the spread at the entry minute.
- A sell closes by buying back at the ask, so it pays the spread at the exit minute.
- The tester passes the spread of that minute from the price data (in points; 1 point = $0.01).
- A spread of 0 (1.5% of 1-minute bars) counts as 14 points. The same 14-point rule is the tester's (L3).
- If the spread is not given, the line says "not given".

## Where the numbers come from, and what they are not

**Rupees per dollar** — `data/usd_inr.csv`, 96 months, Jan 2016 – Dec 2023. It is in git.
- The first line lists the source pages. They are the RBI's *Handbook of Statistics on the Indian Economy*, the
  table "Exchange Rate of the Indian Rupee … (Monthly Average and end-Month Rates)", US Dollar monthly average.
- Its note says: RBI reference rates from May 2012; FBIL (Financial Benchmarks India) from 10 July 2018.
- The `source` column of each row says which one it is. July 2018 is both.
- Each edition shows about 4½ years, so 6 editions were read (2019 … 2025-26). Where they overlap, the newest wins.
- 84 months are in two or more editions. 5 differ: four by ₹0.0001 (rounding), and Sep 2020 by ₹0.01 in the 2022
  edition only. Three other editions say 73.4817, and that is used.
- Check: the mean of each year's 12 months equals the RBI's own yearly-average table within ₹0.0001, for all 8 years.
  A wrong column or a shifted month cannot pass this. A test repeats it.
- Which month: the month the position closed (Claude's reading). A position open across a month end uses one rate.
- Remake it (needs the internet): `python -m machine.data.usd_inr_fetch`.

**Broker's swap** — `data/swap.csv`, one row per reading. The newest row is used. It is in git.
- Read by `python -m machine.data.swap_read` from MT5 on Tejas's PC (🪟), 28 Sep 2026. The server is
  **MetaQuotes-Demo, a demo account** — the server the price data came from. **It is not XM** (his broker).
- The same values (−12.6 / −4.6, Wednesday) were read by the old project (`strategy/strategy-understanding.md` §21).
- It is **today's** swap, used for every year 2016–2023. The real swap changed as interest rates changed, so this line
  is a rough size, not history.
- Only swap mode POINTS (and "no swap") is priced. Another mode makes `extra_lines` stop with a clear message.
- No price was read or printed from MT5, and no order was sent.

**Real spread** — the `spread` column of the price data (MetaQuotes-Demo, 1-minute bars; median 19 points = $0.19).
XM's spread may differ. The ₹40 is what Tejas actually paid.
