# Sample trader notes — MADE UP for the public demo

> These notes are invented for `python -m machine.demo`. They use common textbook words (liquidity, break of
> structure, fair value gap). They are NOT the trader's private notes, which stay in the private repo.

## 1. Liquidity

- A session high or low (Asia, London, New York) is liquidity: stop orders rest just beyond it.
- Liquidity is taken when price trades beyond the level; a wick beyond it is enough.
- "The high is taken" = price went above that high, even for one candle.
- A level that was already taken is finished: it is not liquidity any more.

## 2. The break

- After liquidity is taken, wait for a break of structure the other way: a candle closes beyond the last swing.
- No break, no trade: the setup is dropped.

## 3. The gap

- A gap (fair value gap) is three candles where the first and the third do not overlap.
- Use the first gap of the breaking move. Enter with a limit order at the near edge of the gap.

## 4. Exit

- The stop goes beyond the extreme of the breaking move.
- The target is the next liquidity in the trade's direction.

## 5. Time

- Trade on the 5-minute chart.
