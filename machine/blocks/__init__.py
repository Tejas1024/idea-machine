"""The building blocks of his ideas (L6.1, L6.2, L6b). Every row is stamped `known_at` - the moment it became known."""
# L6.1: swings · structure (his order block and the break of structure) · fvg (the gap, his near edge, the first visit,
# the gaps of the breaking move) · fib (Fibonacci, the plain reading) · sessions ("so far" and "final") · adr (his one
# indicator). L6.2: equal_levels (equal highs / lows, a pair, really equal) · liquidity (his obvious liquidity, swing
# roles, the sweep and "a level swept once is finished", recent first). L6b.3: bias (his entry run on the hour / day /
# week candles, the real all-time high, the doji) · premium_discount (his ADR zones) · hammer (#115) · halfway (both
# readings, four-boxes B2-05). L6b.1: trendlines (3 touches that respect the line, falling / rising / flat; 2-touch
# lines for "a line in the path"; his touch entry, Type 2, whose "steep" waits for his answer, B2-01) · patterns
# (triangles known when their lines have narrowed, their false breaks, false breaks of his liquidity, wedges; "big
# move" waits for his answer, B2-02). L6b.2: zones (his strong zones - time at price on 5-minute and hour candles, the
# quiet main zone and his second choice, touches and roles, the real break, liquidity inside it, the zone entry, the
# zone before the target) · fvg.unfilled_past (#87). Definitions come only from machine/rules.py
# and machine/settings.py (likhith.md R11). Each block passes the cut-off-the-future test (tests/test_blocks.py,
# tests/test_liquidity.py, tests/test_trendlines.py, tests/test_zones.py) and is compared with the old engine
# (plan/work/L6-differences.md,
# L6b3-differences.md, L6b1-differences.md, L6b2-differences.md).
