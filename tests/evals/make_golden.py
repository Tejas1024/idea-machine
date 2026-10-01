"""Builds tests/evals/golden.jsonl from plan/work/golden-ideas.md (T4) + the expected answers below.

The words are cut from golden-ideas.md by code (word for word). The expected answers are golden-ideas.md's own answers,
written in the form's words (block names, entry / hedge / target). Only what golden-ideas.md STATES is scored.
Written before the talker ever ran.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
md = (ROOT / "plan/work/golden-ideas.md").read_text(encoding="utf-8")

words, source, real = {}, {}, {}
for block in re.split(r"\n(?=### G\d\d)", md):
    m = re.match(r"### (G\d\d) · (real|style)(?: · ([^\n]*))?", block)
    if not m:
        continue
    gid = m.group(1)
    real[gid] = m.group(2) == "real"
    source[gid] = (m.group(3) or "").split(" — ")[0].strip() or "in his style (Claude)"
    quote = [ln[1:].strip() for ln in block.splitlines() if ln.startswith(">")]
    text = " ".join(quote)
    assert text.startswith('"') and text.endswith('"'), (gid, text)
    words[gid] = text[1:-1]

HEDGE = "hedge break_of_structure.extreme"       # his default hedge: the grab's extreme (#9, #33, #111)
GAP = "limit gap_of_the_move.near"                # his default entry: the near edge of the first gap (#61, #62, #65)
NEXT = "next_liquidity*"                          # his default target: the next liquidity (#119)
BIAS = "bias|higher_timeframe"
TYPE1 = ["sweep", "break_of_structure", "gap_of_the_move"]


def ask(*kw):
    return {"kind": "ask", "any": list(kw)}


def version(*kw):
    return {"kind": "version", "any": list(kw)}


E = {
    "G01": dict(kind="idea", timeframe="1h", directions=["buy", "sell"], needs=["fib_move"], blocked=False,
                entry="limit fib_move.level_*", target="price fib_move.level_0.5",
                voice=[{"heard": "1 is to 1", "any": ["1:1"]}],
                unclear=[ask("0.68", "0.618", "golden"), ask("stop", "hedge", "1:1", "1 is to 1", "risk"),
                         ask("break", "higher high", "one side", "one-sided", "which move")],
                must_not_ask=["what 'break of structure' means (#56-#60)"], already_seen=["fib_0618"]),
    "G02": dict(timeframe="5m", directions=["buy"], needs=TYPE1 + [BIAS], blocked=True, entry=GAP, exit=HEDGE,
                target=NEXT, voice=[{"heard": "1et fvg", "any": ["first", "1st"]}], unclear=[],
                must_not_ask=["how 'up' is read (#70-#74)"]),
    "G03": dict(timeframe="5m", directions=["buy", "sell"], needs=TYPE1, blocked=False, entry=GAP, exit=HEDGE,
                target=NEXT, unclear=[], must_not_ask=["buy or sell"]),
    "G04": dict(kind="rule:skip", needs=["target_already_reached"], blocked=False,
                voice=[{"heard": "S", "any": ["skip"]}], unclear=[], must_not_ask=["what 'S' means"]),
    "G05": dict(kind="rule:entry", timeframe="5m", directions=["buy", "sell"], needs=["gap_of_the_move"],
                blocked=False, entry=GAP, exit=HEDGE, target=NEXT, unclear=[]),
    "G06": dict(timeframe="5m", directions=["buy"], needs=["trendline", "sweep", "gap_of_the_move", BIAS],
                blocked=True, entry=GAP, exit=HEDGE, target=NEXT, unclear=[],
                must_not_ask=["what counts as a trendline (#92, #94)"]),
    "G07": dict(timeframe="5m", directions=["buy"], needs=TYPE1, blocked=False, entry=GAP, exit=HEDGE, target=NEXT,
                unclear=[], must_not_ask=["which gap (#61, #62)", "'discount' (AMBIGUITY A4, settled)"]),
    "G08": dict(kind="rule:exit", timeframe="5m", needs=["trendline"], blocked=True,
                target=["price trendline*", "next_liquidity*trendline*"],
                voice=[{"heard": "favorable gap", "any": ["fair value gap", "fvg"]}], unclear=[],
                must_not_ask=["how many touches (#92)"]),
    "G09": dict(timeframe="5m", directions=["buy"], needs=["trendline", BIAS], blocked=True,
                entry="limit trendline*", exit="hedge trendline*", target=NEXT,
                unclear=[ask("steep"), version("pip", "below the line", "hedge distance", "how far", "trail")],
                must_not_ask=["what a trendline is (#92, #94)"]),
    "G10": dict(timeframe="5m", directions=["buy"], needs=["zone|zone_entry", "sweep", BIAS], blocked=True,
                entry="limit zone*", exit="hedge zone*", target=NEXT,
                voice=[{"heard": "ob obvious", "any": ["obvious"]}, {"heard": "heads point", "any": ["hedge point"]},
                       {"heard": "favorable gap", "any": ["fair value gap", "fvg"]}],
                unclear=[], must_not_ask=["what a strong zone is (#79-#81)", "what obvious liquidity is (#84, #86)"]),
    "G11": dict(kind="rule:filter", timeframe="5m", directions=["sell"], needs=["premium_discount|adr", BIAS],
                blocked=True, unclear=[], must_not_ask=["what discount is (strategy.md section 7)"]),
    "G12": dict(kind="rule:exit", needs=[BIAS], blocked=True, target="next_liquidity*swing*",
                voice=[{"heard": "trial", "any": ["trail"]}], unclear=[]),
    "G13": dict(timeframe="5m", needs=["zone|zone_break", BIAS], blocked=True, entry="limit zone*",
                exit="hedge zone*", target=NEXT, unclear=[], must_not_ask=["how far the break must go (#98)"]),
    "G14": dict(timeframe="5m", directions=["buy"], needs=["zone|zone_break", "sweep", "gap_of_the_move"],
                blocked=True, entry=GAP, exit=["hedge sweep.*", HEDGE], target=NEXT,
                voice=[{"heard": "five hourly gap", "any": ["5-minute", "five-minute", "5 min", "five minute",
                                                             "5m", "5-min"]}], unclear=[]),
    "G15": dict(timeframe="5m", directions=["buy"], needs=["gap_of_the_move", BIAS], blocked=True, entry=GAP,
                exit=HEDGE, target=NEXT, unclear=[ask("very strong", "strong", "confidence")],
                must_not_ask=["which gap is 'first' (#61, #62)"]),
    "G16": dict(kind="idea", directions=["buy", "sell"], needs=["sweep", "level_waiting|equal_levels"],
                blocked=False,
                target=["price level_waiting.price", "price equal_levels.price", "next_liquidity*equal*"],
                unclear=[version("eventually", "how long", "wait", "timing", "time")],
                must_not_ask=["what equal means (#88-#90)"], already_seen=[]),
    "G17": dict(timeframe="5m", directions=["sell"], needs=TYPE1, entry=GAP, exit=HEDGE,
                target=["price level_waiting.price", "next_liquidity*session*"], unclear=[],
                must_not_ask=["the London / Asia hours (strategy.md section 1)"]),
    "G18": dict(timeframe="5m", directions=["buy"], needs=["all_time_high"] + TYPE1, blocked=True, entry=GAP,
                target=NEXT, unclear=[], must_not_ask=["which all-time high (#72: the real one)"]),
    "G19": dict(timeframe="5m", directions=["sell"], needs=["news"] + TYPE1, blocked=True, entry=GAP, exit=HEDGE,
                target=NEXT, unclear=[version("soon", "after the news", "news window", "how long", "minutes", "news")],
                must_not_ask=["what 'big news' is (red + orange, news.md)"], already_seen=["news_trap"]),
    "G20": dict(timeframe="5m", needs=["gap_of_the_move"], entry=GAP, target=NEXT,
                unclear=[ask("strong trend", "strong"), version("near", "hedge distance", "how far", "how close")]),
    "G21": dict(timeframe="5m", directions=["sell"], needs=["triangle", "gap_of_the_move"], blocked=True, entry=GAP,
                exit=HEDGE, target=["price triangle*", "price false_breakout*"], unclear=[ask("big move", "big")],
                must_not_ask=["what a triangle is (#116, #118)"]),
    "G22": dict(timeframe="5m", directions=["buy"], needs=["trendline"], blocked=True, entry="limit trendline*",
                exit="hedge trendline*", target=NEXT,
                unclear=[ask("steep"), version("pip", "below the line", "how far", "distance")],
                must_not_ask=["'strong uptrend' (#116, #127)"]),
    "G23": dict(timeframe="1h", needs=["halfway"] + TYPE1, blocked=True, unclear=[ask("halfway", "half")],
                already_seen=["halfway_rule"]),
    "G24": dict(kind="rule:management",
                unclear=[version("evening", "hours", "before the close", "friday", "weekend", "when to close")]),
    "G25": dict(timeframe="5m", directions=["buy"], needs=["premium_discount|adr", "news"] + TYPE1, blocked=True,
                entry=GAP, exit=HEDGE, target=NEXT,
                unclear=[version("green", "adr", "discount", "how near"), version("news")],
                must_not_ask=["what the green zone is (strategy.md section 7)"]),
}
assert sorted(E) == sorted(words) and len(words) == 25, sorted(set(words) ^ set(E))

out = []
for gid in sorted(words):
    row = {"id": gid, "real": real[gid], "source": source[gid], "words": words[gid], "expect": E[gid]}
    out.append(json.dumps(row, ensure_ascii=False))
(ROOT / "tests/evals/golden.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
n_ask = sum(u["kind"] == "ask" for e in E.values() for u in e.get("unclear", []))
n_ver = sum(u["kind"] == "version" for e in E.values() for u in e.get("unclear", []))
print(len(out), "rows;", sum(real.values()), "real;", n_ask, "ask;", n_ver, "version")
