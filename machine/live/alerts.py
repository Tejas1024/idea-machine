"""The live news alerts' words (L17a): one 30 minutes before a release of his news, one when its number is out.

Each says when (India time), what (his colour), his news rule (#110: a trade in profit closes; at break-even or a loss
it is hedged at the price then - news never closes a losing trade), and 'the last 10 times' from
machine/news/similar.py - 2016-2023 only (T14), a lookup, never a prediction. Before the number: the 10 latest
releases of the same report, and where gold and his chart are now (live prices, T14 use 1): the price, the day so
far, his hour / day / week bias, his two nearest levels. After it: gold since the number, and the 10 most like it -
the same kind of surprise (L12b.1's groups), then the nearest by surprise size and by the situation at the release
minute (machine/live/situation.py: his session, day and week bias, nearest levels, ADR, the day so far, the 2 hours
before) - next to every release with that kind of surprise. Without live prices the alerts are step 1's.
Short lines, easy English (LEARNED #113); the numbers come from similar.counts(), never from parsing its text.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from machine.news import similar

RULE = ("Your news rule (#110): a trade in profit → close it. At break-even or a loss → hedge it at the price then. "
        "News never closes a losing trade.")
FEW = "10 is few: a hint, not a promise."
GROUP_WORDS = {"big good for the dollar": "a big surprise, good for the dollar",
               "big bad for the dollar": "a big surprise, bad for the dollar",
               "small": "a small surprise (close to the forecast)"}


def _ist(t: Any) -> str:
    return pd.Timestamp(t).tz_convert("Asia/Kolkata").strftime("%H:%M")


def _word(c: dict, minutes: int) -> str:
    """The way most of them went: "down" or "up" ("up" on a tie)."""
    return "down" if c[f"move_{minutes}_down"] > c[f"move_{minutes}_up"] else "up"


def _way(c: dict, minutes: int) -> str:
    up, down, n = c[f"move_{minutes}_up"], c[f"move_{minutes}_down"], c["n"]
    if up == down:
        return f"up {up}, down {down}"
    return f"{_word(c, minutes)} {max(up, down)} of {n}"


def _pool_way(c: dict, minutes: int, word: str) -> str:
    """The same way (`word`) counted on a bigger pool: 'in 100' when it holds 20 or more, else 'of n'."""
    n = c["n"]
    if n == 0:
        return ""
    x = c[f"move_{minutes}_{word}"]
    return f"{word} {round(100 * x / n)} in 100" if n >= 20 else f"{word} {x} of {n}"


def _money(c: dict) -> str:
    if not (isinstance(c["range_usd"], float) and math.isfinite(c["range_usd"])):
        return ""
    return f"${c['range_usd']:.2f} from top to bottom (₹{c['range_inr']:,.0f} on 0.01 lot)"


def _sweep(c: dict) -> str:
    """The biggest of the three cases: his nearest level above taken first, the one below, or neither in the hour."""
    case = max(("neither", c["neither"]), ("above", c["above"]), ("below", c["below"]), key=lambda kv: kv[1])
    if case[0] == "neither":
        return f"in {c['neither']} of {c['n']}, gold took neither of your nearest levels in the first hour"
    return f"in {case[1]} of {c['n']}, gold took your nearest level {case[0].upper()} first"


def lookup_lines(res: similar.Result, pool_words: str) -> list[str]:
    """The bullets of 'the last 10 times' (empty when there is nothing to compare with). pool_words names the bigger
    pool each count is shown next to, e.g. "NFP days" or "times like this"."""
    if res is None or not len(res.near):
        return []
    c = similar.counts(res.near)
    pool = res.base if not res.query.has_number else res.base[res.base["group"] == res.query.group]
    p = similar.counts(pool)

    def beside(minutes: int) -> str:
        return f" (on all {p['n']} {pool_words}: {_pool_way(p, minutes, _word(c, minutes))})" if p["n"] else ""

    lines = [f"• first 5 min: gold {_way(c, 5)}{beside(5)}", f"• after 1 hour: gold {_way(c, 60)}{beside(60)}"]
    money = _money(c)
    if money:
        lines.append(f"• the first hour: {money}")
    lines.append(f"• {_sweep(c)}")
    if c["calm_n"] and isinstance(c["calm"], float) and math.isfinite(c["calm"]):
        lines.append(f"• calm again after about {c['calm']:.0f} min")
    return lines


def _numbers(rel: dict) -> str:
    h = rel["head"]
    bits = []
    if h.get("forecast"):
        bits.append(f"forecast {h['forecast']}")
    if h.get("previous"):
        bits.append(f"last time {h['previous']}")
    return " · ".join(bits)


BIAS_WORDS = {1.0: "up", -1.0: "down", 0.0: "neutral"}


def now_lines(s: dict) -> list[str]:
    """Where gold is now, from situation.whole() taken now: the price and the day so far; his bias on the hour, day
    and week charts; his two nearest levels by name and distance. Empty when the market is shut or no live price
    came; his bias and levels only when his charts were read."""
    if not s.get("p0"):
        return []
    first = f"Gold now: {s['p0']:,.2f}."
    if "adr_used" in s:
        first += (f" Today it has moved ${s['adr_used'] * s['adr']:,.2f} from its low to its high "
                  f"({round(100 * s['adr_used'])} in 100 of your 10-day ADR, ${s['adr']:,.2f}).")
    out = [first]
    if all(k in s for k in ("bias_h", "bias_d", "bias_w")):
        w = {k: BIAS_WORDS.get(float(s[k]), "not known") for k in ("bias_h", "bias_d", "bias_w")}
        out.append(f"Your bias now: hour {w['bias_h']} · day {w['bias_d']} · week {w['bias_w']}.")
    if "up_level_kind" in s:
        names, parts = s.get("level_names") or {}, []
        for side, word in (("up", "above"), ("dn", "below")):
            kind = s.get(f"{side}_level_kind")
            if isinstance(kind, str):
                parts.append(f"{word}, {names.get(side, kind.replace('_', ' '))} (${s[f'{side}_level_dist']:,.2f} away)")
            else:
                parts.append(f"{word}, no level of yours left")
        out.append("Your nearest levels: " + " · ".join(parts) + ".")
    return out


def since_line(rel: dict, since: dict | None) -> str:
    """'Gold since the number: UP $3.10 in 2 minutes (2,650.20 → 2,653.30).' - the last price before the release minute
    to the newest closed bar; empty without live prices."""
    if not since:
        return ""
    way = "UP" if since["move"] > 0 else "DOWN" if since["move"] < 0 else "not moved,"
    mins = int(since["minutes"])
    return (f"Gold since the number: {way} ${abs(since['move']):,.2f} in {mins} minute{'s' if mins != 1 else ''} "
            f"({since['p0']:,.2f} → {since['last']:,.2f}).")


def before_text(rel: dict, res: similar.Result | None, now: dict | None = None) -> str:
    label = similar.name_of(rel["report"])[1]
    head = f"⏰ In 30 min, {_ist(rel['utc'])} India time: {label} ({rel['colour']} news)."
    nums = _numbers(rel)
    out = [head] + ([nums] if nums else []) + now_lines(now or {}) + [RULE, ""]
    lines = lookup_lines(res, f"{label} days")
    if lines:
        out += [f"The last 10 {label} days (2016–2023):"] + lines + [FEW]
    else:
        out.append(f"No {label} release in 2016–2023 to compare with.")
    return "\n".join(out)


MATCHED = "Picked by: {what}."


def _matched(res: similar.Result) -> str:
    """What the 10 were matched on, in his words; empty when only the surprise (or nothing) was used."""
    s = res.query.situation
    parts = [w for k, w in (("session", "your session"), ("bias_d", "your day and week bias"),
                            ("up_level_kind", "your nearest levels"), ("adr", "your ADR"),
                            ("adr_used", "gold's move today"), ("before_120_adr", "its move in the 2 hours before"))
             if k in s]
    if not parts:
        return ""
    first = ["the surprise (kind and size)"] if res.query.has_number else []
    what = first + parts
    return MATCHED.format(what=", ".join(what[:-1]) + " and " + what[-1] if len(what) > 1 else what[0])


def after_text(rel: dict, score: dict, res: similar.Result | None, since: dict | None = None) -> str:
    label = similar.name_of(rel["report"])[1]
    h = rel["head"]
    if h.get("actual"):
        head = f"📰 {label} is out ({_ist(rel['utc'])} India time): {h['actual']}" + \
               (f" (forecast {h['forecast']})." if h.get("forecast") else ".")
    else:
        head = f"📰 {label} is out ({_ist(rel['utc'])} India time)."
    out = [head]
    if score.get("group") in GROUP_WORDS:
        out.append(f"That is {GROUP_WORDS[score['group']]}.")
    elif score.get("why"):
        out.append(f"No surprise size: {score['why']}.")
    gold = since_line(rel, since)
    if gold:
        out.append(gold)
    out.append("")
    matched = _matched(res) if res is not None else ""
    if res is not None and res.query.has_number:
        title = "The 10 times most like this (2016–2023):" if matched else "The last 10 times like this (2016–2023):"
        lines = lookup_lines(res, "times like this")
    else:
        title = f"The 10 {label} days most like today (2016–2023):" if matched else \
            f"The last 10 {label} days (2016–2023):"
        lines = lookup_lines(res, f"{label} days")
    if lines:
        out += [title] + ([matched] if matched else []) + lines + [FEW]
    else:
        out.append(f"No {label} release in 2016–2023 to compare with.")
    return "\n".join(out)


def question(rel: dict, score: dict | None = None, situation: dict | None = None) -> similar.Query:
    """The lookup's question for a live release. Before the number: no situation (it would be measured 30 minutes before
    the release, every past one AT its minute - not the same), so the 10 latest releases of the report. After it: the
    surprise when it was scored, and the situation at the release minute when live prices gave it
    (machine/live/situation.for_lookup: session, day range, the day so far, the 2 hours before, and his day and week
    bias and nearest levels when his charts were read)."""
    t = pd.Timestamp(rel["server"])
    sit = dict(situation or {})
    if score and score.get("group") in GROUP_WORDS:
        h = rel["head"]
        return similar.Query(rel["report"], t, "after", z=float(score["z"]), group=score["group"], situation=sit,
                             number=f"{h['actual']} (forecast {h['forecast']})")
    return similar.Query(rel["report"], t, "before", situation=sit, note=(score or {}).get("why", ""))
