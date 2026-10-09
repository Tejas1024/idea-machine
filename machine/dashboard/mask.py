"""What free text may reach the dashboard (L16.2). Hard rule 1: the page never shows a 2024+ price.

Two masks, chosen by where a text comes from:

    every_digit(text)   every digit -> "#", except clock times ("23:15"). For any text the live alerts (L17a) can
                        write: data/host/alerts.log, the nightly job's error. An alert can carry a live price
                        ("4,123.45"), a move since a release, or a news number of this week. A clock time is none of
                        them (times=False hides those too).
    prices(text)        every number shaped like a gold price -> "####": 4 or more digits, with or without a thousands
                        comma or decimals ("4,123.45", "4123.5", "2,650", "4123"). Gold has been above $1,000 since
                        2009, so every gold price has 4 digits before the point. Kept: rupees (₹ right before the
                        number), short numbers (counts, minutes, 3-digit ids; "932.10" is no gold price), digits
                        inside a word or an id ("v-a0708fb6bd"), a whole number counting something ("1,440 one-minute
                        rows", "34.3 MB") and - with years=True - a plain year 2010-2029.
                        For texts only the machine writes, about 2016-2023 (the intake, the ideas' states, the daily
                        run's lines, Windows' reasons): none of them can hold a 2024+ price (machine.guards), so the
                        mask is a second lock. His own words take years=False: there a plain "2050" may be a price.
"""
from __future__ import annotations

import re

# a number of 4+ digits (or comma groups), maybe with decimals; not right after a word character, a point, a comma,
# "₹" / "₹ ", or "<letter>-" (an id like v-8331245570); not followed by a word character
NUMBER = re.compile(r"(?<![\w.,₹])(?<!₹ )(?<![A-Za-z]-)(?:\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?(?!\w)")
YEAR = re.compile(r"20[12]\d")
# what a whole number may count, right after it ("1,440 one-minute rows"); sizes may have decimals ("34.3 MB")
COUNTED = re.compile(r" (?:one-minute |five-minute |price |trial |new |old )?(?:rows?|files?|trades?|days?|minutes?|"
                     r"hours?|signals?|setups?|tests?|passed|runs?|versions?|charts?|labels?|positions?|bars?|beats?|"
                     r"messages?|lines?|times|ideas?|tables?|pictures?|answers?)\b")
SIZE = re.compile(r" (?:[kMG]B)\b")
CLOCK = re.compile(r"(?<![\d.:,])\d{1,2}:\d{2}(?::\d{2})?(?!\d|:\d|[.,]\d)")
MASK = "####"


def every_digit(text, *, times: bool = True) -> str:
    text = str(text)
    if not times:
        return re.sub(r"\d", "#", text)
    out, last = [], 0
    for m in CLOCK.finditer(text):
        out += [re.sub(r"\d", "#", text[last:m.start()]), m.group(0)]
        last = m.end()
    return "".join(out) + re.sub(r"\d", "#", text[last:])


def prices(text, *, years: bool = True) -> str:
    def one(m: re.Match) -> str:
        s, rest = m.group(0), m.string[m.end():m.end() + 30]
        if years and YEAR.fullmatch(s):
            return s
        if ("." not in s and COUNTED.match(rest)) or SIZE.match(rest):
            return s
        return MASK
    return NUMBER.sub(one, str(text))
