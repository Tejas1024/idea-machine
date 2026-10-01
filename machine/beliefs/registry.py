"""Every Box 4 belief (plan/work/four-boxes.md) and where its numbers are made (L13b).

One module per belief, in Box 4's order. A row that says two things is two beliefs (B4-04, B4-08, B4-11, B4-13,
B4-14: "a" and "b"). Two rows are answered elsewhere: B4-07 (the news trap) by L12b.2, B4-23 ("the account doubles in
7-8 days") by L20.
"""
from __future__ import annotations

from machine.beliefs import (adr_break_further, adr_green_bounce, confluence, equal_exact, equal_first, equal_magnet,
                             equal_pair, equal_three_touches, false_break_other_way, false_break_returns, gap_magnet,
                             grab_then_move, hammer_gap, hedging_sideways, hour_zones, new_york_biggest, news_technical,
                             open_gap_magnet, recent_first, sideways_most, tight_gaps, trendline_retest,
                             zone_first_touch, zone_grab_gap, zone_turn, zone_two_quiet)
from machine.beliefs.engine import Belief

MODULES = (
    grab_then_move,         # B4-01
    equal_first,            # B4-02
    equal_magnet,           # B4-03
    equal_exact,            # B4-04a
    equal_three_touches,    # B4-04b
    equal_pair,             # B4-05
    recent_first,           # B4-06
    gap_magnet,             # B4-08a
    open_gap_magnet,        # B4-08b
    hammer_gap,             # B4-09
    tight_gaps,             # B4-10
    zone_turn,              # B4-11a
    zone_grab_gap,          # B4-11b
    hour_zones,             # B4-12
    zone_first_touch,       # B4-13a
    zone_two_quiet,         # B4-13b
    false_break_other_way,  # B4-14a
    false_break_returns,    # B4-14b
    trendline_retest,       # B4-15
    new_york_biggest,       # B4-16
    adr_green_bounce,       # B4-17
    adr_break_further,      # B4-18
    confluence,             # B4-19
    news_technical,         # B4-20
    hedging_sideways,       # B4-21
    sideways_most,          # B4-22
)

ELSEWHERE = {
    "B4-07": "the news trap - L12b.2, machine/news/beliefs.py (belief-news-trap): plan/work/beliefs/news.md section 3",
    "B4-23": "'the account doubles in about 7-8 days' - L20 (the money plan)",
}


def families(only=None) -> dict[str, list[Belief]]:
    """{belief: its versions}, in Box 4's order. only: belief names ("belief-equal-magnet" or "equal-magnet")."""
    out = {m.FAMILY: list(m.VERSIONS) for m in MODULES}
    if only:
        want = {o if o.startswith("belief-") else f"belief-{o}" for o in only}
        unknown = sorted(want - set(out))
        if unknown:
            raise KeyError(f"no belief {unknown}; the beliefs are {', '.join(f.removeprefix('belief-') for f in out)}")
        out = {k: v for k, v in out.items() if k in want}
    return out


def get(idea_id: str, version_id: str) -> Belief:
    for versions in families().values():
        for b in versions:
            if (b.idea_id, b.version_id) == (idea_id, version_id):
                return b
    raise KeyError(f"no belief {idea_id} version {version_id}")
