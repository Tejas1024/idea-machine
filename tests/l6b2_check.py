"""L6b.2 side by side: his zone charts (strategy/pictures/, via their LEARNED entries) next to what the new zone blocks
see on the SAME candles. 2016-2020 only (the tune years); the panels of his cards from 2021-2023 are the hidden years
and are not read (like L6b.1, L6b.3). No money.

    .venv\\Scripts\\python -m tests.l6b2_check        writes plan/work/L6b2-check/*.jpg and README.md

Left = his picture (the panel he answered about, cropped). Right = the machine's chart of the same stretch: its strong
zones there (tier 1 = his main, quiet zone; tier 2 = his second choice; tier 3 = not handed on), and for the zone
entry / the zone break what the blocks and his hour + day filter do.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from machine import config, settings
from machine.blocks import zones as ZN
from machine.charts import draw as D
from machine.charts import lint
from machine.ideas import compile as C
from machine.ideas.form import Idea
from tests.l6b1_check import side_by_side, wrap
from tests.test_zones import ZB, ZE

OUT = config.WORK_DIR / "L6b2-check"
PICS = config.ROOT / "strategy" / "pictures"
TL_, TR_, BL_, BR_ = (0, 85, 650, 660), (650, 85, 1300, 660), (0, 660, 650, 1240), (650, 660, 1300, 1240)
TIER = {1: "his main zone (tier 1)", 2: "his second choice (tier 2)", 3: "tier 3 (not handed on)"}
COL = {1: "#9fd3c7", 2: "#cfd8e3", 3: "#eeeeee"}


@dataclass
class Case:
    key: str
    lesson: str
    picture: str
    crop: tuple | None
    kind: str                   # zone / entry / break
    tf: str
    start: str                  # India time
    end: str                    # India time (his picture's last candle)
    band: tuple[float, float]   # the zone he saw
    direction: str | None
    his: str


CASES = [
    Case("01-Q78-2016-03-17", "LEARNED #79", "Q78-C1b-which-zone.png", None, "zone", "1h", "2016-02-26 05:00",
         "2016-03-17 04:30", (1230.5, 1233.7), None,
         "A (1,231) YES - 'it is a recent one ... the candles themselves form one range'; B (1,252-1,257) is a typical "
         "support and resistance - valid, not what he trades (#79)"),
    Case("02-Q79B-2018-01-04", "LEARNED #80", "Q79-C1c-quiet-bar.png", TR_, "zone", "1h", "2017-12-26 13:30",
         "2018-01-04 05:00", (1293.9, 1294.9), None, "B YES - 'B and d are the my kind of zones' (#80)"),
    Case("03-Q79D-2017-05-09", "LEARNED #80", "Q79-C1c-quiet-bar.png", BR_, "zone", "1h", "2017-05-01 20:30",
         "2017-05-09 04:30", (1254.8, 1257.2), None, "D YES - 'B and d are the my kind of zones' (#80)"),
    Case("04-Q86A-2020-10-15", "LEARNED #86", "Q86-D6-zone-entry.png", TL_, "entry", "15m", "2020-10-14 07:15",
         "2020-10-15 04:30", (1893.9, 1897.8), "buy",
         "A TAKE - 'I would enter at the zone without gap ... if it also supports higher time frame, definitely yes'"),
    Case("05-Q86C-2020-04-21", "LEARNED #86", "Q86-D6-zone-entry.png", BL_, "entry", "15m", "2020-04-20 05:00",
         "2020-04-21 14:00", (1674.9, 1680.4), "buy",
         "C TAKE - 'it needs higher time frame confidence ... relative equal highs there'"),
    Case("06-Q86D-2016-03-31", "LEARNED #86", "Q86-D6-zone-entry.png", BR_, "entry", "15m", "2016-03-30 08:45",
         "2016-03-31 14:15", (1235.7, 1238.2), "sell",
         "D TAKE - 'not because it took the equal highs, but there is a liquidity below, equal lows are there'"),
    Case("07-Q97A-2017-01-04", "LEARNED #97", "Q97-D7-zone-break.png", TL_, "break", "5m", "2017-01-04 04:40",
         "2017-01-04 08:00", (1156.79, 1158.37), "buy",
         "all four: 'we will not directly enter when it breaks ... it should come again for retesting' (#97)"),
    Case("08-Q97B-2018-02-23", "LEARNED #97", "Q97-D7-zone-break.png", TR_, "break", "5m", "2018-02-23 09:05",
         "2018-02-23 12:30", (1328.0, 1331.89), "sell",
         "B - he named it: 'it came back to retrace at 11.35 to [13]28.5 ... there I would enter' (#97)"),
    Case("09-Q98A-2019-04-05", "LEARNED #98", "Q98-D8-break-distance.png", TL_, "break", "5m", "2019-04-05 06:30",
         "2019-04-05 13:00", (1290.4, 1293.4), "sell",
         "A VALID - 'if I want to sell there, there should be liquidity ... and the higher time frame should also "
         "suggest it is bearish' (#98)"),
    Case("10-Q98C-2017-06-09", "LEARNED #98", "Q98-D8-break-distance.png", BL_, "break", "5m", "2017-06-09 20:35",
         "2017-06-09 22:20", (1265.6, 1268.2), "buy",
         "C VALID, but a triangle inside the zone on the 1-minute chart: 'I will wait for the fair value gap in five "
         "minutes only' (#98, #99)"),
]


class Machine:
    def __init__(self):
        self.m = C.Market.load("2016", "2020")
        from machine.data.bars import load_bars
        self.b15 = load_bars("15m", "2016", "2020")
        self.fp = settings.fingerprint()
        m5 = self.m.bars("5m")
        self.ist5 = pd.DatetimeIndex(m5["utc"]).tz_convert("Asia/Kolkata").tz_localize(None)
        self.life = self.m.zone_life(self.fp)
        self.entries = self.m.zone_entries(self.fp)
        self.breaks = self.m.zone_breaks(self.fp)
        his_filter = [dict(name="higher_timeframe", words="the higher time frame must be with me (#84, #86, #97)")]
        self.ze = C.compile_idea(Idea.load({**ZE, "filters": his_filter}), self.m).signals
        self.zb = C.compile_idea(Idea.load({**ZB, "filters": his_filter}), self.m).signals

    def i5(self, ist: str) -> int:
        return int(np.flatnonzero(self.ist5 >= pd.Timestamp(ist))[0])

    def ist(self, t) -> pd.Timestamp:
        return self.ist5[self.m.bars("5m").index.get_loc(pd.Timestamp(t))]

    def bars(self, tf: str) -> pd.DataFrame:
        return self.b15 if tf == "15m" else self.m.bars(tf)

    def server(self, ist: str, tf: str) -> pd.Timestamp:
        b = self.bars(tf)
        idx = pd.DatetimeIndex(b["utc"]).tz_convert("Asia/Kolkata").tz_localize(None) if "utc" in b else None
        return b.index[int(np.flatnonzero(idx >= pd.Timestamp(ist))[0])]

    def near_zones(self, i: int, band, n: int = 3) -> pd.DataFrame:
        a = ZN.alive(self.life, i)
        lo, hi = band
        ov = (np.minimum(a["hi"], hi) - np.maximum(a["lo"], lo)).clip(lower=0)
        a = a.assign(overlap=ov / (np.maximum(a["hi"], hi) - np.minimum(a["lo"], lo)))
        return a[a["overlap"] > 0].sort_values(["overlap", "tier"], ascending=[False, True]).head(n)

    def bias(self, t) -> str:
        w = {1: "up", -1: "down", 0: "neutral"}
        return f"hour {w[self.m.bias_at('1h', self.fp, t)]}, day {w[self.m.bias_at('1d', self.fp, t)]}"


def zone_items(z: pd.DataFrame) -> list:
    """One box per band: zones of two windows over the same prices are drawn once (their names would sit on each other)."""
    out, seen = [], []
    for r in z.sort_values("tier").itertuples():
        if any(min(r.hi, hi) - max(r.lo, lo) > 0.5 * (r.hi - r.lo) for lo, hi in seen):
            continue
        seen.append((r.lo, r.hi))
        out.append(D.Zone(float(r.lo), float(r.hi), f"{TIER[int(r.tier)]}, calm {r.calm:.2f}", color=COL[int(r.tier)]))
    return out


def machine_case(M: Machine, c: Case, path: Path) -> dict:
    tf = c.tf
    bars = M.bars(tf)
    lo_t, hi_t = M.server(c.start, tf), M.server(c.end, tf)
    win = bars.loc[lo_t:hi_t]
    i_end = M.i5(c.end)
    items, said = [D.Candles()], ""
    if c.kind == "zone":
        z = M.near_zones(i_end, c.band, 2 if c.key.startswith("01") else 1)
        if c.key.startswith("01"):
            z = pd.concat([z, M.near_zones(i_end, (1252.0, 1257.0), 2)])
        items += zone_items(z)
        said = "; ".join(f"{r.lo:.2f}-{r.hi:.2f}: {TIER[int(r.tier)]}, calm {r.calm:.2f} ({r.window} window)"
                         for r in z.itertuples()) or "no live zone there"
    elif c.kind == "entry":
        day = pd.Timestamp(c.end).normalize()
        e = M.entries[(M.entries["direction"] == c.direction)]
        e = e[[(pd.Timestamp(c.end) - pd.Timedelta(days=1)) <= M.ist(b) <= pd.Timestamp(c.end) for b in e["bar"]]]
        if len(e):
            r = e.iloc[-1]
            sig = [s for s in M.ze if pd.Timestamp(s.known_at) == pd.Timestamp(r.known_at) and s.direction == c.direction]
            items += [D.Zone(float(r.lo), float(r.hi), TIER[int(r.tier)], color=COL[int(r.tier)]),
                      D.Level(float(r.entry), "entry", start=r.bar), D.Level(float(r.hedge), "hedge", start=r.bar,
                                                                              color=D.RED),
                      D.Span(r.bar, r.bar + pd.Timedelta(minutes=5), f"{r.level_kind.replace('_', ' ')} taken")]
            said = (f"zone entry {c.direction} at {M.ist(r.bar):%d %b %H:%M} India time: {r.level_kind.replace('_', ' ')} "
                    f"{r.level_price:.2f} taken, zone {r.lo:.2f}-{r.hi:.2f} ({TIER[int(r.tier)]}), entry {r.entry:.2f}, "
                    f"hedge {r.hedge:.2f}; {M.bias(r.known_at)} -> "
                    + ("an order (his hour + day filter passes)" if sig else "no order (his hour + day filter: not "
                                                                               "both with the trade)"))
            if len(e) > 1:
                said += f"; {len(e) - 1} earlier take(s) at zones that day"
        else:
            z = M.near_zones(i_end, c.band, 1)
            items += zone_items(z)
            said = "no zone entry that day" + ("; the zone there: " + "; ".join(
                f"{r.lo:.2f}-{r.hi:.2f} {TIER[int(r.tier)]}" for r in z.itertuples()) if len(z) else "")
    else:
        b = M.breaks[(M.breaks["direction"] == c.direction)]
        b = b[[(pd.Timestamp(c.start) - pd.Timedelta(hours=6)) <= M.ist(x) <= pd.Timestamp(c.end) for x in b["bar"]]]
        lo, hi = c.band
        b = b[(b["hi"] >= lo - 1) & (b["lo"] <= hi + 1)]
        if len(b):
            r = b.iloc[0]
            m5 = M.m.bars("5m")
            k = m5.index.get_loc(r.bar)
            h, l_ = m5["high"].to_numpy(), m5["low"].to_numpy()
            seg = (h[k + 1:] >= r.edge) if c.direction == "sell" else (l_[k + 1:] <= r.edge)
            fill = m5.index[k + 1 + int(np.argmax(seg))] if seg.any() else None
            sig = [s for s in M.zb if pd.Timestamp(s.known_at) == pd.Timestamp(r.known_at)]
            items += [D.Zone(float(r.lo), float(r.hi), TIER[int(r.tier)], color=COL[int(r.tier)]),
                      D.Span(r.bar, r.bar + pd.Timedelta(minutes=5), "the real break"),
                      D.Level(float(r.edge), "retest entry", start=r.bar)]
            if fill is not None and fill <= M.server(c.end, "5m") + pd.Timedelta(hours=6):
                items.append(D.Span(fill, fill + pd.Timedelta(minutes=5), "back at the zone"))
            said = (f"real break {c.direction} at {M.ist(r.bar):%H:%M} India time (closed {r.depth:.2f} of the zone's "
                    f"height through {r.lo:.2f}-{r.hi:.2f}, {TIER[int(r.tier)]}); the limit at {r.edge:.2f} "
                    + (f"is reached at {M.ist(fill):%H:%M}" if fill is not None else "is never reached")
                    + f"; {M.bias(r.known_at)} -> "
                    + ("an order (his hour + day filter passes)" if sig else "no order (his hour + day filter: not "
                                                                               "both with the trade)"))
        else:
            z = M.near_zones(i_end, c.band, 1)
            items += zone_items(z)
            said = ("no real break: " + ("price came into the zone from ABOVE and left upward - the zone held "
                                         "(a break goes through, T06 / #98)" if c.key.startswith("07") else "none found")
                    + ("; the zone: " + "; ".join(f"{r.lo:.2f}-{r.hi:.2f} {TIER[int(r.tier)]}, calm {r.calm:.2f}"
                                                  for r in z.itertuples()) if len(z) else ""))
    d = pd.Timestamp(c.end)
    panel = D.Panel(win, tf, items, title=f"the machine · {d:%d %b %Y}", clock="ist")
    boxes = any(isinstance(x, D.Zone) for x in items)
    pic = D.Picture("trade", f"L6b.2 check {c.key}", f"THE MACHINE · {c.key[3:]}", [[panel]], sub=wrap(said, 80),
                    footer_left="green = his main zone (tier 1) · grey = his second choice (tier 2)" if boxes else "",
                    footer_right="L6b.2")
    drawn = D.render(pic, path)
    return dict(said=said, lint=lint.check(drawn))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    M = Machine()
    rows = []
    for c in CASES:
        m = OUT / f"_m-{c.key}.jpg"
        got = machine_case(M, c, m)
        crop = c.crop or (0, 0, *Image.open(PICS / c.picture).size)
        final = OUT / f"{c.key}.jpg"
        side_by_side(PICS / c.picture, crop, m, final,
                     f"LEFT: his picture ({c.picture}, {c.lesson})   RIGHT: the machine, the same candles (L6b.2)",
                     f"HIS ANSWER: {c.his[:150]}")
        m.unlink()
        rows.append((c, got, final))
    lines = ["# L6b.2 — his zone charts, side by side with the machine", "",
             "> Made by `python -m tests.l6b2_check`. LEFT = his picture (the panel he answered about); RIGHT = what "
             "the zone blocks see on the same candles, India time. 2016–2020 only: the other panels of these cards "
             "(Q76 B–C, Q79 A and C, Q86 B, Q97 C–D, Q98 B and D) are 2021–2023, the hidden years — not read (like "
             "L6b.1, L6b.3). No money.", "",
             "| picture | his answer | the machine | lint |", "|---|---|---|---|"]
    for c, got, final in rows:
        lines.append(f"| `{final.name}` | {c.his} | {got['said']} | {'clean' if not got['lint'] else len(got['lint'])} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
