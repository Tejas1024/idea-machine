"""One label chart (L15.1): 1-hour candles, ONE band, the question written on the picture (#132), and nothing
after the moment the band is finished (lesson B16). His drawings and the daily charts get the SAME look, so a model
learns the candles and the band - not where a chart came from.

Blind, like his Q79 card (old/build/concept_chart.py C1C: "no calm number on the picture, and each chart ends at the
zone's birth, so he judges the candles, not an outcome"): no calm, no tier, no candle count, no date on the picture
(the x axis shows days and hours, India time), no money. The label file keeps the times (machine/labels/store.py).

    h = hour_candles(m5, first, last)          # the 5-minute candles he sees, grouped into hours (server clock)
    drawn = draw(word, h, lo, hi, "x.jpg", subject="...")     # rendered + every picture rule of his checked
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from machine.charts import draw as D
from machine.charts import lint
from machine.labels.words import Word

# The band: the drawer's own zone colour, filled darker than on other pictures - here the band IS the question (#118
# "that zone you marked it VERY LIGHTLY, it's hard to read"). Not the grey of his old zone cards: on these pictures the
# candles are grey (his approved quiet candles), and a grey band hid the very candles he judges (tried 30 Sep 2026).
BAND_COLOUR = D.ZONEC
BAND_ALPHA = 0.30


class LabelPictureError(RuntimeError):
    pass


def hour_candles(m5: pd.DataFrame, first, last) -> pd.DataFrame:
    """The 5-minute candles from `first` to `last` (MT5 server time, both included) grouped into 1-hour candles on the
    server clock = MT5's hour candles, except an hour seen only in part (then only that part - what he saw)."""
    w = m5.loc[pd.Timestamp(first):pd.Timestamp(last), ["open", "high", "low", "close"]]
    if w.empty:
        raise LabelPictureError(f"no 5-minute candles from {first} to {last}")
    h = w.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return h


def calm(h: pd.DataFrame, lo: float, hi: float) -> float:
    """machine/blocks/zones.py's calm, on the chart's own hour candles: the median range of the candles whose BODY
    touches the band / the median range of all of them (#79: "the candles were very small"). A measure for L15.2's
    comparison (the rule against his eye) - never drawn on the picture."""
    o, c = h["open"].to_numpy(float), h["close"].to_numpy(float)
    rng = (h["high"] - h["low"]).to_numpy(float)
    inb = (np.maximum(o, c) >= lo) & (np.minimum(o, c) <= hi)
    if not inb.any():
        return float("nan")
    all_rng = float(np.median(rng))
    return float(np.median(rng[inb]) / all_rng) if all_rng > 0 else float("nan")


def picture(word: Word, h: pd.DataFrame, lo: float, hi: float, *, subject: str) -> D.Picture:
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise LabelPictureError(f"a band needs a low under its high (got {lo} - {hi})")
    panel = D.Panel(h, "1h", [D.Candles(), D.Zone(float(lo), float(hi), word.band_name, color=BAND_COLOUR,
                                                  alpha=BAND_ALPHA)], role="main", clock="ist")
    return D.Picture(kind="question", subject=subject, heading=word.heading, sub=word.sub,
                     question=D.Question(word.ask, word.how), rows=[[panel]], entry_at=h.index[-1], entry_tf="1h")


def draw(word: Word, h: pd.DataFrame, lo: float, hi: float, path: str | Path, *, subject: str,
         shown: Path | None = None) -> D.Drawn:
    """Render and check every picture rule of his (machine/charts/lint.py). A broken rule = no picture."""
    drawn = D.render(picture(word, h, lo, hi, subject=subject), path)
    problems = lint.check(drawn, shown=shown)
    if problems:
        Path(path).unlink(missing_ok=True)
        raise LabelPictureError(f"{Path(path).name}: " + "; ".join(map(str, problems)))
    return drawn


def code_fingerprint() -> str:
    """The code that draws a label chart (this file, the drawer, the lint): a label's picture is redrawn when it
    changes (lesson B19: a cache carries the fingerprint of the code that built it)."""
    here = Path(__file__).resolve()
    files = [here, here.parent.parent / "charts" / "draw.py", here.parent.parent / "charts" / "lint.py",
             here.parent / "words.py"]
    sha = hashlib.sha256()
    for f in files:
        sha.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return sha.hexdigest()[:16]
