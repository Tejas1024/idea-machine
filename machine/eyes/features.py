"""Model A's numbers (L15.2): what the chart he answered shows, as numbers - from the SAME hour candles the picture
draws (machine/labels/picture.py hour_candles: first to last candle, nothing after the band is finished).

    from machine.eyes import features
    X = features.build(lab, m5)        # one row per answer, the columns of COLUMNS, index = label_id

"In the band" = the candle's BODY touches the band (the test zones.py and his calm use). Each number and the words of
his it comes from (strategy/LEARNED.md #78-#81, strategy-understanding §65s-§65v):
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.labels import picture as P

# name -> its meaning in easy English (#113) and where it comes from
COLUMNS: dict[str, str] = {
    "calm": "how small the candles in the band are against all the chart's candles (his rule's number; #79 'the "
            "candles were very small')",
    "calm_touch": "the same, for every candle that touches the band with any part",
    "body_small": "how small the BODIES in the band are against the chart's candles",
    "time_in_band": "the share of the chart's candles that sit in the band (#78 'spend more time')",
    "longest_stay": "the longest run of candles in the band, one after another, as a share of the chart (#78 "
                    "'continuous ... instead of counting different candles')",
    "stays": "how many separate times price sat in the band (#79: two quiet stretches at the same price are stronger)",
    "wick_touches": "candles that touch the band only with a wick, as a share of the chart (#79 option B: the classic "
                    "support and resistance that price spiked into)",
    "band_thick": "how tall the band is, in typical candles",
    "band_of_chart": "how tall the band is against the chart's whole price range",
    "band_place": "where the band sits in the chart's price range (0 = bottom, 1 = top)",
    "band_when": "when price sat in the band (0 = the chart's first candle, 1 = its last; #79 'a recent one')",
    "left_by": "how far price went away from the band after it last sat there, against the chart's range",
    "trend": "how much the chart went up (+) or down (-) from its first to its last candle, against its range",
    "busy": "how big the chart's candles are against the price (per 1,000)",
}


def of_chart(h: pd.DataFrame, lo: float, hi: float) -> dict[str, float]:
    """The numbers of one chart: h = its hour candles (open, high, low, close), the band lo-hi."""
    o, hh, ll, c = (h[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(h)
    if n < 2:
        raise ValueError("a chart needs at least 2 candles")
    rng = hh - ll
    med = float(np.median(rng))
    top, bot = np.maximum(o, c), np.minimum(o, c)
    body_in = (top >= lo) & (bot <= hi)
    wick_in = (hh >= lo) & (ll <= hi)
    span = float(hh.max() - ll.min())
    nan = float("nan")

    def ratio(a: float, b: float) -> float:
        return float(a / b) if b > 0 and np.isfinite(a) else nan

    idx = np.flatnonzero(body_in)
    runs = _runs(body_in)
    if idx.size:
        after = slice(idx[-1] + 1, n)
        away = max(float(np.max(hh[after] - hi, initial=0.0)), float(np.max(lo - ll[after], initial=0.0)), 0.0)
    else:
        away = nan
    return {
        "calm": ratio(float(np.median(rng[body_in])), med) if idx.size else nan,
        "calm_touch": ratio(float(np.median(rng[wick_in])), med) if wick_in.any() else nan,
        "body_small": ratio(float(np.median(np.abs(c - o)[body_in])), med) if idx.size else nan,
        "time_in_band": float(body_in.mean()),
        "longest_stay": float(max(runs, default=0) / n),
        "stays": float(len(runs)),
        "wick_touches": float((wick_in & ~body_in).mean()),
        "band_thick": ratio(hi - lo, med),
        "band_of_chart": ratio(hi - lo, span),
        "band_place": ratio((lo + hi) / 2 - float(ll.min()), span),
        "band_when": float(idx.mean() / (n - 1)) if idx.size else nan,
        "left_by": ratio(away, span),
        "trend": ratio(float(c[-1] - o[0]), span),
        "busy": ratio(1000.0 * med, float(np.median(c))),
    }


def build(lab: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """Every answer's numbers, from the 5-minute candles grouped into the hours the picture drew."""
    rows = []
    for r in lab.itertuples(index=False):
        h = P.hour_candles(m5, r.first_candle, r.last_candle)
        rows.append({"label_id": r.label_id, **of_chart(h, float(r.lo), float(r.hi))})
    return pd.DataFrame(rows, columns=["label_id", *COLUMNS]).set_index("label_id")


def _runs(mask: np.ndarray) -> list[int]:
    """The lengths of the runs of True, in order."""
    out, k = [], 0
    for m in mask:
        if m:
            k += 1
        elif k:
            out.append(k)
            k = 0
    if k:
        out.append(k)
    return out
