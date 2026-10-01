"""Pictures Tejas can read in 5 seconds, drawn HIS way (L4).

    from machine.charts import draw as D, lint
    pic = D.trade_picture(...)  /  D.question_picture(...)  /  D.whole_trade_picture(...)  /  D.arrows_picture(...)
    drawn = D.render(pic, "plan/work/samples/1-trade.jpg")    # .jpg = a REAL JPEG (#106), .png = a PNG
    problems = lint.check(drawn)                               # one check per rule of his (machine/charts/lint.py)
    (a news release, L12b.3: machine/news/similar.py builds a Picture of kind "news" from these items)

A picture is rows of PANELS; a panel is one chart (its bars = the time axis) with ITEMS drawn on it:
  Candles    the price action - the one thing nothing may sit on (#116)
  Level      a price line, its name in the margin just above it (or just below it) - never on the line it names (#115)
  Box        the gap: a BLUE box nothing covers, its size in cents (#110, his T09)
  Zone       a band across the chart (a strong zone, #113 rule 3: "zone = the box")
  Line       a trendline through the wick ends (#112) with its touches NUMBERED (#113 rule 3); two Lines in one
             group = a triangle, drawn with the price inside (#116)
  Fib        Fibonacci: its lines, each with its number (#113 rule 3)
  ADRBand    the day's ADR zones, his Pine study's two bands (strategy-code.md section 16)
  Arrow      his blue arrows walking through the price (#122); two points = one arrow
  Marker     B1 / S1 / H1 / B1 C with ONE keyword (#115, #121) - in the empty lane, a vertical leader to its candle
  Position   one position: a thick line at its price from where it opened to where it closed (his P01 picture)
  Span       a light band over a stretch of time (where the 5-minute chart below sits on the hour chart)
  Moment     one moment as a vertical line between two candles, named above the chart (a news release, L12b.3)

His rules and where they live:
  #109  pictures, not links; plain names ........................ words are plain English, entry names spelled out
  #110  the 5-minute chart, titled, the timeframe named; the gap a blue box; a small 1-hour picture .... every
        panel title says its timeframe AND its clock (lesson B24); trade_picture puts the hour chart on top
  #113  rule 3: every thing named is drawn ...................... lint.named_not_drawn
  #115  1-2 letter labels that never cover the line they name ... _Placer + lint.label_overlap, lint.point_words
  #116  nothing on the price action; words and arrows in empty space; leaders VERTICAL ...... _Placer
  #117  never the same picture twice; every entry and exit its own 5-minute panel, the hour chart above
  #120  the reference picture: old/ml/agent/logs/simple/one/A01.jpg (the first chart he did not reject)
  #121  ONE image per trade, like his TradingView screenshots ... one render() = one file
  #122  / #123 the blue ARROWS (no candles) AND the real candles - arrows_picture + whole_trade_picture
  #132  the question is written ON the picture ................. the yellow band (Question)
  #106  a .jpg is a real JPEG .................................. render() writes it with Pillow
  B16   a question picture ENDS at the entry candle and shows no money ...... question_picture cuts the bars
  B8    "$" is money here, not maths ........................... text.parse_math is off
Ideas copied (not imported) from old/build/one_chart.py (draw_one), one_image.py, arrow_image.py, pair_pack.py,
long_trade.py (ask_band) and simple_trades.py.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from machine.data import clock  # noqa: E402
from machine.tester.models import TF_MINUTES  # noqa: E402

plt.rcParams["text.parse_math"] = False           # lesson B8: "$5" is five dollars, not maths

# ---- the look he approved (old/build/one_chart.py, arrow_image.py) ---------------------------------------------
GREEN, RED, INK, MUTED, GRID = "#1b8a5a", "#d0342c", "#1f2328", "#6e7781", "#e3e6ea"
C_UP, C_DN = "#b7c0ca", "#69717a"                 # quiet candles: the positions are the story
BLUE = "#2f6fdb"                                  # the gap (#110) and his arrows (#122)
PURPLE = "#7a3fb0"                                # a line price takes (liquidity, trendline)
ZONEC = "#8a6d3b"
FIBC = "#b36b00"
ADR_TOP, ADR_BOT = "#8b2346", "#2e7d4f"           # his Pine study: maroon upper zone, green lower zone
SPANC = "#9aa4ae"
ASKBG, ASKEDGE, ASKINK = "#fff6e0", "#e0c890", "#7a4f00"

F_HEAD, F_RESULT, F_SUB, F_PANEL, F_TAG, F_NAME, F_AXIS, F_FOOT = 21, 23, 13.5, 13, 13.5, 12.5, 10.5, 10
F_POINT, F_Q_HEAD, F_Q = 11.5, 15, 12.5
DPI = 150
WIDTH_IN = 10.4

TF_WORDS = {"1m": "1-minute", "5m": "5-minute", "15m": "15-minute", "1h": "1-hour", "4h": "4-hour", "1d": "daily",
            "1w": "weekly"}
CLOCK_WORDS = {"ist": "India time", "server": "MT5 server time", "utc": "UTC"}
ROLE_HEIGHT_IN = {"main": 4.6, "hour": 2.3, "point": 1.9}

Time = Any                                        # anything pd.Timestamp takes; MT5 SERVER time, no time zone
Side = Literal["buy", "sell"]


def _ts(t: Time) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        raise ValueError(f"{t!r} has a time zone; items take MT5 server time without one (like load_bars' index)")
    return ts


def side_colour(side: str) -> str:
    return GREEN if side == "buy" else RED


# =================================================================================================================
# the items
# =================================================================================================================

@dataclass(frozen=True)
class Candles:
    """The panel's own bars as candles. Leave it out and the panel has a time axis but no candles (#122)."""
    width: float = 0.72


@dataclass(frozen=True)
class Level:
    """A price line. Its name goes in the margin, just above the line - never on it (#115). name_below: just under
    it instead (the lower of two close lines, so their names never meet)."""
    price: float
    name: str
    start: Time | None = None                     # None = from the left edge
    end: Time | None = None                       # None = to the right edge
    color: str = INK
    style: str = "--"
    width: float = 1.7
    name_below: bool = False


@dataclass(frozen=True)
class Box:
    """The gap: a blue box nothing covers (#110), from its first candle to where price came back. Named with its
    size in cents (his T09: a 16-cent gap on a three-dollar chart is a thing he cannot see)."""
    start: Time
    end: Time
    lo: float
    hi: float
    name: str = "gap"
    color: str = BLUE
    named: bool = True                            # False: a point's keyword names it ("B1 gap"), no callout

    @property
    def label(self) -> str:
        c = round((self.hi - self.lo) * 100)
        return f"{self.name} {c} cent{'' if c == 1 else 's'}"


@dataclass(frozen=True)
class Zone:
    """A band across the chart (#113 rule 3: a zone is drawn as its box). Name in the left margin. alpha: how dark
    it is filled (a label chart, where the band IS the question, fills it darker - his #118 "that zone you marked it
    VERY LIGHTLY, it's hard to read")."""
    lo: float
    hi: float
    name: str = "strong zone"
    start: Time | None = None
    end: Time | None = None
    color: str = ZONEC
    alpha: float = 0.16


@dataclass(frozen=True)
class Line:
    """A trendline through the wick ends (#112), from `start` to `end`, drawn on to `extend_to`.
    Every touch is NUMBERED 1, 2, 3 on the empty side of the line (#113 rule 3, #114 "I want to clearly see the
    touches"). Two Lines with the same `group` are one shape - "triangle": lint checks the price is inside (#116)."""
    start: tuple[Time, float]
    end: tuple[Time, float]
    touches: tuple[Time, ...] = ()
    name: str = "trendline"
    extend_to: Time | None = None
    group: str | None = None
    color: str = PURPLE


@dataclass(frozen=True)
class Fib:
    """Fibonacci from `start` (level 1) to `end` (level 0), the way TradingView draws a retracement: each line
    with its number in the right margin (#113 rule 3)."""
    start: tuple[Time, float]
    end: tuple[Time, float]
    levels: tuple[float, ...] = (0.0, 0.5, 0.618, 1.0)
    name: str = "Fibonacci"
    color: str = FIBC

    def price(self, level: float) -> float:
        return self.end[1] + (self.start[1] - self.end[1]) * level


@dataclass(frozen=True)
class ADRBand:
    """The day's ADR zones, his Pine study (strategy-code.md section 16): upper zone from dayOpen + ADR5/2 to
    dayOpen + ADR10/2 (maroon), lower zone the mirror (green). Drawn over the day's own hours."""
    start: Time
    end: Time
    day_open: float
    adr5: float
    adr10: float
    name: str = "ADR"

    @property
    def upper(self) -> tuple[float, float]:
        a, b = sorted((self.adr5, self.adr10))
        return self.day_open + a / 2, self.day_open + b / 2

    @property
    def lower(self) -> tuple[float, float]:
        a, b = sorted((self.adr5, self.adr10))
        return self.day_open - b / 2, self.day_open - a / 2


@dataclass(frozen=True)
class Arrow:
    """His blue arrows (#122): one arrowhead per turn of the path. Two points = one arrow."""
    points: tuple[tuple[Time, float], ...]
    color: str = BLUE
    width: float = 2.4


@dataclass(frozen=True)
class Marker:
    """A moment of the trade: B1 (1st buy), S1 (1st sell), H1, and "B1 C" where B1 closed (#115, #121), with ONE
    keyword saying why ("gap", "target", "stop"). No money on the point, no word "hedge" (#116)."""
    time: Time
    price: float
    tag: str
    side: Side
    word: str = ""
    label: bool = True                            # False on a close-up that names the point in its title

    @property
    def text(self) -> str:
        return f"{self.tag}\n{self.word}" if self.word else self.tag


@dataclass(frozen=True)
class Position:
    """One position: a thick line at its price from where it opened to where it closed (his P01 picture)."""
    start: Time
    end: Time
    price: float
    side: Side


@dataclass(frozen=True)
class Span:
    """A light band over a stretch of time, named above the chart (the 5-minute chart's hours on the hour chart)."""
    start: Time
    end: Time
    name: str


@dataclass(frozen=True)
class Moment:
    """One moment as a vertical line (a news release, L12b.3): at its exact place in time - a release at a candle's
    open sits BETWEEN two candles, on no candle (#116) - from the bottom of the chart to just over the candles, its
    name in the empty lane above."""
    time: Time
    name: str
    color: str = INK


Item = Candles | Level | Box | Zone | Line | Fib | ADRBand | Arrow | Marker | Position | Span | Moment


# =================================================================================================================
# panels and pictures
# =================================================================================================================

@dataclass
class Panel:
    """One chart. `bars` (server-time index = bar OPEN times; open high low close) is its time axis.
    The title always ends with the timeframe and the clock (#110, lesson B24): "... · 5-minute chart · India time"."""
    bars: pd.DataFrame
    tf: str
    items: list = field(default_factory=list)
    title: str = ""
    role: Literal["main", "hour", "point"] = "main"
    clock: Literal["ist", "server", "utc"] = "ist"
    price_axis: bool = True
    entry_at: Time | None = None                  # a question picture with several setups (L10b SHOWN): the entry
                                                  # candle THIS chart ends at (None = the picture's entry_at)

    @property
    def full_title(self) -> str:
        tail = f"{TF_WORDS[self.tf]} chart · {CLOCK_WORDS[self.clock]}"
        return f"{self.title}  ·  {tail}" if self.title else tail


@dataclass(frozen=True)
class Question:
    """The question, written ON the picture (#132): the question in bold, then how to answer, in plain words."""
    ask: str
    how: tuple[str, ...] = ()


@dataclass
class Picture:
    kind: Literal["trade", "question", "whole_trade", "arrows", "news"]
    subject: str                                  # what it shows - never the same picture twice (#117)
    heading: str                                  # "SELL GOLD"
    rows: list[list[Panel]]
    heading_color: str = INK
    result: str | None = None                     # "+₹790" - never on a question (lesson B16)
    result_color: str = INK
    sub: str = ""                                 # one plain sentence; "\n" for a second line
    question: Question | None = None
    footer_left: str = ""
    footer_right: str = ""
    entry_at: Time | None = None                  # the entry candle's OPEN time: a question picture ends here (B16)
    entry_tf: str = "5m"


# =================================================================================================================
# what was drawn - the record lint.py reads (display pixels, y up)
# =================================================================================================================

@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, o: "Rect", shrink: float = 1.0) -> bool:
        return (min(self.x1, o.x1) - max(self.x0, o.x0) > 2 * shrink and
                min(self.y1, o.y1) - max(self.y0, o.y0) > 2 * shrink)

    def holds(self, x: float, y: float, shrink: float = 1.0) -> bool:
        return self.x0 + shrink < x < self.x1 - shrink and self.y0 + shrink < y < self.y1 - shrink


@dataclass
class TextOnPicture:
    text: str
    role: str        # heading result sub question footer panel_title tick tag name touch number span
    panel: int | None
    box: Rect
    names: int | None = None                      # the item it names (index into Drawn.items)


@dataclass
class ItemOnPicture:
    kind: str
    panel: int
    name: str
    visible: bool
    info: dict = field(default_factory=dict)
    paths: list = field(default_factory=list)     # display polylines: a list of (N, 2) arrays
    boxes: list = field(default_factory=list)     # display rectangles


@dataclass
class PanelOnPicture:
    index: int
    tf: str
    role: str
    title: str
    clock: str
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    n_bars: int
    area: Rect
    candles: list = field(default_factory=list)   # one Rect per candle, wick end to wick end
    bar_minutes: float | None = None              # the usual time between two candles (None: one candle only)
    entry_at: pd.Timestamp | None = None          # this chart's own entry candle (Panel.entry_at), if it has one


@dataclass
class Drawn:
    path: Path
    kind: str
    subject: str
    size: tuple[int, int]
    panels: list[PanelOnPicture]
    items: list[ItemOnPicture]
    texts: list[TextOnPicture]
    question: Question | None
    entry_at: pd.Timestamp | None
    entry_tf: str
    parse_math: bool
    pixel_sha256: str
    crowded: list[str] = field(default_factory=list)


# =================================================================================================================
# drawing
# =================================================================================================================

def render(pic: Picture, path: str | Path) -> Drawn:
    """Draw the picture into `path` (.jpg / .jpeg = a real JPEG, #106; .png = PNG) and return what was drawn."""
    path = Path(path)
    kind = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{path.name}: a picture is .jpg or .png")
    with plt.rc_context({"text.parse_math": False}):
        parse_math = bool(plt.rcParams["text.parse_math"])
        fig, drawn = _draw(pic, path, parse_math)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white", dpi=DPI)
        plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if kind == "JPEG":
        img.save(tmp, format="JPEG", quality=92, optimize=True)
    else:
        img.save(tmp, format="PNG", optimize=True)
    tmp.replace(path)
    drawn.pixel_sha256 = pixel_sha256(path)
    return drawn


def pixel_sha256(path: str | Path) -> str:
    """The hash of what the picture SHOWS (its decoded pixels), not of the file's bytes."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()


def _draw(pic: Picture, path: Path, parse_math: bool) -> tuple[plt.Figure, Drawn]:
    sub_lines = pic.sub.count("\n") + 1 if pic.sub else 0
    head_in = 0.62 + 0.30 * sub_lines
    q_in = (0.34 + 0.29 * (1 + len(pic.question.how))) if pic.question else 0.0
    rows_in = [0.40 + max(ROLE_HEIGHT_IN[p.role] for p in row) + 0.34 + 0.14 for row in pic.rows]
    foot_in = 0.50
    height_in = head_in + q_in + (0.12 if q_in else 0) + sum(rows_in) + foot_in
    fig = plt.figure(figsize=(WIDTH_IN, height_in), dpi=DPI)
    fig.patch.set_facecolor("white")
    W, H = WIDTH_IN, height_in
    fx = lambda inch: inch / W                    # noqa: E731
    fy = lambda inch: inch / H                    # noqa: E731

    # ---- heading: the side on one line with the result opposite it, one plain sentence underneath
    fig.text(fx(0.35), fy(H - 0.14), pic.heading, fontsize=F_HEAD, fontweight="bold", color=pic.heading_color,
             ha="left", va="top")
    if pic.result:
        fig.text(fx(W - 0.35), fy(H - 0.12), pic.result, fontsize=F_RESULT, fontweight="bold",
                 color=pic.result_color, ha="right", va="top")
    if pic.sub:
        fig.text(fx(0.35), fy(H - 0.62), pic.sub, fontsize=F_SUB, color=INK, ha="left", va="top", linespacing=1.35)
    y_top = H - head_in

    # ---- the question, ON the picture (#132): old/build/long_trade.py ask_band
    if pic.question:
        q = pic.question
        fig.add_artist(Rectangle((fx(0.25), fy(y_top - q_in)), fx(W - 0.50), fy(q_in),
                                 transform=fig.transFigure, fc=ASKBG, ec=ASKEDGE, lw=1.1, zorder=0))
        fig.text(fx(0.40), fy(y_top - 0.14), q.ask, fontsize=F_Q_HEAD, fontweight="bold", color=ASKINK,
                 ha="left", va="top")
        for k, line in enumerate(q.how):
            fig.text(fx(0.40), fy(y_top - 0.14 - 0.34 - 0.29 * k), line, fontsize=F_Q, color=INK, ha="left",
                     va="top")
        y_top -= q_in + 0.12

    # ---- the rows of charts
    drawn = Drawn(path=path, kind=pic.kind, subject=pic.subject, size=(0, 0), panels=[], items=[], texts=[],
                  question=pic.question, entry_at=None if pic.entry_at is None else _ts(pic.entry_at),
                  entry_tf=pic.entry_tf, parse_math=parse_math, pixel_sha256="")
    runs: list[_Run] = []
    for row, row_in in zip(pic.rows, rows_in):
        k = len(row)
        right_pad = 0.72 if any(p.price_axis for p in row) else 0.25
        gap = 0.28
        w_each = (W - 0.35 - right_pad - (k - 1) * (gap + (right_pad - 0.25))) / k
        axes_h = max(ROLE_HEIGHT_IN[p.role] for p in row)
        y_axes = y_top - 0.40 - axes_h
        for j, panel in enumerate(row):
            x0 = 0.35 + j * (w_each + gap + (right_pad - 0.25))
            ax = fig.add_axes([fx(x0), fy(y_axes), fx(w_each), fy(axes_h)])
            runs.append(_Run(fig, ax, panel, len(runs), (w_each * DPI, axes_h * DPI)))
        y_top -= row_in

    # ---- footer: what the letters mean, and the money's settings
    if pic.footer_left:
        fig.text(fx(0.35), fy(0.14), pic.footer_left, fontsize=F_FOOT, color=MUTED, ha="left", va="bottom")
    if pic.footer_right:
        fig.text(fx(W - 0.35), fy(0.14), pic.footer_right, fontsize=F_FOOT, color=MUTED, ha="right", va="bottom")

    for run in runs:
        run.draw(drawn)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    drawn.size = (int(round(W * DPI)), int(round(H * DPI)))
    roles = {pic.heading: "heading", pic.result: "result", pic.sub: "sub", pic.footer_left: "footer",
             pic.footer_right: "footer"}
    for t in fig.texts:
        s = t.get_text()
        role = "question" if pic.question and (s == pic.question.ask or s in pic.question.how) else roles.get(s, "text")
        drawn.texts.append(TextOnPicture(s, role, None, _extent(t, renderer)))
    for run in runs:
        run.record(drawn, renderer)
    return fig, drawn


def _extent(t, renderer) -> Rect:
    """The box a piece of text covers, INCLUDING its white label box when it has one."""
    bb = t.get_window_extent(renderer)
    pad = 0.0
    patch = t.get_bbox_patch()
    if patch is not None:
        pad = 0.30 * t.get_fontsize() * DPI / 72 + 1.5
    return Rect(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


class _Run:
    """One panel being drawn."""

    def __init__(self, fig, ax, panel: Panel, index: int, size_px: tuple[float, float]):
        self.fig, self.ax, self.p, self.index = fig, ax, panel, index
        self.w_px, self.h_px = size_px
        b = panel.bars
        if b is None or b.empty:
            raise ValueError("a panel needs bars (its time axis)")
        if b.index.tz is not None:
            raise ValueError("a panel's bars are indexed by MT5 server time without a time zone")
        self.idx = pd.DatetimeIndex(b.index)
        self.n = len(b)
        self.o, self.h = b["open"].to_numpy(float), b["high"].to_numpy(float)
        self.l, self.c = b["low"].to_numpy(float), b["close"].to_numpy(float)
        self.has_candles = any(isinstance(it, Candles) for it in panel.items)
        self.labels: list[tuple[Any, str, int | None]] = []      # (text artist, role, item index)
        self.item_records: list[ItemOnPicture] = []
        self.crowded: list[str] = []
        self.leaders: list = []

    # ---- time and price -> chart position ------------------------------------------------------------------------
    def x(self, t: Time) -> float:
        """The bar a time falls in (a time before the chart = -1; after its last bar = the last bar)."""
        return float(int(self.idx.searchsorted(_ts(t), side="right")) - 1)

    def in_view(self, t: Time) -> bool:
        """Is this time inside the chart's candles (from the first bar's open to the last bar's close)?"""
        return self.idx[0] <= _ts(t) < self.idx[-1] + pd.Timedelta(minutes=TF_MINUTES[self.p.tf])

    def x_or_edge(self, t: Time | None, left: bool) -> float:
        if t is None:
            return -0.5 if left else self.n - 0.5
        return float(np.clip(self.x(t), -0.5, self.n - 0.5))

    def x_exact(self, t: Time) -> float:
        """A moment's exact place: a candle's OPEN is its left edge (x - 0.5), so a release at 18:00 on a 5-minute
        chart sits between the 17:55 and 18:00 candles; a minute inside a candle sits that far into it."""
        j = self.x(t)
        if j < 0:
            return -0.5
        into = (_ts(t) - self.idx[int(j)]) / pd.Timedelta(minutes=TF_MINUTES[self.p.tf])
        return float(min(j - 0.5 + min(max(into, 0.0), 1.0), self.n - 0.5))

    def low_at(self, x: float) -> float:
        i = int(np.clip(round(x), 0, self.n - 1))
        return float(self.l[i]) if self.has_candles else float("nan")

    def high_at(self, x: float) -> float:
        i = int(np.clip(round(x), 0, self.n - 1))
        return float(self.h[i]) if self.has_candles else float("nan")

    # ---- the picture -----------------------------------------------------------------------------------------------
    def draw(self, drawn: Drawn) -> None:
        ax, p = self.ax, self.p
        items = [it for it in p.items if not isinstance(it, Candles)]

        # what must fit on the price scale: the candles and every drawn thing
        ys = list(self.l) + list(self.h) if self.has_candles else []
        for it in items:
            ys += _prices(it)
        if not ys:
            ys = list(self.l) + list(self.h)
        lo, hi = float(np.nanmin(ys)), float(np.nanmax(ys))
        rng = max(hi - lo, 1e-6)

        callouts_below = sum(1 for it in items if (isinstance(it, Marker) and it.label) or isinstance(it, Line)
                             or (isinstance(it, Box) and it.named))
        callouts_above = sum(1 for it in items if isinstance(it, (Span, Line, Moment)))
        renderer = self.fig.canvas.get_renderer()
        probe = ax.text(0, 0, "B1\ngap", fontsize=F_TAG, fontweight="bold", linespacing=1.15,
                        bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=INK))
        lane_px = _extent(probe, renderer).y1 - _extent(probe, renderer).y0 + 6
        probe.remove()
        f = lane_px / self.h_px
        n_below = 0 if callouts_below == 0 else (1 if callouts_below <= 3 else 2 if callouts_below <= 6 else 3)
        n_above = 0 if callouts_above == 0 else 1
        B = n_below * f + (0.05 if n_below else 0.06)
        T = n_above * f + (0.05 if n_above else 0.06)
        if 1 - B - T < 0.30:                     # a small panel: fewer lanes, and lint will say if it got crowded
            n_below = min(n_below, 1)
            B = n_below * f + 0.05
            T = n_above * f + 0.05
        total = rng / max(1 - B - T, 0.2)
        self.y0, self.y1 = lo - B * total, hi + T * total
        self.lanes_below = [lo - (0.035 + k * f) * total for k in range(n_below)]
        self.lanes_above = [hi + (0.030 + k * f) * total for k in range(n_above)]
        self.price_lo, self.price_hi = lo, hi
        ax.set_ylim(self.y0, self.y1)

        # margins: the left one for level and zone names, the right one for Fibonacci numbers and ADR names
        left_names = [it.name for it in items if isinstance(it, Level) and it.start is None]
        left_names += [it.name for it in items if isinstance(it, Zone)]
        right_names = [it.name for it in items if isinstance(it, Level) and it.start is not None and it.end is None]
        right_names += [f"{lv:g}" for it in items if isinstance(it, Fib) for lv in it.levels]
        right_names += [f"{it.name} {w}" for it in items if isinstance(it, ADRBand) for w in ("top", "bottom")]
        lm_px = max([_text_px(ax, s, F_NAME, renderer) for s in left_names], default=0.0)
        rm_px = max([_text_px(ax, s, F_NAME, renderer) for s in right_names], default=0.0)
        lm_px = lm_px + 14 if lm_px else 8.0
        rm_px = rm_px + 14 if rm_px else 8.0
        per_bar = (self.w_px - lm_px - rm_px) / self.n
        self.xl0, self.xl1 = -0.5 - lm_px / per_bar, self.n - 0.5 + rm_px / per_bar
        ax.set_xlim(self.xl0, self.xl1)

        # the drawing, back to front
        for k, it in enumerate(items):
            rec = ItemOnPicture(kind=type(it).__name__, panel=self.index, name=getattr(it, "name", "") or "",
                                visible=False)
            rec.info["item"] = it
            self.item_records.append(rec)
        if self.has_candles:
            cw = next(it for it in p.items if isinstance(it, Candles)).width
            for j in range(self.n):
                col = C_UP if self.c[j] >= self.o[j] else C_DN
                ax.vlines(j, self.l[j], self.h[j], color=col, lw=1.0 if self.n < 200 else 0.7, zorder=2)
                ax.add_patch(Rectangle((j - cw / 2, min(self.o[j], self.c[j])), cw,
                                       max(abs(self.c[j] - self.o[j]), 1e-6), color=col, zorder=2, lw=0))
        for k, it in enumerate(items):
            getattr(self, "_" + type(it).__name__.lower())(it, k)

        # axes: time underneath in the panel's clock, price on the right like his MT5 / TradingView
        tk = np.linspace(0, self.n - 1, 4 if p.role != "point" else 2).round().astype(int)
        shown = _clock_times(self.idx[tk], p.clock)
        fmt = "%d %b %H:%M" if TF_MINUTES[p.tf] < 1440 else "%d %b %Y"
        ax.set_xticks(tk)
        labs = ax.set_xticklabels([t.strftime(fmt) for t in shown], color=MUTED,
                                  fontsize=F_AXIS if p.role != "point" else F_AXIS - 1.5)
        labs[0].set_horizontalalignment("left")   # the first date starts at its candle, not off the picture
        if not p.price_axis:                      # a close-up's last date ends at its candle, not in the next chart
            labs[-1].set_horizontalalignment("right")
        if p.price_axis:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelsize=F_AXIS, colors=MUTED)
            ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        else:
            ax.set_yticks([])
        for sp in ("top", "left"):
            ax.spines[sp].set_visible(False)
        for sp in ("right", "bottom"):
            ax.spines[sp].set_color(GRID)
        self.title_artist = ax.set_title(p.full_title, loc="left", fontweight="bold", color=INK, pad=6,
                                         fontsize=F_PANEL if p.role != "point" else F_POINT)

        # the words go last, into the empty space (#115, #116)
        self.fig.canvas.draw()
        placer = _Placer(self)
        for k, it in enumerate(items):
            if isinstance(it, Marker) and it.label:
                self._callout(placer, it.text, self.x(it.time), it.price, side_colour(it.side), k, "tag", True)
            elif isinstance(it, Box) and it.named:
                x = self.x(it.start) + 1
                self._callout(placer, it.label, x, (it.lo + it.hi) / 2, it.color, k, "name", True)
            elif isinstance(it, Line):
                self._line_touches(placer, it, k)
            elif isinstance(it, Span):
                xm = (self.x_or_edge(it.start, True) + self.x_or_edge(it.end, False)) / 2
                self._callout(placer, it.name, xm, None, MUTED, k, "span", False, leader=False)
            elif isinstance(it, Moment):             # above the line; slid aside, a leader still joins them
                self._callout(placer, it.name, self.x_exact(it.time), self.moment_top, it.color, k, "name",
                              False, to_price=True)
        self.placer = placer

    # ---- one method per item -------------------------------------------------------------------------------------
    def _level(self, it: Level, k: int) -> None:
        x0, x1 = self.x_or_edge(it.start, True), self.x_or_edge(it.end, False)
        self.ax.plot([x0, x1], [it.price] * 2, color=it.color, lw=it.width, ls=it.style, zorder=4)
        rec = self.item_records[k]
        rec.paths.append([(x0, it.price), (x1, it.price)])
        rec.visible = x1 > x0 and self.y0 < it.price < self.y1
        rec.info["last_time"] = None if it.end is None else _ts(it.end)
        pad = 0.012 * (self.y1 - self.y0)
        # the line runs on, dotted, under its name in the margin - so the word sits ON nothing but points at it
        if it.start is None:
            self.ax.plot([self.xl0 + 0.1, -0.5], [it.price] * 2, color=it.color, lw=1.0, ls=":", zorder=4)
        elif it.end is None:
            self.ax.plot([self.n - 0.5, self.xl1], [it.price] * 2, color=it.color, lw=1.0, ls=":", zorder=4)
        y, va = (it.price - pad, "top") if it.name_below else (it.price + pad, "bottom")
        if it.start is None:                      # the name in the left margin, just above (or below) the line
            t = self.ax.text(self.xl0 + 0.15, y, it.name, color=it.color, fontsize=F_NAME,
                             fontweight="bold", ha="left", va=va, zorder=7)
        elif it.end is None:                      # ... or the right margin
            t = self.ax.text(self.n - 0.5 + 0.25, y, it.name, color=it.color, fontsize=F_NAME,
                             fontweight="bold", ha="left", va=va, zorder=7)
        else:                                     # a short line: the name just above its right end
            t = self.ax.text(x1, y, it.name, color=it.color, fontsize=F_NAME, fontweight="bold",
                             ha="right", va=va, zorder=7)
        self.labels.append((t, "name", k))

    def _box(self, it: Box, k: int) -> None:
        x0, x1 = self.x(it.start) - 0.45, self.x(it.end) + 0.45
        x0, x1 = max(x0, -0.5), min(x1, self.n - 0.5)
        rec = self.item_records[k]
        if x1 > x0:
            self.ax.add_patch(Rectangle((x0, it.lo), x1 - x0, max(it.hi - it.lo, 1e-6), fc=it.color, alpha=0.30,
                                        ec=it.color, lw=1.8, zorder=3))
            rec.boxes.append((x0, it.lo, x1, it.hi))
            rec.visible = (_ts(it.end) >= self.idx[0] and self.in_view(max(_ts(it.start), self.idx[0]))
                           and self.y0 < it.hi and it.lo < self.y1)
        rec.info["last_time"] = _ts(it.end)

    def _zone(self, it: Zone, k: int) -> None:
        x0, x1 = self.x_or_edge(it.start, True), self.x_or_edge(it.end, False)
        if it.start is None:
            x0 = self.xl0
        self.ax.add_patch(Rectangle((x0, it.lo), x1 - x0, max(it.hi - it.lo, 1e-6), fc=it.color, alpha=it.alpha,
                                    ec=it.color, lw=1.0, zorder=1))
        rec = self.item_records[k]
        rec.boxes.append((x0, it.lo, x1, it.hi))
        rec.visible = x1 > x0 and self.y0 < it.hi and it.lo < self.y1
        t = self.ax.text(self.xl0 + 0.15, it.hi + 0.012 * (self.y1 - self.y0), it.name, color=it.color,
                         fontsize=F_NAME, fontweight="bold", ha="left", va="bottom", zorder=7)
        self.labels.append((t, "name", k))

    def _line_xy(self, it: Line) -> tuple[float, float, float, float, float]:
        xa, xb = self.x(it.start[0]), self.x(it.end[0])
        pa, pb = float(it.start[1]), float(it.end[1])
        slope = (pb - pa) / (xb - xa) if xb != xa else 0.0
        x_end = self.x(it.extend_to) if it.extend_to is not None else xb
        return xa, pa, slope, max(xa, -0.5), min(max(x_end, xb), self.n - 0.5)

    def _line(self, it: Line, k: int) -> None:
        xa, pa, slope, x0, x1 = self._line_xy(it)
        rec = self.item_records[k]
        if x1 > x0:
            y0_, y1_ = pa + slope * (x0 - xa), pa + slope * (x1 - xa)
            self.ax.plot([x0, x1], [y0_, y1_], color=it.color, lw=2.4, zorder=4, solid_capstyle="round")
            rec.paths.append([(x0, y0_), (x1, y1_)])
            rec.visible = True
        rec.info.update(touches=len(it.touches), group=it.group, first_time=_ts(it.start[0]),
                        last_time=_ts(it.extend_to) if it.extend_to is not None else _ts(it.end[0]))

    def _fib(self, it: Fib, k: int) -> None:
        x0 = max(self.x(it.start[0]), -0.5)
        x1 = self.n - 0.5
        rec = self.item_records[k]
        numbers = []
        for lv in it.levels:
            px = it.price(lv)
            self.ax.plot([x0, x1], [px, px], color=it.color, lw=1.4, ls="-", zorder=4, alpha=0.9)
            rec.paths.append([(x0, px), (x1, px)])
            t = self.ax.text(self.n - 0.5 + 0.25, px, f"{lv:g}", color=it.color, fontsize=F_NAME,
                             fontweight="bold", ha="left", va="center", zorder=7)
            self.labels.append((t, "number", k))
            numbers.append(f"{lv:g}")
        rec.visible = x1 > x0
        rec.info["numbers"] = numbers

    def _adrband(self, it: ADRBand, k: int) -> None:
        x0, x1 = self.x_or_edge(it.start, True), self.x_or_edge(it.end, False)
        rec = self.item_records[k]
        for (a, b), col, word in ((it.upper, ADR_TOP, "top"), (it.lower, ADR_BOT, "bottom")):
            self.ax.add_patch(Rectangle((x0, a), x1 - x0, max(b - a, 1e-6), fc=col, alpha=0.17, ec=col, lw=0.9,
                                        zorder=1))
            rec.boxes.append((x0, a, x1, b))
            t = self.ax.text(self.n - 0.5 + 0.25, (a + b) / 2, f"{it.name} {word}", color=col, fontsize=F_NAME,
                             fontweight="bold", ha="left", va="center", zorder=7)
            self.labels.append((t, "name", k))
        rec.visible = x1 > x0 and self.y0 < it.upper[1] and it.lower[0] < self.y1

    def _arrow(self, it: Arrow, k: int) -> None:
        pts = [(self.x(t), float(px)) for t, px in it.points]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            self.ax.annotate("", xy=(xb, yb), xytext=(xa, ya), zorder=6,
                             arrowprops=dict(arrowstyle="-|>", color=it.color, lw=it.width, shrinkA=0, shrinkB=0,
                                             mutation_scale=15))
        rec = self.item_records[k]
        rec.paths.append(pts)
        rec.visible = len(pts) >= 2
        rec.info["last_time"] = max(_ts(t) for t, _ in it.points)

    def _marker(self, it: Marker, k: int) -> None:
        x = self.x(it.time)
        col = side_colour(it.side)
        self.ax.plot([x], [it.price], "o", ms=9.5, mfc="none", mec=col, mew=2.6, zorder=8)
        rec = self.item_records[k]
        rec.visible = self.in_view(it.time)
        rec.name = it.tag
        rec.info.update(tag=it.tag, word=it.word, side=it.side, time=_ts(it.time), last_time=_ts(it.time))

    def _position(self, it: Position, k: int) -> None:
        x0, x1 = self.x_or_edge(it.start, True), self.x_or_edge(it.end, False)
        self.ax.plot([x0, x1], [it.price] * 2, color=side_colour(it.side), lw=3.6, solid_capstyle="round",
                     zorder=5, alpha=0.9)
        rec = self.item_records[k]
        rec.paths.append([(x0, it.price), (x1, it.price)])
        rec.visible = x1 > x0
        rec.info["last_time"] = _ts(it.end)

    def _span(self, it: Span, k: int) -> None:
        x0, x1 = self.x_or_edge(it.start, True) - 0.5, self.x_or_edge(it.end, False) + 0.5
        x0, x1 = max(x0, -0.5), min(x1, self.n - 0.5)
        self.ax.add_patch(Rectangle((x0, self.y0), x1 - x0, self.y1 - self.y0, fc=SPANC, alpha=0.16, lw=0,
                                    zorder=0))
        rec = self.item_records[k]
        rec.boxes.append((x0, self.y0, x1, self.y1))
        rec.visible = x1 > x0
        rec.info["last_time"] = _ts(it.end)

    @property
    def moment_top(self) -> float:
        """A Moment's line ends just over the candles; its name sits in the lane above."""
        return self.price_hi + 0.012 * (self.y1 - self.y0)

    def _moment(self, it: Moment, k: int) -> None:
        x = self.x_exact(it.time)
        top = self.moment_top
        self.ax.plot([x, x], [self.y0, top], color=it.color, lw=1.5, ls=(0, (5, 3)), zorder=3)
        rec = self.item_records[k]
        rec.paths.append([(x, self.y0), (x, top)])
        rec.visible = self.in_view(it.time)
        rec.info.update(time=_ts(it.time), last_time=_ts(it.time))

    # ---- words in the empty space ------------------------------------------------------------------------------------
    def _callout(self, placer: "_Placer", text: str, x: float, price: float | None, colour: str, k: int, role: str,
                 below: bool, leader: bool = True, to_price: bool = False) -> None:
        t = self.ax.text(x, 0, text, fontsize=F_TAG if role == "tag" else F_NAME, fontweight="bold", color=colour,
                         ha="center", va="top", zorder=9, linespacing=1.15,
                         bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=colour, lw=1.3))
        lanes = ([(y, "top") for y in self.lanes_below] + [(y, "bottom") for y in self.lanes_above]) if below else \
                ([(y, "bottom") for y in self.lanes_above] + [(y, "top") for y in self.lanes_below])
        ok = placer.place(t, x, lanes)
        if not ok:
            self.crowded.append(text.replace("\n", " "))
        self.labels.append((t, role, k))
        if not leader:
            return
        lx, ly = t.get_position()
        va = t.get_va()
        span = self.y1 - self.y0
        if va == "top":                          # the label is under the chart: up to just under the candle
            end_y = (self.low_at(x) if self.has_candles and not to_price else price) - 0.015 * span
            turn_y = min(end_y, self.price_lo - 0.012 * span)
            start_y = ly + 0.004 * span
        else:                                    # above the chart: down to just over the candle
            end_y = (self.high_at(x) if self.has_candles and not to_price else price) + 0.015 * span
            turn_y = max(end_y, self.price_hi + 0.012 * span)
            start_y = ly - 0.004 * span
        # a VERTICAL line under the candle it names (#116), then across the empty lane to the label
        self.ax.plot([x, x, lx], [end_y, turn_y, start_y], color=colour, lw=1.0, ls=(0, (2, 2)), alpha=0.8,
                     zorder=3)

    def _line_touches(self, placer: "_Placer", it: Line, k: int) -> None:
        """Numbers 1, 2, 3 at the touches, on the side of the line where price is NOT, clear of the candles."""
        xa, pa, slope, x0, x1 = self._line_xy(it)
        renderer = self.fig.canvas.get_renderer()
        numbered = 0
        for n_, tt in enumerate(it.touches, start=1):
            xt = self.x(tt)
            if not (0 <= xt <= self.n - 1):
                continue
            yt = pa + slope * (xt - xa)
            above = (not self.has_candles) or yt >= self.c[int(round(xt))]
            offsets = (9, 14, 20, 28, 38, 50)
            for off in offsets:
                t = self.ax.annotate(str(n_), xy=(xt, yt), xytext=(0, off if above else -off),
                                     textcoords="offset points", ha="center", va="bottom" if above else "top",
                                     fontsize=F_NAME + 1, fontweight="bold", color=it.color, zorder=9)
                if placer.free(_extent(t, renderer)):
                    break
                if off != offsets[-1]:            # no room anywhere: the last try stays, and lint will say so
                    t.remove()
            else:
                self.crowded.append(f"touch {n_}")
            self.labels.append((t, "touch", k))
            placer.add(_extent(t, renderer))
            numbered += 1
        self.item_records[k].info["numbered"] = numbered
        # the name: in the lane on the empty side, a vertical leader down (or up) to the line's END
        xe = float(np.clip(np.floor(x1), 0, self.n - 1))
        ye = pa + slope * (xe - xa)
        above = (not self.has_candles) or ye >= self.c[int(xe)]
        self._callout(placer, it.name, xe, ye, it.color, k, "name", below=not above, to_price=True)

    # ---- the record ------------------------------------------------------------------------------------------------
    def record(self, drawn: Drawn, renderer) -> None:
        ax, p = self.ax, self.p
        tr = ax.transData.transform
        bb = ax.get_window_extent(renderer)
        candles = []
        if self.has_candles:
            cw = next(it for it in p.items if isinstance(it, Candles)).width
            xs = np.arange(self.n)
            a = tr(np.column_stack([xs - cw / 2, self.l]))
            b = tr(np.column_stack([xs + cw / 2, self.h]))
            candles = [Rect(*a[j], *b[j]) for j in range(self.n)]
        drawn.panels.append(PanelOnPicture(
            index=self.index, tf=p.tf, role=p.role, title=self.title_artist.get_text(), clock=p.clock,
            first_bar=self.idx[0], last_bar=self.idx[-1], n_bars=self.n,
            area=Rect(bb.x0, bb.y0, bb.x1, bb.y1), candles=candles,
            bar_minutes=(self.idx[1:] - self.idx[:-1]).median().total_seconds() / 60 if self.n > 1 else None,
            entry_at=None if p.entry_at is None else _ts(p.entry_at)))
        base = len(drawn.items)
        for rec in self.item_records:
            it = rec.info.pop("item")
            rec.info["times"] = _all_times(it)
            rec.paths = [_sample(tr(np.asarray(path, float))) for path in rec.paths]
            rec.boxes = [Rect(*tr((x0, y0)), *tr((x1, y1))) for x0, y0, x1, y1 in rec.boxes]
            if isinstance(it, Line) and it.group:
                rec.info["inside_share"] = None   # filled below, once both lines of the group are known
            drawn.items.append(rec)
        self._triangles(drawn, base)
        drawn.texts.append(TextOnPicture(self.title_artist.get_text(), "panel_title", self.index,
                                         _extent(self.title_artist, renderer)))
        shown = [(lab, loc, ax.get_xlim()) for lab, loc in zip(ax.get_xticklabels(), ax.get_xticks())]
        if p.price_axis:
            shown += [(lab, loc, ax.get_ylim()) for lab, loc in zip(ax.get_yticklabels(), ax.get_yticks())]
        for lab, loc, (lo, hi) in shown:          # only the ticks inside the chart are drawn
            if lab.get_text() and lab.get_visible() and lo <= loc <= hi:
                drawn.texts.append(TextOnPicture(lab.get_text(), "tick", self.index, _extent(lab, renderer)))
        for t, role, k in self.labels:
            drawn.texts.append(TextOnPicture(t.get_text(), role, self.index, _extent(t, renderer), base + k))
        drawn.crowded += [f"panel {self.index + 1}: {c}" for c in self.crowded]

    def _triangles(self, drawn: Drawn, base: int) -> None:
        """For two Lines in one group: the share of candle closes between them, where both exist (#116)."""
        groups: dict[str, list[tuple[int, Line]]] = {}
        for k, it in enumerate([it for it in self.p.items if not isinstance(it, Candles)]):
            if isinstance(it, Line) and it.group:
                groups.setdefault(it.group, []).append((k, it))
        for members in groups.values():
            share = 0.0
            if len(members) == 2:
                (ka, a), (kb, b) = members
                xa0, pa, sa, a0, a1 = self._line_xy(a)
                xb0, pb, sb, b0, b1 = self._line_xy(b)
                xs = np.arange(int(np.ceil(max(a0, b0, 0))), int(np.floor(min(a1, b1, self.n - 1))) + 1)
                if len(xs):
                    la, lb = pa + sa * (xs - xa0), pb + sb * (xs - xb0)
                    inside = (self.c[xs] >= np.minimum(la, lb)) & (self.c[xs] <= np.maximum(la, lb))
                    share = float(inside.mean())
            for k, _ in members:
                drawn.items[base + k].info["inside_share"] = share


class _Placer:
    """Finds empty room for a label: lane by lane, sliding left and right, until it touches nothing - no candle,
    no gap box, no line, no other word (#115, #116). Copied from old/build/arrow_image.py and made two-sided."""

    def __init__(self, run: _Run):
        self.run = run
        self.ax = run.ax
        self.renderer = run.fig.canvas.get_renderer()
        tr = self.ax.transData.transform
        self.rects: list[Rect] = []
        self.points = np.zeros((0, 2))
        if run.has_candles:
            cw = next(it for it in run.p.items if isinstance(it, Candles)).width
            xs = np.arange(run.n)
            a = tr(np.column_stack([xs - cw / 2, run.l]))
            b = tr(np.column_stack([xs + cw / 2, run.h]))
            self.rects += [Rect(*a[j], *b[j]) for j in range(run.n)]
        for rec in run.item_records:
            if rec.kind == "Box":
                self.rects += [Rect(*tr((x0, y0)), *tr((x1, y1))) for x0, y0, x1, y1 in rec.boxes]
            if rec.kind in ("Level", "Line", "Fib", "Position", "Arrow", "Moment"):
                for path in rec.paths:
                    self.points = np.vstack([self.points, _sample(tr(np.asarray(path, float)))])
        for t, _role, _k in run.labels:
            self.rects.append(_extent(t, self.renderer))
        bb = self.ax.get_window_extent(self.renderer)
        self.area = Rect(bb.x0, bb.y0, bb.x1, bb.y1)

    def add(self, r: Rect) -> None:
        self.rects.append(r)

    def free(self, r: Rect) -> bool:
        if any(r.overlaps(q) for q in self.rects):
            return False
        if len(self.points):
            px, py = self.points[:, 0], self.points[:, 1]
            if np.any((px > r.x0) & (px < r.x1) & (py > r.y0) & (py < r.y1)):
                return False
        return True

    def place(self, t, x: float, lanes: list[tuple[float, str]]) -> bool:
        r = _extent(t, self.renderer)
        w_data = (r.x1 - r.x0) / (self.area.x1 - self.area.x0) * (self.run.xl1 - self.run.xl0)
        lo, hi = self.run.xl0 + 0.55 * w_data, self.run.xl1 - 0.55 * w_data
        steps = [0.0] + [s * k * 0.35 * w_data for k in range(1, 40) for s in (1, -1)]
        for y, va in lanes:
            t.set_va(va)
            for dx in steps:
                t.set_position((float(np.clip(x + dx, lo, hi)), y))
                r = _extent(t, self.renderer)
                if self.free(r):
                    self.add(r)
                    return True
        if lanes:                                 # nowhere free: leave it in the first lane; lint will say so
            t.set_va(lanes[0][1])
            t.set_position((float(np.clip(x, lo, hi)), lanes[0][0]))
        self.add(_extent(t, self.renderer))
        return False


def _sample(pts: np.ndarray, step: float = 3.0) -> np.ndarray:
    """A polyline as points every `step` pixels (to test it against the words)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.vstack(out)


def _prices(it) -> list[float]:
    if isinstance(it, Level | Position):
        return [it.price]
    if isinstance(it, Box | Zone):
        return [it.lo, it.hi]
    if isinstance(it, Line):
        return [float(it.start[1]), float(it.end[1])]
    if isinstance(it, Fib):
        return [it.price(lv) for lv in it.levels]
    if isinstance(it, ADRBand):
        return [*it.upper, *it.lower]
    if isinstance(it, Arrow):
        return [float(p) for _, p in it.points]
    if isinstance(it, Marker):
        return [it.price]
    return []


def _text_px(ax, s: str, size: float, renderer) -> float:
    t = ax.text(0, 0, s, fontsize=size, fontweight="bold")
    w = t.get_window_extent(renderer).width
    t.remove()
    return w


def _clock_times(idx: pd.DatetimeIndex, which: str) -> pd.DatetimeIndex:
    if which == "ist":
        return clock.to_ist(idx)
    if which == "utc":
        return clock.server_to_utc(idx)
    return idx


# =================================================================================================================
# the four pictures
# =================================================================================================================

@dataclass(frozen=True)
class Point:
    """One moment of a trade, in his shorthand: B1 in, S1 in, B1 C out (#121)."""
    time: Time
    price: float
    tag: str
    side: Side
    act: Literal["in", "out"]
    word: str = ""


def points_from_trade(trade, word_in: str = "gap") -> tuple[list[Point], list[Position]]:
    """A tester Trade (machine/tester) as his points: B1 / S1 where it filled, "B1 C" at each close."""
    side = trade.direction
    tag = "B1" if side == "buy" else "S1"
    pts = [Point(trade.fill_ts, trade.fill_px, tag, side, "in", word_in)]
    for e in trade.exits:
        pts.append(Point(e.ts, e.px, f"{tag} C", side, "out", {"past at fill": "at once"}.get(e.reason, e.reason)))
    end = trade.exits[-1].ts if trade.exits else trade.fill_ts
    return pts, [Position(trade.fill_ts, end, trade.fill_px, side)]


def money_words(trade) -> tuple[str, str, str]:
    """The result in rupees (after the Rs 40 a position, all-in) and in gold dollars, with its settings (#54, #115)."""
    from machine import costs
    net = trade.net_inr
    moved = sum((e.px - trade.fill_px) * (1 if trade.direction == "buy" else -1) * e.lots / 0.01
                for e in trade.exits) / (trade.lots / 0.01)
    return (costs.inr(net), GREEN if net >= 0 else RED,
            f"gold moved {moved:+.2f} dollars · {trade.lots:.2f} lot · after the ₹40 cost a position")


def _all_times(it) -> list[pd.Timestamp]:
    """Every time an item reaches (lesson B16: nothing on a question picture may reach past the entry candle)."""
    ts: list = []
    for name in ("start", "end", "time", "extend_to"):
        v = getattr(it, name, None)
        if isinstance(v, tuple):
            v = v[0]
        if v is not None:
            ts.append(_ts(v))
    ts += [_ts(t) for t in getattr(it, "touches", ())]
    ts += [_ts(t) for t, _ in getattr(it, "points", ())]
    return ts


def _times(items: Sequence[Item]) -> list[pd.Timestamp]:
    """The earliest time each item needs on the chart (a gap's first candle, a line's first touch ...)."""
    out = []
    for it in items:
        if isinstance(it, Box | Position | Span | ADRBand):
            out.append(_ts(it.start))
        elif isinstance(it, Line | Fib):
            out.append(_ts(it.start[0]))
        elif isinstance(it, Level | Zone) and it.start is not None:
            out.append(_ts(it.start))
        elif isinstance(it, Marker | Moment):
            out.append(_ts(it.time))
    return out


def trade_picture(trade, m5: pd.DataFrame, h1: pd.DataFrame, *, subject: str, sub: str, word_in: str = "gap",
                  items_5m: Sequence[Item] = (), items_1h: Sequence[Item] = (), before: int = 30, after: int = 14,
                  hour_bars: int = 36) -> Picture:
    """A short trade, the way A01 was drawn (#120): a SMALL 1-hour picture on top (#110) with a grey band where the
    5-minute chart sits, then the 5-minute chart with the gap box, the lines, B1 / S1 and "C" where it closed."""
    pts, positions = points_from_trade(trade, word_in)
    t_first = min([_ts(trade.fill_ts)] + _times(items_5m))
    i0 = max(0, int(m5.index.searchsorted(t_first)) - before)
    i1 = min(len(m5) - 1, int(m5.index.searchsorted(_ts(pts[-1].time))) + after)
    m5w = m5.iloc[i0:i1 + 1]
    j1 = int(h1.index.searchsorted(m5w.index[-1], side="right")) + 3
    h1w = h1.iloc[max(0, j1 - hour_bars):min(len(h1), j1)]
    main = [Candles(), *items_5m, *positions]
    main += [Marker(p.time, p.price, p.tag, p.side, p.word) for p in pts]
    hour = [Candles(), Span(m5w.index[0], m5w.index[-1], "the 5-minute chart below"), *items_1h]
    hour += [Marker(p.time, p.price, p.tag, p.side) for p in pts]      # his tag only: no mark without a word
    side_word = "BUY GOLD" if trade.direction == "buy" else "SELL GOLD"
    result, rcol, moved = money_words(trade)
    tag = pts[0].tag
    return Picture(
        kind="trade", subject=subject, heading=side_word, heading_color=side_colour(trade.direction),
        result=result, result_color=rcol, sub=sub,
        rows=[[Panel(h1w, "1h", hour, "the same hours", role="hour")],
              [Panel(m5w, "5m", main, "the trade", role="main")]],
        footer_left=f"{tag} = your 1st {trade.direction} · C = closed",
        footer_right=moved)


def whole_trade_picture(trade, m5: pd.DataFrame, h1: pd.DataFrame, *, subject: str, sub: str, word_in: str = "gap",
                        items_1h: Sequence[Item] = (), point_items: dict | None = None, before: int = 30,
                        after: int = 12, min_hour_bars: int = 60) -> Picture:
    """A long trade (#117, #123): THE WHOLE TRADE on the 1-hour chart on top, then one 5-minute panel for EVERY entry
    and EVERY exit, three to a row. point_items: {point tag + act ("B1 in", "B1 C out"): [items]}."""
    pts, positions = points_from_trade(trade, word_in)
    a = int(h1.index.searchsorted(_ts(trade.fill_ts), side="right")) - 1 - before
    b = int(h1.index.searchsorted(_ts(pts[-1].time), side="right")) - 1 + after
    if b - a < min_hour_bars:                     # a short trade on the hour chart is two candles: widen it,
        a = b - min_hour_bars                     # backwards first (what happened BEFORE matters more)
    h1w = h1.iloc[max(0, a):min(len(h1), b + 1)]
    top = [Candles(), *items_1h, *positions]
    top += [Marker(p.time, p.price, p.tag, p.side, p.word) for p in pts]
    rows: list[list[Panel]] = [[Panel(h1w, "1h", top, "THE WHOLE TRADE", role="main")]]
    small: list[Panel] = []
    for p in pts:
        key = f"{p.tag} {p.act}"
        extra = list((point_items or {}).get(key, ()))
        i = int(m5.index.searchsorted(_ts(p.time), side="right")) - 1
        starts = [_ts(it.start) for it in extra if isinstance(it, Box)]
        i0 = i - 26
        if starts:                                # reach back to the gap's first candle (old one_image.py, T11)
            i0 = min(i0, int(m5.index.searchsorted(min(starts))) - 3)
        w = m5.iloc[max(0, i0):min(len(m5), i + 11)]
        its = [Candles(), *extra, Marker(p.time, p.price, p.tag, p.side, p.word, label=False)]
        title = f"{p.tag} {p.act}" + (f" · {p.word}" if p.word else "")
        small.append(Panel(w, "5m", its, title, role="point", price_axis=False))
    for k in range(0, len(small), 3):
        rows.append(small[k:k + 3])
    side_word = "BUY GOLD" if trade.direction == "buy" else "SELL GOLD"
    result, rcol, moved = money_words(trade)
    return Picture(
        kind="whole_trade", subject=subject, heading=side_word, heading_color=side_colour(trade.direction),
        result=result, result_color=rcol, sub=sub, rows=rows,
        footer_left=f"{pts[0].tag} = your 1st {trade.direction} · C = closed · small charts are 5-minute",
        footer_right=moved)


def price_path(bars: pd.DataFrame, min_move: float) -> list[tuple[pd.Timestamp, float]]:
    """The price as a person draws it: turn, turn, turn - a swing kept only when price moved `min_move` against the
    last one (old/build/arrow_image.py zigzag)."""
    h, l, c, idx = bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float), bars.index
    pts = [(0, float(c[0]))]
    direction, ext_i, ext_p = 0, 0, float(c[0])
    for i in range(1, len(bars)):
        if direction >= 0 and h[i] > ext_p:
            ext_i, ext_p, direction = i, float(h[i]), 1
        elif direction <= 0 and l[i] < ext_p:
            ext_i, ext_p, direction = i, float(l[i]), -1
        elif direction == 1 and ext_p - l[i] >= min_move:
            pts.append((ext_i, ext_p))
            ext_i, ext_p, direction = i, float(l[i]), -1
        elif direction == -1 and h[i] - ext_p >= min_move:
            pts.append((ext_i, ext_p))
            ext_i, ext_p, direction = i, float(h[i]), 1
    pts += [(ext_i, ext_p), (len(bars) - 1, float(c[-1]))]
    out: list[tuple[int, float]] = []
    for p in pts:
        if not out or p[0] > out[-1][0]:
            out.append(p)
    return [(idx[i], px) for i, px in out]


def arrows_picture(trade, h1: pd.DataFrame, *, subject: str, sub: str, min_move: float, word_in: str = "gap",
                   items: Sequence[Item] = (), before: int = 30, after: int = 12, min_hour_bars: int = 60) -> Picture:
    """The same trade in his ARROWS, no candles (#122) - the pair of whole_trade_picture (#123: "both images")."""
    pts, positions = points_from_trade(trade, word_in)
    a = int(h1.index.searchsorted(_ts(trade.fill_ts), side="right")) - 1 - before
    b = int(h1.index.searchsorted(_ts(pts[-1].time), side="right")) - 1 + after
    if b - a < min_hour_bars:
        a = b - min_hour_bars
    h1w = h1.iloc[max(0, a):min(len(h1), b + 1)]
    path = price_path(h1w, min_move)
    for p in pts:                                 # every moment sits ON the path, so no label points at nothing
        path.append((h1w.index[int(h1w.index.searchsorted(_ts(p.time), side="right")) - 1], p.price))
    path = sorted(dict((t, (t, px)) for t, px in path).values())
    its = [Arrow(tuple(path)), *items, *positions]
    its += [Marker(p.time, p.price, p.tag, p.side, p.word) for p in pts]
    side_word = "BUY GOLD" if trade.direction == "buy" else "SELL GOLD"
    result, rcol, moved = money_words(trade)
    return Picture(
        kind="arrows", subject=subject, heading=side_word, heading_color=side_colour(trade.direction),
        result=result, result_color=rcol, sub=sub,
        rows=[[Panel(h1w, "1h", its, "THE WHOLE TRADE in arrows (no candles)", role="main")]],
        footer_left=f"blue arrows = price · {pts[0].tag} = your 1st {trade.direction} · C = closed",
        footer_right=moved)


def question_picture(m5: pd.DataFrame, h1: pd.DataFrame, *, entry_at: Time, question: Question, subject: str,
                     heading: str, sub: str, items_5m: Sequence[Item] = (), items_1h: Sequence[Item] = (),
                     bars_5m: int = 70, hour_bars: int = 36, footer_left: str = "", footer_right: str = "") -> Picture:
    """A question (#132): the question ON the picture, and the charts END at the entry candle (lesson B16) - the
    hour chart shows only hours that had CLOSED by then. No money anywhere (B16)."""
    entry = _ts(entry_at)
    m5w = m5.loc[:entry].iloc[-bars_5m:]
    if m5w.empty or m5w.index[-1] != entry:
        raise ValueError(f"{entry} is not a 5-minute candle in the bars given")
    closed_by = entry + pd.Timedelta(minutes=5)
    h1c = h1[h1.index + pd.Timedelta(hours=1) <= closed_by]
    h1w = h1c.iloc[-hour_bars:]
    hour = [Candles(), Span(max(m5w.index[0], h1w.index[0]), h1w.index[-1], "the 5-minute chart below"),
            *items_1h]
    return Picture(
        kind="question", subject=subject, heading=heading, sub=sub, question=question,
        rows=[[Panel(h1w, "1h", hour, "before it", role="hour")],
              [Panel(m5w, "5m", [Candles(), *items_5m], "it ends at the candle the question is about", role="main")]],
        footer_left=footer_left, footer_right=footer_right, entry_at=entry, entry_tf="5m")
