"""The demo as a recording (L11): `python -m machine.demo` drawn as an animated terminal, about 2 minutes.

    .venv\\Scripts\\python -m machine.demo.record docs/demo.gif

It runs the real demo (machine/demo/run.py) in this process, keeps every line it prints with the real time it
appeared, and draws a terminal frame for each moment with Pillow (no screen recorder needed). Shown at reading pace:
a pause after each step; a long wait for the computer (the referee's exam) is shortened, and the frame says so and
shows the real time it took. The text is the run's own, not edited.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 112, 36             # characters a line, text lines on screen
FONT_PX = 15
PAD = 14
TARGET_SECONDS = 120             # about 2 minutes in all
LINE_SECONDS = 0.12              # each new line shows for at least this long (a terminal printing)
WAIT_SHOWN = 1.6                 # a long wait for the computer is shown this long, with its real length written
BG, FG, DIM = (13, 17, 23), (201, 209, 217), (125, 133, 144)
HEAD, GOOD, BAD, NOTE = (88, 166, 255), (126, 231, 135), (255, 123, 114), (210, 168, 255)


@dataclass
class Line:
    t: float
    text: str
    msg: int                     # the message (one say() call) it came from: one message appears at once


def capture(full: bool = False) -> tuple[list[Line], float]:
    from machine.demo.run import main_steps
    lines: list[Line] = []
    t0 = time.time()
    count = [0]

    def say(text: str = "") -> None:
        count[0] += 1
        for part in str(text).split("\n"):
            lines.append(Line(time.time() - t0, part, count[0]))
        print(text, flush=True)
    main_steps(full=full, say=say)
    return lines, time.time() - t0


def _font() -> ImageFont.FreeTypeFont:
    import matplotlib
    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(path), FONT_PX)


def _color(text: str) -> tuple[int, int, int]:
    if text.startswith("===="):
        return HEAD
    if any(w in text for w in ("REFUSED", "CAUGHT", "!!")):
        return BAD
    if "NOT FOUND" in text:
        return FG
    if any(w in text for w in ("-> FOUND", "passes", "'works' 0 of", "FORM DRAFTED")):
        return GOOD
    if text.startswith("(") or "stand-in" in text:
        return NOTE
    return FG


def _wrap(text: str) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=COLS, subsequent_indent="      ", break_long_words=True,
                         replace_whitespace=False, drop_whitespace=False) or [""]


def frames(lines: list[Line], real_total: float):
    """[screen lines, status text, seconds shown, reading weight] per frame: one frame per printed line (lines printed
    together, e.g. one multi-line message, appear together). The last frame of a step carries a reading weight = the
    lines the step printed; scale() turns the weights into pauses."""
    out, screen, i, since = [], [], 0, 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j].msg == lines[i].msg:
            j += 1
        for ln in lines[i:j]:
            wrapped = _wrap(ln.text)
            screen += wrapped
            since += len(wrapped)
        gap = (lines[j].t - lines[j - 1].t) if j < len(lines) else 0.0
        step_end = j >= len(lines) or lines[j].text.startswith("====")
        out.append([list(screen[-ROWS:]), "", LINE_SECONDS, since if step_end else 0])
        if step_end:
            since = 0
        if gap > 4.0:                                     # the computer is working: say so, briefly
            out.append([list(screen[-ROWS:]), f"... working - {gap:.0f} s of real time, shortened here",
                        WAIT_SHOWN, 0])
        i = j
    return out


def scale(timeline) -> None:
    """Spread the time left (to TARGET_SECONDS) over the pauses at the end of each step, more for a longer step:
    time to read it."""
    fixed = sum(f[2] for f in timeline)
    weight = sum(f[3] for f in timeline) or 1
    left = max(0.0, TARGET_SECONDS - fixed)
    for f in timeline:
        f[2] += left * f[3] / weight


def draw(timeline, path: Path, title: str) -> dict:
    font = _font()
    cw = font.getbbox("M")[2]
    lh = int(FONT_PX * 1.35)
    w, h = PAD * 2 + cw * COLS, PAD * 2 + lh * (ROWS + 3)          # + the title row and the status row
    palette = Image.new("P", (1, 1))
    palette.putpalette([c for rgb in (BG, FG, DIM, HEAD, GOOD, BAD, NOTE) for c in rgb] + [0, 0, 0] * 249)
    images, durations, shown = [], [], 0.0
    for screen, status, seconds, _ in timeline:
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        clock = f"{int(shown // 60)}:{int(shown % 60):02d}"
        d.text((PAD, PAD), title, font=font, fill=DIM)
        d.text((w - PAD - cw * len(clock), PAD), clock, font=font, fill=DIM)
        for k, text in enumerate(screen):
            d.text((PAD, PAD + lh * (k + 1.5)), text, font=font, fill=_color(text.lstrip()))
        if status:
            d.text((PAD, h - PAD - lh), status, font=font, fill=NOTE)
        images.append(img.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(int(round(seconds * 1000)))
        shown += seconds
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True,
                   disposal=1)
    return {"frames": len(images), "seconds": round(shown), "bytes": os.path.getsize(path), "size": (w, h)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.demo.record", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the GIF to write, e.g. docs/demo.gif")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    lines, real = capture()
    timeline = frames(lines, real)
    scale(timeline)
    info = draw(timeline, a.out, f"idea-machine · python -m machine.demo · made-up data · real run {real:.0f} s")
    print(f"\n{a.out}: {info['frames']} frames, {info['seconds']} s shown, {info['bytes'] / 1e6:.1f} MB, "
          f"{info['size'][0]}x{info['size'][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
