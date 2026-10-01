"""The label file: data/labels/<word>.parquet (L15.1) - one row per label, with its picture and its times.

    .venv\\Scripts\\python -m machine.labels build            (re)write it from what is in git + the price data

Rows come from two places, and git holds everything needed to rebuild the file on any PC with the price data:
  "his drawing"   machine/labels/drawings.py LABELS - his answers on the zone cards in strategy/pictures/
  "telegram"      plan/work/labels/<word>/charts.jsonl (every chart sent) + replies.jsonl (his yes / no / not sure; the
                  newest reply to a chart wins, every reply stays in the file)

Columns (times are MT5 server time unless named _utc; "known" = the moment the chart ends, when the band is finished):
  label_id, word, source, label, his_words, lessons, from_picture, first_candle, last_candle, known_utc, year, years,
  window, lo, hi, calm_on_chart, zone_calm, zone_tier, rule_says, step, step_words, step_size, step_wanted, asked,
  answered, replies, image (JPEG bytes; None for the hidden years), image_sha256 (what the picture shows), drawn_by
  (the code that drew it), note.
  years = "tune" (2016-2020) or "hidden" (2021-2023: his words are kept, the candles are never drawn or read).
  rule_says = what his own rule says (rules.py: his main zone = calm at most 0.74), measured on the chart - the
  comparison L15.2 needs ("does the picture model see more than the rule?").

The pictures are drawn once into data/labels/<word>/ (his/<key>.jpg; charts/<qid>.jpg are the ones sent) and drawn
again only when missing or when the drawing code changed (lesson B19). The file is written to a temporary name and
then moved into place (lesson B10).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, rules
from machine.data import clock
from machine.labels import drawings as DW
from machine.labels import picture as P
from machine.labels import replies as LR
from machine.labels import words as W

COLUMNS = ("label_id", "word", "source", "label", "his_words", "lessons", "from_picture", "first_candle",
           "last_candle", "known_utc", "year", "years", "window", "lo", "hi", "calm_on_chart", "zone_calm", "zone_tier",
           "rule_says", "step", "step_words", "step_size", "step_wanted", "asked", "answered", "replies", "image",
           "image_sha256", "drawn_by", "note")


def parquet_path(word: str, folder: Path | None = None) -> Path:
    return Path(folder or config.LABELS_DIR) / f"{word}.parquet"


def rule_says(calm: float) -> str:
    """His rule on one number: his main zone = calm at most 0.74 (rules.py what_a_strong_zone_is)."""
    if not np.isfinite(calm):
        return "no band candles"
    return "yes" if calm <= float(rules.get("what_a_strong_zone_is")["calm_max"]) else "no"


def build(word: str = W.STRONG_ZONE.slug, *, m5: pd.DataFrame | None = None, base: Path | None = None,
          folder: Path | None = None) -> pd.DataFrame:
    """Every label of `word` as one table (and its pictures, drawn when missing). m5: the 5-minute bars 2016-2020
    (loaded when None)."""
    w = W.get(word)
    folder = Path(folder or config.LABELS_DIR)
    pics = folder / word
    if m5 is None:
        from machine.data.bars import load_bars
        m5 = load_bars("5m", str(min(config.TUNE_YEARS)), str(max(config.TUNE_YEARS)))
    fp = P.code_fingerprint()
    rows = []
    for x in DW.labels(word):
        first, last = x.span()
        row = dict(label_id=f"his-{x.key}", word=word, source="his drawing", label=x.label, his_words=x.his_words,
                   lessons=", ".join(x.lessons), from_picture=f"strategy/pictures/{x.picture} panel {x.panel}"
                   + (f" (also {x.same_as})" if x.same_as else ""), first_candle=first, last_candle=last,
                   known_utc=pd.Timestamp(x.last_utc) + pd.Timedelta(minutes=5), year=int(first.year),
                   years="hidden" if x.hidden else "tune", window="his card", lo=x.lo, hi=x.hi, zone_calm=np.nan,
                   zone_tier=np.nan, step=np.nan, step_words="", step_size=np.nan, step_wanted=np.nan, asked=pd.NaT,
                   answered=pd.NaT, replies=1, note=x.note)
        if x.hidden:
            row.update(calm_on_chart=np.nan, rule_says="not drawn (hidden years)", image=None, image_sha256="",
                       drawn_by="")
        else:
            h = P.hour_candles(m5, first, last)
            c = P.calm(h, x.lo, x.hi)
            path = pics / "his" / f"{x.key}.jpg"
            img, sha = _picture(w, h, x.lo, x.hi, path, f"label {word} his {x.key}", fp)
            row.update(calm_on_chart=c, rule_says=rule_says(c), image=img, image_sha256=sha, drawn_by=fp)
        rows.append(row)
    answers = LR.latest(word, base)
    for ch in LR.read_charts(word, base):
        rep = answers.get(ch["qid"])
        if rep is None:
            continue                                   # asked, not answered yet: not a label
        first, last = pd.Timestamp(ch["first"]), pd.Timestamp(ch["last"])
        path = config.ROOT / ch["picture"] if not Path(ch["picture"]).is_absolute() else Path(ch["picture"])
        if path.is_file() and ch.get("drawn_by") == fp:
            img, sha = path.read_bytes(), ch["pixel_sha256"]
        else:                                          # not on this PC, or the drawing code changed: draw it again
            h = P.hour_candles(m5, first, last)
            img, sha = _picture(w, h, ch["lo"], ch["hi"], pics / "redrawn" / f"{ch['qid']}.jpg",
                                f"label {word} redrawn {ch['qid']}", fp)
        rows.append(dict(label_id=f"tg-{ch['qid']}", word=word, source="telegram", label=rep["label"],
                         his_words=rep["words"], lessons="", from_picture=ch["picture"], first_candle=first,
                         last_candle=last, known_utc=_utc(pd.Timestamp(ch["known_at"])), year=int(first.year),
                         years="tune", window=ch["window"], lo=ch["lo"], hi=ch["hi"],
                         calm_on_chart=ch["calm_on_chart"], zone_calm=ch["calm"], zone_tier=ch["tier"],
                         rule_says=rule_says(ch["calm_on_chart"]), step=ch["step"], step_words=ch["step_words"],
                         step_size=ch["step_size"], step_wanted=ch["step_wanted"], asked=pd.Timestamp(ch["queued"]),
                         answered=pd.Timestamp(rep["at"]).tz_localize(None), replies=rep["replies"], image=img,
                         image_sha256=sha, drawn_by=fp, note=f"Telegram message {rep['message_id']}"))
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    check(df)
    return df


def write(df: pd.DataFrame, word: str, folder: Path | None = None) -> Path:
    path = parquet_path(word, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def check(df: pd.DataFrame) -> None:
    """What must be true of a label file: known answers only, every tune-years row has its picture, no hidden-years
    row has one, nothing from the sealed years, one row per label."""
    bad = []
    if not df["label"].isin(W.LABELS).all():
        bad.append(f"labels other than {W.LABELS}: {sorted(set(df['label']) - set(W.LABELS))}")
    tune = df["years"] == "tune"
    if df.loc[tune, "image"].isna().any():
        bad.append("a tune-years label has no picture")
    if df.loc[~tune, "image"].notna().any():
        bad.append("a hidden-years label has a picture (those candles are never drawn)")
    if (pd.to_datetime(df["last_candle"]) >= pd.Timestamp(config.SEALED_FROM)).any():
        bad.append("a label from the sealed years")
    if (tune & (df["year"] > max(config.TUNE_YEARS))).any():
        bad.append("a 'tune' label after the tune years")
    if df["label_id"].duplicated().any():
        bad.append("a label twice")
    if bad:
        raise ValueError("the label file is wrong: " + "; ".join(bad))


def summary(df: pd.DataFrame) -> str:
    """Counts in plain words (for the session's reply and the notes)."""
    if df.empty:
        return "no labels yet"
    lines = []
    for src, g in df.groupby("source", sort=False):
        by = g["label"].value_counts()
        drawn = int(g["image"].notna().sum())
        lines.append(f"{src}: {len(g)} labels - " + ", ".join(f"{k} {by.get(k, 0)}" for k in W.LABELS)
                     + f"; {drawn} with a picture, {len(g) - drawn} not drawn (hidden years)")
    return "\n".join(lines)


def _picture(w: W.Word, h: pd.DataFrame, lo: float, hi: float, path: Path, subject: str,
             fp: str) -> tuple[bytes, str]:
    stamp = path.with_suffix(".drawn_by")
    if path.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == fp:
        from machine.charts.draw import pixel_sha256
        return path.read_bytes(), pixel_sha256(path)
    drawn = P.draw(w, h, lo, hi, path, subject=subject)
    stamp.write_text(fp + "\n", encoding="utf-8")
    return path.read_bytes(), drawn.pixel_sha256


def _utc(server: pd.Timestamp) -> pd.Timestamp:
    return clock.server_to_utc(server).tz_localize(None)
