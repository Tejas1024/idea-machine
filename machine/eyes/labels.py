"""The answers the two models learn (L15.2): the label file of L15.1, only the labels with a picture.

    from machine.eyes import labels
    lab = labels.load()            # one row per answer with a picture, oldest chart first; y = 1 for his yes

Left out, and said so: his 2 answers in the hidden years 2021-2023 (Q79 A, C - never drawn, those years stay closed)
and any "not sure" (neither yes nor no). Refused: a chart outside 2016-2020, a label twice, a label without its JPEG.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config
from machine.labels import store

KEEP = ("label_id", "source", "label", "his_words", "first_candle", "last_candle", "known_utc", "year", "window", "lo",
        "hi", "calm_on_chart", "rule_says", "step", "step_size", "image", "image_sha256")


class LabelsError(ValueError):
    pass


def load(word: str | None = None, folder: Path | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The answers with a picture, oldest chart first (ties: by label id). df: a label table already in memory."""
    word = word or config.EYES_WORD
    if df is None:
        path = store.parquet_path(word, folder)
        if not path.is_file():
            raise LabelsError(f"{path} is missing - run: python -m machine.labels build")
        df = pd.read_parquet(path)
    out = df[df["image"].notna() & df["label"].isin(("yes", "no"))].copy()
    out = out[[c for c in KEEP if c in out.columns]]
    years = set(int(y) for y in out["year"])
    bad = sorted(years - set(config.TUNE_YEARS))
    if bad:
        raise LabelsError(f"labels from {bad}: the models learn from 2016-2020 only (the hidden years stay closed)")
    if (pd.to_datetime(out["last_candle"]) >= pd.Timestamp(config.SEALED_FROM)).any():
        raise LabelsError("a label from the sealed years")
    if out["label_id"].duplicated().any():
        raise LabelsError("a label twice")
    out["first_candle"] = pd.to_datetime(out["first_candle"])
    out["last_candle"] = pd.to_datetime(out["last_candle"])
    out["y"] = (out["label"] == "yes").astype(int)
    out["rule_yes"] = (out["rule_says"] == "yes").astype(int)      # his rule as the label file recorded it (L15.1)
    out = out.sort_values(["first_candle", "label_id"], kind="mergesort").reset_index(drop=True)
    return out


def left_out(word: str | None = None, folder: Path | None = None, df: pd.DataFrame | None = None) -> dict:
    """What load() leaves out, counted (for the report)."""
    word = word or config.EYES_WORD
    if df is None:
        df = pd.read_parquet(store.parquet_path(word, folder))
    no_pic = df[df["image"].isna()]
    unsure = df[df["image"].notna() & ~df["label"].isin(("yes", "no"))]
    return {"all_labels": int(len(df)), "no_picture": int(len(no_pic)),
            "no_picture_ids": sorted(no_pic["label_id"].tolist()), "not_sure": int(len(unsure))}


def fingerprint(lab: pd.DataFrame) -> str:
    """The answers and the pictures they are about: the same answers on the same pictures = the same fingerprint."""
    h = hashlib.sha256()
    for r in lab[["label_id", "label", "image_sha256"]].itertuples(index=False):
        h.update(f"{r.label_id}|{r.label}|{r.image_sha256}\n".encode("utf-8"))
    return h.hexdigest()


def counts(lab: pd.DataFrame) -> pd.DataFrame:
    """yes / no per year (first candle)."""
    t = pd.crosstab(lab["year"], lab["label"]).reindex(columns=["yes", "no"], fill_value=0)
    t["all"] = t.sum(axis=1)
    return t


def made_up(lab: pd.DataFrame, how: str, seed: int = 0) -> np.ndarray:
    """Made-up answers for the exams (never his):
    noise    his yes / no counts of each year shuffled among that year's charts - no link to the pictures at all
    rule     his written rule as the answer: yes when calm is at most 0.74 (a thing the pictures do show)"""
    if how == "noise":
        rng = np.random.default_rng(seed)
        y = lab["y"].to_numpy().copy()
        for yr in sorted(lab["year"].unique()):
            idx = np.flatnonzero(lab["year"].to_numpy() == yr)
            y[idx] = rng.permutation(y[idx])
        return y
    if how == "rule":
        return lab["rule_yes"].to_numpy().copy()
    raise ValueError(f"no made-up answers called {how!r}")
