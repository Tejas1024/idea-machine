"""The exams on made-up answers (L15.2) - run BEFORE his answers are scored, so the measuring is checked first
(as L13.2 and L14.2 did): the same charts, the same split, the same models; only the answers are made up.

    noise   his yes / no counts of each year shuffled among that year's charts: nothing in a picture tells them
            apart, so a model must come out no better than "always no" except by luck (about 1 run in 10 at the 90%
            level, half of them better and half worse). Model A config.EYES_NOISE_RUNS_A times, model B
            config.EYES_NOISE_RUNS_B times (one start each - it is slow).
    rule    his written rule as the answer (yes when calm is at most 0.74): a thing the pictures DO show. Model A has
            calm among its numbers, so it must find it nearly always; model B (its 3 starts, as in the real run) shows
            how well a picture network sees small candles in a band at 224 x 448.

    python -m machine.eyes exams      -> plan/work/eyes/exams.json + exams.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from machine import config
from machine.eyes import labels as LB
from machine.eyes import run as RUN


def run_all(data: RUN.Data, *, noise_a: int | None = None, noise_b: int | None = None, rule_b: bool = True,
            pretrained: bool = True, say=print) -> dict:
    noise_a = config.EYES_NOISE_RUNS_A if noise_a is None else noise_a
    noise_b = config.EYES_NOISE_RUNS_B if noise_b is None else noise_b
    out: dict = {"noise A": [], "noise B": []}
    for i in range(noise_a):
        r = RUN.measure(data, LB.made_up(data.lab, "noise", seed=i), models=("model A",))
        out["noise A"].append(_line(r, "model A", seed=i))
    say(f"noise, model A: {noise_a} runs done")
    for i in range(noise_b):
        r = RUN.measure(data, LB.made_up(data.lab, "noise", seed=1000 + i), models=("model B",), seeds=(0,),
                        pretrained=pretrained, say=say)
        out["noise B"].append(_line(r, "model B", seed=1000 + i))
        say(f"noise, model B: run {i + 1} of {noise_b} done")
    y = LB.made_up(data.lab, "rule")
    r = RUN.measure(data, y, models=("model A", "model B") if rule_b else ("model A",), pretrained=pretrained, say=say)
    out["rule"] = {g: _line(r, g) for g in ("model A", "model B") if g in r["scores"]}
    out["rule"]["his rule"] = _line(r, "his rule")
    out["summary"] = summary(out)
    return out


def _line(r: dict, g: str, seed: int | None = None) -> dict:
    s, base = r["scores"][g], r["scores"]["always no"]
    p = r["pairs"].get(f"always no vs {g}")
    return {"seed": seed, "n": s["n"], "right": s["right"], "accuracy": s["accuracy"], "always_no": base["accuracy"],
            "better_than_always_no": None if p is None else bool(p["beyond_luck"] and p["only_b"] > p["only_a"]),
            "worse_than_always_no": None if p is None else bool(p["beyond_luck"] and p["only_a"] > p["only_b"]),
            "ranks": s["ranks"], "yes_found": s["yes_found"], "yes_all": s["yes_all"]}


def summary(out: dict) -> dict:
    s: dict = {}
    for key in ("noise A", "noise B"):
        rows = out[key]
        if rows:
            gap = [x["accuracy"] - x["always_no"] for x in rows]
            s[key] = {"runs": len(rows), "middle_gap": round(float(np.median(gap)), 4),
                      "better_beyond_luck": sum(bool(x["better_than_always_no"]) for x in rows),
                      "worse_beyond_luck": sum(bool(x["worse_than_always_no"]) for x in rows),
                      "middle_ranks": round(float(np.median([x["ranks"] for x in rows])), 4)}
    s["rule"] = {g: {"accuracy": x["accuracy"], "always_no": x["always_no"], "ranks": x["ranks"]}
                 for g, x in out.get("rule", {}).items()}
    return s


def write(out: dict, folder: Path | None = None) -> tuple[Path, Path]:
    folder = Path(folder or config.EYES_WORK_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    js, md = folder / "exams.json", folder / "exams.md"
    _put(js, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    _put(md, page(out))
    return js, md


def page(out: dict) -> str:
    s = out["summary"]
    L = ["# L15.2 - the exams on made-up answers (before his answers were scored)", "",
         "Same charts, same split (2018 / 2019 / 2020 each guessed by models that learned only from the charts "
         "before), same models. Only the answers are made up - so these numbers say how the measuring behaves, "
         "not how well a model sees his zones.", ""]
    for key, words in (("noise A", "model A (numbers)"), ("noise B", "model B (pictures, one start)")):
        if key in s:
            x = s[key]
            L += [f"## Noise - {words}: {x['runs']} runs",
                  f"His yes / no counts of each year shuffled among that year's charts (nothing in a picture tells "
                  f"them apart). Middle result: {_pts(x['middle_gap'])} against 'always no'. **Better than 'always "
                  f"no' beyond luck: {x['better_beyond_luck']} of {x['runs']} runs** - each would be a false alarm; "
                  f"with nothing to find, at most about 1 run in 20 may do it by luck. Worse beyond luck: "
                  f"{x['worse_beyond_luck']} of {x['runs']} - expected here: with nothing to find, every 'yes' a "
                  f"model says is right only as often as a yes is common, so 'always no' is the best guess. A yes "
                  f"chart ranks higher, middle: {x['middle_ranks']:.2f} (0.50 = a coin).",
                  ""]
    if s.get("rule"):
        L += ["## His written rule planted as the answer (yes when calm is at most 0.74)", "",
              "| who | right | 'always no' on the same answers | yes chart ranks higher |", "|---|---|---|---|"]
        for g, x in s["rule"].items():
            L.append(f"| {RUN.NAMES[g]} | {x['accuracy']:.1%} | {x['always_no']:.1%} | {x['ranks']:.2f} |")
        L += ["", "His rule scores 100% here by construction (it IS the answer) - a check that the measuring reads "
              "the answers right."]
    if out.get("recipes"):
        r = out["recipes"]
        L += ["", "## Other ways to teach model B, tried on the same planted answers (one start each)", "",
              f"Bar set BEFORE these runs: {r['bar']}", "",
              "| how model B was taught | right | 'always no' | yes chart ranks higher | planted yes found |",
              "|---|---|---|---|---|"]
        for name, x in r["tried"].items():
            L.append(f"| {name} | {x['accuracy']:.1%} | {x['always_no']:.1%} | {x['ranks']:.2f} | "
                     f"{x['yes_found']} of {x['yes_all']} |")
        L += ["", r["chosen"]]
    L += ["", "Every run: plan/work/eyes/exams.json. Made by `python -m machine.eyes exams`."]
    return "\n".join(L) + "\n"


def _pts(gap: float) -> str:
    return f"{gap * 100:+.1f} points"


def _put(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
