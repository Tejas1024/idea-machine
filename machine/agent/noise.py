"""The noise exam (L14.2): made-up gold prices with NO edge + one idea's research plan with a small budget -> the
research agent must answer "chance" (never "works").

    .venv\\Scripts\\python -m machine.agent noise --idea idea-002 --versions 3

How: the idea's SAVED research plan (research-plan.json - Claude's own plan, made before any money) is cut to its first
N versions and run exactly the agent's way (research.run: choose_slices, propose_version, run_version, write_report -
the referee chooses one version on 2016-2020, opens 2021-2023 once, judges), with the REAL compiler, tester, referee
and report - on a random walk of 1-minute bars 2016-2023 instead of gold. The idea runs as a copy (idea-002 ->
idea-902) in a temporary folder with its own trial files, check-opened file and tools' cache: nothing reaches
plan/work/ (the count of real tries stays real, R14), no real price is read, no Claude call is made. The learner is
not run on made-up prices (the report says "not measured yet").

The prices: one 1-minute bar every weekday minute; each minute's move is drawn at random (0.019% - about $0.25 at
$1,300, so a day's range is near gold's $15-20), and no minute says anything about the next one: there is no edge.
"""
from __future__ import annotations

import functools
import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from machine import config

FOLDERS = ("WORK_DIR", "IDEAS_DIR", "TRIALS_DIR", "QUESTION_QUEUE_DIR", "MCP_DIR", "SHOWN_PICTURES", "SEALED_LOG",
           "BELIEFS_DIR", "LABELS_DIR", "LABELS_WORK_DIR")


def walk_m1(seed: int = 7, start: str = "2016-01-04", end: str = "2023-12-29 23:59", price: float = 1075.0,
            step: float = 0.00019) -> pd.DataFrame:
    """A random walk of 1-minute bars (weekdays, server clock), the price data's columns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min", name="server_time")
    idx = idx[idx.dayofweek < 5]
    n = len(idx)
    close = price * np.exp(rng.normal(0.0, step, n).cumsum())
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0.0, step * 0.4, (2, n))) * close
    return pd.DataFrame({"open": open_.round(2), "high": (np.maximum(open_, close) + wick[0]).round(2),
                         "low": (np.minimum(open_, close) - wick[1]).round(2), "close": close.round(2),
                         "vol": 1, "spread": rng.choice([15, 20, 25, 30], n)}, index=idx)


@contextmanager
def sandbox(root: Path):
    """Every folder the research writes into -> a temporary folder (and back afterwards)."""
    new = {"WORK_DIR": root / "work", "IDEAS_DIR": root / "work" / "ideas", "TRIALS_DIR": root / "work" / "trials",
           "QUESTION_QUEUE_DIR": root / "work" / "question-queue", "MCP_DIR": root / "mcp",
           "SHOWN_PICTURES": root / "work" / "pictures-shown.jsonl", "SEALED_LOG": root / "sealed_access.log",
           "BELIEFS_DIR": root / "work" / "beliefs", "LABELS_DIR": root / "labels-data",
           "LABELS_WORK_DIR": root / "work" / "labels"}
    old = {k: getattr(config, k) for k in FOLDERS}
    for k, v in new.items():
        setattr(config, k, v)
    try:
        yield new
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def bench(m1: pd.DataFrame, seed: int, say: Callable = print):
    """The research tools' workbench on the made-up prices: the real compiler, tester, referee and report."""
    from machine.data import bars as BARS
    from machine.ideas import compile as C
    from machine.mcp import tools as T
    from machine.referee import judge, tune
    from machine.report import run as R
    from machine.tester import run as tester_run
    tune_m1 = m1.loc[:str(config.TUNE_YEARS[-1])]

    def load_bars(tf, lo, hi):
        part = m1.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
        return part if tf == "1m" else BARS.aggregate(part, tf)

    return T.Workbench(
        load_market=lambda tfs: C.Market.from_m1(tune_m1, tfs=tfs),
        tester=lambda sigs, tf, note: tester_run(sigs, tf=tf, m1=tune_m1, note=note),
        walk_forward=functools.partial(judge.judge_walk_forward, m1=tune_m1),
        tune=functools.partial(tune.run_versions, m1=tune_m1),
        check=lambda idea, who, log=print: R.check_idea(idea, who=who, log=log, m1=m1),
        learn=lambda idea, log=print: log("the learner is not run on made-up prices"),
        load_bars=load_bars, prices_hash=lambda: f"made-up random walk 2016-2023, seed {seed}", log=say)


def copy_idea(idea: str, copy: str, form_json: dict) -> None:
    """The idea's form as `copy`, at SHOWN with his YES - inside the sandbox (never his real idea's files)."""
    from machine.ideas import store
    from machine.ideas.form import Idea
    from machine.intake import shown
    form = Idea.load({**form_json, "id": copy, "status": "SHOWN", "parent_idea": None})
    store.create(form)
    shown.save({"idea": copy, "n": 1, "qid": shown.qid(copy), "status": "yes",
                "note": f"the noise exam: a made-up copy of {idea}"})


def exam(idea: str = "idea-002", versions: int = 3, seed: int = 7, say: Callable = print,
         keep: Path | None = None) -> dict:
    """The exam. Returns {"passed", "answer", "outcome", ...}; prints what happened."""
    from machine.agent import research
    from machine.ideas import store
    plan = research.load_plan(idea)
    if plan is None:
        raise research.ResearchError(f"{idea} has no saved research plan: python -m machine.agent plan {idea}")
    if not plan["versions"]:
        raise research.ResearchError(f"{idea}'s plan has no version (it asks a question first)")
    form_json = json.loads((store.folder(idea) / store.FORM).read_text(encoding="utf-8"))
    real_trials = len(list(Path(config.TRIALS_DIR).glob("*.json")))
    copy = "idea-9" + idea.split("-")[1][-2:]
    t0 = time.time()
    say(f"the noise exam: {idea}'s plan {plan['sha256'][:12]} (made by {plan['made_by']['calls'][0].get('model')}), "
        f"its first {versions} of {len(plan['versions'])} versions, on a random walk 2016-2023 (seed {seed})")
    m1 = walk_m1(seed)
    say(f"  {len(m1):,} made-up 1-minute bars in {time.time() - t0:.0f} s")
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="l142-noise-"))
    try:
        with sandbox(root):
            copy_idea(idea, copy, form_json)
            out = research.run(copy, planner=research.saved_planner(plan, versions), bench=bench(m1, seed, say),
                               send=False, say=say)
            final = out.final or ""
            files = sorted(p.name for p in Path(config.TRIALS_DIR).glob("*.json"))
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    still = len(list(Path(config.TRIALS_DIR).glob("*.json")))
    passed = out.status == "reported" and out.answer == "chance" and still == real_trials
    say(f"\n{final}\n\nthe answer on made-up prices with no edge: {out.answer!r} ({out.status}) - "
        f"{'PASSED' if passed else 'NOT PASSED'} · {len(files)} trial files, all in the temporary folder; "
        f"plan/work/trials/ {real_trials} -> {still} · {time.time() - t0:.0f} s")
    return {"passed": passed, "answer": out.answer, "status": out.status, "said": out.said, "final": final,
            "trial_files": len(files), "seconds": round(time.time() - t0)}
