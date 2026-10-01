"""The daily label charts on Telegram (L15.1) - ONLY after Tejas opts in (T9). Until then nothing here sends anything.

    .venv\\Scripts\\python -m machine.labels send                today's 10 charts into the bot's outbox (once a day)
    .venv\\Scripts\\python -m machine.labels send --dry-run      draw them into data/labels/<word>/dry-run/, send nothing
                                                                (never SHOW these to him: they are the very charts that
                                                                day sends - #117 never the same picture twice)
    .venv\\Scripts\\python -m machine.labels opt-in --answer "opt in" --words "<his words>"   T9: his answer + the task

Every day at config.LABELS_AT (Task Scheduler, added only when he opts in), `send`:
  1. stops unless plan/work/labels/opt-in.md's newest entry says "opt in" (T9);
  2. stops if today's charts were already queued, or if config.LABELS_WAITING_MAX charts still wait for his answer;
  3. picks config.LABELS_PER_DAY zones (pick): the zone block's zones (machine/blocks/zones.py, all tiers) of the
     config.LABELS_WINDOWS window, built and known in config.LABELS_YEARS (2016-2020: the hidden years stay hidden),
     spread over steps of how quiet their candles are (config.LABELS_CALM_EDGES / LABELS_PER_STEP; the first step ends
     at his rule's 0.74). Never a zone sent before, never on a day already used, never the same band on overlapping
     candles, no two charts of one day with a candle in common, never the candles of one of his own drawn answers
     (those labels stay separate). The same day always picks the same charts (the random draw is seeded by the word
     and the day);
  4. draws each one blind (machine/labels/picture.py: the zone's own 5 days of hour candles, the band, the question;
     every picture rule of his checked, never a picture he has seen);
  5. writes one line per chart to plan/work/labels/<word>/charts.jsonl and the lines TO him into the outbox
     (plan/work/telegram-outbox.jsonl) - the ONE bot sends them (L9a) and files his replies (machine/labels/replies.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, rules
from machine.blocks import zones as ZN
from machine.labels import drawings as DW
from machine.labels import picture as P
from machine.labels import replies as LR
from machine.labels import words as W
from machine.labels.optin import opt_in, opt_in_state  # noqa: F401 - his T9 answer (read here, written by the CLI)


class LabelsOff(RuntimeError):
    """He has not opted in (T9): nothing is sent."""


# ============================================================================================== the charts to ask

def steps() -> list[tuple[float, float]]:
    """The calm steps: from 0 to his rule's 0.74, then config.LABELS_CALM_EDGES, then above."""
    edges = [0.0, float(rules.get("what_a_strong_zone_is")["calm_max"]), *config.LABELS_CALM_EDGES, np.inf]
    return list(zip(edges[:-1], edges[1:]))


def step_words(s: int) -> str:
    lo, hi = steps()[s]
    return f"calm {lo:.2f}-{hi:.2f}" if np.isfinite(hi) else f"calm above {lo:.2f}"


def candidates(word: str, m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """Every zone that may be asked: the zone block's zones of the chosen windows, built from candles of
    config.LABELS_YEARS and known before those years end, not on the candles of one of his drawn answers."""
    W.get(word)
    years = config.LABELS_YEARS
    z = ZN.strong_zones(m5, h1, tiers=(1, 2, 3))
    z = z[z["window"].isin(config.LABELS_WINDOWS)]
    z = z[(z["src_first"] >= pd.Timestamp(f"{min(years)}-01-01"))
          & (z["known_at"] <= pd.Timestamp(f"{max(years) + 1}-01-01")) & (z["src_last"].dt.year <= max(years))]
    keep = np.ones(len(z), bool)
    for x in DW.labels(word):
        if x.hidden:
            continue
        a, b = x.span()
        keep &= ~((z["src_first"] <= b) & (a <= z["src_last"])).to_numpy()
    z = z[keep].copy()
    cuts = [hi for _lo, hi in steps()[:-1]]
    z["step"] = np.searchsorted(np.asarray(cuts), z["calm"].to_numpy(float), side="left")
    z["day"] = z["known_at"].dt.normalize()
    return z.dropna(subset=["calm"]).reset_index(drop=True)


def _clash(r, old: list[dict], batch: list[dict]) -> bool:
    """Not with a chart sent before: the same day of zones, or the same band on candles it showed. Not with a chart of
    the same day's batch: any candle in common (ten charts a day = ten different stretches of price)."""
    def overlap(o) -> bool:
        return pd.Timestamp(r["src_first"]) <= pd.Timestamp(o["src_last"]) and \
            pd.Timestamp(o["src_first"]) <= pd.Timestamp(r["src_last"])
    for o in old + batch:
        if pd.Timestamp(r["day"]) == pd.Timestamp(o["day"]):
            return True
        if overlap(o) and min(r["hi"], o["hi"]) > max(r["lo"], o["lo"]):
            return True
    return any(overlap(o) for o in batch)


def pick(pool: pd.DataFrame, sent: list[dict], word: str, day: dt.date,
         per_step: tuple[int, ...] = config.LABELS_PER_STEP) -> list[dict]:
    """The day's charts: per_step[s] from calm step s (a step that runs short gives its place to the steps with
    charts left, nearest first). Each pick says its step, how many were left in that step, and how many were wanted,
    so L15.2 can weigh the labels back to how often each kind of zone really happens."""
    if len(per_step) != len(steps()):
        raise ValueError(f"per_step needs one number per calm step ({len(steps())})")
    seed = int(hashlib.sha256(f"{word}|{day:%Y-%m-%d}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    done_ids = {c["zone_id"] for c in sent}
    old = [{"day": c["known_day"], "src_first": c["first"], "src_last": c["last"], "lo": c["lo"], "hi": c["hi"]}
           for c in sent]
    free = pool[~pool["zone_id"].isin(done_ids)]
    by_step = {s: free[free["step"] == s] for s in range(len(per_step))}
    orders = {s: rng.permutation(len(t)) for s, t in by_step.items()}
    taken = {s: 0 for s in by_step}
    picks: list[dict] = []

    def take(s: int, want: int) -> int:
        got = 0
        t, order = by_step[s], orders[s]
        while got < want and taken[s] < len(order):
            r = t.iloc[order[taken[s]]].to_dict()
            taken[s] += 1
            if _clash(r, old, picks):
                continue
            picks.append({**r, "step_size": len(t), "step_wanted": per_step[s]})
            got += 1
        return got

    short = {s: n - take(s, n) for s, n in enumerate(per_step)}
    for s, missing in short.items():
        for near in sorted(by_step, key=lambda k: (abs(k - s), k)):
            if missing <= 0:
                break
            if near != s:
                missing -= take(near, missing)
    order = rng.permutation(len(picks))                   # the steps are not guessable from the order
    return [picks[i] for i in order]


# ============================================================================================== one day

def send(word: str = W.STRONG_ZONE.slug, *, today: dt.date | None = None, dry: bool = False,
         base: Path | None = None, outbox: Path | None = None, shown: Path | None = None,
         pictures: Path | None = None, opt_in_path: Path | None = None,
         m5: pd.DataFrame | None = None, h1: pd.DataFrame | None = None) -> dict:
    """Today's charts. Returns what was done, in words for the log ("said") and as numbers."""
    w = W.get(word)
    today = today or dt.date.today()
    if not dry and opt_in_state(opt_in_path)["answer"] != "opt in":
        raise LabelsOff("Tejas has not opted in to the daily charts (T9: plan/work/labels/opt-in.md) - nothing sent")
    sent = LR.read_charts(word, base)
    if any(c["day"] == f"{today:%Y-%m-%d}" for c in sent) and not dry:
        return {"sent": 0, "said": f"today's charts ({today}) were already queued"}
    answered = LR.latest(word, base)
    waiting = [c for c in sent if c["qid"] not in answered]
    if len(waiting) >= config.LABELS_WAITING_MAX and not dry:
        return {"sent": 0, "said": f"{len(waiting)} charts still wait for his answer - no new ones today"}
    if m5 is None or h1 is None:
        from machine.data.bars import load_bars
        years = config.LABELS_YEARS
        m5 = load_bars("5m", str(min(years)), str(max(years)))
        h1 = load_bars("1h", str(min(years)), str(max(years)))
    pool = candidates(word, m5, h1)
    chosen = pick(pool, sent, word, today)
    folder = Path(pictures or config.LABELS_DIR / word) / ("dry-run" if dry else "charts")
    fp = P.code_fingerprint()
    rows, lines = [], []
    for k, r in enumerate(chosen, 1):
        q = LR.qid(word, today, k)
        h = P.hour_candles(m5, r["src_first"], r["src_last"])
        path = folder / f"{q}.jpg"
        drawn = P.draw(w, h, r["lo"], r["hi"], path, subject=f"label chart {word} {r['zone_id']}", shown=shown)
        rows.append({"qid": q, "day": f"{today:%Y-%m-%d}", "n": k, "of": len(chosen), "word": word,
                     "zone_id": r["zone_id"], "window": r["window"], "known_at": _t(r["known_at"]),
                     "known_day": _t(r["day"]), "first": _t(r["src_first"]), "last": _t(r["src_last"]),
                     "lo": round(float(r["lo"]), 3), "hi": round(float(r["hi"]), 3), "calm": round(float(r["calm"]), 4),
                     "tier": int(r["tier"]), "calm_on_chart": round(P.calm(h, r["lo"], r["hi"]), 4),
                     "step": int(r["step"]), "step_words": step_words(int(r["step"])),
                     "step_size": int(r["step_size"]), "step_wanted": int(r["step_wanted"]),
                     "picture": _show(path), "pixel_sha256": drawn.pixel_sha256, "drawn_by": fp,
                     "queued": dt.datetime.now().isoformat(timespec="seconds")})
        lines.append({"id": f"label-{q}", "qid": q, "photo": _show(path), "why": "L15.1 label chart",
                      "text": f"Zone chart {k} of {len(chosen)} ({today:%d %b}). Your kind of zone? "
                              "Reply to this picture: yes / no / not sure",
                      "shown": {"subject": f"label chart {word} {r['zone_id']}", "pixel_sha256": drawn.pixel_sha256,
                                "kind": "question", "file": _show(path), "why": f"L15.1 label chart {q}"}})
    intro = {"id": f"labels-{word}-{today:%Y%m%d}-intro", "why": "L15.1 label charts",
             "text": (f"Today's {len(chosen)} zone charts.\n"
                      f"On each: {w.ask}\n"
                      "Reply to each picture: yes / no / not sure. \"Not sure\" is a real answer - don't overthink.\n"
                      "Or answer all in one reply to any of them: 1 yes 2 no 3 not sure ...")}
    out = {"sent": 0 if dry else len(rows), "dry_run": dry, "rows": rows, "lines": [intro, *lines],
           "pool": len(pool), "said": ""}
    if not rows and not dry:
        return _finished(word, len(sent), len(answered), outbox)
    if dry:
        out["said"] = f"dry run: {len(rows)} charts drawn in {_show(folder)}"
        return out
    charts = LR.charts_file(word, base)
    charts.parent.mkdir(parents=True, exist_ok=True)
    with open(charts, "a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    box = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    box.parent.mkdir(parents=True, exist_ok=True)
    with open(box, "a", encoding="utf-8", newline="\n") as fh:
        for ln in out["lines"]:
            fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
    out["said"] = f"{len(rows)} charts queued for Telegram ({_show(charts)})"
    return out


def _finished(word: str, sent: int, answered: int, outbox: Path | None) -> dict:
    """No chart is left that the rules allow (about 610 for "strong zone" in 2016-2020 - measured 30 Sep 2026): he is
    told ONCE (the line's id is fixed and written once; the bot sends an id once)."""
    box = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    lid = f"labels-{word}-finished"
    have = box.read_text(encoding="utf-8") if box.is_file() else ""
    if f'"id": "{lid}"' not in have:
        box.parent.mkdir(parents=True, exist_ok=True)
        with open(box, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"id": lid, "why": "L15.1 label charts", "text": (
                f"No more zone charts: every one the rules allow has been sent ({sent}; you answered {answered}). "
                "Thank you. Nothing more comes until a new set is built.")}, ensure_ascii=False) + "\n")
    return {"sent": 0, "said": f"no chart left to ask for {word} ({sent} sent, {answered} answered) - he was told once"}


def log(word: str, text: str, folder: Path | None = None) -> None:
    folder = Path(folder or config.LABELS_DIR / word)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "daily.log", "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {text}\n")


# ============================================================================================== Task Scheduler

def install(at: str = config.LABELS_AT, *, root: Path = config.ROOT, opt_in_path: Path | None = None) -> str:
    """Adds (or replaces) the daily task - only after he opted in. pythonw (no window), every day at `at`; a day missed
    while the PC was off runs when it is on again."""
    from machine.telegram.nightly import _ps, _run
    if opt_in_state(opt_in_path)["answer"] != "opt in":
        raise LabelsOff("no daily task before he opts in (T9)")
    if os.name != "nt":
        raise RuntimeError("Task Scheduler is on Windows only")
    exe = Path(root) / ".venv" / "Scripts" / "pythonw.exe"
    if not exe.is_file():
        raise RuntimeError(f"{exe} is missing")
    if not re.fullmatch(r"\d{1,2}:\d{2}", at):
        raise ValueError(f"--at wants a time like 21:00, not {at!r}")
    ps = ("$a = New-ScheduledTaskAction -Execute {exe} -Argument '-m machine.labels send' -WorkingDirectory {root};"
          "$t = New-ScheduledTaskTrigger -Daily -At {at};"
          "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
          "-ExecutionTimeLimit (New-TimeSpan -Hours 1);"
          "Register-ScheduledTask -TaskName {name} -Action $a -Trigger $t -Settings $s -Force "
          "-Description {desc} | Out-Null;"
          "(Get-ScheduledTask -TaskName {name} | Get-ScheduledTaskInfo).NextRunTime.ToString('yyyy-MM-dd HH:mm')"
          ).format(exe=_ps(str(exe)), root=_ps(str(root)), at=_ps(at), name=_ps(config.LABELS_TASK_NAME),
                   desc=_ps("10 zone charts a day on Telegram for his yes / no / not sure (plan/likhith.md L15.1)."))
    done = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=False)
    if done.returncode:
        raise RuntimeError("Task Scheduler said no: " + (done.stderr or b"").decode("utf-8", "replace")[:300])
    return f"'{config.LABELS_TASK_NAME}' next runs {done.stdout.decode('utf-8', 'replace').strip()}"


def uninstall() -> str:
    from machine.telegram.nightly import _ps, _run
    if os.name != "nt":
        return "Task Scheduler is on Windows only - nothing to remove"
    _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
          f"Unregister-ScheduledTask -TaskName {_ps(config.LABELS_TASK_NAME)} -Confirm:$false"], check=False)
    return f"removed '{config.LABELS_TASK_NAME}' (if it was there)"


def _t(x) -> str:
    return pd.Timestamp(x).isoformat(sep=" ")


def _show(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()
