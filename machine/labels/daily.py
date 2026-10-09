"""The daily label charts on Telegram (L15.1) - ONLY after Tejas opts in (T9). Until then nothing here sends anything.

    .venv\\Scripts\\python -m machine.labels send                today's 10 charts into the bot's outbox (once a day)
    .venv\\Scripts\\python -m machine.labels send --dry-run      draw them into data/labels/<word>/dry-run/, send nothing
                                                                (never SHOW these to him: they are the very charts that
                                                                day sends - #117 never the same picture twice)
    .venv\\Scripts\\python -m machine.labels next                the next day's 10 charts NOW, if he answered every chart
    .venv\\Scripts\\python -m machine.labels opt-in --answer "opt in" --words "<his words>"   T9: his answer + the task

Since 7 Oct 2026 (his words: "you can send me that 200 or 600 things that needed me") the next day's charts also go
as soon as he has answered every chart sent - `send_next`, started by each intake run (the bot starts one after each of
his answers), while config.LABELS_NEXT_AT_ONCE is on. Every day at config.LABELS_AT (Task Scheduler, added only when he
opts in), `send`:
  1. stops unless plan/work/labels/opt-in.md's newest entry says "opt in" (T9);
  2. stops if today's charts were already queued, or if config.LABELS_WAITING_MAX charts still wait for his answer
     (one day's: a plain list of numbers is then always for the right day, #166) - and then TELLS him, once that day;
  3. picks config.LABELS_PER_DAY zones (pick): the zone block's zones (machine/blocks/zones.py, all tiers) of the
     config.LABELS_WINDOWS window, built and known in config.LABELS_YEARS (2016-2020: the hidden years stay hidden),
     spread over steps of how quiet their candles are (config.LABELS_CALM_EDGES / LABELS_PER_STEP; the first step ends
     at his rule's 0.74). Never a zone sent before, never on a day already used, never a candle an earlier chart
     showed (#117 "never send the same chart twice": a week shown again with the band moved "looked similar" to him,
     4 Oct 2026 - so about 200 charts in all, not 612), never the candles of one of his own drawn answers
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
from machine.host.lock import one_run
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
    """Not with a chart sent before or one of the same day's batch: the same day of zones, or ANY candle in common -
    every chart a stretch of price he has never seen. His #117: "I don't want the same charts again"; 4 Oct 2026
    (inbox 2304) two charts of the same week with the band at another price "looked similar" to him. Until then only
    the same band was refused across days, and 470 of the 585 charts still to come showed a week he had seen; now 198
    charts in all (measured 4 Oct), the end he is told once (_finished)."""
    def overlap(o) -> bool:
        return pd.Timestamp(r["src_first"]) <= pd.Timestamp(o["src_last"]) and \
            pd.Timestamp(o["src_first"]) <= pd.Timestamp(r["src_last"])
    return any(pd.Timestamp(r["day"]) == pd.Timestamp(o["day"]) or overlap(o) for o in old + batch)


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
    """Today's charts. Returns what was done, in words for the log ("said") and as numbers. One sender at a time:
    the 21:00 task and send_next never queue the same day twice."""
    W.get(word)
    if not dry and opt_in_state(opt_in_path)["answer"] != "opt in":
        raise LabelsOff("Tejas has not opted in to the daily charts (T9: plan/work/labels/opt-in.md) - nothing sent")
    if dry:
        return _send(word, today or dt.date.today(), dry=True, base=base, outbox=outbox, shown=shown,
                     pictures=pictures, m5=m5, h1=h1)
    with one_run(_lock_path(word, pictures)) as mine:
        if not mine:
            return {"sent": 0, "said": "another run is queueing charts right now - nothing from this one"}
        return _send(word, today or dt.date.today(), base=base, outbox=outbox, shown=shown, pictures=pictures,
                     m5=m5, h1=h1)


def _send(word: str, today: dt.date, *, dry: bool = False, base: Path | None = None, outbox: Path | None = None,
          shown: Path | None = None, pictures: Path | None = None, m5: pd.DataFrame | None = None,
          h1: pd.DataFrame | None = None, early: dt.datetime | None = None, early_note: str = "") -> dict:
    """send's work (inside its lock). early: when send_next sent this day's charts ahead of their day; early_note: the
    line it adds to the first message (his labels so far)."""
    w = W.get(word)
    sent = LR.read_charts(word, base)
    if any(c["day"] == f"{today:%Y-%m-%d}" for c in sent) and not dry:
        return {"sent": 0, "said": f"today's charts ({today}) were already queued"}
    answered = LR.latest(word, base)
    waiting = [c for c in sent if c["qid"] not in answered]
    if len(waiting) >= config.LABELS_WAITING_MAX and not dry:
        return _held(word, today, waiting, outbox)
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
    # a set sent early is named by its number, not its day: "12 Oct" read as THIS October's gold to him (7 Oct 06:14,
    # "i hope it's not about this year October") - every chart is 2016-2020; the bot's "Saved for the charts of <day>"
    # still names the day (its reader is unchanged), so the first message says the day is only a name
    where = f"set {len({str(c['day']) for c in sent}) + 1}" if early else f"{today:%d %b}"
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
                     "queued": dt.datetime.now().isoformat(timespec="seconds"),
                     **({"early": early.isoformat(timespec="seconds")} if early else {})})
        lines.append({"id": f"label-{q}", "qid": q, "photo": _show(path), "why": "L15.1 label chart",
                      "text": f"Zone chart {k} of {len(chosen)} ({where}). Your kind of zone? "
                              "Reply to this picture: yes / no / not sure",
                      "shown": {"subject": f"label chart {word} {r['zone_id']}", "pixel_sha256": drawn.pixel_sha256,
                                "kind": "question", "file": _show(path), "why": f"L15.1 label chart {q}"}})
    first = (f"Today's {len(chosen)} zone charts.\n" if early is None else
             f"Zone charts, {where}: {len(chosen)} more, because you answered the ones before.\n"
             f"Every chart is gold from {min(config.LABELS_YEARS)}-{max(config.LABELS_YEARS)}. When the bot says "
             f"\"charts of {today:%d %b}\", that date is only this set's name.\n")
    then = ("" if early is None else
            f"\nYour next list of numbers answers THESE {len(chosen)}. The next ones come as soon as you answer."
            + (f"\n{early_note}" if early_note else ""))
    intro = {"id": f"labels-{word}-{today:%Y%m%d}-intro", "why": "L15.1 label charts",
             "text": (first
                      + f"On each: {w.ask}\n"
                      "Easiest: send ONE message with the numbers of YOUR zones, like: 1 3 7. "
                      "The charts you don't list count as not yours. None of them yours? Send: none\n"
                      "Or reply to a picture: yes / no / not sure. \"Not sure\" is a real answer - don't overthink."
                      + then)}
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


def send_next(word: str = W.STRONG_ZONE.slug, *, now: dt.datetime | None = None, base: Path | None = None,
              outbox: Path | None = None, shown: Path | None = None, pictures: Path | None = None,
              opt_in_path: Path | None = None, m5: pd.DataFrame | None = None,
              h1: pd.DataFrame | None = None) -> dict:
    """His words, 7 Oct 2026 (strategy/learned-inbox/2026-10-07-0016-session.md): "you can send me that 200 or 600
    things that needed me". While config.LABELS_NEXT_AT_ONCE is on, the NEXT day's charts go as soon as he has answered
    every chart sent - the intake starts this first, and the bot starts the intake after each of his answers. The set is
    the one planned for the next day no chart has used (the newest day sent + 1, today at the earliest): one day's
    charts still wait at most, so his plain list of numbers is always for them (#166 - the bot's reader is unchanged),
    and the 21:00 task finds its day already queued. Nothing while a chart waits, before he opts in (T9), before the
    first day's charts (the 21:00 task sends those) or with the switch off; once his labels reach L15.2's need he is
    told, once. Returns send's dict, plus "told"."""
    now = now or dt.datetime.now()
    if not config.LABELS_NEXT_AT_ONCE:
        return {"sent": 0, "told": False, "said": "the next charts come at 21:00 (config.LABELS_NEXT_AT_ONCE is off)"}
    if opt_in_state(opt_in_path)["answer"] != "opt in":
        return {"sent": 0, "told": False, "said": "he has not opted in to the daily charts (T9) - nothing sent"}
    with one_run(_lock_path(word, pictures)) as mine:
        if not mine:
            return {"sent": 0, "told": False, "said": "another run is queueing charts right now"}
        sent = LR.read_charts(word, base)
        answered = LR.latest(word, base)
        have, need = labels_so_far(base), l152_needs()
        told = _tell_reached(word, have, need, outbox)
        waiting = [c for c in sent if c["qid"] not in answered]
        if waiting or not sent:
            return {"sent": 0, "told": told, "said": (f"{len(waiting)} charts wait for his answer" if waiting else
                                                      "no chart sent yet - the first ones come at 21:00")}
        day = max(max(dt.date.fromisoformat(str(c["day"])) for c in sent) + dt.timedelta(days=1), now.date())
        note = f"Labels so far: {have}" + (f" - the L15.2 test needs {need}." if need and have < need else ".")
        out = _send(word, day, base=base, outbox=outbox, shown=shown, pictures=pictures, m5=m5, h1=h1, early=now,
                    early_note=note)
        return {**out, "told": told}


def labels_so_far(base: Path | None = None) -> int:
    """Every label of his: the count next_prompt holds L15.2 against (his drawn answers + the charts he answered)."""
    from machine.tools.next_prompt import count_labels
    return count_labels(base)


def l152_needs() -> int | None:
    """L15.2's "≥ N labels" from plan/PROGRESS.md - the same cell next_prompt reads; None when it cannot be read."""
    try:
        from machine.tools.next_prompt import parse_progress
        rows = parse_progress(Path(config.PROGRESS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return next((r.labels_needed for r in rows if r.id == "L15.2" and r.labels_needed), None)


def _tell_reached(word: str, have: int, need: int | None, outbox: Path | None) -> bool:
    """His labels reached L15.2's need: he is told ONCE (a fixed id, written once; the bot sends an id once) - he cannot
    keep a chat open (#172), so the machine says when the prompt can start."""
    if not need or have < need:
        return False
    box = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    lid = f"labels-{word}-reached-{need}"
    if f'"id": "{lid}"' in (box.read_text(encoding="utf-8") if box.is_file() else ""):
        return False
    box.parent.mkdir(parents=True, exist_ok=True)
    with open(box, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"id": lid, "why": "L15.1 label charts: L15.2's need reached", "text": (
            f"{have} labels now - enough for the L15.2 test (PyTorch vs LightGBM, your \"600 thing\": it needs "
            f"{need}).\nIt can start when you want: open a new Claude chat and paste L15.2 (plan/likhith.md).")},
            ensure_ascii=False) + "\n")
    return True


def _lock_path(word: str, pictures: Path | None = None) -> Path:
    """One sender at a time (machine/host/lock.py: let go of by itself when the program stops)."""
    return Path(pictures or config.LABELS_DIR / word) / "send.lock"


def _held(word: str, today: dt.date, waiting: list[dict], outbox: Path | None) -> dict:
    """No new charts while config.LABELS_WAITING_MAX wait - and he is TOLD, once that day. On 4 Oct 2026 nothing came
    and nothing said why: he thought the charts had not been sent (they had, on 2 and 3 Oct)."""
    when = ", ".join(dt.date.fromisoformat(d).strftime("%d %b") for d in sorted({str(c["day"]) for c in waiting}))
    box = Path(outbox or Path(config.WORK_DIR) / "telegram-outbox.jsonl")
    lid = f"labels-{word}-{today:%Y%m%d}-waiting"
    have = box.read_text(encoding="utf-8") if box.is_file() else ""
    if f'"id": "{lid}"' not in have:
        box.parent.mkdir(parents=True, exist_ok=True)
        with open(box, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"id": lid, "why": "L15.1 label charts held back", "text": (
                f"No new zone charts today: the {len(waiting)} charts of {when} still wait for your answer.\n"
                "Scroll up to them and send the numbers of YOUR zones, like: 1 3 7 - or: none.\n"
                + ("The next ones come as soon as you answer." if config.LABELS_NEXT_AT_ONCE else
                   "New charts come at the next 9 pm after you answer."))}, ensure_ascii=False) + "\n")
    return {"sent": 0, "said": f"{len(waiting)} charts still wait for his answer - no new ones today (he was told)"}


def _finished(word: str, sent: int, answered: int, outbox: Path | None) -> dict:
    """No chart is left that the rules allow (198 for "strong zone" in 2016-2020 since 4 Oct 2026, when a candle
    already shown was refused; about 610 before - measured 30 Sep 2026): he is told ONCE (the line's id is fixed and
    written once; the bot sends an id once)."""
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
