"""One idea through the referee, start to end (L10). L10b's intake calls the same functions by itself.

    .venv\\Scripts\\python -m machine.report tune idea-001                 every version on 2016-2020; ONE chosen
    .venv\\Scripts\\python -m machine.report check idea-001 --who Tejas    the check years opened ONCE for it; judged
    .venv\\Scripts\\python -m machine.report build idea-001 [--send]       the report (sent = REPORTED)

tune: the idea must have no question left (every unclear part a VERSION, or answered). It is then CONFIRMED &
      LOCKED before any money is seen (lesson 16): its meaning can no longer change. Every reading is compiled on
      2016-2020 (a version each, <= the budget of 20), tested, and one is chosen by the referee (machine/referee/
      tune.py) -> TUNE RUN.
check: the check years 2021-2023 are opened ONCE for the chosen version (check-opened.json, who and why), the chosen
      version is compiled again on 2016-2023 - its 2016-2020 signals must be exactly the tune signals (nothing after a
      moment changes a signal known at that moment) - and its 2021-2023 signals are judged (judge.judge_check)
      -> CHECK OPENED.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from machine import config
from machine.ideas import store
from machine.ideas.compile import Market, compile_idea, needed_tfs, prepare
from machine.ideas.form import LOCKED_FROM, STATES, reading_choices
from machine.referee import judge, prior, splits, tune


class RunError(ValueError):
    """An idea is not ready for this step, or a step would repeat what may happen only once."""


def about(form, readings: dict[str, str]) -> str:
    """What a version is, in his words: each unclear part's chosen reading (its meaning), no part ids."""
    by_id = {u.id: u for u in form.unclear}
    parts = []
    for part, rid in readings.items():
        r = next(x for x in by_id[part].readings if x.id == rid)
        parts.append(r.means)
    return " · ".join(parts) or "the idea as written (nothing unclear)"


def lock(idea_id: str) -> None:
    """ANSWERED with no question left -> CONFIRMED & LOCKED (the rule of machine.questions.interpret.lock: every ASK
    part answered). Refused while a question is open - never guess a meaning (R13)."""
    form = store.load_form(idea_id)
    state = store.read_state(idea_id)["state"]
    if STATES.index(state) >= STATES.index(LOCKED_FROM):
        return
    open_asks = [u.id for u in form.unclear if u.kind == "ask" and u.answer is None]
    if open_asks or state != "ANSWERED":
        raise RunError(f"{idea_id} is {state}; questions still open: {open_asks or 'none'} - it is tested only once "
                       "every unclear part is answered or is a version the machine tries (R13: never guess)")
    store.set_state(idea_id, "CONFIRMED & LOCKED", "L10: no question left - every unclear part is a version the "
                    "machine tries; locked before any money is seen (lesson 16)")


def tune_idea(idea_id: str, *, log=print) -> tune.Tuned:
    lock(idea_id)
    form = store.load_form(idea_id)
    if tune.latest(idea_id) is not None:
        raise RunError(f"{idea_id} was already tuned (a referee-tune trial file exists); its versions are counted "
                       "tries - running them again adds nothing")
    choices = reading_choices(form)
    if len(choices) > config.VERSION_BUDGET:
        raise RunError(f"{idea_id} has {len(choices)} readings, over the budget of {config.VERSION_BUDGET} versions")
    t0 = time.time()
    market = Market.load(str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1]))
    log(f"bars 2016-2020 read in {time.time() - t0:.0f} s")
    versions, abouts, readings = {}, {}, {}
    for k, rd in enumerate(choices, 1):
        t1 = time.time()
        c = compile_idea(form, market, readings=rd)
        if c.version_id in versions:
            raise RunError(f"two readings compile to the same version {c.version_id}: {readings[c.version_id]} and {rd}")
        versions[c.version_id], abouts[c.version_id], readings[c.version_id] = c.signals, about(form, rd), rd
        log(f"  {k}/{len(choices)} {c.version_id}: {len(c.signals)} signals ({time.time() - t1:.0f} s) - "
            f"{abouts[c.version_id]}")
    t2 = time.time()
    t = tune.run_versions(versions, tf=form.timeframe, about=abouts, readings=readings,
                          note=f"L10: {idea_id}, every reading on 2016-2020")
    log(f"tested and chosen in {time.time() - t2:.0f} s")
    store.set_state(idea_id, "TUNE RUN", f"L10: {len(versions)} versions on 2016-2020; chosen {t.chosen} "
                    f"({abouts[t.chosen]})", waits_for="the check years 2021-2023, opened once for the chosen version")
    return t


def check_idea(idea_id: str, *, who: str, why: str | None = None, n_random: int | None = None,
               log=print, m1: pd.DataFrame | None = None) -> judge.Report:
    """m1: made-up 1-minute bars of 2016-2023 instead of the real prices (the research agent's noise exam, L14.2) -
    every check below runs the same way on them."""
    form = store.load_form(idea_id)
    tf = tune.latest(idea_id)
    if tf is None:
        raise RunError(f"{idea_id} has no tune file yet: python -m machine.report tune {idea_id}")
    chosen = tf["result"]["chosen"]
    row = tf["result"]["versions"][chosen]
    done = [f for f in judge.trials.load() if f.get("kind") == "referee" and f.get("idea_id") == idea_id
            and str(f["settings"].get("judged_on", "")).startswith("the check years")]
    if done:
        raise RunError(f"{idea_id}: the check years were already judged ({done[0]['made_at'][:16]}) - they are "
                       "judged ONCE")
    # the family the tune counted in: the form's parent, or the older idea the research tools found it is a version
    # of (L14.1, section D) - the tune file's signals carry it
    parent = form.parent_idea or tf.get("parent_idea")
    if parent and not form.parent_idea:
        form = form.model_copy(update={"parent_idea": parent})
    opening = splits.family_opening(idea_id, parent)
    if opening is None:
        n = len(tf["result"]["versions"])
        why = why or (f"{chosen} ({row['about']}) chosen on 2016-2020 by the highest Deflated Sharpe of {n} versions "
                      f"(tune file {tf['trial_id']})")
        rec = splits.open_check(idea_id, chosen, parent, who=who, why=why)
        log(f"check years opened: {rec['opened_at']} for {chosen} by {who}")
        store.set_state(idea_id, "CHECK OPENED", f"L10: 2021-2023 opened once for {chosen}",
                        waits_for="the referee's answer on 2021-2023")
    elif (opening["idea_id"], opening.get("version_id")) != (idea_id, chosen):
        raise RunError(f"the {opening['family']} family opened its check years for {opening['idea_id']} "
                       f"{opening.get('version_id')}, not {chosen}")
    else:
        log(f"check years already opened for {chosen} ({opening.get('opened_at')}): judging (a run that stopped)")
    t0 = time.time()
    tfs = needed_tfs(prepare(form, row["readings"], row.get("changes") or None)[0])   # its own charts (dimension 1)
    market = (Market.load(str(config.TUNE_YEARS[0]), str(config.CHECK_YEARS[-1]), tfs=tfs) if m1 is None else
              Market.from_m1(m1.loc[str(config.TUNE_YEARS[0]):str(config.CHECK_YEARS[-1])], tfs=tfs))
    c = compile_idea(form, market, readings=row["readings"], changes=row.get("changes") or None)
    if c.version_id != chosen:
        raise RunError(f"the chosen readings compile to {c.version_id}, not {chosen}: the form or the code changed")
    first_check = pd.Timestamp(f"{config.CHECK_YEARS[0]}-01-01")
    last_tune_bar = market.bars("5m").loc[:str(config.TUNE_YEARS[-1])].index[-1]
    last_tune = last_tune_bar + pd.Timedelta(minutes=5)                       # the last tune bar's close
    before = [s for s in c.signals if pd.Timestamp(s.known_at) <= last_tune]
    if tune.signals_sha256(before) != row["signals_sha256"]:
        raise RunError(f"the 2016-2020 signals of {chosen} compiled on 2016-2023 are not the tune signals "
                       f"({len(before)} vs {row['signals']}): a signal read bars after its moment - stop and find it")
    check = [s for s in c.signals if pd.Timestamp(s.known_at) > last_tune]
    early = [s for s in check if pd.Timestamp(s.known_at) < first_check]
    if early:
        raise RunError(f"{len(early)} signals are known after the last 2020 bar but before 2021: {early[0].id}")
    log(f"compiled 2016-2023 in {time.time() - t0:.0f} s: {len(before)} signals up to 2020 (the same as the tune), "
        f"{len(check)} in 2021-2023")
    t1 = time.time()
    rep = judge.judge_check(check, tf=c.tf, key=prior.key_for_form(c.form), n_random=n_random,
                            m1=None if m1 is None else m1.loc[str(config.CHECK_YEARS[0]):str(config.CHECK_YEARS[-1])],
                            note=f"L10: {idea_id} {chosen} on the check years")
    log(f"judged in {time.time() - t1:.0f} s: {rep.word}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("tune", "check", "build"))
    ap.add_argument("idea")
    ap.add_argument("--who", help="check: who opens the check years (goes into check-opened.json)")
    ap.add_argument("--why", help="check: why (default: the tune choice)")
    ap.add_argument("--send", action="store_true", help="build: put the report into the Telegram outbox")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.step == "tune":
        print(tune_idea(a.idea).text())
    elif a.step == "check":
        if not a.who:
            ap.error("check needs --who (the check years open once; the file says who opened them)")
        print(check_idea(a.idea, who=a.who, why=a.why).text())
    else:
        from machine.report import build as B
        b = B.build(a.idea)
        print(b.text)
        print(f"\npicture: {b.picture['path']}\npage: {b.page_path}")
        if a.send:
            line = B.send(b)
            store.set_state(a.idea, "REPORTED", f"L10: report sent ({line['id']})",
                            waits_for="the machine: its choice after the report (LEARNED #162 - it decides and "
                                      "tells him)")
            print(f"in the outbox: {line['id']} (the bot sends it)")
    return 0
