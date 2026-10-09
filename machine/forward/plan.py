"""The forward plan (L17c): written and locked BEFORE any live bar, for an idea only new days can judge.

The machine chooses "FORWARD TEST" (machine/intake/state.py choice; machine/strategy/parked.py choice) for two kinds of
idea, and for no other (Claude's reading of plan/README.md section 7):
    plain stop  the referee said "chance — needs fresh years" (the old project already used its years) and EVERY other
                check passed;
    his hedge   tested with his hedge on 2016-2020, it made money AND beat 95 of 100 random entries with his hedge -
                but its check years were already used.
likhith.md L17c: "BEFORE any live data, write and commit plan/work/ideas/idea-XXX/forward-plan.md — the locked form,
the checks (the same verdict rules), the number of trades needed (from the power table), and the end date. Then
paper-trade it live (L17b's machinery) until that many trades exist; the referee judges once at the end. No changes to
the form during the test (a change = a new test)."

Two files in the idea's folder, written ONCE and never rewritten (a change is a new test with a new plan):
    forward-plan.json   what the pretend trades (L17b) read: the lock, the trades needed, the dates
    forward-plan.md     the same in words
They reach main with the nightly job (config.BOT_GIT_FILES has plan/work/ideas/*/*). The trades needed come from the
referee's own rule - an edge of Rs POWER_NEEDED_EDGE a trade caught at least POWER_NEEDED_SHARE of the time, never
fewer than MIN_TRADES - planted in the idea's own trades (machine/referee/power.py). Live prices are T14 use 1 (alerts
and pretend trades only): nothing here reads one.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

from machine import config, trials
from machine.ideas import store
from machine.referee import power, tune
from machine.referee import verdict as V

PLAN_JSON = "forward-plan.json"
PLAN_MD = "forward-plan.md"
WHO = "machine.forward (L17c, the forward plan)"
PLAIN, HEDGE = "plain stop", "his hedge"


class ForwardError(ValueError):
    """No forward plan can be written for this idea (not a forward-test idea, or a file it needs is missing)."""


# ================================================================================================= what it rests on

def _sha(path: Path) -> str:
    return trials.code_sha256([path])                        # line endings count as LF: the same on every PC


def _row(idea: str, version: str) -> dict:
    t = tune.latest(idea)
    if t is None or version not in t["result"]["versions"]:
        raise ForwardError(f"{idea}: the tune file has no version {version}")
    return t["result"]["versions"][version]


def facts(idea: str) -> dict:
    """Which road made it a forward-test idea, the version and its trades (the rupees of each closed trade - the
    noise the trades needed are counted in), its trades a week, and what is locked."""
    from machine.report import build as RB
    from machine.strategy import parked
    try:
        f = RB.facts(idea)
    except (RB.ReportError, store.StoreError, OSError, ValueError, KeyError) as err:
        raise ForwardError(f"{idea}: its report cannot be read ({type(err).__name__}: {err})") from err
    ans = f.r["answer"]
    failed = [c["name"] for c in ans["checks"] if not c["passed"] and c["name"] != "not already seen"]
    form = store.folder(idea) / "form.json"
    if ans["word"] == V.FRESH and not failed:
        vid = f.judge["version_id"]
        cols = f.r["trades_table"]["columns"]
        rows = [dict(zip(cols, r)) for r in f.r["trades_table"]["rows"]]
        net = [float(r["net_inr"]) for r in rows if r.get("status") == "closed" and r.get("net_inr") is not None]
        return {"idea": idea, "road": PLAIN, "version": vid, "about": _row(idea, vid).get("about", ""),
                "timeframe": f.judge["settings"]["timeframe"], "trades": net,
                "trades_a_week": float(f.r["power"]["trades_per_week"]),
                "tested_on": f"{RB._where_judged(f)} - the referee's file {f.judge_path}",
                "answer": f"{ans['word']} - {ans['why']}",
                "management": {"kind": PLAIN, "what": "the tester (L3): the form's stop and targets, one trade at a "
                                                      "time, Rs 40 a position all-in"},
                "form_sha256": _sha(form), "signals_sha256": _row(idea, vid).get("signals_sha256"),
                "readings": _row(idea, vid).get("readings"), "changes": _row(idea, vid).get("changes")}
    t = tune.latest(idea)
    h = parked.done(idea, t["result"]["chosen"]) if t is not None else None
    if h is not None:
        r = h["result"]
        head, _ = parked.choice(r, r["his_hedging"])
        if head == "FORWARD TEST":
            net = r["his_hedging"].get("trade_net_inr")
            if not net:
                raise ForwardError(f"{idea}: its test with his hedge ({h['trial_id']}) kept no rupees per trade (made "
                                   "before 9 Oct 2026) - run it again: python -m machine.strategy parked " + idea)
            w = h["settings"]["window"]
            weeks = (dt.datetime.fromisoformat(w["last_bar"]) - dt.datetime.fromisoformat(w["first_bar"])).days / 7
            vid = h["version_id"]
            name = next(Path(config.TRIALS_DIR).glob(f"{h['trial_id']}-*.json")).name
            mgmt = {"kind": HEDGE, "what": "his hedge (L19a / L19d: machine/tester/hedging.py), the rules of the test "
                                           "it passed", "rules": h["settings"]["management_hedging"]}
            return {"idea": idea, "road": HEDGE, "version": vid, "about": _row(idea, vid).get("about", ""),
                    "timeframe": h["settings"]["timeframe"], "trades": [float(x) for x in net],
                    "trades_a_week": len(net) / max(weeks, 1.0),
                    "tested_on": f"{w['years']} with his hedge - trial file plan/work/trials/{name}",
                    "answer": f"with his hedge: {r['his_hedging']['net_inr_total']:,.0f} rupees, better than "
                              f"{r['his_beats_random']['with_hedging']} of {r['runs']} random entries with his hedge",
                    "management": mgmt,
                    "form_sha256": _sha(form), "signals_sha256": h["settings"].get("signals_sha256"),
                    "readings": h["settings"].get("readings"), "changes": h["settings"].get("changes")}
    why = (f"the referee said {ans['word'].upper()}" + (f" and it failed: {', '.join(failed)}" if failed else ""))
    raise ForwardError(f"{idea} is not a forward-test idea: {why} (a forward test is for an idea only new days can "
                       "judge that passed everything else - plan/README.md section 7)")


# ================================================================================================= the numbers

def trades_needed(noise, trades_a_week: float, *, sims: int = config.POWER_SIMS,
                  resamples: int = config.POWER_RESAMPLES, cap: int = config.FORWARD_MAX_TRADES,
                  seed: int = 0) -> dict:
    """The fewest trades at which the referee's luck test catches an edge of Rs POWER_NEEDED_EDGE a trade at least
    POWER_NEEDED_SHARE of the time, planted in the idea's own trades (power.caught_share, the power table's cells) -
    never fewer than MIN_TRADES. Doubles from MIN_TRADES, then halves the gap to about 1 in 100. None = not even at
    `cap` trades."""
    tried: dict[int, float] = {}

    def share(n: int) -> float:
        if n not in tried:
            tried[n] = power.caught_share(noise, config.POWER_NEEDED_EDGE, n, trades_a_week, sims=sims,
                                          resamples=resamples, seed=seed + 7919 * n + config.POWER_NEEDED_EDGE)
        return tried[n]

    def enough(n: int) -> bool:
        return share(n) >= config.POWER_NEEDED_SHARE

    lo, hi = None, config.MIN_TRADES
    while not enough(hi):
        if hi >= cap:
            return {"trades": None, "more_than": cap, "tried": {str(k): round(v, 3) for k, v in sorted(tried.items())}}
        lo, hi = hi, min(cap, 2 * hi)
    if lo is not None:
        while hi - lo > max(1, hi // 100):
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if enough(mid) else (mid, hi)
    return {"trades": hi, "caught": round(share(hi), 3), "tried": {str(k): round(v, 3) for k, v in sorted(tried.items())}}


def dates(written: dt.date, trades: int, trades_a_week: float) -> dict:
    """The first Monday at least FORWARD_START_DAYS days after `written`; the weeks `trades` take at the tested pace;
    the end date = the Friday of week (those weeks x FORWARD_DATE_ROOM), rounded up."""
    start = written + dt.timedelta(days=config.FORWARD_START_DAYS)
    start += dt.timedelta(days=(7 - start.weekday()) % 7)
    weeks = math.ceil(trades / max(trades_a_week, 1e-9))
    room = math.ceil(weeks * config.FORWARD_DATE_ROOM)
    return {"start": start.isoformat(), "weeks_at_pace": weeks, "weeks_at_most": room,
            "end": (start + dt.timedelta(weeks=room) - dt.timedelta(days=3)).isoformat()}


# ================================================================================================= the plan

def path_of(idea: str) -> Path:
    return store.folder(idea) / PLAN_JSON


def read(idea: str) -> dict | None:
    p = path_of(idea)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write(idea: str, *, now: dt.datetime | None = None, log=print, **power_kw) -> dict:
    """Write the idea's forward plan once (forward-plan.json, then forward-plan.md) and return it. A plan already
    there is returned as it is ("already": True) - never rewritten; its words are written again only if missing."""
    old = read(idea)
    if old is not None:
        md = store.folder(idea) / PLAN_MD
        if not md.is_file():
            store._write(md, text(old))
        log(f"{idea}: its forward plan was written already ({old['written']}) - kept as it is")
        return {**old, "already": True}
    now = now or dt.datetime.now().astimezone()
    fx = facts(idea)
    log(f"{idea}: forward plan - {fx['road']}, {len(fx['trades'])} tested trades, "
        f"{fx['trades_a_week']:.2f} a week; counting the trades it needs")
    need = trades_needed(fx["trades"], fx["trades_a_week"], **power_kw)
    n = need["trades"] or need["more_than"]
    lock = {"idea": idea, "version": fx["version"], "about": fx["about"], "timeframe": fx["timeframe"],
            "readings": fx["readings"], "changes": fx["changes"], "form_sha256": fx["form_sha256"],
            "signals_sha256_2016_2020": fx["signals_sha256"], "management": fx["management"],
            "management_sha256": trials.sha256_of(fx["management"]), "rules_sha256": V.RULES_SHA256,
            "code": trials.git_state()["commit"]}
    plan = {"kind": "forward-plan", "idea": idea, "written": now.isoformat(timespec="seconds"), "who": WHO,
            "road": fx["road"], "tested_on": fx["tested_on"], "answer": fx["answer"], "lock": lock,
            "trades_needed": need["trades"], "power": {**need, "edge_inr": config.POWER_NEEDED_EDGE,
                                                       "share": config.POWER_NEEDED_SHARE,
                                                       "noise_from": f"its own {len(fx['trades'])} tested trades",
                                                       "trades_a_week": round(fx["trades_a_week"], 3)},
            **dates(now.date(), n, fx["trades_a_week"]),
            "numbers": {"FORWARD_START_DAYS": config.FORWARD_START_DAYS,
                        "FORWARD_DATE_ROOM": config.FORWARD_DATE_ROOM,
                        "FORWARD_MAX_TRADES": config.FORWARD_MAX_TRADES, "MIN_TRADES": config.MIN_TRADES}}
    store._write(path_of(idea), json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    store._write(store.folder(idea) / PLAN_MD, text(plan))
    log(f"{idea}: forward plan written - {n} trades, start {plan['start']}, end {plan['end']}")
    return plan


def _day(iso: str) -> str:
    return f"{dt.date.fromisoformat(iso):%a %d %b %Y}"


def text(p: dict) -> str:
    """forward-plan.md: the plan in words."""
    lk, pw = p["lock"], p["power"]
    n = p["trades_needed"]
    count = n or pw["more_than"]
    need =(f"**{n} trades.**" if n else f"**more than {pw['more_than']:,} trades** - more than this test can wait "
            "for: on the end date it is judged with what there is, and the referee will most likely say "
            f"\"{V.FEW}\".")
    return "\n".join([
        f"# {p['idea']} - the forward plan (L17c)", "",
        f"**Written:** {p['written']} by {p['who']}, BEFORE any live bar. Never rewritten: a change to anything "
        "below is a new test with a new plan (likhith.md L17c).", "",
        "## Why a forward test", "",
        f"- {p['answer']}.",
        f"- Tested on: {p['tested_on']}.",
        "- Its tested years cannot judge it again, and 2024-2026 stay sealed (LEARNED #96; the final exam needs his "
        "yes again on that day). Only new days can.", "",
        "## Locked (the pretend trades stop if any of this differs)", "",
        f"- Idea {lk['idea']}, version `{lk['version']}`: {lk['about']}",
        f"- Chart: {lk['timeframe']} · readings {json.dumps(lk['readings'], ensure_ascii=False)} · changes "
        f"{json.dumps(lk['changes'], ensure_ascii=False)}",
        f"- form.json fingerprint `{lk['form_sha256']}`",
        f"- Its setups on 2016-2020 (the same compile must give them again): `{lk['signals_sha256_2016_2020']}`",
        f"- Trades closed by: {lk['management']['what']} - fingerprint `{lk['management_sha256']}`",
        f"- The referee's rules: fingerprint `{lk['rules_sha256']}` (below) · code at commit `{lk['code']}`",
        "- 0.01 lot · Rs 40 per position, all-in (LEARNED #54, #115) · one trade at a time · MT5 server time · "
        "pretend trades on live prices only (T14 use 1) - no money.", "",
        "## How many trades", "",
        f"{need} The referee's own rule: with this many trades an edge of Rs {pw['edge_inr']} a trade is caught at "
        f"least {round(100 * pw['share'])} times in 100 (never fewer than {p['numbers']['MIN_TRADES']}) - counted "
        f"in {pw['noise_from']}, moved to no edge (machine/referee/power.py)."
        + (f" At {n}: caught {round(100 * pw['caught'])} times in 100." if n else ""),
        f"Counted at: {', '.join(f'{k} trades {round(100 * v)} in 100' for k, v in pw['tried'].items())}.", "",
        "## When", "",
        f"- **Starts {_day(p['start'])} at the earliest** - the first Monday at least "
        f"{p['numbers']['FORWARD_START_DAYS']} days after this plan was written, and only if, before that day's first "
        "bar, git shows this plan on main AND its pretend trades (L17b) are running. If not, it starts on the first "
        "Monday when both are true, and the end date moves by the same whole weeks. No live bar before the start "
        "counts.",
        f"- At its tested pace ({pw['trades_a_week']} trades a week) {count:,} trades take about "
        f"{p['weeks_at_pace']} weeks.",
        f"- **Judged ONCE, at the first of:** {count:,} closed trades, or **{_day(p['end'])}** (those weeks x "
        f"{p['numbers']['FORWARD_DATE_ROOM']}, room for a slower market). Nothing is judged before.", "",
        "## The checks at the end (the same verdict rules; the new days stand in for the check years)", "",
        "```", V.RULES, "```", "",
        "Check 6 (\"not already seen\") passes by itself: the new days were never seen. The numbers marked Claude's "
        f"here (FORWARD_START_DAYS {p['numbers']['FORWARD_START_DAYS']}, FORWARD_DATE_ROOM "
        f"{p['numbers']['FORWARD_DATE_ROOM']}, FORWARD_MAX_TRADES {p['numbers']['FORWARD_MAX_TRADES']:,}) are "
        "Claude's, not Tejas's rules (machine/config.py).", ""])


def telegram_text(p: dict, name: str) -> str:
    """The one line he gets (it tells, never asks - LEARNED #162). No rupees: nothing was tested here."""
    n = p["trades_needed"]
    many = f"{n:,} trades" if n else f"more than {p['power']['more_than']:,} trades"
    a_week = p["power"]["trades_a_week"]
    return "\n".join([
        f"{name}: its forward test is planned and locked (L17c). Its rules and its version cannot change now.",
        f"- It needs {many} on new days - about {p['weeks_at_pace']} weeks (it made about {a_week:.1f} trades a "
        "week in the years it was tested on).",
        f"- It starts {_day(p['start'])} at the earliest. It is judged once: at {many} or on {_day(p['end'])}, "
        "whichever comes first, by the same rules as every idea.",
        "- Pretend trades on live prices only. No money.",
        "- The pretend trades themselves (L17b) are not built yet. Until they run, the test waits, and both dates "
        "move by the same weeks.",
        "You don't need to do anything."])
