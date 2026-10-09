"""The daily check of the live feed (L17a: "live MT5 bars in SERVER time + a daily check against history").

For each finished server day: the 1-minute bars the live passes saw (LIVE_TAPE_DIR/<day>.csv, machine/live/prices.py)
against MT5's own history of that day, kept by the daily run (L16.1: data/sealed/mt5/M1/<day>.csv). Both are L1's own
text, so they are compared as TEXT, line by line - no price is computed on or shown here (one is parsed only to see
whether a live high and low lie inside the history's, below). One line per day in LIVE_DIR/checks.jsonl:
    history  bars in MT5's history of the day
    seen     bars the passes saw live; same = the very same line
    other    bars seen with another price (open, high, low or close) than the history - the alerts saw other prices
             than the history the tests use: the one case that tells Tejas
    partial  of those, bars the live feed got only PART of the ticks of: fewer ticks than the history, and its high
             and low inside the history's (Tue 6 Oct 2026: MT5 lost its login mid-minute, 159 of 210 ticks; Thu 8 Oct:
             the last 5 of a busy minute's 936 ticks came after the pass read it - not a login loss)
    minor    the same prices, another tick count or spread (no alert reads those)
    missed   bars of the history no pass saw live (the PC was off, MT5 was not connected, a pass came late)
    extra    bars seen live that the history does not have
A day before the first tape day is not checked (the tape did not exist yet). Reading the kept day is T14 use 1 too
(live-trial: the alerts' own feed), logged with the week's opening (machine/live/opened.py).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from machine import config
from machine.host import alert as HA
from machine.live import opened

WHO = "machine.live (L17a, the daily check of the live feed)"
WHY = "the live 1-minute bars against MT5's history of the same day, compared as text (T14 use 1)"


def _lines(path: Path) -> dict[str, str]:
    """{time text: the whole line} of an L1-format day file (the header left out)."""
    out = {}
    for raw in Path(path).read_text(encoding="ascii").splitlines()[1:]:
        if raw.strip():
            out[raw.split(",", 1)[0]] = raw
    return out


def _part_of(mine: str, line: str) -> bool:
    """True when the live bar holds only part of the history bar's ticks: fewer ticks (L1's vol column), and its high
    and low inside the history's - what a bar built from some of the same ticks looks like. Nothing is kept."""
    a, b = mine.split(","), line.split(",")
    try:
        return int(a[5]) < int(b[5]) and float(a[2]) <= float(b[2]) and float(a[3]) >= float(b[3])
    except (IndexError, ValueError):
        return False


def compare(tape: Path, history: Path) -> dict:
    """The counts of one day (see the module text) and the first few times of each kind of difference."""
    seen = _lines(tape) if Path(tape).is_file() else {}
    hist = _lines(history)
    same = other = partial = minor = 0
    first_missed, other_at = [], []
    for t, line in hist.items():
        mine = seen.get(t)
        if mine is None:
            if len(first_missed) < 3:
                first_missed.append(t[11:16])
            continue
        if mine == line:
            same += 1
        elif mine.split(",")[1:5] != line.split(",")[1:5]:
            other += 1
            partial += _part_of(mine, line)
            other_at.append(t[11:16])
        else:
            minor += 1
    missed = sum(1 for t in hist if t not in seen)
    return dict(history=len(hist), seen=len(seen), same=same, other=other, partial=partial, minor=minor,
                missed=missed, extra=sum(1 for t in seen if t not in hist), first_other=other_at[:3],
                first_missed=first_missed,
                other_at=other_at)                       # every such minute (server HH:MM): explained() reads them


def _done(folder: Path) -> set[str]:
    return {r.get("day") for r in HA.read_jsonl(folder / "checks.jsonl")}


def pending(folder: Path | None = None, tape_dir: Path | None = None, kept_dir: Path | None = None) -> list[str]:
    """The days to check now: kept by the daily run, on or after the first tape day, not checked yet."""
    folder = Path(folder or config.LIVE_DIR)
    tapes = sorted(p.stem for p in Path(tape_dir or config.LIVE_TAPE_DIR).glob("*.csv"))
    if not tapes:
        return []
    kept = Path(kept_dir or Path(config.SEALED_MT5_DIR) / "M1")
    done = _done(folder)
    return [p.stem for p in sorted(kept.glob("*.csv")) if p.stem >= tapes[0] and p.stem not in done]


def run(*, folder: Path | None = None, tape_dir: Path | None = None, kept_dir: Path | None = None,
        tell=HA.alert) -> list[dict]:
    """Check every pending day; write checks.jsonl; tell Tejas only when the live prices differed from the history."""
    folder = Path(folder or config.LIVE_DIR)
    tape_dir = Path(tape_dir or config.LIVE_TAPE_DIR)
    kept_dir = Path(kept_dir or Path(config.SEALED_MT5_DIR) / "M1")
    out = []
    for day in pending(folder, tape_dir, kept_dir):
        d = dt.date.fromisoformat(day)
        opened.week(pd.DatetimeIndex([pd.Timestamp(day)]), d - dt.timedelta(days=(d.weekday() + 1) % 7),
                    use="live-trial", who=WHO, why=WHY)
        rec = {"day": day, "at": HA.iso(HA.now()), **compare(tape_dir / f"{day}.csv", kept_dir / f"{day}.csv")}
        HA.append_jsonl(folder / "checks.jsonl", rec)
        if rec["other"] and not explained(rec, folder):            # explained: the day's report says it, calmly
            tell(f"⚠️ Live prices check, {d:%a %d %b}: on {rec['other']} minute(s) the live alerts saw other prices than "
                 f"MT5's history of that day (first at {', '.join(rec['first_other'])} server time). The alerts that "
                 "day may have used wrong prices. Nothing for you to do: it is written down for the next session "
                 "(data/live/checks.jsonl).")
        out.append(rec)
    return out


def words(rec: dict) -> str:
    """One short line for STATUS."""
    d = dt.date.fromisoformat(rec["day"])
    seen = rec["history"] - rec["missed"]
    s = f"{d:%a %d %b}: {seen:,} of {rec['history']:,} minutes seen live"
    if rec["other"]:
        return s + f", {rec['other']} with OTHER prices than the history"
    return s + (", every one the same as the history" if seen else "")


# ===================================================================================================== the day's report

MT5_TROUBLE = ("no live prices", "live prices failed")
TROUBLE = ("FAILED", "could not run", "the daily check failed", "report failed")


def _server_days(times: list[str]) -> list[str]:
    """The server day (machine/data/clock.py) of each ISO time with its offset ("" for one that is not a time)."""
    from machine.data import clock
    out = []
    for t in times:
        try:
            ts = pd.Timestamp(t)
            out.append("" if ts is pd.NaT or ts.tzinfo is None else
                       clock.utc_to_server(pd.DatetimeIndex([ts.tz_convert("UTC")]))[0].strftime("%Y-%m-%d"))
        except (ValueError, TypeError):
            out.append("")
    return out


def _india(server_times: list[str]) -> list[str]:
    """HH:MM India time of server times (YYYY-MM-DD HH:MM...)."""
    if not server_times:
        return []
    from machine.data import clock
    return [f"{HA.india(t.to_pydatetime()):%H:%M}" for t in clock.server_to_utc(pd.DatetimeIndex(server_times))]


def _near(day: str) -> set[str]:
    """The UTC / India dates a server day lies in."""
    d = dt.date.fromisoformat(day)
    return {(d + dt.timedelta(days=k)).isoformat() for k in (-1, 0, 1)}


def _sent_of(day: str, folder: Path) -> list[dict]:
    """The alerts queued on this server day (data/live/sent.jsonl)."""
    sent = [r for r in HA.read_jsonl(folder / "sent.jsonl") if str(r.get("at", ""))[:10] in _near(day)]
    return [r for r, sd in zip(sent, _server_days([r.get("at", "") for r in sent])) if sd == day]


def _log_of(day: str, folder: Path) -> list[str]:
    """The lines of live.log written on this server day (their text, after the time)."""
    log = folder / "live.log"
    raw = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
    logged = [ln.split(" ", 1) for ln in raw if " " in ln and ln[:10] in _near(day)]
    return [x for (_, x), sd in zip(logged, _server_days([t for t, _ in logged])) if sd == day]


def explained(rec: dict, folder: Path | None = None) -> bool:
    """True when the day's minutes with OTHER prices need no one's look: the live feed got only part of each one's
    ticks (`partial`, compare) AND no alert (news or big move) was queued within config.LIVE_OTHER_NEAR_MINUTES after
    any of those minutes, so no alert read them. A check written before 9 Oct 2026 has no `partial`: there MT5 must
    have said it lost its connection that day instead (a "no live prices" line in live.log - Tue 6 and Wed 7 Oct). A
    day whose every such minute is not known (a check from before 8 Oct kept only the first 3) is never explained."""
    if not rec.get("other"):
        return True
    folder = Path(folder or config.LIVE_DIR)
    day = rec["day"]
    at = rec.get("other_at") or rec.get("first_other") or []
    if len(at) < rec["other"]:
        return False
    if "partial" in rec:
        if rec["partial"] < rec["other"]:
            return False
    elif not any(any(k in x for k in MT5_TROUBLE) for x in _log_of(day, folder)):
        return False
    from machine.data import clock
    bad = clock.server_to_utc(pd.DatetimeIndex([f"{day} {t}" for t in at]))
    near = pd.Timedelta(minutes=config.LIVE_OTHER_NEAR_MINUTES)
    for r in _sent_of(day, folder):
        if r.get("kind") not in ("before", "after", "move"):
            continue
        q = pd.Timestamp(r["at"]).tz_convert("UTC")
        if any(b <= q <= b + near for b in bad):
            return False
    return True


def day_report(rec: dict, folder: Path | None = None, *, bot_sent: Path | None = None,
               tape_dir: Path | None = None) -> str:
    """The day's report for his phone, sent once right after the day's check (machine/live/run.locked_once). 6 Oct
    2026: he cannot keep a Claude chat open for the daily checks, and Claude Code refuses to start a Claude agent no
    one watches - so the machine says them itself: the news alerts of that server day and whether the bot sent them,
    the big moves, the check (minutes seen live, the same as MT5's history or not), MT5 trouble, failed lines in
    live.log, the trading days so far, the next news. Counts and times only - no price."""
    from machine.news import similar
    folder = Path(folder or config.LIVE_DIR)
    day = rec["day"]
    d = dt.date.fromisoformat(day)
    sent = _sent_of(day, folder)
    news = [r for r in sent if r.get("kind") in ("before", "after")]
    moves = [r for r in sent if r.get("kind") == "move"]
    bot = {r.get("id") for r in HA.read_jsonl(Path(bot_sent or Path(config.TELEGRAM_DIR) / "sent.jsonl"))}
    unsent = [r for r in news + moves if r.get("sent") == "queued" and r.get("id") not in bot]
    held = [r for r in news + moves if str(r.get("sent", "")).startswith("not sent")]
    lines = [f"📋 Live alerts, {d:%a %d %b} - the daily check"]
    if news:
        by: dict[str, list[str]] = {}
        for r in news:
            by.setdefault(similar.name_of(r.get("report") or "?")[1], []).append(r["kind"])
        what = "; ".join(f"{name}: {' and '.join(dict.fromkeys(kinds))}" for name, kinds in by.items())
        lines.append(f"• News alerts: {len(news)} - {what}.")
    else:
        lines.append("• News alerts: none (none of your news that day).")
    if moves:
        lines.append(f"• Big moves with no news: {len(moves)}, at {', '.join(_india([r['bar'] for r in moves]))} "
                     "India time.")
    else:
        lines.append("• Big moves with no news: none.")
    if unsent:
        lines.append(f"⚠️ {len(unsent)} alert(s) queued but NOT sent by the bot.")
    elif news or moves:
        lines.append("• The bot sent every one." if not held else "• The bot sent every one it was given.")
    if held:
        lines.append(f"• {len(held)} held back because you had typed STOP.")
    seen = rec["history"] - rec["missed"]
    calm = explained(rec, folder)
    if rec["other"] and calm and "partial" in rec:
        at = _india([f"{day} {t}" for t in rec.get("first_other") or []])
        one = rec["other"] == 1
        lines.append(f"• Prices: {seen - rec['other']:,} of the {seen:,} minutes seen live had the same prices as MT5's "
                     f"own history; {rec['other']} {'was' if one else 'were'} not complete (first at {', '.join(at)} "
                     f"India time) - the live feed got only part of {'its' if one else 'their'} ticks, and no alert "
                     f"was made within {config.LIVE_OTHER_NEAR_MINUTES} minutes after {'it' if one else 'them'}, so no "
                     f"alert read {'it' if one else 'them'}.")
    elif rec["other"] and calm:
        at = _india([f"{day} {t}" for t in rec.get("first_other") or []])
        lines.append(f"• Prices: {seen - rec['other']:,} of the {seen:,} minutes seen live had the same prices as MT5's "
                     f"own history; {rec['other']} were not complete (first at {', '.join(at)} India time) - MT5 lost its "
                     f"connection that day, and no alert was made within {config.LIVE_OTHER_NEAR_MINUTES} minutes "
                     "after them, so no alert read them.")
    elif rec["other"]:
        at = _india([f"{day} {t}" for t in rec.get("first_other") or []])
        lines.append(f"⚠️ Prices: on {rec['other']} minute(s) the alerts saw OTHER prices than MT5's own history "
                     f"(first at {', '.join(at)} India time).")
    elif seen:
        lines.append(f"• Prices: the alerts saw {seen:,} of {rec['history']:,} minutes live - every one the same as "
                     "MT5's own history.")
    else:
        lines.append(f"• Prices: the alerts saw none of the day's {rec['history']:,} minutes live.")
    if rec["missed"] and seen:
        lines.append(f"• Minutes not seen live: {rec['missed']:,} (the PC was off, the live job was not running yet, "
                     "or MT5 did not answer).")
    texts = _log_of(day, folder)
    mt5 = sum(1 for x in texts if any(k in x for k in MT5_TROUBLE))
    bad = [x for x in texts if any(k in x for k in TROUBLE) and not any(k in x for k in MT5_TROUBLE)]
    if mt5:
        lines.append(f"• MT5 did not answer: {mt5} time(s) (each pass after it tried again).")
    if bad:
        lines.append(f"⚠️ {len(bad)} failed line(s) in the live log, first: {bad[0][:120]}")
    tapes = sorted(p.stem for p in Path(tape_dir or config.LIVE_TAPE_DIR).glob("*.csv"))
    if tapes:
        lines.append(f"Trading days the live alerts have run: {sum(1 for t in tapes if t <= day)} (since "
                     f"{dt.date.fromisoformat(tapes[0]):%a %d %b}).")
    try:
        nxt = json.loads((folder / "state.json").read_text(encoding="utf-8")).get("next")
    except (OSError, ValueError):
        nxt = None
    if nxt:
        t = pd.Timestamp(nxt["utc"]).tz_convert("Asia/Kolkata")
        lines.append(f"Next news alert: {similar.name_of(nxt['report'])[1]} ({nxt['colour']}), {t:%a %d %b %H:%M}.")
    look = (rec["other"] and not calm) or unsent or bad
    lines.append("⚠️ Something above needs a look: open a Claude chat and paste your L17a prompt." if look
                 else "Nothing for you to do.")
    return "\n".join(lines)
