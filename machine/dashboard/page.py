"""The dashboard's one page (L16.2), drawn from read.snapshot(): easy English (LEARNED #113), every text escaped, no
script at all, light and dark colours. The colours follow the status palette (good / warning / bad, always with an
icon and a word - never the colour alone)."""
from __future__ import annotations

import datetime as dt
import html

from machine.dashboard import read

REFRESH_SECONDS = 60
ICON = {"good": "✓", "warning": "!", "bad": "✕", "unknown": "?", "waiting": "…"}
KINDS = {"tester": "single test runs", "belief": "his beliefs counted", "money-plan": "money plans",
         "hedging-2x2": "hedge tests", "strategy": "his whole strategy's runs", "referee-tune": "choices among versions",
         "referee": "answers", "learner": "condition tables", "learner-model": "learned-model checks",
         "learner-pooled": "pooled checks", "referee-control": "control checks", "strategy-2x2": "his strategy's hedge "
         "test"}                                       # the trial files' kinds, in words (machine/trials.py README)
WORD = {"good": "OK", "warning": "Look", "bad": "Problem", "unknown": "Not known", "waiting": "Waiting"}

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--hair:#e1e0d9;--ring:rgba(11,11,11,.10);--good:#0ca30c;--warning:#fab219;--bad:#d03b3b;--soft:#e1e0d9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--hair:#2c2c2a;--ring:rgba(255,255,255,.10);
--soft:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--hair:#2c2c2a;--ring:rgba(255,255,255,.10);--soft:#383835}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:28px 0 8px}
h3{font-size:15px;margin:16px 0 6px;color:var(--ink2)}
p{margin:4px 0}
.sub,.note,small{color:var(--ink2)}
.note{font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:12px;margin-top:16px}
.tile,.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px}
.tile{padding:12px 14px}
.tile .name{color:var(--ink2);font-size:13px}
.tile .value{display:flex;gap:8px;align-items:center;font-size:17px;font-weight:600;margin:6px 0 4px}
.tile .line{font-size:13px;color:var(--ink2)}
.dot{flex:none;display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;
font-size:12px;font-weight:700;color:#fff;background:var(--muted)}
.dot.good{background:var(--good)}.dot.warning{background:var(--warning);color:#0b0b0b}.dot.bad{background:var(--bad)}
.card{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--hair);vertical-align:top}
tr:last-child td{border-bottom:0}
th{color:var(--ink2);font-weight:600;font-size:13px;white-space:nowrap}
td.n{white-space:nowrap}
.state{font-weight:600;white-space:nowrap}
.steps{margin:0;padding:0;list-style:none}
.steps li{margin:2px 0}
ul.plain{margin:0;padding:10px 14px 10px 30px}
ul.plain li{margin:3px 0}
.strip{display:grid;grid-template-columns:repeat(24,minmax(0,1fr));gap:2px;padding:14px 14px 4px}
.cell{height:24px;border-radius:4px;background:var(--soft)}
.cell.beat{background:var(--good)}.cell.none{background:var(--bad)}
.ticks{display:grid;grid-template-columns:repeat(24,minmax(0,1fr));gap:2px;padding:0 14px 10px;font-size:12px;
color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;padding:0 14px 12px;font-size:13px;color:var(--ink2)}
.key{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;
background:var(--soft)}
.key.beat{background:var(--good)}.key.none{background:var(--bad)}
.ok{color:var(--good);font-weight:700}.no{color:var(--bad);font-weight:700}
footer{margin-top:32px;font-size:13px;color:var(--ink2)}
"""


def e(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def inr(x) -> str:
    if x is None:
        return "no trades"
    return f"-₹{-x:,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def _t(iso, t: dt.datetime) -> str:
    return read._hm(read.when(iso), t) if iso else "?"


def _since(iso, t: dt.datetime) -> str:
    at = read.when(iso)
    if at is None:
        return "?"
    days = (t.date() - at.date()).days
    return f"{at:%d %b}" + (f" ({days} day{'s' if days != 1 else ''})" if days else " (today)")


def _failed(part) -> str | None:
    return part.get("failed") if isinstance(part, dict) else None


# ============================================================================================================ the parts

def tiles(snap: dict) -> str:
    out = []
    for x in snap.get("now") or []:
        st = x["state"]
        out.append(f'<div class="tile"><div class="name">{e(x["name"])}</div><div class="value">'
                   f'<span class="dot {e(st)}" role="img" aria-label="{e(WORD.get(st, st))}">{e(ICON.get(st, "?"))}'
                   f'</span>{e(x["value"])}</div><div class="line">{e(x["line"])}</div></div>')
    return '<section class="tiles" aria-label="Right now">' + "".join(out) + "</section>"


def ideas(snap: dict, t: dt.datetime) -> str:
    part = snap.get("ideas") or {}
    head = "<h2>Ideas</h2>"
    if _failed(part):
        return head + f'<p class="note">Could not be read: {e(part["failed"])}</p>'
    rows = []
    for r in part.get("rows") or []:
        name = e(r["name"]) + (" <small>(runs now)</small>" if r["running"] else "")
        words = f'<b>{e(r["title"])}</b><br>' if r["title"] else ""
        words += f'<small>"{e(r["words"])}"</small>' if r["words"] else ""
        wait = e(r["waits"]) + (' <span class="no">- needs a Claude session</span>' if r["needs_session"] else "")
        if r.get("decided"):
            wait = f'<small>The machine\'s choice:</small> {e(r["decided"])}'
        rows.append(f'<tr><td class="n">{name}</td><td>{words}</td><td class="state">{e(r["state"])}</td>'
                    f'<td class="n">{e(_since(r["since"], t))}</td><td>{wait}</td></tr>')
    body = ('<div class="card"><table><thead><tr><th>Idea</th><th>His words</th><th>State</th><th>Since</th>'
            '<th>Waits for</th></tr></thead><tbody>' + ("".join(rows) or
            '<tr><td colspan="5">No idea yet.</td></tr>') + "</tbody></table></div>")
    extra = []
    if part.get("queue"):
        extra.append(f'{part["queue"]} new idea(s) wait their turn in the queue (the oldest since '
                     f'{e(_since(part["queue_since"], t))}). One idea runs at a time.')
    if part.get("paused"):
        extra.append(e(part["paused"][0].upper() + part["paused"][1:]) + ".")
    return head + body + "".join(f'<p class="note">{x}</p>' for x in extra)


def versions(snap: dict, t: dt.datetime) -> str:
    part = snap.get("versions") or {}
    head = "<h2>Versions tried</h2>"
    if _failed(part):
        return head + f'<p class="note">Could not be read: {e(part["failed"])}</p>'
    names = {r["id"]: r["name"] for r in (snap.get("ideas") or {}).get("rows") or []}
    rows = []
    for r in part.get("rows") or []:
        if r.get("failed"):
            rows.append(f'<tr><td class="n">{e(names.get(r["id"], r["id"]))}</td><td colspan="3"><small>Could not be '
                        f'counted: {e(r["failed"])}</small></td></tr>')
            continue
        tried = f'{r["tried"]} of {r["budget"]}' + (f' <small>(+{r["waiting"]} waiting)</small>' if r["waiting"]
                                                     else "")
        if r.get("family"):
            tried += f'<br><small>family of {e(r["family"])}</small>'
        c = r.get("chosen")
        if c:
            chosen = (f'<b>{e(c["version"])}</b> <small>(one of {c["of"]})</small><br>{_numbers(c)}'
                      + (" <small>with his hedge</small>" if c["hedged"] else ""))
        else:
            chosen = "<small>none chosen yet</small>"
        rows.append(f'<tr><td class="n">{e(names.get(r["id"], r["id"]))}</td><td class="n">{tried}</td>'
                    f'<td>{chosen}</td><td>{_answer(r.get("answer"))}{_hedge(r.get("hedge"))}</td></tr>')
    for a in part.get("others") or []:
        rows.append(f'<tr><td class="n">{e(_other_name(a["name"]))}</td><td class="n">-</td><td>-</td>'
                    f'<td>{_answer(a)}</td></tr>')
    body = ('<div class="card"><table><thead><tr><th>Idea</th><th>Versions tried</th>'
            '<th>The referee\'s choice, on 2016-2020</th><th>The answer, on the check years</th></tr></thead><tbody>'
            + ("".join(rows) or '<tr><td colspan="4">No version tried yet.</td></tr>') + "</tbody></table></div>")
    notes = ['Rupees at 0.01 lot, after ₹40 per position all-in (the spread included). Each version is one try; '
             'at most 20 per idea. The check years 2021-2023 open once per idea.']
    p = part.get("project") or {}
    if p:
        by = p.get("by_word") or {}
        notes.append(f'The whole project: {p.get("answers", 0)} answers - ' + " · ".join(
            f'{e(w)} {by.get(w, 0)}' for w in read.ANSWER_WORDS) + f'. Luck alone would give about '
            f'{p.get("false_works_expected", 0):g} "works" by now. The check years were opened '
            f'{p.get("check_opened", 0)} times in the whole project.')
    kinds = part.get("kinds") or {}
    notes.append(f'Test records: {part.get("files", 0)} trial files (' + " · ".join(
        f"{e(KINDS.get(k, k))} {n}" for k, n in kinds.items()) + "). MLflow holds one run per file; the files stay "
        "the count.")
    return head + body + "".join(f'<p class="note">{x}</p>' for x in notes)


def _other_name(idea: str) -> str:
    return {"his-strategy": "his whole strategy"}.get(idea, idea)


def _numbers(n: dict) -> str:
    w = n.get("worst_year") or {}
    worst = f', worst year {e(w.get("year"))} {inr(w.get("net_inr"))}' if w else ""
    return (f'{n.get("trades") or 0} trades, {inr(n.get("per_trade"))} a trade, total {inr(n.get("total"))}'
            f'{worst}')


def _hedge(h: dict | None) -> str:
    if not h:
        return ""
    beat = (f'; it beat {h["beats"]} of {h["runs"]} random entries with the same hedge' if h.get("beats") is not None
            and h.get("runs") else "")
    choice = f' <b>{e(h["choice"])}</b>' if h.get("choice") else ""
    return (f'<br><small>With his hedge instead of a stop ({e(h.get("years") or "2016-2020")}): {_numbers(h)}{beat}.'
            f'</small>{choice}')


def _answer(a: dict | None) -> str:
    if not a:
        return "<small>not judged yet</small>"
    years = f' <small>({e(a["years"])})</small>' if a.get("years") else ""
    why = f'<br><small>{e(a["why"])}</small>' if a.get("why") else ""
    return f'<b>{e(str(a.get("word") or "?").upper())}</b>{years}<br>{_numbers(a)}{why}'


def runs(snap: dict, t: dt.datetime) -> str:
    part = snap.get("runs") or {}
    head = "<h2>Last runs</h2>"
    if _failed(part):
        return head + f'<p class="note">Could not be read: {e(part["failed"])}</p>'
    rows = []
    for r in part.get("daily") or []:
        steps = "".join(f'<li><span class="{"ok" if s["ok"] else "no"}">{"✓" if s["ok"] else "✕"}</span> '
                        f'{e(s["line"] or s["name"])}</li>' for s in r["steps"])
        result = ('<span class="ok">✓ all steps worked</span>' if r["ok"] else '<span class="no">✕ FAILED</span>')
        if r["test"]:
            result += f'<br><small>a TEST: "{e(r["test"])}" failed on purpose</small>'
        rows.append(f'<tr><td class="n">{e(_t(r["at"], t))}</td><td class="n">{result}</td>'
                    f'<td><ul class="steps">{steps}</ul></td></tr>')
    daily = ('<h3>The daily run (every day at 07:00 on Tejas\'s PC, L16.1), last 7 days</h3><div class="card"><table>'
             '<thead><tr><th>When</th><th>Result</th><th>Its steps</th></tr></thead><tbody>'
             + ("".join(rows) or '<tr><td colspan="3">No daily run on this computer.</td></tr>')
             + "</tbody></table></div>")
    intake = "".join(f'<li>{e(_t(x["at"], t))} - {e(x["text"])}</li>' for x in part.get("intake") or [])
    intake = ('<h3>The ideas\' last changes (the intake, L10b)</h3><div class="card"><ul class="plain">'
              + (intake or "<li>Nothing on this computer.</li>") + "</ul></div>")
    nightly = []
    for n in part.get("nightly") or []:
        what = (f'<span class="no">✕ FAILED</span>: {e(n["error"])}' if n["error"] else
                f'{e(n["branch"])} · {n["files"]} file(s)' + (" · the link went to Tejas" if n["link"] else ""))
        nightly.append(f'<li>{e(_t(n["at"], t))} - {what}' + (" <small>(a test run)</small>" if n["test"] else "")
                       + "</li>")
    nightly_html = ('<h3>The nightly copy of the bot\'s files to GitHub (23:30, L9c)</h3><div class="card">'
                    '<ul class="plain">' + ("".join(nightly) or "<li>No run on this computer.</li>") + "</ul></div>")
    bot = ""
    if part.get("bot_started"):
        bot = f'<p class="note">The Telegram bot last started {e(_t(part["bot_started"], t))}.</p>'
    return head + daily + intake + nightly_html + bot


def heartbeat(snap: dict, t: dt.datetime) -> str:
    part = snap.get("heartbeat") or {}
    head = "<h2>Heartbeat</h2>"
    if _failed(part):
        return head + f'<p class="note">Could not be read: {e(part["failed"])}</p>'
    if not part.get("count"):
        return head + ('<p class="note">No heartbeat on this computer. It runs on Tejas\'s PC every hour (L16.1) and '
                       'writes data/host/heartbeat.jsonl.</p>')
    cells, ticks = [], []
    for i, c in enumerate(part.get("strip") or []):
        start = read.when(c["from"])
        label = {"beat": f'{c["beats"]} beat(s)', "none": "no beat", "waiting": "its beat comes at :15",
                 "before": "before the first beat"}[c["kind"]]
        cells.append(f'<div class="cell {e(c["kind"])}" title="{start:%a %H}:00 - {label}"></div>')
        ticks.append(f"<span>{start:%H}:00</span>" if i % 6 == 0 else "<span></span>")
    last = part.get("last") or {}
    bot = {True: "the bot running", False: "the bot NOT running", None: "the bot not known"}[last.get("bot")]
    probs = ", ".join(read.PROBLEMS.get(p, p) for p in last.get("problems") or []) or "no problems"
    disk = f', {last["disk_free_gb"]} GB free' if last.get("disk_free_gb") is not None else ""
    strip = (f'<div class="card"><div class="strip" aria-hidden="true">{"".join(cells)}</div>'
             f'<div class="ticks" aria-hidden="true">{"".join(ticks)}</div>'
             '<div class="legend"><span><i class="key beat"></i>a beat came</span><span><i class="key none"></i>no beat '
             '(the PC was off, or the heartbeat did not run)</span><span><i class="key"></i>not yet</span></div></div>'
             f'<p class="note">The last {read.STRIP_HOURS} hours, one box per hour. The last beat: '
             f'{e(_t(last.get("at"), t))} - {e(bot)}{e(disk)}, {e(probs)}. Outside watcher: {e(last.get("ping"))}.</p>')
    missed = []
    for m in part.get("missed") or []:
        offs = "; ".join(f'{_t(o["off"], t) if o.get("off") else "?"} to {_t(o["on"], t)}, {o["why"]}'
                         for o in m["offs"])
        missed.append(f'<li>{e(_t(m["since"], t))} to {e(_t(m["at"], t))}: no beat for {e(m["minutes"])} min.'
                      + (f" Windows says the PC was off: {e(offs)}." if offs else
                         " Windows says the PC did not stop: the heartbeat itself did not run.") + "</li>")
    missed_html = ('<h3>Missed beats, last 7 days</h3><div class="card"><ul class="plain">'
                   + ("".join(missed) or "<li>None.</li>") + "</ul></div>")
    opened = "".join(f'<li>{e(p["what"])} - since {e(_t(p["since"], t))}'
                     + ("" if p.get("told") else " (not told yet: no network then)") + "</li>"
                     for p in part.get("open") or [])
    open_html = ('<h3>Open problems (told once when each starts and once when it ends)</h3><div class="card">'
                 '<ul class="plain">' + (opened or "<li>None.</li>") + "</ul></div>")
    return head + strip + missed_html + open_html


def errors(snap: dict, t: dt.datetime) -> str:
    items = snap.get("errors") or []
    rows = "".join(f'<tr><td class="n">{e(_t(x["at"], t))}</td><td class="n">{e(x["where"])}'
                   + (f'<br><small>{e(x["kind"])}</small>' if x.get("kind") else "")
                   + f'</td><td>{e(x["text"])}</td></tr>' for x in items)
    return ('<h2>Alerts and errors, last 7 days</h2><div class="card"><table><thead><tr><th>When</th><th>Where</th>'
            '<th>What</th></tr></thead><tbody>' + (rows or '<tr><td colspan="3">None.</td></tr>')
            + '</tbody></table></div><p class="note">In a message\'s text every digit is hidden (#): the live alerts '
            'write here too, and their prices are never shown on this page.</p>')


# ============================================================================================================ the page

def html_page(snap: dict) -> str:
    t = read.when(snap["made_at"]) or read.now()
    body = (f'<main><header><h1>The idea machine</h1><p class="sub">How it is doing. Read only: this page changes '
            f'nothing. India time. It refreshes every minute.</p></header>{tiles(snap)}{ideas(snap, t)}'
            f'{versions(snap, t)}{runs(snap, t)}{heartbeat(snap, t)}{errors(snap, t)}'
            f'<footer>Made {t:%a %d %b %H:%M} from the files in "{e(snap.get("root"))}". It never shows a live price '
            "or anything from 2024 on (hard rule 1): the live alerts' own files are not read at all.</footer></main>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}"><title>Idea machine dashboard</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
