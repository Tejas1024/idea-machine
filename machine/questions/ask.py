"""Make the questions for one unclear part of an idea (L9b steps 1-3). Nothing is sent here: the weekly batch
(machine/questions/batch.py) sends them.

    python -m machine.questions ask idea-001            every open ASK part of the idea
    python -m machine.questions ask idea-001 --part U2

For one ASK part (a meaning only he knows; a VERSION part is never asked - the machine tries it):
  1. The #129 search first (machine.search.answered, on the part's own words). FOUND -> no question; the part record
     cites where (a session applies it: "a BUILD, not a question"). A hit does NOT count as an answer when it is
       - the part's own source: the words of his that raised the doubt (e.g. his "too near" in LEARNED #149);
       - a line Claude chose ("Who chose it: Claude", "(Claude's reading)") - not his answer;
       - a place a session already read and wrote why it does not answer (`python -m machine.questions checked`).
  2. The readings on the TUNE years, signals only (machine/questions/disagree.py). 95% or more the same -> no
     question ("readings agree").
  3. The moments where they trade differently, most different first: one per reading (2-4, config), in different
     weeks, every one drawable on ONE 5-minute chart that ends at the entry and passes every picture rule; plus one
     more moment for the confirm pictures. One picture per reading of the SAME moment (A / B / C), the question ON it,
     "or draw your own".
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
import tempfile
from pathlib import Path

from machine import config
from machine.charts import draw as D
from machine.charts import lint
from machine.data.clock import to_ist
from machine.ideas import store
from machine.ideas.form import Idea, open_parts
from machine.questions import disagree, pictures
from machine.questions import queue as Q

CLAUDES = re.compile(r"Who chose it: Claude|\(Claude's reading\)|Claude's reading", re.IGNORECASE)


class AskError(ValueError):
    """The part cannot be asked (not an ASK part, answered already, questions already out)."""


# ================================================================================================= the #129 search

def _names(text: str) -> set[str]:
    """The places a text names, in one spelling: 'inbox 2026-09-28-1151', 'LEARNED #149', or the text itself."""
    out = {text.strip().lower()}
    out |= {f"inbox {m}" for m in re.findall(r"inbox[ /]+(\d{4}-\d{2}-\d{2}-\d{4})", text)}
    out |= {f"inbox {m}" for m in re.findall(r"learned-inbox/(?:merged/)?(\d{4}-\d{2}-\d{2}-\d{4})", text)}
    out |= {f"learned #{m}" for m in re.findall(r"LEARNED #(\d+)", text, flags=re.IGNORECASE)}
    return out


def search_part(form: Idea, part, *, search, checked: list[dict]) -> dict:
    """The #129 search for one part: FOUND stops the question unless every hit is set aside (see the module text)."""
    query = part.part
    a = search(query)
    own = set()
    for s in part.sources:
        own |= _names(s.ref)
    seen = {c["id"]: c["why"] for c in checked}
    cites = []
    for h in a.cites:
        names = _names(h.id) | set().union(*(_names(x) for x in h.see)) if h.see else _names(h.id)
        if names & own:
            why_not = "its own source: his words that raised this question"
        elif CLAUDES.search(h.quote or ""):
            why_not = "Claude's own line, not his answer"
        elif h.id in seen or any(x in seen for x in h.see):
            why_not = "read by a session: " + seen.get(h.id, next((seen[x] for x in h.see if x in seen), ""))
        else:
            why_not = None
        cites.append({"id": h.id, "see": list(h.see), "quote": h.quote, "why_not": why_not})
    stops = bool(a.found) and any(c["why_not"] is None for c in cites)
    return {"query": query, "found": bool(a.found), "stops": stops, "sentence": a.sentence, "cites": cites,
            "closest": getattr(a.closest, "id", None), "how": getattr(a, "how", "")}


def why_ask(rec: dict) -> str:
    """The #129 sentence for him, short and in easy English (part.md has why each place does not answer)."""
    s = rec["search"]
    head = "Why I ask: I searched your documents and did not find the answer."
    near = [c["id"] for c in s["cites"] if c.get("why_not")] or ([s["closest"]] if s.get("closest") else [])
    return head + (f" Nearest: {', '.join(near)} - {'they do' if len(near) > 1 else 'it does'} not say it."
                   if near else "")


# ================================================================================================= one part

HIS_NUMBER_WORDS = ("His words, 29 Sep 2026 (inbox 2026-09-29-1155): \"These type of ambiguity it should comeup with "
                    "results ... it would be better if it around 7 candles, the edge cases and other probabilities it "
                    "should only check for the best results. And should Inform me.\"")


def _number_keys(part) -> set[str]:
    return {k for r in part.readings for k in r.changes}


def is_a_number(part) -> bool:
    """Every reading sets the SAME one thing to a number (one reading may leave it as written): how near, how far,
    how big. That is never a question to him - it is a VERSION part (his words above)."""
    values = [v for r in part.readings for v in r.changes.values()]
    return (len(_number_keys(part)) == 1 and bool(values)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values))


def letters_for(part) -> dict[str, str]:
    ids = [r.id for r in part.readings]
    if all(re.fullmatch(r"[A-D]", x) for x in ids):
        return {x: x for x in ids}
    return {x: "ABCD"[k] for k, x in enumerate(ids)}


def new_record(form: Idea, part, now: dt.datetime) -> dict:
    return {"idea": form.id, "part": part.id, "kind": "readings", "topic": part.part, "words": part.words,
            "readings": {r.id: r.means for r in part.readings}, "letters": letters_for(part),
            "made_by": "L9b machine.questions.ask", "made": now.isoformat(timespec="seconds"), "status": "new",
            "checked": [], "questions": [], "confirm": None, "history": []}


def make_part(idea: str, part_id: str, *, market=None, search=None, now: dt.datetime | None = None,
              again: bool = False, log=print, base: Path | None = None) -> dict:
    """Steps 1-3 for one part. Returns its record (also saved). market: a compile Market of the TUNE years (loaded
    here when None). search: machine.search.answered (the tests give a fake)."""
    now = now or dt.datetime.now().astimezone()
    form = store.load_form(idea)
    part = next((u for u in form.unclear if u.id == part_id), None)
    if part is None:
        raise AskError(f"{idea} has no unclear part {part_id}")
    if part.kind != "ask":
        raise AskError(f"{idea} {part_id} is a VERSION part: the machine tries each reading, it is never asked")
    if part.answer is not None:
        raise AskError(f"{idea} {part_id} is answered already ({part.answer}) - nothing to ask")
    if is_a_number(part):
        raise AskError(f"{idea} {part_id} is a NUMBER ({', '.join(sorted(_number_keys(part)))}): the machine tries "
                       f"each one and tells him which works better - never asked. {HIS_NUMBER_WORDS}")
    rec = Q.load(idea, part_id, base)
    if rec and any(q.get("answer") for q in rec.get("questions", [])):
        raise AskError(f"{idea} {part_id}: he answered one of its questions already ({rec['status']}) - never asked "
                       "twice")
    if rec and any(q.get("status") not in ("waiting", "withdrawn") for q in rec.get("questions", [])):
        raise AskError(f"{idea} {part_id}: its questions went out already ({rec['status']}) - withdraw them first "
                       "(python -m machine.questions withdraw ...) if they must be made again")
    if rec and any(q.get("status") == "waiting" for q in rec.get("questions", [])) and not again:
        raise AskError(f"{idea} {part_id}: questions are made already and waiting (use --again to make new ones)")
    rec = rec or new_record(form, part, now)
    if search is None:
        from machine.search import rank
        search = rank.answered
    rec["search"] = search_part(form, part, search=search, checked=rec.get("checked", []))
    if rec["search"]["stops"]:
        Q.set_status(rec, "found", now)
        rec["questions"], rec["confirm"] = [], None
        log(f"{idea} {part_id}: FOUND in his files - no question: " + "; ".join(
            c["id"] for c in rec["search"]["cites"] if c["why_not"] is None))
        _save(rec, base)
        return rec
    if market is None:
        from machine.ideas.compile import Market
        market = Market.load(str(min(config.TUNE_YEARS)), str(max(config.TUNE_YEARS)))
    d = disagree.compare(form, market, part_id)
    log(d.text())
    rec["held"] = d.held
    rec["numbers"] = {"years": list(d.years), "signals": d.counts, "same": d.same, "different": d.different,
                      "share_same": round(d.share_same, 4), "moments": len(d.moments)}
    if d.agree:
        Q.set_status(rec, "readings agree", now)
        _save(rec, base)
        return rec
    shown_before = {q["end"] for q in rec.get("withdrawn", [])}
    want = max(config.QUESTION_MOMENTS_MIN, min(config.QUESTION_MOMENTS_MAX, len(part.readings)))
    chosen = choose(form, part, d, market, want + 1, avoid=shown_before, log=log)
    if len(chosen) < config.QUESTION_MOMENTS_MIN + 1:
        Q.set_status(rec, "not drawable", now)
        log(f"{idea} {part_id}: only {len(chosen)} moments fit one chart - a session looks at it")
        _save(rec, base)
        return rec
    asked, spare = chosen[:-1], chosen[-1]
    out = Q.folder(idea, part_id, base)
    for old in out.glob("*.jpg"):
        old.unlink()                                    # only waiting (never shown) pictures lie here
    rec["questions"] = []
    m5 = market.bars("5m")
    before = len(rec.get("withdrawn", []))                  # a withdrawn question's id is never used again
    for k, (m, sc) in enumerate(asked, 1):
        n = before + k
        qid = f"{idea}-{part_id}-{n}"
        pic = _render(form, part, sc, m5, out / f"q{n}.jpg", subject=f"L9b {qid} (one picture)",
                      letters=rec["letters"], number=f"QUESTION {k}")
        rec["questions"].append({"qid": qid, "n": n, "direction": m.direction, "end": str(sc.end),
                                 "end_ist": to_ist(sc.end).strftime("%d %b %Y %H:%M"), "score": m.score,
                                 "why": m.why, "trading": [r for r, s in sc.signals.items() if s is not None],
                                 "letters": [rec["letters"][r.id] for r in part.readings],
                                 "pictures": [pic], "status": "waiting", "answer": None})
    m, sc = spare
    rec["confirm"] = {"qid": f"{idea}-{part_id}-confirm", "end": str(sc.end),
                      "end_ist": to_ist(sc.end).strftime("%d %b %Y %H:%M"), "status": None, "proposed": None,
                      "pictures": {r.id: _render(form, part, sc, m5, out / f"confirm-{rec['letters'][r.id]}.jpg",
                                                 subject=f"L9b {idea}-{part_id}-confirm {rec['letters'][r.id]}",
                                                 letters=rec["letters"], number="", confirm=r.id)
                                   for r in part.readings}}
    Q.set_status(rec, "waiting", now)
    Q.note(rec, f"{len(asked)} questions made, ONE picture each ({', '.join(q['end_ist'] for q in rec['questions'])})"
                f", confirm moment {rec['confirm']['end_ist']}", now)
    _save(rec, base)
    log(f"{idea} {part_id}: {len(asked)} questions made -> {Q.rel(out)}")
    return rec


def withdraw(idea: str, part_id: str, why: str, *, now: dt.datetime | None = None, base: Path | None = None) -> dict:
    """Take back questions that were sent but not answered (e.g. he said the pictures are not clear). Their pictures
    move to withdrawn-<date>/ (they were shown: kept); a question he answered is never withdrawn."""
    now = now or dt.datetime.now().astimezone()
    rec = Q.load(idea, part_id, base)
    if rec is None:
        raise AskError(f"no questions for {idea} {part_id}")
    if any(q.get("answer") for q in rec["questions"]):
        raise AskError(f"{idea} {part_id}: he answered already - never withdrawn")
    folder = Q.folder(idea, part_id, base)
    keep = folder / f"withdrawn-{now:%Y-%m-%d-%H%M}"
    keep.mkdir(parents=True, exist_ok=True)
    for q in rec["questions"]:
        for p in q["pictures"]:
            src = config.ROOT / p["file"] if not Path(p["file"]).is_absolute() else Path(p["file"])
            if src.is_file():
                shutil.move(str(src), str(keep / src.name))
                p["file"] = Q.rel(keep / src.name)
        q["status"] = "withdrawn"
        q["withdrawn"] = why
    for p in (rec.get("confirm") or {}).get("pictures", {}).values():
        src = config.ROOT / p["file"] if not Path(p["file"]).is_absolute() else Path(p["file"])
        if src.is_file():
            src.unlink()                                    # never shown: made again with the new questions
    rec.setdefault("withdrawn", []).extend(rec["questions"])
    rec["questions"], rec["confirm"] = [], None
    Q.set_status(rec, "withdrawn", now)
    Q.note(rec, f"withdrawn: {why}", now)
    _save(rec, base)
    return rec


def choose(form: Idea, part, d: disagree.Disagreement, market, n: int, *, avoid: set | None = None,
           log=print) -> list[tuple]:
    """The n most different moments whose ONE picture passes every picture rule, in different weeks, never a moment
    already shown (avoid); a moment whose trading readings differ from those already chosen goes first when two are
    equally different."""
    m5 = market.bars("5m")
    letters = letters_for(part)
    chosen, weeks, kinds = [], set(), set()
    tmp = Path(tempfile.mkdtemp(prefix="l9b-"))
    try:
        pool = list(d.moments)
        while pool and len(chosen) < n:
            top = pool[0].score
            same_score = [m for m in pool if m.score == top]
            m = next((x for x in same_score if tuple(x.trading()) not in kinds), same_score[0])
            pool.remove(m)
            week = m.at.isocalendar()[:2]
            if week in weeks:
                continue
            sc = pictures.scene(m, d.compiled, market)
            if sc is None or str(sc.end) in (avoid or set()):
                continue
            pic = pictures.picture(form, part, sc, m5, subject=f"trial {m.at}", letters=letters, number="QUESTION 1")
            if pic is None or lint.check(D.render(pic, tmp / "trial.jpg"), shown=tmp / "none.jsonl"):
                continue
            chosen.append((m, sc))
            weeks.add(week)
            kinds.add(tuple(m.trading()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return chosen


def _render(form, part, sc, m5, path: Path, *, subject, letters, number, confirm: str | None = None) -> dict:
    pic = pictures.picture(form, part, sc, m5, subject=subject, letters=letters, number=number, confirm=confirm)
    drawn = D.render(pic, path)
    problems = lint.check(drawn)
    if problems:
        raise AskError(f"{path.name} breaks a picture rule: " + "; ".join(str(p) for p in problems))
    return {"letter": letters[confirm] if confirm else None, "reading": confirm, "file": Q.rel(path),
            "subject": subject, "pixel_sha256": drawn.pixel_sha256, "end": str(sc.end),
            "ends_at": str(drawn.entry_at), "last_candle": str(pic.rows[0][0].bars.index[-1]),
            "question": pic.question.ask, "lint": f"all {len(lint.RULES)} rules pass",
            "texts": [t.text for t in drawn.texts]}


def open_ask_parts(idea: str) -> list[str]:
    return [u.id for u in open_parts(store.load_form(idea)) if u.kind == "ask"]


def add_checked(idea: str, part_id: str, hit_id: str, why: str, *, by: str = "a Claude session",
                now: dt.datetime | None = None, base: Path | None = None) -> dict:
    """A session read a FOUND place and it does not answer the part: written down with why (then ask again)."""
    now = now or dt.datetime.now().astimezone()
    if len(why.strip()) < 10:
        raise AskError("say WHY the place does not answer the question (a sentence)")
    rec = Q.load(idea, part_id, base)
    if rec is None:
        form = store.load_form(idea)
        part = next(u for u in form.unclear if u.id == part_id)
        rec = new_record(form, part, now)
    rec.setdefault("checked", []).append({"id": hit_id, "why": why.strip(), "by": by,
                                          "at": now.isoformat(timespec="seconds")})
    Q.note(rec, f"checked {hit_id}: {why.strip()}", now)
    _save(rec, base)
    return rec


# ================================================================================================= the page

def _save(rec: dict, base: Path | None) -> None:
    Q.save(rec, base)
    (Q.folder(rec["idea"], rec["part"], base) / "part.md").write_text(page(rec), encoding="utf-8", newline="\n")


def page(rec: dict) -> str:
    L = [f"# {rec['idea']} {rec['part']} — {rec['topic']}", "",
         f"> Made by `{rec['made_by']}` on {rec['made'][:10]}. The machine's record is `part.json` (this page is the "
         "same, for people). No money anywhere in a question (lesson B16).", "",
         f"**Status:** {rec['status']}", ""]
    if rec.get("words"):
        L += [f"**His words that leave it open:** \"{rec['words']}\"", ""]
    L += ["| letter | reading |", "|---|---|"]
    L += [f"| {rec['letters'][r]} | {m} |" for r, m in rec["readings"].items()]
    s = rec.get("search")
    if s:
        L += ["", "## The #129 search first", "",
              f"Asked: \"{s['query']}\" → {'FOUND' if s['found'] else 'NOT FOUND'} ({s.get('how', '')}). "
              + ("**It stops the question** (a session applies the place, or writes why it does not answer)."
                 if s["stops"] else "No place answers it — the question goes ahead."), ""]
        for c in s["cites"]:
            L.append(f"- {c['id']}: \"{c['quote']}\" — " + (c["why_not"] or "**counts as an answer**"))
        L += ["", f"For him: {why_ask(rec)}"]
    n = rec.get("numbers")
    if n:
        L += ["", "## The readings on the TUNE years (signals only, no money)", "",
              f"{n['years'][0]}–{n['years'][-1]}, other open parts held at "
              f"{', '.join(f'{k}={v}' for k, v in (rec.get('held') or {}).items()) or 'none'}: "
              + ", ".join(f"{rec['letters'][r]} {c:,} signals" for r, c in n["signals"].items())
              + f"; the same in every reading {n['same']:,} of {n['same'] + n['different']:,} = "
              f"{n['share_same']:.0%} (they agree at {config.READINGS_AGREE_SHARE:.0%}); {n['moments']:,} moments "
              "where they trade differently."]
    if rec.get("questions"):
        L += ["", "## The questions (ONE picture each: what A and B read differently is marked with a ?, both answers "
              "written on it; ending at the entry)", "",
              "| question | moment (India time) | direction | what differs | picture | status | his answer |",
              "|---|---|---|---|---|---|---|"]
        for q in rec["questions"]:
            pics = " · ".join(f"[picture]({Path(p['file']).name})" for p in q["pictures"])
            ans = (q.get("answer") or {}).get("letter") or (q.get("answer") or {}).get("kind") or ""
            L.append(f"| {q['qid']} | {q['end_ist']} | {q['direction']} | {q['why']} | {pics} | {q['status']} | "
                     f"{ans} |")
        c = rec.get("confirm") or {}
        if c:
            L += ["", f"Confirm pictures (another moment, {c['end_ist']}, sent only after his answer): "
                  + " · ".join(f"[{p['letter']}]({Path(p['file']).name})" for p in c["pictures"].values())
                  + f". Proposed: {c.get('proposed') or '—'} · status: {c.get('status') or '—'}."]
    if rec.get("withdrawn"):
        L += ["", "## Withdrawn (sent, never answered)", ""]
        L += [f"- {q['qid']} ({q['end_ist']}): {q.get('withdrawn', '')} — "
              + ", ".join(f"`{p['file']}`" for p in q["pictures"]) for q in rec["withdrawn"]]
    if rec.get("checked"):
        L += ["", "## Places a session read (they do not answer it)", ""]
        L += [f"- {c['id']} — {c['why']} ({c['by']}, {c['at'][:10]})" for c in rec["checked"]]
    L += ["", "## History", ""] + [f"- {h['at']} {h['what']}" for h in rec.get("history", [])]
    return "\n".join(L) + "\n"
