"""The MISSING BLOCK path (L7b): a form that needs a block, rule or filter that does not exist yet.

    from machine.ideas import blocked
    try:
        out = compile_idea(form, market)
    except MissingBlockError as err:
        blocked.request(form, err)              # the idea is BLOCKED; the request, the Telegram line (+ a question)

What request() does, once for each missing name:
  1. the idea's state -> BLOCKED, and what it waits for (state.json keeps since when - machine/ideas/store.py);
  2. one line in plan/work/block-requests.md: the date, the block, the idea, his words that need it, the prompt that
     builds it and its owner (plan/PROGRESS.md). No prompt builds it yet = the reusable prompt L6-new, owner "either";
  3. one Telegram line to Tejas in plan/work/telegram-outbox.jsonl - the bot (L9a) sends the lines it has not sent yet;
  4. if the block's MEANING is unclear (unclear={name: (the question, the #129 search sentence)}), its question is
     queued at once in plan/work/question-queue/ - priority 1 of the weekly batch: it unblocks an idea (L9b).
The same block for the same idea is asked for once.
waiting() and status_lines() say what waits and for how many days (STATUS, L10b). built(block): the block exists now
- its requests close, and an idea that waits for nothing else goes back to FORM DRAFTED to be compiled again.
built(block, says={...}): the block was built under another name than the made-up one in the waiting forms (L6-new:
'london_session_high_only' -> the sweep step's params.which_session) - the forms are rewritten first.

    .venv\\Scripts\\python -m machine.ideas.blocked          STATUS: every block request, and how long it has waited
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from machine import config
from machine.ideas import store
from machine.ideas.compile import MissingBlockError
from machine.ideas.form import Idea

HEADER = [
    "# Blocks the ideas are waiting for",
    "",
    "> Written by `machine/ideas/blocked.py` (L7b) — never by hand. One row per block, rule or filter an idea needs",
    "> that the machine does not have yet. The idea is BLOCKED until the prompt in \"built by\" adds it (no prompt yet =",
    "> the reusable prompt L6-new). STATUS: `python -m machine.ideas.blocked`.",
    "",
    "| asked | block | idea | where in the form | the words that need it | built by | owner | status |",
    "|---|---|---|---|---|---|---|---|",
]
WAITING = "waiting"
_PROMPT = re.compile(r"\b(L\d+[a-z]?(?:\.\d+)?(?:-new)?)\b")


def requests_path() -> Path:
    return Path(config.WORK_DIR) / "block-requests.md"


def outbox_path() -> Path:
    """Lines for Tejas's Telegram, one JSON each; the bot (L9a) sends every line whose id it has not sent yet."""
    return Path(config.WORK_DIR) / "telegram-outbox.jsonl"


def queue_dir() -> Path:
    return Path(config.WORK_DIR) / "question-queue"


# ================================================================================================== ask for blocks

def request(form: Idea, err: MissingBlockError, *, unclear: dict[str, tuple[str, str]] | None = None,
            now: dt.datetime | None = None) -> list[dict]:
    """The idea is BLOCKED by err.missing. Returns the NEW request rows (a block already asked for is not asked
    again). unclear: {block name: (the question in his language, the #129 search sentence)} for a block whose MEANING
    is unclear - queued at once. Never a question without the search sentence (#129)."""
    if err.idea_id != form.id:
        raise ValueError(f"the error is about {err.idea_id}, the form is {form.id}")
    unclear = dict(unclear or {})
    names = {n for n, _, _ in err.missing}
    for name, qa in unclear.items():
        if name not in names:
            raise ValueError(f"'{name}' is not one of the missing names {sorted(names)}")
        if len(qa) != 2 or not all(str(x).strip() for x in qa):
            raise ValueError(f"'{name}': a question needs its words AND the #129 search sentence ('I searched ... "
                             "for X and found nothing - the closest line is ...')")
    now = now or dt.datetime.now().astimezone()
    if not (store.folder(form.id) / store.FORM).is_file():
        store.create(form)
    rows = read_requests()
    new = []
    for name, where, built_by in err.missing:
        if any(r["block"] == name and r["idea"] == form.id and r["status"] == WAITING for r in rows + new):
            continue
        prompt = _prompt(built_by)
        row = {"asked": now.date().isoformat(), "block": name, "idea": form.id, "where": where,
               "words": _words(form, name, where), "built_by": prompt, "owner": owner_of(prompt), "status": WAITING}
        new.append(row)
        asked = name in unclear
        _append_row(row)
        _telegram(form, row, asked, now)
        if asked:
            _queue_question(form, row, *unclear[name], now=now)
    waits = sorted({f"{r['block']} ({r['built_by']}, {r['owner']})" for r in read_requests()
                    if r["idea"] == form.id and r["status"] == WAITING})
    if waits:
        state = store.read_state(form.id)
        if state["state"] != "BLOCKED" or new:
            store.set_state(form.id, "BLOCKED", "needs " + ", ".join(n for n, _, _ in err.missing),
                            waits_for="the blocks " + "; ".join(waits))
    return new


def _prompt(built_by: str) -> str:
    m = _PROMPT.search(built_by)
    return m.group(1) if m else "L6-new"


def owner_of(prompt: str) -> str:
    """The owner of a prompt's row in plan/PROGRESS.md ("either" for the reusable L6-new; "?" if no row)."""
    path = Path(config.PROGRESS_FILE)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) > 2 and cells[1].replace("⭐", "").strip() == prompt:
                return cells[2].replace("*", "").strip() or "?"
    return "either" if prompt == "L6-new" else "?"


def _words(form: Idea, name: str, where: str) -> str:
    """The words in the form that need the block: his words from the part's sources, else what the part says."""
    got: list[str] = []
    m = re.match(r"step '([a-z0-9_]+)'", where)
    if m:
        step = next(s for s in form.steps if s.id == m.group(1))
        got = [x.words for x in step.sources if x.words] or [step.what]
    elif where in ("do_not_enter", "filters"):
        for r in getattr(form, where):
            if r.name == name:
                got = [x.words for x in r.sources if x.words] or ([r.words] if r.words else [])
    elif where.startswith("targets"):
        got = [x.words for t in form.targets for x in t.sources if x.words]
    text = " / ".join(got) or form.words
    text = " ".join(text.split())
    return text if len(text) <= 240 else text[:237] + "..."


# ================================================================================================ the request file

def _cell(x: str) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _uncell(x: str) -> str:
    return x.strip().replace("\\|", "|")


_KEYS = ("asked", "block", "idea", "where", "words", "built_by", "owner", "status")


def read_requests() -> list[dict]:
    """Every row of plan/work/block-requests.md, oldest first."""
    path = requests_path()
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*\d{4}-\d\d-\d\d\s*\|", line):
            continue
        cells = [_uncell(c) for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if len(cells) == len(_KEYS):
            out.append(dict(zip(_KEYS, cells)))
    return out


def _row_line(r: dict) -> str:
    return "| " + " | ".join(_cell(r[k]) for k in _KEYS) + " |"


def _append_row(row: dict) -> None:
    path = requests_path()
    if not path.is_file():
        store._write(path, "\n".join(HEADER) + "\n")
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(_row_line(row) + "\n")


def _rewrite(rows: list[dict]) -> None:
    store._write(requests_path(), "\n".join(HEADER + [_row_line(r) for r in rows]) + "\n")


# ======================================================================================== Telegram and the question

def _label(idea_id: str) -> str:
    return f"idea #{int(idea_id.split('-')[1])}"


def _telegram(form: Idea, row: dict, asked: bool, now: dt.datetime) -> None:
    """One line to Tejas, in easy English (CLAUDE.md), no money. The bot (L9a) sends it."""
    owner = row["owner"].lower()
    who = ("You build" if owner.startswith("tejas") else "Likhith or you build" if owner in ("either", "?")
           else f"{row['owner']} builds")
    text = (f"{_label(form.id)} ({form.title}) is waiting. The machine cannot see \"{row['block']}\" yet. "
            f"{who} it in {row['built_by']}. "
            + ("One question about what it means is in this week's picture batch." if asked
               else "Nothing for you to do now."))
    line = {"id": f"{row['asked']}-{form.id}-block-{row['block']}", "at": now.isoformat(timespec="seconds"),
            "to": "Tejas", "text": text, "why": "block request (L7b)"}
    path = outbox_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def question_id(idea_id: str, block: str) -> str:
    return f"{idea_id}-block-{block}"


def _queue_question(form: Idea, row: dict, question: str, searched: str, *, now: dt.datetime) -> Path:
    qid = question_id(form.id, row["block"])
    path = queue_dir() / f"{qid}.md"
    text = "\n".join([
        f"# {qid}", "",
        "| field | value |", "|---|---|",
        "| status | waiting for the weekly batch |",
        f"| made by | L7b `machine/ideas/blocked.py` ({now.date().isoformat()}) |",
        f"| idea | {_label(form.id)} — {_cell(form.title)} |",
        "| priority (L9b) | (1) it unblocks a BLOCKED idea |",
        f"| block | `{row['block']}` — built by {row['built_by']} ({row['owner']}) once its meaning is clear |",
        "| picture | none yet: the question loop (L9b) draws one, ending at the entry, no money |", "",
        "## The question", "", question.strip(), "",
        "## Why ask (the #129 search first)", "", searched.strip(), "",
        "## The words that need it", "", f"> {row['words']}", "",
    ])
    store._write(path, text)
    return path


# =================================================================================================== STATUS, built

def waiting(now: dt.datetime | None = None) -> list[dict]:
    """The requests still waiting, each with the days it has waited (STATUS, L10b)."""
    today = (now or dt.datetime.now().astimezone()).date()
    out = []
    for r in read_requests():
        if r["status"] == WAITING:
            out.append({**r, "days": (today - dt.date.fromisoformat(r["asked"])).days})
    return out


def status_lines(now: dt.datetime | None = None) -> list[str]:
    """One line per idea that waits for a block: since when, for how many days, for what, and who builds it."""
    by_idea: dict[str, list[dict]] = {}
    for r in waiting(now):
        by_idea.setdefault(r["idea"], []).append(r)
    lines = []
    for idea, rows in sorted(by_idea.items()):
        days = max(r["days"] for r in rows)
        since = min(r["asked"] for r in rows)
        what = "; ".join(f"{r['block']} ({r['built_by']}, {r['owner']})" for r in rows)
        lines.append(f"{idea} BLOCKED since {since} ({days} day{'s' if days != 1 else ''}): waits for {what}")
    return lines


def built(block: str, *, by: str, now: dt.datetime | None = None, says: dict | None = None) -> list[str]:
    """The block exists now (built by prompt `by`): its requests close with the days they waited, and every idea that
    waits for nothing else goes back to FORM DRAFTED. Returns those ideas.

    says: the block was built under ANOTHER name than the one the waiting forms use. The talker makes up a name for
    what it cannot say yet (the filter 'london_session_high_only'); the prompt builds it where it belongs (the sweep
    step's params.sessions, L6-new 30 Sep). {dotted path: new value} (form.apply_changes), applied to every form that
    waits for this block BEFORE any request closes. Refused - and nothing is written - if a rewritten form is not a
    good form or still needs `block`. The rewrite goes into the idea's history. His words are never changed by it."""
    now = now or dt.datetime.now().astimezone()
    rows = read_requests()
    touched = {r["idea"] for r in rows if r["block"] == block and r["status"] == WAITING}
    if not touched:
        return []
    rewritten = {}
    if says:
        from machine.ideas.compile import missing_in_any_reading
        from machine.ideas.form import apply_changes
        if any(str(p).split(".")[0] in ("words", "id") for p in says):
            raise ValueError("says may not change an idea's words or id: they are his")
        for idea in sorted(touched):
            new = apply_changes(store.load_form(idea), says, who=f"{block} built by {by}")
            if any(n == block for n, _, _ in missing_in_any_reading(new)):
                raise ValueError(f"{idea}: the rewritten form still needs '{block}' - says must put the built name "
                                 "in its place")
            rewritten[idea] = new
        for new in rewritten.values():
            store.save_form(new)
    for r in rows:
        if r["block"] == block and r["status"] == WAITING:
            days = (now.date() - dt.date.fromisoformat(r["asked"])).days
            r["status"] = f"built {now.date().isoformat()} by {by} ({days} day{'s' if days != 1 else ''})"
    _rewrite(rows)
    said = (": its form now says " + "; ".join(f"{p} = {json.dumps(v, ensure_ascii=False)}" for p, v in says.items()
                                                 if str(p) != "notes")) if rewritten else ""
    freed = []
    for idea in sorted(touched):
        still = [r for r in rows if r["idea"] == idea and r["status"] == WAITING]
        if still:
            store.set_state(idea, "BLOCKED", f"{block} built by {by}{said}",
                            waits_for="the blocks " + "; ".join(sorted({f"{r['block']} ({r['built_by']}, "
                                                                        f"{r['owner']})" for r in still})))
        elif store.read_state(idea)["state"] == "BLOCKED":
            store.set_state(idea, "FORM DRAFTED", f"{block} built by {by}{said}: compile it again")
            freed.append(idea)
    return freed


def main() -> None:
    rows = waiting()
    print("STATUS - ideas waiting for a block (plan/work/block-requests.md)")
    for line in status_lines() or ["  none: no idea waits for a block"]:
        print("  " + line)
    for r in rows:
        print(f"  - {r['block']} for {r['idea']}: asked {r['asked']} ({r['days']} days), {r['built_by']}, "
              f"{r['owner']}")


if __name__ == "__main__":
    main()
