"""Gives the lessons waiting in strategy/learned-inbox/ their numbers in strategy/LEARNED.md (L9a).

ONLY ON MAIN, ONLY IN THE GITHUB ACTION (.github/workflows/learned-merge.yml) - two branches would both write #137
(likhith.md R12, B26). Nobody numbers by hand.

    python -m machine.tools.learned_merge            shows what it WOULD do (any branch, any PC; changes nothing)
    python -m machine.tools.learned_merge --apply    the Action only: numbers, appends, moves to merged/

Numbering: N = the highest number in LEARNED.md + 1, counting BOTH heading styles ("## #N  ·  <date> ..." and
"## #N — <date>. <title>"). #21-#24 exist twice (6 and 7 Sep): that does not change the highest. The waiting files
are numbered oldest first, by the time in their name (<YYYY-MM-DD-HHMM>-<source>[-2].md; a file without that time
waits and is reported). Each becomes

    ## #N — <date>. <title>

    <the file without its first heading>

    (from `strategy/learned-inbox/merged/<name>`; numbered on main by learned-merge, L9a)

and moves to strategy/learned-inbox/merged/. LEARNED.md is only ever appended to (it is add-only).
The title: a line "<!-- title: ... -->" in the file if there is one; else the first sentence of its first "= " line
(the session's reading of his point); else what the file is ("Lesson (Telegram)") and his first words, quoted.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from machine import config

HEAD = re.compile(r"^## #(\d+)(?=\s|$)", re.MULTILINE)          # both heading styles start this way
NAME = re.compile(r"^(\d{4}-\d\d-\d\d)-(\d{4})-(.+?)(?:-(\d+))?$")
TITLE_TAG = re.compile(r"^\s*<!--\s*title:\s*(.*?)\s*-->\s*$", re.IGNORECASE)
TITLE_MAX = 150


@dataclass
class Waiting:
    path: Path
    date: str
    title: str
    body: str
    n: int = 0


def highest(learned_text: str) -> int:
    """The highest lesson number in LEARNED.md, both heading styles counted (0 if none)."""
    return max((int(m.group(1)) for m in HEAD.finditer(learned_text)), default=0)


def _order(path: Path):
    m = NAME.match(path.stem)
    return (m.group(1), m.group(2), int(m.group(4) or 1), m.group(3)) if m else None


def waiting(inbox_dir: Path) -> tuple[list[Path], list[str]]:
    """(the inbox files to number, oldest first; the files that wait because something is wrong with them)."""
    ok, problems = [], []
    for p in sorted(Path(inbox_dir).glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        if _order(p) is None:
            problems.append(f"{p.name}: the name has no time (<YYYY-MM-DD-HHMM>-<source>.md) - left waiting")
            continue
        if HEAD.search(p.read_text(encoding="utf-8")):
            problems.append(f"{p.name}: it has a '## #N' line inside, which would break the numbering - left waiting")
            continue
        ok.append(p)
    return sorted(ok, key=_order), problems


def _clean(text: str) -> str:
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^\(\d+\)\s*", "", " ".join(text.split()))
    return text.strip()


def _cut(text: str, n: int = TITLE_MAX) -> str:
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(" ,;:—-") + " …"


def title_of(text: str) -> str:
    lines = text.splitlines()
    for line in lines:
        m = TITLE_TAG.match(line)
        if m and m.group(1).strip():
            return _cut(_clean(m.group(1)))
    for i, line in enumerate(lines):                      # the first "= " paragraph, its first sentence
        if line.startswith("= "):
            para = [line[2:]]
            for nxt in lines[i + 1:]:
                if not nxt.strip() or nxt.startswith("= ") or nxt.startswith("#"):
                    break
                para.append(nxt)
            first = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'*])", _clean(" ".join(para)), maxsplit=1)[0]
            return _cut(first)
    what = next((ln[2:] for ln in lines if ln.startswith("# ")), "Lesson")
    what = re.sub(r"\s*—\s*waiting for a number\s*$", "", what)
    what = re.sub(r",?\s*\d{1,2} [A-Z][a-z]{2} \d{4}(?: \d\d:\d\d)?", "", what).replace("(,", "(").strip()
    words = next((ln[1:].strip() for ln in lines if ln.startswith(">") and ln[1:].strip()), "")
    return _cut(f'{what}: "{_clean(words)}"' if words else what)


def body_of(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    lines = [ln for ln in lines if not TITLE_TAG.match(ln)]
    return "\n".join(lines).strip("\n")


def plan(root: Path | None = None) -> tuple[list[Waiting], list[str]]:
    """What --apply would do: each waiting file with its number, date and title; and the files left waiting."""
    root = Path(root or config.ROOT)
    learned = root / "strategy" / "LEARNED.md"
    paths, problems = waiting(root / "strategy" / "learned-inbox")
    n = highest(learned.read_text(encoding="utf-8"))
    out = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        n += 1
        out.append(Waiting(p, NAME.match(p.stem).group(1), title_of(text), body_of(text), n))
    return out, problems


def entry(w: Waiting) -> str:
    return (f"## #{w.n} — {w.date}. {w.title}\n\n{w.body}\n\n"
            f"(from `strategy/learned-inbox/merged/{w.path.name}`; numbered on main by learned-merge, L9a)\n")


def refuse_off_main(env: dict | None = None) -> None:
    """--apply runs only inside the GitHub Action, on main. Anywhere else it stops before touching a file."""
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("learned-merge numbers lessons ONLY in the GitHub Action on main (likhith.md R12) - not on a "
                         "PC. Push the inbox files to main; the Action numbers them. (Without --apply it only shows.)")
    if env.get("GITHUB_REF") != "refs/heads/main":
        raise SystemExit(f"learned-merge refuses to run off main (this run is on {env.get('GITHUB_REF')!r}): two "
                         "branches would both write the same number.")


def apply(root: Path | None = None, *, env: dict | None = None) -> list[Waiting]:
    """Appends every waiting lesson to LEARNED.md with its number, then moves its file to merged/."""
    refuse_off_main(env)
    root = Path(root or config.ROOT)
    todo, _ = plan(root)
    if not todo:
        return []
    learned = root / "strategy" / "LEARNED.md"
    raw = learned.read_bytes()
    nl = "\r\n" if raw[-4096:].count(b"\r\n") * 2 > raw[-4096:].count(b"\n") else "\n"
    text = ("" if raw.endswith(b"\n") or not raw else nl) + "".join(nl + entry(w).replace("\n", nl) for w in todo)
    with open(learned, "a", encoding="utf-8", newline="") as fh:     # append only: LEARNED.md is add-only
        fh.write(text)
    merged = root / "strategy" / "learned-inbox" / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    for w in todo:
        target = merged / w.path.name
        if target.exists():
            raise SystemExit(f"{target} exists already - two inbox files with one name; nothing more moved")
        os.replace(w.path, target)
    return todo


def message(done: list[Waiting]) -> str:
    if not done:
        return "learned-merge: nothing waiting"
    span = f"#{done[0].n}" if len(done) == 1 else f"#{done[0].n}-#{done[-1].n}"
    return (f"learned-merge: {span} from the inbox ({len(done)} file{'s' if len(done) != 1 else ''})\n\n"
            + "\n".join(f"#{w.n} {w.path.name}: {w.title}" for w in done) + "\n")


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.tools.learned_merge")
    ap.add_argument("--apply", action="store_true", help="number them (the GitHub Action on main only)")
    ap.add_argument("--message-file", help="--apply: write the commit message here")
    args = ap.parse_args(argv)
    if args.apply:
        done = apply()
        print(message(done))
        if args.message_file:
            Path(args.message_file).write_text(message(done), encoding="utf-8")
        return
    todo, problems = plan()
    print(f"WOULD number {len(todo)} file(s) (nothing changed - only the Action on main numbers):")
    for w in todo:
        print(f"  ## #{w.n} — {w.date}. {w.title}    <- {w.path.name}")
    for p in problems:
        print("  WAITS: " + p)


if __name__ == "__main__":
    main()
