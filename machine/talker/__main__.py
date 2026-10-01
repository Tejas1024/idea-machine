"""python -m machine.talker "his words"  -> the draft form, its unclear parts, the checks and the cost (L8).

    python -m machine.talker "london high taken and it give bos and fvg in 5 min then sell at fvg, target asian low"
    python -m machine.talker --file plan/work/idea-001/words.md --save plan/work/ideas/idea-001/talker-draft.json
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from machine import config
from machine.talker.translate import translate


def words_from_md(path: Path) -> str:
    """His quoted words from a markdown file: every '> ' line (quotes joined, the surrounding quote marks dropped)."""
    quotes = [ln[1:].strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.startswith(">")]
    text = "\n".join(q for q in quotes if q)
    return re.sub(r'(^|\n)"|"(\n|$)', r"\1\2", text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="?", help="his words")
    ap.add_argument("--file", help="a markdown file: his words are its '> ' quote lines")
    ap.add_argument("--id", default="idea-900")
    ap.add_argument("--model", default=config.TALKER_MODEL)
    ap.add_argument("--effort", default=config.TALKER_EFFORT)
    ap.add_argument("--backend", default=config.TALKER_BACKEND, choices=("claude-code", "api"),
                    help="claude-code = his own login, no bill per call (default); api = ANTHROPIC_API_KEY, paid")
    ap.add_argument("--save", help="write the form (JSON) here")
    args = ap.parse_args(argv)
    words = words_from_md(Path(args.file)) if args.file else args.words
    if not words:
        ap.error("give his words, or --file")
    d = translate(words, args.id, model=args.model, effort=args.effort, backend=args.backend)
    print(d.text())
    if args.save and d.form is not None:
        Path(args.save).write_text(d.form.to_json(), encoding="utf-8")
        print(f"saved: {args.save}")
    return 0 if d.valid else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
