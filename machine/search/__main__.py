"""python -m machine.search "your question" - is it already in his documents? (the #129 search, L8b)"""
from __future__ import annotations

import argparse
import sys

from machine.search import index as ix
from machine.search.rank import answered


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    ap.add_argument("--embeddings", default="auto", choices=("auto", "off", "local", "voyage"))
    ap.add_argument("--hits", type=int, default=0, help="also print this many ranked hits")
    args = ap.parse_args(argv)
    index = ix.get(embeddings=args.embeddings)
    a = answered(" ".join(args.question), index=index)
    print(a)
    print(f"(search: {a.how})")
    for h in a.hits[: args.hits]:
        print(f"  #{h.rank:<2} tier {h.tier} {h.id:<45} meaning {h.meaning:.2f} covered {h.covered:.2f} "
              f"words {h.words:5.1f} {'passes' if h.passes else ''}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
