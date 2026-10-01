"""python -m machine.mcp --idea idea-002   - the research tools over MCP (stdio), for ONE idea (L14.1).

    .venv\\Scripts\\python -m machine.mcp --idea idea-002      serve (Claude Code starts it: machine/mcp/README.md)
    .venv\\Scripts\\python -m machine.mcp                      serve read-only (get_idea, list_blocks, search_docs,
                                                              list_trials with no hidden year)
    .venv\\Scripts\\python -m machine.mcp --plan                the research agent's planning server (L14.2): ONLY
                                                              get_idea, list_blocks, search_docs
    .venv\\Scripts\\python -m machine.mcp --list [--idea ...]  print the tools and stop
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idea", help="the ONE idea this research run is about (idea-NNN); without it: read-only")
    ap.add_argument("--who", help="who runs it (goes into the version records and check-opened.json)")
    ap.add_argument("--plan", action="store_true", help="the research agent's planning server (L14.2): read-only, "
                    "only get_idea, list_blocks and search_docs")
    ap.add_argument("--list", action="store_true", help="print the tools and what they guard, then stop")
    a = ap.parse_args(argv)
    from machine.mcp import server
    srv = server.build(a.idea, who=a.who, plan=a.plan)
    if a.list:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(server.describe(srv))
        return 0
    srv.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
