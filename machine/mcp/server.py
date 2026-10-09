"""The MCP server (L14.1): the research tools over the Model Context Protocol, for the research agent (L14.2).

    .venv\\Scripts\\python -m machine.mcp --idea idea-002       stdio - what Claude Code starts (machine/mcp/README.md)
    .venv\\Scripts\\python -m machine.mcp --list                 the tools and what each one guards, then stop

Every tool and every guard is machine/mcp/tools.py; this file only hands them over the protocol (the official MCP
Python SDK, `mcp`). Three things are added here, at the door:
  - an argument a tool does not have is REFUSED, not dropped (the SDK would drop it silently): `"unlock": true` or
    `"use": "final-exam"` is a trick, and it is written to data/mcp/refusals.jsonl like any other refusal;
  - a refusal reaches the agent as the tool's error, in its own words (which rule, and why);
  - the tools run one at a time on a worker thread (tools.LOCK), so a long test never stops the server answering.
While it serves, the SDK points the process's stdout at stderr: nothing a tool prints can reach the protocol.
"""
from __future__ import annotations

import inspect
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, TextContent

from machine.mcp import tools as T

NAME = "idea-machine-research"
INSTRUCTIONS = """The idea machine's research tools (L14.1). Tejas's gold trading ideas, tested on 2016-2020 gold.
The AI writes rules; code judges. One run = ONE idea (its research target).
Order: get_idea -> search_docs (his documents first, #129) -> choose_slices (at most 8, before any money) ->
propose_version (at most 20 per family: the timeframe or the idea's own numbers - level, stop, target, wait) ->
run_version (2016-2020 only) -> draw_chart (to look at a trade) -> write_report (the referee chooses ONE version by the
highest Deflated Sharpe and opens 2021-2023 once for it).
Never: 2024 or later; 2021-2023 before write_report opened them for that version; his fixed rules; a slice dressed as a
version. Every refusal says why - read it; do not try another way round it (each attempt is written down).
Money: rupees at 0.01 lot, Rs 40 per position all-in (LEARNED #54, #115). Say the try count with every number."""


PLAN_TOOLS = T.PLAN_TOOLS
PLAN_INSTRUCTIONS = """The idea machine's research tools, PLANNING ONLY (the research agent, L14.2). Read the idea
(get_idea), what a version and a slice may be (list_blocks) and his documents (search_docs - his #129 rule), then write
the research plan. Nothing runs and no money is shown while the plan is made: the plan is saved first, then code runs
exactly that plan."""


# Claude Code keeps a long tool answer away from Claude: past about 50,000 characters it saves the answer to a file and
# shows a short preview, past its token limit it gives an error. The planner has no tool that reads a file, so it then
# plans WITHOUT the answer. list_blocks was cut this way in every plan so far (idea #2: "Output too large (48.9KB)";
# his whole strategy and idea #3's rehearsal: "result (65,918 characters) exceeds maximum allowed tokens") - found in
# the 8 Oct rehearsal (L10b): idea #3's planner never saw the slice names and code refused its plan twice. This note on
# a tool tells Claude Code to pass its whole answer (500,000 is the most it allows; checked with a real call: all 11
# parts arrived). A cut answer is an error in llm.read_agent_stream, and tests/test_mcp.py keeps the size under it.
WHOLE = {"anthropic/maxResultSizeChars": 500_000}

STRATEGY_PLAN_TOOLS = T.STRATEGY_PLAN_TOOLS
STRATEGY_PLAN_INSTRUCTIONS = """The idea machine's research tools, PLANNING ONLY, for Tejas's WHOLE strategy (L19d): his
entries together, one trade at a time, his hedge and his management. Read it (get_strategy), what a version and a
slice may be (list_blocks) and his documents (search_docs - his #129 rule), then write the research plan. Nothing runs
and no money is shown while the plan is made: the plan is saved first, then code runs exactly that plan."""


def build(idea: str | None = None, *, bench: T.Workbench | None = None, who: str | None = None,
          plan: bool = False, strategy_plan: bool = False) -> MCPServer:
    """The server for ONE research run (idea = its target; None = read-only). bench: hand-made prices (tests).
    plan=True: the research agent's PLANNING server (L14.2) - read-only, with ONLY get_idea, list_blocks and
    search_docs: no tool that proposes, runs, draws, reports or shows a trial file is in it.
    strategy_plan=True: the planning server for his whole strategy (L19d) - ONLY get_strategy, list_blocks and
    search_docs."""
    if (plan or strategy_plan) and idea is not None:
        raise ValueError("the planning server is read-only: it has no research target")
    if plan and strategy_plan:
        raise ValueError("one planning server at a time: an idea's (plan) or his whole strategy's (strategy_plan)")
    T.start(idea, who)
    bench = bench or T.BENCH
    server = MCPServer(name=NAME, instructions=(PLAN_INSTRUCTIONS if plan else STRATEGY_PLAN_INSTRUCTIONS
                                                if strategy_plan else INSTRUCTIONS), version="L14.1")

    def call(fn, **kwargs) -> dict[str, Any]:
        try:
            return fn(**kwargs)
        except T.Refused as err:
            raise ToolError(f"REFUSED - {err}") from None

    @server.tool(meta=WHOLE)
    def get_idea(idea: str) -> dict[str, Any]:
        """One idea as the machine holds it: his words, its steps, entry, exit and targets, its unclear parts and
        their readings, its slices, its state, its family, the version budget and the check years. No money."""
        return call(T.get_idea, idea=idea)

    @server.tool(meta=WHOLE)
    def list_blocks() -> dict[str, Any]:
        """Every block, do-not-enter rule and filter a form may name; his fixed rules (never changed); the numbers a
        version may change; the rules of a version; the conditions a slice may name; the budgets and the years."""
        return call(T.list_blocks)

    @server.tool(meta=WHOLE)
    def search_docs(query: str, hits: int = 5) -> dict[str, Any]:
        """Search his documents (strategy.md, LEARNED.md, strategy-understanding.md, his rules, the lesson inbox, his
        answers) before anything is assumed - his #129 rule. FOUND with citations, or NOT FOUND with the closest line.
        Words only: no prices, no results."""
        return call(T.search_docs, query=query, hits=hits)

    @server.tool(meta=WHOLE)
    def get_strategy() -> dict[str, Any]:
        """His WHOLE strategy (L19d): the entries in it (his words, steps, entry, hedge, target), the entries that
        wait for his words, one trade at a time, his management (fixed rows and the test rows a version may change),
        what a version may change, the version budget, the years, already seen. No money."""
        return call(T.get_strategy)

    @server.tool(meta=WHOLE)
    def propose_version(idea: str, why: str, readings: dict[str, str] | None = None,
                        changes: dict[str, Any] | None = None) -> dict[str, Any]:
        """Propose ONE version of the research idea: `readings` = a reading for every VERSION part (get_idea
        "unclear"), `changes` = dotted paths of the timeframe or the idea's own numbers, e.g.
        {"timeframe": "15m"}, {"steps.gap.max_wait_bars": 8}, {"targets.0.ratio": 2.0}, {"settings.near_zone_adr":
        0.1}. Refused: his fixed rules, a direction or filter (that is a slice), anything that changes what the idea
        IS, the 21st version of the family. Nothing runs; it counts as a try from now on. `why`: your reason."""
        return call(T.propose_version, idea=idea, why=why, readings=readings, changes=changes)

    @server.tool(meta=WHOLE)
    def run_version(idea: str, version: str, how: str = "tune") -> dict[str, Any]:
        """Run a version on 2016-2020: how="tune" = through the tester (trades, rupees by year, one trial file);
        how="walk-forward" = the referee judges the idea's versions inside 2016-2020 (choose on the years before,
        judge the next). Never 2021-2023, never 2024+. A long call: minutes."""
        return call(T.run_version, idea=idea, version=version, how=how, bench=bench)

    @server.tool(meta=WHOLE)
    def list_trials(idea: str) -> dict[str, Any]:
        """Every test run of the idea's family (plan/work/trials/): what ran, when, on which years, its numbers - the
        2021-2023 numbers only for this run's own idea once opened for that version. And the version budget."""
        return call(T.list_trials, idea=idea)

    @server.tool(meta=WHOLE)
    def draw_chart(idea: str, version: str, trade: int | str | None = None, start: str | None = None,
                   end: str | None = None, tf: str = "5m") -> dict[str, Any]:
        """ONE picture (a JPEG path): a trade of a version that ran (trade = 1 for its first closed trade, or a signal
        id), OR a stretch of time with the version's setups marked (start and end in MT5 server time, e.g.
        "2019-03-11 08:00", no time zone; tf 5m / 15m / 1h / 4h / 1d). Not sent to Tejas."""
        return call(T.draw_chart, idea=idea, version=version, trade=trade, start=start, end=end, tf=tf, bench=bench)

    @server.tool(meta=WHOLE)
    def write_report(idea: str, send: bool = False) -> dict[str, Any]:
        """The report, through L10: every version of the idea on 2016-2020, the referee chooses ONE (the highest
        Deflated Sharpe - the chance its edge is real, allowing for every version tried), 2021-2023 open ONCE for
        the family for that version and it is judged (a family that used them: walk-forward). Then the report and
        its picture. send=true also queues it for Tejas on Telegram. A long call: tens of minutes on real prices."""
        return call(T.write_report, idea=idea, send=send, bench=bench)

    @server.tool(meta=WHOLE)
    def choose_slices(idea: str, slices: list[dict[str, Any]], why: str) -> dict[str, Any]:
        """Choose the slices BEFORE any money is seen for the idea (at most 8 in all): each
        {"name": a condition from list_blocks, "dimension": its dimension, "how": in words, "why": ...}. Only these
        can become "works best in / fails in"; every other split is "looked at, not tested"."""
        return call(T.choose_slices, idea=idea, slices=slices, why=why)

    keep = PLAN_TOOLS if plan else STRATEGY_PLAN_TOOLS if strategy_plan else None
    for t in list(server._tool_manager.list_tools()):
        if (keep is not None and t.name not in keep) or (keep is None and t.name == "get_strategy"):
            server.remove_tool(t.name)          # get_strategy lives only on his whole strategy's planning server
    allowed ={name: set(inspect.signature(fn).parameters) for name, fn in
               ((t.name, t.fn) for t in server._tool_manager.list_tools())}

    async def strict_arguments(ctx, call_next):
        """Refuse an argument a tool does not have (the SDK would drop it without a word)."""
        if ctx.method == "tools/call" and isinstance(ctx.params, dict):
            name = ctx.params.get("name")
            args = ctx.params.get("arguments") or {}
            extra = sorted(set(args) - allowed.get(name, set())) if name in allowed and isinstance(args, dict) else []
            if extra:
                why = (f"REFUSED - {name} has no argument {', '.join(map(repr, extra))}. Its arguments: "
                       f"{', '.join(sorted(allowed[name]))}. Nothing unlocks the sealed years or opens the check "
                       "years from outside the tools.")
                T._log_refusal(name, args, why)
                return CallToolResult(content=[TextContent(type="text", text=why)], is_error=True)
        return await call_next(ctx)

    server.middleware.append(strict_arguments)
    return server


def describe(server: MCPServer) -> str:
    lines = [f"{NAME}: {len(server._tool_manager.list_tools())} tools · research target: "
             f"{T.SESSION.target or 'none (read-only)'}"]
    for t in server._tool_manager.list_tools():
        lines.append(f"  {t.name}({', '.join(inspect.signature(t.fn).parameters)})")
        lines.append(f"      {' '.join((t.description or '').split())[:300]}")
    return "\n".join(lines)
