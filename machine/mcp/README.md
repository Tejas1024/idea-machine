# machine/mcp — the research tools (L14.1)

The research agent (L14.2) researches ONE of Tejas's ideas by calling these tools over MCP (the Model Context
Protocol). The guards are **inside the tools**, so the agent cannot get around them by asking cleverly: none of the
strategies an LLM found survived honest judging (Gençay 2026, arXiv 2608.27734). The AI writes rules; code judges.

| file | what |
|---|---|
| `tools.py` | the 9 tools as plain Python, every guard inside them |
| `versions.py` | plan/likhith.md section D as code: what a version may change, its family, the 20-version budget |
| `server.py` | the tools over MCP (the official `mcp` SDK, pinned in requirements.txt); an argument a tool does not have is refused, not dropped |
| `__main__.py` | `python -m machine.mcp --idea idea-002` (stdio) · `--list` |

## The tools

| tool | does | refuses |
|---|---|---|
| `get_idea` | the idea: his words, steps, unclear parts, slices, state, family, version budget, check years — no money | not one of his ideas (`idea-NNN`) |
| `list_blocks` | blocks, rules, filters; his fixed rules; the numbers a version may change; slice conditions | — |
| `search_docs` | his #129 search of his documents (FOUND with citations / NOT FOUND + closest line) | — (words only: no prices, no results) |
| `choose_slices` | at most 8 slices, chosen before any money | after money was seen (lesson 18); a 9th; an unknown condition |
| `propose_version` | one version = readings of the VERSION parts + changes of the timeframe or the idea's own numbers | his fixed rules (his words), a slice dressed as a version, what the idea IS, the 21st version of the family |
| `run_version` | `tune`: the tester on 2016–2020 (a trial file) · `walk-forward`: the referee inside 2016–2020 | any other years; a version not proposed; the budget |
| `list_trials` | the family's trial files and the version budget | 2021–2023 numbers unless opened for this run's idea and that version; 2024+; kinds whose years it cannot prove |
| `draw_chart` | ONE picture: a trade of a version, or a stretch of time (server time, no time zone) | 2024+ always; 2021–2023 before opened for that version |
| `write_report` | via L10: every version on 2016–2020, the referee chooses ONE (highest Deflated Sharpe), 2021–2023 open ONCE per family for it, the report | before his YES at SHOWN; no version yet |

**One run = one idea.** `--idea idea-002` binds the server: only that idea's versions can be proposed, run, charted
or reported, and only its own opened check years are ever shown. Without `--idea` the server is read-only.

**Money waits for him.** No tool shows money for an idea until its meaning is locked AND he said YES to the 3 setups
the machine found (SHOWN, L10b) — otherwise his NO could no longer reopen it (lesson 16).

Every refusal says which rule and why, and is written to `data/mcp/refusals.jsonl` (`MLBOT_MCP_DIR` moves it).

## How the research agent (L14.2) uses them

`machine/agent/research.py` splits the work: **Claude only plans**, through the PLANNING server
`python -m machine.mcp --plan` - read-only, and only `get_idea`, `list_blocks`, `search_docs` exist in it (no tool that
proposes, runs, draws, reports or shows a trial file). Code checks the plan with the same rules
(`versions.check_version`, nothing counted), saves it, and then calls the other tools itself, in process
(`choose_slices`, `propose_version`, `run_version`, `write_report`) - exactly the plan, every guard here still inside.
The call is `machine/talker/llm.agent_call` (the command below, with `--output-format stream-json` so every tool call
lands in the research log, and `--json-schema` for the plan's shape).

## How an agent gets them (Claude Code, his login — no API key)

```
claude -p "<the research prompt>" --tools "" --strict-mcp-config --mcp-config research.json ^
       --allowedTools "mcp__research__*" --no-session-persistence
```
with `research.json`:
```json
{"mcpServers": {"research": {"command": "C:\\ML Bot\\.venv\\Scripts\\python.exe",
                             "args": ["-m", "machine.mcp", "--idea", "idea-002"],
                             "env": {"PYTHONPATH": "C:\\ML Bot"}}}}
```
Claude Code's MCP config has no working-folder field: without `PYTHONPATH` the server starts in the agent's folder
and cannot find `machine` (seen 1 Oct 2026: "No module named 'machine'", and the agent got no tools). Write this file
with Python's `json.dumps` - a Git Bash heredoc halves the backslashes. Checked 1 Oct 2026 with a real agent
(`--model claude-haiku-4-5-20251001`, his login): it called get_idea, then tried a 2024 chart and `how='check'` -
both refused, both written to the refusals file.

- `--tools ""` turns off EVERY built-in tool (Bash, Read, Edit, Grep …). **This is the one thing the guards cannot do
  for themselves**: an agent that can read `data/` or `plan/work/trials/` directly walks past every guard here.
- `--strict-mcp-config`: only this server, no other MCP server of the PC.
- A long call (`run_version` on real prices ≈ 2 minutes; `write_report` tens of minutes): give Claude Code room with
  `MCP_TOOL_TIMEOUT` (milliseconds) in its environment.
- What a write_report leaves (state TUNE RUN / CHECK OPENED, the report) is what the intake (L10b) reads: on his PC
  its next run sends the report to Telegram if `send` was false.
