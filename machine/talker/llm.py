"""One call to Claude, with the answer cache and the cost (L8). The only file in machine/ that calls Claude.

Two ways to reach Claude (Request.backend, default config.TALKER_BACKEND):
    "claude-code"  Tejas's own Claude Code login (his Claude Pro plan) through `claude -p` on his PC: no key, no bill
                   per call - it uses his plan's usage allowance. His words, 28 Sep 2026: no money on the Claude API
                   (inbox 2026-09-28-2020), "You can use the claude itself right the one I'm using now?"
    "api"          the Anthropic API with ANTHROPIC_API_KEY (paid per token) - kept for a PC or CI that has a key

    from machine.talker import llm
    req = llm.Request(system=[...], tools=[TOOL], user="...", tool="fill_idea_form")
    ans = llm.call(req)                  # ans.input = the tool's input (a dict), ans.cost_usd, ans.from_cache
    answers = llm.call_batch({"G01": req1, "G02": req2})     # the Message Batches API: half the price, minutes to wait
    a = llm.agent_call(system=..., user=..., schema=..., servers={...}, allowed=(...), model=..., effort=...)
                                         # Claude Code as an AGENT with MCP tools only (the research agent's plan, L14.2)

Answers are cached by (prompt hash, model): data/cache/talker/<hash>.json, where the hash covers everything sent
(system, tools, the message, the model, thinking, effort, max_tokens). The same prompt to the same model is never
paid for twice; any change - a new lesson in the search results, a new setting - is a new prompt.
Cost = the response's own token counts x the price table below (Anthropic's list prices, cached 24 Jun 2026 in the
claude-api skill: cache writes 1.25x, cache reads 0.1x the input price; the Batches API half of everything).

"api": the key is ANTHROPIC_API_KEY in the environment, or in .env (machine.config.env). Never printed, never cached.
"claude-code": the key is never passed on - ANTHROPIC_API_KEY is removed from its environment so it can never bill.
A refused request (stop_reason "refusal") is run again on another model by the API itself (server-side fallbacks,
"default" routing - the claude-api skill's default for this model); the model that answered is recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machine import config

CACHE_DIR = config.DATA_DIR / "cache" / "talker"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# $ per million tokens: (input, output). Cache write = 1.25x input (5-minute cache), cache read = 0.1x input.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}
CACHE_WRITE, CACHE_READ, BATCH = 1.25, 0.10, 0.50


class LLMError(RuntimeError):
    """The API gave no usable answer: no key, a refusal, the answer cut off, no tool call."""


@dataclass(frozen=True)
class Request:
    system: tuple[dict, ...]             # system blocks; the stable one carries cache_control
    tools: tuple[dict, ...]
    user: str
    tool: str                            # the tool the answer must call
    model: str = config.TALKER_MODEL
    effort: str = config.TALKER_EFFORT
    max_tokens: int = 16000
    backend: str = ""                        # "claude-code" | "api"; "" = config.TALKER_BACKEND when made

    def __post_init__(self):
        if not self.backend:
            object.__setattr__(self, "backend", config.TALKER_BACKEND)
        if self.backend not in ("claude-code", "api"):
            raise ValueError(f"backend {self.backend!r}: claude-code or api")

    def body(self) -> dict:
        """What is sent (and hashed). Thinking is adaptive (Claude decides how much); tool_choice stays auto - a
        forced tool choice cannot be combined with thinking - and the prompt says to call the tool."""
        return {"model": self.model, "max_tokens": self.max_tokens, "system": list(self.system),
                "tools": list(self.tools), "tool_choice": {"type": "auto"}, "thinking": {"type": "adaptive"},
                "output_config": {"effort": self.effort},
                "messages": [{"role": "user", "content": self.user}]}

    def key(self) -> str:
        text = json.dumps({"backend": self.backend, **self.body()}, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Tokens:
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    def __add__(self, o: "Tokens") -> "Tokens":
        return Tokens(self.input + o.input, self.output + o.output, self.cache_write + o.cache_write,
                      self.cache_read + o.cache_read)


@dataclass(frozen=True)
class Answer:
    input: dict | None                   # the tool call's input (None: no usable answer)
    model: str                           # the model that answered
    tokens: Tokens
    cost_usd: float                      # what this answer cost when it was made
    from_cache: bool
    batch: bool
    stop_reason: str
    error: str | None = None
    key: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def paid_now(self) -> float:
        return 0.0 if self.from_cache else self.cost_usd


def cost(model: str, t: Tokens, batch: bool = False) -> float:
    """Dollars for these tokens at the list price (an unknown model is priced as Opus 5, and says so in the report)."""
    inp, out = PRICES.get(model, PRICES["claude-opus-5"])
    usd = (t.input * inp + t.cache_write * inp * CACHE_WRITE + t.cache_read * inp * CACHE_READ + t.output * out) / 1e6
    return usd * (BATCH if batch else 1.0)


def _tokens(usage: Any) -> Tokens:
    g = (lambda k: getattr(usage, k, None) or 0) if not isinstance(usage, Mapping) else (lambda k: usage.get(k) or 0)
    return Tokens(g("input_tokens"), g("output_tokens"), g("cache_creation_input_tokens"), g("cache_read_input_tokens"))


# ------------------------------------------------------------------------------------------------ the answer cache
def _cache_file(key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{key}.json"


def cached(req: Request, cache_dir: Path | None = None) -> Answer | None:
    f = _cache_file(req.key(), cache_dir or CACHE_DIR)
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return Answer(d["input"], d["model"], Tokens(**d["tokens"]), d["cost_usd"], True, d.get("batch", False),
                  d["stop_reason"], None, req.key(), d.get("extra", {}))


def _save(req: Request, a: Answer, cache_dir: Path | None) -> None:
    if a.input is None:
        return                              # only usable answers are kept: a failure is tried again next time
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    d = {"model_asked": req.model, "model": a.model, "input": a.input, "tokens": a.tokens.__dict__,
         "cost_usd": a.cost_usd, "batch": a.batch, "stop_reason": a.stop_reason, "extra": a.extra,
         "made": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = _cache_file(req.key(), cache_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_cache_file(req.key(), cache_dir))


# ------------------------------------------------------------------------------------------------ the client
def client():
    """The Anthropic client. Raises LLMError (not a stack trace) when there is no key."""
    key = config.env("ANTHROPIC_API_KEY")
    try:
        import anthropic
    except ImportError as err:                       # pragma: no cover - requirements.txt has it
        raise LLMError("the anthropic package is missing: python -m pip install -r requirements.txt") from err
    if key:
        return anthropic.Anthropic(api_key=key)
    if config.env("ANTHROPIC_AUTH_TOKEN"):
        return anthropic.Anthropic()
    raise LLMError("no Anthropic key: put ANTHROPIC_API_KEY=... in .env (repo root, never committed) or in the "
                   "environment, then run it again")


def _plain(err: Exception) -> LLMError | None:
    """The SDK's typed errors, most specific first, as one plain line (None: not an API error - let it raise)."""
    try:
        import anthropic
    except ImportError:                              # pragma: no cover
        return None
    if isinstance(err, anthropic.AuthenticationError):
        return LLMError("the Anthropic key was refused (401): check ANTHROPIC_API_KEY in .env")
    if isinstance(err, anthropic.PermissionDeniedError):
        return LLMError(f"the key may not use this model or feature (403): {err.message}")
    if isinstance(err, anthropic.NotFoundError):
        return LLMError(f"not found (404) - is the model name right? {err.message}")
    if isinstance(err, anthropic.RateLimitError):
        return LLMError("rate limited (429): wait a minute and run again (answers so far are cached), or use --batch")
    if isinstance(err, anthropic.BadRequestError):
        return LLMError(f"the request was refused (400): {err.message}")
    if isinstance(err, anthropic.APIStatusError):
        return LLMError(f"Anthropic's side failed ({err.status_code}): run again later")
    if isinstance(err, anthropic.APIConnectionError):
        return LLMError("no connection to the Anthropic API: check the internet and run again")
    return None


def _create(api, **kw):
    try:
        return api.beta.messages.create(**kw, betas=[FALLBACK_BETA], fallbacks="default")
    except Exception as err:                         # noqa: BLE001 - only the SDK's own errors are turned into words
        plain = _plain(err)
        if plain is None:
            raise
        raise plain from err


def _read(msg: Any, req: Request, batch: bool) -> Answer:
    """One response -> an Answer: the tool call's input, or the reason there is none."""
    tokens = _tokens(msg.usage)
    served = getattr(msg, "model", req.model) or req.model
    usd = cost(served, tokens, batch)
    stop = getattr(msg, "stop_reason", "") or ""
    blocks = [b for b in msg.content if getattr(b, "type", "") == "tool_use" and getattr(b, "name", "") == req.tool]
    if stop == "refusal":
        details = getattr(msg, "stop_details", None)
        why = getattr(details, "category", None) if details else None
        return Answer(None, served, tokens, usd, False, batch, stop, f"refused ({why or 'no category'})", req.key())
    if stop == "max_tokens":
        return Answer(None, served, tokens, usd, False, batch, stop, "the answer was cut off (max_tokens)", req.key())
    if not blocks:
        return Answer(None, served, tokens, usd, False, batch, stop, f"no {req.tool} call in the answer", req.key())
    data = blocks[0].input
    if isinstance(data, str):                        # the SDK gives a dict; a raw string is parsed, never matched
        data = json.loads(data)
    return Answer(dict(data), served, tokens, usd, False, batch, stop, None, req.key())


# ------------------------------------------------------------------------------------------------ his Claude Code
CLAUDE_CODE_TIMEOUT = 900        # seconds for one answer (Claude's number: a form takes 1-3 minutes)


def claude_exe() -> str:
    """The Claude Code program itself. npm's `claude.cmd` wrapper runs through cmd.exe, which reads < > | & inside
    the form's schema as commands - so the real claude.exe behind it is called. CLAUDE_EXE in .env overrides."""
    given = config.env("CLAUDE_EXE")
    if given:
        return given
    found = shutil.which("claude")
    if not found:
        raise LLMError("Claude Code is not found: the 'claude' command is not on this PC's PATH (or set CLAUDE_EXE "
                       "in .env to claude.exe)")
    if found.lower().endswith((".cmd", ".ps1", ".bat")):
        exe = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if exe.is_file():
            return str(exe)
    return found


def _claude_code(req: Request, runner=None, timeout: float = CLAUDE_CODE_TIMEOUT) -> Answer:
    """`claude -p` with his login: no tools, his system text from a file (a Windows command line holds 32,767
    characters; the text is ~60,000), the message on standard input, the answer forced into the tool's schema
    (--json-schema). Its tokens are counted; the dollars are 0 - his plan pays (Claude Code's own price estimate is
    kept as api_price_usd). His global ~/.claude/CLAUDE.md is still read by Claude Code; only the form is used."""
    runner = runner or subprocess.run
    tool = next(t for t in req.tools if t["name"] == req.tool)
    work = CACHE_DIR / "claude-code-cwd"            # an empty folder: the project's CLAUDE.md is not read
    work.mkdir(parents=True, exist_ok=True)
    sysfile = work / f"{req.key()[:16]}.system.txt"
    sysfile.write_text("\n\n".join(b["text"] for b in req.system), encoding="utf-8")
    args = [claude_exe(), "-p", "--tools", "", "--no-session-persistence", "--output-format", "json",
            "--model", req.model, "--effort", req.effort, "--system-prompt-file", str(sysfile),
            "--json-schema", json.dumps(tool["input_schema"], ensure_ascii=False, separators=(",", ":"))]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    try:
        p = runner(args, input=req.user, capture_output=True, text=True, encoding="utf-8", cwd=str(work), env=env,
                   timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))  # the intake (L10b) runs
        # with no window on his PC: without this flag each call would open a black window
    except subprocess.TimeoutExpired:
        raise LLMError(f"claude -p gave no answer in {timeout:.0f} seconds") from None
    finally:
        sysfile.unlink(missing_ok=True)
    try:
        d = json.loads(p.stdout)
    except ValueError:
        raise LLMError(f"claude -p did not answer in JSON (exit {p.returncode}): "
                       f"{(p.stderr or p.stdout or '').strip()[:300]}") from None
    u = d.get("usage") or {}
    tokens = _tokens(u)
    models = [m for m in (d.get("modelUsage") or {}) if "haiku" not in m] or [req.model]
    extra = {"via": "claude-code (his login; no bill per call)", "api_price_usd": d.get("total_cost_usd"),
             "seconds": round((d.get("duration_ms") or 0) / 1000, 1)}
    if d.get("is_error"):
        why = str(d.get("result") or d.get("subtype") or "an error")[:300]
        if "limit" in why.lower():
            why += " - his plan's usage limit: run again when it resets (answers so far are cached)"
        return Answer(None, models[0], tokens, 0.0, False, False, d.get("subtype") or "error", f"claude -p: {why}",
                      req.key(), extra)
    form = d.get("structured_output")
    if not isinstance(form, dict):
        return Answer(None, models[0], tokens, 0.0, False, False, d.get("subtype") or "", "claude -p: no form in the "
                      "answer", req.key(), extra)
    return Answer(form, models[0], tokens, 0.0, False, False, d.get("subtype") or "success", None, req.key(), extra)


@dataclass(frozen=True)
class AgentAnswer:
    """What one agent run of Claude Code gave back (agent_call)."""
    output: dict | None                  # its answer in the schema's shape (None: no usable answer - see error)
    model: str
    tool_calls: tuple[dict, ...]         # every tool it called, in order: {"tool", "input", "error", "said"}
    turns: int
    seconds: float
    tokens: Tokens
    api_price_usd: float | None          # Claude Code's own estimate; his plan pays (0 extra)
    error: str | None = None
    text: str = ""                       # its last words outside the answer


def agent_call(*, system: str, user: str, schema: dict, servers: dict, allowed: tuple[str, ...], model: str,
               effort: str, timeout: float = CLAUDE_CODE_TIMEOUT, runner=None, work: Path | None = None) -> AgentAnswer:
    """Claude Code (his login, like _claude_code) as an AGENT: it may call ONLY the MCP tools named in `allowed`, from
    ONLY the servers given (`--strict-mcp-config`), and NO built-in tool (`--tools ""`: no Bash, Read, Edit - an agent
    that can read files walks past every guard inside the tools, machine/mcp/README.md). Its answer is forced into
    `schema` (--json-schema); every tool call it made comes back too (--output-format stream-json), for the log.
    servers: {"name": {"command", "args", "env"}} - written with json.dumps (a Git Bash heredoc halves backslashes).
    A server that did not start gives an error here, never a quiet answer without tools (seen in L14.1). The same
    for a tool answer Claude Code cut for its length (seen in the 8 Oct rehearsal, L10b)."""
    runner = runner or subprocess.run
    work = Path(work or CACHE_DIR / "claude-agent-cwd")   # an empty folder: the project's CLAUDE.md is not read
    work.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256(f"{time.time_ns()}{user}".encode("utf-8")).hexdigest()[:16]
    sysfile, cfg = work / f"{stamp}.system.txt", work / f"{stamp}.mcp.json"
    sysfile.write_text(system, encoding="utf-8")
    cfg.write_text(json.dumps({"mcpServers": servers}, ensure_ascii=False), encoding="utf-8")
    args = [claude_exe(), "-p", "--tools", "", "--strict-mcp-config", "--mcp-config", str(cfg),
            "--allowedTools", ",".join(allowed), "--no-session-persistence", "--output-format", "stream-json",
            "--verbose", "--model", model, "--effort", effort, "--system-prompt-file", str(sysfile),
            "--json-schema", json.dumps(schema, ensure_ascii=False, separators=(",", ":"))]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    t0 = time.monotonic()
    try:
        p = runner(args, input=user, capture_output=True, text=True, encoding="utf-8", cwd=str(work), env=env,
                   timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return AgentAnswer(None, model, (), 0, timeout, Tokens(), None, f"claude -p gave no answer in {timeout:.0f} s")
    finally:
        sysfile.unlink(missing_ok=True)
        cfg.unlink(missing_ok=True)
    return read_agent_stream(p.stdout or "", model=model, seconds=time.monotonic() - t0,
                             stderr=p.stderr or "", servers=tuple(servers))


def _cut(said: str | None) -> bool:
    """Claude Code did not give Claude this tool answer: it saved it to a file ('<persisted-output> Output too large')
    or refused it ('exceeds maximum allowed tokens'). Claude has no tool that reads a file here."""
    s = (said or "").lstrip()
    return s.startswith("<persisted-output>") or "exceeds maximum allowed tokens" in s[:200]


def read_agent_stream(out: str, *, model: str, seconds: float, stderr: str = "",
                      servers: tuple[str, ...] = ()) -> AgentAnswer:
    """claude -p's stream-json lines -> an AgentAnswer: the tool calls in order (each with its result), the answer."""
    calls: dict[str, dict] = {}
    order: list[str] = []
    started: dict[str, str] = {}
    result: dict | None = None
    text = ""
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        kind = d.get("type")
        if kind == "system" and d.get("subtype") == "init":
            started = {s.get("name", "?"): s.get("status", "?") for s in d.get("mcp_servers") or []}
        elif kind == "assistant":
            for c in (d.get("message") or {}).get("content") or []:
                if c.get("type") == "tool_use" and c.get("name") != "StructuredOutput":
                    calls[c["id"]] = {"tool": str(c.get("name", "")).split("__")[-1], "input": c.get("input"),
                                      "error": None, "said": None}
                    order.append(c["id"])
                elif c.get("type") == "text" and c.get("text"):
                    text = c["text"]
        elif kind == "user":
            for c in (d.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("tool_use_id") in calls:
                    body = c.get("content")
                    if isinstance(body, list):
                        body = " ".join(str(x.get("text", "")) for x in body if isinstance(x, dict))
                    calls[c["tool_use_id"]].update(error=bool(c.get("is_error")), said=str(body or "")[:400])
        elif kind == "result":
            result = d
    tool_calls = tuple(calls[i] for i in order)
    bad = [f"{n}: {s}" for n, s in started.items() if s != "connected"] + [n for n in servers if n not in started]
    if result is None:
        return AgentAnswer(None, model, tool_calls, 0, seconds, Tokens(), None,
                           f"claude -p gave no result (stderr: {stderr.strip()[:300]})", text)
    tokens = _tokens(result.get("usage") or {})
    used = [m for m in (result.get("modelUsage") or {}) if "haiku" not in m] or [model]
    turns = int(result.get("num_turns") or 0)
    common = dict(model=used[0], tool_calls=tool_calls, turns=turns, seconds=round(seconds, 1), tokens=tokens,
                  api_price_usd=result.get("total_cost_usd"), text=text)
    if bad:
        return AgentAnswer(None, error=f"its tools did not start ({'; '.join(bad)}) - the reason is in the newest "
                                       r"file of ~\.claude\debug (grep 'MCP server')", **common)
    if result.get("is_error"):
        why = str(result.get("result") or result.get("subtype") or "an error")[:300]
        if "limit" in why.lower():
            why += " - his plan's usage limit: run again when it resets"
        return AgentAnswer(None, error=f"claude -p: {why}", **common)
    cut = [c["tool"] for c in tool_calls if _cut(c["said"])]
    if cut:                                                # Claude answered WITHOUT that tool's answer (L10b rehearsal)
        return AgentAnswer(None, error=f"Claude Code cut the answer of {', '.join(dict.fromkeys(cut))} (too long "
                                       "for it), so Claude answered without reading it - see WHOLE in "
                                       "machine/mcp/server.py", **common)
    answer = result.get("structured_output")
    if not isinstance(answer, dict):
        return AgentAnswer(None, error="claude -p: no answer in the plan's shape", **common)
    return AgentAnswer(answer, **common)


def call(req: Request, *, api=None, runner=None, cache_dir: Path | None = None, use_cache: bool = True) -> Answer:
    """One request, real time. From the cache when the same prompt went to the same model the same way before."""
    if use_cache:
        hit = cached(req, cache_dir)
        if hit is not None:
            return hit
    if req.backend == "claude-code":
        a = _claude_code(req, runner)
        _save(req, a, cache_dir)
        return a
    api = api or client()
    body = req.body()
    msg = _create(api, **body)
    a = _read(msg, req, batch=False)
    if a.input is None and a.stop_reason not in ("refusal", "max_tokens"):
        # it answered in words instead of the tool: ask once more, on the same conversation (append-only)
        again = dict(body, messages=body["messages"] + [
            {"role": "assistant", "content": [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b
                                              for b in msg.content]},
            {"role": "user", "content": f"Answer only by calling {req.tool}."}])
        msg2 = _create(api, **again)
        b = _read(msg2, req, batch=False)
        a = Answer(b.input, b.model, a.tokens + b.tokens, a.cost_usd + b.cost_usd, False, False, b.stop_reason,
                   b.error, req.key())
    _save(req, a, cache_dir)
    return a


def call_batch(reqs: Mapping[str, Request], *, api=None, cache_dir: Path | None = None, use_cache: bool = True,
               poll_seconds: float = 20.0, max_wait_seconds: float = 3 * 3600, say=print) -> dict[str, Answer]:
    """Many requests through the Message Batches API (half price; usually minutes, at most 24 hours). Cached ones are
    not sent. The results come back in any order: they are matched by custom_id, never by position."""
    out: dict[str, Answer] = {}
    todo = {}
    for cid, req in reqs.items():
        hit = cached(req, cache_dir) if use_cache else None
        if hit is not None:
            out[cid] = hit
        else:
            todo[cid] = req
    if not todo:
        return out
    if any(r.backend == "claude-code" for r in todo.values()):     # his login has no batches: one by one
        for cid, req in todo.items():
            try:
                out[cid] = call(req, api=api, cache_dir=cache_dir, use_cache=False)
            except LLMError as err:
                out[cid] = Answer(None, req.model, Tokens(), 0.0, False, False, "", str(err), req.key())
            say(f"{cid}: " + (out[cid].error or f"{out[cid].extra.get('seconds', '?')} s"))
        return out
    api = api or client()
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest
    try:
        batch = api.messages.batches.create(requests=[
            BatchRequest(custom_id=cid, params=MessageCreateParamsNonStreaming(**r.body())) for cid, r in todo.items()])
    except Exception as err:                         # noqa: BLE001 - only the SDK's own errors are turned into words
        plain = _plain(err)
        if plain is None:
            raise
        raise plain from err
    say(f"batch {batch.id}: {len(todo)} requests sent")
    t0 = time.monotonic()
    while True:
        b = api.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        if time.monotonic() - t0 > max_wait_seconds:
            raise LLMError(f"batch {batch.id} still running after {max_wait_seconds / 3600:.1f} hours - its answers "
                           "stay at Anthropic for 29 days; run again later (finished answers come from the batch)")
        say(f"batch {batch.id}: {b.request_counts.processing} still running")
        time.sleep(poll_seconds)
    for res in api.messages.batches.results(batch.id):
        req = todo[res.custom_id]
        if res.result.type == "succeeded":
            a = _read(res.result.message, req, batch=True)
        else:
            err = getattr(getattr(res.result, "error", None), "type", None) or res.result.type
            a = Answer(None, req.model, Tokens(), 0.0, False, True, "", f"batch request {res.result.type}: {err}",
                       req.key())
        _save(req, a, cache_dir)
        out[res.custom_id] = a
    for cid, req in todo.items():                    # a request the batch never reported
        out.setdefault(cid, Answer(None, req.model, Tokens(), 0.0, False, True, "", "no result in the batch",
                                   req.key()))
    return out
