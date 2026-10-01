"""The talker's exam (L8): tests/evals/golden.jsonl - 25 sentences (16 his real words) from plan/work/golden-ideas.md
(T4), each with the answer his documents give. Written before the talker ran (git 099b917).

    python -m machine.talker.evals                    # the exam, through his own Claude Code login ($0 extra)
    python -m machine.talker.evals --backend api --batch   # with an API key instead: paid, half price in a batch
    python -m machine.talker.evals --dry-run          # no call: the prompt sizes and what it would cost
    python -m machine.talker.evals --only G01,G09     # some rows (no pass mark)

PASS MARK (plan/likhith.md L8): >= 90% of fields right, >= 90% of unclear parts found, 0 invalid forms, <= 1 needless
question per 10 sentences (<= 2 for 25). Every run is saved to plan/work/evals/ (the number of runs is the number of
tries) with the cost.

What is scored - only what golden-ideas.md STATES for a sentence (a field it leaves open is not scored):
  kind         "idea", or the rule kind: "rule:skip" "rule:entry" "rule:exit" "rule:filter" "rule:management"
  timeframe    the idea's chart          directions   buy / sell
  entry        "limit <block>.<what>" of the step the entry names, e.g. "limit gap_of_the_move.near" (a * = anything)
  exit         "<hedge|stop> <block>.<what>", e.g. "hedge break_of_structure.extreme"
  target       any target: "ratio 1", "price <block>.<what>", "next_liquidity <kinds>"
  needs        each name (block, rule, filter, liquidity kind, or a not-built name) the idea must use; "a|b" = either
  blocked      whether it waits for a piece that is not built
  voice        each voice-typing mistake read right (by the glossary or by the talker)
  already_seen the old tries it must name (prior-tries.csv); [] = it must name none
  unclear      each ASK / VERSION part, found when a part of the SAME kind mentions one of its words (part, his words,
               term); matched one to one. A needless question = an ASK part that matches none.
An invalid form (no answer, a form that does not load, a check that refuses it, a sanity "problem") scores 0 on every
field of its sentence and finds none of its unclear parts.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from machine import config
from machine.talker import llm, translate as T

GOLDEN = config.ROOT / "tests" / "evals" / "golden.jsonl"
OUT_DIR = config.WORK_DIR / "evals"
PASS_FIELDS = 0.90
PASS_UNCLEAR = 0.90
PASS_INVALID = 0
PASS_FALSE_PER_10 = 1
PLAN_USD = 1.00            # the plan: "print the API cost (<= about $1 a run)"
GUESS_OUT_TOKENS = 3000    # dry run only: Claude's guess of one answer's output + thinking tokens


def load(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------ what a draft says
def rendered(d: T.Draft) -> dict:
    f = d.form
    blocks = {s.id: s.block for s in f.steps}

    def ref(r):
        step, _, what = (r or "").partition(".")
        return f"{blocks.get(step, '?')}.{what}"

    targets = []
    for t in f.targets:
        if t.kind == "ratio":
            targets.append(f"ratio {t.ratio:g}")
        elif t.kind == "price":
            targets.append(f"price {ref(t.at)}")
        else:
            targets.append("next_liquidity" + (f" {','.join(t.kinds)}" if t.kinds else ""))
    return {"kind": d.kind, "timeframe": f.timeframe, "directions": sorted(f.directions),
            "entry": f"limit {ref(f.entry.at)}" if f.entry.how == "limit" else "next_open",
            "exit": f"{f.exit.kind} {ref(f.exit.at)}", "targets": targets, "needs": d.needs(),
            "blocked": d.status == "BLOCKED", "already_seen": sorted(d.already_seen),
            "voice": [(r.heard, r.means) for r in d.glossary.replaced] + list(d.voice)}


def _any(value: str, patterns) -> bool:
    patterns = [patterns] if isinstance(patterns, str) else patterns
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)


def _word_in(small: str, big: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(small.lower())}(?!\w)", big.lower()) is not None


def _voice_ok(item: dict, voice: list[tuple[str, str]]) -> bool:
    for heard, means in voice:
        if _word_in(item["heard"], heard) or _word_in(heard, item["heard"]):
            if any(k.lower() in means.lower() for k in item["any"]):
                return True
    return False


def _match(expected: list[dict], produced: list[tuple[str, str]]) -> dict[int, int]:
    """One-to-one matching (augmenting paths): expected part i -> produced part j of the same kind whose text has one
    of its words."""
    edges = {i: [j for j, (kind, text) in enumerate(produced)
                 if kind == e["kind"] and any(k.lower() in text for k in e["any"])] for i, e in enumerate(expected)}
    owner: dict[int, int] = {}

    def augment(i: int, seen: set) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in edges:
        augment(i, set())
    return {i: j for j, i in owner.items()}


# ------------------------------------------------------------------------------------------------ one sentence
@dataclass
class RowScore:
    id: str
    real: bool
    fields: list[tuple[str, bool, object, object]] = field(default_factory=list)   # (field, right, expected, got)
    unclear_expected: int = 0
    unclear_found: int = 0
    missed: list[str] = field(default_factory=list)
    wrong_kind: list[str] = field(default_factory=list)
    false_questions: list[str] = field(default_factory=list)
    extra_versions: list[str] = field(default_factory=list)
    invalid: bool = False
    invalid_why: list[str] = field(default_factory=list)
    draft: T.Draft | None = None

    @property
    def right(self) -> int:
        return sum(ok for _, ok, _, _ in self.fields)


def score_row(row: dict, d: T.Draft) -> RowScore:
    e = row["expect"]
    s = RowScore(row["id"], row["real"], draft=d)
    s.invalid_why = list(d.problems) + [f"sanity: {c.message}" for c in d.sanity if c.severity == "problem"]
    s.invalid = not d.valid or bool(s.invalid_why)
    got = rendered(d) if d.form is not None and not s.invalid else None

    def add(name, ok, exp, have):
        s.fields.append((name, bool(ok) and got is not None, exp, have))

    g = got or {}
    for name in ("kind", "timeframe"):
        if name in e:
            add(name, g.get(name) == e[name], e[name], g.get(name))
    if "directions" in e:
        add("directions", g.get("directions") == sorted(e["directions"]), e["directions"], g.get("directions"))
    for name in ("entry", "exit"):
        if name in e:
            add(name, got and _any(g[name], e[name]), e[name], g.get(name))
    if "target" in e:
        add("target", got and any(_any(t, e["target"]) for t in g["targets"]), e["target"], g.get("targets"))
    if "blocked" in e:
        add("blocked", g.get("blocked") == e["blocked"], e["blocked"], g.get("blocked"))
    for need in e.get("needs", []):
        add(f"needs {need}", got and any(a in g["needs"] for a in need.split("|")), need,
            sorted(g.get("needs", [])))
    for v in e.get("voice", []):
        add(f"voice {v['heard']!r}", got and _voice_ok(v, g["voice"]), v["any"], g.get("voice"))
    if "already_seen" in e:
        exp = sorted(e["already_seen"])
        ok = got and (set(exp) <= set(g["already_seen"]) if exp else not g["already_seen"])
        add("already_seen", ok, exp, g.get("already_seen"))
    # unclear parts
    expected = e.get("unclear", [])
    s.unclear_expected = len(expected)
    produced = []
    if got is not None:
        for q, u in zip(d.questions, d.form.unclear):
            produced.append((u.kind, " ".join(x for x in (u.part, u.words or "", q.term) if x).lower()))
    m = _match(expected, produced)
    s.unclear_found = len(m)
    for i, x in enumerate(expected):
        if i not in m:
            s.missed.append(f"{x['kind']} {x['any']}")
            other = [(("version" if k == "ask" else "ask"), t) for k, t in produced]
            if _match([x], other):
                s.wrong_kind.append(f"{x['kind']} {x['any']} was marked {'version' if x['kind'] == 'ask' else 'ask'}")
    used = set(m.values())
    for j, (kind, text) in enumerate(produced):
        if j in used:
            continue
        label = f"{d.form.unclear[j].id}: {d.form.unclear[j].part}"
        (s.false_questions if kind == "ask" else s.extra_versions).append(label)
    return s


def totals(scores: list[RowScore]) -> dict:
    n = len(scores)
    fields = sum(len(s.fields) for s in scores)
    right = sum(s.right for s in scores)
    ue = sum(s.unclear_expected for s in scores)
    uf = sum(s.unclear_found for s in scores)
    drafts = [s.draft for s in scores if s.draft is not None]
    answers = [a for d in drafts for a in d.answers() if a.input is not None]
    tokens = llm.Tokens()
    for a in answers:
        tokens = tokens + a.tokens
    return {
        "sentences": n, "real": sum(s.real for s in scores),
        "fields_right": right, "fields_total": fields, "fields_rate": right / fields if fields else 0.0,
        "unclear_found": uf, "unclear_total": ue, "unclear_rate": uf / ue if ue else 1.0,
        "invalid": sum(s.invalid for s in scores),
        "false_questions": sum(len(s.false_questions) for s in scores),
        "false_limit": PASS_FALSE_PER_10 * n // 10,
        "extra_versions": sum(len(s.extra_versions) for s in scores),
        "unknown_citations": sum(len(d.unknown_citations) for d in drafts),
        "answers": len(answers), "from_cache": sum(a.from_cache for a in answers),
        "fixups": sum(bool(d.fixup) for d in drafts),
        "fixups_that_made_it_valid": sum(bool(d.fixup and d.valid and d.other and not d.other.valid) for d in drafts),
        "models": sorted({a.model for a in answers}),
        "paid_now_usd": round(sum(a.paid_now for a in answers), 4),
        "api_price_usd": round(sum(a.extra.get("api_price_usd") or 0 for a in answers), 4),
        "seconds": round(sum(a.extra.get("seconds") or 0 for a in answers if not a.from_cache), 1),
        "cost_all_usd": round(sum(a.cost_usd for a in answers), 4),
        "tokens": tokens.__dict__,
    }


def passed(t: dict) -> bool:
    return (t["fields_rate"] >= PASS_FIELDS and t["unclear_rate"] >= PASS_UNCLEAR and t["invalid"] <= PASS_INVALID
            and t["false_questions"] <= t["false_limit"])


def report(scores: list[RowScore], t: dict, meta: dict) -> str:
    via = meta.get("backend", "api")
    L = [f"talker exam - {meta['when']} - model {meta['model']} (effort {meta['effort']}) - via {via}"
         f"{' - batch' if meta['batch'] and via == 'api' else ''} - search: {meta['search']}"]
    if meta.get("note"):
        L.append(f"NOTE: {meta['note']}")
    L.append("")
    for s in scores:
        flag = "INVALID" if s.invalid else f"{s.right}/{len(s.fields)} fields"
        d = s.draft
        if d is not None and d.fixup:
            flag += f" (after a fix-up: {len(d.fixup)} findings; kept {d.kept})"
        L.append(f"{s.id} {'real ' if s.real else 'style'} {flag}; unclear {s.unclear_found}/{s.unclear_expected}"
                 + (f"; NEEDLESS QUESTIONS {s.false_questions}" if s.false_questions else ""))
        for why in s.invalid_why:
            L.append(f"      invalid: {why[:400]}")
        for name, ok, exp, have in s.fields:
            if not ok:
                L.append(f"      wrong {name}: expected {exp}, got {have}")
        for x in s.missed:
            L.append(f"      missed unclear: {x}")
        for x in s.wrong_kind:
            L.append(f"      {x}")
        for x in s.extra_versions:
            L.append(f"      extra version (not a question): {x}")
    L += ["",
          f"fields right: {t['fields_right']} of {t['fields_total']} = {t['fields_rate']:.0%} (mark >= {PASS_FIELDS:.0%})",
          f"unclear parts found: {t['unclear_found']} of {t['unclear_total']} = {t['unclear_rate']:.0%} "
          f"(mark >= {PASS_UNCLEAR:.0%})",
          f"invalid forms: {t['invalid']} (mark {PASS_INVALID})",
          f"needless questions: {t['false_questions']} in {t['sentences']} sentences (mark <= {t['false_limit']})",
          f"extra versions (tried by the machine, not asked): {t['extra_versions']} · citations not shown to it: "
          f"{t['unknown_citations']}",
          f"answers: {t['answers']} ({t['from_cache']} from the cache) · models: {', '.join(t['models']) or '-'} · "
          f"fix-up calls: {t['fixups']} ({t['fixups_that_made_it_valid']} turned a refused form into a good one)",
          (f"COST: $0 extra - his own Claude Code login (Claude Pro plan usage); the same tokens at API prices would "
           f"be ${t['api_price_usd']:.2f} (Claude Code's own estimate) · {t['seconds']:.0f} s of answering · tokens "
           f"{t['tokens']}" if via == "claude-code" else
           f"API COST: ${t['paid_now_usd']:.2f} paid by this run (the answers themselves cost ${t['cost_all_usd']:.2f} "
           f"when made; the plan: about ${PLAN_USD:.0f} a run) · tokens {t['tokens']}")]
    if meta.get("full"):
        L.append(f"PASS MARK: {'PASS' if passed(t) else 'NOT MET'}")
    else:
        L.append("(some rows only: no pass mark)")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ running it
def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001 - a missing git only loses the stamp
        return "unknown"


def save(scores: list[RowScore], t: dict, meta: dict, text: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob("L8-talker-*-run*.json"))) + 1
    base = out_dir / f"L8-talker-{meta['when']}-run{n}"
    rows = []
    for s in scores:
        d = s.draft
        rows.append({"id": s.id, "invalid": s.invalid, "invalid_why": s.invalid_why,
                     "fields": [{"field": f, "right": ok, "expected": exp, "got": have if not isinstance(have, set)
                                 else sorted(have)} for f, ok, exp, have in s.fields],
                     "unclear_found": s.unclear_found, "unclear_expected": s.unclear_expected, "missed": s.missed,
                     "false_questions": s.false_questions, "extra_versions": s.extra_versions,
                     "status": d.status if d else None, "answer": d.wire if d else None,
                     "fixup": d.fixup if d else [], "kept": d.kept if d else "",
                     "other_answer": d.other.wire if d and d.other else None,
                     "error": d.answer.error if d and d.answer else None})
    doc = {"run": n, **meta, "totals": t, "passed": passed(t) if meta.get("full") else None, "rows": rows}
    base.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str) + "\n",
                                         encoding="utf-8")
    base.with_suffix(".md").write_text(f"# L8 talker exam, run {n}\n\n```text\n{text}\n```\n", encoding="utf-8")
    return base


def run(rows: list[dict], *, model: str, effort: str, batch: bool = False, use_cache: bool = True,
        budget_usd: float | None = None, search=None, api=None, runner=None, backend: str | None = None,
        cache_only: bool = False, fixups: int | None = None, say=print) -> list[RowScore]:
    """Every row: its first answer, then - for the rows code refused or whose checks require a question - ONE fix-up
    answer (config.TALKER_FIXUPS), the better form kept. Both rounds run the same way (real time, batch, or only
    the saved answers)."""
    search = search or T.default_search()
    prepared = {}
    for r in rows:
        g, searches, req = T.prepare(r["words"], search, model, effort, backend)
        prepared[r["id"]] = (g, searches, req)
    spent = [0.0]

    def answer_all(reqs: dict[str, llm.Request], label: str) -> dict[str, llm.Answer]:
        if cache_only:                              # re-score saved answers: never asks Claude
            return {k: llm.cached(q) or llm.Answer(None, model, llm.Tokens(), 0.0, False, False, "",
                                                   "not in the cache (--cache-only)", q.key()) for k, q in reqs.items()}
        if batch:
            return llm.call_batch(reqs, api=api, use_cache=use_cache, say=say)
        out = {}
        for k, q in reqs.items():
            if budget_usd is not None and spent[0] >= budget_usd and llm.cached(q) is None:
                out[k] = llm.Answer(None, model, llm.Tokens(), 0.0, False, False, "",
                                    f"not run: the budget ${budget_usd:.2f} was spent", q.key())
                continue
            try:
                a = llm.call(q, api=api, runner=runner, use_cache=use_cache)
            except llm.LLMError as err:
                a = llm.Answer(None, model, llm.Tokens(), 0.0, False, False, "", str(err), q.key())
            spent[0] += a.paid_now
            out[k] = a
            done = "cache" if a.from_cache else (f"{a.extra.get('seconds', '?')} s, $0 extra (his plan)"
                                                 if q.backend == "claude-code" else f"${a.cost_usd:.3f}")
            say(f"{k}{label}: {done}" + (f" - {a.error}" if a.error else ""))
        return out

    first = answer_all({k: v[2] for k, v in prepared.items()}, "")
    drafts = {}
    for k, r in enumerate(rows, 1):
        g, searches, _ = prepared[r["id"]]
        drafts[r["id"]] = T.build(r["words"], first.get(r["id"]), g, searches, f"idea-9{k:02d}", search)
    if (config.TALKER_FIXUPS if fixups is None else fixups) > 0:
        findings = {i: f for i, d in drafts.items() if (f := T.fixup_findings(d))}
        second = answer_all({i: T.fixup_request(prepared[i][2], drafts[i], f) for i, f in findings.items()},
                            " (fix-up)")
        for k, r in enumerate(rows, 1):
            i = r["id"]
            if i in findings:
                g, searches, _ = prepared[i]
                d2 = T.build(r["words"], second.get(i), g, searches, f"idea-9{k:02d}", search)
                drafts[i] = T.better(drafts[i], d2, findings[i])
    return [score_row(r, drafts[r["id"]]) for r in rows]


def dry_run(rows: list[dict], model: str, effort: str, search=None, backend: str | None = None) -> str:
    search = search or T.default_search()
    ref = T.reference()
    ref_tokens = len(ref) // 4 + len(json.dumps(T.tool())) // 4
    user_tokens = []
    for r in rows:
        _, _, req = T.prepare(r["words"], search, model, effort, backend)
        user_tokens.append(len(req.user) // 4)
    n = len(rows)
    per = llm.Tokens(input=sum(user_tokens), output=GUESS_OUT_TOKENS * n, cache_write=ref_tokens,
                     cache_read=ref_tokens * (n - 1))
    now, half = llm.cost(model, per), llm.cost(model, per, batch=True)
    return (f"dry run, no API call: {n} sentences, model {model} (effort {effort})\n"
            f"the same for every sentence (cached by the API): ~{ref_tokens:,} tokens\n"
            f"each sentence's own part: ~{min(user_tokens):,} to ~{max(user_tokens):,} tokens "
            f"(~{sum(user_tokens) // n:,} on average)\n"
            f"output: a guess of {GUESS_OUT_TOKENS:,} tokens each (thinking + the form)\n"
            + (f"via his Claude Code login: $0 extra (his Claude Pro plan's usage); at API prices it would be ~${now:.2f}"
               if (backend or config.TALKER_BACKEND) == "claude-code" else
               f"estimated cost: ${now:.2f} real time, ${half:.2f} with --batch")
            + " (tokens = characters / 4, a rough rule)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="rows, e.g. G01,G09")
    ap.add_argument("--model", default=config.TALKER_MODEL)
    ap.add_argument("--effort", default=config.TALKER_EFFORT, choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--backend", default=config.TALKER_BACKEND, choices=("claude-code", "api"),
                    help="claude-code = his own login, no bill per call (default); api = ANTHROPIC_API_KEY, paid")
    ap.add_argument("--batch", action="store_true", help="api only: the Message Batches API, half the price")
    ap.add_argument("--no-cache", action="store_true", help="ask again even when the same prompt was answered")
    ap.add_argument("--budget-usd", type=float, default=3.0, help="real time: stop asking after this much is spent")
    ap.add_argument("--dry-run", action="store_true", help="no API call: prompt sizes and a cost estimate")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--cache-only", action="store_true", help="re-score saved answers only: never asks Claude")
    ap.add_argument("--note", default="", help="one line kept with the run (why it was run)")
    ap.add_argument("--fixups", type=int, default=config.TALKER_FIXUPS, help="fix-up calls per form (0 = none)")
    args = ap.parse_args(argv)
    rows = load()
    full = not args.only
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        rows = [r for r in rows if r["id"] in want]
    search = T.default_search()
    if args.dry_run:
        print(dry_run(rows, args.model, args.effort, search, args.backend))
        return 0
    from machine.search import index as ix
    meta = {"when": date.today().isoformat(), "model": args.model, "effort": args.effort, "batch": args.batch,
            "backend": args.backend, "note": args.note, "cache_only": args.cache_only, "fixups": args.fixups,
            "search": ix.get().how, "git": _git(), "full": full, "golden": str(GOLDEN.relative_to(config.ROOT)),
            "prompt": hashlib.sha256(T.reference().encode("utf-8")).hexdigest()[:12]}
    scores = run(rows, model=args.model, effort=args.effort, batch=args.batch, use_cache=not args.no_cache,
                 budget_usd=args.budget_usd, search=search, backend=args.backend, cache_only=args.cache_only,
                 fixups=args.fixups)
    t = totals(scores)
    text = report(scores, t, meta)
    print(text)
    if t["answers"] == 0:
        print("\nNo answer at all - nothing saved. " + (scores[0].draft.problems[0] if scores and scores[0].draft
                                                        and scores[0].draft.problems else ""))
        return 2
    if not args.no_save:
        print(f"\nsaved: {save(scores, t, meta, text).relative_to(config.ROOT)}.json / .md")
    return 0 if (not full or passed(t)) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
