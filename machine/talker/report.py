"""A talker draft as plain text (L8): what was read, searched, drafted, asked, checked, and what it cost."""
from __future__ import annotations

import csv
from functools import lru_cache

from machine import config
from machine.ideas.form import REF


@lru_cache(maxsize=1)
def _prior() -> dict[str, dict]:
    with open(config.PRIOR_TRIES_CSV, encoding="utf-8", newline="") as f:
        return {r["idea_key"]: r for r in csv.DictReader(f)}


def _wait(w) -> str:
    return "-" if w is None else ("no limit" if w == "no limit" else f"{w} bars")


def form_lines(f) -> list[str]:
    blocks = {s.id: s.block for s in f.steps}

    def price(ref):
        m = REF.fullmatch(ref or "")
        return f"{ref} ({blocks.get(m.group(1), '?')})" if m else str(ref)

    out = [f"chart {f.timeframe} · {', '.join(f.directions)}", "steps:"]
    for k, s in enumerate(f.steps, 1):
        p = f" {s.params}" if s.params else ""
        out.append(f"  {k}. {s.id}: {s.block} on {s.tf}, wait {_wait(s.max_wait_bars)}{p} - {s.what}")
    e = f.entry
    out.append("entry: " + (f"limit at {price(e.at)}, waits {_wait(e.max_wait_bars)}" if e.how == "limit"
                            else "at the next bar's open"))
    x = f.exit
    out.append(f"{x.kind}: at {price(x.at)}" + (f" + ${x.beyond_usd:g}" if x.beyond_usd else "")
               + (f", management {x.management}" if getattr(x, "management", None) else ""))
    for t in f.targets:
        what = {"ratio": lambda: f"{t.ratio:g} x the {x.kind} distance",
                "price": lambda: price(t.at),
                "next_liquidity": lambda: f"the next liquidity ({', '.join(t.kinds) if t.kinds else 'his usual'}; "
                                          f"{t.pick})"}[t.kind]()
        out.append(f"target: {what}")
    for title, rows in (("do not enter", f.do_not_enter), ("filter", f.filters)):
        for r in rows:
            out.append(f"{title}: {r.name}" + (f" {r.params}" if r.params else ""))
    if f.settings:
        out.append(f"settings: {f.settings}")
    return out


def draft_text(d) -> str:
    L = [f"HIS WORDS: {d.words}"]
    g = d.glossary
    for line in g.lines():
        L.append(f"  voice typing: {line}")
    for heard, means in d.voice:
        L.append(f'  voice typing (the talker): "{heard}" = "{means}"')
    L.append(f"SEARCH (#129): {len(d.searches)} terms")
    for s in d.searches:
        pick = d.picks.get(s.term.name, "(no pick)")
        first = s.hits[0].id if s.hits else "-"
        L.append(f'  "{s.term.name}" (he said "{s.term.said}"): search {"FOUND" if s.found else "NOT FOUND"}, first '
                 f"{first}; the talker's pick: {pick or 'none answers it'}")
    if d.form is None:
        L.append("FORM: none")
    else:
        L.append(f"FORM: {d.status} · {d.kind} · {d.form.title}")
        L += ["  " + x for x in form_lines(d.form)]
        for name, where, prompt in d.missing:
            L.append(f"  needs (not built): {name} ({where}) -> {prompt}")
        for m in (d.wire or {}).get("missing", []):
            L.append(f"  needs (no name covers it): {m['name']} - {m['what']}")
    if d.questions:
        L.append("UNCLEAR:")
        for q, u in zip(d.questions, d.form.unclear):
            L.append(f"  {u.id} {u.kind.upper()} (term \"{q.term}\"): {u.part}"
                     + (f' - his words "{u.words}"' if u.words else ""))
            for r in u.readings:
                L.append(f"     {r.id}: {r.means}" + (f"  {r.changes}" if r.changes else "  (the form as written)"))
            if u.kind == "ask":
                L.append(f"     {q.searched}")
                if q.answered_by:
                    L.append(f"     !! the talker's own pick says {q.answered_by} answers this term")
    elif d.form is not None:
        L.append("UNCLEAR: none")
    for key in d.already_seen:
        r = _prior().get(key)
        L.append(f"ALREADY SEEN: {key}" + (f" - {r['tries']} tries on {r['years']}: those years are not fresh for it"
                                           if r else " (not in prior-tries.csv)"))
    for c in d.sanity:
        L.append(f"CHECK [{c.severity}{'' if c.ok else ', not ok'}] {c.name}: {c.message}")
    for p in d.problems:
        L.append(f"PROBLEM: {p}")
    if d.fixup:
        L.append(f"FIX-UP: code sent {len(d.fixup)} findings back once; kept {d.kept}")
        for x in d.fixup:
            L.append(f"  - {x[:300]}")
    for u in d.unknown_citations:
        L.append(f"CITATION NOT SHOWN TO IT: {u}")
    for a in d.answers():
        if a.input is None:
            continue
        t = a.tokens
        toks = (f"tokens: {t.input:,} in, {t.cache_read:,} cached in, {t.cache_write:,} written to the cache, "
                f"{t.output:,} out")
        if str(a.extra.get("via", "")).startswith("claude-code"):
            L.append(f"CLAUDE: {a.model} through his own Claude Code login - $0 extra (his plan's usage)"
                     f"{', from the cache' if a.from_cache else ''}; at API prices ${a.extra.get('api_price_usd') or 0:.2f}"
                     f"; {a.extra.get('seconds', '?')} s ({toks})")
        else:
            how = "from the cache, first made for" if a.from_cache else "cost"
            L.append(f"API: {a.model}, {how} ${a.cost_usd:.4f} ({toks}{'; batch' if a.batch else ''})")
    return "\n".join(L)
