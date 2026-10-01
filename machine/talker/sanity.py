"""Code checks on a talker draft (L8): what a language model can get wrong quietly, checked by code.

    from machine.talker import sanity
    for c in sanity.check(draft):         # machine.talker.translate.Draft
        print(c.severity, c.name, c.message)

The checks (plan/likhith.md L8 step 3):
  golden_ratio       a number next to "golden ratio" / "fib" that is not a Fibonacci ratio - "0.68" next to "golden
                     ratio" (the golden ratio is 0.618, his F1 question): the draft must ASK it, never pick one
  impossible_levels  the hedge or a target at the entry price; Fibonacci levels in an order no trade can have (a buy's
                     target below its entry); "1:1" with a Fibonacci entry and target that need a stop he never gave
                     (from 0.618 to 0.5 at 1:1 the stop must be near 0.736 - his F2 question): the draft must ASK it
  unknown_timeframe  a chart he names that the machine does not have (2-hour, 30-minute, monthly); a chart he names
                     that the form never uses
  fixed_rule         the form changes one of his fixed rules (refused with his words), or sets one of his settings
                     to something else (only a comparison: it is listed)
Severity: "problem" = the form cannot stand as written (an invalid form); "ask" = a question the draft must have
and does not; "note" = worth reading, nothing wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from machine import rules, settings
from machine.ideas.form import REF, TF

FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 0.886)   # the usual Fibonacci levels (0.705: his OTE set)
GOLDEN = 0.618
NEAR_WORDS = 12                                                 # "next to" = within this many words


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    severity: str        # "problem" | "ask" | "note"
    message: str


def _words(text: str) -> list[str]:
    return re.findall(r"[\w.:]+", text.lower())


def _asks(d, *needles: str) -> bool:
    """An ASK part of the draft whose part / his words / term mention any of the needles."""
    for q, u in zip(d.questions, d.form.unclear if d.form else ()):
        if u.kind != "ask":
            continue
        text = " ".join(x for x in (u.part, u.words or "", q.term) if x).lower()
        if any(n in text for n in needles):
            return True
    return False


# ------------------------------------------------------------------------------------------------ golden ratio
def golden_ratio(d) -> list[Check]:
    w = _words(d.glossary.text)
    out = []
    for i, tok in enumerate(w):
        if not re.fullmatch(r"0?\.\d{1,3}", tok):
            continue
        x = float(tok)
        near = " ".join(w[max(0, i - NEAR_WORDS): i + NEAR_WORDS + 1])
        if not re.search(r"golden|fib|ratio|retrac", near) or any(abs(x - r) < 1e-9 for r in FIB_RATIOS):
            continue
        golden = "golden" in near
        why = (f'"{tok}" next to "golden ratio": the golden ratio is {GOLDEN}' if golden
               else f'"{tok}" next to a Fibonacci word is not a Fibonacci ratio ({", ".join(map(str, FIB_RATIOS))})')
        asked = _asks(d, tok, "0.618", "golden", "ratio")
        out.append(Check("golden_ratio", asked, "note" if asked else "ask",
                         why + ("; the draft asks him which one" if asked else
                                " - the draft must ASK him which one, not choose (his F1 question)")))
    return out


# ------------------------------------------------------------------------------------------------ impossible levels
def _fib(ref: str | None, blocks: dict[str, str]) -> float | None:
    if not ref:
        return None
    m = REF.fullmatch(ref)
    if not m or blocks.get(m.group(1)) != "fib_move" or not m.group(2).startswith("level_"):
        return None
    try:
        return float(m.group(2)[len("level_"):])
    except ValueError:
        return None


def _ratio_said(text: str) -> float | None:
    m = re.search(r"\b1\s*(?:is to|:|to)\s*(\d+(?:\.\d+)?)\b", text.lower())
    return float(m.group(1)) if m else None


def impossible_levels(d) -> list[Check]:
    f = d.form
    if f is None:
        return []
    out = []
    blocks = {s.id: s.block for s in f.steps}
    e_at = f.entry.at
    if e_at and f.exit.at == e_at and f.exit.beyond_usd <= 0:       # "a few pips past the line" (Type 2) is fine
        out.append(Check("impossible_levels", False, "problem",
                         f"the {f.exit.kind} is at the entry price ({e_at}): no distance, no trade"))
    for k, t in enumerate(f.targets):
        if e_at and t.at == e_at:
            out.append(Check("impossible_levels", False, "problem", f"targets[{k}] is the entry price ({e_at})"))
    e, s = _fib(e_at, blocks), _fib(f.exit.at, blocks)
    for k, t in enumerate(f.targets):
        tt = _fib(t.at, blocks)
        if e is not None and tt is not None and not tt < e:
            out.append(Check("impossible_levels", False, "problem",
                             f"targets[{k}] level {tt:g} is not past the entry level {e:g} toward the move (a target "
                             "must be a smaller retracement than the entry)"))
    if e is not None and s is not None and not s > e:
        out.append(Check("impossible_levels", False, "problem",
                         f"the {f.exit.kind} level {s:g} is not beyond the entry level {e:g} (it must be a deeper "
                         "retracement)"))
    n = _ratio_said(d.glossary.text) or next((v for h, v in _voice_ratios(d)), None)
    fib_targets = [_fib(t.at, blocks) for t in f.targets if _fib(t.at, blocks) is not None]
    if n and e is not None and fib_targets:
        tt = fib_targets[0]
        need = e + (e - tt) / n
        if s is None or abs(s - need) > 0.01:
            asked = _asks(d, "stop", "hedge", "1:", "1 is to")
            out.append(Check("impossible_levels", asked, "note" if asked else "ask",
                             f"1:{n:g} from level {e:g} to level {tt:g} needs the {f.exit.kind} near level "
                             f"{need:.3f}; the form has {f.exit.at}"
                             + ("; the draft asks where his stop is" if asked else
                                " - the draft must ASK where his stop is (his F2 question)")))
    if f.exit.beyond_usd > 50:
        out.append(Check("impossible_levels", False, "note",
                         f"the {f.exit.kind} is ${f.exit.beyond_usd:g} past its level - a very long way"))
    return out


def _voice_ratios(d):
    for heard, means in d.voice:
        v = _ratio_said(means)
        if v:
            yield heard, v


# ------------------------------------------------------------------------------------------------ timeframes
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15, "thirty": 30, "sixty": 60}
_TF_RX = re.compile(r"(?<![\w.])(\d+|one|two|three|four|five|ten|fifteen|thirty|sixty)\s*-?\s*"
                    r"(minutes?|mins?|m|hours?|hourly|hrs?|h)\b(?!\s*(?:is\b|:))"
                    r"|\b(daily|weekly|monthly|yearly)\b", re.IGNORECASE)
_KNOWN = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h", 1440: "1d", 10080: "1w"}


def charts_said(text: str) -> list[tuple[str, str | None]]:
    """(his words, the machine's chart or None) for every chart he names."""
    out = []
    for m in _TF_RX.finditer(text):
        if m.group(3):
            word = m.group(3).lower()
            minutes = {"daily": 1440, "weekly": 10080}.get(word)
        else:
            g1 = m.group(1).lower()
            n = int(g1) if g1.isdigit() else _NUM[g1]
            unit = m.group(2).lower()
            minutes = n * (60 if unit.startswith("h") else 1)
        out.append((m.group(0), _KNOWN.get(minutes) if minutes else None))
    return out


def timeframes(d) -> list[Check]:
    out = []
    used = set()
    if d.form is not None:
        used = {d.form.timeframe} | {s.tf for s in d.form.steps}
    for said, tf in charts_said(d.glossary.text):
        if tf is None:
            out.append(Check("unknown_timeframe", False, "ask",
                             f'he names "{said}": the machine has no such chart ({", ".join(TF.__args__)}) - ask, '
                             "never swap in another chart"))
        elif d.form is not None and tf not in used and tf not in ("1d", "1w"):
            out.append(Check("unknown_timeframe", False, "note",
                             f'he names "{said}" ({tf}); the form never uses the {tf} chart'))
    return out


# ------------------------------------------------------------------------------------------------ fixed rules
def fixed_rules(d) -> list[Check]:
    out = [Check("fixed_rule", False, "problem", p) for p in d.problems if p.startswith("FixedRuleError")]
    out += [Check("fixed_rule", False, "problem", p) for p in d.problems if "FixedRuleError" in p
            and not p.startswith("FixedRuleError")]
    f = d.form
    if f is not None:
        try:
            r = settings.resolve(f.settings)
            out += [Check("fixed_rule", True, "note", line) for line in r.breaks_his_rules]
        except (settings.SettingError, rules.FixedRuleError):
            pass                                            # already a problem of the form
    return out


def check(d) -> list[Check]:
    return golden_ratio(d) + impossible_levels(d) + timeframes(d) + fixed_rules(d)
