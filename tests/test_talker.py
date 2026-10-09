"""L8, the talker: the glossary, the terms, the tool, the answer -> form, the checks, the cache, the cost, the exam.

No test calls the API: a fake client answers (the real run is `python -m machine.talker.evals`, with a key).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from machine import config
from machine.ideas import compile as C
from machine.ideas.form import Idea
from machine.search.rank import Answer as SearchAnswer
from machine.search.rank import Hit
from machine.talker import evals, glossary, llm, sanity, terms
from machine.talker import translate as T

ROOT = config.ROOT


# ================================================================================================ fakes
class _Block(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def _msg(inp=None, stop="tool_use", model=None, text=None, tool=T.TOOL_NAME,
         usage=(1000, 2000, 0, 15000)):
    content = []
    if text:
        content.append(_Block(type="text", text=text))
    if inp is not None:
        content.append(_Block(type="tool_use", id="toolu_1", name=tool, input=inp))
    u = dict(zip(("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"), usage))
    return SimpleNamespace(content=content, stop_reason=stop, model=model, usage=u, stop_details=None)


class FakeAPI:
    """Answers from a list (a message, or a function of the request). Records every request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(batches=SimpleNamespace(create=self._bcreate, retrieve=self._bget,
                                                                results=self._bresults))
        self._batch: list = []

    def _next(self, kw):
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        m = copy.copy(a(kw) if callable(a) else a)
        if m.model is None:                                   # the model asked for answers, unless a test says otherwise
            m.model = kw["model"]
        return m

    def _create(self, **kw):
        self.calls.append(kw)
        return self._next(kw)

    def _bcreate(self, requests):
        self._batch = [(r["custom_id"], dict(r["params"])) for r in requests]
        return SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    def _bget(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=SimpleNamespace(processing=0))

    def _bresults(self, bid):
        for cid, params in reversed(self._batch):          # any order: matched by custom_id
            self.calls.append(params)
            yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=self._next(params)))


def _hit(id, quote="a quote", date="2026-09-20", rank=1, found=True):
    return Hit(id, "LEARNED", 1, date, "strategy/LEARNED.md", 1, quote, 3.0, 0.8, 0.6, rank, found)


def fake_search(found: bool = True):
    """Every query: 3 hits; FOUND or NOT FOUND for all."""
    def search(q):
        hits = (_hit("LEARNED #127", "the steep of the line decides"), _hit("LEARNED #92", "3 touches", rank=2),
                _hit("understanding §66bz", "F1 0.68 - the golden ratio is 0.618", rank=3))
        if found:
            return SearchAnswer(q, True, hits[:1], hits[0], f'Found in {hits[0].id}: "{hits[0].quote}"', "fake", hits)
        s = f'Searched strategy.md, LEARNED.md, strategy-understanding.md, rules, pictures for "{q}" — closest line: x'
        return SearchAnswer(q, False, (), hits[0], s, "fake", hits)
    return search


def wire(**changes) -> dict:
    w = copy.deepcopy(T.EXAMPLE)
    w.update(copy.deepcopy(changes))
    return w


def draft(w: dict, words: str = "sell after the london high is taken and the fvg", found: bool = True) -> T.Draft:
    g = glossary.apply(words)
    searches = T.run_searches(g.text, fake_search(found))
    a = llm.Answer(w, "claude-opus-5", llm.Tokens(), 0.0, False, False, "tool_use")
    return T.build(words, a, g, searches, "idea-900", search=fake_search(found))


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "talker-cache")
    monkeypatch.setattr(config, "TALKER_BACKEND", "api")          # the fake client; the claude-code tests say so
    monkeypatch.setattr(config, "env", lambda name, default=None: "test-key" if name == "ANTHROPIC_API_KEY" else default)


# ================================================================================================ 1. the glossary
def _learned_notes() -> list[str]:
    """Every "X" his voice typing wrote, from the speech-to-text / voice-typing notes of LEARNED.md."""
    heard = []
    for line in (ROOT / "strategy/LEARNED.md").read_text(encoding="utf-8").splitlines():
        if not re.search(r"speech-to-text|voice typing", line, re.I):
            continue
        for sent in re.split(r"(?<=\.)\s+(?=[A-Z*\"(])", line):
            if not re.search(r"speech-to-text|voice typing", sent, re.I):
                continue
            for clause in re.split(r"[,;]\s*", sent):
                if "=" not in clause:
                    continue
                left = clause.split("=")[0]
                # quotes on each side of "voice typing" apart: a quoted question before it must not pair its closing
                # quote with the note's opening one (LEARNED #141: '...to you?" Voice typing: "forest factory" =')
                heard += [q.lower() for part in re.split(r"speech-to-text|voice typing", left, flags=re.I)
                          for q in re.findall(r'"([^"]+)"', part)]
    return heard


def test_every_speech_to_text_note_in_learned_has_a_row():
    rows = {r.heard for r in glossary.load()}
    notes = _learned_notes()
    assert len(set(notes)) >= 19, notes
    missing = sorted(set(notes) - rows)
    assert not missing, f"voice-typing notes with no glossary row: {missing}"
    assert "the liquid" in rows                                     # LEARNED #136: Likhith


def test_every_glossary_row_is_in_the_lesson_it_names():
    text = (ROOT / "strategy/LEARNED.md").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=## #\d+)", text)
    for r in glossary.load():
        nums = re.findall(r"#(\d+)", r.lesson)
        assert nums, r
        bodies = [e for e in entries if any(re.match(rf"## #{n}\b", e) for n in nums)]
        assert any(r.heard in b.lower() for b in bodies), f"{r.heard!r} is not in {r.lesson}"


def test_the_glossary_replaces_phrases_and_only_hints_normal_words():
    g = glossary.apply("the Favorable Gap after the heads point, stop loss below; the liquid will test the liquidity")
    assert g.text == "the fair value gap after the hedge point, stop loss below; Likhith will test the liquidity"
    assert [(r.heard, r.means) for r in g.replaced] == [("Favorable Gap", "fair value gap"),
                                                      ("heads point", "hedge point"), ("the liquid", "Likhith")]
    assert [h.heard for h in g.hints] == ["loss"]                     # "stop loss" is never changed to "stop lows"
    assert "loss" in " ".join(g.lines()) and "not replaced" in " ".join(g.lines())
    assert glossary.apply("liquidity only").replaced == ()


# ================================================================================================ 2. the terms
def test_terms_vague_words_first_then_his_order():
    t = terms.find("in strong uptrend buy at steep trendline 3rd touch no fvg needed hedge few pips below")
    names = [x.name for x in t]
    assert {"steep", "few pips", "very strong / strong trend", "trendline", "third touch", "hedge"} <= set(names)
    assert names == sorted(names, key=lambda n: next(x.at for x in t if x.name == n))
    steep = next(x for x in t if x.name == "steep")
    assert steep.vague and "steep trendline" in steep.query
    many = terms.find(" ".join(["steep", "fvg", "trendline", "zone", "news", "hedge", "target", "retest", "wedge",
                                "triangle", "near", "halfway", "stop", "liquidity"]))
    assert len(many) == terms.MAX_TERMS and {"steep", "near", "halfway"} <= {x.name for x in many}


# ================================================================================================ 3. the tool
def _walk(s, path="$"):
    if isinstance(s, dict):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False, path
            assert sorted(s["required"]) == sorted(s["properties"]), path
        for k in ("minimum", "maximum", "minLength", "maxLength", "pattern"):
            assert k not in s, f"{path}: {k} is not allowed in strict mode"
        for k, v in s.items():
            _walk(v, f"{path}.{k}")
    elif isinstance(s, list):
        for i, v in enumerate(s):
            _walk(v, f"{path}[{i}]")


def test_the_tool_is_strict_and_the_example_is_a_good_form():
    t = T.tool()
    assert t["strict"] is True and t["name"] == "fill_idea_form"
    _walk(t["input_schema"])
    f = Idea.load(T.to_form(T.EXAMPLE, "idea-900", "words"))
    assert T.check_form(f) == ([], [])
    assert json.dumps(T.EXAMPLE, ensure_ascii=False) in T.reference()


def test_the_reference_holds_every_rule_setting_block_box_and_old_try():
    ref = T.reference()
    from machine import rules, settings
    for name in list(rules.ALL) + list(settings.ALL) + list(C.BLOCKS) + list(C.PLANNED):
        assert name in ref, name
    for needle in ("Entry TF = **5-min always**", "## Box 2", "B2-01", "B3-16", "fib_0618", "halfway_rule",
                   "Type 1", "#61, #62, #131"):
        assert needle in ref, needle


def test_check_without_bars_blocks_refuses_and_passes():
    f = Idea.load(T.to_form(T.EXAMPLE, "idea-900", "w"))
    assert C.check_without_bars(f) == []
    tl = copy.deepcopy(T.EXAMPLE)
    tl["steps"][0]["block"] = "trendline"
    missing = C.check_without_bars(Idea.load(T.to_form(tl, "idea-900", "w")))
    assert [m[0] for m in missing] == ["trendline"] and "L6b.1" in missing[0][2]


# ================================================================================================ 4. answer -> draft
def test_a_good_answer_makes_a_drafted_form_with_its_checks():
    w = wire(terms=[{"term": "session high / low", "answered_by": "LEARNED #127"}])
    d = draft(w)
    assert d.valid and d.status == "FORM DRAFTED" and d.form.status == "FORM DRAFTED"
    assert d.form.words == "sell after the london high is taken and the fvg"          # his words, as typed
    assert d.picks == {"session high / low": "LEARNED #127"} and not d.unknown_citations
    assert "FORM: FORM DRAFTED" in d.text() and "UNCLEAR: none" in d.text()


def test_a_limit_at_a_gap_gets_his_g8_rule_by_code():
    """strategy.md section 2 G8: 'If price took the target-side liquidity before coming back -> no entry', and his #68
    first visit (the setup's gaps, §65e). Idea #4's form (7 Oct 2026) was drafted without them; code adds both to
    every limit order at a gap, once."""
    both = ["target_reached_first", "price_comes_back_into_another_gap_first"]
    d = draft(wire())                                          # his usual setup: a limit at the gap's near edge
    assert d.valid and not d.missing and [r.name for r in d.form.do_not_enter] == both
    assert "G8" in d.form.do_not_enter[0].sources[0].ref and "#68" in d.form.do_not_enter[1].sources[0].ref
    w = wire(do_not_enter=[{"name": "target_reached_first", "params": [], "words": None, "sources": ["his words"]}])
    assert [r.name for r in draft(w).form.do_not_enter] == both                               # never twice
    assert draft(w).form.do_not_enter[0].sources[0].ref == "his words"                        # his own kept
    w = wire(entry={"how": "next_open", "at": None, "max_wait": None, "sources": ["his words"]})
    assert not draft(w).form.do_not_enter                       # no limit order: nothing to cancel
    w = wire()
    w["entry"]["at"] = "brk.extreme"                            # a limit, but not at a gap
    assert not T.with_his_gap_rules(T.to_form(w, "idea-900", "w"))["do_not_enter"]


def test_a_kind_named_with_its_side_is_its_family_not_a_missing_block():
    """Found on the first real intake run (L10b, 29 Sep): the talker wrote "session_high" (the side in the name) and
    the idea was BLOCKED for a block that is built. The side comes from the trade's direction."""
    w = wire()
    w["steps"][0]["params"] = [{"name": "kinds", "value_json": "[\"session_high\", \"equal_highs\"]"}]
    w["targets"][0]["kinds"] = ["session_low", "swing_low"]
    d = draft(w)
    assert d.valid and d.status == "FORM DRAFTED" and not d.missing
    assert tuple(d.form.steps[0].params["kinds"]) == ("session", "equal")
    assert tuple(d.form.targets[0].kinds) == ("session", "swing")
    assert T.family_kinds(["zone_high", "trendline"]) == ["zone_high", "trendline"]   # still missing: still asked for


def test_a_block_that_is_not_built_makes_the_idea_blocked_not_invalid():
    w = wire()
    w["steps"][0]["block"] = "trendline"
    w["filters"] = [{"name": "news", "params": [{"name": "only", "value_json": "\"red\""}], "words": None,
                     "sources": ["his words"]}]
    d = draft(w)
    assert d.valid and d.status == "BLOCKED"
    assert {"trendline", "news"} <= {m[0] for m in d.missing} and {"trendline", "news"} <= d.needs()
    w2 = wire(missing=[{"name": "close_before_weekend", "what": "close the trade on Friday evening"}])
    d2 = draft(w2)
    assert d2.valid and d2.status == "BLOCKED" and "close_before_weekend" in d2.needs()


def test_an_invented_setting_a_fixed_rule_or_a_bad_reading_is_an_invalid_form():
    d = draft(wire(settings=[{"name": "close_before_weekend", "value_json": "true"}]))
    assert not d.valid and d.status == "NO FORM" and "SettingError" in d.problems[0]
    w = wire()
    w["entry"]["at"] = "gap.far"                                   # his #65: the NEAR edge, a fixed rule
    d = draft(w)
    assert not d.valid and "FixedRuleError" in d.problems[0] and "#65" in d.problems[0]
    assert any(c.name == "fixed_rule" and c.severity == "problem" for c in d.sanity)
    w = wire(unclear=[{"id": "U1", "kind": "version", "term": "near", "part": "how near", "words": None,
                       "readings": [{"id": "A", "means": "as written", "changes": []},
                                    {"id": "B", "means": "a step that is not there",
                                     "changes": [{"path": "steps.nope.block", "value_json": "\"sweep\""}]}]}])
    d = draft(w)
    assert not d.valid and "reading B of U1" in d.problems[0]
    d = draft(wire(settings=[{"name": "state_timeout_bars", "value_json": "thirty"}]))
    assert not d.valid and "not JSON" in d.problems[0]


def test_a_reading_may_use_the_tools_own_shape():
    """Run 2 of the exam: 4 forms were refused only because a reading wrote a whole filter / target list the way the
    tool writes it (sources as strings, params as name + value_json). Code converts it; a real error still fails."""
    filt = [{"name": "bias", "params": [{"name": "charts", "value_json": "[\"1w\"]"}], "words": None,
             "sources": ["his words", "LEARNED #70"]}]
    stop = {"kind": "stop", "at": "brk.extreme", "beyond_usd": 0, "management": [], "sources": ["his words"]}
    u = {"id": "U1", "kind": "version", "term": "near", "part": "which week", "words": None,
         "readings": [{"id": "A", "means": "as written", "changes": []},
                      {"id": "B", "means": "the week must agree",
                       "changes": [{"path": "filters", "value_json": json.dumps(filt)}]},
                      {"id": "C", "means": "a plain stop", "changes": [{"path": "exit", "value_json": json.dumps(stop)}]}]}
    d = draft(wire(unclear=[u]))
    assert d.valid, d.problems
    ch = d.form.unclear[0].readings[1].changes["filters"][0]
    assert ch["params"] == {"charts": ["1w"]} and ch["sources"] == [{"ref": "his words"}, {"ref": "LEARNED #70"}]
    assert "management" not in d.form.unclear[0].readings[2].changes["exit"]
    bad = copy.deepcopy(u)
    bad["readings"][1]["changes"] = [{"path": "steps", "value_json": json.dumps([
        {"id": "gap", "what": "g", "block": "gap_of_the_move", "tf": "5m", "max_wait": 6, "params": [], "sources": []},
        {"id": "sweep", "what": "s", "block": "sweep", "tf": "5m", "max_wait": "first step", "params": [],
         "sources": []}])}]
    assert not draft(wire(unclear=[bad])).valid                        # a reading that breaks the blocks still fails


def test_citations_must_be_among_the_quotes_it_was_shown():
    w = wire(terms=[{"term": "session high / low", "answered_by": "LEARNED #999"}])
    w["steps"][0]["sources"] = ["LEARNED #127", "made up place"]
    d = draft(w)
    assert d.valid
    assert any("LEARNED #999" in u for u in d.unknown_citations)
    assert any("made up place" in u for u in d.unknown_citations)
    assert not any("LEARNED #127" in u for u in d.unknown_citations)
    w = wire(terms=[{"term": "session high / low", "answered_by": "term1: LEARNED #127"}])
    w["steps"][0]["sources"] = ['his words ("london high taken")', "strategy.md section 10", "four-boxes B3-16",
                                "not built yet: trendline"]
    d = draft(w)
    assert d.picks["session high / low"] == "LEARNED #127" and not d.unknown_citations, d.unknown_citations
    d = draft(wire(already_seen=["no_such_try", "fib_0618"]))
    assert any("no_such_try" in u for u in d.unknown_citations) and not any("fib_0618" in u for u in
                                                                              d.unknown_citations)


def test_an_ask_carries_the_129_sentence_made_by_code_and_a_pick_is_flagged():
    ask = {"id": "Q1", "kind": "ask", "term": "session high / low", "part": "which session high counts",
           "words": "london high", "readings": [{"id": "A", "means": "London only", "changes": []},
                                                {"id": "B", "means": "any session", "changes": []}]}
    d = draft(wire(unclear=[ask]), found=False)
    q = d.questions[0]
    assert q.searched.startswith("Searched strategy.md, LEARNED.md") and "closest line" in q.searched
    assert d.form.unclear[0].searched == q.searched
    d = draft(wire(unclear=[ask], terms=[{"term": "session high / low", "answered_by": "LEARNED #127"}]))
    assert d.questions[0].answered_by == "LEARNED #127" and "!! the talker's own pick" in d.text()
    assert "none says what he means" in d.questions[0].searched       # FOUND by the search, but no quote answers
    ask2 = dict(ask, term="flying pigs")                               # never searched: code searches it now
    d = draft(wire(unclear=[ask2]))
    assert "searched after the talker answered" in d.questions[0].searched


# ================================================================================================ 5. the API wrapper
def test_translate_calls_once_then_answers_from_the_cache_per_model(tmp_path):
    api = FakeAPI(_msg(wire()))
    d = T.translate("sell after the london high is taken", search=fake_search(), api=api)
    assert d.valid and len(api.calls) == 1
    kw = api.calls[0]
    assert kw["tools"][0]["name"] == "fill_idea_form" and kw["tools"][0]["strict"] is True
    assert kw["thinking"] == {"type": "adaptive"} and kw["tool_choice"] == {"type": "auto"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"} and kw["fallbacks"] == "default"
    assert kw["model"] == config.TALKER_MODEL
    assert d.answer.cost_usd == pytest.approx((1000 * 5 + 15000 * 5 * 0.1 + 2000 * 25) / 1e6)
    d2 = T.translate("sell after the london high is taken", search=fake_search(), api=api)
    assert len(api.calls) == 1 and d2.answer.from_cache and d2.cost_usd == 0.0
    d3 = T.translate("sell after the london high is taken", search=fake_search(), api=api, model="claude-sonnet-5")
    assert len(api.calls) == 2 and not d3.answer.from_cache
    assert d3.answer.cost_usd == pytest.approx((1000 * 2 + 15000 * 2 * 0.1 + 2000 * 10) / 1e6)


def test_words_instead_of_the_tool_are_asked_once_more_refusals_and_cutoffs_are_errors():
    api = FakeAPI(_msg(text="Here is the form ..."), _msg(wire()))
    d = T.translate("sell at the fvg", search=fake_search(), api=api)
    assert d.valid and len(api.calls) == 2
    assert api.calls[1]["messages"][-1] == {"role": "user", "content": "Answer only by calling fill_idea_form."}
    for stop, why in (("refusal", "refused"), ("max_tokens", "cut off")):
        d = T.translate(f"buy at the fvg {stop}", search=fake_search(), api=FakeAPI(_msg(stop=stop)))
        assert not d.valid and why in d.problems[0]


def test_api_errors_become_one_plain_line():
    import anthropic
    import httpx2
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    for err, words in ((anthropic.RateLimitError("slow", response=httpx2.Response(429, request=req), body=None),
                        "rate limited"),
                       (anthropic.AuthenticationError("no", response=httpx2.Response(401, request=req), body=None),
                        "key was refused"),
                       (anthropic.BadRequestError("bad schema", response=httpx2.Response(400, request=req),
                                                  body=None), "refused (400)"),
                       (anthropic.APIConnectionError(request=req), "no connection")):
        def boom(kw, err=err):
            raise err
        d = T.translate(f"buy at the fvg {words}", search=fake_search(), api=FakeAPI(boom))
        assert not d.valid and words in d.problems[0], (words, d.problems)


def test_no_key_says_so_in_plain_words(monkeypatch):
    monkeypatch.setattr(config, "env", lambda name, default=None: default)
    d = T.translate("buy at the fvg", search=fake_search())
    assert not d.valid and "ANTHROPIC_API_KEY" in d.problems[0]


def test_a_refused_form_gets_one_fixup_call_with_codes_findings():
    bad = wire(settings=[{"name": "close_before_weekend", "value_json": "true"}])
    api = FakeAPI(_msg(bad), _msg(wire()))
    d = T.translate("sell after the london high is taken", search=fake_search(), api=api)
    assert len(api.calls) == 2 and d.valid and d.kept == "the fix-up's form"
    assert any("SettingError" in f and "close_before_weekend" in f for f in d.fixup)
    second = api.calls[1]["messages"][0]["content"]
    assert second.startswith(api.calls[0]["messages"][0]["content"])      # the same message, the findings added
    assert "What code found:" in second and "close_before_weekend" in second and "YOUR FIRST FORM" in second
    assert api.calls[1]["system"] == api.calls[0]["system"]                # the fixed text is unchanged: cached
    assert d.cost_usd == pytest.approx(sum(a.cost_usd for a in d.answers())) and len(d.answers()) == 2
    assert "FIX-UP: code sent 1 findings back once" in d.text()


def test_a_worse_fixup_keeps_the_first_form_and_a_good_form_gets_none():
    ask_missing = fib_wire()                                               # 0.68 next to "golden ratio", never asked
    api = FakeAPI(_msg(ask_missing), _msg(wire(settings=[{"name": "nope", "value_json": "1"}])))
    d = T.translate(FIB_WORDS, search=fake_search(), api=api)
    assert len(api.calls) == 2 and d.valid and d.kept.startswith("the first form")
    assert any("golden_ratio" in f for f in d.fixup) and len(d.answers()) == 2
    api = FakeAPI(_msg(wire()))
    assert T.translate("sell after the london high is taken", search=fake_search(), api=api).fixup == []
    assert len(api.calls) == 1
    api = FakeAPI(_msg(wire(settings=[{"name": "nope", "value_json": "1"}])))
    d = T.translate("sell after the london high", search=fake_search(), api=api, fixups=0)
    assert len(api.calls) == 1 and not d.valid


def test_the_exam_makes_the_fixup_round_and_counts_it():
    rows = [r for r in evals.load() if r["id"] == "G03"]

    def answer(kw):
        fixing = "What code found:" in kw["messages"][0]["content"]
        return _msg(wire() if fixing else wire(settings=[{"name": "nope", "value_json": "1"}]))

    scores = evals.run(rows, model="claude-opus-5", effort="high", search=fake_search(), api=FakeAPI(answer),
                       say=lambda *_: None)
    assert not scores[0].invalid and scores[0].draft.kept == "the fix-up's form"
    t = evals.totals(scores)
    assert t["answers"] == 2 and t["fixups"] == 1 and t["fixups_that_made_it_valid"] == 1


def test_the_batch_matches_answers_by_custom_id_and_caches_them():
    reqs = {cid: T.request(glossary.apply(w), (), None, None) for cid, w in (("A", "buy at the fvg"),
                                                                              ("B", "sell at the fvg"))}

    def answer(params):
        text = params["messages"][0]["content"]
        return _msg(wire(title="buy" if "buy at" in text else "sell"))

    api = FakeAPI(answer)
    out = llm.call_batch(reqs, api=api, say=lambda *_: None)
    assert out["A"].input["title"] == "buy" and out["B"].input["title"] == "sell" and out["A"].batch
    assert out["A"].cost_usd == pytest.approx(llm.cost("claude-opus-5", out["A"].tokens) / 2)
    again = llm.call_batch(reqs, api=FakeAPI(answer), say=lambda *_: None)
    assert again["A"].from_cache and again["B"].from_cache


# ================================================================================================ 5b. his Claude Code login
class FakeClaudeCode:
    """Stands in for `claude -p`: records what it was given (and the system file while it exists)."""

    def __init__(self, out: dict | str, fail=None):
        self.out, self.fail, self.calls = out, fail, []

    def __call__(self, args, **kw):
        sysfile = Path(args[args.index("--system-prompt-file") + 1])
        self.calls.append({"args": args, **kw, "system": sysfile.read_text(encoding="utf-8")})
        if self.fail:
            raise self.fail
        out = self.out if isinstance(self.out, str) else json.dumps(self.out)
        return SimpleNamespace(stdout=out, stderr="", returncode=0)


def _cc_out(form, **extra):
    return {"type": "result", "subtype": "success", "is_error": False, "result": "done", "structured_output": form,
            "usage": {"input_tokens": 9, "output_tokens": 1800, "cache_read_input_tokens": 17000,
                      "cache_creation_input_tokens": 1200},
            "modelUsage": {"claude-haiku-4-5-20251001": {}, "claude-opus-5": {}}, "total_cost_usd": 0.12,
            "duration_ms": 65000, **extra}


def test_claude_code_uses_his_login_never_a_key_and_costs_nothing_extra(monkeypatch):
    monkeypatch.setattr(config, "TALKER_BACKEND", "claude-code")
    monkeypatch.setattr(llm, "claude_exe", lambda: "claude.exe")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-be-passed-on")
    run = FakeClaudeCode(_cc_out(wire()))
    d = T.translate("sell after the london high is taken", search=fake_search(), runner=run)
    assert d.valid and len(run.calls) == 1
    c = run.calls[0]
    a = c["args"]
    assert a[:4] == ["claude.exe", "-p", "--tools", ""] and "--no-session-persistence" in a
    assert a[a.index("--model") + 1] == config.TALKER_MODEL and a[a.index("--effort") + 1] == config.TALKER_EFFORT
    assert json.loads(a[a.index("--json-schema") + 1]) == T.tool()["input_schema"]
    assert c["system"] == T.reference()                              # the fixed text, from a file
    assert c["input"].startswith("His words, as typed:")             # his sentence + its search, on standard input
    assert "ANTHROPIC_API_KEY" not in c["env"] and "ANTHROPIC_AUTH_TOKEN" not in c["env"]
    assert Path(c["cwd"]).name == "claude-code-cwd" and not list(Path(c["cwd"]).glob("*.system.txt"))
    ans = d.answer
    assert ans.model == "claude-opus-5" and ans.cost_usd == 0.0 and ans.extra["api_price_usd"] == 0.12
    assert ans.tokens == llm.Tokens(9, 1800, 1200, 17000) and ans.extra["seconds"] == 65.0
    d2 = T.translate("sell after the london high is taken", search=fake_search(), runner=run)
    assert d2.answer.from_cache and len(run.calls) == 1              # cached: never asked twice
    d3 = T.translate("sell after the london high is taken", search=fake_search(), runner=run, backend="api",
                     api=FakeAPI(_msg(wire())))
    assert not d3.answer.from_cache                                   # the two ways never share an answer


def test_claude_code_failures_say_why(monkeypatch):
    import subprocess
    monkeypatch.setattr(config, "TALKER_BACKEND", "claude-code")
    monkeypatch.setattr(llm, "claude_exe", lambda: "claude.exe")
    cases = ((FakeClaudeCode(_cc_out(None, is_error=True, result="Claude usage limit reached")), "usage limit"),
             (FakeClaudeCode(_cc_out(None)), "no form"),
             (FakeClaudeCode("Error: not logged in"), "did not answer in JSON"),
             (FakeClaudeCode({}, fail=subprocess.TimeoutExpired("claude", 900)), "no answer in 900 seconds"))
    for k, (run, words) in enumerate(cases):
        d = T.translate(f"buy at the fvg {k}", search=fake_search(), runner=run)
        assert not d.valid and words in d.problems[0], (words, d.problems)


def test_the_real_claude_exe_is_found_behind_the_npm_wrapper(tmp_path, monkeypatch):
    exe = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(llm.shutil, "which", lambda name: str(tmp_path / "claude.CMD"))
    assert llm.claude_exe() == str(exe)                               # never through cmd.exe (it reads < > | &)
    monkeypatch.setattr(config, "env", lambda name, default=None: "D:/x/claude.exe" if name == "CLAUDE_EXE" else default)
    assert llm.claude_exe() == "D:/x/claude.exe"
    monkeypatch.setattr(config, "env", lambda name, default=None: default)
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    with pytest.raises(llm.LLMError, match="not found"):
        llm.claude_exe()


# ================================================================================================ 6. sanity
FIB_WORDS = ("in one hour time frame it should higher high then it will come back to exact golden ratio that is "
             "uh, 0.68 it will definitely give 1 is to 1 the next that is 0.5 is the target")


def fib_wire(entry="fib.level_0.68", stop="fib.level_0.786", target="fib.level_0.5", unclear=()):
    return wire(title="fib", timeframe="1h",
                steps=[{"id": "fib", "what": "a higher high and its levels", "block": "fib_move", "tf": "1h",
                        "max_wait": "first step", "params": [], "sources": ["his words"]}],
                entry={"how": "limit", "at": entry, "max_wait": "no limit", "sources": ["his words"]},
                exit={"kind": "stop", "at": stop, "beyond_usd": 0.0, "management": [], "sources": ["his words"]},
                targets=[{"kind": "price", "ratio": None, "at": target, "kinds": None, "pick": "nearest",
                          "sources": ["his words"]}], unclear=list(unclear))


def _checks(d, name):
    return [c for c in d.sanity if c.name == name]


def test_sanity_068_next_to_golden_ratio_must_be_asked():
    d = draft(fib_wire(), FIB_WORDS)
    assert d.valid
    g = _checks(d, "golden_ratio")
    assert len(g) == 1 and not g[0].ok and g[0].severity == "ask" and "0.618" in g[0].message
    ask = {"id": "F1", "kind": "ask", "term": "golden ratio / fibonacci", "part": "0.68 or the golden ratio 0.618?",
           "words": "golden ratio that is uh, 0.68",
           "readings": [{"id": "A", "means": "0.68", "changes": []},
                        {"id": "B", "means": "0.618", "changes": [{"path": "entry.at",
                                                                   "value_json": "\"fib.level_0.618\""}]}]}
    d = draft(fib_wire(unclear=[ask]), FIB_WORDS)
    assert d.valid and _checks(d, "golden_ratio")[0].ok


def test_sanity_1_to_1_needs_a_stop_he_never_gave_and_levels_in_an_impossible_order():
    d = draft(fib_wire(entry="fib.level_0.618"), FIB_WORDS)
    one = [c for c in _checks(d, "impossible_levels") if "1:1" in c.message]
    assert one and one[0].severity == "ask" and "0.736" in one[0].message
    d = draft(fib_wire(entry="fib.level_0.618", target="fib.level_0.786", stop="fib.level_0.5"), "buy at the fib")
    bad = [c for c in _checks(d, "impossible_levels") if c.severity == "problem"]
    assert len(bad) == 2                                   # the target behind the entry, the stop in front of it
    w = wire()
    w["exit"]["at"] = "gap.near"                           # the hedge at the entry price
    d = draft(w)
    assert any(c.severity == "problem" and "no distance" in c.message for c in _checks(d, "impossible_levels"))
    w["exit"]["beyond_usd"] = 0.2                          # a few cents past it (his Type 2 hedge): fine
    assert not any(c.severity == "problem" for c in _checks(draft(w), "impossible_levels"))


def test_sanity_charts_he_names():
    said = dict(sanity.charts_said("on the 2 hour chart, 5 min fvg, one-hour trendline, 1 is to 1, daily up, "
                                   "30 minutes, monthly"))
    assert said["2 hour"] is None and said["5 min"] == "5m" and said["one-hour"] == "1h"
    assert said["daily"] == "1d" and said["30 minutes"] is None and said["monthly"] is None
    assert not any("is" in k for k in said)                # "1 is to 1" is not a chart
    d = draft(wire(), "sell on the 2 hour chart after the london high is taken")
    assert any(c.severity == "ask" and "2 hour" in c.message for c in _checks(d, "unknown_timeframe"))
    d = draft(wire(), "in one hour time frame the london high is taken")
    assert any(c.severity == "note" and "1h" in c.message for c in _checks(d, "unknown_timeframe"))


# ================================================================================================ 7. the exam
def test_golden_jsonl_is_golden_ideas_md_word_for_word():
    rows = evals.load()
    assert len(rows) == 25 and sum(r["real"] for r in rows) == 16
    md = (ROOT / "plan/work/golden-ideas.md").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", " ".join(ln[1:].strip() for ln in md.splitlines() if ln.startswith(">")))
    for r in rows:
        assert r["words"] in flat, r["id"]
    kinds = [u["kind"] for r in rows for u in r["expect"].get("unclear", [])]
    assert kinds.count("ask") == 9 and kinds.count("version") == 8      # golden-ideas.md's own count


def test_golden_jsonl_names_only_real_or_planned_pieces_and_real_old_tries():
    known = set(C.BLOCKS) | set(C.DO_NOT_ENTER) | set(C.FILTERS) | set(C.PLANNED) | set(C.BUILT_KINDS) | {"adr"}
    for r in evals.load():
        e = r["expect"]
        for need in e.get("needs", []):
            assert set(need.split("|")) <= known, (r["id"], need)
        for key in e.get("already_seen", []):
            assert key in T.prior_keys(), (r["id"], key)
        for name in ("entry", "exit", "target"):
            for p in ([e[name]] if isinstance(e.get(name), str) else e.get(name, [])):
                block = p.split(" ", 1)[1].split(".")[0].rstrip("*") if " " in p else ""
                assert not block or any(k.startswith(block) for k in known), (r["id"], p)
        if "kind" in e:
            assert e["kind"] in T.RULE_KINDS


ROW = {"id": "X1", "real": True, "words": "sell after the london high is taken and the fvg",
       "expect": {"kind": "idea", "timeframe": "5m", "directions": ["buy", "sell"], "entry": "limit gap_of_the_move.near",
                  "exit": "hedge break_of_structure.extreme", "target": "next_liquidity*", "blocked": False,
                  "needs": ["sweep", "break_of_structure", "gap_of_the_move", "bias|sweep"],
                  "voice": [{"heard": "1et fvg", "any": ["first"]}],
                  "unclear": [{"kind": "ask", "any": ["steep"]}, {"kind": "version", "any": ["near", "pips"]}]}}


def _u(id, kind, part, term="x"):
    return {"id": id, "kind": kind, "term": term, "part": part, "words": None,
            "readings": [{"id": "A", "means": "a", "changes": []}, {"id": "B", "means": "b", "changes": []}]}


def test_scoring_fields_unclear_parts_and_needless_questions():
    w = wire(voice_typing=[{"heard": "1et fvg", "means": "the first fvg"}],
             unclear=[_u("Q1", "ask", "how steep the line must be"), _u("Q2", "version", "how many pips below"),
                      _u("Q3", "ask", "what is a gap"), _u("Q4", "version", "how long to wait")])
    s = evals.score_row(ROW, draft(w))
    assert not s.invalid and s.right == len(s.fields) == 12
    assert (s.unclear_found, s.unclear_expected) == (2, 2)
    assert s.false_questions == ["Q3: what is a gap"] and s.extra_versions == ["Q4: how long to wait"]
    # the same word as a VERSION instead of an ASK: not found, and said so
    s = evals.score_row(ROW, draft(wire(unclear=[_u("Q1", "version", "how steep")])))
    assert s.unclear_found == 0 and s.wrong_kind and s.right == 11            # the voice item is missed too
    # an invalid form scores nothing
    s = evals.score_row(ROW, draft(wire(settings=[{"name": "nope", "value_json": "1"}])))
    assert s.invalid and s.right == 0 and s.unclear_found == 0


def test_matching_is_one_to_one():
    exp = [{"kind": "ask", "any": ["strong"]}, {"kind": "ask", "any": ["very strong"]}]
    assert len(evals._match(exp, [("ask", "very strong trend")])) == 1
    assert len(evals._match(exp, [("ask", "very strong trend"), ("ask", "strong move")])) == 2


def test_the_pass_mark():
    t = {"fields_rate": 0.9, "unclear_rate": 0.9, "invalid": 0, "false_questions": 2, "false_limit": 2}
    assert evals.passed(t)
    for k, v in (("fields_rate", 0.89), ("unclear_rate", 0.88), ("invalid", 1), ("false_questions", 3)):
        assert not evals.passed({**t, k: v}), k
    assert evals.totals([evals.RowScore(f"G{k:02d}", True) for k in range(25)])["false_limit"] == 2


def test_the_exam_runs_end_to_end_in_a_batch_and_is_saved(tmp_path):
    rows = [r for r in evals.load() if r["id"] in ("G03", "G17")]
    api = FakeAPI(_msg(wire()))
    scores = evals.run(rows, model="claude-opus-5", effort="medium", batch=True, search=fake_search(), api=api,
                       say=lambda *_: None)
    assert [s.id for s in scores] == ["G03", "G17"] and all(not s.invalid for s in scores)
    g03 = {f: ok for f, ok, _, _ in scores[0].fields}
    assert g03["entry"] and g03["exit"] and g03["target"] and g03["timeframe"] and not g03["directions"] is None
    t = evals.totals(scores)
    assert t["answers"] == 2 and t["paid_now_usd"] > 0
    meta = {"when": "2026-09-28", "model": "claude-opus-5", "effort": "medium", "batch": True, "search": "fake",
            "full": False}
    text = evals.report(scores, t, meta)
    assert "API COST" in text and "no pass mark" in text
    base = evals.save(scores, t, meta, text, out_dir=tmp_path)
    doc = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert doc["run"] == 1 and doc["rows"][0]["answer"]["title"] == T.EXAMPLE["title"]
    assert evals.save(scores, t, meta, text, out_dir=tmp_path).name.endswith("run2")


# ================================================================================================ 8. the search exam's pick step
def test_the_search_exam_with_claudes_pick(tmp_path):
    from machine.search import exam
    from machine.search import index as ix
    from machine.search.rank import answered
    index = ix.build(embeddings="off")
    q_found, q_absent = "where does he enter the fair value gap", "what colour is the moon on tuesday"
    hits = answered(q_found, index=index).hits
    rows = [{"id": "F1", "expect": "found", "q": q_found, "cite": [hits[1].id]},
            {"id": "N1", "expect": "absent", "q": q_absent}]
    path = tmp_path / "exam.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def answer(params):
        ids = params["tools"][0]["input_schema"]["properties"]["answer"]["enum"]
        text = params["messages"][0]["content"]
        return _msg({"answer": ids[1] if f"Question: {q_found}" in text else "none", "why": "x"}, tool="pick_citation")

    picked = exam.run_pick(path, index, api=FakeAPI(answer))
    assert [p.pick for p in picked] == [hits[1].id, None]
    assert picked[0].first_right and picked[1].ok
    assert "SEEN" in exam.report_pick(picked, index.how)
