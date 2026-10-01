"""The document search (L8b): how his files are cut and cited, the words search, the trust order, FOUND / NOT FOUND
and the #129 sentence, the embedding cache and the Voyage call (both with fakes: no network, no model download), and
the exam on the real files."""
import functools
import hashlib
import json
import math
import urllib.request
from dataclasses import replace

import numpy as np
import pytest

from machine import config
from machine.search import answered, index as ix, search
from machine.search import exam
from machine.search import rank
from machine.search.rank import Hit

LONG_LINE = " ".join(f"word{i} and gold moves." for i in range(400))

FILES = {
    "strategy/LEARNED.md": """# LEARNED.md
> the preamble is not a lesson

---

## #1  ·  2026-08-30 14:22  ·  LESSON · on card **Q56-order-block** (SELL grade question, swept CONCEPT A1 - which red candle)

> Equal highs and equal lows are a pair, price travels between them.

*(His answer is a DRAWING, saved as `ml/agent/logs/questions/Q56-his-answer.png`.)*

## #21  ·  2026-09-06  ·  TEJAS — THE HEDGE GOES AT THE GRAB'S EXTREME

> the hedge sits at the swing high where it took the liquidity

## #21  ·  2026-09-07 23:01  ·  LESSON

> it is not a proper break of structure

## #129 — 2026-09-21. A LEVEL SWEPT ONCE IS DEAD

> once it poke on any high or low then that's closed, why are you asking again

## #130 — 2026-09-22. LONG

> """ + LONG_LINE + "\n",
    "strategy/strategy-understanding.md": """# strategy-understanding.md
intro line

## 0. The one-line version
zero text

### 7b. ✅ readable
seven b text

## §43. first forty three
a

### What must NOT be concluded
still inside 43

## §43. second forty three
b

### §66bt. LEARNED #129 — swept once is dead
dead levels
""",
    "strategy/strategy.md": """# Gold (XAUUSD) Trading Strategy
> header line

## 1. Liquidity ✅
- **Every sweep counts** — once a high or low is swept, it is finished (#129).

| a | b |
|---|---|

## Glossary
- **Hedge** — the opposite position opened at the hedge point.
""",
    "strategy/strategy-code.md": """# strategy-code.md
## 7. Sweep detection
```
# A sweep = price PENETRATES a level by a wick.
```
after the code
""",
    "strategy/DECISIONS.md": "# DECISIONS\n## C2 · where the hedge goes\nat the swing extreme\n",
    "strategy/AMBIGUITY-REGISTER.md": '# REG\n## TIER 1 — settle these\n### A2 · "sweep" — how far past the level?\nany poke\n',
    "strategy/EVERYTHING-YOU-TAUGHT.md": "# Everything\n## Liquidity\n- **#21** equal highs and equal lows are a pair\n",
    "strategy/telegram-archive.jsonl": (
        json.dumps({"n": 1, "ts": "2026-08-30 14:22", "kind": "lesson", "text": "equal highs pair words"}) + "\n"
        + json.dumps({"n": 21, "ts": "2026-09-07 23:01", "kind": "correction", "text": "not a proper break"}) + "\n"),
    "strategy/learned-inbox/README.md": "# learned-inbox\nnot a lesson\n",
    "strategy/learned-inbox/2026-09-28-1151-session.md": "# Lesson (session) — waiting\n\n> 33.yes 34.no lows too near\n",
    "plan/work/ideas/idea-001/answers.md": ("# Answers - idea-001: x\n\nTejas's answers, word for word, newest at the bottom.\n\n"
                                           "## 2026-09-29 Q1\n> yes that is my idea\n"),
    "plan/work/ideas/idea-002/answers.md": "# Answers - idea-002: y\n\nTejas's answers, word for word.\n",
}


@pytest.fixture
def root(tmp_path):
    for rel, text in FILES.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    pic = tmp_path / "strategy" / "pictures" / "Q56-his-answer.png"
    pic.parent.mkdir(parents=True)
    pic.write_bytes(b"")
    return tmp_path


def _no_rules(monkeypatch):
    """The fake root has no rules.py / settings.py of its own; leave them out of the fake index."""
    monkeypatch.setattr(ix, "SOURCES", tuple(s for s in ix.SOURCES if s is not ix.rules_and_settings))


# ------------------------------------------------------------------------------------------------ cutting and ids
def test_every_source_is_cut_and_cited(root, monkeypatch):
    _no_rules(monkeypatch)
    ids = {p.id for p in ix.all_pieces(root)}
    for want in ("LEARNED #1", "LEARNED #21 (6 Sep)", "LEARNED #21 (7 Sep)", "LEARNED #129", "LEARNED #130",
                 "understanding §0", "understanding §7b", "understanding §66bt", "strategy-code §7", "DECISIONS C2",
                 "AMBIGUITY-REGISTER A2", "inbox 2026-09-28-1151-session", "idea-001 answers — 2026-09-29 Q1",
                 "picture Q56-order-block", "picture Q56-his-answer"):
        assert want in ids, want
    assert "LEARNED #21" not in ids                        # a number used twice always carries its day
    assert not any(i.startswith("inbox README") for i in ids)
    assert not any(i.startswith("idea-002") for i in ids)  # no answers yet: nothing to find
    assert any(i.startswith("strategy.md §1 line ") for i in ids)
    assert any(i.startswith("strategy.md glossary line ") for i in ids)
    assert any(i.startswith("EVERYTHING-YOU-TAUGHT line ") for i in ids)


def test_learned_both_heading_styles_dates_and_titles(root):
    pieces = {p.id: p for p in ix.learned(root)}
    assert pieces["LEARNED #21 (6 Sep)"].date == "2026-09-06"
    assert pieces["LEARNED #21 (7 Sep)"].date == "2026-09-07"
    assert pieces["LEARNED #129"].date == "2026-09-21"      # the "## #N —" style
    assert "A LEVEL SWEPT ONCE IS DEAD" in pieces["LEARNED #129"].title
    assert all(p.tier == 1 for p in pieces.values())
    assert not any("preamble" in p.text for p in pieces.values())


def test_a_long_voice_note_is_cut_into_pieces_of_one_idea(root):
    long = [p for p in ix.learned(root) if p.id == "LEARNED #130"]
    assert len(long) > 3
    assert all(len(p.text.split()) <= ix.MAX_WORDS + 10 for p in long)
    lines = FILES["strategy/LEARNED.md"].splitlines()
    assert long[0].line == lines.index("## #130 — 2026-09-22. LONG") + 1 and "LONG" in long[0].text
    assert all(p.line == lines.index("> " + LONG_LINE) + 1 for p in long[1:])   # each piece names its line


def test_understanding_sections_subsections_and_twice_used_numbers(root):
    pieces = ix.understanding(root)
    ids = [p.id for p in pieces]
    twice = sorted({i for i in ids if i.startswith("understanding §43")})
    assert len(twice) == 2 and all("(line " in i for i in twice)
    inside = next(p for p in pieces if "still inside 43" in p.text)
    assert inside.id.startswith("understanding §43")        # a heading without a number stays in its section
    assert all(p.tier == 2 for p in pieces)


def test_a_heading_inside_a_code_block_is_not_a_section(root):
    pieces = ix.strategy_code(root)
    assert {p.id for p in pieces} == {"strategy-code top", "strategy-code §7"}
    assert "PENETRATES" in next(p for p in pieces if p.id == "strategy-code §7").text


def test_strategy_md_is_cited_by_line_and_section_and_skips_table_rules(root):
    pieces = ix.strategy_md(root)
    sweep = next(p for p in pieces if "Every sweep counts" in p.text)
    assert sweep.id == "strategy.md §1 line 5" and sweep.line == 5 and sweep.tier == 4
    assert not any(set(p.text) <= set("|-: ") for p in pieces)


def test_the_telegram_archive_is_filed_under_its_lesson_number(root):
    pieces = ix.telegram(root)
    assert {p.id for p in pieces} == {"LEARNED #1", "LEARNED #21 (7 Sep)"}
    assert all(p.path == "strategy/telegram-archive.jsonl" for p in pieces)


def test_a_picture_is_found_through_the_lesson_that_names_it(root):
    pics = {p.id: p for p in ix.pictures(root)}
    card, drawn = pics["picture Q56-order-block"], pics["picture Q56-his-answer"]
    assert card.see == ("LEARNED #1",) and drawn.see == ("LEARNED #1",)
    assert drawn.path == "strategy/pictures/Q56-his-answer.png"
    assert card.path == "strategy/LEARNED.md"                # no file kept: cited through the lesson
    assert "which red candle" in card.text                   # what the picture asked


def test_the_inbox_has_the_newest_date_and_the_top_tier(root):
    (p,) = ix.inbox(root)
    assert p.date == "2026-09-28" and p.tier == 1 and "lows too near" in p.text


# ------------------------------------------------------------------------------------------------ words
def test_terms_make_the_same_thing_one_word():
    assert ix.terms("FVG") == ix.terms("fair value gap")
    assert ix.terms("hedged") == ix.terms("hedging") == ix.terms("hedge")
    assert ix.terms("swept") == ix.terms("sweep")
    assert ix.terms("time frame") == ix.terms("timeframe")
    assert ix.terms("the 5-minute chart") == ix.terms("5 min chart")
    assert ix.terms("uh okay the and") == []


def test_bm25_ranks_the_piece_with_the_rare_word_first():
    docs = [ix.terms(t) for t in ("gold gold gold", "gold equal lows pair", "gold hedge")]
    s = ix.BM25(docs).scores(ix.terms("equal lows"))
    assert int(np.argmax(s)) == 1 and s[0] == 0


# ------------------------------------------------------------------------------------------------ search + trust order
class FakeEmbedder:
    """Bag of words hashed into 64 numbers: close meaning = shared words. Deterministic, offline."""

    recipe = "fake|bag-of-words-64"

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = np.zeros(64, np.float32)
        for t in ix.terms(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def documents(self, texts):
        self.calls += len(texts)
        return np.array([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_index(root, tmp_path, monkeypatch):
    _no_rules(monkeypatch)
    monkeypatch.setitem(rank.LINES, "fake|", {"meaning": 0.30, "covered": 0.45, "band": 0.30})
    return ix.build(root, cache_dir=tmp_path / "cache", embedder=FakeEmbedder())


def test_search_finds_and_quotes_with_the_line(fake_index):
    hits = search("is a level dead once it is swept?", index=fake_index)
    assert hits[0].id in ("LEARNED #129", "strategy.md §1 line 5", "understanding §66bt")
    h = next(h for h in hits if h.id == "strategy.md §1 line 5")
    assert "swept" in h.quote and h.line == 5 and h.path == "strategy/strategy.md"


def test_found_cites_his_words_before_strategy_md(fake_index):
    a = answered("is a high or low finished once it is swept?", index=fake_index)
    assert a.found
    tiers = [h.tier for h in a.cites]
    assert tiers == sorted(tiers)                             # the order of trust
    assert a.cites[0].id == "LEARNED #129"
    assert str(a).startswith("FOUND")


def test_not_found_says_the_129_sentence_with_the_closest_line(fake_index):
    a = answered("Does he use the Ichimoku cloud?", index=fake_index)
    assert not a.found and a.cites == ()
    assert a.sentence.startswith('Searched strategy.md, LEARNED.md, strategy-understanding.md, rules, pictures for '
                                 '"Does he use the Ichimoku cloud" — closest line: ')
    assert a.closest is not None and a.closest.id in a.sentence


def _hit(id, tier, rank_, meaning, date="", source="LEARNED", words=10.0):
    return Hit(id, source, tier, date, "f", 1, "q", words, meaning, 0.9, rank_, True)


def test_trust_order_inside_the_band_his_words_first_pictures_after():
    hits = [_hit("strategy.md §1 line 76", 4, 1, 0.800, source="strategy.md"),
            _hit("picture Q95", 1, 2, 0.800, "2026-09-21", source="picture"),
            _hit("LEARNED #95", 1, 3, 0.795, "2026-09-18"),
            _hit("LEARNED #129", 1, 4, 0.792, "2026-09-21"),
            _hit("understanding §66bt", 2, 5, 0.700, source="understanding")]
    out = rank._trust_order(hits, {"meaning": 0.68, "covered": 0.45, "band": 0.01})
    # his words first; inside his words the more relevant first (not the newer: exam run 1, plan/work/L8b-notes.md)
    assert [h.id for h in out] == ["LEARNED #95", "LEARNED #129", "picture Q95", "strategy.md §1 line 76",
                                   "understanding §66bt"]      # §66bt is 0.10 below: outside the band, keeps its place


def test_trust_order_never_lifts_a_hit_far_below_the_most_relevant():
    hits = [_hit("strategy.md §10 line 127", 4, 1, 0.80, source="strategy.md", words=12.0),
            _hit("LEARNED #124", 1, 2, 0.70, "2026-09-20", words=8.0)]
    out = rank._trust_order(hits, {"band": 0.01})
    assert out[0].id == "strategy.md §10 line 127"
    words_only = rank._trust_order([replace(h, meaning=math.nan) for h in hits], {"band_words": 1.0})
    assert words_only[0].id == "strategy.md §10 line 127"


# ------------------------------------------------------------------------------------------------ embeddings
def test_the_embedding_cache_reembeds_only_changed_pieces_and_keeps_models_apart(root, tmp_path, monkeypatch):
    _no_rules(monkeypatch)
    cache = tmp_path / "cache"
    e1 = FakeEmbedder()
    ix.build(root, cache_dir=cache, embedder=e1)
    first = e1.calls
    assert first > 0
    e2 = FakeEmbedder()
    ix.build(root, cache_dir=cache, embedder=e2)
    assert e2.calls == 0                                      # all read back
    f = root / "strategy" / "DECISIONS.md"
    f.write_text(f.read_text(encoding="utf-8") + "and a new line he taught\n", encoding="utf-8")
    e3 = FakeEmbedder()
    ix.build(root, cache_dir=cache, embedder=e3)
    assert e3.calls == 1                                      # only the changed piece
    other = FakeEmbedder()
    other.recipe = "fake|another-model"
    ix.build(root, cache_dir=cache, embedder=other)
    assert other.calls >= first                               # a different model never reads these vectors
    assert len(list(cache.glob("emb-*.npz"))) == 2


def test_voyage_is_called_with_the_key_and_the_right_input_type(monkeypatch):
    sent = []

    class Reply:
        def __init__(self, n):
            self.n = n

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            data = [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(self.n))]
            return json.dumps({"data": data}).encode()

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        sent.append((req.full_url, req.get_header("Authorization"), body["input_type"], len(body["input"]),
                     body["model"]))
        return Reply(len(body["input"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    v = ix.VoyageEmbedder("KEY123")
    docs = v.documents([f"t{i}" for i in range(130)])
    q = v.query("where is the hedge")
    assert docs.shape == (130, 2) and np.allclose(docs[0], [0.6, 0.8]) and np.allclose(q, [0.6, 0.8])
    assert [s[2:4] for s in sent] == [("document", 128), ("document", 2), ("query", 1)]
    assert all(s[0] == "https://api.voyageai.com/v1/embeddings" and s[1] == "Bearer KEY123" for s in sent)
    assert sent[0][4] == config.SEARCH_VOYAGE_MODEL


def test_choosing_the_embedder(monkeypatch):
    env = {}
    monkeypatch.setattr(config, "env", lambda name, default=None: env.get(name, default))
    env["SEARCH_EMBEDDINGS"] = "off"
    e, how = ix.choose_embedder()
    assert e is None and "words only" in how
    env.clear()
    env["VOYAGE_API_KEY"] = "k"
    e, how = ix.choose_embedder()
    assert isinstance(e, ix.VoyageEmbedder) and "Voyage" in how
    e, how = ix.choose_embedder("off")
    assert e is None


# ------------------------------------------------------------------------------------------------ the real files
@pytest.mark.private
def test_the_real_documents_are_all_indexed():
    pieces = ix.all_pieces()
    ids = {p.id for p in pieces}
    for want in ("LEARNED #21 (6 Sep)", "LEARNED #24 (7 Sep)", "LEARNED #136", "understanding §66ca",
                 "rules.py which_gap_is_the_entry_gap", "settings.py which_leg_may_close_while_covered",
                 "DECISIONS C2", "AMBIGUITY-REGISTER A1", "picture Q98-D8-break-distance"):
        assert want in ids, want
    assert any(p.path == "strategy/telegram-archive.jsonl" and p.id == "LEARNED #54" for p in pieces)
    # a lesson numbered by learned-merge (L9a) keeps its inbox id as a second name
    assert any(p.id == "LEARNED #149" and "inbox 2026-09-28-1151-session" in p.see for p in pieces)
    pic = next(p for p in pieces if p.id == "picture Q98-D8-break-distance")
    assert "LEARNED #98" in pic.see
    assert not any(p.path.startswith("old/") and p.source != "picture" for p in pieces)   # old/ is never searched


@pytest.mark.private
@pytest.mark.parametrize("path", [exam.EXAM, exam.DEV])
def test_every_exam_citation_still_exists(path):
    """If a file moves a line, its label must move too - this names the ids that vanished. A numbered lesson answers
    to its old inbox id too (L9a)."""
    pieces = ix.all_pieces()
    ids = {p.id for p in pieces} | {s for p in pieces for s in p.see}
    rows = exam.load(path)
    missing = [(r["id"], c) for r in rows for c in r.get("cite", []) if c not in ids]
    assert missing == []


def test_the_exam_has_30_found_and_10_not_there():
    rows = exam.load(exam.EXAM)
    assert sum(r["expect"] == "found" for r in rows) == 30 and sum(r["expect"] == "not found" for r in rows) == 10
    assert all(r.get("from") for r in rows)
    assert len({r["q"] for r in rows} & {r["q"] for r in exam.load(exam.DEV)}) == 0   # tuning never saw the exam


@functools.lru_cache(maxsize=1)
def _exam_run():
    """The exam, once per test session, with the local model. The first run on a PC embeds every piece (about
    4 minutes on Tejas's PC), then they are read back from data/cache/search/. SEARCH_EMBEDDINGS and VOYAGE_API_KEY
    are ignored here: they would change what is measured."""
    index = ix.get(embeddings="local")
    results = exam.run(exam.EXAM, index)
    text = exam.report(results, index.how)
    print("\n" + text)
    return exam.score(results), text


@pytest.mark.private
@pytest.mark.search_model
def test_the_exam_never_gets_worse():
    s, text = _exam_run()
    assert s["found_right_first"] >= exam.RECORDED["found_right_first"], text
    assert s["false_found"] <= exam.PASS_FALSE_FOUND, text


@pytest.mark.search_model
@pytest.mark.xfail(strict=True, reason="L8b pass mark not reached yet: 20 of 30 (plan/work/L8b-notes.md). When this "
                                       "XPASSes, remove this line and raise exam.RECORDED.")
def test_the_exam_reaches_the_pass_mark():
    """The plan's Done-when (L8b): >= 27 of 30 found with the right FIRST citation, <= 2 of 10 false 'found'."""
    s, text = _exam_run()
    assert exam.passed(s), text
