"""L10 — the report (machine/report/build.py) and the referee's new parts it reads (tune.py, the path counts).

The prompt's tests: every number from the referee · banned words · CHECK once · the "already seen" line · hedged
wording. Hand-made bars are random walks, not gold; every file goes to temporary folders (tests/conftest.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from machine import config
from machine.ideas import store
from machine.ideas.form import Idea
from machine.referee import judge, prior, splits, tune
from machine.referee.replays import outcome_counts, path_counts
from machine.report import build as B
from machine.tester import Signal
from machine.tester.engine import replay
from tests.test_referee import signals_for, walk

IDEA = "idea-901"
REAL_FORM = config.ROOT / "plan" / "work" / "ideas" / "idea-001" / "form.json"


# ================================================================================================================
# a whole idea on hand-made bars: form -> tune (2 versions, 2019) -> check years opened -> judged (2021)
# ================================================================================================================

def make_form(idea: str = IDEA, hedged: bool = True) -> Idea:
    d = json.loads(REAL_FORM.read_text(encoding="utf-8"))
    d.update(id=idea, status="ANSWERED")
    if not hedged:
        d["exit"] = {"kind": "stop", "at": d["exit"]["at"], "beyond_usd": 0.0, "sources": d["exit"]["sources"]}
    return Idea.load(d)


def judged_idea(idea: str = IDEA, *, hedged: bool = True, seen: bool = False, n_random: int = 8) -> dict:
    form = make_form(idea, hedged)
    store.create(form)
    m1 = walk("2019-03-11", days=2)
    versions = {"v1": signals_for(m1, idea, "v1"), "v2": signals_for(m1, idea, "v2", every=45)}
    t = tune.run_versions(versions, tf="5m", m1=m1, about={"v1": "every 30 minutes", "v2": "every 45 minutes"},
                          readings={"v1": {"U1": "A"}, "v2": {"U1": "B"}})
    splits.open_check(idea, t.chosen, who="test", why="chosen on the tune years")
    m1c = walk("2021-03-08", days=4)
    key = (prior.IdeaKey.make("liquidity_sweep+break_of_structure+fvg", "5m", "gap:near") if seen
           else prior.key_for_form(form))
    rep = judge.judge_check(signals_for(m1c, idea, t.chosen), tf="5m", key=key, m1=m1c, n_random=n_random)
    return {"form": form, "tune": t, "report": rep}


@pytest.fixture
def hedged_idea():
    return judged_idea()


def numbers_in(obj) -> list[float]:
    """Every number anywhere in a trial file's result."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in numbers_in(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in numbers_in(v)]
    return []


# ================================================================================================================
# 1. every number from the referee
# ================================================================================================================

def test_every_rupee_in_the_text_is_a_number_of_the_referee_files(hedged_idea):
    b = B.build(IDEA)
    f = b.facts
    allowed = {B.inr(x) for x in numbers_in(f.judge["result"]) + numbers_in(f.tune["result"])}
    allowed |= {"₹40", "₹0", "₹100", "₹10", "₹25", "₹50"}          # the cost, and the rules' own numbers
    shown = re.findall(r"-?₹[\d,]+", b.text)
    assert shown, "the report shows no rupees"
    for amount in shown:
        assert amount in allowed, f"{amount} is not a number in the referee's files"


def test_the_report_reads_the_file_not_a_sum_of_its_own(hedged_idea):
    """Change one number in the referee's trial file: the report follows it (nothing is recomputed from trades)."""
    path = hedged_idea["report"].trial_path
    body = json.loads(path.read_text(encoding="utf-8"))
    body["result"]["a_year"]["net_inr"] = 123456.0
    body["result"]["outcomes"]["in_profit"] = 77
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    text = B.telegram_text(B.facts(IDEA))
    assert "₹123,456 a year" in text and "(77 · " in text


def test_the_lines_of_the_plan_are_all_there(hedged_idea):
    text = B.build(IDEA).text
    for words in ("Idea #901", "family of: new", "Came to the level after the setup", "Fake levels",
                  "Money after costs, 0.01 lot", "a year (2016–2020, the best of 2 versions there)",
                  "a year (2021–2023, hidden while choosing)", "With the broker's swap", "nights held",
                  "with the real spread", "Worst year", "without the best 3 trades", "Random entries: it beats",
                  "setups skipped (one trade at a time)", "Versions tried: 2 of 20", "slices chosen before: 8 of 8",
                  "looked at, not tested: 0", "Smallest edge this test could see", "Already seen by the old project",
                  "Hidden years (2021–2023) opened in the whole project: 1 time",
                  "Works best in / fails in: not measured yet", "Project so far: 1 idea judged", "ANSWER:",
                  "The 6 checks", "1. luck test", "6. not already seen", "report.html"):
        assert words in text, words
    assert len(text) <= 4096


def test_a_version_of_an_older_idea_names_that_idea_in_family_of():
    """An idea with the same blocks, chart and entry as an older one is a version OF it (likhith.md section D): the
    research tools find the older idea and the trial files carry it, the form does not. Found 8 Oct on made-up prices:
    such an idea, judged inside 2016-2020 because the family's hidden years were used, was reported as "family of: new
    (no older idea)" - the reader could not tell why 2021-2023 was not used."""
    from types import SimpleNamespace
    form = make_form("idea-902")
    assert form.parent_idea is None
    line = B.family_line(SimpleNamespace(form=form, judge={"parent_idea": "idea-900"}, mode="walk-forward"))
    assert line.startswith("Idea #900 (the same blocks, chart and entry: a version of it)")
    assert "the hidden years 2021–2023 were opened once for that family" in line and "inside 2016–2020" in line
    assert B.check_words(line) == []
    assert B.family_line(SimpleNamespace(form=form, judge={"parent_idea": None}, mode="check")) == \
        "new (no older idea)"
    assert B.family_line(SimpleNamespace(form=form, judge={"parent_idea": "entry-type1"}, mode="check")) \
        .startswith("Type 1: liquidity taken, break, the first gap")


def test_path_and_outcome_counts_on_hand_made_bars():
    """A limit that comes back, one that sees its target first (cancelled: never came back), one skipped while a
    trade is on, and a trade still open at the end - which is never a profit (lesson B6)."""
    m1 = walk("2019-03-11", days=1, seed=3)
    t0 = m1.index[10]
    px = float(m1.close.loc[t0])
    base = dict(idea_id="idea-x", version_id="v1", direction="buy", levels="from_fill")
    sigs = [Signal(id="a", known_at=t0 + pd.Timedelta(minutes=1), stop=50.0, targets=(50.0,), **base),   # stays open
            Signal(id="b", known_at=t0 + pd.Timedelta(minutes=30), stop=1.0, targets=(1.0,), **base)]   # skipped
    rep = replay(sigs, m1, "5m", 1)
    p, o = path_counts([rep]), outcome_counts([rep])
    assert p["skipped_trade_on"] == 1 and p["had_chance"] == 0 and p["next_open_filled"] == 1
    assert o["open_at_end"] == 1 and o["in_profit"] == 0 and o["closed"] == 0
    lim = [Signal(id="c", known_at=t0 + pd.Timedelta(minutes=1), direction="buy", entry_type="limit",
                  entry_price=round(px - 0.01, 2), stop=round(px - 5, 2), targets=(round(px + 5, 2),), expiry_bars=50,
                  idea_id="idea-x", version_id="v1"),
           Signal(id="d", known_at=t0 + pd.Timedelta(minutes=600), direction="buy", entry_type="limit",
                  entry_price=round(px - 200, 2), stop=round(px - 205, 2), targets=(round(px - 190, 2),),
                  expiry_bars=2, idea_id="idea-x", version_id="v1")]
    p = path_counts([replay(lim, m1, "5m", 1)])
    assert p["never_came_back"]["expired"] + p["never_came_back"]["target reached first"] >= 1
    assert p["had_chance"] == p["came_back"] + sum(p["never_came_back"].values())


# ================================================================================================================
# 2. banned words
# ================================================================================================================

@pytest.mark.parametrize("bad", ["the leftover leg", "the verdict", "a placebo test", "the baseline", "1.5 R",
                                 "a basket of trades", "p < 0.05", "the p-value", "t = 0.41", "n = 25",
                                 "the Deflated Sharpe is 0.9", "PBO 40"])
def test_banned_words_are_caught(bad):
    assert B.check_words(f"The report: {bad} here."), bad


@pytest.mark.parametrize("fine", ["the chance its edge is real (Deflated Sharpe)", "PBO (the chance the choice was "
                                  "luck)", "₹1,234 a year", "the hedge point", "harm, alarm, warm, R&D-free"])
def test_plain_words_pass(fine):
    assert B.check_words(fine) == [], fine


def test_the_text_the_page_and_the_picture_have_no_banned_word(hedged_idea):
    b = B.build(IDEA)
    assert B.check_words(b.text) == []
    assert B.check_words(B._visible(b.page_path.read_text(encoding="utf-8"))) == []
    assert all(B.check_words(t) == [] for t in b.picture["texts"])
    assert "(p <" not in b.page_path.read_text(encoding="utf-8")


# ================================================================================================================
# 3. CHECK once
# ================================================================================================================

def test_the_report_counts_the_openings_and_never_opens_them(hedged_idea):
    before = splits.opened_count()
    b = B.build(IDEA)
    assert splits.opened_count() == before == 1
    assert "opened in the whole project: 1 time" in b.text
    assert b.facts.mode == "check" and b.facts.warnings == []


def test_a_check_judged_for_another_version_is_refused(hedged_idea):
    path = config.IDEAS_DIR / IDEA / "check-opened.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["version_id"] = "v-other"
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(B.ReportError, match="ONCE, for one version"):
        B.facts(IDEA)


def test_a_second_look_at_the_check_years_is_named_and_the_first_counts(hedged_idea):
    first = hedged_idea["report"]
    m1c = walk("2021-03-08", days=4, seed=11)
    judge.judge_check(signals_for(m1c, IDEA, first.version_id), tf="5m", key=prior.key_for_form(make_form()),
                      m1=m1c, n_random=4)
    f = B.facts(IDEA)
    assert f.judge["trial_id"] == json.loads(first.trial_path.read_text(encoding="utf-8"))["trial_id"]
    assert any("judged 2 times" in w for w in f.warnings)
    assert "judged 2 times" in B.telegram_text(f)


def test_no_referee_answer_no_report():
    store.create(make_form("idea-902"))
    with pytest.raises(B.ReportError, match="no referee answer yet"):
        B.facts("idea-902")


def test_changed_rules_refuse_the_report(hedged_idea):
    path = hedged_idea["report"].trial_path
    body = json.loads(path.read_text(encoding="utf-8"))
    body["settings"]["rules_sha256"] = "0" * 64
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(B.ReportError, match="rules"):
        B.facts(IDEA)


# ================================================================================================================
# 4. the "already seen" line
# ================================================================================================================

def test_a_fresh_idea_names_its_nearest_old_tries_in_words(hedged_idea):
    text = B.build(IDEA).text
    line = next(ln for ln in text.splitlines() if ln.startswith("Already seen"))
    assert line.startswith("Already seen by the old project: no. Nearest old tries (not the same idea):")
    assert "Type 1: liquidity grab → break → gap (5-minute)" in line and "29 tries, 2016–2023" in line
    assert "type1_grab_gap" not in line                               # #113: no short names


def test_a_seen_idea_says_yes_and_needs_fresh_years():
    judged_idea("idea-903", seen=True, n_random=4)
    text = B.build("idea-903").text
    assert "Already seen by the old project: YES" in text and "not fresh" in text
    assert "ANSWER: CHANCE — NEEDS FRESH YEARS" in text


def test_idea_1_key_is_his_grab_break_gap_on_equal_levels():
    key = prior.key_for_form(Idea.load(REAL_FORM))
    assert key.blocks == {"liquidity_sweep", "break_of_structure", "fvg", "equal_highs_lows"}
    assert key.entry == "gap:near" and key.timeframes == {"5m"} and key.mgmt == {"hedge"}
    s = prior.already_seen(key, config.CHECK_YEARS)
    assert not s.seen
    assert prior.closest(key, s)[0].idea_key == "type1_grab_gap"


# ================================================================================================================
# 5. hedged wording
# ================================================================================================================

def test_a_hedged_idea_is_never_told_hit_the_target_before_the_stop(hedged_idea):
    text = B.build(IDEA).text
    assert "closed in profit" in text and "closed at a loss" in text and "still open at the end" in text
    assert "hedge point was treated as a plain stop" in text
    assert "hit the target before the stop" not in text
    o = hedged_idea["report"].numbers["outcomes"]
    assert o["in_profit"] + o["at_a_loss"] == o["closed"]


def test_a_stop_idea_says_hit_the_target_before_the_stop():
    judged_idea("idea-904", hedged=False, n_random=4)
    text = B.build("idea-904").text
    assert "hit the target before the stop" in text
    assert "closed in profit" not in text and "hedge" not in text.lower().split("already seen")[0]


# ================================================================================================================
# the picture, the page, the send; the tune step
# ================================================================================================================

def test_the_picture_is_a_real_jpeg_with_the_hidden_years_shaded(hedged_idea):
    b = B.build(IDEA)
    raw = Path(b.picture["path"]).read_bytes()
    assert raw[:3] == b"\xff\xd8\xff"
    assert b.picture["years"] == list(range(2016, 2024)) and b.picture["hidden"] == [2021, 2022, 2023]
    assert "hidden while choosing" in b.picture["texts"]


def test_the_page_shows_every_trade(hedged_idea):
    b = B.build(IDEA)
    page = b.page_path.read_text(encoding="utf-8")
    n_check = len(b.facts.judge["result"]["trades_table"]["rows"])
    n_tune = len(b.facts.tune["result"]["trades_table"]["rows"])
    assert page.count("<tr><td>") >= n_check + n_tune
    assert "India time" in page and "THE RULES" in page


def test_send_puts_one_line_in_the_outbox_and_never_the_same_picture_twice(hedged_idea, tmp_path):
    b = B.build(IDEA)
    outbox, shown = tmp_path / "outbox.jsonl", tmp_path / "shown.jsonl"
    line = B.send(b, outbox=outbox, shown=shown)
    got = [json.loads(x) for x in outbox.read_text(encoding="utf-8").splitlines()]
    assert got == [line] and line["photo"].endswith(B.PICTURE) and line["shown"]["pixel_sha256"]
    shown.write_text(json.dumps({"pixel_sha256": line["shown"]["pixel_sha256"]}) + "\n", encoding="utf-8")
    with pytest.raises(B.ReportError, match="never the same picture twice"):
        B.send(b, outbox=outbox, shown=shown)


def test_tune_tests_every_version_and_records_one_choice(_trials_in_tmp):
    m1 = walk("2019-03-11", days=2)
    vs = {"v1": signals_for(m1, "idea-a", "v1"), "v2": signals_for(m1, "idea-a", "v2", every=45)}
    t = tune.run_versions(vs, tf="5m", m1=m1)
    kinds = sorted(json.loads(p.read_text(encoding="utf-8"))["kind"] for p in _trials_in_tmp.glob("*.json"))
    assert kinds == ["referee-tune", "tester", "tester"]
    best = max(t.choice["versions"], key=lambda k: t.choice["versions"][k]["deflated_sharpe"])
    assert t.chosen == best and tune.latest("idea-a")["result"]["chosen"] == best
    assert t.numbers["trades_table"]["rows"]


def test_tune_never_reads_the_check_years():
    m1 = walk("2020-12-30", days=4)                                   # reaches into 2021
    with pytest.raises(ValueError, match="outside the judged years"):
        tune.run_versions({"v1": signals_for(m1, "idea-a", "v1")}, tf="5m", m1=m1)


# ================================================================================================================
# the cancel price (#119: price reached the target before it came back) in the referee's comparisons (L10 fix)
# ================================================================================================================

def _with_cancel(m1):
    """The limit signals of signals_for, each with a cancel price at its target (like idea #1: compile.py)."""
    out = []
    for s in signals_for(m1):
        if s.entry_type == "limit":
            s = Signal(**{**s.model_dump(), "cancel_at": s.targets[0]})
        out.append(s)
    return out


def test_fake_levels_move_the_cancel_price_with_the_target():
    from machine.referee import fake_levels
    m1 = walk("2019-03-11")
    lim = next(s for s in _with_cancel(m1) if s.entry_type == "limit")          # a sell limit, target $1.5 below
    for step in config.FAKE_LEVEL_STEPS:                                         # -2 used to refuse: cancel above entry
        m = fake_levels.moved(lim, step)
        assert m.cancel_at - m.entry_price == pytest.approx(lim.cancel_at - lim.entry_price)
        assert m.cancel_at == m.targets[0]


def test_random_entries_keep_the_cancel_distance():
    from machine.referee import random_entries
    from machine.referee.replays import make_segment
    m1 = walk("2019-03-11")
    sigs = _with_cancel(m1)
    seg = make_segment("t", (2019,), "5m", sigs, m1, strict=True)
    lim = next(s for s in sigs if s.entry_type == "limit")
    t = random_entries.template(seg, lim)
    assert t.cancel_d == pytest.approx(1.5)
    r = random_entries.make_signal(seg, t, seg.known_times()[500], 0.8, "r1")
    assert r.entry_price - r.cancel_at == pytest.approx(1.5) and r.cancel_at == r.targets[0]
    nxt = random_entries.template(seg, sigs[0])
    assert nxt.cancel_d is None


def test_the_report_speaks_to_him():
    assert B.to_him("no minimum (his written rule) · this · his 'Ok'") == "no minimum (your written rule) · this · your 'Ok'"
