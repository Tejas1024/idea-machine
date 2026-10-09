"""Labels for "his eyes" (L15.1): his drawn answers read word for word, the label file, the daily charts (only after
his T9 opt-in), his replies through the ONE bot (a label is not a lesson; more words still are). No money anywhere."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from machine import config
from machine.charts import draw as D
from machine.labels import daily, drawings as DW, picture as P, replies as LR, store, words as W
from machine.telegram import bot as B
from machine.telegram import inbox
from tests.telegram_fakes import TEJAS, FakeTelegram, chart_picture

WORD = W.STRONG_ZONE.slug
DAY = dt.date(2026, 10, 1)


# ============================================================================================ his drawings

def test_every_picture_of_his_in_the_folder_was_read():
    """Every file in strategy/pictures/ is in drawings.PICTURES, except pictures the bot saved from Telegram after L9a
    (a later session reads those); and nothing listed is missing."""
    folder = config.ROOT / "strategy" / "pictures"
    have = {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    bot_made = {n for n in have if any(fnmatchcase(f"strategy/pictures/{n}", pat) for pat in config.BOT_GIT_FILES)}
    listed = {d.file for d in DW.PICTURES}
    assert not (have - bot_made) - listed, f"not read yet: {sorted((have - bot_made) - listed)}"
    assert not listed - have, f"listed but not in the folder: {sorted(listed - have)}"
    assert len(DW.PICTURES) == len(listed) == 43


def _lessons_text() -> dict[int, str]:
    text = (config.ROOT / "strategy" / "LEARNED.md").read_text(encoding="utf-8")
    return {int(m.group(1)): " ".join(p.split()) for p in re.split(r"(?m)^(?=## #\d+)", text)
            if (m := re.match(r"## #(\d+)\s", p))}


def test_his_words_are_word_for_word_in_the_lessons_they_cite():
    lessons = _lessons_text()
    for x in DW.LABELS:
        cited = [int(s.split("#")[1]) for s in x.lessons]
        for piece in x.his_words.split(" … "):
            piece = " ".join(piece.split())
            assert any(piece in lessons[n] for n in cited), f"{x.key}: '{piece}' is not in {x.lessons}"


def test_the_labels_are_his_answers():
    got = {x.key: x.label for x in DW.labels(WORD)}
    assert got == {"Q78-A": "yes", "Q78-B": "no", "Q79-B": "yes", "Q79-D": "yes", "Q79-A": "no", "Q79-C": "no"}
    files = {d.file: d for d in DW.PICTURES}
    for x in DW.LABELS:
        assert files[x.picture].word == "strong zone" and x.label in W.LABELS


def test_the_hidden_years_are_kept_but_never_drawn():
    hidden = {x.key for x in DW.LABELS if x.hidden}
    assert hidden == {"Q79-A", "Q79-C"}
    for x in DW.LABELS:
        assert (x.hi is None) == x.hidden                     # their band's top was never read


def test_his_candles_are_on_the_server_clock():
    q79b = next(x for x in DW.LABELS if x.key == "Q79-B")
    assert q79b.span() == (pd.Timestamp("2017-12-26 10:10"), pd.Timestamp("2018-01-03 23:50"))   # UTC+2 in winter
    q79d = next(x for x in DW.LABELS if x.key == "Q79-D")
    assert q79d.span() == (pd.Timestamp("2017-05-01 18:05"), pd.Timestamp("2017-05-08 23:50"))   # UTC+3 in summer


def test_the_catalogue_page_lists_every_picture_and_label():
    page = DW.page()
    for d in DW.PICTURES:
        assert f"`{d.file}`" in page
    for x in DW.LABELS:
        assert f"| {x.key} | **{x.label}** |" in page


# ============================================================================================ his replies

@pytest.mark.parametrize("words,want", [
    ("yes", [(None, "yes")]), ("Yes.", [(None, "yes")]), ("YEAH", [(None, "yes")]), ("y", [(None, "yes")]),
    ("no", [(None, "no")]), ("Nope", [(None, "no")]), ("No.", [(None, "no")]),
    ("not sure", [(None, "not sure")]), ("Not sure.", [(None, "not sure")]), ("notsure", [(None, "not sure")]),
    ("don't know", [(None, "not sure")]), ("maybe", [(None, "not sure")]),
    ("1 yes 2 no 3 not sure", [(1, "yes"), (2, "no"), (3, "not sure")]),
    ("1-yes, 2-no", [(1, "yes"), (2, "no")]), ("10. no", [(10, "no")]),
    ("no, the candles are too big", [(None, "no")]),
    ("B", []), ("no-trade zone", []), ("yesterday it was quiet", []), ("", []),
])
def test_reading_yes_no_not_sure(words, want):
    assert LR.read(words) == want


@pytest.mark.parametrize("words,more", [
    ("yes", False), ("Yes. This is my kind of zone", False), ("1 yes 2 no 3 not sure", False),
    ("no, the candles are too big here", True), ("not sure - it moved too much before the band", True),
])
def test_more_than_a_label_is_also_a_lesson(words, more):
    assert LR.more_than_labels(words) is more


def test_a_label_question_id():
    q = LR.qid(WORD, DAY, 3)
    assert q == "LBL-strong_zone-20261001-03" and LR.is_label_question({"qid": q})
    assert LR.parts(q) == (WORD, "20261001", 3)
    assert not LR.is_label_question({"qid": "L6.2-Q1-lows-too-near"}) and not LR.is_label_question(None)


# ============================================================================================ the bot

@pytest.fixture
def where(tmp_path) -> inbox.Where:
    return inbox.Where(inbox=tmp_path / "learned-inbox", pictures=tmp_path / "pictures", bot=tmp_path / "bot",
                       outbox=tmp_path / "work" / "telegram-outbox.jsonl", intake=tmp_path / "work" / "intake.jsonl",
                       labels=tmp_path / "labels")


@pytest.fixture
def tg() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
def bot(tg, where) -> B.Bot:
    return B.Bot(tg, TEJAS, where, echo=False)


def _charts(where, n: int = 3) -> list[str]:
    """n charts 'sent' today: their lines in charts.jsonl and in the outbox (as daily.send writes them)."""
    qids = [LR.qid(WORD, DAY, k) for k in range(1, n + 1)]
    path = LR.charts_file(WORD, where.labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    where.outbox.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh, open(where.outbox, "a", encoding="utf-8") as ob:
        for k, q in enumerate(qids, 1):
            fh.write(json.dumps({"qid": q, "day": f"{DAY}", "n": k}) + "\n")
            ob.write(json.dumps({"id": f"label-{q}", "qid": q, "text": f"Zone chart {k} of {n}. Your kind of zone? "
                                 "Reply to this picture: yes / no / not sure"}) + "\n")
    return qids


def _sent_id(tg, qid: str) -> int:
    return next(s["message_id"] for s in tg.sent if s.get("text", "").startswith("Zone chart")
                and s["text"].split()[2] == str(LR.parts(qid)[2]))


def _inbox(where) -> list[Path]:
    return sorted(where.inbox.glob("*.md")) if where.inbox.is_dir() else []


def test_a_plain_label_goes_to_the_label_file_not_the_lesson_inbox(bot, tg, where):
    qids = _charts(where)
    bot.send_outbox()
    u = tg.text("Yes.", reply_to=_sent_id(tg, qids[1]))
    bot.poll_once()
    [row] = LR.read_replies(WORD, where.labels)
    assert (row["qid"], row["label"], row["words"]) == (qids[1], "yes", "Yes.")
    assert _inbox(where) == []                                           # a label is not a lesson
    reply = tg.replies_to(u)[0]
    assert reply.startswith("Saved: yes.") and "answered so far: 1" in reply


def test_a_label_with_a_reason_is_also_a_lesson(bot, tg, where):
    qids = _charts(where)
    bot.send_outbox()
    u = tg.text("no, the candles are too big here", reply_to=_sent_id(tg, qids[0]))
    bot.poll_once()
    assert [r["label"] for r in LR.read_replies(WORD, where.labels)] == ["no"]
    [lesson] = _inbox(where)
    text = lesson.read_text(encoding="utf-8")
    assert text.startswith(f"# Answer to label chart {qids[0]} (Telegram, ") and "> no, the candles are too big" in text
    assert "also saved as a lesson" in tg.replies_to(u)[0]


def test_words_that_are_no_label_are_filed_as_a_lesson_and_the_chart_stays_open(bot, tg, where):
    qids = _charts(where)
    bot.send_outbox()
    u = tg.text("B", reply_to=_sent_id(tg, qids[2]))
    bot.poll_once()
    assert LR.read_replies(WORD, where.labels) == []
    assert len(_inbox(where)) == 1
    assert "could not find yes / no / not sure" in tg.replies_to(u)[0]


def test_numbered_answers_label_that_days_charts_and_a_missing_number_is_refused(bot, tg, where):
    qids = _charts(where, 3)
    bot.send_outbox()
    u = tg.text("1 yes 2 no 3 not sure 7 yes", reply_to=_sent_id(tg, qids[2]))
    bot.poll_once()
    got = {r["qid"]: r["label"] for r in LR.read_replies(WORD, where.labels)}
    assert got == {qids[0]: "yes", qids[1]: "no", qids[2]: "not sure"}
    reply = tg.replies_to(u)[0]
    assert "chart 1 yes, chart 2 no, chart 3 not sure" in reply and "no chart 7 that day" in reply


def test_a_drawing_on_a_label_chart_keeps_the_picture_and_the_label(bot, tg, where):
    qids = _charts(where)
    bot.send_outbox()
    tg.photo(chart_picture(), "no", reply_to=_sent_id(tg, qids[0]))
    bot.poll_once()
    assert [r["label"] for r in LR.read_replies(WORD, where.labels)] == ["no"]
    [copy] = list(where.pictures.iterdir())
    assert copy.name == f"{qids[0]}-his-answer.jpg"
    assert fnmatchcase(f"strategy/pictures/{copy.name}", "strategy/pictures/*-his-answer*.jpg")   # the nightly job
    assert copy.name in _inbox(where)[0].read_text(encoding="utf-8")


def test_he_may_change_his_mind_the_newest_reply_counts(bot, tg, where):
    qids = _charts(where)
    bot.send_outbox()
    mid = _sent_id(tg, qids[0])
    tg.text("yes", reply_to=mid)
    tg.text("no", reply_to=mid)
    bot.poll_once()
    last = LR.latest(WORD, where.labels)[qids[0]]
    assert (last["label"], last["replies"]) == ("no", 2) and len(LR.read_replies(WORD, where.labels)) == 2


def test_status_counts_the_label_charts(bot, tg, where):
    qids = _charts(where, 3)
    bot.send_outbox()
    tg.text("yes", reply_to=_sent_id(tg, qids[0]))
    bot.poll_once()
    line = next(ln for ln in bot.status_text().splitlines() if ln.startswith("Label charts"))
    assert "6 from your drawings; 3 sent, 1 answered, 2 waiting" in line and "not asked yet (T9)" in line


# his own way, 1 Oct 2026 (Telegram message 915, word for word): not a reply, his zones listed, the rest not his
HIS_LIST = ("Below are my zones, which I did not mentioned are not mine.\n\n"
            "Also I already taught you all , again you are asking \n\n1.3.7.")


@pytest.mark.parametrize("words,want", [
    (HIS_LIST, [1, 3, 7]), ("1 3 7", [1, 3, 7]), ("my zones: 1, 3 and 7", [1, 3, 7]), ("1,3,7,3", [1, 3, 7]),
    ("10", [10]), ("3", [3]), ("my zone: 3", [3]), ("all my zones: 3", [3]), ("these are all my zones 1 3", [1, 3]),
    ("0.74", []), ("sell at 1:4", []), ("1 yes 2 no", []), ("2350", []), ("12", []), ("1.3.17", []),
    ("I like chart 3 because the candles are small", []), ("", []),
    ("All 3 ok", []), ("all 3 are ok", []), ("ok all 3", []), ("All the 3 are okay", []), ("All 10", []),  # all three
])
def test_his_list_of_chart_numbers(words, want):
    assert LR.listed(words) == want


def test_all_3_ok_while_charts_wait_is_a_lesson_not_his_zone_3(bot, tg, where):
    """"All 3 ok" typed on its own on 8 Oct means the picture of 3 setups (idea #3), not "chart 3 is my zone": the
    bot files it as a lesson - the intake reads it as his YES (tests/test_intake.py) - and labels nothing."""
    _charts(where, 10)
    bot.send_outbox()
    tg.text("All 3 ok")                                                # NOT a reply to a picture
    bot.poll_once()
    assert LR.read_replies(WORD, where.labels) == [] and len(_inbox(where)) == 1
    [logged] = [json.loads(ln) for ln in (where.bot / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged["kind"] == "lesson" and not logged.get("labels")


def test_his_list_says_yes_to_the_listed_charts_and_no_to_the_rest():
    assert LR.read_all("1.3.7.", list(range(1, 11))) == [(n, "yes" if n in (1, 3, 7) else "no") for n in range(1, 11)]
    assert LR.read_all("1 3 12", [1, 2, 3]) == []                      # 12 is no chart number: no list at all
    assert LR.read_all("1 yes 2 no", [1, 2, 3]) == [(1, "yes"), (2, "no")]   # yes / no wins: only what he said


def test_his_list_as_a_plain_message_answers_the_newest_days_charts(bot, tg, where):
    qids = _charts(where, 10)
    bot.send_outbox()
    u = tg.text(HIS_LIST)                                              # NOT a reply to a picture
    bot.poll_once()
    got = {r["qid"]: (r["label"], r["replied_to"]) for r in LR.read_replies(WORD, where.labels)}
    assert got == {q: ("yes" if k in (1, 3, 7) else "no", None) for k, q in enumerate(qids, 1)}
    reply = tg.replies_to(u)[0]
    assert reply.startswith("Saved for the charts of 01 Oct: your zones 1, 3, 7; not yours 2, 4, 5, 6, 8, 9, 10.")
    assert "answered so far: 10" in reply and "also saved as a lesson" in reply   # his other words are a lesson
    [lesson] = _inbox(where)
    assert "not a reply to a picture" in lesson.read_text(encoding="utf-8")
    [logged] = [json.loads(ln) for ln in (where.bot / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged["kind"] == "answer" and logged["labels"]


@pytest.mark.parametrize("words,want", [
    ("none", True), ("None.", True), ("None of them are mine", True), ("nothing today", True), ("0", True),
    ("zero", True), ("No zones today", True), ("all no", True), ("Not even one", True),
    (HIS_LIST.replace("1.3.7.", "none"), True),
    ("no", False), ("No zone", False), ("not mine", False), ("1 3 7", False), ("1 yes\nnone", False),
    ("0.74", False), ("none of the sell trades worked", False), ("Nothing works in Asia", False), ("", False),
])
def test_his_list_of_none(words, want):
    assert LR.none_listed(words) is want


def test_his_list_of_none_says_no_to_every_chart_of_that_day():
    assert LR.read_all("No zones today", [1, 2, 3]) == [(1, "no"), (2, "no"), (3, "no")]
    assert LR.read_all("no", [1, 2, 3]) == [(None, "no")]               # "no" alone still answers its picture
    assert not LR.more_than_labels("None of them are mine")              # only labels: no lesson
    assert LR.more_than_labels(HIS_LIST.replace("1.3.7.", "none"))       # his other words: a lesson as well


def test_none_as_a_plain_message_answers_the_newest_days_charts(bot, tg, where):
    qids = _charts(where, 10)
    bot.send_outbox()
    u = tg.text("None of them are mine")                               # NOT a reply to a picture
    bot.poll_once()
    assert {r["qid"]: r["label"] for r in LR.read_replies(WORD, where.labels)} == {q: "no" for q in qids}
    reply = tg.replies_to(u)[0]
    assert reply.startswith("Saved for the charts of 01 Oct: your zones none; not yours 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.")
    assert _inbox(where) == []


def test_his_list_as_a_reply_to_one_picture_answers_that_whole_day(bot, tg, where):
    qids = _charts(where, 4)
    bot.send_outbox()
    tg.text("2 4", reply_to=_sent_id(tg, qids[0]))
    bot.poll_once()
    assert {r["qid"]: r["label"] for r in LR.read_replies(WORD, where.labels)} == dict(
        zip(qids, ["no", "yes", "no", "yes"]))
    assert _inbox(where) == []


def test_a_line_of_numbers_is_a_lesson_when_no_label_chart_waits(bot, tg, where):
    tg.text("1.3.7.")                                                  # no label charts at all
    bot.poll_once()
    assert len(_inbox(where)) == 1
    qids = _charts(where, 3)
    bot.send_outbox()
    for q in qids:
        tg.text("no", reply_to=_sent_id(tg, q))
    tg.text("1 3")                                                     # every chart of the day is answered
    bot.poll_once()
    assert [r["label"] for r in LR.read_replies(WORD, where.labels)] == ["no", "no", "no"]
    assert len(_inbox(where)) == 2


def test_a_lesson_with_numbers_in_it_stays_a_lesson_while_charts_wait(bot, tg, where):
    _charts(where, 3)
    bot.send_outbox()
    tg.text("Sell after 2 gaps, the stop at 1:4")
    bot.poll_once()
    assert LR.read_replies(WORD, where.labels) == [] and len(_inbox(where)) == 1


def test_the_bots_label_path_draws_nothing():
    """The ONE bot reads replies and STATUS through light modules: no pictures library in its process."""
    import subprocess
    import sys
    code = ("import sys; from machine.labels import replies as r; r.read('1 yes'); r.status_line(); "
            "print('matplotlib' in sys.modules)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=config.ROOT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_the_nightly_job_carries_the_label_files():
    for f in ("plan/work/labels/strong_zone/charts.jsonl", "plan/work/labels/strong_zone/replies.jsonl"):
        assert any(fnmatchcase(f, p) for p in config.BOT_GIT_FILES)
        assert any(fnmatchcase(f, p) for p in config.BOT_APPEND_ONLY)
    assert inbox.Where.real().labels == config.LABELS_WORK_DIR
    assert config.STRATEGY_DIR not in Path(inbox.Where.test(Path("x")).labels).parents


# ============================================================================================ T9: his answer

def test_nothing_is_sent_before_he_opts_in(tmp_path):
    with pytest.raises(daily.LabelsOff):
        daily.send(WORD, today=DAY, m5=pd.DataFrame(), h1=pd.DataFrame())
    daily.opt_in("no", "No, not now.", who="test")
    with pytest.raises(daily.LabelsOff):
        daily.send(WORD, today=DAY, m5=pd.DataFrame(), h1=pd.DataFrame())
    with pytest.raises(daily.LabelsOff):
        daily.install()


def test_his_answer_is_added_never_rewritten_and_the_newest_counts():
    t0 = dt.datetime(2026, 10, 1, 21, 0)
    path = daily.opt_in("opt in", "ok opt in, send them", who="Tejas (T9 session)", now=t0)
    first = path.read_text(encoding="utf-8")
    assert daily.opt_in_state() == {"answer": "opt in", "when": "2026-10-01 21:00"}
    daily.opt_in("no", "stop the charts", who="Tejas", now=t0 + dt.timedelta(days=9))
    text = path.read_text(encoding="utf-8")
    assert text.startswith(first) and "> stop the charts" in text
    assert daily.opt_in_state()["answer"] == "no"
    with pytest.raises(ValueError):
        daily.opt_in("opt in", "   ", who="x")


# ============================================================================================ choosing the charts

def _pool(n_days: int = 200, per_day: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for d in range(n_days):
        day = pd.Timestamp("2017-01-02") + pd.Timedelta(days=d)
        for k in range(per_day):
            lo = 1200 + 10 * k + d
            rows.append(dict(zone_id=f"week:{day.date()}:{lo}", window="week", known_at=day, day=day,
                             src_first=day - pd.Timedelta(days=5), src_last=day - pd.Timedelta(minutes=10),
                             lo=float(lo), hi=float(lo + 2), calm=float(rng.uniform(0.4, 1.4)), tier=2))
    z = pd.DataFrame(rows)
    cuts = [hi for _lo, hi in daily.steps()[:-1]]
    z["step"] = np.searchsorted(np.asarray(cuts), z["calm"].to_numpy(float), side="left")
    return z


def _as_sent(picks: list[dict]) -> list[dict]:
    return [{"zone_id": p["zone_id"], "known_day": str(p["day"]), "first": str(p["src_first"]),
             "last": str(p["src_last"]), "lo": p["lo"], "hi": p["hi"]} for p in picks]


def test_the_same_day_picks_the_same_charts_and_the_steps_are_kept():
    pool = _pool()
    a = daily.pick(pool, [], WORD, DAY)
    assert [p["zone_id"] for p in a] == [p["zone_id"] for p in daily.pick(pool, [], WORD, DAY)]
    assert [p["zone_id"] for p in a] != [p["zone_id"] for p in daily.pick(pool, [], WORD, DAY + dt.timedelta(1))]
    assert len(a) == sum(config.LABELS_PER_STEP)
    want = dict(enumerate(config.LABELS_PER_STEP))
    got = pd.Series([p["step"] for p in a]).value_counts().to_dict()
    assert all(got.get(s, 0) == n for s, n in want.items() if (pool["step"] == s).sum() >= 3 * n)


def test_never_the_same_zone_day_or_candle_twice():
    pool = _pool(n_days=800)                                          # 5-day charts: room for 4 days of 10
    sent: list[dict] = []
    for d in range(4):
        picks = daily.pick(pool, sent, WORD, DAY + dt.timedelta(d))
        assert len(picks) == sum(config.LABELS_PER_STEP)
        assert not {p["zone_id"] for p in picks} & {s["zone_id"] for s in sent}
        days = [pd.Timestamp(p["day"]) for p in picks] + [pd.Timestamp(s["known_day"]) for s in sent]
        assert len(days) == len(set(days))                            # one chart per day of zones
        spans = sorted((p["src_first"], p["src_last"]) for p in picks)
        assert all(b[0] > a[1] for a, b in zip(spans, spans[1:]))     # ten different stretches of price
        for p in picks:                                               # #117: never a candle shown before, any band
            for s in sent:
                assert not (p["src_first"] <= pd.Timestamp(s["last"]) and pd.Timestamp(s["first"]) <= p["src_last"])
        sent += _as_sent(picks)


def test_the_charts_end_when_every_stretch_of_price_was_shown():
    pool = _pool(n_days=60)                                           # room for about 12 charts of 5 days
    sent: list[dict] = []
    for d in range(10):
        sent += _as_sent(daily.pick(pool, sent, WORD, DAY + dt.timedelta(d)))
    assert 0 < len(sent) <= 60 // 5 + 1
    assert daily.pick(pool, sent, WORD, DAY + dt.timedelta(10)) == []
    spans = sorted((pd.Timestamp(s["first"]), pd.Timestamp(s["last"])) for s in sent)
    assert all(b[0] > a[1] for a, b in zip(spans, spans[1:]))


def test_when_no_chart_is_left_he_is_told_once(monkeypatch, tmp_path):
    daily.opt_in("opt in", "opt in", who="test")
    empty = _pool().iloc[0:0]
    monkeypatch.setattr(daily, "candidates", lambda word, m5, h1: empty)
    box = tmp_path / "outbox.jsonl"
    for d in range(3):
        out = daily.send(WORD, today=DAY + dt.timedelta(d), outbox=box, m5=pd.DataFrame(), h1=pd.DataFrame())
        assert out["sent"] == 0 and "no chart left" in out["said"]
    [line] = [json.loads(x) for x in box.read_text(encoding="utf-8").splitlines()]
    assert line["id"] == f"labels-{WORD}-finished" and "No more zone charts" in line["text"]


def test_a_short_step_gives_its_place_to_the_nearest_steps():
    pool = _pool()
    pool.loc[pool["step"] == 0, "step"] = 1                          # nobody quiet enough for his rule
    picks = daily.pick(pool, [], WORD, DAY)
    assert len(picks) == sum(config.LABELS_PER_STEP) and all(p["step"] != 0 for p in picks)
    assert all(p["step_size"] > 0 for p in picks)


# ============================================================================================ with the price data

@pytest.fixture(scope="module")
def bars():
    from machine.data.bars import load_bars
    return load_bars("5m", "2016", "2020"), load_bars("1h", "2016", "2020")


@pytest.mark.data
def test_the_candidates_are_the_tune_years_week_window_and_not_his_drawings(bars):
    m5, h1 = bars
    pool = daily.candidates(WORD, m5, h1)
    assert len(pool) > 2000 and set(pool["window"]) == {"week"}
    assert pool["src_first"].min() >= pd.Timestamp("2016-01-01")
    assert pool["src_last"].max() < pd.Timestamp("2021-01-01") and pool["known_at"].max() <= pd.Timestamp("2021-01-01")
    for x in DW.labels(WORD):
        if not x.hidden:
            a, b = x.span()
            assert not ((pool["src_first"] <= b) & (a <= pool["src_last"])).any(), x.key
    assert (pool["step"] == 0).sum() == (pool["calm"] <= 0.74).sum()


@pytest.mark.data
def test_a_day_of_charts_end_to_end(bars, tmp_path):
    m5, h1 = bars
    daily.opt_in("opt in", "opt in", who="test", now=dt.datetime(2026, 10, 1, 20, 0))
    box = tmp_path / "outbox.jsonl"
    out = daily.send(WORD, today=DAY, outbox=box, m5=m5, h1=h1)
    rows = LR.read_charts(WORD)
    assert out["sent"] == len(rows) == config.LABELS_PER_DAY
    lines = [json.loads(x) for x in box.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == config.LABELS_PER_DAY + 1 and "qid" not in lines[0]
    for row, ln in zip(rows, lines[1:]):
        assert ln["qid"] == row["qid"] and ln["shown"]["pixel_sha256"] == row["pixel_sha256"]
        pic = config.ROOT / ln["photo"] if not Path(ln["photo"]).is_absolute() else Path(ln["photo"])
        assert pic.is_file() and pic.read_bytes()[:3] == b"\xff\xd8\xff"
        assert pd.Timestamp(row["last"]) < pd.Timestamp(row["known_at"])          # nothing after the band is known
        assert pd.Timestamp(row["last"]).year <= 2020
        assert 0 <= row["step"] < len(config.LABELS_PER_STEP) and row["tier"] in (1, 2, 3)
    assert daily.send(WORD, today=DAY, outbox=box, m5=m5, h1=h1)["sent"] == 0     # once a day
    # he does not answer: the next day brings no new charts (one day waits at most, #166), and says so
    assert daily.send(WORD, today=DAY + dt.timedelta(1), outbox=box, m5=m5, h1=h1)["sent"] == 0
    told = [json.loads(x) for x in box.read_text(encoding="utf-8").splitlines()][len(lines):]
    assert len(told) == 1 and "still wait" in told[0]["text"] and "qid" not in told[0]


def test_no_new_charts_while_a_day_waits_and_he_is_told_once_that_day(tmp_path):
    daily.opt_in("opt in", "opt in", who="test", now=dt.datetime(2026, 10, 1, 20, 0))
    charts = LR.charts_file(WORD)
    charts.parent.mkdir(parents=True, exist_ok=True)
    charts.write_text("".join(json.dumps({"qid": LR.qid(WORD, DAY, n), "day": f"{DAY:%Y-%m-%d}", "n": n}) + "\n"
                              for n in range(1, config.LABELS_PER_DAY + 1)), encoding="utf-8")
    box = tmp_path / "outbox.jsonl"
    for _ in range(2):                                    # the job runs twice that day: one line only
        out = daily.send(WORD, today=DAY + dt.timedelta(1), outbox=box)
        assert out["sent"] == 0 and "told" in out["said"]
    lines = [json.loads(x) for x in box.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and "qid" not in lines[0]
    assert f"{DAY:%d %b}" in lines[0]["text"] and "none" in lines[0]["text"]
    assert config.LABELS_WAITING_MAX == config.LABELS_PER_DAY             # one day's charts wait at most (#166)
    assert daily.send(WORD, today=DAY + dt.timedelta(2), outbox=box)["sent"] == 0
    assert len(box.read_text(encoding="utf-8").splitlines()) == 2         # a new day: told again, once


@pytest.mark.data
def test_a_label_picture_ends_where_the_band_is_finished_and_names_no_numbers(bars, tmp_path):
    m5, _h1 = bars
    x = next(x for x in DW.LABELS if x.key == "Q79-B")
    h = P.hour_candles(m5, *x.span())
    assert h.index[-1] == pd.Timestamp("2018-01-03 23:00") and h.index[0] == pd.Timestamp("2017-12-26 10:00")
    drawn = P.draw(W.STRONG_ZONE, h, x.lo, x.hi, tmp_path / "q79b.jpg", subject="test q79b")
    words = " ".join(t.text for t in drawn.texts if t.role != "tick")
    assert not re.search(r"calm|tier|0\.\d\d|₹|\$", words)                     # blind: no calm, no money
    assert round(P.calm(h, x.lo, x.hi), 2) == 0.56                              # the card was built on 0.54


def test_the_band_default_did_not_change_for_other_pictures():
    assert D.Zone(1.0, 2.0).alpha == 0.16


@pytest.mark.data
def test_the_label_file_from_his_drawings(bars, tmp_path):
    m5, _h1 = bars
    df = store.build(WORD, m5=m5)
    assert list(df.columns) == list(store.COLUMNS)
    assert len(df) == 6 and set(df["source"]) == {"his drawing"}
    assert dict(zip(df["label_id"], df["label"])) == {"his-Q78-A": "yes", "his-Q78-B": "no", "his-Q79-B": "yes",
                                                      "his-Q79-D": "yes", "his-Q79-A": "no", "his-Q79-C": "no"}
    drawn = df[df["image"].notna()]
    assert set(drawn["label_id"]) == {"his-Q78-A", "his-Q78-B", "his-Q79-B", "his-Q79-D"}
    for r in drawn.itertuples():
        im = Image.open(io.BytesIO(r.image))
        assert im.format == "JPEG" and im.size == (1560, 1191)
        assert r.image_sha256 == D.pixel_sha256(io.BytesIO(r.image)) and r.years == "tune"
    assert set(df.loc[df["image"].isna(), "years"]) == {"hidden"}
    says = dict(zip(df["label_id"], df["rule_says"]))
    assert says["his-Q79-B"] == "yes" and says["his-Q79-D"] == "yes" and says["his-Q78-B"] == "no"
    path = store.write(df, WORD)
    back = pd.read_parquet(path)
    assert back["label_id"].tolist() == df["label_id"].tolist() and back["image"].iloc[0] == df["image"].iloc[0]


@pytest.mark.data
def test_the_label_file_takes_his_newest_telegram_answer(bars, tmp_path):
    m5, h1 = bars
    daily.opt_in("opt in", "opt in", who="test")
    daily.send(WORD, today=DAY, outbox=tmp_path / "o.jsonl", m5=m5, h1=h1)
    q1, q2 = LR.qid(WORD, DAY, 1), LR.qid(WORD, DAY, 2)
    now = dt.datetime(2026, 10, 1, 21, 30).astimezone()
    LR.record(None, question=q1, got=[(None, "yes")], words="yes", message_id=1, when=now)
    LR.record(None, question=q1, got=[(None, "not sure")], words="not sure", message_id=2, when=now)
    LR.record(None, question=q2, got=[(None, "no")], words="no", message_id=3, when=now)
    df = store.build(WORD, m5=m5)
    tg_rows = df[df["source"] == "telegram"].set_index("label_id")
    assert tg_rows.loc[f"tg-{q1}", "label"] == "not sure" and tg_rows.loc[f"tg-{q1}", "replies"] == 2
    assert tg_rows.loc[f"tg-{q2}", "label"] == "no" and len(tg_rows) == 2           # 8 charts wait: not labels
    assert tg_rows["image"].notna().all() and (tg_rows["years"] == "tune").all()


def test_the_label_file_check_refuses_what_must_never_be_there():
    ok = dict(label_id="a", word=WORD, source="telegram", label="yes", years="tune", year=2019, image=b"x",
              last_candle=pd.Timestamp("2019-05-01"))
    store.check(pd.DataFrame([ok]))
    for bad in ({"label": "maybe"}, {"years": "hidden"}, {"image": None},
                {"last_candle": pd.Timestamp("2024-02-01")}, {"year": 2022}):
        with pytest.raises(ValueError):
            store.check(pd.DataFrame([{**ok, **bad}]))
    with pytest.raises(ValueError):
        store.check(pd.DataFrame([ok, ok]))


# ============================================================================================ the next ones at once
# His words, 7 Oct 2026: "you can send me that 200 or 600 things that needed me" - the next day's charts as soon as he
# has answered, not at the next 21:00 (config.LABELS_NEXT_AT_ONCE, daily.send_next, started by every intake run).

NO_BARS = dict(m5=pd.DataFrame(), h1=pd.DataFrame())
LATE = dt.datetime.combine(DAY, dt.time(23, 30))


def _no_drawing(monkeypatch) -> None:
    """send without the price data: a pool of zones, and a picture that is only a file (drawing is tested above)."""
    pool = _pool(n_days=800)
    monkeypatch.setattr(daily, "candidates", lambda word, m5, h1: pool)
    monkeypatch.setattr(daily.P, "hour_candles", lambda m5, a, b: None)
    monkeypatch.setattr(daily.P, "calm", lambda h, lo, hi: 0.5)
    monkeypatch.setattr(daily.P, "code_fingerprint", lambda: "test")

    def draw(w, h, lo, hi, path, *, subject, shown=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(chart_picture(200, 120))
        return SimpleNamespace(pixel_sha256=hashlib.sha256(subject.encode()).hexdigest())
    monkeypatch.setattr(daily.P, "draw", draw)


def _answer(day: dt.date, words: str = "2 5", n: int = 10) -> None:
    """His plain list for that day's charts 1..n, saved as the bot saves it."""
    LR.record(None, question=LR.qid(WORD, day, 1), got=LR.read_all(words, list(range(1, n + 1))), words=words,
              message_id=1, when=dt.datetime.combine(day, dt.time(22, 0)), as_list=True, reply=False)


def _lines(box: Path) -> list[dict]:
    return [json.loads(x) for x in box.read_text(encoding="utf-8").splitlines()] if box.is_file() else []


@pytest.mark.private                                    # the label count L15.2 needs is read from plan/PROGRESS.md
def test_the_next_days_charts_come_as_soon_as_he_has_answered_every_chart(monkeypatch, tmp_path):
    _no_drawing(monkeypatch)
    daily.opt_in("opt in", "opt in", who="test", now=dt.datetime(2026, 10, 1, 20, 0))
    box = tmp_path / "outbox.jsonl"
    assert daily.send(WORD, today=DAY, outbox=box, **NO_BARS)["sent"] == 10
    out = daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)
    assert out["sent"] == 0 and "10 charts wait" in out["said"]                 # not before he answered
    _answer(DAY, n=9)                                                           # 9 of the 10: one still waits
    assert daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)["sent"] == 0
    _answer(DAY)
    have, before = daily.labels_so_far(), len(_lines(box))
    out = daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)
    nxt = DAY + dt.timedelta(days=1)
    assert out["sent"] == 10 and {r["day"] for r in out["rows"]} == {f"{nxt}"}
    assert all(r["early"] == LATE.isoformat(timespec="seconds") for r in out["rows"])
    [intro, *charts] = _lines(box)[before:]
    assert intro["id"] == f"labels-{WORD}-{nxt:%Y%m%d}-intro" and intro["text"].startswith("Zone charts, set 2: 10 more")
    assert "Every chart is gold from 2016-2020" in intro["text"] and "as soon as you answer" in intro["text"]
    assert f"\"charts of {nxt:%d %b}\", that date is only this set's name" in intro["text"]   # 7 Oct: "not this year"
    assert f"Labels so far: {have} - the L15.2 test needs {daily.l152_needs()}." in intro["text"]
    assert [c["qid"] for c in charts] == [LR.qid(WORD, nxt, k) for k in range(1, 11)]
    assert all(c["text"].startswith(f"Zone chart {k} of 10 (set 2).") for k, c in enumerate(charts, 1))
    assert daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)["sent"] == 0  # one day's charts wait at most
    assert "already queued" in daily.send(WORD, today=nxt, outbox=box, **NO_BARS)["said"]   # that day's 21:00 run
    _answer(nxt, "none")
    out = daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)
    assert {r["day"] for r in out["rows"]} == {f"{nxt + dt.timedelta(days=1)}"}
    zones = [c["zone_id"] for c in LR.read_charts(WORD)]
    assert len(zones) == 30 == len(set(zones))                                  # never the same chart twice


def test_the_set_sent_early_is_for_today_at_the_earliest(monkeypatch, tmp_path):
    _no_drawing(monkeypatch)
    daily.opt_in("opt in", "opt in", who="test")
    box = tmp_path / "outbox.jsonl"
    daily.send(WORD, today=DAY, outbox=box, **NO_BARS)
    _answer(DAY)
    later = dt.datetime.combine(DAY + dt.timedelta(days=5), dt.time(8, 0))     # he answered 5 days later
    assert {r["day"] for r in daily.send_next(WORD, now=later, outbox=box, **NO_BARS)["rows"]} == {f"{later.date()}"}


def test_nothing_at_once_before_he_opts_in_before_the_first_day_or_with_the_switch_off(monkeypatch, tmp_path):
    _no_drawing(monkeypatch)
    box = tmp_path / "outbox.jsonl"
    out = daily.send_next(WORD, outbox=box, **NO_BARS)                          # T9: never before his opt-in
    assert out["sent"] == 0 and "not opted in" in out["said"] and not box.exists()
    daily.opt_in("opt in", "opt in", who="test")
    out = daily.send_next(WORD, outbox=box, **NO_BARS)                          # the first day's: the 21:00 task
    assert out["sent"] == 0 and "no chart sent yet" in out["said"] and not box.exists()
    daily.send(WORD, today=DAY, outbox=box, **NO_BARS)
    _answer(DAY)
    monkeypatch.setattr(config, "LABELS_NEXT_AT_ONCE", False)
    out = daily.send_next(WORD, outbox=box, **NO_BARS)
    assert out["sent"] == 0 and "is off" in out["said"]


def test_two_senders_never_queue_the_same_day(monkeypatch, tmp_path):
    from machine.host.lock import one_run
    _no_drawing(monkeypatch)
    daily.opt_in("opt in", "opt in", who="test")
    box = tmp_path / "outbox.jsonl"
    daily.send(WORD, today=DAY, outbox=box, **NO_BARS)
    _answer(DAY)
    with one_run(daily._lock_path(WORD)) as mine:                              # another run queues charts right now
        assert mine
        assert daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)["sent"] == 0
        assert daily.send(WORD, today=DAY + dt.timedelta(days=1), outbox=box, **NO_BARS)["sent"] == 0
    assert daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)["sent"] == 10  # free again: one set, not two


def test_his_plain_list_answers_the_set_sent_early_through_the_bot(monkeypatch, bot, tg, where):
    _no_drawing(monkeypatch)
    daily.opt_in("opt in", "opt in", who="test")
    daily.send(WORD, today=DAY, outbox=where.outbox, **NO_BARS)
    bot.send_outbox()
    tg.text("1 3 7")
    bot.poll_once()
    assert daily.send_next(WORD, now=LATE, outbox=where.outbox, **NO_BARS)["sent"] == 10
    bot.send_outbox()
    nxt = DAY + dt.timedelta(days=1)
    u = tg.text("2")                                                           # NOT a reply: the set that waits
    bot.poll_once()
    got = {r["qid"]: r["label"] for r in LR.read_replies(WORD) if LR.parts(r["qid"])[1] == f"{nxt:%Y%m%d}"}
    assert got == {LR.qid(WORD, nxt, k): "yes" if k == 2 else "no" for k in range(1, 11)}
    assert tg.replies_to(u)[0].startswith(f"Saved for the charts of {nxt:%d %b}: your zones 2;")


def test_when_his_labels_reach_l152s_need_he_is_told_once(monkeypatch, tmp_path):
    _no_drawing(monkeypatch)
    daily.opt_in("opt in", "opt in", who="test")
    box = tmp_path / "outbox.jsonl"
    daily.send(WORD, today=DAY, outbox=box, **NO_BARS)
    need = daily.labels_so_far() + 10                                         # reached with these 10 answers
    monkeypatch.setattr(daily, "l152_needs", lambda: need)
    assert daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)["told"] is False
    _answer(DAY)
    out = daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)
    assert out["told"] is True and out["sent"] == 10
    [said] = [x for x in _lines(box) if x["id"] == f"labels-{WORD}-reached-{need}"]
    assert said["text"].startswith(f"{need} labels now") and "paste L15.2" in said["text"] and "qid" not in said
    intro = next(x for x in _lines(box) if x["id"].endswith("-intro") and "Zone charts, set" in x["text"])
    assert f"Labels so far: {need}." in intro["text"] and "needs" not in intro["text"]
    _answer(DAY + dt.timedelta(days=1))
    assert daily.send_next(WORD, now=LATE, outbox=box, **NO_BARS)["told"] is False
    assert sum(x["id"] == said["id"] for x in _lines(box)) == 1


def test_l152s_need_is_the_number_next_prompt_reads():
    from machine.tools import next_prompt as NP
    rows = NP.parse_progress(config.PROGRESS_FILE.read_text(encoding="utf-8"))
    assert daily.l152_needs() == next(r.labels_needed for r in rows if r.id == "L15.2") > 0


@pytest.mark.parametrize("at_once,says", [(True, "The next ones come as soon as you answer."),
                                          (False, "New charts come at the next 9 pm after you answer.")])
def test_the_held_line_says_when_the_next_charts_come(monkeypatch, tmp_path, at_once, says):
    monkeypatch.setattr(config, "LABELS_NEXT_AT_ONCE", at_once)
    daily.opt_in("opt in", "opt in", who="test")
    charts = LR.charts_file(WORD)
    charts.parent.mkdir(parents=True, exist_ok=True)
    charts.write_text("".join(json.dumps({"qid": LR.qid(WORD, DAY, n), "day": f"{DAY:%Y-%m-%d}", "n": n}) + "\n"
                              for n in range(1, config.LABELS_PER_DAY + 1)), encoding="utf-8")
    box = tmp_path / "outbox.jsonl"
    daily.send(WORD, today=DAY + dt.timedelta(1), outbox=box)
    [line] = _lines(box)
    assert line["text"].endswith(says)


def test_each_intake_run_sends_the_next_charts_first_and_a_failure_never_stops_it(monkeypatch):
    from machine.intake import __main__ as IM
    from machine.intake import state as S
    calls: list[tuple] = []
    monkeypatch.setattr(S, "run", lambda *a, **k: calls.append(("intake",)) or [])
    monkeypatch.setattr(daily, "send_next", lambda word: calls.append(("next", word)) or
                        {"sent": 0, "told": False, "said": "10 charts wait for his answer"})
    log = config.LABELS_DIR / WORD / "daily.log"
    assert IM.main(["run"]) == 0 and calls == [("next", WORD), ("intake",)]
    assert not log.exists()                                                    # nothing done: no line every 30 min
    monkeypatch.setattr(daily, "send_next", lambda word: {"sent": 10, "told": True, "said": "10 charts queued"})
    assert IM.main(["run"]) == 0
    assert "next at once: 10 charts queued + told: L15.2's need reached" in log.read_text(encoding="utf-8")

    def broken(word):
        raise RuntimeError("a picture rule broke")
    monkeypatch.setattr(daily, "send_next", broken)
    calls.clear()
    assert IM.main(["run"]) == 0 and calls == [("intake",)]                    # the intake goes on
    text = log.read_text(encoding="utf-8")
    assert "next at once FAILED" in text and "a picture rule broke" in text
