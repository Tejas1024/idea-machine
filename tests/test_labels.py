"""Labels for "his eyes" (L15.1): his drawn answers read word for word, the label file, the daily charts (only after
his T9 opt-in), his replies through the ONE bot (a label is not a lesson; more words still are). No money anywhere."""
from __future__ import annotations

import datetime as dt
import io
import json
import re
from fnmatch import fnmatchcase
from pathlib import Path

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


def test_never_the_same_zone_day_or_band_twice_and_one_days_charts_share_no_candle():
    pool = _pool()
    sent: list[dict] = []
    for d in range(4):
        picks = daily.pick(pool, sent, WORD, DAY + dt.timedelta(d))
        assert len(picks) == sum(config.LABELS_PER_STEP)
        assert not {p["zone_id"] for p in picks} & {s["zone_id"] for s in sent}
        days = [pd.Timestamp(p["day"]) for p in picks] + [pd.Timestamp(s["known_day"]) for s in sent]
        assert len(days) == len(set(days))                            # one chart per day of zones
        spans = sorted((p["src_first"], p["src_last"]) for p in picks)
        assert all(b[0] > a[1] for a, b in zip(spans, spans[1:]))     # ten different stretches of price
        for p in picks:                                               # never the same band on candles shown before
            for s in sent:
                shared = p["src_first"] <= pd.Timestamp(s["last"]) and pd.Timestamp(s["first"]) <= p["src_last"]
                assert not (shared and min(p["hi"], s["hi"]) > max(p["lo"], s["lo"]))
        sent += _as_sent(picks)


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
    # he stops answering: no new charts once LABELS_WAITING_MAX wait
    daily.send(WORD, today=DAY + dt.timedelta(1), outbox=box, m5=m5, h1=h1)
    assert daily.send(WORD, today=DAY + dt.timedelta(2), outbox=box, m5=m5, h1=h1)["sent"] == 0


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
