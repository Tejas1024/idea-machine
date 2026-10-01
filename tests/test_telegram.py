"""The Telegram bot (L9a): the routing table for each kind, photo + caption -> both copies, a photo is never a
command, lines TO Tejas sent once, only his chat, nothing lost or filed twice, one poller. All on a fake Telegram."""
from __future__ import annotations

import datetime as dt
import json
import urllib.error
from pathlib import Path

import pytest
from PIL import Image

from machine import config
from machine.telegram import api as A
from machine.telegram import bot as B
from machine.telegram import inbox
from machine.telegram import route as R
from tests.telegram_fakes import TEJAS, WHEN, FakeTelegram, chart_picture, phone_screenshot

STAMP = inbox.stamp(dt.datetime.fromtimestamp(WHEN).astimezone())


@pytest.fixture
def where(tmp_path) -> inbox.Where:
    return inbox.Where(inbox=tmp_path / "learned-inbox", pictures=tmp_path / "pictures", bot=tmp_path / "bot",
                       outbox=tmp_path / "work" / "telegram-outbox.jsonl", intake=tmp_path / "work" / "intake.jsonl")


@pytest.fixture
def tg() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
def bot(tg, where) -> B.Bot:
    return B.Bot(tg, TEJAS, where, echo=False)


def _inbox(where) -> list[Path]:
    return sorted(where.inbox.glob("*.md")) if where.inbox.is_dir() else []


def _outbox(where, *lines: dict) -> None:
    where.outbox.parent.mkdir(parents=True, exist_ok=True)
    with open(where.outbox, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def _msg(**kw) -> R.Message:
    base = dict(update_id=1, message_id=2, chat_id=TEJAS, date=dt.datetime(2026, 9, 29, 14, 2).astimezone(), words="")
    return R.Message(**{**base, **kw})


# ============================================================================================ the routing table

@pytest.mark.parametrize("words,kind,command", [
    ("STOP", R.COMMAND, "STOP"), ("START", R.COMMAND, "START"), ("STATUS", R.COMMAND, "STATUS"),
    ("Stop.", R.COMMAND, "STOP"), ("  status ", R.COMMAND, "STATUS"), ("start!", R.COMMAND, "START"),
    ("stop sending me charts", R.LESSON, None), ("STOP STOP", R.LESSON, None), ("status?", R.LESSON, None),
    ("IDEA buy the second touch of equal lows", R.IDEA, None), ("Idea: sell at 0.618", R.IDEA, None),
    ("idea - fade the London high", R.IDEA, None), ("IDEA", R.LESSON, None), ("Ideally I wait for London", R.LESSON, None),
    ("the gap must be the first one", R.LESSON, None),
])
def test_routing_words(words, kind, command):
    r = R.route(_msg(words=words), {})
    assert (r.kind, r.command) == (kind, command)


def test_routing_idea_keeps_his_words_without_the_word_idea():
    assert R.route(_msg(words="IDEA: buy at the gap's near edge"), {}).words == "buy at the gap's near edge"


def test_routing_reply_to_a_question_is_an_answer_even_if_it_says_stop():
    q = {"id": "q-line", "qid": "L6.2-Q1", "idea": "idea-001"}
    r = R.route(_msg(words="STOP", reply_to_id=77), {77: q})
    assert r.kind == R.ANSWER and r.question == q
    assert R.route(_msg(words="A", reply_to_id=78), {77: q}).kind == R.LESSON    # a reply to something else


@pytest.mark.parametrize("caption", ["STOP", "START", "STATUS", "IDEA buy the dip", "status", ""])
def test_routing_a_photo_is_never_a_command_or_an_idea(caption):
    assert R.route(_msg(words=caption, photo_id="f1"), {}).kind == R.LESSON


def test_routing_an_edited_message_is_never_a_command():
    assert R.route(_msg(words="STOP", edited=True), {}).kind == R.LESSON


def test_routing_no_words_no_picture_is_unreadable():
    assert R.route(_msg(words="", other="voice note"), {}).kind == R.UNREADABLE


def test_parse_reads_caption_biggest_photo_image_file_and_reply(tg):
    u = tg.photo(b"x", "look here", reply_to=None)
    m = R.parse(u)
    assert (m.words, m.photo_id, m.chat_id) == ("look here", "file-1", TEJAS)
    m2 = R.parse(tg.photo(b"y", as_file=True))
    assert m2.photo_id == "file-2" and m2.words == ""
    assert R.parse({"update_id": 5, "channel_post": {}}) is None


# ============================================================================================ lessons

def test_a_lesson_lands_in_the_inbox_and_the_bot_says_where(bot, tg, where):
    u = tg.text("The gap must be the first one after the break.\nNot the second.")
    bot.poll_once()
    files = _inbox(where)
    assert [f.name for f in files] == [f"{STAMP}-telegram.md"]
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# Lesson (Telegram, ") and "— waiting for a number" in text.splitlines()[0]
    assert "> The gap must be the first one after the break.\n> Not the second." in text
    [reply] = tg.replies_to(u)
    assert "Saved as a lesson." in reply and f"{STAMP}-telegram.md" in reply
    assert bot.state["offset"] == u["update_id"] + 1


def test_two_lessons_in_one_minute_get_two_files(bot, tg, where):
    tg.text("one")
    tg.text("two")
    bot.poll_once()
    assert [f.name for f in _inbox(where)] == [f"{STAMP}-telegram-2.md", f"{STAMP}-telegram.md"]
    assert "> two" in (where.inbox / f"{STAMP}-telegram-2.md").read_text(encoding="utf-8")


def test_a_reply_to_a_plain_bot_line_is_a_lesson_that_quotes_it(bot, tg, where):
    _outbox(where, {"id": "b1", "text": "idea #1 is waiting for a block."})
    bot.send_outbox()
    u = tg.text("ok build it", reply_to=tg.sent[0]["message_id"])
    bot.poll_once()
    assert 'a reply to the bot\'s message: "idea #1 is waiting for a block."' in _inbox(where)[0].read_text("utf-8")
    assert "Saved as a lesson." in tg.replies_to(u)[0]


# ============================================================================================ photos

def test_photo_and_caption_make_both_copies_and_the_inbox_links_them(bot, tg, where):
    u = tg.photo(phone_screenshot(), "this is the gap I mean")
    bot.poll_once()
    [original] = list(where.photos.iterdir())
    [copy] = list(where.pictures.iterdir())
    assert original.name == f"{STAMP}-m{u['message']['message_id']}.jpg"
    assert copy.name == f"{STAMP}-telegram-m{u['message']['message_id']}-his-picture.jpg"
    text = _inbox(where)[0].read_text(encoding="utf-8")
    assert "> this is the gap I mean" in text
    assert inbox.show(copy) in text and inbox.show(original) in text and "top bar and buttons cut off" in text
    # the copy lost the red status bar and the blue buttons; the original kept them
    cut = Image.open(copy).convert("RGB")
    assert cut.size == (600, 1335 - round(1335 * 0.045) - round(1335 * 0.05))
    assert _white(cut.getpixel((300, 2))) and _white(cut.getpixel((300, cut.size[1] - 3)))
    assert Image.open(original).convert("RGB").getpixel((300, 2))[0] > 200 > Image.open(original).convert(
        "RGB").getpixel((300, 2))[2]
    assert "Picture:" in tg.replies_to(u)[0] and "with your picture" in tg.replies_to(u)[0]


def _white(px) -> bool:
    return all(c > 230 for c in px)


def test_a_wide_picture_is_copied_uncut(bot, tg, where):
    tg.photo(chart_picture(), "drawn on your chart", as_file=True)
    bot.poll_once()
    [copy] = list(where.pictures.iterdir())
    im = Image.open(copy).convert("RGB")
    assert im.size == (800, 500) and im.getpixel((400, 3))[0] > 200 > im.getpixel((400, 3))[2]
    assert "not a phone screenshot, so not cut" in _inbox(where)[0].read_text(encoding="utf-8")


def test_the_crop_uses_his_phone_measurements():
    im = Image.new("RGB", (1200, 2670))
    cut, was = inbox.crop_phone(im)
    assert was and cut.size == (1200, 2670 - 120 - 134)
    assert inbox.crop_phone(Image.new("RGB", (1000, 1600)))[1] is False      # not phone-shaped: left alone


@pytest.mark.parametrize("caption", ["STOP", "IDEA buy the second touch", "STATUS"])
def test_a_photo_never_triggers_an_action(bot, tg, where, caption):
    _outbox(where, {"id": "x1", "text": "a line for Tejas"})
    u = tg.photo(phone_screenshot(), caption)
    bot.poll_once()
    assert bot.state["sending"] is True                      # STOP in a caption did not stop anything
    assert not where.intake.exists()                         # IDEA in a caption started no idea
    assert len(_inbox(where)) == 1 and f"> {caption}" in _inbox(where)[0].read_text(encoding="utf-8")
    reply = tg.replies_to(u)[0]
    assert "Saved as a lesson" in reply and "A photo never starts a command or an idea" in reply
    assert not reply.startswith("STATUS")


def test_photos_sent_together_go_into_one_lesson(bot, tg, where):
    tg.photo(phone_screenshot(), "these two", group="album-7")
    u2 = tg.photo(chart_picture(), group="album-7")
    bot.poll_once()
    [lesson] = _inbox(where)
    assert lesson.read_text(encoding="utf-8").count("picture: `") == 2
    assert len(list(where.pictures.iterdir())) == 2
    assert "Added this picture to the same lesson." in tg.replies_to(u2)[0]


def test_a_photo_with_no_caption_is_still_a_lesson(bot, tg, where):
    tg.photo(phone_screenshot())
    bot.poll_once()
    assert "> (a picture, no words)" in _inbox(where)[0].read_text(encoding="utf-8")


# ============================================================================================ answers

def _question(bot, tg, where, qid="L6.2-Q1-lows-too-near", idea="idea-001") -> int:
    _outbox(where, {"id": f"q-{qid}", "text": "Would you take this sell? YES or NO", "qid": qid, "idea": idea})
    bot.send_outbox()
    return tg.sent[-1]["message_id"]


def test_a_reply_to_a_question_is_an_answer_with_the_picture_named_by_idea_and_question(bot, tg, where):
    answers = Path(config.IDEAS_DIR) / "idea-001" / "answers.md"          # tmp (conftest)
    answers.parent.mkdir(parents=True)
    answers.write_text("# Answers - idea-001\n", encoding="utf-8")
    qmid = _question(bot, tg, where)
    u = tg.photo(phone_screenshot(), "NO, the lows are too near", reply_to=qmid)
    bot.poll_once()
    [copy] = list(where.pictures.iterdir())
    assert copy.name == "idea-001-L6.2-Q1-lows-too-near-his-answer.jpg"
    text = _inbox(where)[0].read_text(encoding="utf-8")
    assert text.startswith("# Answer to question L6.2-Q1-lows-too-near, idea-001 (Telegram, ")
    assert "> NO, the lows are too near" in text and copy.name in text
    got = answers.read_text(encoding="utf-8")
    assert "L6.2-Q1-lows-too-near" in got and "> NO, the lows are too near" in got and copy.name in got
    reply = tg.replies_to(u)[0]
    assert "answer to question L6.2-Q1-lows-too-near (idea #1)" in reply and "Nothing changes until you say yes" in reply


def test_an_answer_in_test_mode_goes_to_the_test_folder(tg, tmp_path):
    where = inbox.Where.test(tmp_path / "telegram-test")
    where.outbox = tmp_path / "outbox.jsonl"
    bot = B.Bot(tg, TEJAS, where, echo=False)
    qmid = _question(bot, tg, where, qid="Q9", idea="idea-004")
    tg.text("B", reply_to=qmid)
    bot.poll_once()
    assert "> B" in (tmp_path / "telegram-test" / "answers" / "idea-004.md").read_text(encoding="utf-8")
    assert all(str(p).startswith(str(tmp_path / "telegram-test")) for p in (where.inbox, where.pictures, where.intake))


def test_where_test_never_points_at_his_real_files(tmp_path):
    w = inbox.Where.test(tmp_path / "t")
    for p in (w.inbox, w.pictures, w.bot, w.intake, w.answers):
        assert config.STRATEGY_DIR not in Path(p).parents and Path(config.WORK_DIR) not in Path(p).parents
    real = inbox.Where.real()
    assert real.inbox == config.INBOX_DIR and real.pictures == config.PICTURES_DIR and real.answers is None


# ============================================================================================ ideas and commands

def test_idea_goes_to_the_intake_queue(bot, tg, where):
    u = tg.text("IDEA sell when London takes the Asia high")
    bot.poll_once()
    [line] = [json.loads(x) for x in where.intake.read_text(encoding="utf-8").splitlines()]
    assert line["words"] == "sell when London takes the Asia high" and line["state"] == "RECEIVED"
    assert not _inbox(where)
    said = tg.replies_to(u)[0]
    assert "Saved as a new idea." in said and "The machine starts on it now, by itself." in said   # L10b runs it


def test_stop_pauses_the_lines_to_him_but_he_is_still_heard(bot, tg, where):
    tg.text("STOP")
    bot.poll_once()
    _outbox(where, {"id": "b1", "text": "a line for Tejas"})
    u = tg.text("a lesson while stopped")
    bot.poll_once()
    assert [s for s in tg.sent if s.get("reply_to") is None] == []            # nothing sent to him unasked
    assert len(_inbox(where)) == 1 and "Saved as a lesson." in tg.replies_to(u)[0]
    u2 = tg.text("START")
    bot.poll_once()
    assert "Sending again. 1 message waiting." == tg.replies_to(u2)[0]
    assert [s["text"] for s in tg.sent if s.get("reply_to") is None] == ["a line for Tejas"]
    assert bot.state["sending"] is True


def test_status_says_what_waits(bot, tg, where):
    tg.text("STOP")
    bot.poll_once()
    _outbox(where, {"id": "b1", "text": "x"})
    tg.text("a lesson")
    u = tg.text("STATUS")
    bot.poll_once()
    reply = tg.replies_to(u)[0]
    assert reply.startswith("STATUS") and "PAUSED" in reply and "Waiting to be sent: 1." in reply
    assert "waiting for a number: 1." in reply
    assert len(_inbox(where)) == 1                           # the command itself is not a lesson


def test_a_voice_note_is_not_filed_and_he_is_told(bot, tg, where):
    u = tg.voice()
    bot.poll_once()
    assert not _inbox(where) and "I can read only words and pictures" in tg.replies_to(u)[0]


# ============================================================================================ only his chat; never lost

def test_another_chat_is_ignored(bot, tg, where):
    u = tg.text("STOP", chat="999")
    bot.poll_once()
    assert not _inbox(where) and not tg.sent and bot.state["sending"] is True
    assert bot.state["offset"] == u["update_id"] + 1


def test_a_message_given_twice_is_filed_once(bot, tg, where):
    u = tg.text("once only")
    bot.handle(u)
    bot.state["offset"] = 0                                    # as if the bot crashed before saving the offset
    bot.handle(u)
    assert len(_inbox(where)) == 1 and len(tg.replies_to(u)) == 1


def test_the_offset_survives_a_restart(tg, where):
    b1 = B.Bot(tg, TEJAS, where, echo=False)
    u = tg.text("first")
    b1.poll_once()
    b2 = B.Bot(tg, TEJAS, where, echo=False)
    assert b2.state["offset"] == u["update_id"] + 1 and u["update_id"] in b2.done


def test_a_bug_keeps_the_message_whole_and_moves_on(bot, tg, where, monkeypatch):
    monkeypatch.setattr(inbox, "write_inbox", lambda *a, **k: 1 / 0)
    u = tg.text("this will hit a bug")
    bot.poll_once()
    kept = where.bot / "failed" / f"{u['update_id']}.json"
    assert json.loads(kept.read_text(encoding="utf-8"))["update"] == u
    assert bot.state["offset"] == u["update_id"] + 1
    assert "I could not file this message" in tg.replies_to(u)[0]


# ============================================================================================ lines TO him

def test_outbox_lines_are_sent_once_and_recorded(bot, tg, where):
    _outbox(where, {"id": "2026-09-28-idea-001-block-x", "text": "idea #1 is waiting.", "to": "Tejas"},
            {"id": "b2", "text": "second"})
    assert bot.send_outbox() == 2
    assert bot.send_outbox() == 0
    sent = [json.loads(x) for x in (where.bot / "sent.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [s["id"] for s in sent] == ["2026-09-28-idea-001-block-x", "b2"]
    assert [s["text"] for s in tg.sent] == ["idea #1 is waiting.", "second"]
    _outbox(where, {"id": "b3", "text": "third"})
    bot.send_outbox()
    assert [s["text"] for s in tg.sent][-1] == "third" and len(tg.sent) == 3


def test_a_bad_outbox_line_is_skipped_not_sent(bot, tg, where):
    where.outbox.parent.mkdir(parents=True)
    where.outbox.write_text('not json\n{"text": "no id"}\n{"id": "ok", "text": "fine"}\n', encoding="utf-8")
    assert bot.send_outbox() == 1 and [s["text"] for s in tg.sent] == ["fine"]


def test_an_outbox_line_with_a_picture_is_sent_as_a_photo(bot, tg, where, tmp_path):
    pic = tmp_path / "q.jpg"
    pic.write_bytes(chart_picture())
    _outbox(where, {"id": "q1", "text": "Would you take it?", "photo": str(pic), "qid": "Q1", "idea": "idea-001"})
    bot.send_outbox()
    assert tg.sent[0]["photo"] == str(pic) and tg.sent[0]["caption"] == "Would you take it?"
    qs = bot.questions()
    assert list(qs) == [tg.sent[0]["message_id"]] and (qs[tg.sent[0]["message_id"]]["qid"], qs[tg.sent[0][
        "message_id"]]["idea"]) == ("Q1", "idea-001")


def test_the_real_l7b_outbox_line_shape_is_read(bot, where):
    """plan/work/telegram-outbox.jsonl is written by machine/ideas/blocked.py (L7b): id, at, to, text, why."""
    _outbox(where, {"id": "2026-09-28-idea-001-block-trendline", "at": "2026-09-28T10:00:00+05:30", "to": "Tejas",
                    "text": "idea #1 (x) is waiting.", "why": "block request (L7b)"})
    assert [ln["id"] for ln in bot.unsent()] == ["2026-09-28-idea-001-block-trendline"]


# ============================================================================================ running it

def test_a_second_poller_stops_the_bot(bot, tg):
    tg.fail = "conflict"
    with pytest.raises(SystemExit, match="only ONE"):
        B.run(bot, sleep=lambda s: None)


def test_a_network_error_is_retried(bot, tg, where):
    tg.fail = "network"
    waits = []

    def sleep(s):
        waits.append(s)
        tg.text("after the network came back")
        raise KeyboardInterrupt                               # stop the loop after one retry

    with pytest.raises(KeyboardInterrupt):
        B.run(bot, sleep=sleep)
    assert waits == [5]
    bot.poll_once()
    assert len(_inbox(where)) == 1


def test_only_one_bot_per_pc(tmp_path):
    lock = tmp_path / "bot.lock"
    with B.only_one(lock):
        with pytest.raises(SystemExit, match="Only ONE"):
            with B.only_one(lock):
                pass
    with B.only_one(lock):                                    # free again once the first one stopped
        pass


def test_the_key_never_shows_in_an_error(monkeypatch):
    key = "123456:SECRET-KEY-abc"

    def boom(req, timeout=0):
        raise urllib.error.URLError(f"cannot reach {req.full_url}")

    monkeypatch.setattr(A.urllib.request, "urlopen", boom)
    with pytest.raises(A.TelegramError) as err:
        A.TelegramAPI(key).send_message("1", "hi")
    assert key not in str(err.value) and "<key>" in str(err.value) and err.value.kind == "network"


def test_copy_old_keys_never_shows_them(tmp_path):
    src = tmp_path / "old" / "ml" / "agent" / "agent_config.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps({"telegram_token": "999:TOKEN-VALUE", "telegram_chat_id": "424242"}), encoding="utf-8")
    said = B.copy_old_keys(tmp_path)
    assert not any("TOKEN-VALUE" in s or "424242" in s for s in said)
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN=999:TOKEN-VALUE" in env and "TELEGRAM_CHAT_ID=424242" in env
    assert all("already in .env" in s for s in B.copy_old_keys(tmp_path))
    assert (tmp_path / ".env").read_text(encoding="utf-8") == env


def test_the_bot_needs_a_chat_id(tg, where):
    with pytest.raises(ValueError):
        B.Bot(tg, "", where)
