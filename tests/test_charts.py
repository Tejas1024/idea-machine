"""L4: the pictures and their lint - every rule of his passes once and FAILS once.

Hand-made bars (tests/chart_fakes.py), so these run everywhere, CI included. Wherever it can, a failure is made by
really drawing the mistake (a label that lands on the candles, a word naming a thing not drawn, a PNG saved as .jpg),
not by editing the record by hand.
"""
from __future__ import annotations

import dataclasses
import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from machine.charts import draw as D
from machine.charts import lint

from tests import chart_fakes as F


@pytest.fixture(scope="module")
def bars():
    m5 = F.bars_5m()
    return m5, F.to_1h(m5)


@pytest.fixture
def shown(tmp_path):
    return tmp_path / "pictures-shown.jsonl"


def names(problems):
    return {p.rule for p in problems}


def fails(problems, rule_name):
    assert rule_name in names(problems), f"expected [{rule_name}] to fail, got: {problems}"
    return [str(p) for p in problems if p.rule == rule_name]


# ---- every rule has a test both ways ------------------------------------------------------------------------------

def test_every_rule_is_listed_with_his_words():
    assert set(lint.RULES) == {
        "label_overlap", "words_fit", "named_not_drawn", "banned_words", "his_labels", "question_on_picture",
        "no_money_in_question", "ends_at_entry", "never_twice", "real_file_type", "title_timeframe_clock",
        "every_point_own_panel", "parse_math_off"}
    for name, (_fn, his) in lint.RULES.items():
        assert "#" in his or "lesson" in his, name


# ---- the good pictures pass every rule ------------------------------------------------------------------------------

@pytest.mark.parametrize("make, suffix", [(F.good_trade_picture, ".jpg"), (F.good_question_picture, ".jpg"),
                                          (F.good_whole_trade_picture, ".png")])
def test_good_pictures_pass_every_rule(bars, tmp_path, shown, make, suffix):
    drawn = D.render(make(*bars), tmp_path / f"good{suffix}")
    assert lint.check(drawn, shown=shown) == []


def test_good_arrows_picture_passes(bars, tmp_path, shown):
    m5, h1 = bars
    t = F.fake_trade(m5, 180, 330, "buy")
    pic = D.arrows_picture(t, h1, subject="arrows 1", sub="a test", min_move=1.5, min_hour_bars=30,
                           items=[D.Box(m5.index[170], m5.index[180], t.fill_px - 0.7, t.fill_px, named=False),
                                  D.Level(t.exits[0].px, "target", start=m5.index[180])])
    drawn = D.render(pic, tmp_path / "arrows.png")
    assert lint.check(drawn, shown=shown) == []
    assert not any(i.kind == "Candles" for i in drawn.items) and drawn.panels[0].candles == []   # #122: no candles


# ---- each rule fails once ---------------------------------------------------------------------------------------------

def test_label_overlap_fails_when_a_name_lands_on_the_candles(bars, tmp_path, shown):
    m5, h1 = bars
    pic = F.good_trade_picture(m5, h1)
    main = pic.rows[1][0]
    mid = float(main.bars["close"].iloc[20])
    # a short line in the middle of the candles: its name goes just above its right end - on the price action
    main.items.append(D.Level(mid, "the price it broke", start=main.bars.index[5], end=main.bars.index[30]))
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "label_overlap")
    assert any("the price it broke" in m and "candles" in m for m in msgs)


def test_label_overlap_fails_when_a_word_covers_the_gap(bars, tmp_path, shown):
    drawn = D.render(F.good_trade_picture(*bars), tmp_path / "x.jpg")
    gap = next(i for i in drawn.items if i.kind == "Box")
    tag = next(t for t in drawn.texts if t.role == "tag" and t.panel == gap.panel)
    moved = dataclasses.replace(tag, box=gap.boxes[0])       # the S1 label put ON the gap box (#110)
    drawn.texts[drawn.texts.index(tag)] = moved
    assert any("covers the gap" in m for m in fails(lint.check(drawn, shown=shown), "label_overlap"))


def test_words_fit_fails_on_a_line_that_runs_off_the_edge(bars, tmp_path, shown):
    pic = F.good_trade_picture(*bars)
    pic.sub = "11 Mar 2019 · " + "a very long sentence that nobody will ever finish reading on a phone " * 3
    fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "words_fit")


def test_named_not_drawn_fails_on_a_zone_that_is_not_drawn(bars, tmp_path, shown):
    pic = F.good_trade_picture(*bars)
    pic.sub = "11 Mar 2019 · price came back to the strong zone"
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "named_not_drawn")
    assert any("zone" in m for m in msgs)


def test_named_not_drawn_fails_on_a_triangle_without_the_price_inside(bars, tmp_path, shown):
    m5, h1 = bars
    pic = F.good_trade_picture(m5, h1)
    main = pic.rows[1][0]
    b = main.bars
    top, low = float(b["high"].max()), float(b["low"].min())
    # two lines far BELOW every candle: they converge, but the price is not inside them (#116)
    main.items += [D.Line((b.index[5], low - 1.0), (b.index[60], low - 2.0), touches=(), group="triangle",
                          name="triangle"),
                   D.Line((b.index[5], low - 5.0), (b.index[60], low - 3.0), touches=(), group="triangle",
                          name="triangle")]
    assert top > low
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "named_not_drawn")
    assert any("triangle" in m and "inside" in m for m in msgs)


def test_named_not_drawn_passes_a_triangle_with_the_price_inside(bars, tmp_path, shown):
    m5, h1 = bars
    pic = F.good_trade_picture(m5, h1)
    main = pic.rows[1][0]
    b = main.bars.iloc[10:50]
    hi, lo = float(b["high"].max()), float(b["low"].min())
    main.items += [D.Line((b.index[0], hi + 0.2), (b.index[-1], hi + 0.1), group="triangle", name="triangle"),
                   D.Line((b.index[0], lo - 0.2), (b.index[-1], lo - 0.1), group="triangle", name="triangle")]
    problems = lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown)
    assert "triangle" not in " ".join(str(p) for p in problems if p.rule == "named_not_drawn")


def test_banned_words_fails(bars, tmp_path, shown):
    pic = F.good_trade_picture(*bars)
    pic.sub = "the leftover leg ran 3R"
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "banned_words")
    assert any("leftover" in m for m in msgs) and any("'R'" in m for m in msgs)


def test_his_labels_fail_on_long_names_money_and_hedge(bars, tmp_path, shown):
    m5, h1 = bars
    pic = F.good_trade_picture(m5, h1)
    main = pic.rows[1][0]
    t = main.bars.index[40]
    px = float(main.bars["close"].iloc[40])
    main.items += [D.Marker(t, px, "BUY1", "buy", "hedge"), D.Marker(main.bars.index[60], px, "S2", "sell", "+₹500")]
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "his_labels")
    assert any("BUY1" in m for m in msgs) and any("hedge" in m for m in msgs) and any("money" in m for m in msgs)


def test_question_on_picture_fails_without_a_real_question(bars, tmp_path, shown):
    pic = F.good_question_picture(*bars)
    pic.question = D.Question("Look at this", ())
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "question_on_picture")
    assert any("'?'" in m for m in msgs) and any("how to answer" in m for m in msgs)


def test_no_money_in_question_fails(bars, tmp_path, shown):
    pic = F.good_question_picture(*bars)
    pic.sub = "11 Mar 2019 · this made ₹500 profit"
    fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "no_money_in_question")


def test_ends_at_entry_fails_when_the_chart_shows_the_future(bars, tmp_path, shown):
    m5, h1 = bars
    pic = F.good_question_picture(m5, h1)
    main = pic.rows[1][0]
    entry = main.bars.index[-1]
    main.bars = m5.loc[main.bars.index[0]:entry + pd.Timedelta(minutes=30)]      # 6 candles AFTER the entry
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "ends_at_entry")
    assert any("after the entry candle closed" in m for m in msgs)


def test_ends_at_entry_fails_when_a_line_reaches_past_the_entry(bars, tmp_path, shown):
    pic = F.good_question_picture(*bars)
    main = pic.rows[1][0]
    line = next(i for i in main.items if isinstance(i, D.Line))
    main.items[main.items.index(line)] = dataclasses.replace(
        line, extend_to=pd.Timestamp(line.extend_to) + pd.Timedelta(hours=2))
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "ends_at_entry")
    assert any("equal highs" in m for m in msgs)


def test_question_picture_refuses_an_entry_that_is_not_a_candle(bars):
    m5, h1 = bars
    with pytest.raises(ValueError, match="not a 5-minute candle"):
        D.question_picture(m5, h1, entry_at=m5.index[100] + pd.Timedelta(minutes=2),
                           question=D.Question("Is it?", ("YES or NO",)), subject="x", heading="Q", sub="")


def test_never_twice_fails_on_the_same_picture_and_the_same_trade(bars, tmp_path, shown):
    drawn = D.render(F.good_trade_picture(*bars), tmp_path / "a.jpg")
    assert lint.check(drawn, shown=shown) == []
    lint.mark_shown(drawn, "test", shown=shown)
    assert json.loads(shown.read_text(encoding="utf-8").splitlines()[0])["subject"] == drawn.subject
    msgs = fails(lint.check(drawn, shown=shown), "never_twice")
    assert any("exact picture" in m for m in msgs)
    # the same trade drawn differently: refused, unless there is a reason to show it again
    pic = F.good_trade_picture(*bars)
    pic.sub = "11 Mar 2019 · the same trade, drawn again"
    again = D.render(pic, tmp_path / "b.jpg")
    assert any("already shown" in m for m in fails(lint.check(again, shown=shown), "never_twice"))
    assert lint.check(again, shown=shown, again_because="he asked to see it fixed") == []


def test_real_file_type_fails_on_a_png_saved_as_jpg(bars, tmp_path, shown):
    drawn = D.render(F.good_trade_picture(*bars), tmp_path / "a.jpg")
    assert drawn.path.read_bytes()[:3] == b"\xff\xd8\xff"                            # #106: a real JPEG
    D.render(F.good_trade_picture(*bars), tmp_path / "b.png")
    drawn.path.write_bytes((tmp_path / "b.png").read_bytes())                       # the old bug: PNG bytes, .jpg name
    assert any("really PNG" in m for m in fails(lint.check(drawn, shown=shown), "real_file_type"))


def test_title_timeframe_clock_fails_when_the_chart_is_not_what_it_says(bars, tmp_path, shown):
    m5, h1 = bars
    pic = F.good_trade_picture(m5, h1)
    pic.rows[0][0].tf = "4h"                     # an hour chart titled "4-hour"
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "title_timeframe_clock")
    assert any("60 minutes apart" in m for m in msgs)
    drawn = D.render(F.good_trade_picture(m5, h1), tmp_path / "y.jpg")
    drawn.panels[1].title = "the trade"         # no timeframe, no clock
    assert len(fails(lint.check(drawn, shown=shown), "title_timeframe_clock")) == 2


def test_every_point_own_panel_fails_when_an_exit_has_no_close_up(bars, tmp_path, shown):
    pic = F.good_whole_trade_picture(*bars)
    pic.rows[1] = pic.rows[1][:1]                # keep B1's 5-minute panel, drop B1 C's
    msgs = fails(lint.check(D.render(pic, tmp_path / "x.jpg"), shown=shown), "every_point_own_panel")
    assert any("B1 C" in m for m in msgs)


def test_parse_math_is_off_even_if_someone_turned_it_on(bars, tmp_path, shown):
    old = plt.rcParams["text.parse_math"]
    plt.rcParams["text.parse_math"] = True
    try:
        pic = F.good_trade_picture(*bars)
        pic.sub = "gold moved $5 and then $6"
        drawn = D.render(pic, tmp_path / "x.jpg")
    finally:
        plt.rcParams["text.parse_math"] = old
    assert drawn.parse_math is False
    assert "parse_math_off" not in names(lint.check(drawn, shown=shown))
    drawn.parse_math = True
    fails(lint.check(drawn, shown=shown), "parse_math_off")


# ---- the drawing itself -----------------------------------------------------------------------------------------------

def test_his_shorthand_from_a_trade(bars):
    m5, _ = bars
    pts, positions = D.points_from_trade(F.fake_trade(m5, 10, 20, "sell"))
    assert [(p.tag, p.act, p.word) for p in pts] == [("S1", "in", "gap"), ("S1 C", "out", "target")]
    assert positions[0].side == "sell"


def test_items_refuse_a_time_with_a_time_zone(bars, tmp_path):
    m5, h1 = bars
    pic = F.good_trade_picture(m5, h1)
    pic.rows[1][0].items.append(D.Marker(pd.Timestamp("2019-03-11 10:00", tz="UTC"), 1290.0, "B2", "buy"))
    with pytest.raises(ValueError, match="time zone"):
        D.render(pic, tmp_path / "x.jpg")


def test_the_gap_box_and_every_word_are_recorded_where_they_were_drawn(bars, tmp_path):
    drawn = D.render(F.good_trade_picture(*bars), tmp_path / "x.jpg")
    gap = next(i for i in drawn.items if i.kind == "Box")
    main = drawn.panels[1]
    assert gap.visible and main.area.x0 <= gap.boxes[0].x0 < gap.boxes[0].x1 <= main.area.x1
    words = {t.text for t in drawn.texts}
    assert {"B1\ngap", "B1 C\ntarget", "gap 70 cents", "target", "stop", "ADR top", "ADR bottom"} <= words
    assert drawn.size == (1560, drawn.size[1]) and drawn.pixel_sha256 == D.pixel_sha256(drawn.path)
