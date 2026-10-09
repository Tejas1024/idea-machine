"""L15.2 - his eyes: the two models of his zone answers and how they are measured (machine/eyes/).

No test reads his real answers: the label file is made up here (or the real price data with made-up charts), the
eyes folders are temporary (conftest), and model B learns on tiny made-up pictures without the ImageNet download.
"""
import io
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from machine import config
from machine.eyes import features as FE
from machine.eyes import labels as LB
from machine.eyes import measure as M
from machine.eyes import model_a, split
from machine.eyes import plan as PL
from machine.eyes import report as RP
from machine.eyes import run as RUN


# ---- made-up answers -----------------------------------------------------------------------------------------------

def jpeg(w=1560, h=1191, colour=(255, 255, 255)) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(b, "JPEG")
    return b.getvalue()


def fake_labels(per_year=12, seed=0, years=(2016, 2017, 2018, 2019, 2020)) -> pd.DataFrame:
    """A label table like the real one: charts of 5 days, spread over each year, yes about 1 in 3."""
    rng = np.random.default_rng(seed)
    rows, pic = [], jpeg()
    for y in years:
        step = 350 // per_year
        for k in range(per_year):
            first = pd.Timestamp(f"{y}-01-08") + pd.Timedelta(days=int(k * step + rng.integers(0, max(1, step - 6))))
            calm = float(rng.uniform(0.4, 1.6))
            rows.append(dict(label_id=f"tg-{y}-{k:02d}", source="telegram", label="yes" if calm <= 0.74 else "no",
                             his_words="", first_candle=first, last_candle=first + pd.Timedelta(days=4, hours=22),
                             known_utc=first + pd.Timedelta(days=5), year=y, window="week", lo=1300.0, hi=1302.0,
                             calm_on_chart=calm, rule_says="yes" if calm <= 0.74 else "no", step=0, step_size=10,
                             image=pic, image_sha256=f"sha-{y}-{k}"))
    return pd.DataFrame(rows)


# ---- the measures ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k,n", [(84, 120), (90, 120), (7, 10), (1, 39), (120, 120), (0, 30)])
def test_wilson_matches_scipy(k, n):
    want = stats.binomtest(k, n).proportion_ci(confidence_level=0.90, method="wilson")
    lo, hi = M.wilson(k, n, 0.90)
    assert lo == pytest.approx(want.low, abs=1e-9) and hi == pytest.approx(want.high, abs=1e-9)


def test_wilson_edges_and_width():
    assert M.wilson(0, 0) != M.wilson(0, 0)                   # nan: no answers, no interval
    assert M.wilson(0, 50)[0] == 0.0 and M.wilson(50, 50)[1] == 1.0
    assert M.half_width(120) > M.half_width(400) > 0          # more answers, a narrower interval
    assert 0.06 < M.half_width(120) < 0.075                   # the "about 7 points" said in the report


@pytest.mark.parametrize("only_a,only_b", [(10, 2), (3, 3), (0, 7), (0, 0), (25, 14)])
def test_pair_is_mcnemar_exact(only_a, only_b):
    a = np.array([True] * only_a + [False] * only_b + [True] * 5 + [False] * 4)
    b = np.array([False] * only_a + [True] * only_b + [True] * 5 + [False] * 4)
    got = M.pair(a, b)
    assert (got["only_a"], got["only_b"]) == (only_a, only_b)
    want = 1.0 if only_a + only_b == 0 else stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    assert got["p"] == pytest.approx(want)
    assert got["beyond_luck"] == (want < 0.10)


def test_ranks_higher_counts_pairs():
    y = np.array([1, 1, 0, 0, 0])
    assert M.ranks_higher(y, np.array([0.9, 0.8, 0.1, 0.2, 0.3])) == 1.0
    assert M.ranks_higher(y, np.array([0.1, 0.2, 0.9, 0.8, 0.7])) == 0.0
    assert M.ranks_higher(y, np.full(5, 0.5)) == 0.5            # ties count half
    s = np.array([0.6, 0.2, 0.4, 0.2, 0.1])
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in s[y == 1] for b in s[y == 0]])
    assert M.ranks_higher(y, s) == pytest.approx(brute)
    lo, hi = M.ranks_interval(y, s, runs=200)
    assert 0 <= lo <= hi <= 1


def test_score_table_and_years():
    y = np.array([1, 1, 0, 0, 0, 1])
    said = np.array([1, 0, 0, 1, 0, 1])
    s = M.score("x", y, said, said.astype(float), np.array([2018, 2018, 2019, 2019, 2020, 2020]))
    assert s.right == 4 and s.n == 6
    assert s.table == {"he yes, it yes": 2, "he yes, it no": 1, "he no, it yes": 1, "he no, it no": 2}
    assert (s.yes_found, s.yes_all, s.no_found, s.no_all) == (2, 3, 2, 3)
    assert s.by_year == {2018: [1, 2], 2019: [1, 2], 2020: [2, 2]}


# ---- model A's numbers -----------------------------------------------------------------------------------------------

def candles(rows) -> pd.DataFrame:
    idx = pd.date_range("2019-03-04 01:00", periods=len(rows), freq="1h")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def test_features_of_a_hand_made_chart():
    # 8 hour candles; the band 100-101. Candles 2, 3 sit in it (small); candle 5 only pokes it with a wick.
    h = candles([[96, 98, 95, 97], [97, 100, 96, 99.5], [100.2, 100.6, 99.9, 100.4], [100.4, 100.8, 100.1, 100.5],
                 [101.5, 104, 101.4, 103], [103, 103.5, 100.8, 102], [102, 106, 101.8, 105], [105, 109, 104, 108]])
    f = FE.of_chart(h, 100.0, 101.0)
    rng = (h["high"] - h["low"]).to_numpy()
    assert f["calm"] == pytest.approx(np.median(rng[[2, 3]]) / np.median(rng))
    assert f["time_in_band"] == pytest.approx(2 / 8)
    assert f["longest_stay"] == pytest.approx(2 / 8) and f["stays"] == 1
    assert f["wick_touches"] == pytest.approx(2 / 8)               # candle 1 reaches 100, candle 5 dips to 100.8
    assert f["band_when"] == pytest.approx(np.mean([2, 3]) / 7)
    span = 109 - 95
    assert f["band_of_chart"] == pytest.approx(1 / span) and f["band_place"] == pytest.approx((100.5 - 95) / span)
    assert f["left_by"] == pytest.approx((109 - 101) / span)       # after its last stay price went up to 109
    assert f["trend"] == pytest.approx((108 - 96) / span)
    assert set(f) == set(FE.COLUMNS)


def test_features_two_stays_and_no_body_in_band():
    h = candles([[100.2, 100.5, 100, 100.3], [101, 103, 100.9, 102.5], [100.4, 100.6, 100.1, 100.5],
                 [103, 104, 102, 103.5]])
    f = FE.of_chart(h, 100.0, 100.8)
    assert f["stays"] == 2 and f["longest_stay"] == pytest.approx(1 / 4)
    g = FE.of_chart(h, 110.0, 111.0)                               # a band no candle reaches
    assert np.isnan(g["calm"]) and np.isnan(g["band_when"]) and g["time_in_band"] == 0


def test_feature_calm_is_the_label_files_calm():
    from machine.labels import picture as P
    rng = np.random.default_rng(3)
    for _ in range(20):
        o = 1300 + np.cumsum(rng.normal(0, 1, 60))
        c = o + rng.normal(0, 1, 60)
        h = candles(np.c_[o, np.maximum(o, c) + rng.uniform(0, 1, 60), np.minimum(o, c) - rng.uniform(0, 1, 60), c])
        lo = float(np.quantile(c, 0.4))
        assert FE.of_chart(h, lo, lo + 1.5)["calm"] == pytest.approx(P.calm(h, lo, lo + 1.5), nan_ok=True)


# ---- the split by time -----------------------------------------------------------------------------------------------

def test_folds_learn_only_from_before_with_a_gap():
    lab = LB.load(df=fake_labels())
    # a chart that runs over the new year into 2019, and one that ends 3 days before 2020's first chart
    first19 = lab.loc[lab["year"] == 2019, "first_candle"].min()
    first20 = lab.loc[lab["year"] == 2020, "first_candle"].min()
    extra = fake_labels(per_year=2, seed=9, years=(2018,)).iloc[:2].copy()
    extra["label_id"] = ["tg-over-new-year", "tg-close-to-2020"]
    extra["first_candle"] = [first19 - pd.Timedelta(days=3), first20 - pd.Timedelta(days=8)]
    extra["last_candle"] = [first19 + pd.Timedelta(days=2), first20 - pd.Timedelta(days=3)]
    extra["year"] = [int(x.year) for x in extra["first_candle"]]
    lab = LB.load(df=pd.concat([fake_labels(), extra], ignore_index=True))
    fs = split.folds(lab)
    assert [f.year for f in fs] == [2018, 2019, 2020]
    for f in fs:
        test_first = lab["first_candle"].iloc[f.test].min()
        assert (lab["first_candle"].iloc[f.test].dt.year == f.year).all()
        assert (lab["last_candle"].iloc[f.train] < test_first - pd.Timedelta(days=7)).all()
        assert not set(f.train) & set(f.test)
    ids = lambda f, part: set(lab["label_id"].iloc[getattr(f, part)])       # noqa: E731
    assert "tg-over-new-year" not in ids(fs[1], "train")         # it ends inside 2019's first week: never taught 2019
    assert "tg-over-new-year" in ids(fs[2], "train")
    assert "tg-close-to-2020" not in ids(fs[2], "train")         # 3 days before 2020's first chart: purged
    tested = split.tested(fs)
    assert set(lab["label_id"].iloc[tested]) == set(lab.loc[lab["first_candle"].dt.year >= 2018, "label_id"])


def test_folds_refuse_an_empty_year():
    with pytest.raises(ValueError, match="no chart starts in 2018"):
        split.folds(LB.load(df=fake_labels(years=(2016, 2017, 2019, 2020))))


# ---- the answers -----------------------------------------------------------------------------------------------------

def test_load_keeps_pictured_yes_no_only_and_refuses_bad_rows():
    df = fake_labels(per_year=3)
    hidden = df.iloc[:1].copy().assign(label_id="his-Q79-A", image=None, year=2022, label="no",
                                       first_candle=pd.Timestamp("2021-12-29"), last_candle=pd.Timestamp("2022-01-05"))
    unsure = df.iloc[:1].copy().assign(label_id="tg-unsure", label="not sure")
    lab = LB.load(df=pd.concat([df, hidden, unsure], ignore_index=True))
    assert len(lab) == len(df) and set(lab["label"]) <= {"yes", "no"}
    assert (lab["y"] == (lab["label"] == "yes")).all()
    assert lab["first_candle"].is_monotonic_increasing
    assert LB.left_out(df=pd.concat([df, hidden, unsure]))["no_picture"] == 1
    with pytest.raises(LB.LabelsError, match="2016-2020 only"):
        LB.load(df=pd.concat([df, hidden.assign(image=jpeg())], ignore_index=True))
    with pytest.raises(LB.LabelsError, match="twice"):
        LB.load(df=pd.concat([df, df.iloc[:1]], ignore_index=True))


def test_made_up_answers():
    lab = LB.load(df=fake_labels(per_year=20))
    for seed in range(3):
        y = LB.made_up(lab, "noise", seed=seed)
        for yr in lab["year"].unique():                     # the same yes count in every year, only moved
            m = lab["year"].to_numpy() == yr
            assert y[m].sum() == lab["y"].to_numpy()[m].sum()
    assert (LB.made_up(lab, "rule") == lab["rule_yes"].to_numpy()).all()
    assert not np.array_equal(LB.made_up(lab, "noise", 0), LB.made_up(lab, "noise", 1))
    assert LB.fingerprint(lab) == LB.fingerprint(lab.copy())
    assert LB.fingerprint(lab) != LB.fingerprint(lab.assign(label=lab["label"].iloc[::-1].to_numpy()))


# ---- model A ---------------------------------------------------------------------------------------------------------

def fake_data(per_year=24, seed=0, planted=True):
    lab = LB.load(df=fake_labels(per_year=per_year, seed=seed))
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(len(lab), len(FE.COLUMNS))), columns=list(FE.COLUMNS),
                     index=lab["label_id"])
    if planted:
        X["calm"] = lab["calm_on_chart"].to_numpy()
    return RUN.Data(lab, X, np.zeros((len(lab), 1, 1, 3), np.uint8))


def test_model_a_finds_a_planted_answer_and_not_noise():
    d = fake_data()
    r = RUN.measure(d, d.lab["rule_yes"].to_numpy(), models=("model A",))
    assert r["scores"]["model A"]["accuracy"] >= 0.90             # small trees on ~50-100 answers: a few near 0.74
    assert next(iter(r["model A uses"])) == "calm"
    assert all(int(k) == y for k, y in zip(r["scores"]["model A"]["by_year"], (2018, 2019, 2020)))
    noise = fake_data(planted=False)
    accs = [RUN.measure(noise, LB.made_up(noise.lab, "noise", s), models=("model A",))["scores"]["model A"]
            ["accuracy"] for s in range(5)]
    base = 1 - noise.lab.loc[noise.lab["year"] >= 2018, "y"].mean()
    assert np.mean(accs) < base + 0.08                        # nothing to find: about "always no" or below


def test_measure_result_shape():
    d = fake_data()
    r = RUN.measure(d, d.lab["y"].to_numpy(), models=("model A",))
    assert r["n"] == 72 and r["years"] == [2018, 2019, 2020]
    assert set(r["scores"]) == {"always no", "his rule", "model A"}
    assert r["scores"]["always no"]["yes_found"] == 0
    assert r["winner"]["name"] is None                        # one model only: no winner named
    assert {"always no vs model A", "his rule vs model A"} <= set(r["pairs"])
    assert len(r["guesses"]) == 72 and set(r["guesses"]["his"]) <= {0, 1}


def test_winner_rule():
    def sc(right):
        return M.Score("x", 100, right, right / 100, 0, 1, 0, 0, 0, 0, {})
    pairs = {"model A vs model B": {"only_a": 9, "only_b": 3, "p": 0.146, "beyond_luck": False}}
    w = RUN.winner({"model A": sc(80), "model B": sc(74)}, pairs)
    assert w["name"] == "model A" and w["by"] == 6 and w["beyond_luck"] is False
    assert RUN.winner({"model A": sc(77), "model B": sc(77)}, pairs)["name"] is None


# ---- model B (PyTorch) - tiny made-up pictures, no download ------------------------------------------------------

def tiny_pictures(n=16, seed=0):
    """Left-dark or right-dark 32 x 64 pictures: an easy thing to see."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    imgs = np.full((n, 32, 64, 3), 255, np.uint8)
    for i, v in enumerate(y):
        x0 = 8 if v else 40
        imgs[i, 8:24, x0:x0 + 16] = rng.integers(0, 60)
    return imgs, y


def test_model_b_learns_runs_the_same_twice_and_keeps_frozen_layers():
    pytest.importorskip("torchvision")
    from machine.eyes import model_b
    imgs, y = tiny_pictures()
    quick = {"epochs": 3, "batch": 8}
    m1 = model_b.fit(imgs, y, 0, pretrained=False, settings=quick, threads=1)
    m2 = model_b.fit(imgs, y, 0, pretrained=False, settings=quick, threads=1)
    p1, p2 = model_b.predict(m1, imgs), model_b.predict(m2, imgs)
    assert p1.shape == (16,) and ((p1 >= 0) & (p1 <= 1)).all()
    assert np.allclose(p1, p2, atol=1e-5)                    # the same start = the same answers
    fresh, frozen = model_b.net(pretrained=False)
    names = [n for n, p in m1.named_parameters() if not p.requires_grad]
    assert names and all(n.split(".")[0] in ("conv1", "bn1", "layer1", "layer2") for n in names)
    trained = model_b.fit(imgs, y, 0, pretrained=False, settings={**quick, "epochs": 1}, threads=1)
    import torch
    torch.manual_seed(0)
    again, _ = model_b.net(pretrained=False)                  # the same random start as fit's
    for (n, a), (_, b) in zip(trained.state_dict().items(), again.state_dict().items()):
        if n.split(".")[0] in ("conv1", "layer1", "layer2") and "num_batches" not in n:
            assert torch.equal(a, b), f"{n} changed although it is frozen"


def test_model_b_walk_averages_its_starts():
    pytest.importorskip("torchvision")
    from machine.eyes import model_b
    d = fake_data(per_year=8)
    imgs, _ = tiny_pictures(len(d.lab))
    folds = split.folds(d.lab)
    p, by = model_b.walk(imgs, d.lab["y"].to_numpy(), folds, seeds=(0, 1), pretrained=False,
                         settings={"epochs": 1, "batch": 8})
    rows = split.tested(folds)
    assert np.isnan(np.delete(p, rows)).all() and np.isfinite(p[rows]).all()
    assert np.allclose(p[rows], (by[0][rows] + by[1][rows]) / 2)


def test_picture_crop_and_size():
    from machine.eyes import model_b
    a = model_b.pixels(jpeg())
    assert a.shape == (*config.EYES_IMAGE_HW, 3) and a.dtype == np.uint8
    with pytest.raises(ValueError, match="1560"):
        model_b.pixels(jpeg(800, 600))
    left, top, right, bottom = config.EYES_CROP
    assert 0 <= left < right <= 1560 and 0 <= top < bottom <= 1191


# ---- the plan comes first --------------------------------------------------------------------------------------------

class FakeGit:
    def __init__(self, tracked=True, same=True, clean=True, pushed=True):
        self.tracked, self.same, self.clean, self.pushed = tracked, same, clean, pushed

    def __call__(self, *args, ok_codes=(0,)):
        if args[0] == "ls-files":
            return args[-1] if self.tracked else None
        if args[0] == "diff":
            return "" if self.same else None
        if args[0] == "status":
            return "" if self.clean else " M machine/eyes/run.py\n"
        if args[0] == "log":
            return "abc1234def5678\n"
        if args[0] == "fetch":
            return ""
        if args[0] == "merge-base":
            return "" if self.pushed else None
        raise AssertionError(args)


def test_plan_check_refuses_until_committed_pushed_and_unchanged():
    lab = LB.load(df=fake_labels())
    with pytest.raises(PL.PlanError, match="no plan"):
        PL.check(lab, git=FakeGit())
    PL.write(PL.make(lab, {"summary": {"rule": {}}}))
    assert PL.check(lab, git=FakeGit())["plan_commit"] == "abc1234def5678"
    for bad, words in ((FakeGit(tracked=False), "not committed"), (FakeGit(same=False), "differs from the commit"),
                       (FakeGit(clean=False), "measuring code differs"), (FakeGit(pushed=False), "not on GitHub")):
        with pytest.raises(PL.PlanError, match=words):
            PL.check(lab, git=bad)
    with pytest.raises(PL.PlanError, match="answers"):
        PL.check(lab.assign(label=np.where(lab["label"] == "yes", "no", "yes")), git=FakeGit())


def test_plan_check_refuses_changed_settings_or_code(monkeypatch):
    lab = LB.load(df=fake_labels())
    PL.write(PL.make(lab, None))
    monkeypatch.setattr(config, "EYES_SEEDS", (0, 1, 2, 3))
    with pytest.raises(PL.PlanError, match="settings"):
        PL.check(lab, git=FakeGit())
    monkeypatch.setattr(config, "EYES_SEEDS", (0, 1, 2))
    monkeypatch.setattr(PL, "code_sha256", lambda: "changed")
    with pytest.raises(PL.PlanError, match="measuring code"):
        PL.check(lab, git=FakeGit())


def test_run_refuses_a_second_scoring_without_a_reason(monkeypatch, capsys):
    from machine.eyes import __main__ as CLI
    lab = LB.load(df=fake_labels())
    monkeypatch.setattr(LB, "load", lambda *a, **k: lab)
    monkeypatch.setattr(PL, "check", lambda *a, **k: {"plan_commit": "abc"})
    folder = config.EYES_WORK_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "runs.jsonl").write_text(json.dumps({"at": "2026-10-08T03:00+05:30"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(RUN, "prepare", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scored again")))
    assert CLI.main(["run"]) == 1
    assert "--again" in capsys.readouterr().out


# ---- the report ------------------------------------------------------------------------------------------------------

def fake_result():
    d = fake_data()
    r = RUN.measure(d, d.lab["y"].to_numpy(), models=("model A",))
    g = r["guesses"]
    g["model B chance"] = np.clip(g["model A chance"] * 0.8 + 0.1, 0, 1)
    g["model B said"] = (g["model B chance"] >= 0.5).astype(int)
    years = g["year"].to_numpy()
    r["scores"]["model B"] = M.score("model B", g["his"], g["model B said"], g["model B chance"], years).as_dict()
    right = {k: g[f"{k} said"].to_numpy() == g["his"].to_numpy() for k in RUN.GUESSERS}
    for i, a in enumerate(RUN.GUESSERS):
        for b in RUN.GUESSERS[i + 1:]:
            r["pairs"][f"{a} vs {b}"] = M.pair(right[a], right[b])
    r["winner"] = RUN.winner({k: M.Score(**v) for k, v in r["scores"].items()}, r["pairs"])
    r["model B each start"] = {0: 0.7, 1: 0.71, 2: 0.69}
    return d, r


def test_report_page_easy_words_and_numbers():
    from machine.charts.lint import BANNED
    import re
    d, r = fake_result()
    plan = PL.make(d.lab, None) | {"plan_commit": "abc1234"}
    text = RP.page(r, plan, made_at="2026-10-08T03:00+05:30", commit="def5678", run_no=1, exams=None,
                   left={"no_picture": 2, "not_sure": 0}, lab=d.lab)
    assert not re.search(r"\b(" + "|".join(BANNED) + r")\b", text, flags=re.IGNORECASE)
    assert "90% interval" in text and 0 < text.index("## The answer") < text.index("## The numbers")
    assert f"{r['scores']['model A']['right']} of {r['n']}" in text
    assert "scored once" in text and "#117" in text
    assert "two readings" in text and ("What the numbers support" in text or "support reading 1" in text)
    assert f"about {M.half_width(r['n']) * 100:.1f} points either side" in text


def test_mistake_captions_use_the_pictures_india_time():
    # chart 1 of 1 Oct: 20-24 Jul 2020 on the server clock; its picture's axis runs 20 Jul 03:30 - 25 Jul 01:30 (IST)
    assert RP.days_ist("2020-07-20 01:05", "2020-07-24 23:50") == "20 Jul - 25 Jul 2020 (India time)"
    # winter (server UTC+2): 23:00 server = 02:30 the next day in India
    assert RP.days_ist("2019-12-02 01:05", "2019-12-06 23:50") == "02 Dec - 07 Dec 2019 (India time)"


def test_report_writes_everything_and_counts_tries():
    d, r = fake_result()
    plan = PL.make(d.lab, None) | {"plan_commit": "abc1234"}
    line = RP.write_all(r, plan, d.lab, exams=None, left={"no_picture": 2, "not_sure": 0},
                        data_fp={"labels_sha256": "x", "gold_m1_content_sha256": "y"}, run_no=1)
    folder = config.EYES_WORK_DIR
    assert (folder / "mistakes-model-a.jpg").is_file() and (folder / "mistakes-model-b.jpg").is_file()
    assert config.EYES_REPORT.is_file() and (folder / "guesses.csv").is_file()
    files = sorted(config.TRIALS_DIR.glob("*.json"))
    assert len(files) == 2 and len(line["trial_files"]) == 2
    t = json.loads(files[0].read_text(encoding="utf-8"))
    assert t["kind"] == config.EYES_KIND and t["idea_id"] == "eyes-strong_zone" and t["result"]["n"] == r["n"]
    assert RP.runs()[-1]["run_no"] == 1
    before = config.EYES_REPORT.read_text(encoding="utf-8")
    RP.rebuild(plan, d.lab, exams=None, left={"no_picture": 2, "not_sure": 0})      # no new scoring
    assert len(sorted(config.TRIALS_DIR.glob("*.json"))) == 2 and len(RP.runs()) == 1
    assert config.EYES_REPORT.read_text(encoding="utf-8") == before


# ---- with the real price data ------------------------------------------------------------------------------------

@pytest.mark.data
def test_features_never_read_past_the_charts_last_candle():
    """The "cut off the future" test (R4): a chart's numbers are the same with every later candle cut away."""
    from machine import guards
    from machine.data.bars import load_bars
    m5 = load_bars("5m", "2019-03-01", "2019-04-30")
    starts = pd.date_range("2019-03-04 01:05", periods=6, freq="7D")
    lab = pd.DataFrame({"label_id": [f"c{i}" for i in range(6)], "first_candle": starts,
                        "last_candle": starts + pd.Timedelta(days=4, hours=22, minutes=45)})
    lab["lo"] = [float(m5.loc[s:s + pd.Timedelta(days=2), "close"].median()) for s in starts]
    lab["hi"] = lab["lo"] + 1.5

    def fn(bars: pd.DataFrame) -> pd.DataFrame:
        done = lab[lab["last_candle"] <= bars.index[-1]]
        return FE.build(done, bars).set_index(pd.DatetimeIndex(done["last_candle"].to_numpy()))
    cuts = [m5.index.get_indexer([t], method="ffill")[0] for t in lab["last_candle"].iloc[1:-1]]
    guards.assert_no_future(fn, m5, cuts)


@pytest.mark.data
@pytest.mark.parametrize("first,last", [("2019-05-06 01:05", "2019-05-10 23:50"),
                                        ("2016-03-14 01:05", "2016-03-18 23:50")])
def test_the_crop_is_the_chart_area_of_a_label_picture(tmp_path, first, last):
    """Draw a real label chart the label way: the crop holds the whole chart area (so every candle) and cuts away
    the heading, the question, the price and the date labels around it (at most a few white dots kept)."""
    from machine.data.bars import load_bars
    from machine.labels import picture as P
    from machine.labels import words as W
    m5 = load_bars("5m", first[:10], last[:10])
    h = P.hour_candles(m5, first, last)
    lo = float(h["close"].median())
    d = P.draw(W.get(config.EYES_WORD), h, lo, lo + 2.0, tmp_path / "x.jpg", subject="test")
    assert d.size == (1560, 1191)
    H = d.size[1]
    area = d.panels[0].area                                   # matplotlib's pixels: y counts up from the bottom
    left, top, right, bottom = config.EYES_CROP
    assert left <= area.x0 and area.x1 <= right + 0.5 and top <= H - area.y1 and H - area.y0 <= bottom
    assert area.x0 - left < 6 and right - area.x1 < 6 and (H - area.y1) - top < 6 and bottom - (H - area.y0) < 6
    for c in d.panels[0].candles:
        assert left <= c.x0 and c.x1 <= right and top <= H - c.y1 and H - c.y0 <= bottom
