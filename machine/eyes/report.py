"""The report of the one real run (L15.2): plan/work/eyes-model.md (easy English, #113), the pictures of mistakes,
every held-out guess (plan/work/eyes/guesses.csv), one trial file per model (plan/work/trials/, kind "eyes") and one
line in plan/work/eyes/runs.jsonl.

The pictures of mistakes stay in the report: they are charts he already answered, so they are never sent to him
again (#117 "I don't want the same charts again").
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config
from machine import trials as trial_files
from machine.data import clock
from machine.eyes import features as FE
from machine.eyes import measure as M
from machine.eyes import run as RUN

MODELS = ("model A", "model B")
SLUG = {"model A": "model-a", "model B": "model-b"}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%" if np.isfinite(x) else "-"


def interval(s: dict) -> str:
    return f"{pct(s['lo'])} to {pct(s['hi'])}"


def page(r: dict, plan: dict, *, made_at: str, commit: str, run_no: int, exams: dict | None, left: dict,
         lab: pd.DataFrame) -> str:
    sc, pr, w = r["scores"], r["pairs"], r["winner"]
    n, base = r["n"], sc["always no"]
    L = ["# His eyes (L15.2) - can a model see his kind of zone?", "",
         "> Written for Tejas, Likhith and anyone reading this repo. Made by `python -m machine.eyes run` on "
         f"{made_at} (commit `{commit[:10]}`). The plan was fixed and pushed first (commit "
         f"`{plan.get('plan_commit', '?')[:10]}`, `plan/work/eyes/plan.md`). His held-out answers were scored "
         f"{'once' if run_no == 1 else f'{run_no} times - see runs.jsonl'}.", "",
         "## The answer", ""]
    L += answer_lines(r)
    L += ["", f"## The numbers - {n} charts held out ({r['years'][0]}-{r['years'][-1]}: he said yes to {r['yes']}, "
          f"no to {r['no']})", "",
          "Each year was guessed by models that learned ONLY from the charts before it (2018 from 2016-2017, 2019 "
          "from 2016-2018, 2020 from 2016-2019).", "",
          "| who | right | accuracy | 90% interval | his yes found | his no found | a yes chart ranks higher |",
          "|---|---|---|---|---|---|---|"]
    for g in RUN.GUESSERS:
        if g in sc:
            s = sc[g]
            rank = "-" if not np.isfinite(s["ranks"]) else f"{s['ranks']:.2f} ({s['ranks_lo']:.2f}-{s['ranks_hi']:.2f})"
            L.append(f"| {RUN.NAMES[g]} | {s['right']} of {s['n']} | **{pct(s['accuracy'])}** | {interval(s)} | "
                     f"{s['yes_found']} of {s['yes_all']} | {s['no_found']} of {s['no_all']} | {rank} |")
    hw, hw400 = M.half_width(n), M.half_width(400)
    L += ["", f"**How wide:** with {n} held-out answers each 90% interval is about {hw * 100:.1f} points either side "
          f"(at 400 it would be about {hw400 * 100:.1f}). There are about 200 charts, not 600, because no chart may "
          "show a candle twice (his #117, #168).",
          "",
          "*A yes chart ranks higher* = take one chart he said yes to and one he said no to: how often the guesser "
          "gives the yes chart the higher chance (0.50 = a coin, 1.00 = always; in brackets its 90% interval). His "
          "rule's chance = how small calm is.", "",
          "## Is a difference luck?", "",
          "Only the charts where exactly one of the two is right count. With no real difference each of them is a "
          "coin toss; *beyond luck* = fewer than 10 in 100 coin tosses split that unevenly.", "",
          "| pair | only the first right | only the second right | chance it is luck | beyond luck? |",
          "|---|---|---|---|---|"]
    for key, p in pr.items():
        a, b = key.split(" vs ")
        L.append(f"| {RUN.NAMES[a]} vs {RUN.NAMES[b]} | {p['only_a']} | {p['only_b']} | {p['p']:.2f} | "
                 f"{'yes' if p['beyond_luck'] else 'no'} |")
    L += ["", "## Year by year", "", "| who | " + " | ".join(str(y) for y in r["years"]) + " | worst year |",
          "|---|" + "---|" * (len(r["years"]) + 1)]
    for g in RUN.GUESSERS:
        if g in sc:
            by = {int(k): v for k, v in sc[g]["by_year"].items()}       # keys are text after result.json
            cells = [f"{by[y][0]} of {by[y][1]} ({pct(by[y][0] / by[y][1])})" for y in r["years"]]
            worst = min(r["years"], key=lambda y: by[y][0] / by[y][1])
            L.append(f"| {RUN.NAMES[g]} | " + " | ".join(cells) + f" | {worst} |")
    L += ["", "## His answer against its answer", ""]
    for g in ("his rule", *MODELS):
        if g in sc:
            t = sc[g]["table"]
            L += [f"**{RUN.NAMES[g]}**", "", "| | it said yes | it said no |", "|---|---|---|",
                  f"| he said yes | {t['he yes, it yes']} | {t['he yes, it no']} |",
                  f"| he said no | {t['he no, it yes']} | {t['he no, it no']} |", ""]
    if r.get("model A uses"):
        L += ["## What model A leaned on", "",
              "How much each number helped model A's trees (its share of all the help, the 3 years' models together):",
              "", "| number | share | what it is |", "|---|---|---|"]
        for k, v in list(r["model A uses"].items())[:8]:
            L.append(f"| {k} | {pct(v)} | {FE.COLUMNS[k]} |")
        L.append("")
    if r.get("model B each start"):
        each = ", ".join(f"{pct(v)}" for v in r["model B each start"].values())
        L += ["## Model B's three starts", "",
              f"Model B was taught 3 times from different random starts; each start alone was right on {each} of "
              "the held-out charts. Its answer above is the average of the three chances.", ""]
    L += ["## Pictures of mistakes", "",
          "The charts each model was surest about and wrong - he said yes and it said no (left), he said no and it "
          "said yes (right). They stay in this file: he has seen these charts, so they are never sent again (#117).",
          ""]
    for g in MODELS:
        if g in sc:
            L += [f"![{RUN.NAMES[g]} - its surest mistakes](eyes/mistakes-{SLUG[g]}.jpg)", ""]
    L += how_lines(r, plan, exams, left, lab)
    L += honest_lines(r, lab)
    return "\n".join(L) + "\n"


def answer_lines(r: dict) -> list[str]:
    sc, pr, w = r["scores"], r["pairs"], r["winner"]
    base, rule = sc["always no"], sc["his rule"]
    out = []
    if w.get("name"):
        loser = "model B" if w["name"] == "model A" else "model A"
        out.append(f"- **The winner on these charts: {RUN.NAMES[w['name']]}** - right on "
                   f"{sc[w['name']]['right']} of {r['n']} ({pct(sc[w['name']]['accuracy'])}), against "
                   f"{sc[loser]['right']} ({pct(sc[loser]['accuracy'])}) for {RUN.NAMES[loser]}. "
                   + ("That gap is beyond luck." if w["beyond_luck"] else
                      f"That gap is NOT beyond luck (they differ on {w['only_a'] + w['only_b']} charts; a gap this "
                      f"size comes by luck {w['p'] * 100:.0f} times in 100)."))
    else:
        out.append(f"- **No winner:** {w['said']}.")
    for g in MODELS:
        if g in sc:
            pb = pr[f"always no vs {g}"]
            ph = pr[f"his rule vs {g}"]
            vs_base = ("better than 'always no' beyond luck" if pb["beyond_luck"] and pb["only_b"] > pb["only_a"]
                       else "worse than 'always no' beyond luck" if pb["beyond_luck"]
                       else "not beyond luck against 'always no'")
            vs_rule = ("better than his rule beyond luck" if ph["beyond_luck"] and ph["only_b"] > ph["only_a"]
                       else "worse than his rule beyond luck" if ph["beyond_luck"]
                       else "not beyond luck against his rule")
            out.append(f"- {RUN.NAMES[g]}: {pct(sc[g]['accuracy'])} ({interval(sc[g])}) - {vs_base}; {vs_rule}.")
    out.append(f"- For comparison: 'always no' is right on {pct(base['accuracy'])} of these charts "
               f"({base['right']} of {r['n']}), his written rule on {pct(rule['accuracy'])}.")
    return out + meaning_lines(r)


def meaning_lines(r: dict) -> list[str]:
    """Two readings and what the numbers support (CLAUDE.md, the judgement guard: the verdict is his)."""
    sc, pr = r["scores"], r["pairs"]
    if not all(g in sc for g in MODELS):
        return []
    rule, t = sc["his rule"], sc["his rule"]["table"]
    differ = t["he yes, it no"] + t["he no, it yes"]
    found = [sc[g]["yes_found"] for g in (*MODELS, "his rule")]
    better = [g for g in MODELS
              if pr[f"his rule vs {g}"]["beyond_luck"] and pr[f"his rule vs {g}"]["only_b"] > pr[f"his rule vs {g}"]["only_a"]]
    accs = ", ".join(f"{RUN.NAMES[g].split(' - ')[0]} {pct(sc[g]['accuracy'])}" for g in (*MODELS, "his rule"))
    out = ["", "**What it means - two readings:**",
           f"1. *His eye sees more than his written rule, and more answers would teach it to a model.* For it: he "
           f"answers differently from his rule on {differ} of the {r['n']} charts (yes to {t['he yes, it no']} it "
           f"calls busy, no to {t['he no, it yes']} it calls quiet), and every guesser misses most of his yes "
           f"({min(found)}-{max(found)} of {rule['yes_all']} found).",
           "2. *Where he answers differently from his rule, it is a one-look answer that could go either way.* For "
           "it: neither model, taught on his own answers, is right more often than his rule beyond luck.", ""]
    if better:
        out.append(f"**The numbers support reading 1:** {' and '.join(RUN.NAMES[g] for g in better)} is right more "
                   "often than his written rule beyond luck.")
    else:
        out.append(f"**What the numbers support:** on these answers a model sees about what his written rule sees - "
                   f"no more ({accs}). They cannot tell the two readings apart; that needs his answers on charts he "
                   "has not seen.")
    return out


def how_lines(r: dict, plan: dict, exams: dict | None, left: dict, lab: pd.DataFrame) -> list[str]:
    s = plan["settings"]
    by_src = lab["source"].value_counts().to_dict()
    L = ["## How it was done", "",
         f"- **His answers:** {len(lab)} with a picture - {by_src.get('telegram', 0)} zone charts he answered on "
         f"Telegram (1-7 Oct 2026) and {by_src.get('his drawing', 0)} of his drawn cards (Q78, Q79); "
         f"{int(lab['y'].sum())} yes, {int((1 - lab['y']).sum())} no. The question on every picture: \"Is the brown "
         "band YOUR kind of zone - the quiet one you trade?\" (L15.1). Left out: "
         f"{left['no_picture']} answers in the hidden years 2021-2023 (never drawn - those years stay closed) and "
         f"{left['not_sure']} 'not sure'.",
         "- **The split by time:** " + "; ".join(r["folds"]) + f". A chart teaches a year only if it ended "
         f"{s['purge_days']} days or more before that year's first chart.",
         f"- **Model A:** LightGBM on {len(FE.COLUMNS)} numbers from the same hour candles the picture draws "
         f"(below); {s['lgbm_rounds']} small trees, no tuning.",
         f"- **Model B:** ResNet18 (PyTorch, torchvision) with its ImageNet weights, shown only the candle area of "
         f"the picture he answered at {s['image_hw'][0]} x {s['image_hw'][1]}; layers from "
         f"{s['net']['train_from']} on learn for {s['net']['epochs']} rounds (the picture moves up to "
         f"{s['net']['shift_px']} dots each round, never flipped); taught {len(s['seeds'])} times, the average.",
         f"- **Yes or no:** a model says yes when its chance of his yes is {s['threshold']} or more.",
         "- **Fixed first:** every setting was written before any model saw his answers: exams on made-up answers, "
         "then `plan/work/eyes/plan.json` committed and pushed, then this ONE scoring. `python -m machine.eyes run` "
         "refuses if the plan, the measuring code or his answers changed.", ""]
    if exams and exams.get("summary"):
        x = exams["summary"]
        L.append("**The exams on made-up answers** (`plan/work/eyes/exams.md`):")
        for key, words in (("noise A", "model A"), ("noise B", "model B")):
            if key in x:
                v = x[key]
                L.append(f"- noise ({words}, {v['runs']} runs, answers shuffled so nothing can be seen): middle "
                         f"{v['middle_gap'] * 100:+.1f} points against 'always no'; better than it beyond luck in "
                         f"{v['better_beyond_luck']} of {v['runs']} (a false alarm - about 1 in 20 may happen by "
                         "luck).")
        for g, v in x.get("rule", {}).items():
            if g in MODELS:
                L.append(f"- his written rule planted as the answer: {RUN.NAMES[g]} right on "
                         f"{100 * v['accuracy']:.1f}% ('always no' {100 * v['always_no']:.1f}%).")
        if exams.get("recipes"):
            tried = [k for k in exams["recipes"]["tried"] if not k.startswith("the first way")]
            L.append(f"- {len(tried)} other ways to teach model B were tried on the same planted answers; none cleared "
                     "the bar written before they ran, so the first way stayed (exams.md).")
        L.append("")
    L += ["**Model A's numbers** (machine/eyes/features.py; \"in the band\" = the candle's body touches the band):",
          "", "| number | what it is |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in FE.COLUMNS.items()]
    return L + [""]


def honest_lines(r: dict, lab: pd.DataFrame) -> list[str]:
    n = r["n"]
    return [
        "## What could make this wrong", "",
        f"- **Few answers.** {n} held-out answers give intervals about {M.half_width(n) * 100:.1f} points wide either "
        "side; a gap of a few points between two guessers is inside them.",
        f"- **'Always no' is hard to beat on accuracy.** He said no to {r['no']} of the {n} held-out charts, so a "
        "guesser that never says yes is right that often. The columns 'his yes found' and 'a yes chart ranks higher' "
        "show what accuracy alone hides.",
        "- **His rule's number came from his own answers.** 0.74 was set between his Q79 answers (#80), and Q79's "
        "two drawn yes cards are among the answers here (in 2017, so only as teaching, never guessed).",
        "- **How the charts were picked.** 3 / 3 / 2 / 2 a day by how quiet the band was (L15.1), not as zones come; "
        "on zones as they come, most are busy and 'always no' would be right even more often.",
        "- **The first 30 charts (1-3 Oct) could share candles** (#167: two showed the same week). The split keeps "
        "a teaching chart 7 days or more away from the year it guesses, so no candle is both taught and guessed.",
        "- **One look per chart.** He answered each chart once and quickly (many sets in one morning); nobody knows "
        "how often he would answer the same chart the same way - no chart is shown twice. That limit holds for "
        "every guesser here.",
        "- **Model B's settings were not tuned on his answers** (only the exams' made-up ones were seen before), so "
        "a better recipe may exist; trying recipes on these 120 answers would spend them.",
        ""]


def days_ist(first_candle, last_candle) -> str:
    """The days of a label chart's first and last HOUR candle in India time - the clock of its x axis (lesson B24;
    the label file's times are MT5 server time)."""
    first = clock.to_ist(pd.Timestamp(first_candle).floor("1h"))
    last = clock.to_ist(pd.Timestamp(last_candle).floor("1h"))
    return f"{first:%d %b} - {last:%d %b %Y} (India time)"


def mistakes_picture(guesses: pd.DataFrame, lab: pd.DataFrame, g: str, path: Path) -> list[str]:
    """The surest mistakes of guesser g: he said yes / it said no (left), he said no / it said yes (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams["text.parse_math"] = False            # lesson B8
    by_id = lab.set_index("label_id")
    ch, said = guesses[f"{g} chance"], guesses[f"{g} said"]

    def pick(mask: pd.Series, surest_first: bool) -> pd.DataFrame:
        sub = guesses[mask].copy()
        sub["c"] = ch[mask].to_numpy()          # only these rows (an empty pick stays empty)
        return sub.sort_values("c", ascending=not surest_first)
    miss_yes = pick((guesses["his"] == 1) & (said == 0), surest_first=False)    # lowest chance of yes first
    miss_no = pick((guesses["his"] == 0) & (said == 1), surest_first=True)      # highest chance of yes first
    k = int(config.EYES_MISTAKES_EACH)
    cols = [miss_yes.head(k), miss_no.head(k)]
    rows = max(1, max(len(c) for c in cols))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.9 * rows + 0.9), squeeze=False)
    fig.suptitle(f"{RUN.NAMES[g]}: its surest mistakes on the held-out charts", fontsize=15, fontweight="bold",
                 x=0.01, ha="left")
    heads = ("He said YES - the model said no", "He said NO - the model said yes")
    shown = []
    for j, col in enumerate(cols):
        for i in range(rows):
            ax = axes[i][j]
            ax.axis("off")
            if i >= len(col):
                if i == 0:
                    ax.text(0.5, 0.5, "no mistake of this kind", ha="center", va="center", fontsize=12)
                continue
            row = col.iloc[i]
            b = by_id.loc[row["label_id"]]
            im = Image.open(io.BytesIO(b["image"])).convert("RGB").crop(config.EYES_CROP)
            ax.imshow(np.asarray(im))
            chance = "under 1" if row["c"] < 0.005 else f"{100 * row['c']:.0f}"
            ax.set_title(f"{heads[j]}\nits chance of his yes: {chance} in 100 · calm "
                         f"{float(b['calm_on_chart']):.2f} · {days_ist(b['first_candle'], b['last_candle'])}",
                         fontsize=10.5, loc="left")
            shown.append(str(row["label_id"]))
    fig.text(0.01, 0.005, "1-hour candles · dates in India time · the brown band = the zone he was asked about · the "
             "price and date labels of the original picture are cut away", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.jpg")
    fig.savefig(tmp, dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)
    os.replace(tmp, path)
    return shown


def record(r: dict, plan: dict, data_fp: dict, *, run_no: int, note: str = "") -> list[Path]:
    """One trial file per model (kind config.EYES_KIND) - the count of scorings of his held-out answers."""
    out = []
    sc = r["scores"]
    for g in MODELS:
        if g not in sc:
            continue
        s = sc[g]
        part = {"model A": {"lgbm_params": plan["settings"]["lgbm_params"],
                            "lgbm_rounds": plan["settings"]["lgbm_rounds"]},
                "model B": {"net": plan["settings"]["net"], "image_hw": plan["settings"]["image_hw"],
                            "crop": plan["settings"]["crop"], "seeds": plan["settings"]["seeds"]}}[g]
        settings = {"model": g, "test_years": plan["settings"]["test_years"],
                    "purge_days": plan["settings"]["purge_days"], "threshold": plan["settings"]["threshold"], **part}
        result = {"n": s["n"], "right": s["right"], "accuracy": s["accuracy"], "lo": s["lo"], "hi": s["hi"],
                  "level": config.EYES_LEVEL, "yes_found": s["yes_found"], "yes_all": s["yes_all"],
                  "no_found": s["no_found"], "no_all": s["no_all"], "ranks_higher": s["ranks"],
                  "by_year": {str(y): v for y, v in s["by_year"].items()},
                  "always_no_accuracy": sc["always no"]["accuracy"], "his_rule_accuracy": sc["his rule"]["accuracy"],
                  "vs_always_no": r["pairs"][f"always no vs {g}"], "vs_his_rule": r["pairs"][f"his rule vs {g}"],
                  "winner": r["winner"].get("name"), "run_no": run_no}
        version = f"{SLUG[g]}-{trial_files.sha256_of(settings)[:10]}"
        out.append(trial_files.record(
            kind=config.EYES_KIND, idea_id=f"eyes-{config.EYES_WORD}", version_id=version, parent_idea=None,
            settings=settings, data=data_fp, result=result,
            note=(f"L15.2: {g} on his held-out zone answers (2018-2020), the plan of commit "
                  f"{plan.get('plan_commit', '?')[:10]}. {note}").strip()))
    return out


def write_all(r: dict, plan: dict, lab: pd.DataFrame, *, exams: dict | None, left: dict, data_fp: dict,
              run_no: int, again: str = "", folder: Path | None = None, report: Path | None = None) -> dict:
    folder = Path(folder or config.EYES_WORK_DIR)
    report = Path(report or config.EYES_REPORT)
    folder.mkdir(parents=True, exist_ok=True)
    made_at = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    commit = (trial_files.git_state() or {}).get("commit", "?")
    g = r["guesses"].copy()
    _put(folder / "guesses.csv", g.round(4).to_csv(index=False, lineterminator="\n"))
    kept = {k: v for k, v in r.items() if k != "guesses"}
    _put(folder / "result.json", json.dumps({"made_at": made_at, "commit": commit, "run_no": run_no, **kept},
                                            indent=2, ensure_ascii=False, default=_plain) + "\n")
    shown = {m: mistakes_picture(g, lab, m, folder / f"mistakes-{SLUG[m]}.jpg") for m in MODELS if m in r["scores"]}
    _put(report, page(r, plan, made_at=made_at, commit=commit, run_no=run_no, exams=exams, left=left, lab=lab))
    trials = record(r, plan, data_fp, run_no=run_no, note=f"again: {again}" if again else "")
    line = {"at": made_at, "run_no": run_no, "again": again or None, "commit": commit,
            "plan_commit": plan.get("plan_commit"), "labels": plan["labels"]["fingerprint"], "n": r["n"],
            "accuracy": {k: v["accuracy"] for k, v in r["scores"].items()}, "winner": r["winner"].get("name"),
            "trial_files": [p.name for p in trials], "mistakes_shown": shown, "seconds": r.get("seconds")}
    with (folder / "runs.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line


def rebuild(plan: dict, lab: pd.DataFrame, *, exams: dict | None, left: dict, folder: Path | None = None,
            report: Path | None = None) -> Path:
    """The report and pictures again from result.json + guesses.csv - his answers are NOT scored again (no trial
    file, no runs line): for wording fixes after the one run."""
    folder = Path(folder or config.EYES_WORK_DIR)
    report = Path(report or config.EYES_REPORT)
    saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    r = {**saved, "guesses": pd.read_csv(folder / "guesses.csv")}
    for g in MODELS:
        if g in r["scores"]:
            mistakes_picture(r["guesses"], lab, g, folder / f"mistakes-{SLUG[g]}.jpg")
    _put(report, page(r, plan, made_at=saved["made_at"], commit=saved["commit"], run_no=saved["run_no"],
                      exams=exams, left=left, lab=lab))
    return report


def _plain(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f"cannot save a {type(x).__name__}")


def runs(folder: Path | None = None) -> list[dict]:
    p = Path(folder or config.EYES_WORK_DIR) / "runs.jsonl"
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _put(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
