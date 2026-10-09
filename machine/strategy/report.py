"""The reports of his whole strategy (L19d): one for the whole, one per entry type - made ONLY from the files
(research-plan.json and the trial files), so `python -m machine.strategy reports` makes them again without running
anything. Easy English (his #113; machine.report.build.check_words is his rule as a test). Money: Rs after Rs 40 a
position (0.01 lot a leg, all-in), a trade counts in the year its last leg closed; trades still open at the end of a
window are not counted (said).

    plan/work/L19d/README.md                 the whole: the answer, the plan, every version, the 2x2 year by year
    plan/work/L19d/whole-years.jpg           his strategy year by year, beside random entries with his hedge
    plan/work/L19d/whole-boxes.jpg           the four boxes, all years
    plan/work/L19d/<entry>.md (+ .jpg)       each entry type: its part of the whole, or why it is not in it
    plan/work/ideas/his-strategy/research-final.md     the final message: the answer + every version
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from machine import config, costs
from machine.entries import forms
from machine.referee import prior, splits, verdict
from machine.strategy import research as SR
from machine.strategy import whole as W
from machine.tester import hedging_2x2 as H2

inr = costs.inr
BOX_WORDS = ("his entries + his hedging", "his entries + a plain stop", "random entries + his hedging",
             "random entries + a plain stop")
GROUP_ALL = "all"


class ReportError(ValueError):
    """Nothing to report yet, or the files disagree."""


# ================================================================================================= the facts

def facts() -> dict:
    """Everything the reports read, from the files only."""
    plan = SR.load_plan()
    if plan is None:
        raise ReportError("no research plan yet: python -m machine.strategy plan")
    runs = {f["version_id"]: f for f in SR.files(W.KIND)}
    tune = (SR.files(SR.TUNE_KIND) or [None])[-1]
    chosen = tune["result"]["chosen"] if tune else None
    tt = [f for f in SR.files(W.KIND_2X2) if f["version_id"] == chosen]
    ref = [f for f in SR.files("referee") if f["version_id"] == chosen]
    return {"plan": plan, "runs": runs, "tune": tune, "chosen": chosen, "tune_2x2": tt[-1] if tt else None,
            "judged": ref[-1] if ref else None, "opening": splits.family_opening(W.STRATEGY),
            "budget": SR.budget(), "opened_count": splits.opened_count()}


def _n(k, word: str) -> str:
    k = int(k)
    if word == "try":
        return f"{k} try" if k == 1 else f"{k} tries"
    return f"{k} {word}" + ("" if k == 1 else "s")


def _years_all(fx: dict) -> list[str]:
    ys = [str(y) for y in config.TUNE_YEARS]
    if fx["judged"]:
        ys += [str(y) for y in config.CHECK_YEARS]
    return ys


def _merged(fx: dict, group: str) -> dict | None:
    """The 2x2 of one group over every year shown: 2016-2020 (the tune window) and 2021-2023 (the judged window),
    run by run (random run k of each window added up)."""
    parts = []
    for f in (fx["tune_2x2"], fx["judged"]):
        if f is None:
            continue
        g = f["result"]["two_by_two"]["groups"].get(group)
        if g is not None:
            parts.append(g)
    if not parts:
        return None
    years = [y for g in parts for y in g["years"]]
    boxes = []
    for b in range(4):
        per = {y: v for g in parts for y, v in g["boxes"][b]["per_year"].items()}
        trades = sum(g["boxes"][b]["trades"] for g in parts)
        open_end = sum(g["boxes"][b]["open_at_end"] for g in parts)
        boxes.append({"words": BOX_WORDS[b], "per_year": per, "trades": trades, "open_at_end": open_end,
                      "random": b >= 2})
    n = min(len(g["run_totals"]["hedge"]) for g in parts)
    tot = {box: np.sum([np.array(g["run_totals"][box][:n]) for g in parts], axis=0) for box in ("hedge", "stop")}
    his_h = sum(boxes[0]["per_year"].values())
    his_s = sum(boxes[1]["per_year"].values())
    for b, box in ((2, "hedge"), (3, "stop")):
        boxes[b]["total"] = float(np.median(tot[box]))
        boxes[b]["total_5_95"] = [float(np.percentile(tot[box], 5)), float(np.percentile(tot[box], 95))]
    for b in (0, 1):
        boxes[b]["total"] = float(sum(boxes[b]["per_year"].values()))
    return {"years": years, "boxes": boxes, "runs": n, "his_beats_random_hedging": int((tot["hedge"] < his_h).sum()),
            "his_beats_random_stop": int((tot["stop"] < his_s).sum()),
            "hedging_beats_stop_in_random_runs": int(((tot["hedge"] - tot["stop"]) > 0).sum()),
            "beats_by_year": {y: v for g in parts for y, v in g["his_beats_random"]["with_hedging_by_year"].items()}}


# ================================================================================================= the pictures

def _figure_text_check(texts: list[str]) -> None:
    from machine.report.build import check_words
    problems = [p for t in texts for p in check_words(t)]
    if problems:
        raise ReportError("the picture's words: " + "; ".join(problems))


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", dpi=150)
    plt.close(fig)
    buf.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.open(buf).convert("RGB").save(tmp, format="JPEG", quality=92, optimize=True)
    tmp.replace(path)


def picture_years(m: dict, path: Path, title: str) -> list[str]:
    """ONE chart: his strategy's rupees year by year (bars), beside the middle random run with his hedge (a black dot
    each year); a line between the years it was chosen on and the years it was judged on once."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from machine.charts import draw as D
    plt.rcParams["text.parse_math"] = False
    years = m["years"]
    his = [m["boxes"][0]["per_year"][y] for y in years]
    mids = _random_mid_by_year(m)
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.8), dpi=D.DPI)
    x = np.arange(len(years))
    ax.bar(x, his, width=0.6, color=[D.GREEN if v > 0 else D.RED for v in his], zorder=2,
           label="his entries + his hedging (his strategy)")
    ax.scatter(x, mids, color=D.INK, s=46, zorder=3, label="random entries + his hedging (the middle of the random runs)")
    ax.axhline(0, color=D.INK, linewidth=1.0, zorder=1)
    texts = ["his entries + his hedging (his strategy)", "random entries + his hedging (the middle of the random runs)"]
    ticks = [f"{y}\n{inr(v)}" for y, v in zip(years, his)]         # his rupees under each year: no word on a bar
    texts += ticks
    if any(int(y) >= config.CHECK_YEARS[0] for y in years) and any(int(y) <= config.TUNE_YEARS[-1] for y in years):
        cut = next(k for k, y in enumerate(years) if int(y) >= config.CHECK_YEARS[0]) - 0.5
        ax.axvline(cut, color=D.MUTED, linestyle="--", linewidth=1.2)
        top = ax.get_ylim()[1]
        for t, xx, ha in (("chosen on these years", cut - 0.1, "right"), ("judged once on these years", cut + 0.1,
                                                                            "left")):
            ax.text(xx, top, t, ha=ha, va="top", fontsize=D.F_AXIS, color=D.MUTED)
            texts.append(t)
    ax.set_xticks(x, ticks, fontsize=D.F_AXIS, color=D.INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="y", labelsize=D.F_AXIS, colors=D.MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=D.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    # the key above the chart, never over a bar (the first picture's key covered 2018)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.02, 0.89), fontsize=D.F_AXIS,
               frameon=False, ncol=1)
    sub = "money after costs, each year · 0.01 lot a leg · ₹40 a position, all-in · a trade counts in the year it closed"
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=D.F_SUB + 2, color=D.INK, fontweight="bold")
    fig.text(0.02, 0.915, sub, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    texts += [title, sub]
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.12)
    _figure_text_check(texts)
    _save(fig, path)
    return texts


def _random_mid_by_year(m: dict) -> list[float]:
    """The middle random run with his hedge, per year (each window's own middle)."""
    out = []
    for y in m["years"]:
        out.append(m["boxes"][2]["per_year"][y])
    return out


def picture_boxes(m: dict, path: Path, title: str, line: str | None = None, *, seen: bool = True) -> list[str]:
    """The four boxes over every year shown (random = the middle run of the window totals added run by run)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from machine.charts import draw as D
    plt.rcParams["text.parse_math"] = False
    vals = [float(b["total"]) for b in m["boxes"]]
    trades = [b["trades"] for b in m["boxes"]]
    names = [f"{b['words']}\n({t:g} trades{', middle of ' + str(m['runs']) + ' runs' if b['random'] else ''})"
             for b, t in zip(m["boxes"], trades)]
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.6), dpi=D.DPI)
    y = list(range(len(vals)))[::-1]
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    span = (hi - lo) or 1.0
    ax.barh(y, vals, height=0.58, color=[D.GREEN if v > 0 else D.RED for v in vals], zorder=2)
    ax.axvline(0, color=D.INK, linewidth=1.0, zorder=3)
    texts = list(names)
    for yy, v in zip(y, vals):
        t = inr(v)
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), yy, t, va="center", ha="left" if v >= 0 else "right",
                fontsize=D.F_NAME, color=D.INK, fontweight="bold")
        texts.append(t)
    ax.set_yticks(y, names, fontsize=D.F_AXIS, color=D.INK)
    ax.set_xlim(lo - 0.3 * span, hi + 0.3 * span)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="x", labelsize=D.F_AXIS, colors=D.MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=D.GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ys = m["years"][0] if m["years"][0] == m["years"][-1] else f"{m['years'][0]}–{m['years'][-1]}"
    sub = f"money after costs, {ys} · 0.01 lot a leg · ₹40 a position, all-in"
    sub2 = ("already seen by the old project: the answer can only be 'chance — needs fresh years'" if seen else
            "the years it was chosen on - a look, not the referee's judging")
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=D.F_SUB + 2, color=D.INK, fontweight="bold")
    y0 = 0.9
    if line:
        fig.text(0.02, y0, line, ha="left", fontsize=D.F_AXIS + 1, color=D.INK)
        texts.append(line)
        y0 -= 0.045
    fig.text(0.02, y0, sub, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    fig.text(0.02, y0 - 0.045, sub2, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    texts += [title, sub, sub2]
    fig.subplots_adjust(left=0.33, right=0.97, top=y0 - 0.1, bottom=0.1)
    _figure_text_check(texts)
    _save(fig, path)
    return texts


# ================================================================================================= the whole

def _ref_line(fx: dict) -> str:
    j = fx["judged"]
    if j is None:
        return "**Not judged yet** - the referee has not opened 2021-2023 for a version."
    a = j["result"]["answer"]
    return f"**The answer: {a['word']}.** {a['why'].capitalize()}."


def _money_line(label: str, s: dict) -> str:
    w = s["worst_year"]
    return (f"- **{label}:** {_n(s['trades'], 'trade')} closed ({s['open_at_end']} still open at the end, not counted), "
            f"{inr(s['net_inr_total'])} after cost"
            + (f" = {inr(s['net_inr_per_trade'])} a trade" if s["trades"] else "")
            + f"; worst year {w['year']} ({inr(w['net_inr'])}).")


def whole_md(fx: dict) -> str:
    plan, b, chosen = fx["plan"], fx["budget"], fx["chosen"]
    L = ["# His whole strategy through the research loop (L19d)", "",
         "> Made by `python -m machine.strategy reports` from the files only · MT5 server time · 5-minute chart · "
         "0.01 lot a leg · ₹40 a position, all-in (LEARNED #54, #115) · a trade counts in the year its last leg closed "
         "· a trade still open at the end of a window is not counted.", "", _ref_line(fx), ""]
    run_c = fx["runs"].get(chosen) if chosen else None
    if fx["judged"]:
        L.append(_money_line(f"{config.CHECK_YEARS[0]}–{config.CHECK_YEARS[-1]}, judged once, with his hedge",
                             fx["judged"]["result"]["summary"]))
    if run_c:
        L.append(_money_line(f"{config.TUNE_YEARS[0]}–{config.TUNE_YEARS[-1]}, the years it was chosen on, with his "
                             "hedge", run_c["result"]["hedged"]))
    seen = prior.already_seen(W.SEEN_KEY, config.TUNE_YEARS + config.CHECK_YEARS)
    L += [f"- **Tries:** {_n(b['used'], 'version')} of his whole strategy here (each one a try of the family "
          f"`{W.STRATEGY}`, budget {b['budget']}), plus the old project's {seen.tries:,} tries of his entries with his "
          "management on these years.", ""]
    L += ["## What his whole strategy is here", "", W.WHAT_IT_IS, "",
          "| his entry | in the whole? | setups 2016–2020 |" + (" setups 2021–2023 |" if fx["judged"] else ""),
          "|---|---|---:|" + ("---:|" if fx["judged"] else "")]
    waits = W.waiting()                          # today's reason (the plan keeps the words of the day it was made)
    for e in forms.ENTRY_TYPES.values():
        if e.id in plan["entries_in"]:
            n1 = run_c["result"]["setups"].get(e.id, "-") if run_c else "-"
            n2 = fx["judged"]["result"]["by_entry"][e.id]["setups"] if fx["judged"] else None
            L.append(f"| [{e.title}]({e.id}.md) | yes | {n1} |" + (f" {n2} |" if n2 is not None else ""))
        else:
            L.append(f"| [{e.title}]({e.id}.md) | no: {waits.get(e.id, 'not in it')} | - |"
                     + (" - |" if fx["judged"] else ""))
    L += ["", "Why three entries are not in it: when this ran, each still had an open word in its form, and the "
          "machine never picks a meaning by money (lesson 16). Checked after the run: none of the three can change "
          "the answer - 'steep' and 'big move' are numbers the machine tests itself (his #154; L19b tried every value "
          "and found very few trades), and both answers to 'very strong' make Type 1's same trades. With his hedge, "
          "Type 1 with all five of his entries closing legs makes the same rupees (L19b). So nothing is asked.", ""]
    L += ["## The plan - made before any money", "", f"Saved {plan['made_at'][:16].replace('T', ' ')} "
          f"(fingerprint `{plan['sha256'][:12]}`), before any version ran. Claude made it with the planning tools "
          "only (his whole strategy, the list of what may change, his documents); code checked it and ran exactly it.",
          "", f"**What must be true if his strategy works:** {plan['hypothesis']}", "",
          f"**When it stops:** {plan['stop_rule']['in_words']}", ""]
    L += ["## Every version on 2016–2020 (each one is a try)", "",
          "| # | version | what it changes | how often the change acted | with his hedge: trades (open at end) | "
          "₹ with his hedge | worst year | ₹ with a plain stop (trades) |", "|---|---|---|---|---:|---:|---|---:|"]
    v1 = fx["runs"].get(W.version({}).version_id)
    same_as_1 = []
    for p in plan["versions"]:
        f = fx["runs"].get(p["version_id"])
        mark = " **← chosen**" if p["version_id"] == chosen else ""
        if f is None:
            L.append(f"| {p['k']} | {p['name']}{mark} | {p['about']} | | did not run | | | |")
            continue
        h, s = f["result"]["hedged"], f["result"]["stop"]
        w = h["worst_year"]
        acted = _acted(p, h)
        if v1 is not None and p["changes"] and f["result"]["hedged_trades"] == v1["result"]["hedged_trades"]:
            same_as_1.append(str(p["k"]))
        L.append(f"| {p['k']} | {p['name']}{mark} | {p['about']} | {acted} | {h['trades']} ({h['open_at_end']}) | "
                 f"{inr(h['net_inr_total'])} | {w['year']} {inr(w['net_inr'])} | {inr(s['net_inr_total'])} "
                 f"({s['trades']}) |")
    L += ["", "The plain-stop column is the same for every version that changes only his management: a plain stop has "
          "no hedge to manage."]
    if same_as_1:
        L.append(f"Versions {', '.join(same_as_1)} made exactly the same trades as version 1: their rule never "
                 "changed a trade of these years. They still count as tries (each is a different rule).")
    if fx["tune"]:
        t = fx["tune"]["result"]
        L += ["", f"The referee chose ONE: the highest score for \"how likely the edge is real, allowing for every try\" "
              f"(Deflated Sharpe) of the hedged trades, counting {_n(t['tries'], 'try')}. Never the agent's pick, "
              "never the most rupees."]
        if t.get("left_out_too_few"):
            L.append(f"Left out of the choice (fewer than 2 closed trades): {', '.join(t['left_out_too_few'])}.")
    L.append("")
    if fx["judged"]:
        j = fx["judged"]["result"]
        L += [f"## The chosen version, judged once on {config.CHECK_YEARS[0]}–{config.CHECK_YEARS[-1]}", "",
              "The referee's 6 checks (the same for every idea, fixed before any result):", ""]
        for c in j["answer"]["checks"]:
            L.append(f"{c['number']}. **{c['name']}: {'yes' if c['passed'] else 'NO'}** — {c['shown']}")
        L += ["", f"The check years opened for this family: once ({fx['opening']['opened_at'][:16]} by "
              f"{fx['opening']['who']}). In the whole project they were opened {fx['opened_count']} times.", ""]
    m = _merged(fx, GROUP_ALL)
    if m:
        L += _boxes_md(m, "## Year by year: the four boxes (₹ after cost)")
        L += ["", "![his strategy year by year](whole-years.jpg)", "", "![the four boxes, all years](whole-boxes.jpg)",
              ""]
    L += _entries_md(fx)
    L += _slices_md(fx)
    L += _guard_md(fx, m)
    L += ["## Already seen", "", seen.text(), "",
          "Each entry alone: its own page below, and L19b's page for the entry alone on 2016–2020 "
          "(`plan/work/L19b/`).", "",
          "## The referee's rules (fixed before any result was seen)", "", "```", verdict.RULES, "```", ""]
    return "\n".join(L) + "\n"


def _boxes_md(m: dict, head: str) -> list[str]:
    ys = m["years"]
    L = [head, "", "| box | " + " | ".join(ys) + " | all years | trades | open at end |", "|---|" + "---:|" * (len(ys) + 3)]
    for bx in m["boxes"]:
        L.append(f"| {bx['words']}{' (middle run)' if bx['random'] else ''} | "
                 + " | ".join(inr(bx["per_year"][y]) for y in ys)
                 + f" | **{inr(bx['total'])}** | {bx['trades']:g} | {bx['open_at_end']:g} |")
    L += ["", f"Random boxes: the middle one of {m['runs']} random runs (half did better, half worse)."
          + (" All years: the random runs of each window added up run by run." if m.get("windows", 2) > 1 else "")]
    if not m["boxes"][0]["trades"]:
        return L + ["- With his hedging these setups opened no trade (a trade of his was always on): nothing to compare "
                    "with his hedge.",
                    f"- With a plain stop: more than **{m['his_beats_random_stop']} of {m['runs']}** random runs."]
    L += [f"- With his hedging, his entries made more than **{m['his_beats_random_hedging']} of {m['runs']}** random "
          "runs, all years. By year: " + ", ".join(f"{y} {v}" for y, v in m["beats_by_year"].items()) + ".",
          f"- With a plain stop: more than **{m['his_beats_random_stop']} of {m['runs']}**.",
          f"- His hedging did better than a plain stop in **{m['hedging_beats_stop_in_random_runs']} of {m['runs']}** "
          "random runs."]
    return L


SHORT = {forms.TYPE1.id: "Type 1", forms.ZONE_BREAK.id: "the zone break", forms.ZONE_ENTRY.id: "the zone entry",
         forms.SECOND_GAP.id: "the second gap", forms.TRENDLINE_TOUCH.id: "the trendline touch",
         forms.TRIANGLE_FALSE_BREAK.id: "the triangle false break"}
ACTED_WORDS = {"real_stops": "a real stop", "cap_stops": "the cap of hedges reached", "friday": "a Friday entry",
               "escapes": "the trend escape", "zone_hedges": "a hedge at a zone", "zone_small": "a target cut back",
               "dbl_booked": "a booking at a zone", "dbl_back": "back in after a booking",
               "split_holds": "held with the hour and day split"}


def _acted(p: dict, h: dict) -> str:
    """How often the version's change acted (the hedging tester counts each test row of his table)."""
    if not p["changes"]:
        return "-"
    if "trades_at_once" in p["changes"]:
        return "every setup of each entry type"
    did = h.get("test_rows_acted") or {}
    if p["changes"].get("exit.management.news_events"):
        return "at news (fewer news moments)"
    if not did:
        return "never"
    return ", ".join(f"{ACTED_WORDS.get(k, k)} {v}" for k, v in did.items())


def _entries_md(fx: dict) -> list[str]:
    L = ["## Each entry type's part of the whole", "",
         "A trade belongs to the entry that OPENED it (after a close it is handed to his next setup, M15 - it stays "
         "that entry's trade).", "",
         "| entry | setups | trades it opened, with his hedge | setups skipped while a trade was on | ₹ with his hedge "
         "| ₹ with a plain stop | his beat random (with his hedge, of runs) |", "|---|---:|---:|---:|---:|---:|---|"]
    none = []
    for eid in fx["plan"]["entries_in"]:
        m = _merged(fx, eid)
        e = forms.get(eid)
        setups = _setups(fx, eid)
        if m is None:
            continue
        skipped = sum(_by_entry(f)[eid]["skipped_hedged"] for f in (fx["tune_2x2"], fx["judged"]) if f is not None)
        n = m["boxes"][0]["trades"]
        if not n:
            none.append(SHORT.get(eid, eid))
        beat = f"{m['his_beats_random_hedging']} of {m['runs']}" if n else "no trade opened - nothing to compare"
        L.append(f"| [{e.title}]({eid}.md) | {setups} | {n:g} | {skipped} | {inr(m['boxes'][0]['total'])} "
                 f"| {inr(m['boxes'][1]['total'])} | {beat} |")
    if none:
        L += ["", f"**What this shows:** {' and '.join(none).capitalize()} never opened a trade with his hedge. A hedged trade stays "
              "open for weeks, Type 1 makes many setups a week, and his rule is one trade at a time across all of "
              "them - so when a trade ends, the next setup that comes is almost always a Type 1. Inside his whole "
              "strategy, the other entries only ever close legs (as his setups, M10); with a plain stop they do trade. "
              "Version 9 (one trade at a time PER entry type) is the only version where they open trades of their "
              "own."]
    return L + [""]


def _by_entry(f: dict) -> dict:
    """{entry id: its part} from a 2x2 file (kind strategy-2x2) or the referee's file."""
    r = f["result"]
    return r["by_entry"] if "by_entry" in r else r["summary"]["by_entry"]


def _hedged(f: dict) -> dict:
    """His hedge's summary (hedging.summarize) from a 2x2 file or the referee's file."""
    r = f["result"]
    return r["summary"]["hedged"] if "hedged" in r["summary"] else r["summary"]


def _setups(fx: dict, eid: str) -> int:
    return sum(_by_entry(f)[eid]["setups"] for f in (fx["tune_2x2"], fx["judged"]) if f is not None)


def _slices_md(fx: dict) -> list[str]:
    L = ["## The slices (chosen in the plan, before any money)", ""]
    sl = (fx["tune_2x2"] or {}).get("result", {}).get("slices") if fx["tune_2x2"] else None
    if not fx["plan"]["slices"]:
        n = sum(_hedged(f)["trades"] for f in (fx["tune_2x2"], fx["judged"]) if f is not None)
        return L + [f"The plan chose no slice. (Claude's first answer named its slices in its own words, not by the "
                    f"machine's names; code refused them; on its one more try it left them out.) Nothing is lost: a "
                    f"slice is tested only with at least {config.SLICE_MIN_TRADES} trades in a part and as many in the "
                    f"rest, and his hedged trades are {n} in all.", ""]
    if not sl:
        return L + ["Not measured yet.", ""]
    for r in sl.get("slices", []):
        t = r.get("test") or {}
        word = t.get("claim") or ("too few trades to test" if not t.get("tested") else "no difference that passed "
                                                                                          "the 3 checks")
        L.append(f"- **{r['label']}** ({r['dimension']}): {word}.")
        for g in r.get("groups", []):
            pt = g.get("per_trade")
            L.append(f"  - {g['group']}: {_n(g['trades'], 'trade')}" + (f", {inr(pt)} a trade" if pt is not None
                                                                        else ""))
    c = sl.get("counts", {})
    L += ["", f"Every other split was looked at, not tested ({c.get('looked_at', 0)} of them) - nothing in them is a "
          "finding.", ""]
    return L


def _guard_md(fx: dict, m: dict | None) -> list[str]:
    L = ["## The judgement guard (#114)", ""]
    if m is None:
        return L + ["Nothing judged yet.", ""]
    b = m["boxes"]
    days = [f"{_hedged(f)['days_open_median']:g} days in {w}" for f, w in ((fx["tune_2x2"], "2016–2020"),
            (fx["judged"], "2021–2023")) if f is not None and _hedged(f)["days_open_median"] is not None]
    L.append(f"1. **Same risk?** All four boxes trade the same months, directions and distances, one trade at a time; "
             f"between two boxes side by side only one thing changes. With his hedge his entries took "
             f"{_n(b[0]['trades'], 'trade')}, with a plain stop {_n(b[1]['trades'], 'trade')}: a hedged trade stays "
             f"open for weeks (the middle one: {', '.join(days)}), so the "
             "two managements do not take the same number of trades. That is most of why his hedging beats a plain "
             f"stop in {m['hedging_beats_stop_in_random_runs']} of {m['runs']} random runs: far fewer trades, far less "
             "cost - and it still loses.")
    L.append(f"   **His entries against random entries, both with his hedge:** his made more than "
             f"{m['his_beats_random_hedging']} of {m['runs']} random runs over all the years. Two readings: (a) his "
             "entry is a worse start than a random one at the same distances; (b) luck - few trades, and after a "
             "trade's first close it is handed to his next setup (M15), so his trades and the random ones end much "
             "the same way (L19a found the same). Year by year his entries beat the middle of the random runs in "
             f"{sum(v > m['runs'] / 2 for v in m['beats_by_year'].values())} of {len(m['beats_by_year'])} years, "
             f"with {_n(b[0]['trades'], 'trade')} in all. The total leans to (a), the years are mixed, so neither is "
             "sure. What both readings agree on: once the trade is hedged, his entry does not add money.")
    worst = min(m["years"], key=lambda y: b[0]["per_year"][y])
    L.append(f"2. **Bad years:** with his hedge the worst year is {worst} ({inr(b[0]['per_year'][worst])}). "
             f"Years with a profit: {sum(b[0]['per_year'][y] > 0 for y in m['years'])} of {len(m['years'])}.")
    L.append(f"3. **Tries:** {_n(fx['budget']['used'], 'version')} of his whole strategy here; the old project's "
             f"{prior.already_seen(W.SEEN_KEY, config.TUNE_YEARS + config.CHECK_YEARS).tries:,} on the same years.")
    for label, f in (("2021–2023", fx["judged"]),):
        if f is None:
            continue
        wb = f["result"]["without_best"]
        if wb.get("best_inr"):
            L.append(f"4. **How much is 3 trades?** On {label} the best 3 trades made {inr(sum(wb['best_inr']))}; "
                     f"without them the other {_n(wb['trades'], 'trade')} made {inr(wb['net_inr'])}.")
    L.append("5. **What would make it wrong:** his answer to 'the setup is good' (when the hedge gets room), the "
             "zones and lines drawn a little differently, and above all new prices - these years were already seen. "
             "(The three entries left out cannot change it: see 'What his whole strategy is here'.)")
    return L + [""]


# ================================================================================================= each entry

def entry_md(fx: dict, e: forms.EntryType) -> str:
    plan = fx["plan"]
    years = config.TUNE_YEARS + config.CHECK_YEARS
    L = [f"# {e.title}", "", "> Part of his whole strategy (L19d) · made by `python -m machine.strategy reports` · MT5 "
         "server time · 5-minute chart · 0.01 lot a leg · ₹40 a position, all-in.", "",
         f"His words: *\"{e.idea().words}\"*", ""]
    if e.id not in plan["entries_in"]:
        L += [f"**Not in the whole:** {W.waiting().get(e.id, 'it is not in it')}.", "",
              "No money is shown for its readings here (lesson 16: a meaning is chosen before money is seen; its money "
              "alone is on L19b's page). What each reading finds (setups only, from L19b's counts on 2016–2020):", "",
              "| reading | means | setups 2016–2020 | a week |", "|---|---|---:|---:|"]
        for c in _l19b_counts(e):
            name = ", ".join(f"reading {v}" for v in c["readings"].values()) or "as written"
            means = re.sub(r"^B\d-\d+ [A-Z]: ", "", c["means"])          # his words, not a question code (#113)
            L.append(f"| {name} | {means} | {c['signals']} | {c['signals_a_week']} |")
        L += ["", f"The entry alone: `plan/work/L19b/{e.id}.md`.", "", prior.already_seen(e.seen, years).text(), ""]
        return "\n".join(L) + "\n"
    m = _merged(fx, e.id)
    L += ["**In the whole: yes.** Below: the trades this entry OPENED inside his whole strategy (the chosen version), "
          "one trade at a time with his other entries - a setup of his while a trade is on is skipped, whichever entry "
          "it is.", ""]
    for label, f in (("2016–2020", fx["tune_2x2"]), ("2021–2023", fx["judged"])):
        if f is None:
            continue
        be = _by_entry(f)[e.id]
        h, s = be["hedged"], be["stop"]
        L.append(f"- **{label}:** {be['setups']} setups · with his hedge it opened {_n(h['trades'], 'trade')} "
                 f"({inr(h['net_inr_total'])}; {be['skipped_hedged']} setups skipped while a trade was on) · with a "
                 f"plain stop {_n(s['trades'], 'trade')} ({inr(s['net_inr_total'])}).")
    if m:
        L += [""] + _boxes_md(m, "## Its four boxes inside the whole (₹ after cost)")
        L += ["", f"Random entries here = the random twins of THIS entry's setups, inside the same random runs of the "
              f"whole.", "", f"![its four boxes]({e.id}-boxes.jpg)", ""]
    L += ["## Already seen", "", prior.already_seen(e.seen, years).text(), "",
          f"The entry ALONE (one trade at a time with itself only), 2016–2020: `plan/work/L19b/{e.id}.md`.", ""]
    return "\n".join(L) + "\n"


def _l19b_counts(e: forms.EntryType) -> list[dict]:
    folder = config.WORK_DIR / "L19b" / e.id
    out = []
    for p in sorted(folder.glob("*-counts.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"readings": d.get("readings", {}), "means": d.get("means", ""), "signals": d.get("signals"),
                    "signals_a_week": d.get("signals_a_week")})
    return sorted(out, key=lambda c: json.dumps(c["readings"]))


# ================================================================================================= write them all

def final_message(fx: dict) -> str:
    """The research's final message: the answer + every version (2016-2020 only), the chosen one marked."""
    plan = fx["plan"]
    lines = [f"His whole strategy - the research loop (L19d). {_ref_line(fx).replace('**', '')}"]
    if fx["judged"]:
        s = fx["judged"]["result"]["summary"]
        lines.append(f"2021-2023 (judged once, with his hedge): {s['trades']} trades, {inr(s['net_inr_total'])} after "
                     f"cost; worst year {s['worst_year']['year']} {inr(s['worst_year']['net_inr'])}.")
    lines.append(f"The plan was saved {plan['made_at'][:16].replace('T', ' ')}, before any money (fingerprint "
                 f"{plan['sha256'][:12]}). It said: {plan['hypothesis']}")
    lines.append(f"Every version tried (2016-2020 only; rupees at 0.01 lot a leg, ₹40 a position all-in; "
                 f"{_n(len(fx['runs']), 'try')}):")
    for p in plan["versions"]:
        f = fx["runs"].get(p["version_id"])
        if f is None:
            lines.append(f"{p['k']}. {p['name']}: did not run (research-log.md says why)")
            continue
        h = f["result"]["hedged"]
        lines.append(f"{p['k']}. {p['name']}: {h['trades']} trades with his hedge, {inr(h['net_inr_total'])}"
                     + (" - CHOSEN by the referee" if p["version_id"] == fx["chosen"] else ""))
    lines.append("The reports: plan/work/L19d/README.md and one page per entry type.")
    return "\n".join(lines)


def out_dir() -> Path:
    return Path(config.WORK_DIR) / "L19d"


def write_all() -> list[Path]:
    from machine.report.build import check_words
    fx = facts()
    OUT = out_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    m = _merged(fx, GROUP_ALL)
    if m:
        p = OUT / "whole-years.jpg"
        picture_years(m, p, "His whole strategy, year by year")
        paths.append(p)
        p = OUT / "whole-boxes.jpg"
        picture_boxes(m, p, "His whole strategy: the four boxes",
                      "his entries together, one trade at a time, his hedge and his management")
        paths.append(p)
    for e in forms.ENTRY_TYPES.values():
        if e.id in fx["plan"]["entries_in"]:
            me = _merged(fx, e.id)
            if me:
                p = OUT / f"{e.id}-boxes.jpg"
                picture_boxes(me, p, e.title.split(":")[0] + ", inside his whole strategy",
                              "the trades this entry opened, one trade at a time with his other entries")
                paths.append(p)
        p = OUT / f"{e.id}.md"
        text = entry_md(fx, e)
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    text = whole_md(fx)
    problems = [x for x in check_words(text.split("## The referee's rules")[0])]
    if problems:
        raise ReportError("the report's words: " + "; ".join(problems[:5]))
    p = OUT / "README.md"
    p.write_text(text, encoding="utf-8")
    paths.append(p)
    final = final_message(fx)
    SR._write(SR.folder() / SR.FINAL, final + "\n")
    paths.append(SR.folder() / SR.FINAL)
    return paths
