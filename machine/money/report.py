"""The money plan's report (L20): runs machine/money/plan.py and writes what Tejas reads.

    .venv\\Scripts\\python -m machine.money.plan                 # idea #1's plain trades 2016-2023
    .venv\\Scripts\\python -m machine.money.plan --runs 50 --no-write

Writes (write=True):
  plan/work/money-plan.md            the report, easy English (every sentence passes report.build.check_words)
  plan/work/money-plan/account.jpg   the account in the real order: its money, the blow-ups, the doubles
  plan/work/money-plan/orders.jpg    the same months in 1,000 other orders: where your money ended
  plan/work/money-plan/*.csv         every trade with the plan's lots, every blow-up and double, how many trades at
                                     each lot size, the Monday numbers, the dips inside the hedged trades
  plan/work/trials/<id>.json         ONE trial file, kind "money-plan" (version 1 + the 9 one-change rows)
The words BLOW-UP, REFILL, a double: plan/work/L20-notes.md section 1 (fixed before the first run on real trades).
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from machine import config, trials  # noqa: E402
from machine.charts import draw as D  # noqa: E402
from machine.costs import inr  # noqa: E402
from machine.money import plan as MP  # noqa: E402
from machine.report.build import check_words  # noqa: E402

plt.rcParams["text.parse_math"] = False           # lesson B8: "$5" is five dollars, not maths

OUT_MD = config.WORK_DIR / "money-plan.md"
OUT_DIR = config.WORK_DIR / "money-plan"
NOTES = "plan/work/L20-notes.md"
MD_LABEL = "PLAIN TRADES — NOT YOUR HEDGING"      # MP.LABEL ("... NOT HIS HEDGING") said to him
HIS_DAYS = 8                                      # his "doubles in ~7-8 days" (strategy.md line 67, section 12)
HIS_BLOWUPS = (2, 4)                              # capital-plan.md: "2-4 full blow-ups a year accepted"
SOON = 7                                          # "right after a refill": the trade opened within 7 days of one
ROW_WORDS = {                                     # the rows in the report's own words (Plan.change is the code's)
    "row 1": "take the profit out at +75%, not at a double",
    "row 2": "the account 5% of your money (₹10,000 in, ₹190,000 outside)",
    "row 3": "the account 20% of your money (₹40,000 in, ₹160,000 outside)",
    "row 4": "never take the profit out",
    "row 5": "no chocolate rule: 0.01 lot always",
    "row 6": "your sheet's number: ₹7,100 for 0.01 lot every week, not 2 × Monday's ADR",
    "row 7": "refill only up to the money needed for 0.01 lot",
    "row 8": "a blow-up only when the account reaches ₹0 (below the line it still trades 0.01 lot)",
    "row 9": "NOT REAL: before the ₹40 cost",
}


class MoneyReportError(ValueError):
    """The report's own words broke the easy-English rules."""


def signed(x: float) -> str:
    """+₹1,234 / -₹1,234 (a plain ₹0 has no sign)."""
    return inr(x) if round(x) <= 0 else "+" + inr(x)


def pc(x: float) -> str:
    return f"{x * 100:+.1f}%"


def num(x, fmt: str = ".0f") -> str:
    return "none" if x is None or not np.isfinite(x) else format(x, fmt)


def days_list(days) -> str:
    d = sorted(int(round(x)) for x in days)
    if not d:
        return "none"
    if len(d) == 1:
        return f"{d[0]} days"
    return ", ".join(str(x) for x in d[:-1]) + f" and {d[-1]} days"


def _rel(path: Path) -> str:
    try:
        return Path(path).relative_to(config.ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ================================================================================================================
# the run
# ================================================================================================================

def run(n_runs: int = MP.RESHUFFLES, write: bool = True, log=print) -> dict:
    """Idea #1's plain trades (made again, checked against the recorded runs) through the plan; write=True also
    writes the report, its pictures and tables, and the one trial file."""
    from machine.data import bars as BARS
    frame, info, m5 = MP.idea_trades(log)
    stories = [json.loads(line) for line in MP.HEDGED_STORIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    m1 = BARS.load_bars("1m", str(config.TUNE_YEARS[0]), str(config.TUNE_YEARS[-1]))
    data = {"prices": _rel(config.GOLD_M1), "gold_m1_content_sha256": BARS.m1_fingerprint()}
    out = build(frame, info, MP.weekly_need(m5), stories, m1, n_runs, log, data=data)
    if write:
        publish(out, log=log)
    return out


def build(frame: pd.DataFrame, info: dict, needs: pd.DataFrame, stories: list[dict], m1: pd.DataFrame,
          n_runs: int = MP.RESHUFFLES, log=print, data: dict | None = None) -> dict:
    """Every number of the report (no file is written). data: the fingerprint of the prices the trades came from
    (the trial file records it)."""
    t0 = time.time()
    trades, dropped = MP.attach_need(frame, needs)
    ts = MP.TradeSet.build(trades, last_month=f"{config.CHECK_YEARS[-1]}-12")
    real, shuf = MP.real_order(ts), MP.reshuffled(ts, n_runs, MP.SEED)
    log(f"{len(ts)} trades in the plan ({dropped} before the first Monday number); {len(needs)} Monday numbers; "
        f"{n_runs} reshuffles of {ts.n_months} months, seed {MP.SEED}")

    rows = []
    for plan in (MP.VERSION_1, *MP.EXTRA_ROWS):
        one = MP.simulate(ts, real, plan, trace=True)
        many = MP.simulate(ts, shuf, plan)
        rows.append({"plan": plan, "one": one, "many": many, "head": MP.headline(one), "spread": MP.spread(many)})
        log(f"  {plan.name}: real order {signed(rows[-1]['head']['result'])}, the middle of {n_runs} orders "
            f"{signed(rows[-1]['spread']['all_years']['result_mid'])}")
    v1 = rows[0]
    one, many = v1["one"], v1["many"]
    tr = one.trace.copy()
    f = ts.frame
    for col in ("signal", "direction", "open", "close", "closed_by", "monday", "net_inr", "gross_inr", "cost_inr"):
        tr[col] = f[col].to_numpy()[tr["trade"]] if col in f.columns else None
    steps = np.round(tr["lots"].to_numpy() / config.LOT_STEP)
    refills = np.array([e["t"] for e in one.events if e["what"] == "blow-up"])
    best = []
    for r in tr.sort_values("inr", ascending=False).head(3).itertuples():
        opened = float(ts.open_d[r.trade])
        since = opened - refills[refills <= opened + 1e-9]
        best.append({"open": pd.Timestamp(r.open), "lots": float(r.lots), "at_001": float(r.net_inr),
                     "inr": float(r.inr), "price_move": float(round(r.lots / config.LOT_STEP) * r.gross_inr),
                     "days_after_refill": float(since.min()) if len(since) else None})
    year_of_open = pd.DatetimeIndex(f["open"]).year
    facts = {
        "trades": len(ts), "dropped": dropped, "first_open": f["open"].iloc[0], "last_close": f["close"].iloc[-1],
        "flat_total": float(f["net_inr"].sum()), "flat_gross": float(f["gross_inr"].sum()),
        "flat_cost": float(f["cost_inr"].sum()),
        "flat_by_year": {int(y): float(v) for y, v in f.groupby(year_of_open)["net_inr"].sum().items()},
        "plan_gross": float((steps * tr["gross_inr"].to_numpy()).sum()),
        "plan_cost": float((steps * tr["cost_inr"].to_numpy()).sum()),
        "lot_max": float(tr["lots"].max()), "lot_small": int((tr["lots"] <= 0.02 + 1e-9).sum()),
        "best3": best, "best3_sum": float(sum(b["inr"] for b in best)),
        "best3_price_moves": float(sum(b["price_move"] for b in best)),
        "lots_avg": float(steps.mean()),
        "orders_better": int((many.result > one.result[0] + 1e-6).sum()),
        "orders_worse": int((many.result < one.result[0] - 1e-6).sum()),
        "orders_best": float(many.result.max()), "orders_made_money": int((many.result > 0).sum()),
        "refills": [float(e["money_in_out"]) for e in one.events if e["what"] == "blow-up"],
        "doubles_in_his_days": int((many.dbl_days <= HIS_DAYS).sum()), "doubles_all": int(len(many.dbl_days)),
        "runs_with_double_by_year": [int((many.doubles[:, y] > 0).sum()) for y in range(many.paths.n_years)],
        "need_min": float(needs["need_inr"].min()), "need_max": float(needs["need_inr"].max()),
    }
    monday_year = pd.DatetimeIndex(needs["day"]).year
    facts["need_by_year"] = {int(y): (float(g.min()), float(g.max()))
                             for y, g in needs.groupby(monday_year)["need_inr"]}
    mean_by_year = needs.groupby(monday_year)["need_inr"].mean()
    facts["quiet_years"] = sorted(int(y) for y in mean_by_year.nsmallest(3).index)    # where the rule buys most
    worst = MP.simulate(ts, shuf.one(v1["spread"]["worst"]["run"]), MP.VERSION_1, trace=True)
    dips = MP.hedged_dips(stories, m1, needs) if stories else pd.DataFrame()
    out = {"data": data or {"prices": "given to build()"}, "info": info, "facts": facts, "rows": rows,
           "years": MP.year_rows(one, ts), "trace": tr, "ts": ts, "needs": needs, "dips": dips, "worst": worst,
           "n_runs": n_runs, "seconds": round(time.time() - t0)}
    out["text"] = summary_text(out)
    return out


def publish(out: dict, *, md_path: Path = OUT_MD, folder: Path = OUT_DIR, record: bool = True, log=print) -> None:
    """The pictures, the tables, the trial file (record=True) and the report. The report's words are checked FIRST:
    a report that cannot be written leaves no picture, no table and no trial file behind."""
    _checked(report_md(out))
    folder.mkdir(parents=True, exist_ok=True)
    out["pictures"] = [account_picture(out, folder / "account.jpg"), orders_picture(out, folder / "orders.jpg")]
    _write_tables(out, folder)
    if record:
        out["trial"] = _record(out).name
    md = _checked(report_md(out))
    tmp = md_path.with_name(md_path.name + ".tmp")
    tmp.write_text(md, encoding="utf-8", newline="\n")
    tmp.replace(md_path)
    log(f"wrote {_rel(md_path)}, {_rel(folder)}/" + (f" and trial {out['trial']}" if record else ""))


def _checked(md: str) -> str:
    problems = check_words(md)
    if problems:
        raise MoneyReportError("the report's words: " + "; ".join(problems[:6]))
    return md


def summary_text(out: dict) -> str:
    h, s, fx = out["rows"][0]["head"], out["rows"][0]["spread"], out["facts"]
    a = s["all_years"]
    return "\n".join([
        f"MONEY PLAN (L20) - {MP.LABEL}: idea #1 {MP.VERSION}, {fx['trades']} trades 2016-2023, Rs 40 a position",
        f"  real order : your Rs 200,000 -> {inr(h['your_money_end'])} ({signed(h['result'])}); {h['doubles']} doubles "
        f"({h['days']}); {h['blowups']} blow-ups; taken out {inr(h['out'])}, added back {inr(h['added'])}",
        f"  {s['runs']} orders: the middle {signed(a['result_mid'])}, the bad 10% {signed(a['result_bad'])}; "
        f"{fx['orders_worse']} did worse than the real order; blow-ups {a['blowups_mid']:.0f} (bad 10% "
        f"{a['blowups_bad']:.0f}); the reserve used up in {a['runs_reserve_used_up']} runs",
        f"  0.01 lot always: {signed(fx['flat_total'])}",
    ])


# ================================================================================================================
# the pictures (each draws every thing it names; one chart each)
# ================================================================================================================

def _save(fig, path: Path, texts: list[str]) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", dpi=D.DPI)
    plt.close(fig)
    buf.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.open(buf).convert("RGB").save(tmp, format="JPEG", quality=92, optimize=True)
    tmp.replace(path)
    problems = [p for t in texts for p in check_words(t)]
    if problems:
        raise MoneyReportError("the picture's words: " + "; ".join(problems))
    return {"path": _rel(path), "texts": texts, "pixel_sha256": D.pixel_sha256(path)}


def account_path(out: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The account's money through the real order: after every trade, and after every withdrawal and refill."""
    one, ts, tr = out["rows"][0]["one"], out["ts"], out["trace"]
    pts = [(ts.open_d[0], one.plan.account_size, 0)]
    pts += [(ts.close_d[k], a, 1) for k, a in zip(tr["trade"], tr["account_after_trade"])]
    pts += [(e["t"], e["account_after"], 2) for e in one.events if e["what"] != "the reserve is used up"]
    pts.sort(key=lambda p: (p[0], p[2]))
    return pd.DatetimeIndex([MP.when(p[0]) for p in pts]), np.array([p[1] for p in pts])


def account_picture(out: dict, path: Path) -> dict:
    """ONE chart: the money in the account, the money needed for 0.01 lot under it, each blow-up and each double."""
    one, tr, ts, h = out["rows"][0]["one"], out["trace"], out["ts"], out["rows"][0]["head"]
    when, money = account_path(out)
    size, double = one.plan.account_size, one.plan.double_at * one.plan.account_size
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.9), dpi=D.DPI)
    texts = []
    ax.step(when, money, where="post", color=D.INK, linewidth=1.3, zorder=3)
    need_t = pd.DatetimeIndex(ts.frame["open"].to_numpy()[tr["trade"]])
    ax.step(need_t, tr["need_inr"], where="post", color=D.RED, linewidth=1.1, zorder=2)
    ax.axhline(size, color=D.MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(double, color=D.GREEN, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ev = [e for e in one.events if e["what"] == "blow-up"]
    dv = [e for e in one.events if e["what"] not in ("blow-up", "the reserve is used up")]
    ax.scatter([MP.when(e["t"]) for e in ev], [e["account_before"] for e in ev], marker="v", s=95, color=D.RED,
               zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter([MP.when(e["t"]) for e in dv], [e["account_before"] for e in dv], marker="^", s=105, color=D.GREEN,
               zorder=5, edgecolors="white", linewidths=0.8)
    ax.set_ylim(0, max(float(money.max()), double) * 1.30)
    x0 = when[0] - pd.Timedelta(days=40)
    x1 = pd.Timestamp(f"{config.CHECK_YEARS[-1] + 1}-01-01") + pd.Timedelta(days=40)
    ax.set_xlim(x0, x1)

    def put(y, s, colour):
        texts.append(s.strip())
        ax.text(x1, y, s, ha="left", va="center", fontsize=D.F_AXIS, color=colour, clip_on=False)

    put(size, f" {inr(size)}\n the full account", D.MUTED)
    put(double, f" {inr(double)}\n a double", D.GREEN)
    put(float(tr["need_inr"].iloc[-1]), " money needed\n for 0.01 lot", D.RED)
    key = [(D.INK, "line", "the money in the account"),
           (D.GREEN, "^", f"a double ({h['doubles']}): the profit is taken out, {inr(h['out'])} in all"),
           (D.RED, "v", f"a blow-up ({h['blowups']}): under the red line, refilled from the reserve, "
                        f"{inr(h['added'])} in all")]
    handles = []
    for colour, mark, words in key:
        texts.append(words)
        if mark == "line":
            handles.append(plt.Line2D([], [], color=colour, linewidth=1.6, label=words))
        else:
            handles.append(plt.Line2D([], [], color=colour, marker=mark, linestyle="", markersize=9, label=words))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=D.F_AXIS + 0.5,
              labelcolor=D.INK, handlelength=1.6, borderaxespad=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="both", labelsize=D.F_AXIS, colors=D.MUTED)
    first_years = range(when[0].year, config.CHECK_YEARS[-1] + 1)               # a tick at each 1 January; no tick
    ax.set_xticks([pd.Timestamp(year=y, month=1, day=1) for y in first_years],   # for the year after the last one
                  [str(y) for y in first_years])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=D.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    title = "The account at the broker, 2016-2023, the real order"
    sub = ("your money plan as written · idea #1's plain trades (a stop at the hedge point, NOT your hedging) · "
           "₹40 a position per 0.01 lot")
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=D.F_HEAD, color=D.INK, fontweight="bold")
    fig.text(0.06, 0.905, sub, ha="left", fontsize=D.F_AXIS, color=D.MUTED)
    texts += [title, sub]
    fig.subplots_adjust(left=0.085, right=0.865, top=0.85, bottom=0.08)
    return _save(fig, path, texts)


def orders_picture(out: dict, path: Path) -> dict:
    """ONE chart: where your money ended in each of the reshuffled orders, and the real order among them."""
    v1, fx = out["rows"][0], out["facts"]
    one, many = v1["one"], v1["many"]
    end = many.your_money_end
    real_end = float(one.your_money_end[0])
    start = one.plan.your_money
    flat_end = start + fx["flat_total"]
    fig, ax = plt.subplots(figsize=(D.WIDTH_IN, 5.8), dpi=D.DPI)
    texts = []
    lo = min(float(end.min()), real_end) - 8_000
    hi = max(start, float(end.max()), real_end, flat_end) + 8_000
    bins = np.arange(np.floor(lo / 5_000) * 5_000, np.ceil(hi / 5_000) * 5_000 + 1, 5_000)
    counts, _, _ = ax.hist(end, bins=bins, color="#b7c0ca", edgecolor="white", linewidth=0.6, zorder=2)
    top = float(counts.max()) * 1.55
    ax.set_ylim(0, top)
    ax.set_xlim(bins[0], bins[-1])

    def mark(x, colour, words, level, ha):
        ax.axvline(x, color=colour, linewidth=1.8, zorder=4, ymax=level)
        texts.append(words)
        ax.text(x, top * (level + 0.012), words, ha=ha, va="bottom", fontsize=D.F_AXIS + 0.5, color=colour,
                fontweight="bold", zorder=5)

    mid = float(np.percentile(end, 50))
    mark(mid, D.RED, f"the middle order\n{inr(mid)}", 0.72, "center")
    mark(real_end, D.INK, f"the real order\n{inr(real_end)}", 0.72, "center")
    mark(flat_end, D.BLUE, f"0.01 lot always, every order\n{inr(flat_end)}", 0.86, "center")
    mark(start, D.GREEN, f"the start\n{inr(start)}", 0.72, "right")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: inr(v)))
    ax.tick_params(axis="both", labelsize=D.F_AXIS, colors=D.MUTED)
    ylab, xlab = "how many of the orders", "your money at the end of the 8 years"
    ax.set_ylabel(ylab, fontsize=D.F_AXIS + 0.5, color=D.MUTED)
    ax.set_xlabel(xlab, fontsize=D.F_AXIS + 0.5, color=D.MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=D.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    title = f"The same months in {many.paths.P:,} other orders"
    sub = (f"grey bars = the {many.paths.P:,} orders · {fx['orders_worse']:,} of them ended with less than the real "
           f"order\nyour money plan as written · idea #1's plain trades (NOT your hedging) · ₹40 a position per "
           f"0.01 lot")
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=D.F_HEAD, color=D.INK, fontweight="bold")
    fig.text(0.06, 0.925, sub, ha="left", va="top", fontsize=D.F_AXIS, color=D.MUTED, linespacing=1.5)
    texts += [title, sub, xlab, ylab]
    fig.subplots_adjust(left=0.085, right=0.975, top=0.82, bottom=0.115)
    return _save(fig, path, texts)


# ================================================================================================================
# the files behind the report
# ================================================================================================================

def _write_tables(out: dict, folder: Path) -> None:
    tr, one = out["trace"], out["rows"][0]["one"]
    t = pd.DataFrame({
        "signal": tr["signal"], "direction": tr["direction"], "opened": tr["open"], "closed": tr["close"],
        "closed_by": tr["closed_by"], "monday": tr["monday"], "money_needed_for_0.01_lot": tr["need_inr"].round(2),
        "inr_at_0.01_lot": tr["net_inr"].round(2), "lots": tr["lots"], "inr_at_that_lot": tr["inr"].round(2),
        "lowest_point_inside_inr": tr["lowest_inside_inr"].round(2),
        "account_after_the_trade": tr["account_after_trade"].round(2), "taken_out": tr["taken_out"].round(2)})
    _csv(t, folder / "trades.csv")
    ev = pd.DataFrame(one.events)
    if len(ev):
        ev.insert(0, "when", [MP.when(x).round("s") for x in ev.pop("t")])
    _csv(_rounded(ev, 2), folder / "events.csv")
    lots = [{"year": MP.EPOCH.year + y, "lot": round(s * config.LOT_STEP, 2), "trades": int(c)}
            for y, row in enumerate(one.steps) for s, c in enumerate(row) if c > 0 and s > 0]
    _csv(pd.DataFrame(lots, columns=["year", "lot", "trades"]), folder / "lots.csv")    # counts, never an average
    _csv(_rounded(out["needs"], 4), folder / "mondays.csv")
    if len(out["dips"]):
        _csv(_rounded(out["dips"].drop(columns=["need_known_at"]), 3), folder / "hedged-dips.csv")


def _rounded(frame: pd.DataFrame, places: int) -> pd.DataFrame:
    """The number columns rounded (times and words left as they are)."""
    numbers = frame.select_dtypes("number").columns
    return frame.assign(**{c: frame[c].round(places) for c in numbers})


def _csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, pd.Timestamp):
        return str(x)
    return x


def _record(out: dict) -> Path:
    fx, info, v1, dips = out["facts"], out["info"], out["rows"][0], out["dips"]
    fingerprint = out["ts"].frame[["signal", "open", "close", "gross_inr", "cost_inr", "low_inr", "need_inr"]]
    settings = {
        "trades": {"label": MP.LABEL, "idea_id": MP.IDEA, "version_id": MP.VERSION, "readings": MP.READINGS,
                   "made_by": "the tester (L3): a stop at his hedge point, his target, one trade at a time, 0.01 lot",
                   "segments": info["segments"], "in_the_plan": fx["trades"], "before_the_first_monday": fx["dropped"],
                   "2016-2020": "the years this version was CHOSEN on (the best of 18, L10)",
                   "2021-2023": "the check years, opened once for this version (L10, 29 Sep 2026); nothing chosen here",
                   "trades_sha256": trials.sha256_of(_rounded(fingerprint, 6).astype(str).to_dict("list"))},
        "version_1": MP.VERSION_1.as_dict(),
        "extra_rows": [r["plan"].as_dict() for r in out["rows"][1:]],
        "money_needed_for_0.01_lot": "2 x that week's Monday ADR (the larger of the 5- and 10-day average day range, "
                                     "finished broker days, strategy-code.md section 16 USE 3) x 1 ounce x the rupee "
                                     "rate of that Monday's month; known when the Monday's first 5-minute bar closed",
        "definitions": NOTES + " section 1 (committed before the first run on real trades)",
        "reshuffles": {"runs": out["n_runs"], "seed": MP.SEED, "blocks": "calendar months, each used once",
                       "months": out["ts"].n_months},
        "year_rule": "a trade counts in the year it OPENED",
        "cost": f"Rs {config.COST_INR_PER_POSITION} per position per 0.01 lot, all-in (LEARNED #54, #115)",
        "already_seen": {"account_doubles_7_8_days": "1 try, 2016-2026 (prior-tries)",
                         "capital_adr_sizing": "10 tries, 2016-2023 (prior-tries)"},
        "hedged_dips": {"trades": f"{_rel(MP.HEDGED_STORIES)} (L19a, trial {MP.HEDGED_TRIAL})",
                        "note": "as recorded by L19a, before trendlines counted as liquidity (L6b.1)"},
        "code_sha256": trials.code_sha256([Path(MP.__file__), Path(__file__)]),
    }
    result = {
        "flat_0.01_lot": {"total": fx["flat_total"], "price_moves": fx["flat_gross"], "cost": fx["flat_cost"],
                          "by_year_opened": fx["flat_by_year"]},
        "version_1": {"real_order": {**v1["head"], "years": out["years"],
                                     "price_moves_at_the_plans_lots": fx["plan_gross"],
                                     "cost_at_the_plans_lots": fx["plan_cost"], "largest_lot": fx["lot_max"],
                                     "trades_at_0.01_or_0.02": fx["lot_small"], "best_3_trades": fx["best3"],
                                     "orders_that_did_better": fx["orders_better"],
                                     "orders_that_did_worse": fx["orders_worse"]},
                      "reshuffles": {**v1["spread"], "doubles_within_8_days": fx["doubles_in_his_days"],
                                     "runs_with_a_double_by_year": fx["runs_with_double_by_year"],
                                     "best_order_result": fx["orders_best"],
                                     "orders_above_the_start": fx["orders_made_money"]}},
        "rows": [{"name": r["plan"].name, "change": r["plan"].change, "real_order": r["head"],
                  "reshuffles": {k: r["spread"][k] for k in ("all_years", "any_year", "worst")}}
                 for r in out["rows"][1:]],
        "hedged_dips": {"trades": 0} if not len(dips) else {
            "trades": int(len(dips)), "still_open_at_the_end": int(dips["still_open"].sum()),
            "at_or_past_1x_the_money_needed": int((dips["low_in_needs"] >= 1).sum()),
            "account_at_zero_or_less_at_the_plans_lots": int((dips["account_at_low"] <= 0).sum()),
            "deepest_inr_at_0.01_lot": float(dips["low_inr"].min()),
            "deepest_in_money_needed": float(dips["low_in_needs"].max())},
        "pictures": [{k: p[k] for k in ("path", "pixel_sha256")} for p in out.get("pictures", [])],
    }
    return trials.record(
        kind="money-plan", idea_id=MP.IDEA, version_id=MP.VERSION, parent_idea=None, settings=settings,
        data=out["data"], result=_plain(result),
        note="L20: his money plan on idea #1's plain trades (not his hedging) - the real order, 1,000 reshuffles of "
             "whole months, 9 one-change rows; the same version of idea #1, not a new version")


# ================================================================================================================
# the report: easy English (CLAUDE.md #113, #114)
# ================================================================================================================

def report_md(out: dict) -> str:
    v1, fx, years, n = out["rows"][0], out["facts"], out["years"], out["n_runs"]
    h, s, one = v1["head"], v1["spread"], v1["one"]
    a, plan = s["all_years"], one.plan
    start = plan.your_money
    real_days = one.dbl_days
    worst_year = min(years, key=lambda r: r["result"])
    seg = out["info"]["segments"]
    lost = h["result"] < 0
    dips = out["dips"]
    deep = dips[dips["low_in_needs"] >= 1].sort_values("low_in_needs", ascending=False) if len(dips) else dips
    L: list[str] = []
    add = L.append

    add("# Your money plan, measured (L20)")
    add("")
    add(f"> For Tejas · made by `python -m machine.money.plan` · trial file `{out.get('trial', 'not written')}` · "
        f"what each word means was fixed before the run: `{NOTES}` section 1.")
    add(f"> **{MD_LABEL}.** The trades are idea #1's {fx['trades']} trades, 2016–2023: a plain stop at the hedge "
        f"point, one trade at a time, ₹40 a position per 0.01 lot. Your entry types with your hedging are not ready "
        f"yet (L19b). Idea #1 is dropped. It is used because it is the only idea with trades in the new machine.")
    add("> **Already seen:** the old project looked at \"doubles in 7–8 days\" once (2016–2026) and at the capital "
        "rule 10 ways (2016–2023). These are not fresh years for either.")
    add("")
    add("## The answer")
    add("")
    slow = not np.isfinite(h["days_mid"]) or h["days_mid"] > HIS_DAYS
    add(f"**On these trades the plan {'lost' if lost else 'made'} money, and "
        + ("the account did not double every 7–8 days.**" if slow
           else f"the middle double took {h['days_mid']:.0f} days.**"))
    add("")
    add("| | the real order, 2016–2023 |")
    add("|---|---|")
    add(f"| your {inr(start)} | became **{inr(h['your_money_end'])}** ({signed(h['result'])}) |")
    add(f"| doubles | **{h['doubles']} in {len(years)} years**"
        + (f": after {days_list(real_days)} |" if len(real_days) else " |"))
    add(f"| blow-ups | **{h['blowups']} in {len(years)} years**, about {h['blowups'] / len(years):.1f} a year "
        f"(you accept {HIS_BLOWUPS[0]} to {HIS_BLOWUPS[1]}) |")
    add(f"| taken out | {inr(h['out'])} |")
    each = (f" ({inr(min(fx['refills']))} to {inr(max(fx['refills']))} each time)" if len(fx["refills"]) > 1 else "")
    add(f"| added back after blow-ups | {inr(h['added'])}{each} |")
    add(f"| the worst year | {worst_year['year']}: {signed(worst_year['result'])} ({worst_year['blowups']} "
        f"blow-ups, {worst_year['doubles']} doubles) |")
    add("")
    share = fx["orders_worse"] / n
    kind = "a lucky one" if share >= 0.8 else ("an unlucky one" if share <= 0.2 else "an ordinary one")
    add(f"**The real order was {kind}.** The same months in {n:,} other orders: {fx['orders_worse']:,} ended "
        f"with less. The middle one ended at {inr(start + a['result_mid'])} ({signed(a['result_mid'])}).")
    b = fx["best3"]
    soon = [x for x in b if x["days_after_refill"] is not None and x["days_after_refill"] <= SOON]
    rest3 = h["result"] - fx["best3_sum"]
    luck = (f"The 3 best trades made {inr(fx['best3_sum'])}. Without them the real order "
            f"{'loses' if rest3 < 0 else 'makes'} {inr(abs(rest3))}.")
    if soon:
        where = " and ".join(f"{x['lots']:.2f} lot in {x['open']:%b %Y}" for x in soon)
        luck += (f" {len(soon)} of the 3 opened within a week after a refill, when the account was full again "
                 f"({where}).")
    add(luck)
    add("")
    add("**Why it lost:**" if lost else "**Why:**")
    add(f"- The trades themselves {'lose' if fx['flat_total'] < 0 else 'make'} {inr(abs(fx['flat_total']))} at "
        f"0.01 lot always: the price moves made {signed(fx['flat_gross'])}, the ₹40 cost took "
        f"{inr(fx['flat_cost'])}.")
    add(f"- The plan trades bigger lots: about {fx['lots_avg']:.0f} times 0.01 lot on average, up to "
        f"{fx['lot_max']:.2f}. Every 0.01 lot pays ₹40, so the cost was {inr(fx['plan_cost'])}.")
    rest_moves = fx["plan_gross"] - fx["best3_price_moves"]
    quiet, chosen = _quiet_years(fx)
    more = fx["plan_gross"] > fx["lots_avg"] * fx["flat_gross"]
    add(f"- At the plan's lots the price moves made {signed(fx['plan_gross'])}"
        + (". That is more than the bigger lots alone explain." if more else ".")
        + f" The 3 best trades made {inr(fx['best3_price_moves'])} of it (without them: {signed(rest_moves)}). "
          f"The Monday number was smallest in {quiet} (the year table), so the rule bought the most lots then"
        + (f":{chosen}, which look better than they are." if chosen else "."))
    add(f"- Left after the cost: {signed(fx['plan_gross'] - fx['plan_cost'])}.")
    add("")
    if len(dips):
        zero = int((dips["account_at_low"] <= 0).sum())
        add(f"**Your hedging, measured apart ({len(dips)} hedged trades of idea #1 from L19a):** in {len(deep)} of "
            f"them the trade went down by as much as the money needed for 0.01 lot, or more, before it closed. With "
            f"every rupee at work, a full account would be at zero inside {zero} of them. The table is further down.")
        add("")
    add("![The account in the real order](money-plan/account.jpg)")
    add("")
    add(f"How to read it: the black line is the money in the account. It sinks to the red line (the money needed "
        f"for 0.01 lot). That is a blow-up (red mark): the reserve fills it back to {inr(plan.account_size)}. A "
        f"green mark is a double: the profit is taken out.")
    add("")

    add("## The plan, as the machine ran it")
    add("")
    add("| your rule | how the machine does it |")
    add("|---|---|")
    add(f"| money needed for 0.01 lot = 2 × that Monday's ADR | in rupees, kept for the week. It was between "
        f"{inr(fx['need_min'])} and {inr(fx['need_max'])} in these 8 years |")
    add("| the ₹100-of-chocolate rule | lots = the money in the account ÷ the money needed for 0.01 lot, rounded "
        "down to 0.01, 0.02, 0.03 … |")
    add("| one trade at a time | a trade opens only after the last one closed |")
    add(f"| a burner account of about 10% | your money {inr(start)}: {inr(plan.account_size)} in the account, "
        f"{inr(plan.reserve_size)} kept outside (the reserve) |")
    add("| withdraw at each double | when the account reaches 2 × its starting money, everything above the "
        "starting money is taken out |")
    add("| **a blow-up** | the account is below the money needed for 0.01 lot (no trade open) |")
    add(f"| **the refill** | the reserve brings the account back to {inr(plan.account_size)} |")
    add("")
    nb = fx["need_by_year"]
    if 2018 in nb and 2023 in nb:
        lo18, hi18 = nb[2018]
        add(f"The number you remembered: about ₹7,000 to ₹8,000 (#114). Later you said it was 2023 and \"don't take "
            f"my word for it\" (#133). By the rule, 2023 went from {inr(nb[2023][0])} to {inr(nb[2023][1])}: your "
            f"number was a high week of 2023. 2018 went from {inr(lo18)} to {inr(hi18)}. So in 2018 a full "
            f"{inr(plan.account_size)} buys {np.floor(plan.account_size / hi18) * 0.01:.2f} to "
            f"{np.floor(plan.account_size / lo18) * 0.01:.2f} lot, not the 0.02 of your example. Row 6 below uses "
            f"your sheet's ₹7,100.")
        add("")

    add("## Year by year, the real order")
    add("")
    add("| year | trades | money needed for 0.01 lot | lots used | blow-ups | doubles | taken out | added back | "
        "the year's result | growth of your money | the same trades at 0.01 lot always |")
    add("|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|")
    for r in years:
        y = r["year"] - MP.EPOCH.year
        dd = one.dbl_days[one.dbl_year == y]
        add(f"| {r['year']} | {r['trades']} | {inr(r['need_lo'])} to {inr(r['need_hi'])} | {_lot_range(one.steps[y])} "
            f"| {r['blowups']} | {r['doubles']}{' (' + days_list(dd) + ')' if len(dd) else ''} | "
            f"{inr(r['taken_out'])} | {inr(r['added_back'])} | {signed(r['result'])} | {pc(r['growth'])} | "
            f"{signed(fx['flat_by_year'].get(r['year'], 0.0))} |")
    add(f"| **all** | **{h['taken']}** | | 0.01 to {fx['lot_max']:.2f} | **{h['blowups']}** | **{h['doubles']}** | "
        f"**{inr(h['out'])}** | **{inr(h['added'])}** | **{signed(h['result'])}** | | "
        f"**{signed(fx['flat_total'])}** |")
    add("")
    add("- A trade counts in the year it opened.")
    add("- Growth = the year's result ÷ your money at the start of that year (the reserve + the account + "
        "everything taken out).")
    add(f"- {fx['lot_small']} of the {fx['trades']} trades were at 0.01 or 0.02 lot. After losses the account is "
        f"small, so the lots are small. How many trades at each lot size, year by year: "
        f"`plan/work/money-plan/lots.csv`.")
    add("- 2016–2020 are the years this version of idea #1 was chosen on (the best of 18). Those years look better "
        "than they are. 2021–2023 were hidden while choosing.")
    add("")

    add(f"## The same months in {n:,} other orders")
    add("")
    add(f"The {out['ts'].n_months} months are put in a random order, each month used once. The trades inside a "
        f"month stay as they were. The plan runs through each order from the same start. Seed {MP.SEED}.")
    add("")
    add("![Where your money ended in each order](money-plan/orders.jpg)")
    add("")
    add("| over the 8 years | the middle order | the bad 10% |")
    add("|---|---:|---:|")
    add(f"| your {inr(start)} ends at | {inr(start + a['result_mid'])} ({signed(a['result_mid'])}) | "
        f"{inr(start + a['result_bad'])} or less ({signed(a['result_bad'])}) |")
    add(f"| taken out | {inr(a['out_mid'])} | {inr(a['out_bad'])} or less |")
    add(f"| added back after blow-ups | {inr(a['added_mid'])} | {inr(a['added_bad'])} or more |")
    add(f"| blow-ups | {a['blowups_mid']:.0f} | {a['blowups_bad']:.0f} or more |")
    add(f"| doubles | {a['doubles_mid']:.0f} | {a['runs_no_double']:,} of the {n:,} orders had none |")
    add("")
    add("\"The bad 10%\" = 1 order in 10 was this bad or worse.")
    best = fx["orders_best"]
    add((f"None of the {n:,} orders ended above the start. " if fx["orders_made_money"] == 0
         else f"{fx['orders_made_money']:,} of the {n:,} orders ended above the start. ")
        + f"The best one ended at {inr(start + best)} ({signed(best)}).")
    add("")
    add("| year of the plan | growth of your money: middle | bad 10% | blow-ups: middle | bad 10% | orders with a "
        "double that year | days per double: middle | slow 10% |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for y, r in enumerate(s["per_year"]):
        add(f"| {r['year']} | {pc(r['growth_mid'])} | {pc(r['growth_bad'])} | {r['blowups_mid']:.0f} | "
            f"{r['blowups_bad']:.0f} | {fx['runs_with_double_by_year'][y]:,} of {n:,} | {num(r['days_mid'])} | "
            f"{num(r['days_slow'])} |")
    any_ = s["any_year"]
    add(f"| **any year** | {pc(any_['growth_mid'])} | {pc(any_['growth_bad'])} | {any_['blowups_mid']:.0f} | "
        f"{any_['blowups_bad']:.0f} | | {num(any_['days_mid'])} | {num(any_['days_slow'])} |")
    add("")
    add(f"- Days per double counts only the times the account did double ({fx['doubles_all']:,} in all the "
        f"orders). More often it blew up first ({int(v1['many'].blowups.sum()):,} blow-ups in all the orders).")
    add(f"- {fx['doubles_in_his_days']:,} of those {fx['doubles_all']:,} doubles came within {HIS_DAYS} days.")
    add("- \"Slow 10%\" = 1 double in 10 took this long or longer.")
    w = s["worst"]
    used = (f"The reserve ran out in year {w['used_up_year']} of the plan." if w["used_up_year"]
            else f"The reserve did not run out: {inr(float(out['worst'].reserve[0]))} of it was left.")
    add("")
    add(f"**The worst order of the {n:,}:** your {inr(start)} ended at {inr(w['your_money_end'])} "
        f"({signed(w['result'])}). {w['blowups']} blow-ups, {w['doubles']} doubles, {inr(w['out'])} taken out, "
        f"{inr(w['added'])} added back. {used}")
    add(f"The reserve ran out in {a['runs_reserve_used_up']:,} of the {n:,} orders.")
    add("")

    add("## One change at a time (shown, not recommended)")
    add("")
    add("Each row changes ONE thing from the plan above. Same trades.")
    add("")
    add("| | the one change | real order: your money ends at | blow-ups | times profit was taken out | taken out | "
        f"added back | {n:,} orders: the middle ends at | the bad 10% | orders where the reserve ran out |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in out["rows"]:
        hh, aa, p = r["head"], r["spread"]["all_years"], r["plan"]
        name = "**the plan**" if not p.change else p.name
        stop = f" (the reserve ran out, {hh['reserve_used_up']})" if hh["reserve_used_up"] else ""
        add(f"| {name} | {ROW_WORDS.get(p.name, 'as written')} | {inr(hh['your_money_end'])} "
            f"({signed(hh['result'])}){stop} | {hh['blowups']} | "
            f"{hh['doubles'] if p.withdraw_at is not None else 0} | {inr(hh['out'])} | {inr(hh['added'])} | "
            f"{inr(p.your_money + aa['result_mid'])} ({signed(aa['result_mid'])}) | "
            f"{inr(p.your_money + aa['result_bad'])} | {aa['runs_reserve_used_up']:,} of {n:,} |")
    add("")
    add("What the rows show:")
    for line in _row_notes(out):
        add(f"- {line}")
    add("")

    if len(dips):
        _hedged_section(out, deep, add)

    _two_readings(out, deep, add)

    add("## What would make this wrong")
    add("")
    add("- **Trades that make money after the ₹40 cost.** Then the plan would grow them. This test cannot say how "
        "fast.")
    quiet, chosen = _quiet_years(fx)
    add(f"- **The years.** The Monday number was smallest in {quiet}, so the plan's biggest lots fell there"
        + (f":{chosen}." if chosen else ".") + " On other trades the biggest lots may fall in their worst years.")
    add("- **Your hedging (L19b).** These are plain trades. A hedged trade stays open for weeks and dips deeper.")
    add(f"- **One idea only.** Idea #1, {fx['trades']} trades, and it is dropped.")
    add(f"- **The start.** {inr(plan.account_size)} in the account is your own number (#114, #115). With another "
        f"start every rupee number changes (rows 2 and 3).")
    add("- **The refill.** The prompt's words, read to the letter, refill only up to the money needed for 0.01 lot. "
        "That is row 7. The plan above refills to the full account.")
    add("")
    add(f"**Tries:** {len(out['rows'])} settings of the plan (the plan + {len(out['rows']) - 1} one-change rows) on "
        f"one set of trades. Nothing was chosen. The old project: 10 tries of the capital rule, 1 of the doubling.")
    add("")

    add("## Where the numbers are")
    add("")
    add("- every trade with its lot and money: `plan/work/money-plan/trades.csv`")
    add("- every blow-up and double in the real order: `plan/work/money-plan/events.csv`")
    add("- how many trades at each lot size, year by year: `plan/work/money-plan/lots.csv`")
    add(f"- the {len(out['needs'])} Monday numbers: `plan/work/money-plan/mondays.csv`")
    add(f"- the trades are the recorded ones: {seg['2016-2020']['trades']} in 2016–2020 "
        f"({signed(seg['2016-2020']['net_inr'])} at 0.01 lot) and {seg['2021-2023']['trades']} in 2021–2023 "
        f"({signed(seg['2021-2023']['net_inr'])}), the same as the L10 runs")
    add("- the code: `machine/money/plan.py`, `machine/money/report.py`; the tests: `tests/test_money_plan.py`")
    add("")
    return "\n".join(L)


def _years_text(years: list[int]) -> str:
    """2017, 2018, 2019 -> '2017–2019'; 2016, 2018 -> '2016 and 2018'."""
    ys = sorted(years)
    if len(ys) > 1 and ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]}–{ys[-1]}"
    return ", ".join(str(y) for y in ys[:-1]) + f" and {ys[-1]}" if len(ys) > 1 else str(ys[0])


def _quiet_years(fx: dict) -> tuple[str, str]:
    """(the 3 years with the smallest Monday number, what they were for this idea): the rule buys the most lots in
    them. If they are all years the version was chosen on, the second part says so; else it is empty."""
    ys = fx["quiet_years"]
    chosen_on = all(y in config.TUNE_YEARS for y in ys)
    return _years_text(ys), (" years this version of idea #1 was chosen on" if chosen_on else "")


def _two_readings(out: dict, deep: pd.DataFrame, add) -> None:
    """The judgement guard (#114): two readings, what each would mean, which one the numbers fit - and stop.
    Written for trades that LOSE after cost. Every sentence is checked against the numbers first: with other trades
    (L19b's) this refuses, and the session writes the readings again instead of printing a claim that is not true."""
    rows, fx, n, dips = out["rows"], out["facts"], out["n_runs"], out["dips"]
    one = rows[0]["one"]
    flat, nocost = rows[5]["many"].result, rows[9]["spread"]["all_years"]["result_mid"]
    checks = {
        "the plan lost in the real order": rows[0]["head"]["result"] < 0,
        "the trades lose at 0.01 lot always": fx["flat_total"] < 0,
        "0.01 lot always ends the same in every order": float(np.ptp(flat)) < 1.0,
        "more money at work lost more (5% < 10% < 20%)": _mid(out, 2) > _mid(out, 0) > _mid(out, 3),
        "before the cost the middle order made money": nocost > 0,
        "doubles within his 7-8 days were rare (under 1 in 20)":
            fx["doubles_in_his_days"] * 20 < max(fx["doubles_all"], 1),
    }
    wrong = [k for k, ok in checks.items() if not ok]
    if wrong:
        raise MoneyReportError("the report's 'two ways to read this' no longer fits the numbers (" + "; ".join(wrong)
                               + "): write report._two_readings again for these trades")
    add("## Two ways to read this")
    add("")
    add("**Reading 1: the money plan does not work.** Then the plan itself (the burner, the lot rule, taking profit "
        "out at a double) should be dropped.")
    add("")
    add("**Reading 2: the plan cannot rescue trades that lose after the ₹40 cost.** These trades do, so the plan "
        "lost more. Then the plan can only be judged on trades that make money after cost (your hedging, L19b).")
    add("")
    add("**The numbers fit reading 2:**")
    add(f"- At 0.01 lot always, the same trades end at {signed(float(np.median(flat)))} in every one of the {n:,} "
        f"orders. The order does not matter when the lot never changes.")
    add(f"- The more money at work, the bigger the loss. Middle order: account 5% {signed(_mid(out, 2))}, "
        f"account 10% {signed(_mid(out, 0))}, account 20% {signed(_mid(out, 3))}.")
    quiet, chosen = _quiet_years(fx)
    add(f"- Before the ₹40 cost (not real) the same plan ends the middle order at {signed(nocost)}. So the ₹40 at "
        f"bigger lots is what turns it into a loss."
        + (f" (Part of that {signed(nocost)} is the biggest lots landing in {quiet},{chosen}.)" if chosen else ""))
    add("")
    add("**Two things the test does say about the plan itself:**")
    real_days = one.dbl_days
    fastest = int(round(float(np.min(real_days)))) if len(real_days) else None
    add("- \"Doubles in 7–8 days\" did not happen. "
        + (f"The fastest double in the real order took {fastest} days. " if fastest else "No double in the real "
           "order. ")
        + f"In the {n:,} orders, {fx['doubles_in_his_days']:,} of {fx['doubles_all']:,} doubles came within "
          f"{HIS_DAYS} days. (Already seen by the old project.)")
    if len(dips):
        add(f"- For your hedging, 2 × the Monday ADR was not enough room in {len(deep)} of {len(dips)} trades.")
    add("")
    add("The decision is yours.")
    add("")


def _hedged_section(out: dict, deep: pd.DataFrame, add) -> None:
    """What plain trades cannot show: the dips inside L19a's hedged trades against the money needed for 0.01 lot."""
    dips, plan = out["dips"], out["rows"][0]["one"].plan
    zero = int((dips["account_at_low"] <= 0).sum())
    still = int(dips["still_open"].sum())
    add("## Your hedging: how far a trade dips before it closes")
    add("")
    add(f"Plain trades cannot show this, so it is measured apart. L19a ran idea #1 with your hedging (every "
        f"position 0.01 lot, 2016–2020): {len(dips)} trades"
        + (f", {still} of them still open at the end of 2020" if still else "")
        + ". For each one: the lowest point the whole trade reached before it closed, next to the money needed for "
          "0.01 lot the week it opened.")
    add("")
    add(f"- In **{len(deep)} of the {len(dips)}** trades the lowest point was as much as the money needed for 0.01 "
        f"lot, or more.")
    add(f"- The deepest: {inr(-float(dips['low_inr'].min()))} down at 0.01 lot. That is "
        f"{float(dips['low_in_needs'].max()):.1f} times the money needed that week.")
    add(f"- With the chocolate rule every rupee in the account is at work. So in **{zero}** of these trades a full "
        f"{inr(plan.account_size)} account would be at zero or below before the trade closed.")
    add("")
    if len(deep):
        add("| opened | days open | positions | it ended at (0.01 lot) | its lowest point (0.01 lot) | money needed "
            "for 0.01 lot that week | lowest point ÷ money needed |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for r in deep.itertuples():
            end = "still open at the end of 2020" if r.still_open else signed(r.net_inr)
            add(f"| {pd.Timestamp(r.open):%d %b %Y} | {r.days:.0f} | {r.positions} | {end} | {inr(r.low_inr)} | "
                f"{inr(r.need_inr)} | {r.low_in_needs:.1f} |")
        add("")
    rest = dips[dips["low_in_needs"] < 1]
    won = int((deep["net_inr"] > 0).sum()) if len(deep) else 0
    if len(rest):
        add(f"- The other {len(rest)} stayed under ({float(rest['low_in_needs'].max()):.1f} times at most).")
    if won:
        add(f"- {won} of the {len(deep)} ended in profit, but only after that dip.")
    add(f"- All {len(dips)}: `plan/work/money-plan/hedged-dips.csv`.")
    add("- These trades are as L19a recorded them, before trendlines counted as liquidity. L19b will make them "
        "again.")
    add("")


def _lot_range(row) -> str:
    used = [s for s, c in enumerate(row) if c > 0 and s > 0]
    if not used:
        return "-"
    return f"{used[0] * 0.01:.2f}" if len(used) == 1 else f"{used[0] * 0.01:.2f} to {used[-1] * 0.01:.2f}"


def _mid(out: dict, i: int) -> float:
    return out["rows"][i]["spread"]["all_years"]["result_mid"]


def _row_notes(out: dict) -> list[str]:
    rows, n = out["rows"], out["n_runs"]
    mid = lambda i: signed(rows[i]["spread"]["all_years"]["result_mid"])                    # noqa: E731
    same = float(np.ptp(rows[5]["many"].result)) < 1.0
    size = rows[0]["plan"].account_size
    sheet_lot = np.floor(size / rows[6]["plan"].sheet_inr) * 0.01
    return [
        f"Row 1 (take out at +75%): middle order {mid(1)}. The plan: {mid(0)}.",
        f"Rows 2 and 3 (the account 5% or 20% of your money): middle order {mid(2)} and {mid(3)}. At 20% the "
        f"reserve ran out in {rows[3]['spread']['all_years']['runs_reserve_used_up']:,} of the {n:,} orders.",
        f"Row 4 (never take out): {mid(4)} in the middle order. Money taken out is safe from the next blow-up.",
        f"Row 5 (0.01 lot always): {signed(rows[5]['head']['result'])} in "
        + ("every order." if same else f"the real order, {mid(5)} in the middle order."),
        f"Row 6 (your sheet's ₹7,100 for 0.01 lot): a full account trades {sheet_lot:.2f} lot. Middle order {mid(6)}.",
        f"Row 7 (refill only up to the money needed for 0.01 lot): {rows[7]['head']['blowups']} blow-ups in the "
        f"real order. After the first one the account sits on the line, so almost every loss is a new blow-up.",
        f"Row 8 (a blow-up only at ₹0): {rows[8]['head']['blowups']} blow-ups in the real order. Each one burns the "
        f"whole account.",
        f"Row 9 (before the ₹40 cost, NOT REAL): {mid(9)} in the middle order. It only shows how much the cost "
        f"takes.",
    ]
