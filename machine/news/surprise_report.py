"""The checks printed by `python -m machine.news.surprise` (L12b.1). Counts are 2016-2023 (2015 only warms up the
rolling "usual miss"). Nothing here reads a gold price."""
from __future__ import annotations

import numpy as np
import pandas as pd

from machine.news import forexfactory, surprise

YEARS = ("2016-01-01", "2023-12-31")


def _in_years(df: pd.DataFrame) -> pd.DataFrame:
    day = df["ny_date"] if "ny_date" in df else df["utc"].dt.tz_convert(surprise.NY).dt.tz_localize(None)
    return df[(day >= YEARS[0]) & (day <= YEARS[1] + " 23:59")]


def headline_events(signs: pd.DataFrame) -> list[str]:
    scored = signs[signs["method"] != "flagged"]
    return list(scored.sort_values(["report", "rank"]).drop_duplicates("report")["event"])


def report(table: pd.DataFrame, events: pd.DataFrame | None = None, ff: pd.DataFrame | None = None,
           out=print) -> dict:
    from machine.news import calendar
    signs = surprise.load_signs()
    events = events if events is not None else calendar.load_events(importance=("high", "moderate"))
    ff = ff if ff is not None else forexfactory.load()
    t = _in_years(table)
    found: dict = {}

    out("\n== 1. signs.csv ==")
    out(f"{len(signs)} event kinds: {int((signs['method'] == 'forecast').sum())} scored by the forecast, "
        f"{int((signs['method'] == '2y yield').sum())} by the 2-year yield (the Fed's decisions), "
        f"{int((signs['method'] == 'flagged').sum())} flagged (no forecast, no market series). "
        f"Turned round (lower = good for the dollar): {list(signs.loc[signs['sign'] == -1, 'event'])}")

    out("\n== 2. Each report's number, 2016-2023: found in Forex Factory? scored? ==")
    heads = headline_events(signs)
    rows = []
    for ev in heads:
        r = t[(t["event"] == ev) & t["headline"]]
        rows.append({"event": ev, "releases": len(r),
                     "FF same minute": int((r["ff_match"] == "same minute").sum()),
                     "FF same day": int((r["ff_match"] == "same day").sum()),
                     "FF none": int((r["ff_match"] == "none").sum()) if (r["method"] == "forecast").any() else "-",
                     "red": int((r["ff_colour"] == "red").sum()), "orange": int((r["ff_colour"] == "orange").sum()),
                     "yellow": int((r["ff_colour"] == "yellow").sum()),
                     "scored": int(r["z"].notna().sum())})
    heads_table = pd.DataFrame(rows)
    out(heads_table.to_string(index=False))
    found["heads"] = heads_table

    out("\n== 3. MT5 against Forex Factory, the same releases 2016-2023 (every line scored by the forecast; the same = "
        "within Forex Factory's rounding) ==")
    m = t[(t["ff_match"] != "none") & (t["method"] == "forecast")]
    m = m.merge(events[["value_id", "forecast", "multiplier"]], on="value_id", how="left")
    m = m.merge(ff[["id", "actual", "forecast"]].rename(columns={"id": "ff_id", "actual": "ff_actual_text",
                                                                 "forecast": "ff_forecast_text"}), on="ff_id",
                how="left")
    m["mt5_forecast"] = m["forecast"] * m["multiplier"].fillna("").map(surprise.MULTIPLIER).fillna(1.0)
    a = m[m["mt5_actual"].notna() & m["ff_actual"].notna()]
    f = m[m["mt5_forecast"].notna() & m["ff_forecast"].notna()]
    act_same = pd.Series(((a["mt5_actual"] - a["ff_actual"]).abs() <= a["ff_actual_text"].map(forexfactory.half_step)
                          + 1e-9).to_numpy(), index=a.index)
    fc_same = pd.Series(((f["mt5_forecast"] - f["ff_forecast"]).abs() <= f["ff_forecast_text"].map(
        forexfactory.half_step) + 1e-9).to_numpy(), index=f.index)
    out(f"actual number: the same on {int(act_same.sum())} of {len(a)} rows ({act_same.mean():.1%})")
    out(f"forecast: the same on {int(fc_same.sum())} of {len(f)} rows ({fc_same.mean():.1%}) - MT5's forecast is "
        f"not the economists' consensus")
    by_year = f.assign(year=f["ny_date"].dt.year, same=fc_same).groupby("year")["same"].mean().round(3)
    out("forecast the same, share by year: " + ", ".join(f"{y} {v:.0%}" for y, v in by_year.items()))
    nfp = f[f["event"] == "Nonfarm Payrolls"].assign(same=fc_same[f["event"] == "Nonfarm Payrolls"])
    worst = nfp.assign(gap=(nfp["mt5_forecast"] - nfp["ff_forecast"]).abs()).nlargest(5, "gap")
    out("NFP, the 5 biggest forecast gaps (jobs):")
    for r in worst.itertuples():
        out(f"  {r.ny_date:%Y-%m-%d}: MT5 {r.mt5_forecast:,.0f}  Forex Factory {r.ff_forecast:,.0f}  "
            f"actual {r.ff_actual:,.0f}")
    diff_act = a[~act_same.to_numpy()]
    if len(diff_act):
        out(f"actual numbers that differ ({len(diff_act)}), first 8:")
        for r in diff_act.head(8).itertuples():
            out(f"  {r.ny_date:%Y-%m-%d} {r.event}: MT5 {r.mt5_actual:,.4g}  Forex Factory {r.ff_actual:,.4g}")
    found["actual_same"], found["forecast_same"] = act_same.mean(), fc_same.mean()

    out("\n== 4. The signs against Forex Factory's own mark (it colours an actual 'better' or 'worse' for the "
        "dollar) ==")
    c = t[t["dollar_surprise"].notna() & (t["dollar_surprise"] != 0) & t["ff_better_worse"].isin(["better", "worse"])
          & (t["method"] == "forecast")]
    agree = (c["dollar_surprise"] > 0) == (c["ff_better_worse"] == "better")
    per = c.assign(agree=agree).groupby("event")["agree"].agg(["sum", "size"])
    per["share"] = per["sum"] / per["size"]
    out(f"all lines: agree on {int(agree.sum())} of {len(c)} releases ({agree.mean():.1%})")
    low = per[per["share"] < 0.9].sort_values("share")
    out("events agreeing on less than 90%:" + ("" if len(low) else " none"))
    for ev, r in low.iterrows():
        out(f"  {ev}: {int(r['sum'])} of {int(r['size'])}")
    found["sign_agree"], found["sign_low"] = agree.mean(), low

    out("\n== 5. Groups by year - each report's number (big good for the dollar / small / big bad / not scored) ==")
    g = t[t["headline"] & (t["method"] != "flagged")]
    key = {"big good for the dollar": "good", "small": "small", "big bad for the dollar": "bad", "not scored": "-"}
    for ev in ("Nonfarm Payrolls", "Core CPI m/m", "Initial Jobless Claims", "Fed Interest Rate Decision",
               "ISM Manufacturing PMI", "Retail Sales m/m", "GDP q/q"):
        e = g[g["event"] == ev]
        if not len(e):
            continue
        tab = pd.crosstab(e["ny_date"].dt.year, e["group"].map(key)).reindex(columns=["good", "small", "bad", "-"],
                                                                            fill_value=0)
        out(f"{ev}:  " + " | ".join(f"{y}: {r['good']}/{r['small']}/{r['bad']}/{r['-']}" for y, r in tab.iterrows()))
    allg = g["group"].map(key).value_counts()
    out(f"every report's number, 2016-2023: good {allg.get('good', 0)} / small {allg.get('small', 0)} / bad "
        f"{allg.get('bad', 0)} / not scored {allg.get('-', 0)}")
    why = g.loc[g["group"] == "not scored", "why_not_scored"].str.replace(r"only \d+ known", "only N known",
                                                                         regex=True).value_counts()
    out("not scored, why: " + "; ".join(f"{k} ({v})" for k, v in why.items()))

    out("\n== 6. The Fed's decision days: the 2-year yield, daily, measured after the release ==")
    fed = t[(t["report"] == "FOMC decision") & t["headline"]].copy()
    fed["bp"] = fed["surprise"] * 100
    for r in fed.itertuples():
        z = "  -  " if np.isnan(r.z) else f"{r.z:+.2f}"
        out(f"  {r.ny_date:%Y-%m-%d} {r.event[:26]:26s} {r.yield_before:5.2f} -> {r.yield_after:5.2f} "
            f"({r.bp:+4.0f} bp)  z {z}  {r.group}")
    found["fed_days"] = len(fed)

    out("\n== 7. His news (Forex Factory red + orange, US) that the calendar (MT5 high + moderate) lacks, 2016-2023 ==")
    ffy = _in_years(ff[ff["colour"].isin(["red", "orange"])].assign(
        ny_date=ff["utc"].dt.tz_convert(surprise.NY).dt.tz_localize(None).dt.normalize()))
    mapped = set(n for names in signs["ff_name"] for n in names.split("|") if n)
    used = set(zip(table["ff_name_found"].dropna(), table.loc[table["ff_name_found"].notna(), "ny_date"]))
    ffy = ffy.assign(in_calendar=[(n, d) in used for n, d in zip(ffy["name"], ffy["ny_date"])],
                     named=ffy["name"].isin(mapped))
    miss = ffy[~ffy["in_calendar"]].groupby(["name", "colour"]).size().sort_values(ascending=False)
    out(f"{int(ffy['in_calendar'].sum())} of {len(ffy)} red / orange rows are in the calendar. Missing, by name:")
    for (name, colour), n in miss.head(30).items():
        out(f"  {n:4d}  {colour:6s} {name}{'' if name in mapped else '   (no line in signs.csv names it)'}")
    found["ff_missing"] = miss
    return found
