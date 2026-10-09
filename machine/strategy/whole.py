"""His whole strategy (L19d): his entry types TOGETHER, his hedge, his management - one research family.

    from machine.strategy import whole as W
    win = W.Window.tune()                          # 2016-2020: the years versions are tried on
    v = W.version({})                              # version 1: his strategy as written
    run = W.run(win, v)                            # his entries with his hedge, and with a plain stop
    print(W.run_summary(run)["hedged"]["net_inr_total"])

What "his whole strategy" is (plan/work/L19d-notes.md):
  his entries     every entry type of his (machine/entries/forms.py, L19b) whose words are settled - today Type 1
                  (the first gap), the zone break (the retest) and the zone entry. The trendline touch ('steep',
                  B2-01), the triangle false break ('big move', B2-02) and the second gap ('very strong', B2-03) wait
                  for his answers in the weekly picture batch: an open word is never chosen by the machine (R13), and
                  never by money (lesson 16: meaning is chosen before money is seen). The day his answer is in the
                  entry's form, the entry joins (the second gap then takes Type 1's place: it IS Type 1 plus its
                  exception, #124).
  one at a time   his M18 (#55): one trade at a time across everything; a waiting order is cancelled by a newer setup
                  of ANY of his entries (M39); a setup while a trade is on is skipped.
  his hedge       the hedging tester (L19a) with his management (T12): legs closed by any of his entries (M10) - the
                  closing setups ARE the whole's own setups, with the checks his orders get while they wait (#98 B,
                  #87, L19b).
  a plain stop    the tester (L3): a stop at the hedge point, the same target - the 2x2's other management.
A VERSION of the whole (one try of the family his-strategy) changes ONLY:
  exit.management.<name>     his management's TEST rows (L19c, machine/tester/management.py VERSIONS)
  trades_at_once             M18's test part: "one_per_entry" = one trade at a time PER entry type (T12 marks it a
                             test version; he never approved it - #118)
  settings.<name>            a number the machine chose (settings.py, status "test") - read by every entry
  <entry id>/<path>          one entry's own numbers (dimension 10): steps.<id>.max_wait_bars, steps.<id>.params.<n>,
                             entry.max_wait_bars, exit.beyond_usd, do_not_enter.<n>.params.<n>
Never his fixed rules (refused with his words, #121), never a reading of an open word, never a timeframe: the hedging
tester runs the whole on the 5-minute chart, and a timeframe is a version of ONE entry (its own family).
Money: Rs after Rs 40 a position (0.01 lot a leg, all-in, #54, #115); a trade counts in the year its last leg closed;
a trade still open at the window end is not counted (said).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from machine import config, guards, settings
from machine.entries import closers, forms
from machine.ideas.compile import Compiled, Market, compile_idea, version_of
from machine.mcp import versions as V
from machine.mcp.versions import Refused
from machine.referee import prior, stats
from machine.referee.replays import Segment
from machine.referee.tune import signals_sha256
from machine.tester import engine, hedging
from machine.tester import hedging_2x2 as H2
from machine.tester import management as MG
from machine.tester.run import summarize as stop_summary

STRATEGY = "his-strategy"                   # the research family (plan/work/ideas/his-strategy/)
KIND = "strategy"                           # one trial file per version run on 2016-2020: the try
KIND_2X2 = "strategy-2x2"                   # the chosen version's 2x2 on 2016-2020 (the same version: no new try)
TF = "5m"
TFS = ("1m", "5m", "1h", "1d", "1w")        # the 1-minute chart: the zone break's lines inside the zone (#98 C)
ONE, PER_ENTRY = "one_across_everything", "one_per_entry"
AT_ONCE = "trades_at_once"
# what the whole IS in the old project's list: his entries with his management - combo_750 and the other his_entries
# rows (plan/work/prior-tries.csv) ran them on 2016-2023, so these years are not fresh for it (CLAUDE.md rule 18)
SEEN_KEY = prior.IdeaKey.make(prior.HIS_ENTRIES, TF, "any")
REPLACES = {forms.SECOND_GAP.id: forms.TYPE1.id}     # when the second gap is in, it takes Type 1's place (#124)
MINUTE = pd.Timedelta(minutes=1)

WHAT_IT_IS = (
    "His entries together, one trade at a time across all of them (his rule #55). A waiting order is cancelled when a "
    "newer setup of any of his entries comes (his \"cancel and take new\"). A setup that comes while a trade is on is "
    "skipped. Each trade is managed with his hedge and his management (his table T12): a leg closes only when a full "
    "opposite setup of his forms - any of the same entries.")


# ================================================================================================= his entries

def open_asks(e: forms.EntryType) -> list[str]:
    """His open words in this entry's form (ASK parts with no answer yet)."""
    return [u.id for u in e.idea().unclear if u.kind == "ask" and u.answer is None]


def entries_in() -> list[forms.EntryType]:
    """The entries in the whole today: every entry type whose words are settled (the second gap in Type 1's place
    once his 'very strong' is answered)."""
    ins = [e for e in forms.ENTRY_TYPES.values() if not open_asks(e)]
    gone = {REPLACES[e.id] for e in ins if e.id in REPLACES}
    return [e for e in ins if e.id not in gone]


# Why each entry with an open word is left out - checked after the run (2 Oct 2026, the T11 session; plan/work/ideas/
# his-strategy/decision.md): 'steep' and 'big move' are NUMBERS (his #154: the machine tests numbers itself and never
# asks; L19b tested every value alone), and both answers to 'very strong' make Type 1's same 15 hedged trades. None of
# them can change the whole's answer, so none is asked.
LEFT_OUT = {
    forms.TRENDLINE_TOUCH.id: "its word 'steep' is a number - the machine tests numbers itself and never asks (his "
                              "#154); L19b tested every value alone: 11 / 5 / 0 / 0 setups in five years, too few to "
                              "change the whole",
    forms.TRIANGLE_FALSE_BREAK.id: "its word 'big move' is a number - the machine tests numbers itself and never asks "
                                   "(his #154); L19b tested every value alone: 29 / 11 / 5 setups in five years, too "
                                   "few to change the whole",
    forms.SECOND_GAP.id: "both answers to 'very strong' make the same 15 trades with his hedge as Type 1 alone - so "
                         "it is not asked (questions only where the answers make different trades)",
}


def waiting() -> dict[str, str]:
    """{entry id: why it is not in the whole} - an entry whose form still has an open word (never chosen by money,
    lesson 16)."""
    return {e.id: LEFT_OUT.get(e.id, f"waits for his answer to {e.open_word or ', '.join(open_asks(e))} (the weekly "
                                     "picture batch) - never chosen by the machine (R13), never by money (lesson 16)")
            for e in forms.ENTRY_TYPES.values() if open_asks(e)}


# ================================================================================================= a version

@dataclass(frozen=True)
class Version:
    """One version of the whole: its changes (no-ops left out), split by what they change, and its id."""
    version_id: str
    changes: dict                      # path -> value, as checked
    management: dict                   # his management's test rows: name -> value
    settings: dict                     # the machine's numbers: name -> value (every entry reads them)
    entries: dict                      # entry id -> {path: value} (one entry's own numbers)
    trades_at_once: str
    ins: tuple                         # the entry types in the whole (forms.EntryType)
    dimensions: dict                   # path -> what it is
    about: str

    @property
    def entry_ids(self) -> list[str]:
        return [e.id for e in self.ins]

    def entry_changes(self, e: forms.EntryType) -> dict:
        """What the compiler gets for this entry: the whole's numbers + the entry's own."""
        return {**{f"settings.{k}": x for k, x in self.settings.items()}, **self.entries.get(e.id, {})}

    def as_dict(self) -> dict:
        return {"version_id": self.version_id, "changes": self.changes, "management": self.management,
                "settings": self.settings, "entries": self.entries, "trades_at_once": self.trades_at_once,
                "entries_in": self.entry_ids, "dimensions": self.dimensions, "about": self.about}


def _form_value(e: forms.EntryType, path: str) -> Any:
    return V._get(e.idea().model_dump(mode="json"), path)


def _check_one(path: str, value: Any, ins: list[forms.EntryType]) -> tuple[str, str, str | None]:
    """(bucket, what it is, entry id or None) for one change - or Refused, with the reason."""
    if not isinstance(path, str) or not path.strip():
        raise Refused(f"a change needs a path like 'exit.management.on_a_friday', not {path!r}")
    if path == AT_ONCE:
        if value not in (ONE, PER_ENTRY):
            raise Refused(f"'{path}' = {value!r}: '{ONE}' (his rule M18, #55) or '{PER_ENTRY}' (M18's test part)")
        return "at_once", ("his M18's test part: one trade at a time PER entry type (T12 marks it a test version; he "
                           "never approved it - #118)"), None
    if path.startswith(MG.PATH):
        try:
            return "management", MG.check_value(forms.TYPE1.idea(), MG.name_of(path) or path, value, path), None
        except MG.Refused as err:
            raise Refused(str(err)) from None
    if path.startswith("settings."):
        name = path[len("settings."):]
        v = MG.VERSIONS.get(name)
        if v is not None and not v.in_manager:
            raise Refused(f"'{path}': it moves the trendline touch entry's own hedge (M27) - that entry waits for his "
                          "word 'steep' (B2-01), so it is not in his whole strategy yet")
        return "settings", V.check_change(forms.TYPE1.idea(), path, value), None
    if "/" in path:
        eid, sub = path.split("/", 1)
        if eid not in forms.ENTRY_TYPES:
            raise Refused(f"'{path}': {eid!r} is not one of his entry types ({', '.join(forms.ENTRY_TYPES)})")
        e = forms.ENTRY_TYPES[eid]
        if e not in ins:
            why = waiting().get(eid, "it is not in the whole")
            raise Refused(f"'{path}': {eid} is not in his whole strategy yet - it {why}")
        if sub == "timeframe" or sub.endswith(".tf"):
            raise Refused(f"'{path}': the hedging tester runs his whole strategy on the 5-minute chart - a timeframe "
                          f"is a version of ONE entry (its own family), {V.SECTION_D}")
        if sub.startswith(("settings.", "exit.management.")) or sub == AT_ONCE:
            raise Refused(f"'{path}': that is a change of the whole - write it without '{eid}/'")
        return "entry", f"{e.title.split(':')[0]}: " + V.check_change(e.idea(), sub, value), eid
    if path == "timeframe" or path.endswith(".tf"):
        raise Refused(f"'{path}': the hedging tester runs his whole strategy on the 5-minute chart - a timeframe is "
                      "a version of ONE entry (its own family)")
    raise Refused(f"'{path}': a version of his whole strategy changes only his management's test rows "
                  f"('exit.management.<name>'), '{AT_ONCE}', a number the machine chose ('settings.<name>') or one "
                  f"entry's own numbers ('<entry id>/steps.<id>.max_wait_bars' ...) - {V.SECTION_D}")


def _default(bucket: str, path: str, eid: str | None) -> Any:
    if bucket == "at_once":
        return ONE
    if bucket == "management":
        return settings.ALL[MG.name_of(path)].value
    if bucket == "settings":
        return settings.ALL[path[len("settings."):]].value
    return _form_value(forms.ENTRY_TYPES[eid], path.split("/", 1)[1])


def _same(a: Any, b: Any) -> bool:
    return a == b and type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                              and not isinstance(a, bool) and not isinstance(b, bool) and a == b)


def version(changes: Mapping[str, Any] | None = None) -> Version:
    """A version of the whole from its changes (dotted paths, see the module text). Each change is checked; one that
    sets version 1's own value is left out (the same rule, not a new try). Refused with the reason."""
    ins = entries_in()
    if not ins:
        raise Refused("no entry of his is settled - nothing to run")
    changes = dict(changes or {})
    kept, dims = {}, {}
    management, sett, entries, at_once = {}, {}, {}, ONE
    for path, value in changes.items():
        bucket, what, eid = _check_one(path, value, ins)
        if _same(value, _default(bucket, path, eid)):
            continue
        kept[path], dims[path] = value, what
        if bucket == "at_once":
            at_once = value
        elif bucket == "management":
            management[MG.name_of(path)] = value
        elif bucket == "settings":
            sett[path[len("settings."):]] = value
        else:
            entries.setdefault(eid, {})[path.split("/", 1)[1]] = value
    try:
        MG.check_needs(forms.TYPE1.idea(), {MG.PATH + k: x for k, x in management.items()})
    except MG.Refused as err:
        raise Refused(str(err)) from None
    rule = {"entries": {}, "management": sorted((k, repr(x)) for k, x in management.items()),
            "trades_at_once": at_once}
    for e in ins:
        own = {**{f"settings.{k}": x for k, x in sett.items()}, **entries.get(e.id, {})}
        try:
            rule["entries"][e.id] = version_of(e.idea(), None, own or None)
        except Exception as err:                          # noqa: BLE001 - the compiler's refusal, in its words
            raise Refused(f"{e.id}: {err}") from None
    vid = "s-" + hashlib.sha256(json.dumps(rule, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return Version(vid, kept, management, sett, entries, at_once, tuple(ins), dims, about(kept, ins))


def about(changes: Mapping[str, Any], ins: list[forms.EntryType] | None = None) -> str:
    """What a version is, in words."""
    ins = ins if ins is not None else entries_in()
    parts = []
    for path, value in changes.items():
        if path == AT_ONCE:
            parts.append("one trade at a time PER entry type (version 1: one across all of them, his #55)")
        elif path.startswith(MG.PATH):
            name = MG.name_of(path)
            parts.append(f"{name} = {value!r} (his management, T12 {MG.VERSIONS[name].row}; version 1 is "
                         f"{settings.ALL[name].value!r})")
        elif path.startswith("settings."):
            name = path[len("settings."):]
            parts.append(f"{name} = {value!r} (version 1 is {settings.ALL[name].value!r})")
        else:
            eid, sub = path.split("/", 1)
            e = forms.ENTRY_TYPES[eid]
            parts.append(f"{e.title.split(':')[0]}: {sub} = {value!r} (version 1 is {_form_value(e, sub)!r})")
    return " · ".join(parts) or "his strategy as written (version 1 of his management)"


# ================================================================================================= the window

@dataclass
class Window:
    """The years one run reads: its market (bars for the compile, warm-up included) and its 1-minute book (where
    trades fill and close). Compiles and market facts are kept for the next version."""
    label: str
    years: tuple[int, ...]
    market: Market
    m1: pd.DataFrame
    book: Any
    _comp: dict = field(default_factory=dict, repr=False)
    _facts: dict = field(default_factory=dict, repr=False)
    _seg: Segment | None = field(default=None, repr=False)

    @classmethod
    def tune(cls, log=print) -> "Window":
        """2016-2020: the years versions are tried and chosen on."""
        t0 = time.time()
        years = tuple(config.TUNE_YEARS)
        market = Market.load(str(years[0]), str(years[-1]), tfs=TFS)
        m1 = market.bars("1m")
        log(f"bars {years[0]}-{years[-1]} read in {time.time() - t0:.0f} s")
        return cls(f"{years[0]}-{years[-1]}", years, market, m1, engine.bars_book(m1, TF))

    @classmethod
    def check(cls, version_id: str, log=print) -> "Window":
        """2021-2023 - only once the family's check-opened.json names this version (guards.check_opened). The market
        starts in 2016 (the blocks' warm-up and the all-time high); trades fill and close only in 2021-2023."""
        guards.check_opened(STRATEGY, version=version_id)
        t0 = time.time()
        years = tuple(config.CHECK_YEARS)
        market = Market.load(str(config.TUNE_YEARS[0]), str(years[-1]), tfs=TFS)
        m1 = market.bars("1m").loc[str(years[0]):str(years[-1])]
        guards.check_opened(STRATEGY, [m1.index[0], m1.index[-1]], version=version_id)
        log(f"bars {config.TUNE_YEARS[0]}-{years[-1]} read in {time.time() - t0:.0f} s (trades in {years[0]}-"
            f"{years[-1]} only)")
        return cls(f"{years[0]}-{years[-1]}", years, market, m1, engine.bars_book(m1, TF))

    def inside(self, s) -> bool:
        t = pd.Timestamp(s.known_at)
        return self.book.first <= t <= self.book.last + MINUTE

    def compiled(self, e: forms.EntryType, changes: dict) -> Compiled:
        key = (e.id, json.dumps(changes, sort_keys=True, default=str))
        if key not in self._comp:
            self._comp[key] = compile_idea(e.idea(), self.market, changes=changes or None)
        return self._comp[key]

    def segment(self, signals: list) -> Segment:
        """The referee's window for the random twins (their bar closes, the price when a signal is known)."""
        if self._seg is None:
            self._seg = Segment(self.label, self.years, TF, [], self.m1, self.book,
                                self.m1["close"].to_numpy(dtype=float))
        return Segment(self._seg.label, self._seg.years, TF, list(signals), self._seg.m1, self._seg.book,
                       self._seg.close)


def signals_of(win: Window, v: Version, log=print) -> dict[str, list]:
    """{entry id: its signals known inside the window} under the version's numbers."""
    out = {}
    for e in v.ins:
        t0 = time.time()
        fresh = (e.id, json.dumps(v.entry_changes(e), sort_keys=True, default=str)) not in win._comp
        c = win.compiled(e, v.entry_changes(e))
        out[e.id] = [s for s in c.signals if win.inside(s)]
        if fresh:
            log(f"  {e.id}: {len(out[e.id])} setups in {win.label} ({time.time() - t0:.0f} s)")
    return out


def tune_part(win: Window, v: Version) -> dict[str, list]:
    """{entry id: its signals known up to the close of the last tune bar} - on the check window these must be
    exactly the tune run's signals (nothing after a moment changes a signal known then)."""
    out = {}
    last = win.market.bars("5m").loc[:str(config.TUNE_YEARS[-1])].index[-1] + pd.Timedelta(minutes=5)
    for e in v.ins:
        out[e.id] = [s for s in win.compiled(e, v.entry_changes(e)).signals if pd.Timestamp(s.known_at) <= last]
    return out


def _order(v: Version, sigs: dict[str, list]) -> list:
    """Every signal of the whole in known_at order; at the same moment, his entries in their order (forms.py)."""
    rank = {e.id: k for k, e in enumerate(v.ins)}
    return sorted((s for x in sigs.values() for s in x), key=lambda s: (pd.Timestamp(s.known_at), rank[s.idea_id],
                                                                         s.id))


# ================================================================================================= one run

@dataclass
class Run:
    version: Version
    window: str
    years: list[int]
    signals: dict                      # entry id -> its signals in the window
    hedged: hedging.HedgeReplay        # his entries with his hedge
    stop: engine.Replay                # his entries with a plain stop
    facts: hedging.MarketFacts
    cancel: dict                       # signal id -> the moment a check cancels its waiting order
    rules: dict
    seconds: float

    def entry_of(self, signal_id: str) -> str:
        return signal_id.split(".", 1)[0]


def _merge_hedged(parts: list[hedging.HedgeReplay]) -> hedging.HedgeReplay:
    p0 = parts[0]
    trades = sorted((t for p in parts for t in p.trades), key=lambda t: (t.fill_ts, t.signal_id))
    return hedging.HedgeReplay(trades, [x for p in parts for x in p.skipped], [x for p in parts for x in p.not_filled],
                               p0.first_bar, p0.last_bar, p0.tf, p0.rules, sum(p.signals for p in parts), p0.setups)


def _merge_stop(parts: list[engine.Replay]) -> engine.Replay:
    p0 = parts[0]
    trades = sorted((t for p in parts for t in p.trades), key=lambda t: (pd.Timestamp(t.fill_ts), t.signal_id))
    return engine.Replay(trades, [x for p in parts for x in p.skipped], [x for p in parts for x in p.not_filled],
                         p0.first_bar, p0.last_bar, p0.tf, len(parts), sum(p.signals for p in parts),
                         ["one trade at a time PER entry type (M18's test part)"])


def facts_for(win: Window, v: Version, sigs: dict[str, list], log=print) -> tuple[hedging.MarketFacts, dict, str]:
    """(the market facts the manager asks, the checks while an order waits, the settings' fingerprint) - his closing
    setups are the whole's own setups (M10)."""
    allsig = _order(v, sigs)
    gap = sigs.get(forms.TYPE1.id) or sigs.get(forms.SECOND_GAP.id) or []
    with settings.using(v.settings) as res:
        fp = settings.fingerprint(res)
        cancel: dict[str, int] = {}
        for e in v.ins:
            cancel.update(closers.fill_checks(e, sigs[e.id], win.market, fp, gap))
        key = (fp, signals_sha256(allsig), json.dumps(sorted(cancel.items())))
        if key not in win._facts:
            t0 = time.time()
            setups = hedging.setups_from(allsig, win.book, cancel)
            facts = hedging.MarketFacts.build(win.market, fp, win.book, setups=setups)
            facts.setup_signals = len(allsig)
            facts.closers = {**{e.id: len(sigs[e.id]) for e in v.ins}, **waiting()}
            win._facts.clear()                      # one set at a time (the market's own caches stay)
            win._facts[key] = facts
            log(f"  his closing setups: {len(setups)} filled of {len(allsig)}; news moments {len(facts.news)} "
                f"({time.time() - t0:.0f} s)")
    return win._facts[key], cancel, fp


def replays(v: Version, sigs: dict[str, list], facts, cancel: dict, book) -> tuple[hedging.HedgeReplay, engine.Replay,
                                                                                    dict]:
    """His hedge and a plain stop on these signals, one trade at a time (or one per entry type: M18's test part)."""
    with settings.using({**v.settings, **v.management}):
        R = hedging.Rules.now()
        groups = [_order(v, sigs)] if v.trades_at_once == ONE else [
            _order(v, {e.id: sigs.get(e.id, [])}) for e in v.ins]
        groups = [g for g in groups if g] or [[]]
        hs = [hedging.replay(g, facts, TF, R, cancel_by=cancel) for g in groups]
        ss = [engine.replay(g, None, TF, config.TRADES_AT_ONCE, book=book, cancel_by=cancel) for g in groups]
    if len(hs) == 1:
        return hs[0], ss[0], R.as_dict()
    return _merge_hedged(hs), _merge_stop(ss), R.as_dict()


def run(win: Window, v: Version, *, log=print) -> Run:
    """One version of the whole on the window: his entries with his hedge, and with a plain stop."""
    t0 = time.time()
    sigs = signals_of(win, v, log=log)
    facts, cancel, _ = facts_for(win, v, sigs, log=log)
    hedged, stop, R = replays(v, sigs, facts, cancel, win.book)
    return Run(v, win.label, list(win.years), sigs, hedged, stop, facts, cancel, R, time.time() - t0)


# ================================================================================================= the numbers

def closed_hedged(trades) -> pd.DataFrame:
    """Closed hedged trades: net_inr, exit_time, signal (the referee's shape)."""
    rows = [(t.net_inr, t.exit_ts, t.signal_id) for t in trades if t.status == "closed"]
    return pd.DataFrame(rows, columns=["net_inr", "exit_time", "signal"])


def closed_stop(trades) -> pd.DataFrame:
    rows = [(t.net_inr, pd.Timestamp(t.exit_ts), t.signal_id) for t in trades if t.status == "closed"]
    return pd.DataFrame(rows, columns=["net_inr", "exit_time", "signal"])


def _part(trades, years, entry: str | None) -> dict:
    """Trades of one entry type (None: all): per year, count, open at end."""
    mine = [t for t in trades if entry is None or t.signal_id.split(".", 1)[0] == entry]
    closed = [t for t in mine if t.status == "closed"]
    per = {str(y): 0.0 for y in years}
    for t in closed:
        per[str(pd.Timestamp(t.exit_ts).year)] += t.net_inr
    net = sum(t.net_inr for t in closed)
    worst = min(per, key=per.get) if per else None
    return {"trades": len(closed), "net_inr_total": round(net, 2),
            "net_inr_per_trade": round(net / len(closed), 2) if closed else None,
            "won": sum(t.net_inr > 0 for t in closed),
            "per_year": {y: round(x, 2) for y, x in per.items()}, "worst_year": worst,
            "open_at_end": sum(t.status != "closed" for t in mine)}


def by_entry(run_: Run) -> dict:
    """Each entry type's part of the whole: the trades it STARTED (a hedged trade is handed to later setups of his,
    M15 - it stays the trade of the entry that opened it)."""
    out = {}
    for e in run_.version.ins:
        n = len(run_.signals.get(e.id, []))
        out[e.id] = {"setups": n, "hedged": _part(run_.hedged.trades, run_.years, e.id),
                     "stop": _part(run_.stop.trades, run_.years, e.id),
                     "skipped_hedged": sum(s.startswith(e.id + ".") for s in run_.hedged.skipped),
                     "skipped_stop": sum(s.startswith(e.id + ".") for s in run_.stop.skipped)}
    return out


def run_summary(run_: Run) -> dict:
    """What a version's trial file keeps: his hedge (hedging.summarize), the plain stop (the tester's), per entry type,
    the checks while an order waits, his closing setups, every entry's signals fingerprint."""
    cancelled_h = sum(why == engine.CHECK_BEFORE_FILL for _, why in run_.hedged.not_filled)
    cancelled_s = sum(why == engine.CHECK_BEFORE_FILL for _, why in run_.stop.not_filled)
    return {
        "window": {"years": run_.window, "first_bar": str(run_.hedged.first_bar), "last_bar": str(run_.hedged.last_bar)},
        "hedged": hedging.summarize(run_.hedged),
        "stop": stop_summary(run_.stop),
        "by_entry": by_entry(run_),
        "setups": {e: len(s) for e, s in run_.signals.items()},
        "signals_sha256": {e: signals_sha256(s) for e, s in run_.signals.items()},
        "fill_checks": {"orders": len(run_.cancel), "cancelled_hedged": cancelled_h, "cancelled_stop": cancelled_s},
        "closing_setups": {"which": run_.facts.closers, "signals": run_.facts.setup_signals,
                           "filled": len(run_.facts.setups)},
        "rules": run_.rules, "seconds": round(run_.seconds, 1),
    }


# ================================================================================================= the 2x2

def two_by_two(win: Window, run_: Run, n_runs: int = config.RANDOM_RUNS, *, log=print) -> dict:
    """{his entries, random entries} x {his hedging, a plain stop} for one version on the window (L19a, #114 - never
    buy-and-hold): random entries = one twin per signal of the whole (machine/tester/hedging_2x2.twins: the same
    month, direction, entry type, distances, waiting time; a random bar close in that month, a limit distance drawn
    from his own), the SAME twins through both managements, one at a time the version's way. The checks while an order
    waits are his entries' own rules: twins do not get them. Per entry type too (the twins of that entry's setups).
    Also the referee's facts for the hedged box: random runs' rupees a trade and every random trade (the power table).
    """
    v = run_.version
    allsig = _order(v, run_.signals)
    years = list(win.years)
    seg = win.segment(allsig)
    seed = int(hashlib.sha256(json.dumps([s.id for s in allsig]).encode("utf-8")).hexdigest()[:12], 16)
    groups = [None, *v.entry_ids]
    rand: dict = {g: {"hedge": [], "stop": []} for g in groups}
    per_trade, counts, nets = [], [], []
    t0 = time.time()
    for r in range(n_runs):
        tw = H2.twins(seg, allsig, np.random.default_rng([seed, r]), r)
        by = {e.id: [t for t in tw if t.idea_id == e.id] for e in v.ins}
        h, s, _ = replays(v, by, run_.facts, {}, win.book)
        for g in groups:
            hh = SimpleNamespace(trades=[t for t in h.trades if g is None or t.idea_id == g])
            ss = SimpleNamespace(trades=[t for t in s.trades if g is None or t.idea_id == g])
            rand[g]["hedge"].append(H2._hedge_by_year(hh, years))
            rand[g]["stop"].append(H2._stop_by_year(ss, years))
        c = [t.net_inr for t in h.trades if t.status == "closed"]
        nets += c
        counts.append(len(c))
        per_trade.append(float(np.mean(c)) if c else None)
        if (r + 1) % 25 == 0:
            log(f"    random run {r + 1}/{n_runs} ({time.time() - t0:.0f} s)")
    out = {"runs": n_runs, "seed": seed, "years": [str(y) for y in years], "groups": {}}
    for g in groups:
        his_h = SimpleNamespace(trades=[t for t in run_.hedged.trades if g is None or t.idea_id == g])
        his_s = SimpleNamespace(trades=[t for t in run_.stop.trades if g is None or t.idea_id == g])
        res = H2._result(his_h, his_s, rand[g], years)
        # every random run's total (so two windows' runs can be added up, run by run, for all the years)
        res["run_totals"] = {box: [round(float(sum(d.values())), 2) for d, _, _ in rand[g][box]]
                             for box in ("hedge", "stop")}
        out["groups"]["all" if g is None else g] = res
    pts = np.array([x for x in per_trade if x is not None])
    his = closed_hedged(run_.hedged.trades)
    mine = float(his["net_inr"].mean()) if len(his) else None
    beaten = int((pts < mine).sum()) if mine is not None and pts.size else 0
    out["random_hedged"] = {
        "runs": n_runs, "runs_with_trades": int(pts.size), "beaten": beaten,
        "beaten_share": round(beaten / pts.size, 4) if pts.size else 0.0,
        "per_trade_median": round(float(np.median(pts)), 2) if pts.size else None,
        "per_trade_5_95": ([round(float(np.percentile(pts, 5)), 2), round(float(np.percentile(pts, 95)), 2)]
                           if pts.size else None),
        "trades_median": int(np.median(counts)) if counts else 0, "trades_wanted": len(his)}
    out["random_hedged_nets"] = [round(float(x), 2) for x in nets]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def years_of(trades: pd.DataFrame) -> set[int]:
    return set(pd.DatetimeIndex(trades["exit_time"]).year) if len(trades) else set()


def week_luck(trades: pd.DataFrame) -> float | None:
    """The referee's luck test on these trades (their weeks drawn again in a world with no edge)."""
    if len(trades) < 2:
        return None
    return float(stats.bootstrap_p(trades["net_inr"].to_numpy(dtype=float), stats.week_ids(trades["exit_time"])))
