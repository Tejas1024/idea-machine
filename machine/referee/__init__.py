"""Judges a tested idea: works / chance / chance — needs fresh years / not enough trades to tell (L5)."""
# judge.py            the way in: judge_check (the check years, opened once) and judge_walk_forward (later versions)
# tune.py             every version on 2016-2020 (a trial file each), ONE chosen by Deflated Sharpe (L10)
# trials.py           versions per idea, family and project, from plan/work/trials/*.json; the version budget
# splits.py           tune 2016-2020 · check 2021-2023 opened ONCE per family (check-opened.json) · walk-forward folds
# prior.py            "already seen": plan/work/prior-tries.csv
# random_entries.py   200 runs: same exits, distances, cost, trades per month; random times and limit levels
# fake_levels.py      the level moved to its neighbours
# stats.py            per year, worst year, without the best 3, the weekly luck test; Deflated Sharpe + PBO (tune only)
# power.py            the smallest edge the test could see
# verdict.py          the rules, fixed in advance, printed in every report
# project.py          ideas judged, "works" count, "works" expected by luck
# selftest.py         the referee's own exams: python -m machine.referee.selftest [--monday]
