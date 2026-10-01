"""How many tries (L5, step 1): counted from the trial files in plan/work/trials/, never from memory.

A VERSION is one (idea, version) pair that has at least one trial file. An idea FAMILY is an idea and every idea whose
parent_idea chain leads back to it (likhith.md section D: an idea with the same blocks, timeframe and entry as an older
idea is a version OF that older idea). The family's root is the oldest idea in the chain.

    from machine.referee import trials
    c = trials.count("idea-001")
    print(c.text())        # "idea-001: 3 versions · family idea-001: 5 versions of 2 ideas (budget 20) · project: ..."
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from machine import config
from machine import trials as trial_files


def load(folder: Path | None = None) -> list[dict]:
    """Every trial file, oldest first."""
    return trial_files.read_all(folder)


def parents(files: list[dict]) -> dict[str, str]:
    """idea -> its parent idea, from every trial file that names one. Two different parents for one idea is refused:
    the family count would depend on which file was read first."""
    out: dict[str, str] = {}
    for f in files:
        idea, parent = f.get("idea_id"), f.get("parent_idea")
        if not idea or not parent:
            continue
        if out.get(idea, parent) != parent:
            raise ValueError(f"{idea} names two parent ideas in plan/work/trials/: {out[idea]} and {parent}")
        out[idea] = parent
    return out


def root_of(idea: str, links: dict[str, str], parent: str | None = None) -> str:
    """The oldest idea of this idea's family (itself when it has no parent)."""
    links = dict(links)
    if parent:
        if links.get(idea, parent) != parent:
            raise ValueError(f"{idea} already has the parent {links[idea]} in plan/work/trials/, not {parent}")
        links[idea] = parent
    seen = [idea]
    while seen[-1] in links:
        nxt = links[seen[-1]]
        if nxt in seen:
            raise ValueError(f"the parent_idea chain loops: {' -> '.join([*seen, nxt])}")
        seen.append(nxt)
    return seen[-1]


@dataclass
class Counts:
    idea_id: str
    family: str
    idea_versions: list[str]                    # this idea's versions, oldest first
    family_ideas: list[str]
    family_versions: list[tuple[str, str]]      # (idea, version), oldest first
    project_ideas: int                          # ideas with a trial file (controls and samples included)
    project_families: int
    project_versions: int
    trial_files: int
    budget: int = field(default_factory=lambda: config.VERSION_BUDGET)

    @property
    def budget_left(self) -> int:
        return self.budget - len(self.family_versions)

    def text(self) -> str:
        fam = (f"family {self.family}: {len(self.family_versions)} versions of {len(self.family_ideas)} ideas"
               if len(self.family_ideas) > 1 else "no other ideas in its family")
        return (f"Tries: {self.idea_id} {len(self.idea_versions)} version(s) · {fam} (budget {self.budget}) · "
                f"project: {self.project_versions} versions of {self.project_ideas} ideas, "
                f"{self.trial_files} test runs")

    def as_dict(self) -> dict:
        return {"idea_id": self.idea_id, "family": self.family, "idea_versions": self.idea_versions,
                "family_ideas": self.family_ideas, "family_versions": [list(v) for v in self.family_versions],
                "project_ideas": self.project_ideas, "project_families": self.project_families,
                "project_versions": self.project_versions, "trial_files": self.trial_files, "budget": self.budget}


def count(idea: str, parent: str | None = None, folder: Path | None = None,
          extra: tuple[str, str] | None = None) -> Counts:
    """The counts for one idea, its family and the project.

    parent: the idea's parent when it has no trial file yet. extra: an (idea, version) about to be tried, counted as if
    its file already existed (the referee counts the run it is judging, and its file in trial_files).
    """
    files = load(folder)
    links = parents(files)
    if parent:
        links = {**links, idea: links.get(idea, parent)}
    pairs: list[tuple[str, str]] = []
    for f in files:
        pair = (f["idea_id"], f["version_id"])
        if pair not in pairs:
            pairs.append(pair)
    if extra and extra not in pairs:
        pairs.append(extra)
    family = root_of(idea, links)
    fam_of = {i: root_of(i, links) for i, _ in pairs} | {idea: family}
    fam_pairs = [p for p in pairs if fam_of[p[0]] == family]
    by_family: dict[str, set[str]] = defaultdict(set)
    for i, _ in pairs:
        by_family[fam_of[i]].add(i)
    return Counts(
        idea_id=idea, family=family,
        idea_versions=[v for i, v in pairs if i == idea],
        family_ideas=sorted({i for i, _ in fam_pairs} | {idea}),
        family_versions=fam_pairs,
        project_ideas=len({i for i, _ in pairs}), project_families=len(by_family), project_versions=len(pairs),
        trial_files=len(files) + (1 if extra else 0),          # extra: this run's own file, about to be written
    )


def check_budget(idea: str, version: str, parent: str | None = None, folder: Path | None = None) -> Counts:
    """Refuse a NEW version once the family has used its budget (config.VERSION_BUDGET, a setting — likhith.md D)."""
    c = count(idea, parent, folder)
    if (idea, version) not in c.family_versions and c.budget_left <= 0:
        raise ValueError(f"the {c.family} family already has {len(c.family_versions)} versions — the budget is "
                         f"{c.budget} per idea (config.VERSION_BUDGET). A new version is not tried.")
    return c
