"""The public copy of the machine (L11): what goes to the public GitHub repo, and the check that nothing else does.

    .venv\\Scripts\\python -m machine.tools.publish <folder>            write the public copy into <folder>
    .venv\\Scripts\\python -m machine.tools.publish <folder> --check    only list what would go and run the checks

What goes out (an ALLOW-list: a file not named here never goes):
  - committed files only (git HEAD - never an uncommitted or untracked file of this folder) under PUBLIC_DIRS, and the
    exact files in PUBLIC_FILES;
  - the files in public/ (this repo), copied to the copy's root: the public README, CI, .gitignore and the trimmed
    plan/ README (public/README.md -> README.md ...);
  - PRIVATE-PARTS.md at the root, made here: what was left out and why (tests/conftest.py reads it - a test that needs
    a left-out file is reported as skipped, with the file it needs).
What never goes (Tejas, 1 Oct 2026, inbox 1745 and 1752; plan/likhith.md L11): strategy/ (his notes, LEARNED.md, his
pictures, the Telegram archive), old/, data/ except four small public tables, plan/ except the trimmed README,
prior-tries.csv and the demo script, the private CLAUDE.md / README / SETUP and CI.
The checks (any finding: nothing is written, exit 1): no path under a private folder; no text that names the private
repo, a personal e-mail of this repo's git history, a local user folder, or anything shaped like a key (Telegram bot
token, Anthropic / OpenAI / GitHub / AWS / Voyage keys, a private key block). gitleaks then scans the public repo's
full history (plan/work/demo-script.md says how).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from machine import config

PUBLIC_DIRS = ("machine/", "tests/", "docs/")
PUBLIC_FILES = (
    "pyproject.toml", "requirements.txt", "requirements-search.txt", "requirements-live.txt", ".env.example",
    # four small public tables the money and news code needs (not price data): Rs per $ by month (RBI / FBIL), the
    # demo broker's swap readings, which way each US number is good for the dollar, the 2-year Treasury yield (FRED)
    "data/usd_inr.csv", "data/swap.csv", "data/news/signs.csv", "data/news/dgs2.csv",
    # the old project's tries (the referee's rule 6: "years already seen") and the demo script
    "plan/work/prior-tries.csv", "plan/work/demo-script.md",
)
OVERLAY_DIR = "public"                                 # public/<path> -> <path> in the copy
LEFT_OUT = (                                           # (path, why) - written into PRIVATE-PARTS.md
    ("strategy/", "the trader's own notes: strategy.md, the numbered lessons (LEARNED.md), his marked chart "
                  "pictures and the Telegram archive - his private knowledge"),
    ("old/", "the old system (before 27 Sep 2026), kept for reference only"),
    ("data/", "the gold prices (1-minute bars 2016-2023, and the sealed 2024+ rows), the news calendar, the bot's "
              "files - only four small public tables are in this copy"),
    ("plan/work/", "his ideas, his answers, the question queue, every trial file (one per test run), the notes of "
                   "each build step - only prior-tries.csv and demo-script.md are in this copy"),
    ("plan/PROGRESS.md, plan/tejas.md, plan/likhith.md", "the working plan of the two people who build it"),
)
NEVER = ("strategy/", "old/", "CLAUDE.md", ".gitleaksignore", ".github/gitleaks-new-commits.sh")
SKIP_IN_DIRS = ("docs/SETUP.md",)                      # the private repo's setup (its clone address, his PC)

# anything shaped like a key or a token (any of these in the copy = stop)
SECRET_PATTERNS = (
    ("telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("anthropic key", re.compile(r"sk-ant-[a-z]+\d{2}-[A-Za-z0-9_-]{40,}")),     # sk-ant-api03-... (a real key's shape)
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
    ("github token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}")),
    ("aws key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("voyage key", re.compile(r"\bpa-[A-Za-z0-9_-]{30,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
# text that names private places (the private repo's own name is read from its git remote: private_names)
PRIVATE_TEXT = (
    ("a local user folder", re.compile(r"[Cc]:[\\/]+Users[\\/]+[A-Za-z]|[Cc]:[\\/]+Tej\b|/c/Users/[A-Za-z]")),
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".jsonl", ".csv", ".yml", ".yaml", ".cfg", ".ini", ".sh",
                 ".bat", ".ps1", ".html", ".example", ".gitignore", ".gitattributes", ""}


class PublishError(RuntimeError):
    pass


@dataclass
class Plan:
    files: dict[str, bytes] = field(default_factory=dict)      # path in the copy -> bytes
    findings: list[str] = field(default_factory=list)


def _git(*args: str, root: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def personal_emails(root: Path) -> set[str]:
    """Every author / committer e-mail of this repo's history that is not a no-reply address (they must not leak)."""
    out = _git("log", "--all", "--format=%ae%n%ce", root=root).decode("utf-8", "replace").split()
    return {e.lower() for e in out if "@" in e and "noreply" not in e.lower()}


def private_names(root: Path) -> set[str]:
    """The private repo's own name, from its remotes' addresses (github.com/<owner>/<name>.git -> <name>). Read here,
    not written in this file, so the public copy does not carry it."""
    out = _git("remote", "-v", root=root).decode("utf-8", "replace").split()
    names = {re.sub(r"\.git$", "", u.rstrip("/").split("/")[-1].split(":")[-1]) for u in out if "/" in u or ":" in u}
    return {n.lower() for n in names if len(n) >= 6}


def plan_copy(root: Path | None = None) -> Plan:
    """What the copy holds (from git HEAD + public/) and every finding of the checks. Writes nothing."""
    root = Path(root or config.ROOT)
    tracked = _git("ls-files", "-z", root=root).decode("utf-8").split("\0")
    p = Plan()
    for path in tracked:
        if not path:
            continue
        if path.startswith(OVERLAY_DIR + "/"):
            dest = path[len(OVERLAY_DIR) + 1:]
        elif path in PUBLIC_FILES or (path.startswith(PUBLIC_DIRS) and path not in SKIP_IN_DIRS):
            dest = path
        else:
            continue
        p.files[dest] = _git("show", f"HEAD:{path}", root=root)
    for need in PUBLIC_FILES:
        if need not in p.files:
            p.findings.append(f"missing: {need} is not committed")
    p.files["PRIVATE-PARTS.md"] = private_parts().encode("utf-8")
    p.findings += check(p.files, personal_emails(root), private_names(root))
    return p


def private_parts() -> str:
    rows = "\n".join(f"| `{path}` | {why} |" for path, why in LEFT_OUT)
    return f"""# What this public copy leaves out

This repo is the public copy of a private working repo, made by `machine/tools/publish.py` (an allow-list: a file
not on its list never comes here). Left out on purpose:

| Path | What it is |
|---|---|
{rows}

**The tests:** `tests/conftest.py` sees this file. A test that fails only because it needs one of the files above is
reported as **skipped**, with the file it needs (`pytest -rs` lists them). Tests marked `private` skip here too.
Tests that need the real gold prices skip as before (`data` marker). Everything else runs on made-up data.

**The demo:** `python -m machine.demo` runs every step on made-up prices and made-up notes in `tests/data/demo/`.
"""


def check(files: dict[str, bytes], emails: set[str], names: set[str] = frozenset()) -> list[str]:
    """Every finding: a private path, key-shaped text, a local user folder, a personal e-mail, a private name."""
    out = []
    for path, body in sorted(files.items()):
        if path.startswith(NEVER) or any(path == n.rstrip("/") for n in NEVER):
            out.append(f"{path}: a private path")
        if Path(path).suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = body.decode("utf-8", "replace")
        for name, rx in SECRET_PATTERNS + PRIVATE_TEXT:
            m = rx.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                out.append(f"{path}:{line}: {name}")
        low = text.lower()
        for e in emails:
            if e in low:
                out.append(f"{path}: a personal e-mail of this repo's history")
        for n in names:
            if n in low:
                out.append(f"{path}: the private repo's name")
    return out


def write(p: Plan, target: Path) -> None:
    """Replace everything in `target` except its .git folder with the copy."""
    if p.findings:
        raise PublishError("not written - the checks found:\n  " + "\n  ".join(p.findings))
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for path, body in p.files.items():
        out = target / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m machine.tools.publish", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="the public repo's folder (its .git is kept)")
    ap.add_argument("--check", action="store_true", help="only list and check; write nothing")
    a = ap.parse_args(argv)
    if Path(a.target).resolve() == Path(config.ROOT).resolve():
        print("refused: the target is this repo", file=sys.stderr)
        return 2
    p = plan_copy()
    size = sum(len(b) for b in p.files.values())
    tops = sorted({f.split("/")[0] for f in p.files})
    print(f"{len(p.files)} files, {size / 1e6:.1f} MB: {', '.join(tops)}")
    if p.findings:
        print("the checks found:\n  " + "\n  ".join(p.findings), file=sys.stderr)
        return 1
    print("checks: no private path, no private text, nothing shaped like a key")
    if not a.check:
        write(p, a.target)
        print(f"written to {a.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
