"""A `git pull` for Tejas's PC, the bot host (L9c). Use it instead of `git pull` in c:\\ML Bot.

    .venv\\Scripts\\python -m machine.tools.pull

Why: the bot writes its files into this clone and the nightly job (machine/telegram/nightly.py) puts copies of them on
GitHub. After he merges that pull request, main has files that are ALSO lying here, uncommitted - and a plain
`git pull` refuses ("untracked working tree files would be overwritten"), even when the two are the same, byte for
byte (tested with git 2.43).

What it does: fetch; find every file here that stands in the way of the pull; then, per file -
    the same as GitHub's                          -> set aside, pull, the pulled copy is the same (the aside is deleted)
    add-only file (config.BOT_APPEND_ONLY) = GitHub's + lines the bot added since -> set aside, pull, put back
    anything else                                 -> STOP before the pull, touch nothing, name the files
and a lesson still waiting here that GitHub already numbered (the same file in learned-inbox/merged/) is removed after
the pull (it would be counted twice). Set-aside copies wait in data/pull-aside/<time>/ until the pull worked; if it
fails they stay there and nothing is lost.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from machine import config
from machine.telegram.nightly import blob_id, cat, git, matches

INBOX = "strategy/learned-inbox/"


@dataclass
class Aside:
    path: str
    kind: str          # same / more / numbered / differs
    local_id: str
    content: bytes = b""


def upstream(repo: Path) -> str:
    branch = git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.decode().strip()
    if not branch:
        raise SystemExit("not on a branch (a detached HEAD) - check out main first")
    ref = f"origin/{branch}"
    if git(repo, "rev-parse", "--verify", "-q", ref, check=False).returncode:
        raise SystemExit(f"{ref} does not exist on GitHub")
    return ref


def local_changes(repo: Path) -> tuple[set[str], set[str]]:
    """(untracked files, tracked files changed here)."""
    out = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.decode("utf-8").split("\0")
    untracked, changed, i = set(), set(), 0
    while i < len(out):
        e = out[i]
        i += 1
        if len(e) < 4:
            continue
        xy, path = e[:2], e[3:]
        if xy[0] in "RC":
            i += 1
        (untracked if xy == "??" else changed).add(path)
    return untracked, changed


def plan(repo: Path, up: str) -> list[Aside]:
    """What stands in the way of merging `up`, and what to do with each."""
    base = git(repo, "merge-base", "HEAD", up).stdout.decode().strip()
    incoming = [p for p in git(repo, "diff", "--no-renames", "--name-only", "-z", base, up).stdout.decode("utf-8")
                .split("\0") if p]
    untracked, changed = local_changes(repo)
    out = []
    for path in incoming:
        if path not in untracked and path not in changed:
            continue
        file = repo / path
        if not file.is_file():
            continue
        raw = file.read_bytes()
        here = git(repo, "hash-object", "-w", "--stdin", f"--path={path}", input=raw).stdout.decode().strip()
        theirs = blob_id(repo, up, path)
        if here == theirs:
            out.append(Aside(path, "same", here))
        elif theirs and matches(path, config.BOT_APPEND_ONLY) and cat(repo, here).startswith(cat(repo, theirs)):
            out.append(Aside(path, "more", here, cat(repo, here)))
        else:
            out.append(Aside(path, "differs", here))
    for path in sorted(untracked):
        if path.startswith(INBOX) and path.count("/") == 2 and blob_id(repo, up, path) is None:
            raw = (repo / path).read_bytes()
            here = git(repo, "hash-object", "-w", "--stdin", f"--path={path}", input=raw).stdout.decode().strip()
            if blob_id(repo, up, INBOX + "merged/" + path.rsplit("/", 1)[1]) == here:
                out.append(Aside(path, "numbered", here))
    return out


def pull(repo: Path = config.ROOT, *, log=print) -> int:
    repo = Path(repo).resolve()
    fetched = git(repo, "fetch", "origin", check=False)
    if fetched.returncode:
        log("could not fetch from GitHub: " + fetched.stderr.decode("utf-8", "replace").strip()[:300])
        return 1
    up = upstream(repo)
    todo = plan(repo, up)
    bad = [a for a in todo if a.kind == "differs"]
    if bad:
        log("STOPPED before the pull - these files here differ from GitHub's, and they are not simply the bot's "
            "extra lines. Nothing was touched:")
        for a in bad:
            log(f"  {a.path}")
        return 1
    folder = config.DATA_DIR / "pull-aside" / dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    for a in todo:
        dest = folder / a.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(repo / a.path), str(dest))
    merged = git(repo, "merge", "--no-edit", up, check=False)
    log((merged.stdout + merged.stderr).decode("utf-8", "replace").strip())
    if merged.returncode:
        for a in todo:                       # put back what is not there now; the set-aside copies stay as well
            if not (repo / a.path).exists():
                shutil.copy2(folder / a.path, repo / a.path)
        log(f"the pull did NOT work (above). The bot's files were put back; copies also wait in {folder}")
        return 1
    kept = []
    for a in todo:
        now = repo / a.path
        if a.kind == "more":
            now.write_bytes(a.content)                    # GitHub's copy + the lines the bot added since
        if a.kind == "numbered":
            (folder / a.path).unlink()
            continue
        check = git(repo, "hash-object", f"--path={a.path}", str(now)).stdout.decode().strip() if now.is_file() \
            else None
        if check == a.local_id:
            (folder / a.path).unlink()
        else:
            kept.append(a.path)
    for d in sorted(folder.rglob("*"), reverse=True) if folder.exists() else []:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if folder.exists() and not any(folder.iterdir()):
        folder.rmdir()
    counts = {k: sum(a.kind == k for a in todo) for k in ("same", "more", "numbered")}
    if any(counts.values()):
        log(f"bot files: {counts['same']} already on GitHub, {counts['more']} with lines the bot added since "
            f"(kept), {counts['numbered']} numbered lessons removed from the inbox (they are in merged/)")
    if kept:
        log(f"check these (a copy waits in {folder}): " + ", ".join(kept))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argparse.ArgumentParser(prog="python -m machine.tools.pull", description=__doc__.split("\n\n")[0]).parse_args(argv)
    return pull()


if __name__ == "__main__":
    raise SystemExit(main())
