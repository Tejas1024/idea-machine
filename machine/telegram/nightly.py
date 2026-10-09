"""The bot's files reach git every night (L9c). Runs on Tejas's PC from Task Scheduler.

    .venv\\Scripts\\python -m machine.telegram.nightly               the job: copy, commit, push, pull request / link
    .venv\\Scripts\\python -m machine.telegram.nightly --dry-run     show what it would copy; change nothing, send nothing
    .venv\\Scripts\\python -m machine.telegram.nightly --install     add the nightly task to Task Scheduler (--at 23:30)
    .venv\\Scripts\\python -m machine.telegram.nightly --uninstall   remove it
    .venv\\Scripts\\python -m machine.telegram.nightly --last        the last run's result

The bot writes into THIS clone (the one it runs from, c:\\ML Bot): lessons, cut pictures, answers, idea states, trial
files. The job never commits here (a Claude session may be working in it). Instead, in a SEPARATE clone that only the
job uses (config.BOT_CLONE = c:\\ML Bot-bot; made the first time, marked as the job's own):

    1. fetch GitHub; start the branch bot/<date> from GitHub's main (bot/<date>-2 if that name is taken);
    2. copy in the bot's files that GitHub's main does not have yet - only paths in config.BOT_GIT_FILES, never data/
       or .env, never a file that looks like it holds a key, never a file git ignores;
    3. commit, push the branch, open a pull request with `gh pr create` (when gh is installed and logged in), else
       message Tejas the link that opens it;
    4. anything that fails -> a message to Tejas on Telegram (and data/telegram/nightly.log).

Which version of a file goes in (three copies: HIS PC now, his PC's last commit = the base, GitHub's main):
    same as main                          -> nothing to do
    main still = the base                 -> his PC's copy (new, or changed only on his PC)
    his PC still = the base               -> nothing (only GitHub changed it)
    a lesson GitHub already numbered      -> nothing (it sits in learned-inbox/merged/ on main)
    add-only file (config.BOT_APPEND_ONLY), both added lines -> GitHub's lines, then the lines his PC added
    anything else                         -> not copied; the message names it ("changed on both")
Files are compared by git's own id (so Windows line endings never count as a change). Deleting is never copied.

A message goes to Tejas only when there is something to merge or something went wrong; a quiet night is only logged
(the bot's STATUS shows the last run). Sending a message is not polling: it never disturbs the one bot (B12).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from machine import config

MARKER = "mlbot.nightlyclone"             # `git config` key set in the job's own clone: only there does it reset/clean
MAIN = "main"
KEY_NAMES = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_TEST_BOT_TOKEN", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "MT5_PASSWORD")
KEY_SHAPES = (re.compile(rb"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"),       # a Telegram bot key
              re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),              # an Anthropic key
              re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"))             # a GitHub key


class NightlyError(RuntimeError):
    """A failure with its reason in plain words (the message to Tejas quotes it)."""


# ================================================================================================= which files

def matches(path: str, patterns=config.BOT_GIT_FILES) -> bool:
    """path (posix, from the repo root) fits one pattern: same number of parts, each part fits (a * never crosses a
    folder)."""
    parts = path.split("/")
    for pat in patterns:
        pp = pat.split("/")
        if len(pp) == len(parts) and all(fnmatchcase(a, b) for a, b in zip(parts, pp)):
            return True
    return False


def forbidden(path: str) -> str | None:
    """A path that must never reach git, whatever the patterns say."""
    parts = path.split("/")
    if parts[0] == "data":
        return "data/ never goes to git"
    if any(p == ".git" for p in parts):
        return "inside .git"
    if parts[-1].lower().startswith(".env"):
        return ".env never goes to git"
    return None


def candidates(dev: Path) -> list[str]:
    """The bot's files that are new or changed in the bot's clone (git status: untracked, added or modified)."""
    out = git(dev, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.decode("utf-8")
    items, entries, i = [], out.split("\0"), 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if len(e) < 4:
            continue
        xy, path = e[:2], e[3:]
        if xy[0] in "RC":                                  # a rename: the old name follows; skip it
            i += 1
        if xy == "??" or ("D" not in xy and any(c in "MARC" for c in xy)):
            if matches(path) and not forbidden(path):
                items.append(path)
    return sorted(set(items))


# ================================================================================================= one file

@dataclass
class Pick:
    path: str
    action: str                 # copy / join / skip / conflict / refused
    why: str
    content: bytes | None = None


def decide(dev: Path, clone: Path, path: str, main_ref: str) -> Pick:
    local_path = dev / path
    if not local_path.is_file():
        return Pick(path, "skip", "gone from his PC")
    reason = forbidden(path)
    if reason:
        return Pick(path, "refused", reason)
    raw = local_path.read_bytes()
    key = has_key(raw)
    if key:
        return Pick(path, "refused", f"looks like it holds a key ({key})")
    if git(clone, "check-ignore", "-q", "--no-index", "--", path, check=False).returncode == 0:
        return Pick(path, "refused", "git ignores this path")
    here = git(clone, "hash-object", "-w", "--stdin", f"--path={path}", input=raw).stdout.decode().strip()
    base = blob_id(dev, "HEAD", path)
    main = blob_id(clone, main_ref, path)
    if here == main:
        return Pick(path, "skip", "GitHub has it already")
    if path.startswith("strategy/learned-inbox/") and main is None:
        numbered = blob_id(clone, main_ref, "strategy/learned-inbox/merged/" + path.rsplit("/", 1)[1])
        if numbered == here:
            return Pick(path, "skip", "GitHub numbered it already (learned-inbox/merged/)")
        if numbered is not None:
            return Pick(path, "conflict", "GitHub numbered an older copy of it; his PC has more in it")
    text = cat(clone, here)
    if main == base:
        return Pick(path, "copy", "new" if main is None else "changed only on his PC", text)
    if here == base:
        return Pick(path, "skip", "only GitHub changed it")
    if matches(path, config.BOT_APPEND_ONLY):
        theirs = cat(clone, main) if main else b""
        mine_base = cat(dev, base) if base else b""
        if text.startswith(theirs):
            return Pick(path, "copy", "his PC added lines after GitHub's", text)
        if theirs.startswith(text):
            return Pick(path, "skip", "GitHub has all of it and more")
        if text.startswith(mine_base) and theirs.startswith(mine_base):
            added = text[len(mine_base):]
            joined = theirs + (b"" if not theirs or theirs.endswith(b"\n") else b"\n") + added
            return Pick(path, "join", "both added lines: GitHub's first, then his PC's", joined)
    return Pick(path, "conflict", "changed both on his PC and on GitHub")


def has_key(data: bytes) -> str | None:
    for name in KEY_NAMES:
        value = config.env(name)
        if value and len(value) >= 8 and value.encode() in data:
            return name
    for shape in KEY_SHAPES:
        if shape.search(data):
            return "the shape of a key"
    return None


# ================================================================================================= the run

@dataclass
class Result:
    at: str
    branch: str | None = None
    commit: str | None = None
    files: list[dict] = field(default_factory=list)       # every Pick (without its content)
    pr: str | None = None                                   # the pull request's address (gh)
    link: str | None = None                                 # the link that opens one (no gh)
    older: list[str] = field(default_factory=list)          # older bot/ branches not merged yet
    error: str | None = None
    dry: bool = False

    def of(self, *actions: str) -> list[dict]:
        return [f for f in self.files if f["action"] in actions]


def run(*, dev: Path = config.ROOT, clone: Path = config.BOT_CLONE, today: dt.date | None = None, dry: bool = False,
        gh: str | list | None = "auto", log=print) -> Result:
    """The whole job. Raises NightlyError (plain words) or anything else on failure; main() turns that into a message."""
    dev, clone = Path(dev).resolve(), Path(clone).resolve()
    today = today or dt.date.today()
    res = Result(at=_now(), dry=dry)
    prepare_clone(dev, clone, log=log)
    fetched = git(clone, "fetch", "--prune", "origin", check=False)
    if fetched.returncode:
        raise NightlyError("could not reach GitHub to fetch: " + _first_line(fetched))
    main_ref = f"origin/{MAIN}"
    if git(clone, "rev-parse", "--verify", "-q", main_ref, check=False).returncode:
        raise NightlyError("GitHub has no main branch")
    git(clone, "checkout", "-q", "-f", "--detach", main_ref)      # the job's own clone: GitHub's main, nothing else
    git(clone, "clean", "-fdq")
    picks = [decide(dev, clone, p, main_ref) for p in candidates(dev)]
    res.files = [{"path": p.path, "action": p.action, "why": p.why} for p in picks]
    for p in picks:
        log(f"  {p.action:8} {p.path} ({p.why})")
    res.older = older_branches(clone, main_ref)
    writes = [p for p in picks if p.action in ("copy", "join")]
    if dry or not writes:
        log("dry run: nothing changed" if dry else "nothing new for GitHub tonight")
        return res
    branch = free_branch(clone, today)
    git(clone, "checkout", "-q", "-B", branch)
    for p in writes:
        target = clone / p.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.content)
    git(clone, "add", "--", *[p.path for p in writes])
    staged = set(git(clone, "diff", "--cached", "--name-only", "-z").stdout.decode().split("\0")) - {""}
    stray = staged - {p.path for p in writes}
    if stray:
        raise NightlyError("the commit would hold files it should not: " + ", ".join(sorted(stray)))
    for path in staged:
        why = forbidden(path)
        if why:
            raise NightlyError(f"refused to commit {path}: {why}")
    git(clone, "commit", "-q", "-m", commit_message(today, writes))
    res.branch = branch
    res.commit = git(clone, "rev-parse", "HEAD").stdout.decode().strip()
    pushed = git(clone, "push", "-q", "-u", "origin", branch, check=False)
    if pushed.returncode:
        raise NightlyError("GitHub did not take the branch " + branch + ": " + _first_line(pushed))
    log(f"pushed {branch} ({len(writes)} files, commit {res.commit[:7]})")
    res.older = [b for b in older_branches(clone, main_ref) if b != branch]
    res.pr = open_pull_request(clone, branch, today, writes, gh=gh, log=log)
    if not res.pr:
        res.link = compare_link(git(clone, "remote", "get-url", "origin").stdout.decode().strip(), branch)
    return res


def prepare_clone(dev: Path, clone: Path, *, log=print) -> None:
    """The job's own clone: made the first time (and marked); a folder the job did not make is never touched."""
    if clone == dev:
        raise NightlyError("the job's clone must not be the bot's own clone")
    if not (clone / ".git").exists():
        if clone.exists() and any(clone.iterdir()):
            raise NightlyError(f"{clone} exists but is not a git clone - the job will not touch it")
        url = git(dev, "remote", "get-url", "origin").stdout.decode().strip()
        log(f"making the job's own clone: {clone}")
        made = git(dev, "clone", "-q", url, str(clone), check=False)
        if made.returncode:
            raise NightlyError(f"could not make the clone {clone}: " + _first_line(made))
        git(clone, "config", MARKER, "true")
        return
    if git(clone, "config", "--get", MARKER, check=False).stdout.decode().strip() != "true":
        raise NightlyError(f"{clone} is a clone the job did not make (no '{MARKER}' mark) - it will not reset it. "
                           f"Mark it with: git -C \"{clone}\" config {MARKER} true")


def free_branch(clone: Path, today: dt.date) -> str:
    """bot/<date>, or bot/<date>-2, -3 ... when that name is on GitHub already (a second run the same day)."""
    taken = set(git(clone, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/").stdout
                .decode().split())
    base = f"{config.BOT_BRANCH_PREFIX}{today.isoformat()}"
    n = 1
    while True:
        name = base if n == 1 else f"{base}-{n}"
        if f"origin/{name}" not in taken:
            return name
        n += 1


def older_branches(clone: Path, main_ref: str) -> list[str]:
    """bot/ branches on GitHub whose files are not all on main yet (a pull request he has not merged)."""
    out = []
    for ref in git(clone, "for-each-ref", "--format=%(refname:short)",
                   f"refs/remotes/origin/{config.BOT_BRANCH_PREFIX}").stdout.decode().split():
        if git(clone, "merge-base", "--is-ancestor", ref, main_ref, check=False).returncode == 0:
            continue
        changed = [p for p in git(clone, "diff", "--name-only", "-z", f"{main_ref}...{ref}").stdout.decode()
                   .split("\0") if p]
        if changed and git(clone, "diff", "--quiet", main_ref, ref, "--", *changed, check=False).returncode:
            out.append(ref.removeprefix("origin/"))
    return out


def commit_message(today: dt.date, writes: list[Pick]) -> str:
    lines = [f"bot: his files of {today.day} {today.strftime('%b %Y')} ({len(writes)} files, L9c nightly)", ""]
    lines += [f"- {p.path} ({p.why})" for p in writes]
    return "\n".join(lines) + "\n"


def open_pull_request(clone: Path, branch: str, today: dt.date, writes: list[Pick], *, gh: str | list | None,
                      log=print) -> str | None:
    """`gh pr create` when gh is installed and logged in; None otherwise (the message then carries the link)."""
    exe = shutil.which("gh") if gh == "auto" else gh
    if not exe:
        return None
    exe = [exe] if isinstance(exe, str) else list(exe)            # a list: the tests' fake gh
    body = ("The bot's files from Tejas's PC (L9c nightly job). Only these paths; never data/ or .env.\n\n"
            + "\n".join(f"- `{p.path}` ({p.why})" for p in writes))
    done = _run([*exe, "pr", "create", "--base", MAIN, "--head", branch,
                 "--title", f"bot: his files of {today.isoformat()}", "--body", body], cwd=clone, check=False)
    if done.returncode:
        log("gh could not open the pull request (" + _first_line(done) + "); sending the link instead")
        return None
    urls = re.findall(r"https://\S+", done.stdout.decode("utf-8", "replace"))
    return urls[-1] if urls else None


def compare_link(origin_url: str, branch: str) -> str | None:
    """The GitHub page that opens a pull request for the branch (None when origin is not on GitHub)."""
    m = re.match(r"(?:https://github\.com/|git@github\.com:)([^/]+)/(.+?)(?:\.git)?/?$", origin_url)
    return f"https://github.com/{m.group(1)}/{m.group(2)}/compare/{MAIN}...{branch}?expand=1" if m else None


# ================================================================================================= telling him

def kinds(paths: list[str]) -> str:
    """'2 lessons, 1 picture, 1 answer file' - what the files are, in his words."""
    n = {"lesson": 0, "picture": 0, "answer file": 0, "other file": 0}
    for p in paths:
        if p.startswith("strategy/learned-inbox/"):
            n["lesson"] += 1
        elif p.startswith("strategy/pictures/"):
            n["picture"] += 1
        elif p.endswith("/answers.md"):
            n["answer file"] += 1
        else:
            n["other file"] += 1
    return ", ".join(f"{v} {k}{'' if v == 1 else 's'}" for k, v in n.items() if v)


def message(res: Result) -> str | None:
    """The Telegram message for this run, or None (a quiet night)."""
    if res.error:
        return ("The nightly copy of your bot's files to GitHub did NOT work tonight.\n"
                f"Why: {res.error}\n"
                "Nothing is lost: every file is still on your PC. It tries again tomorrow night.\n"
                "A Claude session can look at data/telegram/nightly.log.")
    lines = []
    copied = [f["path"] for f in res.of("copy", "join")]
    if res.branch:
        if res.older:            # FIRST, before any link (4 Oct 2026: the older one was merged first, this one clashed)
            lines.append(f"{len(res.older) + 1} links wait. This one also has everything from "
                         f"{', '.join(res.older)}. Merge only this one; close the older one - merging it first makes "
                         "this one clash.")
        lines.append(f"Your bot's files from today are on GitHub: {kinds(copied)}.")
        if res.pr:
            lines += ["Open this pull request and press \"Merge pull request\":", res.pr]
        elif res.link:
            lines += ["Open this link, press \"Create pull request\", then \"Merge pull request\":", res.link]
        else:
            lines.append(f"They are on the branch {res.branch}.")
        lines.append("GitHub's main does not change until you merge.")
    bad = res.of("conflict")
    if bad:
        lines.append("Not copied, because it changed both on your PC and on GitHub (a Claude session can join it): "
                     + ", ".join(f["path"] for f in bad))
    refused = res.of("refused")
    if refused:
        lines.append("Not copied, because it must never go to GitHub: "
                     + "; ".join(f"{f['path']} ({f['why']})" for f in refused))
    return "\n".join(lines) or None


def telegram_tell():
    """A function that sends Tejas one message with the real bot key (sending does not poll: no clash with the bot)."""
    from machine.telegram.api import TelegramAPI
    key, chat = config.env("TELEGRAM_BOT_TOKEN"), config.env("TELEGRAM_CHAT_ID")
    if not key or not chat:
        return None
    api = TelegramAPI(key)
    return lambda text: api.send_message(chat, text[:4096])


# ================================================================================================= records

def record(res: Result, folder: Path = config.TELEGRAM_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "nightly.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")


def last(folder: Path = config.TELEGRAM_DIR) -> dict | None:
    path = Path(folder) / "nightly.jsonl"
    if not path.is_file():
        return None
    rows = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return json.loads(rows[-1]) if rows else None


def status_line(folder: Path = config.TELEGRAM_DIR) -> str:
    """One line for the bot's STATUS."""
    r = last(folder)
    if not r:
        return "Nightly copy to GitHub: has not run yet."
    when = r["at"][:16].replace("T", " ")
    if r.get("error"):
        return f"Nightly copy to GitHub: FAILED on {when} - {r['error'][:120]}"
    if r.get("branch"):
        n = sum(1 for f in r["files"] if f["action"] in ("copy", "join"))
        return f"Nightly copy to GitHub: {when}, {n} file{'s' if n != 1 else ''} on {r['branch']} (merge it on GitHub)."
    return f"Nightly copy to GitHub: {when}, nothing new."


# ================================================================================================= Task Scheduler

def install(at: str = config.NIGHTLY_AT, *, root: Path = config.ROOT) -> str:
    """Adds (or replaces) the nightly task: pythonw (no window) -m machine.telegram.nightly, in this folder, every day
    at `at`; a night missed while the PC was off runs as soon as it is on again."""
    if os.name != "nt":
        raise NightlyError("Task Scheduler is on Windows only")
    exe = Path(root) / ".venv" / "Scripts" / "pythonw.exe"
    if not exe.is_file():
        raise NightlyError(f"{exe} is missing (make the .venv first: docs/SETUP.md)")
    if not re.fullmatch(r"\d{1,2}:\d{2}", at):
        raise NightlyError(f"--at wants a time like 23:30, not {at!r}")
    ps = ("$a = New-ScheduledTaskAction -Execute {exe} -Argument '-m machine.telegram.nightly' -WorkingDirectory {root};"
          "$t = New-ScheduledTaskTrigger -Daily -At {at};"
          "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
          "-ExecutionTimeLimit (New-TimeSpan -Hours 1);"
          "Register-ScheduledTask -TaskName {name} -Action $a -Trigger $t -Settings $s -Force "
          "-Description {desc} | Out-Null;"
          "(Get-ScheduledTask -TaskName {name} | Get-ScheduledTaskInfo).NextRunTime.ToString('yyyy-MM-dd HH:mm')"
          ).format(exe=_ps(str(exe)), root=_ps(str(root)), at=_ps(at), name=_ps(config.NIGHTLY_TASK_NAME),
                   desc=_ps("Copies the Telegram bot's files to GitHub every night (plan/likhith.md L9c)."))
    done = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=False)
    if done.returncode:
        raise NightlyError("Task Scheduler said no: " + _first_line(done))
    return done.stdout.decode("utf-8", "replace").strip()


def uninstall() -> None:
    _run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
          f"Unregister-ScheduledTask -TaskName {_ps(config.NIGHTLY_TASK_NAME)} -Confirm:$false"], check=False)


def _ps(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


# ================================================================================================= git plumbing

def git(repo: Path, *args: str, input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["git", "-C", str(repo), *args], input=input, check=check)


def _run(cmd: list[str], *, cwd: Path | None = None, input: bytes | None = None,
         check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}   # a lost login fails, never waits
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)                              # no window flashing at night
    done = subprocess.run(cmd, cwd=cwd, input=input, capture_output=True, env=env, creationflags=flags)
    if check and done.returncode:
        raise NightlyError(f"`{' '.join(cmd[3:5] if cmd[0] == 'git' else cmd[:2])}` failed: {_first_line(done)}")
    return done


def blob_id(repo: Path, ref: str, path: str) -> str | None:
    done = git(repo, "rev-parse", "--verify", "-q", f"{ref}:{path}", check=False)
    if done.returncode:
        return None
    return done.stdout.decode().strip() or None


def cat(repo: Path, blob: str) -> bytes:
    return git(repo, "cat-file", "blob", blob).stdout


def _first_line(done: subprocess.CompletedProcess) -> str:
    text = (done.stderr or b"").decode("utf-8", "replace") + (done.stdout or b"").decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lower().startswith("warning:")]
    return (lines[0] if lines else f"exit code {done.returncode}")[:300]


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ================================================================================================= main

def main(argv: list[str] | None = None, *, tell=None, dev: Path | None = None, state_dir: Path | None = None) -> int:
    """tell / dev / state_dir: for the tests (a fake Telegram, a throwaway repo, a throwaway data/telegram/)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m machine.telegram.nightly", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="show what would be copied; change and send nothing")
    ap.add_argument("--install", action="store_true", help="add the nightly task to Task Scheduler")
    ap.add_argument("--at", default=config.NIGHTLY_AT, help="the time for --install (default %(default)s)")
    ap.add_argument("--uninstall", action="store_true", help="remove the nightly task")
    ap.add_argument("--last", action="store_true", help="print the last run's result")
    ap.add_argument("--clone", type=Path, default=config.BOT_CLONE, help="the job's own clone")
    args = ap.parse_args(argv)
    if args.install:
        print(f"Task Scheduler: '{config.NIGHTLY_TASK_NAME}' every day at {args.at}; next run {install(args.at)}")
        return 0
    if args.uninstall:
        uninstall()
        print(f"removed '{config.NIGHTLY_TASK_NAME}'")
        return 0
    if args.last:
        print(json.dumps(last(state_dir or config.TELEGRAM_DIR), indent=2, ensure_ascii=False))
        return 0
    state_dir = Path(state_dir or config.TELEGRAM_DIR)
    log_path = state_dir / "nightly.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(text: str) -> None:
        line = f"{_now()} {text}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"nightly job started{' (dry run)' if args.dry_run else ''}")
    try:
        res = run(dev=dev or config.ROOT, clone=args.clone, dry=args.dry_run, log=log)
    except Exception as e:
        log("FAILED:\n" + traceback.format_exc())
        res = Result(at=_now(), error=str(e) if isinstance(e, NightlyError) else f"a bug in the job ({e!r:.200})",
                     dry=args.dry_run)
    if res.dry:
        return 1 if res.error else 0
    record(res, state_dir)
    text = message(res)
    if text:
        tell = tell or telegram_tell()
        if tell is None:
            log("could not message Tejas: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env")
            return 2
        try:
            tell(text)
            log("messaged Tejas")
        except Exception as e:
            log(f"could not message Tejas ({e})")
            return 2
    return 1 if res.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
