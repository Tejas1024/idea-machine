"""One run at a time (the prompt's "a lock file"): the same lock the bot and the intake use.

The lock is a byte locked in a file, not the file itself: when the program stops - even killed, even a power cut - the
system lets go of it, so a lock is never left behind.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def one_run(path: Path):
    """Yields True when this run holds the lock, False when another run holds it (then do nothing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    got = True
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        got = False
    try:
        yield got
    finally:
        if got and os.name == "nt":
            try:
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        fh.close()
