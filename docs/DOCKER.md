# Docker: the tests and the dashboard in one command (L16.2)

Everything of the idea machine that does not need MT5 runs in Docker: the tests (the same as GitHub's), a read-only
dashboard, and the demo.

```
docker compose up
```

| service | what it does |
|---|---|
| `tests` | the whole test suite in Linux: Python 3.13, the pinned packages of `requirements.txt`, UTC - like GitHub's run, so the price-data and MT5 tests skip themselves. **No network**: a test can never send a message, call Claude or reach GitHub. |
| `dashboard` | **http://127.0.0.1:8050** (this computer only): the ideas and their states, the versions tried, the last runs, the heartbeat, alerts and errors. Read only; it reloads every minute. |
| `demo` | only when asked (a profile): one made-up idea through every step, made-up prices, no Claude call (about 2 minutes). |

## Commands

| command | |
|---|---|
| `docker compose up` | build the image if needed, then the tests and the dashboard side by side (Ctrl+C stops both) |
| `docker compose run --rm tests` | only the tests; the exit code is pytest's |
| `docker compose up -d dashboard` | only the dashboard, in the background; `docker compose down` stops it |
| `docker compose --profile demo run --rm demo` | the demo |
| `docker compose build` | build again (needed after a change to `requirements.txt`; code changes only redo the last layers) |
| `MACHINE_HOME="C:/ML Bot" docker compose up` | from another folder (a worktree): the dashboard reads the files of the folder named |
| `python -m machine.dashboard` | the same dashboard without Docker; `--check` prints its six boxes once |

## What is not in Docker, on purpose

- **The Telegram bot, the live alerts, the daily run and the heartbeat.** They need MT5 or Windows, and there is ONE of
  each, on the host PC (Task Scheduler). A second bot would read the chat twice.
- **The data and the keys.** `.dockerignore` is an allow-list: the image gets only the files git keeps - never `data/`
  (except four small public tables), the price files, `.env` or the MT5 login. The image then re-reads its files the
  way git keeps them (Unix line endings, one commit), so the tests see exactly the files of a fresh checkout.

## How the dashboard never shows a 2024+ price

1. **It opens only a fixed list of files** (`READS` in `machine/dashboard/read.py`). A test opens the page with
   Python's audit hook on and fails on any other file. Never: the live alerts' folder, the sealed prices, the bot's
   outbox and sent messages, the owner's messages, `.env`.
2. **In Docker it does not even get them:** `compose.yaml` mounts only `plan/work`, `data/host`, `data/intake` and
   `data/telegram`, all read only.
3. **Free text is masked.** A text the live alerts can write loses every digit except clock times; any other text
   loses every number shaped like a gold price (4 digits or more, with or without a comma or decimals). Rupees, years,
   counts and sizes stay. A test plants "live prices" in every file the page reads and finds none of them on the page.
4. **Read only, in every layer:** GET and HEAD only (anything else: 405), every mount `:ro`, the container's own files
   read only, no Linux capabilities, the port open to this computer only.

## Memory

Docker Desktop on Windows runs a Linux machine (WSL 2). Without a limit it may take half the PC's memory (8 GB of 16),
and Kubernetes or other projects in the same Docker fill it at once. Measured on a 16 GB PC that also runs MT5 and the
alerts (6 Oct 2026): free memory fell to 0.6 GB and Windows paged MT5 and the bot out. So on such a PC:

1. **Give it a limit** - a file `%USERPROFILE%\.wslconfig`:

   ```
   [wsl2]
   memory=5GB
   ```

   then `wsl --shutdown`; it applies at the next start. With it, the whole test run kept at least 2.8 GB free.
2. **Stop it after use** (it keeps its memory until it stops):

   ```
   docker desktop stop
   wsl --shutdown
   ```

   (`wsl --shutdown` also stops any other Linux of WSL that is running.)
