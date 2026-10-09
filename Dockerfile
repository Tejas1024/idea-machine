# The idea machine in Docker (L16.2): everything that does not need MT5 - the tests, the read-only dashboard, the demo.
# Run it with `docker compose up` (compose.yaml); docs/DOCKER.md says more.
#
# The image holds the code as git keeps it, Python 3.13 and requirements.txt - never data/, .env or the price files
# (.dockerignore is an allow-list). Like GitHub's tests (.github/workflows/tests.yml): Linux, UTC, no price data, so
# the price-data and MT5 tests skip themselves.

# Python 3.13.16 on Debian 13 (trixie), pinned by its digest (pulled 6 Oct 2026; the same digest serves Intel and Apple
# chips). For a newer Python: docker pull python:3.13-slim, put its digest here, run the tests.
FROM python:3.13-slim@sha256:2b6e177adb67a564bba97e6f0eab8760a7dc3645582d70277a055a4327a737e5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# git: a trial file records the commit, and some tests run git (as on GitHub). libgomp1: LightGBM needs it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 machine && chown machine:machine /app
USER machine
COPY --chown=machine:machine . .

# The files exactly as git keeps them: a Windows PC's copies have Windows line endings (core.autocrlf), GitHub's have
# Unix ones. git reads every file the way it does for a commit and writes it back, and the image gets ONE commit, like
# GitHub's shallow checkout (the real commit id is not in the image). Its author is a no-reply address: the public
# copy's check (machine/tools/publish.py) counts every other address of the history as personal, and this file is in
# the public copy - "docker@localhost" here failed tests/test_publish.py inside the image (6 Oct 2026).
RUN git init -q \
 && git config core.autocrlf input \
 && git add -A \
 && git -c user.name=docker -c user.email=noreply@localhost commit -qm "the files as git keeps them" \
 && git rm -rq --cached . \
 && git reset -q --hard

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
