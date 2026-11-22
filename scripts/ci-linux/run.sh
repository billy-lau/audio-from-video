#!/usr/bin/env bash
# Run the CI Linux checks locally in a throwaway x86_64 container, before pushing.
#
# Usage: scripts/ci-linux/run.sh [python-version] [pytest arguments...]
#   scripts/ci-linux/run.sh 3.11            # core tests, like CI's "ubuntu / 3.11" job
#   scripts/ci-linux/run.sh 3.12 -v         # GUI tests too, like CI's "GUI / ubuntu" job
#   scripts/ci-linux/run.sh 3.12 tests/gui/test_screens.py -k summary
#
# Needs a Docker-compatible runtime (for example, Colima: brew install colima docker;
# colima start --vm-type vz --vz-rosetta). The working tree is mounted read-only and
# copied inside the container, so uncommitted changes are tested and nothing is written
# back. The image is rebuilt automatically when pyproject.toml changes.
set -euo pipefail

PY="${1:-3.12}"
shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEPS_HASH="$(cat "$REPO/pyproject.toml" "$HERE/Dockerfile" | shasum -a 256 | cut -c1-12)"
IMAGE="afv-ci-linux:${PY}-${DEPS_HASH}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE (first run for this Python and dependency set)..."
  docker build --platform linux/amd64 --build-arg "PY=${PY}" -t "$IMAGE" \
    -f "$HERE/Dockerfile" "$REPO"
fi

docker run --rm --platform linux/amd64 -v "$REPO:/repo:ro" "$IMAGE" bash -c '
  set -e
  tar -C /repo --exclude=./.venv --exclude=./build --exclude=./.git -cf - . | tar -C /work -xf -
  cd /work
  pip install -q --no-deps -e .
  echo "== $(python --version), $(uname -m)"
  ruff check .
  ruff format --check src tests scripts packaging
  python -m pytest -p no:cacheprovider --capture=sys "$@"
' _ "$@"
