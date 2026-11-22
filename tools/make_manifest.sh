#!/usr/bin/env bash
# Regenerate MANIFEST.sha256 for the release tree.
#
#   ./tools/make_manifest.sh
#
# Paths are relative to the repository root. The manifest itself and anything
# under .git are excluded. Run this after any change that should be part of a
# release, then commit the new manifest together with the change.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

find . \
    -path ./.git -prune -o \
    -name MANIFEST.sha256 -prune -o \
    -name '__pycache__' -prune -o \
    -name '*.pyc' -prune -o \
    -type f -print \
  | sed 's|^\./||' \
  | LC_ALL=C sort \
  | xargs -r sha256sum > MANIFEST.sha256

echo "wrote MANIFEST.sha256 with $(wc -l < MANIFEST.sha256) entries"
