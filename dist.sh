#!/usr/bin/env bash
# Release eyck: ./dist.sh <version>, from a clean main that is already pushed.
# The version comes from the git tag (setuptools_scm).
set -euo pipefail
version="${1:?usage: ./dist.sh <version>}"
test -z "$(git status --porcelain --untracked-files=no)" || { echo "commit your changes first" >&2; exit 1; }

git tag -a "v${version}" -m "v${version}"
rm -rf dist
uv build --out-dir dist
git push origin main "v${version}"
gh release create "v${version}" -t "v${version}" -n "v${version}" dist/eyck-"${version}"*

set -a; . ~/.secrets/pypi.env; set +a
uv publish dist/eyck-"${version}"*
