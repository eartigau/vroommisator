#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

echo "[1/5] Building docs assets"
python make_webpage_assets.py

echo "[2/5] Creating docs/.nojekyll"
touch docs/.nojekyll

echo "[3/5] Checking git branch"
CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Current branch: $CUR_BRANCH"

echo "[4/5] Pushing docs subtree to gh-pages"
# This publishes the docs/ folder root to branch gh-pages.
git subtree push --prefix docs origin gh-pages

echo "[5/5] Done"
echo "If Pages is configured for branch gh-pages (root), your site will update shortly."
