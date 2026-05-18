#!/usr/bin/env bash
# paper/scripts/finalize_public_push.sh
#
# Substitutes URL placeholders in README.md + paper/paper_draft.md
# with the real hosted-repo URLs after `gh repo create --push` (or
# equivalent) has been run for each of the 3 repos.
#
# Usage:
#   ./paper/scripts/finalize_public_push.sh <main_url> <exo_url> <gem5_url>
#
# After running this script, the only remaining step is to push the
# README + paper_draft.md commit on the main repo.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  cat <<USAGE
Usage: $0 <main_url> <exo_url> <gem5_url>

Example:
  $0 \\
    https://github.com/USER/precision-routing-rvv-fa \\
    https://github.com/USER/exo-saturn-rvv \\
    https://github.com/USER/gem5-saturn-fu

This script must be run from the main repo root (the directory
that contains README.md + paper/paper_draft.md).
USAGE
  exit 1
fi

MAIN_URL="$1"
EXO_URL="$2"
GEM5_URL="$3"

# Sanity checks: we must be in the main repo root, and the placeholders
# must still be present (i.e. the script has not already run).
if [[ ! -f README.md || ! -f paper/paper_draft.md ]]; then
  echo "ERROR: must be run from the main repo root (README.md + paper/paper_draft.md not found)." >&2
  exit 1
fi

if ! grep -q '<TODO_MAIN_URL>' README.md; then
  echo "WARNING: README.md does not contain <TODO_MAIN_URL> — already finalized?" >&2
fi

# Substitute (use ~ as sed delimiter so URL slashes don't break parsing).
sed -i \
  -e "s~<TODO_MAIN_URL>~${MAIN_URL}~g" \
  -e "s~<TODO_EXO_URL>~${EXO_URL}~g" \
  -e "s~<TODO_GEM5_URL>~${GEM5_URL}~g" \
  README.md paper/paper_draft.md

# Verify the substitutions actually took effect.
if grep -q '<TODO_.*_URL>' README.md paper/paper_draft.md 2>/dev/null; then
  echo "ERROR: leftover <TODO_..._URL> placeholders after substitution:" >&2
  grep -n '<TODO_.*_URL>' README.md paper/paper_draft.md
  exit 1
fi

echo "Substituted:"
echo "  <TODO_MAIN_URL> -> $MAIN_URL"
echo "  <TODO_EXO_URL>  -> $EXO_URL"
echo "  <TODO_GEM5_URL> -> $GEM5_URL"
echo
echo "Diff of changes:"
git diff --stat README.md paper/paper_draft.md
echo
echo "Next: git add README.md paper/paper_draft.md && git commit -m 'Public push: hosted-repo URLs in README + paper §9' && git push"
