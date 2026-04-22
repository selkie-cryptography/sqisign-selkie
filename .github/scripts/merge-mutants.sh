#!/usr/bin/env bash
# Merge incremental mutants results into an existing baseline.
#
# Usage: merge-mutants.sh <incremental.json> <sha> [baseline-url]
#
# Fetches the existing baseline from the CI site, replaces survivors
# for files touched in the incremental run, keeps the rest, and
# recomputes summary stats. Outputs merged JSON to stdout.
set -euo pipefail

INC="${1:?usage: merge-mutants.sh <incremental.json> <sha> [baseline-url]}"
SHA="${2:?usage: merge-mutants.sh <incremental.json> <sha> [baseline-url]}"
BASELINE_URL="${3:-https://sqisign-selkie-ci.fly.dev/mutants/latest.json}"

# Fetch existing baseline (empty object if none exists yet).
curl -sf "$BASELINE_URL" -o /tmp/mutants-baseline.json 2>/dev/null \
  || echo '{}' > /tmp/mutants-baseline.json

# Merge: incremental results win for files in the diff,
# baseline survivors for untouched files are preserved.
jq -s --arg sha "$SHA" '
  (.[0] // {}) as $base |
  (.[1] // {}) as $inc |
  # Files touched in the incremental run.
  ($inc.survivors // [] | map(.file) | unique) as $touched |
  # Keep baseline survivors for untouched files.
  ($base.survivors // [] | map(select(.file as $f | $touched | index($f) | not))) as $kept |
  # Merge survivors.
  ($kept + ($inc.survivors // [])) as $all |
  # Recompute summary from merged data.
  ($base.summary // {caught:0,missed:0,timeout:0,unviable:0,total:0}) as $bs |
  ($inc.summary // {caught:0,missed:0,timeout:0,unviable:0,total:0}) as $is |
  {
    sha: $sha,
    updated_at: $inc.updated_at,
    summary: {
      caught:   ($bs.caught + $is.caught),
      missed:   ([$all | length, $bs.missed + $is.missed] | max),
      timeout:  ($bs.timeout + $is.timeout),
      unviable: ($bs.unviable + $is.unviable),
      total:    ($bs.total + $is.total)
    },
    survivors: ($all | sort_by(.file, .line))
  }
' /tmp/mutants-baseline.json "$INC"
