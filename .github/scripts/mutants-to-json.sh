#!/usr/bin/env bash
# Convert cargo-mutants outcomes.json to the CI site JSON format.
#
# Usage: mutants-to-json.sh <outcomes.json> <sha>
#
# Reads one or more outcomes.json files (for sharded runs, cat them
# together first or pass the merged file).
set -euo pipefail

OUTCOMES="${1:?usage: mutants-to-json.sh <outcomes.json> <sha>}"
SHA="${2:?usage: mutants-to-json.sh <outcomes.json> <sha>}"

jq --arg sha "$SHA" '{
  sha: $sha,
  updated_at: (now | todate),
  summary: {
    caught:  [.outcomes[] | select(.summary == "CaughtMutant")] | length,
    missed:  [.outcomes[] | select(.summary == "MissedMutant")] | length,
    timeout: [.outcomes[] | select(.summary == "Timeout")]      | length,
    unviable:[.outcomes[] | select(.summary == "Unviable")]     | length,
    total:   ([.outcomes[] | select(.summary != "Success")] | length)
  },
  survivors: [
    .outcomes[]
    | select(.summary == "MissedMutant")
    | {
        name:     .scenario.Mutant.name,
        file:     .scenario.Mutant.file,
        function: .scenario.Mutant.function.function_name,
        line:     .scenario.Mutant.function.span.start.line
      }
  ] | sort_by(.file, .line)
}' "$OUTCOMES"
