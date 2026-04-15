#!/usr/bin/env bash
# Parse divan benchmark output into github-action-benchmark's
# customSmallerIsBetter JSON format.
#
# Input: divan's table output on stdin (with --color never).
# Output: JSON array on stdout.
#
# Divan output has group headers like:
#   bigint   fastest  │ slowest  │ median  │ mean  │ samples │ iters
# followed by result lines like:
#   ├─ mul   2.6 ns   │ 2.7 ns   │ 2.6 ns  │ 2.6 ns│ 100     │ 204800
#
# We extract group::name and the "mean" column (4th timing value).

set -euo pipefail

echo "["
first=true
group=""

while IFS= read -r line; do
    # Detect group headers: a word followed by "fastest".
    if echo "$line" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_]* +fastest'; then
        group=$(echo "$line" | awk '{print $1}')
        continue
    fi

    # Match result lines (tree chars followed by name and timings).
    if echo "$line" | grep -qE '^[│├╰─ ]+[a-zA-Z_]'; then
        # Extract name: strip tree characters, take first word.
        name=$(echo "$line" | sed 's/[│├╰─ ]*//' | awk '{print $1}')
        [ -z "$name" ] && continue

        # Prefix with group if we have one.
        if [ -n "$group" ]; then
            name="${group}::${name}"
        fi

        # Extract the 4th timing field (mean).
        mean_field=$(echo "$line" | awk -F '│' '{print $4}' | xargs)
        [ -z "$mean_field" ] && continue

        # Parse value and unit.
        value=$(echo "$mean_field" | awk '{print $1}')
        unit=$(echo "$mean_field" | awk '{print $2}')
        [ -z "$value" ] && continue

        # Normalize to nanoseconds.
        case "$unit" in
            ps) value=$(echo "$value * 0.001" | bc -l) ;;
            ns) ;;
            µs) value=$(echo "$value * 1000" | bc -l) ;;
            ms) value=$(echo "$value * 1000000" | bc -l) ;;
            s)  value=$(echo "$value * 1000000000" | bc -l) ;;
            *)  continue ;;
        esac

        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        printf '  {"name": "%s", "unit": "ns/iter", "value": %s}' "$name" "$value"
    fi
done

echo ""
echo "]"
