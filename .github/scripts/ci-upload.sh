#!/usr/bin/env bash
# Upload CI data (coverage or benchmarks) to the Fly.io CI site.
#
# Usage: ci-upload.sh <coverage|bench> <json-file> <sha>
#
# - Stores per-commit data at /data/<kind>/<sha>.json
# - Updates /data/<kind>/latest.json
# - Maintains /data/<kind>/index.json (last 50 summaries)
# - Prunes per-commit files beyond 30 entries
# - Writes status.json before/after for the site's indicator
set -euo pipefail

KIND="${1:?usage: ci-upload.sh <coverage|bench> <json-file> <sha>}"
JSON="${2:?usage: ci-upload.sh <coverage|bench> <json-file> <sha>}"
SHA="${3:?usage: ci-upload.sh <coverage|bench> <json-file> <sha>}"

APP="sqisign-selkie-ci"
SITE="https://sqisign-selkie-ci.fly.dev"
MAX_INDEX=50
MAX_FILES=30
DIR="/data/${KIND}"

# Build index entry fields. Coverage includes percent; benchmarks don't.
UPDATED=$(jq -r '.updated_at' "$JSON")
if [ "$KIND" = "coverage" ]; then
  TOTAL_PCT=$(jq -r '.total.percent' "$JSON")
  INDEX_JQ='map(select(.sha != $sha))
    | [{sha: $sha, percent: ($pct | tonumber), updated_at: $at}] + .
    | .[0:$max]'
  JQ_ARGS=(--arg sha "$SHA" --arg pct "$TOTAL_PCT" --arg at "$UPDATED" --argjson max "$MAX_INDEX")
else
  INDEX_JQ='map(select(.sha != $sha))
    | [{sha: $sha, updated_at: $at}] + .
    | .[0:$max]'
  JQ_ARGS=(--arg sha "$SHA" --arg at "$UPDATED" --argjson max "$MAX_INDEX")
fi

# Wake the app.
flyctl apps restart "$APP" --skip-health-checks
sleep 5

# Signal "running".
echo "{\"state\":\"running\",\"sha\":\"${SHA}\"}" > "/tmp/${KIND}-status.json"
flyctl ssh sftp shell -a "$APP" <<SFTP
put /tmp/${KIND}-status.json ${DIR}/status.json
SFTP

# Upload per-commit and latest.
flyctl ssh console -a "$APP" -C "rm -f ${DIR}/latest.json ${DIR}/${SHA}.json"
flyctl ssh sftp shell -a "$APP" <<SFTP
put ${JSON} ${DIR}/${SHA}.json
put ${JSON} ${DIR}/latest.json
SFTP

# Update index.
curl -sf "${SITE}/${KIND}/index.json" -o "/tmp/${KIND}-index.json" 2>/dev/null \
  || echo '[]' > "/tmp/${KIND}-index.json"
jq "${JQ_ARGS[@]}" "$INDEX_JQ" "/tmp/${KIND}-index.json" > "/tmp/${KIND}-index-new.json"
flyctl ssh console -a "$APP" -C "rm -f ${DIR}/index.json"
flyctl ssh sftp shell -a "$APP" <<SFTP
put /tmp/${KIND}-index-new.json ${DIR}/index.json
SFTP

# Prune old per-commit files.
KEEP_SHAS=$(jq -r ".[0:${MAX_FILES}] | .[].sha" "/tmp/${KIND}-index-new.json")
flyctl ssh console -a "$APP" -C "ls ${DIR}/" \
  | tr -d '\r' | while IFS= read -r f; do
    case "$f" in
      latest.json|index.json|status.json) continue ;;
      *.json) ;;
      *) continue ;;
    esac
    file_sha="${f%.json}"
    if ! echo "$KEEP_SHAS" | grep -qxF "$file_sha"; then
      flyctl ssh console -a "$APP" -C "rm -f ${DIR}/$f" || true
    fi
  done

# Signal "done".
echo "{\"state\":\"done\",\"sha\":\"${SHA}\"}" > "/tmp/${KIND}-status.json"
flyctl ssh console -a "$APP" -C "rm -f ${DIR}/status.json"
flyctl ssh sftp shell -a "$APP" <<SFTP
put /tmp/${KIND}-status.json ${DIR}/status.json
SFTP
