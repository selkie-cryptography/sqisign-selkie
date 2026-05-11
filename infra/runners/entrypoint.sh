#!/usr/bin/env bash
# Ephemeral runner entrypoint. Expects $JITCONFIG to be a base64-
# encoded just-in-time runner registration blob, minted by the
# orchestrator via the GitHub App and passed in via Fly Machine env.
#
# Runner registers itself, picks up exactly one job, and exits. The
# Fly Machine is created with auto_destroy=true so exit triggers
# destruction.
set -euo pipefail

if [[ -z "${JITCONFIG:-}" ]]; then
  echo "ERROR: JITCONFIG env var is empty — orchestrator must inject it." >&2
  exit 64
fi

cd /home/runner/actions-runner

# `--jitconfig` consumes the runner registration in one shot. No
# state persists to disk; the registration is single-use.
exec ./run.sh --jitconfig "$JITCONFIG"
