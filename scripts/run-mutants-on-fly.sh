#!/usr/bin/env bash
#
# Run cargo-mutants on a one-off Fly Machine using the runner image,
# and pull mutants.out/ back to the local working tree.
#
# Usage:
#   scripts/run-mutants-on-fly.sh [file] [filter] [vm-size]
#
# Defaults (override positionally):
#   file    = src/keys/verifying.rs
#   filter  = 'not test(/sign_kat_derand_/)'   (nextest filter expr)
#   vm-size = performance-8x
#
# Examples:
#   scripts/run-mutants-on-fly.sh
#   scripts/run-mutants-on-fly.sh src/keys/signing.rs
#   scripts/run-mutants-on-fly.sh src/keys/verifying.rs \
#       'not test(/sign_kat_derand_|nonce_independence|sign_derand_is_deterministic|sig_binds_to_message/)'
#   scripts/run-mutants-on-fly.sh src/keys/verifying.rs 'all()' performance-4x
#
# What it does:
#   1. Bundles the current working tree (committed + staged + dirty,
#      via `git stash create`) into a tarball.
#   2. Spins up a one-off Fly Machine on `sqisign-infra-runners` with
#      the runner image, overriding the GH Actions entrypoint so the
#      Machine just sleeps and accepts ssh.
#   3. SFTPs the source tarball up; ssh'es in to extract, install
#      cargo-mutants, and run it; tar's `mutants.out/` to /tmp.
#   4. SFTPs the result tarball back; extracts to `./mutants.out-fly/`.
#   5. Always destroys the Machine on exit (success, failure, or Ctrl-C).
#
# Notes:
#   * Does NOT use a GitHub token — we upload local source directly,
#     so private-repo auth on the Machine is unnecessary.
#   * `cargo install` of cargo-mutants takes ~3–4 min cold. The Rust
#     toolchain and cargo-nextest are pre-baked in the runner image.
#   * `--lib` is intentionally NOT passed: cargo-mutants runs the
#     full `cargo test` (lib + integration + doc) per mutant. Slow
#     integration tests should be excluded via the filter input.

set -euo pipefail

FILE=${1:-src/keys/verifying.rs}
FILTER=${2:-'not test(/sign_kat_derand_/)'}
SIZE=${3:-performance-8x}

APP=sqisign-infra-runners
IMG=registry.fly.io/sqisign-infra-runners:latest
LOCAL_OUT=mutants.out-fly
TMP_SRC=$(mktemp /tmp/selkie-src.XXXXXX.tar.gz)
TMP_OUT=$(mktemp /tmp/mutants-out.XXXXXX.tar.gz)

cleanup() {
  rm -f "$TMP_SRC" "$TMP_OUT"
  if [ -n "${MACHINE_ID:-}" ]; then
    echo "==> destroying Machine $MACHINE_ID" >&2
    flyctl machine destroy "$MACHINE_ID" --app "$APP" -f >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# --- pack current working tree (incl. uncommitted/staged) ---
echo "==> packing local source tree"
STASH_SHA=$(git stash create 2>/dev/null || true)
git archive --format=tar.gz \
  -o "$TMP_SRC" \
  "${STASH_SHA:-HEAD}"
SRC_BYTES=$(wc -c <"$TMP_SRC" | tr -d ' ')
echo "    $TMP_SRC ($SRC_BYTES bytes)"

# --- spin up the Machine ---
# `flyctl machine run` has no `--json`, so we parse the 14-hex-char
# Machine ID out of its stdout (the line containing "Machine ID").
echo "==> starting Fly Machine ($SIZE)"
RUN_LOG=$(mktemp /tmp/fly-run.XXXXXX.log)
flyctl machine run "$IMG" \
  --app "$APP" \
  --vm-size "$SIZE" \
  --entrypoint /bin/sleep \
  --autostart=true \
  -- 14400 2>&1 | tee "$RUN_LOG"
MACHINE_ID=$(grep -oE '[0-9a-f]{14}' "$RUN_LOG" | head -1)
rm -f "$RUN_LOG"
if [ -z "$MACHINE_ID" ]; then
  echo "error: could not parse machine id from flyctl output" >&2
  exit 1
fi
echo "    machine: $MACHINE_ID"

# --- wait for ssh ---
echo "==> waiting for ssh"
for _ in $(seq 1 60); do
  if flyctl ssh console -a "$APP" -s "$MACHINE_ID" -C true >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# --- upload source ---
echo "==> uploading source"
flyctl ssh sftp shell -a "$APP" -s "$MACHINE_ID" <<EOF
put $TMP_SRC /tmp/selkie-src.tar.gz
EOF

# --- run mutants ---
echo "==> running cargo-mutants (file=$FILE)"
flyctl ssh console -a "$APP" -s "$MACHINE_ID" -C "bash -lc '
  set -euo pipefail
  rm -rf /tmp/work
  mkdir -p /tmp/work
  tar xzf /tmp/selkie-src.tar.gz -C /tmp/work
  cd /tmp/work
  # cargo-mutants is not in the runner image; cargo-nextest is.
  command -v cargo-mutants >/dev/null || cargo install --locked cargo-mutants
  cargo mutants \
    --file \"$FILE\" \
    --test-tool nextest \
    --in-place \
    --no-shuffle \
    -vV \
    -- -E \"$FILTER\" || true
  tar czf /tmp/mutants-out.tar.gz mutants.out
'"

# --- pull artifact back ---
echo "==> pulling mutants.out back"
flyctl ssh sftp shell -a "$APP" -s "$MACHINE_ID" <<EOF
get /tmp/mutants-out.tar.gz $TMP_OUT
EOF
rm -rf "$LOCAL_OUT"
mkdir -p "$LOCAL_OUT"
tar xzf "$TMP_OUT" -C "$LOCAL_OUT" --strip-components=1

echo "==> done. results in ./$LOCAL_OUT"
echo
echo "summary:"
jq -r '.outcomes | group_by(.summary) | map({summary: .[0].summary, count: length}) | .[] | "  \(.summary): \(.count)"' \
  "$LOCAL_OUT/outcomes.json" 2>/dev/null || {
  echo "  (could not parse outcomes.json; listing files instead)"
  ls -la "$LOCAL_OUT"
}
