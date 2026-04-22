#!/usr/bin/env bash
# Build and run the SQIsign C reference to capture outer-chain
# kernel points for cross-checking against `src/deuring/mod.rs`'s
# `[OUTER_KER]` diagnostic.
#
# Assumes the C reference is checked out at
# `${CREF_ROOT:-$HOME/src/github.com/SQIsign/the-sqisign}` and
# pinned to the commit recorded in the fixture headers under
# `tests/fixtures/`.
#
# Usage:
#   scripts/cref_outer_ker.sh              # full run, writes to /tmp/cref-kat/
#   scripts/cref_outer_ker.sh --vector 0   # just KAT seed 0's first block
#
# Output:
#   /tmp/cref-kat/stderr.log                  — raw C-ref stderr (~60k lines)
#   /tmp/cref-kat/outer_ker_vector_<N>.txt    — N-th OUTER_KER block
#
# The diagnostic prints live in `src/id2iso/ref/lvlx/dim2id2iso.c`
# around `ker.T1` and are always-on (no `NDEBUG` guard), so any
# build produces them. Re-run this script after touching the C ref
# so the captured kernel stays in sync with the pinned commit.

set -euo pipefail

CREF_ROOT="${CREF_ROOT:-$HOME/src/github.com/SQIsign/the-sqisign}"
BUILD_DIR="${CREF_ROOT}/build"
KATGEN="${BUILD_DIR}/apps/PQCgenKAT_sign_lvl1"
WORK=/tmp/cref-kat

vector=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vector) vector="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${CREF_ROOT}" ]]; then
    echo "C reference not found at ${CREF_ROOT}. Set CREF_ROOT." >&2
    exit 1
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "configuring cmake build in ${BUILD_DIR}"
    cmake -S "${CREF_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release >/dev/null
fi

echo "building PQCgenKAT_sign_lvl1"
cmake --build "${BUILD_DIR}" --target PQCgenKAT_sign_lvl1 >/dev/null

mkdir -p "${WORK}"
rm -f "${WORK}/PQCsignKAT_353_SQIsign_lvl1."{req,rsp} "${WORK}/stderr.log"

echo "running KAT generator (100 signatures, ~30s)"
( cd "${WORK}" && "${KATGEN}" 2>"${WORK}/stderr.log" >/dev/null )

total=$(grep -c '^OUTER_KER T1.P1_x_re' "${WORK}/stderr.log" || true)
echo "captured ${total} outer-chain invocations in ${WORK}/stderr.log"

# Split into per-invocation blocks. Each block is exactly 7 lines:
#   T1.P1, T1.P2, T2.P1, T2.P2, E1_j, E2_j, exp
awk '
    /^OUTER_KER T1.P1_x_re/ { n++; out = sprintf("'"${WORK}"'/outer_ker_vector_%d.txt", n - 1); block = 7 }
    block > 0 && /^OUTER_KER/ { print > out; block-- }
' "${WORK}/stderr.log"

if [[ -n "${vector}" ]]; then
    cat "${WORK}/outer_ker_vector_${vector}.txt"
fi
