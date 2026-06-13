#!/usr/bin/env bash
# Builds the cross-test oracle against a prebuilt the-sqisign C reference.
#
# Requires the C reference already built (its CMake `build-minigmp-ref`
# tree, which links mini-gmp -- the `mpz_probab_prime_p` whose
# trial-division prime set the Rust pre-screen matches). Override the
# checkout with SQISIGN_C_REF; the build tree defaults to
# $SQISIGN_C_REF/build-minigmp-ref.
#
# Emits the oracle path on stdout (the only stdout line), so callers can
# `ORACLE=$(tools/cref_xtest/build.sh)`.
set -euo pipefail

ref="${SQISIGN_C_REF:-$HOME/src/github.com/SQISign/the-sqisign}"
build="${SQISIGN_C_REF_BUILD:-$ref/build-minigmp-ref}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/oracle"

if [[ ! -d "$ref" ]]; then
    echo "C reference not found at $ref (set SQISIGN_C_REF)" >&2
    exit 1
fi
if [[ ! -f "$build/src/libsqisign_lvl1_test_nistapi.a" ]]; then
    echo "C reference not built at $build (expected libsqisign_lvl1_test_nistapi.a)" >&2
    exit 1
fi

includes=(
    "$ref/include"
    "$ref/src/nistapi/lvl1"
    "$ref/src/signature/ref/include"
    "$ref/src/quaternion/ref/generic/include"
    "$ref/src/precomp/ref/lvl1/include"
    "$ref/src/mp/ref/generic/include"
    "$ref/src/ec/ref/include"
    "$ref/src/gf/ref/include"
    "$ref/src/common/generic/include"
    "$ref/src/mini-gmp"
)
inc_flags=()
for i in "${includes[@]}"; do inc_flags+=("-I$i"); done

# Link order mirrors build-minigmp-ref/apps/.../link.txt for
# PQCgenKAT_sign_lvl1 (the test_nistapi archive supplies randombytes /
# randombytes_init); ../libGMP.a is mini-gmp built static.
libs=(
    "$build/src/libsqisign_lvl1_test_nistapi.a"
    "$build/src/libsqisign_lvl1_test.a"
    "$build/src/signature/ref/lvl1/libsqisign_signature_lvl1.a"
    "$build/src/verification/ref/lvl1/libsqisign_verification_lvl1.a"
    "$build/src/id2iso/ref/lvl1/libsqisign_id2iso_lvl1.a"
    "$build/src/quaternion/ref/generic/libsqisign_quaternion_generic.a"
    "$build/src/hd/ref/lvl1/libsqisign_hd_lvl1.a"
    "$build/src/ec/ref/lvl1/libsqisign_ec_lvl1.a"
    "$build/src/gf/ref/lvl1/libsqisign_gf_lvl1.a"
    "$build/src/mp/ref/generic/libsqisign_mp_generic.a"
    "$build/src/precomp/ref/lvl1/libsqisign_precomp_lvl1.a"
    "$build/libGMP.a"
    "$build/src/common/generic/libsqisign_common_test.a"
)

# oracle.c includes only api.h / rng.h / sqisign_namespace.h, which need
# the build-type and variant macros but not the target-arch or mini-gmp
# macros (those guard the C ref's own sources, already compiled into the
# archives). Keeping this set minimal makes the build host-agnostic
# (Apple arm64 and Linux x86_64 alike).
defines=(-DENABLE_SIGN -DSQISIGN_VARIANT=lvl1 -DSQISIGN_BUILD_TYPE_REF)

cc -O2 -std=c11 -Wall \
    "${defines[@]}" \
    "${inc_flags[@]}" \
    "$here/oracle.c" -o "$out" \
    "${libs[@]}" -lm

echo "$out"
