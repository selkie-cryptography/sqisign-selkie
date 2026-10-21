#!/usr/bin/env bash
# Builds the cross-test oracle against a prebuilt the-sqisign C reference.
#
# Requires the C reference already configured and built with CMake
# (`-DSQISIGN_BUILD_TYPE=ref`). Override the checkout with SQISIGN_C_REF,
# the build tree with SQISIGN_C_REF_BUILD (default $SQISIGN_C_REF/build-ref),
# and the parameter set with SQISIGN_VARIANT (default p324_3).
#
# The static archives, their link order, and the compile flags are read
# from the CMake link and flag files of the reference's own KAT generator,
# so the oracle links exactly what PQCgenKAT_sign does.
#
# Emits the oracle path on stdout (the only stdout line), so callers can
# `ORACLE=$(tools/cref_xtest/build.sh)`.
set -euo pipefail

ref="${SQISIGN_C_REF:-$HOME/src/github.com/SQISign/the-sqisign}"
build="${SQISIGN_C_REF_BUILD:-$ref/build-ref}"
variant="${SQISIGN_VARIANT:-p324_3}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/oracle"

kat_dir="$build/apps/CMakeFiles/PQCgenKAT_sign_${variant}.dir"
if [[ ! -f "$kat_dir/link.txt" || ! -f "$kat_dir/flags.make" ]]; then
    echo "C reference not built at $build for $variant (expected $kat_dir/link.txt)" >&2
    exit 1
fi

# Archives in CMake's order, resolved against the apps/ working directory
# the link line assumes. `-lm` is re-added below.
libs=()
while IFS= read -r lib; do
    libs+=("$build/apps/$lib")
done < <(grep -o '\.\./[^ ]*\.a' "$kat_dir/link.txt")

# Defines and include paths exactly as the KAT generator was compiled.
defines=$(sed -n 's/^C_DEFINES = //p' "$kat_dir/flags.make")
includes=$(sed -n 's/^C_INCLUDES = //p' "$kat_dir/flags.make")

# shellcheck disable=SC2086
cc -O2 -std=c11 -Wall \
    $defines $includes \
    "$here/oracle.c" -o "$out" \
    "${libs[@]}" -lm

echo "$out"
