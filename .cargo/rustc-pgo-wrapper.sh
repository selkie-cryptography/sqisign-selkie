#!/usr/bin/env bash
# Cargo rustc-wrapper that adds `-Cprofile-use=<absolute path>` for
# the `sqisign_selkie` crate only, using a committed profile at
# tools/pgo/<target-triple>.profdata.
#
# Why this exists: cargo passes `[build] rustflags` to every crate in
# the dependency graph, but rustc resolves `-Cprofile-use=<path>`
# relative to the directory rustc was invoked from -- which is each
# crate's own source dir, NOT the workspace root. A workspace-relative
# committed-profile path in `.cargo/config.toml` rustflags therefore
# breaks every dependency build with "profile data does not exist".
# Resolving the path here (using BASH_SOURCE to find the repo root)
# and scoping to our crate (via --crate-name) sidesteps both problems.
#
# Build scripts, proc-macros, and every dep crate pass through
# untouched. Triples without a committed profile (e.g. Linux runners
# whose profiles haven't been uploaded yet) also pass through
# untouched; the wrapper is silent in that case so partial coverage
# does not break the build.

set -euo pipefail

# .cargo/rustc-pgo-wrapper.sh -> .cargo -> repo root
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

# argv[0] is the actual rustc path cargo selected; rest is the
# original rustc invocation.
rustc="$1"
shift

# Parse --crate-name and --target out of the rustc args.
crate_name=""
target_triple=""
prev=""
for arg in "$@"; do
    case "$prev" in
        --crate-name) crate_name="$arg" ;;
        --target)     target_triple="$arg" ;;
    esac
    prev="$arg"
done

# Default to the host triple if cargo did not pass --target.
if [[ -z "$target_triple" ]]; then
    target_triple="$("$rustc" -vV | awk '/^host:/{print $2}')"
fi

# Only PGO our own crate. Build scripts and deps are out of scope.
if [[ "$crate_name" == "sqisign_selkie" ]]; then
    profile="$repo_root/tools/pgo/$target_triple.profdata"
    if [[ -f "$profile" ]]; then
        exec "$rustc" "$@" -Cprofile-use="$profile"
    fi
fi

exec "$rustc" "$@"
