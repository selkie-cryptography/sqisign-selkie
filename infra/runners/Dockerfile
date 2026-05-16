# Multi-stage runner image.
#
# Stage `base` — substrate: rustup + system tools (no Sage, no
# texlive — see comment below the bootstrap layer for why).
# Built rarely (~10–15 min cold). Pushed to
# `registry.fly.io/sqisign-infra-runners:base` (overwritten on each
# roll — git history is the audit trail).
#
# Stage `runtime` — thin layer: actions-runner binary + entrypoint.
# Built fast (~30s) because its `FROM` is the `:base` image already
# in the registry; BuildKit fetches it instead of rebuilding.
#
# Build modes:
#   deploy-runner-base  -> fly deploy --build-target base --image-label base
#   deploy-runners      -> fly deploy --build-target runtime --image-label latest
#
# `--build-target base` only builds the `base` stage; BuildKit
# ignores the `runtime` stage's `FROM` (not reachable in the DAG),
# so the first base build doesn't depend on its own previous output.

# ---------- Stage: base ----------

FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# HTTPS bootstrap.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# No `texlive-full` and no Sage here.
#
# Fly Machines enforce an 8 GB *uncompressed* image-unpack ceiling
# at the platform level (separate from per-Machine rootfs sizing).
# `texlive-full` alone is ~5 GB; a Sage 10.6 conda env adds ~4-5 GB.
# Including both pushed past the ceiling. We include the smaller
# `texlive-latex-extra` family below (~1-1.5 GB) so `latex.yml`
# can run on Fly with direct `latexmk`. Sage stays out — workflows
# that need it install at job time via conda + actions/cache.
#
# Workflows that need these:
#   - `latex.yml` -> Fly self-hosted, direct `latexmk` against the
#     `texlive-latex-extra` set below. If a tex package is missing,
#     add the specific `texlive-*` sub-package to the apt list.
#   - `sage-precompute-check` -> Fly self-hosted with conda install
#     at job time. The orchestrator sizes the runtime overlayfs via
#     `rootfs.size_gb` (see `orchestrator/src/fly.rs`) to leave room
#     for the ~4 GB conda env, and `actions/cache` keeps warm runs
#     to ~1-2 min.

# Rustup stable + nightly + clippy/rustfmt. Cross-compile targets
# stay on GitHub-hosted (Fly is x86_64-linux only).
#
# Pre-create the `runner` user so RUSTUP_HOME / CARGO_HOME can be
# chowned. Job-time `dtolnay/rust-toolchain` invocations refresh
# rustup and need write access to RUSTUP_HOME.
RUN useradd -m -s /bin/bash runner \
  && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --no-modify-path \
  && rustup toolchain install nightly --profile minimal \
  && rustup component add rustfmt clippy --toolchain nightly \
  && chown -R runner:runner "$RUSTUP_HOME" "$CARGO_HOME"

# System tools + TeX Live (latex-extra family, ~1-1.5 GB).
#
# `libclang-dev` is needed by the `crabgrind` build script (uses
# `bindgen` to generate Valgrind FFI). Without it, every `cargo
# build` of the main crate panics with "Unable to find libclang".
# Previously came in transitively via Sage's conda env; pinned
# explicitly now that Sage is no longer in the image.
#
# `mold` is wired into `.cargo/config.toml` as the linker for
# `x86_64-unknown-linux-gnu`. Cuts cargo link time ~15-30%.
#
# `texlive-*` + `latexmk` + `biber` power `latex.yml` on Fly
# self-hosted (replacing the previous `xu-cheng/latex-action`
# Docker-action path which required a Docker daemon we don't run).
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
      biber \
      build-essential \
      cmake \
      docker.io \
      git \
      jq \
      latexmk \
      libclang-dev \
      libicu-dev \
      libssl-dev \
      mold \
      pkg-config \
      python3 \
      python3-pip \
      sudo \
      texlive-bibtex-extra \
      texlive-fonts-extra \
      texlive-latex-extra \
      texlive-science \
      valgrind \
  && rm -rf /var/lib/apt/lists/*

# ---------- Stage: runtime ----------

FROM registry.fly.io/sqisign-infra-runners:base AS runtime

ARG RUNNER_VERSION=2.334.0

RUN mkdir -p /home/runner/actions-runner \
  && chown runner:runner /home/runner/actions-runner

WORKDIR /home/runner/actions-runner

USER runner
RUN curl -L -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
  && tar xzf runner.tar.gz \
  && rm runner.tar.gz
USER root
RUN ./bin/installdependencies.sh

# Entrypoint reads JITCONFIG and execs one runner registration.
# Machine is spawned with auto_destroy=true → exit destroys the VM.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER runner
ENTRYPOINT ["/entrypoint.sh"]
