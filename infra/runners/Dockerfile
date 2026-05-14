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
# A Sage 10.6 conda env adds ~4-5 GB; with texlive removed and Sage
# removed, the image stays well under 8 GB and unpacks fine on
# `shared-cpu-2x`.
#
# Workflows that need these:
#   - `latex.yml` -> `ubuntu-latest` + `xu-cheng/latex-action`
#     (Docker action — Fly runners have no Docker daemon).
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

# Smaller system tools. Separate layer so adding/removing one of
# these doesn't invalidate texlive above.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      docker.io \
      git \
      jq \
      libicu-dev \
      libssl-dev \
      pkg-config \
      python3 \
      python3-pip \
      sudo \
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
