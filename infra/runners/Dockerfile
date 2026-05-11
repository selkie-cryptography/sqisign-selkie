# GitHub Actions self-hosted runner, pre-baked with the toolchain
# every Selkie CI job needs. Built once via `fly deploy --build-only`,
# spawned per-job by the orchestrator via the Fly Machines API.
#
# The image is intentionally fat: every cargo subcommand and tool the
# workflows use is pre-installed so the per-job startup cost is just
# Machine boot + GitHub JIT registration (target: < 5 s).

FROM ubuntu:24.04

ARG RUNNER_VERSION=2.328.0

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Base OS + build tooling.
# - build-essential, gcc-multilib: Rust + cross-target builds (i686).
# - cmake, libssl-dev: dependency of some crates and cref-bench.
# - valgrind: required by iai-callgrind for instruction counting.
# - git, curl, ca-certificates, jq: registration + general use.
# - sudo, libicu-dev: required by the GitHub Actions runner.
# - python3: kat-matrix step + various scripts.
# - docker.io: latex-action runs in a Docker container (DinD pattern).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      docker.io \
      gcc-multilib \
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

# Rust toolchain. Pre-install both stable and nightly; clippy/fmt/fuzz
# need nightly while gate/test/kat are stable.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --no-modify-path \
  && rustup toolchain install nightly --profile minimal \
  && rustup component add rustfmt clippy --toolchain nightly \
  && rustup target add i686-unknown-linux-gnu --toolchain stable \
  && rustup target add aarch64-unknown-linux-gnu --toolchain stable

# Cargo subcommands the workflows invoke. Bump the image tag when
# versions advance.
RUN cargo install --locked \
      cargo-nextest \
      cargo-deny \
      cargo-mutants \
      iai-callgrind-runner

# GitHub Actions runner binary.
RUN useradd -m -s /bin/bash runner \
  && mkdir -p /home/runner/actions-runner \
  && chown runner:runner /home/runner/actions-runner

WORKDIR /home/runner/actions-runner

USER runner
RUN curl -L -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
  && tar xzf runner.tar.gz \
  && rm runner.tar.gz
USER root
RUN ./bin/installdependencies.sh

# Entrypoint reads JITCONFIG env var, configures runner, runs once,
# exits. The Machine is configured with auto_destroy=true so exit
# triggers destruction.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER runner
ENTRYPOINT ["/entrypoint.sh"]
