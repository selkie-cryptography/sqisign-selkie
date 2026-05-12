# GitHub Actions self-hosted runner image. Spawned per-job by the
# orchestrator via the Fly Machines API.
#
# Substrate only: OS, system libs, rustup toolchains, Sage, TeX Live,
# and the runner binary. Rust-ecosystem tools (cargo-nextest, deny,
# mutants, llvm-cov, fuzz, ...) are installed per-job by workflow
# `taiki-e/install-action` steps — workflow YAML is the single source
# of truth for tool versions.
#
# Layer order is heaviest-and-most-stable first so RUNNER_VERSION
# bumps don't bust the multi-GB texlive / Sage layers.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    SAGE_HOME=/opt/sage-env \
    PATH=/opt/sage-env/bin:/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# HTTPS bootstrap.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# TeX Live (~5-10 GB). latex.yml runs latexmk directly.
RUN apt-get update \
  && apt-get install -y --no-install-recommends texlive-full \
  && rm -rf /var/lib/apt/lists/*

# Miniforge — split from the Sage env so bumping sage= only rebuilds
# the env layer, not the installer.
RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -o /tmp/miniforge.sh \
  && bash /tmp/miniforge.sh -b -p /opt/miniforge \
  && rm /tmp/miniforge.sh

# Sage (~3-4 GB).
RUN /opt/miniforge/bin/conda create -y -p ${SAGE_HOME} sage=10.6 -c conda-forge \
  && /opt/miniforge/bin/conda clean -afy

# Rustup stable + nightly + clippy/rustfmt. Cross-compile targets
# stay on GitHub-hosted (Fly is x86_64-linux only).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --no-modify-path \
  && rustup toolchain install nightly --profile minimal \
  && rustup component add rustfmt clippy --toolchain nightly

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

# GitHub Actions runner binary. ARG placed here so bumps invalidate
# only from this layer down.
ARG RUNNER_VERSION=2.334.0

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

# Entrypoint reads JITCONFIG and execs one runner registration.
# Machine is spawned with auto_destroy=true → exit destroys the VM.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER runner
ENTRYPOINT ["/entrypoint.sh"]
