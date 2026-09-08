# Multi-stage build for the orchestrator binary.
#
# Build context is `infra/` (the runner-stack workspace root):
#
#   cd infra
#   fly deploy --app sqisign-infra-orchestrator \
#     --config orchestrator/fly.toml \
#     --dockerfile orchestrator/Dockerfile

FROM rust:slim-bookworm AS builder

# aws-lc-sys (jsonwebtoken's signing backend) builds its C sources
# through CMake's Makefile generator; the slim image has gcc but
# neither cmake nor make.
RUN apt-get update \
  && apt-get install -y --no-install-recommends cmake make \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Manifests first so `cargo fetch` caches in its own layer.
# Both workspace members need stubs so `cargo fetch` succeeds against
# the root `Cargo.toml`'s `members = ["orchestrator", "ops"]` list;
# we then overlay the real `orchestrator/src` for the actual build.
COPY Cargo.toml Cargo.lock ./
COPY orchestrator/Cargo.toml orchestrator/Cargo.toml
COPY ops/Cargo.toml ops/Cargo.toml

RUN mkdir -p orchestrator/src ops/src \
  && echo "fn main() {}" > orchestrator/src/main.rs \
  && echo "fn main() {}" > ops/src/main.rs \
  && cargo fetch --locked

COPY orchestrator/src orchestrator/src

RUN cargo build --release --locked --package orchestrator

FROM debian:bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /work/target/release/orchestrator /usr/local/bin/orchestrator

# Orchestrator runtime config. Loaded at startup from
# `ORCHESTRATOR_CONFIG` (default below); editing this file in tree
# and redeploying is the supported way to add a tier, tune the
# reaper, or change the default. See
# `infra/orchestrator/runners.toml` for the schema.
COPY orchestrator/runners.toml /etc/orchestrator/runners.toml

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/orchestrator"]
