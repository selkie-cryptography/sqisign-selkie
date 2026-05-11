# Multi-stage build for the orchestrator binary. Final image is a
# minimal Debian slim with the orchestrator binary + ca-certs.
#
# Build context is `infra/` (the runner-stack workspace root). Run:
#
#   cd infra
#   fly deploy --app sqisign-infra-orchestrator \
#     --config orchestrator/fly.toml \
#     --dockerfile orchestrator/Dockerfile

FROM rust:slim-bookworm AS builder

WORKDIR /work

# Cache deps separately from source. Copy manifests first; the layer
# is reused until any Cargo.toml changes.
COPY Cargo.toml Cargo.lock ./
COPY orchestrator/Cargo.toml orchestrator/Cargo.toml

# Stub source file so cargo fetch + first compile populate the dep
# graph without compiling real code yet.
RUN mkdir -p orchestrator/src \
  && echo "fn main() {}" > orchestrator/src/main.rs \
  && cargo fetch --locked

# Real source.
COPY orchestrator/src orchestrator/src

RUN cargo build --release --locked --package orchestrator

FROM debian:bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /work/target/release/orchestrator /usr/local/bin/orchestrator

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/orchestrator"]
