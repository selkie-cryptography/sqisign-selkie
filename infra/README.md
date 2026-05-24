# infra/

Self-hosted GitHub Actions runners on Fly Machines. Cuts Linux CI cost
versus GH-hosted minutes; macOS and Windows jobs stay on GH-hosted.

## Layout

```
infra/
├── github-app/      one-time GitHub App creation runbook + manifest
├── orchestrator/    long-lived Axum service on Fly: webhook -> JIT -> spawn Machine
├── runners/         runner image, split across two Dockerfiles:
│                      Dockerfile.base    -> :base    (rustup + system tools, ~3 GB, rare rebuilds)
│                      Dockerfile.runtime -> :latest  (thin: FROM :base + actions-runner)
│                      fly.base.toml      drives the :base build
│                      fly.toml           drives the :latest build
└── ops/             laptop-driven deploy CLI
```

Fly apps:
- `sqisign-infra-orchestrator` — the always-on webhook service.
- `sqisign-infra-runners` — namespace for ephemeral runner Machines.
  The same app hosts the image at `registry.fly.io/sqisign-infra-runners:latest`.

## Bootstrap (one-time)

1. `fly apps create sqisign-infra-orchestrator`
2. `fly apps create sqisign-infra-runners`
3. Open `infra/github-app/create-form.html` in a browser and follow
   the manifest-based create flow. See `github-app/setup.md` for the
   step-by-step. Saves: app ID, private key (PEM), webhook secret.
4. Install the GitHub App on the repo/org.
5. Set orchestrator secrets:
   ```
   fly secrets set -a sqisign-infra-orchestrator \
     GITHUB_APP_ID=... \
     GITHUB_APP_PRIVATE_KEY="$(cat path/to/key.pem)" \
     WEBHOOK_SECRET=... \
     FLY_API_TOKEN="$(fly tokens create deploy -a sqisign-infra-runners)" \
     RUNNER_IMAGE=registry.fly.io/sqisign-infra-runners:latest
   ```
6. First deploy:
   ```
   cd infra
   cargo run -p ops -- deploy-runner-base   # publishes :base; needed before :latest can FROM it
   cargo run -p ops -- deploy-all           # orchestrator + :latest runner image + cleanup
   ```
   `deploy-all` deliberately doesn't touch `:base` — that's an
   explicit bump via `deploy-runner-base`.

## Regular deploys

All deploys go through `infra/ops/` (Rust CLI calling `flyctl`).
`cargo run -p ops` only resolves from a directory whose workspace
contains the `ops` crate — that's `infra/`, not the repo root.
Run from `infra/` or pass `--manifest-path infra/Cargo.toml`:

```
cd infra
cargo run -p ops -- deploy-orchestrator   # build + deploy orchestrator
cargo run -p ops -- deploy-runners        # build + push runtime image as :latest (fast, FROM the pinned :base)
cargo run -p ops -- deploy-all            # orchestrator + :latest + cleanup, prints verify hints
cargo run -p ops -- smoke-test            # workflow_dispatch runner-smoke-test.yml + tail logs
cargo run -p ops -- cleanup-orphans       # destroy leaked `fly-<jobid>-<hex>` Machines
```

Trailing args after `--` are forwarded to `fly deploy`:
```
cargo run -p ops -- deploy-orchestrator -- --strategy immediate
```

The compiled `ops` binary uses `CARGO_MANIFEST_DIR` (baked in at
build time) to find `infra/` regardless of where it's run from, so
once built you can invoke `./target/debug/ops <cmd>` from anywhere.
The `cargo run` constraint above is purely about workspace
resolution.

### Rolling the base image

The runner image is split across two Dockerfiles:

- **`runners/Dockerfile.base`** — rustup, system tools, TeX Live,
  AWS CLI, `gh`. ~3 GB uncompressed. Rebuilt rarely (Dockerfile
  edits, the weekly cron in `infra-ci.yml`, or manual roll). Built
  by `deploy-runner-base` via `fly.base.toml`, pushed as
  `registry.fly.io/sqisign-infra-runners:base`.
- **`runners/Dockerfile.runtime`** — `FROM registry.fly.io/sqisign-infra-runners:base`
  + the actions-runner binary + `entrypoint.sh`. ~600 MB on top of
  `:base`. Built by `deploy-runners` via `fly.toml`, pushed as
  `:latest`. Fast because the heavy layers come from the registry.

The image deliberately stays under Fly's **8 GB uncompressed
image-unpack ceiling**. Tools that don't fit (Sage, texlive-full)
run on `ubuntu-latest` instead.

```
cd infra
cargo run -p ops -- deploy-runner-base    # ~10–15 min cold; pushes :base
cargo run -p ops -- deploy-runners        # ~30s; runtime FROM :base, pushes :latest
```

No pinning state to track. The `:base` tag is overwritten on each
roll; git history of `runners/Dockerfile.base` is the audit trail.
To roll back: `git revert` the offending change and re-run
`deploy-runner-base`, then `deploy-runners` so `:latest` picks up
the rolled-back `:base`.

CI automates this: `infra-ci.yml` watches both Dockerfiles on push
to `main`, rebuilds `:base` then `:latest` in order on a
`Dockerfile.base` edit, and rebuilds `:latest` only on a
`Dockerfile.runtime` edit.

## Runtime flow (per job)

1. GitHub fires a `workflow_job` webhook to
   `https://sqisign-infra-orchestrator.fly.dev/webhook`.
2. Orchestrator verifies `X-Hub-Signature-256` (HMAC-SHA256, constant-time).
3. Filter: keep only `action == "queued"` with `fly` in the label list.
4. Mint a JIT runner config:
   - Build a GitHub App JWT (RS256, 10 min TTL).
   - Exchange for an installation access token.
   - `POST /orgs/{org}/actions/runners/generate-jitconfig`.
5. Pick Machine size from labels via `MachineSize::from_labels`:
   - `perf-2x` / `4x` / `8x` / `16x` — Fly performance CPUs.
   - `shared-2x` / `4x` — Fly shared CPUs.
   - Plain `x64` (no sizing label) defaults to `shared-4x`. Jobs that
     genuinely need dedicated cores opt up with `perf-2x` / `perf-4x` /
     `perf-8x`; everything else rides shared CPU.
6. `POST /v1/apps/sqisign-infra-runners/machines` with
   `auto_destroy: true` and `JITCONFIG` in the env.
7. Machine boots → `entrypoint.sh` execs `./run.sh --jitconfig "$JITCONFIG"`.
8. Runner registers JIT, picks up exactly one job, exits.
9. Fly destroys the Machine.

Workflows that target Fly use `runs-on: [self-hosted, fly, linux, x64, ...]`.
The full set of labels accepted is declared in `.github/actionlint.yaml`.

## Fork-PR safety (public repo)

Fork PRs run arbitrary contributor code and must NOT execute on Fly.
The trust boundary in workflows is the conditional:

```yaml
runs-on: ${{ fromJSON((github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) && '["self-hosted","fly","linux","x64"]' || '["ubuntu-latest"]') }}
```

Push to main, schedule, `workflow_dispatch`, and same-repo PRs route
to Fly; fork PRs route to `ubuntu-latest`. The orchestrator does not
implement an authorization step beyond this — write-access to the repo
is the trust line. Defense in depth: GitHub Settings → Actions →
"Require approval for first-time contributors".

## Secret rotation

Webhook secret:
```
# 1. GitHub App settings → "Webhook secret" → regenerate
fly secrets set -a sqisign-infra-orchestrator WEBHOOK_SECRET=...
```

GitHub App private key:
```
# 1. GitHub App settings → "Private keys" → "Generate a private key"
# 2. (download the new .pem, delete the old one in the same UI)
fly secrets set -a sqisign-infra-orchestrator \
  GITHUB_APP_PRIVATE_KEY="$(cat new-key.pem)"
```

Fly API token (deploy token for `sqisign-infra-runners`):
```
fly tokens revoke <old-token-id>          # see `fly tokens list`
NEW=$(fly tokens create deploy -a sqisign-infra-runners)
fly secrets set -a sqisign-infra-orchestrator FLY_API_TOKEN="$NEW"
```

Rotations restart the orchestrator Machine; in-flight runner Machines
keep their JIT and finish their jobs.

## Diagnostics

```
fly logs -a sqisign-infra-orchestrator                 # webhook handler logs
fly machines list -a sqisign-infra-runners             # current runners
fly machines list -a sqisign-infra-runners --json | jq '.[] | {id, state, config: .config.env}'
curl https://sqisign-infra-orchestrator.fly.dev/healthz
```

The orchestrator logs every webhook with the full anyhow chain on
error (`error = format!("{e:#}")`). HMAC verification failures and
JIT-mint failures are the two common shapes.

## Adding a Linux job to Fly

In the workflow file:
```yaml
jobs:
  my-job:
    runs-on: ${{ fromJSON((github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) && '["self-hosted","fly","linux","x64"]' || '["ubuntu-latest"]') }}
```

If the job needs more CPU, append a size label:
`["self-hosted","fly","linux","x64","perf-8x"]`. Sizes are declared
in `.github/actionlint.yaml`; if you need a new size, add it both
there and to `MachineSize::from_labels` in `orchestrator/src/fly.rs`.

If the job only needs to run on push/schedule/dispatch (never PR),
the conditional is redundant — use the unconditional form:
```yaml
runs-on: [self-hosted, fly, linux, x64]
if: github.event_name != 'pull_request'
```
