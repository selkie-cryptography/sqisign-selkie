# infra/

Self-hosted GitHub Actions runners on Fly Machines. Cuts Linux CI cost
versus GH-hosted minutes; macOS and Windows jobs stay on GH-hosted.

## Layout

```
infra/
├── github-app/      one-time GitHub App creation runbook + manifest
├── orchestrator/    long-lived Axum service on Fly: webhook -> JIT -> spawn Machine
├── runners/         multi-stage runner image:
│                      stage `base` (rustup + system tools)
│                      stage `runtime` (thin: FROM pinned base + actions-runner)
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
6. First deploy: `cd infra && cargo run -p ops -- deploy-all`.

## Regular deploys

All deploys go through `infra/ops/` (Rust CLI calling `flyctl`). Run
from anywhere — the binary locates the workspace via `CARGO_MANIFEST_DIR`.

```
cd infra
cargo run -p ops -- deploy-orchestrator   # build + deploy orchestrator
cargo run -p ops -- deploy-runners        # build + push runtime image as `latest` (fast, FROM the pinned base)
cargo run -p ops -- deploy-all            # orchestrator + runners + cleanup, prints verify hints
cargo run -p ops -- smoke-test            # workflow_dispatch runner-smoke-test.yml + tail logs
cargo run -p ops -- cleanup-orphans       # destroy leaked `fly-<jobid>-<hex>` Machines
```

Trailing args after `--` are forwarded to `fly deploy`:
```
cargo run -p ops -- deploy-orchestrator -- --strategy immediate
```

### Rolling the base image

The runner image has two stages in a single `Dockerfile`. The
**base** stage (rustup + system tools, ~3 GB uncompressed) is
rebuilt rarely. The thin **runtime** stage (actions-runner binary
+ entrypoint, ~600 MB on top of base) is rebuilt whenever the
runner version bumps or the entrypoint changes — fast because its
`FROM` is `registry.fly.io/sqisign-infra-runners:base`, already in
the registry.

The image deliberately stays under Fly's **8 GB uncompressed
image-unpack ceiling**. Tools that don't fit (Sage, texlive-full)
run on `ubuntu-latest` instead.

```
cd infra
cargo run -p ops -- deploy-runner-base    # ~10–15 min cold; pushes :base
cargo run -p ops -- deploy-runners        # ~30s; runtime FROM :base, pushes :latest
```

No pinning state to track. The `:base` tag is overwritten on each
roll; git history of `runners/Dockerfile` is the audit trail. If
you want to roll back, `git revert` and re-run `deploy-runner-base`.

`--build-target base` makes BuildKit ignore the `runtime` stage's
`FROM registry.fly.io/.../runners:base`, so the first base build
doesn't depend on its own previous output. After the first base is
published, subsequent runtime builds pull `:base` from the registry
and skip rebuilding the heavy layers.

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
   - Plain `x64` (no sizing label) defaults to `shared-2x`.
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
