//! Fly Machines API client.
//!
//! API reference: <https://fly.io/docs/machines/api/>

use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

const FLY_API_BASE: &str = "https://api.machines.dev/v1";

/// Machine sizes we route runner workloads to.
///
/// Maps onto Fly's named guest sizes. Add variants when routing
/// new job classes to bigger/smaller hardware.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MachineSize {
    SharedCpu2x,
    SharedCpu4x,
    PerformanceCpu2x,
    PerformanceCpu4x,
    PerformanceCpu8x,
    PerformanceCpu16x,
}

impl MachineSize {
    /// Fly's `guest.cpu_kind` + `guest.cpus` representation.
    pub fn as_fly(&self) -> (&'static str, u32) {
        match self {
            Self::SharedCpu2x => ("shared", 2),
            Self::SharedCpu4x => ("shared", 4),
            Self::PerformanceCpu2x => ("performance", 2),
            Self::PerformanceCpu4x => ("performance", 4),
            Self::PerformanceCpu8x => ("performance", 8),
            Self::PerformanceCpu16x => ("performance", 16),
        }
    }

    /// Match a `runs-on:` label list (e.g. `[self-hosted, fly, perf-8x]`)
    /// to a machine size. Returns `None` if no sizing label is set.
    ///
    /// The default for `x64` (no explicit size hint) is
    /// `PerformanceCpu4x`. Cargo parallelizes well to 4 cores; past
    /// that, returns diminish (linker is a tail-singleton). Jobs
    /// that want cheap shared CPU can opt-in via `shared-2x` /
    /// `shared-4x` labels; bigger workloads scale up via `perf-8x`
    /// / `perf-16x`.
    pub fn from_labels(labels: &[String]) -> Option<Self> {
        if labels.iter().any(|l| l == "perf-16x") {
            Some(Self::PerformanceCpu16x)
        } else if labels.iter().any(|l| l == "perf-8x") {
            Some(Self::PerformanceCpu8x)
        } else if labels.iter().any(|l| l == "perf-4x") {
            Some(Self::PerformanceCpu4x)
        } else if labels.iter().any(|l| l == "perf-2x") {
            Some(Self::PerformanceCpu2x)
        } else if labels.iter().any(|l| l == "shared-4x") {
            Some(Self::SharedCpu4x)
        } else if labels.iter().any(|l| l == "shared-2x") {
            Some(Self::SharedCpu2x)
        } else if labels.iter().any(|l| l == "x64") {
            Some(Self::PerformanceCpu4x)
        } else {
            None
        }
    }
}

/// Authenticated Fly Machines API client scoped to a single app.
#[derive(Debug, Clone)]
pub struct FlyClient {
    api_token: String,
    app: String,
    region: String,
    image_ref: String,
    http: reqwest::Client,
}

impl FlyClient {
    /// Construct a client bound to a Fly app, region, and image ref.
    pub fn new(api_token: String, app: String, region: String, image_ref: String) -> Self {
        Self {
            api_token,
            app,
            region,
            image_ref,
            http: reqwest::Client::new(),
        }
    }

    /// Spawn an ephemeral runner Machine.
    ///
    /// `jit_config` is the base64 JIT blob from
    /// [`crate::github::GitHubAppClient::mint_jit_config`], injected
    /// via the `JITCONFIG` env var. The Machine is created with
    /// `auto_destroy: true` so it self-destroys when the runner
    /// finishes its single job and exits.
    pub async fn spawn_runner(&self, size: MachineSize, jit_config: &str) -> Result<MachineId> {
        let (cpu_kind, cpus) = size.as_fly();

        let body = SpawnMachineRequest {
            region: &self.region,
            config: SpawnMachineConfig {
                image: &self.image_ref,
                env: [("JITCONFIG", jit_config)].into(),
                init: SpawnInit {
                    exec: vec!["/entrypoint.sh"],
                },
                guest: SpawnGuest {
                    cpu_kind,
                    cpus,
                    memory_mb: cpus * 2048, // 2 GB per vCPU
                },
                // Extend the runtime overlayfs so jobs that install
                // heavy deps at job time (e.g. `sage-precompute-check`
                // conda-installing Sage ~4 GB into ~/sage-env) have
                // room to write. The image-unpack ceiling (~8 GB,
                // separate hard limit) isn't affected — the slim
                // image still has to fit that on its own.
                rootfs: SpawnRootfs { size_gb: 30 },
                auto_destroy: true,
                restart: SpawnRestart { policy: "no" },
            },
        };

        let url = format!("{FLY_API_BASE}/apps/{}/machines", self.app);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("send POST {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read response body: {e}>"));
            anyhow::bail!("POST {url} returned HTTP {status}; body: {body}");
        }

        let parsed: SpawnMachineResponse =
            resp.json().await.context("parse Machines API response")?;
        Ok(MachineId(parsed.id))
    }

    /// List every Machine in the bound app. Used by the reaper to
    /// find leaked / stale-image runner Machines.
    pub async fn list_machines(&self) -> Result<Vec<Machine>> {
        let url = format!("{FLY_API_BASE}/apps/{}/machines", self.app);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await
            .with_context(|| format!("send GET {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read response body: {e}>"));
            anyhow::bail!("GET {url} returned HTTP {status}; body: {body}");
        }

        resp.json().await.context("parse Machines list response")
    }

    /// Resolves the tag in `self.image_ref` to its current sha256
    /// digest via the registry's manifest endpoint. Uses Basic auth
    /// with `x:$FLY_API_TOKEN` (same credential as `docker login
    /// registry.fly.io`). Single-platform manifests only — a
    /// multi-arch index would return a digest that won't match any
    /// Machine's platform digest.
    pub async fn resolve_latest_digest(&self) -> Result<String> {
        let (host, rest) = self
            .image_ref
            .split_once('/')
            .context("image_ref missing '/' separator")?;
        let (name, tag) = rest
            .split_once(':')
            .context("image_ref missing ':tag' suffix")?;

        let url = format!("https://{host}/v2/{name}/manifests/{tag}");
        let resp = self
            .http
            .head(&url)
            .header(
                "Accept",
                "application/vnd.oci.image.manifest.v1+json,\
                 application/vnd.docker.distribution.manifest.v2+json",
            )
            .basic_auth("x", Some(&self.api_token))
            .send()
            .await
            .with_context(|| format!("send HEAD {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("HEAD {url} returned HTTP {status}");
        }

        let digest = resp
            .headers()
            .get("docker-content-digest")
            .and_then(|v| v.to_str().ok())
            .context("response missing Docker-Content-Digest header")?
            .to_string();

        Ok(digest)
    }

    /// Force-destroy a Machine. Equivalent to `flyctl machine destroy
    /// --force`: skips graceful shutdown and removes the Machine
    /// immediately. The reaper only ever targets Machines it has
    /// already classified as zombies, so the runner-side shutdown
    /// dance has no value here.
    pub async fn destroy_machine(&self, id: &MachineId) -> Result<()> {
        let url = format!(
            "{FLY_API_BASE}/apps/{}/machines/{}?force=true",
            self.app, id.0
        );
        let resp = self
            .http
            .delete(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await
            .with_context(|| format!("send DELETE {url}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|e| format!("<failed to read response body: {e}>"));
            anyhow::bail!("DELETE {url} returned HTTP {status}; body: {body}");
        }
        Ok(())
    }
}

/// Opaque Machine identifier returned by the Fly Machines API.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MachineId(pub String);

/// Subset of `GET /v1/apps/<app>/machines` response fields the reaper
/// needs. The full payload has many more fields; deserializing only
/// what is used keeps the schema decoupled from upstream churn.
#[derive(Debug, Clone, Deserialize)]
pub struct Machine {
    /// Machine identifier (e.g. `9080d1ddc21078`).
    pub id: MachineId,
    /// RFC 3339 timestamp at which Fly created the Machine record,
    /// e.g. `2026-05-15T17:22:25Z`. Parsed lazily by
    /// [`Machine::created_at_systemtime`].
    pub created_at: String,
    /// Container image the Machine is running. The reaper compares
    /// `image_ref.digest` across Machines to detect stale-image
    /// survivors after an image rollover.
    pub image_ref: MachineImageRef,
}

impl Machine {
    /// Parses `created_at` into a [`SystemTime`].
    ///
    /// # Errors
    ///
    /// Returns an error if `created_at` is not valid RFC 3339 or if
    /// the parsed instant is before the Unix epoch (which Fly never
    /// emits in practice, but the conversion would still fail).
    pub fn created_at_systemtime(&self) -> Result<SystemTime> {
        let dt = OffsetDateTime::parse(&self.created_at, &Rfc3339)
            .with_context(|| format!("parse created_at: {}", self.created_at))?;
        let secs =
            u64::try_from(dt.unix_timestamp()).context("created_at predates the Unix epoch")?;
        Ok(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
    }

    /// Returns how long ago this Machine was created, relative to
    /// `now`.
    ///
    /// # Errors
    ///
    /// Returns an error if `created_at` fails to parse or is in the
    /// future relative to `now` (impossible in normal operation, but
    /// would indicate clock skew between the orchestrator and Fly).
    pub fn age(&self, now: SystemTime) -> Result<Duration> {
        let created = self.created_at_systemtime()?;
        now.duration_since(created)
            .context("Machine created_at is in the future")
    }
}

/// Image reference fields the reaper inspects.
#[derive(Debug, Clone, Deserialize)]
pub struct MachineImageRef {
    /// Content-addressable image digest (e.g.
    /// `sha256:3b266fd9…`). Reapers compare digests across Machines
    /// to detect a rollover.
    pub digest: String,
}

// Request/response shapes for `POST /v1/apps/<app>/machines`.
// See <https://fly.io/docs/machines/api/> for field semantics.

#[derive(Serialize)]
struct SpawnMachineRequest<'a> {
    region: &'a str,
    config: SpawnMachineConfig<'a>,
}

#[derive(Serialize)]
struct SpawnMachineConfig<'a> {
    image: &'a str,
    env: std::collections::HashMap<&'a str, &'a str>,
    init: SpawnInit,
    guest: SpawnGuest,
    rootfs: SpawnRootfs,
    auto_destroy: bool,
    restart: SpawnRestart,
}

#[derive(Serialize)]
struct SpawnInit {
    exec: Vec<&'static str>,
}

#[derive(Serialize)]
struct SpawnGuest {
    cpu_kind: &'static str,
    cpus: u32,
    memory_mb: u32,
}

/// Overlayfs sizing for the Machine's writable rootfs space. Maps
/// to flyctl's `--rootfs-size` flag and to `MachineRootfs.SizeGB`
/// in superfly/fly-go's `machine_types.go`. Lets jobs install
/// runtime-heavy deps without filling the default 8 GB.
///
/// Separate from the image-unpack ceiling — that hard limit applies
/// before the overlay is created, so the image itself still has to
/// fit Fly's 8 GB-uncompressed budget.
#[derive(Serialize)]
struct SpawnRootfs {
    size_gb: u32,
}

#[derive(Serialize)]
struct SpawnRestart {
    policy: &'static str,
}

#[derive(Deserialize)]
struct SpawnMachineResponse {
    id: String,
}
