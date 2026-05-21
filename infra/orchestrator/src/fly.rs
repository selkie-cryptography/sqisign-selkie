//! Fly Machines API client.
//!
//! API reference: <https://fly.io/docs/machines/api/>

#[cfg(test)]
mod tests;

use std::{
    collections::BTreeMap,
    fmt::Write,
    path::Path,
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

const FLY_API_BASE: &str = "https://api.machines.dev/v1";

/// Per-vCPU memory allowance applied when a size omits `memory_mb`.
/// 2 GB is Fly's recommendation for general-purpose Rust builds and
/// matches the prior hardcoded ratio.
const DEFAULT_MEMORY_MB_PER_CPU: u32 = 2048;

/// Overlayfs size applied when a size omits `rootfs_gb`. Held at
/// 30 GB because `sage-precompute-check` conda-installs Sage ~4 GB
/// into the overlay on the default tier; cutting the default would
/// silently break it. Sizes that genuinely run small (lint-only)
/// opt down via the per-size `rootfs_gb` field.
const DEFAULT_ROOTFS_GB: u32 = 30;

/// A single Fly Machine guest size: kind ("shared"/"performance") +
/// CPU count, plus optional per-tier memory and rootfs overrides.
/// Loaded from `runners.toml`, not enumerated in code, so adding a
/// new tier is a config-file edit rather than a Rust patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineSize {
    /// Fly `guest.cpu_kind`: `"shared"` or `"performance"`.
    pub cpu_kind: String,

    /// Fly `guest.cpus`.
    pub cpus: u32,

    /// Optional explicit memory allowance. Reads back through
    /// [`Self::memory_mb`], which applies the per-vCPU default when
    /// unset.
    #[serde(default)]
    memory_mb: Option<u32>,

    /// Optional overlayfs size. Reads back through
    /// [`Self::rootfs_gb`], which applies [`DEFAULT_ROOTFS_GB`] when
    /// unset.
    #[serde(default)]
    rootfs_gb: Option<u32>,
}

impl MachineSize {
    /// Effective memory allowance in MB. Falls back to
    /// `cpus * DEFAULT_MEMORY_MB_PER_CPU` when the size declares no
    /// explicit value.
    #[must_use]
    pub fn memory_mb(&self) -> u32 {
        self.memory_mb
            .unwrap_or(self.cpus * DEFAULT_MEMORY_MB_PER_CPU)
    }

    /// Effective overlayfs size in GB. Falls back to a module-level
    /// default (30 GB) when the size declares no explicit value.
    #[must_use]
    pub fn rootfs_gb(&self) -> u32 {
        self.rootfs_gb.unwrap_or(DEFAULT_ROOTFS_GB)
    }
}

/// Top-level deserialization target for `runners.toml`. Owned by
/// the orchestrator's `AppState` (loaded once at startup) and shared
/// by the `render-actionlint` admin binary so config edits stay the
/// single source of truth for both runtime sizing and lint config.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Label → size table plus the `x64` default. Flattened into
    /// the top-level TOML so `default = ...` and `[sizes.*]` are
    /// siblings of `[reaper]` rather than nested under a `[sizes]`
    /// wrapper.
    #[serde(flatten)]
    pub sizes: RunnerSizes,

    /// Reaper task tuning. Section is optional in the TOML; omitted
    /// values fall back to the per-field defaults.
    #[serde(default)]
    pub reaper: ReaperConfig,
}

impl Config {
    /// Read + parse `runners.toml`. Validates cross-field invariants
    /// (e.g., `default` names a declared size) so a typo can't
    /// silently route every job to the fallback path.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read orchestrator config: {}", path.display()))?;
        let parsed: Self = toml::from_str(&raw)
            .with_context(|| format!("parse orchestrator config: {}", path.display()))?;

        if !parsed.sizes.sizes.contains_key(&parsed.sizes.default) {
            return Err(anyhow!(
                "orchestrator config {}: `default = \"{}\"` is not a declared size",
                path.display(),
                parsed.sizes.default,
            ));
        }

        Ok(parsed)
    }
}

/// Default reaper sweep interval in seconds (10 min).
const DEFAULT_REAPER_SWEEP_SECS: u64 = 600;

/// Default reaper max-age in seconds (45 min). Should comfortably
/// exceed the longest single job that may run on a runner Machine.
const DEFAULT_REAPER_MAX_AGE_SECS: u64 = 45 * 60;

/// Tunables for the background reaper task. Each field reverts to
/// its module-level default when omitted from the TOML; the
/// orchestrator's `REAPER_*` env vars still override at runtime as
/// an escape hatch for emergency tuning without a redeploy.
#[derive(Debug, Clone, Deserialize)]
pub struct ReaperConfig {
    /// Seconds between reaper sweeps.
    #[serde(default = "default_reaper_sweep_secs")]
    pub sweep_interval_secs: u64,

    /// Seconds a stale-digest Machine may live before the reaper
    /// force-destroys it.
    #[serde(default = "default_reaper_max_age_secs")]
    pub max_age_secs: u64,
}

impl Default for ReaperConfig {
    fn default() -> Self {
        Self {
            sweep_interval_secs: DEFAULT_REAPER_SWEEP_SECS,
            max_age_secs: DEFAULT_REAPER_MAX_AGE_SECS,
        }
    }
}

/// Function-form default for serde's `#[serde(default = "...")]`,
/// which requires a function path rather than a literal.
fn default_reaper_sweep_secs() -> u64 {
    DEFAULT_REAPER_SWEEP_SECS
}

/// Function-form default for serde's `#[serde(default = "...")]`,
/// which requires a function path rather than a literal.
fn default_reaper_max_age_secs() -> u64 {
    DEFAULT_REAPER_MAX_AGE_SECS
}

/// The label→size table plus the implicit default for plain `x64`.
/// Held inside [`Config`]; lives as its own type so the renderer
/// and `from_labels` lookup can be written against just the sizing
/// half of the config.
#[derive(Debug, Clone, Deserialize)]
pub struct RunnerSizes {
    /// Name of the entry plain `x64` resolves to when no explicit
    /// tier label is present.
    default: String,

    /// Label → size table. BTreeMap keeps `labels()` output stable
    /// across runs so the generated actionlint.yaml doesn't churn.
    sizes: BTreeMap<String, MachineSize>,
}

impl RunnerSizes {
    /// Match a `runs-on:` label list (e.g. `[self-hosted, fly, perf-8x]`)
    /// to a machine size. Explicit tier labels (anything in `sizes`)
    /// take precedence; plain `x64` with no explicit tier resolves to
    /// the configured `default`. Returns `None` if the label list
    /// names neither — caller treats that as "not a Fly job".
    pub fn from_labels(&self, labels: &[String]) -> Option<&MachineSize> {
        for label in labels {
            if let Some(size) = self.sizes.get(label.as_str()) {
                return Some(size);
            }
        }

        if labels.iter().any(|l| l == "x64") {
            self.sizes.get(&self.default)
        } else {
            None
        }
    }

    /// All declared size labels, in stable order. Used by
    /// `render-actionlint` to emit the lint config's allowed-label
    /// list.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.sizes.keys().map(String::as_str)
    }

    /// Resolve the configured default, e.g. for logging at startup.
    pub fn default_label(&self) -> &str {
        &self.default
    }

    /// The size plain `x64` resolves to. Guaranteed non-None by
    /// the validation in [`Config::load`].
    pub fn default_size(&self) -> &MachineSize {
        &self.sizes[&self.default]
    }

    /// Renders the `self-hosted-runner.labels` block actionlint
    /// expects, with the `fly` marker label followed by every
    /// configured size label in stable order.
    ///
    /// Output is committed verbatim to `.github/actionlint.yaml`;
    /// `infra-ci.yml` re-runs the renderer with `--check` to fail
    /// loudly on drift.
    #[must_use]
    pub fn actionlint_yaml(&self) -> String {
        let mut out = String::new();

        out.push_str("# Generated by `cargo run -p ops --bin render-actionlint`.\n");
        out.push_str("# Source of truth: `infra/orchestrator/runners.toml`.\n");
        out.push_str("# `infra-ci.yml` runs `--check` to fail loudly on drift.\n");

        out.push_str("self-hosted-runner:\n");
        out.push_str("  labels:\n");
        out.push_str("    - fly\n");
        for label in self.labels() {
            let _ = writeln!(out, "    - {label}");
        }

        out
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
    pub async fn spawn_runner(&self, size: &MachineSize, jit_config: &str) -> Result<MachineId> {
        let body = SpawnMachineRequest {
            region: &self.region,
            config: SpawnMachineConfig {
                image: &self.image_ref,
                env: [("JITCONFIG", jit_config)].into(),
                init: SpawnInit {
                    exec: vec!["/entrypoint.sh"],
                },
                guest: SpawnGuest {
                    cpu_kind: &size.cpu_kind,
                    cpus: size.cpus,
                    memory_mb: size.memory_mb(),
                },
                // Per-tier overlayfs sizing. Jobs that install heavy
                // deps at job time (e.g. `sage-precompute-check`
                // conda-installing Sage ~4 GB into ~/sage-env) set a
                // larger `rootfs_gb` on the size they route to. The
                // image-unpack ceiling (~8 GB, separate hard limit)
                // isn't affected — the slim image still has to fit
                // that on its own.
                rootfs: SpawnRootfs {
                    size_gb: size.rootfs_gb(),
                },
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

    /// First Machine whose `name` starts with `prefix`, or `None` if
    /// no match. Used by the `completed` webhook handler to find the
    /// Machine spawned for a given `workflow_job.id` without keeping
    /// any in-memory mapping.
    pub async fn find_machine_by_name_prefix(&self, prefix: &str) -> Result<Option<Machine>> {
        Ok(self
            .list_machines()
            .await?
            .into_iter()
            .find(|m| m.name.starts_with(prefix)))
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
    /// Machine name. Orchestrator-spawned runner Machines are named
    /// `fly-{workflow_job.id}-{short_hex}`, used by the completed
    /// webhook handler to look up the Machine without per-replica
    /// in-memory state.
    pub name: String,
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
    guest: SpawnGuest<'a>,
    rootfs: SpawnRootfs,
    auto_destroy: bool,
    restart: SpawnRestart,
}

#[derive(Serialize)]
struct SpawnInit {
    exec: Vec<&'static str>,
}

#[derive(Serialize)]
struct SpawnGuest<'a> {
    cpu_kind: &'a str,
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
