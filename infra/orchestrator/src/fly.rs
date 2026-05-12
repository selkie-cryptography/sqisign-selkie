//! Fly Machines API client.
//!
//! API reference: <https://fly.io/docs/machines/api/>

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
        } else if labels.iter().any(|l| l == "shared-2x" || l == "x64") {
            Some(Self::SharedCpu2x)
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
}

/// Opaque Machine identifier returned by the Fly Machines API.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MachineId(pub String);

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

#[derive(Serialize)]
struct SpawnRestart {
    policy: &'static str,
}

#[derive(Deserialize)]
struct SpawnMachineResponse {
    id: String,
}
