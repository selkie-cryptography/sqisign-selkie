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
    pub fn new(api_token: String, app: String, region: String, image_ref: String) -> Self {
        Self {
            api_token,
            app,
            region,
            image_ref,
            http: reqwest::Client::new(),
        }
    }

    /// Spawn an ephemeral runner Machine from the pre-baked image.
    ///
    /// `jit_config` is a base64-encoded JIT runner registration blob
    /// minted by [`crate::github::GitHubAppClient::mint_jit_config`].
    /// It's passed to the runner via the `JITCONFIG` env var; the
    /// runner consumes it on startup, registers with GitHub, and
    /// runs exactly one job.
    ///
    /// `auto_destroy: true` means the Machine self-destroys on exit
    /// (which happens after the runner finishes its single job).
    pub async fn spawn_runner(&self, size: MachineSize, jit_config: &str) -> Result<MachineId> {
        let (cpu_kind, cpus) = size.as_fly();

        let body = SpawnMachineRequest {
            region: &self.region,
            config: SpawnMachineConfig {
                image: &self.image_ref,
                env: [("JITCONFIG", jit_config)].into(),
                init: SpawnInit {
                    // entrypoint.sh in the runner image consumes JITCONFIG
                    // and execs `./run.sh --jitconfig $JITCONFIG`.
                    exec: vec!["/entrypoint.sh"],
                },
                guest: SpawnGuest {
                    cpu_kind,
                    cpus,
                    memory_mb: cpus * 2048, // 2 GB per vCPU
                },
                auto_destroy: true,
                restart: SpawnRestart {
                    policy: "no", // ephemeral; failure → destroy, no retry
                },
            },
        };

        let url = format!("{FLY_API_BASE}/apps/{}/machines", self.app);
        let resp: SpawnMachineResponse = self
            .http
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&body)
            .send()
            .await
            .context("POST /machines")?
            .error_for_status()?
            .json()
            .await?;

        Ok(MachineId(resp.id))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MachineId(pub String);

// ---- Request/response shapes (private to this module) ----------

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
