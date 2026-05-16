//! GitHub Actions runner orchestrator on Fly.
//!
//! Receives `workflow_job` webhooks; for each job that targets the
//! self-hosted Fly label set, mints a JIT runner config and spawns
//! an ephemeral Fly Machine to run it.
//!
//! A background [`Reaper`] task periodically force-destroys Machines
//! that leaked past their job (e.g. runner crash without clean exit,
//! or stale image after a runner-image rollover).

mod fly;
mod github;
mod reaper;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use tracing::{error, info, warn};

use crate::{
    fly::{FlyClient, MachineId, MachineSize},
    github::{GitHubAppClient, WorkflowJobEvent, verify_webhook_signature},
    reaper::Reaper,
};

/// Default reaper sweep interval (10 min). Overridable via
/// `REAPER_SWEEP_INTERVAL_SECS`.
const DEFAULT_REAPER_INTERVAL_SECS: u64 = 600;

/// Default max age before a stale-digest Machine is reaped (45 min).
/// Overridable via `REAPER_MAX_AGE_SECS`. Should comfortably exceed
/// the longest single job that may run on a runner Machine.
const DEFAULT_REAPER_MAX_AGE_SECS: u64 = 45 * 60;

struct AppState {
    fly: FlyClient,
    github: GitHubAppClient,
    webhook_secret: Vec<u8>,
    /// Maps `workflow_job.id` → the Machine the orchestrator spawned
    /// for that job. Populated on `queued`, consumed on `completed`
    /// so the orchestrator can deterministically destroy the Machine
    /// even when its runner crashes without exiting (the case that
    /// breaks Fly's `auto_destroy: true` happy path).
    spawned: Mutex<HashMap<u64, MachineId>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,orchestrator=debug".into()),
        )
        .init();

    let state = Arc::new(AppState::from_env().context("loading env")?);

    let reaper_interval = env_secs("REAPER_SWEEP_INTERVAL_SECS", DEFAULT_REAPER_INTERVAL_SECS);
    let reaper_max_age = env_secs("REAPER_MAX_AGE_SECS", DEFAULT_REAPER_MAX_AGE_SECS);
    tokio::spawn(Reaper::new(state.fly.clone(), reaper_interval, reaper_max_age).run());

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/webhook", post(webhook))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!("orchestrator listening on :8080");
    axum::serve(listener, app).await?;
    Ok(())
}

impl AppState {
    fn from_env() -> Result<Self> {
        let app_id: u64 = require_env("GITHUB_APP_ID")?.parse()?;
        let installation_id: u64 = require_env("GITHUB_APP_INSTALLATION_ID")?.parse()?;
        let private_key = require_env("GITHUB_APP_PRIVATE_KEY")?;
        let webhook_secret = require_env("GITHUB_WEBHOOK_SECRET")?.into_bytes();
        let fly_token = require_env("FLY_API_TOKEN")?;
        let fly_app =
            std::env::var("FLY_RUNNER_APP").unwrap_or_else(|_| "sqisign-infra-runners".into());
        // Region runner Machines spawn into. Distinct from `FLY_REGION`,
        // which Fly auto-injects with the orchestrator's *own* region —
        // we want the runners' region to be settable independently.
        //
        // `iad` (Ashburn, VA) is co-located with Azure East US 2 where
        // GitHub Actions + its blob-storage backend run, so cache-heavy
        // CI restore time is materially faster than from `ord`. Set
        // `FLY_RUNNER_REGION=ord` (Fly secret) to fall back if iad
        // capacity is constrained.
        let fly_region = std::env::var("FLY_RUNNER_REGION").unwrap_or_else(|_| "iad".into());
        let image_ref = std::env::var("FLY_RUNNER_IMAGE")
            .unwrap_or_else(|_| format!("registry.fly.io/{fly_app}:latest"));
        let org = std::env::var("GITHUB_ORG").unwrap_or_else(|_| "selkie-cryptography".into());

        Ok(Self {
            fly: FlyClient::new(fly_token, fly_app, fly_region, image_ref),
            github: GitHubAppClient::new(app_id, installation_id, private_key, org),
            webhook_secret,
            spawned: Mutex::new(HashMap::new()),
        })
    }
}

fn require_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| anyhow::anyhow!("missing required env var: {name}"))
}

/// Reads an optional integer env var with a fallback default, in
/// seconds. Logs and falls back to the default on a parse error so a
/// typo in a Fly secret never takes the orchestrator down.
fn env_secs(name: &str, default: u64) -> Duration {
    let secs = match std::env::var(name) {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(n) => n,
            Err(e) => {
                warn!(name, raw, error = %e, "invalid env var, using default");
                default
            }
        },
        Err(_) => default,
    };
    Duration::from_secs(secs)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn webhook(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let Some(signature) = headers
        .get("X-Hub-Signature-256")
        .and_then(|v| v.to_str().ok())
    else {
        warn!("webhook missing X-Hub-Signature-256");
        return (StatusCode::UNAUTHORIZED, "missing signature").into_response();
    };

    if let Err(e) = verify_webhook_signature(&body, signature, &state.webhook_secret) {
        warn!(error = %e, "webhook signature mismatch");
        return (StatusCode::UNAUTHORIZED, "bad signature").into_response();
    }

    let event_kind = headers
        .get("X-GitHub-Event")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if event_kind != "workflow_job" {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"ignored": event_kind})),
        )
            .into_response();
    }

    let event: WorkflowJobEvent = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            error!(
                error = format!("{e:#}"),
                "failed to parse workflow_job payload"
            );
            return (StatusCode::BAD_REQUEST, "bad payload").into_response();
        }
    };

    match event.action.as_str() {
        "queued" => handle_queued(&state, event).await,
        "completed" => handle_completed(&state, event).await,
        other => (
            StatusCode::OK,
            Json(serde_json::json!({"ignored_action": other})),
        )
            .into_response(),
    }
}

/// `workflow_job: queued` handler. Mints a JIT config, spawns a
/// Machine, and records the `(job_id, machine_id)` mapping so the
/// matching `completed` event can deterministically destroy the
/// Machine even if the runner inside crashes without a clean exit.
async fn handle_queued(state: &Arc<AppState>, event: WorkflowJobEvent) -> axum::response::Response {
    if !event.workflow_job.labels.iter().any(|l| l == "fly") {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"not_for_us": event.workflow_job.labels})),
        )
            .into_response();
    }

    let size =
        MachineSize::from_labels(&event.workflow_job.labels).unwrap_or(MachineSize::SharedCpu4x);

    info!(
        job = %event.workflow_job.name,
        job_id = event.workflow_job.id,
        labels = ?event.workflow_job.labels,
        size = ?size,
        "spawning runner"
    );

    let runner_name = format!("fly-{}-{}", event.workflow_job.id, short_hex_now());
    let label_refs: Vec<&str> = event
        .workflow_job
        .labels
        .iter()
        .map(String::as_str)
        .collect();
    let jit = match state
        .github
        .mint_jit_config(&runner_name, &label_refs)
        .await
    {
        Ok(j) => j,
        Err(e) => {
            error!(error = format!("{e:#}"), "failed to mint JIT config");
            return (StatusCode::INTERNAL_SERVER_ERROR, "jit mint failed").into_response();
        }
    };

    match state.fly.spawn_runner(size, &jit).await {
        Ok(id) => {
            info!(machine = ?id, job_id = event.workflow_job.id, "runner spawned");
            // Track the spawn so a later `completed` webhook can find
            // the Machine. Mutex is uncontended (single insert per
            // webhook) and the critical section never awaits, so a
            // std Mutex is the right primitive here.
            match state.spawned.lock() {
                Ok(mut map) => {
                    map.insert(event.workflow_job.id, id.clone());
                }
                Err(poisoned) => {
                    warn!("spawned map poisoned; recovering");
                    poisoned
                        .into_inner()
                        .insert(event.workflow_job.id, id.clone());
                }
            }
            (StatusCode::OK, Json(serde_json::json!({"machine": id}))).into_response()
        }
        Err(e) => {
            error!(error = format!("{e:#}"), "failed to spawn Machine");
            (StatusCode::INTERNAL_SERVER_ERROR, "spawn failed").into_response()
        }
    }
}

/// `workflow_job: completed` handler. Looks up the Machine the
/// orchestrator spawned for this job and force-destroys it.
///
/// Defense in depth against Fly's `auto_destroy: true` happy path:
/// when a runner crashes hard enough that `./run.sh --jitconfig`
/// never returns, the Machine survives the job and would otherwise
/// only get reaped by the periodic [`Reaper`] sweep after
/// `REAPER_MAX_AGE_SECS`. Reacting to the `completed` event makes
/// the common case sub-second instead of minutes.
///
/// No-op for jobs we did not spawn (e.g. GH-hosted runs, or
/// orchestrator restarts that lost the in-memory mapping). The
/// periodic reaper still catches those.
async fn handle_completed(
    state: &Arc<AppState>,
    event: WorkflowJobEvent,
) -> axum::response::Response {
    let machine = match state.spawned.lock() {
        Ok(mut map) => map.remove(&event.workflow_job.id),
        Err(poisoned) => poisoned.into_inner().remove(&event.workflow_job.id),
    };

    let Some(machine_id) = machine else {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"untracked_job": event.workflow_job.id})),
        )
            .into_response();
    };

    info!(
        job_id = event.workflow_job.id,
        machine = %machine_id.0,
        "destroying Machine for completed job"
    );

    if let Err(e) = state.fly.destroy_machine(&machine_id).await {
        // Common-case: `auto_destroy: true` already fired and the
        // Machine is gone (HTTP 404). Log at info; the periodic
        // reaper would catch the rare leak that survives.
        info!(
            machine = %machine_id.0,
            error = format!("{e:#}"),
            "destroy on completed (likely already auto-destroyed)"
        );
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({"destroyed": machine_id})),
    )
        .into_response()
}

fn short_hex_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{now:x}")
}
