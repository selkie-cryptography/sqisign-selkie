//! GitHub Actions runner orchestrator on Fly.
//!
//! Receives `workflow_job` webhooks; for each job that targets the
//! self-hosted Fly label set, mints a JIT runner config and spawns
//! an ephemeral Fly Machine to run it.
//!
//! A background [`Reaper`] task periodically force-destroys Machines
//! that leaked past their job (e.g. runner crash without clean exit,
//! or stale image after a runner-image rollover).

use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use orchestrator::{
    fly::{Config, FlyClient, Machine, SpawnOutcome},
    github::{GitHubAppClient, WorkflowJobEvent, verify_webhook_signature},
    reaper::Reaper,
    reconciler::Reconciler,
};
use tracing::{error, info, warn};

struct AppState {
    fly: FlyClient,
    github: GitHubAppClient,
    config: Config,
    webhook_secret: Vec<u8>,
    /// Repo the reconcile loop polls for queued jobs (`name` only; the
    /// org is held by the GitHub client). Webhook spawns don't need it.
    repo: String,
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

    // Reaper tunables come from `runners.toml`, edited and redeployed
    // like any other config (config changes always ride a redeploy, so a
    // runtime env override would buy nothing).
    let reaper_interval = Duration::from_secs(state.config.reaper.sweep_interval_secs);
    let reaper_max_age = Duration::from_secs(state.config.reaper.max_age_secs);
    tokio::spawn(Reaper::new(state.fly.clone(), reaper_interval, reaper_max_age).run());

    // Reconcile loop: a backstop alongside the webhook path that recovers
    // queued jobs the webhook missed or that lost their runner to
    // job-stealing. Always on, like the reaper. Spawns are bounded by the
    // dedup against live Machines and Fly's machine limit.
    tokio::spawn(
        Reconciler::new(
            state.github.clone(),
            state.fly.clone(),
            state.config.sizes.clone(),
            state.repo.clone(),
        )
        .run(),
    );

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
        let repo = std::env::var("GITHUB_REPO").unwrap_or_else(|_| "sqisign-selkie".into());

        // Path inside the runtime image. Dockerfile.runtime COPYs
        // `runners.toml` to this location; overridable for local
        // smoke testing without a rebuild.
        let config_path: PathBuf = std::env::var_os("ORCHESTRATOR_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/etc/orchestrator/runners.toml"));
        let config = Config::load(&config_path).with_context(|| {
            format!("loading orchestrator config from {}", config_path.display())
        })?;
        info!(
            path = %config_path.display(),
            default = config.sizes.default_label(),
            reaper_sweep_secs = config.reaper.sweep_interval_secs,
            reaper_max_age_secs = config.reaper.max_age_secs,
            "loaded orchestrator config",
        );

        Ok(Self {
            fly: FlyClient::new(fly_token, fly_app, fly_region, image_ref),
            github: GitHubAppClient::new(app_id, installation_id, private_key, org),
            config,
            webhook_secret,
            repo,
        })
    }
}

fn require_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| anyhow::anyhow!("missing required env var: {name}"))
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

    // `from_labels` returns `None` only when the label list has no
    // explicit tier *and* no `x64`. The earlier "fly" check gates us
    // to Fly jobs, so this should be unreachable in practice, but
    // fall back to the configured default instead of panicking.
    let size = state
        .config
        .sizes
        .from_labels(&event.workflow_job.labels)
        .unwrap_or_else(|| state.config.sizes.default_size());

    info!(
        job = %event.workflow_job.name,
        job_id = event.workflow_job.id,
        labels = ?event.workflow_job.labels,
        cpu_kind = %size.cpu_kind,
        cpus = size.cpus,
        "spawning runner"
    );

    let runner_name = Machine::runner_name(event.workflow_job.id);
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
        Ok(SpawnOutcome::Spawned(id)) => {
            info!(machine = ?id, job_id = event.workflow_job.id, "runner spawned");
            (StatusCode::OK, Json(serde_json::json!({"machine": id}))).into_response()
        }
        Ok(SpawnOutcome::AtCapacity) => {
            // At the org machine limit. Leave the job queued on GitHub;
            // the reconcile loop places it once a slot frees. Ack 200 so
            // GitHub doesn't treat the delivery as failed.
            info!(
                job_id = event.workflow_job.id,
                "at capacity; job left queued for reconcile loop"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({"queued": "at_capacity"})),
            )
                .into_response()
        }
        Err(e) => {
            error!(error = format!("{e:#}"), "failed to spawn Machine");
            (StatusCode::INTERNAL_SERVER_ERROR, "spawn failed").into_response()
        }
    }
}

/// `workflow_job: completed` handler. Finds the Machine the
/// orchestrator spawned for this job by name prefix and force-destroys
/// it, defending against the case where `auto_destroy: true` doesn't
/// fire (runner crashed without clean exit).
///
/// Machine name is `fly-{workflow_job.id}-{short_hex}`, so the prefix
/// `fly-{job_id}-` uniquely identifies the spawn. No per-replica
/// in-memory state needed: every orchestrator replica sees the same
/// Fly Machines API, so this works correctly under any replica count.
///
/// No-op for jobs we did not spawn or that `auto_destroy` already
/// cleaned up. The periodic reaper covers anything missed.
async fn handle_completed(
    state: &Arc<AppState>,
    event: WorkflowJobEvent,
) -> axum::response::Response {
    let job_id = event.workflow_job.id;
    let prefix = format!("fly-{job_id}-");

    let machine = match state.fly.find_machine_by_name_prefix(&prefix).await {
        Ok(Some(m)) => m,
        Ok(None) => {
            return (
                StatusCode::OK,
                Json(serde_json::json!({"untracked_job": job_id})),
            )
                .into_response();
        }
        Err(e) => {
            warn!(
                error = format!("{e:#}"),
                "list_machines failed in completed handler"
            );
            return (StatusCode::INTERNAL_SERVER_ERROR, "list failed").into_response();
        }
    };

    info!(
        job_id,
        machine = %machine.id.0,
        name = %machine.name,
        "destroying Machine for completed job"
    );

    if let Err(e) = state.fly.destroy_machine(&machine.id).await {
        // Common-case: `auto_destroy: true` raced us and the Machine
        // is already gone (HTTP 404). The periodic reaper would catch
        // anything else that survives.
        info!(
            machine = %machine.id.0,
            error = format!("{e:#}"),
            "destroy on completed (likely already auto-destroyed)"
        );
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({"destroyed": machine.id})),
    )
        .into_response()
}
