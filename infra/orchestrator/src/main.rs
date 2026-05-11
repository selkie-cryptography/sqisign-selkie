//! GitHub Actions runner orchestrator on Fly.
//!
//! Receives `workflow_job` webhooks; for each job that targets the
//! self-hosted Fly label set, mints a JIT runner config and spawns
//! an ephemeral Fly Machine to run it.

mod fly;
mod github;

use std::sync::Arc;

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
    fly::{FlyClient, MachineSize},
    github::{GitHubAppClient, WorkflowJobEvent, verify_webhook_signature},
};

#[derive(Clone)]
struct AppState {
    fly: FlyClient,
    github: GitHubAppClient,
    webhook_secret: Vec<u8>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,orchestrator=debug,runner_core=debug".into()),
        )
        .init();

    let state = AppState::from_env().context("loading env")?;

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/webhook", post(webhook))
        .with_state(Arc::new(state));

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
        let fly_app = std::env::var("FLY_RUNNER_APP").unwrap_or_else(|_| "sqisign-infra-runners".into());
        let fly_region = std::env::var("FLY_REGION").unwrap_or_else(|_| "ord".into());
        let image_ref = std::env::var("FLY_RUNNER_IMAGE")
            .unwrap_or_else(|_| format!("registry.fly.io/{fly_app}:latest"));
        let org = std::env::var("GITHUB_ORG").unwrap_or_else(|_| "selkie-cryptography".into());

        Ok(Self {
            fly: FlyClient::new(fly_token, fly_app, fly_region, image_ref),
            github: GitHubAppClient::new(app_id, installation_id, private_key, org),
            webhook_secret,
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
    // 1. Verify HMAC signature.
    let signature = match headers
        .get("X-Hub-Signature-256")
        .and_then(|v| v.to_str().ok())
    {
        Some(s) => s,
        None => {
            warn!("webhook missing X-Hub-Signature-256");
            return (StatusCode::UNAUTHORIZED, "missing signature").into_response();
        }
    };

    if let Err(e) = verify_webhook_signature(&body, signature, &state.webhook_secret) {
        warn!(error = %e, "webhook signature mismatch");
        return (StatusCode::UNAUTHORIZED, "bad signature").into_response();
    }

    // 2. Only act on workflow_job events.
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

    // 3. Parse, route by labels.
    let event: WorkflowJobEvent = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => {
            error!(error = %e, "failed to parse workflow_job payload");
            return (StatusCode::BAD_REQUEST, "bad payload").into_response();
        }
    };

    if event.action != "queued" {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"ignored_action": event.action})),
        )
            .into_response();
    }

    // Only spawn for jobs that opted into the self-hosted Fly pool.
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
        labels = ?event.workflow_job.labels,
        size = ?size,
        "spawning runner"
    );

    // 4. Mint JIT config.
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
            error!(error = %e, "failed to mint JIT config");
            return (StatusCode::INTERNAL_SERVER_ERROR, "jit mint failed").into_response();
        }
    };

    // 5. Spawn Machine.
    match state.fly.spawn_runner(size, &jit).await {
        Ok(id) => {
            info!(machine = ?id, "runner spawned");
            (StatusCode::OK, Json(serde_json::json!({"machine": id}))).into_response()
        }
        Err(e) => {
            error!(error = %e, "failed to spawn Machine");
            (StatusCode::INTERNAL_SERVER_ERROR, "spawn failed").into_response()
        }
    }
}

fn short_hex_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{now:x}")
}
