//! GitHub App auth, JIT runner-config minting, webhook HMAC verify.
//!
//! Refs: [App auth][app], [JIT][jit], `X-Hub-Signature-256` for HMAC.
//!
//! [app]: https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/about-authentication-with-a-github-app
//! [jit]: https://docs.github.com/en/rest/actions/self-hosted-runners#create-configuration-for-a-just-in-time-runner-for-an-organization

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use hmac::{Hmac, Mac};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// GitHub App client. Holds the App's private key + installation ID
/// and mints short-lived installation access tokens on demand.
#[derive(Debug, Clone)]
pub struct GitHubAppClient {
    app_id: u64,
    installation_id: u64,
    private_key_pem: String,
    org: String,
    http: reqwest::Client,
}

impl GitHubAppClient {
    pub fn new(app_id: u64, installation_id: u64, private_key_pem: String, org: String) -> Self {
        Self {
            app_id,
            installation_id,
            private_key_pem,
            org,
            http: reqwest::Client::new(),
        }
    }

    /// Mint a base64 JIT runner registration config for a single job.
    /// `labels` must match the job's `runs-on:`.
    pub async fn mint_jit_config(&self, runner_name: &str, labels: &[&str]) -> Result<String> {
        let token = self.installation_token().await?;

        #[derive(Serialize)]
        struct Req<'a> {
            name: &'a str,
            runner_group_id: u32,
            labels: &'a [&'a str],
            work_folder: &'a str,
        }

        #[derive(Deserialize)]
        struct Resp {
            encoded_jit_config: String,
        }

        let url = format!(
            "https://api.github.com/orgs/{}/actions/runners/generate-jitconfig",
            self.org
        );
        let resp: Resp = self
            .http
            .post(&url)
            .bearer_auth(&token)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "sqisign-infra-orchestrator")
            .json(&Req {
                name: runner_name,
                runner_group_id: 1, // org default group
                labels,
                work_folder: "_work",
            })
            .send()
            .await
            .context("POST /generate-jitconfig")?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.encoded_jit_config)
    }

    /// Lists the currently `queued` workflow jobs in `repo` whose labels
    /// include `fly` (the ones this orchestrator is responsible for).
    ///
    /// Used by the reconcile loop to recover from missed/lost `queued`
    /// webhooks and from runner job-stealing (an ephemeral runner spawned
    /// for job A legitimately takes a different label-matching job B,
    /// leaving A queued with no runner). Walks the repo's `queued` runs,
    /// then the jobs of each, since GitHub exposes no org-wide
    /// list-queued-jobs endpoint. Low-volume repo, so the runs->jobs walk
    /// is cheap; revisit with pagination if job volume grows.
    pub async fn list_queued_fly_jobs(&self, repo: &str) -> Result<Vec<QueuedJob>> {
        let token = self.installation_token().await?;

        #[derive(Deserialize)]
        struct RunsResp {
            workflow_runs: Vec<Run>,
        }
        #[derive(Deserialize)]
        struct Run {
            id: u64,
        }
        #[derive(Deserialize)]
        struct JobsResp {
            jobs: Vec<Job>,
        }
        #[derive(Deserialize)]
        struct Job {
            id: u64,
            name: String,
            status: String,
            labels: Vec<String>,
        }

        let runs: RunsResp = self
            .http
            .get(format!(
                "https://api.github.com/repos/{}/{repo}/actions/runs?status=queued&per_page=100",
                self.org
            ))
            .bearer_auth(&token)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "sqisign-infra-orchestrator")
            .send()
            .await
            .context("GET /actions/runs?status=queued")?
            .error_for_status()?
            .json()
            .await?;

        let mut queued = Vec::new();
        for run in runs.workflow_runs {
            let jobs: JobsResp = self
                .http
                .get(format!(
                    "https://api.github.com/repos/{}/{repo}/actions/runs/{}/jobs?filter=latest&per_page=100",
                    self.org, run.id
                ))
                .bearer_auth(&token)
                .header("Accept", "application/vnd.github+json")
                .header("X-GitHub-Api-Version", "2022-11-28")
                .header("User-Agent", "sqisign-infra-orchestrator")
                .send()
                .await
                .with_context(|| format!("GET /actions/runs/{}/jobs", run.id))?
                .error_for_status()?
                .json()
                .await?;

            for job in jobs.jobs {
                if job.status == "queued" && job.labels.iter().any(|l| l == "fly") {
                    queued.push(QueuedJob {
                        id: job.id,
                        name: job.name,
                        labels: job.labels,
                    });
                }
            }
        }

        Ok(queued)
    }

    /// Mint a ~1 hr installation access token by signing a JWT with
    /// the App's private key and exchanging it via the App API.
    async fn installation_token(&self) -> Result<String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let claims = JwtClaims {
            iat: now - 60,     // GitHub allows 60s clock skew
            exp: now + 9 * 60, // 9 min, max is 10
            iss: self.app_id,
        };
        let key = EncodingKey::from_rsa_pem(self.private_key_pem.as_bytes())
            .context("invalid GitHub App private key (expected RSA PEM)")?;
        let jwt = encode(&Header::new(Algorithm::RS256), &claims, &key)?;

        #[derive(Deserialize)]
        struct Resp {
            token: String,
        }

        let url = format!(
            "https://api.github.com/app/installations/{}/access_tokens",
            self.installation_id
        );
        let resp: Resp = self
            .http
            .post(&url)
            .bearer_auth(&jwt)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "sqisign-infra-orchestrator")
            .send()
            .await
            .context("POST /installation/access_tokens")?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.token)
    }
}

#[derive(Serialize)]
struct JwtClaims {
    iat: u64,
    exp: u64,
    iss: u64,
}

/// Verify an `X-Hub-Signature-256` header against the request body.
/// Constant-time via `hmac::Mac::verify_slice`. `signature_header`
/// is the raw header value (`sha256=<hex>`); `secret` is the webhook
/// shared secret.
pub fn verify_webhook_signature(body: &[u8], signature_header: &str, secret: &[u8]) -> Result<()> {
    let expected_hex = signature_header
        .strip_prefix("sha256=")
        .ok_or_else(|| anyhow!("signature header missing sha256= prefix"))?;
    let expected = hex_decode(expected_hex)?;

    let mut mac = HmacSha256::new_from_slice(secret).context("invalid webhook secret length")?;
    mac.update(body);
    mac.verify_slice(&expected)
        .map_err(|_| anyhow!("webhook HMAC mismatch"))?;
    Ok(())
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if s.len() & 1 != 0 {
        return Err(anyhow!("hex string has odd length"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| anyhow!("invalid hex: {e}")))
        .collect()
}

/// Subset of GitHub's `workflow_job` webhook payload we read.
#[derive(Debug, Deserialize)]
pub struct WorkflowJobEvent {
    pub action: String, // "queued", "in_progress", "completed", "waiting"
    pub workflow_job: WorkflowJob,
}

#[derive(Debug, Deserialize)]
pub struct WorkflowJob {
    pub id: u64,
    pub name: String,
    pub labels: Vec<String>,
}

/// A currently-queued `fly` job, from polling the Actions API rather
/// than a webhook. Mirrors the fields of [`WorkflowJob`] the spawn path
/// needs (id for the runner-name prefix, labels for sizing/JIT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedJob {
    pub id: u64,
    pub name: String,
    pub labels: Vec<String>,
}
