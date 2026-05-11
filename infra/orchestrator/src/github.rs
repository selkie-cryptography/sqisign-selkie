//! GitHub App authentication + JIT runner-config minting + webhook
//! HMAC verification.
//!
//! API references:
//! - App auth: <https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/about-authentication-with-a-github-app>
//! - JIT runner: `POST /orgs/{org}/actions/runners/generate-jitconfig`
//! - Webhook signing: `X-Hub-Signature-256` header is `sha256=<hmac>`

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use hmac::{Hmac, Mac};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Authenticated GitHub App client. Holds the App's private key and
/// installation ID; mints short-lived installation access tokens
/// on demand.
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

    /// Mint a JIT runner registration config for a single job.
    ///
    /// `labels` are the runner labels matching the job's `runs-on:`.
    /// Returns the base64 JIT blob to pass to the runner via the
    /// `JITCONFIG` env var.
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

    /// Mint a short-lived (~1 hr) installation access token by signing
    /// a JWT with the App's private key, exchanging it for an
    /// installation token.
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
///
/// `signature_header` is the raw header value, e.g. `sha256=abcd...`.
/// `secret` is the webhook secret configured in the GitHub App.
///
/// Returns `Ok(())` if the signature is valid, `Err` otherwise.
/// Constant-time comparison via `hmac::Mac::verify_slice`.
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

/// Webhook payload subset we care about. GitHub's `workflow_job`
/// event has many fields; this is just what the orchestrator reads.
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
