//! Ops CLI for the Fly-hosted runner stack.
//!
//! Wraps `flyctl` and `gh` so the deploy + smoke-test flow runs
//! reproducibly from a laptop, no GH-hosted CI required.
//!
//! Usage:
//!   cargo run -p ops -- deploy-orchestrator
//!   cargo run -p ops -- deploy-runner-base    # slow, run rarely
//!   cargo run -p ops -- deploy-runners        # fast, FROM the pinned base
//!   cargo run -p ops -- cleanup-orphans
//!   cargo run -p ops -- deploy-all
//!   cargo run -p ops -- smoke-test

use std::{path::PathBuf, process::Command};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

const ORCHESTRATOR_APP: &str = "sqisign-infra-orchestrator";
const RUNNERS_APP: &str = "sqisign-infra-runners";

#[derive(Parser)]
#[command(name = "ops", about = "Fly runner stack ops")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build + deploy the orchestrator service to Fly.
    DeployOrchestrator {
        /// Extra args forwarded to `fly deploy`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra: Vec<String>,
    },
    /// Build + push the heavy base runner image (texlive + Sage +
    /// rustup + system tools), tagged `:base`. Slow (~2h on a cold
    /// builder); run rarely. Overwrites the previous `:base` —
    /// audit trail lives in git history of `runners/Dockerfile`.
    DeployRunnerBase {
        /// Extra args forwarded to `fly deploy`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra: Vec<String>,
    },
    /// Build + push the runtime runner image (thin layer on top of
    /// the pinned base), tagged `latest`.
    DeployRunners {
        /// Extra args forwarded to `fly deploy`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra: Vec<String>,
    },
    /// Destroy orphan Machines in the runners app.
    CleanupOrphans,
    /// Deploy orchestrator + runners + cleanup, in order. Does NOT
    /// touch the base — bump it explicitly with `deploy-runner-base`.
    DeployAll,
    /// Dispatch the runner-smoke-test workflow + tail orchestrator logs.
    SmokeTest,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::DeployOrchestrator { extra } => deploy_orchestrator(&extra),
        Cmd::DeployRunnerBase { extra } => deploy_runner_base(&extra),
        Cmd::DeployRunners { extra } => deploy_runners(&extra),
        Cmd::CleanupOrphans => cleanup_orphans(),
        Cmd::DeployAll => {
            deploy_orchestrator(&[])?;
            deploy_runners(&[])?;
            cleanup_orphans()?;
            println!();
            println!("done. verify:");
            println!("  curl https://{ORCHESTRATOR_APP}.fly.dev/healthz");
            println!("  cargo run -p ops -- smoke-test");
            Ok(())
        }
        Cmd::SmokeTest => smoke_test(),
    }
}

/// `infra/` workspace root, computed from this crate's manifest location.
fn infra_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent")
        .to_path_buf()
}

/// Run a `Command` and bail with a descriptive error if it fails.
fn run(label: &str, cmd: &mut Command) -> Result<()> {
    let status = cmd.status().with_context(|| format!("spawning {label}"))?;
    if !status.success() {
        bail!("{label} failed (exit {:?})", status.code());
    }
    Ok(())
}

fn deploy_orchestrator(extra: &[String]) -> Result<()> {
    println!("==> deploy orchestrator");
    let mut cmd = Command::new("fly");
    cmd.current_dir(infra_dir());
    // Unlike runners, the orchestrator IS a long-running service
    // that needs the full deploy step (Machine rolling update,
    // not just a registry push). `[deploy] strategy = "immediate"`
    // in `orchestrator/fly.toml` keeps the rollout short.
    cmd.args([
        "deploy",
        "--app",
        ORCHESTRATOR_APP,
        "--config",
        "orchestrator/fly.toml",
        "--dockerfile",
        "orchestrator/Dockerfile",
    ]);
    cmd.args(extra);
    run("fly deploy orchestrator", &mut cmd)
}

fn deploy_runners(extra: &[String]) -> Result<()> {
    println!("==> deploy runtime runner image (FROM pinned base)");
    let mut cmd = Command::new("fly");
    cmd.current_dir(infra_dir().join("runners"));
    // `--build-only --push` skips flyctl's deploy step (the legacy
    // builder was hitting h2c gRPC failures after push completed).
    // The orchestrator spawns Machines from the registry image on
    // demand; no Machine deploy needed here.
    //
    // `--buildkit` opts into Fly's newer remote builder, which
    // connects over Flycast IPv6 (avoiding the public IPv4 + h2c
    // auth chain that was the source of those errors) and uses a
    // registry mirror for fast push.
    cmd.args([
        "deploy",
        "--app",
        RUNNERS_APP,
        "--image-label",
        "latest",
        "--build-target",
        "runtime",
        "--build-only",
        "--push",
        "--buildkit",
    ]);
    cmd.args(extra);
    run("fly deploy runners", &mut cmd)
}

fn deploy_runner_base(extra: &[String]) -> Result<()> {
    println!("==> deploy runner base image as :base");

    let mut cmd = Command::new("fly");
    cmd.current_dir(infra_dir().join("runners"));
    // See `deploy_runners` for `--build-only --push --buildkit`.
    cmd.args([
        "deploy",
        "--app",
        RUNNERS_APP,
        "--image-label",
        "base",
        "--build-target",
        "base",
        "--build-only",
        "--push",
        "--buildkit",
    ]);
    cmd.args(extra);
    run("fly deploy runner base", &mut cmd)?;

    println!();
    println!("==> base pushed. next: cargo run -p ops -- deploy-runners");
    Ok(())
}

fn cleanup_orphans() -> Result<()> {
    println!("==> cleanup orphans in {RUNNERS_APP}");

    let output = Command::new("fly")
        .args(["machines", "list", "-a", RUNNERS_APP, "--json"])
        .output()
        .context("fly machines list")?;
    if !output.status.success() {
        bail!(
            "fly machines list failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let machines: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("parsing fly machines list output")?;
    let orphans: Vec<&str> = machines
        .as_array()
        .context("expected JSON array from fly machines list")?
        .iter()
        .filter_map(|m| {
            let name = m.get("name")?.as_str()?;
            let id = m.get("id")?.as_str()?;
            // Orchestrator names its runners `fly-<jobid>-<hex>`. Any
            // other Machine in this app is an orphan from `fly deploy`.
            (!name.starts_with("fly-")).then_some(id)
        })
        .collect();

    if orphans.is_empty() {
        println!("  no orphans");
        return Ok(());
    }

    println!("  destroying {} orphan(s)", orphans.len());
    for id in orphans {
        println!("    {id}");
        let mut cmd = Command::new("fly");
        cmd.args(["machines", "destroy", id, "-a", RUNNERS_APP, "--force"]);
        run("fly machines destroy", &mut cmd)?;
    }
    Ok(())
}

fn smoke_test() -> Result<()> {
    println!("==> dispatch runner-smoke-test");
    let mut cmd = Command::new("gh");
    cmd.args(["workflow", "run", "runner-smoke-test.yml"]);
    run("gh workflow run", &mut cmd)?;

    println!("==> tailing orchestrator logs (Ctrl-C to stop)");
    let mut cmd = Command::new("fly");
    cmd.args(["logs", "-a", ORCHESTRATOR_APP]);
    run("fly logs", &mut cmd)
}
