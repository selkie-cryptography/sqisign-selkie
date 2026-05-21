//! Renders `.github/actionlint.yaml` from
//! `infra/orchestrator/runners.toml`.
//!
//! Both files declare the same self-hosted-runner label set; this
//! binary makes the TOML the single source of truth so `actionlint`
//! and the orchestrator can't drift. `infra-ci.yml` runs us with
//! `--check` to fail loudly on uncommitted drift.
//!
//! # Examples
//!
//! Print the rendered YAML to stdout (default):
//!
//! ```text
//! cargo run -p ops --bin render-actionlint
//! ```
//!
//! Verify the committed `.github/actionlint.yaml` matches the
//! orchestrator's view of the size table:
//!
//! ```text
//! cargo run -p ops --bin render-actionlint -- \
//!     --check .github/actionlint.yaml
//! ```

use std::{path::PathBuf, process::ExitCode};

use anyhow::{Context, Result, bail};
use clap::Parser;
use orchestrator::fly::Config;

/// Renders `.github/actionlint.yaml` from `runners.toml`.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the runners config. Defaults to the in-tree copy
    /// when launched via `cargo run`; falls back to the conventional
    /// workspace-relative path otherwise.
    #[arg(long, default_value_os_t = default_config_path())]
    config: PathBuf,

    /// Write the rendered YAML to this path. Mutually exclusive
    /// with `--check`. If neither flag is given, prints to stdout.
    #[arg(long, conflicts_with = "check")]
    output: Option<PathBuf>,

    /// Verify this file already matches the renderer's output;
    /// exits non-zero on drift. Used by CI.
    #[arg(long)]
    check: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("render-actionlint: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Entry point factored out so `?` works against `Result<()>` and
/// `main` can map the error into an `ExitCode`.
fn run() -> Result<()> {
    let args = Args::parse();
    let config =
        Config::load(&args.config).with_context(|| format!("loading {}", args.config.display()))?;
    let rendered = config.sizes.actionlint_yaml();

    if let Some(path) = &args.check {
        let actual =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

        if actual != rendered {
            eprintln!(
                "actionlint config drift: {} does not match \
                 runners.toml.\nRegenerate with:\n  \
                 cargo run -p ops --bin render-actionlint -- --output {}",
                path.display(),
                path.display(),
            );
            bail!("drift detected");
        }

        return Ok(());
    }

    if let Some(path) = &args.output {
        std::fs::write(path, &rendered).with_context(|| format!("writing {}", path.display()))?;
        return Ok(());
    }

    print!("{rendered}");
    Ok(())
}

/// Walks up from `CARGO_MANIFEST_DIR` to find `runners.toml`. Keeps
/// the common-case `cargo run` invocation argless without baking in
/// a workspace-absolute path.
fn default_config_path() -> PathBuf {
    if let Some(dir) = std::env::var_os("CARGO_MANIFEST_DIR") {
        // ops/Cargo.toml → ../orchestrator/runners.toml
        let p = PathBuf::from(dir).join("../orchestrator/runners.toml");
        if p.exists() {
            return p;
        }
    }

    PathBuf::from("infra/orchestrator/runners.toml")
}
