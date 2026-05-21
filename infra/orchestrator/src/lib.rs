//! Shared library for the orchestrator binary and its helpers.
//!
//! Exposes the modules consumed by both `src/main.rs` (the long-
//! lived HTTP service) and `src/bin/render-actionlint.rs` (the
//! actionlint-config generator). Anything that needs to be shared
//! by both lives here; binary-private logic stays in its own file.

pub mod fly;
pub mod github;
pub mod reaper;
