//! Capture the public API surface and output JSON for the dashboard.
//!
//! Usage: api-surface <sha>
//!
//! Runs `cargo public-api` and counts public items by category.
//!
//! Compile: `rustc -O api-surface.rs -o api-surface`

use std::env;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: api-surface <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    let output = Command::new("cargo")
        .args(["public-api"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to run cargo public-api");

    let text = String::from_utf8_lossy(&output.stdout);

    // Count by category.
    let mut fns = 0u32;
    let mut structs = 0u32;
    let mut enums = 0u32;
    let mut traits = 0u32;
    let mut impls = 0u32;
    let mut consts = 0u32;
    let mut types = 0u32;
    let mut other = 0u32;
    let mut items: Vec<String> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        items.push(line.to_string());
        if line.starts_with("pub fn ") || line.contains("pub fn ") {
            fns += 1;
        } else if line.starts_with("pub struct ") || line.contains("pub struct ") {
            structs += 1;
        } else if line.starts_with("pub enum ") || line.contains("pub enum ") {
            enums += 1;
        } else if line.starts_with("pub trait ") || line.contains("pub trait ") {
            traits += 1;
        } else if line.starts_with("impl ") {
            impls += 1;
        } else if line.starts_with("pub const ") || line.contains("pub const ") {
            consts += 1;
        } else if line.starts_with("pub type ") || line.contains("pub type ") {
            types += 1;
        } else {
            other += 1;
        }
    }

    let total = items.len();

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());

    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"total\": {},", total)?;
    writeln!(w, "  \"functions\": {},", fns)?;
    writeln!(w, "  \"structs\": {},", structs)?;
    writeln!(w, "  \"enums\": {},", enums)?;
    writeln!(w, "  \"traits\": {},", traits)?;
    writeln!(w, "  \"impls\": {},", impls)?;
    writeln!(w, "  \"consts\": {},", consts)?;
    writeln!(w, "  \"types\": {},", types)?;
    writeln!(w, "  \"other\": {}", other)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let secs = dur.as_secs();
    let (h, m, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    let mut y = 1970i64;
    let mut rem = (secs / 86400) as i64;
    loop {
        let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if rem < yd { break }
        rem -= yd;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let md = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut mo = 0;
    for &d in &md { if rem < d { break } rem -= d; mo += 1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo + 1, rem + 1, h, m, s)
}
