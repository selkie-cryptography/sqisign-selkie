//! Count panic-related call sites in a release rlib, attributed to the
//! calling Rust function via llvm-objdump disassembly.
//!
//! Usage: panic-count <sha>
//!
//! Emits JSON with both aggregate counts (back-compat with the
//! dashboard's Risks panel) and a per-function `sites` array suitable
//! for actionable triage.
//!
//! Compile: `rustc -O panic-count.rs -o panic-count`

use std::collections::BTreeMap;
use std::env;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: panic-count <sha>");
        std::process::exit(1);
    }
    let sha = &args[1];

    // Build release with line-tables-only debug info. This is just
    // enough for llvm-objdump's `--demangle` to attribute call sites
    // to their calling Rust function via the standard symbol table —
    // we do not parse DWARF directly.
    eprintln!("[panic-count] building release...");
    let status = Command::new("cargo")
        .env("RUSTFLAGS", "-C debuginfo=line-tables-only")
        .args(["build", "--release"])
        .status()
        .expect("failed to run cargo build");
    if !status.success() {
        eprintln!("[panic-count] cargo build failed");
        std::process::exit(1);
    }

    let lib = match find_rlib() {
        Some(l) => l,
        None => {
            eprintln!("[panic-count] could not find rlib");
            std::process::exit(1);
        }
    };
    eprintln!("[panic-count] analyzing {lib}");

    let objdump = match find_llvm_objdump() {
        Some(p) => p,
        None => {
            eprintln!("[panic-count] llvm-objdump not found (need llvm-tools-preview)");
            std::process::exit(1);
        }
    };
    eprintln!("[panic-count] using {objdump}");

    let dis = Command::new(&objdump)
        .args([
            "--disassemble",
            "--no-show-raw-insn",
            "--demangle",
            "--reloc",
            &lib,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    let sites = collect_sites(&dis);

    // Aggregate counts.
    let (mut bounds, mut asserts, mut expects, mut unwraps, mut panic_fmt) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for s in &sites {
        match s.kind.as_str() {
            "bounds_check" => bounds += s.count,
            "assert" => asserts += s.count,
            "expect" => expects += s.count,
            "unwrap" => unwraps += s.count,
            "panic_fmt" => panic_fmt += s.count,
            _ => {}
        }
    }
    let total = bounds + asserts + expects + unwraps + panic_fmt;

    let updated_at = iso8601_now();

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&updated_at))?;
    writeln!(w, "  \"total\": {total},")?;
    writeln!(w, "  \"bounds_check\": {bounds},")?;
    writeln!(w, "  \"assert\": {asserts},")?;
    writeln!(w, "  \"expect\": {expects},")?;
    writeln!(w, "  \"unwrap\": {unwraps},")?;
    writeln!(w, "  \"panic_fmt\": {panic_fmt},")?;
    writeln!(w, "  \"sites\": [")?;
    for (i, s) in sites.iter().enumerate() {
        write!(
            w,
            "    {{\"kind\": {}, \"function\": {}, \"count\": {}}}",
            json_str(&s.kind),
            json_str(&s.function),
            s.count
        )?;
        if i + 1 < sites.len() {
            writeln!(w, ",")?;
        } else {
            writeln!(w)?;
        }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

#[derive(Clone)]
struct Site {
    kind: String,
    function: String,
    count: u64,
}

/// Parse llvm-objdump output, extract panic call sites attributed to
/// the calling function, and return them grouped by (kind, normalized
/// function name) with counts summed across monomorphizations.
fn collect_sites(dis: &str) -> Vec<Site> {
    let mut grouped: BTreeMap<(String, String), u64> = BTreeMap::new();
    let mut current_caller: Option<String> = None;

    for line in dis.lines() {
        // Call/branch lines look like:
        //   <addr>:    bl    0x<addr> <symbol+offset>          (ARM64)
        //   <addr>:    callq 0x<addr> <symbol+offset>          (x86_64)
        // Extract the symbol from `<symbol+offset>`.
        if let Some(sym) = extract_call_target_symbol(line) {
            current_caller = Some(sym);
            continue;
        }

        // Relocation lines look like:
        //   <addr>:  ARM64_RELOC_BRANCH26  <mangled_helper>
        //   <addr>:  R_X86_64_PLT32        <mangled_helper>
        // Substring match identifies which panic helper.
        let kind = match panic_kind_in_line(line) {
            Some(k) => k,
            None => continue,
        };

        let Some(caller) = current_caller.as_ref() else {
            continue;
        };
        let normalized = normalize_function(caller);
        if normalized.is_empty() {
            continue;
        }
        *grouped
            .entry((kind.to_string(), normalized))
            .or_insert(0) += 1;
    }

    let mut sites: Vec<Site> = grouped
        .into_iter()
        .map(|((kind, function), count)| Site { kind, function, count })
        .collect();
    // Most-frequent first; tiebreak on function name for determinism.
    sites.sort_by(|a, b| b.count.cmp(&a.count).then(a.function.cmp(&b.function)));
    sites
}

/// `<addr>: bl 0x<...> <symbol+0xNN>` → returns the symbol body.
fn extract_call_target_symbol(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    // First token should be an address followed by `:`.
    let after_colon = trimmed.split_once(':')?.1.trim_start();
    if !(after_colon.starts_with("bl ")
        || after_colon.starts_with("bl\t")
        || after_colon.starts_with("blr ")
        || after_colon.starts_with("call ")
        || after_colon.starts_with("call\t")
        || after_colon.starts_with("callq ")
        || after_colon.starts_with("callq\t"))
    {
        return None;
    }
    let lt = line.rfind('<')?;
    let gt = line.rfind('>')?;
    if gt <= lt + 1 {
        return None;
    }
    let inside = &line[lt + 1..gt];
    let body = inside.trim_end_matches(|c: char| c.is_whitespace());
    // Strip `+0xNNN` byte-offset suffix.
    let body = match body.rfind("+0x") {
        Some(idx) => &body[..idx],
        None => body,
    };
    Some(body.trim().to_string())
}

/// Identify which panic helper a relocation/call references.
fn panic_kind_in_line(line: &str) -> Option<&'static str> {
    if line.contains("panic_bounds_check") {
        Some("bounds_check")
    } else if line.contains("expect_failed") {
        Some("expect")
    } else if line.contains("unwrap_failed") {
        Some("unwrap")
    } else if line.contains("panic_fmt") {
        Some("panic_fmt")
    } else if line.contains("assert_failed") {
        Some("assert")
    } else {
        None
    }
}

/// Strip the trailing `::h<hex>` monomorphization hash and other
/// post-name decorations (e.g. `(.llvm.<digits>)`), then decode common
/// Rust mangling escapes (`$LT$`, `$GT$`, `$u20$`, etc.) so dashboards
/// show readable signatures.
///
/// Returns an empty string for callers that can't be attributed to a
/// real Rust function (e.g. anonymous trampolines like `ltmp0`).
fn normalize_function(raw: &str) -> String {
    let mut s = raw.to_string();

    // Drop `(.llvm.NNNNN)` suffix.
    if let Some(p) = s.find(" (.llvm.") {
        s.truncate(p);
    }
    // Drop trailing `::h<16 hex chars>` monomorphization hash.
    if let Some(p) = s.rfind("::h") {
        let tail = &s[p + 3..];
        if tail.len() >= 8 && tail.chars().all(|c| c.is_ascii_hexdigit()) {
            s.truncate(p);
        }
    }
    // Decode $X$ escapes.
    let escapes = [
        ("$LT$", "<"),
        ("$GT$", ">"),
        ("$u20$", " "),
        ("$u27$", "'"),
        ("$u7b$", "{"),
        ("$u7d$", "}"),
        ("$u5b$", "["),
        ("$u5d$", "]"),
        ("$RF$", "&"),
        ("$BP$", "*"),
        ("$LP$", "("),
        ("$RP$", ")"),
        ("$C$", ","),
        ("..", "::"),
        ("_$", "<"),
        ("$_", ">"),
    ];
    for (from, to) in escapes {
        s = s.replace(from, to);
    }
    // Replace remaining `_$` artifacts and `4_usize` → `4` for readability.
    s = s.replace("_usize", "");
    if !s.starts_with("sqisign_selkie") {
        return String::new();
    }
    s
}

fn find_rlib() -> Option<String> {
    for entry in std::fs::read_dir("target/release").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "libsqisign_selkie.rlib" {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }
    let mut candidates: Vec<_> = std::fs::read_dir("target/release/deps")
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.starts_with("libsqisign_selkie-") && n.ends_with(".rlib")
        })
        .collect();
    // Newest mtime first (handles stale rlibs from previous toolchains).
    candidates.sort_by_key(|e| {
        std::fs::metadata(e.path())
            .and_then(|m| m.modified())
            .ok()
    });
    candidates
        .last()
        .map(|e| e.path().to_string_lossy().to_string())
}

/// Find llvm-objdump from the active toolchain's
/// `lib/rustlib/<target>/bin/`, falling back to `rustup which` and
/// finally a plain `PATH` lookup.
fn find_llvm_objdump() -> Option<String> {
    // Toolchain sysroot + host triple is the canonical location after
    // `rustup component add llvm-tools-preview`.
    let sysroot = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());
    let host = Command::new("rustc")
        .args(["-vV"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("host: ").map(|t| t.trim().to_string()))
        });
    if let (Some(root), Some(triple)) = (sysroot.as_ref(), host.as_ref()) {
        let cand = format!("{root}/lib/rustlib/{triple}/bin/llvm-objdump");
        if std::path::Path::new(&cand).exists() {
            return Some(cand);
        }
    }
    // `rustup which llvm-objdump` works in some setups where the binary
    // is exposed at the toolchain root.
    if let Ok(out) = Command::new("rustup")
        .args(["which", "llvm-objdump"])
        .output()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() && std::path::Path::new(&s).exists() {
            return Some(s);
        }
    }
    if Command::new("llvm-objdump")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some("llvm-objdump".to_string());
    }
    None
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
