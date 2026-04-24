//! Collect binary size info and cargo-bloat top functions. Output JSON.
//!
//! Usage: size-report <sha>
//!
//! Expects: `cargo build --release` already done, `cargo bloat` installed.
//!
//! Compile: `rustc -O size-report.rs -o size-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 { eprintln!("usage: size-report <sha>"); std::process::exit(1); }
    let sha = &args[1];

    // Find the rlib.
    let lib = find_lib().unwrap_or_default();

    // Get text/data/bss via `size`.
    let (text, data, bss) = if !lib.is_empty() {
        parse_size_output(&lib)
    } else {
        (0, 0, 0)
    };

    // Total file size.
    let total_bytes = if !lib.is_empty() {
        fs::metadata(&lib).map(|m| m.len()).unwrap_or(0)
    } else { 0 };

    let lib_name = lib.rsplit('/').next().unwrap_or("unknown");

    // Cargo bloat top functions.
    let bloat_output = Command::new("cargo")
        .args(["bloat", "--release", "-n", "20", "--message-format=json"])
        .stdout(Stdio::piped()).stderr(Stdio::null())
        .output().ok();

    let mut top_functions = Vec::new();
    if let Some(output) = bloat_output {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            // Parse the JSON manually — extract function entries.
            // Format: {"file-size":N,"text-section-size":N,"functions":[{"name":"...","size":N}, ...]}
            if let Some(fns_start) = text.find("\"functions\"") {
                let rest = &text[fns_start..];
                if let Some(arr_start) = rest.find('[') {
                    for chunk in rest[arr_start..].split('{').skip(1).take(20) {
                        let name = extract_str_from(chunk, "name");
                        let size = extract_num_from(chunk, "size");
                        if !name.is_empty() {
                            top_functions.push((name, size));
                        }
                    }
                }
            }
        }
    }

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"binary\": {{")?;
    writeln!(w, "    \"file\": {},", json_str(lib_name))?;
    writeln!(w, "    \"total_bytes\": {},", total_bytes)?;
    writeln!(w, "    \"text\": {},", text)?;
    writeln!(w, "    \"data\": {},", data)?;
    writeln!(w, "    \"bss\": {}", bss)?;
    writeln!(w, "  }},")?;
    writeln!(w, "  \"top_functions\": [")?;
    for (i, (name, size)) in top_functions.iter().enumerate() {
        write!(w, "    {{\"name\": {}, \"size\": {}}}", json_str(name), size)?;
        if i + 1 < top_functions.len() { writeln!(w, ",")?; } else { writeln!(w)?; }
    }
    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn find_lib() -> Option<String> {
    for entry in fs::read_dir("target/release").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".rlib") || name.ends_with(".so") || name.ends_with(".a") {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }
    None
}

fn parse_size_output(lib: &str) -> (u64, u64, u64) {
    let output = Command::new("size").arg(lib)
        .stdout(Stdio::piped()).stderr(Stdio::null())
        .output().ok();
    match output {
        Some(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            // Last line has: text data bss ...
            if let Some(line) = text.lines().last() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    return (
                        parts[0].parse().unwrap_or(0),
                        parts[1].parse().unwrap_or(0),
                        parts[2].parse().unwrap_or(0),
                    );
                }
            }
            (0, 0, 0)
        }
        _ => (0, 0, 0),
    }
}

fn extract_str_from(chunk: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    chunk.find(&needle).and_then(|i| {
        let rest = &chunk[i + needle.len()..];
        rest.find(':').and_then(|c| {
            let after = rest[c + 1..].trim_start();
            if after.starts_with('"') {
                let mut end = 1;
                let bytes = after.as_bytes();
                while end < bytes.len() {
                    if bytes[end] == b'"' && (end == 1 || bytes[end - 1] != b'\\') { break }
                    end += 1;
                }
                Some(after[1..end].to_string())
            } else { None }
        })
    }).unwrap_or_default()
}

fn extract_num_from(chunk: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    chunk.find(&needle).and_then(|i| {
        let rest = &chunk[i + needle.len()..];
        rest.find(':').map(|c| {
            let after = rest[c + 1..].trim_start();
            let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
            after[..end].parse().unwrap_or(0)
        })
    }).unwrap_or(0)
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len()+2); o.push('"');
    for c in s.chars() { match c { '"'=>o.push_str("\\\""), '\\'=>o.push_str("\\\\"),
        '\n'=>o.push_str("\\n"), '\r'=>o.push_str("\\r"), '\t'=>o.push_str("\\t"),
        c if (c as u32)<0x20 => o.push_str(&format!("\\u{:04x}",c as u32)), _=>o.push(c) } }
    o.push('"'); o
}
fn iso8601_now() -> String {
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    let s = dur.as_secs(); let (h,m,sc)=((s%86400)/3600,(s%3600)/60,s%60);
    let mut y=1970i64; let mut r=(s/86400) as i64;
    loop { let yd=if y%4==0&&(y%100!=0||y%400==0){366}else{365}; if r<yd{break} r-=yd; y+=1; }
    let lp=y%4==0&&(y%100!=0||y%400==0);
    let md=[31,if lp{29}else{28},31,30,31,30,31,31,30,31,30,31];
    let mut mo=0; for &d in &md { if r<d{break} r-=d; mo+=1; }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",y,mo+1,r+1,h,m,sc)
}
