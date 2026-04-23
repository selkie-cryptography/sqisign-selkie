//! Run ctgrind tests under Valgrind memcheck and output JSON.
//!
//! Usage: ctgrind-report <sha>
//!
//! Builds the ctgrind test, runs it under valgrind --tool=memcheck,
//! counts errors from XML output, and produces JSON.
//!
//! Compile: `rustc -O ctgrind-report.rs -o ctgrind-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 { eprintln!("usage: ctgrind-report <sha>"); std::process::exit(1); }
    let sha = &args[1];

    // Build the ctgrind test binary.
    eprintln!("[ctgrind-report] building...");
    let status = Command::new("cargo")
        .args(["test", "--test", "ctgrind", "--features", "expose-internals", "--no-run"])
        .status().expect("failed to build");
    if !status.success() {
        eprintln!("[ctgrind-report] build failed");
        std::process::exit(1);
    }

    // Find the test binary.
    let bin = find_binary("ctgrind");
    let bin = match bin {
        Some(b) => b,
        None => { eprintln!("[ctgrind-report] binary not found"); std::process::exit(1); }
    };

    // Run under valgrind.
    eprintln!("[ctgrind-report] running under valgrind...");
    let xml_file = "/tmp/ctgrind-valgrind.xml";
    let output = Command::new("valgrind")
        .args([
            "--tool=memcheck", "--error-exitcode=0",
            "--xml=yes", &format!("--xml-file={xml_file}"),
            &bin, "--test-threads=1",
        ])
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .output().expect("failed to run valgrind");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let tests = stdout.lines()
        .filter(|l| l.contains("test ") && (l.contains("... ok") || l.contains("... FAILED")))
        .count();

    // Count errors from XML.
    let errors = fs::read_to_string(xml_file).ok()
        .map(|xml| xml.matches("<error>").count())
        .unwrap_or(0);

    let status_str = if errors == 0 { "pass" } else { "fail" };

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"status\": {},", json_str(status_str))?;
    writeln!(w, "  \"errors\": {},", errors)?;
    writeln!(w, "  \"tests\": {}", tests)?;
    writeln!(w, "}}")?;
    Ok(())
}

fn find_binary(prefix: &str) -> Option<String> {
    for entry in fs::read_dir("target/debug/deps").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(prefix) && !name.contains('.') {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() && meta.len() > 0 {
                    return Some(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    None
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
