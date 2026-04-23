//! Run `cargo check` with the MSRV toolchain and output JSON.
//!
//! Usage: msrv-report <sha> <msrv>
//!
//! Compile: `rustc -O msrv-report.rs -o msrv-report`

use std::env;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 { eprintln!("usage: msrv-report <sha> <msrv>"); std::process::exit(1); }
    let sha = &args[1];
    let msrv = &args[2];

    let output = Command::new("cargo").args(["check"])
        .stdout(Stdio::piped()).stderr(Stdio::piped()).output()
        .expect("failed to run cargo check");
    let text = String::from_utf8_lossy(&output.stderr);
    let errors = text.lines().filter(|l| l.contains("error")).count();
    let status = if output.status.success() { "pass" } else { "fail" };

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"msrv\": {},", json_str(msrv))?;
    writeln!(w, "  \"status\": {},", json_str(status))?;
    writeln!(w, "  \"errors\": {}", errors)?;
    writeln!(w, "}}")?;
    Ok(())
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
