//! Wrap the zeroize-report.json with sha/timestamp. Output JSON.
//!
//! Usage: zeroize-report <sha> [zeroize-report.json]
//!
//! Compile: `rustc -O zeroize-report.rs -o zeroize-report`

use std::env;
use std::fs;
use std::io::{self, Write};
use std::time::SystemTime;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 { eprintln!("usage: zeroize-report <sha> [report.json]"); std::process::exit(1); }
    let sha = &args[1];
    let path = args.get(2).map(|s| s.as_str()).unwrap_or("zeroize-report.json");

    let (total_bytes, nonzero, status) = fs::read_to_string(path).ok()
        .map(|c| {
            let tb = extract_num(&c, "total_bytes");
            let nz = extract_num(&c, "nonzero_after_drop");
            let st = extract_str(&c, "status");
            (tb, nz, st)
        })
        .unwrap_or((0, 0, "skip".to_string()));

    let out = io::stdout();
    let mut w = io::BufWriter::new(out.lock());
    writeln!(w, "{{")?;
    writeln!(w, "  \"sha\": {},", json_str(sha))?;
    writeln!(w, "  \"updated_at\": {},", json_str(&iso8601_now()))?;
    writeln!(w, "  \"total_bytes\": {},", total_bytes)?;
    writeln!(w, "  \"nonzero_after_drop\": {},", nonzero)?;
    writeln!(w, "  \"status\": {}", json_str(&status))?;
    writeln!(w, "}}")?;
    Ok(())
}

fn extract_num(json: &str, key: &str) -> u64 {
    let needle = format!("\"{}\"", key);
    json.find(&needle).and_then(|i| {
        let rest = &json[i + needle.len()..];
        rest.find(':').map(|c| {
            let after = rest[c + 1..].trim_start();
            let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
            after[..end].parse().unwrap_or(0)
        })
    }).unwrap_or(0)
}

fn extract_str(json: &str, key: &str) -> String {
    let needle = format!("\"{}\"", key);
    json.find(&needle).and_then(|i| {
        let rest = &json[i + needle.len()..];
        rest.find(':').and_then(|c| {
            let after = rest[c + 1..].trim_start();
            if after.starts_with('"') {
                let mut end = 1;
                let bytes = after.as_bytes();
                while end < bytes.len() {
                    if bytes[end] == b'"' && bytes[end - 1] != b'\\' { break }
                    end += 1;
                }
                Some(after[1..end].to_string())
            } else { None }
        })
    }).unwrap_or_else(|| "skip".to_string())
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
