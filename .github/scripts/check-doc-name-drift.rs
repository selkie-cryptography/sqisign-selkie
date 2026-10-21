//! Flags code-formatted identifiers in the doc surfaces that no longer
//! exist anywhere under `src/`.
//!
//! Surfaces: `README.md`, `latex/sections/*.tex`, `.github/site/*.html`.
//! A candidate is a token inside Markdown backticks, HTML `<code>`
//! elements, or LaTeX `\texttt{}` that is shaped like a type name: a capitalized word
//! with at least one lowercase letter (`Curve`, `Fp2`, `TorsionBasis`).
//! Function and variable names are out of scope; the paper quotes C
//! reference symbols and listing locals too freely for that to be a
//! useful gate.  Paths (`a::B`) contribute their last segment;
//! generics, call parens, and macro bangs are stripped.  A candidate
//! passes if the name occurs as a whole word in some `.rs` file under
//! one of the source roots, or is listed in the allowlist.
//!
//! Usage: `check-doc-name-drift [--allow FILE] [--src DIR]...
//! [--surfaces readme,site,paper] [ROOT]`.  `--src` repeats; the
//! default roots are `src`, `tests`, `benches`, `examples`, `fuzz`.
//! CI checks the README and site against the current tree and the
//! paper against the current tree plus the `v2.0.1-final` tree, whose
//! implementation the history sections describe.  Exits 1 on drift.
//! Stdlib only, compiled with `rustc -O`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut allow_path = PathBuf::from(".github/doc-drift-allow.txt");
    let mut root = PathBuf::from(".");
    let mut src_dirs: Vec<PathBuf> = Vec::new();
    let mut surface_names = vec!["readme".to_owned(), "site".to_owned(), "paper".to_owned()];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--allow" => allow_path = PathBuf::from(args.next().expect("--allow needs a path")),
            "--src" => src_dirs.push(PathBuf::from(args.next().expect("--src needs a dir"))),
            "--surfaces" => {
                surface_names = args
                    .next()
                    .expect("--surfaces needs a list")
                    .split(',')
                    .map(str::to_owned)
                    .collect();
            }
            _ => root = PathBuf::from(a),
        }
    }
    if src_dirs.is_empty() {
        for d in ["src", "tests", "benches", "examples", "fuzz"] {
            src_dirs.push(root.join(d));
        }
    }

    let mut src_idents = HashSet::new();
    for dir in &src_dirs {
        src_idents.extend(source_identifiers(dir));
    }
    let allow = allowlist(&root.join(allow_path));

    let mut surfaces = Vec::new();
    for name in &surface_names {
        match name.as_str() {
            "readme" => surfaces.push(root.join("README.md")),
            "site" => surfaces.extend(files_with_ext(&root.join(".github/site"), "html")),
            "paper" => surfaces.extend(files_with_ext(&root.join("latex/sections"), "tex")),
            other => panic!("unknown surface {}", other),
        }
    }

    let mut drift = 0usize;
    for path in surfaces {
        let Ok(text) = fs::read_to_string(&path) else { continue };
        let kind = match path.extension().and_then(|e| e.to_str()) {
            Some("tex") => Kind::Tex,
            Some("html") => Kind::Html,
            _ => Kind::Markdown,
        };
        for (lineno, line) in text.lines().enumerate() {
            for span in code_spans(line, kind) {
                for token in candidates(&span) {
                    if !src_idents.contains(&token) && !allow.contains(&token) {
                        println!(
                            "{}:{}: `{}` not found in src/",
                            path.strip_prefix(&root).unwrap_or(&path).display(),
                            lineno + 1,
                            token
                        );
                        drift += 1;
                    }
                }
            }
        }
    }

    if drift > 0 {
        eprintln!("{drift} stale identifier(s); rename the doc or add to the allowlist");
        std::process::exit(1);
    }
    println!("doc surfaces name no missing identifiers");
}

/// Every `[A-Za-z_][A-Za-z0-9_]*` word in any `.rs` file under `dir`.
fn source_identifiers(dir: &Path) -> HashSet<String> {
    let mut set = HashSet::new();
    for file in files_with_ext(dir, "rs") {
        let Ok(text) = fs::read_to_string(&file) else { continue };
        for word in words(&text) {
            set.insert(word);
        }
    }
    set
}

fn allowlist(path: &Path) -> HashSet<String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

/// Recursively lists files with the given extension.
fn files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(files_with_ext(&path, ext));
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Splits text into identifier-shaped words.
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Doc surface syntax.
#[derive(Clone, Copy)]
enum Kind {
    Markdown,
    Html,
    Tex,
}

/// Code spans on one line: Markdown backtick spans, HTML `<code>`
/// bodies, or `\texttt{...}` bodies with `\_` unescaped.
fn code_spans(line: &str, kind: Kind) -> Vec<String> {
    match kind {
        Kind::Tex => delimited(line, "\\texttt{", "}")
            .into_iter()
            .map(|s| s.replace("\\_", "_"))
            .collect(),
        Kind::Html => delimited(line, "<code>", "</code>"),
        Kind::Markdown => {
            let mut spans = Vec::new();
            let mut parts = line.split('`');
            // Even-indexed parts are outside backticks.
            parts.next();
            while let Some(inside) = parts.next() {
                spans.push(inside.to_owned());
                parts.next();
            }
            spans
        }
    }
}

/// Bodies between `open` and the next `close` on one line.
fn delimited(line: &str, open: &str, close: &str) -> Vec<String> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(i) = rest.find(open) {
        let body = &rest[i + open.len()..];
        let Some(j) = body.find(close) else { break };
        spans.push(body[..j].to_owned());
        rest = &body[j + close.len()..];
    }
    spans
}

/// Type-shaped tokens worth checking inside one code span.
fn candidates(span: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in span.split(|c: char| c.is_whitespace() || "(),;=&[]|+*/!".contains(c)) {
        let raw = raw.rsplit("::").next().unwrap_or(raw);
        let raw = raw.split('<').next().unwrap_or(raw);
        let raw = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
        if looks_like_type_name(raw) {
            out.push(raw.to_owned());
        }
    }
    out
}

/// A capitalized word with a lowercase letter somewhere: `Curve`,
/// `Fp2`, `HnfLattice`.  Acronyms, constants, numbers, snake_case, and
/// file names do not qualify.
fn looks_like_type_name(s: &str) -> bool {
    s.len() >= 2
        && s.starts_with(|c: char| c.is_ascii_uppercase())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.chars().any(|c| c.is_ascii_lowercase())
}
