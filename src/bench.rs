//! Lightweight benchmark probes for load, verify, and planning paths.

use crate::archive::Archive;
use crate::plan::{Plan, PlanOptions, build_plan};
use crate::verify::verify_archive;
use anyhow::{Context, Result, anyhow, bail};
use glob::glob;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::Serialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Serialize)]
pub struct BenchLoadResult {
    pub input: String,
    pub kind: BenchKind,
    pub elapsed_ms: f64,
    pub files: usize,
    pub tensors_or_pages: usize,
    pub bytes: u64,
    pub largest: Vec<BenchItem>,
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct BenchPlanResult {
    pub input: String,
    pub open_ms: f64,
    pub verify_ms: f64,
    pub plan_ms: f64,
    pub page_count: usize,
    pub archive_bytes: u64,
    pub plan: Plan,
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchKind {
    HuggingFace,
    ThinArchive,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchItem {
    pub id: String,
    pub bytes: u64,
}

pub fn bench_load(path: &Path) -> Result<BenchLoadResult> {
    if path.is_dir() {
        bench_hf_load(path)
    } else {
        bench_thin_load(path)
    }
}

pub fn bench_plan(path: &Path, options: PlanOptions) -> Result<BenchPlanResult> {
    let start = Instant::now();
    let archive = Archive::open(path)?;
    let open_ms = elapsed_ms(start);

    let start = Instant::now();
    let report = verify_archive(&archive)?;
    let verify_ms = elapsed_ms(start);
    if !report.is_ok() {
        bail!("archive verification failed:\n{}", report.errors.join("\n"));
    }

    let start = Instant::now();
    let plan = build_plan(archive.manifest(), options);
    let plan_ms = elapsed_ms(start);

    Ok(BenchPlanResult {
        input: path.display().to_string(),
        open_ms,
        verify_ms,
        plan_ms,
        page_count: archive.records().len(),
        archive_bytes: archive.file_len(),
        plan,
        rss_bytes: proc_status_bytes("VmRSS"),
        peak_rss_bytes: proc_status_bytes("VmHWM"),
    })
}

fn bench_hf_load(path: &Path) -> Result<BenchLoadResult> {
    let start = Instant::now();
    let safetensors = collect_safetensors(path)?;
    if safetensors.is_empty() {
        bail!("no safetensors files found in {}", path.display());
    }

    let mut tensors = 0_usize;
    let mut bytes = 0_u64;
    let mut largest = Vec::new();

    for file_path in &safetensors {
        let file =
            File::open(file_path).with_context(|| format!("open {}", file_path.display()))?;
        // SAFETY: read-only mapping; benchmark never mutates the mapped file.
        let mmap =
            unsafe { Mmap::map(&file).with_context(|| format!("mmap {}", file_path.display()))? };
        let parsed = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("read {}", file_path.display()))?;

        for (name, tensor) in parsed.iter() {
            let len = tensor.data().len() as u64;
            tensors += 1;
            bytes += len;
            largest.push(BenchItem {
                id: name.to_string(),
                bytes: len,
            });
        }
    }

    largest.sort_by(|left, right| right.bytes.cmp(&left.bytes).then(left.id.cmp(&right.id)));
    largest.truncate(8);

    Ok(BenchLoadResult {
        input: path.display().to_string(),
        kind: BenchKind::HuggingFace,
        elapsed_ms: elapsed_ms(start),
        files: safetensors.len(),
        tensors_or_pages: tensors,
        bytes,
        largest,
        rss_bytes: proc_status_bytes("VmRSS"),
        peak_rss_bytes: proc_status_bytes("VmHWM"),
    })
}

fn bench_thin_load(path: &Path) -> Result<BenchLoadResult> {
    let start = Instant::now();
    let archive = Archive::open(path)?;
    let elapsed_ms = elapsed_ms(start);

    let mut largest: Vec<_> = archive
        .records()
        .iter()
        .map(|record| BenchItem {
            id: record.page_id.clone(),
            bytes: record.stored_size,
        })
        .collect();
    largest.sort_by(|left, right| right.bytes.cmp(&left.bytes).then(left.id.cmp(&right.id)));
    largest.truncate(8);

    Ok(BenchLoadResult {
        input: path.display().to_string(),
        kind: BenchKind::ThinArchive,
        elapsed_ms,
        files: 1,
        tensors_or_pages: archive.records().len(),
        bytes: archive.file_len(),
        largest,
        rss_bytes: proc_status_bytes("VmRSS"),
        peak_rss_bytes: proc_status_bytes("VmHWM"),
    })
}

fn collect_safetensors(hf_dir: &Path) -> Result<Vec<PathBuf>> {
    let pattern = hf_dir.join("*.safetensors");
    let pattern = pattern
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8"))?;
    let mut paths = Vec::new();
    for entry in glob(pattern)? {
        paths.push(entry?);
    }
    paths.sort();
    Ok(paths)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn proc_status_bytes(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        let (name, rest) = line.split_once(':')?;
        if name != key {
            continue;
        }
        let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return kb.checked_mul(1024);
    }
    None
}
