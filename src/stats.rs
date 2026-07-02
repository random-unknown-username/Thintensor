//! Archive introspection used by `inspect`, `stats`, and optimization passes.

use crate::archive::Archive;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize)]
pub struct ArchiveStats {
    pub archive_bytes: u64,
    pub manifest_bytes: u64,
    pub page_table_records: u64,
    pub data_offset: u64,
    pub page_count: usize,
    pub total_raw_bytes: u64,
    pub total_stored_bytes: u64,
    pub model: ModelStats,
    pub largest_pages: Vec<PageStats>,
    pub per_layer: Vec<LayerStats>,
    pub per_op: Vec<BucketStats>,
    pub by_kind: Vec<BucketStats>,
    pub by_dtype: Vec<BucketStats>,
    pub by_backend_layout: Vec<BucketStats>,
    pub execution: ExecutionStats,
    pub unknown_op_pages: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStats {
    pub arch: String,
    pub raw_arch: Option<String>,
    pub layers: u32,
    pub hidden_size: u64,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: Option<u64>,
    pub dtype: String,
    pub source_dtype: Option<String>,
    pub vocab_size: Option<u64>,
    pub tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PageStats {
    pub id: String,
    pub kind: String,
    pub layer: Option<u32>,
    pub op: String,
    pub dtype: String,
    pub backend_layout: String,
    pub raw_bytes: u64,
    pub stored_bytes: u64,
    pub referenced_stages: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerStats {
    pub layer: u32,
    pub pages: usize,
    pub raw_bytes: u64,
    pub stored_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BucketStats {
    pub name: String,
    pub pages: usize,
    pub raw_bytes: u64,
    pub stored_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionStats {
    pub stage_count: usize,
    pub page_ref_count: usize,
    pub referenced_page_count: usize,
    pub empty_stage_count: usize,
    pub unreferenced_pages: Vec<String>,
    pub duplicate_page_refs: Vec<DuplicateRefStats>,
    pub first_stages: Vec<StageStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DuplicateRefStats {
    pub page_id: String,
    pub refs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageStats {
    pub stage: String,
    pub page_refs: usize,
    pub raw_bytes: u64,
    pub stored_bytes: u64,
}

pub fn build_stats(archive: &Archive) -> ArchiveStats {
    let manifest = archive.manifest();
    let record_bytes: BTreeMap<_, _> = archive
        .records()
        .iter()
        .map(|record| {
            (
                record.page_id.as_str(),
                (record.raw_size, record.stored_size),
            )
        })
        .collect();
    let mut stage_ref_counts = BTreeMap::<&str, usize>::new();
    let mut referenced = BTreeSet::<&str>::new();

    for stage in &manifest.execution_tape {
        for page_id in &stage.page_refs {
            referenced.insert(page_id);
            *stage_ref_counts.entry(page_id).or_default() += 1;
        }
    }

    let mut page_stats = Vec::with_capacity(manifest.pages.len());
    let total_raw_bytes = archive.records().iter().map(|record| record.raw_size).sum();
    let total_stored_bytes = archive
        .records()
        .iter()
        .map(|record| record.stored_size)
        .sum();
    let mut unknown_op_pages = Vec::new();

    for page in &manifest.pages {
        if page.kind == "fused_physical" {
            continue;
        }
        let (raw_bytes, stored_bytes) = record_bytes
            .get(page.id.as_str())
            .copied()
            .unwrap_or((page.size, page.size));
        if page.op == "unknown" {
            unknown_op_pages.push(page.id.clone());
        }
        page_stats.push(PageStats {
            id: page.id.clone(),
            kind: page.kind.clone(),
            layer: page.layer,
            op: page.op.clone(),
            dtype: page.dtype.clone(),
            backend_layout: page.backend_layout.clone(),
            raw_bytes,
            stored_bytes,
            referenced_stages: stage_ref_counts
                .get(page.id.as_str())
                .copied()
                .unwrap_or_default(),
        });
    }

    let mut largest_pages = page_stats.clone();
    sort_pages_by_size(&mut largest_pages);
    largest_pages.truncate(16);

    ArchiveStats {
        archive_bytes: archive.file_len(),
        manifest_bytes: archive.header().manifest_len,
        page_table_records: archive.header().page_count,
        data_offset: archive.header().data_off,
        page_count: page_stats.len(),
        total_raw_bytes,
        total_stored_bytes,
        model: ModelStats {
            arch: manifest.model.arch.clone(),
            raw_arch: manifest.model.raw_arch.clone(),
            layers: manifest.model.layers,
            hidden_size: manifest.model.hidden_size,
            heads: manifest.model.heads,
            kv_heads: manifest.model.kv_heads,
            head_dim: manifest.model.head_dim,
            dtype: manifest.model.dtype.clone(),
            source_dtype: manifest.model.source_dtype.clone(),
            vocab_size: manifest.model.vocab_size,
            tie_word_embeddings: manifest.model.tie_word_embeddings,
        },
        per_layer: per_layer(&page_stats),
        per_op: buckets(&page_stats, |page| page.op.as_str()),
        by_kind: buckets(&page_stats, |page| page.kind.as_str()),
        by_dtype: buckets(&page_stats, |page| page.dtype.as_str()),
        by_backend_layout: buckets(&page_stats, |page| page.backend_layout.as_str()),
        execution: execution_stats(archive, &record_bytes, referenced, stage_ref_counts),
        largest_pages,
        unknown_op_pages,
    }
}

fn per_layer(pages: &[PageStats]) -> Vec<LayerStats> {
    let mut layers = BTreeMap::<u32, LayerStats>::new();
    for page in pages {
        let Some(layer) = page.layer else {
            continue;
        };
        let entry = layers.entry(layer).or_insert_with(|| LayerStats {
            layer,
            pages: 0,
            raw_bytes: 0,
            stored_bytes: 0,
        });
        entry.pages += 1;
        entry.raw_bytes = entry.raw_bytes.saturating_add(page.raw_bytes);
        entry.stored_bytes = entry.stored_bytes.saturating_add(page.stored_bytes);
    }

    layers.into_values().collect()
}

fn buckets<'a>(pages: &'a [PageStats], key: impl Fn(&'a PageStats) -> &'a str) -> Vec<BucketStats> {
    let mut buckets = BTreeMap::<String, BucketStats>::new();
    for page in pages {
        let name = key(page).to_string();
        let entry = buckets.entry(name.clone()).or_insert_with(|| BucketStats {
            name,
            pages: 0,
            raw_bytes: 0,
            stored_bytes: 0,
        });
        entry.pages += 1;
        entry.raw_bytes = entry.raw_bytes.saturating_add(page.raw_bytes);
        entry.stored_bytes = entry.stored_bytes.saturating_add(page.stored_bytes);
    }

    let mut items: Vec<_> = buckets.into_values().collect();
    items.sort_by(|left, right| {
        right
            .raw_bytes
            .cmp(&left.raw_bytes)
            .then(left.name.cmp(&right.name))
    });
    items
}

fn execution_stats<'a>(
    archive: &'a Archive,
    record_bytes: &BTreeMap<&'a str, (u64, u64)>,
    referenced: BTreeSet<&'a str>,
    stage_ref_counts: BTreeMap<&'a str, usize>,
) -> ExecutionStats {
    let manifest = archive.manifest();
    let unreferenced_pages = manifest
        .pages
        .iter()
        .filter(|page| !referenced.contains(page.id.as_str()))
        .map(|page| page.id.clone())
        .collect();
    let duplicate_page_refs = stage_ref_counts
        .iter()
        .filter(|(_, refs)| **refs > 1)
        .map(|(page_id, refs)| DuplicateRefStats {
            page_id: (*page_id).to_string(),
            refs: *refs,
        })
        .collect();
    let first_stages = manifest
        .execution_tape
        .iter()
        .take(16)
        .map(|stage| {
            let mut raw_bytes = 0_u64;
            let mut stored_bytes = 0_u64;
            for page_id in &stage.page_refs {
                if let Some((raw, stored)) = record_bytes.get(page_id.as_str()) {
                    raw_bytes = raw_bytes.saturating_add(*raw);
                    stored_bytes = stored_bytes.saturating_add(*stored);
                }
            }
            StageStats {
                stage: stage.stage.clone(),
                page_refs: stage.page_refs.len(),
                raw_bytes,
                stored_bytes,
            }
        })
        .collect();

    ExecutionStats {
        stage_count: manifest.execution_tape.len(),
        page_ref_count: stage_ref_counts.values().sum(),
        referenced_page_count: referenced.len(),
        empty_stage_count: manifest
            .execution_tape
            .iter()
            .filter(|stage| stage.page_refs.is_empty())
            .count(),
        unreferenced_pages,
        duplicate_page_refs,
        first_stages,
    }
}

fn sort_pages_by_size(pages: &mut [PageStats]) {
    pages.sort_by(|left, right| {
        right
            .raw_bytes
            .cmp(&left.raw_bytes)
            .then(left.id.cmp(&right.id))
    });
}
