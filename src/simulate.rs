//! Load-order and bytes-moved simulation for ThinTensor archives.

use crate::archive::Archive;
use crate::plan::{Plan, PlanOptions, build_plan};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize)]
pub struct LoadSimulation {
    pub backend: String,
    pub ctx_tokens: u64,
    pub batch_size: u64,
    pub plan: Plan,
    pub stage_count: usize,
    pub resident_pages: usize,
    pub streamed_pages: usize,
    pub evicted_pages: Vec<String>,
    pub initial_archive_read_bytes: u64,
    pub resident_read_groups: usize,
    pub resident_prefetch_io_ops_initial: usize,
    pub max_resident_group_pages: usize,
    pub max_resident_group_bytes: u64,
    pub streamed_weight_bytes_per_token: u64,
    pub kv_write_bytes_per_token: u64,
    pub estimated_bytes_moved_per_token: u64,
    pub peak_memory_bytes: u64,
    pub prefetch_pages: usize,
    pub stream_read_groups: usize,
    pub prefetch_io_ops_per_token: usize,
    pub max_stream_group_pages: usize,
    pub max_stream_group_bytes: u64,
    pub recommended_staging_bytes: u64,
    pub execution_page_refs: usize,
    pub sequential_transitions: usize,
    pub nonsequential_transitions: usize,
    pub backward_jumps: usize,
    pub sequential_read_ratio: f64,
    pub stage_loads: Vec<StageLoad>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageLoad {
    pub stage: String,
    pub page_refs: usize,
    pub resident_refs: usize,
    pub streamed_refs: usize,
    pub raw_bytes: u64,
    pub streamed_bytes: u64,
}

pub fn simulate_load(
    archive: &Archive,
    options: PlanOptions,
    prefetch_pages: usize,
) -> LoadSimulation {
    let plan = build_plan(archive.manifest(), options.clone());
    let resident_set: BTreeSet<_> = plan.resident.iter().map(String::as_str).collect();
    let page_bytes: BTreeMap<_, _> = archive
        .records()
        .iter()
        .map(|record| (record.page_id.as_str(), record.raw_size))
        .collect();
    let record_order: BTreeMap<_, _> = archive
        .records()
        .iter()
        .enumerate()
        .map(|(index, record)| (record.page_id.as_str(), index))
        .collect();

    let mut evicted = BTreeSet::new();
    let mut stage_loads = Vec::with_capacity(archive.manifest().execution_tape.len());
    let mut execution_positions = Vec::new();
    let mut streamed_positions = Vec::new();
    let mut resident_positions = Vec::new();
    let mut streamed_weight_bytes_per_token = 0_u64;
    let mut initial_archive_read_bytes = 0_u64;

    for page_id in &plan.resident {
        let bytes = page_bytes
            .get(page_id.as_str())
            .copied()
            .unwrap_or_default();
        initial_archive_read_bytes = initial_archive_read_bytes.saturating_add(bytes);
        if let Some(position) = record_order.get(page_id.as_str()) {
            resident_positions.push((*position, bytes));
        }
    }

    for stage in &archive.manifest().execution_tape {
        let mut resident_refs = 0_usize;
        let mut streamed_refs = 0_usize;
        let mut raw_bytes = 0_u64;
        let mut streamed_bytes = 0_u64;

        for page_id in &stage.page_refs {
            let bytes = page_bytes
                .get(page_id.as_str())
                .copied()
                .unwrap_or_default();
            raw_bytes = raw_bytes.saturating_add(bytes);
            if let Some(position) = record_order.get(page_id.as_str()) {
                execution_positions.push(*position);
            }

            if resident_set.contains(page_id.as_str()) {
                resident_refs += 1;
            } else {
                streamed_refs += 1;
                streamed_bytes = streamed_bytes.saturating_add(bytes);
                evicted.insert(page_id.clone());
                if let Some(position) = record_order.get(page_id.as_str()) {
                    streamed_positions.push((*position, bytes));
                }
            }
        }

        streamed_weight_bytes_per_token =
            streamed_weight_bytes_per_token.saturating_add(streamed_bytes);
        stage_loads.push(StageLoad {
            stage: stage.stage.clone(),
            page_refs: stage.page_refs.len(),
            resident_refs,
            streamed_refs,
            raw_bytes,
            streamed_bytes,
        });
    }

    let locality = locality_stats(&execution_positions);
    resident_positions.sort_by_key(|(position, _)| *position);
    let resident_groups = stream_group_stats(&resident_positions, prefetch_pages.max(1));
    let stream_groups = stream_group_stats(&streamed_positions, prefetch_pages.max(1));
    let kv_write_bytes_per_token = kv_write_bytes_per_token(&plan);
    let estimated_bytes_moved_per_token = streamed_weight_bytes_per_token
        .saturating_add(kv_write_bytes_per_token)
        .saturating_add(plan.scratch_bytes);

    LoadSimulation {
        backend: options.backend,
        ctx_tokens: options.ctx_tokens,
        batch_size: options.batch_size.max(1),
        resident_pages: plan.resident.len(),
        streamed_pages: evicted.len(),
        evicted_pages: evicted.into_iter().collect(),
        initial_archive_read_bytes,
        resident_read_groups: resident_groups.groups,
        resident_prefetch_io_ops_initial: resident_groups.prefetch_io_ops,
        max_resident_group_pages: resident_groups.max_pages,
        max_resident_group_bytes: resident_groups.max_bytes,
        streamed_weight_bytes_per_token,
        kv_write_bytes_per_token,
        estimated_bytes_moved_per_token,
        peak_memory_bytes: plan.total_bytes,
        prefetch_pages: prefetch_pages.max(1),
        stream_read_groups: stream_groups.groups,
        prefetch_io_ops_per_token: stream_groups.prefetch_io_ops,
        max_stream_group_pages: stream_groups.max_pages,
        max_stream_group_bytes: stream_groups.max_bytes,
        recommended_staging_bytes: stream_groups.recommended_staging_bytes,
        execution_page_refs: execution_positions.len(),
        sequential_transitions: locality.sequential,
        nonsequential_transitions: locality.nonsequential,
        backward_jumps: locality.backward,
        sequential_read_ratio: locality.ratio,
        stage_count: stage_loads.len(),
        stage_loads,
        plan,
    }
}

fn kv_write_bytes_per_token(plan: &Plan) -> u64 {
    if plan.ctx_tokens == 0 {
        return 0;
    }
    plan.kv_cache_bytes / plan.ctx_tokens
}

#[derive(Debug, Clone, Copy)]
struct LocalityStats {
    sequential: usize,
    nonsequential: usize,
    backward: usize,
    ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct StreamGroupStats {
    groups: usize,
    prefetch_io_ops: usize,
    max_pages: usize,
    max_bytes: u64,
    recommended_staging_bytes: u64,
}

fn stream_group_stats(streamed: &[(usize, u64)], prefetch_pages: usize) -> StreamGroupStats {
    if streamed.is_empty() {
        return StreamGroupStats {
            groups: 0,
            prefetch_io_ops: 0,
            max_pages: 0,
            max_bytes: 0,
            recommended_staging_bytes: 0,
        };
    }

    let mut groups = Vec::new();
    let mut current_pages = 0_usize;
    let mut current_bytes = 0_u64;
    let mut previous_position = None;

    for (position, bytes) in streamed {
        let contiguous = previous_position
            .map(|previous| *position == previous + 1)
            .unwrap_or(true);
        if !contiguous && current_pages > 0 {
            groups.push((current_pages, current_bytes));
            current_pages = 0;
            current_bytes = 0;
        }
        current_pages += 1;
        current_bytes = current_bytes.saturating_add(*bytes);
        previous_position = Some(*position);
    }
    if current_pages > 0 {
        groups.push((current_pages, current_bytes));
    }

    let mut prefetch_io_ops = 0_usize;
    let mut max_pages = 0_usize;
    let mut max_bytes = 0_u64;
    for (pages, bytes) in &groups {
        prefetch_io_ops += pages.div_ceil(prefetch_pages);
        max_pages = max_pages.max(*pages);
        max_bytes = max_bytes.max(*bytes);
    }

    StreamGroupStats {
        groups: groups.len(),
        prefetch_io_ops,
        max_pages,
        max_bytes,
        recommended_staging_bytes: recommended_staging_bytes(streamed, prefetch_pages),
    }
}

fn recommended_staging_bytes(streamed: &[(usize, u64)], prefetch_pages: usize) -> u64 {
    streamed
        .windows(prefetch_pages)
        .map(|window| {
            window
                .iter()
                .map(|(_, bytes)| *bytes)
                .fold(0_u64, u64::saturating_add)
        })
        .max()
        .unwrap_or_else(|| streamed.iter().map(|(_, bytes)| *bytes).sum())
}

fn locality_stats(positions: &[usize]) -> LocalityStats {
    let mut sequential = 0_usize;
    let mut nonsequential = 0_usize;
    let mut backward = 0_usize;

    for pair in positions.windows(2) {
        let [left, right] = [pair[0], pair[1]];
        if right == left + 1 {
            sequential += 1;
        } else {
            nonsequential += 1;
            if right < left {
                backward += 1;
            }
        }
    }

    let total = sequential + nonsequential;
    let ratio = if total == 0 {
        1.0
    } else {
        sequential as f64 / total as f64
    };

    LocalityStats {
        sequential,
        nonsequential,
        backward,
        ratio,
    }
}
