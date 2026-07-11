//! Target-VRAM profile generation for runtime residency decisions.

use crate::archive::Archive;
use crate::plan::{Plan, PlanOptions, PlanStatus, WeightResidency, build_plan};
use crate::stats::{BucketStats, LayerStats, PageStats, build_stats};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct ProfileOptions {
    pub backend: String,
    pub target_vram_bytes: u64,
    pub ctx_tokens: u64,
    pub batch_size: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeProfile {
    pub backend: String,
    pub target_vram_bytes: u64,
    pub ctx_tokens: u64,
    pub batch_size: u64,
    pub status: PlanStatus,
    pub selected_candidate: Option<String>,
    pub recommended_layout: String,
    pub kv_codec_recommendation: String,
    pub always_hot_pages: Vec<String>,
    pub streamable_pages: Vec<String>,
    pub cpu_offload_candidates: Vec<PageStats>,
    pub biggest_memory_offenders: Vec<PageStats>,
    pub per_layer: Vec<LayerStats>,
    pub per_op: Vec<BucketStats>,
    pub candidates: Vec<ProfileCandidate>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileCandidate {
    pub name: String,
    pub status: PlanStatus,
    pub total_bytes: u64,
    pub usable_vram_bytes: u64,
    pub resident_weight_bytes: u64,
    pub streamed_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub metadata_bytes: u64,
    pub kv_dtype: String,
    pub weight_residency: WeightResidency,
    pub offload_layers: u32,
    pub suggestions: Vec<String>,
}

pub fn build_profile(archive: &Archive, options: ProfileOptions) -> RuntimeProfile {
    let stats = build_stats(archive);
    let candidates = candidate_plans(archive, &options);
    let selected_index = candidates
        .iter()
        .position(|(_, plan)| plan.status == PlanStatus::Ok)
        .or_else(|| {
            candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, plan))| plan.total_bytes)
                .map(|(index, _)| index)
        });
    let selected_plan = selected_index.map(|index| &candidates[index].1);
    let selected_name = selected_index.map(|index| candidates[index].0.clone());
    let status = selected_plan
        .map(|plan| plan.status)
        .unwrap_or(PlanStatus::NotEnoughVram);
    let kv_codec_recommendation = selected_plan
        .map(|plan| plan.kv_dtype.clone())
        .unwrap_or_else(|| "q2".to_string());
    let recommended_layout = selected_plan
        .map(recommended_layout)
        .unwrap_or("hot_stream_v1")
        .to_string();

    RuntimeProfile {
        backend: options.backend,
        target_vram_bytes: options.target_vram_bytes,
        ctx_tokens: options.ctx_tokens,
        batch_size: options.batch_size.max(1),
        status,
        selected_candidate: selected_name,
        recommended_layout,
        kv_codec_recommendation,
        always_hot_pages: always_hot_pages(archive),
        streamable_pages: streamable_pages(archive),
        cpu_offload_candidates: cpu_offload_candidates(&stats.largest_pages),
        biggest_memory_offenders: stats.largest_pages.iter().take(16).cloned().collect(),
        per_layer: stats.per_layer.clone(),
        per_op: stats.per_op.iter().take(16).cloned().collect(),
        candidates: candidates
            .iter()
            .map(|(name, plan)| summarize_candidate(name.clone(), plan))
            .collect(),
        notes: profile_notes(&stats, selected_plan),
    }
}

fn candidate_plans(archive: &Archive, options: &ProfileOptions) -> Vec<(String, Plan)> {
    let layers = archive.manifest().model.layers;
    let quarter = (layers / 4).max(1);
    let half = (layers / 2).max(1);
    let candidates = [
        ("all_weights_kv_q4", WeightResidency::All, 0, "q4"),
        ("all_weights_kv_q2", WeightResidency::All, 0, "q2"),
        (
            "offload_middle_quarter_kv_q4",
            WeightResidency::OffloadMiddleOut,
            quarter,
            "q4",
        ),
        (
            "offload_middle_half_kv_q4",
            WeightResidency::OffloadMiddleOut,
            half,
            "q4",
        ),
        ("stream_weights_kv_q4", WeightResidency::Stream, 0, "q4"),
        ("stream_weights_kv_q2", WeightResidency::Stream, 0, "q2"),
    ];

    candidates
        .into_iter()
        .map(|(name, residency, offload_layers, kv_dtype)| {
            let plan = build_plan(
                archive.manifest(),
                PlanOptions {
                    backend: options.backend.clone(),
                    vram_bytes: options.target_vram_bytes,
                    ctx_tokens: options.ctx_tokens,
                    batch_size: options.batch_size,
                    kv_dtype: kv_dtype.to_string(),
                    weight_residency: residency,
                    offload_layers,
                    gpu_fraction: 1.0,
                },
            );
            (name.to_string(), plan)
        })
        .collect()
}

fn summarize_candidate(name: String, plan: &Plan) -> ProfileCandidate {
    ProfileCandidate {
        name,
        status: plan.status,
        total_bytes: plan.total_bytes,
        usable_vram_bytes: plan.usable_vram_bytes,
        resident_weight_bytes: plan.weights_bytes,
        streamed_weight_bytes: plan.streamed_weight_bytes,
        kv_cache_bytes: plan.kv_cache_bytes,
        scratch_bytes: plan.scratch_bytes,
        metadata_bytes: plan.metadata_bytes,
        kv_dtype: plan.kv_dtype.clone(),
        weight_residency: plan.weight_residency,
        offload_layers: plan.offload_layers,
        suggestions: plan.suggestions.clone(),
    }
}

fn recommended_layout(plan: &Plan) -> &'static str {
    if plan.streamed_weight_bytes > 0 || plan.weight_residency != WeightResidency::All {
        "hot_stream_v1"
    } else {
        "execution_ordered_v1"
    }
}

fn always_hot_pages(archive: &Archive) -> Vec<String> {
    archive
        .manifest()
        .pages
        .iter()
        .filter(|page| {
            page.layer.is_none() || matches!(page.kind.as_str(), "embedding" | "lm_head")
        })
        .map(|page| page.id.clone())
        .collect()
}

fn streamable_pages(archive: &Archive) -> Vec<String> {
    let manifest = archive.manifest();
    let layered: BTreeSet<&str> = manifest
        .pages
        .iter()
        .filter(|page| page.layer.is_some())
        .map(|page| page.id.as_str())
        .collect();
    let mut seen = BTreeSet::new();
    let mut pages = Vec::new();
    for stage in &manifest.execution_tape {
        for page_id in &stage.page_refs {
            if layered.contains(page_id.as_str()) && seen.insert(page_id.as_str()) {
                pages.push(page_id.clone());
            }
        }
    }
    pages
}

fn cpu_offload_candidates(largest_pages: &[PageStats]) -> Vec<PageStats> {
    largest_pages
        .iter()
        .filter(|page| page.layer.is_some() && is_big_weight(page))
        .take(32)
        .cloned()
        .collect()
}

fn is_big_weight(page: &PageStats) -> bool {
    let op = page.op.as_str();
    op.starts_with("attn_") && op.ends_with("_proj")
        || op.starts_with("mlp_") && op.ends_with("_proj")
        || matches!(op, "moe_gate_up" | "moe_down")
        || op.starts_with("linear_attn_")
            && matches!(
                op,
                "linear_attn_input_projection"
                    | "linear_attn_depthwise_conv"
                    | "linear_attn_output_projection"
            )
        || matches!(op, "per_layer_model_projection" | "per_layer_projection")
}

fn profile_notes(stats: &crate::stats::ArchiveStats, selected_plan: Option<&Plan>) -> Vec<String> {
    let mut notes = Vec::new();
    if let Some(plan) = selected_plan {
        if plan.shared_weight_savings_bytes > 0 {
            notes.push(format!(
                "exact duplicate/shared tensors save {} runtime bytes",
                plan.shared_weight_savings_bytes
            ));
        }
        if plan.streamed_weight_bytes > 0 {
            notes.push(format!(
                "selected plan streams or offloads {} weight bytes",
                plan.streamed_weight_bytes
            ));
        }
        if plan.status == PlanStatus::NotEnoughVram {
            notes.push("no candidate fits the target VRAM budget".to_string());
        }
    }
    if stats.execution.unreferenced_pages.is_empty() {
        notes.push("execution tape references every manifest page".to_string());
    }
    notes
}
