//! Runtime memory budget compiler.

use crate::manifest::{Manifest, PageSpec};
use serde::Serialize;
use std::collections::BTreeSet;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct PlanOptions {
    pub backend: String,
    pub vram_bytes: u64,
    pub ctx_tokens: u64,
    pub batch_size: u64,
    pub kv_dtype: String,
    pub weight_residency: WeightResidency,
    pub offload_layers: u32,
    pub gpu_fraction: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum WeightResidency {
    All,
    Stream,
    OffloadLastN,
    OffloadFirstN,
}

impl FromStr for WeightResidency {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "all" => Ok(Self::All),
            "stream" => Ok(Self::Stream),
            "offload-last-n" => Ok(Self::OffloadLastN),
            "offload-first-n" => Ok(Self::OffloadFirstN),
            _ => Err(format!("unknown weight residency mode {value}")),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub backend: String,
    pub vram_bytes: u64,
    pub usable_vram_bytes: u64,
    pub ctx_tokens: u64,
    pub batch_size: u64,
    pub kv_dtype: String,
    pub weight_residency: WeightResidency,
    pub offload_layers: u32,
    pub resident: Vec<String>,
    pub kv_policy: KvPolicy,
    pub scratch_bytes: u64,
    pub weights_bytes: u64,
    pub total_weight_bytes: u64,
    pub physical_weight_bytes: u64,
    pub shared_weight_savings_bytes: u64,
    pub streamed_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub metadata_bytes: u64,
    pub total_bytes: u64,
    pub status: PlanStatus,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KvPolicy {
    pub recent_tokens_high_precision: u64,
    pub old_tokens_codec: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PlanStatus {
    Ok,
    NotEnoughVram,
}

pub fn build_plan(manifest: &Manifest, options: PlanOptions) -> Plan {
    let batch_size = options.batch_size.max(1);
    let usable_vram_bytes = usable_vram_bytes(options.vram_bytes, options.gpu_fraction);
    let physical_weight_bytes = physical_weight_bytes(manifest, |_| true);
    let total_weight_bytes = unique_weight_bytes(manifest, |_| true);
    let shared_weight_savings_bytes = physical_weight_bytes.saturating_sub(total_weight_bytes);
    let resident = resident_pages(manifest, &options);
    let resident_set: BTreeSet<_> = resident.iter().map(String::as_str).collect();
    let weights_bytes =
        unique_weight_bytes(manifest, |page| resident_set.contains(page.id.as_str()));
    let streamed_weight_bytes = total_weight_bytes.saturating_sub(weights_bytes);
    let scratch_bytes = manifest
        .memory_plan
        .scratch_bytes
        .saturating_mul(batch_size);
    let kv_cache_bytes = estimate_kv_cache_bytes(manifest, &options, batch_size);
    let metadata_bytes = estimate_metadata_bytes(manifest);
    let total_bytes = weights_bytes
        .saturating_add(scratch_bytes)
        .saturating_add(kv_cache_bytes)
        .saturating_add(metadata_bytes);
    let status = if usable_vram_bytes < manifest.memory_plan.scratch_bytes
        || total_bytes > usable_vram_bytes
    {
        PlanStatus::NotEnoughVram
    } else {
        PlanStatus::Ok
    };
    let suggestions = suggestions(
        manifest,
        &options,
        weights_bytes,
        scratch_bytes,
        metadata_bytes,
        status,
        usable_vram_bytes,
    );

    Plan {
        backend: options.backend,
        vram_bytes: options.vram_bytes,
        usable_vram_bytes,
        ctx_tokens: options.ctx_tokens,
        batch_size,
        kv_dtype: options.kv_dtype.clone(),
        weight_residency: options.weight_residency,
        offload_layers: options.offload_layers,
        resident,
        kv_policy: KvPolicy {
            recent_tokens_high_precision: manifest
                .memory_plan
                .kv_cache
                .recent_tokens_high_precision,
            old_tokens_codec: options.kv_dtype,
        },
        scratch_bytes,
        weights_bytes,
        total_weight_bytes,
        physical_weight_bytes,
        shared_weight_savings_bytes,
        streamed_weight_bytes,
        kv_cache_bytes,
        metadata_bytes,
        total_bytes,
        status,
        suggestions,
    }
}

fn resident_pages(manifest: &Manifest, options: &PlanOptions) -> Vec<String> {
    manifest
        .pages
        .iter()
        .filter(|page| {
            is_resident_kind(page)
                && layout_matches(page, &options.backend)
                && page_is_resident(manifest, page, options)
        })
        .map(|page| page.id.clone())
        .collect()
}

fn layout_matches(page: &PageSpec, backend: &str) -> bool {
    page.backend_layout == "generic" || page.backend_layout == backend
}

fn page_is_resident(manifest: &Manifest, page: &PageSpec, options: &PlanOptions) -> bool {
    match options.weight_residency {
        WeightResidency::All => true,
        WeightResidency::Stream => page.layer.is_none(),
        WeightResidency::OffloadLastN => {
            let Some(layer) = page.layer else {
                return true;
            };
            let offload = options.offload_layers.min(manifest.model.layers);
            layer < manifest.model.layers.saturating_sub(offload)
        }
        WeightResidency::OffloadFirstN => {
            let Some(layer) = page.layer else {
                return true;
            };
            layer >= options.offload_layers.min(manifest.model.layers)
        }
    }
}

fn estimate_kv_cache_bytes(manifest: &Manifest, options: &PlanOptions, batch_size: u64) -> u64 {
    let model = &manifest.model;
    if model.heads == 0 {
        return 0;
    }

    let head_dim = model
        .head_dim
        .unwrap_or(model.hidden_size / model.heads as u64);
    let scalars_per_token = 2_u64 * model.layers as u64 * model.kv_heads as u64 * head_dim;
    let recent = options
        .ctx_tokens
        .min(manifest.memory_plan.kv_cache.recent_tokens_high_precision);
    let old = options.ctx_tokens.saturating_sub(recent);

    let recent_bytes = scalars_per_token as f64 * recent as f64 * 2.0;
    let old_bytes = scalars_per_token as f64 * old as f64 * codec_bytes(&options.kv_dtype);

    ((recent_bytes + old_bytes) * batch_size as f64).ceil() as u64
}

fn codec_bytes(codec: &str) -> f64 {
    match codec.to_ascii_lowercase().as_str() {
        "q2" => 0.25,
        "q3" => 0.375,
        "q4" => 0.5,
        "q5" => 0.625,
        "q6" => 0.75,
        "q8" | "fp8" => 1.0,
        "fp16" | "bf16" | "high_precision" => 2.0,
        "fp32" => 4.0,
        _ => 2.0,
    }
}

fn usable_vram_bytes(vram_bytes: u64, gpu_fraction: f64) -> u64 {
    if !gpu_fraction.is_finite() {
        return vram_bytes;
    }
    let fraction = gpu_fraction.clamp(0.05, 1.0);
    (vram_bytes as f64 * fraction).floor() as u64
}

fn unique_weight_bytes(manifest: &Manifest, include: impl Fn(&PageSpec) -> bool) -> u64 {
    let mut seen = BTreeSet::new();
    let mut total = 0_u64;

    for page in &manifest.pages {
        if !is_resident_kind(page) || !include(page) {
            continue;
        }
        let key = (page.size, page.checksum.as_str());
        if seen.insert(key) {
            total = total.saturating_add(page.size);
        }
    }

    total
}

fn physical_weight_bytes(manifest: &Manifest, include: impl Fn(&PageSpec) -> bool) -> u64 {
    manifest
        .pages
        .iter()
        .filter(|page| is_resident_kind(page) && include(page))
        .map(|page| page.size)
        .sum()
}

fn estimate_metadata_bytes(manifest: &Manifest) -> u64 {
    let page_table_bytes = manifest.pages.iter().fold(0_u64, |acc, page| {
        acc.saturating_add(2 + page.id.len() as u64 + 8 + 8 + 8 + 4 + 32)
    });
    let tape_bytes = manifest.execution_tape.iter().fold(0_u64, |acc, stage| {
        acc.saturating_add(stage.stage.len() as u64).saturating_add(
            stage
                .page_refs
                .iter()
                .map(|page| page.len() as u64)
                .sum::<u64>(),
        )
    });
    page_table_bytes.saturating_add(tape_bytes)
}

fn suggestions(
    manifest: &Manifest,
    options: &PlanOptions,
    weights_bytes: u64,
    scratch_bytes: u64,
    metadata_bytes: u64,
    status: PlanStatus,
    usable_vram_bytes: u64,
) -> Vec<String> {
    if status == PlanStatus::Ok {
        return Vec::new();
    }

    let mut suggestions = Vec::new();
    let fixed = weights_bytes
        .saturating_add(scratch_bytes)
        .saturating_add(metadata_bytes);
    if usable_vram_bytes > fixed {
        let max_ctx = max_context_for_budget(
            manifest,
            options,
            usable_vram_bytes - fixed,
            options.ctx_tokens,
        );
        if max_ctx < options.ctx_tokens {
            suggestions.push(format!("lower context to {max_ctx}"));
        }
    }

    if options.batch_size > 1 {
        suggestions.push("reduce batch size".to_string());
    }
    if !matches!(options.kv_dtype.as_str(), "q4" | "q3" | "q2") {
        suggestions.push("enable kv codec q4".to_string());
    } else if options.kv_dtype != "q2" {
        suggestions.push("try kv codec q2 for maximum context length".to_string());
    }
    if options.weight_residency == WeightResidency::All && manifest.model.layers > 1 {
        suggestions.push("use weight residency stream".to_string());
        suggestions.push(format!(
            "use weight residency offload-last-n with --offload-layers {}",
            (manifest.model.layers / 4).max(1)
        ));
    } else if manifest.model.layers > 1 {
        let start = manifest.model.layers / 2;
        suggestions.push(format!(
            "offload layers {start}..{}",
            manifest.model.layers.saturating_sub(1)
        ));
    } else if let Some(page) = largest_page(manifest) {
        suggestions.push(format!("offload page {}", page.id));
    }

    suggestions
}

fn max_context_for_budget(
    manifest: &Manifest,
    options: &PlanOptions,
    kv_budget: u64,
    upper: u64,
) -> u64 {
    let mut lo = 0;
    let mut hi = upper;
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let mut probe = options.clone();
        probe.ctx_tokens = mid;
        let batch_size = probe.batch_size.max(1);
        if estimate_kv_cache_bytes(manifest, &probe, batch_size) <= kv_budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

fn largest_page(manifest: &Manifest) -> Option<&PageSpec> {
    manifest
        .pages
        .iter()
        .filter(|page| is_resident_kind(page))
        .max_by_key(|page| page.size)
}

fn is_resident_kind(page: &PageSpec) -> bool {
    matches!(page.kind.as_str(), "weight" | "embedding" | "lm_head")
}
