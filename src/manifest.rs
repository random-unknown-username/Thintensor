//! Human-readable ThinTensor manifest schema and semantic validation.

use crate::error::Report;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FORMAT_NAME: &str = "thintensor";
pub const FORMAT_VERSION: u32 = 0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub model: ModelSpec,
    pub execution_tape: Vec<ExecutionStage>,
    pub pages: Vec<PageSpec>,
    pub memory_plan: MemoryPlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_arch: Option<String>,
    pub hidden_size: u64,
    pub layers: u32,
    pub heads: u32,
    pub kv_heads: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<u64>,
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_norm_eps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_eps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_theta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qkv_bias: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_naming_scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention_variants: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_precision_modes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub no_rope_layers: Vec<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_rope_layer_interval: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_sliding_window: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_max_position_embeddings: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_variant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlp_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layer_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_conv_kernel_dim: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_key_head_dim: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_value_head_dim: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_num_key_heads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_num_value_heads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_output_gate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_key_value_heads: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_size_per_layer_input: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size_per_layer_input: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_double_wide_mlp: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_sinks: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_local_experts: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts_per_token: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_topk_prob: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swiglu_alpha: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swiglu_limit: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_weight_offset: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_logit_softcap: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcap: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_operators: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_json_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_config_json_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStage {
    pub stage: String,
    #[serde(default)]
    pub page_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSpec {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    pub op: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub backend_layout: String,
    pub size: u64,
    pub checksum: String,
    #[serde(default)]
    pub experimental: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_to: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_offset: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits_per_weight: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_group_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_page: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub scratch_bytes: u64,
    pub min_vram_bytes: u64,
    pub recommended_vram_bytes: u64,
    pub kv_cache: KvCachePlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    pub policy: String,
    pub recent_tokens_high_precision: u64,
    pub old_tokens_codec: String,
}

pub fn validate_manifest(manifest: &Manifest) -> Report {
    let mut report = Report::default();

    if manifest.format != FORMAT_NAME {
        report.error(format!(
            "unknown manifest format {}; expected {FORMAT_NAME}",
            manifest.format
        ));
    }
    if manifest.version != FORMAT_VERSION {
        report.error(format!(
            "unknown format version {}; expected {FORMAT_VERSION}",
            manifest.version
        ));
    }

    validate_model(manifest, &mut report);
    let pages = validate_pages(manifest, &mut report);
    validate_tape(manifest, &pages, &mut report);
    validate_memory_plan(manifest, &mut report);

    report
}

fn validate_model(manifest: &Manifest, report: &mut Report) {
    let model = &manifest.model;

    require_name("model.arch", &model.arch, report);
    require_name("model.dtype", &model.dtype, report);

    if model.hidden_size == 0 {
        report.error("model.hidden_size must be non-zero");
    }
    if model.layers == 0 {
        report.error("model.layers must be non-zero");
    }
    if model.heads == 0 {
        report.error("model.heads must be non-zero");
    }
    if model.kv_heads == 0 {
        report.error("model.kv_heads must be non-zero");
    }
    if model.kv_heads > model.heads {
        report.error("model.kv_heads cannot exceed model.heads");
    }
    if model.heads != 0 && !model.hidden_size.is_multiple_of(model.heads as u64) {
        report.error("model.hidden_size must be divisible by model.heads");
    }
    if model.head_dim == Some(0) {
        report.error("model.head_dim must be non-zero when present");
    }
}

fn validate_pages(manifest: &Manifest, report: &mut Report) -> BTreeMap<String, PageSpec> {
    let mut pages = BTreeMap::new();

    if manifest.pages.is_empty() {
        report.error("pages must contain at least one page");
    }

    for page in &manifest.pages {
        require_id("pages[].id", &page.id, report);
        require_name("pages[].op", &page.op, report);
        require_name("pages[].dtype", &page.dtype, report);
        require_name("pages[].backend_layout", &page.backend_layout, report);

        if pages.insert(page.id.clone(), page.clone()).is_some() {
            report.error(format!("duplicate page id {}", page.id));
        }
        if !is_known_kind(&page.kind) && !page.experimental {
            report.error(format!(
                "unknown page kind {} for {}; set experimental=true to allow it",
                page.kind, page.id
            ));
        }
        if let Some(layer) = page.layer
            && layer >= manifest.model.layers
        {
            report.error(format!(
                "page {} targets layer {}, but model has {} layers",
                page.id, layer, manifest.model.layers
            ));
        }
        if page.shape.is_empty() {
            report.error(format!("page {} shape must not be empty", page.id));
        }
        if page.shape.contains(&0) {
            report.error(format!("page {} shape cannot contain zero", page.id));
        }
        if checked_shape_elems(&page.shape).is_none() {
            report.error(format!("page {} shape overflows u64", page.id));
        }
        if page.size == 0 {
            report.error(format!("page {} size must be non-zero", page.id));
        }
        if !is_lower_hex_32(&page.checksum) {
            report.error(format!(
                "page {} checksum must be 32-byte lowercase hex",
                page.id
            ));
        }
    }

    for page in &manifest.pages {
        if let Some(parent_id) = &page.fused_to {
            let Some(parent) = pages.get(parent_id) else {
                report.error(format!(
                    "logical page {} refers to missing fused page {}",
                    page.id, parent_id
                ));
                continue;
            };
            if parent.kind != "fused_physical" {
                report.error(format!(
                    "logical page {} fused parent {} is not fused_physical",
                    page.id, parent_id
                ));
            }
            let offset = page.fused_offset.unwrap_or(0);
            if offset
                .checked_add(page.size)
                .is_none_or(|end| end > parent.size)
            {
                report.error(format!(
                    "logical page {} slice is outside fused parent {}",
                    page.id, parent_id
                ));
            }
        } else if page.fused_offset.is_some() {
            report.error(format!(
                "page {} has fused_offset without fused_to",
                page.id
            ));
        }
    }

    pages
}

fn validate_tape(manifest: &Manifest, pages: &BTreeMap<String, PageSpec>, report: &mut Report) {
    if manifest.execution_tape.is_empty() {
        report.error("execution_tape must contain at least one stage");
    }

    let mut stages = BTreeSet::new();
    for stage in &manifest.execution_tape {
        require_id("execution_tape[].stage", &stage.stage, report);
        if !stages.insert(stage.stage.clone()) {
            report.error(format!("duplicate execution stage {}", stage.stage));
        }
        for page_id in &stage.page_refs {
            if !pages.contains_key(page_id) {
                report.error(format!(
                    "execution stage {} references missing page {}",
                    stage.stage, page_id
                ));
            }
        }
    }
}

fn validate_memory_plan(manifest: &Manifest, report: &mut Report) {
    let plan = &manifest.memory_plan;

    if plan.min_vram_bytes < plan.scratch_bytes {
        report.error("memory_plan.min_vram_bytes must be >= memory_plan.scratch_bytes");
    }
    if plan.recommended_vram_bytes < plan.min_vram_bytes {
        report.error("memory_plan.recommended_vram_bytes must be >= memory_plan.min_vram_bytes");
    }
    require_name("memory_plan.kv_cache.policy", &plan.kv_cache.policy, report);
    require_name(
        "memory_plan.kv_cache.old_tokens_codec",
        &plan.kv_cache.old_tokens_codec,
        report,
    );
}

fn is_known_kind(kind: &str) -> bool {
    matches!(kind, "weight" | "embedding" | "lm_head" | "fused_physical")
}

fn checked_shape_elems(shape: &[u64]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
}

fn require_id(field: &str, value: &str, report: &mut Report) {
    require_name(field, value, report);
    if value.contains('/') || value.contains('\\') {
        report.error(format!("{field} must not contain path separators"));
    }
    if value.chars().any(char::is_whitespace) {
        report.error(format!("{field} must not contain whitespace"));
    }
}

fn require_name(field: &str, value: &str, report: &mut Report) {
    if value.trim().is_empty() {
        report.error(format!("{field} must not be empty"));
    }
}

fn is_lower_hex_32(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
