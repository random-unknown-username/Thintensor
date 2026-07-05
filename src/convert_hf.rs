//! Hugging Face safetensors to ThinTensor v0 converter.

use crate::archive::{Archive, ArchivePage, ArchivePageSource, write_archive_pages};
use crate::manifest::{
    ExecutionStage, FORMAT_NAME, FORMAT_VERSION, KvCachePlan, Manifest, MemoryPlan, ModelSpec,
    PageSpec,
};
use crate::verify::verify_archive;
use anyhow::{Context, Result, anyhow, bail};
use glob::glob;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

const EMBED: &str = "model.embed_tokens.weight";
const FINAL_NORM: &str = "model.norm.weight";
const LM_HEAD: &str = "lm_head.weight";

#[derive(Debug, Clone)]
pub struct ConvertHfOptions {
    pub hf_dir: PathBuf,
    pub out_path: PathBuf,
    pub arch_override: Option<String>,
    pub include_tokenizer_hashes: bool,
}

#[derive(Debug)]
pub struct ConvertHfResult {
    pub archive: Archive,
    pub warnings: Vec<String>,
}

pub fn convert_hf(options: ConvertHfOptions) -> Result<ConvertHfResult> {
    let config_path = options.hf_dir.join("config.json");
    if !config_path.exists() {
        bail!("config.json missing in {}", options.hf_dir.display());
    }

    let config: Value = serde_json::from_reader(
        File::open(&config_path).with_context(|| format!("open {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))?;
    let safetensor_paths = collect_safetensors(&options.hf_dir)?;
    if safetensor_paths.is_empty() {
        bail!("no safetensors files found in {}", options.hf_dir.display());
    }

    let mut warnings = Vec::new();
    let tensors = collect_tensors(&safetensor_paths)?;
    let layers = required_u32(&config, "num_hidden_layers")?;
    let model = build_model_spec(&config, layers, &tensors, &options, &mut warnings)?;
    let pages = build_pages(&tensors, &config)?;
    let execution_tape = build_execution_tape(layers, &tensors, &mut warnings)?;
    warn_unknown_tensors(&tensors, layers, &execution_tape, &mut warnings);
    let archive_pages = archive_pages(&tensors, &execution_tape);

    let mut manifest = Manifest {
        format: FORMAT_NAME.to_string(),
        version: FORMAT_VERSION,
        model,
        execution_tape,
        pages,
        memory_plan: build_memory_plan(&config, &tensors),
    };
    add_tokenizer_hashes(&mut warnings, &mut manifest.model, &options)?;

    write_archive_pages(&options.out_path, &manifest, &archive_pages)?;
    let archive = Archive::open(&options.out_path)?;
    let report = verify_archive(&archive)?;
    if !report.is_ok() {
        bail!(
            "archive verify failed after convert:\n{}",
            report.errors.join("\n")
        );
    }

    Ok(ConvertHfResult { archive, warnings })
}

fn collect_safetensors(hf_dir: &Path) -> Result<Vec<PathBuf>> {
    let pattern = hf_dir.join("*.safetensors");
    let pattern = pattern
        .to_str()
        .ok_or_else(|| anyhow!("HF dir path is not valid UTF-8"))?;
    let mut paths = Vec::new();
    for entry in glob(pattern)? {
        paths.push(entry?);
    }
    paths.sort();
    Ok(paths)
}

fn collect_tensors(paths: &[PathBuf]) -> Result<BTreeMap<String, TensorPage>> {
    let mut tensors = BTreeMap::new();

    for path in paths {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only mapping; the file is never mutated by the converter.
        let mmap = unsafe { Mmap::map(&file).with_context(|| format!("mmap {}", path.display()))? };
        let safetensors =
            SafeTensors::deserialize(&mmap).with_context(|| format!("read {}", path.display()))?;
        let base = mmap.as_ptr() as usize;

        for (name, tensor) in safetensors.iter() {
            if name.is_empty() {
                bail!("tensor name in {} is empty", path.display());
            }
            let data = tensor.data();
            let data_start = data.as_ptr() as usize;
            let offset = data_start
                .checked_sub(base)
                .ok_or_else(|| anyhow!("tensor {name} data pointer is outside mmap"))?;
            let checksum = blake3::hash(data);
            let shape = tensor
                .shape()
                .iter()
                .map(|dim| u64::try_from(*dim).context("tensor shape dimension overflows u64"))
                .collect::<Result<Vec<_>>>()?;

            let page = TensorPage {
                id: name.to_string(),
                path: path.clone(),
                offset: offset as u64,
                size: data.len() as u64,
                checksum: *checksum.as_bytes(),
                dtype: format!("{:?}", tensor.dtype()).to_ascii_lowercase(),
                shape,
            };
            if tensors.insert(page.id.clone(), page).is_some() {
                bail!("tensor {name} appears twice");
            }
        }
    }

    Ok(tensors)
}

fn build_model_spec(
    config: &Value,
    layers: u32,
    tensors: &BTreeMap<String, TensorPage>,
    options: &ConvertHfOptions,
    warnings: &mut Vec<String>,
) -> Result<ModelSpec> {
    let raw_arch = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let arch = options
        .arch_override
        .clone()
        .or_else(|| raw_arch.as_deref().map(normalize_arch))
        .unwrap_or_else(|| {
            warnings.push("architectures[0] missing; using arch=unknown".to_string());
            "unknown".to_string()
        });

    let hidden_size = optional_u64(config, "hidden_size")
        .or_else(|| infer_hidden_size(tensors))
        .ok_or_else(|| anyhow!("config.json missing hidden_size and tensor inference failed"))?;
    if optional_u64(config, "hidden_size").is_none() {
        warnings.push(format!(
            "hidden_size inferred from tensor shapes as {hidden_size}"
        ));
    }

    let heads = optional_u32(config, "num_attention_heads")
        .or_else(|| infer_attention_heads(config, hidden_size))
        .ok_or_else(|| {
            anyhow!("config.json missing num_attention_heads and tensor inference failed")
        })?;
    if optional_u32(config, "num_attention_heads").is_none() {
        warnings.push(format!("num_attention_heads inferred as {heads}"));
    }

    let kv_heads = optional_u32(config, "num_key_value_heads")
        .or_else(|| infer_kv_heads(config, tensors))
        .unwrap_or(heads);
    if optional_u32(config, "num_key_value_heads").is_none() && kv_heads != heads {
        warnings.push(format!("num_key_value_heads inferred as {kv_heads}"));
    }
    let source_dtype = optional_string(config, "torch_dtype");
    let dtype = source_dtype
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let num_local_experts = optional_u64(config, "num_local_experts")
        .or_else(|| optional_u64(config, "num_experts"))
        .or_else(|| optional_u64(config, "n_routed_experts"));
    let num_experts_per_token = optional_u64(config, "num_experts_per_token")
        .or_else(|| optional_u64(config, "experts_per_token"))
        .or_else(|| optional_u64(config, "num_experts_per_tok"))
        .or_else(|| optional_u64(config, "num_selected_experts"));
    let is_moe = num_local_experts.is_some()
        || tensors.keys().any(|name| {
            name.contains(".mlp.experts.") || name.contains(".block_sparse_moe.experts.")
        });
    let attention_kind = if kv_heads == heads {
        "mha"
    } else if kv_heads == 1 {
        "mqa"
    } else {
        "gqa"
    };
    let attention_sinks = tensors
        .keys()
        .any(|name| name.ends_with(".self_attn.sinks"));
    let mut required_operators = vec![
        "embedding".to_string(),
        "rms_norm".to_string(),
        "qkv_projection".to_string(),
        "rope".to_string(),
        format!("{attention_kind}_attention"),
        "o_projection".to_string(),
        "lm_head".to_string(),
    ];
    if attention_sinks {
        required_operators.push("attention_sinks".to_string());
    }
    if is_moe {
        required_operators.extend([
            "topk_router".to_string(),
            "sparse_experts".to_string(),
            "expert_weighted_sum".to_string(),
        ]);
    } else {
        required_operators.extend([
            "gated_activation".to_string(),
            "down_projection".to_string(),
        ]);
    }
    let mut supported_precision_modes = vec!["bf16".to_string(), "head8".to_string()];
    if !is_moe {
        supported_precision_modes.extend([
            "down_fp8".to_string(),
            "gate_up_fp8".to_string(),
            "mlp_fp8".to_string(),
        ]);
    }
    if let Some(method) = config
        .get("quantization_config")
        .and_then(|value| value.get("quant_method"))
        .and_then(Value::as_str)
    {
        supported_precision_modes.insert(0, format!("native_{method}"));
    }

    Ok(ModelSpec {
        arch,
        model_type: optional_string(config, "model_type"),
        raw_arch,
        hidden_size,
        layers,
        heads,
        kv_heads,
        head_dim: optional_u64(config, "head_dim"),
        dtype,
        intermediate_size: optional_u64(config, "intermediate_size"),
        rms_norm_eps: optional_f64(config, "rms_norm_eps"),
        norm_kind: if optional_f64(config, "rms_norm_eps").is_some() {
            Some("rms_norm".to_string())
        } else if optional_f64(config, "layer_norm_eps").is_some() {
            Some("layer_norm".to_string())
        } else {
            None
        },
        norm_eps: optional_f64(config, "rms_norm_eps")
            .or_else(|| optional_f64(config, "layer_norm_eps")),
        rope_theta: optional_f64(config, "rope_theta"),
        partial_rotary_factor: optional_f64(config, "partial_rotary_factor"),
        rope_scaling: config
            .get("rope_scaling")
            .filter(|value| !value.is_null())
            .cloned(),
        source_dtype,
        vocab_size: optional_u64(config, "vocab_size"),
        tie_word_embeddings: config.get("tie_word_embeddings").and_then(Value::as_bool),
        activation: optional_string(config, "hidden_act"),
        qkv_bias: config
            .get("qkv_bias")
            .or_else(|| config.get("attention_bias"))
            .and_then(Value::as_bool),
        attention_bias: config.get("attention_bias").and_then(Value::as_bool),
        tensor_naming_scheme: Some("hf_decoder_layers".to_string()),
        attention_variants: vec!["causal_kv".to_string(), "current_only_smoke".to_string()],
        supported_precision_modes,
        no_rope_layers: config
            .get("no_rope_layers")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_bool().or_else(|| value.as_u64().map(|v| v != 0)))
                    .collect()
            })
            .unwrap_or_default(),
        no_rope_layer_interval: optional_u64(config, "no_rope_layer_interval"),
        sliding_window: optional_u64(config, "sliding_window"),
        use_sliding_window: config.get("use_sliding_window").and_then(Value::as_bool),
        max_position_embeddings: optional_u64(config, "max_position_embeddings"),
        original_max_position_embeddings: optional_u64(
            config,
            "original_max_position_embeddings",
        ),
        rope_variant: config
            .get("rope_scaling")
            .and_then(|value| value.get("rope_type").or_else(|| value.get("type")))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        architecture_family: Some(
            if is_moe {
                "decoder_moe"
            } else {
                "decoder_dense"
            }
            .to_string(),
        ),
        attention_kind: Some(attention_kind.to_string()),
        mlp_kind: Some(if is_moe { "sparse_moe" } else { "gated_dense" }.to_string()),
        layer_types: config
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        attention_sinks: Some(attention_sinks),
        num_local_experts,
        num_experts_per_token,
        swiglu_alpha: optional_f64(config, "swiglu_alpha"),
        swiglu_limit: optional_f64(config, "swiglu_limit"),
        rope_parameters: config
            .get("rope_parameters")
            .or_else(|| config.get("rope_scaling"))
            .filter(|value| !value.is_null())
            .cloned(),
        quantization_config: config
            .get("quantization_config")
            .filter(|value| !value.is_null())
            .cloned(),
        required_operators,
        tokenizer_json_hash: None,
        tokenizer_config_json_hash: None,
    })
}

fn add_tokenizer_hashes(
    warnings: &mut Vec<String>,
    model: &mut ModelSpec,
    options: &ConvertHfOptions,
) -> Result<()> {
    if !options.include_tokenizer_hashes {
        return Ok(());
    }

    let tokenizer = options.hf_dir.join("tokenizer.json");
    if tokenizer.exists() {
        model.tokenizer_json_hash = Some(hash_file(&tokenizer)?);
    } else {
        warnings.push("tokenizer.json missing; continuing without tokenizer hash".to_string());
    }

    let tokenizer_config = options.hf_dir.join("tokenizer_config.json");
    if tokenizer_config.exists() {
        model.tokenizer_config_json_hash = Some(hash_file(&tokenizer_config)?);
    }

    Ok(())
}

fn build_pages(tensors: &BTreeMap<String, TensorPage>, config: &Value) -> Result<Vec<PageSpec>> {
    let quant = config.get("quantization_config");
    let global_scheme = quant
        .and_then(|value| value.get("quant_method"))
        .and_then(Value::as_str);
    let global_bits = quant
        .and_then(|value| {
            value
                .get("bits")
                .or_else(|| value.get("weight_bits"))
                .or_else(|| value.get("num_bits"))
        })
        .and_then(Value::as_u64)
        .and_then(|value| u8::try_from(value).ok());
    let global_group = quant
        .and_then(|value| value.get("group_size").or_else(|| value.get("block_size")))
        .and_then(Value::as_u64);
    tensors
        .values()
        .map(|tensor| {
            let is_scale = tensor.id.ends_with("_scales")
                || tensor.id.ends_with(".scales")
                || tensor.id.ends_with("_scale")
                || tensor.id.ends_with(".weight_scale");
            let is_packed = tensor.id.ends_with("_blocks")
                || tensor.id.contains("qweight")
                || matches!(
                    tensor.dtype.to_ascii_uppercase().as_str(),
                    "U8" | "I8" | "I32" | "UINT8" | "INT8" | "INT32"
                ) && !is_scale;
            let quant_scheme = if tensor.id.ends_with("_blocks") {
                Some("mxfp4".to_string())
            } else if is_packed {
                global_scheme.map(ToOwned::to_owned)
            } else {
                None
            };
            let bits_per_weight = if tensor.id.ends_with("_blocks") {
                Some(4)
            } else if is_packed {
                global_bits
            } else {
                None
            };
            let quant_group_size = if tensor.id.ends_with("_blocks") {
                Some(32)
            } else if is_packed {
                global_group
            } else {
                None
            };
            let scale_page = if tensor.id.ends_with("_blocks") {
                Some(tensor.id.replace("_blocks", "_scales"))
            } else {
                None
            };
            Ok(PageSpec {
                id: tensor.id.clone(),
                kind: page_kind(&tensor.id).to_string(),
                layer: layer_from_tensor(&tensor.id),
                op: op_from_tensor(&tensor.id).to_string(),
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                backend_layout: backend_layout(&tensor.id),
                size: tensor.size,
                checksum: hex::encode(tensor.checksum),
                experimental: false,
                fused_to: None,
                fused_offset: None,
                quant_scheme,
                bits_per_weight,
                quant_group_size,
                scale_page,
            })
        })
        .collect()
}

fn build_execution_tape(
    layers: u32,
    tensors: &BTreeMap<String, TensorPage>,
    warnings: &mut Vec<String>,
) -> Result<Vec<ExecutionStage>> {
    if !tensors.contains_key(EMBED) {
        bail!("required tensor {EMBED} missing");
    }

    let mut stages = Vec::new();
    stages.push(stage("embed", vec![EMBED.to_string()]));

    let mut missing_q_norm = 0_u32;
    let mut missing_k_norm = 0_u32;
    for layer in 0..layers {
        let input_norm = layer_tensor(layer, "input_layernorm.weight");
        let q_proj = layer_tensor(layer, "self_attn.q_proj.weight");
        let k_proj = layer_tensor(layer, "self_attn.k_proj.weight");
        let v_proj = layer_tensor(layer, "self_attn.v_proj.weight");
        let fused_qkv = layer_tensor(layer, "self_attn.qkv_proj.weight");
        let q_norm = layer_tensor(layer, "self_attn.q_norm.weight");
        let k_norm = layer_tensor(layer, "self_attn.k_norm.weight");
        let o_proj = layer_tensor(layer, "self_attn.o_proj.weight");
        let post_norm = layer_tensor(layer, "post_attention_layernorm.weight");
        let gate_proj = layer_tensor(layer, "mlp.gate_proj.weight");
        let up_proj = layer_tensor(layer, "mlp.up_proj.weight");
        let fused_gate_up = layer_tensor(layer, "mlp.gate_up_proj.weight");
        let down_proj = layer_tensor(layer, "mlp.down_proj.weight");
        let router = first_existing(
            tensors,
            &[
                layer_tensor(layer, "mlp.router.weight"),
                layer_tensor(layer, "block_sparse_moe.gate.weight"),
                layer_tensor(layer, "mlp.gate.weight"),
            ],
        );
        let packed_gate_up = first_existing(
            tensors,
            &[
                layer_tensor(layer, "mlp.experts.gate_up_proj"),
                layer_tensor(layer, "mlp.experts.gate_up_proj_blocks"),
            ],
        );
        let packed_down = first_existing(
            tensors,
            &[
                layer_tensor(layer, "mlp.experts.down_proj"),
                layer_tensor(layer, "mlp.experts.down_proj_blocks"),
            ],
        );

        require_tensor(tensors, &input_norm)?;
        let fused_qkv_refs = tensor_group(tensors, &fused_qkv).ok();
        let separate_qkv_refs = if fused_qkv_refs.is_none() {
            Some([
                tensor_group(tensors, &q_proj)?,
                tensor_group(tensors, &k_proj)?,
                tensor_group(tensors, &v_proj)?,
            ])
        } else {
            None
        };
        let o_projection_refs = tensor_group(tensors, &o_proj)?;
        require_tensor(tensors, &post_norm)?;

        let mut input_norm_refs = vec![input_norm.clone()];
        push_if_present(
            tensors,
            &mut input_norm_refs,
            layer_tensor(layer, "input_layernorm.bias"),
        );
        stages.push(stage(
            format!("layer_{layer}_input_norm"),
            input_norm_refs,
        ));

        let mut qkv_refs = if let Some(refs) = fused_qkv_refs {
            refs
        } else {
            separate_qkv_refs
                .expect("separate QKV groups were built")
                .into_iter()
                .flatten()
                .collect()
        };
        for suffix in [
            "self_attn.qkv_proj.bias",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.sinks",
        ] {
            let name = layer_tensor(layer, suffix);
            if tensors.contains_key(&name) {
                qkv_refs.push(name);
            }
        }
        if tensors.contains_key(&q_norm) {
            qkv_refs.push(q_norm);
        } else {
            missing_q_norm += 1;
        }
        if tensors.contains_key(&k_norm) {
            qkv_refs.push(k_norm);
        } else {
            missing_k_norm += 1;
        }
        stages.push(stage(format!("layer_{layer}_attn_qkv"), qkv_refs));
        stages.push(stage(format!("layer_{layer}_rope"), Vec::new()));
        let mut o_refs = o_projection_refs;
        let o_bias = layer_tensor(layer, "self_attn.o_proj.bias");
        if tensors.contains_key(&o_bias) {
            o_refs.push(o_bias);
        }
        stages.push(stage(format!("layer_{layer}_attn_out"), o_refs));
        let mut post_norm_refs = vec![post_norm];
        push_if_present(
            tensors,
            &mut post_norm_refs,
            layer_tensor(layer, "post_attention_layernorm.bias"),
        );
        stages.push(stage(
            format!("layer_{layer}_post_attn_norm"),
            post_norm_refs,
        ));
        if let (Some(router), Some(packed_gate_up), Some(packed_down)) =
            (router.clone(), packed_gate_up, packed_down)
        {
            let mut router_refs = vec![router];
            push_if_present(
                tensors,
                &mut router_refs,
                layer_tensor(layer, "mlp.router.bias"),
            );
            stages.push(stage(format!("layer_{layer}_moe_router"), router_refs));

            let mut gate_up_refs = vec![packed_gate_up];
            for suffix in [
                "mlp.experts.gate_up_proj_scales",
                "mlp.experts.gate_up_proj_bias",
            ] {
                push_if_present(tensors, &mut gate_up_refs, layer_tensor(layer, suffix));
            }
            stages.push(stage(format!("layer_{layer}_moe_gate_up"), gate_up_refs));

            let mut down_refs = vec![packed_down];
            for suffix in ["mlp.experts.down_proj_scales", "mlp.experts.down_proj_bias"] {
                push_if_present(tensors, &mut down_refs, layer_tensor(layer, suffix));
            }
            stages.push(stage(format!("layer_{layer}_moe_down"), down_refs));
        } else if let Some(router) = router {
            let mut router_refs = vec![router];
            push_if_present(
                tensors,
                &mut router_refs,
                layer_tensor(layer, "mlp.router.bias"),
            );
            stages.push(stage(format!("layer_{layer}_moe_router"), router_refs));
            let prefix_a = format!("model.layers.{layer}.block_sparse_moe.");
            let prefix_b = format!("model.layers.{layer}.mlp.experts.");
            let refs: Vec<String> = tensors
                .keys()
                .filter(|name| name.starts_with(&prefix_a) || name.starts_with(&prefix_b))
                .cloned()
                .collect();
            if refs.is_empty() {
                bail!("layer {layer} declares an MoE router but has no expert tensors");
            }
            stages.push(stage(format!("layer_{layer}_moe_experts"), refs));
        } else {
            let fused_gate_refs = tensor_group(tensors, &fused_gate_up).ok();
            let separate_gate_refs = if fused_gate_refs.is_none() {
                Some((
                    tensor_group(tensors, &gate_proj)?,
                    tensor_group(tensors, &up_proj)?,
                ))
            } else {
                None
            };
            let down_refs = tensor_group(tensors, &down_proj)?;
            stages.push(stage(format!("layer_{layer}_mlp_gate_up"), {
                if let Some(mut refs) = fused_gate_refs {
                    push_if_present(
                        tensors,
                        &mut refs,
                        layer_tensor(layer, "mlp.gate_up_proj.bias"),
                    );
                    refs
                } else {
                    let (gate, up) =
                        separate_gate_refs.expect("separate gate/up groups were built");
                    gate.into_iter().chain(up).collect()
                }
            }));
            stages.push(stage(format!("layer_{layer}_mlp_down"), down_refs));
        }
    }

    if missing_q_norm > 0 {
        warnings.push(format!(
            "q_norm.weight missing for {missing_q_norm} layers; optional"
        ));
    }
    if missing_k_norm > 0 {
        warnings.push(format!(
            "k_norm.weight missing for {missing_k_norm} layers; optional"
        ));
    }

    if tensors.contains_key(FINAL_NORM) {
        let mut final_norm_refs = vec![FINAL_NORM.to_string()];
        push_if_present(
            tensors,
            &mut final_norm_refs,
            "model.norm.bias".to_string(),
        );
        stages.push(stage("final_norm", final_norm_refs));
    }
    if tensors.contains_key(LM_HEAD) {
        stages.push(stage("lm_head", vec![LM_HEAD.to_string()]));
    } else {
        warnings.push("lm_head.weight missing; continuing without lm_head stage".to_string());
    }

    Ok(stages)
}

fn warn_unknown_tensors(
    tensors: &BTreeMap<String, TensorPage>,
    layers: u32,
    execution_tape: &[ExecutionStage],
    warnings: &mut Vec<String>,
) {
    let mut known = BTreeSet::from([
        EMBED.to_string(),
        FINAL_NORM.to_string(),
        LM_HEAD.to_string(),
    ]);
    for layer in 0..layers {
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            known.insert(layer_tensor(layer, suffix));
        }
    }
    known.extend(
        execution_tape
            .iter()
            .flat_map(|stage| stage.page_refs.iter().cloned()),
    );

    let unknown: Vec<_> = tensors
        .keys()
        .filter(|name| !known.contains(*name))
        .take(8)
        .cloned()
        .collect();
    if !unknown.is_empty() {
        warnings.push(format!(
            "unknown extra tensors present: {}{}",
            unknown.join(", "),
            if tensors.keys().filter(|name| !known.contains(*name)).count() > unknown.len() {
                ", ..."
            } else {
                ""
            }
        ));
    }
}

fn build_memory_plan(config: &Value, tensors: &BTreeMap<String, TensorPage>) -> MemoryPlan {
    let weights_bytes = tensors.values().map(|tensor| tensor.size).sum::<u64>();
    let hidden_size = optional_u64(config, "hidden_size")
        .or_else(|| infer_hidden_size(tensors))
        .unwrap_or(4096);
    let intermediate_size = optional_u64(config, "intermediate_size").unwrap_or(hidden_size * 4);
    let scratch_bytes = (64 * 1024 * 1024).max(
        hidden_size
            .saturating_mul(intermediate_size)
            .saturating_mul(2),
    );
    let min_vram_bytes = weights_bytes.saturating_add(scratch_bytes);
    let recommended_vram_bytes = min_vram_bytes.max(8 * 1024 * 1024 * 1024);

    MemoryPlan {
        scratch_bytes,
        min_vram_bytes,
        recommended_vram_bytes,
        kv_cache: KvCachePlan {
            policy: "paged".to_string(),
            recent_tokens_high_precision: 256,
            old_tokens_codec: "q4".to_string(),
        },
    }
}

fn archive_pages(
    tensors: &BTreeMap<String, TensorPage>,
    execution_tape: &[ExecutionStage],
) -> Vec<ArchivePage<'static>> {
    let mut ordered = Vec::with_capacity(tensors.len());
    let mut seen = BTreeSet::new();
    for page_id in execution_tape
        .iter()
        .flat_map(|stage| stage.page_refs.iter())
        .chain(tensors.keys())
    {
        if !seen.insert(page_id.clone()) {
            continue;
        }
        let Some(tensor) = tensors.get(page_id) else {
            continue;
        };
        ordered.push(ArchivePage {
            id: tensor.id.clone(),
            size: tensor.size,
            checksum: tensor.checksum,
            source: ArchivePageSource::FileRange {
                path: tensor.path.clone(),
                offset: tensor.offset,
            },
        });
    }
    ordered
}

fn require_tensor(tensors: &BTreeMap<String, TensorPage>, name: &str) -> Result<()> {
    if tensors.contains_key(name) {
        Ok(())
    } else {
        bail!("required layer tensor {name} missing")
    }
}

fn tensor_group(tensors: &BTreeMap<String, TensorPage>, weight_id: &str) -> Result<Vec<String>> {
    if tensors.contains_key(weight_id) {
        return Ok(vec![weight_id.to_string()]);
    }
    let stem = weight_id.strip_suffix(".weight").unwrap_or(weight_id);
    let primary = [
        format!("{stem}.qweight"),
        format!("{stem}.weight_packed"),
        format!("{stem}.weight_blocks"),
    ]
    .into_iter()
    .find(|name| tensors.contains_key(name));
    let Some(primary) = primary else {
        bail!("required tensor role {weight_id} has no weight or packed weight page");
    };
    let mut refs = vec![primary];
    for suffix in [
        "scales",
        "qzeros",
        "g_idx",
        "weight_scale",
        "weight_zero_point",
        "input_scale",
    ] {
        let name = format!("{stem}.{suffix}");
        if tensors.contains_key(&name) {
            refs.push(name);
        }
    }
    Ok(refs)
}

fn infer_hidden_size(tensors: &BTreeMap<String, TensorPage>) -> Option<u64> {
    tensors
        .get(EMBED)
        .and_then(|tensor| tensor.shape.last().copied())
        .or_else(|| {
            tensors
                .get(&layer_tensor(0, "self_attn.q_proj.weight"))
                .and_then(|tensor| tensor.shape.get(1).copied())
        })
}

fn infer_attention_heads(config: &Value, hidden_size: u64) -> Option<u32> {
    let head_dim = optional_u64(config, "head_dim")?;
    if head_dim == 0 || !hidden_size.is_multiple_of(head_dim) {
        return None;
    }
    u32::try_from(hidden_size / head_dim).ok()
}

fn infer_kv_heads(config: &Value, tensors: &BTreeMap<String, TensorPage>) -> Option<u32> {
    let head_dim = optional_u64(config, "head_dim")?;
    if head_dim == 0 {
        return None;
    }
    let k_proj = tensors.get(&layer_tensor(0, "self_attn.k_proj.weight"))?;
    let rows = *k_proj.shape.first()?;
    if !rows.is_multiple_of(head_dim) {
        return None;
    }
    u32::try_from(rows / head_dim).ok()
}

fn stage(stage: impl Into<String>, page_refs: Vec<String>) -> ExecutionStage {
    ExecutionStage {
        stage: stage.into(),
        page_refs,
    }
}

fn first_existing(tensors: &BTreeMap<String, TensorPage>, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|name| tensors.contains_key(*name))
        .cloned()
}

fn push_if_present(tensors: &BTreeMap<String, TensorPage>, refs: &mut Vec<String>, name: String) {
    if tensors.contains_key(&name) {
        refs.push(name);
    }
}

fn layer_tensor(layer: u32, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

fn layer_from_tensor(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse().ok()
}

fn op_from_tensor(name: &str) -> &str {
    if name == EMBED {
        "embed_tokens"
    } else if name == FINAL_NORM {
        "final_norm"
    } else if name == LM_HEAD {
        "lm_head"
    } else if name.ends_with("input_layernorm.weight") {
        "input_layernorm"
    } else if name.ends_with("q_proj.weight") {
        "attn_q_proj"
    } else if name.ends_with("k_proj.weight") {
        "attn_k_proj"
    } else if name.ends_with("v_proj.weight") {
        "attn_v_proj"
    } else if name.ends_with("o_proj.weight") {
        "attn_o_proj"
    } else if name.ends_with("q_norm.weight") {
        "attn_q_norm"
    } else if name.ends_with("k_norm.weight") {
        "attn_k_norm"
    } else if name.ends_with("post_attention_layernorm.weight") {
        "post_attention_layernorm"
    } else if name.ends_with("self_attn.sinks") {
        "attention_sinks"
    } else if name.ends_with("mlp.router.weight") || name.ends_with("block_sparse_moe.gate.weight")
    {
        "moe_router"
    } else if name.contains(".mlp.experts.gate_up_proj")
        || name.contains(".block_sparse_moe.experts.") && name.contains(".w1.")
        || name.contains(".block_sparse_moe.experts.") && name.contains(".w3.")
    {
        "moe_gate_up"
    } else if name.contains(".mlp.experts.down_proj")
        || name.contains(".block_sparse_moe.experts.") && name.contains(".w2.")
    {
        "moe_down"
    } else if name.ends_with("gate_proj.weight") {
        "mlp_gate_proj"
    } else if name.ends_with("up_proj.weight") {
        "mlp_up_proj"
    } else if name.ends_with("down_proj.weight") {
        "mlp_down_proj"
    } else {
        "unknown"
    }
}

fn page_kind(name: &str) -> &str {
    if name == EMBED {
        "embedding"
    } else if name == LM_HEAD {
        "lm_head"
    } else {
        "weight"
    }
}

fn backend_layout(name: &str) -> String {
    if name.ends_with("_blocks") {
        "mxfp4_group32_packed".to_string()
    } else if name.ends_with("_scales") {
        "mxfp8_e8m0_scales".to_string()
    } else if name.contains(".mlp.experts.") || name.contains(".block_sparse_moe.experts.") {
        "expert_major_contiguous".to_string()
    } else {
        "row_major_contiguous".to_string()
    }
}

fn normalize_arch(raw: &str) -> String {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("qwen") {
        "qwen".to_string()
    } else if lower.contains("llama") {
        "llama".to_string()
    } else {
        lower
    }
}

fn hash_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn required_u64(config: &Value, key: &str) -> Result<u64> {
    optional_u64(config, key).ok_or_else(|| anyhow!("config.json missing {key}"))
}

fn required_u32(config: &Value, key: &str) -> Result<u32> {
    let value = required_u64(config, key)?;
    u32::try_from(value).with_context(|| format!("config {key} overflows u32"))
}

fn optional_u64(config: &Value, key: &str) -> Option<u64> {
    config.get(key).and_then(Value::as_u64)
}

fn optional_u32(config: &Value, key: &str) -> Option<u32> {
    optional_u64(config, key).and_then(|value| u32::try_from(value).ok())
}

fn optional_f64(config: &Value, key: &str) -> Option<f64> {
    config.get(key).and_then(Value::as_f64)
}

fn optional_string(config: &Value, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

#[derive(Debug, Clone)]
struct TensorPage {
    id: String,
    path: PathBuf,
    offset: u64,
    size: u64,
    checksum: [u8; 32],
    dtype: String,
    shape: Vec<u64>,
}
