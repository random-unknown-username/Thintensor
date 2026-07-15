//! Hugging Face safetensors to ThinTensor v0 converter.

use crate::archive::{
    Archive, ArchivePage, ArchivePageSource, ResumableArchiveWriter, planned_archive_size,
    write_archive_pages,
};
use crate::manifest::{
    ExecutionStage, FORMAT_NAME, FORMAT_VERSION, KvCachePlan, Manifest, MemoryPlan, ModelSpec,
    PageSpec,
};
use crate::verify::verify_archive;
use anyhow::{Context, Result, anyhow, bail};
use fs2::available_space;
use glob::glob;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
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
    pub streaming_pack: bool,
    pub consume_source_shards: bool,
    pub minimum_free_bytes: u64,
    pub resume: bool,
}

#[derive(Debug)]
pub struct ConvertHfResult {
    pub archive: Archive,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversionDeletionPoint {
    pub source_shard: PathBuf,
    pub source_bytes: u64,
    pub retained_pages: usize,
    pub after_page_index: Option<usize>,
    pub after_page_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversionDryRunReport {
    pub schema: String,
    pub hf_dir: PathBuf,
    pub out_path: PathBuf,
    pub shard_order: Vec<PathBuf>,
    pub expected_archive_bytes: u64,
    pub archive_payload_bytes: u64,
    pub peak_temporary_bytes_upper_bound: u64,
    pub available_bytes_before: u64,
    pub projected_minimum_available_bytes: u64,
    pub minimum_free_bytes: u64,
    pub consume_source_shards: bool,
    pub payload_checksums_computed: bool,
    pub meets_minimum_free_requirement: bool,
    pub deletion_points: Vec<ConversionDeletionPoint>,
    pub warnings: Vec<String>,
}

struct PreparedHfConversion {
    manifest: Manifest,
    archive_pages: Vec<ArchivePage<'static>>,
    safetensor_paths: Vec<PathBuf>,
    warnings: Vec<String>,
}

pub fn convert_hf(options: ConvertHfOptions) -> Result<ConvertHfResult> {
    if options.streaming_pack && options.resume && conversion_plan_path(&options.out_path).exists()
    {
        return resume_streaming_conversion(&options);
    }
    let prepared = prepare_hf_conversion(&options)?;
    if options.streaming_pack {
        return write_streaming_conversion(
            &options,
            prepared.manifest,
            prepared.archive_pages,
            prepared.safetensor_paths,
            prepared.warnings,
        );
    }

    write_archive_pages(
        &options.out_path,
        &prepared.manifest,
        &prepared.archive_pages,
    )?;
    let archive = Archive::open(&options.out_path)?;
    let report = verify_archive(&archive)?;
    if !report.is_ok() {
        bail!(
            "archive verify failed after convert:\n{}",
            report.errors.join("\n")
        );
    }

    Ok(ConvertHfResult {
        archive,
        warnings: prepared.warnings,
    })
}

fn prepare_hf_conversion(options: &ConvertHfOptions) -> Result<PreparedHfConversion> {
    let config_path = options.hf_dir.join("config.json");
    if !config_path.exists() {
        bail!("config.json missing in {}", options.hf_dir.display());
    }

    let source_config: Value = serde_json::from_reader(
        File::open(&config_path).with_context(|| format!("open {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))?;
    let config = effective_text_config(&source_config);
    let safetensor_paths = collect_safetensors(&options.hf_dir)?;
    if safetensor_paths.is_empty() {
        bail!("no safetensors files found in {}", options.hf_dir.display());
    }

    let mut warnings = Vec::new();
    let raw_tensors = collect_tensors(&safetensor_paths)?;
    let (tensors, excluded_tensors) = canonical_text_tensors(&source_config, raw_tensors)?;
    let tensors = canonical_common_tensors(tensors)?;
    if excluded_tensors > 0 {
        warnings.push(format!(
            "excluded {excluded_tensors} non-text vision/MTP tensors from the native text archive"
        ));
    }
    let layers = required_u32(&config, "num_hidden_layers")?;
    let model = build_model_spec(&config, layers, &tensors, options, &mut warnings)?;
    let pages = build_pages(&tensors, &config)?;
    let execution_tape = build_execution_tape(layers, &tensors, &config, &mut warnings)?;
    warn_unknown_tensors(&tensors, layers, &execution_tape, &mut warnings);
    let archive_pages = archive_pages(&tensors, &execution_tape);
    let memory_plan = build_memory_plan(&config, &tensors, &execution_tape);

    let mut manifest = Manifest {
        format: FORMAT_NAME.to_string(),
        version: FORMAT_VERSION,
        model,
        execution_tape,
        pages,
        memory_plan,
    };
    add_tokenizer_hashes(&mut warnings, &mut manifest.model, options)?;

    Ok(PreparedHfConversion {
        manifest,
        archive_pages,
        safetensor_paths,
        warnings,
    })
}

pub fn dry_run_hf(options: &ConvertHfOptions) -> Result<ConversionDryRunReport> {
    if options.resume {
        bail!("--dry-run cannot be combined with --resume");
    }
    // Manifest checksums serialize as JSON byte arrays, so their decimal
    // widths affect the exact archive prefix length. A truly exact dry-run
    // must therefore hash payloads just like conversion; header-only planning
    // is suitable for a bound, but not an exact byte claim.
    let prepared = prepare_hf_conversion(options)?;
    let archive_payload_bytes = prepared.archive_pages.iter().try_fold(0_u64, |acc, page| {
        acc.checked_add(page.size)
            .ok_or_else(|| anyhow!("archive payload size overflows u64"))
    })?;
    let expected_archive_bytes = planned_archive_size(&prepared.manifest, &prepared.archive_pages)?;
    let output_parent = options
        .out_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let available_bytes_before = available_space(output_parent)
        .with_context(|| format!("query free space for {}", output_parent.display()))?;
    let prefix_bytes = expected_archive_bytes.saturating_sub(archive_payload_bytes);
    let source_indices = prepared.archive_pages.iter().enumerate().fold(
        BTreeMap::<PathBuf, Vec<usize>>::new(),
        |mut result, (index, page)| {
            if let ArchivePageSource::FileRange { path, .. } = &page.source {
                result.entry(path.clone()).or_default().push(index);
            }
            result
        },
    );
    let mut deletion_points = Vec::with_capacity(prepared.safetensor_paths.len());
    let mut projected_available = available_bytes_before.saturating_sub(prefix_bytes);
    let mut projected_minimum = projected_available;
    for source in &prepared.safetensor_paths {
        let source_bytes = source
            .metadata()
            .with_context(|| format!("stat {}", source.display()))?
            .len();
        let indices = source_indices.get(source);
        let after_page_index = indices.and_then(|values| values.last().copied());
        deletion_points.push(ConversionDeletionPoint {
            source_shard: source.clone(),
            source_bytes,
            retained_pages: indices.map_or(0, Vec::len),
            after_page_index,
            after_page_id: after_page_index.map(|index| prepared.archive_pages[index].id.clone()),
        });
        if options.consume_source_shards && indices.is_none() {
            projected_available = projected_available.saturating_add(source_bytes);
        }
    }
    for (index, page) in prepared.archive_pages.iter().enumerate() {
        projected_available = projected_available.saturating_sub(page.size);
        projected_minimum = projected_minimum.min(projected_available);
        if options.consume_source_shards {
            for point in deletion_points
                .iter()
                .filter(|point| point.after_page_index == Some(index))
            {
                projected_available = projected_available.saturating_add(point.source_bytes);
            }
        }
    }
    let manifest_bytes = serde_json::to_vec_pretty(&prepared.manifest)?.len() as u64;
    let plan_records_bytes = prepared.archive_pages.iter().fold(0_u64, |acc, page| {
        acc.saturating_add(page.id.len() as u64 + 256)
    });
    let peak_temporary_bytes_upper_bound = manifest_bytes
        .saturating_add(plan_records_bytes)
        .saturating_add(1024 * 1024);

    Ok(ConversionDryRunReport {
        schema: "thintensor.conversion_dry_run.v1".to_string(),
        hf_dir: options.hf_dir.clone(),
        out_path: options.out_path.clone(),
        shard_order: prepared.safetensor_paths,
        expected_archive_bytes,
        archive_payload_bytes,
        peak_temporary_bytes_upper_bound,
        available_bytes_before,
        projected_minimum_available_bytes: projected_minimum,
        minimum_free_bytes: options.minimum_free_bytes,
        consume_source_shards: options.consume_source_shards,
        payload_checksums_computed: true,
        meets_minimum_free_requirement: projected_minimum >= options.minimum_free_bytes,
        deletion_points,
        warnings: prepared.warnings,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamingPage {
    id: String,
    size: u64,
    checksum: String,
    source_path: PathBuf,
    source_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamingConversionPlan {
    schema_version: u32,
    hf_dir: PathBuf,
    out_path: PathBuf,
    manifest: Manifest,
    pages: Vec<StreamingPage>,
    source_shards: Vec<PathBuf>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamingConversionJournal {
    schema_version: u32,
    plan_blake3: String,
    completed_pages: usize,
    verified_sources: BTreeSet<PathBuf>,
    deleted_sources: BTreeSet<PathBuf>,
    completed: bool,
}

fn write_streaming_conversion(
    options: &ConvertHfOptions,
    manifest: Manifest,
    pages: Vec<ArchivePage<'static>>,
    source_shards: Vec<PathBuf>,
    warnings: Vec<String>,
) -> Result<ConvertHfResult> {
    if options.out_path.exists() {
        bail!("output {} already exists", options.out_path.display());
    }
    let hf_dir = fs::canonicalize(&options.hf_dir)
        .with_context(|| format!("canonicalize {}", options.hf_dir.display()))?;
    let out_path = absolute_output_path(&options.out_path)?;
    let source_shards = source_shards
        .iter()
        .map(|path| {
            fs::canonicalize(path).with_context(|| format!("canonicalize {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let pages = pages
        .into_iter()
        .map(|page| {
            let ArchivePageSource::FileRange { path, offset } = page.source else {
                bail!("streaming HF conversion requires file-range page sources")
            };
            Ok(StreamingPage {
                id: page.id,
                size: page.size,
                checksum: hex::encode(page.checksum),
                source_path: fs::canonicalize(&path)
                    .with_context(|| format!("canonicalize {}", path.display()))?,
                source_offset: offset,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let plan = StreamingConversionPlan {
        schema_version: 1,
        hf_dir,
        out_path,
        manifest,
        pages,
        source_shards,
        warnings,
    };
    let plan_bytes = serde_json::to_vec_pretty(&plan).context("serialize conversion plan")?;
    let plan_hash = blake3::hash(&plan_bytes).to_hex().to_string();
    write_atomic_bytes(&conversion_plan_path(&options.out_path), &plan_bytes)?;
    let journal = StreamingConversionJournal {
        schema_version: 1,
        plan_blake3: plan_hash,
        completed_pages: 0,
        verified_sources: BTreeSet::new(),
        deleted_sources: BTreeSet::new(),
        completed: false,
    };
    write_atomic_json(&conversion_journal_path(&options.out_path), &journal)?;
    run_streaming_conversion(options, plan, journal)
}

fn resume_streaming_conversion(options: &ConvertHfOptions) -> Result<ConvertHfResult> {
    let plan_path = conversion_plan_path(&options.out_path);
    let journal_path = conversion_journal_path(&options.out_path);
    let plan_bytes =
        fs::read(&plan_path).with_context(|| format!("read {}", plan_path.display()))?;
    let plan: StreamingConversionPlan =
        serde_json::from_slice(&plan_bytes).context("parse streaming conversion plan")?;
    let journal: StreamingConversionJournal = serde_json::from_reader(
        File::open(&journal_path).with_context(|| format!("open {}", journal_path.display()))?,
    )
    .context("parse streaming conversion journal")?;
    let actual_hash = blake3::hash(&plan_bytes).to_hex().to_string();
    if actual_hash != journal.plan_blake3 {
        bail!("streaming conversion plan checksum does not match journal");
    }
    if absolute_output_path(&options.out_path)? != plan.out_path {
        bail!("resume output path does not match conversion plan");
    }
    if options.out_path.exists() {
        let archive = Archive::open(&options.out_path)?;
        let report = verify_archive(&archive)?;
        if !report.is_ok() {
            bail!(
                "completed archive failed verification:\n{}",
                report.errors.join("\n")
            );
        }
        return Ok(ConvertHfResult {
            archive,
            warnings: plan.warnings,
        });
    }
    run_streaming_conversion(options, plan, journal)
}

fn run_streaming_conversion(
    options: &ConvertHfOptions,
    plan: StreamingConversionPlan,
    mut journal: StreamingConversionJournal,
) -> Result<ConvertHfResult> {
    let pages = plan
        .pages
        .iter()
        .map(streaming_archive_page)
        .collect::<Result<Vec<_>>>()?;
    let mut writer = ResumableArchiveWriter::open(
        &options.out_path,
        &plan.manifest,
        &pages,
        journal.completed_pages,
    )?;
    let source_indices = source_page_indices(&plan);

    // A shard with no retained text tensors can be consumed as soon as the
    // durable plan proves that no archive page depends on it.
    if options.consume_source_shards {
        for source in &plan.source_shards {
            if !source_indices.contains_key(source) && source.exists() {
                validate_consumable_source(&plan.hf_dir, source)?;
                fs::remove_file(source).with_context(|| {
                    format!("delete excluded source shard {}", source.display())
                })?;
                journal.verified_sources.insert(source.clone());
                journal.deleted_sources.insert(source.clone());
                write_atomic_json(&conversion_journal_path(&options.out_path), &journal)?;
            }
        }
    }

    for (index, page) in pages.iter().enumerate().skip(journal.completed_pages) {
        enforce_free_space(&options.out_path, options.minimum_free_bytes, page.size)?;
        let source = plan.pages[index].source_path.clone();
        if !source.exists() {
            bail!(
                "source shard {} is missing before page {} was committed",
                source.display(),
                plan.pages[index].id
            );
        }
        writer.append(page)?;
        journal.completed_pages = index + 1;
        write_atomic_json(&conversion_journal_path(&options.out_path), &journal)?;

        let indices = source_indices
            .get(&source)
            .ok_or_else(|| anyhow!("source {} is absent from conversion plan", source.display()))?;
        if indices.last() == Some(&index) && !journal.verified_sources.contains(&source) {
            writer.verify_pages(indices)?;
            journal.verified_sources.insert(source.clone());
            write_atomic_json(&conversion_journal_path(&options.out_path), &journal)?;
            if options.consume_source_shards {
                validate_consumable_source(&plan.hf_dir, &source)?;
                fs::remove_file(&source).with_context(|| {
                    format!("delete committed source shard {}", source.display())
                })?;
                journal.deleted_sources.insert(source);
                write_atomic_json(&conversion_journal_path(&options.out_path), &journal)?;
            }
        }
    }
    writer.finish()?;
    let archive = Archive::open(&options.out_path)?;
    let report = verify_archive(&archive)?;
    if !report.is_ok() {
        bail!(
            "archive verify failed after streaming convert:\n{}",
            report.errors.join("\n")
        );
    }
    journal.completed = true;
    write_atomic_json(&conversion_journal_path(&options.out_path), &journal)?;
    Ok(ConvertHfResult {
        archive,
        warnings: plan.warnings,
    })
}

fn streaming_archive_page(page: &StreamingPage) -> Result<ArchivePage<'static>> {
    let decoded = hex::decode(&page.checksum)
        .with_context(|| format!("decode checksum for page {}", page.id))?;
    let checksum: [u8; 32] = decoded
        .try_into()
        .map_err(|_| anyhow!("page {} checksum must contain 32 bytes", page.id))?;
    Ok(ArchivePage {
        id: page.id.clone(),
        size: page.size,
        checksum,
        source: ArchivePageSource::FileRange {
            path: page.source_path.clone(),
            offset: page.source_offset,
        },
    })
}

fn source_page_indices(plan: &StreamingConversionPlan) -> BTreeMap<PathBuf, Vec<usize>> {
    let mut result: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, page) in plan.pages.iter().enumerate() {
        result
            .entry(page.source_path.clone())
            .or_default()
            .push(index);
    }
    result
}

fn validate_consumable_source(hf_dir: &Path, source: &Path) -> Result<()> {
    let canonical = fs::canonicalize(source)
        .with_context(|| format!("canonicalize source shard {}", source.display()))?;
    if !canonical.starts_with(hf_dir)
        || canonical.parent() != Some(hf_dir)
        || canonical.extension().and_then(|value| value.to_str()) != Some("safetensors")
    {
        bail!(
            "refusing to consume source outside the direct HF directory: {}",
            canonical.display()
        );
    }
    Ok(())
}

fn enforce_free_space(path: &Path, minimum_free: u64, next_page: u64) -> Result<()> {
    if minimum_free == 0 {
        return Ok(());
    }
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let available = available_space(parent)
        .with_context(|| format!("query free space for {}", parent.display()))?;
    let required = minimum_free
        .checked_add(next_page)
        .ok_or_else(|| anyhow!("minimum free-space requirement overflows u64"))?;
    if available < required {
        bail!(
            "streaming conversion free-space gate: {} bytes available, {} required before next page",
            available,
            required
        );
    }
    Ok(())
}

fn conversion_plan_path(out: &Path) -> PathBuf {
    sidecar_path(out, "conversion-plan.json")
}

fn conversion_journal_path(out: &Path) -> PathBuf {
    sidecar_path(out, "conversion-journal.json")
}

fn sidecar_path(out: &Path, suffix: &str) -> PathBuf {
    out.with_file_name(format!(
        "{}.{}",
        out.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("model.thin"),
        suffix,
    ))
}

fn absolute_output_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn write_atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("serialize conversion state")?;
    write_atomic_bytes(path, &bytes)
}

fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("conversion-state"),
    ));
    let mut file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write {}", temporary.display()))?;
    file.sync_all()
        .with_context(|| format!("sync {}", temporary.display()))?;
    drop(file);
    fs::rename(&temporary, path).with_context(|| format!("replace {}", path.display()))?;
    File::open(parent)?
        .sync_all()
        .context("sync conversion-state directory")?;
    Ok(())
}

fn effective_text_config(source: &Value) -> Value {
    let Some(text) = source.get("text_config").and_then(Value::as_object) else {
        return source.clone();
    };
    let mut config = text.clone();
    for key in ["architectures", "tie_word_embeddings"] {
        if let Some(value) = source.get(key) {
            config.insert(key.to_string(), value.clone());
        }
    }
    if let Some(model_type) = source.get("model_type") {
        config.insert("model_type".to_string(), model_type.clone());
    }
    let skipped_layers = cross_attention_layers(source).len();
    if skipped_layers > 0
        && let Some(layer_count) = optional_u64(&Value::Object(config.clone()), "num_hidden_layers")
    {
        config.insert(
            "num_hidden_layers".to_string(),
            Value::from(layer_count.saturating_sub(skipped_layers as u64)),
        );
    }
    Value::Object(config)
}

fn canonical_text_tensors(
    source_config: &Value,
    tensors: BTreeMap<String, TensorPage>,
) -> Result<(BTreeMap<String, TensorPage>, usize)> {
    if source_config.get("text_config").is_none() {
        return Ok((tensors, 0));
    }
    const PREFIXES: [&str; 2] = ["model.language_model.", "language_model."];
    let cross_layers = cross_attention_layers(source_config);
    let mut canonical = BTreeMap::new();
    let mut excluded = 0;
    for (name, mut page) in tensors {
        // Some multimodal checkpoints keep the text decoder below
        // `model.language_model.*` but store its untied execution head at the
        // repository root.  Preserve that head before filtering non-text
        // tensors; dropping it produces an archive that verifies structurally
        // but cannot execute logits.
        if name == LM_HEAD {
            page.id = LM_HEAD.to_string();
            if canonical.insert(page.id.clone(), page).is_some() {
                bail!("canonical text tensor name collision for {name}");
            }
            continue;
        }
        let Some(suffix) = PREFIXES.iter().find_map(|prefix| name.strip_prefix(prefix)) else {
            excluded += 1;
            continue;
        };
        if is_text_adapter_tensor(suffix) {
            excluded += 1;
            continue;
        }
        let Some(page_id) = canonical_text_page_id(suffix, &cross_layers) else {
            excluded += 1;
            continue;
        };
        page.id = page_id;
        if canonical.insert(page.id.clone(), page).is_some() {
            bail!("canonical text tensor name collision for {name}");
        }
    }
    if canonical.is_empty() {
        bail!(
            "text_config exists but no supported text tensor prefix was found ({})",
            PREFIXES.join(", ")
        );
    }
    Ok((canonical, excluded))
}

fn canonical_common_tensors(
    tensors: BTreeMap<String, TensorPage>,
) -> Result<BTreeMap<String, TensorPage>> {
    let mut canonical = BTreeMap::new();
    for (source_name, mut page) in tensors {
        let page_id = canonical_common_page_id(&source_name);
        page.id = page_id;
        if canonical.insert(page.id.clone(), page).is_some() {
            bail!(
                "canonical tensor name collision for {source_name}; role adapters must be unambiguous"
            );
        }
    }
    Ok(canonical)
}

fn canonical_common_page_id(name: &str) -> String {
    let globals = [
        ("transformer.wte.weight", EMBED),
        ("model.decoder.embed_tokens.weight", EMBED),
        ("transformer.ln_f.weight", FINAL_NORM),
        ("model.decoder.final_layer_norm.weight", FINAL_NORM),
        ("output.weight", LM_HEAD),
    ];
    if let Some((_, canonical)) = globals.iter().find(|(source, _)| name == *source) {
        return (*canonical).to_string();
    }
    let layer_prefixes = ["transformer.h.", "model.decoder.layers."];
    let Some(rest) = layer_prefixes
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))
    else {
        return name.to_string();
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return name.to_string();
    };
    let suffix = match suffix {
        "ln_1.weight" => "input_layernorm.weight",
        "ln_1.bias" => "input_layernorm.bias",
        "ln_2.weight" => "post_attention_layernorm.weight",
        "ln_2.bias" => "post_attention_layernorm.bias",
        "attn.c_attn.weight" => "self_attn.qkv_proj.weight",
        "attn.c_attn.bias" => "self_attn.qkv_proj.bias",
        "attn.c_proj.weight" => "self_attn.o_proj.weight",
        "attn.c_proj.bias" => "self_attn.o_proj.bias",
        other => other,
    };
    format!("model.layers.{layer}.{suffix}")
}

fn cross_attention_layers(source_config: &Value) -> BTreeSet<u32> {
    source_config
        .get("text_config")
        .and_then(|text| text.get("cross_attention_layers"))
        .and_then(Value::as_array)
        .map(|layers| {
            layers
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|layer| u32::try_from(layer).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn canonical_text_page_id(suffix: &str, cross_layers: &BTreeSet<u32>) -> Option<String> {
    if suffix == "lm_head.weight" {
        return Some(suffix.to_string());
    }
    let normalized = if suffix.starts_with("model.") {
        suffix.to_string()
    } else {
        format!("model.{suffix}")
    };
    let Some(rest) = normalized.strip_prefix("model.layers.") else {
        return Some(normalized);
    };
    let Some((layer_text, layer_suffix)) = rest.split_once('.') else {
        return Some(normalized);
    };
    let Ok(layer) = layer_text.parse::<u32>() else {
        return Some(normalized);
    };
    if cross_layers.contains(&layer) {
        return None;
    }
    let skipped_before = cross_layers
        .iter()
        .filter(|skipped| **skipped < layer)
        .count() as u32;
    let compact_layer = layer.saturating_sub(skipped_before);
    Some(format!("model.layers.{compact_layer}.{layer_suffix}"))
}

fn is_text_adapter_tensor(suffix: &str) -> bool {
    suffix.contains(".cross_attn.")
        || suffix.contains(".cross_attn_attn_gate")
        || suffix.contains(".cross_attn_mlp_gate")
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
            let checksum = *blake3::hash(data).as_bytes();
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
                checksum,
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
    let source_dtype =
        optional_string(config, "torch_dtype").or_else(|| optional_string(config, "dtype"));
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
    let norm_operator = norm_operator_from_config(config);
    let mut required_operators = vec![
        "embedding".to_string(),
        norm_operator.to_string(),
        "qkv_projection".to_string(),
        "rope".to_string(),
        format!("{attention_kind}_attention"),
        "o_projection".to_string(),
        "lm_head".to_string(),
    ];
    let has_linear_attention = config
        .get("layer_types")
        .and_then(Value::as_array)
        .is_some_and(|values| {
            values
                .iter()
                .any(|value| value.as_str() == Some("linear_attention"))
        });
    if has_linear_attention {
        required_operators.extend([
            "gated_delta_net".to_string(),
            "depthwise_causal_conv1d".to_string(),
            "recurrent_state_cache".to_string(),
            "gated_full_attention".to_string(),
        ]);
    }
    let has_per_layer_semantics = [
        "hidden_size_per_layer_input",
        "vocab_size_per_layer_input",
        "num_kv_shared_layers",
        "global_head_dim",
    ]
    .iter()
    .any(|field| optional_u64(config, field).is_some_and(|value| value > 0));
    if has_per_layer_semantics {
        required_operators.extend([
            "per_layer_embeddings".to_string(),
            "variable_head_dim_attention".to_string(),
            "shared_kv_attention".to_string(),
            "post_attention_norm".to_string(),
            "post_feedforward_norm".to_string(),
        ]);
    }
    let has_dual_post_norm = tensors
        .keys()
        .any(|name| name.ends_with("pre_feedforward_layernorm.weight"))
        && tensors
            .keys()
            .any(|name| name.ends_with("post_feedforward_layernorm.weight"));
    if has_dual_post_norm {
        required_operators.extend([
            "post_attention_norm".to_string(),
            "post_feedforward_norm".to_string(),
        ]);
    }
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
        intermediate_size: optional_u64(config, "intermediate_size")
            .or_else(|| infer_intermediate_size(tensors)),
        rms_norm_eps: optional_f64(config, "rms_norm_eps").or_else(|| {
            (config.get("model_type").and_then(Value::as_str) == Some("olmoe")).then_some(1e-5)
        }),
        norm_kind: if optional_f64(config, "rms_norm_eps").is_some()
            || config.get("model_type").and_then(Value::as_str) == Some("olmoe")
        {
            Some("rms_norm".to_string())
        } else if optional_f64(config, "layer_norm_eps").is_some() {
            Some("layer_norm".to_string())
        } else {
            None
        },
        norm_eps: optional_f64(config, "rms_norm_eps")
            .or_else(|| optional_f64(config, "layer_norm_eps"))
            .or_else(|| {
                (config.get("model_type").and_then(Value::as_str) == Some("olmoe")).then_some(1e-5)
            }),
        rope_theta: optional_f64(config, "rope_theta").or_else(|| {
            config
                .get("rope_parameters")
                .and_then(|value| value.get("rope_theta"))
                .and_then(Value::as_f64)
        }),
        partial_rotary_factor: optional_f64(config, "partial_rotary_factor").or_else(|| {
            config
                .get("rope_parameters")
                .and_then(|value| value.get("partial_rotary_factor"))
                .and_then(Value::as_f64)
        }),
        rope_scaling: config
            .get("rope_scaling")
            .filter(|value| !value.is_null())
            .cloned(),
        source_dtype,
        vocab_size: optional_u64(config, "vocab_size"),
        tie_word_embeddings: Some(
            config
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| {
                    matches!(
                        config.get("model_type").and_then(Value::as_str),
                        Some("gemma" | "gemma2")
                    )
                }),
        ),
        activation: optional_string(config, "hidden_act")
            .or_else(|| optional_string(config, "hidden_activation")),
        qkv_bias: config
            .get("qkv_bias")
            .or_else(|| config.get("attention_bias"))
            .and_then(Value::as_bool),
        attention_bias: config.get("attention_bias").and_then(Value::as_bool),
        tensor_naming_scheme: Some("hf_decoder_layers".to_string()),
        attention_variants: {
            let mut variants = vec!["causal_kv".to_string(), "current_only_smoke".to_string()];
            let has_layer_types = config
                .get("layer_types")
                .and_then(Value::as_array)
                .is_some_and(|values| !values.is_empty());
            let uses_sliding_window = config
                .get("use_sliding_window")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            if has_layer_types
                || optional_u64(config, "sliding_window").is_some() && uses_sliding_window
            {
                variants.push("sliding_causal_kv".to_string());
            }
            variants
        },
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
        original_max_position_embeddings: optional_u64(config, "original_max_position_embeddings"),
        rope_variant: config
            .get("rope_scaling")
            .and_then(|value| value.get("rope_type").or_else(|| value.get("type")))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        architecture_family: Some(
            if has_linear_attention {
                "hybrid_decoder"
            } else if is_moe {
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
            .unwrap_or_else(|| {
                if config.get("model_type").and_then(Value::as_str) == Some("gemma2") {
                    (0..layers)
                        .map(|layer| {
                            if layer % 2 == 0 {
                                "sliding_attention"
                            } else {
                                "full_attention"
                            }
                            .to_string()
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            }),
        linear_conv_kernel_dim: optional_u64(config, "linear_conv_kernel_dim"),
        linear_key_head_dim: optional_u64(config, "linear_key_head_dim"),
        linear_value_head_dim: optional_u64(config, "linear_value_head_dim"),
        linear_num_key_heads: optional_u64(config, "linear_num_key_heads"),
        linear_num_value_heads: optional_u64(config, "linear_num_value_heads"),
        attention_output_gate: config.get("attn_output_gate").and_then(Value::as_bool),
        global_head_dim: optional_u64(config, "global_head_dim"),
        num_global_key_value_heads: optional_u64(config, "num_global_key_value_heads"),
        num_kv_shared_layers: optional_u64(config, "num_kv_shared_layers"),
        hidden_size_per_layer_input: optional_u64(config, "hidden_size_per_layer_input"),
        vocab_size_per_layer_input: optional_u64(config, "vocab_size_per_layer_input"),
        use_double_wide_mlp: config.get("use_double_wide_mlp").and_then(Value::as_bool),
        attention_sinks: Some(attention_sinks),
        num_local_experts,
        num_experts_per_token,
        norm_topk_prob: config.get("norm_topk_prob").and_then(Value::as_bool),
        swiglu_alpha: optional_f64(config, "swiglu_alpha"),
        swiglu_limit: optional_f64(config, "swiglu_limit"),
        norm_weight_offset: if matches!(
            config.get("model_type").and_then(Value::as_str),
            Some("gemma2" | "qwen3_5")
        ) {
            Some(1.0)
        } else {
            None
        },
        embedding_scale: if config
            .get("model_type")
            .and_then(Value::as_str)
            .is_some_and(|value| value == "gemma2" || value.starts_with("gemma4"))
        {
            Some((hidden_size as f64).sqrt())
        } else {
            None
        },
        query_pre_attn_scalar: optional_f64(config, "query_pre_attn_scalar"),
        attention_logit_softcap: optional_f64(config, "attn_logit_softcapping"),
        final_logit_softcap: optional_f64(config, "final_logit_softcapping"),
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
    config: &Value,
    warnings: &mut Vec<String>,
) -> Result<Vec<ExecutionStage>> {
    if !tensors.contains_key(EMBED) {
        bail!("required tensor {EMBED} missing");
    }

    let mut stages = Vec::new();
    stages.push(stage("embed", vec![EMBED.to_string()]));
    let per_layer_input_refs = [
        "model.embed_tokens_per_layer.weight",
        "model.per_layer_model_projection.weight",
        "model.per_layer_projection_norm.weight",
    ];
    if per_layer_input_refs
        .iter()
        .all(|name| tensors.contains_key(*name))
    {
        stages.push(stage(
            "per_layer_inputs",
            per_layer_input_refs
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        ));
    }

    let mut missing_q_norm = 0_u32;
    let mut missing_k_norm = 0_u32;
    for layer in 0..layers {
        let is_linear_attention = config
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|values| values.get(layer as usize))
            .and_then(Value::as_str)
            == Some("linear_attention");
        let first_kv_shared_layer = layers.saturating_sub(
            optional_u64(config, "num_kv_shared_layers")
                .unwrap_or(0)
                .try_into()
                .unwrap_or(layers),
        );
        let is_kv_shared = layer >= first_kv_shared_layer;
        let input_norm = layer_tensor(layer, "input_layernorm.weight");
        let q_proj = layer_tensor(layer, "self_attn.q_proj.weight");
        let k_proj = layer_tensor(layer, "self_attn.k_proj.weight");
        let v_proj = layer_tensor(layer, "self_attn.v_proj.weight");
        let fused_qkv = layer_tensor(layer, "self_attn.qkv_proj.weight");
        let q_norm = layer_tensor(layer, "self_attn.q_norm.weight");
        let k_norm = layer_tensor(layer, "self_attn.k_norm.weight");
        let o_proj = layer_tensor(layer, "self_attn.o_proj.weight");
        let post_norm = layer_tensor(layer, "post_attention_layernorm.weight");
        let pre_feedforward_norm = layer_tensor(layer, "pre_feedforward_layernorm.weight");
        let post_feedforward_norm = layer_tensor(layer, "post_feedforward_layernorm.weight");
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
        let fused_qkv_refs = if is_linear_attention {
            None
        } else {
            tensor_group(tensors, &fused_qkv).ok()
        };
        let separate_qkv_refs = if !is_linear_attention && !is_kv_shared && fused_qkv_refs.is_none()
        {
            Some([
                tensor_group(tensors, &q_proj)?,
                tensor_group(tensors, &k_proj)?,
                tensor_group(tensors, &v_proj)?,
            ])
        } else {
            None
        };
        let o_projection_refs = if is_linear_attention {
            Vec::new()
        } else {
            tensor_group(tensors, &o_proj)?
        };
        require_tensor(tensors, &post_norm)?;

        let mut input_norm_refs = vec![input_norm.clone()];
        push_if_present(
            tensors,
            &mut input_norm_refs,
            layer_tensor(layer, "input_layernorm.bias"),
        );
        stages.push(stage(format!("layer_{layer}_input_norm"), input_norm_refs));

        if is_linear_attention {
            let mut refs = Vec::new();
            for suffix in [
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.A_log",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ] {
                let name = layer_tensor(layer, suffix);
                require_tensor(tensors, &name)?;
                refs.push(name);
            }
            stages.push(stage(format!("layer_{layer}_linear_attention"), refs));
        } else {
            let mut qkv_refs = if let Some(refs) = fused_qkv_refs {
                refs
            } else if is_kv_shared {
                tensor_group(tensors, &q_proj)?
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
            } else if !is_kv_shared {
                missing_k_norm += 1;
            }
            stages.push(stage(format!("layer_{layer}_attn_qkv"), qkv_refs));
            stages.push(stage(format!("layer_{layer}_rope"), Vec::new()));
            let hidden_size = optional_u64(config, "hidden_size")
                .or_else(|| infer_hidden_size(tensors))
                .unwrap_or(1);
            let heads = optional_u64(config, "num_attention_heads")
                .or_else(|| infer_attention_heads(config, hidden_size).map(u64::from))
                .unwrap_or(1);
            let kv_heads = optional_u64(config, "num_key_value_heads")
                .or_else(|| infer_kv_heads(config, tensors).map(u64::from))
                .unwrap_or(heads);
            let head_dim = optional_u64(config, "head_dim")
                .unwrap_or_else(|| hidden_size.saturating_div(heads).max(1));
            let attention_operator = if kv_heads == heads {
                "mha_attention"
            } else if kv_heads == 1 {
                "mqa_attention"
            } else {
                "gqa_attention"
            };
            let mut params = BTreeMap::from([
                ("causal".to_string(), Value::Bool(true)),
                ("heads".to_string(), Value::from(heads)),
                ("kv_heads".to_string(), Value::from(kv_heads)),
                ("head_dim".to_string(), Value::from(head_dim)),
            ]);
            let layer_type = config
                .get("layer_types")
                .and_then(Value::as_array)
                .and_then(|values| values.get(layer as usize))
                .and_then(Value::as_str);
            if layer_type.is_some_and(|kind| kind.contains("sliding"))
                && let Some(window) = optional_u64(config, "sliding_window")
            {
                params.insert("window".to_string(), Value::from(window));
            }
            stages.push(operator_stage(
                format!("layer_{layer}_attention"),
                attention_operator,
                params,
                Vec::new(),
            ));
        }
        if !is_linear_attention {
            let mut o_refs = o_projection_refs;
            let o_bias = layer_tensor(layer, "self_attn.o_proj.bias");
            if tensors.contains_key(&o_bias) {
                o_refs.push(o_bias);
            }
            stages.push(stage(format!("layer_{layer}_attn_out"), o_refs));
        }
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
        if tensors.contains_key(&pre_feedforward_norm) {
            stages.push(stage(
                format!("layer_{layer}_pre_feedforward_norm"),
                vec![pre_feedforward_norm],
            ));
        }
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
        if tensors.contains_key(&post_feedforward_norm) {
            stages.push(stage(
                format!("layer_{layer}_post_feedforward_norm"),
                vec![post_feedforward_norm],
            ));
        }
        let per_layer_gate = layer_tensor(layer, "per_layer_input_gate.weight");
        let per_layer_projection = layer_tensor(layer, "per_layer_projection.weight");
        let post_per_layer_norm = layer_tensor(layer, "post_per_layer_input_norm.weight");
        let layer_scalar = layer_tensor(layer, "layer_scalar");
        if tensors.contains_key(&per_layer_gate)
            && tensors.contains_key(&per_layer_projection)
            && tensors.contains_key(&post_per_layer_norm)
        {
            let mut refs = vec![per_layer_gate, per_layer_projection, post_per_layer_norm];
            push_if_present(tensors, &mut refs, layer_scalar);
            stages.push(stage(format!("layer_{layer}_per_layer_input"), refs));
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
        push_if_present(tensors, &mut final_norm_refs, "model.norm.bias".to_string());
        stages.push(stage("final_norm", final_norm_refs));
    }
    if tensors.contains_key(LM_HEAD) {
        stages.push(stage("lm_head", vec![LM_HEAD.to_string()]));
    } else {
        warnings.push("lm_head.weight missing; continuing without lm_head stage".to_string());
    }

    let norm_operator = norm_operator_from_config(config);
    let norm_eps = optional_f64(config, "rms_norm_eps")
        .or_else(|| optional_f64(config, "layer_norm_eps"))
        .or_else(|| {
            (config.get("model_type").and_then(Value::as_str) == Some("olmoe")).then_some(1e-5)
        });
    for stage in &mut stages {
        if stage.operator == "rms_norm" {
            stage.operator = norm_operator.to_string();
            if let Some(eps) = norm_eps {
                stage
                    .operator_params
                    .insert("eps".to_string(), Value::from(eps));
            }
        }
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
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
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

fn build_memory_plan(
    config: &Value,
    tensors: &BTreeMap<String, TensorPage>,
    execution_tape: &[ExecutionStage],
) -> MemoryPlan {
    let weights_bytes = unique_tensor_bytes(tensors.values());
    let hidden_size = optional_u64(config, "hidden_size")
        .or_else(|| infer_hidden_size(tensors))
        .unwrap_or(4096);
    let intermediate_size = optional_u64(config, "intermediate_size")
        .or_else(|| infer_intermediate_size(tensors))
        .unwrap_or(hidden_size.saturating_mul(4));
    let scratch_bytes = estimate_live_runtime_bytes(config, hidden_size, intermediate_size);

    // Globals (especially embeddings and the execution head) stay exact and hot. Layer
    // stages may stream independently, so minimum VRAM is the hot set plus the largest
    // stage working set rather than the size of every model weight.
    let global_bytes = unique_tensor_bytes(
        tensors
            .values()
            .filter(|tensor| layer_from_tensor(&tensor.id).is_none()),
    );
    let largest_stage_bytes = execution_tape
        .iter()
        .map(|stage| {
            unique_tensor_bytes(
                stage
                    .page_refs
                    .iter()
                    .filter_map(|id| tensors.get(id))
                    // Global tensors are already included in `global_bytes`.
                    // Counting the embedding/head stage again can make the
                    // advertised minimum exceed the all-resident plan.
                    .filter(|tensor| layer_from_tensor(&tensor.id).is_some()),
            )
        })
        .max()
        .unwrap_or(0);
    let streaming_working_set_bytes = global_bytes
        .saturating_add(largest_stage_bytes)
        .saturating_add(scratch_bytes);
    let recommended_vram_bytes = weights_bytes.saturating_add(scratch_bytes);
    // The hot global set and the largest stage are deduplicated separately.
    // A tied/aliased tensor can therefore occur in both subtotals even though
    // the all-resident plan counts it once. The streaming minimum cannot be
    // larger than that verified all-resident upper bound.
    let min_vram_bytes = streaming_working_set_bytes.min(recommended_vram_bytes);

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

fn unique_tensor_bytes<'a>(tensors: impl IntoIterator<Item = &'a TensorPage>) -> u64 {
    let mut seen = BTreeSet::new();
    tensors.into_iter().fold(0_u64, |total, tensor| {
        if seen.insert((tensor.size, tensor.checksum)) {
            total.saturating_add(tensor.size)
        } else {
            total
        }
    })
}

fn estimate_live_runtime_bytes(config: &Value, hidden_size: u64, intermediate_size: u64) -> u64 {
    let heads = optional_u64(config, "num_attention_heads")
        .unwrap_or(1)
        .max(1);
    let kv_heads = optional_u64(config, "num_key_value_heads")
        .unwrap_or(heads)
        .max(1);
    let head_dim = optional_u64(config, "head_dim")
        .unwrap_or_else(|| hidden_size.saturating_div(heads).max(1));
    let qkv_elements =
        hidden_size.saturating_add(2_u64.saturating_mul(kv_heads).saturating_mul(head_dim));
    let attention_elements = qkv_elements.saturating_add(3_u64.saturating_mul(hidden_size));

    let top_k = optional_u64(config, "num_experts_per_token")
        .or_else(|| optional_u64(config, "experts_per_token"))
        .or_else(|| optional_u64(config, "num_experts_per_tok"))
        .or_else(|| optional_u64(config, "num_selected_experts"))
        .unwrap_or(1)
        .max(1);
    let experts = optional_u64(config, "num_local_experts")
        .or_else(|| optional_u64(config, "num_experts"))
        .or_else(|| optional_u64(config, "n_routed_experts"))
        .unwrap_or(0);
    let mlp_elements = if experts > 0 {
        top_k
            .saturating_mul(
                3_u64
                    .saturating_mul(intermediate_size)
                    .saturating_add(hidden_size),
            )
            .saturating_add(2_u64.saturating_mul(hidden_size))
            .saturating_add(experts)
    } else {
        3_u64
            .saturating_mul(intermediate_size)
            .saturating_add(2_u64.saturating_mul(hidden_size))
    };

    let linear_layers = config
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|types| {
            types
                .iter()
                .filter(|kind| kind.as_str() == Some("linear_attention"))
                .count() as u64
        })
        .unwrap_or(0);
    let key_heads = optional_u64(config, "linear_num_key_heads").unwrap_or(0);
    let value_heads = optional_u64(config, "linear_num_value_heads").unwrap_or(0);
    let key_dim = optional_u64(config, "linear_key_head_dim").unwrap_or(0);
    let value_dim = optional_u64(config, "linear_value_head_dim").unwrap_or(0);
    let conv_kernel = optional_u64(config, "linear_conv_kernel_dim").unwrap_or(0);
    let conv_dim = 2_u64
        .saturating_mul(key_heads)
        .saturating_mul(key_dim)
        .saturating_add(value_heads.saturating_mul(value_dim));
    let linear_state_bytes = linear_layers
        .saturating_mul(conv_dim.saturating_mul(conv_kernel).saturating_mul(2))
        .saturating_add(
            linear_layers
                .saturating_mul(value_heads)
                .saturating_mul(key_dim)
                .saturating_mul(value_dim)
                .saturating_mul(4),
        );

    let workspace_bytes = attention_elements
        .max(mlp_elements)
        .saturating_mul(2)
        .saturating_add(linear_state_bytes);
    align_up(workspace_bytes.max(64 * 1024 * 1024), 2 * 1024 * 1024)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value
        .saturating_add(alignment.saturating_sub(1))
        .saturating_div(alignment)
        .saturating_mul(alignment)
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

fn infer_intermediate_size(tensors: &BTreeMap<String, TensorPage>) -> Option<u64> {
    for (id, divisor, dimension) in [
        (layer_tensor(0, "mlp.gate_proj.weight"), 1, 0),
        (layer_tensor(0, "mlp.gate_up_proj.weight"), 2, 0),
        (layer_tensor(0, "mlp.experts.gate_up_proj_blocks"), 2, 1),
        (layer_tensor(0, "mlp.experts.0.gate_proj.weight"), 1, 0),
        (
            layer_tensor(0, "block_sparse_moe.experts.0.w1.weight"),
            1,
            0,
        ),
    ] {
        if let Some(size) = tensors
            .get(&id)
            .and_then(|tensor| tensor.shape.get(dimension))
            .copied()
            && size % divisor == 0
        {
            return Some(size / divisor);
        }
    }
    None
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
    let stage = stage.into();
    ExecutionStage {
        operator: operator_from_stage(&stage).to_string(),
        operator_params: BTreeMap::new(),
        stage,
        page_refs,
    }
}

fn operator_stage(
    stage: impl Into<String>,
    operator: impl Into<String>,
    operator_params: BTreeMap<String, Value>,
    page_refs: Vec<String>,
) -> ExecutionStage {
    ExecutionStage {
        stage: stage.into(),
        operator: operator.into(),
        operator_params,
        page_refs,
    }
}

fn operator_from_stage(stage: &str) -> &'static str {
    if stage == "embed" {
        "embedding"
    } else if stage == "lm_head" {
        "lm_head"
    } else if stage.contains("linear_attention") {
        "gated_delta_net"
    } else if stage.ends_with("_attn_qkv") {
        "qkv_projection"
    } else if stage.ends_with("_rope") {
        "rope"
    } else if stage.ends_with("_attn_out") {
        "o_projection"
    } else if stage.ends_with("_moe_router") {
        "topk_router"
    } else if stage.ends_with("_moe_gate_up")
        || stage.ends_with("_moe_down")
        || stage.ends_with("_moe_experts")
    {
        "sparse_experts"
    } else if stage.ends_with("_mlp_gate_up") {
        "gated_activation"
    } else if stage.ends_with("_mlp_down") {
        "down_projection"
    } else if stage.contains("norm") {
        "rms_norm"
    } else if stage.contains("per_layer") {
        "per_layer_embeddings"
    } else {
        "unknown"
    }
}

fn norm_operator_from_config(config: &Value) -> &'static str {
    if optional_f64(config, "rms_norm_eps").is_none()
        && optional_f64(config, "layer_norm_eps").is_some()
    {
        "layer_norm"
    } else {
        "rms_norm"
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
    } else if name.contains("input_layernorm.") || name.contains(".ln_1.") {
        "input_layernorm"
    } else if name.contains("self_attn.qkv_proj.") || name.contains(".attn.c_attn.") {
        "attn_qkv_proj"
    } else if name.contains("self_attn.q_proj.") {
        "attn_q_proj"
    } else if name.contains("self_attn.k_proj.") {
        "attn_k_proj"
    } else if name.contains("self_attn.v_proj.") {
        "attn_v_proj"
    } else if name.contains("self_attn.o_proj.") || name.contains(".attn.c_proj.") {
        "attn_o_proj"
    } else if name.ends_with("q_norm.weight") {
        "attn_q_norm"
    } else if name.ends_with("k_norm.weight") {
        "attn_k_norm"
    } else if name.contains("post_attention_layernorm.") || name.contains(".ln_2.") {
        "post_attention_layernorm"
    } else if name.ends_with("pre_feedforward_layernorm.weight") {
        "pre_feedforward_layernorm"
    } else if name.ends_with("post_feedforward_layernorm.weight") {
        "post_feedforward_layernorm"
    } else if name == "model.embed_tokens_per_layer.weight" {
        "per_layer_token_embeddings"
    } else if name == "model.per_layer_model_projection.weight" {
        "per_layer_model_projection"
    } else if name == "model.per_layer_projection_norm.weight" {
        "per_layer_projection_norm"
    } else if name.ends_with("per_layer_input_gate.weight") {
        "per_layer_input_gate"
    } else if name.ends_with("per_layer_projection.weight") {
        "per_layer_projection"
    } else if name.ends_with("post_per_layer_input_norm.weight") {
        "post_per_layer_input_norm"
    } else if name.ends_with("layer_scalar") {
        "layer_scalar"
    } else if name.contains(".linear_attn.in_proj_") {
        "linear_attn_input_projection"
    } else if name.ends_with(".linear_attn.conv1d.weight") {
        "linear_attn_depthwise_conv"
    } else if name.ends_with(".linear_attn.A_log") || name.ends_with(".linear_attn.dt_bias") {
        "linear_attn_recurrence_parameters"
    } else if name.ends_with(".linear_attn.norm.weight") {
        "linear_attn_gated_norm"
    } else if name.ends_with(".linear_attn.out_proj.weight") {
        "linear_attn_output_projection"
    } else if name.ends_with("self_attn.sinks") {
        "attention_sinks"
    } else if name.contains("mlp.router.") || name.contains("block_sparse_moe.gate.") {
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
    } else if name.contains("mlp.gate_up_proj.") {
        "mlp_gate_up_proj"
    } else if name.contains("mlp.gate_proj.") {
        "mlp_gate_proj"
    } else if name.contains("mlp.up_proj.") {
        "mlp_up_proj"
    } else if name.contains("mlp.down_proj.") {
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
    if lower.contains("qwen3_5") || lower.contains("qwen3.5") {
        "qwen3_5".to_string()
    } else if lower.contains("qwen") {
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
    let aliases: &[&str] = match key {
        "hidden_size" => &["hidden_size", "n_embd", "d_model"],
        "intermediate_size" => &[
            "intermediate_size",
            "ffn_dim",
            "ffn_hidden_size",
            "n_inner",
            "d_ff",
            "moe_intermediate_size",
            "expert_intermediate_size",
        ],
        "num_hidden_layers" => &["num_hidden_layers", "n_layer", "num_layers"],
        "num_attention_heads" => &["num_attention_heads", "n_head", "num_heads"],
        "num_key_value_heads" => &["num_key_value_heads", "num_kv_heads", "n_head_kv"],
        "vocab_size" => &["vocab_size", "n_vocab", "padded_vocab_size"],
        "max_position_embeddings" => &[
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "seq_length",
        ],
        _ => return config.get(key).and_then(Value::as_u64),
    };
    aliases
        .iter()
        .find_map(|alias| config.get(*alias).and_then(Value::as_u64))
}

fn optional_u32(config: &Value, key: &str) -> Option<u32> {
    optional_u64(config, key).and_then(|value| u32::try_from(value).ok())
}

fn optional_f64(config: &Value, key: &str) -> Option<f64> {
    match key {
        "layer_norm_eps" => config
            .get("layer_norm_eps")
            .or_else(|| config.get("layer_norm_epsilon"))
            .and_then(Value::as_f64),
        _ => config.get(key).and_then(Value::as_f64),
    }
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
