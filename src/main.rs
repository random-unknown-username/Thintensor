use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command as ProcessCommand;
use std::str::FromStr;
use thintensor::archive::{PackOptions, pack_archive};
use thintensor::bench::{BenchLoadResult, BenchPlanResult, bench_load, bench_plan};
use thintensor::convert_hf::{ConvertHfOptions, convert_hf};
use thintensor::plan::{PlanStatus, WeightResidency, build_plan};
use thintensor::profile::{ProfileOptions, RuntimeProfile, build_profile};
use thintensor::repack::{RepackOptions, repack_archive};
use thintensor::simulate::{LoadSimulation, simulate_load};
use thintensor::stats::{ArchiveStats, build_stats};
use thintensor::units::{format_bytes, parse_bytes};
use thintensor::verify::verify_archive;
use thintensor::{Archive, Plan, Report};

#[derive(Debug, Parser)]
#[command(name = "thintensor-core")]
#[command(about = "Internal archive core used by the thintensor CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Pack {
        manifest: PathBuf,
        pages_dir: PathBuf,
        out: PathBuf,
    },
    Inspect {
        archive: PathBuf,
    },
    Stats {
        archive: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Verify {
        archive: PathBuf,
    },
    Extract {
        archive: PathBuf,
        out_dir: PathBuf,
    },
    Plan {
        archive: PathBuf,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        vram: String,
        #[arg(long, default_value_t = 4096)]
        ctx: u64,
        #[arg(long, default_value_t = 1)]
        batch: u64,
        #[arg(long = "kv-dtype", default_value = "q4")]
        kv_dtype: String,
        #[arg(long = "weight-residency", default_value = "all")]
        weight_residency: String,
        #[arg(long = "offload-layers", default_value_t = 0)]
        offload_layers: u32,
        #[arg(long = "gpu-fraction", default_value_t = 1.0)]
        gpu_fraction: f64,
        #[arg(long)]
        json: bool,
    },
    Profile {
        archive: PathBuf,
        #[arg(long = "target-vram")]
        target_vram: String,
        #[arg(long, default_value = "cuda")]
        backend: String,
        #[arg(long, default_value_t = 4096)]
        ctx: u64,
        #[arg(long, default_value_t = 1)]
        batch: u64,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    SimulateLoad {
        archive: PathBuf,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        vram: String,
        #[arg(long, default_value_t = 4096)]
        ctx: u64,
        #[arg(long, default_value_t = 1)]
        batch: u64,
        #[arg(long = "kv-dtype", default_value = "q4")]
        kv_dtype: String,
        #[arg(long = "weight-residency", default_value = "all")]
        weight_residency: String,
        #[arg(long = "offload-layers", default_value_t = 0)]
        offload_layers: u32,
        #[arg(long = "gpu-fraction", default_value_t = 1.0)]
        gpu_fraction: f64,
        #[arg(long = "prefetch-pages", default_value_t = 4)]
        prefetch_pages: usize,
        #[arg(long)]
        json: bool,
    },
    ConvertHf {
        hf_dir: PathBuf,
        out: PathBuf,
        #[arg(long)]
        arch: Option<String>,
        #[arg(long)]
        no_tokenizer: bool,
    },
    BenchLoad {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    BenchPlan {
        archive: PathBuf,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        vram: String,
        #[arg(long, default_value_t = 4096)]
        ctx: u64,
        #[arg(long, default_value_t = 1)]
        batch: u64,
        #[arg(long = "kv-dtype", default_value = "q4")]
        kv_dtype: String,
        #[arg(long = "weight-residency", default_value = "all")]
        weight_residency: String,
        #[arg(long = "offload-layers", default_value_t = 0)]
        offload_layers: u32,
        #[arg(long = "gpu-fraction", default_value_t = 1.0)]
        gpu_fraction: f64,
        #[arg(long)]
        json: bool,
    },
    BenchBaseline {
        hf_dir: PathBuf,
        #[arg(
            long,
            default_value = "Write one short paragraph about tensor layouts."
        )]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        tokens: u32,
        #[arg(long, default_value_t = 2048)]
        ctx: u32,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = 87)]
        max_gpu_temp: u32,
        #[arg(long)]
        json: bool,
    },
    BenchArchive {
        archive: PathBuf,
        #[arg(long = "hf-dir")]
        hf_dir: PathBuf,
        #[arg(
            long,
            default_value = "Write one short paragraph about tensor layouts."
        )]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        tokens: u32,
        #[arg(long, default_value_t = 2048)]
        ctx: u32,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value_t = 87)]
        max_gpu_temp: u32,
        #[arg(long)]
        json: bool,
    },
    Repack {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "execution_ordered_v1")]
        layout: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Pack {
            manifest,
            pages_dir,
            out,
        } => {
            let archive = pack_archive(PackOptions {
                manifest_path: manifest,
                pages_dir,
                out_path: out.clone(),
            })?;
            println!(
                "packed {} pages into {}",
                archive.records().len(),
                out.display()
            );
        }
        Command::Inspect { archive } => {
            let archive = Archive::open(archive)?;
            print_inspect(&archive);
        }
        Command::Stats { archive, json } => {
            let archive = Archive::open(archive)?;
            let report = verify_archive(&archive)?;
            if !report.is_ok() {
                print_report(&report);
                bail!("cannot stat invalid archive");
            }
            let stats = build_stats(&archive);
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &stats)?;
                println!();
            } else {
                print_stats(&stats);
            }
        }
        Command::Verify { archive } => {
            let archive = Archive::open(archive)?;
            let report = verify_archive(&archive)?;
            print_report(&report);
            if !report.is_ok() {
                bail!("archive verification failed");
            }
        }
        Command::Extract { archive, out_dir } => {
            let archive = Archive::open(archive)?;
            let report = verify_archive(&archive)?;
            if !report.is_ok() {
                print_report(&report);
                bail!("cannot extract invalid archive");
            }
            archive.extract(&out_dir)?;
            println!("extracted archive into {}", out_dir.display());
        }
        Command::Plan {
            archive,
            backend,
            vram,
            ctx,
            batch,
            kv_dtype,
            weight_residency,
            offload_layers,
            gpu_fraction,
            json,
        } => {
            let archive = Archive::open(archive)?;
            let report = verify_archive(&archive)?;
            if !report.is_ok() {
                print_report(&report);
                bail!("cannot plan invalid archive");
            }
            let plan = build_plan(
                archive.manifest(),
                thintensor::plan::PlanOptions {
                    backend,
                    vram_bytes: parse_bytes(&vram)?,
                    ctx_tokens: ctx,
                    batch_size: batch,
                    kv_dtype: parse_kv_dtype(&kv_dtype)?,
                    weight_residency: parse_weight_residency(&weight_residency)?,
                    offload_layers,
                    gpu_fraction: parse_gpu_fraction(gpu_fraction)?,
                },
            );
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &plan)?;
                println!();
            } else {
                print_plan(&plan);
            }
        }
        Command::Profile {
            archive,
            target_vram,
            backend,
            ctx,
            batch,
            out,
            json,
        } => {
            let archive_path = archive;
            let archive = Archive::open(&archive_path)?;
            let report = verify_archive(&archive)?;
            if !report.is_ok() {
                print_report(&report);
                bail!("cannot profile invalid archive");
            }
            let profile = build_profile(
                &archive,
                ProfileOptions {
                    backend,
                    target_vram_bytes: parse_bytes(&target_vram)?,
                    ctx_tokens: ctx,
                    batch_size: batch,
                },
            );
            let out_path = out.unwrap_or_else(|| default_profile_path(&archive_path, &target_vram));
            let file = std::fs::File::create(&out_path)?;
            serde_json::to_writer_pretty(file, &profile)?;
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &profile)?;
                println!();
            } else {
                print_profile(&profile, &out_path);
            }
        }
        Command::SimulateLoad {
            archive,
            backend,
            vram,
            ctx,
            batch,
            kv_dtype,
            weight_residency,
            offload_layers,
            gpu_fraction,
            prefetch_pages,
            json,
        } => {
            let archive = Archive::open(&archive)?;
            let report = verify_archive(&archive)?;
            if !report.is_ok() {
                print_report(&report);
                bail!("cannot simulate invalid archive");
            }
            let simulation = simulate_load(
                &archive,
                thintensor::plan::PlanOptions {
                    backend,
                    vram_bytes: parse_bytes(&vram)?,
                    ctx_tokens: ctx,
                    batch_size: batch,
                    kv_dtype: parse_kv_dtype(&kv_dtype)?,
                    weight_residency: parse_weight_residency(&weight_residency)?,
                    offload_layers,
                    gpu_fraction: parse_gpu_fraction(gpu_fraction)?,
                },
                prefetch_pages,
            );
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &simulation)?;
                println!();
            } else {
                print_simulation(&simulation);
            }
        }
        Command::ConvertHf {
            hf_dir,
            out,
            arch,
            no_tokenizer,
        } => {
            let result = convert_hf(ConvertHfOptions {
                hf_dir,
                out_path: out.clone(),
                arch_override: arch,
                include_tokenizer_hashes: !no_tokenizer,
            })?;
            for warning in &result.warnings {
                eprintln!("warning: {warning}");
            }
            println!(
                "converted {} pages into {}",
                result.archive.records().len(),
                out.display()
            );
        }
        Command::BenchLoad { input, json } => {
            let result = bench_load(&input)?;
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &result)?;
                println!();
            } else {
                print_bench_load(&result);
            }
        }
        Command::BenchPlan {
            archive,
            backend,
            vram,
            ctx,
            batch,
            kv_dtype,
            weight_residency,
            offload_layers,
            gpu_fraction,
            json,
        } => {
            let result = bench_plan(
                &archive,
                thintensor::plan::PlanOptions {
                    backend,
                    vram_bytes: parse_bytes(&vram)?,
                    ctx_tokens: ctx,
                    batch_size: batch,
                    kv_dtype: parse_kv_dtype(&kv_dtype)?,
                    weight_residency: parse_weight_residency(&weight_residency)?,
                    offload_layers,
                    gpu_fraction: parse_gpu_fraction(gpu_fraction)?,
                },
            )?;
            if json {
                serde_json::to_writer_pretty(std::io::stdout(), &result)?;
                println!();
            } else {
                print_bench_plan(&result);
            }
        }
        Command::BenchBaseline {
            hf_dir,
            prompt,
            tokens,
            ctx,
            device,
            max_gpu_temp,
            json,
        } => run_runtime_bench(RuntimeBenchArgs {
            mode: "hf",
            path: hf_dir,
            hf_dir: None,
            prompt,
            tokens,
            ctx,
            device,
            max_gpu_temp,
            json,
        })?,
        Command::BenchArchive {
            archive,
            hf_dir,
            prompt,
            tokens,
            ctx,
            device,
            max_gpu_temp,
            json,
        } => run_runtime_bench(RuntimeBenchArgs {
            mode: "thin",
            path: archive,
            hf_dir: Some(hf_dir),
            prompt,
            tokens,
            ctx,
            device,
            max_gpu_temp,
            json,
        })?,
        Command::Repack {
            input,
            output,
            layout,
        } => {
            let archive = repack_archive(RepackOptions {
                input,
                output: output.clone(),
                layout,
            })?;
            println!(
                "repacked {} pages into {}",
                archive.records().len(),
                output.display()
            );
        }
    }

    Ok(())
}

fn run_runtime_bench(args: RuntimeBenchArgs) -> Result<()> {
    let mut command = ProcessCommand::new("python3");
    command
        .arg("scripts/bench_runtime.py")
        .arg(args.mode)
        .arg(args.path)
        .arg("--prompt")
        .arg(args.prompt)
        .arg("--tokens")
        .arg(args.tokens.to_string())
        .arg("--ctx")
        .arg(args.ctx.to_string())
        .arg("--device")
        .arg(args.device)
        .arg("--max-gpu-temp")
        .arg(args.max_gpu_temp.to_string());
    if let Some(hf_dir) = args.hf_dir {
        command.arg("--hf-dir").arg(hf_dir);
    }
    if args.json {
        command.arg("--json");
    }

    let status = command.status()?;
    if !status.success() {
        bail!("runtime benchmark failed with status {status}");
    }
    Ok(())
}

struct RuntimeBenchArgs {
    mode: &'static str,
    path: PathBuf,
    hf_dir: Option<PathBuf>,
    prompt: String,
    tokens: u32,
    ctx: u32,
    device: String,
    max_gpu_temp: u32,
    json: bool,
}

fn parse_weight_residency(value: &str) -> Result<WeightResidency> {
    WeightResidency::from_str(value).map_err(anyhow::Error::msg)
}

fn parse_kv_dtype(value: &str) -> Result<String> {
    let value = value.to_ascii_lowercase();
    match value.as_str() {
        "fp16" | "bf16" | "fp8" | "q8" | "q6" | "q5" | "q4" | "q3" | "q2" => Ok(value),
        _ => bail!("unsupported kv dtype {value}; expected fp16|bf16|fp8|q8|q4|q3|q2"),
    }
}

fn parse_gpu_fraction(value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        bail!("--gpu-fraction must be > 0 and <= 1")
    }
}

fn default_profile_path(archive: &std::path::Path, target_vram: &str) -> PathBuf {
    let safe_target: String = target_vram
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect();
    archive.with_file_name(format!(
        "{}.profile-{safe_target}.json",
        archive
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.thin")
    ))
}

fn print_bench_load(result: &BenchLoadResult) {
    println!("Input: {}", result.input);
    println!("Kind: {:?}", result.kind);
    println!("Load: {:.2} ms", result.elapsed_ms);
    println!("Files: {}", result.files);
    println!("Tensors/pages: {}", result.tensors_or_pages);
    println!("Bytes: {}", format_bytes(result.bytes));
    if let Some(rss) = result.rss_bytes {
        println!("RSS: {}", format_bytes(rss));
    }
    if let Some(peak) = result.peak_rss_bytes {
        println!("Peak RSS: {}", format_bytes(peak));
    }
    println!("Largest:");
    for item in &result.largest {
        println!("  {} {}", item.id, format_bytes(item.bytes));
    }
}

fn print_bench_plan(result: &BenchPlanResult) {
    println!("Input: {}", result.input);
    println!("Open: {:.2} ms", result.open_ms);
    println!("Verify: {:.2} ms", result.verify_ms);
    println!("Plan: {:.2} ms", result.plan_ms);
    println!("Archive: {}", format_bytes(result.archive_bytes));
    println!("Pages: {}", result.page_count);
    if let Some(rss) = result.rss_bytes {
        println!("RSS: {}", format_bytes(rss));
    }
    if let Some(peak) = result.peak_rss_bytes {
        println!("Peak RSS: {}", format_bytes(peak));
    }
    println!();
    print_plan(&result.plan);
}

fn print_inspect(archive: &Archive) {
    let header = archive.header();
    let manifest = archive.manifest();
    let stats = build_stats(archive);

    println!("ThinTensor .thin");
    println!("version: {}", manifest.version);
    println!(
        "model: {} layers={} hidden={} heads={} kv_heads={} head_dim={} tied={} dtype={}",
        manifest.model.arch,
        manifest.model.layers,
        manifest.model.hidden_size,
        manifest.model.heads,
        manifest.model.kv_heads,
        manifest
            .model
            .head_dim
            .map_or_else(|| "inferred".to_string(), |value| value.to_string()),
        manifest
            .model
            .tie_word_embeddings
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
        manifest.model.dtype
    );
    println!(
        "manifest: {} bytes @ {}",
        header.manifest_len, header.manifest_off
    );
    println!(
        "page table: {} records @ {}",
        header.page_count, header.page_table_off
    );
    println!("data: @ {}", header.data_off);
    println!("weights: {}", format_bytes(stats.total_raw_bytes));
    println!("pages: {}", stats.page_count);
    println!("execution stages: {}", stats.execution.stage_count);
    println!(
        "scratch: {}",
        format_bytes(manifest.memory_plan.scratch_bytes)
    );
    println!(
        "kv policy: recent {} high precision, older {}",
        manifest.memory_plan.kv_cache.recent_tokens_high_precision,
        manifest.memory_plan.kv_cache.old_tokens_codec
    );
    println!("largest pages:");
    for page in stats.largest_pages.iter().take(8) {
        println!(
            "  {} {} layer={}",
            page.id,
            format_bytes(page.raw_bytes),
            page.layer
                .map(|layer| layer.to_string())
                .unwrap_or_else(|| "global".to_string())
        );
    }
}

fn print_stats(stats: &ArchiveStats) {
    println!("ThinTensor stats");
    println!(
        "model: {} layers={} hidden={} heads={} kv_heads={} head_dim={} tied={} dtype={}",
        stats.model.arch,
        stats.model.layers,
        stats.model.hidden_size,
        stats.model.heads,
        stats.model.kv_heads,
        stats
            .model
            .head_dim
            .map_or_else(|| "inferred".to_string(), |value| value.to_string()),
        stats
            .model
            .tie_word_embeddings
            .map_or_else(|| "unknown".to_string(), |value| value.to_string()),
        stats.model.dtype
    );
    if let Some(raw_arch) = &stats.model.raw_arch {
        println!("raw arch: {raw_arch}");
    }
    println!("archive: {}", format_bytes(stats.archive_bytes));
    println!("manifest: {}", format_bytes(stats.manifest_bytes));
    println!("data offset: {}", stats.data_offset);
    println!("pages: {}", stats.page_count);
    println!("raw bytes: {}", format_bytes(stats.total_raw_bytes));
    println!("stored bytes: {}", format_bytes(stats.total_stored_bytes));
    println!();
    println!("Execution:");
    println!("  stages: {}", stats.execution.stage_count);
    println!("  page refs: {}", stats.execution.page_ref_count);
    println!(
        "  referenced pages: {}",
        stats.execution.referenced_page_count
    );
    println!("  empty stages: {}", stats.execution.empty_stage_count);
    println!(
        "  unreferenced pages: {}",
        stats.execution.unreferenced_pages.len()
    );
    println!();
    println!("Largest pages:");
    for page in stats.largest_pages.iter().take(12) {
        println!(
            "  {} {} kind={} op={} layer={}",
            page.id,
            format_bytes(page.raw_bytes),
            page.kind,
            page.op,
            page.layer
                .map(|layer| layer.to_string())
                .unwrap_or_else(|| "global".to_string())
        );
    }
    print_buckets("Per op", &stats.per_op, 12);
    print_buckets("By dtype", &stats.by_dtype, 12);
    print_buckets("By backend layout", &stats.by_backend_layout, 12);

    println!();
    println!("Per layer:");
    for layer in stats.per_layer.iter().take(16) {
        println!(
            "  layer_{} {} pages={} stored={}",
            layer.layer,
            format_bytes(layer.raw_bytes),
            layer.pages,
            format_bytes(layer.stored_bytes)
        );
    }
    if stats.per_layer.len() > 16 {
        println!("  ... {} more layers", stats.per_layer.len() - 16);
    }

    if !stats.unknown_op_pages.is_empty() {
        println!();
        println!("Unknown op pages:");
        for page in stats.unknown_op_pages.iter().take(12) {
            println!("  {page}");
        }
        if stats.unknown_op_pages.len() > 12 {
            println!("  ... {} more", stats.unknown_op_pages.len() - 12);
        }
    }
}

fn print_buckets(label: &str, buckets: &[thintensor::stats::BucketStats], limit: usize) {
    println!();
    println!("{label}:");
    for bucket in buckets.iter().take(limit) {
        println!(
            "  {} {} pages={} stored={}",
            bucket.name,
            format_bytes(bucket.raw_bytes),
            bucket.pages,
            format_bytes(bucket.stored_bytes)
        );
    }
    if buckets.len() > limit {
        println!("  ... {} more", buckets.len() - limit);
    }
}

fn print_report(report: &Report) {
    if report.is_ok() {
        println!("verify: ok");
    } else {
        println!("verify: failed");
        for error in &report.errors {
            println!("error: {error}");
        }
    }

    for warning in &report.warnings {
        println!("warning: {warning}");
    }
}

fn print_profile(profile: &RuntimeProfile, out_path: &std::path::Path) {
    println!("ThinTensor profile");
    println!("Backend: {}", profile.backend);
    println!("Target VRAM: {}", format_bytes(profile.target_vram_bytes));
    println!("Context: {}", profile.ctx_tokens);
    println!("Batch: {}", profile.batch_size);
    println!(
        "Status: {}",
        match profile.status {
            PlanStatus::Ok => "OK",
            PlanStatus::NotEnoughVram => "NOT ENOUGH VRAM",
        }
    );
    if let Some(candidate) = &profile.selected_candidate {
        println!("Selected: {candidate}");
    }
    println!("Recommended layout: {}", profile.recommended_layout);
    println!("KV codec: {}", profile.kv_codec_recommendation);
    println!("Sidecar: {}", out_path.display());
    println!();
    println!("Hot pages: {}", profile.always_hot_pages.len());
    println!("Streamable pages: {}", profile.streamable_pages.len());
    println!(
        "CPU offload candidates: {}",
        profile.cpu_offload_candidates.len()
    );
    println!();
    println!("Candidates:");
    for candidate in &profile.candidates {
        println!(
            "  {} status={:?} total={} resident={} streamed={} kv={}",
            candidate.name,
            candidate.status,
            format_bytes(candidate.total_bytes),
            format_bytes(candidate.resident_weight_bytes),
            format_bytes(candidate.streamed_weight_bytes),
            format_bytes(candidate.kv_cache_bytes)
        );
    }
    if !profile.biggest_memory_offenders.is_empty() {
        println!();
        println!("Biggest memory offenders:");
        for page in profile.biggest_memory_offenders.iter().take(8) {
            println!(
                "  {} {} op={} layer={}",
                page.id,
                format_bytes(page.raw_bytes),
                page.op,
                page.layer
                    .map(|layer| layer.to_string())
                    .unwrap_or_else(|| "global".to_string())
            );
        }
    }
}

fn print_simulation(simulation: &LoadSimulation) {
    println!("ThinTensor load simulation");
    println!("Backend: {}", simulation.backend);
    println!("Context: {}", simulation.ctx_tokens);
    println!("Batch: {}", simulation.batch_size);
    println!("Stages: {}", simulation.stage_count);
    println!("Execution page refs: {}", simulation.execution_page_refs);
    println!(
        "Sequential read ratio: {:.3}",
        simulation.sequential_read_ratio
    );
    println!(
        "Transitions: sequential={} nonsequential={} backward={}",
        simulation.sequential_transitions,
        simulation.nonsequential_transitions,
        simulation.backward_jumps
    );
    println!();
    println!("Residency:");
    println!("  resident pages: {}", simulation.resident_pages);
    println!("  streamed/evicted pages: {}", simulation.streamed_pages);
    println!(
        "  initial archive read: {}",
        format_bytes(simulation.initial_archive_read_bytes)
    );
    println!(
        "  resident read groups: {}",
        simulation.resident_read_groups
    );
    println!(
        "  resident prefetch IO ops: {}",
        simulation.resident_prefetch_io_ops_initial
    );
    println!(
        "  max resident group: {} pages, {}",
        simulation.max_resident_group_pages,
        format_bytes(simulation.max_resident_group_bytes)
    );
    println!(
        "  peak memory: {}",
        format_bytes(simulation.peak_memory_bytes)
    );
    println!();
    println!("Streaming / prefetch:");
    println!("  prefetch pages: {}", simulation.prefetch_pages);
    println!("  stream read groups: {}", simulation.stream_read_groups);
    println!(
        "  prefetch IO ops/token: {}",
        simulation.prefetch_io_ops_per_token
    );
    println!(
        "  max stream group: {} pages, {}",
        simulation.max_stream_group_pages,
        format_bytes(simulation.max_stream_group_bytes)
    );
    println!(
        "  recommended staging: {}",
        format_bytes(simulation.recommended_staging_bytes)
    );
    println!();
    println!("Estimated bytes moved per generated token:");
    println!(
        "  streamed weights: {}",
        format_bytes(simulation.streamed_weight_bytes_per_token)
    );
    println!(
        "  kv writes: {}",
        format_bytes(simulation.kv_write_bytes_per_token)
    );
    println!(
        "  total estimate: {}",
        format_bytes(simulation.estimated_bytes_moved_per_token)
    );
    println!();
    println!("Largest streamed stages:");
    let mut stages = simulation.stage_loads.clone();
    stages.sort_by(|left, right| {
        right
            .streamed_bytes
            .cmp(&left.streamed_bytes)
            .then(left.stage.cmp(&right.stage))
    });
    for stage in stages
        .iter()
        .filter(|stage| stage.streamed_bytes > 0)
        .take(8)
    {
        println!(
            "  {} streamed={} refs={}/{}",
            stage.stage,
            format_bytes(stage.streamed_bytes),
            stage.streamed_refs,
            stage.page_refs
        );
    }
}

fn print_plan(plan: &Plan) {
    println!("Backend: {}", plan.backend);
    println!("VRAM budget: {}", format_bytes(plan.vram_bytes));
    if plan.usable_vram_bytes != plan.vram_bytes {
        println!("Usable VRAM: {}", format_bytes(plan.usable_vram_bytes));
    }
    println!("Context: {}", plan.ctx_tokens);
    println!("Batch: {}", plan.batch_size);
    println!("KV dtype: {}", plan.kv_dtype);
    println!(
        "Weight residency: {}",
        weight_residency_name(plan.weight_residency)
    );
    if plan.offload_layers > 0 {
        println!("Offload layers: {}", plan.offload_layers);
    }
    println!();
    println!("Resident:");
    if plan.resident.is_empty() {
        println!("  none");
    } else {
        for page_id in &plan.resident {
            println!("  {page_id}");
        }
    }
    println!();
    println!("KV policy:");
    println!(
        "  recent {} tokens: high precision",
        plan.kv_policy.recent_tokens_high_precision
    );
    println!(
        "  older tokens: {} compressed",
        plan.kv_policy.old_tokens_codec
    );
    println!();
    println!("Scratch:");
    println!("  {}", format_bytes(plan.scratch_bytes));
    println!();
    println!("Estimated:");
    println!("  resident weights: {}", format_bytes(plan.weights_bytes));
    if plan.streamed_weight_bytes > 0 {
        println!(
            "  streamed/offloaded weights: {}",
            format_bytes(plan.streamed_weight_bytes)
        );
    }
    println!(
        "  physical archive weights: {}",
        format_bytes(plan.physical_weight_bytes)
    );
    if plan.shared_weight_savings_bytes > 0 {
        println!(
            "  exact shared-weight savings: {}",
            format_bytes(plan.shared_weight_savings_bytes)
        );
    }
    println!(
        "  total unique weights: {}",
        format_bytes(plan.total_weight_bytes)
    );
    println!("  scratch: {}", format_bytes(plan.scratch_bytes));
    println!(
        "  kv/cache at {} ctx: {}",
        plan.ctx_tokens,
        format_bytes(plan.kv_cache_bytes)
    );
    println!("  metadata: {}", format_bytes(plan.metadata_bytes));
    println!("  total: {}", format_bytes(plan.total_bytes));
    println!();
    match plan.status {
        PlanStatus::Ok => println!("Status: OK"),
        PlanStatus::NotEnoughVram => {
            println!("Status: NOT ENOUGH VRAM");
            println!();
            println!("Required:");
            println!("  resident weights: {}", format_bytes(plan.weights_bytes));
            println!("  scratch: {}", format_bytes(plan.scratch_bytes));
            println!("  kv cache: {}", format_bytes(plan.kv_cache_bytes));
            println!("  metadata: {}", format_bytes(plan.metadata_bytes));
            println!("  total: {}", format_bytes(plan.total_bytes));
            println!();
            println!("Budget:");
            println!("  {}", format_bytes(plan.usable_vram_bytes));
            if !plan.suggestions.is_empty() {
                println!();
                println!("Suggestions:");
                for suggestion in &plan.suggestions {
                    println!("  {suggestion}");
                }
            }
        }
    }
}

fn weight_residency_name(value: WeightResidency) -> &'static str {
    match value {
        WeightResidency::All => "all",
        WeightResidency::Stream => "stream",
        WeightResidency::OffloadLastN => "offload-last-n",
        WeightResidency::OffloadFirstN => "offload-first-n",
    }
}
