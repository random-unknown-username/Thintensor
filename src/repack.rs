//! Archive repacking without changing raw page bytes.

use crate::archive::{Archive, ArchivePage, ArchivePageSource, write_archive_pages};
use crate::manifest::{Manifest, PageSpec};
use crate::verify::verify_archive;
use anyhow::{Result, anyhow, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct RepackOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub layout: String,
}

pub fn repack_archive(options: RepackOptions) -> Result<Archive> {
    if options.input == options.output {
        bail!("repack output must differ from input");
    }
    let archive = Archive::open(&options.input)?;
    let report = verify_archive(&archive)?;
    if !report.is_ok() {
        bail!(
            "input archive failed verification:\n{}",
            report.errors.join("\n")
        );
    }
    if options.layout == "fused_decode_v1" {
        return repack_fused_decode(&archive, &options.output);
    }

    let records: BTreeMap<_, _> = archive
        .records()
        .iter()
        .map(|record| (record.page_id.as_str(), record))
        .collect();
    let order = match options.layout.as_str() {
        "execution_ordered_v1" => execution_order(&archive),
        "hot_stream_v1" => hot_stream_order(&archive),
        _ => bail!("unsupported repack layout {}", options.layout),
    };
    let mut pages = Vec::with_capacity(archive.records().len());
    for page_id in order {
        let Some(record) = records.get(page_id.as_str()) else {
            continue;
        };
        pages.push(ArchivePage {
            id: record.page_id.clone(),
            size: record.raw_size,
            checksum: record.checksum,
            source: ArchivePageSource::FileRange {
                path: archive.path().to_path_buf(),
                offset: record.offset,
            },
        });
    }

    write_archive_pages(&options.output, archive.manifest(), &pages)?;
    let output = Archive::open(&options.output)?;
    let report = verify_archive(&output)?;
    if !report.is_ok() {
        bail!(
            "repacked archive failed verification:\n{}",
            report.errors.join("\n")
        );
    }
    Ok(output)
}

fn repack_fused_decode(archive: &Archive, output: &Path) -> Result<Archive> {
    let records: BTreeMap<_, _> = archive
        .records()
        .iter()
        .map(|record| (record.page_id.clone(), record))
        .collect();
    let mut manifest = archive.manifest().clone();
    let mut page_indexes: BTreeMap<_, _> = manifest
        .pages
        .iter()
        .enumerate()
        .map(|(index, page)| (page.id.clone(), index))
        .collect();
    let mut fused_blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut fused_by_child: BTreeMap<String, String> = BTreeMap::new();

    for layer in 0..manifest.model.layers {
        add_fused_group(
            archive,
            &records,
            &mut manifest,
            &mut page_indexes,
            &mut fused_blobs,
            &mut fused_by_child,
            layer,
            "attn_qkv",
            &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
        )?;
        add_fused_group(
            archive,
            &records,
            &mut manifest,
            &mut page_indexes,
            &mut fused_blobs,
            &mut fused_by_child,
            layer,
            "mlp_gate_up",
            &["mlp.gate_proj.weight", "mlp.up_proj.weight"],
        )?;
    }

    let mut physical_pages = Vec::new();
    let mut seen = BTreeSet::new();
    for stage in &manifest.execution_tape {
        for page_id in &stage.page_refs {
            let physical_id = fused_by_child.get(page_id).unwrap_or(page_id);
            if !seen.insert(physical_id.clone()) {
                continue;
            }
            push_physical_page(
                archive,
                &records,
                &mut fused_blobs,
                physical_id,
                &mut physical_pages,
            )?;
        }
    }
    for page in &manifest.pages {
        if page.fused_to.is_some() || !seen.insert(page.id.clone()) {
            continue;
        }
        push_physical_page(
            archive,
            &records,
            &mut fused_blobs,
            &page.id,
            &mut physical_pages,
        )?;
    }

    write_archive_pages(output, &manifest, &physical_pages)?;
    let output_archive = Archive::open(output)?;
    let report = verify_archive(&output_archive)?;
    if !report.is_ok() {
        bail!(
            "fused repacked archive failed verification:\n{}",
            report.errors.join("\n")
        );
    }
    Ok(output_archive)
}

#[allow(clippy::too_many_arguments)]
fn add_fused_group(
    archive: &Archive,
    records: &BTreeMap<String, &crate::archive::PageTableRecord>,
    manifest: &mut Manifest,
    page_indexes: &mut BTreeMap<String, usize>,
    fused_blobs: &mut BTreeMap<String, Vec<u8>>,
    fused_by_child: &mut BTreeMap<String, String>,
    layer: u32,
    op: &str,
    suffixes: &[&str],
) -> Result<()> {
    let child_ids: Vec<_> = suffixes
        .iter()
        .map(|suffix| format!("model.layers.{layer}.{suffix}"))
        .collect();
    let child_indexes: Vec<_> = child_ids
        .iter()
        .map(|id| {
            page_indexes
                .get(id)
                .copied()
                .ok_or_else(|| anyhow!("cannot fuse missing page {id}"))
        })
        .collect::<Result<_>>()?;
    let first = &manifest.pages[child_indexes[0]];
    if first.shape.len() != 2 {
        bail!("cannot fuse non-matrix page {}", first.id);
    }
    let dtype = first.dtype.clone();
    let cols = first.shape[1];
    for index in &child_indexes {
        let page = &manifest.pages[*index];
        if page.shape.len() != 2 || page.shape[1] != cols || page.dtype != dtype {
            bail!("fused group {op} layer {layer} has incompatible tensors");
        }
    }

    let parent_id = format!("layer_{layer}_{op}_fused");
    let mut blob = Vec::new();
    let mut offset = 0_u64;
    for (child_id, index) in child_ids.iter().zip(child_indexes.iter().copied()) {
        let record = records
            .get(child_id)
            .ok_or_else(|| anyhow!("page table record missing for {child_id}"))?;
        let bytes = archive.read_page(record)?;
        blob.extend_from_slice(&bytes);
        let page = &mut manifest.pages[index];
        page.fused_to = Some(parent_id.clone());
        page.fused_offset = Some(offset);
        fused_by_child.insert(child_id.clone(), parent_id.clone());
        offset += page.size;
    }
    let checksum = hex::encode(blake3::hash(&blob).as_bytes());
    let parent = PageSpec {
        id: parent_id.clone(),
        kind: "fused_physical".to_string(),
        layer: Some(layer),
        op: op.to_string(),
        dtype: "u8".to_string(),
        shape: vec![offset],
        backend_layout: "row_concat_v1".to_string(),
        size: offset,
        checksum,
        experimental: false,
        fused_to: None,
        fused_offset: None,
        quant_scheme: None,
        bits_per_weight: None,
        quant_group_size: None,
        scale_page: None,
    };
    page_indexes.insert(parent_id.clone(), manifest.pages.len());
    manifest.pages.push(parent);
    fused_blobs.insert(parent_id, blob);
    Ok(())
}

fn push_physical_page<'a>(
    archive: &'a Archive,
    records: &BTreeMap<String, &'a crate::archive::PageTableRecord>,
    fused_blobs: &mut BTreeMap<String, Vec<u8>>,
    page_id: &str,
    pages: &mut Vec<ArchivePage<'a>>,
) -> Result<()> {
    if let Some(bytes) = fused_blobs.remove(page_id) {
        let checksum = *blake3::hash(&bytes).as_bytes();
        pages.push(ArchivePage {
            id: page_id.to_string(),
            size: bytes.len() as u64,
            checksum,
            source: ArchivePageSource::Owned(bytes),
        });
        return Ok(());
    }
    let record = records
        .get(page_id)
        .ok_or_else(|| anyhow!("physical page {page_id} has no source record"))?;
    pages.push(ArchivePage {
        id: page_id.to_string(),
        size: record.raw_size,
        checksum: record.checksum,
        source: ArchivePageSource::FileRange {
            path: archive.path().to_path_buf(),
            offset: record.offset,
        },
    });
    Ok(())
}

fn execution_order(archive: &Archive) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();

    for stage in &archive.manifest().execution_tape {
        for page_id in &stage.page_refs {
            if seen.insert(page_id.clone()) {
                order.push(page_id.clone());
            }
        }
    }

    for page in &archive.manifest().pages {
        if seen.insert(page.id.clone()) {
            order.push(page.id.clone());
        }
    }

    order
}

fn hot_stream_order(archive: &Archive) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();

    for page in &archive.manifest().pages {
        if is_hot_page(page) && seen.insert(page.id.clone()) {
            order.push(page.id.clone());
        }
    }

    for stage in &archive.manifest().execution_tape {
        for page_id in &stage.page_refs {
            if seen.insert(page_id.clone()) {
                order.push(page_id.clone());
            }
        }
    }

    for page in &archive.manifest().pages {
        if seen.insert(page.id.clone()) {
            order.push(page.id.clone());
        }
    }

    order
}

fn is_hot_page(page: &crate::manifest::PageSpec) -> bool {
    page.layer.is_none() || matches!(page.kind.as_str(), "embedding" | "lm_head")
}
