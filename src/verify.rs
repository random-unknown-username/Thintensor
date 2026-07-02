//! Full archive verifier: manifest contract, page table, offsets, and checksums.

use crate::archive::{Archive, HEADER_LEN, PageTableRecord};
use crate::error::Report;
use crate::manifest::{FORMAT_VERSION, PageSpec, validate_manifest};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};

pub fn verify_archive(archive: &Archive) -> Result<Report> {
    let mut report = validate_manifest(archive.manifest());

    verify_header(archive, &mut report);
    verify_manifest_vs_page_table(archive, &mut report);
    verify_record_ranges(archive, &mut report);
    verify_payload_checksums(archive, &mut report)?;

    Ok(report)
}

fn verify_header(archive: &Archive, report: &mut Report) {
    let header = archive.header();

    if header.header_len != HEADER_LEN {
        report.error(format!("header_len {} is not supported", header.header_len));
    }
    if header.version != FORMAT_VERSION {
        report.error(format!("unknown format version {}", header.version));
    }
    if header.manifest_off != HEADER_LEN as u64 {
        report.error("manifest_off must equal header_len in v0");
    }

    let Some(manifest_end) = header.manifest_off.checked_add(header.manifest_len) else {
        report.error("manifest range overflows u64");
        return;
    };
    if manifest_end != header.page_table_off {
        report.error("page_table_off must immediately follow manifest bytes");
    }
    if archive.page_table_end() != header.data_off {
        report.error("data_off must immediately follow page table records");
    }
    if header.data_off > archive.file_len() {
        report.error("data_off is outside file");
    }
    if header.page_count as usize != archive.records().len() {
        report.error("header page_count does not match parsed page table count");
    }
    if header.archive_hash != [0_u8; 32] {
        report.error("archive_hash is reserved and must be zero in v0");
    }
}

fn verify_manifest_vs_page_table(archive: &Archive, report: &mut Report) {
    let mut manifest_pages = BTreeMap::new();
    for page in &archive.manifest().pages {
        manifest_pages.insert(page.id.as_str(), page);
    }

    let mut record_ids = BTreeSet::new();
    for record in archive.records() {
        if !record_ids.insert(record.page_id.as_str()) {
            report.error(format!("duplicate page table id {}", record.page_id));
        }

        let Some(page) = manifest_pages.get(record.page_id.as_str()) else {
            report.error(format!(
                "page table entry {} is not present in manifest",
                record.page_id
            ));
            continue;
        };
        if page.fused_to.is_some() {
            report.error(format!(
                "logical fused page {} must not have its own page table record",
                page.id
            ));
            continue;
        }

        verify_record_against_page(record, page, report);
    }

    for page in &archive.manifest().pages {
        if page.fused_to.is_none() && !record_ids.contains(page.id.as_str()) {
            report.error(format!("manifest page {} missing from page table", page.id));
        }
    }
}

fn verify_record_against_page(record: &PageTableRecord, page: &PageSpec, report: &mut Report) {
    if record.flags != 0 {
        report.error(format!(
            "page {} uses unsupported flags {}",
            record.page_id, record.flags
        ));
    }
    if record.stored_size != record.raw_size {
        report.error(format!(
            "page {} has stored_size {} but raw_size {}",
            record.page_id, record.stored_size, record.raw_size
        ));
    }
    if record.stored_size != page.size {
        report.error(format!(
            "page {} declared size {} but page table stores {}",
            record.page_id, page.size, record.stored_size
        ));
    }
    let table_checksum = hex::encode(record.checksum);
    if table_checksum != page.checksum {
        report.error(format!(
            "page {} manifest checksum differs from page table checksum",
            record.page_id
        ));
    }
}

fn verify_record_ranges(archive: &Archive, report: &mut Report) {
    let mut ranges = Vec::new();

    for record in archive.records() {
        if record.stored_size == 0 {
            report.error(format!(
                "page {} stored_size must be non-zero",
                record.page_id
            ));
        }
        if record.offset < archive.header().data_off {
            report.error(format!(
                "page {} blob starts before data_off",
                record.page_id
            ));
        }

        let Some(end) = record.offset.checked_add(record.stored_size) else {
            report.error(format!(
                "page {} offset + size overflows u64",
                record.page_id
            ));
            continue;
        };
        if end > archive.file_len() {
            report.error(format!("page {} blob extends outside file", record.page_id));
        }

        ranges.push((record.page_id.as_str(), record.offset, end));
    }

    ranges.sort_by_key(|(_, start, _)| *start);
    for pair in ranges.windows(2) {
        let (left_id, _, left_end) = pair[0];
        let (right_id, right_start, _) = pair[1];
        if left_end > right_start {
            report.error(format!("page blobs {left_id} and {right_id} overlap"));
        }
    }
}

fn verify_payload_checksums(archive: &Archive, report: &mut Report) -> Result<()> {
    let records: BTreeMap<_, _> = archive
        .records()
        .iter()
        .map(|record| (record.page_id.as_str(), record))
        .collect();
    let fused_parent_ids: BTreeSet<_> = archive
        .manifest()
        .pages
        .iter()
        .filter_map(|page| page.fused_to.as_deref())
        .collect();
    let mut fused_payloads = BTreeMap::new();
    for record in archive.records() {
        let bytes = match archive.read_page(record) {
            Ok(bytes) => bytes,
            Err(err) => {
                report.error(err.to_string());
                continue;
            }
        };
        if bytes.len() as u64 != record.stored_size {
            report.error(format!(
                "page {} read {} bytes, expected {}",
                record.page_id,
                bytes.len(),
                record.stored_size
            ));
        }
        let actual = blake3::hash(&bytes);
        if actual.as_bytes() != &record.checksum {
            report.error(format!("page {} checksum mismatch", record.page_id));
        }
        if fused_parent_ids.contains(record.page_id.as_str()) {
            fused_payloads.insert(record.page_id.as_str(), bytes);
        }
    }

    for page in &archive.manifest().pages {
        let Some(parent_id) = page.fused_to.as_deref() else {
            continue;
        };
        let Some(_parent) = records.get(parent_id) else {
            continue;
        };
        let Some(bytes) = fused_payloads.get(parent_id) else {
            report.error(format!("fused parent {} payload was not read", parent_id));
            continue;
        };
        let offset = page.fused_offset.unwrap_or(0) as usize;
        let size = page.size as usize;
        let Some(slice) = bytes.get(offset..offset.saturating_add(size)) else {
            report.error(format!(
                "logical page {} slice is outside fused parent {}",
                page.id, parent_id
            ));
            continue;
        };
        if hex::encode(blake3::hash(slice).as_bytes()) != page.checksum {
            report.error(format!("logical page {} checksum mismatch", page.id));
        }
    }

    Ok(())
}
