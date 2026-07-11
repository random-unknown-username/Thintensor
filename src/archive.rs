//! ThinTensor `.thin` binary archive read/write.
//!
//! v0 layout is intentionally plain: fixed header, JSON manifest, page table, raw blobs.

use crate::manifest::{FORMAT_VERSION, Manifest, validate_manifest};
use anyhow::{Context, Result, anyhow, bail};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub const MAGIC: &[u8; 8] = b"THINv0\0\0";
pub const HEADER_LEN: u32 = 88;
pub const PAGE_FLAG_ZSTD: u32 = 1 << 0;
pub const PAGE_FLAG_ENCRYPTED: u32 = 1 << 1;
pub const PAGE_FLAG_DELTA_OVERLAY: u32 = 1 << 2;

#[derive(Debug, Clone)]
pub struct Header {
    pub header_len: u32,
    pub version: u32,
    pub manifest_off: u64,
    pub manifest_len: u64,
    pub page_table_off: u64,
    pub page_count: u64,
    pub data_off: u64,
    pub archive_hash: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct PageTableRecord {
    pub page_id: String,
    pub offset: u64,
    pub stored_size: u64,
    pub raw_size: u64,
    pub flags: u32,
    pub checksum: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct PackOptions {
    pub manifest_path: PathBuf,
    pub pages_dir: PathBuf,
    pub out_path: PathBuf,
}

#[derive(Debug)]
pub struct ArchivePage<'a> {
    pub id: String,
    pub size: u64,
    pub checksum: [u8; 32],
    pub source: ArchivePageSource<'a>,
}

#[derive(Debug)]
pub enum ArchivePageSource<'a> {
    Owned(Vec<u8>),
    Bytes(&'a [u8]),
    FileRange { path: PathBuf, offset: u64 },
}

#[derive(Debug, Clone)]
pub struct Archive {
    path: PathBuf,
    header: Header,
    manifest: Manifest,
    records: Vec<PageTableRecord>,
    page_table_end: u64,
    file_len: u64,
}

impl Archive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file =
            File::open(&path).with_context(|| format!("open archive {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let header = read_header(&mut file)?;

        let manifest_end = checked_add(header.manifest_off, header.manifest_len, "manifest end")?;
        if manifest_end > file_len {
            bail!("manifest range is outside archive");
        }

        file.seek(SeekFrom::Start(header.manifest_off))
            .context("seek manifest")?;
        let mut manifest_bytes = vec![0_u8; usize_len(header.manifest_len, "manifest")?];
        file.read_exact(&mut manifest_bytes)
            .context("read manifest")?;
        let manifest: Manifest =
            serde_json::from_slice(&manifest_bytes).context("parse manifest")?;

        file.seek(SeekFrom::Start(header.page_table_off))
            .context("seek page table")?;
        let mut records = Vec::with_capacity(usize_len(header.page_count, "page count")?);
        for _ in 0..header.page_count {
            records.push(read_page_table_record(&mut file)?);
        }
        let page_table_end = file.stream_position().context("read page table end")?;

        Ok(Self {
            path,
            header,
            manifest,
            records,
            page_table_end,
            file_len,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn records(&self) -> &[PageTableRecord] {
        &self.records
    }

    pub fn page_table_end(&self) -> u64 {
        self.page_table_end
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn read_page(&self, record: &PageTableRecord) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(record.offset))
            .with_context(|| format!("seek page {}", record.page_id))?;
        let mut bytes = vec![0_u8; usize_len(record.stored_size, "page blob")?];
        file.read_exact(&mut bytes)
            .with_context(|| format!("read page {}", record.page_id))?;
        Ok(bytes)
    }

    pub fn extract(&self, out_dir: impl AsRef<Path>) -> Result<()> {
        let out_dir = out_dir.as_ref();
        let pages_dir = out_dir.join("pages");
        fs::create_dir_all(&pages_dir)
            .with_context(|| format!("create {}", pages_dir.display()))?;

        let manifest_path = out_dir.join("manifest.json");
        let manifest_file = File::create(&manifest_path)
            .with_context(|| format!("create {}", manifest_path.display()))?;
        serde_json::to_writer_pretty(manifest_file, &self.manifest)
            .with_context(|| format!("write {}", manifest_path.display()))?;

        for record in &self.records {
            let bytes = self.read_page(record)?;
            let out_path = pages_dir.join(page_filename(&record.page_id)?);
            fs::write(&out_path, bytes).with_context(|| format!("write {}", out_path.display()))?;
        }

        Ok(())
    }
}

pub fn pack_archive(options: PackOptions) -> Result<Archive> {
    let manifest_bytes = fs::read(&options.manifest_path)
        .with_context(|| format!("read {}", options.manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parse {}", options.manifest_path.display()))?;

    let report = validate_manifest(&manifest);
    if !report.is_ok() {
        bail!("manifest invalid:\n{}", report.errors.join("\n"));
    }

    let page_files = index_page_files(&options.pages_dir)?;
    let mut pages = Vec::with_capacity(manifest.pages.len());

    for page in &manifest.pages {
        if page.fused_to.is_some() {
            continue;
        }
        let source = page_files.get(&page.id).ok_or_else(|| {
            anyhow!(
                "missing page file for {} in {}",
                page.id,
                options.pages_dir.display()
            )
        })?;
        let bytes = fs::read(source).with_context(|| format!("read page {}", source.display()))?;
        if bytes.len() as u64 != page.size {
            bail!(
                "page {} declared size {} but file has {} bytes",
                page.id,
                page.size,
                bytes.len()
            );
        }
        let checksum = blake3::hash(&bytes);
        let expected = hex::decode(&page.checksum)
            .with_context(|| format!("decode checksum for page {}", page.id))?;
        if checksum.as_bytes().as_slice() != expected.as_slice() {
            bail!("page {} checksum does not match manifest", page.id);
        }

        pages.push(ArchivePage {
            id: page.id.clone(),
            size: page.size,
            checksum: *checksum.as_bytes(),
            source: ArchivePageSource::Owned(bytes),
        });
    }

    write_archive_pages(&options.out_path, &manifest, &pages)?;
    Archive::open(&options.out_path)
}

pub fn write_archive_pages(
    path: &Path,
    manifest: &Manifest,
    pages: &[ArchivePage<'_>],
) -> Result<()> {
    let report = validate_manifest(manifest);
    if !report.is_ok() {
        bail!("manifest invalid:\n{}", report.errors.join("\n"));
    }

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let manifest_bytes = serde_json::to_vec_pretty(manifest).context("serialize manifest")?;
    let page_table_len = pages
        .iter()
        .try_fold(0_u64, |acc, page| {
            acc.checked_add(page_record_len(&page.id))
        })
        .ok_or_else(|| anyhow!("page table length overflows u64"))?;

    let manifest_off = HEADER_LEN as u64;
    let manifest_len = manifest_bytes.len() as u64;
    let page_table_off = checked_add(manifest_off, manifest_len, "page_table_off")?;
    let data_off = checked_add(page_table_off, page_table_len, "data_off")?;

    let mut offset = data_off;
    let mut records = Vec::with_capacity(pages.len());
    for page in pages {
        records.push(PageTableRecord {
            page_id: page.id.clone(),
            offset,
            stored_size: page.size,
            raw_size: page.size,
            flags: 0,
            checksum: page.checksum,
        });
        offset = checked_add(offset, page.size, "page offset")?;
    }

    let header = Header {
        header_len: HEADER_LEN,
        version: FORMAT_VERSION,
        manifest_off,
        manifest_len,
        page_table_off,
        page_count: pages.len() as u64,
        data_off,
        archive_hash: [0_u8; 32],
    };

    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("thintensor"),
        std::process::id(),
    ));
    let mut file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    write_header(&mut file, &header)?;
    file.write_all(&manifest_bytes).context("write manifest")?;
    for record in &records {
        write_page_table_record(&mut file, record)?;
    }
    for page in pages {
        write_page_source(&mut file, page).with_context(|| format!("write page {}", page.id))?;
    }
    file.flush().context("flush archive")?;
    file.sync_all().context("sync archive")?;
    drop(file);
    fs::rename(&temporary, path).with_context(|| {
        format!(
            "atomically replace {} with {}",
            path.display(),
            temporary.display()
        )
    })?;

    Ok(())
}

fn write_page_source(writer: &mut File, page: &ArchivePage<'_>) -> Result<()> {
    match &page.source {
        ArchivePageSource::Owned(bytes) => {
            if bytes.len() as u64 != page.size {
                bail!(
                    "page {} source has {} bytes, expected {}",
                    page.id,
                    bytes.len(),
                    page.size
                );
            }
            writer.write_all(bytes)?;
        }
        ArchivePageSource::Bytes(bytes) => {
            if bytes.len() as u64 != page.size {
                bail!(
                    "page {} source has {} bytes, expected {}",
                    page.id,
                    bytes.len(),
                    page.size
                );
            }
            writer.write_all(bytes)?;
        }
        ArchivePageSource::FileRange { path, offset } => {
            let mut input = File::open(path).with_context(|| format!("open {}", path.display()))?;
            input
                .seek(SeekFrom::Start(*offset))
                .with_context(|| format!("seek {}", path.display()))?;
            let copied = std::io::copy(&mut input.take(page.size), writer)
                .with_context(|| format!("copy {}", path.display()))?;
            if copied != page.size {
                bail!(
                    "page {} copied {} bytes, expected {}",
                    page.id,
                    copied,
                    page.size
                );
            }
        }
    }
    Ok(())
}

fn read_header(reader: &mut File) -> Result<Header> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic).context("read magic")?;
    if &magic != MAGIC {
        bail!("not a ThinTensor v0 archive");
    }

    let header = Header {
        header_len: reader.read_u32::<LittleEndian>()?,
        version: reader.read_u32::<LittleEndian>()?,
        manifest_off: reader.read_u64::<LittleEndian>()?,
        manifest_len: reader.read_u64::<LittleEndian>()?,
        page_table_off: reader.read_u64::<LittleEndian>()?,
        page_count: reader.read_u64::<LittleEndian>()?,
        data_off: reader.read_u64::<LittleEndian>()?,
        archive_hash: read_hash(reader)?,
    };

    if header.header_len != HEADER_LEN {
        bail!("unsupported header_len {}", header.header_len);
    }
    if header.version != FORMAT_VERSION {
        bail!("unknown format version {}", header.version);
    }

    Ok(header)
}

fn write_header(writer: &mut File, header: &Header) -> Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_u32::<LittleEndian>(header.header_len)?;
    writer.write_u32::<LittleEndian>(header.version)?;
    writer.write_u64::<LittleEndian>(header.manifest_off)?;
    writer.write_u64::<LittleEndian>(header.manifest_len)?;
    writer.write_u64::<LittleEndian>(header.page_table_off)?;
    writer.write_u64::<LittleEndian>(header.page_count)?;
    writer.write_u64::<LittleEndian>(header.data_off)?;
    writer.write_all(&header.archive_hash)?;
    Ok(())
}

fn read_page_table_record(reader: &mut File) -> Result<PageTableRecord> {
    let id_len = reader.read_u16::<LittleEndian>()? as usize;
    if id_len == 0 {
        bail!("page table record has empty id");
    }
    if id_len > 4096 {
        bail!("page table id length {id_len} is too large");
    }

    let mut id = vec![0_u8; id_len];
    reader.read_exact(&mut id).context("read page id")?;
    let page_id = String::from_utf8(id).context("page id must be UTF-8")?;

    Ok(PageTableRecord {
        page_id,
        offset: reader.read_u64::<LittleEndian>()?,
        stored_size: reader.read_u64::<LittleEndian>()?,
        raw_size: reader.read_u64::<LittleEndian>()?,
        flags: reader.read_u32::<LittleEndian>()?,
        checksum: read_hash(reader)?,
    })
}

fn write_page_table_record(writer: &mut File, record: &PageTableRecord) -> Result<()> {
    let id = record.page_id.as_bytes();
    if id.len() > u16::MAX as usize {
        bail!("page id {} is too long", record.page_id);
    }

    writer.write_u16::<LittleEndian>(id.len() as u16)?;
    writer.write_all(id)?;
    writer.write_u64::<LittleEndian>(record.offset)?;
    writer.write_u64::<LittleEndian>(record.stored_size)?;
    writer.write_u64::<LittleEndian>(record.raw_size)?;
    writer.write_u32::<LittleEndian>(record.flags)?;
    writer.write_all(&record.checksum)?;
    Ok(())
}

fn read_hash(reader: &mut File) -> Result<[u8; 32]> {
    let mut hash = [0_u8; 32];
    reader.read_exact(&mut hash).context("read hash")?;
    Ok(hash)
}

fn index_page_files(pages_dir: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut files = BTreeMap::new();

    for entry in WalkDir::new(pages_dir).follow_links(false) {
        let entry = entry.with_context(|| format!("walk {}", pages_dir.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.into_path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(id) = name.strip_suffix(".bin") else {
            continue;
        };
        let id = id.to_string();
        if files.insert(id.clone(), path).is_some() {
            bail!("duplicate page file for id {id}");
        }
    }

    Ok(files)
}

fn page_record_len(page_id: &str) -> u64 {
    2 + page_id.len() as u64 + 8 + 8 + 8 + 4 + 32
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{label} overflows u64"))
}

fn usize_len(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).with_context(|| format!("{label} length does not fit usize"))
}

fn page_filename(page_id: &str) -> Result<String> {
    if page_id.is_empty() || page_id.contains('/') || page_id.contains('\\') {
        bail!("unsafe page id {page_id:?}");
    }
    Ok(format!("{page_id}.bin"))
}
