import mmap
import json
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


HEADER_STRUCT = struct.Struct("<8sIIQQQQQ32s")
PREFIX_STRUCT = struct.Struct("<H")
TAIL_STRUCT = struct.Struct("<QQQI32s")
MAGIC = b"THINv0\0\0"
HEADER_LEN = 88


class ThinArchive:
    """
    Python parser for ThinTensor .thin archives.

    runtime features:
    - mmap-backed zero-copy page views
    - O(1) page metadata lookup
    - O(1) manifest page lookup
    - fused logical page resolution without linear scans
    - explicit get_page_view() for no-copy runtime loading
    """

    def __init__(
        self,
        path: Path | str,
        run_verify: bool = False,
        bin_path: Optional[str] = None,
    ):
        self.path = Path(path)
        self._file = None
        self._mmap = None
        self._fallback_bytes: Optional[bytes] = None

        self.manifest: Dict[str, Any] = {}
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.manifest_pages: Dict[str, Dict[str, Any]] = {}
        self.tensor_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.fused_children: Dict[str, list[str]] = {}

        if run_verify:
            self.verify_via_cli(bin_path)

        self._open()
        self._parse()

    def _open(self) -> None:
        self._file = self.path.open("rb")
        try:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._mmap = None
            self._file.seek(0)
            self._fallback_bytes = self._file.read()

    @property
    def file_size(self) -> int:
        return self.path.stat().st_size

    def verify_via_cli(self, bin_path: Optional[str] = None) -> None:
        if bin_path is None:
            for p in ("target/release/thintensor", "target/debug/thintensor"):
                candidate = Path(p)
                if candidate.exists():
                    bin_path = str(candidate)
                    break
            if bin_path is None:
                bin_path = "thintensor"

        try:
            subprocess.run(
                [bin_path, "verify", str(self.path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"CLI verify failed: {stderr}") from e

    def _slice(self, offset: int, length: int) -> memoryview:
        if offset < 0 or length < 0 or offset + length > self.file_size:
            raise ValueError(f"slice out of range: offset={offset}, length={length}")

        if self._mmap is not None:
            return memoryview(self._mmap)[offset : offset + length]

        assert self._fallback_bytes is not None
        return memoryview(self._fallback_bytes)[offset : offset + length]

    def _read_bytes(self, offset: int, length: int) -> bytes:
        # Explicit copy. Use only for tiny header/table/json parsing.
        return self._slice(offset, length).tobytes()

    def _parse(self) -> None:
        if self.file_size < HEADER_STRUCT.size:
            raise ValueError("File is too small to contain a ThinTensor header")

        (
            magic,
            header_len,
            version,
            manifest_off,
            manifest_len,
            page_table_off,
            page_count,
            data_off,
            archive_hash,
        ) = HEADER_STRUCT.unpack(self._read_bytes(0, HEADER_STRUCT.size))

        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")
        if header_len != HEADER_LEN:
            raise ValueError(f"Invalid header length: {header_len}, expected {HEADER_LEN}")
        if version != 0:
            raise ValueError(f"Invalid format version: {version}, expected 0")
        if manifest_off + manifest_len > self.file_size:
            raise ValueError("Manifest range exceeds file size")
        if page_table_off > self.file_size:
            raise ValueError("Page table offset exceeds file size")

        self.header = {
            "magic": magic,
            "header_len": header_len,
            "version": version,
            "manifest_off": manifest_off,
            "manifest_len": manifest_len,
            "page_table_off": page_table_off,
            "page_count": page_count,
            "data_off": data_off,
            "archive_hash": archive_hash,
        }

        manifest_bytes = self._read_bytes(manifest_off, manifest_len)
        self.manifest = json.loads(manifest_bytes.decode("utf-8"))

        self._parse_page_table(page_table_off, page_count, data_off)
        self._index_manifest_pages()
        self._validate_manifest_vs_table()
        self._validate_no_overlaps()

    def _parse_page_table(self, page_table_off: int, page_count: int, data_off: int) -> None:
        pos = page_table_off

        for i in range(page_count):
            if pos + PREFIX_STRUCT.size > self.file_size:
                raise ValueError(f"Truncated page table record at page index {i}")

            (id_len,) = PREFIX_STRUCT.unpack(self._read_bytes(pos, PREFIX_STRUCT.size))
            pos += PREFIX_STRUCT.size

            if id_len == 0:
                raise ValueError(f"Empty page id at page index {i}")
            if pos + id_len > self.file_size:
                raise ValueError(f"Truncated page id at page index {i}")

            page_id = self._read_bytes(pos, id_len).decode("utf-8")
            pos += id_len

            if page_id in self.pages:
                raise ValueError(f"Duplicate page id in page table: {page_id}")

            if pos + TAIL_STRUCT.size > self.file_size:
                raise ValueError(f"Truncated page record tail for page {page_id}")

            offset, stored_size, raw_size, flags, checksum = TAIL_STRUCT.unpack(
                self._read_bytes(pos, TAIL_STRUCT.size)
            )
            pos += TAIL_STRUCT.size

            if offset < data_off:
                raise ValueError(f"Page {page_id} offset starts before data_off")
            if offset + stored_size > self.file_size:
                raise ValueError(f"Page {page_id} offset+size exceeds file size")
            if stored_size == 0 and raw_size != 0:
                raise ValueError(f"Page {page_id} has zero stored_size but nonzero raw_size")

            self.pages[page_id] = {
                "offset": int(offset),
                "size": int(stored_size),
                "stored_size": int(stored_size),
                "raw_size": int(raw_size),
                "flags": int(flags),
                "checksum": checksum,
            }

    def _index_manifest_pages(self) -> None:
        for page in self.manifest.get("pages", []):
            page_id = page.get("id")
            if not page_id:
                raise ValueError("Manifest page missing id")
            if page_id in self.manifest_pages:
                raise ValueError(f"Duplicate page id in manifest: {page_id}")

            self.manifest_pages[page_id] = page

            fused_to = page.get("fused_to")
            if fused_to is not None:
                self.fused_children.setdefault(fused_to, []).append(page_id)

    def _validate_manifest_vs_table(self) -> None:
        for page_id, page in self.manifest_pages.items():
            fused_to = page.get("fused_to")
            if fused_to is not None:
                if fused_to not in self.pages:
                    raise ValueError(f"Logical page {page_id} refers to missing fused page {fused_to}")
                continue

            kind = page.get("kind")
            if kind == "fused_logical":
                continue

            if page_id not in self.pages:
                raise ValueError(f"Manifest page {page_id} missing from page table")

        for page_id in self.pages:
            if page_id not in self.manifest_pages:
                # Some v0 archives may allow physical fused pages with no logical metadata,
                # but normal ThinTensor archives should keep table and manifest aligned.
                pass

    def _validate_no_overlaps(self) -> None:
        sorted_pages = sorted(self.pages.items(), key=lambda item: item[1]["offset"])
        for idx in range(len(sorted_pages) - 1):
            name1, p1 = sorted_pages[idx]
            name2, p2 = sorted_pages[idx + 1]
            if p1["offset"] + p1["size"] > p2["offset"]:
                raise ValueError(f"Overlapping pages detected: {name1} overlaps with {name2}")

    def has_page(self, page_id: str) -> bool:
        return page_id in self.manifest_pages or page_id in self.pages

    def get_manifest_page(self, page_id: str) -> Dict[str, Any]:
        try:
            return self.manifest_pages[page_id]
        except KeyError as exc:
            raise KeyError(f"Page {page_id} not in manifest") from exc

    def get_page_record(self, page_id: str) -> Dict[str, Any]:
        try:
            return self.pages[page_id]
        except KeyError as exc:
            raise KeyError(f"Page {page_id} not in page table") from exc

    def get_page_view(self, page_id: str) -> memoryview:
        """
        Zero-copy view of physical or fused logical page bytes.
        Use this in torch.frombuffer instead of get_page_bytes().
        """
        manifest_page = self.get_manifest_page(page_id)

        fused_to = manifest_page.get("fused_to")
        if fused_to is not None:
            fused_record = self.get_page_record(fused_to)
            fused_offset = int(manifest_page.get("fused_offset", 0))
            size = int(manifest_page["size"])

            if fused_offset < 0 or fused_offset + size > fused_record["size"]:
                raise ValueError(f"Fused logical page {page_id} slice outside parent {fused_to}")

            return self._slice(fused_record["offset"] + fused_offset, size)

        record = self.get_page_record(page_id)
        return self._slice(record["offset"], record["size"])

    def get_page_bytes(self, page_id: str) -> bytes:
        """
        Explicit copy fallback. Avoid in hot paths.
        """
        return self.get_page_view(page_id).tobytes()

    def get_tensor_metadata(self, page_id: str) -> Dict[str, Any]:
        cached = self.tensor_metadata_cache.get(page_id)
        if cached is not None:
            return cached

        manifest_page = self.get_manifest_page(page_id)

        fused_to = manifest_page.get("fused_to")
        if fused_to is not None:
            fused_record = self.get_page_record(fused_to)
            offset = fused_record["offset"] + int(manifest_page.get("fused_offset", 0))
            size = int(manifest_page["size"])
        else:
            record = self.get_page_record(page_id)
            offset = record["offset"]
            size = record["size"]

        meta = {
            "id": page_id,
            "dtype": manifest_page["dtype"],
            "shape": manifest_page["shape"],
            "offset": offset,
            "size": size,
            "checksum": manifest_page.get("checksum"),
            "backend_layout": manifest_page.get("backend_layout", "unknown"),
            "kind": manifest_page.get("kind"),
            "layer": manifest_page.get("layer"),
            "op": manifest_page.get("op"),
            "fused_to": manifest_page.get("fused_to"),
            "fused_offset": manifest_page.get("fused_offset"),
        }
        self.tensor_metadata_cache[page_id] = meta
        return meta

    def physical_page_ids(self) -> list[str]:
        return list(self.pages.keys())

    def tensor_page_ids(self) -> list[str]:
        return list(self.manifest_pages.keys())

    def close(self) -> None:
        # Closing mmap while torch.frombuffer views exist can raise BufferError.
        # Runtime should close only after tensor views are gone.
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._fallback_bytes = None

    def __enter__(self) -> "ThinArchive":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()