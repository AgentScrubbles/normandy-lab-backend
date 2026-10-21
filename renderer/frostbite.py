"""
Frostbite engine data reader for Dragon Age: Inquisition.

Reads TOC, SB (superbundle), CAT (content address table), and CAS (content archive) files.
Based on the FrostyToolsuite SDK (https://github.com/CadeEvs/FrostyToolsuite).
"""

import struct
import zlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional


# --- DA:I TOC/SB Deobfuscation ---

class DADeobfuscator:
    """Handles DA:I's XOR-based TOC obfuscation."""

    MAGIC_LEGACY = 0x01CED100
    MAGIC_STANDARD = 0x03CED100

    def __init__(self):
        self.key: Optional[bytes] = None
        self.data_offset: int = 0

    def initialize(self, stream: BinaryIO) -> int:
        """Read magic, extract key if needed, return data start offset."""
        magic = struct.unpack('<I', stream.read(4))[0]

        if magic not in (self.MAGIC_LEGACY, self.MAGIC_STANDARD):
            stream.seek(0)
            return 0  # No obfuscation

        if magic == self.MAGIC_LEGACY:
            stream.seek(0x128)
            raw_key = bytearray(stream.read(260))
            self.key = bytes(b ^ 123 for b in raw_key)

        self.data_offset = 0x22C
        stream.seek(self.data_offset)
        return self.data_offset

    def deobfuscate(self, data: bytes, file_position: int) -> bytes:
        """XOR-deobfuscate a buffer read from the given file position."""
        if self.key is None:
            return data
        start = file_position - self.data_offset
        result = bytearray(data)
        for i in range(len(result)):
            result[i] ^= self.key[(start + i) % 257]
        return bytes(result)


# --- 7-bit encoded integer (LEB128 variant used by Frostbite) ---

def read_7bit_encoded_int(stream: BinaryIO) -> int:
    result = 0
    shift = 0
    while True:
        b = stream.read(1)[0]
        result |= (b & 0x7F) << shift
        if (b >> 7) == 0:
            return result
        shift += 7


def read_7bit_encoded_long(stream: BinaryIO) -> int:
    result = 0
    shift = 0
    while True:
        b = stream.read(1)[0]
        result |= (b & 0x7F) << shift
        if (b >> 7) == 0:
            return result
        shift += 7


# --- DbObject types (tagged binary format) ---

class DbType:
    INVALID = 0
    LIST = 1
    OBJECT = 2
    BOOLEAN = 6
    STRING = 7
    INT = 8
    LONG = 9
    FLOAT = 11
    DOUBLE = 12
    GUID = 15
    SHA1 = 16
    BYTE_ARRAY = 19


def read_null_terminated_string(stream: BinaryIO) -> str:
    chars = []
    while True:
        b = stream.read(1)
        if b == b'\x00' or len(b) == 0:
            break
        chars.append(b)
    return b''.join(chars).decode('utf-8', errors='replace')


def read_sized_string(stream: BinaryIO, length: int) -> str:
    return stream.read(length).decode('utf-8', errors='replace')


def read_sha1(stream: BinaryIO) -> bytes:
    return stream.read(20)


def read_guid(stream: BinaryIO) -> bytes:
    return stream.read(16)


class DbObject:
    """A parsed Frostbite database object (dict or list)."""

    def __init__(self, data=None):
        if isinstance(data, dict):
            self._dict = data
            self._list = None
        elif isinstance(data, list):
            self._dict = None
            self._list = data
        else:
            self._dict = {} if data is None else None
            self._list = [] if data is not None else None

    def get(self, key: str, default=None):
        if self._dict is None:
            return default
        return self._dict.get(key, default)

    def __getitem__(self, key):
        if isinstance(key, str) and self._dict is not None:
            return self._dict[key]
        if isinstance(key, int) and self._list is not None:
            return self._list[key]
        raise KeyError(key)

    def __contains__(self, key):
        if self._dict is not None:
            return key in self._dict
        return False

    def __iter__(self):
        if self._list is not None:
            return iter(self._list)
        if self._dict is not None:
            return iter(self._dict)
        return iter([])

    def __len__(self):
        if self._list is not None:
            return len(self._list)
        if self._dict is not None:
            return len(self._dict)
        return 0

    @property
    def is_list(self):
        return self._list is not None

    @property
    def is_dict(self):
        return self._dict is not None

    def keys(self):
        return self._dict.keys() if self._dict else []

    def items(self):
        return self._dict.items() if self._dict else []

    def values(self):
        return self._dict.values() if self._dict else []

    def to_dict(self) -> Any:
        """Recursively convert to plain Python dicts/lists for JSON serialization."""
        if self._dict is not None:
            return {k: _serialize(v) for k, v in self._dict.items()}
        if self._list is not None:
            return [_serialize(v) for v in self._list]
        return None

    def __repr__(self):
        if self._dict is not None:
            return f"DbObject({{{', '.join(f'{k!r}: ...' for k in list(self._dict.keys())[:5])}}})"
        if self._list is not None:
            return f"DbObject([{len(self._list)} items])"
        return "DbObject(None)"


def _serialize(v):
    if isinstance(v, DbObject):
        return v.to_dict()
    if isinstance(v, bytes):
        return v.hex()
    return v


def read_db_object(stream: BinaryIO) -> Optional[DbObject]:
    """Read a top-level DbObject from a stream."""
    result = _read_db_value(stream)
    if isinstance(result, DbObject):
        return result
    return None


def _read_db_value(stream: BinaryIO):
    """Read a single typed value from the DbObject stream."""
    raw = stream.read(1)
    if len(raw) == 0:
        return None
    tmp = raw[0]

    obj_type = tmp & 0x1F
    if obj_type == DbType.INVALID:
        return None

    name = None
    if (tmp & 0x80) == 0:
        name = read_null_terminated_string(stream)

    value = None
    if obj_type == DbType.LIST:
        size = read_7bit_encoded_long(stream)
        offset = stream.tell()
        items = []
        while stream.tell() - offset < size:
            sub = _read_db_value(stream)
            if sub is None:
                break
            items.append(sub)
        value = DbObject(items)

    elif obj_type == DbType.OBJECT:
        size = read_7bit_encoded_long(stream)
        offset = stream.tell()
        fields = {}
        while stream.tell() - offset < size:
            sub = _read_db_value(stream)
            if sub is None:
                break
            # The sub-value's name was already parsed
        # Actually we need to re-approach this — the name is part of each sub-value
        # Let me re-read the C# code more carefully...
        # In DbReader.ReadDbObject, for Object type, it reads key-value pairs where
        # each pair is: type_byte + name + value
        # So we need to capture the name during recursive reads
        value = DbObject(fields)

    elif obj_type == DbType.BOOLEAN:
        value = stream.read(1)[0] == 1

    elif obj_type == DbType.STRING:
        length = read_7bit_encoded_int(stream)
        value = read_sized_string(stream, length)

    elif obj_type == DbType.INT:
        value = struct.unpack('<i', stream.read(4))[0]

    elif obj_type == DbType.LONG:
        value = struct.unpack('<q', stream.read(8))[0]

    elif obj_type == DbType.FLOAT:
        value = struct.unpack('<f', stream.read(4))[0]

    elif obj_type == DbType.DOUBLE:
        value = struct.unpack('<d', stream.read(8))[0]

    elif obj_type == DbType.GUID:
        value = read_guid(stream)

    elif obj_type == DbType.SHA1:
        value = read_sha1(stream)

    elif obj_type == DbType.BYTE_ARRAY:
        length = read_7bit_encoded_int(stream)
        value = stream.read(length)

    return (name, value)


# The above has a design issue — we need name + value together for Object fields.
# Let me redo this properly:

def read_db_object_v2(stream: BinaryIO) -> Optional[Any]:
    """Read a complete DbObject hierarchy from a Frostbite SB stream."""
    name, value = _read_db_entry(stream)
    return value


def _read_db_entry(stream: BinaryIO) -> tuple:
    """Read one entry: returns (name, value) tuple."""
    raw = stream.read(1)
    if len(raw) == 0:
        return (None, None)

    tmp = raw[0]
    obj_type = tmp & 0x1F
    if obj_type == DbType.INVALID:
        return (None, None)

    # Read name unless anonymous (bit 7 set)
    name = ""
    if (tmp & 0x80) == 0:
        name = read_null_terminated_string(stream)

    if obj_type == DbType.LIST:
        size = read_7bit_encoded_long(stream)
        start = stream.tell()
        items = []
        while stream.tell() - start < size:
            _, sub_value = _read_db_entry(stream)
            if sub_value is None:
                break
            items.append(sub_value)
        return (name, DbObject(items))

    elif obj_type == DbType.OBJECT:
        size = read_7bit_encoded_long(stream)
        start = stream.tell()
        fields = {}
        while stream.tell() - start < size:
            sub_name, sub_value = _read_db_entry(stream)
            if sub_value is None:
                break
            if sub_name:
                fields[sub_name] = sub_value
        return (name, DbObject(fields))

    elif obj_type == DbType.BOOLEAN:
        return (name, stream.read(1)[0] == 1)

    elif obj_type == DbType.STRING:
        length = read_7bit_encoded_int(stream)
        return (name, read_sized_string(stream, length))

    elif obj_type == DbType.INT:
        return (name, struct.unpack('<i', stream.read(4))[0])

    elif obj_type == DbType.LONG:
        return (name, struct.unpack('<q', stream.read(8))[0])

    elif obj_type == DbType.FLOAT:
        return (name, struct.unpack('<f', stream.read(4))[0])

    elif obj_type == DbType.DOUBLE:
        return (name, struct.unpack('<d', stream.read(8))[0])

    elif obj_type == DbType.GUID:
        return (name, read_guid(stream))

    elif obj_type == DbType.SHA1:
        return (name, read_sha1(stream))

    elif obj_type == DbType.BYTE_ARRAY:
        length = read_7bit_encoded_int(stream)
        return (name, stream.read(length))

    return (name, None)


# --- CAT Reader (Content Address Table) ---

CAT_MAGIC = b"NyanNyanNyanNyan"


@dataclass
class CatEntry:
    sha1: bytes
    offset: int
    size: int
    archive_index: int


def read_cat(cat_path: Path) -> dict[bytes, CatEntry]:
    """Read a DA:I CAT file, returning a SHA1 → CatEntry mapping."""
    entries = {}
    with open(cat_path, 'rb') as f:
        magic = f.read(16)
        if magic != CAT_MAGIC:
            raise ValueError(f"Invalid CAT magic: {magic!r}")

        # DA:I legacy format: no count header, entries are 0x20 bytes each
        data = f.read()

    entry_size = 0x20  # SHA1(20) + offset(4) + size(4) + archive_index(4) = 32
    num_entries = len(data) // entry_size

    for i in range(num_entries):
        offset = i * entry_size
        sha1 = data[offset:offset + 20]
        entry_offset = struct.unpack_from('<I', data, offset + 20)[0]
        entry_size_val = struct.unpack_from('<I', data, offset + 24)[0]
        archive_index = struct.unpack_from('<i', data, offset + 28)[0] & 0xFF
        entries[sha1] = CatEntry(
            sha1=sha1,
            offset=entry_offset,
            size=entry_size_val,
            archive_index=archive_index,
        )

    return entries


# --- CAS Reader (Content Archive Stream) ---

def read_cas_blocks(stream: BinaryIO, size: int) -> bytes:
    """Read and decompress CAS blocks from a stream.

    CAS files store data in blocks:
      - 4 bytes BE: decompressed size
      - 2 bytes: compression type
      - 2 bytes BE: compressed buffer size
    """
    output = BytesIO()
    end = stream.tell() + size

    while stream.tell() < end:
        header = stream.read(8)
        if len(header) < 8:
            break

        decompressed_size = struct.unpack('>I', header[0:4])[0]
        compression_type = struct.unpack('<H', header[4:6])[0]
        buffer_size = struct.unpack('>H', header[6:8])[0]

        # Extended buffer size in flags
        flags = (compression_type & 0xFF00) >> 8
        if (flags & 0x0F) != 0:
            buffer_size = ((flags & 0x0F) << 16) + buffer_size

        # Strip high bits from decompressed size
        decompressed_size &= 0x00FFFFFF

        compression_type = compression_type & 0x7F

        compressed_data = stream.read(buffer_size)

        if compression_type == 0x00:  # Uncompressed
            output.write(compressed_data)
        elif compression_type == 0x02:  # ZLib
            output.write(zlib.decompress(compressed_data))
        elif compression_type == 0x09:  # LZ4
            try:
                import lz4.block
                output.write(lz4.block.decompress(compressed_data, uncompressed_size=decompressed_size))
            except ImportError:
                raise ImportError("lz4 package required for LZ4 decompression: pip install lz4")
        else:
            raise ValueError(f"Unsupported compression type: 0x{compression_type:02x}")

    return output.getvalue()


def extract_cas_data(cas_dir: Path, cat_entry: CatEntry) -> bytes:
    """Extract raw data for a CAT entry from the appropriate CAS file."""
    cas_file = cas_dir / f"cas_{cat_entry.archive_index:02d}.cas"
    with open(cas_file, 'rb') as f:
        f.seek(cat_entry.offset)
        return read_cas_blocks(f, cat_entry.size)


# --- TOC/SB Reader ---

@dataclass
class BundleEntry:
    path: str
    ebx: list = field(default_factory=list)
    res: list = field(default_factory=list)
    chunks: list = field(default_factory=list)


@dataclass
class AssetEntry:
    name: str
    sha1: bytes
    size: int
    original_size: int
    asset_type: str = ""  # "ebx", "res", "chunk"
    res_type: int = 0     # resource type ID (for res entries)
    res_meta: bytes = b""


def read_toc(toc_path: Path) -> DbObject:
    """Read a TOC file with DA:I deobfuscation."""
    deobfuscator = DADeobfuscator()
    with open(toc_path, 'rb') as f:
        data_offset = deobfuscator.initialize(f)
        remaining = f.read()

    # Deobfuscate if needed
    if deobfuscator.key is not None:
        remaining = deobfuscator.deobfuscate(remaining, data_offset + len(remaining))

    stream = BytesIO(remaining)
    return read_db_object_v2(stream)


def read_superbundle(sb_path: Path) -> list[BundleEntry]:
    """Read a superbundle file, returning all bundle entries with their assets."""
    with open(sb_path, 'rb') as f:
        data = f.read()

    stream = BytesIO(data)
    root = read_db_object_v2(stream)

    bundles = []
    if root is None:
        return bundles

    bundle_list = root.get('bundles')
    if bundle_list is None:
        return bundles

    for bundle_obj in bundle_list:
        if not isinstance(bundle_obj, DbObject):
            continue

        entry = BundleEntry(path=bundle_obj.get('path', ''))

        # Parse EBX entries
        ebx_list = bundle_obj.get('ebx')
        if ebx_list and isinstance(ebx_list, DbObject) and ebx_list.is_list:
            for item in ebx_list:
                if isinstance(item, DbObject):
                    entry.ebx.append(AssetEntry(
                        name=item.get('name', ''),
                        sha1=item.get('sha1', b''),
                        size=item.get('size', 0),
                        original_size=item.get('originalSize', 0),
                        asset_type='ebx',
                    ))

        # Parse RES entries
        res_list = bundle_obj.get('res')
        if res_list and isinstance(res_list, DbObject) and res_list.is_list:
            for item in res_list:
                if isinstance(item, DbObject):
                    entry.res.append(AssetEntry(
                        name=item.get('name', ''),
                        sha1=item.get('sha1', b''),
                        size=item.get('size', 0),
                        original_size=item.get('originalSize', 0),
                        asset_type='res',
                        res_type=item.get('resType', 0),
                        res_meta=item.get('resMeta', b''),
                    ))

        # Parse chunk entries
        chunk_list = bundle_obj.get('chunks')
        if chunk_list and isinstance(chunk_list, DbObject) and chunk_list.is_list:
            for item in chunk_list:
                if isinstance(item, DbObject):
                    entry.chunks.append(AssetEntry(
                        name=item.get('id', b'').hex() if isinstance(item.get('id'), bytes) else str(item.get('id', '')),
                        sha1=item.get('sha1', b''),
                        size=item.get('size', 0),
                        original_size=item.get('originalSize', 0),
                        asset_type='chunk',
                    ))

        bundles.append(entry)

    return bundles


# --- High-level API ---

class FrostbiteArchive:
    """High-level interface to a DA:I game data directory."""

    def __init__(self, game_path: str | Path):
        self.game_path = Path(game_path)
        self.data_path = self.game_path / "Data"
        self.win32_path = self.data_path / "Win32"
        self._cat: Optional[dict[bytes, CatEntry]] = None

    @property
    def cat(self) -> dict[bytes, CatEntry]:
        if self._cat is None:
            self._cat = read_cat(self.data_path / "cas.cat")
            # Also load patch CAT if it exists
            patch_cat = self.data_path / "Update" / "Patch" / "Data" / "cas.cat"
            if patch_cat.exists():
                self._cat.update(read_cat(patch_cat))
        return self._cat

    def list_superbundles(self) -> list[str]:
        """List all .toc files in Win32 directory."""
        return sorted(p.stem for p in self.win32_path.glob("*.toc"))

    def read_superbundle(self, name: str) -> list[BundleEntry]:
        """Read a named superbundle (e.g., 'chargen')."""
        sb_path = self.win32_path / f"{name}.sb"
        return read_superbundle(sb_path)

    def extract_by_sha1(self, sha1: bytes) -> Optional[bytes]:
        """Extract asset data by SHA1 hash."""
        entry = self.cat.get(sha1)
        if entry is None:
            return None
        return extract_cas_data(self.data_path, entry)

    def find_asset(self, name_pattern: str, superbundle: str = None) -> list[tuple[str, AssetEntry]]:
        """Search for assets by name pattern across superbundles."""
        results = []
        sbs = [superbundle] if superbundle else self.list_superbundles()

        for sb_name in sbs:
            try:
                bundles = self.read_superbundle(sb_name)
            except Exception:
                continue

            for bundle in bundles:
                for asset in bundle.ebx + bundle.res + bundle.chunks:
                    if name_pattern.lower() in asset.name.lower():
                        results.append((sb_name, asset))

        return results
