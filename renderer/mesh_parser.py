"""
Frostbite MeshSet resource parser for DA:I.

Extracts vertices, indices, UVs, normals from MeshSet .res files.
Based on FrostyToolsuite MeshSetPlugin/Resources/MeshSet.cs.
"""

import struct
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional
import numpy as np


# --- Vertex element enums ---

class VertexUsage:
    UNKNOWN = 0x00
    POS = 0x01
    BONE_INDICES = 0x02
    BONE_INDICES2 = 0x03
    BONE_WEIGHTS = 0x04
    BONE_WEIGHTS2 = 0x05
    NORMAL = 0x06
    TANGENT = 0x07
    BINORMAL = 0x08
    BINORMAL_SIGN = 0x09
    COLOR0 = 0x1E
    COLOR1 = 0x1F
    TEXCOORD0 = 0x21
    TEXCOORD1 = 0x22
    TEXCOORD2 = 0x23
    TEXCOORD3 = 0x24
    TEXCOORD4 = 0x25

    _NAMES = {
        0x01: "Pos", 0x02: "BoneIndices", 0x03: "BoneIndices2",
        0x04: "BoneWeights", 0x05: "BoneWeights2", 0x06: "Normal",
        0x07: "Tangent", 0x08: "Binormal", 0x09: "BinormalSign",
        0x1E: "Color0", 0x1F: "Color1",
        0x21: "TexCoord0", 0x22: "TexCoord1", 0x23: "TexCoord2",
        0x24: "TexCoord3", 0x25: "TexCoord4",
    }

    @classmethod
    def name(cls, u):
        return cls._NAMES.get(u, f"0x{u:02x}")


class VertexFormat:
    NONE = 0x00
    FLOAT = 0x01
    FLOAT2 = 0x02
    FLOAT3 = 0x03
    FLOAT4 = 0x04
    HALF = 0x05
    HALF2 = 0x06
    HALF3 = 0x07
    HALF4 = 0x08
    BYTE4 = 0x0A
    BYTE4N = 0x0B
    UBYTE4 = 0x0C
    UBYTE4N = 0x0D
    SHORT2 = 0x0F
    SHORT4 = 0x11
    SHORT2N = 0x13
    SHORT4N = 0x15
    USHORT2 = 0x16
    USHORT4 = 0x17
    USHORT2N = 0x18
    USHORT4N = 0x19
    INT = 0x1A
    UINT = 0x20
    UBYTEN = 0x32

    _SIZES = {
        0x00: 0, 0x01: 4, 0x02: 8, 0x03: 12, 0x04: 16,
        0x05: 2, 0x06: 4, 0x07: 6, 0x08: 8,
        0x0A: 4, 0x0B: 4, 0x0C: 4, 0x0D: 4,
        0x0E: 2, 0x0F: 4, 0x10: 6, 0x11: 8,
        0x12: 2, 0x13: 4, 0x14: 6, 0x15: 8,
        0x16: 4, 0x17: 8, 0x18: 4, 0x19: 8,
        0x1A: 4, 0x1B: 8, 0x1C: 16,
        0x1D: 4, 0x1E: 8, 0x1F: 16,
        0x20: 4, 0x21: 8, 0x22: 16,
        0x23: 4, 0x24: 8,
        0x32: 1, 0x33: 12, 0x34: 12,
    }

    @classmethod
    def size(cls, f):
        return cls._SIZES.get(f, 0)


# DA:I constants
MAX_ELEMENTS = 16
MAX_STREAMS = 4  # DA:I specific
MAX_LODS = 6
MAX_CATEGORIES = 4  # DA:I specific
SECTION_UNKNOWN_BYTES = 44  # DA:I default
DECL_COUNT = 1  # DA:I


@dataclass
class VertexElement:
    usage: int
    format: int
    offset: int
    stream_index: int

    @property
    def size(self):
        return VertexFormat.size(self.format)


@dataclass
class GeometryDecl:
    elements: list  # list of VertexElement
    streams: list   # list of (stride, classification)
    element_count: int
    stream_count: int


@dataclass
class MeshSection:
    material_name: str = ""
    material_id: int = 0
    primitive_count: int = 0
    start_index: int = 0
    vertex_offset: int = 0
    vertex_count: int = 0
    vertex_stride: int = 0
    primitive_type: int = 0
    bones_per_vertex: int = 0
    bone_count: int = 0
    bone_list: list = field(default_factory=list)
    geom_decl: Optional[GeometryDecl] = None
    tex_coord_ratios: list = field(default_factory=list)


@dataclass
class MeshLod:
    mesh_type: int = 0
    max_instances: int = 0
    section_count: int = 0
    sections: list = field(default_factory=list)
    flags: int = 0
    index_buffer_format: int = 0  # 0=16bit, 1=32bit
    index_buffer_size: int = 0
    vertex_buffer_size: int = 0
    adjacency_buffer_size: int = 0
    chunk_id: bytes = b''
    inline_data_offset: int = 0
    name: str = ""
    short_name: str = ""
    bone_indices: list = field(default_factory=list)
    bone_short_names: list = field(default_factory=list)


@dataclass
class MeshSet:
    bounding_box_min: tuple = (0, 0, 0)
    bounding_box_max: tuple = (0, 0, 0)
    fullname: str = ""
    name: str = ""
    name_hash: int = 0
    mesh_type: int = 0
    flags: int = 0
    lod_count: int = 0
    section_count: int = 0
    lods: list = field(default_factory=list)


class MeshSetParser:
    """Parse a MeshSet .res file for DA:I."""

    def __init__(self, data: bytes, res_meta: bytes = b''):
        self.stream = BytesIO(data)
        self.data = data
        self.res_meta = res_meta

    def _read(self, fmt):
        return struct.unpack(fmt, self.stream.read(struct.calcsize(fmt)))

    def _read_u8(self): return self._read('<B')[0]
    def _read_u16(self): return self._read('<H')[0]
    def _read_u32(self): return self._read('<I')[0]
    def _read_i32(self): return self._read('<i')[0]
    def _read_u64(self): return self._read('<Q')[0]
    def _read_i64(self): return self._read('<q')[0]
    def _read_f32(self): return self._read('<f')[0]
    def _read_guid(self): return self.stream.read(16)

    @property
    def _pos(self):
        return self.stream.tell()

    @_pos.setter
    def _pos(self, v):
        self.stream.seek(v)

    def _pad(self, n):
        while self._pos % n != 0:
            self._pos += 1

    def _read_vec3(self):
        x, y, z, _ = self._read('<4f')
        return (x, y, z)

    def _read_null_terminated(self):
        chars = []
        while True:
            b = self.stream.read(1)
            if b == b'\x00' or len(b) == 0:
                break
            chars.append(b)
        return b''.join(chars).decode('utf-8', errors='replace')

    def parse(self) -> MeshSet:
        mesh = MeshSet()

        # Bounding box
        mesh.bounding_box_min = self._read_vec3()
        mesh.bounding_box_max = self._read_vec3()

        # LOD offsets (6 x int64)
        lod_offsets = [self._read_i64() for _ in range(MAX_LODS)]

        # DA:I: no extra long here (not like newer games)

        # String offsets
        fullname_offset = self._read_i64()
        name_offset = self._read_i64()

        mesh.name_hash = self._read_u32()
        mesh.mesh_type = self._read_u32()

        # DA:I: no lodFadeDistanceFactors
        # DA:I: flags as uint32 (not uint64)
        mesh.flags = self._read_u32()

        # DA:I: no NewPartBoneLayout fields
        mesh.lod_count = self._read_u16()
        mesh.section_count = self._read_u16()

        # Parse each LOD
        section_index = 0
        for i in range(mesh.lod_count):
            if lod_offsets[i] == 0:
                continue
            self._pos = lod_offsets[i]
            lod = self._parse_lod(section_index)
            mesh.lods.append(lod)
            section_index += lod.section_count

        # Read strings
        if fullname_offset != 0:
            self._pos = fullname_offset
            mesh.fullname = self._read_null_terminated()
        if name_offset != 0:
            self._pos = name_offset
            mesh.name = self._read_null_terminated()

        return mesh

    def _parse_lod(self, section_index: int) -> MeshLod:
        lod = MeshLod()
        lod.mesh_type = self._read_u32()
        lod.max_instances = self._read_u32()
        lod.section_count = self._read_i32()
        section_offset = self._read_i64()

        # Read sections
        saved = self._pos
        self._pos = section_offset
        for i in range(lod.section_count):
            section = self._parse_section(section_index + i)
            lod.sections.append(section)
        self._pos = saved

        # Category subsets (4 categories for DA:I)
        for _ in range(MAX_CATEGORIES):
            count = self._read_i32()
            cat_offset = self._read_i64()
            # Skip reading actual category data for now

        lod.flags = self._read_u32()

        # DA:I: index buffer format as IndexBufferFormat enum
        lod.index_buffer_format = self._read_i32()

        lod.index_buffer_size = self._read_u32()
        lod.vertex_buffer_size = self._read_u32()

        # DA:I has adjacency in mesh
        lod.adjacency_buffer_size = self._read_i32()

        lod.chunk_id = self._read_guid()
        lod.inline_data_offset = self._read_u32()

        # Adjacency buffer offset
        adj_offset = self._read_i64()

        # String offsets
        str_offset1 = self._read_i64()
        str_offset2 = self._read_i64()
        str_offset3 = self._read_i64()

        name_hash = self._read_u32()
        self._read_i64()  # skip

        # DA:I uses old bone/part layout (not HasNewPartBoneLayout)
        bone_part_count = self._read_u32()
        bone_offset1 = 0
        bone_offset2 = 0

        if lod.mesh_type >= 1:  # Skinned or Composite
            bone_offset1 = self._read_i64()
            bone_offset2 = self._read_i64()

            if lod.mesh_type == 2:  # Composite
                self._read_i64()  # part offset

        self._pad(16)

        # Read bone indices
        if lod.mesh_type == 1 and bone_offset1 != 0:  # Skinned
            saved2 = self._pos
            self._pos = bone_offset1
            for _ in range(bone_part_count):
                lod.bone_indices.append(self._read_u32())
            if bone_offset2 != 0:
                self._pos = bone_offset2
                for _ in range(bone_part_count):
                    lod.bone_short_names.append(self._read_u32())
            self._pos = saved2

        # Read strings
        saved2 = self._pos
        if str_offset2 != 0:
            self._pos = str_offset2
            lod.name = self._read_null_terminated()
        if str_offset3 != 0:
            self._pos = str_offset3
            lod.short_name = self._read_null_terminated()
        self._pos = saved2

        return lod

    def _parse_section(self, index: int) -> MeshSection:
        section = MeshSection()

        self._read_i64()  # runtime ptr (always 0)
        string_offset = self._read_i64()

        bone_count = 0
        section.material_id = self._read_i32()
        # DA:I: no lightMapUvMappingIndex

        section.primitive_count = self._read_u32()
        section.start_index = self._read_u32()
        section.vertex_offset = self._read_u32()
        section.vertex_count = self._read_u32()

        section.vertex_stride = self._read_u8()
        section.primitive_type = self._read_u8()
        section.bones_per_vertex = self._read_u8()
        bone_count = self._read_u8()

        bone_list_offset = self._read_i64()

        # Geometry declaration (1 for DA:I)
        elements = []
        for _ in range(MAX_ELEMENTS):
            usage = self._read_u8()
            fmt = self._read_u8()
            offset = self._read_u8()
            stream_idx = self._read_u8()
            elements.append(VertexElement(usage, fmt, offset, stream_idx))

        streams = []
        for _ in range(MAX_STREAMS):
            stride = self._read_u8()
            classification = self._read_u8()
            streams.append((stride, classification))

        elem_count = self._read_u8()
        stream_count = self._read_u8()
        self.stream.read(2)  # padding

        section.geom_decl = GeometryDecl(
            elements=elements[:elem_count],
            streams=streams[:stream_count],
            element_count=elem_count,
            stream_count=stream_count,
        )

        # TexCoord ratios
        for _ in range(6):
            section.tex_coord_ratios.append(self._read_f32())

        # Unknown data block (44 bytes for DA:I)
        self.stream.read(SECTION_UNKNOWN_BYTES)

        # Read bone list
        saved = self._pos
        if bone_list_offset != 0 and bone_count > 0:
            self._pos = bone_list_offset
            for _ in range(bone_count):
                section.bone_list.append(self._read_u16())

        # Read material name
        if string_offset != 0:
            self._pos = string_offset
            section.material_name = self._read_null_terminated()
        self._pos = saved

        return section


def extract_vertices(mesh: MeshSet, lod_index: int, chunk_data: bytes) -> dict:
    """Extract vertex data from a mesh LOD's chunk data.

    Returns dict with 'positions', 'normals', 'uvs', 'indices' as numpy arrays.
    """
    if lod_index >= len(mesh.lods):
        raise ValueError(f"LOD {lod_index} not found (mesh has {len(mesh.lods)} LODs)")

    lod = mesh.lods[lod_index]

    all_positions = []
    all_normals = []
    all_uvs = []
    all_indices = []
    vertex_offset_total = 0

    for section in lod.sections:
        if section.geom_decl is None:
            continue

        # Find position, normal, UV elements
        pos_elem = None
        normal_elem = None
        uv_elem = None

        for elem in section.geom_decl.elements:
            if elem.usage == VertexUsage.POS:
                pos_elem = elem
            elif elem.usage == VertexUsage.NORMAL:
                normal_elem = elem
            elif elem.usage == VertexUsage.TEXCOORD0:
                uv_elem = elem

        if pos_elem is None:
            continue

        # Extract vertices — vertex_offset is already in bytes
        vb_start = section.vertex_offset
        positions = []
        normals = []
        uvs = []

        for v in range(section.vertex_count):
            vert_start = vb_start + v * section.vertex_stride

            # Position
            pos_data = chunk_data[vert_start + pos_elem.offset:]
            pos = _read_vertex_data(pos_data, pos_elem.format)
            positions.append(pos[:3])

            # Normal
            if normal_elem:
                norm_data = chunk_data[vert_start + normal_elem.offset:]
                norm = _read_vertex_data(norm_data, normal_elem.format)
                normals.append(norm[:3] if len(norm) >= 3 else norm)

            # UV
            if uv_elem:
                uv_data = chunk_data[vert_start + uv_elem.offset:]
                uv = _read_vertex_data(uv_data, uv_elem.format)
                uvs.append(uv[:2] if len(uv) >= 2 else uv)

        # Extract indices
        ib_start = lod.vertex_buffer_size + section.start_index * (2 if lod.index_buffer_format == 0 else 4)
        index_size = 2 if lod.index_buffer_format == 0 else 4
        indices = []

        for i in range(section.primitive_count * 3):  # triangles
            idx_data = chunk_data[ib_start + i * index_size:]
            if index_size == 2:
                idx = struct.unpack_from('<H', idx_data)[0]
            else:
                idx = struct.unpack_from('<I', idx_data)[0]
            indices.append(idx + vertex_offset_total)

        all_positions.extend(positions)
        all_normals.extend(normals)
        all_uvs.extend(uvs)
        all_indices.extend(indices)
        vertex_offset_total += section.vertex_count

    result = {
        'positions': np.array(all_positions, dtype=np.float32) if all_positions else np.empty((0, 3)),
        'indices': np.array(all_indices, dtype=np.int32).reshape(-1, 3) if all_indices else np.empty((0, 3), dtype=np.int32),
    }
    if all_normals:
        result['normals'] = np.array(all_normals, dtype=np.float32)
    if all_uvs:
        result['uvs'] = np.array(all_uvs, dtype=np.float32)

    return result


def _read_vertex_data(data: bytes, fmt: int) -> tuple:
    """Read vertex data based on format type."""
    if fmt == VertexFormat.FLOAT:
        return struct.unpack_from('<f', data)
    elif fmt == VertexFormat.FLOAT2:
        return struct.unpack_from('<2f', data)
    elif fmt == VertexFormat.FLOAT3:
        return struct.unpack_from('<3f', data)
    elif fmt == VertexFormat.FLOAT4:
        return struct.unpack_from('<4f', data)
    elif fmt == VertexFormat.HALF:
        h1 = struct.unpack_from('<H', data)[0]
        return (_half_to_float(h1),)
    elif fmt == VertexFormat.HALF2:
        h1, h2 = struct.unpack_from('<2H', data)
        return (_half_to_float(h1), _half_to_float(h2))
    elif fmt == VertexFormat.HALF3:
        h1, h2, h3 = struct.unpack_from('<3H', data)
        return (_half_to_float(h1), _half_to_float(h2), _half_to_float(h3))
    elif fmt == VertexFormat.HALF4:
        h1, h2, h3, h4 = struct.unpack_from('<4H', data)
        return (_half_to_float(h1), _half_to_float(h2), _half_to_float(h3), _half_to_float(h4))
    elif fmt == VertexFormat.BYTE4N:
        b1, b2, b3, b4 = struct.unpack_from('<4B', data)
        return (b1 / 255.0 * 2 - 1, b2 / 255.0 * 2 - 1, b3 / 255.0 * 2 - 1, b4 / 255.0 * 2 - 1)
    elif fmt == VertexFormat.UBYTE4N:
        b1, b2, b3, b4 = struct.unpack_from('<4B', data)
        return (b1 / 255.0, b2 / 255.0, b3 / 255.0, b4 / 255.0)
    elif fmt == VertexFormat.UBYTE4:
        return struct.unpack_from('<4B', data)
    elif fmt == VertexFormat.SHORT2N:
        s1, s2 = struct.unpack_from('<2h', data)
        return (s1 / 32767.0, s2 / 32767.0)
    elif fmt == VertexFormat.SHORT4N:
        s1, s2, s3, s4 = struct.unpack_from('<4h', data)
        return (s1 / 32767.0, s2 / 32767.0, s3 / 32767.0, s4 / 32767.0)
    elif fmt == VertexFormat.USHORT2N:
        s1, s2 = struct.unpack_from('<2H', data)
        return (s1 / 65535.0, s2 / 65535.0)
    elif fmt == VertexFormat.USHORT4N:
        s1, s2, s3, s4 = struct.unpack_from('<4H', data)
        return (s1 / 65535.0, s2 / 65535.0, s3 / 65535.0, s4 / 65535.0)
    else:
        return (0.0,)


def _half_to_float(h: int) -> float:
    """Convert IEEE 754 half-precision to float."""
    try:
        return struct.unpack('e', struct.pack('H', h))[0]
    except (struct.error, OverflowError):
        return 0.0
