"""
Morph target engine for DA:I character heads.

Applies morph target deformations to the base head mesh based on slider values.
Reads the MorphTargetsResource binary format.
"""

import struct
from dataclasses import dataclass, field
from io import BytesIO
import numpy as np


@dataclass
class ShapeTarget:
    """A mesh shape target (absolute positions for a specific mesh variant)."""
    name: str
    data_offset: int      # offset to float4 position data
    norm_offset: int       # offset to float4 normal data
    lod0_verts: int


@dataclass
class TwoWaySlider:
    """A two-way morph slider with left and right deltas."""
    left_data_offset: int    # offset to left delta positions
    left_norm_offset: int    # offset to left delta normals
    right_data_offset: int   # offset to right delta positions
    right_norm_offset: int   # offset to right delta normals
    left_max: float
    right_max: float


@dataclass
class OneWaySlider:
    """A one-way morph slider with a single delta direction."""
    data_offset: int      # offset to delta positions
    norm_offset: int      # offset to delta normals
    target_max: float


class MorphTargetResource:
    """Parses and manages morph target data from a DA:I MorphTargetsResource."""

    def __init__(self, data: bytes):
        self.data = data
        self._parse()

    def _parse(self):
        header = struct.unpack_from('<12I', self.data, 0)
        self.num_unknown0 = header[0]      # 17
        self.num_lods = header[1]           # 3
        self.num_unknown2 = header[2]       # 8
        self.num_shapes_sec1 = header[3]    # 6
        self.num_shapes_sec4 = header[4]    # 36
        self.num_twoway = header[5]         # 68
        self.num_oneway = header[6]         # 49
        self.num_total = header[7]          # 211
        self.num_unknown8 = header[8]       # 1
        self.lod0_verts = header[9]         # 5522
        self.lod1_verts = header[10]        # 3969
        self.lod2_verts = header[11]        # 1898
        self.total_verts = self.lod0_verts + self.lod1_verts + self.lod2_verts

        # Parse offset table (9 uint64 entries starting at 0x3c)
        self.offsets = []
        for i in range(9):
            off = struct.unpack_from('<Q', self.data, 0x3c + i * 8)[0]
            self.offsets.append(off)

        # Parse Section 1: shape targets (head mesh variants)
        self.shape_targets = []
        sec1_start = self.offsets[1]  # 0x3f0
        for i in range(self.num_shapes_sec1):
            base = sec1_start + i * 28  # 7 uint32s per entry
            vals = struct.unpack_from('<7I', self.data, base)
            str_off = vals[0]
            data_off = vals[2]
            norm_off = vals[4]

            name = self._read_string(str_off)
            self.shape_targets.append(ShapeTarget(
                name=name,
                data_offset=data_off,
                norm_offset=norm_off,
                lod0_verts=self.lod0_verts,
            ))

        # Parse Section 2: two-way sliders
        self.twoway_sliders = []
        sec2_start = self.offsets[2]  # 0x4a0
        entry_size = 44  # 11 uint32s
        for i in range(self.num_twoway):
            base = sec2_start + i * entry_size
            left_off = struct.unpack_from('<Q', self.data, base)[0]
            left_norm = struct.unpack_from('<Q', self.data, base + 8)[0]
            right_off = struct.unpack_from('<Q', self.data, base + 16)[0]
            right_norm = struct.unpack_from('<Q', self.data, base + 24)[0]
            left_max = struct.unpack_from('<f', self.data, base + 32)[0]
            right_max = struct.unpack_from('<f', self.data, base + 36)[0]

            self.twoway_sliders.append(TwoWaySlider(
                left_data_offset=left_off,
                left_norm_offset=left_norm,
                right_data_offset=right_off,
                right_norm_offset=right_norm,
                left_max=left_max,
                right_max=right_max,
            ))

        # Parse Section 3: one-way sliders
        self.oneway_sliders = []
        sec3_start = self.offsets[3]  # 0x1050
        sec3_size = self.offsets[4] - self.offsets[3]
        oneway_entry_size = sec3_size // self.num_oneway if self.num_oneway > 0 else 0

        if oneway_entry_size > 0:
            for i in range(self.num_oneway):
                base = sec3_start + i * oneway_entry_size
                data_off = struct.unpack_from('<Q', self.data, base)[0]
                norm_off = struct.unpack_from('<Q', self.data, base + 8)[0]
                target_max = struct.unpack_from('<f', self.data, base + 16)[0]

                self.oneway_sliders.append(OneWaySlider(
                    data_offset=data_off,
                    norm_offset=norm_off,
                    target_max=target_max,
                ))

    def _read_string(self, offset: int) -> str:
        if offset == 0 or offset >= len(self.data):
            return ""
        end = self.data.index(b'\x00', offset) if b'\x00' in self.data[offset:] else len(self.data)
        return self.data[offset:end].decode('utf-8', errors='replace')

    def read_positions(self, offset: int) -> np.ndarray:
        """Read LOD0 vertex positions from a data offset."""
        positions = np.zeros((self.lod0_verts, 3), dtype=np.float32)
        for v in range(self.lod0_verts):
            off = offset + v * 16
            if off + 16 <= len(self.data):
                x, y, z = struct.unpack_from('<3f', self.data, off)
                positions[v] = [x, y, z]
        return positions

    def get_shape_positions(self, shape_idx: int) -> np.ndarray:
        """Get absolute vertex positions for a shape target."""
        if shape_idx >= len(self.shape_targets):
            raise IndexError(f"Shape index {shape_idx} >= {len(self.shape_targets)}")
        return self.read_positions(self.shape_targets[shape_idx].data_offset)

    def get_twoway_deltas(self, slider_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get left and right position deltas for a two-way slider."""
        if slider_idx >= len(self.twoway_sliders):
            raise IndexError(f"Slider index {slider_idx} >= {len(self.twoway_sliders)}")
        slider = self.twoway_sliders[slider_idx]
        left = self.read_positions(slider.left_data_offset)
        right = self.read_positions(slider.right_data_offset)
        return left, right

    def get_oneway_deltas(self, slider_idx: int) -> np.ndarray:
        """Get position deltas for a one-way slider."""
        if slider_idx >= len(self.oneway_sliders):
            raise IndexError(f"Slider index {slider_idx} >= {len(self.oneway_sliders)}")
        return self.read_positions(self.oneway_sliders[slider_idx].data_offset)

    def apply_morphs(self, base_positions: np.ndarray,
                     twoway_weights: list[float] = None,
                     oneway_weights: list[float] = None) -> np.ndarray:
        """Apply morph target deformations to base positions.

        Args:
            base_positions: (N, 3) base vertex positions
            twoway_weights: list of weights for two-way sliders (-1 to +1)
                           negative = left direction, positive = right
            oneway_weights: list of weights for one-way sliders (0 to 1)

        Returns:
            (N, 3) deformed vertex positions
        """
        result = base_positions.copy()

        if twoway_weights:
            for i, weight in enumerate(twoway_weights):
                if i >= len(self.twoway_sliders) or abs(weight) < 1e-6:
                    continue

                left_delta, right_delta = self.get_twoway_deltas(i)

                if weight < 0:
                    # Apply left delta
                    result += left_delta * abs(weight)
                else:
                    # Apply right delta
                    result += right_delta * weight

        if oneway_weights:
            for i, weight in enumerate(oneway_weights):
                if i >= len(self.oneway_sliders) or abs(weight) < 1e-6:
                    continue

                delta = self.get_oneway_deltas(i)
                result += delta * weight

        return result
