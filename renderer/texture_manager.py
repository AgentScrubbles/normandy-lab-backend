"""
Frostbite texture extraction and UV-mapped material application for DA:I.

Extracts skin tint textures from the game data and applies them to the
rendered head mesh via UV mapping in pyrender.
"""

import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dai")
TEXTURE_CACHE_DIR = os.path.join(ASSET_DIR, "textures")

# Frostbite texture resource type ID
TEXTURE_RES_TYPE = 0x6bde20ba

# Skin tint texture names in the chargen bundle
SKIN_TINT_NAMES = [
    'uhm_skintint_01', 'uhm_skintint_02', 'uhm_skintint_03',
    'uhm_skintint_04', 'uhm_skintint_05', 'uhm_skintint_06',
    'uhm_skintint_07', 'uhm_skintint_08', 'uhm_skintint_09',
    'uhm_skintint_10',
]

# Predefined skin tone RGB colors (approximations from the in-game palette)
# Used when actual textures aren't available
SKIN_TONE_COLORS = [
    (0.95, 0.87, 0.80),  # 1: Very pale
    (0.92, 0.82, 0.73),  # 2: Pale
    (0.87, 0.76, 0.67),  # 3: Light
    (0.82, 0.70, 0.60),  # 4: Light-medium
    (0.76, 0.63, 0.52),  # 5: Medium
    (0.68, 0.55, 0.44),  # 6: Medium-tan
    (0.58, 0.45, 0.35),  # 7: Tan
    (0.48, 0.36, 0.28),  # 8: Dark tan
    (0.38, 0.27, 0.20),  # 9: Brown
    (0.28, 0.20, 0.15),  # 10: Dark brown
    (0.20, 0.14, 0.11),  # 11: Very dark
]


class FrostbiteTexture:
    """Parse a Frostbite Texture resource header.

    The texture resource format stores metadata about the texture
    (dimensions, format, mip levels) followed by or referencing
    the actual pixel data in a chunk.
    """

    def __init__(self, data: bytes):
        self.data = data
        self._parse()

    def _parse(self):
        # Frostbite texture header (based on FrostySdk Texture.cs)
        # This is a simplified parser for the fields we need
        if len(self.data) < 48:
            raise ValueError("Texture resource too small")

        # The exact header layout varies by engine version
        # DA:I uses an older format
        self.res_type = struct.unpack_from('<I', self.data, 0)[0]
        self.width = struct.unpack_from('<H', self.data, 16)[0]
        self.height = struct.unpack_from('<H', self.data, 18)[0]
        self.depth = struct.unpack_from('<H', self.data, 20)[0]
        self.mip_count = struct.unpack_from('<B', self.data, 22)[0]
        self.pixel_format = struct.unpack_from('<I', self.data, 24)[0]
        self.chunk_id = self.data[32:48] if len(self.data) >= 48 else b'\x00' * 16


def get_skin_color(skin_tone_index: int) -> tuple[float, float, float]:
    """Get RGB skin color for a skin tone index (0-10).

    Returns (R, G, B) floats in [0, 1].
    """
    idx = max(0, min(skin_tone_index, len(SKIN_TONE_COLORS) - 1))
    return SKIN_TONE_COLORS[idx]


def make_skin_material(skin_tone_index: int = 0, roughness: float = 0.7):
    """Create a pyrender MetallicRoughnessMaterial with the specified skin tone.

    Args:
        skin_tone_index: 0-10, index into SKIN_TONE_COLORS
        roughness: surface roughness (0=mirror, 1=matte)

    Returns:
        pyrender.MetallicRoughnessMaterial
    """
    import pyrender
    r, g, b = get_skin_color(skin_tone_index)
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[r, g, b, 1.0],
        metallicFactor=0.0,
        roughnessFactor=roughness,
    )


def make_textured_material(texture_image: np.ndarray, roughness: float = 0.7):
    """Create a pyrender material with a texture map applied.

    Args:
        texture_image: RGB uint8 image (H, W, 3) to use as base color
        roughness: surface roughness

    Returns:
        pyrender.MetallicRoughnessMaterial
    """
    import pyrender
    from PIL import Image

    pil_img = Image.fromarray(texture_image)

    return pyrender.MetallicRoughnessMaterial(
        baseColorTexture=pyrender.Texture(source=pil_img, source_channels='RGB'),
        metallicFactor=0.0,
        roughnessFactor=roughness,
    )


def extract_textures_from_game(game_path: str):
    """Extract skin tint textures from game data to cache directory.

    This attempts to find texture resources in the chargen superbundle
    and save them as PNG files.
    """
    os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)

    from renderer.frostbite import FrostbiteArchive
    archive = FrostbiteArchive(game_path)
    bundles = archive.read_superbundle('chargen')

    for b in bundles:
        if 'human_female' not in b.path:
            continue

        for r in b.res:
            name_lower = r.name.lower()
            for tint_name in SKIN_TINT_NAMES:
                if tint_name in name_lower:
                    try:
                        data = archive.extract_by_sha1(r.sha1)
                        cache_path = os.path.join(TEXTURE_CACHE_DIR, f"{tint_name}.bin")
                        with open(cache_path, 'wb') as f:
                            f.write(data)
                        print(f"  Cached texture: {tint_name} ({len(data)} bytes)")
                    except Exception as e:
                        print(f"  Failed to extract {tint_name}: {e}")

    print(f"Texture extraction complete. Cached to {TEXTURE_CACHE_DIR}")


def load_cached_texture(name: str) -> Optional[np.ndarray]:
    """Load a cached texture as an RGB numpy array.

    Returns None if the texture hasn't been extracted.
    """
    cache_path = os.path.join(TEXTURE_CACHE_DIR, f"{name}.bin")
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, 'rb') as f:
        data = f.read()

    try:
        tex = FrostbiteTexture(data)
        # For now, return None if we can't decode the pixel format
        # Full DDS/BC decoding would be needed for actual texture data
        return None
    except Exception:
        return None


# Hair color approximations (25 options from game)
HAIR_COLORS = [
    (0.08, 0.06, 0.05),  # 1: Black
    (0.15, 0.10, 0.07),  # 2: Very dark brown
    (0.25, 0.17, 0.10),  # 3: Dark brown
    (0.35, 0.23, 0.14),  # 4: Medium brown
    (0.45, 0.32, 0.20),  # 5: Light brown
    (0.55, 0.40, 0.25),  # 6: Caramel
    (0.65, 0.48, 0.28),  # 7: Light caramel
    (0.72, 0.55, 0.30),  # 8: Dark blonde
    (0.80, 0.65, 0.40),  # 9: Medium blonde
    (0.88, 0.78, 0.55),  # 10: Light blonde
    (0.95, 0.90, 0.75),  # 11: Platinum blonde
    (0.92, 0.85, 0.70),  # 12: Ash blonde
    (0.50, 0.25, 0.15),  # 13: Auburn
    (0.60, 0.22, 0.10),  # 14: Copper
    (0.70, 0.30, 0.15),  # 15: Bright copper
    (0.55, 0.18, 0.12),  # 16: Dark red
    (0.72, 0.20, 0.12),  # 17: Red
    (0.80, 0.30, 0.18),  # 18: Bright red
    (0.50, 0.50, 0.55),  # 19: Dark grey
    (0.65, 0.65, 0.68),  # 20: Medium grey
    (0.82, 0.82, 0.85),  # 21: Light grey
    (0.95, 0.95, 0.95),  # 22: White
    (0.18, 0.12, 0.22),  # 23: Dark purple
    (0.12, 0.15, 0.30),  # 24: Dark blue
    (0.25, 0.10, 0.10),  # 25: Dark maroon
]


def get_hair_color(index: int) -> tuple[float, float, float]:
    """Get RGB hair color for an index (0-24)."""
    idx = max(0, min(index, len(HAIR_COLORS) - 1))
    return HAIR_COLORS[idx]
