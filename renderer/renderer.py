"""
Headless mesh renderer using pyrender for DA:I character faces.

Renders extracted meshes with pyrender's offscreen EGL/OSMesa backend.
Supports multi-section head mesh (face/eyes/lashes) with distinct materials,
compositing hair meshes on top, makeup overlays, complexion blending,
and proper eye rendering with inner/outer iris colors.
"""

import numpy as np
import trimesh
import pyrender
import os
from typing import Optional

# Force offscreen rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dai")
HAIR_DIR = os.path.join(ASSET_DIR, "hair")
TEX_DIR = os.path.join(ASSET_DIR, "textures")

# Section boundaries from mesh_info.json
FACE_VERTS = 4794
EYE_VERTS = 446
LASH_VERTS = 282
FACE_TRIS = 8910
EYE_TRIS = 864
LASH_TRIS = 424

# Cached textures (loaded lazily)
_texture_cache = {}

# Complexion overlay textures — 14 options grouped by visual effect
# Index 0 = no complexion (default), others blend overlays
# type: 'blend' = lerp with tinted overlay, 'darken' = darken skin in mask areas
COMPLEXION_OVERLAYS = [
    None,                                              # 0: Default
    ('uhm_age_d', 'blend', 0.15),                     # 1: Light aging
    ('uhm_age_d', 'blend', 0.35),                     # 2: Medium aging
    ('uhm_age_d', 'blend', 0.55),                     # 3: Heavy aging
    ('uhm_rough_d', 'blend', 0.15),                   # 4: Light roughness
    ('uhm_rough_d', 'blend', 0.35),                   # 5: Medium roughness
    ('uhm_rough_d', 'blend', 0.55),                   # 6: Heavy roughness
    ('uh_bruised_d', 'blend', 0.15),                  # 7: Light bruising
    ('uh_bruised_d', 'blend', 0.35),                  # 8: Heavy bruising
    None,                                              # 9: (blight too colored, skip)
    ('uhm_scar01_m', 'darken', 0.4),                  # 10: Scar 1
    ('uhm_scar01_m', 'darken', 0.6),                  # 11: Scar 2 (reuse scar01, scars_02 is channel-packed)
    None,                                              # 12: (blight skip)
    ('uhm_rough_d', 'blend', 0.45),                   # 13: Weathered
]

# Eye iris colors — 51 distinct options
EYE_COLORS = [
    (0.35, 0.25, 0.15),  # 0: Dark brown
    (0.45, 0.30, 0.15),  # 1: Brown
    (0.55, 0.38, 0.20),  # 2: Light brown
    (0.50, 0.42, 0.22),  # 3: Hazel
    (0.42, 0.45, 0.25),  # 4: Green-hazel
    (0.30, 0.45, 0.25),  # 5: Green
    (0.25, 0.50, 0.30),  # 6: Forest green
    (0.20, 0.45, 0.40),  # 7: Teal
    (0.25, 0.40, 0.50),  # 8: Blue-grey
    (0.25, 0.35, 0.55),  # 9: Blue
    (0.20, 0.30, 0.60),  # 10: Deep blue
    (0.15, 0.25, 0.55),  # 11: Royal blue
    (0.30, 0.30, 0.50),  # 12: Grey-blue
    (0.40, 0.40, 0.45),  # 13: Grey
    (0.50, 0.48, 0.45),  # 14: Light grey
    (0.55, 0.50, 0.35),  # 15: Amber
    (0.60, 0.45, 0.20),  # 16: Gold
    (0.55, 0.35, 0.15),  # 17: Copper
    (0.50, 0.25, 0.15),  # 18: Dark amber
    (0.45, 0.20, 0.15),  # 19: Deep brown
    (0.38, 0.35, 0.30),  # 20: Warm grey
    (0.42, 0.38, 0.28),  # 21: Khaki
    (0.48, 0.40, 0.25),  # 22: Olive
    (0.35, 0.42, 0.30),  # 23: Sage
    (0.28, 0.40, 0.35),  # 24: Sea green
    (0.22, 0.38, 0.45),  # 25: Steel blue
    (0.30, 0.32, 0.48),  # 26: Slate
    (0.35, 0.28, 0.42),  # 27: Violet-grey
    (0.42, 0.30, 0.35),  # 28: Mauve
    (0.48, 0.32, 0.28),  # 29: Russet
    (0.52, 0.35, 0.22),  # 30: Ochre
    (0.45, 0.42, 0.35),  # 31: Taupe
    (0.38, 0.38, 0.42),  # 32: Pewter
    (0.32, 0.35, 0.45),  # 33: Storm blue
    (0.28, 0.42, 0.42),  # 34: Aquamarine
    (0.35, 0.45, 0.28),  # 35: Leaf green
    (0.45, 0.48, 0.22),  # 36: Yellow-green
    (0.55, 0.45, 0.18),  # 37: Honey
    (0.62, 0.42, 0.15),  # 38: Bright amber
    (0.50, 0.30, 0.20),  # 39: Chestnut
    (0.40, 0.22, 0.18),  # 40: Mahogany
    (0.30, 0.18, 0.15),  # 41: Near-black
    (0.55, 0.52, 0.48),  # 42: Silver
    (0.45, 0.35, 0.50),  # 43: Lavender
    (0.35, 0.25, 0.45),  # 44: Purple
    (0.25, 0.30, 0.50),  # 45: Indigo
    (0.20, 0.40, 0.50),  # 46: Cerulean
    (0.25, 0.48, 0.38),  # 47: Jade
    (0.30, 0.50, 0.22),  # 48: Emerald
    (0.48, 0.50, 0.18),  # 49: Chartreuse
    (0.58, 0.48, 0.15),  # 50: Topaz
]

# 12-color inner/outer iris palette (color picker options)
IRIS_PALETTE = [
    (0.35, 0.25, 0.15),  # 0: Brown
    (0.50, 0.35, 0.18),  # 1: Amber
    (0.55, 0.45, 0.22),  # 2: Hazel
    (0.30, 0.45, 0.25),  # 3: Green
    (0.25, 0.42, 0.38),  # 4: Teal
    (0.25, 0.35, 0.55),  # 5: Blue
    (0.40, 0.40, 0.45),  # 6: Grey
    (0.20, 0.28, 0.52),  # 7: Dark blue
    (0.42, 0.22, 0.18),  # 8: Dark brown
    (0.55, 0.50, 0.45),  # 9: Light grey
    (0.48, 0.42, 0.30),  # 10: Gold-brown
    (0.22, 0.45, 0.32),  # 11: Emerald
]


def _load_texture(name: str) -> Optional[np.ndarray]:
    """Load a texture PNG from the textures directory as RGB, with caching."""
    if name in _texture_cache:
        return _texture_cache[name]
    path = os.path.join(TEX_DIR, f'{name}.png')
    if not os.path.exists(path):
        _texture_cache[name] = None
        return None
    from PIL import Image as PILImage
    img = np.array(PILImage.open(path).convert('RGB'))
    _texture_cache[name] = img
    return img


def _load_texture_rgba(name: str) -> Optional[np.ndarray]:
    """Load a texture PNG with alpha channel."""
    key = name + '_rgba'
    if key in _texture_cache:
        return _texture_cache[key]
    path = os.path.join(TEX_DIR, f'{name}.png')
    if not os.path.exists(path):
        _texture_cache[key] = None
        return None
    from PIL import Image as PILImage
    img = np.array(PILImage.open(path).convert('RGBA'))
    _texture_cache[key] = img
    return img


def _make_tinted_face_texture(skin_tone: int = 0, complexion: int = 0,
                                makeup: dict = None) -> np.ndarray:
    """Create a face texture with skin tint, complexion overlay, and makeup.

    Layers (bottom to top):
    1. Base diffuse (hf_hed_inq_d) × skin tint color
    2. Complexion overlay (age/roughness/scars) blended at intensity
    3. Makeup overlays (blush, lips, eye shadow) painted on
    """
    base = _load_texture('hf_hed_inq_d')
    if base is None:
        from renderer.texture_manager import get_skin_color
        r, g, b = get_skin_color(skin_tone)
        flat = np.full((256, 256, 3), [int(r*255), int(g*255), int(b*255)], dtype=np.uint8)
        return flat

    # Load skin tint ramp
    tint_names = ['uhm_skintint_01', 'uhm_skintint_02', 'uhm_skintint_03',
                  None, 'uhm_skintint_05', 'uhm_skintint_06', 'uhm_skintint_07',
                  'uhm_skintint_08', 'uhm_skintint_09', 'uhm_skintint_10']

    idx = min(skin_tone, len(tint_names) - 1)
    tint_name = tint_names[idx]
    tint = _load_texture(tint_name) if tint_name else None

    if tint is not None:
        mid_row = tint.shape[0] // 2
        mid_col = tint.shape[1] // 2
        tint_color = tint[mid_row, mid_col].astype(np.float32) / 255.0
    else:
        from renderer.texture_manager import get_skin_color
        tint_color = np.array(get_skin_color(skin_tone), dtype=np.float32)

    # Layer 1: Base × tint
    rgb = base[:, :, :3].astype(np.float32) if base.shape[2] >= 3 else base.astype(np.float32)
    result = (rgb * tint_color[np.newaxis, np.newaxis, :]).clip(0, 255)

    # Layer 2: Complexion overlay
    if complexion > 0 and complexion < len(COMPLEXION_OVERLAYS):
        overlay_info = COMPLEXION_OVERLAYS[complexion]
        if overlay_info is not None:
            overlay_name, blend_mode, intensity = overlay_info
            overlay = _load_texture(overlay_name)
            if overlay is not None:
                from PIL import Image as PILImage
                if overlay.shape[:2] != result.shape[:2]:
                    pil_ov = PILImage.fromarray(overlay).resize(
                        (result.shape[1], result.shape[0]), PILImage.BILINEAR)
                    overlay = np.array(pil_ov).astype(np.float32)
                else:
                    overlay = overlay.astype(np.float32)

                if blend_mode == 'darken':
                    # Scar masks: darken skin where the mask has content (any channel)
                    mask = np.max(overlay, axis=2) / 255.0  # 0..1 per pixel
                    darken = mask[:, :, np.newaxis] * intensity
                    result = result * (1 - darken * 0.5)  # darken up to 50%
                else:
                    # Blend: lerp base texture with overlay tinted by skin color
                    ov_tinted = overlay * tint_color[np.newaxis, np.newaxis, :]
                    result = result * (1 - intensity) + ov_tinted * intensity

    return result.clip(0, 255).astype(np.uint8)



def _make_eye_texture(eye_color_idx: int = 0, outer_iris_idx: int = -1,
                       inner_iris_idx: int = -1, size: int = 512) -> np.ndarray:
    """Generate a proper eye texture with sclera, limbal ring, outer iris, inner iris, and pupil.

    The eye mesh UV maps the eyeball surface, with the iris area centered.
    This generates a texture that looks like an actual eye rather than a flat color fill.
    """
    img = np.ones((size, size, 3), dtype=np.float32)

    cy, cx = size // 2, size // 2

    # Define radii as fractions of size
    sclera_r = size * 0.48       # white of eye
    limbal_r = size * 0.38       # dark ring around iris
    outer_iris_r = size * 0.35   # outer iris zone
    inner_iris_r = size * 0.20   # inner iris zone
    pupil_r = size * 0.12        # pupil
    highlight_r = size * 0.04    # specular highlight

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float32)

    # Get colors
    main_ec = EYE_COLORS[min(eye_color_idx, len(EYE_COLORS) - 1)]
    if outer_iris_idx >= 0:
        outer_ec = IRIS_PALETTE[min(outer_iris_idx, len(IRIS_PALETTE) - 1)]
    else:
        outer_ec = main_ec
    if inner_iris_idx >= 0:
        inner_ec = IRIS_PALETTE[min(inner_iris_idx, len(IRIS_PALETTE) - 1)]
    else:
        # Inner iris is usually lighter/warmer than outer
        inner_ec = tuple(min(1.0, c + 0.1) for c in main_ec)

    # Sclera (off-white with slight pink towards edges)
    sclera_color = np.array([0.93, 0.91, 0.90])
    sclera_edge = np.array([0.85, 0.80, 0.78])
    sclera_t = np.clip(dist / sclera_r, 0, 1)
    for c in range(3):
        img[:, :, c] = sclera_color[c] * (1 - sclera_t) + sclera_edge[c] * sclera_t

    # Limbal ring (dark ring around iris edge)
    limbal_color = np.array([0.12, 0.10, 0.08])
    limbal_mask = (dist < limbal_r + 2) & (dist > outer_iris_r - 1)
    limbal_t = 1.0 - np.abs(dist - (limbal_r + outer_iris_r) / 2) / ((limbal_r - outer_iris_r) / 2 + 1)
    limbal_t = np.clip(limbal_t, 0, 1) ** 2
    for c in range(3):
        img[:, :, c] = np.where(limbal_mask,
                                 img[:, :, c] * (1 - limbal_t) + limbal_color[c] * limbal_t,
                                 img[:, :, c])

    # Outer iris — radial pattern with color variation
    outer_mask = dist < outer_iris_r
    outer_color = np.array(outer_ec, dtype=np.float32)
    outer_dark = outer_color * 0.6  # darker at edges
    iris_t = np.clip(dist / outer_iris_r, 0, 1)  # 0 at center, 1 at edge

    # Add radial streaks for natural iris pattern
    angle = np.arctan2(y - cy, x - cx).astype(np.float32)
    streak = (np.sin(angle * 12) * 0.5 + 0.5) * 0.15  # subtle radial variation

    for c in range(3):
        iris_val = outer_color[c] * (1 - iris_t * 0.4) + outer_dark[c] * (iris_t * 0.4) + streak * 0.1
        img[:, :, c] = np.where(outer_mask, iris_val, img[:, :, c])

    # Inner iris — lighter/warmer core
    inner_mask = dist < inner_iris_r
    inner_color = np.array(inner_ec, dtype=np.float32)
    inner_t = np.clip(dist / inner_iris_r, 0, 1)
    for c in range(3):
        blended = inner_color[c] * (1 - inner_t * 0.6) + outer_color[c] * (inner_t * 0.6)
        img[:, :, c] = np.where(inner_mask, blended, img[:, :, c])

    # Pupil with soft edge
    pupil_mask = dist < pupil_r
    pupil_edge = dist < pupil_r * 1.2
    pupil_t = np.clip((pupil_r * 1.2 - dist) / (pupil_r * 0.2 + 1), 0, 1)
    for c in range(3):
        img[:, :, c] = np.where(pupil_edge,
                                 img[:, :, c] * (1 - pupil_t) + 0.02 * pupil_t,
                                 img[:, :, c])
    for c in range(3):
        img[:, :, c] = np.where(pupil_mask, 0.02, img[:, :, c])

    # Specular highlights
    for hx_off, hy_off, hr_mult in [(-0.10, -0.10, 1.0), (0.06, -0.06, 0.4)]:
        h_cx = cx + size * hx_off
        h_cy = cy + size * hy_off
        h_dist = np.sqrt((x - h_cx) ** 2 + (y - h_cy) ** 2).astype(np.float32)
        h_r = highlight_r * hr_mult
        h_mask = h_dist < h_r
        h_t = np.clip(1.0 - h_dist / h_r, 0, 1) ** 2
        for c in range(3):
            img[:, :, c] = np.where(h_mask, img[:, :, c] + h_t * 0.8, img[:, :, c])

    return (img.clip(0, 1) * 255).astype(np.uint8)


def load_mesh_from_arrays(positions: np.ndarray, indices: np.ndarray,
                          normals: np.ndarray = None, uvs: np.ndarray = None) -> trimesh.Trimesh:
    """Create a trimesh from extracted vertex arrays."""
    mesh = trimesh.Trimesh(
        vertices=positions,
        faces=indices,
        vertex_normals=normals,
        process=False,
    )
    if uvs is not None:
        # Flip V for OpenGL convention (V=0 at bottom, texture stored top-down)
        flipped_uvs = uvs.copy()
        flipped_uvs[:, 1] = 1.0 - flipped_uvs[:, 1]
        mesh.visual = trimesh.visual.TextureVisuals(uv=flipped_uvs)
    return mesh


def load_mesh_from_npy(asset_dir: str) -> trimesh.Trimesh:
    """Load a mesh from saved numpy arrays."""
    positions = np.load(f'{asset_dir}/base_head_positions.npy')
    indices = np.load(f'{asset_dir}/base_head_indices.npy')
    normals = None
    uvs = None
    normals_path = f'{asset_dir}/base_head_normals.npy'
    uvs_path = f'{asset_dir}/base_head_uvs.npy'
    if os.path.exists(normals_path):
        normals = np.load(normals_path)
    if os.path.exists(uvs_path):
        uvs = np.load(uvs_path)
    return load_mesh_from_arrays(positions, indices, normals, uvs)


def _split_head_sections(positions: np.ndarray, indices: np.ndarray,
                         normals: np.ndarray = None, uvs: np.ndarray = None):
    """Split the combined head mesh into FACE, EYE, LASHES sections."""
    face_pos = positions[:FACE_VERTS]
    eye_pos = positions[FACE_VERTS:FACE_VERTS + EYE_VERTS]
    lash_pos = positions[FACE_VERTS + EYE_VERTS:]

    face_idx = indices[:FACE_TRIS]
    eye_idx = indices[FACE_TRIS:FACE_TRIS + EYE_TRIS] - FACE_VERTS
    lash_idx = indices[FACE_TRIS + EYE_TRIS:] - (FACE_VERTS + EYE_VERTS)

    face_norms = normals[:FACE_VERTS] if normals is not None else None
    eye_norms = normals[FACE_VERTS:FACE_VERTS + EYE_VERTS] if normals is not None else None
    lash_norms = normals[FACE_VERTS + EYE_VERTS:] if normals is not None else None

    face_uvs = uvs[:FACE_VERTS] if uvs is not None else None
    eye_uvs = uvs[FACE_VERTS:FACE_VERTS + EYE_VERTS] if uvs is not None else None
    lash_uvs = uvs[FACE_VERTS + EYE_VERTS:] if uvs is not None else None

    return (face_pos, face_idx, face_norms, face_uvs,
            eye_pos, eye_idx, eye_norms, eye_uvs,
            lash_pos, lash_idx, lash_norms, lash_uvs)


def _load_hair_mesh(hair_name: str) -> Optional[trimesh.Trimesh]:
    """Load a hair mesh from cached numpy arrays, including UVs."""
    pos_path = os.path.join(HAIR_DIR, f'{hair_name}_positions.npy')
    idx_path = os.path.join(HAIR_DIR, f'{hair_name}_indices.npy')
    if not os.path.exists(pos_path):
        return None
    pos = np.load(pos_path)
    idx = np.load(idx_path)
    norms_path = os.path.join(HAIR_DIR, f'{hair_name}_normals.npy')
    norms = np.load(norms_path) if os.path.exists(norms_path) else None
    uvs_path = os.path.join(HAIR_DIR, f'{hair_name}_uvs.npy')
    uvs = np.load(uvs_path) if os.path.exists(uvs_path) else None
    return load_mesh_from_arrays(pos, idx, norms, uvs)


def _make_tinted_hair_texture(hair_color: tuple, tex_name: str = 'hair_wavy_thick_d') -> np.ndarray:
    """Create a hair card texture tinted by hair color, preserving alpha."""
    base = _load_texture_rgba(tex_name)
    if base is None:
        flat = np.full((256, 256, 4), [int(hair_color[0]*255),
                                        int(hair_color[1]*255),
                                        int(hair_color[2]*255), 255], dtype=np.uint8)
        return flat

    hc = np.array([hair_color[0], hair_color[1], hair_color[2], 1.0], dtype=np.float32)
    tinted = (base.astype(np.float32) * hc[np.newaxis, np.newaxis, :]).clip(0, 255).astype(np.uint8)
    return tinted


def get_available_hair_styles() -> list[str]:
    """List available hair mesh names."""
    if not os.path.exists(HAIR_DIR):
        return []
    import json
    manifest_path = os.path.join(HAIR_DIR, 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return [h['name'] for h in json.load(f)]
    return sorted(set(
        f.replace('_positions.npy', '')
        for f in os.listdir(HAIR_DIR)
        if f.endswith('_positions.npy')
    ))


def render_face(mesh: trimesh.Trimesh, width: int = 512, height: int = 512,
                yaw: float = 0.0, pitch: float = 0.0,
                material: 'pyrender.MetallicRoughnessMaterial' = None,
                skin_tone: int = None) -> np.ndarray:
    """Render a single face mesh (legacy API, no eyes/hair)."""
    if material is None:
        if skin_tone is not None:
            from renderer.texture_manager import make_skin_material
            material = make_skin_material(skin_tone)
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.7, 0.65, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.7,
            )
    return _render_scene([mesh], [material], width, height, yaw, pitch)


def render_character(positions: np.ndarray, indices: np.ndarray,
                     normals: np.ndarray = None, uvs: np.ndarray = None,
                     width: int = 512, height: int = 512,
                     yaw: float = 0.0, pitch: float = 0.0,
                     skin_tone: int = 0, eye_color: int = 0,
                     hair_style: str = None, hair_color: tuple = None,
                     makeup: dict = None, complexion: int = 0,
                     outer_iris_color: int = -1, inner_iris_color: int = -1,
                     eyelash_color: tuple = None) -> np.ndarray:
    """Render a complete character with face, eyes, lashes, and hair.

    Args:
        positions: (N, 3) vertex positions for full head mesh (face+eye+lash)
        indices: (M, 3) face indices for full head mesh
        normals: optional (N, 3) vertex normals
        uvs: optional (N, 2) UV coordinates
        width, height: image dimensions
        yaw, pitch: camera angles in degrees
        skin_tone: skin tone index (0-10)
        eye_color: eye color index (0-50) — main color
        hair_style: hair mesh name or None for bald
        hair_color: (R,G,B) tuple for hair
        makeup: dict with blush/lip/eye_shadow/eyeliner params
        complexion: complexion index (0-13)
        outer_iris_color: outer iris color picker index (0-11), -1 = use eye_color
        inner_iris_color: inner iris color picker index (0-11), -1 = use eye_color
        eyelash_color: (R,G,B) tuple for lash color, None = dark
    """
    from PIL import Image as PILImage

    # Load UVs from file if not provided
    if uvs is None:
        uvs_path = os.path.join(ASSET_DIR, 'base_head_uvs.npy')
        if os.path.exists(uvs_path):
            uvs = np.load(uvs_path)

    # Split head into sections
    (face_pos, face_idx, face_norms, face_uvs,
     eye_pos, eye_idx, eye_norms, eye_uvs,
     lash_pos, lash_idx, lash_norms, lash_uvs) = _split_head_sections(
        positions, indices, normals, uvs)

    meshes = []
    materials = []

    # Face section — textured with diffuse + skin tint + complexion + makeup
    face_tex = _make_tinted_face_texture(skin_tone, complexion, makeup)
    face_pil = PILImage.fromarray(face_tex)
    face_mat = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=pyrender.Texture(source=face_pil, source_channels='RGB'),
        metallicFactor=0.0,
        roughnessFactor=0.6,
    )
    face_mesh = load_mesh_from_arrays(face_pos, face_idx, face_norms, face_uvs)
    meshes.append(face_mesh)
    materials.append(face_mat)

    # Eye section — proper eye texture with sclera/iris/pupil
    eye_tex = _make_eye_texture(eye_color, outer_iris_color, inner_iris_color)
    eye_pil = PILImage.fromarray(eye_tex)
    eye_mat = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=pyrender.Texture(source=eye_pil, source_channels='RGB'),
        metallicFactor=0.1,
        roughnessFactor=0.2,
    )
    eye_mesh = load_mesh_from_arrays(eye_pos, eye_idx, eye_norms, eye_uvs)
    meshes.append(eye_mesh)
    materials.append(eye_mat)

    # Lashes section — colored by eyelash_color param
    if eyelash_color is not None:
        lash_rgb = [eyelash_color[0], eyelash_color[1], eyelash_color[2], 1.0]
    else:
        lash_rgb = [0.05, 0.04, 0.03, 1.0]
    lash_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=lash_rgb,
        metallicFactor=0.0,
        roughnessFactor=0.4,
    )
    meshes.append(load_mesh_from_arrays(lash_pos, lash_idx, lash_norms, lash_uvs))
    materials.append(lash_mat)

    # Hair mesh
    if hair_style:
        hair_mesh = _load_hair_mesh(hair_style)
        if hair_mesh is not None:
            hc = hair_color or (0.25, 0.17, 0.10)
            hair_tex = _make_tinted_hair_texture(hc)
            hair_pil = PILImage.fromarray(hair_tex)
            hair_mat = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=pyrender.Texture(source=hair_pil, source_channels='RGBA'),
                metallicFactor=0.05,
                roughnessFactor=0.55,
                alphaMode='MASK',
                alphaCutoff=0.3,
            )
            meshes.append(hair_mesh)
            materials.append(hair_mat)

    rendered = _render_scene(meshes, materials, width, height, yaw, pitch)

    # Post-process: apply makeup overlays on the rendered image
    # Uses landmark detection (slower but precise), falls back to approx
    if makeup:
        rendered = _apply_makeup_post(rendered, makeup)

    return rendered


def _apply_makeup_post(image: np.ndarray, makeup: dict) -> np.ndarray:
    """Apply makeup to rendered image using face landmark detection.

    Uses dlib face landmarks to find precise eye, lip, cheek regions
    and applies colored overlays with blend modes.

    Falls back to approximate image-space placement if face detection fails.
    """
    # Check if any makeup has meaningful intensity
    has_makeup = False
    for key in ['lip_color', 'blush', 'eye_shadow', 'eyeliner']:
        if makeup.get(key, {}).get('intensity', 0) > 0.05:
            has_makeup = True
            break
    if not has_makeup:
        return image

    # Try face detection for landmark-based makeup
    landmarks = None
    try:
        import face_recognition
        img_contiguous = np.ascontiguousarray(image)
        locations = face_recognition.face_locations(img_contiguous, model='cnn')
        if locations:
            landmarks_list = face_recognition.face_landmarks(img_contiguous, locations)
            if landmarks_list:
                landmarks = landmarks_list[0]
    except Exception:
        pass

    if landmarks is None:
        # Fallback: approximate placement based on image center
        return _apply_makeup_approx(image, makeup)

    return _apply_makeup_landmarks(image, makeup, landmarks)


# Cache for face_recognition model loading (first call is slow, subsequent fast)
_face_model_loaded = False


def _apply_makeup_landmarks(image: np.ndarray, makeup: dict, landmarks: dict) -> np.ndarray:
    """Apply makeup using detected face landmarks."""
    import cv2
    result = image.astype(np.float32)
    h, w = result.shape[:2]

    def _fill_region_soft(pts, color, intensity, expand=1.0):
        pts_arr = np.array(pts, dtype=np.int32)
        if expand != 1.0:
            center = pts_arr.mean(axis=0)
            pts_arr = ((pts_arr - center) * expand + center).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_arr], 255)
        ksize = max(3, int(min(h, w) * 0.03)) | 1
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0) / 255.0
        color_f = np.array(color, dtype=np.float32) * 255
        alpha = mask_f[:, :, np.newaxis] * intensity
        return alpha, color_f

    # Lip color
    lip = makeup.get('lip_color', {})
    if lip.get('intensity', 0) > 0.05 and 'top_lip' in landmarks and 'bottom_lip' in landmarks:
        lip_pts = landmarks['top_lip'] + landmarks['bottom_lip']
        alpha, color_f = _fill_region_soft(lip_pts, lip['color'], lip['intensity'] * 0.5)
        result = result * (1 - alpha) + color_f * alpha + result * alpha * 0.3

    # Blush
    blush = makeup.get('blush', {})
    if blush.get('intensity', 0) > 0.05 and 'chin' in landmarks:
        chin = landmarks['chin']
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            left_eye_center = np.array(landmarks['left_eye']).mean(axis=0)
            right_eye_center = np.array(landmarks['right_eye']).mean(axis=0)
            left_cheek_center = (left_eye_center + np.array(chin[2])) / 2
            right_cheek_center = (right_eye_center + np.array(chin[-3])) / 2
            cheek_r = int(np.linalg.norm(left_eye_center - right_eye_center) * 0.25)

            for cc in [left_cheek_center, right_cheek_center]:
                y_coords, x_coords = np.ogrid[:h, :w]
                dist = np.sqrt((x_coords - cc[0]) ** 2 + (y_coords - cc[1]) ** 2).astype(np.float32)
                mask = np.clip(1.0 - dist / cheek_r, 0, 1) ** 2
                color_f = np.array(blush['color'], dtype=np.float32) * 255
                alpha = mask[:, :, np.newaxis] * blush['intensity'] * 0.35
                result = result * (1 - alpha) + color_f * alpha + result * alpha * 0.2

    # Eye shadow
    eye_shadow = makeup.get('eye_shadow', {})
    if eye_shadow.get('intensity', 0) > 0.05:
        for eye_key in ['left_eyebrow', 'right_eyebrow']:
            if eye_key in landmarks:
                brow = landmarks[eye_key]
                eye_key2 = 'left_eye' if 'left' in eye_key else 'right_eye'
                if eye_key2 in landmarks:
                    eye = landmarks[eye_key2]
                    shadow_pts = list(brow) + list(reversed(eye))
                    alpha, color_f = _fill_region_soft(shadow_pts, eye_shadow['color'],
                                                        eye_shadow['intensity'] * 0.4, expand=1.1)
                    result = result * (1 - alpha) + color_f * alpha + result * alpha * 0.2

    # Eyeliner
    eyeliner = makeup.get('eyeliner', {})
    if eyeliner.get('intensity', 0) > 0.05:
        for eye_key in ['left_eye', 'right_eye']:
            if eye_key in landmarks:
                eye_pts = landmarks[eye_key]
                alpha, color_f = _fill_region_soft(eye_pts, eyeliner['color'],
                                                    eyeliner['intensity'] * 0.5)
                result = result * (1 - alpha) + color_f * alpha

    return result.clip(0, 255).astype(np.uint8)


def _apply_makeup_approx(image: np.ndarray, makeup: dict) -> np.ndarray:
    """Apply makeup using approximate image-space positions (no face detection).

    Uses fixed proportions relative to image center, assuming a centered frontal face.
    Less precise than landmark-based but much faster.
    """
    result = image.astype(np.float32)
    h, w = result.shape[:2]
    cx, cy = w * 0.5, h * 0.42  # face center (slightly above image center)

    y_coords, x_coords = np.ogrid[:h, :w]

    def _oval_blend(center_x, center_y, rx, ry, color, intensity):
        nonlocal result
        dist = ((x_coords - center_x) / rx) ** 2 + ((y_coords - center_y) / ry) ** 2
        mask = np.clip(1.0 - dist, 0, 1) ** 2
        color_f = np.array(color, dtype=np.float32) * 255
        alpha = mask[:, :, np.newaxis] * intensity
        result = result * (1 - alpha) + color_f * alpha + result * alpha * 0.2

    face_w = w * 0.35  # approximate face width

    # Lip color
    lip = makeup.get('lip_color', {})
    if lip.get('intensity', 0) > 0.05:
        _oval_blend(cx, cy + h * 0.15, face_w * 0.25, h * 0.025,
                     lip['color'], lip['intensity'] * 0.45)

    # Blush
    blush = makeup.get('blush', {})
    if blush.get('intensity', 0) > 0.05:
        for offset in [-0.12, 0.12]:
            _oval_blend(cx + w * offset, cy + h * 0.05,
                         face_w * 0.15, h * 0.05,
                         blush['color'], blush['intensity'] * 0.3)

    # Eye shadow
    eye_shadow = makeup.get('eye_shadow', {})
    if eye_shadow.get('intensity', 0) > 0.05:
        for offset in [-0.08, 0.08]:
            _oval_blend(cx + w * offset, cy - h * 0.03,
                         face_w * 0.12, h * 0.025,
                         eye_shadow['color'], eye_shadow['intensity'] * 0.35)

    # Eyeliner
    eyeliner = makeup.get('eyeliner', {})
    if eyeliner.get('intensity', 0) > 0.05:
        for offset in [-0.08, 0.08]:
            _oval_blend(cx + w * offset, cy,
                         face_w * 0.10, h * 0.008,
                         eyeliner['color'], eyeliner['intensity'] * 0.4)

    return result.clip(0, 255).astype(np.uint8)


def _render_scene(meshes: list[trimesh.Trimesh],
                  materials: list['pyrender.MetallicRoughnessMaterial'],
                  width: int, height: int,
                  yaw: float, pitch: float) -> np.ndarray:
    """Render a scene with multiple meshes."""
    scene = pyrender.Scene(
        bg_color=[0.2, 0.2, 0.2, 1.0],
        ambient_light=[0.3, 0.3, 0.3],
    )

    # Compute combined bounds for camera placement
    all_bounds_min = np.full(3, np.inf)
    all_bounds_max = np.full(3, -np.inf)

    for mesh, mat in zip(meshes, materials):
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
        scene.add(py_mesh)
        all_bounds_min = np.minimum(all_bounds_min, mesh.bounds[0])
        all_bounds_max = np.maximum(all_bounds_max, mesh.bounds[1])

    center = (all_bounds_min + all_bounds_max) / 2
    extent = (all_bounds_max - all_bounds_min).max()
    camera_distance = extent * 2.5

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.radians(30), aspectRatio=width / height)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    cam_x = center[0] + camera_distance * np.sin(yaw_rad) * np.cos(pitch_rad)
    cam_y = center[1] + camera_distance * np.sin(pitch_rad)
    cam_z = center[2] + camera_distance * np.cos(yaw_rad) * np.cos(pitch_rad)
    camera_pose = _look_at(np.array([cam_x, cam_y, cam_z]), center, np.array([0.0, 1.0, 0.0]))
    scene.add(camera, pose=camera_pose)

    # 3-point lighting
    key_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=3.0)
    scene.add(key_light, pose=_look_at(
        center + np.array([0.5, 0.5, 1.0]) * extent, center, np.array([0.0, 1.0, 0.0])))

    fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.5)
    scene.add(fill_light, pose=_look_at(
        center + np.array([-0.5, 0.2, 0.8]) * extent, center, np.array([0.0, 1.0, 0.0])))

    rim_light = pyrender.DirectionalLight(color=[0.9, 0.9, 1.0], intensity=1.0)
    scene.add(rim_light, pose=_look_at(
        center + np.array([0.0, 0.3, -1.0]) * extent, center, np.array([0.0, 1.0, 0.0])))

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()
    return color


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 look-at camera matrix (OpenGL convention)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


if __name__ == '__main__':
    import sys
    from PIL import Image

    asset_dir = sys.argv[1] if len(sys.argv) > 1 else 'assets/dai'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/dai_face_render.png'

    print(f'Loading mesh from {asset_dir}...')
    mesh = load_mesh_from_npy(asset_dir)
    print(f'  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}')

    print(f'Rendering...')
    image = render_face(mesh, width=512, height=512)
    Image.fromarray(image).save(output_path)
    print(f'Saved to {output_path}')
