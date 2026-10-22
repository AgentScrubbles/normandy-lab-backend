#!/usr/bin/env python3
"""
DA:I standalone character data collection tool.

Generates random morph slider values, renders faces using the standalone
renderer (no game needed), and saves screenshots for training data.

Generates quantized slider values matching the 56 game UI inputs,
encodes them in underscore-separated filenames for parse_filename().
Also extracts facial metrics (25 ratios + 128-dim encoding) per image.

Usage:
    python tools/dai_standalone_capture.py --count 100
    python tools/dai_standalone_capture.py --count 10000 --width 256 --height 256
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from renderer.morph_engine import MorphTargetResource
from renderer.input_mapper import build_input_mapping, ui_values_to_morph_weights, load_mapping
from renderer.renderer import render_character, get_available_hair_styles
from renderer.texture_manager import SKIN_TONE_COLORS, get_hair_color
from PIL import Image


# Paths
GAME_PATH = "/mnt/10TBRaid/games/drop/dragon-age-inquisition"
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dai")
MORPH_RES_CACHE = os.path.join(ASSET_DIR, "morph_targets.res")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "dai", "female")

# UI input option counts (matching dai_human_female.py schema)
UI_OPTION_COUNTS = [
    # General (6)
    7, 11, 14, 51, 28, 25,
    # Head/Ears grids (7): each grid = 10x10
    100, 100, 100, 100, 100, 100, 100,
    # Eyes (11)
    8, 100, 100, 100, 23, 10, 14, 12, 12,
    # Nose (9)
    8, 100, 100, 100, 100, 100, 100, 100, 100,
    # Mouth/Jaw (11)
    9, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # Makeup (14): colors=12, intensities=11
    12, 11, 12, 12, 11, 12, 12, 11, 12, 11, 12, 11, 12, 11,
]

# Makeup color palettes — 12 colors each (indexed by ui_values)
MAKEUP_PALETTE = {
    'MakeupColorEyesA': [  # Eye shadow
        (0.15, 0.10, 0.08), (0.25, 0.15, 0.12), (0.35, 0.20, 0.15),
        (0.20, 0.12, 0.25), (0.15, 0.10, 0.30), (0.10, 0.15, 0.25),
        (0.12, 0.20, 0.18), (0.25, 0.22, 0.15), (0.30, 0.25, 0.20),
        (0.18, 0.18, 0.22), (0.08, 0.08, 0.10), (0.35, 0.30, 0.28),
    ],
    'MakeupColorEyesB': [  # Under-eye
        (0.20, 0.12, 0.10), (0.30, 0.18, 0.15), (0.25, 0.15, 0.20),
        (0.18, 0.12, 0.25), (0.15, 0.12, 0.28), (0.12, 0.15, 0.22),
        (0.20, 0.20, 0.18), (0.28, 0.22, 0.18), (0.22, 0.18, 0.15),
        (0.15, 0.15, 0.18), (0.10, 0.08, 0.12), (0.30, 0.25, 0.22),
    ],
    'MakeupColorUnderbrow': [
        (0.85, 0.80, 0.75), (0.78, 0.72, 0.68), (0.72, 0.65, 0.60),
        (0.82, 0.78, 0.72), (0.75, 0.70, 0.68), (0.68, 0.62, 0.58),
        (0.80, 0.75, 0.70), (0.72, 0.68, 0.62), (0.65, 0.60, 0.55),
        (0.78, 0.74, 0.70), (0.70, 0.65, 0.60), (0.62, 0.58, 0.52),
    ],
    'MakeupColorEyeliner': [
        (0.05, 0.04, 0.03), (0.12, 0.08, 0.06), (0.20, 0.12, 0.08),
        (0.08, 0.06, 0.15), (0.06, 0.06, 0.20), (0.10, 0.12, 0.08),
        (0.15, 0.10, 0.08), (0.18, 0.15, 0.12), (0.08, 0.08, 0.08),
        (0.25, 0.18, 0.12), (0.03, 0.03, 0.05), (0.30, 0.22, 0.15),
    ],
    'MakeupColorCheeks': [  # Blush
        (0.75, 0.45, 0.42), (0.80, 0.50, 0.45), (0.85, 0.55, 0.48),
        (0.70, 0.40, 0.38), (0.72, 0.42, 0.40), (0.78, 0.48, 0.42),
        (0.82, 0.52, 0.45), (0.68, 0.38, 0.35), (0.75, 0.45, 0.40),
        (0.80, 0.50, 0.42), (0.65, 0.35, 0.32), (0.85, 0.55, 0.50),
    ],
    'MakeupColorLips': [  # Lip color
        (0.65, 0.25, 0.22), (0.72, 0.30, 0.25), (0.80, 0.35, 0.28),
        (0.58, 0.20, 0.18), (0.50, 0.18, 0.22), (0.45, 0.15, 0.25),
        (0.70, 0.28, 0.24), (0.78, 0.32, 0.26), (0.85, 0.38, 0.30),
        (0.55, 0.22, 0.20), (0.40, 0.12, 0.15), (0.90, 0.42, 0.35),
    ],
    'MakeupColorLipLiner': [
        (0.45, 0.18, 0.15), (0.52, 0.22, 0.18), (0.60, 0.25, 0.20),
        (0.38, 0.15, 0.12), (0.35, 0.12, 0.18), (0.30, 0.10, 0.20),
        (0.50, 0.20, 0.16), (0.55, 0.22, 0.18), (0.62, 0.28, 0.22),
        (0.40, 0.16, 0.14), (0.28, 0.08, 0.10), (0.65, 0.30, 0.25),
    ],
}


def extract_morph_resource():
    """Extract the morph targets resource from the game if not cached."""
    if os.path.exists(MORPH_RES_CACHE):
        return

    print("Extracting morph target resource from game data...")
    from renderer.frostbite import FrostbiteArchive
    archive = FrostbiteArchive(GAME_PATH)
    bundles = archive.read_superbundle('chargen')

    for b in bundles:
        if 'human_female' not in b.path:
            continue
        for r in b.res:
            if 'humanfemaletargets' in r.name and r.asset_type == 'res':
                data = archive.extract_by_sha1(r.sha1)
                with open(MORPH_RES_CACHE, 'wb') as f:
                    f.write(data)
                print(f"  Saved {len(data)} bytes to {MORPH_RES_CACHE}")
                return
        break

    raise RuntimeError("Failed to find morph targets resource")


def generate_random_ui_values(rng: np.random.Generator) -> list[int]:
    """Generate random quantized UI input values matching the 56 inputs.

    Returns list of 56 integer values, each in [0, option_count-1].
    """
    values = []
    for count in UI_OPTION_COUNTS:
        values.append(int(rng.integers(0, count)))
    return values


def values_to_filename(ui_values: list[int]) -> str:
    """Encode UI values as underscore-separated filename."""
    return '_'.join(str(v) for v in ui_values)


def extract_makeup_params(ui_values: list[int], mapping) -> dict:
    """Extract makeup rendering parameters from UI values.

    Returns dict with keys: eye_shadow, under_eye, blush, lip_color, lip_liner,
    eyeliner, eyelash_color, each with 'color' and 'intensity' (0-1).
    """
    params = {}

    # Map UI indices to makeup params
    # [42] Eye Shadow Color, [43] Eye Shadow Intensity
    params['eye_shadow'] = {
        'color': MAKEUP_PALETTE['MakeupColorEyesA'][ui_values[42] % 12],
        'intensity': ui_values[43] / 10.0,  # 0-10 → 0-1
    }
    # [44] Under-Eye Color (intensity shared with eye shadow)
    params['under_eye'] = {
        'color': MAKEUP_PALETTE['MakeupColorEyesB'][ui_values[44] % 12],
        'intensity': ui_values[43] / 10.0,
    }
    # [45] Under-Brow Color, [46] Under-Brow Intensity
    params['under_brow'] = {
        'color': MAKEUP_PALETTE['MakeupColorUnderbrow'][ui_values[45] % 12],
        'intensity': ui_values[46] / 10.0,
    }
    # [47] Eyelash Color
    params['eyelash_color'] = MAKEUP_PALETTE['MakeupColorEyeliner'][ui_values[47] % 12]
    # [48] Eyeliner Color, [49] Eyeliner Intensity
    params['eyeliner'] = {
        'color': MAKEUP_PALETTE['MakeupColorEyeliner'][ui_values[48] % 12],
        'intensity': ui_values[49] / 10.0,
    }
    # [50] Blush Color, [51] Blush Intensity
    params['blush'] = {
        'color': MAKEUP_PALETTE['MakeupColorCheeks'][ui_values[50] % 12],
        'intensity': ui_values[51] / 10.0,
    }
    # [52] Lip Color, [53] Lip Color Intensity
    params['lip_color'] = {
        'color': MAKEUP_PALETTE['MakeupColorLips'][ui_values[52] % 12],
        'intensity': ui_values[53] / 10.0,
    }
    # [54] Lip Liner Color (intensity shared with lip color)
    params['lip_liner'] = {
        'color': MAKEUP_PALETTE['MakeupColorLipLiner'][ui_values[54] % 12],
        'intensity': ui_values[53] / 10.0,
    }
    # [55] Lip Shine
    params['lip_shine'] = ui_values[55] / 10.0

    return params


def main():
    parser = argparse.ArgumentParser(description="DA:I standalone face capture")
    parser.add_argument("--count", type=int, default=100, help="Number of faces to generate")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--scale", type=float, default=0.5, help="Morph intensity (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--metrics", action="store_true", help="Extract facial metrics per image")
    parser.add_argument("--angles", type=int, default=1, help="Number of camera angles per face")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Extract and load morph data
    extract_morph_resource()

    print("Loading morph target resource...")
    with open(MORPH_RES_CACHE, 'rb') as f:
        morph_data = f.read()
    morph = MorphTargetResource(morph_data)
    print(f"  Two-way sliders: {len(morph.twoway_sliders)}")
    print(f"  One-way sliders: {len(morph.oneway_sliders)}")
    print(f"  Morph shapes: {len(morph.morph_shapes)}")

    print("Loading base mesh...")
    base_pos = np.load(os.path.join(ASSET_DIR, "base_head_positions.npy"))
    base_idx = np.load(os.path.join(ASSET_DIR, "base_head_indices.npy"))
    base_normals_path = os.path.join(ASSET_DIR, "base_head_normals.npy")
    base_normals = np.load(base_normals_path) if os.path.exists(base_normals_path) else None
    print(f"  Vertices: {len(base_pos)}, Faces: {len(base_idx)}")

    # Load input mapping
    print("Loading input mapping...")
    try:
        mapping = load_mapping()
    except FileNotFoundError:
        mapping = build_input_mapping()
    print(f"  {len(mapping)} UI inputs mapped")

    # Load available hair styles
    hair_styles = get_available_hair_styles()
    print(f"  Hair styles available: {len(hair_styles)}")

    # Setup metrics extraction
    metrics_data = {}
    if args.metrics:
        try:
            from renderer.face_metrics import extract_all, NUM_RATIOS, NUM_ENCODING
            print("  Face metrics extraction enabled")
        except ImportError as e:
            print(f"  Warning: face_metrics not available: {e}")
            args.metrics = False

    existing = set(f.replace('.png', '') for f in os.listdir(args.output) if f.endswith('.png'))
    print(f"Existing images: {len(existing)}")

    # Camera angles for multi-view capture
    angles = [(0, 0)]  # frontal
    if args.angles > 1:
        angles += [(15, 0), (-15, 0)]  # slight left/right
    if args.angles > 3:
        angles += [(0, 5), (0, -5)]  # slight up/down

    print(f"\nGenerating {args.count} faces at {args.width}x{args.height}, "
          f"scale={args.scale}, angles={len(angles)}...")
    start_time = time.time()
    generated = 0

    for i in range(args.count):
        ui_values = generate_random_ui_values(rng)
        file_id = values_to_filename(ui_values)

        if file_id in existing:
            continue

        # Convert to morph weights using the proper input mapping
        twoway, oneway, shapes = ui_values_to_morph_weights(
            mapping, ui_values, scale=args.scale)

        # Apply morphs
        morphed = morph.apply_morphs(base_pos,
                                     twoway_weights=twoway,
                                     oneway_weights=oneway,
                                     shape_weights=shapes)

        # Extract visual params from UI values
        skin_tone_idx = ui_values[1] % len(SKIN_TONE_COLORS)
        eye_color_idx = ui_values[3]  # Eye Color (0-50)
        hair_idx = ui_values[4]       # Hair style (0-27)
        hair_color_idx = ui_values[5]  # Hair Color (0-24)

        # Map hair index to available hair meshes (0 = bald option)
        hair_name = None
        if hair_styles and hair_idx > 0:
            hair_name = hair_styles[(hair_idx - 1) % len(hair_styles)]
        hair_color = get_hair_color(hair_color_idx)

        # Extract makeup parameters
        makeup = extract_makeup_params(ui_values, mapping)

        # Complexion index
        complexion_idx = ui_values[2]

        # Inner/outer iris colors
        outer_iris_idx = ui_values[20] if len(ui_values) > 20 else 0
        inner_iris_idx = ui_values[21] if len(ui_values) > 21 else 0

        for angle_idx, (yaw, pitch) in enumerate(angles):
            image = render_character(morphed, base_idx, base_normals,
                                     width=args.width, height=args.height,
                                     yaw=yaw, pitch=pitch,
                                     skin_tone=skin_tone_idx,
                                     eye_color=eye_color_idx,
                                     hair_style=hair_name,
                                     hair_color=hair_color,
                                     makeup=makeup,
                                     complexion=complexion_idx,
                                     outer_iris_color=outer_iris_idx,
                                     inner_iris_color=inner_iris_idx,
                                     eyelash_color=makeup.get('eyelash_color'))

            if angle_idx == 0:
                filename = file_id + ".png"
            else:
                filename = f"{file_id}_{angle_idx}.png"

            Image.fromarray(image).save(os.path.join(args.output, filename))

            # Extract metrics from frontal view only
            if args.metrics and angle_idx == 0:
                result = extract_all(image)
                if result is not None:
                    _, ratios, encoding = result
                    metrics_data[file_id] = {
                        'ratios': ratios.tolist(),
                        'encoding': encoding.tolist(),
                    }

        generated += 1

        if generated % 10 == 0:
            elapsed = time.time() - start_time
            rate = generated / elapsed
            print(f"  {generated}/{args.count} ({rate:.1f} faces/sec)")

    # Save metrics
    if args.metrics and metrics_data:
        metrics_path = os.path.join(args.output, "metrics.npz")
        ratios_arr = np.array([metrics_data[k]['ratios'] for k in sorted(metrics_data)])
        encodings_arr = np.array([metrics_data[k]['encoding'] for k in sorted(metrics_data)])
        keys_arr = np.array(sorted(metrics_data.keys()))
        np.savez(metrics_path, ratios=ratios_arr, encodings=encodings_arr, keys=keys_arr)
        print(f"Saved metrics for {len(metrics_data)} images to {metrics_path}")

    elapsed = time.time() - start_time
    rate = generated / elapsed if elapsed > 0 else 0
    print(f"\nDone! {generated} new faces in {elapsed:.1f}s ({rate:.1f} faces/sec)")


if __name__ == "__main__":
    main()
