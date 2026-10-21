#!/usr/bin/env python3
"""
DA:I standalone character data collection tool.

Generates random morph slider values, renders faces using the standalone
renderer (no game needed), and saves screenshots for training data.

Usage:
    python tools/dai_standalone_capture.py --count 100
    python tools/dai_standalone_capture.py --count 10000 --width 256 --height 256
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'

from renderer.morph_engine import MorphTargetResource
from renderer.renderer import load_mesh_from_arrays, render_face
from PIL import Image


# Paths
GAME_PATH = "/mnt/10TBRaid/games/drop/dragon-age-inquisition"
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dai")
MORPH_RES_CACHE = os.path.join(ASSET_DIR, "morph_targets.res")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "dai", "female")

NUM_TWOWAY = 68
NUM_ONEWAY = 49


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


def generate_random_weights(rng: np.random.Generator, scale: float = 0.5):
    """Generate random slider weights for a face.

    Args:
        rng: numpy random generator
        scale: how extreme the morphs are (0.0 = base, 1.0 = full range)

    Returns:
        (twoway_weights, oneway_weights) tuple
    """
    twoway = (rng.random(NUM_TWOWAY) * 2 - 1) * scale
    oneway = rng.random(NUM_ONEWAY) * scale
    return twoway, oneway


def weights_to_filename(twoway: np.ndarray, oneway: np.ndarray) -> str:
    """Convert slider weights to a compact filename.

    Quantizes to 0-20 range per slider, encodes as sequential index + hash.
    Uses index-based naming: {sequential_id}_{hash}.
    The weights are saved in a separate manifest for exact recovery.
    """
    # Quantize to 21 levels (0-20) for reasonable granularity
    tw_q = np.clip(np.round((twoway + 1) * 10).astype(int), 0, 20)
    ow_q = np.clip(np.round(oneway * 20).astype(int), 0, 20)
    all_q = np.concatenate([tw_q, ow_q])
    # Create a short hash from the quantized values
    h = hash(tuple(all_q.tolist())) & 0xFFFFFFFF
    return f"{h:08x}"


def filename_to_weights(filename: str, manifest_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a filename back into slider weights using the manifest."""
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    entry = manifest.get(filename.replace('.png', ''))
    if entry is None:
        raise ValueError(f"Filename {filename} not found in manifest")
    return np.array(entry['twoway']), np.array(entry['oneway'])


def main():
    parser = argparse.ArgumentParser(description="DA:I standalone face capture")
    parser.add_argument("--count", type=int, default=100, help="Number of faces to generate")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--scale", type=float, default=0.5, help="Morph intensity (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
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

    print("Loading base mesh...")
    base_pos = np.load(os.path.join(ASSET_DIR, "base_head_positions.npy"))
    base_idx = np.load(os.path.join(ASSET_DIR, "base_head_indices.npy"))
    print(f"  Vertices: {len(base_pos)}, Faces: {len(base_idx)}")

    # Load or create manifest
    import json
    manifest_path = os.path.join(args.output, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    existing = set(os.listdir(args.output))
    print(f"Existing images: {len(existing)}")

    print(f"\nGenerating {args.count} faces at {args.width}x{args.height}, scale={args.scale}...")
    start_time = time.time()
    generated = 0

    for i in range(args.count):
        twoway, oneway = generate_random_weights(rng, args.scale)
        file_id = weights_to_filename(twoway, oneway)
        filename = file_id + ".png"

        if filename in existing:
            continue

        # Apply morphs
        morphed = morph.apply_morphs(base_pos,
                                     twoway_weights=twoway.tolist(),
                                     oneway_weights=oneway.tolist())

        # Render
        mesh = load_mesh_from_arrays(morphed, base_idx)
        image = render_face(mesh, args.width, args.height)

        # Save image and update manifest
        Image.fromarray(image).save(os.path.join(args.output, filename))
        manifest[file_id] = {
            "twoway": [round(float(w), 4) for w in twoway],
            "oneway": [round(float(w), 4) for w in oneway],
        }
        generated += 1

        if generated % 10 == 0:
            elapsed = time.time() - start_time
            rate = generated / elapsed
            print(f"  {generated}/{args.count} ({rate:.1f} faces/sec)")

    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=1)

    elapsed = time.time() - start_time
    rate = generated / elapsed if elapsed > 0 else 0
    print(f"\nDone! {generated} new faces in {elapsed:.1f}s ({rate:.1f} faces/sec)")


if __name__ == "__main__":
    main()
