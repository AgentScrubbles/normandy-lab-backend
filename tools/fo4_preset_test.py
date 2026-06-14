"""
Fallout 4 LooksMenu preset test tool.

Usage:
    python -m tools.fo4_preset_test generate --count 5 --output-dir /tmp/fo4_test
    python -m tools.fo4_preset_test inspect --file preset.json
    python -m tools.fo4_preset_test roundtrip [--count 100]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_standalone_preset(seed: int = None) -> dict:
    """
    Generate a sample LooksMenu preset without requiring the schema.
    Useful for format exploration before the full schema is wired up.
    """
    if seed is not None:
        random.seed(seed)

    morph_regions = {}
    for i in range(1, 9):
        key = f"0000000{i}"
        floats = [0.0] * 8
        floats[0] = round(random.random(), 4)
        morph_regions[key] = floats

    preset = {
        "version": 1,
        "Actor": "00000014",
        "isFemale": True,
        "weight": {
            "Thin": round(random.random(), 4),
            "Muscular": round(random.random(), 4),
            "Large": round(random.random(), 4),
        },
        "headParts": [
            {"type": 1, "formId": f"00{random.randint(0, 19):06X}"},
            {"type": 9, "formId": f"01{random.randint(0, 9):06X}"},
        ],
        "morphRegionData": morph_regions,
        "hairColor": f"00{random.randint(0, 9):06X}",
        "skinTone": round(random.random(), 4),
    }
    return preset


def cmd_generate(args):
    """Generate sample preset JSON files."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to use the schema if available, fall back to standalone
    use_schema = not args.standalone
    schema = None
    if use_schema:
        try:
            from schemas.fo4_female import FO4_Female_Schema
            schema = FO4_Female_Schema()
        except ImportError:
            print("Schema not available, using standalone generator.")
            schema = None

    for i in range(args.count):
        if schema:
            code = schema.generate_code()
            preset = schema._code_to_preset(code)
            filename = f"cc_{code}.json"
        else:
            preset = build_standalone_preset(seed=i if args.deterministic else None)
            filename = f"sample_{i:04d}.json"

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(preset, f, indent=2)

        if schema:
            print(f"  {filename} (code: {code})")
        else:
            print(f"  {filename}")

    print(f"\nGenerated {args.count} presets in {output_dir}")


def cmd_inspect(args):
    """Read and pretty-print a preset JSON."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)

    with open(filepath) as f:
        preset = json.load(f)

    print(f"=== {filepath.name} ===")
    print(f"Version: {preset.get('version')}")
    print(f"Actor:   {preset.get('Actor')}")
    print(f"Female:  {preset.get('isFemale')}")

    # Weight
    weight = preset.get("weight", {})
    print(f"\nWeight:")
    for k, v in weight.items():
        print(f"  {k}: {v}")

    # Head parts
    print(f"\nHead Parts:")
    type_names = {1: "Hair", 2: "Facial Hair", 3: "Eyes (Mesh)", 9: "Eyes (Color)"}
    for hp in preset.get("headParts", []):
        t = hp.get("type", "?")
        name = type_names.get(t, f"Type {t}")
        print(f"  {name}: {hp.get('formId')}")

    # Hair/skin
    print(f"\nHair Color: {preset.get('hairColor')}")
    print(f"Skin Tone:  {preset.get('skinTone')}")

    # Morph regions
    morph_data = preset.get("morphRegionData", {})
    if morph_data:
        print(f"\nMorph Regions ({len(morph_data)}):")
        for key in sorted(morph_data.keys()):
            floats = morph_data[key]
            nonzero = [(i, v) for i, v in enumerate(floats) if v != 0.0]
            if nonzero:
                parts = ", ".join(f"[{i}]={v:.4f}" for i, v in nonzero)
                print(f"  {key}: {parts}")
            else:
                print(f"  {key}: (all zero)")

    # Tint layers (if present in real presets)
    tints = preset.get("tintData", [])
    if tints:
        print(f"\nTint Layers ({len(tints)}):")
        for t in tints[:10]:
            print(f"  index={t.get('index')} color={t.get('color')} texture={t.get('texture')}")
        if len(tints) > 10:
            print(f"  ... and {len(tints) - 10} more")

    # Try round-trip through schema
    try:
        from schemas.fo4_female import FO4_Female_Schema
        schema = FO4_Female_Schema()
        code = schema._preset_to_code(preset)
        print(f"\nEncoded code: {code}")
    except ImportError:
        pass


def cmd_roundtrip(args):
    """Test round-trip: code -> preset -> code."""
    from schemas.fo4_female import FO4_Female_Schema
    schema = FO4_Female_Schema()

    passed = 0
    failed = 0
    failures = []

    for i in range(args.count):
        code = schema.generate_code()
        preset = schema._code_to_preset(code)
        recovered = schema._preset_to_code(preset)

        if code == recovered:
            passed += 1
        else:
            failed += 1
            if len(failures) < 5:
                failures.append((code, recovered))

    print(f"Round-trip test: {passed}/{args.count} passed, {failed} failed")
    for orig, recovered in failures:
        print(f"  FAIL: {orig} -> {recovered}")
        # Show which positions differ
        for j, (a, b) in enumerate(zip(orig, recovered)):
            if a != b:
                inputs = schema.get_inputs()
                name = inputs[j].name if j < len(inputs) else f"pos {j}"
                print(f"    pos {j} ({name}): '{a}' vs '{b}'")

    if failed == 0:
        print("All round-trips passed!")
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fallout 4 LooksMenu preset test tool")
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate sample preset JSONs")
    gen.add_argument("--count", "-n", type=int, default=5)
    gen.add_argument("--output-dir", "-o", default="/tmp/fo4_test")
    gen.add_argument("--standalone", action="store_true", help="Use standalone generator (no schema)")
    gen.add_argument("--deterministic", action="store_true", help="Deterministic output (standalone only)")

    # inspect
    ins = sub.add_parser("inspect", help="Inspect a preset JSON file")
    ins.add_argument("--file", "-f", required=True)

    # roundtrip
    rt = sub.add_parser("roundtrip", help="Test code->preset->code round-trip")
    rt.add_argument("--count", "-n", type=int, default=100)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "roundtrip":
        cmd_roundtrip(args)


if __name__ == "__main__":
    main()
