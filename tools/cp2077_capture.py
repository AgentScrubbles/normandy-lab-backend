#!/usr/bin/env python3
"""
Cyberpunk 2077 character data collection tool.

Generates random presets, loads them via ACU + AutoPresetLoader,
and captures face screenshots for training data.

Usage:
    python tools/cp2077_capture.py --count 10
    python tools/cp2077_capture.py --count 1000 --delay 4
"""

import argparse
import os
import random
import subprocess
import time
from pathlib import Path

from PIL import ImageGrab

# Paths
GAME_DIR = "/mnt/10TBRaid/SteamLibrary/steamapps/common/Cyberpunk 2077"
PRESET_DIR = os.path.join(GAME_DIR, "bin/x64/plugins/cyber_engine_tweaks/mods/AppearanceChangeUnlocker/character-presets/female")
AUTO_PRESET = os.path.join(PRESET_DIR, "_auto.preset")
DEFAULT_PRESET = os.path.join(PRESET_DIR, "default.preset")
OUTPUT_DIR = "/mnt/10TBRaid/src/character_creator/dataset/cp2077/female"

# Game window
WINDOW_TITLE = "Cyberpunk 2077 (C) 2020 by CD Projekt RED"
FACE_CROP = (520, 160, 1050, 860)  # Relative to game window

# Stable options we randomize (line index -> (name, max_options))
RANDOMIZE_STABLE = {
    0: ("Skin Tone", 12),
    1: ("Skin Type", 4),
    2: ("Hairstyle", 51),
    4: ("Eyes", 6),       # First 6 are normal eye shapes
    5: ("Eye Color", 6),  # First 6 are normal eye colors
    6: ("Eyebrows", 12),
    9: ("Nose", 22),
    10: ("Mouth", 22),
    11: ("Jaw", 22),
    12: ("Ears", 22),
}

# Variable LocKey options — we randomize the value even though the LocKey
# may not match after a style change. When it does match, we get color variety.
# When it doesn't, the game uses its default for that style.
RANDOMIZE_VARIABLE = {
    3: ("Hair Color", 30),          # Hair color palette
    7: ("Eyebrow Color", 10),       # Eyebrow color
    8: ("Eyelash Color", 10),       # Eyelash color
    17: ("Piercing Color", 10),     # Won't apply (piercings disabled)
}

# Options to force to 0 (cyberware, scars, tattoos, piercings, teeth, lip makeup, body stuff)
FORCE_ZERO = set(range(13, 28)) | set(range(28, 38))  # 13-27: cyberware through blemishes, 28+: body


def find_game_window():
    """Find game window position and size."""
    result = subprocess.run(['wmctrl', '-lG'], capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        if '(C) 2020 by CD Projekt RED' in line:
            parts = line.split(None, 7)
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            return x, y, w, h
    return None


def load_template():
    """Load the default preset as template."""
    with open(DEFAULT_PRESET, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                key, val = line.rsplit(':', 1)
                lines.append((key, int(val)))
    return lines


def generate_preset(template):
    """Generate a randomized preset, returning (lines, code_dict)."""
    new_lines = []
    code = {}

    for i, (key, val) in enumerate(template):
        if i in RANDOMIZE_STABLE:
            name, max_opts = RANDOMIZE_STABLE[i]
            val = random.randint(0, max_opts - 1)
            code[name] = val
        elif i in RANDOMIZE_VARIABLE:
            name, max_opts = RANDOMIZE_VARIABLE[i]
            val = random.randint(0, max_opts - 1)
            code[name] = val
        if i in FORCE_ZERO:
            val = 0
        new_lines.append(f"{key}:{val}")

    return new_lines, code


def code_to_string(code):
    """Convert code dict to a filename-safe string."""
    all_keys = {}
    all_keys.update(RANDOMIZE_STABLE)
    all_keys.update(RANDOMIZE_VARIABLE)
    parts = []
    for i in sorted(all_keys.keys()):
        name = all_keys[i][0]
        parts.append(str(code.get(name, 0)))
    return '_'.join(parts)


def write_preset(lines):
    """Write preset to _auto.preset."""
    with open(AUTO_PRESET, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def focus_game():
    """Focus the game window."""
    subprocess.run(['wmctrl', '-a', WINDOW_TITLE])


def press_f8():
    """Press F8 to trigger preset load."""
    subprocess.run(['xdotool', 'key', 'F8'])


def capture_face(gx, gy):
    """Capture the face region."""
    x1, y1, x2, y2 = FACE_CROP
    return ImageGrab.grab(bbox=(gx + x1, gy + y1, gx + x2, gy + y2))


def main():
    parser = argparse.ArgumentParser(description="CP2077 character data collection")
    parser.add_argument("--count", type=int, default=10, help="Number of characters to capture")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to wait after loading preset")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find game window
    win = find_game_window()
    if not win:
        print("ERROR: Cyberpunk 2077 window not found!")
        return
    gx, gy, gw, gh = win
    print(f"Game window: pos=({gx},{gy}) size={gw}x{gh}")

    # Load template
    template = load_template()
    print(f"Template loaded: {len(template)} options")

    # Focus game
    focus_game()
    time.sleep(1)

    print(f"\nCapturing {args.count} characters...")
    for i in range(args.count):
        # Generate and write preset
        lines, code = generate_preset(template)
        write_preset(lines)
        code_str = code_to_string(code)

        # Load preset in game
        press_f8()
        time.sleep(args.delay)

        # Capture
        img = capture_face(gx, gy)
        filename = f"{code_str}.png"
        filepath = os.path.join(args.output, filename)
        img.save(filepath)

        print(f"[{i+1}/{args.count}] {filename}")

    print(f"\nDone! {args.count} images saved to {args.output}")


if __name__ == "__main__":
    main()
