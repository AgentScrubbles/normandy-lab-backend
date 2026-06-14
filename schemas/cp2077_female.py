"""
Cyberpunk 2077 - Female V Schema (ACU preset file mode)

Uses ACU (Appearance Change Unlocker) mod to save/load character presets.
Preset files use LocKey#<id>:<value> format.

Filenames encode 14 values separated by underscores:
  SkinTone_SkinType_Hairstyle_HairColor_Eyes_EyeColor_Eyebrows_
  EyebrowColor_EyelashColor_Nose_Mouth_Jaw_Ears_PiercingColor
"""

import os
import subprocess
import time
from typing import List, Optional

from base_schema import BaseSchema, Slider, register_schema, InputType, Cyberpunk2077


# Reuse constants from the capture tool
GAME_DIR = "/mnt/10TBRaid/SteamLibrary/steamapps/common/Cyberpunk 2077"
PRESET_DIR = os.path.join(GAME_DIR, "bin/x64/plugins/cyber_engine_tweaks/mods/AppearanceChangeUnlocker/character-presets/female")
AUTO_PRESET = os.path.join(PRESET_DIR, "_auto.preset")
DEFAULT_PRESET = os.path.join(PRESET_DIR, "default.preset")
WINDOW_TITLE = "Cyberpunk 2077 (C) 2020 by CD Projekt RED"
FACE_CROP = (520, 160, 1050, 860)  # Relative to game window

# Maps input index (in our 14-input schema) to preset file line index
INPUT_TO_LINE = {
    0: 0,    # Skin Tone
    1: 1,    # Skin Type
    2: 2,    # Hairstyle
    3: 3,    # Hair Color
    4: 4,    # Eyes
    5: 5,    # Eye Color
    6: 6,    # Eyebrows
    7: 7,    # Eyebrow Color
    8: 8,    # Eyelash Color
    9: 9,    # Nose
    10: 10,  # Mouth
    11: 11,  # Jaw
    12: 12,  # Ears
    13: 17,  # Piercing Color (line 17 in preset)
}

# Lines to force to 0 (cyberware, cosmetics, body)
FORCE_ZERO = set(range(13, 28)) | set(range(28, 38))


@register_schema("cp2077_female")
class CP2077_Female_Schema(BaseSchema):
    """Cyberpunk 2077 Female V character creator schema."""

    GAME = Cyberpunk2077

    OUTPUT_DIR = "dataset/cp2077/female"
    CODE_FILE = "dataset/cp2077/female_codes.txt"

    FILENAME_SEPARATOR = "_"

    def get_inputs(self) -> List[InputType]:
        pos = 0
        inputs = []

        pos += 1; inputs.append(Slider(pos, "Skin Tone", 12, biased=True))
        pos += 1; inputs.append(Slider(pos, "Skin Type", 4))
        pos += 1; inputs.append(Slider(pos, "Hairstyle", 51, biased=True))
        pos += 1; inputs.append(Slider(pos, "Hair Color", 30, biased=True))
        pos += 1; inputs.append(Slider(pos, "Eyes", 6, biased=True))
        pos += 1; inputs.append(Slider(pos, "Eye Color", 6, biased=True))
        pos += 1; inputs.append(Slider(pos, "Eyebrows", 12, biased=True))
        pos += 1; inputs.append(Slider(pos, "Eyebrow Color", 10))
        pos += 1; inputs.append(Slider(pos, "Eyelash Color", 10))
        pos += 1; inputs.append(Slider(pos, "Nose", 22, biased=True))
        pos += 1; inputs.append(Slider(pos, "Mouth", 22, biased=True))
        pos += 1; inputs.append(Slider(pos, "Jaw", 22, biased=True))
        pos += 1; inputs.append(Slider(pos, "Ears", 22))
        pos += 1; inputs.append(Slider(pos, "Piercing Color", 10))

        return inputs

    def parse_filename(self, stem: str) -> Optional[List[int]]:
        """Parse underscore-separated integer values from filename."""
        parts = stem.split('_')
        inputs = self.get_inputs()
        if len(parts) != len(inputs):
            return None
        labels = []
        for part, inp in zip(parts, inputs):
            val = int(part)
            val = min(val, inp.num_classes() - 1)
            labels.append(val)
        return labels

    def get_code_file(self) -> str:
        return self.CODE_FILE

    def get_output_dir(self) -> str:
        return self.OUTPUT_DIR

    def _load_template(self):
        """Load the default preset as template."""
        with open(DEFAULT_PRESET, 'r') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    key, val = line.rsplit(':', 1)
                    lines.append((key, int(val)))
        return lines

    def _find_game_window(self):
        """Find game window position and size."""
        result = subprocess.run(['wmctrl', '-lG'], capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if '(C) 2020 by CD Projekt RED' in line:
                parts = line.split(None, 7)
                return int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        return None

    def _values_to_preset(self, values: List[int]) -> List[str]:
        """Convert predicted values to preset file lines."""
        template = self._load_template()
        new_lines = []
        for i, (key, val) in enumerate(template):
            # Check if this line maps to one of our inputs
            for input_idx, line_idx in INPUT_TO_LINE.items():
                if i == line_idx:
                    val = values[input_idx]
                    break
            if i in FORCE_ZERO:
                val = 0
            new_lines.append(f"{key}:{val}")
        return new_lines

    def setup_window(self):
        subprocess.run(['wmctrl', '-a', WINDOW_TITLE])

    def send_code_to_game(self, code: str):
        """Decode code, write preset file, and trigger F8 to load it."""
        decoded = self.decode_code(code)
        values = [value for _, value in decoded]

        preset_lines = self._values_to_preset(values)
        with open(AUTO_PRESET, 'w') as f:
            f.write('\n'.join(preset_lines) + '\n')

        self.setup_window()
        time.sleep(0.3)
        subprocess.run(['xdotool', 'key', 'F8'])
        time.sleep(3)

    def capture_image(self, code: str):
        """Write preset, load in game, and capture face screenshot."""
        self.send_code_to_game(code)

        win = self._find_game_window()
        if not win:
            print("WARNING: Game window not found")
            return None

        gx, gy, _, _ = win
        x1, y1, x2, y2 = FACE_CROP

        from PIL import ImageGrab
        return ImageGrab.grab(bbox=(gx + x1, gy + y1, gx + x2, gy + y2))
