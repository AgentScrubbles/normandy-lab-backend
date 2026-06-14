"""
Fallout 4 - Female Schema (LooksMenu preset file mode)

Uses LooksMenu mod to save/load face configurations as JSON preset files.
No UI automation needed for applying settings -- just write a JSON file.

Input types (all Sliders, discretized to 10 bins unless noted):
- Hair Style (20 options, biased)
- Hair Color (10, biased)
- Eye Color (10, biased)
- Skin Tone (10, biased)
- Face Morph Regions 0-7 (10 each, one float per region)
- Weight Thin/Muscular/Large (10 each)

Form IDs for HeadParts are placeholders until we can examine a real preset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from base_schema import BaseSchema, Slider, register_schema, InputType, Fallout4


# Morph region hex keys from F4EE source (CharGenInterface.cpp)
# Each region has an 8-float array; we control one representative float per region
MORPH_REGION_KEYS = [
    "00000001",  # Region 0: Forehead
    "00000002",  # Region 1: Eyes
    "00000003",  # Region 2: Nose
    "00000004",  # Region 3: Mouth
    "00000005",  # Region 4: Cheeks
    "00000006",  # Region 5: Jaw
    "00000007",  # Region 6: Chin
    "00000008",  # Region 7: Neck
]

# Placeholder form IDs for female hair styles
# These will be replaced with real form IDs from examining actual presets
HAIR_FORM_IDS = [f"00{i:06X}" for i in range(20)]

# Placeholder form IDs for eye colors
EYE_FORM_IDS = [f"01{i:06X}" for i in range(10)]

# Number of discretization bins for continuous values
NUM_BINS = 10


def float_from_bin(bin_idx: int, num_bins: int = NUM_BINS) -> float:
    """Convert a bin index (0 to num_bins-1) to a float in [0.0, 1.0]."""
    if num_bins <= 1:
        return 0.0
    return bin_idx / (num_bins - 1)


def bin_from_float(value: float, num_bins: int = NUM_BINS) -> int:
    """Convert a float in [0.0, 1.0] to the nearest bin index."""
    clamped = max(0.0, min(1.0, value))
    return round(clamped * (num_bins - 1))


@register_schema("fo4_female")
class FO4_Female_Schema(BaseSchema):
    """Fallout 4 Female character creator schema (LooksMenu presets)."""

    GAME = Fallout4

    OUTPUT_DIR = "dataset/fo4/female"
    CODE_FILE = "dataset/fo4/female_codes.txt"

    # Default presets directory (overridable via FO4_PRESETS_DIR env var)
    DEFAULT_PRESETS_DIR = os.path.expanduser(
        "~/.steam/steam/steamapps/compatdata/377160/pfx/drive_c/"
        "users/steamuser/My Documents/My Games/Fallout4/F4SE/Plugins/F4EE/Presets"
    )

    def __init__(self):
        self.presets_dir = os.environ.get("FO4_PRESETS_DIR", self.DEFAULT_PRESETS_DIR)

    def get_inputs(self) -> List[InputType]:
        pos = 0
        inputs = []

        # === General ===
        pos += 1
        inputs.append(Slider(pos, "Hair Style", 20, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Hair Color", NUM_BINS, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Eye Color", NUM_BINS, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Skin Tone", NUM_BINS, biased=True))

        # === Face Morphs (regions 0-7) ===
        for i in range(8):
            pos += 1
            inputs.append(Slider(pos, f"Morph Region {i}", NUM_BINS))

        # === Body Weights ===
        pos += 1
        inputs.append(Slider(pos, "Weight Thin", NUM_BINS))
        pos += 1
        inputs.append(Slider(pos, "Weight Muscular", NUM_BINS))
        pos += 1
        inputs.append(Slider(pos, "Weight Large", NUM_BINS))

        return inputs

    def get_code_file(self) -> str:
        return self.CODE_FILE

    def get_output_dir(self) -> str:
        return self.OUTPUT_DIR

    def setup_window(self):
        """Focus the Fallout 4 window."""
        try:
            from window_helpers import focus_window, find_window
            game_window = find_window(self.GAME.WINDOW_KEYWORD)
            if not game_window:
                raise RuntimeError(f"Window '{self.GAME.WINDOW_KEYWORD}' not found. Is FO4 running?")
            focus_window(game_window)
            return game_window
        except ImportError:
            return None

    def capture_image(self, code: str):
        """Capture a screenshot of the character."""
        try:
            from PIL import ImageGrab
            bbox = (0, 0, 800, 600)  # Placeholder - needs calibration
            return ImageGrab.grab(bbox=bbox)
        except ImportError:
            return None

    def send_code_to_game(self, code: str):
        """Write a LooksMenu preset JSON file from the given code."""
        preset = self._code_to_preset(code)
        Path(self.presets_dir).mkdir(parents=True, exist_ok=True)
        preset_path = os.path.join(self.presets_dir, f"cc_{code}.json")
        with open(preset_path, "w") as f:
            json.dump(preset, f, indent=2)
        return preset_path

    def _code_to_preset(self, code: str) -> Dict[str, Any]:
        """Convert a character code to a LooksMenu preset JSON dict."""
        decoded = self.decode_code(code)
        values = {inp.name: val for inp, val in decoded}

        # Build morph regions array
        morph_regions = {}
        for i, key in enumerate(MORPH_REGION_KEYS):
            bin_val = values[f"Morph Region {i}"]
            # Each region is an 8-float array; we set index 0, rest at 0.0
            floats = [0.0] * 8
            floats[0] = float_from_bin(bin_val)
            morph_regions[key] = floats

        # Build head parts
        hair_idx = values["Hair Style"]
        hair_form_id = HAIR_FORM_IDS[hair_idx] if hair_idx < len(HAIR_FORM_IDS) else HAIR_FORM_IDS[0]

        eye_idx = values["Eye Color"]
        eye_form_id = EYE_FORM_IDS[eye_idx] if eye_idx < len(EYE_FORM_IDS) else EYE_FORM_IDS[0]

        preset = {
            "version": 1,
            "Actor": "00000014",  # Player reference
            "isFemale": True,
            "weight": {
                "Thin": float_from_bin(values["Weight Thin"]),
                "Muscular": float_from_bin(values["Weight Muscular"]),
                "Large": float_from_bin(values["Weight Large"]),
            },
            "headParts": [
                {"type": 1, "formId": hair_form_id},   # Hair
                {"type": 9, "formId": eye_form_id},     # Eyes
            ],
            "morphRegionData": morph_regions,
            "hairColor": f"00{values['Hair Color']:06X}",
            "skinTone": float_from_bin(values["Skin Tone"]),
        }
        return preset

    def _preset_to_code(self, preset: Dict[str, Any]) -> str:
        """Convert a LooksMenu preset JSON dict back to a character code."""
        inputs = self.get_inputs()
        values = {}

        # Hair Style: find index in HAIR_FORM_IDS
        hair_form_id = None
        for hp in preset.get("headParts", []):
            if hp.get("type") == 1:
                hair_form_id = hp.get("formId")
                break
        if hair_form_id and hair_form_id in HAIR_FORM_IDS:
            values["Hair Style"] = HAIR_FORM_IDS.index(hair_form_id)
        else:
            values["Hair Style"] = 0

        # Hair Color
        hair_color_hex = preset.get("hairColor", "00000000")
        try:
            values["Hair Color"] = int(hair_color_hex, 16) & 0xFF
        except (ValueError, TypeError):
            values["Hair Color"] = 0
        values["Hair Color"] = min(values["Hair Color"], NUM_BINS - 1)

        # Eye Color: find index in EYE_FORM_IDS
        eye_form_id = None
        for hp in preset.get("headParts", []):
            if hp.get("type") == 9:
                eye_form_id = hp.get("formId")
                break
        if eye_form_id and eye_form_id in EYE_FORM_IDS:
            values["Eye Color"] = EYE_FORM_IDS.index(eye_form_id)
        else:
            values["Eye Color"] = 0

        # Skin Tone
        values["Skin Tone"] = bin_from_float(preset.get("skinTone", 0.0))

        # Morph regions
        morph_data = preset.get("morphRegionData", {})
        for i, key in enumerate(MORPH_REGION_KEYS):
            region_floats = morph_data.get(key, [0.0] * 8)
            values[f"Morph Region {i}"] = bin_from_float(region_floats[0] if region_floats else 0.0)

        # Body weights
        weight = preset.get("weight", {})
        values["Weight Thin"] = bin_from_float(weight.get("Thin", 0.0))
        values["Weight Muscular"] = bin_from_float(weight.get("Muscular", 0.0))
        values["Weight Large"] = bin_from_float(weight.get("Large", 0.0))

        # Encode
        code_chars = []
        for inp in inputs:
            val = values.get(inp.name, 0)
            val = max(0, min(val, inp.num_classes() - 1))
            code_chars.append(inp.encode(val))
        return ''.join(code_chars)
