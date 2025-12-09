"""
Dragon Age: Inquisition - Human Female Schema

This schema defines the character creator inputs for human female characters
in Dragon Age: Inquisition. Unlike Mass Effect, DAI uses direct UI manipulation
(clicking on coordinate grids and color wheels) instead of code paste.

Input Types:
- Sliders: 15 inputs with varying option counts (7-51)
- Coordinate Grids: 26 inputs (2D X,Y pickers)
- Color Pickers: 10 inputs (color wheel selections)
"""

from typing import List
import os

from base_schema import (
    BaseSchema, Slider, CoordinateGrid, ColorPicker,
    register_schema, InputType, DragonAgeInquisition
)

if os.getenv("APP_MODE") != "docker":
    try:
        import pyautogui
    except ImportError:
        pyautogui = None
    try:
        from PIL import ImageGrab
    except ImportError:
        ImageGrab = None


# Default configurations for DAI inputs
# These will be calibrated per-machine using the coordinate configurator tool
DEFAULT_GRID_BOUNDS = (0, 0, 100, 100)  # Placeholder - needs calibration
DEFAULT_WHEEL_CENTER = (50, 50)         # Placeholder - needs calibration
DEFAULT_WHEEL_RADIUS = 40               # Placeholder - needs calibration

# Grid resolution: 10x10 = 100 options per coordinate input
GRID_DIVISIONS = 10

# Color palette: 12 hues × 3 saturations = 36 colors
COLOR_PALETTE_SIZE = 36
HUE_DIVISIONS = 12
SATURATION_LEVELS = 3


@register_schema("dai_human_female")
class DAI_Human_Female_Schema(BaseSchema):
    """Dragon Age: Inquisition Human Female character creator schema."""

    # --- Game Configuration ---
    GAME = DragonAgeInquisition

    # --- Schema Configuration ---
    OUTPUT_DIR = "dataset/dai/human_female"
    CODE_FILE = "dataset/dai/human_female_codes.txt"

    # Window and capture configuration (needs calibration per-machine)
    WINDOW_LOCATION = (0, 0)  # Placeholder
    CAPTURE_BOX = (0, 0, 800, 600)  # Placeholder

    def get_inputs(self) -> List[InputType]:
        """
        Define all character creator inputs for DAI Human Female.

        Organized by category as they appear in the game UI.
        Position numbers are sequential for code encoding.
        """
        pos = 0  # Track position for each input

        inputs = []

        # =================================================================
        # GENERAL
        # =================================================================
        pos += 1
        inputs.append(Slider(pos, "Face Shape", 7, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Skin Tone", 11, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Complexion", 14))
        pos += 1
        inputs.append(Slider(pos, "Eye Color", 51, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Hair", 28, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Hair Color", 25, biased=True))

        # =================================================================
        # HEAD AND EARS
        # =================================================================
        pos += 1
        inputs.append(CoordinateGrid(pos, "Ear Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Ear Position", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Earlobe Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Cheekbones", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Cheek Shape", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Brow Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Forehead", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))

        # =================================================================
        # EYES
        # =================================================================
        pos += 1
        inputs.append(Slider(pos, "Eye Shape", 8, biased=True))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Eye Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Eye Position", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Eye Rotation and Spacing", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(Slider(pos, "Eyebrow Style", 23, biased=True))
        pos += 1
        inputs.append(Slider(pos, "Eyebrow Position", 10))
        pos += 1
        inputs.append(Slider(pos, "Eyelash Style", 14))
        pos += 1
        inputs.append(ColorPicker(pos, "Outer Iris Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS, biased=True))
        pos += 1
        inputs.append(ColorPicker(pos, "Inner Iris Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))

        # =================================================================
        # NOSE
        # =================================================================
        pos += 1
        inputs.append(Slider(pos, "Nose Shape", 8, biased=True))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Nose Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Nose Position", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Nose Bridge Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Bridge Position", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Nose Tip Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Nose Tip Position", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Nostril Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Broken Nose", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))

        # =================================================================
        # MOUTH AND JAW
        # =================================================================
        pos += 1
        inputs.append(Slider(pos, "Mouth Shape", 9, biased=True))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Mouth Width", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Mouth Position", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Lip Thickness", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Jawline", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Jaw Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Jowls", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Chin Size", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Chin Shape", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Double Chin", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))
        pos += 1
        inputs.append(CoordinateGrid(pos, "Adam's Apple", GRID_DIVISIONS, GRID_DIVISIONS, DEFAULT_GRID_BOUNDS))

        # =================================================================
        # MAKEUP
        # =================================================================
        pos += 1
        inputs.append(ColorPicker(pos, "Eye Shadow Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(Slider(pos, "Eye Shadow Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Under-Eye Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(ColorPicker(pos, "Under-Brow Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(Slider(pos, "Under-Brow Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Eyelash Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(ColorPicker(pos, "Eyeliner Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(Slider(pos, "Eyeliner Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Blush Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(Slider(pos, "Blush Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Lip Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(Slider(pos, "Lip Color Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Lip Liner Color", COLOR_PALETTE_SIZE, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, HUE_DIVISIONS, SATURATION_LEVELS))
        pos += 1
        inputs.append(Slider(pos, "Lip Shine", 11))

        return inputs

    def get_code_file(self) -> str:
        return self.CODE_FILE

    def get_output_dir(self) -> str:
        return self.OUTPUT_DIR

    def setup_window(self):
        """Focus and position the DAI game window."""
        try:
            from window_helpers import setup_window
            return setup_window(self.GAME.WINDOW_KEYWORD, self.WINDOW_LOCATION[0], self.WINDOW_LOCATION[1])
        except ImportError:
            return None

    def capture_image(self, code: str):
        """Capture a screenshot of the character."""
        try:
            if ImageGrab is None:
                raise RuntimeError("ImageGrab not available in this environment")
            self.send_code_to_game(code)
            shot = ImageGrab.grab(bbox=self.CAPTURE_BOX)
            return shot
        except Exception:
            return None

    # =========================================================================
    # DAI-Specific Methods
    # =========================================================================

    def apply_input_value(self, inp: InputType, value: int, click_delay: float = 0.15):
        """
        Apply a value to an input in the DAI UI.

        For DAI, ALL inputs require clicking (sliders, grids, and color pickers).
        This uses calibration data if available.

        Args:
            inp: The input type to apply
            value: The value index (0-indexed) to apply
            click_delay: Delay in seconds after clicking
        """
        if pyautogui is None:
            return

        # Use calibrated coordinates if available
        self.apply_calibrated_input(inp, value, click_delay)

    def apply_all_inputs_with_navigation(self, code: str, click_delay: float = 0.15):
        """
        Apply all inputs with menu navigation.

        Groups inputs by menu and navigates efficiently (once per category).
        This is more efficient than apply_all_inputs() which doesn't batch.

        Args:
            code: The character code to apply
            click_delay: Delay after each click
        """
        import time
        decoded = self.decode_code(code)
        cal = self.load_calibration()

        if not cal:
            # Fall back to basic application without navigation
            self.apply_all_inputs(code, click_delay)
            return

        # Group inputs by menu
        by_menu: dict[str, list[tuple[InputType, int]]] = {}
        for inp, value in decoded:
            menu = cal.inputs.get(inp.name)
            if menu:
                menu_key = menu.menu
            else:
                menu_key = "UNKNOWN"

            if menu_key not in by_menu:
                by_menu[menu_key] = []
            by_menu[menu_key].append((inp, value))

        # Process each menu group
        current_menu = None
        for menu_key, inputs in by_menu.items():
            # Navigate to menu if different from current
            if menu_key != current_menu and menu_key != "UNKNOWN":
                self.navigate_to_menu(menu_key)
                current_menu = menu_key
                time.sleep(0.3)  # Wait for menu transition

            # Apply all inputs in this menu
            for inp, value in inputs:
                self.apply_input_value(inp, value, click_delay)

    def send_code_to_game(self, code: str):
        """
        For DAI, we don't paste codes - we apply inputs directly.

        Uses navigation-aware application if calibration is available.
        """
        self.setup_window()

        cal = self.load_calibration()
        if cal and cal.navigation:
            self.apply_all_inputs_with_navigation(code)
        else:
            self.apply_all_inputs(code)
