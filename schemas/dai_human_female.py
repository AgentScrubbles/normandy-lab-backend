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

# Color palette: Just the hue ring (no saturation variation)
# 12 hues around the color wheel (30-degree increments)
NUM_HUES = 12


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
        inputs.append(ColorPicker(pos, "Outer Iris Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS, biased=True))
        pos += 1
        inputs.append(ColorPicker(pos, "Inner Iris Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))

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
        inputs.append(ColorPicker(pos, "Eye Shadow Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(Slider(pos, "Eye Shadow Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Under-Eye Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(ColorPicker(pos, "Under-Brow Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(Slider(pos, "Under-Brow Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Eyelash Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(ColorPicker(pos, "Eyeliner Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(Slider(pos, "Eyeliner Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Blush Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(Slider(pos, "Blush Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Lip Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(Slider(pos, "Lip Color Intensity", 11))
        pos += 1
        inputs.append(ColorPicker(pos, "Lip Liner Color", NUM_HUES, DEFAULT_WHEEL_CENTER, DEFAULT_WHEEL_RADIUS))
        pos += 1
        inputs.append(Slider(pos, "Lip Shine", 11))

        return inputs

    def get_code_file(self) -> str:
        return self.CODE_FILE

    def get_output_dir(self) -> str:
        return self.OUTPUT_DIR

    def setup_window(self):
        """Focus the DAI game window (don't move - use calibrated position)."""
        try:
            from window_helpers import focus_window, find_window
            import pyautogui
            import time

            game_window = find_window(self.GAME.WINDOW_KEYWORD)
            if not game_window:
                raise RuntimeError(f"Window titled {self.GAME.WINDOW_KEYWORD} was not found. Is it running?")

            # Use wmctrl to bring window to front
            focus_window(game_window)
            time.sleep(0.1)

            # Click on the game window to ensure it has keyboard focus
            # Use calibrated window position if available
            cal = self.load_calibration()
            if cal and cal.window:
                # Offset further into the window to avoid resize borders
                click_x = cal.window.reference_position[0] + 150
                click_y = cal.window.reference_position[1] + 150
            else:
                # Fallback: click near top-left of screen
                click_x, click_y = 150, 150

            print(f"    Clicking at ({click_x}, {click_y}) to focus game", flush=True)
            pyautogui.click(click_x, click_y)
            time.sleep(0.2)

            # Click again to make sure - some games need double activation
            pyautogui.click(click_x, click_y)
            time.sleep(0.3)

            # Send a harmless key to "wake up" the game's input handling
            # Use a key that won't do anything in the character creator
            pyautogui.press('shift')
            time.sleep(0.1)

            return game_window
        except ImportError:
            return None

    def capture_image(self, code: str):
        """Capture a screenshot of the character."""
        try:
            if ImageGrab is None:
                raise RuntimeError("ImageGrab not available in this environment")

            # Apply all inputs
            self.send_code_to_game(code)

            # Reset view (zoom in) before capture
            self.reset_view_for_capture()

            # Use calibrated capture box if available
            cal = self.load_calibration()
            if cal and cal.capture:
                bbox = cal.capture.box
            else:
                bbox = self.CAPTURE_BOX

            shot = ImageGrab.grab(bbox=bbox)
            return shot
        except Exception:
            return None

    # =========================================================================
    # DAI-Specific Methods (Keyboard-based navigation)
    # =========================================================================

    def send_code_to_game(self, code: str):
        """
        For DAI, we don't paste codes - we apply inputs using keyboard navigation.

        Uses calibration data to navigate menus and inputs via keyboard,
        with mouse clicks only for coordinate grids and color pickers.
        """
        self.setup_window()

        cal = self.load_calibration()
        if cal:
            # Use keyboard-based navigation
            self.apply_all_calibrated_inputs(code)
        else:
            print("Warning: No calibration data. Run calibrator first.")
