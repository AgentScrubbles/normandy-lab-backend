#!/usr/bin/env python3
"""
Interactive calibration tool for game character creator schemas.

This tool guides users through recording UI element coordinates for game
automation. It uses the existing schema definitions to organize the workflow.

Usage:
    python -m tools.calibrator --schema dai_human_female
    python -m tools.calibrator --schema dai_human_female --resume
    python -m tools.calibrator --schema dai_human_female --input "Ear Size"
    python -m tools.calibrator --schema dai_human_female --list
    python -m tools.calibrator --position
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_schema import get_schema, all_schemas, BaseSchema

# Import schemas to trigger registration
# Each import is separate to allow partial success
def _import_schemas():
    """Import available schemas, handling missing dependencies gracefully."""
    schemas_imported = []
    try:
        from schemas.dai_human_female import DAI_Human_Female_Schema
        schemas_imported.append('dai_human_female')
    except ImportError:
        pass
    try:
        from schemas.me1_female import ME1_Female_Schema
        schemas_imported.append('me1_female')
    except ImportError:
        pass
    try:
        from schemas.me1_male import ME1_Male_Schema
        schemas_imported.append('me1_male')
    except ImportError:
        pass
    try:
        from schemas.me3_female import ME3_Female_Schema
        schemas_imported.append('me3_female')
    except ImportError:
        pass
    try:
        from schemas.me3_male import ME3_Male_Schema
        schemas_imported.append('me3_male')
    except ImportError:
        pass
    return schemas_imported

_import_schemas()
from input_types import Slider, CoordinateGrid, ColorPicker, InputType
from tools.calibration_schema import (
    CalibrationData, InputCalibration, WindowInfo, CaptureInfo,
    NavigationInfo, MenuNavigation,
    SliderControl, GridControl, ColorPickerControl,
)
from tools.calibration_ui import MouseCapture, CaptureAborted, continuous_position_display
from calibrations import load_calibration, save_calibration, get_calibration_path


# Category markers used in DAI schema comments
CATEGORY_PATTERN = re.compile(r'^=+\s*\n\s*#\s*(.+?)\s*\n\s*=+', re.MULTILINE)


def extract_categories_from_schema(schema: BaseSchema) -> Dict[str, List[InputType]]:
    """
    Extract input categories from schema.

    For DAI, categories are defined by comment blocks in get_inputs().
    This function groups inputs by their logical category.

    Falls back to a single "ALL" category if parsing fails.
    """
    inputs = schema.get_inputs()

    # Known DAI categories in order (from schema comments)
    dai_categories = [
        ("GENERAL", ["Face Shape", "Skin Tone", "Complexion", "Eye Color", "Hair", "Hair Color"]),
        ("HEAD_AND_EARS", ["Ear Size", "Ear Position", "Earlobe Size", "Cheekbones",
                          "Cheek Shape", "Brow Size", "Forehead"]),
        ("EYES", ["Eye Shape", "Eye Size", "Eye Position", "Eye Rotation and Spacing",
                  "Eyebrow Style", "Eyebrow Position", "Eyelash Style",
                  "Outer Iris Color", "Inner Iris Color"]),
        ("NOSE", ["Nose Shape", "Nose Size", "Nose Position", "Nose Bridge Size",
                  "Bridge Position", "Nose Tip Size", "Nose Tip Position",
                  "Nostril Size", "Broken Nose"]),
        ("MOUTH_AND_JAW", ["Mouth Shape", "Mouth Width", "Mouth Position", "Lip Thickness",
                          "Jawline", "Jaw Size", "Jowls", "Chin Size", "Chin Shape",
                          "Double Chin", "Adam's Apple"]),
        ("MAKEUP", ["Eye Shadow Color", "Eye Shadow Intensity", "Under-Eye Color",
                    "Under-Brow Color", "Under-Brow Intensity", "Eyelash Color",
                    "Eyeliner Color", "Eyeliner Intensity", "Blush Color",
                    "Blush Intensity", "Lip Color", "Lip Color Intensity",
                    "Lip Liner Color", "Lip Shine"]),
    ]

    # Build lookup from input name to input object
    input_by_name = {inp.name: inp for inp in inputs}

    # Try to match inputs to categories
    categories: Dict[str, List[InputType]] = {}
    assigned_names = set()

    for cat_name, input_names in dai_categories:
        cat_inputs = []
        for name in input_names:
            if name in input_by_name:
                cat_inputs.append(input_by_name[name])
                assigned_names.add(name)
        if cat_inputs:
            categories[cat_name] = cat_inputs

    # Add any unassigned inputs to "OTHER"
    unassigned = [inp for inp in inputs if inp.name not in assigned_names]
    if unassigned:
        categories["OTHER"] = unassigned

    # If no categories matched, fall back to single category
    if not categories:
        categories["ALL"] = inputs

    return categories


def get_input_type_name(inp: InputType) -> str:
    """Get the type string for an input."""
    if isinstance(inp, Slider):
        return "slider"
    elif isinstance(inp, CoordinateGrid):
        return "coordinate_grid"
    elif isinstance(inp, ColorPicker):
        return "color_picker"
    return "unknown"


def find_input_category(inp: InputType, categories: Dict[str, List[InputType]]) -> str:
    """Find which category an input belongs to."""
    for cat_name, cat_inputs in categories.items():
        if any(i.name == inp.name for i in cat_inputs):
            return cat_name
    return "OTHER"


class Calibrator:
    """Interactive calibration workflow manager."""

    def __init__(self, schema: BaseSchema, calibration: CalibrationData):
        self.schema = schema
        self.calibration = calibration
        self.inputs = schema.get_inputs()
        self._capture: Optional[MouseCapture] = None  # Lazy init
        self.categories = extract_categories_from_schema(schema)
        self._current_menu: Optional[str] = None

    @property
    def capture(self) -> MouseCapture:
        """Lazy-initialize mouse capture (requires pynput)."""
        if self._capture is None:
            self._capture = MouseCapture()
            # Apply window bounds if already calibrated (silently)
            self._apply_window_bounds(verbose=False)
        return self._capture

    def _apply_window_bounds(self, verbose: bool = True):
        """Apply window bounds to mouse capture if calibration exists."""
        if not self.calibration.window:
            return
        x, y = self.calibration.window.reference_position
        w, h = self.calibration.window.expected_size
        # If capture already exists, update it; otherwise it will be set on lazy init
        if self._capture is not None:
            self._capture.set_window_bounds(x, y, w, h)
        if verbose:
            print(f"  [Click filtering enabled: only clicks within ({x},{y}) to ({x+w},{y+h}) will be recorded]")

    def _get_game_slug(self) -> str:
        """Get the game slug for file paths."""
        if self.schema.GAME:
            # Convert class name to snake_case
            name = self.schema.GAME.__name__
            # Insert underscore before capitals and lowercase
            slug = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
            return slug
        return "unknown"

    def _save(self):
        """Save current calibration state."""
        game_slug = self._get_game_slug()
        save_calibration(self.calibration, game_slug, self.calibration.schema_name)
        print(f"  [Saved to {get_calibration_path(game_slug, self.calibration.schema_name)}]")

    def run_full_calibration(self):
        """Run complete calibration workflow."""
        print("=" * 70)
        print("CHARACTER CREATOR CALIBRATION TOOL")
        print(f"Schema: {self.schema.__class__.__name__}")
        print(f"Game: {self.schema.game_name}")
        print(f"Inputs: {len(self.inputs)}")
        print("=" * 70)

        try:
            # Step 1: Window setup
            self._calibrate_window()

            # Step 2: Capture region
            self._calibrate_capture()

            # Step 3: Menu navigation
            self._calibrate_menus()

            # Step 4: All inputs
            self._calibrate_all_inputs()

            print("\n" + "=" * 70)
            print("CALIBRATION COMPLETE!")
            print(f"Saved to: {get_calibration_path(self._get_game_slug(), self.calibration.schema_name)}")
            print("=" * 70)

        except CaptureAborted:
            print("\n\nCalibration aborted by user.")
            print(f"Progress saved. Use --resume to continue.")

    def _calibrate_window(self):
        """Calibrate window position by detecting from running windows."""
        print("\n" + "-" * 50)
        print("WINDOW CALIBRATION")
        print("-" * 50)

        # List all running windows
        try:
            from window_helpers import list_windows, find_window_info
            windows = list_windows()

            print("\nRunning windows:")
            print("-" * 70)
            for i, win in enumerate(windows):
                print(f"  {i+1:2}. {win.title[:50]:<50}")
                print(f"      Position: ({win.x}, {win.y})  Size: {win.width}x{win.height}")
            print("-" * 70)

            # Try to auto-detect game window
            window_keyword = self.schema.GAME.WINDOW_KEYWORD if self.schema.GAME else None
            detected_window = None

            if window_keyword:
                detected_window = find_window_info(window_keyword)
                if detected_window:
                    print(f"\n  Auto-detected game window: '{detected_window.title}'")
                    print(f"  Position: ({detected_window.x}, {detected_window.y})")
                    print(f"  Size: {detected_window.width}x{detected_window.height}")

                    use_detected = input("\n  Use detected window? [Y/n]: ").strip().lower()
                    if use_detected != 'n':
                        self.calibration.window = WindowInfo(
                            reference_position=(detected_window.x, detected_window.y),
                            expected_size=(detected_window.width, detected_window.height),
                        )
                        self.calibration.resolution = (detected_window.width, detected_window.height)
                        self._save()
                        self._apply_window_bounds()
                        print(f"\n  Window calibration saved.")
                        return

            # Manual selection if auto-detect failed or was rejected
            if not detected_window:
                print(f"\n  Could not auto-detect window (looking for: '{window_keyword}')")

            print("\n  Enter window number from list above, or press ENTER to input manually:")
            selection = input("  Selection: ").strip()

            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(windows):
                    win = windows[idx]
                    self.calibration.window = WindowInfo(
                        reference_position=(win.x, win.y),
                        expected_size=(win.width, win.height),
                    )
                    self.calibration.resolution = (win.width, win.height)
                    self._save()
                    self._apply_window_bounds()
                    print(f"\n  Window calibration saved: {win.title}")
                    return

        except ImportError:
            print("\n  Warning: window_helpers not available, using manual input")
        except Exception as e:
            print(f"\n  Warning: Could not list windows: {e}")

        # Fallback to manual input
        print("\n  Manual window configuration:")
        x_str = input("    Window X position [0]: ").strip()
        y_str = input("    Window Y position [0]: ").strip()
        width_str = input("    Window width [1920]: ").strip()
        height_str = input("    Window height [1080]: ").strip()

        x = int(x_str) if x_str else 0
        y = int(y_str) if y_str else 0
        width = int(width_str) if width_str else 1920
        height = int(height_str) if height_str else 1080

        self.calibration.window = WindowInfo(
            reference_position=(x, y),
            expected_size=(width, height),
        )
        self.calibration.resolution = (width, height)
        self._save()
        self._apply_window_bounds()

    def _calibrate_capture(self):
        """Calibrate screenshot capture region (rclone-style)."""
        print("\n" + "-" * 50)
        print("CAPTURE REGION CALIBRATION")
        print("-" * 50)

        print("\n--- Capture Box ---")
        if self.calibration.capture:
            print(f"  Current: {self.calibration.capture.box}")
        else:
            print(f"  Current: (not set)")
        print(f"  [F12 to keep, or click TOP-LEFT then BOTTOM-RIGHT to set]")

        try:
            clicks = self.capture.capture_click_sequence(
                "Click TOP-LEFT, then BOTTOM-RIGHT, or F12 to keep",
                min_clicks=0,
                max_clicks=2,
            )
            if len(clicks) >= 2:
                p1, p2 = clicks[0], clicks[1]
                x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
                y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
                bounds = (x1, y1, x2, y2)
                self.calibration.capture = CaptureInfo(box=bounds)
                self._save()
                print(f"  -> Set to: {bounds}")
            elif len(clicks) == 1:
                print(f"  Need 2 clicks for bounding box, got 1. Keeping existing.")
            else:
                print(f"  -> Keeping existing")
        except CaptureAborted:
            pass

    def _calibrate_menus(self):
        """Calibrate menu navigation paths (rclone-style: show existing, F12 to keep)."""
        print("\n" + "-" * 50)
        print("MENU NAVIGATION CALIBRATION")
        print("-" * 50)

        print("\nFor each item: F12 = keep existing/skip, Click = record new value, ESC = abort")

        # Load existing values
        menus: Dict[str, MenuNavigation] = {}
        back_button = None
        back_wait_ms = 300

        if self.calibration.navigation:
            menus = dict(self.calibration.navigation.menus)
            back_button = self.calibration.navigation.back_button
            back_wait_ms = self.calibration.navigation.back_wait_ms

        category_list = list(self.categories.keys())

        # Each category menu
        for i, cat_name in enumerate(category_list):
            print(f"\n--- {cat_name} ({i+1}/{len(category_list)}) ---")

            existing = menus.get(cat_name)
            if existing and existing.path:
                print(f"  Current: {len(existing.path)} click(s) - {existing.path}")
            else:
                print(f"  Current: (not set)")
            print(f"  [F12 to keep, or click path to {cat_name}]")

            try:
                path = self.capture.capture_click_sequence(
                    f"Click path to {cat_name}, or F12 to keep",
                    min_clicks=0,
                    max_clicks=5,
                )
                if path:
                    menus[cat_name] = MenuNavigation(path=path, wait_after_ms=existing.wait_after_ms if existing else 300)
                    print(f"  -> Set to: {path}")
                elif not existing:
                    # No existing and no new clicks - set empty
                    menus[cat_name] = MenuNavigation(path=[], wait_after_ms=300)
            except CaptureAborted:
                if not existing:
                    menus[cat_name] = MenuNavigation(path=[], wait_after_ms=300)

            # Save after each menu
            self.calibration.navigation = NavigationInfo(
                menus=menus,
                back_button=back_button,
                back_wait_ms=back_wait_ms,
            )
            self._save()

            # After first category, ask for back button (only once, only if not set)
            # User is now IN a submenu, so they can click the back button
            if i == 0 and back_button is None:
                print(f"\n--- Back Button ---")
                print(f"  Current: (not set)")
                print(f"  [F12 to skip, or click the back button]")

                try:
                    clicks = self.capture.capture_click_sequence(
                        "Click the back button, or F12 to skip",
                        min_clicks=0,
                        max_clicks=1,
                    )
                    if clicks:
                        back_button = clicks[0]
                        print(f"  -> Set to: {back_button}")
                        # Save back button immediately
                        self.calibration.navigation = NavigationInfo(
                            menus=menus,
                            back_button=back_button,
                            back_wait_ms=back_wait_ms,
                        )
                        self._save()
                except CaptureAborted:
                    pass

            # After each category (except last), let user navigate back to main menu
            if i < len(category_list) - 1:
                print(f"\n  [Navigate back to main menu, then F12 when ready]")
                try:
                    self.capture.capture_click_sequence(
                        "Click back to main menu, then F12",
                        min_clicks=0,
                        max_clicks=10,  # Ignore these clicks
                    )
                except CaptureAborted:
                    pass

        # Timing adjustment (optional)
        print("\n" + "-" * 50)
        print("TIMING CONFIGURATION (optional)")
        print("-" * 50)
        print("\nDefault wait time is 300ms. Press Enter to keep defaults, or enter new values.")

        adjust = input("\nAdjust timing values? [y/N]: ").strip().lower()
        if adjust == 'y':
            # Back button timing
            if back_button:
                wait_str = input(f"  Wait after back button (ms) [{back_wait_ms}]: ").strip()
                if wait_str:
                    back_wait_ms = int(wait_str)

            # Per-menu timing
            for cat_name, menu_nav in menus.items():
                if menu_nav.path:
                    wait_str = input(f"  Wait after navigating to {cat_name} (ms) [{menu_nav.wait_after_ms}]: ").strip()
                    if wait_str:
                        menus[cat_name] = MenuNavigation(path=menu_nav.path, wait_after_ms=int(wait_str))

            # Save updated timings
            self.calibration.navigation = NavigationInfo(
                menus=menus,
                back_button=back_button,
                back_wait_ms=back_wait_ms,
            )
            self._save()

        print("\n  Menu navigation calibration complete.")

    def _calibrate_all_inputs(self):
        """Calibrate all inputs organized by category."""
        print("\n" + "-" * 50)
        print("INPUT CALIBRATION")
        print("-" * 50)

        total = len(self.inputs)
        calibrated, _ = self.calibration.get_progress([i.name for i in self.inputs])
        print(f"\nProgress: {calibrated}/{total} inputs calibrated")

        for cat_name, cat_inputs in self.categories.items():
            print(f"\n{'=' * 50}")
            print(f"Category: {cat_name} ({len(cat_inputs)} inputs)")
            print("=" * 50)

            # Navigate to this category
            self._navigate_to_menu(cat_name)

            for inp in cat_inputs:
                if inp.name in self.calibration.inputs:
                    print(f"\n  [SKIP] {inp.name} (already calibrated)")
                    continue

                try:
                    self._calibrate_input(inp, cat_name)
                except CaptureAborted:
                    print(f"\n  Calibration aborted at {inp.name}")
                    raise

    def _navigate_to_menu(self, menu_key: str):
        """Prompt user to navigate to a menu (calibration never clicks)."""
        if menu_key == self._current_menu:
            return  # Already there

        print(f"\n  Please navigate to {menu_key} menu, then press F12 when ready.")
        self.capture.capture_click_sequence(
            f"Navigate to {menu_key}, then F12 when ready",
            min_clicks=0,
            max_clicks=20,  # Ignore all clicks during navigation
        )
        self._current_menu = menu_key

    def _calibrate_input(self, inp: InputType, menu: str):
        """Calibrate a single input."""
        input_type = get_input_type_name(inp)

        print(f"\n  --- {inp.name} ({input_type}) ---")

        # Step 1: Activation click
        activation = self.capture.capture_single_click(
            f"Click on '{inp.name}' row to select/activate it"
        )

        # Step 2: Type-specific control calibration
        if isinstance(inp, Slider):
            control = self._calibrate_slider_control(inp)
        elif isinstance(inp, CoordinateGrid):
            control = self._calibrate_grid_control(inp)
        elif isinstance(inp, ColorPicker):
            control = self._calibrate_colorpicker_control(inp)
        else:
            print(f"  [ERROR] Unknown input type: {type(inp)}")
            return

        # Create and save calibration
        input_cal = InputCalibration(
            type=input_type,
            activation_click=activation,
            control=control,
            menu=menu,
        )

        self.calibration.set_input(inp.name, input_cal)
        self._save()

        calibrated, total = self.calibration.get_progress([i.name for i in self.inputs])
        print(f"  [OK] {inp.name} calibrated ({calibrated}/{total})")

    def _calibrate_slider_control(self, inp: Slider) -> SliderControl:
        """Calibrate slider track endpoints."""
        print(f"  Slider has {inp.num_classes()} options")

        start, end = self.capture.capture_two_points(
            "Click LEFT/START of slider track",
            "Click RIGHT/END of slider track"
        )

        return SliderControl(track_start=start, track_end=end)

    def _calibrate_grid_control(self, inp: CoordinateGrid) -> GridControl:
        """Calibrate coordinate grid bounds."""
        print(f"  Grid is {inp.x_divisions}x{inp.y_divisions} = {inp.num_classes()} options")

        bounds = self.capture.capture_bounding_box(
            "Click TOP-LEFT corner of the grid",
            "Click BOTTOM-RIGHT corner of the grid"
        )

        return GridControl(bounds=bounds)

    def _calibrate_colorpicker_control(self, inp: ColorPicker) -> ColorPickerControl:
        """Calibrate color wheel center and radius."""
        print(f"  Color wheel has {inp.num_classes()} colors")

        center, radius = self.capture.capture_center_and_radius(
            "Click CENTER of the color wheel",
            "Click EDGE of the color wheel"
        )

        return ColorPickerControl(wheel_center=center, wheel_radius=radius)

    def resume_calibration(self):
        """Resume calibration from where it left off."""
        uncalibrated = self.calibration.get_uncalibrated_inputs([i.name for i in self.inputs])

        if not uncalibrated:
            print("All inputs are already calibrated!")
            return

        print(f"Resuming calibration: {len(uncalibrated)} inputs remaining")
        print(f"Uncalibrated: {', '.join(uncalibrated[:5])}{'...' if len(uncalibrated) > 5 else ''}")

        # Check if window/capture/menus are done
        if not self.calibration.window:
            self._calibrate_window()
        if not self.calibration.capture:
            self._calibrate_capture()
        if not self.calibration.navigation:
            self._calibrate_menus()

        # Calibrate remaining inputs
        try:
            for cat_name, cat_inputs in self.categories.items():
                remaining = [inp for inp in cat_inputs if inp.name in uncalibrated]
                if not remaining:
                    continue

                print(f"\n{'=' * 50}")
                print(f"Category: {cat_name} ({len(remaining)} remaining)")
                print("=" * 50)

                self._navigate_to_menu(cat_name)

                for inp in remaining:
                    self._calibrate_input(inp, cat_name)

            print("\n" + "=" * 70)
            print("CALIBRATION COMPLETE!")
            print("=" * 70)

        except CaptureAborted:
            print("\n\nCalibration paused. Use --resume to continue.")

    def calibrate_single_input(self, input_name: str):
        """Calibrate or recalibrate a single input by name."""
        inp = next((i for i in self.inputs if i.name == input_name), None)
        if not inp:
            print(f"Input '{input_name}' not found in schema")
            print(f"Available inputs: {', '.join(i.name for i in self.inputs[:10])}...")
            return

        cat_name = find_input_category(inp, self.categories)

        print(f"\nCalibrating: {input_name}")
        print(f"Category: {cat_name}")
        print(f"Type: {get_input_type_name(inp)}")

        # Navigate to menu if we have navigation data
        if self.calibration.navigation:
            self._navigate_to_menu(cat_name)

        try:
            self._calibrate_input(inp, cat_name)
            print("\nCalibration saved.")
        except CaptureAborted:
            print("\nCalibration cancelled.")

    def list_status(self):
        """Show calibration status for all inputs."""
        print(f"\nSchema: {self.calibration.schema_name}")
        print(f"Game: {self.calibration.game}")
        print()

        calibrated, total = self.calibration.get_progress([i.name for i in self.inputs])
        print(f"Progress: {calibrated}/{total} inputs calibrated")
        print()

        # Window/Capture/Navigation status
        print("Setup:")
        print(f"  Window:     {'OK' if self.calibration.window else 'Not calibrated'}")
        print(f"  Capture:    {'OK' if self.calibration.capture else 'Not calibrated'}")
        print(f"  Navigation: {'OK' if self.calibration.navigation else 'Not calibrated'}")
        print()

        # Per-category status
        print("Inputs by category:")
        for cat_name, cat_inputs in self.categories.items():
            cal_count = sum(1 for i in cat_inputs if i.name in self.calibration.inputs)
            status = "DONE" if cal_count == len(cat_inputs) else f"{cal_count}/{len(cat_inputs)}"
            print(f"  {cat_name}: {status}")

            # Show uncalibrated inputs
            uncal = [i.name for i in cat_inputs if i.name not in self.calibration.inputs]
            if uncal:
                for name in uncal[:3]:
                    print(f"    - {name}")
                if len(uncal) > 3:
                    print(f"    ... and {len(uncal) - 3} more")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive calibration tool for game character creator schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.calibrator --schema dai_human_female       # Full calibration
  python -m tools.calibrator --schema dai_human_female --resume  # Continue from last
  python -m tools.calibrator --schema dai_human_female --input "Ear Size"  # Single input
  python -m tools.calibrator --schema dai_human_female --list    # Show status
  python -m tools.calibrator --position                      # Show mouse position
  python -m tools.calibrator --schemas                       # List available schemas
        """
    )

    parser.add_argument("--schema", help="Schema name to calibrate (e.g., dai_human_female)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing calibration")
    parser.add_argument("--input", help="Calibrate single input by name")
    parser.add_argument("--list", action="store_true", help="Show calibration status")
    parser.add_argument("--position", action="store_true", help="Show mouse position (debug)")
    parser.add_argument("--schemas", action="store_true", help="List available schemas")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Duration for position display (default: 60s)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Position display mode
    if args.position:
        continuous_position_display(args.duration)
        return

    # List schemas
    if args.schemas:
        print("Available schemas:")
        for name in all_schemas():
            print(f"  - {name}")
        return

    # Schema required for other operations
    if not args.schema:
        print("Error: --schema is required")
        print("Use --schemas to list available schemas")
        print("Use --position for mouse position display")
        return

    # Load schema
    try:
        schema = get_schema(args.schema)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --schemas to list available schemas")
        return

    # Determine game slug
    if schema.GAME:
        game_slug = re.sub(r'(?<!^)(?=[A-Z])', '_', schema.GAME.__name__).lower()
    else:
        game_slug = "unknown"

    # Load or create calibration
    calibration = load_calibration(game_slug, args.schema)
    if calibration is None:
        calibration = CalibrationData(
            schema_name=args.schema,
            game=game_slug,
        )
        print(f"Creating new calibration for {args.schema}")
    else:
        print(f"Loaded existing calibration for {args.schema}")

    # Create calibrator
    calibrator = Calibrator(schema, calibration)

    # Execute requested operation
    if args.list:
        calibrator.list_status()
    elif args.input:
        calibrator.calibrate_single_input(args.input)
    elif args.resume:
        calibrator.resume_calibration()
    else:
        calibrator.run_full_calibration()


if __name__ == "__main__":
    main()
