#!/usr/bin/env python3
"""
Interactive calibration tool for game character creator schemas.

This tool uses keyboard-based navigation (arrow keys + enter) with mouse
only for coordinate grids and color pickers.

Usage:
    python -m tools.calibrator --schema dai_human_female
    python -m tools.calibrator --schema dai_human_female --list
    python -m tools.calibrator --position
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_schema import get_schema, all_schemas, BaseSchema

# Import schemas to trigger registration
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
    CalibrationData, WindowInfo, CaptureInfo,
    MenuInfo, InputInfo, GridBounds, ColorWheelInfo, KeyboardTiming,
)
from tools.calibration_ui import MouseCapture, CaptureAborted, continuous_position_display
from calibrations import load_calibration, save_calibration, get_calibration_path


def _load_categories_for_schema(schema_name: str) -> Dict[str, List[str]]:
    """Load menu categories from YAML macro file, falling back to empty dict."""
    try:
        from macro_schema import get_macro_config
        config = get_macro_config(schema_name)
        if config:
            return config.get_menu_categories()
    except ImportError:
        pass
    return {}


def extract_categories_from_schema(schema: BaseSchema) -> Dict[str, List[InputType]]:
    """Extract input categories from schema, loading from YAML macro if available."""
    inputs = schema.get_inputs()
    input_by_name = {inp.name: inp for inp in inputs}

    # Try to load categories from YAML macro
    schema_name = schema._get_schema_name()
    yaml_categories = _load_categories_for_schema(schema_name)

    categories: Dict[str, List[InputType]] = {}
    assigned_names = set()

    for cat_name, input_names in yaml_categories.items():
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

    return categories


class Calibrator:
    """Interactive calibration workflow manager using keyboard navigation."""

    def __init__(self, schema: BaseSchema, calibration: CalibrationData):
        self.schema = schema
        self.calibration = calibration
        self.inputs = schema.get_inputs()
        self._capture: Optional[MouseCapture] = None
        self.categories = extract_categories_from_schema(schema)

    @property
    def capture(self) -> MouseCapture:
        """Lazy-initialize mouse capture."""
        if self._capture is None:
            self._capture = MouseCapture()
            if self.calibration.window:
                x, y = self.calibration.window.reference_position
                w, h = self.calibration.window.expected_size
                self._capture.set_window_bounds(x, y, w, h)
        return self._capture

    def _get_game_slug(self) -> str:
        """Get the game slug for file paths."""
        if self.schema.GAME:
            name = self.schema.GAME.__name__
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return "unknown"

    def _save(self):
        """Save current calibration state."""
        game_slug = self._get_game_slug()
        save_calibration(self.calibration, game_slug, self.calibration.schema_name)
        print(f"  [Saved]")

    def run_full_calibration(self):
        """Run complete calibration workflow."""
        print("=" * 60)
        print("CHARACTER CREATOR CALIBRATION (Keyboard Mode)")
        print(f"Schema: {self.schema.__class__.__name__}")
        print(f"Game: {self.schema.game_name}")
        print("=" * 60)
        print("\nThis calibrator uses keyboard navigation.")
        print("You'll enter numbers for menu/input positions.")
        print("Mouse is only needed for grid bounds and color wheel.")

        try:
            self._calibrate_window()
            self._calibrate_capture()
            self._calibrate_menus()
            self._calibrate_inputs()
            self._calibrate_grid_bounds()
            self._calibrate_color_wheel()

            print("\n" + "=" * 60)
            print("CALIBRATION COMPLETE!")
            print(f"Saved to: {get_calibration_path(self._get_game_slug(), self.calibration.schema_name)}")
            print("=" * 60)

        except (CaptureAborted, KeyboardInterrupt):
            print("\n\nCalibration interrupted. Progress saved.")

    def _calibrate_window(self):
        """Calibrate window position."""
        print("\n" + "-" * 50)
        print("WINDOW CALIBRATION")
        print("-" * 50)

        try:
            from window_helpers import list_windows, find_window_info
            windows = list_windows()

            print("\nRunning windows:")
            for i, win in enumerate(windows):
                print(f"  {i+1:2}. {win.title[:50]}")
                print(f"      Pos: ({win.x}, {win.y})  Size: {win.width}x{win.height}")

            # Try auto-detect
            window_keyword = self.schema.GAME.WINDOW_KEYWORD if self.schema.GAME else None
            detected = find_window_info(window_keyword) if window_keyword else None

            if detected:
                print(f"\nAuto-detected: '{detected.title}'")
                print(f"  Pos: ({detected.x}, {detected.y})  Size: {detected.width}x{detected.height}")

            if self.calibration.window:
                print(f"\nCurrent: pos={self.calibration.window.reference_position}, size={self.calibration.window.expected_size}")

            choice = input("\nEnter window number, 'a' for auto-detect, or Enter to keep current: ").strip()

            if choice == 'a' and detected:
                self.calibration.window = WindowInfo(
                    reference_position=(detected.x, detected.y),
                    expected_size=(detected.width, detected.height),
                )
                self.calibration.resolution = (detected.width, detected.height)
                self._save()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(windows):
                    win = windows[idx]
                    self.calibration.window = WindowInfo(
                        reference_position=(win.x, win.y),
                        expected_size=(win.width, win.height),
                    )
                    self.calibration.resolution = (win.width, win.height)
                    self._save()
            elif choice == '' and self.calibration.window:
                print("  -> Keeping existing")

        except ImportError:
            print("window_helpers not available")

    def _calibrate_capture(self):
        """Calibrate screenshot capture region and view reset."""
        print("\n" + "-" * 50)
        print("CAPTURE REGION")
        print("-" * 50)

        # Current values
        current_box = self.calibration.capture.box if self.calibration.capture else None
        current_reset = self.calibration.capture.reset_view_position if self.calibration.capture else None
        current_scroll = self.calibration.capture.reset_scroll_count if self.calibration.capture else 10

        # Capture box
        print("\n--- Capture Box ---")
        if current_box:
            print(f"Current: {current_box}")
        else:
            print("Current: (not set)")
        print("[Click TOP-LEFT then BOTTOM-RIGHT of face region, or F12 to skip]")

        box = current_box
        try:
            clicks = self.capture.capture_click_sequence(
                "Click TOP-LEFT, BOTTOM-RIGHT, or F12 to skip",
                min_clicks=0,
                max_clicks=2,
            )
            if len(clicks) >= 2:
                p1, p2 = clicks[0], clicks[1]
                x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
                y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
                box = (x1, y1, x2, y2)
                print(f"  -> Set to: {box}")
        except CaptureAborted:
            pass

        # Reset view position (where to hover and scroll to zoom in)
        print("\n--- Reset View Position ---")
        print("After applying inputs, the character zooms out.")
        print("Click where to hover the mouse for scroll-wheel zoom reset.")
        if current_reset:
            print(f"Current: {current_reset}")
        else:
            print("Current: (not set)")
        print("[Click on the character, or F12 to skip]")

        reset_pos = current_reset
        try:
            clicks = self.capture.capture_click_sequence(
                "Click on character for zoom reset position, or F12 to skip",
                min_clicks=0,
                max_clicks=1,
            )
            if clicks:
                reset_pos = clicks[0]
                print(f"  -> Set to: {reset_pos}")
        except CaptureAborted:
            pass

        # Scroll count
        print(f"\n--- Scroll Count ---")
        print(f"How many scroll-wheel-ups to zoom in? [current: {current_scroll}]")
        scroll_input = input("Scroll count (Enter to keep): ").strip()
        scroll_count = int(scroll_input) if scroll_input.isdigit() else current_scroll

        # Save if we have a box
        if box:
            self.calibration.capture = CaptureInfo(
                box=box,
                reset_view_position=reset_pos,
                reset_scroll_count=scroll_count,
            )
            self._save()

    def _calibrate_menus(self):
        """Calibrate menu navigation (down counts)."""
        print("\n" + "-" * 50)
        print("MENU NAVIGATION")
        print("-" * 50)
        print("\nFor each menu, enter how many DOWN presses from the top.")
        print("(GENERAL is at top = 0, HEAD_AND_EARS = 1, etc.)")

        category_list = list(self.categories.keys())

        for i, cat_name in enumerate(category_list):
            existing = self.calibration.menus.get(cat_name)
            current = existing.down_count if existing else None

            prompt = f"\n{cat_name}: "
            if current is not None:
                prompt += f"[current: {current}] "
            prompt += "DOWN count (Enter to keep): "

            value = input(prompt).strip()

            if value.isdigit():
                self.calibration.menus[cat_name] = MenuInfo(down_count=int(value))
                self._save()
            elif value == '' and current is not None:
                pass  # Keep existing
            elif value == '' and current is None:
                # Default: assume sequential from 0
                self.calibration.menus[cat_name] = MenuInfo(down_count=i)
                self._save()

    def _calibrate_inputs(self):
        """Calibrate input navigation (down counts within each menu)."""
        print("\n" + "-" * 50)
        print("INPUT NAVIGATION")
        print("-" * 50)
        print("\nFor each input, enter DOWN presses from first input in that menu.")
        print("(First input = 0, second = 1, etc.)")
        print("Press Enter to accept default (sequential order).")

        for cat_name, cat_inputs in self.categories.items():
            print(f"\n=== {cat_name} ===")

            for i, inp in enumerate(cat_inputs):
                existing = self.calibration.inputs.get(inp.name)
                current = existing.down_count if existing else None

                input_type = "slider" if isinstance(inp, Slider) else "grid" if isinstance(inp, CoordinateGrid) else "color"
                prompt = f"  {inp.name} ({input_type}): "
                if current is not None:
                    prompt += f"[current: {current}] "
                else:
                    prompt += f"[default: {i}] "

                value = input(prompt).strip()

                if value.isdigit():
                    self.calibration.inputs[inp.name] = InputInfo(down_count=int(value), menu=cat_name)
                    self._save()
                elif value == '':
                    # Use existing or default
                    if current is None:
                        self.calibration.inputs[inp.name] = InputInfo(down_count=i, menu=cat_name)
                        self._save()

    def _calibrate_grid_bounds(self):
        """Calibrate grid bounds (shared by all CoordinateGrid inputs)."""
        print("\n" + "-" * 50)
        print("GRID BOUNDS (shared by all grid inputs)")
        print("-" * 50)

        # Check if schema has any grids
        has_grids = any(isinstance(inp, CoordinateGrid) for inp in self.inputs)
        if not has_grids:
            print("No grid inputs in this schema, skipping.")
            return

        if self.calibration.grid_bounds:
            gb = self.calibration.grid_bounds
            print(f"Current: ({gb.x1}, {gb.y1}) to ({gb.x2}, {gb.y2})")
        else:
            print("Current: (not set)")

        print("\nNavigate to any grid input in the game, then:")
        print("[Click TOP-LEFT then BOTTOM-RIGHT of the grid, or F12 to skip]")

        try:
            clicks = self.capture.capture_click_sequence(
                "Click TOP-LEFT, BOTTOM-RIGHT of grid, or F12 to skip",
                min_clicks=0,
                max_clicks=2,
            )
            if len(clicks) >= 2:
                p1, p2 = clicks[0], clicks[1]
                x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
                y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
                self.calibration.grid_bounds = GridBounds(x1=x1, y1=y1, x2=x2, y2=y2)
                self._save()
                print(f"  -> Set to: ({x1}, {y1}) to ({x2}, {y2})")
        except CaptureAborted:
            pass

    def _calibrate_color_wheel(self):
        """Calibrate color wheel hue ring (shared by all ColorPicker inputs)."""
        print("\n" + "-" * 50)
        print("COLOR WHEEL HUE RING")
        print("-" * 50)

        # Check if schema has any color pickers
        has_colors = any(isinstance(inp, ColorPicker) for inp in self.inputs)
        if not has_colors:
            print("No color picker inputs in this schema, skipping.")
            return

        if self.calibration.color_wheel:
            cw = self.calibration.color_wheel
            print(f"Current: center=({cw.center_x}, {cw.center_y}), ring_radius={cw.ring_radius}")
        else:
            print("Current: (not set)")

        print("\nThe color picker has an outer HUE RING and inner square.")
        print("We only use the outer hue ring for color selection.")
        print("\nNavigate to any color picker in the game, then:")
        print("  1. Click the CENTER of the inner square (the middle of the whole wheel)")
        print("  2. Click anywhere ON the hue ring itself (the colored outer circle)")
        print("[F12 to skip]")

        try:
            clicks = self.capture.capture_click_sequence(
                "Click center of wheel, then on the hue ring, or F12 to skip",
                min_clicks=0,
                max_clicks=2,
            )
            if len(clicks) >= 2:
                center = clicks[0]
                ring_point = clicks[1]
                import math
                ring_radius = int(math.sqrt((ring_point[0] - center[0])**2 + (ring_point[1] - center[1])**2))
                self.calibration.color_wheel = ColorWheelInfo(
                    center_x=center[0],
                    center_y=center[1],
                    ring_radius=ring_radius,
                )
                self._save()
                print(f"  -> Set to: center=({center[0]}, {center[1]}), ring_radius={ring_radius}")
        except CaptureAborted:
            pass

    def list_status(self):
        """Show calibration status."""
        print(f"\nSchema: {self.calibration.schema_name}")
        print(f"Game: {self.calibration.game}")
        print(f"Version: {self.calibration.version}")
        print()

        # Setup status
        print("Setup:")
        print(f"  Window:      {'OK' if self.calibration.window else 'Not set'}")
        print(f"  Capture:     {'OK' if self.calibration.capture else 'Not set'}")
        print(f"  Grid bounds: {'OK' if self.calibration.grid_bounds else 'Not set'}")
        print(f"  Color wheel: {'OK' if self.calibration.color_wheel else 'Not set'}")
        print()

        # Menu status
        print(f"Menus: {len(self.calibration.menus)}/{len(self.categories)}")
        for cat_name in self.categories:
            info = self.calibration.menus.get(cat_name)
            if info:
                print(f"  {cat_name}: down={info.down_count}")
            else:
                print(f"  {cat_name}: NOT SET")
        print()

        # Input status
        total_inputs = len(self.inputs)
        calibrated_inputs = len(self.calibration.inputs)
        print(f"Inputs: {calibrated_inputs}/{total_inputs}")

        for cat_name, cat_inputs in self.categories.items():
            cat_calibrated = sum(1 for inp in cat_inputs if inp.name in self.calibration.inputs)
            print(f"  {cat_name}: {cat_calibrated}/{len(cat_inputs)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration tool for game character creator (keyboard navigation)",
    )
    parser.add_argument("--schema", help="Schema name to calibrate")
    parser.add_argument("--list", action="store_true", help="Show calibration status")
    parser.add_argument("--position", action="store_true", help="Show mouse position")
    parser.add_argument("--schemas", action="store_true", help="List available schemas")
    parser.add_argument("--duration", type=float, default=60.0, help="Position display duration")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.position:
        continuous_position_display(args.duration)
        return

    if args.schemas:
        print("Available schemas:")
        for name in all_schemas():
            print(f"  - {name}")
        return

    if not args.schema:
        print("Error: --schema is required")
        print("Use --schemas to list available schemas")
        return

    try:
        schema = get_schema(args.schema)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Get game slug
    if schema.GAME:
        game_slug = re.sub(r'(?<!^)(?=[A-Z])', '_', schema.GAME.__name__).lower()
    else:
        game_slug = "unknown"

    # Load or create calibration
    calibration = load_calibration(game_slug, args.schema)
    if calibration is None:
        calibration = CalibrationData(schema_name=args.schema, game=game_slug)
        print(f"Creating new calibration for {args.schema}")
    else:
        print(f"Loaded existing calibration for {args.schema}")

    calibrator = Calibrator(schema, calibration)

    if args.list:
        calibrator.list_status()
    else:
        calibrator.run_full_calibration()


if __name__ == "__main__":
    main()
