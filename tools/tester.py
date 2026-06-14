#!/usr/bin/env python3
"""
Calibration tester tool for verifying game automation.

This tool generates a single random character and applies it to the game,
reporting detailed information about what should happen at each step for
verification purposes.

Usage:
    python -m tools.tester --schema dai_human_female
    python -m tools.tester --schema dai_human_female --code ABC123...
    python -m tools.tester --schema dai_human_female --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_schema import get_schema, all_schemas, BaseSchema
from input_types import Slider, CoordinateGrid, ColorPicker, InputType

# Import schemas to trigger registration
def _import_schemas():
    """Import available schemas, handling missing dependencies gracefully."""
    try:
        from schemas.dai_human_female import DAI_Human_Female_Schema
    except ImportError:
        pass
    try:
        from schemas.me1_female import ME1_Female_Schema
    except ImportError:
        pass
    try:
        from schemas.me1_male import ME1_Male_Schema
    except ImportError:
        pass
    try:
        from schemas.me3_female import ME3_Female_Schema
    except ImportError:
        pass
    try:
        from schemas.me3_male import ME3_Male_Schema
    except ImportError:
        pass

_import_schemas()

from calibrations import load_calibration, get_calibration_path


def _load_categories_for_schema(schema_name: str) -> Dict[str, List[str]]:
    """Load menu categories from YAML macro file."""
    try:
        from macro_schema import get_macro_config
        config = get_macro_config(schema_name)
        if config:
            return config.get_menu_categories()
    except ImportError:
        pass
    return {}


class CalibrationTester:
    """Calibration verification tester."""

    def __init__(self, schema: BaseSchema):
        self.schema = schema
        self.calibration = schema.load_calibration()
        self.inputs = schema.get_inputs()
        self.input_by_name = {inp.name: inp for inp in self.inputs}
        self._categories = _load_categories_for_schema(schema._get_schema_name())

    def _get_game_slug(self) -> str:
        """Get the game slug for file paths."""
        if self.schema.GAME:
            name = self.schema.GAME.__name__
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return "unknown"

    def check_calibration(self) -> bool:
        """Check if calibration is complete enough for testing."""
        if not self.calibration:
            print("ERROR: No calibration found!")
            print(f"Run: python -m tools.calibrator --schema {self.schema._get_schema_name()}")
            return False

        issues = []

        if not self.calibration.window:
            issues.append("- Window not calibrated")
        if not self.calibration.capture:
            issues.append("- Capture region not calibrated")
        if not self.calibration.menus:
            issues.append("- No menus calibrated")
        if not self.calibration.inputs:
            issues.append("- No inputs calibrated")

        # Check for grids and colors
        has_grids = any(isinstance(inp, CoordinateGrid) for inp in self.inputs)
        has_colors = any(isinstance(inp, ColorPicker) for inp in self.inputs)

        if has_grids and not self.calibration.grid_bounds:
            issues.append("- Grid bounds not calibrated (needed for CoordinateGrid inputs)")
        if has_colors and not self.calibration.color_wheel:
            issues.append("- Color wheel not calibrated (needed for ColorPicker inputs)")

        if issues:
            print("CALIBRATION INCOMPLETE:")
            for issue in issues:
                print(issue)
            return False

        return True

    def print_calibration_summary(self):
        """Print a summary of the calibration state."""
        cal = self.calibration
        print("=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)

        if cal.window:
            print(f"Window: pos={cal.window.reference_position}, size={cal.window.expected_size}")
        else:
            print("Window: NOT SET")

        if cal.capture:
            print(f"Capture box: {cal.capture.box}")
            if cal.capture.reset_view_position:
                print(f"Reset position: {cal.capture.reset_view_position}, scroll={cal.capture.reset_scroll_count}")
        else:
            print("Capture: NOT SET")

        if cal.grid_bounds:
            gb = cal.grid_bounds
            print(f"Grid bounds: ({gb.x1}, {gb.y1}) to ({gb.x2}, {gb.y2})")
        else:
            print("Grid bounds: NOT SET")

        if cal.color_wheel:
            cw = cal.color_wheel
            print(f"Color wheel: center=({cw.center_x}, {cw.center_y}), radius={cw.ring_radius}")
        else:
            print("Color wheel: NOT SET")

        print(f"\nMenus calibrated: {len(cal.menus)}")
        print(f"Inputs calibrated: {len(cal.inputs)}")
        print()

    def decode_and_describe(self, code: str) -> List[Tuple[InputType, int, str]]:
        """
        Decode a code and return descriptions for each input.

        Returns:
            List of (input, value, description) tuples
        """
        decoded = self.schema.decode_code(code)
        results = []

        for inp, value in decoded:
            if isinstance(inp, Slider):
                desc = f"slider position {value}/{inp.num_classes()-1}"
            elif isinstance(inp, CoordinateGrid):
                col = value % inp.x_divisions
                row = value // inp.x_divisions
                desc = f"grid position ({col}, {row}) of ({inp.x_divisions}x{inp.y_divisions})"
            elif isinstance(inp, ColorPicker):
                # Convert value to approximate hue description
                hue_names = ["red", "orange", "yellow", "lime", "green", "teal",
                            "cyan", "azure", "blue", "violet", "magenta", "rose"]
                if value < len(hue_names):
                    desc = f"hue {value} ({hue_names[value]})"
                else:
                    desc = f"hue {value}/{inp.num_classes()-1}"
            else:
                desc = f"value {value}"

            results.append((inp, value, desc))

        return results

    def print_character_description(self, code: str):
        """Print a human-readable description of what the character should look like."""
        print("=" * 60)
        print("CHARACTER DESCRIPTION")
        print("=" * 60)
        print(f"Code: {code}")
        print()

        decoded = self.decode_and_describe(code)
        cal = self.calibration

        # Group by category
        input_to_category = {}
        for cat_name, input_names in self._categories.items():
            for name in input_names:
                input_to_category[name] = cat_name

        current_category = None
        for inp, value, desc in decoded:
            cat = input_to_category.get(inp.name, "OTHER")
            if cat != current_category:
                current_category = cat
                print(f"\n--- {cat} ---")

            # Get action info
            action_info = ""
            if isinstance(inp, Slider):
                action_info = f" -> LEFT to reset, RIGHT {value} times"
            elif isinstance(inp, CoordinateGrid) and cal and cal.grid_bounds:
                x, y = cal.grid_bounds.get_click_position(value, inp.x_divisions, inp.y_divisions)
                action_info = f" -> click ({x}, {y})"
            elif isinstance(inp, ColorPicker) and cal and cal.color_wheel:
                x, y = cal.color_wheel.get_click_position(value, inp.num_classes())
                action_info = f" -> click ({x}, {y})"

            print(f"  {inp.name}: {desc}{action_info}")

        print()

    def print_action_sequence(self, code: str):
        """Print the exact sequence of actions that will be taken."""
        print("=" * 60)
        print("ACTION SEQUENCE (what the tool will do)")
        print("=" * 60)

        decoded = self.decode_and_describe(code)
        cal = self.calibration

        if not cal:
            print("No calibration - cannot show action sequence")
            return

        # Group by menu for efficient traversal
        by_menu: Dict[str, List[Tuple[InputType, int, str]]] = {}
        for inp, value, desc in decoded:
            if inp.name in cal.inputs:
                menu = cal.inputs[inp.name].menu
                if menu not in by_menu:
                    by_menu[menu] = []
                by_menu[menu].append((inp, value, desc))

        step = 0
        for menu_key in cal.menus.keys():
            if menu_key not in by_menu:
                continue

            menu_info = cal.menus[menu_key]
            step += 1
            print(f"\n{step}. Navigate to {menu_key}")
            print(f"   - Press UP x20 (ensure at top)")
            print(f"   - Press DOWN x{menu_info.down_count}")
            print(f"   - Press ENTER (enter menu)")

            # Sort inputs by down_count
            menu_inputs = sorted(by_menu[menu_key], key=lambda x: cal.inputs[x[0].name].down_count)

            current_idx = 0
            for inp, value, desc in menu_inputs:
                input_info = cal.inputs[inp.name]
                target_idx = input_info.down_count

                step += 1
                if target_idx > current_idx:
                    print(f"\n{step}. {inp.name}")
                    print(f"   - Press DOWN x{target_idx - current_idx}")
                elif target_idx < current_idx:
                    print(f"\n{step}. {inp.name}")
                    print(f"   - Press UP x{current_idx - target_idx}")
                else:
                    print(f"\n{step}. {inp.name} (already selected)")

                if isinstance(inp, Slider):
                    print(f"   - Press LEFT x{inp.num_classes()} (reset to 0)")
                    print(f"   - Press RIGHT x{value} (set to {value})")
                elif isinstance(inp, CoordinateGrid) and cal.grid_bounds:
                    x, y = cal.grid_bounds.get_click_position(value, inp.x_divisions, inp.y_divisions)
                    col = value % inp.x_divisions
                    row = value // inp.x_divisions
                    print(f"   - Click grid at ({x}, {y}) for cell ({col}, {row})")
                elif isinstance(inp, ColorPicker) and cal.color_wheel:
                    x, y = cal.color_wheel.get_click_position(value, inp.num_classes())
                    print(f"   - Click hue ring at ({x}, {y}) for hue {value}")

                current_idx = target_idx

        # Reset view
        step += 1
        print(f"\n{step}. Reset view for capture")
        print(f"   - Press ESCAPE")
        if cal.capture and cal.capture.reset_view_position:
            x, y = cal.capture.reset_view_position
            print(f"   - Move mouse to ({x}, {y})")
            print(f"   - Scroll up x{cal.capture.reset_scroll_count} (zoom in)")

        # Capture
        step += 1
        print(f"\n{step}. Capture screenshot")
        if cal.capture:
            print(f"   - Grab region: {cal.capture.box}")

        print()

    def run_test(self, code: Optional[str] = None, dry_run: bool = False):
        """
        Run the calibration test.

        Args:
            code: Specific code to use, or None to generate random
            dry_run: If True, only show what would happen without applying
        """
        print("=" * 60)
        print("CALIBRATION TESTER")
        print(f"Schema: {self.schema.__class__.__name__}")
        print(f"Game: {self.schema.game_name}")
        print("=" * 60)

        # Check calibration
        if not self.check_calibration():
            return

        self.print_calibration_summary()

        # Generate or use provided code
        if code:
            try:
                # Validate code length
                self.schema.decode_code(code)
            except ValueError as e:
                print(f"ERROR: Invalid code - {e}")
                return
        else:
            code = self.schema.generate_code(biased=True)
            print(f"Generated random character code: {code}")

        # Show character description
        self.print_character_description(code)

        # Show action sequence
        self.print_action_sequence(code)

        if dry_run:
            print("=" * 60)
            print("DRY RUN - no actions taken")
            print("=" * 60)
            return

        # Wait for user confirmation
        print("=" * 60)
        print("READY TO APPLY")
        print("=" * 60)
        print("\nMake sure:")
        print("1. The game is running and in the character creator")
        print("2. You are on the main menu (not in a submenu)")
        print("3. The character is at default/starting position")
        print()
        input("Press ENTER to apply the character, or Ctrl+C to cancel...")

        # Apply the character
        print("\nStarting in 3 seconds... (switch to game window now)")
        import time
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        print("Applying character...")
        try:
            # Focus the game window
            try:
                self.schema.setup_window()
                print("  Window focused")
            except RuntimeError as e:
                print(f"WARNING: {e}")
                print("Continuing anyway - make sure game is focused!")

            # Small delay to ensure window is ready
            time.sleep(0.5)

            print("  Applying inputs...")
            self.schema.apply_all_calibrated_inputs(code)
            print("Inputs applied!")

            # Reset view
            print("Resetting view for capture...")
            self.schema.reset_view_for_capture()
            print("View reset!")

            # Capture screenshot
            print("Capturing screenshot...")
            screenshot = self.schema.capture_image(code)
            if screenshot:
                # Save to test output
                output_dir = Path(self.schema.get_output_dir()) / "test"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_{timestamp}.png"
                filepath = output_dir / filename
                screenshot.save(filepath)
                print(f"Screenshot saved: {filepath}")
            else:
                print("WARNING: Could not capture screenshot")

        except Exception as e:
            print(f"ERROR during application: {e}")
            import traceback
            traceback.print_exc()
            return

        # Final summary
        print()
        print("=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"\nCode applied: {code}")
        print("\nPlease verify:")
        print("1. Each slider moved to the correct position")
        print("2. Grid inputs clicked the right cells")
        print("3. Color pickers selected the right hues")
        print("4. The final screenshot matches the character")
        print()
        print("If anything is wrong, check the calibration values and re-run.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test calibration by generating and applying a single character",
    )
    parser.add_argument("--schema", required=True, help="Schema name to test")
    parser.add_argument("--code", help="Specific code to use (default: random)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would happen without applying")
    parser.add_argument("--schemas", action="store_true", help="List available schemas")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.schemas:
        print("Available schemas:")
        for name in all_schemas():
            print(f"  - {name}")
        return

    try:
        schema = get_schema(args.schema)
    except ValueError as e:
        print(f"Error: {e}")
        return

    tester = CalibrationTester(schema)
    tester.run_test(code=args.code, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
