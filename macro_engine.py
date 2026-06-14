"""
Macro execution engine for YAML-defined game automation.

Provides:
- ActionRunner: Executes atomic actions (key press, click, wait)
- MenuNavigator: Recursive menu traversal using the YAML tree
- InputApplier: Dispatches to handler by input type
- MacroEngine: Top-level orchestrator
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from macro_schema import MacroConfig, MenuNode, InputDef, TimingConfig, NavigationConfig
from input_types import InputType, Slider, CoordinateGrid, ColorPicker

if TYPE_CHECKING:
    from tools.calibration_schema import CalibrationData

# Conditional import of pyautogui
_pyautogui = None
if os.getenv("APP_MODE") != "docker":
    try:
        import pyautogui
        _pyautogui = pyautogui
        _pyautogui.FAILSAFE = False
        _pyautogui.PAUSE = 0.01
    except ImportError:
        pass


class ActionRunner:
    """Executes atomic actions: key presses, mouse clicks, waits."""

    def __init__(self, timing: TimingConfig, dry_run: bool = False):
        self.timing = timing
        self.dry_run = dry_run
        self.log: List[str] = []

    def execute_action(self, action: str, context: Optional[dict] = None):
        """
        Execute an action string like "UP x20", "RIGHT x{value}", "SPACE".

        Args:
            action: The action string, possibly with {variables}
            context: Variable substitutions for the action string
        """
        if context:
            action = action.format(**context)

        # Parse "KEY xN" pattern
        match = re.match(r'(\w+)\s+x(\d+)', action)
        if match:
            key = match.group(1).lower()
            count = int(match.group(2))
            self._press_key(key, count)
        else:
            self._press_key(action.lower(), 1)

    def press_key(self, key: str, count: int = 1):
        """Press a key N times with timing delay."""
        self._press_key(key, count)

    def _press_key(self, key: str, count: int):
        delay = self.timing.key_press_ms / 1000
        delay = max(delay, 0.1)

        for _ in range(count):
            self.log.append(f"KEY: {key}")
            if not self.dry_run and _pyautogui:
                _pyautogui.press(key)
            time.sleep(delay if not self.dry_run else 0)

    def click(self, x: int, y: int):
        """Click at screen coordinates."""
        self.log.append(f"CLICK: ({x}, {y})")
        if not self.dry_run and _pyautogui:
            _pyautogui.click(x, y)
            time.sleep(0.1)

    def wait(self, ms: int):
        """Wait for a specified number of milliseconds."""
        self.log.append(f"WAIT: {ms}ms")
        if not self.dry_run:
            time.sleep(ms / 1000)


class MenuNavigator:
    """
    Navigates the game's menu hierarchy using keyboard.

    Tracks current position as a path through the menu tree.
    Uses menu_reset (UP x20) + DOWN to reach target menus.
    """

    def __init__(self, config: MacroConfig, runner: ActionRunner):
        self.config = config
        self.runner = runner
        self.nav = config.navigation
        self._current_menu: Optional[str] = None
        self._current_input_idx: int = 0

    def navigate_to_menu(self, menu_name: str):
        """Navigate to a top-level menu by name."""
        if menu_name not in self.config.menus:
            return

        menu = self.config.menus[menu_name]

        # If we're in a submenu, press back to go up
        if self._current_menu is not None:
            self.runner.press_key(self.nav.back_key, 1)
            self.runner.wait(self.config.timing.after_back_ms)

        # Reset to top of menu list
        self.runner.execute_action(self.nav.menu_reset)
        self.runner.wait(300)

        # Force-activate by going down one first
        self.runner.press_key("down", 1)
        self.runner.wait(300)

        if menu.position == 0:
            self.runner.press_key("up", 1)
            self.runner.wait(300)
        elif menu.position > 1:
            self.runner.press_key("down", menu.position - 1)
            self.runner.wait(300)

        # Select the menu
        self.runner.press_key(self.nav.select_key, 1)
        self.runner.wait(self.config.timing.after_select_ms)

        self._current_menu = menu_name
        self._current_input_idx = 0

    def navigate_to_input_index(self, target_idx: int):
        """Navigate to an input at the given index within the current menu."""
        if target_idx > self._current_input_idx:
            self.runner.press_key("down", target_idx - self._current_input_idx)
        elif target_idx < self._current_input_idx:
            self.runner.press_key("up", self._current_input_idx - target_idx)
        self._current_input_idx = target_idx


class InputApplier:
    """Applies values to inputs based on their type."""

    def __init__(self, config: MacroConfig, runner: ActionRunner,
                 calibration: Optional["CalibrationData"] = None):
        self.config = config
        self.runner = runner
        self.calibration = calibration

    def apply_slider(self, inp: Slider, value: int):
        """Apply a slider value: LEFT to reset, RIGHT to target."""
        # Reset to minimum
        self.runner.press_key("left", inp.num_classes())
        self.runner.wait(self.config.timing.after_slider_reset_ms)
        # Move to target
        self.runner.press_key("right", value)

    def apply_grid(self, inp: CoordinateGrid, value: int):
        """Apply a grid value: mouse click at calibrated position."""
        cal = self.calibration
        if not cal or not cal.grid_bounds:
            return
        x, y = cal.grid_bounds.get_click_position(value, inp.x_divisions, inp.y_divisions)
        self.runner.click(x, y)

    def apply_color(self, inp: ColorPicker, value: int):
        """Apply a color value: mouse click on the hue ring."""
        cal = self.calibration
        if not cal or not cal.color_wheel:
            return
        x, y = cal.color_wheel.get_click_position(value, inp.num_classes())
        self.runner.click(x, y)

    def apply(self, inp: InputType, value: int):
        """Apply a value to any input type."""
        if isinstance(inp, Slider):
            self.apply_slider(inp, value)
        elif isinstance(inp, CoordinateGrid):
            self.apply_grid(inp, value)
        elif isinstance(inp, ColorPicker):
            self.apply_color(inp, value)


class MacroEngine:
    """
    Top-level orchestrator that applies a code to a game using YAML config.

    Combines MenuNavigator + InputApplier to process all inputs.
    """

    def __init__(self, config: MacroConfig,
                 calibration: Optional["CalibrationData"] = None,
                 dry_run: bool = False):
        self.config = config
        self.calibration = calibration
        self.dry_run = dry_run
        self.runner = ActionRunner(config.timing, dry_run=dry_run)
        self.navigator = MenuNavigator(config, self.runner)
        self.applier = InputApplier(config, self.runner, calibration)

    def apply_code(self, code: str, inputs: List[InputType]):
        """
        Apply a decoded code to the game.

        Args:
            code: The character code string
            inputs: The list of InputType objects (from schema.get_inputs())
        """
        mode = self.config.navigation.mode

        if mode == "direct_ui":
            self._apply_direct_ui(code, inputs)
        elif mode == "code_paste":
            self._apply_code_paste(code)
        elif mode == "preset_file":
            self._apply_preset_file(code)

    def _apply_direct_ui(self, code: str, inputs: List[InputType]):
        """Apply inputs using keyboard navigation + mouse clicks."""
        from base_schema import BaseSchema

        # Build input->value mapping from the code
        # We need a temporary schema-like decode
        decoded = self._decode_code(code, inputs)

        if not self.config.menus:
            return

        # Group inputs by menu
        input_defs = self.config.get_all_input_defs()
        input_name_to_def = {d.name: d for d in input_defs}

        # Build menu -> [(input, value, idx_in_menu)] mapping
        by_menu: Dict[str, List[Tuple[InputType, int, int]]] = {}
        input_idx = 0
        for menu_name, menu_node in self.config.menus.items():
            menu_inputs = []
            for local_idx, inp_def in enumerate(menu_node.inputs):
                if input_idx < len(decoded):
                    inp, value = decoded[input_idx]
                    menu_inputs.append((inp, value, local_idx))
                input_idx += 1
            if menu_inputs:
                by_menu[menu_name] = menu_inputs

        # Process each menu
        for menu_name in self.config.menus:
            if menu_name not in by_menu:
                continue

            self.navigator.navigate_to_menu(menu_name)

            for inp, value, local_idx in by_menu[menu_name]:
                self.navigator.navigate_to_input_index(local_idx)
                self.applier.apply(inp, value)

    def _apply_code_paste(self, code: str):
        """Apply code via clipboard paste."""
        if self.dry_run:
            self.runner.log.append(f"CLIPBOARD: {code}")
            self.runner.log.append("PASTE: paste_button")
            return

        try:
            from window_helpers import add_to_clipboard
            add_to_clipboard(code)
        except ImportError:
            pass

    def _apply_preset_file(self, code: str):
        """Apply code by writing a preset file."""
        if not self.config.preset:
            return
        self.runner.log.append(f"PRESET_FILE: {self.config.preset.filename.format(code=code)}")

    def _decode_code(self, code: str, inputs: List[InputType]) -> List[Tuple[InputType, int]]:
        """Decode a code string using the provided input list."""
        expected_len = sum(inp.encoding_width for inp in inputs)
        if len(code) != expected_len:
            raise ValueError(f"Code '{code}' length {len(code)} != expected {expected_len}")

        result = []
        offset = 0
        for inp in inputs:
            width = inp.encoding_width
            symbol = code[offset:offset + width]
            value = inp.decode(symbol)
            result.append((inp, value))
            offset += width
        return result

    def get_action_log(self) -> List[str]:
        """Return the log of all actions taken (useful for dry_run)."""
        return self.runner.log
