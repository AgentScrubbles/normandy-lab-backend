from abc import ABC, abstractmethod
import os
import re
import random
import string
import time
from typing import ClassVar, Dict, List, Optional, Tuple, Type, TYPE_CHECKING
import warnings

# Import InputType hierarchy from the new module
from input_types import InputType, Slider, CoordinateGrid, ColorPicker, AnyInputType

# Import Game hierarchy
from game import Game, AutomationMode, MassEffect1, MassEffect3, DragonAgeInquisition, Fallout4, Cyberpunk2077

# Calibration imports (lazy loaded to avoid circular imports)
if TYPE_CHECKING:
    from tools.calibration_schema import CalibrationData, InputCalibration

# Conditional import of pyautogui (not available in docker mode)
_pyautogui = None
if os.getenv("APP_MODE") != "docker":
    try:
        import pyautogui
        _pyautogui = pyautogui
        # Disable failsafe (moving mouse to corner aborts) and reduce default pause
        _pyautogui.FAILSAFE = False
        _pyautogui.PAUSE = 0.01  # Minimal pause between pyautogui calls
    except ImportError:
        pass

# Re-export for backward compatibility and convenience
__all__ = [
    # Schema classes
    'BaseSchema', 'register_schema', 'get_schema', 'all_schemas',
    # Input types
    'Slider', 'InputType', 'CoordinateGrid', 'ColorPicker', 'AnyInputType',
    # Game classes
    'Game', 'AutomationMode', 'MassEffect1', 'MassEffect3', 'DragonAgeInquisition', 'Fallout4',
]

_schema_registry: dict[str, type["BaseSchema"]] = {}

def register_schema(name: str):
    """Decorator to register a BaseSchema subclass by name."""
    def wrapper(cls):
        _schema_registry[name] = cls
        return cls
    return wrapper

def get_schema(name: str, *args, **kwargs) -> "BaseSchema":
    cls = _schema_registry.get(name)
    if cls is None:
        raise ValueError(f"No schema registered with name '{name}'")
    return cls(*args, **kwargs)

def all_schemas() -> list[str]:
    return list(_schema_registry.keys())


class BaseSchema(ABC):
    """
    Abstract base class for game-specific character creator schemas.

    Each schema defines:
    - The game it belongs to (GAME class variable)
    - The inputs available in the character creator (sliders, color pickers, etc.)
    - How to encode/decode character codes
    - Game automation (window management, clicking, capturing)

    Subclasses must:
    - Set GAME to their game class (MassEffect1, DragonAgeInquisition, etc.)
    - Implement get_inputs() to define their input types
    - Implement abstract methods for game automation

    The deprecated get_sliders() method is still supported for backward compatibility.
    """

    # Subclasses must set this to their game class
    # Default to None for backward compatibility during migration
    GAME: ClassVar[Optional[Type[Game]]] = None

    @abstractmethod
    def get_inputs(self) -> List[InputType]:
        """
        Return list of all input types for this schema.

        This is the primary method for defining schema inputs.
        Override this in subclasses.
        """
        pass

    def get_sliders(self) -> List[Slider]:
        """
        Deprecated: Use get_inputs() instead.

        Returns only Slider-type inputs for backward compatibility.
        """
        return [inp for inp in self.get_inputs() if isinstance(inp, Slider)]

    def get_options_per_index(self) -> List[int]:
        """Return list of option counts for each input position."""
        return [inp.num_classes() for inp in self.get_inputs()]

    @abstractmethod
    def get_code_file(self) -> str:
        """Path to file containing generated character codes."""
        pass

    @abstractmethod
    def setup_window(self):
        """Setup game window (focus, position, etc.)."""
        pass

    @abstractmethod
    def send_code_to_game(self, code: str):
        """Send a character code to the game (e.g., via clipboard paste)."""
        pass

    @abstractmethod
    def get_output_dir(self) -> str:
        """Directory for saving captured images."""
        pass

    @abstractmethod
    def capture_image(self, code):
        """Capture a screenshot of the character with the given code."""
        pass

    def generate_code(self, biased=True) -> str:
        """
        Generate a random character code.

        Args:
            biased: If True, preferentially sample from biased inputs (key features)

        Returns:
            Generated code string
        """
        code_chars = []
        for inp in self.get_inputs():
            # Determine which value to sample
            if biased and inp.biased:
                # Sample uniformly for key features
                value = random.randint(0, inp.num_classes() - 1)
            elif inp.num_classes() > 20:
                # For large option counts, sample from a subset
                subset_size = min(4, inp.num_classes())
                subset = random.sample(range(inp.num_classes()), subset_size)
                value = random.choice(subset)
            else:
                value = random.randint(0, inp.num_classes() - 1)

            # Encode the value
            code_chars.append(inp.encode(value))
        return ''.join(code_chars)

    def generate_codes(self, n=10000) -> List[str]:
        """Generate n unique character codes."""
        codes = set()
        while len(codes) < n:
            code = self.generate_code(biased=True)
            codes.add(code)
        return sorted(codes)

    def option_chars(self, n: int) -> List[str]:
        """
        Return the list of valid characters for n options.

        Deprecated: Use InputType.encode() instead.
        Kept for backward compatibility with existing ME schemas.

        This uses legacy 1-indexed encoding ('1'-'9', 'A'-'Z').
        For extended encoding (62+ chars), use the InputType methods directly.
        """
        if n <= 9:
            return [str(i) for i in range(1, n+1)]
        elif n <= 35:
            # Legacy encoding: 9 digits + up to 26 letters = 35 max
            return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase[:n-9])
        else:
            # For n > 35, return full legacy alphabet (caller should use InputType.encode())
            warnings.warn(
                f"option_chars({n}) exceeds legacy limit of 35. Use InputType.encode() instead.",
                DeprecationWarning
            )
            return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase)

    def parse_filename(self, stem: str) -> Optional[List[int]]:
        """
        Parse a filename stem into label indices.

        Override in subclasses that use non-standard filename encoding
        (e.g., underscore-separated integers instead of concatenated chars).

        Returns None to use the default character-based decoding.
        """
        return None

    def decode_code(self, code: str) -> List[Tuple[InputType, int]]:
        """
        Decode a code string into a list of (input, value) pairs.

        Args:
            code: The encoded character code string

        Returns:
            List of (InputType, decoded_value) tuples
        """
        inputs = self.get_inputs()
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

    def apply_input_value(self, inp: InputType, value: int, click_delay: float = 0.1):
        """
        Apply a value to an input in the game UI.

        For CoordinateGrid and ColorPicker inputs, this clicks the calculated
        screen coordinates. For Slider inputs in code-paste games (ME), this
        is a no-op since sliders are handled by code paste.

        Override in subclasses for game-specific automation (e.g., to handle
        sliders via direct UI clicks in games without code paste).

        Args:
            inp: The input type to apply
            value: The value index (0-indexed) to apply
            click_delay: Delay in seconds after clicking (for UI to respond)
        """
        if _pyautogui is None:
            # pyautogui not available (docker mode or not installed)
            return

        if isinstance(inp, (CoordinateGrid, ColorPicker)):
            x, y = inp.get_click_coordinates(value)
            _pyautogui.click(x, y)
            time.sleep(click_delay)
        # Sliders in ME-style games are handled by code paste, not direct clicks
        # Subclasses can override for games requiring direct UI manipulation (DAI)

    def apply_all_inputs(self, code: str, click_delay: float = 0.1, inter_input_delay: float = 0.05):
        """
        Apply all input values from a code string to the game UI.

        This is the primary method for games that use direct UI manipulation
        (like Dragon Age Inquisition) instead of code paste.

        Args:
            code: The encoded character code string
            click_delay: Delay after each click (for UI to respond)
            inter_input_delay: Delay between inputs (for UI stability)

        Note:
            For Mass Effect-style games, use send_code_to_game() instead.
        """
        decoded = self.decode_code(code)
        for inp, value in decoded:
            self.apply_input_value(inp, value, click_delay=click_delay)
            time.sleep(inter_input_delay)

    @property
    def game(self) -> Optional[Type[Game]]:
        """Return the game class for this schema."""
        return self.GAME

    @property
    def game_name(self) -> str:
        """Return the human-readable game name."""
        if self.GAME is not None:
            return self.GAME.GAME_NAME
        return "Unknown Game"

    @property
    def uses_code_paste(self) -> bool:
        """
        Whether this schema uses clipboard code paste (ME-style) or direct UI manipulation (DAI-style).

        Determined by the GAME class. Default is True for backward compatibility with ME schemas.
        """
        if self.GAME is not None:
            return self.GAME.uses_code_paste()
        # Default for backward compatibility during migration
        return True

    @property
    def uses_direct_ui(self) -> bool:
        """
        Whether this schema uses direct UI manipulation (DAI-style).

        Determined by the GAME class.
        """
        if self.GAME is not None:
            return self.GAME.uses_direct_ui()
        return False

    @property
    def uses_preset_file(self) -> bool:
        """Whether this schema writes preset files (FO4-style)."""
        if self.GAME is not None:
            return self.GAME.uses_preset_file()
        return False

    @property
    def has_direct_ui_inputs(self) -> bool:
        """
        Whether this schema has any inputs requiring direct UI clicks.

        Returns True if any input is a CoordinateGrid or ColorPicker.
        """
        return any(isinstance(inp, (CoordinateGrid, ColorPicker)) for inp in self.get_inputs())

    # =========================================================================
    # Calibration Support (Keyboard-based navigation)
    # =========================================================================

    _calibration: Optional["CalibrationData"] = None
    _calibration_loaded: bool = False
    _current_menu: Optional[str] = None
    _current_input_idx: int = 0

    def _get_game_slug(self) -> str:
        """Get the game slug for calibration file paths."""
        if self.GAME:
            name = self.GAME.__name__
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return "unknown"

    def _get_schema_name(self) -> str:
        """Get the registered name of this schema."""
        for name, cls in _schema_registry.items():
            if isinstance(self, cls):
                return name
        return self.__class__.__name__.lower()

    def load_calibration(self) -> Optional["CalibrationData"]:
        """Load calibration data for this schema."""
        if self._calibration_loaded:
            return self._calibration

        try:
            from calibrations import load_calibration
            game_slug = self._get_game_slug()
            schema_name = self._get_schema_name()
            self._calibration = load_calibration(game_slug, schema_name)
            self._calibration_loaded = True
        except ImportError:
            self._calibration_loaded = True

        return self._calibration

    def _press_key(self, key: str, count: int = 1):
        """Press a key multiple times with calibrated delay."""
        if _pyautogui is None:
            return

        cal = self.load_calibration()
        # Use longer delay to ensure game registers each keypress
        delay = cal.timing.key_delay_ms / 1000 if cal else 0.05
        delay = max(delay, 0.1)  # Minimum 100ms between keypresses

        for _ in range(count):
            _pyautogui.press(key)
            time.sleep(delay)

    def _press_enter(self):
        """Press enter/select with calibrated delay. Uses Space for DAI."""
        if _pyautogui is None:
            return

        cal = self.load_calibration()
        delay = cal.timing.enter_delay_ms / 1000 if cal else 0.2
        delay = max(delay, 0.3)  # Minimum 300ms after selection

        # DAI uses Space for selection, not Enter
        print("    [SPACE]", flush=True)
        _pyautogui.press('space')
        time.sleep(delay)

    def _press_escape(self):
        """Press escape with calibrated delay."""
        if _pyautogui is None:
            return

        cal = self.load_calibration()
        delay = cal.timing.escape_delay_ms / 1000 if cal else 0.2
        delay = max(delay, 0.3)  # Minimum 300ms after escape

        print("    [ESC]", flush=True)
        _pyautogui.press('escape')
        time.sleep(delay)

    def navigate_to_menu(self, menu_key: str) -> bool:
        """
        Navigate to a menu using keyboard (UP to top, DOWN to menu, ENTER).

        Args:
            menu_key: The menu key to navigate to

        Returns:
            True if navigation was performed
        """
        cal = self.load_calibration()
        if not cal or menu_key not in cal.menus:
            return False

        if _pyautogui is None:
            return False

        print(f"  Navigating to menu: {menu_key}", flush=True)

        # If we're in a submenu, press ESC to go back to menu list
        if self._current_menu is not None:
            self._press_escape()

        # Press UP many times to ensure we're at the top
        self._press_key('up', 20)
        time.sleep(0.3)

        # Always go down one first to force selection, then navigate to target
        # This ensures the menu system is "active" and not just visually highlighted
        down_count = cal.menus[menu_key].down_count
        print(f"    DOWN x1 (force activate)", flush=True)
        self._press_key('down', 1)
        time.sleep(0.3)

        if down_count == 0:
            # Target is first item, go back up
            print(f"    UP x1 (back to target)", flush=True)
            self._press_key('up', 1)
            time.sleep(0.3)
        elif down_count > 1:
            # Target is further down, continue down
            print(f"    DOWN x{down_count - 1} (to target)", flush=True)
            self._press_key('down', down_count - 1)
            time.sleep(0.3)
        # If down_count == 1, we're already there
        else:
            time.sleep(0.3)

        # Press SPACE to select the menu (this also selects first input)
        self._press_enter()

        self._current_menu = menu_key
        self._current_input_idx = 0
        return True

    def navigate_to_input(self, inp: InputType) -> bool:
        """
        Navigate to a specific input using keyboard.

        Args:
            inp: The input to navigate to

        Returns:
            True if navigation was performed
        """
        cal = self.load_calibration()
        if not cal or inp.name not in cal.inputs:
            return False

        input_info = cal.inputs[inp.name]

        # Navigate to the menu if needed
        if self._current_menu != input_info.menu:
            self.navigate_to_menu(input_info.menu)

        # Calculate how many DOWNs needed from current position
        target_idx = input_info.down_count
        if target_idx > self._current_input_idx:
            self._press_key('down', target_idx - self._current_input_idx)
        elif target_idx < self._current_input_idx:
            self._press_key('up', self._current_input_idx - target_idx)

        self._current_input_idx = target_idx
        return True

    def apply_slider_value(self, inp: Slider, value: int):
        """
        Apply a slider value using keyboard (LEFT to reset, RIGHT to value).

        Args:
            inp: The slider input
            value: The value to set (0-indexed)
        """
        if _pyautogui is None:
            return

        cal = self.load_calibration()
        reset_delay = cal.timing.slider_reset_delay_ms / 1000 if cal else 0.1

        # Press LEFT many times to reset to 0
        self._press_key('left', inp.num_classes())
        time.sleep(reset_delay)

        # Press RIGHT to reach target value
        self._press_key('right', value)

    def apply_grid_value(self, inp: CoordinateGrid, value: int):
        """
        Apply a coordinate grid value using mouse click.

        Args:
            inp: The grid input
            value: The value to set (0-indexed)
        """
        cal = self.load_calibration()
        if not cal or not cal.grid_bounds:
            return

        if _pyautogui is None:
            return

        x, y = cal.grid_bounds.get_click_position(value, inp.x_divisions, inp.y_divisions)
        _pyautogui.click(x, y)
        time.sleep(0.1)

    def apply_color_value(self, inp: ColorPicker, value: int):
        """
        Apply a color picker value using mouse click on the hue ring.

        Args:
            inp: The color picker input
            value: The hue value to set (0-indexed)
        """
        cal = self.load_calibration()
        if not cal or not cal.color_wheel:
            return

        if _pyautogui is None:
            return

        # Click on the hue ring at the calculated angle
        x, y = cal.color_wheel.get_click_position(value, inp.num_classes())
        _pyautogui.click(x, y)
        time.sleep(0.1)

    def apply_calibrated_input(self, inp: InputType, value: int):
        """
        Apply a value to an input using keyboard navigation.

        This method:
        1. Navigates to the input (keyboard)
        2. Applies the value (keyboard for sliders, mouse for grids/colors)

        Args:
            inp: The input to apply
            value: The value index to apply
        """
        # Navigate to the input
        self.navigate_to_input(inp)

        # Apply value based on type
        if isinstance(inp, Slider):
            self.apply_slider_value(inp, value)
        elif isinstance(inp, CoordinateGrid):
            self.apply_grid_value(inp, value)
        elif isinstance(inp, ColorPicker):
            self.apply_color_value(inp, value)

    def apply_all_calibrated_inputs(self, code: str):
        """
        Apply all input values from a code string using keyboard navigation.

        Optimized to process inputs menu by menu to minimize navigation.

        Args:
            code: The encoded character code string
        """
        cal = self.load_calibration()
        if not cal:
            return

        decoded = self.decode_code(code)

        # Group by menu for efficient navigation
        by_menu: Dict[str, List[Tuple[InputType, int]]] = {}
        for inp, value in decoded:
            if inp.name in cal.inputs:
                menu = cal.inputs[inp.name].menu
                if menu not in by_menu:
                    by_menu[menu] = []
                by_menu[menu].append((inp, value))

        # Process each menu
        for menu_key in cal.menus.keys():
            if menu_key not in by_menu:
                continue

            # Navigate to menu
            self.navigate_to_menu(menu_key)

            # Sort inputs by down_count for efficient traversal
            menu_inputs = sorted(by_menu[menu_key], key=lambda x: cal.inputs[x[0].name].down_count)

            for inp, value in menu_inputs:
                self.apply_calibrated_input(inp, value)

    def reset_view_for_capture(self):
        """
        Reset the camera view after applying inputs.

        In DAI, the character zooms out during editing. This method:
        1. Escapes back to main menu
        2. Moves mouse to the character
        3. Scrolls up to zoom in

        Should be called before capturing a screenshot.
        """
        cal = self.load_calibration()
        if not cal or not cal.capture or not cal.capture.reset_view_position:
            return

        if _pyautogui is None:
            return

        # Escape back to main menu
        self._press_escape()

        # Move to reset position and scroll up to zoom in
        x, y = cal.capture.reset_view_position
        _pyautogui.moveTo(x, y)
        time.sleep(0.2)

        # Scroll up to zoom in
        for _ in range(cal.capture.reset_scroll_count):
            _pyautogui.scroll(1)  # Positive = scroll up = zoom in
            time.sleep(0.05)

        time.sleep(0.3)  # Wait for view to settle
