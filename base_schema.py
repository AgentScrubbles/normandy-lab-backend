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
from game import Game, AutomationMode, MassEffect1, MassEffect3, DragonAgeInquisition

# Calibration imports (lazy loaded to avoid circular imports)
if TYPE_CHECKING:
    from tools.calibration_schema import CalibrationData, InputCalibration

# Conditional import of pyautogui (not available in docker mode)
_pyautogui = None
if os.getenv("APP_MODE") != "docker":
    try:
        import pyautogui
        _pyautogui = pyautogui
    except ImportError:
        pass

# Re-export for backward compatibility and convenience
__all__ = [
    # Schema classes
    'BaseSchema', 'register_schema', 'get_schema', 'all_schemas',
    # Input types
    'Slider', 'InputType', 'CoordinateGrid', 'ColorPicker', 'AnyInputType',
    # Game classes
    'Game', 'AutomationMode', 'MassEffect1', 'MassEffect3', 'DragonAgeInquisition',
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
    def has_direct_ui_inputs(self) -> bool:
        """
        Whether this schema has any inputs requiring direct UI clicks.

        Returns True if any input is a CoordinateGrid or ColorPicker.
        """
        return any(isinstance(inp, (CoordinateGrid, ColorPicker)) for inp in self.get_inputs())

    # =========================================================================
    # Calibration Support
    # =========================================================================

    _calibration: Optional["CalibrationData"] = None
    _calibration_loaded: bool = False

    def _get_game_slug(self) -> str:
        """Get the game slug for calibration file paths."""
        if self.GAME:
            # Convert class name to snake_case
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
        """
        Load calibration data for this schema.

        Returns:
            CalibrationData if file exists, None otherwise
        """
        if self._calibration_loaded:
            return self._calibration

        try:
            from calibrations import load_calibration
            game_slug = self._get_game_slug()
            schema_name = self._get_schema_name()
            self._calibration = load_calibration(game_slug, schema_name)
            self._calibration_loaded = True
        except ImportError:
            # Calibration module not available
            self._calibration_loaded = True

        return self._calibration

    def get_calibrated_input(self, inp: InputType) -> InputType:
        """
        Return input with calibration coordinates applied.

        If calibration exists for this input, returns a copy with updated
        screen coordinates. Otherwise returns the original input unchanged.

        Args:
            inp: The input to calibrate

        Returns:
            Input with calibrated coordinates (or original if no calibration)
        """
        cal = self.load_calibration()
        if not cal or inp.name not in cal.inputs:
            return inp

        input_cal = cal.inputs[inp.name]

        if isinstance(inp, CoordinateGrid):
            if hasattr(input_cal.control, 'bounds'):
                return inp.with_bounds(input_cal.control.bounds)
        elif isinstance(inp, ColorPicker):
            if hasattr(input_cal.control, 'wheel_center'):
                return inp.with_wheel(
                    input_cal.control.wheel_center,
                    input_cal.control.wheel_radius
                )

        return inp

    def get_activation_click(self, inp: InputType) -> Optional[Tuple[int, int]]:
        """
        Get the activation click position for an input.

        The activation click is used to select/highlight an input row
        before interacting with its control.

        Args:
            inp: The input to get activation for

        Returns:
            (x, y) tuple or None if no calibration
        """
        cal = self.load_calibration()
        if not cal or inp.name not in cal.inputs:
            return None
        return cal.inputs[inp.name].activation_click

    def get_input_menu(self, inp: InputType) -> Optional[str]:
        """Get the menu key that contains this input."""
        cal = self.load_calibration()
        if not cal or inp.name not in cal.inputs:
            return None
        return cal.inputs[inp.name].menu

    _current_menu: Optional[str] = None  # Track current menu for navigation

    def navigate_to_menu(self, menu_key: str, click_back: bool = True) -> bool:
        """
        Navigate to a menu using calibrated click path.

        Args:
            menu_key: The menu key to navigate to
            click_back: If True, click the back button first to return to main menu

        Returns:
            True if navigation was performed, False otherwise
        """
        cal = self.load_calibration()
        if not cal or not cal.navigation:
            return False

        # Skip if already at target menu
        if menu_key == self._current_menu:
            return True

        menu_nav = cal.navigation.menus.get(menu_key)
        if not menu_nav or not menu_nav.path:
            self._current_menu = menu_key
            return False

        if _pyautogui is None:
            return False

        # Click back button first to return to main menu (if we're in a submenu)
        if click_back and self._current_menu is not None and cal.navigation.back_button:
            back = cal.navigation.back_button
            _pyautogui.click(back[0], back[1])
            time.sleep(cal.navigation.back_wait_ms / 1000)

        # Navigate to target menu
        for x, y in menu_nav.path:
            _pyautogui.click(x, y)
            time.sleep(menu_nav.wait_after_ms / 1000)

        self._current_menu = menu_key
        return True

    def apply_calibrated_input(self, inp: InputType, value: int, click_delay: float = 0.1):
        """
        Apply a value to an input using calibrated coordinates.

        This method:
        1. Clicks the activation position (if calibrated)
        2. Calculates and clicks the control position

        Args:
            inp: The input to apply
            value: The value index to apply
            click_delay: Delay after clicking
        """
        if _pyautogui is None:
            return

        # Get calibrated version of input
        calibrated_inp = self.get_calibrated_input(inp)

        # Click activation position if available
        activation = self.get_activation_click(inp)
        if activation:
            _pyautogui.click(activation[0], activation[1])
            time.sleep(click_delay)

        # Apply the value based on input type
        if isinstance(calibrated_inp, (CoordinateGrid, ColorPicker)):
            x, y = calibrated_inp.get_click_coordinates(value)
            _pyautogui.click(x, y)
            time.sleep(click_delay)
        elif isinstance(calibrated_inp, Slider):
            # For calibrated sliders, use track interpolation
            self._apply_calibrated_slider(inp, value, click_delay)

    def _apply_calibrated_slider(self, inp: Slider, value: int, click_delay: float):
        """Apply a slider value using calibrated track endpoints."""
        cal = self.load_calibration()
        if not cal or inp.name not in cal.inputs:
            return

        input_cal = cal.inputs[inp.name]
        if not hasattr(input_cal.control, 'track_start'):
            return

        # Use SliderControl's get_click_position method
        x, y = input_cal.control.get_click_position(value, inp.num_classes())

        if _pyautogui is not None:
            _pyautogui.click(x, y)
            time.sleep(click_delay)
