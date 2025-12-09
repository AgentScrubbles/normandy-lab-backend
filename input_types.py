"""
Input type hierarchy for character creator schemas.

This module defines the input types that can be used in character
creator schemas. Each input type knows how to encode/decode its values and
calculate click coordinates for game automation.

Supported types:
- Slider: Single-dimension discrete slider (backward compatible with existing schemas)
- CoordinateGrid: 2D X,Y picker discretized to a grid (Stage 3)
- ColorPicker: Color wheel discretized to a palette (Stage 3)

Design note: Slider maintains the original constructor signature for backward
compatibility. New input types follow a different pattern but implement the
same interface methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Protocol, runtime_checkable
import string


# Extended alphabet for encoding: 0-9, A-Z, a-z = 62 characters
# Legacy (ME): 1-indexed using '1'-'9', 'A'-'Z' (35 chars max)
# Extended: 0-indexed using '0'-'9', 'A'-'Z', 'a'-'z' (62 chars, or 3844 with 2-char)
ALPHABET = list(string.digits + string.ascii_uppercase)
EXTENDED_ALPHABET = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

# Reverse lookup for extended alphabet
EXTENDED_CHAR_TO_IDX = {ch: i for i, ch in enumerate(EXTENDED_ALPHABET)}


def extended_encode(value: int, num_classes: int) -> str:
    """
    Encode a value using the extended alphabet (0-indexed).

    - For num_classes <= 62: single character encoding
    - For num_classes <= 3844 (62²): two character encoding

    Args:
        value: The value to encode (0-indexed)
        num_classes: Total number of classes for this input

    Returns:
        Encoded string (1 or 2 characters)
    """
    if value < 0 or value >= num_classes:
        raise ValueError(f"Value {value} out of range [0, {num_classes})")

    if num_classes <= 62:
        return EXTENDED_ALPHABET[value]
    elif num_classes <= 62 * 62:
        # 2-char encoding: high digit first, then low digit
        high = value // 62
        low = value % 62
        return EXTENDED_ALPHABET[high] + EXTENDED_ALPHABET[low]
    else:
        raise ValueError(f"num_classes {num_classes} exceeds maximum of {62 * 62}")


def extended_decode(encoded: str, num_classes: int) -> int:
    """
    Decode a string back to a value using the extended alphabet.

    Args:
        encoded: The encoded string (1 or 2 characters)
        num_classes: Total number of classes for this input

    Returns:
        Decoded value (0-indexed)
    """
    if num_classes <= 62:
        if len(encoded) != 1:
            raise ValueError(f"Expected single character for {num_classes} classes, got '{encoded}'")
        if encoded not in EXTENDED_CHAR_TO_IDX:
            raise ValueError(f"Invalid character '{encoded}'")
        return EXTENDED_CHAR_TO_IDX[encoded]
    elif num_classes <= 62 * 62:
        if len(encoded) != 2:
            raise ValueError(f"Expected 2 characters for {num_classes} classes, got '{encoded}'")
        high = EXTENDED_CHAR_TO_IDX.get(encoded[0])
        low = EXTENDED_CHAR_TO_IDX.get(encoded[1])
        if high is None or low is None:
            raise ValueError(f"Invalid characters in '{encoded}'")
        return high * 62 + low
    else:
        raise ValueError(f"num_classes {num_classes} exceeds maximum of {62 * 62}")


def extended_encoding_width(num_classes: int) -> int:
    """Return the number of characters needed to encode num_classes options."""
    if num_classes <= 62:
        return 1
    elif num_classes <= 62 * 62:
        return 2
    else:
        raise ValueError(f"num_classes {num_classes} exceeds maximum of {62 * 62}")


def extended_symbol_list(num_classes: int) -> List[str]:
    """Return list of all valid encoded symbols for num_classes options."""
    return [extended_encode(i, num_classes) for i in range(num_classes)]


@runtime_checkable
class InputType(Protocol):
    """
    Protocol defining the interface for all input types.

    All input types must implement these methods/properties.
    Using Protocol allows duck typing - Slider can maintain its
    original field order while still being recognized as an InputType.
    """
    name: str
    biased: bool

    def num_classes(self) -> int:
        """Return the number of discrete classes/options for this input."""
        ...

    def encode(self, value: int) -> str:
        """Encode a class index (0-indexed) to a string representation."""
        ...

    def decode(self, encoded: str) -> int:
        """Decode a string representation back to a class index."""
        ...

    @property
    def encoding_width(self) -> int:
        """Number of characters this input uses in the code string."""
        ...


@dataclass
class Slider:
    """
    A single-dimension slider with N discrete options.

    This is the basic input type, backward-compatible with the original
    Mass Effect schema system. Each slider maps to a single character
    in the code string.

    Constructor signature matches original: Slider(digit, name, options, biased=False)

    Encoding uses characters 1-9, then A-Z for options 10+.
    Note: We use 1-indexed characters (1-9, not 0-9) for backward
    compatibility with existing Mass Effect codes.

    Example:
        Slider with 9 options: encodes as '1' through '9'
        Slider with 30 options: encodes as '1'-'9', 'A'-'U'
    """
    # Field order matches original Slider for backward compatibility
    digit: int           # Position in code (kept as 'digit' for backward compat)
    name: str            # Human-readable name
    options: int         # Number of discrete options (1 to 36 for single-char)
    biased: bool = False # Whether to bias sampling toward this input

    @property
    def position(self) -> int:
        """Alias for digit (new InputType interface)."""
        return self.digit

    def num_classes(self) -> int:
        """Return the number of discrete options for this slider."""
        return self.options

    @property
    def encoding_width(self) -> int:
        """Single character for options <= 36 (current limit)."""
        return 1

    def encode(self, value: int) -> str:
        """
        Encode a value (0-indexed) to a single character.

        Mass Effect codes use 1-indexed characters:
        - Values 0-8 encode as '1'-'9'
        - Values 9+ encode as 'A', 'B', 'C', ...
        """
        if value < 0 or value >= self.options:
            raise ValueError(f"Value {value} out of range [0, {self.options})")

        if value < 9:
            return str(value + 1)  # 0 -> '1', 1 -> '2', ..., 8 -> '9'
        else:
            # 9 -> 'A', 10 -> 'B', etc.
            return chr(ord('A') + (value - 9))

    def decode(self, encoded: str) -> int:
        """
        Decode a single character to a value (0-indexed).
        """
        if len(encoded) != 1:
            raise ValueError(f"Expected single character, got '{encoded}'")

        ch = encoded.upper()
        if ch.isdigit():
            # '1' -> 0, '2' -> 1, ..., '9' -> 8
            return int(ch) - 1
        else:
            # 'A' -> 9, 'B' -> 10, etc.
            return ord(ch) - ord('A') + 9

    def get_symbol_list(self) -> List[str]:
        """
        Return the list of valid symbols for this slider.
        Used for building CHAR2IDX/IDX2CHAR mappings.
        """
        symbols = []
        for i in range(self.options):
            symbols.append(self.encode(i))
        return symbols


# ---------------------------------------------------------------------------
# New input types for Stage 3 - placeholders with full interface
# ---------------------------------------------------------------------------

@dataclass
class CoordinateGrid:
    """
    A 2D coordinate picker discretized to an X*Y grid.

    Used for inputs like "nose width/height" where the user selects
    a point on a 2D grid. The grid is discretized and each cell
    maps to a class index.

    Uses extended encoding (0-indexed, supports up to 3844 options with 2-char).
    """
    position: int        # Position in code
    name: str            # Human-readable name
    x_divisions: int     # Number of divisions along X axis
    y_divisions: int     # Number of divisions along Y axis
    grid_bounds: Tuple[int, int, int, int] = (0, 0, 100, 100)  # screen coords (x1, y1, x2, y2)
    biased: bool = False

    def num_classes(self) -> int:
        return self.x_divisions * self.y_divisions

    @property
    def encoding_width(self) -> int:
        """Returns 1 for <=62 options, 2 for <=3844 options."""
        return extended_encoding_width(self.num_classes())

    def encode(self, value: int) -> str:
        """Encode using extended alphabet (0-indexed, supports multi-char)."""
        return extended_encode(value, self.num_classes())

    def decode(self, encoded: str) -> int:
        """Decode using extended alphabet."""
        return extended_decode(encoded, self.num_classes())

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (x, y) grid coordinates."""
        x = idx % self.x_divisions
        y = idx // self.x_divisions
        return (x, y)

    def xy_to_index(self, x: int, y: int) -> int:
        """Convert (x, y) grid coordinates to flat index."""
        return y * self.x_divisions + x

    def get_click_coordinates(self, idx: int) -> Tuple[int, int]:
        """Get screen coordinates for clicking on this grid cell."""
        x_grid, y_grid = self.index_to_xy(idx)
        x1, y1, x2, y2 = self.grid_bounds

        cell_width = (x2 - x1) / self.x_divisions
        cell_height = (y2 - y1) / self.y_divisions

        # Click center of cell
        screen_x = int(x1 + (x_grid + 0.5) * cell_width)
        screen_y = int(y1 + (y_grid + 0.5) * cell_height)
        return (screen_x, screen_y)

    def get_symbol_list(self) -> List[str]:
        """Return the list of valid symbols for this input."""
        return extended_symbol_list(self.num_classes())

    def with_bounds(self, bounds: Tuple[int, int, int, int]) -> "CoordinateGrid":
        """Return a copy with updated grid bounds from calibration."""
        return CoordinateGrid(
            position=self.position,
            name=self.name,
            x_divisions=self.x_divisions,
            y_divisions=self.y_divisions,
            grid_bounds=bounds,
            biased=self.biased,
        )

    def update_bounds(self, bounds: Tuple[int, int, int, int]) -> None:
        """Update grid bounds in place from calibration data."""
        # Note: dataclass is not frozen, so we can mutate
        object.__setattr__(self, 'grid_bounds', bounds)


@dataclass
class ColorPicker:
    """
    A color wheel/picker discretized to a palette.

    Used for inputs like "skin color" or "hair color" where the user
    selects from a color wheel or palette. Colors are discretized
    using HSL representation for perceptually meaningful sampling.

    Uses extended encoding (0-indexed, supports up to 3844 options with 2-char).
    """
    position: int        # Position in code
    name: str            # Human-readable name
    palette_size: int    # Total number of colors in palette
    wheel_center: Tuple[int, int] = (0, 0)   # Screen coordinates
    wheel_radius: int = 100
    hue_divisions: int = 12       # 30-degree increments
    saturation_levels: int = 3    # e.g., 33%, 66%, 100%
    lightness_levels: int = 1     # Usually fixed at 50%
    biased: bool = False

    def num_classes(self) -> int:
        return self.palette_size

    @property
    def encoding_width(self) -> int:
        """Returns 1 for <=62 options, 2 for <=3844 options."""
        return extended_encoding_width(self.num_classes())

    def encode(self, value: int) -> str:
        """Encode using extended alphabet (0-indexed, supports multi-char)."""
        return extended_encode(value, self.num_classes())

    def decode(self, encoded: str) -> int:
        """Decode using extended alphabet."""
        return extended_decode(encoded, self.num_classes())

    def index_to_hsl(self, idx: int) -> Tuple[float, float, float]:
        """Convert palette index to HSL values (0-1 range each)."""
        import math
        # Decompose index into H, S, L components
        h_idx = idx % self.hue_divisions
        remaining = idx // self.hue_divisions
        s_idx = remaining % self.saturation_levels
        l_idx = remaining // self.saturation_levels

        h = h_idx / self.hue_divisions  # 0-1
        s = (s_idx + 1) / self.saturation_levels  # Avoid 0 saturation
        l = 0.5 if self.lightness_levels == 1 else (0.25 + l_idx * 0.25)
        return (h, s, l)

    def get_click_coordinates(self, idx: int) -> Tuple[int, int]:
        """Get screen coordinates for clicking on color wheel."""
        import math
        h, s, _ = self.index_to_hsl(idx)

        # Convert to polar coordinates on wheel
        angle = h * 2 * math.pi
        radius = s * self.wheel_radius

        x = int(self.wheel_center[0] + radius * math.cos(angle))
        y = int(self.wheel_center[1] + radius * math.sin(angle))
        return (x, y)

    def get_symbol_list(self) -> List[str]:
        """Return the list of valid symbols for this input."""
        return extended_symbol_list(self.num_classes())

    def with_wheel(self, center: Tuple[int, int], radius: int) -> "ColorPicker":
        """Return a copy with updated wheel geometry from calibration."""
        return ColorPicker(
            position=self.position,
            name=self.name,
            palette_size=self.palette_size,
            wheel_center=center,
            wheel_radius=radius,
            hue_divisions=self.hue_divisions,
            saturation_levels=self.saturation_levels,
            lightness_levels=self.lightness_levels,
            biased=self.biased,
        )

    def update_wheel(self, center: Tuple[int, int], radius: int) -> None:
        """Update wheel geometry in place from calibration data."""
        object.__setattr__(self, 'wheel_center', center)
        object.__setattr__(self, 'wheel_radius', radius)


# Type alias for any input type
AnyInputType = Slider | CoordinateGrid | ColorPicker
