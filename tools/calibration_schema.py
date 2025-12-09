"""
Calibration data schema for game character creator UI automation.

This module defines dataclasses for storing calibration data using
keyboard-based navigation (arrow keys + enter) with mouse only for
coordinate grids and color pickers.

Navigation Model:
- Menus are selected by DOWN arrow count from top, then ENTER
- Inputs within menus are selected by DOWN arrow count from first input
- Sliders are controlled by LEFT (reset to 0) then RIGHT (to value)
- Grids and color pickers require mouse clicks at calibrated positions
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class WindowInfo:
    """Window positioning information."""
    reference_position: Tuple[int, int]  # Where to position the window
    expected_size: Tuple[int, int]       # Expected window dimensions

    def to_dict(self) -> dict:
        return {
            'reference_position': list(self.reference_position),
            'expected_size': list(self.expected_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WindowInfo':
        return cls(
            reference_position=tuple(data['reference_position']),
            expected_size=tuple(data['expected_size']),
        )


@dataclass
class CaptureInfo:
    """Screenshot capture region configuration."""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 bounding box
    reset_view_position: Optional[Tuple[int, int]] = None  # Where to hover for zoom reset
    reset_scroll_count: int = 10  # How many scroll-ups to zoom in

    def to_dict(self) -> dict:
        d = {'box': list(self.box)}
        if self.reset_view_position:
            d['reset_view_position'] = list(self.reset_view_position)
        d['reset_scroll_count'] = self.reset_scroll_count
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'CaptureInfo':
        return cls(
            box=tuple(data['box']),
            reset_view_position=tuple(data['reset_view_position']) if data.get('reset_view_position') else None,
            reset_scroll_count=data.get('reset_scroll_count', 10),
        )


@dataclass
class KeyboardTiming:
    """Timing configuration for keyboard inputs."""
    key_delay_ms: int = 50        # Delay between key presses
    enter_delay_ms: int = 200     # Delay after ENTER (menu transitions)
    escape_delay_ms: int = 200    # Delay after ESC (back to menu)
    slider_reset_delay_ms: int = 100  # Delay after resetting slider to 0

    def to_dict(self) -> dict:
        return {
            'key_delay_ms': self.key_delay_ms,
            'enter_delay_ms': self.enter_delay_ms,
            'escape_delay_ms': self.escape_delay_ms,
            'slider_reset_delay_ms': self.slider_reset_delay_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KeyboardTiming':
        return cls(
            key_delay_ms=data.get('key_delay_ms', 50),
            enter_delay_ms=data.get('enter_delay_ms', 200),
            escape_delay_ms=data.get('escape_delay_ms', 200),
            slider_reset_delay_ms=data.get('slider_reset_delay_ms', 100),
        )


@dataclass
class MenuInfo:
    """Keyboard navigation info for a menu category."""
    down_count: int  # Number of DOWN presses from top to reach this menu

    def to_dict(self) -> dict:
        return {'down_count': self.down_count}

    @classmethod
    def from_dict(cls, data: dict) -> 'MenuInfo':
        return cls(down_count=data['down_count'])


@dataclass
class InputInfo:
    """Keyboard navigation info for an input within a menu."""
    down_count: int  # Number of DOWN presses from first input in menu
    menu: str        # Which menu this input belongs to

    def to_dict(self) -> dict:
        return {
            'down_count': self.down_count,
            'menu': self.menu,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InputInfo':
        return cls(
            down_count=data['down_count'],
            menu=data['menu'],
        )


@dataclass
class GridBounds:
    """Screen coordinates for coordinate grid inputs (shared by all grids)."""
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}

    @classmethod
    def from_dict(cls, data: dict) -> 'GridBounds':
        return cls(x1=data['x1'], y1=data['y1'], x2=data['x2'], y2=data['y2'])

    def get_click_position(self, value: int, x_divisions: int, y_divisions: int) -> Tuple[int, int]:
        """Calculate click position for a grid value (0-indexed)."""
        # Value is encoded as row * x_divisions + col
        col = value % x_divisions
        row = value // x_divisions

        # Calculate position within bounds
        # Add 0.5 to click center of cell
        x = self.x1 + int((col + 0.5) * (self.x2 - self.x1) / x_divisions)
        y = self.y1 + int((row + 0.5) * (self.y2 - self.y1) / y_divisions)
        return (x, y)


@dataclass
class ColorWheelInfo:
    """
    Screen coordinates for color picker hue ring.

    DAI uses a hue ring with a saturation/brightness square inside.
    We only care about the outer hue ring for color selection.
    """
    center_x: int
    center_y: int
    ring_radius: int  # Radius to the middle of the hue ring

    def to_dict(self) -> dict:
        return {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'ring_radius': self.ring_radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ColorWheelInfo':
        # Support both old 'radius' and new 'ring_radius' field names
        radius = data.get('ring_radius', data.get('radius', 50))
        return cls(
            center_x=data['center_x'],
            center_y=data['center_y'],
            ring_radius=radius,
        )

    def get_click_position(self, value: int, num_hues: int) -> Tuple[int, int]:
        """
        Calculate click position on the hue ring.

        Args:
            value: Hue index (0 to num_hues-1)
            num_hues: Total number of hue divisions

        Returns:
            (x, y) screen coordinates on the hue ring
        """
        import math

        # Angle for hue (0 = right/blue, going counter-clockwise)
        # Offset by -90 degrees so 0 = top if desired, adjust as needed
        angle = 2 * math.pi * value / num_hues

        x = self.center_x + int(self.ring_radius * math.cos(angle))
        y = self.center_y - int(self.ring_radius * math.sin(angle))  # Y inverted
        return (x, y)


@dataclass
class CalibrationData:
    """
    Complete calibration data for a schema using keyboard navigation.

    This is the top-level container that gets serialized to/from JSON.
    """
    version: str = '2.0'  # Version 2.0 = keyboard-based navigation
    schema_name: str = ''
    game: str = ''
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Window and capture
    resolution: Optional[Tuple[int, int]] = None
    window: Optional[WindowInfo] = None
    capture: Optional[CaptureInfo] = None

    # Keyboard timing
    timing: KeyboardTiming = field(default_factory=KeyboardTiming)

    # Menu navigation (menu_name -> down_count)
    menus: Dict[str, MenuInfo] = field(default_factory=dict)

    # Input navigation (input_name -> InputInfo)
    inputs: Dict[str, InputInfo] = field(default_factory=dict)

    # Shared control positions (only need one of each)
    grid_bounds: Optional[GridBounds] = None
    color_wheel: Optional[ColorWheelInfo] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'version': self.version,
            'schema_name': self.schema_name,
            'game': self.game,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'timing': self.timing.to_dict(),
        }

        if self.resolution:
            d['resolution'] = list(self.resolution)
        if self.window:
            d['window'] = self.window.to_dict()
        if self.capture:
            d['capture'] = self.capture.to_dict()
        if self.menus:
            d['menus'] = {k: v.to_dict() for k, v in self.menus.items()}
        if self.inputs:
            d['inputs'] = {k: v.to_dict() for k, v in self.inputs.items()}
        if self.grid_bounds:
            d['grid_bounds'] = self.grid_bounds.to_dict()
        if self.color_wheel:
            d['color_wheel'] = self.color_wheel.to_dict()

        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        """Create from dictionary (deserialized JSON)."""
        return cls(
            version=data.get('version', '2.0'),
            schema_name=data.get('schema_name', ''),
            game=data.get('game', ''),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            resolution=tuple(data['resolution']) if data.get('resolution') else None,
            window=WindowInfo.from_dict(data['window']) if data.get('window') else None,
            capture=CaptureInfo.from_dict(data['capture']) if data.get('capture') else None,
            timing=KeyboardTiming.from_dict(data['timing']) if data.get('timing') else KeyboardTiming(),
            menus={k: MenuInfo.from_dict(v) for k, v in data.get('menus', {}).items()},
            inputs={k: InputInfo.from_dict(v) for k, v in data.get('inputs', {}).items()},
            grid_bounds=GridBounds.from_dict(data['grid_bounds']) if data.get('grid_bounds') else None,
            color_wheel=ColorWheelInfo.from_dict(data['color_wheel']) if data.get('color_wheel') else None,
        )

    @classmethod
    def load(cls, path: Path) -> 'CalibrationData':
        """Load calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Save calibration to JSON file."""
        self.updated_at = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_menu_count(self) -> int:
        """Return number of calibrated menus."""
        return len(self.menus)

    def get_input_count(self) -> int:
        """Return number of calibrated inputs."""
        return len(self.inputs)

    def is_complete(self, menu_names: List[str], input_names: List[str],
                    has_grids: bool, has_colors: bool) -> bool:
        """Check if calibration is complete for the given schema."""
        if not all(m in self.menus for m in menu_names):
            return False
        if not all(i in self.inputs for i in input_names):
            return False
        if has_grids and not self.grid_bounds:
            return False
        if has_colors and not self.color_wheel:
            return False
        return True
