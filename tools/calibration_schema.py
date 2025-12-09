"""
Calibration data schema for game character creator UI automation.

This module defines dataclasses for storing and loading calibration data
that maps game UI elements to screen coordinates.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Union
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
    face_center: Optional[Tuple[int, int]] = None  # Optional center point

    def to_dict(self) -> dict:
        d = {'box': list(self.box)}
        if self.face_center:
            d['face_center'] = list(self.face_center)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'CaptureInfo':
        return cls(
            box=tuple(data['box']),
            face_center=tuple(data['face_center']) if data.get('face_center') else None,
        )


@dataclass
class MenuNavigation:
    """Navigation path to reach a menu/submenu."""
    path: List[Tuple[int, int]]  # Sequence of clicks to reach this menu
    wait_after_ms: int = 200     # Wait time after clicking (for UI transitions)

    def to_dict(self) -> dict:
        return {
            'path': [list(p) for p in self.path],
            'wait_after_ms': self.wait_after_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MenuNavigation':
        return cls(
            path=[tuple(p) for p in data['path']],
            wait_after_ms=data.get('wait_after_ms', 200),
        )


@dataclass
class NavigationInfo:
    """All menu navigation paths."""
    menus: Dict[str, MenuNavigation]
    back_button: Optional[Tuple[int, int]] = None  # Button to return to main menu
    back_wait_ms: int = 300  # Wait time after clicking back
    confirm_button: Optional[Tuple[int, int]] = None
    cancel_button: Optional[Tuple[int, int]] = None

    def to_dict(self) -> dict:
        d = {
            'menus': {k: v.to_dict() for k, v in self.menus.items()},
        }
        if self.back_button:
            d['back_button'] = list(self.back_button)
            d['back_wait_ms'] = self.back_wait_ms
        if self.confirm_button:
            d['confirm_button'] = list(self.confirm_button)
        if self.cancel_button:
            d['cancel_button'] = list(self.cancel_button)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'NavigationInfo':
        return cls(
            menus={k: MenuNavigation.from_dict(v) for k, v in data['menus'].items()},
            back_button=tuple(data['back_button']) if data.get('back_button') else None,
            back_wait_ms=data.get('back_wait_ms', 300),
            confirm_button=tuple(data['confirm_button']) if data.get('confirm_button') else None,
            cancel_button=tuple(data['cancel_button']) if data.get('cancel_button') else None,
        )


@dataclass
class SliderControl:
    """Control data for slider-type inputs."""
    track_start: Tuple[int, int]  # Left/start of slider track
    track_end: Tuple[int, int]    # Right/end of slider track

    def to_dict(self) -> dict:
        return {
            'track_start': list(self.track_start),
            'track_end': list(self.track_end),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SliderControl':
        return cls(
            track_start=tuple(data['track_start']),
            track_end=tuple(data['track_end']),
        )

    def get_click_position(self, value: int, num_options: int) -> Tuple[int, int]:
        """Calculate click position for a given value."""
        if num_options <= 1:
            t = 0.5
        else:
            t = value / (num_options - 1)

        x = int(self.track_start[0] + t * (self.track_end[0] - self.track_start[0]))
        y = int(self.track_start[1] + t * (self.track_end[1] - self.track_start[1]))
        return (x, y)


@dataclass
class GridControl:
    """Control data for coordinate grid inputs."""
    bounds: Tuple[int, int, int, int]  # x1, y1, x2, y2

    def to_dict(self) -> dict:
        return {'bounds': list(self.bounds)}

    @classmethod
    def from_dict(cls, data: dict) -> 'GridControl':
        return cls(bounds=tuple(data['bounds']))


@dataclass
class ColorPickerControl:
    """Control data for color picker/wheel inputs."""
    wheel_center: Tuple[int, int]
    wheel_radius: int

    def to_dict(self) -> dict:
        return {
            'wheel_center': list(self.wheel_center),
            'wheel_radius': self.wheel_radius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ColorPickerControl':
        return cls(
            wheel_center=tuple(data['wheel_center']),
            wheel_radius=data['wheel_radius'],
        )


ControlType = Union[SliderControl, GridControl, ColorPickerControl]


@dataclass
class InputCalibration:
    """Calibration data for a single input."""
    type: Literal['slider', 'coordinate_grid', 'color_picker']
    activation_click: Tuple[int, int]  # Click to select/activate this input
    control: ControlType               # Type-specific control data
    menu: str                          # Menu key this input belongs to

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'activation_click': list(self.activation_click),
            'control': self.control.to_dict(),
            'menu': self.menu,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InputCalibration':
        input_type = data['type']
        if input_type == 'slider':
            control = SliderControl.from_dict(data['control'])
        elif input_type == 'coordinate_grid':
            control = GridControl.from_dict(data['control'])
        elif input_type == 'color_picker':
            control = ColorPickerControl.from_dict(data['control'])
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        return cls(
            type=input_type,
            activation_click=tuple(data['activation_click']),
            control=control,
            menu=data['menu'],
        )


@dataclass
class CalibrationLogEntry:
    """Log entry for when an input was calibrated."""
    input: str
    timestamp: str
    method: str = 'interactive'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationLogEntry':
        return cls(**data)


@dataclass
class CalibrationData:
    """
    Complete calibration data for a schema.

    This is the top-level container that gets serialized to/from JSON.
    """
    version: str = '1.0'
    schema_name: str = ''
    game: str = ''
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolution: Optional[Tuple[int, int]] = None
    window: Optional[WindowInfo] = None
    capture: Optional[CaptureInfo] = None
    navigation: Optional[NavigationInfo] = None
    inputs: Dict[str, InputCalibration] = field(default_factory=dict)
    calibration_log: List[CalibrationLogEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'version': self.version,
            'schema_name': self.schema_name,
            'game': self.game,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

        if self.resolution:
            d['resolution'] = list(self.resolution)
        if self.window:
            d['window'] = self.window.to_dict()
        if self.capture:
            d['capture'] = self.capture.to_dict()
        if self.navigation:
            d['navigation'] = self.navigation.to_dict()
        if self.inputs:
            d['inputs'] = {k: v.to_dict() for k, v in self.inputs.items()}
        if self.calibration_log:
            d['calibration_log'] = [e.to_dict() for e in self.calibration_log]

        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        """Create from dictionary (deserialized JSON)."""
        return cls(
            version=data.get('version', '1.0'),
            schema_name=data.get('schema_name', ''),
            game=data.get('game', ''),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            resolution=tuple(data['resolution']) if data.get('resolution') else None,
            window=WindowInfo.from_dict(data['window']) if data.get('window') else None,
            capture=CaptureInfo.from_dict(data['capture']) if data.get('capture') else None,
            navigation=NavigationInfo.from_dict(data['navigation']) if data.get('navigation') else None,
            inputs={k: InputCalibration.from_dict(v) for k, v in data.get('inputs', {}).items()},
            calibration_log=[CalibrationLogEntry.from_dict(e) for e in data.get('calibration_log', [])],
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

    def get_input(self, name: str) -> Optional[InputCalibration]:
        """Get calibration for a specific input."""
        return self.inputs.get(name)

    def set_input(self, name: str, calibration: InputCalibration) -> None:
        """Set/update calibration for an input."""
        self.inputs[name] = calibration
        self.calibration_log.append(CalibrationLogEntry(
            input=name,
            timestamp=datetime.now().isoformat(),
            method='interactive',
        ))

    def get_uncalibrated_inputs(self, input_names: List[str]) -> List[str]:
        """Return list of input names that haven't been calibrated."""
        return [name for name in input_names if name not in self.inputs]

    def is_complete(self, input_names: List[str]) -> bool:
        """Check if all inputs have been calibrated."""
        return all(name in self.inputs for name in input_names)

    def get_progress(self, input_names: List[str]) -> Tuple[int, int]:
        """Return (calibrated_count, total_count)."""
        calibrated = sum(1 for name in input_names if name in self.inputs)
        return (calibrated, len(input_names))
