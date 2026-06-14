"""
YAML-based universal macro schema system.

Parses YAML macro files into typed config objects and provides MacroSchema,
a BaseSchema subclass that constructs itself entirely from YAML definitions.

This replaces the need to write Python schema files for each game/gender combo.
The YAML's nested menu structure mirrors the game's actual menu hierarchy.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from input_types import InputType, Slider, CoordinateGrid, ColorPicker
from game import (
    Game, AutomationMode,
    MassEffect1, MassEffect3, DragonAgeInquisition, Fallout4, Cyberpunk2077,
)

# Game name -> Game class lookup
GAME_REGISTRY: Dict[str, type] = {
    "mass_effect_1": MassEffect1,
    "mass_effect_3": MassEffect3,
    "dragon_age_inquisition": DragonAgeInquisition,
    "fallout_4": Fallout4,
    "cyberpunk_2077": Cyberpunk2077,
}


# =========================================================================
# Config Dataclasses
# =========================================================================

@dataclass
class TimingConfig:
    key_press_ms: int = 100
    after_select_ms: int = 300
    after_back_ms: int = 300
    after_slider_reset_ms: int = 100


@dataclass
class InputHandlerConfig:
    """Configuration for how a specific input type is handled."""
    # For slider
    reset: Optional[str] = None
    apply: Optional[str] = None
    encoding: Optional[str] = None  # "legacy" for ME-style 1-indexed
    # For grid/color
    method: Optional[str] = None  # "mouse_click"
    reference: Optional[str] = None  # "grid_bounds", "color_wheel"


@dataclass
class NavigationConfig:
    mode: str = "direct_ui"  # "direct_ui", "code_paste", "preset_file"
    select_key: str = "space"
    back_key: str = "escape"
    menu_reset: str = "UP x20"
    input_reset_key: str = "left"


@dataclass
class CaptureConfig:
    method: str = "screenshot"
    reset_actions: List[str] = field(default_factory=list)
    scroll_up: int = 10


@dataclass
class PasteConfig:
    actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PresetConfig:
    format: str = "looksmenu_json"
    output_dir: str = ""
    filename: str = "cc_{code}.json"


@dataclass
class InputDef:
    """A single input definition from YAML."""
    name: str
    type: str  # "slider", "grid", "color"
    options: int = 0
    x_divisions: int = 0
    y_divisions: int = 0
    palette_size: int = 0
    biased: bool = False


@dataclass
class MenuNode:
    """A menu in the game's hierarchy. Can contain inputs and/or submenus."""
    name: str
    position: int
    inputs: List[InputDef] = field(default_factory=list)
    submenus: Dict[str, "MenuNode"] = field(default_factory=dict)


@dataclass
class MacroConfig:
    """Top-level parsed YAML configuration."""
    version: str
    game: str
    schema: str
    window_keyword: str = ""
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    input_handlers: Dict[str, InputHandlerConfig] = field(default_factory=dict)
    capture: Optional[CaptureConfig] = None
    paste: Optional[PasteConfig] = None
    preset: Optional[PresetConfig] = None
    # For direct_ui games: nested menu hierarchy
    menus: Dict[str, MenuNode] = field(default_factory=dict)
    # For code_paste/preset_file games: flat input list
    flat_inputs: List[InputDef] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MacroConfig":
        """Parse a YAML file into a MacroConfig."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "MacroConfig":
        """Parse a dict (from YAML) into a MacroConfig."""
        # Navigation
        nav_data = data.get("navigation", {})
        navigation = NavigationConfig(
            mode=nav_data.get("mode", "direct_ui"),
            select_key=nav_data.get("select_key", "space"),
            back_key=nav_data.get("back_key", "escape"),
            menu_reset=nav_data.get("menu_reset", "UP x20"),
            input_reset_key=nav_data.get("input_reset_key", "left"),
        )

        # Timing
        timing_data = data.get("timing", {})
        timing = TimingConfig(
            key_press_ms=timing_data.get("key_press_ms", 100),
            after_select_ms=timing_data.get("after_select_ms", 300),
            after_back_ms=timing_data.get("after_back_ms", 300),
            after_slider_reset_ms=timing_data.get("after_slider_reset_ms", 100),
        )

        # Input handlers
        handlers = {}
        for handler_name, handler_data in data.get("input_handlers", {}).items():
            handlers[handler_name] = InputHandlerConfig(
                reset=handler_data.get("reset"),
                apply=handler_data.get("apply"),
                encoding=handler_data.get("encoding"),
                method=handler_data.get("method"),
                reference=handler_data.get("reference"),
            )

        # Capture
        capture = None
        cap_data = data.get("capture")
        if cap_data:
            reset_view = cap_data.get("reset_view", {})
            capture = CaptureConfig(
                method=cap_data.get("method", "screenshot"),
                reset_actions=reset_view.get("actions", []),
                scroll_up=reset_view.get("scroll_up", 10),
            )

        # Paste
        paste = None
        paste_data = data.get("paste")
        if paste_data:
            paste = PasteConfig(actions=paste_data.get("actions", []))

        # Preset
        preset = None
        preset_data = data.get("preset")
        if preset_data:
            preset = PresetConfig(
                format=preset_data.get("format", "looksmenu_json"),
                output_dir=preset_data.get("output_dir", ""),
                filename=preset_data.get("filename", "cc_{code}.json"),
            )

        # Window keyword
        window_data = data.get("window", {})
        window_keyword = window_data.get("keyword", "")

        # Menus (for direct_ui) or flat inputs (for code_paste/preset_file)
        menus = {}
        flat_inputs = []

        if "menus" in data:
            menus = _parse_menus(data["menus"])
        if "inputs" in data:
            flat_inputs = [_parse_input_def(inp) for inp in data["inputs"]]

        return cls(
            version=data.get("version", "1.0"),
            game=data.get("game", ""),
            schema=data.get("schema", ""),
            window_keyword=window_keyword,
            navigation=navigation,
            timing=timing,
            input_handlers=handlers,
            capture=capture,
            paste=paste,
            preset=preset,
            menus=menus,
            flat_inputs=flat_inputs,
        )

    def get_all_input_defs(self) -> List[InputDef]:
        """Return all input definitions in order (menus then flat)."""
        inputs = []
        if self.menus:
            for menu in self.menus.values():
                inputs.extend(_collect_inputs_from_menu(menu))
        inputs.extend(self.flat_inputs)
        return inputs

    def get_menu_categories(self) -> Dict[str, List[str]]:
        """Return menu name -> list of input names, for calibrator/tester compatibility."""
        categories = {}
        if self.menus:
            for menu_name, menu_node in self.menus.items():
                categories[menu_name] = [inp.name for inp in menu_node.inputs]
        return categories


def _parse_input_def(data: dict) -> InputDef:
    """Parse a single input definition from YAML."""
    return InputDef(
        name=data["name"],
        type=data["type"],
        options=data.get("options", 0),
        x_divisions=data.get("x_divisions", 0),
        y_divisions=data.get("y_divisions", 0),
        palette_size=data.get("palette_size", 0),
        biased=data.get("biased", False),
    )


def _parse_menus(menus_data: dict) -> Dict[str, MenuNode]:
    """Parse the menus section of the YAML, handling nested submenus."""
    result = {}
    for menu_name, menu_data in menus_data.items():
        result[menu_name] = _parse_menu_node(menu_name, menu_data)
    return result


def _parse_menu_node(name: str, data: dict) -> MenuNode:
    """Parse a single menu node, recursively handling submenus."""
    inputs = [_parse_input_def(inp) for inp in data.get("inputs", [])]
    submenus = {}
    if "submenus" in data:
        for sub_name, sub_data in data["submenus"].items():
            submenus[sub_name] = _parse_menu_node(sub_name, sub_data)
    return MenuNode(
        name=name,
        position=data.get("position", 0),
        inputs=inputs,
        submenus=submenus,
    )


def _collect_inputs_from_menu(menu: MenuNode) -> List[InputDef]:
    """Recursively collect all inputs from a menu and its submenus."""
    inputs = list(menu.inputs)
    for sub in menu.submenus.values():
        inputs.extend(_collect_inputs_from_menu(sub))
    return inputs


def _input_def_to_input_type(inp_def: InputDef, position: int) -> InputType:
    """Convert an InputDef to the appropriate InputType instance."""
    if inp_def.type == "slider":
        return Slider(
            digit=position,
            name=inp_def.name,
            options=inp_def.options,
            biased=inp_def.biased,
        )
    elif inp_def.type == "grid":
        return CoordinateGrid(
            position=position,
            name=inp_def.name,
            x_divisions=inp_def.x_divisions,
            y_divisions=inp_def.y_divisions,
            biased=inp_def.biased,
        )
    elif inp_def.type == "color":
        return ColorPicker(
            position=position,
            name=inp_def.name,
            palette_size=inp_def.palette_size,
            biased=inp_def.biased,
        )
    else:
        raise ValueError(f"Unknown input type: {inp_def.type}")


# =========================================================================
# MacroSchema: YAML-backed BaseSchema
# =========================================================================

from base_schema import BaseSchema, register_schema, _schema_registry


class MacroSchema(BaseSchema):
    """A BaseSchema constructed entirely from a YAML macro file."""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.config = MacroConfig.from_yaml(yaml_path)
        self.GAME = GAME_REGISTRY.get(self.config.game)
        self._inputs: Optional[List[InputType]] = None

    def get_inputs(self) -> List[InputType]:
        if self._inputs is None:
            self._inputs = []
            pos = 0
            for inp_def in self.config.get_all_input_defs():
                pos += 1
                self._inputs.append(_input_def_to_input_type(inp_def, pos))
        return self._inputs

    def get_code_file(self) -> str:
        game_slug = self.config.game.replace(" ", "_")
        return f"dataset/{game_slug}/{self.config.schema}_codes.txt"

    def get_output_dir(self) -> str:
        game_slug = self.config.game.replace(" ", "_")
        return f"dataset/{game_slug}/{self.config.schema}"

    def setup_window(self):
        if not self.config.window_keyword:
            return None
        try:
            from window_helpers import focus_window, find_window
            game_window = find_window(self.config.window_keyword)
            if not game_window:
                raise RuntimeError(
                    f"Window '{self.config.window_keyword}' not found. Is the game running?"
                )
            focus_window(game_window)
            return game_window
        except ImportError:
            return None

    def send_code_to_game(self, code: str):
        mode = self.config.navigation.mode
        if mode == "code_paste":
            self.setup_window()
            # Delegate to paste logic or calibrated method
            try:
                from window_helpers import add_to_clipboard
                add_to_clipboard(code)
                self.setup_window()
            except ImportError:
                pass
        elif mode == "direct_ui":
            self.setup_window()
            cal = self.load_calibration()
            if cal:
                self.apply_all_calibrated_inputs(code)
            else:
                print("Warning: No calibration data. Run calibrator first.")
        elif mode == "preset_file":
            # Preset file mode: subclasses or game-specific logic handles this
            print(f"Preset file mode not yet implemented for macro schemas.")

    def capture_image(self, code):
        try:
            from PIL import ImageGrab
            cal = self.load_calibration()
            if cal and cal.capture:
                bbox = cal.capture.box
            else:
                bbox = (0, 0, 800, 600)
            return ImageGrab.grab(bbox=bbox)
        except ImportError:
            return None


# =========================================================================
# Auto-discovery and registration of YAML macro schemas
# =========================================================================

_macros_dir = Path(__file__).parent / "macros"
_loaded_macro_schemas: Dict[str, MacroSchema] = {}


def load_macro_schema(yaml_path: str) -> MacroSchema:
    """Load a MacroSchema from a YAML file and register it."""
    schema = MacroSchema(yaml_path)
    name = schema.config.schema
    _loaded_macro_schemas[name] = schema
    return schema


def discover_macros():
    """
    Discover all YAML files in the macros/ directory.

    Returns a dict of schema_name -> yaml_path for all found macros.
    Does NOT auto-register them (to avoid conflicts with existing Python schemas).
    """
    macros = {}
    if _macros_dir.exists():
        for yaml_file in sorted(_macros_dir.glob("*.yaml")):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                schema_name = data.get("schema", yaml_file.stem)
                macros[schema_name] = str(yaml_file)
            except Exception:
                continue
    return macros


def register_macro_schemas(override: bool = False):
    """
    Register all discovered YAML macro schemas in the schema registry.

    Args:
        override: If True, override existing Python schemas with YAML versions.
                  If False (default), skip schemas already registered.
    """
    for schema_name, yaml_path in discover_macros().items():
        if not override and schema_name in _schema_registry:
            continue
        schema = MacroSchema(yaml_path)
        # Register a factory that returns a fresh MacroSchema
        _schema_registry[schema_name] = _make_macro_schema_class(yaml_path, schema_name)
        _loaded_macro_schemas[schema_name] = schema


def get_macro_config(schema_name: str) -> Optional[MacroConfig]:
    """Get the MacroConfig for a schema, loading from YAML if needed."""
    if schema_name in _loaded_macro_schemas:
        return _loaded_macro_schemas[schema_name].config
    macros = discover_macros()
    if schema_name in macros:
        schema = load_macro_schema(macros[schema_name])
        return schema.config
    return None


def _make_macro_schema_class(yaml_path: str, name: str) -> type:
    """Create a MacroSchema subclass that can be instantiated by the registry."""
    class _DynamicMacroSchema(MacroSchema):
        def __init__(self):
            super().__init__(yaml_path)
    _DynamicMacroSchema.__name__ = f"MacroSchema_{name}"
    _DynamicMacroSchema.__qualname__ = f"MacroSchema_{name}"
    return _DynamicMacroSchema
