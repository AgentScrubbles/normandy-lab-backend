# Multi-Game Character Creator Migration Plan

## Overview

This document outlines the migration plan to extend the AI character creator from Mass Effect-only support to a multi-game architecture capable of handling Dragon Age Inquisition and future games with complex input types.

**Current State:** Mass Effect 1 & 3 (male/female variants) with simple slider-based inputs
**Target State:** Extensible architecture supporting multiple games with sliders, color pickers, 2D coordinate grids, and 100+ option inputs

---

## Current Progress

| Stage | Status | Description |
|-------|--------|-------------|
| **Stage 1** | ✅ Complete | Abstraction Layer |
| **Stage 2** | ✅ Complete | Extended Encoding (62+ options) |
| **Stage 3** | ✅ Complete | New Input Type Implementations |
| **Stage 3.5** | ✅ Complete | Game Type Hierarchy |
| **Stage 4** | ⏳ Pending | Training Enhancements |
| **Stage 5** | ⏳ Pending | Data Collection Tooling |
| **Stage 6** | 🔄 In Progress | Dragon Age Inquisition Schema + Standalone Renderer |

---

## Stage 1: Abstraction Layer ✅ COMPLETE

**Goal:** Define polymorphic input type hierarchy to support different input types beyond sliders.

### What Was Implemented

#### New File: `input_types.py`

Created a new module defining the input type hierarchy:

```python
# Protocol for all input types (duck typing for backward compat)
class InputType(Protocol):
    name: str
    biased: bool
    def num_classes(self) -> int: ...
    def encode(self, value: int) -> str: ...
    def decode(self, encoded: str) -> int: ...
    @property
    def encoding_width(self) -> int: ...

# Backward-compatible Slider (same constructor signature as original)
@dataclass
class Slider:
    digit: int           # Position in code
    name: str            # Human-readable name
    options: int         # Number of discrete options
    biased: bool = False # Bias sampling toward this input

# Placeholders for Stage 3
class CoordinateGrid: ...  # 2D X,Y picker
class ColorPicker: ...     # Color wheel/palette
```

#### Modified: `base_schema.py`

- Added `get_inputs() -> List[InputType]` as the primary abstract method
- Deprecated `get_sliders()` - now calls `get_inputs()` and filters for Slider types
- Updated `generate_code()` to use `InputType.encode()`
- Added `apply_input_value()` method for future game automation
- Extended `option_chars()` limit from 23 to 26 letters

#### Modified: `trainer.py`

- Changed `self.schema.get_sliders()` → `self.schema.get_inputs()`
- Updated `NUM_CLASSES_PER_DIGIT` to use `inp.num_classes()`
- Symbol maps now built using `inp.get_symbol_list()`
- Updated `eval_epoch()` to use `self.inputs` instead of `get_sliders()`
- Updated `CodeImageDataset._code_to_label()` to use `get_inputs()`

#### Modified: Schema Files

All four ME schemas updated:
- `schemas/me1_male.py`
- `schemas/me1_female.py`
- `schemas/me3_male.py`
- `schemas/me3_female.py`

Changes:
- Renamed `get_sliders()` → `get_inputs()`
- Updated return type to `List[InputType]`
- Added import for `InputType` from `base_schema`

### Backward Compatibility

- ✅ `Slider(digit, name, options, biased=False)` constructor unchanged
- ✅ `get_sliders()` still works (deprecated wrapper)
- ✅ Existing character codes remain valid
- ✅ All syntax checks pass

---

## Stage 2: Extended Encoding ✅ COMPLETE

**Goal:** Support 100+ options per position with extended alphabet encoding.

### What Was Implemented

#### `input_types.py`

Added extended encoding helper functions:
```python
# Extended alphabet: 0-9, A-Z, a-z (62 chars)
EXTENDED_ALPHABET = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
EXTENDED_CHAR_TO_IDX = {ch: i for i, ch in enumerate(EXTENDED_ALPHABET)}

# New helper functions
def extended_encode(value: int, num_classes: int) -> str:
    """Encode using 0-indexed extended alphabet (1-char for ≤62, 2-char for ≤3844)"""

def extended_decode(encoded: str, num_classes: int) -> int:
    """Decode back to 0-indexed value"""

def extended_encoding_width(num_classes: int) -> int:
    """Return 1 for ≤62 options, 2 for ≤3844 options"""

def extended_symbol_list(num_classes: int) -> List[str]:
    """Return all valid symbols for num_classes options"""
```

**Key Design Decision:** Slider keeps its legacy 1-indexed encoding ('1'-'9', 'A'-'Z') for ME backward compatibility. CoordinateGrid and ColorPicker use the new 0-indexed extended encoding.

#### `CoordinateGrid` and `ColorPicker`

Updated to use extended encoding:
- Single-char for ≤62 options (e.g., 8×8 grid = 64 → 2 chars)
- Two-char for ≤3844 options (e.g., 10×10 grid = 100 options)
- Removed `NotImplementedError` placeholders

#### `trainer.py`

Updated `CodeImageDataset._code_to_label()`:
- Now uses `encoding_width` to extract correct number of characters per input
- Supports multi-char symbols (not just single char iteration)
- Handles both legacy (uppercase) and extended (case-sensitive) encodings

#### `base_schema.py`

Updated `option_chars()`:
- Added deprecation warning for n > 35
- Clarified it's for legacy ME encoding only

### Tests Added

New test file: `tests/test_encoding.py` (24 tests)
- Extended encoding helper function tests
- Slider legacy encoding tests
- CoordinateGrid encoding tests
- ColorPicker encoding tests
- Backward compatibility tests

### Backward Compatibility

- ✅ Slider encoding unchanged ('1'-'9', 'A'-'Z')
- ✅ Existing ME character codes remain valid
- ✅ All ME schemas load and generate codes correctly

---

## Stage 3: New Input Type Implementations ✅ COMPLETE

**Goal:** Fully implement CoordinateGrid and ColorPicker with game automation.

### What Was Implemented

#### CoordinateGrid (in `input_types.py`)

Fully implemented 2D coordinate picker with:
- `num_classes()`: Returns x_divisions × y_divisions
- `encode()`/`decode()`: Extended encoding (0-indexed, supports multi-char for >62 options)
- `index_to_xy()`/`xy_to_index()`: Grid coordinate conversion
- `get_click_coordinates()`: Calculates screen position for clicking grid cells
- `get_symbol_list()`: Returns valid encoded symbols

#### ColorPicker (in `input_types.py`)

Fully implemented color wheel picker with:
- `num_classes()`: Returns palette_size
- `encode()`/`decode()`: Extended encoding
- `index_to_hsl()`: Converts palette index to HSL values
- `get_click_coordinates()`: Calculates polar coordinates on color wheel
- `get_symbol_list()`: Returns valid encoded symbols

#### Game Automation (in `base_schema.py`)

New methods added:

```python
def decode_code(self, code: str) -> List[Tuple[InputType, int]]:
    """Decode a code string into (input, value) pairs."""

def apply_input_value(self, inp: InputType, value: int, click_delay: float = 0.1):
    """Apply a value via click for CoordinateGrid/ColorPicker inputs."""

def apply_all_inputs(self, code: str, click_delay: float = 0.1, inter_input_delay: float = 0.05):
    """Apply all input values from a code string to the game UI."""

@property
def uses_code_paste(self) -> bool:
    """Whether schema uses clipboard paste (ME-style) vs direct UI (DAI-style)."""

@property
def has_direct_ui_inputs(self) -> bool:
    """Whether schema has any CoordinateGrid or ColorPicker inputs."""
```

### Tests Added

12 new tests in `tests/test_encoding.py`:
- `TestCoordinateGridClickCoordinates`: Grid click coordinate calculations
- `TestColorPickerClickCoordinates`: Color wheel click coordinates, HSL conversion
- `TestSchemaDecodeCode`: Code decoding, round-trip, extended encoding
- `TestSchemaProperties`: `uses_code_paste`, `has_direct_ui_inputs`

**Total tests: 38 (up from 24)**

---

## Stage 3.5: Game Type Hierarchy ✅ COMPLETE

**Goal:** Create a Game abstraction layer to separate game-level configuration from schema-level input definitions.

### What Was Implemented

#### New File: `game.py`

Created a Game base class with automation mode distinction:

```python
class AutomationMode(Enum):
    CODE_PASTE = "code_paste"      # Mass Effect style: paste a code string
    DIRECT_UI = "direct_ui"        # Dragon Age style: click UI elements directly

class Game(ABC):
    GAME_NAME: ClassVar[str]
    WINDOW_KEYWORD: ClassVar[str]
    AUTOMATION_MODE: ClassVar[AutomationMode]

    @classmethod
    def uses_code_paste(cls) -> bool: ...
    @classmethod
    def uses_direct_ui(cls) -> bool: ...

# Game implementations
class MassEffect1(Game):
    GAME_NAME = "Mass Effect 1"
    WINDOW_KEYWORD = "Mass Effect"
    AUTOMATION_MODE = AutomationMode.CODE_PASTE

class MassEffect3(Game):
    GAME_NAME = "Mass Effect 3"
    WINDOW_KEYWORD = "Mass Effect 3"
    AUTOMATION_MODE = AutomationMode.CODE_PASTE

class DragonAgeInquisition(Game):
    GAME_NAME = "Dragon Age: Inquisition"
    WINDOW_KEYWORD = "Dragon Age Inquisition"
    AUTOMATION_MODE = AutomationMode.DIRECT_UI
    DEFAULT_GRID_DIVISIONS = 10
    DEFAULT_PALETTE_SIZE = 36
```

#### Modified: `base_schema.py`

- Added `GAME: ClassVar[Optional[Type[Game]]]` class variable
- Added `game` property to return the game class
- Added `game_name` property for human-readable game name
- Added `uses_direct_ui` property (complements `uses_code_paste`)
- Updated `uses_code_paste` to check game's automation mode
- Re-exports Game classes for convenience

#### Modified: All ME Schemas

Updated all four Mass Effect schemas to use the Game hierarchy:

```python
# Before
class ME1_Female_Schema(BaseSchema):
    WINDOW_KEYWORD = "Mass Effect"  # Duplicated in each schema
    ...

# After
class ME1_Female_Schema(BaseSchema):
    GAME = MassEffect1  # References game class
    ...
    def setup_window(self):
        setup_window(self.GAME.WINDOW_KEYWORD, ...)  # Uses game config
```

### Game Hierarchy Diagram

```
Game (ABC)
├── GAME_NAME, WINDOW_KEYWORD, AUTOMATION_MODE
├── uses_code_paste(), uses_direct_ui()
│
├── MassEffect1 (CODE_PASTE)
│   ├── ME1_Female_Schema
│   └── ME1_Male_Schema
│
├── MassEffect3 (CODE_PASTE)
│   ├── ME3_Female_Schema
│   └── ME3_Male_Schema
│
└── DragonAgeInquisition (DIRECT_UI)
    ├── DAI_Human_Female (implemented)
    ├── DAI_Human_Male (future)
    ├── DAI_Elf_Female (future)
    └── ... (other species/genders)
```

### Tests Added

7 new tests in `tests/test_encoding.py`:
- `TestGameHierarchy`: Automation modes, game names
- `TestDAISchema`: DAI Human Female loading, input counts, code generation

**Total tests: 45 (up from 38)**

---

## Stage 4: Training Enhancements ⏳ PENDING

**Goal:** Improve training for heterogeneous input types with varying class counts.

### Loss Normalization

Current loss sums cross-entropy without normalization, biasing gradients toward high-class inputs:

```python
# Current (trainer.py:204-205)
for d, out in enumerate(outs):
    loss = loss + F.cross_entropy(out, labels[:, d])

# Proposed: Normalize by log(num_classes)
for d, (out, inp) in enumerate(zip(outs, self.inputs)):
    ce = F.cross_entropy(out, labels[:, d])
    normalized = ce / math.log(inp.num_classes() + 1)
    loss = loss + normalized
```

### Per-Input-Type Metrics

Track accuracy grouped by input type in TensorBoard:
```python
metrics_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
for inp, acc in zip(self.inputs, per_digit_acc):
    type_name = type(inp).__name__
    metrics_by_type[type_name]['correct'] += acc
    metrics_by_type[type_name]['total'] += 1
```

### Augmentation Adjustment

Disable color jitter when schema has ColorPicker inputs:
```python
def make_transforms(self, img_size, aug=True, has_color_inputs=False):
    if aug and not has_color_inputs:
        transforms_list.append(ColorJitter(...))
```

---

## Stage 5: Data Collection Tooling ⏳ PENDING

**Goal:** Create tools for configuring game automation and collecting training data.

### New Directory: `tools/`

#### `tools/coordinate_configurator.py`

Interactive GUI for recording click coordinates:
- Click to record points
- Drag to record regions (bounding boxes)
- Save/load JSON configs
- Test mode to verify automation

#### `tools/color_data_collector.py`

Automated color picker training data collector:
- Iterate through palette systematically
- Click color wheel at calculated positions
- Capture screenshots
- Save with encoded color value in filename

#### `tools/macro_tester.py`

Test automation macros without full data collection:
- Test individual input value application
- Dry-run mode (log without clicking)
- Cycle through all inputs with test values

#### `tools/schema_validator.py`

Validate schema configurations:
- Check coordinate bounds
- Verify encoding capacity
- Validate input configurations

---

## Stage 6: Dragon Age Inquisition Schema 🔄 IN PROGRESS

**Goal:** Implement first new game as MVP (Human Female first).

### Implemented: `schemas/dai_human_female.py`

The first DAI schema is complete with all 56 inputs:

```python
@register_schema("dai_human_female")
class DAI_Human_Female_Schema(BaseSchema):
    GAME = DragonAgeInquisition
    OUTPUT_DIR = "dataset/dai/human_female"

    def get_inputs(self) -> List[InputType]:
        # 56 total inputs organized by category:
        # - General: 6 sliders (Face Shape, Skin Tone, Complexion, Eye Color, Hair, Hair Color)
        # - Head and Ears: 7 coordinate grids
        # - Eyes: 4 sliders + 3 grids + 2 color pickers
        # - Nose: 1 slider + 8 grids
        # - Mouth and Jaw: 1 slider + 10 grids
        # - Makeup: 6 sliders + 8 color pickers
        ...
```

### DAI Human Female Input Summary

| Category | Sliders | Grids | Colors | Total |
|----------|---------|-------|--------|-------|
| General | 6 | 0 | 0 | 6 |
| Head and Ears | 0 | 7 | 0 | 7 |
| Eyes | 4 | 3 | 2 | 9 |
| Nose | 1 | 8 | 0 | 9 |
| Mouth and Jaw | 1 | 10 | 0 | 11 |
| Makeup | 6 | 0 | 8 | 14 |
| **Total** | **18** | **28** | **10** | **56** |

**Code Length:** 18 (sliders × 1 char) + 56 (grids × 2 chars) + 10 (colors × 1 char) = **84 characters**

### Key Differences from Mass Effect

| Feature | Mass Effect | Dragon Age Inquisition |
|---------|-------------|------------------------|
| Character codes | Clipboard paste | Direct UI manipulation |
| Input types | Sliders only (35-37) | Mixed: 18 sliders + 28 grids + 10 colors |
| Code length | 35-37 chars | 84 chars |
| Data collection | Paste code, screenshot | Click each input, screenshot |
| Automation mode | `CODE_PASTE` | `DIRECT_UI` |

### What's Working

- ✅ Schema loads and registers correctly
- ✅ All 56 inputs defined with correct types
- ✅ Code generation works (84-char codes)
- ✅ Code decode/encode round-trips correctly
- ✅ `uses_direct_ui` returns True
- ✅ `has_direct_ui_inputs` returns True

### What Needs Calibration (Stage 5)

- ⏳ Grid bounds for each CoordinateGrid (currently placeholder `(0, 0, 100, 100)`)
- ⏳ Color wheel center/radius for each ColorPicker (currently placeholder)
- ⏳ Slider click positions (DAI sliders need clicking, not code paste)
- ⏳ Window location and capture box

### Remaining DAI Schemas (Future)

| Schema | Status |
|--------|--------|
| `dai_human_female` | ✅ Implemented |
| `dai_human_male` | ⏳ Planned |
| `dai_elf_female` | ⏳ Planned |
| `dai_elf_male` | ⏳ Planned |
| `dai_dwarf_female` | ⏳ Planned |
| `dai_dwarf_male` | ⏳ Planned |
| `dai_qunari_female` | ⏳ Planned |
| `dai_qunari_male` | ⏳ Planned |

---

## Dependency Graph

```
Stage 1: Abstraction Layer ✅
    │
    ├──→ Stage 2: Extended Encoding ✅
    │         │
    │         ▼
    ├──→ Stage 3: New Input Types ✅
    │         │
    │         ▼
    ├──→ Stage 3.5: Game Hierarchy ✅
    │         │
    │         ├──→ Stage 6: DAI Schemas 🔄 (dai_human_female done)
    │         │
    │         ▼
    └──→ Stage 4: Training Enhancements ⏳
                    │
                    ▼
          Stage 5: Data Collection Tooling ⏳
                    │
                    ▼
          Stage 6: DAI Calibration & Remaining Schemas ⏳
```

---

## Files Summary

### New Files (Created)

| File | Stage | Purpose |
|------|-------|---------|
| `input_types.py` | 1 ✅ | InputType protocol, Slider, CoordinateGrid, ColorPicker |
| `tests/test_encoding.py` | 2 ✅ | Encoding tests (45 tests total) |
| `game.py` | 3.5 ✅ | Game base class, AutomationMode, MassEffect1/3, DragonAgeInquisition |
| `schemas/dai_human_female.py` | 6 🔄 | DAI Human Female schema (56 inputs) |
| `tools/coordinate_configurator.py` | 5 ⏳ | Interactive coordinate recording (planned) |
| `tools/color_data_collector.py` | 5 ⏳ | Automated color data collection (planned) |
| `tools/macro_tester.py` | 5 ⏳ | Automation testing (planned) |
| `tools/schema_validator.py` | 5 ⏳ | Schema validation (planned) |

### Modified Files

| File | Stage | Changes |
|------|-------|---------|
| `base_schema.py` | 1-3.5 ✅ | `get_inputs()`, `decode_code()`, `apply_input_value()`, `apply_all_inputs()`, `GAME` class var, Game imports |
| `trainer.py` | 1 ✅, 2 ✅ | Use `get_inputs()`, variable-width encoding support |
| `input_types.py` | 1-3 ✅ | Extended encoding, CoordinateGrid, ColorPicker with click coordinates |
| `schemas/me1_female.py` | 1 ✅, 3.5 ✅ | `get_inputs()`, `GAME = MassEffect1` |
| `schemas/me1_male.py` | 1 ✅, 3.5 ✅ | `get_inputs()`, `GAME = MassEffect1` |
| `schemas/me3_female.py` | 1 ✅, 3.5 ✅ | `get_inputs()`, `GAME = MassEffect3` |
| `schemas/me3_male.py` | 1 ✅, 3.5 ✅ | `get_inputs()`, `GAME = MassEffect3` |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| DAI has no character codes | High | High | Plan for direct UI manipulation from start |
| Color wheel geometry varies | Medium | Medium | Make geometry configurable per ColorPicker |
| Combinatorial explosion | High | Medium | Smart sampling, focus on biased inputs |
| Backward compatibility breaks | Low | High | Extensive testing on ME schemas |

---

## Testing Checklist

### Stage 1 ✅
- [x] Slider constructor backward compatible
- [x] `get_sliders()` still works
- [x] Syntax checks pass on all files
- [x] InputType protocol works with isinstance()
- [x] Encode/decode round-trip for Slider

### Stage 2 ✅
- [x] Extended alphabet encode/decode
- [x] Multi-char encoding for >62 options
- [x] Parse synthetic codes with lowercase
- [x] Existing ME codes still work
- [x] Slider backward compatibility preserved
- [x] CoordinateGrid uses extended encoding
- [x] ColorPicker uses extended encoding
- [x] trainer.py handles variable-width encodings

### Stage 3 ✅
- [x] CoordinateGrid index ↔ xy conversion
- [x] ColorPicker index ↔ HSL conversion
- [x] Click coordinate calculations
- [x] Schema decode_code method
- [x] Schema properties (uses_code_paste, has_direct_ui_inputs)
- [ ] Visual validation of click positions (deferred to Stage 5 tooling)

### Stage 3.5 ✅
- [x] Game base class with AutomationMode enum
- [x] MassEffect1, MassEffect3, DragonAgeInquisition game classes
- [x] BaseSchema GAME class variable integration
- [x] All ME schemas updated to use GAME
- [x] uses_code_paste/uses_direct_ui derived from game
- [x] Tests for game automation modes

### Stage 4 (Pending)
- [ ] Loss normalization reduces gradient bias
- [ ] Per-type metrics in TensorBoard
- [ ] Color jitter disabled for color schemas

### Stage 5 (Pending)
- [ ] Coordinate configurator saves/loads configs
- [ ] Color collector iterates palette correctly
- [ ] Macro tester validates clicks
- [ ] Schema validator catches errors

### Stage 6 (In Progress)
- [x] DAI Human Female schema loads
- [x] All 56 inputs configured (18 sliders, 28 grids, 10 colors)
- [x] Code generation works (84-char codes)
- [x] Code encode/decode round-trips correctly
- [x] Frostbite data extraction pipeline (TOC/SB/CAT/CAS, EBX parser, mesh/morph/texture extraction)
- [x] Standalone renderer — headless pyrender with PBR materials, no game needed
- [x] Morph engine — two-way (68), one-way (49), Section 4 (36) morph targets
- [x] Input mapper — all 56 UI inputs → morph weights + shader params
- [x] PBR quality pass — normal maps, roughness maps, mesh subdivision, diffuse smoothing
- [x] Eye texture calibrated to actual mesh UV mapping (iris/sclera proportions)
- [x] Hair rendering — 16 styles, RGBA card textures with BLEND alpha for strand transparency
- [x] Makeup post-process — landmark-based (lip, blush, eye shadow, eyeliner)
- [x] Capture tool — 512x512 default, supersampling, underscore-separated filenames
- [ ] Generate 10k dataset
- [ ] Train dual-input model (CNN + face metrics)
- [ ] Evaluate prediction accuracy

---

*Last Updated: October 22, 2026*
*Current Stage: Stage 3.5 Complete, Stage 6 (DAI) in progress — standalone renderer complete, dataset generation next*
