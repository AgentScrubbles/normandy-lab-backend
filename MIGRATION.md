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
| **Stage 3** | ⏳ Pending | New Input Type Implementations |
| **Stage 4** | ⏳ Pending | Training Enhancements |
| **Stage 5** | ⏳ Pending | Data Collection Tooling |
| **Stage 6** | ⏳ Pending | Dragon Age Inquisition Schema |

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

## Stage 3: New Input Type Implementations ⏳ PENDING

**Goal:** Fully implement CoordinateGrid and ColorPicker with game automation.

### CoordinateGrid

For 2D inputs like "nose width/height" where user selects a point on a grid:

```python
@dataclass
class CoordinateGrid:
    position: int
    name: str
    x_divisions: int     # e.g., 10
    y_divisions: int     # e.g., 10
    grid_bounds: Tuple[int, int, int, int]  # screen coords (x1, y1, x2, y2)
    biased: bool = False

    def num_classes(self) -> int:
        return self.x_divisions * self.y_divisions  # e.g., 100

    def index_to_xy(self, idx: int) -> Tuple[int, int]:
        return (idx % self.x_divisions, idx // self.x_divisions)

    def get_click_coordinates(self, idx: int) -> Tuple[int, int]:
        # Calculate screen position for clicking this grid cell
        ...
```

### ColorPicker

For color wheels discretized to a palette using HSL representation:

```python
@dataclass
class ColorPicker:
    position: int
    name: str
    palette_size: int        # e.g., 36 (12 hues × 3 saturations)
    wheel_center: Tuple[int, int]
    wheel_radius: int
    hue_divisions: int = 12
    saturation_levels: int = 3
    biased: bool = False

    def index_to_hsl(self, idx: int) -> Tuple[float, float, float]:
        # Convert palette index to H, S, L values
        ...

    def get_click_coordinates(self, idx: int) -> Tuple[int, int]:
        # Calculate screen position on color wheel
        ...
```

### Game Automation

Add to `base_schema.py`:
```python
def apply_input_value(self, input: InputType, value: int):
    if isinstance(input, CoordinateGrid):
        x, y = input.get_click_coordinates(value)
        pyautogui.click(x, y)
    elif isinstance(input, ColorPicker):
        x, y = input.get_click_coordinates(value)
        pyautogui.click(x, y)
    # Sliders handled by code paste (existing behavior)
```

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

## Stage 6: Dragon Age Inquisition Schema ⏳ PENDING

**Goal:** Implement first new game as MVP (Human only).

### New File: `schemas/dai_human.py`

```python
@register_schema("dai_human")
class DAI_Human_Schema(BaseSchema):
    WINDOW_KEYWORD = "Dragon Age Inquisition"
    OUTPUT_DIR = "dataset/dai/human"

    def get_inputs(self) -> List[InputType]:
        return [
            # Standard sliders
            Slider(position=1, name="Face Shape", options=10),

            # 2D coordinate grids
            CoordinateGrid(
                position=2, name="Eye Position",
                x_divisions=10, y_divisions=10,
                grid_bounds=(100, 200, 300, 400)
            ),

            # Color pickers
            ColorPicker(
                position=3, name="Skin Color",
                palette_size=36,
                wheel_center=(500, 500),
                wheel_radius=100
            ),
            # ... more inputs
        ]
```

### Key Differences from Mass Effect

| Feature | Mass Effect | Dragon Age Inquisition |
|---------|-------------|------------------------|
| Character codes | Clipboard paste | Direct UI manipulation |
| Input types | Sliders only | Sliders + Colors + 2D grids |
| Data collection | Paste code, screenshot | Click each input, screenshot |

### Species Strategy

Start with **Human only** as MVP. Other species (Elf, Dwarf, Qunari) have mostly the same sliders and can be added as separate schema files following the same pattern.

### Implementation Steps

1. Document DAI character creator UI (window title, input locations)
2. Use `coordinate_configurator` to record all click positions
3. Define schema with mix of Slider, CoordinateGrid, ColorPicker
4. Implement `navigate_to_customization()` for UI navigation
5. Implement `apply_all_inputs()` for direct value application
6. Run data collection (sample from combinatorial space)
7. Train and evaluate

---

## Dependency Graph

```
Stage 1: Abstraction Layer ✅
    │
    ├──→ Stage 2: Extended Encoding
    │         │
    │         ▼
    ├──→ Stage 3: New Input Types
    │         │
    │         ▼
    └──→ Stage 4: Training Enhancements
                    │
                    ▼
          Stage 5: Data Collection Tooling
                    │
                    ▼
          Stage 6: Dragon Age Inquisition Schema
```

---

## Files Summary

### New Files (Created)

| File | Stage | Purpose |
|------|-------|---------|
| `input_types.py` | 1 ✅ | InputType protocol, Slider, CoordinateGrid, ColorPicker |
| `tests/test_encoding.py` | 2 ✅ | Extended encoding tests (24 tests) |
| `tools/coordinate_configurator.py` | 5 | Interactive coordinate recording |
| `tools/color_data_collector.py` | 5 | Automated color data collection |
| `tools/macro_tester.py` | 5 | Automation testing |
| `tools/schema_validator.py` | 5 | Schema validation |
| `schemas/dai_human.py` | 6 | Dragon Age Inquisition Human schema |

### Modified Files

| File | Stage | Changes |
|------|-------|---------|
| `base_schema.py` | 1 ✅, 2 ✅ | `get_inputs()`, deprecate `get_sliders()`, `apply_input_value()`, `option_chars()` deprecation warning |
| `trainer.py` | 1 ✅, 2 ✅, 4 | Use `get_inputs()`, variable-width encoding support, loss normalization (pending) |
| `input_types.py` | 1 ✅, 2 ✅ | Extended encoding helpers, CoordinateGrid/ColorPicker encoding |
| `schemas/me1_male.py` | 1 ✅ | Rename `get_sliders()` → `get_inputs()` |
| `schemas/me1_female.py` | 1 ✅ | Rename `get_sliders()` → `get_inputs()` |
| `schemas/me3_male.py` | 1 ✅ | Rename `get_sliders()` → `get_inputs()` |
| `schemas/me3_female.py` | 1 ✅ | Rename `get_sliders()` → `get_inputs()` |

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

### Stage 3 (Pending)
- [ ] CoordinateGrid index ↔ xy conversion
- [ ] ColorPicker index ↔ HSL conversion
- [ ] Click coordinate calculations
- [ ] Visual validation of click positions

### Stage 4 (Pending)
- [ ] Loss normalization reduces gradient bias
- [ ] Per-type metrics in TensorBoard
- [ ] Color jitter disabled for color schemas

### Stage 5 (Pending)
- [ ] Coordinate configurator saves/loads configs
- [ ] Color collector iterates palette correctly
- [ ] Macro tester validates clicks
- [ ] Schema validator catches errors

### Stage 6 (Pending)
- [ ] DAI Human schema loads
- [ ] All inputs configured
- [ ] Data collection pilot successful
- [ ] Training converges

---

*Last Updated: December 2024*
*Current Stage: 2 Complete, proceeding to Stage 3*
