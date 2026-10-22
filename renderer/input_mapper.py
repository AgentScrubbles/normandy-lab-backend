"""
Maps DA:I game UI inputs (56 controls) to internal morph sliders and shader parameters.

The game UI has:
- 6 General preset/option sliders (Face Shape, Skin Tone, Complexion, Eye Color, Hair, Hair Color)
- 7 Head/Ears grids (each grid = 2 morph axes)
- 11 Eyes inputs (3 grids + 3 sliders + 2 colors + preset sliders)
- 9 Nose inputs (1 preset + 8 grids)
- 11 Mouth/Jaw inputs (1 preset + 10 grids)
- 12 Makeup inputs (colors + intensity sliders — shader only, no geometry)

The morph resource has:
- 68 two-way sliders (each has left/right delta, weight -1 to +1)
- 49 one-way sliders (each has single delta, weight 0 to 1)
- 36 morph shapes (Section 4, delta-based)
- 6 shape targets (Section 1, absolute positions for face presets)

Category enum from EBX: Head=0, Mouth=1, Eyes=2, Nose=3, Ears=4, Neck=5, Jaw=6, Cheeks=7, Teeth=8, Lips=9
"""

import json
import os
from dataclasses import dataclass, field

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dai")
MAPPING_PATH = os.path.join(ASSET_DIR, "input_mapping.json")

# Category names from the MorphSliderType enum
CATEGORIES = {
    0: 'Head', 1: 'Mouth', 2: 'Eyes', 3: 'Nose', 4: 'Ears',
    5: 'Neck', 6: 'Jaw', 7: 'Cheeks', 8: 'Teeth', 9: 'Lips',
}


@dataclass
class UIInput:
    """A single game UI input and its mapping to morph/shader parameters."""
    name: str
    category: str
    input_type: str    # 'slider', 'grid', 'color'
    morph_type: str = ''        # 'twoway', 'oneway', 'shape', 'shader', 'preset'
    x_morph_idx: int = -1       # morph slider index for X axis (grids)
    y_morph_idx: int = -1       # morph slider index for Y axis (grids)
    morph_idx: int = -1         # single morph slider index (for 1D sliders)
    shape_indices: list = field(default_factory=list)  # for face/nose/mouth/eye shape presets
    shader_param: str = ''      # shader parameter name
    option_count: int = 0
    # For grids that use mixed slider types (e.g. one twoway + one oneway)
    x_slider_type: str = ''     # 'twoway' or 'oneway'
    y_slider_type: str = ''     # 'twoway' or 'oneway'


def build_category_index(morph_mapping: dict) -> dict:
    """Build index of morph sliders grouped by category.

    Returns dict mapping category_name -> {'twoway': [indices], 'oneway': [indices], 'shapes': [indices]}
    """
    index = {}
    for cat_name in CATEGORIES.values():
        index[cat_name] = {'twoway': [], 'oneway': [], 'shapes': []}

    for i, s in enumerate(morph_mapping['morph_sliders']['two_way']):
        cat = s['category']
        if cat in index:
            index[cat]['twoway'].append(i)

    for i, s in enumerate(morph_mapping['morph_sliders']['one_way']):
        cat = s['category']
        if cat in index:
            index[cat]['oneway'].append(i)

    for i, s in enumerate(morph_mapping['morph_shapes']['shapes']):
        cat = s['category']
        if cat in index:
            index[cat]['shapes'].append(i)

    return index


def build_input_mapping() -> list[UIInput]:
    """Build the mapping from 56 UI inputs to morph sliders.

    Uses the morph_mapping.json category assignments to correlate UI inputs
    with internal morph sliders. Grid inputs map X/Y to consecutive morph
    slider pairs within the matching category.
    """
    mapping_path = os.path.join(ASSET_DIR, "morph_mapping.json")
    with open(mapping_path) as f:
        morph_mapping = json.load(f)

    cat_idx = build_category_index(morph_mapping)

    # Track consumption of morph sliders per category
    cat_consumed = {cat: {'twoway': 0, 'oneway': 0, 'shapes': 0} for cat in cat_idx}

    def _next_twoway_pair(category: str) -> tuple[int, int]:
        """Get next two consecutive two-way slider indices for a grid."""
        pos = cat_consumed[category]['twoway']
        sliders = cat_idx[category]['twoway']
        if pos + 1 < len(sliders):
            cat_consumed[category]['twoway'] = pos + 2
            return sliders[pos], sliders[pos + 1]
        return -1, -1

    def _next_oneway_pair(category: str) -> tuple[int, int]:
        """Get next two consecutive one-way slider indices for a grid."""
        pos = cat_consumed[category]['oneway']
        sliders = cat_idx[category]['oneway']
        if pos + 1 < len(sliders):
            cat_consumed[category]['oneway'] = pos + 2
            return sliders[pos], sliders[pos + 1]
        return -1, -1

    def _next_twoway(category: str) -> int:
        """Get next single two-way slider index."""
        pos = cat_consumed[category]['twoway']
        sliders = cat_idx[category]['twoway']
        if pos < len(sliders):
            cat_consumed[category]['twoway'] = pos + 1
            return sliders[pos]
        return -1

    def _next_oneway(category: str) -> int:
        """Get next single one-way slider index."""
        pos = cat_consumed[category]['oneway']
        sliders = cat_idx[category]['oneway']
        if pos < len(sliders):
            cat_consumed[category]['oneway'] = pos + 1
            return sliders[pos]
        return -1

    def _grid_from_category(name: str, category: str) -> UIInput:
        """Create a grid UIInput, using twoway pair if available, falling back to oneway."""
        x_tw, y_tw = _next_twoway_pair(category)
        if x_tw >= 0:
            return UIInput(name, category, 'grid', morph_type='twoway',
                           x_morph_idx=x_tw, y_morph_idx=y_tw,
                           x_slider_type='twoway', y_slider_type='twoway')
        # Fall back to one-way pair
        x_ow, y_ow = _next_oneway_pair(category)
        if x_ow >= 0:
            return UIInput(name, category, 'grid', morph_type='oneway',
                           x_morph_idx=x_ow, y_morph_idx=y_ow,
                           x_slider_type='oneway', y_slider_type='oneway')
        # Mix: try 1 remaining twoway + 1 oneway
        x_tw2 = _next_twoway(category)
        y_ow2 = _next_oneway(category)
        if x_tw2 >= 0 or y_ow2 >= 0:
            return UIInput(name, category, 'grid', morph_type='twoway',
                           x_morph_idx=x_tw2, y_morph_idx=y_ow2,
                           x_slider_type='twoway' if x_tw2 >= 0 else '',
                           y_slider_type='oneway' if y_ow2 >= 0 else '')
        return UIInput(name, category, 'grid', morph_type='twoway')

    inputs = []

    # === GENERAL ===
    # Face Shape: preset selector → Section 1 shape targets (6) + Section 4 Head shapes
    # Head category Section 4 shapes: [0,1,2,3,4,5,6,19,31,32,33,34,35] = 13 shapes
    # Face Shape has 7 presets, so we map to first 7 Head-category Section 4 shapes
    head_sec4 = cat_idx['Head']['shapes']  # [0,1,2,3,4,5,6,19,31,32,33,34,35]
    inputs.append(UIInput('Face Shape', 'General', 'slider', morph_type='preset',
                          shape_indices=list(range(6)),  # Section 1 shapes
                          option_count=7))
    # Skin Tone: shader only
    inputs.append(UIInput('Skin Tone', 'General', 'slider', morph_type='shader',
                          shader_param='SkinTint', option_count=11))
    # Complexion: shader only
    inputs.append(UIInput('Complexion', 'General', 'slider', morph_type='shader',
                          shader_param='ComplextionBlend_Diff_Norm_Spec', option_count=14))
    # Eye Color: shader only
    inputs.append(UIInput('Eye Color', 'General', 'slider', morph_type='shader',
                          shader_param='EyeColor', option_count=51))
    # Hair: preset selector → hair mesh index
    inputs.append(UIInput('Hair', 'General', 'slider', morph_type='preset',
                          option_count=28))
    # Hair Color: shader only
    inputs.append(UIInput('Hair Color', 'General', 'slider', morph_type='shader',
                          shader_param='hairColor', option_count=25))

    # === HEAD AND EARS ===
    inputs.append(_grid_from_category('Ear Size', 'Ears'))
    inputs.append(_grid_from_category('Ear Position', 'Ears'))
    inputs.append(_grid_from_category('Earlobe Size', 'Ears'))
    inputs.append(_grid_from_category('Cheekbones', 'Cheeks'))
    inputs.append(_grid_from_category('Cheek Shape', 'Cheeks'))
    inputs.append(_grid_from_category('Brow Size', 'Head'))
    inputs.append(_grid_from_category('Forehead', 'Head'))

    # === EYES ===
    # Eye Shape: preset → Section 4 Eyes shapes [8-14] (7 shapes for 8 presets)
    eye_sec4 = cat_idx['Eyes']['shapes']  # [8,9,10,11,12,13,14]
    inputs.append(UIInput('Eye Shape', 'Eyes', 'slider', morph_type='preset',
                          shape_indices=eye_sec4, option_count=8))
    inputs.append(_grid_from_category('Eye Size', 'Eyes'))
    inputs.append(_grid_from_category('Eye Position', 'Eyes'))
    inputs.append(_grid_from_category('Eye Rotation and Spacing', 'Eyes'))
    # Eyebrow Style: shader parameter
    inputs.append(UIInput('Eyebrow Style', 'Eyes', 'slider', morph_type='shader',
                          shader_param='Eyebrow_BrowSel_Opacity_VOffset', option_count=23))
    # Eyebrow Position: one-way morph
    ow_idx = _next_oneway('Eyes')
    inputs.append(UIInput('Eyebrow Position', 'Eyes', 'slider', morph_type='oneway',
                          morph_idx=ow_idx, option_count=10))
    # Eyelash Style: shader
    inputs.append(UIInput('Eyelash Style', 'Eyes', 'slider', morph_type='shader',
                          shader_param='LashStyle', option_count=14))
    # Iris colors: shader
    inputs.append(UIInput('Outer Iris Color', 'Eyes', 'color', morph_type='shader',
                          shader_param='TintMid', option_count=12))
    inputs.append(UIInput('Inner Iris Color', 'Eyes', 'color', morph_type='shader',
                          shader_param='TintInner', option_count=12))

    # === NOSE ===
    # Nose Shape: preset → Section 4 Nose shapes [24-30] (7 shapes for 8 presets)
    nose_sec4 = cat_idx['Nose']['shapes']  # [24,25,26,27,28,29,30]
    inputs.append(UIInput('Nose Shape', 'Nose', 'slider', morph_type='preset',
                          shape_indices=nose_sec4, option_count=8))
    inputs.append(_grid_from_category('Nose Size', 'Nose'))
    inputs.append(_grid_from_category('Nose Position', 'Nose'))
    inputs.append(_grid_from_category('Nose Bridge Size', 'Nose'))
    inputs.append(_grid_from_category('Bridge Position', 'Nose'))
    inputs.append(_grid_from_category('Nose Tip Size', 'Nose'))
    inputs.append(_grid_from_category('Nose Tip Position', 'Nose'))
    inputs.append(_grid_from_category('Nostril Size', 'Nose'))
    inputs.append(_grid_from_category('Broken Nose', 'Nose'))

    # === MOUTH AND JAW ===
    # Mouth Shape: preset → Section 4 Mouth shapes [15-18, 20-23] (8 shapes for 9 presets)
    mouth_sec4 = cat_idx['Mouth']['shapes']  # [15,16,17,18,20,21,22,23]
    inputs.append(UIInput('Mouth Shape', 'Mouth', 'slider', morph_type='preset',
                          shape_indices=mouth_sec4, option_count=9))
    inputs.append(_grid_from_category('Mouth Width', 'Mouth'))
    inputs.append(_grid_from_category('Mouth Position', 'Mouth'))
    inputs.append(_grid_from_category('Lip Thickness', 'Lips'))
    inputs.append(_grid_from_category('Jawline', 'Jaw'))
    inputs.append(_grid_from_category('Jaw Size', 'Jaw'))
    inputs.append(_grid_from_category('Jowls', 'Jaw'))
    inputs.append(_grid_from_category('Chin Size', 'Jaw'))       # was broken, now uses oneway fallback
    inputs.append(_grid_from_category('Chin Shape', 'Cheeks'))   # was broken, now uses oneway fallback
    inputs.append(_grid_from_category('Double Chin', 'Neck'))
    inputs.append(_grid_from_category('Adam\'s Apple', 'Neck'))  # was broken, now uses remaining Neck slider

    # === MAKEUP (all shader-only) ===
    makeup = [
        ('Eye Shadow Color', 'color', 'MakeupColorEyesA', 12),
        ('Eye Shadow Intensity', 'slider', 'MakeupLevels_EyeSh_Eye_Bls_Lp', 11),
        ('Under-Eye Color', 'color', 'MakeupColorEyesB', 12),
        ('Under-Brow Color', 'color', 'MakeupColorUnderbrow', 12),
        ('Under-Brow Intensity', 'slider', 'MakeupRecession_Eye_Blush_Ub', 11),
        ('Eyelash Color', 'color', 'LashColor', 12),
        ('Eyeliner Color', 'color', 'MakeupColorEyeliner', 12),
        ('Eyeliner Intensity', 'slider', 'MakeupLevels_EyeSh_Eye_Bls_Lp', 11),
        ('Blush Color', 'color', 'MakeupColorCheeks', 12),
        ('Blush Intensity', 'slider', 'MakeupLevels_EyeSh_Eye_Bls_Lp', 11),
        ('Lip Color', 'color', 'MakeupColorLips', 12),
        ('Lip Color Intensity', 'slider', 'MakeupLevels_EyeSh_Eye_Bls_Lp', 11),
        ('Lip Liner Color', 'color', 'MakeupColorLipLiner', 12),
        ('Lip Shine', 'slider', 'MakeupGlitter_Eye_Lip_Lev_In_ub', 11),
    ]
    for name, itype, param, count in makeup:
        inputs.append(UIInput(name, 'Makeup', itype, morph_type='shader',
                              shader_param=param, option_count=count))

    return inputs


def get_geometry_input_indices(mapping: list[UIInput]) -> list[int]:
    """Return indices of UI inputs that affect mesh geometry (not shader-only)."""
    return [i for i, m in enumerate(mapping) if m.morph_type in ('twoway', 'oneway', 'preset')]


def get_shader_input_indices(mapping: list[UIInput]) -> list[int]:
    """Return indices of UI inputs that are shader/visual only."""
    return [i for i, m in enumerate(mapping) if m.morph_type == 'shader']


def ui_values_to_morph_weights(mapping: list[UIInput], ui_values: list[int],
                                num_twoway: int = 68, num_oneway: int = 49,
                                num_shapes: int = 36, scale: float = 0.5
                                ) -> tuple[list[float], list[float], list[float]]:
    """Convert 56 UI input values to morph slider weights.

    Args:
        mapping: list of UIInput from build_input_mapping()
        ui_values: list of 56 integer values (quantized game UI values)
        num_twoway: total number of two-way sliders
        num_oneway: total number of one-way sliders
        num_shapes: total number of Section 4 shapes
        scale: morph intensity multiplier

    Returns:
        (twoway_weights, oneway_weights, shape_weights) ready for apply_morphs()
    """
    twoway = [0.0] * num_twoway
    oneway = [0.0] * num_oneway
    shapes = [0.0] * num_shapes

    for inp, val in zip(mapping, ui_values):
        if inp.morph_type == 'twoway' or (inp.morph_type == 'oneway' and inp.input_type == 'grid'):
            if inp.input_type == 'grid':
                # Grid value 0-99 → x=val%10, y=val//10 → normalized
                x_norm = (val % 10) / 9.0   # 0..1
                y_norm = (val // 10) / 9.0   # 0..1

                if inp.x_morph_idx >= 0:
                    if inp.x_slider_type == 'oneway':
                        if inp.x_morph_idx < num_oneway:
                            oneway[inp.x_morph_idx] = x_norm * scale
                    else:  # twoway
                        if inp.x_morph_idx < num_twoway:
                            twoway[inp.x_morph_idx] = (x_norm * 2 - 1) * scale
                if inp.y_morph_idx >= 0:
                    if inp.y_slider_type == 'oneway':
                        if inp.y_morph_idx < num_oneway:
                            oneway[inp.y_morph_idx] = y_norm * scale
                    else:  # twoway
                        if inp.y_morph_idx < num_twoway:
                            twoway[inp.y_morph_idx] = (y_norm * 2 - 1) * scale

        elif inp.morph_type == 'oneway' and inp.input_type == 'slider':
            if inp.morph_idx >= 0 and inp.morph_idx < num_oneway:
                # Slider value 0..(option_count-1) → normalized 0..1
                max_val = max(inp.option_count - 1, 1)
                oneway[inp.morph_idx] = (val / max_val) * scale

        elif inp.morph_type == 'preset' and inp.shape_indices:
            # Preset selector: activate one Section 4 shape based on the index
            # preset 0 = default (no shape), preset N activates shape N-1
            if val > 0 and val <= len(inp.shape_indices):
                shape_idx = inp.shape_indices[val - 1]
                if shape_idx < num_shapes:
                    shapes[shape_idx] = 1.0 * scale

    return twoway, oneway, shapes


def save_mapping(mapping: list[UIInput], path: str = MAPPING_PATH):
    """Save mapping to JSON."""
    data = []
    for inp in mapping:
        data.append({
            'name': inp.name,
            'category': inp.category,
            'input_type': inp.input_type,
            'morph_type': inp.morph_type,
            'x_morph_idx': inp.x_morph_idx,
            'y_morph_idx': inp.y_morph_idx,
            'morph_idx': inp.morph_idx,
            'shape_indices': inp.shape_indices,
            'shader_param': inp.shader_param,
            'option_count': inp.option_count,
            'x_slider_type': inp.x_slider_type,
            'y_slider_type': inp.y_slider_type,
        })
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_mapping(path: str = MAPPING_PATH) -> list[UIInput]:
    """Load mapping from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [UIInput(**d) for d in data]


if __name__ == '__main__':
    mapping = build_input_mapping()
    print(f"Built mapping for {len(mapping)} UI inputs:")
    for i, m in enumerate(mapping):
        detail = ''
        if m.input_type == 'grid':
            x_type = m.x_slider_type or '?'
            y_type = m.y_slider_type or '?'
            detail = f' → {x_type}[{m.x_morph_idx}], {y_type}[{m.y_morph_idx}]'
        elif m.morph_type == 'oneway':
            detail = f' → oneway[{m.morph_idx}]'
        elif m.morph_type == 'preset':
            detail = f' → shapes{m.shape_indices}'
        elif m.morph_type == 'shader':
            detail = f' → {m.shader_param}'
        status = '  OK' if (m.x_morph_idx >= 0 or m.y_morph_idx >= 0 or
                            m.morph_idx >= 0 or m.shape_indices or
                            m.shader_param) else '  UNMAPPED'
        print(f"  [{i:2d}] {m.name:30s} ({m.category:8s}/{m.input_type:6s}){detail}{status}")

    geo = get_geometry_input_indices(mapping)
    shader = get_shader_input_indices(mapping)
    print(f"\nGeometry inputs: {len(geo)}, Shader-only inputs: {len(shader)}")

    save_mapping(mapping)
    print(f"Saved to {MAPPING_PATH}")
