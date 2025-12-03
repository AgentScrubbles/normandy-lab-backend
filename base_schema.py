from abc import ABC, abstractmethod
import random
import string
from typing import List
import warnings

# Import InputType hierarchy from the new module
from input_types import InputType, Slider, CoordinateGrid, ColorPicker, AnyInputType

# Re-export for backward compatibility
__all__ = ['BaseSchema', 'Slider', 'InputType', 'CoordinateGrid', 'ColorPicker',
           'AnyInputType', 'register_schema', 'get_schema', 'all_schemas']

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
    - The inputs available in the character creator (sliders, color pickers, etc.)
    - How to encode/decode character codes
    - Game automation (window management, clicking, capturing)

    Subclasses should implement get_inputs() to define their input types.
    The deprecated get_sliders() method is still supported for backward compatibility.
    """

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

    def apply_input_value(self, input: InputType, value: int):
        """
        Apply a value to an input in the game UI.

        Override in subclasses for game-specific automation.
        Default implementation does nothing (used for code-paste games like ME).
        """
        # Sliders in ME-style games are handled by code paste, not direct clicks
        # Subclasses can override for games requiring direct UI manipulation
        pass
