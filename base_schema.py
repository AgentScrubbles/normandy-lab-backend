from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import string

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

@dataclass
class Slider():
    digit: int
    name: str
    options: int
    biased: bool = False

class BaseSchema(ABC):

    @abstractmethod
    def get_sliders(self) -> list[Slider]:
        pass

    @abstractmethod
    def get_options_per_index(self) -> list[int]:
        pass

    @abstractmethod
    def get_code_file(self) -> str:
        pass

    @abstractmethod
    def setup_window(self):
        pass

    @abstractmethod
    def send_code_to_game(self, code: str):
        pass

    @abstractmethod
    def get_output_dir(self) -> str:
        pass

    @abstractmethod
    def capture_image(self, code):
        pass

    def generate_code(self, biased=True):
        code_chars = []
        for i, slider in enumerate(self.get_sliders()):
            c = self.option_chars(slider.options)
            if biased and slider.biased:
                # Exhaustively loop through key options
                char = random.choice(c)
            elif slider.options > 20:
                # Sample a smaller subset for big sliders
                char = random.choice(random.sample(c, 4))
            else:
                char = random.choice(c)
            code_chars.append(char)
        return ''.join(code_chars)
    
    def generate_codes(self, n=10000):
        codes = set()
        while len(codes) < n:
            code = self.generate_code(biased=True)
            codes.add(code)
        return sorted(codes)
    
    def option_chars(self, n):
        if n <= 9:
            return [str(i) for i in range(1, n+1)]
        else:
            max_letter_index = min(n - 9, 23)  # 'A'..'W' => 23 letters
            return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase[:max_letter_index])
