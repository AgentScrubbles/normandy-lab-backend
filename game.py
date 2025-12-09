"""
Game type hierarchy for the character creator system.

This module defines the Game base class and game-specific subclasses.
Each game defines:
- Automation mode (code_paste vs direct_ui)
- Window management configuration
- Common capture/automation logic

Schemas inherit from their game class to get game-level configuration
while defining their own inputs and capture specifics.

Hierarchy:
    Game (abstract)
    ├── MassEffect1
    │   ├── ME1_Female_Schema
    │   └── ME1_Male_Schema
    ├── MassEffect3
    │   ├── ME3_Female_Schema
    │   └── ME3_Male_Schema
    └── DragonAgeInquisition
        ├── DAI_Human_Female
        ├── DAI_Human_Male
        └── ... (other species/genders)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, List, Optional, Tuple
import os

# Conditional imports for GUI automation
_pyautogui = None
_ImageGrab = None
if os.getenv("APP_MODE") != "docker":
    try:
        import pyautogui
        _pyautogui = pyautogui
    except ImportError:
        pass
    try:
        from PIL import ImageGrab
        _ImageGrab = ImageGrab
    except ImportError:
        pass


class AutomationMode(Enum):
    """How the game accepts character configurations."""
    CODE_PASTE = "code_paste"      # Mass Effect style: paste a code string
    DIRECT_UI = "direct_ui"        # Dragon Age style: click UI elements directly


class Game(ABC):
    """
    Abstract base class for games.

    Each game defines its automation mode and window management.
    Schemas inherit from their game to get this configuration.

    Class Attributes:
        GAME_NAME: Human-readable game name
        WINDOW_KEYWORD: String to identify game window
        AUTOMATION_MODE: How to apply character configurations
    """

    # Subclasses must define these
    GAME_NAME: ClassVar[str]
    WINDOW_KEYWORD: ClassVar[str]
    AUTOMATION_MODE: ClassVar[AutomationMode]

    @classmethod
    def uses_code_paste(cls) -> bool:
        """Whether this game uses clipboard code paste."""
        return cls.AUTOMATION_MODE == AutomationMode.CODE_PASTE

    @classmethod
    def uses_direct_ui(cls) -> bool:
        """Whether this game uses direct UI manipulation."""
        return cls.AUTOMATION_MODE == AutomationMode.DIRECT_UI

    @classmethod
    def find_window(cls) -> bool:
        """
        Find and focus the game window.

        Returns:
            True if window found and focused, False otherwise
        """
        # Implementation depends on platform
        # Linux: wmctrl, Windows: pygetwindow, etc.
        try:
            from window_helpers import setup_window
            return setup_window(cls.WINDOW_KEYWORD, 0, 0) is not None
        except ImportError:
            return False


# =============================================================================
# Mass Effect Games (Code Paste Mode)
# =============================================================================

class MassEffect1(Game):
    """Mass Effect 1 game configuration."""

    GAME_NAME = "Mass Effect 1"
    WINDOW_KEYWORD = "Mass Effect"
    AUTOMATION_MODE = AutomationMode.CODE_PASTE


class MassEffect3(Game):
    """Mass Effect 3 game configuration."""

    GAME_NAME = "Mass Effect 3"
    WINDOW_KEYWORD = "Mass Effect 3"
    AUTOMATION_MODE = AutomationMode.CODE_PASTE


# =============================================================================
# Dragon Age Games (Direct UI Mode)
# =============================================================================

class DragonAgeInquisition(Game):
    """Dragon Age: Inquisition game configuration."""

    GAME_NAME = "Dragon Age: Inquisition"
    WINDOW_KEYWORD = "Dragon Age Inquisition"
    AUTOMATION_MODE = AutomationMode.DIRECT_UI

    # Default grid/color picker configurations for DAI
    # These can be overridden in specific schemas if needed
    DEFAULT_GRID_DIVISIONS = 10  # 10x10 = 100 options per coordinate grid
    DEFAULT_PALETTE_SIZE = 36    # 12 hues × 3 saturations for color pickers
