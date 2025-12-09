import subprocess
import time
import re
from typing import Optional, List, Tuple, NamedTuple

try:
    import pyperclip
except ImportError:
    pyperclip = None


class WindowInfo(NamedTuple):
    """Information about a window."""
    window_id: str
    desktop: int
    x: int
    y: int
    width: int
    height: int
    hostname: str
    title: str


def list_windows() -> List[WindowInfo]:
    """
    List all windows with their geometry.

    Returns:
        List of WindowInfo tuples with id, position, size, and title
    """
    # Use wmctrl -lG to get geometry info
    result = subprocess.run(['wmctrl', '-lG'], capture_output=True, text=True)
    if result.returncode != 0:
        return []

    windows = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        # Format: window_id desktop x y width height hostname title
        # Example: 0x04a00003  0 0    0    1920 1080 hostname Window Title Here
        parts = line.split(None, 7)  # Split on whitespace, max 8 parts
        if len(parts) >= 8:
            try:
                windows.append(WindowInfo(
                    window_id=parts[0],
                    desktop=int(parts[1]),
                    x=int(parts[2]),
                    y=int(parts[3]),
                    width=int(parts[4]),
                    height=int(parts[5]),
                    hostname=parts[6],
                    title=parts[7],
                ))
            except (ValueError, IndexError):
                continue

    return windows


def find_window(title_keyword: str) -> Optional[str]:
    """Find a window by title keyword, return the title."""
    result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True)
    windows = result.stdout.strip().split('\n')
    for window in windows:
        if title_keyword.lower() in window.lower():
            parts = re.split(r'\s+', window, maxsplit=3)
            if len(parts) == 4:
                return parts[3]
    return None


def find_window_info(title_keyword: str) -> Optional[WindowInfo]:
    """
    Find a window by title keyword and return full info including geometry.

    Args:
        title_keyword: Substring to search for in window titles (case-insensitive)

    Returns:
        WindowInfo if found, None otherwise
    """
    for window in list_windows():
        if title_keyword.lower() in window.title.lower():
            return window
    return None


def get_window_geometry(title_keyword: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Get window geometry (x, y, width, height) by title keyword.

    Args:
        title_keyword: Substring to search for in window titles

    Returns:
        (x, y, width, height) tuple or None if not found
    """
    info = find_window_info(title_keyword)
    if info:
        return (info.x, info.y, info.width, info.height)
    return None

def focus_window(window_title: str):
    subprocess.run(['wmctrl', '-a', window_title])

def move_window(window_title: str, x: int, y: int):
    """
    Move the window with the given title to coordinates (x, y) 
    without changing its size.
    """
    # -1 means "keep current width/height"
    geometry = f"0,{x},{y},-1,-1"
    subprocess.run(['wmctrl', '-r', window_title, '-e', geometry])

def add_to_clipboard(text):
    if pyperclip is None:
        raise RuntimeError("pyperclip is required for clipboard operations")
    pyperclip.copy(text)
    time.sleep(0.5)

def setup_window(window_keyword: str, x: int, y: int):
    game_window = find_window(window_keyword)
    
    if not game_window:
        print("Game window not found.")
        raise RuntimeError(f"Window titled {window_keyword} was not found. Is it running?")
    move_window(window_keyword, x, y)
    focus_window(game_window)
    time.sleep(0.1)