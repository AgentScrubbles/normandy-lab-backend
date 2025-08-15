import subprocess
import time
import pyperclip
import re

def find_window(title_keyword: str):
    result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True)
    windows = result.stdout.strip().split('\n')
    for window in windows:
        if title_keyword.lower() in window.lower():
            parts = re.split(r'\s+', window, maxsplit=3)
            if len(parts) == 4:
                return parts[3]
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