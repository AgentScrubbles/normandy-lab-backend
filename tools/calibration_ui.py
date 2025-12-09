"""
UI capture utilities for calibration tool.

Provides interactive mouse position capture using pynput for non-blocking input.
"""

import math
import sys
import time
from typing import Tuple, List, Optional, Callable
from threading import Event

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class CaptureAborted(Exception):
    """Raised when user aborts capture with ESC."""
    pass


class MouseCapture:
    """
    Captures mouse positions with keyboard triggers.

    Uses pynput for non-blocking click detection. User clicks to record
    positions, ESC to cancel, ENTER to finish sequences.

    Supports window bounds filtering to ignore clicks outside the game window.
    """

    def __init__(self, window_bounds: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize mouse capture.

        Args:
            window_bounds: Optional (x, y, width, height) tuple defining the
                          valid click region. Clicks outside this region are ignored.
        """
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput is required for mouse capture. Install with: pip install pynput")

        self._click_event = Event()
        self._key_event = Event()
        self._last_click: Optional[Tuple[int, int]] = None
        self._last_key: Optional[str] = None
        self._mouse_listener: Optional[mouse.Listener] = None
        self._keyboard_listener: Optional[keyboard.Listener] = None
        self._window_bounds = window_bounds

    def set_window_bounds(self, x: int, y: int, width: int, height: int):
        """Set the valid click region (window position and size)."""
        self._window_bounds = (x, y, width, height)

    def clear_window_bounds(self):
        """Clear window bounds, allowing clicks anywhere."""
        self._window_bounds = None

    def _is_in_bounds(self, x: int, y: int) -> bool:
        """Check if a point is within the window bounds."""
        if self._window_bounds is None:
            return True
        wx, wy, ww, wh = self._window_bounds
        return wx <= x < wx + ww and wy <= y < wy + wh

    def _on_click(self, x: int, y: int, button, pressed: bool):
        """Mouse click callback - filters clicks outside window bounds."""
        if pressed and button == mouse.Button.left:
            if self._is_in_bounds(x, y):
                self._last_click = (x, y)
                self._click_event.set()
            # Clicks outside bounds are silently ignored

    def _on_key(self, key):
        """Keyboard callback."""
        try:
            if key == keyboard.Key.esc:
                self._last_key = 'esc'
                self._key_event.set()
                self._click_event.set()  # Also unblock click wait
            elif key == keyboard.Key.f12:
                # F12 to confirm/finish (instead of Enter which games often use)
                self._last_key = 'confirm'
                self._key_event.set()
                self._click_event.set()
        except AttributeError:
            pass

    def _start_listeners(self):
        """Start mouse and keyboard listeners."""
        self._click_event.clear()
        self._key_event.clear()
        self._last_click = None
        self._last_key = None

        self._mouse_listener = mouse.Listener(on_click=self._on_click)
        self._keyboard_listener = keyboard.Listener(on_press=self._on_key)
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def _stop_listeners(self):
        """Stop listeners."""
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _wait_for_click_or_key(self, timeout: float = None) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        """Wait for either a click or key press."""
        self._click_event.wait(timeout)
        return self._last_click, self._last_key

    def capture_single_click(self, prompt: str) -> Tuple[int, int]:
        """
        Wait for user to click and return the position.

        Args:
            prompt: Message to display to user

        Returns:
            (x, y) tuple of clicked position

        Raises:
            CaptureAborted: If user pressed ESC
        """
        print(f"\n  {prompt}")
        print("  [Left-click to record, ESC to cancel]")

        self._start_listeners()
        try:
            while True:
                click, key = self._wait_for_click_or_key()
                if key == 'esc':
                    raise CaptureAborted("User cancelled with ESC")
                if click:
                    print(f"  -> Recorded: ({click[0]}, {click[1]})")
                    return click
                # Reset for next wait
                self._click_event.clear()
                self._last_click = None
        finally:
            self._stop_listeners()

    def capture_two_points(self, prompt1: str, prompt2: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Capture two points (e.g., for bounding box corners or slider track).

        Args:
            prompt1: Prompt for first point
            prompt2: Prompt for second point

        Returns:
            Tuple of two (x, y) positions
        """
        p1 = self.capture_single_click(prompt1)
        p2 = self.capture_single_click(prompt2)
        return (p1, p2)

    def capture_bounding_box(self, prompt_tl: str = "Click TOP-LEFT corner",
                             prompt_br: str = "Click BOTTOM-RIGHT corner") -> Tuple[int, int, int, int]:
        """
        Capture a bounding box (x1, y1, x2, y2).

        Returns:
            (x1, y1, x2, y2) tuple normalized so x1 < x2 and y1 < y2
        """
        p1, p2 = self.capture_two_points(prompt_tl, prompt_br)

        # Normalize to ensure top-left < bottom-right
        x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
        y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])

        return (x1, y1, x2, y2)

    def capture_center_and_radius(self, prompt_center: str = "Click CENTER of wheel",
                                  prompt_edge: str = "Click EDGE of wheel") -> Tuple[Tuple[int, int], int]:
        """
        Capture center point and calculate radius from edge click.

        Args:
            prompt_center: Prompt for center point
            prompt_edge: Prompt for edge point

        Returns:
            Tuple of (center (x,y), radius)
        """
        center = self.capture_single_click(prompt_center)
        edge = self.capture_single_click(prompt_edge)

        radius = int(math.sqrt(
            (edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2
        ))
        print(f"  -> Calculated radius: {radius}")

        return (center, radius)

    def capture_click_sequence(self, prompt: str, min_clicks: int = 1,
                               max_clicks: int = 10) -> List[Tuple[int, int]]:
        """
        Capture a sequence of clicks (for menu navigation paths).

        Args:
            prompt: Message to display
            min_clicks: Minimum required clicks
            max_clicks: Maximum allowed clicks

        Returns:
            List of (x, y) positions

        Raises:
            CaptureAborted: If user pressed ESC
        """
        print(f"\n  {prompt}")
        print(f"  [Click to add points (min {min_clicks}), F12 when done, ESC to cancel]")

        points: List[Tuple[int, int]] = []
        self._start_listeners()

        try:
            while len(points) < max_clicks:
                self._click_event.clear()
                self._key_event.clear()
                self._last_click = None
                self._last_key = None

                click, key = self._wait_for_click_or_key()

                if key == 'esc':
                    raise CaptureAborted("User cancelled with ESC")

                if key == 'confirm':  # F12
                    if len(points) >= min_clicks:
                        break
                    else:
                        print(f"  Need at least {min_clicks} point(s), have {len(points)}")
                        continue

                if click:
                    points.append(click)
                    print(f"  -> Point {len(points)}: ({click[0]}, {click[1]})")

            print(f"  -> Sequence complete: {len(points)} point(s)")
            return points

        finally:
            self._stop_listeners()

    def capture_optional_click(self, prompt: str, timeout: float = 5.0) -> Optional[Tuple[int, int]]:
        """
        Optionally capture a click with timeout.

        Args:
            prompt: Message to display
            timeout: Seconds to wait before returning None

        Returns:
            (x, y) if clicked, None if timeout or ESC
        """
        print(f"\n  {prompt}")
        print(f"  [Click to record, F12 to skip, ESC to cancel] ({timeout}s timeout)")

        self._start_listeners()
        try:
            start = time.time()
            while time.time() - start < timeout:
                self._click_event.clear()
                self._last_click = None
                self._last_key = None

                remaining = timeout - (time.time() - start)
                click, key = self._wait_for_click_or_key(timeout=remaining)

                if key == 'esc':
                    raise CaptureAborted("User cancelled with ESC")
                if key == 'confirm':  # F12
                    print("  -> Skipped")
                    return None
                if click:
                    print(f"  -> Recorded: ({click[0]}, {click[1]})")
                    return click

            print("  -> Timeout, skipped")
            return None

        finally:
            self._stop_listeners()


def continuous_position_display(duration_seconds: float = 60.0):
    """
    Display mouse position continuously (improved gui_debug.py replacement).

    Args:
        duration_seconds: How long to run (default 60s)
    """
    if not PYAUTOGUI_AVAILABLE:
        print("pyautogui not available, cannot display position")
        return

    print(f"Displaying mouse position for {duration_seconds} seconds...")
    print("Press Ctrl+C to stop early.\n")

    start = time.time()
    last_pos = None

    try:
        while time.time() - start < duration_seconds:
            x, y = pyautogui.position()
            current_pos = (x, y)

            # Only update display if position changed
            if current_pos != last_pos:
                elapsed = int(time.time() - start)
                remaining = int(duration_seconds - elapsed)
                print(f"\rPosition: ({x:5d}, {y:5d})  |  Time remaining: {remaining:3d}s  ", end="", flush=True)
                last_pos = current_pos

            time.sleep(0.02)  # 50 Hz update rate

    except KeyboardInterrupt:
        pass

    print("\n\nPosition tracking stopped.")


def get_current_position() -> Tuple[int, int]:
    """Get current mouse position."""
    if PYAUTOGUI_AVAILABLE:
        return pyautogui.position()
    raise RuntimeError("pyautogui not available")


def click_position(x: int, y: int, delay: float = 0.1):
    """Click at a position."""
    if PYAUTOGUI_AVAILABLE:
        pyautogui.click(x, y)
        time.sleep(delay)
    else:
        raise RuntimeError("pyautogui not available")


if __name__ == '__main__':
    # Quick test / standalone position display
    if len(sys.argv) > 1 and sys.argv[1] == '--position':
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        continuous_position_display(duration)
    else:
        print("Usage: python -m tools.calibration_ui --position [duration]")
        print("       Shows mouse position continuously for debugging")
