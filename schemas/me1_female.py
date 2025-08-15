from base_schema import BaseSchema, Slider
import string
import pyautogui
import time
import subprocess
import re
import os
from pathlib import Path
from PIL import ImageGrab
import pyperclip

from window_helpers import add_to_clipboard, setup_window


class ME1_Female_Schema(BaseSchema):

    # --- Configuration ---
    WINDOW_KEYWORD = "Mass Effect"  # Update this
    CODE_FILE = 'female_codes.txt'
    RENDER_WAIT_TIME = 1
    ANGLES = 5
    DRAG_PIXELS = 50
    OUTPUT_DIR = "dataset/me1/female/"
    DEFAULT_CODE = '0' * 37

    WINDOW_LOCATION = (4000, 2000)
    PASTE_BUTTON = (4575, 2925)
    CENTER_FACE = (5475, 2585)
    CAPTURE_BOX = (5035, 2040, 5730, 2925)

    def get_sliders(self) -> list[Slider]:
        return [
            Slider(1, "Facial Structure", 9),
            Slider(2, "Skin Tone", 30),
            Slider(3, "Complexion", 3),
            Slider(4, "Neck Thickness", 33),
            Slider(5, "Face Size", 33),
            Slider(6, "Cheek Width", 33),
            Slider(7, "Cheek Bones", 33),
            Slider(8, "Cheek Gaunt", 33),
            Slider(9, "Ears Size", 33),
            Slider(10, "Ears Orientation", 33),
            Slider(11, "Eye Shape", 9, True),  # KEY
            Slider(12, "Eye Height", 33),
            Slider(13, "Eye Width", 33),
            Slider(14, "Eye Depth", 33),
            Slider(15, "Brow Depth", 33),
            Slider(16, "Brow Height", 33),
            Slider(17, "Iris Color", 18, True),  # KEY
            Slider(18, "Chin Height", 33),
            Slider(19, "Chin Depth", 33),
            Slider(20, "Chin Width", 33),
            Slider(21, "Jaw Width", 33),
            Slider(22, "Mouth Shape", 11, True),  # KEY
            Slider(23, "Mouth Depth", 33),
            Slider(24, "Mouth Width", 33),
            Slider(25, "Mouth Lip Size", 33),
            Slider(26, "Mouth Height", 33),
            Slider(27, "Nose Shape", 13, True),  # KEY
            Slider(28, "Nose Height", 33),
            Slider(29, "Nose Depth", 33),
            Slider(30, "Hair Color", 23, True),  # KEY
            Slider(31, "Hair Style", 21, True),  # KEY
            Slider(32, "Brow", 17, True),        # KEY
            Slider(33, "Brow Color", 21),
            Slider(34, "Blush Color", 33),
            Slider(35, "Lip Color", 34),
            Slider(36, "Eye Shadow Color", 36),
            Slider(37, "Scar", 12, True)         # KEY
        ]

    def get_options_per_index(self) -> list[int]:
        chars = []
        for index, slider in enumerate(self.get_sliders()):
            chars[index] = self.option_chars(slider.options)

        return [slider.options for slider in self.get_sliders()]

    def option_chars(self, n):
        """Return the list of characters available for n options."""
        if n <= 9:
            return [str(i) for i in range(1, n+1)]
        else:
            return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase[:n-9])
        
    def paste_code(self):
        pyautogui.click(x=self.PASTE_BUTTON[0], y=self.PASTE_BUTTON[1])
        pyautogui.sleep(0.1)
        pyautogui.click(x=self.PASTE_BUTTON[0], y=self.PASTE_BUTTON[1])
        pyautogui.sleep(0.1)

    def get_code_file(self) -> str:
        return self.CODE_FILE
    
    def get_output_dir(self):
        return self.OUTPUT_DIR
    
    def setup_window(self):
        return setup_window(self.WINDOW_KEYWORD, self.WINDOW_LOCATION[0], self.WINDOW_LOCATION[1])
    
    def send_code_to_game(self, code):
        add_to_clipboard(code)
        setup_window(self.WINDOW_KEYWORD, self.WINDOW_LOCATION[0], self.WINDOW_LOCATION[1])
        self.paste_code()
        
    def capture_image(self, code):
        try:
            add_to_clipboard(code)
            setup_window(self.WINDOW_KEYWORD, self.WINDOW_LOCATION[0], self.WINDOW_LOCATION[1])
            self.paste_code()
            shot = ImageGrab.grab(bbox=self.CAPTURE_BOX)
            return shot
        except:
            return None