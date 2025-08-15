from base_schema import BaseSchema, Slider, register_schema
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

@register_schema("me3_female")
class ME3_Female_Schema(BaseSchema):

    # --- Configuration ---
    WINDOW_KEYWORD = "Mass Effect 3"  # Update this
    CODE_FILE = 'dataset/me3/female_codes.txt'
    RENDER_WAIT_TIME = 1
    ANGLES = 5
    DRAG_PIXELS = 50
    OUTPUT_DIR = "dataset/me3/female"
    DEFAULT_CODE = '0' * 37

    WINDOW_LOCATION = (4000, 2000)
    CODE_BUTTON = (4300, 2850)
    PASTE_BUTTON = (4580, 2966)
    CENTER_FACE = (5475, 2585)
    CAPTURE_BOX = (5028, 2052, 5750, 2850)

    def get_sliders(self) -> list[Slider]:
        return [
            Slider(1, "Facial Structure", 9),
            Slider(2, "Skin Tone", 29),
            Slider(3, "Complexion", 3),
            Slider(4, "Neck Thickness", 32),
            Slider(5, "Face Size", 32),
            Slider(6, "Cheek Width", 32),
            Slider(7, "Cheek Bones", 32),
            Slider(8, "Cheek Gaunt", 32),
            Slider(9, "Ears Size", 32),
            Slider(10, "Ears Orientation", 32),
            Slider(11, "Eye Shape", 9),  # KEY
            Slider(12, "Eye Height", 32),
            Slider(13, "Eye Width", 32),
            Slider(14, "Eye Depth", 32),
            Slider(15, "Brow Depth", 32),
            Slider(16, "Brow Height", 32),
            Slider(17, "Iris Color", 17),  # KEY
            Slider(18, "Chin Height", 32),
            Slider(19, "Chin Depth", 32),
            Slider(20, "Chin Width", 32),
            Slider(21, "Jaw Width", 32),
            Slider(22, "Mouth Shape", 10),  # KEY
            Slider(23, "Mouth Depth", 32),
            Slider(24, "Mouth Width", 32),
            Slider(25, "Mouth Lip Size", 32),
            Slider(26, "Mouth Height", 32),
            Slider(27, "Nose Shape", 12),  # KEY
            Slider(28, "Nose Height", 32),
            Slider(29, "Nose Depth", 32),
            Slider(30, "Hair Color", 22),  # KEY
            Slider(31, "Hair Style", 20),  # KEY
            Slider(32, "Brow", 16),        # KEY
            Slider(33, "Brow Color", 20),
            Slider(34, "Blush Color", 32),
            Slider(35, "Lip Color", 33),
            Slider(36, "Eye Shadow Color", 35),
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
        
    def get_code_file(self) -> str:
        return self.CODE_FILE
    
    def setup_window(self):
        return setup_window(self.WINDOW_KEYWORD, self.WINDOW_LOCATION[0], self.WINDOW_LOCATION[1])
    
    def send_code_to_game(self, code):
        add_to_clipboard(code)
        setup_window(self.WINDOW_KEYWORD, self.WINDOW_LOCATION[0], self.WINDOW_LOCATION[1])
        self.paste_code()

    def paste_code(self):
        pyautogui.click(x=self.CODE_BUTTON[0], y=self.CODE_BUTTON[1])
        pyautogui.sleep(0.1)
        pyautogui.click(x=self.PASTE_BUTTON[0], y=self.PASTE_BUTTON[1])
        pyautogui.sleep(0.1)
        pyautogui.click(x=self.PASTE_BUTTON[0], y=self.PASTE_BUTTON[1])
        pyautogui.sleep(0.1)

    def get_output_dir(self):
        return self.OUTPUT_DIR
        
    def capture_image(self, code):
        try:
            self.setup_window()
            self.send_code_to_game(code)
            shot = ImageGrab.grab(bbox=self.CAPTURE_BOX)
            return shot
        except:
            return None