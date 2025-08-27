import pyautogui
import time
import subprocess
import re
import os
from pathlib import Path
from PIL import ImageGrab, ImageOps
import pyperclip
import argparse

from schemas.me1_male import ME1_Male_Schema
from schemas.me3_female import ME3_Female_Schema
from schemas.me3_male import ME3_Male_Schema
from base_schema import BaseSchema, get_schema
from window_helpers import setup_window

RENDER_WAIT_TIME = 1


# --- Character Feature Encoding ---
# --- Game Automation ---

def print_update(code, all_times, index, skipped, total):
    completed = index + 1 - skipped
    todo = total - completed - skipped
    sum = 0
    for t in all_times:
        sum += t
    avg_time = sum / completed
    remaining_time = avg_time * todo
    remaining_human = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

    print(f'[{index + 1}/{total}][avg={avg_time}s][est. remaining={remaining_human}]')


def parse_args():
    ap = argparse.ArgumentParser(description="Image Capturer")
    ap.add_argument('--op', type=str, help='Operation to perform, generate_codes or capture_images')
    ap.add_argument('--schema', type=str, help='Schema to use')
    args = ap.parse_args()
    return args

def generate_codes(schema: BaseSchema):
    codes = schema.generate_codes(n=10000)
    os.makedirs(os.path.dirname(schema.get_code_file()), exist_ok=True)
    with open(schema.get_code_file(), "w") as f:
        for c in codes:
            f.write(c + "\n")
    print(f"✅ Generated {len(codes)} unique codes.")

def capture_images(schema: BaseSchema):
    schema.setup_window()
    code_file = schema.get_code_file()
    time.sleep(1)

    with open(code_file, 'r') as file:
        codes = file.readlines()

    times = []
    skipped = 0
    total = len(codes)

    os.makedirs(schema.get_output_dir(), exist_ok=True)
    for idx, code in enumerate(codes):
        start_time = time.time()
        code = code.rstrip('\n')
        image_path = f"{schema.get_output_dir()}/{code}.png"
        if (os.path.exists(image_path)):
            print(f"{idx + 1} of {len(codes)} Skipping code {code} (Already Exists)")
            skipped = skipped + 1
            continue
        code = code.rstrip('\n')
        time.sleep(RENDER_WAIT_TIME)
        
        image = schema.capture_image(code)
        if (image is None):
            raise RuntimeError(f"Could not capture image for code {code}")
        image.save(image_path)
        time.sleep(0.5)
        image_reverse = schema.capture_image(code)
        if (image_reverse is not None):
            image_reverse = ImageOps.mirror(image_reverse)
            image_path_rev = f"{schema.get_output_dir()}/{code}_1.png"
            image_reverse.save(image_path_rev)
        # print(f"{idx + 1} of {len(codes)} Captured code {code}")
        elapsed = time.time() - start_time
        times.append(elapsed)
        print_update(code, times, idx, skipped, total)


if __name__ == "__main__":
    args = parse_args()
    schema = get_schema(args.schema)
    if (args.op == 'generate_codes'):
        generate_codes(schema)
    if (args.op == 'capture_images'):
        capture_images(schema)
