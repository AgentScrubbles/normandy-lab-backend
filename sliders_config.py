import string

def option_chars(n):
    """Return the list of characters available for n options."""
    if n <= 9:
        return [str(i) for i in range(1, n+1)]
    else:
        return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase[:n-9])
    
me1_female_sliders = [
    {"digit": 1, "name": "Facial Structure", "options": 9},
    {"digit": 2, "name": "Skin Tone", "options": 30},
    {"digit": 3, "name": "Complexion", "options": 3},
    {"digit": 4, "name": "Neck Thickness", "options": 33},
    {"digit": 5, "name": "Face Size", "options": 33},
    {"digit": 6, "name": "Cheek Width", "options": 33},
    {"digit": 7, "name": "Cheek Bones", "options": 33},
    {"digit": 8, "name": "Cheek Gaunt", "options": 33},
    {"digit": 9, "name": "Ears Size", "options": 33},
    {"digit": 10, "name": "Ears Orientation", "options": 33},
    {"digit": 11, "name": "Eye Shape", "options": 9},  # KEY
    {"digit": 12, "name": "Eye Height", "options": 33},
    {"digit": 13, "name": "Eye Width", "options": 33},
    {"digit": 14, "name": "Eye Depth", "options": 33},
    {"digit": 15, "name": "Brow Depth", "options": 33},
    {"digit": 16, "name": "Brow Height", "options": 33},
    {"digit": 17, "name": "Iris Color", "options": 18},  # KEY
    {"digit": 18, "name": "Chin Height", "options": 33},
    {"digit": 19, "name": "Chin Depth", "options": 33},
    {"digit": 20, "name": "Chin Width", "options": 33},
    {"digit": 21, "name": "Jaw Width", "options": 33},
    {"digit": 22, "name": "Mouth Shape", "options": 11},  # KEY
    {"digit": 23, "name": "Mouth Depth", "options": 33},
    {"digit": 24, "name": "Mouth Width", "options": 33},
    {"digit": 25, "name": "Mouth Lip Size", "options": 33},
    {"digit": 26, "name": "Mouth Height", "options": 33},
    {"digit": 27, "name": "Nose Shape", "options": 13},  # KEY
    {"digit": 28, "name": "Nose Height", "options": 33},
    {"digit": 29, "name": "Nose Depth", "options": 33},
    {"digit": 30, "name": "Hair Color", "options": 23},  # KEY
    {"digit": 31, "name": "Hair Style", "options": 21},  # KEY
    {"digit": 32, "name": "Brow", "options": 17},        # KEY
    {"digit": 33, "name": "Brow Color", "options": 21},
    {"digit": 34, "name": "Blush Color", "options": 33},
    {"digit": 35, "name": "Lip Color", "options": 34},
    {"digit": 36, "name": "Eye Shadow Color", "options": 36},
    {"digit": 37, "name": "Scar", "options": 12}         # KEY
]

# Add encoding chars to each slider
for slider in me1_female_sliders:
    slider["chars"] = option_chars(slider["options"])

# List of NUM_CLASSES_PER_DIGIT (needed for training)
NUM_CLASSES_PER_DIGIT_ME1_FEMALE = [slider["options"] for slider in me1_female_sliders]
