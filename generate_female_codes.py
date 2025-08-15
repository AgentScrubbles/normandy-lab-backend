import random
import string
from sliders_config import me1_female_sliders

# Create digit encoding map
def option_chars(n):
    if n <= 9:
        return [str(i) for i in range(1, n+1)]
    else:
        max_letter_index = min(n - 9, 23)  # 'A'..'W' => 23 letters
        return [str(i) for i in range(1, 10)] + list(string.ascii_uppercase[:max_letter_index])


# Prepare encoding map
for slider in me1_female_sliders:
    slider['chars'] = option_chars(slider['options'])

# Key sliders to diversify fully
key_slider_indices = [10, 16, 21, 26, 29, 30, 31, 36]  # zero-based
key_slider_names = [me1_female_sliders[i]['name'] for i in key_slider_indices]

# Generate code
def generate_female_code(biased=False):
    code_chars = []
    for i, slider in enumerate(me1_female_sliders):
        if biased and i in key_slider_indices:
            # Exhaustively loop through key options
            char = random.choice(slider['chars'])
        elif slider['options'] > 20:
            # Sample a smaller subset for big sliders
            char = random.choice(random.sample(slider['chars'], 4))
        else:
            char = random.choice(slider['chars'])
        code_chars.append(char)

    return ''.join(code_chars)

# Generate N codes
def generate_code_set(n=1000):
    codes = set()
    while len(codes) < n:
        code = generate_female_code(biased=True)
        codes.add(code)
    return sorted(codes)

# Example: generate 100 codes
if __name__ == "__main__":
    codes = generate_code_set(n=10000)
    with open("female_codes.txt", "w") as f:
        for c in codes:
            f.write(c + "\n")
    print(f"✅ Generated {len(codes)} unique female codes.")
