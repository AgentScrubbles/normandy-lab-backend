import random
import string
from typing import Dict, List

from schemas.me1_female import ME1_Female_Schema
from schemas.me3_female import ME3_Female_Schema
from schemas.me1_male import ME1_Male_Schema

from base_schema import BaseSchema, get_schema

schema = get_schema("me1_male")

# Example: generate 100 codes
if __name__ == "__main__":
    codes = schema.generate_codes(10000)
    with open(schema.get_code_file(), "w") as f:
        for c in codes:
            f.write(c + "\n")
    print(f"✅ Generated {len(codes)} unique codes.")
