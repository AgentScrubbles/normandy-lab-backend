#!/usr/bin/env python3
"""
Batch evaluation tool for trained models.

Loads a checkpoint and runs inference on dataset images,
comparing predictions against ground truth from filenames.

Usage:
    python -m tools.eval_batch --schema cp2077_female --ckpt runs/cp2077_female_v1/best.pt --count 100
    python -m tools.eval_batch --schema cp2077_female --ckpt runs/cp2077_female_v1/best.pt --all
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from base_schema import get_schema, all_schemas, BaseSchema

# Import schemas to trigger registration
from schemas.me1_female import ME1_Female_Schema
from schemas.me3_female import ME3_Female_Schema
from schemas.me1_male import ME1_Male_Schema
from schemas.me3_male import ME3_Male_Schema
try:
    from schemas.cp2077_female import CP2077_Female_Schema
except ImportError:
    pass

from trainer import CodePredictor


def load_model(schema: BaseSchema, ckpt_path: str, device: torch.device):
    inputs = schema.get_inputs()
    num_classes = [inp.num_classes() for inp in inputs]
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CodePredictor(num_classes).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(model, img_tensor, device):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    outs = model(img_tensor)
    return [out.argmax(dim=1).item() for out in outs]


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate model on dataset images")
    parser.add_argument("--schema", required=True, choices=all_schemas())
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    parser.add_argument("--count", type=int, default=100, help="Number of images to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all images")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Print every prediction")
    args = parser.parse_args()

    schema = get_schema(args.schema)
    inputs = schema.get_inputs()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(schema, args.ckpt, device)
    tf = make_transform(args.img_size)

    data_dir = Path(schema.get_output_dir())
    all_images = sorted(data_dir.glob("*.png"))

    if not all_images:
        print(f"No images found in {data_dir}")
        return

    if not args.all:
        random.seed(args.seed)
        all_images = random.sample(all_images, min(args.count, len(all_images)))

    print(f"Evaluating {len(all_images)} images from {data_dir}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print()

    correct_per_input = [0] * len(inputs)
    within1_per_input = [0] * len(inputs)
    exact_match = 0
    total = 0

    for img_path in all_images:
        stem = img_path.stem
        ground_truth = schema.parse_filename(stem)
        if ground_truth is None:
            print(f"  SKIP: {stem} (unparseable)")
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = tf(img)
        preds = predict(model, img_tensor, device)

        all_correct = True
        for i, (pred, gt) in enumerate(zip(preds, ground_truth)):
            if pred == gt:
                correct_per_input[i] += 1
            else:
                all_correct = False
            if abs(pred - gt) <= 1:
                within1_per_input[i] += 1

        if all_correct:
            exact_match += 1
        total += 1

        if args.verbose:
            match = "OK" if all_correct else "MISS"
            diffs = []
            for i, (p, g) in enumerate(zip(preds, ground_truth)):
                if p != g:
                    diffs.append(f"{inputs[i].name}: {p}!={g}")
            diff_str = f" [{', '.join(diffs)}]" if diffs else ""
            print(f"  [{match}] {stem}{diff_str}")

    print()
    print("=" * 70)
    print(f"RESULTS ({total} images)")
    print("=" * 70)
    print(f"Exact match: {exact_match}/{total} ({100*exact_match/total:.1f}%)")
    print()
    print(f"{'Input':<20} {'Exact':>8} {'Within1':>8} {'Classes':>8}")
    print("-" * 50)
    for i, inp in enumerate(inputs):
        acc = correct_per_input[i] / total
        w1 = within1_per_input[i] / total
        print(f"{inp.name:<20} {100*acc:>7.1f}% {100*w1:>7.1f}% {inp.num_classes():>8}")
    print()

    overall_acc = sum(correct_per_input) / (total * len(inputs))
    overall_w1 = sum(within1_per_input) / (total * len(inputs))
    print(f"{'Overall':<20} {100*overall_acc:>7.1f}% {100*overall_w1:>7.1f}%")


if __name__ == "__main__":
    main()
