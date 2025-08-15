#!/usr/bin/env python3
"""
Train a multi-head classifier that maps a face image -> 37-digit character code.

Dataset layout (as provided):
    dataset/female/{CODE}.png

Each filename's stem is the 37-char code using digits 0-9A-Z.

The model:
- Backbone: MobileNetV3-Small (lightweight, CPU friendly at inference)
- 37 classification heads (one per code position)

Features per position are class-counts from the table the user supplied.
We encode symbols for N options as: ["0","1",...,"9","A","B",...,"Z"][:N]

Usage examples:
    python train_code_predictor.py \
        --data-root dataset/female \
        --epochs 20 --batch-size 64 --img-size 256 --out-dir runs/female_v1 \
        --device cuda  # or cpu

After training, run a quick test inference:
    python train_code_predictor.py --infer-image path/to/photo.jpg --ckpt runs/female_v1/best.pt

The script will also output top-N full-code suggestions (beam search) with probabilities.
"""

import argparse
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms
from PIL import Image
import face_recognition
from torch.utils.tensorboard import SummaryWriter

from base_schema import BaseSchema, all_schemas, get_schema
from schemas.me1_female import ME1_Female_Schema
from schemas.me3_female import ME3_Female_Schema
from trainer import Trainer


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train / infer character code predictor")
    ap.add_argument('--schema', type=str, choices=all_schemas(), help='Which schema to run against', required=True)

    ap.add_argument('--out-dir', type=str, default='runs/female_v1')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Inference
    ap.add_argument('--infer-image', type=str, default=None, help='Path to image to run inference on')
    ap.add_argument('--ckpt', type=str, default=None, help='Checkpoint path for inference')
    ap.add_argument('--beam-width', type=int, default=3)
    ap.add_argument('--topn', type=int, default=5)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--capture-live-dir', type=str, default=None, help='If the game is running, save a live capture of this inference to the location')

    args = ap.parse_args()

    if args.schema is None:
        ap.error('--schema is required')

    # Basic arg validation
    if args.infer_image is None:
        # training mode: need data-root
        # if not args.data_root:
        #     ap.error('--data-root is required for training')
        pass
    else:
        # inference mode: need ckpt
        if not args.ckpt:
            ap.error('--ckpt is required for inference')

    return args


if __name__ == '__main__':
    args = parse_args()

    schema = get_schema(args.schema)

    trainer = Trainer(args.shema)
    if args.infer_image is None:
        trainer.train(args)
    else:
        trainer.infer(args)
