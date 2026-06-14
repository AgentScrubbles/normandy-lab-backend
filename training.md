# Training Guide

## Hardware

Tested on an RTX 3090 (24GB VRAM).

## Recommended Settings by GPU

| GPU | VRAM | Batch Size | Image Size | Notes |
|-----|------|-----------|-----------|-------|
| RTX 3090 | 24GB | 256-512 | 384 | ~13GB VRAM at batch 256/img 384 |
| RTX 3080 | 10GB | 128 | 256 | Untested, estimate |
| RTX 3060 | 12GB | 128-256 | 256 | Untested, estimate |

## Key Parameters

- **batch-size**: Larger = faster epochs but fewer gradient updates. Scale epochs accordingly.
  - Batch 64: ~141 steps/epoch (10k dataset) — slow, low GPU utilization on 3090
  - Batch 256: ~35 steps/epoch — good balance
  - Batch 512: ~18 steps/epoch — fastest, needs 60+ epochs to compensate
- **img-size**: Larger = more facial detail for the model. 384 recommended over 256 for subtle features (nose, jaw).
- **epochs**: 30 is not enough — val loss was still improving at epoch 28/30. Use 60+ epochs, especially with larger batch sizes.
- **lr**: Default 1e-3 with CosineAnnealingLR scheduler. No resume support yet — each run starts fresh.

## Training Commands

### Mass Effect
```bash
./venv/bin/python3 train.py --schema me1_female --epochs 24 --batch-size 128 --img-size 256 --out-dir runs/me1/female_v1 --device cuda
```

### Cyberpunk 2077
```bash
# v1 (baseline)
./venv/bin/python3 train.py --schema cp2077_female --epochs 30 --batch-size 64 --img-size 256 --out-dir runs/cp2077_female_v1 --device cuda

# v2 (recommended — better GPU utilization, higher resolution)
./venv/bin/python3 train.py --schema cp2077_female --epochs 60 --batch-size 256 --img-size 384 --out-dir runs/cp2077_female_v2 --device cuda
```

## TensorBoard

```bash
./venv/bin/python3 -m tensorboard.main --logdir runs/ --bind_all
```

## Testing Inference

Run inference on test images with game capture (game must be running in character creator):
```bash
./venv/bin/python3 train.py --schema cp2077_female --infer-image tests/FJones40.jpg --ckpt runs/cp2077_female_v2/best.pt --device cuda --verbose --capture-live-dir tests/results/
```

Batch run on all test images:
```bash
for img in tests/*.jpg; do
  ./venv/bin/python3 train.py --schema cp2077_female --infer-image "$img" \
    --ckpt runs/cp2077_female_v2/best.pt --device cuda \
    --capture-live-dir tests/results/
done
```

Results are saved as side-by-side PNGs (game screenshot + input photo) in `tests/results/`.

## Model Architecture

- **Backbone**: MobileNetV3-Small (lightweight, CPU-friendly at inference)
- **Heads**: One classification head per input position
- **Augmentation**: Random crop, color jitter, rotation, blur, perspective

## Observations (RTX 3090, CP2077 Female)

| Setting | Batch | Img Size | VRAM Used | Epochs | Best Epoch | Notes |
|---------|-------|----------|-----------|--------|------------|-------|
| v1 | 64 | 256 | ~2GB | 30 | 28 | Low GPU util, still improving |
| v2 | 256 | 384 | ~13GB | 60 | TBD | Good GPU util (98%) |

### Per-Input Accuracy Notes
- **Hair/hairstyle**: Trains fastest, most visually distinct
- **Skin tone, skin type**: Converge quickly
- **Nose, mouth, jaw**: Subtle differences, need more epochs and higher img-size
- **Hair color**: Weakest — uses variable LocKeys so training data has inconsistent mappings
