# Character Creator

AI-powered character creator that uses computer vision to reverse-engineer game character settings from face photos. Upload a photo, the CNN predicts character creator settings, and (optionally) applies them directly in-game.

## Supported Games

| Game | Schema | Inputs | Mode | Dataset |
|------|--------|--------|------|---------|
| Mass Effect 1 | `me1_female`, `me1_male` | 37 sliders | Code paste | `dataset/female/` |
| Mass Effect 3 | `me3_female`, `me3_male` | 37 sliders | Code paste | `dataset/me3/female/` |
| Dragon Age: Inquisition | `dai_human_female` | 56 (sliders + grids + colors) | Standalone renderer | `dataset/dai/female/` |
| Cyberpunk 2077 | `cp2077_female` | 14 sliders | Preset file (ACU mod) | `dataset/cp2077/female/` |

See [Cyberpunk 2077 setup](cyberpunk2077.md) for mod stack and data collection details.
DA:I uses a standalone renderer (no game needed) — see `renderer/` and `tools/dai_standalone_capture.py`.

## How It Works

1. **Data collection**: Generate random characters in-game, capture face screenshots with settings encoded in filenames
2. **Training**: MobileNetV3-Small CNN with one classification head per input — learns to map face images to character settings
3. **Inference**: Upload a photo, model predicts settings, optionally applies them in-game via automation
4. **Deployment**: Flask API in Docker, checkpoints stored in S3 (DigitalOcean Spaces)

## Quick Start

```bash
# Setup
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Train
./venv/bin/python3 train.py --schema cp2077_female --epochs 60 --batch-size 256 --img-size 384 --out-dir runs/cp2077_female_v2 --device cuda

# Infer
./venv/bin/python3 train.py --schema cp2077_female --infer-image photo.jpg --ckpt runs/cp2077_female_v2/best.pt --device cuda --verbose

# Serve API
flask run
```

## Project Structure

```
train.py              - CLI entry point (train / infer)
trainer.py            - Training loop, dataset, model, inference
app.py                - Flask API server
base_schema.py        - Abstract schema base, registry, calibration
game.py               - Game hierarchy (ME1, ME3, DAI, CP2077)
input_types.py        - Slider, CoordinateGrid, ColorPicker
macro_schema.py       - YAML-based schema definitions
macro_engine.py       - YAML macro execution engine

schemas/              - Game-specific schema implementations
macros/               - YAML macro definitions per game
renderer/             - DA:I standalone renderer (Frostbite mesh/morph/PBR pipeline)
tools/                - Calibration, capture, testing tools
tests/                - Test images and unit tests
calibrations/         - Calibration JSON files per game
dataset/              - Training images (not in repo)
runs/                 - Training checkpoints and logs (not in repo)
```

## Training

See [training.md](training.md) for GPU settings, recommended parameters, and performance observations.

## API

- `GET /schemas` — list available schemas
- `POST /infer/<schema>` — upload image, returns cropped face with predicted code in `X-Code` header
- `GET /metrics` — Prometheus metrics

## Deployment

```bash
docker build -t character-creator .
docker run -p 5000:5000 --env-file .env character-creator
```

Checkpoints are downloaded from S3 on first request. Set `CKPT_<SCHEMA_UPPER>=<s3_key>` in `.env`.
