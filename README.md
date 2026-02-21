# openwakeword_training

Custom wake word model training environment using NVIDIA AI Workbench and openWakeWord.
Trains models for deployment to ESPHome-based voice assistants integrated with Home Assistant.

## Get Started (NVIDIA AI Workbench)

1. Open this project in NVIDIA AI Workbench on CABAL
2. Click **Start** on the Project Container — it will build the PyTorch + CUDA environment
3. Once running, click **Open JupyterLab**
4. Open `code/automatic_model_training.ipynb`
5. Run all cells in order — the full pipeline runs automatically
6. Trained model will appear at `models/my_custom_model/<phrase>.onnx`

> **Note:** Training executes remotely on **EVE** (NVIDIA Tesla P4 GPU). No Colab timeouts, no GPU limits.

## Overview

This project provides a containerized, GPU-accelerated training pipeline for custom openWakeWord
models. It runs via NVIDIA AI Workbench with remote execution on EVE, producing trained models
ready for deployment to Home Assistant's openWakeWord integration and ESPHome voice assistant devices.

The first wake word trained with this pipeline is **"Hey Glitch"** — the wake word for Glitch,
a Raspberry Pi-based home assistant satellite with a HAL 9000 voice personality
(*2001: A Space Odyssey*).

## Infrastructure

| Machine | Role |
|---------|------|
| CABAL (Windows PC) | Development, NVIDIA AI Workbench UI |
| EVE (Ubuntu + Tesla P4 GPU) | Remote training execution |
| Aurora3 (Home Assistant) | Deployment target for trained models |
| ESPHome devices | Voice assistant endpoints |

All machines are connected via Tailscale.

## Environment

- **Base image:** PyTorch 2.6 with CUDA 12.6 (Ubuntu 24.04)
- **Applications:** JupyterLab, TensorBoard
- **GPU:** NVIDIA Tesla P4 — 8GB GDDR5 (EVE, 100.106.1.66)

## Training Pipeline

The notebook `code/automatic_model_training.ipynb` handles the full pipeline automatically:

1. **Environment Setup** — git clones, pip installs
2. **Download Data** — AudioSet + FMA background audio
3. **Step 1: Generate synthetic clips** — TTS samples via Piper (~5,000 per phrase)
4. **Step 2: Augment clips** — mix with background noise for robustness
5. **Step 3: Train model** — 25,000 steps on GPU
6. **Output:** `models/my_custom_model/<phrase>.onnx`

### Training Configuration

Only one line needs to change to train a new wake word:

```python
config["target_phrase"] = ["hey glitch"]  # ← change this
config["n_samples"] = 5000
config["n_samples_val"] = 1000
config["steps"] = 25000
config["target_accuracy"] = 0.6
config["target_recall"] = 0.5
```

### Python 3.12 Compatibility Fixes (already applied to notebook)

| Issue | Fix |
|-------|-----|
| `tensorflow-addons` unavailable on 3.12 | Commented out `onnx_tf` install line |
| `piper-phonemize` no wheel for 3.12 | Direct URL to cp312 wheel |
| `os.makedirs` fails if dir exists | Added `exist_ok=True` |
| `torchaudio.set_audio_backend` removed in newer torchaudio | Monkey-patch cell added before Step 1 |

## Output & Deployment

Training produces a `.onnx` model. Convert to TFLite for Home Assistant and ESPHome deployment:

```bash
pip install onnx2tf
onnx2tf -i models/my_custom_model/hey_glitch.onnx -o models/tflite_output -osd
# Use: models/tflite_output/hey_glitch_float32.tflite
```

Deploy `hey_glitch.tflite` to:
- **Home Assistant:** `/share/openwakeword/`
- **ESPHome:** configure via HA openWakeWord integration

## Wake Words

| Phrase | Personality | Voice | Status |
|--------|------------|-------|--------|
| `hey glitch` | Glitch — Pi Zero 2W satellite | HAL 9000 (*2001: A Space Odyssey*) | ✅ Trained |

## Related

- [SpiritualCreations42/glitch](https://github.com/SpiritualCreations42/glitch) — Original Glitch
  satellite project (Wyoming protocol — superseded by ESPHome)
- EVE training files: `~/wakeword-training/` on 100.106.1.66
