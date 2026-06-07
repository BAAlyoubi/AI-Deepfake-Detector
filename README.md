# AI Deepfake Detector

Source code for the study:

**Attention-Guided Spatial-Sequential Feature Learning for Robust Deepfake Detection using MobileNetV2-LSTM Architecture**

This repository implements a lightweight image-level deepfake detector that combines:

- **MobileNetV2** for efficient spatial feature extraction.
- **CBAM attention** to emphasize manipulation-sensitive facial regions such as boundaries, blending artifacts, and texture inconsistencies.
- **LSTM sequential modeling** over the spatial feature map to learn inter-region dependencies.

The reported study trains on an 80 percent training and 20 percent validation split across **39,428 images** and evaluates with accuracy, precision, recall, and F1-score. The paper reports **98.81 percent classification accuracy** on the primary dataset and **94.2 percent average cross-dataset accuracy** on FaceForensics++, Celeb-DF, and DFDC.

## Repository Structure

```text
AI-Deepfake-Detector/
  configs/default.yaml              # Default training configuration
  scripts/prepare_split.py          # 80/20 dataset split helper
  src/deepfake_detector/
    attention.py                    # CBAM attention blocks
    data.py                         # ImageFolder datasets and transforms
    evaluate.py                     # Checkpoint evaluation script
    metrics.py                      # Accuracy, precision, recall, F1
    model.py                        # MobileNetV2-CBAM-LSTM model
    predict.py                      # Single-image inference
    train.py                        # Training loop
    utils.py                        # Shared helpers
  requirements.txt
  README.md
```

## Dataset Layout

The code uses the standard `torchvision.datasets.ImageFolder` layout:

```text
data/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

Class folder names must match `class_names` in `configs/default.yaml`.

To create an 80/20 split from a source folder:

```bash
python scripts/prepare_split.py --source-dir path/to/source --output-dir data --val-ratio 0.2 --copy
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Training

Update `configs/default.yaml` with your dataset paths, then run:

```bash
python -m deepfake_detector.train --config configs/default.yaml
```

The best checkpoint is saved to:

```text
runs/mobilenetv2_cbam_lstm/best_model.pt
```

Training uses:

- AdamW optimizer
- Cross-entropy loss with light label smoothing
- Optional mixed precision on CUDA
- Early stopping on validation F1-score
- Image augmentations for training robustness

## Evaluation

```bash
python -m deepfake_detector.evaluate ^
  --checkpoint runs/mobilenetv2_cbam_lstm/best_model.pt ^
  --data-dir data/test ^
  --output runs/mobilenetv2_cbam_lstm/test_metrics.json
```

The output includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Prediction

```bash
python -m deepfake_detector.predict ^
  --checkpoint runs/mobilenetv2_cbam_lstm/best_model.pt ^
  --image path/to/image.jpg
```

## Model Design

The architecture follows the study design:

1. A pretrained MobileNetV2 backbone extracts compact convolutional features.
2. CBAM applies channel and spatial attention to highlight suspicious visual evidence.
3. The attended feature map is reshaped into a spatial sequence.
4. An LSTM learns dependencies among facial regions.
5. A compact classifier predicts `real` or `fake`.

This design keeps the model efficient enough for resource-limited use while improving robustness beyond plain CNN feature pooling.

## Reproducing the Study Setup

Recommended settings for the primary dataset:

- Total images: 39,428
- Split: 80 percent train, 20 percent validation
- Input size: 224 x 224
- Backbone: ImageNet-pretrained MobileNetV2
- Metrics: accuracy, precision, recall, F1-score
- Cross-dataset tests: FaceForensics++, Celeb-DF, DFDC

Example benchmark flow:

```bash
python -m deepfake_detector.train --config configs/default.yaml
python -m deepfake_detector.evaluate --checkpoint runs/mobilenetv2_cbam_lstm/best_model.pt --data-dir data/test
python -m deepfake_detector.evaluate --checkpoint runs/mobilenetv2_cbam_lstm/best_model.pt --data-dir data/faceforensicspp
python -m deepfake_detector.evaluate --checkpoint runs/mobilenetv2_cbam_lstm/best_model.pt --data-dir data/celebdf
python -m deepfake_detector.evaluate --checkpoint runs/mobilenetv2_cbam_lstm/best_model.pt --data-dir data/dfdc
```

## Citation

If you use this repository, cite the associated study:

```bibtex
@article{attention_mobilenetv2_lstm_deepfake,
  title = {Attention-Guided Spatial-Sequential Feature Learning for Robust Deepfake Detection using MobileNetV2-LSTM Architecture},
  year = {2026},
  note = {Source code repository}
}
```

## Ethical Use

This project is intended for research, media verification, and defensive detection systems. Do not use it to target individuals, evade detection, or support harmful synthetic media workflows.
