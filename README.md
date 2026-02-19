# RealVision - Deepfake Video Detection with Explainable AI (XAI)

**RealVision** is a deep learning project for **binary deepfake video classification** *(Real vs. Fake)* using:
- a **video-only baseline** (EfficientNet-B0 + temporal average pooling), and  
- an **audio-visual model** that fuses video features with audio MFCC features, with temporal modeling via a Transformer encoder.

The project also includes **Explainable AI (XAI)** using **Grad-CAM** to visually explain model decisions.

---

## Dataset
- **FaceForensics++** (Real + Manipulated)
- Binary labels:
  - `0` = Real
  - `1` = Fake (manipulated)

> This repo does **not** include the full dataset files (due to size/licensing).  
> Run the notebook on Kaggle and attach the dataset there.

---

## Key Configuration (from the notebook)
- Video: `k_frames=16`, `img_size=224`
- Audio: `sample_rate=16000`, `audio_seconds=4`, `n_mfcc=40`
- Batch size: `4`
- Training:
  - Baseline epochs: `4`, LR: `3e-4`
  - AV epochs: `6`, LR: `2e-4`
  - Optimizer: AdamW, scheduler: CosineAnnealingLR
  - Early stopping on **Val F1** with `patience=2`

---

## Models

### 1) Baseline (Video-only)
- Backbone: `tf_efficientnet_b0` (via `timm`)
- Frame-wise features → **temporal average pooling** → classifier

### 2) RealVision AV (Video + Audio)
**Video branch**
- EfficientNet-B0 backbone
- Temporal modeling: Transformer Encoder (`nlayers=2`, `nheads=4`)
- Video embedding: 256-d

**Audio branch**
- MFCC input (40 × time)
- Conv1D feature extractor
- Audio embedding: 256-d

**Fusion**
- Concatenation (video + audio) → MLP classifier

---

## Results (as reported in the notebook)
- **Baseline (Video-only)**: Accuracy **71.3%**, F1 **0.710**, AUC **0.82**
- **AV (Video+Audio)**: Accuracy **73.7%**, F1 **0.737**, AUC **0.81**

Observation: AV improves accuracy/F1 while changing the FP/FN trade-off.

---

## Explainability (XAI)
- **Grad-CAM** is applied to inspect which facial regions influence the prediction.
- Implemented with a wrapper to handle temporal (3D) video inputs.

---

## Repository Contents
- `RealVision_Deepfake_Detection.ipynb` — full end-to-end pipeline:
  - environment checks
  - video/audio preprocessing
  - dataset loading + splits
  - training (baseline + AV)
  - evaluation (metrics/curves)
  - Grad-CAM analysis
- `requirements.txt` — minimal dependencies (for local runs)

---

## How to Run (Recommended: Kaggle)
1. Upload this notebook to Kaggle or open it directly in Kaggle.
2. Attach the FaceForensics++ dataset as a Kaggle input.
3. Run cells top-to-bottom.

> If running locally/Colab:  
> - replace Kaggle paths like `/kaggle/input/...`  
> - ensure **ffmpeg** is installed/available (audio extraction)

---

## Authors
- Menalu Chekol
- Odelya Datski
- Zohar Shalom

Course: Deep Learning | Lecturer: Idan Tobis
