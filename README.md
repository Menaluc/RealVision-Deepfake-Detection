# RealVision – Deepfake Video Detection with Explainable AI (XAI)

**RealVision** is a deep learning system for **binary deepfake video classification** *(Real vs. Fake)* using:
- a **video-only baseline** (EfficientNet-B0 + temporal average pooling), and  
- a **multimodal audio-visual model** that fuses video features with audio MFCC features, with temporal modeling via a Transformer encoder.

The system also integrates **Explainable AI (XAI)** using **Grad-CAM** to provide visual interpretation of model predictions.

---

## Overview

Deepfakes pose a serious threat to digital trust, enabling misinformation and identity impersonation.  
This project aims to detect manipulated videos while providing **interpretable and trustworthy predictions**.

---

## Dataset

- **FaceForensics++** (Real + Manipulated)
- 2,000 videos (balanced):
  - 1,000 real  
  - 1,000 fake  
- Split:
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%  

> This repo does **not** include the full dataset files (due to size/licensing).  
> Run the notebook on Kaggle and attach the dataset there.

---

## Key Configuration

- Video: `k_frames=16`, `img_size=224`
- Audio: `sample_rate=16000`, `audio_seconds=4`, `n_mfcc=40`
- Batch size: `4`

### Training
- Baseline:
  - Epochs: `4`, LR: `3e-4`
- AV model:
  - Epochs: `6`, LR: `2e-4`

- Optimizer: **AdamW**  
- Scheduler: **CosineAnnealingLR**  
- Early stopping on **Val F1** (`patience=2`)

---

## Models

### Baseline (Video-only)
- Backbone: `tf_efficientnet_b0` (via `timm`)
- Frame-wise feature extraction  
- Temporal average pooling  
- Linear classifier  

### RealVision AV (Multimodal)

**Video branch**
- EfficientNet-B0 backbone  
- Transformer Encoder (`2 layers`, `4 heads`)  
- 256-d embedding  

**Audio branch**
- MFCC input (40 coefficients)  
- Conv1D feature extractor  
- 256-d embedding  

**Fusion**
- Concatenation (video + audio)  
- Fully connected classifier  

---

## Results

| Model            | Accuracy | F1 Score | AUC  |
|------------------|---------|---------|------|
| Baseline         | 71.3%   | 0.710   | 0.82 |
| Multimodal (AV)  | 73.7%   | 0.737   | 0.81 |

### Key Insight
The multimodal model improves accuracy and F1 score, but introduces a trade-off:  
- fewer false positives  
- more missed deepfakes (false negatives)

---

## Evaluation & Visual Results

### Confusion Matrices

#### Baseline Model
<img src="https://github.com/user-attachments/assets/914c6ffb-4ec6-4735-a8f8-6f68ae0d0a74" width="500"/>

#### Multimodal (AV) Model
<img src="https://github.com/user-attachments/assets/43910825-41b8-447e-b715-8211620a6c32" width="500"/>

---

### ROC & Precision-Recall

#### Baseline
<img src="https://github.com/user-attachments/assets/c5a9e76d-441c-419c-a76d-b4d8b9983cac" width="500"/>

#### Multimodal (AV)
<img src="https://github.com/user-attachments/assets/beb702d7-a47e-4228-94ce-20c301807388" width="500"/>

---

## Explainability (XAI)

- Grad-CAM applied to validate model decisions  
- Model focuses on **facial regions rather than background noise**  
- Improves interpretability and trust  

---

## Repository Contents

- `RealVision_Deepfake_Detection.ipynb`  
  - Full pipeline:
    - preprocessing  
    - training  
    - evaluation  
    - Grad-CAM analysis  

- `requirements.txt`

---

## How to Run (Recommended: Kaggle)

1. Upload the notebook to Kaggle  
2. Attach FaceForensics++ dataset  
3. Run all cells  

> For local/Colab:
> - replace `/kaggle/input/...` paths  
> - ensure **ffmpeg** is installed  

---

## Authors

- Menalu Chekol  
- Odelya Datski  
- Zohar Shalom  

Course: Deep Learning | Lecturer: Idan Tobis
