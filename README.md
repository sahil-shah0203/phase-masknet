
# PhaseMaskNet: Predictive Phase Mask Generation for Holography

## Overview
This project develops a deep learning model, **PhaseMaskNet**, designed to predict phase masks that can reconstruct cross-sectional images at specified z-heights.  
The goal is to use these phase masks in **metasurfaces** for **holographic projection** and ultimately accelerate **additive manufacturing** workflows.

This repository contains:
- `phase_mask_training.ipynb`: Train PhaseMaskNet on single-layer cross-sectional slices.
- `phase_mask_inference.ipynb`: Load a trained model and generate phase masks without retraining.
- `phase_mask_multilayer_training.ipynb`: Train PhaseMaskNet to handle multi-layer (3-depth) projection.
- `phase_mask_multilayer_inference.ipynb`: Load and run predictions for multi-layer projections.

---

## Setup Instructions
1. Install Python packages:
   ```bash
   pip install torch torchvision matplotlib pillow piq
   ```
2. (Optional) Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. This project was originally developed on **Google Colab Pro** using a Tesla T4 GPU.

---

## Training PhaseMaskNet (2D Single Slice)
- Open `phase_mask_training.ipynb`.
- Place cross-sectional `.png` images into a folder.
- Adjust `img_dir` path to your dataset.
- Run all cells to train the model.
- Outputs a single phase mask prediction per slice.

---

## Running Inference (2D)
- Open `phase_mask_inference.ipynb`.
- Load the trained `phase_mask_net.pth` weights.
- Input a cross-sectional slice.
- Output a phase mask prediction.

---

## Multi-Layer Phase Mask Prediction
- `phase_mask_multilayer_training.ipynb` trains the model to predict a phase mask that reconstructs three different slices at different z-heights.
- `phase_mask_multilayer_inference.ipynb` tests and evaluates multi-depth holographic projections.

This uses angular spectrum propagation to simulate real-world light behavior.

---

## Key Parameters
| Parameter | Value | Notes |
|:----------|:------|:------|
| Image Size | 512×512 | Future work: 1024×1024 |
| Loss | MSE + Gradient | Structural and intensity matching |
| Model | DeepCGHUNet | U-Net variation |
| 3D Training? | No (multi-depth only) | Full 3D Conv to come later |

---

## Future Work
- Higher resolution phase masks (e.g., 1024×1024).
- Full volumetric (3D) convolution models.
- Real experimental data training.
- Phase mask fabrication and testing.

---

## Authors
- Sahil Shah, Neal Jere, Will Tidwell, Aydin Gomez, Rushil Patange
- Packaged with AI assistance.
