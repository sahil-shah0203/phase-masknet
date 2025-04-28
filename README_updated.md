
# PhaseMaskNet: Predictive Phase Mask Generation for Holography

## Overview
This project develops a deep learning model, **PhaseMaskNet**, designed to predict phase masks that can reconstruct cross-sectional images at specified z-heights.  
The goal is to use these phase masks in **metasurfaces** for **holographic projection** and ultimately accelerate **additive manufacturing** workflows.

This repository contains:
- `phase_mask_training.ipynb`: Train PhaseMaskNet on cross-sectional slices.
- `phase_mask_inference.ipynb`: Load a trained model and generate phase masks without retraining.

---

## Setup Instructions
1. Install Python packages:
   ```bash
   pip install torch torchvision matplotlib pillow piq
   ```
2. (Optional) If using Google Colab, mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. (Optional) This project was originally developed on **Google Colab Pro** using a paid GPU instance (Tesla T4 or better recommended).

---

## Training PhaseMaskNet
- Open `phase_mask_training.ipynb`.
- Configure hyperparameters (batch size, learning rate, number of epochs) if needed.
- Place your cross-sectional `.png` slices into a folder (example structure shown inside notebook).
- Update `img_dir` path in the code to your dataset.
- Run all cells to train the model.
- Model weights will be saved to `phase_mask_net.pth`.

**Important Notes:**
- Current training uses low-resolution (128×128) slices for demonstration.
- Training was conducted on **Google Colab Pro GPU instances**; local training may require sufficient VRAM (~8GB+ recommended).
- Future experiments should use 256×256 or 512×512 resolution for sharper outputs.
- Model converges around 30 epochs for current dataset.

---

## Running Inference
- Open `phase_mask_inference.ipynb`.
- Upload the trained `phase_mask_net.pth` file.
- Load your cross-sectional slice image(s).
- Run predictions to generate corresponding phase masks.

This allows generating new phase masks without retraining the model.

---

## Key Parameters
| Parameter | Description | Recommendation |
|:----------|:-------------|:---------------|
| `size` in dataset | Input image size | 128×128 for now; increase to 512×512 for production |
| `batch_size` | Training batch size | Increase if GPU allows |
| `learning_rate` | Learning rate | 1e-3 |
| Model structure | U-Net based | **Do not change** without careful testing |
| Loss function | SSIM loss | Recommended to keep for perceptual quality |

---

## Future Work
- **Increase Input Resolution**: Move to 512×512 or higher fidelity inputs.
- **Train on Complex Shapes**: Use real-world `.stl` cross sections (not just synthetic spheres).
- **Full 3D Training**: Extend model to 3D convolutional architectures.
- **Model Deployment**: Package PhaseMaskNet into a cloud API for faster inference.

---

## Authors
- Developed collaboratively with AI assistance.
- For further work, please credit the original PhaseMaskNet model architecture when modifying.
