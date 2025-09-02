## Note - Project Shifted to Hugging Face with Demo ![](https://huggingface.co/spaces/xPushpeshx/flare_removal_2.0)
---
title: Flare Removal 2.0
emoji: 🌖
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: wtfpl
---

# 🌌 Lens Flare Removal – Research Implementations

This repository provides **two end-to-end implementations** of state-of-the-art lens flare removal methods:

- **Wu et al., 2020** – *How to Train Neural Networks for Flare Removal*  
- **Zhou et al., 2023 (ICCV)** – *Improving Lens Flare Removal with General-Purpose Pipeline and Multiple Light Sources Recovery*  

Both tackle the same problem — **removing lens flare artifacts** to restore clean, high-quality images — but differ in their **training data synthesis strategies** and **light source recovery mechanisms**.

---

## 📖 Overview

**Lens flare** occurs when strong light sources (e.g., sun, street lights, car headlights) scatter or reflect inside a camera lens, producing streaks, blobs, or haze that degrade both **visual quality** and **downstream computer vision tasks**.

This project explores two influential research directions:

1. **Wu et al. (2020)** – Physics-inspired flare modeling (scattering + reflective) with additive composition and thresholded recovery.  
2. **Zhou et al. (2023)** – ISP/AE-aware synthesis pipeline with smooth, threshold-free multi-source recovery.  

---

## 🏗 Implementations

### 🔹 Wu et al. (2020) – Baseline
**Key Ideas**
- **Flare Modeling:**
  - *Scattering flare* – simulated via Fourier optics with random aperture defects.  
  - *Reflective flare* – captured on a rotation stage with HDR imaging.  
- **Training Pair Synthesis:**  
  ![eq1](https://latex.codecogs.com/png.latex?I_F%20=%20I_0%20+%20F%20+%20\mathcal{N}(0,%20\sigma^2))  
- **Losses:** Combined **image loss** (L1 + perceptual) and **residual flare loss**.  
- **Light Source Handling:** Mask saturated pixels (>0.99), ignore them in training, then feather original source back after inference.  

**Limitations**
- Direct addition causes **global brightening** and **clipping**.  
- Assumes **one dominant light source**.  
- Limited **cross-device generalization**.  

---

### 🔹 Zhou et al. (2023) – Improved Pipeline
**Key Ideas**
- **ISP-Aware Data Synthesis:**
  - Convex blending in **inverse-gamma space** (instead of direct addition).  
  - Sigmoid **weight map** balances scene vs. flare, simulating **auto-exposure darkening**.  
  - Noise sampled from ![eqn](https://latex.codecogs.com/png.latex?0.01%20%5Cchi%5E2).  
- **Convex Blending Equation:**  
  ![eq2](https://latex.codecogs.com/png.latex?I%20=%20(1-W)%20%5Codot%20S%20+%20W%20%5Codot%20F%20+%20%5Cmathcal{N}(0,%20%5Csigma%5E2))  
- **Threshold-Free Light Source Recovery:**  
  ![eq3](https://latex.codecogs.com/png.latex?W%20=%20%5Ctext%7Bnorm%7D(I_%7B%5Ctext%7Billum%7D%7D)%5E%7B%5Calpha%7D)  
  - Typically \(\alpha = 15\).  
  - Recovers **multiple emitters** naturally, without threshold tuning.  
- **Evaluation:** Tested on a **Consumer Electronics dataset** (10 devices, varied flare shapes).  

**Improvements Over Wu et al.**
- More realistic synthesis → avoids distribution shift.  
- Robust recovery → works for **multiple light sources**.  
- Stronger **cross-device generalization**.  
- Better downstream task performance (e.g., object detection after deflaring).  

---

## 🔑 Key Differences

| Aspect                  | Wu et al. (2020) | Zhou et al. (2023) |
|-------------------------|------------------|--------------------|
| **Flare Modeling**      | Physics-driven (sim + captured) | Focus on ISP/AE realism |
| **Pair Synthesis**      | Direct addition in linear space | Convex blending in inverse-gamma space |
| **Light Source Handling** | Threshold mask + paste-back | Smooth power-law recovery (multi-source) |
| **Generalization**      | Good for similar devices | Robust across devices & flare shapes |
| **Focus**               | Flare physics | Camera pipeline + multi-source realism |

---

## ⚙️ Usage

### 📦 Environment Setup
```bash
uv venv
.venv/Scripts/activate
uv add -r requirements.txt


