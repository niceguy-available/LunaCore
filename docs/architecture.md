# ChandraAlign: Multi-Modal Lunar Image Registration Architecture

**ISRO Smart India Hackathon 2026 | Problem Statement 26166**  
**Theme:** Space Technology  
**Category:** Software (Image Processing & Computer Vision)

---

## 1. System Overview

ChandraAlign is a comprehensive, end-to-end multi-modal image co-registration platform engineered specifically for lunar optical orbital sensors aboard **Chandrayaan-2**:
- **OHRC (Orbiter High Resolution Camera):** ~0.25 m/pixel panchromatic
- **TMC-2 (Terrain Mapping Camera-2):** ~5.0 m/pixel panchromatic stereo triplets
- **IIRS (Imaging Infrared Spectrometer):** ~80.0 m/pixel hyperspectral (0.8–5.0 µm)

```
                       ┌──────────────────────────────────────┐
                       │       Chandrayaan-2 / LRO Inputs     │
                       │   (OHRC / TMC-2 / IIRS / LRO NAC)    │
                       └──────────────────┬───────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │  STAGE A: Radiometric Normalization│
                        │    - Log-Gabor Phase Congruency    │
                        │    - Difference-of-Log (DoL)       │
                        │    - Weber Excitation (WLD)        │
                        └─────────────────┬──────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │  STAGE B: GSD-Scale Space & Pyramid│
                        │    - Sensor Resolution Priors      │
                        │    - Coarse-to-Fine Chaining       │
                        └─────────────────┬──────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │  STAGE C: Feature Detect & Match   │
                        │    - Grid-based Spatial NMS        │
                        │    - SIFT / RIFT MIM Descriptors   │
                        │    - Lowe's 2-NN Ratio Matching    │
                        └─────────────────┬──────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │  STAGE D: Transform Estimation     │
                        │    - USAC / MAGSAC++ Outlier Rej.  │
                        │    - Homography / Affine / TPS     │
                        │    - ECC Intensity Refinement      │
                        └─────────────────┬──────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │  STAGE E: Sub-Pixel Refinement     │
                        │    - Parabolic Peak Fitting        │
                        │    - Lucas-Kanade Sparse Flow      │
                        └─────────────────┬──────────────────┘
                                          │
                        ┌─────────────────▼──────────────────┐
                        │  STAGE F: Evaluation Dashboard     │
                        │    - RMSE / SSIM / NCC / MI        │
                        │    - 8x8 Spatial Uniformity Map    │
                        │    - Reprojection Histogram        │
                        │    - CSV / GeoJSON Point Export    │
                        └────────────────────────────────────┘
```

---

## 2. Key Algorithmic Innovations

### 2.1 Illumination Invariance via Phase Congruency
Standard gradient detectors (Sobel, SIFT DoG) fail when sun angles invert shadows across crater rims. ChandraAlign utilizes a Log-Gabor frequency filterbank to extract **Phase Congruency (PC)**:
$$PC(x, y) = \frac{\sum_o \max(E_o(x, y) - T, 0)}{\sum_o \sum_s A_{s, o}(x, y) + \epsilon}$$
Where $E_o$ is local energy at orientation $o$, $A_{s,o}$ is amplitude at scale $s$, and $T$ is noise threshold. Because phase alignment occurs at geometric edges regardless of illumination intensity, the resulting representation is inherently sun-angle invariant.

### 2.2 Radiation-Invariant Feature Transform (RIFT)
For cross-sensor modalities (e.g., optical reflectance vs. infrared absorption in IIRS), intensity correlations break down. ChandraAlign implements RIFT descriptors using **Maximum Index Maps (MIM)** of phase congruency orientation, building spatial orientation histograms that correlate structure rather than raw radiometric counts.

### 2.3 Grid-Based Spatial Non-Max Suppression (Uniformity)
To fulfill the SIH mandate for *"uniform distribution across the images"*, ChandraAlign implements an $N \times N$ spatial tessellation grid (default $8 \times 8$). Strong features are capped per grid bucket, preventing 90%+ clustering on high-contrast crater boundaries and forcing matches across low-contrast lunar mare plains.

### 2.4 Sub-Pixel Quadric Correlation Peak Refinement
Matched keypoints undergo localized Normalized Cross-Correlation (NCC) with parabolic sub-pixel surface interpolation:
$$\Delta x = \frac{f(x-1, y) - f(x+1, y)}{2(f(x-1, y) - 2f(x, y) + f(x+1, y))}$$
$$\Delta y = \frac{f(x, y-1) - f(x, y+1)}{2(f(x, y-1) - 2f(x, y) + f(x, y+1))}$$
Achieving sub-pixel localization accuracy with residuals $< 0.5$ pixels.

---

## 3. Technology Stack

- **Backend:** Python 3.10+, OpenCV 4.8+ (with USAC MAGSAC++), NumPy, SciPy, Scikit-Image, FastAPI, Uvicorn, SQLite3
- **Frontend:** Next.js 14 (App Router), TypeScript, Tailwind CSS, Server-Sent Events (SSE)
- **CLI Suite:** Standalone Python CLI (`scripts/cli_register.py`) for headless, serverless execution.
