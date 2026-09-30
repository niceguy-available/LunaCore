# Quantitative Evaluation Metrics & Validation Reference

**ChandraAlign — ISRO SIH 2026 (Problem Statement 26166)**

---

## 1. Summary of Evaluated Metrics

ChandraAlign evaluates every registration run across multiple complementary mathematical dimensions:

| Metric | Domain | Target Range | Meaning & Significance |
|---|---|---|---|
| **Quality Score** | Combined | 0 – 100 (Higher is better) | Composite confidence score weighting correlation, SSIM, inlier ratio, uniformity, and sub-pixel accuracy. |
| **Normalized Cross-Correlation (NCC)** | Photometric | 0.0 – 1.0 (Higher is better) | Linear similarity between transformed source and reference pixels over overlapping valid masks. |
| **Structural Similarity (SSIM)** | Perceptual | -1.0 – 1.0 (Higher is better) | High-order structural and topographical fidelity comparing luminance, contrast, and structure. |
| **Root Mean Square Error (RMSE)** | Radiometric | $0 - \infty$ (Lower is better) | Standard root mean squared intensity error: $\sqrt{\frac{1}{N}\sum (I_{ref} - I_{reg})^2}$. |
| **Mean Absolute Error (MAE)** | Radiometric | $0 - \infty$ (Lower is better) | Mean absolute intensity residual: $\frac{1}{N}\sum \|I_{ref} - I_{reg}\|$. |
| **Mutual Information (MI)** | Information Theory | $\ge 0.0$ (Higher is better) | Shared entropy across sensor modalities: $MI(X, Y) = H(X) + H(Y) - H(X, Y)$. |
| **Gradient Correlation** | Edge Structure | -1.0 – 1.0 (Higher is better) | Alignment of Sobel crater rims and ridge features between registered outputs. |
| **Inlier Ratio** | Geometric | 0.0 – 1.0 (Higher is better) | Fraction of candidate matches that satisfy the epipolar/homography geometry via MAGSAC++. |
| **Reprojection RMSE** | Point Accuracy | $< 1.0$ px (Sub-pixel) | Residual geometric distance when projecting source keypoints through transformation matrix $H$. |
| **Sub-Pixel Ratio** | Precision | 0.0 – 1.0 (Higher is better) | Percentage of inlier matches with residual reprojection error $< 1.0$ pixel. |
| **Distribution Entropy & Uniformity** | Spatial Distribution | 0 – 100 (Higher is better) | Shannon entropy across an $8 \times 8$ grid measuring spatial uniformity of match distribution. |

---

## 2. Spatial Uniformity Formula

To evaluate whether matches are distributed uniformly across the entire scene rather than clustered locally on a single crater:

1. The reference image is partitioned into an $M \times M$ grid ($M=8$, total 64 cells).
2. The empirical probability distribution $p_i = \frac{n_i}{N}$ is computed for non-empty cells ($n_i > 0$).
3. **Normalized Spatial Entropy:**
$$H_{norm} = \frac{-\sum_{i=1}^{K} p_i \log_2(p_i)}{\log_2(M^2)}$$
4. **Uniformity Score:**
$$\text{Score}_{uniform} = 0.60 \times H_{norm} \times 100 + 0.40 \times \left(\frac{K_{occupied}}{M^2}\right) \times 100$$
Where $K_{occupied}$ is the number of cells containing at least one valid match.

---

## 3. Export Formats

- **JSON:** Complete metadata, transformation matrix, per-stage latency, and quantitative metrics.
- **CSV:** Point correspondence list (`idx, src_x, src_y, dst_x, dst_y, reproj_error`).
- **GeoJSON:** GeoJSON `FeatureCollection` with `LineString` correspondences for GIS overlay (QGIS, ArcGIS, NASA Web WorldWind).
