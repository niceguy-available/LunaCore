"""
validation.py — Enhanced Registration Evaluation & Metrics
ISRO SIH 2026 | Problem Statement 26166

Implements Stage F: comprehensive evaluation metrics.

Metrics computed:
  • RMSE of matched points vs. ground-control/check points
  • Inlier count and inlier ratio (post-RANSAC / pre-RANSAC)
  • Match distribution uniformity score (grid-occupancy entropy, CV)
  • Reprojection error histogram
  • Processing time per stage
  • Image-level quality: SSIM, NCC, mutual information, gradient correlation
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from typing import Dict, List, Optional, Tuple, Any
import json
import time
import math


class RegistrationEvaluator:
    """
    Comprehensive evaluation of registration quality.
    
    Computes all metrics required by the SIH problem statement:
    RMSE, inlier statistics, match distribution uniformity,
    reprojection errors, and image similarity metrics.
    """

    def __init__(self, grid_size: int = 8):
        """
        Args:
            grid_size: Grid divisions for match distribution analysis.
        """
        self.grid_size = grid_size

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self,
                 reference: np.ndarray,
                 registered: np.ndarray,
                 src_pts: Optional[np.ndarray] = None,
                 dst_pts: Optional[np.ndarray] = None,
                 all_matches_count: int = 0,
                 inlier_matches_count: int = 0,
                 homography: Optional[np.ndarray] = None,
                 stage_timings: Optional[Dict[str, float]] = None,
                 ground_truth_pts: Optional[Tuple[np.ndarray, np.ndarray]] = None
                 ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Returns a comprehensive metrics dictionary ready for the
        evaluation dashboard and PDF/CSV export.
        """
        metrics: Dict[str, Any] = {}

        # --- Image-level metrics ---
        img_metrics = self.compute_image_metrics(reference, registered)
        metrics["image_metrics"] = img_metrics

        # --- Match statistics ---
        metrics["match_stats"] = {
            "total_matches": int(all_matches_count),
            "inlier_count": int(inlier_matches_count),
            "inlier_ratio": float(inlier_matches_count / max(all_matches_count, 1)),
        }

        # --- Reprojection errors ---
        if src_pts is not None and dst_pts is not None and homography is not None:
            reproj = self.compute_reprojection_errors(src_pts, dst_pts, homography)
            metrics["reprojection"] = reproj
        else:
            metrics["reprojection"] = {
                "rmse": 0.0, "mean": 0.0, "median": 0.0,
                "max": 0.0, "std": 0.0, "errors": [],
                "histogram_bins": [], "histogram_counts": [],
            }

        # --- Match distribution uniformity ---
        if dst_pts is not None:
            uniformity = self.compute_distribution_uniformity(
                dst_pts, reference.shape, self.grid_size
            )
            metrics["distribution"] = uniformity
        else:
            metrics["distribution"] = {
                "entropy": 0.0, "normalized_entropy": 0.0,
                "cv": 0.0, "occupied_cells": 0,
                "total_cells": self.grid_size ** 2,
                "occupancy_ratio": 0.0, "grid": [],
            }

        # --- Ground truth accuracy ---
        if ground_truth_pts is not None:
            gt_src, gt_dst = ground_truth_pts
            if homography is not None:
                gt_metrics = self.compute_ground_truth_accuracy(
                    gt_src, gt_dst, homography
                )
                metrics["ground_truth"] = gt_metrics

        # --- Stage timings ---
        if stage_timings:
            metrics["timing"] = stage_timings
            metrics["timing"]["total"] = sum(stage_timings.values())

        # --- Overall quality score ---
        metrics["quality_score"] = self.compute_quality_score(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Image-level metrics
    # ------------------------------------------------------------------

    def compute_image_metrics(self, reference: np.ndarray,
                               registered: np.ndarray) -> Dict[str, float]:
        """Compute image similarity metrics between reference and registered."""
        if reference.shape != registered.shape:
            # Resize registered to match reference
            registered = cv2.resize(registered,
                                     (reference.shape[1], reference.shape[0]))

        ref_f = reference.astype(np.float32)
        reg_f = registered.astype(np.float32)

        # Mask: only evaluate where both images have data
        mask = (reference > 0) & (registered > 0)
        if mask.sum() < 100:
            return {"rmse": float('inf'), "mae": float('inf'),
                    "correlation": 0.0, "ssim": 0.0,
                    "mutual_info": 0.0, "gradient_corr": 0.0}

        metrics = {}

        # RMSE
        diff = ref_f - reg_f
        metrics["rmse"] = float(np.sqrt(np.mean(diff[mask] ** 2)))

        # MAE
        metrics["mae"] = float(np.mean(np.abs(diff[mask])))

        # Normalized Cross-Correlation
        ref_masked = ref_f[mask]
        reg_masked = reg_f[mask]
        corr = np.corrcoef(ref_masked.flatten(), reg_masked.flatten())
        metrics["correlation"] = float(corr[0, 1]) if not np.isnan(corr[0, 1]) else 0.0

        # SSIM
        try:
            metrics["ssim"] = float(ssim_metric(reference, registered, data_range=255))
        except Exception:
            metrics["ssim"] = 0.0

        # Mutual Information
        metrics["mutual_info"] = float(self._mutual_information(reference, registered))

        # Gradient Correlation
        metrics["gradient_corr"] = float(self._gradient_correlation(reference, registered))

        return metrics

    # ------------------------------------------------------------------
    # Reprojection errors
    # ------------------------------------------------------------------

    def compute_reprojection_errors(self, src_pts: np.ndarray,
                                     dst_pts: np.ndarray,
                                     H: np.ndarray) -> Dict[str, Any]:
        """
        Compute per-point reprojection error: ||H·src - dst||.
        
        Also generates histogram data for the dashboard visualization.
        """
        src = src_pts.reshape(-1, 2)
        dst = dst_pts.reshape(-1, 2)

        # Project source points through homography
        src_h = np.hstack([src, np.ones((len(src), 1))]).T
        projected = (H @ src_h).T
        projected = projected[:, :2] / (projected[:, 2:3] + 1e-8)

        errors = np.linalg.norm(projected - dst, axis=1)

        # Histogram
        max_err = max(float(np.max(errors)), 1.0)
        hist_counts, hist_edges = np.histogram(errors, bins=20, range=(0, max_err))

        return {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
            "min": float(np.min(errors)),
            "std": float(np.std(errors)),
            "errors": errors.tolist(),
            "histogram_bins": ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist(),
            "histogram_counts": hist_counts.tolist(),
            "sub_pixel_count": int(np.sum(errors < 1.0)),
            "sub_pixel_ratio": float(np.sum(errors < 1.0) / len(errors)),
        }

    # ------------------------------------------------------------------
    # Match distribution uniformity
    # ------------------------------------------------------------------

    def compute_distribution_uniformity(self, points: np.ndarray,
                                         image_shape: Tuple[int, ...],
                                         grid_size: int = 8
                                         ) -> Dict[str, Any]:
        """
        Evaluate spatial uniformity of match point distribution.
        
        Uses grid-occupancy entropy and coefficient of variation.
        Higher entropy = more uniform distribution (desired).
        Lower CV = more uniform distribution (desired).
        """
        h, w = image_shape[:2]
        cell_h = h / grid_size
        cell_w = w / grid_size

        pts = points.reshape(-1, 2)
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)

        for pt in pts:
            cx = min(int(pt[0] / cell_w), grid_size - 1)
            cy = min(int(pt[1] / cell_h), grid_size - 1)
            if 0 <= cx < grid_size and 0 <= cy < grid_size:
                grid[cy, cx] += 1

        total = grid.sum()
        if total == 0:
            return {
                "entropy": 0.0, "normalized_entropy": 0.0,
                "cv": float('inf'), "occupied_cells": 0,
                "total_cells": grid_size ** 2,
                "occupancy_ratio": 0.0,
                "grid": grid.tolist(),
                "uniformity_score": 0.0,
            }

        # Entropy
        probs = grid.flatten().astype(np.float64) / total
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
        max_entropy = np.log2(grid_size ** 2)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Coefficient of variation
        mean_count = np.mean(grid)
        std_count = np.std(grid)
        cv = std_count / mean_count if mean_count > 0 else float('inf')

        occupied = int(np.count_nonzero(grid))

        # Combined uniformity score (0-100, higher = more uniform)
        uniformity_score = norm_entropy * 60 + (occupied / (grid_size**2)) * 40

        return {
            "entropy": float(entropy),
            "normalized_entropy": float(norm_entropy),
            "cv": float(cv),
            "occupied_cells": occupied,
            "total_cells": grid_size ** 2,
            "occupancy_ratio": float(occupied / (grid_size ** 2)),
            "grid": grid.tolist(),
            "uniformity_score": float(np.clip(uniformity_score, 0, 100)),
        }

    # ------------------------------------------------------------------
    # Ground truth accuracy
    # ------------------------------------------------------------------

    def compute_ground_truth_accuracy(self, gt_src: np.ndarray,
                                       gt_dst: np.ndarray,
                                       H: np.ndarray) -> Dict[str, float]:
        """
        Compute accuracy against ground-control/check points.
        """
        src = gt_src.reshape(-1, 2)
        dst = gt_dst.reshape(-1, 2)

        src_h = np.hstack([src, np.ones((len(src), 1))]).T
        projected = (H @ src_h).T
        projected = projected[:, :2] / (projected[:, 2:3] + 1e-8)

        errors = np.linalg.norm(projected - dst, axis=1)

        return {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "std_error": float(np.std(errors)),
            "n_checkpoints": len(src),
        }

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    def compute_quality_score(self, metrics: Dict) -> float:
        """
        Compute overall quality score (0-100).
        
        Weighted combination of:
          - Image correlation (25%)
          - SSIM (25%)
          - Inlier ratio (20%)
          - Distribution uniformity (15%)
          - Sub-pixel accuracy ratio (15%)
        """
        score = 0.0

        img = metrics.get("image_metrics", {})
        match = metrics.get("match_stats", {})
        dist = metrics.get("distribution", {})
        reproj = metrics.get("reprojection", {})

        # Correlation: 0-1 → 0-25
        corr = max(0, img.get("correlation", 0))
        score += corr * 25

        # SSIM: -1 to 1 → 0-25
        ssim_val = img.get("ssim", 0)
        score += ((ssim_val + 1) / 2) * 25

        # Inlier ratio: 0-1 → 0-20
        inlier_ratio = match.get("inlier_ratio", 0)
        score += inlier_ratio * 20

        # Distribution uniformity: 0-100 → 0-15
        uniformity = dist.get("uniformity_score", 0)
        score += (uniformity / 100) * 15

        # Sub-pixel ratio: 0-1 → 0-15
        subpix = reproj.get("sub_pixel_ratio", 0)
        score += subpix * 15

        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mutual_information(img1: np.ndarray, img2: np.ndarray,
                             bins: int = 64) -> float:
        """Compute mutual information between two images."""
        h1 = np.histogram(img1.flatten(), bins=bins, range=(0, 255))[0]
        h2 = np.histogram(img2.flatten(), bins=bins, range=(0, 255))[0]
        h12 = np.histogram2d(img1.flatten(), img2.flatten(),
                              bins=bins, range=[[0, 255], [0, 255]])[0]

        h1 = h1.astype(np.float64) + 1e-10
        h2 = h2.astype(np.float64) + 1e-10
        h12 = h12.astype(np.float64) + 1e-10

        h1 /= h1.sum()
        h2 /= h2.sum()
        h12 /= h12.sum()

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if h12[i, j] > 1e-10:
                    mi += h12[i, j] * np.log2(h12[i, j] / (h1[i] * h2[j]))
        return mi

    @staticmethod
    def _gradient_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute correlation of image gradients."""
        gx1 = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
        gx2 = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
        gy2 = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)

        gx_corr = np.corrcoef(gx1.flatten(), gx2.flatten())[0, 1]
        gy_corr = np.corrcoef(gy1.flatten(), gy2.flatten())[0, 1]

        gx_corr = gx_corr if not np.isnan(gx_corr) else 0.0
        gy_corr = gy_corr if not np.isnan(gy_corr) else 0.0

        return (gx_corr + gy_corr) / 2

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_matches_csv(src_pts: np.ndarray, dst_pts: np.ndarray,
                            errors: Optional[np.ndarray] = None,
                            output_path: str = "matches.csv") -> str:
        """Export match points to CSV format."""
        src = src_pts.reshape(-1, 2)
        dst = dst_pts.reshape(-1, 2)
        lines = ["idx,src_x,src_y,dst_x,dst_y,reproj_error"]
        for i in range(len(src)):
            err = errors[i] if errors is not None and i < len(errors) else ""
            lines.append(f"{i},{src[i,0]:.4f},{src[i,1]:.4f},"
                         f"{dst[i,0]:.4f},{dst[i,1]:.4f},{err}")
        csv_text = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(csv_text)
        return csv_text

    @staticmethod
    def export_matches_geojson(src_pts: np.ndarray, dst_pts: np.ndarray,
                                output_path: str = "matches.geojson") -> Dict:
        """Export match points as GeoJSON (useful if georeferenced)."""
        src = src_pts.reshape(-1, 2)
        dst = dst_pts.reshape(-1, 2)
        features = []
        for i in range(len(src)):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(src[i, 0]), float(src[i, 1])],
                        [float(dst[i, 0]), float(dst[i, 1])]
                    ]
                },
                "properties": {"match_id": i}
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)
        return geojson

    @staticmethod
    def metrics_to_summary_text(metrics: Dict) -> str:
        """Format metrics as a readable text summary."""
        lines = ["=" * 60, "  REGISTRATION EVALUATION REPORT", "=" * 60, ""]

        img = metrics.get("image_metrics", {})
        lines.append("Image Quality Metrics:")
        lines.append(f"  RMSE:              {img.get('rmse', 0):.2f}")
        lines.append(f"  MAE:               {img.get('mae', 0):.2f}")
        lines.append(f"  Correlation (NCC): {img.get('correlation', 0):.4f}")
        lines.append(f"  SSIM:              {img.get('ssim', 0):.4f}")
        lines.append(f"  Mutual Info:       {img.get('mutual_info', 0):.4f}")
        lines.append(f"  Gradient Corr:     {img.get('gradient_corr', 0):.4f}")

        match = metrics.get("match_stats", {})
        lines.append("\nMatch Statistics:")
        lines.append(f"  Total matches:     {match.get('total_matches', 0)}")
        lines.append(f"  Inlier count:      {match.get('inlier_count', 0)}")
        lines.append(f"  Inlier ratio:      {match.get('inlier_ratio', 0):.2%}")

        reproj = metrics.get("reprojection", {})
        lines.append("\nReprojection Error:")
        lines.append(f"  RMSE:              {reproj.get('rmse', 0):.4f} px")
        lines.append(f"  Mean:              {reproj.get('mean', 0):.4f} px")
        lines.append(f"  Median:            {reproj.get('median', 0):.4f} px")
        lines.append(f"  Sub-pixel ratio:   {reproj.get('sub_pixel_ratio', 0):.2%}")

        dist = metrics.get("distribution", {})
        lines.append("\nMatch Distribution:")
        lines.append(f"  Uniformity score:  {dist.get('uniformity_score', 0):.1f}/100")
        lines.append(f"  Grid occupancy:    {dist.get('occupied_cells', 0)}/{dist.get('total_cells', 0)}")
        lines.append(f"  Entropy:           {dist.get('entropy', 0):.2f} bits")

        timing = metrics.get("timing", {})
        if timing:
            lines.append("\nProcessing Time:")
            for stage, t in timing.items():
                lines.append(f"  {stage:20s} {t:.3f}s")

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  OVERALL QUALITY SCORE: {metrics.get('quality_score', 0):.1f} / 100")
        lines.append(f"{'=' * 60}")

        return "\n".join(lines)
