"""
feature_extraction.py — Multi-Modal Feature Detection & Matching
ISRO SIH 2026 | Problem Statement 26166

Implements Stage B + C of the pipeline:
  B) Scale-space & multi-resolution handling
  C) Feature detection & description (multi-modal robust)

Key features:
  • Classical detectors: SIFT, ORB, AKAZE with tuned parameters
  • RIFT-style descriptor: Maximum Index Map of phase congruency
    orientation + histogram of oriented phase — radiation-invariant
  • Grid-based Non-Max Suppression (NMS) for spatially uniform
    keypoint distribution (explicitly required by problem statement)
  • Multi-scale matching with GSD-informed scale prior
  • Coarse-to-fine chaining support (OHRC→TMC→IIRS)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import math


# ---------------------------------------------------------------------------
# Grid-based Non-Max Suppression
# ---------------------------------------------------------------------------

def grid_nms(keypoints: List[cv2.KeyPoint],
             image_shape: Tuple[int, int],
             grid_size: int = 8,
             max_per_cell: int = 50) -> List[cv2.KeyPoint]:
    """
    Grid-based Non-Max Suppression for spatially uniform keypoint distribution.
    
    Divides the image into an NxN grid and keeps only the top-K strongest
    keypoints per cell. This directly satisfies the problem statement's
    requirement for "uniform distribution across the images."
    
    Args:
        keypoints:    List of detected keypoints.
        image_shape:  (height, width) of the image.
        grid_size:    Number of grid divisions per axis (e.g., 8 → 64 cells).
        max_per_cell: Maximum keypoints retained per cell.
    
    Returns:
        Filtered list of keypoints with uniform spatial distribution.
    """
    if not keypoints:
        return []

    h, w = image_shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    # Bucket keypoints into grid cells
    grid: Dict[Tuple[int, int], List[cv2.KeyPoint]] = {}
    for kp in keypoints:
        cx = min(int(kp.pt[0] / cell_w), grid_size - 1)
        cy = min(int(kp.pt[1] / cell_h), grid_size - 1)
        grid.setdefault((cy, cx), []).append(kp)

    # Keep top-K per cell (sorted by response/strength)
    result = []
    for cell_kps in grid.values():
        sorted_kps = sorted(cell_kps, key=lambda k: k.response, reverse=True)
        result.extend(sorted_kps[:max_per_cell])

    return result


def compute_grid_occupancy(keypoints: List[cv2.KeyPoint],
                            image_shape: Tuple[int, int],
                            grid_size: int = 8) -> Dict[str, Any]:
    """
    Compute grid occupancy statistics for keypoint distribution analysis.
    
    Returns occupancy map, entropy, and coefficient of variation — metrics
    for evaluating spatial uniformity of match distribution.
    """
    h, w = image_shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    for kp in keypoints:
        cx = min(int(kp.pt[0] / cell_w), grid_size - 1)
        cy = min(int(kp.pt[1] / cell_h), grid_size - 1)
        grid[cy, cx] += 1

    total = grid.sum()
    if total == 0:
        return {"grid": grid, "entropy": 0.0, "cv": 0.0, "occupied_cells": 0,
                "total_cells": grid_size * grid_size}

    # Occupancy entropy (higher = more uniform)
    probs = grid.flatten().astype(np.float64) / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(grid_size * grid_size)  # perfectly uniform
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Coefficient of variation (lower = more uniform)
    mean_count = np.mean(grid)
    std_count = np.std(grid)
    cv = std_count / mean_count if mean_count > 0 else float('inf')

    occupied = np.count_nonzero(grid)

    return {
        "grid": grid,
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "cv": float(cv),
        "occupied_cells": int(occupied),
        "total_cells": grid_size * grid_size,
        "occupancy_ratio": float(occupied / (grid_size * grid_size)),
    }


# ---------------------------------------------------------------------------
# RIFT-Style Descriptor (Radiation-Invariant Feature Transform)
# ---------------------------------------------------------------------------

def compute_rift_descriptor(image: np.ndarray,
                             keypoints: List[cv2.KeyPoint],
                             pc_orient: Optional[np.ndarray] = None,
                             n_orient_bins: int = 6,
                             n_spatial_bins: int = 4,
                             patch_size: int = 48
                             ) -> Optional[np.ndarray]:
    """
    Compute RIFT-style descriptors using Maximum Index Map (MIM) of
    phase congruency orientation.
    
    RIFT descriptors are designed for multi-modal remote sensing image
    matching. Instead of using intensity gradients (which fail across
    modalities), they use the orientation of phase congruency, which is
    invariant to intensity transformations.
    
    Process:
      1. For each keypoint, extract a local patch.
      2. Compute the Maximum Index Map: for each pixel, the orientation
         bin with maximum phase congruency response.
      3. Build a histogram of MIM indices over spatial sub-regions of
         the patch (similar to SIFT's spatial binning).
    
    Returns:
        descriptors: Nx(n_spatial_bins^2 * n_orient_bins) float32 array
    """
    if not keypoints:
        return None

    half = patch_size // 2
    h, w = image.shape[:2]
    desc_dim = n_spatial_bins * n_spatial_bins * n_orient_bins
    descriptors = []

    # If no precomputed phase congruency orientation, compute gradient orientation
    if pc_orient is None:
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        orient_map = np.arctan2(dy, dx)
        magnitude = np.sqrt(dx**2 + dy**2)
    else:
        orient_map = pc_orient
        magnitude = image.astype(np.float32) / 255.0  # use PC magnitude as weight

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Extract patch (with boundary padding)
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x - half), min(w, x + half)

        if y2 - y1 < 8 or x2 - x1 < 8:
            descriptors.append(np.zeros(desc_dim, dtype=np.float32))
            continue

        orient_patch = orient_map[y1:y2, x1:x2]
        mag_patch = magnitude[y1:y2, x1:x2]

        # Resize to standard size for consistent descriptor
        orient_patch = cv2.resize(orient_patch, (patch_size, patch_size))
        mag_patch = cv2.resize(mag_patch, (patch_size, patch_size))

        # Maximum Index Map: quantize orientation into bins
        mim = ((orient_patch + np.pi) / (2 * np.pi) * n_orient_bins).astype(np.int32)
        mim = np.clip(mim, 0, n_orient_bins - 1)

        # Spatial binning
        descriptor = np.zeros(desc_dim, dtype=np.float32)
        sub_h = patch_size // n_spatial_bins
        sub_w = patch_size // n_spatial_bins

        for sy in range(n_spatial_bins):
            for sx in range(n_spatial_bins):
                sub_mim = mim[sy*sub_h:(sy+1)*sub_h, sx*sub_w:(sx+1)*sub_w]
                sub_mag = mag_patch[sy*sub_h:(sy+1)*sub_h, sx*sub_w:(sx+1)*sub_w]
                idx_base = (sy * n_spatial_bins + sx) * n_orient_bins
                for b in range(n_orient_bins):
                    descriptor[idx_base + b] = np.sum(sub_mag[sub_mim == b])

        # L2 normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
        descriptors.append(descriptor)

    return np.array(descriptors, dtype=np.float32) if descriptors else None


# ---------------------------------------------------------------------------
# Feature Engine — Main Class
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Multi-modal feature detection and matching engine.
    
    Supports classical (SIFT/ORB/AKAZE/RIFT) and provides infrastructure
    for the deep-learning engine (LoFTR/SuperGlue) in Stage C.
    """

    SUPPORTED_DETECTORS = ("sift", "orb", "akaze", "rift")

    def __init__(self,
                 detector: str = "sift",
                 grid_size: int = 8,
                 max_per_cell: int = 80,
                 matcher: str = "flann",
                 ratio_threshold: float = 0.75,
                 ransac_threshold: float = 5.0):
        """
        Args:
            detector:         Feature detector type ('sift', 'orb', 'akaze', 'rift').
            grid_size:        Grid divisions for spatial NMS.
            max_per_cell:     Max keypoints per grid cell.
            matcher:          Matcher type ('flann' or 'bf').
            ratio_threshold:  Lowe's ratio test threshold.
            ransac_threshold: RANSAC inlier threshold (pixels).
        """
        self.detector_name = detector.lower()
        self.grid_size = grid_size
        self.max_per_cell = max_per_cell
        self.matcher_type = matcher.lower()
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self._detector = self._create_detector()

    def _create_detector(self):
        """Initialize the OpenCV feature detector."""
        if self.detector_name == "sift":
            return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03)
        elif self.detector_name == "orb":
            return cv2.ORB_create(nfeatures=10000, scaleFactor=1.2, nlevels=12)
        elif self.detector_name == "akaze":
            return cv2.AKAZE_create()
        elif self.detector_name == "rift":
            # RIFT uses SIFT keypoints with custom descriptors
            return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02)
        else:
            return cv2.SIFT_create()

    # ----- Detection -----

    def detect(self, image: np.ndarray,
               pc_orient: Optional[np.ndarray] = None,
               apply_grid_nms: bool = True
               ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors.
        
        If detector is 'rift', uses SIFT keypoints but computes RIFT
        descriptors from phase congruency orientation instead of intensity
        gradients.
        
        Args:
            image:          Preprocessed grayscale image (uint8).
            pc_orient:      Phase congruency orientation map (for RIFT).
            apply_grid_nms: Whether to apply grid-based NMS for uniformity.
            
        Returns:
            keypoints:    List of cv2.KeyPoint.
            descriptors:  Nx128 (SIFT) or Nx96 (RIFT) float32 array.
        """
        if image is None or image.size == 0:
            return [], None

        # Detect keypoints + compute descriptors
        if self.detector_name == "rift":
            keypoints = self._detector.detect(image, None)
            if apply_grid_nms:
                keypoints = grid_nms(keypoints, image.shape,
                                     self.grid_size, self.max_per_cell)
            descriptors = compute_rift_descriptor(image, keypoints, pc_orient)
        else:
            keypoints, descriptors = self._detector.detectAndCompute(image, None)
            if apply_grid_nms and keypoints:
                # Re-detect descriptors only for kept keypoints
                keypoints = grid_nms(keypoints, image.shape,
                                     self.grid_size, self.max_per_cell)
                if keypoints:
                    keypoints, descriptors = self._detector.compute(image, keypoints)

        return keypoints or [], descriptors

    def detect_multiscale(self, image: np.ndarray,
                          scales: List[float] = None,
                          pc_orient: Optional[np.ndarray] = None
                          ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extract features at multiple scales for scale-space robustness.
        
        Scales keypoint coordinates back to the original image coordinate
        system after detection at each scale.
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]

        h, w = image.shape[:2]
        all_kps = []
        all_descs = []

        for scale in scales:
            sh, sw = int(h * scale), int(w * scale)
            if sh < 16 or sw < 16:
                continue
            scaled_img = cv2.resize(image, (sw, sh))

            scaled_orient = None
            if pc_orient is not None:
                scaled_orient = cv2.resize(pc_orient, (sw, sh))

            kps, descs = self.detect(scaled_img, scaled_orient, apply_grid_nms=False)

            # Map keypoints back to original coordinates
            for kp in kps:
                kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                kp.size = kp.size / scale
                all_kps.append(kp)

            if descs is not None:
                all_descs.append(descs)

        if all_descs:
            combined_descs = np.vstack(all_descs)
        else:
            combined_descs = None

        # Apply grid NMS to the combined set
        all_kps = grid_nms(all_kps, image.shape, self.grid_size, self.max_per_cell)

        # Re-index descriptors to match kept keypoints
        # (simplified: just detect on kept keypoints if possible)
        if combined_descs is not None and len(all_kps) > 0:
            _, combined_descs = self._detector.compute(image, all_kps)

        return all_kps, combined_descs

    # ----- Matching -----

    def match(self, desc1: np.ndarray, desc2: np.ndarray
              ) -> List[cv2.DMatch]:
        """
        Match descriptors using Lowe's ratio test.
        
        Returns list of good matches (after ratio test, before RANSAC).
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        matcher = self._create_matcher(desc1)
        try:
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good.append(m)
        return good

    def match_and_filter(self, kp1: List[cv2.KeyPoint], desc1: np.ndarray,
                         kp2: List[cv2.KeyPoint], desc2: np.ndarray,
                         method: str = "usac_magsac"
                         ) -> Dict[str, Any]:
        """
        Full matching pipeline: ratio test → geometric verification.
        
        Supports RANSAC, MAGSAC++, and USAC methods.
        
        Returns dict with:
          'all_matches'     — pre-RANSAC matches (after ratio test)
          'inlier_matches'  — post-RANSAC inlier matches
          'outlier_matches' — rejected matches
          'src_pts'         — inlier source points Nx1x2
          'dst_pts'         — inlier destination points Nx1x2
          'homography'      — estimated homography matrix (or None)
          'inlier_mask'     — boolean mask of inliers
          'inlier_count'    — number of inliers
          'inlier_ratio'    — inliers / total matches
        """
        all_matches = self.match(desc1, desc2)

        if len(all_matches) < 4:
            return {
                "all_matches": all_matches, "inlier_matches": [],
                "outlier_matches": all_matches,
                "src_pts": None, "dst_pts": None,
                "homography": None, "inlier_mask": None,
                "inlier_count": 0, "inlier_ratio": 0.0,
            }

        src_pts = np.float32([kp1[m.queryIdx].pt for m in all_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in all_matches]).reshape(-1, 1, 2)

        # Geometric verification
        usac_method = self._get_ransac_flag(method)
        H, mask = cv2.findHomography(src_pts, dst_pts, usac_method,
                                      self.ransac_threshold)

        if mask is None:
            mask = np.zeros(len(all_matches), dtype=np.uint8)

        mask_flat = mask.flatten().astype(bool)
        inlier_matches = [all_matches[i] for i in range(len(all_matches)) if mask_flat[i]]
        outlier_matches = [all_matches[i] for i in range(len(all_matches)) if not mask_flat[i]]

        inlier_src = src_pts[mask_flat] if mask_flat.any() else None
        inlier_dst = dst_pts[mask_flat] if mask_flat.any() else None

        return {
            "all_matches": all_matches,
            "inlier_matches": inlier_matches,
            "outlier_matches": outlier_matches,
            "src_pts": inlier_src,
            "dst_pts": inlier_dst,
            "all_src_pts": src_pts,
            "all_dst_pts": dst_pts,
            "homography": H,
            "inlier_mask": mask_flat,
            "inlier_count": int(mask_flat.sum()),
            "inlier_ratio": float(mask_flat.sum() / len(all_matches)) if all_matches else 0.0,
        }

    def _create_matcher(self, desc: np.ndarray):
        """Create appropriate feature matcher based on descriptor type."""
        if desc.dtype == np.uint8:
            if self.matcher_type == "flann":
                index_params = dict(algorithm=6, table_number=12,
                                    key_size=20, multi_probe_level=2)
                search_params = dict(checks=100)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            if self.matcher_type == "flann":
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=100)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    @staticmethod
    def _get_ransac_flag(method: str) -> int:
        """Map method name to OpenCV RANSAC flag."""
        flags = {
            "ransac": cv2.RANSAC,
            "lmeds": cv2.LMEDS,
            "rho": cv2.RHO,
        }
        # MAGSAC++ via USAC framework (OpenCV 4.5+)
        if hasattr(cv2, "USAC_MAGSAC"):
            flags["usac_magsac"] = cv2.USAC_MAGSAC
            flags["usac_accurate"] = cv2.USAC_ACCURATE
            flags["usac_fast"] = cv2.USAC_FAST
        else:
            # Fallback to RANSAC if USAC not available
            flags["usac_magsac"] = cv2.RANSAC
            flags["usac_accurate"] = cv2.RANSAC
            flags["usac_fast"] = cv2.RANSAC

        return flags.get(method.lower(), cv2.RANSAC)

    # ----- Visualization helpers -----

    @staticmethod
    def draw_matches(img1: np.ndarray, kp1: List[cv2.KeyPoint],
                     img2: np.ndarray, kp2: List[cv2.KeyPoint],
                     inlier_matches: List[cv2.DMatch],
                     outlier_matches: Optional[List[cv2.DMatch]] = None,
                     max_display: int = 200) -> np.ndarray:
        """
        Draw match visualization with inliers (green) and outliers (red).
        """
        # Draw outliers first (behind, red)
        if outlier_matches:
            vis = cv2.drawMatches(
                img1, kp1, img2, kp2,
                outlier_matches[:max_display], None,
                matchColor=(60, 60, 180),  # muted red
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        else:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            h = max(h1, h2)
            vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
            vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
            vis[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2

        # Draw inliers on top (bright green/cyan)
        if inlier_matches:
            vis = cv2.drawMatches(
                img1, kp1, img2, kp2,
                inlier_matches[:max_display], vis,
                matchColor=(0, 255, 180),  # cyan-green
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
            )

        return vis

    @staticmethod
    def keypoints_to_array(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """Convert keypoints to Nx2 numpy array of (x, y) coordinates."""
        if not keypoints:
            return np.array([], dtype=np.float32).reshape(0, 2)
        return np.float32([kp.pt for kp in keypoints])

    @staticmethod
    def matches_to_point_pairs(kp1: List[cv2.KeyPoint],
                                kp2: List[cv2.KeyPoint],
                                matches: List[cv2.DMatch]
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert matches to paired point arrays."""
        if not matches:
            return (np.array([], dtype=np.float32).reshape(0, 2),
                    np.array([], dtype=np.float32).reshape(0, 2))
        src = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst = np.float32([kp2[m.trainIdx].pt for m in matches])
        return src, dst
