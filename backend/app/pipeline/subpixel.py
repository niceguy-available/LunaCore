"""
subpixel.py — Sub-Pixel Refinement Module
ISRO SIH 2026 | Problem Statement 26166

Implements Stage E: after initial correspondence, refine keypoint locations
to sub-pixel accuracy.

Methods:
  • Parabolic peak fitting on the correlation surface
  • Lucas-Kanade optical flow local refinement
  • Phase correlation for global sub-pixel shift
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import math


class SubPixelRefiner:
    """
    Sub-pixel refinement of matched keypoint locations.
    
    After the initial feature matching pipeline produces integer-pixel or
    coarse floating-point keypoint coordinates, this module refines them
    to sub-pixel accuracy — justifying the "sub-pixel accuracy" claim
    required by the problem statement.
    """

    def __init__(self, method: str = "parabolic"):
        """
        Args:
            method: 'parabolic', 'lucas_kanade', 'phase_correlation', or 'combined'.
        """
        self.method = method

    def refine(self, img_src: np.ndarray, img_dst: np.ndarray,
               src_pts: np.ndarray, dst_pts: np.ndarray,
               patch_radius: int = 15
               ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Refine matched point pairs to sub-pixel accuracy.
        
        Args:
            img_src:      Source image (grayscale uint8).
            img_dst:      Destination/reference image (grayscale uint8).
            src_pts:      Nx2 source points (float32).
            dst_pts:      Nx2 destination points (float32).
            patch_radius: Half-size of local patch for refinement.
            
        Returns:
            refined_src: Nx2 refined source points.
            refined_dst: Nx2 refined destination points.
            stats:       Refinement statistics dict.
        """
        if src_pts is None or len(src_pts) == 0:
            return src_pts, dst_pts, {"method": self.method, "refined": 0}

        src = src_pts.reshape(-1, 2).copy()
        dst = dst_pts.reshape(-1, 2).copy()

        if self.method == "parabolic":
            refined_src, refined_dst, stats = self._parabolic_refine(
                img_src, img_dst, src, dst, patch_radius
            )
        elif self.method == "lucas_kanade":
            refined_src, refined_dst, stats = self._lk_refine(
                img_src, img_dst, src, dst
            )
        elif self.method == "phase_correlation":
            refined_src, refined_dst, stats = self._phase_corr_refine(
                img_src, img_dst, src, dst, patch_radius
            )
        elif self.method == "combined":
            # Phase correlation first (global), then L-K (local)
            refined_src, refined_dst, stats1 = self._phase_corr_refine(
                img_src, img_dst, src, dst, patch_radius
            )
            refined_src, refined_dst, stats2 = self._lk_refine(
                img_src, img_dst, refined_src, refined_dst
            )
            stats = {**stats1, **stats2, "method": "combined"}
        else:
            refined_src, refined_dst = src, dst
            stats = {"method": self.method, "refined": 0}

        return refined_src, refined_dst, stats

    # ------------------------------------------------------------------
    # Parabolic peak fitting
    # ------------------------------------------------------------------

    def _parabolic_refine(self, img_src, img_dst, src_pts, dst_pts,
                           patch_radius):
        """
        Refine correspondences via parabolic (quadratic) peak fitting
        on the normalized cross-correlation surface.
        
        For each match point pair:
          1. Extract a patch around the source keypoint.
          2. Compute NCC template matching in a small search window
             around the destination keypoint.
          3. Fit a 2D parabola to the correlation peak to find the
             sub-pixel maximum.
        """
        refined_dst = dst_pts.copy()
        shifts = []
        r = patch_radius
        search = r + 4  # slightly larger search window

        for i in range(len(src_pts)):
            sx, sy = int(round(src_pts[i, 0])), int(round(src_pts[i, 1]))
            dx, dy = int(round(dst_pts[i, 0])), int(round(dst_pts[i, 1]))

            # Extract template from source
            t_y1 = max(0, sy - r)
            t_y2 = min(img_src.shape[0], sy + r + 1)
            t_x1 = max(0, sx - r)
            t_x2 = min(img_src.shape[1], sx + r + 1)
            template = img_src[t_y1:t_y2, t_x1:t_x2].astype(np.float32)

            if template.shape[0] < 5 or template.shape[1] < 5:
                continue

            # Search region in destination
            s_y1 = max(0, dy - search)
            s_y2 = min(img_dst.shape[0], dy + search + 1)
            s_x1 = max(0, dx - search)
            s_x2 = min(img_dst.shape[1], dx + search + 1)
            search_region = img_dst[s_y1:s_y2, s_x1:s_x2].astype(np.float32)

            if (search_region.shape[0] < template.shape[0] or
                    search_region.shape[1] < template.shape[1]):
                continue

            # NCC template matching
            result = cv2.matchTemplate(search_region, template,
                                        cv2.TM_CCORR_NORMED)

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Parabolic sub-pixel fitting around the peak
            py, px = max_loc[1], max_loc[0]
            if 0 < py < result.shape[0] - 1 and 0 < px < result.shape[1] - 1:
                # Fit parabola in x direction
                fx_m = result[py, px - 1]
                fx_0 = result[py, px]
                fx_p = result[py, px + 1]
                dx_sub = 0.5 * (fx_m - fx_p) / (fx_m - 2*fx_0 + fx_p + 1e-10)

                # Fit parabola in y direction
                fy_m = result[py - 1, px]
                fy_0 = result[py, px]
                fy_p = result[py + 1, px]
                dy_sub = 0.5 * (fy_m - fy_p) / (fy_m - 2*fy_0 + fy_p + 1e-10)

                # Clamp sub-pixel offset
                dx_sub = np.clip(dx_sub, -0.5, 0.5)
                dy_sub = np.clip(dy_sub, -0.5, 0.5)

                refined_x = s_x1 + px + dx_sub + r
                refined_y = s_y1 + py + dy_sub + r
                shift = math.sqrt((refined_x - dst_pts[i, 0])**2 +
                                  (refined_y - dst_pts[i, 1])**2)
                shifts.append(shift)

                refined_dst[i, 0] = refined_x
                refined_dst[i, 1] = refined_y

        stats = {
            "method": "parabolic",
            "refined": len(shifts),
            "mean_shift_px": float(np.mean(shifts)) if shifts else 0.0,
            "max_shift_px": float(np.max(shifts)) if shifts else 0.0,
        }

        return src_pts, refined_dst, stats

    # ------------------------------------------------------------------
    # Lucas-Kanade optical flow refinement
    # ------------------------------------------------------------------

    def _lk_refine(self, img_src, img_dst, src_pts, dst_pts):
        """
        Refine correspondences using Lucas-Kanade sparse optical flow.
        
        OpenCV's cornerSubPix + calcOpticalFlowPyrLK provides sub-pixel
        tracking inherently.
        """
        # First, refine source keypoint positions to sub-pixel
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

        src_refined = cv2.cornerSubPix(
            img_src, src_pts.astype(np.float32),
            winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
        )

        # Use L-K optical flow to refine destination positions
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )

        dst_init = dst_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_flow, status, err = cv2.calcOpticalFlowPyrLK(
            img_src, img_dst,
            src_refined.reshape(-1, 1, 2),
            dst_init, **lk_params
        )

        # Only use successfully tracked points
        refined_dst = dst_pts.copy()
        tracked = 0
        shifts = []
        if status is not None:
            for i in range(len(status)):
                if status[i, 0] == 1:
                    shift = math.sqrt(
                        (dst_flow[i, 0, 0] - dst_pts[i, 0])**2 +
                        (dst_flow[i, 0, 1] - dst_pts[i, 1])**2
                    )
                    if shift < 5.0:  # reject large jumps
                        refined_dst[i] = dst_flow[i, 0]
                        tracked += 1
                        shifts.append(shift)

        stats = {
            "method": "lucas_kanade",
            "refined": tracked,
            "total": len(src_pts),
            "mean_shift_px": float(np.mean(shifts)) if shifts else 0.0,
            "max_shift_px": float(np.max(shifts)) if shifts else 0.0,
        }

        return src_refined.reshape(-1, 2), refined_dst, stats

    # ------------------------------------------------------------------
    # Phase correlation refinement
    # ------------------------------------------------------------------

    def _phase_corr_refine(self, img_src, img_dst, src_pts, dst_pts,
                            patch_radius):
        """
        Phase correlation for local sub-pixel shift estimation.
        
        For each match, extract local patches and compute the phase
        correlation to find the sub-pixel translation between them.
        """
        refined_dst = dst_pts.copy()
        shifts = []
        r = patch_radius

        for i in range(len(src_pts)):
            sx, sy = int(round(src_pts[i, 0])), int(round(src_pts[i, 1]))
            dx, dy = int(round(dst_pts[i, 0])), int(round(dst_pts[i, 1]))

            # Extract patches
            ps = self._safe_patch(img_src, sy, sx, r)
            pd = self._safe_patch(img_dst, dy, dx, r)

            if ps is None or pd is None:
                continue
            if ps.shape != pd.shape:
                continue

            # Phase correlation
            shift, response = cv2.phaseCorrelate(
                ps.astype(np.float64),
                pd.astype(np.float64)
            )

            if response > 0.1:  # confidence threshold
                dx_sub, dy_sub = shift
                if abs(dx_sub) < 3.0 and abs(dy_sub) < 3.0:
                    refined_dst[i, 0] += dx_sub
                    refined_dst[i, 1] += dy_sub
                    shifts.append(math.sqrt(dx_sub**2 + dy_sub**2))

        stats = {
            "method": "phase_correlation",
            "refined": len(shifts),
            "mean_shift_px": float(np.mean(shifts)) if shifts else 0.0,
        }

        return src_pts, refined_dst, stats

    @staticmethod
    def _safe_patch(image, cy, cx, radius):
        """Extract a square patch with boundary checking."""
        h, w = image.shape[:2]
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)
        patch = image[y1:y2, x1:x2]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            return None
        # Ensure even dimensions for FFT
        ph, pw = patch.shape[:2]
        ph = ph - ph % 2
        pw = pw - pw % 2
        return patch[:ph, :pw]
