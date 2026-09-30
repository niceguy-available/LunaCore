"""
registration.py — Geometric Transform Estimation & Image Warping
ISRO SIH 2026 | Problem Statement 26166

Implements Stage D of the pipeline: outlier rejection & transform estimation.

Key additions:
  • MAGSAC++ via OpenCV USAC framework
  • Thin-Plate Spline (TPS) warp for terrain-relief distortion
  • Polynomial warp (2nd/3rd order) for flexible distortion modeling
  • Piecewise affine transform for local deformation handling
  • Composition & inversion of transforms
"""

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Dict, Optional, Any
import math


class TransformEstimator:
    """
    Geometric transformation estimation and image warping.
    
    Supports: homography, affine, rigid, similarity, TPS, polynomial.
    """

    SUPPORTED_METHODS = (
        "homography", "affine", "rigid", "similarity",
        "tps", "polynomial"
    )

    def __init__(self, method: str = "homography"):
        self.method = method

    # ------------------------------------------------------------------
    # Transform estimation
    # ------------------------------------------------------------------

    def estimate(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                 method: Optional[str] = None,
                 ransac_threshold: float = 5.0,
                 ransac_method: str = "usac_magsac"
                 ) -> Dict[str, Any]:
        """
        Estimate geometric transformation from matched point pairs.
        
        Args:
            src_pts: Nx2 source point coordinates.
            dst_pts: Nx2 destination point coordinates.
            method:  Override self.method if provided.
            ransac_threshold: Inlier threshold for RANSAC (pixels).
            ransac_method:    'ransac', 'usac_magsac', 'lmeds', etc.
            
        Returns:
            dict with 'matrix', 'params', 'inlier_mask', 'method', 'error'.
        """
        method = method or self.method
        src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
        dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)

        if len(src) < 4:
            return {"matrix": None, "params": {}, "inlier_mask": None,
                    "method": method, "error": "insufficient_points"}

        if method == "homography":
            return self._estimate_homography(src, dst, ransac_threshold, ransac_method)
        elif method == "affine":
            return self._estimate_affine(src, dst, ransac_threshold)
        elif method == "rigid":
            return self._estimate_rigid(src, dst)
        elif method == "similarity":
            return self._estimate_similarity(src, dst)
        elif method == "tps":
            return self._estimate_tps(src, dst)
        elif method == "polynomial":
            return self._estimate_polynomial(src, dst, order=2)
        else:
            return self._estimate_homography(src, dst, ransac_threshold, ransac_method)

    def _estimate_homography(self, src, dst, threshold, ransac_method):
        """Homography estimation with MAGSAC++ / RANSAC."""
        flag = self._ransac_flag(ransac_method)
        H, mask = cv2.findHomography(src, dst, flag, threshold)
        if H is None:
            return {"matrix": None, "params": {}, "inlier_mask": None,
                    "method": "homography", "error": "estimation_failed"}
        return {
            "matrix": H,
            "params": self._decompose_homography(H),
            "inlier_mask": mask.flatten().astype(bool) if mask is not None else None,
            "method": "homography",
            "error": None,
        }

    def _estimate_affine(self, src, dst, threshold):
        """Full affine transform (6 DOF)."""
        src_r = src.reshape(-1, 1, 2)
        dst_r = dst.reshape(-1, 1, 2)
        M, mask = cv2.estimateAffine2D(src_r, dst_r, ransacReprojThreshold=threshold)
        if M is None:
            return {"matrix": None, "params": {}, "inlier_mask": None,
                    "method": "affine", "error": "estimation_failed"}
        H = np.vstack([M, [0, 0, 1]])
        return {
            "matrix": H,
            "params": self._decompose_affine(M),
            "inlier_mask": mask.flatten().astype(bool) if mask is not None else None,
            "method": "affine",
            "error": None,
        }

    def _estimate_rigid(self, src, dst):
        """Rigid transformation (rotation + translation, no scale)."""
        src_c = np.mean(src, axis=0)
        dst_c = np.mean(dst, axis=0)
        src_centered = src - src_c
        dst_centered = dst - dst_c

        H_mat = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H_mat)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_c - R @ src_c
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t

        angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        return {
            "matrix": T,
            "params": {"translation": tuple(t), "rotation_deg": float(angle), "scale": 1.0},
            "inlier_mask": None,
            "method": "rigid",
            "error": None,
        }

    def _estimate_similarity(self, src, dst):
        """Similarity transformation (rotation + scale + translation)."""
        src_c = np.mean(src, axis=0)
        dst_c = np.mean(dst, axis=0)
        src_centered = src - src_c
        dst_centered = dst - dst_c

        src_scale = np.sqrt(np.sum(src_centered ** 2))
        dst_scale = np.sqrt(np.sum(dst_centered ** 2))
        scale = dst_scale / (src_scale + 1e-8)

        src_norm = src_centered / (src_scale + 1e-8)
        dst_norm = dst_centered / (dst_scale + 1e-8)

        H_mat = src_norm.T @ dst_norm
        U, _, Vt = np.linalg.svd(H_mat)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        T = np.eye(3)
        T[:2, :2] = scale * R
        T[:2, 2] = dst_c - scale * R @ src_c

        angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        return {
            "matrix": T,
            "params": {"translation": tuple(T[:2, 2]), "rotation_deg": float(angle),
                       "scale": float(scale)},
            "inlier_mask": None,
            "method": "similarity",
            "error": None,
        }

    def _estimate_tps(self, src, dst):
        """
        Thin-Plate Spline (TPS) — nonrigid transformation for terrain-relief.
        
        Uses scipy RBF interpolation. For warping, stores the control points
        and the RBF model; warping is done in apply_tps_warp().
        """
        try:
            rbf_x = RBFInterpolator(src, dst[:, 0], kernel='thin_plate_spline', smoothing=0.1)
            rbf_y = RBFInterpolator(src, dst[:, 1], kernel='thin_plate_spline', smoothing=0.1)
            return {
                "matrix": None,  # TPS doesn't have a matrix form
                "tps_model": {"rbf_x": rbf_x, "rbf_y": rbf_y,
                              "src_pts": src, "dst_pts": dst},
                "params": {"type": "tps", "n_control_points": len(src)},
                "inlier_mask": None,
                "method": "tps",
                "error": None,
            }
        except Exception as e:
            return {"matrix": None, "params": {}, "inlier_mask": None,
                    "method": "tps", "error": str(e)}

    def _estimate_polynomial(self, src, dst, order=2):
        """
        Polynomial warp (2nd or 3rd order) for flexible distortion modeling.
        
        Fits: x' = Σ a_ij * x^i * y^j
              y' = Σ b_ij * x^i * y^j
        """
        n = len(src)
        if order == 2:
            # 6 coefficients: 1, x, y, x^2, xy, y^2
            A = np.column_stack([
                np.ones(n), src[:, 0], src[:, 1],
                src[:, 0]**2, src[:, 0]*src[:, 1], src[:, 1]**2
            ])
        elif order == 3:
            A = np.column_stack([
                np.ones(n), src[:, 0], src[:, 1],
                src[:, 0]**2, src[:, 0]*src[:, 1], src[:, 1]**2,
                src[:, 0]**3, src[:, 0]**2*src[:, 1],
                src[:, 0]*src[:, 1]**2, src[:, 1]**3
            ])
        else:
            A = np.column_stack([np.ones(n), src[:, 0], src[:, 1]])

        try:
            coeff_x, _, _, _ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
            coeff_y, _, _, _ = np.linalg.lstsq(A, dst[:, 1], rcond=None)
        except np.linalg.LinAlgError as e:
            return {"matrix": None, "params": {}, "inlier_mask": None,
                    "method": "polynomial", "error": str(e)}

        return {
            "matrix": None,
            "poly_model": {"coeff_x": coeff_x, "coeff_y": coeff_y, "order": order},
            "params": {"type": "polynomial", "order": order},
            "inlier_mask": None,
            "method": "polynomial",
            "error": None,
        }

    # ------------------------------------------------------------------
    # Warping
    # ------------------------------------------------------------------

    def warp(self, image: np.ndarray, transform_result: Dict,
             output_shape: Optional[Tuple[int, int]] = None
             ) -> np.ndarray:
        """
        Apply a transformation to warp an image.
        
        Dispatches to the appropriate warping function based on the
        transform method (matrix-based, TPS, or polynomial).
        """
        if output_shape is None:
            output_shape = image.shape[:2]
        h, w = output_shape

        method = transform_result.get("method", "homography")

        if method in ("homography", "affine", "rigid", "similarity"):
            T = transform_result.get("matrix")
            if T is None:
                return image
            if T.shape == (3, 3):
                return cv2.warpPerspective(image, T, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT)
            else:
                return cv2.warpAffine(image, T[:2], (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)

        elif method == "tps":
            return self._apply_tps_warp(image, transform_result, (h, w))

        elif method == "polynomial":
            return self._apply_poly_warp(image, transform_result, (h, w))

        return image

    def _apply_tps_warp(self, image, transform_result, shape):
        """Apply TPS warp using stored RBF models."""
        tps = transform_result.get("tps_model")
        if tps is None:
            return image
        h, w = shape
        # Build a mesh of destination coordinates
        yy, xx = np.mgrid[0:h:4, 0:w:4]  # subsample for speed
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        mapped_x = tps["rbf_x"](pts).reshape(yy.shape)
        mapped_y = tps["rbf_y"](pts).reshape(yy.shape)

        # Upsample mapping
        map_x = cv2.resize(mapped_x.astype(np.float32), (w, h))
        map_y = cv2.resize(mapped_y.astype(np.float32), (w, h))

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def _apply_poly_warp(self, image, transform_result, shape):
        """Apply polynomial warp."""
        poly = transform_result.get("poly_model")
        if poly is None:
            return image
        h, w = shape
        order = poly["order"]
        cx, cy = poly["coeff_x"], poly["coeff_y"]

        yy, xx = np.mgrid[0:h:4, 0:w:4]
        pts_x = xx.ravel().astype(np.float64)
        pts_y = yy.ravel().astype(np.float64)

        if order == 2:
            A = np.column_stack([
                np.ones_like(pts_x), pts_x, pts_y,
                pts_x**2, pts_x*pts_y, pts_y**2
            ])
        elif order == 3:
            A = np.column_stack([
                np.ones_like(pts_x), pts_x, pts_y,
                pts_x**2, pts_x*pts_y, pts_y**2,
                pts_x**3, pts_x**2*pts_y, pts_x*pts_y**2, pts_y**3
            ])
        else:
            A = np.column_stack([np.ones_like(pts_x), pts_x, pts_y])

        mapped_x = (A @ cx).reshape(yy.shape)
        mapped_y = (A @ cy).reshape(yy.shape)

        map_x = cv2.resize(mapped_x.astype(np.float32), (w, h))
        map_y = cv2.resize(mapped_y.astype(np.float32), (w, h))

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    # ------------------------------------------------------------------
    # Refinement (intensity-based)
    # ------------------------------------------------------------------

    def refine_intensity(self, src_img: np.ndarray, dst_img: np.ndarray,
                         initial_transform: np.ndarray,
                         max_iterations: int = 100
                         ) -> Tuple[np.ndarray, float]:
        """
        Refine transformation using Enhanced Correlation Coefficient (ECC).
        
        ECC is significantly more robust than SSD for cross-modal images.
        """
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    max_iterations, 1e-6)

        T = initial_transform.copy().astype(np.float32)

        try:
            if T.shape == (3, 3):
                warp_mode = cv2.MOTION_HOMOGRAPHY
            else:
                warp_mode = cv2.MOTION_AFFINE
                T = T[:2, :].astype(np.float32)

            _, T_refined = cv2.findTransformECC(
                dst_img.astype(np.float32),
                src_img.astype(np.float32),
                T, warp_mode, criteria,
                inputMask=None, gaussFiltSize=5
            )

            if warp_mode == cv2.MOTION_AFFINE:
                T_refined = np.vstack([T_refined, [0, 0, 1]])

            # Compute final error
            warped = cv2.warpPerspective(src_img, T_refined.astype(np.float64),
                                          (dst_img.shape[1], dst_img.shape[0]))
            mask = (warped > 0) & (dst_img > 0)
            if mask.sum() > 0:
                error = np.sqrt(np.mean(
                    (warped[mask].astype(np.float32) - dst_img[mask].astype(np.float32))**2
                ))
            else:
                error = float('inf')

            return T_refined.astype(np.float64), float(error)

        except cv2.error:
            # ECC failed to converge — return original transform
            return initial_transform, float('inf')

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _decompose_homography(self, H):
        """Extract translation, rotation, and scale from homography."""
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        scale = math.sqrt(a**2 + b**2)
        angle = math.degrees(math.atan2(c, a))
        return {
            "translation": (float(H[0, 2]), float(H[1, 2])),
            "rotation_deg": float(angle),
            "scale": float(scale),
        }

    def _decompose_affine(self, M):
        """Extract params from 2x3 affine matrix."""
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        scale_x = math.sqrt(a**2 + c**2)
        scale_y = math.sqrt(b**2 + d**2)
        angle = math.degrees(math.atan2(c, a))
        return {
            "translation": (float(M[0, 2]), float(M[1, 2])),
            "rotation_deg": float(angle),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "scale": float((scale_x + scale_y) / 2),
        }

    @staticmethod
    def _ransac_flag(method: str) -> int:
        flags = {"ransac": cv2.RANSAC, "lmeds": cv2.LMEDS, "rho": cv2.RHO}
        if hasattr(cv2, "USAC_MAGSAC"):
            flags["usac_magsac"] = cv2.USAC_MAGSAC
            flags["usac_accurate"] = cv2.USAC_ACCURATE
        else:
            flags["usac_magsac"] = cv2.RANSAC
            flags["usac_accurate"] = cv2.RANSAC
        return flags.get(method.lower(), cv2.RANSAC)

    @staticmethod
    def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """Compose two 3x3 transformations: T2 ∘ T1."""
        return T2 @ T1

    @staticmethod
    def invert(T: np.ndarray) -> Optional[np.ndarray]:
        """Invert a 3x3 transformation matrix."""
        try:
            return np.linalg.inv(T)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def compute_reprojection_error(src_pts: np.ndarray, dst_pts: np.ndarray,
                                    H: np.ndarray) -> np.ndarray:
        """
        Compute per-point reprojection error: ||H·src - dst||.
        Returns 1D array of errors for each point pair.
        """
        src_h = np.hstack([src_pts.reshape(-1, 2),
                           np.ones((len(src_pts), 1))]).T  # 3xN
        projected = (H @ src_h).T  # Nx3
        projected = projected[:, :2] / (projected[:, 2:3] + 1e-8)
        errors = np.linalg.norm(projected - dst_pts.reshape(-1, 2), axis=1)
        return errors
