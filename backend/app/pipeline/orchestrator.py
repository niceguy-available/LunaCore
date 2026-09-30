"""
orchestrator.py — Pipeline Orchestrator with Stage Events
ISRO SIH 2026 | Problem Statement 26166

Refactored from the original pipeline.py to:
  • Emit stage events for real-time progress in the web UI
  • Return intermediate images at each stage
  • Support engine selection: 'classical' | 'deep' | 'hybrid'
  • Record per-stage timing for the evaluation dashboard
  • Export match points as CSV/GeoJSON

The orchestrator drives the 6-stage pipeline:
  A) Preprocessing / illumination normalization
  B) Multi-resolution scale handling
  C) Feature detection & matching
  D) Outlier rejection & transform estimation
  E) Sub-pixel refinement
  F) Evaluation metrics
"""

import cv2
import numpy as np
import time
import json
import base64
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Tuple
from io import BytesIO

from .preprocessing import LunarPreprocessor
from .feature_extraction import FeatureEngine, compute_grid_occupancy
from .registration import TransformEstimator
from .subpixel import SubPixelRefiner
from .validation import RegistrationEvaluator


# Type for stage event callback: (stage_name, status, data_dict)
StageCallback = Callable[[str, str, Dict[str, Any]], None]


def _img_to_base64(img: np.ndarray, max_dim: int = 512) -> str:
    """Encode image as base64 JPEG string for JSON transport."""
    if img is None:
        return ""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    if img.ndim == 2:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    else:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


class RegistrationOrchestrator:
    """
    Master orchestrator for the lunar image registration pipeline.
    
    Coordinates all pipeline stages and emits real-time progress events
    for the web UI.
    """

    STAGES = [
        "preprocessing",
        "feature_detection",
        "feature_matching",
        "transform_estimation",
        "subpixel_refinement",
        "evaluation",
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self._init_components()

    def _default_config(self) -> Dict:
        return {
            "engine": "classical",              # 'classical', 'deep', 'hybrid'
            "illumination_method": "phase_congruency",  # 'phase_congruency', 'dol', 'wld', 'clahe', 'combined'
            "feature_detector": "sift",          # 'sift', 'orb', 'akaze', 'rift'
            "matcher": "flann",                  # 'flann', 'bf'
            "ratio_threshold": 0.75,
            "ransac_method": "usac_magsac",      # 'ransac', 'usac_magsac', 'lmeds'
            "ransac_threshold": 5.0,
            "transform_type": "homography",      # 'homography', 'affine', 'rigid', 'similarity', 'tps', 'polynomial'
            "grid_size": 8,
            "max_per_cell": 80,
            "subpixel_method": "parabolic",      # 'parabolic', 'lucas_kanade', 'phase_correlation', 'combined'
            "refine_intensity": True,
            "multiscale": False,
            "source_sensor": None,
            "reference_sensor": None,
            "debug": False,
        }

    def _init_components(self):
        c = self.config
        self.preprocessor = LunarPreprocessor(
            method=c["illumination_method"],
            debug=c.get("debug", False)
        )
        self.feature_engine = FeatureEngine(
            detector=c["feature_detector"],
            grid_size=c["grid_size"],
            max_per_cell=c["max_per_cell"],
            matcher=c["matcher"],
            ratio_threshold=c["ratio_threshold"],
            ransac_threshold=c["ransac_threshold"],
        )
        self.transform_estimator = TransformEstimator(method=c["transform_type"])
        self.subpixel_refiner = SubPixelRefiner(method=c["subpixel_method"])
        self.evaluator = RegistrationEvaluator(grid_size=c["grid_size"])

    def register(self,
                 source_image: np.ndarray,
                 reference_image: np.ndarray,
                 callback: Optional[StageCallback] = None,
                 source_sensor: Optional[str] = None,
                 reference_sensor: Optional[str] = None,
                 sun_azimuth: Optional[float] = None,
                 sun_elevation: Optional[float] = None,
                 ) -> Dict[str, Any]:
        """
        Run the full registration pipeline.
        
        Args:
            source_image:     Source image (grayscale or BGR, uint8/uint16).
            reference_image:  Reference image.
            callback:         Optional callback for stage progress events.
            source_sensor:    Sensor code for source ('OHRC', 'TMC2', etc.).
            reference_sensor: Sensor code for reference.
            sun_azimuth:      Sun azimuth for shadow estimation.
            sun_elevation:    Sun elevation for shadow estimation.
            
        Returns:
            Comprehensive result dict with all outputs and metrics.
        """
        result: Dict[str, Any] = {
            "status": "running",
            "stages": {},
            "config": self.config,
        }
        stage_timings: Dict[str, float] = {}

        def _emit(stage: str, status: str, data: Dict = None):
            data = data or {}
            result["stages"][stage] = {"status": status, **data}
            if callback:
                callback(stage, status, data)

        total_start = time.time()

        # ── STAGE A: Preprocessing ──────────────────────────────────────
        _emit("preprocessing", "running")
        t0 = time.time()

        src_gray = self._to_gray(source_image)
        ref_gray = self._to_gray(reference_image)

        src_prep = self.preprocessor.preprocess(
            src_gray, sensor_type=source_sensor,
            sun_azimuth=sun_azimuth, sun_elevation=sun_elevation
        )
        ref_prep = self.preprocessor.preprocess(
            ref_gray, sensor_type=reference_sensor
        )

        stage_timings["preprocessing"] = time.time() - t0
        _emit("preprocessing", "complete", {
            "source_shape": list(src_gray.shape),
            "reference_shape": list(ref_gray.shape),
            "method": self.config["illumination_method"],
            "thumbnail_src": _img_to_base64(src_prep["normalized"]),
            "thumbnail_ref": _img_to_base64(ref_prep["normalized"]),
        })

        # ── STAGE B+C: Feature Detection ────────────────────────────────
        _emit("feature_detection", "running")
        t0 = time.time()

        src_norm = src_prep["normalized"]
        ref_norm = ref_prep["normalized"]
        src_pc_orient = src_prep.get("pc_orient")
        ref_pc_orient = ref_prep.get("pc_orient")

        if self.config.get("multiscale", False):
            src_kp, src_desc = self.feature_engine.detect_multiscale(
                src_norm, pc_orient=src_pc_orient
            )
            ref_kp, ref_desc = self.feature_engine.detect_multiscale(
                ref_norm, pc_orient=ref_pc_orient
            )
        else:
            src_kp, src_desc = self.feature_engine.detect(
                src_norm, pc_orient=src_pc_orient
            )
            ref_kp, ref_desc = self.feature_engine.detect(
                ref_norm, pc_orient=ref_pc_orient
            )

        # Grid occupancy of detected keypoints
        src_occupancy = compute_grid_occupancy(src_kp, src_norm.shape, self.config["grid_size"])
        ref_occupancy = compute_grid_occupancy(ref_kp, ref_norm.shape, self.config["grid_size"])

        stage_timings["feature_detection"] = time.time() - t0
        _emit("feature_detection", "complete", {
            "source_keypoints": len(src_kp),
            "reference_keypoints": len(ref_kp),
            "detector": self.config["feature_detector"],
            "source_occupancy": src_occupancy.get("occupancy_ratio", 0),
            "reference_occupancy": ref_occupancy.get("occupancy_ratio", 0),
        })

        if len(src_kp) < 4 or len(ref_kp) < 4:
            result["status"] = "failed"
            result["error"] = "insufficient_keypoints"
            return result

        # ── STAGE C: Feature Matching ───────────────────────────────────
        _emit("feature_matching", "running")
        t0 = time.time()

        match_result = self.feature_engine.match_and_filter(
            src_kp, src_desc, ref_kp, ref_desc,
            method=self.config["ransac_method"]
        )

        stage_timings["feature_matching"] = time.time() - t0

        all_matches = match_result["all_matches"]
        inlier_matches = match_result["inlier_matches"]
        src_pts = match_result["src_pts"]
        dst_pts = match_result["dst_pts"]

        # Generate match visualization
        match_vis = self.feature_engine.draw_matches(
            src_norm, src_kp, ref_norm, ref_kp,
            inlier_matches, match_result["outlier_matches"]
        )

        _emit("feature_matching", "complete", {
            "total_matches": len(all_matches),
            "inlier_count": len(inlier_matches),
            "inlier_ratio": match_result["inlier_ratio"],
            "ransac_method": self.config["ransac_method"],
            "match_visualization": _img_to_base64(match_vis, max_dim=1024),
        })

        if len(inlier_matches) < 4:
            result["status"] = "failed"
            result["error"] = "insufficient_matches"
            return result

        # ── STAGE D: Transform Estimation ───────────────────────────────
        _emit("transform_estimation", "running")
        t0 = time.time()

        transform_result = self.transform_estimator.estimate(
            src_pts, dst_pts,
            method=self.config["transform_type"],
            ransac_threshold=self.config["ransac_threshold"],
            ransac_method=self.config["ransac_method"],
        )

        # Intensity-based refinement (ECC)
        refinement_error = None
        if self.config.get("refine_intensity", True) and transform_result.get("matrix") is not None:
            refined_T, ref_err = self.transform_estimator.refine_intensity(
                src_norm, ref_norm, transform_result["matrix"]
            )
            if ref_err < float('inf'):
                transform_result["matrix"] = refined_T
                refinement_error = ref_err

        stage_timings["transform_estimation"] = time.time() - t0
        _emit("transform_estimation", "complete", {
            "method": transform_result["method"],
            "params": transform_result.get("params", {}),
            "refinement_error": refinement_error,
        })

        if transform_result.get("matrix") is None and transform_result.get("tps_model") is None:
            result["status"] = "failed"
            result["error"] = "transform_estimation_failed"
            return result

        # ── STAGE E: Sub-pixel Refinement ───────────────────────────────
        _emit("subpixel_refinement", "running")
        t0 = time.time()

        refined_src, refined_dst, subpix_stats = self.subpixel_refiner.refine(
            src_norm, ref_norm, src_pts, dst_pts
        )

        stage_timings["subpixel_refinement"] = time.time() - t0
        _emit("subpixel_refinement", "complete", subpix_stats)

        # Apply transformation to warp source → reference space
        registered_image = self.transform_estimator.warp(
            src_gray, transform_result,
            output_shape=ref_gray.shape
        )

        # ── STAGE F: Evaluation ─────────────────────────────────────────
        _emit("evaluation", "running")
        t0 = time.time()

        H = transform_result.get("matrix")
        metrics = self.evaluator.evaluate(
            reference=ref_gray,
            registered=registered_image,
            src_pts=refined_src,
            dst_pts=refined_dst,
            all_matches_count=len(all_matches),
            inlier_matches_count=len(inlier_matches),
            homography=H,
            stage_timings=stage_timings,
        )

        stage_timings["evaluation"] = time.time() - t0
        metrics["timing"] = stage_timings
        metrics["timing"]["total"] = time.time() - total_start

        _emit("evaluation", "complete", {
            "quality_score": metrics["quality_score"],
            "image_metrics": metrics["image_metrics"],
            "match_stats": metrics["match_stats"],
        })

        # ── Assemble final result ───────────────────────────────────────

        # Checkerboard overlay
        checkerboard = self._create_checkerboard(ref_gray, registered_image)

        # Blended overlay
        blended = self._create_blend(ref_gray, registered_image)

        # Match point data for export
        match_points = []
        if refined_src is not None and refined_dst is not None:
            rs = refined_src.reshape(-1, 2)
            rd = refined_dst.reshape(-1, 2)
            for i in range(len(rs)):
                match_points.append({
                    "src_x": float(rs[i, 0]),
                    "src_y": float(rs[i, 1]),
                    "dst_x": float(rd[i, 0]),
                    "dst_y": float(rd[i, 1]),
                })

        result.update({
            "status": "complete",
            "source_image": _img_to_base64(src_gray),
            "reference_image": _img_to_base64(ref_gray),
            "registered_image": _img_to_base64(registered_image),
            "checkerboard": _img_to_base64(checkerboard),
            "blended": _img_to_base64(blended),
            "match_visualization": _img_to_base64(match_vis, max_dim=1024),
            "transformation_matrix": H.tolist() if H is not None else None,
            "transformation_params": transform_result.get("params", {}),
            "refinement_error": refinement_error,
            "subpixel_stats": subpix_stats,
            "metrics": metrics,
            "match_points": match_points,
            "keypoint_counts": {
                "source": len(src_kp),
                "reference": len(ref_kp),
            },
            "processing_time": time.time() - total_start,
        })

        # Store raw images for file saving (not serialized)
        result["_raw_registered"] = registered_image
        result["_raw_source"] = src_gray
        result["_raw_reference"] = ref_gray
        result["_raw_match_vis"] = match_vis

        return result

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype == np.uint16:
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return image

    @staticmethod
    def _create_checkerboard(img1: np.ndarray, img2: np.ndarray,
                              block_size: int = 32) -> np.ndarray:
        """Create a checkerboard overlay of two images."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        h, w = img1.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                by = min(y + block_size, h)
                bx = min(x + block_size, w)
                if ((y // block_size) + (x // block_size)) % 2 == 0:
                    block = img1[y:by, x:bx]
                else:
                    block = img2[y:by, x:bx]
                if block.ndim == 2:
                    result[y:by, x:bx] = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
                else:
                    result[y:by, x:bx] = block
        return result

    @staticmethod
    def _create_blend(img1: np.ndarray, img2: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """Create a blended overlay of two images."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        c1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
        c2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2
        return cv2.addWeighted(c1, alpha, c2, 1 - alpha, 0)
