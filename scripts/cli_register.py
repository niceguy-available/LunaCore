#!/usr/bin/env python3
"""
cli_register.py — Standalone CLI for Lunar Image Registration
ISRO SIH 2026 | Problem Statement 26166

Standalone software deliverable that runs without a web server.
Produces: registered image, match points CSV, metrics JSON, visualization PNG.

Usage:
    python cli_register.py --source ohrc.tif --reference tmc.tif --output results/
    python cli_register.py --source ohrc.tif --reference tmc.tif --engine classical --detector sift
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Lunar Image Registration — CLI Tool (ISRO SIH 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source source.png --reference reference.png
  %(prog)s --source ohrc.tif --reference tmc.tif --output results/ --detector rift
  %(prog)s --source img1.png --reference img2.png --illumination phase_congruency --transform affine
        """
    )

    # Required
    parser.add_argument("--source", "-s", required=True, help="Path to source image")
    parser.add_argument("--reference", "-r", required=True, help="Path to reference image")

    # Output
    parser.add_argument("--output", "-o", default="./registration_output",
                        help="Output directory (default: ./registration_output)")

    # Pipeline config
    parser.add_argument("--engine", choices=["classical", "deep", "hybrid"],
                        default="classical", help="Registration engine (default: classical)")
    parser.add_argument("--illumination",
                        choices=["phase_congruency", "dol", "wld", "clahe", "combined"],
                        default="phase_congruency",
                        help="Illumination normalization method (default: phase_congruency)")
    parser.add_argument("--detector", choices=["sift", "orb", "akaze", "rift"],
                        default="sift", help="Feature detector (default: sift)")
    parser.add_argument("--transform",
                        choices=["homography", "affine", "rigid", "similarity", "tps", "polynomial"],
                        default="homography",
                        help="Transform estimation method (default: homography)")
    parser.add_argument("--subpixel",
                        choices=["parabolic", "lucas_kanade", "phase_correlation", "combined"],
                        default="parabolic",
                        help="Sub-pixel refinement method (default: parabolic)")
    parser.add_argument("--ransac-method",
                        choices=["ransac", "usac_magsac", "lmeds"],
                        default="usac_magsac",
                        help="RANSAC variant (default: usac_magsac)")
    parser.add_argument("--ransac-threshold", type=float, default=5.0,
                        help="RANSAC inlier threshold in pixels (default: 5.0)")
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Grid NMS size for uniform distribution (default: 8)")
    parser.add_argument("--max-per-cell", type=int, default=80,
                        help="Max keypoints per grid cell (default: 80)")
    parser.add_argument("--ratio-threshold", type=float, default=0.75,
                        help="Lowe's ratio test threshold (default: 0.75)")

    # Sensors
    parser.add_argument("--source-sensor",
                        choices=["OHRC", "TMC2", "TMC", "IIRS", "LRO_NAC", "LRO_WAC"],
                        default=None, help="Source sensor type")
    parser.add_argument("--reference-sensor",
                        choices=["OHRC", "TMC2", "TMC", "IIRS", "LRO_NAC", "LRO_WAC"],
                        default=None, help="Reference sensor type")

    # Flags
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip intensity-based refinement")
    parser.add_argument("--multiscale", action="store_true",
                        help="Enable multi-scale feature detection")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.source).exists():
        print(f"ERROR: Source image not found: {args.source}")
        sys.exit(1)
    if not Path(args.reference).exists():
        print(f"ERROR: Reference image not found: {args.reference}")
        sys.exit(1)

    # Load images
    print(f"\n{'=' * 65}")
    print(f"  LUNAR IMAGE REGISTRATION -- CLI")
    print(f"  ISRO SIH 2026 | Problem Statement 26166")
    print(f"{'=' * 65}\n")

    print(f"Source:     {args.source}")
    print(f"Reference:  {args.reference}")
    print(f"Engine:     {args.engine}")
    print(f"Detector:   {args.detector}")
    print(f"Illum:      {args.illumination}")
    print(f"Transform:  {args.transform}")
    print()

    src_img = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    ref_img = cv2.imread(args.reference, cv2.IMREAD_UNCHANGED)

    if src_img is None:
        print(f"ERROR: Could not read source image: {args.source}")
        sys.exit(1)
    if ref_img is None:
        print(f"ERROR: Could not read reference image: {args.reference}")
        sys.exit(1)

    print(f"Source shape:    {src_img.shape}")
    print(f"Reference shape: {ref_img.shape}")

    # Build config
    config = {
        "engine": args.engine,
        "illumination_method": args.illumination,
        "feature_detector": args.detector,
        "matcher": "flann",
        "ratio_threshold": args.ratio_threshold,
        "ransac_method": args.ransac_method,
        "ransac_threshold": args.ransac_threshold,
        "transform_type": args.transform,
        "grid_size": args.grid_size,
        "max_per_cell": args.max_per_cell,
        "subpixel_method": args.subpixel,
        "refine_intensity": not args.no_refine,
        "multiscale": args.multiscale,
        "source_sensor": args.source_sensor,
        "reference_sensor": args.reference_sensor,
        "debug": args.verbose,
    }

    # Import and run pipeline
    from app.pipeline import RegistrationOrchestrator

    def cli_callback(stage, status, data):
        if status == "running":
            print(f"  [>] {stage}...", end="", flush=True)
        elif status == "complete":
            # Print key info
            info_parts = []
            for key in ["source_keypoints", "reference_keypoints",
                         "total_matches", "inlier_count", "inlier_ratio",
                         "quality_score", "method", "refined"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, float):
                        info_parts.append(f"{key}={val:.3f}")
                    else:
                        info_parts.append(f"{key}={val}")
            info = ", ".join(info_parts[:4])
            print(f" [OK] ({info})")

    print(f"\n{'-' * 65}")
    print(f"Running pipeline...")
    print(f"{'-' * 65}\n")

    orchestrator = RegistrationOrchestrator(config)
    result = orchestrator.register(
        src_img, ref_img,
        callback=cli_callback,
        source_sensor=args.source_sensor,
        reference_sensor=args.reference_sensor,
    )

    # Check result
    if result["status"] != "complete":
        print(f"\n[FAIL] Registration FAILED: {result.get('error', 'unknown')}")
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    print(f"\n{'-' * 65}")
    print(f"Saving outputs to: {out_dir}")
    print(f"{'-' * 65}\n")

    # Registered image
    raw_reg = result.pop("_raw_registered", None)
    raw_src = result.pop("_raw_source", None)
    raw_ref = result.pop("_raw_reference", None)
    raw_vis = result.pop("_raw_match_vis", None)

    if raw_reg is not None:
        cv2.imwrite(str(out_dir / "registered.png"), raw_reg)
        print(f"  [OK] registered.png")

    if raw_vis is not None:
        cv2.imwrite(str(out_dir / "match_visualization.png"), raw_vis)
        print(f"  [OK] match_visualization.png")

    # Transformation matrix
    T = result.get("transformation_matrix")
    if T is not None:
        np.save(str(out_dir / "transformation.npy"), np.array(T))
        print(f"  [OK] transformation.npy")

    # Metrics JSON
    metrics = result.get("metrics", {})
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  [OK] metrics.json")

    # Match points CSV
    match_points = result.get("match_points", [])
    if match_points:
        csv_lines = ["idx,src_x,src_y,dst_x,dst_y"]
        for i, mp in enumerate(match_points):
            csv_lines.append(
                f"{i},{mp['src_x']:.4f},{mp['src_y']:.4f},"
                f"{mp['dst_x']:.4f},{mp['dst_y']:.4f}"
            )
        with open(out_dir / "matches.csv", "w") as f:
            f.write("\n".join(csv_lines))
        print(f"  [OK] matches.csv ({len(match_points)} points)")

    # Full result JSON (without base64 images)
    export_result = {k: v for k, v in result.items()
                     if not isinstance(v, str) or len(v) < 1000}
    with open(out_dir / "result.json", "w") as f:
        json.dump(export_result, f, indent=2, default=str)
    print(f"  [OK] result.json")

    # Print summary
    from app.pipeline.validation import RegistrationEvaluator
    summary = RegistrationEvaluator.metrics_to_summary_text(metrics)
    print(f"\n{summary}")

    print(f"\nProcessing time: {result.get('processing_time', 0):.2f}s")
    print(f"Output saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
