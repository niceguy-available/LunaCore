"""End-to-end orchestration of the six registration phases."""

from __future__ import annotations

import csv
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .errors import GeometryError, HarmonizationError, LabelParseError, MatchingError
from .evaluation import (checkpoint_rmse, gradient_magnitude, uniformity, write_geotiff,
                         write_report, zncc)
from .geometry import (HomographyEstimate, estimate_homography, refine_ecc,
                       reprojection_errors, warp_to_reference)
from .harmonization import HarmonizedPair, harmonize
from .ingestion import Pds4Product, infer_gsd_from_footprint, load_pds4
from .matching import Matcher, Matches, build_matcher, match_images
from .wms import ReferenceMap, fetch_reference, load_reference

log = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    H_source_to_reference: np.ndarray     # full-resolution source px -> reference px
    registered_path: Path
    report_path: Path
    metrics: Dict[str, Any]
    timings_s: Dict[str, float] = field(default_factory=dict)


@contextmanager
def _timed(timings: Dict[str, float], phase: str) -> Iterator[None]:
    t0 = time.perf_counter()
    log.info("── %s", phase)
    try:
        yield
    finally:
        timings[phase] = round(time.perf_counter() - t0, 3)


def _polarities(pair: HarmonizedPair, try_inverted: bool) -> List[Tuple[str, np.ndarray]]:
    options = [("normal", pair.src_canvas_u8)]
    if try_inverted:
        inv = 255 - pair.src_canvas_u8
        inv[~pair.src_canvas_valid] = 0
        options.append(("inverted", inv))
    return options


def _match_and_estimate(pair: HarmonizedPair, matcher: Matcher, cfg: PipelineConfig
                        ) -> Tuple[str, np.ndarray, Matches, HomographyEstimate]:
    """Phases 3-4 for each shadow polarity; keep the one with the most inliers.

    Sun-azimuth changes of ~180° invert crater shading (and infrared vs.
    optical can invert contrast); matching the negated source as well makes
    the pipeline invariant to that without retraining the network.
    """
    same_frame = pair.prior_used != "scale"
    h, w = pair.src_canvas_u8.shape
    best = None
    errors = []
    for name, img in _polarities(pair, cfg.matching.try_inverted_polarity):
        matches = match_images(img, pair.src_canvas_valid, pair.ref_u8, pair.ref_valid,
                               matcher, cfg.matching, same_frame)
        try:
            est = estimate_homography(matches, cfg.geometry, (w, h))
        except GeometryError as exc:
            errors.append(f"{name}: {exc}")
            continue
        log.info("polarity %s: %d inliers", name, est.n_inliers)
        if best is None or est.n_inliers > best[3].n_inliers:
            best = (name, img, matches, est)
    if best is None:
        raise GeometryError("registration failed for all polarities: " + "; ".join(errors))
    return best


def _write_matches_csv(path: Path, matches: Matches, inliers: np.ndarray,
                       H_canvas_to_ref_full: np.ndarray, D_ref_inv: np.ndarray) -> None:
    from .geometry import apply_homography

    ref_full = apply_homography(D_ref_inv, matches.pts1)
    pred_full = apply_homography(H_canvas_to_ref_full, matches.pts0)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["src_canvas_x", "src_canvas_y", "ref_x", "ref_y", "pred_ref_x",
                     "pred_ref_y", "confidence", "inlier"])
        for i in range(len(matches)):
            wr.writerow([f"{matches.pts0[i, 0]:.4f}", f"{matches.pts0[i, 1]:.4f}",
                         f"{ref_full[i, 0]:.4f}", f"{ref_full[i, 1]:.4f}",
                         f"{pred_full[i, 0]:.4f}", f"{pred_full[i, 1]:.4f}",
                         f"{matches.confidence[i]:.4f}", int(inliers[i])])


def register(label_path: str | Path, out_dir: str | Path,
             config: Optional[PipelineConfig] = None,
             reference_path: Optional[str | Path] = None,
             gsd_override_m: Optional[float] = None,
             matcher: Optional[Matcher] = None,
             session=None) -> RegistrationResult:
    """Register a Chandrayaan-2 PDS4 product onto an LRO reference map.

    Args:
        label_path: PDS4 ``.xml`` label of the source product.
        out_dir: Output directory (GeoTIFF, report, match CSV, WMS download).
        config: Pipeline configuration; defaults are tuned for OHRC/TMC-2 on WAC.
        reference_path: Use this GeoTIFF instead of fetching one over WMS.
        gsd_override_m: Source GSD when the label lacks one.
        matcher: Inject a matcher (e.g. a shared LoFTR instance).
        session: Optional ``requests.Session`` for the WMS download.
    """
    cfg = config or PipelineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    # ── Phase 1: ingestion & reference retrieval ────────────────────────────
    with _timed(timings, "1_ingestion"):
        product: Pds4Product = load_pds4(label_path, gsd_override_m)
        if product.gsd_m is None:
            product.gsd_m = infer_gsd_from_footprint(product)
            if product.gsd_m is None:
                raise LabelParseError("no GSD in label and no footprint to infer it from")
            log.warning("GSD inferred from footprint: %.3f m/px", product.gsd_m)
        if reference_path is not None:
            reference: ReferenceMap = load_reference(reference_path)
        else:
            if product.bbox is None:
                raise LabelParseError("label has no footprint; supply reference_path")
            reference = fetch_reference(product.bbox, product.gsd_m, cfg.wms, out,
                                        session=session)

    # ── Phase 2: harmonisation ──────────────────────────────────────────────
    with _timed(timings, "2_harmonization"):
        pair = harmonize(product, reference, cfg.harmonization)

    # ── Phases 3-4: matching & outlier rejection ────────────────────────────
    with _timed(timings, "3_4_matching_outliers"):
        matcher = matcher or build_matcher(cfg.matching)
        polarity, src_img, matches, est = _match_and_estimate(pair, matcher, cfg)

    H_res = est.H
    ecc_cc = (float("nan"), float("nan"))
    if cfg.geometry.ecc_refine:
        with _timed(timings, "4b_ecc_refinement"):
            H_res, *ecc_cc = refine_ecc(src_img, pair.src_canvas_valid, pair.ref_u8,
                                        pair.ref_valid, H_res, cfg.geometry)

    # ── Phase 5: warp ──────────────────────────────────────────────────────
    with _timed(timings, "5_warp"):
        D_ref_inv = np.linalg.inv(pair.D_ref)
        M_small_to_ref = D_ref_inv @ H_res @ pair.H0          # src small -> ref full
        M_small_to_ref /= M_small_to_ref[2, 2]
        H_total = M_small_to_ref @ pair.D_src                 # src full -> ref full
        H_total /= H_total[2, 2]
        warped, warped_valid = warp_to_reference(pair.src_small, M_small_to_ref,
                                                 reference.shape)

    # ── Phase 6: evaluation & output ───────────────────────────────────────
    with _timed(timings, "6_evaluation_output"):
        inl = est.inliers
        src_in, dst_in = matches.pts0[inl], matches.pts1[inl]
        fit_rmse = float(np.sqrt(np.mean(reprojection_errors(H_res, src_in, dst_in) ** 2)))
        cp_rmse, n_cp = checkpoint_rmse(src_in, dst_in, cfg.evaluation.holdout_fraction,
                                        cfg.evaluation.random_seed)
        to_ref_px = pair.matching_gsd_m / reference.gsd_m
        overlap = warped_valid & np.isfinite(reference.image)
        h, w = pair.src_canvas_u8.shape
        metrics: Dict[str, Any] = {
            "n_matches": len(matches),
            "n_inliers": est.n_inliers,
            "inlier_ratio": est.n_inliers / max(len(matches), 1),
            "polarity": polarity,
            "prior": pair.prior_used,
            "scale_ratio_S": pair.scale_ratio,
            "source_gsd_m": product.gsd_m,
            "reference_gsd_m": reference.gsd_m,
            "matching_gsd_m": pair.matching_gsd_m,
            "rmse_fit_ref_px": fit_rmse * to_ref_px,
            "rmse_checkpoint_ref_px": cp_rmse * to_ref_px,
            "rmse_checkpoint_m": cp_rmse * pair.matching_gsd_m,
            "n_checkpoints": n_cp,
            "subpixel": bool(np.isfinite(cp_rmse) and cp_rmse * to_ref_px < 1.0),
            "zncc": zncc(reference.image, warped, overlap),
            "zncc_gradient": zncc(gradient_magnitude(reference.image),
                                  gradient_magnitude(warped), overlap),
            "overlap_fraction": float(overlap.mean()),
            "ecc_cc_before": ecc_cc[0],
            "ecc_cc_after": ecc_cc[1],
            **{f"uniformity_{k}": v for k, v in
               uniformity(src_in, w, h, cfg.evaluation.uniformity_grid,
                          footprint=pair.src_canvas_valid).items()},
            **pair.info,
        }
        registered = out / "registered.tif"
        write_geotiff(registered, warped, reference, {
            "LUNACORE_SOURCE": str(Path(label_path).name),
            "LUNACORE_HOMOGRAPHY": ",".join(f"{v:.12g}" for v in H_total.ravel()),
            "LUNACORE_RMSE_CHECKPOINT_PX": f"{metrics['rmse_checkpoint_ref_px']:.4f}",
            "LUNACORE_ZNCC": f"{metrics['zncc']:.4f}",
        }, cfg.evaluation.geotiff_compress)
        _write_matches_csv(out / "matches.csv", matches, inl,
                           D_ref_inv @ H_res, D_ref_inv)
        report_path = out / "report.json"
        write_report(report_path, {
            "source": str(label_path),
            "reference": str(reference.path) if reference.path else None,
            "instrument": product.instrument,
            "illumination_metadata": product.metadata,
            "homography_source_to_reference": H_total,
            "metrics": metrics,
            "timings_s": timings,
            "config": cfg.to_dict(),
        })

    log.info("done: %d inliers, checkpoint RMSE %.3f ref px, ZNCC %.3f",
             metrics["n_inliers"], metrics["rmse_checkpoint_ref_px"], metrics["zncc"])
    return RegistrationResult(H_total, registered, report_path, metrics, timings)


__all__ = ["RegistrationResult", "register", "HarmonizationError", "MatchingError"]
