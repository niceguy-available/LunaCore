# `lunacore` — 6-phase registration pipeline

`lunacore/` is a self-contained Python package that registers a raw
Chandrayaan-2 PDS4 product (OHRC, TMC-2, IIRS) onto an LRO base map fetched
automatically over WMS, and writes a map-projected GeoTIFF plus accuracy report.
It is independent of the FastAPI demo backend in `backend/`.

```bash
pip install -e ".[deep,pds4,dev]"

# Reference fetched automatically from the footprint in the label
python -m lunacore register ch2_ohr_xxx.xml -o out/ --layer LROC_WAC

# Local reference, CPU-only classical matcher
python -m lunacore register ch2_tmc_xxx.xml -o out/ --reference wac.tif --matcher sift
```

Outputs in `out/`: `registered.tif` (float32, reference grid, lunar CRS, NaN
no-data, homography in the tags), `report.json` (metrics, timings, config),
`matches.csv`, and `reference_wms.tif` when the base map was downloaded.

## Phases and modules

| Phase | Module | What it does |
|---|---|---|
| 1 Ingestion | `ingestion.py`, `wms.py` | Namespace-agnostic PDS4 label parsing (GSD with unit conversion, corner lat/lon, `cart:` bounding box, special constants, scaling); the `.dat` is **memory-mapped**, not loaded. `pds4_tools` is the fallback for layouts the native reader rejects. The footprint (padded 15 %) sizes a WMS `GetMap` so ground pixels are square; retries with exponential backoff, streamed download, `ServiceExceptionReport` detection, georeferencing taken from the request. |
| 2 Harmonisation | `harmonization.py` | Two-pass **streaming PCA** (covariance accumulated strip by strip, `eigh`, PC1 sign fixed to brightness). `S = GSD_ref / GSD_src`; NaN-aware exact block-mean + `INTER_AREA` reduction with the pixel-centre mapping tracked as a 3×3 matrix. Percentile stretch + **CLAHE 8×8**. A **geo-prior** homography from the label's corner coordinates pre-warps the source north-up into the reference frame. |
| 3 Matching | `matching.py` | kornia **LoFTR** (fp16 on CUDA, `inference_mode`, max 1024 px/side, co-located tiling when the prior already aligned the frames); SIFT fallback. Confidence + no-data mask filtering, then an adaptive **quad-tree** keeps the best few matches per leaf for an even spread. Matching is repeated on the **negated** source and the polarity with more inliers wins (inverted shadows / IR vs optical). |
| 4 Outliers | `geometry.py` | **MAGSAC++** (`cv2.USAC_MAGSAC`), then the consensus re-solved by **Hartley-normalised DLT via SVD** (`Ah = 0`, smallest singular vector) and polished with Levenberg–Marquardt on pixel reprojection error; sanity checks on orientation, scale and the line at infinity. Optional ECC dense refinement (off by default; see below). |
| 5 Warp | `geometry.py` | `H_total = D_ref⁻¹ · H_res · H0 · D_src`; **bicubic** inverse-mapped warp from the anti-aliased reduced source onto the reference grid, with a validity mask that excludes pixels whose 4×4 support touches no-data. |
| 6 Evaluation | `evaluation.py` | RMSE on inliers **and on held-out checkpoints** (20 % not used in the fit), **ZNCC** on intensities and on gradient magnitude (modality-robust), grid coverage/entropy **uniformity** over the source footprint, GeoTIFF + JSON output. |

## Notes and decisions

* **CRS.** `EPSG:32630` is a terrestrial UTM zone and is wrong for the Moon.
  Planetary WMS servers publish simple-cylindrical lunar lon/lat under
  `EPSG:4326`; the output GeoTIFF is tagged with the IAU 2015 lunar sphere
  (`IAU_2015:30100`). WMS 1.3.0 swaps the `EPSG:4326` bbox to lat/lon order —
  handled in `build_getmap_params`.
* **Why a geo-prior.** LoFTR is not rotation invariant and OHRC strips are in
  sensor geometry. Pre-warping with the label's corner coordinates leaves the
  network only a small residual; without corners the pipeline falls back to a
  scale-only prior (`--prior scale|bbox|corners|auto`).
* **ECC is opt-in (`--ecc`).** It maximises intensity correlation, which is
  biased when sun angle or modality differ; on the synthetic tests it made a
  0.15 px registration worse. It is also auto-rejected when it lowers the
  correlation or moves any corner by more than 3 px.
* **Limits.** Footprints crossing ±180° longitude are rejected; polar
  products need a polar-stereographic reference (supply one with
  `--reference`; projected CRSs are reprojected for the prior). Outputs larger
  than 32767 px per side exceed OpenCV's warp limit.
* **LoFTR weights** download from the kornia model zoo on first use; pass a
  local `.ckpt` path via `MatchingConfig.loftr_weights` for offline machines.

## Tests

`pytest` runs 39 tests, including end-to-end runs on a synthetic cratered
terrain with a known ground-truth homography (4× scale difference, 6°
rotation, pointing error, different or opposite sun azimuth). They use the
SIFT matcher so they run offline; the LoFTR wrapper is exercised with random
weights. Every end-to-end case recovers the ground truth to < 0.75 reference
pixels with sub-pixel checkpoint RMSE.
