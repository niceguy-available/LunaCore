import cv2
import numpy as np
import sys
sys.path.insert(0, "backend")
from app.pipeline import RegistrationOrchestrator
from scripts.generate_sample_data import generate_lunar_surface

# Generate realistic lunar pair
print("Generating pair...")
ref_1 = generate_lunar_surface(512, 512, seed=101, num_craters=45, sun_azimuth_deg=45.0, sun_elevation_deg=40.0)
src_raw = generate_lunar_surface(512, 512, seed=101, num_craters=45, sun_azimuth_deg=90.0, sun_elevation_deg=35.0)

# Rigid transform: +12px x, -8px y, 3 deg rotation
center = (256, 256)
M = cv2.getRotationMatrix2D(center, 3.0, 1.0)
M[0, 2] += 12
M[1, 2] -= 8
src_1 = cv2.warpAffine(src_raw, M, (512, 512), borderMode=cv2.BORDER_REFLECT)

config = {
    "engine": "classical",
    "illumination_method": "combined",
    "feature_detector": "sift",
    "matcher": "flann",
    "ratio_threshold": 0.85,
    "ransac_method": "usac_magsac",
    "ransac_threshold": 5.0,
    "transform_type": "homography",
    "grid_size": 8,
    "max_per_cell": 80,
    "subpixel_method": "parabolic",
    "refine_intensity": True,
    "multiscale": False,
}

orchestrator = RegistrationOrchestrator(config)
res = orchestrator.register(src_1, ref_1)

print("Status:", res["status"])
print("Inlier count:", res["metrics"]["match_stats"]["inlier_count"])
print("Inlier ratio:", f"{res['metrics']['match_stats']['inlier_ratio']*100:.1f}%")
print("Reproj RMSE:", res["metrics"]["reprojection"]["rmse"])
print("Uniformity:", res["metrics"]["distribution"]["uniformity_score"])
print("Quality Score:", res["metrics"]["quality_score"])
H = np.array(res["transformation_matrix"])
rot = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
tx, ty = H[0, 2], H[1, 2]
print(f"Recovered: rot={rot:.2f} deg (target 3.0), tx={tx:.2f} (target 12), ty={ty:.2f} (target -8)")
