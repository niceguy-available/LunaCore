import cv2
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path("backend").resolve()))

from app.pipeline.preprocessing import LunarPreprocessor, compute_phase_congruency
from app.pipeline.feature_extraction import FeatureEngine
from app.pipeline.registration import TransformEstimator
from app.pipeline.subpixel import SubPixelRefiner
from app.pipeline.validation import RegistrationEvaluator

src = cv2.imread("data/samples/pair_sun_angle/source.png", cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("data/samples/pair_sun_angle/reference.png", cv2.IMREAD_GRAYSCALE)

print(f"Loaded src: {src.shape}, ref: {ref.shape}")

# Test 1: Phase Congruency magnitude only
pc_src, orient_src, _ = compute_phase_congruency(src)
pc_ref, orient_ref, _ = compute_phase_congruency(ref)

pc_src_u8 = (pc_src * 255).astype(np.uint8)
pc_ref_u8 = (pc_ref * 255).astype(np.uint8)

# Test SIFT on PC magnitude
sift = cv2.SIFT_create(contrastThreshold=0.01)
kp1, desc1 = sift.detectAndCompute(pc_src_u8, None)
kp2, desc2 = sift.detectAndCompute(pc_ref_u8, None)

print(f"Keypoints on PC: src={len(kp1)}, ref={len(kp2)}")

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(desc1, desc2, k=2)
good = [m for m, n in matches if m.distance < 0.85 * n.distance]
print(f"Good matches with ratio 0.85: {len(good)}")

if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"USAC_MAGSAC Inliers on pure PC: {inliers} / {len(good)}")
    if H is not None:
        # Check recovered rotation and translation
        a, b = H[0, 0], H[0, 1]
        rot = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
        tx, ty = H[0, 2], H[1, 2]
        print(f"Recovered: rot={rot:.2f} deg, tx={tx:.2f}, ty={ty:.2f} (Target: rot=4.0, tx=15, ty=-10)")

# Test 2: Morphological Gradient / Sobel Magnitude
grad_src = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
grad_ref = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
g_src_u8 = clahe.apply(grad_src)
g_ref_u8 = clahe.apply(grad_ref)

kp1_g, desc1_g = sift.detectAndCompute(g_src_u8, None)
kp2_g, desc2_g = sift.detectAndCompute(g_ref_u8, None)
matches_g = matcher.knnMatch(desc1_g, desc2_g, k=2)
good_g = [m for m, n in matches_g if m.distance < 0.85 * n.distance]
print(f"Morph Gradient Good matches: {len(good_g)}")
if len(good_g) >= 4:
    src_pts_g = np.float32([kp1_g[m.queryIdx].pt for m in good_g]).reshape(-1, 1, 2)
    dst_pts_g = np.float32([kp2_g[m.trainIdx].pt for m in good_g]).reshape(-1, 1, 2)
    H_g, mask_g = cv2.findHomography(src_pts_g, dst_pts_g, cv2.USAC_MAGSAC, 5.0)
    inliers_g = int(mask_g.sum()) if mask_g is not None else 0
    print(f"USAC_MAGSAC Inliers on Morph Gradient: {inliers_g} / {len(good_g)}")
