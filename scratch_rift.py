import cv2
import numpy as np

def compute_improved_rift(pc_img, pc_orient, kps, patch_size=48, n_orient_bins=6, n_spatial_bins=4):
    half = patch_size // 2
    h, w = pc_img.shape[:2]
    desc_dim = n_spatial_bins * n_spatial_bins * n_orient_bins
    descriptors = []
    
    # Wrap orientation to [0, pi) - symmetric edge direction
    orient_mod = np.mod(pc_orient, np.pi)
    
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        y1, y2 = max(0, y - half), min(h, y + half)
        x1, x2 = max(0, x - half), min(w, x + half)
        if y2 - y1 < 8 or x2 - x1 < 8:
            descriptors.append(np.zeros(desc_dim, dtype=np.float32))
            continue
            
        orient_patch = orient_mod[y1:y2, x1:x2]
        mag_patch = pc_img[y1:y2, x1:x2].astype(np.float32) / 255.0
        
        orient_patch = cv2.resize(orient_patch, (patch_size, patch_size))
        mag_patch = cv2.resize(mag_patch, (patch_size, patch_size))
        
        # Quantize [0, pi) into n_orient_bins
        mim = (orient_patch / np.pi * n_orient_bins).astype(np.int32)
        mim = np.clip(mim, 0, n_orient_bins - 1)
        
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
                    
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm
        descriptors.append(descriptor)
        
    return np.array(descriptors, dtype=np.float32)

import sys
sys.path.insert(0, "backend")
from app.pipeline.preprocessing import compute_phase_congruency
src = cv2.imread("data/samples/pair_sun_angle/source.png", cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("data/samples/pair_sun_angle/reference.png", cv2.IMREAD_GRAYSCALE)

pc_src, orient_src, _ = compute_phase_congruency(src)
pc_ref, orient_ref, _ = compute_phase_congruency(ref)

# Keypoints from Harris/FAST
corners_src = cv2.goodFeaturesToTrack((pc_src * 255).astype(np.uint8), maxCorners=2000, qualityLevel=0.005, minDistance=6)
corners_ref = cv2.goodFeaturesToTrack((pc_ref * 255).astype(np.uint8), maxCorners=2000, qualityLevel=0.005, minDistance=6)

kps1 = [cv2.KeyPoint(float(p[0, 0]), float(p[0, 1]), 20) for p in corners_src]
kps2 = [cv2.KeyPoint(float(p[0, 0]), float(p[0, 1]), 20) for p in corners_ref]

desc1 = compute_improved_rift((pc_src * 255).astype(np.uint8), orient_src, kps1)
desc2 = compute_improved_rift((pc_ref * 255).astype(np.uint8), orient_ref, kps2)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.knnMatch(desc1, desc2, k=2)
good = [m for m, n in matches if m.distance < 0.92 * n.distance]

print(f"Good matches: {len(good)}")
src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
inliers = int(mask.sum()) if mask is not None else 0
print(f"USAC_MAGSAC Inliers: {inliers} / {len(good)}")
if H is not None:
    rot = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    tx, ty = H[0, 2], H[1, 2]
    print(f"Recovered: rot={rot:.2f} deg, tx={tx:.2f}, ty={ty:.2f} (Target: rot=4.0, tx=15, ty=-10)")
