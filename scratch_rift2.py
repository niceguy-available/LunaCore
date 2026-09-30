import cv2
import numpy as np
import sys
sys.path.insert(0, "backend")
from app.pipeline.preprocessing import compute_phase_congruency

def compute_rotation_invariant_rift(pc_img, pc_orient, kps, patch_size=48, n_orient_bins=6, n_rings=4, n_rays=8):
    h, w = pc_img.shape[:2]
    desc_dim = n_rings * n_rays * n_orient_bins
    descriptors = []
    
    # Pre-build log-polar grid sample points
    radii = np.linspace(4, patch_size // 2 - 2, n_rings)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    
    orient_mod = np.mod(pc_orient, np.pi)
    
    for kp in kps:
        cx, cy = kp.pt
        
        # Determine dominant orientation within patch
        half = patch_size // 2
        y1, y2 = max(0, int(cy - half)), min(h, int(cy + half))
        x1, x2 = max(0, int(cx - half)), min(w, int(cx + half))
        if y2 - y1 < 10 or x2 - x1 < 10:
            descriptors.append(np.zeros(desc_dim, dtype=np.float32))
            continue
            
        local_orient = orient_mod[y1:y2, x1:x2]
        local_mag = pc_img[y1:y2, x1:x2].astype(np.float32) / 255.0
        
        # Dominant orientation histogram
        hist, _ = np.histogram(local_orient, bins=n_orient_bins, range=(0, np.pi), weights=local_mag)
        dom_idx = np.argmax(hist)
        dom_angle = (dom_idx + 0.5) * (np.pi / n_orient_bins)
        
        descriptor = []
        # Sample concentric rings relative to dominant angle
        for r in radii:
            for ray_idx, base_ray in enumerate(angles):
                sample_theta = base_ray + dom_angle
                sx = cx + r * np.cos(sample_theta)
                sy = cy + r * np.sin(sample_theta)
                
                if 0 <= sx < w - 1 and 0 <= sy < h - 1:
                    # Bilinear sample
                    ix, iy = int(sx), int(sy)
                    fx, fy = sx - ix, sy - iy
                    samp_orient = orient_mod[iy, ix] * (1 - fx) * (1 - fy) + orient_mod[iy, ix+1] * fx * (1 - fy) + orient_mod[iy+1, ix] * (1 - fx) * fy + orient_mod[iy+1, ix+1] * fx * fy
                    samp_mag = (pc_img[iy, ix] * (1 - fx) * (1 - fy) + pc_img[iy, ix+1] * fx * (1 - fy) + pc_img[iy+1, ix] * (1 - fx) * fy + pc_img[iy+1, ix+1] * fx * fy) / 255.0
                    
                    # Bin orientation relative to dominant orientation
                    rel_orient = np.mod(samp_orient - dom_angle, np.pi)
                    bin_idx = int(np.clip(rel_orient / np.pi * n_orient_bins, 0, n_orient_bins - 1))
                    
                    bin_vec = np.zeros(n_orient_bins, dtype=np.float32)
                    bin_vec[bin_idx] = samp_mag
                    descriptor.extend(bin_vec)
                else:
                    descriptor.extend(np.zeros(n_orient_bins, dtype=np.float32))
                    
        desc_arr = np.array(descriptor, dtype=np.float32)
        norm = np.linalg.norm(desc_arr)
        if norm > 0:
            desc_arr /= norm
        descriptors.append(desc_arr)
        
    return np.array(descriptors, dtype=np.float32)

src = cv2.imread("data/samples/pair_sun_angle/source.png", cv2.IMREAD_GRAYSCALE)
ref = cv2.imread("data/samples/pair_sun_angle/reference.png", cv2.IMREAD_GRAYSCALE)

pc_src, orient_src, _ = compute_phase_congruency(src)
pc_ref, orient_ref, _ = compute_phase_congruency(ref)

corners_src = cv2.goodFeaturesToTrack((pc_src * 255).astype(np.uint8), maxCorners=2000, qualityLevel=0.005, minDistance=6)
corners_ref = cv2.goodFeaturesToTrack((pc_ref * 255).astype(np.uint8), maxCorners=2000, qualityLevel=0.005, minDistance=6)

kps1 = [cv2.KeyPoint(float(p[0, 0]), float(p[0, 1]), 20) for p in corners_src]
kps2 = [cv2.KeyPoint(float(p[0, 0]), float(p[0, 1]), 20) for p in corners_ref]

desc1 = compute_rotation_invariant_rift((pc_src * 255).astype(np.uint8), orient_src, kps1)
desc2 = compute_rotation_invariant_rift((pc_ref * 255).astype(np.uint8), orient_ref, kps2)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.knnMatch(desc1, desc2, k=2)
good = [m for m, n in matches if m.distance < 0.88 * n.distance]

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
