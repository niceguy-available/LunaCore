import cv2
import numpy as np
import sys
sys.path.insert(0, "backend")
from scratch_physical import generate_lunar_terrain_model, render_lunar_image

heightmap, albedo_map = generate_lunar_terrain_model(512, 512, seed=123, num_craters=60)
ref_img = render_lunar_image(heightmap, albedo_map, sun_azimuth_deg=45.0, sun_elevation_deg=45.0)
src_raw = render_lunar_image(heightmap, albedo_map, sun_azimuth_deg=65.0, sun_elevation_deg=40.0)

# Ground truth: 4 deg rotation, tx=15, ty=-10
M_gt = cv2.getRotationMatrix2D((256, 256), 4.0, 1.0)
M_gt[0, 2] += 15
M_gt[1, 2] -= 10
src_img = cv2.warpAffine(src_raw, M_gt, (512, 512), borderMode=cv2.BORDER_REFLECT)

# Test multiple detectors
for det_name in ["sift", "akaze", "orb"]:
    if det_name == "sift":
        detector = cv2.SIFT_create(contrastThreshold=0.01)
        norm_type = cv2.NORM_L2
    elif det_name == "akaze":
        detector = cv2.AKAZE_create(threshold=0.0005)
        norm_type = cv2.NORM_HAMMING
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm_type = cv2.NORM_HAMMING

    # Contrast enhance first
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    s_enh = clahe.apply(src_img)
    r_enh = clahe.apply(ref_img)

    kp1, desc1 = detector.detectAndCompute(s_enh, None)
    kp2, desc2 = detector.detectAndCompute(r_enh, None)

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < 0.78 * n.distance]

    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Test Affine with RANSAC
        M_est, inliers_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        in_cnt = int(inliers_mask.sum()) if inliers_mask is not None else 0
        
        if M_est is not None:
            rot = np.degrees(np.arctan2(M_est[1, 0], M_est[0, 0]))
            tx, ty = M_est[0, 2], M_est[1, 2]
            print(f"[{det_name.upper()}] Matches={len(good)}, Inliers={in_cnt}, Recovered: rot={rot:.2f} deg, tx={tx:.2f}, ty={ty:.2f}")
        else:
            print(f"[{det_name.upper()}] Failed to estimate affine")
