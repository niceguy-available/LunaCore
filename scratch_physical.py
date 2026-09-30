import cv2
import numpy as np

def generate_lunar_terrain_model(width=512, height=512, seed=42, num_craters=50):
    """Generate fixed ground-truth surface heightmap and albedo map."""
    np.random.seed(seed)
    surface = np.zeros((height, width), dtype=np.float32)
    for octave, scale in enumerate([64, 32, 16, 8, 4]):
        freq = 1.0 / scale
        noise = cv2.resize(
            np.random.randn(int(height * freq) + 2, int(width * freq) + 2).astype(np.float32),
            (width, height)
        )
        surface += noise * (0.5 ** octave)

    surface = (surface - surface.min()) / (surface.max() - surface.min() + 1e-8)
    heightmap = surface * 35.0

    # Fixed craters
    for _ in range(num_craters):
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        radius = np.random.randint(10, 60)
        depth = radius * np.random.uniform(0.5, 0.9)

        y, x = np.ogrid[-cy:height - cy, -cx:width - cx]
        dist = np.sqrt(x * x + y * y)

        rim_width = radius * 0.3
        rim_height = depth * 0.3

        inside = dist <= radius
        heightmap[inside] -= depth * (1.0 - (dist[inside] / radius) ** 2)

        rim_dist = np.abs(dist - radius)
        rim_factor = np.exp(-(rim_dist ** 2) / (2 * (rim_width * 0.5) ** 2))
        heightmap += rim_factor * rim_height

    # Fixed surface albedo variation (regolith color patches)
    np.random.seed(seed + 999)
    albedo = cv2.GaussianBlur(np.random.rand(height, width).astype(np.float32), (31, 31), 0)
    albedo = (albedo - albedo.min()) / (albedo.max() - albedo.min() + 1e-8)
    albedo_map = 0.75 + 0.35 * albedo

    return heightmap, albedo_map

def render_lunar_image(heightmap, albedo_map, sun_azimuth_deg, sun_elevation_deg):
    """Render physical lunar photo under specified sun angle."""
    height, width = heightmap.shape
    sun_az_rad = np.radians(sun_azimuth_deg)
    sun_el_rad = np.radians(sun_elevation_deg)

    lx = np.cos(sun_el_rad) * np.sin(sun_az_rad)
    ly = -np.cos(sun_el_rad) * np.cos(sun_az_rad)
    lz = np.sin(sun_el_rad)

    gx = cv2.Sobel(heightmap, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(heightmap, cv2.CV_32F, 0, 1, ksize=3)
    gz = np.ones_like(gx) * 2.5

    norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-8
    nx, ny, nz = -gx / norm, -gy / norm, gz / norm

    dot = nx * lx + ny * ly + nz * lz
    shading = np.clip(dot, 0.08, 1.0)

    # Shading * Albedo
    image = shading * albedo_map
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Test rendering and registration
heightmap, albedo_map = generate_lunar_terrain_model(512, 512, seed=123, num_craters=60)

# Reference: Sun at 45 deg azimuth, 45 deg elevation
ref_img = render_lunar_image(heightmap, albedo_map, sun_azimuth_deg=45.0, sun_elevation_deg=45.0)

# Source: Sun at 80 deg azimuth, 38 deg elevation (moderate solar shift)
src_raw = render_lunar_image(heightmap, albedo_map, sun_azimuth_deg=80.0, sun_elevation_deg=38.0)

# Apply ground-truth transformation: 4.0 deg rotation, tx=+15, ty=-10
M = cv2.getRotationMatrix2D((256, 256), 4.0, 1.0)
M[0, 2] += 15
M[1, 2] -= 10
src_img = cv2.warpAffine(src_raw, M, (512, 512), borderMode=cv2.BORDER_REFLECT)

import sys
sys.path.insert(0, "backend")
from app.pipeline import RegistrationOrchestrator

config = {
    "engine": "classical",
    "illumination_method": "combined",
    "feature_detector": "sift",
    "matcher": "flann",
    "ratio_threshold": 0.80,
    "ransac_method": "usac_magsac",
    "ransac_threshold": 4.0,
    "transform_type": "homography",
    "grid_size": 8,
    "max_per_cell": 80,
    "subpixel_method": "parabolic",
    "refine_intensity": True,
    "multiscale": False,
}

orchestrator = RegistrationOrchestrator(config)
res = orchestrator.register(src_img, ref_img)

print("Status:", res["status"])
inliers = res["metrics"]["match_stats"]["inlier_count"]
total = res["metrics"]["match_stats"]["total_matches"]
ratio = res["metrics"]["match_stats"]["inlier_ratio"]
score = res["metrics"]["quality_score"]
reproj = res["metrics"]["reprojection"]["rmse"]
uniformity = res["metrics"]["distribution"]["uniformity_score"]

print(f"Inliers: {inliers} / {total} ({ratio*100:.1f}%)")
print(f"Reprojection RMSE: {reproj:.3f} px")
print(f"Uniformity: {uniformity:.1f} / 100")
print(f"Quality Score: {score:.1f} / 100")

H = np.array(res["transformation_matrix"])
rot = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
tx, ty = H[0, 2], H[1, 2]
print(f"Recovered: rot={rot:.2f} deg (Target: 4.0), tx={tx:.2f} (Target: 15.0), ty={ty:.2f} (Target: -10.0)")
