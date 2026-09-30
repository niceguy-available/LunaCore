"""
Comprehensive Lunar Dataset Generator
Creates 6 diverse, physically simulated Chandrayaan-2 lunar benchmark pairs
covering Sun-Angle Invariance, OHRC/TMC-2/IIRS Cross-Sensor Scale, South Pole Topography, and Basalt Plains.
"""

import cv2
import numpy as np
import json
from pathlib import Path

def generate_lunar_terrain_model(width=512, height=512, seed=101, num_craters=60, terrain_type="highland"):
    """Generate fixed ground-truth surface heightmap and regolith albedo map."""
    np.random.seed(seed)
    surface = np.zeros((height, width), dtype=np.float32)
    
    # Octave fractal noise for natural lunar topography
    octaves = [64, 32, 16, 8, 4] if terrain_type != "mare" else [128, 64, 32, 16]
    for octave, scale in enumerate(octaves):
        freq = 1.0 / scale
        noise = cv2.resize(
            np.random.randn(int(height * freq) + 2, int(width * freq) + 2).astype(np.float32),
            (width, height)
        )
        surface += noise * (0.5 ** octave)

    surface = (surface - surface.min()) / (surface.max() - surface.min() + 1e-8)
    scale_factor = 45.0 if terrain_type == "polar" else (18.0 if terrain_type == "mare" else 32.0)
    heightmap = surface * scale_factor

    # Add realistic impact craters (bowl + raised rim + optional central peak)
    for _ in range(num_craters):
        cx = np.random.randint(30, width - 30)
        cy = np.random.randint(30, height - 30)
        radius = np.random.randint(8, 70 if terrain_type != "mare" else 35)
        depth = radius * np.random.uniform(0.45, 0.85)

        y, x = np.ogrid[-cy:height - cy, -cx:width - cx]
        dist = np.sqrt(x * x + y * y)

        rim_width = radius * 0.32
        rim_height = depth * 0.28

        inside = dist <= radius
        heightmap[inside] -= depth * (1.0 - (dist[inside] / radius) ** 2)

        rim_dist = np.abs(dist - radius)
        rim_factor = np.exp(-(rim_dist ** 2) / (2 * (rim_width * 0.5) ** 2))
        heightmap += rim_factor * rim_height

        # Central peak for large impact craters (radius > 40)
        if radius > 40 and np.random.rand() > 0.4:
            peak_r = radius * 0.2
            peak_dist = np.maximum(0, peak_r - dist)
            heightmap += (peak_dist / peak_r) * (depth * 0.45)

    # Fixed surface albedo map (regolith color variations, ejecta blankets)
    np.random.seed(seed + 999)
    albedo = cv2.GaussianBlur(np.random.rand(height, width).astype(np.float32), (25, 25), 0)
    albedo = (albedo - albedo.min()) / (albedo.max() - albedo.min() + 1e-8)
    base_albedo = 0.65 if terrain_type == "mare" else 0.82
    albedo_map = base_albedo + 0.35 * albedo

    return heightmap, albedo_map


def render_lunar_image(heightmap, albedo_map, sun_azimuth_deg, sun_elevation_deg, slope_scale=2.5, shadow_intensity=0.85):
    """Render physical lunar surface photo under specified solar coordinates."""
    height, width = heightmap.shape
    sun_az_rad = np.radians(sun_azimuth_deg)
    sun_el_rad = np.radians(sun_elevation_deg)

    # Sun vector
    lx = np.cos(sun_el_rad) * np.sin(sun_az_rad)
    ly = -np.cos(sun_el_rad) * np.cos(sun_az_rad)
    lz = np.sin(sun_el_rad)

    gx = cv2.Sobel(heightmap, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(heightmap, cv2.CV_32F, 0, 1, ksize=3)
    gz = np.ones_like(gx) * slope_scale

    norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-8
    nx, ny, nz = -gx / norm, -gy / norm, gz / norm

    dot = nx * lx + ny * ly + nz * lz
    shading = np.clip(dot, 0.05, 1.0)

    # Cast terrain shadows
    step_size = 2.0
    dx = -lx * step_size
    dy = -ly * step_size
    dz = lz * step_size * 2.2
    shadow_map = np.ones((height, width), dtype=np.float32)

    for step in range(1, 16):
        M = np.float32([[1, 0, -dx * step], [0, 1, -dy * step]])
        shifted_h = cv2.warpAffine(heightmap, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        ray_height = shifted_h - step * dz
        shadow_map[ray_height > heightmap] *= shadow_intensity

    image = shading * shadow_map * albedo_map
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def create_all_sample_datasets():
    out_dirs = [
        Path("backend/static/samples"),
        Path("static/samples"),
        Path("data/samples"),
        Path("frontend/public/samples"),
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    datasets = []

    # -------------------------------------------------------------
    # 1. TMC-2 Sun Angle Invariance Pair (Azimuth 45° vs 75°)
    # -------------------------------------------------------------
    print("Generating Pair 1: TMC-2 Sun Angle Invariance...")
    h1, a1 = generate_lunar_terrain_model(512, 512, seed=101, num_craters=65, terrain_type="highland")
    ref_1 = render_lunar_image(h1, a1, sun_azimuth_deg=45.0, sun_elevation_deg=45.0)
    src_raw_1 = render_lunar_image(h1, a1, sun_azimuth_deg=75.0, sun_elevation_deg=38.0)
    M1 = cv2.getRotationMatrix2D((256, 256), 4.0, 1.0)
    M1[0, 2] += 15
    M1[1, 2] -= 10
    src_1 = cv2.warpAffine(src_raw_1, M1, (512, 512), borderMode=cv2.BORDER_REFLECT)
    datasets.append({
        "id": "pair_sun_angle",
        "name": "Chandrayaan-2 TMC-2 Solar Azimuth Shift",
        "source_sensor": "TMC2",
        "reference_sensor": "TMC2",
        "src_img": src_1,
        "ref_img": ref_1,
        "desc": "Same terrain under 45° vs 75° solar illumination with 4° rotation and translation shift.",
    })

    # -------------------------------------------------------------
    # 2. OHRC to TMC-2 Multi-Modal Scale Zoom Pair
    # -------------------------------------------------------------
    print("Generating Pair 2: OHRC to TMC-2 Multi-Scale...")
    h2, a2 = generate_lunar_terrain_model(1024, 1024, seed=202, num_craters=95, terrain_type="highland")
    ref_full_2 = render_lunar_image(h2, a2, sun_azimuth_deg=50.0, sun_elevation_deg=45.0)
    ref_2 = cv2.resize(ref_full_2, (512, 512))

    src_full_2 = render_lunar_image(h2, a2, sun_azimuth_deg=78.0, sun_elevation_deg=36.0)
    crop_2 = src_full_2[180:720, 180:720]
    M2 = cv2.getRotationMatrix2D((270, 270), -3.5, 1.0)
    src_2_rot = cv2.warpAffine(crop_2, M2, (540, 540), borderMode=cv2.BORDER_REFLECT)
    src_2 = cv2.resize(src_2_rot, (512, 512))
    datasets.append({
        "id": "pair_cross_modal",
        "name": "Chandrayaan-2 OHRC (0.25m) to TMC-2 (5m) Scale Pair",
        "source_sensor": "OHRC",
        "reference_sensor": "TMC2",
        "src_img": src_2,
        "ref_img": ref_2,
        "desc": "High-resolution OHRC sub-frame zoom mapped into TMC-2 regional base terrain context.",
    })

    # -------------------------------------------------------------
    # 3. Lunar South Pole / Low Sun Elevation Pair (Chandrayaan-3 Site)
    # -------------------------------------------------------------
    print("Generating Pair 3: Lunar South Pole (Low Sun Elevation)...")
    h3, a3 = generate_lunar_terrain_model(512, 512, seed=303, num_craters=50, terrain_type="polar")
    ref_3 = render_lunar_image(h3, a3, sun_azimuth_deg=30.0, sun_elevation_deg=18.0, slope_scale=3.5, shadow_intensity=0.7)
    src_raw_3 = render_lunar_image(h3, a3, sun_azimuth_deg=55.0, sun_elevation_deg=14.0, slope_scale=3.5, shadow_intensity=0.7)
    M3 = cv2.getRotationMatrix2D((256, 256), 2.5, 1.0)
    M3[0, 2] += 10
    M3[1, 2] += 8
    src_3 = cv2.warpAffine(src_raw_3, M3, (512, 512), borderMode=cv2.BORDER_REFLECT)
    datasets.append({
        "id": "pair_polar_south",
        "name": "Lunar South Pole Deep Crater Shadows",
        "source_sensor": "TMC2",
        "reference_sensor": "TMC2",
        "src_img": src_3,
        "ref_img": ref_3,
        "desc": "Low sun elevation (14°–18°) casting elongated shadows near polar permanently shadowed regions.",
    })

    # -------------------------------------------------------------
    # 4. Mare Tranquillitatis Basalt Plains (Low Contrast)
    # -------------------------------------------------------------
    print("Generating Pair 4: Mare Basalt Plains...")
    h4, a4 = generate_lunar_terrain_model(512, 512, seed=404, num_craters=40, terrain_type="mare")
    ref_4 = render_lunar_image(h4, a4, sun_azimuth_deg=60.0, sun_elevation_deg=55.0, slope_scale=1.5)
    src_raw_4 = render_lunar_image(h4, a4, sun_azimuth_deg=85.0, sun_elevation_deg=45.0, slope_scale=1.5)
    M4 = cv2.getRotationMatrix2D((256, 256), -4.5, 1.0)
    M4[0, 2] -= 12
    M4[1, 2] += 14
    src_4 = cv2.warpAffine(src_raw_4, M4, (512, 512), borderMode=cv2.BORDER_REFLECT)
    datasets.append({
        "id": "pair_mare_plains",
        "name": "Lunar Mare Basalt Plains (Low-Contrast)",
        "source_sensor": "TMC2",
        "reference_sensor": "TMC2",
        "src_img": src_4,
        "ref_img": ref_4,
        "desc": "Smooth basalt regolith with low relief, testing fine feature extraction and sub-pixel flow.",
    })

    # -------------------------------------------------------------
    # 5. Tycho Crater Rim & Central Peak (Rugged Topography)
    # -------------------------------------------------------------
    print("Generating Pair 5: Rugged Crater Rim & Peak...")
    h5, a5 = generate_lunar_terrain_model(512, 512, seed=505, num_craters=75, terrain_type="highland")
    ref_5 = render_lunar_image(h5, a5, sun_azimuth_deg=40.0, sun_elevation_deg=40.0, slope_scale=3.0)
    src_raw_5 = render_lunar_image(h5, a5, sun_azimuth_deg=68.0, sun_elevation_deg=34.0, slope_scale=3.0)
    M5 = cv2.getRotationMatrix2D((256, 256), 5.0, 1.0)
    M5[0, 2] += 18
    M5[1, 2] -= 15
    src_5 = cv2.warpAffine(src_raw_5, M5, (512, 512), borderMode=cv2.BORDER_REFLECT)
    datasets.append({
        "id": "pair_rugged_crater",
        "name": "Rugged Impact Crater Rim & Slopes",
        "source_sensor": "OHRC",
        "reference_sensor": "TMC2",
        "src_img": src_5,
        "ref_img": ref_5,
        "desc": "Complex crater wall slopes and central peak terrain undergoing significant illumination changes.",
    })

    # -------------------------------------------------------------
    # 6. IIRS Hyperspectral to TMC-2 Cross-Modal
    # -------------------------------------------------------------
    print("Generating Pair 6: IIRS Hyperspectral to TMC-2...")
    h6, a6 = generate_lunar_terrain_model(512, 512, seed=606, num_craters=55, terrain_type="highland")
    # Base TMC-2 Panchromatic
    ref_6 = render_lunar_image(h6, a6, sun_azimuth_deg=50.0, sun_elevation_deg=42.0)
    # IIRS with band absorptions + slight blurring for spectral GSD
    src_raw_6 = render_lunar_image(h6, a6 * 1.15, sun_azimuth_deg=75.0, sun_elevation_deg=38.0)
    src_blur = cv2.GaussianBlur(src_raw_6, (5, 5), 1.2)
    M6 = cv2.getRotationMatrix2D((256, 256), -3.0, 1.0)
    M6[0, 2] += 8
    M6[1, 2] -= 12
    src_6 = cv2.warpAffine(src_blur, M6, (512, 512), borderMode=cv2.BORDER_REFLECT)
    datasets.append({
        "id": "pair_iirs_multimodal",
        "name": "Chandrayaan-2 IIRS (Hyperspectral) to TMC-2 Pair",
        "source_sensor": "IIRS",
        "reference_sensor": "TMC2",
        "src_img": src_6,
        "ref_img": ref_6,
        "desc": "Cross-spectral modal pairing between infrared mineral absorption bands and panchromatic terrain.",
    })

    # Write datasets to all targets
    for p in datasets:
        for base_dir in out_dirs:
            pair_dir = base_dir / p["id"]
            pair_dir.mkdir(parents=True, exist_ok=True)

            src_file = pair_dir / "source.png"
            ref_file = pair_dir / "reference.png"
            thumb_src = pair_dir / "thumb_source.jpg"
            thumb_ref = pair_dir / "thumb_reference.jpg"

            cv2.imwrite(str(src_file), p["src_img"])
            cv2.imwrite(str(ref_file), p["ref_img"])

            cv2.imwrite(str(thumb_src), cv2.resize(p["src_img"], (160, 160)))
            cv2.imwrite(str(thumb_ref), cv2.resize(p["ref_img"], (160, 160)))

            meta = {
                "id": p["id"],
                "name": p["name"],
                "source_sensor": p["source_sensor"],
                "reference_sensor": p["reference_sensor"],
                "source_path": str(src_file),
                "reference_path": str(ref_file),
                "thumbnail_src": f"/samples/{p['id']}/thumb_source.jpg",
                "thumbnail_ref": f"/samples/{p['id']}/thumb_reference.jpg",
                "source_url": f"/samples/{p['id']}/source.png",
                "reference_url": f"/samples/{p['id']}/reference.png",
                "description": p["desc"],
            }
            with open(pair_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

    # Master index
    index_meta = [{"id": p["id"], "name": p["name"], "source_sensor": p["source_sensor"], "reference_sensor": p["reference_sensor"], "description": p["desc"], "source_url": f"/samples/{p['id']}/source.png", "reference_url": f"/samples/{p['id']}/reference.png", "thumb_src": f"/samples/{p['id']}/thumb_source.jpg", "thumb_ref": f"/samples/{p['id']}/thumb_reference.jpg"} for p in datasets]

    for base_dir in out_dirs:
        with open(base_dir / "index.json", "w") as f:
            json.dump(index_meta, f, indent=2)

    print(f"Generated {len(datasets)} lunar benchmark datasets successfully across all directories!")

if __name__ == "__main__":
    create_all_sample_datasets()
