"""
example_db_usage.py — shows how to wire RegistrationDB into the
existing LunarImageRegistrationPipeline (pipeline.py).

Run:  python example_db_usage.py
"""

import json
from pathlib import Path

from db import RegistrationDB

# from pipeline import LunarImageRegistrationPipeline  # uncomment in the real project


def main():
    db = RegistrationDB("lunar_registration.db")

    # 1. Register the images you're about to process (idempotent: same
    #    file_path is upserted, not duplicated)
    src_id = db.add_image(
        "data/ohrc_source.png",
        sensor_code="OHRC",
        sun_azimuth_deg=142.3,
        sun_elevation_deg=35.1,
        width_px=4096, height_px=4096,
        gsd_m=0.25,
    )
    ref_id = db.add_image(
        "data/tmc_reference.png",
        sensor_code="TMC2",
        sun_azimuth_deg=98.7,
        sun_elevation_deg=41.4,
        width_px=2048, height_px=2048,
        gsd_m=5.0,
    )

    # 2. Save the config used for this run (from config.json)
    with open("config.json") as f:
        full_config = json.load(f)
    config_id = db.add_config("default-sift-flann-homography", full_config)

    # 3. Run the actual pipeline (commented here since no real images exist
    #    in this sandbox) and log the result:
    #
    # config = {
    #     'feature_method': 'sift', 'matcher_method': 'flann',
    #     'transformation_type': 'homography',
    #     'enhance_preprocessing': True, 'remove_illumination_bias': True,
    #     'refine_alignment': True,
    # }
    # pipeline = LunarImageRegistrationPipeline(config)
    # result = pipeline.register_pair(
    #     "data/ohrc_source.png", "data/tmc_reference.png",
    #     "output/registered.png",
    # )
    # run_id = db.log_run(src_id, ref_id, result, config_id=config_id,
    #                      registered_image_path="output/registered.png")

    # --- for this demo, simulate a result_dict like pipeline.py returns ---
    fake_result = {
        "transformation_matrix": [[1.02, 0.01, 25.3], [-0.01, 1.01, -15.2], [0, 0, 1]],
        "transformation_params": {"translation": (25.3, -15.2), "rotation": 5.8, "scale": 1.02},
        "keypoints_src": 1234,
        "keypoints_ref": 1456,
        "matches": 156,
        "metrics": {
            "rmse": 15.23, "mae": 12.45, "correlation": 0.9234,
            "ssim": 0.8756, "mutual_info": 2.3456, "gradient_corr": 0.8945,
        },
        "quality_score": 85.34,
        "processing_time": 12.56,
    }
    run_id = db.log_run(src_id, ref_id, fake_result, config_id=config_id,
                         registered_image_path="output/registered.png",
                         transformation_type="homography")
    print(f"Logged run_id={run_id}")

    # 4. Query it back
    print("\n--- Run detail ---")
    print(json.dumps(db.get_run(run_id), indent=2, default=str))

    print("\n--- Top runs ---")
    for r in db.top_runs(limit=5):
        print(f"run {r['run_id']}: {r['source_sensor']} -> {r['reference_sensor']} "
              f"quality={r['quality_score']:.2f} matches={r['num_matches']}")

    print("\n--- Summary stats ---")
    print(db.summary_stats())

    db.close()


if __name__ == "__main__":
    main()
