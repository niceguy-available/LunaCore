"""
db.py — SQLite persistence layer for the Lunar Image Registration System
ISRO SIH 2026 | Problem Statement 26166

Wraps schema.sql and exposes a small, dependency-free API that plugs
directly into pipeline.py's `result_dict` output. No ORM required —
just sqlite3 from the standard library.

Typical usage
-------------
    from db import RegistrationDB

    db = RegistrationDB("lunar_registration.db")   # creates file + schema if needed

    src_id = db.add_image("data/ohrc_source.png", sensor_code="OHRC",
                           sun_azimuth_deg=142.3, sun_elevation_deg=35.1)
    ref_id = db.add_image("data/tmc_reference.png", sensor_code="TMC2")

    config_id = db.add_config("default-sift-flann", full_config_dict)

    result = pipeline.register_pair(src_path, ref_path, out_path)
    run_id = db.log_run(src_id, ref_id, result, config_id=config_id)

    # querying
    best = db.top_runs(limit=10)
    history = db.runs_for_image(src_id)
"""

import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class RegistrationDB:
    def __init__(self, db_path="lunar_registration.db"):
        self.db_path = str(db_path)
        first_time = not Path(self.db_path).exists()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _init_schema(self):
        with open(SCHEMA_PATH, "r") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    @contextmanager
    def cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------
    # sensors
    # ------------------------------------------------------------------
    def get_sensor_id(self, code):
        row = self.conn.execute(
            "SELECT sensor_id FROM sensors WHERE code = ?", (code,)
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Unknown sensor code '{code}'. Add it via add_sensor() first."
            )
        return row["sensor_id"]

    def add_sensor(self, code, name, platform=None, modality=None, native_gsd_m=None, notes=None):
        with self.cursor() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO sensors (code, name, platform, modality, native_gsd_m, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (code, name, platform, modality, native_gsd_m, notes),
            )
        return self.get_sensor_id(code)

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------
    def add_image(self, file_path, sensor_code, **kwargs):
        """
        Register an image (source or reference) in the catalog.
        kwargs may include: original_url, role_default, width_px, height_px,
        bit_depth, gsd_m, sun_azimuth_deg, sun_elevation_deg,
        incidence_angle_deg, emission_angle_deg, center_lat, center_lon,
        footprint_geojson, acquisition_time, checksum_sha256
        """
        sensor_id = self.get_sensor_id(sensor_code)
        cols = ["sensor_id", "file_path"] + list(kwargs.keys())
        vals = [sensor_id, str(file_path)] + list(kwargs.values())
        placeholders = ", ".join(["?"] * len(vals))
        with self.cursor() as cur:
            cur.execute(
                f"""INSERT INTO images ({", ".join(cols)}) VALUES ({placeholders})
                    ON CONFLICT(file_path) DO UPDATE SET
                        sensor_id=excluded.sensor_id""",
                vals,
            )
        row = self.conn.execute(
            "SELECT image_id FROM images WHERE file_path = ?", (str(file_path),)
        ).fetchone()
        return row["image_id"]

    def get_image(self, image_id):
        row = self.conn.execute(
            "SELECT * FROM images WHERE image_id = ?", (image_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # configs
    # ------------------------------------------------------------------
    def add_config(self, name, config_dict):
        config_json = json.dumps(config_dict)
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO pipeline_configs
                       (name, config_json, feature_method, matcher_method, transformation_type)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET config_json = excluded.config_json""",
                (
                    name,
                    config_json,
                    config_dict.get("feature_method") or config_dict.get("feature_extraction", {}).get("method"),
                    config_dict.get("matcher_method") or config_dict.get("feature_matching", {}).get("method"),
                    config_dict.get("transformation_type") or config_dict.get("registration", {}).get("transformation_type"),
                ),
            )
        row = self.conn.execute(
            "SELECT config_id FROM pipeline_configs WHERE name = ?", (name,)
        ).fetchone()
        return row["config_id"]

    # ------------------------------------------------------------------
    # registration runs — main integration point with pipeline.py
    # ------------------------------------------------------------------
    def log_run(self, source_image_id, reference_image_id, result_dict,
                config_id=None, registered_image_path=None, status="success",
                error_message=None, store_matches=False, src_pts=None, dst_pts=None,
                transformation_type=None):
        """
        Persist one pipeline.register_pair() result.

        result_dict is the dict returned by LunarImageRegistrationPipeline.register_pair():
            transformation_matrix, transformation_params, keypoints_src,
            keypoints_ref, matches, metrics, quality_score, processing_time

        If result_dict is None (pipeline returned failure), pass status=
        'failed'/'insufficient_keypoints'/'insufficient_matches'/'error'
        and error_message instead.
        """
        if result_dict is None:
            with self.cursor() as cur:
                cur.execute(
                    """INSERT INTO registration_runs
                           (source_image_id, reference_image_id, config_id, status, error_message)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source_image_id, reference_image_id, config_id, status, error_message),
                )
                return cur.lastrowid

        params = result_dict.get("transformation_params", {}) or {}
        translation = params.get("translation", (None, None))
        metrics = result_dict.get("metrics", {}) or {}

        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO registration_runs (
                       source_image_id, reference_image_id, config_id, status,
                       keypoints_src, keypoints_ref, num_matches,
                       transformation_type, transformation_matrix,
                       translation_x, translation_y, rotation_deg, scale,
                       refinement_error, registered_image_path,
                       quality_score, processing_time_sec
                   ) VALUES (?, ?, ?, 'success', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_image_id, reference_image_id, config_id,
                    result_dict.get("keypoints_src"),
                    result_dict.get("keypoints_ref"),
                    result_dict.get("matches"),
                    transformation_type,
                    json.dumps(result_dict.get("transformation_matrix")),
                    translation[0] if translation else None,
                    translation[1] if translation else None,
                    params.get("rotation"),
                    params.get("scale"),
                    result_dict.get("refinement_error"),
                    str(registered_image_path) if registered_image_path else None,
                    result_dict.get("quality_score"),
                    result_dict.get("processing_time"),
                ),
            )
            run_id = cur.lastrowid

            cur.execute(
                """INSERT INTO metrics (run_id, rmse, mae, correlation, ssim, mutual_info, gradient_corr)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    metrics.get("rmse"),
                    metrics.get("mae"),
                    metrics.get("correlation"),
                    metrics.get("ssim"),
                    metrics.get("mutual_info"),
                    metrics.get("gradient_corr"),
                ),
            )

            if store_matches and src_pts is not None and dst_pts is not None:
                rows = [
                    (run_id, float(s[0]), float(s[1]), float(d[0]), float(d[1]), None)
                    for s, d in zip(src_pts, dst_pts)
                ]
                cur.executemany(
                    """INSERT INTO matches (run_id, src_x, src_y, dst_x, dst_y, distance)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    rows,
                )

        return run_id

    # ------------------------------------------------------------------
    # batch jobs
    # ------------------------------------------------------------------
    def start_batch(self, job_type, label=None, notes=None):
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO batch_jobs (job_type, label, notes) VALUES (?, ?, ?)",
                (job_type, label, notes),
            )
            return cur.lastrowid

    def finish_batch(self, batch_id):
        with self.cursor() as cur:
            cur.execute(
                "UPDATE batch_jobs SET finished_at = datetime('now') WHERE batch_id = ?",
                (batch_id,),
            )

    def link_run_to_batch(self, batch_id, run_id):
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO batch_job_runs (batch_id, run_id) VALUES (?, ?)",
                (batch_id, run_id),
            )

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def get_run(self, run_id):
        row = self.conn.execute(
            "SELECT * FROM v_run_report WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def top_runs(self, limit=10, min_matches=0):
        rows = self.conn.execute(
            """SELECT * FROM v_run_report
               WHERE status = 'success' AND num_matches >= ?
               ORDER BY quality_score DESC
               LIMIT ?""",
            (min_matches, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def runs_for_image(self, image_id):
        rows = self.conn.execute(
            """SELECT * FROM v_run_report
               WHERE source_path = (SELECT file_path FROM images WHERE image_id = ?)
                  OR reference_path = (SELECT file_path FROM images WHERE image_id = ?)
               ORDER BY created_at DESC""",
            (image_id, image_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def runs_by_sensor_pair(self, source_sensor_code, reference_sensor_code):
        rows = self.conn.execute(
            """SELECT * FROM v_run_report
               WHERE source_sensor = ? AND reference_sensor = ?
               ORDER BY quality_score DESC""",
            (source_sensor_code, reference_sensor_code),
        ).fetchall()
        return [dict(r) for r in rows]

    def summary_stats(self):
        row = self.conn.execute(
            """SELECT
                   COUNT(*) AS total_runs,
                   SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS successful_runs,
                   AVG(CASE WHEN status='success' THEN quality_score END) AS avg_quality_score,
                   AVG(CASE WHEN status='success' THEN processing_time_sec END) AS avg_processing_time
               FROM registration_runs"""
        ).fetchone()
        return dict(row)
