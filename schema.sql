-- =====================================================================
-- Lunar Image Registration System — Database Schema (SQLite)
-- ISRO SIH 2026 | Problem Statement 26166
-- =====================================================================
-- Tracks datasets/images (OHRC, TMC-2, IIRS, LRO-NAC), every registration
-- run against the pipeline, the resulting transformation, quality
-- metrics, and (optionally) per-run match/keypoint detail.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- 1. sensors — lookup of the imaging payloads referenced by the PS
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    code          TEXT NOT NULL UNIQUE,      -- e.g. 'OHRC', 'TMC2', 'IIRS', 'LRO_NAC', 'SELENE'
    name          TEXT NOT NULL,             -- full name
    platform      TEXT,                      -- 'Chandrayaan-2', 'LRO', 'SELENE'
    modality      TEXT,                      -- 'panchromatic','multispectral','hyperspectral'
    native_gsd_m  REAL,                      -- ground sample distance / resolution in metres
    notes         TEXT
);

-- ---------------------------------------------------------------------
-- 2. images — every source/reference image ingested into the system
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS images (
    image_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_id       INTEGER NOT NULL REFERENCES sensors(sensor_id),
    file_path       TEXT NOT NULL,           -- local/relative path or URI
    original_url    TEXT,                    -- source download link (chmapbrowse, lroc, etc.)
    role_default    TEXT CHECK (role_default IN ('source','reference','either')) DEFAULT 'either',
    width_px        INTEGER,
    height_px       INTEGER,
    bit_depth       INTEGER,
    gsd_m           REAL,                    -- actual resolution of this image
    sun_azimuth_deg REAL,                    -- illumination geometry
    sun_elevation_deg REAL,
    incidence_angle_deg REAL,
    emission_angle_deg  REAL,
    center_lat      REAL,
    center_lon      REAL,
    footprint_geojson TEXT,                  -- optional polygon footprint as GeoJSON text
    acquisition_time  TEXT,                  -- ISO-8601 timestamp
    checksum_sha256   TEXT,                  -- integrity check for the file
    created_at        TEXT DEFAULT (datetime('now')),
    UNIQUE (file_path)
);

CREATE INDEX IF NOT EXISTS idx_images_sensor ON images(sensor_id);
CREATE INDEX IF NOT EXISTS idx_images_latlon  ON images(center_lat, center_lon);

-- ---------------------------------------------------------------------
-- 3. pipeline_configs — named/versioned config snapshots (config.json)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_configs (
    config_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    config_json   TEXT NOT NULL,             -- full JSON blob as-is
    feature_method        TEXT,              -- denormalized for quick filtering
    matcher_method         TEXT,
    transformation_type    TEXT,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE (name)
);

-- ---------------------------------------------------------------------
-- 4. registration_runs — one row per pipeline.register_pair() call
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS registration_runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_image_id   INTEGER NOT NULL REFERENCES images(image_id),
    reference_image_id INTEGER NOT NULL REFERENCES images(image_id),
    config_id         INTEGER REFERENCES pipeline_configs(config_id),
    status            TEXT CHECK (status IN ('success','failed','insufficient_keypoints',
                                              'insufficient_matches','error')) NOT NULL,
    error_message     TEXT,

    -- feature extraction stats
    keypoints_src     INTEGER,
    keypoints_ref     INTEGER,
    num_matches       INTEGER,

    -- transformation output
    transformation_type TEXT,                -- homography / affine / rigid / similarity
    transformation_matrix TEXT,               -- JSON-serialized 3x3 matrix
    translation_x      REAL,
    translation_y      REAL,
    rotation_deg        REAL,
    scale               REAL,
    refinement_error     REAL,

    -- outputs
    registered_image_path TEXT,
    quality_score          REAL,             -- 0-100
    processing_time_sec    REAL,

    created_at         TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_src   ON registration_runs(source_image_id);
CREATE INDEX IF NOT EXISTS idx_runs_ref   ON registration_runs(reference_image_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON registration_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_quality ON registration_runs(quality_score);

-- ---------------------------------------------------------------------
-- 5. metrics — validation.py output, one row per run (1:1, but kept
--    separate from registration_runs for clarity / easy extension)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metrics (
    metric_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL UNIQUE REFERENCES registration_runs(run_id) ON DELETE CASCADE,
    rmse            REAL,
    mae             REAL,
    correlation     REAL,
    ssim            REAL,
    mutual_info     REAL,
    gradient_corr   REAL
);

-- ---------------------------------------------------------------------
-- 6. matches — optional fine-grained keypoint correspondences per run
--    (populate only if you need point-level auditing/visual overlays;
--     safe to leave empty for normal operation)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    match_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES registration_runs(run_id) ON DELETE CASCADE,
    src_x        REAL NOT NULL,
    src_y        REAL NOT NULL,
    dst_x        REAL NOT NULL,
    dst_y        REAL NOT NULL,
    distance     REAL              -- descriptor distance / match quality
);

CREATE INDEX IF NOT EXISTS idx_matches_run ON matches(run_id);

-- ---------------------------------------------------------------------
-- 7. batch_jobs — groups many runs together (register_series / batch_register)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type      TEXT CHECK (job_type IN ('series','batch','single')) NOT NULL,
    label         TEXT,
    started_at    TEXT DEFAULT (datetime('now')),
    finished_at   TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS batch_job_runs (
    batch_id   INTEGER NOT NULL REFERENCES batch_jobs(batch_id) ON DELETE CASCADE,
    run_id     INTEGER NOT NULL REFERENCES registration_runs(run_id) ON DELETE CASCADE,
    PRIMARY KEY (batch_id, run_id)
);

-- ---------------------------------------------------------------------
-- Convenience view: full run report joining images, metrics, config
-- ---------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_run_report AS
SELECT
    r.run_id,
    r.status,
    src.file_path   AS source_path,
    ssens.code      AS source_sensor,
    ref.file_path   AS reference_path,
    rsens.code      AS reference_sensor,
    r.transformation_type,
    r.keypoints_src,
    r.keypoints_ref,
    r.num_matches,
    r.quality_score,
    r.processing_time_sec,
    m.rmse, m.mae, m.correlation, m.ssim, m.mutual_info, m.gradient_corr,
    r.translation_x, r.translation_y, r.rotation_deg, r.scale,
    pc.name AS config_name,
    r.created_at
FROM registration_runs r
JOIN images src   ON src.image_id = r.source_image_id
JOIN images ref   ON ref.image_id = r.reference_image_id
JOIN sensors ssens ON ssens.sensor_id = src.sensor_id
JOIN sensors rsens ON rsens.sensor_id = ref.sensor_id
LEFT JOIN metrics m       ON m.run_id = r.run_id
LEFT JOIN pipeline_configs pc ON pc.config_id = r.config_id;

-- ---------------------------------------------------------------------
-- Seed the sensor lookup with the payloads named in the problem statement
-- ---------------------------------------------------------------------
INSERT OR IGNORE INTO sensors (code, name, platform, modality, native_gsd_m) VALUES
    ('OHRC',   'Orbiter High Resolution Camera', 'Chandrayaan-2', 'panchromatic', 0.25),
    ('TMC2',   'Terrain Mapping Camera-2',       'Chandrayaan-2', 'panchromatic', 5.0),
    ('IIRS',   'Imaging Infrared Spectrometer',  'Chandrayaan-2', 'hyperspectral', 80.0),
    ('LRO_NAC','Narrow Angle Camera',            'LRO',           'panchromatic', 0.5),
    ('SELENE', 'SELENE Terrain Camera',          'SELENE (Kaguya)', 'panchromatic', 10.0);
