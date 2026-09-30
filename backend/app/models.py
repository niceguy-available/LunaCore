"""
models.py — Pydantic schemas for the API
ISRO SIH 2026 | Problem Statement 26166
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class EngineType(str, Enum):
    classical = "classical"
    deep = "deep"
    hybrid = "hybrid"


class IlluminationMethod(str, Enum):
    phase_congruency = "phase_congruency"
    dol = "dol"
    wld = "wld"
    clahe = "clahe"
    combined = "combined"


class DetectorType(str, Enum):
    sift = "sift"
    orb = "orb"
    akaze = "akaze"
    rift = "rift"


class TransformType(str, Enum):
    homography = "homography"
    affine = "affine"
    rigid = "rigid"
    similarity = "similarity"
    tps = "tps"
    polynomial = "polynomial"


class SubpixelMethod(str, Enum):
    parabolic = "parabolic"
    lucas_kanade = "lucas_kanade"
    phase_correlation = "phase_correlation"
    combined = "combined"


class RegistrationConfig(BaseModel):
    """Configuration for a registration job."""
    engine: EngineType = EngineType.classical
    illumination_method: IlluminationMethod = IlluminationMethod.phase_congruency
    feature_detector: DetectorType = DetectorType.sift
    matcher: str = "flann"
    ratio_threshold: float = Field(default=0.75, ge=0.3, le=1.0)
    ransac_method: str = "usac_magsac"
    ransac_threshold: float = Field(default=5.0, ge=1.0, le=20.0)
    transform_type: TransformType = TransformType.homography
    grid_size: int = Field(default=8, ge=2, le=20)
    max_per_cell: int = Field(default=80, ge=10, le=500)
    subpixel_method: SubpixelMethod = SubpixelMethod.parabolic
    refine_intensity: bool = True
    multiscale: bool = False
    source_sensor: Optional[str] = None
    reference_sensor: Optional[str] = None


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class StageUpdate(BaseModel):
    """Progress update for a pipeline stage."""
    stage: str
    status: str
    data: Dict[str, Any] = {}
    timestamp: float = 0.0


class JobResponse(BaseModel):
    """Response for a registration job."""
    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_stage: Optional[str] = None
    stages: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0.0


class SampleDataset(BaseModel):
    """Metadata for a sample image pair."""
    id: str
    name: str
    source_sensor: str
    reference_sensor: str
    source_path: str
    reference_path: str
    description: str = ""
    thumbnail_src: Optional[str] = None
    thumbnail_ref: Optional[str] = None


class CompareRequest(BaseModel):
    """Request to compare two engines on the same image pair."""
    job_id_a: str
    job_id_b: str


class MetricsExportFormat(str, Enum):
    json = "json"
    csv = "csv"
    geojson = "geojson"
