"""
Datasets Router — Upload & Manage Image Pairs
"""

import cv2
import numpy as np
import uuid
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, List

router = APIRouter()

UPLOAD_DIR = Path("static/uploads")
SAMPLES_DIR = Path("static/samples")

# Built-in sample datasets (populated on first run or via seed script)
SAMPLE_DATASETS = [
    {
        "id": "synthetic_pair_1",
        "name": "Synthetic Crater Field (Easy)",
        "source_sensor": "TMC2",
        "reference_sensor": "TMC2",
        "description": "Same-sensor pair with slight rotation/translation. Good for testing basic registration.",
    },
    {
        "id": "synthetic_cross_modal",
        "name": "Synthetic OHRC→TMC Cross-Sensor",
        "source_sensor": "OHRC",
        "reference_sensor": "TMC2",
        "description": "Cross-sensor pair with 20:1 scale ratio and different illumination. Tests multi-modal capability.",
    },
]


@router.post("/upload")
async def upload_images(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    source_sensor: Optional[str] = Form(default=None),
    reference_sensor: Optional[str] = Form(default=None),
    name: Optional[str] = Form(default=None),
):
    """Upload a pair of images for registration."""
    pair_id = str(uuid.uuid4())[:8]
    pair_dir = UPLOAD_DIR / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Save source
    src_path = pair_dir / f"source_{source.filename}"
    with open(src_path, "wb") as f:
        f.write(await source.read())

    # Save reference
    ref_path = pair_dir / f"reference_{reference.filename}"
    with open(ref_path, "wb") as f:
        f.write(await reference.read())

    # Generate thumbnails
    src_img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

    thumb_src = thumb_ref = None
    if src_img is not None:
        h, w = src_img.shape
        scale = min(1.0, 256 / max(h, w))
        thumb = cv2.resize(src_img, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(pair_dir / "thumb_source.jpg"), thumb)
        thumb_src = f"/static/uploads/{pair_id}/thumb_source.jpg"

    if ref_img is not None:
        h, w = ref_img.shape
        scale = min(1.0, 256 / max(h, w))
        thumb = cv2.resize(ref_img, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(pair_dir / "thumb_reference.jpg"), thumb)
        thumb_ref = f"/static/uploads/{pair_id}/thumb_reference.jpg"

    # Save metadata
    metadata = {
        "id": pair_id,
        "name": name or f"Upload {pair_id}",
        "source_sensor": source_sensor,
        "reference_sensor": reference_sensor,
        "source_path": str(src_path),
        "reference_path": str(ref_path),
        "source_filename": source.filename,
        "reference_filename": reference.filename,
        "source_shape": list(src_img.shape) if src_img is not None else None,
        "reference_shape": list(ref_img.shape) if ref_img is not None else None,
        "thumbnail_src": thumb_src,
        "thumbnail_ref": thumb_ref,
    }
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


@router.get("/samples")
async def list_samples():
    """List available sample image pairs."""
    samples = []

    # Check for uploaded pairs
    if UPLOAD_DIR.exists():
        for pair_dir in sorted(UPLOAD_DIR.iterdir()):
            meta_path = pair_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    samples.append(json.load(f))

    # Check for sample pairs
    if SAMPLES_DIR.exists():
        for pair_dir in sorted(SAMPLES_DIR.iterdir()):
            meta_path = pair_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    samples.append(json.load(f))

    return {"samples": samples, "built_in": SAMPLE_DATASETS}


@router.get("/sensors")
async def list_sensors():
    """List supported sensors and their metadata."""
    return {
        "sensors": [
            {"code": "OHRC", "name": "Orbiter High Resolution Camera",
             "platform": "Chandrayaan-2", "gsd_m": 0.25,
             "modality": "panchromatic", "swath": "3 km"},
            {"code": "TMC2", "name": "Terrain Mapping Camera-2",
             "platform": "Chandrayaan-2", "gsd_m": 5.0,
             "modality": "panchromatic", "swath": "20 km"},
            {"code": "IIRS", "name": "Imaging Infrared Spectrometer",
             "platform": "Chandrayaan-2", "gsd_m": 80.0,
             "modality": "hyperspectral", "swath": "20 km"},
            {"code": "LRO_NAC", "name": "Narrow Angle Camera",
             "platform": "LRO", "gsd_m": 0.5,
             "modality": "panchromatic", "swath": "5 km"},
            {"code": "LRO_WAC", "name": "Wide Angle Camera",
             "platform": "LRO", "gsd_m": 100.0,
             "modality": "multispectral", "swath": "100 km"},
        ]
    }


@router.get("/image/{pair_id}/{image_type}")
async def get_image(pair_id: str, image_type: str):
    """Serve an uploaded or sample image."""
    # Check uploads
    pair_dir = UPLOAD_DIR / pair_id
    if not pair_dir.exists():
        pair_dir = SAMPLES_DIR / pair_id
    if not pair_dir.exists():
        raise HTTPException(404, f"Pair {pair_id} not found")

    meta_path = pair_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "Metadata not found")

    with open(meta_path) as f:
        meta = json.load(f)

    if image_type == "source":
        path = meta.get("source_path")
    elif image_type == "reference":
        path = meta.get("reference_path")
    elif image_type == "thumb_source":
        path = str(pair_dir / "thumb_source.jpg")
    elif image_type == "thumb_reference":
        path = str(pair_dir / "thumb_reference.jpg")
    else:
        raise HTTPException(400, f"Unknown image type: {image_type}")

    if path and Path(path).exists():
        return FileResponse(path)
    raise HTTPException(404, "Image file not found")
