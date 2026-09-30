"""
Registration Router — Submit & Monitor Registration Jobs
"""

import cv2
import numpy as np
import asyncio
import json
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional

from ..tasks import task_manager
from ..models import RegistrationConfig, JobStatus

router = APIRouter()


@router.post("/register")
async def submit_registration(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    config_json: str = Form(default="{}"),
    source_sensor: Optional[str] = Form(default=None),
    reference_sensor: Optional[str] = Form(default=None),
):
    """
    Submit a registration job.
    
    Upload source and reference images along with pipeline configuration.
    Returns a job_id for tracking progress via SSE.
    """
    # Parse config
    try:
        config_dict = json.loads(config_json)
        config = RegistrationConfig(**config_dict)
    except Exception as e:
        raise HTTPException(400, f"Invalid config: {e}")

    # Read images
    src_bytes = await source.read()
    ref_bytes = await reference.read()

    src_arr = np.frombuffer(src_bytes, np.uint8)
    ref_arr = np.frombuffer(ref_bytes, np.uint8)

    src_img = cv2.imdecode(src_arr, cv2.IMREAD_UNCHANGED)
    ref_img = cv2.imdecode(ref_arr, cv2.IMREAD_UNCHANGED)

    if src_img is None or ref_img is None:
        raise HTTPException(400, "Could not decode one or both images")

    # Submit job
    job_id = task_manager.submit_job(
        src_img, ref_img, config,
        source_sensor=source_sensor,
        reference_sensor=reference_sensor,
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of a registration job."""
    job = task_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "current_stage": job["current_stage"],
        "stages": job["stages"],
        "error": job.get("error"),
        "created_at": job["created_at"],
    }


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get full results of a completed registration job."""
    job = task_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job["status"] == JobStatus.pending or job["status"] == JobStatus.running:
        return {"status": job["status"], "progress": job["progress"],
                "message": "Job still running"}

    if job["status"] == JobStatus.failed:
        return {"status": "failed", "error": job.get("error")}

    result = job.get("result", {})

    # Build response (strip heavy raw data, keep base64 images)
    return {
        "job_id": job_id,
        "status": "complete",
        "source_image": result.get("source_image", ""),
        "reference_image": result.get("reference_image", ""),
        "registered_image": result.get("registered_image", ""),
        "checkerboard": result.get("checkerboard", ""),
        "blended": result.get("blended", ""),
        "match_visualization": result.get("match_visualization", ""),
        "transformation_matrix": result.get("transformation_matrix"),
        "transformation_params": result.get("transformation_params", {}),
        "metrics": result.get("metrics", {}),
        "match_points": result.get("match_points", []),
        "keypoint_counts": result.get("keypoint_counts", {}),
        "subpixel_stats": result.get("subpixel_stats", {}),
        "processing_time": result.get("processing_time", 0),
        "config": job.get("config", {}),
    }


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events (SSE) stream for real-time job progress.
    
    Frontend connects to this endpoint and receives stage updates
    as they happen.
    """
    job = task_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    async def event_generator():
        last_event_count = 0
        while True:
            job = task_manager.get_job(job_id)
            if not job:
                break

            events = task_manager.get_events(job_id)
            new_events = events[last_event_count:]
            last_event_count = len(events)

            for event in new_events:
                # Limit data size for SSE (remove large base64 images)
                event_data = {**event}
                if "data" in event_data:
                    data = {**event_data["data"]}
                    # Keep thumbnails but remove large match visualizations from stream
                    for key in ["match_visualization"]:
                        if key in data and len(str(data[key])) > 10000:
                            data[key] = "[available via /result endpoint]"
                    event_data["data"] = data

                yield f"data: {json.dumps(event_data, default=str)}\n\n"

            # Check completion
            if job["status"] in (JobStatus.complete, JobStatus.failed):
                final = {
                    "stage": "final",
                    "status": str(job["status"]),
                    "quality_score": (job.get("result") or {}).get("metrics", {}).get("quality_score"),
                    "processing_time": (job.get("result") or {}).get("processing_time"),
                }
                yield f"data: {json.dumps(final)}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/matches")
async def download_matches(job_id: str, format: str = "csv"):
    """Download match points as CSV or GeoJSON."""
    job = task_manager.get_job(job_id)
    if not job or job["status"] != JobStatus.complete:
        raise HTTPException(404, "Job not found or not complete")

    match_points = (job.get("result") or {}).get("match_points", [])

    if format == "geojson":
        features = []
        for i, mp in enumerate(match_points):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [mp["src_x"], mp["src_y"]],
                        [mp["dst_x"], mp["dst_y"]]
                    ]
                },
                "properties": {"match_id": i}
            })
        geojson = {"type": "FeatureCollection", "features": features}
        return JSONResponse(geojson, headers={
            "Content-Disposition": f"attachment; filename=matches_{job_id}.geojson"
        })
    else:
        lines = ["idx,src_x,src_y,dst_x,dst_y"]
        for i, mp in enumerate(match_points):
            lines.append(f"{i},{mp['src_x']:.4f},{mp['src_y']:.4f},"
                         f"{mp['dst_x']:.4f},{mp['dst_y']:.4f}")
        csv_text = "\n".join(lines)
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=matches_{job_id}.csv"}
        )


@router.get("/jobs")
async def list_jobs():
    """List all registration jobs."""
    return task_manager.list_jobs()
