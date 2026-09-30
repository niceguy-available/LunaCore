"""
Evaluation Router — Compare engines & export reports
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional

from ..tasks import task_manager
from ..models import JobStatus

router = APIRouter()


@router.get("/evaluate/{job_id}")
async def get_evaluation(job_id: str):
    """Get detailed evaluation metrics for a completed job."""
    job = task_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != JobStatus.complete:
        raise HTTPException(400, "Job not yet complete")

    result = job.get("result", {})
    metrics = result.get("metrics", {})

    return {
        "job_id": job_id,
        "metrics": metrics,
        "match_points_count": len(result.get("match_points", [])),
        "config": job.get("config", {}),
    }


@router.get("/compare")
async def compare_jobs(job_a: str, job_b: str):
    """Compare metrics from two registration jobs side by side."""
    a = task_manager.get_job(job_a)
    b = task_manager.get_job(job_b)

    if not a or not b:
        raise HTTPException(404, "One or both jobs not found")
    if a["status"] != JobStatus.complete or b["status"] != JobStatus.complete:
        raise HTTPException(400, "Both jobs must be complete")

    result_a = a.get("result", {})
    result_b = b.get("result", {})

    metrics_a = result_a.get("metrics", {})
    metrics_b = result_b.get("metrics", {})

    # Build comparison
    comparison = {
        "job_a": {
            "job_id": job_a,
            "config": a.get("config", {}),
            "metrics": metrics_a,
            "processing_time": result_a.get("processing_time", 0),
            "match_visualization": result_a.get("match_visualization", ""),
            "registered_image": result_a.get("registered_image", ""),
            "checkerboard": result_a.get("checkerboard", ""),
        },
        "job_b": {
            "job_id": job_b,
            "config": b.get("config", {}),
            "metrics": metrics_b,
            "processing_time": result_b.get("processing_time", 0),
            "match_visualization": result_b.get("match_visualization", ""),
            "registered_image": result_b.get("registered_image", ""),
            "checkerboard": result_b.get("checkerboard", ""),
        },
        "winner": {},
    }

    # Determine winner for each metric
    img_a = metrics_a.get("image_metrics", {})
    img_b = metrics_b.get("image_metrics", {})

    for metric in ["correlation", "ssim", "gradient_corr"]:
        va = img_a.get(metric, 0)
        vb = img_b.get(metric, 0)
        comparison["winner"][metric] = "a" if va >= vb else "b"

    for metric in ["rmse", "mae"]:
        va = img_a.get(metric, float('inf'))
        vb = img_b.get(metric, float('inf'))
        comparison["winner"][metric] = "a" if va <= vb else "b"

    # Overall
    qa = metrics_a.get("quality_score", 0)
    qb = metrics_b.get("quality_score", 0)
    comparison["winner"]["overall"] = "a" if qa >= qb else "b"

    return comparison


@router.get("/export/{job_id}")
async def export_report(job_id: str, format: str = "json"):
    """Export evaluation report as JSON, CSV, or GeoJSON."""
    job = task_manager.get_job(job_id)
    if not job or job["status"] != JobStatus.complete:
        raise HTTPException(404, "Job not found or not complete")

    result = job.get("result", {})

    if format == "json":
        report = {
            "job_id": job_id,
            "config": job.get("config", {}),
            "metrics": result.get("metrics", {}),
            "transformation_params": result.get("transformation_params", {}),
            "keypoint_counts": result.get("keypoint_counts", {}),
            "match_points_count": len(result.get("match_points", [])),
            "processing_time": result.get("processing_time", 0),
        }
        return JSONResponse(report, headers={
            "Content-Disposition": f"attachment; filename=report_{job_id}.json"
        })

    elif format == "csv":
        # Flatten metrics to CSV
        metrics = result.get("metrics", {})
        img = metrics.get("image_metrics", {})
        match = metrics.get("match_stats", {})
        reproj = metrics.get("reprojection", {})
        dist = metrics.get("distribution", {})

        lines = [
            "metric,value",
            f"quality_score,{metrics.get('quality_score', 0)}",
            f"rmse,{img.get('rmse', 0)}",
            f"mae,{img.get('mae', 0)}",
            f"correlation,{img.get('correlation', 0)}",
            f"ssim,{img.get('ssim', 0)}",
            f"mutual_info,{img.get('mutual_info', 0)}",
            f"gradient_corr,{img.get('gradient_corr', 0)}",
            f"total_matches,{match.get('total_matches', 0)}",
            f"inlier_count,{match.get('inlier_count', 0)}",
            f"inlier_ratio,{match.get('inlier_ratio', 0)}",
            f"reproj_rmse,{reproj.get('rmse', 0)}",
            f"reproj_mean,{reproj.get('mean', 0)}",
            f"sub_pixel_ratio,{reproj.get('sub_pixel_ratio', 0)}",
            f"uniformity_score,{dist.get('uniformity_score', 0)}",
            f"grid_occupancy,{dist.get('occupancy_ratio', 0)}",
            f"processing_time,{result.get('processing_time', 0)}",
        ]
        csv_text = "\n".join(lines)
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=metrics_{job_id}.csv"}
        )

    raise HTTPException(400, f"Unsupported format: {format}")
