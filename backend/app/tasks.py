"""
tasks.py — Background Job Manager
ISRO SIH 2026 | Problem Statement 26166

Manages registration jobs using asyncio + ThreadPoolExecutor.
Jobs run in a background thread to avoid blocking the API server,
and emit real-time stage events via Server-Sent Events (SSE).
"""

import asyncio
import uuid
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, List
from pathlib import Path

from .pipeline import RegistrationOrchestrator
from .models import JobStatus, RegistrationConfig


class JobStore:
    """In-memory job storage (sufficient for hackathon demo)."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, List[Dict[str, Any]]] = {}

    def create(self, job_id: str, config: Dict) -> Dict:
        job = {
            "job_id": job_id,
            "status": JobStatus.pending,
            "progress": 0.0,
            "current_stage": None,
            "stages": {},
            "result": None,
            "error": None,
            "config": config,
            "created_at": time.time(),
        }
        self._jobs[job_id] = job
        self._events[job_id] = []
        return job

    def get(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs):
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    def add_event(self, job_id: str, event: Dict):
        if job_id in self._events:
            self._events[job_id].append(event)

    def get_events(self, job_id: str) -> List[Dict]:
        return self._events.get(job_id, [])

    def list_all(self) -> List[Dict]:
        return list(self._jobs.values())


class TaskManager:
    """Manages background registration jobs."""

    STAGE_WEIGHTS = {
        "preprocessing": 0.15,
        "feature_detection": 0.20,
        "feature_matching": 0.20,
        "transform_estimation": 0.15,
        "subpixel_refinement": 0.15,
        "evaluation": 0.15,
    }

    def __init__(self, max_workers: int = 2):
        self.store = JobStore()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._upload_dir = Path("static/uploads")
        self._results_dir = Path("static/results")
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def submit_job(self, source_image: np.ndarray,
                   reference_image: np.ndarray,
                   config: RegistrationConfig,
                   source_sensor: Optional[str] = None,
                   reference_sensor: Optional[str] = None,
                   ) -> str:
        """Submit a new registration job. Returns job ID."""
        job_id = str(uuid.uuid4())[:8]
        config_dict = config.model_dump()
        self.store.create(job_id, config_dict)

        # Run in background thread
        self.executor.submit(
            self._run_job, job_id, source_image, reference_image,
            config_dict, source_sensor, reference_sensor
        )

        return job_id

    def _run_job(self, job_id: str,
                 source_image: np.ndarray,
                 reference_image: np.ndarray,
                 config: Dict,
                 source_sensor: Optional[str],
                 reference_sensor: Optional[str]):
        """Execute a registration job in background thread."""
        self.store.update(job_id, status=JobStatus.running)

        completed_stages = set()

        def stage_callback(stage: str, status: str, data: Dict):
            event = {
                "stage": stage,
                "status": status,
                "data": data,
                "timestamp": time.time(),
            }
            self.store.add_event(job_id, event)
            self.store.update(
                job_id,
                current_stage=stage,
                stages={**self.store.get(job_id)["stages"], stage: {"status": status, **data}},
            )

            if status == "complete":
                completed_stages.add(stage)
                progress = sum(
                    self.STAGE_WEIGHTS.get(s, 0)
                    for s in completed_stages
                )
                self.store.update(job_id, progress=min(progress, 0.99))

        try:
            orchestrator = RegistrationOrchestrator(config)
            result = orchestrator.register(
                source_image, reference_image,
                callback=stage_callback,
                source_sensor=source_sensor,
                reference_sensor=reference_sensor,
            )

            if result["status"] == "complete":
                # Save registered image to disk
                result_dir = self._results_dir / job_id
                result_dir.mkdir(parents=True, exist_ok=True)

                raw_reg = result.pop("_raw_registered", None)
                raw_src = result.pop("_raw_source", None)
                raw_ref = result.pop("_raw_reference", None)
                raw_vis = result.pop("_raw_match_vis", None)

                if raw_reg is not None:
                    cv2.imwrite(str(result_dir / "registered.png"), raw_reg)
                if raw_src is not None:
                    cv2.imwrite(str(result_dir / "source.png"), raw_src)
                if raw_ref is not None:
                    cv2.imwrite(str(result_dir / "reference.png"), raw_ref)
                if raw_vis is not None:
                    cv2.imwrite(str(result_dir / "match_visualization.png"), raw_vis)

                # Save metrics JSON
                import json
                metrics_path = result_dir / "metrics.json"
                metrics_data = result.get("metrics", {})
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=2, default=str)

                # Save match points CSV
                match_points = result.get("match_points", [])
                if match_points:
                    csv_lines = ["idx,src_x,src_y,dst_x,dst_y"]
                    for i, mp in enumerate(match_points):
                        csv_lines.append(
                            f"{i},{mp['src_x']:.4f},{mp['src_y']:.4f},"
                            f"{mp['dst_x']:.4f},{mp['dst_y']:.4f}"
                        )
                    with open(result_dir / "matches.csv", "w") as f:
                        f.write("\n".join(csv_lines))

                self.store.update(
                    job_id,
                    status=JobStatus.complete,
                    progress=1.0,
                    result=result,
                )
            else:
                self.store.update(
                    job_id,
                    status=JobStatus.failed,
                    error=result.get("error", "unknown"),
                    result=result,
                )

        except Exception as e:
            import traceback
            self.store.update(
                job_id,
                status=JobStatus.failed,
                error=str(e),
            )
            traceback.print_exc()

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.store.get(job_id)

    def get_events(self, job_id: str) -> List[Dict]:
        return self.store.get_events(job_id)

    def list_jobs(self) -> List[Dict]:
        jobs = self.store.list_all()
        # Return lightweight summaries
        return [{
            "job_id": j["job_id"],
            "status": j["status"],
            "progress": j["progress"],
            "current_stage": j["current_stage"],
            "created_at": j["created_at"],
            "quality_score": (j.get("result") or {}).get("metrics", {}).get("quality_score"),
        } for j in jobs]


# Global singleton task manager instance
task_manager = TaskManager(max_workers=2)
