"""In-memory job store and background execution for ANPR video processing."""

import os
import time
import uuid
from typing import Optional

import cv2

import anpr

THUMB_COUNT = 4

# CPU-friendly defaults chosen from the Phase 0 benchmark (imgsz=1920 was ~135min
# for a 60s clip; imgsz=960/stride=2 matched imgsz=960/stride=1 plate detections
# at ~2x the speed).
DEFAULT_IMGSZ = 960
DEFAULT_FRAME_STRIDE = 2

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGIONS = {"US", "UK", "HK", "AUTO"}

JOBS: dict[str, dict] = {}


def create_job(input_path: str, region: str, fps: float, width: int, height: int, total_frames: int) -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "region": region,
        "input_path": input_path,
        "output_path": os.path.join(OUTPUT_DIR, f"{job_id}.mp4"),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames_processed": 0,
        "created_at": time.time(),
        "error": None,
        "result": None,
        "thumbs": [],
    }
    return job_id


def _extract_thumbnails(video_path: str, job_id: str) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    paths = []
    for i in range(THUMB_COUNT):
        frame_idx = min(total - 1, int((i + 1) * total / (THUMB_COUNT + 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        thumb_path = os.path.join(OUTPUT_DIR, f"{job_id}_thumb_{i}.jpg")
        cv2.imwrite(thumb_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(thumb_path)
    cap.release()
    return paths


def get_job(job_id: str) -> Optional[dict]:
    return JOBS.get(job_id)


def run_job(job_id: str) -> None:
    job = JOBS[job_id]
    job["status"] = "processing"

    def on_progress(frame_num, total_frames, locked_plates):
        job["frames_processed"] = frame_num
        job["total_frames"] = total_frames or job["total_frames"]

    try:
        result = anpr.process_video(
            video_path=job["input_path"],
            output_path=job["output_path"],
            region=job["region"],
            imgsz=DEFAULT_IMGSZ,
            frame_stride=DEFAULT_FRAME_STRIDE,
            progress_callback=on_progress,
        )
        final_video = anpr.reencode_h264(result["output_path"])
        result["output_path"] = final_video
        job["output_path"] = final_video
        job["result"] = result
        job["frames_processed"] = result["frames_processed"]
        job["thumbs"] = _extract_thumbnails(final_video, job_id)
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        if os.path.exists(job["input_path"]):
            os.remove(job["input_path"])
