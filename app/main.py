"""FastAPI backend for the ANPR web app.

Upload a video + region -> async background job -> poll for progress ->
fetch results JSON + download the annotated video.
"""

import os
import shutil
import uuid

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import jobs
import anpr

app = FastAPI(title="Bright Node ANPR")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.on_event("startup")
def warm_models():
    anpr.get_detector()
    anpr.get_ocr()


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...), region: str = Form("AUTO")):
    region = region.upper()
    if region not in jobs.REGIONS:
        raise HTTPException(status_code=422, detail=f"region must be one of {sorted(jobs.REGIONS)}")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    input_path = os.path.join(jobs.UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(input_path, "wb") as dest:
        shutil.copyfileobj(file.file, dest)

    cap = cv2.VideoCapture(input_path)
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not opened:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="Could not read this video. The file may be corrupt or use a codec the server can't decode.")

    job_id = jobs.create_job(input_path, region, fps, width, height, total_frames)
    background_tasks.add_task(jobs.run_job, job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    payload = {
        "id": job["id"],
        "status": job["status"],
        "region": job["region"],
        "fps": job["fps"],
        "width": job["width"],
        "height": job["height"],
        "total_frames": job["total_frames"],
        "frames_processed": job["frames_processed"],
        "error": job["error"],
    }
    if job["status"] == "done":
        payload["result"] = job["result"]
        payload["video_url"] = f"/api/jobs/{job_id}/video"
        payload["thumb_urls"] = [f"/api/jobs/{job_id}/thumb/{i}" for i in range(len(job["thumbs"]))]
    return JSONResponse(payload)


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="job is not finished yet")
    return FileResponse(job["output_path"], media_type="video/mp4", filename="annotated.mp4")


@app.get("/api/jobs/{job_id}/thumb/{index}")
async def job_thumb(job_id: str, index: int):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if index < 0 or index >= len(job["thumbs"]):
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return FileResponse(job["thumbs"][index], media_type="image/jpeg")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
