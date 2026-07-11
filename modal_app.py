"""Modal deployment entrypoint for the ANPR FastAPI app.

Model weights are NOT baked into the image or committed to git — they're pushed to
a Modal Volume once, ahead of time:

    modal volume create anpr-weights
    modal volume put anpr-weights best_carplateocr_30062026.pt /best_carplateocr_30062026.pt

Then deploy:

    modal deploy modal_app.py
"""

import os

import modal

WEIGHTS_VOLUME_NAME = "anpr-weights"
WEIGHTS_FILENAME = "best_carplateocr_30062026.pt"

app = modal.App("anpr")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi==0.139.0",
        "uvicorn[standard]==0.51.0",
        "python-multipart==0.0.32",
        "opencv-python-headless==5.0.0.93",
        "numpy==2.5.1",
        "ultralytics==8.3.253",
        "fast-plate-ocr[onnx]==1.1.0",
    )
    .add_local_file("anpr.py", remote_path="/root/anpr.py")
    .add_local_dir("app", remote_path="/root/app")
    .add_local_dir("frontend", remote_path="/root/frontend")
)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)

# Free-tier demo constraints: the job store in app/jobs.py is in-memory and files
# live on local container disk, so this only works correctly with a single
# container instance. max_containers=1 pins that; scaledown_window keeps the
# container (and its in-progress background jobs) alive across the polling
# interval the frontend uses.
@app.function(
    image=image,
    volumes={"/vol": weights_volume},
    timeout=1800,
    max_containers=1,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    import anpr

    anpr.MODEL_PATH = f"/vol/{WEIGHTS_FILENAME}"

    from app.main import app as web_app

    return web_app
