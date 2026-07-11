"""
ANPR (Automatic Number Plate Recognition) — cleaned from the Kaggle notebook.

Pipeline:
    YOLOv8 plate detector  ->  fast-plate-ocr  ->  temporal stabilization (voting/locking)

Regions supported (map to the web app's user-facing options):
    USA -> "US" | UK -> "UK" | Hong Kong -> "HK" | Universal -> "AUTO"

Requirements:
    pip install ultralytics "fast-plate-ocr[onnx]" opencv-python numpy
    (ffmpeg on PATH is optional — used only for the final H.264 re-encode)

All Kaggle paths removed. Model weights and the sample video are expected in this folder.
The core detection/OCR/stabilization logic is unchanged from the notebook so accuracy is preserved.
"""

import os
import re
import shutil
import builtins
import subprocess
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# --------------------------------------------------------------------------------------
# Paths (local — no Kaggle)
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_carplateocr_30062026.pt")
DEFAULT_VIDEO = os.path.join(BASE_DIR, "sample_video.mp4")

# --------------------------------------------------------------------------------------
# Model loading (lazy + cached so importing this module is cheap and web-app friendly)
# --------------------------------------------------------------------------------------
_model = None
_plate_ocr = None


def get_detector():
    """Load and cache the YOLOv8 plate detector."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def get_ocr():
    """Load and cache the fast-plate-ocr recognizer (CPU, ONNX)."""
    global _plate_ocr
    if _plate_ocr is None:
        _plate_ocr = LicensePlateRecognizer("cct-s-v2-global-model")
    return _plate_ocr


# --------------------------------------------------------------------------------------
# Region validation / patterns
# --------------------------------------------------------------------------------------
BLOCKLIST = {"POLICE", "TAXI", "TAX1", "STOP", "BUS", "SLOW", "AMBULANCE", "FIRE", "SCHOOL"}

ALL_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$"),   # UK current:        AB12CDE
    re.compile(r"^[A-Z]\d{1,3}[A-Z]{3}$"),    # UK older prefix:   A123BCD
    re.compile(r"^[A-Z]{1,3}\d{1,4}$"),       # HK / Asian:        SM7080, FX18
    re.compile(r"^[A-Z]{3}\d{3,4}$"),         # US common / EU:    ABC1234
    re.compile(r"^\d{3}[A-Z]{3}$"),           # US variant:        123ABC
    re.compile(r"^[A-Z]{2}\d{3,4}$"),         # EU / generic:      AB1234
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}$"),   # older EU:          AB12CD
    re.compile(r"^\d{1,4}[A-Z]{1,3}$"),       # digits-first
    re.compile(r"^[A-Z]\d{3}[A-Z]{2}$"),      # mixed variant:     A123BC
    re.compile(r"^[A-Z]{2}\d{5}$"),           # India-style tail:  AB12345
    re.compile(r"^[A-Z]{3}\d{4}$"),           # Australia / others:ABC1234
]

REGION_PATTERNS = {
    "UK":    [ALL_PATTERNS[0], ALL_PATTERNS[1]],
    "HK":    [ALL_PATTERNS[2]],
    "US":    [ALL_PATTERNS[3], ALL_PATTERNS[4]],
    "EU":    [ALL_PATTERNS[5], ALL_PATTERNS[6]],
    "INDIA": [ALL_PATTERNS[9]],
    "AUTO":  ALL_PATTERNS,
}


def validate_plate(text, region="AUTO"):
    """Return the cleaned plate string if it is valid for the given region, else None."""
    if not text or text == "N/A":
        return None
    cleaned = "".join(c for c in text.upper() if c.isalnum())
    if cleaned in BLOCKLIST:
        return None
    if not (4 <= len(cleaned) <= 8):
        return None
    patterns = REGION_PATTERNS.get(region, ALL_PATTERNS)
    if any(p.match(cleaned) for p in patterns):
        return cleaned
    return None


# --------------------------------------------------------------------------------------
# OCR correction helpers
# --------------------------------------------------------------------------------------
DIGIT_FIX = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6", "Z": "2",
             "Q": "0", "M": "11", "W": "11", "D": "0", "A": "4", "T": "7"}
LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"}


def smart_correct(text):
    """Heuristic fix: first two chars as letters, remainder as digits (AB1234-style)."""
    if not text or len(text) < 3:
        return None
    t = "".join(c for c in text.upper() if c.isalnum())
    if len(t) < 3:
        return None
    letters = "".join(ch if ch.isalpha() else LETTER_FIX.get(ch, ch) for ch in t[:2])
    digits = "".join(ch if ch.isdigit() else DIGIT_FIX.get(ch, ch) for ch in t[2:])
    cand = letters + digits
    if re.match(r"^[A-Z]{2}\d{2,4}$", cand):
        return cand
    return None


# --------------------------------------------------------------------------------------
# Plate stabilizer (temporal voting / locking across frames)
# --------------------------------------------------------------------------------------
class PlateStabilizer:
    def __init__(self, window=30, min_confidence=0.15, min_length=3,
                 lock_threshold=5, persist_frames=90):
        self.window = window
        self.min_conf = min_confidence
        self.min_length = min_length
        self.lock_threshold = lock_threshold
        self.persist_frames = persist_frames
        self.histories = {}
        self.locked_text = {}
        self.lock_count = {}
        self.last_seen = {}

    def update(self, plate_id, text, conf, frame_num):
        self.last_seen[plate_id] = frame_num
        if plate_id not in self.histories:
            self.histories[plate_id] = deque(maxlen=self.window)
            self.lock_count[plate_id] = 0
        if not text or conf < self.min_conf or len(text) < self.min_length:
            return
        self.histories[plate_id].append(text)
        counter = Counter(self.histories[plate_id])
        best_text, votes = counter.most_common(1)[0]

        if plate_id not in self.locked_text:
            total = sum(counter.values())
            if votes >= self.lock_threshold or (total >= 6 and votes >= 3):
                self.locked_text[plate_id] = best_text
                self.lock_count[plate_id] = votes
        else:
            if best_text != self.locked_text[plate_id] and votes >= self.lock_threshold + 2:
                self.locked_text[plate_id] = best_text
            elif best_text == self.locked_text[plate_id]:
                self.lock_count[plate_id] += 1

    def get_stable(self, plate_id, frame_num):
        last = self.last_seen.get(plate_id, 0)
        if frame_num - last > self.persist_frames:
            return None
        return self.locked_text.get(plate_id, None)

    def reset_plate(self, plate_id):
        self.histories.pop(plate_id, None)
        self.locked_text.pop(plate_id, None)
        self.lock_count.pop(plate_id, None)
        self.last_seen.pop(plate_id, None)

    def clear_lost(self, active_ids, frame_num):
        for pid in list(self.last_seen.keys()):
            last = self.last_seen.get(pid, 0)
            if pid not in active_ids and frame_num - last > self.persist_frames:
                self.reset_plate(pid)


# --------------------------------------------------------------------------------------
# Image / geometry helpers
# --------------------------------------------------------------------------------------
def preprocess_plate(crop):
    """Optional enhancement of a plate crop (grayscale, denoise, CLAHE, sharpen, threshold)."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop
    new_w = builtins.max(1, int(w * (120 / h)))
    crop = cv2.resize(crop, (new_w, 120), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def read_plate(crop):
    """Run OCR on a plate crop. Returns (text, confidence)."""
    if crop is None or crop.size == 0:
        return None, 0.0
    try:
        preds = get_ocr().run(crop)
        if preds:
            text = preds[0].plate.upper().strip()
            text = "".join(c for c in text if c.isalnum())
            if text:
                return text, 0.9
    except Exception:
        pass
    return "N/A", 0.0


def draw_plate_label(frame, text, x1, y1, x2, y2):
    if not text:
        return
    fw = x2 - x1
    fs = builtins.max(0.7, builtins.min(1.5, fw / 150))
    th_ = builtins.max(2, int(fs * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, fs, th_)
    tx = builtins.max(0, x1)
    ty = builtins.max(th + bl + 6, y1 - 6)
    cv2.rectangle(frame, (tx - 4, ty - th - bl - 6), (tx + tw + 6, ty + bl + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), font, fs, (0, 255, 0), th_)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


def merge_overlapping_boxes(bl, iou_threshold=0.3):
    if len(bl) <= 1:
        return bl

    def iou(b1, b2):
        x1 = builtins.max(b1[0], b2[0]); y1 = builtins.max(b1[1], b2[1])
        x2 = builtins.min(b1[2], b2[2]); y2 = builtins.min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter)

    bl = sorted(bl, key=lambda b: b[4], reverse=True)
    keep = []
    while bl:
        best = bl.pop(0)
        keep.append(best)
        bl = [b for b in bl if iou(best, b) < iou_threshold]
    return keep


def is_valid_plate_shape(x1, y1, x2, y2, fw, fh, min_width=18, min_height=8,
                         min_area=150, min_aspect=1.0, max_aspect=5.5):
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    if w < min_width or h < min_height:
        return False
    area = w * h
    asp = w / h
    if area < min_area:
        return False
    if not (min_aspect <= asp <= max_aspect):
        return False
    return True


def find_matching_plate_id(x1, y1, x2, y2, pl, stab, fc, max_distance=80, max_age=30):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bid, bd = None, max_distance
    for pid, (px1, py1, px2, py2, _) in pl.items():
        if fc - stab.last_seen.get(pid, 0) > max_age:
            continue
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        d = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        if d < bd:
            bd, bid = d, pid
    return bid


# --------------------------------------------------------------------------------------
# Main processing entry point (importable by the web app)
# --------------------------------------------------------------------------------------
def process_video(video_path=DEFAULT_VIDEO, output_path=None, region="AUTO",
                  start_frame=0, end_frame=None, imgsz=1920, conf=0.15,
                  frame_stride=1, progress_callback=None):
    """
    Detect and read license plates in a video.

    Args:
        video_path:  input video file.
        output_path: annotated output video (mp4v). Defaults to <BASE_DIR>/output.mp4.
        region:      "US" | "UK" | "HK" | "AUTO" (Universal).
        imgsz:       YOLO inference size (lower = faster on CPU).
        frame_stride: process every Nth frame (>1 speeds up CPU runs).
        progress_callback: optional fn(frame_num, total_frames, locked_plates).

    Returns:
        dict with keys: output_path, plates (sorted unique locked plates),
        fps, width, height, frames_processed.
    """
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "output.mp4")

    model = get_detector()
    stabilizer = PlateStabilizer(window=30, min_confidence=0.15, min_length=4,
                                 lock_threshold=3, persist_frames=10)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (wv, hv))

    frame_count = start_frame
    persistent_labels = {}
    detected = {}  # plate_text -> {"region": ..., "first_frame": ..., "last_frame": ...}
    conf_history = {}  # plate_text -> list of OCR confidences that contributed to its lock

    while cap.isOpened():
        if end_frame is not None and frame_count >= end_frame:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_count += 1

        if frame_stride > 1 and (frame_count - start_frame) % frame_stride != 0:
            out.write(frame)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, verbose=False, imgsz=imgsz, conf=conf,
                        iou=0.4, max_det=20)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        boxes = results[0].boxes
        width, height = frame.shape[1], frame.shape[0]
        active_ids = []
        raw_boxes = []
        for i in range(len(boxes)):
            c = boxes.conf[i].item()
            if c < 0.10:
                continue
            x1 = int(boxes.xyxyn[i][0].item() * width)
            y1 = int(boxes.xyxyn[i][1].item() * height)
            x2 = int(boxes.xyxyn[i][2].item() * width)
            y2 = int(boxes.xyxyn[i][3].item() * height)
            if not is_valid_plate_shape(x1, y1, x2, y2, width, height):
                continue
            raw_boxes.append((x1, y1, x2, y2, c))

        for (x1, y1, x2, y2, c) in merge_overlapping_boxes(raw_boxes, 0.3):
            p = 5
            x1p = int(np.clip(x1 - p, 0, width)); y1p = int(np.clip(y1 - p, 0, height))
            x2p = int(np.clip(x2 + p, 0, width)); y2p = int(np.clip(y2 + p, 0, height))
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            text, oc = read_plate(crop)
            pid = find_matching_plate_id(x1, y1, x2, y2, persistent_labels,
                                         stabilizer, frame_count, 80, 30)
            if pid is None:
                pid = f"plate_{frame_count}_{x1}_{y1}"
            active_ids.append(pid)
            if text and text != "N/A" and validate_plate(text, region) and oc >= 0.6:
                stabilizer.update(pid, text, oc, frame_count)
                conf_history.setdefault(validate_plate(text, region), []).append(oc)
            st = stabilizer.get_stable(pid, frame_count)
            if st:
                persistent_labels[pid] = (x1, y1, x2, y2, st)
            else:
                prev = persistent_labels.get(pid)
                persistent_labels[pid] = (x1, y1, x2, y2, prev[4] if prev else None)

        stabilizer.clear_lost(active_ids, frame_count)
        for pid in list(persistent_labels.keys()):
            if pid not in active_ids and frame_count - stabilizer.last_seen.get(pid, 0) > 45:
                del persistent_labels[pid]

        locked_this_frame = []
        for pid, (x1, y1, x2, y2, text) in persistent_labels.items():
            valid = validate_plate(text, region) if text else None
            if valid:
                draw_plate_label(frame, text, x1, y1, x2, y2)
                locked_this_frame.append(valid)
                rec = detected.setdefault(
                    valid, {"region": region, "first_frame": frame_count,
                            "last_frame": frame_count})
                rec["last_frame"] = frame_count

        out.write(frame)

        if progress_callback and frame_count % 30 == 0:
            progress_callback(frame_count, total_frames, sorted(set(locked_this_frame)))

    cap.release()
    out.release()

    plates = [
        {"plate": t, "region": v["region"],
         "first_frame": v["first_frame"], "last_frame": v["last_frame"],
         "first_time_sec": round(v["first_frame"] / fps, 2),
         "confidence": round(sum(conf_history[t]) / len(conf_history[t]), 3) if conf_history.get(t) else 0.9}
        for t, v in sorted(detected.items())
    ]

    return {
        "output_path": output_path,
        "plates": plates,
        "fps": fps,
        "width": wv,
        "height": hv,
        "frames_processed": frame_count - start_frame,
    }


def reencode_h264(input_path, output_path=None):
    """Re-encode to browser-friendly H.264 using ffmpeg if available. Returns output path."""
    if shutil.which("ffmpeg") is None:
        return input_path
    if output_path is None:
        root, _ = os.path.splitext(input_path)
        output_path = root + "_h264.mp4"
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", "23",
         "-preset", "fast", "-pix_fmt", "yuv420p", output_path, "-y"],
        check=True,
    )
    return output_path


if __name__ == "__main__":
    print("Loading models...")
    get_detector()
    get_ocr()
    print("Models ready. Processing sample video...")

    def _cb(fn, total, locked):
        print(f"  frame {fn}/{total} | locked: {locked}")

    result = process_video(region="AUTO", progress_callback=_cb)
    print(f"\nDone. Annotated video: {result['output_path']}")
    print(f"Detected {len(result['plates'])} plate(s):")
    for p in result["plates"]:
        print(f"  {p['plate']}  (region={p['region']}, first@{p['first_time_sec']}s)")

    final = reencode_h264(result["output_path"])
    if final != result["output_path"]:
        print(f"H.264 version: {final}")
