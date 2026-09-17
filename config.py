"""Configuration for RoadIQ Triple-Riding Detection using pretrained YOLO models only."""
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_DIR = BASE_DIR / "evidence"
for d in (WEIGHTS_DIR, VIDEOS_DIR, OUTPUTS_DIR, EVIDENCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODE_IMAGE = "📷 Image Inference"
MODE_VIDEO = "🎬 Video Inference"
MODES = [MODE_IMAGE, MODE_VIDEO]

TASK_TRIPLE_RIDING = "🚨 Triple Riding Detection"

# Pretrained Ultralytics models. No custom-trained weights are required.
DETECTOR_MODELS = {
    "YOLO26 Nano": "yolo26n.pt",
    "YOLO26 Small": "yolo26s.pt",
    "YOLO26 Medium": "yolo26m.pt",
    "YOLO11 Nano": "yolo11n.pt",
    "YOLO11 Small": "yolo11s.pt",
}

POSE_MODELS = {
    "YOLO26 Pose Nano": "yolo26n-pose.pt",
    "YOLO26 Pose Small": "yolo26s-pose.pt",
    "YOLO11 Pose Nano": "yolo11n-pose.pt",
    "YOLO11 Pose Small": "yolo11s-pose.pt",
}

DEFAULT_DETECTOR = "YOLO26 Nano"
DEFAULT_POSE = "YOLO26 Pose Nano"

# COCO class IDs.
PERSON_CLASS = 0
MOTORCYCLE_CLASS = 3

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.50
DEFAULT_IMAGE_SIZE = 640

# Association is deliberately transparent and pretrained-model based.
MOTORCYCLE_EXPANSION = 0.25
MAX_ASSOCIATION_DISTANCE = 1.35
MIN_ASSOCIATION_IOU = 0.01
POSE_BONUS_WEIGHT = 0.35
DISTANCE_WEIGHT = 0.45
OVERLAP_WEIGHT = 0.20

DEFAULT_ASSOCIATION_THRESHOLD = 0.52
DEFAULT_CONFIRMATION_FRAMES = 5
MIN_CONFIRMATION_HITS = 3
TRACK_BUFFER = 30
TRACKER = "bytetrack.yaml"

VIDEO_DISPLAY_WIDTH = 960
VIDEO_DISPLAY_HEIGHT = 540
VIDEO_SKIP_FRAMES = 0

SAVE_EVIDENCE = True
EVIDENCE_FRAME_COUNT = 5

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def use_local_weights_dir() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_name: str) -> str:
    local = WEIGHTS_DIR / model_name
    return str(local) if local.exists() else model_name


def sweep_stray_weights() -> None:
    """Move auto-downloaded project-root .pt files into weights/."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for file in BASE_DIR.glob("*.pt"):
        destination = WEIGHTS_DIR / file.name
        if destination.exists():
            continue
        try:
            shutil.move(str(file), str(destination))
        except Exception:
            pass
