"""Configuration for RoadIQ traffic violation detection."""

from pathlib import Path
import shutil


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_DIR = BASE_DIR / "weights"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_DIR = BASE_DIR / "evidence"

for directory in (
    WEIGHTS_DIR,
    VIDEOS_DIR,
    OUTPUTS_DIR,
    EVIDENCE_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# INFERENCE MODES
# ============================================================

MODE_IMAGE = "📷 Image Inference"
MODE_VIDEO = "🎬 Video Inference"

MODES = [
    MODE_IMAGE,
    MODE_VIDEO,
]


# ============================================================
# DETECTION TASKS
# ============================================================

TASK_TRIPLE_RIDING = "🚨 Triple Riding Detection"
TASK_NO_HELMET = "🪖 No-Helmet Detection"
TASK_COMBINED = "🚨 All Violations (Triple Riding + No Helmet)"
TASK_WRONG_WAY = "🚫 Wrong-Way Detection"

TASKS = [
    TASK_TRIPLE_RIDING,
    TASK_NO_HELMET,
    TASK_COMBINED,
    TASK_WRONG_WAY,
]


# ============================================================
# OBJECT DETECTION MODELS
# ============================================================

DETECTOR_MODELS = {
    "YOLO11 Nano": "yolo11n.pt",
    "YOLO11 Small": "yolo11s.pt",
    "YOLO26 Nano": "yolo26n.pt",
    "YOLO26 Small": "yolo26s.pt",
    "YOLO26 Medium": "yolo26m.pt",
}

DEFAULT_DETECTOR = "YOLO11 Nano"


# ============================================================
# POSE MODELS
# ============================================================

POSE_MODELS = {
    "YOLO11 Pose Nano": "yolo11n-pose.pt",
    "YOLO11 Pose Small": "yolo11s-pose.pt",
    "YOLO26 Pose Nano": "yolo26n-pose.pt",
    "YOLO26 Pose Small": "yolo26s-pose.pt",
}

DEFAULT_POSE = "YOLO11 Pose Nano"


# ============================================================
# HELMET MODELS
# ============================================================

HELMET_MODELS = {
    "YOLO11 Small Helmet (High Accuracy)": "helmet_yolo11s.pt",
    "YOLOv8 Nano Helmet (Fast)": "helmet_yolov8n.pt",
}

DEFAULT_HELMET_MODEL = (
    "YOLO11 Small Helmet (High Accuracy)"
)

DEFAULT_HELMET_CONFIDENCE = 0.35


HELMET_CLASS_WITH = 0
HELMET_CLASS_WITHOUT = 1


# ============================================================
# COCO CLASS IDs
# ============================================================

PERSON_CLASS = 0
MOTORCYCLE_CLASS = 3


# ============================================================
# GENERAL DETECTION SETTINGS
# ============================================================

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.50
DEFAULT_IMAGE_SIZE = 640


# ============================================================
# RIDER ASSOCIATION SETTINGS
# ============================================================

MOTORCYCLE_EXPANSION = 0.25

MAX_ASSOCIATION_DISTANCE = 1.35

MIN_ASSOCIATION_IOU = 0.01

POSE_BONUS_WEIGHT = 0.35
DISTANCE_WEIGHT = 0.45
OVERLAP_WEIGHT = 0.20

DEFAULT_ASSOCIATION_THRESHOLD = 0.52


# ============================================================
# TEMPORAL / TRACKING SETTINGS
# ============================================================

DEFAULT_CONFIRMATION_FRAMES = 5

MIN_CONFIRMATION_HITS = 3

TRACK_BUFFER = 30

TRACKER = "bytetrack.yaml"


# ============================================================
# VIDEO SETTINGS
# ============================================================

VIDEO_DISPLAY_WIDTH = 960
VIDEO_DISPLAY_HEIGHT = 540

VIDEO_SKIP_FRAMES = 0


# ============================================================
# EVIDENCE SETTINGS
# ============================================================

SAVE_EVIDENCE = True

EVIDENCE_FRAME_COUNT = 5


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
]


# ============================================================
# MODEL HELPERS
# ============================================================

def use_local_weights_dir() -> None:
    """Create the local weights directory if required."""

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def resolve_model_path(model_name: str) -> str:
    """
    Use a local weight file when available.

    Otherwise return the model name so that
    Ultralytics can load/download the official model.
    """

    local_path = WEIGHTS_DIR / model_name

    if local_path.exists():
        return str(local_path)

    return model_name


def sweep_stray_weights() -> None:
    """
    Move .pt files from the project root into weights/.
    """

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in BASE_DIR.glob("*.pt"):

        destination = WEIGHTS_DIR / file.name

        if destination.exists():
            continue

        try:
            shutil.move(
                str(file),
                str(destination),
            )

        except Exception:
            pass