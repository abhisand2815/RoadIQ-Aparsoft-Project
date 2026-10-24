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
EVENT_LOG = OUTPUTS_DIR / "road_violations.json"
OUTPUT_VIDEO = OUTPUTS_DIR / "road_violations_annotated.mp4"

# MacBook M4 performance settings
DEFAULT_IMAGE_SIZE = 480
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.50

# Use Apple Silicon GPU when available
DEFAULT_DEVICE = "mps"

# Video preview performance
VIDEO_SKIP_FRAMES = 0
LIVE_PREVIEW_EVERY_N_FRAMES = 3
VIDEO_DISPLAY_WIDTH = 640
VIDEO_DISPLAY_HEIGHT = 360

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
TASK_STOP_LINE = "🛑 Red Light & Stop-Line Detection"

TASK_YOLOE = "YOLOE (text-prompted segmentation)"

TASKS = [
    TASK_TRIPLE_RIDING,
    TASK_NO_HELMET,
    TASK_COMBINED,
    TASK_WRONG_WAY,
    TASK_STOP_LINE,
]


# ============================================================
# YOLO26-ONLY MODEL CATALOG
# The existing RoadIQ services consume DETECTOR_MODELS, POSE_MODELS,
# and HELMET_MODELS. Keep these names stable for compatibility.
# ============================================================

_MODEL_SIZES = ("n", "s", "m", "l", "x")

DETECTOR_MODELS = {
    f"YOLO26 {size.upper()}" + (" — recommended" if size == "n" else ""): f"yolo26{size}.pt"
    for size in _MODEL_SIZES
}
DEFAULT_DETECTOR = "YOLO26 N — recommended"

POSE_MODELS = {
    f"YOLO26 Pose {size.upper()}": f"yolo26{size}-pose.pt"
    for size in _MODEL_SIZES
}
DEFAULT_POSE = "YOLO26 Pose N"

# Project-specific checkpoint; this is not a generic pretrained model.
HELMET_MODELS = {
    "RoadIQ PPE YOLO26-L (custom)": "ppe_yolo26l_96e.pt",
}
DEFAULT_HELMET_MODEL = "RoadIQ PPE YOLO26-L (custom)"
DEFAULT_HELMET_CONFIDENCE = 0.35
HELMET_CLASS_WITH = 0
HELMET_CLASS_WITHOUT = 1

# Optional YOLO26 task catalogs. These do not automatically add inference
# support to existing RoadIQ workflows; use only when the relevant task is
# supported by the installed Ultralytics release.
SEGMENTATION_MODELS = {
    f"YOLO26 Segmentation {size.upper()}": f"yolo26{size}-seg.pt"
    for size in _MODEL_SIZES
}
CLASSIFICATION_MODELS = {
    f"YOLO26 Classification {size.upper()}": f"yolo26{size}-cls.pt"
    for size in _MODEL_SIZES
}
ORIENTED_BOX_MODELS = {
    f"YOLO26 OBB {size.upper()}": f"yolo26{size}-obb.pt"
    for size in _MODEL_SIZES
}

# Defaults consumed by older model-loading helpers.
DETECT_MODEL = "yolo26n.pt"
FALLBACK_DETECT_MODEL = "yolo26n.pt"
POSE_MODEL = "yolo26n-pose.pt"
FALLBACK_POSE_MODEL = "yolo26n-pose.pt"

# ============================================================
# COCO CLASS IDs
# ============================================================
# COCO pretrained detection labels: person=0, motorcycle=3.
# Both naming conventions are kept because existing modules use both.
PERSON_CLASS = 0
PERSON_CLASS_ID = PERSON_CLASS
MOTORCYCLE_CLASS = 3
MOTORCYCLE_CLASS_ID = MOTORCYCLE_CLASS


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
    """Create the project's local weights directory and prefer it in Ultralytics."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS["weights_dir"] = str(WEIGHTS_DIR)
    except (ImportError, KeyError, TypeError):
        # Keep config importable even before optional ML dependencies are installed.
        pass


def resolve_model_path(model_name: str) -> str:
    """Resolve an existing model path, preferring this project's ``weights/`` folder.

    If the requested file is not present locally, return the model identifier so
    Ultralytics can resolve an official checkpoint when the caller loads it.
    """
    candidate = Path(model_name).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate.resolve())
    if candidate.is_file():
        return str(candidate.resolve())

    local_path = WEIGHTS_DIR / candidate.name
    if local_path.is_file():
        return str(local_path.resolve())

    root_path = BASE_DIR / candidate.name
    if root_path.is_file():
        return str(root_path.resolve())

    return model_name


def sweep_stray_weights() -> None:
    """Move stray checkpoint files from the project root into ``weights/`` safely."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for file in BASE_DIR.glob("*.pt"):
        destination = WEIGHTS_DIR / file.name
        if destination.exists():
            continue
        try:
            shutil.move(str(file), str(destination))
        except OSError:
            # A file move should never stop inference.
            continue
