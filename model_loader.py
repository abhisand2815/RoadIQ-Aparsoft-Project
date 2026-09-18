"""
Centralized model loading with Streamlit caching.

Supports:
- Standard YOLO detection
- YOLO pose
- YOLO World
- YOLOE
- RoadIQ Triple Riding detector + pose models

Models are loaded once and reused across Streamlit reruns.
"""

from typing import Optional, Union, List, Tuple

import torch
import streamlit as st
from ultralytics import YOLO, YOLOWorld, YOLOE

import config


# -------------------------------------------------------------------
# LOCAL WEIGHTS
# -------------------------------------------------------------------

# Reuse models from ./weights/ when available.
# Otherwise Ultralytics can download the required pretrained weights.
config.use_local_weights_dir()


# -------------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------------

def get_device() -> str:
    """
    Return the best available inference device.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _ensure_device(model):
    """
    Move model to the best available device.

    CUDA is used when available, otherwise CPU.
    """
    device = get_device()

    try:
        # Move the underlying PyTorch model first.
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)

        # Move the Ultralytics wrapper as well.
        if hasattr(model, "to"):
            model.to(device)

    except Exception:
        # Ultralytics can handle device placement during inference.
        pass

    return model


# -------------------------------------------------------------------
# STANDARD YOLO LOADER
# -------------------------------------------------------------------

@st.cache_resource(max_entries=64)
def _load_yolo_cached(model_name: str) -> YOLO:
    """
    Load a standard pretrained YOLO model.

    Cached so Streamlit does not reload the model every time
    the UI changes.
    """
    path = config.resolve_model_path(model_name)

    model = YOLO(path)

    config.sweep_stray_weights()

    return _ensure_device(model)


# -------------------------------------------------------------------
# ROADIQ DETECTOR LOADER
# -------------------------------------------------------------------

def load_detector(model_name: str) -> YOLO:
    """
    Load a pretrained detection model.

    Used by:
        triple_riding_service.py
        image_service.py
        video_service.py

    Example:
        load_detector("yolo26l.pt")
    """
    return _load_yolo_cached(model_name)


# -------------------------------------------------------------------
# ROADIQ POSE LOADER
# -------------------------------------------------------------------

def load_pose_model(model_name: str) -> YOLO:
    """
    Load a pretrained YOLO pose model.

    Used by the RoadIQ triple-riding pipeline.

    Example:
        load_pose_model("yolo26l-pose.pt")
    """
    return _load_yolo_cached(model_name)


# -------------------------------------------------------------------
# WORLD MODEL
# -------------------------------------------------------------------

@st.cache_resource(max_entries=16)
def load_world_model(model_name: str) -> YOLOWorld:
    """
    Load a YOLO-World model.
    """
    path = config.resolve_model_path(model_name)

    model = YOLOWorld(path)

    config.sweep_stray_weights()

    return _ensure_device(model)


# -------------------------------------------------------------------
# YOLOE MODEL
# -------------------------------------------------------------------

@st.cache_resource(max_entries=16)
def load_yoloe_model(model_name: str) -> YOLOE:
    """
    Load a YOLOE model.
    """
    path = config.resolve_model_path(model_name)

    model = YOLOE(path)

    config.sweep_stray_weights()

    return _ensure_device(model)


# -------------------------------------------------------------------
# SAFE PROMPTED MODEL LOADING
# -------------------------------------------------------------------

@st.cache_resource(
    max_entries=8,
    show_spinner="Encoding text prompt..."
)
def _load_prompted_model(
    task: str,
    model_name: str,
    classes: Tuple[str, ...],
):
    """
    Load YOLO-World / YOLOE and embed the requested classes.

    Moving to CPU before set_classes() avoids CPU/CUDA tensor
    mismatch errors.
    """

    path = config.resolve_model_path(model_name)

    if task == config.TASK_YOLOE:
        model = YOLOE(path)
    else:
        model = YOLOWorld(path)

    config.sweep_stray_weights()

    # ---------------------------------------------------------------
    # IMPORTANT:
    # set_classes() can create CPU tensors.
    # Therefore configure classes on CPU first.
    # ---------------------------------------------------------------

    try:
        model.to("cpu")
    except Exception:
        pass

    model.set_classes(list(classes))

    # Move complete model back to best device.
    _ensure_device(model)

    return model


# -------------------------------------------------------------------
# WORLD CLASS SETTER
# -------------------------------------------------------------------

def _set_world_classes(
    model: YOLOWorld,
    classes: List[str],
) -> None:
    """
    Safely set YOLO-World classes without CPU/CUDA mismatch.
    """

    try:
        model.to("cpu")
    except Exception:
        pass

    model.set_classes(classes)

    _ensure_device(model)


# -------------------------------------------------------------------
# YOLOE CLASS SETTER
# -------------------------------------------------------------------

def _set_yoloe_classes(
    model: YOLOE,
    classes: List[str],
) -> None:
    """
    Safely set YOLOE classes without CPU/CUDA mismatch.
    """

    try:
        model.to("cpu")
    except Exception:
        pass

    model.set_classes(classes)

    _ensure_device(model)


# -------------------------------------------------------------------
# TASK MODEL ROUTER
# -------------------------------------------------------------------

def get_model_for_task(
    task: str,
    world_classes: Optional[List[str]] = None,
    model_name: Optional[str] = None,
):
    """
    Return the appropriate model for the selected task.

    Supports:
        Detection
        Segmentation
        Pose
        YOLO World
        YOLOE
        Triple Riding
    """

    # ---------------------------------------------------------------
    # RoadIQ Triple Riding
    # ---------------------------------------------------------------

    if task == getattr(config, "TASK_TRIPLE_RIDING", "Triple Riding"):

        detector_name = model_name or config.DEFAULT_DETECTOR
        pose_name = config.DEFAULT_POSE

        # Resolve friendly names -> actual .pt files.
        if detector_name in config.DETECTOR_MODELS:
            detector_file = config.DETECTOR_MODELS[detector_name]
        else:
            detector_file = detector_name

        if pose_name in config.POSE_MODELS:
            pose_file = config.POSE_MODELS[pose_name]
        else:
            pose_file = pose_name

        detector = load_detector(detector_file)
        pose = load_pose_model(pose_file)

        return detector, pose

    # ---------------------------------------------------------------
    # Original YOLO Vision Studio tasks
    # ---------------------------------------------------------------

    defaults = {
        getattr(config, "TASK_DETECT", "Detection"):
            getattr(config, "DETECTION_MODEL", None),

        getattr(config, "TASK_SEGMENT", "Segmentation"):
            getattr(config, "SEGMENTATION_MODEL", None),

        getattr(config, "TASK_POSE", "Pose"):
            getattr(config, "POSE_MODEL", None),

        getattr(config, "TASK_WORLD", "YOLO World"):
            getattr(config, "YOLO_WORLD_MODEL", None),

        getattr(config, "TASK_YOLOE", "YOLOE"):
            getattr(config, "YOLOE_MODEL", None),
    }

    name = model_name or defaults.get(task)

    if not name:
        raise ValueError(
            f"No model configured for task: {task}"
        )

    # ---------------------------------------------------------------
    # YOLO World / YOLOE
    # ---------------------------------------------------------------

    if task in (
        getattr(config, "TASK_WORLD", "YOLO World"),
        getattr(config, "TASK_YOLOE", "YOLOE"),
    ):

        if not world_classes:

            if task == getattr(
                config,
                "TASK_YOLOE",
                "YOLOE",
            ):
                return load_yoloe_model(name)

            return load_world_model(name)

        return _load_prompted_model(
            task,
            name,
            tuple(world_classes),
        )

    # ---------------------------------------------------------------
    # Standard YOLO
    # ---------------------------------------------------------------

    return _load_yolo_cached(name)


# -------------------------------------------------------------------
# FRESH DETECTOR
# -------------------------------------------------------------------

def load_fresh_detector(model_name: str) -> YOLO:
    """
    Load a fresh detector instance.

    Useful for independent video tracking sessions.
    """

    path = config.resolve_model_path(model_name)

    model = YOLO(path)

    config.sweep_stray_weights()

    return _ensure_device(model)


# -------------------------------------------------------------------
# FRESH POSE MODEL
# -------------------------------------------------------------------

def load_fresh_pose_model(model_name: str) -> YOLO:
    """
    Load a fresh pose model instance.

    Useful for independent video tracking sessions.
    """

    path = config.resolve_model_path(model_name)

    model = YOLO(path)

    config.sweep_stray_weights()

    return _ensure_device(model)


# -------------------------------------------------------------------
# FRESH WORLD MODEL
# -------------------------------------------------------------------

def load_fresh_world_model(
    model_name: str,
    classes: Optional[List[str]] = None,
) -> YOLOWorld:

    path = config.resolve_model_path(model_name)

    model = YOLOWorld(path)

    config.sweep_stray_weights()

    if classes:
        _set_world_classes(model, classes)

    return model


# -------------------------------------------------------------------
# FRESH YOLOE MODEL
# -------------------------------------------------------------------

def load_fresh_yoloe_model(
    model_name: str,
    classes: Optional[List[str]] = None,
) -> YOLOE:

    path = config.resolve_model_path(model_name)

    model = YOLOE(path)

    config.sweep_stray_weights()

    if classes:
        _set_yoloe_classes(model, classes)

    return model


# -------------------------------------------------------------------
# GENERIC FRESH MODEL
# -------------------------------------------------------------------

def load_fresh_model(
    task: str,
    world_classes: Optional[List[str]] = None,
    model_name: Optional[str] = None,
):
    """
    Generic uncached model loader.

    Preserves compatibility with the original YOLO Vision Studio
    architecture.
    """

    defaults = {
        getattr(config, "TASK_DETECT", "Detection"):
            getattr(config, "DETECTION_MODEL", None),

        getattr(config, "TASK_SEGMENT", "Segmentation"):
            getattr(config, "SEGMENTATION_MODEL", None),

        getattr(config, "TASK_POSE", "Pose"):
            getattr(config, "POSE_MODEL", None),

        getattr(config, "TASK_WORLD", "YOLO World"):
            getattr(config, "YOLO_WORLD_MODEL", None),

        getattr(config, "TASK_YOLOE", "YOLOE"):
            getattr(config, "YOLOE_MODEL", None),
    }

    name = model_name or defaults.get(task)

    if not name:
        raise ValueError(
            f"No model configured for task: {task}"
        )

    # ---------------------------------------------------------------
    # YOLO World
    # ---------------------------------------------------------------

    if task == getattr(
        config,
        "TASK_WORLD",
        "YOLO World",
    ):

        return load_fresh_world_model(
            name,
            world_classes,
        )

    # ---------------------------------------------------------------
    # YOLOE
    # ---------------------------------------------------------------

    if task == getattr(
        config,
        "TASK_YOLOE",
        "YOLOE",
    ):

        return load_fresh_yoloe_model(
            name,
            world_classes,
        )

    # ---------------------------------------------------------------
    # Standard YOLO
    # ---------------------------------------------------------------

    path = config.resolve_model_path(name)

    model = YOLO(path)

    config.sweep_stray_weights()

    return _ensure_device(model)