import os
import streamlit as st

from wrong_way import WrongWayDetector, process_video


def render_image(*args, **kwargs):
    st.info(
        "Wrong-Way Detection requires video because "
        "vehicle movement is tracked across frames."
    )


def render_video(
    confidence,
    model_path,
    video_path,
    zones=None,
    confirm_frames=4,
):

    if not video_path:
        st.error("No video was provided.")
        return None

    if not os.path.exists(video_path):
        st.error(f"Video was not found: {video_path}")
        return None

    try:
        detector = WrongWayDetector(
            model_path=model_path,
            zones=zones,
            conf=float(confidence),
            imgsz=640,
            history_len=12,
            min_displacement=10,
            confirm_frames=int(confirm_frames),
            warmup_frames=30,
            opposite_angle=135,
            tracker="bytetrack.yaml",
        )

    except Exception as exc:
        st.error(f"Could not load YOLO model: {exc}")
        return None

    progress = st.progress(0)
    status = st.empty()

    def update_progress(value):
        progress.progress(float(value))
        status.text(
            f"Processing wrong-way detection: "
            f"{int(value * 100)}%"
        )

    output_dir = os.path.join(
        "outputs",
        "wrong_way_video"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "wrong_way_result.mp4"
    )

    result = process_video(
        in_path=video_path,
        out_path=output_path,
        detector=detector,
        progress_cb=update_progress,
    )

    progress.empty()
    status.empty()

    if result is None:
        st.error("Output video was not created.")
        return None

    if not os.path.exists(result):
        st.error("Output video does not exist.")
        return None

    if os.path.getsize(result) == 0:
        st.error("Output video is empty.")
        return None

    if detector.flagged:
        st.error(
            f"Wrong-way vehicle(s) detected: "
            f"{len(detector.flagged)}"
        )
    else:
        st.success(
            "No confirmed wrong-way vehicles detected."
        )

    return result