"""Video inference with YOLO tracking and temporal confirmation."""
from collections import Counter
from pathlib import Path
import cv2
import streamlit as st
from triple_riding_service import TripleRidingDetector
import config


def render(confidence, association_threshold, confirmation_frames, detector_name, pose_name):
    st.header("Triple Riding — Video Inference")
    source = st.radio("Video source", ["Upload Video", "Stored Video"], horizontal=True)

    if source == "Upload Video":
        uploaded = st.file_uploader("Upload road video", type=[x.lstrip('.') for x in config.VIDEO_EXTENSIONS])
        if uploaded is None:
            st.info("Upload a video to start inference.")
            return
        input_path = config.VIDEOS_DIR / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())
    else:
        videos = [p for p in config.VIDEOS_DIR.iterdir() if p.suffix.lower() in config.VIDEO_EXTENSIONS]
        if not videos:
            st.info("No stored videos found in videos/.")
            return
        selected = st.selectbox("Select video", videos, format_func=lambda p: p.name)
        input_path = selected

    if not st.button("Run Triple Riding Detection", type="primary"):
        return

    detector = TripleRidingDetector(confidence, association_threshold, detector_name, pose_name)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open the selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_path = config.OUTPUTS_DIR / f"{input_path.stem}_triple_riding.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    history = {}
    hits = Counter()
    confirmed = set()
    frame_slot = st.empty()
    progress = st.progress(0.0)
    frame_idx = 0
    violations = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.VIDEO_SKIP_FRAMES and frame_idx % (config.VIDEO_SKIP_FRAMES + 1) != 0:
                frame_idx += 1
                continue

            analysis = detector.analyze(frame, tracking=True)
            for bike_id in list(history):
                history[bike_id] = max(0, history[bike_id] - 1)
            for bike_id in analysis.triple_bikes:
                history[bike_id] = confirmation_frames
                hits[bike_id] += 1
                if hits[bike_id] >= config.MIN_CONFIRMATION_HITS:
                    confirmed.add(bike_id)
            # Retire stale confirmation state.
            active_ids = {b.track_id for b in analysis.bikes}
            confirmed &= active_ids | set(history)

            annotated = detector.annotate(frame, analysis, confirmed)
            writer.write(annotated)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            if total:
                progress.progress(min(1.0, (frame_idx + 1) / total))
            violations = max(violations, len(confirmed))
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        progress.progress(1.0)

    st.success(f"Finished. Processed {frame_idx} frames.")
    st.metric("Confirmed triple-riding tracks", violations)
    st.video(str(output_path))
    st.download_button("Download annotated video", output_path.read_bytes(), file_name=output_path.name, mime="video/mp4")
