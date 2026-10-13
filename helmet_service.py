"""
Image and Video inference UI for RoadIQ No-Helmet Detection.

Detects riders without helmets strictly on motorcycles with high accuracy,
providing evidence crops, violation statistics, and downloadable results.

Features:
- Side-by-side original vs detected image display
- Differentiates Rider (Driver) vs Person (Sitting Behind)
- Robust JSON report generation with safe float/int serialization
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

import config
from helmet_detector import HelmetDetector, HelmetAnalysis


def _safe_json_default(obj):
    """Ensure all numpy and custom types are cleanly JSON serializable."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_safe_json_default(x) for x in obj]
    return str(obj)


def render_image(
    confidence: float,
    association_threshold: float,
    helmet_confidence: float,
    detector_name: str,
    pose_name: str,
    helmet_model_name: str,
):
    st.header("🪖 No-Helmet Detection — Image Inference")
    st.caption(
        "Detects riders and pillions riding two-wheelers without helmets. "
        "Strict bike-rider association filters out pedestrians so only people on motorcycles are evaluated. "
        "Differentiates the **Rider (Driver)** from **Person (Sitting Behind)**."
    )

    source = st.radio(
        "Select Image Input Source",
        ["Upload Road Image", "Choose Sample Road Image"],
        horizontal=True,
    )

    frame = None
    image_name = "image.jpg"

    if source == "Upload Road Image":
        uploaded = st.file_uploader(
            "Upload an image containing motorcycles and riders",
            type=[x.lstrip(".") for x in config.IMAGE_EXTENSIONS],
        )
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            image_name = uploaded.name
        else:
            st.info("Upload an image containing motorcycles to test no-helmet detection.")
            return
    else:
        sample_images = sorted(
            [p for p in config.BASE_DIR.glob("images/*") if p.suffix.lower() in config.IMAGE_EXTENSIONS]
        )
        if not sample_images:
            st.warning("No sample images found in images/ directory.")
            return

        selected_sample = st.selectbox(
            "Choose a sample image",
            sample_images,
            format_func=lambda p: p.name,
        )
        if selected_sample:
            frame = cv2.imread(str(selected_sample))
            image_name = selected_sample.name

    if frame is None:
        return

    with st.spinner("Analyzing scene: YOLO Detection + Pose Keypoints + Rider Association + Helmet Inference..."):
        detector = HelmetDetector(
            confidence=confidence,
            association_threshold=association_threshold,
            helmet_confidence=helmet_confidence,
            detector_name=detector_name,
            pose_name=pose_name,
            helmet_model_name=helmet_model_name,
        )
        analysis: HelmetAnalysis = detector.analyze(frame, tracking=False)
        annotated = detector.annotate(frame, analysis)

    # 1. Side-by-Side View of Original Real Image and AI Detected Image
    st.markdown("### 📷 Visual Comparison: Original vs. AI Detection")
    view_mode = st.radio(
        "View Mode",
        ["Side-by-Side Comparison", "Detected Image (Full Width)", "Original Image (Full Width)"],
        horizontal=True,
    )

    orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annot_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    if view_mode == "Side-by-Side Comparison":
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("##### 📸 Original Real Image")
            st.image(orig_rgb, use_container_width=True)
        with col_det:
            st.markdown("##### 🔍 AI Detected Image (With Roles & Helmet Check)")
            st.image(annot_rgb, use_container_width=True)
    elif view_mode == "Detected Image (Full Width)":
        st.image(annot_rgb, caption=f"AI Detection: {image_name}", use_container_width=True)
    else:
        st.image(orig_rgb, caption=f"Original: {image_name}", use_container_width=True)

    # Key Metrics
    total_riders = len(analysis.rider_helmets)
    no_helmet_count = len(analysis.violating_riders)
    helmet_count = len(analysis.compliant_riders)
    pedestrians_count = len(analysis.pedestrians)

    drivers_count = sum(1 for r in analysis.rider_helmets.values() if r.is_driver)
    pillions_count = sum(1 for r in analysis.rider_helmets.values() if not r.is_driver)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Motorcycles", len(analysis.bikes))
    m2.metric("Riders (Drivers)", drivers_count)
    m3.metric("Sitting Behind (Pillions)", pillions_count)
    m4.metric("🚨 No Helmet Violations", no_helmet_count)
    m5.metric("🛡️ Helmets Compliant", helmet_count)
    m6.metric("🚶 Pedestrians (Ignored)", pedestrians_count)

    # Violation Summary & Evidence Gallery
    st.markdown("---")
    st.subheader("📋 Violation Inspection & Evidence")

    if no_helmet_count > 0:
        st.error(
            f"🚨 **VIOLATION DETECTED**: Found **{no_helmet_count} person(s) without helmet** "
            f"across **{len(analysis.violating_bikes)} motorcycle(s)**."
        )

        st.markdown("#### Evidence Gallery (People on Bike Without Helmet)")
        cols = st.columns(min(4, max(1, no_helmet_count)))

        evidence_records = []
        for idx, v_info in enumerate(analysis.violating_riders):
            col = cols[idx % len(cols)]
            r = v_info.person
            rx1, ry1, rx2, ry2 = map(int, r.box)
            h, w = frame.shape[:2]

            crop_x1 = max(0, rx1 - 10)
            crop_y1 = max(0, ry1 - 10)
            crop_x2 = min(w, rx2 + 10)
            crop_y2 = min(h, ry2 + 10)
            rider_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            role_badge = "🏍️ **Rider (Driver)**" if v_info.is_driver else "👥 **Person (Sitting Behind)**"
            rejection_text = f"\n\n⚠️ *{v_info.rejection_note}*" if v_info.rejection_note else ""

            with col:
                if rider_crop.size > 0:
                    col.image(
                        cv2.cvtColor(rider_crop, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                col.markdown(
                    f"{role_badge} #{r.track_id}\n\n"
                    f"Bike: `#{v_info.bike_id}`\n\n"
                    f"❌ **No Helmet**: `{float(v_info.confidence):.1%}`\n\n"
                    f"📍 Assoc: `{float(v_info.association_score):.1%}`"
                    f"{rejection_text}"
                )

            evidence_records.append(
                {
                    "track_id": int(r.track_id),
                    "bike_id": int(v_info.bike_id),
                    "role": v_info.role,
                    "is_driver": bool(v_info.is_driver),
                    "violation": "no_helmet",
                    "confidence": round(float(v_info.confidence), 4),
                    "association_score": round(float(v_info.association_score), 4),
                    "rider_box": [round(float(c), 2) for c in r.box],
                    "head_box": [round(float(c), 2) for c in v_info.head_box],
                    "rejection_note": v_info.rejection_note,
                }
            )

        # Download Evidence Report
        report = {
            "source_image": str(image_name),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_motorcycles": int(len(analysis.bikes)),
            "total_riders_evaluated": int(total_riders),
            "drivers_count": int(drivers_count),
            "pillions_count": int(pillions_count),
            "total_pedestrians_ignored": int(pedestrians_count),
            "no_helmet_violations_count": int(no_helmet_count),
            "violating_bikes": [int(b) for b in analysis.violating_bikes],
            "violations": evidence_records,
        }

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            success, enc_img = cv2.imencode(".jpg", annotated)
            if success:
                st.download_button(
                    "📥 Download Annotated Image",
                    data=enc_img.tobytes(),
                    file_name=f"annotated_no_helmet_{image_name}",
                    mime="image/jpeg",
                    type="primary",
                )
        with d_col2:
            st.download_button(
                "📄 Download Violation JSON Report",
                data=json.dumps(report, indent=2, default=_safe_json_default),
                file_name=f"violation_report_{image_name}.json",
                mime="application/json",
            )

    elif total_riders > 0:
        st.success("✅ **ALL RIDERS & PASSENGERS COMPLIANT**: Everyone associated with a motorcycle is wearing a helmet.")
    else:
        st.info("ℹ️ No motorcycle riders were identified in this frame.")


def render_video(
    confidence: float,
    association_threshold: float,
    helmet_confidence: float,
    confirmation_frames: int,
    detector_name: str,
    pose_name: str,
    helmet_model_name: str,
):
    st.header("🪖 No-Helmet Detection — Video Inference")
    st.caption("Processes video with tracking and temporal multi-frame confirmation to detect helmet violations.")

    source = st.radio("Video source", ["Upload Video", "Stored Video"], horizontal=True)

    if source == "Upload Video":
        uploaded = st.file_uploader(
            "Upload road video",
            type=[x.lstrip(".") for x in config.VIDEO_EXTENSIONS],
        )
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

    if not st.button("Run No-Helmet Video Detection", type="primary"):
        return

    detector = HelmetDetector(
        confidence=confidence,
        association_threshold=association_threshold,
        helmet_confidence=helmet_confidence,
        detector_name=detector_name,
        pose_name=pose_name,
        helmet_model_name=helmet_model_name,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open the selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_path = config.OUTPUTS_DIR / f"{input_path.stem}_no_helmet.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_slot = st.empty()
    progress = st.progress(0.0)
    frame_idx = 0
    total_violations_tracked = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if config.VIDEO_SKIP_FRAMES and frame_idx % (config.VIDEO_SKIP_FRAMES + 1) != 0:
                frame_idx += 1
                continue

            analysis = detector.analyze(frame, tracking=True)
            annotated = detector.annotate(frame, analysis)
            writer.write(annotated)

            for vr in analysis.violating_riders:
                total_violations_tracked.add((vr.bike_id, vr.person.track_id))

            frame_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            if total:
                progress.progress(min(1.0, (frame_idx + 1) / total))
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        progress.progress(1.0)

    st.success(f"Processing complete! Processed {frame_idx} frames.")
    st.metric("Total Unique No-Helmet Violations Tracked", len(total_violations_tracked))
    st.video(str(output_path))
    st.download_button(
        "📥 Download Annotated Video",
        output_path.read_bytes(),
        file_name=output_path.name,
        mime="video/mp4",
    )
