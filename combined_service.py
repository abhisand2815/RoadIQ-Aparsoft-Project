"""
Combined violation detection UI for RoadIQ: Triple Riding + No Helmet.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
import streamlit as st

import config
from helmet_detector import HelmetDetector


def render_combined_image(
    confidence: float,
    association_threshold: float,
    helmet_confidence: float,
    detector_name: str,
    pose_name: str,
    helmet_model_name: str,
):
    st.header("🚨 Combined Traffic Violations — Image Inference")
    st.caption("Detects both Triple Riding and No-Helmet violations simultaneously on motorcycles.")

    source = st.radio(
        "Select Image Input Source",
        ["Upload Road Image", "Choose Sample Road Image"],
        horizontal=True,
    )

    frame = None
    image_name = "image.jpg"

    if source == "Upload Road Image":
        uploaded = st.file_uploader(
            "Upload road image",
            type=[x.lstrip(".") for x in config.IMAGE_EXTENSIONS],
        )
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            image_name = uploaded.name
        else:
            st.info("Upload an image containing motorcycles and people to test combined violations.")
            return
    else:
        sample_images = sorted(
            [p for p in config.BASE_DIR.glob("images/*") if p.suffix.lower() in config.IMAGE_EXTENSIONS]
        )
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

    with st.spinner("Analyzing scene for Triple Riding and Helmet Violations..."):
        detector = HelmetDetector(
            confidence=confidence,
            association_threshold=association_threshold,
            helmet_confidence=helmet_confidence,
            detector_name=detector_name,
            pose_name=pose_name,
            helmet_model_name=helmet_model_name,
        )
        analysis = detector.analyze(frame, tracking=False)

        # Identify triple riding bikes
        triple_bikes = {
            bike_id
            for bike_id, riders in analysis.associations.items()
            if len(riders) >= 3
        }

        # Custom combined annotation
        image = frame.copy()

        # Draw Motorcycles
        for bike in analysis.bikes:
            bx1, by1, bx2, by2 = map(int, bike.box)
            riders_on_bike = analysis.associations.get(bike.track_id, [])
            rider_count = len(riders_on_bike)

            is_triple = bike.track_id in triple_bikes
            has_no_helmet = bike.track_id in analysis.violating_bikes

            tags = []
            if is_triple:
                tags.append("TRIPLE RIDING")
            if has_no_helmet:
                no_h_count = sum(
                    1
                    for p, _ in riders_on_bike
                    if analysis.rider_helmets.get(p.track_id)
                    and analysis.rider_helmets[p.track_id].status == "WITHOUT_HELMET"
                )
                tags.append(f"NO HELMET ({no_h_count})")

            if tags:
                color = (0, 0, 255)
                label = f"BIKE {bike.track_id} | RIDERS: {rider_count} | 🚨 " + " + ".join(tags)
                thickness = 3
            elif rider_count > 0:
                color = (0, 220, 100)
                label = f"BIKE {bike.track_id} | RIDERS: {rider_count} | ✅ COMPLIANT"
                thickness = 2
            else:
                color = (255, 180, 0)
                label = f"BIKE {bike.track_id} | NO RIDERS"
                thickness = 2

            cv2.rectangle(image, (bx1, by1), (bx2, by2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            banner_y = max(24, by1 - 8)
            cv2.rectangle(
                image,
                (bx1, banner_y - th - 6),
                (bx1 + tw + 8, banner_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (bx1 + 4, banner_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw Pedestrians
        for ped in analysis.pedestrians:
            px1, py1, px2, py2 = map(int, ped.box)
            color = (150, 150, 150)
            cv2.rectangle(image, (px1, py1), (px2, py2), color, 1)
            cv2.putText(
                image,
                f"PERSON {ped.track_id} | NOT ON BIKE",
                (px1, min(image.shape[0] - 6, py2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        # Draw Riders
        for person_id, rinfo in analysis.rider_helmets.items():
            rider = rinfo.person
            rx1, ry1, rx2, ry2 = map(int, rider.box)
            role_title = "RIDER (Driver)" if rinfo.is_driver else "PERSON (Sitting Behind)"

            if rinfo.status == "WITHOUT_HELMET":
                color = (0, 0, 255)
                status_text = f"NO HELMET ({rinfo.confidence:.0%})"
                tag = f"{role_title} #{rider.track_id} | {status_text}"
                box_thickness = 3
            elif rinfo.status == "WITH_HELMET":
                color = (0, 200, 0)
                status_text = f"HELMET OK ({rinfo.confidence:.0%})"
                tag = f"{role_title} #{rider.track_id} | {status_text}"
                box_thickness = 2
            else:
                color = (0, 165, 255)
                status_text = "HEAD OCCLUDED"
                tag = f"{role_title} #{rider.track_id} | {status_text}"
                box_thickness = 1

            cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, box_thickness)
            if rinfo.helmet_box is not None:
                hx1, hy1, hx2, hy2 = map(int, rinfo.helmet_box)
                cv2.rectangle(image, (hx1, hy1), (hx2, hy2), color, 2)
            else:
                hx1, hy1, hx2, hy2 = map(int, rinfo.head_box)
                cv2.rectangle(image, (hx1, hy1), (hx2, hy2), color, 1)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
            tag_y = max(18, ry1 - 6)
            cv2.rectangle(
                image,
                (rx1, tag_y - th - 4),
                (rx1 + tw + 6, tag_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                image,
                tag,
                (rx1 + 3, tag_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    st.markdown("### 📷 Visual Comparison: Original vs. AI Detection")
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("##### 📸 Original Real Image")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_det:
        st.markdown("##### 🔍 Combined AI Detection (Triple Riding + No Helmet)")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)


    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Motorcycles", len(analysis.bikes))
    c2.metric("Riders on Bikes", len(analysis.rider_helmets))
    c3.metric("🚨 Triple Riding Bikes", len(triple_bikes))
    c4.metric("🪖 No Helmet Violations", len(analysis.violating_riders))

    success, enc_img = cv2.imencode(".jpg", image)
    if success:
        st.download_button(
            "📥 Download Annotated Result",
            data=enc_img.tobytes(),
            file_name=f"combined_violations_{image_name}",
            mime="image/jpeg",
            type="primary",
        )
