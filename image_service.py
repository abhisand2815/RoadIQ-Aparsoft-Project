"""Image inference UI for RoadIQ."""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from triple_riding_service import TripleRidingDetector
from stop_line_service import (
    StopLineService,
    draw_stop_line_analysis,
)


def render(
    confidence,
    association_threshold,
    detector_name,
    pose_name,
):
    st.header("RoadIQ — Image Analysis")

    # Violation type is selected in the main RoadIQ workflow selector.
    # This service handles triple-riding image inference only.
    st.caption("Upload an image to compare the original and annotated result side by side.")

    # Image upload
    uploaded = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if not uploaded:
        st.info("Upload a road image to begin analysis.")
        return

    # Load image
    image = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR,
    )

    # =========================================================
    # TRIPLE RIDING
    # =========================================================
    st.subheader("Triple-riding detection")

    with st.spinner("Running triple-riding analysis..."):
        detector = TripleRidingDetector(
            confidence,
            association_threshold,
            detector_name,
            pose_name,
        )
        triple_analysis = detector.analyze(frame, tracking=False)
        triple_result = detector.annotate(frame, triple_analysis)

    original_col, result_col = st.columns(2)
    with original_col:
        st.markdown("#### Original image")
        st.image(image, use_container_width=True)
    with result_col:
        st.markdown("#### Annotated result")
        st.image(
            cv2.cvtColor(triple_result, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )

    cols = st.columns(4)
    cols[0].metric("Motorcycles", len(triple_analysis.bikes))
    cols[1].metric("People", len(triple_analysis.people))
    cols[2].metric(
        "Riders",
        sum(len(v) for v in triple_analysis.associations.values()),
    )
    cols[3].metric("Triple Riding", len(triple_analysis.triple_bikes))

