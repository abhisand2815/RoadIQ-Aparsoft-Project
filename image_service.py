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

    # ---------------------------------------------------------
    # SELECT DETECTION TYPE
    # ---------------------------------------------------------

    analysis_type = st.radio(
        "Select Detection Type",
        [
            "Triple Riding",
            "Stop-Line Crossing",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # IMAGE UPLOAD
    # ---------------------------------------------------------

    uploaded = st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ],
    )

    if not uploaded:
        st.info("Upload a road image to begin analysis.")
        return

    # ---------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------

    image = Image.open(uploaded).convert("RGB")

    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR,
    )

    # =========================================================
    # TRIPLE RIDING
    # =========================================================

    if analysis_type == "Triple Riding":

        st.subheader("Triple-Riding Detection")

        with st.spinner("Running triple-riding analysis..."):

            detector = TripleRidingDetector(
                confidence,
                association_threshold,
                detector_name,
                pose_name,
            )

            triple_analysis = detector.analyze(
                frame,
                tracking=False,
            )

            triple_result = detector.annotate(
                frame,
                triple_analysis,
            )

        # Display result
        st.image(
            cv2.cvtColor(
                triple_result,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

        # Metrics
        cols = st.columns(4)

        cols[0].metric(
            "Motorcycles",
            len(triple_analysis.bikes),
        )

        cols[1].metric(
            "People",
            len(triple_analysis.people),
        )

        cols[2].metric(
            "Riders",
            sum(
                len(v)
                for v in triple_analysis.associations.values()
            ),
        )

        cols[3].metric(
            "Triple Riding",
            len(triple_analysis.triple_bikes),
        )

    # =========================================================
    # STOP-LINE CROSSING
    # =========================================================

    else:

        st.subheader("Stop-Line Crossing Detection")

        approach_direction = st.selectbox(
            "Camera Approach Direction",
            [
                "Foreground / Bottom (Approaching Upwards)",
                "Background / Top (Approaching Downwards)",
                "Uncalibrated (Side-only)",
            ],
            index=0,
            help="Specify which side of the stop line vehicles approach from.",
        )
        approach_side_sign = (
            1
            if "Bottom" in approach_direction
            else (-1 if "Top" in approach_direction else None)
        )

        with st.spinner(
            "Detecting stop line, traffic signal, and vehicles..."
        ):

            stop_line_service = StopLineService(
                confidence=confidence,
                detector_name=detector_name,
                approach_side_sign=approach_side_sign,
            )

            stop_line_analysis = (
                stop_line_service.analyze(frame)
            )

            stop_line_result = draw_stop_line_analysis(
                frame,
                stop_line_analysis,
            )

        # Display result
        st.image(
            cv2.cvtColor(
                stop_line_result,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

        # Stop line status
        if stop_line_analysis.stop_line is None:

            st.error(
                "Stop line could not be detected."
            )

        else:

            cols = st.columns(2)

            sig_state = (
                stop_line_analysis.traffic_signal.state
                if stop_line_analysis.traffic_signal
                else "UNKNOWN"
            )
            cols[0].metric(
                "Traffic Signal",
                sig_state,
            )

            active_beyond_count = len(
                stop_line_analysis.violation_candidates
                if (
                    stop_line_analysis.traffic_signal
                    and stop_line_analysis.traffic_signal.state == "RED"
                )
                else (
                    []
                    if (
                        stop_line_analysis.traffic_signal
                        and stop_line_analysis.traffic_signal.state != "RED"
                    )
                    else stop_line_analysis.potential_violations
                )
            )

            cols[1].metric(
                "Beyond Stop Line",
                active_beyond_count,
            )