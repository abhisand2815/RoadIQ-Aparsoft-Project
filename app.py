import os
import streamlit as st
import config
import image_service
import video_service
import helmet_service
import combined_service
import wrong_way_service


st.set_page_config(
    page_title="RoadIQ — Traffic Violation Detection",
    page_icon="🏍️",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏍️ RoadIQ")

    task = st.selectbox(
        "Detection Task",
        config.TASKS,
        index=0
    )

    st.caption(
        "AI Road-Safety & Traffic Violation Detection"
    )

    mode = st.radio(
        "Inference Mode",
        config.MODES
    )

    st.markdown("---")

    detector_name = st.selectbox(
        "Object detector",
        list(config.DETECTOR_MODELS.keys()),
        index=list(config.DETECTOR_MODELS).index(
            config.DEFAULT_DETECTOR
        ),
    )

    pose_name = st.selectbox(
        "Pose model",
        list(config.POSE_MODELS.keys()),
        index=list(config.POSE_MODELS).index(
            config.DEFAULT_POSE
        ),
    )

    helmet_model_name = config.DEFAULT_HELMET_MODEL
    helmet_conf = config.DEFAULT_HELMET_CONFIDENCE

    if task in (
        config.TASK_NO_HELMET,
        config.TASK_COMBINED
    ):

        helmet_model_name = st.selectbox(
            "Helmet model",
            list(config.HELMET_MODELS.keys()),
            index=list(config.HELMET_MODELS).index(
                config.DEFAULT_HELMET_MODEL
            ),
        )

        helmet_conf = st.slider(
            "Helmet confidence threshold",
            0.10,
            0.90,
            config.DEFAULT_HELMET_CONFIDENCE,
            0.05,
        )

    confidence = st.slider(
        "Detection confidence",
        0.10,
        0.90,
        config.DEFAULT_CONFIDENCE,
        0.05
    )

    association_threshold = st.slider(
        "Rider association threshold",
        0.20,
        0.90,
        config.DEFAULT_ASSOCIATION_THRESHOLD,
        0.05,
    )

    st.markdown("---")

    if task == config.TASK_TRIPLE_RIDING:

        st.caption(
            "Triple-Riding: Uses YOLO and pose models "
            "with geometric rider association."
        )

    elif task == config.TASK_NO_HELMET:

        st.caption(
            "No-Helmet: Checks helmet compliance for "
            "riders associated with motorcycles."
        )

    elif task == config.TASK_COMBINED:

        st.caption(
            "Combined: Detects triple riding and "
            "no-helmet violations together."
        )

    elif task == config.TASK_WRONG_WAY:

        st.caption(
            "Wrong-Way: Tracks vehicles with ByteTrack "
            "and checks movement against the allowed direction."
        )


# ============================================================
# TRIPLE RIDING
# ============================================================

if task == config.TASK_TRIPLE_RIDING:

    if mode == config.MODE_IMAGE:

        image_service.render(
            confidence,
            association_threshold,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
        )

    else:

        confirmation_frames = st.sidebar.slider(
            "Confirmation window",
            1,
            15,
            config.DEFAULT_CONFIRMATION_FRAMES
        )

        video_service.render(
            confidence,
            association_threshold,
            confirmation_frames,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
        )


# ============================================================
# NO HELMET
# ============================================================

elif task == config.TASK_NO_HELMET:

    if mode == config.MODE_IMAGE:

        helmet_service.render_image(
            confidence,
            association_threshold,
            helmet_conf,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
            config.HELMET_MODELS[helmet_model_name],
        )

    else:

        confirmation_frames = st.sidebar.slider(
            "Confirmation window",
            1,
            15,
            config.DEFAULT_CONFIRMATION_FRAMES
        )

        helmet_service.render_video(
            confidence,
            association_threshold,
            helmet_conf,
            confirmation_frames,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
            config.HELMET_MODELS[helmet_model_name],
        )


# ============================================================
# COMBINED
# ============================================================

elif task == config.TASK_COMBINED:

    if mode == config.MODE_IMAGE:

        combined_service.render_combined_image(
            confidence,
            association_threshold,
            helmet_conf,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
            config.HELMET_MODELS[helmet_model_name],
        )

    else:

        confirmation_frames = st.sidebar.slider(
            "Confirmation window",
            1,
            15,
            config.DEFAULT_CONFIRMATION_FRAMES
        )

        helmet_service.render_video(
            confidence,
            association_threshold,
            helmet_conf,
            confirmation_frames,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
            config.HELMET_MODELS[helmet_model_name],
        )


# ============================================================
# WRONG WAY
# ============================================================

elif task == config.TASK_WRONG_WAY:

    st.info(
        "Upload a road video. RoadIQ tracks vehicles and "
        "checks whether they move against the allowed direction."
    )

    uploaded = st.file_uploader(
        "Upload road video",
        type=[
            x.lstrip(".")
            for x in config.VIDEO_EXTENSIONS
        ],
    )

    if uploaded:

        os.makedirs(
            str(config.OUTPUTS_DIR),
            exist_ok=True,
        )

        video_path = (
            config.OUTPUTS_DIR
            / "wrong_way_input_video.mp4"
        )

        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.video(str(video_path))

        if st.button(
            "Run Wrong-Way Detection",
            type="primary",
        ):

            model_path = config.DETECTOR_MODELS[
                detector_name
            ]

            result = wrong_way_service.render_video(
                confidence=confidence,
                model_path=model_path,
                video_path=str(video_path),
            )

            if result:

                st.success(
                    "Wrong-way detection completed."
                )

                st.video(result)

                with open(result, "rb") as video_file:

                    st.download_button(
                        "Download Wrong-Way Video",
                        video_file,
                        file_name="wrong_way_result.mp4",
                        mime="video/mp4",
                    )