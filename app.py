import os
import streamlit as st
import config
import image_service
import video_service
import helmet_service
import combined_service
import wrong_way_service
import stop_line_service


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

    elif task == config.TASK_STOP_LINE:

        st.caption(
            "Stop-Line: Detects vehicles crossing the stop line "
            "during a red traffic signal."
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
        "Upload a road video or choose a sample video. RoadIQ tracks vehicles and "
        "checks whether they move against the allowed direction."
    )

    video_source = st.radio(
        "Select Video Input Source",
        ["Upload Road Video", "Choose Sample Road Video"],
        horizontal=True,
    )

    video_path = None

    if video_source == "Upload Road Video":
        uploaded = st.file_uploader(
            "Upload road video",
            type=[
                x.lstrip(".")
                for x in config.VIDEO_EXTENSIONS
            ],
        )

        if uploaded:
            if uploaded.size < 1000:
                st.warning(
                    "⚠️ The uploaded file is smaller than 1 KB and appears to be a Git LFS pointer file or empty placeholder. "
                    "Please upload a valid MP4 video."
                )
            else:
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
    else:
        # Filter stored videos with actual content (> 10 KB)
        sample_videos = sorted(
            [
                p
                for p in config.VIDEOS_DIR.iterdir()
                if p.suffix.lower() in config.VIDEO_EXTENSIONS and p.stat().st_size > 10000
            ],
            key=lambda p: p.name,
        )

        if not sample_videos:
            st.info("No stored sample videos found in videos/.")
        else:
            selected_video = st.selectbox(
                "Choose a sample road video",
                sample_videos,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024*1024):.1f} MB)",
            )
            if selected_video:
                video_path = selected_video
                st.video(str(video_path))

    if video_path and st.button(
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


# ============================================================
# STOP LINE / RED LIGHT VIOLATION
# ============================================================

elif task == config.TASK_STOP_LINE:

    if mode == config.MODE_IMAGE:

        stop_line_service.render_image(
            confidence=confidence,
            detector_name=config.DETECTOR_MODELS[detector_name],
        )

    else:

        stop_line_service.render_video(
            confidence=confidence,
            detector_name=config.DETECTOR_MODELS[detector_name],
        )
