import streamlit as 
import config
import image_service
import video_service
import helmet_service
import combined_service

st.set_page_config(page_title="RoadIQ — Traffic Violation Detection", page_icon="🏍️", layout="wide")

with st.sidebar:
    st.title("🏍️ RoadIQ")
    task = st.selectbox("Detection Task", config.TASKS, index=0)
    st.caption("AI Road-Safety & Traffic Violation Detection")
    mode = st.radio("Inference Mode", config.MODES)
    st.markdown("---")

    detector_name = st.selectbox(
        "Object detector",
        list(config.DETECTOR_MODELS.keys()),
        index=list(config.DETECTOR_MODELS).index(config.DEFAULT_DETECTOR),
    )
    pose_name = st.selectbox(
        "Pose model",
        list(config.POSE_MODELS.keys()),
        index=list(config.POSE_MODELS).index(config.DEFAULT_POSE),
    )

    if task in (config.TASK_NO_HELMET, config.TASK_COMBINED):
        helmet_model_name = st.selectbox(
            "Helmet model",
            list(config.HELMET_MODELS.keys()),
            index=list(config.HELMET_MODELS).index(config.DEFAULT_HELMET_MODEL),
        )
        helmet_conf = st.slider(
            "Helmet confidence threshold",
            0.10,
            0.90,
            config.DEFAULT_HELMET_CONFIDENCE,
            0.05,
        )

    confidence = st.slider(
        "Detection confidence", 0.10, 0.90, config.DEFAULT_CONFIDENCE, 0.05
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
            "Triple-Riding: Uses pretrained YOLO detection + pose models with transparent geometric rider association."
        )
    elif task == config.TASK_NO_HELMET:
        st.caption(
            "No-Helmet: Evaluates helmet compliance strictly for riders associated with motorcycles, ignoring pedestrians."
        )
    else:
        st.caption(
            "Combined: Detects both triple riding and no-helmet violations on motorcycles simultaneously."
        )

# Route execution to appropriate service
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
            "Confirmation window", 1, 15, config.DEFAULT_CONFIRMATION_FRAMES
        )
        video_service.render(
            confidence,
            association_threshold,
            confirmation_frames,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
        )
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
            "Confirmation window", 1, 15, config.DEFAULT_CONFIRMATION_FRAMES
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
else:
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
            "Confirmation window", 1, 15, config.DEFAULT_CONFIRMATION_FRAMES
        )
        # For combined video, render helmet video with tracking
        helmet_service.render_video(
            confidence,
            association_threshold,
            helmet_conf,
            confirmation_frames,
            config.DETECTOR_MODELS[detector_name],
            config.POSE_MODELS[pose_name],
            config.HELMET_MODELS[helmet_model_name],
        )

