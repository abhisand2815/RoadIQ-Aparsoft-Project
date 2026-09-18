import streamlit as st
import config
import image_service
import video_service

st.set_page_config(page_title="RoadIQ — Triple Riding", page_icon="🏍️", layout="wide")

with st.sidebar:
    st.title("🏍️ RoadIQ")
    st.caption("Triple-Riding Detection — pretrained models")
    mode = st.radio("Inference Mode", config.MODES)
    st.markdown("---")

    detector_name = st.selectbox("Object detector", list(config.DETECTOR_MODELS.keys()), index=list(config.DETECTOR_MODELS).index(config.DEFAULT_DETECTOR))
    pose_name = st.selectbox("Pose model", list(config.POSE_MODELS.keys()), index=list(config.POSE_MODELS).index(config.DEFAULT_POSE))
    confidence = st.slider("Detection confidence", 0.10, 0.90, config.DEFAULT_CONFIDENCE, 0.05)
    association_threshold = st.slider("Rider association threshold", 0.20, 0.90, config.DEFAULT_ASSOCIATION_THRESHOLD, 0.05)

    st.markdown("---")
    st.caption("No custom-trained model is required. The system uses pretrained YOLO detection + pose models and transparent association logic.")

if mode == config.MODE_IMAGE:
    image_service.render(confidence, association_threshold, config.DETECTOR_MODELS[detector_name], config.POSE_MODELS[pose_name])
else:
    confirmation_frames = st.sidebar.slider("Confirmation window", 1, 15, config.DEFAULT_CONFIRMATION_FRAMES)
    video_service.render(confidence, association_threshold, confirmation_frames, config.DETECTOR_MODELS[detector_name], config.POSE_MODELS[pose_name])
