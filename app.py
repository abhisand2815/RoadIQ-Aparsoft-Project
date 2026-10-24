"""RoadIQ Streamlit frontend: focused, workflow-first UI.

Uses the existing RoadIQ inference service APIs. UI changes only; detection
algorithms are not implemented or modified here.
"""
from pathlib import Path
import streamlit as st
import config
import image_service
import video_service
import helmet_service
import combined_service
import wrong_way_service
import stop_line_service

st.set_page_config(
    page_title="RoadIQ | Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual design
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
      --rq-bg: #0d1015; --rq-panel: #171b22; --rq-sidebar: #20232c;
      --rq-border: rgba(255,255,255,.09); --rq-green: #19b78a;
      --rq-text: #f4f7f8; --rq-muted: #aab4c0;
    }
    .stApp { background: var(--rq-bg); }
    [data-testid="stSidebar"] { background: var(--rq-sidebar); border-right: 1px solid var(--rq-border); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
    .block-container { max-width: 1460px; padding-top: 1.45rem; padding-bottom: 3rem; }
    .rq-brand { font-size: 2.05rem; font-weight: 850; letter-spacing: -1.5px; color: var(--rq-text); line-height: 1.05; }
    .rq-brand span { color: var(--rq-green); }
    .rq-subbrand { color: var(--rq-muted); font-size: .74rem; letter-spacing: 1.5px; text-transform: uppercase; margin-top: .45rem; }
    .rq-muted { color: var(--rq-muted); font-size: .9rem; }
    [data-testid="stFileUploader"] { background: rgba(255,255,255,.025); border: 1px dashed rgba(25,183,138,.35); border-radius: 13px; padding: .45rem; }
    .stButton > button[kind="primary"] { background: #0d8e6c; border: 1px solid var(--rq-green); border-radius: 10px; font-weight: 700; }
    hr { border-color: var(--rq-border); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Workflow copy and recommendations
# -----------------------------------------------------------------------------
TASK_COPY = {
    config.TASK_TRIPLE_RIDING: {
        "title": "Triple-riding detection",
        "description": "Identify motorcycles carrying more than two riders using vehicle detection, pose estimation and rider association.",
        "recommendation": "Start with YOLO26 Nano for detection and YOLO26 Pose Nano for pose estimation. Move to a larger model if accuracy matters more than speed.",
        "models": ("detector", "pose"),
    },
    config.TASK_NO_HELMET: {
        "title": "Helmet compliance",
        "description": "Review motorcycle riders for helmet compliance using your custom-trained helmet checkpoint.",
        "recommendation": "Use the RoadIQ custom helmet/PPE checkpoint. A generic COCO detector alone does not reliably classify helmet compliance.",
        "models": ("detector", "pose", "helmet"),
    },
    config.TASK_COMBINED: {
        "title": "Combined violation analysis",
        "description": "Run the configured triple-riding and helmet-compliance workflows in one place.",
        "recommendation": "Use the Nano detector and pose model for a faster baseline, plus the custom helmet checkpoint. Confirm each result on representative road footage.",
        "models": ("detector", "pose", "helmet"),
    },
    config.TASK_WRONG_WAY: {
        "title": "Wrong-way driving",
        "description": "Track vehicles across video frames and identify movement against the configured traffic direction.",
        "recommendation": "Use a detection checkpoint with reliable vehicle detection. Wrong-way decisions also depend on tracking and the road-direction configuration, not just the model size.",
        "models": ("detector",),
    },
    config.TASK_STOP_LINE: {
        "title": "Red-light and stop-line detection",
        "description": "Review vehicle movement relative to traffic signals and the configured stop line.",
        "recommendation": "Begin with YOLO26 Nano for speed. Validate signal classes, stop-line placement and camera perspective for your scene.",
        "models": ("detector",),
    },
}

# Config compatibility: prefer the keys used by the current config.py.
def _model_choices(catalog):
    return list(catalog.keys())

# -----------------------------------------------------------------------------
# Sidebar: only the decisions needed for the selected workflow
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="rq-brand">Road<span>IQ</span></div><div class="rq-subbrand">Road Safety Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("#### 1 · Choose a workflow")
    task = st.selectbox("Violation type", config.TASKS, index=0, label_visibility="collapsed")
    mode = st.radio("2 · Input type", config.MODES, horizontal=False)

    task_info = TASK_COPY.get(task, {})
    needed_models = task_info.get("models", ("detector",))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 3 · Model setup")
    detector_names = _model_choices(config.DETECTOR_MODELS)
    detector_default = getattr(config, "DEFAULT_DETECTOR", detector_names[0])
    if detector_default not in detector_names:
        detector_default = detector_names[0]
    detector_name = st.selectbox(
        "Object detector",
        detector_names,
        index=detector_names.index(detector_default),
        help="YOLO26 checkpoint used for general object detection.",
    )
    selected_detector = config.DETECTOR_MODELS[detector_name]

    selected_pose = None
    if "pose" in needed_models:
        pose_names = _model_choices(config.POSE_MODELS)
        pose_default = getattr(config, "DEFAULT_POSE", pose_names[0])
        if pose_default not in pose_names:
            pose_default = pose_names[0]
        pose_name = st.selectbox(
            "Pose model",
            pose_names,
            index=pose_names.index(pose_default),
            help="Used by workflows that estimate rider/person keypoints.",
        )
        selected_pose = config.POSE_MODELS[pose_name]

    helmet_model_name = None
    helmet_conf = float(getattr(config, "DEFAULT_HELMET_CONFIDENCE", 0.35))
    if "helmet" in needed_models:
        helmet_names = list(config.HELMET_MODELS.keys())
        helmet_default = getattr(config, "DEFAULT_HELMET_MODEL", helmet_names[0])
        if helmet_default not in helmet_names:
            helmet_default = helmet_names[0]
        helmet_model_name = st.selectbox(
            "Custom helmet model",
            helmet_names,
            index=helmet_names.index(helmet_default),
            help="A custom-trained checkpoint is required for helmet classification.",
        )
        helmet_conf = st.slider("Helmet confidence", 0.10, 0.90, helmet_conf, 0.05)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 4 · Detection settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, float(config.DEFAULT_CONFIDENCE), 0.05)
    association_threshold = float(getattr(config, "DEFAULT_ASSOCIATION_THRESHOLD", 0.52))
    if task in (config.TASK_TRIPLE_RIDING, config.TASK_NO_HELMET, config.TASK_COMBINED):
        association_threshold = st.slider("Rider association threshold", 0.20, 0.90, association_threshold, 0.05)

    confirmation_frames = int(getattr(config, "DEFAULT_CONFIRMATION_FRAMES", 5))
    if mode == config.MODE_VIDEO and task in (config.TASK_TRIPLE_RIDING, config.TASK_NO_HELMET, config.TASK_COMBINED):
        confirmation_frames = st.slider("Confirmation window (frames)", 1, 15, confirmation_frames)

    st.caption("Model selection changes which checkpoint is requested; it does not train a model.")

# -----------------------------------------------------------------------------
# Inference workspace
# Keep the main area focused on the selected workflow and its actual controls.
# -----------------------------------------------------------------------------
st.markdown("---")

# -----------------------------------------------------------------------------
# Existing service dispatch: do not reimplement inference here
# -----------------------------------------------------------------------------
if task == config.TASK_TRIPLE_RIDING:
    if mode == config.MODE_IMAGE:
        image_service.render(confidence, association_threshold, selected_detector, selected_pose)
    else:
        video_service.render(confidence, association_threshold, confirmation_frames, selected_detector, selected_pose)

elif task == config.TASK_NO_HELMET:
    helmet_filename = config.HELMET_MODELS[helmet_model_name]
    if mode == config.MODE_IMAGE:
        helmet_service.render_image(confidence, association_threshold, helmet_conf, selected_detector, selected_pose, helmet_filename)
    else:
        helmet_service.render_video(confidence, association_threshold, helmet_conf, confirmation_frames, selected_detector, selected_pose, helmet_filename)

elif task == config.TASK_COMBINED:
    helmet_filename = config.HELMET_MODELS[helmet_model_name]
    if mode == config.MODE_IMAGE:
        combined_service.render_combined_image(confidence, association_threshold, helmet_conf, selected_detector, selected_pose, helmet_filename)
    else:
        # Keep the current service routing unchanged until a dedicated combined-video API exists.
        helmet_service.render_video(confidence, association_threshold, helmet_conf, confirmation_frames, selected_detector, selected_pose, helmet_filename)

elif task == config.TASK_WRONG_WAY:
    if mode == config.MODE_IMAGE:
        wrong_way_service.render_image(confidence=confidence, model_path=selected_detector)
    else:
        st.markdown("#### Road video")
        source = st.radio("Video source", ["Upload a video", "Use a sample video"], horizontal=True, label_visibility="collapsed")
        video_path = None
        if source == "Upload a video":
            uploaded = st.file_uploader(
                "Upload road video",
                type=[ext.lstrip(".") for ext in config.VIDEO_EXTENSIONS],
                key="wrong_way_upload",
                help="Choose an MP4, AVI, MOV, MKV, or WEBM file.",
            )
            if uploaded is not None:
                suffix = Path(uploaded.name).suffix.lower()
                if suffix not in config.VIDEO_EXTENSIONS:
                    st.error("Unsupported video format.")
                elif uploaded.size < 1000:
                    st.error("The selected file is too small to be a valid video.")
                else:
                    input_path = config.OUTPUTS_DIR / f"wrong_way_input{suffix}"
                    input_path.write_bytes(uploaded.getbuffer())
                    video_path = input_path
                    st.video(str(video_path))
        else:
            samples = sorted(
                p for p in config.VIDEOS_DIR.iterdir()
                if p.suffix.lower() in config.VIDEO_EXTENSIONS and p.stat().st_size > 10000
            )
            if samples:
                video_path = st.selectbox(
                    "Sample video",
                    samples,
                    format_func=lambda p: f"{p.name} · {p.stat().st_size / (1024 * 1024):.1f} MB",
                )
                st.video(str(video_path))
            else:
                st.info("No sample videos found. Add sample clips to the project's `videos/` folder.")

        if video_path and st.button("Run wrong-way analysis", type="primary", use_container_width=True):
            with st.spinner("Tracking vehicles and analyzing their direction…"):
                result = wrong_way_service.render_video(
                    confidence=confidence,
                    model_path=selected_detector,
                    video_path=str(video_path),
                )
            if result and Path(result).is_file():
                st.success("Wrong-way analysis completed.")
                st.video(str(result))
                with open(result, "rb") as result_file:
                    st.download_button(
                        "Download annotated video",
                        result_file,
                        file_name="wrong_way_result.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

elif task == config.TASK_STOP_LINE:
    if mode == config.MODE_IMAGE:
        stop_line_service.render_image(confidence=confidence, detector_name=selected_detector)
    else:
        stop_line_service.render_video(confidence=confidence, detector_name=selected_detector)
