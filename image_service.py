"""Image inference UI for RoadIQ Triple Riding."""
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from triple_riding_service import TripleRidingDetector


def render(confidence, association_threshold, detector_name, pose_name):
    st.header("Triple Riding — Image Inference")
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not uploaded:
        st.info("Upload an image containing motorcycles and people to test the detector.")
        return

    image = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Running YOLO detection + pose + rider association..."):
        detector = TripleRidingDetector(confidence, association_threshold, detector_name, pose_name)
        analysis = detector.analyze(frame, tracking=False)
        result = detector.annotate(frame, analysis)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
    cols = st.columns(4)
    cols[0].metric("Motorcycles", len(analysis.bikes))
    cols[1].metric("People", len(analysis.people))
    cols[2].metric("Riders", sum(len(v) for v in analysis.associations.values()))
    cols[3].metric("Triple Riding", len(analysis.triple_bikes))
