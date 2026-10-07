import os
import cv2
import streamlit as st
from ultralytics import YOLO
from wrong_way import WrongWayDetector


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def render_image(*args, **kwargs):
    st.warning(
        "Wrong-Way Detection requires a video because vehicle movement direction is required."
    )


def render_video(
    confidence,
    model_path,
    video_path,
    traffic_direction="right",
):

    # Load YOLO model
    model = load_model(model_path)

    # Open input video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Could not open the video.")
        return None

    # Video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        cap.release()
        st.error("Invalid video dimensions.")
        return None

    # Output folder
    output_dir = "outputs/wrong_way_video"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "wrong_way_result.mp4"
    )

    # OpenCV VideoWriter
    fourcc_function = getattr(
        cv2,
        "VideoWriter_fourcc"
    )

    fourcc = fourcc_function(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        st.error(
            "Could not create the output video. "
            "Please check OpenCV video codec support."
        )
        return None

    # Wrong-way detector
    detector = WrongWayDetector(
        traffic_direction=traffic_direction,
        history_size=10,
        min_movement=10.0,
        confirmation_frames=5
    )

    # Streamlit UI
    progress = st.progress(0)
    status = st.empty()
    preview = st.empty()

    wrong_way_ids = set()
    frame_count = 0

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        # YOLO tracking
        results = model.track(
            frame,
            conf=float(confidence),
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        # Compatible with current Ultralytics setup
        result = next(iter(results))

        output = frame.copy()

        # ====================================================
        # VEHICLE DETECTION
        # ====================================================

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id.tolist()
                )

            else:

                track_ids = [
                    None
                ] * len(boxes)

            # =================================================
            # PROCESS EACH VEHICLE
            # =================================================

            for box, class_id, track_id in zip(
                boxes,
                classes,
                track_ids
            ):

                class_id = int(class_id)

                # Ignore non-vehicle classes
                if class_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                vehicle_name = (
                    VEHICLE_CLASSES[class_id]
                )

                wrong_way = False

                # =============================================
                # TRACK MOVEMENT
                # =============================================

                if track_id is not None:

                    track_id = int(track_id)

                    center = (
                        (x1 + x2) / 2,
                        (y1 + y2) / 2
                    )

                    wrong_way = detector.update(
                        track_id,
                        center
                    )

                    if wrong_way:

                        wrong_way_ids.add(
                            track_id
                        )

                # =============================================
                # DRAW RESULT
                # =============================================

                if wrong_way:

                    box_color = (
                        0,
                        0,
                        255
                    )

                    text = (
                        f"WRONG WAY | "
                        f"{vehicle_name}"
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    text = vehicle_name

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.putText(
                    output,
                    text,
                    (
                        x1,
                        max(y1 - 10, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

        # ====================================================
        # WRITE OUTPUT FRAME
        # ====================================================

        writer.write(output)

        frame_count += 1

        # ====================================================
        # PROGRESS
        # ====================================================

        if total_frames > 0:

            progress.progress(
                min(
                    frame_count / total_frames,
                    1.0
                )
            )

        status.text(
            f"Processing frame "
            f"{frame_count}/{total_frames}"
        )

        # Show preview every 10 frames
        if frame_count % 10 == 0:

            preview.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB"
            )

    # ========================================================
    # CLOSE VIDEO
    # ========================================================

    cap.release()
    writer.release()

    progress.empty()
    status.empty()
    preview.empty()

    # ========================================================
    # CHECK OUTPUT
    # ========================================================

    if not os.path.exists(output_path):

        st.error(
            "Output video was not created."
        )

        return None

    # ========================================================
    # RESULT
    # ========================================================

    if wrong_way_ids:

        st.warning(
            f"Wrong-way vehicles detected: "
            f"{len(wrong_way_ids)}"
        )

    else:

        st.info(
            "No wrong-way vehicles detected."
        )

    return output_path