from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

import config
from association import BikeObservation, PersonObservation, associate_people_to_bikes


def _load_model(primary: str, fallback: str):
    local_primary = config.WEIGHTS_DIR / primary
    local_fallback = config.WEIGHTS_DIR / fallback

    try:
        if local_primary.exists():
            return YOLO(str(local_primary))
        return YOLO(primary)
    except Exception:
        if local_fallback.exists():
            return YOLO(str(local_fallback))
        return YOLO(fallback)


class TripleRidingEngine:
    def __init__(
        self,
        confidence=0.40,
        association_iou=0.05,
        confirmation_frames=5,
        min_rider_score=0.45,
    ):
        self.confidence = confidence
        self.association_iou = association_iou
        self.confirmation_frames = confirmation_frames
        self.min_rider_score = min_rider_score

        self.detector = _load_model(
            config.DETECT_MODEL,
            config.FALLBACK_DETECT_MODEL,
        )
        self.pose = _load_model(
            config.POSE_MODEL,
            config.FALLBACK_POSE_MODEL,
        )

    @staticmethod
    def _extract_boxes(result, wanted_class):
        out = []
        if result.boxes is None:
            return out

        names = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)

        for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
            if cls_id != wanted_class:
                continue
            obj_id = int(ids[i]) if ids is not None else i
            out.append((obj_id, tuple(map(float, box)), float(conf)))
        return out

    def _detect(self, frame):
        det_result = self.detector.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker=config.TRACKER,
            verbose=False,
        )[0]

        pose_result = self.pose.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker=config.TRACKER,
            verbose=False,
        )[0]

        bikes_raw = self._extract_boxes(
            det_result, config.MOTORCYCLE_CLASS_ID
        )

        people = []
        if pose_result.boxes is not None:
            boxes = pose_result.boxes.xyxy.cpu().numpy()
            classes = pose_result.boxes.cls.cpu().numpy().astype(int)
            confs = pose_result.boxes.conf.cpu().numpy()
            ids = (
                pose_result.boxes.id.cpu().numpy().astype(int)
                if pose_result.boxes.id is not None
                else np.arange(len(boxes))
            )

            keypoints = None
            if pose_result.keypoints is not None:
                keypoints = pose_result.keypoints.data.cpu().numpy()

            for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
                if cls_id != config.PERSON_CLASS_ID:
                    continue
                kp = keypoints[i] if keypoints is not None else None
                people.append(
                    PersonObservation(
                        person_id=int(ids[i]),
                        box=tuple(map(float, box)),
                        keypoints=kp,
                        confidence=float(conf),
                    )
                )

        bikes = [
            BikeObservation(
                bike_id=bike_id,
                box=box,
                confidence=conf,
            )
            for bike_id, box, conf in bikes_raw
        ]
        return people, bikes

    def _annotate(self, frame, people, bikes, associations, confirmed):
        canvas = frame.copy()

        # Motorcycle boxes.
        for bike in bikes:
            x1, y1, x2, y2 = map(int, bike.box)
            riders = associations.get(bike.bike_id, [])
            rider_count = len(riders)
            is_triple = bike.bike_id in confirmed

            label = f"BIKE ID:{bike.bike_id} | riders:{rider_count}"
            if is_triple:
                label += " | TRIPLE RIDING"

            cv2.rectangle(
                canvas, (x1, y1), (x2, y2),
                (0, 0, 255) if is_triple else (255, 180, 0),
                3 if is_triple else 2,
            )
            cv2.putText(
                canvas, label, (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if is_triple else (255, 180, 0),
                2, cv2.LINE_AA,
            )

        # Person boxes and association scores.
        associated_ids = {
            p.person_id for values in associations.values()
            for p, _ in values
        }
        score_by_person = {
            p.person_id: score
            for values in associations.values()
            for p, score in values
        }

        for p in people:
            x1, y1, x2, y2 = map(int, p.box)
            if p.person_id in associated_ids:
                score = score_by_person[p.person_id]
                label = f"RIDER ID:{p.person_id} | {score:.0%}"
                thickness = 2
                box_color = (0, 255, 0)
            else:
                label = f"PERSON ID:{p.person_id} | STANDING/NEARBY"
                thickness = 1
                box_color = (180, 180, 180)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
            cv2.putText(
                canvas, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                box_color, 1, cv2.LINE_AA,
            )

        return canvas

    def run(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Could not open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(config.OUTPUT_VIDEO),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_placeholder = st.empty()
        c1, c2, c3, c4 = st.columns(4)
        bike_metric = c1.empty()
        triple_metric = c2.empty()
        max_metric = c3.empty()
        fps_metric = c4.empty()

        # bike_id -> consecutive frames with >= 3 associated people
        triple_hits = Counter()
        max_riders = defaultdict(int)
        event_written = set()

        frame_no = 0
        start = time.time()
        last_time = start
        last_fps = 0.0

        with open(config.EVENT_LOG, "w", encoding="utf-8") as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_no += 1
                people, bikes = self._detect(frame)

                associations = associate_people_to_bikes(
                    people,
                    bikes,
                    min_score=self.min_rider_score,
                )

                confirmed = set()
                for bike in bikes:
                    count = len(associations.get(bike.bike_id, []))
                    max_riders[bike.bike_id] = max(
                        max_riders[bike.bike_id], count
                    )

                    if count >= 3:
                        triple_hits[bike.bike_id] += 1
                    else:
                        triple_hits[bike.bike_id] = 0

                    if triple_hits[bike.bike_id] >= self.confirmation_frames:
                        confirmed.add(bike.bike_id)

                        if bike.bike_id not in event_written:
                            event = {
                                "type": "triple_riding",
                                "frame": frame_no,
                                "bike_track_id": bike.bike_id,
                                "riders": count,
                                "frames_confirmed": triple_hits[bike.bike_id],
                                "confidence": round(
                                    float(np.mean([
                                        p.confidence for p, _ in
                                        associations[bike.bike_id]
                                    ])),
                                    4,
                                ) if associations[bike.bike_id] else 0.0,
                            }
                            log.write(json.dumps(event) + "\n")
                            log.flush()
                            event_written.add(bike.bike_id)

                annotated = self._annotate(
                    frame, people, bikes, associations, confirmed
                )
                writer.write(annotated)

                now = time.time()
                inst_fps = 1.0 / max(now - last_time, 1e-6)
                last_fps = inst_fps if last_fps == 0 else (
                    0.9 * last_fps + 0.1 * inst_fps
                )
                last_time = now

                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                bike_metric.metric("Bikes", len(bikes))
                triple_metric.metric(
                    "Confirmed triple bikes", len(confirmed)
                )
                max_metric.metric(
                    "Max riders on a bike",
                    max(max_riders.values(), default=0),
                )
                fps_metric.metric("FPS", f"{last_fps:.1f}")

        cap.release()
        writer.release()

        st.success("Processing complete.")
        st.video(str(config.OUTPUT_VIDEO))
        st.caption(f"Evidence log: {config.EVENT_LOG}")
