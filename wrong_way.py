import os
import math
import shutil
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = [2, 3, 5, 7]


class WrongWayDetector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        zones=None,
        conf=0.25,
        imgsz=640,
        history_len=12,
        min_displacement=10,
        confirm_frames=4,
        warmup_frames=30,
        opposite_angle=135,
        tracker="bytetrack.yaml",
    ):
        self.model = YOLO(model_path)

        self.conf = conf
        self.imgsz = imgsz
        self.history_len = history_len
        self.min_displacement = min_displacement
        self.confirm_frames = confirm_frames
        self.warmup_frames = warmup_frames
        self.opposite_angle = opposite_angle
        self.tracker = tracker

        self.zones = zones

        self.reset()

    def reset(self):
        self.frame_index = 0

        self.history = defaultdict(
            lambda: deque(
                maxlen=self.history_len
            )
        )

        self.last_seen = {}
        self.wrong_frames = defaultdict(int)

        self.flagged = set()

        self.flow_samples = []
        self.flow_directions = []

        self.flow_ready = False

    @staticmethod
    def unit(vector):
        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        length = np.linalg.norm(vector)

        if length < 1e-6:
            return None

        return vector / length

    @staticmethod
    def angle(a, b):
        a = WrongWayDetector.unit(a)
        b = WrongWayDetector.unit(b)

        if a is None or b is None:
            return 180.0

        value = np.clip(
            np.dot(a, b),
            -1.0,
            1.0
        )

        return math.degrees(
            math.acos(value)
        )

    def movement(self, track_id):

        points = self.history[track_id]

        if len(points) < 4:
            return None

        old = np.asarray(
            points[-4],
            dtype=np.float32
        )

        new = np.asarray(
            points[-1],
            dtype=np.float32
        )

        move = new - old

        if np.linalg.norm(move) < self.min_displacement:
            return None

        return move

    def learn_flow(self):

        if len(self.flow_samples) < 10:
            return

        directions = []

        for sample in self.flow_samples:

            direction = self.unit(sample)

            if direction is None:
                continue

            duplicate = False

            for existing in directions:

                if self.angle(
                    direction,
                    existing
                ) < 45:

                    duplicate = True
                    break

            if not duplicate:
                directions.append(direction)

            if len(directions) >= 3:
                break

        if directions:
            self.flow_directions = directions
            self.flow_ready = True

    def is_wrong_way(self, move):

        if not self.flow_ready:
            return False

        direction = self.unit(move)

        if direction is None:
            return False

        best_angle = 180

        for normal in self.flow_directions:

            value = self.angle(
                direction,
                normal
            )

            best_angle = min(
                best_angle,
                value
            )

        return best_angle >= self.opposite_angle

    def process_frame(self, frame):

        self.frame_index += 1

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=VEHICLE_CLASSES,
            verbose=False
        )

        result = None

        for item in results:
            result = item
            break

        if result is None:
            return frame, set()

        wrong_now = set()

        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            return frame, wrong_now

        boxes = result.boxes.xyxy.tolist()

        ids = result.boxes.id.tolist()

        if result.boxes.conf is not None:
            confidences = result.boxes.conf.tolist()
        else:
            confidences = [0.0] * len(boxes)

        for box, track_id, confidence in zip(
            boxes,
            ids,
            confidences
        ):

            track_id = int(track_id)

            x1, y1, x2, y2 = map(
                int,
                box
            )

            point = (
                (x1 + x2) / 2,
                y2
            )

            self.history[track_id].append(
                point
            )

            self.last_seen[track_id] = (
                self.frame_index
            )

            move = self.movement(
                track_id
            )

            # Learn normal traffic movement.
            if (
                move is not None
                and self.frame_index
                <= self.warmup_frames
            ):

                self.flow_samples.append(move)

                if len(self.flow_samples) > 300:
                    self.flow_samples = (
                        self.flow_samples[-300:]
                    )

            if (
                not self.flow_ready
                and self.frame_index
                >= self.warmup_frames
            ):
                self.learn_flow()

            wrong = False

            if (
                move is not None
                and self.flow_ready
                and self.frame_index
                > self.warmup_frames
            ):
                wrong = self.is_wrong_way(
                    move
                )

            if wrong:
                self.wrong_frames[track_id] += 1
            else:
                self.wrong_frames[track_id] = max(
                    0,
                    self.wrong_frames[track_id] - 1
                )

            if (
                self.wrong_frames[track_id]
                >= self.confirm_frames
            ):
                self.flagged.add(track_id)

            if track_id in self.flagged:

                color = (
                    0,
                    0,
                    255
                )

                text = (
                    f"WRONG WAY "
                    f"ID:{track_id}"
                )

                wrong_now.add(track_id)

            else:

                color = (
                    0,
                    200,
                    0
                )

                text = (
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                text,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        status = (
            "TRAFFIC FLOW READY"
            if self.flow_ready
            else "LEARNING TRAFFIC FLOW..."
        )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return frame, wrong_now


def process_video(
    in_path,
    out_path,
    detector,
    progress_cb=None
):

    cap = cv2.VideoCapture(in_path)

    if not cap.isOpened():
        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total <= 0:
        total = 1

    os.makedirs(
        os.path.dirname(out_path)
        or ".",
        exist_ok=True
    )

    raw_path = (
        out_path
        + ".raw.mp4"
    )

    if os.path.exists(raw_path):
        os.remove(raw_path)

    if os.path.exists(out_path):
        os.remove(out_path)

    # mp4v FOURCC integer.
    fourcc = 1983148141

    writer = cv2.VideoWriter(
        raw_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        return None

    detector.reset()

    count = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        output, _ = detector.process_frame(
            frame
        )

        writer.write(output)

        count += 1

        if progress_cb:
            progress_cb(
                min(
                    count / total,
                    1.0
                )
            )

    cap.release()
    writer.release()

    if not os.path.exists(raw_path):
        return None

    if os.path.getsize(raw_path) == 0:
        return None

    # Convert to browser-friendly H264.
    if shutil.which("ffmpeg"):

        command = [
            "ffmpeg",
            "-y",
            "-i",
            raw_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            out_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if (
            result.returncode == 0
            and os.path.exists(out_path)
        ):

            os.remove(raw_path)

            return out_path

    os.replace(
        raw_path,
        out_path
    )

    return out_path