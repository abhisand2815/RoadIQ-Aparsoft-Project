import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

import config
from model_loader import load_detector
from stop_line_detector import (
    AutomaticStopLineDetector,
    StopLine,
)
from traffic_signal_detector import (
    TrafficSignal,
    TrafficSignalDetector,
)




VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}




@dataclass
class VehicleObservation:
    class_id: int
    name: str
    confidence: float
    box: Tuple[int, int, int, int]
    bottom_center: Tuple[int, int]
    position: str
    signed_distance: float | None = None
    reference_reliable: bool = True
    track_id: int | str | None = None
    confirmed_red_light_violation: bool = False
    is_violation_candidate: bool = False

    @property
    def reference_point(self) -> Tuple[int, int]:
        """Return the lower-center ground-contact proxy for this detection."""
        return self.bottom_center




@dataclass
class StopLineAnalysis:
    stop_line: StopLine | None
    vehicles: List[VehicleObservation]
    potential_violations: List[VehicleObservation]
    approach_side_sign: int | None = None
    traffic_signal: TrafficSignal | None = None
    violation_candidates: List[VehicleObservation] = field(default_factory=list)




class StopLineService:

    def __init__(
        self,
        confidence: float | None = None,
        detector_name: str | None = None,
        approach_side_sign: int | None = None,
        approach_direction: str | None = "auto",
        line_offset: int = 0,
    ):
        self.confidence = (
            confidence
            if confidence is not None
            else 0.25
        )
        self.small_object_confidence = 0.20
        self.detector_name = (
            detector_name
            if detector_name is not None
            else config.DETECTOR_MODELS[
                config.DEFAULT_DETECTOR
            ]
        )
        if approach_side_sign not in (None, -1, 1):
            raise ValueError(
                "approach_side_sign must be -1, 1, or None."
            )
        self.approach_side_sign = approach_side_sign
        self.approach_direction = approach_direction or "auto"
        self.line_offset = line_offset

        self.stop_line_detector = (
            AutomaticStopLineDetector()
        )
        self.model = load_detector(
            self.detector_name
        )
        self.traffic_signal_detector = (
            TrafficSignalDetector(detector=self.model)
        )

    def analyze(
        self,
        frame: np.ndarray,
    ) -> StopLineAnalysis:
        if frame is None or frame.size == 0:
            raise ValueError("Input image is empty.")

        hint = self.approach_direction if self.approach_direction in ("top", "bottom", "left", "right") else None
        if hint is None and self.approach_side_sign == 1:
            hint = "bottom"
        elif hint is None and self.approach_side_sign == -1:
            hint = "top"

        stop_line = self.stop_line_detector.detect(frame, approach_hint=hint)

        # Apply fine-tuning offset if specified
        if stop_line is not None and self.line_offset != 0:
            dx = stop_line.p2[0] - stop_line.p1[0]
            dy = stop_line.p2[1] - stop_line.p1[1]
            length = max(1.0, float(np.hypot(dx, dy)))
            nx = -dy / length
            ny = dx / length
            h_img, w_img = frame.shape[:2]
            p1_sh = (
                int(np.clip(stop_line.p1[0] + nx * self.line_offset, 0, w_img - 1)),
                int(np.clip(stop_line.p1[1] + ny * self.line_offset, 0, h_img - 1)),
            )
            p2_sh = (
                int(np.clip(stop_line.p2[0] + nx * self.line_offset, 0, w_img - 1)),
                int(np.clip(stop_line.p2[1] + ny * self.line_offset, 0, h_img - 1)),
            )
            stop_line.p1 = p1_sh
            stop_line.p2 = p2_sh

        traffic_signal = (
            self.traffic_signal_detector.detect(frame)
            if hasattr(self, "traffic_signal_detector")
            and self.traffic_signal_detector is not None
            else None
        )

        vehicles = self._detect_vehicles(frame)
        potential_violations = []
        violation_candidates = []

        if stop_line is not None:
            image_height, image_width = frame.shape[:2]
            line_dx = stop_line.p2[0] - stop_line.p1[0]
            line_dy = stop_line.p2[1] - stop_line.p1[1]
            line_length = float(np.hypot(line_dx, line_dy))

            if line_length == 0:
                raise ValueError("Detected stop line has zero length.")

            # Resolve approach direction
            if self.approach_direction in ("top", "bottom", "left", "right"):
                use_dir = self.approach_direction
            elif self.approach_side_sign == 1:
                use_dir = "bottom"
            elif self.approach_side_sign == -1:
                use_dir = "top"
            else:
                # Auto direction based on road geometry and vehicle clustering
                if abs(line_dy) > abs(line_dx) * 1.5:
                    mean_vx = np.mean([(v.box[0] + v.box[2]) / 2.0 for v in vehicles]) if vehicles else 0
                    line_mid_x = (stop_line.p1[0] + stop_line.p2[0]) / 2.0
                    use_dir = "left" if mean_vx < line_mid_x else "right"
                else:
                    mean_vy = np.mean([v.box[3] for v in vehicles]) if vehicles else 0
                    line_mid_y = (stop_line.p1[1] + stop_line.p2[1]) / 2.0
                    use_dir = "top" if mean_vy < line_mid_y else "bottom"

            tau = max(10.0, min(image_height, image_width) * 0.015)

            for vehicle in vehicles:
                bx1, by1, bx2, by2 = vehicle.box
                cx, cy = vehicle.bottom_center

                if use_dir == "left":
                    front_x = bx2
                    ref_x = stop_line.p1[0] if line_dx == 0 else stop_line.p1[0] + (cy - stop_line.p1[1]) * line_dx / line_dy
                    diff = front_x - ref_x
                elif use_dir == "right":
                    front_x = bx1
                    ref_x = stop_line.p1[0] if line_dx == 0 else stop_line.p1[0] + (cy - stop_line.p1[1]) * line_dx / line_dy
                    diff = ref_x - front_x
                elif use_dir == "bottom":
                    front_y = by1
                    ref_y = stop_line.p1[1] if line_dx == 0 else stop_line.p1[1] + (cx - stop_line.p1[0]) * line_dy / line_dx
                    diff = ref_y - front_y
                else:  # "top"
                    front_y = by2
                    ref_y = stop_line.p1[1] if line_dx == 0 else stop_line.p1[1] + (cx - stop_line.p1[0]) * line_dy / line_dx
                    diff = front_y - ref_y

                vehicle.signed_distance = float(diff)
                vehicle.reference_reliable = True

                if diff > tau:
                    vehicle.position = "BEYOND_LINE"
                    potential_violations.append(vehicle)
                    if traffic_signal is not None and traffic_signal.state == "RED":
                        vehicle.confirmed_red_light_violation = True
                        vehicle.is_violation_candidate = True
                        violation_candidates.append(vehicle)
                elif abs(diff) <= tau:
                    vehicle.position = "AT_LINE"
                else:
                    vehicle.position = "BEFORE_LINE"

        if traffic_signal is None or traffic_signal.state != "RED":
            violation_candidates = list(potential_violations)

        return StopLineAnalysis(
            stop_line=stop_line,
            vehicles=vehicles,
            potential_violations=potential_violations,
            approach_side_sign=self.approach_side_sign,
            traffic_signal=traffic_signal,
            violation_candidates=violation_candidates,
        )

    @staticmethod
    def _position_for_distance(
        signed_distance: float,
        tolerance: float,
        approach_side_sign: int | None,
    ) -> str:
        if approach_side_sign not in (None, -1, 1):
            raise ValueError(
                "approach_side_sign must be -1, 1, or None."
            )
        if abs(signed_distance) <= tolerance:
            return "AT_LINE"
        if approach_side_sign is None:
            return (
                "SIDE_POSITIVE"
                if signed_distance > 0
                else "SIDE_NEGATIVE"
            )
        return (
            "BEFORE_LINE"
            if signed_distance * approach_side_sign > 0
            else "BEYOND_LINE"
        )

    @staticmethod
    def confirms_red_light_crossing(
        *,
        previous_track_id: int | str | None,
        current_track_id: int | str | None,
        previous_signed_distance: float,
        current_signed_distance: float,
        approach_side_sign: int | None,
        previous_timestamp: str | datetime | None,
        current_timestamp: str | datetime | None,
        previous_camera_id: str | None,
        current_camera_id: str | None,
        signal_camera_id: str | None,
        signal_phase: str | None,
        signal_timestamp: str | datetime | None,
        signal_max_age_seconds: float | None,
    ) -> bool:
        """Confirm only a tracked crossing with fresh RED state for that camera."""
        if approach_side_sign not in (-1, 1):
            return False

        if (
            previous_track_id is None
            or previous_track_id != current_track_id
            or not previous_camera_id
            or previous_camera_id != current_camera_id
            or previous_camera_id != signal_camera_id
            or signal_phase is None
            or signal_phase.strip().lower() != "red"
            or signal_max_age_seconds is None
            or previous_timestamp is None
            or current_timestamp is None
            or signal_timestamp is None
        ):
            return False

        if signal_max_age_seconds <= 0:
            raise ValueError(
                "signal_max_age_seconds must be greater than zero."
            )

        previous_time = StopLineService._parse_timestamp(
            previous_timestamp
        )
        current_time = StopLineService._parse_timestamp(
            current_timestamp
        )
        signal_time = StopLineService._parse_timestamp(
            signal_timestamp
        )

        previous_approach_distance = (
            previous_signed_distance * approach_side_sign
        )
        current_approach_distance = (
            current_signed_distance * approach_side_sign
        )

        if (
            previous_approach_distance <= 0
            or current_approach_distance >= 0
            or current_time <= previous_time
        ):
            return False

        distance_sum = (
            previous_approach_distance
            - current_approach_distance
        )
        crossing_fraction = (
            previous_approach_distance / distance_sum
        )
        crossing_time = previous_time + (
            current_time - previous_time
        ) * crossing_fraction

        signal_age = (
            crossing_time - signal_time
        ).total_seconds()
        return (
            0 <= signal_age <= signal_max_age_seconds
        )

    @staticmethod
    def _parse_timestamp(
        value: str | datetime | None,
    ) -> datetime:
        if isinstance(value, str):
            timestamp = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        elif isinstance(value, datetime):
            timestamp = value
        else:
            raise ValueError(
                "A timestamp is required to verify a crossing."
            )

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                "Crossing and signal timestamps must include a timezone."
            )

        return timestamp



    def _detect_vehicles(
        self,
        frame: np.ndarray,
    ) -> List[VehicleObservation]:

        height, width = frame.shape[:2]

        detections = []



        results = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False,
        )

        detections.extend(
            self._extract_vehicle_detections(
                results,
                offset_x=0,
                offset_y=0,
                scale_x=1.0,
                scale_y=1.0,
            )
        )



        crop_y1 = int(
            height * 0.25
        )

        crop_y2 = height

        crop = frame[
            crop_y1:crop_y2,
            0:width,
        ]

        if (
            crop is not None
            and crop.size > 0
        ):

            scale = 1.75

            enlarged = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

            crop_results = self.model.predict(
                enlarged,
                conf=self.small_object_confidence,
                verbose=False,
            )

            detections.extend(
                self._extract_vehicle_detections(
                    crop_results,
                    offset_x=0,
                    offset_y=crop_y1,
                    scale_x=scale,
                    scale_y=scale,
                )
            )



        return self._remove_duplicates(
            detections
        )



    def _extract_vehicle_detections(
        self,
        results,
        offset_x: int,
        offset_y: int,
        scale_x: float,
        scale_y: float,
    ) -> List[VehicleObservation]:

        vehicles = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for index in range(
                len(boxes)
            ):

                class_id = int(
                    boxes.cls[index].item()
                )

                class_name = getattr(self.model, "names", {}).get(class_id, "").lower()
                is_vehicle = class_id in VEHICLE_CLASSES or any(
                    v in class_name for v in ("car", "motorcycle", "bus", "truck", "vehicle")
                )
                if not is_vehicle:
                    continue

                vehicle_name = VEHICLE_CLASSES.get(class_id, class_name or "vehicle")

                confidence = float(
                    boxes.conf[index].item()
                )

                raw_x1, raw_y1, raw_x2, raw_y2 = (
                    boxes.xyxy[index].tolist()
                )


                x1 = int(
                    raw_x1 / scale_x
                    + offset_x
                )

                y1 = int(
                    raw_y1 / scale_y
                    + offset_y
                )

                x2 = int(
                    raw_x2 / scale_x
                    + offset_x
                )

                y2 = int(
                    raw_y2 / scale_y
                    + offset_y
                )

                x1 = max(
                    0,
                    x1,
                )

                y1 = max(
                    0,
                    y1,
                )

                x2 = max(
                    x1,
                    x2,
                )

                y2 = max(
                    y1,
                    y2,
                )

                bottom_center = (
                    int(
                        (x1 + x2) / 2
                    ),
                    y2,
                )

                vehicles.append(
                    VehicleObservation(
                        class_id=class_id,
                        name=vehicle_name,
                        confidence=confidence,
                        box=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        bottom_center=(
                            bottom_center
                        ),
                        position="UNKNOWN",
                    )
                )

        return vehicles



    @staticmethod
    def _line_intersects_box(
        stop_line: StopLine,
        box: Tuple[int, int, int, int],
    ) -> bool:
        """
        Check whether the diagonal stop line intersects
        the vehicle bounding box.
        """

        x1, y1, x2, y2 = box

        lx1, ly1 = stop_line.p1
        lx2, ly2 = stop_line.p2

        def side(
            px: float,
            py: float,
        ) -> float:

            return (
                (lx2 - lx1)
                * (py - ly1)
                - (ly2 - ly1)
                * (px - lx1)
            )

        values = [
            side(x1, y1),
            side(x2, y1),
            side(x2, y2),
            side(x1, y2),
        ]

        has_positive = any(
            value > 0
            for value in values
        )

        has_negative = any(
            value < 0
            for value in values
        )

        if (
            has_positive
            and has_negative
        ):
            return True

   
        line_length = max(
            1.0,
            np.hypot(
                lx2 - lx1,
                ly2 - ly1,
            ),
        )

        distances = [
            abs(value) / line_length
            for value in values
        ]

        return (
            min(distances) <= 15.0
        )

    @staticmethod
    def _segment_intersects_box(
        stop_line: StopLine,
        box: Tuple[int, int, int, int],
    ) -> bool:
        """Return whether the finite stop-line segment intersects a box."""
        x1, y1, x2, y2 = box
        line_x1, line_y1 = stop_line.p1
        line_x2, line_y2 = stop_line.p2
        dx = line_x2 - line_x1
        dy = line_y2 - line_y1

        lower = 0.0
        upper = 1.0
        for p, q in (
            (-dx, line_x1 - x1),
            (dx, x2 - line_x1),
            (-dy, line_y1 - y1),
            (dy, y2 - line_y1),
        ):
            if p == 0:
                if q < 0:
                    return False
                continue
            ratio = q / p
            if p < 0:
                lower = max(lower, ratio)
            else:
                upper = min(upper, ratio)
            if lower > upper:
                return False

        return True


    @staticmethod
    def _remove_duplicates(
        vehicles: List[VehicleObservation],
        iou_threshold: float = 0.50,
    ) -> List[VehicleObservation]:

        ordered = sorted(
            vehicles,
            key=lambda vehicle:
                vehicle.confidence,
            reverse=True,
        )

        kept = []

        for vehicle in ordered:

            duplicate = False

            for existing in kept:

                if (
                    vehicle.class_id
                    != existing.class_id
                ):
                    continue

                # IoU.
                if (
                    StopLineService._box_iou(
                        vehicle.box,
                        existing.box,
                    )
                    >= iou_threshold
                ):

                    duplicate = True
                    break

                # Center distance.
                distance = (
                    StopLineService
                    ._center_distance(
                        vehicle.box,
                        existing.box,
                    )
                )

                width = max(
                    vehicle.box[2]
                    - vehicle.box[0],

                    existing.box[2]
                    - existing.box[0],
                )

                if (
                    width > 0
                    and distance
                    < width * 0.30
                ):

                    duplicate = True
                    break

            if not duplicate:
                kept.append(
                    vehicle
                )

        return kept



    @staticmethod
    def _box_iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        iw = max(
            0,
            ix2 - ix1,
        )

        ih = max(
            0,
            iy2 - iy1,
        )

        intersection = (
            iw * ih
        )

        area_a = (
            max(
                0,
                ax2 - ax1,
            )
            *
            max(
                0,
                ay2 - ay1,
            )
        )

        area_b = (
            max(
                0,
                bx2 - bx1,
            )
            *
            max(
                0,
                by2 - by1,
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection / union
        )



    @staticmethod
    def _center_distance(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:

        ax = (
            box_a[0]
            + box_a[2]
        ) / 2

        ay = (
            box_a[1]
            + box_a[3]
        ) / 2

        bx = (
            box_b[0]
            + box_b[2]
        ) / 2

        by = (
            box_b[1]
            + box_b[3]
        ) / 2

        return float(
            np.hypot(
                ax - bx,
                ay - by,
            )
        )



    @staticmethod
    def _line_y_at_x(
        stop_line: StopLine,
        x: float,
    ) -> float:

        x1, y1 = stop_line.p1
        x2, y2 = stop_line.p2

        if x2 == x1:

            return float(
                (y1 + y2) / 2.0
            )

        ratio = (
            x - x1
        ) / float(
            x2 - x1
        )

        return float(
            y1
            + ratio
            * (
                y2 - y1
            )
        )

    @staticmethod
    def _signed_line_distance(
        stop_line: StopLine,
        x: float,
        y: float,
    ) -> float:

        x1, y1 = stop_line.p1
        x2, y2 = stop_line.p2
        line_length = max(
            1.0,
            float(
                np.hypot(
                    x2 - x1,
                    y2 - y1,
                )
            ),
        )

        return (
            (x2 - x1) * (y - y1)
            - (y2 - y1) * (x - x1)
        ) / line_length




def _layout_vehicle_labels(
    labels: List[Tuple[str, Tuple[int, int, int, int]]],
    image_width: int,
    image_height: int,
    reserved_top: int,
) -> List[Tuple[str, int, int, float]]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    padding = 4
    layouts = []
    placed_rectangles = []
    vehicle_boxes = [
        vehicle_box
        for _, vehicle_box in labels
    ]

    for label, box in labels:
        x1, y1, x2, y2 = box
        max_text_width = max(1, image_width - padding * 2)
        scale = 0.60
        text_size, baseline_offset = cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )
        while text_size[0] > max_text_width and scale > 0.35:
            scale = max(0.35, scale - 0.05)
            text_size, baseline_offset = cv2.getTextSize(
                label,
                font,
                scale,
                thickness,
            )

        if text_size[0] > max_text_width:
            while len(label) > 4 and text_size[0] > max_text_width:
                label = label[:-4] + "..."
                text_size, baseline_offset = cv2.getTextSize(
                    label,
                    font,
                    scale,
                    thickness,
                )

        text_width, text_height = text_size
        min_x = padding
        max_x = max(min_x, image_width - text_width - padding)
        min_baseline = text_height + padding
        max_baseline = max(
            min_baseline,
            image_height - baseline_offset - padding,
        )

        vertical_step = text_height + baseline_offset + 6
        preferred_positions = (
            (x1, y1 - 8),
            (x1, y2 + text_height + baseline_offset + 8),
            (x2 - text_width, y1 - 8),
            (x2 - text_width, y2 + text_height + baseline_offset + 8),
        )
        preferred_positions = tuple(
            (
                min(max(x, min_x), max_x),
                min(max(baseline, min_baseline), max_baseline),
            )
            for x, baseline in preferred_positions
        )
        x_step = max(8, min(24, text_width // 8 or 8))
        x_candidates = list(range(min_x, max_x + 1, x_step))
        if not x_candidates or x_candidates[-1] != max_x:
            x_candidates.append(max_x)
        baseline_candidates = list(
            range(
                max(min_baseline, reserved_top + text_height + padding),
                max_baseline + 1,
                vertical_step,
            )
        )
        if not baseline_candidates:
            baseline_candidates = [min_baseline]
        if baseline_candidates[-1] != max_baseline:
            baseline_candidates.append(max_baseline)

        best_candidate = None
        best_cost = float("inf")

        for candidate_x in x_candidates:
            for candidate_baseline in baseline_candidates:
                rect = (
                    candidate_x - padding,
                    candidate_baseline - text_height - padding,
                    candidate_x + text_width + padding,
                    candidate_baseline + baseline_offset + padding,
                )
                distance = min(
                    abs(candidate_x - preferred_x_x)
                    + abs(candidate_baseline - preferred_baseline)
                    for preferred_x_x, preferred_baseline
                    in preferred_positions
                )
                cost = float(distance)
                if rect[1] < reserved_top:
                    cost += 1_000_000_000

                for placed in placed_rectangles:
                    overlap_width = max(
                        0,
                        min(rect[2], placed[2])
                        - max(rect[0], placed[0]),
                    )
                    overlap_height = max(
                        0,
                        min(rect[3], placed[3])
                        - max(rect[1], placed[1]),
                    )
                    if overlap_width and overlap_height:
                        cost += 1_000_000_000 + (
                            overlap_width * overlap_height * 1000
                        )

                for vehicle_box in vehicle_boxes:
                    overlap_width = max(
                        0,
                        min(rect[2], vehicle_box[2])
                        - max(rect[0], vehicle_box[0]),
                    )
                    overlap_height = max(
                        0,
                        min(rect[3], vehicle_box[3])
                        - max(rect[1], vehicle_box[1]),
                    )
                    if overlap_width and overlap_height:
                        cost += 10_000_000 + (
                            overlap_width * overlap_height * 100
                        )

                if cost < best_cost:
                    best_cost = cost
                    best_candidate = (
                        candidate_x,
                        candidate_baseline,
                        rect,
                    )

        if best_candidate is None:
            raise RuntimeError("Unable to place a vehicle annotation.")

        label_x, label_baseline, label_rect = best_candidate
        layouts.append((label, label_x, label_baseline, scale))
        placed_rectangles.append(label_rect)

    return layouts


def draw_stop_line_analysis(
    frame: np.ndarray,
    analysis: StopLineAnalysis,
) -> np.ndarray:

    result = frame.copy()
    image_height, image_width = result.shape[:2]



    if analysis.stop_line is not None:

        line = analysis.stop_line

        cv2.line(
            result,
            line.p1,
            line.p2,
            (0, 255, 255),
            5,
            cv2.LINE_AA,
        )

        cv2.putText(
            result,
            (
                "STOP LINE | "
                f"score={line.score:.2f}"
            ),
            (
                line.p1[0],
                max(
                    35,
                    line.p1[1] - 12,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if analysis.vehicles and analysis.approach_side_sign is None:
            cv2.putText(
                result,
                "APPROACH DIRECTION UNKNOWN | POSITION ONLY",
                (30, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    else:

        cv2.putText(
            result,
            "STOP LINE: NOT DETECTED",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )



    if analysis.traffic_signal and analysis.traffic_signal.box:
        tx1, ty1, tx2, ty2 = analysis.traffic_signal.box
        sig_state = analysis.traffic_signal.state
        if sig_state == "RED":
            sig_box_color = (0, 0, 255)
        elif sig_state == "GREEN":
            sig_box_color = (0, 255, 0)
        elif sig_state == "YELLOW":
            sig_box_color = (0, 255, 255)
        else:
            sig_box_color = (180, 180, 180)

        cv2.rectangle(
            result,
            (tx1, ty1),
            (tx2, ty2),
            sig_box_color,
            3,
        )
        sig_badge = f"SIGNAL: {sig_state} ({analysis.traffic_signal.confidence:.2f})"
        cv2.putText(
            result,
            sig_badge,
            (tx1, max(24, ty1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            sig_box_color,
            2,
            cv2.LINE_AA,
        )



    label_data = []
    vehicle_colors = []
    for index, vehicle in enumerate(
        analysis.vehicles,
        start=1,
    ):

        confirmed_violation = (
            vehicle.confirmed_red_light_violation
            and analysis.approach_side_sign in (-1, 1)
            and vehicle.reference_reliable
            and vehicle.position == "BEYOND_LINE"
        )
        is_candidate_violation = (
            vehicle.is_violation_candidate
            and analysis.approach_side_sign in (-1, 1)
            and vehicle.reference_reliable
            and vehicle.position == "BEYOND_LINE"
        )

        if vehicle.position == "BEYOND_LINE":
            box_color = (0, 0, 255)  # Red for ahead/crossed
            if vehicle.confirmed_red_light_violation:
                label = f"🚨 RED-LIGHT VIOLATION | {vehicle.name} {vehicle.confidence:.0%}"
            else:
                label = f"🚨 AHEAD OF LINE | {vehicle.name} {vehicle.confidence:.0%}"
        elif vehicle.position == "AT_LINE":
            box_color = (0, 165, 255)  # Amber/Orange for at line
            label = f"⏸️ AT STOP LINE | {vehicle.name} {vehicle.confidence:.0%}"
        else:
            box_color = (0, 200, 0)  # Green for behind/safe
            label = f"✅ BEHIND LINE | {vehicle.name} {vehicle.confidence:.0%}"

        if vehicle.signed_distance is not None:
            label += f" ({vehicle.signed_distance:+.0f}px)"

        label_data.append((label, vehicle.box))
        vehicle_colors.append(box_color)

    reserved_top = 160 if analysis.vehicles else 90
    label_layouts = _layout_vehicle_labels(
        label_data,
        image_width,
        image_height,
        reserved_top,
    )

    for (label, label_x, label_baseline, font_scale), box_color in zip(
        label_layouts,
        vehicle_colors,
    ):
        (text_width, text_height), baseline_offset = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2,
        )
        cv2.rectangle(
            result,
            (label_x - 4, label_baseline - text_height - 4),
            (
                label_x + text_width + 4,
                label_baseline + baseline_offset + 4,
            ),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            result,
            label,
            (label_x, label_baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            box_color,
            2,
            cv2.LINE_AA,
        )

    for vehicle, box_color in zip(
        analysis.vehicles,
        vehicle_colors,
    ):
        x1, y1, x2, y2 = vehicle.box
        cx, cy = vehicle.bottom_center
        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            box_color,
            3,
        )
        cv2.circle(
            result,
            (cx, cy),
            7,
            box_color,
            -1,
        )



    if analysis.traffic_signal:
        sig_state = analysis.traffic_signal.state
        sig_conf = analysis.traffic_signal.confidence
        if sig_state == "RED":
            sig_txt = f"TRAFFIC SIGNAL: RED ({sig_conf:.2f})"
            sig_txt_color = (0, 0, 255)
        elif sig_state == "GREEN":
            sig_txt = f"TRAFFIC SIGNAL: GREEN ({sig_conf:.2f})"
            sig_txt_color = (0, 255, 0)
        elif sig_state == "YELLOW":
            sig_txt = f"TRAFFIC SIGNAL: YELLOW ({sig_conf:.2f})"
            sig_txt_color = (0, 255, 255)
        else:
            sig_txt = f"TRAFFIC SIGNAL: UNKNOWN ({analysis.traffic_signal.reason})"
            sig_txt_color = (200, 200, 200)

        cv2.putText(
            result,
            sig_txt,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            sig_txt_color,
            2,
            cv2.LINE_AA,
        )

    active_beyond_count = len(
        analysis.violation_candidates
        if (analysis.traffic_signal and analysis.traffic_signal.state == "RED")
        else (
            []
            if (analysis.traffic_signal and analysis.traffic_signal.state != "RED")
            else analysis.potential_violations
        )
    )

    cv2.putText(
        result,
        f"Beyond Stop Line candidates: {active_beyond_count}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def render_image(
    confidence: float,
    detector_name: str,
):
    st.header("🛑 Red Light & Stop-Line Violation Detection")
    st.caption("Detects painted stop lines on the road, evaluates traffic signal states, and determines whether vehicles cross or stop safely before the line.")

    source = st.radio(
        "Select Image Input Source",
        ["Upload Road Image", "Choose Sample Road Image"],
        horizontal=True,
    )

    frame = None
    image_name = "image.jpg"

    if source == "Upload Road Image":
        uploaded = st.file_uploader(
            "Upload an image containing road traffic and stop line",
            type=[x.lstrip(".") for x in config.IMAGE_EXTENSIONS],
        )
        if uploaded is not None:
            buf = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            image_name = uploaded.name
    else:
        sample_images = [
            p for p in Path("images").iterdir()
            if p.suffix.lower() in config.IMAGE_EXTENSIONS
        ]
        if not sample_images:
            st.info("No sample images found in images/ directory.")
            return

        selected_sample = st.selectbox(
            "Choose a sample road image",
            sample_images,
            format_func=lambda p: p.name,
        )
        if selected_sample:
            frame = cv2.imread(str(selected_sample))
            image_name = selected_sample.name

    if frame is None:
        return

    with st.expander("⚙️ Stop-Line & Zebra-Crossing Controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            approach_opt = st.selectbox(
                "Vehicle Approach Direction",
                [
                    "Auto-Detect from Scene (Recommended)",
                    "Traffic Moving Downward (Top to Bottom)",
                    "Traffic Moving Upward (Bottom to Top)",
                    "Traffic Moving Rightward (Left to Right)",
                    "Traffic Moving Leftward (Right to Left)",
                ],
            )
            if "Downward" in approach_opt:
                app_dir = "top"
            elif "Upward" in approach_opt:
                app_dir = "bottom"
            elif "Rightward" in approach_opt:
                app_dir = "left"
            elif "Leftward" in approach_opt:
                app_dir = "right"
            else:
                app_dir = "auto"
        with c2:
            signal_override = st.selectbox(
                "Traffic Signal Phase",
                [
                    "Auto-Detect from Scene",
                    "Enforce RED Signal (Active Red-Light Check)",
                    "Enforce GREEN Signal (Pass Allowed)",
                    "Enforce YELLOW Signal (Caution)",
                ],
            )
        with c3:
            line_offset = st.slider(
                "Stop-Line Fine-Tuning (pixels)",
                min_value=-150,
                max_value=150,
                value=0,
                step=5,
                help="Nudge the detected zebra crossing stop line forward or backward for precise alignment on any road camera.",
            )

    with st.spinner("Analyzing stop line, traffic signal, and vehicle positions..."):
        service = StopLineService(
            confidence=confidence,
            detector_name=detector_name,
            approach_direction=app_dir,
            line_offset=line_offset,
        )
        analysis = service.analyze(frame)

        if "RED" in signal_override:
            analysis.traffic_signal = TrafficSignal(state="RED", confidence=1.0, reason="Manual Red Phase Override")
            analysis.violation_candidates = [v for v in analysis.vehicles if v.position == "BEYOND_LINE" and v.reference_reliable]
            for v in analysis.vehicles:
                if v.position == "BEYOND_LINE":
                    v.confirmed_red_light_violation = True
        elif "GREEN" in signal_override:
            analysis.traffic_signal = TrafficSignal(state="GREEN", confidence=1.0, reason="Manual Green Phase Override")
            analysis.violation_candidates = []
            for v in analysis.vehicles:
                v.confirmed_red_light_violation = False
        elif "YELLOW" in signal_override:
            analysis.traffic_signal = TrafficSignal(state="YELLOW", confidence=1.0, reason="Manual Yellow Phase Override")
            analysis.violation_candidates = []
            for v in analysis.vehicles:
                v.confirmed_red_light_violation = False

        annotated = draw_stop_line_analysis(frame, analysis)

    st.markdown("### 📷 Visual Comparison: Original vs. AI Detection")
    view_mode = st.radio(
        "View Mode",
        ["Side-by-Side Comparison", "Detected Image (Full Width)", "Original Image (Full Width)"],
        horizontal=True,
    )

    orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annot_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    if view_mode == "Side-by-Side Comparison":
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("##### 📸 Original Real Image")
            st.image(orig_rgb, use_container_width=True)
        with col_det:
            st.markdown("##### 🔍 AI Detected Image (With Zebra Stop Line & Vehicle Positions)")
            st.image(annot_rgb, use_container_width=True)
    elif view_mode == "Detected Image (Full Width)":
        st.image(annot_rgb, caption=f"Detection: {image_name}", use_container_width=True)
    else:
        st.image(orig_rgb, caption=f"Original: {image_name}", use_container_width=True)

    total_vehicles = len(analysis.vehicles)
    ahead_vehicles = [v for v in analysis.vehicles if v.position == "BEYOND_LINE"]
    behind_vehicles = [v for v in analysis.vehicles if v.position == "BEFORE_LINE"]
    at_line_vehicles = [v for v in analysis.vehicles if v.position == "AT_LINE"]
    sig_state = analysis.traffic_signal.state if analysis.traffic_signal else "UNKNOWN"

    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Traffic Signal", sig_state)
    m2.metric("🚨 Ahead of Stop Line", len(ahead_vehicles))
    m3.metric("✅ Behind Stop Line", len(behind_vehicles))
    m4.metric("⏸️ At Stop Line", len(at_line_vehicles))
    m5.metric("🚗 Total Vehicles", total_vehicles)

    st.markdown("---")
    st.subheader("📊 Comparison & Violation Summary")
    st.info(
        f"**Vehicle Comparison**: **{len(ahead_vehicles)} vehicle(s) ahead of the stop line** vs. "
        f"**{len(behind_vehicles)} vehicle(s) safely behind the stop line** (Total monitored: {total_vehicles})."
    )

    if sig_state == "RED":
        if ahead_vehicles:
            st.error(f"🚨 **RED LIGHT VIOLATIONS DETECTED**: Found **{len(ahead_vehicles)} vehicle(s)** that crossed past the stop line into the crosswalk during a RED signal!")
        else:
            st.success("✅ **ALL VEHICLES COMPLIANT**: All waiting vehicles are safely behind the zebra crossing stop line during RED signal.")
    elif sig_state == "GREEN":
        st.success("🟢 **SIGNAL IS GREEN**: Vehicles are permitted to proceed across the stop line.")
    else:
        st.info(f"Stop-line monitoring active: {len(ahead_vehicles)} vehicle(s) ahead of line, {len(behind_vehicles)} vehicle(s) behind line.")

    if analysis.vehicles:
        st.markdown("#### 📋 Vehicle Position & Violation Breakdown")
        table_rows = []
        for idx, v in enumerate(analysis.vehicles, 1):
            dist_str = f"{v.signed_distance:+.1f} px" if v.signed_distance is not None else "N/A"
            if v.position == "BEYOND_LINE":
                if sig_state == "RED":
                    status_badge = "🚨 RED LIGHT VIOLATION (Crossed)"
                else:
                    status_badge = "⚠️ AHEAD OF STOP LINE (Crossed)"
            elif v.position == "BEFORE_LINE":
                status_badge = "✅ COMPLIANT (Behind Line)"
            elif v.position == "AT_LINE":
                status_badge = "⏸️ AT STOP LINE (Waiting)"
            else:
                status_badge = f"ℹ️ {v.position}"

            table_rows.append({
                "Vehicle #": idx,
                "Type": v.name.title(),
                "Confidence": f"{v.confidence:.0%}",
                "Position": v.position,
                "Distance to Line": dist_str,
                "Status": status_badge,
            })
        st.dataframe(table_rows, use_container_width=True)

    _, encoded_img = cv2.imencode(".jpg", annotated)
    st.download_button(
        "📥 Download Annotated Stop-Line Image",
        encoded_img.tobytes(),
        file_name=f"stop_line_analysis_{image_name}",
        mime="image/jpeg",
    )


def render_video(
    confidence: float,
    detector_name: str,
):
    st.header("🛑 Red Light & Stop-Line Detection — Video Inference")
    st.caption("Tracks moving vehicles across video frames and identifies red-light stop-line crossings in real time.")

    video_source = st.radio(
        "Select Video Input Source",
        ["Upload Road Video", "Choose Sample Road Video"],
        horizontal=True,
    )

    input_path = None

    if video_source == "Upload Road Video":
        uploaded = st.file_uploader(
            "Upload a road video",
            type=[x.lstrip(".") for x in config.VIDEO_EXTENSIONS],
        )
        if uploaded is not None:
            if uploaded.size < 1000:
                st.warning("Uploaded file is too small or an empty placeholder. Please select a valid video.")
                return
            input_path = config.VIDEOS_DIR / uploaded.name
            input_path.write_bytes(uploaded.getbuffer())
    else:
        sample_videos = sorted(
            [p for p in config.VIDEOS_DIR.iterdir() if p.suffix.lower() in config.VIDEO_EXTENSIONS and p.stat().st_size > 10000],
            key=lambda p: p.name,
        )
        if not sample_videos:
            st.info("No sample videos found in videos/.")
            return
        selected = st.selectbox("Choose a sample video", sample_videos, format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024*1024):.1f} MB)")
        if selected:
            input_path = selected

    if not input_path:
        return

    st.video(str(input_path))

    with st.expander("⚙️ Stop-Line Detection Controls", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            approach_opt = st.selectbox(
                "Vehicle Approach Direction",
                [
                    "Approaching Upward (Bottom to Top - Standard)",
                    "Approaching Downward (Top to Bottom)",
                    "Automatic / Unconstrained",
                ],
                key="video_approach_opt",
            )
            approach_sign = 1 if "Bottom to Top" in approach_opt else (-1 if "Top to Bottom" in approach_opt else None)
        with c2:
            signal_override = st.selectbox(
                "Traffic Signal Phase",
                [
                    "Auto-Detect from Scene",
                    "Enforce RED Signal (Active Red-Light Check)",
                    "Enforce GREEN Signal (Pass Allowed)",
                    "Enforce YELLOW Signal (Caution)",
                ],
                key="video_signal_override",
            )

    if not st.button("Run Stop-Line Video Detection", type="primary"):
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open the selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    raw_output_path = config.OUTPUTS_DIR / f"{input_path.stem}_stop_line_raw.mp4"
    output_path = config.OUTPUTS_DIR / f"{input_path.stem}_stop_line_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))

    service = StopLineService(
        confidence=confidence,
        detector_name=detector_name,
        approach_side_sign=approach_sign,
    )

    frame_slot = st.empty()
    progress = st.progress(0.0)
    status_text = st.empty()

    frame_idx = 0
    unique_crossed = set()
    unique_stopped = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            analysis = service.analyze(frame)

            if "RED" in signal_override:
                analysis.traffic_signal = TrafficSignal(state="RED", confidence=1.0, reason="Manual Red Phase Override")
                analysis.violation_candidates = [v for v in analysis.vehicles if v.position == "BEYOND_LINE" and v.reference_reliable]
            elif "GREEN" in signal_override:
                analysis.traffic_signal = TrafficSignal(state="GREEN", confidence=1.0, reason="Manual Green Phase Override")
                analysis.violation_candidates = []
            elif "YELLOW" in signal_override:
                analysis.traffic_signal = TrafficSignal(state="YELLOW", confidence=1.0, reason="Manual Yellow Phase Override")
                analysis.violation_candidates = []

            annotated = draw_stop_line_analysis(frame, analysis)
            writer.write(annotated)

            for v in analysis.vehicles:
                t_id = v.track_id if v.track_id is not None else f"{v.name}_{v.box[0]}"
                if v.position == "BEYOND_LINE":
                    unique_crossed.add(t_id)
                elif v.position in ("BEFORE_LINE", "AT_LINE"):
                    unique_stopped.add(t_id)

            frame_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            progress.progress(min(1.0, (frame_idx + 1) / total))
            stopped_compliant = len(unique_stopped - unique_crossed)
            status_text.text(f"Processed frame {frame_idx + 1}/{total} | Crossed: {len(unique_crossed)} | Stopped: {stopped_compliant}")
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        progress.progress(1.0)

    # Convert to browser-compatible H264 if ffmpeg is available
    display_video_path = raw_output_path
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            str(output_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            display_video_path = output_path

    st.success(f"Processing complete! Analyzed {frame_idx} frames.")
    stopped_compliant_count = len(unique_stopped - unique_crossed)
    total_vehicles_count = len(unique_crossed | unique_stopped)

    c1, c2, c3 = st.columns(3)
    c1.metric("🚨 Crossed Stop Line", len(unique_crossed))
    c2.metric("✅ Stopped Safely Before Line", stopped_compliant_count)
    c3.metric("🚗 Total Unique Vehicles", total_vehicles_count)

    st.video(str(display_video_path))
    with open(display_video_path, "rb") as vf:
        st.download_button(
            "📥 Download Annotated Stop-Line Video",
            vf.read(),
            file_name=f"{input_path.stem}_stop_line_result.mp4",
            mime="video/mp4",
        )
