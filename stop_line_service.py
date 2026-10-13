from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np

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
            raise ValueError(
                "Input image is empty."
            )


        stop_line = (
            self.stop_line_detector.detect(
                frame
            )
        )



        traffic_signal = (
            self.traffic_signal_detector.detect(frame)
            if hasattr(self, "traffic_signal_detector")
            and self.traffic_signal_detector is not None
            else None
        )



        vehicles = self._detect_vehicles(
            frame
        )

        potential_violations = []
        violation_candidates = []



        if stop_line is not None:

            relevant_vehicles = []
            image_height, image_width = frame.shape[:2]
            line_length = float(
                np.hypot(
                    stop_line.p2[0] - stop_line.p1[0],
                    stop_line.p2[1] - stop_line.p1[1],
                )
            )

            if line_length == 0:
                raise ValueError(
                    "Detected stop line has zero length."
                )

            for vehicle in vehicles:

                x1, y1, x2, y2 = vehicle.box
                x_base, y_base = vehicle.bottom_center

            
                if y2 < int(image_height * 0.22):
                    continue

                line_dx = stop_line.p2[0] - stop_line.p1[0]
                line_dy = stop_line.p2[1] - stop_line.p1[1]

   
                contact_projection = (
                    (x_base - stop_line.p1[0]) * line_dx
                    + (y_base - stop_line.p1[1]) * line_dy
                ) / (line_length * line_length)

                vehicle_width = max(1, x2 - x1)
                vehicle_height = max(1, y2 - y1)

                base_distance = self._signed_line_distance(
                    stop_line,
                    x_base,
                    y_base,
                )
                vehicle.signed_distance = base_distance

           
                distance_tolerance = min(
                    image_height * 0.025,
                    max(
                        8.0,
                        min(image_height, image_width) * 0.01,
                        vehicle_height * 0.10,
                    ),
                )

              
                max_allowed_distance = image_height * 0.75
                if abs(base_distance) > max_allowed_distance:
                    continue

               
                mid_x = (x1 + x2) / 2.0
                if self.approach_side_sign == 1:
                    # Approaching upwards: front is lower y (y1), rear is higher y (y2)
                    if vehicle.class_id in (3, 1):  # motorcycle, bicycle
                        y_front = y1 + 0.35 * vehicle_height
                        y_rear = float(y2)
                        pts_front = [(mid_x, y_front)]
                        pts_rear = [(mid_x, y_rear)]
                    else:  # car, truck, bus
                        eff_h = min(vehicle_height, vehicle_width * 1.0)
                        y_front = y2 - eff_h * 0.70
                        y_rear = float(y2)
                        pts_front = [
                            (x1 + 0.15 * vehicle_width, y_front),
                            (mid_x, y_front),
                            (x2 - 0.15 * vehicle_width, y_front),
                        ]
                        pts_rear = [
                            (x1 + 0.15 * vehicle_width, y_rear),
                            (mid_x, y_rear),
                            (x2 - 0.15 * vehicle_width, y_rear),
                        ]
                    d_front_vals = [self._signed_line_distance(stop_line, px, py) for px, py in pts_front]
                    d_rear_vals = [self._signed_line_distance(stop_line, px, py) for px, py in pts_rear]
                    app_front = min(d_front_vals)
                    app_base = max(d_rear_vals)
                    app_mid = (app_front + app_base) / 2.0
                elif self.approach_side_sign == -1:
                    # Approaching downwards: front is higher y (y2), rear is lower y (y1)
                    if vehicle.class_id in (3, 1):
                        y_front = y2 - 0.35 * vehicle_height
                        y_rear = float(y1)
                        pts_front = [(mid_x, y_front)]
                        pts_rear = [(mid_x, y_rear)]
                    else:
                        eff_h = min(vehicle_height, vehicle_width * 1.0)
                        y_front = y1 + eff_h * 0.70
                        y_rear = float(y1)
                        pts_front = [
                            (x1 + 0.15 * vehicle_width, y_front),
                            (mid_x, y_front),
                            (x2 - 0.15 * vehicle_width, y_front),
                        ]
                        pts_rear = [
                            (x1 + 0.15 * vehicle_width, y_rear),
                            (mid_x, y_rear),
                            (x2 - 0.15 * vehicle_width, y_rear),
                        ]
                    d_front_vals = [-self._signed_line_distance(stop_line, px, py) for px, py in pts_front]
                    d_rear_vals = [-self._signed_line_distance(stop_line, px, py) for px, py in pts_rear]
                    app_front = min(d_front_vals)
                    app_base = max(d_rear_vals)
                    app_mid = (app_front + app_base) / 2.0
                else:
                    app_base = base_distance
                    app_front = base_distance
                    app_mid = base_distance

                
                if app_base < -distance_tolerance:
                    departure_lat_margin = max(0.28, vehicle_width / line_length)
                    if not (-departure_lat_margin <= contact_projection <= 1.0 + departure_lat_margin):
                        continue
                else:
                    approach_lat_margin = max(0.65, vehicle_width / line_length)
                    if not (-approach_lat_margin <= contact_projection <= 1.0 + approach_lat_margin):
                        continue


                is_clipped = (
                    x1 <= 0
                    or x2 >= image_width - 1
                    or y2 >= image_height - 1
                )
                if is_clipped and abs(base_distance) <= image_height * 0.15:
                    vehicle.reference_reliable = False
                else:
                    vehicle.reference_reliable = True

              
                if not vehicle.reference_reliable:
                    vehicle.position = "REFERENCE_UNCERTAIN"
                    relevant_vehicles.append(vehicle)
                    continue

                if self.approach_side_sign is None:
                    
                    if self._segment_intersects_box(stop_line, vehicle.box) and not (0.0 <= contact_projection <= 1.0):
                        vehicle.position = "SEGMENT_UNCERTAIN"
                    else:
                        vehicle.position = self._position_for_distance(
                            base_distance,
                            distance_tolerance,
                            None,
                        )
                else:
                    tau = distance_tolerance
                    if app_front < -tau:
                        # Reliable leading ground contact has crossed to the departure side
                        vehicle.position = "BEYOND_LINE"
                    elif abs(app_front) <= tau or abs(app_base) <= tau or abs(app_mid) <= tau:
                        # Vehicle contact is at/on the stop line zone within tolerance
                        vehicle.position = "AT_LINE"
                    elif app_front > tau and app_base > tau:
                        # Both leading and trailing contacts are on the approach side
                        vehicle.position = "BEFORE_LINE"
                    else:
                        vehicle.position = self._position_for_distance(
                            base_distance,
                            distance_tolerance,
                            self.approach_side_sign,
                        )

                if vehicle.position == "BEYOND_LINE":
                    potential_violations.append(vehicle)
                    if (
                        traffic_signal is not None
                        and traffic_signal.state == "RED"
                        and self.approach_side_sign in (-1, 1)
                        and vehicle.reference_reliable
                    ):
                        vehicle.is_violation_candidate = True
                        violation_candidates.append(vehicle)

                relevant_vehicles.append(vehicle)

            vehicles = relevant_vehicles

        if traffic_signal is None:
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

                if class_id not in VEHICLE_CLASSES:
                    continue

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
                        name=VEHICLE_CLASSES[
                            class_id
                        ],
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

        if confirmed_violation:
            box_color = (0, 0, 255)
            display_id = (
                vehicle.track_id
                if vehicle.track_id is not None
                else index
            )
            label = (
                "RED-LIGHT VIOLATION | "
                f"ID {display_id} {vehicle.name}"
            )
        elif is_candidate_violation:
            box_color = (0, 0, 255)
            display_id = (
                vehicle.track_id
                if vehicle.track_id is not None
                else index
            )
            label = (
                f"{display_id}: {vehicle.name} {vehicle.confidence:.2f}"
                " | RED-LIGHT CANDIDATE"
            )
        elif (
            vehicle.position in (
                "AT_LINE",
                "REFERENCE_UNCERTAIN",
                "SEGMENT_UNCERTAIN",
            )
            or not vehicle.reference_reliable
        ):
            box_color = (0, 165, 255)
            label = (
                f"{index}: {vehicle.name} {vehicle.confidence:.2f}"
                f" | {vehicle.position}"
            )
        else:
            box_color = (255, 0, 0)
            label = (
                f"{index}: {vehicle.name} {vehicle.confidence:.2f}"
                f" | {vehicle.position}"
            )

        if vehicle.signed_distance is not None:
            label += f" d={vehicle.signed_distance:.1f}px"
            if not vehicle.reference_reliable:
                label += "*"

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
            (0, 0, 255),
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
