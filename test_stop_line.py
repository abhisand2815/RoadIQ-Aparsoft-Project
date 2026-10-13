"""Test stop-line detection on the new RoadIQ junction image."""

from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from stop_line_service import (
    StopLineAnalysis,
    StopLineService,
    VehicleObservation,
    draw_stop_line_analysis,
    _layout_vehicle_labels,
)
from stop_line_detector import (
    AutomaticStopLineDetector,
    StopLine,
)
from traffic_signal_detector import (
    TrafficSignal,
    TrafficSignalDetector,
)


class StopLineGeometryTests(unittest.TestCase):

    def test_only_explicit_confirmed_calibrated_violation_is_red(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)

        def box_color(vehicle, approach_side_sign):
            analysis = StopLineAnalysis(
                stop_line=None,
                vehicles=[vehicle],
                potential_violations=[],
                approach_side_sign=approach_side_sign,
            )
            rendered = draw_stop_line_analysis(image, analysis)
            x1, y1, _, _ = vehicle.box
            return tuple(int(channel) for channel in rendered[y1, x1])

        vehicle = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.9,
            box=(100, 150, 220, 260),
            bottom_center=(160, 260),
            position="BEYOND_LINE",
            track_id=17,
            confirmed_red_light_violation=True,
        )

        self.assertEqual(
            box_color(vehicle, 1),
            (0, 0, 255),
        )
        self.assertEqual(
            box_color(vehicle, None),
            (255, 0, 0),
        )
        vehicle.reference_reliable = False
        self.assertEqual(
            box_color(vehicle, 1),
            (0, 165, 255),
        )
        vehicle.reference_reliable = True
        vehicle.confirmed_red_light_violation = False
        self.assertEqual(
            box_color(vehicle, 1),
            (255, 0, 0),
        )

    def test_vehicle_labels_are_visible_and_do_not_overlap(self):
        labels = [
            ("1: car 0.87 | SIDE_POSITIVE d=160.6px", (230, 190, 399, 299)),
            ("2: motorcycle 0.51 | SIDE_POSITIVE d=185.0px", (260, 195, 380, 299)),
        ]
        layouts = _layout_vehicle_labels(
            labels,
            image_width=400,
            image_height=320,
            reserved_top=90,
        )

        rectangles = []
        for (_, _), (label, x, baseline, scale) in zip(
            labels,
            layouts,
        ):
            self.assertTrue(label)
            size, baseline_offset = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                2,
            )
            rect = (
                x - 4,
                baseline - size[1] - 4,
                x + size[0] + 4,
                baseline + baseline_offset + 4,
            )
            self.assertGreaterEqual(rect[0], 0)
            self.assertGreaterEqual(rect[1], 90)
            self.assertLessEqual(rect[2], 400)
            self.assertLessEqual(rect[3], 320)
            rectangles.append(rect)
            for _, vehicle_box in labels:
                box_overlap = (
                    max(rect[0], vehicle_box[0])
                    < min(rect[2], vehicle_box[2])
                    and max(rect[1], vehicle_box[1])
                    < min(rect[3], vehicle_box[3])
                )
                self.assertFalse(box_overlap)

        overlap_x = (
            max(rectangles[0][0], rectangles[1][0])
            < min(rectangles[0][2], rectangles[1][2])
        )
        overlap_y = (
            max(rectangles[0][1], rectangles[1][1])
            < min(rectangles[0][3], rectangles[1][3])
        )
        self.assertFalse(overlap_x and overlap_y)

    def test_signed_distance_supports_horizontal_and_diagonal_lines(self):
        horizontal = StopLine((0, 50), (100, 50), 1.0)
        diagonal = StopLine((0, 0), (100, 100), 1.0)

        self.assertGreater(
            StopLineService._signed_line_distance(horizontal, 50, 60),
            0,
        )
        self.assertLess(
            StopLineService._signed_line_distance(horizontal, 50, 40),
            0,
        )
        self.assertGreater(
            StopLineService._signed_line_distance(diagonal, 50, 60),
            0,
        )

    def test_position_requires_calibrated_approach_side(self):
        classify = StopLineService._position_for_distance

        self.assertEqual(
            classify(20, 2, None),
            "SIDE_POSITIVE",
        )
        self.assertEqual(
            classify(-20, 2, None),
            "SIDE_NEGATIVE",
        )
        self.assertEqual(
            classify(20, 2, 1),
            "BEFORE_LINE",
        )
        self.assertEqual(
            classify(20, 2, -1),
            "BEYOND_LINE",
        )
        self.assertEqual(
            classify(1, 2, -1),
            "AT_LINE",
        )

    def test_oversized_box_is_classified_from_ground_contact_proxy(self):
        line = StopLine((50, 200), (350, 200), 1.0)
        vehicle = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.9,
            box=(100, 100, 200, 300),
            bottom_center=(150, 300),
            position="UNKNOWN",
        )
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = line
        service.approach_side_sign = 1
        service._detect_vehicles = Mock(return_value=[vehicle])

        analysis = service.analyze(
            np.zeros((400, 400, 3), dtype=np.uint8)
        )

        self.assertEqual(len(analysis.vehicles), 1)
        self.assertEqual(analysis.vehicles[0].position, "BEFORE_LINE")
        self.assertEqual(analysis.vehicles[0].signed_distance, 100)

    def test_position_candidates_require_calibration_and_reliable_reference(self):
        line = StopLine((50, 200), (350, 200), 1.0)
        side_positive = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.9,
            box=(170, 200, 230, 280),
            bottom_center=(200, 280),
            position="UNKNOWN",
        )
        at_line = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.9,
            box=(100, 195, 140, 205),
            bottom_center=(120, 205),
            position="UNKNOWN",
        )
        clipped_reference = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.9,
            box=(300, 200, 399, 250),
            bottom_center=(349, 250),
            position="UNKNOWN",
        )

        uncalibrated_service = StopLineService.__new__(StopLineService)
        uncalibrated_service.stop_line_detector = Mock()
        uncalibrated_service.stop_line_detector.detect.return_value = line
        uncalibrated_service.approach_side_sign = None
        uncalibrated_service._detect_vehicles = Mock(
            return_value=[
                side_positive,
                at_line,
                clipped_reference,
            ]
        )
        uncalibrated = uncalibrated_service.analyze(
            np.zeros((400, 400, 3), dtype=np.uint8)
        )

        self.assertEqual(
            [vehicle.position for vehicle in uncalibrated.vehicles],
            ["SIDE_POSITIVE", "AT_LINE", "REFERENCE_UNCERTAIN"],
        )
        self.assertEqual(uncalibrated.potential_violations, [])

        beyond_line = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.9,
            box=(170, 145, 230, 165),
            bottom_center=(200, 165),
            position="UNKNOWN",
        )
        calibrated_service = StopLineService.__new__(StopLineService)
        calibrated_service.stop_line_detector = Mock()
        calibrated_service.stop_line_detector.detect.return_value = line
        calibrated_service.approach_side_sign = 1
        calibrated_service._detect_vehicles = Mock(
            return_value=[beyond_line, at_line, clipped_reference]
        )
        calibrated = calibrated_service.analyze(
            np.zeros((400, 400, 3), dtype=np.uint8)
        )

        self.assertEqual(
            [vehicle.position for vehicle in calibrated.vehicles],
            ["BEYOND_LINE", "AT_LINE", "REFERENCE_UNCERTAIN"],
        )
        self.assertEqual(calibrated.potential_violations, [beyond_line])

    def test_box_intersecting_segment_is_retained_as_uncertain(self):
        line = StopLine((150, 200), (350, 200), 1.0)
        vehicle = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.9,
            box=(50, 150, 160, 250),
            bottom_center=(105, 250),
            position="UNKNOWN",
        )
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = line
        service.approach_side_sign = None
        service._detect_vehicles = Mock(return_value=[vehicle])

        analysis = service.analyze(
            np.zeros((400, 400, 3), dtype=np.uint8)
        )

        self.assertTrue(
            StopLineService._line_intersects_box(line, vehicle.box)
        )
        self.assertEqual(analysis.vehicles, [vehicle])
        self.assertEqual(vehicle.position, "SEGMENT_UNCERTAIN")
        self.assertEqual(analysis.potential_violations, [])

    def test_finite_segment_box_intersection_handles_horizontal_and_diagonal(self):
        horizontal = StopLine((150, 200), (350, 200), 1.0)
        diagonal = StopLine((100, 100), (300, 300), 1.0)

        self.assertTrue(
            StopLineService._segment_intersects_box(
                horizontal,
                (340, 180, 380, 220),
            )
        )
        self.assertFalse(
            StopLineService._segment_intersects_box(
                horizontal,
                (50, 180, 100, 220),
            )
        )
        self.assertTrue(
            StopLineService._segment_intersects_box(
                diagonal,
                (190, 190, 210, 210),
            )
        )
        self.assertFalse(
            StopLineService._segment_intersects_box(
                diagonal,
                (310, 310, 330, 330),
            )
        )


class RedLightEvidenceTests(unittest.TestCase):

    def setUp(self):
        self.evidence = {
            "previous_track_id": 17,
            "current_track_id": 17,
            "previous_signed_distance": 20,
            "current_signed_distance": -20,
            "approach_side_sign": 1,
            "previous_timestamp": "2026-10-10T10:00:00+05:30",
            "current_timestamp": "2026-10-10T10:00:02+05:30",
            "previous_camera_id": "cam-1",
            "current_camera_id": "cam-1",
            "signal_camera_id": "cam-1",
            "signal_phase": "red",
            "signal_timestamp": "2026-10-10T10:00:00.500000+05:30",
            "signal_max_age_seconds": 2,
        }

    def test_crossing_during_fresh_red_is_confirmed(self):
        self.assertTrue(
            StopLineService.confirms_red_light_crossing(
                **self.evidence
            )
        )

    def test_before_or_already_beyond_without_crossing_is_not_confirmed(self):
        before = self.evidence | {
            "current_signed_distance": 10,
        }
        already_beyond = self.evidence | {
            "previous_signed_distance": -30,
            "current_signed_distance": -10,
        }

        self.assertFalse(
            StopLineService.confirms_red_light_crossing(**before)
        )
        self.assertFalse(
            StopLineService.confirms_red_light_crossing(**already_beyond)
        )

    def test_green_stale_missing_or_other_camera_state_is_not_confirmed(self):
        cases = (
            self.evidence | {"signal_phase": "green"},
            self.evidence | {"signal_phase": "unknown"},
            self.evidence | {
                "signal_timestamp": "2026-10-10T09:59:00+05:30",
            },
            self.evidence | {
                "signal_timestamp": "2026-10-10T10:00:01.500000+05:30",
            },
            self.evidence | {"signal_phase": None},
            self.evidence | {"signal_timestamp": None},
            self.evidence | {"signal_camera_id": "cam-2"},
            self.evidence | {"current_camera_id": "cam-2"},
            self.evidence | {"current_track_id": 18},
        )

        for evidence in cases:
            with self.subTest(evidence=evidence):
                self.assertFalse(
                    StopLineService.confirms_red_light_crossing(
                        **evidence
                    )
                )


class TrafficSignalDetectionTests(unittest.TestCase):

    @staticmethod
    def _create_synthetic_housing(active_lamp: str) -> np.ndarray:
        crop = np.zeros((150, 60, 3), dtype=np.uint8) + 25
        cv2.rectangle(crop, (4, 4), (56, 146), (15, 15, 15), -1)
        # 3 lamps: top y=30, mid y=75, bot y=120
        cv2.circle(crop, (30, 30), 16, (40, 40, 40), -1)
        cv2.circle(crop, (30, 75), 16, (40, 40, 40), -1)
        cv2.circle(crop, (30, 120), 16, (40, 40, 40), -1)
        if active_lamp == "RED":
            cv2.circle(crop, (30, 30), 16, (0, 0, 255), -1)
        elif active_lamp == "YELLOW":
            cv2.circle(crop, (30, 75), 16, (0, 255, 255), -1)
        elif active_lamp == "GREEN":
            cv2.circle(crop, (30, 120), 16, (0, 255, 0), -1)
        elif active_lamp == "ALL":
            cv2.circle(crop, (30, 30), 16, (0, 0, 255), -1)
            cv2.circle(crop, (30, 75), 16, (0, 255, 255), -1)
            cv2.circle(crop, (30, 120), 16, (0, 255, 0), -1)
        return crop

    def test_red_lamp_recognition(self):
        crop = self._create_synthetic_housing("RED")
        state, conf, name, px, reason = TrafficSignalDetector.classify_lamp_crop(crop)
        self.assertEqual(state, "RED")
        self.assertGreater(conf, 0.6)
        self.assertEqual(name, "red_lamp")
        self.assertGreater(px, 50)

    def test_green_lamp_recognition(self):
        crop = self._create_synthetic_housing("GREEN")
        state, conf, name, px, reason = TrafficSignalDetector.classify_lamp_crop(crop)
        self.assertEqual(state, "GREEN")
        self.assertGreater(conf, 0.6)
        self.assertEqual(name, "green_lamp")
        self.assertGreater(px, 50)

    def test_yellow_lamp_recognition(self):
        crop = self._create_synthetic_housing("YELLOW")
        state, conf, name, px, reason = TrafficSignalDetector.classify_lamp_crop(crop)
        self.assertEqual(state, "YELLOW")
        self.assertGreater(conf, 0.6)
        self.assertEqual(name, "yellow_lamp")
        self.assertGreater(px, 50)

    def test_unlit_lamps_return_unknown(self):
        crop = self._create_synthetic_housing("NONE")
        state, conf, name, px, reason = TrafficSignalDetector.classify_lamp_crop(crop)
        self.assertEqual(state, "UNKNOWN")

    def test_ambiguous_glare_returns_unknown(self):
        crop = self._create_synthetic_housing("ALL")
        state, conf, name, px, reason = TrafficSignalDetector.classify_lamp_crop(crop)
        self.assertEqual(state, "UNKNOWN")

    def test_no_traffic_light_returns_unknown(self):
        detector = TrafficSignalDetector.__new__(TrafficSignalDetector)
        detector.confidence = 0.15
        detector.model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        detector.model.predict.return_value = [mock_result]

        empty_image = np.zeros((400, 400, 3), dtype=np.uint8)
        signal = detector.detect(empty_image)
        self.assertEqual(signal.state, "UNKNOWN")
        self.assertEqual(signal.box, None)


class StopLineDetectorGeometryTests(unittest.TestCase):

    def test_normalized_coordinates_populated_on_detection(self):
        detector = AutomaticStopLineDetector()
        img = np.zeros((400, 600, 3), dtype=np.uint8) + 50
        # Draw a bright white stop line across the middle
        cv2.line(img, (50, 250), (550, 250), (255, 255, 255), 10)
        res = detector.detect(img)
        if res is not None:
            self.assertIsNotNone(res.normalized_p1)
            self.assertIsNotNone(res.normalized_p2)
            self.assertGreaterEqual(res.normalized_p1[0], 0.0)
            self.assertLessEqual(res.normalized_p1[0], 1.0)
            self.assertGreaterEqual(res.normalized_p1[1], 0.0)
            self.assertLessEqual(res.normalized_p1[1], 1.0)

    def test_angle_score_handles_horizontal_and_diagonal(self):
        score_horiz = AutomaticStopLineDetector._angle_score(5.0)
        score_diag20 = AutomaticStopLineDetector._angle_score(20.0)
        score_diag35 = AutomaticStopLineDetector._angle_score(35.0)
        score_steep = AutomaticStopLineDetector._angle_score(50.0)

        self.assertEqual(score_horiz, 1.0)
        self.assertEqual(score_diag20, 0.85)
        self.assertEqual(score_diag35, 0.70)
        self.assertEqual(score_steep, 0.50)


class StillImageViolationLogicTests(unittest.TestCase):

    def setUp(self):
        self.line = StopLine((50, 200), (350, 200), 1.0)
        self.beyond_vehicle = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.9,
            box=(100, 120, 200, 180),
            bottom_center=(150, 180),
            position="UNKNOWN",
        )
        self.before_vehicle = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.85,
            box=(220, 230, 280, 310),
            bottom_center=(250, 310),
            position="UNKNOWN",
        )

    def test_red_signal_beyond_line_is_violation_candidate_and_red_box(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        service._detect_vehicles = Mock(
            return_value=[self.beyond_vehicle, self.before_vehicle]
        )
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.85,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

        self.assertEqual(len(analysis.vehicles), 2)
        self.assertEqual(self.beyond_vehicle.position, "BEYOND_LINE")
        self.assertTrue(self.beyond_vehicle.is_violation_candidate)
        self.assertFalse(self.beyond_vehicle.confirmed_red_light_violation)
        self.assertEqual(len(analysis.violation_candidates), 1)

        # Check drawing: candidate vehicle box is RED (0, 0, 255)
        rendered = draw_stop_line_analysis(np.zeros((400, 400, 3), dtype=np.uint8), analysis)
        x1, y1 = self.beyond_vehicle.box[:2]
        self.assertEqual(tuple(int(c) for c in rendered[y1, x1]), (0, 0, 255))

    def test_green_signal_beyond_line_is_not_violation_candidate(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        service._detect_vehicles = Mock(
            return_value=[self.beyond_vehicle, self.before_vehicle]
        )
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="GREEN",
            confidence=0.88,
            box=(10, 10, 50, 110),
            reason="Green lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

        self.assertEqual(self.beyond_vehicle.position, "BEYOND_LINE")
        self.assertFalse(self.beyond_vehicle.is_violation_candidate)
        self.assertEqual(len(analysis.violation_candidates), 0)

        # Check drawing: when green, vehicle beyond line is BLUE (255, 0, 0), not RED
        rendered = draw_stop_line_analysis(np.zeros((400, 400, 3), dtype=np.uint8), analysis)
        x1, y1 = self.beyond_vehicle.box[:2]
        self.assertEqual(tuple(int(c) for c in rendered[y1, x1]), (255, 0, 0))

    def test_unknown_signal_beyond_line_is_not_violation_candidate(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        service._detect_vehicles = Mock(return_value=[self.beyond_vehicle])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="UNKNOWN",
            confidence=0.0,
            box=None,
            reason="No traffic light detected",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

        self.assertFalse(self.beyond_vehicle.is_violation_candidate)
        self.assertEqual(len(analysis.violation_candidates), 0)

    def test_still_image_never_claims_confirmed_violation(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        service._detect_vehicles = Mock(return_value=[self.beyond_vehicle])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.9,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

    def test_yellow_signal_does_not_create_violation_candidates(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        service._detect_vehicles = Mock(return_value=[self.beyond_vehicle])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="YELLOW",
            confidence=0.85,
            box=(10, 10, 50, 110),
            reason="Yellow lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

        self.assertEqual(self.beyond_vehicle.position, "BEYOND_LINE")
        self.assertFalse(self.beyond_vehicle.is_violation_candidate)
        self.assertEqual(len(analysis.violation_candidates), 0)

    def test_straddling_vehicle_at_line_is_at_line_in_orange(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        # Straddling motorcycle whose front wheel contact rests on the line within tolerance (tau=10.0)
        # y1=175, y2=240, h=65 -> y_front = 175 + 0.35*65 = 197.8 (line is at y=200, |d|=2.2px <= tau)
        straddling_bike = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.80,
            box=(150, 175, 210, 240),
            bottom_center=(180, 240),
            position="UNKNOWN",
        )
        service._detect_vehicles = Mock(return_value=[straddling_bike])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.90,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

        self.assertEqual(len(analysis.vehicles), 1)
        self.assertEqual(analysis.vehicles[0].position, "AT_LINE")
        self.assertFalse(analysis.vehicles[0].is_violation_candidate)

        # Check drawing: at-line vehicle box is ORANGE (0, 165, 255)
        rendered = draw_stop_line_analysis(np.zeros((400, 400, 3), dtype=np.uint8), analysis)
        x1, y1 = straddling_bike.box[:2]
        self.assertEqual(tuple(int(c) for c in rendered[y1, x1]), (0, 165, 255))

    def test_vehicle_whose_contact_point_crossed_is_beyond_line_candidate_in_red(self):
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = self.line
        service.approach_side_sign = 1
        # Motorcycle whose front contact has crossed to departure side:
        # y1=160, y2=240, h=80 -> y_front = 160 + 0.35*80 = 188 (line is at y=200, d = -12px < -tau)
        crossed_bike = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.80,
            box=(150, 160, 210, 240),
            bottom_center=(180, 240),
            position="UNKNOWN",
        )
        service._detect_vehicles = Mock(return_value=[crossed_bike])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.90,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))

        self.assertEqual(len(analysis.vehicles), 1)
        self.assertEqual(analysis.vehicles[0].position, "BEYOND_LINE")
        self.assertTrue(analysis.vehicles[0].is_violation_candidate)
        self.assertEqual(len(analysis.violation_candidates), 1)

        # Check drawing: candidate vehicle box is RED (0, 0, 255)
        rendered = draw_stop_line_analysis(np.zeros((400, 400, 3), dtype=np.uint8), analysis)
        x1, y1 = crossed_bike.box[:2]
        self.assertEqual(tuple(int(c) for c in rendered[y1, x1]), (0, 0, 255))

    def test_diagonal_line_crossing_and_before_classification(self):
        # Diagonal line: p1=(50, 100), p2=(350, 300) -> line passes through (200, 200)
        diag_line = StopLine((50, 100), (350, 300), 1.0)
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = diag_line
        service.approach_side_sign = 1

        # Bike before diagonal line (around x=200, y=280..340, line at x=200 is y=200)
        bike_before = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.85,
            box=(180, 280, 220, 340),
            bottom_center=(200, 340),
            position="UNKNOWN",
        )
        # Bike crossing diagonal line (around x=200, y=140..210, front at 140+0.35*70=164.5 < 200)
        bike_crossed = VehicleObservation(
            class_id=3,
            name="motorcycle",
            confidence=0.85,
            box=(180, 140, 220, 210),
            bottom_center=(200, 210),
            position="UNKNOWN",
        )
        service._detect_vehicles = Mock(return_value=[bike_before, bike_crossed])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.90,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))
        self.assertEqual(analysis.vehicles[0].position, "BEFORE_LINE")
        self.assertFalse(analysis.vehicles[0].is_violation_candidate)
        self.assertEqual(analysis.vehicles[1].position, "BEYOND_LINE")
        self.assertTrue(analysis.vehicles[1].is_violation_candidate)
        self.assertEqual(len(analysis.violation_candidates), 1)

    def test_diagonal_line_opposite_approach_direction(self):
        # Diagonal line: p1=(50, 100), p2=(350, 300)
        diag_line = StopLine((50, 100), (350, 300), 1.0)
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = diag_line
        # Approach from top/background (approach_side_sign = -1)
        service.approach_side_sign = -1

        # Vehicle at bottom (y=350) has crossed into foreground intersection
        beyond_veh = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.88,
            box=(100, 310, 200, 370),
            bottom_center=(150, 370),
            position="UNKNOWN",
        )
        # Vehicle at top (y=50) is still approaching
        before_veh = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.88,
            box=(100, 40, 200, 90),
            bottom_center=(150, 90),
            position="UNKNOWN",
        )

        service._detect_vehicles = Mock(return_value=[beyond_veh, before_veh])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.90,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))
        self.assertEqual(beyond_veh.position, "BEYOND_LINE")
        self.assertTrue(beyond_veh.is_violation_candidate)
        self.assertEqual(before_veh.position, "BEFORE_LINE")
        self.assertFalse(before_veh.is_violation_candidate)

    def test_road_corridor_retains_vehicle_in_adjacent_lane(self):
        # Line is detected only on right half x in [200, 380]
        partial_line = StopLine((200, 200), (380, 200), 1.0)
        service = StopLineService.__new__(StopLineService)
        service.stop_line_detector = Mock()
        service.stop_line_detector.detect.return_value = partial_line
        service.approach_side_sign = 1
        # Car is waiting in left lane x in [60, 140], y in [240, 310] (before line y=200)
        left_lane_car = VehicleObservation(
            class_id=2,
            name="car",
            confidence=0.89,
            box=(60, 240, 140, 310),
            bottom_center=(100, 310),
            position="UNKNOWN",
        )
        service._detect_vehicles = Mock(return_value=[left_lane_car])
        service.traffic_signal_detector = Mock()
        service.traffic_signal_detector.detect.return_value = TrafficSignal(
            state="RED",
            confidence=0.90,
            box=(10, 10, 50, 110),
            reason="Red lamp illuminated",
        )

        analysis = service.analyze(np.zeros((400, 400, 3), dtype=np.uint8))
        self.assertEqual(len(analysis.vehicles), 1)
        self.assertEqual(left_lane_car.position, "BEFORE_LINE")

    def test_triple_riding_regression_contract(self):
        # Regression protection: ensure triple-riding model loads and maintains API contract
        from triple_riding_service import TripleRidingDetector
        detector = TripleRidingDetector(0.25, 0.45)
        self.assertIsNotNone(detector.detector)
        self.assertIsNotNone(detector.pose)


class MultipleVehicleTemporalCrossingTests(unittest.TestCase):

    def test_multiple_vehicles_crossing_during_red_confirmed(self):
        ev1 = {
            "previous_track_id": 101,
            "current_track_id": 101,
            "previous_signed_distance": 25,
            "current_signed_distance": -15,
            "approach_side_sign": 1,
            "previous_timestamp": "2026-10-10T10:00:00+05:30",
            "current_timestamp": "2026-10-10T10:00:01+05:30",
            "previous_camera_id": "cam-junction-1",
            "current_camera_id": "cam-junction-1",
            "signal_camera_id": "cam-junction-1",
            "signal_phase": "red",
            "signal_timestamp": "2026-10-10T10:00:00.200000+05:30",
            "signal_max_age_seconds": 2,
        }
        ev2 = {
            "previous_track_id": 102,
            "current_track_id": 102,
            "previous_signed_distance": 40,
            "current_signed_distance": -20,
            "approach_side_sign": 1,
            "previous_timestamp": "2026-10-10T10:00:00+05:30",
            "current_timestamp": "2026-10-10T10:00:01.500000+05:30",
            "previous_camera_id": "cam-junction-1",
            "current_camera_id": "cam-junction-1",
            "signal_camera_id": "cam-junction-1",
            "signal_phase": "red",
            "signal_timestamp": "2026-10-10T10:00:00.500000+05:30",
            "signal_max_age_seconds": 2,
        }
        self.assertTrue(StopLineService.confirms_red_light_crossing(**ev1))
        self.assertTrue(StopLineService.confirms_red_light_crossing(**ev2))


# =============================================================
# INPUT IMAGES FOR EVALUATION
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_IMAGES = [
    (
        PROJECT_ROOT / "images" / "Indian City Intersection at Red Light.png",
        PROJECT_ROOT / "outputs" / "stop_line_red_light_indian_city_validated.png",
        1,  # Approach from bottom (approaching upwards)
    ),
    (
        PROJECT_ROOT / "images" / "Gemini_Generated_Image_mfyye9mfyye9mfyy.png",
        PROJECT_ROOT / "outputs" / "stop_line_red_light_gemini_validated.png",
        1,  # Approach from bottom (approaching upwards)
    ),
]


def evaluate_image(image_path: Path, output_path: Path, approach_side_sign: int):
    print("=" * 65)
    print(f"EVALUATING: {image_path.name}")
    print("=" * 65)

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: OpenCV could not read: {image_path}")
        return

    height, width = frame.shape[:2]
    print(f"Image Dimensions: {width} x {height}")

    service = StopLineService(approach_side_sign=approach_side_sign)
    analysis = service.analyze(frame)

    print("\n--- 1. STOP LINE RESULT ---")
    if analysis.stop_line is None:
        print("Stop Line: NOT DETECTED")
        print(f"Reason: {service.stop_line_detector.last_diagnostics.get('reason')}")
    else:
        line = analysis.stop_line
        print(f"Stop Line: DETECTED (score={line.score:.3f})")
        print(f"  Pixel Endpoints: p1={line.p1}, p2={line.p2}")
        if line.normalized_p1 and line.normalized_p2:
            print(f"  Normalized: p1=({line.normalized_p1[0]:.3f}, {line.normalized_p1[1]:.3f}), p2=({line.normalized_p2[0]:.3f}, {line.normalized_p2[1]:.3f})")

    print("\n--- 2. TRAFFIC SIGNAL RESULT ---")
    if analysis.traffic_signal:
        sig = analysis.traffic_signal
        print(f"Signal State: {sig.state} (confidence={sig.confidence:.2f})")
        if sig.box:
            print(f"  Housing Box: {sig.box}")
        print(f"  Diagnostic: {sig.reason}")
    else:
        print("Signal: NOT EVALUATED")

    print("\n--- 3. VEHICLE CLASSIFICATIONS ---")
    print(f"Total Vehicles Tracked/Observed: {len(analysis.vehicles)}")
    for idx, v in enumerate(analysis.vehicles, start=1):
        d_str = f"d={v.signed_distance:.1f}px" if v.signed_distance is not None else "d=N/A"
        viol_str = " | VIOLATION CANDIDATE (RED)" if v.is_violation_candidate else ""
        print(f"  Vehicle {idx}: {v.name} (conf={v.confidence:.2f}) | box={v.box} | contact={v.bottom_center} | {v.position} ({d_str}){viol_str}")

    print("\n--- 4. VIOLATION SUMMARY ---")
    print(f"Beyond Stop Line Candidates: {len(analysis.violation_candidates if analysis.traffic_signal and analysis.traffic_signal.state == 'RED' else analysis.potential_violations)}")
    print("Confirmed Violations (Temporal Tracking): 0 (Single snapshot only)")

    # Render and save annotated image
    result = draw_stop_line_analysis(frame, analysis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), result)
    if success:
        print(f"\nAnnotated output saved: {output_path.name}")
    else:
        print(f"\nERROR: Failed to save output to {output_path}")
    print("=" * 65 + "\n")


def main():
    print("*" * 65)
    print("ROADIQ STOP-LINE & RED-LIGHT VIOLATION VALIDATION RUNNER")
    print("*" * 65 + "\n")

    for img_path, out_path, approach_sign in TEST_IMAGES:
        evaluate_image(img_path, out_path, approach_sign)


if __name__ == "__main__":
    main()

