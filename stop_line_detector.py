from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StopLine:

    p1: Tuple[int, int]
    p2: Tuple[int, int]
    score: float
    normalized_p1: Optional[Tuple[float, float]] = None
    normalized_p2: Optional[Tuple[float, float]] = None


class AutomaticStopLineDetector:


    def __init__(
        self,
        min_line_ratio: float = 0.08,
        min_score: float = 0.30,
    ):
        self.min_line_ratio = min_line_ratio
        self.min_score = min_score
        self.last_diagnostics: dict[str, object] = {
            "status": "not_run",
            "reason": "Detection has not run.",
            "candidates": [],
            "selected": None,
        }

    def detect(
        self,
        frame: np.ndarray,
    ) -> Optional[StopLine]:

        if frame is None or frame.size == 0:
            self.last_diagnostics = {
                "status": "uncertain",
                "reason": "Input image is empty.",
                "candidates": [],
                "selected": None,
            }
            return None

        self.last_diagnostics = {
            "status": "uncertain",
            "reason": "No reliable stop-line candidate found.",
            "candidates": [],
            "selected": None,
        }

        height, width = frame.shape[:2]



        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road_values = hsv[
            int(height * 0.20):,
            :,
            2,
        ]
        value_threshold = int(
            np.clip(
                np.percentile(road_values, 95) + 10,
                150,
                175,
            )
        )

        lower_white = np.array(
            [0, 0, value_threshold],
            dtype=np.uint8,
        )

        upper_white = np.array(
            [180, 100, 255],
            dtype=np.uint8,
        )

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white,
        )

       
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )


        road_mask = np.zeros_like(white_mask)

        cv2.fillPoly(
            road_mask,
            [
                np.array(
                    [
                        [0, int(height * 0.20)],
                        [width, int(height * 0.20)],
                        [width, height],
                        [0, height],
                    ],
                    dtype=np.int32,
                )
            ],
            255,
        )

        white_mask = cv2.bitwise_and(
            white_mask,
            road_mask,
        )



        edges = cv2.Canny(
            white_mask,
            40,
            120,
            apertureSize=3,
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=30,
            minLineLength=max(
                30,
                int(width * self.min_line_ratio),
            ),
            maxLineGap=max(
                10,
                int(width * 0.025),
            ),
        )

        if lines is None:
            self.last_diagnostics["reason"] = (
                "Hough transform found no line segments."
            )
            return None

        candidates = []



        for raw_line in lines:

            line = np.asarray(
                raw_line
            ).reshape(-1)

            if line.size != 4:
                continue

            x1, y1, x2, y2 = map(
                int,
                line,
            )

            dx = x2 - x1
            dy = y2 - y1

            length = float(
                np.hypot(dx, dy)
            )

            if length < width * self.min_line_ratio:
                continue

            angle = float(
                np.degrees(
                    np.arctan2(dy, dx)
                )
            )

            if angle < 0:
                angle += 180

            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            angle_from_horizontal = min(
                angle,
                180.0 - angle,
            )

            if angle_from_horizontal > 38.0:
                continue

            if not (
                height * 0.22
                <= center_y
                <= height * 0.92
            ):
                continue

            whiteness = self._line_whiteness(
                white_mask,
                x1,
                y1,
                x2,
                y2,
            )

            if whiteness < 0.25:
                continue

            angle_score = self._angle_score(
                angle_from_horizontal
            )

            position_score = self._position_score(
                center_y,
                height,
            )

            candidates.append(
                {
                    "length": length,
                    "whiteness": whiteness,
                    "angle_score": angle_score,
                    "position_score": position_score,
                    "angle": angle,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )

        if not candidates:
            self.last_diagnostics["reason"] = (
                "No Hough segments met the marking geometry and whiteness checks."
            )
            return None


        for candidate in candidates:
            x1 = candidate["x1"]
            y1 = candidate["y1"]
            x2 = candidate["x2"]
            y2 = candidate["y2"]
            dx = x2 - x1
            dy = y2 - y1
            norm = max(
                1.0,
                float(np.hypot(dx, dy)),
            )

            aligned = []
            parallel_offsets = set()

            for other in candidates:
                angle_difference = abs(
                    candidate["angle"] - other["angle"]
                )
                angle_difference = min(
                    angle_difference,
                    180.0 - angle_difference,
                )

                if angle_difference > 4.0:
                    continue

                overlap = min(
                    max(x1, x2),
                    max(other["x1"], other["x2"]),
                ) - max(
                    min(x1, x2),
                    min(other["x1"], other["x2"]),
                )
                shorter_span = min(
                    abs(x2 - x1),
                    abs(other["x2"] - other["x1"]),
                )
                if shorter_span <= 0:
                    continue

                center_x = other["center_x"]
                center_y = other["center_y"]
                signed_distance = (
                    dx * (center_y - y1)
                    - dy * (center_x - x1)
                ) / norm

                alignment_tolerance = max(
                    6.0,
                    min(width, height) * 0.025,
                )
                if abs(signed_distance) <= alignment_tolerance:
                    aligned.append(other)
                    continue

                if overlap < shorter_span * 0.5:
                    continue

                parallel_min = max(
                    8.0,
                    min(width, height) * 0.025,
                )
                parallel_max = min(width, height) * 0.18
                if parallel_min <= abs(signed_distance) <= parallel_max:
                    parallel_offsets.add(
                        round(signed_distance / (parallel_min * 0.75))
                    )

            if not aligned:
                aligned = [candidate]

            span_left = min(
                min(item["x1"], item["x2"])
                for item in aligned
            )
            span_right = max(
                max(item["x1"], item["x2"])
                for item in aligned
            )
            span_score = min(
                1.0,
                (span_right - span_left) / (width * 0.45),
            )
            continuity = self._line_whiteness(
                white_mask,
                span_left,
                int(
                    self._y_at_x(
                        span_left,
                        x1,
                        y1,
                        dx,
                        dy,
                    )
                ),
                span_right,
                int(
                    self._y_at_x(
                        span_right,
                        x1,
                        y1,
                        dx,
                        dy,
                    )
                ),
            )
            parallel_penalty = min(
                0.30,
                len(parallel_offsets) * 0.10,
            )
            candidate["rank_score"] = (
                0.40 * span_score
                + 0.20 * candidate["whiteness"]
                + 0.15 * candidate["angle_score"]
                + 0.10 * candidate["position_score"]
                + 0.15 * continuity
                - parallel_penalty
            )
            candidate["group_span"] = (
                span_left,
                span_right,
            )
            candidate["continuity"] = continuity
            candidate["parallel_penalty"] = parallel_penalty

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                item["rank_score"],
                item["length"],
            ),
            reverse=True,
        )
        seed = ranked_candidates[0]
        candidate_diagnostics = [
            {
                "p1": (candidate["x1"], candidate["y1"]),
                "p2": (candidate["x2"], candidate["y2"]),
                "rank_score": float(candidate["rank_score"]),
                "length": float(candidate["length"]),
                "whiteness": float(candidate["whiteness"]),
                "group_span": candidate["group_span"],
                "continuity": float(candidate["continuity"]),
                "parallel_penalty": float(
                    candidate["parallel_penalty"]
                ),
            }
            for candidate in ranked_candidates[:5]
        ]
        self.last_diagnostics.update(
            {
                "reason": (
                    "Top candidates are too close in support."
                ),
                "candidates": candidate_diagnostics,
                "selected": {
                    "p1": (seed["x1"], seed["y1"]),
                    "p2": (seed["x2"], seed["y2"]),
                    "rank_score": float(seed["rank_score"]),
                },
            }
        )


        ambiguity_distance = max(
            12.0,
            min(width, height) * 0.035,
        )
        for other in ranked_candidates[1:]:
            angle_difference = abs(
                seed["angle"] - other["angle"]
            )
            angle_difference = min(
                angle_difference,
                180.0 - angle_difference,
            )

            if angle_difference > 4.0:
                continue

            seed_center_x = (
                seed["x1"] + seed["x2"]
            ) / 2.0
            seed_center_y = (
                seed["y1"] + seed["y2"]
            ) / 2.0
            other_center_x = other["center_x"]
            other_center_y = other["center_y"]
            line_distance = abs(
                (seed["x2"] - seed["x1"])
                * (other_center_y - seed_center_y)
                - (seed["y2"] - seed["y1"])
                * (other_center_x - seed_center_x)
            ) / max(
                1.0,
                seed["length"],
            )

            if (
                line_distance > ambiguity_distance
                and seed["rank_score"] - other["rank_score"] < 0.025
            ):
                self.last_diagnostics.update(
                    {
                        "status": "ambiguous",
                        "reason": (
                            "Spatially separate marking candidates "
                            "have nearly equal support."
                        ),
                        "competing_candidate": {
                            "p1": (other["x1"], other["y1"]),
                            "p2": (other["x2"], other["y2"]),
                            "rank_score": float(
                                other["rank_score"]
                            ),
                        },
                        "selected": None,
                    }
                )
                return None

        # ---------------------------------------------------------
        # 5. Collect all Hough segments belonging to the same
        #    physical stop line.
        # ---------------------------------------------------------

        seed_x1 = seed["x1"]
        seed_y1 = seed["y1"]
        seed_x2 = seed["x2"]
        seed_y2 = seed["y2"]

        a = seed_y1 - seed_y2
        b = seed_x2 - seed_x1
        c = (
            seed_x1 * seed_y2
            - seed_x2 * seed_y1
        )

        norm = float(
            np.hypot(a, b)
        )

        if norm == 0:
            self.last_diagnostics.update(
                {
                    "reason": "Selected candidate has zero length.",
                    "selected": None,
                }
            )
            return None

        points = []

        for candidate in candidates:

            angle_difference = abs(
                candidate["angle"]
                - seed["angle"]
            )

            angle_difference = min(
                angle_difference,
                180.0 - angle_difference,
            )

            if angle_difference > 4.0:
                continue

            distance = abs(
                a * candidate["center_x"]
                + b * candidate["center_y"]
                + c
            ) / norm

            if distance > max(
                6.0,
                min(width, height) * 0.025,
            ):
                continue

            points.append(
                (
                    candidate["x1"],
                    candidate["y1"],
                )
            )

            points.append(
                (
                    candidate["x2"],
                    candidate["y2"],
                )
            )

        if len(points) < 4:
            self.last_diagnostics.update(
                {
                    "status": "uncertain",
                    "reason": (
                        "Too few aligned Hough segments support the "
                        "selected candidate."
                    ),
                    "selected": None,
                }
            )
            return None

        points = np.asarray(
            points,
            dtype=np.float32,
        )



        fitted = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        )



        vx = float(fitted[0][0])
        vy = float(fitted[1][0])
        x0 = float(fitted[2][0])
        y0 = float(fitted[3][0])

        if abs(vx) < 1e-6:
            self.last_diagnostics.update(
                {
                    "reason": "Fitted line is nearly vertical.",
                    "selected": None,
                }
            )
            return None

        self.last_diagnostics["reason"] = (
            "Top candidate was separated from competitors and fitted."
        )


        x_left = max(
            0,
            int(np.min(points[:, 0])),
        )
        x_right = min(
            width - 1,
            int(np.max(points[:, 0])),
        )

        if x_right <= x_left:
            self.last_diagnostics.update(
                {
                    "reason": "Fitted line has no horizontal span.",
                    "selected": None,
                }
            )
            return None

        y_left = self._y_at_x(
            x_left,
            x0,
            y0,
            vx,
            vy,
        )

        y_right = self._y_at_x(
            x_right,
            x0,
            y0,
            vx,
            vy,
        )

        y_left = int(
            np.clip(
                y_left,
                0,
                height - 1,
            )
        )

        y_right = int(
            np.clip(
                y_right,
                0,
                height - 1,
            )
        )



        confidence = min(
            0.99,
            max(
                seed["rank_score"],
                0.0,
            ),
        )

        if confidence < self.min_score:
            self.last_diagnostics.update(
                {
                    "reason": (
                        "Best candidate support is below the minimum score."
                    ),
                    "selected": None,
                }
            )
            return None

        norm_p1 = (float(x_left / width), float(y_left / height))
        norm_p2 = (float(x_right / width), float(y_right / height))
        self.last_diagnostics.update(
            {
                "status": "detected",
                "reason": "A candidate passed the current support checks.",
                "selected": {
                    "p1": (x_left, y_left),
                    "p2": (x_right, y_right),
                    "normalized_p1": norm_p1,
                    "normalized_p2": norm_p2,
                    "rank_score": float(confidence),
                },
            }
        )
        return StopLine(
            p1=(
                x_left,
                y_left,
            ),
            p2=(
                x_right,
                y_right,
            ),
            score=float(
                confidence
            ),
            normalized_p1=norm_p1,
            normalized_p2=norm_p2,
        )

    @staticmethod
    def _y_at_x(
        x,
        x0,
        y0,
        vx,
        vy,
    ):
        return (
            y0
            + (vy / vx)
            * (x - x0)
        )

    @staticmethod
    def _angle_score(
        angle: float,
    ) -> float:

        if angle <= 10:
            return 1.0

        if angle <= 25:
            return 0.85

        if angle <= 38:
            return 0.70

        return 0.50

    @staticmethod
    def _position_score(
        center_y: float,
        height: int,
    ) -> float:

        y = center_y / height
        return float(
            np.clip(
                (y - 0.22) / 0.70,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _line_whiteness(
        mask: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> float:

        length = max(
            20,
            int(
                np.hypot(
                    x2 - x1,
                    y2 - y1,
                )
            ),
        )

        xs = np.linspace(
            x1,
            x2,
            length,
        ).astype(int)

        ys = np.linspace(
            y1,
            y2,
            length,
        ).astype(int)

        h, w = mask.shape[:2]

        valid = (
            (xs >= 0)
            & (xs < w)
            & (ys >= 0)
            & (ys < h)
        )

        if not np.any(valid):
            return 0.0

        values = mask[
            ys[valid],
            xs[valid],
        ]

        return float(
            np.mean(values > 0)
        )


def draw_stop_line(
    frame: np.ndarray,
    stop_line: Optional[StopLine],
) -> np.ndarray:

    result = frame.copy()

    if stop_line is None:

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

        return result

    cv2.line(
        result,
        stop_line.p1,
        stop_line.p2,
        (0, 255, 255),
        6,
        cv2.LINE_AA,
    )

    label = (
        f"STOP LINE | "
        f"score={stop_line.score:.2f}"
    )

    cv2.putText(
        result,
        label,
        (
            stop_line.p1[0],
            max(
                35,
                stop_line.p1[1] - 12,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result