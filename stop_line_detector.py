from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StopLine:
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    score: float
    line_type: str = "STOP LINE"
    normalized_p1: Optional[Tuple[float, float]] = None
    normalized_p2: Optional[Tuple[float, float]] = None


class AutomaticStopLineDetector:
    """
    Intelligent Stop-Line and Zebra-Crossing Detector.
    Detects both:
    1. Zebra Crossings (Pedestrian crosswalks with parallel white or yellow stripes)
       and fits the entrance baseline facing oncoming traffic as the stop line.
    2. Solid transverse stop lines painted on the asphalt.
    """

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
        approach_hint: Optional[str] = None,
    ) -> Optional[StopLine]:
        if frame is None or frame.size == 0:
            self.last_diagnostics = {
                "status": "uncertain",
                "reason": "Input image is empty.",
                "candidates": [],
                "selected": None,
            }
            return None

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # 1. Multi-spectrum Road Marking Segmentation (White + Yellow + TopHat)
        # -------------------------------------------------------------
        # White road markings
        lower_white = np.array([0, 0, 130], dtype=np.uint8)
        upper_white = np.array([180, 90, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Yellow road markings (for crosswalks like stop_line4.png)
        lower_yellow = np.array([14, 50, 90], dtype=np.uint8)
        upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # High-contrast road features via morphological Top-Hat
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, tophat_mask = cv2.threshold(tophat, 32, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, tophat_mask)

        # Mask out sky, billboards, or upper 15% of frame
        combined_mask[: int(height * 0.15), :] = 0

        # Clean noise
        clean_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        clean_mask = cv2.morphologyEx(
            clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        # -------------------------------------------------------------
        # 2. Zebra Crossing Stripe Detection
        # -------------------------------------------------------------
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        stripes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 200 or area > 85000:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            if rw < 6 or rh < 6:
                continue

            aspect = max(rw, rh) / min(rw, rh)
            if aspect < 1.25:
                continue

            box = cv2.boxPoints(rect)
            v0 = box[1] - box[0]
            v1 = box[2] - box[1]
            if np.linalg.norm(v0) > np.linalg.norm(v1):
                end1 = (box[0] + box[3]) / 2.0
                end2 = (box[1] + box[2]) / 2.0
            else:
                end1 = (box[0] + box[1]) / 2.0
                end2 = (box[2] + box[3]) / 2.0

            stripes.append({
                "center": (cx, cy),
                "end1": end1,
                "end2": end2,
                "box": box,
                "area": area,
                "rect": rect,
                "aspect": aspect,
            })

        # -------------------------------------------------------------
        # 3. Solid Transverse Line Candidates (HoughLinesP)
        # -------------------------------------------------------------
        edges = cv2.Canny(clean_mask, 40, 130)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=max(30, int(width * self.min_line_ratio)),
            maxLineGap=25,
        )

        transverse_lines = []
        if lines is not None:
            for raw_line in lines:
                pts = raw_line.reshape(-1)
                if pts.size != 4:
                    continue
                x1, y1, x2, y2 = map(int, pts)
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < width * self.min_line_ratio:
                    continue
                angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
                y_mid = (y1 + y2) / 2.0
                if y_mid > height * 0.22:
                    transverse_lines.append({
                        "p1": (x1, y1),
                        "p2": (x2, y2),
                        "length": length,
                        "angle": angle,
                        "y_mid": y_mid,
                    })

        # -------------------------------------------------------------
        # 4. Synthesize Stop Line: Check Zebra Crossing First
        # -------------------------------------------------------------
        if len(stripes) >= 3:
            # Derive dominant stripe orientation geometrically:
            # Cars drive along the long axis of the zebra stripes.
            stripe_dx = []
            stripe_dy = []
            for s in stripes:
                v = s["end2"] - s["end1"]
                stripe_dx.append(abs(v[0]))
                stripe_dy.append(abs(v[1]))
            mean_dx = float(np.mean(stripe_dx)) if stripe_dx else 1.0
            mean_dy = float(np.mean(stripe_dy)) if stripe_dy else 1.0

            if approach_hint is None or approach_hint == "auto":
                if mean_dx > mean_dy * 1.3:
                    chosen_dir = "left"
                else:
                    chosen_dir = "top"
            else:
                chosen_dir = approach_hint

            entrance_pts = []
            for s in stripes:
                e1, e2 = s["end1"], s["end2"]
                if chosen_dir == "left":
                    pt = e1 if e1[0] < e2[0] else e2
                elif chosen_dir == "right":
                    pt = e1 if e1[0] > e2[0] else e2
                elif chosen_dir == "bottom":
                    pt = e1 if e1[1] > e2[1] else e2
                else:  # top / moving down or diagonal
                    pt = e1 if e1[1] < e2[1] else e2
                entrance_pts.append(pt)

            entrance_pts = np.asarray(entrance_pts, dtype=np.float32)
            if len(entrance_pts) >= 3:
                fitted = cv2.fitLine(entrance_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                vx = float(fitted[0][0])
                vy = float(fitted[1][0])
                x0 = float(fitted[2][0])
                y0 = float(fitted[3][0])

                if chosen_dir in ("left", "right") and abs(vy) > 0.30:
                    # Vertical / steep baseline facing left/right traffic
                    min_y = max(0, int(np.min(entrance_pts[:, 1]) - 30))
                    max_y = min(height - 1, int(np.max(entrance_pts[:, 1]) + 30))
                    if abs(vy) > 1e-4:
                        x1 = int(np.clip(x0 + (vx / vy) * (min_y - y0), 0, width - 1))
                        x2 = int(np.clip(x0 + (vx / vy) * (max_y - y0), 0, width - 1))
                    else:
                        x1, x2 = int(x0), int(x0)
                    p1, p2 = (x1, min_y), (x2, max_y)
                else:
                    # Transverse baseline across road width
                    min_x = max(0, int(np.min(entrance_pts[:, 0]) - 30))
                    max_x = min(width - 1, int(np.max(entrance_pts[:, 0]) + 30))
                    if abs(vx) > 1e-4:
                        y1 = int(np.clip(y0 + (vy / vx) * (min_x - x0), 0, height - 1))
                        y2 = int(np.clip(y0 + (vy / vx) * (max_x - x0), 0, height - 1))
                    else:
                        y1, y2 = int(y0), int(y0)
                    p1, p2 = (min_x, y1), (max_x, y2)

                norm_p1 = (float(p1[0] / width), float(p1[1] / height))
                norm_p2 = (float(p2[0] / width), float(p2[1] / height))

                self.last_diagnostics = {
                    "status": "detected",
                    "reason": f"Detected zebra crossing entrance with {len(stripes)} stripes.",
                    "line_type": "ZEBRA CROSSING START",
                    "selected": {"p1": p1, "p2": p2, "rank_score": 0.88},
                }

                return StopLine(
                    p1=p1,
                    p2=p2,
                    score=0.88,
                    line_type="ZEBRA CROSSING START",
                    normalized_p1=norm_p1,
                    normalized_p2=norm_p2,
                )

        # -------------------------------------------------------------
        # 5. Check Solid Transverse Stop Lines
        # -------------------------------------------------------------
        if transverse_lines:
            best_line = max(transverse_lines, key=lambda l: l["length"])
            p1 = best_line["p1"]
            p2 = best_line["p2"]
            norm_p1 = (float(p1[0] / width), float(p1[1] / height))
            norm_p2 = (float(p2[0] / width), float(p2[1] / height))

            self.last_diagnostics = {
                "status": "detected",
                "reason": "Detected solid transverse stop line.",
                "line_type": "SOLID STOP LINE",
                "selected": {"p1": p1, "p2": p2, "rank_score": 0.82},
            }

            return StopLine(
                p1=p1,
                p2=p2,
                score=0.82,
                line_type="SOLID STOP LINE",
                normalized_p1=norm_p1,
                normalized_p2=norm_p2,
            )

        # -------------------------------------------------------------
        # 6. Fallback to Sensible Default Road Baseline
        # -------------------------------------------------------------
        default_y = int(height * 0.65)
        p1 = (0, default_y)
        p2 = (width - 1, default_y)
        norm_p1 = (0.0, 0.65)
        norm_p2 = (1.0, 0.65)

        self.last_diagnostics = {
            "status": "estimated",
            "reason": "Estimated road baseline across lower road section.",
            "line_type": "ROAD BASELINE (ESTIMATED)",
            "selected": {"p1": p1, "p2": p2, "rank_score": 0.50},
        }

        return StopLine(
            p1=p1,
            p2=p2,
            score=0.50,
            line_type="ROAD BASELINE (ESTIMATED)",
            normalized_p1=norm_p1,
            normalized_p2=norm_p2,
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

    cv2.line(result, stop_line.p1, stop_line.p2, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.line(result, stop_line.p1, stop_line.p2, (0, 255, 255), 5, cv2.LINE_AA)

    label = f"🛑 {stop_line.line_type} | score={stop_line.score:.2f}"
    text_x = min(stop_line.p1[0], stop_line.p2[0])
    text_y = max(35, min(stop_line.p1[1], stop_line.p2[1]) - 12)

    cv2.putText(
        result,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result
