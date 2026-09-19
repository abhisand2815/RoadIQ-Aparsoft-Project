"""Pretrained-model rider-to-motorcycle association.

No custom classifier is used. Person and motorcycle boxes come from YOLO;
pose keypoints and transparent geometry are combined into an association score.
"""

from dataclasses import dataclass
import math
import numpy as np
import config


@dataclass
class Person:
    track_id: int
    box: tuple[float, float, float, float]
    keypoints: object = None
    confidence: float = 0.0


@dataclass
class Motorcycle:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float = 0.0


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(a) + _area(b) - inter

    return inter / union if union > 0 else 0.0


def _expanded(box, ratio):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    return (
        x1 - w * ratio,
        y1 - h * ratio,
        x2 + w * ratio,
        y2 + h * ratio,
    )


def _normalized_distance(person_box, bike_box):
    px, py = _bottom_center(person_box)
    bx, by = _center(bike_box)

    bw = max(1.0, bike_box[2] - bike_box[0])
    bh = max(1.0, bike_box[3] - bike_box[1])

    return math.hypot(
        (px - bx) / bw,
        (py - by) / bh,
    )


def _rider_position_score(person_box, bike_box):
    """Check whether the person's lower body is positioned over the bike."""

    px, py = _bottom_center(person_box)

    bx1, by1, bx2, by2 = bike_box

    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    # Keep the horizontal rider area close to the motorcycle.
    horizontal_margin = bw * 0.10

    if px < bx1 - horizontal_margin or px > bx2 + horizontal_margin:
        return 0.0

    # A rider's lower body should normally be around the motorcycle
    # rather than clearly above it.
    bike_center_y = (by1 + by2) / 2.0

    if py < bike_center_y - (bh * 0.10):
        return 0.0

    if py > by2 + (bh * 0.20):
        return 0.0

    # Normalize the horizontal position.
    horizontal_distance = abs(px - ((bx1 + bx2) / 2.0)) / bw

    return max(
        0.0,
        1.0 - horizontal_distance
    )


def _pose_score(person, bike):
    kp = person.keypoints

    if kp is None or len(kp) == 0:
        return 0.0

    # Use a smaller expansion than the full association region.
    bike_region = _expanded(bike.box, 0.10)

    bx1, by1, bx2, by2 = bike_region

    useful = []

    # COCO pose:
    # 11/12 = hips
    # 15/16 = ankles
    for idx in (11, 12, 15, 16):

        if idx >= len(kp):
            continue

        x = float(kp[idx][0])
        y = float(kp[idx][1])

        conf = (
            float(kp[idx][2])
            if len(kp[idx]) > 2
            else 1.0
        )

        if conf >= 0.25:
            useful.append((x, y, conf))

    if not useful:
        return 0.0

    inside = sum(
        1
        for x, y, _ in useful
        if bx1 <= x <= bx2 and by1 <= y <= by2
    )

    return inside / len(useful)


def association_score(person, bike):
    """Return a transparent 0..1 pretrained-model association score."""

    # Use a limited expansion for actual rider association.
    expanded = _expanded(bike.box, 0.10)

    overlap = _iou(person.box, expanded)

    # Require some actual overlap with the motorcycle region.
    if overlap < config.MIN_ASSOCIATION_IOU:
        return 0.0

    distance = _normalized_distance(
        person.box,
        bike.box
    )

    if distance > config.MAX_ASSOCIATION_DISTANCE:
        return 0.0

    # Reject people whose lower body is clearly outside the bike.
    rider_position = _rider_position_score(
        person.box,
        bike.box
    )

    if rider_position <= 0.0:
        return 0.0

    distance_score = max(
        0.0,
        1.0 - distance / config.MAX_ASSOCIATION_DISTANCE
    )

    pose_score = _pose_score(person, bike)

    score = (
        config.OVERLAP_WEIGHT * overlap
        + config.DISTANCE_WEIGHT * distance_score
        + config.POSE_BONUS_WEIGHT * pose_score
    )

    # Rider position is an additional check, not a replacement
    # for the original pretrained-model association score.
    score *= (0.70 + 0.30 * rider_position)

    return float(np.clip(score, 0.0, 1.0))


def associate_people_to_bikes(
    persons,
    motorcycles,
    threshold=None
):
    """Associate each detected person with the best motorcycle."""

    threshold = (
        config.DEFAULT_ASSOCIATION_THRESHOLD
        if threshold is None
        else threshold
    )

    result = {
        bike.track_id: []
        for bike in motorcycles
    }

    for person in persons:

        # Ignore very low-confidence detections.
        if (
            person.confidence > 0.0
            and person.confidence < 0.30
        ):
            continue

        candidates = []

        for bike in motorcycles:

            score = association_score(
                person,
                bike
            )

            if score >= threshold:
                candidates.append(
                    (score, bike.track_id)
                )

        # Associate a person with only the best motorcycle.
        if candidates:

            best_score, best_bike_id = max(
                candidates,
                key=lambda item: item[0]
            )

            result[best_bike_id].append(
                (person, best_score)
            )

    return result
