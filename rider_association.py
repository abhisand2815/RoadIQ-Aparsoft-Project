
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


def _pose_score(person, bike):
    kp = person.keypoints

    if kp is None or len(kp) == 0:
        return 0.0

    bike_region = _expanded(
        bike.box,
        config.MOTORCYCLE_EXPANSION
    )

    bx1, by1, bx2, by2 = bike_region

    useful = []

    for idx in (11, 12, 15, 16):
        if idx >= len(kp):
            continue

        x = float(kp[idx][0])
        y = float(kp[idx][1])

        conf = float(kp[idx][2]) if len(kp[idx]) > 2 else 1.0

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

    expanded = _expanded(
        bike.box,
        config.MOTORCYCLE_EXPANSION
    )

    overlap = _iou(person.box, expanded)

    distance = _normalized_distance(
        person.box,
        bike.box
    )

    # Reject people that are too far away from the motorcycle.
    if distance > config.MAX_ASSOCIATION_DISTANCE:
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

    return float(np.clip(score, 0.0, 1.0))


def associate_people_to_bikes(persons, motorcycles, threshold=None):
    """Associate detected people with motorcycles.

    A person is associated with only one motorcycle.
    Each motorcycle is limited to a maximum of four associated people.
    """

    threshold = (
        config.DEFAULT_ASSOCIATION_THRESHOLD
        if threshold is None
        else threshold
    )

    result = {bike.track_id: [] for bike in motorcycles}

    # Store all valid person -> motorcycle associations first.
    candidates_by_bike = {
        bike.track_id: []
        for bike in motorcycles
    }

    for person in persons:

        # Ignore very low-confidence person detections.
        if person.confidence > 0.0 and person.confidence < 0.30:
            continue

        candidates = []

        for bike in motorcycles:

            score = association_score(person, bike)

            if score >= threshold:
                candidates.append(
                    (score, bike.track_id)
                )

        # A person can belong to only one motorcycle.
        if candidates:
            best_score, best_bike_id = max(
                candidates,
                key=lambda item: item[0]
            )

            candidates_by_bike[best_bike_id].append(
                (person, best_score)
            )

    # Keep only the strongest associations for each motorcycle.
    # This prevents unrelated detections from turning into extra riders.
    MAX_RIDERS_PER_MOTORCYCLE = 4

    for bike_id, candidates in candidates_by_bike.items():

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        result[bike_id] = candidates[
            :MAX_RIDERS_PER_MOTORCYCLE
        ]

    return result
