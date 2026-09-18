from dataclasses import dataclass
from math import hypot
from typing import Optional

import numpy as np


@dataclass
class PersonObservation:
    person_id: int
    box: tuple[float, float, float, float]
    keypoints: Optional[np.ndarray]
    confidence: float


@dataclass
class BikeObservation:
    bike_id: int
    box: tuple[float, float, float, float]
    confidence: float


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _expand_box(box, x_ratio=0.55, y_ratio=0.85):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        x1 - w * x_ratio,
        y1 - h * y_ratio,
        x2 + w * x_ratio,
        y2 + h * y_ratio,
    )


def _point_in_box(point, box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pose_geometry_score(person: PersonObservation, bike: BikeObservation) -> float:
    """
    Heuristic pose score used only for association.

    A standing pedestrian normally has a foot point well below the bike and
    a torso/hip structure outside the motorcycle footprint. A seated rider's
    lower body and torso tend to remain spatially coupled to the motorcycle.

    This is intentionally an association feature, not the final violation
    classifier. The future capstone model should learn this relationship from
    labelled riding/non-riding examples.
    """
    kps = person.keypoints
    if kps is None or len(kps) < 17:
        return 0.0

    # COCO keypoints: shoulder 5/6, hip 11/12, ankle 15/16.
    pts = []
    for idx in (5, 6, 11, 12, 15, 16):
        x, y, conf = kps[idx]
        if conf >= 0.25:
            pts.append((float(x), float(y), float(conf)))

    if not pts:
        return 0.0

    bx = _expand_box(bike.box)
    hips = []
    ankles = []
    for idx in (11, 12):
        x, y, c = kps[idx]
        if c >= 0.25:
            hips.append((float(x), float(y)))
    for idx in (15, 16):
        x, y, c = kps[idx]
        if c >= 0.25:
            ankles.append((float(x), float(y)))

    score = 0.0

    # Hip inside/near bike is a strong signal for seated riders.
    if any(_point_in_box(p, bx) for p in hips):
        score += 0.45

    # At least one ankle close to the bike footprint supports riding.
    if any(_point_in_box(p, bx) for p in ankles):
        score += 0.25

    # Torso center close to motorcycle center.
    if hips:
        hx = sum(p[0] for p in hips) / len(hips)
        hy = sum(p[1] for p in hips) / len(hips)
        cx, cy = _center(bike.box)
        bw = max(1.0, bike.box[2] - bike.box[0])
        bh = max(1.0, bike.box[3] - bike.box[1])
        d = hypot((hx - cx) / bw, (hy - cy) / bh)
        score += max(0.0, 0.30 - 0.15 * d)

    return min(score, 1.0)


def rider_association_score(
    person: PersonObservation,
    bike: BikeObservation,
) -> float:
    """
    Score whether a person is riding a particular motorcycle.

    Combines:
      - person/bike overlap
      - lower-body position
      - distance between person and bike
      - pose geometry

    Returns 0..1.
    """
    expanded = _expand_box(bike.box)
    iou = box_iou(person.box, expanded)

    p_bottom = _bottom_center(person.box)
    in_expanded = _point_in_box(p_bottom, expanded)

    bx, by = _center(bike.box)
    px, py = _center(person.box)

    bw = max(1.0, bike.box[2] - bike.box[0])
    bh = max(1.0, bike.box[3] - bike.box[1])
    normalized_distance = hypot((px - bx) / bw, (py - by) / bh)
    distance_score = max(0.0, 1.0 - normalized_distance / 2.0)

    geometry = _pose_geometry_score(person, bike)

    score = (
        0.35 * min(iou / 0.35, 1.0)
        + 0.25 * float(in_expanded)
        + 0.20 * distance_score
        + 0.20 * geometry
    )

    # Strong penalty for a person whose lower body is clearly outside the
    # expanded motorcycle neighbourhood.
    if not in_expanded and iou < 0.01:
        score *= 0.35

    return float(max(0.0, min(1.0, score)))


def associate_people_to_bikes(
    people: list[PersonObservation],
    bikes: list[BikeObservation],
    min_score: float = 0.45,
):
    associations = {bike.bike_id: [] for bike in bikes}

    for person in people:
        candidates = []
        for bike in bikes:
            score = rider_association_score(person, bike)
            if score >= min_score:
                candidates.append((score, bike.bike_id))

        if candidates:
            # A person is assigned to the most plausible bike.
            score, bike_id = max(candidates)
            associations[bike_id].append((person, score))

    return associations
