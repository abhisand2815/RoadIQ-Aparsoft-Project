"""
Helmet Detection pipeline for RoadIQ.

Focuses specifically on detecting helmet compliance (helmet on / no helmet)
strictly for riders associated with motorcycles, filtering out pedestrians.

Differentiates:
- Rider: The person driving the bike or scooty (in front, operating the controls).
- Person (Sitting Behind): Pillion passenger(s) sitting behind the driver.
- Pedestrian (Not on Bike): Anyone walking or standing nearby (ignored for helmet compliance).

Includes anatomical verification to prevent false-positives from face masks and hats/caps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set
import math
import cv2
import numpy as np

import config
from model_loader import load_detector, load_pose_model, load_helmet_model
from rider_association import Person, Motorcycle, associate_people_to_bikes


@dataclass
class HelmetDetection:
    label: str          # "With Helmet" or "Without Helmet"
    confidence: float
    box: Tuple[float, float, float, float]
    class_id: int


@dataclass
class RiderHelmetInfo:
    person: Person
    bike_id: int
    is_rider: bool                   # True if on a bike
    is_driver: bool                  # True if driving the bike/scooty
    role: str                        # "Rider (Driver)" or "Person (Sitting Behind)"
    status: str                      # "WITHOUT_HELMET", "WITH_HELMET", "UNKNOWN"
    confidence: float
    helmet_box: Optional[Tuple[float, float, float, float]]
    head_box: Tuple[float, float, float, float]
    head_center: Tuple[float, float]
    association_score: float
    rejection_note: Optional[str] = None  # e.g. "Mask detected instead of helmet"


@dataclass
class HelmetAnalysis:
    bikes: List[Motorcycle]
    people: List[Person]
    associations: Dict[int, List[Tuple[Person, float]]]
    rider_helmets: Dict[int, RiderHelmetInfo]      # person_id -> RiderHelmetInfo
    violating_riders: List[RiderHelmetInfo]        # without helmet riders
    compliant_riders: List[RiderHelmetInfo]        # with helmet riders
    violating_bikes: Set[int]                      # bike ids with >= 1 no-helmet rider
    all_helmet_detections: List[HelmetDetection]
    pedestrians: List[Person]


class HelmetDetector:
    def __init__(
        self,
        confidence: Optional[float] = None,
        association_threshold: Optional[float] = None,
        helmet_confidence: Optional[float] = None,
        detector_name: Optional[str] = None,
        pose_name: Optional[str] = None,
        helmet_model_name: Optional[str] = None,
    ):
        self.confidence = (
            float(confidence) if confidence is not None else float(config.DEFAULT_CONFIDENCE)
        )
        self.association_threshold = (
            float(association_threshold)
            if association_threshold is not None
            else float(config.DEFAULT_ASSOCIATION_THRESHOLD)
        )
        self.helmet_confidence = (
            float(helmet_confidence)
            if helmet_confidence is not None
            else float(config.DEFAULT_HELMET_CONFIDENCE)
        )

        detector_name = detector_name or config.DETECTOR_MODELS[config.DEFAULT_DETECTOR]
        pose_name = pose_name or config.POSE_MODELS[config.DEFAULT_POSE]
        helmet_model_name = helmet_model_name or config.HELMET_MODELS[config.DEFAULT_HELMET_MODEL]

        self.detector = load_detector(detector_name)
        self.pose = load_pose_model(pose_name)
        self.helmet_model = load_helmet_model(helmet_model_name)

    @staticmethod
    def _get_boxes(result, class_id: int):
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else np.arange(len(boxes))
        )
        return [
            (int(ids[i]), tuple(map(float, boxes[i])), float(confs[i]))
            for i in range(len(boxes))
            if classes[i] == class_id
        ]

    @staticmethod
    def _get_people(result):
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else np.arange(len(boxes))
        )
        keypoints = (
            result.keypoints.data.cpu().numpy()
            if result.keypoints is not None
            else None
        )
        people = []
        for i in range(len(boxes)):
            if classes[i] != config.PERSON_CLASS:
                continue
            kp = keypoints[i] if keypoints is not None else None
            people.append(
                Person(
                    track_id=int(ids[i]),
                    box=tuple(map(float, boxes[i])),
                    keypoints=kp,
                    confidence=float(confs[i]),
                )
            )
        return people

    @staticmethod
    def get_head_geometry(person: Person) -> Tuple[Tuple[float, float], Tuple[float, float, float, float]]:
        """
        Derive head center and bounding box from pose keypoints (if available)
        or fallback to upper body proportional geometry.
        """
        px1, py1, px2, py2 = person.box
        pw = px2 - px1
        ph = py2 - py1
        kp = person.keypoints

        if kp is not None and len(kp) >= 7:
            head_pts = [kp[i][:2] for i in range(5) if kp[i][2] >= 0.20]
            shoulders = [kp[i][:2] for i in (5, 6) if kp[i][2] >= 0.20]

            if head_pts:
                hx_min = min(p[0] for p in head_pts) - 0.15 * pw
                hx_max = max(p[0] for p in head_pts) + 0.15 * pw
                hy_min = min(p[1] for p in head_pts) - 0.25 * pw
                hy_max = max(p[1] for p in head_pts) + 0.10 * pw

                if shoulders:
                    shoulder_y = min(p[1] for p in shoulders)
                    hy_max = max(hy_max, shoulder_y)

                head_box = (
                    float(max(0.0, hx_min)),
                    float(max(0.0, min(py1, hy_min))),
                    float(hx_max),
                    float(hy_max),
                )
                center_x = float(sum(p[0] for p in head_pts) / len(head_pts))
                center_y = float(sum(p[1] for p in head_pts) / len(head_pts))
                return (center_x, center_y), head_box

        # Fallback: Top 32% of person box
        head_box = (float(px1), float(py1), float(px2), float(py1 + 0.32 * ph))
        head_center = (float((px1 + px2) / 2.0), float(py1 + 0.16 * ph))
        return head_center, head_box

    @staticmethod
    def validate_helmet_vs_mask_or_hat(person: Person, h_box: Tuple[float, float, float, float], label: str) -> Tuple[bool, Optional[str]]:
        """
        Anatomical verification:
        Ensures that face masks and hats/caps are not falsely accepted as motorcycle helmets.

        1. Face Mask Rejection:
           A mask covers only the lower half of the face (nose/mouth/chin).
           A real helmet MUST cover the upper cranium and forehead, so its top edge
           (hy1) must extend well above the eye line. If hy1 >= eye_y - 0.08*hh,
           it covers only the mouth/chin and is a MASK, not a helmet!

        2. Hat / Baseball Cap Rejection:
           A baseball cap or sun hat is very shallow, covers only the crown,
           and leaves the ears completely exposed below it. A motorcycle helmet
           is rounded/bulbous (height/width ratio >= 0.45) and covers down to ears/jawline.
        """
        if "Without" in label:
            return True, None

        hx1, hy1, hx2, hy2 = h_box
        hw = max(1.0, hx2 - hx1)
        hh = max(1.0, hy2 - hy1)
        kp = person.keypoints

        if kp is not None and len(kp) >= 5:
            eyes = [kp[i][1] for i in (1, 2) if kp[i][2] > 0.25]
            nose = kp[0][1] if kp[0][2] > 0.25 else None
            ears = [kp[i] for i in (3, 4) if kp[i][2] > 0.25]

            # 1. Mask Check: If top of detection is at/below the eye level, it's a face mask!
            if eyes:
                eye_y = sum(eyes) / len(eyes)
                if hy1 >= eye_y - 0.08 * hh:
                    return False, "Face mask detected (mouth/chin covered, cranium exposed)"

            if nose is not None and not eyes:
                if hy1 >= nose - 0.08 * hh:
                    return False, "Face mask detected (lower face covered)"

            # 2. Hat / Shallow Cap Check:
            if ears and eyes:
                ear_y = sum(p[1] for p in ears) / len(ears)
                # If ears are far below the box and the box is very shallow -> Hat/Cap
                if hy2 < ear_y and (hh / hw) < 0.52:
                    return False, "Hat / Cap detected (ears exposed, shallow brim)"

        # 3. Geometric Aspect Ratio: A motorcycle helmet is rounded/deep (hh/hw >= 0.45)
        if (hh / hw) < 0.42:
            return False, "Hat / Cap detected (flat brim aspect ratio)"

        return True, None

    @staticmethod
    def assign_rider_roles(riders: List[Person], bike: Motorcycle) -> Dict[int, Tuple[str, bool]]:
        """
        Differentiate who is driving the bike/scooty (Rider) vs sitting behind (Person):
        - If 1 person on bike: Rider (Driver)
        - If >=2 people on bike:
          Determines travel/facing direction (right vs left) using head/facial keypoints.
          The person positioned furthest forward in the direction of travel is the Driver (Rider).
          The person(s) positioned behind are Person (Sitting Behind).
        """
        if not riders:
            return {}
        if len(riders) == 1:
            return {riders[0].track_id: ("Rider (Driver)", True)}

        # Determine travel/facing direction (right vs left)
        votes = {"right": 0, "left": 0}
        for r in riders:
            kp = r.keypoints
            if kp is not None and len(kp) >= 5:
                if kp[0][2] > 0.20:
                    ear_xs = [kp[i][0] for i in (3, 4) if kp[i][2] > 0.20]
                    if ear_xs:
                        ear_x = sum(ear_xs) / len(ear_xs)
                        if kp[0][0] > ear_x:
                            votes["right"] += 1
                        else:
                            votes["left"] += 1

        facing = "right" if votes["right"] >= votes["left"] else "left"

        # If facing right, person with largest center_x is in front (Driver)
        # If facing left, person with smallest center_x is in front (Driver)
        def sort_key(r):
            cx = (r.box[0] + r.box[2]) / 2.0
            return -cx if facing == "right" else cx

        sorted_riders = sorted(riders, key=sort_key)
        roles = {}
        roles[sorted_riders[0].track_id] = ("Rider (Driver)", True)
        for r in sorted_riders[1:]:
            roles[r.track_id] = ("Person (Sitting Behind)", False)
        return roles

    def _extract_helmet_detections(self, frame: np.ndarray) -> List[HelmetDetection]:
        results = self.helmet_model.predict(
            frame,
            conf=self.helmet_confidence,
            verbose=False,
            imgsz=config.DEFAULT_IMAGE_SIZE,
        )[0]

        detections = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confs):
            label = self.helmet_model.names.get(cls_id, str(cls_id))
            detections.append(
                HelmetDetection(
                    label=label,
                    confidence=float(conf),
                    box=tuple(map(float, box)),
                    class_id=int(cls_id),
                )
            )
        return detections

    def _crop_head_inference(
        self, frame: np.ndarray, head_box: Tuple[float, float, float, float]
    ) -> Optional[HelmetDetection]:
        """Crop head region for targeted fallback classification."""
        h, w = frame.shape[:2]
        hx1, hy1, hx2, hy2 = map(int, head_box)
        pad_x = int(0.20 * (hx2 - hx1))
        pad_y = int(0.20 * (hy2 - hy1))

        cx1 = max(0, hx1 - pad_x)
        cy1 = max(0, hy1 - pad_y)
        cx2 = min(w, hx2 + pad_x)
        cy2 = min(h, hy2 + pad_y)

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        results = self.helmet_model.predict(
            crop,
            conf=0.18,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        best_idx = int(np.argmax(confs))
        cls_id = int(classes[best_idx])
        conf = float(confs[best_idx])
        label = self.helmet_model.names.get(cls_id, str(cls_id))

        local_b = boxes[best_idx]
        global_b = (
            float(local_b[0] + cx1),
            float(local_b[1] + cy1),
            float(local_b[2] + cx1),
            float(local_b[3] + cy1),
        )

        return HelmetDetection(label=label, confidence=conf, box=global_b, class_id=cls_id)

    def analyze(self, frame: np.ndarray, tracking: bool = False) -> HelmetAnalysis:
        # 1. Detect motorcycles and people
        if tracking:
            det = self.detector.track(
                frame,
                conf=self.confidence,
                iou=config.DEFAULT_IOU,
                imgsz=config.DEFAULT_IMAGE_SIZE,
                persist=True,
                tracker=config.TRACKER,
                verbose=False,
            )[0]
            pose = self.pose.track(
                frame,
                conf=self.confidence,
                iou=config.DEFAULT_IOU,
                imgsz=config.DEFAULT_IMAGE_SIZE,
                persist=True,
                tracker=config.TRACKER,
                verbose=False,
            )[0]
        else:
            det = self.detector.predict(
                frame,
                conf=self.confidence,
                iou=config.DEFAULT_IOU,
                imgsz=config.DEFAULT_IMAGE_SIZE,
                verbose=False,
            )[0]
            pose = self.pose.predict(
                frame,
                conf=self.confidence,
                iou=config.DEFAULT_IOU,
                imgsz=config.DEFAULT_IMAGE_SIZE,
                verbose=False,
            )[0]

        bikes = [
            Motorcycle(*item)
            for item in self._get_boxes(det, config.MOTORCYCLE_CLASS)
        ]
        people = self._get_people(pose)

        # 2. Strict Rider Association: Only associate people with bikes
        associations = associate_people_to_bikes(
            people, bikes, self.association_threshold
        )

        # Assign driver vs sitting behind (pillion) roles for each motorcycle
        rider_roles: Dict[int, Tuple[str, bool]] = {}
        rider_people_ids = set()
        rider_bike_map = {}
        rider_assoc_scores = {}

        for bike in bikes:
            riders_on_this_bike = [p for p, _ in associations.get(bike.track_id, [])]
            roles_for_bike = self.assign_rider_roles(riders_on_this_bike, bike)
            rider_roles.update(roles_for_bike)

            for person, score in associations.get(bike.track_id, []):
                rider_people_ids.add(person.track_id)
                rider_bike_map[person.track_id] = bike.track_id
                rider_assoc_scores[person.track_id] = float(score)

        pedestrians = [p for p in people if p.track_id not in rider_people_ids]
        riders = [p for p in people if p.track_id in rider_people_ids]

        # 3. Detect helmets in full frame
        helmet_detections = self._extract_helmet_detections(frame)

        # 4. Associate helmet detections specifically with riders' heads
        rider_matched_dets: Dict[int, List[HelmetDetection]] = {
            r.track_id: [] for r in riders
        }

        for h_det in helmet_detections:
            hx1, hy1, hx2, hy2 = h_det.box
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            best_rider = None
            min_dist = float("inf")

            for r in riders:
                (rcx, rcy), head_box = self.get_head_geometry(r)
                rx1, ry1, rx2, ry2 = r.box
                rw = rx2 - rx1
                rh = ry2 - ry1

                # Helmet center must lie within the upper region of the rider
                if (rx1 - 0.25 * rw) <= hcx <= (rx2 + 0.25 * rw) and (
                    ry1 - 0.20 * rh
                ) <= hcy <= (ry1 + 0.45 * rh):
                    dist = math.hypot(hcx - rcx, hcy - rcy)
                    if dist < min_dist:
                        min_dist = dist
                        best_rider = r

            if best_rider is not None:
                rider_matched_dets[best_rider.track_id].append(h_det)

        # 5. Build RiderHelmetInfo with role & mask/hat validation
        rider_helmets: Dict[int, RiderHelmetInfo] = {}
        violating_riders: List[RiderHelmetInfo] = []
        compliant_riders: List[RiderHelmetInfo] = []
        violating_bikes: Set[int] = set()

        for r in riders:
            bike_id = rider_bike_map[r.track_id]
            assoc_score = rider_assoc_scores[r.track_id]
            role, is_driver = rider_roles.get(r.track_id, ("Rider", True))
            head_center, head_box = self.get_head_geometry(r)
            matched = rider_matched_dets[r.track_id]

            selected_det: Optional[HelmetDetection] = None

            if matched:
                no_helmets = [d for d in matched if "Without" in d.label]
                with_helmets = [d for d in matched if "Without" not in d.label]

                if no_helmets and with_helmets:
                    best_no = max(no_helmets, key=lambda d: d.confidence)
                    best_with = max(with_helmets, key=lambda d: d.confidence)
                    selected_det = best_no if best_no.confidence >= best_with.confidence else best_with
                elif no_helmets:
                    selected_det = max(no_helmets, key=lambda d: d.confidence)
                else:
                    selected_det = max(with_helmets, key=lambda d: d.confidence)
            else:
                # Fallback: Targeted head crop inference
                selected_det = self._crop_head_inference(frame, head_box)

            rejection_note = None
            if selected_det is not None:
                is_no_helmet = "Without" in selected_det.label
                status = "WITHOUT_HELMET" if is_no_helmet else "WITH_HELMET"
                conf = float(selected_det.confidence)
                helmet_box = selected_det.box

                # Anatomical validation: Check if candidate helmet is actually a mask or hat
                if status == "WITH_HELMET" and helmet_box is not None:
                    is_valid, reason = self.validate_helmet_vs_mask_or_hat(
                        r, helmet_box, selected_det.label
                    )
                    if not is_valid:
                        # Reclassify to WITHOUT_HELMET (mask/hat is NOT a helmet)
                        status = "WITHOUT_HELMET"
                        rejection_note = reason
                        conf = max(0.85, conf)
            else:
                status = "UNKNOWN"
                conf = 0.0
                helmet_box = None

            info = RiderHelmetInfo(
                person=r,
                bike_id=int(bike_id),
                is_rider=True,
                is_driver=bool(is_driver),
                role=role,
                status=status,
                confidence=float(conf),
                helmet_box=helmet_box,
                head_box=head_box,
                head_center=head_center,
                association_score=float(assoc_score),
                rejection_note=rejection_note,
            )
            rider_helmets[r.track_id] = info

            if status == "WITHOUT_HELMET":
                violating_riders.append(info)
                violating_bikes.add(int(bike_id))
            elif status == "WITH_HELMET":
                compliant_riders.append(info)

        return HelmetAnalysis(
            bikes=bikes,
            people=people,
            associations=associations,
            rider_helmets=rider_helmets,
            violating_riders=violating_riders,
            compliant_riders=compliant_riders,
            violating_bikes=violating_bikes,
            all_helmet_detections=helmet_detections,
            pedestrians=pedestrians,
        )

    @staticmethod
    def annotate(frame: np.ndarray, analysis: HelmetAnalysis) -> np.ndarray:
        """
        Render visual overlays:
        - Red warnings for riders / pillions without helmet
        - Green badges for riders / pillions with helmet
        - Explicit Driver (Rider) vs Person (Sitting Behind) identification
        - Mask/Hat alerts if a non-helmet head covering was worn
        - Subtle gray boxes for pedestrians (explaining they are not on bike)
        """
        image = frame.copy()

        # 1. Draw Motorcycles
        for bike in analysis.bikes:
            bx1, by1, bx2, by2 = map(int, bike.box)
            riders_on_bike = analysis.associations.get(bike.track_id, [])
            rider_count = len(riders_on_bike)

            has_no_helmet = bike.track_id in analysis.violating_bikes
            no_helmet_count = sum(
                1
                for p, _ in riders_on_bike
                if analysis.rider_helmets.get(p.track_id)
                and analysis.rider_helmets[p.track_id].status == "WITHOUT_HELMET"
            )

            if has_no_helmet:
                color = (0, 0, 255)  # BGR Red
                label = f"BIKE {bike.track_id} | NO-HELMET VIOLATION ({no_helmet_count}/{rider_count} without helmet)"
                thickness = 3
            elif rider_count > 0:
                color = (0, 220, 100)  # BGR Green
                label = f"BIKE {bike.track_id} | RIDERS: {rider_count} | HELMETS OK"
                thickness = 2
            else:
                color = (255, 180, 0)
                label = f"BIKE {bike.track_id} | PARKED / NO RIDER"
                thickness = 2

            cv2.rectangle(image, (bx1, by1), (bx2, by2), color, thickness)

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            banner_y = max(24, by1 - 8)
            cv2.rectangle(
                image,
                (bx1, banner_y - th - 6),
                (bx1 + tw + 8, banner_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (bx1 + 4, banner_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Pedestrians (Non-riders)
        for ped in analysis.pedestrians:
            px1, py1, px2, py2 = map(int, ped.box)
            color = (150, 150, 150)
            cv2.rectangle(image, (px1, py1), (px2, py2), color, 1)
            label = f"PERSON {ped.track_id} | NOT ON BIKE"
            cv2.putText(
                image,
                label,
                (px1, min(image.shape[0] - 6, py2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Riders on Bikes
        for person_id, rinfo in analysis.rider_helmets.items():
            rider = rinfo.person
            rx1, ry1, rx2, ry2 = map(int, rider.box)

            # Differentiate: Driver vs Person (Sitting Behind)
            role_title = "RIDER (Driver)" if rinfo.is_driver else "PERSON (Sitting Behind)"

            if rinfo.status == "WITHOUT_HELMET":
                color = (0, 0, 255)  # Red
                if rinfo.rejection_note:
                    status_text = f"NO HELMET - {rinfo.rejection_note} ({rinfo.confidence:.0%})"
                else:
                    status_text = f"NO HELMET VIOLATION ({rinfo.confidence:.0%})"
                tag = f"{role_title} #{rider.track_id} | {status_text}"
                box_thickness = 3
            elif rinfo.status == "WITH_HELMET":
                color = (0, 200, 0)  # Green
                status_text = f"HELMET OK ({rinfo.confidence:.0%})"
                tag = f"{role_title} #{rider.track_id} | {status_text}"
                box_thickness = 2
            else:
                color = (0, 165, 255)  # Orange
                status_text = "HEAD OCCLUDED"
                tag = f"{role_title} #{rider.track_id} | {status_text}"
                box_thickness = 1

            cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, box_thickness)

            # Draw rider head indicator / helmet bounding box
            if rinfo.helmet_box is not None:
                hx1, hy1, hx2, hy2 = map(int, rinfo.helmet_box)
                cv2.rectangle(image, (hx1, hy1), (hx2, hy2), color, 2)
            else:
                hx1, hy1, hx2, hy2 = map(int, rinfo.head_box)
                cv2.rectangle(image, (hx1, hy1), (hx2, hy2), color, 1, lineType=cv2.LINE_4)

            # Rider label tag
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
            tag_y = max(18, ry1 - 6)
            cv2.rectangle(
                image,
                (rx1, tag_y - th - 4),
                (rx1 + tw + 6, tag_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                image,
                tag,
                (rx1 + 3, tag_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return image
