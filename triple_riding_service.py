"""Core Triple Riding pipeline using pretrained YOLO detection + pose."""
from dataclasses import dataclass
import cv2
import numpy as np
import config
from model_loader import load_detector, load_pose_model
from rider_association import Person, Motorcycle, associate_people_to_bikes


@dataclass
class FrameAnalysis:
    bikes: list
    people: list
    associations: dict
    triple_bikes: set


class TripleRidingDetector:
    def __init__(self, confidence=None, association_threshold=None, detector_name=None, pose_name=None):
        self.confidence = confidence if confidence is not None else config.DEFAULT_CONFIDENCE
        self.association_threshold = association_threshold if association_threshold is not None else config.DEFAULT_ASSOCIATION_THRESHOLD
        detector_name = detector_name or config.DETECTOR_MODELS[config.DEFAULT_DETECTOR]
        pose_name = pose_name or config.POSE_MODELS[config.DEFAULT_POSE]
        self.detector = load_detector(detector_name)
        self.pose = load_pose_model(pose_name)

    @staticmethod
    def _get_boxes(result, class_id):
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(boxes))
        return [
            (int(ids[i]), tuple(map(float, boxes[i])), float(confs[i]))
            for i in range(len(boxes)) if classes[i] == class_id
        ]

    @staticmethod
    def _get_people(result):
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(boxes))
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
        people = []
        for i in range(len(boxes)):
            if classes[i] != config.PERSON_CLASS:
                continue
            kp = keypoints[i] if keypoints is not None else None
            people.append(Person(int(ids[i]), tuple(map(float, boxes[i])), kp, float(confs[i])))
        return people

    def analyze(self, frame, tracking=True):
        if tracking:
            det = self.detector.track(frame, conf=self.confidence, iou=config.DEFAULT_IOU,
                                      imgsz=config.DEFAULT_IMAGE_SIZE, persist=True,
                                      tracker=config.TRACKER, verbose=False)[0]
            pose = self.pose.track(frame, conf=self.confidence, iou=config.DEFAULT_IOU,
                                   imgsz=config.DEFAULT_IMAGE_SIZE, persist=True,
                                   tracker=config.TRACKER, verbose=False)[0]
        else:
            det = self.detector.predict(frame, conf=self.confidence, iou=config.DEFAULT_IOU,
                                        imgsz=config.DEFAULT_IMAGE_SIZE, verbose=False)[0]
            pose = self.pose.predict(frame, conf=self.confidence, iou=config.DEFAULT_IOU,
                                     imgsz=config.DEFAULT_IMAGE_SIZE, verbose=False)[0]

        bikes = [Motorcycle(*item) for item in self._get_boxes(det, config.MOTORCYCLE_CLASS)]
        people = self._get_people(pose)
        associations = associate_people_to_bikes(people, bikes, self.association_threshold)
        triple = {bid for bid, riders in associations.items() if len(riders) >= 3}
        return FrameAnalysis(bikes, people, associations, triple)

    @staticmethod
    def annotate(frame, analysis: FrameAnalysis, confirmed_bikes=None):
        confirmed_bikes = confirmed_bikes or set()
        image = frame.copy()
        rider_ids = {p.track_id for riders in analysis.associations.values() for p, _ in riders}
        scores = {p.track_id: score for riders in analysis.associations.values() for p, score in riders}

        for bike in analysis.bikes:
            x1, y1, x2, y2 = map(int, bike.box)
            count = len(analysis.associations.get(bike.track_id, []))
            is_triple = bike.track_id in analysis.triple_bikes
            is_confirmed = bike.track_id in confirmed_bikes
            color = (0, 0, 255) if is_confirmed else ((0, 165, 255) if is_triple else (255, 180, 0))
            label = f"BIKE {bike.track_id} | RIDERS {count}"
            if is_confirmed:
                label += " | TRIPLE RIDING CONFIRMED"
            elif is_triple:
                label += " | CANDIDATE"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if is_confirmed else 2)
            cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        for person in analysis.people:
            x1, y1, x2, y2 = map(int, person.box)
            if person.track_id in rider_ids:
                color = (0, 255, 0)
                label = f"RIDER {person.track_id} | {scores[person.track_id]:.0%}"
            else:
                color = (170, 170, 170)
                label = f"PERSON {person.track_id} | NOT ASSOCIATED"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, min(image.shape[0] - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        return image
