

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from model_loader import load_detector


@dataclass
class TrafficSignal:
   

    state: str  # "RED", "GREEN", "YELLOW", "UNKNOWN"
    confidence: float
    box: Optional[Tuple[int, int, int, int]] = None
    lamp_name: Optional[str] = None
    lamp_pixels: int = 0
    reason: str = ""


class TrafficSignalDetector:


    def __init__(
        self,
        detector=None,
        detector_name: Optional[str] = None,
        confidence: float = 0.15,
    ):
        self.confidence = confidence
        if detector is not None:
            self.model = detector
        else:
            self.model = load_detector(detector_name or "yolo26n.pt")

    def detect(self, frame: np.ndarray) -> TrafficSignal:

        if frame is None or frame.size == 0:
            return TrafficSignal(
                state="UNKNOWN",
                confidence=0.0,
                box=None,
                reason="Input image is empty.",
            )

        height, width = frame.shape[:2]

        
        results = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False,
        )

        candidates = []
        for result in results:
            if result.boxes is None:
                continue
            for index in range(len(result.boxes)):
                class_id = int(result.boxes.cls[index].item())
                if class_id != 9:  # COCO class 9: traffic light
                    continue

                conf = float(result.boxes.conf[index].item())
                raw_x1, raw_y1, raw_x2, raw_y2 = (
                    result.boxes.xyxy[index].tolist()
                )
                x1 = max(0, int(raw_x1))
                y1 = max(0, int(raw_y1))
                x2 = min(width - 1, int(raw_x2))
                y2 = min(height - 1, int(raw_y2))

                box_w = x2 - x1
                box_h = y2 - y1

                
                if y1 > height * 0.78:
                    continue

                
                if box_w < 6 or box_h < 12 or (box_w * box_h) < 70:
                    continue

                candidates.append((conf, (x1, y1, x2, y2)))

        if not candidates:
            return TrafficSignal(
                state="UNKNOWN",
                confidence=0.0,
                box=None,
                reason="No traffic light housing detected in scene.",
            )

        
        candidates.sort(
            key=lambda c: (c[0] * 0.6 + min(1.0, (c[1][2] - c[1][0]) * (c[1][3] - c[1][1]) / 5000.0) * 0.4),
            reverse=True,
        )

        best_signal = None
        for box_conf, box in candidates:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            state, lamp_conf, lamp_name, lamp_px, reason = self.classify_lamp_crop(crop)

            
            signal_confidence = min(0.99, max(0.10, float(box_conf * 0.5 + lamp_conf * 0.5)))

            if state in ("RED", "GREEN", "YELLOW"):
                return TrafficSignal(
                    state=state,
                    confidence=signal_confidence,
                    box=box,
                    lamp_name=lamp_name,
                    lamp_pixels=lamp_px,
                    reason=reason,
                )
            elif best_signal is None:
                best_signal = TrafficSignal(
                    state="UNKNOWN",
                    confidence=signal_confidence,
                    box=box,
                    lamp_name=lamp_name,
                    lamp_pixels=lamp_px,
                    reason=reason,
                )

        return best_signal or TrafficSignal(
            state="UNKNOWN",
            confidence=0.0,
            box=None,
            reason="Detected traffic lights showed ambiguous or unlit lamp states.",
        )

    @staticmethod
    def classify_lamp_crop(
        crop: np.ndarray,
    ) -> Tuple[str, float, Optional[str], int, str]:

        if crop is None or crop.size == 0:
            return "UNKNOWN", 0.0, None, 0, "Empty crop"

        ch, cw = crop.shape[:2]
        if ch < 10 or cw < 5:
            return "UNKNOWN", 0.0, None, 0, "Crop resolution too small for lamp analysis"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

 
        red_mask = ((h_chan < 12) | (h_chan > 165)) & (s_chan > 70) & (v_chan > 130)

        yellow_mask = (h_chan >= 15) & (h_chan <= 36) & (s_chan > 70) & (v_chan > 130)

        green_mask = (h_chan >= 38) & (h_chan <= 92) & (s_chan > 65) & (v_chan > 130)


        is_vertical = ch >= cw

        if is_vertical:
           
            top_bound = int(ch * 0.45)
            mid_start = int(ch * 0.25)
            mid_end = int(ch * 0.75)
            bot_bound = int(ch * 0.55)

            red_pixels = int(np.sum(red_mask[:top_bound, :]))
            yellow_pixels = int(np.sum(yellow_mask[mid_start:mid_end, :]))
            green_pixels = int(np.sum(green_mask[bot_bound:, :]))

            red_peak = int(np.max(v_chan[:top_bound, :])) if top_bound > 0 else 0
            yellow_peak = int(np.max(v_chan[mid_start:mid_end, :])) if mid_end > mid_start else 0
            green_peak = int(np.max(v_chan[bot_bound:, :])) if ch > bot_bound else 0
        else:
           
            left_bound = int(cw * 0.45)
            mid_start = int(cw * 0.25)
            mid_end = int(cw * 0.75)
            right_bound = int(cw * 0.55)

            red_pixels = int(np.sum(red_mask[:, :left_bound]))
            yellow_pixels = int(np.sum(yellow_mask[:, mid_start:mid_end]))
            green_pixels = int(np.sum(green_mask[:, right_bound:]))

            red_peak = int(np.max(v_chan[:, :left_bound])) if left_bound > 0 else 0
            yellow_peak = int(np.max(v_chan[:, mid_start:mid_end])) if mid_end > mid_start else 0
            green_peak = int(np.max(v_chan[:, right_bound:])) if cw > right_bound else 0

       
        min_lamp_pixels = max(5, int(ch * cw * 0.012))

       
        if (
            red_pixels >= min_lamp_pixels
            and red_peak >= 155
            and red_pixels > 1.8 * max(yellow_pixels, green_pixels)
        ):
            lamp_conf = min(0.98, 0.65 + min(0.33, red_pixels / 300.0))
            return "RED", lamp_conf, "red_lamp", red_pixels, f"Illuminated red lamp detected ({red_pixels} active pixels, peak={red_peak})"

        
        if (
            green_pixels >= min_lamp_pixels
            and green_peak >= 155
            and green_pixels > 1.8 * max(red_pixels, yellow_pixels)
        ):
            lamp_conf = min(0.98, 0.65 + min(0.33, green_pixels / 300.0))
            return "GREEN", lamp_conf, "green_lamp", green_pixels, f"Illuminated green lamp detected ({green_pixels} active pixels, peak={green_peak})"

     
        if (
            yellow_pixels >= min_lamp_pixels
            and yellow_peak >= 155
            and yellow_pixels > 1.8 * max(red_pixels, green_pixels)
        ):
            lamp_conf = min(0.98, 0.65 + min(0.33, yellow_pixels / 300.0))
            return "YELLOW", lamp_conf, "yellow_lamp", yellow_pixels, f"Illuminated yellow lamp detected ({yellow_pixels} active pixels, peak={yellow_peak})"

       
        active_counts = [cnt for cnt in (red_pixels, yellow_pixels, green_pixels) if cnt >= min_lamp_pixels]
        if len(active_counts) >= 2:
            return "UNKNOWN", 0.30, None, max(red_pixels, yellow_pixels, green_pixels), "Ambiguous lamp illumination (multiple colored apertures detected)"

        return "UNKNOWN", 0.20, None, 0, "No illuminated lamp aperture confidently detected in housing"
