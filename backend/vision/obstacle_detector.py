"""
Saarthi — Obstacle Detector Module
====================================
Uses a YOLOv8-nano model to detect navigable hazards and everyday
objects that are relevant for a visually impaired user. Only objects
from a curated target list are returned, filtered by a confidence
threshold.

Usage:
    from backend.vision.obstacle_detector import ObstacleDetector

    detector = ObstacleDetector()
    detections = detector.detect(frame)
"""

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
#  Detection result data class
# ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object with spatial awareness."""
    label: str
    confidence: float
    bbox: tuple       # (x1, y1, x2, y2) — top-left and bottom-right corners
    direction: str = ""        # "Left", "Center", or "Right"
    distance_m: float = -1.0   # Estimated distance in meters (-1 = unknown)
    is_hazard: bool = False    # True if within the immediate hazard zone



# ──────────────────────────────────────────────────────────────
#  Target hazard & object classes
# ──────────────────────────────────────────────────────────────

# COCO class names relevant for navigation assistance.
# These are the objects we care about for a visually impaired user.
TARGET_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "cat",
    "dog",
    "chair",
    "couch",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "bottle",
    "cup",
    "bed",
    "door",
    "bench",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "fire hydrant",
    "stop sign",
    "traffic light",
    "potted plant",
    "stairs",
]


# ──────────────────────────────────────────────────────────────
#  Known real-world heights (meters) for monocular distance
# ──────────────────────────────────────────────────────────────

# Approximate average real-world heights used for the pinhole
# camera distance formula:  distance = (real_h × focal_len) / pixel_h
KNOWN_HEIGHTS_M = {
    "person":        1.70,
    "bicycle":       1.10,
    "car":           1.50,
    "motorcycle":    1.10,
    "bus":           3.00,
    "truck":         3.50,
    "cat":           0.25,
    "dog":           0.45,
    "chair":         0.90,
    "couch":         0.85,
    "dining table":  0.75,
    "toilet":        0.40,
    "tv":            0.50,
    "laptop":        0.25,
    "bottle":        0.25,
    "cup":           0.15,
    "bed":           0.60,
    "door":          2.00,
    "bench":         0.90,
    "backpack":      0.50,
    "umbrella":      1.00,
    "handbag":       0.35,
    "suitcase":      0.60,
    "fire hydrant":  0.50,
    "stop sign":     2.00,
    "traffic light": 0.80,
    "potted plant":  0.40,
    "stairs":        1.50,
}

# Approximate focal length in pixels for a typical smartphone
# camera streaming at 640×480. Adjust if your camera differs.
FOCAL_LENGTH_PX = 600

# Immediate hazard zone — only objects closer than this trigger warnings
HAZARD_ZONE_M = 2.5

# Sector boundaries (fraction of frame width)
LEFT_BOUNDARY = 0.35
RIGHT_BOUNDARY = 0.65


# ──────────────────────────────────────────────────────────────
#  Obstacle Detector
# ──────────────────────────────────────────────────────────────

class ObstacleDetector:
    """
    Loads a YOLOv8 model and runs inference on video frames,
    returning only detections that belong to the curated
    TARGET_CLASSES list and exceed the confidence threshold.

    Args:
        model_path:  Path to the YOLOv8 weights file.
        conf_thresh: Minimum confidence to keep a detection (default 0.50).
    """

    # Default weights path relative to the project root
    _DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "yolov8n.pt"
    )

    def __init__(self, model_path: str = None, conf_thresh: float = 0.50):
        self.model_path = model_path or self._DEFAULT_MODEL_PATH
        self.conf_thresh = conf_thresh

        # Load the YOLO model
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"[ObstacleDetector] Model weights not found at: {self.model_path}\n"
                "  Run: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\" "
                "and move the file to models/yolov8n.pt"
            )

        self.model = YOLO(self.model_path)

        # Build a lookup set of target class names (lowercased) for fast filtering
        self._target_set = {name.lower() for name in TARGET_CLASSES}

        # Map COCO class indices → names from the model itself
        self._class_names = self.model.names  # dict {0: 'person', 1: 'bicycle', ...}

        print(f"[ObstacleDetector] Model loaded: {self.model_path}")
        print(f"[ObstacleDetector] Confidence threshold: {self.conf_thresh}")
        print(f"[ObstacleDetector] Tracking {len(self._target_set)} target classes")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame and enrich each
        detection with direction, estimated distance, and hazard flag.

        Args:
            frame: A BGR numpy array (H, W, 3) from OpenCV.

        Returns:
            A list of Detection objects with spatial fields populated.
        """
        frame_h, frame_w = frame.shape[:2]

        # Run inference — verbose=False to suppress per-frame logs
        results = self.model(frame, conf=self.conf_thresh, verbose=False)

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract class index and look up the name
                cls_id = int(box.cls[0])
                label = self._class_names.get(cls_id, "unknown").lower()

                # Filter: only keep objects in our target list
                if label not in self._target_set:
                    continue

                confidence = float(box.conf[0])

                # Bounding box: (x1, y1, x2, y2) in pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                # ── Spatial enrichment ──
                direction = self.estimate_direction(bbox, frame_w)
                distance_m = self.estimate_distance(label, bbox)
                is_hazard = distance_m < HAZARD_ZONE_M if distance_m > 0 else False

                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        direction=direction,
                        distance_m=round(distance_m, 2),
                        is_hazard=is_hazard,
                    )
                )

        return detections

    # ──────────────────────────────────────────────────────────
    #  Spatial helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def estimate_direction(bbox: tuple, frame_width: int) -> str:
        """
        Classify the horizontal position of a detection into a sector.

        Sectors (based on bbox center X as a fraction of frame width):
            - "Left"   :  X_center < 35 %
            - "Center" :  35 % ≤ X_center ≤ 65 %
            - "Right"  :  X_center > 65 %

        Args:
            bbox:        (x1, y1, x2, y2) pixel coordinates.
            frame_width: Width of the frame in pixels.

        Returns:
            One of "Left", "Center", or "Right".
        """
        x1, _, x2, _ = bbox
        x_center_frac = ((x1 + x2) / 2.0) / frame_width

        if x_center_frac < LEFT_BOUNDARY:
            return "Left"
        elif x_center_frac > RIGHT_BOUNDARY:
            return "Right"
        else:
            return "Center"

    @staticmethod
    def estimate_distance(label: str, bbox: tuple) -> float:
        """
        Estimate the physical distance (meters) to an object using
        the pinhole camera model:

            distance = (real_height × focal_length) / bbox_pixel_height

        Args:
            label: Class name of the detected object.
            bbox:  (x1, y1, x2, y2) pixel coordinates.

        Returns:
            Estimated distance in meters, or -1.0 if the object's
            real-world height is unknown.
        """
        real_height = KNOWN_HEIGHTS_M.get(label)
        if real_height is None:
            return -1.0

        _, y1, _, y2 = bbox
        pixel_height = abs(y2 - y1)

        if pixel_height == 0:
            return -1.0

        distance = (real_height * FOCAL_LENGTH_PX) / pixel_height
        return distance

    def get_hazards(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter a list of detections to only those within the
        immediate hazard zone (< 2.5 m).

        Args:
            detections: Full list of Detection objects from detect().

        Returns:
            A filtered list containing only hazardous detections.
        """
        return [d for d in detections if d.is_hazard]



# ──────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import sys

    print("=" * 50)
    print("  Saarthi — Obstacle Detector + Spatial Test")
    print("=" * 50)

    detector = ObstacleDetector()

    # Try local webcam for a quick test
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Test] Cannot open webcam. Exiting.")
        sys.exit(1)

    print("[Test] Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        hazards = detector.get_hazards(detections)

        # Draw detections on the frame
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Red for hazards, green for safe
            color = (0, 0, 255) if det.is_hazard else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label with distance and direction
            dist_str = f"{det.distance_m:.1f}m" if det.distance_m > 0 else "?m"
            text = f"{det.label} {dist_str} [{det.direction}]"
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        # HUD overlay
        cv2.putText(
            frame,
            f"Objects: {len(detections)}  |  HAZARDS: {len(hazards)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if hazards else (0, 255, 0),
            2,
        )

        cv2.imshow("Saarthi — Hazard Perception", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Test] Done.")

