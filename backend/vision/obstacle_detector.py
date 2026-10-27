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
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set

import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
#  Detection result data class
# ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object with spatial awareness and temporal confirmation."""
    label: str
    confidence: float
    bbox: tuple       # (x1, y1, x2, y2) — top-left and bottom-right corners
    direction: str = ""        # "Left", "Center", or "Right"
    distance_m: float = -1.0   # Estimated distance in meters (-1 = unknown)
    is_hazard: bool = False    # True if within the immediate hazard zone
    is_confirmed: bool = False # Detected for >= 3 consecutive frames (anti-hallucination)
    is_eligible: bool = False  # Confirmed and elapsed cooldown >= 5.0s



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
    "mouse",
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
    "mouse":         0.04,
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
        conf_thresh: Minimum confidence to keep a detection (default 0.35).
    """

    # Default weights path relative to the project root
    _DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "yolov8s.pt"
    )

    def __init__(self, model_path: str = None, conf_thresh: float = 0.35, synthesizer = None):
        self.model_path = model_path or self._DEFAULT_MODEL_PATH
        self.conf_thresh = conf_thresh
        self.synthesizer = synthesizer

        # Step 2: Object Memory (Class-Based Cooldowns)
        self.cooldowns: Dict[str, float] = {}
        self.cooldown_seconds: float = 5.0

        # Step 3: Frame Confirmation (Anti-Hallucination)
        self.frame_counters: Dict[str, int] = {}
        self.min_confirm_frames: int = 3

        # Load the YOLO model (auto-download via Ultralytics if not found locally)
        if os.path.isfile(self.model_path):
            self.model = YOLO(self.model_path)
        elif os.path.isfile(os.path.basename(self.model_path)):
            self.model = YOLO(os.path.basename(self.model_path))
        else:
            model_name = os.path.basename(self.model_path)
            print(f"[ObstacleDetector] '{self.model_path}' not found locally. Ultralytics will auto-download {model_name}...")
            self.model = YOLO(model_name)

        # Map COCO class indices → names from the model itself
        self._class_names = self.model.names  # dict {0: 'person', 1: 'bicycle', ...}

        # Whitelist is bypassed: detect ALL classes recognised by YOLO model
        self._target_set = None

        print(f"[ObstacleDetector] Model loaded: {self.model_path}")
        print(f"[ObstacleDetector] Confidence threshold: {self.conf_thresh}")
        print(f"[ObstacleDetector] Detecting all {len(self._class_names)} classes (3-frame confirmation, 5.0s cooldown)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame, detecting all objects.
        Applies 3-frame confirmation temporal smoothing to eliminate 1-frame
        hallucinations, and enforces a 5-second per-class announcement cooldown.

        Args:
            frame: A BGR numpy array (H, W, 3) from OpenCV.

        Returns:
            A list of Detection objects with spatial and temporal fields populated.
        """
        frame_h, frame_w = frame.shape[:2]

        # Run inference — verbose=False to suppress per-frame logs
        results = self.model(frame, conf=self.conf_thresh, verbose=False)

        current_frame_classes: Set[str] = set()
        raw_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Extract class index and look up the name
                cls_id = int(box.cls[0])
                label = self._class_names.get(cls_id, "unknown").lower()
                if label == "unknown":
                    continue

                confidence = float(box.conf[0])

                # Bounding box: (x1, y1, x2, y2) in pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                # ── Spatial enrichment ──
                direction = self.estimate_direction(bbox, frame_w)
                distance_m = self.estimate_distance(label, bbox)
                is_hazard = distance_m < HAZARD_ZONE_M if distance_m > 0 else False

                current_frame_classes.add(label)
                raw_detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "direction": direction,
                    "distance_m": distance_m,
                    "is_hazard": is_hazard,
                })

        # ── Step 3: Frame Confirmation (Anti-Hallucination) ──
        # Increment consecutive frame counter for all detected classes
        for label in current_frame_classes:
            self.frame_counters[label] = self.frame_counters.get(label, 0) + 1

        # If an object disappears before 3 frames (or at any time), reset counter to 0
        for tracked_label in list(self.frame_counters.keys()):
            if tracked_label not in current_frame_classes:
                self.frame_counters[tracked_label] = 0

        # ── Step 2: Object Memory (Class-Based Cooldowns) ──
        now = time.time()
        detections: List[Detection] = []

        for item in raw_detections:
            label = item["label"]
            consecutive_frames = self.frame_counters.get(label, 0)
            is_confirmed = consecutive_frames >= self.min_confirm_frames

            # An object must be detected for >= 3 consecutive frames before
            # passing it to cooldown logic or audio synthesizer.
            # If confirmed, check if time elapsed since last alert >= 5.0 seconds.
            is_eligible = False
            if is_confirmed:
                last_time = self.cooldowns.get(label, 0.0)
                if (now - last_time) >= self.cooldown_seconds:
                    is_eligible = True

            detections.append(
                Detection(
                    label=label,
                    confidence=item["confidence"],
                    bbox=item["bbox"],
                    direction=item["direction"],
                    distance_m=round(item["distance_m"], 2),
                    is_hazard=item["is_hazard"],
                    is_confirmed=is_confirmed,
                    is_eligible=is_eligible,
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
            Estimated distance in meters.
        """
        # Use known height if calibrated; fallback to 0.50m for unlisted COCO classes
        real_height = KNOWN_HEIGHTS_M.get(label, 0.50)

        _, y1, _, y2 = bbox
        pixel_height = abs(y2 - y1)

        if pixel_height == 0:
            return -1.0

        distance = (real_height * FOCAL_LENGTH_PX) / pixel_height
        return distance

    def get_hazards(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter a list of detections to only confirmed hazards eligible for announcement.

        Criteria:
          1. is_hazard (< 2.5 m)
          2. is_confirmed (detected for >= 3 consecutive frames - anti-hallucination)
          3. is_eligible (more than 5.0 seconds elapsed since last announcement for this class)

        Updates the class cooldown timestamp when a hazard is returned for announcement.

        Args:
            detections: Full list of Detection objects from detect().

        Returns:
            A filtered list containing confirmed, eligible hazardous detections.
        """
        now = time.time()
        hazards: List[Detection] = []
        for d in detections:
            if d.is_hazard and d.is_confirmed and d.is_eligible:
                hazards.append(d)
                self.cooldowns[d.label] = now
                d.is_eligible = False  # Consumed for this frame
        return hazards

    def get_confirmed(self, detections: List[Detection]) -> List[Detection]:
        """Return only detections confirmed across >= 3 frames."""
        return [d for d in detections if d.is_confirmed]



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

